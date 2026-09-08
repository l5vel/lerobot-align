#!/usr/bin/env python
"""Score `--plan.subtasks_path` alignment against the human labels in
``meta/lerobot_annotations.json``.

Runs the real ``_align_given_subtasks`` path, so it tracks whatever the module
currently does, and reports how far the spans it produces sit from the human
boundaries.

    python eval_align.py ROOT --episode 0 --fps 1.0 --labels gt

``--labels gt`` supplies the human labels (isolates the timing from any
label-mismatch). ``--labels-file labels.json`` supplies a non-empty ordered
JSON list of your own labels; ``--labels user`` requires that file.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from lerobot_align.alignment_metrics import evaluate_alignment
from lerobot_align.config import PlanConfig, VlmConfig
from lerobot_align.diagnostics._camera import MULTI_CAMERA_HELP, resolve_camera_keys
from lerobot_align.diagnostics._provenance import api_key, validate_options, validate_truth
from lerobot_align.frames import make_frame_provider
from lerobot_align.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
)
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import make_vlm_client



def ground_truth(root: Path, episode: int) -> list[dict]:
    payload = json.loads((root / "meta" / "lerobot_annotations.json").read_text(encoding="utf-8"))
    spans = payload["episodes"][str(episode)]["subtasks"]
    return validate_truth({episode: spans})[episode]


def label_at(spans: list[dict], t: float, key: str) -> str | None:
    """The label active at ``t``, reading ``key`` for the text field."""
    for span in spans:
        if float(span["start"]) <= t < float(span["end"]):
            return str(span[key])
    return str(spans[-1][key]) if spans and t >= float(spans[-1]["end"]) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--labels-file", type=Path, help="JSON list of ordered subtask labels")
    ap.add_argument("--api-key-env", default="LEROBOT_VLM_API_KEY", help="environment variable containing the endpoint API key")
    ap.add_argument("--labels", default="gt", choices=["gt", "user"])
    ap.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help=MULTI_CAMERA_HELP,
    )
    ap.add_argument(
        "--sampling",
        default="uniform",
        choices=["uniform", "motion_stratified"],
        help="timestamp selection strategy",
    )
    ap.add_argument(
        "--frame-format",
        default="contact_sheet",
        choices=["contact_sheet", "video"],
        help="how sampled frames reach the model (video = one re-encoded clip)",
    )
    ap.add_argument("--frame-width", type=int, default=336)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--model", default=VlmConfig().model_id)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--dump", type=Path, default=None, help="write the raw reply + spans here")
    args = ap.parse_args()
    validate_options(args, ap)
    endpoint_key = api_key(args, ap)
    if args.labels == 'user' and args.labels_file is None:
        ap.error('--labels user requires --labels-file; no dataset-specific labels are built in')
    user_labels = None
    if args.labels_file is not None:
        try:
            user_labels = json.loads(args.labels_file.read_text())
        except (OSError, ValueError) as exc:
            ap.error(f'Cannot read --labels-file: {exc}')
        if not isinstance(user_labels, list) or not user_labels or any(
            not isinstance(label, str) or not label.strip() for label in user_labels
        ):
            ap.error('--labels-file must contain a non-empty list of non-empty strings')
        args.labels = 'user'
    cameras = resolve_camera_keys(args.root, args.cameras)

    gt = ground_truth(args.root, args.episode)
    labels = [s["label"] for s in gt] if args.labels == "gt" else user_labels

    print(
        f"### episode {args.episode}  labels={args.labels}  fps={args.fps}  "
        f"cameras={cameras}  sampling={args.sampling}  format={args.frame_format}"
    )
    for i, text in enumerate(labels):
        print(f"    {i}. {text}")

    vlm_cfg = VlmConfig(
        model_id=args.model,
        api_base=args.api_base,
        api_key=endpoint_key,
        auto_serve=False,
        camera_key=cameras[0],
        chat_template_kwargs={"enable_thinking": False},
        max_new_tokens=4096,
        temperature=args.temperature,
        client_concurrency=4,
    )
    vlm = make_vlm_client(vlm_cfg)

    captured: list = []
    original = vlm.generate_json

    def spy(messages_batch, **kwargs):
        results = original(messages_batch, **kwargs)
        captured.append((messages_batch, kwargs, results))
        return results

    vlm.generate_json = spy  # type: ignore[method-assign]

    # PlanConfig wants a path. Create it atomically with an unpredictable name
    # so concurrent diagnostics cannot overwrite one another's labels, and
    # close the descriptor returned by mkstemp before reopening it.
    subtasks_fd, subtasks_name = tempfile.mkstemp(prefix="lerobot-align-labels-", suffix=".json")
    os.close(subtasks_fd)
    subtasks_path = Path(subtasks_name)

    def cleanup_subtasks() -> None:
        subtasks_path.unlink(missing_ok=True)

    atexit.register(cleanup_subtasks)
    subtasks_path.write_text(json.dumps(labels), encoding="utf-8")

    plan_cfg = PlanConfig(
        subtasks_path=subtasks_path,
        frames_per_second=args.fps,
        max_frames_per_prompt=args.max_frames,
        contact_sheet_frame_width=args.frame_width,
        subtask_align_min_fraction=0.0,
        subtask_align_camera_keys=tuple(cameras),
        subtask_align_sampling=args.sampling,
        subtask_align_frame_format=args.frame_format,
        n_task_rephrasings=0,
        emit_plan=False,
        emit_memory=False,
    )
    provider = make_frame_provider(args.root, camera_key=cameras[0])
    module = PlanSubtasksMemoryModule(vlm=vlm, config=plan_cfg, frame_provider=provider, root=args.root)

    record = next(r for r in iter_episodes(args.root) if r.episode_index == args.episode)
    duration = float(record.frame_timestamps[-1]) - float(record.frame_timestamps[0])
    print(f"\nepisode frames={record.row_count} duration={duration:.2f}s task={record.episode_task!r}")
    resolved_cameras = module._align_camera_keys(warn=False)
    timestamps = module._align_sample_timestamps(record, camera_count=len(resolved_cameras))
    print(
        f"planned timestamps: {len(timestamps)}  "
        f"planned camera-frames: {len(timestamps) * len(resolved_cameras)}  "
        f"resolved cameras: {resolved_cameras}"
    )

    started = time.perf_counter()
    spans = module._align_given_subtasks(record, labels, record.episode_task)
    elapsed = time.perf_counter() - started
    print(f"alignment wall time: {elapsed:.2f}s")

    for call_i, (batch, kwargs, results) in enumerate(captured):
        n_sheets = sum(
            1
            for m in batch[0]
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") == "image"
        )
        print(f"\n===== VLM call {call_i}: {len(batch)} message(s), {n_sheets} sheets, kwargs={kwargs}")
        for result in results:
            if result is None:
                print("    *** None — JSON parse failed after retry (reply dropped entirely)")
            else:
                print(f"    {json.dumps(result)[:800]}")
    if len(captured) != 1:
        print(f"\n*** {len(captured)} VLM calls — alignment must issue exactly one")

    # ---- boundary error, only meaningful when the labels ARE the ground truth
    print("\n===== spans =====")
    if args.labels == "gt":
        by_text = {str(s["text"]): s for s in spans}
        print(f"  {'idx':>3}  {'predicted':>17}  {'ground truth':>17}  {'|dstart|':>8}  label")
        errs = []
        for i, label in enumerate(labels):
            g = (float(gt[i]["start"]), float(gt[i]["end"]))
            hit = by_text.get(label)
            if hit is None:
                print(f"  {i:>3}  {'(not placed)':>17}  {g[0]:7.2f}->{g[1]:7.2f}  {'--':>8}  {label}")
                continue
            p = (float(hit["start"]), float(hit["end"]))
            errs.append(abs(p[0] - g[0]))
            print(f"  {i:>3}  {p[0]:7.2f}->{p[1]:7.2f}  {g[0]:7.2f}->{g[1]:7.2f}  {errs[-1]:8.2f}  {label}")
        if errs:
            print(
                f"\n  placed {len(by_text)}/{len(labels)}   mean |start error| = {sum(errs) / len(errs):.2f}s"
            )

        n = int(duration) + 1
        ok = sum(1 for k in range(n) if label_at(spans, float(k), "text") == label_at(gt, float(k), "label"))
        print(f"  per-second label agreement: {ok}/{n} = {ok / n:.1%}")

        metrics = evaluate_alignment(labels, gt, spans)
        finite_summary = (
            f"MAE={metrics.boundary_mae:.2f}s median={metrics.boundary_median:.2f}s "
            f"p90={metrics.boundary_p90:.2f}s"
            if metrics.boundary_mae is not None
            and metrics.boundary_median is not None
            and metrics.boundary_p90 is not None
            else "no internal boundaries placed"
        )
        hit_summary = " ".join(
            f"B@{threshold:g}s={rate:.1%}" for threshold, rate in metrics.boundary_hit_rates.items()
        )
        print(
            "  robust metrics: "
            f"placed={metrics.placed_fraction:.1%} macro-tIoU={metrics.macro_temporal_iou:.1%} "
            f"internal-boundaries[{finite_summary}; {hit_summary}]"
        )
    else:
        metrics = None
        print("  [predicted]")
        for s in spans:
            print(f"     {s['start']:7.2f} -> {s['end']:7.2f}  {s['text']}")
        print("  [ground truth]")
        for s in gt:
            print(f"     {s['start']:7.2f} -> {s['end']:7.2f}  {s['label']}")

    if args.dump:
        args.dump.write_text(
            json.dumps(
                {
                    "episode": args.episode,
                    "labels": labels,
                    "gt": gt,
                    "spans": spans,
                    "raw": [r for _b, _k, rs in captured for r in rs],
                    "metrics": asdict(metrics) if metrics is not None else None,
                    "elapsed_s": elapsed,
                    "requested_cameras": cameras,
                    "resolved_cameras": resolved_cameras,
                    "sampling": args.sampling,
                    "frames_per_second": args.fps,
                    "max_camera_frames": args.max_frames,
                    "frame_width": args.frame_width,
                    "temperature": args.temperature,
                    "model_id": args.model,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.dump}")
    cleanup_subtasks()
    atexit.unregister(cleanup_subtasks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
