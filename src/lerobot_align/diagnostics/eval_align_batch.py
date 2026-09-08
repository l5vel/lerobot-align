#!/usr/bin/env python
"""Score fixed-label alignment across a whole dataset, one condition per format.

Runs the real ``_align_given_subtasks`` path over every episode that carries
human spans in ``meta/lerobot_annotations.json`` and reports per-episode and
aggregate agreement.

Both frame formats are evaluated per episode against the *same* decoded frames:
the uniform grid depends only on the episode and the budget, so the second
condition reads the first one's warm cache. That keeps the comparison paired
(no sampling difference between arms) and halves the decode cost.

    python eval_align_batch.py ROOT --formats contact_sheet video --out results.json

Results are appended to ``--out``-adjacent JSONL as they complete, so a run
that dies partway can still be aggregated: ``fit_align_calibration.py`` accepts
that sidecar in place of the ``--out`` array, taking the last row written for
each ``(episode, format)``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from contextlib import ExitStack
from unittest.mock import patch
from pathlib import Path

from lerobot_align.alignment_metrics import evaluate_alignment
from lerobot_align.config import PlanConfig, VlmConfig
from lerobot_align.diagnostics._camera import MULTI_CAMERA_HELP, resolve_camera_keys
from lerobot_align.diagnostics._provenance import api_key, evaluation_config, fingerprint, validate_options, validate_truth
from lerobot_align.frames import VideoFrameProvider
from lerobot_align.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
)
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import make_vlm_client


def human_spans(root: Path) -> dict[int, list[dict]]:
    payload = json.loads((root / "meta" / "lerobot_annotations.json").read_text(encoding="utf-8"))
    out: dict[int, list[dict]] = {}
    for key, episode in (payload.get("episodes") or {}).items():
        spans = episode.get("subtasks") or []
        if spans:
            out[int(key)] = spans
    return validate_truth(out)


def sidecar_path(out: Path) -> Path:
    """Where the append-only row log goes for a given ``--out``.

    ``--out run.jsonl`` would otherwise resolve the sidecar onto the output
    itself: the run appends rows to it and then overwrites them with the final
    array, so a second run that dies leaves a file that is neither form and the
    crash-recovery contract is void. Only that collision moves the name;
    ordinary ``--out x.json`` keeps its established ``x.jsonl``.
    """
    beside = out.with_suffix(".jsonl")
    return out.with_name(out.name + ".rows.jsonl") if beside == out else beside


def main() -> int:
    with ExitStack() as stack:
        return _main(stack)


def _main(stack: ExitStack) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all annotated")
    ap.add_argument("--camera", action="append", dest="cameras", help=MULTI_CAMERA_HELP)
    ap.add_argument(
        "--formats", nargs="+", default=["contact_sheet", "video"], choices=["contact_sheet", "video"]
    )
    ap.add_argument("--sampling", default="uniform", choices=["uniform", "motion_stratified"])
    ap.add_argument(
        "--video-fallback",
        default="contact_sheet",
        choices=["contact_sheet", "error"],
        help="match the evaluation arm's behavior when native-video encoding fails",
    )
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--frame-width", type=int, default=336)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--model", default=VlmConfig().model_id)
    ap.add_argument("--api-key-env", default="LEROBOT_VLM_API_KEY", help="environment variable containing the endpoint API key")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--out", type=Path, default=Path("align_batch.json"))
    ap.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="per-boundary offsets from fit_align_calibration.py",
    )
    ap.add_argument(
        "--video-prompt",
        default=None,
        help="prompt name to use instead of plan_subtask_align_video (A/B prompt wording)",
    )
    ap.add_argument(
        "--processor-fps",
        type=float,
        default=None,
        help="mm_processor_kwargs fps for the video path; the model default is 2, "
        "which discards frames from a denser clip",
    )
    args = ap.parse_args()
    validate_options(args, ap)
    endpoint_key = api_key(args, ap)
    cameras = resolve_camera_keys(args.root, args.cameras)

    if args.processor_fps:
        # The Qwen3-VL video processor resamples to 2 fps by default, silently
        # discarding frames the caller deliberately sampled. Asking for the
        # clip's own rate keeps all of them.
        import lerobot_align.modules.plan_subtasks_memory as plan_module

        base_block = plan_module.to_video_url_block

        def block_override(url, fps=None, mm_kwargs=None):
            return base_block(url, fps=args.processor_fps, mm_kwargs=mm_kwargs)

        stack.enter_context(patch.object(plan_module, "to_video_url_block", block_override))
        print(f"processor fps override: {args.processor_fps}")

    if args.video_prompt:
        # Swap only the video prompt, leaving the contact-sheet path alone, so
        # a prompt A/B is a one-variable change.
        import lerobot_align.modules.plan_subtasks_memory as plan_module

        base_load = plan_module.load_prompt

        def load_override(name: str) -> str:
            return base_load(args.video_prompt if name == "plan_subtask_align_video" else name)

        stack.enter_context(patch.object(plan_module, "load_prompt", load_override))
        print(f"video prompt override: {args.video_prompt}")

    truth = human_spans(args.root)
    wanted = args.episodes if args.episodes is not None else sorted(truth)
    episodes = [e for e in wanted if e in truth]
    missing = [e for e in wanted if e not in truth]
    if missing:
        print(f"skipping {len(missing)} episode(s) without human spans: {missing}")
    print(f"scoring {len(episodes)} episode(s) x {len(args.formats)} format(s): {args.formats}")

    if not episodes:
        ap.error("No selected episodes have human spans; nothing was evaluated")
    provenance = evaluation_config(args, cameras, args.root)
    config_fingerprint = fingerprint(provenance)
    vlm = make_vlm_client(
        VlmConfig(
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
    )

    # One provider for every episode and format: the uniform grid is identical
    # across formats, so the second format reads the first's decode. The cache
    # must hold a whole episode's frames or it thrashes and decodes twice.
    provider = VideoFrameProvider(
        root=args.root, camera_key=cameras[0], cache_size=max(512, args.max_frames * 2)
    )

    modules = {}
    for frame_format in args.formats:
        # No ``subtasks_path``: labels are passed to ``_align_given_subtasks``
        # per episode, which matters because they are not identical across the
        # dataset (episode 18 carries five spans, the rest six).
        cfg = PlanConfig(
            frames_per_second=args.fps,
            max_frames_per_prompt=args.max_frames,
            contact_sheet_frame_width=args.frame_width,
            subtask_align_min_fraction=0.0,
            subtask_align_camera_keys=tuple(cameras),
            subtask_align_sampling=args.sampling,
            subtask_align_frame_format=frame_format,
            subtask_video_fallback=args.video_fallback,
            subtask_align_calibration_path=args.calibration,
            n_task_rephrasings=0,
            emit_plan=False,
            emit_memory=False,
        )
        modules[frame_format] = PlanSubtasksMemoryModule(
            vlm=vlm, config=cfg, frame_provider=provider, root=args.root
        )

    jsonl = sidecar_path(args.out)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    # Stamped on every row because the sidecar is append-only: without it, two
    # scoring passes with different --model, --fps or --camera are
    # indistinguishable once they share a file, and whatever reads it back
    # cannot tell a retry of one condition from a mix of two.
    run_id = uuid.uuid4().hex[:8]
    print(f"run id: {run_id}")
    rows: list[dict] = []
    started_all = time.perf_counter()

    records = {r.episode_index: r for r in iter_episodes(args.root) if r.episode_index in set(episodes)}

    for position, episode in enumerate(episodes, start=1):
        record = records.get(episode)
        if record is None or record.row_count == 0:
            raise ValueError(f"episode {episode}: human spans exist but no dataset frames were found")
        gt = truth[episode]
        labels = [str(span["label"]) for span in gt]
        duration = float(record.frame_timestamps[-1]) - float(record.frame_timestamps[0])

        for frame_format in args.formats:
            started = time.perf_counter()
            try:
                spans = modules[frame_format]._align_given_subtasks(record, labels, record.episode_task)
                metrics = evaluate_alignment(labels, gt, spans)
                row = {
                    "episode": episode,
                    "run_id": run_id,
                    "config_fingerprint": config_fingerprint,
                    "provenance": provenance,
                    "format": frame_format,
                    "labels": len(labels),
                    "duration_s": round(duration, 2),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                    "spans": spans,
                    **asdict(metrics),
                }
            except Exception as exc:  # keep going; one bad episode is not the run
                row = {
                    "episode": episode,
                    "run_id": run_id,
                    "config_fingerprint": config_fingerprint,
                    "provenance": provenance,
                    "format": frame_format,
                    "labels": len(labels),
                    "duration_s": round(duration, 2),
                    "elapsed_s": round(time.perf_counter() - started, 1),
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=4),
                }
                print(f"    !! ep {episode} {frame_format}: {row['error']}")
            rows.append(row)
            with jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())

            if "error" not in row:
                first = row["boundary_errors"][0] if row["boundary_errors"] else None
                print(
                    f"[{position}/{len(episodes)}] ep {episode:>2} {frame_format:<13} "
                    f"tIoU={row['macro_temporal_iou']:6.1%} "
                    f"MAE={row['boundary_mae'] if row['boundary_mae'] is None else round(row['boundary_mae'], 2)}s "
                    f"B@3={row['boundary_hit_rates'].get(3.0, 0):5.1%} "
                    f"first={('--' if first is None else round(first, 2))}s "
                    f"placed={row['placed_fraction']:.0%} ({row['elapsed_s']}s)"
                )

    args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out} and {jsonl} in {(time.perf_counter() - started_all) / 60:.1f} min")
    summarize(rows, args.formats)
    return 1 if missing or any("error" in row for row in rows) or not rows else 0


def summarize(rows: list[dict], formats: list[str]) -> None:
    def mean(values):
        values = [v for v in values if v is not None]
        return statistics.fmean(values) if values else None

    def fmt(value, spec: str, suffix: str = "") -> str:
        return "--" if value is None else f"{value:{spec}}{suffix}"

    print("\n## Aggregate (macro-average over episodes)\n")
    print("| format | eps | errors | placed | macro-tIoU | boundary MAE | median | B@1 | B@3 | B@5 | first-boundary MAE |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for frame_format in formats:
        good = [r for r in rows if r["format"] == frame_format and "error" not in r]
        bad = [r for r in rows if r["format"] == frame_format and "error" in r]
        if not good:
            print(f"| {frame_format} | 0 | {len(bad)} | -- | -- | -- | -- | -- | -- | -- | -- |")
            continue
        firsts = [r["boundary_errors"][0] for r in good if r["boundary_errors"]]
        print(
            f"| {frame_format} | {len(good)} | {len(bad)} "
            f"| {fmt(mean(r['placed_fraction'] for r in good), '.1%')} "
            f"| {fmt(mean(r['macro_temporal_iou'] for r in good), '.2%')} "
            f"| {fmt(mean(r['boundary_mae'] for r in good), '.2f', 's')} "
            f"| {fmt(mean(r['boundary_median'] for r in good), '.2f', 's')} "
            f"| {fmt(mean(r['boundary_hit_rates'].get(1.0) for r in good), '.1%')} "
            f"| {fmt(mean(r['boundary_hit_rates'].get(3.0) for r in good), '.1%')} "
            f"| {fmt(mean(r['boundary_hit_rates'].get(5.0) for r in good), '.1%')} "
            f"| {fmt(mean(firsts), '.2f', 's')} |"
        )

    if len(formats) == 2:
        a, b = formats
        by_ep = {}
        for row in rows:
            if "error" not in row:
                by_ep.setdefault(row["episode"], {})[row["format"]] = row
        paired = [v for v in by_ep.values() if a in v and b in v]
        if paired:
            wins = sum(1 for v in paired if v[b]["macro_temporal_iou"] > v[a]["macro_temporal_iou"])
            deltas = [v[b]["macro_temporal_iou"] - v[a]["macro_temporal_iou"] for v in paired]
            mae_d = [
                v[b]["boundary_mae"] - v[a]["boundary_mae"]
                for v in paired
                if v[a]["boundary_mae"] is not None and v[b]["boundary_mae"] is not None
            ]
            print(
                f"\nPaired over {len(paired)} episode(s): {b} beats {a} on macro-tIoU in "
                f"{wins}/{len(paired)} ({wins / len(paired):.0%}); "
                f"mean delta {statistics.fmean(deltas):+.2%}, median {statistics.median(deltas):+.2%}"
            )
            if mae_d:
                print(
                    f"  boundary MAE delta: mean {statistics.fmean(mae_d):+.2f}s, "
                    f"median {statistics.median(mae_d):+.2f}s"
                )


if __name__ == "__main__":
    sys.exit(main())
