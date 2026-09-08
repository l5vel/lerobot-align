#!/usr/bin/env python
"""Check fix 4a (clock-formatted "MM:SS.ss" timestamps in the align reply)
against a REAL episode, not the tiny synthetic fixture the unit test uses.

This is a one-off verification script, not a pytest test: it needs a real
dataset root, and ``--live`` needs a served model.

Two checks:

1. Deterministic, no VLM call: craft a reply that echoes the real human
   ground-truth boundaries (meta/lerobot_annotations.json) in clock format,
   feed it through the real ``_align_given_subtasks`` pipeline via a stub VLM
   client, and confirm the recovered spans exactly match ground truth. Also
   replays the same reply through the PRE-FIX coercion logic to show what
   would have happened without the fix (every entry dropped).

2. Live, one VLM call per episode: ask the real model to answer in "MM:SS.ss"
   format explicitly, to see whether it can even follow that instruction, and
   whether the real reply parses.

    python -m lerobot_align.diagnostics.check_clock_format ROOT --episode 0 --live
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lerobot_align.config import PlanConfig, VlmConfig
from lerobot_align.diagnostics._camera import CAMERA_HELP, resolve_camera_key
from lerobot_align.frames import make_frame_provider
from lerobot_align.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
    _coerce_timestamp,
    _contact_sheet_preamble,
)
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import StubVlmClient, make_vlm_client


def _clock(t: float) -> str:
    """Real seconds -> "MM:SS.ss" -- the format fix 4a added support for."""
    m, s = divmod(t, 60.0)
    return f"{int(m):02d}:{s:05.2f}s"


def _pre_fix_coerce(raw) -> float | None:
    """``_coerce_timestamp`` as it existed before fix 4a: strips a trailing
    "s" and calls ``float()`` -- no colon handling. Reproduced here (not
    imported) so this script keeps working after the real function changes
    further; it only needs to reflect what the OLD code did."""
    if isinstance(raw, str):
        raw = raw.strip().removesuffix("s")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def deterministic_check(root: Path, episode: int, camera: str) -> bool:
    ann = json.loads((root / "meta" / "lerobot_annotations.json").read_text(encoding="utf-8"))
    gt = ann["episodes"][str(episode)]["subtasks"]
    labels = [s["label"] for s in gt]

    print(f"### deterministic check: episode {episode}, {len(labels)} real ground-truth subtasks")
    print("    reply built by reformatting real GT boundaries as clock time, e.g.:")
    for s in gt[:2]:
        print(
            f"      {s['start']:7.2f}s -> {_clock(s['start'])!r}   {s['end']:7.2f}s -> {_clock(s['end'])!r}"
        )

    clock_reply = {
        "subtasks": [
            {"index": i, "start": _clock(s["start"]), "end": _clock(s["end"])} for i, s in enumerate(gt)
        ]
    }

    # Show what the PRE-FIX parser would have done with this exact reply.
    pre_fix_placed = sum(
        1
        for entry in clock_reply["subtasks"]
        if _pre_fix_coerce(entry["start"]) is not None and _pre_fix_coerce(entry["end"]) is not None
    )
    print(
        f"\n    pre-fix _coerce_timestamp would have placed {pre_fix_placed}/{len(labels)} "
        f"subtasks from this exact reply (0 expected: no colon handling)"
    )

    # Real pipeline, real episode, stubbed VLM returning the clock reply.
    def responder(_messages):
        return clock_reply

    vlm = StubVlmClient(responder=responder)
    plan_cfg = PlanConfig(
        subtasks_path=None,  # not used by _align_given_subtasks directly
        frames_per_second=1.0,
        max_frames_per_prompt=300,
        n_task_rephrasings=0,
        emit_plan=False,
        emit_memory=False,
    )
    provider = make_frame_provider(root, camera_key=camera)
    module = PlanSubtasksMemoryModule(vlm=vlm, config=plan_cfg, frame_provider=provider, root=root)
    record = next(r for r in iter_episodes(root) if r.episode_index == episode)

    spans = module._align_given_subtasks(record, labels, record.episode_task)

    print(
        f"\n    real pipeline (real episode {episode}, real frame timestamps) recovered {len(spans)}/{len(labels)} spans:"
    )
    ok = len(spans) == len(labels)
    for i, (span, truth) in enumerate(zip(spans, gt, strict=False)):
        start_ok = abs(float(span["start"]) - float(truth["start"])) < 0.5
        text_ok = span["text"] == truth["label"]
        ok = ok and start_ok and text_ok
        mark = "OK" if start_ok and text_ok else "MISMATCH"
        print(
            f"      [{mark:8}] idx={i} start={span['start']:7.2f}s (gt {truth['start']:7.2f}s)  {span['text']}"
        )
    print(
        f"\n    RESULT: {'PASS' if ok else 'FAIL'} — clock-formatted reply recovered exactly onto real GT boundaries"
    )
    return ok


def live_check(root: Path, episode: int, camera: str, model: str, api_base: str) -> None:
    print(f"\n### live check: episode {episode}, asking the real model to answer in MM:SS.ss")
    ann = json.loads((root / "meta" / "lerobot_annotations.json").read_text(encoding="utf-8"))
    gt = ann["episodes"][str(episode)]["subtasks"]
    labels = [s["label"] for s in gt]

    vlm_cfg = VlmConfig(
        model_id=model,
        api_base=api_base,
        api_key="EMPTY",
        auto_serve=False,
        camera_key=camera,
        chat_template_kwargs={"enable_thinking": False},
        max_new_tokens=2048,
        temperature=0.2,
        client_concurrency=4,
    )
    vlm = make_vlm_client(vlm_cfg)

    plan_cfg = PlanConfig(frames_per_second=3.0, max_frames_per_prompt=300, contact_sheet_frame_width=336)
    provider = make_frame_provider(root, camera_key=camera)
    module = PlanSubtasksMemoryModule(vlm=vlm, config=plan_cfg, frame_provider=provider, root=root)
    record = next(r for r in iter_episodes(root) if r.episode_index == episode)
    duration = float(record.frame_timestamps[-1]) - float(record.frame_timestamps[0])

    blocks = module._episode_video_block(record)
    prompt = _contact_sheet_preamble(plan_cfg.contact_sheet_columns) + (
        "You are timing a FIXED, ORDERED list of subtasks against a robot "
        "demonstration shown as timestamped contact sheets.\n\n"
        f'The operator\'s goal was: "{record.episode_task}"\n\n'
        "Subtasks:\n" + "\n".join(f"{i}. {t}" for i, t in enumerate(labels)) + "\n\n"
        f"The episode runs from 0.00s to {duration:.2f}s.\n\n"
        "For each subtask, give its start and end time. Report each time as "
        'clock format "MM:SS.ss" (minutes:seconds), NOT plain seconds.\n\n'
        'Output strictly valid JSON: {"subtasks": [{"index": 0, "start": "MM:SS.ss", "end": "MM:SS.ss"}, ...]}'
    )
    result = vlm.generate_json([[{"role": "user", "content": [*blocks, {"type": "text", "text": prompt}]}]])[
        0
    ]
    print(f"    raw reply: {json.dumps(result)[:600]}")
    if not isinstance(result, dict) or not isinstance(result.get("subtasks"), list):
        print("    *** no usable reply")
        return

    n_colon = sum(
        1
        for e in result["subtasks"]
        if isinstance(e, dict)
        for v in (e.get("start"), e.get("end"))
        if isinstance(v, str) and ":" in v
    )
    n_total = sum(
        1 for e in result["subtasks"] if isinstance(e, dict) for v in (e.get("start"), e.get("end"))
    )
    print(f"    model used clock format for {n_colon}/{n_total} timestamp fields when explicitly asked")

    n_parsed = sum(
        1
        for e in result["subtasks"]
        if isinstance(e, dict)
        and _coerce_timestamp(e.get("start")) is not None
        and _coerce_timestamp(e.get("end")) is not None
    )
    print(
        f"    current _coerce_timestamp parses {n_parsed}/{len(result['subtasks'])} entries from the real reply"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--camera", default=None, help=CAMERA_HELP)
    ap.add_argument("--model", default=VlmConfig().model_id)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--live", action="store_true", help="also ask the real model for clock-format output")
    args = ap.parse_args()

    camera = resolve_camera_key(args.root, args.camera)

    ok = deterministic_check(args.root, args.episode, camera)
    if args.live:
        live_check(args.root, args.episode, camera, args.model, args.api_base)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
