#!/usr/bin/env python
"""Reproduce the comparison that retired the per-tile alignment reply format.

An earlier align path asked for one [timestamp, index] pair per contact-sheet
tile and recovered spans by run-length encoding. The shipped path no longer
does: it asks for one {start, end} per supplied label, in a single call. This
probe issues that span-shaped request over the production sampler's own contact
sheets and scores it against ``meta/lerobot_annotations.json`` -- the
measurement the decision rested on. It is kept as that reproduction, not as a
live A/B: the per-tile arm it was compared against is gone from the production
code, so only the surviving format is exercised here.

Where the outcome is recorded: the docstring of
``PlanSubtasksMemoryModule._align_given_subtasks``
(``src/lerobot_align/modules/plan_subtasks_memory.py``, lines 1586-1598).

``SPAN_PROMPT`` below is this probe's own copy of the request, so it does not
track later edits to the shipped ``prompts/plan_subtask_align.txt``.

    python probe_align_format.py ROOT --episode 0 --fps 1.0
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
    _contact_sheet_preamble,
)
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import make_vlm_client

SPAN_PROMPT = """You are timing a FIXED, ORDERED list of subtasks against a robot
demonstration shown as timestamped contact sheets.

The operator's goal was: "{episode_task}"

These subtasks describe what the robot does, in the order it does them. Do NOT
rewrite, rename, split, merge, reorder or invent subtasks. Your ONLY job is to
say WHEN each one happens.

Subtasks:
{subtask_list}

The episode runs from 0.00s to {duration:.2f}s.

For each subtask, give the start and end time in seconds, read off the burned-in
tile timestamps. The subtasks are consecutive: each one's end is the next one's
start. The first starts at 0.00 and the last ends at {duration:.2f}.

If a listed subtask genuinely never happens, give it "start": null, "end": null.

Output strictly valid JSON and nothing else:

  {{
    "subtasks": [
      {{"index": 0, "start": <seconds>, "end": <seconds>}},
      ...
    ]
  }}
"""


def gt_index_at(gt: list[dict], t: float) -> int | None:
    for i, span in enumerate(gt):
        if float(span["start"]) <= t < float(span["end"]):
            return i
    return len(gt) - 1 if t >= float(gt[-1]["end"]) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--camera", default=None, help=CAMERA_HELP)
    ap.add_argument("--frame-width", type=int, default=336)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--model", default=VlmConfig().model_id)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()
    camera = resolve_camera_key(args.root, args.camera)

    ann = json.loads((args.root / "meta" / "lerobot_annotations.json").read_text(encoding="utf-8"))
    gt = ann["episodes"][str(args.episode)]["subtasks"]
    labels = [s["label"] for s in gt]

    vlm_cfg = VlmConfig(
        model_id=args.model,
        api_base=args.api_base,
        api_key="EMPTY",
        auto_serve=False,
        camera_key=camera,
        chat_template_kwargs={"enable_thinking": False},
        max_new_tokens=2048,
        temperature=args.temperature,
        client_concurrency=4,
    )
    vlm = make_vlm_client(vlm_cfg)

    record = next(r for r in iter_episodes(args.root) if r.episode_index == args.episode)
    duration = float(record.frame_timestamps[-1]) - float(record.frame_timestamps[0])
    n_tiles = max(1, round(duration * args.fps) + 1)

    plan_cfg = PlanConfig(
        frames_per_second=args.fps,
        max_frames_per_prompt=n_tiles,
        contact_sheet_frame_width=args.frame_width,
    )
    provider = make_frame_provider(args.root, camera_key=camera)
    module = PlanSubtasksMemoryModule(vlm=vlm, config=plan_cfg, frame_provider=provider, root=args.root)

    blocks = module._episode_video_block(record)
    print(f"episode {args.episode}: {duration:.2f}s, {n_tiles} tiles, {len(blocks)} sheets")

    prompt = _contact_sheet_preamble(plan_cfg.contact_sheet_columns) + SPAN_PROMPT.format(
        episode_task=record.episode_task,
        subtask_list="\n".join(f"{i}. {t}" for i, t in enumerate(labels)),
        duration=duration,
    )
    messages = [{"role": "user", "content": [*blocks, {"type": "text", "text": prompt}]}]

    results = vlm.generate_json([messages] * args.repeat, max_new_tokens=2048)

    for run, result in enumerate(results):
        print(f"\n===== run {run} =====")
        if not isinstance(result, dict):
            print(f"  bad reply: {result!r}")
            continue
        entries = result.get("subtasks")
        if not isinstance(entries, list):
            print(f"  no 'subtasks' key; got {list(result)}")
            continue
        pred: dict[int, tuple[float, float]] = {}
        for e in entries:
            if not isinstance(e, dict):
                continue
            idx = e.get("index")
            s, t_end = e.get("start"), e.get("end")
            if idx is None or s is None or t_end is None:
                continue
            try:
                pred[int(idx)] = (float(s), float(t_end))
            except (TypeError, ValueError):
                continue

        print(f"  {'idx':>3}  {'predicted':>17}  {'ground truth':>17}  {'|dstart|':>8}  label")
        errs = []
        for i, label in enumerate(labels):
            g = (float(gt[i]["start"]), float(gt[i]["end"]))
            if i not in pred:
                print(f"  {i:>3}  {'(not placed)':>17}  {g[0]:7.2f}->{g[1]:7.2f}  {'--':>8}  {label}")
                continue
            p = pred[i]
            d = abs(p[0] - g[0])
            errs.append(d)
            print(f"  {i:>3}  {p[0]:7.2f}->{p[1]:7.2f}  {g[0]:7.2f}->{g[1]:7.2f}  {d:8.2f}  {label}")
        if errs:
            print(f"  placed {len(pred)}/{len(labels)}   mean |start error| = {sum(errs) / len(errs):.2f}s")

        # per-frame agreement, sampled at 1 Hz
        n = int(duration) + 1
        ok = 0
        for k in range(n):
            t = float(k)
            truth = gt_index_at(gt, t)
            got = next((i for i, (s, e) in sorted(pred.items()) if s <= t < e), None)
            if truth is not None and got == truth:
                ok += 1
        print(f"  per-second label agreement: {ok}/{n} = {ok / n:.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
