#!/usr/bin/env python
"""Does the model's native video timestamp equal episode time?

Burns the TRUE episode timestamp into each sampled frame, encodes the clip with
the production encoder, sends it as native video, and asks what is printed at a
given *native* time. Agreement means clip time == episode time, independent of
any task semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from lerobot_align.config import PlanConfig, VlmConfig
from lerobot_align.diagnostics._camera import CAMERA_HELP, resolve_camera_key
from lerobot_align.frames import (
    _draw_timestamp_badge,
    _frame_to_pil,
    encode_frames_to_clip,
    make_frame_provider,
    to_video_url_block,
)
from lerobot_align.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
)
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import make_vlm_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--camera", default=None, help=CAMERA_HELP)
    ap.add_argument("--model", default=VlmConfig().model_id)
    ap.add_argument("--probe-times", type=float, nargs="+", default=[10.0, 30.0, 60.0, 85.0])
    ap.add_argument("--processor-fps", type=float, default=None,
                    help="mm_processor_kwargs fps; the model default is 2, which drops frames")
    args = ap.parse_args()
    camera = resolve_camera_key(args.root, args.camera)

    vlm = make_vlm_client(
        VlmConfig(
            model_id=args.model,
            api_base="http://127.0.0.1:8000/v1",
            api_key="EMPTY",
            auto_serve=False,
            camera_key=camera,
            chat_template_kwargs={"enable_thinking": False},
            max_new_tokens=512,
            temperature=0.0,
            client_concurrency=1,
        )
    )
    plan_cfg = PlanConfig(
        frames_per_second=3.0,
        max_frames_per_prompt=300,
        contact_sheet_frame_width=336,
        subtask_align_camera_keys=(camera,),
        subtask_align_frame_format="video",
    )
    provider = make_frame_provider(args.root, camera_key=camera)
    module = PlanSubtasksMemoryModule(
        vlm=vlm, config=plan_cfg, frame_provider=provider, root=args.root
    )
    record = next(r for r in iter_episodes(args.root) if r.episode_index == args.episode)

    timestamps = module._align_sample_timestamps(record, camera_count=1)
    frames = provider.frames_at(record, timestamps, camera_key=camera)
    print(
        f"episode {args.episode}: {len(frames)} frames, "
        f"{timestamps[0]:.2f}s -> {timestamps[-1]:.2f}s"
    )

    badged = [_draw_timestamp_badge(_frame_to_pil(f).convert("RGB"), float(t))
              for f, t in zip(frames, timestamps, strict=True)]

    clip_fd, clip_name = tempfile.mkstemp(prefix="lerobot-align-timeline-", suffix=".mp4")
    os.close(clip_fd)
    clip = Path(clip_name)
    fps = encode_frames_to_clip(badged, timestamps, clip, frame_width=336)
    print(f"encoded at {fps:.4f} fps -> {clip.stat().st_size / 1e6:.1f} MB")

    asked = ", ".join(f"{t:.1f}" for t in args.probe_times)
    prompt = (
        "Every frame of this video has a number printed in its top-left corner, "
        "like 012.50s.\n\n"
        f"For each of these video times — {asked} seconds — report the number "
        "printed on the frame at that time.\n\n"
        'Answer strictly as JSON: {"readings": [{"asked": <seconds>, '
        '"printed": <the printed number, in seconds>}, ...]}'
    )
    blocks = to_video_url_block(clip.as_uri(), fps=args.processor_fps)
    messages = [{"role": "user", "content": [*blocks, {"type": "text", "text": prompt}]}]
    try:
        result = vlm.generate_json([messages], max_new_tokens=512)[0]
    finally:
        clip.unlink(missing_ok=True)

    print("\nraw:", json.dumps(result))
    readings = (result or {}).get("readings") or []
    print(f"\n  {'asked':>8} {'printed':>8} {'drift':>8}")
    drifts = []
    for item in readings:
        try:
            asked_t = float(item["asked"])
            printed = float(str(item["printed"]).rstrip("s"))
        except (KeyError, TypeError, ValueError):
            continue
        drifts.append(abs(printed - asked_t))
        print(f"  {asked_t:>8.2f} {printed:>8.2f} {printed - asked_t:>+8.2f}")
    if drifts:
        print(f"\nmean |drift| = {sum(drifts) / len(drifts):.2f}s over {len(drifts)} probe(s)")
        verdict = "clip time == episode time" if max(drifts) < 2.0 else "*** TIMELINE MISMATCH"
        print("VERDICT:", verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
