#!/usr/bin/env python
"""Prompt-token cost of the same episode 0 frames as contact sheets vs one clip."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path

from lerobot_align.config import PlanConfig, VlmConfig
from lerobot_align.diagnostics._camera import CAMERA_HELP, resolve_camera_key
from lerobot_align.frames import (
    encode_frames_to_clip,
    make_frame_provider,
    to_contact_sheet_blocks,
    to_video_url_block,
)
from lerobot_align.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
)
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import _to_openai_messages, make_vlm_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--camera", default=None, help=CAMERA_HELP)
    ap.add_argument("--model", default=VlmConfig().model_id)
    args = ap.parse_args()
    camera = resolve_camera_key(args.root, args.camera)

    cfg = PlanConfig(
        frames_per_second=3.0,
        max_frames_per_prompt=300,
        contact_sheet_frame_width=336,
        subtask_align_camera_keys=(camera,),
    )
    vlm = make_vlm_client(
        VlmConfig(model_id=args.model, api_base="http://127.0.0.1:8000/v1", api_key="EMPTY",
                  auto_serve=False, camera_key=camera, client_concurrency=1)
    )
    provider = make_frame_provider(args.root, camera_key=camera)
    module = PlanSubtasksMemoryModule(vlm=vlm, config=cfg, frame_provider=provider, root=args.root)
    record = next(r for r in iter_episodes(args.root) if r.episode_index == args.episode)

    timestamps = module._align_sample_timestamps(record, camera_count=1)
    frames = provider.frames_at(record, timestamps, camera_key=camera)
    print(f"episode {args.episode}: {len(frames)} frames")

    sheets = to_contact_sheet_blocks(
        frames, timestamps,
        columns=cfg.contact_sheet_columns,
        frames_per_sheet=cfg.contact_sheet_frames_per_sheet,
        frame_width=cfg.contact_sheet_frame_width,
        quality=cfg.contact_sheet_quality,
    )
    clip_fd, clip_name = tempfile.mkstemp(prefix="lerobot-align-tokens-", suffix=".mp4")
    os.close(clip_fd)
    clip = Path(clip_name)
    encode_frames_to_clip(frames, timestamps, clip, frame_width=cfg.contact_sheet_frame_width)
    video = to_video_url_block(clip.as_uri(), fps=None)
    print(f"contact sheets: {len(sheets)} images    clip: {clip.stat().st_size / 1e6:.1f} MB")

    def prompt_tokens(blocks, label):
        content = [*blocks, {"type": "text", "text": "Reply with 1."}]
        messages = [{"role": "user", "content": content}]
        api_messages, mm_kwargs = _to_openai_messages(messages)
        payload = {"model": args.model, "messages": api_messages, "max_tokens": 1, "temperature": 0}
        if mm_kwargs:
            payload["mm_processor_kwargs"] = mm_kwargs
        req = urllib.request.Request(
            "http://127.0.0.1:8000/v1/chat/completions",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"},
        )
        usage = json.load(urllib.request.urlopen(req, timeout=600))["usage"]
        print(f"{label:<16} prompt_tokens = {usage['prompt_tokens']:>7,}  "
              f"({usage['prompt_tokens'] / len(frames):.1f} per frame)")
        return usage["prompt_tokens"]

    try:
        sheet_tokens = prompt_tokens(sheets, "contact sheets")
        video_tokens = prompt_tokens(video, "video clip")
    finally:
        clip.unlink(missing_ok=True)

    print(f"\nvideo uses {sheet_tokens / video_tokens:.2f}x fewer prompt tokens "
          f"({sheet_tokens - video_tokens:,} saved)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
