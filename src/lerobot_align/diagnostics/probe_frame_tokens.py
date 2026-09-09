#!/usr/bin/env python
"""Compare official Qwen visual-token counts for one matched frame set.

This probe counts processor-expanded visual placeholders from ``grid_thw``.
It intentionally does not use an inference server's ``usage.prompt_tokens``:
that telemetry has been observed to count image and video payloads differently
and cannot support a cross-format comparison.
"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

from lerobot_align.diagnostics._camera import CAMERA_HELP, resolve_camera_key
from lerobot_align.frames import encode_frames_to_clip, make_frame_provider, to_contact_sheet_blocks
from lerobot_align.modules.plan_subtasks_memory import PlanSubtasksMemoryModule
from lerobot_align.reader import iter_episodes


def _visual_tokens(grids: list[list[int]], merge_size: int) -> int:
    divisor = merge_size**2
    total = 0
    for grid in grids:
        patches = math.prod(grid)
        if patches % divisor:
            raise ValueError(f"grid {grid!r} is not divisible by merge_size**2={divisor}")
        total += patches // divisor
    return total


def _inspect(name: str, batch: dict, processor: object, *, video: bool) -> int:
    grid_key = "video_grid_thw" if video else "image_grid_thw"
    processor_part = processor.video_processor if video else processor.image_processor
    token_id = processor.video_token_id if video else processor.image_token_id
    grids = batch[grid_key].tolist()
    visual_tokens = _visual_tokens(grids, int(processor_part.merge_size))
    placeholders = int((batch["input_ids"][0] == token_id).sum().item())
    if placeholders != visual_tokens:
        raise ValueError(
            f"{name}: grid count {visual_tokens} != input placeholder count {placeholders}"
        )
    print(
        f"{name:<16} visual_tokens = {visual_tokens:>7,}  "
        f"total_input_ids = {int(batch['input_ids'].shape[-1]):>7,}  grids = {grids}"
    )
    return visual_tokens


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default=None, help=CAMERA_HELP)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B", help="processor model ID")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--frame-width", type=int, default=224)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="load processor files only from the local Hugging Face cache",
    )
    args = parser.parse_args()
    if args.fps <= 0 or args.max_frames <= 0 or args.frame_width <= 0:
        parser.error("--fps, --max-frames, and --frame-width must be positive")

    from transformers import AutoProcessor

    camera = resolve_camera_key(args.root, args.camera)
    record = next(iter_episodes(args.root, only_episodes=(args.episode,)), None)
    if record is None:
        parser.error(f"episode {args.episode} does not exist")
    duration = float(record.frame_timestamps[-1]) - float(record.frame_timestamps[0])
    count = min(max(1, round(duration * args.fps) + 1), args.max_frames)
    timestamps = PlanSubtasksMemoryModule._uniform_episode_timestamps(record, count)
    provider = make_frame_provider(args.root, camera_key=camera)
    frames = provider.frames_at(record, timestamps, camera_key=camera, fail_on_error=True)
    if len(frames) != len(timestamps):
        raise ValueError(f"decoded {len(frames)} of {len(timestamps)} requested frames")

    processor = AutoProcessor.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
    )
    prompt = "Reply with the single character 1."
    sheets = to_contact_sheet_blocks(
        frames,
        timestamps,
        columns=5,
        frames_per_sheet=20,
        frame_width=args.frame_width,
        quality=84,
    )
    image_messages = [
        {
            "role": "user",
            "content": [
                *({"type": "image", "image": block["image"]} for block in sheets),
                {"type": "text", "text": prompt},
            ],
        }
    ]
    image_batch = processor.apply_chat_template(
        image_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    with tempfile.TemporaryDirectory(prefix="lerobot-align-tokens-") as temp_dir:
        clip = Path(temp_dir) / "sampled.mp4"
        encoded_fps = encode_frames_to_clip(
            frames,
            timestamps,
            clip,
            frame_width=args.frame_width,
            crf=0,
        )
        if encoded_fps is None:
            raise ValueError("video encoding failed")
        video_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(clip)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        video_batch = processor.apply_chat_template(
            video_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_sample_frames": False},
        )

    print(
        f"episode {args.episode}: {len(frames)} matched frames, "
        f"{len(sheets)} contact sheets, encoded video {encoded_fps:.6g} fps"
    )
    sheet_tokens = _inspect("contact sheets", image_batch, processor, video=False)
    video_tokens = _inspect("native video", video_batch, processor, video=True)
    reduction = 1.0 - video_tokens / sheet_tokens
    print(f"native video uses {reduction:.1%} fewer processor-expanded visual tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
