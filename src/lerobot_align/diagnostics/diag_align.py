#!/usr/bin/env python
"""Run one alignment call against the live VLM and dump the raw reply.

python diag_align.py /path/to/dataset_root [episode_index] [fps] --subtasks labels.json

``--subtasks`` is the same file ``PlanConfig.subtasks_path`` takes: a JSON list
of labels for every episode, or an object keyed by episode index.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import tempfile
from pathlib import Path

from lerobot_align.config import PlanConfig, VlmConfig
from lerobot_align.diagnostics._camera import CAMERA_HELP, resolve_camera_key
from lerobot_align.frames import make_frame_provider
from lerobot_align.modules.plan_subtasks_memory import (
    PlanSubtasksMemoryModule,
    _load_subtasks_file,
)
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import make_vlm_client


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("episode", type=int, nargs="?", default=0)
    ap.add_argument("fps", type=float, nargs="?", default=1.0)
    ap.add_argument(
        "--subtasks",
        type=Path,
        required=True,
        help="JSON file of subtask labels, in PlanConfig.subtasks_path format",
    )
    ap.add_argument("--camera", default=None, help=CAMERA_HELP)
    ap.add_argument("--model", default=VlmConfig().model_id)
    ap.add_argument("--api-base", default="http://127.0.0.1:8000/v1")
    args = ap.parse_args()

    root = args.root
    want_ep = args.episode
    fps = args.fps
    subtasks = _load_subtasks_file(args.subtasks).for_episode(want_ep)
    camera = resolve_camera_key(root, args.camera)

    # PlanConfig wants a path of its own. Write the resolved labels to a
    # private temporary file rather than into this package's directory, which
    # under `pip install` is site-packages; close the descriptor mkstemp
    # returns before reopening the path, and remove it on exit.
    subtasks_fd, subtasks_name = tempfile.mkstemp(prefix="lerobot-align-diag-", suffix=".json")
    os.close(subtasks_fd)
    subtasks_path = Path(subtasks_name)

    def cleanup_subtasks() -> None:
        subtasks_path.unlink(missing_ok=True)

    atexit.register(cleanup_subtasks)
    subtasks_path.write_text(json.dumps(subtasks), encoding="utf-8")

    vlm_cfg = VlmConfig(
        model_id=args.model,
        api_base=args.api_base,
        api_key="EMPTY",
        auto_serve=False,
        camera_key=camera,
        chat_template_kwargs={"enable_thinking": False},
        max_new_tokens=4096,
        client_concurrency=4,
    )
    vlm = make_vlm_client(vlm_cfg)

    raw_calls: list = []
    original = vlm.generate_json

    def spy(messages_batch, **kwargs):
        results = original(messages_batch, **kwargs)
        raw_calls.append((messages_batch, kwargs, results))
        return results

    vlm.generate_json = spy  # type: ignore[method-assign]

    plan_cfg = PlanConfig(
        subtasks_path=subtasks_path,
        frames_per_second=fps,
        max_frames_per_prompt=300,
        contact_sheet_frame_width=336,
        n_task_rephrasings=0,
        emit_plan=False,
        emit_memory=False,
    )
    provider = make_frame_provider(root, camera_key=camera)
    print(f"frame provider: {type(provider).__name__} camera={getattr(provider, 'camera_key', None)}")

    module = PlanSubtasksMemoryModule(vlm=vlm, config=plan_cfg, frame_provider=provider, root=root)

    record = next(r for r in iter_episodes(root) if r.episode_index == want_ep)
    print(
        f"episode {record.episode_index}: {record.row_count} frames, "
        f"{record.frame_timestamps[0]:.2f} -> {record.frame_timestamps[-1]:.2f}s"
    )

    tiles = module._align_sample_timestamps(record)
    print(f"tiles requested: {len(tiles)}  (first 5: {[round(t, 2) for t in tiles[:5]]})")

    spans = module._align_given_subtasks(record, subtasks, record.episode_task)

    for i, (batch, kwargs, results) in enumerate(raw_calls):
        print(f"\n===== VLM call {i}  kwargs={kwargs} =====")
        n_img = sum(
            1
            for m in batch[0]
            for b in (m.get("content") or [])
            if isinstance(b, dict) and b.get("type") != "text"
        )
        print(f"images in first message: {n_img}")
        for j, result in enumerate(results):
            print(f"--- reply {j}: type={type(result).__name__}")
            text = json.dumps(result)[:1500] if result is not None else "None"
            print(f"    {text}")
            if isinstance(result, dict):
                print(f"    keys = {list(result.keys())}")

    print(f"\n===== resulting spans ({len(spans)}) =====")
    for s in spans:
        print(f"  {s['start']:7.2f} -> {s['end']:7.2f}  {s['text']}")
    cleanup_subtasks()
    atexit.unregister(cleanup_subtasks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
