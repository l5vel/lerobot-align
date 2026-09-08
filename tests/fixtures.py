from __future__ import annotations

from pathlib import Path

import pandas as pd
from lerobot.datasets.io_utils import write_tasks
from lerobot.utils.io_utils import write_json


def _write_sarm_episode_meta(
    root: Path,
    episode_specs: list[tuple[int, int, str]],
    episode_subtasks: dict[int, list[tuple[str, float, float]]],
    prefix: str,
) -> None:
    """Write ``meta/episodes`` carrying SARM-style subtask columns."""
    stem = f"{prefix}_" if prefix else ""
    rows = []
    for episode_index, _num_frames, _task_text in episode_specs:
        triples = episode_subtasks.get(episode_index)
        rows.append(
            {
                "episode_index": episode_index,
                f"{stem}subtask_names": [t[0] for t in triples] if triples else None,
                f"{stem}subtask_start_times": [float(t[1]) for t in triples] if triples else None,
                f"{stem}subtask_end_times": [float(t[2]) for t in triples] if triples else None,
            }
        )
    meta_dir = root / "meta" / "episodes" / "chunk-000"
    meta_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(meta_dir / "file-000.parquet", index=False)


def build_annotation_dataset(
    root: Path,
    episode_specs: list[tuple[int, int, str]],
    fps: int = 10,
    episode_frame_tasks: dict[int, list[str]] | None = None,
    episode_subtasks: dict[int, list[tuple[str, float, float]]] | None = None,
    sarm_prefix: str = "dense",
) -> Path:
    """Build a minimal on-disk LeRobot dataset fixture for annotation tests."""
    data_dir = root / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame_tasks_by_episode = episode_frame_tasks or {}

    tasks: dict[int, str] = {}

    def _task_index(task_text: str) -> int:
        if task_text not in tasks.values():
            tasks[len(tasks)] = task_text
        return next(k for k, v in tasks.items() if v == task_text)

    for episode_index, num_frames, task_text in episode_specs:
        per_frame = frame_tasks_by_episode.get(episode_index)
        if per_frame is not None:
            if len(per_frame) != num_frames:
                raise ValueError(
                    f"episode {episode_index}: episode_frame_tasks has {len(per_frame)} entries "
                    f"but the episode has {num_frames} frames"
                )
            task_indices = [_task_index(t) for t in per_frame]
        else:
            task_indices = [_task_index(task_text)] * num_frames
        frame = pd.DataFrame(
            {
                "episode_index": [episode_index] * num_frames,
                "frame_index": list(range(num_frames)),
                "timestamp": [round(i / fps, 6) for i in range(num_frames)],
                "task_index": task_indices,
                "subtask_index": [0] * num_frames,
            }
        )
        frame.to_parquet(data_dir / f"file-{episode_index:03d}.parquet", index=False)

    if episode_subtasks:
        _write_sarm_episode_meta(root, episode_specs, episode_subtasks, sarm_prefix)

    tasks_df = pd.DataFrame(
        {"task_index": list(tasks.keys())},
        index=pd.Index(list(tasks.values()), name="task"),
    )
    write_tasks(tasks_df, root)

    write_json(
        {
            "codebase_version": "v3.1",
            "fps": fps,
            "features": {},
            "total_episodes": len(episode_specs),
        },
        root / "meta" / "info.json",
    )
    return root


def add_synthetic_video(root: Path, *, camera: str = "observation.images.top") -> Path:
    """Encode original synthetic frames and complete v3 video metadata, without Hub access."""
    import json
    from PIL import Image, ImageDraw
    from lerobot_align.frames import encode_frames_to_clip
    from lerobot_align.reader import iter_episodes

    records = list(iter_episodes(root))
    rows = []
    for record in records:
        path = root / "videos" / camera / "chunk-000" / f"file-{record.episode_index:03d}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        frames = []
        for index, _timestamp in enumerate(record.frame_timestamps):
            frame = Image.new("RGB", (160, 120), (20, 40, 80))
            ImageDraw.Draw(frame).rectangle((10 + index, 20, 40 + index, 60), fill=(220, 80, 40))
            frames.append(frame)
        encode_frames_to_clip(frames, list(record.frame_timestamps), path, frame_width=160)
        rows.append(
            {
                "episode_index": record.episode_index,
                "data/chunk_index": 0,
                "data/file_index": record.episode_index,
                f"videos/{camera}/chunk_index": 0,
                f"videos/{camera}/file_index": record.episode_index,
                f"videos/{camera}/from_timestamp": 0.0,
                f"videos/{camera}/to_timestamp": record.frame_timestamps[-1] + 0.1,
            }
        )
    path = root / "meta/episodes/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text())
    info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    info["features"][camera] = {
        "dtype": "video",
        "shape": [120, 160, 3],
        "names": ["height", "width", "channels"],
        "info": {"video.fps": info["fps"], "video.is_depth_map": False},
    }
    info_path.write_text(json.dumps(info))
    return root
