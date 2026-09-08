#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Datatrove-shaped reader.

The reader walks ``data/chunk-*/file-*.parquet`` and yields one record per
episode containing:

- ``episode_index``: int
- ``frame_timestamps``: tuple[float, ...]
- ``frame_indices``: tuple[int, ...]
- ``episode_task``: str (canonical task from ``meta/tasks.parquet``)
- ``frame_tasks``: tuple[str, ...] (the task string of *each* frame)
- ``data_path``: pathlib.Path of the source parquet shard
- ``frames_df``: pandas.DataFrame slice for the episode (only loaded on demand)

This shape lets each module operate per-episode without loading all parquet
rows into memory at once.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from lerobot.datasets.io_utils import load_tasks
from lerobot.datasets.utils import DEFAULT_TASKS_PATH

_SUPPORTED_CODEBASE_VERSION = re.compile(
    r"^v3\.(?:0|[1-9]\d*)(?:\.(?:0|[1-9]\d*))?$"
)


@dataclass
class EpisodeRecord:
    """Per-episode record yielded by the reader."""

    episode_index: int
    episode_task: str
    frame_timestamps: tuple[float, ...]
    frame_indices: tuple[int, ...]
    data_path: Path
    row_offset: int  # row offset within the parquet file where this episode starts
    row_count: int  # number of rows for this episode

    # Per-frame task string, resolved through ``meta/tasks.parquet``.
    # ``episode_task`` is just ``frame_tasks[0]``; the full sequence is kept
    # because a dataset recorded with the task changing mid-episode already
    # carries a subtask segmentation there (see ``subtask_import``). Costs one
    # pointer per frame, the same as ``frame_timestamps`` / ``frame_indices``.
    frame_tasks: tuple[str, ...] = ()

    # Memoized parquet slice — populated on first ``frames_df()`` call so
    # repeat queries from different modules don't re-read the whole shard.
    _frames_df_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def frames_df(self):  # type: ignore[no-untyped-def]
        """Lazy-load the pandas slice for this episode (memoized)."""
        if self._frames_df_cache is None:
            import pandas as pd  # noqa: PLC0415  - deferred for optional dataset extra

            table = pq.read_table(self.data_path)
            df: pd.DataFrame = table.to_pandas()
            self._frames_df_cache = df.iloc[self.row_offset : self.row_offset + self.row_count].reset_index(
                drop=True
            )
        return self._frames_df_cache

    def frame_columns(self, names: Sequence[str]) -> dict[str, list[Any]]:
        """Read selected per-frame columns for this episode only.

        Motion-aware alignment needs a few numeric observation vectors but not
        the images, language columns, or every other field in a potentially
        large shard. Reading only the requested parquet columns avoids the
        full-shard pandas conversion performed by :meth:`frames_df`.
        """
        unique_names = list(dict.fromkeys(str(name) for name in names if name))
        if not unique_names:
            return {}
        table = pq.read_table(self.data_path, columns=unique_names).slice(self.row_offset, self.row_count)
        return {name: table.column(name).to_pylist() for name in unique_names}


def reconstruct_subtask_spans(
    rows: Sequence[dict[str, Any]],
    *,
    episode_end_t: float | None = None,
) -> list[dict[str, Any]]:
    """Turn ``style="subtask"`` rows into ``{text, start, end}`` spans.

    Each span's ``end`` is the next span's ``start``. The final span's
    ``end`` defaults to its own ``start`` (zero-duration) — pass
    ``episode_end_t`` to extend it to the episode's last frame instead,
    which is what downstream consumers (memory, interjection boundary
    selection) expect.

    Used by the ``plan`` module (plan-update pass) and the
    ``interjections`` module (interjection anchoring), which both need the
    same span shape.
    """
    sorted_rows = sorted(
        (r for r in rows if r.get("style") == "subtask"),
        key=lambda r: float(r["timestamp"]),
    )
    spans: list[dict[str, Any]] = []
    for r in sorted_rows:
        t = float(r["timestamp"])
        if spans:
            spans[-1]["end"] = t
        spans.append({"text": r.get("content") or "", "start": t, "end": t})
    if spans and episode_end_t is not None and float(episode_end_t) > spans[-1]["start"]:
        spans[-1]["end"] = float(episode_end_t)
    return spans


def snap_to_frame(t: float, frame_timestamps: Sequence[float]) -> float:
    """Snap an arbitrary float to the nearest exact source frame timestamp.

    Modules use this when emitting event-style rows so the row's
    timestamp matches a real parquet frame: event rows must land on an
    exact frame, otherwise the per-frame event lookup the writer does
    would never match them.
    """
    if not frame_timestamps:
        return float(t)
    nearest = min(frame_timestamps, key=lambda f: abs(f - t))
    return float(nearest)


def _load_tasks_lookup(root: Path) -> dict[int, str]:
    """Map ``task_index -> task`` from ``meta/tasks.parquet``.

    Returns an empty dict when the file is absent — the task description is
    derived later from the video if needed. Reuses the library-level
    :func:`lerobot.datasets.io_utils.load_tasks`, which returns the tasks
    frame indexed by task string with a ``task_index`` column.
    """
    if not (root / DEFAULT_TASKS_PATH).exists():
        return {}
    tasks = load_tasks(root)
    return {int(idx): str(task) for task, idx in zip(tasks.index, tasks["task_index"], strict=True)}


def _require_parquet_storage(root: Path) -> None:
    """Fail clearly for dataset layouts this in-place writer cannot update."""
    info_path = root / "meta" / "info.json"
    metadata: dict[str, Any] = {}
    if info_path.exists():
        metadata = json.loads(info_path.read_text(encoding="utf-8"))
    storage_format = metadata.get("storage_format")

    # Current LeRobot Lance datasets declare the format in metadata and keep
    # their tabular rows in ``frames.lance``. Check both so a partially copied
    # dataset cannot silently look like an empty Parquet dataset.
    has_lance_table = (root / "frames.lance").exists()
    if storage_format not in (None, "lerobot") or has_lance_table:
        declared = storage_format if storage_format is not None else "lance"
        raise NotImplementedError(
            "lerobot-align supports only LeRobot's Parquet/MP4 storage layout; "
            f"dataset {root} uses storage_format={declared!r}. Convert the dataset "
            "to the default layout before annotating it."
        )

    codebase_version = metadata.get("codebase_version")
    if not isinstance(codebase_version, str) or _SUPPORTED_CODEBASE_VERSION.fullmatch(codebase_version) is None:
        raise ValueError(
            "lerobot-align supports only LeRobot Parquet datasets with a numeric v3.x "
            f"codebase_version; dataset {root} declares {codebase_version!r}. "
            "Convert the dataset to v3.0 or newer before annotating it."
        )


def iter_episodes(root: Path, *, only_episodes: tuple[int, ...] | None = None) -> Iterator[EpisodeRecord]:
    """Yield :class:`EpisodeRecord` for every episode under ``root/data/``.

    Episodes are yielded in ascending ``episode_index`` order. The reader does
    not assume a specific chunk/file layout: it scans every ``*.parquet``
    under ``data/`` and groups by ``episode_index``.
    """
    _require_parquet_storage(root)
    tasks = _load_tasks_lookup(root)
    data_dir = root / "data"
    parquet_files = sorted(data_dir.rglob("*.parquet"))

    only_set = set(only_episodes) if only_episodes is not None else None

    for path in parquet_files:
        yield from _iter_one_path(path, tasks, only_set)


def _iter_one_path(path: Path, tasks: dict[int, str], only_set: set[int] | None) -> Iterator[EpisodeRecord]:
    table = pq.read_table(path)
    names = table.column_names
    if "episode_index" not in names:
        return
    episode_col = table.column("episode_index").to_pylist()
    if "timestamp" not in names:
        raise ValueError(f"{path}: required timestamp column is missing")
    timestamp_col = table.column("timestamp").to_pylist()
    frame_col = (
        table.column("frame_index").to_pylist() if "frame_index" in names else list(range(len(episode_col)))
    )
    task_col = table.column("task_index").to_pylist() if "task_index" in names else None

    def _build(
        ep: int,
        start: int,
        end: int,
        ts_buf: list[float],
        fi_buf: list[int],
        ti_buf: list[int],
    ) -> EpisodeRecord | None:
        if only_set is not None and ep not in only_set:
            return None
        if any(isinstance(t, bool) or not isinstance(t, (int, float))
               or not math.isfinite(t) or t < 0 for t in ts_buf):
            raise ValueError(f"{path}: episode {ep} timestamps must be finite non-negative numbers")
        if any(b <= a for a, b in zip(ts_buf, ts_buf[1:], strict=False)):
            raise ValueError(f"{path}: episode {ep} timestamps must be strictly increasing")
        # Resolved per frame, not once per episode: an episode whose
        # ``task_index`` changes mid-way is already segmented, and collapsing
        # it here would throw that away.
        frame_tasks = tuple(tasks.get(task_idx, "") for task_idx in ti_buf)
        return EpisodeRecord(
            episode_index=ep,
            episode_task=frame_tasks[0] if frame_tasks else "",
            frame_timestamps=tuple(ts_buf),
            frame_indices=tuple(fi_buf),
            frame_tasks=frame_tasks,
            data_path=path,
            row_offset=start,
            row_count=end - start,
        )

    cur_ep: int | None = None
    start_offset = 0
    ts_buf: list[float] = []
    fi_buf: list[int] = []
    ti_buf: list[int] = []

    for i, ep in enumerate(episode_col):
        if cur_ep is None or ep != cur_ep:
            if cur_ep is not None:
                rec = _build(cur_ep, start_offset, i, ts_buf, fi_buf, ti_buf)
                if rec is not None:
                    yield rec
            cur_ep = ep
            start_offset = i
            ts_buf = [timestamp_col[i]]
            fi_buf = [frame_col[i]]
            ti_buf = [task_col[i]] if task_col is not None else []
        else:
            ts_buf.append(timestamp_col[i])
            fi_buf.append(frame_col[i])
            if task_col is not None:
                ti_buf.append(task_col[i])

    if cur_ep is not None:
        rec = _build(cur_ep, start_offset, len(episode_col), ts_buf, fi_buf, ti_buf)
        if rec is not None:
            yield rec
