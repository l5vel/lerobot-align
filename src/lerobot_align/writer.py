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
"""Final parquet rewrite.

For every episode the writer:

1. reads the staged module outputs,
2. partitions them into a persistent slice (PERSISTENT_STYLES) and an event
   slice (EVENT_ONLY_STYLES + style=None tool-call atoms),
3. sorts each slice deterministically,
4. broadcasts the persistent slice across every frame in the episode,
5. for each frame, materializes the sublist of event rows whose timestamp
   exactly equals that frame's timestamp,
6. drops the legacy ``subtask_index`` column,
7. writes the parquet shard back in place.

The writer does NOT add a dataset-level ``tools`` column. Tool *calls* are
emitted per-row via the existing ``tool_calls`` field on the v3.1 row
struct for every speech atom. The tool *schema* (the description
of the ``say`` function and its parameters) is a fixed code constant —
``SAY_TOOL_SCHEMA`` below — and downstream chat-template consumers import
it directly rather than reading a redundant per-row column.

Invariants enforced here (and re-checked by the validator):

- per-episode persistent slice is byte-identical across every frame;
- ``language_events`` rows on a frame all have ``timestamp == frame_ts``
  (timestamps come straight from the source parquet — never recomputed);
- every row passes ``column_for_style(style)``.
"""

from __future__ import annotations

import hashlib
import logging
import os
import stat
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.io_utils import write_table_one_row_group_per_episode
from lerobot.datasets.language import (
    EVENT_ONLY_STYLES,
    LANGUAGE_EVENTS,
    LANGUAGE_PERSISTENT,
    PERSISTENT_STYLES,
    column_for_style,
    validate_camera_field,
)

from .reader import EpisodeRecord
from .staging import EpisodeStaging

logger = logging.getLogger(__name__)


@contextmanager
def dataset_run_lock(root: Path) -> Iterator[None]:
    """Hold an exclusive lease for one complete local annotation run.

    Episode staging is shared beneath the dataset by default, so shard-level
    locks alone cannot stop overlapping runs from mixing module outputs. The
    non-blocking lease fails the second run immediately with an actionable
    error instead of letting it wait while a VLM server consumes resources.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX compatibility fallback
        raise RuntimeError("Dataset rewriting requires POSIX file locking; use a supported Linux system") from None

    lock_dir = root / ".annotate_staging" / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / "run.lock"
    with lock_path.open("a+b") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another lerobot-align process is already annotating {root}. "
                "Wait for it to finish before starting another run. "
                f"Coordination lock: {lock_path}"
            ) from exc
        try:
            from .transaction import recover_dataset
            recover_dataset(root)
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_shard_lock(root: Path, path: Path) -> Iterator[None]:
    """Serialize the read-modify-replace cycle for a selected shard."""
    try:
        import fcntl
    except ImportError:  # pragma: no cover - non-POSIX compatibility fallback
        logger.warning("File locking is unavailable on this platform; concurrent shard writes are unsafe")
        yield
        return

    digest = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()
    # Keep persistent lock in the staging tree rather than dataset content.
    # Lock files must not be unlinked after use: removing one while another
    # process waits on its inode can let a third process acquire a new inode.
    lock_dir = root / ".annotate_staging" / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{digest}.lock").open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


# Columns the writer strips out of every shard. ``meta/info.json`` must drop
# them too: a feature declared in metadata but absent from the parquet does not
# fail the load, it makes ``LeRobotDataset`` hand back ``None`` for that key on
# every frame. Keeping the names here lets the metadata sync in ``executor.py``
# stay in step with what the writer actually removes.
LEGACY_SUBTASK_INDEX_COLUMN = "subtask_index"
LEGACY_TOOLS_COLUMN = "tools"
LEGACY_ANNOTATION_COLUMNS = (LEGACY_SUBTASK_INDEX_COLUMN, LEGACY_TOOLS_COLUMN)


# Tool schema constants live in lerobot.datasets.language — single
# source of truth. Re-exported here so existing imports
# (``from lerobot_align.writer import SAY_TOOL_SCHEMA``)
# keep working.
from lerobot.datasets.language import DEFAULT_TOOLS, SAY_TOOL_SCHEMA  # noqa: F401, E402


def _row_persistent_sort_key(row: dict[str, Any]) -> tuple:
    return (float(row["timestamp"]), row.get("style") or "", row.get("role") or "")


def _row_event_sort_key(row: dict[str, Any]) -> tuple:
    # events are bucketed per-frame, but within a frame we still want determinism
    return (
        row.get("style") or "",
        row.get("role") or "",
        row.get("camera") or "",
    )


def _normalize_row(row: dict[str, Any], style: str | None, *, with_timestamp: bool) -> dict[str, Any]:
    """Coerce a staged row into the language-column struct shape.

    Key order matches ``PERSISTENT_ROW_FIELDS`` / ``EVENT_ROW_FIELDS`` — the
    writer infers the parquet struct schema from insertion order, so
    ``timestamp`` (persistent rows only) sits between ``style`` and ``camera``.
    """
    camera = row.get("camera")
    validate_camera_field(style, camera)
    out: dict[str, Any] = {
        "role": str(row["role"]),
        "content": None if row.get("content") is None else str(row["content"]),
        "style": style,
    }
    if with_timestamp:
        out["timestamp"] = float(row["timestamp"])
    out["camera"] = None if camera is None else str(camera)
    out["tool_calls"] = _normalize_tool_calls(row.get("tool_calls"))
    return out


def _normalize_persistent_row(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce a staged row into the persistent column's struct shape."""
    style = row.get("style")
    if style not in PERSISTENT_STYLES:
        raise ValueError(
            f"persistent slice contains row with non-persistent style {style!r}; "
            "row would be misrouted under column_for_style()"
        )
    if "timestamp" not in row:
        raise ValueError(f"persistent row missing timestamp: {row!r}")
    if "role" not in row:
        # Friendly error from the writer instead of a raw KeyError below;
        # the validator doesn't check ``role`` yet.
        raise ValueError(f"persistent row missing role: {row!r}")
    return _normalize_row(row, style, with_timestamp=True)


def _normalize_event_row(row: dict[str, Any]) -> dict[str, Any]:
    """Coerce a staged row into the event column's struct shape (no timestamp)."""
    style = row.get("style")
    if style is not None and style not in EVENT_ONLY_STYLES:
        raise ValueError(
            f"event slice contains row with style {style!r}; expected None or one of {EVENT_ONLY_STYLES}"
        )
    if column_for_style(style) != LANGUAGE_EVENTS:
        raise ValueError(f"event row with style {style!r} would not route to language_events")
    if "role" not in row:
        raise ValueError(f"event row missing role: {row!r}")
    return _normalize_row(row, style, with_timestamp=False)


def _normalize_tool_calls(value: Any) -> list[Any] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"tool_calls must be a list or None, got {type(value).__name__}")
    return list(value)


def _validate_atom_invariants(row: dict[str, Any]) -> None:
    """At-least-one of content/tool_calls; style=None implies tool_calls."""
    has_content = row.get("content") is not None
    has_tools = row.get("tool_calls") is not None
    if not (has_content or has_tools):
        raise ValueError(f"row has neither content nor tool_calls: {row!r}")
    if row.get("style") is None and not has_tools:
        raise ValueError(f"style=None requires tool_calls: {row!r}")


def _validate_speech_atom(row: dict[str, Any]) -> None:
    """Speech atoms: role=assistant, style=None, content=None, say tool call."""
    if row.get("style") is not None:
        return  # not a speech atom
    if row.get("role") != "assistant":
        raise ValueError(f"speech atom must have role=assistant: {row!r}")
    if row.get("content") is not None:
        raise ValueError(f"speech atom must have content=null: {row!r}")
    tool_calls = row.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        raise ValueError(f"speech atom must have non-empty tool_calls list: {row!r}")
    first = tool_calls[0]
    if not isinstance(first, dict):
        raise ValueError(f"speech atom tool_calls[0] must be a dict: {row!r}")
    if first.get("type") != "function":
        raise ValueError(f"speech atom tool_calls[0].type must be 'function': {row!r}")
    fn = first.get("function") or {}
    if fn.get("name") != "say":
        raise ValueError(f"speech atom tool_calls[0].function.name must be 'say': {row!r}")
    args = fn.get("arguments") or {}
    if not isinstance(args, dict) or "text" not in args or not isinstance(args["text"], str):
        raise ValueError(f"speech atom must carry 'text' string in arguments: {row!r}")


@dataclass
class LanguageColumnsWriter:
    """Rewrite ``data/chunk-*/file-*.parquet`` with the two language columns."""

    drop_existing_subtask_index: bool = True

    @property
    def dropped_columns(self) -> tuple[str, ...]:
        """Columns this writer removes from the shards it rewrites.

        ``executor.py`` mirrors these into ``meta/info.json`` so metadata never
        advertises a column the parquet no longer has.
        """
        if self.drop_existing_subtask_index:
            return LEGACY_ANNOTATION_COLUMNS
        return (LEGACY_TOOLS_COLUMN,)

    def write_all(
        self,
        records: Sequence[EpisodeRecord],
        staging_dir: Path,
        root: Path,
        *,
        scope_validated: bool = False,
        skip_episode_indices: Sequence[int] = (),
    ) -> list[Path]:
        if not scope_validated:
            self.validate_write_scope(records, root)
        skipped = set(skip_episode_indices)
        episodes_by_path: dict[Path, list[EpisodeRecord]] = defaultdict(list)
        for record in records:
            # Retain every preflighted shard for schema migration, even one
            # containing only rejected episodes. An empty selection copies
            # existing language rows (or empty columns on the first run),
            # keeping meta/info.json consistent across the complete dataset.
            episodes = episodes_by_path[record.data_path]
            if record.episode_index not in skipped:
                episodes.append(record)

        written: list[Path] = []
        for path, eps in episodes_by_path.items():
            self._rewrite_one(path, eps, staging_dir, root)
            written.append(path)
        return written

    def validate_write_scope(self, records: Sequence[EpisodeRecord], root: Path) -> None:
        """Require shards containing unselected episodes to be canonical.

        A first partial run cannot safely update ``meta/info.json`` while
        leaving other shards on the legacy schema, nor can it migrate a packed
        shard without also changing unselected episodes in that file. Require
        one full migration first. Later partial reruns are safe because every
        affected shard already carries the canonical columns.
        """
        selected_by_path: dict[Path, set[int]] = defaultdict(set)
        for record in records:
            selected_by_path[record.data_path.resolve()].add(record.episode_index)
        all_paths = {path.resolve() for path in (root / "data").rglob("*.parquet")}
        incompatible: list[Path] = []
        for path in sorted(all_paths):
            names = set(pq.read_schema(path).names)
            has_language_columns = {LANGUAGE_PERSISTENT, LANGUAGE_EVENTS}.issubset(names)
            has_dropped_columns = any(name in names for name in self.dropped_columns)
            has_canonical_schema = has_language_columns and not has_dropped_columns
            if has_canonical_schema:
                continue

            selected_episodes = selected_by_path.get(path, set())
            episode_indices = set(pq.read_table(path, columns=["episode_index"]).column(0).to_pylist())
            if selected_episodes != episode_indices:
                incompatible.append(path)
        if incompatible:
            preview = ", ".join(str(path) for path in incompatible[:3])
            if len(incompatible) > 3:
                preview += f", ... ({len(incompatible)} total)"
            raise ValueError(
                "A partial annotation run cannot safely migrate parquet shards that contain "
                "unselected episodes. "
                "Run once without --only_episodes to migrate the complete dataset, then partial reruns "
                f"are safe. Incompatible shard(s): {preview}"
            )

    def _rewrite_one(
        self,
        path: Path,
        episodes: Sequence[EpisodeRecord],
        staging_dir: Path,
        root: Path,
    ) -> None:
        with _exclusive_shard_lock(root, path):
            self._rewrite_one_locked(path, episodes, staging_dir)

    def _rewrite_one_locked(
        self,
        path: Path,
        episodes: Sequence[EpisodeRecord],
        staging_dir: Path,
    ) -> None:
        table = pq.read_table(path)
        n_rows = table.num_rows

        # Only selected episodes are materialized from staging. Existing
        # language rows for every other episode in a packed shard are copied
        # through unchanged, making ``--only_episodes`` non-destructive.
        staged_per_ep: dict[int, dict[str, list[dict[str, Any]]]] = {}
        for record in episodes:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            staged_per_ep[record.episode_index] = staging.read_all()

        persistent_by_ep: dict[int, list[dict[str, Any]]] = {}
        events_by_ep_ts: dict[int, dict[float, list[dict[str, Any]]]] = {}

        for ep_index, ep_staged in staged_per_ep.items():
            persistent_rows: list[dict[str, Any]] = []
            event_rows: list[dict[str, Any]] = []  # carry timestamp until bucketed
            for _module_name, rows in ep_staged.items():
                for row in rows:
                    style = row.get("style")
                    if column_for_style(style) == LANGUAGE_PERSISTENT:
                        persistent_rows.append(row)
                    else:
                        event_rows.append(row)

            persistent_rows.sort(key=_row_persistent_sort_key)
            normalized_persistent = []
            for r in persistent_rows:
                _validate_atom_invariants(r)
                _validate_speech_atom(r)
                normalized_persistent.append(_normalize_persistent_row(r))
            persistent_by_ep[ep_index] = normalized_persistent

            buckets: dict[float, list[dict[str, Any]]] = defaultdict(list)
            for r in event_rows:
                _validate_atom_invariants(r)
                _validate_speech_atom(r)
                ts = float(r["timestamp"])
                buckets[ts].append(_normalize_event_row(r))
            for ts in list(buckets.keys()):
                buckets[ts].sort(key=_row_event_sort_key)
            events_by_ep_ts[ep_index] = buckets

        episode_col = (
            table.column("episode_index").to_pylist() if "episode_index" in table.column_names else None
        )
        ts_col = table.column("timestamp").to_pylist() if "timestamp" in table.column_names else None
        if episode_col is None or ts_col is None:
            raise ValueError(f"{path} is missing 'episode_index' or 'timestamp' — required by the writer.")

        existing_persistent = (
            table.column(LANGUAGE_PERSISTENT).to_pylist()
            if LANGUAGE_PERSISTENT in table.column_names
            else [[] for _ in range(n_rows)]
        )
        existing_events = (
            table.column(LANGUAGE_EVENTS).to_pylist()
            if LANGUAGE_EVENTS in table.column_names
            else [[] for _ in range(n_rows)]
        )
        selected_episodes = set(staged_per_ep)
        per_row_persistent: list[list[dict[str, Any]]] = []
        per_row_events: list[list[dict[str, Any]]] = []
        for i in range(n_rows):
            ep = episode_col[i]
            if ep not in selected_episodes:
                per_row_persistent.append(existing_persistent[i] or [])
                per_row_events.append(existing_events[i] or [])
                continue
            ts = float(ts_col[i])
            per_row_persistent.append(persistent_by_ep.get(ep, []))
            buckets = events_by_ep_ts.get(ep, {})
            per_row_events.append(buckets.get(ts, []))

        new_table = self._materialize_table(
            table, per_row_persistent, per_row_events, drop_old=self.drop_existing_subtask_index
        )
        # Re-emit one row group per episode (a bulk pq.write_table would collapse
        # them into one). Write to a sibling tmp path and atomically rename so a
        # crash mid-write can't leave a half-written shard.
        fd, tmp_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            os.chmod(tmp_path, stat.S_IMODE(path.stat().st_mode))
            write_table_one_row_group_per_episode(new_table, tmp_path)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def _materialize_table(
        self,
        table: pa.Table,
        persistent: list[list[dict[str, Any]]],
        events: list[list[dict[str, Any]]],
        *,
        drop_old: bool,
    ) -> pa.Table:
        cols = []
        names = []
        for name in table.column_names:
            if drop_old and name == LEGACY_SUBTASK_INDEX_COLUMN:
                continue
            if name in (LANGUAGE_PERSISTENT, LANGUAGE_EVENTS):
                continue  # we'll re-add canonical versions
            # Strip any legacy ``tools`` column previously emitted by older
            # writers — the schema no longer uses it (constant lives in
            # SAY_TOOL_SCHEMA / DEFAULT_TOOLS).
            if name == LEGACY_TOOLS_COLUMN:
                continue
            cols.append(table.column(name))
            names.append(name)

        # We let pyarrow infer struct/list schema rather than passing the
        # canonical type from `lerobot.datasets.language` directly: that type
        # uses `pa.json_()` for the `tool_calls` element type, which
        # `pa.array(..., type=...)` cannot materialize from Python lists on
        # current pyarrow versions. The inferred schema round-trips through
        # parquet and `LeRobotDataset` correctly — upstream LeRobot's own
        # `tests/datasets/test_language.py` exercises the same flow.
        persistent_arr = pa.array(persistent)
        events_arr = pa.array(events)

        cols.extend([persistent_arr, events_arr])
        names.extend([LANGUAGE_PERSISTENT, LANGUAGE_EVENTS])

        return pa.Table.from_arrays(cols, names=names)


def speech_atom(timestamp: float, text: str) -> dict[str, Any]:
    """Build a canonical speech tool-call atom for the events column."""
    return {
        "role": "assistant",
        "content": None,
        "style": None,
        "timestamp": float(timestamp),
        "camera": None,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "say",
                    "arguments": {"text": text},
                },
            }
        ],
    }
