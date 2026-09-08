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
"""Import subtask spans a dataset already records, instead of asking the VLM.

Four sources, tried in this order under ``auto``:

1. ``sarm`` — ``{prefix}_subtask_names`` / ``_start_times`` / ``_end_times`` on
   ``meta/episodes/*.parquet``, written by
   ``lerobot.data_processing.sarm_annotations.subtask_annotation``. The
   unprefixed legacy columns are accepted too.
2. ``lerobot_annotations`` — spans from ``meta/lerobot_annotations.json`` in
   either interval or timestamped-atom form.
3. ``language`` — ``style="subtask"`` rows already sitting in the episode's
   ``language_persistent`` column, i.e. the output of a previous
   ``lerobot-align`` run.
4. ``task_index`` — runs of the per-frame ``task_index`` column, for datasets
   recorded with the task string changing mid-episode.

All four carry their own timings, so an import costs no VLM call at all: the
spans drop straight into the same ``_clean_spans`` -> ``_stitch_full_coverage``
tail that generated spans go through.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.datasets.language import LANGUAGE_PERSISTENT
from lerobot.datasets.utils import EPISODES_DIR

from .reader import EpisodeRecord, reconstruct_subtask_spans

logger = logging.getLogger(__name__)

OFF = "off"
AUTO = "auto"
SARM = "sarm"
LEROBOT_ANNOTATIONS = "lerobot_annotations"
LANGUAGE = "language"
TASK_INDEX = "task_index"

#: Resolution order used by ``auto``, best-grounded source first.
IMPORT_SOURCES = (SARM, LEROBOT_ANNOTATIONS, LANGUAGE, TASK_INDEX)

_SARM_SUFFIXES = ("subtask_names", "subtask_start_times", "subtask_end_times")
_LEROBOT_ANNOTATIONS_PATH = Path("meta") / "lerobot_annotations.json"


def _is_missing(value: Any) -> bool:
    """True for the None / NaN a parquet column uses for "not annotated"."""
    return value is None or (isinstance(value, float) and value != value)


def _sarm_columns(columns: Any, prefix: str) -> tuple[str, str, str] | None:
    """Pick the SARM column triple to read, preferring ``prefix``.

    Falls back to the other granularity and then to the unprefixed legacy
    columns, matching what ``subtask_annotation.load_annotations_from_dataset``
    accepts.
    """
    other = "sparse" if prefix == "dense" else "dense"
    for candidate in (f"{prefix}_", f"{other}_", ""):
        triple = tuple(f"{candidate}{suffix}" for suffix in _SARM_SUFFIXES)
        if all(name in columns for name in triple):
            return triple  # type: ignore[return-value]
    return None


def _load_episode_meta(root: Path | None) -> Any:
    """``meta/episodes`` as a frame indexed by ``episode_index``, or ``None``.

    Returns ``None`` (rather than raising) whenever the metadata simply isn't
    there — plenty of valid datasets have no ``meta/episodes`` at all, and the
    caller falls through to the next source.
    """
    if root is None:
        return None
    if not (root / EPISODES_DIR).exists():
        return None
    from lerobot.datasets.io_utils import load_episodes  # noqa: PLC0415

    try:
        episodes = load_episodes(root)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.warning("subtask import: could not read %s/%s: %s", root, EPISODES_DIR, exc)
        return None
    frame = episodes.to_pandas()
    if "episode_index" in frame.columns:
        frame = frame.set_index("episode_index")
    return frame


def _load_lerobot_annotations(root: Path | None) -> dict[str, Any] | None:
    """Load the episode mapping from meta/lerobot_annotations.json once.

    A missing file is an ordinary unavailable source. Invalid JSON or an
    incompatible top-level schema is warned about and ignored so auto can
    continue to the next source.
    """
    if root is None:
        return None
    path = root / _LEROBOT_ANNOTATIONS_PATH
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning("subtask import: could not read %s: %s", path, exc)
        return None
    if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), dict):
        logger.warning(
            "subtask import: %s must contain an object-valued 'episodes' field",
            path,
        )
        return None
    return payload["episodes"]


def _annotation_time(value: Any) -> float:
    """A finite JSON number, excluding booleans and numeric-looking strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("annotation time must be a JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("annotation time must be finite")
    return result


def _from_sarm(record: EpisodeRecord, episodes: Any, prefix: str) -> list[dict[str, Any]]:
    """Spans from the SARM subtask columns on ``meta/episodes``.

    Reads the raw ``*_start_times`` / ``*_end_times`` float columns directly
    rather than going through ``load_annotations_from_dataset``, which
    round-trips the times through an ``mm:ss`` string and so quantises every
    boundary to a whole second.
    """
    if episodes is None or record.episode_index not in episodes.index:
        return []
    triple = _sarm_columns(episodes.columns, prefix)
    if triple is None:
        return []
    col_names, col_start, col_end = triple
    row = episodes.loc[record.episode_index]
    names, starts, ends = row[col_names], row[col_start], row[col_end]
    if _is_missing(names) or _is_missing(starts) or _is_missing(ends):
        return []
    try:
        triples = list(zip(names, starts, ends, strict=True))
    except (TypeError, ValueError) as exc:
        logger.warning(
            "episode %d: malformed SARM subtask columns (%s): %s", record.episode_index, col_names, exc
        )
        return []
    spans: list[dict[str, Any]] = []
    for name, start, end in triples:
        text = str(name).strip()
        if not text:
            continue
        spans.append({"text": text, "start": float(start), "end": float(end)})
    return spans


def _warn_malformed_lerobot_annotations(record: EpisodeRecord, detail: str) -> None:
    logger.warning(
        "episode %d: malformed %s entry: %s",
        record.episode_index,
        _LEROBOT_ANNOTATIONS_PATH,
        detail,
    )


def _from_lerobot_annotations(
    record: EpisodeRecord, episodes: Mapping[str, Any] | None
) -> list[dict[str, Any]]:
    """Read one episode from either supported lerobot-annotations schema.

    Interval form stores a subtasks list of label/start/end objects. Atom form
    stores timestamped language rows; subtask ends are reconstructed from the
    next subtask start and the episode's final frame. If both fields exist,
    non-empty interval form wins because it is the explicit authored segmentation.

    Parsing is atomic per episode. A malformed subtask returns no spans instead
    of silently stretching a partial annotation over the omitted behavior.
    """
    if episodes is None:
        return []
    episode = episodes.get(str(record.episode_index))
    if episode is None:
        return []
    if not isinstance(episode, Mapping):
        _warn_malformed_lerobot_annotations(record, "episode must be an object")
        return []

    if "subtasks" in episode:
        raw_subtasks = episode["subtasks"]
        if not isinstance(raw_subtasks, list):
            _warn_malformed_lerobot_annotations(record, "'subtasks' must be a list")
            return []
        spans: list[dict[str, Any]] = []
        for index, raw in enumerate(raw_subtasks):
            if not isinstance(raw, Mapping):
                _warn_malformed_lerobot_annotations(record, f"subtask {index} must be an object")
                return []
            label = raw.get("label")
            if not isinstance(label, str) or not label.strip():
                _warn_malformed_lerobot_annotations(record, f"subtask {index} needs a non-empty string label")
                return []
            try:
                start = _annotation_time(raw["start"])
                end = _annotation_time(raw["end"])
            except (KeyError, TypeError, ValueError) as exc:
                _warn_malformed_lerobot_annotations(record, f"subtask {index} has invalid start/end ({exc})")
                return []
            if end <= start:
                _warn_malformed_lerobot_annotations(
                    record, f"subtask {index} must have end greater than start"
                )
                return []
            spans.append({"text": label.strip(), "start": start, "end": end})
        if spans or "atoms" not in episode:
            return spans

    if "atoms" not in episode:
        return []
    atoms = episode["atoms"]
    if not isinstance(atoms, list):
        _warn_malformed_lerobot_annotations(record, "'atoms' must be a list")
        return []

    subtask_rows: list[dict[str, Any]] = []
    for index, raw in enumerate(atoms):
        if not isinstance(raw, Mapping):
            _warn_malformed_lerobot_annotations(record, f"atom {index} must be an object")
            return []
        if raw.get("style") != "subtask":
            continue
        content = raw.get("content")
        if not isinstance(content, str) or not content.strip():
            _warn_malformed_lerobot_annotations(
                record, f"subtask atom {index} needs non-empty string content"
            )
            return []
        try:
            timestamp = _annotation_time(raw["timestamp"])
        except (KeyError, TypeError, ValueError) as exc:
            _warn_malformed_lerobot_annotations(record, f"subtask atom {index} has invalid timestamp ({exc})")
            return []
        subtask_rows.append({"style": "subtask", "content": content.strip(), "timestamp": timestamp})

    if not subtask_rows:
        return []
    episode_end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
    return reconstruct_subtask_spans(subtask_rows, episode_end_t=episode_end_t)


def _from_language_persistent(record: EpisodeRecord) -> list[dict[str, Any]]:
    """Spans from ``style="subtask"`` rows a previous annotate run wrote.

    The persistent slice is byte-identical on every frame of the episode, so
    the first row's list is the whole episode's state.
    """
    frame = record.frames_df()
    if LANGUAGE_PERSISTENT not in frame.columns or len(frame) == 0:
        return []
    raw = frame[LANGUAGE_PERSISTENT].iloc[0]
    if _is_missing(raw):
        return []
    rows = [dict(entry) for entry in raw if isinstance(entry, dict)]
    if not rows:
        return []
    episode_end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
    return reconstruct_subtask_spans(rows, episode_end_t=episode_end_t)


def _from_task_index(record: EpisodeRecord) -> list[dict[str, Any]]:
    """Spans from runs of the per-frame task string.

    Returns ``[]`` for the overwhelmingly common single-task episode: one span
    covering everything carries no subtask information beyond the episode task
    that every prompt already receives.
    """
    tasks = record.frame_tasks
    timestamps = record.frame_timestamps
    if not tasks or len(tasks) != len(timestamps):
        return []
    spans: list[dict[str, Any]] = []
    start_i = 0
    for i in range(1, len(tasks) + 1):
        if i < len(tasks) and tasks[i] == tasks[start_i]:
            continue
        text = (tasks[start_i] or "").strip()
        if text:
            end = float(timestamps[i]) if i < len(timestamps) else float(timestamps[-1])
            spans.append({"text": text, "start": float(timestamps[start_i]), "end": end})
        start_i = i
    return spans if len(spans) > 1 else []


@dataclass
class SubtaskImporter:
    """Resolve recorded subtask spans for an episode.

    ``source`` is either ``auto`` (try :data:`IMPORT_SOURCES` in order and take
    the first that yields spans) or one specific source name. The distinction
    matters to the caller, not here: an explicit source that comes up empty is
    an error worth surfacing, whereas ``auto`` coming up empty just means this
    dataset records nothing and the VLM should generate instead.
    """

    root: Path | None = None
    source: str = AUTO
    sarm_prefix: str = "dense"

    # Loaded once at construction: the episode phase runs on a thread pool, and
    # re-reading shared metadata per episode would dominate a VLM-free import.
    _episodes: Any = field(default=None, init=False, repr=False)
    _lerobot_annotation_episodes: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.source not in IMPORT_SOURCES and self.source != AUTO:
            raise ValueError(
                f"subtask_import={self.source!r} is not one of {(AUTO, *IMPORT_SOURCES)} (or 'off')"
            )
        if self.source in (AUTO, SARM):
            self._episodes = _load_episode_meta(self.root)
            if self.source == SARM and self._episodes is None:
                logger.warning(
                    "subtask_import='sarm' but no readable %s under %s — every episode will "
                    "import zero subtasks.",
                    EPISODES_DIR,
                    self.root,
                )
        if self.source in (AUTO, LEROBOT_ANNOTATIONS):
            self._lerobot_annotation_episodes = _load_lerobot_annotations(self.root)
            if self.source == LEROBOT_ANNOTATIONS and self._lerobot_annotation_episodes is None:
                logger.warning(
                    "subtask_import=%r but no readable %s under %s — every episode will "
                    "import zero subtasks.",
                    LEROBOT_ANNOTATIONS,
                    _LEROBOT_ANNOTATIONS_PATH,
                    self.root,
                )

    @property
    def tried(self) -> tuple[str, ...]:
        """The sources :meth:`spans_for` will consult, in order."""
        return IMPORT_SOURCES if self.source == AUTO else (self.source,)

    def spans_for(self, record: EpisodeRecord) -> tuple[list[dict[str, Any]], str | None]:
        """Return ``(spans, source_name)``; ``([], None)`` when nothing matched."""
        for name in self.tried:
            if name == SARM:
                spans = _from_sarm(record, self._episodes, self.sarm_prefix)
            elif name == LEROBOT_ANNOTATIONS:
                spans = _from_lerobot_annotations(record, self._lerobot_annotation_episodes)
            elif name == LANGUAGE:
                spans = _from_language_persistent(record)
            else:
                spans = _from_task_index(record)
            if spans:
                return spans, name
        return [], None
