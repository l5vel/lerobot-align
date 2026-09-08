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
"""``plan`` module: subtask decomposition + plan + memory (PERSISTENT styles)."""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.datasets.io_utils import load_info

from ..config import PlanConfig
from ..frames import (
    FrameProvider,
    encode_frames_to_clip,
    null_provider,
    to_contact_sheet_blocks,
    to_multiview_contact_sheet_blocks,
    to_stacked_view_frames,
    to_video_url_block,
)
from ..motion_sampling import select_motion_stratified_timestamps
from ..prompts import load as load_prompt
from ..reader import EpisodeRecord, reconstruct_subtask_spans, snap_to_frame
from ..staging import EpisodeStaging
from ..subtask_import import AUTO as AUTO_IMPORT, OFF as IMPORT_OFF, SubtaskImporter
from ..vlm_client import VlmClient

logger = logging.getLogger(__name__)


class AlignmentCoverageError(ValueError):
    """A fixed-label alignment placed fewer labels than required."""


def _has_video_url_block(blocks: Sequence[Any] | None) -> bool:
    """Whether an OpenAI content-block list contains a usable video URL."""
    return bool(blocks) and any(
        isinstance(block, dict)
        and block.get("type") == "video_url"
        and isinstance(block.get("video_url"), dict)
        and bool(block["video_url"].get("url"))
        for block in blocks or ()
    )


# Prepended to every describe / segment prompt so the VLM knows the images are
# timestamped contact-sheet grids, not a single video, and reads the burned-in
# per-tile timestamp when choosing boundaries.
def _contact_sheet_preamble(columns: int, camera_keys: Sequence[str] = ()) -> str:
    camera_note = ""
    if len(camera_keys) > 1:
        view_map = ", ".join(
            f"VIEW {index} = {key}" for index, key in enumerate(camera_keys, start=1)
        )
        camera_note = (
            "- Every timestamp tile vertically stacks synchronized camera views "
            f"of the SAME moment ({view_map}). Time advances once per complete "
            "stacked tile. Use whichever view makes each boundary clearest.\n"
        )
    return (
        "CONTACT SHEETS — how to read the images below:\n"
        f"- Each image is a grid of sampled video frames, {columns} per row, "
        "with time running left-to-right then top-to-bottom (row-major).\n"
        "- Each frame has its timestamp burned into the top-left corner, e.g. "
        '"012.50s". Use that printed timestamp (not the tile position) when you '
        "choose start/end times; boundaries should land on or near a printed "
        "timestamp.\n"
        "- Frames continue across grids: an action may span the end of one sheet "
        "and the start of the next, so do not place a boundary just because a new "
        "image begins.\n"
        f"{camera_note}\n"
    )


# Appended to every describe (and segment) prompt. A visual, causal definition
# of where one event ends and the next begins — adapted from macrodata/refiner —
# to sharpen cut points while the existing prompt keeps owning the imperative
# phrasing.
_CAUSAL_BOUNDARY_RULES = (
    "EVENT BOUNDARIES — where one event ends and the next begins:\n"
    "- Start a new event whenever the world state changes: an object becomes "
    "held (the gripper closes on it), an object is released (the gripper opens "
    "and it stays put), an object reaches a new location, a lid/door/drawer "
    "changes open/closed state, a tool starts or stops affecting a surface, or "
    "contents visibly move (e.g. poured).\n"
    "- If a single action changes the same state gradually and continuously, "
    "keep it as ONE event — do not split it.\n"
    "- If the same action repeats on different objects or target locations, "
    "treat each repetition as a separate event.\n"
    "- Do NOT create boundaries for idle time, camera motion, hesitation, or "
    "tiny hand adjustments."
)


@dataclass
class _GivenSubtasks:
    """Subtask labels supplied by the user, keyed by episode."""

    default: list[str]
    by_episode: dict[int, list[str]]

    def for_episode(self, episode_index: int) -> list[str]:
        return self.by_episode.get(episode_index, self.default)


@dataclass(frozen=True)
class _GenerationClip:
    """Temporary native-video clip prepared for one generation VLM call."""

    path: Path
    duration: float
    fps: float


def _coerce_label_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: expected a non-empty list; got {type(value).__name__}")
    labels = [item.strip() for item in value if isinstance(item, str)]
    if len(labels) != len(value) or not all(labels):
        raise ValueError(f"{where}: every subtask must be a non-empty string")
    return labels


def _load_subtasks_file(path: Path) -> _GivenSubtasks:
    """Read ``PlanConfig.subtasks_path`` into per-episode label lists.

    Accepts a flat list (applied to every episode) or an object keyed by
    episode index with an optional ``"default"`` entry. Raises on a bad file
    rather than falling back to VLM generation: a typo'd path that silently
    reverted to generated labels is exactly the failure this mode exists to
    avoid.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _GivenSubtasks(default=_coerce_label_list(payload, str(path)), by_episode={})
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON list or object, got {type(payload).__name__}")
    default: list[str] = []
    by_episode: dict[int, list[str]] = {}
    for key, value in payload.items():
        if key == "default":
            default = _coerce_label_list(value, f"{path}['default']")
            continue
        try:
            episode_index = int(key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{path}: key {key!r} is neither 'default' nor an episode index"
            ) from exc
        by_episode[episode_index] = _coerce_label_list(value, f"{path}[{key!r}]")
    if not default and not by_episode:
        raise ValueError(f"{path}: no subtasks found")
    return _GivenSubtasks(default=default, by_episode=by_episode)


@dataclass(frozen=True)
class _AlignCalibration:
    """Per-boundary offsets, and optionally the priors for the DP solver.

    ``mode`` selects whether ``offsets`` are seconds (``absolute``) or fractions
    of the episode's duration (``fraction_of_duration``). Fractions travel
    better between episodes of different length, which is most of a dataset.

    Supplying ``segment_fractions`` — the expected length of each subtask as a
    fraction of the episode — switches on the dynamic-programming solver, which
    chooses the whole ordered boundary set at once instead of shifting each
    boundary independently. That both uses the duration prior and makes
    out-of-order output impossible, rather than repairing it afterwards.
    """

    offsets: tuple[float, ...]
    labels: tuple[str, ...] = ()
    mode: str = "absolute"
    segment_fractions: tuple[float, ...] = ()
    residual_scale: float = 0.05
    duration_scale: float = 0.05
    duration_weight: float = 1.0
    # ``None`` preserves the legacy hand-written offset-file behavior. New
    # fitted files declare their applicability instead of clearing ``labels``
    # to silently bypass the exact-label guard.
    label_scope: str | None = None
    n_segments: int | None = None

    def __post_init__(self) -> None:
        if self.label_scope not in (None, "exact", "segment_count"):
            raise ValueError("'label_scope' must be 'exact' or 'segment_count' when present")
        if self.n_segments is not None and (
            type(self.n_segments) is not int or self.n_segments < 2
        ):
            raise ValueError("'n_segments' must be an integer of at least 2")
        if self.label_scope == "segment_count":
            if self.n_segments is None or self.n_segments != len(self.offsets) + 1:
                raise ValueError("segment_count calibration requires n_segments == len(offsets) + 1")
            if self.labels:
                raise ValueError("segment_count calibration must have an empty 'labels' list")
        elif self.label_scope == "exact":
            if not self.labels or any(not label.strip() for label in self.labels):
                raise ValueError("exact calibration requires a non-empty 'labels' list")
            if len(self.labels) != len(self.offsets) + 1:
                raise ValueError("exact calibration requires one offset per internal label boundary")
            if self.n_segments is not None and self.n_segments != len(self.labels):
                raise ValueError("exact calibration requires n_segments == len(labels)")

    def shift_for(self, boundary: int, duration: float) -> float:
        """Seconds to subtract from the model's ``boundary``-th internal start."""
        if boundary < 0 or boundary >= len(self.offsets):
            return 0.0
        offset = self.offsets[boundary]
        return offset * duration if self.mode == "fraction_of_duration" else offset

    def applies_to(self, labels: Sequence[str]) -> bool:
        """Whether the labels satisfy the fitted file's applicability schema.

        Exact schemas preserve positional label identity. A count schema
        explicitly assumes corresponding boundary positions within the same
        task, even when episode labels use different wording. Only legacy
        files without a declared scope retain the unchecked empty-label form.
        """
        if self.label_scope == "segment_count":
            return len(labels) == self.n_segments
        return not self.labels or tuple(labels) == self.labels


def _load_align_calibration(path: Path) -> _AlignCalibration:
    """Read a calibration file written by ``fit_align_calibration.py``.

    Accepts a bare list of offsets or an object with ``offsets`` and an
    optional ``labels``. Raises rather than falling back to uncalibrated
    output: a typo'd path that silently produced uncorrected spans is the
    failure this mode exists to avoid.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_offsets = payload.get("offsets")
        raw_labels = (
            payload.get("labels", []) if "label_scope" in payload else payload.get("labels") or []
        )
    elif isinstance(payload, list):
        raw_offsets, raw_labels = payload, []
    else:
        raise ValueError(f"{path}: expected a JSON list or object, got {type(payload).__name__}")
    if not isinstance(raw_offsets, list) or not raw_offsets:
        raise ValueError(f"{path}: 'offsets' must be a non-empty list of seconds")
    try:
        offsets = tuple(float(value) for value in raw_offsets)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: every offset must be a number of seconds") from exc
    if any(not math.isfinite(value) for value in offsets):
        raise ValueError(f"{path}: offsets must be finite")
    if not isinstance(raw_labels, list) or any(not isinstance(item, str) for item in raw_labels):
        raise ValueError(f"{path}: 'labels' must be a list of strings when present")

    extra = payload if isinstance(payload, dict) else {}
    if "label_scope" in extra and extra["label_scope"] not in ("exact", "segment_count"):
        raise ValueError(f"{path}: 'label_scope' must be 'exact' or 'segment_count'")
    mode = str(extra.get("mode", "absolute"))
    if mode not in {"absolute", "fraction_of_duration"}:
        raise ValueError(
            f"{path}: 'mode' must be 'absolute' or 'fraction_of_duration', got {mode!r}"
        )

    raw_segments = extra.get("segment_fractions") or []
    if not isinstance(raw_segments, list):
        raise ValueError(f"{path}: 'segment_fractions' must be a list when present")
    try:
        segments = tuple(float(value) for value in raw_segments)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: every segment fraction must be a number") from exc
    if any(not math.isfinite(value) or value < 0 for value in segments):
        raise ValueError(f"{path}: segment fractions must be finite and non-negative")
    if segments and len(segments) != len(offsets) + 1:
        raise ValueError(
            f"{path}: expected {len(offsets) + 1} segment fraction(s) for {len(offsets)} "
            f"boundary offset(s), got {len(segments)}"
        )

    def _positive(name: str, default: float) -> float:
        value = float(extra.get(name, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{path}: '{name}' must be a positive number")
        return value

    weight = float(extra.get("duration_weight", 1.0))
    if not math.isfinite(weight) or weight < 0:
        raise ValueError(f"{path}: 'duration_weight' must be a non-negative number")

    return _AlignCalibration(
        offsets=offsets,
        labels=tuple(raw_labels),
        mode=mode,
        segment_fractions=segments,
        residual_scale=_positive("residual_scale", 0.05),
        duration_scale=_positive("duration_scale", 0.05),
        duration_weight=weight,
        label_scope=extra.get("label_scope"),
        n_segments=extra.get("n_segments"),
    )


def _align_token_budget(n_subtasks: int) -> int:
    """Reply budget for one alignment call.

    The reply is one ``{"index", "start", "end"}`` object per supplied subtask
    (~25 tokens each), so the shared ``VlmConfig.max_new_tokens`` default of
    512 would truncate a long list mid-array — damage the JSON repair pass
    cannot undo.
    """
    return max(512, 64 * n_subtasks + 256)


def _coerce_timestamp(raw: Any) -> float | None:
    """Seconds from a reply's timestamp field, or ``None`` if unreadable.

    Boundaries are read off the tiles' burned-in badges, which
    ``_draw_timestamp_badge`` renders as ``f"{t:06.2f}s"`` — so a model quoting
    what it saw answers with the string ``"050.04s"``, not a number. A bare
    ``float()`` rejects every one of those, which silently discards the whole
    alignment.

    Also accepts a clock-style ``"MM:SS.ss"`` (or ``"H:MM:SS.ss"``) string.
    The badge is never rendered that way, but a model answering with a
    timestamp format of its own choosing rather than the one it read has been
    observed to produce one — the same failure class the plain-seconds case
    above guards against.
    """
    if isinstance(raw, str):
        text = raw.strip().removesuffix("s")
        if ":" in text:
            parts = text.split(":")
            if len(parts) not in {2, 3} or any(not part.isdigit() for part in parts[:-1]):
                return None
            try:
                seconds = float(parts[-1])
                if not 0 <= seconds < 60 or (len(parts) == 3 and int(parts[1]) >= 60):
                    return None
                for i, part in enumerate(reversed(parts[:-1])):
                    seconds += float(part) * 60 ** (i + 1)
            except ValueError:
                return None
            return seconds
        raw = text
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _parse_align_spans(result: Any, n_subtasks: int) -> dict[int, tuple[float, float]]:
    """Map one alignment reply onto ``{subtask index: (start, end)}``.

    Accepts ``{"subtasks": [{"index": i, "start": s, "end": e}, ...]}`` plus the
    positional ``[i, s, e]`` and ``[s, e]`` forms. ``index`` falls back to the
    entry's position in the list, since a model that emits the subtasks in the
    order it was given often drops the redundant field.

    An entry whose index is out of range, or whose start/end is null or
    unreadable, is skipped: that subtask simply was not placed. The caller
    reports the omission and stitches over it, which is the honest outcome for
    a subtask the episode never performs.
    """
    out: dict[int, tuple[float, float]] = {}
    if not isinstance(result, dict):
        return out
    entries = result.get("subtasks")
    if entries is not None and not isinstance(entries, list):
        raise ValueError("alignment subtasks must be a list")
    if not isinstance(entries, list):
        return out
    for position, entry in enumerate(entries):
        if isinstance(entry, dict):
            raw_idx = entry.get("index")
            raw_idx = position if raw_idx is None else raw_idx
            raw_start, raw_end = entry.get("start"), entry.get("end")
        elif isinstance(entry, list | tuple) and len(entry) >= 3:
            raw_idx, raw_start, raw_end = entry[0], entry[1], entry[2]
        elif isinstance(entry, list | tuple) and len(entry) == 2:
            raw_idx, (raw_start, raw_end) = position, entry
        else:
            raise ValueError(f"alignment subtask {position} must be an object or timestamp tuple")
        try:
            if isinstance(raw_idx, bool) or (isinstance(raw_idx, float) and not raw_idx.is_integer()):
                raise ValueError('non-integral index')
            idx = int(raw_idx)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'alignment subtask {position} needs an integer index') from exc
        # First entry wins: a duplicated index means the model contradicted
        # itself, and the earlier answer is the one the rest of the reply was
        # written against.
        if not 0 <= idx < n_subtasks or idx in out:
            continue
        start, end = _coerce_timestamp(raw_start), _coerce_timestamp(raw_end)
        if any(raw is not None and (isinstance(raw, bool) or value is None
                                   or not math.isfinite(value))
               for raw, value in ((raw_start, start), (raw_end, end))):
            raise ValueError(f"alignment subtask {position} needs finite start/end timestamps")
        if start is None or end is None:
            continue
        out[idx] = (start, end) if end >= start else (end, start)
    return out


@dataclass
class PlanSubtasksMemoryModule:
    """Generate subtask spans, plan, and memory rows.

    All output is persistent (lives in ``language_persistent``):

    - ``subtask`` rows: one per span, stamped at the span's *start* timestamp
      (snapped to an exact frame).
    - ``plan`` rows: emitted at every subtask boundary as the deterministic
      numbered list of remaining subtasks. Generated interjections share
      those boundaries; :meth:`run_plan_updates` only fills a missing plan
      for an external/non-boundary interjection.
    - ``memory`` rows: emitted at each subtask boundary (= subtask start
      timestamp from the second subtask onward).

    Subtask spans have three possible origins, and everything downstream —
    plan, memory, interjection anchoring — is identical whichever one is used:

    - **generate** (default): the VLM writes the labels and provisional
      boundaries. With ``subtask_realign_generated``, the labels then go
      through the fixed-label aligner and only its boundaries are kept.
    - **align** (``PlanConfig.subtasks_path``): labels are supplied, the VLM
      only times them — see :meth:`_align_given_subtasks`.
    - **import** (``PlanConfig.subtask_import``): both labels and boundaries
      are read from what the dataset already records, with no VLM call — see
      :mod:`..subtask_import`.
    """

    vlm: VlmClient
    config: PlanConfig
    frame_provider: FrameProvider = field(default_factory=null_provider)
    # Dataset root, needed to read SARM ``meta/episodes``,
    # ``meta/lerobot_annotations.json``, and ``meta/info.json`` for motion
    # feature discovery; every other path works off ``EpisodeRecord`` alone.
    root: Path | None = None

    # Both resolved once at construction (not lazily per episode) so a bad path
    # or missing metadata fails before any GPU time is spent, and so the
    # episode thread pool shares one immutable copy.
    _given: _GivenSubtasks | None = field(default=None, init=False, repr=False)
    _importer: SubtaskImporter | None = field(default=None, init=False, repr=False)
    _motion_feature_keys: tuple[str, ...] = field(default=(), init=False, repr=False)
    _calibration: _AlignCalibration | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.subtask_align_sampling not in {"uniform", "motion_stratified"}:
            raise ValueError(
                "subtask_align_sampling must be 'uniform' or 'motion_stratified', "
                f"got {self.config.subtask_align_sampling!r}"
            )
        importing = self.config.subtask_import != IMPORT_OFF
        if self.config.subtasks_path is not None and importing:
            raise ValueError(
                "subtasks_path (align) and subtask_import (import) are mutually exclusive: "
                "the first supplies labels for the VLM to time, the second reads labels AND "
                "times from the dataset. Pick one."
            )
        if self.config.subtasks_path is not None:
            self._given = _load_subtasks_file(Path(self.config.subtasks_path))
        elif importing:
            self._importer = SubtaskImporter(
                root=Path(self.root) if self.root is not None else None,
                source=self.config.subtask_import,
                sarm_prefix=self.config.subtask_import_sarm_prefix,
            )
        if self.config.subtask_align_calibration_path is not None:
            self._calibration = _load_align_calibration(
                Path(self.config.subtask_align_calibration_path)
            )
        if self.config.subtask_align_sampling == "motion_stratified":
            self._motion_feature_keys = self._resolve_motion_feature_keys()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def run_episode(self, record: EpisodeRecord, staging: EpisodeStaging) -> None:
        rows: list[dict[str, Any]] = []
        # Task driving every plan-module prompt: canonical episode_task, or a
        # video-derived one when it's empty/placeholder (see derive_task_*).
        effective_task = self._resolve_effective_task(record)
        # Establish valid visual grounding before any text-only augmentation.
        subtask_spans = self._subtask_spans(record, effective_task)
        # task_aug rows at t=0: phrasings the renderer rotates ${task} through.
        # Either the structured 5-axis taxonomy (task_aug_axes.enabled) or
        # free-form n_task_rephrasings; the effective task is always emitted
        # first so the rotation covers the source-of-truth phrasing.
        t0 = float(record.frame_timestamps[0]) if record.frame_timestamps else 0.0
        variants: list[str] | None = None
        if self.config.task_aug_axes.enabled and effective_task:
            variants = self._generate_task_aug_by_axes(effective_task, self.config.task_aug_axes)
        elif self.config.n_task_rephrasings > 0 and effective_task:
            variants = self._generate_task_rephrasings(
                effective_task, n=self.config.n_task_rephrasings
            )
        if variants is not None:
            rows.extend(self._task_aug_rows([effective_task, *variants], t0))

        # subtask rows
        for span in subtask_spans:
            rows.append(
                {
                    "role": "assistant",
                    "content": span["text"],
                    "style": "subtask",
                    "timestamp": snap_to_frame(span["start"], record.frame_timestamps),
                    "tool_calls": None,
                }
            )
        # Plan rows at every subtask boundary (incl. t=0). The plan is a
        # numbered list of still-todo subtasks, so re-emitting at each
        # boundary makes it shrink as work progresses — ${plan} at frame t is
        # exactly what's left to do.
        if self.config.emit_plan:
            for span in subtask_spans:
                boundary_t = snap_to_frame(span["start"], record.frame_timestamps)
                plan_text = self._generate_plan(record, subtask_spans, refresh_t=boundary_t)
                if plan_text is not None:
                    rows.append(
                        {
                            "role": "assistant",
                            "content": plan_text,
                            "style": "plan",
                            "timestamp": float(boundary_t),
                            "tool_calls": None,
                        }
                    )
        # memory rows at every subtask boundary except the very first start;
        # skipped entirely when ``emit_memory`` is False (subtasks-only / plan-only).
        prior_memory = ""
        memory_boundaries = enumerate(subtask_spans[1:], start=1) if self.config.emit_memory else []
        for i, span in memory_boundaries:
            completed = subtask_spans[i - 1]["text"]
            remaining = [s["text"] for s in subtask_spans[i:]]
            mem_text = self._generate_memory(
                record, prior_memory, completed, remaining, task=effective_task
            )
            if mem_text:
                ts = snap_to_frame(span["start"], record.frame_timestamps)
                rows.append(
                    {
                        "role": "assistant",
                        "content": mem_text,
                        "style": "memory",
                        "timestamp": ts,
                        "tool_calls": None,
                    }
                )
                prior_memory = mem_text
        staging.write("plan", rows)

    # ------------------------------------------------------------------
    # Task derivation + rephrasings
    # ------------------------------------------------------------------

    _PLACEHOLDER_TASKS: frozenset[str] = frozenset(
        {
            "debug",
            "test",
            "tbd",
            "todo",
            "n/a",
            "na",
            "untitled",
            "unnamed",
            "default",
            "placeholder",
        }
    )

    def _resolve_effective_task(self, record: EpisodeRecord) -> str:
        """Decide which task string drives the ``plan`` module for this episode.

        Returns the user-supplied ``record.episode_task`` unless
        ``derive_task_from_video`` says otherwise (see config docstring).
        Falls back gracefully to the canonical task if video derivation
        fails.
        """
        canonical = (record.episode_task or "").strip()
        mode = (self.config.derive_task_from_video or "off").strip().lower()
        if mode == "always":
            derived = self._derive_task_from_video(record)
            return derived or canonical
        if mode == "if_short" and self._task_seems_bad(canonical):
            derived = self._derive_task_from_video(record)
            if derived:
                return derived
        return canonical

    def _task_seems_bad(self, task: str) -> bool:
        if not task:
            return True
        if len(task.split()) < int(self.config.derive_task_min_words):
            return True
        return task.lower() in self._PLACEHOLDER_TASKS

    @staticmethod
    def _task_aug_rows(phrasings: Sequence[str], t0: float) -> list[dict[str, Any]]:
        """Build deduplicated ``task_aug`` rows (role=user) at ``t0``."""
        seen: set[str] = set()
        rows: list[dict[str, Any]] = []
        for phrasing in phrasings:
            key = phrasing.strip()
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "role": "user",
                    "content": key,
                    "style": "task_aug",
                    "timestamp": t0,
                    "tool_calls": None,
                }
            )
        return rows

    # ------------------------------------------------------------------
    # VLM call helpers — every plan-module prompt follows the same shape:
    # build messages → single VLM call → pull a named field.
    # ------------------------------------------------------------------

    def _vlm_field(self, messages: list[dict[str, Any]], field: str) -> Any:
        """Run a single VLM call and return ``result[field]`` or ``None``.

        Centralizes the ``vlm.generate_json([m])[0]`` + ``isinstance(dict)``
        dance every prompt-call site needs.
        """
        result = self.vlm.generate_json([messages])[0]
        if isinstance(result, dict):
            return result.get(field)
        return None

    @staticmethod
    def _text_message(text: str) -> list[dict[str, Any]]:
        """One-shot text-only user message wrapped for ``generate_json``."""
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    def _video_message(
        self,
        record: EpisodeRecord,
        prompt: str,
        window: tuple[float, float] | None = None,
        *,
        require_frames: bool = False,
    ) -> list[dict[str, Any]]:
        """User message combining the (optionally windowed) contact sheets with ``prompt``.

        The prompt is always prefixed with a short explanation of how to read
        the timestamped grids, so the model treats them as one ordered
        sequence of frames rather than unrelated images.
        """
        prompt = _contact_sheet_preamble(self.config.contact_sheet_columns) + prompt
        content = [
            *self._episode_video_block(
                record,
                window=window,
                fail_on_error=require_frames,
            ),
            {"type": "text", "text": prompt},
        ]
        return [{"role": "user", "content": content}]

    def _generation_visual_field(
        self,
        record: EpisodeRecord,
        *,
        field: str,
        contact_sheet_prompt: str,
        native_video_prompt: str,
        window: tuple[float, float] | None = None,
    ) -> tuple[Any, bool]:
        """Run one open-ended visual prompt in the configured representation.

        ``video`` is deliberately opt-in and reaches the model as a real
        ``video_url`` block. The internal ``type=video`` representation is not
        suitable here because the OpenAI-compatible client expands it into a
        sequence of independent image blocks.

        The returned boolean says whether replies use a clip-relative timeline.
        It matters for a whole episode whose first timestamp is not zero. If
        video encoding fails after frames were decoded, the contact-sheet
        fallback also uses relative badges; describe and segment therefore
        cannot disagree about the time origin if only one encode succeeds.
        Provider/decode errors and endpoint/model errors propagate clearly.
        """
        if self.config.subtask_generate_frame_format == "video":
            clip = self._encode_generation_clip(record, window=window)
            if clip is None and self.config.subtask_video_fallback == "error":
                raise RuntimeError(
                    f"episode {record.episode_index}: native-video {field} could not produce "
                    "a non-empty clip; contact-sheet fallback is disabled by "
                    "plan.subtask_video_fallback=error, so no alternate VLM request was sent"
                )
            if clip is None:
                logger.warning(
                    "episode %d: native-video %s is unavailable; using contact sheets",
                    record.episode_index,
                    field,
                )
            if clip is not None:
                try:
                    # Ask vLLM's processor for the clip's encoded rate so a
                    # non-default generation density (for example 3 fps) is
                    # not silently thinned back to the model's usual 2 fps.
                    # The OpenAI adapter forwards this for its default
                    # auto-served vLLM, or when the external-server opt-in env
                    # var documented in README is set.
                    blocks = None
                    try:
                        blocks = to_video_url_block(clip.path.as_uri(), fps=clip.fps)
                    except Exception as exc:
                        if self.config.subtask_video_fallback == "error":
                            raise RuntimeError(
                                f"episode {record.episode_index}: native-video {field} could not "
                                "construct its video_url block; contact-sheet fallback is disabled "
                                "by plan.subtask_video_fallback=error, so no alternate VLM request "
                                "was sent"
                            ) from exc
                        logger.warning(
                            "episode %d: native-video %s could not construct its video_url block; "
                            "using contact sheets: %s",
                            record.episode_index,
                            field,
                            exc,
                        )
                    if _has_video_url_block(blocks):
                        prompt = (
                            "NATIVE VIDEO — use the video's own frame timestamps. "
                            f"This clip runs from 0.00s to {clip.duration:.3f}s; "
                            "when the prompt asks for times, report them relative to the "
                            "start of this clip.\n\n"
                            f"{native_video_prompt}"
                        )
                        messages = [
                            {
                                "role": "user",
                                "content": [*blocks, {"type": "text", "text": prompt}],
                            }
                        ]
                        try:
                            return self._vlm_field(messages, field), True
                        except Exception as exc:
                            raise RuntimeError(
                                f"episode {record.episode_index}: native-video generation request "
                                f"failed: {exc}. Confirm the endpoint supports video_url input, "
                                "or use --plan.subtask_generate_frame_format=contact_sheet."
                            ) from exc
                    if self.config.subtask_video_fallback == "error":
                        raise RuntimeError(
                            f"episode {record.episode_index}: native-video {field} could not "
                            "produce a usable video_url block; contact-sheet fallback is disabled "
                            "by plan.subtask_video_fallback=error, so no alternate VLM request "
                            "was sent"
                        )
                    if blocks is not None:
                        logger.warning(
                            "episode %d: native-video %s produced no usable video_url block; "
                            "using contact sheets",
                            record.episode_index,
                            field,
                        )
                finally:
                    try:
                        clip.path.unlink(missing_ok=True)
                    except OSError:
                        # Cleanup must never hide a model/endpoint exception
                        # (or turn an otherwise successful request into one).
                        logger.warning(
                            "could not remove temporary generation clip %s",
                            clip.path,
                            exc_info=True,
                        )

        relative_timeline = self.config.subtask_generate_frame_format == "video"
        fallback_window = window
        if relative_timeline and fallback_window is None and record.frame_timestamps:
            fallback_window = (
                float(record.frame_timestamps[0]),
                float(record.frame_timestamps[-1]),
            )
        try:
            messages = self._video_message(
                record,
                contact_sheet_prompt,
                window=fallback_window,
                require_frames=True,
            )
        except Exception as exc:
            if relative_timeline:
                raise RuntimeError(
                    f"episode {record.episode_index}: native-video generation could not build "
                    f"its contact-sheet fallback: {type(exc).__name__}: {exc} No text-only VLM "
                    "request was sent. Fix the dataset/video/cache access before retrying; "
                    "contact-sheet mode also requires decodable frames."
                ) from exc
            raise
        if not any(
            block.get("type") in {"image", "image_url", "video", "video_url"}
            for block in messages[0]["content"]
        ):
            raise RuntimeError(
                f"episode {record.episode_index}: visual generation could not produce a "
                "clip or any contact-sheet images; refusing to send a text-only VLM request. "
                "Check video decoding and image conversion before retrying."
            )
        return self._vlm_field(messages, field), relative_timeline

    def _generation_sample_timestamps(
        self,
        record: EpisodeRecord,
        window: tuple[float, float] | None = None,
    ) -> list[float]:
        """Uniform timestamps shared by contact-sheet and native generation.

        Keeping this selector common makes an A/B change only the media
        representation and prompt wording on normal regularly timed LeRobot
        episodes. The contact-sheet default retains its legacy behavior on a
        sparse/irregular source: if the budget can show every real frame, it
        uses each frame's exact timestamp. Native clips must instead synthesize
        an even grid because one encoded FPS cannot represent irregular gaps.
        """
        if not record.frame_timestamps:
            return []
        if window is not None:
            w0, w1 = float(window[0]), float(window[1])
            duration = max(0.0, w1 - w0)
            n = max(1, int(round(duration * self.config.frames_per_second)) + 1)
            n = min(n, self.config.max_frames_per_prompt)
            if n <= 1 or duration <= 0.0:
                return [0.5 * (w0 + w1)]
            step = duration / (n - 1)
            return [w0 + i * step for i in range(n)]

        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        duration = t_last - t0
        n = max(1, int(round(duration * self.config.frames_per_second)) + 1)
        n = min(n, self.config.max_frames_per_prompt)
        if self.config.subtask_generate_frame_format == "contact_sheet":
            return self._uniform_episode_timestamps(record, n)
        if n <= 1 or duration <= 0.0:
            return [t0]
        step = duration / (n - 1)
        return [t0 + i * step for i in range(n)]

    def _encode_generation_clip(
        self,
        record: EpisodeRecord,
        *,
        window: tuple[float, float] | None = None,
    ) -> _GenerationClip | None:
        """Encode the generation samples as a lossless native-video clip.

        A native timeline needs at least two uniformly spaced timestamps.
        Single-frame/zero-duration inputs and clip-encoding failures use the
        contact-sheet path instead, but provider/decode failures propagate so
        native mode can never degrade into a text-only request. The caller
        owns and always deletes a successfully returned temporary path.
        """
        timestamps = self._generation_sample_timestamps(record, window=window)
        if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
            logger.warning(
                "episode %d: native-video subtask generation needs at least two timestamps; "
                "a native clip is unavailable for this prompt",
                record.episode_index,
            )
            return None

        try:
            frames = self.frame_provider.frames_at(record, timestamps, fail_on_error=True)
        except Exception as exc:
            raise RuntimeError(
                f"episode {record.episode_index}: native-video generation could not decode its "
                f"source frames: {type(exc).__name__}: {exc} Native video was explicitly "
                "selected, so no text-only VLM request was sent. Fix the dataset/cache access "
                "before retrying; contact-sheet mode also requires decodable frames."
            ) from exc
        if not frames or len(frames) != len(timestamps):
            raise RuntimeError(
                f"episode {record.episode_index}: native-video generation decoded "
                f"{len(frames)} frame(s) for {len(timestamps)} timestamp(s). Native video was "
                "explicitly selected, so no text-only VLM request was sent. Check the video "
                "shard and episode metadata before retrying; contact-sheet mode also requires "
                "a complete frame set."
            )

        path: Path | None = None
        try:
            handle, name = tempfile.mkstemp(prefix="lerobot-generate-", suffix=".mp4")
            path = Path(name)
            os.close(handle)
            fps = encode_frames_to_clip(
                frames,
                timestamps,
                path,
                frame_width=self.config.contact_sheet_frame_width,
            )
            if fps is None or not path.exists() or path.stat().st_size == 0:
                logger.warning(
                    "episode %d: native generation clip encoded empty",
                    record.episode_index,
                )
                path.unlink(missing_ok=True)
                return None
        except Exception as exc:
            logger.warning(
                "episode %d: could not encode the native generation clip: %s",
                record.episode_index,
                exc,
            )
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "could not remove failed generation clip %s", path, exc_info=True
                    )
            return None

        duration = float(timestamps[-1]) - float(timestamps[0])
        logger.info(
            "episode %d: prepared a %d-frame native generation clip at %.4f fps "
            "(%.2fs of playback)",
            record.episode_index,
            len(timestamps),
            fps,
            duration,
        )
        return _GenerationClip(
            path=path,
            duration=duration,
            fps=float(fps),
        )

    def _derive_task_from_video(self, record: EpisodeRecord) -> str | None:
        """Ask the VLM "what is this video about" with no task hint at all."""
        prompt = load_prompt("plan_video_task")
        text, _uses_clip_time = self._generation_visual_field(
            record,
            field="task",
            contact_sheet_prompt=prompt,
            native_video_prompt=prompt,
        )
        return text.strip() if isinstance(text, str) and text.strip() else None

    def _generate_task_rephrasings(self, base_task: str, *, n: int) -> list[str]:
        """Request ``n`` text-only paraphrases and retain up to ``n`` valid strings.

        Final deduplication, including against the canonical task, happens in
        :meth:`_task_aug_rows`. The requested count is a diversity target.
        """
        if n <= 0 or not base_task:
            return []
        prompt = load_prompt("plan_task_rephrasings").format(base_task=base_task, n=n)
        raw = self._vlm_field(self._text_message(prompt), "rephrasings")
        if not isinstance(raw, list):
            return []
        out = [item.strip().strip('"').strip("'") for item in raw if isinstance(item, str)]
        return [s for s in out if s][:n]

    # ------------------------------------------------------------------
    # Structured 5-axis task augmentation (EgoMimic-style taxonomy)
    # ------------------------------------------------------------------

    def _generate_task_aug_by_axes(self, base_task: str, axes_cfg: Any) -> list[str]:
        """One VLM call → variants along the 5-axis taxonomy.

        Variants from all axes are flattened into a single list (the
        downstream pipeline doesn't need to know about the per-axis
        bucketing — every variant becomes a ``task_aug`` row). Order
        is preserved for reproducibility: synonym_paraphrase first,
        then omit_arm, then omit_orientation, then omit_grasp_method,
        then combined_omissions.
        """
        if not base_task:
            return []
        prompt = load_prompt("plan_task_aug_axes").format(
            base_task=base_task,
            n_synonym=axes_cfg.synonym_paraphrase,
            n_omit_arm=axes_cfg.omit_arm,
            n_omit_orientation=axes_cfg.omit_orientation,
            n_omit_grasp_method=axes_cfg.omit_grasp_method,
            n_combined=axes_cfg.combined_omissions,
        )
        result = self.vlm.generate_json([self._text_message(prompt)])[0]
        if not isinstance(result, dict):
            raise RuntimeError("Structured task augmentation returned no valid JSON object.")
        axis_targets = (
            ("synonym_paraphrase", axes_cfg.synonym_paraphrase),
            ("omit_arm", axes_cfg.omit_arm),
            ("omit_orientation", axes_cfg.omit_orientation),
            ("omit_grasp_method", axes_cfg.omit_grasp_method),
            ("combined_omissions", axes_cfg.combined_omissions),
        )
        flat: list[str] = []
        seen: set[str] = {base_task.strip()}
        for axis, target in axis_targets:
            entries = result.get(axis)
            if not isinstance(entries, list):
                raise RuntimeError(
                    f"Structured task augmentation field {axis!r} must be a JSON list."
                )
            cleaned_axis: list[str] = []
            for item in entries:
                if not isinstance(item, str):
                    raise RuntimeError(
                        f"Structured task augmentation field {axis!r} contains a non-string entry."
                    )
                key = item.strip().strip('"').strip("'")
                if not key:
                    raise RuntimeError(
                        f"Structured task augmentation field {axis!r} contains an empty entry."
                    )
                if key in seen:
                    raise RuntimeError(
                        "Structured task augmentation returned duplicate variants across the "
                        f"canonical task/axes: {key!r}."
                    )
                seen.add(key)
                cleaned_axis.append(key)
            if len(cleaned_axis) > target:
                raise RuntimeError(
                    f"Structured task augmentation field {axis!r} returned "
                    f"{len(cleaned_axis)} entries, above configured target {target}."
                )
            if axis == "synonym_paraphrase" and len(cleaned_axis) != target:
                raise RuntimeError(
                    "Structured task augmentation returned "
                    f"{len(cleaned_axis)} synonym paraphrase(s), expected exactly {target}."
                )
            flat.extend(cleaned_axis)
        return flat

    def _episode_video_block(
        self,
        record: EpisodeRecord,
        window: tuple[float, float] | None = None,
        *,
        fail_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        """Timestamped contact sheets for the describe / segmentation prompts.

        Always renders the (optionally windowed) episode as contact sheets:
        frames sampled at ``frames_per_second`` and packed into timestamped
        JPEG grids. ``max_frames_per_prompt`` caps the frame count; whole
        episodes that exceed it are windowed upstream in
        :meth:`_generate_subtasks` so each call stays within budget while the
        full episode keeps its sampling density.

        When ``window=(w0, w1)`` is given the badges are WINDOW-RELATIVE
        (``ts - w0``) to match the window-relative time frame the
        segmentation prompt works in (spans are offset back to absolute time
        afterwards).
        """
        if not record.frame_timestamps:
            return []
        timestamps = self._generation_sample_timestamps(record, window=window)
        if fail_on_error:
            frames = self.frame_provider.frames_at(record, timestamps, fail_on_error=True)
        else:
            # Keep the established FrameProvider call shape for default
            # contact-sheet mode (including third-party/test providers that
            # predate the opt-in strict keyword).
            frames = self.frame_provider.frames_at(record, timestamps)
        if fail_on_error and len(frames) != len(timestamps):
            raise RuntimeError(f'episode {record.episode_index}: decoded {len(frames)} frames '
                               f'for {len(timestamps)} timestamps; refusing incomplete visual input')
        if window is not None:
            w0 = float(window[0])
            rel = [ts - w0 for ts in timestamps[: len(frames)]]
            return self._contact_sheet_blocks(frames, rel)
        return self._contact_sheet_blocks(frames, timestamps[: len(frames)])

    @staticmethod
    def _uniform_episode_timestamps(record: EpisodeRecord, n: int) -> list[float]:
        """``n`` episode-relative timestamps spanning ``[t0, t_last]`` uniformly."""
        ts = record.frame_timestamps
        if n >= len(ts):
            return [float(t) for t in ts]
        t0, t_last = float(ts[0]), float(ts[-1])
        if t_last <= t0 or n <= 1:
            return [t0] * max(1, n)
        step = (t_last - t0) / (n - 1)
        return [t0 + i * step for i in range(n)]

    def _contact_sheet_blocks(
        self, frames: list[Any], timestamps: list[float]
    ) -> list[dict[str, Any]]:
        """Build timestamped contact-sheet image blocks from decoded frames."""
        return to_contact_sheet_blocks(
            frames,
            timestamps,
            columns=self.config.contact_sheet_columns,
            frames_per_sheet=self.config.contact_sheet_frames_per_sheet,
            frame_width=self.config.contact_sheet_frame_width,
            quality=self.config.contact_sheet_quality,
        )

    def run_plan_updates(
        self,
        record: EpisodeRecord,
        staging: EpisodeStaging,
        interjection_times: Sequence[float],
    ) -> None:
        """Ensure a deterministic plan is present at each interjection time.

        Interjections are intentionally emitted at subtask boundaries, which
        already carry the correct remaining-subtasks plan. Their prompts may
        only cue that fixed upcoming trajectory, so no semantic replanning is
        needed. The fallback below covers external/non-boundary interjections
        without introducing a VLM plan that could diverge from the video.
        """
        if not self.config.emit_plan:
            return
        existing = staging.read("plan")
        # Pass the last frame timestamp so the final span is closed (else its
        # end == start, zero duration, and a refresh inside it is missed).
        episode_end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
        spans = reconstruct_subtask_spans(existing, episode_end_t=episode_end_t)
        already_planned: set[float] = {
            float(row["timestamp"])
            for row in existing
            if row.get("style") == "plan" and row.get("timestamp") is not None
        }
        new_rows = list(existing)

        for raw_t in interjection_times:
            t = snap_to_frame(raw_t, record.frame_timestamps)
            if t in already_planned:
                continue
            already_planned.add(t)
            plan_text = self._generate_plan(record, spans, refresh_t=t)
            if plan_text is not None:
                new_rows.append(
                    {
                        "role": "assistant",
                        "content": plan_text,
                        "style": "plan",
                        "timestamp": t,
                        "tool_calls": None,
                    }
                )
        staging.write("plan", new_rows)

    def _generate_subtasks(
        self, record: EpisodeRecord, *, task: str | None = None
    ) -> list[dict[str, Any]]:
        """Generate subtask spans, optionally via a multi-call quality chain.

        Single call (default): watch video → emit subtask JSON.

        Multi-call (opt-in, higher quality, more VLM calls):
          1. ``subtask_describe_first`` — a grounding pass that narrates
             ONLY what is visible (no JSON commitment to subtasks yet);
             its description is injected into the segmentation prompt so
             the model segments its own grounded observations instead of
             pattern-matching the task text.
          2. segmentation — emit subtask JSON (as before).
        """
        if record.row_count == 0 or not record.frame_timestamps:
            return []
        episode_duration = record.frame_timestamps[-1] - record.frame_timestamps[0]
        effective_task = task if task is not None else record.episode_task

        # ---- Auto-windowing (keeps the full sampling density) --------
        # Contact sheets are cheap, but a whole long episode sampled at
        # ``frames_per_second`` can still exceed ``max_frames_per_prompt``.
        # When it does, split into consecutive frame-budgeted windows (one
        # describe→segment call each, still at the full sampling density), then
        # merge + stitch — so an episode of any length is covered at full
        # density rather than subsampled into one sparse call. The final pair is
        # balanced instead of leaving a tiny greedy tail.
        fps = max(1e-6, float(self.config.frames_per_second))
        n_whole = int(round(episode_duration * fps)) + 1
        if n_whole > self.config.max_frames_per_prompt:
            return self._generate_subtasks_windowed(record, effective_task)

        # ---- Pass 1 (optional): grounding description ----------------
        observation_block = ""
        if getattr(self.config, "subtask_describe_first", False):
            description = self._describe_episode(record, effective_task)
            if description:
                observation_block = (
                    "You watched this video and described, chronologically, "
                    "ONLY what the robot actually does:\n"
                    f'"""{description}"""\n\n'
                    "Segment THAT grounded description (cross-checked against "
                    "the video) into atomic subtasks. Do not introduce any "
                    "action that is not in your description above.\n\n"
                )

        # ---- Pass 2: segmentation ------------------------------------
        prompt_args = {
            "episode_task": effective_task,
            "min_subtask_seconds": self.config.min_subtask_seconds,
            "max_steps": self.config.plan_max_steps,
            "episode_duration": f"{episode_duration:.3f}",
            "observation_block": observation_block,
        }
        contact_sheet_prompt = self._with_causal_rules(
            load_prompt("plan_subtasks").format(**prompt_args)
        )
        native_video_prompt = self._with_causal_rules(
            load_prompt("plan_subtasks_video").format(**prompt_args)
        )
        spans, uses_clip_time = self._generation_visual_field(
            record,
            field="subtasks",
            contact_sheet_prompt=contact_sheet_prompt,
            native_video_prompt=native_video_prompt,
        )
        if uses_clip_time:
            # Native video and its contact-sheet fallback both start at zero
            # even when the dataset's episode clock does not. Validate in clip
            # time first, then restore absolute time before frame snapping.
            cleaned = self._clean_spans(
                spans,
                record,
                bounds=(0.0, episode_duration),
                dedupe=False,
            )
            episode_start = float(record.frame_timestamps[0])
            for span in cleaned:
                span["start"] = episode_start + float(span["start"])
                span["end"] = episode_start + float(span["end"])
            cleaned = self._clean_spans(cleaned, record)
        else:
            cleaned = self._clean_spans(spans, record)
        if not cleaned:
            return []

        # ---- Full-episode coverage stitch ----------------------------
        # The VLM can start after t0 or leave gaps, so frames fall through
        # with no active subtask. Always stitch into a contiguous
        # [t0, t_last] cover.
        cleaned = self._stitch_full_coverage(cleaned, record)

        return cleaned

    def _seeded_relabel(
        self, record: EpisodeRecord, spans: list[dict[str, Any]], task: str
    ) -> list[dict[str, Any]]:
        """Re-label each span using prev/current/next segment contact sheets.

        Boundaries are kept fixed; only ``text`` is refined. The original
        ("seed") label is passed as a strong prior so the model verifies and
        minimally corrects it rather than re-describing from scratch — the
        macrodata seeded-relabeling step. One VLM call per span.
        """
        n = len(spans)
        out: list[dict[str, Any]] = []
        for i, span in enumerate(spans):
            content: list[dict[str, Any]] = []
            if i > 0:
                content += self._segment_sheet(record, spans[i - 1])
            content += self._segment_sheet(record, span)
            if i < n - 1:
                content += self._segment_sheet(record, spans[i + 1])
            prompt = load_prompt("plan_subtask_relabel").format(
                episode_task=task,
                seed_label=span["text"],
                segment_index=i + 1,
                segment_count=n,
                start=float(span["start"]),
                end=float(span["end"]),
            )
            content.append({"type": "text", "text": prompt})
            label = self._vlm_field([{"role": "user", "content": content}], "label")
            if not isinstance(label, str) or not label.strip():
                raise RuntimeError(
                    f"episode {record.episode_index}: seeded relabel returned no valid label "
                    f"for subtask {i + 1}/{n}; refusing to publish partially relabeled output"
                )
            text = label.strip()
            out.append({**span, "text": text})
        return out

    def _segment_sheet(self, record: EpisodeRecord, span: dict[str, Any]) -> list[dict[str, Any]]:
        """Contact-sheet block(s) for one span: up to N frames sampled uniformly."""
        s, e = float(span["start"]), float(span["end"])
        n = max(1, int(self.config.subtask_relabel_frames))
        if e <= s or n == 1:
            timestamps = [s]
        else:
            step = (e - s) / (n - 1)
            timestamps = [s + i * step for i in range(n)]
        frames = self.frame_provider.frames_at(record, timestamps)
        return self._contact_sheet_blocks(frames, timestamps[: len(frames)])

    def _generation_windows(
        self,
        record: EpisodeRecord,
    ) -> list[tuple[float, float]]:
        """Consecutive full-density windows without a pathological tail.

        Full-size prefix windows keep their established seams. When the final
        greedy window would be short, only the last pair is rebalanced in
        sample-interval units. The pair therefore keeps the same total frame
        work and shared endpoint while neither call receives a tiny tail clip.
        """
        if not record.frame_timestamps:
            return []
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        duration = t_last - t0
        if duration <= 0.0:
            return []

        fps = max(1e-6, float(self.config.frames_per_second))
        frame_budget = int(self.config.max_frames_per_prompt)
        if frame_budget <= 1:
            # Full-density endpoint pairs cannot fit in a one-frame prompt.
            # Divide the episode evenly so this supported debug configuration
            # still avoids a pathological final sliver and always terminates.
            window_count = max(1, math.ceil(duration * fps))
            return [
                (
                    t0 + duration * index / window_count,
                    t0 + duration * (index + 1) / window_count,
                )
                for index in range(window_count)
            ]

        capacity = frame_budget - 1
        total_intervals = max(1, int(round(duration * fps)))
        window_count = max(1, math.ceil(total_intervals / capacity))
        if window_count == 1:
            return [(t0, t_last)]

        windows = []
        start = t0
        prefix_count = max(0, window_count - 2)
        for _ in range(prefix_count):
            end = start + capacity / fps
            windows.append((start, end))
            start = end

        remaining_intervals = total_intervals - prefix_count * capacity
        left_intervals = remaining_intervals // 2
        remaining_duration = t_last - start
        middle = start + remaining_duration * left_intervals / remaining_intervals
        windows.extend(((start, middle), (middle, t_last)))
        return windows

    def _generate_subtasks_windowed(self, record: EpisodeRecord, task: str) -> list[dict[str, Any]]:
        """Subtask generation in bounded, tail-balanced windows at constant fps.

        Splits ``[t0, t_last]`` into consecutive windows that each honor the
        frame budget. Full prefix windows retain their old seams; the final pair
        is balanced to avoid a very short tail request. Each window runs the
        describe -> segment chain at ``frames_per_second``; spans are then offset
        to absolute time, merged, and stitched into a whole-episode cover.
        """
        all_spans: list[dict[str, Any]] = []
        windows = self._generation_windows(record)
        for w0, w1 in windows:
            all_spans.extend(self._subtasks_for_window(record, task, w0, w1))
        logger.info(
            "episode %d: windowed subtask gen over %d tail-balanced window(s), "
            "each capped at %d sampled frame(s) -> %d raw spans",
            record.episode_index,
            len(windows),
            self.config.max_frames_per_prompt,
            len(all_spans),
        )
        # Merge across windows: clamp to the absolute episode, sort, and
        # frame-snap to distinct starts (handles any boundary collisions).
        cleaned = self._clean_spans(all_spans, record)
        if not cleaned:
            return []
        return self._stitch_full_coverage(cleaned, record)

    def _subtasks_for_window(
        self, record: EpisodeRecord, task: str, w0: float, w1: float
    ) -> list[dict[str, Any]]:
        """Run describe -> segment on one ``[w0, w1]`` window.

        The model works in window-RELATIVE time ``[0, L]`` (it perceives
        the window as a clip starting at 0); spans are offset back to
        absolute ``[w0, w1]`` before returning.
        """
        window = (w0, w1)
        win_len = max(0.0, w1 - w0)

        observation_block = ""
        if getattr(self.config, "subtask_describe_first", False):
            description = self._describe_episode(record, task, window=window)
            if description:
                observation_block = (
                    "You watched this video clip and described, chronologically, "
                    "ONLY what the robot actually does:\n"
                    f'"""{description}"""\n\n'
                    "Segment THAT grounded description (cross-checked against "
                    "the clip) into atomic subtasks. Do not introduce any "
                    "action that is not in your description above.\n\n"
                )

        prompt_args = {
            "episode_task": task,
            "min_subtask_seconds": self.config.min_subtask_seconds,
            "max_steps": self.config.plan_max_steps,
            "episode_duration": f"{win_len:.3f}",
            "observation_block": observation_block,
        }
        contact_sheet_prompt = self._with_causal_rules(
            load_prompt("plan_subtasks").format(**prompt_args)
        )
        native_video_prompt = self._with_causal_rules(
            load_prompt("plan_subtasks_video").format(**prompt_args)
        )
        spans, _uses_clip_time = self._generation_visual_field(
            record,
            field="subtasks",
            contact_sheet_prompt=contact_sheet_prompt,
            native_video_prompt=native_video_prompt,
            window=window,
        )
        # Window-relative clamp; no frame-snap dedupe yet (done on the
        # merged absolute set).
        cleaned = self._clean_spans(spans, record, bounds=(0.0, win_len), dedupe=False)
        if not cleaned:
            return []

        # Offset window-relative spans back to absolute episode time.
        for s in cleaned:
            s["start"] = w0 + float(s["start"])
            s["end"] = w0 + float(s["end"])
        return cleaned

    # ------------------------------------------------------------------
    # Subtask origin: import -> align -> generate
    # ------------------------------------------------------------------

    def _subtask_spans(self, record: EpisodeRecord, task: str) -> list[dict[str, Any]]:
        """Subtask spans for ``record``, from whichever origin is configured."""
        given = self._given_subtasks(record)
        if given:
            # Labels are supplied: the VLM only decides when each one happens.
            return self._align_given_subtasks(record, given, task)

        if self._importer is not None:
            imported = self._import_subtask_spans(record)
            if imported is not None:
                return imported

        generated = self._generate_subtasks(record, task=task)
        if self.config.subtask_seeded_relabel and generated:
            generated = self._seeded_relabel(record, generated, task)
        if self.config.subtask_realign_generated and generated:
            generated = self._realign_generated_subtasks(record, generated, task)
        return generated

    def _realign_generated_subtasks(
        self, record: EpisodeRecord, generated: Sequence[dict[str, Any]], task: str
    ) -> list[dict[str, Any]]:
        """Refine generated boundaries only when fixed-label alignment is complete.

        Only ordered text labels cross the stage boundary. In particular, the
        alignment prompt never sees provisional times, so it cannot anchor
        itself to discovery's window seams or coarse boundary guesses.
        Repeated labels are deliberately retained: their list indices, not
        their text, identify distinct occurrences for the aligner.
        A failed refinement retains the entire generated cover; mixing a
        partial alignment with generated spans would lose label identity.
        """
        labels: list[str] = []
        for index, span in enumerate(generated):
            text = span.get("text") if isinstance(span, dict) else None
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(
                    f"episode {record.episode_index}: generated subtask {index + 1}/"
                    f"{len(generated)} has no valid label; refusing to start fixed-label alignment"
                )
            labels.append(text.strip())

        logger.info(
            "episode %d: realigning %d generated subtask label(s) as one whole-episode %s pass",
            record.episode_index,
            len(labels),
            self.config.subtask_align_frame_format,
        )
        try:
            aligned = self._align_given_subtasks(
                record,
                labels,
                task,
                require_labeled_calibration=True,
            )
        except AlignmentCoverageError as exc:
            # Only an incomplete refinement is recoverable here. Inference,
            # configuration and strict video-input failures still propagate.
            logger.warning("%s; keeping generated boundaries", exc)
            return [dict(span) for span in generated]
        if not aligned:
            logger.warning(
                "episode %d: generated-subtask realignment produced no spans for %d label(s); "
                "keeping generated boundaries",
                record.episode_index,
                len(labels),
            )
            return [dict(span) for span in generated]
        aligned_labels = [span.get("text") if isinstance(span, dict) else None for span in aligned]
        if aligned_labels != labels:
            logger.warning(
                "episode %d: generated-subtask realignment did not preserve the exact ordered "
                "generated label list (expected %d, got %d); keeping generated boundaries",
                record.episode_index,
                len(labels),
                len(aligned_labels),
            )
            return [dict(span) for span in generated]
        return aligned

    def _import_subtask_spans(self, record: EpisodeRecord) -> list[dict[str, Any]] | None:
        """Spans read from the dataset, or ``None`` to fall through to the VLM.

        Falling through is only allowed under ``subtask_import="auto"``, which
        means "use what's recorded if anything is". An explicitly named source
        that comes up empty returns ``[]`` instead: the user asked for that
        source, so silently spending a GPU on generated labels would be the
        wrong answer to give them.
        """
        assert self._importer is not None
        spans, source = self._importer.spans_for(record)
        if spans:
            logger.info(
                "episode %d: imported %d subtask(s) from %r (no VLM call)",
                record.episode_index,
                len(spans),
                source,
            )
            cleaned = self._clean_spans(spans, record)
            # Stitched like generated spans so the "every frame has exactly one
            # active subtask" contract holds for the renderer, even if the
            # recorded annotation deliberately left idle gaps.
            return self._stitch_full_coverage(cleaned, record) if cleaned else []
        if self._importer.source == AUTO_IMPORT:
            logger.warning(
                "episode %d: no recorded subtasks found (tried %s) — generating with the VLM instead",
                record.episode_index,
                ", ".join(self._importer.tried),
            )
            return None
        logger.warning(
            "episode %d: subtask_import=%r found nothing to import; this episode gets no subtasks "
            "(and so no plan or memory). Use subtask_import='auto' to fall back to generation.",
            record.episode_index,
            self._importer.source,
        )
        return []

    # ------------------------------------------------------------------
    # Fixed-label alignment (labels fixed, boundaries predicted)
    # ------------------------------------------------------------------

    def _given_subtasks(self, record: EpisodeRecord) -> list[str]:
        """Labels supplied for ``record``, or ``[]`` to generate them instead."""
        if self._given is None:
            return []
        labels = self._given.for_episode(record.episode_index)
        if not labels:
            raise ValueError(f'No supplied subtasks for episode {record.episode_index}; '
                             'add that episode or a default label list to subtasks_path')
        return labels

    def _align_given_subtasks(
        self,
        record: EpisodeRecord,
        subtasks: Sequence[str],
        task: str,
        *,
        require_labeled_calibration: bool = False,
    ) -> list[dict[str, Any]]:
        """Time a fixed, ordered list of subtask labels against the episode.

        ONE call: the whole (sub-sampled) episode goes up in the configured
        contact-sheet or native-video format, and the model returns a
        start/end for each supplied label.

        An earlier design asked instead for a per-tile classification — "which
        subtask is active in this tile?" for every sampled frame — and recovered
        spans by run-length encoding. That reply degenerates on real episodes: a
        VLM asked for hundreds of near-identical decisions in one autoregressive
        reply pattern-completes rather than looks. Measured against the human
        labels in ``meta/lerobot_annotations.json``, a 280-tile episode came back
        as four runs of exactly 100/50/30/100 with *synthesised* timestamps (the
        arithmetic sampling grid, not the burned-in badges), and a 94-tile
        episode came back as one constant index — 6-8% agreement. Splitting the
        tiles across calls only traded that for chunk-local indices restarting at
        0. The same model on the same sheets, asked for one span per label,
        places every label within a few seconds (~75% agreement). Hence: one
        call, one span per label.
        """
        if record.row_count == 0 or not record.frame_timestamps or not subtasks:
            return []
        camera_keys = self._align_camera_keys()
        # A stacked video clip carries every view inside one frame, so its cost
        # scales with pixels rather than with timeline length; only contact
        # sheets, which lay views out as separate tiles, pay per camera.
        budget_cameras = (
            1 if self.config.subtask_align_frame_format == "video" else len(camera_keys)
        )
        timestamps = self._align_sample_timestamps(record, camera_count=budget_cameras)
        decoded_views: list[tuple[str, list[Any]]] = []
        for camera_key in camera_keys:
            frames = self.frame_provider.frames_at(record, timestamps, camera_key=camera_key)
            if frames and len(frames) != len(timestamps):
                logger.warning(
                    "episode %d: alignment camera %r returned %d frame(s) for %d timestamp(s); "
                    "skipping the view to preserve frame/timestamp identity",
                    record.episode_index,
                    camera_key,
                    len(frames),
                    len(timestamps),
                )
                continue
            if not frames:
                logger.warning(
                    "episode %d: alignment camera %r produced no frames; skipping the view",
                    record.episode_index,
                    camera_key,
                )
                continue
            display_key = (
                camera_key or getattr(self.frame_provider, "camera_key", None) or "default"
            )
            decoded_views.append((str(display_key), frames))
        if not decoded_views:
            if (
                self.config.subtask_align_frame_format == "video"
                and self.config.subtask_video_fallback == "error"
            ):
                raise RuntimeError(
                    f"episode {record.episode_index}: native-video alignment could not decode "
                    "a complete selected camera view; contact-sheet fallback is disabled by "
                    "plan.subtask_video_fallback=error, so no VLM request was sent"
                )
            raise RuntimeError(
                f"episode {record.episode_index}: alignment requires a complete decoded camera view"
            )

        # A camera may exist in global metadata but be absent or undecodable in
        # one episode. Re-spend the total camera-frame budget on the surviving
        # views so a graceful fallback does not silently halve temporal density.
        if len(decoded_views) < len(camera_keys):
            retry_cameras = (
                1 if self.config.subtask_align_frame_format == "video" else len(decoded_views)
            )
            retry_timestamps = self._align_sample_timestamps(record, camera_count=retry_cameras)
            if retry_timestamps != timestamps:
                retried_views: list[tuple[str, list[Any]]] = []
                for camera_key, _frames in decoded_views:
                    frames = self.frame_provider.frames_at(
                        record,
                        retry_timestamps,
                        camera_key=camera_key,
                    )
                    if not frames or len(frames) != len(retry_timestamps):
                        break
                    retried_views.append((camera_key, frames))
                if len(retried_views) == len(decoded_views):
                    timestamps = retry_timestamps
                    decoded_views = retried_views
                else:
                    logger.warning(
                        "episode %d: could not re-budget alignment frames after a camera failure; "
                        "using the complete lower-density decode",
                        record.episode_index,
                    )

        clip_path = self._align_encode_clip(record, decoded_views, timestamps)
        if (
            clip_path is None
            and self.config.subtask_align_frame_format == "video"
            and self.config.subtask_video_fallback == "error"
        ):
            raise RuntimeError(
                f"episode {record.episode_index}: native-video alignment could not produce a "
                "non-empty clip; contact-sheet fallback is disabled by "
                "plan.subtask_video_fallback=error, so no alternate VLM request was sent"
            )
        if clip_path is None and self.config.subtask_align_frame_format == "video":
            logger.warning(
                "episode %d: native-video alignment is unavailable; using contact sheets",
                record.episode_index,
            )
        try:
            use_video = clip_path is not None
            if use_video:
                blocks = None
                try:
                    blocks = to_video_url_block(clip_path.as_uri(), fps=None)
                except Exception as exc:
                    if self.config.subtask_video_fallback == "error":
                        raise RuntimeError(
                            f"episode {record.episode_index}: native-video alignment could not "
                            "construct its video_url block; contact-sheet fallback is disabled "
                            "by plan.subtask_video_fallback=error, so no alternate VLM request "
                            "was sent"
                        ) from exc
                    logger.warning(
                        "episode %d: native-video alignment could not construct its video_url "
                        "block; using contact sheets: %s",
                        record.episode_index,
                        exc,
                    )
                if not _has_video_url_block(blocks):
                    if self.config.subtask_video_fallback == "error":
                        raise RuntimeError(
                            f"episode {record.episode_index}: native-video alignment could not "
                            "produce a usable video_url block; contact-sheet fallback is disabled "
                            "by plan.subtask_video_fallback=error, so no alternate VLM request "
                            "was sent"
                        )
                    if blocks is not None:
                        logger.warning(
                            "episode %d: native-video alignment produced no usable video_url "
                            "block; using contact sheets",
                            record.episode_index,
                        )
                    use_video = False
            if not use_video and self.config.subtask_align_frame_format == "video":
                decoded_views, timestamps = self._rebudget_align_contact_sheet(
                    record,
                    decoded_views,
                    timestamps,
                )
            if use_video:
                # The clip's own timeline starts at zero, so the model is asked
                # about clip time and the reply is shifted back to episode time
                # below. On these datasets the offset is zero, but deriving it
                # keeps the two timelines from silently diverging.
                offset = float(timestamps[0])
                prompt_template = "plan_subtask_align_video"
                if len(decoded_views) > 1:
                    view_map = ", ".join(
                        f"VIEW {i} = {key}" for i, (key, _f) in enumerate(decoded_views, start=1)
                    )
                    preamble_video = (
                        "Each video frame vertically stacks synchronized camera views of the "
                        f"SAME moment ({view_map}). Time advances once per stacked frame — the "
                        "views are simultaneous, not consecutive. Use whichever view makes each "
                        "boundary clearest.\n\n"
                    )
                else:
                    preamble_video = ""
                episode_start = 0.0
                episode_end = float(timestamps[-1]) - offset
                preamble = preamble_video
            else:
                if len(decoded_views) == 1:
                    blocks = self._contact_sheet_blocks(decoded_views[0][1], timestamps)
                else:
                    blocks = to_multiview_contact_sheet_blocks(
                        decoded_views,
                        timestamps,
                        columns=self.config.contact_sheet_columns,
                        frames_per_sheet=self.config.contact_sheet_frames_per_sheet,
                        frame_width=self.config.contact_sheet_frame_width,
                        quality=self.config.contact_sheet_quality,
                    )
                offset = 0.0
                prompt_template = "plan_subtask_align"
                episode_start = float(record.frame_timestamps[0])
                episode_end = float(record.frame_timestamps[-1])
                preamble = _contact_sheet_preamble(
                    self.config.contact_sheet_columns,
                    tuple(camera_key for camera_key, _frames in decoded_views),
                )
            if not blocks:
                return []

            prompt = preamble + load_prompt(prompt_template).format(
                episode_task=task,
                subtask_list="\n".join(f"{i}. {text}" for i, text in enumerate(subtasks)),
                episode_start=f"{episode_start:.2f}",
                episode_end=f"{episode_end:.2f}",
            )
            messages = [{"role": "user", "content": [*blocks, {"type": "text", "text": prompt}]}]
            result = self.vlm.generate_json(
                [messages], max_new_tokens=_align_token_budget(len(subtasks))
            )[0]
        finally:
            if clip_path is not None:
                try:
                    clip_path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "could not remove temporary alignment clip %s",
                        clip_path,
                        exc_info=True,
                    )

        parsed = _parse_align_spans(result, len(subtasks))
        if offset:
            parsed = {
                index: (start + offset, end + offset) for index, (start, end) in parsed.items()
            }
        parsed = self._apply_align_calibration(
            record,
            subtasks,
            parsed,
            require_labeled_calibration=require_labeled_calibration,
        )
        spans = self._align_spans_in_order(record, subtasks, parsed)
        self._check_align_coverage(record, len(spans), len(subtasks))
        cleaned = self._clean_spans(spans, record)
        if not cleaned:
            return []
        return self._stitch_full_coverage(cleaned, record)

    def _apply_align_calibration(
        self,
        record: EpisodeRecord,
        subtasks: Sequence[str],
        parsed: dict[int, tuple[float, float]],
        *,
        require_labeled_calibration: bool = False,
    ) -> dict[int, tuple[float, float]]:
        """Subtract the fitted per-boundary offsets from the model's answer.

        Offset ``i`` corrects the start of subtask ``i + 1`` — the first
        internal boundary — so the first subtask's start is left alone; it is
        forced to the episode start downstream anyway. Each corrected start
        also becomes the previous subtask's end, keeping the spans consecutive
        before :meth:`_align_spans_in_order` re-checks that invariant.

        A boundary with no fitted offset is left as the model reported it, so a
        short calibration corrects the boundaries it covers instead of failing.
        """
        def finish(
            result: dict[int, tuple[float, float]], reason: str, method: str | None = None
        ) -> dict[int, tuple[float, float]]:
            # One log call keeps the JSON event atomic with concurrent episodes.
            # The evaluation harness consumes actual runtime outcomes instead
            # of inferring application from which calibration path was routed.
            logger.info(
                "ALIGN_CALIBRATION %s",
                json.dumps(
                    {
                        "episode": int(record.episode_index),
                        "applied": reason == "applied",
                        "reason": reason,
                        "method": method,
                    },
                    sort_keys=True,
                ),
            )
            return result

        calibration = self._calibration
        if calibration is None:
            return finish(parsed, "not_configured")
        if not parsed:
            return finish(parsed, "empty_prediction")
        if require_labeled_calibration and (
            calibration.label_scope == "segment_count" or not calibration.labels
        ):
            logger.warning(
                "episode %d: alignment calibration has no recorded subtask list; "
                "generated-label realignment requires an exact schema match, leaving this "
                "episode uncalibrated",
                record.episode_index,
            )
            return finish(parsed, "generated_requires_exact_labels")
        if not calibration.applies_to(subtasks):
            if calibration.label_scope == "segment_count":
                logger.warning(
                    "episode %d: alignment calibration was fit for %d segment(s), got %d; "
                    "leaving this episode uncalibrated",
                    record.episode_index,
                    calibration.n_segments,
                    len(subtasks),
                )
                return finish(parsed, "segment_count_mismatch")
            logger.warning(
                "episode %d: alignment calibration was fit for a different subtask list "
                "(%d label(s) vs %d); leaving this episode uncalibrated",
                record.episode_index,
                len(calibration.labels),
                len(subtasks),
            )
            return finish(parsed, "label_mismatch")

        if not record.frame_timestamps:
            return finish(parsed, "invalid_duration")
        episode_start = float(record.frame_timestamps[0])
        episode_end = float(record.frame_timestamps[-1])
        duration = episode_end - episode_start
        if not math.isfinite(duration) or duration <= 0:
            return finish(parsed, "invalid_duration")

        solved = self._solve_align_boundaries(
            record, subtasks, parsed, calibration, episode_start, episode_end
        )
        if solved is not None:
            return finish(solved, "applied", "duration_prior")

        corrected = dict(parsed)
        for index in sorted(parsed):
            shift = calibration.shift_for(index - 1, duration)
            if not shift:
                continue
            start, end = corrected[index]
            # Raw VLM ends are discarded by full-coverage stitching. They must
            # not cap corrections: the seed diagnostic sees stitched ends.
            new_start = max(episode_start, min(start - shift, episode_end))
            corrected[index] = (new_start, max(new_start, end))
            previous = corrected.get(index - 1)
            if previous is not None:
                corrected[index - 1] = (previous[0], max(previous[0], new_start))
        return finish(corrected, "applied", "offsets")

    def _solve_align_boundaries(
        self,
        record: EpisodeRecord,
        subtasks: Sequence[str],
        parsed: dict[int, tuple[float, float]],
        calibration: _AlignCalibration,
        episode_start: float,
        episode_end: float,
    ) -> dict[int, tuple[float, float]] | None:
        """Choose every internal boundary at once by dynamic programming.

        Balances two costs on a 0.1s grid: staying near the model's
        offset-corrected answer, and matching the seed set's expected subtask
        lengths. ``duration_weight`` trades between them.

        Solving jointly is what makes the ordering constraint structural — the
        independent per-boundary shift below can push one boundary past its
        neighbour, which cost a dropped label on a real episode. Returns
        ``None`` when the priors do not apply, so the caller falls back.
        """
        segments = calibration.segment_fractions
        if not segments or len(segments) != len(subtasks):
            return None
        # The solver places every internal boundary, so it needs the model's
        # full ordered answer; a partly-unplaced reply falls back to shifting.
        if sorted(parsed) != list(range(len(subtasks))):
            return None

        duration = episode_end - episode_start
        n_boundaries = len(subtasks) - 1
        if n_boundaries <= 0:
            return None

        grid = np.arange(episode_start, episode_end + 1e-9, 0.1)
        if grid.size < 2:
            return None

        model = np.array(
            [
                min(
                    max(parsed[i + 1][0] - calibration.shift_for(i, duration), episode_start),
                    episode_end,
                )
                for i in range(n_boundaries)
            ]
        )
        want = np.array(segments) * duration
        fit_scale = max(1e-6, calibration.residual_scale * duration)
        seg_scale = max(1e-6, calibration.duration_scale * duration)
        weight = calibration.duration_weight

        fit = np.abs(grid[None, :] - model[:, None]) / fit_scale
        # Cost of the run-up to the first boundary, measured from the episode start.
        cost = fit[0] + weight * np.abs((grid - episode_start) - want[0]) / seg_scale
        back = np.zeros((n_boundaries, grid.size), dtype=np.int32)
        for i in range(1, n_boundaries):
            step = np.empty(grid.size)
            choice = np.empty(grid.size, dtype=np.int32)
            for j in range(grid.size):
                # Boundaries must advance, so only grid points up to j are legal
                # predecessors; the prior pulls the gap toward want[i].
                candidates = (
                    cost[: j + 1] + weight * np.abs((grid[j] - grid[: j + 1]) - want[i]) / seg_scale
                )
                k = int(np.argmin(candidates))
                step[j] = candidates[k]
                choice[j] = k
            cost = step + fit[i]
            back[i] = choice

        tail = weight * np.abs((episode_end - grid) - want[n_boundaries]) / seg_scale
        index = int(np.argmin(cost + tail))
        starts = [0.0] * n_boundaries
        for i in range(n_boundaries - 1, -1, -1):
            starts[i] = float(grid[index])
            index = int(back[i, index])

        solved: dict[int, tuple[float, float]] = {0: (episode_start, starts[0])}
        for i, start in enumerate(starts):
            end = starts[i + 1] if i + 1 < n_boundaries else episode_end
            solved[i + 1] = (start, end)

        self._warn_on_low_alignment_confidence(record, model, starts, duration, calibration)
        return solved

    def _warn_on_low_alignment_confidence(
        self,
        record: EpisodeRecord,
        model: np.ndarray,
        solved: Sequence[float],
        duration: float,
        calibration: _AlignCalibration,
    ) -> None:
        """Warn when the model and the fitted priors disagree unusually badly.

        Calibration hides the symptom it used to be safe to look for. Before it,
        a failed alignment looked failed — boundaries bunched together or out of
        order. The solver now returns a well-ordered segmentation matching
        plausible subtask lengths *whatever the model said*, so a degenerate
        episode is no longer visible in the output, and
        ``subtask_align_min_fraction`` cannot catch it either because every label
        is placed.

        The disagreement between the model's answer and the solved boundaries is
        already computed, and on a healthy episode it stays within the residual
        spread the seed set showed. Well outside that, the spans downstream are
        mostly prior, not observation — still plausible, but no longer evidence.
        """
        if not len(solved) or duration <= 0:
            return
        scale = max(1e-6, calibration.residual_scale * duration)
        disagreement = max(abs(float(m) - float(b)) for m, b in zip(model, solved, strict=True))
        # Three residual sigmas: comfortably past the seed set's own spread
        # without firing on the ordinary tail.
        if disagreement > 3.0 * scale:
            logger.warning(
                "episode %d: alignment calibration moved a boundary %.1fs (%.1f residual sigma) "
                "from the model's answer; the spans are mostly the duration prior rather than "
                "observation — check this episode before trusting it",
                record.episode_index,
                disagreement,
                disagreement / scale,
            )

    def _rebudget_align_contact_sheet(
        self,
        record: EpisodeRecord,
        decoded_views: Sequence[tuple[str, list[Any]]],
        timestamps: Sequence[float],
    ) -> tuple[list[tuple[str, list[Any]]], list[float]]:
        """Keep a native-video fallback within the camera-frame budget.

        Native multiview video spends the budget once per timestamp because its
        synchronized views are stacked inside each video frame. Contact sheets
        account for every camera frame separately. If video construction fails,
        deterministically thin the already-decoded synchronized grid before
        building sheets; this avoids another decode and keeps badge timestamps
        identical to the retained source frames.
        """
        views = [(key, list(frames)) for key, frames in decoded_views]
        times = [float(timestamp) for timestamp in timestamps]
        if len(views) <= 1 or not times:
            return views, times

        per_view = max(1, int(self.config.max_frames_per_prompt) // len(views))
        if len(times) <= per_view:
            return views, times
        if per_view == 1:
            indices = [0]
        else:
            last = len(times) - 1
            indices = [round(index * last / (per_view - 1)) for index in range(per_view)]

        logger.info(
            "episode %d: thinning native-video fallback from %d to %d timestamp(s) "
            "per view so %d contact-sheet camera frame(s) honor the budget",
            record.episode_index,
            len(times),
            len(indices),
            len(indices) * len(views),
        )
        return (
            [(key, [frames[index] for index in indices]) for key, frames in views],
            [times[index] for index in indices],
        )

    def _align_encode_clip(
        self,
        record: EpisodeRecord,
        decoded_views: Sequence[tuple[str, list[Any]]],
        timestamps: Sequence[float],
    ) -> Path | None:
        """Encode the alignment frames as one clip, or ``None`` for contact sheets.

        Rejections return ``None`` to the caller. It either uses the decoded
        frames as contact sheets or raises when native video is configured as a
        hard transport contract.
        """
        if self.config.subtask_align_frame_format != "video":
            return None
        if self.config.subtask_align_sampling != "uniform":
            logger.warning(
                "episode %d: subtask_align_frame_format='video' needs the uniform timestamp grid "
                "to keep clip time equal to episode time; a native clip is unavailable",
                record.episode_index,
            )
            return None

        # Synchronized views are stacked into one image per timestamp, so a
        # multiview clip keeps the full timeline instead of trading it away —
        # the contact-sheet path splits its budget across cameras, which halved
        # temporal density. Frame count turned out not to drive accuracy here
        # (32 frames scored as well as 280), so spending pixels on views rather
        # than timestamps is the better use of the same prompt.
        if len(decoded_views) > 1:
            clip_frames: Sequence[Any] = to_stacked_view_frames(
                decoded_views,
                len(timestamps),
                frame_width=self.config.contact_sheet_frame_width,
            )
            if not clip_frames:
                logger.warning(
                    "episode %d: could not stack %d synchronized views for the alignment clip; "
                    "a native clip is unavailable",
                    record.episode_index,
                    len(decoded_views),
                )
                return None
        else:
            clip_frames = decoded_views[0][1]

        path: Path | None = None
        try:
            handle, name = tempfile.mkstemp(prefix="lerobot-align-", suffix=".mp4")
            path = Path(name)
            os.close(handle)
            fps = encode_frames_to_clip(
                clip_frames,
                timestamps,
                path,
                frame_width=self.config.contact_sheet_frame_width,
            )
        except Exception as exc:
            logger.warning(
                "episode %d: could not create the native alignment clip: %s",
                record.episode_index,
                exc,
            )
            if path is not None:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "could not remove failed alignment clip %s",
                        path,
                        exc_info=True,
                    )
            return None
        assert path is not None
        try:
            clip_is_empty = fps is None or not path.exists() or path.stat().st_size == 0
        except OSError as exc:
            logger.warning(
                "episode %d: could not inspect the native alignment clip: %s",
                record.episode_index,
                exc,
            )
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "could not remove uninspectable alignment clip %s",
                    path,
                    exc_info=True,
                )
            return None
        if clip_is_empty:
            logger.warning("episode %d: the alignment clip encoded empty", record.episode_index)
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning(
                    "could not remove empty alignment clip %s",
                    path,
                    exc_info=True,
                )
            return None
        logger.info(
            "episode %d: aligning from a %d-frame clip at %.4f fps (%.2fs of playback)",
            record.episode_index,
            len(timestamps),
            fps,
            float(timestamps[-1]) - float(timestamps[0]),
        )
        return path

    def _align_camera_keys(self, *, warn: bool = True) -> list[str | None]:
        """Resolve configured alignment views without assuming camera roles.

        Empty configuration preserves the historical provider-default call.
        Explicit keys are intersected with the provider's decodable video keys
        in stable order. If none survive, fall back to the provider default so
        a dataset-specific optional view never turns alignment into a no-op.
        """
        requested = list(
            dict.fromkeys(str(key) for key in self.config.subtask_align_camera_keys if key)
        )
        if not requested:
            return [None]

        available = list(getattr(self.frame_provider, "camera_keys", []) or [])
        selected = [key for key in requested if key in available]
        missing = [key for key in requested if key not in available]
        if missing and warn:
            logger.warning(
                "alignment camera key(s) unavailable and skipped: %s; available=%s",
                missing,
                available,
            )
        camera_budget = max(1, int(self.config.max_frames_per_prompt))
        if len(selected) > camera_budget:
            if warn:
                logger.warning(
                    "alignment camera-frame budget %d cannot retain all %d requested cameras; "
                    "using the first %d",
                    camera_budget,
                    len(selected),
                    camera_budget,
                )
            selected = selected[:camera_budget]
        if selected:
            return selected

        default = getattr(self.frame_provider, "camera_key", None)
        if default is None and available:
            default = available[0]
        if default is not None:
            if warn:
                logger.warning(
                    "none of the configured alignment cameras were available; falling back to %r",
                    default,
                )
            return [str(default)]
        return [None]

    def _check_align_coverage(self, record: EpisodeRecord, n_aligned: int, n_given: int) -> None:
        """Abort when too few of the given subtasks were placed.

        The stitch closes the gap left by every dropped subtask, so a mostly
        failed alignment still yields spans that cover the episode and pass
        validation — plausible-looking output that is substantially wrong. A
        run that gets this far is better stopped than published.
        """
        threshold = float(self.config.subtask_align_min_fraction)
        if threshold <= 0.0 or n_given == 0:
            return
        if n_aligned / n_given >= threshold:
            return
        raise AlignmentCoverageError(
            f"episode {record.episode_index}: aligned only {n_aligned}/{n_given} given subtask(s), "
            f"below subtask_align_min_fraction={threshold:g}. The supplied list may not match this "
            f"episode, or the camera may not show the events (try --vlm.camera_key=<wider view>). "
            f"Set --plan.subtask_align_min_fraction=0 to stitch over the gaps instead of failing."
        )

    def _align_sample_timestamps(
        self, record: EpisodeRecord, *, camera_count: int | None = None
    ) -> list[float]:
        """Tile timestamps for the alignment call, at ``frames_per_second``.

        Capped so the total across synchronized cameras stays within
        ``max_frames_per_prompt`` and the whole episode fits in ONE call —
        alignment must never be split, because each chunk would be a separate
        prompt with no idea which part of the episode it covers, and the model
        restarts its indices at 0 in every one (measured).

        Sub-sampling costs almost nothing here: the reply is one span per label,
        not one decision per tile, so the model only needs enough tiles to see
        each event happen. Against the human labels, 94 tiles (1 fps) and 280
        tiles (3 fps) of the same episode scored within 3 points of each other.
        """
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        fps = max(1e-6, float(self.config.frames_per_second))
        n = max(1, int(round((t_last - t0) * fps)) + 1)
        if camera_count is None:
            camera_count = len(self._align_camera_keys(warn=False))
        per_camera_budget = max(1, int(self.config.max_frames_per_prompt) // max(1, camera_count))
        n = min(n, per_camera_budget)
        uniform = self._uniform_episode_timestamps(record, n)
        if self.config.subtask_align_sampling == "uniform" or n < 2:
            return uniform

        if not self._motion_feature_keys:
            logger.info(
                "episode %d: no usable numeric motion features; using uniform alignment sampling",
                record.episode_index,
            )
            return uniform
        try:
            signal_groups = record.frame_columns(self._motion_feature_keys)
        except Exception as exc:
            logger.warning(
                "episode %d: failed to read motion features %s; using uniform alignment sampling: %s",
                record.episode_index,
                list(self._motion_feature_keys),
                exc,
            )
            return uniform

        selected = select_motion_stratified_timestamps(record.frame_timestamps, signal_groups, n)
        if selected is None:
            logger.info(
                "episode %d: motion features were constant, malformed, or too noisy; "
                "using uniform alignment sampling",
                record.episode_index,
            )
            return uniform
        logger.info(
            "episode %d: motion-stratified alignment sampling selected %d timestamp(s) from %s",
            record.episode_index,
            len(selected),
            list(self._motion_feature_keys),
        )
        return selected

    def _resolve_motion_feature_keys(self) -> tuple[str, ...]:
        """Find robot-agnostic numeric observation vectors from metadata."""
        if self.root is None:
            logger.warning(
                "motion-stratified alignment requested without a dataset root; uniform fallback will be used"
            )
            return ()
        try:
            features = load_info(Path(self.root)).features
        except Exception as exc:
            logger.warning(
                "could not load dataset metadata for motion-stratified alignment; "
                "uniform fallback will be used: %s",
                exc,
            )
            return ()

        def usable(key: str) -> bool:
            feature = features.get(key)
            if not isinstance(feature, dict) or not str(feature.get("dtype", "")).startswith(
                "float"
            ):
                return False
            shape = feature.get("shape")
            if not isinstance(shape, list | tuple) or len(shape) != 1:
                return False
            try:
                dimension = int(shape[0])
            except (TypeError, ValueError, OverflowError):
                return False
            return 0 < dimension <= 256

        requested = tuple(dict.fromkeys(self.config.subtask_align_motion_feature_keys))
        if requested:
            selected = tuple(key for key in requested if usable(key))
            rejected = [key for key in requested if key not in selected]
            if rejected:
                logger.warning(
                    "motion feature key(s) missing or not floating-point vectors and skipped: %s",
                    rejected,
                )
            return selected

        observations = tuple(
            key for key in features if key.startswith("observation.") and usable(key)
        )
        if observations:
            return observations
        # Commanded actions may lead the physical/video response, so they are a
        # fallback only when the dataset carries no measured numeric state.
        return ("action",) if usable("action") else ()

    def _align_spans_in_order(
        self,
        record: EpisodeRecord,
        subtasks: Sequence[str],
        placed: dict[int, tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Ordered spans from ``{index: (start, end)}``, dropping what won't fit.

        Starts are forced strictly increasing: a subtask placed at or before the
        previously kept one is dropped rather than reordered, because the
        supplied list is the authority on ordering.

        Dropped subtasks are logged loudly — the stitch below closes the gap
        they leave, so an unaligned subtask is otherwise invisible in the
        output.
        """
        spans: list[dict[str, Any]] = []
        dropped: list[str] = []
        last_start: float | None = None
        for idx, text in enumerate(subtasks):
            hit = placed.get(idx)
            if hit is None:
                dropped.append(f"{idx}:{text!r} (never assigned)")
                continue
            start, end = hit
            if last_start is not None and start <= last_start:
                dropped.append(f"{idx}:{text!r} (out of order at {start:.2f}s)")
                continue
            spans.append({"text": text, "start": start, "end": end})
            last_start = start

        if dropped:
            logger.warning(
                "episode %d: aligned %d/%d given subtask(s); dropped %s. The "
                "remaining spans are stitched over the gaps, so check that the "
                "supplied list matches what this episode actually shows.",
                record.episode_index,
                len(spans),
                len(subtasks),
                "; ".join(dropped),
            )
        return spans

    def _stitch_full_coverage(
        self, spans: list[dict[str, Any]], record: EpisodeRecord
    ) -> list[dict[str, Any]]:
        """Make subtask spans tile the full episode with no gaps.

        * The first subtask starts at the episode's first frame ``t0``
          (any idle / approach before the first labelled action is folded
          into it), so every early frame has an active subtask.
        * Each subtask's ``end`` is snapped to the next subtask's
          ``start`` (gaps between spans are closed), and the final
          subtask's ``end`` extends to the last frame ``t_last``.

        Starts are otherwise left as the (already frame-snapped, distinct)
        values the VLM produced — only the FIRST start is pulled
        back to ``t0``, which can't collide with a later span because it
        was already the earliest. Purely deterministic; runs after the
        VLM passes.
        """
        if not spans or not record.frame_timestamps:
            return spans
        t0 = float(record.frame_timestamps[0])
        t_last = float(record.frame_timestamps[-1])
        spans = sorted(spans, key=lambda s: float(s["start"]))
        spans[0]["start"] = t0
        for i in range(len(spans) - 1):
            spans[i]["end"] = float(spans[i + 1]["start"])
        spans[-1]["end"] = t_last
        for s in spans:
            if float(s["end"]) < float(s["start"]):
                s["end"] = float(s["start"])
        return spans

    @staticmethod
    def _with_causal_rules(prompt: str) -> str:
        """Append the causal event-boundary rules to a describe/segment prompt."""
        return f"{prompt}\n\n{_CAUSAL_BOUNDARY_RULES}"

    def _clean_spans(
        self,
        spans: Any,
        record: EpisodeRecord,
        bounds: tuple[float, float] | None = None,
        dedupe: bool = True,
    ) -> list[dict[str, Any]]:
        """Clamp / sort / (optionally) dedupe raw VLM subtask spans into valid rows.

        ``bounds`` overrides the clamp range — pass the window's
        ``(w_lo, w_hi)`` when cleaning window-relative spans, or leave
        ``None`` to clamp to the whole episode ``[t0, t_last]``.
        ``dedupe`` runs the frame-snap distinct-start step; skip it for
        window-relative spans (frame snapping is done once on the merged,
        absolute-time set).
        """
        if spans is None:
            return []
        if not isinstance(spans, list):
            raise ValueError(f"episode {record.episode_index}: subtasks must be a list of span objects")
        if bounds is not None:
            lo, hi = float(bounds[0]), float(bounds[1])
        else:
            lo = record.frame_timestamps[0]
            hi = record.frame_timestamps[-1]
        cleaned: list[dict[str, Any]] = []
        for index, span in enumerate(spans):
            if not isinstance(span, dict):
                raise ValueError(f"episode {record.episode_index}: subtask {index} must be an object")
            try:
                if any(isinstance(span.get(key), bool) for key in ("start", "end")):
                    raise ValueError("boolean timestamp")
                start = float(span["start"])
                end = float(span["end"])
                if not math.isfinite(start) or not math.isfinite(end):
                    raise ValueError("non-finite timestamp")
            except (KeyError, ValueError, TypeError) as exc:
                raise ValueError(
                    f"episode {record.episode_index}: subtask {index} needs finite start/end timestamps"
                ) from exc
            raw_text = span.get("text") if isinstance(span, dict) else None
            if not isinstance(raw_text, str):
                raise ValueError(f"episode {record.episode_index}: subtask {index} needs string text")
            text = raw_text.strip()
            start = max(lo, min(start, hi))
            end = max(lo, min(end, hi))
            if end < start:
                start, end = end, start
            if not text:
                raise ValueError(f'episode {record.episode_index}: subtask {index} needs non-empty text')
            cleaned.append({"text": text, "start": start, "end": end})
        cleaned.sort(key=lambda s: s["start"])
        if dedupe:
            return self._dedupe_starts_to_distinct_frames(cleaned, record)
        return cleaned

    def _describe_episode(
        self, record: EpisodeRecord, task: str, window: tuple[float, float] | None = None
    ) -> str:
        """Grounding pass: free-form chronological description of the (windowed) video."""
        contact_sheet_prompt = self._with_causal_rules(
            load_prompt("plan_subtask_describe").format(episode_task=task)
        )
        native_video_prompt = self._with_causal_rules(
            load_prompt("plan_subtask_describe_video").format(episode_task=task)
        )
        text, _uses_clip_time = self._generation_visual_field(
            record,
            field="description",
            contact_sheet_prompt=contact_sheet_prompt,
            native_video_prompt=native_video_prompt,
            window=window,
        )
        return text.strip() if isinstance(text, str) and text.strip() else ""

    @staticmethod
    def _dedupe_starts_to_distinct_frames(
        spans: list[dict[str, Any]], record: EpisodeRecord
    ) -> list[dict[str, Any]]:
        """Bump same-frame subtask starts onto distinct frames.

        Two consecutive VLM spans whose ``start`` rounds to the same
        source frame (after :func:`snap_to_frame`) would otherwise emit
        two ``style=subtask`` rows at the identical persistent
        timestamp. The training-time renderer's ``active_at(t,
        style=subtask)`` resolver can't disambiguate that and raises
        ``Ambiguous resolver for style='subtask'``.

        Walk the (sorted-by-start) spans, snap each to its frame, and
        if the snapped frame is already taken push the span onto the
        next unused frame so both subtasks survive on distinct
        timestamps. If the episode ends before a free frame is found,
        the trailing span is dropped with a warning — better than
        poisoning the render.
        """
        if not spans:
            return spans
        frames = record.frame_timestamps
        if not frames:
            return spans
        used: set[float] = set()
        out: list[dict[str, Any]] = []
        for span in spans:
            ts = snap_to_frame(span["start"], frames)
            if ts in used:
                next_ts = next((f for f in frames if f > ts and f not in used), None)
                if next_ts is None:
                    logger.warning(
                        "episode %d: subtask %r snapped to occupied frame "
                        "%.3f and no free later frame exists — dropping",
                        record.episode_index,
                        span.get("text"),
                        ts,
                    )
                    continue
                ts = next_ts
            used.add(ts)
            new_span = {**span, "start": ts}
            if float(new_span.get("end", ts)) < ts:
                new_span["end"] = ts
            out.append(new_span)
        return out

    def _generate_plan(
        self,
        record: EpisodeRecord,  # noqa: ARG002  (kept for signature stability)
        subtask_spans: Sequence[dict[str, Any]],
        *,
        refresh_t: float | None = None,
    ) -> str | None:
        """Deterministic plan = numbered list of *still-todo* subtasks.

        No VLM call: a plain numbered list keeps the plan aligned with the
        upcoming subtasks (the old VLM "compact hierarchical plan" prompt
        cost a round-trip per episode/refresh and could diverge).

            1. <subtask 1>
            2. <subtask 2>

        At ``refresh_t`` (used for each boundary and for the non-boundary
        fallback in ``run_plan_updates``), only subtasks starting at or after
        that time are included, so the value always describes what's left.
        """
        if not subtask_spans:
            return None
        remaining = [
            s
            for s in subtask_spans
            if refresh_t is None or float(s.get("start", 0.0)) >= float(refresh_t)
        ]
        if not remaining:
            # Past the last subtask boundary at a late timestamp — nothing
            # left to plan; emit None so the caller skips the row.
            return None
        return "\n".join(
            f"{i}. {span.get('text', '').strip()}" for i, span in enumerate(remaining, start=1)
        )

    def _generate_memory(
        self,
        record: EpisodeRecord,
        prior_memory: str,
        completed: str,
        remaining: Sequence[str],
        *,
        task: str | None = None,
    ) -> str:
        prompt = load_prompt("plan_memory").format(
            episode_task=(task if task is not None else record.episode_task),
            prior_memory=prior_memory or "(none)",
            completed_subtask=completed,
            remaining_subtasks=", ".join(remaining) if remaining else "(none)",
        )
        memory = self._vlm_field(self._text_message(prompt), "memory")
        return memory.strip() if isinstance(memory, str) else ""
