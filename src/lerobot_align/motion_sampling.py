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
"""Robot-agnostic motion-aware timestamp sampling.

The selector in this module deliberately knows nothing about robot layouts,
feature names, or tasks. It consumes one or more two-dimensional numeric
signal groups sampled at exact frame timestamps. A group may contain joint
positions, base pose, end-effector pose, or any other continuous state.

Motion only chooses *within* equal-time strata. Consequently a short burst of
activity cannot consume the whole frame budget, and stationary parts of an
episode remain represented. The first and last timestamp are always retained.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def select_motion_stratified_timestamps(
    timestamps: Sequence[float] | np.ndarray,
    signal_groups: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    budget: int,
    *,
    motion_weight: float = 0.65,
    smoothing_window: int | None = None,
    smoothing_seconds: float = 0.1,
    min_finite_fraction: float = 0.8,
) -> list[float] | None:
    """Select exact timestamps using temporal coverage and motion transitions.

    Args:
        timestamps: Timestamp for every signal row. Input may be unsorted and
            may contain duplicates; signals are reordered with it and the first
            row for each duplicate timestamp is retained.
        signal_groups: A mapping or sequence of numeric ``(time, dimensions)``
            arrays. A single 2-D array is accepted as a convenience. Invalid
            groups and unusable dimensions are ignored. Groups are scored
            independently before taking their maximum so a high-dimensional
            or duplicated group does not overwhelm another signal source.
        budget: Maximum number of timestamps to return. When enough timestamps
            exist, exactly ``budget`` are returned. A multi-frame episode needs
            a budget of at least two so both endpoints can be preserved.
        motion_weight: Weight of motion salience versus proximity to the center
            of an equal-time stratum.
        smoothing_window: Optional moving-average width in source frames.
            Primarily useful for controlled tests. When omitted, the width is
            derived from ``smoothing_seconds`` and the median source-frame
            interval so filtering has the same physical meaning across FPS.
        smoothing_seconds: Physical smoothing duration used when
            ``smoothing_window`` is omitted.
        min_finite_fraction: Minimum finite-sample coverage required for an
            individual signal dimension.

    Returns:
        Sorted, unique timestamps drawn exactly from ``timestamps``, or
        ``None`` when the timestamps or every supplied signal group are
        unusable. ``None`` is an explicit instruction for a caller to fall
        back to its uniform sampler.

    Notes:
        Each signal dimension with adequate coverage is linearly filled across
        non-finite samples, winsorized to its 5th--95th percentile range, and
        robustly scaled. Dense frame-to-frame jitter with little episode-scale
        range is rejected as noise. Derivatives use the actual timestamp
        deltas. The per-frame salience favors changes in smoothed motion
        (onsets and offsets) over sustained high velocity.
    """
    if budget <= 0:
        raise ValueError("budget must be positive")
    if not math.isfinite(motion_weight) or not 0.0 <= motion_weight <= 1.0:
        raise ValueError("motion_weight must be between 0 and 1")
    if smoothing_window is not None and smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive")
    if not math.isfinite(smoothing_seconds) or smoothing_seconds <= 0:
        raise ValueError("smoothing_seconds must be positive")
    if not math.isfinite(min_finite_fraction) or not 0.0 < min_finite_fraction <= 1.0:
        raise ValueError("min_finite_fraction must be between 0 and 1")

    prepared_timestamps = _prepare_timestamps(timestamps)
    if prepared_timestamps is None:
        return None
    sorted_timestamps, row_indices, original_count = prepared_timestamps
    timestamp_count = len(sorted_timestamps)
    if timestamp_count < 2:
        return None
    if budget < 2:
        raise ValueError("budget must be at least 2 when multiple timestamps are available")

    groups = _prepare_signal_groups(
        signal_groups,
        sorted_timestamps,
        row_indices=row_indices,
        original_count=original_count,
        min_finite_fraction=min_finite_fraction,
    )
    if not groups:
        return None

    selected_count = min(budget, timestamp_count)
    if selected_count == timestamp_count:
        return sorted_timestamps.tolist()

    resolved_smoothing_window = (
        smoothing_window
        if smoothing_window is not None
        else _smoothing_window_for_timestamps(sorted_timestamps, smoothing_seconds)
    )
    salience = np.max(
        [
            _motion_transition_salience(sorted_timestamps, group, resolved_smoothing_window)
            for group in groups
        ],
        axis=0,
    )
    selected_indices = _select_from_time_strata(
        sorted_timestamps,
        salience,
        selected_count,
        motion_weight=motion_weight,
    )
    return [float(sorted_timestamps[index]) for index in selected_indices]


def _prepare_timestamps(
    timestamps: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return sorted unique timestamps and their original row indices."""
    try:
        values = np.asarray(timestamps, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        return None

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    keep = np.ones(len(sorted_values), dtype=bool)
    keep[1:] = sorted_values[1:] != sorted_values[:-1]
    return sorted_values[keep], order[keep], len(values)


def _prepare_signal_groups(
    signal_groups: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    timestamps: np.ndarray,
    *,
    row_indices: np.ndarray,
    original_count: int,
    min_finite_fraction: float,
) -> list[np.ndarray]:
    """Coerce groups and retain only robustly varying numeric dimensions."""
    groups: list[np.ndarray] = []
    for raw_group in _iter_raw_groups(signal_groups):
        try:
            values = np.asarray(raw_group, dtype=np.float64)
        except (TypeError, ValueError):
            continue
        if values.ndim != 2 or values.shape[0] != original_count or values.shape[1] == 0:
            continue
        values = values[row_indices]

        normalized_dimensions: list[np.ndarray] = []
        for dimension in values.T:
            normalized = _normalize_dimension(
                timestamps,
                dimension,
                min_finite_fraction=min_finite_fraction,
            )
            if normalized is not None:
                normalized_dimensions.append(normalized)
        if normalized_dimensions:
            groups.append(np.column_stack(normalized_dimensions))
    return groups


def _iter_raw_groups(
    signal_groups: Mapping[str, Any] | Sequence[Any] | np.ndarray,
) -> list[Any]:
    """Disambiguate one 2-D Python list from a list of 2-D groups."""
    if isinstance(signal_groups, Mapping):
        return list(signal_groups.values())
    if isinstance(signal_groups, np.ndarray):
        return [signal_groups]

    try:
        possible_single_group = np.asarray(signal_groups)
    except (TypeError, ValueError):
        possible_single_group = None
    if possible_single_group is not None and possible_single_group.ndim == 2:
        return [signal_groups]
    try:
        return list(signal_groups)
    except TypeError:
        return []


def _normalize_dimension(
    timestamps: np.ndarray,
    values: np.ndarray,
    *,
    min_finite_fraction: float = 0.8,
) -> np.ndarray | None:
    """Fill, winsorize, and robustly scale one signal dimension."""
    finite = np.isfinite(values)
    finite_count = int(np.count_nonzero(finite))
    if finite_count < 2 or finite_count / len(values) < min_finite_fraction:
        return None

    filled = np.interp(timestamps, timestamps[finite], values[finite])
    q05, median, q95 = np.quantile(filled, [0.05, 0.5, 0.95])
    clipped = np.clip(filled, q05, q95)
    q25, q75 = np.quantile(clipped, [0.25, 0.75])
    scale = float(q75 - q25)
    magnitude = max(1.0, abs(float(q05)), abs(float(q95)))
    tolerance = np.finfo(np.float64).eps * magnitude * 16
    if scale <= tolerance:
        scale = float(q95 - q05)
    if scale <= tolerance:
        return None
    normalized = (clipped - median) / scale

    # Robust scaling can magnify tiny, dense sensor jitter into an apparently
    # active trace. Reject it only when most intervals change yet the whole
    # episode's robust range is less than six typical changes. Sparse steps
    # (such as a gripper toggle) bypass this gate.
    steps = np.abs(np.diff(normalized))
    step_tolerance = max(np.finfo(np.float64).eps * 16, float(q95 - q05) * 1e-6 / scale)
    active_steps = steps[steps > step_tolerance]
    # Short traces do not contain enough intervals to distinguish jitter from
    # an intentionally rapid action; leave them usable and rely on stratified
    # temporal coverage.
    if len(steps) >= 30 and len(active_steps) / len(steps) >= 0.25:
        typical_step = float(np.median(active_steps))
        robust_range = float(np.quantile(normalized, 0.95) - np.quantile(normalized, 0.05))
        if typical_step > 0.0 and robust_range / typical_step < 6.0:
            return None
    return normalized


def _smoothing_window_for_timestamps(timestamps: np.ndarray, duration_s: float) -> int:
    """Convert a physical smoothing duration to a nearby odd frame width."""
    positive_deltas = np.diff(timestamps)
    positive_deltas = positive_deltas[positive_deltas > 0.0]
    if positive_deltas.size == 0:
        return 1
    width = max(1, int(round(duration_s / float(np.median(positive_deltas)))))
    return width + 1 if width % 2 == 0 else width


def _motion_transition_salience(
    timestamps: np.ndarray,
    normalized_group: np.ndarray,
    smoothing_window: int,
) -> np.ndarray:
    """Compute group-size-neutral motion-onset/offset salience."""
    delta_t = np.diff(timestamps)
    velocity = np.diff(normalized_group, axis=0) / delta_t[:, np.newaxis]
    # RMS gives each normalized dimension equal influence; taking the mean
    # inside the square root prevents larger groups from winning by dimension
    # count alone.
    interval_speed = np.sqrt(np.mean(np.square(velocity), axis=1))
    interval_speed = _winsorize_nonnegative(interval_speed)
    interval_speed = _smooth(interval_speed, smoothing_window)

    frame_motion = np.empty(len(timestamps), dtype=np.float64)
    frame_motion[0] = interval_speed[0]
    frame_motion[-1] = interval_speed[-1]
    frame_motion[1:-1] = (interval_speed[:-1] + interval_speed[1:]) / 2

    transition = np.zeros(len(timestamps), dtype=np.float64)
    interval_midpoints = (timestamps[:-1] + timestamps[1:]) / 2
    transition[1:-1] = np.abs(np.diff(interval_speed)) / np.diff(interval_midpoints)
    transition = _winsorize_nonnegative(transition)

    # Boundary location benefits more from a motion onset/offset than from a
    # frame in the middle of a long, steady movement.
    return 0.25 * _unit_score(frame_motion) + 0.75 * _unit_score(transition)


def _winsorize_nonnegative(values: np.ndarray) -> np.ndarray:
    """Cap spikes while retaining a genuinely sparse non-zero signal."""
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    cap = float(np.quantile(values, 0.95))
    if cap <= np.finfo(np.float64).eps:
        cap = float(np.max(values, initial=0.0))
    if cap <= np.finfo(np.float64).eps:
        return np.zeros_like(values)
    return np.minimum(values, cap)


def _smooth(values: np.ndarray, requested_width: int) -> np.ndarray:
    """Centered moving average with edge replication and stable length."""
    width = min(requested_width, len(values))
    if width % 2 == 0:
        width -= 1
    if width <= 1:
        return values.copy()
    radius = width // 2
    padded = np.pad(values, (radius, radius), mode="edge")
    kernel = np.full(width, 1.0 / width)
    return np.convolve(padded, kernel, mode="valid")


def _unit_score(values: np.ndarray) -> np.ndarray:
    """Map a non-negative sequence to ``[0, 1]`` without erasing rare peaks."""
    cap = float(np.quantile(values, 0.95))
    if cap <= np.finfo(np.float64).eps:
        cap = float(np.max(values, initial=0.0))
    if cap <= np.finfo(np.float64).eps:
        return np.zeros_like(values)
    return np.clip(values / cap, 0.0, 1.0)


def _select_from_time_strata(
    timestamps: np.ndarray,
    salience: np.ndarray,
    selected_count: int,
    *,
    motion_weight: float,
) -> list[int]:
    """Keep endpoints and choose one candidate near every interior anchor."""
    if selected_count == 2:
        return [0, len(timestamps) - 1]

    anchors = np.linspace(timestamps[0], timestamps[-1], selected_count)
    selected = {0, len(timestamps) - 1}
    empty_anchor_indices: list[int] = []

    for anchor_index in range(1, selected_count - 1):
        target = anchors[anchor_index]
        left = (anchors[anchor_index - 1] + target) / 2
        right = (target + anchors[anchor_index + 1]) / 2
        candidates = [
            index
            for index in range(1, len(timestamps) - 1)
            if index not in selected
            and timestamps[index] >= left
            and (
                timestamps[index] <= right
                if anchor_index == selected_count - 2
                else timestamps[index] < right
            )
        ]
        if not candidates:
            empty_anchor_indices.append(anchor_index)
            continue

        half_width = max(target - left, right - target)
        scored_candidates: list[tuple[float, float, int, int]] = []
        for index in candidates:
            proximity = max(0.0, 1.0 - abs(float(timestamps[index] - target)) / half_width)
            combined = motion_weight * float(salience[index]) + (1.0 - motion_weight) * proximity
            # Prefer proximity and then the earlier timestamp for exact ties.
            scored_candidates.append((combined, proximity, -index, index))

        selected.add(max(scored_candidates)[-1])

    # A stratum can be empty when frame timestamps are irregular. Fill it with
    # the closest still-unused timestamp, preserving coverage before salience.
    for anchor_index in empty_anchor_indices:
        target = anchors[anchor_index]
        candidates = [index for index in range(1, len(timestamps) - 1) if index not in selected]
        closest = min(
            candidates,
            key=lambda index: (abs(float(timestamps[index] - target)), -float(salience[index]), index),
        )
        selected.add(closest)

    return sorted(selected)
