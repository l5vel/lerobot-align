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
"""Tests for generic motion-stratified timestamp selection."""

from __future__ import annotations

import numpy as np
import pytest

from lerobot_align.motion_sampling import select_motion_stratified_timestamps


def test_motion_transition_can_win_over_stratum_center() -> None:
    timestamps = np.arange(11, dtype=np.float64)
    # Rest, sustained motion beginning at t=3, then rest again. With a
    # three-frame budget the only interior stratum is centered at t=5.
    position = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 5, 5], dtype=np.float64)[:, None]

    selected = select_motion_stratified_timestamps(
        timestamps,
        {"arbitrary_pose": position},
        budget=3,
        smoothing_window=1,
    )

    assert selected == [0.0, 3.0, 10.0]


def test_selection_preserves_endpoints_and_equal_time_coverage() -> None:
    timestamps = np.arange(13, dtype=np.float64)
    # Several dimensions and concentrated activity must not let one temporal
    # region consume all three interior samples.
    signals = np.column_stack(
        [
            np.where(timestamps >= 3, 1000.0, 0.0),
            np.where(timestamps >= 4, -0.002, 0.0),
        ]
    )

    selected = select_motion_stratified_timestamps(
        timestamps,
        [signals],
        budget=5,
        smoothing_window=1,
    )

    assert selected is not None
    assert len(selected) == 5
    assert selected[0] == 0.0
    assert selected[-1] == 12.0
    assert sum(1.5 <= timestamp < 4.5 for timestamp in selected) == 1
    assert sum(4.5 <= timestamp < 7.5 for timestamp in selected) == 1
    assert sum(7.5 <= timestamp <= 10.5 for timestamp in selected) == 1


def test_derivatives_use_actual_timestamp_deltas() -> None:
    timestamps = np.array([0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.float64)
    # Position is exactly linear in time. Treating row deltas as velocity
    # would invent a transition at the three-second sampling gap (t=5).
    position = timestamps[:, None]

    selected = select_motion_stratified_timestamps(
        timestamps,
        position,
        budget=3,
        smoothing_window=1,
    )

    assert selected == [0.0, 6.0, 12.0]


def test_output_is_sorted_unique_and_drawn_from_exact_input_grid() -> None:
    timestamps = [2.7, 0.1, 1.3, 4.9, 1.3, 3.8, 6.2]
    position = np.asarray(timestamps, dtype=np.float64)[:, None]

    selected = select_motion_stratified_timestamps(timestamps, {"state": position}, budget=4)

    assert selected is not None
    assert len(selected) == 4
    assert selected == sorted(set(selected))
    assert set(selected).issubset(set(timestamps))
    assert selected[0] == min(timestamps)
    assert selected[-1] == max(timestamps)


def test_nonfinite_and_constant_dimensions_do_not_poison_valid_signal() -> None:
    timestamps = np.arange(9, dtype=np.float64)
    groups = {
        "malformed": np.ones((4, 2)),
        "mixed": np.column_stack(
            [
                np.ones(9),
                [np.nan, 0, 0, 0, 1, 2, 3, 4, 5],
                np.full(9, np.nan),
            ]
        ),
    }

    selected = select_motion_stratified_timestamps(timestamps, groups, budget=4)

    assert selected is not None
    assert len(selected) == 4
    assert selected[0] == 0.0
    assert selected[-1] == 8.0


def test_low_finite_coverage_requests_uniform_fallback() -> None:
    timestamps = np.arange(100, dtype=np.float64) / 10
    sparse = np.full((100, 1), np.nan)
    sparse[::10, 0] = np.arange(10)

    assert select_motion_stratified_timestamps(timestamps, sparse, budget=10) is None


@pytest.mark.parametrize("fps", [5, 10, 50])
def test_dense_high_frequency_noise_requests_uniform_fallback(fps: int) -> None:
    timestamps = np.arange(20 * fps, dtype=np.float64) / fps
    noise = np.random.default_rng(1729).normal(size=(len(timestamps), 3))

    assert select_motion_stratified_timestamps(timestamps, noise, budget=20) is None


def test_default_smoothing_has_consistent_physical_boundary_across_fps() -> None:
    selections = []
    for fps in (10, 50):
        timestamps = np.arange(10 * fps + 1, dtype=np.float64) / fps
        # A two-second move begins at the same physical time in each trace.
        position = np.clip(timestamps - 4.0, 0.0, 2.0)[:, None]
        selected = select_motion_stratified_timestamps(timestamps, position, budget=3)
        assert selected is not None
        selections.append(selected[1])

    assert selections[0] == pytest.approx(selections[1], abs=0.15)


@pytest.mark.parametrize(
    "signals",
    [
        {"constant": np.ones((20, 6))},
        [np.full((20, 2), np.nan), np.full((20, 1), np.inf)],
        {"wrong_rows": np.ones((19, 3))},
    ],
)
def test_unusable_signals_request_uniform_fallback(signals) -> None:
    assert select_motion_stratified_timestamps(np.arange(20), signals, budget=5) is None


def test_winsorization_rejects_a_single_sample_spike() -> None:
    timestamps = np.arange(101, dtype=np.float64)
    glitch = np.zeros((101, 1), dtype=np.float64)
    glitch[50] = 1e12

    assert select_motion_stratified_timestamps(timestamps, glitch, budget=8) is None


def test_multiple_groups_accept_different_dimension_counts() -> None:
    timestamps = np.arange(10, dtype=np.float64)
    one_dimension = timestamps[:, None] / 1000
    seven_dimensions = np.repeat(np.maximum(timestamps - 4, 0)[:, None], 7, axis=1)

    selected = select_motion_stratified_timestamps(
        timestamps,
        [one_dimension, seven_dimensions],
        budget=4,
        smoothing_window=1,
    )

    assert selected is not None
    assert len(selected) == 4


def test_budget_larger_than_grid_returns_every_exact_timestamp() -> None:
    timestamps = [0.0, 0.15, 0.41, 1.2]
    signal = np.asarray(timestamps)[:, None]

    selected = select_motion_stratified_timestamps(timestamps, signal, budget=20)

    assert selected == timestamps


def test_invalid_budget_for_endpoint_preservation_is_rejected() -> None:
    timestamps = np.arange(5, dtype=np.float64)
    signal = timestamps[:, None]

    with pytest.raises(ValueError, match="at least 2"):
        select_motion_stratified_timestamps(timestamps, signal, budget=1)
