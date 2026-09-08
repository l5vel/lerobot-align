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

from lerobot_align.alignment_metrics import evaluate_alignment


def _span(text: str, start: float, end: float) -> dict:
    return {"text": text, "start": start, "end": end}


def test_alignment_metrics_perfect_internal_boundaries() -> None:
    labels = ["approach", "grasp", "place"]
    truth = [
        {"label": "approach", "start": 0.0, "end": 2.0},
        {"label": "grasp", "start": 2.0, "end": 5.0},
        {"label": "place", "start": 5.0, "end": 8.0},
    ]
    metrics = evaluate_alignment(
        labels,
        truth,
        [_span("approach", 0.0, 2.0), _span("grasp", 2.0, 5.0), _span("place", 5.0, 8.0)],
    )

    assert metrics.placed_fraction == 1.0
    assert metrics.macro_temporal_iou == 1.0
    assert metrics.boundary_errors == (0.0, 0.0)
    assert metrics.boundary_hit_rates == {1.0: 1.0, 3.0: 1.0, 5.0: 1.0}


def test_alignment_metrics_penalize_missing_labels_and_exclude_forced_first_start() -> None:
    labels = ["approach", "grasp", "place"]
    truth = [
        {"label": "approach", "start": 0.0, "end": 2.0},
        {"label": "grasp", "start": 2.0, "end": 5.0},
        {"label": "place", "start": 5.0, "end": 8.0},
    ]
    metrics = evaluate_alignment(
        labels,
        truth,
        [_span("approach", 0.0, 2.0), _span("place", 6.0, 8.0)],
    )

    assert metrics.placed_fraction == 2 / 3
    assert metrics.macro_temporal_iou == (1.0 + 0.0 + 2 / 3) / 3
    assert metrics.boundary_errors == (None, 1.0)
    # The missing internal boundary stays in the denominator.
    assert metrics.boundary_hit_rates[1.0] == 0.5
    # Its summary-statistic penalty is the worst possible in-episode error:
    # max(|2-0|, |8-2|) = 6 seconds.
    assert metrics.boundary_mae == 3.5
