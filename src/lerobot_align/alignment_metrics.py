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
"""Metrics for fixed-label subtask alignment diagnostics."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlignmentMetrics:
    placed_fraction: float
    macro_temporal_iou: float
    boundary_errors: tuple[float | None, ...]
    boundary_mae: float | None
    boundary_median: float | None
    boundary_p90: float | None
    boundary_hit_rates: dict[float, float]


def _ordered_span_matches(labels: list[str], predicted: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Greedily match predicted spans to the authoritative ordered labels."""
    matches: list[dict[str, Any] | None] = []
    cursor = 0
    for label in labels:
        match = None
        for index in range(cursor, len(predicted)):
            if str(predicted[index].get("text", "")) == label:
                match = predicted[index]
                cursor = index + 1
                break
        matches.append(match)
    return matches


def _temporal_iou(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_start, left_end = float(left["start"]), float(left["end"])
    right_start, right_end = float(right["start"]), float(right["end"])
    intersection = max(0.0, min(left_end, right_end) - max(left_start, right_start))
    union = max(left_end, right_end) - min(left_start, right_start)
    if union <= 0.0:
        return 1.0 if left_start == right_start and left_end == right_end else 0.0
    return intersection / union


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def evaluate_alignment(
    labels: list[str],
    ground_truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    *,
    hit_thresholds: tuple[float, ...] = (1.0, 3.0, 5.0),
) -> AlignmentMetrics:
    """Score final stitched spans and their unforced internal boundaries.

    Temporal IoU evaluates the final stitched segmentation, including its fixed
    episode endpoints. Boundary metrics do not count those endpoints and use
    only the ``len(labels) - 1`` internal starts. Missing labels remain in every
    denominator: their temporal IoU is zero, their boundary is a miss for hit
    rates, and error summaries assign the worst possible in-episode error.
    """
    if len(ground_truth) != len(labels):
        raise ValueError(
            f"ground truth has {len(ground_truth)} span(s) for {len(labels)} authoritative label(s)"
        )

    matches = _ordered_span_matches(labels, predicted)
    placed = sum(match is not None for match in matches)
    ious = [
        _temporal_iou(match, truth) if match is not None else 0.0
        for match, truth in zip(matches, ground_truth, strict=True)
    ]

    boundary_errors: list[float | None] = []
    for index in range(1, len(labels)):
        match = matches[index]
        boundary_errors.append(
            abs(float(match["start"]) - float(ground_truth[index]["start"])) if match is not None else None
        )
    # A placed-only mean can look better as alignment drops difficult labels.
    # Give every missing boundary its maximum possible in-episode error, while
    # retaining ``None`` in ``boundary_errors`` so callers can distinguish a
    # miss from an inaccurate placement.
    if ground_truth:
        episode_start = float(ground_truth[0]["start"])
        episode_end = float(ground_truth[-1]["end"])
    else:
        episode_start = episode_end = 0.0
    scored_errors = [
        error
        if error is not None
        else max(
            abs(float(ground_truth[index]["start"]) - episode_start),
            abs(episode_end - float(ground_truth[index]["start"])),
        )
        for index, error in enumerate(boundary_errors, start=1)
    ]
    denominator = len(boundary_errors)
    hit_rates = {
        threshold: (
            sum(error is not None and error <= threshold for error in boundary_errors) / denominator
            if denominator
            else 1.0
        )
        for threshold in hit_thresholds
    }
    return AlignmentMetrics(
        placed_fraction=placed / len(labels) if labels else 1.0,
        macro_temporal_iou=sum(ious) / len(ious) if ious else 1.0,
        boundary_errors=tuple(boundary_errors),
        boundary_mae=sum(scored_errors) / len(scored_errors) if scored_errors else None,
        boundary_median=statistics.median(scored_errors) if scored_errors else None,
        boundary_p90=_percentile(scored_errors, 0.9),
        boundary_hit_rates=hit_rates,
    )
