"""Label-free temporal segmentation metrics.

Every function here scores geometry only: where the cuts fall and how the
spans overlap. Nothing in this module looks at the label text, so a tool is
never rewarded or punished for wording. Semantic scoring lives in
``semantic.py``; the joint metrics that need both live in ``joint.py``.

The predicted and reference segmentations are each an ordered, non-overlapping
list of spans. They are NOT required to have the same number of segments --
that difference is the thing we are measuring -- but both are assumed to
describe the same episode over the same ``[0, duration]`` support.

Conventions
-----------
A *span* is a mapping with float ``start`` and ``end`` (seconds) and, where
relevant, a ``text`` label. An *internal boundary* is a start time other than
the episode start: a segmentation with ``M`` spans has ``M - 1`` of them. The
episode endpoints are fixed by construction for both tools, so scoring them
would inflate every arm equally and hide real differences.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Span = dict[str, Any]


# --------------------------------------------------------------------------
# basic geometry
# --------------------------------------------------------------------------

def span_bounds(span: Span) -> tuple[float, float]:
    return float(span["start"]), float(span["end"])


def intersection(a: Span, b: Span) -> float:
    a0, a1 = span_bounds(a)
    b0, b1 = span_bounds(b)
    return max(0.0, min(a1, b1) - max(a0, b0))


def union(a: Span, b: Span) -> float:
    a0, a1 = span_bounds(a)
    b0, b1 = span_bounds(b)
    return max(a1, b1) - min(a0, b0)


def iou(a: Span, b: Span) -> float:
    """Temporal IoU. Two zero-length spans at the same instant score 1.0."""
    u = union(a, b)
    if u <= 0.0:
        a0, a1 = span_bounds(a)
        b0, b1 = span_bounds(b)
        return 1.0 if (a0, a1) == (b0, b1) else 0.0
    return intersection(a, b) / u


def duration(span: Span) -> float:
    a0, a1 = span_bounds(span)
    return max(0.0, a1 - a0)


def internal_boundaries(spans: Sequence[Span]) -> list[float]:
    """Start times excluding the first, i.e. the cuts the tool actually chose."""
    return [float(s["start"]) for s in spans[1:]]


def validate_segmentation(spans: Sequence[Span], *, name: str = "segmentation") -> None:
    """Reject anything that would silently corrupt a metric.

    We are strict here on purpose. A tool that emits overlapping or
    out-of-order spans is producing a different kind of object than the one
    these metrics define, and quietly repairing it would hide a real quality
    difference between the arms.
    """
    if not spans:
        raise ValueError(f"{name} is empty")
    prev_end: float | None = None
    for index, span in enumerate(spans):
        start, end = span_bounds(span)
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"{name}[{index}] has non-finite bounds ({start}, {end})")
        if end < start:
            raise ValueError(f"{name}[{index}] ends before it starts ({start} > {end})")
        if prev_end is not None and start < prev_end - 1e-6:
            raise ValueError(f"{name}[{index}] starts at {start} before previous end {prev_end}")
        prev_end = end


# --------------------------------------------------------------------------
# boundary detection: did the tool cut in the right places?
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class BoundaryScore:
    tolerance: float
    n_pred: int
    n_true: int
    n_matched: int
    precision: float
    recall: float
    f1: float
    matched_errors: tuple[float, ...]

    @property
    def mae_matched(self) -> float | None:
        return sum(self.matched_errors) / len(self.matched_errors) if self.matched_errors else None


def _monotone_boundary_match(
    pred: Sequence[float], true: Sequence[float], tolerance: float
) -> list[tuple[int, int]]:
    """Order-preserving maximum matching between two boundary lists.

    Boundaries are inherently ordered in time, so a matching that crosses
    (pred i -> true j, pred i' > i -> true j' < j) is not physically
    meaningful. A Hungarian assignment would happily produce one. We instead
    run the standard DP over the two sequences, maximising the number of
    matched pairs first and minimising total absolute error to break ties.
    That makes the score invariant to how we enumerate candidates.
    """
    n, m = len(pred), len(true)
    # best[i][j] = (matches, -total_error) achievable from pred[i:], true[j:]
    best: list[list[tuple[int, float]]] = [[(0, 0.0)] * (m + 1) for _ in range(n + 1)]
    choice: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            skip_pred = best[i + 1][j]
            skip_true = best[i][j + 1]
            candidate = skip_pred if skip_pred >= skip_true else skip_true
            taken = 1 if skip_pred >= skip_true else 2
            error = abs(pred[i] - true[j])
            if error <= tolerance:
                nxt = best[i + 1][j + 1]
                pair = (nxt[0] + 1, nxt[1] - error)
                if pair > candidate:
                    candidate = pair
                    taken = 3
            best[i][j] = candidate
            choice[i][j] = taken
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        move = choice[i][j]
        if move == 3:
            pairs.append((i, j))
            i += 1
            j += 1
        elif move == 1:
            i += 1
        else:
            j += 1
    return pairs


def boundary_score(
    predicted: Sequence[Span], reference: Sequence[Span], *, tolerance: float
) -> BoundaryScore:
    """Precision/recall/F1 over internal boundaries at a time tolerance.

    This is the primary naming-independent metric. It is well defined when the
    two segmentations have different lengths, which is exactly the case that
    fixed-label alignment metrics cannot handle, and it degrades gracefully:
    a tool that emits too many cuts loses precision, one that emits too few
    loses recall.
    """
    pred = internal_boundaries(predicted)
    true = internal_boundaries(reference)
    if not pred and not true:
        return BoundaryScore(tolerance, 0, 0, 0, 1.0, 1.0, 1.0, ())
    pairs = _monotone_boundary_match(pred, true, tolerance)
    errors = tuple(abs(pred[i] - true[j]) for i, j in pairs)
    matched = len(pairs)
    precision = matched / len(pred) if pred else 0.0
    recall = matched / len(true) if true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return BoundaryScore(tolerance, len(pred), len(true), matched, precision, recall, f1, errors)


def boundary_distance_summary(
    predicted: Sequence[Span], reference: Sequence[Span]
) -> dict[str, float | None]:
    """Nearest-boundary distances in both directions, with no tolerance.

    Tolerance-free distances complement the F1 scores: they say how far off
    the cuts are rather than how many landed inside a window, and the two
    directions separate "missed a real boundary" from "invented a spurious
    one". Reported in seconds.
    """
    pred = internal_boundaries(predicted)
    true = internal_boundaries(reference)
    if not pred or not true:
        return {"pred_to_true_mae": None, "true_to_pred_mae": None, "chamfer": None}
    p2t = [min(abs(p - t) for t in true) for p in pred]
    t2p = [min(abs(t - p) for p in pred) for t in true]
    forward = sum(p2t) / len(p2t)
    backward = sum(t2p) / len(t2p)
    return {
        "pred_to_true_mae": forward,
        "true_to_pred_mae": backward,
        "chamfer": (forward + backward) / 2.0,
    }


# --------------------------------------------------------------------------
# partition overlap: how well do the two segmentations cover each other?
# --------------------------------------------------------------------------

def segmentation_covering(reference: Sequence[Span], predicted: Sequence[Span]) -> float:
    """Duration-weighted best-overlap covering of ``reference`` by ``predicted``.

    Borrowed from the image-segmentation literature (Arbelaez et al.), where
    it is the standard way to compare two partitions with different numbers of
    regions. Each reference span contributes its best IoU against any predicted
    span, weighted by how long it is. It needs no labels, no equal segment
    counts, and no matching heuristic, which makes it the most assumption-free
    number we report.

    It is deliberately asymmetric: covering(GT, pred) rewards finding the real
    structure, covering(pred, GT) punishes inventing structure. Report both.
    """
    total = sum(duration(span) for span in reference)
    if total <= 0.0:
        return 1.0 if not predicted else 0.0
    score = 0.0
    for span in reference:
        best = max((iou(span, other) for other in predicted), default=0.0)
        score += duration(span) * best
    return score / total


def _monotone_span_match(
    predicted: Sequence[Span], reference: Sequence[Span]
) -> list[tuple[int, int, float]]:
    """Order-preserving one-to-one span matching that maximises total IoU."""
    n, m = len(predicted), len(reference)
    best: list[list[float]] = [[0.0] * (m + 1) for _ in range(n + 1)]
    choice: list[list[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            skip_pred = best[i + 1][j]
            skip_true = best[i][j + 1]
            take = iou(predicted[i], reference[j]) + best[i + 1][j + 1]
            if take >= skip_pred and take >= skip_true:
                best[i][j], choice[i][j] = take, 3
            elif skip_pred >= skip_true:
                best[i][j], choice[i][j] = skip_pred, 1
            else:
                best[i][j], choice[i][j] = skip_true, 2
    out: list[tuple[int, int, float]] = []
    i = j = 0
    while i < n and j < m:
        move = choice[i][j]
        if move == 3:
            out.append((i, j, iou(predicted[i], reference[j])))
            i += 1
            j += 1
        elif move == 1:
            i += 1
        else:
            j += 1
    return out


def matched_iou(
    predicted: Sequence[Span], reference: Sequence[Span]
) -> dict[str, Any]:
    """Mean IoU over reference spans under an order-preserving 1:1 matching.

    Unmatched reference spans score zero, so dropping a segment is penalised
    rather than ignored. This is the closest analogue to the existing
    ``macro_temporal_iou`` in ``lerobot_align.alignment_metrics``, but it does
    not require the label lists to match, so it works for open-ended output.
    """
    pairs = _monotone_span_match(predicted, reference)
    by_ref = {j: value for _, j, value in pairs}
    ious = [by_ref.get(j, 0.0) for j in range(len(reference))]
    return {
        "macro_iou": sum(ious) / len(ious) if ious else 1.0,
        "matched": len(pairs),
        "n_pred": len(predicted),
        "n_true": len(reference),
        "per_reference_iou": ious,
        "pairs": [(i, j) for i, j, _ in pairs],
    }


# --------------------------------------------------------------------------
# segment-count behaviour
# --------------------------------------------------------------------------

def segmentation_counts(
    predicted: Sequence[Span], reference: Sequence[Span]
) -> dict[str, float]:
    """Over- and under-segmentation, reported without a sign convention trick.

    ``ratio`` above 1 means the tool cut the episode into more pieces than the
    human did. ``log_ratio`` is what should be averaged across episodes: the
    raw ratio is bounded below by 0 and unbounded above, so its mean would
    make over-segmentation look worse than the symmetric error it is.
    """
    n_pred, n_true = len(predicted), len(reference)
    ratio = n_pred / n_true if n_true else float("inf")
    return {
        "n_pred": float(n_pred),
        "n_true": float(n_true),
        "count_error": float(n_pred - n_true),
        "abs_count_error": float(abs(n_pred - n_true)),
        "ratio": ratio,
        "log_ratio": math.log(ratio) if ratio > 0 and math.isfinite(ratio) else float("nan"),
    }
