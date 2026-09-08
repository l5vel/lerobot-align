"""Joint metrics: scores that need both the timing and the label.

These are the standard temporal-action-segmentation and dense-captioning
measures, adapted so that "same class" is decided by a pluggable semantic
``Matcher`` rather than by an integer class id. Time is represented by continuous spans. Semantic matching replaces class
equality; the dense-caption diagnostic is reference-averaged semantic recall,
not the official ActivityNet caption precision metric.

Implemented here:

``f1_at_iou``      Lea et al. (MS-TCN convention) segmental F1@{10,25,50}
``edit_score``     normalised Levenshtein over the label sequence
``mof``            mean-over-frames label accuracy
``dense_caption``  reference-averaged semantic recall over IoU thresholds
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .segmentation import Span, duration, iou, span_bounds
from .semantic import Matcher, normalise_label


def f1_at_iou(
    predicted: Sequence[Span],
    reference: Sequence[Span],
    matcher: Matcher,
    *,
    threshold: float,
) -> dict[str, float]:
    """Segmental F1 at an IoU threshold, the MS-TCN convention.

    A predicted segment is a true positive when it overlaps a semantically
    equivalent reference segment with IoU >= ``threshold`` and that reference
    segment has not already been claimed. Each reference segment can be
    matched at most once. Distinct predicted label runs compete for that one
    match, so extra runs are false positives. Cuts within an identical-label
    run are removed by the MS-TCN run-length encoding convention.

    Predictions are visited in temporal order. Each chooses its best equivalent
    reference, including already claimed references; a claimed best match is an
    FP, with no fallback. Ties choose the first reference, as in the official
    implementation: https://github.com/yabufarha/ms-tcn/blob/master/eval.py.
    Adjacent identical labels are merged before segmental scoring. Similarity
    thresholds are not used to merge labels, since they are not transitive.
    """
    if not 0 < threshold <= 1:
        raise ValueError("IoU threshold must lie in (0, 1]")
    predicted, reference = collapse_spans(predicted), collapse_spans(reference)
    used: set[int] = set()
    tp = 0
    for pred in predicted:
        overlaps = [iou(pred, ref) if matcher.equivalent(
            str(pred.get("text", "")), str(ref.get("text", ""))) else 0.0
            for ref in reference]
        if not overlaps:
            continue
        j = max(range(len(overlaps)), key=overlaps.__getitem__)
        if overlaps[j] >= threshold and j not in used:
            used.add(j)
            tp += 1
    fp = len(predicted) - tp
    fn = len(reference) - tp
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        f"f1@{int(threshold * 100)}": f1,
        f"precision@{int(threshold * 100)}": precision,
        f"recall@{int(threshold * 100)}": recall,
        f"tp@{int(threshold * 100)}": float(tp),
    }


def collapse_spans(spans: Sequence[Span]) -> list[Span]:
    """Merge contiguous runs of the same normalised label without mutating input."""
    out: list[Span] = []
    for span in spans:
        if (out and float(out[-1]["end"]) == float(span["start"])
                and normalise_label(out[-1].get("text", ""))
                == normalise_label(span.get("text", ""))):
            out[-1]["end"] = span["end"]
        else:
            out.append(dict(span))
    return out


def collapsed_labels(spans: Sequence[Span]) -> list[str]:
    """Run-length encode label identity; semantic thresholds are not equivalence classes."""
    out: list[str] = []
    for span in spans:
        label = str(span.get("text", ""))
        if not out or normalise_label(out[-1]) != normalise_label(label):
            out.append(label)
    return out


def edit_score(
    predicted: Sequence[Span], reference: Sequence[Span], matcher: Matcher
) -> float:
    """Normalised segmental edit distance over the two label sequences.

    Collapses each segmentation to its ordered label sequence and computes
    1 - Levenshtein / max(len). Time is ignored entirely, so this isolates
    whether the tool recovered the right *sequence of steps* -- the ordering
    and multiplicity -- from whether it placed them accurately. It is the
    standard companion to F1@IoU in the action-segmentation literature and is
    insensitive to extra cuts within an identical-label run.
    """
    a = collapsed_labels(predicted)
    b = collapsed_labels(reference)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, right in enumerate(b, start=1):
            cost = 0 if matcher.equivalent(left, right) else 1
            current[j] = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost)
        previous = current
    distance = previous[-1]
    return 1.0 - distance / max(len(a), len(b))


def mof(
    predicted: Sequence[Span],
    reference: Sequence[Span],
    matcher: Matcher,
    *,
    step: float = 0.1,
) -> dict[str, float]:
    """Mean-over-frames accuracy on a fixed time grid.

    The fraction of episode time whose predicted label is semantically
    equivalent to the reference label there. Sampled on a regular ``step``
    grid rather than at native fps so the cost does not scale with the 50 Hz
    source and so every dataset is weighted the same way per second.

    Reported alongside F1@IoU because the two fail differently: frame accuracy
    is dominated by long segments and is nearly blind to over-segmentation,
    while F1@IoU is blind to how much of the episode is actually right.
    """
    if step <= 0:
        raise ValueError("step must be positive")
    if not reference:
        return {"mof": 0.0, "n_samples": 0.0}
    start = float(reference[0]["start"])
    end = float(reference[-1]["end"])
    if end <= start:
        return {"mof": 0.0, "n_samples": 0.0}

    def label_at(spans: Sequence[Span], t: float) -> str | None:
        for span in spans:
            lo, hi = span_bounds(span)
            if lo <= t < hi:
                return str(span.get("text", ""))
        # No extrapolation. Falling back to the last span's label would let a
        # prediction that ends early -- or lies entirely outside the episode --
        # claim credit for every uncovered instant, so a prediction covering
        # nothing could score the same as one that is genuinely a third right.
        # Uncovered time is simply wrong, which is what it is.
        return None

    n = max(1, int((end - start) / step))
    hits = 0
    sample_step = min(step, end - start)
    for k in range(n):
        t = start + (k + 0.5) * sample_step
        left, right = label_at(predicted, t), label_at(reference, t)
        if left is not None and right is not None and matcher.equivalent(left, right):
            hits += 1
    return {"mof": hits / n, "n_samples": float(n)}


def dense_caption_score(
    predicted: Sequence[Span],
    reference: Sequence[Span],
    matcher: Matcher,
    *,
    thresholds: Sequence[float] = (0.3, 0.5, 0.7, 0.9),
) -> dict[str, float]:
    """Reference-averaged semantic recall at each temporal IoU threshold.

    Each ground-truth segment contributes its best eligible caption similarity,
    or zero when missed. Its denominator is fixed by ground truth, so deleting
    a prediction cannot improve recall. This is a recall diagnostic inspired by
    dense captioning, not the official Krishna/ActivityNet caption evaluator.
    Extra predictions are penalised by segmental F1 and label precision.
    """
    out: dict[str, float] = {}
    values: list[float] = []
    for threshold in thresholds:
        scores: list[float] = []
        for ref in reference:
            best = 0.0
            for pred in predicted:
                if iou(pred, ref) >= threshold:
                    best = max(best, matcher.similarity(
                        str(pred.get("text", "")), str(ref.get("text", ""))
                    ))
            scores.append(best)
        value = sum(scores) / len(scores) if scores else 0.0
        out[f"dvc@{int(threshold * 100)}"] = value
        values.append(value)
    out["dvc_mean"] = sum(values) / len(values) if values else 0.0
    return out


def coverage_weighted_label_accuracy(
    predicted: Sequence[Span], reference: Sequence[Span], matcher: Matcher
) -> float:
    """Duration-weighted fraction of reference time labelled equivalently.

    Complements ``mof`` by weighting each reference segment by its own length
    rather than by grid samples, which makes it exactly comparable across
    datasets with different episode durations.
    """
    total = sum(duration(span) for span in reference)
    if total <= 0.0:
        return 0.0
    score = 0.0
    for ref in reference:
        text = str(ref.get("text", ""))
        matched = 0.0
        lo, hi = span_bounds(ref)
        for pred in predicted:
            if not matcher.equivalent(str(pred.get("text", "")), text):
                continue
            plo, phi = span_bounds(pred)
            matched += max(0.0, min(hi, phi) - max(lo, plo))
        score += min(matched, duration(ref))
    return score / total


def all_joint_metrics(
    predicted: Sequence[Span], reference: Sequence[Span], matcher: Matcher
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for threshold in (0.10, 0.25, 0.50):
        out.update(f1_at_iou(predicted, reference, matcher, threshold=threshold))
    out["edit_score"] = edit_score(predicted, reference, matcher)
    out.update(mof(predicted, reference, matcher))
    out.update(dense_caption_score(predicted, reference, matcher))
    out["cw_label_accuracy"] = coverage_weighted_label_accuracy(predicted, reference, matcher)
    return out
