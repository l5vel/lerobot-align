"""Cross-episode consistency metrics.

The L5vel tasks are scripted: within a dataset, most episodes execute the same
sequence of steps, and several datasets have exactly one distinct human label
sequence across all 50 episodes. That has two consequences this module exists
to measure.

First, consistency is a real quality axis. A tool that describes the same
physical action as "pick up the red can" in one episode and "grab the drink"
in the next produces a dataset that is harder to train on, even when both
labels are individually defensible. The human annotations set the target: we
measure whether a tool reproduces the *degree* of consistency the humans had,
not merely whether it is self-consistent (a tool that emits one constant label
everywhere would be perfectly self-consistent and useless).

Second, scriptedness is a confound. Because the step sequence is so
predictable, a method with access to even a handful of labelled episodes from
the same dataset can do well without looking at the video at all. The
``script_predictability`` helpers quantify how much of a score is available
from the script prior alone, so the reported gains can be stated net of it.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from typing import Any

from .segmentation import Span
from .semantic import Matcher, normalise_label


def label_sequence(spans: Sequence[Span]) -> tuple[str, ...]:
    return tuple(normalise_label(str(s.get("text", ""))) for s in spans)


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    out = 0.0
    for value in counts.values():
        p = value / total
        out -= p * math.log(p) if p > 0 else 0.0
    return out


def sequence_diversity(episodes: Sequence[Sequence[Span]]) -> dict[str, float]:
    """How many distinct step-sequences a tool produced over a dataset.

    Compared against the same statistic computed on the human annotations.
    Both over- and under-shooting are errors: more distinct sequences than the
    humans means unstable wording or unstable segmentation, fewer means the
    tool is ignoring genuine episode-to-episode variation.
    """
    sequences = [label_sequence(spans) for spans in episodes]
    counts = Counter(sequences)
    n = len(sequences)
    return {
        "n_episodes": float(n),
        "n_distinct_sequences": float(len(counts)),
        "modal_sequence_fraction": (counts.most_common(1)[0][1] / n) if n else 0.0,
        "sequence_entropy": _entropy(counts),
        "normalised_sequence_entropy": (_entropy(counts) / math.log(n)) if n > 1 else 0.0,
    }


def vocabulary_stability(
    episodes: Sequence[Sequence[Span]], matcher: Matcher | None = None
) -> dict[str, float]:
    """Size and concentration of the label vocabulary a tool emits.

    ``distinct_labels_per_segment`` near 1 means the tool invents a fresh
    phrase for nearly every segment -- the classic failure of an open-ended
    captioner used as an annotator. When a ``matcher`` is supplied, labels are
    additionally collapsed into semantic clusters, so a tool is not punished
    for pure formatting variation.
    """
    labels = [normalise_label(str(s.get("text", ""))) for spans in episodes for s in spans]
    counts = Counter(labels)
    total = len(labels)
    out = {
        "n_segments": float(total),
        "n_distinct_labels": float(len(counts)),
        "distinct_labels_per_segment": (len(counts) / total) if total else 0.0,
        "label_entropy": _entropy(counts),
    }
    if matcher is not None and counts:
        clusters: list[str] = []
        for label in counts:
            if not any(matcher.equivalent(label, head) for head in clusters):
                clusters.append(label)
        out["n_semantic_clusters"] = float(len(clusters))
        out["semantic_clusters_per_segment"] = len(clusters) / total if total else 0.0
    return out


def consistency_gap(
    predicted_episodes: Sequence[Sequence[Span]],
    reference_episodes: Sequence[Sequence[Span]],
    matcher: Matcher | None = None,
) -> dict[str, float]:
    """Signed distance between a tool's consistency and the humans'.

    Zero is the target. Positive ``sequence_diversity_gap`` means the tool was
    less consistent than the humans, negative means more. Reporting the signed
    gap rather than the raw value keeps "collapsed to one constant label" from
    reading as a perfect score.
    """
    pred = sequence_diversity(predicted_episodes)
    ref = sequence_diversity(reference_episodes)
    pred_vocab = vocabulary_stability(predicted_episodes, matcher)
    ref_vocab = vocabulary_stability(reference_episodes, matcher)
    return {
        "pred_n_distinct_sequences": pred["n_distinct_sequences"],
        "true_n_distinct_sequences": ref["n_distinct_sequences"],
        "sequence_diversity_gap": pred["n_distinct_sequences"] - ref["n_distinct_sequences"],
        "pred_modal_fraction": pred["modal_sequence_fraction"],
        "true_modal_fraction": ref["modal_sequence_fraction"],
        "modal_fraction_gap": pred["modal_sequence_fraction"] - ref["modal_sequence_fraction"],
        "pred_n_distinct_labels": pred_vocab["n_distinct_labels"],
        "true_n_distinct_labels": ref_vocab["n_distinct_labels"],
        "vocabulary_gap": pred_vocab["n_distinct_labels"] - ref_vocab["n_distinct_labels"],
        "pred_entropy": pred["sequence_entropy"],
        "true_entropy": ref["sequence_entropy"],
        "entropy_gap": pred["sequence_entropy"] - ref["sequence_entropy"],
    }


def segment_count_stability(episodes: Sequence[Sequence[Span]]) -> dict[str, float]:
    """Spread of the per-episode segment count."""
    counts = [float(len(spans)) for spans in episodes]
    if not counts:
        return {"mean_segments": 0.0, "sd_segments": 0.0, "cv_segments": 0.0}
    mean = sum(counts) / len(counts)
    variance = sum((c - mean) ** 2 for c in counts) / len(counts)
    sd = math.sqrt(variance)
    return {
        "mean_segments": mean,
        "sd_segments": sd,
        "cv_segments": (sd / mean) if mean else 0.0,
        "min_segments": min(counts),
        "max_segments": max(counts),
    }


def script_predictability(reference_episodes: Sequence[Sequence[Span]]) -> dict[str, Any]:
    """How much of this dataset is guessable without watching the video.

    Returns the modal human label sequence and the mean relative boundary
    positions for episodes that follow it. These feed the script-prior
    reference arms, which establish the floor that any video-reading method
    must clear to have demonstrated that it is reading the video at all.
    """
    sequences = [label_sequence(spans) for spans in reference_episodes]
    counts = Counter(sequences)
    if not counts:
        return {"modal_sequence": (), "modal_fraction": 0.0, "mean_relative_boundaries": []}
    modal, hits = counts.most_common(1)[0]
    fractions: list[list[float]] = []
    for spans in reference_episodes:
        if label_sequence(spans) != modal or len(spans) < 2:
            continue
        start = float(spans[0]["start"])
        end = float(spans[-1]["end"])
        span = end - start
        if span <= 0:
            continue
        fractions.append([(float(s["start"]) - start) / span for s in spans[1:]])
    if fractions:
        mean_relative = [sum(col) / len(col) for col in zip(*fractions, strict=True)]
        sd_relative = [
            math.sqrt(sum((v - m) ** 2 for v in col) / len(col))
            for col, m in zip(zip(*fractions, strict=True), mean_relative, strict=True)
        ]
    else:
        mean_relative, sd_relative = [], []
    return {
        "modal_sequence": list(modal),
        "modal_fraction": hits / len(sequences),
        "n_following_modal": float(len(fractions)),
        "mean_relative_boundaries": mean_relative,
        "sd_relative_boundaries": sd_relative,
    }
