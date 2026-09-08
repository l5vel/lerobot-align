"""Per-episode scoring driver.

Produces one flat, JSON-serialisable record per (episode, arm) so that every
downstream aggregation, table and plot reads the same primitive. Nothing here
decides anything: it runs every metric and records every value, leaving the
choice of headline number to the analysis stage. That ordering matters --
picking the metric after seeing the numbers is how evaluations get rigged.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .consistency import label_sequence
from .joint import all_joint_metrics
from .segmentation import (
    Span,
    boundary_distance_summary,
    boundary_score,
    matched_iou,
    segmentation_counts,
    segmentation_covering,
    validate_segmentation,
)
from .semantic import Matcher, label_agreement

BOUNDARY_TOLERANCES: tuple[float, ...] = (0.5, 1.0, 2.0, 3.0, 5.0)


@dataclass
class EpisodeScore:
    dataset: str
    episode: int
    arm: str
    matcher: str
    ok: bool
    values: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def flat(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "episode": self.episode,
            "arm": self.arm,
            "matcher": self.matcher,
            "ok": self.ok,
            "error": self.error,
            **self.values,
        }


def score_episode(
    *,
    dataset: str,
    episode: int,
    arm: str,
    predicted: Sequence[Span],
    reference: Sequence[Span],
    matcher: Matcher,
    strict: bool = True,
) -> EpisodeScore:
    """Run the full metric suite for one episode under one arm.

    A tool that produced nothing usable for this episode yields ``ok=False``
    with the reason recorded. Those rows are carried through aggregation and
    reported as a failure rate rather than dropped, because an arm that
    silently skips its hard episodes would otherwise look better than one that
    attempts them.
    """
    try:
        validate_segmentation(reference, name="reference")
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        return EpisodeScore(dataset, episode, arm, matcher.name, False, error=f"bad reference: {exc}")
    if not predicted:
        return EpisodeScore(dataset, episode, arm, matcher.name, False, error="no prediction")
    try:
        validate_segmentation(predicted, name="predicted")
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
        if strict or not isinstance(exc, ValueError):
            return EpisodeScore(
                dataset, episode, arm, matcher.name, False, error=f"bad prediction: {exc}"
            )

    values: dict[str, Any] = {}

    for tolerance in BOUNDARY_TOLERANCES:
        score = boundary_score(predicted, reference, tolerance=tolerance)
        tag = f"{tolerance:g}".replace(".", "p")
        values[f"boundary_f1@{tag}"] = score.f1
        values[f"boundary_precision@{tag}"] = score.precision
        values[f"boundary_recall@{tag}"] = score.recall
    values.update(boundary_distance_summary(predicted, reference))

    values["covering_gt_by_pred"] = segmentation_covering(reference, predicted)
    values["covering_pred_by_gt"] = segmentation_covering(predicted, reference)
    values["covering_harmonic"] = (
        2 * values["covering_gt_by_pred"] * values["covering_pred_by_gt"]
        / (values["covering_gt_by_pred"] + values["covering_pred_by_gt"])
        if (values["covering_gt_by_pred"] + values["covering_pred_by_gt"]) > 0
        else 0.0
    )

    overlap = matched_iou(predicted, reference)
    values["macro_iou"] = overlap["macro_iou"]
    values["n_matched_spans"] = float(overlap["matched"])
    values.update(segmentation_counts(predicted, reference))

    values.update(all_joint_metrics(predicted, reference, matcher))
    values.update(
        label_agreement(
            [str(s.get("text", "")) for s in predicted],
            [str(s.get("text", "")) for s in reference],
            matcher,
        )
    )

    values["pred_sequence"] = list(label_sequence(predicted))
    values["true_sequence"] = list(label_sequence(reference))
    values["episode_duration"] = float(reference[-1]["end"]) - float(reference[0]["start"])

    return EpisodeScore(dataset, episode, arm, matcher.name, True, values)
