#!/usr/bin/env python
"""The three model-free timing floors of the alignment study (plan 5.1).

Fixed-label alignment hands every arm the ordered label list `L` and asks only
WHEN each label happens. That makes a whole class of answers reachable without
a model at all: these tasks are scripted, so "put the boundaries where they
usually are" is a genuine competitor. Without that number on the table the
configuration deltas this study exists to measure are uninterpretable -- a
twelve-point win of native video over contact sheets is twelve points of
nothing if neither arm beats a template that never opened the video
(plan 1.3 and claim A5, contrasts 7 and 8).

Three floors, in increasing order of the supervision they consume:

``floor_uniform``           `|L|` equal parts of the episode.       labels_only
``floor_median_ratio``      seed median RELATIVE boundary           labels_plus_seed
                            positions.
``floor_cumulative_prior``  seed median SEGMENT LENGTHS,            labels_plus_seed
                            normalised to sum to 1, accumulated.

WHY NOT THE DP SOLVER WITH THE MODEL TERM DELETED. That construction looks
like the obvious strongest floor and is unusable, which took measuring to
establish (plan 5.1). Dropping the model term leaves `min sum|g_i - want_i|`
subject to `sum g_i = D`, whose optimum is a whole face of the feasible set
whenever the wanted segment fractions do not sum to `D` -- and they are
per-index MEDIANS across seed episodes, so they never do. Measured here on
`u850-fridge-drink-04-v30`, the medians sum to 1.009476 of the episode; feeding
that solver two opposite model answers still moved boundaries 0.70s apart at a
duration weight of 100,000. The surplus is free to sit anywhere and the tie
break silently lets the model choose, so a "priors-only DP" carries a
permanent, unlabelled dependence on the VLM. These floors normalise instead:
boundary `k` goes at `t0 + D * (sum_{i<k} f_i) / (sum_i f_i)`. No solver, no
tie to break, model-free by construction rather than by hope.

WHY THE ANCHOR IS THE EPISODE START AND NOT ZERO. ``evaluate_alignment``
measures every boundary error from ``ground_truth[0]["start"]`` and scores IoU
over ``[gt[0].start, gt[-1].end]``, while ``reference_arms._equal_split``
anchors its split at `t = 0`. Where the two differ the floor is displaced by a
constant on every boundary of the episode, and a constant displacement of the
floor reads as VLM superiority in contrasts 7 and 8. On Corpus A exactly one
episode is affected (`u850-bag-place-02-FV-v30` episode 24 starts at 0.563s),
so the size of the bug here is small -- but it is 0.563s against a `b_hit@1`
tolerance of one second, it is silent, and it is systematic on any corpus
whose annotations do not start at the first frame.

WHY THE SEED SET IS FILTERED BY LABEL LIST. Both seeded floors average
POSITIONAL boundaries: "the median position of boundary 2". On a component
whose seed episodes mix annotation conventions, boundary 2 is a different
physical event in different episodes and the average is of unrelated things.
The filter is the same one the calibration fit uses
(``fit_align_calibration.py``: exact ordered label tuple, most frequent wins,
ties broken lexicographically so hash order cannot decide it), which also
means the seeded floors and the calibrated VLM arms of contrast 7 consume the
same ten episodes through the same cohort rule. Its outcome -- how many
distinct lists, which episodes were kept, which dropped, whether the mode was
a tie -- is recorded per component, because on `u850-fridge-drink-03-v30` it
discards half the seed set.

WHY THE PRIOR IS NOT APPLIED TO EVERY EVAL EPISODE. Under `oracle_list` each
episode carries its own label list, and a seed-derived positional prior means
nothing for an episode annotated under a different convention. The floors
therefore mirror the tool's own ``applies_to`` rule: prior when the episode's
label tuple matches the cohort's, uniform placement otherwise. That makes the
seeded floors a blended treatment in exactly the way plan 4.2 describes for the
calibrated VLM arms, so ``prior_applied`` is written per episode -- it is the
floors' analogue of the study's ``calibration_applied``, and it is what lets
the report restrict contrast 7 to the subpopulation where the treatment is
uniform. It is deliberately NOT named ``calibration_applied``: no calibration
is involved and the metric-name contract belongs to the scorer.

PROVING THE FLOORS ARE MODEL-FREE (plan 7.4 stage B, gate 11). Claiming a
construction is model-free is worth less than testing it, so ``--model-answers``
loads a prediction file and passes each episode's spans into the arm builder,
which is precisely where the rejected DP floor would have consumed them.
``--self-test`` runs the whole pipeline three times -- no answers, all
boundaries at 2% of the episode, all at 98% -- and asserts every output file is
byte-identical. Passing that is evidence; not having a parameter is not.

Output is the record shape ``run_arm.py`` writes, so ``score_alignment.py``
cannot tell a floor from a VLM arm except by reading fields that say so.
Nothing is cached: these arms cost milliseconds, so a cache could only ever
serve a stale answer.

``--prior-scope segment_count`` is the task-budget protocol: split the task's
same at-most-ten seeds by segment count, fit each count from all its examples
regardless of wording, and freeze those templates before evaluation. Counts
with fewer than ``--min-fit-episodes`` usable seeds receive uniform timing.
The older ``exact`` and ``length`` modes retain their modal exact-label fit;
``length`` relaxes only application and must not be confused with this mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve()

Span = dict[str, Any]

# Supervision levels come from plan 4.1 and decide which contrasts are legal,
# so they are stated here rather than read from a config: a floor that silently
# changed level would let `aggregate.py` compare across levels, which is the
# exact mistake that withdrew the previous study's headline result (plan 1.2).
SUPERVISION = {
    "floor_uniform": "labels_only",
    "floor_median_ratio": "labels_plus_seed",
    "floor_cumulative_prior": "labels_plus_seed",
}


def sha256_file(path: Path) -> str:
    """Full digest, used for the byte-identity check of the model-freedom gate."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_digest(path: str | Path) -> str:
    """Short content hash of a supplied file, exactly as ``run_arm.py`` takes it.

    Same function, same truncation, so a report can compare the label file a
    floor used against the one a VLM arm used by string equality. Content, never
    path: label files are rewritten in place under the same name (defect D1).
    """
    file = Path(path)
    if not file.is_file():
        raise SystemExit(f"supplied file does not exist: {file}")
    return sha256_file(file)[:16]


def config_fingerprint(payload: dict[str, Any]) -> str:
    """Same digest ``run_arm.py`` uses, so both records are keyed comparably."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def episode_extent(spans: Sequence[Span]) -> tuple[float, float]:
    """Episode start and duration, read the way the scorer reads them.

    ``evaluate_alignment`` takes ``episode_start = ground_truth[0]["start"]``
    and ``episode_end = ground_truth[-1]["end"]``; the floors must anchor on the
    same two numbers or they are scored against a different episode than the one
    they placed boundaries in. This is the episode's extent, not its contents:
    it is the one quantity plan 4 gives to every arm, and it reveals no internal
    boundary.
    """
    start = float(spans[0]["start"])
    return start, float(spans[-1]["end"]) - start


@dataclass(frozen=True)
class SeedCohort:
    """Which seed episodes the positional averages may be taken over."""

    labels: tuple[str, ...]
    episodes: tuple[int, ...]
    dropped: tuple[int, ...]
    n_label_lists: int
    modal_count: int
    tie: bool
    runners_up: tuple[tuple[str, ...], ...]


def seed_cohort(seed_truth: dict[int, list[Span]]) -> SeedCohort | None:
    """The modal exact label tuple of the seed episodes, and who shares it.

    Deliberately identical to ``fit_align_calibration.py``'s fit-set selector:
    count exact ordered tuples, take the most frequent, break frequency ties on
    the tuple itself so that PYTHONHASHSEED and input ordering cannot move the
    cohort between runs. Upstream defect D7 was that an earlier selector scored
    by LENGTH over a set and let hash order decide; reproducing that would make
    the floors non-deterministic for the same reason.
    """
    if not seed_truth:
        return None
    lists = {episode: tuple(str(s["text"]) for s in spans) for episode, spans in seed_truth.items()}
    counts = Counter(lists.values())
    modal = min(counts, key=lambda labels: (-counts[labels], labels))
    tied = sorted(t for t, n in counts.items() if n == counts[modal] and t != modal)
    return SeedCohort(
        labels=modal,
        episodes=tuple(sorted(e for e, t in lists.items() if t == modal)),
        dropped=tuple(sorted(e for e, t in lists.items() if t != modal)),
        n_label_lists=len(counts),
        modal_count=counts[modal],
        tie=bool(tied),
        runners_up=tuple(tied),
    )


@dataclass(frozen=True)
class SeedPrior:
    """Positional timing template fitted to one component's seed cohort."""

    cohort: SeedCohort
    relative_boundaries: tuple[float, ...]
    segment_fractions: tuple[float, ...]
    fraction_sum: float
    rows: int
    skipped: tuple[int, ...] = ()

    @property
    def labels(self) -> tuple[str, ...]:
        return self.cohort.labels


def build_seed_prior(
    seed_truth: dict[int, list[Span]], *, cohort: SeedCohort | None = None
) -> SeedPrior | None:
    """Median relative boundaries and median segment fractions over the cohort.

    Medians, not means, and fractions of duration, not seconds -- both for the
    reasons ``fit_align_calibration.fit`` gives: one badly-timed episode should
    not move the template, and a constant second offset over-corrects short
    episodes while under-correcting long ones.

    The two vectors are computed independently rather than one from the other.
    They are not interchangeable: the median of the per-episode differences is
    not the difference of the per-index medians, which is the whole reason
    ``floor_median_ratio`` and ``floor_cumulative_prior`` are different arms and
    the reason the segment fractions need normalising at all.
    """
    cohort = cohort if cohort is not None else seed_cohort(seed_truth)
    if cohort is None or not cohort.episodes:
        return None
    relative: dict[int, list[float]] = {}
    segments: dict[int, list[float]] = {}
    used: list[int] = []
    skipped: list[int] = []
    for episode in cohort.episodes:
        spans = seed_truth[episode]
        if not spans:
            skipped.append(episode)
            continue
        start, duration = episode_extent(spans)
        if not math.isfinite(start) or not math.isfinite(duration) or duration <= 0:
            skipped.append(episode)
            continue
        used.append(episode)
        for index, span in enumerate(spans[1:], start=1):
            relative.setdefault(index, []).append((float(span["start"]) - start) / duration)
        # Segment i runs from its own start to the NEXT span's start, and the
        # last one to the episode end -- the same walk the fitter does, so a
        # gap between two annotated spans is charged to the earlier segment
        # instead of vanishing.
        cursor = start
        for index in range(len(spans)):
            end = (
                float(spans[index + 1]["start"])
                if index + 1 < len(spans)
                else float(spans[-1]["end"])
            )
            segments.setdefault(index, []).append((end - cursor) / duration)
            cursor = end
    if not used:
        return None
    boundaries = [statistics.median(relative[i]) for i in sorted(relative)]
    fractions = [statistics.median(segments[i]) for i in sorted(segments)]
    return SeedPrior(
        cohort=cohort,
        relative_boundaries=tuple(boundaries),
        segment_fractions=tuple(fractions),
        fraction_sum=float(sum(fractions)),
        rows=len(used),
        skipped=tuple(skipped),
    )


def _seed_timing_exclusion(spans: Sequence[Span]) -> str | None:
    """Reject malformed seed timelines before a count prior takes any medians."""
    try:
        starts = [float(span["start"]) for span in spans]
        ends = [float(span["end"]) for span in spans]
    except (KeyError, TypeError, ValueError):
        return "invalid_timestamp"
    if not all(math.isfinite(value) for value in (*starts, *ends)):
        return "nonfinite_timestamp"
    if any(end <= start for start, end in zip(starts, ends, strict=True)):
        return "nonpositive_span"
    if any(right <= left for left, right in zip(starts[:-1], starts[1:], strict=True)):
        return "nonincreasing_starts"
    if any(end > next_start for end, next_start in zip(ends[:-1], starts[1:], strict=True)):
        return "overlapping_spans"
    return None


def build_segment_count_priors(
    seed_truth: dict[int, list[Span]], *, min_fit_episodes: int = 3
) -> tuple[dict[int, SeedPrior], dict[str, dict[str, Any]]]:
    """Fit each count once, sharing one task's fixed seed budget across counts.

    Label wording is never a fit filter here. Boundary position is assumed to
    have a consistent meaning within this task and count, as in the explicit
    segment-count calibration mode. The returned audit includes rejected counts
    so unsupported evaluation episodes cannot appear to receive a learned prior.
    """
    if len(seed_truth) > 10:
        raise ValueError("segment-count priors may use at most 10 seed trajectories per task")
    if type(min_fit_episodes) is not int or not 1 <= min_fit_episodes <= 10:
        raise ValueError("min_fit_episodes must be an integer between 1 and 10")
    groups: dict[int, dict[int, list[Span]]] = {}
    for episode, spans in seed_truth.items():
        groups.setdefault(len(spans), {})[episode] = spans
    priors: dict[int, SeedPrior] = {}
    audit: dict[str, dict[str, Any]] = {}
    for count, group in sorted(groups.items()):
        label_counts = Counter(tuple(str(s["text"]) for s in spans) for spans in group.values())
        exclusions: dict[int, str] = {}
        for episode, spans in group.items():
            reason = "insufficient_segments" if count < 2 else _seed_timing_exclusion(spans)
            if reason is not None:
                exclusions[episode] = reason
        eligible = {episode: spans for episode, spans in group.items() if episode not in exclusions}
        cohort = SeedCohort(
            labels=(),
            episodes=tuple(sorted(eligible)),
            dropped=tuple(sorted(set(seed_truth) - set(group))),
            n_label_lists=len(label_counts),
            modal_count=max(label_counts.values()),
            tie=False,
            runners_up=(),
        )
        prior = build_seed_prior(eligible, cohort=cohort) if eligible else None
        usable = prior.rows if prior is not None else 0
        fitted = prior is not None and usable >= min_fit_episodes
        if fitted:
            priors[count] = prior
        skipped = set(exclusions) | (set(prior.skipped) if prior is not None else set(group))
        audit[str(count)] = {
            "label_scope": "segment_count",
            "n_segments": count,
            "labels": [],
            "seed_episodes": sorted(group),
            "fit_episodes": [episode for episode in cohort.episodes if episode not in skipped],
            "fit_episode_count": usable,
            "skipped_episodes": sorted(skipped),
            "excluded_episodes": [
                {"episode": episode, "reason": reason}
                for episode, reason in sorted(exclusions.items())
            ],
            "n_label_lists": cohort.n_label_lists,
            "min_fit_episodes": min_fit_episodes,
            "fitted": fitted,
            "reason": (
                "fitted" if fitted else "insufficient_segments" if count < 2
                else "insufficient_fit_episodes"
            ),
            "relative_boundaries": list(prior.relative_boundaries) if fitted else None,
            "segment_fractions": list(prior.segment_fractions) if fitted else None,
            "segment_fraction_sum": prior.fraction_sum if fitted else None,
            "normalisation": "segment fractions divided by their sum before accumulation",
        }
    return priors, audit


def prior_applies(prior: SeedPrior | None, labels: Sequence[str], scope: str) -> bool:
    """Whether this episode's label list is one the positional prior describes.

    ``exact`` mirrors the tool's ``_AlignCalibration.applies_to``, so a seeded
    floor and a calibrated VLM arm treat the same episodes as in-scope and
    contrast 7 stays like-for-like. ``length`` is the looser reading -- same
    number of boundaries, different wording -- and exists so the report can show
    the choice does not carry the result.
    """
    if prior is None:
        return False
    if scope == "exact":
        return tuple(str(label) for label in labels) == prior.labels
    if scope == "length":
        return len(labels) == len(prior.labels)
    if scope == "segment_count":
        return len(labels) == len(prior.segment_fractions)
    raise ValueError(f"unknown prior scope {scope!r}")


def _spans_from_fractions(
    labels: Sequence[str], start: float, duration: float, fractions: Sequence[float]
) -> list[Span]:
    """Build `|L|` consecutive spans from `|L|-1` internal boundary fractions.

    Clamped into `[0, 1]` and forced non-decreasing. Per-index medians of
    monotone rows are themselves monotone, so this should never bite; it is here
    because if it ever does, an out-of-order floor would be scored as a wrong
    answer rather than raising, and the study would carry a silent defect.

    Carries ``index`` as well as ``text`` because the scorer matches on the
    returned index, not on the label text (plan 6.1, defect D10): 16.4% of
    Corpus A episodes repeat a label within the episode. A floor always places
    every label, so for these arms the two matchings agree -- writing the index
    anyway means the scorer needs no special case for them.
    """
    monotone: list[float] = []
    previous = 0.0
    for value in fractions:
        previous = min(1.0, max(previous, float(value)))
        monotone.append(previous)
    cuts = [start] + [start + duration * f for f in monotone] + [start + duration]
    return [
        {"start": cuts[i], "end": cuts[i + 1], "text": str(labels[i]), "index": i}
        for i in range(len(labels))
    ]


def floor_uniform(
    labels: Sequence[str],
    *,
    start: float,
    duration: float,
    prior: SeedPrior | None = None,  # noqa: ARG001 - level is labels_only by design
    model_answer: list[Span] | None = None,  # noqa: ARG001 - see module docstring
) -> list[Span]:
    """`|L|` equal parts, anchored at the episode start."""
    count = max(1, len(labels))
    return _spans_from_fractions(labels, start, duration, [i / count for i in range(1, count)])


def floor_median_ratio(
    labels: Sequence[str],
    *,
    start: float,
    duration: float,
    prior: SeedPrior | None,
    model_answer: list[Span] | None = None,  # noqa: ARG001 - see module docstring
) -> list[Span]:
    """Boundary `k` at the seed cohort's median relative position for `k`.

    Falls back to uniform when the caller has already judged the prior
    inapplicable (``prior=None``) or when its width does not match `|L|`. Falling
    back rather than dropping the episode keeps the population at all 637 eval
    episodes, which is the coverage complaint of plan 2.1b.
    """
    if prior is None or len(prior.relative_boundaries) != len(labels) - 1:
        return floor_uniform(labels, start=start, duration=duration)
    return _spans_from_fractions(labels, start, duration, prior.relative_boundaries)


def floor_cumulative_prior(
    labels: Sequence[str],
    *,
    start: float,
    duration: float,
    prior: SeedPrior | None,
    model_answer: list[Span] | None = None,  # noqa: ARG001 - see module docstring
) -> list[Span]:
    """Median segment lengths, normalised to sum to 1, then accumulated.

    The normalisation is the entire point (module docstring): per episode the
    segment fractions sum to exactly 1, but the per-index medians do not, and
    dividing by their sum is what removes the free surplus that made the DP
    formulation quietly model-dependent.
    """
    if prior is None or len(prior.segment_fractions) != len(labels) or prior.fraction_sum <= 0:
        return floor_uniform(labels, start=start, duration=duration)
    cumulative = 0.0
    fractions: list[float] = []
    for value in prior.segment_fractions[:-1]:
        cumulative += value
        fractions.append(cumulative / prior.fraction_sum)
    return _spans_from_fractions(labels, start, duration, fractions)


ARMS = {
    "floor_uniform": floor_uniform,
    "floor_median_ratio": floor_median_ratio,
    "floor_cumulative_prior": floor_cumulative_prior,
}


def _coerce_label_list(value: Any, where: str) -> list[str]:
    """The tool's own validation: a list of non-empty strings, or an error."""
    if not isinstance(value, list):
        raise SystemExit(f"{where}: expected a list of subtask strings, got {type(value).__name__}")
    labels = [item.strip() for item in value if isinstance(item, str)]
    if len(labels) != len(value) or not all(labels):
        raise SystemExit(f"{where}: every subtask must be a non-empty string")
    return labels


def load_label_lists(path: Path) -> tuple[list[str], dict[int, list[str]]]:
    """Read a supplied-label file with the tool's own semantics.

    Mirrors ``plan_subtasks_memory._load_subtasks_file``: a flat list applies to
    every episode, an object is keyed by episode index with an optional
    ``default``. Reading it the same way matters because the floors must receive
    the identical `L` the VLM arms receive; a divergence here would make the
    contrast a comparison of label files. Malformed entries are refused for the
    same reason the tool refuses them -- coercing a non-string into a label here
    would let a file that ABORTS the VLM arm quietly produce a floor.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return _coerce_label_list(payload, str(path)), {}
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON list or object, got {type(payload).__name__}")
    default: list[str] = []
    by_episode: dict[int, list[str]] = {}
    for key, value in payload.items():
        if key == "default":
            default = _coerce_label_list(value, f"{path}['default']")
            continue
        try:
            by_episode[int(key)] = _coerce_label_list(value, f"{path}[{key!r}]")
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{path}: key {key!r} is neither 'default' nor an episode index"
            ) from exc
    if not default and not by_episode:
        raise SystemExit(f"{path}: no subtasks found")
    return default, by_episode


def load_model_answers(path: Path) -> dict[int, list[Span]]:
    """Load a prediction file for the model-freedom probe (plan 7.4 gate 11).

    Parsed and handed to the arm builders for real. The point of the gate is
    that a full, well-formed model answer reaches the place a DP floor would
    have used it and changes nothing.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes") if isinstance(payload, dict) else None
    if not isinstance(episodes, dict):
        raise SystemExit(f"{path}: expected a run_arm.py-shaped record with an 'episodes' object")
    return {int(key): list(value or []) for key, value in episodes.items()}


def synthetic_model_answer(
    dataset: str, truth: dict[int, list[Span]], episodes: Sequence[int], position: float
) -> dict[str, Any]:
    """Two opposite fake model answers: every internal boundary at `position`.

    These are the same probes plan 5.1 used to expose the DP floor -- all
    boundaries at 2% of the episode versus all at 98% -- because a construction
    that is only accidentally model-free tends to survive small perturbations
    and fail on the extremes.
    """
    out: dict[str, list[Span]] = {}
    for episode in episodes:
        spans = truth.get(episode)
        if not spans:
            continue
        start, duration = episode_extent(spans)
        cut = start + duration * position
        placed: list[Span] = []
        for index in range(len(spans)):
            placed.append(
                {
                    "start": start if index == 0 else cut,
                    "end": cut if index == 0 else start + duration,
                    "text": str(spans[index]["text"]),
                    "index": index,
                }
            )
        out[str(episode)] = placed
    return {"dataset": dataset, "arm": f"synthetic_at_{position}", "episodes": out}


Job = tuple[Path, Path, Path | None]


def dataset_jobs(gt: Path, splits: Path, labels: Path | None, label_source: str,
                 datasets: Sequence[str] | None = None) -> list[Job]:
    """Pair up (ground truth, split, labels) for one component or for all of them.

    Directories are accepted because the floors cost no GPU: running all 16
    components in one process is what makes the stage-B gates (11 and 12) a
    single check over the whole corpus instead of sixteen partial ones.

    A label directory is the one ``make_label_files.py`` writes, whose files are
    named ``<dataset>__<source>.json`` -- the source is part of the name because
    a component has both an oracle and a modal list, and picking the wrong one
    would silently change the task rather than fail.
    """
    truth_files = sorted(gt.glob("*.json")) if gt.is_dir() else [gt]
    if not truth_files:
        raise SystemExit(f"no ground truth JSON under {gt}")
    # `--datasets` names components, exactly as `make_label_files.py --datasets`
    # does: every corpus-walking script in this study takes the same flag with the
    # same meaning, so a subset run is described once and cannot mean two things.
    # An unknown name is fatal rather than an empty result -- a typo that quietly
    # scores nothing is how a partial sweep gets reported as a whole one.
    if datasets:
        wanted = list(dict.fromkeys(datasets))
        by_stem = {path.stem: path for path in truth_files}
        unknown = [name for name in wanted if name not in by_stem]
        if unknown:
            raise SystemExit(
                f"--datasets names component(s) with no ground truth under {gt}: "
                f"{unknown}. Known: {sorted(by_stem)}"
            )
        truth_files = [by_stem[name] for name in wanted]
    jobs: list[Job] = []
    for truth_file in truth_files:
        split = splits / truth_file.name if splits.is_dir() else splits
        if not split.exists():
            raise SystemExit(f"no split file for {truth_file.name} at {split}")
        label_file: Path | None = None
        if labels is not None:
            if labels.is_dir():
                candidates = [
                    labels / f"{truth_file.stem}__{label_source}.json",
                    labels / truth_file.name,
                ]
                label_file = next((c for c in candidates if c.exists()), None)
                if label_file is None:
                    raise SystemExit(
                        f"no label file for {truth_file.stem}: tried "
                        f"{[str(c) for c in candidates]}"
                    )
            else:
                label_file = labels
                if not label_file.exists():
                    raise SystemExit(f"no label file at {label_file}")
        jobs.append((truth_file, split, label_file))
    return jobs


def resolve_labels(
    truth: dict[int, list[Span]],
    episodes: Sequence[int],
    label_file: Path | None,
) -> tuple[dict[int, list[str]], str]:
    """`L` per eval episode, plus which route it came by.

    With no file this reproduces `oracle_list` (plan 4.2) exactly: each episode
    keeps its own label list and loses only its boundaries. That is what the
    label file will contain, but deriving it here means the floors can be run --
    and gated -- before ``make_label_files.py`` has written anything. Verified:
    over all 16 components the two routes produce identical predictions.

    The route is recorded as ``label_route`` rather than ``label_source``, which
    is a row field ``score_alignment.py`` computes for itself with a different
    vocabulary; two fields of the same name meaning different things is how a
    table gets misread.

    An episode that resolves to no labels aborts rather than being skipped. The
    tool's own loader takes the same line, and for a good reason: on the VLM
    path an unresolved episode falls through to free generation (defect D3), so
    a quietly empty entry is the failure this whole mode exists to avoid.
    """
    if label_file is None:
        oracle = {episode: [str(s["text"]) for s in truth[episode]] for episode in episodes}
        return oracle, "ground_truth"
    default, by_episode = load_label_lists(label_file)
    resolved: dict[int, list[str]] = {}
    missing: list[int] = []
    for episode in episodes:
        labels = by_episode.get(episode) or default
        if not labels:
            missing.append(episode)
            continue
        resolved[episode] = labels
    if missing:
        raise SystemExit(f"{label_file}: no label list for eval episode(s) {missing}")
    return resolved, "supplied_file"


def run_dataset(
    *,
    truth_path: Path,
    split_path: Path,
    label_path: Path | None,
    out_dir: Path,
    arms: Sequence[str],
    prior_scope: str,
    model_answers: dict[int, list[Span]] | None,
    model_answers_path: Path | None,
    quiet: bool = False,
    min_fit_episodes: int = 3,
) -> list[dict[str, Any]]:
    """Fit the seed template for one component and write one record per arm.

    The seed prior is fitted once and shared by both seeded arms, so they can
    never disagree about which episodes the cohort contained. Every arm is
    written even when its prior turns out to apply nowhere: an arm missing from
    the results directory is indistinguishable from an arm that crashed.
    """
    truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
    dataset = truth_payload["dataset"]
    truth = {int(k): v for k, v in truth_payload["episodes"].items()}

    split = json.loads(split_path.read_text(encoding="utf-8"))
    seed_ids = [int(e) for e in split["seed"]]
    eval_ids = [int(e) for e in split["eval"]]
    if len(set(seed_ids)) > 10:
        raise SystemExit(f"{dataset}: at most 10 seed trajectories per task are allowed")
    overlap = set(seed_ids) & set(eval_ids)
    if overlap:
        raise SystemExit(f"{dataset}: seed and eval episodes overlap: {sorted(overlap)}")

    seed_truth = {e: truth[e] for e in seed_ids if e in truth}
    if not seed_truth and prior_scope != "segment_count":
        raise SystemExit(f"{dataset}: no seed episodes carry ground truth")
    # An eval episode with no human annotation cannot be scored by anyone, so
    # it gets no prediction -- but `n_episodes_requested` keeps the full split
    # size beside `n_episodes_predicted`, so the shortfall is visible.
    scored = [e for e in eval_ids if e in truth]
    priors_by_count: dict[int, SeedPrior] = {}
    count_prior_records: dict[str, dict[str, Any]] = {}
    if prior_scope == "segment_count":
        priors_by_count, count_prior_records = build_segment_count_priors(
            seed_truth, min_fit_episodes=min_fit_episodes
        )
        prior = None
    elif prior_scope in {"exact", "length"}:
        prior = build_seed_prior(seed_truth)
        if prior is None:
            raise SystemExit(f"{dataset}: could not fit a seed prior from {len(seed_truth)} episode(s)")
    else:
        raise SystemExit(f"unknown prior scope {prior_scope!r}")

    labels_by_episode, label_route = resolve_labels(truth, scored, label_path)
    # Not fatal here: gate 7.4.1 owns the label/GT agreement check and this
    # script must not become a second, weaker copy of it. But the scorer raises
    # on a length mismatch, so a silent one would surface as an unexplained hole
    # in the results table hours later.
    mismatched = [e for e in scored if len(labels_by_episode[e]) != len(truth[e])]
    if mismatched and not quiet:
        print(
            f"[floors] WARNING {dataset}: {len(mismatched)} episode(s) whose supplied label "
            f"count differs from ground truth; these cannot be scored: {mismatched[:8]}",
            file=sys.stderr,
        )

    seed_prior_record = {
        "labels": list(prior.labels),
        "fit_episodes": list(prior.cohort.episodes),
        "fit_episode_count": prior.rows,
        "dropped_episodes": list(prior.cohort.dropped),
        # A cohort episode with no usable extent: in the cohort by label, out of
        # the medians. Empty on Corpus A, recorded so it cannot become silent.
        "skipped_episodes": list(prior.skipped),
        "n_label_lists": prior.cohort.n_label_lists,
        "modal_count": prior.cohort.modal_count,
        "modal_tie": prior.cohort.tie,
        "modal_runners_up": [list(t) for t in prior.cohort.runners_up],
        "relative_boundaries": list(prior.relative_boundaries),
        "segment_fractions": list(prior.segment_fractions),
        # Gate 7.4.12: the sum is recorded and the normalisation that divides it
        # out is named, so a reader can see the DP degeneracy this floor avoids.
        "segment_fraction_sum": prior.fraction_sum,
        "normalisation": "segment fractions divided by their sum before accumulation",
    } if prior is not None else None

    label_sha = content_digest(label_path) if label_path is not None else None
    records: list[dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for arm in arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; known: {sorted(ARMS)}")
        started = time.time()
        seeded = SUPERVISION[arm] == "labels_plus_seed"
        predictions: dict[str, list[Span]] = {}
        applied: dict[str, bool] = {}
        applied_by_count: dict[str, dict[str, int]] = {}
        for episode in scored:
            labels = labels_by_episode[episode]
            start, duration = episode_extent(truth[episode])
            selected_prior = priors_by_count.get(len(labels)) if prior_scope == "segment_count" else prior
            in_scope = seeded and prior_applies(selected_prior, labels, prior_scope)
            if prior_scope == "segment_count" and in_scope and arm == "floor_cumulative_prior":
                assert selected_prior is not None
                # The arm falls back to uniform for a degenerate median prior.
                # Report that fallback, not merely count-based eligibility.
                in_scope = math.isfinite(selected_prior.fraction_sum) and selected_prior.fraction_sum > 0
            applied[str(episode)] = in_scope
            coverage = applied_by_count.setdefault(str(len(labels)), {"episodes": 0, "applied": 0})
            coverage["episodes"] += 1
            coverage["applied"] += int(in_scope)
            predictions[str(episode)] = ARMS[arm](
                labels,
                start=start,
                duration=duration,
                prior=selected_prior if in_scope else None,
                model_answer=(model_answers or {}).get(episode),
            )
        elapsed = time.time() - started

        provenance: dict[str, Any] = {
            "dataset": dataset,
            "arm": arm,
            "tool": "reference",
            "arm_family": "floor",
            "uses_video": False,
            "equalised": True,
            "flags": {"prior_scope": prior_scope if seeded else None, "anchor": "episode_start"},
            "model_id": None,
            "supervision": SUPERVISION[arm],
            # PROVENANCE_KEYS that `score_runs.py` copies onto every row. Null
            # here is the honest value and is the point of a floor: no
            # annotation code ran, so no annotation source can have shaped this
            # output. What did shape it is hashed instead.
            "align_source": None,
            "upstream_source": None,
            "effective_prompts": None,
            "prompt_overrides_active": None,
            "floor_source_sha": content_digest(HERE),
            "repeat": 0,
            "seed_episodes": seed_ids,
            "episodes": sorted(scored),
            "calibration_sha": None,
            "subtasks_path": str(label_path) if label_path is not None else None,
            # Content hash, never the path (defect D1): label files are
            # regenerated routinely here, and a path-keyed record cannot tell a
            # rebuilt label file from the one it actually used.
            "subtasks_sha": label_sha,
            "label_route": label_route,
            "seed_prior": seed_prior_record,
        }
        if prior_scope == "segment_count":
            provenance["flags"]["min_fit_episodes"] = min_fit_episodes
            provenance["seed_priors_by_segment_count"] = count_prior_records
        record = {
            **provenance,
            "config_fingerprint": config_fingerprint(provenance),
            "ok": True,
            "returncode": 0,
            "timed_out": False,
            "error": None,
            "elapsed_seconds": elapsed,
            # Measured, not unknown: a floor issues no request, so zero is the
            # true cost. `run_arm.py` reserves None for an unreadable counter.
            "prompt_tokens": 0,
            "generation_tokens": 0,
            "total_tokens": 0,
            "n_episodes_requested": len(eval_ids),
            "n_episodes_predicted": len(predictions),
            "n_label_count_mismatch": len(mismatched),
            # The floors' analogue of the calibration-applied rate (plan 4.2):
            # which episodes actually received the seed template, and which fell
            # back to a uniform split. Absent for `labels_only`, where there is
            # no seed treatment to have applied. Deliberately not written as
            # `calibration_applied`: no calibration exists on this path, and the
            # scorer derives that metric as 0 for an arm with no calibration
            # file, which is the truth about a floor.
            "prior_applied": applied if seeded else None,
            "prior_applied_fraction": (
                sum(applied.values()) / len(applied) if seeded and applied else None
            ),
            "model_answers_probe": str(model_answers_path) if model_answers_path else None,
            "command": [sys.executable, str(HERE), *sys.argv[1:]],
            "log": None,
            "episodes": predictions,
        }
        if prior_scope == "segment_count":
            record["prior_coverage_by_segment_count"] = applied_by_count if seeded else None
        path = out_dir / f"{dataset}__{arm}.json"
        path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        records.append(record)
        if not quiet:
            rate = record["prior_applied_fraction"]
            print(
                f"[floors] {dataset} {arm}: {len(predictions)} episodes"
                + (f" prior_applied={rate:.3f}" if rate is not None else "")
                + f" -> {path}"
            )
    if not quiet and prior_scope == "segment_count":
        print(
            f"[floors] {dataset}: fitted {len(priors_by_count)} segment-count prior(s) "
            f"from {len(seed_truth)}/{len(set(seed_ids))} available shared seed trajectories; "
            f"minimum {min_fit_episodes} usable seeds per count"
        )
    elif not quiet:
        assert prior is not None
        cohort = prior.cohort
        print(
            f"[floors] {dataset} seed filter: {cohort.n_label_lists} label list(s), "
            f"kept {len(cohort.episodes)}/{len(seed_truth)}"
            + (f", dropped {list(cohort.dropped)}" if cohort.dropped else "")
            + (" (MODAL TIE broken lexicographically)" if cohort.tie else "")
            + f", sum(segment_fractions)={prior.fraction_sum:.6f}"
        )
    return records


# The two fields of a record that no model answer can reach and that no two
# runs can share: the wall clock, and the argv line that produced the file. The
# model-freedom gate compares records with these removed -- and then checks that
# they were the ONLY difference, so removing them cannot hide a real one.
VOLATILE_FIELDS = ("elapsed_seconds", "command")


def stable_digest(path: Path) -> tuple[str, dict[str, Any]]:
    """SHA-256 of a written record with the volatile fields taken out."""
    record = json.loads(path.read_text(encoding="utf-8"))
    stripped = {key: value for key, value in record.items() if key not in VOLATILE_FIELDS}
    digest = hashlib.sha256(json.dumps(stripped, sort_keys=True).encode("utf-8")).hexdigest()
    return digest, record


def _changed_keys(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return sorted(
        key for key in set(left) | set(right) if left.get(key, ...) != right.get(key, ...)
    )


def self_test(
    jobs: Sequence[Job], *, arms: Sequence[str], prior_scope: str, min_fit_episodes: int = 3
) -> int:
    """Gate 7.4.11: opposite model answers must produce identical output.

    Runs every component three times -- no model answer, one that puts every
    internal boundary at 2% of the episode, one that puts them all at 98% -- and
    compares what was written. A "priors-only DP" fails this by 0.70s per
    boundary at any duration weight (plan 5.1); these arms must not fail it at
    all.

    The comparison is over the whole record minus ``VOLATILE_FIELDS``, plus an
    explicit audit that those two were the only keys that moved. Comparing raw
    bytes would fail on the wall clock and prove nothing about the model answer;
    dropping keys without auditing them would be a gate that cannot fail.
    """
    import tempfile

    variants = (("none", None), ("at_0.02", 0.02), ("at_0.98", 0.98))
    with tempfile.TemporaryDirectory(prefix="align_floors_selftest_") as workspace:
        work = Path(workspace)
        digests: dict[str, dict[str, str]] = {}
        records: dict[str, dict[str, dict[str, Any]]] = {}
        for label, position in variants:
            variant = work / label
            for truth_path, split_path, label_path in jobs:
                truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
                dataset = truth_payload["dataset"]
                truth = {int(k): v for k, v in truth_payload["episodes"].items()}
                split = json.loads(split_path.read_text(encoding="utf-8"))
                answers = None
                if position is not None:
                    probe = work / f"{label}__{dataset}__probe.json"
                    probe.write_text(
                        json.dumps(
                            synthetic_model_answer(
                                dataset, truth, [int(e) for e in split["eval"]], position
                            ),
                            indent=1,
                        ),
                        encoding="utf-8",
                    )
                    answers = load_model_answers(probe)
                run_dataset(
                    truth_path=truth_path,
                    split_path=split_path,
                    label_path=label_path,
                    out_dir=variant,
                    arms=arms,
                    prior_scope=prior_scope,
                    min_fit_episodes=min_fit_episodes,
                    model_answers=answers,
                    # The probe path is deliberately NOT recorded in the compared
                    # records: the gate would then fail on a filename rather than
                    # on a boundary, which is not what it is asking.
                    model_answers_path=None,
                    quiet=True,
                )
            pairs = {
                path.name: stable_digest(path)
                for path in sorted(variant.glob("*__floor_*.json"))
            }
            digests[label] = {name: digest for name, (digest, _) in pairs.items()}
            records[label] = {name: record for name, (_, record) in pairs.items()}

        names = sorted(digests["none"])
        if not names:
            raise SystemExit("[floors] self-test produced no output files")
        failures: dict[str, list[str]] = {}
        volatile_seen: set[str] = set()
        for name in names:
            for label, _ in variants[1:]:
                if digests[label].get(name) != digests["none"][name]:
                    failures.setdefault(name, []).append(label)
                moved = _changed_keys(records["none"][name], records[label][name])
                volatile_seen.update(moved)
                unexpected = [key for key in moved if key not in VOLATILE_FIELDS]
                if unexpected:
                    failures.setdefault(name, []).append(f"{label}:{','.join(unexpected)}")
        if failures:
            for name, detail in sorted(failures.items()):
                print(f"[floors] self-test {name}: DIFFERS under {detail}", file=sys.stderr)
            print(
                f"[floors] MODEL-FREEDOM GATE FAILED on {len(failures)} of {len(names)} file(s): "
                "the placement depends on the model answer",
                file=sys.stderr,
            )
            return 1
    print(
        f"[floors] model-freedom gate PASSED: {len(names)} record(s) identical with no model "
        "answer, with every boundary at 2% and with every boundary at 98%; the only fields that "
        f"moved between runs were {sorted(volatile_seen) or ['none']}"
    )
    return 0


def declared_arms(paths: Sequence[Path]) -> dict[str, tuple[Path, dict[str, Any]]]:
    """Collect every arm declaration the study's config files make.

    Two shapes, and the floors are now declared in BOTH. ``alignment_arms.yaml``
    is authoritative: the floors are ordinary ``arms:`` entries carrying
    ``tool: reference``, exactly as ``arms.yaml`` declares the generation
    study's four model-free arms that ``reference_arms.py`` rather than
    ``run_arm.py`` produces. ``tool: reference`` is what keeps the launcher off
    them, so listing them beside the launched arms does not invite launching
    them. ``alignment_contrasts.yaml:external_arms`` is kept as the redundant
    contract it describes itself as; ``aggregate.py`` refuses to run if the two
    disagree on ``tool`` or ``supervision``.

    Reading both matters for ``floor_median_ratio``, which no §8 contrast
    references and which therefore has no row in a file keyed by contrast --
    before it was declared in the arm table it was scored and reported with its
    supervision level recorded nowhere.
    """
    try:
        import yaml
    except ImportError:  # the cross-check is a safeguard, never a prerequisite
        print("[floors] arm cross-check skipped: pyyaml not importable", file=sys.stderr)
        return {}
    found: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in paths:
        if not path.exists():
            print(f"[floors] NOTE: no arm declarations at {path}", file=sys.stderr)
            continue
        spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for entry in spec.get("arms") or []:
            if isinstance(entry, dict) and "name" in entry:
                found[str(entry["name"])] = (path, entry)
        for name, entry in (spec.get("external_arms") or {}).items():
            if isinstance(entry, dict):
                found[str(name)] = (path, entry)
    return found


def check_declarations(paths: Sequence[Path], records: Sequence[dict[str, Any]]) -> None:
    """Reconcile what the floors emit against what the configs promise.

    ``alignment_contrasts.yaml`` states of its ``external_arms`` that "the
    floors' own emitted metadata is authoritative and must match". This is the
    matching, run at the moment the metadata is written rather than hours later
    in the aggregation. ``supervision`` is fatal because it decides which
    contrasts ``aggregate.py`` will permit -- comparing across levels is the
    error that withdrew the previous study's headline (plan 1.2) -- while the
    descriptive fields are reported and left to the config's owner.
    """
    declared = declared_arms(paths)
    if not declared:
        return
    for arm in sorted({record["arm"] for record in records}):
        record = next(r for r in records if r["arm"] == arm)
        if arm not in declared:
            print(f"[floors] NOTE: {arm} is declared in none of "
                  f"{[str(p.name) for p in paths]}", file=sys.stderr)
            continue
        path, entry = declared[arm]
        # `alignment_arms.yaml` names the field `supervision`; the contrast
        # table's `external_arms` block names it `level`, after plan 4.1's own
        # column heading. Same contract, so accept either and fail only on a
        # real disagreement.
        level = entry.get("supervision", entry.get("level"))
        if level is None:
            print(f"[floors] NOTE: {arm} in {path.name} states no supervision level; "
                  "contrast_policy fails closed on missing metadata", file=sys.stderr)
            continue
        if level != record["supervision"]:
            raise SystemExit(
                f"{path}: {arm} declares supervision {level!r} but plan 5.1 puts it at "
                f"{record['supervision']!r}; a wrong level silently changes which contrasts "
                "are legal"
            )
        for key in ("tool", "uses_video", "equalised"):
            if key in entry and entry[key] != record[key]:
                print(f"[floors] NOTE: {arm} {key}={entry[key]!r} in {path.name}, "
                      f"{record[key]!r} in the prediction record", file=sys.stderr)
        print(f"[floors] {arm}: supervision {record['supervision']!r} agrees with {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gt", type=Path, required=True,
                        help="ground truth JSON, or a directory of them (one per component)")
    parser.add_argument("--split", type=Path, required=True,
                        help="split JSON with seed/eval lists, or a directory named like --gt")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=None,
                        help="supplied-label JSON (or directory) as passed to the VLM arms via "
                             "--subtasks-path; defaults to the episode's own ground-truth list, "
                             "which is what the oracle list contains")
    parser.add_argument("--label-source", default="oracle",
                        help="which make_label_files.py source to take from a --labels directory; "
                             "'oracle' is the study (plan 4.2), 'modal' is out of scope")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="restrict to these components by name (same flag and meaning as "
                             "make_label_files.py --datasets); default is every component "
                             "with ground truth under --gt")
    parser.add_argument("--arms", nargs="*", default=sorted(ARMS))
    parser.add_argument("--prior-scope", choices=("exact", "length", "segment_count"), default="exact",
                        help="exact/length fit a modal exact-label prior; segment_count fits all "
                             "seeds within each count and applies the corresponding frozen prior")
    parser.add_argument("--min-fit-episodes", type=int, choices=range(1, 11), default=3,
                        help="minimum usable seeds per segment_count prior (default: 3); "
                             "all counts share at most 10 seeds per task")
    parser.add_argument("--model-answers", type=Path, default=None,
                        help="prediction file handed to the arm builders for the model-freedom "
                             "gate; the output must not depend on it")
    parser.add_argument("--self-test", action="store_true",
                        help="run gate 7.4.11 and exit: two opposite synthetic model answers must "
                             "produce identical records, the wall clock excepted and audited")
    parser.add_argument("--arm-tables", type=Path, nargs="*",
                        default=[
                            HERE.parent.parent / "configs" / "alignment_arms.yaml",
                            HERE.parent.parent / "configs" / "alignment_contrasts.yaml",
                        ],
                        help="config files to reconcile the emitted arm metadata against; the "
                             "floors are declared as tool: reference arms in the arm "
                             "table and cross-checked against external_arms in the "
                             "contrast table")
    parser.add_argument("--summary-out", type=Path, default=None,
                        help="write the per-component seed-filter and segment-fraction record "
                             "(gates 7.4.11-12) here; kept out of --out-dir, which is globbed")
    args = parser.parse_args()

    jobs = dataset_jobs(args.gt, args.split, args.labels, args.label_source, args.datasets)
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; known: {sorted(ARMS)}")

    if args.self_test:
        return self_test(
            jobs, arms=args.arms, prior_scope=args.prior_scope,
            min_fit_episodes=args.min_fit_episodes,
        )

    answers = load_model_answers(args.model_answers) if args.model_answers else None
    if answers is not None:
        print(f"[floors] loaded {len(answers)} model answer(s) from {args.model_answers}; "
              "these arms must ignore them (gate 7.4.11)")

    records: list[dict[str, Any]] = []
    for truth_path, split_path, label_path in jobs:
        records.extend(
            run_dataset(
                truth_path=truth_path,
                split_path=split_path,
                label_path=label_path,
                out_dir=args.out_dir,
                arms=args.arms,
                prior_scope=args.prior_scope,
                min_fit_episodes=args.min_fit_episodes,
                model_answers=answers,
                model_answers_path=args.model_answers,
            )
        )

    check_declarations(args.arm_tables, records)

    if args.summary_out:
        summary = {
            "prior_scope": args.prior_scope,
            "anchor": "episode_start",
            "floor_source_sha": content_digest(HERE),
            "components": [
                {
                    "dataset": record["dataset"],
                    "arm": record["arm"],
                    "supervision": record["supervision"],
                    "n_episodes_predicted": record["n_episodes_predicted"],
                    "prior_applied_fraction": record["prior_applied_fraction"],
                    "seed_prior": record["seed_prior"],
                    **({
                        "seed_priors_by_segment_count": record["seed_priors_by_segment_count"],
                        "prior_coverage_by_segment_count": record["prior_coverage_by_segment_count"],
                    } if args.prior_scope == "segment_count" else {}),
                }
                for record in records
            ],
        }
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"[floors] summary -> {args.summary_out}")

    print(f"[floors] wrote {len(records)} prediction file(s) under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
