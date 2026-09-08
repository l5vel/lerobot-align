#!/usr/bin/env python
"""Score fixed-label alignment predictions, one JSONL row per (dataset, episode, arm).

This is Study 2's scorer (`alignment_plan.md`). It **wraps**
``lerobot_align.alignment_metrics.evaluate_alignment`` rather than
reimplementing it -- the arithmetic that produces temporal IoU, hit rates and
the missing-boundary penalty stays the tool's own, so a change there cannot
silently diverge from what this study reports. What the wrapper changes is the
two things that metric gets wrong *for this study*, neither of which can be
fixed in the tool (the annotation code is not modified to improve evaluation
numbers).

**Defect D10 -- repeated labels lose their occurrence identity.**
``_ordered_span_matches`` re-derives the label-to-span correspondence by exact
text with a monotone cursor, discarding the index the aligner actually
returned. With ``L = [A, B, A]`` and the third span dropped, the surviving
``A`` is attributed to occurrence 0 no matter where it sits in time. Measured
on Corpus A this is live on **131 of 797 episodes (16.4%)**, 126 of them the
whole ``clean-table`` family -- and it is harmless only while every label is
placed, which ``_align_spans_in_order`` breaks by design.

The fix here is a *preprocessing* step, not a reimplementation: labels are
re-keyed to ``"<index>\\x1f<text>"``, which is unique by construction, and each
predicted span is stamped with the key of the label index it was resolved to.
Text matching over unique labels then reproduces index matching exactly, and
``evaluate_alignment`` runs untouched underneath. On the duplicate-free subset
the two are provably and (``--self-test``) numerically identical, which is what
makes this safe.

Indices come from, in order of authority:
  1. an explicit ``index`` / ``label_index`` on the span (what the aligner
     parsed out of the model's reply -- use it whenever it survives);
  2. otherwise the maximum order-preserving matching of the predicted texts
     onto ``L``, when exactly one such matching exists. A run that emits one
     span per label in label order has a matching of full size, so a unique
     maximum matching IS the true mapping;
  3. otherwise the mapping is genuinely ambiguous and guessing it is D10 all
     over again, so the episode is refused (``--ambiguous-policy``).

Route 2 also repairs a second, quieter face of the same defect.
``_ordered_span_matches`` is greedy on the label side and advances a cursor
past every predicted span it skips, so it is not always a *maximum* matching:
it can strand a span that does have a home and under-report
``placed_fraction``. Measured over random drop patterns on the 131
duplicate-bearing episodes of Corpus A, about one pattern in five leaves the
text-only mapping genuinely ambiguous (route 3), and of the rest roughly 30%
still score differently -- every one of those because greedy stranded a span,
not because an occurrence moved. Neither face touches a duplicate-free
episode, where the matching is unique and greedy finds it; that is what the
agreement test pins.

**Defect: boundary MAE is a drop counter, not an accuracy metric.**
The penalty for a missing boundary is ``max(|g - start|, |end - g|)``, at least
half the episode for any in-episode ``g``. Over all 3,983 internal boundaries
of Corpus A the *normalised* miss penalty runs min 0.500 / median 0.750 /
mean 0.741 / max 0.978, while a boundary placed at the best measured accuracy
(1.51 s on a ~100 s episode) contributes 0.0151. One dropped label therefore
outweighs roughly **49** accurately placed ones, and any MAE ranking is a
ranking on drop rate. So this scorer emits both halves of the decomposition and
never one alone:

  ``boundary_mae_placed``  accuracy GIVEN placement, over placed boundaries only
  ``placed_fraction``      the drop rate itself, a primary metric
  ``boundary_mae_norm``    the penalised mean over episode duration, diagnostic

``boundary_mae_placed`` improves when an arm drops the boundaries it finds
hard, so it is written into every row that carries it *together with*
``placed_fraction``; :func:`AlignmentScore.row` enforces that pairing rather
than trusting callers to remember it.

Rows and failures
-----------------
Every evaluation episode of every prediction file produces exactly one row.
Failures are never omitted -- the failure rate has to survive into aggregation
-- and they come in two kinds that must not be conflated:

  ``ok=false, scored=true``   the arm produced nothing for this episode. It is
                              scored at the floor (placed_fraction 0, tIoU 0,
                              every boundary a miss), because that is what its
                              output is worth, not a gap in the table.
  ``ok=false, scored=false``  the *episode* is unscorable: the supplied label
                              list does not match ground truth in length (the
                              `modal_list` coverage loss), or the label->span
                              mapping is ambiguous. Metrics are null; scoring
                              these would mean inventing a correspondence.

Hit thresholds are fixed in code at 1/3/5 s. Choosing tau after seeing which
value favours a configuration would invalidate the comparison, so it is not a
command-line knob.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
if str(REPO / "src") not in sys.path:
    # Fallback for running outside the project venv. The venv installs the
    # package editable from this same tree, so both routes load one source.
    sys.path.append(str(REPO / "src"))

Span = dict[str, Any]

# Contract with the aggregation and report agents: these exact strings.
HIT_THRESHOLDS: tuple[float, ...] = (1.0, 3.0, 5.0)
HIT_METRIC_NAMES: dict[float, str] = {1.0: "b_hit@1", 3.0: "b_hit@3", 5.0: "b_hit@5"}

# Separator between the disambiguating index and the label text. Uniqueness is
# already guaranteed by the integer prefix; the unit separator only keeps the
# re-keyed label readable in a traceback.
KEY_SEPARATOR = "\x1f"

# The DP that counts candidate mappings is polynomial, but the count it returns
# is not: a long list of identical labels has combinatorially many. Only "is it
# 1" is ever acted on, so the count saturates rather than growing unbounded.
MAX_EMBEDDING_COUNT = 10_000

METRIC_VERSION = 3


def _evaluate_alignment() -> Any:
    """Import the tool's metric lazily.

    Scoring a JSONL should not require the annotation package (and its heavy
    dependencies) to be importable, but the numbers must come from the real
    implementation when it is. Imported at call time so ``--help`` and a
    ``PYTHONPATH`` mistake fail with something legible.
    """
    from lerobot_align.alignment_metrics import evaluate_alignment

    return evaluate_alignment


# --------------------------------------------------------------------------
# D10: resolving each predicted span to the label index it belongs to
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class IndexResolution:
    """Which label each predicted span belongs to, and how sure we are."""

    indices: tuple[int | None, ...]
    source: str  # "explicit" | "recovered" | "text_greedy"
    n_candidates: int  # distinct maximum-size mappings; 1 when unambiguous


def _explicit_indices(labels: list[str], predicted: list[Span]) -> tuple[int, ...] | None:
    """Read ``index``/``label_index`` off the spans, or ``None`` if absent.

    All-or-nothing: a file where only some spans carry an index is malformed,
    and quietly recovering the rest would hide whatever produced it.
    """
    keys = [
        next((k for k in ("index", "label_index") if k in span and span[k] is not None), None)
        for span in predicted
    ]
    if all(key is None for key in keys):
        return None
    if any(key is None for key in keys):
        raise ValueError("some predicted spans carry a label index and others do not")

    out: list[int] = []
    previous = -1
    for span, key in zip(predicted, keys, strict=True):
        try:
            index = int(span[key])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"label index {span[key]!r} is not an integer") from exc
        if not 0 <= index < len(labels):
            raise ValueError(f"label index {index} outside [0, {len(labels)})")
        if index <= previous:
            # The aligner emits one span per label in label order; anything else
            # means the correspondence has already been scrambled upstream.
            raise ValueError(f"label indices are not strictly increasing: {index} after {previous}")
        text = span.get("text")
        if text is not None and str(text) != labels[index]:
            raise ValueError(
                f"span at label index {index} carries text {str(text)!r}, "
                f"but the supplied label is {labels[index]!r}"
            )
        out.append(index)
        previous = index
    return tuple(out)


def _recover_indices(labels: list[str], texts: list[str]) -> tuple[tuple[int | None, ...], int]:
    """Largest order-preserving text matching, plus how many such matchings exist.

    ``best[j][i]`` is the size of the largest matching of ``texts[j:]`` into
    ``labels[i:]``; ``ways[j][i]`` counts the matchings of that size. Predicted
    span ``j`` is decided first -- left unmatched, or matched to some later
    label ``k`` -- so each distinct matching is generated exactly once and the
    count is a real ambiguity measure rather than a path count.

    A count above 1 means the surviving spans genuinely do not determine which
    occurrence of a repeated label they came from. That is D10, and the caller
    must not paper over it.
    """
    m, n = len(texts), len(labels)
    best = [[0] * (n + 1) for _ in range(m + 1)]
    ways = [[1] * (n + 1) for _ in range(m + 1)]
    for j in range(m - 1, -1, -1):
        for i in range(n, -1, -1):
            size, count = best[j + 1][i], ways[j + 1][i]  # predicted j unmatched
            for k in range(i, n):
                if labels[k] != texts[j]:
                    continue
                candidate = 1 + best[j + 1][k + 1]
                if candidate > size:
                    size, count = candidate, ways[j + 1][k + 1]
                elif candidate == size:
                    count = min(MAX_EMBEDDING_COUNT, count + ways[j + 1][k + 1])
            best[j][i], ways[j][i] = size, count

    # Witness: prefer matching over dropping, and the earliest label that still
    # reaches the maximum. On a duplicate-free list the matching is unique, so
    # this is exactly what ``_ordered_span_matches`` returns.
    indices: list[int | None] = []
    i = 0
    for j in range(m):
        chosen: int | None = None
        for k in range(i, n):
            if labels[k] == texts[j] and 1 + best[j + 1][k + 1] == best[j][i]:
                chosen = k
                break
        indices.append(chosen)
        if chosen is not None:
            i = chosen + 1
    return tuple(indices), ways[0][0]


def resolve_label_indices(labels: list[str], predicted: list[Span]) -> IndexResolution:
    """Attribute every predicted span to a label index (D10).

    ``len(predicted) <= len(labels)`` is asserted, not merely assumed. The
    fixed-label path cannot produce an extra span -- ``_parse_align_spans``
    drops out-of-range and duplicate indices and ``_align_spans_in_order``
    appends at most one span per supplied label -- and extra spans are
    unpunished by the metric (D5), so if that invariant ever breaks it must
    break the scorer rather than quietly inflate a score.
    """
    if len(predicted) > len(labels):
        raise ValueError(
            f"{len(predicted)} predicted span(s) for {len(labels)} supplied label(s): "
            "extra spans cannot occur on the fixed-label path and are unpunished by the metric"
        )
    explicit = _explicit_indices(labels, predicted)
    if explicit is not None:
        return IndexResolution(indices=explicit, source="explicit", n_candidates=1)
    texts = [str(span.get("text", "")) for span in predicted]
    indices, candidates = _recover_indices(labels, texts)
    return IndexResolution(indices=indices, source="recovered", n_candidates=candidates)


# --------------------------------------------------------------------------
# Scoring one episode
# --------------------------------------------------------------------------


@dataclass
class AlignmentScore:
    """One episode's metrics, or the reason there are none."""

    n_labels: int
    n_placed: int = 0
    n_boundaries: int = 0
    n_boundaries_placed: int = 0
    episode_duration: float | None = None
    placed_fraction: float | None = None
    hit_rates: dict[float, float] = field(default_factory=dict)
    boundary_mae_placed: float | None = None
    boundary_mae_norm: float | None = None
    macro_temporal_iou: float | None = None
    index_source: str | None = None
    n_index_candidates: int | None = None
    n_unmatched_predicted: int = 0
    scored: bool = True
    error: str | None = None

    def row(self) -> dict[str, Any]:
        """Flat metric fields under the names the contract fixes.

        ``boundary_mae_placed`` is an average over the boundaries an arm chose
        to place, so it improves when hard labels are dropped. Emitting it
        without ``placed_fraction`` beside it would be reporting the numerator
        of a ratio; the assertion below makes that impossible by construction
        rather than by convention.
        """
        assert self.boundary_mae_placed is None or self.placed_fraction is not None, (
            "placed-only MAE must never be emitted without placed_fraction"
        )
        out: dict[str, Any] = {
            "n_labels": self.n_labels,
            "n_placed": self.n_placed,
            "n_boundaries": self.n_boundaries,
            "n_boundaries_placed": self.n_boundaries_placed,
            "episode_duration": self.episode_duration,
            "placed_fraction": self.placed_fraction,
        }
        for threshold in HIT_THRESHOLDS:
            out[HIT_METRIC_NAMES[threshold]] = self.hit_rates.get(threshold)
        out["boundary_mae_placed"] = self.boundary_mae_placed
        out["boundary_mae_norm"] = self.boundary_mae_norm
        out["macro_temporal_iou"] = self.macro_temporal_iou
        out["index_source"] = self.index_source
        out["n_index_candidates"] = self.n_index_candidates
        out["n_unmatched_predicted"] = self.n_unmatched_predicted
        out["scored"] = self.scored
        return out


def score_episode(
    labels: list[str],
    ground_truth: list[Span],
    predicted: list[Span],
    *,
    ambiguous_policy: str = "fail",
) -> AlignmentScore:
    """Score one episode with index-based label attribution.

    Raises ``ValueError`` only for invariant violations that should stop the
    run (too many spans, a malformed explicit index). Conditions that are
    properties of the *data* -- a label list of the wrong length, an ambiguous
    mapping -- come back as an unscorable ``AlignmentScore`` so they appear in
    the output as rows instead of aborting.
    """
    n_labels = len(labels)
    if len(ground_truth) != n_labels:
        return AlignmentScore(
            n_labels=n_labels,
            scored=False,
            error=(
                f"supplied label list has {n_labels} label(s) for "
                f"{len(ground_truth)} ground-truth span(s)"
            ),
        )

    resolution = resolve_label_indices(labels, predicted)
    if resolution.n_candidates > 1 and ambiguous_policy == "fail":
        return AlignmentScore(
            n_labels=n_labels,
            scored=False,
            index_source=resolution.source,
            n_index_candidates=resolution.n_candidates,
            error=(
                f"{resolution.n_candidates} order-preserving label mappings fit these "
                f"{len(predicted)} span(s); a repeated label lost its occurrence identity "
                "and no returned index survived to recover it (D10)"
            ),
        )

    evaluate = _evaluate_alignment()
    if resolution.n_candidates > 1:
        # ``text_greedy`` is the tool's own guess, so it is taken by handing
        # ``evaluate_alignment`` the untouched labels and spans rather than by
        # reimplementing its cursor. Not a shortcut: the recovery above returns
        # a MAXIMUM order-preserving matching, and ``_ordered_span_matches`` --
        # greedy on the label side, advancing a cursor past every predicted span
        # it skips -- is not always maximum, so it can strand a span that does
        # have a home and under-report ``placed_fraction``. Reproducing the
        # tool's number means calling the tool.
        source = "text_greedy"
        metrics = evaluate(labels, ground_truth, predicted, hit_thresholds=HIT_THRESHOLDS)
    else:
        # Re-key so the tool's text matcher cannot mis-attribute: every label is
        # unique, and each span already carries the key of its resolved index.
        source = resolution.source
        keyed_labels = [f"{index}{KEY_SEPARATOR}{text}" for index, text in enumerate(labels)]
        keyed_predicted = [
            {**span, "text": keyed_labels[index]}
            for span, index in zip(predicted, resolution.indices, strict=True)
            if index is not None
        ]
        metrics = evaluate(
            keyed_labels, ground_truth, keyed_predicted, hit_thresholds=HIT_THRESHOLDS
        )

    placed_errors = [error for error in metrics.boundary_errors if error is not None]
    duration = float(ground_truth[-1]["end"]) - float(ground_truth[0]["start"])
    # Recovered from the ratio so the count agrees with the metric it came from
    # under either branch; both are exact integers at these list lengths.
    n_placed = round(metrics.placed_fraction * n_labels)
    return AlignmentScore(
        n_labels=n_labels,
        n_placed=n_placed,
        n_boundaries=len(metrics.boundary_errors),
        n_boundaries_placed=len(placed_errors),
        episode_duration=duration,
        placed_fraction=metrics.placed_fraction,
        hit_rates=dict(metrics.boundary_hit_rates),
        # Placed-only: the misses are excluded, which is the whole point and
        # also the whole hazard. See the module docstring.
        boundary_mae_placed=(sum(placed_errors) / len(placed_errors) if placed_errors else None),
        # Penalised (misses included at their worst in-episode error) and
        # divided by the episode's extent so components of different length are
        # on one scale. Diagnostic only -- it is dominated by the drop rate.
        boundary_mae_norm=(
            metrics.boundary_mae / duration
            if metrics.boundary_mae is not None and duration > 0
            else None
        ),
        macro_temporal_iou=metrics.macro_temporal_iou,
        index_source=source,
        n_index_candidates=resolution.n_candidates,
        # Spans no label could take. D5: extra spans are unpunished by the
        # metric, so they are counted here instead of disappearing.
        n_unmatched_predicted=len(predicted) - n_placed,
    )


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------


def load_ground_truth(directory: Path) -> dict[str, dict[int, list[Span]]]:
    out: dict[str, dict[int, list[Span]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["dataset"]] = {int(k): v for k, v in payload["episodes"].items()}
    return out


def load_eval_splits(directory: Path) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["dataset"]] = sorted(int(e) for e in payload["eval"])
    return out


def load_label_file(path: Path) -> tuple[list[str], dict[int, list[str]]]:
    """Read a ``subtasks_path`` label file the way the tool reads it.

    Mirrors ``_load_subtasks_file``: a flat list applies to every episode, an
    object is keyed by episode index with an optional ``default``. Scoring an
    arm against a different reading of its own label file would be the quietest
    possible way to get every number wrong.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [str(item) for item in payload], {}
    if not isinstance(payload, dict):
        raise SystemExit(f"{path}: expected a JSON list or object")
    default: list[str] = []
    by_episode: dict[int, list[str]] = {}
    for key, value in payload.items():
        if key == "default":
            default = [str(item) for item in value]
            continue
        try:
            by_episode[int(key)] = [str(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"{path}: key {key!r} is neither 'default' nor an episode index"
            ) from exc
    if not default and not by_episode:
        raise SystemExit(f"{path}: no subtasks found")
    return default, by_episode


def _flag_value(flags: dict[str, Any], suffix: str) -> str | None:
    """Find a draccus flag by field name, ignoring its config section.

    ``run_arm.py`` records the flags it actually passed, with the label and
    calibration placeholders already substituted, so reading them back is the
    only source that cannot disagree with what the tool was given.
    """
    for key, value in flags.items():
        if value is not None and str(key).split(".")[-1] == suffix:
            return str(value)
    return None


def load_calibration_labels(path: Path) -> tuple[str, ...]:
    """The label tuple a calibration file was fit for.

    ``_AlignCalibration.applies_to`` is ``not self.labels or tuple(labels) ==
    self.labels`` -- an empty recorded list applies to *everything*. Stage B of
    the preflight aborts on that (plan §7.4.8); here it is only reported,
    because the scorer's job is to describe what the run did, not to re-gate it.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("labels") if isinstance(payload, dict) else None
    return tuple(str(item) for item in (raw or []))


# --------------------------------------------------------------------------
# Self-test: the agreement deliverable
# --------------------------------------------------------------------------


def _metrics_agree(mine: AlignmentScore, theirs: Any) -> list[str]:
    """Field-by-field comparison against a raw ``evaluate_alignment`` result."""

    def close(a: float | None, b: float | None) -> bool:
        if a is None or b is None:
            return a is None and b is None
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)

    problems: list[str] = []
    if not close(mine.placed_fraction, theirs.placed_fraction):
        problems.append(f"placed_fraction {mine.placed_fraction} != {theirs.placed_fraction}")
    if not close(mine.macro_temporal_iou, theirs.macro_temporal_iou):
        problems.append(
            f"macro_temporal_iou {mine.macro_temporal_iou} != {theirs.macro_temporal_iou}"
        )
    for threshold in HIT_THRESHOLDS:
        if not close(mine.hit_rates.get(threshold), theirs.boundary_hit_rates.get(threshold)):
            problems.append(
                f"{HIT_METRIC_NAMES[threshold]} {mine.hit_rates.get(threshold)} != "
                f"{theirs.boundary_hit_rates.get(threshold)}"
            )
    placed = [e for e in theirs.boundary_errors if e is not None]
    expected_placed = sum(placed) / len(placed) if placed else None
    if not close(mine.boundary_mae_placed, expected_placed):
        problems.append(f"boundary_mae_placed {mine.boundary_mae_placed} != {expected_placed}")
    if mine.episode_duration and theirs.boundary_mae is not None:
        expected_norm = theirs.boundary_mae / mine.episode_duration
        if not close(mine.boundary_mae_norm, expected_norm):
            problems.append(f"boundary_mae_norm {mine.boundary_mae_norm} != {expected_norm}")
    return problems


def _synthetic_prediction(
    spans: list[Span], keep: list[int], rng: random.Random, *, with_index: bool
) -> list[Span]:
    """A plausible aligner output: an ordered subset of the truth, jittered."""
    out: list[Span] = []
    for index in keep:
        start = float(spans[index]["start"]) + rng.uniform(-2.0, 2.0)
        end = max(start, float(spans[index]["end"]) + rng.uniform(-2.0, 2.0))
        span: Span = {"text": str(spans[index]["text"]), "start": start, "end": end}
        if with_index:
            span["index"] = index
        out.append(span)
    return out


def _test_corpus_agreement(gt_dir: Path, draws: int) -> tuple[int, list[str]]:
    """On duplicate-free episodes the two scorers MUST agree exactly.

    This is the deliverable that proves the D10 fix changed only the cases D10
    is about. Every episode whose label list has no repeat is scored both ways,
    over several random drop patterns and including the no-drop case.

    Both index routes are exercised. Spans that carry an explicit ``index`` test
    the route the fix is built on; spans that carry only text test the recovery
    route, which is what today's prediction files need because the index does
    not survive the parquet round-trip. On a duplicate-free list the recovery is
    unique, so it must also report exactly one candidate -- an ambiguity here
    would mean the recovery is guessing where nothing is in doubt.
    """
    evaluate = _evaluate_alignment()
    checked, problems = 0, []
    for path in sorted(gt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        for episode, spans in sorted(payload["episodes"].items(), key=lambda kv: int(kv[0])):
            labels = [str(span["text"]) for span in spans]
            if len(set(labels)) != len(labels):
                continue
            rng = random.Random(f"{dataset}:{episode}")
            patterns = [list(range(len(labels)))]
            for _ in range(draws):
                keep = [i for i in range(len(labels)) if rng.random() > 0.35]
                patterns.append(keep or [rng.randrange(len(labels))])
            for keep in patterns:
                for with_index in (True, False):
                    predicted = _synthetic_prediction(spans, keep, rng, with_index=with_index)
                    mine = score_episode(labels, spans, predicted)
                    theirs = evaluate(labels, spans, predicted, hit_thresholds=HIT_THRESHOLDS)
                    checked += 1
                    where = f"{dataset} ep{episode} keep={keep} index={with_index}"
                    expected_source = "explicit" if with_index else "recovered"
                    if mine.index_source != expected_source:
                        problems.append(f"{where}: index_source {mine.index_source}")
                    if mine.n_index_candidates != 1:
                        problems.append(
                            f"{where}: {mine.n_index_candidates} candidate mappings on a "
                            "duplicate-free label list"
                        )
                    for problem in _metrics_agree(mine, theirs):
                        problems.append(f"{where}: {problem}")
    return checked, problems


def _test_text_greedy_fidelity(gt_dir: Path, draws: int) -> tuple[int, list[str]]:
    """``--ambiguous-policy text_greedy`` must BE the tool, not resemble it.

    The option exists to measure how often the ambiguity bites, so it has to
    reproduce ``evaluate_alignment`` exactly on the episodes where it engages.
    It is taken by calling the tool on the untouched inputs, and this checks
    that nothing in the wrapper leaks into that branch.
    """
    evaluate = _evaluate_alignment()
    checked, problems = 0, []
    for path in sorted(gt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = payload["dataset"]
        for episode, spans in sorted(payload["episodes"].items(), key=lambda kv: int(kv[0])):
            labels = [str(span["text"]) for span in spans]
            if len(set(labels)) == len(labels):
                continue
            rng = random.Random(f"greedy:{dataset}:{episode}")
            for _ in range(draws):
                keep = [i for i in range(len(labels)) if rng.random() > 0.3] or [0]
                predicted = _synthetic_prediction(spans, keep, rng, with_index=False)
                if resolve_label_indices(labels, predicted).n_candidates == 1:
                    continue
                mine = score_episode(labels, spans, predicted, ambiguous_policy="text_greedy")
                theirs = evaluate(labels, spans, predicted, hit_thresholds=HIT_THRESHOLDS)
                checked += 1
                if mine.index_source != "text_greedy":
                    problems.append(
                        f"{dataset} ep{episode} keep={keep}: source {mine.index_source}"
                    )
                for problem in _metrics_agree(mine, theirs):
                    problems.append(f"{dataset} ep{episode} keep={keep}: {problem}")
    return checked, problems


def _test_duplicate_divergence() -> list[str]:
    """A duplicate-label case where index and text matching genuinely differ.

    ``L = [open, take, open]`` with only the LAST span surviving -- the shape
    ``_align_spans_in_order`` produces whenever it drops a label. Text matching
    hands the survivor to occurrence 0 and scores it as a total miss; index
    matching puts it where it belongs and scores it as nearly exact.
    """
    evaluate = _evaluate_alignment()
    labels = ["open the fridge", "take the bottle", "open the fridge"]
    truth = [
        {"start": 0.0, "end": 10.0, "text": labels[0]},
        {"start": 10.0, "end": 20.0, "text": labels[1]},
        {"start": 20.0, "end": 30.0, "text": labels[2]},
    ]
    predicted = [{"index": 2, "text": labels[2], "start": 20.5, "end": 30.0}]

    mine = score_episode(labels, truth, predicted)
    theirs = evaluate(labels, truth, predicted, hit_thresholds=HIT_THRESHOLDS)

    problems: list[str] = []
    if mine.hit_rates[1.0] != 0.5:
        problems.append(f"index matching should hit 1 of 2 boundaries, got {mine.hit_rates[1.0]}")
    if theirs.boundary_hit_rates[1.0] != 0.0:
        problems.append(
            f"text matching should hit none, got {theirs.boundary_hit_rates[1.0]} -- "
            "the defect this test pins may have moved"
        )
    if mine.boundary_mae_placed is None or abs(mine.boundary_mae_placed - 0.5) > 1e-9:
        problems.append(f"placed-only MAE should be 0.5s, got {mine.boundary_mae_placed}")
    if any(error is not None for error in theirs.boundary_errors):
        problems.append("text matching unexpectedly placed a boundary")
    if not (mine.macro_temporal_iou or 0.0) > (theirs.macro_temporal_iou or 0.0):
        problems.append(
            f"tIoU should improve: index {mine.macro_temporal_iou} "
            f"vs text {theirs.macro_temporal_iou}"
        )
    if mine.placed_fraction != theirs.placed_fraction:
        problems.append("placed_fraction should be identical; only the attribution changed")

    # Without a surviving index the mapping is ambiguous (occurrence 0 or 2),
    # and refusing to guess is the point of the fix.
    blind = [{"text": labels[2], "start": 20.5, "end": 30.0}]
    refused = score_episode(labels, truth, blind)
    if refused.scored or refused.n_index_candidates != 2:
        problems.append(
            f"ambiguous mapping should be refused with 2 candidates, got "
            f"scored={refused.scored} candidates={refused.n_index_candidates}"
        )
    guessed = score_episode(labels, truth, blind, ambiguous_policy="text_greedy")
    if guessed.hit_rates.get(1.0) != theirs.boundary_hit_rates[1.0]:
        problems.append("text_greedy policy should reproduce evaluate_alignment exactly")
    return problems


def _test_miss_penalty() -> list[str]:
    """One drop must dominate ~49 accurate placements, and be visible as a drop.

    100 s episode, four internal boundaries at 20/40/60/80 s, every placed
    boundary 1.51 s late -- the best accuracy the prior work reports. Dropping
    the 80 s boundary costs ``max(80, 20) / 100 = 0.80`` normalised, against
    0.0151 for a placed one.
    """
    labels = [f"step {i}" for i in range(5)]
    starts = [0.0, 20.0, 40.0, 60.0, 80.0]
    truth = [
        {"start": starts[i], "end": (starts[i + 1] if i + 1 < 5 else 100.0), "text": labels[i]}
        for i in range(5)
    ]
    everything = [
        {"index": i, "text": labels[i], "start": starts[i] + 1.51, "end": float(truth[i]["end"])}
        for i in range(5)
    ]
    dropped = [span for span in everything if span["index"] != 4]

    full = score_episode(labels, truth, everything)
    partial = score_episode(labels, truth, dropped)

    problems: list[str] = []
    if abs((full.boundary_mae_placed or 0.0) - 1.51) > 1e-9:
        problems.append(f"placed-only MAE should be 1.51s, got {full.boundary_mae_placed}")
    if abs((partial.boundary_mae_placed or 0.0) - 1.51) > 1e-9:
        problems.append(
            "placed-only MAE must be BLIND to the drop -- that is why it is never "
            f"reported alone; got {partial.boundary_mae_placed}"
        )
    if partial.placed_fraction != 0.8 or full.placed_fraction != 1.0:
        problems.append(
            f"placed_fraction must carry the drop: {full.placed_fraction} "
            f"-> {partial.placed_fraction}"
        )
    if partial.hit_rates[3.0] != 0.75 or full.hit_rates[3.0] != 1.0:
        problems.append(
            f"b_hit@3 must score the miss as a miss: {full.hit_rates[3.0]} "
            f"-> {partial.hit_rates[3.0]}"
        )
    # 0.80 for the dropped boundary against 0.0151 for a placed one.
    penalty = (partial.boundary_mae_norm or 0.0) * 4 - (0.0151 * 3)
    ratio = penalty / 0.0151
    if not 45.0 < ratio < 55.0:
        problems.append(f"one drop should outweigh ~49 accurate placements, got {ratio:.1f}")
    if (partial.boundary_mae_norm or 0.0) <= (full.boundary_mae_norm or 0.0):
        problems.append("normalised MAE must worsen on a drop")
    return problems


def _test_invariants() -> list[str]:
    """The assertions that must fail loudly rather than score silently."""
    labels = ["a", "b"]
    truth = [
        {"start": 0.0, "end": 5.0, "text": "a"},
        {"start": 5.0, "end": 10.0, "text": "b"},
    ]
    problems: list[str] = []

    cases: list[tuple[str, list[Span]]] = [
        (
            "extra span",
            [
                {"index": 0, "text": "a", "start": 0.0, "end": 5.0},
                {"index": 1, "text": "b", "start": 5.0, "end": 8.0},
                {"index": 1, "text": "b", "start": 8.0, "end": 10.0},
            ],
        ),
        ("index out of range", [{"index": 7, "text": "a", "start": 0.0, "end": 5.0}]),
        (
            "non-increasing indices",
            [
                {"index": 1, "text": "b", "start": 0.0, "end": 5.0},
                {"index": 0, "text": "a", "start": 5.0, "end": 10.0},
            ],
        ),
        ("index disagrees with text", [{"index": 1, "text": "a", "start": 0.0, "end": 5.0}]),
        (
            "index on only some spans",
            [
                {"index": 0, "text": "a", "start": 0.0, "end": 5.0},
                {"text": "b", "start": 5.0, "end": 10.0},
            ],
        ),
    ]
    for name, predicted in cases:
        try:
            score_episode(labels, truth, predicted)
        except ValueError:
            continue
        problems.append(f"{name}: should have raised")

    # An arm that produced nothing is scored at the floor, not skipped.
    empty = score_episode(labels, truth, [])
    if (
        empty.placed_fraction != 0.0
        or empty.macro_temporal_iou != 0.0
        or empty.hit_rates[5.0] != 0.0
    ):
        problems.append(f"an empty prediction must score zero, got {empty.row()}")
    if not empty.scored:
        problems.append("an empty prediction is scorable at the floor, not unscorable")

    # A label list of the wrong length is coverage loss, reported as a row.
    mismatch = score_episode(["a", "b", "c"], truth, [])
    if mismatch.scored or "3 label(s)" not in (mismatch.error or ""):
        problems.append(f"label-count mismatch should be unscorable, got {mismatch.error}")
    return problems


def run_self_test(gt_dir: Path | None, draws: int) -> int:
    checks: list[tuple[str, list[str]]] = [
        ("duplicate-label divergence", _test_duplicate_divergence()),
        ("miss penalty", _test_miss_penalty()),
        ("invariants", _test_invariants()),
    ]
    if gt_dir is not None:
        checked, problems = _test_corpus_agreement(gt_dir, draws)
        checks.append(
            (f"agreement with evaluate_alignment ({checked} duplicate-free cases)", problems)
        )
        checked, problems = _test_text_greedy_fidelity(gt_dir, 2 * draws)
        checks.append((f"text_greedy is evaluate_alignment ({checked} ambiguous cases)", problems))
    else:
        print("[self-test] no --gt-dir: the corpus agreement test was NOT run", file=sys.stderr)

    failed = 0
    for name, problems in checks:
        if problems:
            failed += 1
            print(f"[self-test] FAIL {name}")
            for problem in problems[:10]:
                print(f"             {problem}")
            if len(problems) > 10:
                print(f"             ... and {len(problems) - 10} more")
        else:
            print(f"[self-test] ok   {name}")
    return 1 if failed else 0


# --------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--gt-dir", type=Path, default=None)
    parser.add_argument("--splits-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None, help="JSONL to write")
    parser.add_argument("--eligibility", type=Path,
                        help="shared task/episode eligibility JSON exported by the cohort merger; applies to every arm")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="restrict to these components by name (same flag and "
                             "meaning as make_label_files.py --datasets); default is "
                             "every component found under --predictions-dir")
    parser.add_argument(
        "--labels-dir", type=Path, default=None,
        help="fallback supplied-label files, <dataset>.json; the arm's own recorded "
             "--subtasks-path flag wins over this",
    )
    parser.add_argument(
        "--ambiguous-policy", choices=("fail", "text_greedy"), default="fail",
        help="what to do when a repeated label's occurrence cannot be recovered: refuse "
             "the episode (default), or reproduce evaluate_alignment's text guess",
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="validate the scorer instead of scoring; with --gt-dir this runs the "
             "agreement test against evaluate_alignment on every duplicate-free episode",
    )
    parser.add_argument(
        "--self-test-draws", type=int, default=3,
        help="random drop patterns per episode in the self-test, on top of the "
             "no-drop case that is always included",
    )
    args = parser.parse_args()

    if args.self_test:
        return run_self_test(args.gt_dir, args.self_test_draws)

    missing = [
        name
        for name, value in (
            ("--predictions-dir", args.predictions_dir),
            ("--gt-dir", args.gt_dir),
            ("--splits-dir", args.splits_dir),
            ("--out", args.out),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"missing required argument(s): {', '.join(missing)}")

    truth = load_ground_truth(args.gt_dir)
    if not truth:
        raise SystemExit(f"no ground truth under {args.gt_dir}")
    splits = load_eval_splits(args.splits_dir)
    shared_eligibility = None
    if args.eligibility:
        mask = json.loads(args.eligibility.read_text())
        if mask.get("version") != 1 or not isinstance(mask.get("episodes"), dict):
            raise SystemExit("invalid shared eligibility mask")
        shared_eligibility = mask["episodes"]

    prediction_files = sorted(args.predictions_dir.glob("*.json"))
    if not prediction_files:
        raise SystemExit(f"no predictions under {args.predictions_dir}")

    # `--datasets` names components, exactly as `make_label_files.py --datasets`
    # and `align_floors.py --datasets` do. Filtering on the RECORDED dataset field
    # rather than on the filename is deliberate: a prediction file is named
    # <dataset>__<arm>.json, but the dataset a row belongs to is the one inside the
    # record, and those are the two things that must never be allowed to disagree.
    if args.datasets:
        wanted = list(dict.fromkeys(args.datasets))
        unknown = [name for name in wanted if name not in truth]
        if unknown:
            raise SystemExit(
                f"--datasets names component(s) with no ground truth under {args.gt_dir}: "
                f"{unknown}. Known: {sorted(truth)}"
            )
        selected = set(wanted)
        prediction_files = [
            path for path in prediction_files
            if json.loads(path.read_text(encoding="utf-8"))["dataset"] in selected
        ]
        if not prediction_files:
            raise SystemExit(
                f"no predictions under {args.predictions_dir} for {wanted}. A subset that "
                "turns out to be empty is a missing run, not a smaller population."
            )

    # Validate every file before opening the output, so a misconfigured run
    # aborts instead of leaving a half-written table that looks complete.
    for path in prediction_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        dataset = record["dataset"]
        if dataset not in truth:
            raise SystemExit(f"{path.name}: no ground truth for {dataset}")
        if not splits.get(dataset):
            raise SystemExit(f"{path.name}: missing or empty evaluation split for {dataset}")
        if shared_eligibility is not None:
            mask_rows = shared_eligibility.get(dataset)
            if (not isinstance(mask_rows, dict) or set(mask_rows) != {str(e) for e in splits[dataset]}
                    or any(type(v) is not bool for v in mask_rows.values())):
                raise SystemExit(f"{dataset}: shared eligibility must cover exactly the evaluation split with booleans")
            for key, value in (record.get("calibration_eligible") or {}).items():
                if key not in mask_rows or value != mask_rows[key]:
                    raise SystemExit(f"{dataset}/{key}: recorded eligibility conflicts with the shared allocation")
        # Never let an episode vanish from the table because it happens to lack
        # ground truth: the row count per arm has to be the split's size, or the
        # failure rate computed downstream is over the wrong denominator.
        missing_truth = set(splits[dataset]) - set(truth[dataset])
        if missing_truth:
            raise SystemExit(
                f"{path.name}: evaluation episodes with no ground truth for {dataset}: "
                f"{sorted(missing_truth)}"
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    unscorable = 0
    label_sources: dict[str, int] = {}
    # Written aside and renamed only on success. An invariant violation aborts
    # mid-stream, and a truncated JSONL at the expected path is indistinguishable
    # from a complete one to everything downstream.
    staging = args.out.with_name(args.out.name + ".partial")
    with staging.open("w", encoding="utf-8") as handle:
        for path in prediction_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            dataset, arm = payload["dataset"], payload["arm"]
            flags = payload.get("flags") or {}
            predictions = {int(k): (v or []) for k, v in (payload.get("episodes") or {}).items()}
            if payload.get("ok") is False:
                # A failed whole job can retain extracted/staged spans, including
                # after a frozen-file or instrumentation check failed. They are
                # not delivered predictions and must score as missing.
                predictions = {}

            # The label list the arm was actually given, from its own recorded
            # flags where possible. Falling back to ground truth is correct only
            # because `oracle_list` IS the episode's own label list (plan §4.2);
            # every row records which route was taken so a modal-list arm
            # scored against the wrong list cannot hide.
            label_path = _flag_value(flags, "subtasks_path")
            label_source = "flags"
            if not label_path:
                # `align_floors.py` records the label file it used at the top
                # level rather than as a tool flag -- it invokes no CLI.
                label_path, label_source = payload.get("subtasks_path"), "provenance"
            if not label_path and args.labels_dir is not None:
                candidate = args.labels_dir / f"{dataset}.json"
                if candidate.exists():
                    label_path, label_source = str(candidate), "labels_dir"
            default_labels: list[str] = []
            by_episode: dict[int, list[str]] = {}
            if label_path:
                default_labels, by_episode = load_label_file(Path(label_path))
            else:
                label_source = "ground_truth"

            calibration_path = _flag_value(flags, "subtask_align_calibration_path")
            runtime_status = {
                int(k): v for k, v in (payload.get("calibration_status") or {}).items()
            }
            calibration_labels: tuple[str, ...] | None = None
            calibration_scope, calibration_count = "exact", None
            if calibration_path and not payload.get("calibration_status_version"):
                if not Path(calibration_path).exists():
                    # A calibrated arm silently scored as uncalibrated would
                    # invalidate claim A3; refuse rather than guess.
                    raise SystemExit(
                        f"{path.name}: arm {arm} declares calibration {calibration_path}, "
                        "which does not exist"
                    )
                calibration_labels = load_calibration_labels(Path(calibration_path))
                calibration_file = json.loads(Path(calibration_path).read_text())
                if isinstance(calibration_file, dict):
                    calibration_scope = calibration_file.get("label_scope", "exact")
                    calibration_count = calibration_file.get("n_segments")
                if not calibration_labels and calibration_scope != "segment_count":
                    print(
                        f"[score] WARNING {dataset} {arm}: calibration records no 'labels'; "
                        "applies_to() accepts every label list (plan §3.3)",
                        file=sys.stderr,
                    )
            episode_calibration = {
                int(k): bool(v) for k, v in (payload.get("calibration_applied") or {}).items()
            }
            eligible = {
                int(k): bool(v) for k, v in (payload.get("calibration_eligible") or {}).items()
            }
            if shared_eligibility is not None:
                eligible = {int(k): v for k, v in shared_eligibility[dataset].items()}

            for episode in splits[dataset]:
                reference = truth[dataset][episode]  # guaranteed by the check above
                labels = by_episode.get(episode) or default_labels
                if not labels:
                    labels = [str(span["text"]) for span in reference]
                predicted = predictions.get(episode) or []

                try:
                    score = score_episode(
                        labels, reference, predicted, ambiguous_policy=args.ambiguous_policy
                    )
                except ValueError as exc:
                    raise SystemExit(f"{dataset} ep{episode} {arm}: {exc}") from exc

                if score.error is not None:
                    ok, error = False, score.error
                elif not predicted:
                    ok, error = False, "arm produced no spans for this episode"
                else:
                    ok, error = True, None
                unscorable += not score.scored

                # Reliability, as plan §6 defines it: the fraction of episodes
                # where applies_to() accepted. The tool also short-circuits when
                # nothing parsed, so an episode with no spans counts as 0.
                status = runtime_status.get(episode)
                if payload.get("ok") is False:
                    applied, applied_source = False, "delivery"
                    status = {"applied": False, "reason": "job_failed"}
                elif status is not None:
                    if type(status.get("applied")) is not bool:
                        raise SystemExit(f"{path.name}: invalid runtime status for episode {episode}")
                    applied = status["applied"]
                    applied_source = status.get("source", "runtime")
                elif payload.get("calibration_status_version"):
                    if predicted:
                        raise SystemExit(f"{path.name}: missing runtime status for episode {episode}")
                    applied, applied_source = False, "no_output"
                elif episode in episode_calibration:
                    applied = episode_calibration[episode]
                    applied_source = "legacy_reported_unverified"
                elif calibration_labels is None:
                    applied = False
                    applied_source = "not_configured"
                else:
                    applied = bool(predicted) and (
                        len(labels) == calibration_count if calibration_scope == "segment_count"
                        else not calibration_labels or tuple(labels) == calibration_labels
                    )
                    applied_source = "legacy_schema_inference"

                row = {
                    "dataset": dataset,
                    "episode": episode,
                    "arm": arm,
                    "repeat": int(payload.get("repeat", 0) or 0),
                    "ok": ok,
                    "error": error,
                    **score.row(),
                    "calibration_applied": int(applied),
                    "calibration_applied_source": applied_source,
                    "calibration_reason": status.get("reason") if status else None,
                    "calibration_eligible": eligible.get(episode),
                    "calibration_eligibility_source": (
                        "shared_seed_count_support" if shared_eligibility is not None
                        else payload.get("calibration_eligibility_source")
                    ),
                    "has_duplicate_labels": len(set(labels)) != len(labels),
                    "label_source": label_source,
                    "arm_has_calibration": calibration_path is not None or any(
                        c.get("calibration_configured") for c in payload.get("cohorts", [])
                    ),
                    "supervision": payload.get("supervision"),
                    "tool": payload.get("tool"),
                    "metric_version": METRIC_VERSION,
                }
                handle.write(json.dumps(row) + "\n")
                written += 1
                label_sources[label_source] = label_sources.get(label_source, 0) + 1

    staging.replace(args.out)
    print(f"[score] wrote {written} rows -> {args.out}")
    print(f"[score] label sources: {label_sources}")
    if unscorable:
        print(f"[score] {unscorable} row(s) unscorable (ok=false, metrics null); see 'error'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
