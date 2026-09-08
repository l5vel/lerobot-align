#!/usr/bin/env python
"""Harvest real (model label, human label) pairs for blind human adjudication.

``calibrate_matcher.py`` selects the embedding threshold from pairs it builds
out of the human vocabulary alone. Every one of its positives is string-
identical after ``normalise_label`` -- ``("open the fridge door", "open the
fridge door.")`` and the article-stripped variant of the same string -- so the
sweep is scored against questions the matcher answers correctly at *any*
threshold, and it can only ever select its own ceiling. The number it reports
is therefore an upper bound on a task nobody is asking it to do. The task it
*is* asked to do is "is the model's paraphrase the same action as the human's
label", and no automatic construction can produce those pairs, because one
side of every one of them has to come out of a model. Until they exist the
threshold is provisional and no label-aware number can be reported
(`analysis/deviations.md`, open items).

This script closes that gap in two halves: ``harvest`` turns prediction files
into a blank review file, and ``validate`` turns the filled review file into
the ``--extra-pairs`` JSON that ``calibrate_matcher.py`` already accepts.

Where the pairs come from
-------------------------
Model spans are paired with human spans by the **same order-preserving IoU
matching that ``metrics.segmentation.matched_iou`` uses to compute
``macro_iou``**, then filtered to pairs with real temporal overlap. So the
adjudicator is shown the questions the scorer will actually ask. Pairing the
labels some other way -- nearest by embedding, say, or by multiset alignment
-- would produce a validation set for a matcher nothing runs.

Why the selection is not random
-------------------------------
A uniform sample of matched pairs is dominated by cases the matcher scores
0.99 or 0.05, whose verdicts are already known and which move no threshold. A
budget of ninety adjudications spent there buys almost nothing. Selection is
therefore concentrated where a small threshold change flips the verdict:
``|similarity - threshold| <= band``. Those are the pairs that decide the
number, and they are also the pairs where the matcher is least trustworthy.

Controls, and why they must be indistinguishable
------------------------------------------------
A hard-only sample cannot tell a careless adjudicator from a difficult task,
so the file also carries pairs the matcher is near-certain about: the highest-
and lowest-similarity candidates. Disagreement there is a signal to look at
the adjudicator, not at the threshold.

Controls only work while they look exactly like everything else. They are
drawn from the same candidate pool, carry the same fields, and are interleaved
by hashing rather than grouped, because an adjudicator who can spot the
attention checks answers them as a different task from the one being measured.
Note the honest limit: a control's "correct" answer is the matcher's own
confident verdict, not ground truth, so a disagreement flags a pair for review
rather than convicting the adjudicator.

What the review file deliberately does not contain
--------------------------------------------------
No arm name, no dataset, no episode, no similarity, no stratum, and no hint of
what the matcher currently thinks. Every one of those anchors the verdict:

* Showing the matcher's guess turns the exercise into a measurement of
  anchoring. Agreement rates collected that way are near-100% whatever the
  threshold is, which is exactly the failure this script exists to avoid
  repeating in a different form.
* Showing the arm is worse than useless, it is a *directional* bias. The person
  adjudicating is the person who wrote one of the two tools; knowing that a
  label came from ``align_video_realign`` rather than ``baseline_upstream``
  lets a favourable reading slip in, and the resulting threshold feeds every
  label-aware metric of every arm. ``metrics.semantic.JudgeMatcher`` is blinded
  for the same reason; a human judge is not exempt from a rule we impose on an
  LLM.
* Even the order within a pair is blinded: it is fixed by a hash of the pair,
  not by which side the model produced, so an adjudicator cannot learn that the
  second column is always the human label and start reading it as the answer.

The mapping back to arm, dataset, similarity and stratum is written to a
separate key file, which ``validate`` needs and the adjudicator must not open.

Why the controls do not go back into ``--extra-pairs``
------------------------------------------------------
They were selected *because the matcher is confident about them*. Feeding them
into the sweep would reward the matcher for the cases it already gets right and
re-inflate precisely the ceiling this harvest exists to escape -- the same
defect as the string-identical auto-positives, laundered through a human. Only
the near-threshold pairs are exported (``--include-controls`` overrides, and
the report records which was done).

That makes the exported set deliberately hard and matcher-selected. The
balanced accuracy ``calibrate_matcher.py`` reports after ingesting it is a
**lower bound**, not an unbiased estimate of matcher quality, and must be
quoted as such. The unbiased quantity this script does produce is the
agreement between the matcher and the adjudicators *inside the near band*, plus
the separability check: if a human-"same" pair scores below a human-"different"
pair then no threshold separates them, and §6.2's rule to demote the embedding
backend applies regardless of what the sweep reports.

Determinism
-----------
Same predictions, same ground truth, same flags, same file -- byte for byte.
Pair identity is a hash of the two normalised labels, so it does not depend on
which arm or episode the pair was first seen in; ordering, orientation and
selection are all functions of that hash and of the recorded similarity. The
annotation runs are not reproducible (neither tool seeds its generation call),
but the harvest of a fixed set of prediction files is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from metrics.segmentation import (  # noqa: E402
    iou,
    matched_iou,
    validate_segmentation,
)
from metrics.semantic import (  # noqa: E402
    EmbeddingMatcher,
    TokenF1Matcher,
    normalise_label,
)

Span = dict[str, Any]

SAME, DIFFERENT, UNSURE = "same", "different", "unsure"
VERDICTS = (SAME, DIFFERENT, UNSURE)

NEAR = "near_threshold"
CONTROL_MATCH = "control_match"
CONTROL_MISMATCH = "control_mismatch"
# The verdict a control is expected to draw. Expected, not known: it is the
# matcher's own confident opinion, which is why disagreement is reported as a
# flag rather than scored as an adjudicator error.
CONTROL_EXPECTED = {CONTROL_MATCH: SAME, CONTROL_MISMATCH: DIFFERENT}

REVIEW_SCHEMA = "harvest_pairs/review/1"
KEY_SCHEMA = "harvest_pairs/key/1"

# Fixed before adjudication starts, because a criterion agreed afterwards is a
# criterion fitted to the cases. Wording follows JudgeMatcher's contract in
# metrics/semantic.py so the human and LLM backends answer the same question.
QUESTION = (
    "Do these two phrases denote the SAME physical robot action on the same object?"
)
INSTRUCTIONS = (
    "Fill in 'adjudicator' with your name, then write one of 'same', "
    "'different' or 'unsure' into every 'verdict' field. Leave nothing else "
    "changed -- editing a label invalidates the pair and validation will "
    "refuse the file.",
    "'same' means a scorer treating the two phrases as one step would be "
    "right: same action, same object, same direction. Differences of wording, "
    "detail or tense do not matter.",
    "'different' means they denote different actions, or the same action in "
    "the opposite direction (open/close, left/right, pick up/put down), or act "
    "on different objects.",
    "'unsure' means the phrases are too vague or too damaged to decide. Use "
    "it rather than guessing; unsure pairs are excluded from calibration and "
    "reported separately.",
    "The two labels are shown in an arbitrary order. Neither column is "
    "consistently the human annotation, and nothing here indicates what the "
    "automatic matcher currently says -- both are withheld on purpose so the "
    "verdicts measure your judgement rather than your agreement with a "
    "machine.",
)


# --------------------------------------------------------------------------
# reading predictions and ground truth
# --------------------------------------------------------------------------

def _coerce_spans(raw: Any) -> list[Span] | None:
    """Coerce one episode's spans, or ``None`` when the record is unusable.

    Prediction files are written by a subprocess whose output we do not
    control, so a span may be missing a bound, carry a non-numeric one, or not
    be a mapping at all. Such an episode is dropped and counted, never
    partially repaired: a half-read episode would contribute pairs whose
    temporal matching is meaningless.
    """
    if not isinstance(raw, list) or not raw:
        return None
    spans: list[Span] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            return None
        spans.append({"start": start, "end": end, "text": str(item.get("text", ""))})
    return spans


def _coerce_episode_map(raw: Any) -> list[tuple[int, list[Span]]]:
    """``{episode: spans}`` -> sorted ``[(episode, spans)]``, dropping junk."""
    out: list[tuple[int, list[Span]]] = []
    if not isinstance(raw, dict):
        return out
    for key, value in raw.items():
        try:
            episode = int(key)
        except (TypeError, ValueError):
            continue
        spans = _coerce_spans(value)
        if spans is not None:
            out.append((episode, spans))
    out.sort(key=lambda item: item[0])
    return out


def load_ground_truth(directory: Path) -> dict[str, dict[int, list[Span]]]:
    out: dict[str, dict[int, list[Span]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[str(payload["dataset"])] = dict(_coerce_episode_map(payload.get("episodes")))
    return out


def load_scope(splits_dir: Path | None, scope: str) -> dict[str, set[int]] | None:
    """Episode filter per dataset, or ``None`` to accept every episode.

    ``eval`` restricts the harvest to the episodes the study actually scores;
    ``seed`` restricts it to episodes that are never scored, which is the only
    scope under which hand-adjudicating pairs cannot touch the test set at all.
    The default is ``all`` because §6.2 prescribes harvesting from the pilot,
    whose episodes come from the evaluation split; the overlap is recorded in
    the key file so the report can state it rather than discover it later.
    """
    if scope == "all":
        return None
    if splits_dir is None:
        raise SystemExit(f"--scope {scope} needs --splits-dir")
    out: dict[str, set[int]] = {}
    for path in sorted(splits_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[str(payload["dataset"])] = {int(e) for e in payload.get(scope) or []}
    if not out:
        raise SystemExit(f"no split files under {splits_dir}")
    return out


# --------------------------------------------------------------------------
# candidate pairs
# --------------------------------------------------------------------------

def pair_id(left: str, right: str) -> str:
    """Order-independent identity of a label pair.

    Hashing the sorted *normalised* forms means the same question is asked
    once however many episodes, arms or spellings produced it, and that the id
    carries no trace of stratum, arm or similarity -- so sorting the review
    file by it interleaves hard pairs and controls without any further
    shuffling.
    """
    a, b = sorted((normalise_label(left), normalise_label(right)))
    return hashlib.sha256(f"{a}\x00{b}".encode()).hexdigest()[:12]


def collect_candidates(
    prediction_files: list[Path],
    truth: dict[str, dict[int, list[Span]]],
    scope: dict[str, set[int]] | None,
    min_iou: float,
) -> tuple[dict[str, dict[str, Any]], Counter]:
    """Deduplicated candidate pairs, plus a census of what was skipped."""
    skipped: Counter = Counter()
    pairs: dict[str, dict[str, Any]] = {}

    for path in prediction_files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped["unreadable_prediction_file"] += 1
            continue
        if not isinstance(payload, dict):
            skipped["unreadable_prediction_file"] += 1
            continue
        # Reference arms replay human labels verbatim, so their pairs teach the
        # matcher nothing about paraphrase and would pad the harvest with
        # trivial positives.
        if payload.get("tool") == "reference":
            skipped["reference_arm"] += 1
            continue
        if payload.get("ok") is False:
            skipped["failed_run"] += 1
            continue
        dataset = str(payload.get("dataset") or "")
        arm = str(payload.get("arm") or "")
        if dataset not in truth:
            skipped["no_ground_truth_for_dataset"] += 1
            continue
        allowed = None if scope is None else scope.get(dataset, set())

        for episode, predicted in _coerce_episode_map(payload.get("episodes")):
            if allowed is not None and episode not in allowed:
                continue
            reference = truth[dataset].get(episode)
            if not reference:
                skipped["episode_without_ground_truth"] += 1
                continue
            try:
                validate_segmentation(reference, name="reference")
                validate_segmentation(predicted, name="predicted")
            except ValueError:
                skipped["malformed_segmentation"] += 1
                continue

            for i, j in matched_iou(predicted, reference)["pairs"]:
                overlap = iou(predicted[i], reference[j])
                if overlap <= min_iou:
                    skipped["pair_below_min_iou"] += 1
                    continue
                model_label = predicted[i]["text"]
                human_label = reference[j]["text"]
                if not normalise_label(model_label) or not normalise_label(human_label):
                    skipped["empty_label"] += 1
                    continue
                if normalise_label(model_label) == normalise_label(human_label):
                    # Nothing to adjudicate, and calibrate_matcher's automatic
                    # positives are already made of exactly these.
                    skipped["identical_after_normalisation"] += 1
                    continue

                identifier = pair_id(model_label, human_label)
                record = pairs.get(identifier)
                if record is None:
                    # First occurrence in a deterministic traversal fixes the
                    # representative wording and the model/human orientation.
                    # Later occurrences only accumulate weight and provenance.
                    pairs[identifier] = {
                        "pair_id": identifier,
                        "model_label": model_label,
                        "human_label": human_label,
                        "occurrences": 1,
                        "datasets": {dataset},
                        "arms": {arm},
                        "first_dataset": dataset,
                        "first_episode": episode,
                        "max_iou": overlap,
                    }
                else:
                    record["occurrences"] += 1
                    record["datasets"].add(dataset)
                    record["arms"].add(arm)
                    record["max_iou"] = max(record["max_iou"], overlap)

    return pairs, skipped


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------

def stratify(similarity: float, threshold: float, band: float) -> str:
    if abs(similarity - threshold) <= band:
        return NEAR
    return CONTROL_MATCH if similarity > threshold else CONTROL_MISMATCH


def round_robin_by_dataset(
    records: list[dict[str, Any]], sort_key, limit: int
) -> list[dict[str, Any]]:
    """Take ``limit`` records, one dataset at a time, best-first within each.

    Straight global ranking lets one component supply the whole budget, and a
    threshold validated on one component's vocabulary is a threshold validated
    on one script. §8.1 makes the component the unit of clustering for the same
    reason, so the harvest spreads across components too.
    """
    if limit <= 0:
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["first_dataset"]].append(record)
    queues = [sorted(group, key=sort_key) for _, group in sorted(grouped.items())]
    out: list[dict[str, Any]] = []
    index = 0
    while len(out) < limit and any(queues):
        progressed = False
        for queue in queues:
            if index < len(queue):
                out.append(queue[index])
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
        index += 1
    return out


def select(
    pairs: dict[str, dict[str, Any]],
    threshold: float,
    band: float,
    n_near: int,
    n_control: int,
) -> list[dict[str, Any]]:
    by_stratum: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pairs.values():
        by_stratum[record["stratum"]].append(record)

    chosen = round_robin_by_dataset(
        by_stratum[NEAR],
        # Closest to the flip point first; frequent pairs break ties, because a
        # wording that occurs four hundred times moves more metric mass than one
        # that occurs once. The id makes the order total.
        lambda r: (abs(r["similarity"] - threshold), -r["occurrences"], r["pair_id"]),
        n_near,
    )
    chosen += round_robin_by_dataset(
        by_stratum[CONTROL_MATCH],
        lambda r: (-r["similarity"], -r["occurrences"], r["pair_id"]),
        n_control,
    )
    chosen += round_robin_by_dataset(
        by_stratum[CONTROL_MISMATCH],
        lambda r: (r["similarity"], -r["occurrences"], r["pair_id"]),
        n_control,
    )
    return chosen


def blind(record: dict[str, Any]) -> tuple[str, str, str]:
    """Present the pair in a hash-determined order. Returns (a, b, model_side)."""
    flip = int(record["pair_id"][:8], 16) % 2 == 1
    if flip:
        return record["human_label"], record["model_label"], "b"
    return record["model_label"], record["human_label"], "a"


# --------------------------------------------------------------------------
# harvest
# --------------------------------------------------------------------------

def build_matcher(backend: str, model: str, threshold: float | None):
    matcher = EmbeddingMatcher(model_name=model) if backend == "embedding" else TokenF1Matcher()
    if threshold is not None:
        matcher.threshold = threshold
    return matcher


def threshold_from_calibration(path: Path, backend: str) -> float:
    """Read the selected threshold, refusing a file calibrated for another backend.

    Cosine and bag-of-words similarities live on incomparable scales, so
    sampling a token_f1 band around an embedding threshold would centre the
    harvest nowhere in particular while still looking calibrated.
    """
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("backend") != backend:
        raise SystemExit(
            f"{path} was calibrated for backend {calibration.get('backend')!r}, "
            f"not {backend!r}; its threshold does not transfer"
        )
    return float(calibration["selected_threshold"])


def run_harvest(args: argparse.Namespace) -> int:
    # Checked before any work, not just before the write: an operator who
    # re-runs the harvest over a file someone has been filling in should be
    # told immediately, not after the encoder has loaded.
    _refuse_to_clobber(args.out, args.force)
    truth = load_ground_truth(args.gt_dir)
    if not truth:
        raise SystemExit(f"no ground truth under {args.gt_dir}")
    prediction_files = sorted(args.predictions_dir.glob("*.json"))
    if not prediction_files:
        raise SystemExit(f"no predictions under {args.predictions_dir}")

    threshold = args.threshold
    if args.calibration is not None:
        threshold = threshold_from_calibration(args.calibration, args.backend)
    matcher = build_matcher(args.backend, args.model, threshold)
    threshold = float(matcher.threshold)

    scope = load_scope(args.splits_dir, args.scope)
    pairs, skipped = collect_candidates(prediction_files, truth, scope, args.min_iou)
    if not pairs:
        raise SystemExit(
            "no candidate pairs. Either the prediction files hold no VLM arm, or every "
            "matched label pair was identical to the human label after normalisation."
        )

    if isinstance(matcher, EmbeddingMatcher):
        matcher.warm(
            [x for r in pairs.values() for x in (r["model_label"], r["human_label"])]
        )
    for record in pairs.values():
        record["similarity"] = matcher.similarity(record["model_label"], record["human_label"])
        record["stratum"] = stratify(record["similarity"], threshold, args.band)

    chosen = select(pairs, threshold, args.band, args.n_near, args.n_control)
    if not chosen:
        raise SystemExit("selection produced no pairs; widen --band or raise --n-near")
    chosen.sort(key=lambda r: r["pair_id"])

    review_pairs = []
    key_pairs: dict[str, Any] = {}
    for record in chosen:
        label_a, label_b, model_side = blind(record)
        review_pairs.append(
            {"pair_id": record["pair_id"], "label_a": label_a, "label_b": label_b, "verdict": ""}
        )
        key_pairs[record["pair_id"]] = {
            "model_label": record["model_label"],
            "human_label": record["human_label"],
            "model_side": model_side,
            "similarity": record["similarity"],
            "matcher_verdict": SAME if record["similarity"] >= threshold else DIFFERENT,
            "stratum": record["stratum"],
            "occurrences": record["occurrences"],
            "datasets": sorted(record["datasets"]),
            "arms": sorted(record["arms"]),
            "max_iou": record["max_iou"],
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "schema": REVIEW_SCHEMA,
                "question": QUESTION,
                "instructions": list(INSTRUCTIONS),
                "verdict_values": list(VERDICTS),
                "adjudicator": "",
                "n_pairs": len(review_pairs),
                "pairs": review_pairs,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    counts = Counter(r["stratum"] for r in chosen)
    args.key.parent.mkdir(parents=True, exist_ok=True)
    args.key.write_text(
        json.dumps(
            {
                "schema": KEY_SCHEMA,
                "warning": "ANSWER KEY -- do not show this to the adjudicator.",
                "review_file": str(args.out),
                "backend": args.backend,
                "model": args.model if args.backend == "embedding" else None,
                "threshold": threshold,
                "threshold_source": str(args.calibration) if args.calibration else "flag/default",
                "band": args.band,
                "min_iou": args.min_iou,
                "scope": args.scope,
                "n_candidate_pairs": len(pairs),
                "n_selected": len(chosen),
                "selected_by_stratum": dict(sorted(counts.items())),
                "candidates_by_stratum": dict(
                    sorted(Counter(r["stratum"] for r in pairs.values()).items())
                ),
                "skipped": dict(sorted(skipped.items())),
                "prediction_files": [p.name for p in prediction_files],
                "pairs": key_pairs,
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    print(
        f"[harvest] {len(prediction_files)} prediction files -> {len(pairs)} distinct "
        f"candidate pairs; selected {len(chosen)} at threshold={threshold:.3f} "
        f"band=+-{args.band:.3f}"
    )
    for stratum in (NEAR, CONTROL_MATCH, CONTROL_MISMATCH):
        print(f"[harvest]   {stratum:18s} {counts.get(stratum, 0):4d}")
    if skipped:
        print("[harvest] skipped: " + ", ".join(f"{k}={v}" for k, v in sorted(skipped.items())))
    print(f"[harvest] review file -> {args.out}")
    print(f"[harvest] answer key  -> {args.key}   (do not open before adjudicating)")
    return 0


def _refuse_to_clobber(path: Path, force: bool) -> None:
    """Never silently destroy adjudication work by re-running the harvest."""
    if force or not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
        filled = sum(1 for p in existing.get("pairs", []) if str(p.get("verdict", "")).strip())
    except (json.JSONDecodeError, AttributeError, TypeError):
        filled = 0
    if filled:
        raise SystemExit(
            f"{path} already carries {filled} filled verdict(s). Refusing to overwrite "
            "hand-adjudicated work; write elsewhere or pass --force."
        )


# --------------------------------------------------------------------------
# validation of a completed review file
# --------------------------------------------------------------------------

def cohen_kappa(left: list[str], right: list[str]) -> float | None:
    """Chance-corrected agreement, or ``None`` where it is undefined.

    Reported next to raw agreement, never instead of it: with a skewed verdict
    distribution -- which this harvest guarantees, since two thirds of the file
    is near-threshold -- kappa is low even when agreement is high, and quoting
    either alone misleads in a different direction.
    """
    n = len(left)
    if n == 0:
        return None
    observed = sum(1 for a, b in zip(left, right, strict=True) if a == b) / n
    counts_left, counts_right = Counter(left), Counter(right)
    expected = sum(
        (counts_left[c] / n) * (counts_right[c] / n) for c in set(left) | set(right)
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def read_review(path: Path, key_pairs: dict[str, Any]) -> tuple[str, dict[str, str]]:
    """Parse one filled review file, verifying it matches the key it claims."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path}: not valid JSON after editing ({exc})") from exc
    if payload.get("schema") != REVIEW_SCHEMA:
        raise SystemExit(f"{path}: not a {REVIEW_SCHEMA} file")

    adjudicator = str(payload.get("adjudicator") or "").strip() or path.stem
    verdicts: dict[str, str] = {}
    problems: list[str] = []
    for entry in payload.get("pairs") or []:
        identifier = str(entry.get("pair_id", ""))
        if identifier not in key_pairs:
            problems.append(f"pair_id {identifier!r} is not in the key file")
            continue
        if identifier in verdicts:
            problems.append(f"pair_id {identifier!r} appears more than once")
            continue
        # A pair whose labels were edited is no longer the pair that was
        # selected, and its recorded similarity no longer describes it.
        if pair_id(str(entry.get("label_a", "")), str(entry.get("label_b", ""))) != identifier:
            problems.append(f"pair_id {identifier!r}: labels were edited")
            continue
        raw = str(entry.get("verdict", "")).strip().casefold()
        if raw and raw not in VERDICTS:
            problems.append(f"pair_id {identifier!r}: verdict {raw!r} is not one of {VERDICTS}")
            continue
        verdicts[identifier] = raw
    if problems:
        for line in problems[:10]:
            print(f"[validate] {path.name}: {line}", file=sys.stderr)
        raise SystemExit(f"{path}: {len(problems)} malformed record(s); not usable")
    return adjudicator, verdicts


def consensus(per_reviewer: dict[str, dict[str, str]], identifiers: list[str]) -> dict[str, Any]:
    """Unanimity or nothing.

    A pair two adjudicators disagree about is a pair whose answer is not known,
    and majority-voting it in would put a contested judgement into the set that
    fixes the threshold. Disagreements are counted and reported instead.
    """
    out: dict[str, Any] = {}
    conflicts: list[str] = []
    for identifier in identifiers:
        filled = [
            verdicts[identifier]
            for verdicts in per_reviewer.values()
            if verdicts.get(identifier)
        ]
        if not filled:
            out[identifier] = None
            continue
        if len(set(filled)) > 1:
            conflicts.append(identifier)
            out[identifier] = None
            continue
        out[identifier] = filled[0]
    return {"verdicts": out, "conflicts": conflicts}


def best_threshold(labelled: list[tuple[float, str]]) -> dict[str, Any]:
    """Sweep the harvested pairs alone. Diagnostic, never authoritative.

    The authoritative sweep is ``calibrate_matcher.py`` over the full pair set;
    this one runs on a sample chosen to sit at the decision boundary, so its
    optimum is fitted to the hardest region of the space and is reported only
    to show whether a separating threshold exists here at all.
    """
    positive = sorted(s for s, v in labelled if v == SAME)
    negative = sorted(s for s, v in labelled if v == DIFFERENT)
    if not positive or not negative:
        return {
            "n_positive": len(positive),
            "n_negative": len(negative),
            "threshold": None,
            "balanced_accuracy": None,
            "separable": None,
            "n_inversions": None,
            "inversion_rate": None,
            "n_ties": None,
            "min_similarity_same": positive[0] if positive else None,
            "max_similarity_different": negative[-1] if negative else None,
        }
    inversions = sum(1 for p in positive for q in negative if p < q)
    ties = sum(1 for p in positive for q in negative if p == q)
    best = {"threshold": None, "balanced_accuracy": -1.0, "true_negative_rate": -1.0}
    for candidate in sorted({s for s, _ in labelled}):
        tpr = sum(1 for s in positive if s >= candidate) / len(positive)
        tnr = sum(1 for s in negative if s < candidate) / len(negative)
        row = {
            "threshold": candidate,
            "balanced_accuracy": (tpr + tnr) / 2,
            "true_negative_rate": tnr,
        }
        if (row["balanced_accuracy"], row["true_negative_rate"]) > (
            best["balanced_accuracy"], best["true_negative_rate"]
        ):
            best = row
    return {
        "n_positive": len(positive),
        "n_negative": len(negative),
        "threshold": best["threshold"],
        "balanced_accuracy": best["balanced_accuracy"],
        "separable": inversions == 0 and ties == 0,
        "n_inversions": inversions,
        "inversion_rate": inversions / (len(positive) * len(negative)),
        "n_ties": ties,
        "min_similarity_same": positive[0],
        "max_similarity_different": negative[-1],
    }


def run_validate(args: argparse.Namespace) -> int:
    key = json.loads(args.key.read_text(encoding="utf-8"))
    if key.get("schema") != KEY_SCHEMA:
        raise SystemExit(f"{args.key}: not a {KEY_SCHEMA} file")
    key_pairs: dict[str, Any] = key["pairs"]
    identifiers = sorted(key_pairs)
    threshold = float(key["threshold"])

    per_reviewer: dict[str, dict[str, str]] = {}
    for path in args.review:
        name, verdicts = read_review(path, key_pairs)
        if name in per_reviewer:
            raise SystemExit(f"two review files claim adjudicator {name!r}; rename one")
        per_reviewer[name] = verdicts
    reviewers = sorted(per_reviewer)

    agreed = consensus(per_reviewer, identifiers)
    verdicts = agreed["verdicts"]

    inter_reviewer = []
    for i, left in enumerate(reviewers):
        for right in reviewers[i + 1:]:
            common = [
                identifier for identifier in identifiers
                if per_reviewer[left].get(identifier) and per_reviewer[right].get(identifier)
            ]
            a = [per_reviewer[left][identifier] for identifier in common]
            b = [per_reviewer[right][identifier] for identifier in common]
            inter_reviewer.append({
                "a": left,
                "b": right,
                "n_common": len(common),
                "percent_agreement": (
                    sum(1 for x, y in zip(a, b, strict=True) if x == y) / len(common)
                    if common else None
                ),
                "cohen_kappa": cohen_kappa(a, b),
            })

    # Controls: agreement with the matcher where the matcher is near-certain.
    control_total = control_hits = 0
    control_by_stratum: dict[str, dict[str, int]] = {
        CONTROL_MATCH: {"n": 0, "agree": 0}, CONTROL_MISMATCH: {"n": 0, "agree": 0}
    }
    control_failures: list[dict[str, Any]] = []
    for identifier in identifiers:
        stratum = key_pairs[identifier]["stratum"]
        if stratum not in CONTROL_EXPECTED:
            continue
        verdict = verdicts.get(identifier)
        if verdict in (None, UNSURE):
            continue
        expected = CONTROL_EXPECTED[stratum]
        control_total += 1
        control_by_stratum[stratum]["n"] += 1
        if verdict == expected:
            control_hits += 1
            control_by_stratum[stratum]["agree"] += 1
        else:
            control_failures.append({
                "pair_id": identifier,
                "model_label": key_pairs[identifier]["model_label"],
                "human_label": key_pairs[identifier]["human_label"],
                "similarity": key_pairs[identifier]["similarity"],
                "matcher_verdict": expected,
                "human_verdict": verdict,
            })
    control_accuracy = control_hits / control_total if control_total else None

    # Matcher-vs-human on the pairs that actually decide the threshold.
    by_stratum: dict[str, dict[str, Any]] = {}
    for stratum in (NEAR, CONTROL_MATCH, CONTROL_MISMATCH):
        rows = [
            (key_pairs[i], verdicts[i]) for i in identifiers
            if key_pairs[i]["stratum"] == stratum and verdicts.get(i) in (SAME, DIFFERENT)
        ]
        if not rows:
            continue
        false_match = sum(
            1 for r, v in rows if r["matcher_verdict"] == SAME and v == DIFFERENT
        )
        false_split = sum(
            1 for r, v in rows if r["matcher_verdict"] == DIFFERENT and v == SAME
        )
        by_stratum[stratum] = {
            "n_adjudicated": len(rows),
            "agreement_at_current_threshold": (len(rows) - false_match - false_split) / len(rows),
            "matcher_says_same_human_says_different": false_match,
            "matcher_says_different_human_says_same": false_split,
        }

    near_labelled = [
        (float(key_pairs[i]["similarity"]), verdicts[i]) for i in identifiers
        if key_pairs[i]["stratum"] == NEAR and verdicts.get(i) in (SAME, DIFFERENT)
    ]
    separability = best_threshold(near_labelled)

    # Export. Controls are withheld by default: they were chosen because the
    # matcher is confident about them, so returning them to the sweep rewards
    # the matcher for the cases it already gets right.
    exported = [
        i for i in identifiers
        if verdicts.get(i) in (SAME, DIFFERENT)
        and (args.include_controls or key_pairs[i]["stratum"] == NEAR)
    ]
    positive = [
        [key_pairs[i]["model_label"], key_pairs[i]["human_label"]]
        for i in exported if verdicts[i] == SAME
    ]
    negative = [
        [key_pairs[i]["model_label"], key_pairs[i]["human_label"]]
        for i in exported if verdicts[i] == DIFFERENT
    ]

    counts = {
        "n_pairs": len(identifiers),
        "n_with_consensus": sum(1 for v in verdicts.values() if v),
        "n_unfilled": sum(1 for v in verdicts.values() if v is None) - len(agreed["conflicts"]),
        "n_conflicting": len(agreed["conflicts"]),
        "n_same": sum(1 for v in verdicts.values() if v == SAME),
        "n_different": sum(1 for v in verdicts.values() if v == DIFFERENT),
        "n_unsure": sum(1 for v in verdicts.values() if v == UNSURE),
    }

    report = {
        "key_file": str(args.key),
        "review_files": [str(p) for p in args.review],
        "reviewers": reviewers,
        "backend": key.get("backend"),
        "model": key.get("model"),
        "threshold": threshold,
        "band": key.get("band"),
        "counts": counts,
        "inter_reviewer": inter_reviewer,
        "control_accuracy": control_accuracy,
        "control_n": control_total,
        "control_by_stratum": {
            k: {**v, "accuracy": (v["agree"] / v["n"] if v["n"] else None)}
            for k, v in control_by_stratum.items()
        },
        "control_disagreements": control_failures,
        "matcher_vs_human": by_stratum,
        "near_band_separability": separability,
        "exported": {
            "include_controls": bool(args.include_controls),
            "n_positive": len(positive),
            "n_negative": len(negative),
            "path": str(args.out),
        },
        "unsure_pairs": [
            {
                "model_label": key_pairs[i]["model_label"],
                "human_label": key_pairs[i]["human_label"],
                "similarity": key_pairs[i]["similarity"],
            }
            for i in identifiers if verdicts.get(i) == UNSURE
        ],
        "conflicting_pairs": [
            {
                "model_label": key_pairs[i]["model_label"],
                "human_label": key_pairs[i]["human_label"],
                "verdicts": {
                    name: per_reviewer[name][i]
                    for name in reviewers if per_reviewer[name].get(i)
                },
            }
            for i in agreed["conflicts"]
        ],
        "interpretation": (
            "The exported pairs are matcher-selected hard cases. Balanced accuracy "
            "reported by calibrate_matcher.py after ingesting them is a LOWER BOUND, "
            "not an unbiased estimate. The unbiased figures here are "
            "matcher_vs_human.near_threshold and near_band_separability."
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "positive": positive,
                "negative": negative,
                "_meta": {
                    "generated_by": "evaluation/scripts/harvest_pairs.py validate",
                    "key_file": str(args.key),
                    "review_files": [str(p) for p in args.review],
                    "reviewers": reviewers,
                    "include_controls": bool(args.include_controls),
                    "control_accuracy": control_accuracy,
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    report_path = args.report or args.out.with_name(args.out.stem + "_report.json")
    report_path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(
        f"[validate] {len(reviewers)} adjudicator(s), {counts['n_with_consensus']}/"
        f"{counts['n_pairs']} pairs decided "
        f"({counts['n_same']} same, {counts['n_different']} different, "
        f"{counts['n_unsure']} unsure, {counts['n_conflicting']} conflicting, "
        f"{counts['n_unfilled']} blank)"
    )
    for row in inter_reviewer:
        agreement = "n/a" if row["percent_agreement"] is None else f"{row['percent_agreement']:.3f}"
        kappa = "undefined" if row["cohen_kappa"] is None else f"{row['cohen_kappa']:.3f}"
        print(
            f"[validate]   {row['a']} vs {row['b']}: n={row['n_common']} "
            f"agreement={agreement} kappa={kappa}"
        )
    if control_accuracy is None:
        print("[validate] no controls adjudicated; adjudicator accuracy is unchecked")
    else:
        print(f"[validate] control accuracy {control_accuracy:.3f} on {control_total} controls")
    for stratum, row in sorted(by_stratum.items()):
        print(
            f"[validate]   {stratum:18s} n={row['n_adjudicated']:3d} "
            f"agreement_at_{threshold:.2f}={row['agreement_at_current_threshold']:.3f} "
            f"(matcher over-matches {row['matcher_says_same_human_says_different']}, "
            f"under-matches {row['matcher_says_different_human_says_same']})"
        )
    if separability["threshold"] is not None:
        print(
            f"[validate] near band: separable={separability['separable']} "
            f"inversions={separability['n_inversions']}/"
            f"{separability['n_positive'] * separability['n_negative']} "
            f"best-on-harvest threshold={separability['threshold']:.3f} "
            f"balacc={separability['balanced_accuracy']:.3f} (diagnostic only)"
        )
    else:
        print(
            "[validate] near band has verdicts of only one class; no separability "
            "statement is possible from this harvest"
        )
    print(
        f"[validate] {len(positive)} positive / {len(negative)} negative pairs -> {args.out}\n"
        f"[validate] report -> {report_path}\n"
        f"[validate] next: calibrate_matcher.py --extra-pairs {args.out}"
    )

    # The summary above is the context for any failure below it, and stdout is
    # block-buffered when piped while stderr is not.
    sys.stdout.flush()
    if control_accuracy is not None and control_accuracy < args.min_control_accuracy:
        print(
            f"[validate] FAIL: control accuracy {control_accuracy:.3f} is below "
            f"--min-control-accuracy={args.min_control_accuracy}. These verdicts are not "
            "reliable enough to fix a threshold every label-aware metric depends on; "
            "re-adjudicate before using the exported pairs.",
            file=sys.stderr,
        )
        return 1
    if counts["n_with_consensus"] < args.min_decided:
        print(
            f"[validate] FAIL: only {counts['n_with_consensus']} pairs decided, "
            f"--min-decided={args.min_decided}.",
            file=sys.stderr,
        )
        return 1
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    harvest = sub.add_parser("harvest", allow_abbrev=False, help="build a blank review file from predictions")
    harvest.add_argument("--predictions-dir", type=Path, required=True)
    harvest.add_argument("--gt-dir", type=Path, required=True)
    harvest.add_argument("--out", type=Path, required=True, help="review file to write")
    harvest.add_argument("--key", type=Path, required=True,
                         help="answer key; needed by `validate`, withheld from the adjudicator")
    harvest.add_argument("--splits-dir", type=Path, default=None)
    harvest.add_argument("--scope", default="all", choices=["all", "eval", "seed"])
    # `exact` is deliberately absent: its similarity is 0 or 1, so there is no
    # near band to sample and the whole selection collapses.
    harvest.add_argument("--backend", default="embedding", choices=["embedding", "token_f1"])
    harvest.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    harvest.add_argument("--calibration", type=Path, default=None,
                         help="matcher_calibration.json; its selected_threshold "
                              "wins over --threshold")
    harvest.add_argument("--threshold", type=float, default=None,
                         help="decision threshold to sample around (default: the backend's)")
    harvest.add_argument("--band", type=float, default=0.05,
                         help="half-width of the near-threshold band")
    harvest.add_argument("--n-near", type=int, default=60)
    harvest.add_argument("--n-control", type=int, default=15,
                         help="controls of EACH kind (clear match, clear mismatch)")
    harvest.add_argument("--min-iou", type=float, default=0.0,
                         help="keep matched spans whose temporal IoU exceeds this")
    harvest.add_argument("--force", action="store_true",
                         help="overwrite a review file that already has verdicts in it")

    validate = sub.add_parser("validate", allow_abbrev=False, help="turn a filled review file into --extra-pairs")
    validate.add_argument("--review", type=Path, action="append", required=True,
                          help="filled review file; repeat for several adjudicators")
    validate.add_argument("--key", type=Path, required=True)
    validate.add_argument("--out", type=Path, required=True,
                          help="--extra-pairs JSON for calibrate_matcher.py")
    validate.add_argument("--report", type=Path, default=None)
    validate.add_argument("--min-control-accuracy", type=float, default=0.90)
    validate.add_argument("--min-decided", type=int, default=1)
    validate.add_argument("--include-controls", action="store_true",
                          help="also export the control pairs; biases the sweep, see the docstring")

    args = parser.parse_args()
    return run_harvest(args) if args.command == "harvest" else run_validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
