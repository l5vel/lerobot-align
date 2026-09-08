#!/usr/bin/env python
"""Validate and threshold the semantic matcher before it is used to score anything.

A label matcher that calls "open the fridge door" and "close the fridge door"
equivalent would silently inflate every label-aware metric for every arm, and
would do so asymmetrically: whichever tool produces vaguer labels benefits
most. So the matcher is validated first, against pairs whose answer we already
know, and the threshold is *chosen* by that validation rather than guessed.

Two pair sets are built automatically from the ground truth itself, so this
needs no hand-labelling and re-derives itself for any new corpus:

**Positives** — the same human label taken from two different datasets or
written with incidental formatting differences. These must match.

**Hard negatives** — pairs drawn from the SAME dataset's label vocabulary.
Within one task the labels are minimally different by construction ("open the
fridge door" / "close the fridge door", "place the drink on top" / "pick up the
drink on top"), which makes them exactly the adversarial cases that a bag-of-
words or a loosely-thresholded embedding will fail. These must NOT match.

**Known weakness of the automatic positives.** They are built from the human
vocabulary alone, so they only ever test identical or trivially reformatted
labels. The case that actually matters -- a model paraphrase such as "pick up
the red can from the refrigerator" against the human "grab the red drink from
the fridge" -- cannot be constructed until model output exists. This script
therefore establishes only a NECESSARY condition (the matcher must not confuse
minimally-different labels within a task). After the pilot, rerun with
``--extra-pairs`` pointing at a hand-labelled sample of real (model, human)
pairs; the threshold selected here is provisional until that has been done, and
the final report must quote the paraphrase-validated number.

An explicit hand-written antonym set is added on top, because the failure modes
we most need to exclude (open/close, left/right, pick/place, on/in) are known
in advance and should not depend on the corpus happening to contain them.

The script sweeps the threshold, reports the full curve, and selects the value
maximising balanced accuracy. If the best achievable balanced accuracy is below
``--min-accuracy``, it exits non-zero: the matcher is not fit for purpose and
the evaluation must fall back to the LLM judge.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from metrics.semantic import (  # noqa: E402
    EmbeddingMatcher,
    TokenF1Matcher,
    normalise_label,
)

ANTONYMS = [
    ("open the fridge door", "close the fridge door"),
    ("open the drawer", "close the drawer"),
    ("pick up the red can", "put down the red can"),
    ("pick up the cup", "place the cup"),
    ("put the cup on the left shelf", "put the cup on the right shelf"),
    ("move the bag to the left", "move the bag to the right"),
    ("put the block in the box", "put the block on the box"),
    ("push the door", "pull the door"),
    ("pick up the red drink", "pick up the blue drink"),
    ("grab the drink from the fridge", "place the drink on top of the fridge"),
]


def load_vocabularies(gt_dir: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in sorted(gt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        labels = {
            normalise_label(str(span.get("text", "")))
            for spans in payload["episodes"].values()
            for span in spans
        }
        out[payload["dataset"]] = sorted(x for x in labels if x)
    return out


def build_pairs(vocabularies: dict[str, list[str]]) -> tuple[list, list]:
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []

    # Positives: identical labels shared across datasets, plus formatting variants.
    seen: dict[str, list[str]] = {}
    for dataset, labels in vocabularies.items():
        for label in labels:
            seen.setdefault(label, []).append(dataset)
    for label, datasets in seen.items():
        if len(datasets) > 1:
            positives.append((label, label))
        positives.append((label, f"{label}."))
        positives.append((label, label.replace(" the ", " ")))

    # Hard negatives: distinct labels from within one dataset's vocabulary.
    for labels in vocabularies.values():
        for left, right in combinations(labels, 2):
            negatives.append((left, right))

    negatives.extend(ANTONYMS)
    positives = [(a, b) for a, b in positives if a and b]
    negatives = [(a, b) for a, b in negatives if a and b and normalise_label(a) != normalise_label(b)]
    return positives, negatives


def sweep(matcher, positives, negatives, thresholds) -> list[dict]:
    pos = [matcher.similarity(a, b) for a, b in positives]
    neg = [matcher.similarity(a, b) for a, b in negatives]
    rows = []
    for threshold in thresholds:
        tp = sum(1 for s in pos if s >= threshold)
        fn = len(pos) - tp
        fp = sum(1 for s in neg if s >= threshold)
        tn = len(neg) - fp
        tpr = tp / len(pos) if pos else 0.0
        tnr = tn / len(neg) if neg else 0.0
        rows.append({
            "threshold": threshold,
            "true_positive_rate": tpr,
            "true_negative_rate": tnr,
            "balanced_accuracy": (tpr + tnr) / 2,
            "false_positives": fp,
            "false_negatives": fn,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", default="embedding", choices=["embedding", "token_f1"])
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    parser.add_argument(
        "--extra-pairs", type=Path, default=None,
        help="JSON {'positive': [[a,b],...], 'negative': [[a,b],...]} of hand-labelled "
             "(model, human) pairs harvested from the pilot. Supplying this turns the "
             "provisional threshold into a paraphrase-validated one.",
    )
    args = parser.parse_args()

    vocabularies = load_vocabularies(args.gt_dir)
    if not vocabularies:
        raise SystemExit(f"no ground truth under {args.gt_dir}")
    positives, negatives = build_pairs(vocabularies)
    paraphrase_validated = False
    if args.extra_pairs:
        extra = json.loads(args.extra_pairs.read_text(encoding="utf-8"))
        positives.extend(tuple(pair) for pair in extra.get("positive", []))
        negatives.extend(tuple(pair) for pair in extra.get("negative", []))
        paraphrase_validated = bool(extra.get("positive"))
    print(f"[calibrate] {len(vocabularies)} datasets, "
          f"{len(positives)} positive pairs, {len(negatives)} hard-negative pairs"
          + ("" if paraphrase_validated else "  [PROVISIONAL: no hand-labelled paraphrase pairs]"))

    if args.backend == "embedding":
        matcher = EmbeddingMatcher(model_name=args.model)
        matcher.warm([x for pair in positives + negatives for x in pair])
        thresholds = [round(0.30 + 0.01 * i, 2) for i in range(66)]
    else:
        matcher = TokenF1Matcher()
        thresholds = [round(0.05 * i, 2) for i in range(1, 21)]

    rows = sweep(matcher, positives, negatives, thresholds)
    best = max(rows, key=lambda r: (r["balanced_accuracy"], r["true_negative_rate"]))

    print(f"\n{'thr':>6s} {'TPR':>7s} {'TNR':>7s} {'balacc':>7s} {'FP':>5s} {'FN':>5s}")
    for row in rows:
        mark = " <-- selected" if row is best else ""
        if row["balanced_accuracy"] > 0.5 or mark:
            print(f"{row['threshold']:6.2f} {row['true_positive_rate']:7.3f} "
                  f"{row['true_negative_rate']:7.3f} {row['balanced_accuracy']:7.3f} "
                  f"{row['false_positives']:5d} {row['false_negatives']:5d}{mark}")

    # Surface the negatives that survive the chosen threshold: these are the
    # pairs the metric will wrongly treat as the same action, and they belong
    # in the report rather than in a footnote.
    residual = sorted(
        ((matcher.similarity(a, b), a, b) for a, b in negatives
         if matcher.similarity(a, b) >= best["threshold"]),
        reverse=True,
    )[:15]
    if residual:
        print(f"\n[calibrate] {len(residual)} hard negatives still match at "
              f"{best['threshold']:.2f} (worst first):")
        for score, a, b in residual:
            print(f"    {score:.3f}  {a!r}  ~  {b!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "backend": args.backend,
        "model": args.model if args.backend == "embedding" else None,
        "n_positive_pairs": len(positives),
        "n_negative_pairs": len(negatives),
        "paraphrase_validated": paraphrase_validated,
        "selected_threshold": best["threshold"],
        "selected": best,
        "sweep": rows,
        "residual_false_positives": [
            {"similarity": s, "a": a, "b": b} for s, a, b in residual
        ],
    }, indent=1), encoding="utf-8")

    print(f"\n[calibrate] selected threshold={best['threshold']:.2f} "
          f"balanced_accuracy={best['balanced_accuracy']:.4f} -> {args.out}")
    if best["balanced_accuracy"] < args.min_accuracy:
        print(f"[calibrate] FAIL: below --min-accuracy={args.min_accuracy}. "
              f"This matcher is not fit to score labels; use the LLM judge.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
