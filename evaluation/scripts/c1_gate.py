#!/usr/bin/env python
"""Compute the decoding-noise band and decide C1 by equivalence testing.

C1 predicts that `lerobot-align` with every new feature off behaves the same as
upstream. It is the study's gate: two tools sharing prompts, defaults and model
should agree, and if they do not, an uncontrolled confound exists and no
downstream number means anything.

The trap this script exists to avoid is deciding C1 by *failing to reject*.
With three components, non-rejection is mostly a statement about power, and
"p > 0.05, therefore identical" would reward an imprecise experiment. So C1 is
decided by two one-sided tests against a margin that is itself measured rather
than chosen:

    margin = the upper bound of the 95% interval on the WITHIN-ARM,
             between-repeat difference for the baseline

Neither tool seeds its generation call, so running the same arm twice already
produces a difference of that size. Anything smaller cannot be attributed to a
mechanism, by either tool. The same margin then floors every C2/C3 effect.

Outcomes: EQUIVALENT (proceed), DIFFERENT (stop and find the confound), or
INCONCLUSIVE (too imprecise to claim either -- also blocks, because an
imprecise C1 is not a passing C1).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Both the evaluation package root (for `metrics.*`) and this directory (for the
# sibling `gate_fingerprint`). Relying on Python adding the script's own
# directory works only when this file is executed directly as a path, and
# breaks under `-m`, under importlib, and when imported from a test.
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import jsonl_io  # noqa: E402
from gate_fingerprint import compute as compute_fingerprint  # noqa: E402
from gate_fingerprint import PROVENANCE_KEYS, source_hashes  # noqa: E402

from metrics.stats import decoding_noise_band, equivalence_test  # noqa: E402

HEADLINE = ("boundary_f1@0p5", "boundary_f1@1", "macro_iou")


def paired_episode_deltas(rows: list[dict], metric: str, arm: str, baseline: str):
    """Average paired repeat differences within each episode before resampling."""
    index: dict[tuple, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["arm"] not in (arm, baseline) or row.get(metric) is None:
            continue
        key = (row["dataset"], row["episode"], int(row.get("repeat", 0) or 0))
        if row["arm"] in index[key]:
            raise ValueError(f"duplicate score for {key}, {row['arm']}")
        index[key][row["arm"]] = float(row[metric])
    episodes: dict[tuple, list[float]] = defaultdict(list)
    unpaired: set[tuple] = set()
    n_pairs = 0
    for (dataset, episode, _repeat), values in sorted(index.items()):
        if arm in values and baseline in values:
            episodes[dataset, episode].append(values[arm] - values[baseline])
            n_pairs += 1
        else:
            unpaired.add((dataset, episode))
    deltas: dict[str, list[float]] = defaultdict(list)
    for (dataset, _episode), values in sorted(episodes.items()):
        deltas[dataset].append(sum(values) / len(values))
    return dict(deltas), len(unpaired), n_pairs


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matcher", default="embedding")
    parser.add_argument("--baseline", default="baseline_upstream")
    parser.add_argument("--arm", default="align_defaults")
    parser.add_argument("--metrics", nargs="*", default=list(HEADLINE))
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--model-id", default="Qwen/Qwen3.8-27B",
                        help="Recorded in the verdict's fingerprint; a later sweep against a "
                             "different model must not reuse this verdict.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Matcher threshold used, recorded in the fingerprint.")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument(
        "--max-failure-rate", type=float, default=0.05,
        help="Refuse to certify equivalence if either arm failed more than this fraction "
             "of episodes. A tool that cannot annotate the corpus has not been shown "
             "equivalent to one that can.",
    )
    parser.add_argument(
        "--max-attrition-gap", type=float, default=0.02,
        help="Refuse to certify equivalence if the two arms' failure rates differ by more "
             "than this. Differential attrition makes the surviving paired episodes an "
             "easy subset and can manufacture equivalence.",
    )
    args = parser.parse_args()

    all_rows = [
        json.loads(line)
        for line in jsonl_io.read_text(args.scores).splitlines()
        if line.strip()
    ]
    # The saved file also contains the main sweep (repeat 0). E1 is the
    # positive-index repeat experiment for THIS pair, not the full sweep.
    all_rows = [r for r in all_rows if r.get("matcher") == args.matcher
                and r["arm"] in (args.baseline, args.arm)
                and int(r.get("repeat", 0) or 0) > 0]
    rows = [r for r in all_rows if r.get("ok")]
    if not rows:
        raise SystemExit(f"no usable rows for matcher {args.matcher}")

    # A verdict must describe the code that actually produced the predictions.
    # run_arm.py stamps each prediction with a content hash of both tools'
    # source; if any contributing row was produced by different source than is
    # on disk now, it came from a cache written by other code and certifying it
    # would attach a current-looking fingerprint to stale outputs.
    full_state = source_hashes()
    current_source = {k: full_state[k] for k in PROVENANCE_KEYS}
    observed: dict[str, set[str]] = {k: set() for k in current_source}
    unstamped = 0
    incomplete: dict[str, int] = dict.fromkeys(current_source, 0)
    for row in all_rows:
        if row["arm"] not in (args.baseline, args.arm):
            continue
        if all(row.get(k) is None for k in current_source):
            unstamped += 1
            continue
        for key in current_source:
            if row.get(key) is None:
                # An absent key is NOT a pass. Treating it as "nothing to
                # compare" is exactly how prompt-override provenance went
                # unverified: the observed set stayed empty and the mismatch
                # branch never ran.
                incomplete[key] += 1
            else:
                observed[key].add(str(row[key]))
    source_problems: list[str] = []
    if unstamped:
        source_problems.append(
            f"{unstamped} row(s) carry no source stamp; they predate provenance "
            "tracking and cannot be shown to come from the current code"
        )
    for key, count in incomplete.items():
        if count:
            source_problems.append(
                f"{key}: {count} row(s) carry no value for this field, so the predictions "
                "cannot be attributed to the current configuration"
            )
    for key, values in observed.items():
        if len(values) > 1:
            source_problems.append(f"{key}: predictions mix {sorted(values)}")
        elif values and next(iter(values)) != current_source[key]:
            source_problems.append(
                f"{key}: predictions were produced with {next(iter(values))!r}, "
                f"but {current_source[key]!r} is in effect now"
            )

    # Differential attrition can manufacture equivalence. If one arm crashes on
    # the hard episodes, only the easy ones survive into the paired comparison,
    # and two arms that behave very differently look identical on the remainder.
    # Dropping failures is exactly how an equivalence claim gets rigged without
    # anyone intending it, so attrition is measured first and can block the
    # verdict on its own.
    attrition: dict[str, dict[str, float]] = {}
    for arm in (args.baseline, args.arm):
        total = sum(1 for r in all_rows if r["arm"] == arm)
        failed = sum(1 for r in all_rows if r["arm"] == arm and not r.get("ok"))
        attrition[arm] = {
            "episodes_attempted": float(total),
            "episodes_failed": float(failed),
            "failure_rate": (failed / total) if total else 0.0,
        }
    baseline_rate = attrition[args.baseline]["failure_rate"]
    arm_rate = attrition[args.arm]["failure_rate"]
    attrition_gap = abs(arm_rate - baseline_rate)
    attrition_blocks = (
        max(arm_rate, baseline_rate) > args.max_failure_rate
        or attrition_gap > args.max_attrition_gap
    )

    repeats = sorted({int(r.get("repeat", 0) or 0) for r in rows})
    if len(repeats) < 2:
        raise SystemExit(
            f"found repeat indices {repeats}; C1 needs at least two repeats of "
            f"'{args.baseline}' to measure the decoding-noise margin. Run "
            "`run_all.sh falsify` first."
        )

    report: dict = {
        "matcher": args.matcher,
        "baseline": args.baseline,
        "arm": args.arm,
        "repeats_found": repeats,
        "alpha": args.alpha,
        "resampling_unit": "dataset, then episode mean of paired repeat differences",
        # A PASS is a statement about THIS configuration. Recording what it was
        # certified against is what lets a later sweep detect that the verdict
        # has gone stale rather than silently inheriting it.
        "environment_fingerprint": compute_fingerprint(
            model_id=args.model_id, matcher=args.matcher, threshold=args.threshold,
            baseline=args.baseline, arm=args.arm, metrics=list(args.metrics),
        ),
        "source_of_predictions": {k: sorted(v) for k, v in observed.items()},
        "source_problems": source_problems,
        "attrition": attrition,
        "attrition_gap": attrition_gap,
        "attrition_blocks": attrition_blocks,
        "metrics": {},
    }
    verdicts = []

    if source_problems:
        for line in source_problems:
            print(f"SOURCE MISMATCH: {line}", file=sys.stderr)

    if attrition_blocks:
        print(
            f"ATTRITION: {args.baseline} failed {baseline_rate:.1%} of episodes, "
            f"{args.arm} failed {arm_rate:.1%} (gap {attrition_gap:.1%}). "
            f"Limits: max {args.max_failure_rate:.1%} per arm, "
            f"{args.max_attrition_gap:.1%} gap.",
            file=sys.stderr,
        )

    for metric in args.metrics:
        # Margin: how much the BASELINE disagrees with itself across repeats.
        noise: dict[str, dict[int, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
        for row in rows:
            if row["arm"] != args.baseline or metric not in row or row[metric] is None:
                continue
            repeat_index = int(row.get("repeat", 0) or 0)
            noise[row["dataset"]][repeat_index][int(row["episode"])] = float(row[metric])
        band = decoding_noise_band(
            {k: {r: dict(e) for r, e in v.items()} for k, v in noise.items()},
            n_boot=args.n_boot, seed=11,
        )
        margin = band.high
        if not math.isfinite(margin) or margin <= 0:
            report["metrics"][metric] = {
                "decoding_noise_band": band.as_dict(),
                "error": "no usable decoding-noise margin: the baseline's repeats share no "
                         "episodes, or all repeats agree exactly. C1 cannot be decided.",
            }
            verdicts.append("INCONCLUSIVE")
            print(f"{metric:20s} margin=UNDEFINED -> INCONCLUSIVE")
            continue

        deltas, unpaired, n_pairs = paired_episode_deltas(
            rows, metric, args.arm, args.baseline
        )

        result = equivalence_test(
            dict(deltas), margin, n_boot=args.n_boot, alpha=args.alpha, seed=13
        )
        # An episode one arm annotated and the other did not is evidence about
        # the arms, not a row to discard. It cannot enter a paired difference,
        # so it is counted and allowed to veto the verdict instead.
        if attrition_blocks and result["verdict"] == "EQUIVALENT":
            result = dict(result)
            result["verdict"] = "INCONCLUSIVE"
            result["reason"] = (
                f"equivalence within the margin, but withheld: the arms lost different "
                f"episodes to failure (gap {attrition_gap:.1%}), so the surviving paired "
                "subset is not a fair sample of the corpus."
            )
        report["metrics"][metric] = {
            "decoding_noise_band": band.as_dict(),
            "margin_used": margin,
            "equivalence": result,
            "n_paired_episodes": sum(len(v) for v in deltas.values()),
            "n_unpaired_episodes": unpaired,
            "n_paired_repeat_observations": n_pairs,
        }
        verdicts.append(result["verdict"])
        print(
            f"{metric:20s} margin={margin:.4f}  "
            f"delta={result.get('point', float('nan')):+.4f} "
            f"[{result.get('ci_low', float('nan')):+.4f},{result.get('ci_high', float('nan')):+.4f}]  "
            f"-> {result['verdict']}"
        )

    overall = (
        "EQUIVALENT" if all(v == "EQUIVALENT" for v in verdicts)
        else "DIFFERENT" if any(v == "DIFFERENT" for v in verdicts)
        else "INCONCLUSIVE"
    )
    report["statistical_verdict"] = overall
    # Whatever the numbers say, a verdict cannot certify outputs it cannot
    # attribute to the current code.
    if source_problems and overall == "EQUIVALENT":
        overall = "INCONCLUSIVE"
        report["withheld_reason"] = (
            "equivalence within the margin, but withheld: the predictions were not "
            "produced by the source now on disk, so this verdict would certify cached "
            "outputs from different code."
        )
    report["verdict"] = overall
    report["gate"] = "PASS" if overall == "EQUIVALENT" else "BLOCK"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(
        f"\nattrition: {args.baseline} {baseline_rate:.1%} failed, "
        f"{args.arm} {arm_rate:.1%} failed, gap {attrition_gap:.1%}"
        + ("  [BLOCKS EQUIVALENCE]" if attrition_blocks else "")
    )
    print(f"C1 verdict: {overall}  ->  gate {report['gate']}")
    if overall == "DIFFERENT":
        print("Two tools sharing prompts, defaults and model disagree. Find the confound "
              "before running anything else.", file=sys.stderr)
    elif overall == "INCONCLUSIVE":
        print("Too imprecise to conclude equivalence. This is NOT a pass: add repeats or "
              "components until the interval fits inside the margin.", file=sys.stderr)
    print(f"wrote {args.out}")
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
