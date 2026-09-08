#!/usr/bin/env python
"""Translate the Corpus B seed/eval splits from ORIGINAL to renumbered indices.

This is the single point where the two index spaces of the Corpus B pipeline
meet, and getting it wrong is silent: every artefact downstream would key on
plausible-looking integers that name the wrong episodes, and the study would
report confident numbers about trajectories it never scored.

  * `select_corpus_b.py` emits splits in ORIGINAL RoboInter `episode_index`,
    because that is what the subset builder consumes.
  * `build_corpus_b_subset.py` renumbers to 0..N-1 -- forced by the v2.1 -> v3.0
    converter, which numbers positionally -- and writes `index_map.json`.
  * Ground truth, labels, calibration and predictions all key on the NEW index.

So this script consumes both and emits splits in the new space, with the
translation asserted rather than assumed:

  1. every original index in the splits appears in the index map;
  2. the map is injective, so no two originals collapse onto one new index;
  3. seed and eval stay disjoint after translation, per task and globally;
  4. every translated index exists in the ground-truth file;
  5. the round trip new -> original recovers the input exactly.

Any failure is fatal. There is no partial translation, because a partially
translated split is indistinguishable from a correct one downstream.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--splits", type=Path, required=True, help="selection splits.json (original indices)")
    parser.add_argument("--index-map", type=Path, required=True, help="<subset>.index_map.json")
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    splits_payload = json.loads(args.splits.read_text(encoding="utf-8"))
    tasks = splits_payload["tasks"]
    entries = json.loads(args.index_map.read_text(encoding="utf-8"))["episodes"]
    truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))["episodes"]

    forward: dict[int, int] = {}
    for entry in entries:
        original, new = int(entry["original_index"]), int(entry["new_index"])
        if original in forward:
            raise SystemExit(f"index map lists original episode {original} twice")
        forward[original] = new
    if len(set(forward.values())) != len(forward):
        raise SystemExit("index map is not injective: two originals share a new index")

    translated: dict[str, dict[str, list[int]]] = {}
    seen_new: dict[int, str] = {}
    for task, groups in sorted(tasks.items()):
        row: dict[str, list[int]] = {}
        for group in ("seed", "eval"):
            out: list[int] = []
            for original in groups[group]:
                original = int(original)
                if original not in forward:
                    raise SystemExit(
                        f"{task}/{group}: original episode {original} is not in the index map, so "
                        "it was not built into the subset. Refusing to emit a split that names "
                        "episodes the dataset does not contain."
                    )
                new = forward[original]
                if str(new) not in truth:
                    raise SystemExit(
                        f"{task}/{group}: original {original} -> new {new} has no ground truth. "
                        "A scored episode without spans would be counted as a total miss."
                    )
                if new in seen_new:
                    raise SystemExit(f"new index {new} claimed by both {seen_new[new]} and {task}/{group}")
                seen_new[new] = f"{task}/{group}"
                out.append(new)
            row[group] = sorted(out)
        overlap = set(row["seed"]) & set(row["eval"])
        if overlap:
            raise SystemExit(f"{task}: seed and eval overlap after translation: {sorted(overlap)[:5]}")
        translated[task] = row

    # Round trip, so a subtly wrong map cannot pass the checks above.
    backward = {new: original for original, new in forward.items()}
    for task, groups in translated.items():
        for group in ("seed", "eval"):
            recovered = sorted(backward[n] for n in groups[group])
            if recovered != sorted(int(x) for x in tasks[task][group]):
                raise SystemExit(f"{task}/{group}: round trip through the index map did not recover the input")

    all_seed = sorted(n for r in translated.values() for n in r["seed"])
    all_eval = sorted(n for r in translated.values() for n in r["eval"])
    if set(all_seed) & set(all_eval):
        raise SystemExit("seed and eval overlap across tasks after translation")

    # Two files, because two consumers want different shapes and neither should be
    # bent to the other. The harness reads a FLAT {dataset, seed, eval} per dataset
    # -- that is the contract make_label_files.py, align_floors.py and run_arm.py
    # already implement for Corpus A, and Corpus B conforms to it rather than
    # changing working code. The per-task structure is what cohorts and the cluster
    # bootstrap need, and it lives beside it.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dataset = args.out.stem
    args.out.write_text(
        json.dumps({
            "dataset": dataset,
            "seed": all_seed,
            "eval": all_eval,
            "n_available": len(all_seed) + len(all_eval),
            "policy": "corpus_b_plan.md 4.2: 10 lowest-index trajectories per task are seed "
                      "and are never scored; the remainder, capped at 40, is eval.",
            "index_space": "RENUMBERED subset indices (0..N-1), translated via index_map.json.",
        }, indent=1),
        encoding="utf-8",
    )
    tasks_out = args.out.parent / f"{dataset}.tasks.json"
    tasks_out.write_text(
        json.dumps({
            "index_space": "RENUMBERED subset indices (0..N-1). Translated from the original "
                           "RoboInter indices via the subset's index_map.json. Everything "
                           "downstream -- ground truth, labels, calibration, predictions -- "
                           "keys on THESE.",
            "source_splits": str(args.splits),
            "index_map": str(args.index_map),
            "n_tasks": len(translated),
            "n_seed": len(all_seed),
            "n_eval": len(all_eval),
            "tasks": translated,
        }, indent=1),
        encoding="utf-8",
    )
    print(json.dumps({
        "tasks": len(translated), "seed": len(all_seed), "eval": len(all_eval),
        "checks": "map injective, all present in ground truth, seed/eval disjoint, round trip exact",
        "out": str(args.out),
        "tasks_out": str(args.out.parent / f"{args.out.stem}.tasks.json"),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
