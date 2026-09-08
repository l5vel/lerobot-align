#!/usr/bin/env python
"""Re-key task-level ground truth and splits from subset indices to ORIGINAL ones.

`aggregate.py` pairs a contrast on `f"{dataset}/{episode}"`, so two arms of the
same contrast must name the same trajectory by the same integer. The VLM arms
arrive from `merge_corpus_b_predictions.py` keyed on the **original RoboInter
episode_index**; `shard_corpus_b_by_task.py` emits the **renumbered subset
index**, because that is the space its converted root uses.

Left alone, the two disagree silently: every dataset name matches, every episode
id differs, and `bootstrap_paired_difference_by_dataset` finds zero paired
episodes for every floor-versus-VLM contrast -- which is the whole of claim B1.
An empty contrast is not a null result, it is a plumbing failure that looks like
one.

So this translates the floors' inputs BEFORE they are used, rather than
translating their outputs afterwards: an arm that reads re-keyed ground truth
writes predictions in the same space, and nothing downstream has to know.

Seed episodes are kept. The floors fit their prior on the seeds
(`align_floors.py`), and `score_alignment.py` reads only the `eval` list, so
carrying the seeds costs nothing and dropping them would break the fit.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_forward(path: Path) -> dict[int, int]:
    """subset index -> original RoboInter index, asserted injective."""
    entries = json.loads(path.read_text(encoding="utf-8"))["episodes"]
    forward = {int(e["new_index"]): int(e["original_index"]) for e in entries}
    if len(forward) != len(entries) or len(set(forward.values())) != len(entries):
        raise SystemExit(f"{path}: index map is not injective")
    return forward


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--index-map", type=Path, required=True,
                        help="<subset>.index_map.json for the root the shards came from")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--gt-out-dir", type=Path, required=True)
    parser.add_argument("--splits-out-dir", type=Path, required=True)
    args = parser.parse_args()

    forward = load_forward(args.index_map)
    for directory in (args.gt_out_dir, args.splits_out_dir):
        directory.mkdir(parents=True, exist_ok=True)

    def translate(episode: Any, where: str) -> int:
        index = int(episode)
        if index not in forward:
            # Refuse rather than skip: a dropped episode would quietly shrink one
            # arm's population and leave the contrast paired on the remainder.
            raise SystemExit(f"{where}: subset episode {index} is absent from {args.index_map}")
        return forward[index]

    space = "ORIGINAL RoboInter episode_index"
    n_datasets = n_gt = n_seed = n_eval = 0
    for gt_path in sorted(args.gt_dir.glob("*.json")):
        dataset = gt_path.stem
        split_path = args.splits_dir / f"{dataset}.json"
        if not split_path.exists():
            raise SystemExit(f"{dataset}: no split beside the ground truth at {split_path}")

        truth = json.loads(gt_path.read_text(encoding="utf-8"))
        episodes = {
            str(translate(k, f"{dataset} ground truth")): v
            for k, v in truth["episodes"].items()
        }
        if len(episodes) != len(truth["episodes"]):
            raise SystemExit(f"{dataset}: two subset episodes translated to one original index")
        truth["episodes"] = {k: episodes[k] for k in sorted(episodes, key=int)}
        truth["index_space"] = space
        (args.gt_out_dir / gt_path.name).write_text(json.dumps(truth, indent=1), encoding="utf-8")

        split = json.loads(split_path.read_text(encoding="utf-8"))
        for group in ("seed", "eval"):
            split[group] = sorted(translate(e, f"{dataset} {group}") for e in split[group])
        if set(split["seed"]) & set(split["eval"]):
            raise SystemExit(f"{dataset}: seed and eval overlap after translation")
        missing = (set(split["seed"]) | set(split["eval"])) - set(map(int, truth["episodes"]))
        if missing:
            raise SystemExit(f"{dataset}: {len(missing)} split episode(s) have no ground truth")
        split["index_space"] = space
        (args.splits_out_dir / split_path.name).write_text(json.dumps(split, indent=1), encoding="utf-8")

        n_datasets += 1
        n_gt += len(truth["episodes"])
        n_seed += len(split["seed"])
        n_eval += len(split["eval"])

    if not n_datasets:
        raise SystemExit(f"no ground-truth files under {args.gt_dir}")
    print(json.dumps({
        "datasets": n_datasets, "episodes_with_gt": n_gt,
        "seed": n_seed, "eval": n_eval, "index_space": space,
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
