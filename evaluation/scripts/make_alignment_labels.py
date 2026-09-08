#!/usr/bin/env python
"""Emit supplied ordered labels for evaluation-only components in any corpus.

Each episode receives its own ground-truth label texts, without boundary times.
There is no modal/default fallback: missing labels must fail, not silently switch
to label generation. Seed trajectories cannot occur in these evaluation splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", default="oracle")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_episodes = 0
    written = 0
    lengths: dict[int, int] = {}

    for split_path in sorted(args.splits_dir.glob("*.json")):
        dataset = split_path.stem
        gt_path = args.gt_dir / f"{dataset}.json"
        if not gt_path.exists():
            raise SystemExit(f"{dataset}: no ground truth at {gt_path}")
        truth = json.loads(gt_path.read_text(encoding="utf-8"))["episodes"]
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("seed"):
            raise SystemExit(
                f"{dataset}: a cohort dataset must hold no seed episodes, found "
                f"{len(split['seed'])}. Seed episodes fit the calibration and must never "
                "appear in a scored dataset."
            )

        payload: dict[str, list[str]] = {}
        for episode in sorted(int(e) for e in split["eval"]):
            spans = truth.get(str(episode))
            if not spans:
                raise SystemExit(f"{dataset}: eval episode {episode} has no ground-truth spans")
            labels = [str(span["text"]).strip() for span in spans]
            if not all(labels):
                raise SystemExit(f"{dataset}: episode {episode} has an empty span label")
            payload[str(episode)] = labels
            lengths[len(labels)] = lengths.get(len(labels), 0) + 1
            total_episodes += 1

        if not payload:
            raise SystemExit(f"{dataset}: no eval episodes")
        # No "default": an absent episode must fail loudly, not silently fall
        # through to the arm generating its own labels and being scored as if
        # it had been handed them.
        (args.out_dir / f"{dataset}__{args.source}.json").write_text(
            json.dumps(payload, indent=1), encoding="utf-8"
        )
        written += 1

    print(
        json.dumps(
            {
                "datasets": written,
                "episodes": total_episodes,
                "labels_per_episode": dict(sorted(lengths.items())),
                "default_key": "absent, deliberately",
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
