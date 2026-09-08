#!/usr/bin/env python
"""Choose seed and evaluation episodes for each dataset.

Two independent leakage risks are handled here.

**Seed leakage.** Arms that fit a boundary prior need labelled episodes. Those
episodes must never be scored. The split is written once, to disk, before any
model runs, and every consumer reads it rather than choosing its own -- so the
supervised arms and the model-free script-prior arm are fitted on exactly the
same episodes, and none of them can quietly borrow an evaluation episode.

**Development contamination.** This tool was developed against these datasets,
and the published experiments fit priors on the first ten episodes of each. To
avoid re-using the very episodes that shaped the design, seeds are drawn from
that same low-index region (episodes 0-9) rather than from fresh ones, and the
evaluation split is taken exclusively from the remainder. That deliberately
concedes the strongest seed episodes to the supervised arms while keeping the
scored episodes as clean as this corpus allows.

Sampling is a seeded, evenly-spaced draw over the eligible indices rather than
a random subset, so the evaluation set spans the whole dataset (early and late
recording sessions alike) and is exactly reproducible.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_SEED_POOL = 10


def choose_evaluation(candidates: list[int], k: int) -> list[int]:
    """Evenly-spaced deterministic subset of ``candidates``.

    Even spacing rather than random sampling because episodes within a dataset
    are recorded in sessions, and a random draw can land disproportionately in
    one session. Spacing guarantees coverage without needing a random seed at
    all, which removes one more thing a reader has to trust.
    """
    if k >= len(candidates):
        return list(candidates)
    if k <= 0:
        return []
    step = len(candidates) / k
    return [candidates[min(len(candidates) - 1, int(i * step))] for i in range(k)]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="ground truth JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-eval", type=int, default=40)
    parser.add_argument("--n-seed", type=int, default=10)
    parser.add_argument("--seed-pool", type=int, default=DEFAULT_SEED_POOL,
                        help="seeds are drawn from episode indices below this value")
    args = parser.parse_args()

    payload = json.loads(args.gt.read_text(encoding="utf-8"))
    available = sorted(int(k) for k in payload["episodes"])
    if not available:
        raise SystemExit("ground truth has no episodes")

    seed_candidates = [e for e in available if e < args.seed_pool]
    if len(seed_candidates) < args.n_seed:
        seed_candidates = available[: args.n_seed]
    seed = seed_candidates[: args.n_seed]

    remainder = [e for e in available if e not in set(seed)]
    evaluation = choose_evaluation(remainder, args.n_eval)

    overlap = set(seed) & set(evaluation)
    if overlap:
        raise SystemExit(f"internal error: overlap {sorted(overlap)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "dataset": payload["dataset"],
                "n_available": len(available),
                "seed": seed,
                "eval": evaluation,
                "policy": {
                    "seed_pool_max_index": args.seed_pool,
                    "n_seed": args.n_seed,
                    "n_eval_requested": args.n_eval,
                    "selection": "evenly spaced over non-seed episodes, deterministic",
                },
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"[split] {payload['dataset']}: {len(available)} annotated, "
        f"seed={len(seed)} eval={len(evaluation)} -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
