#!/usr/bin/env python
"""Present Corpus B to the harness as 39 components, one per task.

The whole Corpus A pipeline -- `make_label_files.py`, `align_floors.py`,
`run_arm.py`, `aggregate.py`, the cluster bootstrap -- keys on a COMPONENT, one
dataset name per unit of analysis, and `aggregate.py` resamples components as the
bootstrap cluster.

Corpus B arrives as a single converted dataset holding 39 tasks. Handed over as
one component it would give the bootstrap **n = 1 cluster**, which is not a
conservative approximation of 39 -- it is no clustering at all, and every interval
in the report would be wrong in the permissive direction. Clustering on the task
is what `corpus_b_plan.md` section 9 pre-registers.

Two ways to fix that: change `aggregate.py` to cluster on a new key, or present
Corpus B in the shape the harness already understands. This does the second. The
analysis code that produced Study 1 and Study 2 is not touched, so a Corpus A
result and a Corpus B result remain comparable by construction rather than by
inspection.

Each task becomes a component named `corpus_b__<task>`:

  * `<source-dir>/corpus_b__<task>` -- a SYMLINK to the one converted dataset
    root. The episodes are already there; only the per-component episode lists
    differ, so copying 39 roots would multiply 4 GB by 39 to no purpose.
  * `ground_truth/corpus_b__<task>.json` -- that task's episodes only
  * `splits/corpus_b__<task>.json` -- that task's seed and eval

Every file keys on the RENUMBERED subset index, which is the index space the
converted root uses. Nothing here re-derives the selection: it partitions what
`translate_corpus_b_splits.py` already validated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="the converted v3.0 dataset root")
    parser.add_argument("--tasks-splits", type=Path, required=True, help="<dataset>.tasks.json")
    parser.add_argument("--ground-truth", type=Path, required=True, help="whole-corpus ground truth")
    parser.add_argument("--source-dir", type=Path, required=True, help="where component roots are looked up")
    parser.add_argument("--gt-out-dir", type=Path, required=True)
    parser.add_argument("--splits-out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="corpus_b__")
    args = parser.parse_args()

    payload = json.loads(args.tasks_splits.read_text(encoding="utf-8"))
    tasks = payload["tasks"]
    truth_payload = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    truth = truth_payload["episodes"]

    args.source_dir.mkdir(parents=True, exist_ok=True)
    args.gt_out_dir.mkdir(parents=True, exist_ok=True)
    args.splits_out_dir.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()

    components = []
    for task, groups in sorted(tasks.items()):
        name = f"{args.prefix}{task}"
        seed = [int(e) for e in groups["seed"]]
        evaluation = [int(e) for e in groups["eval"]]
        missing = [e for e in seed + evaluation if str(e) not in truth]
        if missing:
            raise SystemExit(f"{name}: {len(missing)} episode(s) have no ground truth, e.g. {missing[:5]}")
        if set(seed) & set(evaluation):
            raise SystemExit(f"{name}: seed and eval overlap")

        link = args.source_dir / name
        if link.is_symlink() or link.exists():
            if link.is_symlink() and Path(os.readlink(link)).resolve() == root:
                pass
            else:
                raise SystemExit(
                    f"{link} exists and is not a symlink to {root}. Refusing to replace it; "
                    "a component root pointing somewhere unexpected would be scored silently."
                )
        else:
            link.symlink_to(root, target_is_directory=True)

        (args.gt_out_dir / f"{name}.json").write_text(
            json.dumps({
                "dataset": name,
                "source": str(root),
                "corpus": "robointer",
                "task": task,
                "fps": truth_payload.get("fps", 10),
                "gt_source": truth_payload.get("gt_source"),
                "index_space": "renumbered subset indices",
                "n_episodes_with_gt": len(seed) + len(evaluation),
                "episodes": {str(e): truth[str(e)] for e in sorted(seed + evaluation)},
            }, indent=1),
            encoding="utf-8",
        )
        (args.splits_out_dir / f"{name}.json").write_text(
            json.dumps({
                "dataset": name,
                "task": task,
                "seed": sorted(seed),
                "eval": sorted(evaluation),
                "n_available": len(seed) + len(evaluation),
                "policy": "corpus_b_plan.md 4.2; seed fits calibration and is never scored",
                "index_space": "renumbered subset indices",
            }, indent=1),
            encoding="utf-8",
        )
        components.append({"component": name, "task": task, "n_seed": len(seed), "n_eval": len(evaluation)})

    manifest = {
        "root": str(root),
        "prefix": args.prefix,
        "n_components": len(components),
        "n_seed": sum(c["n_seed"] for c in components),
        "n_eval": sum(c["n_eval"] for c in components),
        "note": "One component per RoboInter task, so aggregate.py's bootstrap clusters on the "
                "task as corpus_b_plan.md section 9 pre-registers. All components share one "
                "dataset root by symlink; only their episode lists differ.",
        "components": components,
    }
    # Beside the ground-truth directory, not inside it: make_label_files.py globs
    # *.json there and treats every hit as a component, so a manifest living among
    # them is read as a 40th dataset with no split.
    (args.gt_out_dir.parent / "corpus_b_components.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_components", "n_seed", "n_eval")}, indent=1))
    print(" ".join(c["component"] for c in components))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
