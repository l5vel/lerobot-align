#!/usr/bin/env python
"""Build one physically separate v3.0 dataset per Corpus B task.

An earlier design pointed all 39 components at ONE converted 1,592-episode root by
symlink, and selected each component's episodes with `--only_episodes`. The tool
refuses that, correctly:

    ValueError: A partial annotation run cannot safely migrate parquet shards that
    contain unselected episodes. Run once without --only_episodes to migrate the
    complete dataset, then partial reruns are safe.

The v3.0 conversion concatenates episodes into a handful of shards, so a shard
holding task A's episodes also holds task B's. Writing annotations back through a
partial run would touch episodes the run never selected. The alternative -- run
once over all 1,592 episodes to migrate, then rerun partially -- is worse: it
annotates the SEED episodes too, and seed episodes fit the calibration, so they
must never be scored.

So each task gets its own dataset, holding only its own seed and eval episodes.
That also fixes the staging cost: `run_arm.py` copies its source root per job, and
a shared 1.6 GB root cost 1.6 GB per job across ~356 jobs, where a per-task root
costs about 40 MB.

Each component is built by the two steps already validated for the whole corpus --
`build_corpus_b_subset.py` then LeRobot's shipped `convert_dataset_v21_to_v30` --
so nothing new is trusted here. Episode indices are renumbered again WITHIN each
component, and each component's own `index_map.json` records the mapping, so the
per-component ground truth and splits are rewritten to match.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def run(cmd: list[str], log: Path, env_extra: dict[str, str] | None = None) -> tuple[bool, str]:
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log.write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""), encoding="utf-8")
    return proc.returncode == 0, (proc.stderr or proc.stdout or "")[-500:]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--selection-splits", type=Path, required=True,
                        help="selection/splits.json, in ORIGINAL RoboInter indices")
    parser.add_argument("--source", type=Path, required=True, help="robointer/ with data/ and videos/")
    parser.add_argument("--meta", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True, help="where component roots are built")
    parser.add_argument("--gt-out-dir", type=Path, required=True)
    parser.add_argument("--splits-out-dir", type=Path, required=True)
    parser.add_argument("--converter", type=Path, required=True)
    parser.add_argument("--lerobot-src", type=Path, required=True)
    parser.add_argument("--python", default="python")
    parser.add_argument("--prefix", default="corpus_b__")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--tasks", nargs="*", default=None)
    parser.add_argument(
        "--groups", type=Path, default=None,
        help="build these groups instead of whole tasks: JSON mapping name -> "
             "{seed: [...], eval: [...]} in ORIGINAL indices. Used to build one dataset "
             "per (task, calibration cohort), because a job must select EVERY episode in "
             "the shards it touches -- a calibrated arm scoring one cohort of a task "
             "leaves the task's other cohorts unselected in the same shard, which the "
             "tool refuses.",
    )
    args = parser.parse_args()

    if args.groups:
        tasks = json.loads(args.groups.read_text(encoding="utf-8"))
    else:
        tasks = json.loads(args.selection_splits.read_text(encoding="utf-8"))["tasks"]
    if args.tasks:
        tasks = {k: v for k, v in tasks.items() if k in set(args.tasks)}
    for d in (args.out_dir, args.gt_out_dir, args.splits_out_dir):
        d.mkdir(parents=True, exist_ok=True)

    slots: queue.Queue[int] = queue.Queue()
    for i in range(args.workers):
        slots.put(i)
    results: dict[str, dict[str, Any]] = {}
    lock = threading.Lock()

    def build(task: str, groups: dict[str, list[int]]) -> None:
        slot = slots.get()
        try:
            name = task if task.startswith(args.prefix) else f"{args.prefix}{task}"
            root = args.out_dir / name
            logs = args.out_dir / "_logs"
            seed = sorted(int(e) for e in groups["seed"])
            evaluation = sorted(int(e) for e in groups["eval"])
            episodes = sorted(set(seed) | set(evaluation))

            # A component built earlier and left converted would be picked up by
            # the converter's <root>_old restore path; refuse rather than reuse.
            for suffix in ("_old", "_v30", ".partial"):
                stale = args.out_dir / f"{name}{suffix}"
                if stale.exists():
                    with lock:
                        results[name] = {"ok": False, "stage": "precheck",
                                         "error": f"stale {stale} present; remove it"}
                    return

            ep_file = args.out_dir / f"{name}.episodes.json"
            ep_file.write_text(json.dumps(episodes), encoding="utf-8")

            ok, tail = run(
                [args.python, str(HERE / "build_corpus_b_subset.py"),
                 "--source", str(args.source), "--meta", str(args.meta),
                 "--episodes", str(ep_file), "--out", str(root)],
                logs / f"{name}.build.log")
            if not ok:
                with lock:
                    results[name] = {"ok": False, "stage": "build", "error": tail}
                return

            ok, tail = run(
                [args.python, str(args.converter), f"--repo-id=corpus_b/{task}",
                 f"--root={root}", "--push-to-hub=false"],
                logs / f"{name}.convert.log",
                {"PYTHONPATH": str(args.lerobot_src)})
            if not ok:
                with lock:
                    results[name] = {"ok": False, "stage": "convert", "error": tail}
                return

            # Ground truth and splits must be rewritten in the component's OWN
            # index space: build_corpus_b_subset renumbers again within each
            # component, so the whole-corpus indices no longer name these episodes.
            ok, tail = run(
                [args.python, str(HERE / "prepare_robointer.py"),
                 "--root", str(root), "--gt-out", str(args.gt_out_dir / f"{name}.json"),
                 "--dataset-name", name, "--fps", str(args.fps), "--provenance-verified"],
                logs / f"{name}.gt.log")
            if not ok:
                with lock:
                    results[name] = {"ok": False, "stage": "ground_truth", "error": tail}
                return

            index_map = json.loads((args.out_dir / f"{name}.index_map.json").read_text(encoding="utf-8"))
            forward = {int(e["original_index"]): int(e["new_index"]) for e in index_map["episodes"]}
            missing = [e for e in episodes if e not in forward]
            if missing:
                with lock:
                    results[name] = {"ok": False, "stage": "index_map",
                                     "error": f"{len(missing)} episode(s) absent, e.g. {missing[:3]}"}
                return
            local_seed = sorted(forward[e] for e in seed)
            local_eval = sorted(forward[e] for e in evaluation)
            if set(local_seed) & set(local_eval):
                with lock:
                    results[name] = {"ok": False, "stage": "splits", "error": "seed/eval overlap"}
                return
            (args.splits_out_dir / f"{name}.json").write_text(
                json.dumps({
                    "dataset": name, "task": task,
                    "seed": local_seed, "eval": local_eval,
                    "n_available": len(local_seed) + len(local_eval),
                    "index_space": f"component-local indices; see {name}.index_map.json",
                    "policy": "corpus_b_plan.md 4.2; seed fits calibration and is never scored",
                }, indent=1), encoding="utf-8")

            with lock:
                results[name] = {"ok": True, "task": task, "root": str(root),
                                 "n_seed": len(local_seed), "n_eval": len(local_eval)}
                done = len(results)
                print(f"  {done}/{len(tasks)} {name}", flush=True)
        finally:
            slots.put(slot)

    threads = [threading.Thread(target=build, args=(t, g)) for t, g in sorted(tasks.items())]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = {k: v for k, v in results.items() if not v["ok"]}
    manifest = {
        "n_components": len(results),
        "n_ok": len(results) - len(failed),
        "n_failed": len(failed),
        "n_seed": sum(v.get("n_seed", 0) for v in results.values() if v["ok"]),
        "n_eval": sum(v.get("n_eval", 0) for v in results.values() if v["ok"]),
        "note": "One physically separate v3.0 dataset per task. Indices are component-local; "
                "each component's index_map.json maps back to the original RoboInter episode.",
        "components": [{"component": k, **v} for k, v in sorted(results.items()) if v["ok"]],
        "failures": failed,
    }
    (args.out_dir.parent / "corpus_b_components.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("n_components", "n_ok", "n_failed", "n_seed", "n_eval")}, indent=1))
    for name, row in list(failed.items())[:5]:
        print(f"  FAILED {name} at {row['stage']}: {str(row['error'])[:150]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
