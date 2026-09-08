#!/usr/bin/env python
"""Apply the Corpus B selection rule and emit the episode list, splits, and audit.

Uses the trajectory and view rules from `corpus_b_plan.md` sections 4.2 and 5.3,
with the ten-seed protocol described in `evaluation/README.md`. Every step is
score-independent: no metric, no model output, and no arm configuration is read.

The rule, in order
------------------
1. RH20T only (Corpus B is RH20T; DROID's 43,025 free-form task strings cannot
   form calibration cohorts -- section 4.1).
2. Trajectory-level. RH20T films each physical trajectory from up to 8 camera
   positions and stores each as its own LeRobot episode, so counting episodes
   would inflate n by ~8x. Trajectory identity is `episode_name` with the
   trailing `_<camera_view>` stripped.
3. Keep trajectories with >= 2 subtask clips. Alignment is undefined when one
   subtask spans the whole recording.
4. Keep trajectories whose spans pass `prepare_robointer.episode_spans`.
5. Keep tasks with >= MIN_USABLE usable trajectories (10 seed + >= 20 eval).
6. Take ALL qualifying tasks. Amended from "the 24 with the lowest task_index"
   after the full census found 42 qualifying; see section 4.2, which records why
   the amendment is legitimate. `--max-tasks` still applies the original cap for
   anyone reproducing the pre-amendment design.
7. Use all remaining trajectories for evaluation. `--cap-eval 40` reproduces
   the historical study's evaluation cap; task-level aggregation gives tasks
   equal weight regardless of how many episodes they contribute.
8. Seed = the 10 lowest-`episode_index` trajectories of each task. Seed
   trajectories fit calibration and are NEVER scored.

One view per trajectory (section 5.3)
-------------------------------------
The view whose duration is closest to the trajectory's median duration, ties
broken by lowest `camera_view` string. Fixed in advance and independent of any
score. The held-back views are recorded in the audit so the camera sub-study
(section 8.1) can use them without re-deriving the selection.

Output
------
`--out-dir` receives:
  `episodes.json`   - the original episode indices to hand to build_corpus_b_subset.py
                      (the CHOSEN view of each selected trajectory, main study only)
  `camera_substudy_episodes.json` - chosen AND held-back views for section 8.1
  `splits.json`     - per task: seed and eval, as ORIGINAL episode indices
  `selection_audit.json` - every task considered and why it was kept or dropped

Splits are emitted in ORIGINAL indices because that is what the subset builder
consumes. It renumbers to 0..N-1 and writes `index_map.json`; the stage that
builds the scoring splits must translate through that map, and the audit records
this explicitly so the two index spaces are never conflated.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_robointer import episode_spans  # noqa: E402

TIME_CLIP = "annotation.time_clip"
SUBTASK = "annotation.substask"
SKILL = "annotation.primitive_skill"
COLUMNS = [TIME_CLIP, SUBTASK, SKILL, "timestamp", "episode_name", "camera_view", "episode_index"]

MIN_USABLE = 30
N_SEED = 10
CHUNK_SIZE = 1000


def task_of(trajectory: str) -> str:
    match = re.match(r"(RH20T_cfg\d+_task_\d+)", trajectory)
    return match.group(1) if match else "unknown"


def scan(root: Path, fps: float) -> dict[str, dict[str, Any]]:
    """One pass over every episode, grouped into trajectories."""
    trajectories: dict[str, dict[str, Any]] = {}
    chunks = sorted(d for d in root.glob("chunk-*") if d.is_dir())
    if not chunks:
        raise SystemExit(f"no chunk-* directories under {root}")
    for chunk in chunks:
        for path in sorted(chunk.glob("*.parquet")):
            try:
                frame = pd.read_parquet(path, columns=COLUMNS)
            except Exception:
                continue
            if not len(frame):
                continue
            name = str(frame["episode_name"].iloc[0])
            camera = str(frame["camera_view"].iloc[0])
            suffix = f"_{camera}"
            key = name[: -len(suffix)] if name.endswith(suffix) else name

            raw = str(frame[TIME_CLIP].iloc[0]).strip()
            try:
                clips = ast.literal_eval(raw) if raw and raw not in {"nan", "None"} else []
            except Exception:
                clips = []

            spans, reason, _gaps = episode_spans(frame.reset_index(drop=True), fps)
            stamps = frame["timestamp"].to_numpy()

            record = trajectories.setdefault(key, {"task": task_of(key), "views": []})
            record["views"].append({
                "episode_index": int(frame["episode_index"].iloc[0]),
                "camera": camera,
                "n_clips": len(clips),
                "n_spans": len(spans) if spans else 0,
                "reject_reason": None if spans else reason,
                "duration": float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0,
            })
    return trajectories


def choose_view(views: list[dict[str, Any]]) -> dict[str, Any]:
    """Section 5.3: closest to the trajectory's median duration, ties by camera id."""
    median = st.median([v["duration"] for v in views])
    return sorted(views, key=lambda v: (abs(v["duration"] - median), v["camera"]))[0]


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory holding chunk-NNN/ of RoboInter parquet")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--min-usable", type=int, default=MIN_USABLE)
    parser.add_argument("--cap-eval", type=int, default=None,
                        help="optional evaluation cap per task; default uses all trajectories after the seeds")
    parser.add_argument("--n-seed", type=int, default=N_SEED)
    parser.add_argument("--camera-substudy-tasks", type=int, default=6,
                        help="how many tasks (lowest id first) get ALL their views included in "
                             "camera_substudy_episodes.json for plan section 8.1. 0 disables it.")
    parser.add_argument("--max-tasks", type=int, default=None,
                        help="cap the number of tasks, lowest task id first. Omit to take all "
                             "that qualify (the amended plan); pass 24 for the original design.")
    args = parser.parse_args()

    if not 1 <= args.n_seed <= 10:
        parser.error("--n-seed must be between 1 and 10 trajectories per task")
    if args.cap_eval is not None and args.cap_eval < 1:
        parser.error("--cap-eval must be positive when supplied")

    trajectories = scan(args.data_root, args.fps)

    # Steps 3 and 4, at trajectory level, using the chosen view.
    usable_by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    drop_reasons: Counter = Counter()
    view_disagreement = 0
    for key, record in sorted(trajectories.items()):
        view = choose_view(record["views"])
        # Usability is decided on the CHOSEN view, because that view's video is
        # what an arm sees and that view's annotation is what it is scored
        # against. When views of one trajectory disagree about usability the
        # choice therefore changes the population, so the rate is recorded rather
        # than left implicit.
        usable_flags = {(v["n_clips"] >= 2 and not v["reject_reason"]) for v in record["views"]}
        if len(usable_flags) > 1:
            view_disagreement += 1
        if view["n_clips"] < 2:
            drop_reasons["single-clip (alignment undefined)"] += 1
            continue
        if view["reject_reason"]:
            drop_reasons[f"spans rejected: {view['reject_reason'].split('(')[0].strip()}"] += 1
            continue
        usable_by_task[record["task"]].append({
            "trajectory": key,
            "episode_index": view["episode_index"],
            "camera": view["camera"],
            "n_spans": view["n_spans"],
            "duration": round(view["duration"], 2),
            "other_views": sorted(
                v["episode_index"] for v in record["views"] if v["episode_index"] != view["episode_index"]
            ),
        })

    # Steps 5 and 6.
    qualifying = sorted(t for t, rows in usable_by_task.items() if len(rows) >= args.min_usable)
    below_bar = {t for t in usable_by_task if t not in qualifying}
    if args.max_tasks is not None:
        cut_by_cap = set(qualifying[args.max_tasks:])
        qualifying = qualifying[: args.max_tasks]
    else:
        cut_by_cap = set()

    if not qualifying:
        raise SystemExit(
            f"no task reached --min-usable={args.min_usable}; "
            f"best was {max((len(r) for r in usable_by_task.values()), default=0)}"
        )

    # Steps 7 and 8.
    splits: dict[str, dict[str, list[int]]] = {}
    episodes: list[int] = []
    per_task_audit: dict[str, Any] = {}
    for task in qualifying:
        rows = sorted(usable_by_task[task], key=lambda r: r["episode_index"])
        seed_rows = rows[: args.n_seed]
        eval_rows = rows[args.n_seed :]
        if args.cap_eval is not None:
            eval_rows = eval_rows[:args.cap_eval]
        seed = [r["episode_index"] for r in seed_rows]
        evaluation = [r["episode_index"] for r in eval_rows]
        # A task that qualifies but has nothing left after the seed contributes no
        # scored trajectory, while still consuming a calibration cohort and a
        # cluster slot in the bootstrap. Drop it and say so. This cannot happen at
        # the plan's parameters (min_usable 30, n_seed 10 -> >=20 eval), but
        # --min-usable is a flag and the failure would be silent.
        if not evaluation:
            drop_reasons[f"task {task}: qualified but 0 eval after seeding"] += 1
            continue
        splits[task] = {"seed": seed, "eval": evaluation}
        episodes.extend(seed + evaluation)
        per_task_audit[task] = {
            "n_usable": len(rows),
            "n_seed": len(seed),
            "n_eval": len(evaluation),
            "n_unused_beyond_cap": len(rows) - len(seed_rows) - len(eval_rows),
            "trajectories": {
                "seed": [r["trajectory"] for r in seed_rows],
                "eval": [r["trajectory"] for r in eval_rows],
            },
            "held_back_views": {
                str(r["episode_index"]): r["other_views"] for r in eval_rows
            },
            "spans_per_eval_trajectory": Counter(r["n_spans"] for r in eval_rows),
        }

    # The camera sub-study (plan section 8.1) needs every VIEW of its eval
    # trajectories, but `episodes` holds only the chosen view of each. Emitting
    # the main list alone would leave the sub-study with no data in the built
    # subset, and it would fail as "nothing to score" rather than as a missing
    # input. Its episodes are listed separately so the operator chooses whether
    # to pay for them: all views of 42 tasks would be roughly six times the
    # subset, where the plan only asks for six tasks.
    qualifying = [t for t in qualifying if t in splits]
    substudy_tasks = qualifying[: args.camera_substudy_tasks] if args.camera_substudy_tasks else []
    substudy_episodes: list[int] = []
    for task in substudy_tasks:
        for index in splits[task]["eval"]:
            substudy_episodes.append(index)
            substudy_episodes.extend(per_task_audit[task]["held_back_views"].get(str(index), []))
    substudy_episodes = sorted(set(substudy_episodes))
    if not qualifying:
        raise SystemExit("every qualifying task had 0 eval trajectories after seeding")
    episodes = sorted(set(episodes))
    overlap = set().union(*(set(s["seed"]) & set(s["eval"]) for s in splits.values())) if splits else set()
    if overlap:
        raise SystemExit(f"seed and eval overlap for {len(overlap)} episode(s): {sorted(overlap)[:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "episodes.json").write_text(json.dumps(episodes), encoding="utf-8")
    (args.out_dir / "camera_substudy_episodes.json").write_text(
        json.dumps({
            "note": "Plan section 8.1. Chosen AND held-back views of the eval trajectories of "
                    "the listed tasks. Superset of those tasks' entries in episodes.json; build "
                    "a separate subset from this if the camera sub-study is being run.",
            "tasks": substudy_tasks,
            "n_episodes": len(substudy_episodes),
            "episodes": substudy_episodes,
        }, indent=1),
        encoding="utf-8",
    )
    (args.out_dir / "splits.json").write_text(
        json.dumps({
            "index_space": "ORIGINAL RoboInter episode_index. build_corpus_b_subset.py "
                           "renumbers to 0..N-1; translate through its index_map.json before "
                           "scoring. Never mix the two.",
            "n_seed_per_task": args.n_seed,
            "cap_eval": args.cap_eval,
            "tasks": splits,
        }, indent=1),
        encoding="utf-8",
    )

    # Built AFTER every drop, so `kept` describes the selection that was actually
    # used. Computing it earlier marked tasks kept that --max-tasks or the
    # zero-eval rule later removed, which would make the audit -- the document
    # whose whole job is to show the selection was not shaped after the fact --
    # disagree with the selection it documents.
    considered = {}
    for task, rows in sorted(usable_by_task.items()):
        if task in splits:
            reason = "selected"
        elif task in below_bar:
            reason = f"below --min-usable ({len(rows)} < {args.min_usable})"
        elif task in cut_by_cap:
            reason = f"cut by --max-tasks={args.max_tasks}"
        else:
            reason = "qualified but 0 eval after seeding"
        considered[task] = {"n_usable": len(rows), "kept": task in splits, "reason": reason}

    audit = {
        "rule": "corpus_b_plan.md sections 4.2 and 5.3",
        "data_root": str(args.data_root),
        "parameters": {
            "min_usable": args.min_usable, "cap_eval": args.cap_eval,
            "n_seed": args.n_seed, "max_tasks": args.max_tasks, "fps": args.fps,
        },
        "totals": {
            "trajectories_scanned": len(trajectories),
            "trajectories_usable": sum(len(r) for r in usable_by_task.values()),
            "tasks_with_any_usable": len(usable_by_task),
            "tasks_qualifying": len([t for t, c in considered.items() if c["kept"]]),
            "tasks_selected": len(qualifying),
            "tasks_dropped_for_zero_eval": sum(
                1 for k in drop_reasons if k.startswith("task ") and k.endswith("0 eval after seeding")
            ),
            "episodes_selected": len(episodes),
            "eval_trajectories": sum(len(s["eval"]) for s in splits.values()),
            "seed_trajectories": sum(len(s["seed"]) for s in splits.values()),
            "trajectories_with_view_disagreement_on_usability": view_disagreement,
            "camera_substudy_tasks": len(substudy_tasks),
            "camera_substudy_episodes": len(substudy_episodes),
        },
        "trajectory_drop_reasons": dict(drop_reasons),
        "tasks_considered": considered,
        "per_task": {k: {**v, "spans_per_eval_trajectory": dict(v["spans_per_eval_trajectory"])}
                     for k, v in per_task_audit.items()},
    }
    (args.out_dir / "selection_audit.json").write_text(json.dumps(audit, indent=1), encoding="utf-8")

    print(json.dumps(audit["totals"], indent=1))
    print(f"drop reasons: {dict(drop_reasons)}")
    print(f"-> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
