#!/usr/bin/env python
"""Census a RoboInter-Data RH20T sample: the measurements behind `corpus_b_plan.md`.

Every *(measured)* number in that plan comes from this script. It is committed so
the plan's figures are reproducible rather than asserted, and so the full 83-chunk
census (`corpus_b_plan.md` section 12) runs the same code as the 6-chunk probe.

It reads ONLY ground-truth structure -- clip counts, boundary positions,
cross-view spread. It never touches an arm, a prompt, or a hyperparameter, which
is why using it to design the study does not contaminate the corpus
(`corpus_b_plan.md` section 4.3).

Three things it exists to establish:

1. **The unit of analysis is the trajectory, not the episode file.** RH20T
   records each physical trajectory from up to 8 camera positions and stores each
   view as its own episode with its own `episode_index`. Counting files as
   independent samples inflates n by ~8x. Trajectory identity is recovered by
   stripping the trailing `_<camera_view>` from `episode_name`.

2. **Calibration cohorts cannot be keyed on the label list.** Corpus A fits per
   component on the modal label tuple; RoboInter's labels are free-form prose, so
   that key yields cohorts of size 1-8 and none large enough to fit. The
   (task, segment-count) key is measured alongside it so the choice is evidenced.

3. **Boundary regularity, which is the whole point of Corpus B.** Corpus A's
   per-boundary SD of normalised position is 0.036, which is why a model-free
   timing template beats every uncalibrated arm there. If Corpus B is markedly
   less regular, that result was corpus-specific.

Cross-view agreement (`--cross-view`) doubles as the human noise floor that
`alignment_plan.md` section 11 called the gating measurement and costed at an
hour of annotator time. The 8 views of one trajectory carry separately-marked
boundaries, so their spread measures annotation consistency directly. Cameras are
NOT frame-synchronised -- median duration spread across views is 6.5 s -- so the
comparison is restricted to duration-matched views or it measures desync instead.
It is a LOWER bound: one team, one event, different angles, not blind
re-annotation.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

TIME_CLIP = "annotation.time_clip"
SUBTASK = "annotation.substask"  # sic: the upstream field name carries this typo
COLUMNS = [TIME_CLIP, SUBTASK, "timestamp", "episode_name", "camera_view"]

# Corpus A's per-boundary SD of normalised position, from the Study 2 run.
CORPUS_A_BOUNDARY_SD = 0.036


def parse_clips(value: Any) -> list[tuple[int, int]]:
    text = str(value).strip()
    if not text or text in {"nan", "None"}:
        return []
    try:
        return [(int(a), int(b)) for a, b in ast.literal_eval(text)]
    except Exception:
        return []


def trajectory_key(episode_name: str, camera_view: str) -> str:
    """Strip the camera suffix so all views of one physical trajectory collapse."""
    suffix = f"_{camera_view}"
    return episode_name[: -len(suffix)] if episode_name.endswith(suffix) else episode_name


def task_of(trajectory: str) -> str:
    match = re.match(r"(RH20T_cfg\d+_task_\d+)", trajectory)
    return match.group(1) if match else "unknown"


def scan(chunk_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    trajectories: dict[str, dict[str, Any]] = {}
    for chunk in chunk_dirs:
        for path in sorted(chunk.glob("*.parquet")):
            try:
                frame = pd.read_parquet(path, columns=COLUMNS)
            except Exception:
                continue
            name = str(frame["episode_name"].iloc[0])
            camera = str(frame["camera_view"].iloc[0])
            clips = parse_clips(frame[TIME_CLIP].iloc[0])
            stamps = frame["timestamp"].to_numpy()
            labels = frame[SUBTASK].tolist()
            key = trajectory_key(name, camera)
            record = trajectories.setdefault(
                key, {"task": task_of(key), "chunk": chunk.name, "views": []}
            )
            record["views"].append(
                {
                    "camera": camera,
                    "clips": clips,
                    "n_frames": len(frame),
                    "duration": float(stamps[-1] - stamps[0]) if len(stamps) > 1 else 0.0,
                    # Boundary times in SECONDS: the onset of every clip after the first.
                    "boundaries": [
                        float(stamps[min(start, len(stamps) - 1)]) for start, _ in clips[1:]
                    ],
                    "labels": tuple(
                        str(labels[start]).strip() for start, _ in clips if start < len(labels)
                    ),
                }
            )
    return trajectories


def modal_clip_count(record: dict[str, Any]) -> int:
    counts = Counter(len(view["clips"]) for view in record["views"])
    return counts.most_common(1)[0][0]


def report(trajectories: dict[str, dict[str, Any]], cross_view: bool) -> dict[str, Any]:
    n_files = sum(len(r["views"]) for r in trajectories.values())
    modal = {key: modal_clip_count(rec) for key, rec in trajectories.items()}
    usable = [k for k, n in modal.items() if n >= 2]

    print(f"episode-view files    : {n_files}")
    print(f"distinct trajectories : {len(trajectories)}")
    print(f"views per trajectory  : {dict(sorted(Counter(len(r['views']) for r in trajectories.values()).items()))}")
    print(f"usable (>=2 clips)    : {len(usable)}/{len(trajectories)} = "
          f"{100 * len(usable) / max(len(trajectories), 1):.1f}%")
    print(f"internal boundaries   : {dict(sorted(Counter(modal[k] - 1 for k in usable).items()))}")

    # Per-task multi-clip rate. This is bimodal -- whole tasks are single-subtask
    # -- which is why selection happens at the task level (plan section 4.2).
    per_task: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for key, rec in trajectories.items():
        per_task[rec["task"]][0] += 1
        if modal[key] >= 2:
            per_task[rec["task"]][1] += 1
    print(f"\nPER-TASK ({len(per_task)} tasks)")
    print(f"{'task':<28} {'traj':>6} {'multi':>6} {'rate':>6}")
    rates = []
    for task, (total, multi) in sorted(per_task.items()):
        print(f"{task:<28} {total:>6d} {multi:>6d} {100 * multi / total:>5.0f}%")
        rates.append(multi / total)
    eligible = sum(1 for _, (_, m) in per_task.items() if m >= 30)
    print(f"  median per-task rate: {st.median(rates):.2f}; "
          f">=50%: {sum(r >= 0.5 for r in rates)}/{len(rates)}; "
          f">=30 usable trajectories (plan 4.2 step 5): {eligible}/{len(per_task)}")

    # Cohort feasibility. The exact-label key is what Corpus A used; it dies here.
    by_count: Counter = Counter()
    by_labels: Counter = Counter()
    for key in usable:
        rec = trajectories[key]
        by_count[(rec["task"], modal[key])] += 1
        modal_labels = Counter(v["labels"] for v in rec["views"]).most_common(1)[0][0]
        by_labels[(rec["task"], modal_labels)] += 1
    print("\nCALIBRATION COHORT FEASIBILITY")
    for label, counter in (("(task, segment count)", by_count), ("(task, exact label list)", by_labels)):
        sizes = sorted(counter.values(), reverse=True)
        print(f"  {label:26s} cohorts {len(counter):5d}  largest {sizes[0] if sizes else 0:3d}  "
              f">=20 trajectories: {sum(s >= 20 for s in sizes)}")

    # Boundary regularity: the measurement Corpus B exists to make.
    slots: dict[tuple, list[float]] = defaultdict(list)
    for key in usable:
        rec = trajectories[key]
        view = rec["views"][0]
        if len(view["clips"]) != modal[key] or view["n_frames"] < 2:
            continue
        for index, (start, _) in enumerate(view["clips"][1:]):
            slots[(rec["task"], modal[key], index)].append(start / (view["n_frames"] - 1))
    spreads = [st.pstdev(v) for v in slots.values() if len(v) >= 8]
    boundary_sd = st.median(spreads) if spreads else None
    print("\nBOUNDARY REGULARITY (SD of normalised position, within task x segment-count)")
    if boundary_sd is None:
        print("  not enough trajectories per boundary slot")
    else:
        print(f"  Corpus B: {boundary_sd:.3f}   ({len(spreads)} slots with >=8 trajectories)")
        print(f"  Corpus A: {CORPUS_A_BOUNDARY_SD:.3f}   ratio {boundary_sd / CORPUS_A_BOUNDARY_SD:.1f}x")

    summary = {
        "episode_view_files": n_files,
        "trajectories": len(trajectories),
        "usable_trajectories": len(usable),
        "tasks": len(per_task),
        "tasks_with_30_usable": eligible,
        "cohorts_by_segment_count_ge20": sum(v >= 20 for v in by_count.values()),
        "cohorts_by_label_list_ge20": sum(v >= 20 for v in by_labels.values()),
        "boundary_sd_corpus_b": boundary_sd,
        "boundary_sd_corpus_a": CORPUS_A_BOUNDARY_SD,
    }
    if cross_view:
        summary["cross_view"] = cross_view_agreement(trajectories, modal)
    return summary


def cross_view_agreement(
    trajectories: dict[str, dict[str, Any]], modal: dict[str, int], tolerance: float = 1.0
) -> dict[str, Any]:
    """Human noise floor from the spread across camera views of one trajectory.

    Restricted to views of near-equal duration: the cameras start and stop
    independently, so an unrestricted comparison measures recording desync rather
    than annotation disagreement.
    """
    deviations: list[float] = []
    n_traj = 0
    for key, rec in trajectories.items():
        views = rec["views"]
        if len(views) < 2 or modal[key] < 2:
            continue
        if len({len(v["boundaries"]) for v in views}) != 1 or not views[0]["boundaries"]:
            continue
        median_duration = st.median([v["duration"] for v in views])
        matched = [v for v in views if abs(v["duration"] - median_duration) <= tolerance]
        if len(matched) < 3:
            continue
        n_traj += 1
        for index in range(len(matched[0]["boundaries"])):
            values = [v["boundaries"][index] for v in matched]
            centre = st.median(values)
            deviations.extend(abs(v - centre) for v in values)
    if not deviations:
        return {"pairs": 0}
    deviations.sort()
    result = {
        "duration_tolerance_s": tolerance,
        "trajectories": n_traj,
        "pairs": len(deviations),
        "mean_s": round(st.mean(deviations), 3),
        "median_s": round(st.median(deviations), 3),
        "p90_s": round(deviations[int(0.9 * len(deviations)) - 1], 3),
        "within_1s": round(sum(d <= 1.0 for d in deviations) / len(deviations), 3),
    }
    print("\nCROSS-VIEW AGREEMENT (human floor, LOWER bound -- see module docstring)")
    print(f"  views matched within {tolerance}s of duration: "
          f"{n_traj} trajectories, {len(deviations)} boundary-view pairs")
    print(f"  |view - view median|: mean {result['mean_s']}s  median {result['median_s']}s  "
          f"p90 {result['p90_s']}s  within 1s {result['within_1s']:.1%}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory holding extracted chunk-NNN/ directories of parquet files")
    parser.add_argument("--cross-view", action="store_true",
                        help="also measure the cross-view human noise floor")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    chunks = sorted(d for d in args.root.glob("chunk-*") if d.is_dir())
    if not chunks:
        raise SystemExit(f"no chunk-* directories under {args.root}")
    print(f"chunks: {[c.name for c in chunks]}\n")

    summary = report(scan(chunks), cross_view=args.cross_view)
    summary["chunks"] = [c.name for c in chunks]
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
        print(f"\nwrote {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
