#!/usr/bin/env python
"""Merge evaluation components into task-level predictions in stable study IDs.

Components separate source datasets and segment counts for inference. The task
is the analysis/bootstrap unit for every corpus. Re-emit predictions, ground
truth and splits in one identity namespace; do not treat components as tasks.
Explicitly failed jobs remain in the scoring population. Fallback jobs keep
their logical calibrated arm and record actual runtime application separately.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from calibration_status import STATUS_VERSION


def prediction_calibration_status(record: dict[str, Any], episode: str) -> dict[str, Any]:
    """Use an observed decision; cohort membership cannot establish application."""
    configured = bool((record.get("flags") or {}).get("plan.subtask_align_calibration_path"))
    status = (record.get("calibration_status") or {}).get(episode)
    if status is None:
        if configured:
            raise ValueError(
                f"episode {episode}: calibration runtime decision is missing; rerun the job"
            )
        return {
            "applied": False,
            "reason": "not_configured",
            "method": None,
            "source": "configuration",
        }
    if type(status.get("applied")) is not bool:
        raise ValueError(f"episode {episode}: calibration runtime decision is invalid")
    if status["applied"] and not configured:
        raise ValueError(f"episode {episode}: uncalibrated arm claims calibration was applied")
    return dict(status)


def load_index_map(path: Path) -> dict[int, int]:
    """local index -> original RoboInter index"""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {int(e["new_index"]): int(e["original_index"]) for e in payload["episodes"]}


def main(
    *,
    default_prefix: str = "study__",
    default_namespace: str = "study_episode_index",
    default_arms_config: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--group-meta", type=Path, required=True)
    parser.add_argument(
        "--cohorts-dir", type=Path, required=True, help="holds <cohort>.index_map.json"
    )
    parser.add_argument("--gt-dir", type=Path, required=True, help="cohort-level ground truth")
    parser.add_argument("--out-predictions", type=Path, required=True)
    parser.add_argument("--out-gt", type=Path, required=True)
    parser.add_argument("--out-splits", type=Path, required=True)
    parser.add_argument("--prefix", default=default_prefix)
    parser.add_argument("--identity-namespace", default=default_namespace)
    parser.add_argument(
        "--arms", nargs="+", help="expected arms for historical runs without a job manifest"
    )
    parser.add_argument(
        "--job-manifest", type=Path, help="defaults to predictions-dir/jobs_manifest.json"
    )
    parser.add_argument(
        "--eligibility-out",
        type=Path,
        help="write the shared task/episode eligibility mask for scoring every arm, including floors",
    )
    parser.add_argument(
        "--arms-config",
        type=Path,
        default=default_arms_config
        or Path(__file__).resolve().parents[1] / "configs/alignment_protocol_arms.yaml",
    )
    args = parser.parse_args()

    group_meta = json.loads(args.group_meta.read_text(encoding="utf-8"))
    arms = {a["name"]: a for a in yaml.safe_load(args.arms_config.read_text())["arms"]}
    manifest_path = args.job_manifest or args.predictions_dir / "jobs_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if args.job_manifest and manifest is None:
        raise SystemExit(f"missing job manifest {manifest_path}")
    expected_arms = args.arms or (manifest or {}).get("arms")
    if not expected_arms or len(set(expected_arms)) != len(expected_arms):
        raise SystemExit(
            "provide a job manifest or explicit unique --arms; observed files cannot define coverage"
        )
    if set(expected_arms) - arms.keys():
        raise SystemExit(f"unknown expected arms: {set(expected_arms) - arms.keys()}")
    expected = {(dataset, arm) for dataset in group_meta for arm in expected_arms}
    if manifest is not None:
        recorded = [(j["dataset"], j["arm"]) for j in manifest.get("jobs", [])]
        if (
            manifest.get("version") != 1
            or set(recorded) != expected
            or len(recorded) != len(expected)
        ):
            raise SystemExit("job manifest differs from the complete expected dataset/arm matrix")
    expected_paths = {f"{d}__{a}.json": (d, a) for d, a in expected}
    observed = {
        p.name
        for p in args.predictions_dir.glob("*.json")
        if p.name not in {"run_summary.json", "jobs_manifest.json", manifest_path.name}
    }
    if observed != expected_paths.keys():
        raise SystemExit(
            f"incomplete or stale prediction matrix: missing={sorted(expected_paths.keys() - observed)}, "
            f"unexpected={sorted(observed - expected_paths.keys())}"
        )

    maps: dict[str, dict[int, int]] = {}
    for dataset in group_meta:
        path = args.cohorts_dir / f"{dataset}.index_map.json"
        if not path.exists():
            raise SystemExit(f"{dataset}: no index map at {path}")
        for metadata in (group_meta[dataset], json.loads(path.read_text())):
            if (
                metadata.get("identity_namespace", args.identity_namespace)
                != args.identity_namespace
            ):
                raise SystemExit(f"{dataset}: identity namespace differs from requested study")
        maps[dataset] = load_index_map(path)

    # ---- ground truth and splits, per task, on original indices ---------
    task_truth: dict[str, dict[str, Any]] = defaultdict(dict)
    for dataset, meta in sorted(group_meta.items()):
        truth = json.loads((args.gt_dir / f"{dataset}.json").read_text(encoding="utf-8"))[
            "episodes"
        ]
        if manifest is not None:
            for job in manifest["jobs"]:
                if job["dataset"] == dataset:
                    ids = job.get("episodes", [])
                    if len(ids) != len(set(ids)) or {str(e) for e in ids} != set(truth):
                        raise SystemExit(
                            f"{dataset}: job manifest episodes differ from evaluation GT"
                        )
        for local, spans in truth.items():
            original = maps[dataset].get(int(local))
            if original is None:
                raise SystemExit(f"{dataset}: local episode {local} absent from its index map")
            if str(original) in task_truth[meta["task"]]:
                raise SystemExit(
                    f"duplicate evaluation trajectory {original} in task {meta['task']}"
                )
            task_truth[meta["task"]][str(original)] = spans

    eligibility = {}
    for dataset, meta in group_meta.items():
        task = f"{args.prefix}{meta['task']}"
        bucket = eligibility.setdefault(task, {})
        for original in maps[dataset].values():
            if str(original) not in task_truth[meta["task"]]:
                raise SystemExit(
                    f"{dataset}: index map contains a trajectory outside evaluation GT"
                )
            bucket[str(original)] = bool(meta.get("calibration_eligible", meta.get("calibrated")))
    expected_outputs = {
        f"{args.prefix}{task}__{arm}.json" for task in task_truth for arm in expected_arms
    }
    stale = {p.name for p in args.out_predictions.glob("*.json")} - expected_outputs
    if stale:
        raise SystemExit(
            f"output directory contains unrequested predictions: {sorted(stale)}; use a new directory"
        )
    for d in (args.out_predictions, args.out_gt, args.out_splits):
        d.mkdir(parents=True, exist_ok=True)
    if args.eligibility_out:
        args.eligibility_out.parent.mkdir(parents=True, exist_ok=True)
        args.eligibility_out.write_text(
            json.dumps(
                {
                    "version": 1,
                    "source": "shared_seed_count_support",
                    "index_space": args.identity_namespace,
                    "episodes": eligibility,
                },
                indent=1,
            )
        )

    for task, episodes in sorted(task_truth.items()):
        name = f"{args.prefix}{task}"
        (args.out_gt / f"{name}.json").write_text(
            json.dumps(
                {
                    "dataset": name,
                    "task": task,
                    "index_space": args.identity_namespace,
                    "n_episodes_with_gt": len(episodes),
                    "episodes": {k: episodes[k] for k in sorted(episodes, key=int)},
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        (args.out_splits / f"{name}.json").write_text(
            json.dumps(
                {
                    "dataset": name,
                    "task": task,
                    "seed": [],
                    "eval": sorted(int(k) for k in episodes),
                    "n_available": len(episodes),
                    "index_space": args.identity_namespace,
                    "policy": "seed episodes fit the calibration and are in no scored dataset at all",
                },
                indent=1,
            ),
            encoding="utf-8",
        )

    # ---- predictions, per (task, logical arm) ---------------------------
    merged: dict[tuple[str, str], dict[str, Any]] = {}
    dropped: list[dict[str, Any]] = []
    calibrated_flag: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)

    for filename, (dataset, arm) in sorted(expected_paths.items()):
        path = args.predictions_dir / filename
        record = json.loads(path.read_text(encoding="utf-8"))
        meta = group_meta[dataset]
        key = (meta["task"], arm)
        entry = merged.setdefault(
            key,
            {
                "dataset": f"{args.prefix}{meta['task']}",
                "task": meta["task"],
                "arm": arm,
                "tool": arms[arm].get("tool"),
                "supervision": arms[arm].get("supervision"),
                "index_space": args.identity_namespace,
                "episodes": {},
                "cohorts": [],
                "n_cohorts": 0,
                "prompt_tokens": 0,
                "generation_tokens": 0,
                "total_tokens": 0,
                "elapsed_seconds": 0.0,
                "ok": True,
                "calibration_status_version": STATUS_VERSION,
                "calibration_eligibility_source": meta.get(
                    "calibration_eligible_source", "cohort_group_metadata"
                ),
                "calibration_status": {},
                "calibration_eligible": {},
                "failed_cohorts": [],
            },
        )
        # Eligibility is useful for a paired analysis, but is kept separate from
        # actual application and exists on the uncalibrated baseline as well.
        for original in maps[dataset].values():
            entry["calibration_eligible"][str(original)] = eligibility[entry["dataset"]][
                str(original)
            ]
        entry["cohorts"].append(
            {
                "dataset": dataset,
                "eligible": bool(meta.get("calibrated")),
                "calibration_configured": bool(
                    (record.get("flags") or {}).get("plan.subtask_align_calibration_path")
                    or record.get("calibration_sha")
                ),
                "calibration_fit_available": record.get("calibration_fit_available"),
                "ran_as": record.get("arm"),
                "ok": bool(record.get("ok")),
                "n": len(record.get("episodes") or {}) if record.get("ok") else 0,
            }
        )
        entry["n_cohorts"] += 1
        for field in ("prompt_tokens", "generation_tokens", "total_tokens"):
            entry[field] += record.get(field) or 0
        entry["elapsed_seconds"] += record.get("elapsed_seconds") or 0.0
        if not record.get("ok"):
            failure = {"file": path.name, "arm": arm, "reason": record.get("error") or "ok=false"}
            dropped.append(failure)
            entry["failed_cohorts"].append(failure)
            continue
        for local, spans in (record.get("episodes") or {}).items():
            original = maps[dataset].get(int(local))
            if original is None:
                raise SystemExit(f"{dataset}: predicted episode {local} absent from its index map")
            if str(original) in entry["episodes"]:
                raise SystemExit(f"{key}: episode {original} predicted twice")
            entry["episodes"][str(original)] = spans
            try:
                status = prediction_calibration_status(record, local)
            except ValueError as exc:
                raise SystemExit(f"{path.name}: {exc}") from exc
            status["episode"] = original
            entry["calibration_status"][str(original)] = status
            calibrated_flag[key][str(original)] = status["applied"]

    for (task, arm), entry in sorted(merged.items()):
        flags = calibrated_flag[(task, arm)]
        for episode in task_truth[task]:
            if episode not in entry["episodes"]:
                flags[episode] = False
                entry["calibration_status"][episode] = {
                    "episode": int(episode),
                    "applied": False,
                    "reason": "no_output",
                    "method": None,
                    "source": "delivery",
                }
        entry["n_episodes_predicted"] = len(entry["episodes"])
        # Per EPISODE, which is the shape score_alignment.py reads. A single
        # fraction would lose exactly the information the covered/uncovered
        # stratum of plan section 6.2 is defined on.
        entry["calibration_applied"] = {
            k: bool(v) for k, v in sorted(flags.items(), key=lambda kv: int(kv[0]))
        }
        entry["calibration_applied_fraction"] = round(
            sum(1 for v in flags.values() if v) / max(len(flags), 1), 4
        )
        entry["episodes"] = {k: entry["episodes"][k] for k in sorted(entry["episodes"], key=int)}
        (args.out_predictions / f"{args.prefix}{task}__{arm}.json").write_text(
            json.dumps(entry, indent=1), encoding="utf-8"
        )

    by_arm: dict[str, int] = defaultdict(int)
    for (_, arm), entry in merged.items():
        by_arm[arm] += entry["n_episodes_predicted"]
    print(
        json.dumps(
            {
                "merged_files": len(merged),
                "tasks": len(task_truth),
                "episodes_per_arm": dict(sorted(by_arm.items())),
                "dropped_failed_jobs": len(dropped),
                "dropped": dropped[:6],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
