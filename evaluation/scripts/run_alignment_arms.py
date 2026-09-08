#!/usr/bin/env python
"""Dispatch evaluation cohorts using frozen, separately fitted arm manifests.

Each calibrated arm requires its own manifest. A missing or rejected count fit
within that manifest uses the identical uncalibrated configuration. Task seed
budgets, episode identities, regime, model, and file digests are checked before
dispatch. Every job still uses run_arm.py's caching and runtime telemetry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import threading
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from fit_task_calibration import episode_ids, fit_regime, validate_fit, validate_task_splits  # noqa: E402
from calibration_status import STATUS_VERSION  # noqa: E402


def load_manifests(paths: list[Path], arms_config: Path, model_id: str) -> dict[str, Any]:
    """Read immutable, arm-specific fit files and validate their task seed budget."""
    manifests = {}
    allocation = None
    for path in paths:
        document = json.loads(path.read_text())
        bundled = document.get("arms")
        entries = (
            bundled.items() if isinstance(bundled, dict) else [(document.get("arm"), document)]
        )
        for arm, manifest in entries:
            if not arm or manifest.get("schema_version") != 2 or manifest.get("arm") != arm:
                raise ValueError(
                    "expected a version 2 per-arm calibration manifest; legacy shared fits are unsafe"
                )
            if arm in manifests:
                raise ValueError(f"duplicate calibration manifest for {arm}")
            if manifest.get("model_id") != model_id:
                raise ValueError(f"{arm}: calibration model differs from inference model")
            regime, _ = fit_regime(arms_config, arm)
            if manifest.get("fit_regime") != regime:
                raise ValueError(f"{arm}: calibration regime differs from inference arm")
            tasks = validate_task_splits(manifest.get("task_splits"))
            minimum = manifest.get("min_cohort")
            if type(minimum) is not int or not 3 <= minimum <= 10:
                raise ValueError(f"{arm}: invalid minimum cohort size")
            identities = manifest.get("episode_identities") or {}
            required = {str(e) for g in tasks.values() for values in g.values() for e in values}
            if set(identities) != required or any(type(v) is not int for v in identities.values()):
                raise ValueError(f"{arm}: missing or malformed episode identities")
            if len(set(identities.values())) != len(identities):
                raise ValueError(f"{arm}: episode identity map is not injective")
            current = (
                tasks,
                manifest.get("identity_namespace"),
                identities,
                manifest.get("seed_sources"),
            )
            if allocation is not None and allocation != current:
                raise ValueError(
                    "all arm calibrations must share the SAME task seed/evaluation allocation"
                )
            allocation = current
            for key, entry in manifest.get("fit_files", {}).items():
                task = entry.get("task_id")
                if task not in tasks or key != f"{task}__n{entry.get('n_segments')}":
                    raise ValueError(f"{arm}/{key}: invalid task or segment count")
                if (
                    entry.get("arm") != arm
                    or entry.get("task_seed_episodes") != tasks[task]["seed"]
                ):
                    raise ValueError(f"{arm}/{key}: fit arm or task seed provenance differs")
                spec = manifest.get("cohorts", {}).get(key)
                if (
                    not spec
                    or spec.get("task") != task
                    or spec.get("n_segments") != entry.get("n_segments")
                ):
                    raise ValueError(f"{arm}/{key}: missing matching cohort provenance")
                fit_path = Path(entry["path"])
                if not fit_path.is_absolute():
                    fit_path = path.parent / fit_path
                if not fit_path.exists() or hashlib.sha256(
                    fit_path.read_bytes()
                ).hexdigest() != entry.get("sha256"):
                    raise ValueError(f"{arm}/{key}: calibration file missing or digest changed")
                payload = json.loads(fit_path.read_text())
                fitted = validate_fit(payload, spec, arm, tasks[task]["seed"], minimum)
                if fitted != entry.get("fit_on_episodes") or len(fitted) != entry.get(
                    "fit_episode_count"
                ):
                    raise ValueError(f"{arm}/{key}: recorded fit provenance does not match file")
                entry["path"] = str(fit_path.resolve())
            manifests[arm] = manifest
    return manifests


def component_episodes(
    dataset: str, source_dir: Path, splits_dir: Path, gt_dir: Path, seen: set[int]
) -> tuple[list[int], dict[int, int], dict[str, Any]]:
    """Verify evaluation identity across the cohort-local and original spaces."""
    split = json.loads((splits_dir / f"{dataset}.json").read_text())
    if split.get("seed"):
        raise ValueError(f"{dataset}: scored dataset must hold no seed episodes")
    evaluation = episode_ids(split["eval"], f"{dataset}/eval")
    truth = json.loads((gt_dir / f"{dataset}.json").read_text())["episodes"]
    if set(map(str, evaluation)) != set(truth):
        raise ValueError(f"{dataset}: ground truth must contain exactly the evaluation episodes")
    map_path = source_dir / f"{dataset}.index_map.json"
    entries = json.loads(map_path.read_text())["episodes"]
    mapping = {int(e["new_index"]): int(e["original_index"]) for e in entries}
    if len(mapping) != len(entries) or len(set(mapping.values())) != len(entries):
        raise ValueError(f"{dataset}: index map must be injective")
    if not set(evaluation) <= mapping.keys():
        raise ValueError(f"{dataset}: evaluation episode absent from index map")
    originals = {mapping[e] for e in evaluation}
    if originals & seen:
        raise ValueError(f"{dataset}: evaluation trajectories appear in multiple components")
    seen.update(originals)
    return evaluation, mapping, truth


def write_failed_job(
    out: Path, job: dict[str, Any], arms: dict[str, Any], error: str, returncode: int
) -> None:
    """Replace any old success with an explicit failure that the merger retains."""
    actual_arm = arms[job["run_as"]]
    flags = dict(actual_arm.get("flags") or {})
    for key, value in flags.items():
        if isinstance(value, str):
            flags[key] = value.replace("{subtasks_path}", str(job["labels"])).replace(
                "{calibration_path}", str(job["calibration"] or "")
            )
    record = {
        "dataset": job["component"],
        "task_id": job["task"],
        "arm": job["run_as"],
        "requested_arm": job["arm"],
        "tool": actual_arm["tool"],
        "supervision": arms[job["arm"]].get("supervision"),
        "flags": flags,
        "ok": False,
        "returncode": returncode,
        "error": error,
        "episodes": {},
        "n_episodes_requested": len(job["episodes"]),
        "n_episodes_predicted": 0,
        "calibration_status_version": STATUS_VERSION,
        "calibration_status": {},
        "calibration_applied": {},
        "calibration_fit_available": bool(job["calibration"]),
        "calibration_fit_path": job["calibration"],
        "calibration_fit_sha256": job["calibration_sha256"],
        "calibration_fallback": job["fallback"],
    }
    temporary = out.with_suffix(".json.partial")
    temporary.write_text(json.dumps(record, indent=1))
    temporary.replace(out)


def verify_frozen_calibration(job: dict[str, Any], tag: str) -> None:
    """Check the manifest's digest both before and after the subprocess."""
    if not job["calibration"]:
        return
    try:
        digest = hashlib.sha256(Path(job["calibration"]).read_bytes()).hexdigest()
    except OSError as exc:
        raise ValueError(f"{tag}: frozen calibration file is missing or unreadable") from exc
    if digest != job["calibration_sha256"]:
        raise ValueError(f"{tag}: calibration digest differs from the frozen manifest")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument(
        "--group-meta",
        type=Path,
        required=True,
        help="cohort_group_meta.json: dataset -> {task, cohort, calibrated}",
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument(
        "--cohort-files",
        type=Path,
        action="append",
        default=[],
        help="version 2 per-arm manifest; repeat for single/stack fits, or supply {arms: {...}}",
    )
    parser.add_argument("--arms-config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--job-splits-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=None)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--ports", default="8000,8001,8002,8003,8004,8005,8006")
    parser.add_argument("--python", default="python")
    parser.add_argument("--label-source", default="oracle")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = yaml.safe_load(args.arms_config.read_text(encoding="utf-8"))
    arms = {a["name"]: a for a in spec["arms"]}
    wanted = args.arms or [n for n, a in arms.items() if a.get("tool") != "reference"]
    unknown = [a for a in wanted if a not in arms]
    if unknown:
        raise SystemExit(f"unknown arm(s): {unknown}")

    group_meta = json.loads(args.group_meta.read_text(encoding="utf-8"))
    try:
        manifests = load_manifests(args.cohort_files, args.arms_config, args.model_id)
        for arm in wanted:
            if (
                "plan.subtask_align_calibration_path" in (arms[arm].get("flags") or {})
                and arm not in manifests
            ):
                raise ValueError(
                    f"{arm}: missing arm-specific calibration manifest; fit this arm first"
                )
        components = {}
        seen: set[int] = set()
        for dataset in sorted(group_meta):
            components[dataset] = component_episodes(
                dataset, args.source_dir, args.splits_dir, args.gt_dir, seen
            )
        if manifests:
            allocation = next(iter(manifests.values()))
            if not allocation.get("identity_namespace"):
                raise ValueError("fit manifest needs a stable study identity namespace")
            for dataset, (evaluation, identities, _) in components.items():
                task = group_meta[dataset]["task"]
                if (
                    group_meta[dataset].get("identity_namespace", allocation["identity_namespace"])
                    != allocation["identity_namespace"]
                ):
                    raise ValueError(f"{dataset}: component and fit identity namespaces differ")
                index_meta = json.loads((args.source_dir / f"{dataset}.index_map.json").read_text())
                if (
                    index_meta.get("identity_namespace", allocation["identity_namespace"])
                    != allocation["identity_namespace"]
                ):
                    raise ValueError(f"{dataset}: index map and fit identity namespaces differ")
                groups = allocation["task_splits"].get(task)
                if groups is None:
                    raise ValueError(f"{dataset}: task absent from calibration task allocation")
                allowed = {allocation["episode_identities"][str(e)] for e in groups["eval"]}
                if not {identities[e] for e in evaluation} <= allowed:
                    raise ValueError(
                        f"{dataset}: evaluation includes seed or unallocated trajectories"
                    )
            required_eval = {
                allocation["episode_identities"][str(e)]
                for groups in allocation["task_splits"].values()
                for e in groups["eval"]
            }
            if seen != required_eval:
                raise ValueError(
                    "evaluation components must cover ALL allocated evaluation trajectories; "
                    f"missing={sorted(required_eval - seen)}, unexpected={sorted(seen - required_eval)}"
                )
        port_names = [p.strip() for p in args.ports.split(",") if p.strip()]
        if not port_names or len(port_names) != len(set(port_names)):
            raise ValueError("ports must be a nonempty unique list")
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(str(exc)) from exc

    # Each dataset IS one calibration cohort (or the uncalibrated remainder of a
    # task), holding ONLY eval episodes. So a job selects every episode in its
    # dataset -- which is what the tool requires, since a partial selection would
    # leave unselected episodes in the same parquet shard -- and its calibration
    # is a property of the dataset rather than of individual episodes.
    # Map each calibrated arm to the arm identical to it except for the
    # calibration flag. Asserted rather than assumed: a fallback that quietly
    # changed another setting would make the uncovered fifth a different
    # experiment from the covered majority.
    CAL_FLAG = "plan.subtask_align_calibration_path"
    counterpart: dict[str, str] = {}
    for name, arm in arms.items():
        flags = dict(arm.get("flags") or {})
        if CAL_FLAG not in flags:
            continue
        stripped = {k: v for k, v in flags.items() if k != CAL_FLAG}
        for other_name, other in arms.items():
            if other_name == name:
                continue
            if dict(other.get("flags") or {}) == stripped and other.get("tool") == arm.get("tool"):
                counterpart[name] = other_name
                break

    jobs: list[dict[str, Any]] = []
    for arm_name in wanted:
        arm = arms[arm_name]
        needs_calibration = any(
            "{calibration_path}" in str(v) for v in (arm.get("flags") or {}).values()
        )
        for dataset, meta in sorted(group_meta.items()):
            evaluation, identities, truth = components[dataset]
            labels = args.labels_dir / f"{dataset}__{args.label_source}.json"
            if not labels.exists():
                raise SystemExit(f"missing label file {labels}")
            label_payload = json.loads(labels.read_text())
            for episode in evaluation:
                supplied = (
                    label_payload
                    if isinstance(label_payload, list)
                    else label_payload.get(str(episode), label_payload.get("default"))
                )
                if supplied != [span["text"] for span in truth[str(episode)]]:
                    raise SystemExit(
                        f"{dataset}/{episode}: oracle labels differ from evaluation ground truth"
                    )

            calibration = None
            run_as = arm_name
            fit_entry = None
            if needs_calibration:
                manifest = manifests.get(arm_name)
                if manifest is not None:
                    task = meta["task"]
                    groups = manifest["task_splits"].get(task)
                    if groups is None:
                        raise SystemExit(f"{dataset}: task absent from {arm_name} manifest")
                    if not manifest.get("identity_namespace"):
                        raise SystemExit("fit manifest needs a stable study identity namespace")
                    fit_identities = manifest["episode_identities"]
                    allowed_eval = {fit_identities[str(e)] for e in groups["eval"]}
                    if not {identities[e] for e in evaluation} <= allowed_eval:
                        raise SystemExit(
                            f"{dataset}: evaluation includes seed or unallocated trajectories"
                        )
                    counts = {len(truth[str(e)]) for e in evaluation}
                    keys = {f"{task}__n{count}" for count in counts}
                    eligible_keys = keys & manifest["fit_files"].keys()
                    if eligible_keys and len(counts) != 1:
                        raise SystemExit(
                            f"{dataset}: mixed counts with available fits; rebuild components by task and count"
                        )
                    if eligible_keys:
                        fit_entry = manifest["fit_files"][next(iter(eligible_keys))]
                        calibration = fit_entry["path"]
                if calibration is None:
                    # The uncalibrated fallback of plan section 6.2, expressed the only
                    # way the tool permits. run_arm.py refuses an arm whose flags carry
                    # {calibration_path} with no path supplied, so "same arm without a
                    # calibration" cannot be requested directly. The counterpart arm IS
                    # that configuration: it is derived below by flag comparison and
                    # asserted to differ ONLY by the calibration flag, so the fallback is
                    # exactly what it claims to be rather than a nearby arm.
                    run_as = counterpart.get(arm_name)
                    if not run_as:
                        raise SystemExit(
                            f"{arm_name}: no uncalibrated counterpart, so the fallback of "
                            f"section 6.2 cannot be run on {dataset}"
                        )
            jobs.append(
                {
                    "arm": arm_name,
                    "run_as": run_as,
                    "component": dataset,
                    "task": meta["task"],
                    "group": "uncal"
                    if not meta.get("calibrated")
                    else str(meta["cohort"]).split("__")[-1],
                    "episodes": evaluation,
                    "calibration": calibration,
                    "labels": labels,
                    "fallback": run_as != arm_name,
                    "calibration_sha256": fit_entry["sha256"] if fit_entry else None,
                }
            )

    n_eps = sum(len(j["episodes"]) for j in jobs)
    print(
        json.dumps(
            {
                "arms": len(wanted),
                "datasets": len(group_meta),
                "jobs": len(jobs),
                "episode_predictions": n_eps,
                "calibrated_jobs": sum(1 for j in jobs if j["calibration"]),
                "fallback_jobs": sum(1 for j in jobs if j.get("fallback")),
                "counterparts": counterpart,
            },
            indent=1,
        )
    )
    if args.dry_run:
        return 0

    args.job_splits_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Persist the complete expected matrix before starting any job. An interrupted
    # dispatcher must not leave a smaller population that the merger accepts.
    job_manifest = {
        "version": 1,
        "arms": wanted,
        "datasets": sorted(group_meta),
        "jobs": [
            {"dataset": j["component"], "arm": j["arm"], "episodes": j["episodes"]} for j in jobs
        ],
    }
    temporary = args.out_dir / "jobs_manifest.json.partial"
    temporary.write_text(json.dumps(job_manifest, indent=1))
    temporary.replace(args.out_dir / "jobs_manifest.json")
    ports: queue.Queue[str] = queue.Queue()
    for port in port_names:
        ports.put(port)

    results: list[dict[str, Any]] = []
    lock = threading.Lock()

    def run_job(job: dict[str, Any]) -> None:
        tag = f"{job['component']}__{job['arm']}"
        split_path = args.job_splits_dir / f"{tag}.json"
        out = args.out_dir / f"{tag}.json"
        port = None
        try:
            split_path.write_text(
                json.dumps(
                    {
                        "dataset": job["component"],
                        "seed": [],
                        "eval": sorted(job["episodes"]),
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
            port = ports.get()  # the read IS the lease
            verify_frozen_calibration(job, tag)
            cmd = [
                args.python,
                str(HERE / "run_arm.py"),
                "--dataset",
                job["component"],
                "--source-root",
                str(args.source_dir / job["component"]),
                "--arm",
                job["run_as"],
                "--arms-config",
                str(args.arms_config),
                # A work root PER JOB. run_arm.py derives its staging root as
                # work_dir/<dataset>__<arm>__r<repeat>, which is identical for every
                # cohort job of one (component, arm) -- and build_working_root wipes
                # and rebuilds it with overwrite=True. Sharing it would have several
                # concurrent jobs deleting each other's staged dataset mid-run.
                "--work-dir",
                str(args.work_dir / tag),
                "--out",
                str(out),
                "--episodes",
                str(split_path),
                "--base-url",
                f"http://127.0.0.1:{port}/v1",
                "--model-id",
                args.model_id,
                "--python",
                args.python,
                "--subtasks-path",
                str(job["labels"]),
            ]
            if job["calibration"]:
                cmd += ["--calibration-path", job["calibration"]]
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = ""
            proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
            verify_frozen_calibration(job, tag)
            record = json.loads(out.read_text()) if out.exists() else {}
            ok = proc.returncode == 0 and bool(record.get("ok"))
            error = (
                None
                if ok
                else (
                    proc.stderr
                    or proc.stdout
                    or record.get("error")
                    or "job produced no successful output"
                )[-400:]
            )
            if ok:
                record.update(
                    {
                        "requested_arm": job["arm"],
                        "calibration_fit_available": bool(job["calibration"]),
                        "calibration_fit_path": job["calibration"],
                        "calibration_fit_sha256": job["calibration_sha256"],
                        "calibration_fallback": job["fallback"],
                    }
                )
                out.write_text(json.dumps(record, indent=1))
            else:
                write_failed_job(out, job, arms, error, proc.returncode)
            with lock:
                results.append(
                    {
                        "tag": tag,
                        "arm": job["arm"],
                        "component": job["component"],
                        "task": job["task"],
                        "group": job["group"],
                        "run_as": job["run_as"],
                        "fallback": job["fallback"],
                        "n_episodes": len(job["episodes"]),
                        "calibrated": bool(job["calibration"]),
                        "ok": ok,
                        "returncode": proc.returncode,
                        "error": error,
                    }
                )
                done = len(results)
                if done % 25 == 0:
                    print(f"  {done}/{len(jobs)} jobs", flush=True)
        except Exception as exc:
            error = str(exc)
            try:
                write_failed_job(out, job, arms, error, 1)
            except OSError as write_error:
                error += f"; could not persist failure record: {write_error}"
            with lock:
                results.append(
                    {
                        "tag": tag,
                        "arm": job["arm"],
                        "ok": False,
                        "component": job["component"],
                        "task": job["task"],
                        "n_episodes": len(job["episodes"]),
                        "error": error,
                        "returncode": 1,
                    }
                )
        finally:
            if port is not None:
                ports.put(port)  # returned whatever the status

    threads = [threading.Thread(target=run_job, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    failed = [r for r in results if not r["ok"]]
    summary = {
        "jobs": len(jobs),
        "ok": len(results) - len(failed),
        "failed": len(failed),
        "episode_predictions": n_eps,
        "failures": failed,
    }
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {k: summary[k] for k in ("jobs", "ok", "failed", "episode_predictions")}, indent=1
        )
    )
    if failed:
        for row in failed[:5]:
            print(f"  FAILED {row['tag']}: {str(row['error'])[:160]}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
