#!/usr/bin/env python
"""Fit separate arm/count calibrations using at most ten total seeds per task.

Also export every evaluation task/count group in stable study trajectory IDs. Group membership never depends on whether an arm's
fit passes its gate. Metadata's legacy ``calibrated`` field denotes sufficient
seed-count support only, explicitly distinguished from runtime application.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
MIN_COHORT = 3
MAX_TASK_SEEDS = 10


def episode_ids(values: Any, where: str) -> list[int]:
    if not isinstance(values, list) or any(type(v) is not int or v < 0 for v in values):
        raise ValueError(f"{where}: episode ids must be nonnegative integers")
    if len(values) != len(set(values)):
        raise ValueError(f"{where}: duplicate episode ids")
    return sorted(values)


def validate_task_splits(
    tasks: dict[str, Any], truth: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Reject budget excess, duplicates and overlap in the global episode namespace."""
    if not isinstance(tasks, dict) or not tasks:
        raise ValueError("task splits must be a nonempty object")
    seen, result = {}, {}
    for task, groups in sorted(tasks.items()):
        if not isinstance(groups, dict) or "seed" not in groups or "eval" not in groups:
            raise ValueError(f"{task}: expected seed and eval lists")
        result[task] = {}
        for group in ("seed", "eval"):
            ids = episode_ids(groups[group], f"{task}/{group}")
            if group == "seed" and len(ids) > MAX_TASK_SEEDS:
                raise ValueError(f"{task}: at most {MAX_TASK_SEEDS} TOTAL seeds per task")
            for episode in ids:
                if episode in seen:
                    raise ValueError(
                        f"episode {episode}: overlap between {seen[episode]} and {task}/{group}"
                    )
                if truth is not None and not truth.get(str(episode)):
                    raise ValueError(f"{task}/{group}: episode {episode} has no ground truth")
                seen[episode] = f"{task}/{group}"
            result[task][group] = ids
    return result


def fit_regime(arms_config: Path, arm: str) -> tuple[dict[str, Any], list[str]]:
    """Derive all exposed batch settings from the inference arm, without overrides."""
    arms = {a["name"]: a for a in yaml.safe_load(arms_config.read_text())["arms"]}
    if arm not in arms or arms[arm].get("tool") != "align":
        raise ValueError(f"unknown alignment arm {arm}")
    flags = arms[arm].get("flags") or {}
    if "plan.subtask_align_calibration_path" not in flags:
        raise ValueError(f"{arm}: select a calibrated inference arm")
    fixed = {
        "plan.min_subtask_seconds": 1.5,
        "plan.contact_sheet_columns": 5,
        "plan.contact_sheet_frames_per_sheet": 20,
        "plan.contact_sheet_quality": 84,
        "plan.subtask_align_min_fraction": 0.0,
        "vlm.max_new_tokens": 4096,
    }
    for key, expected in fixed.items():
        if flags.get(key, expected) != expected:
            raise ValueError(f"{arm}: batch driver cannot reproduce {key}={flags[key]}")
    chat = flags.get("vlm.chat_template_kwargs", '{"enable_thinking":false}')
    chat = json.loads(chat) if isinstance(chat, str) else chat
    if chat != {"enable_thinking": False}:
        raise ValueError(f"{arm}: batch driver requires enable_thinking=false")
    cameras = flags.get("plan.subtask_align_camera_keys")
    cameras = yaml.safe_load(cameras) if isinstance(cameras, str) else cameras
    if cameras is not None and not isinstance(cameras, (list, tuple)):
        raise ValueError(f"{arm}: camera keys must be a list")
    cameras = list(cameras or [flags.get("vlm.camera_key", "observation.images.wrist")])
    if any(not isinstance(c, str) or not c for c in cameras):
        raise ValueError(f"{arm}: invalid camera list")
    if any("{camera_" in c for c in cameras):
        raise ValueError(
            "resolve camera roles with prepare_alignment_study.py and pass its arms.yaml"
        )
    regime = {
        "frame_format": flags.get("plan.subtask_align_frame_format", "contact_sheet"),
        "sampling": flags.get("plan.subtask_align_sampling", "uniform"),
        "fps": flags.get("plan.frames_per_second", 2.0),
        "max_frames": flags.get("plan.max_frames_per_prompt", 60),
        "frame_width": flags.get("plan.contact_sheet_frame_width", 224),
        "temperature": flags.get("vlm.temperature", 0.2),
        "cameras": cameras,
        "video_fallback": flags.get("plan.subtask_video_fallback", "contact_sheet"),
        **fixed,
        "chat_template_kwargs": chat,
    }
    extra = [value for camera in cameras for value in ("--camera", camera)]
    for option, key in (
        ("formats", "frame_format"),
        ("sampling", "sampling"),
        ("fps", "fps"),
        ("max-frames", "max_frames"),
        ("frame-width", "frame_width"),
        ("temperature", "temperature"),
        ("video-fallback", "video_fallback"),
    ):
        extra += [f"--{option}", str(regime[key])]
    return regime, extra


def validate_fit(
    payload: dict[str, Any], spec: dict[str, Any], arm: str, task_seed: list[int], min_cohort: int
) -> list[int]:
    allocated = episode_ids(spec.get("seed_episodes"), "cohort seed_episodes")
    if not min_cohort <= len(allocated) <= MAX_TASK_SEEDS:
        raise ValueError("invalid cohort seed allocation")
    if (
        "seed_episodes" in payload
        and episode_ids(payload["seed_episodes"], "fit seed_episodes") != allocated
    ):
        raise ValueError("fit seed allocation differs from requested cohort")
    for key, expected in (
        ("label_scope", "segment_count"),
        ("n_segments", spec["n_segments"]),
        ("task_id", spec["task"]),
        ("arm", arm),
    ):
        if payload.get(key) != expected:
            raise ValueError(f"fit {key}={payload.get(key)!r}, expected {expected!r}")
    fitted = episode_ids(payload.get("fit_on_episodes"), "fit_on_episodes")
    if (
        payload.get("fit_episode_count") != len(fitted)
        or not min_cohort <= len(fitted) <= MAX_TASK_SEEDS
    ):
        raise ValueError("invalid fitted episode count")
    if not set(fitted) <= set(allocated) <= set(task_seed):
        raise ValueError("fit uses episodes outside the cohort's allocated task seeds")
    return fitted


def evaluation_groups(
    tasks: dict[str, Any],
    truth: dict[str, Any],
    identities: dict[int, int],
    min_cohort: int,
    *,
    prefix: str = "study__",
    namespace: str = "study_episode_index",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Export ALL held-out task/count groups, independent of either arm's fit."""
    groups, metadata = {}, {}
    for task, split in sorted(tasks.items()):
        seeds_by_count, eval_by_count = defaultdict(list), defaultdict(list)
        for episode in split["seed"]:
            seeds_by_count[len(truth[str(episode)])].append(identities[episode])
        for episode in split["eval"]:
            eval_by_count[len(truth[str(episode)])].append(identities[episode])
        for count, evaluation in sorted(eval_by_count.items()):
            cohort = f"{task}__n{count}"
            dataset = cohort if cohort.startswith(prefix) else f"{prefix}{cohort}"
            seeds = sorted(seeds_by_count[count])
            supported = count >= 2 and len(seeds) >= min_cohort
            # Build only the evaluation episodes; seeds remain metadata and can
            # never enter the converted dataset that an inference job scores.
            groups[dataset] = {"seed": [], "eval": sorted(evaluation)}
            metadata[dataset] = {
                "task": task,
                "cohort": cohort,
                "n_segments": count,
                "n_seed": len(seeds),
                "seed_episodes": seeds,
                "task_seed_episodes": sorted(identities[e] for e in split["seed"]),
                "calibrated": supported,
                "calibration_eligible": supported,
                "calibration_eligible_source": "seed_count_support",
                "index_space": namespace,
                "identity_namespace": namespace,
            }
    return groups, metadata


def run(cmd: list[str], log: Path, timeout: float | None = None) -> tuple[bool, str]:
    """Run one step, bounded. An unbounded wait is not a safe default here.

    A replica can accept a connection, report the request as running, and then
    produce no tokens at all. The FIFO lease means the cohort holding that
    replica never returns it and never returns itself, so the driver waits on
    one thread forever with every other cohort already finished. Bounding the
    step turns that into one failed cohort routed to the uncalibrated fallback,
    which is a result the manifest records, rather than a hung run.
    """
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "LEROBOT_OPENAI_SEND_MM_KWARGS": "1"}
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as expired:

        def decode(raw: bytes | str | None) -> str:
            if raw is None:
                return ""
            return raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw

        tail = f"step exceeded {timeout:g} seconds"
        log.write_text(
            decode(expired.stdout)
            + "\n--- stderr ---\n"
            + decode(expired.stderr)
            + f"\n--- {tail} ---\n"
        )
        return False, tail
    log.write_text((proc.stdout or "") + "\n--- stderr ---\n" + (proc.stderr or ""))
    return proc.returncode == 0, (proc.stderr or proc.stdout or "")[-600:]


def main(
    *,
    default_prefix: str = "study__",
    default_namespace: str = "study_episode_index",
    default_arms_config: Path | None = None,
) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    for name in ("root", "tasks-splits", "ground-truth", "out-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    for name in ("arm", "eval-batch", "fit-bin", "model"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--arms-config",
        type=Path,
        default=default_arms_config or HERE.parent / "configs/alignment_protocol_arms.yaml",
    )
    parser.add_argument("--prefix", default=default_prefix)
    parser.add_argument("--identity-namespace", default=default_namespace)
    parser.add_argument(
        "--seed-sources",
        type=Path,
        help="study episode -> {root, episode}; supports seeds distributed across source roots",
    )
    parser.add_argument("--python", default="python")
    parser.add_argument("--ports", default="8000,8001,8002,8003,8004,8005,8006")
    parser.add_argument("--min-cohort", type=int, default=MIN_COHORT)
    parser.add_argument("--min-gain", default="3.0")
    parser.add_argument("--min-first-boundary-mae", default="1.5")
    parser.add_argument(
        "--force", action="store_true", help="override only the fitter recommendation gate"
    )
    parser.add_argument(
        "--step-timeout",
        type=float,
        default=2400.0,
        help="seconds one seed-inference or fit step may take before its cohort "
        "is failed and routed to the uncalibrated fallback (0 disables)",
    )
    parser.add_argument(
        "--index-map",
        type=Path,
        help="original trajectory identity map; defaults to <root>.index_map.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        truth = json.loads(args.ground_truth.read_text())["episodes"]
        tasks = validate_task_splits(json.loads(args.tasks_splits.read_text())["tasks"], truth)
        if not MIN_COHORT <= args.min_cohort <= MAX_TASK_SEEDS:
            raise ValueError("min-cohort must be between 3 and 10")
        regime, extra = fit_regime(args.arms_config, args.arm)
        port_names = [p.strip() for p in args.ports.split(",") if p.strip()]
        if not port_names or len(port_names) != len(set(port_names)):
            raise ValueError("ports must be a nonempty unique list")
        all_ids = {e for g in tasks.values() for values in g.values() for e in values}
        map_path = args.index_map or args.root.with_name(args.root.name + ".index_map.json")
        if map_path.exists():
            entries = json.loads(map_path.read_text())["episodes"]
            identities = {int(r["new_index"]): int(r["original_index"]) for r in entries}
            if len(identities) != len(entries) or len(set(identities.values())) != len(entries):
                raise ValueError("index map must be injective")
            if not all_ids <= identities.keys():
                raise ValueError("index map does not cover every seed/eval episode")
            namespace = args.identity_namespace
            if (
                not namespace
                or json.loads(map_path.read_text()).get("identity_namespace", namespace)
                != namespace
            ):
                raise ValueError("index map identity namespace differs from requested study")
        else:
            raise ValueError(
                f"missing original-trajectory index map {map_path}; provide --index-map"
            )
        seed_sources = json.loads(args.seed_sources.read_text()) if args.seed_sources else None
        if seed_sources is not None:
            allocated_seeds = {str(e) for g in tasks.values() for e in g["seed"]}
            if set(seed_sources) != allocated_seeds:
                raise ValueError("seed sources must cover exactly the allocated seeds")
            source_ids = [
                (str(Path(r["root"]).resolve()), r["episode"]) for r in seed_sources.values()
            ]
            if any(type(e) is not int or e < 0 for _, e in source_ids) or len(
                set(source_ids)
            ) != len(source_ids):
                raise ValueError(
                    "seed sources must contain unique nonnegative source episode identities"
                )
    except (ValueError, KeyError) as exc:
        raise SystemExit(str(exc)) from exc
    step_timeout = args.step_timeout if args.step_timeout and args.step_timeout > 0 else None
    cohorts, routing = {}, {}
    for task, groups in tasks.items():
        by_count = defaultdict(list)
        for episode in groups["seed"]:
            by_count[len(truth[str(episode)])].append(episode)
        for count, seeds in sorted(by_count.items()):
            if count >= 2 and len(seeds) >= args.min_cohort:
                cohorts[f"{task}__n{count}"] = {
                    "task": task,
                    "n_segments": count,
                    "seed_episodes": seeds,
                    "n_seed": len(seeds),
                }
        for episode in groups["eval"]:
            count = len(truth[str(episode)])
            key = f"{task}__n{count}"
            routing[str(episode)] = {
                "task": task,
                "n_segments": count,
                "cohort": key if key in cohorts else None,
                "calibrated": key in cohorts,
            }
    groups, group_meta = evaluation_groups(
        tasks, truth, identities, args.min_cohort, prefix=args.prefix, namespace=namespace
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "arm": args.arm,
                    "task_splits": tasks,
                    "cohorts": cohorts,
                    "fit_regime": regime,
                    "batch_flags": extra,
                    "evaluation_groups": groups,
                    "cohort_group_meta": group_meta,
                },
                indent=1,
            )
        )
        return 0
    out_dir = args.out_dir / args.arm
    if (out_dir / "calibration_routing.json").exists():
        raise SystemExit(
            f"{out_dir}: completed calibration manifest is frozen; choose a new --out-dir"
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("evaluation_groups.json", groups),
        ("cohort_group_meta.json", group_meta),
    ):
        destination = out_dir / name
        temporary = destination.with_suffix(".json.partial")
        temporary.write_text(json.dumps(payload, indent=1))
        temporary.replace(destination)
    ports: queue.Queue[str] = queue.Queue()
    for port in port_names:
        ports.put(port)

    def fit(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        key, spec = item
        port = ports.get()
        try:
            # A new candidate path prevents stale success on a refused/no-op fitter.
            with tempfile.TemporaryDirectory(prefix=f"{key}.", dir=out_dir) as temporary:
                predictions = Path(temporary) / "predictions.json"
                candidate = Path(temporary) / "calibration.json"
                episodes = [str(e) for e in spec["seed_episodes"]]
                if seed_sources is None:
                    batches = {str(args.root): {int(e): int(e) for e in episodes}}
                else:
                    batches = defaultdict(dict)
                    for e in episodes:
                        source = seed_sources[e]
                        batches[source["root"]][source["episode"]] = int(e)
                collected, ok, tail = [], True, ""
                for batch_index, (source_root, local_to_study) in enumerate(
                    sorted(batches.items())
                ):
                    batch_output = Path(temporary) / f"batch-{batch_index}.json"
                    ok, tail = run(
                        [
                            args.python,
                            args.eval_batch,
                            source_root,
                            "--episodes",
                            *map(str, sorted(local_to_study)),
                            *extra,
                            "--model",
                            args.model,
                            "--api-base",
                            f"http://127.0.0.1:{port}/v1",
                            "--out",
                            str(batch_output),
                        ],
                        out_dir / f"{key}.{batch_index}.batch.log",
                        timeout=step_timeout,
                    )
                    if not ok or not batch_output.exists():
                        ok = False
                        tail = tail or "missing seed inference output"
                        break
                    for row in json.loads(batch_output.read_text()):
                        if row["episode"] not in local_to_study:
                            raise ValueError("seed inference returned an unallocated episode")
                        collected.append({**row, "episode": local_to_study[row["episode"]]})
                if ok:
                    predictions.write_text(json.dumps(collected))
                if not ok or not predictions.exists():
                    return key, {
                        **spec,
                        "ok": False,
                        "stage": "eval_align_batch",
                        "error": tail or "missing predictions",
                    }
                prediction_sha = hashlib.sha256(predictions.read_bytes()).hexdigest()
                saved_predictions = out_dir / f"{key}.{prediction_sha}.fit_predictions.json"
                predictions.replace(saved_predictions)
                cmd = [
                    args.fit_bin,
                    str(args.root),
                    str(saved_predictions),
                    "--out",
                    str(candidate),
                    "--format",
                    regime["frame_format"],
                    "--label-scope",
                    "segment_count",
                    "--episodes",
                    *episodes,
                    "--min-fit-episodes",
                    str(args.min_cohort),
                    "--max-fit-episodes",
                    str(MAX_TASK_SEEDS),
                    "--task-id",
                    spec["task"],
                    "--arm",
                    args.arm,
                    "--min-gain",
                    args.min_gain,
                    "--min-first-boundary-mae",
                    args.min_first_boundary_mae,
                ]
                if args.force:
                    cmd += ["--force"]
                ok, tail = run(cmd, out_dir / f"{key}.fit.log", timeout=step_timeout)
                if not ok or not candidate.exists():
                    return key, {
                        **spec,
                        "ok": False,
                        "stage": "fit",
                        "error": tail or "missing fit output",
                    }
                payload = json.loads(candidate.read_text())
                fitted = validate_fit(
                    payload, spec, args.arm, tasks[spec["task"]]["seed"], args.min_cohort
                )
                digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
                frozen = out_dir / f"{key}.{digest}.json"
                candidate.replace(frozen)
                entry = {
                    "path": str(frozen.resolve()),
                    "sha256": digest,
                    "task_id": spec["task"],
                    "arm": args.arm,
                    "n_segments": spec["n_segments"],
                    "fit_on_episodes": fitted,
                    "fit_episode_count": len(fitted),
                    "task_seed_episodes": tasks[spec["task"]]["seed"],
                    "fit_predictions": str(saved_predictions.resolve()),
                    "fit_predictions_sha256": prediction_sha,
                }
                return key, {**spec, "ok": True, "stage": "done", "fit_file": entry}
        except Exception as exc:
            return key, {**spec, "ok": False, "stage": "validation", "error": str(exc)}
        finally:
            ports.put(port)

    with ThreadPoolExecutor(max_workers=len(port_names)) as executor:
        results = dict(executor.map(fit, sorted(cohorts.items())))
    fit_files = {k: r["fit_file"] for k, r in results.items() if r["ok"]}
    for row in routing.values():
        if row["cohort"] not in fit_files:
            row["cohort"], row["calibrated"] = None, False
    resolved = Counter(r["calibrated"] for r in routing.values())
    manifest = {
        "schema_version": 2,
        "arm": args.arm,
        "model_id": args.model,
        "root": str(args.root.resolve()),
        "fit_regime": regime,
        "task_splits": tasks,
        "identity_namespace": namespace,
        "seed_sources": seed_sources,
        "seed_sources_sha256": hashlib.sha256(args.seed_sources.read_bytes()).hexdigest()
        if args.seed_sources
        else None,
        "evaluation_groups": str((out_dir / "evaluation_groups.json").resolve()),
        "cohort_group_meta": str((out_dir / "cohort_group_meta.json").resolve()),
        "episode_identities": {str(e): identities[e] for e in sorted(all_ids)},
        "min_cohort": args.min_cohort,
        "max_task_seeds": MAX_TASK_SEEDS,
        "forced": args.force,
        "ports": port_names,
        "step_timeout_seconds": step_timeout,
        "n_cohorts": len(cohorts),
        "n_cohorts_fitted": len(fit_files),
        "n_cohorts_failed": len(cohorts) - len(fit_files),
        "eval_trajectories": len(routing),
        "calibrated": resolved[True],
        "uncalibrated_fallback": resolved[False],
        "fit_files": fit_files,
        "cohorts": cohorts,
        "results": results,
        "routing": routing,
    }
    destination = out_dir / "calibration_routing.json"
    temporary_manifest = destination.with_suffix(".json.partial")
    temporary_manifest.write_text(json.dumps(manifest, indent=1))
    temporary_manifest.replace(destination)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "arm",
                    "n_cohorts_fitted",
                    "n_cohorts_failed",
                    "calibrated",
                    "uncalibrated_fallback",
                )
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
