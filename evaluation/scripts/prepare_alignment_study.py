#!/usr/bin/env python3
"""Normalize prepared LeRobot sources into the common alignment study contract.

Raw-data conversion stays outside this script. Task membership is explicit;
sources may contain one task or an episode->task map. Seeds are shared across
sources, cameras and count cohorts. Evaluation datasets are physically split
using LeRobot's splitter so annotation never touches unselected seed shards.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re

import yaml

from fit_task_calibration import evaluation_groups, validate_task_splits

HERE = Path(__file__).resolve().parent
PROTOCOL = {
    "version": 1,
    "seed_budget": 10,
    "min_cohort": 3,
    "label_scope": "segment_count",
    "min_gain": 3.0,
    "min_first_boundary_mae": 1.5,
    "force": False,
    "population": "all_scored",
    "bootstrap_unit": "task",
    "seed": 1729,
}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def slug(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"unsafe or empty study/source/task identifier: {value!r}")
    return value


def normalized_spans(spans):
    return [
        {"text": s.get("text", s.get("label")), "start": s["start"], "end": s["end"]} for s in spans
    ]


def resolve(path, base):
    expanded = os.path.expandvars(str(path))
    if re.search(r'\$[{A-Za-z_]', expanded):
        raise ValueError(f'Unset source-path environment variable in {path!r}; see evaluation/RELEASE_DATA.md')
    path = Path(expanded)
    return (path if path.is_absolute() else base / path).resolve()


def split_component(job):
    """Materialize one independent component in an isolated process."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.dataset_tools import split_dataset

    name, source_id, source_root, selected, components_root = job
    dataset = LeRobotDataset(repo_id=source_id, root=Path(source_root))
    split_dataset(dataset, {name: selected}, output_dir=Path(components_root))


def plan_study(path):
    config = json.loads(path.read_text())
    if config.get("version") != 1:
        raise ValueError("expected source manifest version 1")
    name = slug(config["name"])
    namespace = f"{name}:source_episode"
    cameras = config["cameras"]
    if set(cameras) != {"primary", "secondary"} or any(
        not isinstance(v, str) or not v for v in cameras.values()
    ):
        raise ValueError("declare primary and secondary camera feature names")
    sources = config["sources"]
    if not sources or len({s["id"] for s in sources}) != len(sources):
        raise ValueError("source ids must be nonempty and unique")
    manual = any("seed_episodes" in s for s in sources)
    if manual and not all("seed_episodes" in s for s in sources):
        raise ValueError(
            "explicit seed allocations require seed_episodes on every source (possibly empty)"
        )
    records, truth, source_map, hashes, physical = {}, {}, {}, {}, set()
    for source in sorted(sources, key=lambda s: s["id"]):
        source_id = slug(source["id"])
        root = resolve(source["root"], path.parent)
        gt_path = resolve(source["ground_truth"], path.parent)
        data = gt_path.read_bytes()
        hashes[str(gt_path)] = hashlib.sha256(data).hexdigest()
        gt = json.loads(data)["episodes"]
        episodes = source.get("episodes", sorted(map(int, gt)))
        if len(episodes) != len(set(episodes)) or any(
            type(e) is not int or e < 0 for e in episodes
        ):
            raise ValueError(f"{source_id}: invalid episode selection")
        explicit_seeds = source.get("seed_episodes", [])
        if (
            any(type(e) is not int or e < 0 for e in explicit_seeds)
            or len(explicit_seeds) != len(set(explicit_seeds))
            or not set(explicit_seeds) <= set(episodes)
        ):
            raise ValueError(
                f"{source_id}: seed allocation outside selected episodes or duplicated"
            )
        source_map[source_id] = {"root": str(root), "ground_truth": str(gt_path)}
        annotations_path = root / "meta/lerobot_annotations.json"
        annotations = json.loads(annotations_path.read_text())["episodes"]
        hashes[str(annotations_path)] = hashlib.sha256(annotations_path.read_bytes()).hexdigest()
        info_path = root / "meta/info.json"
        info = json.loads(info_path.read_text())
        hashes[str(info_path)] = hashlib.sha256(info_path.read_bytes()).hexdigest()
        for camera in cameras.values():
            if camera not in info["features"]:
                raise ValueError(f"{source_id}: missing camera {camera}")
        if info.get("codebase_version") != "v3.0":
            raise ValueError(f"{source_id}: convert to LeRobot v3.0 before preparing a study")
        for local in sorted(episodes):
            key = (str(root), local)
            if key in physical:
                raise ValueError(f"source trajectory appears twice: {key}")
            physical.add(key)
            task = slug(source.get("task") or source.get("tasks", {}).get(str(local)))
            spans = normalized_spans(gt[str(local)])
            if not spans or any(
                not isinstance(s["text"], str) or not s["text"].strip() for s in spans
            ):
                raise ValueError(f"{source_id}/{local}: missing or invalid labels")
            # Batch inference reads source annotations, not the exported GT.
            if spans != normalized_spans(annotations.get(str(local), {}).get("subtasks", [])):
                raise ValueError(
                    f"{source_id}/{local}: source annotations differ from exported ground truth"
                )
            index = len(records)
            records[str(index)] = {
                "source": source_id,
                "root": str(root),
                "episode": local,
                "task": task,
                "explicit_seed": local in explicit_seeds,
            }
            truth[str(index)] = spans
    by_task = defaultdict(list)
    for index, row in records.items():
        by_task[row["task"]].append(int(index))
    tasks = {}
    for task, indices in sorted(by_task.items()):
        if manual:
            seeds = [e for e in indices if records[str(e)]["explicit_seed"]]
        else:
            # Hash stable source identities, not timings, labels or model quality.
            def priority(e, task=task):
                r = records[str(e)]
                return hashlib.sha256(
                    f"{PROTOCOL['seed']}:{task}:{r['source']}:{r['episode']}".encode()
                ).hexdigest()

            seeds = sorted(indices, key=priority)[: PROTOCOL["seed_budget"]]
        if len(seeds) != PROTOCOL["seed_budget"] or len(indices) <= len(seeds):
            raise ValueError(
                f"{task}: require exactly ten allocated seeds and at least one remaining trajectory"
            )
        tasks[task] = {"seed": sorted(seeds), "eval": sorted(set(indices) - set(seeds))}
    validate_task_splits(tasks, truth)
    _, meta = evaluation_groups(
        tasks, truth, {int(e): int(e) for e in records}, 3, prefix=f"{name}__", namespace=namespace
    )
    components = {}
    for task, split in tasks.items():
        for index in split["eval"]:
            row = records[str(index)]
            cohort = f"{task}__n{len(truth[str(index)])}"
            dataset = f"{name}__{cohort}__{row['source']}"
            component = components.setdefault(
                dataset, {**meta[f"{name}__{cohort}"], "source": row["source"], "episodes": []}
            )
            component["episodes"].append(index)
    return {
        "version": 1,
        "name": name,
        "identity_namespace": namespace,
        "protocol": dict(PROTOCOL),
        "cameras": cameras,
        "sources": source_map,
        "episodes": records,
        "truth": truth,
        "tasks": tasks,
        "components": components,
        "input_sha256": hashes,
        "source_manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def materialize(plan, out):
    if out.exists():
        raise ValueError(f"{out}: use a new preparation directory")
    out.mkdir(parents=True)
    write(out / "preparation_plan.json", plan)
    root = out / "fit_root"
    allocated_seeds = {e for s in plan["tasks"].values() for e in s["seed"]}
    write(
        root / "meta/lerobot_annotations.json",
        {
            "episodes": {
                str(e): {"subtasks": [{**s, "label": s["text"]} for s in plan["truth"][str(e)]]}
                for e in sorted(allocated_seeds)
            }
        },
    )
    write(
        out / "fit_root.index_map.json",
        {
            "identity_namespace": plan["identity_namespace"],
            "episodes": [{"new_index": int(e), "original_index": int(e)} for e in plan["episodes"]],
        },
    )
    write(out / "ground_truth.json", {"episodes": plan["truth"]})
    write(out / "task_splits.json", {"tasks": plan["tasks"]})
    write(
        out / "seed_sources.json",
        {
            str(e): {k: plan["episodes"][str(e)][k] for k in ("root", "episode")}
            for e in sorted(allocated_seeds)
        },
    )
    arms = yaml.safe_load((HERE.parent / "configs/alignment_protocol_arms.yaml").read_text())
    for arm in arms["arms"]:
        for flag, value in (arm.get("flags") or {}).items():
            if isinstance(value, str):
                for role, camera in plan["cameras"].items():
                    value = value.replace("{camera_" + role + "}", camera)
                arm["flags"][flag] = value
    (out / "arms.yaml").write_text(yaml.safe_dump(arms, sort_keys=False))
    split_jobs = []
    for name, component in plan["components"].items():
        source_id = component["source"]
        selected = sorted(plan["episodes"][str(e)]["episode"] for e in component["episodes"])
        split_jobs.append((name, component, source_id, selected))

    workers = int(os.environ.get("ALIGN_PREPARE_WORKERS", "8"))
    if workers < 1:
        raise ValueError("ALIGN_PREPARE_WORKERS must be positive")
    process_jobs = [
        (name, source_id, plan["sources"][source_id]["root"], selected, str(out / "components"))
        for name, _, source_id, selected in split_jobs
    ]
    # LeRobot's splitter uses a process-aware tqdm lock internally. Calling it
    # from several threads races while deleting that lock, so concurrency must
    # be isolated at the process boundary.
    if workers == 1 or len(process_jobs) == 1:
        for job in process_jobs:
            split_component(job)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(process_jobs))) as executor:
            list(executor.map(split_component, process_jobs))

    for name, component, _, selected in split_jobs:
        global_by_local = {plan["episodes"][str(e)]["episode"]: e for e in component["episodes"]}
        mapping = {i: global_by_local[e] for i, e in enumerate(selected)}
        write(
            out / "components" / f"{name}.index_map.json",
            {
                "identity_namespace": plan["identity_namespace"],
                "episodes": [{"new_index": i, "original_index": e} for i, e in mapping.items()],
            },
        )
        write(
            out / "gt" / f"{name}.json",
            {
                "dataset": name,
                "episodes": {str(i): plan["truth"][str(e)] for i, e in mapping.items()},
            },
        )
        write(
            out / "splits" / f"{name}.json",
            {"dataset": name, "seed": [], "eval": list(mapping)},
        )
    write(out / "group_meta.json", plan["components"])
    for task, split in plan["tasks"].items():
        dataset = f"{plan['name']}__{task}"
        write(
            out / "task_gt" / f"{dataset}.json",
            {
                "dataset": dataset,
                "episodes": {str(e): plan["truth"][str(e)] for ids in split.values() for e in ids},
            },
        )
        write(out / "task_eval_splits" / f"{dataset}.json", {"dataset": dataset, **split})
    artifacts = [
        "task_splits.json",
        "seed_sources.json",
        "ground_truth.json",
        "arms.yaml",
        "group_meta.json",
        "fit_root.index_map.json",
        "fit_root/meta/lerobot_annotations.json",
    ]
    artifacts.append("preparation_plan.json")
    for folder in ("gt", "splits", "task_gt", "task_eval_splits"):
        artifacts.extend(str(p.relative_to(out)) for p in sorted((out / folder).glob("*.json")))
    artifacts.extend(
        str(p.relative_to(out)) for p in sorted((out / "components").glob("*.index_map.json"))
    )
    hashes = {p: hashlib.sha256((out / p).read_bytes()).hexdigest() for p in artifacts}
    write(
        out / "study.json",
        {
            "version": 1,
            "name": plan["name"],
            "identity_namespace": plan["identity_namespace"],
            "protocol": PROTOCOL,
            "artifacts_sha256": hashes,
            "arms_template_sha256": hashlib.sha256(
                (HERE.parent / "configs/alignment_protocol_arms.yaml").read_bytes()
            ).hexdigest(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = plan_study(args.manifest)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        materialize(plan, args.out.resolve())


if __name__ == "__main__":
    main()
