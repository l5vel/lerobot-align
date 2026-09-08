#!/usr/bin/env python3
"""One fit -> inference -> floors -> score -> aggregate entry point for any corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

from prepare_alignment_study import PROTOCOL, write

HERE = Path(__file__).resolve().parent
STAGES = ("fit", "run", "score", "aggregate")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commands(study_dir, out, model, python, ports, timeout, n_boot):
    study = json.loads((study_dir / "study.json").read_text())
    if study.get("version") != 1 or study.get("protocol") != PROTOCOL:
        raise ValueError("study was prepared under a different evaluation protocol")
    if study.get("arms_template_sha256") != sha(
        HERE.parent / "configs/alignment_protocol_arms.yaml"
    ):
        raise ValueError("shared arm protocol changed; prepare a new study")
    for name, digest in study["artifacts_sha256"].items():
        if sha(study_dir / name) != digest:
            raise ValueError(f"prepared study artifact changed: {name}")
    plan = json.loads((study_dir / "preparation_plan.json").read_text())
    for source, digest in plan["input_sha256"].items():
        if sha(Path(source)) != digest:
            raise ValueError(f"source metadata or ground truth changed: {source}")
    arms = yaml.safe_load((study_dir / "arms.yaml").read_text())["arms"]
    names = [a["name"] for a in arms if a["tool"] != "reference"]
    calibrated = [
        a["name"] for a in arms if "plan.subtask_align_calibration_path" in a.get("flags", {})
    ]
    script = lambda name: [python, str(HERE / (name + ".py"))]  # noqa: E731
    fits = [out / "fits" / a / "calibration_routing.json" for a in calibrated]
    shared = ["--arms-config", str(study_dir / "arms.yaml")]
    result = {stage: [] for stage in STAGES}
    for arm in calibrated:
        result["fit"].append(
            script("fit_task_calibration")
            + [
                "--root",
                str(study_dir / "fit_root"),
                "--tasks-splits",
                str(study_dir / "task_splits.json"),
                "--ground-truth",
                str(study_dir / "ground_truth.json"),
                "--seed-sources",
                str(study_dir / "seed_sources.json"),
                "--out-dir",
                str(out / "fits"),
                "--arm",
                arm,
                "--model",
                model,
                "--python",
                python,
                "--eval-batch",
                str(HERE.parents[1] / "src/lerobot_align/diagnostics/eval_align_batch.py"),
                "--fit-bin",
                str(Path(python).parent / "lerobot-align-fit"),
                "--prefix",
                study["name"] + "__",
                "--identity-namespace",
                study["identity_namespace"],
                "--ports",
                ports,
                "--step-timeout",
                str(timeout),
                "--min-cohort",
                "3",
                "--min-gain",
                "3.0",
                "--min-first-boundary-mae",
                "1.5",
                *shared,
            ]
        )
    result["run"].append(
        script("make_alignment_labels")
        + [
            "--gt-dir",
            str(study_dir / "gt"),
            "--splits-dir",
            str(study_dir / "splits"),
            "--out-dir",
            str(out / "labels"),
        ]
    )
    result["run"].append(
        script("run_alignment_arms")
        + [
            "--group-meta",
            str(study_dir / "group_meta.json"),
            "--source-dir",
            str(study_dir / "components"),
            "--splits-dir",
            str(study_dir / "splits"),
            "--gt-dir",
            str(study_dir / "gt"),
            "--labels-dir",
            str(out / "labels"),
            *[v for f in fits for v in ("--cohort-files", str(f))],
            "--arms",
            *names,
            "--model-id",
            model,
            "--python",
            python,
            "--ports",
            ports,
            "--out-dir",
            str(out / "predictions"),
            "--work-dir",
            str(out / "work"),
            "--job-splits-dir",
            str(out / "job-splits"),
            *shared,
        ]
    )
    result["score"].append(
        script("merge_alignment_predictions")
        + [
            "--predictions-dir",
            str(out / "predictions"),
            "--group-meta",
            str(study_dir / "group_meta.json"),
            "--cohorts-dir",
            str(study_dir / "components"),
            "--gt-dir",
            str(study_dir / "gt"),
            "--out-predictions",
            str(out / "merged-predictions"),
            "--out-gt",
            str(out / "merged-gt"),
            "--out-splits",
            str(out / "merged-splits"),
            "--eligibility-out",
            str(out / "eligibility.json"),
            "--prefix",
            study["name"] + "__",
            "--identity-namespace",
            study["identity_namespace"],
            *shared,
        ]
    )
    result["score"].append(
        script("align_floors")
        + [
            "--gt",
            str(study_dir / "task_gt"),
            "--split",
            str(study_dir / "task_eval_splits"),
            "--prior-scope",
            "segment_count",
            "--min-fit-episodes",
            "3",
            "--out-dir",
            str(out / "floors"),
        ]
    )
    for name, predictions in [("arms", "merged-predictions"), ("floors", "floors")]:
        result["score"].append(
            script("score_alignment")
            + [
                "--predictions-dir",
                str(out / predictions),
                "--gt-dir",
                str(out / "merged-gt"),
                "--splits-dir",
                str(out / "merged-splits"),
                "--eligibility",
                str(out / "eligibility.json"),
                "--out",
                str(out / f"scores_{name}.jsonl"),
            ]
        )
    for eligible in (False, True):
        result["aggregate"].append(
            script("aggregate")
            + [
                "--scores",
                str(out / "scores.jsonl"),
                "--out",
                str(out / ("aggregate_eligible.json" if eligible else "aggregate.json")),
                "--matcher",
                "any",
                "--contrasts",
                str(HERE.parent / "configs/alignment_protocol_contrasts.yaml"),
                "--output-population",
                "all_scored",
                "--n-boot",
                str(n_boot),
                *shared,
                *(["--require-true", "calibration_eligible"] if eligible else []),
            ]
        )
    return study, result


def combine_scores(out):
    rows, keys_by_arm = [], {}
    for name in ("arms", "floors"):
        for line in (out / f"scores_{name}.jsonl").read_text().splitlines():
            row = json.loads(line)
            key = (row["dataset"], row["episode"])
            keys = keys_by_arm.setdefault(row["arm"], set())
            if key in keys:
                raise ValueError(f"duplicate score row {row['arm']}/{key}")
            keys.add(key)
            rows.append(row)
    expected = {
        a["name"]
        for a in yaml.safe_load((HERE.parent / "configs/alignment_protocol_arms.yaml").read_text())[
            "arms"
        ]
    }
    if set(keys_by_arm) != expected or any(
        keys != next(iter(keys_by_arm.values())) for keys in keys_by_arm.values()
    ):
        raise ValueError("all nine arms must cover exactly the same requested trajectories")
    (out / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def run_command(stage, command, out, environment):
    """Run one stage, accepting only a complete, explicitly recorded inference failure set."""
    completed = subprocess.run(command, check=False, env=environment)
    if completed.returncode == 0:
        return
    is_inference = stage == "run" and Path(command[1]).name == "run_alignment_arms.py"
    summary_path = out / "predictions/run_summary.json"
    manifest_path = out / "predictions/jobs_manifest.json"
    if (
        is_inference
        and completed.returncode == 1
        and summary_path.exists()
        and manifest_path.exists()
    ):
        summary = json.loads(summary_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        expected = len(manifest.get("jobs", []))
        recorded = int(summary.get("ok", -1)) + int(summary.get("failed", -1))
        records = list((out / "predictions").glob("*.json"))
        records = [p for p in records if p.name not in {"jobs_manifest.json", "run_summary.json"}]
        if summary.get("jobs") == expected == recorded == len(records):
            print(
                f"[run] continuing to score {summary['failed']} explicitly recorded failed jobs",
                flush=True,
            )
            return
    raise subprocess.CalledProcessError(completed.returncode, command)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study",
        type=Path,
        required=True,
        help="prepared directory from prepare_alignment_study.py",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--ports", default="8001,8002,8003,8004,8005,8006")
    parser.add_argument("--step-timeout", type=float, default=2400)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--stage", choices=(*STAGES, "all"), default="all")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.n_boot < 1 or args.step_timeout <= 0:
        parser.error("n-boot and step-timeout must be positive")
    study_dir, out = args.study.resolve(), args.out.resolve()
    study, steps = commands(
        study_dir, out, args.model, args.python, args.ports, args.step_timeout, args.n_boot
    )
    identity = {
        "study_sha256": sha(study_dir / "study.json"),
        "model": args.model,
        "protocol": PROTOCOL,
        "arms_sha256": sha(study_dir / "arms.yaml"),
        "contrasts_sha256": sha(HERE.parent / "configs/alignment_protocol_contrasts.yaml"),
        "n_boot": args.n_boot,
    }
    if args.dry_run:
        print(json.dumps({"run": identity, "stages": steps}, indent=2))
        return
    marker = out / "run_manifest.json"
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(
                "existing run uses different study/model/protocol settings; use a new output directory"
            )
    elif out.exists() and any(out.iterdir()):
        raise ValueError("output directory is nonempty without a run manifest")
    write(marker, identity)
    environment = dict(os.environ, LEROBOT_OPENAI_SEND_MM_KWARGS="1")
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in stages:
        if stage == "aggregate":
            combine_scores(out)
        for command in steps[stage]:
            print(f"[{stage}] " + " ".join(command), flush=True)
            run_command(stage, command, out, environment)
    print(f"Completed {args.stage}: {out}", flush=True)


if __name__ == "__main__":
    main()
