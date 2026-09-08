"""Shared A/B protocol, cross-source seed budget, and frozen input regressions."""

import hashlib
import json
from pathlib import Path
import sys

import pytest
import yaml

SCRIPTS = Path(__file__).resolve().parents[1] / "evaluation/scripts"
sys.path.insert(0, str(SCRIPTS))
import prepare_alignment_study as prepare  # noqa: E402
import evaluate_alignment_study as evaluate  # noqa: E402
import fit_task_calibration as fitter  # noqa: E402


def source_manifest(tmp_path, name="corpus_a", secondary="external"):
    sources = []
    for source_id in ("part_a", "part_b"):
        root = tmp_path / source_id
        spans = [{"text": "reach", "start": 0, "end": 1}, {"text": "place", "start": 1, "end": 2}]
        truth = {str(e): spans for e in range(12)}
        prepare.write(root / "gt.json", {"episodes": truth})
        prepare.write(
            root / "meta/lerobot_annotations.json",
            {"episodes": {e: {"subtasks": s} for e, s in truth.items()}},
        )
        prepare.write(
            root / "meta/info.json",
            {"codebase_version": "v3.0", "features": {"wrist": {}, secondary: {}}},
        )
        sources.append(
            {
                "id": source_id,
                "root": str(root),
                "ground_truth": str(root / "gt.json"),
                "task": "shared_task",
            }
        )
    manifest = tmp_path / "sources.json"
    prepare.write(
        manifest,
        {
            "version": 1,
            "name": name,
            "cameras": {"primary": "wrist", "secondary": secondary},
            "sources": sources,
        },
    )
    return manifest


def prepared(tmp_path, monkeypatch, name="corpus_a", secondary="external"):
    # Exercise all production file/mapping generation. Replace only expensive media I/O.
    import lerobot.datasets.lerobot_dataset as dataset_module
    import lerobot.datasets.dataset_tools as dataset_tools

    calls = []
    monkeypatch.setattr(dataset_module, "LeRobotDataset", lambda **kwargs: kwargs)
    monkeypatch.setenv("ALIGN_PREPARE_WORKERS", "1")

    def split(dataset, selections, output_dir):
        calls.append((dataset, selections))
        for component in selections:
            (output_dir / component).mkdir(parents=True)

    monkeypatch.setattr(dataset_tools, "split_dataset", split)
    plan = prepare.plan_study(source_manifest(tmp_path, name, secondary))
    out = tmp_path / "prepared"
    prepare.materialize(plan, out)
    return out, plan, calls


def test_one_budget_across_sources_and_counts(tmp_path):
    manifest = source_manifest(tmp_path)
    plan = prepare.plan_study(manifest)
    assert plan == prepare.plan_study(manifest)
    task = plan["tasks"]["shared_task"]
    assert len(task["seed"]) == 10
    assert len(task["eval"]) == 14
    assert set(task["seed"]).isdisjoint(task["eval"])
    assert len({plan["episodes"][str(e)]["source"] for e in task["seed"]}) == 2
    assert {e for c in plan["components"].values() for e in c["episodes"]} == set(task["eval"])
    assert all(c["calibration_eligible"] for c in plan["components"].values())


@pytest.mark.parametrize(
    "failure", ["double_budget", "duplicate_source", "missing_task", "gt_mismatch"]
)
def test_invalid_preparation_rejected(tmp_path, failure):
    path = source_manifest(tmp_path)
    data = json.loads(path.read_text())
    if failure == "double_budget":
        for source in data["sources"]:
            source["seed_episodes"] = list(range(10))
    elif failure == "duplicate_source":
        data["sources"][1]["root"] = data["sources"][0]["root"]
    elif failure == "missing_task":
        del data["sources"][0]["task"]
    else:
        gt_path = Path(data["sources"][0]["ground_truth"])
        gt = json.loads(gt_path.read_text())
        gt["episodes"]["0"][0]["end"] = 0.5
        prepare.write(gt_path, gt)
    prepare.write(path, data)
    with pytest.raises(ValueError):
        prepare.plan_study(path)


def test_materialization_mapping_and_common_commands(tmp_path, monkeypatch):
    normalized = []
    for name, secondary in [("corpus_a", "right"), ("corpus_b", "primary")]:
        study, plan, calls = prepared(tmp_path / name, monkeypatch, name, secondary)
        seed_ids = set(plan["tasks"]["shared_task"]["seed"])
        eval_ids = set(plan["tasks"]["shared_task"]["eval"])
        sidecar = json.loads((study / "fit_root/meta/lerobot_annotations.json").read_text())
        assert set(map(int, sidecar["episodes"])) == seed_ids
        seen = set()
        for dataset, selections in calls:
            for component, local_ids in selections.items():
                mapping = json.loads(
                    (study / "components" / f"{component}.index_map.json").read_text()
                )
                for row, local in zip(mapping["episodes"], sorted(local_ids), strict=True):
                    global_id = row["original_index"]
                    seen.add(global_id)
                    assert global_id in eval_ids and global_id not in seed_ids
                    assert plan["episodes"][str(global_id)]["episode"] == local
                    assert plan["episodes"][str(global_id)]["root"] == str(dataset["root"])
        assert seen == eval_ids
        arms = yaml.safe_load((study / "arms.yaml").read_text())
        assert "{camera_" not in (study / "arms.yaml").read_text()
        stack = next(a for a in arms["arms"] if a["name"] == "align_video_stack")
        assert json.loads(stack["flags"]["plan.subtask_align_camera_keys"]) == ["wrist", secondary]
        _, commands = evaluate.commands(
            study, tmp_path / "run", "same_model", sys.executable, "8001", 2400, 100
        )
        assert len(commands["fit"]) == 2
        for cmd in commands["fit"]:
            assert "--force" not in cmd
            assert cmd[cmd.index("--min-cohort") + 1] == "3"
            assert "--seed-sources" in cmd
        for cmd in commands["aggregate"]:
            assert cmd[cmd.index("--output-population") + 1] == "all_scored"
            assert cmd[cmd.index("--contrasts") + 1].endswith("alignment_protocol_contrasts.yaml")
        normalized.append(json.dumps(commands).replace(str(study), "STUDY").replace(name, "CORPUS"))
    assert normalized[0] == normalized[1]


@pytest.mark.parametrize(
    "relative", ["task_eval_splits/corpus_a__shared_task.json", "gt", "source"]
)
def test_changed_scoring_or_source_inputs_rejected(tmp_path, monkeypatch, relative):
    study, plan, _ = prepared(tmp_path, monkeypatch)
    path = (
        Path(next(iter(plan["input_sha256"])))
        if relative == "source"
        else next((study / "gt").glob("*.json"))
        if relative == "gt"
        else study / relative
    )
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        evaluate.commands(study, tmp_path / "run", "model", sys.executable, "8001", 2400, 100)


def test_fit_combines_local_seed_ids_from_multiple_roots(tmp_path, monkeypatch):
    study, plan, _ = prepared(tmp_path, monkeypatch)
    calls = []
    sources = json.loads((study / "seed_sources.json").read_text())

    def run(cmd, log, timeout=None):
        calls.append(cmd)
        start = cmd.index("--episodes") + 1
        end = next(i for i in range(start, len(cmd)) if cmd[i].startswith("--"))
        ids = list(map(int, cmd[start:end]))
        output = Path(cmd[cmd.index("--out") + 1])
        if "batch" in cmd:
            root = cmd[cmd.index("batch") + 1]
            assert set(ids) == {r["episode"] for r in sources.values() if r["root"] == root}
            prepare.write(output, [{"episode": e, "source": root} for e in ids])
        else:
            predictions = json.loads(Path(cmd[2]).read_text())
            assert {r["episode"] for r in predictions} == set(map(int, sources))
            for row in predictions:
                assert row["source"] == sources[str(row["episode"])]["root"]
            prepare.write(
                output,
                {
                    "label_scope": "segment_count",
                    "n_segments": 2,
                    "task_id": "shared_task",
                    "arm": "align_video_cal",
                    "fit_on_episodes": ids,
                    "fit_episode_count": len(ids),
                },
            )
        return True, ""

    monkeypatch.setattr(fitter, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fit",
            "--root",
            str(study / "fit_root"),
            "--tasks-splits",
            str(study / "task_splits.json"),
            "--ground-truth",
            str(study / "ground_truth.json"),
            "--seed-sources",
            str(study / "seed_sources.json"),
            "--identity-namespace",
            plan["identity_namespace"],
            "--arms-config",
            str(study / "arms.yaml"),
            "--out-dir",
            str(tmp_path / "fits"),
            "--arm",
            "align_video_cal",
            "--model",
            "model",
            "--ports",
            "8001",
            "--eval-batch",
            "batch",
            "--fit-bin",
            "fitter",
        ],
    )
    assert fitter.main() == 0
    manifest = json.loads((tmp_path / "fits/align_video_cal/calibration_routing.json").read_text())
    assert manifest["n_cohorts_fitted"] == 1
    assert len(calls) == 3  # two source batches, one pooled ten-seed fit
    assert (
        manifest["seed_sources_sha256"]
        == hashlib.sha256((study / "seed_sources.json").read_bytes()).hexdigest()
    )


def test_real_lerobot_split_preserves_selected_trajectory_identity(tmp_path):
    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "source"
    features = {
        camera: {"dtype": "image", "shape": (8, 8, 3), "names": ["height", "width", "channel"]}
        for camera in ("wrist", "external")
    }
    features["action"] = {"dtype": "float32", "shape": (1,), "names": ["identity"]}
    dataset = LeRobotDataset.create(
        repo_id="local/source", root=root, fps=10, features=features, use_videos=False
    )
    for episode in range(12):
        for _ in range(2):
            dataset.add_frame(
                {
                    "wrist": np.zeros((8, 8, 3), dtype=np.uint8),
                    "external": np.zeros((8, 8, 3), dtype=np.uint8),
                    "action": np.array([episode], dtype=np.float32),
                    "task": "shared_task",
                }
            )
        dataset.save_episode()
    dataset.finalize()
    spans = [{"text": "reach", "start": 0, "end": 0.1}, {"text": "place", "start": 0.1, "end": 0.2}]
    prepare.write(
        root / "meta/lerobot_annotations.json",
        {"episodes": {str(e): {"subtasks": spans} for e in range(12)}},
    )
    prepare.write(tmp_path / "gt.json", {"episodes": {str(e): spans for e in range(12)}})
    path = tmp_path / "manifest.json"
    prepare.write(
        path,
        {
            "version": 1,
            "name": "smoke",
            "cameras": {"primary": "wrist", "secondary": "external"},
            "sources": [
                {
                    "id": "source",
                    "root": str(root),
                    "ground_truth": str(tmp_path / "gt.json"),
                    "task": "shared_task",
                    "seed_episodes": list(range(10)),
                }
            ],
        },
    )
    out = tmp_path / "prepared"
    plan = prepare.plan_study(path)
    prepare.materialize(plan, out)
    component = next(iter(plan["components"]))
    result = LeRobotDataset(repo_id=component, root=out / "components" / component)
    assert result.meta.total_episodes == 2
    assert result.hf_dataset["episode_index"] == [0, 0, 1, 1]
    assert [float(v) for v in result.hf_dataset["action"]] == [10.0, 10.0, 11.0, 11.0]


def test_shared_score_and_aggregate_pipeline_includes_failures(tmp_path, monkeypatch):
    import subprocess

    study, plan, _ = prepared(tmp_path, monkeypatch)
    out = tmp_path / "run"
    _, commands = evaluate.commands(study, out, "model", sys.executable, "8001", 2400, 100)
    arms = yaml.safe_load((study / "arms.yaml").read_text())["arms"]
    names = [a["name"] for a in arms if a["tool"] != "reference"]
    jobs = []
    for component in plan["components"]:
        truth = json.loads((study / "gt" / f"{component}.json").read_text())["episodes"]
        for arm in names:
            jobs.append({"dataset": component, "arm": arm, "episodes": list(map(int, truth))})
            prepare.write(
                out / "predictions" / f"{component}__{arm}.json",
                {
                    "arm": arm,
                    "ok": arm != "align_default",
                    "episodes": truth,
                    "calibration_status": {
                        e: {"applied": False, "reason": "not_configured", "method": None}
                        for e in truth
                    },
                },
            )
    prepare.write(
        out / "predictions/jobs_manifest.json", {"version": 1, "arms": names, "jobs": jobs}
    )
    for cmd in commands["score"]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    evaluate.combine_scores(out)
    for cmd in commands["aggregate"]:
        result = subprocess.run(cmd, capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr
    rows = [json.loads(line) for line in (out / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 14 * 9
    failed = [r for r in rows if r["arm"] == "align_default"]
    assert len(failed) == 14 and all(not r["ok"] and r["scored"] for r in failed)


def test_orchestrator_scores_complete_explicit_failures(tmp_path, monkeypatch):
    from types import SimpleNamespace

    predictions = tmp_path / "predictions"
    prepare.write(
        predictions / "jobs_manifest.json",
        {"jobs": [{"dataset": "one", "arm": "arm", "episodes": [0]}]},
    )
    prepare.write(predictions / "run_summary.json", {"jobs": 1, "ok": 0, "failed": 1})
    prepare.write(predictions / "one__arm.json", {"ok": False, "episodes": {}})
    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    evaluate.run_command(
        "run",
        [sys.executable, str(SCRIPTS / "run_alignment_arms.py")],
        tmp_path,
        {},
    )


def test_orchestrator_rejects_incomplete_inference(tmp_path, monkeypatch):
    from types import SimpleNamespace

    predictions = tmp_path / "predictions"
    prepare.write(
        predictions / "jobs_manifest.json",
        {"jobs": [{"dataset": "one", "arm": "arm", "episodes": [0]}]},
    )
    prepare.write(predictions / "run_summary.json", {"jobs": 1, "ok": 0, "failed": 1})
    monkeypatch.setattr(
        evaluate.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )
    with pytest.raises(evaluate.subprocess.CalledProcessError):
        evaluate.run_command(
            "run",
            [sys.executable, str(SCRIPTS / "run_alignment_arms.py")],
            tmp_path,
            {},
        )


def test_source_manifest_environment_is_explicit(tmp_path, monkeypatch):
    monkeypatch.delenv('ALIGN_RELEASE_TEST_ROOT', raising=False)
    with pytest.raises(ValueError, match='Unset source-path environment variable'):
        prepare.resolve('${ALIGN_RELEASE_TEST_ROOT}/dataset', tmp_path)
    monkeypatch.setenv('ALIGN_RELEASE_TEST_ROOT', str(tmp_path))
    assert prepare.resolve('${ALIGN_RELEASE_TEST_ROOT}/dataset', tmp_path) == tmp_path / 'dataset'
