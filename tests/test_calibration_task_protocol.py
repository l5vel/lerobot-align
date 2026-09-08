"""Task-budget, fit provenance, and arm routing regressions without VLM calls."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "evaluation/scripts"
sys.path.insert(0, str(SCRIPTS))
import fit_task_calibration as driver  # noqa: E402
import run_alignment_arms as runner  # noqa: E402

ARMS = REPO / "evaluation/configs/corpus_b_arms.yaml"


def spans(count):
    return [{"text": f"label {i}", "start": i, "end": i + 1} for i in range(count)]


def inputs(tmp_path, tasks=None, counts=None):
    tasks = tasks or {"task": {"seed": [0, 1, 2], "eval": [3, 4]}}
    counts = counts or {e: 2 for groups in tasks.values() for ids in groups.values() for e in ids}
    root = tmp_path / "root"
    root.mkdir()
    truth = tmp_path / "gt.json"
    truth.write_text(json.dumps({"episodes": {str(e): spans(n) for e, n in counts.items()}}))
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"tasks": tasks}))
    root.with_name(root.name + ".index_map.json").write_text(json.dumps({
        "episodes": [{"new_index": e, "original_index": e + 100} for e in counts]}))
    argv = ["fit", "--root", str(root), "--tasks-splits", str(splits),
            "--ground-truth", str(truth), "--out-dir", str(tmp_path / "fits"),
            "--arm", "align_video_cal", "--arms-config", str(ARMS),
            "--eval-batch", "batch", "--fit-bin", "fitter", "--model", "model", "--ports", "8000"]
    return argv, tasks, counts


def install_fake_fit(monkeypatch, counts, *, omit_output=False, reject=False, timeouts=None):
    calls = []
    timeouts = timeouts if timeouts is not None else []

    def fake_run(cmd, log, timeout=None):
        calls.append(cmd)
        timeouts.append(timeout)
        output = Path(cmd[cmd.index("--out") + 1])
        if "batch" in cmd:
            output.write_text("[]")
        elif reject:
            return False, "recommendation gate rejected"
        elif not omit_output:
            start = cmd.index("--episodes") + 1
            end = next(i for i in range(start, len(cmd)) if cmd[i].startswith("--"))
            ids = list(map(int, cmd[start:end]))
            output.write_text(json.dumps({
                "label_scope": "segment_count", "n_segments": counts[ids[0]],
                "task_id": cmd[cmd.index("--task-id") + 1], "arm": cmd[cmd.index("--arm") + 1],
                "fit_on_episodes": ids, "fit_episode_count": len(ids),
            }))
        return True, ""

    monkeypatch.setattr(driver, "run", fake_run)
    return calls


@pytest.mark.parametrize("tasks", [
    {"task": {"seed": list(range(11)), "eval": [11]}},
    {"task": {"seed": [0, 0, 1], "eval": [2]}},
    {"task": {"seed": [0, 1, 2], "eval": [2, 3]}},
    {"a": {"seed": [0, 1, 2], "eval": [3]}, "b": {"seed": [4, 5, 6], "eval": [3]}},
])
def test_invalid_task_split_fails_before_side_effects(tmp_path, monkeypatch, tasks):
    argv, _, counts = inputs(tmp_path, tasks)
    calls = install_fake_fit(monkeypatch, counts)
    monkeypatch.setattr(sys, "argv", [*argv, "--force"])
    with pytest.raises(SystemExit):
        driver.main()
    assert calls == []
    assert not (tmp_path / "fits").exists()


def test_count_cohorts_partition_same_ten_seeds_and_keep_all_eval(tmp_path, monkeypatch):
    tasks = {"task": {"seed": list(range(10)), "eval": list(range(10, 20))}}
    counts = {e: 2 if e < 6 or e >= 10 else 3 for e in range(20)}
    argv, _, _ = inputs(tmp_path, tasks, counts)
    calls = install_fake_fit(monkeypatch, counts)
    monkeypatch.setattr(sys, "argv", argv)
    assert driver.main() == 0
    path = tmp_path / "fits/align_video_cal/calibration_routing.json"
    manifest = json.loads(path.read_text())
    assert manifest["task_splits"] == tasks
    assert len(manifest["routing"]) == 10
    assert {k: len(v["seed_episodes"]) for k, v in manifest["cohorts"].items()} == {
        "task__n2": 6, "task__n3": 4}
    assert len({e for c in manifest["cohorts"].values() for e in c["seed_episodes"]}) == 10
    assert all("--force" not in cmd for cmd in calls)
    for cmd in calls:
        if cmd[0] == "fitter":
            assert cmd[cmd.index("--label-scope") + 1] == "segment_count"
            assert cmd[cmd.index("--max-fit-episodes") + 1] == "10"
        else:
            assert cmd[cmd.index("--temperature") + 1] == "0.2"
            assert cmd[cmd.index("--fps") + 1] == "2.0"
            assert cmd[cmd.index("--frame-width") + 1] == "224"
            assert cmd[cmd.index("--video-fallback") + 1] == "error"
    runner.load_manifests([path], ARMS, "model")
    before = len(calls)
    with pytest.raises(SystemExit, match="frozen"):
        driver.main()
    assert len(calls) == before


@pytest.mark.parametrize("reject", [False, True])
def test_failed_or_missing_fit_does_not_reuse_stale_file(tmp_path, monkeypatch, reject):
    argv, _, counts = inputs(tmp_path)
    old = tmp_path / "fits/align_video_cal/task__n2.json"
    old.parent.mkdir(parents=True)
    old.write_text('{"old": true}')
    install_fake_fit(monkeypatch, counts, omit_output=True, reject=reject)
    monkeypatch.setattr(sys, "argv", argv)
    assert driver.main() == 0
    manifest = json.loads((old.parent / "calibration_routing.json").read_text())
    assert manifest["fit_files"] == {}
    assert manifest["calibrated"] == 0
    assert manifest["uncalibrated_fallback"] == 2


def test_regime_derives_distinct_cameras_and_rejects_arbitrary_flags(tmp_path, monkeypatch):
    single, _ = driver.fit_regime(ARMS, "align_video_cal")
    stack, _ = driver.fit_regime(ARMS, "align_video_stack_cal")
    assert single["cameras"] == ["observation.images.wrist"]
    assert stack["cameras"] == ["observation.images.wrist", "observation.images.primary"]
    argv, _, _ = inputs(tmp_path)
    monkeypatch.setattr(sys, "argv", [*argv, "--fit-flags=--temperature 1.0"])
    with pytest.raises(SystemExit):
        driver.main()


def test_small_count_group_uses_fallback_without_extra_seed_calls(tmp_path, monkeypatch):
    tasks = {"task": {"seed": list(range(10)), "eval": [10, 11]}}
    counts = {e: (3 if e in (8, 9, 11) else 2) for e in range(12)}
    argv, _, _ = inputs(tmp_path, tasks, counts)
    calls = install_fake_fit(monkeypatch, counts)
    monkeypatch.setattr(sys, "argv", [*argv, "--force"])
    assert driver.main() == 0
    manifest = json.loads((tmp_path / "fits/align_video_cal/calibration_routing.json").read_text())
    assert len(calls) == 2
    assert manifest["calibrated"] == 1
    assert manifest["uncalibrated_fallback"] == 1
    assert manifest["routing"]["11"]["cohort"] is None
    assert "--force" in next(cmd for cmd in calls if cmd[0] == "fitter")


def fitted_manifest(tmp_path, monkeypatch):
    argv, _, counts = inputs(tmp_path)
    install_fake_fit(monkeypatch, counts)
    monkeypatch.setattr(sys, "argv", argv)
    assert driver.main() == 0
    return tmp_path / "fits/align_video_cal/calibration_routing.json"


@pytest.mark.parametrize("mutation", ["digest", "arm", "model", "fit_ids", "seeds"])
def test_manifest_rejects_tampered_or_cross_arm_fits(tmp_path, monkeypatch, mutation):
    path = fitted_manifest(tmp_path, monkeypatch)
    manifest = json.loads(path.read_text())
    entry = manifest["fit_files"]["task__n2"]
    if mutation == "digest":
        Path(entry["path"]).write_text("{}")
    elif mutation == "arm":
        entry["arm"] = "align_video_stack_cal"
    elif mutation == "model":
        manifest["model_id"] = "another-model"
    elif mutation == "fit_ids":
        entry["fit_on_episodes"] = [0, 1, 3]
    else:
        manifest["task_splits"]["task"]["seed"] = list(range(11))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        runner.load_manifests([path], ARMS, "model")


def run_inputs(tmp_path, manifest_path, arm):
    source = tmp_path / "components"
    source.mkdir()
    dataset = "component"
    (source / f"{dataset}.index_map.json").write_text(json.dumps({
        "episodes": [{"new_index": 0, "original_index": 103}, {"new_index": 1, "original_index": 104}]}))
    split_dir, gt_dir, labels_dir = [tmp_path / d for d in ("eval_splits", "eval_gt", "labels")]
    for directory in (split_dir, gt_dir, labels_dir):
        directory.mkdir()
    (split_dir / f"{dataset}.json").write_text(json.dumps({"seed": [], "eval": [0, 1]}))
    (gt_dir / f"{dataset}.json").write_text(json.dumps({"episodes": {"0": spans(2), "1": spans(2)}}))
    (labels_dir / f"{dataset}__oracle.json").write_text(json.dumps({"0": ["label 0", "label 1"], "1": ["label 0", "label 1"]}))
    meta = tmp_path / "groups.json"
    meta.write_text(json.dumps({dataset: {"task": "task", "cohort": "task__n2", "calibrated": True}}))
    return [
        "run", "--group-meta", str(meta), "--source-dir", str(source), "--splits-dir", str(split_dir),
        "--gt-dir", str(gt_dir), "--labels-dir", str(labels_dir), "--cohort-files", str(manifest_path),
        "--arms-config", str(ARMS), "--arms", arm, "--model-id", "model",
        "--out-dir", str(tmp_path / "outputs"), "--work-dir", str(tmp_path / "work"),
        "--job-splits-dir", str(tmp_path / "jobs")]


@pytest.mark.parametrize("missing_manifest", [False, True])
def test_stack_requires_own_manifest_and_falls_back_only_within_it(tmp_path, monkeypatch, capsys, missing_manifest):
    manifest_path = fitted_manifest(tmp_path, monkeypatch)
    if not missing_manifest:
        manifest = json.loads(manifest_path.read_text())
        manifest["arm"] = "align_video_stack_cal"
        manifest["fit_regime"], _ = driver.fit_regime(ARMS, manifest["arm"])
        manifest["fit_files"] = {}
        manifest_path = tmp_path / "stack_manifest.json"
        manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(sys, "argv", [*run_inputs(tmp_path, manifest_path, "align_video_stack_cal"), "--dry-run"])
    capsys.readouterr()
    if missing_manifest:
        with pytest.raises(SystemExit, match="missing arm-specific calibration manifest"):
            runner.main()
        assert not (tmp_path / "outputs").exists()
        return
    assert runner.main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["calibrated_jobs"] == 0
    assert report["fallback_jobs"] == 1
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize("failure", ["exception", "early_return"])
@pytest.mark.parametrize("stale_success", [False, True])
def test_dispatch_failure_always_writes_explicit_empty_result(tmp_path, monkeypatch, failure, stale_success):
    manifest_path = fitted_manifest(tmp_path, monkeypatch)
    monkeypatch.setattr(sys, "argv", run_inputs(tmp_path, manifest_path, "align_video_cal"))
    destination = tmp_path / "outputs/component__align_video_cal.json"
    if stale_success:
        destination.parent.mkdir()
        destination.write_text(json.dumps({"ok": True, "episodes": {"0": spans(2)}}))

    def fail(*args, **kwargs):
        if failure == "exception":
            raise FileNotFoundError("missing executable")
        return SimpleNamespace(returncode=1, stdout="", stderr="failed before output")

    monkeypatch.setattr(runner.subprocess, "run", fail)
    assert runner.main() == 1
    result = json.loads(destination.read_text())
    assert result["ok"] is False
    assert result["episodes"] == {}
    assert result["calibration_status"] == {}
    assert result["calibration_applied"] == {}
    assert result["requested_arm"] == "align_video_cal"
    assert result["dataset"] == "component"
    assert result["n_episodes_requested"] == 2
    assert result["calibration_fit_available"] is True
    summary = json.loads((destination.parent / "run_summary.json").read_text())
    assert summary["failed"] == 1
    assert len(summary["failures"]) == 1


@pytest.mark.parametrize("change", ["replace", "delete"])
def test_fit_changed_during_subprocess_invalidates_success(tmp_path, monkeypatch, change):
    manifest_path = fitted_manifest(tmp_path, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    calibration_path = Path(manifest["fit_files"]["task__n2"]["path"])
    monkeypatch.setattr(sys, "argv", run_inputs(tmp_path, manifest_path, "align_video_cal"))

    def pretend_success(cmd, **kwargs):
        output = Path(cmd[cmd.index("--out") + 1])
        output.write_text(json.dumps({"ok": True, "episodes": {"0": spans(2), "1": spans(2)}}))
        if change == "replace":
            calibration_path.write_text("{}")
        else:
            calibration_path.unlink()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", pretend_success)
    assert runner.main() == 1
    output = json.loads((tmp_path / "outputs/component__align_video_cal.json").read_text())
    assert output["ok"] is False
    assert output["episodes"] == {}
    assert "calibration" in output["error"]


def test_runner_refuses_an_omitted_evaluation_component_before_dispatch(tmp_path, monkeypatch):
    manifest_path = fitted_manifest(tmp_path, monkeypatch)
    argv = run_inputs(tmp_path, manifest_path, "align_video_cal")
    # Split the two episodes into two physical components, then accidentally
    # leave the second out of group metadata. Every listed ID remains valid.
    (tmp_path / "eval_splits/component.json").write_text(json.dumps({"seed": [], "eval": [0]}))
    (tmp_path / "eval_gt/component.json").write_text(json.dumps({"episodes": {"0": spans(2)}}))
    (tmp_path / "components/omitted.index_map.json").write_text(json.dumps({
        "episodes": [{"new_index": 0, "original_index": 104}]}))
    (tmp_path / "eval_splits/omitted.json").write_text(json.dumps({"seed": [], "eval": [0]}))
    (tmp_path / "eval_gt/omitted.json").write_text(json.dumps({"episodes": {"0": spans(2)}}))
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(SystemExit, match="ALL allocated evaluation trajectories"):
        runner.main()
    assert calls == []
    assert not (tmp_path / "outputs").exists()


def test_same_original_trajectory_cannot_appear_in_two_components(tmp_path):
    for directory in ("source", "splits", "gt"):
        (tmp_path / directory).mkdir()
    for dataset in ("a", "b"):
        (tmp_path / "source" / f"{dataset}.index_map.json").write_text(json.dumps({
            "episodes": [{"new_index": 0, "original_index": 100}]}))
        (tmp_path / "splits" / f"{dataset}.json").write_text(json.dumps({"seed": [], "eval": [0]}))
        (tmp_path / "gt" / f"{dataset}.json").write_text(json.dumps({"episodes": {"0": spans(2)}}))
    seen = set()
    runner.component_episodes("a", tmp_path / "source", tmp_path / "splits", tmp_path / "gt", seen)
    with pytest.raises(ValueError, match="multiple components"):
        runner.component_episodes("b", tmp_path / "source", tmp_path / "splits", tmp_path / "gt", seen)


def test_exports_all_count_groups_in_original_ids_independent_of_arm_gate(tmp_path, monkeypatch):
    tasks = {"task": {"seed": list(range(10)), "eval": [10, 11, 12]}}
    counts = {e: (3 if e in (8, 9, 11) else 4 if e == 12 else 2) for e in range(13)}
    argv, _, _ = inputs(tmp_path, tasks, counts)
    install_fake_fit(monkeypatch, counts)
    monkeypatch.setattr(sys, "argv", argv)
    assert driver.main() == 0
    single = tmp_path / "fits/align_video_cal"
    groups = json.loads((single / "evaluation_groups.json").read_text())
    assert groups == {
        "study__task__n2": {"seed": [], "eval": [110]},
        "study__task__n3": {"seed": [], "eval": [111]},
        "study__task__n4": {"seed": [], "eval": [112]},
    }
    metadata = json.loads((single / "cohort_group_meta.json").read_text())
    assert metadata["study__task__n2"]["seed_episodes"] == list(range(100, 108))
    assert metadata["study__task__n2"]["calibrated"] is True
    assert metadata["study__task__n3"]["n_seed"] == 2
    assert metadata["study__task__n3"]["calibrated"] is False
    assert metadata["study__task__n4"]["n_seed"] == 0
    assert all(m["calibration_eligible_source"] == "seed_count_support" for m in metadata.values())
    install_fake_fit(monkeypatch, counts, reject=True)
    argv[argv.index("--arm") + 1] = "align_video_stack_cal"
    monkeypatch.setattr(sys, "argv", argv)
    assert driver.main() == 0
    stack = tmp_path / "fits/align_video_stack_cal"
    for name in ("evaluation_groups.json", "cohort_group_meta.json"):
        assert (stack / name).read_bytes() == (single / name).read_bytes()
    assert json.loads((stack / "calibration_routing.json").read_text())["fit_files"] == {}


def test_group_export_requires_original_identity_map_before_calls(tmp_path, monkeypatch):
    argv, _, counts = inputs(tmp_path)
    (tmp_path / "root.index_map.json").unlink()
    calls = install_fake_fit(monkeypatch, counts)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="original-trajectory index map"):
        driver.main()
    assert not calls
    assert not (tmp_path / "fits").exists()


def test_the_step_bound_reaches_the_step_and_is_recorded(tmp_path, monkeypatch):
    timeouts = []
    argv, _, counts = inputs(tmp_path)
    install_fake_fit(monkeypatch, counts, timeouts=timeouts)
    monkeypatch.setattr(sys, "argv", [*argv, "--step-timeout", "900"])
    assert driver.main() == 0
    assert timeouts and all(value == 900.0 for value in timeouts)
    manifest = json.loads((tmp_path / "fits/align_video_cal/calibration_routing.json").read_text())
    assert manifest["step_timeout_seconds"] == 900.0


def test_run_bounds_a_command_that_never_returns(tmp_path):
    """The bound is on the real step, not only on a stand-in."""
    ok, tail = driver.run(["sleep", "60"], tmp_path / "step.log", timeout=0.5)
    assert ok is False
    assert "exceeded" in tail
    assert "exceeded" in (tmp_path / "step.log").read_text()
    assert driver.run(["true"], tmp_path / "ok.log", timeout=60)[0] is True


def test_a_wedged_replica_fails_its_cohort_into_the_fallback(tmp_path, monkeypatch):
    """A replica can accept a request, report it running, and emit no tokens.

    The FIFO lease means that cohort never returns its replica and never
    returns itself, so an unbounded step left the driver waiting on one thread
    with every other cohort already finished -- which is what happened on
    replica 8000 mid-run. Bounded, it must become one recorded failure whose
    trajectories route to the uncalibrated fallback, and the run must complete.
    """
    argv, _, _ = inputs(tmp_path, {"task": {"seed": [0, 1, 2], "eval": [3, 4]}})
    monkeypatch.setattr(driver, "run", lambda cmd, log, timeout=None: (False, "step exceeded 1 seconds"))
    monkeypatch.setattr(sys, "argv", argv)
    assert driver.main() == 0
    manifest = json.loads((tmp_path / "fits/align_video_cal/calibration_routing.json").read_text())
    assert manifest["n_cohorts_fitted"] == 0
    assert manifest["n_cohorts_failed"] == 1
    assert manifest["uncalibrated_fallback"] == 2
    assert all(row["cohort"] is None and row["calibrated"] is False
               for row in manifest["routing"].values())
    assert "exceeded" in manifest["results"]["task__n2"]["error"]
