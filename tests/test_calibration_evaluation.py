"""Runtime instrumentation must survive evaluation and cohort index translation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "evaluation" / "scripts"
sys.path.insert(0, str(SCRIPTS))

import calibration_status  # noqa: E402
import merge_alignment_predictions as merge  # noqa: E402
import score_alignment  # noqa: E402
import select_corpus_b  # noqa: E402


def event(episode=0, applied=True, reason="applied", method="offsets"):
    payload = {"episode": episode, "applied": applied, "reason": reason, "method": method}
    return f'INFO ALIGN_CALIBRATION {json.dumps(payload)}\n'


def test_runtime_decision_not_changed_predictions_establishes_application():
    predictions = {"0": [{"text": "reach", "start": 0, "end": 2}]}
    skipped = event(applied=False, reason="label_mismatch", method=None)
    statuses = calibration_status.run_calibration_status(
        skipped, [0], predictions, configured=True, job_ok=True,
    )
    assert statuses["0"]["applied"] is False
    assert statuses["0"]["reason"] == "label_mismatch"
    with pytest.raises(ValueError, match="no runtime decision"):
        calibration_status.run_calibration_status(
            "changed predictions alone", [0], predictions, configured=True, job_ok=True,
        )


def test_failed_outputs_do_not_count_as_delivered_calibration():
    status = calibration_status.run_calibration_status(
        event(), [0], {}, configured=True, job_ok=False,
    )["0"]
    assert not status["applied"]
    assert status["reason"] == "job_failed"
    assert status["runtime_decision"]["applied"]


def test_duplicate_or_unexpected_runtime_decisions_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        calibration_status.parse_calibration_status(event() + event())
    with pytest.raises(ValueError, match="unrequested"):
        calibration_status.run_calibration_status(event(9), [0], {}, configured=True, job_ok=True)


def test_merger_never_uses_eligibility_as_application():
    record = {"flags": {}, "calibration_applied": {"0": True}}
    assert not merge.prediction_calibration_status(record, "0")["applied"]
    record["flags"]["plan.subtask_align_calibration_path"] = "fit.json"
    with pytest.raises(ValueError, match="runtime decision is missing"):
        merge.prediction_calibration_status(record, "0")


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def test_merge_and_score_preserve_actual_status_and_failed_task(tmp_path, monkeypatch):
    name = "study__task__n2"
    spans = [{"text": "reach", "start": 0, "end": 1}, {"text": "place", "start": 1, "end": 2}]
    write_json(tmp_path / "groups.json", {name: {"task": "task", "cohort": "task__n2", "calibrated": True}})
    write_json(tmp_path / "cohorts" / f"{name}.index_map.json", {
        "episodes": [{"new_index": 0, "original_index": 100}],
    })
    write_json(tmp_path / "gt" / f"{name}.json", {"episodes": {"0": spans}})
    for arm, success in [("align_video", True), ("align_video_cal", True), ("align_video_stack_cal", False)]:
        flags = {} if arm == "align_video" else {"plan.subtask_align_calibration_path": "frozen-fit.json"}
        status = {"applied": False, "reason": "label_mismatch", "method": None}
        write_json(tmp_path / "preds" / f"{name}__{arm}.json", {
            "arm": arm, "ok": success, "flags": flags, "episodes": {"0": spans},
            "calibration_status": {"0": status},
        })
    write_json(tmp_path / "preds/jobs_manifest.json", {
        "version": 1, "arms": ["align_video", "align_video_cal", "align_video_stack_cal"],
        "jobs": [{"dataset": name, "arm": arm, "episodes": [0]}
                 for arm in ["align_video", "align_video_cal", "align_video_stack_cal"]],
    })
    monkeypatch.setattr(sys, "argv", ["merge", "--predictions-dir", str(tmp_path / "preds"),
        "--group-meta", str(tmp_path / "groups.json"), "--cohorts-dir", str(tmp_path / "cohorts"),
        "--gt-dir", str(tmp_path / "gt"), "--out-predictions", str(tmp_path / "merged"),
        "--out-gt", str(tmp_path / "merged_gt"), "--out-splits", str(tmp_path / "splits"),
        "--eligibility-out", str(tmp_path / "eligibility.json")])
    assert merge.main() == 0
    calibrated = json.loads((tmp_path / "merged/study__task__align_video_cal.json").read_text())
    assert calibrated["calibration_eligible"] == {"100": True}
    assert calibrated["calibration_applied"] == {"100": False}
    assert calibrated["calibration_status"]["100"]["reason"] == "label_mismatch"
    failed = json.loads((tmp_path / "merged/study__task__align_video_stack_cal.json").read_text())
    assert failed["episodes"] == {}
    assert failed["calibration_applied"] == {"100": False}
    assert failed["cohorts"][0]["calibration_configured"]
    assert not failed["cohorts"][0]["ok"]
    mask = json.loads((tmp_path / "eligibility.json").read_text())
    assert mask["episodes"] == {"study__task": {"100": True}}
    monkeypatch.setattr(sys, "argv", ["score", "--predictions-dir", str(tmp_path / "merged"),
        "--gt-dir", str(tmp_path / "merged_gt"), "--splits-dir", str(tmp_path / "splits"),
        "--out", str(tmp_path / "scores.jsonl"), "--eligibility", str(tmp_path / "eligibility.json")])
    assert score_alignment.main() == 0
    rows = [json.loads(line) for line in (tmp_path / "scores.jsonl").read_text().splitlines()]
    assert len(rows) == 3
    assert all(r["calibration_applied"] == 0 and r["calibration_eligible"] for r in rows)
    failure = next(r for r in rows if r["arm"] == "align_video_stack_cal")
    assert failure["arm_has_calibration"]
    assert failure["calibration_applied_source"] == "delivery"
    assert all(r["calibration_applied_source"] == "runtime" for r in rows if r["ok"])
    assert not failure["ok"] and failure["scored"] and failure["b_hit@3"] == 0


def test_selection_budget_is_checked_before_scanning(tmp_path, monkeypatch):
    def forbidden(*args):
        pytest.fail("invalid seed budget must not scan data")
    monkeypatch.setattr(select_corpus_b, "scan", forbidden)
    monkeypatch.setattr(sys, "argv", ["select", "--data-root", str(tmp_path),
        "--out-dir", str(tmp_path / "out"), "--n-seed", "11"])
    with pytest.raises(SystemExit):
        select_corpus_b.main()


@pytest.mark.parametrize("cap, expected", [(None, 45), (40, 40)])
def test_selection_uses_ten_seeds_and_all_remaining_by_default(tmp_path, monkeypatch, cap, expected):
    trajectories = {
        f"trajectory_{e}": {"task": "task", "views": [{
            "episode_index": e, "camera": "wrist", "n_clips": 2,
            "n_spans": 2, "duration": 10, "reject_reason": None,
        }]}
        for e in range(55)
    }
    monkeypatch.setattr(select_corpus_b, "scan", lambda *args: trajectories)
    args = ["select", "--data-root", str(tmp_path), "--out-dir", str(tmp_path / "out"),
            "--camera-substudy-tasks", "0"]
    if cap is not None:
        args.extend(["--cap-eval", str(cap)])
    monkeypatch.setattr(sys, "argv", args)
    assert select_corpus_b.main() == 0
    payload = json.loads((tmp_path / "out/splits.json").read_text())
    split = payload["tasks"]["task"]
    assert split["seed"] == list(range(10))
    assert split["eval"] == list(range(10, 10 + expected))


def test_full_dataset_run_accepts_empty_episode_telemetry():
    statuses = calibration_status.run_calibration_status(
        event(0) + event(1, False, "empty_prediction", None), None,
        {"0": [{"text": "reach", "start": 0, "end": 2}]},
        configured=True, job_ok=True,
    )
    assert statuses["0"]["applied"]
    assert not statuses["1"]["applied"]
    assert statuses["1"]["reason"] == "no_output"


def test_scorer_discards_predictions_of_failed_job(tmp_path, monkeypatch):
    spans = [{"text": "reach", "start": 0, "end": 1}, {"text": "place", "start": 1, "end": 2}]
    write_json(tmp_path / "gt/task.json", {"dataset": "task", "episodes": {"0": spans}})
    write_json(tmp_path / "splits/task.json", {"dataset": "task", "seed": [], "eval": [0]})
    write_json(tmp_path / "preds/task.json", {
        "dataset": "task", "arm": "align_video_cal", "ok": False,
        "error": "calibration file changed during inference", "episodes": {"0": spans},
        "calibration_status_version": 1,
        "calibration_status": {"0": {"applied": True, "reason": "applied", "method": "offsets"}},
    })
    monkeypatch.setattr(sys, "argv", ["score", "--predictions-dir", str(tmp_path / "preds"),
        "--gt-dir", str(tmp_path / "gt"), "--splits-dir", str(tmp_path / "splits"),
        "--out", str(tmp_path / "scores.jsonl")])
    assert score_alignment.main() == 0
    row = json.loads((tmp_path / "scores.jsonl").read_text())
    assert not row["ok"] and row["scored"] and row["b_hit@3"] == 0
    assert not row["calibration_applied"]
    assert row["calibration_reason"] == "job_failed"
