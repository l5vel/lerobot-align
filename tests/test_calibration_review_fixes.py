"""Regression coverage for population completeness and calibration parity."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tests.test_calibration_evaluation import merge, score_alignment, write_json
from tests.test_calibration_runtime import _module, _record
from lerobot_align.diagnostics.fit_align_calibration import apply
from lerobot_align.modules.plan_subtasks_memory import _AlignCalibration

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation/scripts"))
import aggregate  # noqa: E402


def test_eligible_population_refuses_missing_floor_mask():
    rows = [{"arm": "align_video", "calibration_eligible": True},
            {"arm": "floor_uniform", "calibration_eligible": None}]
    with pytest.raises(SystemExit, match="missing for arms.*floor_uniform"):
        aggregate.filter_population(rows, ["calibration_eligible"])
    rows[1]["calibration_eligible"] = True
    assert aggregate.filter_population(rows, ["calibration_eligible"]) == rows


def test_offsets_only_matches_production_with_unstitched_model_end():
    record = _record(tuple(i / 10 for i in range(101)))
    module = _module(_AlignCalibration(offsets=(-0.3,), mode="fraction_of_duration"))
    raw = {0: (0, 5), 1: (5, 6)}

    def clean(parsed):
        return module._stitch_full_coverage(module._clean_spans(
            module._align_spans_in_order(record, ["reach", "place"], parsed), record), record)

    runtime = clean(module._apply_align_calibration(record, ["reach", "place"], raw))
    diagnostic = apply(clean(raw), {"offsets": [-0.3]})
    assert runtime == diagnostic
    assert runtime[1]["start"] == 8


@pytest.mark.parametrize("population,expected,count", [("all_scored", 0.5, 2), ("successful", 1.0, 1)])
def test_aggregation_keeps_scored_failures_and_explicit_conditional_view(
    tmp_path, monkeypatch, population, expected, count,
):
    rows = [{"dataset": "task", "episode": e, "arm": a, "ok": e == 0,
             "scored": e != 2, "b_hit@3": 1.0 if e == 0 else 0.0}
            for a in ("a", "b") for e in range(3)]
    (tmp_path / "scores.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    write_json(tmp_path / "arms.json", {"arms": [
        {"name": a, "tool": "align", "supervision": "labels_only", "equalised": True, "flags": {}}
        for a in ("a", "b")]})
    monkeypatch.setattr(sys, "argv", ["aggregate", "--scores", str(tmp_path / "scores.jsonl"),
        "--out", str(tmp_path / "out.json"), "--arms-config", str(tmp_path / "arms.json"),
        "--matcher", "any", "--metrics", "b_hit@3", "--n-boot", "20", "--baseline", "a",
        "--contrast-mode", "global", "--output-population", population])
    assert aggregate.main() == 0
    result = json.loads((tmp_path / "out.json").read_text())
    assert result["per_arm"]["a"]["b_hit@3"]["point"] == expected
    assert result["per_arm"]["a"]["b_hit@3"]["n_units"] == count
    assert result["failure_rates"]["a"]["failed"] == 2
    assert result["contrasts"]["b"]["b_hit@3"]["n_units"] == count


def test_missing_entire_task_arm_is_rejected_before_outputs(tmp_path, monkeypatch):
    names = [f"corpus_b__{t}__n2" for t in ("a", "b")]
    write_json(tmp_path / "groups.json", {n: {"task": n, "calibrated": True} for n in names})
    write_json(tmp_path / "preds" / f"{names[0]}__align_video.json", {"ok": True})
    monkeypatch.setattr(sys, "argv", ["merge", "--predictions-dir", str(tmp_path / "preds"),
        "--arms", "align_video", "--group-meta", str(tmp_path / "groups.json"),
        "--cohorts-dir", str(tmp_path / "components"), "--gt-dir", str(tmp_path / "gt"),
        "--out-predictions", str(tmp_path / "merged"), "--out-gt", str(tmp_path / "merged_gt"),
        "--out-splits", str(tmp_path / "splits")])
    with pytest.raises(SystemExit, match="incomplete or stale prediction matrix"):
        merge.main()
    assert not (tmp_path / "merged").exists()


@pytest.mark.parametrize("missing,conflicting", [(False, False), (True, False), (False, True)])
def test_shared_eligibility_is_applied_to_floor_and_validated(tmp_path, monkeypatch, missing, conflicting):
    spans = [{"text": "a", "start": 0, "end": 1}, {"text": "b", "start": 1, "end": 2}]
    write_json(tmp_path / "gt/task.json", {"dataset": "task", "episodes": {"3": spans}})
    write_json(tmp_path / "split/task.json", {"dataset": "task", "seed": [], "eval": [3]})
    write_json(tmp_path / "preds/floor.json", {"dataset": "task", "arm": "floor_uniform",
        "ok": True, "episodes": {"3": spans},
        **({"calibration_eligible": {"3": False}} if conflicting else {})})
    write_json(tmp_path / "mask.json", {"version": 1, "episodes": {"task": {} if missing else {"3": True}}})
    monkeypatch.setattr(sys, "argv", ["score", "--predictions-dir", str(tmp_path / "preds"),
        "--gt-dir", str(tmp_path / "gt"), "--splits-dir", str(tmp_path / "split"),
        "--eligibility", str(tmp_path / "mask.json"), "--out", str(tmp_path / "scores.jsonl")])
    if missing or conflicting:
        with pytest.raises(SystemExit, match="shared eligibility|conflicts"):
            score_alignment.main()
        assert not (tmp_path / "scores.jsonl").exists()
    else:
        assert score_alignment.main() == 0
        row = json.loads((tmp_path / "scores.jsonl").read_text())
        assert row["calibration_eligible"]
        assert not row["calibration_applied"]
        assert row["calibration_eligibility_source"] == "shared_seed_count_support"
