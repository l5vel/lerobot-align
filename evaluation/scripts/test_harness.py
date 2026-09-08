"""Small CPU fixtures for aggregation, gates, accounting and publication."""
# ruff: noqa: E402

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import aggregate
import breakdown
import c1_gate
import gate_fingerprint
import make_report
import preflight
import reliability
import regenerate_reports
import run_arm
import score_runs
from contrast_policy import contrast_policy
from metrics.hierarchical import _bias_corrected
from metrics.stats import cluster_bootstrap_mean, equivalence_test


def invoke(monkeypatch, module, *args):
    monkeypatch.setattr(sys, "argv", [module.__file__, *map(str, args)])
    return module.main()


def test_regenerator_does_not_mistake_a_crash_for_a_gate_block(monkeypatch):
    def failed(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "", "Traceback: failure")
    monkeypatch.setattr(regenerate_reports.subprocess, "run", failed)
    with pytest.raises(RuntimeError, match="c1_gate.py failed"):
        regenerate_reports.run("c1_gate.py", [], diagnostic=True)


def test_regenerator_keeps_a_completed_block_result(monkeypatch):
    def blocked(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 1, "C1 verdict: INCONCLUSIVE  ->  gate BLOCK\n", "")
    monkeypatch.setattr(regenerate_reports.subprocess, "run", blocked)
    regenerate_reports.run("c1_gate.py", [], diagnostic=True)


@pytest.mark.parametrize("value, inert", [(True, False), (False, True), (0, True),
                                          (1, False), (2.5, False), (None, True)])
def test_new_numeric_defaults_are_not_exempt(value, inert):
    assert preflight.inert_default(value) is inert


def test_tied_bootstrap_is_symmetric():
    ci = cluster_bootstrap_mean({"d": [-1., 1.]}, n_boot=4000, level=.6, seed=3)
    assert (ci.low, ci.high) == (-1, 1)
    ci = _bias_corrected(0, [-1] * 3 + [0] * 4 + [1] * 3, .6, 2, 1)
    assert (ci.low, ci.high) == (-1, 1)


def test_repeats_remain_one_episode_and_can_change_tost_verdict():
    rows = [{"dataset": "d", "episode": ep, "repeat": r, "arm": arm, "macro_iou": value}
            for ep, delta in [(1, -.2), (2, .2)] for r in (1, 2, 3)
            for arm, value in [("baseline", .4), ("candidate", .4 + delta)]]
    deltas, unpaired, n_pairs = c1_gate.paired_episode_deltas(
        rows, "macro_iou", "candidate", "baseline")
    assert len(deltas["d"]) == 2
    assert (unpaired, n_pairs) == (0, 6)
    corrected = equivalence_test(deltas, .15, n_boot=4000, seed=13)
    pseudoreplicated = equivalence_test({"d": [-.2] * 3 + [.2] * 3}, .15,
                                      n_boot=4000, seed=13)
    assert corrected["verdict"] == "INCONCLUSIVE"
    assert pseudoreplicated["verdict"] == "EQUIVALENT"
    reversed_deltas, _, _ = c1_gate.paired_episode_deltas(
        rows, "macro_iou", "baseline", "candidate")
    assert reversed_deltas["d"] == pytest.approx([-x for x in deltas["d"]])


@pytest.mark.parametrize("side", ["candidate", "baseline"])
@pytest.mark.parametrize("bad", ["unknown", "supervision", "quarantine"])
def test_both_contrast_sides_fail_closed(side, bad):
    meta = {k: {"tool": "align", "supervision": "none", "equalised": True}
            for k in ("candidate", "baseline")}
    if bad == "unknown":
        del meta[side]
    elif bad == "supervision":
        del meta[side]["supervision"]
    else:
        meta[side]["quarantined"] = True
    assert contrast_policy("candidate", "baseline", meta)[0]
    result = breakdown.contrast({}, {}, baseline="baseline", arm="candidate", metadata=meta)
    assert result["unavailable"]


def test_breakdown_is_exploratory_and_has_no_pvalues():
    a = {("d", i): dict.fromkeys((*breakdown.HEADLINE, "log_ratio"), 0.2) for i in range(4)}
    b = {k: {m: x + .1 for m, x in v.items()} for k, v in a.items()}
    meta = {k: {"tool": "align", "supervision": "none"} for k in ("a", "b")}
    result = breakdown.contrast(a, b, baseline="a", arm="b", metadata=meta)
    for metric in (*breakdown.HEADLINE, "log_ratio"):
        entry = result[metric]
        assert entry["exploratory"]
        assert entry["point"] == pytest.approx(.1)
        assert not ({"p", "p_holm", "p_bootstrap"} & entry.keys())


def test_aggregate_c3_and_means_use_same_paired_population(tmp_path, monkeypatch):
    names = [f"{a}__p__w" for a in ("baseline_upstream", "align_video", "align_video_realign")]
    config = tmp_path / "arms.yaml"
    config.write_text(yaml.safe_dump({"arms": [{"name": n, "tool": "align", "profile": "p", "camera": "w",
                                                   "supervision": "none", "equalised": True} for n in names]}))
    rows = []
    for arm, values in zip(names, [[.1, .9], [.2, .8], [.4, None]], strict=True):
        for ep, value in enumerate(values):
            rows.append({"dataset": "d", "episode": ep, "arm": arm, "matcher": "exact", "ok": value is not None,
                             "macro_iou": value})
    scores = tmp_path / "scores.jsonl"
    scores.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "report.json"
    assert invoke(monkeypatch, aggregate, "--scores", scores, "--out", out, "--matcher", "exact",
                  "--arms-config", config, "--metrics", "macro_iou", "--n-boot", 100) == 0
    report = json.loads(out.read_text())
    key = names[2] + " [C3]"
    entry = report["contrasts"][key]["macro_iou"]
    assert report["contrast_baselines"][key] == names[1]
    assert entry["n_paired_episodes"] == entry["n_clusters"] == 1
    assert entry["paired_arm_mean"] == .4
    assert entry["paired_baseline_mean"] == .2
    assert entry["point"] == pytest.approx(entry["paired_arm_mean"] - entry["paired_baseline_mean"])
    assert report["per_arm"][names[1]]["macro_iou"]["point"] == .5
    assert report["holm_family_size"] == 3


@pytest.mark.parametrize("delta, lower, marker", [(.1, False, "\\*"), (-.1, False, "**!**"),
                                               (-.1, True, "\\*")])
def test_report_direction_without_aggregate_specific_field(delta, lower, marker):
    entry = {"point": delta, "ci_low": delta, "ci_high": delta, "p_holm": .001, "lower_is_better": lower}
    assert marker in make_report.fmt_delta(entry)
    entry["exploratory"] = True
    assert marker not in make_report.fmt_delta(entry)


def test_published_contrast_keeps_supervision_caveat():
    meta = {"prior": {"tool": "reference", "supervision": "seed_calibration"},
            "video": {"tool": "align", "supervision": "none"}}
    reason, warning = contrast_policy("video", "prior", meta, allow_reference=True)
    assert reason is None
    entry = {"point": -.2, "ci_low": -.3, "ci_high": -.1, "paired_arm_mean": .4,
                 "paired_baseline_mean": .6, "n_paired_episodes": 20, "n_clusters": 2}
    report = {"contrasts": {"video": {"macro_iou": entry}}, "contrast_baselines": {"video": "prior"},
                  "supervision_asymmetry": {"video": warning}}
    assert warning in "\n".join(make_report.paired_table(report))


def test_all_gate_cells_loaded_and_primary_alias_deduplicated(tmp_path):
    gate = {"baseline": "upstream__wrap__wrist", "arm": "align__wrap__wrist", "verdict": "EQUIVALENT", "gate": "PASS"}
    for name in ("c1_gate.json", "c1_gate__wrap__wrist.json"):
        (tmp_path / name).write_text(json.dumps(gate))
    second = {"baseline": "upstream__think__wrist", "arm": "align__think__wrist", "verdict": "INCONCLUSIVE", "gate": "BLOCK"}
    (tmp_path / "c1_gate__think__wrist.json").write_text(json.dumps(second))
    result = make_report.load_gates(tmp_path)
    assert len(result["cells"]) == 2
    assert result["gate"] == "BLOCK"


def test_legacy_fingerprint_validates_definitions_not_hash_key_alias(monkeypatch):
    old = {"arms_yaml": "old-file", "baseline": "b", "arm": "a", "model_id": "m"}
    now = {"arms_certified": "pair", "arms_yaml_file": "new-file", "baseline": "b", "arm": "a", "model_id": "m"}
    monkeypatch.setattr(gate_fingerprint, "legacy_certified_hash", lambda *args: "pair")
    assert not gate_fingerprint.diff(old, now)
    assert gate_fingerprint.diff(old, now | {"arms_certified": "changed-pair"})
    assert gate_fingerprint.diff(old, now | {"model_id": "changed-model"})
    monkeypatch.setattr(gate_fingerprint, "legacy_certified_hash", lambda *args: None)
    assert gate_fingerprint.diff(old, now)


@pytest.mark.parametrize("gpus", ["0,1,2, 4", "0,\t4,1", "4", "0,04", "0,,1", "GPU-4"])
def test_gpu_policy_refuses_whitespace_and_invalid_aliases(gpus):
    result = subprocess.run(["bash", str(HERE / "run_all.sh"), "serve"],
                            env=os.environ | {"ALLOWED_GPUS": gpus}, capture_output=True, text=True)
    assert result.returncode != 0
    assert "FATAL" in result.stderr


def test_no_matching_arm_filter_fails_before_gate_or_dispatch():
    result = subprocess.run(["bash", str(HERE / "run_all.sh"), "full"],
                            env=os.environ | {"ARM_FILTER": "^not_an_arm$", "PYTHON": sys.executable},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert "matches no arms" in result.stderr
    assert "dispatch" not in result.stdout


def test_reliability_decomposes_comparable_and_additional_modes():
    row = {"jobs": 1, "jobs_ok": 1, "episodes_requested": 10, "episodes_published": 10}
    result = reliability.decomposition({"baseline_upstream__p__w": row,
                                        "align_defaults__p__w": row,
                                        "align_video__p__w": row,
                                        "align_video_realign__p__w": row | {"episodes_published": 0}})
    assert result["fork_matched_modes"]["episode_yield"] == 1
    assert result["fork_additional_modes"]["episode_yield"] == 0


def test_score_entrypoint_refuses_missing_split_without_truncating(tmp_path, monkeypatch):
    gt, pred, split = [tmp_path / x for x in ("gt", "pred", "split")]
    for directory in (gt, pred, split):
        directory.mkdir()
    (gt / "d.json").write_text(json.dumps({"dataset": "d", "episodes": {"0": [{"start": 0, "end": 1}]}}))
    (pred / "d.json").write_text(json.dumps({"dataset": "d", "arm": "a", "episodes": {}}))
    out = tmp_path / "scores.jsonl"
    out.write_text("previous complete scores")
    with pytest.raises(SystemExit, match="missing or empty evaluation split"):
        invoke(monkeypatch, score_runs, "--predictions-dir", pred, "--gt-dir", gt,
               "--splits-dir", split, "--out", out, "--matchers", "exact")
    assert out.read_text() == "previous complete scores"


def test_scoring_continues_after_malformed_span_and_records_threshold(tmp_path, monkeypatch):
    from metrics.semantic import ExactMatcher, EmbeddingMatcher
    gt, pred, split = [tmp_path / x for x in ("gt", "pred", "split")]
    for directory in (gt, pred, split):
        directory.mkdir()
    span = {"start": 0, "end": 1, "text": "pick"}
    (gt / "d.json").write_text(json.dumps({"dataset": "d", "episodes": {"0": [span], "1": [span], "2": [span]}}))
    (pred / "d.json").write_text(json.dumps({"dataset": "d", "arm": "a", "episodes": {
        "0": [span], "1": [{"start": 0, "text": "pick"}], "2": [span]}}))
    (split / "d.json").write_text(json.dumps({"dataset": "d", "seed": [0], "eval": [1, 2]}))
    thresholds = []
    def matchers(names, model, threshold):
        thresholds.append(threshold)
        return [ExactMatcher(name="embedding")]
    monkeypatch.setattr(score_runs, "build_matchers", matchers)
    out = tmp_path / "scores.jsonl"
    assert invoke(monkeypatch, score_runs, "--predictions-dir", pred, "--gt-dir", gt,
                  "--splits-dir", split, "--out", out) == 0
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["episode"] for r in rows] == [1, 2]
    assert [r["ok"] for r in rows] == [False, True]
    assert thresholds == [EmbeddingMatcher.threshold] == [.93]
    assert all(r["embedding_threshold"] == .93 and r["metric_version"] == 2 for r in rows)


def test_timeout_writes_failure_artifact(tmp_path, monkeypatch):
    config, split, out = [tmp_path / n for n in ("arms.yaml", "split.json", "out.json")]
    config.write_text(yaml.safe_dump({"arms": [{"name": "a", "tool": "align", "supervision": "none", "flags": {}}]}))
    split.write_text(json.dumps({"eval": [10, 11]}))
    monkeypatch.setattr(run_arm, "source_hashes", lambda: {})
    monkeypatch.setattr(run_arm, "git_commit", lambda *a: "revision")
    monkeypatch.setattr(run_arm, "package_version", lambda *a: "version")
    monkeypatch.setattr(run_arm, "build_working_root", lambda *a, **k: None)
    monkeypatch.setattr(run_arm, "verify_clean", lambda *a: [])
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], kw["timeout"])
    monkeypatch.setattr(run_arm.subprocess, "run", timeout)
    assert invoke(monkeypatch, run_arm, "--dataset", "d", "--source-root", tmp_path,
                  "--arm", "a", "--arms-config", config, "--work-dir", tmp_path / "work",
                  "--out", out, "--episodes", split, "--base-url", "unused", "--model-id", "unused") == 1
    result = json.loads(out.read_text())
    assert result["timed_out"] and not result["ok"]
    assert result["returncode"] == 124
    assert result["n_episodes_requested"] == 2
    assert result["episodes"] == {}
    assert reliability.classify(out.with_suffix(".log"))[0] == "timeout"
