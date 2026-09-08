"""Exercise calibration selection through its CLI without a VLM or dataset loader."""

import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

from lerobot_align.diagnostics import fit_align_calibration as calibration


def make_inputs(tmp_path, label_lists):
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    truth, results = {}, []
    for episode, labels in enumerate(label_lists):
        spans = [
            {"label": label, "start": i * 1.0, "end": i + 1.0} for i, label in enumerate(labels)
        ]
        truth[str(episode)] = {"subtasks": spans}
        predicted = [
            {"text": span["label"], "start": span["start"], "end": span["end"]}
            for span in spans
        ]
        results.append({"episode": episode, "format": "video", "spans": predicted})
    (root / "meta" / "lerobot_annotations.json").write_text(json.dumps({"episodes": truth}))
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(results))
    output = tmp_path / "calibration.json"
    return [str(root), str(results_path), "--out", str(output), "--force", "--allow-legacy-provenance"], output


@pytest.mark.parametrize("count", [1, 2])
def test_tiny_fit_has_numeric_duration_weight(tmp_path, monkeypatch, count):
    args, output = make_inputs(tmp_path, [("pick", "place")] * count)
    monkeypatch.setattr(sys, "argv", ["lerobot-align-fit", *args])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["duration_weight"] == 1.0
    assert payload["fit_episode_count"] == count


@pytest.mark.parametrize("extra", [["--duration-weight", "0"], ["--no-duration-prior"]])
def test_tiny_fit_respects_explicit_prior_settings(tmp_path, monkeypatch, extra):
    args, output = make_inputs(tmp_path, [("pick", "place")])
    monkeypatch.setattr(sys, "argv", ["lerobot-align-fit", *args, *extra])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    if "--no-duration-prior" in extra:
        assert "segment_fractions" not in payload
        assert "duration_weight" not in payload
    else:
        assert payload["duration_weight"] == 0.0


@pytest.mark.parametrize("tied", [False, True])
def test_label_group_selection_is_exact_and_independent_of_hash_seed(tmp_path, tied):
    # Same lengths used to give every label list the same count, even when one
    # of them appears in only one episode. A tie now picks the lexical minimum.
    labels = [("z", "end")] * (3 if tied else 1) + [("a", "end")] * 3
    args, output = make_inputs(tmp_path, labels)
    for seed in range(8):
        result = subprocess.run(
            [sys.executable, "-m", calibration.__name__, *args],
            env={**os.environ, "PYTHONHASHSEED": str(seed), "CUDA_VISIBLE_DEVICES": ""},
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        payload = json.loads(output.read_text())
        assert payload["labels"] == ["a", "end"], f"seed={seed}: {payload}"
        assert payload["fit_on_episodes"] == list(range(len(labels) - 3, len(labels)))


def test_count_scope_uses_all_freeform_seed_labels(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [(f"pick {i}", f"place {i}") for i in range(10)])
    monkeypatch.setattr(sys, "argv", [
        "fit", *args, "--label-scope", "segment_count", "--min-fit-episodes", "3",
        "--episodes", *map(str, range(10)), "--task-id", "task-a", "--arm", "align_video_cal",
    ])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["label_scope"] == "segment_count"
    assert payload["n_segments"] == 2
    assert payload["labels"] == []
    assert payload["seed_episodes"] == payload["fit_on_episodes"] == list(range(10))
    assert payload["fit_episode_count"] == 10
    assert payload["excluded_episodes"] == []
    assert payload["task_id"] == "task-a"
    assert payload["arm"] == "align_video_cal"


def test_count_scope_rejects_mixed_counts_instead_of_selecting_mode(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b")] * 3 + [("a", "b", "c")])
    monkeypatch.setattr(sys, "argv", ["fit", *args, "--label-scope", "segment_count"])
    assert calibration.main() == 1
    assert not output.exists()


def test_seed_allowlist_prevents_evaluation_rows_affecting_fit(tmp_path, monkeypatch):
    # More than ten rows are present, but only three have been allocated as seeds.
    args, output = make_inputs(tmp_path, [("pick", "place")] * 12)
    cli = ["fit", *args, "--episodes", "0", "1", "2", "--label-scope", "segment_count"]
    monkeypatch.setattr(sys, "argv", cli)
    assert calibration.main() == 0
    before = json.loads(output.read_text())
    rows = json.loads(Path(args[1]).read_text())
    annotations_path = tmp_path / "ds" / "meta" / "lerobot_annotations.json"
    annotations = json.loads(annotations_path.read_text())
    for row in rows[3:]:
        row["spans"] = [{"text": "evaluation-only", "start": float("nan"), "end": -1}]
        annotations["episodes"][str(row["episode"])]["subtasks"] = [
            {"label": "other-count", "start": 0, "end": 900}
        ]
    Path(args[1]).write_text(json.dumps(rows))
    annotations_path.write_text(json.dumps(annotations))
    assert calibration.main() == 0
    after = json.loads(output.read_text())
    # Numeric fits still ignore evaluation-only rows. Provenance must record
    # that the source files changed even when their fitting subset did not.
    for key in ('source_results_sha256', 'ground_truth_sha256'):
        assert after.pop(key) != before.pop(key)
    assert after == before
    assert before["fit_on_episodes"] == [0, 1, 2]


def test_allocation_over_ten_is_rejected_even_if_extra_rows_are_unusable(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b")] * 11)
    monkeypatch.setattr(sys, "argv", ["fit", *args])
    assert calibration.main() == 1
    assert not output.exists()


@pytest.mark.parametrize("flags", [
    ["--max-fit-episodes", "11"],
    ["--min-fit-episodes", "0"],
    ["--min-fit-episodes", "4", "--max-fit-episodes", "3"],
    ["--episodes", "0", "0"],
])
def test_invalid_seed_budget_cannot_be_forced(tmp_path, monkeypatch, flags):
    args, output = make_inputs(tmp_path, [("a", "b")])
    monkeypatch.setattr(sys, "argv", ["fit", *args, *flags])
    with pytest.raises(SystemExit, match="2"):
        calibration.main()
    assert not output.exists()


def test_unusable_predictions_are_excluded_and_counted(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 8)
    rows = json.loads(Path(args[1]).read_text())
    rows[3]["spans"].pop(1)  # a,c must not compare c's start to b's GT start
    rows[4]["spans"][1]["text"] = "wrong label"
    rows[5]["spans"][1]["start"] = float("nan")
    for span in rows[6]["spans"]:
        span["start"] = span["end"] = 0
    rows[7]["error"] = "model failed"
    Path(args[1]).write_text(json.dumps(rows))
    monkeypatch.setattr(sys, "argv", ["fit", *args, "--min-fit-episodes", "3"])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["fit_on_episodes"] == [0, 1, 2]
    assert payload["fit_episode_count"] == 3
    assert payload["offsets"] == [0.0, 0.0]
    assert [row["episode"] for row in payload["excluded_episodes"]] == [3, 4, 5, 6, 7]
    reasons = " ".join(row["reason"] for row in payload["excluded_episodes"])
    for phrase in ("incomplete", "labels", "nonfinite", "duration", "model failed"):
        assert phrase in reasons


def test_minimum_applies_after_complete_prediction_validation(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 3)
    rows = json.loads(Path(args[1]).read_text())
    rows[-1]["spans"].pop(1)
    Path(args[1]).write_text(json.dumps(rows))
    monkeypatch.setattr(sys, "argv", ["fit", *args, "--min-fit-episodes", "3"])
    assert calibration.main() == 1
    assert not output.exists()


def test_direct_fit_excludes_partial_prediction_instead_of_corrupting_positions():
    truth = [{"label": label, "start": start, "end": end}
             for label, start, end in [("a", 0, 2), ("b", 2, 5), ("c", 5, 10)]]
    predicted = [{"text": "a", "start": 0, "end": 5},
                 {"text": "c", "start": 5, "end": 10}]
    assert calibration.fit({0: predicted}, {0: truth}, [0]) == {}
    with pytest.raises(ValueError, match="incomplete"):
        calibration.signed_errors(predicted, truth)


def test_no_duration_prior_cross_validation_scores_offsets_only(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 3)
    truth = {
        e: [{"label": label, "start": start, "end": end}
            for label, start, end in [("a", 0, 2), ("b", 2, 5), ("c", 5, 10)]]
        for e in range(3)
    }
    predicted = {
        e: [{"text": label, "start": start, "end": end}
            for label, start, end in [("a", 0, 3 + e), ("b", 3 + e, 6 + e), ("c", 6 + e, 10)]]
        for e in range(3)
    }
    (tmp_path / "ds" / "meta" / "lerobot_annotations.json").write_text(json.dumps({
        "episodes": {str(e): {"subtasks": spans} for e, spans in truth.items()}
    }))
    Path(args[1]).write_text(json.dumps([
        {"episode": e, "format": "video", "spans": spans} for e, spans in predicted.items()
    ]))
    monkeypatch.setattr(sys, "argv", [
        "fit", *args, "--no-duration-prior", "--duration-weight", "4",
    ])
    assert calibration.main() == 0
    corrected, baseline = [], []
    for episode in range(3):
        model = calibration.fit(predicted, truth, [e for e in range(3) if e != episode])
        model.pop("segment_fractions")
        corrected.append(calibration.score(predicted, truth, [episode], model, 4)[0])
        baseline.append(calibration.score(predicted, truth, [episode], {"offsets": [0, 0]})[0])
    expected = 100 * (statistics.fmean(corrected) - statistics.fmean(baseline))
    payload = json.loads(output.read_text())
    assert payload["heldout_gain_points"] == pytest.approx(expected, abs=0.0005)
    assert statistics.fmean(corrected) < 0.7  # the wrong duration-prior CV gives 1.0


def test_diagnostic_solver_drops_collided_labels_like_runtime():
    predicted = [{"text": label, "start": start, "end": end}
                 for label, start, end in [("a", 0, 5), ("b", 5, 5), ("c", 5, 10)]]
    corrected = calibration.apply(predicted, {
        "offsets": [0, 0], "segment_fractions": [0.3, 0.3, 0.4],
        "residual_scale": 0.05, "duration_scale": 0.05,
    }, weight=0)
    assert [span["text"] for span in corrected] == ["a", "b"]
    assert corrected[-1]["end"] == 10


def as_jsonl(args, rows, name="results.jsonl"):
    """Rewrite the results argument as the append-only sidecar eval_align_batch writes."""
    path = Path(args[1]).with_name(name)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return [args[0], str(path), *args[2:]]


def test_jsonl_sidecar_fits_identically_to_the_json_array(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 4)
    monkeypatch.setattr(sys, "argv", ["fit", *args])
    assert calibration.main() == 0
    from_array = json.loads(output.read_text())

    rows = json.loads(Path(args[1]).read_text())
    monkeypatch.setattr(sys, "argv", ["fit", *as_jsonl(args, rows)])
    assert calibration.main() == 0
    from_jsonl = json.loads(output.read_text())

    assert from_array["source_results_format"] == "json"
    assert from_jsonl["source_results_format"] == "jsonl"
    for key in ("offsets", "segment_fractions", "fit_on_episodes", "heldout_gain_points"):
        assert from_jsonl[key] == from_array[key]


def test_crashed_run_rerun_supersedes_earlier_rows_instead_of_excluding_them(
    tmp_path, monkeypatch, capsys
):
    # Run one dies after four episodes; the full re-run appends a second copy of
    # every one of them. Without last-wins these are "duplicate prediction rows".
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 4)
    rows = json.loads(Path(args[1]).read_text())
    first_pass = [dict(row) for row in rows[:2]]
    first_pass[1] = {**first_pass[1], "error": "TimeoutError: boom"}
    first_pass[1].pop("spans")
    monkeypatch.setattr(sys, "argv", ["fit", *as_jsonl(args, [*first_pass, *rows])])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["fit_on_episodes"] == [0, 1, 2, 3]
    assert payload["excluded_episodes"] == []
    out = capsys.readouterr().out
    assert "superseded ep 0 video: line 3 replaces line 1" in out
    assert "superseded ep 1 video: line 4 replaces line 2" in out
    assert "earlier attempt errored: TimeoutError: boom" in out
    assert [(n["episode"], n["resolution"]) for n in payload["resolved_collisions"]] == [
        (0, "superseded"), (1, "superseded"),
    ]


def test_jsonl_keeps_distinct_formats_apart(tmp_path, monkeypatch, capsys):
    # Video first, contact_sheet last: a supersede key that ignored `format`
    # would let the trailing rows displace the ones the fit needs.
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 3)
    rows = json.loads(Path(args[1]).read_text())
    other = [{**row, "format": "contact_sheet"} for row in rows]
    cli = ["fit", *as_jsonl(args, [*rows, *other])]
    monkeypatch.setattr(sys, "argv", cli)
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["fit_on_episodes"] == [0, 1, 2]
    assert payload["resolved_collisions"] == []
    assert "superseded" not in capsys.readouterr().out

    monkeypatch.setattr(sys, "argv", [*cli, "--format", "contact_sheet"])
    assert calibration.main() == 0
    assert json.loads(output.read_text())["fit_on_episodes"] == [0, 1, 2]


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("", "file is empty"),
        ("   \n\n", "file is empty"),
        ('{"episode": 0, "format": "video"}\n{"episode": 1, ', ":2: not valid JSON"),
        ('{"episode": 0, "format": "video"}\n"not-a-row"\n', ":2: expected a JSON object"),
        ('[{"episode": 0, "format": "video"}, 7]', "row 1 is int"),
    ],
)
def test_malformed_results_report_the_location_and_do_not_write(
    tmp_path, monkeypatch, capsys, body, message
):
    args, output = make_inputs(tmp_path, [("a", "b")])
    path = Path(args[1]).with_name("broken.jsonl")
    path.write_text(body)
    monkeypatch.setattr(sys, "argv", ["fit", args[0], str(path), *args[2:]])
    assert calibration.main() == 1
    assert not output.exists()
    assert message in capsys.readouterr().err


def test_partial_last_line_is_reported_not_silently_dropped(tmp_path):
    args, _ = make_inputs(tmp_path, [("a", "b")] * 2)
    rows = json.loads(Path(args[1]).read_text())
    path = Path(args[1]).with_name("partial.jsonl")
    path.write_text(json.dumps(rows[0]) + "\n" + json.dumps(rows[1])[:20])
    with pytest.raises(ValueError, match="partial.jsonl:2: not valid JSON"):
        calibration.load_result_rows(path)


def test_trailing_and_blank_lines_are_tolerated(tmp_path):
    args, _ = make_inputs(tmp_path, [("a", "b")] * 2)
    rows = json.loads(Path(args[1]).read_text())
    path = Path(args[1]).with_name("gappy.jsonl")
    path.write_text("\n".join(["", json.dumps(rows[0]), "", json.dumps(rows[1]), "", ""]))
    loaded, superseded, source_format = calibration.load_result_rows(path)
    assert [row["episode"] for row in loaded] == [0, 1]
    assert superseded == []
    assert source_format == "jsonl"


def test_rows_without_an_episode_key_fail_loudly(tmp_path, monkeypatch, capsys):
    args, output = make_inputs(tmp_path, [("a", "b")] * 2)
    rows = json.loads(Path(args[1]).read_text())
    rows[1].pop("episode")
    Path(args[1]).write_text(json.dumps(rows))
    monkeypatch.setattr(sys, "argv", ["fit", *args])
    assert calibration.main() == 1
    assert not output.exists()
    assert "have no 'episode' key" in capsys.readouterr().err


def test_a_later_failure_never_discards_an_already_scored_row(tmp_path, monkeypatch, capsys):
    # A re-run against a flaky endpoint appends errors for episodes that already
    # scored. Taking those would shrink the cohort and then blame the model.
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 3)
    rows = json.loads(Path(args[1]).read_text())
    retry = {"episode": 0, "format": "video", "error": "ConnectionError: vLLM died"}
    monkeypatch.setattr(sys, "argv", ["fit", *as_jsonl(args, [*rows, retry])])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["fit_on_episodes"] == [0, 1, 2]
    assert payload["excluded_episodes"] == []
    assert payload["resolved_collisions"] == [{
        "episode": 0, "format": "video", "line": 4, "previous_line": 1,
        "resolution": "kept_earlier",
        "detail": "later attempt errored: ConnectionError: vLLM died",
    }]
    assert "kept_earlier ep 0 video: line 4 kept over line 1" in capsys.readouterr().out


def test_a_later_success_still_supersedes_an_earlier_failure(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 3)
    rows = json.loads(Path(args[1]).read_text())
    failed = {"episode": 0, "format": "video", "error": "TimeoutError: stalled"}
    monkeypatch.setattr(sys, "argv", ["fit", *as_jsonl(args, [failed, *rows])])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["fit_on_episodes"] == [0, 1, 2]
    assert payload["resolved_collisions"][0]["resolution"] == "superseded"


def test_two_failures_for_one_episode_still_exclude_it(tmp_path, monkeypatch):
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 4)
    rows = json.loads(Path(args[1]).read_text())
    rows[0] = {"episode": 0, "format": "video", "error": "first"}
    extra = {"episode": 0, "format": "video", "error": "second"}
    monkeypatch.setattr(sys, "argv", ["fit", *as_jsonl(args, [*rows, extra]), "--min-fit-episodes", "3"])
    assert calibration.main() == 0
    payload = json.loads(output.read_text())
    assert payload["fit_on_episodes"] == [1, 2, 3]
    assert payload["excluded_episodes"] == [
        {"episode": 0, "reason": "prediction error: second"}
    ]


def test_collision_notes_name_the_runs_that_produced_them(tmp_path, monkeypatch, capsys):
    args, output = make_inputs(tmp_path, [("a", "b", "c")] * 3)
    rows = json.loads(Path(args[1]).read_text())
    first = [{**row, "run_id": "aaaa1111"} for row in rows]
    second = [{**row, "run_id": "bbbb2222"} for row in rows]
    monkeypatch.setattr(sys, "argv", ["fit", *as_jsonl(args, [*first, *second])])
    assert calibration.main() == 0
    note = json.loads(output.read_text())["resolved_collisions"][0]
    assert (note["run_id"], note["previous_run_id"]) == ("bbbb2222", "aaaa1111")
    assert "[run bbbb2222 vs aaaa1111]" in capsys.readouterr().out


def test_malformed_json_array_names_the_file(tmp_path, monkeypatch, capsys):
    args, output = make_inputs(tmp_path, [("a", "b")])
    path = Path(args[1]).with_name("truncated.json")
    path.write_text('[{"episode": 0, "format": "video"},\n {"episode": 1,\n]')
    monkeypatch.setattr(sys, "argv", ["fit", args[0], str(path), *args[2:]])
    assert calibration.main() == 1
    assert not output.exists()
    err = capsys.readouterr().err
    assert "truncated.json" in err and "line 3" in err
