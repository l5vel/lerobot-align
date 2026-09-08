"""Frozen timing baselines use one shared task seed budget without label filtering."""

from __future__ import annotations

import json
import sys

import pytest

from evaluation.scripts import align_floors as floors


def _spans(episode, fractions, duration=100.0):
    cuts = [0.0, *[duration * value for value in fractions], duration]
    return [
        {"text": f"wording {episode} step {index}", "start": start, "end": end}
        for index, (start, end) in enumerate(zip(cuts[:-1], cuts[1:], strict=True))
    ]


def _inputs(tmp_path, truth, seeds, evaluation):
    truth_path = tmp_path / "task.json"
    truth_path.write_text(json.dumps({"dataset": "task", "episodes": truth}))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"seed": seeds, "eval": evaluation}))
    return truth_path, split_path


def _run(tmp_path, paths, **overrides):
    return floors.run_dataset(
        truth_path=paths[0],
        split_path=paths[1],
        label_path=None,
        out_dir=tmp_path / "out",
        arms=tuple(floors.ARMS),
        prior_scope=overrides.pop("prior_scope", "segment_count"),
        model_answers=None,
        model_answers_path=None,
        quiet=True,
        **overrides,
    )


def test_segment_count_fit_keeps_all_seed_wordings(tmp_path) -> None:
    truth = {i: _spans(i, [value]) for i, value in enumerate([0.1, 0.2, 0.7, 0.9])}
    paths = _inputs(tmp_path, truth, [0, 1, 2], [3])
    count_records = _run(tmp_path, paths)
    for record in count_records[1:]:
        assert record["episodes"]["3"][1]["start"] == pytest.approx(20.0)
        assert record["prior_applied"] == {"3": True}
        metadata = record["seed_priors_by_segment_count"]["2"]
        assert metadata["fit_episodes"] == [0, 1, 2]
        assert metadata["fit_episode_count"] == metadata["n_label_lists"] == 3
        assert record["seed_prior"] is None

    exact_records = _run(tmp_path, paths, prior_scope="exact")
    length_records = _run(tmp_path, paths, prior_scope="length")
    for record in exact_records[1:]:
        assert record["seed_prior"]["fit_episode_count"] == 1
        assert record["episodes"]["3"][1]["start"] == pytest.approx(50.0)
        assert record["prior_applied"] == {"3": False}
    for record in length_records[1:]:
        assert record["seed_prior"]["fit_episode_count"] == 1
        assert record["episodes"]["3"][1]["start"] == pytest.approx(10.0)
        assert "seed_priors_by_segment_count" not in record


def test_count_cohorts_share_ten_seeds_and_fallback_for_unsupported_counts(tmp_path) -> None:
    truth = {
        **{i: _spans(i, [0.2]) for i in range(3)},
        **{i: _spans(i, [0.2, 0.7]) for i in range(3, 6)},
        **{i: _spans(i, [0.1, 0.2, 0.3]) for i in range(6, 8)},
        **{i: _spans(i, [0.1, 0.2, 0.3, 0.4]) for i in range(8, 10)},
        10: _spans(10, [0.9]),
        11: _spans(11, [0.1, 0.9]),
        12: _spans(12, [0.1, 0.8, 0.9]),
        13: _spans(13, [0.1, 0.2, 0.3, 0.8, 0.9]),
    }
    paths = _inputs(tmp_path, truth, list(range(10)), [10, 11, 12, 13])
    uniform, *seeded = _run(tmp_path, paths)
    for record in seeded:
        priors = record["seed_priors_by_segment_count"]
        assert {int(count) for count, prior in priors.items() if prior["fitted"]} == {2, 3}
        assert priors["2"]["fit_episodes"] == [0, 1, 2]
        assert priors["3"]["fit_episodes"] == [3, 4, 5]
        assert {e for prior in priors.values() for e in prior["seed_episodes"]} == set(range(10))
        assert priors["4"]["reason"] == "insufficient_fit_episodes"
        assert record["prior_applied"] == {"10": True, "11": True, "12": False, "13": False}
        assert record["prior_applied_fraction"] == 0.5
        assert record["prior_coverage_by_segment_count"]["4"] == {"episodes": 1, "applied": 0}
        for episode in ("12", "13"):
            assert record["episodes"][episode] == uniform["episodes"][episode]


@pytest.mark.parametrize("scope", ["exact", "length", "segment_count"])
def test_more_than_ten_seed_trajectories_is_rejected(tmp_path, scope) -> None:
    paths = _inputs(tmp_path, {i: _spans(i, [0.2]) for i in range(12)}, list(range(11)), [11])
    with pytest.raises(SystemExit, match="at most 10"):
        _run(tmp_path, paths, prior_scope=scope)


def test_seed_eval_overlap_is_rejected(tmp_path) -> None:
    paths = _inputs(tmp_path, {i: _spans(i, [0.2]) for i in range(3)}, [0, 1, 2], [2])
    with pytest.raises(SystemExit, match="overlap"):
        _run(tmp_path, paths)


def test_eval_internal_times_cannot_change_fitted_priors_or_predictions(tmp_path) -> None:
    truth = {i: _spans(i, [0.2, 0.7]) for i in range(4)}
    paths = _inputs(tmp_path, truth, [0, 1, 2], [3])
    before = _run(tmp_path, paths)
    truth[3] = _spans(3, [0.01, 0.99])
    _inputs(tmp_path, truth, [0, 1, 2], [3])
    after = _run(tmp_path, paths)
    for left, right in zip(before, after, strict=True):
        assert left["seed_priors_by_segment_count"] == right["seed_priors_by_segment_count"]
        assert left["episodes"] == right["episodes"]


def test_minimum_counts_usable_episodes(tmp_path) -> None:
    truth = {i: _spans(i, [0.2]) for i in range(4)}
    truth[2] = _spans(2, [0.2], duration=0.0)
    paths = _inputs(tmp_path, truth, [0, 1, 2], [3])
    for record in _run(tmp_path, paths)[1:]:
        prior = record["seed_priors_by_segment_count"]["2"]
        assert prior["fit_episode_count"] == 2
        assert prior["skipped_episodes"] == [2]
        assert record["prior_applied"] == {"3": False}


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("start", float("nan"), "nonfinite_timestamp"),
        ("end", float("inf"), "nonfinite_timestamp"),
        ("end", -1.0, "nonpositive_span"),
        ("end", 20.0, "nonpositive_span"),
        ("start", -1.0, "nonincreasing_starts"),
        ("end", 90.0, "overlapping_spans"),
        ("start", "invalid", "invalid_timestamp"),
        ("end", None, "invalid_timestamp"),
    ],
)
def test_invalid_internal_seed_timing_is_excluded_before_fit_and_minimum(
    tmp_path, field, value, reason
) -> None:
    truth = {i: _spans(i, [0.2, 0.7]) for i in range(5)}
    truth[3][1][field] = value
    paths = _inputs(tmp_path, truth, [0, 1, 2, 3], [4])
    for record in _run(tmp_path, paths)[1:]:
        prior = record["seed_priors_by_segment_count"]["3"]
        assert prior["fit_episode_count"] == 3
        assert prior["fit_episodes"] == [0, 1, 2]
        assert prior["seed_episodes"] == [0, 1, 2, 3]
        assert prior["skipped_episodes"] == [3]
        assert prior["excluded_episodes"] == [{"episode": 3, "reason": reason}]
        assert [span["start"] for span in record["episodes"]["4"]] == pytest.approx([0, 20, 70])
        assert record["prior_applied"] == {"4": True}

    paths = _inputs(tmp_path, truth, [0, 1, 3], [4])
    uniform, *seeded = _run(tmp_path, paths)
    for record in seeded:
        prior = record["seed_priors_by_segment_count"]["3"]
        assert prior["fit_episode_count"] == 2
        assert prior["reason"] == "insufficient_fit_episodes"
        assert record["prior_applied"] == {"4": False}
        assert record["episodes"] == uniform["episodes"]


def test_cli_forwards_count_scope_and_fit_minimum_to_normal_and_self_test_runs(
    tmp_path, monkeypatch
) -> None:
    paths = _inputs(tmp_path, {i: _spans(i, [0.2]) for i in range(4)}, [0, 1, 2], [3])
    args = [
        "align_floors",
        "--gt",
        str(paths[0]),
        "--split",
        str(paths[1]),
        "--out-dir",
        str(tmp_path / "out"),
        "--prior-scope",
        "segment_count",
        "--min-fit-episodes",
        "4",
        "--arm-tables",
    ]
    monkeypatch.setattr(sys, "argv", args)
    assert floors.main() == 0
    record = json.loads((tmp_path / "out" / "task__floor_median_ratio.json").read_text())
    assert record["prior_applied"] == {"3": False}
    assert record["seed_priors_by_segment_count"]["2"]["min_fit_episodes"] == 4

    original = floors.run_dataset
    seen = []

    def run_spy(**kwargs):
        seen.append((kwargs["prior_scope"], kwargs["min_fit_episodes"]))
        return original(**kwargs)

    monkeypatch.setattr(floors, "run_dataset", run_spy)
    monkeypatch.setattr(sys, "argv", [*args, "--self-test"])
    assert floors.main() == 0
    assert seen == [("segment_count", 4)] * 3
