"""Applicability and truthful application reporting for frozen calibration files."""

from __future__ import annotations

import json
import logging
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

from lerobot_align.config import PlanConfig  # noqa: E402
from lerobot_align.modules.plan_subtasks_memory import (  # noqa: E402
    PlanSubtasksMemoryModule,
    _AlignCalibration,
    _load_align_calibration,
)
from lerobot_align.reader import EpisodeRecord  # noqa: E402
from lerobot_align.vlm_client import StubVlmClient  # noqa: E402


def _record(timestamps: tuple[float, ...] = (0.0, 1.0, 2.0)) -> EpisodeRecord:
    return EpisodeRecord(
        episode_index=17,
        episode_task="task",
        frame_timestamps=timestamps,
        frame_indices=tuple(range(len(timestamps))),
        data_path=Path("unused.parquet"),
        row_offset=0,
        row_count=len(timestamps),
    )


def _module(calibration: _AlignCalibration | None) -> PlanSubtasksMemoryModule:
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _: pytest.fail("calibration must not call a VLM")),
        config=PlanConfig(),
    )
    module._calibration = calibration
    return module


def _count_calibration(**overrides) -> _AlignCalibration:
    return _AlignCalibration(
        **{
            "offsets": (0.2, 0.3),
            "label_scope": "segment_count",
            "n_segments": 3,
            **overrides,
        }
    )


def _events(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(record.getMessage().removeprefix("ALIGN_CALIBRATION "))
        for record in caplog.records
        if record.getMessage().startswith("ALIGN_CALIBRATION ")
    ]


def test_count_scope_applies_across_wording_and_keeps_parameters_frozen(caplog) -> None:
    calibration = _count_calibration()
    module = _module(calibration)
    before_model = asdict(calibration)
    prediction = {0: (0.0, 0.8), 1: (0.8, 1.6), 2: (1.6, 2.0)}
    before_prediction = dict(prediction)
    with caplog.at_level(logging.INFO):
        first = module._apply_align_calibration(_record(), ["a", "b", "c"], prediction)
        second = module._apply_align_calibration(
            replace(_record(), episode_index=18), ["different", "wording", "here"], prediction
        )
    assert first == second
    assert first[1][0] == pytest.approx(0.6)
    assert first[2][0] == pytest.approx(1.3)
    assert asdict(calibration) == before_model
    assert module._calibration is calibration
    assert prediction == before_prediction
    with pytest.raises(FrozenInstanceError):
        calibration.offsets = (100.0, 100.0)
    assert _events(caplog) == [
        {"episode": episode, "applied": True, "reason": "applied", "method": "offsets"}
        for episode in (17, 18)
    ]


@pytest.mark.parametrize(
    ("calibration", "labels", "timestamps", "prediction", "generated", "reason"),
    [
        (None, ["a", "b", "c"], (0.0, 2.0), {0: (0.0, 2.0)}, False, "not_configured"),
        (_count_calibration(), ["a", "b", "c"], (0.0, 2.0), {}, False, "empty_prediction"),
        (
            _count_calibration(),
            ["a", "b"],
            (0.0, 2.0),
            {0: (0.0, 2.0)},
            False,
            "segment_count_mismatch",
        ),
        (
            _AlignCalibration(offsets=(0.1,), labels=("a", "b")),
            ["a", "c"],
            (0.0, 2.0),
            {0: (0.0, 2.0)},
            False,
            "label_mismatch",
        ),
        (
            _count_calibration(),
            ["a", "b", "c"],
            (0.0, 2.0),
            {0: (0.0, 2.0)},
            True,
            "generated_requires_exact_labels",
        ),
        (
            _AlignCalibration(offsets=(0.1,)),
            ["a", "b"],
            (0.0, 2.0),
            {0: (0.0, 2.0)},
            True,
            "generated_requires_exact_labels",
        ),
        (
            _count_calibration(),
            ["a", "b", "c"],
            (1.0, 1.0),
            {0: (0.0, 2.0)},
            False,
            "invalid_duration",
        ),
        (_count_calibration(), ["a", "b", "c"], (), {0: (0.0, 2.0)}, False, "invalid_duration"),
        (
            _count_calibration(),
            ["a", "b", "c"],
            (0.0, float("nan")),
            {0: (0.0, 2.0)},
            False,
            "invalid_duration",
        ),
    ],
)
def test_skips_report_actual_reason(
    calibration, labels, timestamps, prediction, generated, reason, caplog
) -> None:
    with caplog.at_level(logging.INFO):
        result = _module(calibration)._apply_align_calibration(
            _record(timestamps), labels, prediction, require_labeled_calibration=generated
        )
    assert result is prediction
    assert _events(caplog) == [{"episode": 17, "applied": False, "reason": reason, "method": None}]


def test_duration_solver_and_partial_prediction_report_distinct_methods(caplog) -> None:
    module = _module(_count_calibration(segment_fractions=(0.3, 0.35, 0.35)))
    complete = {0: (0.0, 0.8), 1: (0.8, 1.6), 2: (1.6, 2.0)}
    partial = {0: (0.0, 0.8), 2: (1.6, 2.0)}
    with caplog.at_level(logging.INFO):
        result = module._apply_align_calibration(_record(), ["a", "b", "c"], complete)
        module._apply_align_calibration(_record(), ["a", "b", "c"], partial)
    assert len(result) == 3
    assert [event["method"] for event in _events(caplog)] == ["duration_prior", "offsets"]
    assert all(event["applied"] for event in _events(caplog))


def test_explicit_exact_calibration_supports_generated_realign(caplog) -> None:
    module = _module(
        _AlignCalibration(offsets=(0.2,), labels=("a", "b"), label_scope="exact", n_segments=2)
    )
    with caplog.at_level(logging.INFO):
        result = module._apply_align_calibration(
            _record(),
            ["a", "b"],
            {0: (0.0, 1.0), 1: (1.0, 2.0)},
            require_labeled_calibration=True,
        )
    assert result[1][0] == pytest.approx(0.8)
    assert _events(caplog)[0]["applied"] is True


@pytest.mark.parametrize(
    "extra",
    [
        {"label_scope": "segment_count"},
        {"label_scope": "segment_count", "n_segments": 2},
        {"label_scope": "segment_count", "n_segments": 3.0},
        {"label_scope": "segment_count", "n_segments": True},
        {"label_scope": "segment_count", "n_segments": "3"},
        {"label_scope": "segment_count", "n_segments": 3, "labels": {}},
        {"label_scope": "segment_count", "n_segments": 3, "labels": ["a", "b", "c"]},
        {"label_scope": "exact", "labels": []},
        {"label_scope": "exact", "labels": ["a", "b"]},
        {"label_scope": "exact", "labels": ["a", " ", "c"]},
        {"label_scope": "exact", "labels": ["a", "b", "c"], "n_segments": 4},
        {"label_scope": "unknown"},
        {"label_scope": None},
        {"label_scope": []},
    ],
)
def test_explicit_schema_rejects_ambiguous_or_invalid_files(tmp_path, extra) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"offsets": [0.1, 0.2], **extra}))
    with pytest.raises(ValueError):
        _load_align_calibration(path)


def test_new_count_schema_round_trip_and_legacy_files(tmp_path) -> None:
    path = tmp_path / "calibration.json"
    path.write_text(
        json.dumps(
            {"offsets": [0.1, 0.2], "label_scope": "segment_count", "n_segments": 3, "labels": []}
        )
    )
    calibration = _load_align_calibration(path)
    assert calibration.applies_to(["a", "b", "c"])
    assert not calibration.applies_to(["a", "b"])
    for legacy in ([0.1], {"offsets": [0.1], "labels": []}):
        path.write_text(json.dumps(legacy))
        calibration = _load_align_calibration(path)
        assert calibration.label_scope is None
        assert calibration.applies_to(["a", "b", "c"])
