#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Importing subtask spans a dataset already records (no VLM call)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pandas = pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

from lerobot_align.config import PlanConfig  # noqa: E402
from lerobot_align.modules import PlanSubtasksMemoryModule  # noqa: E402
from lerobot_align.reader import iter_episodes  # noqa: E402
from lerobot_align.staging import EpisodeStaging  # noqa: E402
from lerobot_align.subtask_import import (  # noqa: E402
    IMPORT_SOURCES,
    LANGUAGE,
    LEROBOT_ANNOTATIONS,
    SARM,
    TASK_INDEX,
    SubtaskImporter,
    _from_language_persistent,
    _from_lerobot_annotations,
    _from_sarm,
    _from_task_index,
    _load_lerobot_annotations,
)
from lerobot_align.vlm_client import StubVlmClient  # noqa: E402


@dataclass
class _FakeRecord:
    """Enough of ``EpisodeRecord`` for the source functions under test."""

    episode_index: int = 0
    frame_timestamps: tuple[float, ...] = ()
    frame_tasks: tuple[str, ...] = ()
    persistent: Any = None

    def frames_df(self):
        return pandas.DataFrame({"language_persistent": [self.persistent] * len(self.frame_timestamps)})


def _exploding_vlm() -> StubVlmClient:
    """A VLM that fails the test if the pipeline calls it at all."""

    def responder(messages):
        raise AssertionError("import mode must not call the VLM for subtasks")

    return StubVlmClient(responder=responder)


def _ts(n: int, fps: int = 10) -> tuple[float, ...]:
    return tuple(round(i / fps, 6) for i in range(n))


def _write_lerobot_annotations(root: Path, payload: Any) -> Path:
    path = root / "meta" / "lerobot_annotations.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ----------------------------------------------------------------------
# Individual sources
# ----------------------------------------------------------------------


def test_from_sarm_reads_raw_float_times_not_rounded_seconds() -> None:
    """The raw ``*_times`` columns are read directly, so sub-second boundaries
    survive — going via ``load_annotations_from_dataset`` would quantise them
    to whole seconds through its ``mm:ss`` round-trip."""
    episodes = pandas.DataFrame(
        {
            "dense_subtask_names": [["open the drawer", "take the fork"]],
            "dense_subtask_start_times": [[0.0, 1.25]],
            "dense_subtask_end_times": [[1.25, 2.75]],
        },
        index=pandas.Index([0], name="episode_index"),
    )
    spans = _from_sarm(_FakeRecord(episode_index=0), episodes, "dense")
    assert spans == [
        {"text": "open the drawer", "start": 0.0, "end": 1.25},
        {"text": "take the fork", "start": 1.25, "end": 2.75},
    ]


def test_from_sarm_falls_back_to_other_prefix_then_legacy_columns() -> None:
    sparse_only = pandas.DataFrame(
        {
            "sparse_subtask_names": [["fold the towel"]],
            "sparse_subtask_start_times": [[0.0]],
            "sparse_subtask_end_times": [[2.0]],
        },
        index=pandas.Index([0], name="episode_index"),
    )
    assert _from_sarm(_FakeRecord(), sparse_only, "dense")[0]["text"] == "fold the towel"

    legacy = pandas.DataFrame(
        {
            "subtask_names": [["wipe the counter"]],
            "subtask_start_times": [[0.0]],
            "subtask_end_times": [[2.0]],
        },
        index=pandas.Index([0], name="episode_index"),
    )
    assert _from_sarm(_FakeRecord(), legacy, "dense")[0]["text"] == "wipe the counter"


def test_from_sarm_tolerates_unannotated_and_absent_episodes() -> None:
    episodes = pandas.DataFrame(
        {
            "dense_subtask_names": [None],
            "dense_subtask_start_times": [None],
            "dense_subtask_end_times": [None],
        },
        index=pandas.Index([0], name="episode_index"),
    )
    assert _from_sarm(_FakeRecord(episode_index=0), episodes, "dense") == []
    assert _from_sarm(_FakeRecord(episode_index=9), episodes, "dense") == []
    assert _from_sarm(_FakeRecord(), None, "dense") == []


def test_from_lerobot_annotations_reads_interval_schema_and_preserves_raw_times() -> None:
    episodes = {
        "7": {
            "subtasks": [
                {"label": "  reach the drawer  ", "start": 0.25, "end": 1.25, "extra": "ignored"},
                {"label": "open the drawer", "start": 1.25, "end": 2.04},
            ],
            "high_levels": [],
            "atoms": [{"style": "subtask", "content": "must not win", "timestamp": 0.0}],
        }
    }
    spans = _from_lerobot_annotations(_FakeRecord(episode_index=7, frame_timestamps=_ts(21)), episodes)
    assert spans == [
        {"text": "reach the drawer", "start": 0.25, "end": 1.25},
        {"text": "open the drawer", "start": 1.25, "end": 2.04},
    ]
    assert _from_lerobot_annotations(_FakeRecord(episode_index=8), episodes) == []


def test_from_lerobot_annotations_reconstructs_timestamped_atoms() -> None:
    episodes = {
        "0": {
            "subtasks": [],
            "atoms": [
                {"style": "plan", "content": "1. reach", "timestamp": 0.0},
                {"style": "subtask", "content": "put it down", "timestamp": 1.0},
                {"style": "subtask", "content": "  reach  ", "timestamp": 0.25},
            ],
        }
    }
    spans = _from_lerobot_annotations(_FakeRecord(episode_index=0, frame_timestamps=_ts(21)), episodes)
    assert spans == [
        {"text": "reach", "start": 0.25, "end": 1.0},
        {"text": "put it down", "start": 1.0, "end": 2.0},
    ]


@pytest.mark.parametrize(
    "bad_span",
    [
        {"label": "", "start": 0.5, "end": 1.0},
        {"label": "bad", "start": True, "end": 1.0},
        {"label": "bad", "start": "0.5", "end": 1.0},
        {"label": "bad", "start": 0.5, "end": float("inf")},
        {"label": "bad", "start": 1.0, "end": 1.0},
    ],
)
def test_from_lerobot_annotations_rejects_partial_malformed_intervals(
    bad_span: dict[str, Any], caplog
) -> None:
    episodes = {
        "0": {
            "subtasks": [
                {"label": "valid", "start": 0.0, "end": 0.5},
                bad_span,
            ]
        }
    }
    with caplog.at_level("WARNING"):
        assert _from_lerobot_annotations(_FakeRecord(), episodes) == []
    assert "malformed meta/lerobot_annotations.json entry" in caplog.text


def test_from_lerobot_annotations_rejects_partial_malformed_atoms(caplog) -> None:
    episodes = {
        "0": {
            "atoms": [
                {"style": "subtask", "content": "valid", "timestamp": 0.0},
                {"style": "subtask", "content": "", "timestamp": 1.0},
            ]
        }
    }
    with caplog.at_level("WARNING"):
        assert _from_lerobot_annotations(_FakeRecord(frame_timestamps=_ts(21)), episodes) == []
    assert "malformed meta/lerobot_annotations.json entry" in caplog.text


def test_load_lerobot_annotations_handles_missing_and_malformed_files(tmp_path: Path, caplog) -> None:
    assert _load_lerobot_annotations(tmp_path) is None

    path = _write_lerobot_annotations(tmp_path, {"version": 1, "episodes": {"9": {"subtasks": []}}})
    assert _load_lerobot_annotations(tmp_path) == {"9": {"subtasks": []}}

    path.write_text("{not valid json", encoding="utf-8")
    with caplog.at_level("WARNING"):
        assert _load_lerobot_annotations(tmp_path) is None
    assert "could not read" in caplog.text


def test_from_language_persistent_reads_previous_annotate_output() -> None:
    """``style="subtask"`` rows become spans; other styles are ignored and the
    last span is extended to the episode end."""
    rows = [
        {"role": "assistant", "content": "pick up the cup", "style": "subtask", "timestamp": 0.0},
        {"role": "assistant", "content": "1. pick up the cup", "style": "plan", "timestamp": 0.0},
        {"role": "assistant", "content": "put it down", "style": "subtask", "timestamp": 1.0},
    ]
    spans = _from_language_persistent(_FakeRecord(frame_timestamps=_ts(21), persistent=rows))
    assert [s["text"] for s in spans] == ["pick up the cup", "put it down"]
    assert spans[0]["end"] == 1.0
    assert spans[-1]["end"] == 2.0


def test_from_language_persistent_empty_when_column_missing_or_null() -> None:
    assert _from_language_persistent(_FakeRecord(frame_timestamps=_ts(5), persistent=None)) == []
    assert _from_language_persistent(_FakeRecord(frame_timestamps=_ts(5), persistent=[])) == []


def test_from_task_index_segments_runs_of_the_per_frame_task() -> None:
    record = _FakeRecord(
        frame_timestamps=_ts(6),
        frame_tasks=("reach", "reach", "grasp", "grasp", "lift", "lift"),
    )
    assert _from_task_index(record) == [
        {"text": "reach", "start": 0.0, "end": 0.2},
        {"text": "grasp", "start": 0.2, "end": 0.4},
        {"text": "lift", "start": 0.4, "end": 0.5},
    ]


def test_from_task_index_ignores_single_task_episodes() -> None:
    """One run means the 'segmentation' is just the episode task — no
    information, so the caller should fall through rather than emit it."""
    record = _FakeRecord(frame_timestamps=_ts(4), frame_tasks=("tidy up",) * 4)
    assert _from_task_index(record) == []


# ----------------------------------------------------------------------
# Resolution order + module wiring
# ----------------------------------------------------------------------


def test_importer_rejects_unknown_source() -> None:
    with pytest.raises(ValueError, match="not one of"):
        SubtaskImporter(root=None, source="nonsense")


def test_importer_auto_resolution_order() -> None:
    assert IMPORT_SOURCES == (SARM, LEROBOT_ANNOTATIONS, LANGUAGE, TASK_INDEX)
    assert SubtaskImporter(root=None, source="auto").tried == IMPORT_SOURCES


def test_reader_exposes_per_frame_tasks(tmp_path: Path) -> None:
    """An episode whose ``task_index`` changes mid-way keeps every frame's task
    instead of collapsing to the first one."""
    from tests.fixtures import build_annotation_dataset

    root = build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[(0, 4, "tidy the kitchen")],
        fps=10,
        episode_frame_tasks={0: ["open the bin", "open the bin", "drop the can", "drop the can"]},
    )
    record = next(iter_episodes(root))
    assert record.frame_tasks == ("open the bin", "open the bin", "drop the can", "drop the can")
    assert record.episode_task == "open the bin"


def _run_import(root: Path, tmp_path: Path, **config_overrides: Any):
    module = PlanSubtasksMemoryModule(
        vlm=_exploding_vlm(),
        config=PlanConfig(
            n_task_rephrasings=0,
            emit_plan=False,
            emit_memory=False,
            derive_task_from_video="off",
            **config_overrides,
        ),
        root=root,
    )
    record = next(iter_episodes(root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    return record, sorted(
        (r for r in staging.read("plan") if r["style"] == "subtask"),
        key=lambda r: r["timestamp"],
    )


def test_import_sarm_end_to_end_without_any_vlm_call(tmp_path: Path) -> None:
    """SARM columns on ``meta/episodes`` become subtask rows, stitched to a
    full-episode cover, with the VLM never consulted."""
    from tests.fixtures import build_annotation_dataset

    root = build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[(0, 21, "Pour water from the bottle into the cup.")],
        fps=10,
        # Deliberately starts after t0 and stops before the end: the stitch
        # must close both gaps.
        episode_subtasks={0: [("pick up the bottle", 0.5, 1.0), ("pour the water", 1.2, 1.7)]},
    )
    _record, rows = _run_import(root, tmp_path, subtask_import="sarm")

    assert [r["content"] for r in rows] == ["pick up the bottle", "pour the water"]
    assert rows[0]["timestamp"] == 0.0  # pulled back to t0
    assert rows[1]["timestamp"] == 1.2


def test_import_auto_uses_lerobot_annotations_without_a_vlm_call(tmp_path: Path) -> None:
    from tests.fixtures import build_annotation_dataset

    root = build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[(0, 21, "tidy the kitchen")],
        fps=10,
        episode_frame_tasks={0: ["task-index fallback a"] * 10 + ["task-index fallback b"] * 11},
    )
    _write_lerobot_annotations(
        root,
        {
            "version": 1,
            "episodes": {
                "0": {
                    "subtasks": [
                        {"label": "reach the drawer", "start": 0.2, "end": 0.8},
                        {"label": "open the drawer", "start": 0.8, "end": 2.04},
                    ]
                }
            },
        },
    )

    _record, rows = _run_import(root, tmp_path, subtask_import="auto")
    assert [row["content"] for row in rows] == ["reach the drawer", "open the drawer"]
    assert [row["timestamp"] for row in rows] == [0.0, 0.8]


def test_import_explicit_lerobot_annotations_supports_atoms_without_a_vlm_call(tmp_path: Path) -> None:
    from tests.fixtures import build_annotation_dataset

    root = build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[(0, 21, "tidy the kitchen")],
        fps=10,
    )
    _write_lerobot_annotations(
        root,
        {
            "version": 2,
            "episodes": {
                "0": {
                    "subtasks": [],
                    "atoms": [
                        {"style": "plan", "content": "1. reach", "timestamp": 0.0},
                        {"style": "subtask", "content": "put it down", "timestamp": 1.0},
                        {"style": "subtask", "content": "reach", "timestamp": 0.2},
                    ],
                }
            },
        },
    )

    _record, rows = _run_import(root, tmp_path, subtask_import=LEROBOT_ANNOTATIONS)
    assert [row["content"] for row in rows] == ["reach", "put it down"]
    assert [row["timestamp"] for row in rows] == [0.0, 1.0]


def test_import_auto_prefers_sarm_over_lerobot_annotations(tmp_path: Path) -> None:
    from tests.fixtures import build_annotation_dataset

    root = build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[(0, 21, "tidy the kitchen")],
        fps=10,
        episode_subtasks={0: [("SARM wins", 0.0, 2.0)]},
    )
    _write_lerobot_annotations(
        root,
        {"episodes": {"0": {"subtasks": [{"label": "JSON loses", "start": 0.0, "end": 2.04}]}}},
    )

    _record, rows = _run_import(root, tmp_path, subtask_import="auto")
    assert [row["content"] for row in rows] == ["SARM wins"]


def test_import_auto_falls_back_to_task_index(tmp_path: Path) -> None:
    """A JSON file without this episode does not block later auto sources."""

    from tests.fixtures import build_annotation_dataset

    root = build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[(0, 4, "tidy the kitchen")],
        fps=10,
        episode_frame_tasks={0: ["open the bin", "open the bin", "drop the can", "drop the can"]},
    )
    _write_lerobot_annotations(
        root,
        {"episodes": {"9": {"subtasks": [{"label": "other", "start": 0.0, "end": 1.0}]}}},
    )

    _record, rows = _run_import(root, tmp_path, subtask_import="auto")
    assert [r["content"] for r in rows] == ["open the bin", "drop the can"]


def test_import_auto_falls_back_to_generation_when_nothing_recorded(
    fixture_dataset_root: Path, tmp_path: Path, caplog
) -> None:
    """A dataset that records nothing importable still gets annotated — the
    VLM generates, and the fallback is warned about rather than silent."""
    from tests._helpers import SyntheticFrameProvider

    generated = {
        "subtasks": [
            {"text": "grasp the sponge", "start": 0.0, "end": 0.5},
            {"text": "wipe the counter", "start": 0.5, "end": 1.1},
        ]
    }
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda messages: generated),
        config=PlanConfig(
            subtask_import="auto",
            n_task_rephrasings=0,
            emit_plan=False,
            emit_memory=False,
            derive_task_from_video="off",
        ),
        root=fixture_dataset_root,
        frame_provider=SyntheticFrameProvider(),
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    with caplog.at_level("WARNING"):
        module.run_episode(record, staging)

    contents = [r["content"] for r in staging.read("plan") if r["style"] == "subtask"]
    assert contents == ["grasp the sponge", "wipe the counter"]
    assert "generating with the VLM instead" in caplog.text


def test_import_explicit_source_does_not_fall_back_to_generation(
    fixture_dataset_root: Path, tmp_path: Path, caplog
) -> None:
    """Naming a source means that source or nothing — never a surprise GPU
    bill for generated labels the user didn't ask for."""
    module = PlanSubtasksMemoryModule(
        vlm=_exploding_vlm(),
        config=PlanConfig(
            subtask_import="sarm",
            n_task_rephrasings=0,
            emit_plan=False,
            emit_memory=False,
            derive_task_from_video="off",
        ),
        root=fixture_dataset_root,
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    with caplog.at_level("WARNING"):
        module.run_episode(record, staging)

    assert [r for r in staging.read("plan") if r["style"] == "subtask"] == []
    assert "found nothing to import" in caplog.text


def test_import_and_align_are_mutually_exclusive(tmp_path: Path) -> None:
    path = tmp_path / "subtasks.json"
    path.write_text('["a", "b"]', encoding="utf-8")
    with pytest.raises(ValueError, match="mutually exclusive"):
        PlanSubtasksMemoryModule(
            vlm=_exploding_vlm(),
            config=PlanConfig(subtasks_path=path, subtask_import="auto"),
        )
