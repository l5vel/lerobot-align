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
"""Configuration and rerun-safety regression tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot_align import executor as executor_module
from lerobot_align.config import (
    AnnotationPipelineConfig,
    InterjectionsConfig,
    PlanConfig,
    VqaConfig,
)
from lerobot_align.executor import Executor
from lerobot_align.reader import iter_episodes
from lerobot_align.staging import EpisodeStaging
from lerobot_align.writer import LanguageColumnsWriter, dataset_run_lock


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_plan_config_rejects_invalid_frame_budget(value: object) -> None:
    with pytest.raises(ValueError, match="max_frames_per_prompt"):
        PlanConfig(max_frames_per_prompt=value)  # type: ignore[arg-type]


def test_plan_config_rejects_invalid_generation_frame_format() -> None:
    with pytest.raises(ValueError, match="subtask_generate_frame_format"):
        PlanConfig(subtask_generate_frame_format="frames")  # type: ignore[arg-type]


def test_plan_config_rejects_invalid_alignment_frame_format() -> None:
    with pytest.raises(ValueError, match="subtask_align_frame_format"):
        PlanConfig(subtask_align_frame_format="frames")  # type: ignore[arg-type]


def test_plan_config_rejects_invalid_video_fallback_policy() -> None:
    with pytest.raises(ValueError, match="subtask_video_fallback"):
        PlanConfig(subtask_video_fallback="silent")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "value", [0.0, -1.0, float("nan"), float("inf"), float("-inf"), True, "fast"]
)
def test_vqa_config_rejects_invalid_frequency(value: object) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        VqaConfig(vqa_emission_hz=value)  # type: ignore[arg-type]


def test_vqa_config_rejects_empty_question_types() -> None:
    with pytest.raises(ValueError, match="question_types"):
        VqaConfig(question_types=())


def test_disabled_phase_clears_stale_staging(fixture_dataset_root: Path, tmp_path: Path) -> None:
    records = list(iter_episodes(fixture_dataset_root))
    staging_dir = tmp_path / "stage"
    stale = {
        "role": "user",
        "content": "stale question",
        "style": "vqa",
        "timestamp": 0.0,
        "camera": "observation.images.front",
        "tool_calls": None,
    }
    for record in records:
        EpisodeStaging(staging_dir, record.episode_index).write("vqa", [stale])

    class DisabledModule:
        enabled = False

    executor = Executor(
        config=AnnotationPipelineConfig(),
        plan=None,
        interjections=None,
        vqa=None,
        writer=None,  # type: ignore[arg-type]
        validator=None,  # type: ignore[arg-type]
    )
    result = executor._run_module_phase("vqa", records, staging_dir, DisabledModule())

    assert result.episodes_processed == 0
    assert result.episodes_skipped == len(records)
    for record in records:
        assert EpisodeStaging(staging_dir, record.episode_index).read("vqa") == []


def test_reader_rejects_lance_dataset_explicitly(tmp_path: Path) -> None:
    root = tmp_path / "lance_ds"
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(json.dumps({"storage_format": "lance"}), encoding="utf-8")
    (root / "frames.lance").mkdir()

    with pytest.raises(NotImplementedError, match="Parquet/MP4"):
        list(iter_episodes(root))


@pytest.mark.parametrize("version", ["v3.0", "v3.1"])
def test_reader_accepts_supported_v3_versions(fixture_dataset_root: Path, version: str) -> None:
    info_path = fixture_dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["codebase_version"] = version
    info_path.write_text(json.dumps(info))

    assert list(iter_episodes(fixture_dataset_root))


def test_dataset_run_lock_rejects_overlap_and_releases(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    with (
        dataset_run_lock(root),
        pytest.raises(RuntimeError, match="already annotating"),
        dataset_run_lock(root),
    ):
        pytest.fail("an overlapping run acquired the dataset lease")

    # A completed/failed run releases the advisory lease. The persistent lock
    # file remains so waiters never race by locking different inodes.
    with dataset_run_lock(root):
        pass


def test_executor_run_acquires_dataset_lock(tmp_path: Path) -> None:
    executor = Executor(
        config=AnnotationPipelineConfig(),
        plan=None,
        interjections=None,
        vqa=None,
        writer=LanguageColumnsWriter(),
        validator=None,  # type: ignore[arg-type]
    )

    with dataset_run_lock(tmp_path), pytest.raises(RuntimeError, match="already annotating"):
        executor.run(tmp_path)


def test_atomic_info_failure_preserves_original_metadata(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    info_path = fixture_dataset_root / "meta" / "info.json"
    before = info_path.read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("simulated metadata replace failure")

    monkeypatch.setattr(executor_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated metadata replace failure"):
        Executor._ensure_annotation_metadata_in_info(fixture_dataset_root)

    assert info_path.read_bytes() == before
    assert list(info_path.parent.glob(f".{info_path.name}.*.tmp")) == []


def test_metadata_failure_is_not_reported_as_pipeline_success(
    single_episode_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class DisabledModule:
        enabled = False

    class ValidReport:
        ok = True
        warnings: list[str] = []

        @staticmethod
        def summary() -> str:
            return "ok"

    class Validator:
        @staticmethod
        def validate(
            _records: object,
            _staging_dir: Path,
            *,
            config: AnnotationPipelineConfig | None = None,
        ) -> ValidReport:
            del config
            return ValidReport()

    config = AnnotationPipelineConfig(
        plan=PlanConfig(enabled=False),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )
    executor = Executor(
        config=config,
        plan=DisabledModule(),
        interjections=DisabledModule(),
        vqa=DisabledModule(),
        writer=LanguageColumnsWriter(),
        validator=Validator(),  # type: ignore[arg-type]
    )

    def fail_metadata(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated metadata failure")

    monkeypatch.setattr(executor, "_ensure_annotation_metadata_in_info", fail_metadata)

    with pytest.raises(OSError, match="simulated metadata failure"):
        executor.run(single_episode_root)

    assert "pipeline complete" not in capsys.readouterr().out
