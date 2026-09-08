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
"""Enabled-module completeness must be checked before parquet mutation."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot_align.config import (  # noqa: E402
    AnnotationPipelineConfig,
    InterjectionsConfig,
    PlanConfig,
    TaskAugAxesConfig,
    VqaConfig,
)
from lerobot_align.executor import Executor  # noqa: E402
from lerobot_align.frames import null_provider  # noqa: E402
from lerobot_align.modules import (  # noqa: E402
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot_align.reader import iter_episodes  # noqa: E402
from lerobot_align.staging import EpisodeStaging  # noqa: E402
from lerobot_align.validator import StagingValidator  # noqa: E402
from lerobot_align.vlm_client import StubVlmClient  # noqa: E402
from lerobot_align.writer import LanguageColumnsWriter, speech_atom  # noqa: E402

from ._helpers import SyntheticFrameProvider  # noqa: E402


def _persistent(style: str, timestamp: float) -> dict:
    return {
        "role": "assistant",
        "content": f"{style} at {timestamp}",
        "style": style,
        "timestamp": timestamp,
        "tool_calls": None,
    }


def _task_aug(content: str, timestamp: float = 0.0) -> dict:
    return {
        "role": "user",
        "content": content,
        "style": "task_aug",
        "timestamp": timestamp,
        "tool_calls": None,
    }


def _prompt_text(messages: list[dict]) -> str:
    return "\n".join(
        str(block.get("text", ""))
        for message in messages
        for block in message.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    )


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_task_rephrasing_count_must_be_non_negative_integer(value: object) -> None:
    with pytest.raises(ValueError, match="n_task_rephrasings"):
        PlanConfig(n_task_rephrasings=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_task_axis_counts_must_be_non_negative_integers(value: object) -> None:
    with pytest.raises(ValueError, match="task_aug_axes.synonym_paraphrase"):
        TaskAugAxesConfig(synonym_paraphrase=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_seeded_relabel_frame_count_must_be_positive_integer(value: object) -> None:
    with pytest.raises(ValueError, match="subtask_relabel_frames"):
        PlanConfig(subtask_relabel_frames=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_interjection_minimum_must_be_non_negative_integer(value: object) -> None:
    with pytest.raises(ValueError, match="min_speech_atoms_per_episode"):
        InterjectionsConfig(min_speech_atoms_per_episode=value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_vqa_minimum_must_be_non_negative_integer(value: object) -> None:
    with pytest.raises(ValueError, match="min_pairs_per_episode"):
        VqaConfig(min_pairs_per_episode=value)  # type: ignore[arg-type]


def test_plan_requires_outputs_at_every_deterministic_boundary(
    single_episode_root: Path,
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / "stage"
    staging = EpisodeStaging(staging_dir, 0)
    staging.write(
        "plan",
        [
            _persistent("subtask", 0.0),
            _persistent("subtask", 0.5),
            _persistent("plan", 0.0),
        ],
    )
    config = AnnotationPipelineConfig(
        plan=PlanConfig(n_task_rephrasings=0, emit_plan=True, emit_memory=True),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )

    report = StagingValidator().validate(
        list(iter_episodes(single_episode_root)),
        staging_dir,
        config=config,
    )

    assert not report.ok
    assert any("missed plan output" in error and "0.5" in error for error in report.errors)
    assert any("missed memory output" in error and "0.5" in error for error in report.errors)


def test_complete_deterministic_plan_outputs_pass(
    single_episode_root: Path,
    tmp_path: Path,
) -> None:
    staging_dir = tmp_path / "stage"
    EpisodeStaging(staging_dir, 0).write(
        "plan",
        [
            _persistent("subtask", 0.0),
            _persistent("subtask", 0.5),
            _persistent("plan", 0.0),
            _persistent("plan", 0.5),
            _persistent("memory", 0.5),
        ],
    )
    config = AnnotationPipelineConfig(
        plan=PlanConfig(n_task_rephrasings=0, emit_plan=True, emit_memory=True),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )

    report = StagingValidator().validate(
        list(iter_episodes(single_episode_root)),
        staging_dir,
        config=config,
    )

    assert report.ok, report.errors


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_free_form_task_augmentation_warns_on_shortfall_but_enforces_presence_and_cap(
    single_episode_root: Path,
    tmp_path: Path,
    count: int,
) -> None:
    records = list(iter_episodes(single_episode_root))
    staging_dir = tmp_path / "stage"
    staging = EpisodeStaging(staging_dir, 0)
    config = AnnotationPipelineConfig(
        plan=PlanConfig(n_task_rephrasings=2, emit_plan=True, emit_memory=False),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )
    validator = StagingValidator()

    staging.write(
        "plan",
        [
            _persistent("subtask", 0.0),
            _persistent("plan", 0.0),
            *[_task_aug(f"task {i}") for i in range(count)],
        ],
    )
    report = validator.validate(records, staging_dir, config=config)
    if count == 0 or count > 3:
        assert not report.ok
        assert len(report.completeness_errors) == 1
        assert "task_aug" in report.completeness_errors[0]
        assert report.episode_completeness_errors == {0: report.completeness_errors}
        assert not report.warnings
    else:
        assert report.ok, report.errors
        assert not report.completeness_errors
        assert not report.episode_errors
        assert not report.episode_completeness_errors
        if count < 3:
            assert len(report.warnings) == 1
            assert f"ep=0: plan module emitted {count} task_aug row(s)" in report.warnings[0]
            assert "below requested target 3" in report.warnings[0]
        else:
            assert not report.warnings


def test_disabled_task_augmentation_does_not_require_rows(single_episode_root, tmp_path):
    config = AnnotationPipelineConfig(
        plan=PlanConfig(n_task_rephrasings=0, emit_plan=True, emit_memory=False),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )
    staging_dir = tmp_path / "stage"
    EpisodeStaging(staging_dir, 0).write(
        "plan", [_persistent("subtask", 0.0), _persistent("plan", 0.0)]
    )
    report = StagingValidator().validate(
        list(iter_episodes(single_episode_root)), staging_dir, config=config
    )
    assert report.ok, report.errors
    assert not report.warnings


def test_task_augmentation_warning_does_not_hide_missing_subtasks(single_episode_root, tmp_path):
    config = AnnotationPipelineConfig(
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )
    staging_dir = tmp_path / "stage"
    staging = EpisodeStaging(staging_dir, 0)
    staging.write(
        "plan",
        [_persistent("plan", 0.0), _task_aug("canonical"), _task_aug("variant one")],
    )
    report = StagingValidator().validate(
        list(iter_episodes(single_episode_root)), staging_dir, config=config
    )
    assert not report.ok
    assert report.completeness_errors == [
        "ep=0: plan module is enabled but emitted no subtask rows"
    ]
    assert len(report.warnings) == 1


def test_structured_task_augmentation_requires_canonical_and_synonym_targets(
    single_episode_root: Path,
    tmp_path: Path,
) -> None:
    records = list(iter_episodes(single_episode_root))
    staging_dir = tmp_path / "stage"
    staging = EpisodeStaging(staging_dir, 0)
    axes = TaskAugAxesConfig(
        enabled=True,
        synonym_paraphrase=2,
        omit_arm=2,
        omit_orientation=2,
        omit_grasp_method=2,
        combined_omissions=2,
    )
    config = AnnotationPipelineConfig(
        plan=PlanConfig(task_aug_axes=axes, emit_plan=False, emit_memory=False),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )

    # Optional omission axes may be empty, but the canonical task and both
    # unconditional synonym targets are required.
    staging.write(
        "plan",
        [_persistent("subtask", 0.0), _task_aug("canonical"), _task_aug("one synonym")],
    )
    incomplete = StagingValidator().validate(records, staging_dir, config=config)
    assert not incomplete.ok
    assert any("structured augmentation minimum 3" in error for error in incomplete.errors)

    staging.write(
        "plan",
        [
            _persistent("subtask", 0.0),
            _task_aug("canonical"),
            _task_aug("one synonym"),
            _task_aug("another synonym"),
        ],
    )
    complete = StagingValidator().validate(records, staging_dir, config=config)
    assert complete.ok, complete.errors


def test_sparse_modules_use_configurable_minima(
    single_episode_root: Path,
    tmp_path: Path,
) -> None:
    records = list(iter_episodes(single_episode_root))
    staging_dir = tmp_path / "stage"
    config = AnnotationPipelineConfig(
        plan=PlanConfig(enabled=False),
        interjections=InterjectionsConfig(min_speech_atoms_per_episode=1),
        vqa=VqaConfig(min_pairs_per_episode=1),
    )
    validator = StagingValidator(dataset_camera_keys=("observation.images.front",))

    empty_report = validator.validate(records, staging_dir, config=config)
    assert not empty_report.ok
    assert any("speech atom(s)" in error for error in empty_report.errors)
    assert any("complete pair(s)" in error for error in empty_report.errors)

    staging = EpisodeStaging(staging_dir, 0)
    staging.write("interjections", [speech_atom(0.0, "Ready.")])
    staging.write(
        "vqa",
        [
            {
                "role": "user",
                "content": "How many cups?",
                "style": "vqa",
                "timestamp": 0.0,
                "camera": "observation.images.front",
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": json.dumps({"label": "cup", "count": 1}),
                "style": "vqa",
                "timestamp": 0.0,
                "camera": "observation.images.front",
                "tool_calls": None,
            },
        ],
    )

    valid_report = validator.validate(records, staging_dir, config=config)
    assert valid_report.ok, valid_report.errors


@pytest.mark.parametrize("skip_validation", [False, True])
def test_all_invalid_plan_replies_fail_before_parquet_write(
    single_episode_root: Path,
    skip_validation: bool,
) -> None:
    invalid_vlm = StubVlmClient(responder=lambda _messages: None)
    config = AnnotationPipelineConfig(
        plan=PlanConfig(
            n_task_rephrasings=0,
            subtask_describe_first=False,
            emit_plan=False,
            emit_memory=False,
        ),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
        skip_validation=skip_validation,
    )
    executor = Executor(
        config=config,
        plan=PlanSubtasksMemoryModule(
            vlm=invalid_vlm,
            config=config.plan,
            frame_provider=SyntheticFrameProvider(),
            root=single_episode_root,
        ),
        interjections=InterjectionsAndSpeechModule(
            vlm=invalid_vlm,
            config=config.interjections,
            frame_provider=null_provider(),
        ),
        vqa=GeneralVqaModule(
            vlm=invalid_vlm,
            config=config.vqa,
            frame_provider=null_provider(),
        ),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
    )
    shard = next((single_episode_root / "data").rglob("*.parquet"))
    info_path = single_episode_root / "meta" / "info.json"
    before_shard = shard.read_bytes()
    before_info = info_path.read_bytes()

    with pytest.raises(RuntimeError, match="Staging validation failed"):
        executor.run(single_episode_root)

    assert shard.read_bytes() == before_shard
    assert info_path.read_bytes() == before_info


@pytest.mark.parametrize("skip_validation", [False, True])
def test_canonical_only_task_augmentation_warns_and_publishes(
    single_episode_root: Path,
    skip_validation: bool,
) -> None:
    def responder(messages: list[dict]) -> object:
        if "COMPLETED manipulation events" in _prompt_text(messages):
            return {"subtasks": [{"text": "pick up the cup", "start": 0.0, "end": 0.9}]}
        return None

    vlm = StubVlmClient(responder=responder)
    config = AnnotationPipelineConfig(
        plan=PlanConfig(
            n_task_rephrasings=2,
            subtask_describe_first=False,
            emit_plan=True,
            emit_memory=False,
        ),
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
        skip_validation=skip_validation,
    )
    executor = Executor(
        config=config,
        plan=PlanSubtasksMemoryModule(
            vlm=vlm,
            config=config.plan,
            frame_provider=SyntheticFrameProvider(),
            root=single_episode_root,
        ),
        interjections=InterjectionsAndSpeechModule(
            vlm=vlm,
            config=config.interjections,
            frame_provider=null_provider(),
        ),
        vqa=GeneralVqaModule(
            vlm=vlm,
            config=config.vqa,
            frame_provider=null_provider(),
        ),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
    )
    shard = next((single_episode_root / "data").rglob("*.parquet"))
    summary = executor.run(single_episode_root)
    assert summary.validation_report.ok
    assert not summary.skipped_episodes
    assert len(summary.validation_report.warnings) == 1
    assert "emitted 1 task_aug row(s), below requested target 3" in (
        summary.validation_report.warnings[0]
    )
    rows = pq.read_table(shard).to_pylist()[0]["language_persistent"]
    assert [row["content"] for row in rows if row["style"] == "subtask"] == ["pick up the cup"]
    assert [row["content"] for row in rows if row["style"] == "task_aug"] == [
        "Pour water from the bottle into the cup."
    ]


def test_missing_interjection_plan_state_cannot_skip_validation(
    single_episode_root: Path,
) -> None:
    class MissingPlanStateModule:
        enabled = True

        @staticmethod
        def run_episode(record: object, staging: EpisodeStaging) -> None:
            timestamps = record.frame_timestamps  # type: ignore[attr-defined]
            staging.write(
                "plan",
                [
                    _persistent("subtask", float(timestamps[0])),
                    _persistent("subtask", float(timestamps[1])),
                    _persistent("plan", float(timestamps[0])),
                ],
            )

        @staticmethod
        def run_plan_updates(
            _record: object,
            _staging: EpisodeStaging,
            _interjection_times: list[float],
        ) -> None:
            # Simulate a broken/no-op post-pass: validation must still stop the
            # write even when ordinary structural validation is skipped.
            return None

    class InterjectionAtMissingPlanModule:
        enabled = True

        @staticmethod
        def run_episode(record: object, staging: EpisodeStaging) -> None:
            timestamps = record.frame_timestamps  # type: ignore[attr-defined]
            boundary = float(timestamps[1])
            staging.write(
                "interjections",
                [
                    speech_atom(float(timestamps[0]), "Ready."),
                    {
                        "role": "user",
                        "content": "Now pick it up.",
                        "style": "interjection",
                        "timestamp": boundary,
                        "tool_calls": None,
                    },
                    speech_atom(boundary, "On it."),
                ],
            )

    config = AnnotationPipelineConfig(
        plan=PlanConfig(n_task_rephrasings=0, emit_plan=True, emit_memory=False),
        interjections=InterjectionsConfig(min_speech_atoms_per_episode=1),
        vqa=VqaConfig(enabled=False),
        skip_validation=True,
    )
    validator = StagingValidator()
    inert_vlm = StubVlmClient(responder=lambda _messages: None)
    executor = Executor(
        config=config,
        plan=MissingPlanStateModule(),  # type: ignore[arg-type]
        interjections=InterjectionAtMissingPlanModule(),  # type: ignore[arg-type]
        vqa=GeneralVqaModule(
            vlm=inert_vlm,
            config=config.vqa,
            frame_provider=null_provider(),
        ),
        writer=LanguageColumnsWriter(),
        validator=validator,
    )
    shard = next((single_episode_root / "data").rglob("*.parquet"))
    info_path = single_episode_root / "meta" / "info.json"
    before_shard = shard.read_bytes()
    before_info = info_path.read_bytes()

    with pytest.raises(RuntimeError, match="Staging validation failed"):
        executor.run(single_episode_root)

    report = validator.validate(
        list(iter_episodes(single_episode_root)),
        config.resolved_staging_dir(single_episode_root),
        config=config,
    )
    assert any("co-timestamped plan state" in error for error in report.completeness_errors)
    assert shard.read_bytes() == before_shard
    assert info_path.read_bytes() == before_info


@pytest.mark.parametrize(
    "reply",
    [
        None,
        {"synonym_paraphrase": []},
        {
            "synonym_paraphrase": [],
            "omit_arm": [],
            "omit_orientation": [],
            "omit_grasp_method": [],
            "combined_omissions": [],
        },
    ],
)
def test_structured_task_augmentation_rejects_invalid_or_incomplete_reply(
    single_episode_root: Path,
    tmp_path: Path,
    reply: object,
) -> None:
    axes = TaskAugAxesConfig(
        enabled=True,
        synonym_paraphrase=1,
        omit_arm=0,
        omit_orientation=0,
        omit_grasp_method=0,
        combined_omissions=0,
    )
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: reply),
        config=PlanConfig(
            task_aug_axes=axes,
            subtask_describe_first=False,
            emit_plan=False,
            emit_memory=False,
        ),
        frame_provider=SyntheticFrameProvider(),
    )
    record = next(iter_episodes(single_episode_root))

    with pytest.raises(RuntimeError, match="Structured task augmentation"):
        module.run_episode(record, EpisodeStaging(tmp_path / "stage", record.episode_index))


@pytest.mark.parametrize("relabel_reply", [None, {"label": ""}, {"label": 7}])
def test_seeded_relabel_rejects_missing_label(
    single_episode_root: Path,
    tmp_path: Path,
    relabel_reply: object,
) -> None:
    def responder(messages: list[dict]) -> object:
        prompt = _prompt_text(messages)
        if "COMPLETED manipulation events" in prompt:
            return {"subtasks": [{"text": "pick up the cup", "start": 0.0, "end": 0.9}]}
        if "Annotate one fixed segment" in prompt:
            return relabel_reply
        return None

    config = PlanConfig(
        n_task_rephrasings=0,
        subtask_describe_first=False,
        subtask_seeded_relabel=True,
        emit_plan=False,
        emit_memory=False,
    )
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=config,
        frame_provider=SyntheticFrameProvider(),
    )
    record = next(iter_episodes(single_episode_root))

    with pytest.raises(RuntimeError, match="seeded relabel returned no valid label"):
        module.run_episode(record, EpisodeStaging(tmp_path / "stage", record.episode_index))


def test_seeded_relabel_accepts_valid_label(
    single_episode_root: Path,
    tmp_path: Path,
) -> None:
    def responder(messages: list[dict]) -> object:
        prompt = _prompt_text(messages)
        if "COMPLETED manipulation events" in prompt:
            return {"subtasks": [{"text": "pick up", "start": 0.0, "end": 0.9}]}
        if "Annotate one fixed segment" in prompt:
            return {"label": "pick up the cup"}
        return None

    config = PlanConfig(
        n_task_rephrasings=0,
        subtask_describe_first=False,
        subtask_seeded_relabel=True,
        emit_plan=False,
        emit_memory=False,
    )
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=config,
        frame_provider=SyntheticFrameProvider(),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)

    module.run_episode(record, staging)

    subtasks = [row for row in staging.read("plan") if row.get("style") == "subtask"]
    assert [row["content"] for row in subtasks] == ["pick up the cup"]
