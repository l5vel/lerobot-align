"""Regression cases from the release data-integrity audit."""

from dataclasses import fields
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_align.config import (
    AnnotationPipelineConfig,
    ExecutorConfig,
    InterjectionsConfig,
    PlanConfig,
    VlmConfig,
    VqaConfig,
)
from lerobot_align.frames import FrameProviderError, make_frame_provider, null_provider
from lerobot_align.modules import GeneralVqaModule, PlanSubtasksMemoryModule
from lerobot_align.modules.plan_subtasks_memory import _parse_align_spans
from lerobot_align.reader import iter_episodes
from lerobot_align.staging import EpisodeStaging
from lerobot_align.vlm_client import StubVlmClient
from tests._helpers import SyntheticFrameProvider


@pytest.mark.parametrize(
    "config_type",
    [
        PlanConfig,
        InterjectionsConfig,
        VqaConfig,
        VlmConfig,
        ExecutorConfig,
        AnnotationPipelineConfig,
    ],
)
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), "bad", True])
def test_all_numeric_configuration_fields_reject_invalid_values(config_type, bad):
    defaults = config_type()
    for definition in fields(defaults):
        value = getattr(defaults, definition.name)
        if type(value) in (int, float):
            with pytest.raises(ValueError, match=definition.name):
                config_type(**{definition.name: bad})


@pytest.mark.parametrize(
    "value", [float("nan"), float("inf"), float("-inf"), "not a time", True, None]
)
def test_generated_timestamps_are_rejected_before_staging(single_episode_root, tmp_path, value):
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(
            lambda messages: {"subtasks": [{"text": "pick up the cup", "start": value, "end": 0.9}]}
        ),
        config=PlanConfig(n_task_rephrasings=0, subtask_describe_first=False),
        frame_provider=SyntheticFrameProvider(),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "staging", record.episode_index)
    before = record.data_path.read_bytes()
    with pytest.raises(ValueError, match="finite start/end"):
        module.run_episode(record, staging)
    assert staging.read("plan") == []
    assert record.data_path.read_bytes() == before


@pytest.mark.parametrize("value", [5, "bad", {}, [None], ["bad"], [5]])
def test_invalid_subtask_structures_are_rejected(single_episode_root, value):
    module = PlanSubtasksMemoryModule(vlm=StubVlmClient(lambda _: None), config=PlanConfig())
    with pytest.raises(ValueError, match="subtask"):
        module._clean_spans(value, next(iter_episodes(single_episode_root)))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "bad", True])
def test_alignment_rejects_invalid_timestamps_before_clamping(value):
    with pytest.raises(ValueError, match="finite start/end"):
        _parse_align_spans({"subtasks": [{"index": 0, "start": value, "end": 1}]}, 1)


@pytest.mark.parametrize(
    "values", [[0, float("nan")], [0, float("inf")], [0, 0], [1, 0], ["0", "bad"]]
)
def test_source_timestamps_rejected_before_inference(single_episode_root, values):
    path = next((single_episode_root / "data").rglob("*.parquet"))
    table = pq.read_table(path).slice(0, 2)
    table = table.set_column(
        table.schema.get_field_index("timestamp"), "timestamp", pa.array(values)
    )
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="timestamps must be"):
        list(iter_episodes(single_episode_root))


@pytest.mark.parametrize("format", ["contact_sheet", "video"])
@pytest.mark.parametrize("failed", [False, True])
def test_visual_generation_never_calls_vlm_without_images(
    single_episode_root, tmp_path, format, failed
):
    calls = []
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(lambda messages: calls.append(messages)),
        config=PlanConfig(subtask_generate_frame_format=format),
        frame_provider=null_provider(
            initialization_error=OSError("broken video") if failed else None
        ),
    )
    with pytest.raises(RuntimeError):
        module.run_episode(next(iter_episodes(single_episode_root)), EpisodeStaging(tmp_path, 0))
    assert calls == []


def test_cameraless_metadata_does_not_initialize_decoder(single_episode_root, monkeypatch):
    monkeypatch.setattr(
        "lerobot_align.frames.VideoFrameProvider", lambda **_: pytest.fail("decoder called")
    )
    assert make_frame_provider(single_episode_root).camera_keys == []


def test_declared_image_camera_is_not_treated_as_cameraless(single_episode_root, monkeypatch):
    path = single_episode_root / "meta/info.json"
    info = json.loads(path.read_text())
    info["features"]["observation.images.top"] = {"dtype": "image"}
    path.write_text(json.dumps(info))

    def fail(**kwargs):
        raise OSError("missing visual metadata")

    monkeypatch.setattr("lerobot_align.frames.VideoFrameProvider", fail)
    with pytest.raises(FrameProviderError, match="missing visual metadata"):
        make_frame_provider(single_episode_root)


def test_vqa_failed_provider_is_not_camera_less_exemption(single_episode_root, tmp_path):
    module = GeneralVqaModule(
        vlm=StubVlmClient(lambda _: pytest.fail("VLM called")),
        config=VqaConfig(),
        frame_provider=null_provider(initialization_error=OSError("broken video")),
    )
    with pytest.raises(FrameProviderError):
        module.run_episode(next(iter_episodes(single_episode_root)), EpisodeStaging(tmp_path, 0))


def test_vqa_empty_decode_cannot_produce_annotations(single_episode_root, tmp_path):
    class EmptyProvider(SyntheticFrameProvider):
        def frames_at(self, *args, **kwargs):
            return []

    module = GeneralVqaModule(
        vlm=StubVlmClient(lambda _: pytest.fail("VLM called")),
        config=VqaConfig(),
        frame_provider=EmptyProvider(),
    )
    with pytest.raises(RuntimeError, match="no VQA image"):
        module.run_episode(next(iter_episodes(single_episode_root)), EpisodeStaging(tmp_path, 0))


@pytest.mark.parametrize('timestamp', ['1:99', '1:2:3:4', 'a:01', '1:60:00'])
def test_malformed_clock_timestamps_are_rejected(timestamp):
    with pytest.raises(ValueError, match='finite start/end'):
        _parse_align_spans({'subtasks': [{'start': timestamp, 'end': 9}]}, 1)


@pytest.mark.parametrize('index', [True, 0.5, float('nan'), float('inf'), 'first'])
def test_invalid_alignment_indices_are_rejected(index):
    with pytest.raises(ValueError, match='integer index'):
        _parse_align_spans({'subtasks': [{'index': index, 'start': 0, 'end': 9}]}, 1)


def test_partial_frame_set_never_reaches_model(single_episode_root, tmp_path):
    class PartialProvider(SyntheticFrameProvider):
        def frames_at(self, record, timestamps, *args, **kwargs):
            return super().frames_at(record, timestamps[:1])
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(lambda _: pytest.fail('incomplete visual request reached model')),
        config=PlanConfig(), frame_provider=PartialProvider())
    with pytest.raises(RuntimeError, match='incomplete visual input'):
        module.run_episode(next(iter_episodes(single_episode_root)), EpisodeStaging(tmp_path, 0))


@pytest.mark.parametrize('labels', [[], {'9': ['pick up cup']}])
def test_missing_fixed_labels_do_not_start_generation(single_episode_root, tmp_path, labels):
    from lerobot_align.executor import prepare_records_for_run
    from lerobot_align.writer import LanguageColumnsWriter
    path = tmp_path / 'labels.json'
    path.write_text(json.dumps(labels))
    config = AnnotationPipelineConfig(plan=PlanConfig(subtasks_path=path))
    with pytest.raises(ValueError, match='non-empty list|No supplied subtasks'):
        prepare_records_for_run(single_episode_root, config, LanguageColumnsWriter())
