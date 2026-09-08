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
"""Module 1/2/3 unit tests with stubbed VLMs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import PIL.Image
import pytest

# Annotation imports pull in ``lerobot.datasets`` (and therefore the HF
# ``datasets`` library), which only ships under the ``dataset`` extra. Skip
# this module in tiers without it instead of erroring at import.
pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")
pytest.importorskip("pandas", reason="pandas is required (install lerobot[dataset])")

from lerobot_align.config import (  # noqa: E402
    AnnotationPipelineConfig,
    InterjectionsConfig,
    PlanConfig,
    VqaConfig,
)
from lerobot_align.modules import plan_subtasks_memory as plan_module  # noqa: E402
from lerobot_align.modules import (  # noqa: E402
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot_align.frames import null_provider  # noqa: E402
from lerobot_align.reader import iter_episodes  # noqa: E402
from lerobot_align.staging import EpisodeStaging  # noqa: E402
from lerobot_align.validator import StagingValidator  # noqa: E402
from lerobot_align.vlm_client import StubVlmClient  # noqa: E402

from ._helpers import SyntheticFrameProvider, make_canned_responder  # noqa: E402


@dataclass
class _StubFrameProvider:
    """Returns one sentinel object per requested timestamp."""

    # A real (tiny) PIL image so the contact-sheet builder, which resizes and
    # tiles frames, has something to draw. VQA still passes it through by
    # identity via ``to_image_blocks``.
    sentinel: Any = field(default_factory=lambda: PIL.Image.new("RGB", (32, 24)))
    cameras: tuple[str, ...] = ("observation.images.top",)
    failed_cameras: tuple[str, ...] = ()
    calls: list[tuple[int, tuple[float, ...], str | None]] = field(default_factory=list)
    video_calls: list[tuple[int, int, str | None]] = field(default_factory=list)

    @property
    def camera_keys(self) -> list[str]:
        return list(self.cameras)

    def frames_at(self, record, timestamps, camera_key=None, *, fail_on_error=False):
        self.calls.append((record.episode_index, tuple(timestamps), camera_key))
        if camera_key in self.failed_cameras:
            if fail_on_error:
                raise RuntimeError(f"could not decode {camera_key}")
            return []
        return [self.sentinel] * len(timestamps)

    def video_for_episode(self, record, max_frames, camera_key=None):
        self.video_calls.append((record.episode_index, max_frames, camera_key))
        n = min(max_frames, len(record.frame_timestamps))
        return [self.sentinel] * n


def _spy_responder(captured: list[list[dict[str, Any]]], reply: Any):
    def responder(messages):
        captured.append(list(messages))
        return reply

    return StubVlmClient(responder=responder)


def test_module1_plan_memory_subtask_smoke(fixture_dataset_root: Path, tmp_path: Path) -> None:
    vlm = make_canned_responder(
        {
            "COMPLETED manipulation events": {
                "subtasks": [
                    {"text": "grasp the handle of the sponge", "start": 0.0, "end": 0.4},
                    {"text": "wipe the counter from left to right", "start": 0.4, "end": 0.8},
                    {"text": "place the sponge into the sink", "start": 0.8, "end": 1.1},
                ]
            },
            "compressed semantic memory": {"memory": "wiped the counter once"},
        },
    )
    module = PlanSubtasksMemoryModule(vlm=vlm, config=PlanConfig(), frame_provider=SyntheticFrameProvider())
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("plan")

    styles = {r["style"] for r in rows}
    assert {"subtask", "plan", "memory"}.issubset(styles)
    # subtask timestamps must be exact frame timestamps
    frame_set = set(record.frame_timestamps)
    for row in rows:
        assert row["timestamp"] in frame_set
    # one plan row per subtask boundary; the first lands at t0 and each
    # plan is the deterministic numbered list of still-todo subtasks
    plan_rows = sorted((r for r in rows if r["style"] == "plan"), key=lambda r: r["timestamp"])
    subtask_rows = [r for r in rows if r["style"] == "subtask"]
    assert len(plan_rows) == len(subtask_rows)
    assert plan_rows[0]["timestamp"] == record.frame_timestamps[0]
    # the t0 plan enumerates all subtasks; later plans shrink
    assert plan_rows[0]["content"].startswith("1. ")
    assert len(plan_rows[0]["content"].splitlines()) == len(subtask_rows)
    assert len(plan_rows[-1]["content"].splitlines()) == 1


def test_module1_emit_memory_false_skips_memory_keeps_subtasks_and_plan(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """``emit_memory=False`` drops ``memory`` rows (and their VLM calls) while
    leaving subtask + plan generation intact — symmetric to ``emit_plan``."""
    vlm = make_canned_responder(
        {
            "COMPLETED manipulation events": {
                "subtasks": [
                    {"text": "grasp the handle of the sponge", "start": 0.0, "end": 0.4},
                    {"text": "wipe the counter from left to right", "start": 0.4, "end": 0.8},
                    {"text": "place the sponge into the sink", "start": 0.8, "end": 1.1},
                ]
            },
            "compressed semantic memory": {"memory": "wiped the counter once"},
        },
    )
    module = PlanSubtasksMemoryModule(vlm=vlm, config=PlanConfig(emit_memory=False), frame_provider=SyntheticFrameProvider())
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("plan")

    styles = {r["style"] for r in rows}
    assert "memory" not in styles
    assert {"subtask", "plan"}.issubset(styles)


def test_module2_at_t0_emits_speech_only_no_interjection(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    vlm = make_canned_responder(
        {"acknowledgement the robot": {"text": "Sure, on it."}},
    )
    module = InterjectionsAndSpeechModule(
        vlm=vlm,
        config=InterjectionsConfig(max_interjections_per_episode=0),
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("interjections")
    assert len(rows) == 1
    only = rows[0]
    assert only["role"] == "assistant"
    assert only["style"] is None
    assert only["content"] is None
    assert only["timestamp"] == record.frame_timestamps[0]
    assert only["tool_calls"][0]["function"]["name"] == "say"


def test_module2_mid_episode_emits_paired_interjection_and_speech(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """Module 2 anchors interjections on Module 1's subtask boundaries.

    The executor runs Module 1 first, then Module 2 reads the subtask
    rows back from the same staging tree (see
    ``_mid_episode_interjections``). Reproduce that contract here by
    seeding the staging with two subtask rows so a single ``0 → 1``
    boundary exists for Module 2 to anchor on.
    """
    vlm = make_canned_responder(
        {
            "acknowledgement the robot": {"text": "OK."},
            # Marker matches the distinctive line of
            # ``interjections_interjection.txt`` ("Write ONE compact
            # interjection ..."). Keep this in sync with that prompt's
            # wording — the canned responder matches on substring.
            "Write ONE compact interjection": {
                "interjection": "now wipe the counter please",
                "speech": "On it.",
            },
        },
    )
    module = InterjectionsAndSpeechModule(
        vlm=vlm,
        config=InterjectionsConfig(max_interjections_per_episode=1, interjection_min_t=0.2),
        seed=7,
        frame_provider=SyntheticFrameProvider(),
    )
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    # Seed Module 1's subtask staging so Module 2 has a boundary to
    # anchor on (it bails with zero rows when no spans exist — the
    # production executor guarantees Module 1 ran first).
    boundary_ts = float(record.frame_timestamps[len(record.frame_timestamps) // 2])
    staging.write(
        "plan",
        [
            {
                "role": "assistant",
                "content": "grasp the sponge",
                "style": "subtask",
                "timestamp": float(record.frame_timestamps[0]),
                "tool_calls": None,
            },
            {
                "role": "assistant",
                "content": "wipe the counter",
                "style": "subtask",
                "timestamp": boundary_ts,
                "tool_calls": None,
            },
        ],
    )
    module.run_episode(record, staging)
    rows = staging.read("interjections")

    interjections = [r for r in rows if r["style"] == "interjection"]
    speeches = [r for r in rows if r["style"] is None and r["role"] == "assistant"]
    assert len(interjections) == 1
    assert len(speeches) >= 2  # initial t=0 + one paired with the interjection
    inter_t = interjections[0]["timestamp"]
    assert any(abs(s["timestamp"] - inter_t) < 1e-9 for s in speeches)


def test_plan_state_at_generated_interjection_boundary_is_deterministic_noop(
    fixture_dataset_root: Path,
    tmp_path: Path,
) -> None:
    """A generated interjection cues the already-upcoming subtask.

    Its boundary already has the correct remaining-subtasks plan, so the
    post-pass must preserve that row without another VLM call or a divergent
    semantic replan.
    """

    def forbid_replan(_messages: list[dict[str, Any]]) -> Any:
        pytest.fail("co-timestamped boundary plan must not trigger VLM replanning")

    record = next(iter_episodes(fixture_dataset_root))
    boundary = float(record.frame_timestamps[len(record.frame_timestamps) // 2])
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    original = [
        {
            "role": "assistant",
            "content": "reach for the sponge",
            "style": "subtask",
            "timestamp": float(record.frame_timestamps[0]),
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": "wipe the counter",
            "style": "subtask",
            "timestamp": boundary,
            "tool_calls": None,
        },
        {
            "role": "assistant",
            "content": "1. wipe the counter",
            "style": "plan",
            "timestamp": boundary,
            "tool_calls": None,
        },
    ]
    staging.write("plan", original)
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=forbid_replan),
        config=PlanConfig(n_task_rephrasings=0),
    )

    module.run_plan_updates(record, staging, [boundary])

    assert staging.read("plan") == original


def test_module3_vqa_unique_per_frame_and_camera(single_episode_root: Path, tmp_path: Path) -> None:
    payload = {
        "question": "How many cups?",
        "answer": {"label": "cup", "count": 2, "note": "white & blue"},
    }
    vlm = make_canned_responder({"frame-grounded visual question": payload})
    module = GeneralVqaModule(
        vlm=vlm,
        config=VqaConfig(vqa_emission_hz=1.0, K=3),
        seed=1,
        frame_provider=_StubFrameProvider(
            cameras=("observation.images.top", "observation.images.wrist")
        ),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("vqa")
    # every vqa row must carry a camera tag and one of the configured cameras
    for r in rows:
        assert r["style"] == "vqa"
        assert r.get("camera") in {"observation.images.top", "observation.images.wrist"}
    # at most one (vqa, user) and one (vqa, assistant) per (timestamp, camera)
    user_keys = [
        (r["timestamp"], r["camera"]) for r in rows if r["role"] == "user" and r["style"] == "vqa"
    ]
    assistant_keys = [
        (r["timestamp"], r["camera"])
        for r in rows
        if r["role"] == "assistant" and r["style"] == "vqa"
    ]
    assert len(user_keys) == len(set(user_keys))
    assert len(assistant_keys) == len(set(assistant_keys))
    # both cameras must be represented
    assert {c for _, c in user_keys} == {"observation.images.top", "observation.images.wrist"}
    # every emitted timestamp must be an exact source frame timestamp
    frame_set = set(record.frame_timestamps)
    for ts, _ in user_keys + assistant_keys:
        assert ts in frame_set


def test_module1_attaches_contact_sheets_to_subtask_prompt(
    fixture_dataset_root: Path, tmp_path: Path
) -> None:
    """Module 1 sends timestamped contact-sheet image blocks (not a raw video block)."""
    captured: list[list[dict[str, Any]]] = []
    payload = {
        "subtasks": [
            {"text": "grasp the handle of the sponge", "start": 0.0, "end": 0.5},
            {"text": "wipe the counter", "start": 0.5, "end": 1.1},
        ]
    }
    memory_payload = {"memory": "wiped once"}

    def responder(messages):
        captured.append(list(messages))
        text = ""
        for m in messages:
            for block in m.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
        if "compressed semantic memory" in text:
            return memory_payload
        return payload

    provider = _StubFrameProvider()
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        # Disable the rephrasings sub-prompt so the test's only video-bearing
        # call is the subtask one — keeps the assertions below focused on
        # ``_generate_subtasks`` rather than fighting the order of unrelated
        # text-only Module-1 sub-prompts.
        config=PlanConfig(frames_per_second=2.0, max_frames_per_prompt=60, n_task_rephrasings=0),
        frame_provider=provider,
    )
    assert module.config.subtask_generate_frame_format == "contact_sheet"
    assert module.config.subtask_realign_generated is False
    assert module.config.subtask_video_fallback == "contact_sheet"
    record = next(iter_episodes(fixture_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    # Find the call carrying the subtask prompt rather than blindly taking
    # captured[0] — Module 1 issues several sub-prompts and their order is
    # not part of the contract.
    assert captured, "no VLM calls made"

    def _prompt_text(messages):
        for m in messages:
            for block in m.get("content", []):
                if isinstance(block, dict) and block.get("type") == "text":
                    return block.get("text", "")
        return ""

    subtask_calls = [m for m in captured if "COMPLETED manipulation events" in _prompt_text(m)]
    assert len(subtask_calls) == 1, "expected exactly one subtask-prompt VLM call"
    content = subtask_calls[0][0]["content"]
    video_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "video"]
    video_url_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "video_url"]
    image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
    text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
    assert video_blocks == [], "contact-sheet mode must not emit a raw video block"
    assert video_url_blocks == [], "contact-sheet mode must not emit a native video URL"
    assert len(image_blocks) >= 1, f"expected >=1 contact-sheet image block, got {content}"
    assert all(isinstance(b["image"], PIL.Image.Image) for b in image_blocks)
    assert len(text_blocks) == 1
    # the prompt is prefixed with the contact-sheet reading instructions
    assert text_blocks[0]["text"].startswith("CONTACT SHEETS")
    # frames were decoded for this episode at episode-relative timestamps
    assert provider.calls and provider.calls[0][0] == record.episode_index


def test_module3_attaches_frame_image_block_to_prompt(
    single_episode_root: Path, tmp_path: Path
) -> None:
    """Each VQA prompt must carry a single image block at the emission frame."""
    captured: list[list[dict[str, Any]]] = []
    payload = {
        "question": "How many cups?",
        "answer": {"label": "cup", "count": 1},
    }
    provider = _StubFrameProvider()
    module = GeneralVqaModule(
        vlm=_spy_responder(captured, payload),
        config=VqaConfig(vqa_emission_hz=1.0, K=1),
        seed=0,
        frame_provider=provider,
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert captured, "no VLM calls made"
    for messages in captured:
        content = messages[0]["content"]
        image_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "image"]
        text_blocks = [b for b in content if isinstance(b, dict) and b.get("type") == "text"]
        assert len(image_blocks) == 1, f"expected 1 image block per VQA prompt, got {content}"
        assert image_blocks[0]["image"] is provider.sentinel
        assert len(text_blocks) == 1
    # provider was called once per emission per camera with the exact emission timestamp
    for ep_idx, ts_tuple, camera in provider.calls:
        assert ep_idx == record.episode_index
        assert len(ts_tuple) == 1
        assert ts_tuple[0] in record.frame_timestamps
        assert camera in provider.cameras


def test_module3_assistant_content_is_valid_json(single_episode_root: Path, tmp_path: Path) -> None:
    payload = {
        "question": "Where is the cup?",
        "answer": {
            "detections": [{"label": "cup", "bbox_format": "xyxy", "bbox": [10, 20, 50, 80]}]
        },
    }
    vlm = make_canned_responder({"frame-grounded visual question": payload})
    module = GeneralVqaModule(
        vlm=vlm,
        config=VqaConfig(vqa_emission_hz=1.0, K=2),
        seed=2,
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(single_episode_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    rows = staging.read("vqa")
    for row in rows:
        if row["role"] == "assistant" and row["style"] == "vqa":
            decoded = json.loads(row["content"])
            assert "detections" in decoded


# ----------------------------------------------------------------------
# Shared prompt-inspection helpers
# ----------------------------------------------------------------------

_ALIGN_MARKER = "say WHEN each one happens"
_GENERATE_MARKER = "COMPLETED manipulation events"

_GIVEN = ["pick up the cup", "pour the water", "put the cup down"]


def _last_prompt_text(messages: list[dict[str, Any]]) -> str:
    """Last text block across a message list (the prompt the stub answers)."""
    text = ""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
    return text


# ----------------------------------------------------------------------
# Open-ended subtask generation from native video
# ----------------------------------------------------------------------


def _install_fake_native_encoder(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[Path], list[tuple[float, ...]]]:
    """Make native-video tests inspect messages without requiring a valid MP4.

    ``StubVlmClient`` receives the internal message before the OpenAI adapter
    base64-inlines the file, so a few sentinel bytes are sufficient here. The
    production encoder itself has separate coverage in ``test_frames.py``.
    """
    paths: list[Path] = []
    timestamp_batches: list[tuple[float, ...]] = []

    def fake_encode(frames, timestamps, path, **_kwargs):
        assert len(frames) == len(timestamps)
        path.write_bytes(b"fake native video")
        paths.append(path)
        timestamp_batches.append(tuple(float(t) for t in timestamps))
        return 2.0

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fake_encode)
    return paths, timestamp_batches


def _video_url_path(messages: list[dict[str, Any]]) -> Path:
    blocks = messages[0]["content"]
    videos = [block for block in blocks if block.get("type") == "video_url"]
    assert len(videos) == 1
    url = videos[0]["video_url"]["url"]
    assert url.startswith("file://")
    return Path(url.removeprefix("file://"))


def _native_generation_config(**overrides: Any) -> PlanConfig:
    values: dict[str, Any] = {
        "subtask_generate_frame_format": "video",
        "derive_task_from_video": "off",
        "subtask_describe_first": False,
        "n_task_rephrasings": 0,
        "emit_plan": False,
        "emit_memory": False,
    }
    values.update(overrides)
    return PlanConfig(**values)


def test_generation_samples_are_uniform_when_source_timestamps_are_irregular(
    align_dataset_root: Path,
) -> None:
    """Native clip timing must not encode an irregular grid at one average FPS."""
    record = next(iter_episodes(align_dataset_root))
    irregular = replace(record, frame_timestamps=(0.0, 0.2, 1.7, 2.0))
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=_native_generation_config(frames_per_second=1.0, max_frames_per_prompt=60),
        frame_provider=_StubFrameProvider(),
    )

    assert module._generation_sample_timestamps(irregular) == [0.0, 1.0, 2.0]


def test_default_contact_sheet_retains_exact_sparse_source_timestamps(
    align_dataset_root: Path,
) -> None:
    """The new opt-in must not change the established default evidence."""
    record = next(iter_episodes(align_dataset_root))
    irregular = replace(record, frame_timestamps=(0.0, 0.2, 1.7, 2.0))
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=PlanConfig(frames_per_second=2.0, max_frames_per_prompt=60),
        frame_provider=_StubFrameProvider(),
    )

    assert module._generation_sample_timestamps(irregular) == [0.0, 0.2, 1.7, 2.0]


def test_native_generation_uses_video_url_for_derive_describe_and_segment(
    fixture_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every whole-video generation pass uses the native video modality.

    In particular, changing the segmentation pass alone would leave task
    derivation and the grounding description on contact sheets and make the
    option's behavior surprisingly hybrid.
    """
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)
    captured: list[list[dict[str, Any]]] = []

    def responder(messages: list[dict[str, Any]]) -> Any:
        captured.append(messages)
        path = _video_url_path(messages)
        assert path.exists(), "the temporary clip was deleted before the VLM read it"
        prompt = _last_prompt_text(messages)
        if "video above shows" in prompt:
            return {"task": "put the sponge in the sink"}
        if "OBSERVATION pass" in prompt:
            return {"description": "0.0-0.5s: the gripper picks up the sponge"}
        if _GENERATE_MARKER in prompt:
            return {
                "subtasks": [
                    {"text": "pick up the sponge", "start": 0.0, "end": 0.5},
                    {"text": "put the sponge in the sink", "start": 0.5, "end": 1.1},
                ]
            }
        pytest.fail(f"unexpected native-video prompt: {prompt[:120]!r}")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(
            derive_task_from_video="always",
            subtask_describe_first=True,
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(fixture_dataset_root))
    module.run_episode(record, EpisodeStaging(tmp_path / "stage", record.episode_index))

    assert len(captured) == 3
    for messages in captured:
        content = messages[0]["content"]
        video_blocks = [block for block in content if block.get("type") == "video_url"]
        assert len(video_blocks) == 1
        assert video_blocks[0]["fps"] == pytest.approx(2.0)
        assert not any(block.get("type") == "image" for block in content)
        prompt = _last_prompt_text(messages).lower()
        assert not prompt.startswith("contact sheets")
        assert "burned-in timestamp" not in prompt
        assert "timestamped contact sheet" not in prompt
    assert encoded_paths
    assert all(not path.exists() for path in encoded_paths)


def test_native_generation_restores_nonzero_episode_time_origin(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model reports clip time from zero; stored spans use episode time."""
    _install_fake_native_encoder(monkeypatch)
    record = next(iter_episodes(align_dataset_root))
    shifted = replace(
        record,
        frame_timestamps=tuple(float(timestamp) + 10.0 for timestamp in record.frame_timestamps),
    )
    reply = {
        "subtasks": [
            {"text": "pick up the bottle", "start": 0.0, "end": 1.0},
            {"text": "pour into the cup", "start": 1.0, "end": 2.0},
        ]
    }
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda messages: reply),
        config=_native_generation_config(),
        frame_provider=_StubFrameProvider(),
    )

    spans = module._generate_subtasks(shifted, task=shifted.episode_task)

    assert [(span["start"], span["end"]) for span in spans] == [(10.0, 11.0), (11.0, 12.0)]


def test_native_generation_restores_each_window_time_origin(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A native clip restarts at zero in each long-episode window."""
    encoded_paths, timestamp_batches = _install_fake_native_encoder(monkeypatch)
    replies = iter(
        [
            {"subtasks": [{"text": "pick up the bottle", "start": 0.0, "end": 1.0}]},
            {"subtasks": [{"text": "pour into the cup", "start": 0.0, "end": 1.0}]},
        ]
    )
    captured: list[list[dict[str, Any]]] = []

    def responder(messages: list[dict[str, Any]]) -> Any:
        captured.append(messages)
        return next(replies)

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(frames_per_second=1.0, max_frames_per_prompt=2),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    spans = module._generate_subtasks(record, task=record.episode_task)

    assert [(span["text"], span["start"]) for span in spans] == [
        ("pick up the bottle", 0.0),
        ("pour into the cup", 1.0),
    ]
    assert timestamp_batches == [(0.0, 1.0), (1.0, 2.0)]
    assert len(captured) == 2
    assert all(_video_url_path(messages) in encoded_paths for messages in captured)
    assert all(not path.exists() for path in encoded_paths)


@pytest.mark.parametrize(
    ("duration", "expected_frame_counts"),
    [
        (49.5, [100]),
        (50.0, [51, 51]),
        (51.18, [52, 52]),
        (99.0, [100, 100]),
        (100.0, [100, 51, 52]),
        (100.2, [100, 51, 52]),
        (149.5, [100, 100, 51, 52]),
    ],
)
def test_generation_windows_rebalance_short_tail(
    align_dataset_root: Path,
    duration: float,
    expected_frame_counts: list[int],
) -> None:
    """Keep the frame work but split the final pair instead of a tiny tail."""
    record = next(iter_episodes(align_dataset_root))
    shifted = replace(record, frame_timestamps=(10.0, 10.0 + duration))
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=_native_generation_config(
            frames_per_second=2.0,
            max_frames_per_prompt=100,
        ),
        frame_provider=_StubFrameProvider(),
    )

    windows = module._generation_windows(shifted)
    frame_counts = [
        len(module._generation_sample_timestamps(shifted, window=window)) for window in windows
    ]

    assert frame_counts == expected_frame_counts
    assert windows[0][0] == pytest.approx(10.0)
    assert windows[-1][1] == pytest.approx(10.0 + duration)
    assert all(
        left[1] == pytest.approx(right[0])
        for left, right in zip(windows, windows[1:], strict=False)
    )
    assert all(count <= 100 for count in frame_counts)
    expected_total = int(round(duration * 2.0)) + 1
    assert sum(frame_counts) - (len(frame_counts) - 1) == expected_total
    if duration == 51.18:
        assert windows == pytest.approx([(10.0, 35.59), (35.59, 61.18)])


def test_one_frame_generation_budget_still_terminates_and_covers_episode(
    align_dataset_root: Path,
) -> None:
    record = next(iter_episodes(align_dataset_root))
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=_native_generation_config(
            frames_per_second=1.0,
            max_frames_per_prompt=1,
        ),
        frame_provider=_StubFrameProvider(),
    )

    windows = module._generation_windows(record)

    assert windows == [(0.0, 1.0), (1.0, 2.0)]

    fractional_tail = replace(record, frame_timestamps=(0.0, 2.1))
    windows = module._generation_windows(fractional_tail)
    assert len(windows) == 3
    for actual, expected in zip(
        windows,
        [(0.0, 0.7), (0.7, 1.4), (1.4, 2.1)],
        strict=True,
    ):
        assert actual == pytest.approx(expected)


def test_generation_routes_episode_38_shape_through_balanced_windows(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = next(iter_episodes(align_dataset_root))
    shifted = replace(record, frame_timestamps=(10.0, 61.18))
    windows: list[tuple[float, float]] = []
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=_native_generation_config(
            frames_per_second=2.0,
            max_frames_per_prompt=100,
        ),
        frame_provider=_StubFrameProvider(),
    )

    def fake_window(_record: Any, _task: str, start: float, end: float):
        windows.append((start, end))
        return [{"text": f"window {len(windows)}", "start": start, "end": end}]

    monkeypatch.setattr(module, "_subtasks_for_window", fake_window)

    module._generate_subtasks(shifted, task=shifted.episode_task)

    assert windows == pytest.approx([(10.0, 35.59), (35.59, 61.18)])


def test_native_generation_removes_temporary_clip_when_vlm_raises(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)

    def fail_after_opening_clip(messages: list[dict[str, Any]]) -> Any:
        assert _video_url_path(messages).exists()
        raise RuntimeError("inference failed")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=fail_after_opening_clip),
        config=_native_generation_config(),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="inference failed"):
        module._generate_subtasks(record, task=record.episode_task)

    assert encoded_paths
    assert all(not path.exists() for path in encoded_paths)


def test_native_generation_falls_back_to_contact_sheets_when_encode_fails(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fail_encode(*_args, **_kwargs):
        raise OSError("encoder unavailable")

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fail_encode)
    captured: list[list[dict[str, Any]]] = []
    reply = {"subtasks": [{"text": "pour into the cup", "start": 0.0, "end": 2.0}]}

    def responder(messages: list[dict[str, Any]]) -> Any:
        captured.append(messages)
        return reply

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with caplog.at_level("WARNING"):
        spans = module._generate_subtasks(record, task=record.episode_task)

    assert [span["text"] for span in spans] == ["pour into the cup"]
    assert len(captured) == 1
    content = captured[0][0]["content"]
    assert any(block.get("type") == "image" for block in content)
    assert not any(block.get("type") == "video_url" for block in content)
    assert _last_prompt_text(captured[0]).startswith("CONTACT SHEETS")
    assert "encode" in caplog.text.lower()


def test_strict_native_generation_rejects_contact_sheet_fallback(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_encode(*_args, **_kwargs):
        raise OSError("encoder unavailable")

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fail_encode)
    vlm_calls = 0

    def responder(_messages: list[dict[str, Any]]) -> Any:
        nonlocal vlm_calls
        vlm_calls += 1
        pytest.fail("strict native-video mode must not send a fallback request")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(subtask_video_fallback="error"),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="contact-sheet fallback is disabled"):
        module._generate_subtasks(record, task=record.episode_task)

    assert vlm_calls == 0


def test_native_generation_video_url_exception_falls_back_by_default(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)

    def fail_video_url(*_args: Any, **_kwargs: Any):
        raise ValueError("adapter rejected clip")

    monkeypatch.setattr(plan_module, "to_video_url_block", fail_video_url)
    captured: list[list[dict[str, Any]]] = []

    def responder(messages: list[dict[str, Any]]) -> Any:
        captured.append(messages)
        return {"subtasks": [{"text": "pour into the cup", "start": 0.0, "end": 2.0}]}

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with caplog.at_level("WARNING"):
        spans = module._generate_subtasks(record, task=record.episode_task)

    assert [span["text"] for span in spans] == ["pour into the cup"]
    assert any(block.get("type") == "image" for block in captured[0][0]["content"])
    assert "adapter rejected clip" in caplog.text
    assert all(not path.exists() for path in encoded_paths)


def test_strict_native_generation_rejects_empty_video_url_block(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)
    monkeypatch.setattr(plan_module, "to_video_url_block", lambda *_args, **_kwargs: [])
    vlm_calls = 0

    def responder(_messages: list[dict[str, Any]]) -> Any:
        nonlocal vlm_calls
        vlm_calls += 1
        pytest.fail("strict native-video mode must not send a text-only request")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(subtask_video_fallback="error"),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="usable video_url block"):
        module._generate_subtasks(record, task=record.episode_task)

    assert vlm_calls == 0
    assert encoded_paths
    assert all(not path.exists() for path in encoded_paths)


def test_native_generation_decode_failure_never_calls_vlm(
    align_dataset_root: Path,
) -> None:
    root_error = PermissionError("cannot access /nfs/huggingface/cache")
    vlm_calls = 0

    def responder(_messages: list[dict[str, Any]]) -> Any:
        nonlocal vlm_calls
        vlm_calls += 1
        pytest.fail("a decode failure must not become a text-only VLM request")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(),
        frame_provider=null_provider(initialization_error=root_error),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(
        RuntimeError,
        match=r"PermissionError.*cannot access /nfs/huggingface/cache.*no text-only VLM request",
    ) as info:
        module._generate_subtasks(record, task=record.episode_task)

    assert vlm_calls == 0
    assert info.value.__cause__ is not None
    assert info.value.__cause__.__cause__ is root_error


def test_native_generation_rejects_empty_contact_sheet_fallback(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An encoder fallback is safe only if it still carries visual evidence."""

    def fail_encode(*_args, **_kwargs):
        raise OSError("encoder unavailable")

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fail_encode)
    vlm_calls = 0

    def responder(_messages: list[dict[str, Any]]) -> Any:
        nonlocal vlm_calls
        vlm_calls += 1
        pytest.fail("an empty contact-sheet fallback must not reach the VLM")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(),
        frame_provider=_StubFrameProvider(),
    )
    monkeypatch.setattr(module, "_contact_sheet_blocks", lambda _frames, _timestamps: [])
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="refusing to send a text-only VLM request"):
        module._generate_subtasks(record, task=record.episode_task)

    assert vlm_calls == 0


def test_native_generation_falls_back_when_temporary_clip_cannot_be_created(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_mkstemp(*_args, **_kwargs):
        raise OSError("temporary storage unavailable")

    monkeypatch.setattr(plan_module.tempfile, "mkstemp", fail_mkstemp)
    captured: list[list[dict[str, Any]]] = []

    def responder(messages: list[dict[str, Any]]) -> Any:
        captured.append(messages)
        return {"subtasks": [{"text": "pour into the cup", "start": 0.0, "end": 2.0}]}

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    spans = module._generate_subtasks(record, task=record.episode_task)

    assert [span["text"] for span in spans] == ["pour into the cup"]
    assert len(captured) == 1
    assert any(block.get("type") == "image" for block in captured[0][0]["content"])


def test_native_generation_fallback_keeps_clip_relative_time_origin(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A contact-sheet fallback must use the same clock as a native clip."""

    def fail_encode(*_args, **_kwargs):
        raise OSError("no encoder")

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fail_encode)
    record = next(iter_episodes(align_dataset_root))
    shifted = replace(
        record,
        frame_timestamps=tuple(float(timestamp) + 10.0 for timestamp in record.frame_timestamps),
    )
    reply = {
        "subtasks": [
            {"text": "pick up the bottle", "start": 0.0, "end": 1.0},
            {"text": "pour into the cup", "start": 1.0, "end": 2.0},
        ]
    }
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: reply),
        config=_native_generation_config(),
        frame_provider=_StubFrameProvider(),
    )

    spans = module._generate_subtasks(shifted, task=shifted.episode_task)

    assert [(span["start"], span["end"]) for span in spans] == [(10.0, 11.0), (11.0, 12.0)]


def test_native_generation_keeps_seeded_relabel_contact_sheet_based(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The opt-in changes discovery media, not the focused relabel pass."""
    _install_fake_native_encoder(monkeypatch)
    captured: list[list[dict[str, Any]]] = []

    def responder(messages: list[dict[str, Any]]) -> Any:
        captured.append(messages)
        prompt = _last_prompt_text(messages)
        content = messages[0]["content"]
        if _GENERATE_MARKER in prompt:
            assert any(block.get("type") == "video_url" for block in content)
            return {"subtasks": [{"text": "move bottle", "start": 0.0, "end": 2.0}]}
        if "Annotate one fixed segment" in prompt:
            assert any(block.get("type") == "image" for block in content)
            assert not any(block.get("type") == "video_url" for block in content)
            return {"label": "pour bottle into cup"}
        pytest.fail(f"unexpected prompt: {prompt[:120]!r}")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=_native_generation_config(
            subtask_seeded_relabel=True,
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    spans = module._subtask_spans(record, record.episode_task)

    assert [span["text"] for span in spans] == ["pour bottle into cup"]
    assert len(captured) == 2


# ----------------------------------------------------------------------
# Given-subtask alignment (PlanConfig.subtasks_path)
# ----------------------------------------------------------------------


def _spans_reply(*entries: tuple[int, Any, Any]) -> dict[str, Any]:
    """An alignment reply: ``(index, start, end)`` triples, ``None`` = unplaced."""
    return {"subtasks": [{"index": i, "start": s, "end": e} for i, s, e in entries]}


def _align_stub(
    replies: list[Any],
    captured: list[str] | None = None,
    captured_messages: list[list[dict[str, Any]]] | None = None,
) -> StubVlmClient:
    """Answer alignment calls from a queue, in call order.

    A queue (rather than one canned reply) is what lets a test assert how many
    alignment calls were issued — alignment must only ever issue one.
    """
    queue = list(replies)

    def responder(messages: list[dict[str, Any]]) -> Any:
        if captured_messages is not None:
            captured_messages.append(messages)
        text = _last_prompt_text(messages)
        if captured is not None:
            captured.append(text)
        if _ALIGN_MARKER in text:
            return queue.pop(0)
        if "compressed semantic memory" in text:
            return {"memory": "poured the water"}
        return None

    return StubVlmClient(responder=responder)


def _write_subtasks(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "subtasks.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _align_config(tmp_path: Path, payload: Any, **overrides: Any) -> PlanConfig:
    return PlanConfig(
        subtasks_path=_write_subtasks(tmp_path, payload),
        frames_per_second=2.0,
        n_task_rephrasings=0,
        emit_plan=False,
        emit_memory=False,
        **overrides,
    )


def _subtask_rows(staging: EpisodeStaging) -> list[dict[str, Any]]:
    return sorted(
        (r for r in staging.read("plan") if r["style"] == "subtask"),
        key=lambda r: r["timestamp"],
    )


def test_generated_labels_feed_native_video_alignment_in_order(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Discovery supplies labels only; the final boundaries come from alignment.

    Repeated text is intentional: fixed-label alignment identifies occurrences
    by index and must not collapse equal generated labels.
    """
    encoded_paths, _timestamp_batches = _install_fake_native_encoder(monkeypatch)
    captured_messages: list[list[dict[str, Any]]] = []
    labels = ["move the gripper closer", "move the gripper closer"]

    def responder(messages: list[dict[str, Any]]) -> Any:
        captured_messages.append(messages)
        prompt = _last_prompt_text(messages)
        if _GENERATE_MARKER in prompt:
            content = messages[0]["content"]
            assert any(block.get("type") == "image" for block in content)
            assert not any(block.get("type") == "video_url" for block in content)
            return {
                "subtasks": [
                    {"text": labels[0], "start": 0.0, "end": 0.5},
                    {"text": labels[1], "start": 0.5, "end": 2.0},
                ]
            }
        if _ALIGN_MARKER in prompt:
            assert _video_url_path(messages).exists()
            assert not any(block.get("type") == "image" for block in messages[0]["content"])
            return _spans_reply((0, 0.0, 1.5), (1, 1.5, 2.0))
        pytest.fail(f"unexpected prompt: {prompt[:120]!r}")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=responder),
        config=PlanConfig(
            derive_task_from_video="off",
            subtask_describe_first=False,
            n_task_rephrasings=0,
            emit_plan=False,
            emit_memory=False,
            subtask_realign_generated=True,
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    prompts = [_last_prompt_text(messages) for messages in captured_messages]
    assert len(prompts) == 2
    assert _GENERATE_MARKER in prompts[0]
    assert _ALIGN_MARKER in prompts[1]
    assert f"0. {labels[0]}" in prompts[1]
    assert f"1. {labels[1]}" in prompts[1]
    rows = _subtask_rows(staging)
    assert [row["content"] for row in rows] == labels
    # The provisional second boundary was 0.5s; only the 1.5s alignment
    # boundary is published.
    assert [row["timestamp"] for row in rows] == [0.0, 1.5]
    # Generation stayed on its default contact sheets; only alignment encoded
    # a native clip, proving the two media settings are independent.
    assert len(encoded_paths) == 1
    assert all(not path.exists() for path in encoded_paths)


def test_seeded_relabel_precedes_generated_subtask_realign(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The aligner must receive the final relabeled vocabulary, not its seeds."""
    events: list[tuple[str, Any]] = []
    generated = [
        {"text": "move object", "start": 0.0, "end": 1.0},
        {"text": "set object", "start": 1.0, "end": 2.0},
    ]
    relabeled = [
        {"text": "pick up the bottle", "start": 0.0, "end": 1.0},
        {"text": "pour into the cup", "start": 1.0, "end": 2.0},
    ]
    aligned = [
        {"text": "pick up the bottle", "start": 0.0, "end": 1.5},
        {"text": "pour into the cup", "start": 1.5, "end": 2.0},
    ]
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=_native_generation_config(
            subtask_seeded_relabel=True,
            subtask_realign_generated=True,
        ),
        frame_provider=_StubFrameProvider(),
    )

    def fake_generate(_record: Any, *, task: str) -> list[dict[str, Any]]:
        events.append(("generate", task))
        return generated

    def fake_relabel(_record: Any, spans: list[dict[str, Any]], task: str) -> list[dict[str, Any]]:
        events.append(("relabel", ([span["text"] for span in spans], task)))
        return relabeled

    def fake_align(
        _record: Any,
        labels: list[str],
        task: str,
        *,
        require_labeled_calibration: bool = False,
    ) -> list[dict[str, Any]]:
        assert require_labeled_calibration is True
        events.append(("align", (list(labels), task)))
        return aligned

    monkeypatch.setattr(module, "_generate_subtasks", fake_generate)
    monkeypatch.setattr(module, "_seeded_relabel", fake_relabel)
    monkeypatch.setattr(module, "_align_given_subtasks", fake_align)
    record = next(iter_episodes(align_dataset_root))

    result = module._subtask_spans(record, record.episode_task)

    assert [event for event, _payload in events] == ["generate", "relabel", "align"]
    assert events[-1][1][0] == [span["text"] for span in relabeled]
    assert result == aligned


def test_empty_generation_skips_generated_subtask_realign(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=_native_generation_config(subtask_realign_generated=True),
        frame_provider=_StubFrameProvider(),
    )
    monkeypatch.setattr(module, "_generate_subtasks", lambda _record, *, task: [])

    def unexpected_align(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
        pytest.fail("empty discovery must not issue a fixed-label alignment call")

    monkeypatch.setattr(module, "_align_given_subtasks", unexpected_align)
    record = next(iter_episodes(align_dataset_root))

    assert module._subtask_spans(record, record.episode_task) == []


def test_generated_subtask_realign_keeps_generation_on_empty_alignment(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _messages: {}),
        config=_native_generation_config(subtask_realign_generated=True),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    generated = [
        {"text": "pick up the bottle", "start": 0.0, "end": 1.0},
        {"text": "pour into the cup", "start": 1.0, "end": 2.0},
    ]
    monkeypatch.setattr(module, "_align_given_subtasks", lambda *_args, **_kwargs: [])

    assert module._realign_generated_subtasks(record, generated, record.episode_task) == generated
    assert "realignment produced no spans" in caplog.text
    assert "keeping generated boundaries" in caplog.text


def test_generated_subtask_realign_keeps_generation_on_partial_default_threshold(
    align_dataset_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Reject a partial refinement without losing the complete generated spans."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 2.0), (1, None, None))]),
        config=_native_generation_config(subtask_realign_generated=True),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    generated = [
        {"text": "pick up the bottle", "start": 0.0, "end": 1.0},
        {"text": "pour into the cup", "start": 1.0, "end": 2.0},
    ]

    assert module.config.max_frames_per_prompt == 60
    assert module._realign_generated_subtasks(record, generated, record.episode_task) == generated
    assert "exact ordered generated label list (expected 2, got 1)" in caplog.text
    assert "keeping generated boundaries" in caplog.text


def test_generated_subtask_realign_keeps_generation_when_frame_snap_drops_label(
    align_dataset_root: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Coverage is checked before cleaning, so assert the final labels too."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 1.96, 2.0), (1, 1.97, 2.0))]),
        config=_native_generation_config(
            subtask_realign_generated=True,
            subtask_align_min_fraction=1.0,
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    generated = [
        {"text": "move closer", "start": 0.0, "end": 1.0},
        {"text": "move closer", "start": 1.0, "end": 2.0},
    ]

    assert module._realign_generated_subtasks(record, generated, record.episode_task) == generated
    assert "exact ordered generated label list" in caplog.text


@pytest.mark.parametrize("placed", [0, 1, 2])
def test_generated_subtask_realign_falls_back_with_capped_whole_episode_frames(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    placed: int,
) -> None:
    """Run the real sampling, parsing, coverage and staging paths at 60 frames."""
    record = next(iter_episodes(align_dataset_root))
    timestamps = tuple(i / 10 for i in range(1201))
    record = replace(
        record,
        frame_timestamps=timestamps,
        frame_indices=tuple(range(len(timestamps))),
        row_count=len(timestamps),
    )
    generated = [
        {"text": label, "start": i * 30.0, "end": (i + 1) * 30.0}
        for i, label in enumerate(["pick", "pour", "place", "release"])
    ]
    frames = _StubFrameProvider()
    reply = _spans_reply(*[(i, i * 30.0, (i + 1) * 30.0) for i in range(placed)])
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([reply]),
        config=PlanConfig(subtask_realign_generated=True, n_task_rephrasings=0),
        frame_provider=frames,
    )
    monkeypatch.setattr(module, "_generate_subtasks", lambda *_a, **_kw: generated)
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    subtasks = [row for row in staging.read("plan") if row.get("style") == "subtask"]
    assert [(row["content"], row["timestamp"]) for row in subtasks] == [
        (span["text"], span["start"]) for span in generated
    ]
    cfg = AnnotationPipelineConfig(
        plan=module.config,
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
    )
    report = StagingValidator().validate([record], staging.root, config=cfg)
    assert report.ok, report.errors
    assert len(frames.calls[0][1]) == 60
    assert frames.calls[0][1][0] == 0.0
    assert frames.calls[0][1][-1] == 120.0
    assert "keeping generated boundaries" in caplog.text


@pytest.mark.parametrize("reordered", [False, True])
def test_generated_realign_checks_label_identity_and_order_even_at_equal_count(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    reordered: bool,
) -> None:
    record = next(iter_episodes(align_dataset_root))
    generated = [
        {"text": "pick", "start": 0.0, "end": 1.0},
        {"text": "place", "start": 1.0, "end": 2.0},
    ]
    aligned = [dict(span) for span in generated]
    if reordered:
        aligned.reverse()
    else:
        aligned[1]["text"] = "different action"
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _: {}),
        config=PlanConfig(subtask_realign_generated=True),
        frame_provider=_StubFrameProvider(),
    )
    monkeypatch.setattr(module, "_align_given_subtasks", lambda *_a, **_kw: aligned)
    assert module._realign_generated_subtasks(record, generated, record.episode_task) == generated


def test_generated_realign_propagates_runtime_failure(
    align_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = next(iter_episodes(align_dataset_root))
    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=lambda _: {}),
        config=PlanConfig(subtask_realign_generated=True),
        frame_provider=_StubFrameProvider(),
    )

    def fail(*_args, **_kwargs):
        raise RuntimeError("inference failed")

    monkeypatch.setattr(module, "_align_given_subtasks", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        module._realign_generated_subtasks(
            record, [{"text": "pick", "start": 0.0, "end": 2.0}], record.episode_task
        )


def test_generated_subtask_realign_skips_unlabeled_calibration(
    align_dataset_root: Path,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    labels = ["pick up the bottle", "pour into the cup"]
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.8), (1, 0.8, 2.0))]),
        config=_native_generation_config(
            subtask_realign_generated=True,
            subtask_align_min_fraction=1.0,
            subtask_align_calibration_path=_write_calibration(tmp_path, [0.3]),
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    generated = [
        {"text": labels[0], "start": 0.0, "end": 1.0},
        {"text": labels[1], "start": 1.0, "end": 2.0},
    ]

    with caplog.at_level("WARNING"):
        aligned = module._realign_generated_subtasks(record, generated, record.episode_task)

    assert [span["start"] for span in aligned] == pytest.approx([0.0, 0.8])
    assert "calibration has no recorded subtask list" in caplog.text


def test_generated_subtask_realign_applies_exactly_labeled_calibration(
    align_dataset_root: Path,
    tmp_path: Path,
) -> None:
    labels = ["pick up the bottle", "pour into the cup"]
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.8), (1, 0.8, 2.0))]),
        config=_native_generation_config(
            subtask_realign_generated=True,
            subtask_align_min_fraction=1.0,
            subtask_align_calibration_path=_write_calibration(
                tmp_path,
                {"offsets": [0.3], "labels": labels},
            ),
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    generated = [
        {"text": labels[0], "start": 0.0, "end": 1.0},
        {"text": labels[1], "start": 1.0, "end": 2.0},
    ]

    aligned = module._realign_generated_subtasks(record, generated, record.episode_task)

    assert [span["start"] for span in aligned] == pytest.approx([0.0, 0.5])


@pytest.mark.parametrize("bad_text", [None, 7, {"label": "pick"}])
def test_clean_spans_rejects_non_string_labels(
    align_dataset_root: Path,
    bad_text: Any,
) -> None:
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([]),
        config=PlanConfig(),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(ValueError, match="needs string text"):
        module._clean_spans(
            [
                {"text": bad_text, "start": 0.0, "end": 1.0},
                {"text": "valid label", "start": 1.0, "end": 2.0},
            ],
            record,
        )


def test_align_uses_given_labels_and_never_generates_them(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """Given subtasks are timed, not rewritten: labels and order come from the
    file verbatim, boundaries come from the model's spans, and the
    label-writing segmentation prompt is never issued."""
    captured: list[str] = []
    vlm = _align_stub(
        [_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0), (2, 2.0, 2.0))],
        captured=captured,
    )
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(tmp_path, _GIVEN, subtask_video_fallback="error"),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    rows = _subtask_rows(staging)
    assert [r["content"] for r in rows] == _GIVEN
    assert [r["timestamp"] for r in rows] == [0.0, 1.0, 2.0]
    # The VLM was asked to align, never to invent labels.
    assert any(_ALIGN_MARKER in text for text in captured)
    assert not any(_GENERATE_MARKER in text for text in captured)
    # The alignment prompt carries the ordered list the model must choose from.
    align_prompt = next(text for text in captured if _ALIGN_MARKER in text)
    assert "0. pick up the cup" in align_prompt
    assert "2. put the cup down" in align_prompt


def test_native_alignment_falls_back_to_contact_sheets_by_default(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_encode(*_args, **_kwargs):
        raise OSError("encoder unavailable")

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fail_encode)
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub(
            [_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))],
            captured_messages=captured_messages,
        ),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    aligned = module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert [span["text"] for span in aligned] == _GIVEN[:2]
    assert len(captured_messages) == 1
    content = captured_messages[0][0]["content"]
    assert any(block.get("type") == "image" for block in content)
    assert not any(block.get("type") == "video_url" for block in content)


def test_native_alignment_video_url_exception_falls_back_by_default(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)

    def fail_video_url(*_args: Any, **_kwargs: Any):
        raise ValueError("adapter rejected clip")

    monkeypatch.setattr(plan_module, "to_video_url_block", fail_video_url)
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub(
            [_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))],
            captured_messages=captured_messages,
        ),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with caplog.at_level("WARNING"):
        aligned = module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert [span["text"] for span in aligned] == _GIVEN[:2]
    assert any(block.get("type") == "image" for block in captured_messages[0][0]["content"])
    assert "adapter rejected clip" in caplog.text
    assert all(not path.exists() for path in encoded_paths)


def test_native_alignment_temp_allocation_failure_falls_back_by_default(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        plan_module.tempfile,
        "mkstemp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("temporary storage unavailable")),
    )
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub(
            [_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))],
            captured_messages=captured_messages,
        ),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    aligned = module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert [span["text"] for span in aligned] == _GIVEN[:2]
    assert any(block.get("type") == "image" for block in captured_messages[0][0]["content"])


def test_strict_native_alignment_rejects_contact_sheet_fallback(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_encode(*_args, **_kwargs):
        raise OSError("encoder unavailable")

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fail_encode)
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([], captured_messages=captured_messages),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="contact-sheet fallback is disabled"):
        module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert captured_messages == []


def test_strict_native_alignment_rejects_temp_allocation_failure(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(
        plan_module.tempfile,
        "mkstemp",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("temporary storage unavailable")),
    )
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([], captured_messages=captured_messages),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with (
        caplog.at_level("WARNING"),
        pytest.raises(
            RuntimeError,
            match="contact-sheet fallback is disabled",
        ),
    ):
        module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert captured_messages == []
    assert "temporary storage unavailable" in caplog.text


def test_strict_native_alignment_rejects_empty_video_url_block(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)
    monkeypatch.setattr(plan_module, "to_video_url_block", lambda *_args, **_kwargs: [])
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([], captured_messages=captured_messages),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="usable video_url block"):
        module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert captured_messages == []
    assert encoded_paths
    assert all(not path.exists() for path in encoded_paths)


def test_strict_native_alignment_rejects_video_url_exception(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)

    def fail_video_url(*_args: Any, **_kwargs: Any):
        raise ValueError("adapter rejected clip")

    monkeypatch.setattr(plan_module, "to_video_url_block", fail_video_url)
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([], captured_messages=captured_messages),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="construct its video_url block") as info:
        module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert isinstance(info.value.__cause__, ValueError)
    assert captured_messages == []
    assert all(not path.exists() for path in encoded_paths)


def test_native_alignment_cleanup_never_masks_vlm_failure(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    encoded_paths, _ = _install_fake_native_encoder(monkeypatch)
    original_unlink = Path.unlink

    def fail_clip_cleanup(path: Path, *args: Any, **kwargs: Any):
        if path in encoded_paths:
            raise PermissionError("cleanup denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_clip_cleanup)

    def fail_inference(messages: list[dict[str, Any]]) -> Any:
        assert _video_url_path(messages).exists()
        raise RuntimeError("inference failed")

    module = PlanSubtasksMemoryModule(
        vlm=StubVlmClient(responder=fail_inference),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    try:
        with caplog.at_level("WARNING"), pytest.raises(RuntimeError, match="inference failed"):
            module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)
        assert "could not remove temporary alignment clip" in caplog.text
    finally:
        for path in encoded_paths:
            original_unlink(path, missing_ok=True)


def test_strict_native_alignment_rejects_missing_visual_views(
    align_dataset_root: Path,
    tmp_path: Path,
) -> None:
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([], captured_messages=captured_messages),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(failed_cameras=(None,)),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="could not decode a complete selected camera view"):
        module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert captured_messages == []


def test_strict_native_alignment_rejects_motion_sampling_fallback(
    align_dataset_root: Path,
    tmp_path: Path,
) -> None:
    captured_messages: list[list[dict[str, Any]]] = []
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([], captured_messages=captured_messages),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            subtask_align_frame_format="video",
            subtask_align_sampling="motion_stratified",
            subtask_video_fallback="error",
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))

    with pytest.raises(RuntimeError, match="contact-sheet fallback is disabled"):
        module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert captured_messages == []


def test_align_drops_subtask_the_model_did_not_place(
    align_dataset_root: Path, tmp_path: Path, caplog
) -> None:
    """``null`` boundaries mean "this episode never does that": the subtask is
    dropped loudly rather than squeezed into the timeline."""
    vlm = _align_stub([_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0), (2, None, None))])
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(tmp_path, _GIVEN),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    with caplog.at_level("WARNING"):
        module.run_episode(record, staging)

    rows = _subtask_rows(staging)
    assert [r["content"] for r in rows] == ["pick up the cup", "pour the water"]
    assert [r["timestamp"] for r in rows] == [0.0, 1.0]
    assert "aligned 2/3 given subtask(s)" in caplog.text
    assert "put the cup down" in caplog.text


def test_align_drops_out_of_order_spans(align_dataset_root: Path, tmp_path: Path, caplog) -> None:
    """The supplied list is the authority on ordering, so a label the model
    places before its predecessor is dropped, not reordered."""
    vlm = _align_stub([_spans_reply((0, 1.0, 2.0), (1, 0.0, 1.0), (2, 2.0, 2.0))])
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(tmp_path, _GIVEN, subtask_align_min_fraction=0.0),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    with caplog.at_level("WARNING"):
        module.run_episode(record, staging)

    assert [r["content"] for r in _subtask_rows(staging)] == ["pick up the cup", "put the cup down"]
    assert "out of order at 0.00s" in caplog.text


def test_align_starting_late_still_covers_the_episode(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """The model can start the first subtask after t0 — the stitch still hands
    every frame an active subtask."""
    vlm = _align_stub([_spans_reply((0, 1.0, 1.5), (1, 1.5, 2.0))])
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(tmp_path, _GIVEN[:2]),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    rows = _subtask_rows(staging)
    assert [r["content"] for r in rows] == _GIVEN[:2]
    # The idle head is folded into the first subtask rather than left uncovered.
    assert rows[0]["timestamp"] == record.frame_timestamps[0]
    assert rows[1]["timestamp"] == 1.5


def test_align_never_splits_across_calls(align_dataset_root: Path, tmp_path: Path) -> None:
    """Alignment always issues exactly ONE call: a chunk of tiles cannot tell
    which part of the episode it covers, and the model restarts its indices in
    every chunk. ``max_frames_per_prompt`` therefore caps the tiles instead of
    splitting them."""
    captured: list[str] = []
    provider = _StubFrameProvider()
    vlm = _align_stub([_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))], captured=captured)
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        # 5 tiles would be sampled at fps=2.0 over a 2s episode; the cap is 2.
        config=_align_config(tmp_path, _GIVEN[:2], max_frames_per_prompt=2),
        frame_provider=provider,
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert len([text for text in captured if _ALIGN_MARKER in text]) == 1
    # ...and the cap applied, rather than the tiles being chunked into calls.
    assert [len(ts) for _ep, ts, _cam in provider.calls] == [2]
    assert [r["content"] for r in _subtask_rows(staging)] == _GIVEN[:2]


def test_align_uses_synchronized_labeled_camera_views(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """Configured views receive the identical timestamp grid in stable order,
    and each view is labeled so repeating the timeline is unambiguous."""
    captured_messages: list[list[dict[str, Any]]] = []
    provider = _StubFrameProvider(
        cameras=("observation.images.top", "observation.images.wrist"),
    )
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub(
            [_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))],
            captured_messages=captured_messages,
        ),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            # Total camera-frame budget: two cameras receive two timestamps each.
            max_frames_per_prompt=4,
            subtask_align_camera_keys=("observation.images.wrist", "observation.images.top"),
        ),
        frame_provider=provider,
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert [camera for _ep, _timestamps, camera in provider.calls] == [
        "observation.images.wrist",
        "observation.images.top",
    ]
    assert provider.calls[0][1] == provider.calls[1][1]
    assert len(provider.calls[0][1]) == 2
    content = captured_messages[0][0]["content"]
    image_blocks = [block for block in content if block.get("type") == "image"]
    # One chronological sheet sequence: each tile stacks both views instead of
    # presenting two full timelines that the model could concatenate.
    assert len(image_blocks) == 1
    prompt = _last_prompt_text(captured_messages[0])
    assert "VIEW 1 = observation.images.wrist" in prompt
    assert "VIEW 2 = observation.images.top" in prompt
    assert "SAME moment" in prompt


def test_native_multiview_alignment_stacks_views_without_diluting_timeline(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encoded_paths, timestamp_batches = _install_fake_native_encoder(monkeypatch)
    captured_messages: list[list[dict[str, Any]]] = []
    provider = _StubFrameProvider(
        cameras=("observation.images.top", "observation.images.wrist"),
    )
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub(
            [_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))],
            captured_messages=captured_messages,
        ),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            max_frames_per_prompt=4,
            subtask_align_camera_keys=("observation.images.wrist", "observation.images.top"),
            subtask_align_frame_format="video",
            subtask_video_fallback="error",
        ),
        frame_provider=provider,
    )
    record = next(iter_episodes(align_dataset_root))

    aligned = module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert [span["text"] for span in aligned] == _GIVEN[:2]
    assert [(camera, len(timestamps)) for _ep, timestamps, camera in provider.calls] == [
        ("observation.images.wrist", 4),
        ("observation.images.top", 4),
    ]
    assert [len(batch) for batch in timestamp_batches] == [4]
    content = captured_messages[0][0]["content"]
    assert any(block.get("type") == "video_url" for block in content)
    assert not any(block.get("type") == "image" for block in content)
    prompt = _last_prompt_text(captured_messages[0])
    assert "VIEW 1 = observation.images.wrist" in prompt
    assert "VIEW 2 = observation.images.top" in prompt
    assert all(not path.exists() for path in encoded_paths)


def test_native_multiview_contact_sheet_fallback_honors_total_frame_budget(
    align_dataset_root: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_encode(*_args: Any, **_kwargs: Any):
        raise OSError("encoder unavailable")

    monkeypatch.setattr(plan_module, "encode_frames_to_clip", fail_encode)
    fallback_shape: dict[str, Any] = {}

    def capture_sheets(camera_frames: Any, timestamps: Any, **_kwargs: Any):
        fallback_shape["view_lengths"] = [len(frames) for _key, frames in camera_frames]
        fallback_shape["timestamps"] = list(timestamps)
        return [{"type": "image", "image": PIL.Image.new("RGB", (32, 24))}]

    monkeypatch.setattr(plan_module, "to_multiview_contact_sheet_blocks", capture_sheets)
    provider = _StubFrameProvider(
        cameras=("observation.images.top", "observation.images.wrist"),
    )
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            max_frames_per_prompt=4,
            subtask_align_camera_keys=("observation.images.wrist", "observation.images.top"),
            subtask_align_frame_format="video",
        ),
        frame_provider=provider,
    )
    record = next(iter_episodes(align_dataset_root))

    aligned = module._align_given_subtasks(record, _GIVEN[:2], record.episode_task)

    assert [span["text"] for span in aligned] == _GIVEN[:2]
    assert fallback_shape["view_lengths"] == [2, 2]
    assert len(fallback_shape["timestamps"]) == 2
    assert sum(fallback_shape["view_lengths"]) <= module.config.max_frames_per_prompt


def test_align_skips_missing_camera_and_uses_available_view(
    align_dataset_root: Path, tmp_path: Path, caplog
) -> None:
    provider = _StubFrameProvider(cameras=("observation.images.top",))
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            max_frames_per_prompt=2,
            subtask_align_camera_keys=("observation.images.missing", "observation.images.top"),
        ),
        frame_provider=provider,
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    with caplog.at_level("WARNING"):
        module.run_episode(record, staging)

    assert [camera for _ep, _timestamps, camera in provider.calls] == ["observation.images.top"]
    assert "observation.images.missing" in caplog.text
    assert [r["content"] for r in _subtask_rows(staging)] == _GIVEN[:2]


def test_align_rebudgets_surviving_view_after_per_episode_camera_failure(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    provider = _StubFrameProvider(
        cameras=("observation.images.wrist", "observation.images.side"),
        failed_cameras=("observation.images.side",),
    )
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            max_frames_per_prompt=4,
            subtask_align_camera_keys=("observation.images.wrist", "observation.images.side"),
        ),
        frame_provider=provider,
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert [(camera, len(timestamps)) for _ep, timestamps, camera in provider.calls] == [
        ("observation.images.wrist", 2),
        ("observation.images.side", 2),
        ("observation.images.wrist", 4),
    ]
    assert [r["content"] for r in _subtask_rows(staging)] == _GIVEN[:2]


def test_align_camera_count_never_exceeds_total_frame_budget(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    provider = _StubFrameProvider(cameras=("cam.a", "cam.b", "cam.c"))
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN[:2],
            max_frames_per_prompt=2,
            subtask_align_camera_keys=("cam.a", "cam.b", "cam.c"),
        ),
        frame_provider=provider,
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert [(camera, len(timestamps)) for _ep, timestamps, camera in provider.calls] == [
        ("cam.a", 1),
        ("cam.b", 1),
    ]


def test_align_motion_sampling_accepts_one_frame_budget(
    align_dataset_root: Path, tmp_path: Path, monkeypatch
) -> None:
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            max_frames_per_prompt=1,
            subtask_align_sampling="motion_stratified",
        ),
        root=align_dataset_root,
    )
    record = next(iter_episodes(align_dataset_root))
    module._motion_feature_keys = ("observation.synthetic_state",)
    monkeypatch.setattr(
        record,
        "frame_columns",
        lambda _names: {
            "observation.synthetic_state": [[float(index)] for index in range(record.row_count)]
        },
    )

    assert module._align_sample_timestamps(record) == [record.frame_timestamps[0]]


def test_align_motion_sampling_discovers_numeric_observations(
    align_dataset_root: Path, tmp_path: Path, monkeypatch
) -> None:
    data_path = next((align_dataset_root / "data").rglob("*.parquet"))
    frame = pytest.importorskip("pandas").read_parquet(data_path)
    frame["observation.arbitrary_pose"] = [[float(i), float(i % 3)] for i in range(len(frame))]
    frame.to_parquet(data_path, index=False)
    info_path = align_dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"]["observation.arbitrary_pose"] = {
        "dtype": "float32",
        "shape": [2],
        "names": None,
    }
    info_path.write_text(json.dumps(info), encoding="utf-8")

    captured: dict[str, Any] = {}

    def fake_selector(timestamps, signal_groups, budget):
        captured["timestamps"] = timestamps
        captured["groups"] = signal_groups
        captured["budget"] = budget
        return [float(timestamps[0]), 0.7, float(timestamps[-1])]

    monkeypatch.setattr(
        "lerobot_align.modules.plan_subtasks_memory.select_motion_stratified_timestamps",
        fake_selector,
    )
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            max_frames_per_prompt=3,
            subtask_align_sampling="motion_stratified",
        ),
        root=align_dataset_root,
    )
    record = next(iter_episodes(align_dataset_root))

    assert module._align_sample_timestamps(record) == [0.0, 0.7, 2.0]
    assert module._motion_feature_keys == ("observation.arbitrary_pose",)
    assert captured["budget"] == 3
    assert list(captured["groups"]) == ["observation.arbitrary_pose"]
    assert len(captured["groups"]["observation.arbitrary_pose"]) == record.row_count


def test_align_motion_sampling_falls_back_exactly_when_dataset_has_no_signals(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    motion_module = PlanSubtasksMemoryModule(
        vlm=_align_stub([]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            max_frames_per_prompt=3,
            subtask_align_sampling="motion_stratified",
        ),
        root=align_dataset_root,
    )
    uniform_module = PlanSubtasksMemoryModule(
        vlm=_align_stub([]),
        config=_align_config(tmp_path, _GIVEN, max_frames_per_prompt=3),
        root=align_dataset_root,
    )
    record = next(iter_episodes(align_dataset_root))

    assert motion_module._align_sample_timestamps(
        record
    ) == uniform_module._align_sample_timestamps(record)


def test_align_per_episode_file_overrides_default(align_dataset_root: Path, tmp_path: Path) -> None:
    """An episode-keyed entry wins over ``default`` for that episode."""
    vlm = _align_stub([_spans_reply((0, 0.0, 1.0), (1, 1.0, 2.0))])
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(
            tmp_path,
            {"default": ["wrong a", "wrong b"], "0": ["open the drawer", "close the drawer"]},
        ),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    assert [r["content"] for r in _subtask_rows(staging)] == ["open the drawer", "close the drawer"]


def test_align_accepts_badge_formatted_timestamps(align_dataset_root: Path, tmp_path: Path) -> None:
    """The tiles carry their time as a burned-in badge, which
    ``_draw_timestamp_badge`` renders as ``f"{t:06.2f}s"`` — so a model reading
    boundaries off the tiles may well answer ``"000.50s"`` rather than ``0.5``.
    Parsing those as floats must work, or the episode silently aligns 0/N."""
    vlm = _align_stub(
        [
            _spans_reply(
                (0, "000.00s", "001.00s"), (1, "001.00s", "002.00s"), (2, "002.00s", "002.00s")
            )
        ]
    )
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(tmp_path, _GIVEN),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    rows = _subtask_rows(staging)
    assert [r["content"] for r in rows] == _GIVEN
    assert [r["timestamp"] for r in rows] == [0.0, 1.0, 2.0]


def test_align_accepts_clock_formatted_timestamps(align_dataset_root: Path, tmp_path: Path) -> None:
    """The badge is never rendered as ``MM:SS``, but a model answering with a
    format of its own choosing rather than the one it read has been observed
    to use it anyway. Must parse the same as the plain-seconds badge form."""
    vlm = _align_stub(
        [
            _spans_reply(
                (0, "00:00.00s", "00:01.00s"),
                (1, "00:01.00s", "00:02.00s"),
                (2, "00:02.00s", "00:02.00s"),
            )
        ]
    )
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(tmp_path, _GIVEN),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    rows = _subtask_rows(staging)
    assert [r["content"] for r in rows] == _GIVEN
    assert [r["timestamp"] for r in rows] == [0.0, 1.0, 2.0]


def test_align_infers_index_from_reply_order(align_dataset_root: Path, tmp_path: Path) -> None:
    """A model that answers in list order often drops the redundant ``index``
    field; the entry's position stands in for it."""
    vlm = _align_stub(
        [
            {
                "subtasks": [
                    {"start": 0.0, "end": 1.0},
                    {"start": 1.0, "end": 2.0},
                    {"start": 2.0, "end": 2.0},
                ]
            }
        ]
    )
    module = PlanSubtasksMemoryModule(
        vlm=vlm,
        config=_align_config(tmp_path, _GIVEN),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    rows = _subtask_rows(staging)
    assert [r["content"] for r in rows] == _GIVEN
    assert [r["timestamp"] for r in rows] == [0.0, 1.0, 2.0]


def test_align_fails_when_coverage_below_threshold(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """A mostly-failed alignment is stitched into full coverage and passes
    validation, so it must abort rather than publish plausible-looking spans."""
    only_one = [_spans_reply((0, 0.0, 2.0), (1, None, None), (2, None, None))]
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub(only_one),
        config=_align_config(tmp_path, _GIVEN),  # 1 of 3 aligned = 0.33 < 0.5
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    with pytest.raises(ValueError, match="aligned only 1/3"):
        module.run_episode(record, staging)


def test_align_threshold_zero_restores_stitch_and_warn(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """``subtask_align_min_fraction=0`` opts back into covering the gaps."""
    only_one = [_spans_reply((0, 0.0, 2.0), (1, None, None), (2, None, None))]
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub(only_one),
        config=_align_config(tmp_path, _GIVEN, subtask_align_min_fraction=0.0),
        frame_provider=_StubFrameProvider(),
    )
    record = next(iter_episodes(align_dataset_root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)

    rows = _subtask_rows(staging)
    assert [r["content"] for r in rows] == ["pick up the cup"]


def test_align_bad_subtasks_file_fails_at_construction(tmp_path: Path) -> None:
    """A malformed or missing file must raise up front, never silently fall
    back to VLM-generated labels."""
    with pytest.raises(FileNotFoundError):
        PlanSubtasksMemoryModule(
            vlm=_align_stub([]), config=PlanConfig(subtasks_path=tmp_path / "nope.json")
        )
    with pytest.raises(ValueError, match="non-empty string"):
        PlanSubtasksMemoryModule(
            vlm=_align_stub([]),
            config=PlanConfig(subtasks_path=_write_subtasks(tmp_path, ["ok", ""])),
        )


# --- alignment calibration -------------------------------------------------
# The model's boundary error against a human reference is largely a fixed
# per-boundary offset rather than noise, so the pipeline can subtract one fit on
# a labelled subset. These cover the correction, the guard that stops offsets
# fit for one label list being applied to another, and the refusal to run on a
# malformed file. The fixture episode spans 0.0-2.0s at 10 fps, so every
# asserted boundary below is an exact source frame and snapping is a no-op.


def _write_calibration(tmp_path: Path, payload: Any) -> Path:
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _calibrated_module(tmp_path: Path, payload: Any) -> PlanSubtasksMemoryModule:
    return PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.8), (1, 0.8, 1.6), (2, 1.6, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            subtask_align_min_fraction=0.0,
            subtask_align_calibration_path=_write_calibration(tmp_path, payload),
        ),
        frame_provider=_StubFrameProvider(),
    )


def _run_align(
    module: PlanSubtasksMemoryModule, root: Path, tmp_path: Path
) -> list[dict[str, Any]]:
    record = next(iter_episodes(root))
    staging = EpisodeStaging(tmp_path / "stage", record.episode_index)
    module.run_episode(record, staging)
    return _subtask_rows(staging)


def test_align_calibration_shifts_boundaries_earlier(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """Offset ``i`` moves the start of subtask ``i+1`` back by that many seconds."""
    module = _calibrated_module(tmp_path, {"offsets": [0.3, 0.3], "labels": _GIVEN})
    rows = _run_align(module, align_dataset_root, tmp_path)

    assert [r["content"] for r in rows] == _GIVEN
    # The model put the boundaries at 0.8s and 1.6s; each offset pulls its own
    # 0.3s earlier. The first start is forced to the episode start regardless.
    assert rows[0]["timestamp"] == pytest.approx(0.0, abs=0.05)
    assert rows[1]["timestamp"] == pytest.approx(0.5, abs=0.05)
    assert rows[2]["timestamp"] == pytest.approx(1.3, abs=0.05)


def test_align_calibration_is_skipped_for_a_different_label_list(
    align_dataset_root: Path, tmp_path: Path, caplog
) -> None:
    """Offsets are positional, so a calibration fit for other labels must not be
    applied — it would silently move the wrong boundary."""
    module = _calibrated_module(
        tmp_path, {"offsets": [0.3, 0.3], "labels": ["something", "else", "entirely"]}
    )
    with caplog.at_level("WARNING"):
        rows = _run_align(module, align_dataset_root, tmp_path)

    assert rows[1]["timestamp"] == pytest.approx(0.8, abs=0.05)
    assert rows[2]["timestamp"] == pytest.approx(1.6, abs=0.05)
    assert "fit for a different subtask list" in caplog.text


def test_align_calibration_ignores_uncovered_boundaries(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """A calibration shorter than the label list corrects what it covers and
    leaves the rest as the model reported them."""
    module = _calibrated_module(tmp_path, {"offsets": [0.3]})
    rows = _run_align(module, align_dataset_root, tmp_path)

    assert rows[1]["timestamp"] == pytest.approx(0.5, abs=0.05)
    assert rows[2]["timestamp"] == pytest.approx(1.6, abs=0.05)


def test_align_calibration_accepts_a_bare_offset_list(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """A hand-written file may be just the offsets, with no applicability guard."""
    module = _calibrated_module(tmp_path, [0.3, 0.3])
    rows = _run_align(module, align_dataset_root, tmp_path)

    assert rows[1]["timestamp"] == pytest.approx(0.5, abs=0.05)
    assert rows[2]["timestamp"] == pytest.approx(1.3, abs=0.05)


@pytest.mark.parametrize(
    "payload", [{"offsets": []}, {"offsets": ["soon"]}, {"offsets": [float("nan")]}, "nope"]
)
def test_align_calibration_rejects_a_malformed_file(tmp_path: Path, payload: Any) -> None:
    """A bad calibration raises instead of silently emitting uncorrected spans."""
    with pytest.raises(ValueError):
        PlanSubtasksMemoryModule(
            vlm=_align_stub([]),
            config=_align_config(
                tmp_path,
                _GIVEN,
                subtask_align_calibration_path=_write_calibration(tmp_path, payload),
            ),
            frame_provider=_StubFrameProvider(),
        )


# --- DP boundary solver ----------------------------------------------------
# Supplying expected subtask lengths switches the calibration from shifting each
# boundary independently to solving the whole ordered set at once. That uses the
# duration prior and makes out-of-order output impossible by construction — the
# independent shift could push a boundary past its neighbour and cost a label.


def _dp_calibration(**overrides: Any) -> dict[str, Any]:
    payload = {
        "mode": "fraction_of_duration",
        "offsets": [0.0, 0.0],
        "segment_fractions": [1 / 3, 1 / 3, 1 / 3],
        "residual_scale": 0.05,
        "duration_scale": 0.05,
        "duration_weight": 1.0,
        "labels": _GIVEN,
    }
    payload.update(overrides)
    return payload


def test_dp_solver_pulls_boundaries_toward_expected_lengths(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """With equal-thirds priors and a heavy weight, a lopsided reply is evened out."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.2), (1, 0.2, 0.4), (2, 0.4, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            subtask_align_min_fraction=0.0,
            subtask_align_calibration_path=_write_calibration(
                tmp_path, _dp_calibration(duration_weight=50.0)
            ),
        ),
        frame_provider=_StubFrameProvider(),
    )
    rows = _run_align(module, align_dataset_root, tmp_path)

    # The episode runs 0.0-2.0s, so equal thirds put the boundaries near 0.67s
    # and 1.33s rather than the 0.2s/0.4s the model proposed.
    assert rows[1]["timestamp"] == pytest.approx(0.67, abs=0.15)
    assert rows[2]["timestamp"] == pytest.approx(1.33, abs=0.15)


def test_dp_solver_keeps_the_model_when_the_prior_is_weightless(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """``duration_weight=0`` reduces the solver to trusting the model."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.5), (1, 0.5, 1.2), (2, 1.2, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            subtask_align_min_fraction=0.0,
            subtask_align_calibration_path=_write_calibration(
                tmp_path, _dp_calibration(duration_weight=0.0)
            ),
        ),
        frame_provider=_StubFrameProvider(),
    )
    rows = _run_align(module, align_dataset_root, tmp_path)

    assert rows[1]["timestamp"] == pytest.approx(0.5, abs=0.1)
    assert rows[2]["timestamp"] == pytest.approx(1.2, abs=0.1)


def test_dp_solver_never_emits_out_of_order_boundaries(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """Even an inverted reply comes back ordered, with every label placed."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 1.8), (1, 1.8, 0.4), (2, 0.4, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            subtask_align_min_fraction=0.0,
            subtask_align_calibration_path=_write_calibration(tmp_path, _dp_calibration()),
        ),
        frame_provider=_StubFrameProvider(),
    )
    rows = _run_align(module, align_dataset_root, tmp_path)

    assert [r["content"] for r in rows] == _GIVEN
    stamps = [r["timestamp"] for r in rows]
    assert stamps == sorted(stamps)


def test_dp_solver_falls_back_when_a_label_is_unplaced(
    align_dataset_root: Path, tmp_path: Path
) -> None:
    """The solver needs the full ordered answer; a null boundary uses the shift."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.8), (1, None, None), (2, 1.6, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            subtask_align_min_fraction=0.0,
            subtask_align_calibration_path=_write_calibration(
                tmp_path, _dp_calibration(offsets=[0.0, 0.15])
            ),
        ),
        frame_provider=_StubFrameProvider(),
    )
    rows = _run_align(module, align_dataset_root, tmp_path)

    # Only two labels survive, and the second keeps the shifted (not solved)
    # boundary: 1.6s minus 0.15 of the 2.0s episode.
    assert [r["content"] for r in rows] == [_GIVEN[0], _GIVEN[2]]
    assert rows[1]["timestamp"] == pytest.approx(1.3, abs=0.1)


def test_dp_calibration_rejects_a_mismatched_segment_count(tmp_path: Path) -> None:
    """Segment priors must cover exactly one more entry than there are offsets."""
    with pytest.raises(ValueError, match="segment fraction"):
        PlanSubtasksMemoryModule(
            vlm=_align_stub([]),
            config=_align_config(
                tmp_path,
                _GIVEN,
                subtask_align_calibration_path=_write_calibration(
                    tmp_path, _dp_calibration(segment_fractions=[0.5, 0.5])
                ),
            ),
            frame_provider=_StubFrameProvider(),
        )


def test_dp_solver_warns_when_the_prior_overrides_the_model(
    align_dataset_root: Path, tmp_path: Path, caplog
) -> None:
    """Calibration hides the symptom of a failed alignment: the solver returns a
    tidy segmentation whatever the model said, and every label is placed, so
    ``subtask_align_min_fraction`` cannot catch it. The episode must say so."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.1), (1, 0.1, 0.2), (2, 0.2, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            subtask_align_min_fraction=0.0,
            subtask_align_calibration_path=_write_calibration(
                tmp_path, _dp_calibration(duration_weight=50.0, residual_scale=0.02)
            ),
        ),
        frame_provider=_StubFrameProvider(),
    )
    with caplog.at_level("WARNING"):
        rows = _run_align(module, align_dataset_root, tmp_path)

    assert [r["content"] for r in rows] == _GIVEN  # output still looks healthy
    assert "mostly the duration prior rather than observation" in caplog.text


def test_dp_solver_stays_quiet_when_the_model_and_prior_agree(
    align_dataset_root: Path, tmp_path: Path, caplog
) -> None:
    """The guard must not cry wolf on an ordinary episode."""
    module = PlanSubtasksMemoryModule(
        vlm=_align_stub([_spans_reply((0, 0.0, 0.67), (1, 0.67, 1.33), (2, 1.33, 2.0))]),
        config=_align_config(
            tmp_path,
            _GIVEN,
            subtask_align_min_fraction=0.0,
            subtask_align_calibration_path=_write_calibration(tmp_path, _dp_calibration()),
        ),
        frame_provider=_StubFrameProvider(),
    )
    with caplog.at_level("WARNING"):
        _run_align(module, align_dataset_root, tmp_path)

    assert "mostly the duration prior" not in caplog.text
