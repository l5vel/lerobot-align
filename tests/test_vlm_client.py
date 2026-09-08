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
"""Unit tests for ``vlm_client`` helpers."""

from __future__ import annotations

import base64
from types import SimpleNamespace

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot_align.config import VlmConfig
from lerobot_align import vlm_client as vlm_module
from lerobot_align.vlm_client import _bind_serve_port, _default_serve_command, _to_openai_messages


def test_invalid_video_metadata_policy_fails_before_inference():
    with pytest.raises(ValueError, match="video_metadata_source"):
        VlmConfig(video_metadata_source="guess")


def test_bind_serve_port_substitutes_placeholder() -> None:
    # The {port} placeholder is replaced everywhere it appears, regardless of
    # parallel vs single server — the bug was the single-server path passing
    # it through unsubstituted.
    cmd = "vllm serve M --max-model-len 32768 --port {port}"
    assert _bind_serve_port(cmd, 8000) == "vllm serve M --max-model-len 32768 --port 8000"


def test_bind_serve_port_appends_when_missing() -> None:
    assert _bind_serve_port("vllm serve M", 8001) == "vllm serve M --port 8001"


def test_bind_serve_port_leaves_explicit_port_untouched() -> None:
    cmd = "vllm serve M --port 9000"
    assert _bind_serve_port(cmd, 8000) == cmd


def test_default_serve_command_prefers_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lerobot_align.vlm_client.shutil.which",
        lambda executable: (
            f"/usr/bin/{executable}" if executable in {"vllm", "transformers"} else None
        ),
    )

    cmd = _default_serve_command(VlmConfig(model_id="org/model", max_model_len=4096))

    assert cmd.startswith("vllm serve org/model ")
    assert "--host 127.0.0.1" in cmd
    assert "--max-model-len 4096" in cmd
    assert "--media-io-kwargs" in cmd
    assert '"num_frames":-1' in cmd


def test_default_serve_command_uses_transformers_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "lerobot_align.vlm_client.shutil.which",
        lambda executable: "/usr/bin/transformers" if executable == "transformers" else None,
    )

    cmd = _default_serve_command(VlmConfig(model_id="org/model", serve_port=8123))

    assert cmd == "transformers serve org/model --host 127.0.0.1 --port 8123 --continuous-batching"


def test_default_serve_command_reports_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lerobot_align.vlm_client.shutil.which", lambda _executable: None)

    with pytest.raises(RuntimeError, match=r"lerobot-align\[serve\]"):
        _default_serve_command(VlmConfig())


def test_openai_transport_inlines_local_video_without_changing_bytes(tmp_path) -> None:
    clip_path = tmp_path / "native-clip.mp4"
    expected = b"\x00\x00\x00\x18ftypmp42\x00\xffvideo-payload"
    clip_path.write_bytes(expected)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the clip."},
                {
                    "type": "video_url",
                    "video_url": {"url": clip_path.as_uri()},
                    "fps": 3.0,
                },
            ],
        }
    ]

    api_messages, mm_kwargs = _to_openai_messages(messages)

    video_url = api_messages[0]["content"][1]["video_url"]["url"]
    prefix = "data:video/mp4;base64,"
    assert video_url.startswith(prefix)
    assert base64.b64decode(video_url.removeprefix(prefix)) == expected
    assert mm_kwargs == {"fps": 3.0}


def test_default_auto_served_vllm_forwards_video_processor_fps(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict] = []

    class FakeOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (
                        calls.append(kwargs)
                        or SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))]
                        )
                    )
                )
            )

    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setattr(vlm_module, "_server_is_up", lambda _url: False)
    monkeypatch.setattr(
        vlm_module,
        "_spawn_inference_server",
        lambda config: config.api_base,
    )
    monkeypatch.setattr(
        vlm_module.shutil,
        "which",
        lambda executable: "/usr/bin/vllm" if executable == "vllm" else None,
    )
    monkeypatch.delenv("LEROBOT_OPENAI_SEND_MM_KWARGS", raising=False)
    from PIL import Image
    from lerobot_align.frames import encode_frames_to_clip

    clip_path = tmp_path / "clip with spaces.mp4"
    encode_frames_to_clip([Image.new("RGB", (56, 56)) for _ in range(7)],
                          [index / 3 for index in range(7)], clip_path)
    client = vlm_module._make_openai_client(VlmConfig(
        auto_serve=True, client_concurrency=1, video_metadata_source="client",
    ))

    result = client.generate_json(
        [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video_url",
                            "video_url": {"url": clip_path.as_uri()},
                            "fps": 3.0,
                        }
                    ],
                }
            ]
        ]
    )

    assert result == [{"ok": True}]
    assert calls[0]["extra_body"]["mm_processor_kwargs"] == {
        "fps": 3.0,
        "do_sample_frames": False,
        "video_metadata": [{"total_num_frames": 7, "fps": 3.0, "frames_indices": list(range(7))}],
    }

    # Exercise the real processor boundary that failed in the canonical vLLM
    # container, including the temporal rate used by Qwen2.5's time positions.
    import numpy as np
    from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

    processor = Qwen2VLVideoProcessor(size={"shortest_edge": 56 * 56, "longest_edge": 56 * 56})
    processed = processor(videos=[np.zeros((7, 56, 56, 3), dtype=np.uint8)],
                          **calls[0]["extra_body"]["mm_processor_kwargs"], return_tensors="pt",
                          return_metadata=True)
    metadata = processed["video_metadata"][0]
    assert metadata.sampled_fps == 3.0
    assert metadata.timestamps == [index / 3 for index in range(7)]

    _, server_kwargs = _to_openai_messages([{"role": "user", "content": [{
        "type": "video_url", "video_url": {"url": clip_path.as_uri()},
    }]}], video_metadata="server")
    # Qwen3's vLLM adapter supplies video_metadata in its data argument. Sending
    # a second copy in kwargs raises TypeError before the processor can run.
    assert "video_metadata" not in server_kwargs
    assert server_kwargs["do_sample_frames"] is False
