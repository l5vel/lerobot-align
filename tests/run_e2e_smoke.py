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
"""CPU-only visual E2E smoke run.

Builds the shared annotation fixture (:func:`build_annotation_dataset`),
runs the full annotation pipeline against a localhost HTTP test double, and prints a
short report. Run it after installing the development dependencies with either
``uv run --no-sync python -m tests.run_e2e_smoke`` or
``uv run --no-sync python tests/run_e2e_smoke.py``. It exercises the installed CLI against a localhost HTTP test double with real
MP4 decoding and visual payload validation; no model or Hub access is required.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make direct file invocation resolve the local ``tests`` package just like
# ``python -m tests.run_e2e_smoke`` does.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import base64
import io
import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import av
from PIL import Image
import pyarrow.parquet as pq
from tests.fixtures import add_synthetic_video, build_annotation_dataset


def _stub_responder(messages):
    text = ""
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text = block.get("text", "")
            elif isinstance(content, str):
                text = content
    if "OBSERVATION pass" in text:
        return {
            "description": (
                "0.0-0.5s: grasp bottle; 0.5-1.0s: pour into cup; 1.0-1.5s: place bottle down"
            )
        }
    if "Reconstruct the sequence of COMPLETED manipulation events" in text:
        return {
            "subtasks": [
                {"text": "grasp the bottle", "start": 0.0, "end": 0.5},
                {"text": "pour into the cup", "start": 0.5, "end": 1.0},
                {"text": "place the bottle down", "start": 1.0, "end": 1.5},
            ]
        }
    if "compressed semantic memory" in text:
        return {"memory": "poured once"}
    if "acknowledgement the robot" in text:
        return {"text": "Sure."}
    if "compact interjection" in text:
        return {"interjection": "use less water", "speech": "Using less water."}
    if "frame-grounded visual question" in text:
        return {"question": "How many cups?", "answer": {"label": "cup", "count": 1}}
    raise AssertionError(f"smoke VLM has no canned response for prompt: {text[:160]!r}")


def main(*, cli_prefix: list[str] | None = None) -> int:
    observed = []
    errors = []

    class LocalModel(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                assert self.path == "/v1/chat/completions"
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert payload["model"] == "synthetic-smoke"
                blocks = [
                    b for m in payload["messages"] for b in m["content"] if isinstance(b, dict)
                ]
                prompt = "\n".join(b.get("text", "") for b in blocks)
                visual = [b for b in blocks if b["type"] in {"image_url", "video_url"}]
                if any(
                    marker in prompt
                    for marker in (
                        "OBSERVATION pass",
                        "COMPLETED manipulation events",
                        "compact interjection",
                        "frame-grounded visual question",
                    )
                ):
                    assert visual, "visual inference reached the HTTP endpoint without images/video"
                for block in visual:
                    kind = block["type"]
                    data = base64.b64decode(block[kind]["url"].split(",", 1)[1], validate=True)
                    if kind == "image_url":
                        Image.open(io.BytesIO(data)).verify()
                    else:
                        with av.open(io.BytesIO(data)) as clip:
                            assert len(list(clip.decode(video=0))) > 1
                        assert payload["mm_processor_kwargs"]["fps"] > 0
                        assert payload["mm_processor_kwargs"]["do_sample_frames"] is False
                        assert payload["mm_processor_kwargs"]["video_metadata"][0]["fps"] > 0
                    observed.append(kind)
                reply = _stub_responder(payload["messages"])
                body = json.dumps(
                    {
                        "id": "smoke",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "synthetic-smoke",
                        "choices": [
                            {
                                "index": 0,
                                "finish_reason": "stop",
                                "message": {"role": "assistant", "content": json.dumps(reply)},
                            }
                        ],
                    }
                ).encode()
                self.send_response(200)
            except Exception as exc:
                errors.append(str(exc))
                body = json.dumps({"error": str(exc)}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), LocalModel)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for frame_format in ("contact_sheet", "video"):
                root = add_synthetic_video(
                    build_annotation_dataset(
                        Path(tmp) / frame_format,
                        episode_specs=[(0, 30, "Pour water into the cup.")],
                        fps=10,
                    )
                )
                before = pq.read_table(next((root / "data").rglob("*.parquet")))
                command = [
                    *(cli_prefix or [str(Path(sys.executable).parent / "lerobot-align")]),
                    f"--root={root}",
                    "--vlm.model_id=synthetic-smoke",
                    "--vlm.auto_serve=false",
                    "--vlm.video_metadata_source=client",
                    f"--vlm.api_base=http://127.0.0.1:{server.server_port}/v1",
                    "--video_backend=pyav",
                    "--plan.n_task_rephrasings=0",
                    f"--plan.subtask_generate_frame_format={frame_format}",
                    "--plan.subtask_video_fallback=error",
                    "--interjections.max_interjections_per_episode=1",
                    "--interjections.interjection_min_t=0.5",
                    "--vqa.K=1",
                ]
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    env={
                        **os.environ,
                        "HF_HUB_OFFLINE": "1",
                        "HF_HOME": str(Path(tmp) / "hf"),
                        "HF_DATASETS_CACHE": str(Path(tmp) / "hf/datasets"),
                        "HF_LEROBOT_HOME": str(Path(tmp) / "hf/lerobot"),
                        "LEROBOT_OPENAI_SEND_MM_KWARGS": "1",
                    },
                )
                assert result.returncode == 0, result.stdout + result.stderr
                assert not errors, errors
                after = pq.read_table(next((root / "data").rglob("*.parquet")))
                assert after.num_rows == before.num_rows
                for key in ("episode_index", "frame_index", "timestamp", "task_index"):
                    assert after[key].equals(before[key]), key
                events = [atom for row in after["language_events"].to_pylist() for atom in row]
                assert any(atom.get("style") == "interjection" for atom in events)
                assert any(atom.get("style") == "vqa" for atom in events)
                assert any(atom.get("tool_calls") for atom in events)
                persistent = [
                    atom for row in after["language_persistent"].to_pylist() for atom in row
                ]
                assert {"subtask", "plan", "memory"}.issubset({a.get("style") for a in persistent})
                info = json.loads((root / "meta/info.json").read_text())
                assert {"language_persistent", "language_events"}.issubset(info["features"])
                assert not (root / ".lerobot-align-transaction").exists()
                print(
                    f"{frame_format}: real MP4 decoding, HTTP visual payloads, all modules and transactional write passed"
                )
        assert {"image_url", "video_url"}.issubset(observed), observed
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
