#!/usr/bin/env python3
"""Validate analytic Qwen visual-token counts on decoded evaluation media.

This is the deliberately small, expensive companion to
``measure_frame_format_tokens.py``.  It selects fixed quantiles of episode
length from each corpus, decodes the exact evaluation timestamps, constructs
both production frame formats, and runs the official processor.  When an API
base is supplied it also sends both payloads to a live Qwen endpoint.  Server
token telemetry is retained only as a diagnostic; cross-format claims use the
processor-expanded visual placeholders checked against ``grid_thw``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Any
import urllib.error
import urllib.request

from lerobot_align.frames import (
    encode_frames_to_clip,
    make_frame_provider,
    to_contact_sheet_blocks,
    to_video_url_block,
)
from lerobot_align.modules.plan_subtasks_memory import PlanSubtasksMemoryModule
from lerobot_align.reader import iter_episodes
from lerobot_align.vlm_client import _to_openai_messages


DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
PROMPT = "Reply with the single character 1."


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate paired visual-token counts on real evaluation media."
    )
    parser.add_argument("--rows", type=Path, required=True, help="census JSONL[.gz]")
    parser.add_argument(
        "--corpus-root",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="prepared corpus root; pass once per corpus",
    )
    parser.add_argument("--out", type=Path, required=True, help="validation JSON output")
    parser.add_argument("--processor", default=DEFAULT_MODEL)
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forbid processor downloads (default: true)",
    )
    parser.add_argument("--pairs-per-corpus", type=int, default=3)
    parser.add_argument(
        "--api-base",
        default=None,
        help="optional OpenAI-compatible base, for example http://127.0.0.1:8033/v1",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="served model name")
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=None,
        help="optional physical GPU index to record with nvidia-smi",
    )
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    rows: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    if not rows:
        raise ValueError(f"no rows in {path}")
    return rows


def parse_roots(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path:
            raise ValueError(f"invalid --corpus-root {value!r}; expected NAME=PATH")
        if name in result:
            raise ValueError(f"duplicate corpus root {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"corpus root is not a directory: {path}")
        result[name] = path
    return result


def select_length_quantiles(
    rows: Sequence[Mapping[str, Any]], count: int
) -> list[Mapping[str, Any]]:
    """Choose deterministic rank quantiles after sorting by sampled-frame count."""
    if count <= 0:
        raise ValueError("--pairs-per-corpus must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (
            int(row["n_sampled_frames"]),
            str(row["component"]),
            int(row["episode"]),
        ),
    )
    if count >= len(ordered):
        return ordered
    if count == 1:
        return [ordered[(len(ordered) - 1) // 2]]
    indices = [round(i * (len(ordered) - 1) / (count - 1)) for i in range(count)]
    return [ordered[index] for index in indices]


def visual_tokens(grids: Sequence[Sequence[int]], merge_size: int) -> int:
    divisor = merge_size**2
    total = 0
    for grid in grids:
        patches = math.prod(int(value) for value in grid)
        if patches <= 0 or patches % divisor:
            raise ValueError(f"invalid grid {grid!r} for merge_size={merge_size}")
        total += patches // divisor
    return total


def processor_result(batch: Mapping[str, Any], processor: Any, kind: str) -> dict[str, Any]:
    grid_key = "image_grid_thw" if kind == "contact_sheet" else "video_grid_thw"
    token_id = processor.image_token_id if kind == "contact_sheet" else processor.video_token_id
    grids = batch[grid_key].tolist()
    merge_size = int(
        processor.image_processor.merge_size
        if kind == "contact_sheet"
        else processor.video_processor.merge_size
    )
    input_ids = batch["input_ids"][0]
    placeholder_count = int((input_ids == token_id).sum().item())
    grid_count = visual_tokens(grids, merge_size)
    if placeholder_count != grid_count:
        raise ValueError(
            f"{kind}: input_ids contain {placeholder_count} visual placeholders, "
            f"but grid_thw expands to {grid_count}"
        )
    return {
        "grid_thw": grids,
        "visual_tokens": grid_count,
        "visual_placeholders_in_input_ids": placeholder_count,
        "total_input_ids": int(input_ids.numel()),
    }


def apply_processor(processor: Any, content: list[dict[str, Any]], *, video: bool) -> Any:
    messages = [{"role": "user", "content": [*content, {"type": "text", "text": PROMPT}]}]
    kwargs: dict[str, Any] = {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": True,
        "return_tensors": "pt",
    }
    if video:
        # These clips already contain the exact timestamps selected by the
        # evaluation sampler.  A second processor sampling pass would change
        # the experiment.
        kwargs["processor_kwargs"] = {"do_sample_frames": False}
    return processor.apply_chat_template(messages, **kwargs)


def api_request(
    api_base: str,
    model: str,
    content: list[dict[str, Any]],
    *,
    video: bool,
) -> dict[str, Any]:
    messages = [{"role": "user", "content": [*content, {"type": "text", "text": PROMPT}]}]
    api_messages, mm_kwargs = _to_openai_messages(messages)
    if video:
        mm_kwargs["do_sample_frames"] = False
    payload: dict[str, Any] = {
        "model": model,
        "messages": api_messages,
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if mm_kwargs:
        payload["mm_processor_kwargs"] = mm_kwargs
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"GPU request failed with HTTP {exc.code}: {detail[:1000]}") from exc
    return {
        "elapsed_s": round(time.monotonic() - started, 3),
        "finish_reason": result["choices"][0].get("finish_reason"),
        "text": result["choices"][0]["message"].get("content", ""),
        # Diagnostic only.  vLLM usage accounting is modality-dependent and
        # is explicitly excluded from the visual-token comparison.
        "server_usage_diagnostic_only": result.get("usage"),
    }


def get_json(url: str) -> Any:
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def server_provenance(api_base: str | None) -> dict[str, Any] | None:
    if not api_base:
        return None
    root = api_base.removesuffix("/v1").rstrip("/")
    models = get_json(api_base.rstrip("/") + "/models")
    model_rows = models.get("data", []) if isinstance(models, Mapping) else []
    public_models = [
        {key: row[key] for key in ("id", "root", "owned_by", "max_model_len") if key in row}
        for row in model_rows
        if isinstance(row, Mapping)
    ]
    return {
        "api_base": api_base,
        "version": get_json(root + "/version"),
        "models": {"data": public_models},
    }


def gpu_provenance(index: int | None) -> dict[str, Any] | None:
    if index is None:
        return None
    command = [
        "nvidia-smi",
        "-i",
        str(index),
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    parts = [part.strip() for part in completed.stdout.strip().split(",")]
    if len(parts) != 4:
        raise ValueError(f"unexpected nvidia-smi output: {completed.stdout!r}")
    return {
        "physical_index": int(parts[0]),
        "name": parts[1],
        "driver_version": parts[2],
        "memory_total_mib": int(parts[3]),
    }


def inspect_pair(
    row: Mapping[str, Any],
    root: Path,
    processor: Any,
    *,
    api_base: str | None,
    model: str,
) -> dict[str, Any]:
    component_root = root / "components" / str(row["component"])
    episode = int(row["episode"])
    records = list(iter_episodes(component_root, only_episodes=(episode,)))
    if len(records) != 1:
        raise ValueError(
            f"expected one record for {row['component']} episode {episode}, got {len(records)}"
        )
    record = records[0]
    requested = int(row["n_sampled_frames"])
    timestamps = PlanSubtasksMemoryModule._uniform_episode_timestamps(record, requested)
    if len(timestamps) != requested:
        raise ValueError(
            f"sampler returned {len(timestamps)} timestamps, census expected {requested}"
        )
    camera = str(row["camera"])
    provider = make_frame_provider(component_root, camera_key=camera)
    frames = provider.frames_at(record, timestamps, camera_key=camera, fail_on_error=True)
    if len(frames) != requested:
        raise ValueError(f"decoded {len(frames)} frames, expected {requested}")

    sheets = to_contact_sheet_blocks(
        frames,
        timestamps,
        columns=int(row["contact_sheet_columns"]),
        frames_per_sheet=int(row["frames_per_sheet"]),
        frame_width=int(row["frame_width"]),
        quality=84,
    )
    image_content = [{"type": "image", "image": block["image"]} for block in sheets]
    image_batch = apply_processor(processor, image_content, video=False)
    image_result = processor_result(image_batch, processor, "contact_sheet")

    with tempfile.TemporaryDirectory(prefix="qwen-frame-format-") as raw_temp:
        clip = Path(raw_temp) / "sampled.mp4"
        encoded_fps = encode_frames_to_clip(
            frames,
            timestamps,
            clip,
            frame_width=int(row["frame_width"]),
            crf=0,
        )
        if encoded_fps is None or not clip.is_file():
            raise ValueError("lossless video encoding failed")
        video_content_processor = [{"type": "video", "video": str(clip)}]
        video_batch = apply_processor(processor, video_content_processor, video=True)
        video_result = processor_result(video_batch, processor, "video")

        api_results = None
        if api_base:
            video_content_api = to_video_url_block(clip.as_uri(), fps=None)
            api_results = {
                "contact_sheet": api_request(api_base, model, image_content, video=False),
                "video": api_request(api_base, model, video_content_api, video=True),
            }
        clip_bytes = clip.stat().st_size

    expected_image_grids = [row["contact_sheet_grid_thw"]] * int(row["contact_sheet_count"])
    expected_video_grids = [row["video_grid_thw"]]
    checks = {
        "same_timestamp_count": len(timestamps) == requested,
        "decoded_every_timestamp": len(frames) == requested,
        "contact_sheet_grids_match_census": image_result["grid_thw"] == expected_image_grids,
        "video_grid_matches_census": video_result["grid_thw"] == expected_video_grids,
        "contact_sheet_tokens_match_census": image_result["visual_tokens"]
        == int(row["contact_sheet_visual_tokens"]),
        "video_tokens_match_census": video_result["visual_tokens"]
        == int(row["video_visual_tokens"]),
    }
    if not all(checks.values()):
        raise ValueError(
            f"processor validation failed for {row['corpus']}/{row['component']}/{episode}: "
            f"{checks}"
        )
    return {
        "corpus": row["corpus"],
        "task": row["task"],
        "component": row["component"],
        "episode": episode,
        "source_episode": row["source_episode"],
        "n_sampled_frames": requested,
        "duration_s": row["duration_s"],
        "timestamp_first": float(timestamps[0]),
        "timestamp_last": float(timestamps[-1]),
        "contact_sheet_count": len(sheets),
        "encoded_fps": encoded_fps,
        "lossless_clip_bytes": clip_bytes,
        "contact_sheet": image_result,
        "video": video_result,
        "checks": checks,
        "gpu3_inference": api_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    roots = parse_roots(args.corpus_root)
    rows = load_rows(args.rows)
    corpora = sorted({str(row["corpus"]) for row in rows})
    if set(corpora) != set(roots):
        raise ValueError(f"row corpora {corpora} do not match roots {sorted(roots)}")

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        args.processor,
        local_files_only=args.local_files_only,
    )
    selected = [
        row
        for corpus in corpora
        for row in select_length_quantiles(
            [candidate for candidate in rows if candidate["corpus"] == corpus],
            args.pairs_per_corpus,
        )
    ]
    results = [
        inspect_pair(
            row,
            roots[str(row["corpus"])],
            processor,
            api_base=args.api_base,
            model=args.model,
        )
        for row in selected
    ]
    output = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "selection": "deterministic equally spaced ranks after sorting by sampled-frame count",
        "pairs_per_corpus": args.pairs_per_corpus,
        "rows_path": str(args.rows),
        "rows_sha256": sha256_file(args.rows),
        "processor": args.processor,
        "processor_local_files_only": args.local_files_only,
        "prompt": PROMPT,
        "gpu_endpoint_used": bool(args.api_base),
        "server": server_provenance(args.api_base),
        "gpu": gpu_provenance(args.gpu_index),
        "server_usage_policy": (
            "diagnostic only; modality-dependent server counts are not compared or plotted"
        ),
        "pairs": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(f"validated {len(results)} decoded pairs; wrote {args.out}")
    for result in results:
        print(
            f"{result['corpus']} n={result['n_sampled_frames']}: "
            f"sheet={result['contact_sheet']['visual_tokens']} "
            f"video={result['video']['visual_tokens']}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
