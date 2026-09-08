"""Evaluation run identity, independent of local paths and credentials."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
from urllib.parse import urlsplit

from lerobot_align.config import PlanConfig, VlmConfig
from lerobot_align.prompts import load as load_prompt


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fingerprint(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def evaluation_config(args, cameras: list[str], root: Path) -> dict:
    """Record exact local input bytes, code/prompts, effective settings and runtime versions."""
    package = Path(__file__).resolve().parents[1]
    code = {
        str(path.relative_to(package)): digest(path)
        for path in sorted(package.rglob("*"))
        if path.suffix in {".py", ".txt"}
    }
    sources = sorted(
        {
            *root.glob("meta/*.json"),
            *root.glob("meta/*.parquet"),
            *root.glob("meta/episodes/**/*.parquet"),
            *root.glob("data/**/*.parquet"),
            *root.glob("videos/**/*.mp4"),
        }
    )
    source_identity = {
        str(path.relative_to(root)): digest(path) for path in sources if path.is_file()
    }
    settings = {
        name: getattr(args, name, None)
        for name in (
            "model",
            "fps",
            "max_frames",
            "frame_width",
            "temperature",
            "sampling",
            "video_fallback",
            "video_prompt",
            "processor_fps",
        )
    }
    settings["cameras"] = cameras
    settings["video_metadata_source"] = os.environ.get("LEROBOT_VLM_VIDEO_METADATA_SOURCE", "server")
    settings["send_mm_kwargs"] = os.environ.get("LEROBOT_OPENAI_SEND_MM_KWARGS", "").lower() in {
        "1",
        "yes",
        "true",
    }
    settings["calibration_sha256"] = (
        digest(args.calibration) if getattr(args, "calibration", None) else None
    )
    # A model alias can name different deployments. Keep endpoint identity
    # without publishing private hostnames or including authentication details.
    endpoint = urlsplit(args.api_base)
    settings["endpoint_sha256"] = fingerprint({
        "scheme": endpoint.scheme,
        "host": endpoint.hostname,
        "port": endpoint.port,
        "path": endpoint.path.rstrip("/"),
    })
    # The loader supports environment overrides, including an alternate named
    # video prompt. Hash the text actually loaded, not only packaged files.
    prompt_names = {p.stem for p in (package / "prompts").glob("*.txt")}
    if getattr(args, "video_prompt", None):
        prompt_names.add(args.video_prompt)
    effective_prompts = {
        name: hashlib.sha256(load_prompt(name).encode()).hexdigest()
        for name in sorted(prompt_names)
    }
    return {
        "schema_version": 1,
        "settings": settings,
        "dataset_sha256": fingerprint(source_identity),
        "ground_truth_sha256": digest(root / "meta/lerobot_annotations.json"),
        "code_sha256": fingerprint(code),
        "effective_prompts_sha256": fingerprint(effective_prompts),
        "versions": {
            name: version(name)
            for name in ("lerobot-align", "lerobot", "openai", "av", "pillow", "pyarrow")
        },
        "plan_defaults": {
            key: str(val) if isinstance(val, Path) else val
            for key, val in asdict(PlanConfig()).items()
        },
    }


def validate_options(args, parser) -> None:
    """Reject invalid CLI numerics before a decoder, model client or output is opened."""
    try:
        PlanConfig(
            frames_per_second=args.fps,
            max_frames_per_prompt=args.max_frames,
            contact_sheet_frame_width=args.frame_width,
        )
        VlmConfig(temperature=args.temperature)
        fps = getattr(args, "processor_fps", None)
        if fps is not None and (not math.isfinite(fps) or fps <= 0):
            raise ValueError("--processor-fps must be positive and finite")
        episodes = getattr(args, "episodes", None)
        if episodes is not None and (
            not episodes or any(e < 0 for e in episodes) or len(set(episodes)) != len(episodes)
        ):
            raise ValueError("--episodes must be non-empty, unique, non-negative IDs")
        if getattr(args, "episode", 0) < 0:
            raise ValueError("--episode must be non-negative")
    except ValueError as exc:
        parser.error(str(exc))


def api_key(args, parser) -> str:
    name = args.api_key_env
    if name not in os.environ and name != "LEROBOT_VLM_API_KEY":
        parser.error(f"API key environment variable {name!r} is not set")
    return os.environ.get(name, "EMPTY")


def validate_truth(truth: dict) -> dict:
    for episode, spans in truth.items():
        if not isinstance(spans, list) or not spans:
            raise ValueError(f"episode {episode}: ground-truth subtasks must be a non-empty list")
        previous = -1.0
        for index, span in enumerate(spans):
            if (
                not isinstance(span, dict)
                or not isinstance(span.get("label"), str)
                or not span["label"].strip()
            ):
                raise ValueError(
                    f"episode {episode}: ground-truth subtask {index} needs a string label"
                )
            values = [span.get("start"), span.get("end")]
            if any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in values
            ):
                raise ValueError(
                    f"episode {episode}: ground-truth timestamps must be finite numbers"
                )
            start, end = values
            if start < 0 or end <= start or start < previous:
                raise ValueError(
                    f"episode {episode}: ground-truth intervals must be positive and ordered"
                )
            previous = end
    return truth
