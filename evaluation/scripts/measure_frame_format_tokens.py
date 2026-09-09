#!/usr/bin/env python3
"""Measure Qwen visual tokens for matched contact-sheet and video inputs.

This is a metadata-only population audit.  It reproduces the evaluation
sampler and image geometry, then calls the resizing functions shipped with the
installed Qwen processor.  It never decodes media, loads model weights, or
uses server-reported token counters.

The positional inputs are final prepared corpus directories.  Each must carry
``study.json``, ``preparation_plan.json``, ``group_meta.json``, and the
``components/`` and ``splits/`` trees produced by the unified protocol.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version as package_version
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any


SCHEMA_VERSION = 1
CAMERA_DEFAULT = "observation.images.wrist"
SAMPLING_FPS = 2.0
MAX_FRAMES = 300
FRAME_WIDTH = 224
CONTACT_SHEET_COLUMNS = 5
CONTACT_SHEET_ROWS = 4
FRAMES_PER_SHEET = 20
PROCESSOR_DEFAULT = "Qwen/Qwen3.8-27B"
PROCESSOR_FILES = (
    "config.json",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "tokenizer_config.json",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute exact grid-derived Qwen visual-token counts for matched contact-sheet "
            "and native-video representations over final prepared evaluation corpora."
        )
    )
    parser.add_argument(
        "corpus_dirs",
        type=Path,
        nargs="+",
        help="one or more final prepared corpus directories",
    )
    parser.add_argument(
        "--processor",
        default=PROCESSOR_DEFAULT,
        help="official Qwen processor Hub ID or local snapshot path",
    )
    parser.add_argument(
        "--camera",
        default=CAMERA_DEFAULT,
        help=f"video feature whose source geometry is measured (default: {CAMERA_DEFAULT})",
    )
    parser.add_argument("--rows-out", type=Path, required=True, help="output JSONL rows")
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=None,
        help="summary/provenance JSON (default: <rows-out stem>.summary.json)",
    )
    parser.add_argument(
        "--actual-pairs",
        type=Path,
        default=None,
        help=(
            "optional JSON/JSONL processor outputs to verify against the analytic grids; "
            "records are keyed by corpus/component/episode"
        ),
    )
    parser.add_argument(
        "--local-files-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="forbid processor downloads (default: true)",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON from {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def size_edge(size: object, name: str) -> int:
    value = getattr(size, name, None)
    if value is None and isinstance(size, Mapping):
        value = size.get(name)
    if value is None:
        raise ValueError(f"processor size has no {name!r}: {size!r}")
    return int(value)


def processor_provenance(source: str, local_files_only: bool) -> tuple[dict[str, Any], Any]:
    try:
        import transformers
        from transformers import AutoProcessor
        from transformers.utils.hub import cached_file
    except ImportError as exc:
        raise ValueError(
            "transformers with Qwen3-VL support is required; run this script in the project venv"
        ) from exc

    processor = AutoProcessor.from_pretrained(source, local_files_only=local_files_only)
    image = processor.image_processor
    video = processor.video_processor
    required = ("patch_size", "merge_size")
    if any(not hasattr(image, name) for name in required) or any(
        not hasattr(video, name) for name in (*required, "temporal_patch_size")
    ):
        raise ValueError(f"{source!r} did not load a compatible Qwen vision processor")

    resolved_files: dict[str, str] = {}
    resolved_revisions: set[str] = set()
    source_path = Path(source).expanduser()
    for filename in PROCESSOR_FILES:
        candidate: Path | None
        if source_path.is_dir():
            candidate = source_path / filename
        else:
            try:
                cached = cached_file(source, filename, local_files_only=local_files_only)
            except (OSError, ValueError):
                cached = None
            candidate = Path(cached) if cached else None
        if candidate is not None and candidate.is_file():
            resolved_files[filename] = sha256_file(candidate)
            # Keep the cache symlink path here: resolving it points into
            # ``blobs/`` and discards the pinned ``snapshots/<revision>``
            # segment that provides the model revision.
            resolved = candidate.absolute()
            try:
                snapshot_index = resolved.parts.index("snapshots")
            except ValueError:
                pass
            else:
                if snapshot_index + 1 < len(resolved.parts):
                    resolved_revisions.add(resolved.parts[snapshot_index + 1])

    if len(resolved_revisions) > 1:
        raise ValueError(
            f"processor files resolved from multiple cached revisions: {sorted(resolved_revisions)}"
        )

    image_config = {
        "class": type(image).__name__,
        "patch_size": int(image.patch_size),
        "temporal_patch_size": int(getattr(image, "temporal_patch_size", 2)),
        "merge_size": int(image.merge_size),
        "shortest_edge": size_edge(image.size, "shortest_edge"),
        "longest_edge": size_edge(image.size, "longest_edge"),
    }
    video_config = {
        "class": type(video).__name__,
        "patch_size": int(video.patch_size),
        "temporal_patch_size": int(video.temporal_patch_size),
        "merge_size": int(video.merge_size),
        "shortest_edge": size_edge(video.size, "shortest_edge"),
        "longest_edge": size_edge(video.size, "longest_edge"),
        "default_fps": float(getattr(video, "fps", math.nan)),
        "default_do_sample_frames": bool(getattr(video, "do_sample_frames", True)),
        "min_frames": int(getattr(video, "min_frames", 0)),
        "max_frames": int(getattr(video, "max_frames", 0)),
    }
    tokenizer = processor.tokenizer
    provenance = {
        "requested_source": source,
        "resolved_name_or_path": str(getattr(processor, "name_or_path", source)),
        "revision": next(iter(resolved_revisions), None),
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "transformers_version": transformers.__version__,
        "image": image_config,
        "video": video_config,
        "source_files_sha256": resolved_files,
        "configuration_sha256": sha256_json(
            {"image": image_config, "video": video_config, "files": resolved_files}
        ),
    }
    return provenance, processor


def frame_shape(feature: Mapping[str, Any], camera: str) -> tuple[int, int]:
    camera_feature = feature.get(camera)
    if not isinstance(camera_feature, Mapping) or camera_feature.get("dtype") != "video":
        raise ValueError(f"camera {camera!r} is not a video feature")
    shape = camera_feature.get("shape")
    if not isinstance(shape, list) or len(shape) < 2:
        raise ValueError(f"camera {camera!r} has invalid shape {shape!r}")
    names = camera_feature.get("names")
    if isinstance(names, list) and "height" in names and "width" in names:
        height = shape[names.index("height")]
        width = shape[names.index("width")]
    else:
        height, width = shape[:2]
    height, width = int(height), int(width)
    if height <= 0 or width <= 0:
        raise ValueError(f"camera {camera!r} has non-positive shape {shape!r}")
    return height, width


def read_episode_metadata(component_dir: Path) -> dict[int, dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("pyarrow is required to read prepared episode metadata") from exc

    paths = sorted((component_dir / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise ValueError(f"no episode metadata parquet found under {component_dir}")
    wanted = [
        "episode_index",
        "length",
        "stats/timestamp/min",
        "stats/timestamp/max",
    ]
    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        schema_names = set(pq.read_schema(path).names)
        missing = set(wanted) - schema_names
        if missing:
            raise ValueError(f"{path} is missing episode metadata columns {sorted(missing)}")
        for row in pq.read_table(path, columns=wanted).to_pylist():
            episode = int(row["episode_index"])
            if episode in rows:
                raise ValueError(f"duplicate episode {episode} in {component_dir}/meta/episodes")
            rows[episode] = row
    return rows


def scalar_stat(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {key}: {value!r}")
    return result


def grid_tokens(grid: Sequence[int], merge_size: int) -> int:
    if len(grid) != 3 or any(int(value) <= 0 for value in grid):
        raise ValueError(f"invalid grid_thw {grid!r}")
    patches = math.prod(int(value) for value in grid)
    divisor = merge_size**2
    if patches % divisor:
        raise ValueError(f"grid {grid!r} is not divisible by merge_size**2={divisor}")
    return patches // divisor


def analytic_counts(
    *,
    duration_s: float,
    source_height: int,
    source_width: int,
    processor: Any,
) -> dict[str, Any]:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        smart_resize as image_smart_resize,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        smart_resize as video_smart_resize,
    )

    n_frames = min(int(round(duration_s * SAMPLING_FPS)) + 1, MAX_FRAMES)
    if n_frames <= 0:
        raise ValueError(f"sampling produced {n_frames} frames for duration {duration_s}")
    frame_height = max(1, round(source_height * FRAME_WIDTH / source_width))

    sheet_count = math.ceil(n_frames / FRAMES_PER_SHEET)
    sheet_height = frame_height * CONTACT_SHEET_ROWS
    sheet_width = FRAME_WIDTH * CONTACT_SHEET_COLUMNS
    image = processor.image_processor
    image_factor = int(image.patch_size) * int(image.merge_size)
    resized_sheet_height, resized_sheet_width = image_smart_resize(
        sheet_height,
        sheet_width,
        factor=image_factor,
        min_pixels=size_edge(image.size, "shortest_edge"),
        max_pixels=size_edge(image.size, "longest_edge"),
    )
    sheet_grid = [
        1,
        resized_sheet_height // int(image.patch_size),
        resized_sheet_width // int(image.patch_size),
    ]
    sheet_tokens_each = grid_tokens(sheet_grid, int(image.merge_size))
    sheet_tokens = sheet_count * sheet_tokens_each

    # encode_frames_to_clip crops H.264 dimensions to even values before Qwen
    # receives them.  FRAME_WIDTH is even, but keep this explicit so an odd
    # source aspect ratio is reproduced exactly.
    video_input_height = frame_height - frame_height % 2
    video_input_width = FRAME_WIDTH - FRAME_WIDTH % 2
    if video_input_height < 2 or video_input_width < 2:
        raise ValueError(
            f"video input would be too small: {video_input_height}x{video_input_width}"
        )
    video = processor.video_processor
    video_factor = int(video.patch_size) * int(video.merge_size)
    resized_video_height, resized_video_width = video_smart_resize(
        num_frames=n_frames,
        height=video_input_height,
        width=video_input_width,
        temporal_factor=int(video.temporal_patch_size),
        factor=video_factor,
        min_pixels=size_edge(video.size, "shortest_edge"),
        max_pixels=size_edge(video.size, "longest_edge"),
    )
    video_grid = [
        math.ceil(n_frames / int(video.temporal_patch_size)),
        resized_video_height // int(video.patch_size),
        resized_video_width // int(video.patch_size),
    ]
    video_tokens = grid_tokens(video_grid, int(video.merge_size))
    if sheet_tokens <= 0 or video_tokens <= 0:
        raise ValueError("processor produced a non-positive visual-token count")

    return {
        "sampling_fps": SAMPLING_FPS,
        "max_frames": MAX_FRAMES,
        "n_sampled_frames": n_frames,
        "n_frames": n_frames,
        "frame_width": FRAME_WIDTH,
        "frame_height": frame_height,
        "contact_sheet_columns": CONTACT_SHEET_COLUMNS,
        "contact_sheet_rows": CONTACT_SHEET_ROWS,
        "frames_per_sheet": FRAMES_PER_SHEET,
        "contact_sheet_count": sheet_count,
        "contact_sheet_canvas_height": sheet_height,
        "contact_sheet_canvas_width": sheet_width,
        "contact_sheet_resized_height": resized_sheet_height,
        "contact_sheet_resized_width": resized_sheet_width,
        "contact_sheet_grid_thw": sheet_grid,
        "contact_sheet_visual_tokens_per_sheet": sheet_tokens_each,
        "contact_sheet_visual_tokens": sheet_tokens,
        "video_input_height": video_input_height,
        "video_input_width": video_input_width,
        "video_resized_height": resized_video_height,
        "video_resized_width": resized_video_width,
        "video_grid_thw": video_grid,
        "video_visual_tokens": video_tokens,
        "same_frames": True,
        "video_to_contact_ratio": video_tokens / sheet_tokens,
        "contact_to_video_ratio": sheet_tokens / video_tokens,
        "video_reduction_fraction": 1.0 - video_tokens / sheet_tokens,
    }


def component_rows(
    root: Path,
    *,
    camera: str,
    processor: Any,
    corpus: str,
    identity_namespace: str,
    plan: Mapping[str, Any],
    group_meta: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_paths = sorted((root / "splits").glob("*.json"))
    if not split_paths:
        raise ValueError(f"{root} contains no component splits")

    output: list[dict[str, Any]] = []
    task_episode_sets: dict[str, set[int]] = defaultdict(set)
    component_counts: dict[str, int] = {}
    for split_path in split_paths:
        component = split_path.stem
        meta = group_meta.get(component)
        if not isinstance(meta, Mapping):
            raise ValueError(f"{root}/group_meta.json has no record for {component}")
        component_dir = root / "components" / component
        index_map_path = root / "components" / f"{component}.index_map.json"
        if not component_dir.is_dir() or not index_map_path.is_file():
            raise ValueError(f"prepared component or index map is missing for {component}")

        split = load_json(split_path)
        if split.get("dataset") != component:
            raise ValueError(f"{split_path}: dataset does not match filename")
        seed = {int(value) for value in split.get("seed", [])}
        evaluation = [int(value) for value in split.get("eval", [])]
        if len(evaluation) != len(set(evaluation)):
            raise ValueError(f"{split_path}: duplicate eval episode")
        overlap = seed & set(evaluation)
        if overlap:
            raise ValueError(f"{split_path}: seed/eval overlap {sorted(overlap)}")

        index_map = load_json(index_map_path)
        if index_map.get("identity_namespace") != identity_namespace:
            raise ValueError(f"{index_map_path}: identity namespace mismatch")
        local_to_source = {
            int(item["new_index"]): int(item["original_index"])
            for item in index_map.get("episodes", [])
        }
        if len(local_to_source) != len(index_map.get("episodes", [])):
            raise ValueError(f"{index_map_path}: duplicate new_index")
        episode_metadata = read_episode_metadata(component_dir)
        info = load_json(component_dir / "meta" / "info.json")
        source_fps = float(info.get("fps", 0))
        if not math.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"{component}: invalid source fps {source_fps!r}")
        source_height, source_width = frame_shape(info.get("features", {}), camera)
        task = str(meta.get("task", ""))
        source = str(meta.get("source", ""))
        if not task or not source:
            raise ValueError(f"{component}: group metadata lacks task/source")

        for episode in evaluation:
            if episode not in local_to_source or episode not in episode_metadata:
                raise ValueError(f"{component}: eval episode {episode} lacks index or metadata")
            source_episode = local_to_source[episode]
            plan_episode = plan.get("episodes", {}).get(str(source_episode))
            if not isinstance(plan_episode, Mapping):
                raise ValueError(f"{component}: source episode {source_episode} missing from plan")
            if plan_episode.get("task") != task or plan_episode.get("source") != source:
                raise ValueError(
                    f"{component}: source episode {source_episode} disagrees with group metadata"
                )
            metadata = episode_metadata[episode]
            timestamp_min = scalar_stat(metadata, "stats/timestamp/min")
            timestamp_max = scalar_stat(metadata, "stats/timestamp/max")
            duration_s = timestamp_max - timestamp_min
            if duration_s < 0:
                raise ValueError(f"{component} episode {episode}: negative duration")
            length = int(metadata["length"])
            expected_duration = (length - 1) / source_fps if length > 0 else 0.0
            if not math.isclose(duration_s, expected_duration, abs_tol=max(1e-4, 1 / source_fps)):
                raise ValueError(
                    f"{component} episode {episode}: timestamp duration {duration_s} disagrees "
                    f"with length/fps duration {expected_duration}"
                )
            counts = analytic_counts(
                duration_s=duration_s,
                source_height=source_height,
                source_width=source_width,
                processor=processor,
            )
            if int(counts["n_sampled_frames"]) > length:
                raise ValueError(
                    f"{component} episode {episode}: analytic sampler requested "
                    f"{counts['n_sampled_frames']} frames from only {length} source frames"
                )
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "corpus": corpus,
                    "identity_namespace": identity_namespace,
                    "task": task,
                    "component": component,
                    "episode": episode,
                    "component_episode": episode,
                    "source": source,
                    "source_episode": source_episode,
                    "valid": True,
                    "duration_s": round(duration_s, 6),
                    "source_fps": source_fps,
                    "source_length": length,
                    "camera": camera,
                    "source_height": source_height,
                    "source_width": source_width,
                    **counts,
                }
            )
            task_episode_sets[task].add(source_episode)
        component_counts[component] = len(evaluation)

    task_splits = load_json(root / "task_splits.json")
    expected_task_splits = task_splits.get("tasks", {})
    if not isinstance(expected_task_splits, Mapping):
        raise ValueError(f"{root}/task_splits.json: tasks must be an object")
    if set(task_episode_sets) != set(expected_task_splits):
        raise ValueError(f"{root}: component tasks do not equal task_splits tasks")
    for task, actual in task_episode_sets.items():
        expected_values = [int(value) for value in expected_task_splits[task].get("eval", [])]
        if len(expected_values) != len(set(expected_values)):
            raise ValueError(f"{root}/task_splits.json: duplicate eval episode for task {task}")
        expected = set(expected_values)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                f"{root}: task {task} component population mismatch; "
                f"missing={missing[:10]} extra={extra[:10]}"
            )
    audit = {
        "components": len(component_counts),
        "tasks": len(task_episode_sets),
        "eval_episodes": len(output),
        "component_eval_counts": component_counts,
        "task_eval_counts": {key: len(value) for key, value in sorted(task_episode_sets.items())},
    }
    return output, audit


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate zero rows")
    contact = sum(int(row["contact_sheet_visual_tokens"]) for row in rows)
    video = sum(int(row["video_visual_tokens"]) for row in rows)
    episode_ratios = [float(row["video_to_contact_ratio"]) for row in rows]
    episode_reductions = [float(row["video_reduction_fraction"]) for row in rows]
    return {
        "n_episodes": len(rows),
        "n_components": len({str(row["component"]) for row in rows}),
        "n_tasks": len({(str(row["corpus"]), str(row["task"])) for row in rows}),
        "contact_sheet_visual_tokens": contact,
        "video_visual_tokens": video,
        "video_to_contact_ratio": video / contact,
        "contact_to_video_ratio": contact / video,
        "video_reduction_fraction": 1.0 - video / contact,
        "median_episode_video_to_contact_ratio": statistics.median(episode_ratios),
        "median_episode_video_reduction_fraction": statistics.median(episode_reductions),
        "min_episode_video_reduction_fraction": min(episode_reductions),
        "max_episode_video_reduction_fraction": max(episode_reductions),
    }


def grouped_aggregates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_corpus: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_task: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        corpus, task = str(row["corpus"]), str(row["task"])
        by_corpus[corpus].append(row)
        by_task[corpus][task].append(row)
    return {
        "all": aggregate(rows),
        "corpora": {key: aggregate(value) for key, value in sorted(by_corpus.items())},
        "tasks": {
            corpus: {task: aggregate(group) for task, group in sorted(tasks.items())}
            for corpus, tasks in sorted(by_task.items())
        },
    }


def load_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record is not an object") from None
            records.append(record)
        return records
    if isinstance(value, dict):
        value = value.get("rows", value.get("pairs"))
    if not isinstance(value, list) or any(not isinstance(record, dict) for record in value):
        raise ValueError(f"{path}: expected a JSON array, rows/pairs object, or JSONL objects")
    return value


def normalize_grids(value: Any, field: str) -> list[list[int]]:
    if (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(item, (int, float)) for item in value)
    ):
        value = [value]
    if not isinstance(value, list) or any(
        not isinstance(grid, list) or len(grid) != 3 for grid in value
    ):
        raise ValueError(f"{field} must be one grid or a list of grids")
    normalized: list[list[int]] = []
    for grid in value:
        if any(
            not isinstance(item, (int, float))
            or not math.isfinite(item)
            or item <= 0
            or int(item) != item
            for item in grid
        ):
            raise ValueError(f"{field} contains a non-positive or non-integral grid: {grid!r}")
        normalized.append([int(item) for item in grid])
    return normalized


def validate_actual_pairs(
    rows: Sequence[Mapping[str, Any]], actual_path: Path | None
) -> dict[str, Any]:
    if actual_path is None:
        return {"status": "not_requested", "pairs": 0}
    expected = {
        (str(row["corpus"]), str(row["component"]), int(row["episode"])): row for row in rows
    }
    seen: set[tuple[str, str, int]] = set()
    corpus_counts: dict[str, int] = defaultdict(int)
    records = load_records(actual_path)
    if not records:
        raise ValueError(f"{actual_path}: no actual-pair records")
    for record in records:
        missing = [field for field in ("corpus", "component", "episode") if field not in record]
        if missing:
            raise ValueError(f"{actual_path}: actual-pair record lacks {missing}")
        try:
            episode = int(record["episode"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{actual_path}: invalid actual-pair episode {record['episode']!r}"
            ) from exc
        key = (str(record["corpus"]), str(record["component"]), episode)
        if key in seen:
            raise ValueError(f"{actual_path}: duplicate actual-pair identity {key}")
        seen.add(key)
        row = expected.get(key)
        if row is None:
            raise ValueError(f"{actual_path}: actual pair does not match an analytic row: {key}")
        contact_sheet = record.get("contact_sheet")
        video = record.get("video")
        image_value = record.get(
            "contact_sheet_grids_thw",
            record.get(
                "image_grid_thw",
                contact_sheet.get("grid_thw") if isinstance(contact_sheet, Mapping) else None,
            ),
        )
        video_value = record.get(
            "video_grids_thw",
            record.get(
                "video_grid_thw",
                video.get("grid_thw") if isinstance(video, Mapping) else None,
            ),
        )
        actual_image = normalize_grids(image_value, "image_grid_thw")
        actual_video = normalize_grids(video_value, "video_grid_thw")
        expected_image = [row["contact_sheet_grid_thw"]] * int(row["contact_sheet_count"])
        expected_video = [row["video_grid_thw"]]
        if actual_image != expected_image or actual_video != expected_video:
            raise ValueError(
                f"{actual_path}: grid mismatch for {key}: image={actual_image} "
                f"expected={expected_image}; video={actual_video} expected={expected_video}"
            )
        corpus_counts[key[0]] += 1
    return {
        "status": "passed",
        "pairs": len(records),
        "corpora": dict(sorted(corpus_counts.items())),
        "path": str(actual_path),
        "sha256": sha256_file(actual_path),
    }


def validate_identities(rows: Sequence[Mapping[str, Any]]) -> None:
    local: set[tuple[str, str, int]] = set()
    source: set[tuple[str, int]] = set()
    for row in rows:
        local_key = (str(row["corpus"]), str(row["component"]), int(row["episode"]))
        source_key = (str(row["identity_namespace"]), int(row["source_episode"]))
        if local_key in local:
            raise ValueError(f"duplicate output row identity {local_key}")
        if source_key in source:
            raise ValueError(f"source episode appears in more than one component: {source_key}")
        local.add(local_key)
        source.add(source_key)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary_out = args.summary_out or args.rows_out.with_suffix(".summary.json")
    if args.rows_out.resolve() == summary_out.resolve():
        raise ValueError("--rows-out and --summary-out must be different files")

    processor_info, processor = processor_provenance(args.processor, args.local_files_only)
    all_rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    corpus_names: set[str] = set()
    for raw_root in args.corpus_dirs:
        root = raw_root.expanduser().resolve()
        required = (
            "study.json",
            "preparation_plan.json",
            "group_meta.json",
            "task_splits.json",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ValueError(f"{root}: not a final prepared corpus; missing {missing}")
        study = load_json(root / "study.json")
        plan = load_json(root / "preparation_plan.json")
        group_meta = load_json(root / "group_meta.json")
        corpus = str(study.get("name", ""))
        identity_namespace = str(study.get("identity_namespace", ""))
        if not corpus or corpus in corpus_names:
            raise ValueError(f"duplicate or empty corpus name {corpus!r}")
        if plan.get("name") != corpus or plan.get("identity_namespace") != identity_namespace:
            raise ValueError(f"{root}: study and preparation plan identity disagree")
        corpus_names.add(corpus)
        rows, audit = component_rows(
            root,
            camera=args.camera,
            processor=processor,
            corpus=corpus,
            identity_namespace=identity_namespace,
            plan=plan,
            group_meta=group_meta,
        )
        all_rows.extend(rows)
        inputs.append(
            {
                "corpus": corpus,
                "prepared_root": raw_root.name,
                "identity_namespace": identity_namespace,
                "study_sha256": sha256_file(root / "study.json"),
                "preparation_plan_sha256": sha256_file(root / "preparation_plan.json"),
                "group_meta_sha256": sha256_file(root / "group_meta.json"),
                "task_splits_sha256": sha256_file(root / "task_splits.json"),
                **audit,
            }
        )

    validate_identities(all_rows)
    all_rows.sort(key=lambda row: (row["corpus"], row["task"], row["component"], row["episode"]))
    actual_validation = validate_actual_pairs(all_rows, args.actual_pairs)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "method": (
            "Complete prepared split evaluation population; matched frame identities; "
            "grid-derived Qwen visual tokens; no model inference, pixel decoding, or server counters."
        ),
        "protocol": {
            "sampling_fps": SAMPLING_FPS,
            "max_frames": MAX_FRAMES,
            "frame_count_equation": "min(round(duration_s * 2) + 1, 300)",
            "frame_width": FRAME_WIDTH,
            "contact_sheet_columns": CONTACT_SHEET_COLUMNS,
            "contact_sheet_rows": CONTACT_SHEET_ROWS,
            "frames_per_sheet": FRAMES_PER_SHEET,
            "contact_sheet_partial_policy": "full 5x4 canvas padded with black cells",
            "video_frame_policy": "same sampled frames, temporal padding only",
            "camera": args.camera,
        },
        "processor": processor_info,
        "runtime": {
            "python": sys.version.split()[0],
            "pyarrow": package_version("pyarrow"),
        },
        "inputs": inputs,
        "population_validation": {
            "status": "passed",
            "rows": len(all_rows),
            "unique_local_identities": len(all_rows),
            "unique_source_identities": len(all_rows),
        },
        "actual_pair_validation": actual_validation,
        "aggregates": grouped_aggregates(all_rows),
    }
    rows_text = "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in all_rows)
    atomic_write_text(args.rows_out, rows_text)
    atomic_write_text(
        summary_out, json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(f"wrote {len(all_rows)} rows to {args.rows_out}")
    print(f"wrote summary and provenance to {summary_out}")
    for corpus, result in summary["aggregates"]["corpora"].items():
        print(
            f"{corpus}: n={result['n_episodes']} video/contact="
            f"{result['video_to_contact_ratio']:.4f}, visual-token reduction="
            f"{result['video_reduction_fraction']:.1%}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
