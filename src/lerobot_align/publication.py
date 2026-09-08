"""Explicit, symlink-free dataset publication and source-card preservation."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from huggingface_hub import DatasetCard

from .config import AnnotationPipelineConfig

_SHARD = r"chunk-\d+/file-\d+\.parquet"
_METADATA = {"meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/subtasks.parquet"}
_NOTICES = {"LICENSE", "LICENSE.md", "LICENSE.txt", "NOTICE", "NOTICE.md", "NOTICE.txt"}


def publication_manifest(root: Path, staging_dir: Path) -> tuple[Path, ...]:
    """Return only canonical data, declared video cameras, metadata and notices.

    Matching symlinks are rejected, including directory symlinks. Arbitrary
    JSON, hidden files, backups, caches and user-selected staging never enter
    the manifest, even when placed within data/ or videos/.
    """
    root = root.absolute()
    staging_dir = staging_dir.absolute()
    info = root / "meta/info.json"
    _regular_source(info, root)
    features = json.loads(info.read_text(encoding="utf-8")).get("features", {})
    cameras = {name for name, feature in features.items() if feature.get("dtype") == "video"}
    selected = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        video = re.fullmatch(r"videos/([^/]+)/chunk-\d+/file-\d+\.mp4", relative)
        allowed = (
            relative in _METADATA | _NOTICES
            or re.fullmatch(rf"data/{_SHARD}", relative)
            or re.fullmatch(rf"meta/episodes/{_SHARD}", relative)
            or (video and video.group(1) in cameras)
        )
        if not allowed:
            continue
        if path == staging_dir or staging_dir in path.parents:
            raise ValueError(f"Staging overlaps required dataset content: {path}")
        _regular_source(path, root)
        selected.append(path.relative_to(root))
    if not any(p.parts[0] == "data" for p in selected):
        raise ValueError("No canonical data/chunk-*/file-*.parquet files available for publication")
    return tuple(selected)


def _regular_source(path: Path, root: Path) -> None:
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError(f"Refusing to publish symlinked dataset content: {path}")
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Publication requires a regular file within the dataset: {path}")


def copy_publication_files(root: Path, destination: Path, manifest: tuple[Path, ...]) -> None:
    """Upload from a private snapshot, never from the user's working directory."""
    for relative in manifest:
        source = root / relative
        _regular_source(source, root.absolute())
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def source_dataset_card(root: Path, cfg: AnnotationPipelineConfig) -> DatasetCard:
    """Retain the complete source card; never guess or change its license."""
    path = root / "README.md"
    if path.exists() or path.is_symlink():
        _regular_source(path, root.absolute())
        text = path.read_text(encoding="utf-8")
        card = DatasetCard(text)
        if cfg.dataset_license and card.data.get("license") not in (None, cfg.dataset_license):
            raise ValueError("--dataset_license conflicts with the source dataset card license")
    else:
        card = DatasetCard("# Annotated dataset\n")
    if not card.data.get("license"):
        if not cfg.dataset_license:
            raise ValueError(
                "Source dataset card has no license. Preserve/add its original license in README.md "
                "or supply --dataset_license after verifying the dataset's terms. "
                "The software license is never applied to data."
            )
        card.data.license = cfg.dataset_license
    # Preserve original citations, links, restrictions and custom YAML metadata.
    marker = "<!-- lerobot-align provenance -->"
    if marker not in card.text:
        card.text += (
            f"\n\n{marker}\n## Annotation provenance\n\n"
            "Language annotations were produced with lerobot-align. Original dataset "
            "license, attribution and usage restrictions above remain applicable.\n"
        )
        if cfg.repo_id:
            card.text += f"\nSource: https://huggingface.co/datasets/{cfg.repo_id}\n"
    return card
