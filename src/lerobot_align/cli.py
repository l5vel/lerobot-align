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
"""``lerobot-align`` — populate ``language_persistent`` and
``language_events`` columns on a LeRobot dataset.

Annotations live directly in ``data/chunk-*/file-*.parquet``.

Example:

  uv run lerobot-align \\
      --root=/path/to/dataset \\
      --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct

Pass ``--job.target=<flavor>`` to run the same command on a Hugging Face
Jobs GPU instead of this machine:

  uv run lerobot-align \\
      --repo_id=user/dataset \\
      --new_repo_id=user/dataset_annotated \\
      --push_to_hub=true \\
      --job.target=h200
"""

import logging
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError
from lerobot.configs import parser
from lerobot.utils.import_utils import _datasets_available, require_package

from lerobot_align.config import AnnotationPipelineConfig
from lerobot_align.executor import Executor, PipelineRunSummary, prepare_records_for_run
from lerobot_align.frames import make_frame_provider
from lerobot_align.modules import (
    GeneralVqaModule,
    InterjectionsAndSpeechModule,
    PlanSubtasksMemoryModule,
)
from lerobot_align.reader import EpisodeRecord
from lerobot_align.publication import copy_publication_files, publication_manifest, source_dataset_card
from lerobot_align.validator import StagingValidator
from lerobot_align.vlm_client import make_vlm_client
from lerobot_align.writer import LanguageColumnsWriter, dataset_run_lock

if TYPE_CHECKING or _datasets_available:
    from lerobot.datasets.dataset_metadata import CODEBASE_VERSION
    from lerobot.datasets.io_utils import load_info

logger = logging.getLogger(__name__)
_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


@dataclass(frozen=True)
class _HubPushPreflight:
    """Read-only Hub state captured before local annotation work starts."""

    api: HfApi
    repo_id: str
    version_tag: str
    repo_exists: bool
    parent_commit: str | None
    expected_tag_target: str | None


def _resolve_root(cfg: AnnotationPipelineConfig) -> Path:
    if cfg.root is not None:
        return Path(cfg.root)
    if cfg.repo_id is not None:
        # Hub snapshots are a shared, content-addressed cache. Writers must not
        # modify them in place, so copy the file contents (not cache symlinks or
        # read-only blob modes) into an isolated writable worktree.
        snapshot_root = Path(snapshot_download(repo_id=cfg.repo_id, repo_type="dataset"))
        work_root = Path(tempfile.mkdtemp(prefix="lerobot-align-"))
        try:
            shutil.copytree(
                snapshot_root,
                work_root,
                dirs_exist_ok=True,
                copy_function=shutil.copyfile,
            )
        except (OSError, shutil.Error):
            shutil.rmtree(work_root, ignore_errors=True)
            raise
        logger.info("Copied Hub snapshot %s to writable worktree %s", snapshot_root, work_root)
        return work_root
    raise ValueError("Either --root or --repo_id must be provided.")


def _log_retained_worktree(root: Path, cfg: AnnotationPipelineConfig) -> None:
    """Explain the lifecycle of the temporary copy made for a Hub source."""
    if cfg.root is not None or cfg.repo_id is None:
        return
    logger.warning(
        "Writable dataset worktree retained at %s. Remove it after checking the output. "
        "For a durable local destination, download or copy the dataset yourself and pass --root.",
        root,
    )


def _validate_push_target(cfg: AnnotationPipelineConfig) -> None:
    """Reject an upload request with nowhere to send it before annotation starts."""
    if cfg.push_to_hub and cfg.repo_id is None and cfg.new_repo_id is None:
        raise ValueError(
            "--push_to_hub requires --repo_id or --new_repo_id (the dataset repo to push to)."
        )


def _run_prevalidated_local(
    cfg: AnnotationPipelineConfig,
    root: Path,
    writer: LanguageColumnsWriter,
    records: Sequence[EpisodeRecord],
) -> PipelineRunSummary:
    """Construct runtime services and execute while the caller holds the dataset lock."""
    frame_provider = make_frame_provider(
        root, camera_key=cfg.vlm.camera_key, video_backend=cfg.video_backend
    )
    needs_visual_plan = cfg.plan.enabled and cfg.plan.subtask_import == "off"
    if needs_visual_plan:
        for record in records:
            if cfg.plan.subtasks_path is not None and cfg.plan.subtask_align_camera_keys:
                cameras = [c for c in cfg.plan.subtask_align_camera_keys if c in frame_provider.camera_keys]
                frames = next((images for camera in cameras if (images := frame_provider.frames_at(
                    record, [record.frame_timestamps[0]], camera_key=camera
                ))), [])
            else:
                frames = frame_provider.frames_at(record, [record.frame_timestamps[0]], fail_on_error=True)
            if not frames:
                raise RuntimeError(f"episode {record.episode_index}: required visual input could not be loaded")
    vlm = make_vlm_client(cfg.vlm)
    # Surface the resolved cameras up front so a silent vqa-module no-op
    # is obvious in job output rather than discovered post-hoc by counting
    # parquet rows.
    cam_keys = list(getattr(frame_provider, "camera_keys", []) or [])
    logger.info(
        "annotate: frame_provider default camera=%r, all cameras=%s",
        getattr(frame_provider, "camera_key", None),
        cam_keys,
    )
    if cfg.vqa.enabled and not cam_keys:
        logger.warning(
            "annotate: the vqa module is enabled but no cameras were "
            "resolved — it will produce zero VQA rows. Check "
            "meta/info.json for observation.images.* features, or pass "
            "--vlm.camera_key=<key> to seed the cameras list."
        )
    # ``root`` lets the plan module read SARM ``meta/episodes`` and
    # ``meta/lerobot_annotations.json``; other sources use the episode record.
    plan = PlanSubtasksMemoryModule(
        vlm=vlm, config=cfg.plan, frame_provider=frame_provider, root=root
    )
    interjections = InterjectionsAndSpeechModule(
        vlm=vlm, config=cfg.interjections, seed=cfg.seed, frame_provider=frame_provider
    )
    vqa = GeneralVqaModule(vlm=vlm, config=cfg.vqa, seed=cfg.seed, frame_provider=frame_provider)
    validator = StagingValidator(
        dataset_camera_keys=tuple(getattr(frame_provider, "camera_keys", []) or []) or None,
    )

    executor = Executor(
        config=cfg,
        plan=plan,
        interjections=interjections,
        vqa=vqa,
        writer=writer,
        validator=validator,
    )
    return executor.run(
        root,
        prevalidated_records=records,
        dataset_lock_held=True,
    )


@parser.wrap()
def annotate(cfg: AnnotationPipelineConfig) -> None:
    """Run the steerable annotation pipeline against a dataset."""
    _validate_push_target(cfg)
    if cfg.job.is_remote:
        # Imported lazily so local-only runs do not initialize the Jobs client.
        from lerobot_align.jobs import submit_align_to_hf

        return submit_align_to_hf(cfg)

    root = _resolve_root(cfg)
    logger.info("annotate: root=%s", root)

    try:
        # Hold the lease from preflight through metadata mutation. This prevents
        # another run from changing the dataset after validation or mixing rows in
        # the shared episode staging tree.
        with dataset_run_lock(root):
            writer = LanguageColumnsWriter()
            records = prepare_records_for_run(root, cfg, writer)
            hub_preflight = _prepare_hub_push(root, cfg) if cfg.push_to_hub else None
            summary = _run_prevalidated_local(cfg, root, writer, records)
            if cfg.push_to_hub:
                _push_to_hub(root, cfg, preflight=hub_preflight)
        logger.info("annotate: wrote %d shard(s)", len(summary.written_paths))
        for phase in summary.phases:
            logger.info(
                "annotate: phase=%s processed=%d skipped=%d",
                phase.name,
                phase.episodes_processed,
                phase.episodes_skipped,
            )
        if summary.validation_report.warnings:
            for w in summary.validation_report.warnings:
                logger.warning(w)
    finally:
        _log_retained_worktree(root, cfg)


def _prepare_hub_push(root: Path, cfg: AnnotationPipelineConfig) -> _HubPushPreflight:
    """Capture target state before constructing a VLM or mutating the dataset."""
    require_package("datasets", "dataset")
    repo_id = cfg.new_repo_id or cfg.repo_id
    if repo_id is None:
        raise ValueError("A Hub repository ID is required before pushing the dataset.")
    dataset_info = load_info(root)
    version_tag = (
        dataset_info.codebase_version
        if dataset_info.codebase_version.startswith("v")
        else CODEBASE_VERSION
    )
    api = HfApi()
    repo_exists, parent_commit, expected_tag_target = _preflight_hub_push(
        api,
        repo_id=repo_id,
        version_tag=version_tag,
        allow_version_tag_move=cfg.allow_version_tag_move,
    )
    publication_manifest(root, cfg.resolved_staging_dir(root))
    source_dataset_card(root, cfg)
    return _HubPushPreflight(
        api=api,
        repo_id=repo_id,
        version_tag=version_tag,
        repo_exists=repo_exists,
        parent_commit=parent_commit,
        expected_tag_target=expected_tag_target,
    )


def _push_to_hub(
    root: Path,
    cfg: AnnotationPipelineConfig,
    *,
    preflight: _HubPushPreflight | None = None,
) -> None:
    """Upload the annotated dataset directory to the Hub.

    Pushes to ``cfg.new_repo_id`` when set, otherwise back to ``cfg.repo_id``.
    """
    preflight = preflight or _prepare_hub_push(root, cfg)
    repo_id = preflight.repo_id
    commit_message = cfg.push_commit_message or "Add steerable annotations (lerobot-align)"
    dataset_info = load_info(root)
    # Read the version straight from the dataset's own ``meta/info.json`` so
    # the tag cannot drift from what the writer actually wrote.
    version_tag = (
        dataset_info.codebase_version
        if dataset_info.codebase_version.startswith("v")
        else CODEBASE_VERSION
    )
    if version_tag != preflight.version_tag:
        raise RuntimeError(
            f"Dataset version changed after Hub preflight from {preflight.version_tag!r} "
            f"to {version_tag!r}; no Hub changes were made."
        )

    manifest = publication_manifest(root, cfg.resolved_staging_dir(root))
    card = source_dataset_card(root, cfg)

    api = preflight.api
    parent_commit = preflight.parent_commit

    logger.info(f"[lerobot-align] creating/locating dataset repo {repo_id}...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=cfg.push_private,
        exist_ok=True,
    )
    if not preflight.repo_exists:
        # Hub repositories can be truly empty immediately after creation. A
        # concurrent creator/auto-initializer may have established ``main`` in
        # the meantime; guard against that exact head when present.
        parent_commit = _repo_head_or_none(api, repo_id=repo_id)

    _revalidate_tag_before_upload(
        api,
        repo_id=repo_id,
        version_tag=version_tag,
        expected_target=preflight.expected_tag_target,
    )

    logger.info(f"[lerobot-align] uploading {root} -> {repo_id}...")
    with tempfile.TemporaryDirectory(prefix="lerobot-align-publication-") as directory:
        snapshot = Path(directory)
        copy_publication_files(root, snapshot, manifest)
        commit_info = api.upload_folder(
            folder_path=str(snapshot),
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            parent_commit=parent_commit,
        )
    upload_revision = _commit_oid(commit_info, operation="dataset upload")
    logger.info("[lerobot-align] dataset files uploaded; publishing the dataset card...")

    card_commit_info = card.push_to_hub(
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card (lerobot-align)",
        parent_commit=upload_revision,
    )
    final_revision = _commit_oid(card_commit_info, operation="dataset-card upload")
    logger.info("[lerobot-align] dataset card uploaded; publishing the required version tag...")

    # Tag the upload with the codebase version. ``LeRobotDatasetMetadata``
    # resolves the dataset revision via ``get_safe_version`` which scans
    # for tags like ``v3.0``; without a tag it raises
    # ``RevisionNotFoundError``. Point it at the card commit, not the earlier
    # folder commit, so the tagged revision contains the generated README.
    _publish_required_version_tag(
        api,
        repo_id=repo_id,
        version_tag=version_tag,
        revision=final_revision,
        expected_target=preflight.expected_tag_target,
        allow_version_tag_move=cfg.allow_version_tag_move,
    )
    logger.info(
        "[lerobot-align] published https://huggingface.co/datasets/%s at %s "
        "with verified tag %s",
        repo_id,
        final_revision,
        version_tag,
    )


def _find_tag_target(api: HfApi, *, repo_id: str, tag: str) -> tuple[bool, str | None]:
    """Return whether ``tag`` exists and the commit it currently targets."""
    try:
        refs = api.list_repo_refs(repo_id=repo_id, repo_type="dataset")
    except RevisionNotFoundError:
        # A newly created, truly empty Hub repository has no refs yet.
        return False, None
    for ref in refs.tags:
        if ref.name == tag:
            return True, ref.target_commit
    return False, None


def _repo_head_or_none(api: HfApi, *, repo_id: str) -> str | None:
    """Resolve ``main`` for optimistic locking, permitting a truly empty repo."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision="main")
    except RevisionNotFoundError:
        return None
    head = getattr(info, "sha", None)
    if head is None:
        return None
    if not isinstance(head, str) or not head:
        raise RuntimeError(
            f"Could not resolve the current main commit for dataset repository {repo_id!r}."
        )
    return head


def _commit_oid(commit_info: object, *, operation: str) -> str:
    """Extract the exact commit OID returned by a guarded Hub write."""
    revision = getattr(commit_info, "oid", None)
    if not isinstance(revision, str) or not revision:
        raise RuntimeError(
            f"Hub {operation} did not return a commit OID; cannot safely publish a version tag."
        )
    return revision


def _preflight_hub_push(
    api: HfApi,
    *,
    repo_id: str,
    version_tag: str,
    allow_version_tag_move: bool,
) -> tuple[bool, str | None, str | None]:
    """Inspect the target before any mutation and reject unapproved tag moves."""
    try:
        parent_commit = _repo_head_or_none(api, repo_id=repo_id)
    except RepositoryNotFoundError:
        return False, None, None
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not inspect dataset repository {repo_id!r} before upload; no Hub changes were made."
        ) from exc

    try:
        exists, target = _find_tag_target(api, repo_id=repo_id, tag=version_tag)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not inspect version tag {version_tag!r} on {repo_id!r} before upload; "
            "no Hub changes were made."
        ) from exc

    if exists:
        if not isinstance(target, str) or not target:
            raise RuntimeError(
                f"Version tag {version_tag!r} on {repo_id!r} has no resolvable commit target; "
                "no Hub changes were made."
            )
        if not allow_version_tag_move:
            raise RuntimeError(
                f"Version tag {version_tag!r} already exists on {repo_id!r} at {target!r}. "
                "The upload was not started because moving an existing Hub tag is destructive. "
                "Review that release and explicitly pass --allow_version_tag_move=true to use "
                "the verified backup-and-rollback transaction."
            )
        return True, parent_commit, target
    return True, parent_commit, None


def _revalidate_tag_before_upload(
    api: HfApi,
    *,
    repo_id: str,
    version_tag: str,
    expected_target: str | None,
) -> None:
    """Abort before file upload if the version ref changed during local work."""
    try:
        exists, target = _find_tag_target(api, repo_id=repo_id, tag=version_tag)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not revalidate version tag {version_tag!r} on {repo_id!r}; "
            "no dataset files were uploaded."
        ) from exc

    unchanged = (expected_target is None and not exists) or (
        expected_target is not None and exists and target == expected_target
    )
    if not unchanged:
        raise RuntimeError(
            f"Version tag {version_tag!r} changed after preflight on {repo_id!r}: "
            f"expected {expected_target!r}, found {target!r}; no dataset files were uploaded."
        )


def _publish_required_version_tag(
    api: HfApi,
    *,
    repo_id: str,
    version_tag: str,
    revision: str,
    expected_target: str | None,
    allow_version_tag_move: bool,
) -> None:
    """Create or explicitly move the required version tag and verify it."""
    try:
        exists, target = _find_tag_target(api, repo_id=repo_id, tag=version_tag)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not inspect required tag {version_tag!r} on {repo_id!r}; "
            "the dataset upload is incomplete."
        ) from exc

    if exists and target == revision:
        logger.info(
            "[lerobot-align] required tag %s already targets uploaded commit %s",
            version_tag,
            revision,
        )
        return

    if expected_target is None:
        if exists:
            raise RuntimeError(
                f"Required tag {version_tag!r} appeared on {repo_id!r} at {target!r} after "
                "preflight; refusing to overwrite a concurrent publisher."
            )
        _create_and_verify_tag(
            api,
            repo_id=repo_id,
            tag=version_tag,
            revision=revision,
            context="required version tag",
        )
        return

    if not exists or target != expected_target:
        raise RuntimeError(
            f"Required tag {version_tag!r} changed after preflight on {repo_id!r}: "
            f"expected {expected_target!r}, found {target!r}; refusing to overwrite a "
            "concurrent publisher."
        )
    if not allow_version_tag_move:
        raise RuntimeError(
            f"Moving existing required tag {version_tag!r} on {repo_id!r} requires explicit "
            "allow_version_tag_move authorization."
        )

    _move_version_tag_with_backup(
        api,
        repo_id=repo_id,
        version_tag=version_tag,
        old_revision=expected_target,
        new_revision=revision,
    )


def _create_and_verify_tag(
    api: HfApi,
    *,
    repo_id: str,
    tag: str,
    revision: str,
    context: str,
) -> None:
    """Create one tag and require the Hub to report its exact target."""
    try:
        api.create_tag(
            repo_id=repo_id,
            tag=tag,
            revision=revision,
            repo_type="dataset",
        )
        verified, target = _find_tag_target(api, repo_id=repo_id, tag=tag)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to create and verify {context} {tag!r} at {revision!r} on {repo_id!r}."
        ) from exc
    if not verified or target != revision:
        raise RuntimeError(
            f"Failed to verify {context} {tag!r} on {repo_id!r}: "
            f"expected {revision!r}, found {target!r}."
        )


def _unique_backup_tag(api: HfApi, *, repo_id: str, version_tag: str) -> str:
    """Choose an unused, unguessable recovery ref without mutating the Hub."""
    for _ in range(8):
        candidate = f"{version_tag}-lerobot-align-backup-{uuid4().hex}"
        exists, _target = _find_tag_target(api, repo_id=repo_id, tag=candidate)
        if not exists:
            return candidate
    raise RuntimeError("Could not allocate a unique recovery tag after 8 attempts.")


def _move_version_tag_with_backup(
    api: HfApi,
    *,
    repo_id: str,
    version_tag: str,
    old_revision: str,
    new_revision: str,
) -> None:
    """Move a tag under an explicit opt-in, preserving a verified recovery ref."""
    backup_tag = _unique_backup_tag(api, repo_id=repo_id, version_tag=version_tag)
    try:
        _create_and_verify_tag(
            api,
            repo_id=repo_id,
            tag=backup_tag,
            revision=old_revision,
            context="recovery tag",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not secure recovery tag {backup_tag!r}; required tag {version_tag!r} "
            f"on {repo_id!r} was not modified."
        ) from exc

    # This read intentionally sits immediately before deletion. A tag update
    # is independent of the branch parent-commit guard used by the uploads.
    try:
        current, current_target = _find_tag_target(api, repo_id=repo_id, tag=version_tag)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not revalidate required tag {version_tag!r} immediately before its move; "
            f"no deletion was attempted. Verified recovery tag {backup_tag!r} remains at "
            f"{old_revision!r}."
        ) from exc
    if not current or current_target != old_revision:
        raise RuntimeError(
            f"Required tag {version_tag!r} changed immediately before its move: expected "
            f"{old_revision!r}, found {current_target!r}. No deletion was attempted. Verified "
            f"recovery tag {backup_tag!r} remains at {old_revision!r}."
        )

    try:
        api.delete_tag(repo_id=repo_id, tag=version_tag, repo_type="dataset")
        _create_and_verify_tag(
            api,
            repo_id=repo_id,
            tag=version_tag,
            revision=new_revision,
            context="required version tag",
        )
    except Exception as move_exc:  # noqa: BLE001
        try:
            _restore_version_tag_if_absent(
                api,
                repo_id=repo_id,
                version_tag=version_tag,
                old_revision=old_revision,
            )
        except Exception as rollback_exc:  # noqa: BLE001
            raise RuntimeError(
                f"Failed to move required tag {version_tag!r} on {repo_id!r} from "
                f"{old_revision!r} to {new_revision!r} ({move_exc}); safe rollback also failed "
                f"({rollback_exc}). Verified recovery tag {backup_tag!r} remains at "
                f"{old_revision!r}; restore from it manually after reviewing the current tag."
            ) from rollback_exc
        raise RuntimeError(
            f"Failed to move required tag {version_tag!r} on {repo_id!r} to "
            f"{new_revision!r}; its previous target {old_revision!r} was restored and verified. "
            f"Recovery tag {backup_tag!r} was retained for audit/recovery."
        ) from move_exc

    _cleanup_backup_tag(api, repo_id=repo_id, backup_tag=backup_tag)


def _restore_version_tag_if_absent(
    api: HfApi,
    *,
    repo_id: str,
    version_tag: str,
    old_revision: str,
) -> None:
    """Restore the old target only when no concurrent actor owns the tag."""
    exists, target = _find_tag_target(api, repo_id=repo_id, tag=version_tag)
    if exists:
        if target == old_revision:
            return
        raise RuntimeError(
            f"required tag now exists at {target!r}; refusing to overwrite a concurrent publisher"
        )

    _create_and_verify_tag(
        api,
        repo_id=repo_id,
        tag=version_tag,
        revision=old_revision,
        context="rollback tag",
    )


def _cleanup_backup_tag(api: HfApi, *, repo_id: str, backup_tag: str) -> None:
    """Best-effort cleanup after the required tag was successfully verified."""
    try:
        api.delete_tag(repo_id=repo_id, tag=backup_tag, repo_type="dataset")
        exists, target = _find_tag_target(api, repo_id=repo_id, tag=backup_tag)
        if exists:
            raise RuntimeError(f"tag still targets {target!r} after deletion")
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[lerobot-align] required version tag is valid, but recovery tag %s on %s "
            "could not be removed or its removal could not be verified (%s); it is safe "
            "to remove this backup manually",
            backup_tag,
            repo_id,
            exc,
        )


def main() -> None:
    # Configure logging before draccus parses arguments or remote dispatch
    # starts. ``force`` is appropriate at this process-owned entry point and
    # prevents an imported library's placeholder handler from making this a
    # silent no-op.
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, force=True)
    annotate()


def motion_viz_main() -> None:
    """Load the optional plotting CLI with an actionable dependency error."""
    try:
        from lerobot_align.diagnostics.plot_motion_sampling import main as plot_main
    except ModuleNotFoundError as exc:
        if exc.name not in {"matplotlib", "contourpy"}:
            raise
        raise SystemExit(
            "lerobot-motion-viz requires plotting dependencies; "
            "install them with `pip install 'lerobot-align[viz]'`."
        ) from exc
    plot_main()


if __name__ == "__main__":
    main()
