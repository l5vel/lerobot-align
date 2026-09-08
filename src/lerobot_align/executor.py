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
"""In-process executor that runs the annotation phases.

The executor runs **six phases** in dependency order:

    phase 1: ``plan`` module (plan + subtasks + memory)
    phase 2: ``interjections`` module (interjections + speech)
    phase 3: ``plan`` state pass — ensures a co-timestamped deterministic
             plan at every interjection produced by phase 2
    phase 4: ``vqa`` module (VQA)
    phase 5: validator
    phase 6: writer

Generated interjections land at subtask boundaries, where phase 1 already
emits the correct remaining-subtasks plan. Phase 3 preserves a deterministic
fallback for externally staged/non-boundary interjections.

Distributed execution is provided by Hugging Face Jobs (see
``lerobot_align.jobs``, reached via ``--job.target=<flavor>``); the pod inside
the job invokes ``lerobot-align`` which uses this in-process executor.
Episode-level concurrency is controlled by
``ExecutorConfig.episode_parallelism``.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import AnnotationPipelineConfig
from .reader import EpisodeRecord, iter_episodes
from .staging import EpisodeStaging
from .validator import StagingValidator
from .writer import LEGACY_ANNOTATION_COLUMNS, LanguageColumnsWriter, dataset_run_lock

logger = logging.getLogger(__name__)


@dataclass
class PhaseResult:
    """Summary of one pipeline phase across all episodes."""

    name: str
    episodes_processed: int
    episodes_skipped: int


@dataclass
class PipelineRunSummary:
    """Aggregated result returned by :meth:`Executor.run`."""

    phases: list[PhaseResult]
    written_paths: list[Path]
    validation_report: Any  # ValidationReport, kept Any to avoid import cycle
    skipped_episodes: dict[int, list[str]] = field(default_factory=dict)


def prepare_records_for_run(
    root: Path,
    config: AnnotationPipelineConfig,
    writer: LanguageColumnsWriter,
) -> list[EpisodeRecord]:
    """Load and preflight the records before constructing expensive runtime services."""
    # ``iter_episodes`` checks the storage layout/version; ``load_info`` also
    # validates that the typed metadata required after the shard rewrite is
    # readable now, before a VLM server starts or any data is modified.
    from lerobot.datasets.io_utils import load_info  # noqa: PLC0415

    records = list(iter_episodes(root, only_episodes=config.only_episodes))
    load_info(root)
    if not records:
        raise ValueError(f"No episodes found under {root}/data/")
    writer.validate_write_scope(records, root)
    if config.plan.enabled:
        from .modules.plan_subtasks_memory import _load_align_calibration, _load_subtasks_file
        if config.plan.subtasks_path is not None:
            given = _load_subtasks_file(Path(config.plan.subtasks_path))
            missing = [r.episode_index for r in records if not given.for_episode(r.episode_index)]
            if missing:
                raise ValueError(f'No supplied subtasks for episodes {missing}; add labels or a default list')
        if config.plan.subtask_align_calibration_path is not None:
            _load_align_calibration(Path(config.plan.subtask_align_calibration_path))
    return records


@dataclass
class Executor:
    """Run all six phases over a dataset root in-process.

    Episode-level concurrency comes from ``ExecutorConfig.episode_parallelism``
    (a thread pool); cluster-level concurrency comes from running this
    executor inside a Hugging Face Job. Tests construct the executor
    directly with stub modules.
    """

    config: AnnotationPipelineConfig
    plan: Any  # PlanSubtasksMemoryModule
    interjections: Any  # InterjectionsAndSpeechModule
    vqa: Any  # GeneralVqaModule
    writer: LanguageColumnsWriter
    validator: StagingValidator

    def run(
        self,
        root: Path,
        *,
        prevalidated_records: Sequence[EpisodeRecord] | None = None,
        dataset_lock_held: bool = False,
    ) -> PipelineRunSummary:
        """Execute the pipeline under the dataset-wide run lock.

        The CLI acquires the lock before preflight and VLM construction, then
        passes ``dataset_lock_held=True``. Direct callers get the same safety
        automatically here.
        """
        if dataset_lock_held:
            return self._run_locked(root, prevalidated_records=prevalidated_records)
        with dataset_run_lock(root):
            return self._run_locked(root, prevalidated_records=prevalidated_records)

    def _run_locked(
        self,
        root: Path,
        *,
        prevalidated_records: Sequence[EpisodeRecord] | None,
    ) -> PipelineRunSummary:
        records = (
            prepare_records_for_run(root, self.config, self.writer)
            if prevalidated_records is None
            else list(prevalidated_records)
        )
        n = len(records)
        if n == 0:
            raise ValueError(f"No episodes found under {root}/data/")

        logger.info("[annotate] %d episodes total", n)

        staging_dir = self.config.resolved_staging_dir(root)
        staging_dir.mkdir(parents=True, exist_ok=True)

        phases: list[PhaseResult] = []

        # Phase 1: ``plan`` module (plan + subtasks + memory)
        phases.append(self._run_module_phase("plan", records, staging_dir, self.plan))
        # Phase 2: ``interjections`` module (interjections + speech). It
        # reads the ``plan`` module's subtask rows from the same staging
        # tree to ground the interjection prompt in the correct local subtask.
        phases.append(self._run_module_phase("interjections", records, staging_dir, self.interjections))
        # Phase 3: ensure co-timestamped deterministic plan state for interjections.
        phases.append(self._run_plan_update_phase(records, staging_dir))
        # Phase 4: ``vqa`` module (VQA)
        phases.append(self._run_module_phase("vqa", records, staging_dir, self.vqa))

        logger.info("[annotate] running validator...")
        report = self.validator.validate(records, staging_dir, config=self.config)
        logger.info("[annotate] validator: %s", report.summary())
        errors = getattr(report, "errors", [])
        episode_errors = getattr(report, "episode_errors", {})
        skipped = {
            ep: episode_errors[ep]
            for ep in getattr(report, "episode_completeness_errors", {})
        }
        report_path = self._write_validation_report(staging_dir, report, skipped, n)
        for error in errors[:20]:
            # Kept on stdout deliberately: this listing is asserted against
            # captured stdout by tests/test_staging_isolation.py.
            print(f"[annotate] validator error: {error}", flush=True)
        if len(errors) > 20:
            print(f"[annotate] {len(errors) - 20} more error(s); see {report_path}", flush=True)
        for warning in report.warnings[:20]:
            # Kept on stdout deliberately, as for the error listing above.
            print(f"[annotate] validator warning: {warning}", flush=True)
        if len(report.warnings) > 20:
            print(
                f"[annotate] {len(report.warnings) - 20} more warning(s); see {report_path}",
                flush=True,
            )

        # Errors on an excluded episode cannot reach the writer. Structural
        # errors elsewhere retain the existing skip_validation behavior, and
        # completeness errors without an episode identity still fail closed.
        excluded_errors = {error for reasons in skipped.values() for error in reasons}
        remaining_errors = set(errors) - excluded_errors
        unassigned_completeness = set(getattr(report, "completeness_errors", [])) - excluded_errors
        limit = self.config.executor.max_incomplete_episode_fraction
        too_many_incomplete = len(skipped) == n or len(skipped) / n > limit
        structural_failure = not report.ok and (bool(remaining_errors) or not errors)
        if unassigned_completeness or too_many_incomplete or (
            structural_failure and not self.config.skip_validation
        ):
            details = "\n".join(errors[:20])
            raise RuntimeError(
                f"Staging validation failed: {report.summary()}; incomplete={len(skipped)}/{n}, "
                f"max_incomplete_episode_fraction={limit:g}; report={report_path}\n{details}"
            )
        phases.append(PhaseResult("validation", n - len(skipped), len(skipped)))
        if skipped:
            logger.warning(
                "[annotate] skipping %d/%d incomplete episode(s); publishing %d; details: %s",
                len(skipped),
                n,
                n - len(skipped),
                report_path,
            )

        logger.info("[annotate] writing parquet shards into %s/data/...", root)
        from .transaction import rewrite_dataset
        written = rewrite_dataset(
            root, records, staging_dir, self.writer, self._ensure_annotation_metadata_in_info,
            skip_episode_indices=tuple(skipped),
        )
        logger.info("[annotate] wrote %d shard(s); pipeline complete", len(written))

        return PipelineRunSummary(
            phases=phases, written_paths=written, validation_report=report, skipped_episodes=skipped
        )

    @staticmethod
    def _write_validation_report(
        staging_dir: Path, report: Any, skipped: dict[int, list[str]], episodes_checked: int
    ) -> Path:
        """Retain all errors and rejected episode IDs, including on failed runs."""
        path = staging_dir / "validation_report.json"
        payload = {
            "episodes_checked": episodes_checked,
            "errors": getattr(report, "errors", []),
            "completeness_errors": getattr(report, "completeness_errors", []),
            "warnings": report.warnings,
            "episode_errors": getattr(report, "episode_errors", {}),
            "skipped_episodes": skipped,
        }
        fd, tmp_name = tempfile.mkstemp(dir=staging_dir, prefix=f".{path.name}.", suffix=".tmp")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
                file.write("\n")
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return path

    @staticmethod
    def _ensure_annotation_metadata_in_info(
        root: Path, dropped_columns: Sequence[str] = LEGACY_ANNOTATION_COLUMNS
    ) -> None:
        """Sync ``meta/info.json`` with the columns the writer just wrote.

        ``LanguageColumnsWriter`` adds ``language_persistent`` and
        ``language_events`` to parquet shards. The metadata must advertise
        those columns too, otherwise non-streaming ``LeRobotDataset`` loads
        cast against the old schema and fail on the extra parquet columns.

        The writer also *removes* legacy columns (``dropped_columns``), and
        those must come out of the metadata for the mirror-image reason: a
        feature declared in ``info.json`` with no parquet column behind it does
        not raise, it silently yields ``None`` on every frame.

        Preserves all other user metadata.
        """
        from lerobot.datasets.io_utils import load_info  # noqa: PLC0415
        from lerobot.datasets.language import SAY_TOOL_SCHEMA, language_feature_info  # noqa: PLC0415

        info_path = root / "meta" / "info.json"
        info = load_info(root)

        changed = False

        merged_features = {**info.features, **language_feature_info()}
        removed = [name for name in dropped_columns if name in merged_features]
        for name in removed:
            merged_features.pop(name)
        if merged_features != info.features:
            info.features = merged_features
            changed = True

        existing = info.tools or []
        names = {(t.get("function") or {}).get("name") for t in existing if isinstance(t, dict)}
        if SAY_TOOL_SCHEMA["function"]["name"] not in names:
            info.tools = [*existing, SAY_TOOL_SCHEMA]
            changed = True

        if changed:
            Executor._write_info_atomic(info, info_path)
            logger.info(
                "[annotate] meta/info.json: language_features=%s, dropped=%s, tools=%s",
                list(language_feature_info()),
                removed,
                [t["function"]["name"] for t in (info.tools or [])],
            )

    @staticmethod
    def _write_info_atomic(info: Any, info_path: Path) -> None:
        """Durably replace ``meta/info.json`` without exposing a partial file."""
        fd, tmp_name = tempfile.mkstemp(
            dir=info_path.parent,
            prefix=f".{info_path.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            os.chmod(tmp_path, stat.S_IMODE(info_path.stat().st_mode))
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(info.to_dict(), file, indent=4, ensure_ascii=False)
                file.flush()
                os.fsync(file.fileno())
            os.replace(tmp_path, info_path)

            # Persist the directory entry update as well as the file contents.
            dir_fd = os.open(info_path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except BaseException:
            # ``fdopen`` owns the descriptor after it succeeds. If an error is
            # raised before that point, close the raw descriptor here.
            with suppress(OSError):
                os.close(fd)
            raise
        finally:
            tmp_path.unlink(missing_ok=True)

    def _run_module_phase(
        self,
        name: str,
        records: list[EpisodeRecord],
        staging_dir: Path,
        module: Any,
    ) -> PhaseResult:
        if not module.enabled:
            # A staging directory is deliberately reusable across runs. Make
            # a disabled phase explicit for this run so its old JSONL cannot
            # be picked up by the writer and resurrect annotations the user
            # asked not to generate.
            for record in records:
                EpisodeStaging(staging_dir, record.episode_index).write(name, [])
            logger.info("[annotate] phase=%s skipped (module disabled)", name)
            return PhaseResult(name=name, episodes_processed=0, episodes_skipped=len(records))
        n = len(records)
        parallelism = max(1, min(self.config.executor.episode_parallelism, n))
        logger.info(
            "[annotate] phase=%s starting on %d episode(s) (parallelism=%d)", name, n, parallelism
        )
        t0 = time.time()

        def _do(idx_record: tuple[int, EpisodeRecord]) -> tuple[int, int, float]:
            i, record = idx_record
            ep_start = time.time()
            staging = EpisodeStaging(staging_dir, record.episode_index)
            module.run_episode(record, staging)
            return i, record.episode_index, time.time() - ep_start

        processed = 0
        if parallelism == 1:
            for i, record in enumerate(records, 1):
                _, ep_idx, elapsed = _do((i, record))
                processed += 1
                logger.info(
                    "[annotate]   %s episode %d/%d (idx=%d) done in %.1fs", name, i, n, ep_idx, elapsed
                )
        else:
            with ThreadPoolExecutor(max_workers=parallelism) as pool:
                futures = [pool.submit(_do, (i, r)) for i, r in enumerate(records, 1)]
                for fut in as_completed(futures):
                    i, ep_idx, elapsed = fut.result()
                    processed += 1
                    logger.info(
                        "[annotate]   %s episode %d/%d (idx=%d, submit_order=%d) done in %.1fs",
                        name,
                        processed,
                        n,
                        ep_idx,
                        i,
                        elapsed,
                    )
        total = time.time() - t0
        logger.info("[annotate] phase=%s complete: %d/%d in %.1fs", name, processed, n, total)
        return PhaseResult(name=name, episodes_processed=processed, episodes_skipped=0)

    def _run_plan_update_phase(  # noqa: PLR0915
        self, records: list[EpisodeRecord], staging_dir: Path
    ) -> PhaseResult:
        """Ensure ``plan`` state exists where the interjections module emitted.

        Generated interjections only cue the already-upcoming subtask and land
        on its boundary, where the deterministic remaining-subtasks plan is
        normally already present. Calling back into the plan module retains a
        deterministic fallback for externally staged/non-boundary events.
        """
        if not self.plan.enabled or not self.interjections.enabled:
            return PhaseResult(name="plan_update", episodes_processed=0, episodes_skipped=len(records))
        processed = 0
        for record in records:
            staging = EpisodeStaging(staging_dir, record.episode_index)
            interjection_rows = [
                row for row in staging.read("interjections") if row.get("style") == "interjection"
            ]
            interjection_times = [float(row["timestamp"]) for row in interjection_rows]
            if interjection_times:
                self.plan.run_plan_updates(record, staging, interjection_times)
                processed += 1
        # Episodes without any interjections are skipped (no plan state check
        # needed); count them so the summary's processed+skipped == total.
        return PhaseResult(
            name="plan_update",
            episodes_processed=processed,
            episodes_skipped=len(records) - processed,
        )
