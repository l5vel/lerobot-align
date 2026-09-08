"""Failures at every durable dataset publication boundary must be recoverable."""

import json
import os
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

from lerobot_align import transaction
from lerobot_align.executor import Executor
from lerobot_align.reader import iter_episodes
from lerobot_align.writer import LanguageColumnsWriter, dataset_run_lock
from tests.fixtures import build_annotation_dataset


@pytest.fixture
def dataset(tmp_path):
    return build_annotation_dataset(
        tmp_path / "dataset", episode_specs=[(0, 4, "move cup"), (1, 4, "place cup")]
    )


def snapshot(root):
    paths = sorted((root / "data").rglob("*.parquet")) + [root / "meta/info.json"]
    return {path.relative_to(root): path.read_bytes() for path in paths}


def rewrite(root):
    with dataset_run_lock(root):
        return transaction.rewrite_dataset(
            root,
            list(iter_episodes(root)),
            root / ".stage",
            LanguageColumnsWriter(),
            Executor._ensure_annotation_metadata_in_info,
        )


@pytest.mark.parametrize("failure_at", [0, 1, 2, 3])
def test_publication_exception_restores_all_originals(dataset, monkeypatch, failure_at):
    before = snapshot(dataset)
    original = transaction._publish_file
    count = 0

    def inject(source, destination):
        nonlocal count
        if count == failure_at:
            raise OSError("injected disk failure")
        original(source, destination)
        count += 1
        if failure_at == 3 and count == 3:
            raise KeyboardInterrupt("interrupted after metadata publication")

    monkeypatch.setattr(transaction, "_publish_file", inject)
    with pytest.raises((OSError, KeyboardInterrupt)):
        rewrite(dataset)
    assert snapshot(dataset) == before
    assert not (dataset / transaction.TRANSACTION_DIR).exists()


@pytest.mark.parametrize("failure_at", [1, 2, 3])
def test_process_death_recovers_before_next_read(dataset, failure_at):
    before = snapshot(dataset)
    script = """
import os, sys
from pathlib import Path
from lerobot_align import transaction
from lerobot_align.executor import Executor
from lerobot_align.reader import iter_episodes
from lerobot_align.writer import LanguageColumnsWriter, dataset_run_lock
root = Path(sys.argv[1])
original = transaction._publish_file
count = 0
def crash(source, destination):
    global count
    original(source, destination)
    count += 1
    if count == int(sys.argv[2]): os._exit(91)
transaction._publish_file = crash
with dataset_run_lock(root):
    transaction.rewrite_dataset(root, list(iter_episodes(root)), root / '.stage', LanguageColumnsWriter(),
                                Executor._ensure_annotation_metadata_in_info)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(dataset), str(failure_at)],
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    assert result.returncode == 91
    assert (dataset / transaction.TRANSACTION_DIR / "journal.json").exists()
    with dataset_run_lock(dataset):
        assert snapshot(dataset) == before
    assert not (dataset / transaction.TRANSACTION_DIR).exists()


def test_failure_during_staging_never_touches_originals(dataset, monkeypatch):
    before = snapshot(dataset)
    original = LanguageColumnsWriter._rewrite_one
    count = 0

    def inject(self, *args, **kwargs):
        nonlocal count
        original(self, *args, **kwargs)
        count += 1
        if count == 1:
            raise OSError("staging failed after first shard")

    monkeypatch.setattr(LanguageColumnsWriter, "_rewrite_one", inject)
    with pytest.raises(OSError, match="staging failed"):
        rewrite(dataset)
    assert snapshot(dataset) == before


def test_metadata_staging_failure_never_touches_originals(dataset):
    before = snapshot(dataset)

    def fail(*args, **kwargs):
        raise OSError("metadata failed")

    with dataset_run_lock(dataset), pytest.raises(OSError, match="metadata failed"):
        transaction.rewrite_dataset(
            dataset, list(iter_episodes(dataset)), dataset / ".stage", LanguageColumnsWriter(), fail
        )
    assert snapshot(dataset) == before


def test_success_publishes_consistent_shards_and_metadata(dataset):
    paths = rewrite(dataset)
    assert len(paths) == 2
    info = json.loads((dataset / "meta/info.json").read_text())
    for path in paths:
        columns = set(pq.read_schema(path).names)
        assert {"language_events", "language_persistent"}.issubset(columns)
        assert "subtask_index" not in columns
    assert {"language_events", "language_persistent"}.issubset(info["features"])
    assert not (dataset / transaction.TRANSACTION_DIR).exists()


def test_recovery_can_itself_be_interrupted(dataset, monkeypatch):
    before = snapshot(dataset)
    publish = transaction._publish_file
    replace_file = transaction.os.replace
    def fail_commit(source, destination):
        publish(source, destination)
        raise OSError('interrupt commit')
    def fail_rollback(source, destination):
        if str(source.name).startswith('.align-rollback-'):
            raise OSError('interrupt rollback')
        return replace_file(source, destination)
    with monkeypatch.context() as patch:
        patch.setattr(transaction, '_publish_file', fail_commit)
        patch.setattr(transaction.os, 'replace', fail_rollback)
        with pytest.raises(RuntimeError, match='recovery failed'):
            rewrite(dataset)
    assert (dataset / transaction.TRANSACTION_DIR / 'journal.json').exists()
    with dataset_run_lock(dataset):
        assert snapshot(dataset) == before


def test_validation_rejects_corrupted_staged_frame_identity(dataset, monkeypatch):
    before = snapshot(dataset)
    original = LanguageColumnsWriter.write_all
    def corrupt(self, *args, **kwargs):
        paths = original(self, *args, **kwargs)
        table = pq.read_table(paths[0])
        table = table.slice(1)
        pq.write_table(table, paths[0])
        return paths
    monkeypatch.setattr(LanguageColumnsWriter, 'write_all', corrupt)
    with pytest.raises(ValueError, match='wrong row count'):
        rewrite(dataset)
    assert snapshot(dataset) == before
