"""Recoverable, dataset-wide publication of transformed shards and metadata.

Callers hold dataset_run_lock. External readers must remain stopped until the
transaction completes (or recovery finishes): POSIX cannot atomically rename
multiple files. Originals survive every replacement until a durable commit.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from dataclasses import replace

import pyarrow.parquet as pq

TRANSACTION_DIR = ".lerobot-align-transaction"


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _sync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _sync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _journal(txn: Path, payload: dict) -> None:
    path = txn / "journal.json"
    temp = txn / "journal.tmp"
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    _sync_dir(txn)


def _publish_file(source: Path, destination: Path) -> None:
    """One commit replacement; kept separate for failure-injection tests."""
    os.replace(source, destination)
    _sync_dir(destination.parent)


def _safe_relative(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Invalid transaction path: {value!r}")
    if str(path) != "meta/info.json" and not (
        path.parts[0] == "data" and path.suffix == ".parquet"
    ):
        raise ValueError(f"Unexpected transaction target: {value!r}")
    return path


def _assert_regular(path: Path, root: Path) -> None:
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError(f"Transaction requires regular files without symlinks: {path}")
    path.resolve().relative_to(root.resolve())


def recover_dataset(root: Path) -> bool:
    """Roll back interrupted commits. Caller must hold the exclusive run lock.

    Recovery is itself restartable: backup files are copied, never consumed.
    Invalid journals/backups stop the run and retain all recovery evidence.
    """
    txn = root / TRANSACTION_DIR
    if not txn.exists():
        return False
    if txn.is_symlink():
        raise RuntimeError(f"Refusing symlink transaction directory: {txn}")
    journal = txn / "journal.json"
    if not journal.exists():
        # The protocol never mutates a target before publishing its journal.
        shutil.rmtree(txn)
        _sync_dir(root)
        return True
    try:
        payload = json.loads(journal.read_text(encoding="utf-8"))
        if payload["version"] != 1 or payload["state"] not in {
            "prepared",
            "committing",
            "committed",
        }:
            raise ValueError("unsupported transaction journal")
        entries = payload["files"]
        if not entries or len({e["path"] for e in entries}) != len(entries):
            raise ValueError("invalid transaction file inventory")
        if payload["state"] == "committing":
            for entry in entries:
                rel = _safe_relative(entry["path"])
                backup = txn / "original" / rel
                _assert_regular(backup, txn)
                if _digest(backup) != entry["original_sha256"]:
                    raise ValueError(f"backup checksum mismatch: {rel}")
                target = root / rel
                _assert_regular(target, root)
                if _digest(target) not in {entry["original_sha256"], entry["new_sha256"]}:
                    raise ValueError(f"target changed outside the transaction: {rel}")
            for entry in entries:
                rel = _safe_relative(entry["path"])
                target = root / rel
                fd, name = tempfile.mkstemp(prefix=".align-rollback-", dir=target.parent)
                os.close(fd)
                temp = Path(name)
                try:
                    shutil.copy2(txn / "original" / rel, temp)
                    _sync_file(temp)
                    os.replace(temp, target)
                    _sync_dir(target.parent)
                finally:
                    temp.unlink(missing_ok=True)
            # Cleanup can also be interrupted. Persist that rollback finished
            # before deleting a single backup, so recovery can safely resume it.
            payload["state"] = "prepared"
            _journal(txn, payload)
    except Exception as exc:
        raise RuntimeError(
            f"Dataset transaction recovery failed at {txn}: {exc}. "
            "Keep this directory and stop dataset readers; restore from its verified originals "
            "or correct the filesystem error, then run lerobot-align-recover again."
        ) from exc
    shutil.rmtree(txn)
    _sync_dir(root)
    return True


def rewrite_dataset(
    root, records, staging_dir, writer, update_metadata, *, skip_episode_indices=()
):
    """Stage and validate the transformed data before replacing any source file."""
    root = root.resolve()
    txn = root / TRANSACTION_DIR
    if txn.exists():
        raise RuntimeError(f"Pending dataset transaction at {txn}; recover it before writing")
    txn.mkdir(mode=0o700)
    _sync_dir(root)
    try:
        new = txn / "new"
        originals = txn / "original"
        paths = sorted({record.data_path.resolve().relative_to(root) for record in records})
        paths.append(Path("meta/info.json"))  # metadata publication is always last
        # Stage the complete data schema, including canonical unselected shards.
        for source in sorted((root / "data").rglob("*.parquet")) + [root / "meta/info.json"]:
            _assert_regular(source, root)
            rel = source.relative_to(root)
            destination = new / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        staged_records = [
            replace(r, data_path=new / r.data_path.resolve().relative_to(root)) for r in records
        ]
        writer.write_all(
            staged_records,
            staging_dir,
            new,
            scope_validated=True,
            skip_episode_indices=skip_episode_indices,
        )
        update_metadata(new, dropped_columns=writer.dropped_columns)
        info = json.loads((new / "meta/info.json").read_text())
        required = {"language_persistent", "language_events"}
        if not required.issubset(info["features"]):
            raise ValueError("Staged metadata is missing language features")
        for source in sorted((root / "data").rglob("*.parquet")):
            transformed = new / source.relative_to(root)
            before, after = pq.read_table(source), pq.read_table(transformed)
            if before.num_rows != after.num_rows or not required.issubset(after.column_names):
                raise ValueError(f"Staged shard has wrong row count or schema: {source}")
            if set(writer.dropped_columns).intersection(after.column_names):
                raise ValueError(f"Staged shard retains obsolete columns: {source}")
            for identity in ("episode_index", "frame_index", "timestamp"):
                if identity in before.column_names and not before[identity].equals(after[identity]):
                    raise ValueError(f"Staged shard changed {identity}: {source}")
        entries = []
        for rel in paths:
            backup = originals / rel
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(root / rel, backup)
            _sync_file(backup)
            _sync_file(new / rel)
            entries.append(
                {
                    "path": str(rel),
                    "original_sha256": _digest(backup),
                    "new_sha256": _digest(new / rel),
                }
            )
        for directory in sorted(
            (p for p in txn.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True
        ):
            _sync_dir(directory)
        payload = {"version": 1, "state": "prepared", "files": entries}
        _journal(txn, payload)
        payload["state"] = "committing"
        _journal(txn, payload)
        for rel in paths:
            _publish_file(new / rel, root / rel)
        payload["state"] = "committed"
        _journal(txn, payload)
    except BaseException:
        recover_dataset(root)
        raise
    recover_dataset(root)  # committed: only clean up, never roll back
    return [root / rel for rel in paths if rel.parts[0] == "data"]


def main() -> int:
    import argparse
    from .writer import dataset_run_lock

    parser = argparse.ArgumentParser(
        description="Recover an interrupted LeRobot dataset rewrite without inference."
    )
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if not (args.root / "meta/info.json").is_file():
        parser.error("root must contain meta/info.json")
    # Lock acquisition performs recovery before any dataset read.
    with dataset_run_lock(args.root):
        print(f"Dataset recovery complete: {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
