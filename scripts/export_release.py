#!/usr/bin/env python3
"""Export only the exact reviewed files in release/source-files.txt from a commit.

Working-tree changes, untracked files and Git history are never copied. The
destination must not exist; a failed export leaves no partial destination.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import tempfile

MANIFEST = "release/source-files.txt"
RECEIPT = "release-export.json"


def git(repository: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(repository), *args], check=True,
                          capture_output=True).stdout


def export_source(repository: Path, destination: Path, revision: str = "HEAD") -> dict:
    commit = git(repository, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    entries = {}
    for row in git(repository, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if row:
            metadata, name = row.split(b"\t", 1)
            mode, kind, oid = metadata.decode().split()
            entries[name.decode()] = (mode, kind, oid)
    if entries.get(MANIFEST, (None,))[0] != "100644":
        raise ValueError("The selected commit must contain a regular release/source-files.txt")
    names = git(repository, "cat-file", "blob", entries[MANIFEST][2]).decode().splitlines()
    if not names or len(names) != len(set(names)) or MANIFEST not in names:
        raise ValueError("Release manifest must be non-empty, unique, and include itself")
    for name in names:
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or str(path) != name or ".." in path.parts
                or any(part in {".git", ".env", "__pycache__"} for part in path.parts)
                or name.startswith("evaluation/results/") or name == RECEIPT):
            raise ValueError(f"Unsafe release manifest path: {name!r}")
        if entries.get(name, (None,))[0] not in {"100644", "100755"}:
            raise ValueError(f"Manifest entry must be a committed regular file: {name}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Release destination must not exist: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "source_commit": commit, "inherited_git_history": False, "files": []}
    with tempfile.TemporaryDirectory(prefix=".align-export-", dir=destination.parent) as staging:
        root = Path(staging) / "source"
        root.mkdir()
        for name in sorted(names):
            mode, _, oid = entries[name]
            data = git(repository, "cat-file", "blob", oid)
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(int(mode[-3:], 8))
            receipt["files"].append({"path": name, "size": len(data),
                                     "sha256": hashlib.sha256(data).hexdigest()})
        (root / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
        # The only operation that makes the completed source tree visible.
        root.rename(destination)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    receipt = export_source(args.repository, args.destination, args.revision)
    print(f"Exported {len(receipt['files'])} committed files to {args.destination}; no Git history copied")


if __name__ == "__main__":
    main()
