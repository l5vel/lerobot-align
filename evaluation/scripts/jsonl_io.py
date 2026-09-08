"""Transparent reading of score files that may be stored gzipped.

The study writes per-episode score rows as plain ``.jsonl`` during a run, but
the committed evidence is stored ``.jsonl.gz``: the uncompressed Study 1 dump is
146 MB, which is both unpleasant to clone and over GitHub's hard 100 MiB
per-file limit. The newer artifacts under ``results/unified_protocol/`` and
``results/corpus_b_paired_ablations/`` already used ``.gz``; this module makes
the older readers agree, so a caller can name either form and get the rows.

Two properties matter and are the reason this is a module rather than an inline
``gzip.open``:

**A path resolves across the two forms.** ``resolve()`` accepts
``results/scores.jsonl`` and returns ``results/scores.jsonl.gz`` when only the
compressed file is present, and vice versa. Existing commands, docstrings and
``run_all.sh`` stages therefore keep working unchanged whichever form is on
disk, and a fresh run that writes plain ``.jsonl`` is still readable.

**Hashes are taken over the DECOMPRESSED bytes.** ``content_bytes()`` returns
what the file logically contains, not how it is stored. Provenance records such
as ``results/corpus_calibration_diagnosis/provenance.json`` pin the sha256 of
``alignment_scores.jsonl``; hashing the container instead would invalidate a
frozen record the moment the file was compressed, and the record is evidence.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path


def resolve(path: Path) -> Path:
    """The existing file for ``path``, accepting either the plain or .gz form.

    Prefers an exact match, so a caller that deliberately names ``x.jsonl.gz``
    never silently reads a stale plain ``x.jsonl`` sitting beside it.
    """
    path = Path(path)
    if path.exists():
        return path
    if path.suffix == ".gz":
        plain = path.with_suffix("")
        if plain.exists():
            return plain
    else:
        packed = path.with_suffix(path.suffix + ".gz")
        if packed.exists():
            return packed
    # Nothing on disk: return the requested path so the caller raises the
    # ordinary FileNotFoundError naming what it actually asked for.
    return path


def open_text(path: Path):
    """A text-mode handle for ``path``, decompressing when it is gzipped."""
    resolved = resolve(path)
    if resolved.suffix == ".gz":
        return gzip.open(resolved, "rt", encoding="utf-8")
    return resolved.open(encoding="utf-8")


def read_text(path: Path) -> str:
    with open_text(path) as handle:
        return handle.read()


def content_bytes(path: Path) -> bytes:
    """The decompressed bytes of ``path``, for stable content hashing."""
    resolved = resolve(path)
    if resolved.suffix == ".gz":
        with gzip.open(resolved, "rb") as handle:
            return handle.read()
    return resolved.read_bytes()


def iter_lines(path: Path) -> Iterator[bytes]:
    """Decompressed lines as bytes, for streaming hash-and-parse readers."""
    resolved = resolve(path)
    opener = gzip.open if resolved.suffix == ".gz" else open
    with opener(resolved, "rb") as handle:
        yield from handle


def load_jsonl(path: Path) -> list[dict]:
    """Every non-blank row of ``path`` as a dict."""
    with open_text(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]
