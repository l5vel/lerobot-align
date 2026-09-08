#!/usr/bin/env python
"""Build a pseudo-unannotated working copy of a dataset and extract its ground truth.

Both annotation tools mutate the dataset root in place, so every arm needs its
own root. Copying the videos would cost hundreds of gigabytes per arm, and the
tools only ever read them, so ``videos/`` is symlinked and only ``data/`` and
``meta/`` are materialised.

The three jobs this script does are deliberately coupled, because doing them
separately is how ground truth leaks:

0. **Verify provenance** -- refuse to treat machine output as ground truth.
1. **Extract** the human subtask spans to a file OUTSIDE the working root.
2. **Strip** every trace of them from the working root.

Provenance is not optional and is the reason this script exists in this shape.
The `language_persistent` column looks like ground truth in every dataset, but
in 14 of the 32 component datasets here it holds subtask rows that THIS TOOL
generated on an earlier run (their `.done` marker says `mode=generate`, not
`imported_episodes=N`). Scoring against those would be scoring the tool against
its own output. A further two datasets were imported from `maskjp/...` -- the
org the tool pushes its own results to -- and carry an empty
`lerobot_annotations.json`, so their labels are machine-derived too. And the
SARM importer, one of the supported sources, is itself a Qwen3-VL annotator.

We therefore take ground truth ONLY from `meta/lerobot_annotations.json`, the
external human artifact that neither tool ever writes (it is read-only in
`subtask_import.py` and absent from upstream entirely). That file is missing or
empty in exactly the machine-annotated datasets, so requiring it is
self-validating: a dataset that cannot supply it is refused rather than
silently scored.

Stripping is not optional and not partial. The source datasets carry the human
subtasks in ``language_persistent`` as ``style="subtask"`` rows, but the same
column also holds ``style="plan"`` and ``style="memory"`` rows written by an
earlier pipeline run, and those restate the remaining step list in plain text
at each boundary. A tool given those rows would be reading the answer. We
therefore clear the whole language payload rather than filtering by style, and
we delete ``meta/lerobot_annotations.json`` from the copy so that the new
tool's ``subtask_import=auto`` cannot rediscover the labels.

Run ``--verify`` (the default) to assert afterwards that no episode in the
working root retains any annotation state.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

SUBTASK_STYLE = "subtask"
ANNOTATION_COLUMNS = ("language_persistent", "language_events")
ANNOTATION_SIDECARS = ("lerobot_annotations.json", "subtasks.parquet")
# Hub orgs this pipeline pushes its own annotated datasets to. A dataset
# imported from one of these carries machine labels however human they look.
MACHINE_OUTPUT_ORGS = {"maskjp"}
# Staging directory written by a previous annotation run; it holds per-episode
# plan.jsonl files with the full label list and must not survive into a
# working root.
STAGING_DIR = ".annotate_staging"


def _episode_end(frame: pd.DataFrame) -> float:
    return float(frame["timestamp"].iloc[-1])


class ProvenanceError(RuntimeError):
    """Raised when a dataset cannot supply verified human ground truth."""


def verify_human_provenance(source: Path) -> dict[str, Any]:
    """Refuse any dataset whose subtask labels are not verifiably human.

    Three independent checks, all of which must pass:

    1. ``meta/lerobot_annotations.json`` exists and has non-empty subtasks.
       Neither tool writes this file, so its presence is evidence of an
       external annotation process.
    2. The sibling ``.done`` marker, when present, records
       ``imported_episodes=N`` rather than ``mode=generate``.
    3. The marker's source repo is not the org this tool pushes results to.

    Check 1 alone is nearly sufficient -- the machine-annotated datasets have
    no such file -- but checks 2 and 3 catch the one real case where a dataset
    was imported from a previously machine-annotated copy and kept an empty
    annotations file.
    """
    annotations = source / "meta" / ANNOTATION_SIDECARS[0]
    if not annotations.exists():
        raise ProvenanceError(
            f"{source.name}: no meta/lerobot_annotations.json. Its language_persistent "
            "column may hold machine-generated subtasks; refusing to use it as ground truth."
        )
    payload = json.loads(annotations.read_text(encoding="utf-8"))
    episodes = payload.get("episodes") or {}
    annotated = {k: v for k, v in episodes.items() if (v or {}).get("subtasks")}
    if not annotated:
        raise ProvenanceError(
            f"{source.name}: meta/lerobot_annotations.json has no non-empty subtasks. "
            "Its labels came from elsewhere and are not verified human annotations."
        )

    marker = source.parent / f"{source.name}.done"
    info: dict[str, Any] = {"n_annotated": len(annotated), "marker": None, "source_repo": None}
    if marker.exists():
        text = marker.read_text(encoding="utf-8")
        info["marker"] = text.strip().splitlines()[:3]
        if "mode=generate" in text:
            raise ProvenanceError(
                f"{source.name}: .done marker says mode=generate -- these subtasks were "
                "produced by this tool, not by a human."
            )
        for line in text.splitlines():
            if line.startswith("repo="):
                repo = line.split("=", 1)[1].strip()
                info["source_repo"] = repo
                if repo.split("/")[0] in MACHINE_OUTPUT_ORGS:
                    raise ProvenanceError(
                        f"{source.name}: imported from '{repo}', which is an org this tool "
                        "pushes its own output to. Labels are machine-derived."
                    )
    return info


def ground_truth_from_sidecar(source: Path) -> dict[int, list[dict[str, Any]]]:
    """Read human spans from ``meta/lerobot_annotations.json``.

    This, not the parquet column, is the authoritative source. Spans are
    normalised to ``{start, end, text}`` and validated: a dataset with
    overlapping or out-of-order human spans is reported rather than repaired,
    because a silent repair would change what the tools are being scored
    against.
    """
    payload = json.loads((source / "meta" / ANNOTATION_SIDECARS[0]).read_text(encoding="utf-8"))
    out: dict[int, list[dict[str, Any]]] = {}
    anomalies: list[str] = []
    for key, episode in (payload.get("episodes") or {}).items():
        raw = (episode or {}).get("subtasks") or []
        spans = [
            {"start": float(s["start"]), "end": float(s["end"]), "text": str(s["label"])}
            for s in raw
            if s.get("label") and s.get("start") is not None and s.get("end") is not None
        ]
        spans.sort(key=lambda s: s["start"])
        for index in range(1, len(spans)):
            if spans[index]["start"] < spans[index - 1]["end"] - 1e-6:
                anomalies.append(
                    f"episode {key}: span {index} starts at {spans[index]['start']:.3f} "
                    f"before previous end {spans[index - 1]['end']:.3f}"
                )
        if spans:
            out[int(key)] = spans
    if anomalies:
        print(f"[prepare] {len(anomalies)} human-annotation anomaly/ies (reported, not repaired):",
              file=sys.stderr)
        for line in anomalies[:5]:
            print(f"[prepare]   {line}", file=sys.stderr)
    return out


def extract_ground_truth(data_files: list[Path]) -> dict[int, list[dict[str, Any]]]:
    """Reconstruct subtask spans per episode from ``language_persistent``.

    Used to READ BACK what a tool wrote, and as a cross-check against the
    sidecar. It is deliberately NOT the ground-truth source: this column is
    exactly where machine output lives.

    A ``style="subtask"`` row marks the START of a span at its timestamp. The
    span runs until the next subtask row, and the final one runs to the last
    frame timestamp. Reimplemented here without importing either tool, so the
    evaluation's parsing cannot drift with the code under test.
    """
    out: dict[int, list[dict[str, Any]]] = {}
    for path in data_files:
        frame = pd.read_parquet(path)
        if "language_persistent" not in frame.columns:
            continue
        for episode, group in frame.groupby("episode_index"):
            raw = group["language_persistent"].iloc[0]
            if raw is None:
                continue
            rows = [dict(item) for item in raw if isinstance(item, dict)]
            marks = [
                (float(r["timestamp"]), str(r["content"]))
                for r in rows
                if r.get("style") == SUBTASK_STYLE and r.get("content")
            ]
            if not marks:
                continue
            marks.sort(key=lambda item: item[0])
            end_t = _episode_end(group)
            spans: list[dict[str, Any]] = []
            for index, (start, text) in enumerate(marks):
                end = marks[index + 1][0] if index + 1 < len(marks) else end_t
                if end <= start:
                    continue
                spans.append({"start": start, "end": end, "text": text})
            if spans:
                out[int(episode)] = spans
    return out


def strip_annotations(frame: pd.DataFrame) -> pd.DataFrame:
    """Blank every annotation-bearing column, preserving dtype and length."""
    frame = frame.copy()
    for column in ANNOTATION_COLUMNS:
        if column in frame.columns:
            frame[column] = [[] for _ in range(len(frame))]
    if "task_index_high_level" in frame.columns:
        frame["task_index_high_level"] = -1
    return frame


def build_working_root(
    source: Path, target: Path, *, link_videos: bool = True, overwrite: bool = False
) -> None:
    if target.exists():
        if not overwrite:
            raise SystemExit(f"refusing to overwrite existing root: {target}")
        shutil.rmtree(target)
    target.mkdir(parents=True)

    meta_src, meta_dst = source / "meta", target / "meta"
    shutil.copytree(meta_src, meta_dst, copy_function=shutil.copyfile)
    for name in ANNOTATION_SIDECARS:
        victim = meta_dst / name
        if victim.exists():
            victim.unlink()

    for path in sorted((source / "data").rglob("*.parquet")):
        relative = path.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        strip_annotations(pd.read_parquet(path)).to_parquet(destination, index=False)

    videos_src = source / "videos"
    if videos_src.exists():
        if link_videos:
            link_video_files(videos_src, target / "videos")
        else:
            shutil.copytree(videos_src, target / "videos", copy_function=shutil.copyfile)

    for extra in ("README.md",):
        candidate = source / extra
        if candidate.exists():
            shutil.copyfile(candidate, target / extra)

    # A previous run's staging directory holds per-episode plan.jsonl files
    # containing the full label list. It is never copied, but assert it.
    staging = target / STAGING_DIR
    if staging.exists():
        shutil.rmtree(staging)


def link_video_files(source_videos: Path, target_videos: Path) -> None:
    """Mirror the video tree with per-FILE links, never a directory symlink.

    Symlinking ``videos/`` as a whole is the obvious way to avoid copying
    hundreds of gigabytes, and it is wrong. A directory symlink resolves its
    parent through the link target, so ``<working root>/videos/..`` lands in the
    *source* dataset -- which still holds ``meta/lerobot_annotations.json`` and,
    worse, a previous run's ``.annotate_staging/episode_*/plan.jsonl``
    containing the exact human labels and boundary timestamps. Stripping the
    working root is pointless if the answer key is one ``..`` away, and
    ``verify_clean`` would happily pass such a root.

    Mirroring the directory structure for real and linking only the leaf files
    removes the traversal entirely: every directory in the working root is a
    genuine directory whose parent is inside the working root, and a hard link
    carries no path back to where it came from.

    There is deliberately **no symlink fallback**. An earlier version fell back
    to per-file symlinks when the working root sat on a different filesystem,
    and that reopened the leak: each symlink names the absolute source path, so
    ``readlink`` on any video yields the source dataset root, from which the
    human labels and a previous run's ``.annotate_staging/*/plan.jsonl`` are
    both readable -- while ``verify_clean()`` reported the root as clean. A
    cross-filesystem working root therefore fails loudly and the caller must
    either place it on the source filesystem or pass ``--copy-videos``.
    """
    for path in sorted(source_videos.rglob("*")):
        if path.is_dir():
            (target_videos / path.relative_to(source_videos)).mkdir(parents=True, exist_ok=True)
    for path in sorted(source_videos.rglob("*")):
        if not path.is_file():
            continue
        destination = target_videos / path.relative_to(source_videos)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(path, destination)
        except OSError as exc:
            raise SystemExit(
                f"cannot hard-link {path} into {destination} ({exc}).\n"
                "The working root must share a filesystem with the source dataset, because "
                "the symlink alternative leaks the source path and with it the ground truth. "
                "Move the working root onto the source filesystem, or pass --copy-videos."
            ) from exc


def verify_clean(root: Path) -> list[str]:
    """Assert the working root leaks no annotation state. Returns problems."""
    problems: list[str] = []
    for name in ANNOTATION_SIDECARS:
        if (root / "meta" / name).exists():
            problems.append(f"sidecar still present: meta/{name}")
    if (root / STAGING_DIR).exists():
        problems.append(f"previous run's {STAGING_DIR}/ survived into the working root")

    # A directory symlink anywhere in the root re-opens the traversal this
    # module exists to close, so check for it explicitly rather than trusting
    # that the builder did the right thing.
    for path in sorted(root.rglob("*")):
        if path.is_symlink() and path.is_dir():
            problems.append(
                f"directory symlink {path.relative_to(root)} -> {os.readlink(path)}; "
                "its parent resolves outside the working root and exposes the source dataset"
            )
    # And confirm the escape is actually closed for the video tree.
    videos = root / "videos"
    if videos.exists():
        escaped = (videos / "..").resolve()
        if escaped != root.resolve():
            problems.append(
                f"videos/.. resolves to {escaped}, not the working root {root.resolve()}"
            )
    for path in sorted((root / "data").rglob("*.parquet")):
        frame = pd.read_parquet(path)
        for column in ANNOTATION_COLUMNS:
            if column not in frame.columns:
                continue
            non_empty = sum(1 for value in frame[column] if value is not None and len(value) > 0)
            if non_empty:
                problems.append(f"{path.name}: {column} has {non_empty} non-empty rows")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("source", type=Path, help="read-only source dataset root")
    parser.add_argument("target", type=Path, help="working root to create")
    parser.add_argument("--gt-out", type=Path, required=True, help="where to write ground truth JSON")
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--copy-videos", action="store_true", help="copy instead of symlink videos")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    args = parser.parse_args()

    data_files = sorted((args.source / "data").rglob("*.parquet"))
    if not data_files:
        raise SystemExit(f"no data parquet under {args.source}")

    try:
        provenance = verify_human_provenance(args.source)
    except ProvenanceError as exc:
        print(f"[prepare] REFUSED: {exc}", file=sys.stderr)
        return 2
    truth = ground_truth_from_sidecar(args.source)
    if not truth:
        raise SystemExit(f"{args.source.name}: sidecar parsed to zero usable spans")

    # Cross-check the authoritative sidecar against the parquet column. They
    # should agree; a disagreement means the column was overwritten by a later
    # run and is worth knowing about, but the sidecar always wins.
    column = extract_ground_truth(data_files)
    shared = sorted(set(truth) & set(column))
    mismatched = [
        e for e in shared
        if [s["text"] for s in truth[e]] != [s["text"] for s in column[e]]
    ]
    if mismatched:
        print(
            f"[prepare] note: {len(mismatched)}/{len(shared)} episodes differ between "
            f"meta/lerobot_annotations.json and language_persistent; using the sidecar.",
            file=sys.stderr,
        )

    name = args.dataset_name or args.source.name
    args.gt_out.parent.mkdir(parents=True, exist_ok=True)
    args.gt_out.write_text(
        json.dumps(
            {
                "dataset": name,
                "source": str(args.source),
                "n_episodes_with_gt": len(truth),
                "provenance": provenance,
                "gt_source": "meta/lerobot_annotations.json",
                "column_agrees": len(shared) - len(mismatched),
                "column_disagrees": len(mismatched),
                "episodes": {str(k): v for k, v in sorted(truth.items())},
            },
            indent=1,
        ),
        encoding="utf-8",
    )

    build_working_root(
        args.source, args.target, link_videos=not args.copy_videos, overwrite=args.overwrite
    )

    print(f"[prepare] {name}: {len(truth)} episodes with ground truth -> {args.gt_out}")
    print(f"[prepare] working root -> {args.target}")

    if args.verify:
        problems = verify_clean(args.target)
        if problems:
            for problem in problems:
                print(f"[prepare] LEAK: {problem}", file=sys.stderr)
            raise SystemExit("working root still contains annotation state")
        print("[prepare] verified: working root carries no annotation state")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
