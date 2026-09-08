#!/usr/bin/env python
"""Assemble a LeRobot v2.1 root holding just the selected RoboInter episodes.

RoboInter ships RH20T as 83 tar-packed chunks of 1,000 episodes: 82,894 episodes,
17.4 GB of parquet and 75.7 GB of video. The Corpus B design scores roughly 2,100
trajectories of that (`corpus_b_plan.md` section 4.2), so converting the whole
dataset to v3.0 would spend hours rewriting data no arm ever opens.

This builds a v2.1 root containing only the requested episodes, so LeRobot's own
shipped `convert_dataset_v21_to_v30` -- not anything hand-rolled here -- has a
consistent dataset to work on.

Why this file is fussy
----------------------
It is the seam where FIVE index spaces and TWO file trees must agree: the original
RoboInter `episode_index`, the positional index the converter assigns, the
`episode_index` column inside each parquet, the `meta/*.jsonl` records, and the
video filenames the converter enumerates by `sorted()` position. Every defect
found here so far has had the same shape -- a silent mismatch between two things
that must agree, surfacing as a credible-looking wrong number rather than a crash.
The guards below therefore fail loudly and early, and the build is staged so a
failure leaves nothing usable behind.

Episodes are RENUMBERED to 0..N-1, and the converter forces this
----------------------------------------------------------------
The converter numbers positionally --

    for ep_idx, ep_path in enumerate(sorted(data_dir.glob("*/*.parquet"))):
        ep_metadata = {"episode_index": ep_idx, ...}

-- and then asserts four independent sources agree, two of which are those
positional counters, so the metadata must be 0..N-1 contiguous. A scattered
selection otherwise dies with `ValueError: Number of episodes is not the same`.

`index_map.json` is therefore the ONLY link from a result back to a RoboInter
episode. Downstream artefacts key on the new index; nothing may key on the
original silently.

The `<root>_old` trap
---------------------
The converter ends every successful run with

    shutil.move(str(root), str(old_root))      # <root> -> <root>_old
    shutil.move(str(new_root), str(root))      # <root>_v30 -> <root>

so a converted root always leaves a `<root>_old` sibling. On its NEXT run it does,
before any conversion work and after validating the new root:

    if old_root.is_dir() and root.is_dir():
        shutil.rmtree(str(root))               # deletes the freshly built subset
        shutil.move(str(old_root), str(root))  # restores the PREVIOUS selection

A second build->convert cycle at the same `--out` would therefore convert the
previous selection while `index_map.json` describes the new one -- every score
credited to the wrong RH20T episode, with nothing raised, because both roots are
valid v2.1 and both are renumbered 0..N-1 so the four-way assertion passes. This
script refuses to build when a `<out>_old` or `<out>_v30` sibling exists.

Staged build
------------
Everything is written to `<out>.partial` and moved into place only on success, so
`--out` never holds a half-built subset and the success marker cannot outlive the
tree it describes.

Publishing replaces the WHOLE of `--out`, not just the subtrees written here, so
every state of `--out` is classified before any work starts: absent, empty, or
carrying this script's own marker is allowed; anything else is refused. The
staging directory is likewise only reused when it carries the sentinel this
script writes. Both deletions are recursive, and neither may act on a directory
this script cannot prove it created.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

CHUNK_SIZE = 1000
MARKER = "corpus_b_subset_report.json"
STAGING_SENTINEL = ".corpus_b_staging"


def link_or_copy(src: Path, dst: Path) -> str:
    """Materialise `src` at `dst`, REPLACING whatever is there."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
        return "linked"
    except OSError:
        shutil.copy2(src, dst)
        return "copied"


def column_stats(values: list[int]) -> dict[str, list[float]]:
    """Recompute a v2.1 stats blob for one integer column."""
    n = len(values)
    mean = sum(values) / n if n else 0.0
    var = sum((v - mean) ** 2 for v in values) / n if n else 0.0
    return {
        "min": [min(values) if values else 0],
        "max": [max(values) if values else 0],
        "mean": [mean],
        "std": [math.sqrt(var)],
        "count": [n],
    }


def check_paths(out: Path, source: Path, meta: Path) -> None:
    """`--overwrite` deletes whole subtrees, so establish what `--out` is FIRST."""
    out_resolved = out.resolve()
    for label, other in (("--source", source), ("--meta", meta)):
        other_resolved = other.resolve()
        if out_resolved == other_resolved:
            raise SystemExit(f"--out is the same directory as {label} ({out_resolved}); refusing")
        if other_resolved.is_relative_to(out_resolved):
            raise SystemExit(
                f"--out ({out_resolved}) contains {label} ({other_resolved}); "
                "refusing, because --overwrite would delete it"
            )
        if out_resolved.is_relative_to(other_resolved):
            raise SystemExit(
                f"--out ({out_resolved}) is inside {label} ({other_resolved}); "
                "refusing, so a build can never write into or delete its own inputs"
            )


def check_converter_leftovers(out: Path) -> None:
    """Refuse when a previous conversion's siblings would hijack the next one."""
    for suffix in ("_old", "_v30"):
        stale = out.parent / f"{out.name}{suffix}"
        if stale.is_dir():
            raise SystemExit(
                f"{stale} exists, left by a previous conversion of {out.name}.\n"
                "The converter checks for it BEFORE converting and, if it is there, deletes the\n"
                "freshly built root and restores this one instead -- so the run would convert the\n"
                "PREVIOUS selection while index_map.json describes the new one, silently crediting\n"
                "every score to the wrong episode.\n"
                f"Remove it first:  rm -rf {stale}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory holding data/chunk-NNN/ and videos/chunk-NNN/")
    parser.add_argument("--meta", type=Path, required=True,
                        help="directory holding the upstream info.json, episodes.jsonl, "
                             "tasks.jsonl, episodes_stats.jsonl")
    parser.add_argument("--episodes", type=Path, required=True,
                        help="JSON file: a list of original episode indices, or an object with an 'episodes' list")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--video-keys", default=None,
                        help="comma-separated camera keys. Default: every video feature in info.json. "
                             "Any key given must exist there, or the converter and this build would "
                             "disagree about which cameras the dataset has.")
    parser.add_argument("--overwrite", action="store_true",
                        help="rebuild into an existing --out (which must have been built by this script)")
    parser.add_argument("--index-map-out", type=Path, default=None,
                        help="where to write index_map.json. Defaults to a SIBLING of --out, never "
                             "inside it: the converter renames the whole root aside as <root>_old, "
                             "which would take the audit map with it.")
    args = parser.parse_args()

    check_paths(args.out, args.source, args.meta)
    check_converter_leftovers(args.out)

    for required in ("info.json", "episodes.jsonl", "tasks.jsonl", "episodes_stats.jsonl"):
        if not (args.meta / required).exists():
            raise SystemExit(f"--meta is missing {required}; the converter requires all four")

    info: dict[str, Any] = json.loads((args.meta / "info.json").read_text(encoding="utf-8"))
    declared_video_keys = [k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]
    if args.video_keys:
        video_keys = [k for k in args.video_keys.split(",") if k]
        unknown = [k for k in video_keys if k not in declared_video_keys]
        if unknown:
            raise SystemExit(
                f"--video-keys names {unknown}, which info.json does not declare as video features "
                f"(it has {declared_video_keys}). The converter takes its camera list from info.json, "
                "so the build and the conversion would disagree about which cameras exist."
            )
    else:
        video_keys = declared_video_keys
    if not video_keys:
        raise SystemExit("info.json declares no video features")

    payload = json.loads(args.episodes.read_text(encoding="utf-8"))
    wanted = payload if isinstance(payload, list) else payload["episodes"]
    wanted = sorted({int(e) for e in wanted})
    if not wanted:
        raise SystemExit("no episodes requested")

    # new index == rank in the sorted original ordering, so the converter's
    # sorted-glob position matches the metadata index it asserts against.
    renumber = {original: new for new, original in enumerate(wanted)}

    episodes = [
        json.loads(line)
        for line in (args.meta / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_index = {int(e["episode_index"]): e for e in episodes}
    absent = [e for e in wanted if e not in by_index]
    if absent:
        raise SystemExit(f"episodes.jsonl is missing {len(absent)} requested episode(s), e.g. {absent[:5]}")

    stats_by_index: dict[int, dict[str, Any]] = {}
    for line in (args.meta / "episodes_stats.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        index = int(row["episode_index"])
        if index in renumber:
            stats_by_index[index] = row
    # No exemption for the zero-coverage case: it used to short-circuit this check
    # and then skip writing the file entirely, reporting success while producing a
    # root the converter cannot open.
    missing_stats = [e for e in wanted if e not in stats_by_index]
    if missing_stats:
        raise SystemExit(
            f"episodes_stats.jsonl covers {len(stats_by_index)} of {len(wanted)} selected episode(s); "
            f"missing e.g. {missing_stats[:5]}. The converter asserts these agree one-for-one."
        )

    # ---- staged build ---------------------------------------------------
    # Publishing replaces the WHOLE of --out (`shutil.rmtree` then rename), not
    # just the three subtrees this script writes. So every state of --out has to
    # be classified here, before any work: an unrecognised directory must be
    # refused rather than silently widened into by a flag whose stated purpose is
    # "rebuild". An earlier version only checked --out when data/, videos/ or
    # meta/ were present, which left a --out holding anything else unguarded.
    if args.out.exists():
        if not args.out.is_dir():
            raise SystemExit(f"--out ({args.out}) exists and is not a directory; refusing")
        contents = sorted(p.name for p in args.out.iterdir())
        existing = [p for p in ("data", "videos", "meta") if (args.out / p).exists()]
        if not contents:
            pass  # an empty directory is ours to fill
        elif (args.out / MARKER).exists():
            if not args.overwrite:
                raise SystemExit(
                    f"{args.out} already holds a subset built by this script. Pass --overwrite to "
                    "rebuild it. Refusing silently to merge two selections: episode numbering is "
                    "positional, so leftovers pair one selection's ground truth with another's video."
                )
        elif existing:
            raise SystemExit(
                f"{args.out} holds {', '.join(existing)} but no {MARKER}, so it was not built by "
                f"this script. Refusing to delete it.\nIf it is a converted root, remove it AND its "
                f"{args.out.name}_old sibling by hand."
            )
        else:
            raise SystemExit(
                f"--out ({args.out}) is a non-empty directory this script did not create "
                f"(contains {', '.join(contents[:6])}{'...' if len(contents) > 6 else ''}).\n"
                "Publishing would delete all of it. Refusing; choose an empty or new --out."
            )

    # The staging directory is deleted outright too, so it is only ever reused
    # when it carries the sentinel this script writes into it.
    staging = args.out.parent / f"{args.out.name}.partial"
    if staging.exists():
        if not staging.is_dir():
            raise SystemExit(f"{staging} exists and is not a directory; refusing")
        if not (staging / STAGING_SENTINEL).exists():
            raise SystemExit(
                f"{staging} exists but carries no {STAGING_SENTINEL}, so this script did not "
                "create it. Refusing to delete it; move it aside or choose another --out."
            )
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    (staging / STAGING_SENTINEL).write_text(
        "Staging directory written by build_corpus_b_subset.py. Safe to delete.\n", encoding="utf-8")

    counts: dict[str, int] = {}
    index_map: list[dict[str, Any]] = []
    frame_cursor = 0

    for original in wanted:
        new = renumber[original]
        src_chunk = original // CHUNK_SIZE
        dst_chunk = new // CHUNK_SIZE

        src = args.source / "data" / f"chunk-{src_chunk:03d}" / f"episode_{original:06d}.parquet"
        if not src.exists():
            raise SystemExit(
                f"episode {original} has no parquet at {src}. Renumbering is positional, so a "
                "partial build would silently shift every later episode."
            )

        # Every camera must be present BEFORE the data file is written, or an
        # episode ends up with data and no video and the per-camera sorted()
        # enumerations disagree about which episode is which from there on.
        video_sources = {}
        for key in video_keys:
            vsrc = args.source / "videos" / f"chunk-{src_chunk:03d}" / key / f"episode_{original:06d}.mp4"
            if not vsrc.exists():
                raise SystemExit(
                    f"episode {original} has no {key} video at {vsrc}. Every selected episode must "
                    "carry every camera the dataset declares."
                )
            video_sources[key] = vsrc

        table = pd.read_parquet(src)
        # The filename drives the renumbering; the column is what downstream code
        # reads. Confirm they agreed before overwriting the column, so a mis-packed
        # source chunk cannot be laundered into a plausible-looking subset.
        if "episode_index" in table.columns and len(table):
            found = int(table["episode_index"].iloc[0])
            if found != original:
                raise SystemExit(
                    f"{src} is named for episode {original} but its episode_index column says {found}"
                )
        episode_name = (
            str(table["episode_name"].iloc[0])
            if "episode_name" in table.columns and len(table) else None
        )
        if "episode_index" in table.columns:
            table["episode_index"] = new
        if "index" in table.columns:
            table["index"] = range(frame_cursor, frame_cursor + len(table))

        dst = staging / "data" / f"chunk-{dst_chunk:03d}" / f"episode_{new:06d}.parquet"
        dst.parent.mkdir(parents=True, exist_ok=True)
        table.to_parquet(dst, index=False)
        counts["data:rewritten"] = counts.get("data:rewritten", 0) + 1

        for key, vsrc in video_sources.items():
            how = link_or_copy(
                vsrc, staging / "videos" / f"chunk-{dst_chunk:03d}" / key / f"episode_{new:06d}.mp4"
            )
            counts[f"video:{how}"] = counts.get(f"video:{how}", 0) + 1

        index_map.append({
            "new_index": new,
            "original_index": original,
            "episode_name": episode_name,
            "source_chunk": src_chunk,
            "n_frames": len(table),
            "frame_from": frame_cursor,
            "frame_to": frame_cursor + len(table),
        })
        frame_cursor += len(table)

    # ---- meta -----------------------------------------------------------
    meta_out = staging / "meta"
    meta_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.meta / "tasks.jsonl", meta_out / "tasks.jsonl")

    kept_episodes = []
    for original in wanted:
        row = dict(by_index[original])
        row["episode_index"] = renumber[original]
        kept_episodes.append(row)

    kept_stats = []
    for entry in index_map:
        row = dict(stats_by_index[entry["original_index"]])
        row["episode_index"] = entry["new_index"]
        # The nested blob describes COLUMNS, and two of those columns were just
        # rewritten. Carrying the originals forward leaves the dataset asserting
        # statistics for values it no longer contains.
        stats = {k: (dict(v) if isinstance(v, dict) else v) for k, v in row.get("stats", {}).items()}
        if "episode_index" in stats:
            stats["episode_index"] = column_stats([entry["new_index"]] * entry["n_frames"])
        if "index" in stats:
            stats["index"] = column_stats(list(range(entry["frame_from"], entry["frame_to"])))
        row["stats"] = stats
        kept_stats.append(row)

    def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    write_jsonl(meta_out / "episodes.jsonl", kept_episodes)
    write_jsonl(meta_out / "episodes_stats.jsonl", kept_stats)

    info["total_episodes"] = len(kept_episodes)
    info["total_frames"] = frame_cursor
    info["total_chunks"] = len({e["new_index"] // CHUNK_SIZE for e in index_map})
    info["total_videos"] = len(kept_episodes) * len(video_keys)
    if "splits" in info:
        info["splits"] = {"train": f"0:{len(kept_episodes)}"}
    (meta_out / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")

    # ---- integrity, before anything is published -------------------------
    n_data = len(list((staging / "data").glob("*/*.parquet")))
    if n_data != len(kept_episodes):
        raise SystemExit(f"{n_data} parquet file(s) staged but {len(kept_episodes)} episode(s) selected")
    for key in video_keys:
        n_video = len(list((staging / "videos").glob(f"*/{key}/*.mp4")))
        if n_video != len(kept_episodes):
            raise SystemExit(f"{n_video} video file(s) for {key} but {len(kept_episodes)} episode(s) selected")

    index_map_payload = json.dumps({
        "note": "Episodes are renumbered 0..N-1 because the v2.1->v3.0 converter numbers "
                "positionally. Downstream artefacts key on new_index; this map is the only "
                "link back to the RoboInter episode.",
        "source": str(args.source),
        "n_episodes": len(index_map),
        "episodes": index_map,
    }, indent=1)

    report = {
        "source": str(args.source),
        "out": str(args.out),
        "n_requested": len(wanted),
        "n_episodes": len(kept_episodes),
        "n_frames": frame_cursor,
        "renumbered": True,
        "new_index_range": [0, len(kept_episodes) - 1],
        "original_index_range": [wanted[0], wanted[-1]],
        "video_keys": video_keys,
        "file_counts": counts,
        "episodes_stats_carried": len(kept_stats),
    }
    (staging / MARKER).write_text(json.dumps(report, indent=1), encoding="utf-8")
    (staging / "index_map.json").write_text(index_map_payload, encoding="utf-8")

    # ---- publish atomically ---------------------------------------------
    # Safe because the guard above established that --out is absent, empty, or a
    # subset this script built.
    (staging / STAGING_SENTINEL).unlink(missing_ok=True)
    if args.out.exists():
        shutil.rmtree(args.out)
    staging.rename(args.out)

    index_map_out = args.index_map_out or args.out.parent / f"{args.out.name}.index_map.json"
    index_map_out.parent.mkdir(parents=True, exist_ok=True)
    index_map_out.write_text(index_map_payload, encoding="utf-8")
    report["index_map"] = str(index_map_out)
    (args.out.parent / f"{args.out.name}.subset_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")

    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
