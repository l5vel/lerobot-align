#!/usr/bin/env python
"""Prepare a RoboInter-Data subset as the contamination-free evaluation corpus.

Corpus A (the L5vel components) is the corpus this tool was developed on, so no
result measured there can rule out the tuning advantage described in
`evaluation_plan.md` §10.1. RoboInter-Data is external to that development, so
it is the only place a contamination-free claim can be made -- which is why it
is mandatory rather than optional.

Format
------
`Annotation_with_action_lerobotv21/` is LeRobot **v2.1**, so it needs converting
before either tool can read it. The conversion is done by LeRobot's own shipped
`lerobot.scripts.convert_dataset_v21_to_v30`, not by anything written here: a
hand-rolled converter would be one more thing a reader has to trust, and a bug
in it would be indistinguishable from a tool difference.

Ground truth
------------
Per-frame `annotation.time_clip` is a JSON string `[[start_frame, end_frame],
...]`, identical on every frame of an episode, paired with per-frame
`annotation.substask` (the typo is the real field name) and
`annotation.primitive_skill`. At 10 fps a clip `[s, e]` is `[s/10, (e+1)/10]`
seconds -- note the `+1`, because `e` is the last INCLUDED frame, and dropping
it would shift every boundary by 0.1 s in the tool's favour or against it
depending on the metric.

The invariants this relies on are re-checked per episode rather than assumed,
and an episode that violates any of them is skipped and counted:

* `time_clip` constant across the episode's frames
* clips contiguous, `next.start == prev.end + 1`
* clips covering `[0, n_frames - 1]`
* clip starts coinciding with `(substask, primitive_skill)` run changes

Provenance
----------
**This script refuses to run unless `--provenance-verified` is passed.** The
dataset card must be read and its statement about how the boundaries were
produced quoted into `analysis/corpus_b_selection.md` first. If the boundaries
turn out to be model-generated, this corpus measures agreement with another
model and must not be used; that check is a human judgement and is not
automated away here.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

TIME_CLIP = "annotation.time_clip"
SUBTASK = "annotation.substask"  # sic: the upstream field name contains this typo
SKILL = "annotation.primitive_skill"
ANNOTATION_COLUMNS = (TIME_CLIP, SUBTASK, SKILL)


def parse_time_clip(value: Any) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    try:
        raw = json.loads(value) if isinstance(value, str) else list(value)
        return [(int(a), int(b)) for a, b in raw]
    except Exception:
        return None


def episode_spans(
    frame: pd.DataFrame, fps: float
) -> tuple[list[dict[str, Any]] | None, str, int]:
    """Reconstruct spans for one episode, or return why it was rejected.

    Returns (spans, reason, gap_frames). `gap_frames` counts frames lying in an
    annotation gap between two subtasks; it is 0 for a contiguous episode.
    """
    if TIME_CLIP not in frame.columns or SUBTASK not in frame.columns:
        return None, "missing annotation columns", 0

    clips = parse_time_clip(frame[TIME_CLIP].iloc[0])
    if not clips:
        return None, "unparseable time_clip", 0

    distinct = {str(v) for v in frame[TIME_CLIP]}
    if len(distinct) != 1:
        return None, f"time_clip varies within the episode ({len(distinct)} distinct values)", 0

    n = len(frame)
    # Tolerances below are all +/-1 FRAME, which at 10 fps is 0.1 s. They exist
    # because the real annotations carry exactly that much slop and rejecting on
    # it discarded 23.6% of the multi-clip episodes -- 88 of 399 in a 1,414
    # episode probe -- for a discrepancy far below the 0.25 s at which the
    # annotators themselves agree. See `corpus_b_plan.md` section 2.5.
    if clips[0][0] > 1 or abs(clips[-1][1] - (n - 1)) > 1:
        return None, f"clips cover [{clips[0][0]}, {clips[-1][1]}] not [0, {n - 1}]", 0

    # Joins come in three real shapes: exact (next.start == prev.end + 1), a
    # one-frame overlap (next.start == prev.end, e.g. [[0,89],[90,131],[131,172]]),
    # and a genuine GAP of unannotated dead time between two subtasks. Only the
    # first was accepted before. A gap is data, not corruption: it is recorded so
    # scoring can exclude it, and never interpolated away.
    gap_frames = 0
    for index in range(1, len(clips)):
        delta = clips[index][0] - clips[index - 1][1]
        if delta < 0:
            return None, f"clips overlap by {-delta} frames at {index}", 0
        if delta > 1:
            gap_frames += delta - 1

    labels = list(frame[SUBTASK])
    spans: list[dict[str, Any]] = []
    for start, end in clips:
        if start >= len(labels):
            return None, "clip start beyond frame count", 0

        # Take the label from the WHOLE clip, not from labels[start].
        #
        # A single-frame read at the boundary was correct only while clip starts
        # were required to fall exactly on a label run-change: `start` was then
        # the first frame of its own run by construction. The +/-1 tolerance
        # below removes that guarantee, and a clip starting one frame BEFORE its
        # run-change would silently inherit the PREVIOUS clip's label -- which in
        # fixed-label alignment corrupts both the label list handed to the model
        # and the boundaries it is scored against. The mode over the clip's own
        # frames cannot be moved by one frame of slop.
        window = [str(v).strip() for v in labels[start : end + 1] if v is not None and str(v).strip()]
        if not window:
            return None, f"empty subtask label at frame {start}", 0
        counts = Counter(window)
        text, hits = counts.most_common(1)[0]
        # Below this the clip genuinely spans two labels and we do not know which
        # one owns it, so the episode is dropped rather than guessed at. One frame
        # of slop cannot breach it except on a clip only a few frames long.
        if hits / len(window) < 0.6:
            return None, "clip spans two labels with no clear majority", 0

        spans.append(
            {"start": start / fps, "end": (end + 1) / fps, "text": text}
        )

    # The clip boundaries should coincide with label run-changes. Where they
    # disagree by more than a frame the two annotation channels genuinely
    # disagree and we do not know which is authoritative, so the episode is
    # dropped rather than guessed at. Where they disagree by exactly one frame
    # -- 64 of 275 multi-clip episodes in the probe -- it is the same 0.1 s slop
    # as the joins above, and dropping the episode would discard a good
    # annotation over a rounding difference.
    run_changes = {0} | {
        i for i in range(1, len(labels))
        if labels[i] != labels[i - 1]
        or (SKILL in frame.columns and frame[SKILL].iloc[i] != frame[SKILL].iloc[i - 1])
    }
    for start, _ in clips:
        if not any(abs(start - change) <= 1 for change in run_changes):
            return None, "clip starts do not align with subtask/skill run changes", 0

    return spans, "ok", gap_frames


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="converted v3.0 dataset root")
    parser.add_argument("--gt-out", type=Path, required=True)
    parser.add_argument(
        "--annotations-out", type=Path, default=None,
        help="also write the same spans as meta/lerobot_annotations.json, the format "
             "`lerobot-align-fit` reads. RoboInter ships no such file, and the fitter "
             "has no other way in, so calibration cannot be fit without it. This is a "
             "format adapter over the spans already extracted here -- one source of "
             "truth, re-encoded -- not a second reading of the annotations.",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument(
        "--provenance-verified", action="store_true",
        help="Confirm the dataset card's statement on how the boundaries were produced "
             "has been read and quoted into analysis/corpus_b_selection.md.",
    )
    args = parser.parse_args()

    if not args.provenance_verified:
        print(
            "REFUSED: pass --provenance-verified only after reading the RoboInter-Data "
            "card and quoting its statement on annotation provenance into "
            "analysis/corpus_b_selection.md. If the boundaries are model-generated, "
            "this corpus measures agreement with another model and must not be used.",
            file=sys.stderr,
        )
        return 2

    files = sorted((args.root / "data").rglob("*.parquet"))
    if not files:
        raise SystemExit(f"no data parquet under {args.root}")

    truth: dict[int, list[dict[str, Any]]] = {}
    rejected: dict[str, int] = {}
    gaps: dict[int, int] = {}
    for path in files:
        table = pd.read_parquet(path)
        if "episode_index" not in table.columns:
            continue
        for episode, group in table.groupby("episode_index"):
            if args.max_episodes and len(truth) >= args.max_episodes:
                break
            spans, reason, gap_frames = episode_spans(group.reset_index(drop=True), args.fps)
            if spans:
                truth[int(episode)] = spans
                if gap_frames:
                    gaps[int(episode)] = gap_frames
            else:
                rejected[reason] = rejected.get(reason, 0) + 1

    if not truth:
        raise SystemExit(f"no usable episodes; rejections: {rejected}")

    args.gt_out.parent.mkdir(parents=True, exist_ok=True)
    args.gt_out.write_text(
        json.dumps(
            {
                "dataset": args.dataset_name,
                "source": str(args.root),
                "corpus": "robointer",
                "fps": args.fps,
                "gt_source": f"{TIME_CLIP} + {SUBTASK}",
                "n_episodes_with_gt": len(truth),
                "rejected": rejected,
                # Episodes whose subtasks do not tile the recording: the value is
                # the number of frames lying between two annotated clips. Recorded
                # rather than interpolated, so scoring can exclude the dead time.
                "gap_frames": {str(k): v for k, v in sorted(gaps.items())},
                "n_episodes_with_gaps": len(gaps),
                "episodes": {str(k): v for k, v in sorted(truth.items())},
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    if args.annotations_out:
        args.annotations_out.parent.mkdir(parents=True, exist_ok=True)
        args.annotations_out.write_text(
            json.dumps(
                {
                    "source": f"{TIME_CLIP} + {SUBTASK} via prepare_robointer.py",
                    "episodes": {
                        str(episode): {
                            "subtasks": [
                                {"label": span["text"], "start": span["start"], "end": span["end"]}
                                for span in spans
                            ]
                        }
                        for episode, spans in sorted(truth.items())
                    },
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"[robointer] annotations -> {args.annotations_out}")

    print(f"[robointer] {args.dataset_name}: {len(truth)} usable episodes -> {args.gt_out}")
    if gaps:
        print(f"[robointer] {len(gaps)} episode(s) carry annotation gaps (recorded, not interpolated)")
    if rejected:
        print(f"[robointer] rejected {sum(rejected.values())} episode(s): {rejected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
