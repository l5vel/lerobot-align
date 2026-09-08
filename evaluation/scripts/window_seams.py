#!/usr/bin/env python
"""Do window seams leak into the predicted boundaries, and does the tail-balancing help?

Both tools cut a long episode into windows, prompt the model on each window
separately, and stitch the results. A seam is therefore a place where the model
is *structurally* likely to end a span -- it never saw across it -- and seam
boundaries are artifacts of the harness rather than observations about the robot.

The two tools lay windows out differently, and that is the point of measuring:

  upstream  fixed windows of `max_frames_per_prompt / fps` seconds, with
            whatever short tail is left over.
  align     capacity is `max_frames_per_prompt - 1` INTERVALS, so prefix
            windows are slightly shorter, and the final two are rebalanced so
            the episode does not end in a sliver.

Each arm is therefore scored against ITS OWN seam grid. Scoring both against a
single grid is not a comparison, it is a measurement of one tool taken twice --
an earlier version of this script did exactly that, put align's boundaries on
upstream's 30.0s grid, and produced a 22%-vs-7% "advantage" that was an artifact
of the wrong denominator.

`human_on_seam` is the chance baseline: the share of the *annotators'* own
boundaries that fall near the same seams. Seam coincidence above that is
attributable to the harness; at or below it is what any segmentation of this
corpus would show.

    python window_seams.py --predictions-dir DIR --gt-dir DIR --out results/window_seams.json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

# (frames_per_second, max_frames_per_prompt) per profile, mirroring arms.yaml.
PROFILES = {
    "udef": (2.0, 60),
    "wrap": (3.0, 300),
    "udefthink": (2.0, 60),
}


def upstream_seams(duration: float, fps: float, budget: int) -> list[float]:
    """Fixed windows of `budget / fps` seconds; the tail is whatever remains."""
    if int(round(duration * fps)) + 1 <= budget:
        return []
    window_s = budget / fps
    seams, position = [], window_s
    while position < duration - 1e-6:
        seams.append(position)
        position += window_s
    return seams


def align_seams(duration: float, fps: float, budget: int) -> list[float]:
    """Frame-budgeted windows with the final PAIR rebalanced.

    Mirrors `_generation_windows`: capacity is measured in intervals, not
    frames, and the last two windows split the remainder evenly rather than
    leaving a sliver.
    """
    if int(round(duration * fps)) + 1 <= budget or budget <= 1:
        return []
    capacity = budget - 1
    total_intervals = max(1, int(round(duration * fps)))
    window_count = max(1, math.ceil(total_intervals / capacity))
    if window_count == 1:
        return []
    seams, start = [], 0.0
    for _ in range(max(0, window_count - 2)):
        start += capacity / fps
        seams.append(start)
    remaining_intervals = total_intervals - max(0, window_count - 2) * capacity
    if remaining_intervals <= 0:
        return seams
    left = remaining_intervals // 2
    seams.append(start + (duration - start) * left / remaining_intervals)
    return seams


SEAM_RULE = {"upstream": upstream_seams, "align": align_seams}


def interior_boundaries(spans: list[dict]) -> list[float]:
    return [float(s["end"]) for s in spans[:-1]] if len(spans) > 1 else []


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, )
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--gt-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tolerance", type=float, default=0.5)
    args = ap.parse_args()

    truth: dict[tuple[str, int], list[dict]] = {}
    for path in sorted(args.gt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for episode, spans in payload["episodes"].items():
            if spans:
                truth[(payload["dataset"], int(episode))] = spans

    arms: dict[str, dict] = collections.defaultdict(
        lambda: {"boundaries": 0, "on_seam": 0, "human_boundaries": 0, "human_on_seam": 0,
                 "episodes": 0, "windowed_episodes": 0}
    )

    for path in sorted(args.predictions_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm, tool = payload.get("arm"), payload.get("tool")
        if not arm or tool not in SEAM_RULE or payload.get("repeat"):
            continue
        parts = arm.split("__")
        if len(parts) != 3 or parts[1] not in PROFILES:
            continue
        fps, budget = PROFILES[parts[1]]
        rule = SEAM_RULE[tool]
        row = arms[arm]
        for episode, spans in (payload.get("episodes") or {}).items():
            gt = truth.get((payload["dataset"], int(episode)))
            if not spans or not gt:
                continue
            duration = float(gt[-1]["end"]) - float(gt[0]["start"])
            seams = rule(duration, fps, budget)
            row["episodes"] += 1
            if not seams:
                continue
            row["windowed_episodes"] += 1
            for boundary in interior_boundaries(spans):
                row["boundaries"] += 1
                if min(abs(boundary - s) for s in seams) <= args.tolerance:
                    row["on_seam"] += 1
            # Chance baseline on the SAME episodes and the SAME seams.
            for boundary in interior_boundaries(gt):
                row["human_boundaries"] += 1
                if min(abs(boundary - s) for s in seams) <= args.tolerance:
                    row["human_on_seam"] += 1

    report = {}
    for arm, row in sorted(arms.items()):
        report[arm] = row | {
            "seam_share": row["on_seam"] / row["boundaries"] if row["boundaries"] else None,
            "human_seam_share": (
                row["human_on_seam"] / row["human_boundaries"] if row["human_boundaries"] else None
            ),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Seam coincidence within +/-{args.tolerance}s, each arm on its OWN window grid.")
    print(f"{'arm':<40}{'wnd eps':>8}{'bounds':>8}{'on seam':>9}{'human':>8}{'excess':>8}")
    for arm, row in report.items():
        if not row["boundaries"]:
            print(f"{arm:<40}{row['windowed_episodes']:>8}{0:>8}{'--':>9}{'--':>8}{'--':>8}")
            continue
        excess = row["seam_share"] - (row["human_seam_share"] or 0.0)
        print(
            f"{arm:<40}{row['windowed_episodes']:>8}{row['boundaries']:>8}"
            f"{row['seam_share']:>8.1%}{row['human_seam_share']:>8.1%}{excess:>+8.1%}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
