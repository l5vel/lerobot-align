#!/usr/bin/env python
"""Select and render qualitative examples for manual inspection.

Automatic metrics cannot decide whether a differently-worded label is *better*
than the human one, whether a boundary disagreement is the tool's error or the
annotator's, or whether a segment counted as hallucinated is a real event the
human did not label. Those questions need a person, and a person can only look
at a few episodes -- so which few is chosen matters, and choosing them after
seeing the aggregate is how cherry-picking happens.

Selection is therefore mechanical and fixed by rank:

* the ``--k`` best, worst and median episodes per arm, by the ranking metric
* the ``--k`` episodes where the two nominated arms disagree most, in both
  directions

Selecting failures *and* successes, and disagreements in *both* directions, is
what keeps the sample from being a highlight reel. The rendered timelines print
both segmentations against a common time axis so a reader can judge the
disagreement directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

BAR_WIDTH = 88


def render_timeline(spans: list[dict[str, Any]], duration: float, title: str) -> list[str]:
    """ASCII timeline, one line per span plus a shared axis."""
    lines = [f"  {title}"]
    if duration <= 0:
        return lines + ["    (zero-length episode)"]
    for span in spans:
        start, end = float(span["start"]), float(span["end"])
        left = int(BAR_WIDTH * start / duration)
        width = max(1, int(BAR_WIDTH * (end - start) / duration))
        bar = " " * left + "#" * min(width, BAR_WIDTH - left)
        lines.append(f"    |{bar:<{BAR_WIDTH}}| {start:7.2f}-{end:7.2f}  {span.get('text','')}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matcher", default="embedding")
    parser.add_argument("--metric", default="boundary_f1@1")
    parser.add_argument("--arms", nargs=2, default=["baseline_upstream", "align_video_realign"])
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.scores.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if r.get("matcher") == args.matcher and r.get("ok")]
    if not rows:
        raise SystemExit(f"no usable rows for matcher {args.matcher}")

    truth: dict[str, dict[int, list]] = {}
    for path in sorted(args.gt_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        truth[payload["dataset"]] = {int(k): v for k, v in payload["episodes"].items()}

    predictions: dict[tuple[str, str], dict[int, list]] = {}
    for path in sorted(args.predictions_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        predictions[(payload["dataset"], payload["arm"])] = {
            int(k): v for k, v in (payload.get("episodes") or {}).items()
        }

    selections: list[tuple[str, str, int, str]] = []

    for arm in args.arms:
        scored = sorted(
            ((float(r[args.metric]), r["dataset"], r["episode"])
             for r in rows if r["arm"] == arm and args.metric in r),
            key=lambda t: t[0],
        )
        if not scored:
            continue
        middle = len(scored) // 2
        for tag, chosen in (
            ("worst", scored[: args.k]),
            ("median", scored[max(0, middle - args.k // 2): max(0, middle - args.k // 2) + args.k]),
            ("best", scored[-args.k:]),
        ):
            for value, dataset, episode in chosen:
                selections.append((f"{arm}:{tag}({value:.3f})", dataset, episode, arm))

    by_key: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if args.metric in row:
            by_key[(row["dataset"], row["episode"])][row["arm"]] = float(row[args.metric])
    left, right = args.arms
    deltas = sorted(
        ((v[right] - v[left], k[0], k[1]) for k, v in by_key.items() if left in v and right in v),
        key=lambda t: t[0],
    )
    for value, dataset, episode in deltas[: args.k]:
        selections.append((f"disagree:{right}_worse({value:+.3f})", dataset, episode, right))
    for value, dataset, episode in deltas[-args.k:]:
        selections.append((f"disagree:{right}_better({value:+.3f})", dataset, episode, right))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    seen: set[tuple[str, int]] = set()
    with args.out.open("w", encoding="utf-8") as handle:
        handle.write("# Qualitative examples\n\n")
        handle.write(
            f"Selection is mechanical: {args.k} best / median / worst per arm by "
            f"`{args.metric}` ({args.matcher} matcher), plus the {args.k} largest "
            f"disagreements in each direction between `{left}` and `{right}`. "
            "Successes and failures are selected by the same rule, so this is not "
            "a highlight reel.\n\n"
        )
        for tag, dataset, episode, _arm in selections:
            reference = truth.get(dataset, {}).get(episode)
            if not reference:
                continue
            duration = float(reference[-1]["end"]) - float(reference[0]["start"])
            handle.write(f"\n## {dataset} episode {episode} — {tag}\n\n```\n")
            for line in render_timeline(reference, duration, "HUMAN"):
                handle.write(line + "\n")
            for arm in args.arms:
                spans = predictions.get((dataset, arm), {}).get(episode)
                if spans:
                    handle.write("\n")
                    for line in render_timeline(spans, duration, arm.upper()):
                        handle.write(line + "\n")
            handle.write("```\n")
            seen.add((dataset, episode))

    print(f"[examples] {len(seen)} distinct episodes rendered -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
