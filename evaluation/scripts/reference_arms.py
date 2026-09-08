#!/usr/bin/env python
"""Model-free reference arms.

These arms never look at a video frame. They exist because the L5vel tasks are
scripted -- several datasets have a single human label sequence across all
fifty episodes -- so a method that has seen a few labelled episodes can score
well by replaying the script and placing boundaries at their usual relative
positions. Without these numbers on the table, a VLM arm's score is
uninterpretable: there is no way to tell how much of it came from reading the
video and how much from the task simply being predictable.

Arms, in increasing order of the supervision they consume:

``single``        one span covering the whole episode. The floor.
``uniform_true``  equal split into the TRUE number of segments. Consumes the
                  per-episode segment count, which no real tool is given, so
                  it is an oracle-count reference, not a competitor.
``uniform_modal`` equal split into the modal segment count from the seed
                  episodes. Uses only seed supervision.
``script_prior``  modal seed label sequence, boundaries at the mean relative
                  positions observed in the seed episodes. This is the arm to
                  beat: it is the strongest thing obtainable without any
                  perception at all.

Every arm that consumes seed episodes uses the SAME seed split as the
calibrated VLM arms, and that split is disjoint from the evaluation episodes.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

Span = dict[str, Any]


def _modal_sequence(seed: dict[int, list[Span]]) -> list[str]:
    from collections import Counter

    counts = Counter(tuple(str(s["text"]) for s in spans) for spans in seed.values())
    return list(counts.most_common(1)[0][0]) if counts else []


def _mean_relative_boundaries(seed: dict[int, list[Span]], sequence: Sequence[str]) -> list[float]:
    rows: list[list[float]] = []
    for spans in seed.values():
        if tuple(str(s["text"]) for s in spans) != tuple(sequence) or len(spans) < 2:
            continue
        start, end = float(spans[0]["start"]), float(spans[-1]["end"])
        if end <= start:
            continue
        rows.append([(float(s["start"]) - start) / (end - start) for s in spans[1:]])
    if not rows:
        return []
    return [sum(column) / len(column) for column in zip(*rows, strict=True)]


def _modal_count(seed: dict[int, list[Span]]) -> int:
    from collections import Counter

    counts = Counter(len(spans) for spans in seed.values())
    return counts.most_common(1)[0][0] if counts else 1


def _equal_split(duration: float, k: int, labels: Sequence[str] | None = None) -> list[Span]:
    k = max(1, k)
    out: list[Span] = []
    for index in range(k):
        start = duration * index / k
        end = duration * (index + 1) / k
        text = labels[index] if labels is not None and index < len(labels) else f"segment {index + 1}"
        out.append({"start": start, "end": end, "text": text})
    return out


def arm_single(duration: float, **_: Any) -> list[Span]:
    return [{"start": 0.0, "end": duration, "text": "perform the task"}]


def arm_uniform_true(duration: float, *, n_true: int, **_: Any) -> list[Span]:
    return _equal_split(duration, n_true)


def arm_uniform_modal(duration: float, *, seed: dict[int, list[Span]], **_: Any) -> list[Span]:
    return _equal_split(duration, _modal_count(seed))


def arm_script_prior(duration: float, *, seed: dict[int, list[Span]], **_: Any) -> list[Span]:
    """Replay the modal seed script at its mean relative boundary positions."""
    sequence = _modal_sequence(seed)
    if not sequence:
        return arm_single(duration)
    fractions = _mean_relative_boundaries(seed, sequence)
    if len(fractions) != len(sequence) - 1:
        return _equal_split(duration, len(sequence), sequence)
    cuts = [0.0] + [duration * f for f in fractions] + [duration]
    cuts = sorted(cuts)
    return [
        {"start": cuts[i], "end": cuts[i + 1], "text": sequence[i]} for i in range(len(sequence))
    ]


ARMS = {
    "ref_single": arm_single,
    "ref_uniform_true": arm_uniform_true,
    "ref_uniform_modal": arm_uniform_modal,
    "ref_script_prior": arm_script_prior,
}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--gt", type=Path, required=True, help="ground truth JSON from prepare_dataset.py")
    parser.add_argument("--split", type=Path, required=True, help="JSON with seed/eval episode lists")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", default=sorted(ARMS))
    args = parser.parse_args()

    truth_payload = json.loads(args.gt.read_text(encoding="utf-8"))
    dataset = truth_payload["dataset"]
    truth = {int(k): v for k, v in truth_payload["episodes"].items()}

    split = json.loads(args.split.read_text(encoding="utf-8"))
    seed_ids = [int(e) for e in split["seed"]]
    eval_ids = [int(e) for e in split["eval"]]
    overlap = set(seed_ids) & set(eval_ids)
    if overlap:
        raise SystemExit(f"seed and eval episodes overlap: {sorted(overlap)}")

    seed = {e: truth[e] for e in seed_ids if e in truth}
    if not seed:
        raise SystemExit("no seed episodes carry ground truth")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm}; known: {sorted(ARMS)}")
        function = ARMS[arm]
        predictions: dict[str, list[Span]] = {}
        for episode in eval_ids:
            spans = truth.get(episode)
            if not spans:
                continue
            duration = float(spans[-1]["end"]) - float(spans[0]["start"])
            predictions[str(episode)] = function(duration, seed=seed, n_true=len(spans))
        path = args.out_dir / f"{dataset}__{arm}.json"
        path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "tool": "reference",
                    "uses_video": False,
                    "seed_episodes": seed_ids,
                    "episodes": predictions,
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        print(f"[reference] {dataset} {arm}: {len(predictions)} episodes -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
