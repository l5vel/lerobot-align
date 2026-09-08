#!/usr/bin/env python
"""Score every prediction file against ground truth and emit one flat table.

Reads the per-(dataset, arm) prediction JSONs produced by ``run_arm.py`` and
``reference_arms.py``, scores each episode with the full metric suite, and
writes one row per (dataset, episode, arm, matcher) to JSONL.

Every arm is scored under EVERY configured matcher. That is deliberate: the
headline comparison has to be shown to survive a change of matcher, and the
only way to show it is to compute all of them and report the spread. Choosing
the matcher after seeing which one favours the new tool would invalidate the
result, so the set is fixed here in code rather than passed per run.

Episodes an arm failed to annotate are written as ``ok=false`` rows rather
than omitted, so the failure rate survives into the aggregation.

Every row carries its ``repeat`` index. E1 runs the same arm three times to
measure decoding noise, and without that field those repeats are
indistinguishable in ``scores.jsonl``: they would silently triple-count into
the E3 aggregate as if they were separate episodes, and the noise band could
not be recovered even in principle. Aggregation reads repeat 0 by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from gate_fingerprint import PROVENANCE_KEYS  # noqa: E402

from metrics.score import score_episode  # noqa: E402
from metrics.semantic import (  # noqa: E402
    EmbeddingMatcher,
    ExactMatcher,
    TokenF1Matcher,
)


def build_matchers(names: Iterable[str], embedding_model: str, threshold: float) -> list[Any]:
    out: list[Any] = []
    for name in names:
        if name == "exact":
            out.append(ExactMatcher())
        elif name == "token_f1":
            out.append(TokenF1Matcher())
        elif name == "embedding":
            out.append(EmbeddingMatcher(model_name=embedding_model, threshold=threshold))
        else:
            raise SystemExit(f"unknown matcher {name}")
    return out


def load_ground_truth(directory: Path) -> dict[str, dict[int, list[dict[str, Any]]]]:
    out: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        out[payload["dataset"]] = {int(k): v for k, v in payload["episodes"].items()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--matchers", nargs="*", default=["exact", "token_f1", "embedding"])
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-threshold", type=float, default=EmbeddingMatcher.threshold)
    args = parser.parse_args()

    truth = load_ground_truth(args.gt_dir)
    if not truth:
        raise SystemExit(f"no ground truth under {args.gt_dir}")

    splits: dict[str, set[int]] = {}
    for path in sorted(args.splits_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits[payload["dataset"]] = {int(e) for e in payload["eval"]}

    prediction_files = sorted(args.predictions_dir.glob("*.json"))
    if not prediction_files:
        raise SystemExit(f"no predictions under {args.predictions_dir}")

    # Validate before opening the output or loading an encoder. Never infer an
    # evaluation split from whatever human labels happen to be available.
    for path in prediction_files:
        dataset = json.loads(path.read_text(encoding="utf-8"))["dataset"]
        if dataset not in truth:
            raise SystemExit(f"no ground truth for {dataset}")
        if dataset not in splits or not splits[dataset]:
            raise SystemExit(f"missing or empty evaluation split for {dataset}")
        missing = splits[dataset] - set(truth[dataset])
        if missing:
            raise SystemExit(f"evaluation episodes missing ground truth for {dataset}: {sorted(missing)}")

    matchers = build_matchers(args.matchers, args.embedding_model, args.embedding_threshold)
    for matcher in matchers:
        if isinstance(matcher, EmbeddingMatcher):
            vocabulary = {
                str(span.get("text", ""))
                for episodes in truth.values()
                for spans in episodes.values()
                for span in spans
            }
            for path in prediction_files:
                payload = json.loads(path.read_text(encoding="utf-8"))
                for spans in (payload.get("episodes") or {}).values():
                    vocabulary.update(str(s.get("text", "")) for s in (spans or [])
                                      if isinstance(s, dict))
            print(f"[score] warming embedding cache on {len(vocabulary)} distinct labels")
            matcher.warm(vocabulary)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", encoding="utf-8") as handle:
        for path in prediction_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            dataset, arm = payload["dataset"], payload["arm"]
            repeat = int(payload.get("repeat", 0) or 0)
            # Carried so the C1 gate can prove every prediction it certifies was
            # produced by the source it claims, rather than served from a cache
            # written by different code.
            provenance = {key: payload.get(key) for key in PROVENANCE_KEYS}
            if dataset not in truth:
                print(f"[score] SKIP {path.name}: no ground truth for {dataset}", file=sys.stderr)
                continue
            reference_all = truth[dataset]
            scored_episodes = splits[dataset]
            predictions = {int(k): v for k, v in (payload.get("episodes") or {}).items()}

            for episode in sorted(scored_episodes):
                reference = reference_all.get(episode)
                if not reference:
                    continue
                predicted = predictions.get(episode)
                for matcher in matchers:
                    if not predicted:
                        row = {
                            "dataset": dataset, "episode": episode, "arm": arm,
                            "repeat": repeat, **provenance,
                            "matcher": matcher.name, "ok": False,
                            "error": "arm produced no spans for this episode",
                            "supervision": payload.get("supervision"),
                            "tool": payload.get("tool"),
                        }
                    else:
                        score = score_episode(
                            dataset=dataset, episode=episode, arm=arm,
                            predicted=predicted, reference=reference, matcher=matcher,
                        )
                        row = score.flat()
                        row["repeat"] = repeat
                        row.update(provenance)
                        row["supervision"] = payload.get("supervision")
                        row["tool"] = payload.get("tool")
                        row["elapsed_seconds"] = payload.get("elapsed_seconds")
                    row["embedding_threshold"] = (
                        args.embedding_threshold if matcher.name == "embedding" else None)
                    row["embedding_model"] = (
                        args.embedding_model if matcher.name == "embedding" else None)
                    row["metric_version"] = 2
                    handle.write(json.dumps(row) + "\n")
                    written += 1
    print(f"[score] wrote {written} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
