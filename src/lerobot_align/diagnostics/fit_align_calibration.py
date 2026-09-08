#!/usr/bin/env python
"""Fit per-boundary timing offsets from alignment results against human spans.

The model's boundary error against a human reference is largely a fixed offset
rather than noise — it marks subtasks as starting consistently later than the
annotator did — and that offset survives both prompt rewrites and large changes
in frame count. Subtracting it is worth about +9 macro-tIoU points on held-out
episodes when fit on ten labelled ones.

Usage:

    # 1. score some episodes you have human spans for
    python eval_align_batch.py ROOT --episodes 0 1 2 3 4 5 6 7 8 9 \\
        --formats video --out fit.json

    # 2. fit the offsets
    python fit_align_calibration.py ROOT fit.json --out calibration.json

    # ...or, if step 1 died partway, fit from the sidecar it left behind
    python fit_align_calibration.py ROOT fit.jsonl --out calibration.json

    # 3. use them for the rest of the dataset
    lerobot-align --root=ROOT --plan.subtasks_path=labels.json \\
        --plan.subtask_align_frame_format=video \\
        --plan.subtask_align_calibration_path=calibration.json

The offsets encode ONE task and annotation convention. Use ``--episodes`` to
restrict fitting to a fixed seed allocation of at most ten trajectories. The
default exact-label scope preserves the recorded label schema. Explicit
``--label-scope segment_count`` pools differently worded labels within one task
and one segment count; its positional correspondence is an assumption the
caller must establish. Freeze the resulting file before annotating the rest.

Leave-one-out estimates use only fitting episodes. When they also select the
duration weight, their reported gain is optimistic, not an independent test.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from lerobot_align.alignment_metrics import evaluate_alignment
from lerobot_align.diagnostics._provenance import digest, fingerprint, validate_truth


def load_result_rows(path: Path, *, config_fingerprint: str | None = None, frame_format: str | None = None) -> tuple[list[dict], list[str], str]:
    """Read ``eval_align_batch.py`` output as a JSON array or as its JSONL sidecar.

    The sidecar is the only complete record a crashed scoring run leaves behind,
    so refusing to parse it forces a hand re-aggregation at exactly the moment
    the run is least reproducible. Its append-only semantics also make a
    repeated ``(episode, format)`` a later observation rather than the caller
    error the array's duplicate check assumes, so the last line wins -- except
    that an ``error`` row never displaces a scored one. A re-run against a
    flaky endpoint appends failures for episodes that already scored cleanly,
    and taking those would shrink the cohort while recording "prediction error"
    for an episode whose prediction is sitting in the same file.

    Returns the rows, one note per resolved collision, and which form was
    detected, so the calibration payload can record what it was fit from.
    """
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError(f"{path}: file is empty")
    if stripped.startswith("["):
        # Name the file. A bare JSONDecodeError reports only an offset into a
        # file it does not identify, and the caller may be a shell loop.
        try:
            rows = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: not valid JSON ({exc.msg} at line {exc.lineno})") from exc
        if not isinstance(rows, list):
            raise ValueError(f"{path}: expected a JSON array, got {type(rows).__name__}")
        selected = _select_results(_checked_rows(rows, path), path, config_fingerprint, frame_format)
        return selected, [], "json"

    # JSONL. Report the offending line rather than a byte offset into the file:
    # a partial sidecar's last line is routinely a half-written row.
    rows, seen, notes = [], {}, []
    identities = {}
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{number}: expected a JSON object, got {type(row).__name__}")
        selected = _select_results([row], path, config_fingerprint, frame_format, identities)
        if not selected:
            continue
        key = (row.get("episode"), row.get("format"))
        if key not in seen:
            seen[key] = len(rows)
            rows.append({"line": number, "row": row})
            continue
        previous = rows[seen[key]]
        note = _collision_note(key, number, row, previous)
        notes.append(note)
        if note["resolution"] == "superseded":
            rows[seen[key]] = {"line": number, "row": row}
    return _checked_rows([entry["row"] for entry in rows], path), notes, "jsonl"


def _select_results(rows, path, config_fingerprint, frame_format, identities=None):
    identities = {} if identities is None else identities
    selected = []
    for row in rows:
        if frame_format is not None and row.get('format') != frame_format:
            continue
        identity = row.get('config_fingerprint')
        if config_fingerprint is not None and identity != config_fingerprint:
            continue
        if identity is not None:
            provenance = row.get('provenance')
            if not isinstance(provenance, dict) or fingerprint(provenance) != identity:
                raise ValueError(f'{path}: invalid or missing evaluation provenance')
        condition = row.get('format')
        if condition in identities and identities[condition] != identity:
            raise ValueError(f'{path}: mixed evaluation configurations for {condition}; '
                             'select one --config-fingerprint or use separate result files')
        identities[condition] = identity
        selected.append(row)
    return selected


def _collision_note(key: tuple, number: int, row: dict, previous: dict) -> dict:
    """Resolve one repeated ``(episode, format)`` and describe the trade made.

    Kept as a record rather than a printed line because the calibration file is
    frozen and audited later, while the sidecar it names keeps growing: without
    this, the same command re-run after another scoring pass fits a different
    cohort and the artifact reads identically.
    """
    kept_data = "error" not in previous["row"] and "error" in row
    note = {
        "episode": key[0],
        "format": key[1],
        "line": number,
        "previous_line": previous["line"],
        "resolution": "kept_earlier" if kept_data else "superseded",
    }
    for label, source in (("run_id", row), ("previous_run_id", previous["row"])):
        run_id = source.get("run_id")
        if run_id is not None:
            note[label] = run_id
    if kept_data:
        note["detail"] = f"later attempt errored: {row['error']}"
    elif "error" in previous["row"]:
        note["detail"] = f"earlier attempt errored: {previous['row']['error']}"
    return note


def describe_collision(note: dict) -> str:
    """One printable line per collision, in the style of the exclusion reports."""
    verb = "replaces" if note["resolution"] == "superseded" else "kept over"
    runs = ""
    if note.get("run_id") or note.get("previous_run_id"):
        runs = f" [run {note.get('run_id', '?')} vs {note.get('previous_run_id', '?')}]"
    head = (
        f"{note['resolution']} ep {note['episode']} {note['format']}: "
        f"line {note['line']} {verb} line {note['previous_line']}{runs}"
    )
    return head + (f" ({note['detail']})" if "detail" in note else "")


def _checked_rows(rows: list[Any], path: Path) -> list[dict]:
    """Reject non-object rows here so the filters below can index them freely."""
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{path}: row {index} is {type(row).__name__}, expected an object")
    return rows


def human_spans(root: Path) -> dict[int, list[dict]]:
    payload = json.loads((root / "meta" / "lerobot_annotations.json").read_text(encoding="utf-8"))
    out: dict[int, list[dict]] = {}
    for key, episode in (payload.get("episodes") or {}).items():
        spans = episode.get("subtasks") or []
        if spans:
            out[int(key)] = spans
    return validate_truth(out)


def signed_errors(predicted: list[dict], truth: list[dict]) -> dict[int, float]:
    """Predicted-minus-human seconds for a complete, correctly ordered answer.

    A partial answer cannot be compared by list position: its second span may
    refer to the third ground-truth label. Exclude it instead of fitting an
    offset to the wrong semantic boundary.
    """
    reason = unusable_episode_reason(predicted, truth)
    if reason is not None:
        raise ValueError(reason)
    out: dict[int, float] = {}
    for index in range(1, len(truth)):
        out[index - 1] = float(predicted[index]["start"]) - float(truth[index]["start"])
    return out


def episode_duration(spans: list[dict]) -> float:
    return float(spans[-1]["end"]) - float(spans[0]["start"])


def _span_label(span: dict) -> Any:
    # Runtime prediction rows use text; annotation rows use label.
    return span.get("text", span.get("label"))


def unusable_episode_reason(predicted: Any, truth: Any) -> str | None:
    """Return an auditable exclusion reason, or None for a usable full answer."""
    if not isinstance(truth, list) or len(truth) < 2:
        return "ground truth needs at least two segments"
    if not isinstance(predicted, list) or len(predicted) != len(truth):
        return "prediction is incomplete or has a different segment count"
    for name, spans in (("ground truth", truth), ("prediction", predicted)):
        try:
            labels = [_span_label(span) for span in spans]
            starts = [float(span["start"]) for span in spans]
            ends = [float(span["end"]) for span in spans]
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return f"{name} contains malformed spans"
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            return f"{name} contains an empty or invalid label"
        if any(not math.isfinite(value) for value in [*starts, *ends]):
            return f"{name} contains nonfinite timestamps"
        if ends[-1] <= starts[0]:
            return f"{name} has zero or negative episode duration"
        if any(end <= start for start, end in zip(starts, ends, strict=True)):
            return f"{name} contains a zero or negative segment duration"
        if any(right <= left for left, right in zip(starts, starts[1:], strict=False)):
            return f"{name} has non-increasing boundaries"
    if [_span_label(s) for s in predicted] != [_span_label(s) for s in truth]:
        return "prediction labels do not exactly match its ground truth in order"
    return None


def fit(
    results: dict[int, list[dict]], truth: dict[int, list[dict]], episodes: list[int]
) -> dict[str, Any]:
    """Fit offsets and duration priors, both as fractions of episode length.

    Fractions rather than seconds because episodes differ in length — a
    constant second offset over-corrects short episodes and under-corrects long
    ones. Positional estimates are medians; residual scales are pooled standard
    deviations. Only complete predictions enter either set of estimates.
    """
    offsets: dict[int, list[float]] = {}
    segments: dict[int, list[float]] = {}
    n_segments = None
    for episode in episodes:
        spans, gt = results[episode], truth[episode]
        if unusable_episode_reason(spans, gt) is not None:
            continue
        if n_segments is not None and len(gt) != n_segments:
            raise ValueError("a positional calibration requires one segment count")
        n_segments = len(gt)
        duration = episode_duration(spans)
        for boundary, value in signed_errors(spans, gt).items():
            offsets.setdefault(boundary, []).append(value / duration)
        start = float(gt[0]["start"])
        for i in range(len(gt)):
            end = float(gt[i + 1]["start"]) if i + 1 < len(gt) else float(gt[-1]["end"])
            segments.setdefault(i, []).append((end - start) / duration)
            start = end
    if not offsets:
        return {}
    off = [statistics.median(offsets.get(i, [0.0])) for i in range(max(offsets) + 1)]
    seg = [statistics.median(segments.get(i, [0.0])) for i in range(max(segments) + 1)]
    spread = lambda rows, mid: [v - mid[i] for i, vs in rows.items() for v in vs]  # noqa: E731
    return {
        "offsets": off,
        "segment_fractions": seg,
        "residual_scale": max(1e-3, statistics.pstdev(spread(offsets, off)) or 1e-3),
        "duration_scale": max(1e-3, statistics.pstdev(spread(segments, seg)) or 1e-3),
    }


def apply(predicted: list[dict], model: dict[str, Any], weight: float = 1.0) -> list[dict]:
    """Correct boundaries the way the pipeline does, for scoring a candidate fit.

    Mirrors the boundary solver, shift clipping, ordered-label dropping, and
    full-coverage stitching. The diagnostic rows lack source frame timestamps,
    so this cannot reproduce production's final frame snapping.
    """
    offsets = model.get("offsets") or []
    segments = model.get("segment_fractions") or []
    # Nothing was placed, so there is nothing to correct. Returning early also
    # keeps episode_duration off an empty list; scoring still counts the
    # episode, because evaluate_alignment charges every unplaced label.
    if not predicted:
        return []
    duration = episode_duration(predicted)
    if segments and len(segments) == len(predicted) and len(predicted) > 1 and duration >= 0.1:
        spans = _solve(predicted, model, weight)
    else:
        spans = [dict(span) for span in predicted]
        for index in range(1, len(spans)):
            boundary = index - 1
            if boundary >= len(offsets):
                continue
            start = float(spans[index]["start"]) - offsets[boundary] * duration
            start = max(float(predicted[0]["start"]), min(start, float(predicted[-1]["end"])))
            spans[index]["start"] = start
            spans[index]["end"] = max(start, float(spans[index]["end"]))
            spans[index - 1]["end"] = max(float(spans[index - 1]["start"]), start)

    # Match _align_spans_in_order: a collided boundary loses its label instead
    # of receiving credit as a zero-duration placed segment during CV.
    kept = []
    last_start = None
    lo, hi = float(predicted[0]["start"]), float(predicted[-1]["end"])
    for span in spans:
        start, end = float(span["start"]), float(span["end"])
        if last_start is not None and start <= last_start:
            continue
        last_start = start
        start, end = max(lo, min(start, hi)), max(lo, min(end, hi))
        if end < start:
            start, end = end, start
        kept.append({**span, "start": start, "end": end})
    if kept:
        kept.sort(key=lambda span: span["start"])
        kept[0]["start"] = lo
        for index in range(1, len(kept)):
            kept[index - 1]["end"] = kept[index]["start"]
        kept[-1]["end"] = hi
    return kept


def _solve(predicted: list[dict], model: dict[str, Any], weight: float) -> list[dict]:
    """Reference implementation of the pipeline's DP boundary solver."""
    t0 = float(predicted[0]["start"])
    t1 = float(predicted[-1]["end"])
    duration = t1 - t0
    offsets = model["offsets"]
    want = [f * duration for f in model["segment_fractions"]]
    nb = len(predicted) - 1
    grid = np.arange(t0, t1 + 1e-9, 0.1)
    fit_scale = max(1e-6, model["residual_scale"] * duration)
    seg_scale = max(1e-6, model["duration_scale"] * duration)
    m = np.array(
        [min(max(float(predicted[i + 1]["start"]) - offsets[i] * duration, t0), t1) for i in range(nb)]
    )
    fit = np.abs(grid[None, :] - m[:, None]) / fit_scale
    cost = fit[0] + weight * np.abs((grid - t0) - want[0]) / seg_scale
    back = np.zeros((nb, grid.size), dtype=np.int32)
    for i in range(1, nb):
        step = np.empty(grid.size)
        choice = np.empty(grid.size, dtype=np.int32)
        for j in range(grid.size):
            cand = cost[: j + 1] + weight * np.abs((grid[j] - grid[: j + 1]) - want[i]) / seg_scale
            k = int(np.argmin(cand))
            step[j] = cand[k]
            choice[j] = k
        cost = step + fit[i]
        back[i] = choice
    idx = int(np.argmin(cost + weight * np.abs((t1 - grid) - want[nb]) / seg_scale))
    starts = [0.0] * nb
    for i in range(nb - 1, -1, -1):
        starts[i] = float(grid[idx])
        idx = int(back[i, idx])
    spans = [dict(s) for s in predicted]
    for i, start in enumerate(starts):
        spans[i + 1]["start"] = start
    for i in range(1, len(spans)):
        spans[i - 1]["end"] = spans[i]["start"]
    return spans


def score(results, truth, episodes, model, weight: float = 1.0) -> tuple[float, float, float]:
    tiou, mae, hit = [], [], []
    for episode in episodes:
        gt = truth[episode]
        labels = [str(span["label"]) for span in gt]
        corrected = [
            {**span, "text": _span_label(span)}
            for span in apply(results[episode], model, weight)
        ]
        metrics = evaluate_alignment(labels, gt, corrected)
        tiou.append(metrics.macro_temporal_iou)
        if metrics.boundary_mae is not None:
            mae.append(metrics.boundary_mae)
        hit.append(metrics.boundary_hit_rates.get(3.0, 0.0))
    return statistics.fmean(tiou), statistics.fmean(mae or [0.0]), statistics.fmean(hit)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, help="dataset root holding meta/lerobot_annotations.json")
    ap.add_argument(
        "results", type=Path,
        help="eval_align_batch.py --out JSON, or its append-only .jsonl sidecar",
    )
    ap.add_argument('--config-fingerprint', help='select one recorded evaluation configuration before deduplication')
    ap.add_argument('--allow-legacy-provenance', action='store_true',
                    help='explicitly accept historical rows without verifiable configuration identity')
    ap.add_argument("--out", type=Path, default=Path("align_calibration.json"))
    ap.add_argument("--format", default="video", help="which format's rows to fit on")
    ap.add_argument(
        "--episodes", type=int, nargs="+",
        help="fixed seed episode allowlist; all other result rows are ignored",
    )
    ap.add_argument(
        "--label-scope", choices=("exact", "segment_count"), default="exact",
        help="exact ordered labels, or explicit positional pooling within one task and count",
    )
    ap.add_argument("--min-fit-episodes", type=int, default=1,
                    help="minimum usable, complete seeds after filtering (not bypassed by --force)")
    ap.add_argument("--max-fit-episodes", type=int, default=10,
                    help="maximum allocated seeds; cannot exceed the ten-trajectory task budget")
    ap.add_argument("--task-id", help="task identifier recorded for provenance")
    ap.add_argument("--arm", help="prediction arm recorded for provenance")
    ap.add_argument(
        "--duration-weight",
        type=float,
        default=None,
        help="trade-off between the model and the duration prior; "
        "default picks one by leave-one-out on the fitting episodes (1.0 for fewer than 3)",
    )
    ap.add_argument("--no-duration-prior", action="store_true", help="fit offsets only, no solver")
    ap.add_argument(
        "--min-gain",
        type=float,
        default=3.0,
        help="refuse to write a calibration whose held-out gain is below this many "
        "macro-tIoU points; a model that is not biased on this task gains nothing "
        "from correction and loses a little to the duration prior",
    )
    ap.add_argument(
        "--min-first-boundary-mae",
        type=float,
        default=1.5,
        help="refuse when the uncalibrated model is already within this many seconds of the "
        "annotator on the first boundary; there is nothing to correct",
    )
    ap.add_argument("--force", action="store_true", help="write the calibration even if it is not recommended")
    args = ap.parse_args()

    if not 1 <= args.min_fit_episodes <= args.max_fit_episodes <= 10:
        ap.error("require 1 <= --min-fit-episodes <= --max-fit-episodes <= 10")
    if args.episodes is not None and len(set(args.episodes)) != len(args.episodes):
        ap.error("--episodes must contain unique trajectory IDs")
    if args.duration_weight is not None and (
        not math.isfinite(args.duration_weight) or args.duration_weight < 0
    ):
        ap.error("--duration-weight must be finite and nonnegative")
    if not math.isfinite(args.min_gain) or not math.isfinite(args.min_first_boundary_mae):
        ap.error("calibration gate thresholds must be finite")

    truth = human_spans(args.root)
    try:
        all_rows, collisions, source_format = load_result_rows(
            args.results, config_fingerprint=args.config_fingerprint, frame_format=args.format)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"cannot read results: {exc}", file=sys.stderr)
        return 1
    for note in collisions:
        print(describe_collision(note))
    rows = [r for r in all_rows if r.get("format") == args.format]
    if any(row.get('config_fingerprint') is None for row in rows):
        if not args.allow_legacy_provenance:
            print('Results lack configuration provenance. Re-evaluate, or explicitly acknowledge '
                  'unverifiable historical inputs with --allow-legacy-provenance.', file=sys.stderr)
            return 1
        print('WARNING: accepting legacy results with unverified configuration provenance', file=sys.stderr)
    ground_truth_sha256 = digest(args.root / 'meta/lerobot_annotations.json')
    for row in rows:
        if row.get('config_fingerprint') is not None and row['provenance'].get('ground_truth_sha256') != ground_truth_sha256:
            print('Ground truth differs from the evaluation input; refusing calibration.', file=sys.stderr)
            return 1
    missing_episode = [i for i, r in enumerate(rows) if "episode" not in r]
    if missing_episode:
        print(
            f"{args.results}: {len(missing_episode)} {args.format!r} row(s) have no 'episode' key",
            file=sys.stderr,
        )
        return 1
    if args.episodes is not None:
        rows = [r for r in rows if r["episode"] in args.episodes]
    seed_episodes = sorted(args.episodes if args.episodes is not None else {r["episode"] for r in rows})
    if len(seed_episodes) > args.max_fit_episodes:
        print(
            f"refusing {len(seed_episodes)} allocated seeds: maximum is {args.max_fit_episodes}; "
            "pass --episodes with the fixed task seed allocation", file=sys.stderr,
        )
        return 1
    if not seed_episodes:
        print(f"no usable {args.format!r} rows in {args.results}", file=sys.stderr)
        return 1

    # Check the requested GT cohort before discarding failed predictions: a
    # mixed count is a caller/schema error, not grounds to pick another mode.
    if args.label_scope == "segment_count":
        counts = {len(truth[e]) for e in seed_episodes if e in truth}
        if len(counts) != 1 or min(counts) < 2:
            print("segment_count scope requires one ground-truth segment count >= 2; "
                  f"selected counts: {sorted(counts)}", file=sys.stderr)
            return 1

    grouped_rows: dict[int, list[dict]] = {}
    for row in rows:
        grouped_rows.setdefault(row["episode"], []).append(row)
    results, exclusions = {}, []
    for episode in seed_episodes:
        candidates = grouped_rows.get(episode, [])
        if episode not in truth:
            reason = "missing ground-truth spans"
        elif not candidates:
            reason = f"missing {args.format} prediction row"
        elif len(candidates) != 1:
            reason = f"duplicate {args.format} prediction rows"
        elif "error" in candidates[0]:
            reason = f"prediction error: {candidates[0]['error']}"
        else:
            reason = unusable_episode_reason(candidates[0].get("spans"), truth[episode])
        if reason is not None:
            exclusions.append({"episode": episode, "reason": reason})
        else:
            results[episode] = candidates[0]["spans"]
    episodes = sorted(results)
    labels_by_episode = {e: tuple(s["label"] for s in truth[e]) for e in episodes}
    labels = []
    if args.label_scope == "exact" and episodes:
        label_counts = Counter(labels_by_episode.values())
        # Break frequency ties lexicographically, independent of hash seed.
        biggest = min(label_counts, key=lambda labels: (-label_counts[labels], labels))
        labels = list(biggest)
        for episode in episodes:
            if labels_by_episode[episode] != biggest:
                exclusions.append({"episode": episode, "reason": "outside modal exact label schema"})
        episodes = [e for e in episodes if labels_by_episode[e] == biggest]
    for exclusion in exclusions:
        print(f"excluded seed {exclusion['episode']}: {exclusion['reason']}")
    if len(episodes) < args.min_fit_episodes:
        print(f"only {len(episodes)} usable fitting episode(s); require {args.min_fit_episodes} "
              f"from the fixed {len(seed_episodes)}-trajectory allocation", file=sys.stderr)
        return 1

    def fit_candidate(cohort: list[int]) -> dict[str, Any]:
        candidate = fit(results, truth, cohort)
        if args.no_duration_prior:
            candidate.pop("segment_fractions", None)
        return candidate

    model = fit_candidate(episodes)
    if not model:
        print("no internal boundaries found to fit", file=sys.stderr)
        return 1

    # Choose the weight by leave-one-out on the FITTING episodes. Picking it on
    # the episodes you later report would be selection on the test set.
    weight = args.duration_weight
    if weight is None and model.get("segment_fractions") and len(episodes) > 2:
        candidates = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
        best, weight = None, 1.0
        for candidate in candidates:
            scores = []
            for episode in episodes:
                rest = [e for e in episodes if e != episode]
                scores.append(score(results, truth, [episode], fit_candidate(rest), candidate)[0])
            mean = statistics.fmean(scores)
            if best is None or mean > best:
                best, weight = mean, candidate
        print(f"duration weight chosen by leave-one-out: {weight} (held-out tIoU {best:.2%})")
    if weight is None:
        # Match apply()/score()'s documented default when there are too few
        # episodes to select a prior weight. Preserve explicit 0.0 weights.
        weight = 1.0
        if model.get("segment_fractions"):
            print("note: fewer than 3 fitting episodes; using default duration weight 1.0")

    print(f"\nfit on {len(episodes)} episode(s): {episodes}")
    print("\n  boundary  offset (fraction of episode, subtracted)")
    for index, value in enumerate(model["offsets"]):
        print(f"  {index:>8}  {value:+7.4f}")
    if model.get("segment_fractions"):
        print("\n  expected subtask lengths (fraction of episode)")
        print("   " + "  ".join(f"{v:.3f}" for v in model["segment_fractions"]))

    t0, m0, h0 = score(results, truth, episodes, {"offsets": [0.0] * len(model["offsets"])})
    t1, m1, h1 = score(results, truth, episodes, model, weight)
    print(f"\n  in-sample : tIoU {t0:.2%} -> {t1:.2%}   MAE {m0:.2f}s -> {m1:.2f}s   B@3 {h0:.1%} -> {h1:.1%}")

    held, base = [], []
    if len(episodes) > 2:
        for episode in episodes:
            rest = [e for e in episodes if e != episode]
            held.append(score(results, truth, [episode], fit_candidate(rest), weight))
            base.append(score(results, truth, [episode], {"offsets": [0.0] * len(model["offsets"])}))
        print(
            f"  held-out  : tIoU {statistics.fmean(b[0] for b in base):.2%} -> "
            f"{statistics.fmean(h[0] for h in held):.2%}"
            f"   B@3 {statistics.fmean(b[2] for b in base):.1%} -> {statistics.fmean(h[2] for h in held):.1%}"
            f"   (improved in {sum(1 for h, b in zip(held, base, strict=True) if h[0] > b[0])}/{len(held)})"
        )

    gain = None
    if len(episodes) > 2:
        gain = (statistics.fmean(h[0] for h in held) - statistics.fmean(b[0] for b in base)) * 100

    # How far the UNCALIBRATED model sits from this annotator on the first
    # boundary. Measured over 16 datasets and 507 held-out episodes, this is the
    # best single predictor of what calibration is worth (r = +0.87 against the
    # realised gain, versus +0.80 for the leave-one-out estimate below).
    first_errors = [
        abs(signed_errors(results[e], truth[e]).get(0, 0.0))
        for e in episodes
        if signed_errors(results[e], truth[e])
    ]
    first_mae = statistics.fmean(first_errors) if first_errors else 0.0

    print(f"\n  uncalibrated first-boundary MAE: {first_mae:.2f}s")
    if gain is not None:
        print(f"  leave-one-out predicted gain   : {gain:+.2f} points")
        print("    (optimistic by about +10 points on average across 16 datasets — "
              "treat as an upper bound)")

    # Refuse only where the evidence is unambiguous. Below 1.5s of first-boundary
    # error, four datasets gained at most +1.83 and one lost 0.72 while shedding
    # ~9 points of B@3 — there is no bias to remove and the duration prior only
    # adds constraint. Between there and ~5s the relationship is real but noisy
    # (one dataset with 3.5s of bias gained only +0.3 because its raw output was
    # already the best of the set), so that range is reported, not blocked.
    problems = []
    if first_mae < args.min_first_boundary_mae:
        problems.append(
            f"the model is already within {first_mae:.2f}s of this annotator "
            f"(threshold {args.min_first_boundary_mae:.2f}s) — nothing to correct"
        )
    if gain is not None and gain < args.min_gain:
        problems.append(f"held-out gain is only {gain:+.2f} points (threshold {args.min_gain:+.2f})")
    if len(episodes) < 10:
        print(f"\n  NOTE: only {len(episodes)} episode(s) survived fitting validation; "
              "the leave-one-out estimate is less trustworthy than usual")
    if problems:
        print("\n  NOT RECOMMENDED for this dataset:")
        for item in problems:
            print(f"    - {item}")
        print("    Calibrating is likely to cost a little rather than gain.")
        if not args.force:
            print("\n  refusing to write; re-run with --force to override")
            return 2

    payload = {
        "mode": "fraction_of_duration",
        "label_scope": args.label_scope,
        "n_segments": len(truth[episodes[0]]),
        "heldout_gain_points": None if gain is None else round(gain, 3),
        "uncalibrated_first_boundary_mae": round(first_mae, 3),
        "fit_episode_count": len(episodes),
        "offsets": [round(v, 5) for v in model["offsets"]],
        "labels": labels,
        "fit_on_episodes": episodes,
        "seed_episodes": seed_episodes,
        "excluded_episodes": exclusions,
        "max_fit_episodes": args.max_fit_episodes,
        "source_results": str(args.results),
        "source_results_sha256": digest(args.results),
        "ground_truth_sha256": ground_truth_sha256,
        "config_fingerprint": rows[0].get('config_fingerprint') if rows else None,
        "provenance_verified": bool(rows) and all(r.get('config_fingerprint') for r in rows),
        "source_results_format": source_format,
        "resolved_collisions": collisions,
    }
    if args.task_id is not None:
        payload["task_id"] = args.task_id
    if args.arm is not None:
        payload["arm"] = args.arm
    if model.get("segment_fractions"):
        payload.update(
            segment_fractions=[round(v, 5) for v in model["segment_fractions"]],
            residual_scale=round(model["residual_scale"], 5),
            duration_scale=round(model["duration_scale"], 5),
            duration_weight=weight,
        )
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    print(f"  --plan.subtask_align_calibration_path={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
