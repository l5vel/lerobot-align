#!/usr/bin/env python
"""Boundary-calibration adapter, applicable to ANY arm's predictions.

``arms.yaml`` declares ``postprocess: dp_calibration`` on
``baseline_upstream_calibrated``. Nothing read that field, so E5 could not run.
This is what reads it.

Why the adapter is a post-process at all
----------------------------------------
The new tool can calibrate its own boundaries; upstream cannot. Comparing a
calibrated new tool against an uncalibrated baseline would measure the adapter
and report it as a tool difference (plan section 7.4). So the adapter has to be
applicable to spans that have already been written -- upstream's included --
rather than living inside one tool's alignment call.

Why this is a reimplementation and not an import
------------------------------------------------
``lerobot_align.modules.plan_subtasks_memory`` already contains this solver.
Importing it would make the measuring instrument a part of the thing being
measured: an edit to the tool would silently redefine what "calibrated" means
for the *baseline* arm too, and any bug in the tool's solver would be invisible
because both sides of the contrast would share it. The same reasoning already
governs ground-truth extraction (``prepare_dataset.py``) and span parsing, so
it governs this too. The cost function, the 0.1 s grid and the backtracking are
kept deliberately faithful to ``_solve_align_boundaries`` -- this is the same
method, not a similar one.

What is fitted, and what is deliberately NOT
--------------------------------------------
The tool's own fit (``diagnostics/fit_align_calibration.py``) estimates two
things: a signed per-boundary **offset**, and a **duration prior** over segment
lengths. The offset is fitted from *paired* data -- the arm's own predicted
boundary minus the human boundary on each seed episode.

This adapter fits only the second half, from the seed episodes' human spans
alone, and its offsets are identically zero. Two reasons, and both are binding:

* The harness never runs an arm on the seed split (``run_arm.py`` passes the
  *eval* episodes to ``--only_episodes``), so the paired data does not exist
  without spending extra inference on every arm.
* Fitting it would make the fitted adapter arm-specific. A
  calibrated-vs-calibrated contrast would then blend the quality of the tool
  with how well the adapter happened to fit that tool, which is the confound
  the calibrated baseline arm exists to remove.

The consequence must be read into any E5 result: this adapter can reshape a
prediction's timing toward the corpus's typical boundary profile, but it cannot
remove a constant lateness the way the tool's own calibration does. It is
therefore a **lower bound** on what the tool's calibration feature is worth, and
is not a measurement of that feature.

Why the prior is keyed on the segment count
-------------------------------------------
Both halves of the real method are *positional*: offset ``i`` and expected
length ``i`` belong to the i-th subtask. The tool guards that by refusing to
calibrate unless the label list matches the one it was fitted for. That guard
cannot be used here, because every arm invents its own labels and upstream's
never match the human list -- applying it would leave every episode
uncalibrated and E5 would compare two untouched arms while reporting that it
had calibrated them. The count is the same guard in the one coordinate that
survives an arm renaming everything: a prior fitted on 5-segment episodes is
used for 5-segment predictions. For a count the seed split never showed, the
prior is resampled from a pooled progress curve (fraction of segments completed
-> fraction of episode elapsed); ``--count-policy skip`` refuses instead, which
is honest but leaves the over-segmenting VLM arms almost entirely uncalibrated.

Positions, not lengths, are the fitted quantity: taking a median per segment
length gives lengths that do not sum to the episode, so the solver's tail term
would pull every boundary in a direction that is not in the data. Medians of
*positions* are monotone by construction and their differences sum to exactly 1.

Leakage
-------
Fitting on an episode that is later scored would be leakage, and it is the one
mistake here that no reviewer could detect from the output alone. The fit reads
the split file, uses only ``seed``, and asserts that no fitted episode appears
in ``eval``; ``--apply`` asserts the converse against the episodes it touches.

This file is not interchangeable with the tool's calibration format: it carries
no ``offsets`` key, so ``_load_align_calibration`` raises rather than
half-reading it if the two are ever confused on a command line.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from gate_fingerprint import PROVENANCE_KEYS  # noqa: E402

Span = dict[str, Any]

ADAPTER = "dp_calibration"
# Bump when the fitted schema or the solved output changes, so a calibration
# file written by an older version is refused rather than silently reinterpreted.
ADAPTER_VERSION = 1

GRID_SECONDS = 0.1
# 0.1 s over an hour-long episode is 36k grid points and an O(k*G^2) solve.
# Nothing in either corpus is remotely that long, so hitting this means the
# input is wrong; the episode is left uncalibrated and reported rather than
# quietly costing an hour of CPU.
MAX_GRID_POINTS = 20_000
# Spans from both tools tile the episode exactly (`_stitch_full_coverage` sets
# each end from the next start), as do the reference arms and the human spans.
TILING_TOLERANCE = 1e-6

# Fraction of the episode the solver is allowed to move a boundary before it
# stops trusting the arm. Not estimable without paired predictions (see above),
# so it is a declared constant, held identical across arms, and recorded in
# every calibration file.
DEFAULT_RESIDUAL_SCALE = 0.05
DEFAULT_DURATION_WEIGHT = 1.0


# --------------------------------------------------------------------------
# reading spans
# --------------------------------------------------------------------------

def span_bounds(spans: Sequence[Span]) -> tuple[float, float]:
    return float(spans[0]["start"]), float(spans[-1]["end"])


def relative_boundaries(spans: Sequence[Span]) -> list[float]:
    """Internal boundary positions as fractions of the episode, or raise.

    Raises rather than repairing. The human spans carry known anomalies -- a
    back-jump of 4.185 s, two episodes with gaps -- and section 3.2 fixes the
    policy for them as "reported, not repaired": an episode whose geometry is
    not a clean tiling is dropped from the fit and named in the calibration
    file, because averaging a malformed episode into the prior would move every
    boundary of every calibrated episode by an amount nobody could later trace.
    """
    if len(spans) < 1:
        raise ValueError("no spans")
    times: list[float] = []
    for index, span in enumerate(spans):
        start, end = float(span["start"]), float(span["end"])
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError(f"span {index} has non-finite bounds")
        if end < start:
            raise ValueError(f"span {index} ends before it starts")
        if times and abs(start - times[-1]) > TILING_TOLERANCE:
            raise ValueError(
                f"span {index} starts at {start:.6f} but the previous span ends at "
                f"{times[-1]:.6f}; spans must tile the episode"
            )
        times.append(end)
    t0, t1 = span_bounds(spans)
    duration = t1 - t0
    if duration <= 0:
        raise ValueError("episode has non-positive duration")
    return [(float(s["start"]) - t0) / duration for s in spans[1:]]


# --------------------------------------------------------------------------
# the fitted object
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Calibration:
    """Corpus-level prior over boundary positions and segment lengths.

    ``positions_by_count`` holds the per-index median boundary position for
    each segment count the seed split showed often enough to be worth fitting.
    ``curve_*`` is the pooled fallback, a monotone map from fraction-of-segments
    to fraction-of-episode used for counts the seed split never showed.
    """

    dataset: str
    fit_episodes: tuple[int, ...]
    positions_by_count: dict[int, tuple[float, ...]]
    curve_x: tuple[float, ...]
    curve_y: tuple[float, ...]
    residual_scale: float
    duration_scale: float
    duration_weight: float

    def segment_fractions(self, count: int, *, policy: str) -> tuple[float, ...] | None:
        """Expected segment lengths, as fractions of the episode, or ``None``.

        ``None`` means "this adapter has no prior for a prediction of this
        shape", and the caller leaves the episode alone rather than inventing
        one.
        """
        if count < 2:
            return None
        positions = self.positions_by_count.get(count)
        if positions is None:
            if policy != "interpolate" or len(self.curve_x) < 2:
                return None
            positions = tuple(
                float(np.interp(index / count, self.curve_x, self.curve_y))
                for index in range(1, count)
            )
        edges = [0.0, *positions, 1.0]
        # Monotonicity is guaranteed by how the curve and the medians are built;
        # re-established here because a hand-edited calibration file must not be
        # able to produce negative segment lengths downstream.
        for index in range(1, len(edges)):
            edges[index] = min(1.0, max(edges[index], edges[index - 1]))
        return tuple(edges[i + 1] - edges[i] for i in range(count))

    def to_payload(self, extra: dict[str, Any]) -> dict[str, Any]:
        return {
            "adapter": ADAPTER,
            "adapter_version": ADAPTER_VERSION,
            "mode": "fraction_of_duration",
            "dataset": self.dataset,
            "fit_episodes": list(self.fit_episodes),
            "boundary_positions_by_count": {
                str(count): [round(v, 6) for v in values]
                for count, values in sorted(self.positions_by_count.items())
            },
            "progress_curve": {
                "segments_fraction": [round(v, 6) for v in self.curve_x],
                "time_fraction": [round(v, 6) for v in self.curve_y],
            },
            "residual_scale": self.residual_scale,
            "duration_scale": self.duration_scale,
            "duration_weight": self.duration_weight,
            **extra,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, path: Path) -> Calibration:
        """Parse a calibration file, refusing anything it cannot fully validate.

        Every failure here raises. A calibration that silently degraded to a
        weaker prior would still produce plausible spans and plausible scores,
        which is the one outcome this evaluation cannot tolerate.
        """
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a JSON object")
        if payload.get("adapter") != ADAPTER:
            raise ValueError(f"{path}: not a {ADAPTER} calibration ({payload.get('adapter')!r})")
        if payload.get("adapter_version") != ADAPTER_VERSION:
            raise ValueError(
                f"{path}: adapter version {payload.get('adapter_version')!r}, "
                f"this script writes and reads {ADAPTER_VERSION}"
            )

        def _number(key: str, *, positive: bool) -> float:
            value = payload.get(key)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{path}: '{key}' must be a number")
            value = float(value)
            if not math.isfinite(value) or (positive and value <= 0) or value < 0:
                raise ValueError(f"{path}: '{key}' must be a finite non-negative number")
            return value

        raw_counts = payload.get("boundary_positions_by_count")
        if not isinstance(raw_counts, dict):
            raise ValueError(f"{path}: 'boundary_positions_by_count' must be an object")
        positions: dict[int, tuple[float, ...]] = {}
        for key, values in raw_counts.items():
            count = int(key)
            if not isinstance(values, list) or len(values) != count - 1:
                raise ValueError(
                    f"{path}: count {count} needs {count - 1} boundary position(s), "
                    f"got {values if not isinstance(values, list) else len(values)}"
                )
            parsed = tuple(float(v) for v in values)
            if any(not math.isfinite(v) or not 0.0 <= v <= 1.0 for v in parsed):
                raise ValueError(f"{path}: count {count} has a position outside [0, 1]")
            if any(b < a for a, b in zip(parsed, parsed[1:], strict=False)):
                raise ValueError(f"{path}: count {count} has out-of-order positions")
            positions[count] = parsed

        curve = payload.get("progress_curve") or {}
        curve_x = tuple(float(v) for v in (curve.get("segments_fraction") or ()))
        curve_y = tuple(float(v) for v in (curve.get("time_fraction") or ()))
        if len(curve_x) != len(curve_y):
            raise ValueError(f"{path}: progress curve axes have different lengths")
        if any(b < a for a, b in zip(curve_x, curve_x[1:], strict=False)):
            raise ValueError(f"{path}: progress curve x values are not sorted")
        if any(b < a for a, b in zip(curve_y, curve_y[1:], strict=False)):
            raise ValueError(f"{path}: progress curve is not monotone")

        episodes = payload.get("fit_episodes")
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"{path}: 'fit_episodes' must be a non-empty list")
        dataset = payload.get("dataset")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError(f"{path}: 'dataset' must be a non-empty string")

        return cls(
            dataset=dataset,
            fit_episodes=tuple(int(e) for e in episodes),
            positions_by_count=positions,
            curve_x=curve_x,
            curve_y=curve_y,
            residual_scale=_number("residual_scale", positive=True),
            duration_scale=_number("duration_scale", positive=True),
            duration_weight=_number("duration_weight", positive=False),
        )


# --------------------------------------------------------------------------
# fit
# --------------------------------------------------------------------------

def fit_calibration(
    *,
    dataset: str,
    truth: dict[int, list[Span]],
    seed_episodes: Sequence[int],
    residual_scale: float,
    duration_weight: float,
    min_count_episodes: int,
) -> tuple[Calibration, dict[int, str]]:
    """Fit the prior from human spans on the seed episodes."""
    rows_by_count: dict[int, list[list[float]]] = {}
    knots: list[tuple[float, float]] = []
    used: list[int] = []
    rejected: dict[int, str] = {}

    for episode in sorted(dict.fromkeys(seed_episodes)):
        spans = truth.get(episode)
        if not spans:
            rejected[episode] = "no ground truth"
            continue
        try:
            positions = relative_boundaries(spans)
        except ValueError as exc:
            rejected[episode] = str(exc)
            continue
        count = len(spans)
        used.append(episode)
        if count < 2:
            # A single-span episode has no internal boundary to contribute, but
            # it is not an anomaly and is not reported as one.
            continue
        rows_by_count.setdefault(count, []).append(positions)
        knots.extend(((index + 1) / count, value) for index, value in enumerate(positions))

    if not knots:
        raise SystemExit(
            "no seed episode contributed an internal boundary; nothing to fit"
        )

    positions_by_count: dict[int, tuple[float, ...]] = {}
    for count, rows in sorted(rows_by_count.items()):
        if len(rows) < min_count_episodes:
            continue
        medians = [statistics.median(column) for column in zip(*rows, strict=True)]
        # Medians of pointwise-ordered rows are themselves ordered; clamped
        # anyway so a future change to the estimator cannot break the invariant.
        running = 0.0
        ordered: list[float] = []
        for value in medians:
            running = min(1.0, max(value, running))
            ordered.append(running)
        positions_by_count[count] = tuple(ordered)

    curve_x, curve_y = _progress_curve(knots)

    calibration = Calibration(
        dataset=dataset,
        fit_episodes=tuple(used),
        positions_by_count=positions_by_count,
        curve_x=curve_x,
        curve_y=curve_y,
        residual_scale=residual_scale,
        duration_scale=1.0,  # placeholder; measured from the residuals below
        duration_weight=duration_weight,
    )

    # How far a real episode's segment lengths sit from the prior. This is the
    # scale the solver divides the duration term by, so it is what stops the
    # prior from overruling the arm on a corpus where segment lengths vary a
    # lot -- and from being ignored on one where they barely vary at all.
    residuals: list[float] = []
    for episode in used:
        spans = truth[episode]
        if len(spans) < 2:
            continue
        expected = calibration.segment_fractions(len(spans), policy="interpolate")
        if expected is None:
            continue
        edges = [0.0, *relative_boundaries(spans), 1.0]
        observed = [edges[i + 1] - edges[i] for i in range(len(spans))]
        residuals.extend(o - e for o, e in zip(observed, expected, strict=True))
    duration_scale = max(1e-3, statistics.pstdev(residuals) if len(residuals) > 1 else 0.0)

    return replace(calibration, duration_scale=duration_scale), rejected


def _progress_curve(
    knots: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Pooled fraction-of-segments -> fraction-of-episode map.

    Every seed episode contributes one point per internal boundary, whatever its
    segment count, so counts the seed split never showed still get a prior drawn
    from real observations rather than from an equal split. Duplicated x values
    are reduced by median for the same outlier resistance as the per-count fit.
    """
    grouped: dict[float, list[float]] = {}
    for x, y in knots:
        grouped.setdefault(round(float(x), 9), []).append(float(y))
    xs = [0.0]
    ys = [0.0]
    for x in sorted(grouped):
        if x <= 0.0 or x >= 1.0:
            continue
        xs.append(x)
        ys.append(statistics.median(grouped[x]))
    xs.append(1.0)
    ys.append(1.0)
    running = 0.0
    monotone: list[float] = []
    for value in ys:
        running = min(1.0, max(value, running))
        monotone.append(running)
    return tuple(xs), tuple(monotone)


# --------------------------------------------------------------------------
# apply
# --------------------------------------------------------------------------

def solve_boundaries(
    *,
    model: Sequence[float],
    fractions: Sequence[float],
    t0: float,
    t1: float,
    residual_scale: float,
    duration_scale: float,
    duration_weight: float,
    grid_seconds: float,
) -> list[float]:
    """Place every internal boundary at once by dynamic programming.

    Faithful to ``_solve_align_boundaries``: two costs on a fixed grid --
    staying near the arm's answer, and matching the expected segment lengths --
    with the ordering constraint built into the recursion rather than repaired
    afterwards. Shifting each boundary independently, the obvious alternative,
    can push one boundary past its neighbour and produce spans the metrics
    reject outright.
    """
    duration = t1 - t0
    n_boundaries = len(model)
    grid = np.arange(t0, t1 + 1e-9, grid_seconds)
    fit_scale = max(1e-6, residual_scale * duration)
    seg_scale = max(1e-6, duration_scale * duration)
    want = np.asarray(fractions, dtype=float) * duration
    anchors = np.clip(np.asarray(model, dtype=float), t0, t1)

    fit = np.abs(grid[None, :] - anchors[:, None]) / fit_scale
    cost = fit[0] + duration_weight * np.abs((grid - t0) - want[0]) / seg_scale
    back = np.zeros((n_boundaries, grid.size), dtype=np.int32)
    for index in range(1, n_boundaries):
        step = np.empty(grid.size)
        choice = np.empty(grid.size, dtype=np.int32)
        for j in range(grid.size):
            # Boundaries may not go backwards, so only grid points up to j are
            # legal predecessors; the prior pulls the gap toward want[index].
            candidates = (
                cost[: j + 1]
                + duration_weight * np.abs((grid[j] - grid[: j + 1]) - want[index]) / seg_scale
            )
            k = int(np.argmin(candidates))
            step[j] = candidates[k]
            choice[j] = k
        cost = step + fit[index]
        back[index] = choice

    tail = duration_weight * np.abs((t1 - grid) - want[n_boundaries]) / seg_scale
    position = int(np.argmin(cost + tail))
    starts = [0.0] * n_boundaries
    for index in range(n_boundaries - 1, -1, -1):
        starts[index] = float(grid[position])
        position = int(back[index, position])
    return starts


def calibrate_spans(
    spans: Sequence[Span],
    calibration: Calibration,
    *,
    policy: str,
    grid_seconds: float,
) -> tuple[list[Span], str | None]:
    """Adjust boundary times only. Returns ``(spans, reason_left_unchanged)``.

    An episode this adapter has no prior for is returned untouched with the
    reason recorded, not dropped: dropping it would change which episodes each
    arm is scored on, and section 8.2 fixes that failures are data.
    """
    original = [dict(span) for span in spans]
    if len(original) < 2:
        return original, "fewer than two spans: no internal boundary"
    try:
        relative_boundaries(original)
    except ValueError as exc:
        return original, f"prediction geometry rejected: {exc}"

    fractions = calibration.segment_fractions(len(original), policy=policy)
    if fractions is None:
        return original, f"no prior for a {len(original)}-segment prediction"

    t0, t1 = span_bounds(original)
    if (t1 - t0) / grid_seconds + 1 > MAX_GRID_POINTS:
        return original, f"episode of {t1 - t0:.1f}s exceeds the solver grid budget"

    starts = solve_boundaries(
        model=[float(span["start"]) for span in original[1:]],
        fractions=fractions,
        t0=t0,
        t1=t1,
        residual_scale=calibration.residual_scale,
        duration_scale=calibration.duration_scale,
        duration_weight=calibration.duration_weight,
        grid_seconds=grid_seconds,
    )

    calibrated = [dict(span) for span in original]
    for index, start in enumerate(starts):
        calibrated[index + 1]["start"] = start
    calibrated[0]["start"] = t0
    for index in range(1, len(calibrated)):
        calibrated[index - 1]["end"] = calibrated[index]["start"]
    calibrated[-1]["end"] = t1
    assert_timing_only(original, calibrated)
    return calibrated, None


def assert_timing_only(before: Sequence[Span], after: Sequence[Span]) -> None:
    """Refuse to emit anything but a retiming of the input.

    This is a timing adapter, not a relabeller: if it could change a label, a
    count or an order, then an E5 result would no longer isolate boundary
    placement and the calibrated arms would not be comparable to the
    uncalibrated ones at all. Written as explicit raises rather than ``assert``
    statements so that ``python -O`` cannot switch the guarantee off.
    """
    if len(before) != len(after):
        raise AssertionError(f"segment count changed: {len(before)} -> {len(after)}")
    for index, (old, new) in enumerate(zip(before, after, strict=True)):
        if set(old) != set(new):
            raise AssertionError(f"span {index}: field set changed")
        for key in old:
            if key in {"start", "end"}:
                continue
            if old[key] != new[key]:
                raise AssertionError(f"span {index}: field {key!r} changed")
    starts = [float(span["start"]) for span in after]
    ends = [float(span["end"]) for span in after]
    if any(not math.isfinite(v) for v in (*starts, *ends)):
        raise AssertionError("calibrated spans contain a non-finite time")
    if any(b < a for a, b in zip(starts, starts[1:], strict=False)):
        raise AssertionError("calibrated boundaries are out of order")
    for index in range(len(after)):
        if ends[index] < starts[index]:
            raise AssertionError(f"span {index} ends before it starts")
        if index + 1 < len(after) and abs(ends[index] - starts[index + 1]) > TILING_TOLERANCE:
            raise AssertionError(f"span {index} no longer meets span {index + 1}")
    old_t0, old_t1 = span_bounds(before)
    if abs(starts[0] - old_t0) > TILING_TOLERANCE or abs(ends[-1] - old_t1) > TILING_TOLERANCE:
        raise AssertionError("calibration moved the episode endpoints")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def run_fit(args: argparse.Namespace) -> int:
    truth_payload = json.loads(args.gt.read_text(encoding="utf-8"))
    dataset = truth_payload["dataset"]
    truth = {int(k): v for k, v in truth_payload["episodes"].items()}

    split = json.loads(args.split.read_text(encoding="utf-8"))
    if split.get("dataset") != dataset:
        raise SystemExit(f"split is for {split.get('dataset')!r}, ground truth for {dataset!r}")
    seed = [int(e) for e in split["seed"]]
    evaluation = [int(e) for e in split["eval"]]
    leaked = sorted(set(seed) & set(evaluation))
    if leaked:
        raise SystemExit(f"split is not disjoint; seed and eval share {leaked}")

    calibration, rejected = fit_calibration(
        dataset=dataset,
        truth=truth,
        seed_episodes=seed,
        residual_scale=args.residual_scale,
        duration_weight=args.duration_weight,
        min_count_episodes=args.min_count_episodes,
    )

    # The assertion the whole supervised block rests on. Checked against the
    # episodes actually fitted rather than against the requested list, so a
    # future change to which episodes survive filtering cannot slip past it.
    fitted_eval = sorted(set(calibration.fit_episodes) & set(evaluation))
    if fitted_eval:
        raise AssertionError(f"calibration was fitted on evaluation episodes {fitted_eval}")

    payload = calibration.to_payload(
        {
            "gt_source": str(args.gt),
            "gt_sha": _sha16(args.gt),
            "split_source": str(args.split),
            "n_fit_episodes": len(calibration.fit_episodes),
            "rejected_episodes": {str(k): v for k, v in sorted(rejected.items())},
            "eval_episodes_excluded": evaluation,
            "min_count_episodes": args.min_count_episodes,
        }
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    print(
        f"[calibrate] fit {dataset} on {len(calibration.fit_episodes)} seed episode(s); "
        f"counts {sorted(calibration.positions_by_count)}; "
        f"duration_scale={calibration.duration_scale:.4f} -> {args.out}"
    )
    for count, positions in sorted(calibration.positions_by_count.items()):
        print(f"[calibrate]   {count} segments: " + " ".join(f"{v:.3f}" for v in positions))
    for episode, reason in sorted(rejected.items()):
        print(f"[calibrate]   episode {episode} not fitted: {reason}", file=sys.stderr)
    return 0


def _check_arm_declaration(name: str, arms_config: Path) -> None:
    """Cross-check the target arm against ``arms.yaml``.

    Applying this adapter to an arm that already passes
    ``subtask_align_calibration_path`` to the tool would calibrate twice and
    report the result as one adapter, so that is refused. An arm that does not
    declare ``postprocess: dp_calibration`` is warned about rather than refused,
    because sensitivity checks against the reference arms are legitimate -- but
    a silent apply to an undeclared arm is exactly how an unrecorded difference
    enters a table.
    """
    if not arms_config.exists():
        print(
            f"[calibrate] WARNING: {arms_config} not found; arm not cross-checked",
            file=sys.stderr,
        )
        return
    import yaml

    spec = yaml.safe_load(arms_config.read_text(encoding="utf-8"))
    arms = {a["name"]: a for a in spec.get("arms", [])}
    arm = arms.get(name)
    if arm is None:
        print(
            f"[calibrate] WARNING: arm {name!r} is not declared in {arms_config}",
            file=sys.stderr,
        )
        return
    for key, value in (arm.get("flags") or {}).items():
        if "calibration_path" in str(key):
            raise SystemExit(
                f"{name} already calibrates inside the tool ({key}={value}); applying this "
                "adapter as well would calibrate twice"
            )
    if arm.get("postprocess") != ADAPTER:
        print(
            f"[calibrate] WARNING: arm {name!r} does not declare postprocess: {ADAPTER}",
            file=sys.stderr,
        )


def run_apply(args: argparse.Namespace) -> int:
    if args.out.resolve() == args.predictions.resolve():
        raise SystemExit("refusing to overwrite the source predictions in place")

    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    calibration_payload = json.loads(args.calibration.read_text(encoding="utf-8"))
    calibration = Calibration.from_payload(calibration_payload, path=args.calibration)

    dataset = payload["dataset"]
    if dataset != calibration.dataset:
        raise SystemExit(
            f"calibration was fitted on {calibration.dataset!r} but these predictions are "
            f"for {dataset!r}; calibration is a per-corpus adapter (plan section 10.4)"
        )
    source_arm = payload["arm"]
    arm = args.arm_name or source_arm
    _check_arm_declaration(arm, args.arms_config)

    episodes = {int(k): v for k, v in (payload.get("episodes") or {}).items()}
    leaked = sorted(set(episodes) & set(calibration.fit_episodes))
    if leaked:
        raise AssertionError(
            f"episodes {leaked} were used to fit this calibration and are being scored "
            "under it; that is leakage"
        )

    calibrated: dict[str, list[Span]] = {}
    unchanged: dict[str, str] = {}
    shifts: list[float] = []
    for episode in sorted(episodes):
        spans, reason = calibrate_spans(
            episodes[episode], calibration, policy=args.count_policy, grid_seconds=args.grid_seconds
        )
        calibrated[str(episode)] = spans
        if reason is not None:
            unchanged[str(episode)] = reason
            continue
        shifts.extend(
            abs(float(new["start"]) - float(old["start"]))
            for old, new in zip(episodes[episode][1:], spans[1:], strict=True)
        )

    missing = [key for key in PROVENANCE_KEYS if key not in payload]
    record = {
        **payload,
        "arm": arm,
        "episodes": calibrated,
        "calibration": {
            "applied": True,
            "adapter": ADAPTER,
            "adapter_version": ADAPTER_VERSION,
            "calibration_path": str(args.calibration),
            "calibration_sha": _sha16(args.calibration),
            "fit_dataset": calibration.dataset,
            "fit_episodes": list(calibration.fit_episodes),
            "count_policy": args.count_policy,
            "grid_seconds": args.grid_seconds,
            "residual_scale": calibration.residual_scale,
            "duration_scale": calibration.duration_scale,
            "duration_weight": calibration.duration_weight,
            "source_predictions": str(args.predictions),
            "source_arm": source_arm,
            "n_episodes": len(calibrated),
            "n_calibrated": len(calibrated) - len(unchanged),
            "n_unchanged": len(unchanged),
            "unchanged": unchanged,
            "mean_abs_boundary_shift_seconds": (
                round(statistics.fmean(shifts), 4) if shifts else None
            ),
            "max_abs_boundary_shift_seconds": round(max(shifts), 4) if shifts else None,
            # Provenance the schema carries is copied verbatim above; recorded
            # here so a row missing a contracted field is visible at the point
            # the adapter ran rather than only at the gate.
            "missing_provenance_keys": missing,
        },
        # The source fingerprint describes the run that produced the raw spans
        # and says nothing about the adapter. Kept, but not reused: a calibrated
        # file claiming the uncalibrated run's fingerprint would let a cache
        # serve one as the other.
        "source_config_fingerprint": payload.get("config_fingerprint"),
        "config_fingerprint": _fingerprint(
            {
                "source": payload.get("config_fingerprint"),
                "adapter": ADAPTER,
                "adapter_version": ADAPTER_VERSION,
                "calibration_sha": _sha16(args.calibration),
                "count_policy": args.count_policy,
                "grid_seconds": args.grid_seconds,
                "arm": arm,
            }
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1), encoding="utf-8")

    if missing:
        print(
            f"[calibrate] WARNING: source predictions carry no {missing}; the C1 gate treats "
            "rows without those fields as unattributable",
            file=sys.stderr,
        )
    print(
        f"[calibrate] {dataset} {source_arm} -> {arm}: calibrated "
        f"{len(calibrated) - len(unchanged)}/{len(calibrated)} episode(s), "
        f"mean |shift| {record['calibration']['mean_abs_boundary_shift_seconds']}s -> {args.out}"
    )
    for episode, reason in sorted(unchanged.items())[:5]:
        print(f"[calibrate]   episode {episode} left uncalibrated: {reason}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--fit", action="store_true", help="fit a calibration from the seed split")
    mode.add_argument("--apply", action="store_true", help="calibrate an arm's predictions")

    parser.add_argument("--gt", type=Path, help="[fit] ground truth JSON from prepare_dataset.py")
    parser.add_argument("--split", type=Path, help="[fit] split JSON from make_splits.py")
    parser.add_argument("--residual-scale", type=float, default=DEFAULT_RESIDUAL_SCALE,
                        help="[fit] how far, as a fraction of the episode, the solver may move a "
                             "boundary before it stops trusting the arm")
    parser.add_argument("--duration-weight", type=float, default=DEFAULT_DURATION_WEIGHT,
                        help="[fit] weight of the duration prior against the arm's own answer")
    parser.add_argument("--min-count-episodes", type=int, default=2,
                        help="[fit] seed episodes required before a segment count gets its own "
                             "prior rather than the pooled progress curve")

    parser.add_argument("--predictions", type=Path, help="[apply] predictions JSON")
    parser.add_argument("--calibration", type=Path, help="[apply] calibration JSON from --fit")
    parser.add_argument("--arm-name", default=None,
                        help="[apply] arm name for the output, e.g. baseline_upstream_calibrated")
    parser.add_argument("--arms-config", type=Path, default=HERE.parent / "configs" / "arms.yaml")
    parser.add_argument("--count-policy", choices=("interpolate", "skip"), default="interpolate",
                        help="[apply] what to do with a segment count the seed split never showed")
    parser.add_argument("--grid-seconds", type=float, default=GRID_SECONDS,
                        help="[apply] solver grid resolution")

    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.fit:
        for name in ("gt", "split"):
            if getattr(args, name) is None:
                raise SystemExit(f"--fit requires --{name}")
        if not math.isfinite(args.residual_scale) or args.residual_scale <= 0:
            raise SystemExit("--residual-scale must be positive")
        if not math.isfinite(args.duration_weight) or args.duration_weight < 0:
            raise SystemExit("--duration-weight must be non-negative")
        if args.min_count_episodes < 1:
            raise SystemExit("--min-count-episodes must be at least 1")
        return run_fit(args)

    for name in ("predictions", "calibration"):
        if getattr(args, name) is None:
            raise SystemExit(f"--apply requires --{name}")
    if not math.isfinite(args.grid_seconds) or args.grid_seconds <= 0:
        raise SystemExit("--grid-seconds must be positive")
    return run_apply(args)


if __name__ == "__main__":
    raise SystemExit(main())
