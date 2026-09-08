"""Uncertainty and hypothesis testing for the arm comparison.

The sampling structure here is not i.i.d. and pretending otherwise would
produce confidence intervals several times too narrow. Episodes come from 16
component datasets nested in 4 task families; within a component the task is
scripted, so episodes are strongly correlated -- in several components all
fifty episodes share one human label sequence. The effective sample size for a
cross-corpus claim is therefore closer to the number of COMPONENTS than to the
number of episodes, and arguably closer still to the number of FAMILIES.

Small-cluster caveat, stated because it bounds every interval below
-----------------------------------------------------------------
A percentile cluster bootstrap is known to under-cover when the number of
clusters is small, and 16 (let alone 4) is small. We therefore report a
bias-corrected interval rather than a raw percentile one, and every result
carries its cluster count so a reader can discount accordingly. This reduces
but does NOT eliminate the under-coverage: intervals at K=16 should be read as
optimistically narrow, and a marginal result at K=4 should not be believed at
all. The family-level sign test is reported alongside precisely because it
makes no distributional assumption.

Everything in this module follows from that:

* the primary interval is a **two-stage cluster bootstrap** that resamples
  datasets first and episodes within them second;
* the primary test is **paired at the episode level** (both arms annotate the
  identical episode, so pairing removes episode difficulty entirely) but its
  uncertainty is still clustered by dataset;
* a **sign test over datasets** (n = 10) is reported alongside as the
  conservative fallback that assumes nothing about within-dataset structure;
* p-values across arms are corrected with **Holm**, which needs no
  independence assumption between the hypotheses.

All resampling is seeded and deterministic.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Interval:
    point: float
    low: float
    high: float
    level: float
    n_units: int
    n_clusters: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "point": self.point,
            "ci_low": self.low,
            "ci_high": self.high,
            "level": self.level,
            "n_units": self.n_units,
            "n_clusters": self.n_clusters,
        }

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.low:.4f}, {self.high:.4f}]"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _inverse_normal_cdf(p: float) -> float:
    """Acklam's rational approximation; ample for bootstrap quantiles."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def cluster_bootstrap_mean(
    values_by_cluster: dict[str, Sequence[float]],
    *,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
    weight_by: str = "cluster",
) -> Interval:
    """Two-stage bootstrap CI for a mean over clustered observations.

    ``weight_by='cluster'`` gives every dataset equal weight regardless of how
    many episodes it contributes. That is the right default here: the datasets
    have between 98 and 250 episodes, and episode-weighting would let the
    largest corpus decide the headline number. ``weight_by='unit'`` reproduces
    the naive episode-weighted mean and is reported only as a secondary.
    """
    clusters = sorted(values_by_cluster)
    if not clusters:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0, 0)
    rng = random.Random(seed)

    def statistic(sample: Sequence[str], resample_units: bool) -> float:
        per_cluster: list[float] = []
        all_units: list[float] = []
        for name in sample:
            values = values_by_cluster[name]
            if not values:
                continue
            if resample_units:
                drawn = [values[rng.randrange(len(values))] for _ in range(len(values))]
            else:
                drawn = list(values)
            per_cluster.append(_mean(drawn))
            all_units.extend(drawn)
        if weight_by == "unit":
            return _mean(all_units)
        return _mean(per_cluster)

    point = statistic(clusters, resample_units=False)
    draws: list[float] = []
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))]
        value = statistic(sample, resample_units=True)
        if not math.isnan(value):
            draws.append(value)
    draws.sort()
    if not draws or math.isnan(point):
        return Interval(point, float("nan"), float("nan"), level, 0, len(clusters))

    # Bias-corrected percentile interval. The raw percentile bootstrap
    # under-covers badly at the cluster counts used here; the BC adjustment
    # costs nothing and recentres the interval when the bootstrap distribution
    # is skewed relative to the point estimate.
    below = sum((d < point) + 0.5 * (d == point) for d in draws)
    fraction = below / len(draws)
    alpha = (1.0 - level) / 2.0
    if 0.0 < fraction < 1.0:
        z0 = _inverse_normal_cdf(fraction)
        lo_q = _normal_cdf(2 * z0 + _inverse_normal_cdf(alpha))
        hi_q = _normal_cdf(2 * z0 + _inverse_normal_cdf(1.0 - alpha))
    else:
        lo_q, hi_q = alpha, 1.0 - alpha
    low = draws[max(0, min(len(draws) - 1, int(lo_q * len(draws))))]
    high = draws[max(0, min(len(draws) - 1, int(hi_q * len(draws))))]
    n_units = sum(len(v) for v in values_by_cluster.values())
    return Interval(point, low, high, level, n_units, len(clusters))


def paired_cluster_bootstrap(
    deltas_by_cluster: dict[str, Sequence[float]],
    *,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
    weight_by: str = "cluster",
) -> tuple[Interval, float]:
    """CI and two-sided p-value for a paired per-episode difference.

    ``deltas_by_cluster[dataset]`` holds ``metric(new) - metric(baseline)`` for
    each episode of that dataset. The p-value is the bootstrap proportion of
    resamples whose mean lands on the opposite side of zero, doubled -- the
    standard percentile-bootstrap test. It is reported to two significant
    figures and floored at ``1 / n_boot``, because a bootstrap cannot resolve
    a p-value smaller than its own resolution and quoting one would be a
    fabricated precision.
    """
    interval = cluster_bootstrap_mean(
        deltas_by_cluster, n_boot=n_boot, level=level, seed=seed, weight_by=weight_by
    )
    # A NaN point estimate compares False against both 0 tests below, so every
    # resample would look non-crossing and the function would return the SMALLEST
    # p it can express -- reporting maximal significance for a broken input.
    if math.isnan(interval.point):
        return interval, float("nan")
    clusters = sorted(deltas_by_cluster)
    rng = random.Random(seed + 1)
    crossings = 0
    total = 0
    for _ in range(n_boot):
        sample = [clusters[rng.randrange(len(clusters))] for _ in range(len(clusters))]
        per_cluster: list[float] = []
        units: list[float] = []
        for name in sample:
            values = deltas_by_cluster[name]
            if not values:
                continue
            drawn = [values[rng.randrange(len(values))] for _ in range(len(values))]
            per_cluster.append(_mean(drawn))
            units.extend(drawn)
        value = _mean(units) if weight_by == "unit" else _mean(per_cluster)
        if math.isnan(value):
            continue
        total += 1
        if (interval.point > 0 and value <= 0) or (interval.point <= 0 and value >= 0):
            crossings += 1
    if total == 0:
        return interval, float("nan")
    p = min(1.0, 2.0 * crossings / total)
    return interval, max(p, 1.0 / n_boot)


def sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test, ties excluded.

    Applied over COMPONENTS (and separately over FAMILIES), not episodes. Note
    the hard limit: the smallest attainable two-sided p is 2/2**n, so with 4
    families the test can never go below 0.125 and CANNOT reach significance at
    any conventional threshold. That is not a defect to work around; it is the
    honest statement of how much a 4-family corpus can support. With 16
    components the floor is 3.05e-05. Low power is the honest position: a result that only shows up when
    every episode is treated as independent is a result about pseudo-
    replication. When this test and the cluster bootstrap disagree, the
    disagreement is itself the finding and both are reported.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    def comb(a: int, b: int) -> int:
        return math.comb(a, b)
    extreme = min(wins, losses)
    tail = sum(comb(n, k) for k in range(extreme + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def holm_correction(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment.

    Controls the family-wise error rate across the arm comparisons without
    assuming the tests are independent, which they are not: every arm is
    compared against the same baseline on the same episodes.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    n = len(items)
    out: dict[str, float] = {}
    running = 0.0
    for index, (key, value) in enumerate(items):
        adjusted = min(1.0, (n - index) * value)
        running = max(running, adjusted)
        out[key] = running
    return out


def bootstrap_paired_difference_by_dataset(
    per_episode: dict[str, dict[str, float]],
    arm_a: str,
    arm_b: str,
) -> dict[str, list[float]]:
    """Build ``{dataset: [delta, ...]}`` from per-episode records.

    ``per_episode`` maps ``"<dataset>/<episode>"`` to ``{arm: value}``.
    Episodes where either arm is missing are dropped and counted, never
    imputed: silently filling a failed run with a neutral value would reward
    a tool for crashing on hard episodes.
    """
    out: dict[str, list[float]] = {}
    for key, values in sorted(per_episode.items()):
        if arm_a not in values or arm_b not in values:
            continue
        dataset = key.split("/", 1)[0]
        out.setdefault(dataset, []).append(float(values[arm_a]) - float(values[arm_b]))
    return out


def describe_missing(
    per_episode: dict[str, dict[str, float]], arms: Sequence[str]
) -> dict[str, dict[str, int]]:
    """Per-arm count of episodes with no usable result, for the report.

    Failure rate is itself a quality metric and must never be hidden by
    dropping rows.
    """
    out: dict[str, dict[str, int]] = {}
    for key, values in per_episode.items():
        dataset = key.split("/", 1)[0]
        bucket = out.setdefault(dataset, {"total": 0, **dict.fromkeys(arms, 0)})
        bucket["total"] += 1
        for arm in arms:
            if arm not in values:
                bucket[arm] += 1
    return out


def decoding_noise_band(
    repeat_values: dict[str, dict[int, dict[Any, float]]],
    *,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Within-arm, between-repeat difference: the floor on any claimed effect.

    ``repeat_values[cluster][repeat][episode]`` holds one arm's metric value for
    that episode on that repeat. Neither tool seeds its generation call, so
    running the same arm twice gives different answers; the typical spread
    between repeats is the smallest difference that could mean anything, and an
    effect below it is not an effect.

    Episodes are matched **by episode id**, not by position in a list. An
    earlier version zipped per-repeat lists positionally, which is correct only
    while every repeat annotates exactly the same episodes in the same order --
    and the moment one repeat fails on one episode, every later episode is
    paired with the wrong one and the resulting margin is silently meaningless.
    Only episodes present in all repeats contribute.

    The *absolute* pairwise difference is used deliberately: a signed
    difference has expectation zero, so its interval would measure estimation
    error and shrink as data accumulated, making a genuinely identical arm
    harder to certify the more evidence was collected.
    """
    by_cluster: dict[str, list[float]] = {}
    for cluster, repeats in repeat_values.items():
        indices = sorted(repeats)
        if len(indices) < 2:
            continue
        shared = set(repeats[indices[0]])
        for index in indices[1:]:
            shared &= set(repeats[index])
        per_episode: list[float] = []
        for episode in sorted(shared):
            values = [repeats[index][episode] for index in indices]
            pairs = [
                abs(values[a] - values[b])
                for a in range(len(values))
                for b in range(a + 1, len(values))
            ]
            if pairs:
                per_episode.append(sum(pairs) / len(pairs))
        if per_episode:
            by_cluster[cluster] = per_episode
    if not by_cluster:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0, 0)
    return cluster_bootstrap_mean(by_cluster, n_boot=n_boot, level=level, seed=seed)


def equivalence_test(
    deltas_by_cluster: dict[str, Sequence[float]],
    margin: float,
    *,
    n_boot: int = 10000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Two one-sided tests for equivalence within +/- ``margin``.

    C1 predicts a null, and non-rejection is not evidence of equivalence -- at
    three clusters it is mostly evidence of low power. TOST inverts the burden:
    equivalence is *concluded* only when the (1 - 2*alpha) interval on the
    difference lies entirely inside the margin, so an imprecise study fails to
    conclude equivalence rather than being rewarded for imprecision.

    The conventional TOST interval is (1 - 2*alpha), i.e. 90% at alpha = 0.05;
    using a 95% interval here would be conservative but is not the standard
    construction and would understate power.
    """
    if not math.isfinite(margin) or margin <= 0:
        return {"verdict": "UNDEFINED", "reason": "margin is not a positive finite number"}
    interval = cluster_bootstrap_mean(
        deltas_by_cluster, n_boot=n_boot, level=1.0 - 2.0 * alpha, seed=seed
    )
    if math.isnan(interval.point) or math.isnan(interval.low) or math.isnan(interval.high):
        return {"verdict": "INCONCLUSIVE", "reason": "interval undefined", **interval.as_dict()}
    inside = (interval.low > -margin) and (interval.high < margin)
    beyond = (interval.low > margin) or (interval.high < -margin)
    return {
        "verdict": "EQUIVALENT" if inside else ("DIFFERENT" if beyond else "INCONCLUSIVE"),
        "margin": margin,
        "alpha": alpha,
        **interval.as_dict(),
        "reason": (
            "interval lies entirely inside the margin" if inside
            else "interval lies entirely outside the margin" if beyond
            else "interval spans the margin: too imprecise to conclude equivalence, "
                 "which is NOT the same as having shown the arms agree"
        ),
    }
