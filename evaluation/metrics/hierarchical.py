"""Three-level (family / component / episode) bootstrap and its variance components.

Why this module exists
----------------------
`evaluation_plan.md` §8.1 clusters at the COMPONENT: 16 recording sessions,
resampled first, episodes second. That is right as far as it goes -- treating
797 episodes as 797 independent observations would understate every interval
several-fold -- but it stops one level short of the real design. The 16
components nest in only **4 task families** (`base4-clean-table`,
`base4-mobile-door`, `u850-bag-place`, `u850-fridge-drink`), and the round-1
review measured a family-level ICC of roughly 0.33: two episodes drawn from
different components of the *same* family still share about a third of their
variance, because they share a robot, a scene, a script and an annotator
session.

A component-level bootstrap treats those 16 components as 16 independent
draws. They are not. Under family-correlated data the component bootstrap
resamples 16 units whose effective count is closer to 4-6, so its sampling
distribution is too tight and its nominal 95% interval covers less than 95% of
the time. That is the specific failure this module exists to bound: the
component interval is not merely "a bit optimistic", it is a coverage claim
that is not met, and a marginal C2/C3 result decided on it would be decided by
an artefact of the clustering choice.

The honest interval and the optimistic one -- report BOTH
--------------------------------------------------------
`family_cluster_bootstrap` resamples families, then components within each
drawn family, then episodes within each drawn component. **With 4 families
this has very low power by construction**, in the same way and for the same
reason that `stats.sign_test` over 4 families can never go below p = 0.125.
That is not a defect to tune away. It is the honest statement of how much a
4-family corpus can support, and it is the whole reason the corpus limitation
is a headline limitation rather than a footnote.

So:

* the **family-level interval is the honest one** -- it is the interval whose
  coverage claim is actually met when families are correlated;
* the **component-level interval (`stats.cluster_bootstrap_mean`) is the
  optimistic one** -- narrower, and narrower by an amount that has nothing to
  do with evidence;
* **both are reported, on the same row, always.** `report_both_levels` exists
  precisely so that producing one without the other takes deliberate effort.
  Quoting whichever came out narrower would be a post-hoc choice of analysis,
  which is the thing this pre-registration exists to prevent. If the two
  disagree about whether an effect excludes zero, the disagreement *is* the
  finding, exactly as §8.1 already says for the bootstrap-versus-sign-test
  case.

`variance_components` reports the ICC actually measured on the data at hand,
so the report can state the dependence rather than assert it from a review
note. A study that says "clustering matters" without a number is asking to be
taken on faith.

Measured, not asserted: what this actually buys
-----------------------------------------------
Coverage simulation on the real Corpus A layout -- 4 families holding 4/3/4/5
components, 40 evaluation episodes each (640 total) -- with
``y = mu + f_i + c_ij + e_ijk``, 800 replications, 400 bootstrap resamples,
nominal 95%:

| generating variances            | family-level | component-level |
|---------------------------------|--------------|-----------------|
| icc_family 0.33 (0.33/0.17/0.50)| **0.858**    | 0.657           |
| icc_family 0.60 (0.60/0.10/0.30)| **0.835**    | 0.590           |
| icc_family 0.00 (0.00/0.30/0.70)| 0.958        | 0.934           |

and for the paired test, rejection rate of a **true null** at nominal 0.05:
family-level **0.142**, component-level **0.335**.

Three things follow, and all three are stated in the report rather than only
here.

1. The component-level interval does not merely run narrow under family
   correlation -- its 95% interval covers 66% of the time and its 5% test
   rejects a true null one time in three. On family-correlated data it is not
   an approximation to a 95% interval; it is a different, much weaker
   guarantee.
2. The family-level version is a large improvement and is **still
   anticonservative**: 86% coverage, 14% type-I. Four clusters is too few for a
   cluster bootstrap to reach its nominal level, exactly as `stats.py` warns
   for 16. Neither number should be read at face value, and a family-level p
   just under 0.05 is not a 5% result.
3. When there is genuinely no family effect the family-level interval costs
   about 26% extra width for nothing. That is the price of the assumption, and
   it is why `variance_components` measures the ICC rather than assuming it.

Which interval answers which question
-------------------------------------
The two are not competing estimates of one quantity. The family-level interval
supports a claim about *task families in general* -- the families here being
four draws from the space of tasks a user might annotate -- which is the claim
the report wants to make. The component-level interval, at best, supports a
claim conditional on these four specific families and generalising only to new
recording sessions of them. Stating the second while meaning the first is the
error this module exists to prevent.

Conventions are inherited from `stats.py` deliberately: same `Interval`, same
bias-corrected percentile construction, same seeded determinism, same
`(Interval, p)` shape for the paired test. The private helpers imported from
`stats` are private by convention only; re-deriving a normal quantile function
here would risk the two modules' intervals differing for a reason no reader
could see.

Two deliberate divergences from `stats.py`, both noted where they occur:
`weight_by` is validated rather than silently falling through to the default
(a typo there would quietly change the estimand), and the paired p-value is
computed from the *same* resamples as the interval rather than from a second
independent bootstrap, so the two cannot contradict each other through Monte
Carlo noise alone.
"""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

# Private by convention within the package, not by intent to hide: sharing them
# is what keeps a family-level interval and a component-level interval on the
# same row commensurable.
from .stats import (
    Interval,
    _inverse_normal_cdf,
    _mean,
    _normal_cdf,
    cluster_bootstrap_mean,
)

# ``{family: {component: [per-episode value, ...]}}``
Nested = Mapping[str, Mapping[str, Sequence[float]]]

_WEIGHTS = ("family", "component", "unit")

_COMPONENT_INDEX = re.compile(r"\d+")


def family_of(component: str) -> str:
    """Family name for a component id such as ``u850-fridge-drink-03-BC-v30``.

    The corpus names components ``<family>-<NN>-<suffixes>``, so the family is
    everything before the first all-digit token. A name with no such token is
    returned unchanged and therefore becomes a singleton family, which silently
    degrades the analysis back to component-level clustering -- detectable
    because the family count in the reported interval rises towards 16 instead
    of staying at 4. Pass an explicit mapping to `group_by_family` whenever the
    grouping matters, which for the headline numbers it does.
    """
    parts = component.split("-")
    for index, part in enumerate(parts):
        if index > 0 and _COMPONENT_INDEX.fullmatch(part):
            return "-".join(parts[:index])
    return component


def group_by_family(
    values_by_component: Mapping[str, Sequence[float]],
    family_of_component: Mapping[str, str] | None = None,
) -> dict[str, dict[str, list[float]]]:
    """Reshape ``{component: values}`` into ``{family: {component: values}}``.

    An explicit ``family_of_component`` mapping must cover every component: a
    component missing from it raises rather than falling back to `family_of`,
    because a single silently mis-grouped component changes the number of
    clusters and hence every interval in the table.
    """
    out: dict[str, dict[str, list[float]]] = {}
    for component in sorted(values_by_component):
        if family_of_component is None:
            family = family_of(component)
        elif component in family_of_component:
            family = family_of_component[component]
        else:
            raise ValueError(f"no family given for component {component!r}")
        out.setdefault(family, {})[component] = list(values_by_component[component])
    return out


def sanitise_nested(values_by_family: Nested) -> tuple[dict[str, dict[str, list[float]]], int]:
    """Drop non-finite values; return the cleaned structure and the drop count.

    A single NaN metric would otherwise poison its component mean, its family
    mean and the point estimate, turning the whole interval into NaN -- which
    over 640 episodes is close to certain and would make the module useless.
    Dropping instead is safe only because it is visible: the surviving episode
    count is carried into `Interval.n_units`, and the count dropped is returned
    here so the caller can report it. **Check both.** An interval over 340 of
    640 episodes is not an interval over the corpus, and nothing downstream can
    tell the difference on its own.

    This is not in tension with §8.2's "failures are data": an episode an arm
    could not annotate is handled upstream as a failure rate. What is dropped
    here is a metric that came back undefined for an episode both arms did
    produce.
    """
    clean: dict[str, dict[str, list[float]]] = {}
    dropped = 0
    for family in sorted(values_by_family):
        components: dict[str, list[float]] = {}
        for component in sorted(values_by_family[family]):
            kept: list[float] = []
            for value in values_by_family[family][component]:
                number = float(value)
                if math.isfinite(number):
                    kept.append(number)
                else:
                    dropped += 1
            if kept:
                components[component] = kept
        if components:
            clean[family] = components
    return clean, dropped


def count_levels(values_by_family: Nested) -> tuple[int, int, int]:
    """``(families, components, episodes)`` after sanitisation.

    `Interval` carries one cluster count, and for a family bootstrap that slot
    holds the family count -- 4 -- so a reader discounts the interval on sight.
    The component count is still needed for the report, hence this.
    """
    clean, _ = sanitise_nested(values_by_family)
    components = sum(len(v) for v in clean.values())
    units = sum(len(e) for v in clean.values() for e in v.values())
    return len(clean), components, units


def _statistic(
    data: Mapping[str, Mapping[str, Sequence[float]]],
    family_sample: Sequence[str],
    weight_by: str,
    rng: random.Random | None,
) -> float:
    """One evaluation of the estimator over a (possibly repeated) family sample.

    ``rng=None`` disables the lower two resampling stages, giving the point
    estimate. When a family appears twice in ``family_sample`` its components
    and episodes are drawn independently for each appearance; reusing one draw
    for both would remove real resampling variance and narrow the interval.
    """
    family_means: list[float] = []
    component_means: list[float] = []
    units: list[float] = []
    for family in family_sample:
        components = data[family]
        names = sorted(components)
        if rng is not None:
            names = [names[rng.randrange(len(names))] for _ in range(len(names))]
        within: list[float] = []
        for name in names:
            values = components[name]
            if rng is not None:
                drawn = [values[rng.randrange(len(values))] for _ in range(len(values))]
            else:
                drawn = list(values)
            mean = _mean(drawn)
            within.append(mean)
            component_means.append(mean)
            units.extend(drawn)
        if within:
            family_means.append(_mean(within))
    if weight_by == "unit":
        return _mean(units)
    if weight_by == "component":
        return _mean(component_means)
    return _mean(family_means)


def _bias_corrected(
    point: float,
    draws: list[float],
    level: float,
    n_units: int,
    n_clusters: int,
) -> Interval:
    """Bias-corrected percentile interval, identical in form to `stats.py`.

    The BC adjustment matters more here than there, not less: at 4 clusters the
    bootstrap distribution is markedly skewed and a raw percentile interval is
    both mis-centred and too short. It does not repair the small-cluster
    under-coverage, it only stops it being worse than it has to be.
    """
    if not draws or math.isnan(point):
        return Interval(point, float("nan"), float("nan"), level, n_units, n_clusters)
    below = sum((value < point) + 0.5 * (value == point) for value in draws)
    fraction = below / len(draws)
    alpha = (1.0 - level) / 2.0
    if 0.0 < fraction < 1.0:
        z0 = _inverse_normal_cdf(fraction)
        lo_q = _normal_cdf(2 * z0 + _inverse_normal_cdf(alpha))
        hi_q = _normal_cdf(2 * z0 + _inverse_normal_cdf(1.0 - alpha))
    else:
        # Every draw on one side of the point estimate: z0 is infinite and the
        # correction is undefined. Fall back to the raw percentile rather than
        # emit an interval built from an infinity.
        lo_q, hi_q = alpha, 1.0 - alpha
    last = len(draws) - 1
    low = draws[max(0, min(last, int(lo_q * len(draws))))]
    high = draws[max(0, min(last, int(hi_q * len(draws))))]
    return Interval(point, low, high, level, n_units, n_clusters)


def _draws(
    data: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    n_boot: int,
    seed: int,
    weight_by: str,
) -> tuple[float, list[float]]:
    families = sorted(data)
    rng = random.Random(seed)
    point = _statistic(data, families, weight_by, None)
    values: list[float] = []
    for _ in range(n_boot):
        sample = [families[rng.randrange(len(families))] for _ in range(len(families))]
        value = _statistic(data, sample, weight_by, rng)
        if not math.isnan(value):
            values.append(value)
    # Sorted here, once, because `_bias_corrected` indexes by quantile position.
    # Unsorted draws yield an interval whose ends are two arbitrary resamples --
    # frequently inverted, occasionally excluding the point estimate, and never
    # obviously wrong to a reader.
    values.sort()
    return point, values


def family_cluster_bootstrap(
    values_by_family: Nested,
    *,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
    weight_by: str = "family",
) -> Interval:
    """Three-stage bootstrap CI: families, then components, then episodes.

    ``weight_by='family'`` is the default and gives each task family equal
    weight, so the 5-component `u850-fridge-drink` family cannot outvote the
    3-component `base4-mobile-door` one -- the same argument §8.1 makes for
    component weighting, applied one level up. ``'component'`` and ``'unit'``
    reproduce the component- and episode-weighted means for comparison; all
    three are estimating different quantities and the choice is declared, not
    discovered.

    The returned `Interval` reports ``n_clusters`` as the number of FAMILIES,
    which on Corpus A is 4. Read it as a warning label. Degenerate structure is
    represented rather than repaired:

    * **one family** -- the family stage resamples a single unit, so the
      interval carries no between-family variance at all and must not be
      reported as a family-level interval. ``n_clusters == 1`` is the tell.
    * **a family with one component** -- that family contributes no
      between-component variance. Correct, and nothing to fix: the data contain
      no information about component variation inside it.
    * **a component with one episode** -- likewise contributes no within-
      component variance.
    * **empty input** (or input entirely non-finite) -- an all-NaN interval with
      zero units and zero clusters, never a spurious 0.0.
    """
    if weight_by not in _WEIGHTS:
        # stats.py treats any unrecognised value as its default; here a typo
        # would silently swap the estimand, so it is an error instead.
        raise ValueError(f"weight_by must be one of {_WEIGHTS}, got {weight_by!r}")
    data, _ = sanitise_nested(values_by_family)
    n_units = sum(len(e) for v in data.values() for e in v.values())
    if not data:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0, 0)
    point, draws = _draws(data, n_boot=n_boot, seed=seed, weight_by=weight_by)
    return _bias_corrected(point, draws, level, n_units, len(data))


def paired_family_bootstrap(
    deltas_by_family: Nested,
    *,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
    weight_by: str = "family",
) -> tuple[Interval, float]:
    """CI and two-sided p for a paired per-episode difference, clustered by family.

    ``deltas_by_family[family][component]`` holds ``metric(arm) -
    metric(baseline)`` for each episode of that component, both arms having
    annotated the identical episode. Signature and return shape follow
    `stats.paired_cluster_bootstrap`; the p-value is the bootstrap proportion of
    resamples landing on the far side of zero, doubled, and floored at
    ``1 / n_boot`` because a bootstrap cannot resolve finer and quoting more
    would be fabricated precision.

    Unlike `stats.paired_cluster_bootstrap` the p-value is computed from the
    *same* resamples as the interval rather than from a second independent
    bootstrap. Two independent bootstraps can disagree at the margin -- an
    interval excluding zero beside a p above the threshold -- through Monte
    Carlo noise alone, and at 4 clusters that noise is not small. The two can
    still differ because the interval is bias-corrected and the p-value is not,
    which is a real property of the construction rather than an artefact.

    A NaN point estimate returns ``(interval, nan)``: it compares False against
    both sides of zero, so a naive count would see no crossings at all and
    report maximal significance for broken input. This is the same guard, and
    the same reason, as in `stats.py`.

    Expect this to be underpowered. With 4 families the bootstrap has few
    distinct family multisets to draw from, so p-values are coarse and only a
    large, consistent effect will clear any threshold. Report it beside the
    component-level p, not instead of it.
    """
    if weight_by not in _WEIGHTS:
        raise ValueError(f"weight_by must be one of {_WEIGHTS}, got {weight_by!r}")
    data, _ = sanitise_nested(deltas_by_family)
    n_units = sum(len(e) for v in data.values() for e in v.values())
    if not data:
        return Interval(float("nan"), float("nan"), float("nan"), level, 0, 0), float("nan")
    point, draws = _draws(data, n_boot=n_boot, seed=seed, weight_by=weight_by)
    interval = _bias_corrected(point, draws, level, n_units, len(data))
    if math.isnan(point) or not draws:
        return interval, float("nan")
    crossings = sum(
        1 for value in draws
        if (point > 0 and value <= 0) or (point <= 0 and value >= 0)
    )
    p = min(1.0, 2.0 * crossings / len(draws))
    return interval, max(p, 1.0 / n_boot)


@dataclass(frozen=True)
class VarianceComponents:
    """Nested random-effects decomposition of one metric over the corpus."""

    var_family: float
    var_component: float
    var_episode: float
    icc_family: float
    icc_component: float
    n_families: int
    n_components: int
    n_units: int
    mean_cluster_size: float
    design_effect_family: float
    effective_n_family: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "var_family": self.var_family,
            "var_component": self.var_component,
            "var_episode": self.var_episode,
            "icc_family": self.icc_family,
            "icc_component": self.icc_component,
            "n_families": self.n_families,
            "n_components": self.n_components,
            "n_units": self.n_units,
            "mean_cluster_size": self.mean_cluster_size,
            "design_effect_family": self.design_effect_family,
            "effective_n_family": self.effective_n_family,
        }


def _nan_components(n_families: int, n_components: int, n_units: int) -> VarianceComponents:
    nan = float("nan")
    return VarianceComponents(
        nan, nan, nan, nan, nan, n_families, n_components, n_units, nan, nan, nan
    )


def variance_components(values_by_family: Nested) -> VarianceComponents:
    """Two-fold nested ANOVA estimate of family / component / episode variance.

    Gives the report a measured ICC instead of an asserted one:
    ``icc_family`` is the correlation between two episodes from *different
    components of the same family*, and ``icc_component`` the correlation
    between two episodes of the same component. If ``icc_family`` really is
    around 0.33, then clustering at the component treats 16 units as
    independent when their effective count is far smaller, and
    ``effective_n_family`` says by how much.

    Unbalanced expected-mean-square coefficients (Searle et al.) are used
    rather than the balanced-design shortcut, because the families hold 3 to 5
    components and 150 to 249 episodes; the balanced formula would misattribute
    variance between the levels in proportion to that imbalance.

    Precision, because the estimate will be quoted: on the Corpus A layout the
    variance estimates are unbiased in the mean (4000-replication simulation
    recovers 0.3294 / 0.1696 / 0.5000 against a true 0.33 / 0.17 / 0.50) but
    ``var_family`` has a standard deviation of 0.31 and ``icc_family`` one of
    0.18. **The ICC is barely estimable at four families.** Report it as
    "roughly a third", never to two decimals, and do not treat a difference
    between two metrics' ICCs as real.

    Estimates are clamped at zero. An ANOVA variance component can come out
    negative when the true value is near zero and it is not reportable as such;
    the clamp biases the ICC upward slightly, which is the conservative
    direction for the argument being made here (it cannot manufacture a case
    for family clustering that the data do not contain, it can only fail to
    subtract the last of the noise).

    Returns NaNs, never zeros, where a level is not identified: fewer than two
    families leaves the family variance undefined, and one component per family
    confounds family with component entirely (that case reports
    ``var_component = 0`` and folds the shared variance into ``var_family``,
    which is what the data can distinguish and no more).
    """
    data, _ = sanitise_nested(values_by_family)
    families = sorted(data)
    n_families = len(families)
    n_components = sum(len(v) for v in data.values())
    n_units = sum(len(e) for v in data.values() for e in v.values())
    if n_units == 0 or n_families == 0:
        return _nan_components(n_families, n_components, n_units)

    grand = sum(x for v in data.values() for e in v.values() for x in e) / n_units

    ss_episode = 0.0
    ss_component = 0.0
    ss_family = 0.0
    sum_nij_sq = 0.0            # sum over components of n_ij^2
    sum_nij_sq_over_ni = 0.0    # sum over families of (sum_j n_ij^2) / n_i
    sum_ni_sq = 0.0             # sum over families of n_i^2
    for family in families:
        components = data[family]
        n_family = sum(len(v) for v in components.values())
        mean_family = sum(sum(v) for v in components.values()) / n_family
        inner = 0.0
        for component in sorted(components):
            values = components[component]
            size = len(values)
            mean_component = _mean(values)
            ss_episode += sum((x - mean_component) ** 2 for x in values)
            ss_component += size * (mean_component - mean_family) ** 2
            inner += size * size
            sum_nij_sq += size * size
        sum_nij_sq_over_ni += inner / n_family
        sum_ni_sq += n_family * n_family
        ss_family += n_family * (mean_family - grand) ** 2

    df_episode = n_units - n_components
    df_component = n_components - n_families
    df_family = n_families - 1
    if df_family <= 0:
        return _nan_components(n_families, n_components, n_units)

    ms_episode = ss_episode / df_episode if df_episode > 0 else 0.0
    var_episode = ms_episode

    if df_component > 0:
        k1 = (n_units - sum_nij_sq_over_ni) / df_component
        ms_component = ss_component / df_component
        var_component = max(0.0, (ms_component - ms_episode) / k1) if k1 > 0 else 0.0
    else:
        var_component = 0.0

    k2 = (sum_nij_sq_over_ni - sum_nij_sq / n_units) / df_family
    k3 = (n_units - sum_ni_sq / n_units) / df_family
    ms_family = ss_family / df_family
    if k3 <= 0:
        return _nan_components(n_families, n_components, n_units)
    var_family = max(0.0, (ms_family - ms_episode - k2 * var_component) / k3)

    total = var_family + var_component + var_episode
    if total <= 0:
        # Constant metric: every variance is zero and the ratio is 0/0. There is
        # no dependence to report, and reporting 0.0 would claim independence.
        return _nan_components(n_families, n_components, n_units)

    icc_family = var_family / total
    icc_component = (var_family + var_component) / total
    # Kish's average cluster size, not the plain mean: the families hold 150 to
    # 249 episodes and the plain mean understates the design effect when sizes
    # differ.
    mean_cluster_size = sum_ni_sq / n_units
    design_effect = 1.0 + (mean_cluster_size - 1.0) * icc_family
    return VarianceComponents(
        var_family=var_family,
        var_component=var_component,
        var_episode=var_episode,
        icc_family=icc_family,
        icc_component=icc_component,
        n_families=n_families,
        n_components=n_components,
        n_units=n_units,
        mean_cluster_size=mean_cluster_size,
        design_effect_family=design_effect,
        effective_n_family=n_units / design_effect if design_effect > 0 else float("nan"),
    )


def flatten_to_components(values_by_family: Nested) -> dict[str, list[float]]:
    """``{component: values}`` for handing to `stats.cluster_bootstrap_mean`.

    Does NOT sanitise: `cluster_bootstrap_mean` propagates a NaN into its point
    estimate, so pass `sanitise_nested`'s output if the metric can be undefined.
    Raises if a component id appears under two families: silently merging them
    would fabricate a component that exists in neither.
    """
    out: dict[str, list[float]] = {}
    for family in sorted(values_by_family):
        for component in sorted(values_by_family[family]):
            if component in out:
                raise ValueError(f"component {component!r} appears in more than one family")
            out[component] = list(values_by_family[family][component])
    return out


def report_both_levels(
    values_by_family: Nested,
    *,
    n_boot: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, Any]:
    """Family-level and component-level intervals side by side, plus the ICC.

    This function exists to make reporting one interval without the other take
    deliberate effort. The family interval is the honest one and the component
    interval is the optimistic one; the ratio of their widths is the price of
    the clustering assumption, stated in the open. Neither is "the" answer and
    picking the narrower after seeing both would be a post-hoc analysis choice.
    """
    data, dropped = sanitise_nested(values_by_family)
    family = family_cluster_bootstrap(
        data, n_boot=n_boot, level=level, seed=seed, weight_by="family"
    )
    # Sanitised first, deliberately: `cluster_bootstrap_mean` propagates a NaN
    # into its point estimate, so feeding it the raw data would put a NaN
    # component-level interval beside a finite family-level one and invite the
    # reader to quote the only number on the row that survived.
    component = cluster_bootstrap_mean(
        flatten_to_components(data),
        n_boot=n_boot, level=level, seed=seed, weight_by="cluster",
    )
    family_width = family.high - family.low
    component_width = component.high - component.low
    ratio = (
        family_width / component_width
        if math.isfinite(family_width) and math.isfinite(component_width) and component_width > 0
        else float("nan")
    )
    return {
        "family_level": family.as_dict(),
        "component_level": component.as_dict(),
        "width_ratio_family_over_component": ratio,
        "variance_components": variance_components(values_by_family).as_dict(),
        "non_finite_values_dropped": dropped,
        "note": (
            "family_level is the honest interval and component_level the optimistic one; "
            "report both. With 4 families the family-level interval is very low power BY "
            "CONSTRUCTION and a wide interval there is a statement about the corpus, not "
            "about the arm."
        ),
    }
