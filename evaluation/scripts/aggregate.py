#!/usr/bin/env python
"""Aggregate per-episode scores into the comparison tables.

Produces, for one chosen matcher:

* a per-arm summary with cluster-bootstrap confidence intervals
* a per-dataset breakdown for every arm
* paired per-episode contrasts, with clustered CIs, bootstrap p-values, a
  dataset-level sign test and ONE Holm correction over the whole family
* a failure-rate table

The unit of analysis is the COMPONENT dataset, never the episode: within a
component the task is scripted and its episodes are near-replicates.

Two studies, two ways of choosing what gets contrasted
------------------------------------------------------
**Generation** (``evaluation_plan.md``) contrasts every arm against a baseline
named on the command line -- per (profile, camera) cell, or globally. That is
enough when the family is "every arm against one baseline".

**Fixed-label alignment** (``alignment_plan.md`` section 8) pre-registers eight
specific pairs, and one arm is the numerator of more than one of them. A per-arm
baseline key cannot express that set -- it maps each arm to a single baseline --
so the family would have to be assembled from six separate invocations, and Holm
would then correct six small families instead of the single family of 24 tests
that was pre-registered. Splitting a family that way is anticonservative: it is
strictly easier to clear. ``--contrasts`` therefore takes the family as data: a
LIST of (numerator, baseline) pairs computed in one pass, with one Holm
correction over their union (D13).

Three rules are enforced here rather than left to the writer of the report.

**Polarity is declared, not inferred** (D14). ``LOWER_IS_BETTER`` decides which
sign counts as a win. An error metric missing from it is reported with its sign
inverted -- the arm with the SMALLEST boundary error is starred as significantly
worse. That is the class of silent error behind the withdrawn C4 claim, so the
alignment error metrics are registered below.

**Arms are only contrasted against a baseline at the SAME supervision level.**
A calibrated arm compared against an uncalibrated baseline measures the ten
labelled seed episodes, not the configuration. The generation study overrode
that gate from the command line to force C4 through, and C4 is withdrawn. In
contrast-list mode the gate is NOT overridable by any flag. A comparison that
must cross levels declares ``cross_level: true`` in the contrast config, which
permits it, stamps ``supervision_asymmetry`` on every row of the result, and
obliges the report generator to render that stamp. The previous study computed
the stamp and never printed it, which is how a mismatched comparison reached the
headline; ``make_alignment_report.py`` refuses to write a report that drops it.

**Equalisation is scoped to the dimension under test** (D12). ``equalised:
false`` is a property of an ARM ("it sees two cameras"), so using it to exclude
comparisons also excludes the contrast that is ABOUT cameras -- here, the
numerator of the primary claim. In contrast-list mode the gate instead compares
the two arms' declared flags, derives the dimensions on which they actually
differ, and refuses only when they differ on something the contrast does not
declare as under test. A contrast that declares no dimension is not refused:
its confounded dimensions are stamped on every row instead, because a comparison
silently dropped and a comparison silently confounded are both worse than one
that carries its own caveat.

Quarantined arms (those consuming ground truth: oracle labels, oracle segment
count) are excluded from every contrast and printed in a separate section
labelled as upper bounds.

Contrast config format (``configs/alignment_contrasts.yaml``, written by the
harness author, read here)::

    family: alignment_v1            # optional label, recorded in the output
    metrics: [b_hit@3, placed_fraction, macro_temporal_iou]   # the Holm family
    contrasts:
      - id: 4
        numerator: align_video_stack_cal
        baseline: align_video_stack
        level: labels_plus_seed     # recorded; arm metadata stays authoritative
        claim: A3
        cross_level: true           # supervision levels differ, deliberately
        dimension: calibration      # str, list, or "all" for a deliberately
                                    # confounded margin contrast

Only ``numerator`` and ``baseline`` are required. Everything else is optional and
absence is reported rather than assumed.
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
sys.path.insert(0, str(HERE))

import jsonl_io  # noqa: E402
from contrast_policy import contrast_policy  # noqa: E402

from metrics.stats import (  # noqa: E402
    bootstrap_paired_difference_by_dataset,
    cluster_bootstrap_mean,
    holm_correction,
    paired_cluster_bootstrap,
    sign_test,
)

# Metrics where a LOWER value is better. Without this, a "win" is counted as
# any positive difference and an arm that doubles its boundary error would be
# reported as beating the baseline on that row. Polarity is declared here, once,
# rather than inferred from the metric name.
# Only UNSIGNED error magnitudes belong here. `count_error` is signed
# (negative = under-segmentation), so declaring it lower-is-better would report
# systematic under-segmentation as a starred win. Use `abs_count_error`.
LOWER_IS_BETTER = {
    "abs_count_error",
    "pred_to_true_mae",
    "true_to_pred_mae",
    "chamfer",
    "hallucinated",
    "missed",
    # Fixed-label alignment (D14). Both are unsigned error magnitudes, so both
    # belong here; leaving them out inverts the sign of every alignment error
    # row and reports the most accurate arm as significantly the worst.
    # `boundary_mae_placed` averages |error| over PLACED boundaries only and is
    # therefore never reported without `placed_fraction` beside it -- dropping a
    # hard label improves it. `boundary_mae_norm` is the penalised MAE divided by
    # the episode duration, which is comparable across components but is a drop
    # counter rather than an accuracy metric (alignment_plan.md section 6.1).
    "boundary_mae_placed",
    "boundary_mae_norm",
}

# The three metrics on which claims are made, fixed before any VLM ran. The
# model-free script prior already scores 0.97 label_f1 / 0.96 edit_score /
# 0.91 f1@25 on this corpus (results/model_free_floor.json), so those metrics
# have no headroom and cannot discriminate between arms. Only fine-grained
# boundary placement does. See evaluation_plan.md §1.1.
HEADLINE = [
    "boundary_f1@0p5",
    "boundary_f1@1",
    "macro_iou",
]

# Reported for completeness and to characterise behaviour, but NO claim rests
# on them: the script prior is at or near ceiling on every one.
DESCRIPTIVE = [
    "boundary_f1@3",
    "covering_gt_by_pred",
    "f1@25",
    "f1@50",
    "edit_score",
    "mof",
    "label_f1",
    "abs_count_error",
]


# The fixed-label alignment family, pre-registered in alignment_plan.md section
# 8: two primaries plus macro tIoU for continuity with the prior work. Holm
# corrects exactly these across exactly the pre-registered contrasts -- 8 x 3 =
# 24 tests -- and nothing else. B@3 is primary because it is bounded and scores a
# missing label as a miss; `placed_fraction` is primary because it IS the drop
# rate, and every other alignment number is conditional on it.
ALIGNMENT_FAMILY = [
    "b_hit@3",
    "placed_fraction",
    "macro_temporal_iou",
]

# Computed and reported, but deliberately OUTSIDE the Holm family: adding them
# would enlarge the family from 24 to 64 tests and weaken every pre-registered
# result to buy significance for numbers no claim rests on. They are marked
# `exploratory` in the output so no report can star them.
ALIGNMENT_SECONDARY = [
    "b_hit@1",
    "b_hit@5",
    "boundary_mae_placed",
    "boundary_mae_norm",
    "calibration_applied",
]

ALIGNMENT_METRICS = ALIGNMENT_FAMILY + ALIGNMENT_SECONDARY

# What "equalised" means for an alignment arm, dimension by dimension (D12).
# Each entry maps a dimension name to the arm-metadata fields and tool flags that
# realise it. Two arms are equalised FOR A CONTRAST when the dimensions on which
# their configurations differ are the ones that contrast declares to be under
# test -- which is a property of the pair, not of either arm alone.
DIMENSION_SETTINGS: dict[str, tuple[str, ...]] = {
    "frame_format": ("frame_format", "plan.subtask_align_frame_format"),
    "cameras": ("cameras", "camera", "vlm.camera_key", "vlm.camera_keys",
                "plan.subtask_align_camera_keys"),
    "calibration": ("calibration", "plan.subtask_align_calibration_path"),
    "frame_budget": ("profile", "plan.max_frames_per_prompt", "plan.frames_per_second"),
}


def usable_score(row: dict[str, Any], population: str = "all_scored") -> bool:
    """Keep explicit scored failure penalties; never invent scores for unscorable rows.

    Older generation rows lack ``scored`` and retain their existing ok-based
    contract. Successful-output analysis is an explicit secondary population.
    """
    if population == "successful":
        return bool(row.get("ok")) and row.get("scored") is not False
    return bool(row.get("scored", row.get("ok", False)))


def filter_population(rows: list[dict[str, Any]], fields: list[str]) -> list[dict[str, Any]]:
    """Unknown eligibility is not ineligibility; refuse a silently smaller family."""
    for field in fields:
        missing = sorted({r["arm"] for r in rows if r.get(field) is None})
        if missing:
            raise SystemExit(f"population field {field!r} is missing for arms {missing}; "
                             "score every arm with the shared eligibility mask before filtering")
    return [r for r in rows if all(bool(r[field]) for field in fields)]


def load_rows(path: Path, matcher: str | None, repeat: int | None = 0) -> list[dict[str, Any]]:
    """Rows for one matcher and one repeat index.

    E1 runs the same arm several times to measure decoding noise. Those repeats
    live in the same scores file, and pooling them into the main aggregate would
    silently multiply the apparent episode count and shrink every interval.
    ``repeat=None`` returns all of them, which is what the noise-band
    computation wants.

    ``matcher=None`` disables the matcher filter. Fixed-label alignment has no
    matcher dimension at all: the labels are supplied, and `score_alignment.py`
    matches a prediction to its label by the index the tool returned rather than
    by text (D10). Filtering those rows on a matcher name that was never written
    would return nothing and read as "the arm produced no episodes".
    """
    rows = []
    for line in jsonl_io.read_text(path).splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if matcher is not None and row.get("matcher") != matcher:
            continue
        if repeat is not None and int(row.get("repeat", 0) or 0) != repeat:
            continue
        rows.append(row)
    return rows


# Exit status for "the Holm family that was computed is not the one registered".
# Distinguished from 1 (any hard failure) so a caller can tolerate exactly this
# condition -- see the comment at the bottom of main().
EXIT_FAMILY_SHORTFALL = 3


def load_contrast_config(path: Path) -> dict[str, Any]:
    """Read the pre-registered contrast family and validate it structurally.

    The family is data, not code, so it can be pre-registered, diffed and cited.
    Every failure here is fatal rather than skipped: a contrast that silently
    vanishes from the family also silently relaxes the Holm correction applied to
    the ones that remain.
    """
    import yaml

    spec = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = spec.get("contrasts")
    if not isinstance(entries, list) or not entries:
        raise SystemExit(
            f"{path} declares no `contrasts:` list. The pre-registered family is the "
            "input to this mode; refusing to run with an empty one."
        )
    seen: set[str] = set()
    contrasts: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise SystemExit(f"contrast {index} in {path} is not a mapping")
        for field in ("numerator", "baseline"):
            if not entry.get(field):
                raise SystemExit(f"contrast {index} in {path} has no `{field}`")
        identifier = str(entry.get("id") or entry.get("name") or index)
        key = f"{identifier}. {entry['numerator']} vs {entry['baseline']}"
        # Holm keys are "<contrast>::<metric>" and are split on the first "::",
        # so a contrast id containing one would silently mis-route its p-value.
        if "::" in key:
            raise SystemExit(f"contrast id/arm names must not contain '::': {key}")
        if key in seen:
            raise SystemExit(f"duplicate contrast {key} in {path}")
        seen.add(key)
        contrasts.append({**entry, "key": key, "id": identifier})
    correction = str(spec.get("correction", "holm")).lower()
    scope = str(spec.get("family_scope", "single")).lower()
    if correction != "holm" or scope != "single":
        raise SystemExit(
            f"{path} asks for correction='{correction}' family_scope='{scope}'. This script "
            "implements one Holm pass over the union of the listed contrasts and nothing "
            "else; running it while the config asks for something else would report a "
            "correction that was never applied."
        )
    return {
        "path": str(path),
        "study": spec.get("study"),
        "family": spec.get("family"),
        "alpha": float(spec.get("alpha", 0.05)),
        "metrics": spec.get("metrics"),
        "secondary_metrics": spec.get("secondary_metrics"),
        "metric_directions": spec.get("metric_directions") or {},
        "expected_tests": spec.get("expected_tests"),
        # Model-free floors come from align_floors.py, not run_arm.py, so they are
        # absent from the arms file by design. Without their metadata the fail-closed
        # policy would refuse contrasts 7 and 8 -- claim A5 -- citing missing metadata
        # rather than anything real, so the contrast config carries the contract and
        # this script merges it, refusing any disagreement with the arms file.
        "external_arms": spec.get("external_arms") or {},
        "contrasts": contrasts,
    }


def declared_dimensions(entry: dict[str, Any]) -> set[str] | str | None:
    """What the contrast says is under test: a set, the string "all", or None."""
    raw = entry.get("dimension", entry.get("dimensions"))
    if raw is None:
        return None
    values = {str(v).strip() for v in ([raw] if isinstance(raw, str) else raw)}
    values -= {""}
    if not values:
        return None
    if values & {"all", "*"}:
        return "all"
    return values


def configuration_dimensions(
    left: dict[str, Any], right: dict[str, Any]
) -> dict[str, list[str]]:
    """Dimensions on which two arms actually differ, and the settings that say so.

    Read from the arms' declared FLAGS rather than from a hand-maintained label,
    because the flags are what the tool is actually run with -- section 7.1's
    "arms differ only in the dimension under test" is a statement about them. A
    setting that differs and maps to no known dimension is reported under its own
    name rather than ignored, so an accidental difference in seed or temperature
    surfaces as an unequalised contrast instead of being absorbed silently.
    """
    if left.get("tool") != right.get("tool"):
        # A model-free floor runs no VLM and declares no tool flags, so diffing the
        # flag dicts against a VLM arm would list every flag the floor does not have
        # and bury the one difference that matters. The tool itself is the dimension.
        return {"tool": [f"{left.get('tool')} vs {right.get('tool')}"]}
    known = {name: dimension
             for dimension, names in DIMENSION_SETTINGS.items() for name in names}
    left_flags, right_flags = dict(left.get("flags") or {}), dict(right.get("flags") or {})
    differing: dict[str, list[str]] = defaultdict(list)
    for field in sorted(set(known) & (set(left) | set(right))):
        if left.get(field) != right.get(field):
            differing[known[field]].append(field)
    for field in sorted(set(left_flags) | set(right_flags)):
        if left_flags.get(field) != right_flags.get(field):
            differing[known.get(field, f"other:{field}")].append(field)
    return {dimension: sorted(set(fields)) for dimension, fields in sorted(differing.items())}


def alignment_contrast_policy(
    entry: dict[str, Any], metadata: dict[str, dict[str, Any]]
) -> tuple[str | None, str | None, dict[str, Any]]:
    """Eligibility for one pre-registered contrast: (refusal, asymmetry, scope).

    Fail-closed on metadata, like ``contrast_policy``, but it differs from that
    function in the two ways section 4.1 and D12 require:

    * the supervision gate has no command-line override. The only way past it is
      ``cross_level: true`` on the contrast itself -- a declaration that lives in a
      reviewable, pre-registered file rather than in one invocation's arguments,
      and that does not make the asymmetry disappear but publishes it;
    * equalisation is judged for the PAIR, against the dimension the contrast
      declares, instead of on a per-arm boolean that cannot express "this
      comparison is about cameras".
    """
    arm, baseline = entry["numerator"], entry["baseline"]
    for name in (arm, baseline):
        meta = metadata.get(name)
        if not meta or not meta.get("supervision") or not meta.get("tool"):
            return f"missing metadata or supervision for '{name}'", None, {}
        if meta.get("quarantined"):
            return f"quarantined: '{name}' consumes ground truth; upper bound only", None, {}
    left, right = metadata[arm], metadata[baseline]
    warnings: list[str] = []
    differing = configuration_dimensions(left, right)
    declared = declared_dimensions(entry)
    scope: dict[str, Any] = {
        "differs_on": sorted(differing),
        "differing_settings": differing,
        "declared": sorted(declared) if isinstance(declared, set) else declared,
        "unequalised_arms": sorted(
            name for name in (arm, baseline) if metadata[name].get("equalised") is False
        ),
        "warnings": warnings,
    }
    if arm == baseline:
        return "numerator and baseline are the same arm", None, scope
    if not differing:
        warnings.append(
            f"'{arm}' and '{baseline}' declare identical configurations; any difference "
            "measured here is decoding noise, not a configuration effect"
        )
    if declared == "all":
        warnings.append(
            "dimension: all -- every difference between these arms is in scope, so this "
            f"contrast is a total margin over {sorted(differing)} and not a controlled "
            "test of any one of them"
        )
    elif declared is None:
        warnings.append(
            f"no `dimension` declared, so nothing constrains what may differ; measured "
            f"differences: {sorted(differing) or 'none'}"
        )
    elif set(differing) - declared:
        undeclared = {name: differing[name] for name in sorted(set(differing) - declared)}
        return (
            f"not equalised for this contrast: '{arm}' and '{baseline}' differ on "
            f"{undeclared}, which is not among the declared dimension(s) "
            f"{sorted(declared)}. Declare it if the confound is intended, or contrast arms "
            "that differ only in the dimension under test."
        ), None, scope
    if scope["unequalised_arms"]:
        warnings.append(
            f"{scope['unequalised_arms']} are marked `equalised: false`, which is a "
            "property of the arm; admitted here because the pair differs only on the "
            "declared dimension (D12)"
        )

    level = entry.get("level")
    if level and str(level) not in (str(left["supervision"]), str(right["supervision"])):
        warnings.append(
            f"declared level '{level}' matches neither arm's supervision "
            f"('{left['supervision']}', '{right['supervision']}'); the arm metadata is "
            "authoritative for the gate"
        )
    if left["supervision"] != right["supervision"]:
        if not entry.get("cross_level"):
            return (
                f"supervision mismatch: '{arm}' uses '{left['supervision']}', "
                f"'{baseline}' uses '{right['supervision']}'. This gate is not overridable "
                "by a flag. If the asymmetry is intended, declare `cross_level: true` on "
                "this contrast in the config, which computes it, stamps it on every row "
                "and forces the report to render it."
            ), None, scope
        return None, (
            f"Supervision asymmetry (declared cross_level): '{arm}' is at level "
            f"'{left['supervision']}' while '{baseline}' is at '{right['supervision']}'. "
            "The arm at the higher level has been fitted on this component's annotated "
            "seed episodes -- ten of them in this corpus -- which the other has never "
            "seen, so this row measures a system that was given ten labelled examples, "
            "not a like-for-like configuration change. The cost of those ten examples "
            "belongs in any reading of the effect."
        ), scope
    if entry.get("cross_level"):
        warnings.append(
            f"`cross_level: true` is declared but '{arm}' and '{baseline}' are both at "
            f"supervision level '{left['supervision']}'; there is no asymmetry to stamp, "
            "so either the declaration or the arm metadata is wrong"
        )
    return None, None, scope


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--matcher", default="embedding",
        help="Score rows carrying this matcher name. Pass 'any' for fixed-label "
             "alignment, whose rows have no matcher because labels are supplied and "
             "spans are matched by returned index (D10).",
    )
    parser.add_argument(
        "--repeat", type=int, default=0,
        help="Which repeat index to aggregate (default 0). E1's extra repeats live in the "
             "same scores file and must not be pooled into the main aggregate.",
    )
    parser.add_argument("--baseline", default="baseline_upstream",
                        help="Baseline for arms with no cell (e.g. reference arms).")
    parser.add_argument("--contrast-mode", choices=("cell", "global"), default="cell",
                        help="cell: each arm vs the baseline in its own (profile, camera) "
                             "cell (C1/C2/C3). global: every arm vs --baseline (C4).")
    parser.add_argument("--baseline-arm", default="baseline_upstream",
                        help="Base arm name whose per-cell instance each arm is contrasted "
                             "against, e.g. baseline_upstream__wrap__wrist.")
    parser.add_argument("--arms-config", type=Path, default=HERE.parent / "configs" / "arms.yaml")
    parser.add_argument(
        "--contrasts", type=Path, default=None,
        help="YAML file holding the pre-registered contrast family (numerator/baseline "
             "pairs). Switches the contrast set from 'every arm vs one baseline' to the "
             "listed pairs, computed in ONE Holm family (D13). Required for the alignment "
             "study, where one arm is the numerator of several contrasts.",
    )
    parser.add_argument(
        "--family-metrics", nargs="*", default=None,
        help="Metrics that enter the Holm family in --contrasts mode. Everything else in "
             "--metrics is computed and marked `exploratory`, so it is reported but cannot "
             "be starred and does not enlarge the correction. Defaults to the config's "
             "`metrics:` key, then to the pre-registered alignment family.",
    )
    parser.add_argument(
        "--require-true", action="append", default=[], metavar="FIELD",
        help="Keep only score rows whose FIELD is true, and record the restriction in the "
             "output. Used to report a contrast twice: once over the whole eval "
             "population and once over the subpopulation where a treatment is uniform "
             "(alignment_plan.md §4.2 -- calibration only applies to episodes whose label "
             "tuple matches the fit).",
    )
    parser.add_argument(
        "--datasets-matching", action="append", default=[], metavar="SUBSTRING",
        help="Keep only components whose name contains one of these. With "
             "--datasets-not-matching, this is how the pre-registered strata are cut: the "
             "headline population is all 16 components, and the development-adjacent "
             "family is reported SEPARATELY rather than dropped (alignment_plan.md §3.2).",
    )
    parser.add_argument(
        "--datasets-not-matching", action="append", default=[], metavar="SUBSTRING",
        help="Drop components whose name contains one of these.",
    )
    parser.add_argument("--metrics", nargs="*", default=None)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--output-population", choices=("all_scored", "successful"),
                        default="all_scored", help="include scored failure penalties by default; "
                        "successful reports conditional output quality")
    parser.add_argument(
        "--allow-unequalised", action="store_true",
        help="Contrast arms marked `equalised: false`. Off by default: those arms see a "
             "different camera set, so the contrast measures a camera choice, not a mechanism.",
    )
    args = parser.parse_args()

    contrast_spec = load_contrast_config(args.contrasts) if args.contrasts else None
    if contrast_spec is not None and args.allow_unequalised:
        # The flag exists to widen a per-ARM gate. In contrast-list mode the gate is
        # already scoped per PAIR, so the only thing this could still do is hide the
        # refusals that scoping produces -- which is the override that withdrew C4.
        raise SystemExit(
            "--allow-unequalised is refused with --contrasts: equalisation is already "
            "scoped to the dimension each contrast declares. Declare the dimension in the "
            "contrast config instead of widening the gate from the command line."
        )
    if args.metrics is not None:
        metrics = list(args.metrics)
    elif contrast_spec is not None:
        # The config's own two lists, so the set measured here is the set the
        # pre-registration named. dict.fromkeys keeps order and drops duplicates.
        metrics = list(dict.fromkeys(
            (contrast_spec.get("metrics") or ALIGNMENT_FAMILY)
            + (contrast_spec.get("secondary_metrics") or ALIGNMENT_SECONDARY)
        ))
    else:
        metrics = HEADLINE + DESCRIPTIVE
    args.metrics = metrics
    alpha = contrast_spec["alpha"] if contrast_spec else 0.05
    if contrast_spec is not None:
        # Two independent declarations of polarity that must agree before anything is
        # computed. Preferring one silently would mean a typo in either place inverts
        # the sign of an error metric and stars its best arm as the worst (D14); this
        # way the disagreement is a loud failure naming both sides.
        for metric, direction in sorted(contrast_spec["metric_directions"].items()):
            if direction not in ("higher", "lower"):
                raise SystemExit(
                    f"{contrast_spec['path']} declares direction '{direction}' for "
                    f"'{metric}'; only 'higher' and 'lower' mean anything here."
                )
            registered = metric in LOWER_IS_BETTER
            if (direction == "lower") != registered:
                raise SystemExit(
                    f"polarity disagreement for '{metric}': {contrast_spec['path']} says "
                    f"'{direction}', while aggregate.py's LOWER_IS_BETTER "
                    f"{'contains' if registered else 'does not contain'} it. Both must agree "
                    "before any contrast is computed. Register the metric in "
                    "LOWER_IS_BETTER, or correct the direction in the config."
                )
    if contrast_spec is None:
        # Generation study: every computed test is in the family, as before.
        family_metrics = set(metrics)
    else:
        family_metrics = set(
            args.family_metrics or contrast_spec.get("metrics") or ALIGNMENT_FAMILY
        )
        missing_family = sorted(family_metrics - set(metrics))
        if missing_family:
            raise SystemExit(
                f"family metrics {missing_family} are not among --metrics {metrics}: the "
                "Holm family would silently be smaller than the pre-registered one."
            )
        implied = len(contrast_spec["contrasts"]) * len(family_metrics)
        declared_tests = contrast_spec.get("expected_tests")
        if declared_tests is not None and int(declared_tests) != implied:
            raise SystemExit(
                f"{contrast_spec['path']} declares expected_tests={declared_tests} but lists "
                f"{len(contrast_spec['contrasts'])} contrasts x {len(family_metrics)} family "
                f"metrics = {implied}. The config disagrees with itself about the size of the "
                "correction; fix it before any p-value is computed."
            )
    # Alignment rows carry no matcher (see load_rows); 'any' says so explicitly
    # rather than letting a default of 'embedding' silently select nothing.
    matcher = None if args.matcher.strip().lower() in ("any", "none", "") else args.matcher

    import yaml

    spec = yaml.safe_load(args.arms_config.read_text(encoding="utf-8"))
    arm_meta = {a["name"]: a for a in spec["arms"]}
    for name, declared in ((contrast_spec or {}).get("external_arms") or {}).items():
        existing = arm_meta.get(name)
        if existing is None:
            arm_meta[name] = {"name": name, **declared, "external_arm": True}
            continue
        clashes = {field for field in ("tool", "supervision")
                   if existing.get(field) != declared.get(field)}
        if clashes:
            raise SystemExit(
                f"'{name}' is declared in both {args.arms_config} and "
                f"{contrast_spec['path']} and they disagree on {sorted(clashes)}. The gate "
                "reads whichever loaded last, so a disagreement here decides a supervision "
                "verdict by accident."
            )

    rows = load_rows(args.scores, matcher, repeat=args.repeat)
    population = {"filter": list(args.require_true), "rows_before_filter": len(rows)}
    if args.require_true:
        rows = filter_population(rows, args.require_true)
        if not rows:
            raise SystemExit(
                f"no rows in {args.scores} satisfy {args.require_true}. A subpopulation "
                "that turns out to be empty is a finding about the corpus, not a reason to "
                "aggregate the whole population under a restricted label."
            )
    if args.datasets_matching:
        rows = [row for row in rows
                if any(s in str(row.get("dataset")) for s in args.datasets_matching)]
    if args.datasets_not_matching:
        rows = [row for row in rows
                if not any(s in str(row.get("dataset")) for s in args.datasets_not_matching)]
    population["datasets_matching"] = list(args.datasets_matching)
    population["datasets_not_matching"] = list(args.datasets_not_matching)
    population["rows_after_filter"] = len(rows)
    if not rows:
        available = sorted({
            int(json.loads(line).get("repeat", 0) or 0)
            for line in args.scores.read_text(encoding="utf-8").splitlines()
            if line.strip()
        })
        matchers = sorted({
            str(json.loads(line).get("matcher"))
            for line in args.scores.read_text(encoding="utf-8").splitlines()
            if line.strip()
        })
        raise SystemExit(
            f"no rows for matcher '{args.matcher}' at repeat {args.repeat} in {args.scores}. "
            f"Repeat indices present: {available}. Matchers present: {matchers}. E1's "
            "falsification repeats are numbered from 1, while the main sweep uses 0; pass "
            "--repeat to select one. Alignment rows have no matcher at all: pass "
            "--matcher any."
        )

    arms = sorted({r["arm"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})
    # A wholly failed arm can still have valid scored failure penalties and
    # belongs in the comparison. Only explicitly unscorable rows are absent.
    def included(row):
        return usable_score(row, args.output_population)

    scored_arms = {r["arm"] for r in rows if included(r)}

    failures: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "failed": 0})
    for row in rows:
        bucket = failures[row["arm"]]
        bucket["total"] += 1
        if not row.get("ok"):
            bucket["failed"] += 1

    report: dict[str, Any] = {
        "output_population": args.output_population,
        "per_arm_population": "descriptive: each arm's selected scored rows; use paired means for contrasts",
        "matcher": args.matcher,
        "baseline": args.baseline,
        "datasets": datasets,
        "arms": arms,
        "failure_rates": {
            arm: {**v, "rate": v["failed"] / v["total"] if v["total"] else 0.0}
            for arm, v in sorted(failures.items())
        },
        "per_arm": {},
        "per_dataset": {},
        "contrasts": {},
        "excluded_from_contrast": {},
    }
    if contrast_spec is not None:
        # Everything a reader needs to check that the family reported is the family
        # pre-registered: which file, which pairs, which metrics were corrected.
        report["mode"] = "alignment"
        # There is no single baseline in this mode: each contrast names its own, and
        # leaving the CLI default in place would let a reader attribute every row to
        # an arm that appears in only some of them.
        report["baseline"] = None
        report["metrics"] = list(args.metrics)
        report["contrast_config"] = contrast_spec["path"]
        report["contrast_family"] = contrast_spec.get("family")
        report["holm_family_metrics"] = sorted(family_metrics)
        report["alpha"] = alpha
        report["expected_tests"] = contrast_spec.get("expected_tests")
        report["contrast_options"] = {}
        report["population"] = population
        report["config_warnings"] = {}

    for arm in arms:
        report["per_arm"][arm] = {}
        for metric in args.metrics:
            by_dataset: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                if row["arm"] != arm or not included(row) or metric not in row:
                    continue
                value = row[metric]
                # boundary_distance_summary returns None when either side has no
                # internal boundary (a single-span prediction). float(None) is a
                # TypeError that would abort the whole aggregation.
                if value is None:
                    continue
                by_dataset[row["dataset"]].append(float(value))
            if not by_dataset:
                continue
            interval = cluster_bootstrap_mean(by_dataset, n_boot=args.n_boot, seed=17)
            report["per_arm"][arm][metric] = interval.as_dict()

    for dataset in datasets:
        report["per_dataset"][dataset] = {}
        for arm in arms:
            values: dict[str, list[float]] = {}
            for metric in args.metrics:
                collected = [
                    float(r[metric]) for r in rows
                    if r["dataset"] == dataset and r["arm"] == arm and included(r)
                    and r.get(metric) is not None
                ]
                if collected:
                    values[metric] = sum(collected) / len(collected)
            if values:
                report["per_dataset"][dataset][arm] = values

    per_episode: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        if not included(row):
            continue
        key = f"{row['dataset']}/{row['episode']}"
        for metric in args.metrics:
            if metric in row and row[metric] is not None:
                per_episode[metric][key][row["arm"]] = float(row[metric])

    # In the factorial design every arm belongs to a (profile, camera) cell, and
    # the only legitimate contrast is against the baseline in its OWN cell. A
    # global baseline would compare, say, a wrist-camera arm against a
    # left-camera baseline and report the camera difference as a tool effect.
    def cell_baseline(arm: str) -> str:
        # `global` is required for C4, where every VLM arm is contrasted against
        # the one model-free prior; cell mapping there would silently redirect
        # each arm to its tool baseline and the primary claim would never appear.
        if args.contrast_mode == "global":
            return args.baseline
        meta = arm_meta.get(arm) or {}
        profile, camera = meta.get("profile"), meta.get("camera")
        if not profile or not camera:
            return args.baseline
        return f"{args.baseline_arm}__{profile}__{camera}"

    # Retain the historical upstream comparisons and add the registered C3
    # intervention (realign minus plain video) in every available cell.
    # Each comparison carries its contrast-config entry, or None when the pair was
    # derived from the per-arm baseline rule rather than pre-registered as a pair.
    comparisons = [(arm, arm, cell_baseline(arm),
                    "upstream" if args.contrast_mode == "cell" else "C4", None)
                   for arm in arms if arm != cell_baseline(arm) and arm != args.baseline]
    if args.contrast_mode == "cell":
        for arm in arms:
            if arm.startswith("align_video_realign__"):
                baseline = arm.replace("align_video_realign__", "align_video__", 1)
                comparisons.append((arm + " [C3]", arm, baseline, "C3", None))
    if contrast_spec is not None:
        # The pre-registered list replaces the derived one outright. Mixing them
        # would put unregistered comparisons into the family that Holm corrects.
        comparisons = [
            (entry["key"], entry["numerator"], entry["baseline"],
             str(entry.get("claim") or ""), entry)
            for entry in contrast_spec["contrasts"]
        ]

    raw_p: dict[str, float] = {}
    for key, arm, this_baseline, claim, entry in comparisons:
        if entry is None:
            reason, asymmetry = contrast_policy(
                arm, this_baseline, arm_meta,
                allow_unequalised=args.allow_unequalised, allow_reference=True,
            )
            scope = None
        else:
            reason, asymmetry, scope = alignment_contrast_policy(entry, arm_meta)
            if scope.get("warnings"):
                report["config_warnings"][key] = scope["warnings"]
        if reason:
            report["excluded_from_contrast"][key] = reason
            continue
        if this_baseline not in arms:
            report["excluded_from_contrast"][key] = (
                f"comparison baseline '{this_baseline}' is absent from the results")
            continue
        if this_baseline not in scored_arms or arm not in scored_arms:
            report["excluded_from_contrast"][key] = (
                f"'{arm}' or '{this_baseline}' scored zero episodes; no contrast is possible")
            continue
        report["contrasts"][key] = {}
        report.setdefault("contrast_baselines", {})[key] = this_baseline
        report.setdefault("contrast_arms", {})[key] = arm
        report.setdefault("contrast_claims", {})[key] = claim
        if asymmetry:
            report.setdefault("supervision_asymmetry", {})[key] = asymmetry
        if entry is not None:
            report.setdefault("contrast_levels", {})[key] = entry.get("level")
            # Carried so the report can honour what the pre-registration asked of each
            # contrast -- which ones must show the calibration-applied rate, which ones
            # are additionally reported on the uniform-treatment subpopulation -- rather
            # than applying one blanket rule to all eight.
            report["contrast_options"][key] = {
                field: entry[field] for field in
                ("label", "report_calibration_applied", "restrict_to_modal_subpopulation")
                if field in entry
            }
            report.setdefault("contrast_scope", {})[key] = scope
            report.setdefault("contrast_ids", {})[key] = entry["id"]
        # Stamped on every metric row below, not just on the contrast, so a report
        # that renders rows without reading the contrast header still carries the
        # caveat. Section 4.1: a cross-level contrast is computed only on condition
        # that its asymmetry travels with every number it produces.
        # Contrast-list mode only, so the generation study's artifacts are
        # byte-identical to what it produced before this mode existed; there the
        # caveat is already rendered from the top-level map by make_report.py.
        stamp: dict[str, Any] = {}
        if entry is not None:
            if asymmetry:
                stamp["supervision_asymmetry"] = asymmetry
                stamp["cross_level"] = True
            if scope:
                stamp["equalisation_scope"] = scope["differs_on"]
                stamp["equalisation_declared"] = scope["declared"]
        for metric in args.metrics:
            deltas = bootstrap_paired_difference_by_dataset(per_episode[metric], arm, this_baseline)
            if not deltas:
                # Never skip in silence: an absent row and a null result look
                # identical in a table, and the absent one is the dangerous case.
                report["contrasts"][key][metric] = {
                    "unavailable": True,
                    "reason": (
                        f"no episode was scored for BOTH '{arm}' and '{this_baseline}' on "
                        f"'{metric}', so no paired difference exists"
                    ),
                    **stamp,
                }
                continue
            paired_arm: dict[str, list[float]] = defaultdict(list)
            paired_baseline: dict[str, list[float]] = defaultdict(list)
            for episode_key, values in sorted(per_episode[metric].items()):
                if arm in values and this_baseline in values:
                    dataset = episode_key.split("/", 1)[0]
                    paired_arm[dataset].append(values[arm])
                    paired_baseline[dataset].append(values[this_baseline])
            def paired_mean(values):
                return sum(sum(v) / len(v) for v in values.values()) / len(values)
            interval, p = paired_cluster_bootstrap(deltas, n_boot=args.n_boot, seed=23)
            # Orient "win" by the metric's polarity, not by the sign of the raw
            # difference: for an error metric a negative delta is the improvement.
            better = -1.0 if metric in LOWER_IS_BETTER else 1.0
            wins = sum(1 for v in deltas.values() if better * (sum(v) / len(v)) > 0)
            losses = sum(1 for v in deltas.values() if better * (sum(v) / len(v)) < 0)
            report["contrasts"][key][metric] = {
                **interval.as_dict(),
                "paired_arm_mean": paired_mean(paired_arm),
                "paired_baseline_mean": paired_mean(paired_baseline),
                "paired_datasets": sorted(deltas),
                "p_bootstrap": p,
                "lower_is_better": metric in LOWER_IS_BETTER,
                "improves_baseline": better * interval.point > 0,
                "dataset_wins": wins,
                "dataset_losses": losses,
                "p_sign_test": sign_test(wins, losses),
                "n_paired_episodes": sum(len(v) for v in deltas.values()),
                **stamp,
            }
            if metric in family_metrics:
                raw_p[f"{key}::{metric}"] = p
            else:
                # Outside the pre-registered family: computed, reported, never
                # starred, and excluded from the correction so it cannot dilute it.
                report["contrasts"][key][metric]["exploratory"] = True

    # An arm whose every metric was unavailable was never really contrasted;
    # leaving it in `contrasts` with an empty or all-unavailable body would read
    # as "compared, no effect found".
    for arm in list(report["contrasts"]):
        entries = report["contrasts"][arm]
        if not entries or all(e.get("unavailable") for e in entries.values()):
            report["excluded_from_contrast"][arm] = (
                "no metric had episodes scored for both this arm and its baseline; "
                "nothing was compared"
            )
            del report["contrasts"][arm]

    # ONE Holm pass over every test computed in this invocation. In alignment mode
    # that is the whole pre-registered family -- 8 contrasts x 3 metrics -- rather
    # than six per-baseline families of three or four, which would each be
    # corrected against a divisor four to eight times too small (D13).
    report["holm_family_size"] = len(raw_p)
    if raw_p:
        adjusted = holm_correction(raw_p)
        for key, value in adjusted.items():
            arm, metric = key.split("::", 1)
            report["contrasts"][arm][metric]["p_holm"] = value
    family_shortfall = False
    if contrast_spec is not None:
        expected = len(contrast_spec["contrasts"]) * len(family_metrics)
        if len(raw_p) != expected:
            family_shortfall = True
            # The aggregate is still written, because a refused or unpairable contrast
            # is itself a result and has to be inspectable. But the exit status is
            # non-zero: a family that shrank was corrected more weakly than the
            # pre-registration promised, and a pipeline must not sail past that.
            report["holm_family_note"] = (
                f"{len(raw_p)} of the {expected} pre-registered tests "
                f"({len(contrast_spec['contrasts'])} contrasts x {len(family_metrics)} "
                "metrics) were computed; the rest were refused or had no paired episodes. "
                "See excluded_from_contrast. Holm corrected the tests that exist, so the "
                "surviving p-values are LESS conservative than the pre-registered family "
                "would have made them."
            )
            print(f"\n[holm] {report['holm_family_note']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")

    if contrast_spec is None:
        print(f"\nmatcher={args.matcher}  baseline={args.baseline}  datasets={len(datasets)}")
    else:
        print(f"\nalignment family={contrast_spec.get('family')}  "
              f"contrasts={len(contrast_spec['contrasts'])}  datasets={len(datasets)}  "
              f"population={args.require_true or 'all scored episodes'}")
    print(f"\n=== descriptive per-arm ({args.output_population}; use paired contrasts for differences) ===")
    head = args.metrics[:5]
    print(f"{'arm':34s} " + " ".join(f"{m[:15]:>15s}" for m in head))
    for arm in arms:
        cells = []
        for metric in head:
            entry = report["per_arm"].get(arm, {}).get(metric)
            cells.append(f"{entry['point']:15.4f}" if entry else f"{'-':>15s}")
        print(f"{arm:34s} " + " ".join(cells))

    if contrast_spec is None:
        print("\n=== paired contrasts (including registered C3: realign minus video) ===")
    else:
        print(f"\n=== pre-registered contrasts (Holm family of {report['holm_family_size']}"
              f" over {sorted(family_metrics)}) ===")
    # Claim metrics are starred; everything else is marked '+' as descriptive.
    claim_metrics = family_metrics if contrast_spec is not None else set(HEADLINE)
    for arm, metrics in report["contrasts"].items():
        print(f"\n  {arm}   [vs {report.get('contrast_baselines',{}).get(arm, args.baseline)}]")
        asymmetry = report.get("supervision_asymmetry", {}).get(arm)
        if asymmetry and contrast_spec is not None:
            print(f"     ! {asymmetry}")
        scope = report.get("contrast_scope", {}).get(arm)
        if scope:
            print(f"       equalisation: differs on {scope['differs_on'] or 'nothing'}; "
                  f"declared under test: {scope['declared']}")
        for warning in report.get("config_warnings", {}).get(arm, []):
            print(f"       warning: {warning}")
        for metric, entry in metrics.items():
            if entry.get("unavailable"):
                print(f"     {metric:24s} UNAVAILABLE - {entry['reason']}")
                continue
            significant = entry.get("p_holm", 1.0) < alpha
            star = ("*" if entry["improves_baseline"] else "!") if significant else " "
            if metric not in claim_metrics:
                star = star.replace("*", "+")  # descriptive: not a claim
            direction = " (lower better)" if entry["lower_is_better"] else ""
            print(
                f"    {star}{metric:24s}{direction:15s} {entry['point']:+.4f} "
                f"[{entry['ci_low']:+.4f},{entry['ci_high']:+.4f}] "
                f"p={entry['p_bootstrap']:.4f} holm={entry.get('p_holm', float('nan')):.4f} "
                f"wins={entry['dataset_wins']}/{entry['dataset_wins'] + entry['dataset_losses']} "
                f"means={entry['paired_arm_mean']:.4f}/{entry['paired_baseline_mean']:.4f} "
                f"n={entry['n_paired_episodes']} K={entry['n_clusters']}"
            )
    if report["excluded_from_contrast"]:
        print("\n=== excluded from contrast ===")
        for arm, reason in report["excluded_from_contrast"].items():
            print(f"  {arm}: {reason}")
    print(f"\nwrote {args.out}")
    if family_shortfall:
        print(
            f"FAILED: {len(raw_p)} of the {expected} pre-registered tests were computed. "
            "The aggregate was written for inspection, but the family is not the one that "
            "was registered and its Holm correction is weaker than promised.",
            file=sys.stderr,
        )
        # A DISTINCT exit code, not a generic 1. A shortfall is a specific, expected
        # condition -- it is what a deliberately partial run (the model-free floors
        # alone, with no VLM arm to be the numerator of contrasts 7 and 8) produces,
        # and it is what a real run producing a broken family also produces. A caller
        # that wants to tolerate the first must not thereby tolerate a crash, a bad
        # config or an unreadable scores file, all of which exit 1. Nothing here is
        # relaxed: the message, the non-zero status and `holm_family_note` in the
        # written JSON are unchanged, and the report generator still refuses to render
        # a family this size without its own explicit --allow-incomplete.
        return EXIT_FAMILY_SHORTFALL
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
