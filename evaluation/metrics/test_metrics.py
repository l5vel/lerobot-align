"""Unit tests for the evaluation metric suite.

These test the properties the evaluation *relies on*, not merely that the code
runs. Each test corresponds to a way a metric could silently mislead:
over-segmentation going unpunished, order-crossing matches, missing segments
being ignored, error metrics having the wrong polarity, and the semantic
matcher confusing antonyms.

Run: .venv/bin/python -m pytest evaluation/metrics/test_metrics.py -q
"""

from __future__ import annotations

import math

import pytest

from .consistency import consistency_gap, script_predictability, sequence_diversity
from .joint import edit_score, f1_at_iou, mof
from .segmentation import (
    boundary_score,
    matched_iou,
    segmentation_counts,
    segmentation_covering,
    validate_segmentation,
)
from .semantic import (
    ExactMatcher,
    TokenF1Matcher,
    contrast_conflict,
    label_agreement,
    normalise_label,
)
from .stats import cluster_bootstrap_mean, holm_correction, paired_cluster_bootstrap, sign_test


def spans(*triples):
    return [{"start": s, "end": e, "text": t} for s, e, t in triples]


GT = spans((0, 10, "open door"), (10, 20, "pick can"), (20, 30, "close door"))


# --------------------------------------------------------------- segmentation

def test_perfect_prediction_scores_one():
    assert matched_iou(GT, GT)["macro_iou"] == pytest.approx(1.0)
    assert segmentation_covering(GT, GT) == pytest.approx(1.0)
    assert boundary_score(GT, GT, tolerance=0.5).f1 == pytest.approx(1.0)


def test_over_segmentation_is_punished():
    """Ten segments where there are three must lose precision everywhere.

    This is the failure mode a naive frame-accuracy metric misses entirely.
    """
    over = [{"start": i * 3, "end": i * 3 + 3, "text": "open door"} for i in range(10)]
    assert boundary_score(over, GT, tolerance=1.0).precision < 0.3
    assert segmentation_covering(GT, over) < 0.4
    assert edit_score(over, GT, ExactMatcher()) == pytest.approx(1 / 3)
    assert segmentation_counts(over, GT)["log_ratio"] > 1.0


def test_under_segmentation_is_punished():
    single = spans((0, 30, "open door"))
    assert boundary_score(single, GT, tolerance=1.0).recall == pytest.approx(0.0)
    assert segmentation_covering(GT, single) < 0.4
    assert segmentation_counts(single, GT)["log_ratio"] < -1.0


def test_boundary_matching_preserves_order():
    """An order-crossing match must not be counted.

    Boundaries are ordered in time; a matcher that pairs a late prediction with
    an early reference would report agreement that does not exist. Predictions
    in reversed order should therefore match at most one boundary.
    """
    reversed_pred = spans((0, 20, "a"), (20, 21, "b"), (21, 30, "c"))
    score = boundary_score(reversed_pred, GT, tolerance=0.5)
    assert score.n_matched <= 1


def test_missing_segment_counts_as_zero_not_skipped():
    """Dropping a segment must lower the score, never leave it untouched."""
    partial = spans((0, 10, "open door"), (10, 30, "pick can"))
    result = matched_iou(partial, GT)
    assert len(result["per_reference_iou"]) == 3
    assert min(result["per_reference_iou"]) < 1.0
    assert result["macro_iou"] < matched_iou(GT, GT)["macro_iou"]


def test_covering_is_asymmetric():
    """The two directions must differ, or the metric is not doing its job."""
    over = [{"start": i * 3, "end": i * 3 + 3, "text": "x"} for i in range(10)]
    assert segmentation_covering(GT, over) != pytest.approx(segmentation_covering(over, GT))


def test_validate_rejects_overlap_and_disorder():
    with pytest.raises(ValueError):
        validate_segmentation(spans((0, 20, "a"), (10, 30, "b")))
    with pytest.raises(ValueError):
        validate_segmentation(spans((10, 5, "a")))
    with pytest.raises(ValueError):
        validate_segmentation([])


# -------------------------------------------------------------------- semantic

def test_normalisation_ignores_formatting_only():
    assert normalise_label("Pick up the RED can.") == normalise_label("pick up red can")


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("open the fridge door", "close the fridge door"),
        ("put the cup on the left shelf", "put the cup on the right shelf"),
        ("push the door", "pull the door"),
        ("put the block in the box", "put the block on the box"),
    ],
)
def test_contrast_guard_blocks_meaning_flips(left, right):
    """Sentence encoders score these near 1.0; the guard is what saves us."""
    assert contrast_conflict(left, right)


def test_contrast_guard_allows_unequal_specificity():
    """A vaguer label is not a contradiction and must not be blocked."""
    assert not contrast_conflict("put the cup down", "put the cup on the left shelf")
    assert not contrast_conflict("open the door", "open the fridge door")


def test_token_matcher_rejects_antonyms_at_default_threshold():
    matcher = TokenF1Matcher()
    assert not matcher.equivalent("open the fridge door", "close the fridge door")


def test_label_agreement_uses_multiset_not_set():
    """Repeated labels are real in this corpus; collapsing them hides errors."""
    matcher = ExactMatcher()
    reference = ["wipe", "wipe", "wipe"]
    result = label_agreement(["wipe"], reference, matcher)
    assert result["label_recall"] == pytest.approx(1 / 3)
    assert result["missed"] == pytest.approx(2.0)


# ----------------------------------------------------------------------- joint

def test_f1_at_iou_collapses_identical_label_runs():
    """MS-TCN merges adjacent identical labels before segmental scoring."""
    split = spans((0, 3, "open door"), (3, 6, "open door"), (6, 10, "open door"))
    result = f1_at_iou(split, GT[:1], ExactMatcher(), threshold=0.25)
    assert result["tp@25"] == 1.0
    assert result["precision@25"] == 1.0


def test_mof_ignores_wording_when_matcher_says_equivalent():
    aliased = spans((0, 10, "OPEN DOOR."), (10, 20, "pick can"), (20, 30, "close door"))
    assert mof(aliased, GT, ExactMatcher())["mof"] == pytest.approx(1.0)


def test_edit_score_is_time_free():
    """Shifting every boundary must not change the sequence score."""
    shifted = spans((0, 5, "open door"), (5, 25, "pick can"), (25, 30, "close door"))
    assert edit_score(shifted, GT, ExactMatcher()) == pytest.approx(1.0)


# ----------------------------------------------------------------- consistency

def test_sequence_diversity_detects_unstable_wording():
    stable = [GT for _ in range(5)]
    unstable = [spans((0, 10, f"open door {i}"), (10, 30, "pick can")) for i in range(5)]
    assert sequence_diversity(stable)["n_distinct_sequences"] == 1
    assert sequence_diversity(unstable)["n_distinct_sequences"] == 5
    assert consistency_gap(unstable, stable)["sequence_diversity_gap"] > 0


def test_consistency_gap_is_signed_so_collapse_is_not_a_win():
    """A tool emitting one constant label is over-consistent, not perfect."""
    human = [spans((0, 10, f"step {i}"), (10, 30, "pick can")) for i in range(5)]
    collapsed = [spans((0, 10, "do thing"), (10, 30, "do thing")) for _ in range(5)]
    assert consistency_gap(collapsed, human)["sequence_diversity_gap"] < 0


def test_script_predictability_recovers_modal_boundary():
    seed = [spans((0, 15, "a"), (15, 60, "b")) for _ in range(4)]
    result = script_predictability(seed)
    assert result["modal_fraction"] == pytest.approx(1.0)
    assert result["mean_relative_boundaries"][0] == pytest.approx(0.25)


# ----------------------------------------------------------------------- stats

def test_cluster_bootstrap_widens_with_between_cluster_variance():
    """Two corpora with the same mean but different clustering must not get
    the same interval; ignoring clustering is what produces false precision."""
    tight = {f"d{i}": [0.5] * 40 for i in range(10)}
    spread = {f"d{i}": [0.5 + (0.4 if i < 5 else -0.4)] * 40 for i in range(10)}
    a = cluster_bootstrap_mean(tight, n_boot=500, seed=0)
    b = cluster_bootstrap_mean(spread, n_boot=500, seed=0)
    assert (b.high - b.low) > (a.high - a.low)


def test_paired_bootstrap_finds_no_effect_when_there_is_none():
    null = {f"d{i}": [0.0] * 40 for i in range(10)}
    _, p = paired_cluster_bootstrap(null, n_boot=500, seed=0)
    assert p > 0.05


def test_bootstrap_p_is_floored_at_resolution():
    """A bootstrap cannot resolve p below 1/n_boot; quoting less is fabricated."""
    strong = {f"d{i}": [1.0] * 40 for i in range(10)}
    _, p = paired_cluster_bootstrap(strong, n_boot=500, seed=0)
    assert p >= 1 / 500


def test_sign_test_and_holm():
    assert sign_test(10, 0) < 0.01
    assert sign_test(5, 5) == pytest.approx(1.0)
    adjusted = holm_correction({"a": 0.001, "b": 0.02, "c": 0.3})
    assert adjusted["a"] < adjusted["b"] < adjusted["c"]
    assert all(v >= raw for v, raw in zip(adjusted.values(), [0.001, 0.02, 0.3], strict=True))


def test_missing_arm_rows_are_dropped_not_imputed():
    from .stats import bootstrap_paired_difference_by_dataset

    per_episode = {"d/0": {"a": 1.0, "b": 0.5}, "d/1": {"a": 1.0}}
    deltas = bootstrap_paired_difference_by_dataset(per_episode, "a", "b")
    assert deltas == {"d": [0.5]}


def test_log_ratio_is_symmetric_for_double_and_half():
    """Averaging raw count ratios would make over-segmentation look worse."""
    double = segmentation_counts(GT * 2, GT)["log_ratio"]
    half = segmentation_counts(GT[:1], GT[:2])["log_ratio"]
    assert math.isclose(double, -half, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Regression tests for defects found by the round-1 adversarial review.
# Each of these returned a plausible-looking but wrong number before the fix,
# which is exactly the class of bug that survives a green test suite.
# ---------------------------------------------------------------------------

def test_mof_does_not_extrapolate_past_the_prediction():
    """A prediction lying entirely outside the episode must score 0, not 1.

    The original `label_at` fell back to the last span's label for any
    uncovered instant, so a segmentation covering none of the episode was
    credited with all of it.
    """
    reference = spans((0, 30, "a"))
    outside = spans((100, 130, "a"))
    assert mof(outside, reference, ExactMatcher())["mof"] == pytest.approx(0.0)


def test_contrast_guard_allows_inflections_and_synonyms():
    """The guard must fire on opposite meanings, not on different wordings.

    The first version held each contrast as a flat set, so "open" vs "opening"
    and "in" vs "into" counted as opposite sides and hard-zeroed similarity for
    ordinary paraphrases of the corpus's most common labels.
    """
    assert not contrast_conflict("open the fridge door", "opening the fridge door")
    assert not contrast_conflict("put the block into the box", "put the block in the box")
    assert not contrast_conflict("close the door", "closing the door")
    assert contrast_conflict("open the drawer", "shut the drawer")
    assert contrast_conflict("put the block in the box", "put the block on the box")


def test_nan_delta_does_not_report_maximal_significance():
    """A NaN point estimate must yield NaN, not the smallest expressible p.

    NaN compares False against both sign tests, so every resample looked
    non-crossing and the function returned p = 1/n_boot -- maximal significance
    for a broken input.
    """
    deltas = {"a": [0.0] * 10, "b": [float("nan")] + [0.0] * 9}
    _, p = paired_cluster_bootstrap(deltas, n_boot=200, seed=0)
    assert math.isnan(p)


def test_identical_arms_are_not_significant():
    deltas = {f"c{i}": [0.0] * 10 for i in range(10)}
    _, p = paired_cluster_bootstrap(deltas, n_boot=200, seed=0)
    assert p == pytest.approx(1.0)


def test_signed_count_error_is_not_declared_lower_is_better():
    """`count_error` is signed, so treating it as an error magnitude would
    report systematic under-segmentation as an improvement."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "aggregate.py"
    spec = importlib.util.spec_from_file_location("agg", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert "abs_count_error" in module.LOWER_IS_BETTER
    assert "count_error" not in module.LOWER_IS_BETTER


def test_headline_metrics_exclude_the_ceilinged_ones():
    """The model-free script prior scores >0.9 on label_f1/edit_score/f1@25 on
    this corpus, so no claim may rest on them."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "aggregate.py"
    spec = importlib.util.spec_from_file_location("agg2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for ceilinged in ("label_f1", "edit_score", "f1@25", "f1@50", "mof"):
        assert ceilinged not in module.HEADLINE, f"{ceilinged} has no headroom on this corpus"
    assert "boundary_f1@0p5" in module.HEADLINE


# ---------------------------------------------------------------------------
# Round-2: C1/C4 gate plumbing.
# ---------------------------------------------------------------------------

def test_noise_band_matches_episodes_by_id_not_position():
    """Repeats that cover different episode sets must not be mispaired.

    The first version zipped per-repeat lists positionally, so a single missing
    episode in one repeat silently shifted every later comparison onto the
    wrong episode and produced a meaningless margin.
    """
    from .stats import decoding_noise_band

    # Repeat 2 is missing episode 1. Only episodes 2 and 3 are shared:
    # |0.20-0.22| = 0.02 and |0.30-0.33| = 0.03, so the mean is 0.025.
    values = {"c0": {1: {1: 0.10, 2: 0.20, 3: 0.30}, 2: {2: 0.22, 3: 0.33}}}
    band = decoding_noise_band(values, n_boot=200, seed=0)
    assert band.point == pytest.approx(0.025, abs=1e-9)


def test_noise_band_needs_two_repeats():
    from .stats import decoding_noise_band

    band = decoding_noise_band({"c0": {1: {0: 0.5}}}, n_boot=100, seed=0)
    assert math.isnan(band.point)


def test_equivalence_distinguishes_inconclusive_from_equivalent():
    """An imprecise study must not be certified as equivalent.

    This is the whole point of using TOST rather than failure-to-reject: a
    wide interval means 'we cannot tell', not 'they agree'.
    """
    from .stats import equivalence_test

    tight = {f"c{i}": [0.001] * 40 for i in range(16)}
    assert equivalence_test(tight, 0.05, n_boot=400, seed=0)["verdict"] == "EQUIVALENT"

    wide = {f"c{i}": [0.001 + (0.4 if i % 2 else -0.4)] * 40 for i in range(16)}
    assert equivalence_test(wide, 0.05, n_boot=400, seed=0)["verdict"] == "INCONCLUSIVE"

    large = {f"c{i}": [0.20] * 40 for i in range(16)}
    assert equivalence_test(large, 0.05, n_boot=400, seed=0)["verdict"] == "DIFFERENT"


def test_equivalence_rejects_a_nonpositive_margin():
    from .stats import equivalence_test

    assert equivalence_test({"c": [0.0]}, 0.0, n_boot=100)["verdict"] == "UNDEFINED"
    assert equivalence_test({"c": [0.0]}, float("nan"), n_boot=100)["verdict"] == "UNDEFINED"


def test_c4_is_computable_in_the_preregistered_direction():
    """`--baseline ref_script_prior` must not exclude every VLM arm.

    C4 is the study's primary claim. The first fix exempted reference ARMS from
    the supervision guard but never checked the BASELINE, so the direction the
    plan actually states produced an empty contrast table.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "aggregate.py"
    spec = importlib.util.spec_from_file_location("agg3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    metadata = {
        "prior": {"tool": "reference", "supervision": "seed_calibration"},
        "video": {"tool": "align", "supervision": "none"},
    }
    for arm, baseline in (("prior", "video"), ("video", "prior")):
        reason, warning = module.contrast_policy(arm, baseline, metadata, allow_reference=True)
        assert reason is None
        assert "Supervision asymmetry" in warning


def test_differential_attrition_cannot_certify_equivalence():
    """Dropping failed episodes must not manufacture an equivalence verdict.

    If one arm crashes on the hard episodes, only the easy ones survive into
    the paired comparison and two very different arms look identical on the
    remainder. The paired deltas in that situation are genuinely tiny and sit
    well inside the margin, so the equivalence test alone cannot catch it --
    the attrition rates have to veto the verdict.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "c1_gate.py"
    source = path.read_text(encoding="utf-8")
    # The guard must exist, must be wired into the verdict, and must default to
    # blocking rather than warning.
    assert "attrition_blocks" in source
    assert "max_attrition_gap" in source
    assert 'result["verdict"] = "INCONCLUSIVE"' in source
    spec = importlib.util.spec_from_file_location("c1g", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "main")


def test_full_stage_requires_a_passing_c1_gate():
    """`run_all.sh full` spends hours of GPU; it must read the gate it claims to have."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_all.sh"
    ).read_text(encoding="utf-8")
    assert "require_c1_pass" in source
    stage = source[source.index("stage_full()"):]
    stage = stage[: stage.index("\n}")]
    assert "require_c1_pass" in stage, "stage_full does not check the C1 gate"


def test_c1_verdict_is_bound_to_the_configuration_it_certified():
    """A PASS must not survive a change to what it was a statement about.

    Editing one arm flag, checking out different tool source, or pointing at a
    different model all leave a stale PASS on disk that would otherwise
    authorise a sweep it never covered.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "gate_fingerprint.py"
    spec = importlib.util.spec_from_file_location("gfp", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    base = module.compute(
        model_id="m", matcher="embedding", threshold=0.91,
        baseline="baseline_upstream__wrap__wrist", arm="align_defaults__wrap__wrist", metrics=["macro_iou"],
    )
    # Every field that can change the meaning of the verdict must change it.
    for field, changed in (
        ("model_id", {"model_id": "other"}),
        ("matcher", {"matcher": "exact"}),
        ("threshold", {"threshold": 0.5}),
        ("arm", {"arm": "align_video__wrap__wrist"}),
        ("metrics", {"metrics": ["boundary_f1@1"]}),
    ):
        kwargs = {
            "model_id": "m", "matcher": "embedding", "threshold": 0.91,
            "baseline": "baseline_upstream__wrap__wrist", "arm": "align_defaults__wrap__wrist",
            "metrics": ["macro_iou"],
        }
        kwargs.update(changed)
        assert module.diff(base, module.compute(**kwargs)), f"{field} did not invalidate"

    # Source-tree hashes must be real content hashes, not placeholders.
    assert base["align_source"] not in ("absent", "")
    assert base["upstream_source"] not in ("absent", "")
    assert base["arms_certified"] not in ("absent", "")
    assert not module.diff(base, dict(base))


def test_full_stage_checks_verdict_freshness_not_just_the_word_pass():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_all.sh"
    ).read_text(encoding="utf-8")
    gate = source[source.index("require_c1_pass()"):]
    gate = gate[: gate.index("\nstage_full")]
    assert "gate_fingerprint.py" in gate, "stage_full accepts a stale PASS"
    assert "--check" in gate


def test_prediction_cache_keys_on_source_content_not_just_git_head():
    """An uncommitted edit must invalidate cached predictions.

    Keying on `git rev-parse HEAD` leaves the key unchanged across an
    uncommitted edit, so the cache serves predictions produced by different
    code -- and a later C1 verdict certifies them while recording the *new*
    source hash.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_arm.py"
    ).read_text(encoding="utf-8")
    block = source[source.index("provenance = {"):source.index("fingerprint = config_fingerprint")]
    assert "source_hashes()" in block, "cache key omits source content hash"


def test_c1_gate_refuses_predictions_from_other_source():
    """A verdict must not certify outputs it cannot attribute to current code."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "c1_gate.py"
    ).read_text(encoding="utf-8")
    assert "source_problems" in source
    assert "withheld_reason" in source
    # Unstamped rows must be treated as unattributable, not as a pass.
    assert "predate provenance" in source


def test_score_rows_carry_the_full_provenance_contract():
    """The carrier must forward every contracted field, not a hand-picked subset.

    Superseded the earlier version of this test, which checked for two literal
    key names -- and so would have kept passing while `effective_prompts` was
    dropped in transit.
    """
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "score_runs.py"
    ).read_text(encoding="utf-8")
    assert "PROVENANCE_KEYS" in source
    assert "for key in PROVENANCE_KEYS" in source
    # It must not re-list fields, which is how the drift happened.
    assert '("align_source", "upstream_source")' not in source


def test_upstream_fingerprint_is_derived_from_the_import_graph():
    """The hash must cover what the tools actually import, not a hand-list.

    Enumerating directories failed twice: the first version hashed only
    `annotations/steerable_pipeline` and missed the baseline's own entry point;
    the second added five directories by inspection and still missed 39
    imported modules, including `lerobot/__init__.py`, `lerobot_types.py` and
    the whole `processor` and `transforms` packages.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "gate_fingerprint.py"
    spec = importlib.util.spec_from_file_location("gfp2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "_discover_executed_modules")
    assert "lerobot.scripts.lerobot_annotate" in module.ENTRY_POINTS

    files = module._discover_executed_modules()
    relatives = {p.as_posix() for p in files}
    # The modules earlier versions missed must now be present.
    for expected in ("__init__.py", "lerobot_types.py", "processor/pipeline.py",
                     "scripts/lerobot_annotate.py"):
        assert any(r.endswith(expected) for r in relatives), f"{expected} not fingerprinted"
    # A real import graph for this pipeline is dozens of modules, not a handful.
    assert len(files) > 50, f"discovery returned only {len(files)} modules"


def test_upstream_fingerprint_refuses_a_thin_discovery():
    """Silent under-discovery must raise, not produce a weak-but-valid hash."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "gate_fingerprint.py"
    ).read_text(encoding="utf-8")
    assert "REQUIRED_IN_GRAPH" in source
    assert "cannot be trusted" in source
    assert "refusing to fingerprint a" in source


def test_upstream_fingerprint_is_stable_and_counts_modules():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "gate_fingerprint.py"
    spec = importlib.util.spec_from_file_location("gfp3", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    first = module.source_hashes()
    assert first["upstream_source"] not in ("absent", "")
    assert int(first["upstream_module_count"]) > 50
    assert module.source_hashes() == first


def test_fingerprint_covers_runtime_prompt_resources():
    """Prompts are read as data, so no import graph can reveal them.

    Hashing only modules left every prompt unfingerprinted, including
    `plan_subtasks.txt` -- the prompt that produces the segmentation being
    measured. Editing it would have changed every prediction while leaving the
    cache key and the C1 verdict untouched, which is the exact failure the
    provenance machinery exists to prevent.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "gate_fingerprint.py"
    spec = importlib.util.spec_from_file_location("gfp4", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    resources = {p.name for p in module._discover_runtime_resources()}
    for prompt in ("plan_subtasks.txt", "plan_subtask_describe.txt",
                   "plan_subtask_relabel.txt", "plan_memory.txt"):
        assert prompt in resources, f"{prompt} is not fingerprinted"

    hashes = module.source_hashes()
    assert int(hashes["upstream_resource_count"]) >= 10
    # The align side must cover its own prompts too, not only .py files.
    assert ".txt" in module.RESOURCE_SUFFIXES


def test_resource_discovery_refuses_to_miss_the_core_prompts():
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "gate_fingerprint.py"
    ).read_text(encoding="utf-8")
    assert "REQUIRED_RESOURCES" in source
    assert "plan_subtasks.txt" in source
    assert "refusing to fingerprint a" in source


def test_fingerprint_accounts_for_runtime_prompt_overrides(monkeypatch):
    """A packaged-file hash is insufficient when an env override replaces it.

    Both tools honour `LEROBOT_PROMPT_OVERRIDE_<name>`, which supersedes the
    packaged prompt. Hashing only the file means a verdict certified with an
    override active is indistinguishable from one certified without it -- same
    fingerprint, different prompts, different predictions.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "gate_fingerprint.py"
    spec = importlib.util.spec_from_file_location("gfp5", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    monkeypatch.delenv("LEROBOT_PROMPT_OVERRIDE_plan_subtasks", raising=False)
    clean_digest, clean_names = module.effective_prompt_digest()
    assert clean_names == []

    monkeypatch.setenv("LEROBOT_PROMPT_OVERRIDE_plan_subtasks", "a different prompt")
    dirty_digest, dirty_names = module.effective_prompt_digest()
    assert dirty_digest != clean_digest, "override did not change the fingerprint"
    assert dirty_names == ["plan_subtasks"]

    # Two different override texts must not collide either.
    monkeypatch.setenv("LEROBOT_PROMPT_OVERRIDE_plan_subtasks", "yet another prompt")
    assert module.effective_prompt_digest()[0] != dirty_digest


def test_preflight_refuses_while_prompt_overrides_are_active():
    """Comparing prompt FILES proves nothing when the text sent comes from env."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "preflight.py"
    ).read_text(encoding="utf-8")
    assert "check_prompt_overrides" in source
    assert "LEROBOT_PROMPT_OVERRIDE_" in source
    assert "ALLOW_PROMPT_OVERRIDES" in source
    # It must be wired into main(), not merely defined.
    body = source[source.index("def main()"):]
    assert "check_prompt_overrides(problems, notes)" in body


def test_provenance_contract_is_declared_once_and_honoured_everywhere():
    """Producer, carrier and verifier must agree on what a prediction carries.

    They had drifted: `run_arm` recorded `effective_prompts`, `score_runs`
    forwarded only two keys, and `c1_gate` skipped any field whose observed set
    was empty -- so prompt-override provenance was collected and then silently
    discarded before certification.
    """
    import importlib.util
    from pathlib import Path

    scripts = Path(__file__).resolve().parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location("gfp6", scripts / "gate_fingerprint.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "effective_prompts" in module.PROVENANCE_KEYS
    assert "prompt_overrides_active" in module.PROVENANCE_KEYS

    # Every key in the contract must be produced by source_hashes().
    state = module.source_hashes()
    for key in module.PROVENANCE_KEYS:
        assert key in state, f"{key} is in the contract but never produced"

    # The carrier and verifier must import the contract, not re-list it.
    carrier = (scripts / "score_runs.py").read_text(encoding="utf-8")
    verifier = (scripts / "c1_gate.py").read_text(encoding="utf-8")
    assert "PROVENANCE_KEYS" in carrier, "score_runs re-lists provenance keys"
    assert "PROVENANCE_KEYS" in verifier, "c1_gate re-lists provenance keys"


def test_missing_provenance_field_blocks_certification():
    """An absent field is unattributable, not 'nothing to compare'."""
    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "c1_gate.py"
    ).read_text(encoding="utf-8")
    assert "incomplete" in source
    assert "carry no value for this field" in source


def test_report_refuses_to_render_without_evidence(tmp_path):
    """Missing evidence must fail, not footnote.

    The first version printed "INCOMPLETE", wrote the file and exited 0, so a
    pipeline under `set -e` would continue and leave a document on disk that
    reads like a result while resting on nothing.
    """
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
    empty = tmp_path / "results"
    empty.mkdir()
    out = tmp_path / "report.md"

    result = subprocess.run(
        [sys.executable, str(script), "--results", str(empty), "--out", str(out)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, "rendered a report with no evidence"
    assert not out.exists(), "wrote a report file despite refusing"
    assert "REFUSED" in result.stderr

    # The opt-in draft must be written, stamped, and still non-zero.
    result = subprocess.run(
        [sys.executable, str(script), "--results", str(empty), "--out", str(out),
         "--allow-incomplete"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0, "an incomplete draft must not report success"
    assert out.exists()
    assert "NOT A RESULT" in out.read_text(encoding="utf-8")


def test_report_output_handling_is_symlink_safe_and_race_free(tmp_path):
    """Every mutation of --out must go through one verified descriptor.

    Three progressively subtler failures led here. The first version unlinked
    whatever `--out` named on the failure path. The second checked a marker by
    path and then acted by path, so the file could be swapped in between. Both
    followed symlinks, so `--out link.md` wrote through to the link's target.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
    empty = tmp_path / "results"
    empty.mkdir()

    def run(out: Path, *extra: str):
        return subprocess.run(
            [sys.executable, str(script), "--results", str(empty), "--out", str(out), *extra],
            capture_output=True, text=True,
        )

    # A foreign regular file survives both the refusal and the write path.
    foreign = tmp_path / "notes.md"
    foreign.write_text("# important notes\nvaluable content\n", encoding="utf-8")
    assert run(foreign).returncode != 0
    assert "valuable content" in foreign.read_text(encoding="utf-8")
    assert run(foreign, "--allow-incomplete").returncode != 0
    assert "valuable content" in foreign.read_text(encoding="utf-8"), "overwrote a foreign file"

    # A symlinked --out is refused outright, never followed to its target.
    target = tmp_path / "precious.md"
    target.write_text("# precious\n", encoding="utf-8")
    link = tmp_path / "link.md"
    os.symlink(target, link)
    result = run(link, "--allow-incomplete")
    assert result.returncode != 0
    assert "symbolic link" in result.stderr
    assert target.read_text(encoding="utf-8") == "# precious\n", "wrote through a symlink"

    # Our own stale report is cleared IN PLACE, not unlinked: unlink can only be
    # done by path, which would reintroduce the swap race.
    ours = tmp_path / "report.md"
    assert run(ours, "--allow-incomplete").returncode != 0
    assert ours.exists()
    assert run(ours).returncode != 0
    assert ours.exists(), "unlinked instead of clearing in place"
    body = ours.read_text(encoding="utf-8")
    assert "Superseded" in body and "NOT A RESULT" not in body




def test_report_writes_every_byte_and_leaves_no_stale_tail(tmp_path):
    """A short write would produce a truncated report that still looks valid.

    `os.write` may consume only part of its buffer and returns how much it
    took; ignoring that yields a file opening with the generation marker,
    reading as authoritative, and simply stopping partway through the results —
    worse than not writing at all, because nothing signals it is incomplete.

    Rewriting to a SHORTER report must also leave no tail from the longer one.
    """
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
    spec = importlib.util.spec_from_file_location("mr_write", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "report.md"
    for size in (100, 1 << 20, 4 << 20):
        payload = module.GENERATION_MARKER + "\n" + ("x" * size) + "\n"
        target = module.OutputTarget(out)
        target.write(payload)
        target.close()
        assert out.read_text(encoding="utf-8") == payload, f"truncated at {size} bytes"

    # Shrink: the file must equal the new content exactly, with no residue.
    short = module.GENERATION_MARKER + "\nshort\n"
    target = module.OutputTarget(out)
    assert target.is_ours
    target.write(short)
    target.close()
    assert out.read_text(encoding="utf-8") == short, "stale tail survived a shorter write"


def test_failed_write_leaves_the_destination_untouched(tmp_path, monkeypatch):
    """A write that fails midway must not leave a partial report behind.

    Looping until every byte is consumed still leaves a half-written file
    bearing the generation marker when the disk fills or errors partway. It
    reads as authoritative and stops mid-results, and verifying the size
    afterwards cannot undo it. The destination is therefore only ever touched
    by an atomic replace.
    """
    import importlib.util
    import os
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
    spec = importlib.util.spec_from_file_location("mr_atomic", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    out = tmp_path / "report.md"
    good = module.GENERATION_MARKER + "\n# complete previous report\nbody\n"
    target = module.OutputTarget(out)
    target.write(good)
    target.close()

    def flaky(fd, payload):
        os.write(fd, payload[: len(payload) // 2])
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(module.OutputTarget, "_write_all", staticmethod(flaky))
    target = module.OutputTarget(out)
    try:
        with pytest.raises(OSError):
            target.write(module.GENERATION_MARKER + "\n# new\n" + ("y" * 5000))
    finally:
        target.close()

    assert out.read_text(encoding="utf-8") == good, "destination was corrupted by a failed write"
    # The sidecar lock is expected to persist; unlinking it would race with
    # another holder. Only .tmp- residue indicates a leak.
    strays = [p.name for p in tmp_path.iterdir() if ".tmp-" in p.name]
    assert not strays, f"temporary files leaked: {strays}"


def test_publish_refuses_when_the_destination_changed_after_the_check(tmp_path):
    """Atomic publishing must not reintroduce a path-based overwrite.

    `os.replace` acts by path, so publishing that way clobbers whatever is
    there — undoing the descriptor-based ownership check and discarding the
    create-only guarantee of an `O_EXCL` open. Publishing is therefore split:
    `os.link` for a destination that did not exist (fails if one appeared), and
    an inode comparison against the held descriptor before `os.replace` for one
    that did.
    """
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
    spec = importlib.util.spec_from_file_location("mr_publish", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = module.GENERATION_MARKER + "\n# report\n"

    # A file appearing at a previously-absent destination must not be clobbered.
    appeared = tmp_path / "appeared.md"
    target = module.OutputTarget(appeared)
    appeared.write_text("# someone else's file\n", encoding="utf-8")
    try:
        with pytest.raises(OSError, match="appeared after"):
            target.write(body)
    finally:
        target.close()
    assert appeared.read_text(encoding="utf-8") == "# someone else's file\n"

    # Our own file, swapped after verification, must not be replaced.
    swapped = tmp_path / "swapped.md"
    first = module.OutputTarget(swapped)
    first.write(body)
    first.close()
    target = module.OutputTarget(swapped)
    assert target.is_ours
    swapped.unlink()
    swapped.write_text("# swapped in\n", encoding="utf-8")
    try:
        with pytest.raises(OSError, match="replaced after"):
            target.write(body)
    finally:
        target.close()
    assert swapped.read_text(encoding="utf-8") == "# swapped in\n"

    # The ordinary create-then-update path still works and leaves no residue.
    normal = tmp_path / "normal.md"
    for text in (body, body + "updated\n"):
        handle = module.OutputTarget(normal)
        handle.write(text)
        handle.close()
    assert normal.read_text(encoding="utf-8").endswith("updated\n")
    assert not [p for p in tmp_path.iterdir() if ".tmp-" in p.name], "temp residue"


def test_concurrent_runs_serialise_via_a_sidecar_lock(tmp_path):
    """Two runs writing one report must be ordered, including from nothing.

    Locking the report itself does not work, and both failures matter: the
    report is only lockable once it exists, so two runs starting with no report
    both proceed unlocked; and publishing swaps the destination inode, so a lock
    held on it ends up on an orphan while the next run locks the new file. The
    lock is therefore taken on a sidecar that is never replaced.
    """
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
    spec = importlib.util.spec_from_file_location("mr_lock", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    body = module.GENERATION_MARKER + "\n# report\n"

    # Serialised even when the report does not exist yet.
    absent = tmp_path / "absent.md"
    first = module.OutputTarget(absent)
    try:
        assert first.lock_fd is not None
        blocked = module.OutputTarget(absent)
        try:
            assert blocked.lock_fd is None, "second run proceeded with no report present"
            assert "locked by another run" in (blocked.refusal or "")
        finally:
            blocked.close()
    finally:
        first.close()

    # The lock must survive publishing, which replaces the destination inode.
    out = tmp_path / "report.md"
    holder = module.OutputTarget(out)
    try:
        holder.write(body)
        assert holder.lock_fd is not None, "publish released the lock"
        blocked = module.OutputTarget(out)
        try:
            assert blocked.lock_fd is None, "lock did not survive the inode swap"
        finally:
            blocked.close()
    finally:
        holder.close()

    # Released on close.
    after = module.OutputTarget(out)
    try:
        assert after.lock_fd is not None
    finally:
        after.close()




def test_sidecar_lock_is_symlink_safe_and_must_be_a_regular_file(tmp_path):
    """The lock inherits every hazard the destination checks guard against.

    Opened without `O_NOFOLLOW`, a symlinked `.<name>.lock` is followed: the
    process opens an unrelated file for writing, creates it if absent, and takes
    its lock — and two runs pointed at different targets both believe they hold
    the report, defeating serialisation exactly when it matters.
    """
    import importlib.util
    import os
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "make_report.py"
    spec = importlib.util.spec_from_file_location("mr_sidecar", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # A sidecar symlinked at an existing file must not be locked or touched.
    victim = tmp_path / "precious.txt"
    victim.write_text("do not touch\n", encoding="utf-8")
    out = tmp_path / "r.md"
    os.symlink(victim, tmp_path / ".r.md.lock")
    handle = module.OutputTarget(out)
    try:
        assert handle.lock_fd is None, "locked through a symlinked sidecar"
        assert "symbolic link" in (handle.refusal or "")
    finally:
        handle.close()
    assert victim.read_text(encoding="utf-8") == "do not touch\n"

    # A dangling symlink must not cause the target to be created.
    out2 = tmp_path / "s.md"
    target = tmp_path / "created_by_mistake.txt"
    os.symlink(target, tmp_path / ".s.md.lock")
    handle = module.OutputTarget(out2)
    try:
        assert handle.lock_fd is None
    finally:
        handle.close()
    assert not target.exists(), "followed a dangling symlink and created the target"

    # A fifo would block or misbehave under flock.
    out3 = tmp_path / "t.md"
    os.mkfifo(tmp_path / ".t.md.lock")
    handle = module.OutputTarget(out3)
    try:
        assert handle.lock_fd is None, "locked a non-regular file"
    finally:
        handle.close()

    # The ordinary case still works.
    handle = module.OutputTarget(tmp_path / "u.md")
    try:
        assert handle.lock_fd is not None
    finally:
        handle.close()


def test_installed_sibling_package_is_not_part_of_upstream_import_graph(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent / 'scripts/gate_fingerprint.py'
    spec = importlib.util.spec_from_file_location('installed_graph', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fake = {
        'lerobot/__init__.py': '',
        'lerobot/scripts/__init__.py': '',
        'lerobot/scripts/lerobot_annotate.py': 'import lerobot.datasets\nimport lerobot.annotations.steerable_pipeline\n',
        'lerobot/datasets/__init__.py': '',
        'lerobot/annotations/__init__.py': '',
        'lerobot/annotations/steerable_pipeline/__init__.py': '',
        'lerobot_align/__init__.py': '',
        'lerobot_align/cli.py': 'import lerobot_align.config\n',
        'lerobot_align/config.py': '',
    }
    for name, content in fake.items():
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
    monkeypatch.setenv('PYTHONPATH', str(tmp_path))
    paths = module._discover_executed_modules()
    assert len(paths) == 6
    assert not any(path.name == 'config.py' for path in paths)


def test_prompt_digest_uses_installed_package_without_source_checkout(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    import lerobot_align
    path = Path(__file__).resolve().parent.parent / 'scripts/gate_fingerprint.py'
    spec = importlib.util.spec_from_file_location('installed_prompts', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = tmp_path / 'site-packages/lerobot_align'
    (package / 'prompts').mkdir(parents=True)
    prompt = package / 'prompts/installed_only.txt'
    prompt.write_text('first prompt')
    monkeypatch.setattr(lerobot_align, '__file__', str(package / '__init__.py'))
    monkeypatch.setattr(module, 'REPO_DIR', tmp_path / 'no-checkout')
    before, _ = module.effective_prompt_digest()
    prompt.write_text('changed prompt')
    after, _ = module.effective_prompt_digest()
    assert before != after
