"""Diagnostic timing statistics must be independent of the time origin."""

import importlib.util
from pathlib import Path
import sys

import pytest

SCRIPT = Path(__file__).parents[1] / "evaluation/scripts/diagnose_corpus_calibration_gap.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("gap_diagnosis", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def spans(boundary, origin=0, scale=1):
    return [
        {"text": "first", "start": origin, "end": origin + scale * boundary},
        {"text": "second", "start": origin + scale * boundary, "end": origin + scale * 10},
    ]


@pytest.mark.parametrize("origin,scale", [(0, 1), (100, 1), (100, 2)])
def test_shift_and_scale_invariance_of_normalized_diagnostics(origin, scale):
    model = {
        "duration_weight": 1,
        "segment_fractions": [0.5, 0.5],
        "duration_scale": 0.1,
        "residual_scale": 0.1,
        "offsets": [0],
    }
    result = module.cohort_diagnostic(
        "test",
        "single",
        "task",
        model,
        [spans(5, origin, scale)],
        {1: spans(6, origin, scale), 2: spans(8, origin, scale)},
        {},
    )
    assert result["seed_prior_error_norm"] == pytest.approx(0.2)
    assert result["seed_prior_error_seconds"] == pytest.approx(2 * scale)
    assert result["eval_within_cohort_dispersion_norm"] == pytest.approx(0.1)
    assert result["seed_eval_target_median_shift_norm"] == pytest.approx(0.2)
