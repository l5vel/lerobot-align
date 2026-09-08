"""Protect population and occurrence identity in the offline experiment."""

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "evaluation/scripts/experiment_corpus_b_calibration.py"
SPEC = importlib.util.spec_from_file_location("corpus_b_experiment", SCRIPT)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def model():
    return {
        "offsets": [-0.5, 0.0],
        "segment_fractions": [0.3, 0.3, 0.4],
        "duration_weight": 0.0,
        "residual_scale": 0.1,
        "duration_scale": 0.1,
    }


def spans():
    return [
        {"text": "A", "start": 0.0, "end": 3.0},
        {"text": "B", "start": 3.0, "end": 6.0},
        {"text": "A", "start": 6.0, "end": 10.0},
    ]


def test_collision_preserves_repeated_label_occurrence_and_charges_drop():
    truth = spans()
    fitted = model()
    rows = experiment.evaluate_job(({}, ["A", "B", "A"], truth, truth, experiment.variants(fitted)))
    offsets = next(r for r in rows if r["variant"] == "offsets")
    assert offsets["scored"]
    assert [s["index"] for s in offsets["predictions"]] == [0, 1]
    assert offsets["placed_fraction"] == 2 / 3
    assert fitted == model()  # Variants and application must not alter a frozen fit.
    assert truth == spans()


def test_empty_outputs_are_scored_failures_in_every_variant():
    rows = experiment.evaluate_job(({}, ["A", "B", "A"], spans(), [], experiment.variants(model())))
    assert all(r["scored"] and r["b_hit@3"] == 0 for r in rows)
    assert all(r["placed_fraction"] == 0 for r in rows)


def test_partial_predictions_are_unchanged_by_positional_ablations():
    partial = spans()[:2]
    rows = experiment.evaluate_job(
        ({}, ["A", "B", "A"], spans(), partial, experiment.variants(model()))
    )
    assert all(r["predictions"] == rows[0]["predictions"] for r in rows)
    assert all(r["application"] == "incomplete_or_ambiguous" for r in rows[1:])


def test_summary_weights_tasks_equally_and_retains_failed_task():
    rows = []
    for camera in experiment.CAMERAS:
        for episode in range(10):
            successful = episode == 9
            meta = {
                "camera": camera,
                "task": "small" if successful else "large",
                "episode": episode,
            }
            rows.extend(
                experiment.evaluate_job(
                    (meta, ["A", "B", "A"], spans(), spans() if successful else [], None)
                )
            )
    summary = experiment.summarize(rows, 100, 1729)
    for camera in experiment.CAMERAS:
        baseline = summary["cameras"][camera]["arms"]["baseline"]
        assert baseline["task_macro"]["b_hit@3"] == 0.5
        assert baseline["episode_micro"]["b_hit@3"] == 0.1
        assert baseline["scored"] == 10
        assert baseline["empty_outputs"] == 9
    assert all(c["delta"] == 0 and c["holm_p"] == 1 for c in summary["contrasts"])
