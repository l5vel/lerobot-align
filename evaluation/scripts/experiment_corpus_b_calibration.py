#!/usr/bin/env python3
"""Paired offline ablations; see analysis/corpus_b_ablation_plan.md.

Inputs are read-only snapshots of run2 merged outputs and frozen fit manifests.
No parameter fitting or evaluation-based model selection is performed here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from datetime import datetime, UTC
import hashlib
import gzip
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import binomtest, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_alignment import resolve_label_indices, score_episode  # noqa: E402
from lerobot_align.diagnostics.fit_align_calibration import apply, fit  # noqa: E402

VARIANTS = ("baseline", "offsets", "duration", "full", "normalized")
METRICS = ("b_hit@3", "macro_temporal_iou", "placed_fraction")
CAMERAS = {"single": "align_video", "stacked": "align_video_stack"}
CONTRASTS = (
    ("offsets", "baseline"),
    ("duration", "baseline"),
    ("full", "baseline"),
    ("normalized", "full"),
    ("full", "offsets"),
    ("full", "duration"),
)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variants(model):
    out = {v: deepcopy(model) for v in VARIANTS if v != "baseline"}
    out["offsets"].pop("segment_fractions", None)
    out["duration"]["offsets"] = [0.0] * len(model["offsets"])
    fractions = model["segment_fractions"]
    assert sum(fractions) > 0
    out["normalized"]["segment_fractions"] = [v / sum(fractions) for v in fractions]
    return out


def evaluate_job(job):
    meta, labels, truth, predicted, models = job
    resolution = resolve_label_indices(labels, predicted)
    # Preserve occurrence identity if a calibrated collision later drops a label.
    if resolution.n_candidates == 1:
        predicted = [
            {**s, "index": i} if i is not None else dict(s)
            for s, i in zip(predicted, resolution.indices, strict=True)
        ]
    complete = resolution.n_candidates == 1 and resolution.indices == tuple(range(len(labels)))
    reason = (
        "no_fit"
        if models is None
        else "empty_prediction"
        if not predicted
        else "incomplete_or_ambiguous"
        if not complete
        else "applied"
    )
    rows = []
    for variant in VARIANTS:
        corrected = predicted
        if variant != "baseline" and reason == "applied":
            model = models[variant]
            corrected = apply(predicted, model, model.get("duration_weight", 1.0))
        scored = score_episode(labels, truth, corrected)
        rows.append(
            {
                **meta,
                "variant": variant,
                "application": ("baseline" if variant == "baseline" else reason),
                "predictions": corrected,
                "error": scored.error,
                **scored.row(),
            }
        )
    # Ambiguous original predictions are unchanged, so all variants share exclusions.
    assert len({r["scored"] for r in rows}) == 1
    return rows


def summarize(rows, n_boot, seed):
    result = {
        "cameras": {},
        "contrasts": [],
        "n_boot": n_boot,
        "bootstrap": "percentile task-cluster; equal task weights",
        "seed": seed,
    }
    rng = np.random.default_rng(seed)
    for camera in CAMERAS:
        selected = [r for r in rows if r["camera"] == camera]
        grouped = defaultdict(dict)
        for r in selected:
            if r["scored"]:
                grouped[(r["task"], r["episode"])][r["variant"]] = r
        assert all(set(v) == set(VARIANTS) for v in grouped.values())
        tasks = sorted({k[0] for k in grouped})
        task_values = {v: {} for v in VARIANTS}
        for variant in VARIANTS:
            for task in tasks:
                rs = [v[variant] for (t, _), v in grouped.items() if t == task]
                task_values[variant][task] = {
                    m: float(np.mean([r[m] for r in rs])) for m in METRICS
                }
        arm_summary = {}
        for variant in VARIANTS:
            rs = [r for r in selected if r["variant"] == variant]
            arm_summary[variant] = {
                "requested": len(rs),
                "scored": sum(r["scored"] for r in rs),
                "unscorable": sum(not r["scored"] for r in rs),
                "empty_outputs": sum(not r["predictions"] for r in rs),
                "application_counts": dict(Counter(r["application"] for r in rs)),
                "task_macro": {
                    m: float(np.mean([v[m] for v in task_values[variant].values()]))
                    for m in METRICS
                },
                "episode_micro": {
                    m: float(np.mean([r[m] for r in rs if r["scored"]])) for m in METRICS
                },
            }
        result["cameras"][camera] = {"arms": arm_summary, "per_task": task_values}
        # Same draws for the six paired contrasts in a camera arm.
        draws = rng.integers(0, len(tasks), size=(n_boot, len(tasks)))
        for arm, baseline in CONTRASTS:
            deltas = np.array(
                [
                    task_values[arm][t]["b_hit@3"] - task_values[baseline][t]["b_hit@3"]
                    for t in tasks
                ]
            )
            boot = deltas[draws].mean(axis=1)
            wins, losses = int(sum(deltas > 1e-12)), int(sum(deltas < -1e-12))
            p = float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0
            result["contrasts"].append(
                {
                    "camera": camera,
                    "arm": arm,
                    "baseline": baseline,
                    "metric": "b_hit@3",
                    "delta": float(deltas.mean()),
                    "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
                    "task_wins": wins,
                    "task_losses": losses,
                    "task_ties": len(tasks) - wins - losses,
                    "sign_test_p": p,
                    "n_tasks": len(tasks),
                    "n_paired": len(grouped),
                }
            )
    ordered = sorted(result["contrasts"], key=lambda r: r["sign_test_p"])
    adjusted = 0.0
    for i, row in enumerate(ordered):
        adjusted = max(adjusted, min(1.0, (len(ordered) - i) * row["sign_test_p"]))
        row["holm_p"] = adjusted
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-inputs", type=Path, required=True)
    parser.add_argument("--fit-inputs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--n-boot", type=int, default=10000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    sources = {}

    def read(path):
        sources[str(path)] = digest(path)
        return json.loads(path.read_text())

    read(args.fit_inputs / "copy_audit.json")
    annotations = read(args.fit_inputs / "seed_annotations.json")["episodes"]
    manifests, models, frozen, cohort_audit = {}, {}, {}, []
    common_splits = None
    for camera, base in CAMERAS.items():
        arm = base + "_cal"
        directory = args.fit_inputs / "fits" / arm
        manifest = read(directory / "calibration_routing.json")
        manifests[camera] = manifest
        if common_splits is None:
            common_splits = manifest["task_splits"]
        assert manifest["task_splits"] == common_splits
        assert manifest["max_task_seeds"] == 10 and not manifest["forced"]
        identities = manifest["episode_identities"]
        assert len(set(identities.values())) == len(identities)
        all_seeds, all_eval = set(), set()
        for split in common_splits.values():
            assert len(split["seed"]) == len(set(split["seed"])) == 10
            assert not set(split["seed"]) & set(split["eval"])
            assert not all_seeds & set(split["seed"])
            assert not all_eval & set(split["eval"])
            all_seeds.update(split["seed"])
            all_eval.update(split["eval"])
        assert not all_seeds & all_eval
        assert set(map(int, manifest["routing"])) == all_eval
        models[camera], frozen[camera] = {}, {}
        for cohort, entry in manifest["fit_files"].items():
            path = directory / Path(entry["path"]).name
            assert digest(path) == entry["sha256"]
            model = read(path)
            task = entry["task_id"]
            seeds = set(common_splits[task]["seed"])
            retained = model["fit_on_episodes"]
            assert set(retained) <= seeds
            assert set(model["seed_episodes"]) <= seeds
            assert len(retained) == len(set(retained)) == model["fit_episode_count"]
            assert model["task_id"] == task and model["arm"] == arm
            predpath = directory / Path(entry["fit_predictions"]).name
            assert digest(predpath) == entry["fit_predictions_sha256"]
            seed_rows = read(predpath)
            assert {r["episode"] for r in seed_rows} <= seeds
            paired = {r["episode"]: r["spans"] for r in seed_rows if r["episode"] in retained}
            # Only the allowlisted seed annotation rows enter this verification.
            seed_truth = {e: annotations[str(e)]["subtasks"] for e in retained}
            recomputed = fit(paired, seed_truth, retained)
            for field in ("offsets", "segment_fractions", "residual_scale", "duration_scale"):
                assert np.allclose(recomputed[field], model[field], rtol=0, atol=5.01e-6)
            models[camera][cohort] = variants(model)
            frozen[camera][cohort] = {
                "source_sha256": entry["sha256"],
                "variants": models[camera][cohort],
            }
            cohort_audit.append(
                {
                    "camera": camera,
                    "task": task,
                    "cohort": cohort,
                    "usable_seeds": len(retained),
                    "n_segments": model["n_segments"],
                    "duration_weight": model["duration_weight"],
                    "fraction_sum": sum(model["segment_fractions"]),
                    "seed_loo_tiou_gain_points": model["heldout_gain_points"],
                }
            )
    # Freeze all transformations before loading evaluation truth or scoring.
    write_json(args.out / "frozen_variants.json", frozen)
    jobs = []
    expected_scores = {}
    with (args.run_inputs / "scores_arms.jsonl").open() as stream:
        sources[str(args.run_inputs / "scores_arms.jsonl")] = digest(
            args.run_inputs / "scores_arms.jsonl"
        )
        for line in stream:
            row = json.loads(line)
            if row["arm"] in CAMERAS.values():
                expected_scores[(row["dataset"], row["episode"], row["arm"])] = row
    for camera, base in CAMERAS.items():
        manifest = manifests[camera]
        reverse = {
            original: int(converted)
            for converted, original in manifest["episode_identities"].items()
        }
        for task, split in common_splits.items():
            dataset = "corpus_b__" + task
            gt = read(args.run_inputs / "merged-gt" / (dataset + ".json"))["episodes"]
            prediction = read(
                args.run_inputs / "merged-predictions" / (dataset + "__" + base + ".json")
            )
            eval_split = read(args.run_inputs / "merged-splits" / (dataset + ".json"))["eval"]
            assert {reverse[e] for e in eval_split} == set(split["eval"])
            assert set(prediction["episodes"]) <= set(map(str, eval_split))
            for episode in eval_split:
                route = manifest["routing"][str(reverse[episode])]
                assert route["task"] == task
                truth = gt[str(episode)]
                assert len(truth) == route["n_segments"]
                cohort = route["cohort"]
                model = models[camera].get(cohort) if route["calibrated"] else None
                assert not route["calibrated"] or model is not None
                predicted = prediction["episodes"].get(str(episode), [])
                # Merging already discarded failed-cohort predictions; never
                # discard successful sibling cohorts because the task is partial.
                meta = {
                    "camera": camera,
                    "task": task,
                    "dataset": dataset,
                    "episode": episode,
                    "cohort": cohort,
                    "fit_available": model is not None,
                    "eligible": prediction["calibration_eligible"][str(episode)],
                }
                jobs.append((meta, [s["text"] for s in truth], truth, predicted, model))
    print(
        f"Validated ten seeds/task; frozen {len(cohort_audit)} fits; {len(jobs)} camera/episode jobs",
        flush=True,
    )
    rows = []
    with (
        ProcessPoolExecutor(max_workers=args.workers) as pool,
        gzip.open(args.out / "episode_results.jsonl.gz", "wt") as stream,
    ):
        for i, batch in enumerate(pool.map(evaluate_job, jobs, chunksize=4), 1):
            baseline = batch[0]
            old = expected_scores[
                (baseline["dataset"], baseline["episode"], CAMERAS[baseline["camera"]])
            ]
            assert baseline["scored"] == old["scored"]
            for metric in METRICS:
                if baseline["scored"]:
                    assert abs(baseline[metric] - old[metric]) < 1e-10
            for row in batch:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            rows.extend(batch)
            if i % 100 == 0:
                print(f"Scored {i}/{len(jobs)} paired camera/episodes", flush=True)
    result = summarize(rows, args.n_boot, 1729)
    # Descriptive support and seed-CV generalization diagnostics; no tuning.
    for entry in cohort_audit:
        subset = [
            r
            for r in rows
            if r["camera"] == entry["camera"] and r["cohort"] == entry["cohort"] and r["scored"]
        ]
        full = [r for r in subset if r["variant"] == "full"]
        baseline = [r for r in subset if r["variant"] == "baseline"]
        entry["n_eval"] = len(full)
        entry["n_applied"] = sum(r["application"] == "applied" for r in full)
        entry["eval_tiou_gain_points"] = (
            100
            * float(
                np.mean(
                    [
                        a["macro_temporal_iou"] - b["macro_temporal_iou"]
                        for a, b in zip(full, baseline, strict=True)
                    ]
                )
            )
            if full
            else None
        )
    result["cohorts"] = cohort_audit
    result["seed_cv_diagnostics"] = {}
    for camera in CAMERAS:
        cohort = [r for r in cohort_audit if r["camera"] == camera and r["n_eval"]]
        rho = spearmanr(
            [r["seed_loo_tiou_gain_points"] for r in cohort],
            [r["eval_tiou_gain_points"] for r in cohort],
        )
        result["seed_cv_diagnostics"][camera] = {
            "n_cohorts": len(cohort),
            "spearman_rho": float(rho.statistic),
            "eval_negative_gain_cohorts": sum(r["eval_tiou_gain_points"] < 0 for r in cohort),
        }
    write_json(args.out / "summary.json", result)
    sources[str(Path(__file__).resolve())] = digest(Path(__file__).resolve())
    for path in (
        Path(__file__).with_name("score_alignment.py"),
        Path(__file__).parents[2] / "src/lerobot_align/diagnostics/fit_align_calibration.py",
        Path(__file__).parents[2] / "src/lerobot_align/alignment_metrics.py",
    ):
        sources[str(path.resolve())] = digest(path)
    # Source annotations have independent redistribution terms. Keep fingerprints
    # in provenance, but never copy source inputs into a shareable result archive.
    for name, sha in sources.items():
        assert digest(Path(name)) == sha, f"Input changed during experiment: {name}"
    write_json(
        args.out / "provenance.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "command": sys.argv,
            "input_and_code_sha256": sources,
            "seed_budget_per_task": 10,
            "n_tasks": len(common_splits),
            "n_seed_trajectories": sum(len(v["seed"]) for v in common_splits.values()),
            "n_eval_trajectories": sum(len(v["eval"]) for v in common_splits.values()),
            "frozen_variants_sha256": digest(args.out / "frozen_variants.json"),
            "validation": "all frozen fit parameters reproduced from allowlisted seeds; all baseline scores reproduced",
            "limitations": "offline stitched predictions; no final frame snapping; partial outputs unchanged; exploratory",
        },
    )
    print(json.dumps(result["contrasts"], indent=2), flush=True)


if __name__ == "__main__":
    main()
