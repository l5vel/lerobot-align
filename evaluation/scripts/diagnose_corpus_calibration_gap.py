#!/usr/bin/env python3
"""Cross-corpus calibration diagnosis using frozen seeds and saved predictions.

No evaluation labels enter fitting. Evaluation medians below are descriptive
dispersion/shift diagnostics only, never applied to predictions.
"""

from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

import jsonl_io
from experiment_corpus_b_calibration import evaluate_job, variants
from lerobot_align.diagnostics.fit_align_calibration import fit, unusable_episode_reason

ROOT = Path(__file__).resolve().parents[2]
A = Path("/tmp/corpus-a-calibration-diagnosis")
B = Path("/tmp/lerobot-calibration-fixes-run2")
BF = Path("/tmp/lerobot-corpus-b-ablation-inputs")
OUT = ROOT / "evaluation/results/corpus_calibration_diagnosis"
SOURCES = {}


def read(path):
    data = path.read_bytes()
    SOURCES[str(path)] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def macro(rows, metric):
    groups = defaultdict(list)
    for r in rows:
        if r["scored"] and r.get(metric) is not None:
            groups[r["dataset"]].append(r[metric])
    return statistics.mean(map(statistics.mean, groups.values())) if groups else None


def distribution(xs):
    return (
        {
            "n": len(xs),
            "median": float(np.median(xs)),
            "p10": float(np.quantile(xs, 0.1)),
            "p90": float(np.quantile(xs, 0.9)),
            "mean": float(np.mean(xs)),
        }
        if xs
        else {}
    )


def cohort_diagnostic(corpus, camera, name, model, seed_truth, eval_truth, predictions):
    def positions(gt):
        start = gt[0]["start"]
        duration = gt[-1]["end"] - start
        assert duration > 0
        return [(s["start"] - start) / duration for s in gt[1:]]

    seed_positions = np.array([positions(gt) for gt in seed_truth])
    prior = np.median(seed_positions, axis=0)
    target_positions = np.array([positions(gt) for gt in eval_truth.values()])
    if not len(target_positions):
        return None
    target_mid = np.median(target_positions, axis=0)
    normalized_prior_error = np.abs(target_positions - prior).mean(axis=1)
    seconds_prior_error = [
        v * (gt[-1]["end"] - gt[0]["start"])
        for v, gt in zip(normalized_prior_error, eval_truth.values(), strict=True)
    ]
    errors = []
    for episode, gt in eval_truth.items():
        pred = predictions.get(str(episode), [])
        if len(pred) != len(gt) or [s["text"] for s in pred] != [s["text"] for s in gt]:
            continue
        duration = pred[-1]["end"] - pred[0]["start"]
        errors.append(
            [(p["start"] - g["start"]) / duration for p, g in zip(pred[1:], gt[1:], strict=True)]
        )
    diag = {
        "corpus": corpus,
        "camera": camera,
        "cohort": name,
        "seeds": len(seed_truth),
        "n_eval": len(eval_truth),
        "n_segments": len(prior) + 1,
        "duration_weight": model["duration_weight"],
        "prior_fraction_sum": sum(model["segment_fractions"]),
        "seed_prior_error_norm": float(normalized_prior_error.mean()),
        "seed_prior_error_seconds": float(np.mean(seconds_prior_error)),
        "eval_within_cohort_dispersion_norm": float(np.abs(target_positions - target_mid).mean()),
        "seed_eval_target_median_shift_norm": float(np.abs(prior - target_mid).mean()),
        "seed_duration_scale": model["duration_scale"],
        "seed_offset_residual_scale": model["residual_scale"],
    }
    if errors:
        err = np.array(errors)
        mid = np.median(err, axis=0)
        offsets = np.array(model["offsets"])
        nontrivial = np.abs(offsets) >= 0.02
        diag.update(
            n_complete_predictions=len(err),
            offset_median_shift_norm=float(np.abs(mid - offsets).mean()),
            eval_offset_residual_norm=float(np.abs(err - mid).mean()),
            offset_magnitude_norm=float(np.abs(offsets).mean()),
            nontrivial_offsets=int(nontrivial.sum()),
            sign_reversed_offsets=int(((mid * offsets < 0) & nontrivial).sum()),
        )
    return diag


def main():
    OUT.mkdir(exist_ok=True)
    meta_a = read(ROOT / "evaluation/results/alignment_calibration.json")
    old_a = jsonl_io.load_jsonl(ROOT / "evaluation/results/alignment_scores.jsonl")
    old_b = jsonl_io.load_jsonl(B / "scores.jsonl")
    summary = {
        "production": {},
        "offline": {},
        "dataset_stats": {},
        "cohort_diagnostics": [],
        "legacy_fit_validation": [],
        "sensitivity": {},
    }
    for corpus, rows in [("A", old_a), ("B", old_b)]:
        summary["production"][corpus] = {}
        for arm in sorted({r["arm"] for r in rows}):
            rs = [r for r in rows if r["arm"] == arm]
            summary["production"][corpus][arm] = {
                "requested": len(rs),
                "scored": sum(r["scored"] for r in rs),
                "scored_failures": sum(r["scored"] and not r["ok"] for r in rs),
                **{m: macro(rs, m) for m in ["b_hit@3", "macro_temporal_iou", "placed_fraction"]},
            }
    jobs, truths_a, splits_a, models_a = [], {}, {}, {}
    for path in sorted((A / "ground_truth").glob("*.json")):
        task = path.stem
        truth = read(path)["episodes"]
        split = read(A / "splits" / path.name)
        assert len(split["seed"]) == 10 and not set(split["seed"]) & set(split["eval"])
        truths_a[task], splits_a[task] = truth, split
        for camera, base in [("single", "align_video"), ("stacked", "align_video_stack")]:
            arm = base + "_cal"
            model = read(A / "calibration" / arm / (task + ".json"))
            assert set(model["fit_on_episodes"]) <= set(split["seed"])
            models_a[camera, task] = model
            seed_rows = read(A / "calibration" / arm / (task + ".fit_predictions.json"))
            seed_preds = {r["episode"]: r["spans"] for r in seed_rows}
            seed_gt = {
                e: [{**s, "label": s["text"]} for s in truth[str(e)]]
                for e in model["fit_on_episodes"]
            }
            reasons = {e: unusable_episode_reason(seed_preds[e], seed_gt[e]) for e in seed_gt}
            recomputed = fit(seed_preds, seed_gt, model["fit_on_episodes"])
            parameter_match = all(
                np.allclose(recomputed[k], model[k], rtol=0, atol=5.01e-6)
                for k in ["offsets", "segment_fractions", "residual_scale", "duration_scale"]
            )
            summary["legacy_fit_validation"].append(
                {
                    "task": task,
                    "camera": camera,
                    "recorded_seed_count": len(seed_gt),
                    "currently_usable": sum(v is None for v in reasons.values()),
                    "rejections": {e: r for e, r in reasons.items() if r is not None},
                    "matches_current_fitter": parameter_match,
                }
            )
            payload = read(A / "predictions_alignment/oracle" / (task + "__" + base + ".json"))
            preds = payload["episodes"] if payload["ok"] else {}
            matched = {}
            for episode in split["eval"]:
                gt = truth[str(episode)]
                labels = [s["text"] for s in gt]
                supported = labels == model["labels"]
                if supported:
                    matched[episode] = gt
                jobs.append(
                    (
                        {
                            "corpus": "A",
                            "camera": camera,
                            "task": task,
                            "dataset": task,
                            "episode": episode,
                            "fit_available": supported,
                        },
                        labels,
                        gt,
                        preds.get(str(episode), []),
                        variants(model) if supported else None,
                    )
                )
            diag = cohort_diagnostic(
                "A",
                camera,
                task,
                model,
                [truth[str(e)] for e in model["fit_on_episodes"]],
                matched,
                preds,
            )
            summary["cohort_diagnostics"].append(diag)
    with ProcessPoolExecutor(max_workers=6) as pool:
        a_rows = [r for batch in pool.map(evaluate_job, jobs, chunksize=4) for r in batch]
    old_lookup = {(r["dataset"], r["episode"], r["arm"]): r for r in old_a}
    for row in a_rows:
        if row["variant"] == "baseline":
            arm = "align_video" if row["camera"] == "single" else "align_video_stack"
            old = old_lookup[row["dataset"], row["episode"], arm]
            assert old["scored"] == row["scored"]
            if row["scored"]:
                assert abs(old["b_hit@3"] - row["b_hit@3"]) < 1e-10
    with gzip.open(
        ROOT / "evaluation/results/corpus_b_paired_ablations/episode_results.jsonl.gz", "rt"
    ) as f:
        b_rows = [{**json.loads(line), "corpus": "B"} for line in f]
    for corpus, rows in [("A", a_rows), ("B", b_rows)]:
        summary["sensitivity"][corpus] = {}
        summary["offline"][corpus] = {}
        for camera in ["single", "stacked"]:
            summary["offline"][corpus][camera] = {}
            for variant in ["baseline", "offsets", "duration", "full", "normalized"]:
                rs = [r for r in rows if r["camera"] == camera and r["variant"] == variant]
                summary["offline"][corpus][camera][variant] = {
                    **{
                        m: macro(rs, m)
                        for m in ["b_hit@3", "macro_temporal_iou", "placed_fraction"]
                    },
                    "scored": sum(r["scored"] for r in rs),
                    "applied": sum(r["application"] == "applied" for r in rs),
                }
            full = [r for r in rows if r["camera"] == camera and r["variant"] == "full"]
            base = {
                (r["task"], r["episode"]): r
                for r in rows
                if r["camera"] == camera and r["variant"] == "baseline"
            }
            diffs = [
                {**r, "gain": r["b_hit@3"] - base[r["task"], r["episode"]]["b_hit@3"]}
                for r in full
                if r["scored"]
            ]
            summary["offline"][corpus][camera]["applied_subset_gain"] = macro(
                [r for r in diffs if r["application"] == "applied"], "gain"
            )
            cuts = {
                "all": lambda r: True,
                "duration_le_150": lambda r: r["episode_duration"] <= 150,
                "duration_40_to_80": lambda r: 40 <= r["episode_duration"] <= 80,
            }
            if corpus == "A":
                cuts.update(
                    non_bag=lambda r: "bag-place" not in r["task"],
                    non_fridge=lambda r: "fridge-drink" not in r["task"],
                )
            else:
                cuts.update(three_segments=lambda r: r["n_labels"] == 3)
            summary["sensitivity"][corpus][camera] = {
                name: {
                    "n": sum(predicate(r) for r in diffs),
                    "gain": macro([r for r in diffs if predicate(r)], "gain"),
                }
                for name, predicate in cuts.items()
            }
            if corpus == "A":
                gated = [
                    {
                        **r,
                        "gain": r["gain"]
                        if meta_a[
                            r["task"]
                            + "__"
                            + ("align_video_cal" if camera == "single" else "align_video_stack_cal")
                        ]["gate_verdict"]
                        == "RECOMMENDED"
                        else 0.0,
                    }
                    for r in diffs
                ]
                summary["offline"][corpus][camera]["gate_obeyed_gain"] = macro(gated, "gain")
    annotation_b = read(BF / "seed_annotations.json")["episodes"]
    for camera, arm in [("single", "align_video_cal"), ("stacked", "align_video_stack_cal")]:
        manifest = read(BF / "fits" / arm / "calibration_routing.json")
        ids = manifest["episode_identities"]
        for cohort, entry in manifest["fit_files"].items():
            task = entry["task_id"]
            model = read(BF / "fits" / arm / Path(entry["path"]).name)
            truth = read(B / "merged-gt" / ("corpus_b__" + task + ".json"))["episodes"]
            preds = read(
                B
                / "merged-predictions"
                / ("corpus_b__" + task + "__" + arm.removesuffix("_cal") + ".json")
            )["episodes"]
            matched = {
                ids[e]: truth[str(ids[e])]
                for e, r in manifest["routing"].items()
                if r["cohort"] == cohort and r["calibrated"]
            }
            seeds = [
                [{**s, "text": s["label"]} for s in annotation_b[str(e)]["subtasks"]]
                for e in model["fit_on_episodes"]
            ]
            summary["cohort_diagnostics"].append(
                cohort_diagnostic("B", camera, cohort, model, seeds, matched, preds)
            )
    for corpus in ["A", "B"]:
        durations, counts, unique_rates = [], [], []
        cohort_counts = []
        if corpus == "A":
            task_gt = [(t, truths_a[t], [str(e) for e in splits_a[t]["eval"]]) for t in truths_a]
        else:
            task_gt = []
            for path in (B / "merged-gt").glob("*.json"):
                gt = read(path)["episodes"]
                ev = read(B / "merged-splits" / path.name)["eval"]
                task_gt.append((path.stem, gt, list(map(str, ev))))
        for _task, gt, ev in task_gt:
            episodes = [gt[e] for e in ev]
            ds = [s[-1]["end"] - s[0]["start"] for s in episodes]
            durations.extend(ds)
            counts.extend(map(len, episodes))
            unique_rates.append(
                len({tuple(s["text"] for s in ep) for ep in episodes}) / len(episodes)
            )
            cohort_counts.append(len(set(map(len, episodes))))
        summary["dataset_stats"][corpus] = {
            "n_eval": len(durations),
            "duration_seconds": distribution(durations),
            "n_segments": dict(Counter(counts)),
            "distinct_counts_per_task": distribution(cohort_counts),
            "ordered_label_tuple_unique_fraction_per_task": distribution(unique_rates),
            "over_150_seconds": sum(d > 150 for d in durations),
        }
    summary["diagnostic_distributions"] = {}
    for corpus in ["A", "B"]:
        summary["diagnostic_distributions"][corpus] = {}
        for camera in ["single", "stacked"]:
            ds = [
                d
                for d in summary["cohort_diagnostics"]
                if d and d["corpus"] == corpus and d["camera"] == camera
            ]
            summary["diagnostic_distributions"][corpus][camera] = {
                k: distribution([d[k] for d in ds if k in d])
                for k in [
                    "seeds",
                    "seed_prior_error_norm",
                    "seed_prior_error_seconds",
                    "eval_within_cohort_dispersion_norm",
                    "seed_eval_target_median_shift_norm",
                    "offset_median_shift_norm",
                    "eval_offset_residual_norm",
                    "offset_magnitude_norm",
                ]
            }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with gzip.open(OUT / "corpus_a_offline_rows.jsonl.gz", "wt") as f:
        for row in a_rows:
            f.write(json.dumps(row) + "\n")
    for p in [
        Path(__file__),
        ROOT / "evaluation/results/alignment_scores.jsonl",
        B / "scores.jsonl",
        ROOT / "evaluation/results/corpus_b_paired_ablations/episode_results.jsonl.gz",
    ]:
        # Hash the decompressed content so the recorded digest is stable
        # whether the scores are stored .jsonl or .jsonl.gz.
        SOURCES[str(p)] = hashlib.sha256(jsonl_io.content_bytes(p)).hexdigest()
    (OUT / "provenance.json").write_text(json.dumps(SOURCES, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "cohort_diagnostics"}, indent=2))


if __name__ == "__main__":
    main()
