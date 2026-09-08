#!/usr/bin/env python3
"""Validate and render the saved paired Corpus B experiment results."""

from collections import defaultdict
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation/results/corpus_b_paired_ablations"
REPORT = ROOT / "evaluation/analysis/corpus_b_paired_ablation_report.md"
NAMES = {
    "baseline": "Uncalibrated",
    "offsets": "Offsets only",
    "duration": "Duration regularization only",
    "full": "Full calibration",
    "normalized": "Full + normalized duration prior",
}


def main():
    summary = json.loads((OUT / "summary.json").read_text())
    provenance = json.loads((OUT / "provenance.json").read_text())
    with gzip.open(OUT / "episode_results.jsonl.gz", "rt") as stream:
        rows = [json.loads(line) for line in stream]
    lookup = {(r["camera"], r["task"], r["episode"], r["variant"]): r for r in rows}
    assert len(lookup) == len(rows) == 13050
    checks = {
        "unique_rows": len(rows),
        "baseline_scores_reproduced": 2610,
        "frozen_fits_reproduced_from_seeds": len(summary["cohorts"]),
        "n_tasks": provenance["n_tasks"],
        "n_seeds": provenance["n_seed_trajectories"],
    }
    dropped = defaultdict(dict)
    for camera, data in summary["cameras"].items():
        for variant, expected in data["arms"].items():
            selected = [r for r in rows if r["camera"] == camera and r["variant"] == variant]
            scored = [r for r in selected if r["scored"]]
            assert len(scored) == expected["scored"] == 1299
            for metric in ("b_hit@3", "macro_temporal_iou", "placed_fraction"):
                by_task = defaultdict(list)
                for row in scored:
                    by_task[row["task"]].append(row[metric])
                # Independent stdlib arithmetic verifies NumPy summary calculations.
                measured = statistics.mean(statistics.mean(v) for v in by_task.values())
                assert abs(measured - expected["task_macro"][metric]) < 1e-12
            lost = []
            for row in selected:
                base = lookup[(camera, row["task"], row["episode"], "baseline")]
                if row["application"] != "applied":
                    assert row["predictions"] == base["predictions"]
                if row["scored"]:
                    lost.append(base["n_placed"] - row["n_placed"])
            dropped[camera][variant] = {"episodes": sum(v > 0 for v in lost), "labels": sum(lost)}
    for contrast in summary["contrasts"]:
        n = contrast["task_wins"] + contrast["task_losses"]
        k = min(contrast["task_wins"], contrast["task_losses"])
        exact = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)
        assert abs(exact - contrast["sign_test_p"]) < 1e-12
    checks["drop_diagnostics"] = dropped
    checks["assessment"] = "Share with caveats: paired offline ablations, not a production replay"
    (OUT / "validation.json").write_text(json.dumps(checks, indent=2) + "\n")
    with (OUT / "comparison.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["camera", "variant", "n_scored", "B@3_percent", "tIoU_percent", "placement_percent"]
        )
        for camera, data in summary["cameras"].items():
            for variant, arm in data["arms"].items():
                writer.writerow(
                    [
                        camera,
                        variant,
                        arm["scored"],
                        *[
                            100 * arm["task_macro"][m]
                            for m in ("b_hit@3", "macro_temporal_iou", "placed_fraction")
                        ],
                    ]
                )
    with (OUT / "per_task.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["camera", "task", "variant", "B@3", "tIoU", "placement"])
        for camera, data in summary["cameras"].items():
            for variant, tasks in data["per_task"].items():
                for task, metrics in tasks.items():
                    writer.writerow(
                        [
                            camera,
                            task,
                            variant,
                            metrics["b_hit@3"],
                            metrics["macro_temporal_iou"],
                            metrics["placed_fraction"],
                        ]
                    )

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True, sharey=True)
    colors = ["#89939f", "#ce7751", "#397fa5", "#43896b", "#8666a7"]
    for axis, camera in zip(axes, ("single", "stacked"), strict=True):
        values = [
            100 * summary["cameras"][camera]["arms"][v]["task_macro"]["b_hit@3"] for v in NAMES
        ]
        axis.barh(range(5), values, color=colors, height=0.68)
        axis.set_title(
            "Single camera" if camera == "single" else "Stacked cameras", fontweight="bold"
        )
        axis.set_xlim(0, 60)
        axis.set_yticks(range(5), list(NAMES.values()))
        axis.set_xlabel("B@3 (%) · equal weight per task")
        axis.spines[["top", "right"]].set_visible(False)
        for i, value in enumerate(values):
            axis.text(value + 0.7, i, f"{value:.2f}", va="center")
    axes[0].invert_yaxis()
    fig.suptitle(
        "Corpus B: duration regularization supplies most of the observed gain", fontsize=14
    )
    fig.text(
        0.5,
        0.01,
        "Same saved predictions · 10 seeds per task · 39 tasks · 1,299 scorable trajectories per camera\n"
        "Exploratory offline postprocessing; six ambiguous trajectories excluded per camera; failures retained.",
        ha="center",
        fontsize=9,
        color="#4b5563",
    )
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(OUT / "comparison.png", dpi=180)
    fig.savefig(OUT / "comparison.pdf")
    plt.close(fig)

    lines = [
        "# Corpus B: paired calibration ablations",
        "",
        "**Duration regularization supplies most of the observed calibration benefit.** "
        "Offsets alone reduce average boundary accuracy and create dropped labels. "
        "Prior normalization has small, inconsistent effects across cameras.",
        "",
        f"Run completed {provenance['created_utc']}. These are **new offline postprocessing experiments** "
        "on saved run2 predictions, with no new VLM calls. Existing production runs are discussed separately below.",
        "",
        "## Design and seed budget",
        "",
        "All 39 tasks retain their original **ten seed trajectories per task: 390 total**, shared "
        "across both camera arms and all segment-count cohorts. No evaluation trajectory was used "
        "to fit or tune parameters. All 67 retained frozen fits (35 single, 32 stacked) were "
        "hash-verified and their offsets, fractions, and scales independently reproduced from "
        "their allowlisted seeds. Original quality gates and seed-selected duration weights remain fixed. "
        "These are component-removal ablations, not separately optimized replacements.",
        "",
        "Each camera supplies 1,305 requested evaluation trajectories. Six have ambiguous label "
        "correspondence and remain excluded identically across its five variants, leaving **1,299 "
        "scorable trajectories in 39 tasks per camera**. Failed/empty outputs remain scored: "
        "40 single-camera and 78 stacked-camera trajectories. The two cameras' six exclusions "
        "need not be the same episodes; primary contrasts are within camera only.",
        "",
        "Offsets-only removes the duration solver. Duration-only zeroes offsets but retains the "
        "duration solver, scales, and original weight (including zero weights). Normalization divides "
        "the fitted duration fractions by their sum and retains the original scales/weight. "
        "Unrouted, empty, and incomplete/ambiguous outputs are unchanged. Label occurrence indices "
        "are recovered before calibration and retained through any boundary collision.",
        "",
        "## Results",
        "",
        "B@3 is the percentage of internal boundaries within three seconds of ground truth; "
        "missing boundaries are misses. tIoU measures temporal overlap; placement measures "
        "the fraction of supplied labels placed. Higher is better. All values below are "
        "percentages, averaged within task and then equally across tasks.",
        "",
        "| Method | Single B@3 | Single tIoU | Single placement | Stacked B@3 | Stacked tIoU | Stacked placement |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, name in NAMES.items():
        values = [
            100 * summary["cameras"][cam]["arms"][variant]["task_macro"][metric]
            for cam in ("single", "stacked")
            for metric in ("b_hit@3", "macro_temporal_iou", "placed_fraction")
        ]
        lines.append("| " + name + " | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += [
        "",
        "![Paired calibration comparison](../results/corpus_b_paired_ablations/comparison.png)",
        "",
        "## Paired differences and uncertainty",
        "",
        "Ten thousand task-cluster bootstrap draws give marginal percentile 95% intervals. "
        "Exact sign tests use task wins/losses (ties excluded); Holm adjusts the single family "
        "of all 12 planned B@3 contrasts. Intervals are not simultaneous and can exclude zero "
        "when the corrected sign test is not significant. Tasks may remain correlated within "
        "robot configurations; the intervals do not capture that higher-level dependence. "
        "This bootstrap differs from the historical two-stage bias-corrected bootstrap.",
        "",
        "| Camera | Contrast | Δ B@3 (pp) | 95% CI (pp) | Task wins/losses/ties | Holm p |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for c in summary["contrasts"]:
        lines.append(
            f"| {c['camera']} | {c['arm']} − {c['baseline']} | {100 * c['delta']:+.2f} | "
            f"[{100 * c['ci95'][0]:+.2f}, {100 * c['ci95'][1]:+.2f}] | "
            f"{c['task_wins']}/{c['task_losses']}/{c['task_ties']} | {c['holm_p']:.5f} |"
        )
    lines += [
        "",
        "## What the experiments establish",
        "",
        "1. **Duration regularization is the strongest supported contributor.** "
        "It improves B@3 by 5.80 points (single; Holm p=0.00036) and 5.39 points "
        "(stacked; p=0.00819). Full calibration adds only 0.36 points over duration-only "
        "for single camera and subtracts 1.37 for stacked cameras; neither difference "
        "is significant after correction. This does not establish equivalence or prove "
        "that separately refitted offsets cannot help.",
        "",
        "2. **Offsets-only has a concrete collision problem in this diagnostic path.** "
        "It drops 93 labels across 92 single-camera trajectories and 82 labels across "
        "80 stacked-camera trajectories. Full calibration drops only 3 and 7 labels, "
        "respectively; duration-only drops none. This supports checking ordering constraints "
        "and offset stability before relying on the offsets-only path. Its lower mean "
        "B@3 alone does not pass the corrected significance threshold for either camera.",
        "",
        "3. **Normalization is not supported as a general repair.** Normalizing duration "
        "fractions changes B@3 by +0.62 points for single camera and −0.71 for stacked. "
        "Neither corrected test is significant. Fitted fraction sums range from 0.9480 "
        "to 1.0364 (single) and 0.9480 to 1.0283 (stacked); deviations are real but "
        "small in these retained fits. This tests normalization with frozen original "
        "scales/weights, not a wholly refitted normalized method.",
        "",
        "4. **Seed-selected apparent gains transfer unevenly.** Median usable seeds are "
        "8 per retained count cohort for single camera and 9 for stacked. These remain "
        "subsets of the original ten per task. Median seed-LOO tIoU gains are 14.56 and "
        "13.46 points; median evaluation cohort gains are 5.40 and 3.85. Seven of 35 "
        "single-camera fits and ten of 32 stacked fits have negative evaluation tIoU gain. "
        "Spearman correlations between seed-LOO and evaluation gains are only 0.22 and "
        "0.12. Because the gate selected positive LOO gains and the replay is approximate, "
        "this is evidence of limited predictive reliability, not proof of a particular "
        "distribution-shift mechanism.",
        "",
        "## Coverage and limitations",
        "",
        "Calibration is unavailable for 450 single-camera and 521 stacked-camera "
        "evaluation trajectories under the frozen routing. It is applied here to 821 and "
        "753 complete outputs, respectively. Among routed single-camera trajectories, "
        "22 are empty and 12 incomplete/ambiguous; among routed stacked trajectories, "
        "31 are empty. Fallbacks remain in the headline population. On the shared "
        "seed-count-eligible subset, descriptive full-minus-baseline B@3 gains are "
        "+6.91 points (1,057 scorable single-camera trajectories, 38 tasks) and "
        "+4.64 (1,058 stacked-camera trajectories, 38 tasks); no additional hypothesis "
        "tests were performed for this secondary subset.",
        "",
        "The inputs are already stitched and frame-snapped baseline predictions. Production "
        "calibration runs earlier, can correct partial outputs using their original indices, "
        "and snaps frames afterward. This study cannot undo dropped labels or recover raw "
        "model timestamps, and leaves partial outputs unchanged to avoid positional errors. "
        "The experiment isolates postprocessing on common predictions; it is **not an exact "
        "production replay**. Duration-only may make tiny grid-rounding changes even for "
        "zero selected weights (three single-camera fits, one stacked fit).",
        "",
        "The corrected historical production report measured full-calibrated B@3 of 47.45% "
        "(single) and 49.38% (stacked), using separate model calls and arm-specific output "
        "failures. Here the full-calibration values are 47.83% and 47.94% on each base "
        "arm's fixed outputs. Their differences cannot be assigned solely to runtime code "
        "without raw-response replay. No production defaults were changed based on these results.",
        "",
        "## Targeted next experiments",
        "",
        "- Replay cached **raw, indexed responses** through production cleanup/frame snapping "
        "for baseline, full, and zero-offset duration regularization. Cache one response per "
        "trajectory so the comparison does not conflate calibration with model randomness.",
        "- Test monotonic/minimum-duration constraints for offsets-only to separate "
        "boundary collisions from poor offset transfer. Choose any shrinkage strength "
        "using seed-only nested validation, with the same ten seed trajectories.",
        "- Compare 3/5/10-seed subsets drawn strictly from the existing ten, pooling no "
        "extra trajectories. Refit and select gates/weights within each subset; report "
        "coverage and stability rather than choosing a winner on evaluation outcomes.",
        "",
        "## Validation and reproducibility",
        "",
        "Assessment: **share with the offline/exploratory caveats above**. All 2,610 "
        "baseline camera/episode scores reproduced the corrected scorer output. Independent "
        "stdlib calculations verified every task-weighted headline mean and all exact "
        "sign-test p-values. No-fit/partial/empty predictions were verified unchanged, "
        "and 13,050 camera/episode/variant rows are unique. A second execution reproduced "
        "the entire numerical summary exactly. Targeted tests cover missing outputs, "
        "occurrence preservation after collisions, unchanged partial outputs, and task "
        "weighting without excluding failed tasks.",
        "",
        "Artifacts: [summary](../results/corpus_b_paired_ablations/summary.json), "
        "[comparison CSV](../results/corpus_b_paired_ablations/comparison.csv), "
        "[per-task CSV](../results/corpus_b_paired_ablations/per_task.csv), "
        "[episode scores and predictions](../results/corpus_b_paired_ablations/episode_results.jsonl.gz), "
        "[frozen variants](../results/corpus_b_paired_ablations/frozen_variants.json), "
        "[provenance](../results/corpus_b_paired_ablations/provenance.json), "
        "[validation](../results/corpus_b_paired_ablations/validation.json), "
        "[exact input snapshot](../results/corpus_b_paired_ablations/inputs.tar.gz), "
        "[experiment plan](corpus_b_ablation_plan.md).",
        "",
        "To reproduce in this repository environment, extract the input snapshot to a "
        "new directory and run (use a new, nonexistent output directory):",
        "",
        "```bash",
        "mkdir -p /tmp/corpus-b-replay-inputs",
        "tar -xzf evaluation/results/corpus_b_paired_ablations/inputs.tar.gz -C /tmp/corpus-b-replay-inputs",
        ".venv/bin/python evaluation/scripts/experiment_corpus_b_calibration.py \\",
        "  --run-inputs /tmp/corpus-b-replay-inputs/run_inputs \\",
        "  --fit-inputs /tmp/corpus-b-replay-inputs/fit_inputs \\",
        "  --out /tmp/corpus-b-replay-results --workers 6",
        "```",
        "",
        "The snapshot includes the exact experiment/scorer/calibrator/metric source files "
        "under `code/`; provenance records their SHA-256 hashes. The run requires the "
        "repository's Python environment. Re-render the canonical report with "
        "`MPLCONFIGDIR=/tmp/corpus-b-mpl .venv/bin/python evaluation/scripts/report_corpus_b_experiments.py`.",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    checksums = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in OUT.iterdir()
        if p.is_file() and p.name != "artifact_hashes.json"
    }
    (OUT / "artifact_hashes.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(REPORT)


if __name__ == "__main__":
    main()
