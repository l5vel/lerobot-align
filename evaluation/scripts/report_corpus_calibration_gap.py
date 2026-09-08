#!/usr/bin/env python3
"""Build the canonical technical report from reviewed diagnosis outputs."""

from datetime import UTC, datetime
import json
import gzip
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation/results/corpus_calibration_diagnosis"


def main():
    s = json.loads((OUT / "summary.json").read_text())
    title = "Why calibration transfers poorly to Corpus B"
    sections = [
        (
            "summary",
            "Technical summary",
            "**The strongest explanation is a weaker match to the calibration assumptions.** "
            "Corpus B has much less stable relative boundary timing within fitted task/count cohorts, "
            "and seed-fitted timing offsets transfer poorly to the remaining trajectories. "
            "The method still helps on average, principally through duration regularization.\n\n"
            "**The historical comparison also mixes protocols.** Corpus A used finer component-specific "
            "fits, forced rejected calibrations, and an older fitter that accepted incomplete seed predictions. "
            "These differences matter, but sensitivity checks show they do not explain the whole gap.\n\n"
            "**This investigation adds new evidence:** paired offline ablations on Corpus A using the "
            "same diagnostic/scorer as Corpus B; seed-to-evaluation timing diagnostics for both corpora; "
            "gate, duration, development-family, and legacy-bug sensitivity checks. "
            "No extra seed trajectories or VLM calls were used. The results diagnose mechanisms; "
            "they do not constitute a randomized cross-corpus causal experiment.",
            "analysis",
        ),
        (
            "definitions",
            "What is being compared",
            "B@3 is the fraction of internal boundaries within three seconds of the reference, with missing "
            "boundaries counted as misses. Scores average episodes within component/task, then give each "
            "component/task equal weight. Corpus A has 637 evaluation trajectories in 16 components; "
            "Corpus B has 1,305 in 39 tasks. The paired offline analysis includes 637 scorable trajectories "
            "per camera for A and 1,299 for B; six ambiguous B trajectories per camera remain excluded "
            "across all variants. Empty predictions remain scored failures.\n\n"
            "Corpus A’s historical allocation is ten seeds per component (160 allocated total). Corpus B’s "
            "allocation is ten total per task (390 total), shared across camera arms and count cohorts. "
            "For A, exact ordered labels determine applicability; B pools by task and segment count. "
            "All parameters, gates, weights, and seed allocations were frozen before evaluation. "
            "The A and B task populations and grouping granularity are different.",
            "analysis",
        ),
        (
            "observed",
            "The performance gap survives consistent failure handling",
            "Reaggregating saved score rows with all scored failures included gives single-camera B@3 "
            "54.59% → 78.08% for A and 41.67% → 47.45% for B. Stacked-camera values are "
            "57.38% → 79.43% for A and 43.91% → 49.38% for B. These are descriptive arm means; "
            "the B arms have slightly different scorable populations, so the new paired replay is the "
            "cleaner component comparison. A’s old successful-output-only report slightly overstated "
            "its scores, but that accounting issue does not remove the gap.\n\n"
            "The new replay applies each ablation to identical saved uncalibrated predictions within "
            "camera and corpus. It does not recover raw responses or replay final frame snapping. "
            "Incomplete outputs are left unchanged to preserve label identity. Consequently its "
            "absolute scores need not equal the historical production scores.",
            "analysis",
        ),
        (
            "offset",
            "1. Seed offsets transfer much less reliably on B — high confidence",
            "Offsets-only improves A’s B@3 by **14.74 points (single) and 17.74 points (stacked)**. "
            "The same ablation changes B by **−1.13 and −3.62 points**. Full calibration improves "
            "A by 18.61/22.78 points versus B by 6.16/4.03. Thus the small B gain is not merely "
            "an artifact of separately sampled calibrated and uncalibrated model responses.\n\n"
            "For each fitted cohort, I compared the frozen offset with the median signed error "
            "on complete evaluation predictions, normalized by episode duration. The median "
            "cohort discrepancy is **1.62% of duration on A versus 6.23% on B** for single camera, "
            "and 1.88% versus 7.69% for stacked cameras. Evaluation residual variation around "
            "its own median is 3.07% versus 9.46% for single camera. A fixed shift cannot remove "
            "this episode-specific variation.\n\n"
            "Among offsets with magnitude at least 2% of duration, the evaluation median error "
            "has the opposite sign for 2/58 A versus 9/62 B boundaries (single), and 2/57 "
            "versus 10/59 (stacked). This sign threshold is an exploratory descriptive cut. "
            "The evidence supports instability/heterogeneity; it cannot distinguish sampling "
            "error, annotation differences, model noise, and distribution shift as separate causes.",
            "analysis",
        ),
        (
            "timing",
            "2. The timing-template assumption is much weaker on B — high confidence",
            "Calibration assumes that the same segment position has a reasonably repeatable "
            "relative duration. I measured normalized human boundary positions within each "
            "retained fit’s evaluation cohort. Median within-cohort absolute dispersion is "
            "**2.52% of episode duration on A versus 8.10% on B** (single-camera fit cohorts), "
            "about 3.2 times larger. This uses evaluation medians only to measure dispersion; "
            "they are never used to calibrate an output.\n\n"
            "A ground-truth-only median boundary template estimated from the original fitting seeds "
            "has median cohort evaluation MAE **1.95 seconds on A versus 4.21 seconds on B**. "
            "Its normalized errors are 3.37% versus 9.58%. Because this check uses human timings "
            "and ignores model predictions, it demonstrates a predictability difference beyond "
            "video rendering or VLM decoding alone. Scope differs: exact-label A cohorts versus "
            "same-count B cohorts, so this identifies a weakness of the actual pooling strategy, "
            "not an immutable property of every possible B grouping.\n\n"
            "The separate model-free cumulative-duration floor likewise reaches B@3 71.96% on A "
            "versus 48.02% on B. In the paired ablations, duration-only still gives B +5.80/+5.39 "
            "points, but gives A +14.72/+18.24. B benefits from a prior whose predictive value is weaker.",
            "analysis",
        ),
        (
            "coverage",
            "3. Coverage and heterogeneous pooling dilute B’s gain — contributing, not sufficient",
            "A task’s ten seeds must cover more count cohorts on B: the median number of observed "
            "evaluation segment counts is three per B task versus two per A component. "
            "The median fraction of distinct ordered label tuples among evaluation trajectories "
            "is 95.2% per B task versus 6.3% per A component. This lexical measure is not proof "
            "of semantic incompatibility, but it shows why exact-schema calibration transfers poorly "
            "and why B required broader positional pooling.\n\n"
            "Frozen B routing supplies no accepted fit for 450/1,305 single-camera and 521/1,305 "
            "stacked-camera trajectories. The offline routine reaches 821 and 753 complete outputs "
            "respectively. A’s replay reaches 455/637 and 513/637. The median recorded fit contains "
            "nine A seeds; B’s retained fits contain eight single-camera or nine stacked-camera seeds. "
            "The problem is not simply that B has fewer total trajectories.\n\n"
            "Coverage alone cannot explain the gap: on the subset actually processed in the replay, "
            "task-averaged gains are **26.35 versus 8.96 points** for A versus B single camera "
            "and **25.46 versus 5.69** for stacked. These are descriptive conditional subsets, "
            "not matched cross-corpus populations or unbiased treatment-effect estimates.",
            "analysis",
        ),
        (
            "protocol",
            "4. Corpus A is a favorable historical benchmark — verified confounding",
            "The 16 A components come from only four broader task families, giving 30–50 allocated "
            "seed trajectories per family, albeit split across independent component fits. "
            "Calling this “ten per task” requires defining task as component; it is not a "
            "family-level ten-trajectory comparison. The repository also records that prompts and "
            "calibration design were developed on a sibling fridge component, and that the same "
            "16 components informed the calibration gate. A is therefore development-adjacent, "
            "not an independent estimate of out-of-domain generalization.",
            "protocol",
        ),
        (
            "sensitivity",
            "Checks that limit those alternative explanations",
            "Applying A’s recorded gate decisions in the offline replay changes its full gain "
            "from 18.61 to **19.35 points** (single) and 22.78 to **21.65** (stacked). "
            "Thus forcing rejected A fits is not the main source of its advantage. Removing the "
            "development-adjacent fridge family still leaves gains of **18.17 and 24.99 points**.\n\n"
            "Corpus B is shorter, not longer: median evaluation duration is **38.9 seconds versus "
            "53.68 on A**. A fixed three-second tolerance is relatively more permissive on B. "
            "Only 23 B trajectories exceed 150 seconds, the nominal duration where a 2-fps/300-frame "
            "budget binds. Excluding them leaves B’s gains essentially unchanged: **6.16/4.11 points**. "
            "Restricting B to its 925 three-segment episodes gives **6.94/4.32**. Its rare very "
            "large label counts are not driving the main discrepancy.\n\n"
            "Within a common 40–80-second duration band, A still gains **18.67/22.34 points** "
            "(521 episodes) while B gains **2.61/2.08** (491/492 scorable episodes). "
            "This is a sensitivity slice, not matched task difficulty or a causal adjustment. "
            "No new significance claims are made from these post-hoc cuts.",
            "analysis",
        ),
        (
            "bug",
            "5. Legacy fitting and collision behavior need attention",
            "**Seven of A’s 32 frozen fits used incomplete seed outputs that the corrected fitter "
            "rejects.** All seven are bag-placement fits; 26 of 280 recorded camera/seed rows "
            "are affected. The historical fitter pairs prediction and truth by list position up "
            "to the shorter list, so a missing interior label can assign an offset to the wrong "
            "semantic boundary. All 32 files reproduce under the historical code, whereas only "
            "25 reproduce under the corrected fitter: this is a confirmed version difference, "
            "not unexplained file corruption. No overlap between recorded A fitting IDs and its "
            "evaluation IDs was found.\n\n"
            "This bug does not explain away A’s full advantage: excluding all four bag-placement "
            "components leaves **14.19/14.87-point** gains, still larger than B’s 6.16/4.03. "
            "A fresh, protocol-matched rerun is needed before presenting A’s historical values "
            "as a clean benchmark for the corrected fitter.\n\n"
            "On B, offsets-only creates boundary collisions that drop 93 labels across 92 "
            "single-camera trajectories and 82 labels across 80 stacked-camera trajectories. "
            "Full calibration drops only three/seven labels. This is a concrete reason that "
            "unconstrained independent shifts are risky on B. Normalizing duration fractions "
            "changes B@3 by only +0.62/−0.71 points; neither corrected test is significant. "
            "Normalization alone is not supported as a general repair.",
            None,
        ),
        (
            "video",
            "6. Visual-domain differences are plausible, but their causal effect is unmeasured",
            "The inspected A video metadata is 640×480 at 50 fps; B’s study metadata is "
            "320×180 at 10 fps. Both evaluation configurations request two sampled frames per "
            "second and up to 300 frames; the source-fps difference should not be described as "
            "a fivefold difference in frames actually shown. The stacked views also differ: "
            "wrist/right for A versus wrist/primary for B.\n\n"
            "These differences may affect timestamp errors, but we have no matched-resolution, "
            "matched-view experiment. They cannot by themselves explain why a template derived "
            "only from human timings is substantially less accurate on B. Rank them below the "
            "directly measured timing and offset instability, not as a proven root cause.",
            "video",
        ),
        (
            "method",
            "Method and validation",
            "For boundary j in episode i, the fitted offset is the median of "
            "(predicted start − human start)/predicted episode duration over permitted seeds. "
            "The solver combines distance from the offset-corrected prediction with a penalty "
            "for deviating from seed-median segment durations, on a 0.1-second monotone grid. "
            "Weights and gates come from seed leave-one-out scores.\n\n"
            "For timing diagnostics, a human boundary’s relative position is "
            "(boundary − first human start)/(last human end − first human start). "
            "Within-cohort dispersion is the mean absolute deviation from the evaluation "
            "cohort’s median position, averaged across internal boundaries. We report medians "
            "of those cohort means. Offset shift is the mean absolute difference between frozen "
            "seed offsets and evaluation median signed errors, again normalized by duration. "
            "Evaluation medians are diagnostic measurements only. They never enter prediction "
            "generation, calibration fitting, weight selection, or gate selection.\n\n"
            "All 1,274 A baseline camera/episode B@3 scores were reproduced with the current "
            "scorer, including failures. All 32 historical A fits were reconstructed with the "
            "recorded historical fitter; the earlier B study reproduced all 67 retained fits "
            "with the corrected fitter. Source hashes, per-cohort statistics, legacy validation "
            "details, and episode-level replay results are preserved beside this report.",
            None,
        ),
        (
            "next",
            "Recommended next experiments",
            "1. **Make the cross-corpus comparison genuinely protocol-matched.** Decide whether "
            "task means component or broader family; allocate ten total accordingly, use the "
            "same corrected fitter, gate policy, and raw-response replay on A and B.\n"
            "2. **Test safer offsets on B.** Compare full calibration, zero-offset duration "
            "regularization, and seed-selected offset shrinkage with monotone/minimum-duration "
            "constraints. Cache one raw indexed response per trajectory; select every weight "
            "only within the original ten seeds.\n"
            "3. **Test the pooling assumption.** Use label/phase information available at inference "
            "to separate semantically different positional patterns or decline unsupported ones. "
            "Keep the ten-seed budget; report the resulting accuracy–coverage tradeoff.\n"
            "4. **Measure annotation uncertainty and seed sensitivity.** Compare nested 3/5/10 "
            "subsets of the same ten seeds. A separate duplicate-annotation audit can estimate "
            "human agreement, but those audit annotations must not enter fitting or tuning.",
            None,
        ),
        (
            "questions",
            "What remains unresolved",
            "We cannot uniquely apportion B’s instability among true trajectory variation, "
            "annotation convention/noise, chronological seed-to-evaluation shift, and VLM "
            "stochasticity. Independent duplicate labels, collection/session metadata, repeat "
            "inference, and an exact raw-response replay would distinguish those explanations. "
            "We also have not measured the causal effect of lower resolution or the exact gain "
            "under a ten-per-broader-family A protocol. The comparative evidence supports "
            "weaker timing predictability and offset transfer with high confidence; it does "
            "not justify assigning a percentage of the total gap to each correlated cause.",
            None,
        ),
    ]
    blocks = [{"id": "title", "type": "markdown", "body": "# " + title}]
    charts, datasets = [], {}
    for key, heading, body, source in sections:
        block = {"id": key, "type": "markdown", "body": "## " + heading + "\n\n" + body}
        if source:
            block["sourceId"] = source
        blocks.append(block)
        if key == "offset":
            for camera in ["single", "stacked"]:
                data = []
                for corpus in ["A", "B"]:
                    arms = s["offline"][corpus][camera]
                    for variant, name in [
                        ("offsets", "Offsets"),
                        ("duration", "Duration"),
                        ("full", "Full"),
                        ("normalized", "Normalized"),
                    ]:
                        data.append(
                            {
                                "corpus": "Corpus " + corpus,
                                "method": name,
                                "gain_pp": 100
                                * (arms[variant]["b_hit@3"] - arms["baseline"]["b_hit@3"]),
                                "baseline_B3": arms["baseline"]["b_hit@3"],
                                "variant_B3": arms[variant]["b_hit@3"],
                                "n_scored": arms[variant]["scored"],
                                "camera": camera,
                                "placement": arms[variant]["placed_fraction"],
                            }
                        )
                datasets[camera] = data
                charts.append(
                    {
                        "id": camera,
                        "title": camera.capitalize() + "-camera calibration gains",
                        "subtitle": "Paired offline change in B@3, percentage points; tasks weighted equally",
                        "type": "bar",
                        "dataset": camera,
                        "sourceId": "analysis",
                        "valueFormat": "number",
                        "encodings": {
                            "x": {"field": "method", "type": "nominal", "label": "Ablation"},
                            "y": {
                                "field": "gain_pp",
                                "type": "quantitative",
                                "label": "B@3 gain (pp)",
                            },
                            "color": {"field": "corpus", "type": "nominal", "label": "Corpus"},
                        },
                    }
                )
                blocks.append({"id": camera + "-chart", "type": "chart", "chartId": camera})
                blocks.append(
                    {
                        "id": camera + "-interpretation",
                        "type": "markdown",
                        "sourceId": "analysis",
                        "body": "The paired "
                        + camera
                        + "-camera comparison shows a much larger gain on A "
                        "for both offsets and duration regularization. The bars isolate postprocessing "
                        "on fixed predictions within each corpus; differences in task composition, "
                        "fitting version, and seed grouping remain cross-corpus confounders.",
                    }
                )
    sources = [
        {
            "id": "analysis",
            "label": "Cross-corpus diagnostic calculations",
            "path": "evaluation/results/corpus_calibration_diagnosis/summary.json",
        },
        {
            "id": "protocol",
            "label": "Historical study scope and contamination disclosure",
            "path": "evaluation/alignment_plan.md",
        },
        {
            "id": "video",
            "label": "Verified source video metadata",
            "path": "evaluation/results/corpus_calibration_diagnosis/video_metadata.json",
        },
    ]
    # The portable chart contract requires executed SQL provenance. Compute
    # the actual plotted means from episode rows, independently of summary.json.
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute(
        "CREATE TABLE offline_episode_results "
        "(corpus TEXT,camera TEXT,task TEXT,variant TEXT,scored INT,b3 REAL,placement REAL)"
    )
    for corpus, path in [
        ("A", OUT / "corpus_a_offline_rows.jsonl.gz"),
        ("B", ROOT / "evaluation/results/corpus_b_paired_ablations/episode_results.jsonl.gz"),
    ]:
        with gzip.open(path, "rt") as stream:
            for line in stream:
                r = json.loads(line)
                db.execute(
                    "INSERT INTO offline_episode_results VALUES (?,?,?,?,?,?,?)",
                    (
                        corpus,
                        r["camera"],
                        r["task"],
                        r["variant"],
                        r["scored"],
                        r["b_hit@3"],
                        r["placed_fraction"],
                    ),
                )
    for chart in charts:
        camera = chart["dataset"]
        sql = f"""WITH task_means AS (
 SELECT corpus,camera,task,variant,AVG(b3) AS b3,AVG(placement) AS placement,COUNT(*) AS n
 FROM offline_episode_results WHERE scored=1 AND camera='{camera}'
 GROUP BY corpus,camera,task,variant
), arm_means AS (
 SELECT corpus,camera,variant,AVG(b3) AS b3,AVG(placement) AS placement,SUM(n) AS n
 FROM task_means GROUP BY corpus,camera,variant
)
SELECT 'Corpus '||a.corpus AS corpus,a.camera,
 CASE a.variant WHEN 'offsets' THEN 'Offsets' WHEN 'duration' THEN 'Duration'
 WHEN 'full' THEN 'Full' WHEN 'normalized' THEN 'Normalized' END AS method,
 100*(a.b3-b.b3) AS gain_pp,b.b3 AS baseline_B3,a.b3 AS variant_B3,
 a.n AS n_scored,a.placement
FROM arm_means a JOIN arm_means b ON a.corpus=b.corpus AND a.camera=b.camera
WHERE b.variant='baseline' AND a.variant!='baseline'
ORDER BY a.corpus,CASE a.variant WHEN 'offsets' THEN 1 WHEN 'duration' THEN 2
WHEN 'full' THEN 3 ELSE 4 END;"""
        actual = [dict(r) for r in db.execute(sql)]
        expected = {(r["corpus"], r["method"]): r for r in datasets[camera]}
        for r in actual:
            assert abs(r["gain_pp"] - expected[r["corpus"], r["method"]]["gain_pp"]) < 1e-10
        datasets[camera] = actual
        source_id = camera + "_sql"
        chart["sourceId"] = source_id
        (OUT / (source_id + ".sql")).write_text(sql + "\n")
        sources.append(
            {
                "id": source_id,
                "label": camera.capitalize() + " camera episode-level aggregation",
                "path": "evaluation/results/corpus_calibration_diagnosis/" + source_id + ".sql",
                "query": {
                    "engine": "sqlite",
                    "language": "sql",
                    "sql": sql,
                    "tables_used": ["offline_episode_results"],
                    "description": "Episode rows imported from corpus_a_offline_rows.jsonl.gz and corpus_b_paired_ablations/episode_results.jsonl.gz; averages retain scored failures, average within task, then equally across tasks. Upstream experiment provenance is recorded in both result directories.",
                },
            }
        )
    db.close()
    now = datetime.now(UTC).isoformat()
    payload = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "generatedAt": now,
            "blocks": blocks,
            "charts": charts,
            "sources": sources,
        },
        "snapshot": {"version": 1, "generatedAt": now, "status": "ready", "datasets": datasets},
        "sources": sources,
    }
    (OUT / "artifact.json").write_text(json.dumps(payload, indent=2) + "\n")
    (OUT / "report_design.json").write_text(
        json.dumps(
            {
                "audience": "technical",
                "delivery": "html",
                "structure": "Technical summary; definitions before evidence; ranked findings; methodology; limitations; next experiments; unresolved questions.",
                "chart_contract": {
                    "question": "Which calibration components transfer across corpora?",
                    "family": "grouped bar",
                    "rows_per_chart": 8,
                    "charts": 2,
                    "repeated_family_reason": "One panel per camera regime preserves identical comparison and scale semantics.",
                    "series": "corpus",
                    "x": "method",
                    "y": "paired B@3 gain in percentage points",
                    "palette": "shared native categorical tokens; corpus legend and method labels provide non-color context",
                    "limitations": "Different task populations and fit protocols; no causal decomposition claimed.",
                },
                "omitted_visuals": "Exact dispersion and shift values are explained in prose; charts focus on the directly comparable ablations.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
