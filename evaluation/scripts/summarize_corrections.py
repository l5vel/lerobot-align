#!/usr/bin/env python
"""Write the source-backed summary and an exhaustive old/new numeric ledger."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from reliability import decomposition

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metrics.stats import holm_correction  # noqa: E402

HEADLINE = ("boundary_f1@0p5", "boundary_f1@1", "macro_iou")
ABSENT = "<absent>"


def numbers(value, path="") -> dict:
    if isinstance(value, dict):
        return {k: v for name, child in value.items()
                for k, v in numbers(child, path + "/" + str(name)).items()}
    if isinstance(value, list):
        return {k: v for i, child in enumerate(value)
                for k, v in numbers(child, path + "/" + str(i)).items()}
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return {path: value}
    return {}


def implication(artifact: str, path: str) -> str:
    if "c1_gate" in artifact:
        return "C1: episode-level repeat resampling; live source check can withhold certification"
    if "breakdown" in artifact:
        return ("Exploratory factorial analysis: significance claim withdrawn" if "/p" in path
                else "Exploratory factorial analysis: pointwise CI or sample accounting")
    if "decomposition" in artifact:
        return "Reliability: matched generation modes separated from additional fork capabilities"
    if "[C3]" in path:
        return "C3: registered realign-minus-video contrast now measured on shared episodes"
    if "p_holm" in path:
        return "Multiplicity: complete tool family now includes registered C3 tests"
    if "paired" in path:
        return "Population: candidate/baseline means use exactly the contrast's paired sample"
    if "ci_" in path:
        return "Uncertainty: ties receive half weight in bootstrap bias correction"
    if "report_c4" in artifact:
        return "C4: model-free floor comparison; supervision asymmetry retained"
    return "Audit metadata or descriptive/comparison statistic; see artifact and path"


def write_ledger(before: Path, results: Path, analysis: Path) -> list[dict]:
    rows = []
    for path in sorted(results.glob("*.json")):
        old_path = before / "results" / path.name
        old = json.loads(old_path.read_text()) if old_path.exists() else {}
        new = json.loads(path.read_text())
        old_nums, new_nums = numbers(old), numbers(new)
        for key in sorted(old_nums.keys() | new_nums.keys()):
            was, now = old_nums.get(key, ABSENT), new_nums.get(key, ABSENT)
            if was != now:
                effect = implication(path.name, key)
                if (key.endswith("/p_holm") and isinstance(was, (int, float))
                        and isinstance(now, (int, float)) and (was < .05) != (now < .05)):
                    effect += "; crosses the 0.05 significance threshold"
                rows.append({"artifact": "results/" + path.name, "path": key,
                                 "published": was, "corrected": now, "conclusion": effect})
    r = json.loads((results / "report_embedding.json").read_text())
    best = max(e["macro_iou"]["point"] for a, e in r["per_arm"].items()
               if not a.startswith("ref_") and "macro_iou" in e)
    groups = decomposition(json.loads((results / "reliability.json").read_text()))
    manual = [
        ("results/breakdown.json", "significant entries in original 180-test family", 79, 0,
         "Full-family Holm rejects none; corrected breakdown is exploratory and makes no significance claims"),
        ("analysis/report.md", "positive profile cells incorrectly marked significantly worse", 16, 0,
         "Direction fallback fixed; exploratory tables now carry no significance markers"),
        ("analysis/deviations.md", "D50 estimable profile comparisons", 12, 11,
         "Eleven profile comparisons have a macro IoU estimate; the additional stack2 comparison has no shared episodes"),
        ("analysis/deviations.md", "D50 significantly better boundary F1 profile comparisons", 7, ABSENT,
         "Withdrawn because the original breakdown used the wrong multiplicity family"),
        ("analysis/summary.md", "best VLM macro IoU", .483, best,
         "Best VLM score improves; still below the model-free prior"),
        ("analysis/summary.md", "fork episode yield comparison", 8062 / 14014,
         groups["fork_matched_modes"]["episode_yield"],
         "Estimand corrected: 22-arm pooled yield replaced by 12 matched generation arms; additional modes reported separately"),
        ("evaluation_plan.md", "floor measured episodes", 797, 635,
         "Floor sample is 635 scored held-out episodes; 797 remains the full annotated corpus"),
        ("scripts/score_runs.py", "default embedding threshold", .6, .93,
         "Standalone scoring now uses the matcher fallback; saved study threshold remains 0.91"),
        ("scripts/make_report.py", "gate cells reported", 1, 2,
         "Both gate cells are visible; second cell was already blocked"),
    ]
    for name in ("README.md", "evaluation_plan.md", "analysis/deviations.md", "metrics/semantic.py"):
        manual.append((name, "hard negatives admitted at threshold 0.60", 78, 70,
                       "Calibration rationale corrected; 78 belongs to threshold 0.56"))
    for artifact, path, was, now, effect in manual:
        rows.append({"artifact": artifact, "path": path, "published": was, "corrected": now, "conclusion": effect})
    out = analysis / "number_changes.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["artifact", "path", "published", "corrected", "conclusion"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_summary(results: Path, analysis: Path) -> None:
    report = json.loads((results / "report_embedding.json").read_text())
    c4 = json.loads((results / "report_c4_embedding.json").read_text())
    floor = json.loads((results / "model_free_floor.json").read_text())
    prior = floor["component_weighted_means"]["ref_script_prior"]
    groups = decomposition(json.loads((results / "reliability.json").read_text()))
    manifest = json.loads((results / "regeneration.json").read_text())
    gates = [json.loads(p.read_text()) for p in sorted(results.glob("c1_gate__*.json"))]
    lines = ["# What the corrected evaluation supports", "",
             "The saved timing scores show some conditional tool differences, but **neither tool demonstrates an advantage over the model-free script prior**. Reliability remains substantially worse for the fork even after separating comparable modes. Camera/profile tables are exploratory.", "",
             "**C1 does not currently authorize a sweep.** The primary cell remains statistically equivalent after correcting repeat resampling, but its live source fingerprint differs from the saved predictions. Its executable gate is BLOCK. The second gate cell is also BLOCK and was missing from the earlier report.", "",
             f"**Regeneration scope:** {manifest['metric_rescoring']} The statistical layers of both exact and embedding reports were refreshed. Saved span metrics affected by defects 10–13 have not been relabelled as newly scored. The three headline timing metrics do not depend on those definitions.", "",
             "## Reliability on comparable modes", "",
             "| Group | Arms | Episodes published / requested | Yield | Jobs completed |",
             "|---|---|---|---|---|"]
    for name, row in groups.items():
        lines.append(f"| {name} | {len(row['arms'])} | {row['episodes_published']}/{row['episodes_requested']} "
                     f"| {row['episode_yield']:.2%} | {row['jobs_ok']}/{row['jobs']} |")
    lines += ["", "The fork's defaults and video generation modes cover the six upstream profile/camera cells. The ten additional arms perform generate-then-realign, including four stacked-camera arms; upstream has no equivalent mode. The old pooled 8062/14014 (57.5%) is not a like-for-like tool comparison.", "",
              "## Quality and the script prior", "",
              "| Metric | Script prior | Best VLM descriptive mean | Arm attaining that mean |",
              "|---|---|---|---|"]
    for metric in HEADLINE:
        value, arm = max((e[metric]["point"], a) for a, e in report["per_arm"].items()
                         if not a.startswith("ref_") and metric in e)
        lines.append(f"| {metric} | {prior[metric]:.4f} | {value:.4f} | {arm} |")
    c4_wins = [(a, m) for a, es in c4["contrasts"].items() if not a.startswith("ref_")
               for m, e in es.items() if m in HEADLINE and e.get("improves_baseline") and e.get("p_holm", 1) < .05]
    lines += ["", f"VLM headline wins against the prior after Holm: **{len(c4_wins)}**. "
              "The earlier 0.483 best-VLM macro IoU was the upstream wrist value; the actual maximum is 0.5224 for `align_video__udef__wrist`. The boundary maxima and IoU maximum belong to different arms. These are descriptive means on each arm's own surviving episodes; the report presents paired means and sample counts beside every contrast.", "",
              "The prior uses labelled seed episodes unavailable to the VLM arms. Its comparison is a supervision-asymmetric floor check, not an isolated perception intervention. That caveat now appears with every affected published comparison.", "",
              "## Tool differences that survive the complete Holm family", "",
              f"The family contains {report['holm_family_size']} tests, including the previously absent registered C3 comparisons. All results below are conditional on shared surviving episodes and the gate limitations above.", "",
              "| Comparison | Metric | Difference | Paired episodes / datasets | Holm p | Direction |",
              "|---|---|---|---|---|---|"]
    for arm, entries in report["contrasts"].items():
        for metric in HEADLINE:
            e = entries.get(metric, {})
            if e.get("p_holm", 1) < .05:
                baseline = report["contrast_baselines"][arm]
                direction = "better" if e["improves_baseline"] else "worse"
                lines.append(f"| {arm} vs {baseline} | {metric} | {e['point']:+.4f} "
                             f"| {e['n_paired_episodes']} / {e['n_clusters']} | {e['p_holm']:.4f} | {direction} |")
    lines += ["", "## Camera, frame budget, and gates", "",
              "All 79 previous breakdown significance markers are withdrawn. The old per-contrast correction covered four metrics while the script ran 180 tests; applying Holm to that full old family leaves zero significant entries. The corrected tables retain every eligible metric and paired pointwise interval, explicitly as exploratory results. Camera and profile patterns can suggest follow-up work, but the earlier confirmed-effect claims do not stand.", "",
              "| Gate cell | Statistical verdict | Current executable gate |",
              "|---|---|---|"]
    for gate in gates:
        lines.append(f"| {gate['arm']} vs {gate['baseline']} | {gate.get('statistical_verdict', gate['verdict'])} | {gate['gate']} |")
    lines += ["", "Repeated measurements are averaged within episode: the primary C1 cell contributes 119 paired episodes, not 357 independent units. Its statistical equivalence survives; the live gate is withheld because current tool source is different. Untested cells have no C1 certification.", "",
              "## Limits", "",
              "The corpus is scripted, has four task families and no measured inter-annotator floor, and was used to develop fork settings. Attrition is non-random. No conditional quality comparison establishes performance on failed episodes. Batch-fatal staging, realignment label-loss and alignment-fraction failures live in the tool under test and were left unchanged.", "",
              "See [the generated report](report.md), [the exact-matcher report](report_exact.md), [the defect audit](harness_corrections.md), and [every changed number](number_changes.csv).", ""]
    (analysis / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--before-dir", type=Path, required=True)
    args = parser.parse_args()
    old = json.loads((args.before_dir / "results" / "breakdown.json").read_text())
    tests = {f"{block}/{key}/{m}": e["p"]
             for block in ("camera_contrasts", "profile_contrasts", "tool_contrasts")
             for key, entry in old[block].items() for m, e in entry.items()
             if isinstance(e, dict) and "p" in e}
    adjusted = holm_correction(tests)
    if tests:
        audit = {"legacy_tests": len(tests),
                 "legacy_significant_after_full_family_holm": sum(p < .05 for p in adjusted.values()),
                 "smallest_legacy_full_family_p_holm": min(adjusted.values())}
        (args.results / "breakdown_family_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    write_summary(args.results, args.analysis)
    rows = write_ledger(args.before_dir, args.results, args.analysis)
    print(f"wrote summary and {len(rows)} numeric changes -> {args.analysis / 'number_changes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
