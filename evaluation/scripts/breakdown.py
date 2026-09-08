#!/usr/bin/env python
"""Per-cell quality, and paired camera / profile contrasts within each tool.

The 28-arm grid is a factorial of tool x profile x camera, and the interesting
questions are not only "which tool" but "does the view matter" and "does the
frame budget matter". Both are asked here as PAIRED contrasts.

Pairing is not cosmetic. The arms in this study do not all publish the same
episodes -- see reliability.json -- so an unpaired camera mean would compare
`wrist` on the episodes wrist survived against `left` on the episodes left
survived, and attribute the corpus difference to the camera. Every contrast
below is therefore restricted to the episodes present in BOTH sides of that
contrast, and the restricted n is reported next to it.

    python breakdown.py --scores results/scores.jsonl.gz --out results/breakdown.json
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jsonl_io
from contrast_policy import contrast_policy
from metrics.stats import cluster_bootstrap_mean

HEADLINE = ("boundary_f1@0p5", "boundary_f1@1", "macro_iou")
# Segment-count diagnostics: `log_ratio` = log(n_pred / n_true), so a positive
# mean is over-segmentation and a negative one under-segmentation. It is logged
# because the raw ratio is asymmetric (2x too many = 2.0, half as many = 0.5).
DIAGNOSTIC = ("log_ratio", "abs_count_error", "n_pred", "n_true", "edit_score", "label_f1")
ALL_METRICS = HEADLINE + DIAGNOSTIC


def split_arm(arm: str) -> tuple[str, str, str] | None:
    parts = arm.split("__")
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else None


def load(scores: Path, matcher: str) -> dict[str, dict[tuple[str, int], dict]]:
    """arm -> (dataset, episode) -> metrics, for scorable rows only."""
    out: dict[str, dict[tuple[str, int], dict]] = collections.defaultdict(dict)
    with jsonl_io.open_text(scores) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("matcher") != matcher or not row.get("ok") or row.get("repeat"):
                continue
            out[row["arm"]][(row["dataset"], row["episode"])] = row
    return out


def summarise(rows: dict[tuple[str, int], dict]) -> dict:
    by_cluster: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for (dataset, _), row in rows.items():
        for metric in ALL_METRICS:
            value = row.get(metric)
            if value is not None:
                by_cluster[metric][dataset].append(float(value))
    summary = {"n_episodes": len(rows), "n_datasets": len({d for d, _ in rows})}
    for metric in ALL_METRICS:
        clusters = by_cluster.get(metric)
        if not clusters:
            summary[metric] = None
            continue
        interval = cluster_bootstrap_mean(dict(clusters), n_boot=2000, seed=17)
        summary[metric] = {"point": interval.point, "ci_low": interval.low, "ci_high": interval.high}
    return summary


def contrast(a: dict[tuple[str, int], dict], b: dict[tuple[str, int], dict],
             *, baseline: str, arm: str, metadata: dict,
             allow_unequalised: bool = False) -> dict:
    """b - a, over the episodes both sides actually produced."""
    reason, _ = contrast_policy(arm, baseline, metadata, allow_unequalised=allow_unequalised)
    if reason:
        return {"unavailable": True, "reason": reason}
    shared = sorted(set(a) & set(b))
    result = {"n_paired": len(shared), "n_a_only": len(set(a) - set(b)), "n_b_only": len(set(b) - set(a))}
    if not shared:
        return result | dict.fromkeys(HEADLINE)
    for metric in HEADLINE + ("log_ratio",):
        deltas: dict[str, list[float]] = collections.defaultdict(list)
        for key in shared:
            va, vb = a[key].get(metric), b[key].get(metric)
            if va is not None and vb is not None:
                deltas[key[0]].append(float(vb) - float(va))
        if not deltas:
            result[metric] = None
            continue
        interval = cluster_bootstrap_mean(dict(deltas), n_boot=2000, seed=17)
        result[metric] = {
            **interval.as_dict(), "exploratory": True,
            "improves_baseline": interval.point > 0 if metric in HEADLINE else None,
        }
    # These additional factorial analyses are exploratory. Intervals are
    # pointwise, not simultaneous; no p-values or significance claims are made.
    return result


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, )
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--matcher", default="embedding")
    ap.add_argument("--reference-camera", default="wrist")
    ap.add_argument("--reference-profile", default="wrap")
    ap.add_argument("--reference-tool", default="baseline_upstream")
    ap.add_argument("--arms-config", type=Path,
                    default=Path(__file__).resolve().parents[1] / "configs" / "arms.yaml")
    args = ap.parse_args()
    import yaml
    metadata = {a["name"]: a for a in yaml.safe_load(args.arms_config.read_text())["arms"]}

    by_arm = load(args.scores, args.matcher)
    cells = {arm: split_arm(arm) for arm in by_arm}
    cells = {arm: cell for arm, cell in cells.items() if cell}

    report: dict = {
        "matcher": args.matcher, "cells": {},
        "inference": "exploratory; pointwise 95% intervals; no p-values or significance claims",
        "n_boot": 2000,
        "camera_contrasts": {}, "profile_contrasts": {}, "tool_contrasts": {},
    }

    for arm, (tool, profile, camera) in sorted(cells.items()):
        report["cells"][arm] = {"tool": tool, "profile": profile, "camera": camera} | summarise(by_arm[arm])

    index = {cell: arm for arm, cell in cells.items()}

    # Camera: hold tool and profile fixed, vary the view.
    for (tool, profile, camera), arm in sorted(index.items()):
        if camera == args.reference_camera:
            continue
        ref = index.get((tool, profile, args.reference_camera))
        if ref is None:
            continue
        key = f"{tool}__{profile}: {camera} vs {args.reference_camera}"
        report["camera_contrasts"][key] = contrast(by_arm[ref], by_arm[arm], baseline=ref, arm=arm,
                                                     metadata=metadata, allow_unequalised=True)

    # Profile: hold tool and camera fixed, vary the frame budget.
    for (tool, profile, camera), arm in sorted(index.items()):
        if profile == args.reference_profile:
            continue
        ref = index.get((tool, args.reference_profile, camera))
        if ref is None:
            continue
        key = f"{tool}__{camera}: {profile} vs {args.reference_profile}"
        report["profile_contrasts"][key] = contrast(by_arm[ref], by_arm[arm], baseline=ref, arm=arm,
                                                     metadata=metadata, allow_unequalised=True)

    # Tool: hold profile and camera fixed, vary the tool. This is the C2/C3
    # question asked within a cell, which is the only place it can be asked
    # honestly -- the two tools are only equalised within a cell.
    for (tool, profile, camera), arm in sorted(index.items()):
        if tool == args.reference_tool:
            continue
        ref = index.get((args.reference_tool, profile, camera))
        if ref is None:
            # Stacked views have no upstream counterpart: the upstream tool
            # takes a single camera_key and cannot present synchronised views,
            # so there is nothing to contrast against and inventing one would
            # mean comparing across cells.
            report["tool_contrasts"][f"{tool}__{profile}__{camera}"] = {
                "unavailable": True,
                "reason": f"no {args.reference_tool} arm in cell ({profile}, {camera})",
            }
            continue
        key = f"{profile}__{camera}: {tool} vs {args.reference_tool}"
        report["tool_contrasts"][key] = contrast(by_arm[ref], by_arm[arm], baseline=ref, arm=arm, metadata=metadata)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(entry):
        return "--" if not entry else f"{entry['point']:.3f}"

    print(f"=== per-cell means ({args.matcher} matcher; n = episodes that scored) ===")
    print(f"{'arm':<38}{'n':>6}{'bF1@0.5':>9}{'bF1@1':>8}{'mIoU':>7}{'log(np/nt)':>11}{'edit':>7}")
    for arm, cell in sorted(report["cells"].items()):
        print(
            f"{arm:<38}{cell['n_episodes']:>6}{fmt(cell['boundary_f1@0p5']):>9}"
            f"{fmt(cell['boundary_f1@1']):>8}{fmt(cell['macro_iou']):>7}"
            f"{fmt(cell['log_ratio']):>11}{fmt(cell['edit_score']):>7}"
        )

    def delta(entry, metric):
        cell = entry.get(metric)
        if not cell:
            return "--", "--"
        return f"{cell['point']:+.3f}", f"[{cell['ci_low']:+.3f},{cell['ci_high']:+.3f}]"

    for title, block in (("tool", report["tool_contrasts"]),
                         ("camera", report["camera_contrasts"]),
                         ("profile", report["profile_contrasts"])):
        print(f"\n=== {title} contrasts (EXPLORATORY; paired; pointwise intervals) ===")
        print(f"{'contrast':<46}{'paired':>7}{'d bF1@0.5':>11}{'95% CI':>17}{'d mIoU':>9}{'95% CI':>17}")
        for key, entry in sorted(block.items()):
            if entry.get("unavailable"):
                print(f"{key:<46}{'--':>7}   {entry['reason']}")
                continue
            boundary, boundary_p = delta(entry, "boundary_f1@0p5")
            iou, iou_p = delta(entry, "macro_iou")
            print(
                f"{key:<46}{entry['n_paired']:>7}{boundary:>11}{boundary_p:>17}{iou:>9}{iou_p:>17}"
            )

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
