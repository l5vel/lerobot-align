#!/usr/bin/env python
"""Recompute the statistical layer from saved scores, on CPU only.

No annotation, serving, scoring or encoder is invoked. To apply new span metric
definitions, first score saved predictions with score_runs.py and an explicit
matcher/threshold; this script reports the metric versions actually present.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import jsonl_io
from reliability import decomposition
from contrast_policy import contrast_policy

HERE = Path(__file__).resolve().parent


def run(script: str, args: list, log: Path | None = None, diagnostic: bool = False) -> None:
    result = subprocess.run(
        [sys.executable, str(HERE / script), *map(str, args)],
        env=os.environ | {"CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True,
        check=False,
    )
    if log:
        log.write_text(result.stdout, encoding="utf-8")
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")
    # Exit 1 can also be a Python exception. Accept it only when the producer
    # reached its explicit, post-publication diagnostic outcome.
    completed_diagnostic = diagnostic and result.returncode == 1 and (
        (script == "c1_gate.py" and "C1 verdict:" in result.stdout and "->  gate BLOCK" in result.stdout)
        or (script == "make_report.py" and "[report] wrote" in result.stderr
            and "numbers are diagnostic only." in result.stderr)
    )
    if result.returncode != 0 and not completed_diagnostic:
        raise RuntimeError(f"{script} failed ({result.returncode}): {result.stdout}")


def score_manifest(path: Path) -> dict:
    digest = hashlib.sha256()
    versions: Counter = Counter()
    rows = 0
    # Hash the decompressed bytes, not the container, so a manifest stays
    # comparable whether the scores are stored .jsonl or .jsonl.gz.
    for line in jsonl_io.iter_lines(path):
        digest.update(line)
        row = json.loads(line)
        versions[str(row.get("metric_version", "legacy-unversioned"))] += 1
        rows += 1
    return {"sha256": digest.hexdigest(), "rows": rows, "metric_versions": dict(versions)}


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--results", type=Path, default=HERE.parent / "results")
    parser.add_argument("--analysis", type=Path, default=HERE.parent / "analysis")
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--reuse-aggregates", action="store_true",
                        help="Only finalize already regenerated aggregates; do not repeat their bootstrap.")
    parser.add_argument("--before-dir", type=Path, default=None,
                        help="Earlier snapshot of evaluation/ for an exhaustive numeric change ledger.")
    parser.add_argument("--metric-rescoring-note", default="Not performed; statistical refresh uses saved metric values.")
    args = parser.parse_args()
    args.analysis.mkdir(parents=True, exist_ok=True)
    if args.before_dir is None:
        args.before_dir = Path(tempfile.mkdtemp(prefix="evaluation-before-"))
        for directory in (args.results, args.analysis):
            dest = args.before_dir / directory.name
            dest.mkdir()
            for path in directory.iterdir():
                if path.is_file() and path.suffix in (".json", ".md", ".txt"):
                    shutil.copy2(path, dest / path.name)

    if not args.reuse_aggregates:
        for matcher in ("exact", "embedding"):
            for c4 in (False, True):
                stem = f"report_{'c4_' if c4 else ''}{matcher}"
                print(f"[regenerate] {stem}", flush=True)
                run("aggregate.py", ["--scores", args.results / "scores.jsonl", "--out", args.results / f"{stem}.json",
                                     "--matcher", matcher, "--contrast-mode", "global" if c4 else "cell",
                                     "--baseline", "ref_script_prior" if c4 else "baseline_upstream",
                                     "--n-boot", args.n_boot], log=args.results / f"{stem}.txt")
        run("breakdown.py", ["--scores", args.results / "scores.jsonl", "--out", args.results / "breakdown.json"])
    else:
        for matcher in ("exact", "embedding"):
            report = json.loads((args.results / f"report_{matcher}.json").read_text())
            if "holm_family_size" not in report or not any("[C3]" in k for k in report["contrasts"]):
                raise RuntimeError("--reuse-aggregates requires corrected aggregates including C3")

    # Refresh metadata-only caveats without resampling an already computed CI.
    import yaml
    metadata = {a["name"]: a for a in yaml.safe_load((HERE.parent / "configs" / "arms.yaml").read_text())["arms"]}
    for path in args.results.glob("report*.json"):
        report = json.loads(path.read_text())
        for key in report.get("supervision_asymmetry", {}):
            arm = report.get("contrast_arms", {}).get(key, key)
            baseline = report["contrast_baselines"][key]
            _, warning = contrast_policy(arm, baseline, metadata, allow_reference=True)
            if not warning:
                raise RuntimeError(f"supervision metadata changed for {key}; regenerate aggregates")
            report["supervision_asymmetry"][key] = warning
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")

    # Certify all available cells, not just the primary alias. A BLOCK is an
    # experimental result and must be published, while script errors still fail.
    primary = json.loads((args.results / "c1_gate.json").read_text())
    primary_key = (primary["baseline"], primary["arm"])
    cell_paths = sorted(args.results.glob("c1_gate__*.json"))
    if not cell_paths:
        cell_paths = [args.results / "c1_gate.json"]
    for path in cell_paths:
        previous = json.loads(path.read_text())
        fingerprint = previous["environment_fingerprint"]
        print(f"[regenerate] {path.name}", flush=True)
        run("c1_gate.py", ["--scores", args.results / "e1_scores.jsonl", "--out", path,
                           "--baseline", previous["baseline"], "--arm", previous["arm"],
                           "--model-id", fingerprint["model_id"], "--matcher", previous["matcher"],
                           "--threshold", fingerprint["threshold"], "--n-boot", args.n_boot], diagnostic=True)
        current = json.loads(path.read_text())
        if "resampling_unit" not in current:
            raise RuntimeError(f"gate regeneration did not produce a corrected artifact: {path}")
        if (current["baseline"], current["arm"]) == primary_key and path.name != "c1_gate.json":
            shutil.copy2(path, args.results / "c1_gate.json")

    reliability = json.loads((args.results / "reliability.json").read_text())
    (args.results / "reliability_decomposition.json").write_text(json.dumps(decomposition(reliability), indent=2) + "\n")
    manifest = {
        "scope": "statistical reports only; no GPU, annotation or encoder calls",
        "metric_rescoring": args.metric_rescoring_note,
        "scores": score_manifest(args.results / "scores.jsonl"),
        "e1_scores": score_manifest(args.results / "e1_scores.jsonl"),
        "n_boot": args.n_boot,
        "breakdown_inference": "exploratory pointwise intervals; no p-values",
    }
    (args.results / "regeneration.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for matcher in ("embedding", "exact"):
        out = args.analysis / ("report.md" if matcher == "embedding" else "report_exact.md")
        run("make_report.py", ["--results", args.results, "--out", out, "--matcher", matcher], diagnostic=True)
        if "Repeated runs are averaged within episode" not in out.read_text():
            raise RuntimeError(f"report regeneration failed: {out}")
    run("summarize_corrections.py", ["--results", args.results, "--analysis", args.analysis,
                                     "--before-dir", args.before_dir])
    print(f"[regenerate] finished; gate BLOCKs and pending metric rescoring are reported in {args.analysis}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
