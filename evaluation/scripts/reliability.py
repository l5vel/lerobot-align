#!/usr/bin/env python
"""Per-arm reliability: how much requested work each arm actually delivered.

Scoring only ever sees episodes an arm managed to publish, so an arm that
crashes on hard components looks *better* on every quality metric than one that
answers everywhere. This table is the correction: it reports the denominator the
quality tables are missing, and attributes each lost job to the exception that
ended it.

The unit that matters is the JOB (one dataset x one arm, ~40 episodes). All
three failure modes seen here abort the whole job, so a single bad episode can
void ~40 good ones; `waste` counts episodes that were annotated successfully and
discarded anyway.

    python reliability.py --predictions-dir DIR --out results/reliability.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

# Terminal exceptions, mapped to a short cause. Ordered: first match wins.
FAILURE_MODES = (
    ("timeout", re.compile(r"TimeoutExpired:")),
    ("staging_validation",
     re.compile(r"Staging validation failed: checked=(\d+) errors=(\d+)")),
    ("realign_label_loss",
     re.compile(r"realignment did not preserve the exact ordered generated label list")),
    ("align_min_fraction",
     re.compile(r"below subtask_align_min_fraction")),
)


def classify(log_path: Path) -> tuple[str, int | None, int | None]:
    """Return (mode, episodes_checked, episodes_erroring) for a failed job."""
    if not log_path.is_file():
        return "no_log", None, None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    for name, pattern in FAILURE_MODES:
        match = pattern.search(text)
        if match:
            if match.groups():
                return name, int(match.group(1)), int(match.group(2))
            return name, None, None
    if "Traceback" in text:
        # Unclassified crash: report the exception line rather than hiding it in
        # an "other" bucket, so a new failure mode cannot pass unnoticed.
        lines = [ln for ln in text.splitlines() if re.match(r"^\w+(\.\w+)*(Error|Exception):", ln)]
        return (lines[-1].split(":")[0] if lines else "unknown_traceback"), None, None
    return "no_traceback", None, None


def decomposition(report: dict) -> dict:
    """Keep matched modes and additional fork capabilities out of one denominator.

    Upstream supports ordinary generation from one camera. The defaults and
    native-video fork arms cover the same six profile/camera cells. All ten
    generate-then-realign arms are an additional mode (four also stack views).
    """
    upstream = {k: v for k, v in report.items() if k.startswith("baseline_upstream__")}
    cells = {k.split("__", 1)[1] for k in upstream}
    matched = {k: v for k, v in report.items()
               if k.split("__")[0] in ("align_defaults", "align_video")
               and k.split("__", 1)[1] in cells}
    additional = {k: v for k, v in report.items() if k.startswith("align_") and k not in matched}
    groups = {"upstream_matched": upstream, "fork_matched_modes": matched,
              "fork_additional_modes": additional}
    out = {}
    for name, group in groups.items():
        row = {key: sum(v[key] for v in group.values()) for key in
               ("jobs", "jobs_ok", "episodes_requested", "episodes_published")}
        row["arms"] = sorted(group)
        row["episode_yield"] = (row["episodes_published"] / row["episodes_requested"]
                                if row["episodes_requested"] else None)
        out[name] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False, )
    ap.add_argument("--predictions-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    arms: dict[str, dict] = collections.defaultdict(
        lambda: {
            "jobs": 0, "jobs_ok": 0,
            "episodes_requested": 0, "episodes_published": 0,
            "episodes_wasted": 0, "seconds": 0.0,
            "modes": collections.Counter(), "failed_components": [],
        }
    )

    for path in sorted(args.predictions_dir.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        arm = payload.get("arm")
        # Reference arms are computed from ground truth, not run; falsify
        # repeats are a separate design and would double-count components.
        if not arm or arm.startswith("ref_") or payload.get("repeat"):
            continue
        row = arms[arm]
        row["jobs"] += 1
        row["episodes_requested"] += payload.get("n_episodes_requested") or 0
        row["episodes_published"] += payload.get("n_episodes_predicted") or 0
        row["seconds"] += payload.get("elapsed_seconds") or 0.0
        if payload.get("ok"):
            row["jobs_ok"] += 1
            continue
        mode, checked, erroring = (("timeout", None, None) if payload.get("timed_out")
                                   else classify(Path(payload.get("log") or "")))
        row["modes"][mode] += 1
        row["failed_components"].append(payload.get("dataset"))
        if checked is not None and erroring is not None:
            # Episodes the tool annotated correctly and then threw away because
            # a sibling episode in the same batch failed validation.
            row["episodes_wasted"] += max(checked - erroring, 0)

    report = {}
    for arm, row in sorted(arms.items()):
        requested = row["episodes_requested"]
        report[arm] = {
            "jobs": row["jobs"],
            "jobs_ok": row["jobs_ok"],
            "job_success_rate": row["jobs_ok"] / row["jobs"] if row["jobs"] else None,
            "episodes_requested": requested,
            "episodes_published": row["episodes_published"],
            "episode_yield": row["episodes_published"] / requested if requested else None,
            "episodes_wasted_by_batch_abort": row["episodes_wasted"],
            "minutes_per_job": row["seconds"] / 60 / row["jobs"] if row["jobs"] else None,
            "failure_modes": dict(row["modes"]),
            "failed_components": sorted(set(row["failed_components"])),
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    width = max(len(a) for a in report) if report else 10
    print(f"{'arm':<{width}} {'jobs ok':>9} {'ep yield':>9} {'wasted':>7} {'min/job':>8}  failure modes")
    for arm, row in report.items():
        modes = ", ".join(f"{k}={v}" for k, v in sorted(row["failure_modes"].items())) or "-"
        print(
            f"{arm:<{width}} {row['jobs_ok']:>4}/{row['jobs']:<4} "
            f"{row['episode_yield']:>8.1%} {row['episodes_wasted_by_batch_abort']:>7} "
            f"{row['minutes_per_job']:>7.1f}  {modes}"
        )
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
