"""Opt-in integration check against an already running visual model.

Only original synthetic media is sent. This checks compatibility and dataset
integrity, not annotation quality. No model server or paid Job is provisioned.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pyarrow.parquet as pq

from tests.fixtures import add_synthetic_video, build_annotation_dataset


def check_numbers(value):
    if isinstance(value, float):
        assert math.isfinite(value), value
    elif isinstance(value, dict):
        for child in value.values():
            check_numbers(child)
    elif isinstance(value, list):
        for child in value:
            check_numbers(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--video-metadata-source", choices=("server", "client"), default="server")
    parser.add_argument("--output", type=Path, required=True, help="New directory for logs and report")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    with tempfile.TemporaryDirectory(prefix="align-live-vlm-") as tmp:
        for mode, frame_format in ((mode, fmt) for mode in ("generation", "alignment")
                                   for fmt in ("contact_sheet", "video")):
            case = f"{mode}-{frame_format}"
            root = add_synthetic_video(build_annotation_dataset(
                Path(tmp) / case,
                episode_specs=[(0, 30, "Move the red square to the right.")], fps=10,
            ))
            shard = next((root / "data").rglob("*.parquet"))
            before = pq.read_table(shard)
            command = [
                str(Path(sys.executable).parent / "lerobot-align"),
                f"--root={root}", f"--vlm.model_id={args.model}",
                "--vlm.auto_serve=false", f"--vlm.api_base={args.api_base}",
                f"--vlm.video_metadata_source={args.video_metadata_source}",
                "--vlm.max_new_tokens=1024", "--vlm.temperature=0",
                "--video_backend=pyav", "--plan.n_task_rephrasings=0",
                f"--plan.subtask_generate_frame_format={frame_format}",
                "--plan.subtask_video_fallback=error",
                "--interjections.max_interjections_per_episode=1",
                "--interjections.interjection_min_t=0.5", "--vqa.K=1",
            ]
            if mode == "alignment":
                labels = Path(tmp) / "ordered-labels.json"
                labels.write_text(json.dumps(["Start moving the red square right",
                                              "Continue moving the red square right"]))
                command.extend([f"--plan.subtasks_path={labels}",
                                f"--plan.subtask_align_frame_format={frame_format}"])
            result = subprocess.run(command, capture_output=True, text=True, timeout=600, env={
                **os.environ, "HF_HUB_OFFLINE": "1", "HF_HOME": str(Path(tmp) / "hf"),
                "HF_DATASETS_CACHE": str(Path(tmp) / "hf/datasets"),
                "HF_LEROBOT_HOME": str(Path(tmp) / "hf/lerobot"),
                "LEROBOT_OPENAI_SEND_MM_KWARGS": "1",
            })
            (args.output / f"{case}.log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, f"See {args.output / (case + '.log')}"
            after = pq.read_table(shard)
            assert after.num_rows == before.num_rows
            for key in set(before.column_names) - {"subtask_index"}:
                assert after[key].equals(before[key]), key
            events = [atom for row in after["language_events"].to_pylist() for atom in row]
            persistent = [atom for row in after["language_persistent"].to_pylist() for atom in row]
            check_numbers(events + persistent)
            assert "vqa" in {atom.get("style") for atom in events}
            assert any(atom.get("tool_calls") for atom in events)
            assert {"subtask", "plan"} <= {atom.get("style") for atom in persistent}
            # A legitimate one-subtask answer has no later boundary at which to
            # generate memory or an interjection. Verify those modules ran;
            # the deterministic smoke separately requires their emitted atoms.
            for module in ("plan", "interjections", "vqa"):
                assert f"phase={module} processed=1 skipped=0" in result.stderr + result.stdout
            info = json.loads((root / "meta/info.json").read_text())
            assert {"language_events", "language_persistent"} <= info["features"].keys()
            assert not (root / ".lerobot-align-transaction").exists()
            reports.append({"mode": mode, "format": frame_format, "rows": after.num_rows,
                            "all_modules": True, "trajectory_columns_unchanged": True,
                            "finite_annotations": True, "transaction_completed": True,
                            "styles": sorted({a["style"] for a in events + persistent if a.get("style")})})
            (args.output / "report.json").write_text(json.dumps({
                "model": args.model, "quality_assessed": False, "checks": reports,
            }, indent=2) + "\n")
            print(f"{case}: live visual model, all modules and dataset integrity passed", flush=True)


if __name__ == "__main__":
    main()
