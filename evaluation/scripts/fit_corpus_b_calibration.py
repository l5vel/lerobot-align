#!/usr/bin/env python
"""Compatibility entry point. New studies use fit_task_calibration.py."""

from pathlib import Path
from fit_task_calibration import *  # noqa: F403
from fit_task_calibration import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_prefix="corpus_b__",
            default_namespace="robointer_original_episode_index",
            default_arms_config=Path(__file__).resolve().parents[1] / "configs/corpus_b_arms.yaml",
        )
    )
