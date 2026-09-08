#!/usr/bin/env python
"""Compatibility entry point. New studies use merge_alignment_predictions.py."""

from pathlib import Path
from merge_alignment_predictions import *  # noqa: F403
from merge_alignment_predictions import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            default_prefix="corpus_b__",
            default_namespace="robointer_original_episode_index",
            default_arms_config=Path(__file__).resolve().parents[1] / "configs/corpus_b_arms.yaml",
        )
    )
