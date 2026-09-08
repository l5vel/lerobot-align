#!/usr/bin/env python
"""Compatibility entry point. New studies use run_alignment_arms.py."""

from run_alignment_arms import *  # noqa: F403
from run_alignment_arms import main

if __name__ == "__main__":
    raise SystemExit(main())
