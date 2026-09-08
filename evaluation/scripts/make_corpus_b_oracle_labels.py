#!/usr/bin/env python
"""Compatibility entry point. New studies use make_alignment_labels.py."""

from make_alignment_labels import *  # noqa: F403
from make_alignment_labels import main

if __name__ == "__main__":
    raise SystemExit(main())
