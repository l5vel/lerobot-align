#!/usr/bin/env python
"""Print the subtask spans each dataset ended up with, side by side.

python compare_subtasks.py LABEL=/path/to/ds LABEL=/path/to/ds ...
"""

from __future__ import annotations

import sys
from pathlib import Path

from lerobot_align.reader import (
    iter_episodes,
    reconstruct_subtask_spans,
)
from lerobot.datasets.language import LANGUAGE_PERSISTENT


def spans(root: str) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for record in iter_episodes(Path(root)):
        frame = record.frames_df()
        if LANGUAGE_PERSISTENT not in frame.columns or len(frame) == 0:
            out[record.episode_index] = []
            continue
        raw = frame[LANGUAGE_PERSISTENT].iloc[0]
        rows = [dict(e) for e in raw if isinstance(e, dict)] if raw is not None else []
        end_t = float(record.frame_timestamps[-1]) if record.frame_timestamps else None
        out[record.episode_index] = reconstruct_subtask_spans(rows, episode_end_t=end_t)
    return out


def main() -> int:
    sources = [arg.split("=", 1) for arg in sys.argv[1:]]
    tables = {label: spans(path) for label, path in sources}
    episodes = sorted({ep for t in tables.values() for ep in t})

    for ep in episodes:
        print(f"\n{'=' * 70}\nepisode {ep}\n{'=' * 70}")
        for label in tables:
            rows = tables[label].get(ep, [])
            print(f"\n  [{label}] {len(rows)} span(s)")
            for span in rows:
                print(f"      {span['start']:7.2f} -> {span['end']:7.2f}  {span['text']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
