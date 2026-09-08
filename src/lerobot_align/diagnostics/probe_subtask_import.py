#!/usr/bin/env python
"""Report what each subtask-import source finds in a dataset. No VLM involved.

python probe_subtask_import.py /path/to/dataset [--prefix dense] [--max-episodes 5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot_align.reader import iter_episodes
from lerobot_align.subtask_import import (
    IMPORT_SOURCES,
    SubtaskImporter,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--prefix", default="dense")
    ap.add_argument("--max-episodes", type=int, default=5)
    args = ap.parse_args()

    importers = {
        source: SubtaskImporter(root=args.root, source=source, sarm_prefix=args.prefix)
        for source in IMPORT_SOURCES
    }

    for n, record in enumerate(iter_episodes(args.root)):
        if n >= args.max_episodes:
            break
        print(
            f"\n=== episode {record.episode_index} "
            f"({record.row_count} frames, task={record.episode_task!r}) ==="
        )
        for source, importer in importers.items():
            spans, _ = importer.spans_for(record)
            if not spans:
                print(f"  {source:<20} -> nothing")
                continue
            print(f"  {source:<20} -> {len(spans)} span(s)")
            for span in spans[:8]:
                print(f"      [{span['start']:7.2f} -> {span['end']:7.2f}]  {span['text']}")
            if len(spans) > 8:
                print(f"      ... {len(spans) - 8} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
