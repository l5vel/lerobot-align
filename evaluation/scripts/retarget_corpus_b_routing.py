#!/usr/bin/env python
"""Re-key the calibration routing into each component's own index space.

Corpus B now has THREE index spaces, and this is where the last two meet:

  1. ORIGINAL RoboInter `episode_index`, what the census and selection speak.
  2. WHOLE-CORPUS subset indices 0..1591, assigned when the 1,592 selected
     episodes were built into one root. The calibration cohorts and
     `calibration_routing.json` are keyed here.
  3. COMPONENT-LOCAL indices 0..N-1, assigned again when each task was built
     into its own dataset -- which it had to be, because a partial run cannot
     write back through parquet shards holding other tasks' episodes.

Routing keyed in space 2 and applied in space 3 would be silently wrong: local
episode 10 of task_0069 is not whole-corpus episode 10. And it would LOOK right
on the first task checked, because the first component's local indices coincide
with its whole-corpus ones. That is the failure this file exists to prevent.

The calibration FILES need no change -- they hold per-boundary offsets and
duration priors, which name no episode. Only the routing table does. Each
component gets its own, translated local -> original -> whole-corpus -> cohort,
and every eval episode must resolve or this refuses to write anything.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--routing", type=Path, required=True,
                        help="calibration_routing.json, keyed on whole-corpus indices")
    parser.add_argument("--study-index-map", type=Path, required=True,
                        help="study.index_map.json: original -> whole-corpus")
    parser.add_argument("--components-dir", type=Path, required=True,
                        help="holds <component>.index_map.json for each component")
    parser.add_argument("--splits-dir", type=Path, required=True,
                        help="component-local splits, to check every eval episode resolves")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    routing_payload = json.loads(args.routing.read_text(encoding="utf-8"))
    routing = routing_payload["routing"]

    study = json.loads(args.study_index_map.read_text(encoding="utf-8"))["episodes"]
    original_to_whole = {int(e["original_index"]): int(e["new_index"]) for e in study}

    out: dict[str, Any] = {}
    totals = {"components": 0, "eval": 0, "calibrated": 0, "fallback": 0}
    for split_path in sorted(args.splits_dir.glob("*.json")):
        component = split_path.stem
        index_map_path = args.components_dir / f"{component}.index_map.json"
        if not index_map_path.exists():
            raise SystemExit(f"{component}: no index map at {index_map_path}")
        local_to_original = {
            int(e["new_index"]): int(e["original_index"])
            for e in json.loads(index_map_path.read_text(encoding="utf-8"))["episodes"]
        }
        split = json.loads(split_path.read_text(encoding="utf-8"))

        rows: dict[str, Any] = {}
        for local in sorted(int(e) for e in split["eval"]):
            if local not in local_to_original:
                raise SystemExit(f"{component}: local episode {local} is not in its index map")
            original = local_to_original[local]
            if original not in original_to_whole:
                raise SystemExit(
                    f"{component}: original episode {original} was never built into the "
                    "whole-corpus root, so no cohort was ever assigned to it"
                )
            whole = original_to_whole[original]
            row = routing.get(str(whole))
            if row is None:
                raise SystemExit(
                    f"{component}: whole-corpus episode {whole} (local {local}, original "
                    f"{original}) has no routing entry. Refusing to emit a partial table: a "
                    "missing entry would silently become an uncalibrated fallback."
                )
            rows[str(local)] = {
                "local_index": local, "original_index": original, "whole_corpus_index": whole,
                "task": row["task"], "n_segments": row["n_segments"],
                "cohort": row["cohort"], "calibrated": bool(row["calibrated"]),
            }
            totals["eval"] += 1
            totals["calibrated" if row["calibrated"] else "fallback"] += 1

        out[component] = {"n_eval": len(rows), "routing": rows}
        totals["components"] += 1

    payload = {
        "note": "Calibration routing keyed on COMPONENT-LOCAL indices. Translated "
                "local -> original -> whole-corpus -> cohort. The calibration files "
                "themselves are index-independent and are reused unchanged; only this "
                "table is re-keyed.",
        "source_routing": str(args.routing),
        "totals": totals,
        "calibration_applied_fraction": round(totals["calibrated"] / max(totals["eval"], 1), 4),
        "components": out,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({**totals, "fraction": payload["calibration_applied_fraction"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
