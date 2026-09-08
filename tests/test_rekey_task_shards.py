"""The floors and the VLM arms must name the same trajectory by the same integer.

`aggregate.py` pairs a contrast on ``f"{dataset}/{episode}"``. The VLM arms are
merged into ORIGINAL RoboInter indices; `shard_corpus_b_by_task.py` emits the
renumbered subset index. Left unreconciled, every dataset name matches, every
episode id differs, and claim B1's floor contrasts pair on nothing -- reporting
a plumbing failure as a null.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "evaluation/scripts/rekey_task_shards.py"
SUBSET_TO_ORIGINAL = {0: 648, 1: 753, 2: 901}


def _shards(root: Path, *, eval_ids: list[int]) -> tuple[Path, Path, Path]:
    gt_dir, splits_dir = root / "gt", root / "splits"
    gt_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)
    index_map = root / "study.index_map.json"
    index_map.write_text(json.dumps({"episodes": [
        {"new_index": new, "original_index": original}
        for new, original in SUBSET_TO_ORIGINAL.items()
    ]}))
    (gt_dir / "corpus_b__t1.json").write_text(json.dumps({
        "dataset": "corpus_b__t1", "index_space": "renumbered subset indices",
        "episodes": {str(e): [{"start": 0.0, "end": 1.0, "text": "a"},
                              {"start": 1.0, "end": 2.0, "text": "b"}]
                     for e in SUBSET_TO_ORIGINAL},
    }))
    (splits_dir / "corpus_b__t1.json").write_text(json.dumps({
        "dataset": "corpus_b__t1", "seed": [0], "eval": eval_ids,
        "index_space": "renumbered subset indices",
    }))
    return index_map, gt_dir, splits_dir


def _run(root: Path, index_map: Path, gt_dir: Path, splits_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--index-map", str(index_map),
         "--gt-dir", str(gt_dir), "--splits-dir", str(splits_dir),
         "--gt-out-dir", str(root / "gt_out"), "--splits-out-dir", str(root / "splits_out")],
        capture_output=True, text=True,
    )


def test_ground_truth_and_splits_move_together_into_original_indices(tmp_path: Path) -> None:
    index_map, gt_dir, splits_dir = _shards(tmp_path, eval_ids=[1, 2])
    assert _run(tmp_path, index_map, gt_dir, splits_dir).returncode == 0

    split = json.loads((tmp_path / "splits_out/corpus_b__t1.json").read_text())
    truth = json.loads((tmp_path / "gt_out/corpus_b__t1.json").read_text())
    assert split["seed"] == [648]
    assert split["eval"] == [753, 901]
    # The pairing key is the episode id, so the ids the scorer will read from the
    # ground truth must be the same integers the split names.
    assert sorted(int(e) for e in truth["episodes"]) == [648, 753, 901]
    assert split["index_space"] == truth["index_space"] == "ORIGINAL RoboInter episode_index"


def test_an_episode_outside_the_map_is_refused_not_dropped(tmp_path: Path) -> None:
    # Silently dropping it would shrink one arm's population and leave the
    # contrast paired on the remainder, which reads as a smaller but valid n.
    index_map, gt_dir, splits_dir = _shards(tmp_path, eval_ids=[1, 2, 7])
    result = _run(tmp_path, index_map, gt_dir, splits_dir)
    assert result.returncode != 0
    assert "absent from" in result.stderr
    assert not (tmp_path / "splits_out/corpus_b__t1.json").exists()


def test_a_non_injective_map_is_refused(tmp_path: Path) -> None:
    index_map, gt_dir, splits_dir = _shards(tmp_path, eval_ids=[1, 2])
    index_map.write_text(json.dumps({"episodes": [
        {"new_index": 0, "original_index": 648},
        {"new_index": 1, "original_index": 648},
        {"new_index": 2, "original_index": 901},
    ]}))
    result = _run(tmp_path, index_map, gt_dir, splits_dir)
    assert result.returncode != 0
    assert "injective" in result.stderr


@pytest.mark.parametrize("group", ["seed", "eval"])
def test_every_split_episode_must_carry_ground_truth(tmp_path: Path, group: str) -> None:
    index_map, gt_dir, splits_dir = _shards(tmp_path, eval_ids=[1, 2])
    truth = json.loads((gt_dir / "corpus_b__t1.json").read_text())
    truth["episodes"].pop("0" if group == "seed" else "1")
    (gt_dir / "corpus_b__t1.json").write_text(json.dumps(truth))
    result = _run(tmp_path, index_map, gt_dir, splits_dir)
    assert result.returncode != 0
    assert "no ground truth" in result.stderr
