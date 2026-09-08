from pathlib import Path

import pytest

from lerobot_align.diagnostics.eval_align_batch import sidecar_path, summarize


def test_summarize_handles_missing_boundary_metrics(capsys):
    summarize(
        [
            {
                "episode": 1,
                "format": "video",
                "placed_fraction": 0.0,
                "macro_temporal_iou": 0.0,
                "boundary_mae": None,
                "boundary_median": None,
                "boundary_hit_rates": {},
                "boundary_errors": [],
            }
        ],
        ["video"],
    )

    output = capsys.readouterr().out
    assert "| video | 1 | 0 | 0.0% | 0.00% | -- | -- | -- | -- | -- | -- |" in output


@pytest.mark.parametrize(
    ("out", "expected"),
    [
        ("align_batch.json", "align_batch.jsonl"),
        ("results.v2.json", "results.v2.jsonl"),
        ("no_extension", "no_extension.jsonl"),
        # The sidecar must never resolve onto --out itself, or the run appends
        # rows to the file it later overwrites with the array.
        ("run.jsonl", "run.jsonl.rows.jsonl"),
    ],
)
def test_sidecar_never_collides_with_the_output_file(out, expected):
    assert sidecar_path(Path(out)) == Path(expected)
    assert sidecar_path(Path(out)) != Path(out)
