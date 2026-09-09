"""Regression tests for the matched-frame Qwen visual-token audit."""

from copy import deepcopy
import json
from subprocess import CompletedProcess
from types import SimpleNamespace

import pytest

from evaluation.scripts import measure_frame_format_tokens as measure
from evaluation.scripts import plot_frame_format_tokens as plot
from evaluation.scripts import validate_frame_format_tokens as validate
from lerobot_align.diagnostics import probe_frame_tokens as probe


def fake_processor():
    return SimpleNamespace(
        image_processor=SimpleNamespace(
            patch_size=16,
            merge_size=2,
            size={"shortest_edge": 65_536, "longest_edge": 16_777_216},
        ),
        video_processor=SimpleNamespace(
            patch_size=16,
            merge_size=2,
            temporal_patch_size=2,
            size={"shortest_edge": 4_096, "longest_edge": 25_165_824},
        ),
    )


def test_analytic_counts_use_the_evaluation_geometry_and_qwen_resize():
    result = measure.analytic_counts(
        duration_s=9.5,
        source_height=480,
        source_width=640,
        processor=fake_processor(),
    )

    assert result["n_sampled_frames"] == 20
    assert result["frame_height"] == 168
    assert result["contact_sheet_count"] == 1
    assert result["contact_sheet_grid_thw"] == [1, 42, 70]
    assert result["contact_sheet_visual_tokens"] == 735
    assert result["video_grid_thw"] == [10, 10, 14]
    assert result["video_visual_tokens"] == 350
    assert result["same_frames"] is True


def test_nested_decoded_pair_schema_is_checked_and_hashed(tmp_path):
    rows = [
        {
            "corpus": "corpus_a",
            "component": "task__n2__part",
            "episode": 7,
            "contact_sheet_grid_thw": [1, 42, 70],
            "contact_sheet_count": 2,
            "video_grid_thw": [10, 10, 14],
        }
    ]
    actual = tmp_path / "validation.json"
    actual.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "corpus": "corpus_a",
                        "component": "task__n2__part",
                        "episode": 7,
                        "contact_sheet": {"grid_thw": [[1, 42, 70], [1, 42, 70]]},
                        "video": {"grid_thw": [[10, 10, 14]]},
                    }
                ]
            }
        )
    )

    result = measure.validate_actual_pairs(rows, actual)
    assert result["status"] == "passed"
    assert result["pairs"] == 1
    assert result["corpora"] == {"corpus_a": 1}
    assert result["sha256"] == measure.sha256_file(actual)


def test_duplicate_measurement_identity_is_rejected():
    row = {
        "corpus": "corpus_a",
        "component": "component",
        "episode": 1,
        "identity_namespace": "source",
        "source_episode": 2,
    }
    with pytest.raises(ValueError, match="duplicate output row identity"):
        measure.validate_identities([row, deepcopy(row)])


def test_validator_cache_flag_is_explicitly_switchable():
    required = ["--rows", "rows.jsonl", "--corpus-root", "corpus_a=.", "--out", "out.json"]
    assert validate.parse_args(required).local_files_only is True
    assert validate.parse_args([*required, "--no-local-files-only"]).local_files_only is False


def test_validator_quantile_selection_is_deterministic():
    rows = [
        {"n_sampled_frames": frames, "component": component, "episode": episode}
        for frames, component, episode in (
            (300, "z", 0),
            (10, "b", 1),
            (10, "a", 2),
            (100, "m", 3),
            (200, "n", 4),
        )
    ]
    selected = validate.select_length_quantiles(rows, 3)
    assert [(row["n_sampled_frames"], row["component"]) for row in selected] == [
        (10, "a"),
        (100, "m"),
        (300, "z"),
    ]


def test_gpu_provenance_omits_persistent_device_uuid(monkeypatch):
    def fake_run(*args, **kwargs):
        return CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="3, NVIDIA H100 80GB HBM3, 570.172.08, 81559\n",
        )

    monkeypatch.setattr(validate.subprocess, "run", fake_run)
    result = validate.gpu_provenance(3)
    assert result == {
        "physical_index": 3,
        "name": "NVIDIA H100 80GB HBM3",
        "driver_version": "570.172.08",
        "memory_total_mib": 81559,
    }
    assert "uuid" not in result

    def fake_get_json(url):
        if url.endswith("/version"):
            return {"version": "0.17.0"}
        return {
            "object": "list",
            "data": [
                {
                    "id": "Qwen/Qwen3.8-27B",
                    "root": "Qwen/Qwen3.8-27B",
                    "owned_by": "vllm",
                    "max_model_len": 32768,
                    "created": 123,
                    "permission": [{"id": "modelperm-host-instance"}],
                }
            ],
        }

    monkeypatch.setattr(validate, "get_json", fake_get_json)
    server = validate.server_provenance("http://127.0.0.1:8000/v1")
    assert server["models"]["data"] == [
        {
            "id": "Qwen/Qwen3.8-27B",
            "root": "Qwen/Qwen3.8-27B",
            "owned_by": "vllm",
            "max_model_len": 32768,
        }
    ]


def plot_row(corpus: str, source_episode: int) -> dict:
    return {
        "schema_version": 1,
        "corpus": corpus,
        "task": f"{corpus}_task",
        "component": f"{corpus}_component",
        "component_episode": 0,
        "episode": 0,
        "identity_namespace": f"{corpus}_source",
        "source": f"{corpus}_source",
        "source_episode": source_episode,
        "valid": True,
        "same_frames": True,
        "n_frames": 20,
        "n_sampled_frames": 20,
        "contact_sheet_visual_tokens": 100,
        "video_visual_tokens": 50,
        "video_to_contact_ratio": 0.5,
        "contact_to_video_ratio": 2.0,
        "video_reduction_fraction": 0.5,
    }


@pytest.mark.parametrize("field", ["valid", "same_frames"])
def test_plotter_refuses_unvalidated_or_unmatched_rows(field):
    rows = [plot_row("corpus_a", 1), plot_row("corpus_b", 2)]
    rows[0][field] = False
    with pytest.raises(plot.PlotInputError, match=field):
        plot._validate_rows(rows)


def test_plotter_refuses_summary_that_does_not_reconcile_with_rows():
    rows = plot._validate_rows([plot_row("corpus_a", 1), plot_row("corpus_b", 2)])
    derived = {
        corpus: plot._summary_for([row for row in rows if row["corpus"] == corpus])
        for corpus in plot.EXPECTED_CORPORA
    }
    derived["all"] = plot._summary_for(rows)
    summary = {
        "schema_version": 1,
        "population_validation": {
            "status": "passed",
            "rows": 2,
            "unique_local_identities": 2,
            "unique_source_identities": 2,
        },
        "actual_pair_validation": {
            "status": "passed",
            "pairs": 2,
            "corpora": {"corpus_a": 1, "corpus_b": 1},
        },
        "aggregates": measure.grouped_aggregates(rows),
    }
    plot._validate_summary(summary, derived, len(rows))

    summary["aggregates"]["corpora"]["corpus_b"]["video_visual_tokens"] += 1
    with pytest.raises(plot.PlotInputError, match="does not match rows"):
        plot._validate_summary(summary, derived, len(rows))


def test_probe_counts_only_expanded_visual_placeholders():
    assert probe._visual_tokens([[1, 42, 70], [1, 42, 70]], merge_size=2) == 1_470
    with pytest.raises(ValueError, match="not divisible"):
        probe._visual_tokens([[1, 1, 1]], merge_size=2)


def test_svg_writer_replaces_matplotlib_title_with_one_accessible_title(tmp_path):
    svg = tmp_path / "figure.svg"
    svg.write_text(
        '<svg width="648pt" height="403.2pt"><title>Matplotlib title</title>'
        '<metadata><dc:title>metadata title</dc:title></metadata>\n<path d="M 0 0 ">\n</svg>\n'
    )
    plot._write_accessible_svg(svg, "Accessible title", "Accessible description")
    result = svg.read_text()
    assert 'width="900" height="560"' in result
    assert result.count("<title") == 1
    assert '<title id="figure3-title">Accessible title</title>' in result
    assert "Matplotlib title" not in result
    assert not any(line.endswith(" ") for line in result.splitlines())
