"""CPU regressions for episode-local completeness failures and safe publication."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lerobot_align.config import (
    AnnotationPipelineConfig,
    ExecutorConfig,
    InterjectionsConfig,
    VqaConfig,
)
from lerobot_align.executor import Executor
from lerobot_align.frames import null_provider
from lerobot_align.modules import PlanSubtasksMemoryModule
from lerobot_align.reader import iter_episodes
from lerobot_align.staging import EpisodeStaging
from lerobot_align.validator import StagingValidator
from lerobot_align.vlm_client import StubVlmClient
from lerobot_align.writer import LanguageColumnsWriter
from tests.fixtures import build_annotation_dataset


class StagedPlan:
    enabled = True

    def run_episode(self, record, staging):
        pass


class DisabledModule:
    enabled = False


def config(**kwargs):
    return AnnotationPipelineConfig(
        interjections=InterjectionsConfig(enabled=False),
        vqa=VqaConfig(enabled=False),
        **kwargs,
    )


def executor(cfg):
    return Executor(
        config=cfg,
        plan=StagedPlan(),
        interjections=DisabledModule(),
        vqa=DisabledModule(),
        writer=LanguageColumnsWriter(),
        validator=StagingValidator(),
    )


def plan_rows(task_aug_count=11):
    return [
        {"role": "assistant", "style": "subtask", "content": "pick up cup", "timestamp": 0.0},
        {"role": "assistant", "style": "plan", "content": "1. pick up cup", "timestamp": 0.0},
        *[
            {"role": "user", "style": "task_aug", "content": f"task {i}", "timestamp": 0.0}
            for i in range(task_aug_count)
        ],
    ]


@pytest.mark.parametrize("skip_validation", [False, True])
@pytest.mark.parametrize(
    "case",
    [
        "duplicate",
        "canonical",
        "empty",
        "quoted_empty",
        "non_string",
        "short_reply",
        "truncated_before_dedup",
        "complete",
    ],
)
def test_task_rephrasing_filtering_preserves_subtask_publication(
    single_episode_root, monkeypatch, capsys, case, skip_validation
):
    """Exercise the real generation → filtering → validation → parquet path."""
    cfg = config(
        skip_validation=skip_validation,
        executor=ExecutorConfig(max_incomplete_episode_fraction=0),
    )
    record = next(iter_episodes(single_episode_root))
    variants = [f"variant {i}" for i in range(9)]
    replies = {
        "duplicate": [*variants, f"  {variants[-1]}  "],
        "canonical": [*variants, f'"{record.episode_task}"'],
        "empty": [*variants, "  "],
        "quoted_empty": [*variants, '"  "'],
        "non_string": [*variants, {"text": "variant nine"}],
        "short_reply": variants,
        "truncated_before_dedup": [*variants, variants[-1], "variant nine"],
        "complete": [*variants, "variant nine"],
    }
    calls = []

    def responder(messages):
        calls.append(messages)
        return {"rephrasings": replies[case]}

    module = PlanSubtasksMemoryModule(
        config=cfg.plan,
        vlm=StubVlmClient(responder=responder),
        frame_provider=null_provider(),
    )
    monkeypatch.setattr(
        module,
        "_subtask_spans",
        lambda *_: [{"text": "pick up cup", "start": 0.0, "end": record.frame_timestamps[-1]}],
    )
    runner = executor(cfg)
    runner.plan = module
    summary = runner.run(single_episode_root)
    assert len(calls) == 1  # No retry or model/GPU needed to reproduce the shortfall.
    assert "exactly 10" in calls[0][0]["content"][0]["text"]
    assert all(block["type"] == "text" for message in calls[0] for block in message["content"])
    staging_dir = cfg.resolved_staging_dir(single_episode_root)
    staged = EpisodeStaging(staging_dir, record.episode_index).read_all()["plan"]
    expected_phrasings = [record.episode_task, *variants]
    if case == "complete":
        expected_phrasings.append("variant nine")
    assert [row["content"] for row in staged if row["style"] == "task_aug"] == expected_phrasings
    report = summary.validation_report
    assert report.ok, report.errors
    assert not report.completeness_errors
    assert not report.episode_errors
    assert not report.episode_completeness_errors
    assert not summary.skipped_episodes
    output = capsys.readouterr().out
    if case == "complete":
        assert not report.warnings
        assert "validator warning:" not in output
    else:
        assert report.warnings == [
            "ep=0: plan module emitted 10 task_aug row(s), below requested target 11; "
            "retaining available task phrasings and subtask annotations"
        ]
        assert f"[annotate] validator warning: {report.warnings[0]}" in output
    saved = json.loads((staging_dir / "validation_report.json").read_text())
    assert saved["warnings"] == report.warnings
    assert saved["skipped_episodes"] == {}
    assert saved["errors"] == saved["completeness_errors"] == []
    rows = pq.read_table(record.data_path).to_pylist()
    for row in rows:
        assert any(
            atom["style"] == "subtask" and atom["content"] == "pick up cup"
            for atom in row["language_persistent"]
        )
    assert [
        atom["content"] for atom in rows[0]["language_persistent"] if atom["style"] == "task_aug"
    ] == expected_phrasings
    assert StagingValidator().validate([record], staging_dir).ok


def incomplete_plan_rows():
    # Keep exercising structural completeness: every subtask boundary needs
    # its deterministic plan state, regardless of paraphrase diversity.
    return [row for row in plan_rows() if row["style"] != "plan"]


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("skip_validation", [False, True])
def test_incomplete_episode_is_skipped_and_good_episodes_publish(
    tmp_path, capsys, packed, existing, skip_validation
):
    root = build_annotation_dataset(tmp_path / "ds", [(i, 12, "pick up cup") for i in (0, 2, 7)])
    paths = sorted((root / "data").rglob("*.parquet"))
    if packed:
        pq.write_table(pa.concat_tables([pq.read_table(p) for p in paths]), paths[0])
        for path in paths[1:]:
            path.unlink()
    cfg = config(skip_validation=skip_validation)
    staging_dir = cfg.resolved_staging_dir(root)
    if existing:
        for record in iter_episodes(root):
            old_rows = plan_rows()
            old_rows[0]["content"] = "previous annotation"
            EpisodeStaging(staging_dir, record.episode_index).write("plan", old_rows)
        executor(cfg).run(root)
    old = [row for p in (root / "data").rglob("*.parquet") for row in pq.read_table(p).to_pylist()]
    for record in iter_episodes(root):
        EpisodeStaging(staging_dir, record.episode_index).write(
            "plan", incomplete_plan_rows() if record.episode_index == 2 else plan_rows()
        )

    summary = executor(cfg).run(root)

    message = "ep=2: plan module missed plan output at subtask boundary timestamp(s) [0.0]"
    assert summary.skipped_episodes == {2: [message]}
    assert not summary.validation_report.ok
    assert message in capsys.readouterr().out
    saved = json.loads((staging_dir / "validation_report.json").read_text())
    assert saved["skipped_episodes"] == {"2": [message]}
    assert saved["episodes_checked"] == 3
    for path in (root / "data").rglob("*.parquet"):
        table = pq.read_table(path)
        assert "subtask_index" not in table.column_names
        for row in table.to_pylist():
            if row["episode_index"] == 2:
                before = next(
                    r
                    for r in old
                    if r["episode_index"] == 2 and r["frame_index"] == row["frame_index"]
                )
                assert row["language_persistent"] == before.get("language_persistent", [])
                assert row["language_events"] == before.get("language_events", [])
            else:
                assert any(atom["content"] == "pick up cup" for atom in row["language_persistent"])
    # Every migrated shard agrees with the feature metadata, including a shard
    # containing only a rejected episode on the first annotation run.
    features = json.loads((root / "meta" / "info.json").read_text())["features"]
    assert "language_persistent" in features
    assert "subtask_index" not in features


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), float("inf"), True, "0.5"])
def test_invalid_incomplete_fraction_rejected(value):
    with pytest.raises(ValueError, match="max_incomplete_episode_fraction"):
        ExecutorConfig(max_incomplete_episode_fraction=value)


@pytest.mark.parametrize(
    "bad_count,limit,fails",
    [
        (0, 0.0, False),
        (1, 0.0, True),
        (1, 0.5, False),
        (1, 0.49, True),
        (2, 1.0, True),
    ],
)
@pytest.mark.parametrize("skip_validation", [False, True])
def test_incomplete_threshold_enforced_before_mutation(
    fixture_dataset_root, capsys, bad_count, limit, fails, skip_validation
):
    cfg = config(
        executor=ExecutorConfig(max_incomplete_episode_fraction=limit),
        skip_validation=skip_validation,
    )
    staging_dir = cfg.resolved_staging_dir(fixture_dataset_root)
    records = list(iter_episodes(fixture_dataset_root))
    for i, record in enumerate(records):
        EpisodeStaging(staging_dir, record.episode_index).write(
            "plan", incomplete_plan_rows() if i < bad_count else plan_rows()
        )
    paths = [r.data_path for r in records] + [fixture_dataset_root / "meta" / "info.json"]
    before = {p: p.read_bytes() for p in paths}
    if fails:
        with pytest.raises(RuntimeError, match="Staging validation failed.*incomplete=") as exc:
            executor(cfg).run(fixture_dataset_root)
        assert "missed plan output" in str(exc.value)
        assert {p: p.read_bytes() for p in paths} == before
    else:
        summary = executor(cfg).run(fixture_dataset_root)
        assert len(summary.skipped_episodes) == bad_count
    assert (staging_dir / "validation_report.json").exists()
    if bad_count:
        assert "missed plan output" in capsys.readouterr().out


@pytest.mark.parametrize("structural_error_on_skipped_episode", [False, True])
def test_structural_errors_only_excluded_with_their_incomplete_episode(
    fixture_dataset_root, structural_error_on_skipped_episode
):
    cfg = config()
    staging_dir = cfg.resolved_staging_dir(fixture_dataset_root)
    for record in iter_episodes(fixture_dataset_root):
        rows = incomplete_plan_rows() if record.episode_index == 0 else plan_rows()
        if record.episode_index == (0 if structural_error_on_skipped_episode else 1):
            rows.append(
                {"role": "assistant", "style": "unknown", "content": "bad", "timestamp": 0.0}
            )
        EpisodeStaging(staging_dir, record.episode_index).write("plan", rows)
    if structural_error_on_skipped_episode:
        summary = executor(cfg).run(fixture_dataset_root)
        assert len(summary.skipped_episodes[0]) == 2
    else:
        paths = list((fixture_dataset_root / "data").rglob("*.parquet"))
        before = {p: p.read_bytes() for p in paths}
        with pytest.raises(RuntimeError, match="Staging validation failed"):
            executor(cfg).run(fixture_dataset_root)
        assert {p: p.read_bytes() for p in paths} == before
