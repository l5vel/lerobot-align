"""Evaluation CLI authentication, exit status and provenance regression tests."""

import json
import sys

import pytest

from lerobot_align.diagnostics import eval_align, eval_align_batch, fit_align_calibration
from lerobot_align.diagnostics._provenance import fingerprint
from lerobot_align.modules import plan_subtasks_memory
from lerobot_align.vlm_client import StubVlmClient
from tests._helpers import SyntheticFrameProvider


@pytest.fixture
def batch_run(single_episode_root, tmp_path, monkeypatch):
    root = single_episode_root
    (root / "meta/lerobot_annotations.json").write_text(
        json.dumps(
            {
                "episodes": {
                    "0": {
                        "subtasks": [
                            {"label": "pick up cup", "start": 0, "end": 0.5},
                            {"label": "place cup", "start": 0.5, "end": 0.9},
                        ]
                    }
                }
            }
        )
    )
    captured = []

    def make_client(config):
        captured.append(config)
        return StubVlmClient(
            lambda _: {
                "subtasks": [
                    {"index": 0, "start": 0, "end": 0.5},
                    {"index": 1, "start": 0.5, "end": 0.9},
                ]
            }
        )

    monkeypatch.setattr(
        eval_align_batch, "resolve_camera_keys", lambda *_: ["observation.images.top"]
    )
    monkeypatch.setattr(
        eval_align_batch, "VideoFrameProvider", lambda **_: SyntheticFrameProvider()
    )
    monkeypatch.setattr(eval_align_batch, "make_vlm_client", make_client)
    out = tmp_path / "results.json"

    def run(*extra):
        monkeypatch.setattr(
            sys,
            "argv",
            ["eval", str(root), "--formats", "contact_sheet", "--out", str(out), *extra],
        )
        return eval_align_batch.main()

    return run, out, captured


def test_batch_authentication_comes_from_environment_without_logging_key(
    batch_run, monkeypatch, capsys
):
    run, out, captured = batch_run
    monkeypatch.setenv("CUSTOM_ENDPOINT_KEY", "test-secret-do-not-publish")
    assert run("--api-key-env", "CUSTOM_ENDPOINT_KEY") == 0
    assert captured[0].api_key == "test-secret-do-not-publish"
    assert "test-secret-do-not-publish" not in out.read_text() + capsys.readouterr().out
    row = json.loads(out.read_text())[0]
    assert row["config_fingerprint"] == fingerprint(row["provenance"])
    assert row["provenance"]["ground_truth_sha256"]


def test_batch_failure_returns_nonzero_and_keeps_actionable_row(batch_run, monkeypatch):
    run, out, _ = batch_run

    def fail(*args):
        raise RuntimeError("endpoint refused inference")

    monkeypatch.setattr(
        plan_subtasks_memory.PlanSubtasksMemoryModule, "_align_given_subtasks", fail
    )
    assert run() == 1
    row = json.loads(out.read_text())[0]
    assert "endpoint refused inference" in row["error"]
    assert eval_align_batch.sidecar_path(out).read_text()


def test_batch_empty_selection_is_not_success(batch_run):
    run, out, captured = batch_run
    with pytest.raises(SystemExit) as info:
        run("--episodes", "999")
    assert info.value.code == 2
    assert not captured
    assert not out.exists()


@pytest.mark.parametrize(
    "flag,value",
    [("--fps", "nan"), ("--processor-fps", "inf"), ("--max-frames", "0"), ("--temperature", "nan")],
)
def test_invalid_evaluation_numbers_fail_before_inference(batch_run, flag, value):
    run, out, captured = batch_run
    with pytest.raises(SystemExit) as info:
        run(flag, value)
    assert info.value.code == 2
    assert not captured
    assert not out.exists()


def test_missing_explicit_api_key_environment_fails_early(batch_run):
    run, out, captured = batch_run
    with pytest.raises(SystemExit) as info:
        run("--api-key-env", "UNSET_EVALUATION_SECRET")
    assert info.value.code == 2
    assert not captured


def test_user_labels_require_a_file_and_never_use_internal_examples(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["eval", str(tmp_path), "--labels", "user"])
    with pytest.raises(SystemExit) as info:
        eval_align.main()
    assert info.value.code == 2
    assert not hasattr(eval_align, "USER_LABELS")


def test_batch_overrides_do_not_leak_into_later_calls(batch_run):
    run, _, _ = batch_run
    original = plan_subtasks_memory.to_video_url_block
    assert run("--processor-fps", "3") == 0
    assert plan_subtasks_memory.to_video_url_block is original


def test_incompatible_retries_cannot_be_silently_merged(batch_run):
    run, out, _ = batch_run
    assert run("--fps", "1") == 0
    first = json.loads(out.read_text())[0]
    assert run("--fps", "2") == 0
    second = json.loads(out.read_text())[0]
    assert first["config_fingerprint"] != second["config_fingerprint"]
    sidecar = eval_align_batch.sidecar_path(out)
    with pytest.raises(ValueError, match="mixed evaluation configurations"):
        fit_align_calibration.load_result_rows(sidecar)
    rows, collisions, _ = fit_align_calibration.load_result_rows(
        sidecar, config_fingerprint=first["config_fingerprint"]
    )
    assert rows == [first]
    assert collisions == []


def test_same_configuration_retry_keeps_existing_recovery_contract(batch_run):
    run, out, _ = batch_run
    assert run() == 0
    first = json.loads(out.read_text())[0]
    assert run() == 0
    second = json.loads(out.read_text())[0]
    assert first["config_fingerprint"] == second["config_fingerprint"]
    assert first["run_id"] != second["run_id"]
    rows, collisions, _ = fit_align_calibration.load_result_rows(eval_align_batch.sidecar_path(out))
    assert rows == [second]
    assert len(collisions) == 1


def test_legacy_provenance_requires_explicit_acknowledgement(tmp_path, monkeypatch):
    from tests.test_fit_align_calibration import make_inputs

    args, out = make_inputs(tmp_path, [("pick", "place")])
    args.remove("--allow-legacy-provenance")
    monkeypatch.setattr(sys, "argv", ["fit", *args])
    assert fit_align_calibration.main() == 1
    assert not out.exists()


def test_prompt_override_retries_cannot_be_silently_merged(batch_run, monkeypatch):
    run, out, _ = batch_run
    assert run() == 0
    first = json.loads(out.read_text())[0]
    monkeypatch.setenv("LEROBOT_PROMPT_OVERRIDE_plan_subtask_align", "private candidate prompt")
    assert run() == 0
    second = json.loads(out.read_text())[0]
    assert first["config_fingerprint"] != second["config_fingerprint"]
    assert "private candidate prompt" not in json.dumps(second["provenance"])
    with pytest.raises(ValueError, match="mixed evaluation configurations"):
        fit_align_calibration.load_result_rows(eval_align_batch.sidecar_path(out))


def test_blank_prompt_override_preserves_effective_configuration(batch_run, monkeypatch):
    run, out, _ = batch_run
    assert run() == 0
    first = json.loads(out.read_text())[0]
    monkeypatch.setenv("LEROBOT_PROMPT_OVERRIDE_plan_subtask_align", " \n ")
    assert run() == 0
    assert first["config_fingerprint"] == json.loads(out.read_text())[0]["config_fingerprint"]


def test_video_metadata_policy_changes_evaluation_identity(batch_run, monkeypatch):
    run, out, _ = batch_run
    monkeypatch.setenv("LEROBOT_VLM_VIDEO_METADATA_SOURCE", "server")
    assert run() == 0
    first = json.loads(out.read_text())[0]
    monkeypatch.setenv("LEROBOT_VLM_VIDEO_METADATA_SOURCE", "client")
    assert run() == 0
    second = json.loads(out.read_text())[0]
    assert first["config_fingerprint"] != second["config_fingerprint"]
    assert second["provenance"]["settings"]["video_metadata_source"] == "client"


def test_endpoint_identity_distinguishes_servers_without_recording_credentials(batch_run):
    run, out, _ = batch_run
    assert run("--api-base", "https://user:private-key@first.invalid/v1?token=private-query") == 0
    first = json.loads(out.read_text())[0]
    assert run("--api-base", "https://user:rotated-key@first.invalid/v1?token=rotated-query") == 0
    second = json.loads(out.read_text())[0]
    assert first["config_fingerprint"] == second["config_fingerprint"]
    assert run("--api-base", "https://second.invalid/v1") == 0
    third = json.loads(out.read_text())[0]
    assert first["config_fingerprint"] != third["config_fingerprint"]
    for row in (first, second, third):
        provenance = json.dumps(row["provenance"])
        assert "private-key" not in provenance
        assert "private-query" not in provenance
        assert "rotated-key" not in provenance
        assert "rotated-query" not in provenance
        assert ".invalid" not in provenance
