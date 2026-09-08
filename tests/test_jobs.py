# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the package-owned Hugging Face Jobs command builder."""

from __future__ import annotations

import shlex
import subprocess
import tomllib
from pathlib import Path

import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot_align import jobs
from lerobot_align.config import AnnotationJobConfig


def test_pod_setup_runs_when_image_only_provides_python3(tmp_path):
    """The official vLLM image has no `python` alias; exercise the shell itself."""
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    calls = tmp_path / "calls"
    for name, body in {
        "apt-get": "exit 0",
        "python3": 'printf "%s\\n" "$*" >> "$POD_TEST_CALLS"',
    }.items():
        executable = executable_dir / name
        executable.write_text("#!/bin/sh\n" + body + "\n")
        executable.chmod(0o755)
    result = subprocess.run(
        ["/bin/bash", "-c", jobs.build_pod_setup("v0.6.1", "/release/project.whl")],
        env={"PATH": str(executable_dir), "POD_TEST_CALLS": str(calls)},
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    operations = calls.read_text().splitlines()
    assert len(operations) == 6
    assert "locate_file('lerobot_align/jobs-constraints.txt')" in operations[2]
    assert "--constraint /tmp/lerobot-align-job-constraints.txt" in operations[3]
    assert "pycairo>=1.20.0,<2.0.0" in operations[3]
    assert operations[4] == "-m pip check"
    assert operations[5].startswith("-c import av, cv2")


def test_remote_runtime_defaults_are_reproducibly_pinned() -> None:
    config = AnnotationJobConfig()

    assert config.image == (
        "vllm/vllm-openai@sha256:2622f38a0aa646c15ccc27bd5033911a58fd94ac69fd8f86aba0692d77cfe5b9"
    )
    assert config.lerobot_ref == "7e241bd630a3719a56157a497ce5d08f244784f1"


def test_project_dependencies_match_remote_runtime_stack() -> None:
    project = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]

    assert "openai>=2.0,<3.0" in project["dependencies"]
    assert "transformers[serving]>=5.5.3,<5.6.0" in project["optional-dependencies"]["serve"]


def test_unpublished_client_requires_explicit_remote_package():
    with pytest.raises(ValueError, match="--job.package_spec"):
        jobs.resolve_package_spec(None)


def test_resolve_package_spec_preserves_immutable_requirement():
    spec = "lerobot-align @ git+https://example.com/org/align.git@" + "a" * 40
    assert jobs.resolve_package_spec(spec) == spec
    wheel = "https://example.com/lerobot_align-0.1.0-py3-none-any.whl#sha256=" + "b" * 64
    assert jobs.resolve_package_spec(wheel) == wheel


@pytest.mark.parametrize('spec', ['lerobot-align==0.1.0', './local.whl',
                                  'git+https://example.com/repo.git@main',
                                  'https://example.com/latest.whl'])
def test_remote_package_must_be_immutable(spec):
    with pytest.raises(ValueError, match='immutable'):
        jobs.resolve_package_spec(spec)


def test_build_pod_command_installs_and_invokes_standalone_package() -> None:
    package_spec = "lerobot-align @ git+https://example.com/org/align.git@feature branch"
    command = jobs.build_pod_command(
        "user/source",
        "v0.6.1",
        package_spec,
        [
            "--repo_id=other/source",
            "--root",
            "/host/private-dataset",
            "--job.target=h200",
            "--job.package_spec=ignored",
            "--push_to_hub=true",
            "--vlm.model_id=org/model",
            "--plan.subtask_video_fallback=error",
        ],
    )

    assert command[:2] == ["bash", "-c"]
    shell_command = command[2]
    assert shlex.quote(package_spec) in shell_command
    assert "lerobot @ git+https://github.com/huggingface/lerobot.git@v0.6.1" in shell_command
    assert "torchvision>=0.25.0,<0.26.0" in shell_command
    assert "transformers>=5.5.3,<5.6.0" in shell_command
    assert "openai>=2.0,<3.0" in shell_command
    assert "python3 -m pip check" in shell_command
    assert "import av, cv2, datasets, huggingface_hub" in shell_command
    assert "transformers, vllm" in shell_command
    assert "import lerobot_align.cli" in shell_command
    assert shell_command.index("python3 -m pip check") < shell_command.index("import av, cv2")
    assert "lerobot-align --repo_id=user/source" in shell_command
    assert "--push_to_hub=true" in shell_command
    assert "--vlm.model_id=org/model" in shell_command
    assert "--plan.subtask_video_fallback=error" in shell_command
    assert "--job.target=local" in shell_command
    assert "lerobot-annotate" not in shell_command
    assert "/host/private-dataset" not in shell_command
    assert "other/source" not in shell_command
    assert "--job.package_spec" not in shell_command


def _pod_align_command(argv: list[str]) -> str:
    """The `lerobot-align ...` half of the pod's shell command."""
    command = jobs.build_pod_command("user/source", "v0.6.1", "lerobot-align==0.1.0", argv)
    return command[2].split("&& lerobot-align", 1)[1]


def test_remote_job_serves_its_own_model_by_default() -> None:
    """A submitted job must start a server on the pod.

    ``VlmConfig.auto_serve`` is False so that a bare local run does not try to
    spawn a 27B server on a machine that may have no GPU. A pod has no server
    either, and nothing else will start one, so the remote path has to invert
    that default -- otherwise the documented HF Jobs workflow, which passes
    ``--vlm.serve_command`` and ``--vlm.num_gpus`` but never
    ``--vlm.auto_serve``, connects to a dead endpoint after paying for the
    image pull and the dataset download.
    """
    align = _pod_align_command(
        [
            "--root=/host/dataset",
            "--vlm.model_id=org/model",
            "--vlm.num_gpus=1",
            "--vlm.serve_command=vllm serve org/model --port {port}",
            "--job.target=h200",
        ]
    )

    assert "--vlm.auto_serve=true" in align
    assert align.count("--vlm.auto_serve") == 1


@pytest.mark.parametrize(
    ("supplied", "expected"),
    [
        (["--vlm.auto_serve=false"], "--vlm.auto_serve=false"),
        (["--vlm.auto_serve=true"], "--vlm.auto_serve=true"),
        (["--vlm.auto_serve", "false"], "--vlm.auto_serve false"),
    ],
)
def test_explicit_auto_serve_is_never_overridden(supplied: list[str], expected: str) -> None:
    """An explicit choice wins, in either `--flag=value` or `--flag value` form.

    A job pointed at an already-running endpoint says so with
    ``--vlm.auto_serve=false``; appending our own value would both contradict
    the submitter and pass the flag twice.
    """
    align = _pod_align_command(
        ["--root=/host/dataset", "--vlm.api_base=https://remote.example/v1", *supplied]
    )

    assert expected in align
    assert align.count("--vlm.auto_serve") == 1


@pytest.mark.parametrize('field', ['subtasks_path', 'subtask_align_calibration_path', 'staging_dir'])
def test_submitter_local_paths_fail_before_any_paid_job(field, monkeypatch, tmp_path):
    from lerobot_align.config import AnnotationPipelineConfig
    cfg = AnnotationPipelineConfig(repo_id='user/source')
    cfg.job.target = 'h200'
    if field == 'staging_dir':
        cfg.staging_dir = tmp_path / 'stage'
    else:
        setattr(cfg.plan, field, tmp_path / 'input.json')
    monkeypatch.setattr(jobs, 'get_token', lambda: 'test-token')
    monkeypatch.setattr(jobs.sys, 'argv', ['lerobot-align'])
    monkeypatch.setattr(jobs, 'run_job', lambda **_: pytest.fail('paid job submitted'))
    monkeypatch.setattr(jobs, 'HfApi', lambda **_: pytest.fail('Hub operation occurred'))
    with pytest.raises(ValueError, match='submitter-local path'):
        jobs.submit_align_to_hf(cfg)


def test_api_credentials_are_not_embedded_in_remote_command():
    command = _pod_align_command(['--vlm.api_key=must-not-appear', '--vlm.model_id=org/model'])
    assert 'must-not-appear' not in command
    assert '--vlm.api_key' not in command


@pytest.mark.parametrize("available", [False, True])
def test_jobs_never_upload_local_cache_and_forward_visual_settings(available, tmp_path, monkeypatch):
    from types import SimpleNamespace
    import lerobot.jobs.dataset as upstream
    from lerobot_align.config import AnnotationPipelineConfig

    local = tmp_path / "user/source"
    (local / "meta").mkdir(parents=True)
    (local / "meta/info.json").write_text("{}")
    (local / ".env").write_text("SYNTHETIC_PRIVATE_VALUE=must-stay-local")
    monkeypatch.setattr(upstream, "HF_LEROBOT_HOME", tmp_path)
    monkeypatch.setattr(upstream, "LeRobotDataset", lambda *_: pytest.fail("local cache was opened for upload"))
    monkeypatch.setenv("LEROBOT_OPENAI_SEND_MM_KWARGS", "1")
    cfg = AnnotationPipelineConfig(repo_id="user/source")
    cfg.vlm.video_metadata_source = "client"
    cfg.job.package_spec = "https://example.invalid/project.whl#sha256=" + "a" * 64
    calls = []
    monkeypatch.setattr(jobs, "get_token", lambda: "synthetic-test-token")
    monkeypatch.setattr(jobs.sys, "argv", ["lerobot-align"])
    monkeypatch.setattr(jobs, "HfApi", lambda **_: SimpleNamespace(repo_exists=lambda *a, **k: available))
    monkeypatch.setattr(jobs, "run_job", lambda **kwargs: calls.append(kwargs) or SimpleNamespace(id="test"))
    monkeypatch.setattr(jobs, "follow_job", lambda *a, **k: False)
    if not available:
        with pytest.raises(RuntimeError, match="accessible Hub dataset"):
            jobs.submit_align_to_hf(cfg)
        assert not calls
    else:
        jobs.submit_align_to_hf(cfg)
        assert calls[0]["env"] == {
            "LEROBOT_VLM_VIDEO_METADATA_SOURCE": "client",
            "LEROBOT_OPENAI_SEND_MM_KWARGS": "1",
        }
        assert "synthetic-test-token" not in str(calls[0]["command"])
