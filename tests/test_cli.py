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
"""Tests for the standalone command boundary."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot_align import cli
from lerobot_align.config import AnnotationPipelineConfig


class _FakeHfApi:
    def __init__(
        self,
        *,
        tags: dict[str, str] | None = None,
        repo_exists: bool = True,
        head: str | None = "base-commit",
        upload_oid: str = "uploaded-commit",
        card_oid: str = "card-commit",
        create_errors: list[Exception | None] | None = None,
        delete_errors: list[Exception | None] | None = None,
    ) -> None:
        self.tags = dict(tags or {})
        self.repo_exists = repo_exists
        self.head = head
        self.upload_oid = upload_oid
        self.card_oid = card_oid
        self.create_errors = list(create_errors or [])
        self.delete_errors = list(delete_errors or [])
        self.create_repo_calls: list[dict[str, object]] = []
        self.upload_folder_calls: list[dict[str, object]] = []
        self.card_push_calls: list[dict[str, object]] = []
        self.create_tag_calls: list[dict[str, str]] = []
        self.delete_tag_calls: list[dict[str, str]] = []
        self.list_repo_refs_calls = 0
        self.list_repo_refs_hooks: dict[int, Callable[[_FakeHfApi], None]] = {}

    def repo_info(self, **_kwargs: object) -> SimpleNamespace:
        if not self.repo_exists:
            response = httpx.Response(
                404,
                request=httpx.Request("GET", "https://huggingface.co/api/datasets/user/dataset"),
            )
            raise cli.RepositoryNotFoundError("repository not found", response=response)
        return SimpleNamespace(sha=self.head)

    def create_repo(self, **kwargs: object) -> None:
        self.create_repo_calls.append(kwargs)
        if not self.repo_exists:
            self.repo_exists = True
            self.head = None
        return None

    def upload_folder(self, **kwargs: object) -> SimpleNamespace:
        self.upload_folder_calls.append(kwargs)
        if kwargs.get("parent_commit") != self.head:
            raise RuntimeError("stale upload parent")
        self.head = self.upload_oid
        return SimpleNamespace(oid=self.upload_oid)

    def list_repo_refs(self, **_kwargs: object) -> SimpleNamespace:
        if not self.repo_exists:
            raise AssertionError("cannot list refs for a missing repository")
        self.list_repo_refs_calls += 1
        hook = self.list_repo_refs_hooks.get(self.list_repo_refs_calls)
        if hook is not None:
            hook(self)
        return SimpleNamespace(
            tags=[
                SimpleNamespace(name=name, target_commit=target)
                for name, target in self.tags.items()
            ]
        )

    def create_tag(self, **kwargs: str) -> None:
        self.create_tag_calls.append(kwargs)
        if self.create_errors:
            error = self.create_errors.pop(0)
            if error is not None:
                raise error
        tag = kwargs["tag"]
        if tag in self.tags:
            raise RuntimeError(f"tag {tag!r} already exists")
        self.tags[tag] = kwargs["revision"]

    def delete_tag(self, **kwargs: str) -> None:
        self.delete_tag_calls.append(kwargs)
        if self.delete_errors:
            error = self.delete_errors.pop(0)
            if error is not None:
                raise error
        tag = kwargs["tag"]
        if tag not in self.tags:
            raise RuntimeError(f"tag {tag!r} does not exist")
        del self.tags[tag]


class _FakeDatasetCard:
    def __init__(self, api: _FakeHfApi) -> None:
        self.api = api

    def push_to_hub(self, **kwargs: object) -> SimpleNamespace:
        self.api.card_push_calls.append(kwargs)
        if kwargs.get("parent_commit") != self.api.head:
            raise RuntimeError("stale card parent")
        self.api.head = self.api.card_oid
        return SimpleNamespace(oid=self.api.card_oid)


def _install_hub_fakes(monkeypatch: pytest.MonkeyPatch, api: _FakeHfApi) -> None:
    monkeypatch.setattr(cli, "HfApi", lambda: api)
    monkeypatch.setattr(
        cli,
        "source_dataset_card",
        lambda *_args: _FakeDatasetCard(api),
    )


def test_resolve_root_copies_hub_snapshot_to_writable_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "cache" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    source_file = snapshot / "meta" / "info.json"
    source_file.parent.mkdir()
    source_file.write_text("source")
    source_file.chmod(0o444)
    worktree = tmp_path / "worktree"
    worktree.mkdir()

    calls: list[dict[str, str]] = []

    def fake_snapshot_download(**kwargs: str) -> str:
        calls.append(kwargs)
        return str(snapshot)

    monkeypatch.setattr(cli, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda **_kwargs: str(worktree))

    resolved = cli._resolve_root(AnnotationPipelineConfig(repo_id="user/dataset"))

    assert calls == [{"repo_id": "user/dataset", "repo_type": "dataset"}]
    assert resolved == worktree
    copied_file = resolved / "meta" / "info.json"
    copied_file.write_text("changed")
    assert copied_file.read_text() == "changed"
    assert source_file.read_text() == "source"


def test_resolve_root_uses_explicit_local_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "snapshot_download",
        lambda **_kwargs: pytest.fail("explicit --root must not download a Hub snapshot"),
    )

    assert cli._resolve_root(AnnotationPipelineConfig(root=tmp_path)) == tmp_path


def test_hub_worktree_retention_is_reported(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=cli.__name__):
        cli._log_retained_worktree(tmp_path, AnnotationPipelineConfig(repo_id="user/dataset"))

    assert str(tmp_path) in caplog.text
    assert "pass --root" in caplog.text


def test_hub_worktree_retention_is_reported_when_annotation_fails(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _FakeHfApi(tags={"v3.1": "previous-release"})
    _install_hub_fakes(monkeypatch, api)
    monkeypatch.setattr(cli, "_resolve_root", lambda _cfg: fixture_dataset_root)
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)
    cfg = AnnotationPipelineConfig(repo_id="user/dataset", push_to_hub=True)

    with (
        caplog.at_level(logging.WARNING, logger=cli.__name__),
        pytest.raises(RuntimeError, match="--allow_version_tag_move=true"),
    ):
        cli.annotate(cfg)

    assert str(fixture_dataset_root) in caplog.text
    assert "worktree retained" in caplog.text


def test_main_configures_logging_before_cli_parsing(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[tuple[str, dict[str, object] | None]] = []
    monkeypatch.setattr(
        cli.logging,
        "basicConfig",
        lambda **kwargs: events.append(("logging", kwargs)),
    )
    monkeypatch.setattr(cli, "annotate", lambda: events.append(("annotate", None)))

    cli.main()

    assert events[0] == (
        "logging",
        {"level": cli.logging.INFO, "format": cli._LOG_FORMAT, "force": True},
    )
    assert events[1] == ("annotate", None)


def _forbid_vlm_creation(_config: object) -> None:
    pytest.fail("dataset preflight must fail before VLM construction")


def test_missing_push_target_fails_before_vlm_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)
    cfg = AnnotationPipelineConfig(root=tmp_path, push_to_hub=True)

    with pytest.raises(ValueError, match="--push_to_hub requires"):
        cli.annotate(cfg)


def test_lance_storage_fails_before_vlm_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "lance"
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text('{"codebase_version": "v3.0", "storage_format": "lance"}')
    (root / "frames.lance").mkdir()
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)

    with pytest.raises(NotImplementedError, match="Parquet/MP4"):
        cli.annotate(AnnotationPipelineConfig(root=root))


def test_legacy_dataset_version_fails_before_vlm_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "legacy"
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text('{"codebase_version": "v2.1"}')
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)

    with pytest.raises(ValueError, match="Convert the dataset to v3.0 or newer"):
        cli.annotate(AnnotationPipelineConfig(root=root))


@pytest.mark.parametrize("version", [None, "", "3.1", "v3", "v3.x", "v3.01", "v4.0"])
def test_malformed_or_non_v3_version_fails_before_vlm_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: str | None,
) -> None:
    root = tmp_path / "unsupported-version"
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text(json.dumps({"codebase_version": version}))
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)

    with pytest.raises(ValueError, match="numeric v3.x"):
        cli.annotate(AnnotationPipelineConfig(root=root))


def test_malformed_metadata_fails_before_vlm_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "malformed"
    info_path = root / "meta" / "info.json"
    info_path.parent.mkdir(parents=True)
    info_path.write_text("{not-json")
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)

    with pytest.raises(ValueError):
        cli.annotate(AnnotationPipelineConfig(root=root))


def test_v31_dataset_reaches_write_scope_guard_before_vlm_creation(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert '"codebase_version": "v3.1"' in (
        fixture_dataset_root / "meta" / "info.json"
    ).read_text()
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)
    cfg = AnnotationPipelineConfig(root=fixture_dataset_root, only_episodes=(0,))

    with pytest.raises(ValueError, match="without --only_episodes"):
        cli.annotate(cfg)


def _backup_tags(api: _FakeHfApi) -> dict[str, str]:
    return {name: target for name, target in api.tags.items() if "-backup-" in name}


def test_existing_tag_without_opt_in_fails_before_vlm_or_hub_mutation(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeHfApi(tags={"v3.1": "previous-release"})
    _install_hub_fakes(monkeypatch, api)
    monkeypatch.setattr(cli, "make_vlm_client", _forbid_vlm_creation)
    cfg = AnnotationPipelineConfig(
        root=fixture_dataset_root,
        repo_id="user/dataset",
        push_to_hub=True,
    )

    with pytest.raises(RuntimeError, match="--allow_version_tag_move=true"):
        cli.annotate(cfg)

    assert cfg.allow_version_tag_move is False
    assert api.create_repo_calls == []
    assert api.upload_folder_calls == []
    assert api.card_push_calls == []
    assert api.create_tag_calls == []
    assert api.delete_tag_calls == []


def test_push_to_hub_creates_initial_tag_at_final_card_commit(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _FakeHfApi()
    _install_hub_fakes(monkeypatch, api)
    cfg = AnnotationPipelineConfig(root=fixture_dataset_root, repo_id="user/dataset")

    with caplog.at_level(logging.INFO, logger=cli.__name__):
        cli._push_to_hub(fixture_dataset_root, cfg)

    assert api.tags["v3.1"] == "card-commit"
    assert api.upload_folder_calls[0]["parent_commit"] == "base-commit"
    assert api.card_push_calls[0]["parent_commit"] == "uploaded-commit"
    assert api.create_tag_calls == [
        {
            "repo_id": "user/dataset",
            "tag": "v3.1",
            "revision": "card-commit",
            "repo_type": "dataset",
        }
    ]
    assert api.delete_tag_calls == []
    assert "with verified tag v3.1" in caplog.text


def test_push_to_hub_propagates_initial_tag_failure_without_success_log(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _FakeHfApi(create_errors=[RuntimeError("tag create failed")])
    _install_hub_fakes(monkeypatch, api)
    cfg = AnnotationPipelineConfig(root=fixture_dataset_root, repo_id="user/dataset")

    with (
        caplog.at_level(logging.INFO, logger=cli.__name__),
        pytest.raises(RuntimeError, match="Failed to create and verify required version tag"),
    ):
        cli._push_to_hub(fixture_dataset_root, cfg)

    assert api.delete_tag_calls == []
    assert "with verified tag" not in caplog.text


def test_push_to_hub_initializes_new_repo_then_guards_both_commits(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _FakeHfApi(repo_exists=False)
    _install_hub_fakes(monkeypatch, api)
    cfg = AnnotationPipelineConfig(root=fixture_dataset_root, repo_id="user/dataset")

    cli._push_to_hub(fixture_dataset_root, cfg)

    assert api.upload_folder_calls[0]["parent_commit"] is None
    assert api.card_push_calls[0]["parent_commit"] == "uploaded-commit"
    assert api.tags == {"v3.1": "card-commit"}


def test_publish_version_tag_is_noop_when_already_at_target() -> None:
    api = _FakeHfApi(tags={"v3.1": "new-release"})

    cli._publish_required_version_tag(
        api,
        repo_id="user/dataset",
        version_tag="v3.1",
        revision="new-release",
        expected_target="old-release",
        allow_version_tag_move=True,
    )

    assert api.tags == {"v3.1": "new-release"}
    assert api.create_tag_calls == []
    assert api.delete_tag_calls == []


def test_push_to_hub_moves_existing_tag_with_verified_backup(
    fixture_dataset_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _FakeHfApi(tags={"v3.1": "previous-release"})
    _install_hub_fakes(monkeypatch, api)
    cfg = AnnotationPipelineConfig(
        root=fixture_dataset_root,
        repo_id="user/dataset",
        allow_version_tag_move=True,
    )

    with caplog.at_level(logging.INFO, logger=cli.__name__):
        cli._push_to_hub(fixture_dataset_root, cfg)

    assert api.tags == {"v3.1": "card-commit"}
    assert len(api.create_tag_calls) == 2
    backup_create, version_create = api.create_tag_calls
    assert backup_create["tag"].startswith("v3.1-lerobot-align-backup-")
    assert backup_create["revision"] == "previous-release"
    assert version_create["tag"] == "v3.1"
    assert version_create["revision"] == "card-commit"
    assert [call["tag"] for call in api.delete_tag_calls] == [
        "v3.1",
        backup_create["tag"],
    ]
    assert "with verified tag v3.1" in caplog.text


def test_version_tag_delete_failure_keeps_old_tag_and_verified_backup() -> None:
    api = _FakeHfApi(
        tags={"v3.1": "old-release"},
        delete_errors=[RuntimeError("delete failed")],
    )

    with pytest.raises(RuntimeError, match="previous target .* restored and verified"):
        cli._move_version_tag_with_backup(
            api,
            repo_id="user/dataset",
            version_tag="v3.1",
            old_revision="old-release",
            new_revision="new-release",
        )

    assert api.tags["v3.1"] == "old-release"
    assert set(_backup_tags(api).values()) == {"old-release"}


def test_version_tag_create_failure_rolls_back_and_retains_backup() -> None:
    api = _FakeHfApi(
        tags={"v3.1": "old-release"},
        create_errors=[None, RuntimeError("new tag failed"), None],
    )

    with pytest.raises(RuntimeError, match="previous target .* restored and verified"):
        cli._move_version_tag_with_backup(
            api,
            repo_id="user/dataset",
            version_tag="v3.1",
            old_revision="old-release",
            new_revision="new-release",
        )

    assert api.tags["v3.1"] == "old-release"
    assert set(_backup_tags(api).values()) == {"old-release"}


def test_version_tag_rollback_failure_leaves_verified_backup() -> None:
    api = _FakeHfApi(
        tags={"v3.1": "old-release"},
        create_errors=[
            None,
            RuntimeError("new tag failed"),
            RuntimeError("rollback failed"),
        ],
    )

    with pytest.raises(RuntimeError, match="safe rollback also failed") as exc_info:
        cli._move_version_tag_with_backup(
            api,
            repo_id="user/dataset",
            version_tag="v3.1",
            old_revision="old-release",
            new_revision="new-release",
        )

    assert "restore from it manually" in str(exc_info.value)
    assert "v3.1" not in api.tags
    assert set(_backup_tags(api).values()) == {"old-release"}


def test_version_tag_verification_failure_never_overwrites_concurrent_tag() -> None:
    api = _FakeHfApi(tags={"v3.1": "old-release"})
    # Direct move reads: unique-candidate check, backup verification,
    # pre-delete guard, then final-tag verification.
    api.list_repo_refs_hooks[4] = lambda fake: fake.tags.__setitem__(
        "v3.1", "concurrent-release"
    )

    with pytest.raises(RuntimeError, match="safe rollback also failed"):
        cli._move_version_tag_with_backup(
            api,
            repo_id="user/dataset",
            version_tag="v3.1",
            old_revision="old-release",
            new_revision="new-release",
        )

    assert api.tags["v3.1"] == "concurrent-release"
    assert set(_backup_tags(api).values()) == {"old-release"}
    assert [call["tag"] for call in api.delete_tag_calls] == ["v3.1"]


def test_version_tag_change_immediately_before_delete_aborts_without_delete() -> None:
    api = _FakeHfApi(tags={"v3.1": "old-release"})
    api.list_repo_refs_hooks[3] = lambda fake: fake.tags.__setitem__(
        "v3.1", "concurrent-release"
    )

    with pytest.raises(RuntimeError, match="No deletion was attempted"):
        cli._move_version_tag_with_backup(
            api,
            repo_id="user/dataset",
            version_tag="v3.1",
            old_revision="old-release",
            new_revision="new-release",
        )

    assert api.tags["v3.1"] == "concurrent-release"
    assert set(_backup_tags(api).values()) == {"old-release"}
    assert api.delete_tag_calls == []


def test_version_tag_backup_uses_fresh_random_suffix_after_collision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collision = "v3.1-lerobot-align-backup-collision"
    generated = iter(
        [SimpleNamespace(hex="collision"), SimpleNamespace(hex="fresh-random-suffix")]
    )
    monkeypatch.setattr(cli, "uuid4", lambda: next(generated))
    api = _FakeHfApi(
        tags={
            "v3.1": "old-release",
            collision: "unrelated-release",
        }
    )

    cli._move_version_tag_with_backup(
        api,
        repo_id="user/dataset",
        version_tag="v3.1",
        old_revision="old-release",
        new_revision="new-release",
    )

    assert api.tags == {
        "v3.1": "new-release",
        collision: "unrelated-release",
    }
    assert collision not in {call["tag"] for call in api.create_tag_calls}


def test_version_tag_backup_cleanup_failure_warns_after_success(
    caplog: pytest.LogCaptureFixture,
) -> None:
    api = _FakeHfApi(
        tags={"v3.1": "old-release"},
        delete_errors=[None, RuntimeError("cleanup failed")],
    )

    with caplog.at_level(logging.WARNING, logger=cli.__name__):
        cli._move_version_tag_with_backup(
            api,
            repo_id="user/dataset",
            version_tag="v3.1",
            old_revision="old-release",
            new_revision="new-release",
        )

    assert api.tags["v3.1"] == "new-release"
    assert set(_backup_tags(api).values()) == {"old-release"}
    assert "could not be removed" in caplog.text
