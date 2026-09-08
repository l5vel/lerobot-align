"""An export must never carry deleted source material or local secrets."""

import hashlib
import json
import subprocess

import pytest

from scripts.export_release import export_source


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Synthetic Test")
    git(root, "config", "user.email", "test@example.invalid")
    (root / "removed-source.json").write_text('{"private_annotation": "sentinel"}')
    git(root, "add", ".")
    git(root, "commit", "-qm", "historical source")
    git(root, "rm", "removed-source.json")
    (root / "README.md").write_text("Reviewed public source\n")
    (root / "release").mkdir()
    (root / "release/source-files.txt").write_text("README.md\nrelease/source-files.txt\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "source boundary")
    return root


def test_export_ignores_history_secrets_dirty_files_and_unlisted_tracked_files(repository, tmp_path):
    (repository / ".env").write_text("SYNTHETIC_SECRET=do-not-export")
    (repository / "unrelated").mkdir()
    (repository / "unrelated/cached.json").write_text("cached private input")
    git(repository, "add", "unrelated")
    git(repository, "commit", "-qm", "unlisted tracked artifact")
    (repository / "README.md").write_text("uncommitted private note")
    destination = tmp_path / "public"
    receipt = export_source(repository, destination)
    assert (destination / "README.md").read_text() == "Reviewed public source\n"
    assert {str(p.relative_to(destination)) for p in destination.rglob("*") if p.is_file()} == {
        "README.md", "release/source-files.txt", "release-export.json",
    }
    assert not (destination / ".git").exists()
    assert json.loads((destination / "release-export.json").read_text()) == receipt
    for row in receipt["files"]:
        assert hashlib.sha256((destination / row["path"]).read_bytes()).hexdigest() == row["sha256"]


@pytest.mark.parametrize("entry", ["../private", ".env", "evaluation/results/inputs.tar.gz", "missing", "link"])
def test_invalid_manifest_is_rejected_before_creating_destination(repository, tmp_path, entry):
    (repository / "link").symlink_to("README.md")
    manifest = repository / "release/source-files.txt"
    manifest.write_text(manifest.read_text() + entry + "\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "invalid manifest")
    destination = tmp_path / "public"
    with pytest.raises(ValueError):
        export_source(repository, destination)
    assert not destination.exists()


def test_export_does_not_replace_existing_destination(repository, tmp_path):
    destination = tmp_path / "public"
    destination.mkdir()
    sentinel = destination / "README.md"
    sentinel.write_text("existing work")
    with pytest.raises(FileExistsError):
        export_source(repository, destination)
    assert sentinel.read_text() == "existing work"


def test_git_read_failure_leaves_no_partial_export(repository, tmp_path, monkeypatch):
    import scripts.export_release as exporter

    original = exporter.git
    calls = 0

    def fail_on_second_file(*args):
        nonlocal calls
        if args[1:3] == ("cat-file", "blob"):
            calls += 1
            if calls == 3:
                raise OSError("injected object read failure")
        return original(*args)

    monkeypatch.setattr(exporter, "git", fail_on_second_file)
    with pytest.raises(OSError, match="injected"):
        export_source(repository, tmp_path / "public")
    assert not (tmp_path / "public").exists()
    assert not list(tmp_path.glob(".align-export-*"))
