import json

import pytest

from lerobot_align.config import AnnotationPipelineConfig
from lerobot_align.publication import (
    copy_publication_files,
    publication_manifest,
    source_dataset_card,
)


def test_manifest_excludes_secrets_staging_caches_and_unrelated_files(tmp_path):
    root = tmp_path / "dataset"
    good = ["meta/info.json", "meta/tasks.parquet", "meta/stats.json",
            "meta/episodes/chunk-000/file-000.parquet", "data/chunk-000/file-000.parquet",
            "videos/observation.images.camera/chunk-000/file-000.mp4", "LICENSE", "NOTICE"]
    bad = [".env", ".env.local", "credentials.json", "token", "README.md", "notes.txt",
           ".annotate_staging/episode_000000/plan.jsonl", "custom_staging/plan.jsonl",
           "custom_staging/data/chunk-000/file-000.parquet", ".cache/secret.json",
           "data/random.parquet", "data/chunk-000/file-000.parquet.tmp", "meta/private.json",
           "videos/unknown/chunk-000/file-000.mp4", "videos/.cache/a.mp4"]
    for name in good + bad:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic content")
    (root / "meta/info.json").write_text(json.dumps({"features": {
        "observation.images.camera": {"dtype": "video"}}}))
    manifest = publication_manifest(root, root / "custom_staging")
    assert set(map(str, manifest)) == set(good)
    upload = tmp_path / "upload"
    copy_publication_files(root, upload, manifest)
    assert {str(p.relative_to(upload)) for p in upload.rglob("*") if p.is_file()} == set(good)


@pytest.mark.parametrize("name", ["meta/info.json", "data/chunk-000/file-000.parquet", "LICENSE"])
def test_publication_rejects_symlinked_allowed_files(fixture_dataset_root, tmp_path, name):
    root = fixture_dataset_root
    path = root / name
    target = tmp_path / "private"
    target.write_text('{"features": {}}')
    path.unlink(missing_ok=True)
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlink"):
        publication_manifest(root, root / "staging")


def test_staging_cannot_overlap_dataset_files(fixture_dataset_root):
    with pytest.raises(ValueError, match="Staging overlaps"):
        publication_manifest(fixture_dataset_root, fixture_dataset_root / "data")


def test_card_preserves_source_license_attribution_and_custom_metadata(tmp_path):
    original = "---\nlicense: cc-by-nc-sa-4.0\ncustom: provenance\n---\n# Source\n\nCredit Alice.\n[Terms](https://example.org/terms)\n"
    (tmp_path / "README.md").write_text(original)
    cfg = AnnotationPipelineConfig(repo_id="source/dataset", new_repo_id="target/annotated")
    card = source_dataset_card(tmp_path, cfg)
    assert card.data.license == "cc-by-nc-sa-4.0"
    assert card.data.get("custom") == "provenance"
    assert "Credit Alice." in card.text
    assert "[Terms](https://example.org/terms)" in card.text
    assert "https://huggingface.co/datasets/source/dataset" in card.text
    assert "apache-2.0" not in str(card)
    assert (tmp_path / "README.md").read_text() == original
    assert cfg.push_private is True


def test_missing_license_is_not_inferred(tmp_path):
    with pytest.raises(ValueError, match="no license"):
        source_dataset_card(tmp_path, AnnotationPipelineConfig())
    card = source_dataset_card(tmp_path, AnnotationPipelineConfig(dataset_license="cc-by-4.0"))
    assert card.data.license == "cc-by-4.0"


def test_license_override_cannot_replace_source_license(tmp_path):
    (tmp_path / "README.md").write_text("---\nlicense: cc-by-4.0\n---\nCredit\n")
    with pytest.raises(ValueError, match="conflicts"):
        source_dataset_card(tmp_path, AnnotationPipelineConfig(dataset_license="apache-2.0"))


def test_cli_uploads_only_the_manifest(fixture_dataset_root, monkeypatch):
    from lerobot_align import cli
    from tests.test_cli import _FakeHfApi, _install_hub_fakes

    root = fixture_dataset_root
    (root / ".env").write_text("SYNTHETIC_SECRET=do-not-upload")
    (root / "custom_staging").mkdir()
    (root / "custom_staging/private.json").write_text("private work")
    api = _FakeHfApi(repo_exists=False)
    _install_hub_fakes(monkeypatch, api)
    uploaded = []
    original = api.upload_folder

    def capture(**kwargs):
        from pathlib import Path
        directory = Path(kwargs["folder_path"])
        assert directory != root
        uploaded.extend(str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file())
        return original(**kwargs)

    monkeypatch.setattr(api, "upload_folder", capture)
    cli._push_to_hub(root, AnnotationPipelineConfig(
        root=root, new_repo_id="user/annotated", staging_dir=root / "custom_staging"))
    assert ".env" not in uploaded
    assert "custom_staging/private.json" not in uploaded
    assert "meta/info.json" in uploaded
    assert api.create_repo_calls[0]["private"] is True
