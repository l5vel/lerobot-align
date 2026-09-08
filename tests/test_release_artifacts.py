"""Distribution boundary regression tests; source material must stay private."""
import json
import gzip
import hashlib
from pathlib import Path
import re

import pytest


@pytest.mark.parametrize(("corpus", "count"), [("a", 16), ("b", 1)])
def test_released_corpora_have_verified_immutable_upstream_identifiers(corpus, count):
    """Portable local roots alone cannot identify the historical source bytes."""
    root = Path(__file__).parents[1]
    manifest = json.loads((root / f"evaluation/configs/corpus_{corpus}_sources.json").read_text())
    assert len(manifest["sources"]) == count
    check = manifest["hub_verification"]
    assert check["authentication"] == "none"
    assert check["file_downloads_tested"] is False
    assert manifest["reproduction_status"] == (
        "source_revisions_verified_historical_artifacts_not_distributed"
    )
    for source in manifest["sources"]:
        upstream = source["upstream"]
        assert re.fullmatch(r"[\w.-]+/[\w.-]+", upstream["repo_id"])
        assert re.fullmatch(r"[0-9a-f]{40}", upstream["revision"])
        assert upstream["hub"]["http_status"] == 200
        assert upstream["hub"]["resolved_sha"] == upstream["revision"]
        assert upstream["hub"]["private"] is False
        assert upstream["hub"]["gated"] is False
        if corpus == "a":
            assert upstream["repo_id"].split("/")[1] == source["id"]
            assert re.fullmatch(r"[0-9a-f]{64}", upstream["annotation_sidecar"]["sha256"])
            assert upstream["local_historical_match"]["ground_truth_episodes_equal"] is True
        else:
            # The API has no license tag; do not manufacture one from NOTICE
            # or from the software license. The source card has separate terms.
            assert upstream["hub"]["license_metadata"] is None
            assert upstream["local_historical_match"]["matched_episodes"] == len(source["tasks"])
            assert upstream["local_historical_match"]["mismatched_episodes"] == 0


def test_release_tree_does_not_redistribute_raw_input_archives():
    root = Path(__file__).parents[1]
    inventory_path = root / "evaluation/release_artifact_inventory.json"
    if not inventory_path.exists():
        # The package sdist intentionally omits all study results and inputs.
        assert not (root / "evaluation/results").exists()
        return
    inventory = json.loads(inventory_path.read_text())
    assert len(inventory["archives"]) == 2
    for archive in inventory["archives"]:
        assert not (root / archive["archive"]).exists()
        assert archive["counts"]["source_annotations"] > 0
        for member in archive["members"]:
            assert len(member["sha256"]) == 64
            assert not {"subtasks", "labels", "start", "end"}.intersection(member)
    changes = inventory["structured_result_redactions"]
    if not (root / "evaluation/results").exists():
        # A clean source export carries the inventory as a removal record, not
        # the historical results. None of those artifacts may be reintroduced.
        assert all(not (root / change["path"]).exists() for change in changes)
        return
    for change in changes:
        path = root / change["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == change["release_sha256"]
        if path.name.endswith(".jsonl.gz"):
            with gzip.open(path, "rt") as stream:
                for line in stream:
                    row = json.loads(line)
                    assert "true_sequence" not in row
                    assert "pred_sequence" not in row
                    assert all("text" not in span for span in row.get("predictions", []))
