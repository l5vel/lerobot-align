# Reproducing an evaluation

The release contains evaluation software, configuration and the removal inventory.
It contains no dataset, copied source annotations, historical predictions or fits.
Access and redistribution terms belong to each source dataset independently of
the software's Apache-2.0 license. See [RELEASE_DATA.md](RELEASE_DATA.md).

## Install and validate locally

From the source root, with Python 3.12, uv and FFmpeg installed:

```bash
uv sync --locked --extra dev
uv run --no-sync pytest -q evaluation
uv run --no-sync python evaluation/scripts/prepare_alignment_study.py --help
uv run --no-sync python evaluation/scripts/evaluate_alignment_study.py --help
```

The full development extra includes evaluation dependencies. For an evaluation
installation without development tools, use `uv sync --locked --extra eval`.
Set `LEROBOT_VLM_API_KEY` for an authenticated model endpoint; do not put keys
in command lines, source manifests or result directories.

## Inputs and historical limitations

Corpus A comprises 16 component datasets grouped into four task families.
The [source manifest](configs/corpus_a_sources.json) records each upstream
repository ID, immutable revision and original annotation-sidecar SHA-256.
Local snapshots match the historical annotation inputs: 799 total episodes,
797 with annotations. Historical study selections have their own exclusions.

Corpus B originates at `InternRobotics/RoboInter-Data`, revision
`0208b79c34eca4d214cdb5d07f5ae0f7cc634340`. The
[source manifest](configs/corpus_b_sources.json) records its RH20T subdirectory
and verification summary. All 168 local download records name this revision;
original Parquet annotations reconstruct all 1,695 historical study ground-truth
episodes exactly. Do not treat a prepared local directory as proof of origin.

On 2026-09-08, anonymous Hub metadata requests returned HTTP 200 and the exact
requested commit for all 17 sources, with `private=false` and `gated=false`.
These were repository metadata checks, not full file downloads or an independent
preparation rerun. Corpus B's [pinned card](https://huggingface.co/datasets/InternRobotics/RoboInter-Data/blob/0208b79c34eca4d214cdb5d07f5ae0f7cc634340/README.md)
still declares community and underlying dataset terms; public availability is
not a grant of redistribution rights. Review each source card and obtain any
required author permission before downloading or reusing data.

The historical saved model predictions and calibration fits remain private.
Hashes in [release_artifact_inventory.json](release_artifact_inventory.json)
can verify supplied artifacts; they cannot reconstruct missing bytes. Model
reruns, including temperature-zero runs on a different runtime, are new
experiments. The archived performance numbers in the root README are not
independently reproducible from this source release alone.

## Preparation with authorized input access

Keep downloaded and prepared data outside the source checkout. To download
Corpus A's pinned snapshots after reviewing their terms, choose a new directory
and run this from the source root. This downloads the datasets, including videos;
allow for substantial disk space. It does not reproduce historical predictions.

```bash
export ALIGN_CORPUS_A_DOWNLOAD="$HOME/lerobot-align-data/corpus-a-original"
uv run --no-sync python - <<'PY'
import hashlib
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

manifest = json.loads(Path("evaluation/configs/corpus_a_sources.json").read_text())
destination = Path(os.environ["ALIGN_CORPUS_A_DOWNLOAD"])
destination.mkdir(parents=True, exist_ok=False)
for source in manifest["sources"]:
    upstream = source["upstream"]
    root = destination / source["id"]
    snapshot_download(repo_id=upstream["repo_id"], repo_type="dataset",
                      revision=upstream["revision"], local_dir=root, token=False)
    sidecar = upstream["annotation_sidecar"]
    if hashlib.sha256((root / sidecar["path"]).read_bytes()).hexdigest() != sidecar["sha256"]:
        raise RuntimeError(f"{source['id']}: original annotation checksum mismatch")
PY
```

Corpus A's `prepare_dataset.py` requires nonempty original sidecars and checks
any existing sibling `.done` marker for evidence of generated/imported outputs.
It extracts ground truth and strips annotations from a separate working copy;
keep the downloaded original for the shared protocol's annotated-source input.
Markers are optional, and do not independently prove who annotated a dataset.
Do not invent markers or bypass provenance checks. Check its positional source
and target arguments and required `--gt-out` with `--help`.

For Corpus B, use the pinned download command in [RELEASE_DATA.md](RELEASE_DATA.md).

For Corpus B, run these tools in order, using each tool's `--help` to select
your input and new output directories:

1. `census_robointer.py`: inventory the downloaded RH20T source.
2. `select_corpus_b.py`: select episodes and write splits in original indices.
3. `build_corpus_b_subset.py`: assemble a v2.1 subset and retain its index map.
4. `uv run --no-sync python -m lerobot.scripts.convert_dataset_v21_to_v30`:
   convert the subset with the installed matching LeRobot tool.
5. `prepare_robointer.py`: extract ground truth from the converted v3 root.
   Its required `--provenance-verified` assertion means the source boundary
   annotation method has been checked; it does not assert redistribution rights.
6. For the historical per-task layout, `build_corpus_b_components.py`,
   `translate_corpus_b_splits.py` and `shard_corpus_b_by_task.py` construct
   components and translate between original, subset and per-task indices.

Retain source repository IDs/revisions, index maps, selection reports, checksums
and the exact commands privately alongside the data. Historical selection
cannot be recovered from the removed archives' hashes alone.

## Shared alignment protocol

Set `ALIGN_CORPUS_A_ROOT` to the parent of the 16 prepared source roots and
`ALIGN_CORPUS_A_GT` to the directory containing their ground-truth JSON files.
For Corpus B, set `ALIGN_CORPUS_B_ROOT` to its prepared mixed-task v3 dataset
and `ALIGN_CORPUS_B_GT` to its ground-truth JSON file. Missing variables fail
before preparation. For new datasets, create the small manifest described in
[calibration_protocol.md](calibration_protocol.md) with your own task taxonomy.

Use that protocol's preparation and evaluation commands. Start the same model
on the configured localhost ports first; the evaluator does not start a server.
Record the model weight revision and server image digest separately from the
served alias. Preparation validates label/timestamp agreement and reserves ten
seed trajectories per task. Do not overwrite frozen fits or reuse calibration
with a different input/prompt fingerprint. A failed batch returns nonzero while
retaining completed rows and failure records.

For the documented Qwen2.5/vLLM image, export
`LEROBOT_VLM_VIDEO_METADATA_SOURCE=client` and
`LEROBOT_OPENAI_SEND_MM_KWARGS=1` before evaluation. Use metadata source `server`
for Qwen3-VL adapters. New evaluation fingerprints include this choice.

The legacy `run_all.sh` is a site-specific historical driver. The shared Python
drivers above are the supported entry point for a new evaluation.
