# lerobot-align

**`lerobot-align`** is a steerable multimodal annotation, given-subtask alignment, and boundary calibration pipeline for robotics datasets in [LeRobot](https://github.com/huggingface/lerobot) format.

It bridges Vision-Language Models (VLMs) like Qwen-VL with robotics trajectory data, enabling high-precision subtask segmentation, visual question answering, speech/interjection generation, and trajectory keyframe alignment.

---

## Key Features

1. **Given-Subtask Alignment**:
   - Provide fixed subtask labels via JSON, or feed labels discovered by the
     generator directly into the same aligner.
   - The VLM predicts precise temporal boundaries in a single pass.
   - Generated-label realignment keeps the generated boundaries if refinement
     loses or reorders labels; direct fixed-label runs retain the configurable
     `subtask_align_min_fraction` threshold.

2. **Native Video Encoding**:
   - Use `subtask_generate_frame_format=video` for open-ended subtask discovery, or `subtask_align_frame_format=video` when labels are supplied.
   - Encodes sampled frames into a single lossless (`crf=0`) H.264 video clip.
   - Takes advantage of native VLM video timestamp tokens (e.g. `<12.3 seconds>`), eliminating visual clock badge OCR errors.
   - Native-video alignment can also reduce prompt-processing time compared with contact sheets.

3. **Motion-Adaptive Sampling & Stacked Multiview**:
   - Samples keyframes adaptively based on robot joint velocity and motion peaks.
   - Stacks multiple synchronized camera views into single composite frames without diluting temporal resolution.

4. **Dynamic Programming Boundary Calibration**:
   - Solves monotonic boundaries simultaneously via Dynamic Programming using dataset-specific duration priors.
   - Quality guards: warns if boundary moves > 3 residual $\sigma$; calibration fitter refuses calibrations unless empirical error drops significantly.

5. **Flexible Subtask Import**:
   - Import subtasks from SARM metadata, previous `language_persistent` runs, raw `task_index` runs, or `meta/lerobot_annotations.json`.

---

## Installation

### Prerequisites

- **Python 3.12.** The supported baseline (`requires-python = ">=3.12"`).
- **FFmpeg**, on `PATH`. It is a hard requirement, not an optional extra:
  frame decoding and the native-video encoder depend on it, CI installs it
  before anything else runs (`.github/workflows/ci.yml`), the Hugging Face Jobs
  pod installs it into the remote runtime (`src/lerobot_align/jobs.py`), and
  `tests/test_frames.py` skips its video fixtures when it is missing.
- **An OpenAI-compatible VLM endpoint.** See [Serving a model](#serving-a-model).
  The machine that runs `lerobot-align` does not itself need a GPU.

Use the repository root or extract the versioned source distribution
`lerobot_align-0.1.0.tar.gz` and enter `lerobot_align-0.1.0/`. The package is
not yet on PyPI. With [uv](https://docs.astral.sh/uv/getting-started/installation/)
and FFmpeg installed, the canonical quick start is:

```bash
uv sync --locked --extra dev
uv run --no-sync lerobot-align --help
uv run --no-sync python -m tests.run_e2e_smoke
```

No environment activation is needed: all commands use `uv run --no-sync`.
The smoke run creates an original synthetic MP4, runs the installed CLI through
a localhost HTTP test server, checks both contact sheets and native video,
then validates VQA, interjections, plans, memory and the rewritten dataset.
It needs no GPU, token, model download or paid service. Its deterministic
responses verify plumbing; they do not measure model accuracy. Expected output
includes `contact_sheet: ... passed` and `video: ... passed`.

For client-only use, `uv sync --locked` omits development tools. `.[serve]`
installs a Transformers server, **not vLLM**; `.[viz]` adds plotting tools.
The optional Docker workflow below provides vLLM separately and avoids mixing
its CUDA dependencies with the annotation environment.

Input datasets must use the LeRobot v3.x Parquet/MP4 layout. Lance-backed
datasets and legacy v2.1 datasets are rejected before model startup; convert
them with the matching LeRobot migration tool first.

---

## Serving a model

`lerobot-align` is an HTTP client. It never loads model weights: `torch` is
present only as the tensor container LeRobot hands frames back in, and every
inference call goes to an OpenAI-compatible `/v1/chat/completions` endpoint.
Any such endpoint works, including a hosted one — point `--vlm.api_base` at it
and supply `--vlm.api_key`. The examples below use a local vLLM server because
the native-video modes need a video-capable backend.

### Optional GPU server for real annotation

The examples use the same model, `Qwen/Qwen2.5-VL-7B-Instruct`, documented by
[its publisher](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct).
On a Linux NVIDIA GPU host with Docker and NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  vllm/vllm-openai@sha256:2622f38a0aa646c15ccc27bd5033911a58fd94ac69fd8f86aba0692d77cfe5b9 \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --revision cc594898137f460bfe9f0759e9844b3ce807cfb5 \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 32768 \
  --media-io-kwargs '{"video":{"num_frames":-1}}'
```

The container listens on its own interfaces; Docker exposes it only on the
host's loopback interface. Model weights are downloaded on first startup.
This GPU step is separate from the CPU smoke test. Check readiness with
`curl --fail http://127.0.0.1:8000/v1/models` and confirm the model ID before
running annotation. For an authenticated endpoint, set `LEROBOT_VLM_API_KEY`
in the environment; avoid putting credentials in command history.

Native video requires the server's `num_frames=-1` setting above and these
client environment variables for the documented Qwen2.5/vLLM image:

```bash
export LEROBOT_OPENAI_SEND_MM_KWARGS=1
export LEROBOT_VLM_VIDEO_METADATA_SOURCE=client
```

The metadata setting is also available as `--vlm.video_metadata_source=client`.
Use `server` (the default) with Qwen3-VL adapters, which supply their own
metadata; sending a second copy to those adapters causes a processor error.
For locally encoded clips the
client forwards actual FPS and frame indices and disables further sampling;
an explicit different processor FPS remains an intentional resampling request.
Contact-sheet requests need
neither setting. See [vLLM multimodal inputs](https://docs.vllm.ai/en/latest/features/multimodal_inputs/).

To check your running server with original synthetic media, run:

```bash
uv run --no-sync python -m tests.run_live_vlm_smoke \
  --api-base=http://127.0.0.1:8000/v1 --model=Qwen/Qwen2.5-VL-7B-Instruct \
  --video-metadata-source=client --output=/tmp/align-live-check
```

The output directory must be new. This runs generation and fixed-label
alignment with both visual formats and checks dataset integrity. It does not
measure annotation quality or provision a server.

`--vlm.auto_serve=false` is the default. Optional `auto_serve=true` launches
an installed `vllm` or `transformers` executable and binds it to loopback.
Hugging Face Jobs start their own server unless explicitly disabled; their
remote package and input-file requirements are documented in
[the pipeline guide](ANNOTATION_PIPELINE.md#running-on-hugging-face-jobs).

### A multi-replica cluster

[`scripts/start_qwen38.sh`](scripts/start_qwen38.sh) is the launcher used for
the evaluation study: one independent vLLM replica per GPU inside a `tmux`
session, with ports assigned in replica order from `8000`. It is a reference
implementation tuned to one 8xH100 host rather than part of the installable
package, and it has prerequisites of its own — `tmux`, `nvidia-smi`, `curl`,
and a separate vLLM virtualenv (`.venv-vllm` by default). Read
[`scripts/README.md`](scripts/README.md) for its known rough edges first.

---

## Annotating your dataset

### 1. Given-Subtask Alignment (Native Video Mode)

```bash
LEROBOT_OPENAI_SEND_MM_KWARGS=1 \
uv run --no-sync lerobot-align \
  --root=/path/to/my_robot_dataset \
  --plan.subtasks_path=/path/to/subtasks.json \
  --plan.subtask_align_frame_format=video \
  --plan.subtask_video_fallback=error \
  --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct \
  --vlm.api_base=http://127.0.0.1:8000/v1 \
  --vlm.auto_serve=false
```

Fixed-label alignment always uses one whole-episode call. Contact sheets are
the default visual representation; native video keeps the same ordered-label
contract while using the model's video timeline. When multiple alignment
cameras are configured, synchronized views are stacked within each native
video frame rather than being treated as consecutive moments.

`subtask_video_fallback=error` makes native video a hard contract: if a real,
non-empty `video_url` clip cannot be built, the stage fails before sending a
contact-sheet request. The default, `contact_sheet`, preserves compatibility
by warning and using the already-decoded frames as timestamped sheets.

`--root` is rewritten with a recoverable dataset transaction. All transformed
data shards and `meta/info.json` are staged and validated before replacement;
checksummed originals are retained until the commit is durable. Reserve disk
space for the staged data and originals (up to twice the data-shard size).
Stop external dataset readers during a run: multiple file replacements cannot
be made atomically visible to readers that do not use the run lock. After an
interruption, the next run rolls back before reading data. To recover without
starting inference, run `uv run --no-sync lerobot-align-recover /path/to/dataset`.
Do not delete `.lerobot-align-transaction` while recovery is pending.

Work on a copy, or add `--new_repo_id` and
`--push_to_hub=true` when intentionally publishing a separate Hub dataset.
The first annotation pass must cover the full dataset so every packed Parquet
shard receives the canonical language schema. `--only_episodes` is safe only
for later reruns of a dataset that has already completed that full migration.
Publishing over an existing versioned Hub destination is refused by default;
after reviewing the old release, opt in explicitly with
`--allow_version_tag_move=true`. The publisher creates and verifies a recovery
tag before moving the LeRobot version tag and rolls it back on failure.

Hub publication uses an explicit manifest: canonical Parquet shards, declared
video-camera shards, standard LeRobot metadata, and root LICENSE/NOTICE files.
Unrelated files, `.env`, caches and staging are excluded. Symlinked publication
files are rejected. Custom sidecars are not uploaded implicitly.
New destinations are private by default; use `--push_private=false` only when
intentionally making a new dataset public. This flag does not change an existing
repository's visibility. The source README/card, license, attribution and links
are retained, with annotation provenance appended. If its license is missing,
add the original dataset license to the card or supply `--dataset_license`
after checking its terms. The software license is never assigned to data.

### 2. Open-Ended Generation + Native-Video Realignment

Omit `--plan.subtasks_path` to discover labels from the episode. The example
below then passes those generated labels directly to the fixed-label aligner,
so no intermediate JSON file is needed:

```bash
LEROBOT_OPENAI_SEND_MM_KWARGS=1 \
uv run --no-sync lerobot-align \
  --root=/path/to/my_robot_dataset \
  --plan.subtask_generate_frame_format=video \
  --plan.subtask_realign_generated=true \
  --plan.subtask_align_frame_format=video \
  --plan.subtask_video_fallback=error \
  --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct \
  --vlm.api_base=http://127.0.0.1:8000/v1 \
  --vlm.auto_serve=false
```

Contact sheets remain the default for both stages. With the settings above,
the pipeline runs **describe → segment → fixed-label realignment**. Optional
seeded relabeling runs between segmentation and realignment, so the aligner
receives the revised ordered labels. The rough boundaries produced during
segmentation are replaced by one independent whole-episode alignment call
only if it preserves the exact ordered label list, including repeated labels.
If the alignment is empty, below the coverage threshold, or loses/reorders a
label after frame snapping, the tool logs a warning and keeps all generated
boundaries. It never publishes the partial refinement. Inference failures and
strict video-input errors still fail the run.

Generation keeps its automatic episode windowing at the configured sampling
density. When a greedy split would leave a tiny final clip, the last two
windows share their remaining frame intervals evenly (for example, the
51.18-second/2-fps/100-frame case becomes 52 + 52 frames instead of 100 + 4).
Windows are processed independently, so an action crossing a seam may still
produce split or repeated labels; raise `max_frames_per_prompt` when context
allows if that matters for the dataset. Realignment is never windowed:
it samples the whole episode into one call, reducing temporal density as needed
to honor the same frame budget. Task derivation also remains one whole-episode
call capped to that budget. Omit `subtask_realign_generated` to keep the rough
generation boundaries instead.

For an external vLLM server, the environment variable above forwards the
clip's encoded FPS to its processor; also start the server with
`--media-io-kwargs '{"video":{"num_frames":-1}}'`. Without both controls,
vLLM may silently resample the clip or cap it (commonly at 32 frames). A vLLM
process spawned by the tool's default command enables both controls
automatically.

The client inlines each MP4 as base64 for OpenAI-compatible endpoints. If a
large custom frame budget hits proxy request limits or client RAM pressure,
lower `max_frames_per_prompt` and/or `executor.episode_parallelism`.

In every visual generation mode, frame-provider initialization and decode
failures abort before the VLM call and include the underlying error. Under the
default fallback policy, a clip encoder failure may use contact sheets only
after a complete frame set was decoded; the tool never turns a failed
native-video read into a text-only generation request. Use
`subtask_video_fallback=error` when experiments must guarantee native-video
transport rather than merely prefer it.

### 3. Fit Boundary Calibration on Ground-Truth Annotations

```bash
LEROBOT_OPENAI_SEND_MM_KWARGS=1 uv run --no-sync lerobot-align-eval-batch /path/to/my_robot_dataset \
  --episodes 0 1 2 3 4 5 6 7 8 9 \
  --formats video \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --camera observation.images.top \
  --api-base http://127.0.0.1:8000/v1 \
  --out alignment-results.json

uv run --no-sync lerobot-align-fit \
  /path/to/my_robot_dataset \
  alignment-results.json \
  --episodes 0 1 2 3 4 5 6 7 8 9 \
  --min-fit-episodes 3 \
  --out calibration.json
```

These diagnostics use the human intervals in
`meta/lerobot_annotations.json` as ground truth.

`lerobot-align-eval-batch` and `lerobot-align-eval` call a model, so they need
a server already running at `--api-base` and a dataset that really contains the
camera named by `--camera`. Name both explicitly: an omitted `--model` falls
back to the package default, and an omitted `--camera` resolves to the
dataset's first decodable video camera, which is often a wrist close-up that
cannot see the event being annotated. `lerobot-align-fit` is offline — it reads
the JSON written by `lerobot-align-eval-batch` and accepts neither flag.

Evaluation CLIs read `LEROBOT_VLM_API_KEY` by default; `--api-key-env NAME`
selects another environment variable without placing its value in arguments.
`lerobot-align-eval --labels-file labels.json` accepts your own ordered labels;
`--labels user` requires that file. Empty selections, invalid settings and any
failed batch episode/format return a nonzero exit code; successful rows remain
in the JSONL sidecar.

Every new batch row records a stable configuration fingerprint, model identifier,
sampling and camera settings, package versions, code hashes, loaded prompt hashes
(including environment overrides), and input content hashes (hashing large videos
can take time). Endpoint identity is hashed without credentials. Record immutable
model revisions and server settings alongside your experiment: an endpoint and
model alias do not prove which weights a remote server loaded. The fitter refuses mixed
configurations before deduplicating retries. Select one with
`--config-fingerprint SHA256` or keep separate result files. A random `run_id`
still identifies each attempt within that configuration. Historical rows without
this information require `--allow-legacy-provenance`; the output records that
its configuration provenance could not be verified. Regenerate such inputs when
possible. Calibration also records source-result and ground-truth checksums.

**If a scoring run dies partway.** `lerobot-align-eval-batch` appends every row
to an `--out`-adjacent `.jsonl` as it completes, so the episodes that did finish
survive the crash. Pass that sidecar to `lerobot-align-fit` in place of the
`--out` array — it accepts either form:

```bash
uv run --no-sync lerobot-align-fit /path/to/my_robot_dataset alignment-results.jsonl \
  --episodes 0 1 2 3 4 5 6 7 8 9 --out calibration.json
```

Because the sidecar is append-only, re-running the scorer into the same `--out`
leaves two rows for every repeated episode. The later row wins — a re-scored
episode supersedes the stale attempt before it — with one exception: a row that
records an error never displaces one that carries spans, so a retry against a
flaky endpoint cannot quietly shrink the fitting cohort. Every resolved
collision is printed and written into `calibration.json` under
`resolved_collisions`, next to `source_results_format`.

Rows carry the `run_id` of the scoring run that produced them. Check it before
fitting from a sidecar you did not write in one pass: the file records nothing
else about `--model`, `--fps` or `--camera`, so two runs under different
conditions look identical once they share it.

**The ground-truth schema.** `meta/lerobot_annotations.json` maps a **string**
episode index to that episode's ordered subtask intervals. Times are seconds
from the start of the episode, `end` must exceed `start`, and the label key is
`label`:

```json
{
  "episodes": {
    "0": {
      "subtasks": [
        {"label": "open the fridge door", "start": 0.0, "end": 3.4},
        {"label": "pick up the red can", "start": 3.4, "end": 7.1}
      ]
    }
  }
}
```

All three diagnostics read this `subtasks` form only. The importer
(`--plan.subtask_import=lerobot_annotations`) additionally accepts an `atoms`
form, in which an episode carries timestamped rows of
`{"style": "subtask", "content": ..., "timestamp": ...}` and each interval's
end is reconstructed from the next start; a non-empty `subtasks` list wins when
both fields are present.

Reserve at most ten trajectories per task, fit once, then use the frozen file
on the remaining trajectories. The explicit `--episodes` allowlist excludes
all other prediction rows. The [task calibration workflow](evaluation/calibration_protocol.md)
enforces a shared task budget across count cohorts and camera arms, validates
frozen files, and records actual runtime application.

The fitter selects the most frequent exact ordered label tuple, breaking ties
lexicographically. With fewer than three fitting episodes it uses duration
weight `1.0` and reports that fallback; an explicit `--duration-weight` (including
`0`) takes precedence. Such a tiny fit has no held-out gain estimate.

By default (`--label-scope exact`), calibration applies only when an episode's
complete ordered labels exactly match the recorded labels. Explicit
`--label-scope segment_count` pools differently worded seed labels within one
task and count, and applies by count. This assumes comparable task phases at
each position; count alone cannot establish task identity. Open-ended wording
can vary between episodes, so a
calibration used with generated-label realignment will commonly be skipped
unless generation produces the same canonical label list. Legacy bare-offset
calibration files remain accepted for directly supplied fixed labels, but are
ignored with a warning in generated-label realignment because they carry no
schema to verify.

### 4. Evaluate Alignment Against Ground Truth

```bash
lerobot-align-eval \
  /path/to/my_robot_dataset \
  --episode 0 \
  --labels gt \
  --frame-format video \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --camera observation.images.top \
  --api-base http://127.0.0.1:8000/v1
```

Same requirements as section 3: a running server, and a dataset whose
`meta/info.json` declares `<CAMERA-KEY>`.

### 5. Motion-Adaptive Keyframe Visualization

```bash
lerobot-motion-viz \
  /path/to/my_robot_dataset \
  --episodes 0 \
  --output-dir=./viz
```

This one needs no server and takes no `--model` or `--camera`: it plots joint
velocity and the timestamps the sampler would pick, using a stub client. It
does need the plotting extra (`uv pip install -e ".[viz]"`).

---

## Completeness and Partial Publication

Completeness validation rejects individual episodes with missing required
module outputs. Other episodes still publish, and existing language annotations
on rejected episodes are preserved (empty on a first run). The writer migrates
all selected shards to the canonical schema, including shards containing only
rejected episodes, so the dataset metadata stays consistent.

Free-form task augmentation (`plan.n_task_rephrasings`, default 10) requests
alternative wordings of the episode task, in addition to the canonical task.
Filtering and deduplication can leave fewer than the requested alternatives.
This produces a warning and preserves an otherwise-valid episode's subtask
annotations, even if only the canonical task phrasing remains. Entirely missing
`task_aug` output or output above the configured cap still fails completeness.
Required subtask, plan, and memory output and the separate structured
`task_aug_axes` contract remain enforced.

The executor prints the validation summary and the first 20 errors and warnings,
and saves all errors, warnings, and skipped episode IDs in `<staging_dir>/validation_report.json`
(by default, `<root>/.annotate_staging/validation_report.json`). The run summary
also includes `skipped_episodes` and a validation phase with processed/skipped
counts. `--skip_validation=true` never allows incomplete episodes to publish.
All-incomplete runs always fail before parquet writes. Set
`--executor.max_incomplete_episode_fraction=0.1` to also abort when more than
10% of episodes are incomplete; `0` restores strict batch failure, and the
default `1` permits any partial success. Structural validation errors on
episodes selected for publication retain their existing failure behavior.

## CLI Tools

- `lerobot-align`: Main steerable annotation and alignment engine.
- `lerobot-align-fit`: Fits dataset-specific temporal boundary offsets and duration priors.
- `lerobot-align-eval`: Evaluates alignment quality (tIoU, boundary MAE, B@3) against ground truth.
- `lerobot-align-eval-batch`: Paired multi-episode batch evaluation.
- `lerobot-motion-viz`: Visualizes joint motion velocity curves and adaptive sample frames.

---

## Evaluation

The source release includes the metric library, evaluation harness and shared
[calibration protocol](evaluation/calibration_protocol.md). Start with the
[reproduction guide](evaluation/REPRODUCING.md) for installation, authorized
input preparation, configuration and the limits of historical reproduction.
Historical accuracy artifacts, private inputs and engineering notes are excluded
from the clean source export. Their absence is intentional; the software license
does not grant redistribution rights to the original datasets. The reviewed
matched-frame visual-token evidence is public under
[`evaluation/public_artifacts/`](evaluation/public_artifacts/README.md).

### What the numbers actually say

The following are archived study results, not measurements from the release
smoke tests. Their private predictions and fits are not shipped, so these
numbers cannot currently be regenerated from the source release alone.

Under the shared protocol, boundary calibration raises task-macro temporal
IoU by **+0.150 on Corpus A** and **+0.042 on Corpus B** for single-video
input, and by **+0.155 / +0.047** for two-view stacked video. The Corpus B
effects survive the registered 24-test Holm correction (`p=0.0048` single,
`p=0.0024` stacked). The Corpus A effects do **not** (`p=0.1504` and
`p=0.1116`): Corpus A has only four task clusters, so its intervals are wide
despite the larger point estimates.

Two model-free timing references — one placing boundaries from the seed
episodes' median duration ratios, one from their cumulative duration prior —
score at or above the best VLM arm on temporal IoU on **both** corpora (0.696
against 0.687 on A; 0.592 against 0.566 on B), and land within about a point of
it on B@3. Those references are given each episode's annotated extent and
always place every label, so they are timing-prior controls rather than
competing methods — but the headline gains have to be read against them.

The head-to-head against upstream `lerobot-annotate` was measured on one corpus
only: the corpus this package was developed against. That makes it a
tuned-versus-untuned comparison in this package's favour. Every result is
conditional on a single served model alias (Qwen3.8-27B). Neither reference
layer has an independent repeat-annotation agreement measurement, so no score
can be read as exceeding human agreement.

---

## Repository layout

| Path | Contents |
| --- | --- |
| `src/lerobot_align/` | The installable package: CLI, executor, VLM client, frame sampling and video encoding, annotation modules, prompts, writer, and the `diagnostics/` tools behind the four extra console scripts. |
| `tests/` | Unit tests, plus `run_e2e_smoke.py`, the end-to-end smoke run CI executes. |
| `scripts/` | Reference host scripts for serving models and driving batch runs. Not part of the installable package — see [scripts/README.md](scripts/README.md). |
| `evaluation/` | Metric library, harness and protocol — see [evaluation/REPRODUCING.md](evaluation/REPRODUCING.md). Historical artifacts remain outside the release export. |

### Research documents

- [ANNOTATION_PIPELINE.md](./ANNOTATION_PIPELINE.md): Pipeline architecture and configuration reference.

### Project files

- [CONTRIBUTING.md](./CONTRIBUTING.md): dev install, tests, lint, and what not to regenerate.
- [CHANGELOG.md](./CHANGELOG.md): release notes.
- [CITATION.cff](./CITATION.cff): how to cite this software.

---

## License

Apache 2.0. See [LICENSE](LICENSE) for details.
