# Annotation Pipeline

`lerobot-align` watches each episode's video with a vision-language
model (VLM) and writes natural-language annotations back into your
dataset. It fills the two language columns from the
LeRobot language-column recipe conventions —
`language_persistent` and `language_events` — straight into
`data/chunk-*/file-*.parquet`.

In short: point it at a LeRobot dataset, and it adds subtasks, plans,
memory, interjections, speech, and visual Q&A that a policy can be
trained on.

## How it fits together

```text
  your dataset                   lerobot-align
  (LeRobot v3.x)
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │                    read episodes                     │
  └──────────────────────────┬──────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                     ▼
  ┌──────────┐      ┌───────────────┐        ┌──────────┐       one shared Qwen-VL
  │   plan   │      │ interjections │        │   vqa    │  ◀──   server (vLLM, OpenAI
  └────┬─────┘      └───────┬───────┘        └────┬─────┘        API) drives all three
       └────────────────────┼─────────────────────┘
                            │   each module stages raw JSONL
                            ▼   into .annotate_staging/
                  ┌─────────────────┐
                  │    validator    │  ◀──  checks everything
                  └────────┬────────┘
                           ▼
                  ┌─────────────────┐
                  │     writer      │
                  └────────┬────────┘
                           ▼
              data/chunk-*/file-*.parquet
              (+ meta/info.json tools)
```

Three modules (`plan`, `interjections`, `vqa`) all talk to **one** shared
VLM. Each module stages its output to disk, a validator checks it, and a
single writer rewrites the dataset shards in place.

## What the pipeline produces

Each module emits a few kinds of annotation ("styles"), routed to one of
the two language columns:

| Style / atom                                | Column                | Module          |
| ------------------------------------------- | --------------------- | --------------- |
| `subtask` (Pi0.7-style "how, not what")     | `language_persistent` | `plan`          |
| `plan` (remaining subtasks at each boundary) | `language_persistent` | `plan`          |
| `memory` (MEM-style compression)            | `language_persistent` | `plan`          |
| `task_aug` (rephrasings of the task)        | `language_persistent` | `plan`          |
| `interjection`                              | `language_events`     | `interjections` |
| speech tool-call atom (`style=null`, `say`) | `language_events`     | `interjections` |
| `vqa` (user / assistant pair)               | `language_events`     | `vqa`           |

### How subtasks are generated

The `plan` module doesn't ask the VLM for subtasks in one shot. Instead
it uses a two-step **describe → segment** flow:

1. **Describe** — the VLM narrates only what it actually sees in the
   chosen camera (no guessing about the task).
2. **Segment** — that description is fed back in, and the VLM splits the
   episode into consecutive atomic subtasks.

By default, both passes see the episode as **timestamped contact sheets** —
frames sampled at `frames_per_second` (2 fps by default) and packed into JPEG
grids with each frame's time burned into its corner, so the VLM cites exact
boundary times directly. This is far cheaper in vision tokens than one image
per frame, so the sampling can stay dense.

Set `--plan.subtask_generate_frame_format=video` to send those uniformly
sampled frames as a native H.264 clip instead. This applies to video-based task
derivation and the describe→segment passes when subtasks are being generated;
it does not change imported span data, the fixed-label alignment modality, or
the optional seeded-relabel pass. The native clip starts at 0.0 seconds for
each episode or window, allowing a video-native VLM to use its temporal tokens
rather than OCR timestamp badges. Contact sheets remain the default for
compatibility. Task derivation uses one whole-episode sample capped to
`max_frames_per_prompt`; the describe→segment path is the part that windows
long episodes.

In either format, episodes longer than `max_frames_per_prompt` are split into
independent windows at the same density, then their spans are combined and
stitched into a gap-free episode. If a greedy split would create a tiny tail,
the final two windows split their remaining sample intervals approximately
evenly; at 2 fps with a 100-frame budget, a 51.18-second episode therefore uses
52 + 52 frames rather than 100 + 4. An event that crosses a window seam can
still be split or repeated; increase `max_frames_per_prompt` when the model
context allows it if seam-sensitive boundaries matter. Both prompts also carry a
causal **event-boundary** definition (a new event starts when an object becomes
held / is released / reaches a new location / a lid changes state / contents
move) to sharpen where cuts land.

```bash
LEROBOT_OPENAI_SEND_MM_KWARGS=1 uv run --no-sync lerobot-align --root=/path/to/dataset \
    --plan.subtask_generate_frame_format=video \
    --plan.subtask_realign_generated=true \
    --plan.subtask_align_frame_format=video \
    --plan.subtask_video_fallback=error \
    --vlm.auto_serve=false
```

Native input requires a video-capable endpoint. For an external vLLM server,
the environment variable above forwards the clip's encoded FPS; also start the
server with `--media-io-kwargs '{"video":{"num_frames":-1}}'`. Without both
controls, vLLM may silently resample the clip or cap it (commonly at 32
frames). A vLLM process spawned by the tool's default command enables both
controls automatically.

The OpenAI-compatible transport inlines each MP4 as base64. For unusually
large frame budgets, lower `max_frames_per_prompt` or
`executor.episode_parallelism` if the client runs out of memory or a proxy
rejects the request as too large.

Native generation treats visual evidence as required. If the frame provider
cannot initialize or decode the complete requested frame grid, the episode
fails before any VLM request and reports the root exception. Failure of the
clip encoder alone may fall back to contact sheets because the decoded frames
are still available, but an empty visual fallback is rejected rather than
sent as a text-only prompt.

Set `--plan.subtask_video_fallback=error` to make native video a hard contract
for any generation or alignment stage configured as `video`. If encoding or
`video_url` construction fails, the tool then aborts before sending an
alternate contact-sheet request. The default `contact_sheet` policy retains
the compatibility fallback. Explicit contact-sheet stages—including seeded
relabeling—are unaffected by this policy.

Optionally, a third **seeded-relabel** pass (`--plan.subtask_seeded_relabel`)
revisits each span with its previous/current/next segment contact sheets and
minimally corrects the label, using the first label as a prior — it keeps the
boundaries fixed and only sharpens wording, at the cost of one extra call per
subtask.

Set `--plan.subtask_realign_generated=true` to feed the resulting ordered label
texts directly into the fixed-label aligner. No intermediate `subtasks.json` is
written or required. The chained path is:

1. **Describe → segment** — discover labels and rough spans, using generation
   windows when the full-density sample exceeds `max_frames_per_prompt`.
2. **Optional seeded relabel** — revise the label wording while keeping the
   rough spans fixed.
3. **Whole-episode realignment** — discard the rough boundaries and time the
   final ordered labels independently in one call, using the
   `subtask_align_*` settings.

This chained realignment is all-labels-or-error. After parsing, calibration,
frame snapping, cleaning, and stitching, the final labels must still match the
generated ordered list exactly; `subtask_align_min_fraction` cannot permit a
partial generated-label result.

Generation windowing and alignment sampling are intentionally different. A
long episode is split into independent generation windows at full sampling
density, so labels can still split or repeat at a seam. Realignment is never
windowed; it subsamples the complete episode to fit `max_frames_per_prompt` and
therefore has global context for the final boundaries. Choose its visual input
separately with `--plan.subtask_align_frame_format`: setting it to `video`, as
in the example above, sends the generated labels with one native video clip.

`subtask_align_calibration_path`, camera selection, and the other alignment
controls apply to this final pass. Calibration requires an exact match with
its recorded ordered label list. Because open-ended generation can vary wording
or step count between episodes, a calibration will often be skipped unless the
generator produces a canonical, stable label schema. A legacy bare-offset
calibration has no verifiable schema, so generated-label realignment ignores it
with a warning; direct fixed-label alignment retains that legacy behavior.
The option affects only labels created by open-ended generation: labels from
`subtasks_path` already take the fixed-label alignment path, while imported
spans already include their authored boundaries and bypass realignment.

The resulting spans are then stitched into a gap-free, full-episode
cover, so **every frame has exactly one active subtask**. See
[Running on Hugging Face Jobs](#running-on-hugging-face-jobs) for the
production settings (single camera, timestamped contact sheets,
auto-windowed subtask generation).

### Importing subtasks the dataset already records

Before generating anything, check whether your dataset already carries
subtasks. `--plan.subtask_import` reads them straight out of the dataset —
labels *and* boundaries — so subtask generation costs **no VLM call at all**:

```bash
uv run --no-sync lerobot-align --root=/path/to/dataset --plan.subtask_import=auto
```

Four sources are supported, tried in this order under `auto`:

| Source                | Where it lives                                                                             | Written by                                                    |
| --------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------- |
| `sarm`                | `{dense,sparse}_subtask_names` / `_start_times` / `_end_times` on `meta/episodes/*.parquet` | `lerobot.data_processing.sarm_annotations.subtask_annotation` |
| `lerobot_annotations` | interval spans or timestamped atoms in `meta/lerobot_annotations.json`                     | LeRobot Annotate or another compatible annotation tool        |
| `language`            | `style="subtask"` rows in `language_persistent`                                            | a previous `lerobot-align` run                                |
| `task_index`          | runs of the per-frame `task_index` column                                                   | recording with the task changing mid-episode                  |

Pass one of those names instead of `auto` to pin a single source. The
difference matters when nothing is found:

- **`auto`** falls back to generating with the VLM, and warns which sources it
  tried.
- **A named source** does not fall back. If it finds nothing, that episode gets
  no subtasks (and so no plan or memory) and the run warns — naming a source
  means "this or nothing", not "this, or quietly spend a GPU on something
  else".

`--plan.subtask_import_sarm_prefix` (default `dense`) picks which SARM
granularity is preferred; the other prefix and the legacy unprefixed columns
are accepted as fallbacks. The raw `*_times` float columns are read directly,
so sub-second boundaries survive.

`lerobot_annotations` accepts both interval entries with `label`, `start`, and
`end`, and timestamped `style="subtask"` atoms with `content` and `timestamp`
(the schema-v2 layout, detected from the episode fields rather than the
top-level version). When both forms occur in one episode, non-empty interval
entries take precedence as the explicit authored segmentation; an empty list
can fall through to atoms.

Imported spans go through the same gap-closing stitch as generated ones, so
the "every frame has exactly one active subtask" contract still holds even if
the recorded annotation deliberately left idle gaps. `task_index` ignores
single-task episodes, where the "segmentation" would just restate the episode
task.

<Tip>

`subtask_import` and `subtasks_path` are mutually exclusive and the run fails
if both are set. They solve opposite problems: import is for **labels and
times already on disk**; align (below) is for **labels without times**.

</Tip>

### Supplying your own subtasks

If you already know the subtasks — a scripted collection protocol, a
human-authored list, or subtask labels a previous run wrote — you can hand
them to the pipeline and have the VLM only decide **when** each one happens:

```bash
uv run --no-sync lerobot-align --root=/path/to/dataset \
    --plan.subtasks_path=subtasks.json
```

The file is either a flat list applied to every episode:

```json
["pick up the cup", "pour the water into the glass", "put the cup down"]
```

or an object keyed by episode index, with an optional `default`:

```json
{
  "default": ["open the drawer", "take out the fork", "close the drawer"],
  "7": ["open the drawer", "close the drawer"]
}
```

This mode shows the VLM the whole episode together with the ordered list, using
timestamped contact sheets by default or a native clip when
`--plan.subtask_align_frame_format=video`, and asks for a `start`/`end` per
label in a **single call**. Two things fall out of that shape:

- A subtask the episode never performs comes back with `null` boundaries and is
  dropped, rather than being forced into the timeline to satisfy the list.
- Starts are forced strictly increasing — the supplied list is the authority on
  ordering, so a label placed out of order is dropped rather than reordered.

`max_frames_per_prompt` caps the total camera frames for that call. Alignment
is deliberately never split across calls: a chunk of tiles cannot tell which
part of the episode it covers, and the model restarts its numbering inside each
one. Sub-sampling costs little here, because the reply is one span per label
rather than one decision per tile.

Alignment can combine synchronized views by naming their exact dataset keys:

```bash
uv run --no-sync lerobot-align --root=/path/to/dataset \
    --plan.subtasks_path=subtasks.json \
    --plan.subtask_align_camera_keys='["observation.images.wrist", "observation.images.side"]' \
    --plan.max_frames_per_prompt=300
```

Camera roles are not standardized in LeRobot metadata, so the pipeline does
not guess which key is a wrist or side camera. A missing optional key is warned
about and skipped; if none remain, alignment falls back to the provider's
default camera. In contact-sheet mode, the views are labeled and stacked inside
each timestamp tile, and the frame budget is shared across them: two cameras
and a budget of 300 mean at most 150 synchronized timestamps per camera. Native
video stacks the synchronized views into each video frame instead, so it keeps
one chronological sequence and does not divide the timestamp budget by the
number of views.

`--plan.subtask_align_sampling=motion_stratified` is an experimental
alternative to uniform timestamps. It discovers low-dimensional floating-point
`observation.*` vectors without assuming joint names or robot morphology,
robustly scores motion onsets and offsets, and chooses one real frame within
each equal-time stratum. Temporal coverage and episode endpoints are preserved.
Missing, low-coverage, constant, noise-dominated, malformed, or unreadable
state produces the exact uniform grid instead. Uniform remains the default.

`subtask_describe_first` and `subtask_seeded_relabel` don't apply here and are
skipped — the labels are the input, so there is nothing to ground or refine.
Everything downstream is unchanged: plan, memory, and interjection anchoring
all read the resulting spans exactly as they do for generated ones.

<Tip warning={true}>

The labels are taken as ground truth, so an unaligned subtask is invisible in
the output — the stitch closes the gap it leaves, which means a mostly failed
alignment still covers the episode and still passes validation. Because that
is wrong output that looks right, an episode that places fewer than
`--plan.subtask_align_min_fraction` (default `0.5`) of the supplied subtasks
**fails the run** rather than emitting spans. Set it to `0` to stitch over the
gaps and only warn.

Either way, watch for `aligned N/M given subtask(s)` warnings: they mean the
supplied list didn't match what those episodes actually show. This matters
most when one task-level list is applied to every episode, since real
demonstrations vary (retries, skipped steps, early termination). If whole
subtasks are never placed, check that the camera actually shows them —
`--vlm.camera_key` defaults to the first `observation.images.*` key, which may
be a wrist close-up that cannot see, say, a door opening.

</Tip>

### Tools

The writer does **not** add a `tools` column to the parquet. The tool
catalog lives in `meta/info.json["tools"]` instead.
After every run, the pipeline makes sure the canonical `say` schema is in
that list, keeping any tools you declared beforehand.

Want to add your own tool? Edit `meta/info.json["tools"]` directly — the
pipeline preserves whatever is already there. That makes the tool visible
to the chat template, so the model can learn to _generate_ the call. The
runtime layer that actually _executes_ a generated call belongs to the
application consuming the annotated dataset; this package only records the
schema and calls.

## Running on Hugging Face Jobs

Annotating a real dataset needs a GPU big enough to serve the VLM, so
`lerobot-align` can dispatch itself to
[Hugging Face Jobs](https://huggingface.co/docs/hub/en/jobs) — same as
`lerobot-train`. Add `--job.target=<flavor>` to the exact command you'd
run locally and it runs on that hardware instead:

Before submitting, publish an installable wheel to a location the pod can read,
record its SHA256, and set `ALIGN_PACKAGE_SPEC` to that HTTPS wheel URL with a
`#sha256=` fragment. A `git+https` requirement pinned to a full 40-character
commit is also accepted. Mutable branches, local wheels and the unpublished
PyPI version are rejected before provisioning.

```bash
uv run --no-sync hf auth login
: "${ALIGN_PACKAGE_SPEC:?Set an immutable remote wheel URL or VCS requirement}"
: "${SOURCE_DATASET:?Set the source Hub dataset ID}"
: "${DESTINATION_DATASET:?Set a separate destination Hub dataset ID}"
uv run --no-sync lerobot-align \
    --repo_id="$SOURCE_DATASET" \
    --new_repo_id="$DESTINATION_DATASET" \
    --push_to_hub=true --push_private=true \
    --vlm.model_id=Qwen/Qwen2.5-VL-7B-Instruct \
    --vlm.num_gpus=1 \
    --vlm.serve_ready_timeout_s=1800 \
    --job.package_spec="$ALIGN_PACKAGE_SPEC" \
    --job.target=h200
```

Local `plan.subtasks_path`, `plan.subtask_align_calibration_path`, `staging_dir`
and configuration-file arguments are rejected: the pod cannot read this
machine's files. Use local execution for those workflows, or prepare source
annotations in the Hub dataset and select an explicit `plan.subtask_import`
source. The default server command includes native-video processing settings.
Jobs require paid compute and were not executed by the CPU release checks.


That submits a single-GPU `h200` job that:

1. starts from the configured image and installs `lerobot-align` plus its
   compatible LeRobot dependency,
2. boots one vLLM server per GPU and drives it over the OpenAI-compatible API,
3. runs the `plan` / `interjections` / `vqa` modules across the dataset,
4. with `--push_to_hub=true`, uploads the result to `--new_repo_id` (or
   back to `--repo_id` in place if you leave that unset).

The command streams the job's logs; `Ctrl-C` detaches without cancelling
it. List the available flavors and their pricing with `hf jobs hardware`.

Step 2 is why the example passes no `--vlm.auto_serve`. Locally that flag
defaults to `false`, because assuming a bare `lerobot-align` run should spawn a
27B server is wrong on a machine that may have no GPU. A pod is the opposite
case: it is a fresh GPU machine with nothing listening on `--vlm.api_base`, and
nothing else will start a server there. The submitter therefore sends
`--vlm.auto_serve=true` to the pod automatically. Pass the flag yourself only to
override that: `--vlm.auto_serve=false` points the job at an endpoint that is
already running somewhere else, in which case `--vlm.serve_command` and
`--vlm.num_gpus` have no effect.

<Tip warning={true}>

Qwen3.6 ships with thinking enabled, which eats the token budget the
annotator needs for its JSON answer — `--vlm.chat_template_kwargs='{"enable_thinking": false}'`
turns it off. Without `--push_to_hub=true` the annotated dataset is
discarded when the pod exits.

</Tip>

### Job options

| Flag                | Default                   | What it does                                                                    |
| ------------------- | ------------------------- | ------------------------------------------------------------------------------- |
| `--job.target`      | `local`                   | HF Jobs flavor to run on (e.g. `h200`, `h200x4`). Omitted/`local` runs here.    |
| `--job.image`       | vLLM 0.19.1 image digest from `AnnotationJobConfig` | Runtime image for the pod. |
| `--job.timeout`     | `2h`                      | Wall-clock cap. Raise it for large datasets.                                    |
| `--job.detach`      | `false`                   | Submit and exit instead of streaming logs.                                      |
| `--job.lerobot_ref` | `7e241bd630a3719a56157a497ce5d08f244784f1` | Immutable LeRobot v0.6.1 commit installed on the pod. |
| `--job.package_spec` | required for remote Jobs | Immutable HTTPS wheel URL with SHA-256 or VCS requirement with a full commit SHA. Validated before provisioning. |
| `--job.tags`        | `[]`                      | Extra tags on the job and on any dataset it pushes (`lerobot` is always added). |

The default image is pinned by digest to vLLM 0.19.1 because its CUDA 12.9 / PyTorch
2.10 stack includes Transformers 5.5.x, which is compatible with the Hub 1.x
API required by LeRobot 0.6.1. Older vLLM 0.17 images pin Transformers 4.x and
cannot share that dependency set.

The package carries `jobs-constraints.txt`, the 259-distribution Python
constraints recorded from the qualified image. Setup uses `python3` (the image
has no `python` alias), completes its missing Cairo dependency, and runs strict
`pip check` and import checks. Overrides of the image or LeRobot revision need
fresh compatibility qualification; do not delete constraints to force a run.

For a bigger dataset, scale to `h200x4` and raise
`--vlm.parallel_servers` / `--vlm.num_gpus` to match, and give the job
more headroom with e.g. `--job.timeout=8h`.

Remote runs need `--repo_id` (the pod pulls the dataset from the Hub;
`--root` names a directory only your machine has). A dataset that exists
only in your local cache is rejected before provisioning. Publish it separately
through a reviewed explicit file manifest; Jobs never implicitly upload a local
cache. The resolved video metadata policy and multimodal-kwargs opt-in are
forwarded to the pod, while endpoint credentials remain in Jobs secrets.

The annotator accepts LeRobot v3.x datasets in the Parquet/MP4 layout. It
rejects v2.1 and Lance-backed datasets during preflight; migrate those inputs
with LeRobot before starting an annotation run.

## Key options

These are the flags you'll reach for most often. Run
`lerobot-align --help` for everything else; the defaults are tuned for
short manipulation episodes.

### Dataset in / out

| Flag              | Default | What it does                                                            |
| ----------------- | ------- | ----------------------------------------------------------------------- |
| `--repo_id`       | —       | Hub dataset to annotate (downloaded if `--root` unset).                 |
| `--root`          | —       | Annotate a local dataset directory instead.                             |
| `--new_repo_id`   | —       | Push the result to a new repo (leaves the source repo untouched).       |
| `--push_to_hub`   | `false` | Upload after annotating (to `--new_repo_id`, else back to `--repo_id`). |
| `--allow_version_tag_move` | `false` | Explicitly authorize a verified backup-and-rollback update of an existing LeRobot version tag. |
| `--only_episodes` | all     | Select episodes on a canonical rerun; a fresh dataset must run all episodes once. |
| `--seed`          | `1729`  | Seeds the RNGs that pick interjection timestamps + VQA question types.  |

The tag-move opt-in is needed when deliberately republishing to a destination
that already has its `v3.x` release tag. Without it, the destination is checked
before model work starts and the run fails without uploading. The old target is
preserved under a verified recovery tag until the new tag has been verified;
failed moves are rolled back without overwriting a concurrent publisher.

### Which modules run

Every module is on by default and can be toggled independently (set to
`false` to skip it, e.g. to iterate on one module at a time):

| Flag                      | Default | Turns off                           |
| ------------------------- | ------- | ----------------------------------- |
| `--plan.enabled`          | `true`  | subtasks + plan + memory + task_aug |
| `--interjections.enabled` | `true`  | interjections + speech atoms        |
| `--vqa.enabled`           | `true`  | the VQA pairs                       |

### The VLM (`--vlm.*`)

| Flag                       | Default            | What it does                                                                         |
| -------------------------- | ------------------ | ------------------------------------------------------------------------------------ |
| `--vlm.model_id`           | `Qwen/Qwen2.5-VL-7B-Instruct` | The model to serve and prompt.                                                       |
| `--vlm.camera_key`         | first `images.*`   | Which camera every prompt is grounded on.                                            |
| `--vlm.serve_command`      | auto               | The exact `vllm serve …` command (set TP size, GPU memory, `--max-model-len` here).  |
| `--vlm.parallel_servers`   | `1`                | Independent servers for round-robin routing (one per GPU).                           |
| `--vlm.num_gpus`           | `0`                | GPUs per server (`0` = one each).                                                    |
| `--vlm.client_concurrency` | `16`               | In-flight requests across all servers.                                               |
| `--vlm.max_new_tokens`     | `512`              | Generation cap per call.                                                             |
| `--vlm.temperature`        | `0.2`              | Sampling temperature.                                                                |
| `--vlm.reasoning_effort`   | `null`             | Thinking-budget hint (`low`/`medium`/`high`) forwarded to OpenAI-compatible servers. |

### Subtasks / plan / memory (`--plan.*`)

| Flag                            | Default    | What it does                                                                                                                 |
| ------------------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------- |
| `--plan.frames_per_second`      | `2.0`      | Requested sampling rate for generation and alignment visuals (`2.0` = one frame every 0.5s); alignment may subsample to its frame budget. |
| `--plan.max_frames_per_prompt`  | `60`       | Frame budget per VLM call. Describe→segment generation is auto-windowed at full density; alignment instead subsamples the whole episode into one call. |
| `--plan.subtask_generate_frame_format` | `contact_sheet` | Visual input for open-ended task derivation and describe→segment generation: `contact_sheet` or native `video`.       |
| `--plan.subtask_video_fallback` | `contact_sheet` | What a stage configured as native `video` does if it cannot build a usable `video_url`: warn and use `contact_sheet`, or fail with `error`. |
| `--plan.contact_sheet_columns`  | `5`        | Columns per contact-sheet grid (`contact_sheet_frames_per_sheet` tiles, time row-major).                                     |
| `--plan.plan_max_steps`         | `8`        | Upper bound on subtasks per generation call (there is one call per window for long episodes).                              |
| `--plan.subtask_describe_first` | `true`     | Run the describe→segment grounding pass (best subtask quality; +1 call/episode).                                             |
| `--plan.subtask_seeded_relabel` | `false`    | Second pass: re-label each subtask from its prev/current/next contact sheets, seeded with the first label (+1 call/subtask; any missing label aborts). |
| `--plan.subtask_relabel_frames` | `5`        | Frames sampled uniformly per segment sheet in the relabel pass (only used when `subtask_seeded_relabel=true`).               |
| `--plan.subtask_realign_generated` | `false` | Discard generated boundaries and time the final generated labels with one whole-episode fixed-label alignment call. Uses the `subtask_align_*` settings. |
| `--plan.emit_plan`              | `true`     | Emit the numbered `plan` rows (`false` = subtasks + memory only).                                                            |
| `--plan.emit_memory`            | `true`     | Emit the `memory` rows (`false` = subtasks + plan only); symmetric to `emit_plan`.                                           |
| `--plan.n_task_rephrasings`     | `10`       | Exact number of distinct `task_aug` alternatives to emit in addition to the canonical task (`0` disables).                 |
| `--plan.derive_task_from_video` | `if_short` | Use the dataset task as-is (`off`), only when it's missing/short (`if_short`), or always re-derive from video (`always`).    |
| `--plan.subtasks_path`          | unset      | JSON file of subtask labels to align instead of generate (see [Supplying your own subtasks](#supplying-your-own-subtasks)). |
| `--plan.subtask_align_frame_format` | `contact_sheet` | Visual input for fixed-label alignment, including generated-label realignment: `contact_sheet` or native `video`.      |
| `--plan.subtask_align_camera_keys` | `[]`     | Ordered exact cameras for synchronized alignment; contact sheets divide the budget across views, while native video stacks views per timestamp. |
| `--plan.subtask_align_sampling` | `uniform`  | Alignment timestamp selector: `uniform` or experimental `motion_stratified` with automatic uniform fallback.             |
| `--plan.subtask_align_motion_feature_keys` | `[]` | Optional exact numeric feature keys for motion sampling; empty discovers suitable `observation.*` vectors.               |
| `--plan.subtask_align_calibration_path` | unset | Calibration JSON for fixed-label boundaries; applies only when the episode's ordered labels exactly match its fitted labels. |
| `--plan.subtask_align_min_fraction` | `0.5`  | Fail the run when an episode places fewer than this fraction of the supplied subtasks (`0` = stitch and warn instead).      |
| `--plan.subtask_import`         | `off`      | Read recorded subtasks: `auto`, `sarm`, `lerobot_annotations`, `language`, or `task_index`.                                 |
| `--plan.subtask_import_sarm_prefix` | `dense` | Preferred SARM granularity (`dense`/`sparse`); the other prefix and legacy unprefixed columns are fallbacks.                |

### Interjections + VQA

| Flag                                            | Default | What it does                                               |
| ----------------------------------------------- | ------- | ---------------------------------------------------------- |
| `--interjections.max_interjections_per_episode` | `3`     | Cap on interjection/speech pairs per episode.              |
| `--interjections.min_speech_atoms_per_episode`  | `1`     | Fail if an enabled episode emits fewer speech atoms (`0` = best effort). |
| `--vqa.vqa_emission_hz`                         | `1.0`   | How often VQA pairs are emitted.                           |
| `--vqa.min_pairs_per_episode`                   | `1`     | Minimum complete VQA pairs when cameras are available (`0` = best effort). |
| `--vqa.restrict_to_default_camera`              | `false` | Ground VQA only on `--vlm.camera_key` (else every camera). |
| `--executor.episode_parallelism`                | `16`    | Episodes processed concurrently within each phase.         |

## Contributing new modules

The pipeline is built to grow, and **contributions are very welcome** —
a brand-new module (say, trajectory traces or affordances), a new prompt
template, a smarter grounding flow, or quality fixes to the existing
`plan` / `interjections` / `vqa` modules.

Every module lives under
`src/lerobot_align/modules/`, shares the VLM
client and the keyframe cache, writes its raw output to the staging
tree, and plugs into the executor as its own phase. Got an idea? Open an
issue or pull request in this repository.

## How recipes consume the output

The annotations are meant to be read by LeRobot language-column recipes.
Typically:

- low-level / high-level / memory-update branches read
  `subtask` / `plan` / `memory` from `language_persistent`.
- an interjection-response branch reads `interjection` events plus the
  paired speech atom (merged into one assistant turn via `tool_calls_from`)
  and the matching deterministic `plan` state at the same timestamp.
- a VQA branch reads the `(vqa, user)` and `(vqa, assistant)` pairs from
  `language_events`.

## Why state and events are split

Two ideas shape the design:

1. **Persistent state vs. exact events.** Persistent rows (`subtask`,
   `plan`, `memory`) apply to the whole episode and answer "what's true
   right now?". Event rows (`interjection`, `vqa`, speech) appear only on
   the one frame whose timestamp matches. Timestamps are copied straight
   from the source parquet — never recomputed in floating point.
2. **One VLM pass.** All three modules share a single VLM client (the
   OpenAI-compatible client talking to the job's vLLM server), so you pay
   for one model load per dataset, not three.

## Re-running a single module

Each module stages its raw output to
`<root>/.annotate_staging/episode_{N:06d}/<module>.jsonl`. This makes
prompt iteration cheap: re-running one module overwrites only its own
JSONL, then the writer recomposes the final parquet. Disable modules you
don't want with `--plan.enabled=false` (and likewise
`--interjections.enabled` / `--vqa.enabled`) to test one at a time.

## What the validator checks

Before the writer runs, `StagingValidator` confirms:

- every event row lands exactly on a real frame timestamp;
- no speech / interjection pairs are left orphaned;
- deterministic `plan` state exists at every interjection timestamp;
- `memory` rows fall on subtask boundaries (a warning, not an error);
- each VQA assistant `content` is valid JSON in one of the
  bbox / keypoint / count / attribute / spatial shapes;
- every row goes to the column chosen by `column_for_style(style)`.

It also enforces non-empty output from enabled modules: subtasks and requested
plan/memory rows must cover their deterministic boundaries; free-form task
augmentation must contain the canonical task plus exactly the requested number
of alternatives (structured axes require the canonical task and all requested
synonyms, while conditional omission axes may be sparse); and speech/VQA use
the configurable per-episode minima above. Completeness failures always abort
before the writer; `--skip_validation=true` bypasses only the remaining
structural checks while debugging.

## Where each module's ideas come from

- **`plan` — subtasks.** Hi Robot ([Shi 2025](https://arxiv.org/abs/2502.19417))
  for atom granularity ("pick up one piece of lettuce", "place bowl to
  box"); Pi0.7 ([Physical Intelligence 2025](https://pi.website/pi07))
  for "how, not what" detail.
- **`plan` — memory.** MEM ([Torne 2026](https://arxiv.org/abs/2603.03596)):
  keep only the minimal relevant information — preserve outcomes, drop
  specific attributes.
- **`interjections`.** Hi Robot's scenario taxonomy: negative task,
  situated correction, specific constraint, preference. Speech is a
  tool-call-only atom
  (`tool_calls=[{type:function, function:{name:"say", arguments:{text:...}}}]`).
- **`vqa`.** ECoT ([Zawalski 2024](https://arxiv.org/abs/2407.08693)) for
  grounded features (pixel bounding boxes `[x_min, y_min, x_max, y_max]`,
  keypoints) and Steerable VLA Policies
  ([Zhao 2025](https://arxiv.org/abs/2509.07626)) for multi-abstraction
  grounding. Pi0.7 also grounds answers across abstraction levels.

When improving a module, tweak its prompt template in
`src/lerobot_align/prompts/` rather than
rewriting from scratch.

## Roughly how much it costs

Per episode, the pipeline makes about `max_steps` plan calls,
`max_interjections_per_episode` interjection calls, and
`vqa_emission_hz × episode_seconds` VQA calls. With the defaults (8
subtasks, 1 interjection, 1 Hz × 3 pairs) on a 30-second episode, that's
~50 VLM calls.

Storage stays small: `language_persistent` is at most tens of KB per
episode (parquet dictionary-encodes the one entry that repeats across
frames), and `language_events` is empty on most frames — its size scales
with the number of emissions, not `num_frames × num_emissions`.
