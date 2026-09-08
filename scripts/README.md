# `scripts/`

Reference host scripts. **None of this directory is part of the installable
package** — nothing under `src/lerobot_align/` imports or shells out to it, and
`uv build` does not ship it. These files are tuned to the 8xH100 host the
evaluation study ran on, and are published so the study's serving and batch
configuration is inspectable rather than described second-hand. Read them
before running them; adapt paths, GPU counts and virtualenv locations to your
own machine.

CI checks only that they parse (`bash -n scripts/*.sh`).

## Inventory

### `start_qwen38.sh` — multi-replica vLLM launcher

Starts one independent vLLM replica per GPU inside a `tmux` session, serving
`Qwen/Qwen3.8-27B-FP8` under the served name `Qwen/Qwen3.8-27B`. Ports are
assigned in replica order from `8000`. Replicas bind `127.0.0.1`.

This is the only script here with a machine-readable consumer: the evaluation
harness invokes it at `evaluation/scripts/run_all.sh:329` for its `serve`
stage.

Prerequisites: `tmux`, `nvidia-smi`, `curl`, `python` and `awk` on `PATH`, plus
a **separate vLLM virtualenv** (`.venv-vllm` at the repository root by default,
overridable with `QWEN38_VENV`). The script prints the `uv venv` / `uv pip
install vllm` commands to create it if it is missing. It lifts vLLM's 32-frame
video cap with `--media-io-kwargs '{"video": {"num_frames": -1}}'`, which the
native-video annotation modes require.

### `start_qwen38_flashnext.sh` — experimental Docker variant

Serves `Qwen/Qwen3.8-Flash-Next-FP8` from the `vllm/vllm-openai:qwen38-flash-next`
container with expert parallelism and PLE CPU offload. It exists because that
architecture is not in the pip vLLM registry, so it cannot be a flag on
`start_qwen38.sh`.

**Nothing in the evaluation study uses it.** No result in `evaluation/` was
produced with this model or this script.

Prerequisites: a working `docker` the invoking user can reach, the container
image pulled locally, four or more 80 GiB GPUs, and roughly 51 GB of host RAM
for the offloaded embedding table.

### `run_generate_batch.sh` — open-ended generation over a list of datasets

For each repository ID in a text file, stages a working copy, runs
`lerobot-align` in generate mode (describe → segment, plus plan, memory and
task augmentation), and pushes the annotated result to a destination under
`$DST_OWNER`. Keeps per-dataset logs and `.done` markers so an interrupted
batch resumes.

### `run_import_batch.sh` — import existing human spans over a list of datasets

Same shape, but with `--plan.subtask_import=lerobot_annotations` instead of
generation: it imports the human subtask spans already in the dataset and
generates only the plan, memory and task augmentations around them.

Both batch drivers are run by hand — no scheduler, no CI step, no other script
calls them. Each invokes the `lerobot-align` console script directly from
`$ANNOTATE_VENV/bin` (default `.venv/bin`), so the package must already be
installed in that virtualenv. Both require `DST_OWNER`, both default to
`VLM_MODEL=Qwen/Qwen3.8-27B` at `VLM_API_BASE=http://127.0.0.1:8000/v1`, and
both verify that the endpoint actually serves that model before starting any
work.

## Known rough edges

These are real and unfixed. They are listed rather than hidden.

- **`start_qwen38.sh` hardcodes `CUDA_HOME=/usr/local/cuda-12.8`** (line 46)
  and prepends it to `PATH`. There is no environment override. On a host with
  CUDA installed anywhere else, edit the line.

- **The multi-replica path assumes an nginx load balancer on `:9000` whose
  configuration is not shipped.** With more than one replica, the script's
  closing instructions point clients at `VLM_PORT=9000` and state that the
  balancer needs upstreams covering the replica ports. That config file does
  not exist in this repository, and neither does anything that generates it.
  Until you write one, either run one annotation job per replica port or route
  the ports yourself.

- **The Flash-Next container exposes its endpoint only on host loopback**
  (`docker run -p 127.0.0.1:${PORT}:${PORT}`). Its internal `0.0.0.0` bind
  is required for Docker forwarding. Both launchers are unauthenticated local
  servers; use an authenticating proxy if intentionally exposing either one.

- **`run_generate_batch.sh` defaults `CAMERA` to `observation.images.wrist`**,
  a camera name that is specific to the datasets it was written for. Set
  `CAMERA` for any other dataset; a wrist close-up frequently cannot see the
  event the annotation is about.

- **Several defaults name a specific host layout** — `$HOME/.cache` paths,
  GPU counts, `.venv-vllm`, an 8-GPU tray. They are documented in each
  script's header and overridable by environment variable, but they are
  defaults chosen for one machine.
