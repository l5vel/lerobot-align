#!/usr/bin/env bash
set -euo pipefail
umask 077

# ============================================================
# Persistent multi-GPU vLLM launcher for Qwen3.8-27B
#
# Qwen3.8-27B is a dense 27B vision-language model on the Qwen3.5
# architecture, so unlike the 397B MoE it fits on a single H100: about 29 GiB
# for the official FP8 checkpoint, about 52 GiB in BF16. The default layout is
# therefore one independent replica per GPU rather than one replica sharded
# over all of them:
#
#   replicas = number_of_gpus / (tensor_parallel_size * pipeline_parallel_size)
#
# Usage:
#   ./start_qwen38.sh                    # GPUs 0-7, TP 1 -> 8 replicas
#   ./start_qwen38.sh 4                  # GPUs 0-3, TP 1 -> 4 replicas
#   ./start_qwen38.sh --gpus 3           # One replica on GPU 3
#   ./start_qwen38.sh 8 --tp 8           # One replica sharded over GPUs 0-7
#
# Ports are assigned in replica order starting at BASE_PORT, so the live
# backends remain contiguous for run_annotate.sh.
#
# The native context is 262K; this conservative 32K default leaves memory for
# multimodal annotation concurrency and can be overridden with an environment
# variable. Longer contexts need YaRN, not just a larger --max-model-len.
#
# The video frame cap is lifted here by default. vLLM decimates every clip to
# 32 frames unless told otherwise, and this pipeline's per-request fps control
# only takes effect when the server was started with num_frames -1. Both that
# and the client's LEROBOT_OPENAI_SEND_MM_KWARGS=1 fail silently -- see
# ALIGN_HANDOFF.md.
#
# One request-side trap, measured against this build: vLLM 0.17 validates
# reasoning_effort against low|medium|high, while Qwen3.8's chat template
# accepts xhigh|medium|low. Only 'low' and 'medium' clear both -- 'high' and
# 'xhigh' each fail at the other end with a 400. Omit it for the template's
# xhigh default.
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TMPDIR="${QWEN38_TMPDIR:-$HOME/.cache/qwen38-tmp}"
export MAX_JOBS="${QWEN38_MAX_JOBS:-8}"
export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-$HF_HOME/assets}"
export HF_XET_CACHE="${HF_XET_CACHE:-$HF_HOME/xet}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

ORIGINAL_ARGS=( "$@" )

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${QWEN38_VENV:-$REPO_ROOT/.venv-vllm}"

MODEL="${QWEN38_MODEL:-Qwen/Qwen3.8-27B-FP8}"
SERVED_MODEL_NAME="${QWEN38_SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B}"
QUANTIZATION="${QWEN38_QUANTIZATION:-}"

SESSION="${QWEN38_TMUX_SESSION:-qwen38}"

BASE_PORT="${QWEN38_BASE_PORT:-8000}"

# Parallel sizes are resolved after GPU selection. Explicit --tp/--pp values
# always win; otherwise the 27B launcher uses TP 1 + PP 1, one replica per GPU.
TP_SIZE="${QWEN38_TP_SIZE:-}"
PP_SIZE="${QWEN38_PP_SIZE:-}"

TP_SIZE_EXPLICIT=false
PP_SIZE_EXPLICIT=false

if [[ -v QWEN38_TP_SIZE ]]; then
    TP_SIZE_EXPLICIT=true
fi

if [[ -v QWEN38_PP_SIZE ]]; then
    PP_SIZE_EXPLICIT=true
fi

MAX_MODEL_LEN="${QWEN38_MAX_MODEL_LEN:-32768}"

GPU_MEM_UTIL="${QWEN38_GPU_MEM_UTIL:-0.90}"
GPU_MEM_UTIL_EXPLICIT=false

if [[ -v QWEN38_GPU_MEM_UTIL ]]; then
    GPU_MEM_UTIL_EXPLICIT=true
fi

# Qwen3.8 is a gated-delta-net hybrid. Each concurrent decode sequence needs a
# recurrent state block in addition to its attention KV cache, so concurrency
# costs memory even before the KV pages. The default is resolved after the
# checkpoint is known: BF16 weights leave far less room than FP8 weights.
MAX_NUM_SEQS="${QWEN38_MAX_NUM_SEQS:-}"
MAX_NUM_SEQS_EXPLICIT=false

if [[ -v QWEN38_MAX_NUM_SEQS ]]; then
    MAX_NUM_SEQS_EXPLICIT=true
fi

# Pipeline parallelism enables eager execution in auto mode.
ENFORCE_EAGER="${QWEN38_ENFORCE_EAGER:-auto}"

# vLLM recommends data-parallel vision encoding for multimodal Qwen3.5-family
# models. It only applies when a replica spans more than one GPU.
MM_ENCODER_TP_MODE="${QWEN38_MM_ENCODER_TP_MODE:-data}"

# Lifts vLLM's 32-frame-per-clip cap so client-side fps/num_frames requests
# are honoured. Set to the empty string to serve with vLLM's default loader.
MEDIA_IO_KWARGS_DEFAULT='{"video": {"num_frames": -1}}'

# Unset falls back to the default; an explicitly empty value disables the flag.
MEDIA_IO_KWARGS="${QWEN38_MEDIA_IO_KWARGS-$MEDIA_IO_KWARGS_DEFAULT}"

LOG_DIR="${QWEN38_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/lerobot-align/qwen38}"


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

# Reprint this invocation with an explicitly configured cache group activated.
group_rerun_hint() {

    local required_group="${HF_CACHE_GROUP:-}"
    local rerun_command

    printf -v rerun_command '%q ' "$0" "${ORIGINAL_ARGS[@]}"

    if [[ -n "$required_group" ]]; then
        printf 'If the cache requires that group, retry with:\n\n'
        printf '  sg %q -c "bash %s"\n' "$required_group" "$rerun_command"
    else
        echo "Choose a writable HF_HOME or fix the cache directory permissions."
    fi
}


# ------------------------------------------------------------
# Help
# ------------------------------------------------------------

usage() {

cat <<EOF

Usage:

  $0 [COUNT] [options]


GPU selection:

  COUNT

      A bare number is a count: use the first COUNT GPUs.

        $0 8        ->  GPUs 0-7

      Default:

        8


  --gpus LIST

      Comma or space separated list of GPU indices.

        $0 --gpus 0,1,2,3    ->  four single-GPU replicas
        $0 --gpus 3          ->  one replica on GPU 3

      Note the difference: '2' means two GPUs, '--gpus 2'
      means the GPU with index 2.


Options:

  --tp N

      Tensor-parallel size. Must divide the 24 query heads,
      the 16 linear key heads, and the 48 linear value heads,
      so N must be 1, 2, 4, or 8.

        --tp 1   ->  one replica per GPU (default)
        --tp 8   ->  1 replica across 8 GPUs


  --pp N

      Pipeline-parallel size. TP * PP is the number of GPUs
      used by each replica.

        --tp 4 --pp 2   ->  1 replica across 8 GPUs

      Default:

        TP 1 + PP 1.


  --venv PATH

      Python virtual environment containing a vLLM build with
      native Qwen3.5-architecture support.

      Default:

        ${VENV_DIR}


  --help

      Show this help.


Ports are assigned in replica order from ${BASE_PORT}:

  replica 0 -> ${BASE_PORT}
  replica 1 -> $((BASE_PORT + 1))

Environment overrides:

  QWEN38_MODEL, QWEN38_SERVED_MODEL_NAME, QWEN38_QUANTIZATION,
  QWEN38_VENV, QWEN38_TMUX_SESSION, QWEN38_BASE_PORT,
  QWEN38_TP_SIZE, QWEN38_PP_SIZE, QWEN38_MAX_MODEL_LEN,
  QWEN38_MAX_NUM_SEQS, QWEN38_GPU_MEM_UTIL,
  QWEN38_ENFORCE_EAGER, QWEN38_MM_ENCODER_TP_MODE,
  QWEN38_MEDIA_IO_KWARGS, QWEN38_LOG_DIR, QWEN38_TMPDIR,
  QWEN38_MAX_JOBS, HF_HOME, HF_XET_HIGH_PERFORMANCE

Serve the BF16 checkpoint instead of the official FP8 one with:

  QWEN38_MODEL=Qwen/Qwen3.8-27B $0

EOF
}


# ------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------

GPU_SPEC=""
GPU_SPEC_IS_LIST=false

while [[ $# -gt 0 ]]; do

    case "$1" in

        --gpus)

            if [[ $# -lt 2 ]]; then
                echo "ERROR: --gpus requires a value, e.g. --gpus 1,3"
                exit 1
            fi

            GPU_SPEC="$2"
            GPU_SPEC_IS_LIST=true
            shift 2
            ;;


        --tp)

            if [[ $# -lt 2 ]]; then
                echo "ERROR: --tp requires a value, e.g. --tp 4"
                exit 1
            fi

            TP_SIZE="$2"
            TP_SIZE_EXPLICIT=true
            shift 2
            ;;


        --pp)

            if [[ $# -lt 2 ]]; then
                echo "ERROR: --pp requires a value, e.g. --pp 2"
                exit 1
            fi

            PP_SIZE="$2"
            PP_SIZE_EXPLICIT=true
            shift 2
            ;;


        --venv)

            if [[ $# -lt 2 ]]; then
                echo "ERROR: --venv requires a path, e.g. --venv .venv"
                exit 1
            fi

            VENV_DIR="$2"
            shift 2
            ;;


        --help|-h)

            usage
            exit 0
            ;;


        -*)

            echo "ERROR: Unknown option:"
            echo "  $1"
            echo

            usage

            exit 1
            ;;


        *)

            GPU_SPEC="$1"
            GPU_SPEC_IS_LIST=false
            shift
            ;;

    esac

done


mkdir -p "$LOG_DIR" "$TMPDIR"


case "$ENFORCE_EAGER" in
    auto|true|false)
        ;;
    *)
        echo "ERROR: QWEN38_ENFORCE_EAGER must be auto, true, or false."
        exit 1
        ;;
esac


# Validated here because it is later handed to awk as a numeric value.
if ! [[ "$GPU_MEM_UTIL" =~ ^(0?\.[0-9]+|1|1\.0+)$ ]]; then
    echo "ERROR: QWEN38_GPU_MEM_UTIL must be a fraction in (0, 1], got '$GPU_MEM_UTIL'."
    exit 1
fi


# Report cache permission problems before launching long-lived workers.
if [[ ! -d "$HF_HOME" ]] && ! mkdir -p "$HF_HOME" 2>/dev/null; then
    echo "ERROR: could not create the Hugging Face cache directory:"
    echo "  $HF_HOME"
    echo
    group_rerun_hint
    exit 1
fi

if [[ ! -w "$HF_HOME" ]]; then
    echo "ERROR: Hugging Face cache directory is not writable:"
    echo "  $HF_HOME"
    echo
    group_rerun_hint
    exit 1
fi

# The group that owns the cache. Baked into every runner so a tmux window that
# inherited credentials without it can re-acquire it before loading weights.
HF_HOME_GROUP="$(stat -c %G "$HF_HOME" 2>/dev/null || true)"

if [[ "$HF_HOME_GROUP" == "UNKNOWN" ]]; then
    HF_HOME_GROUP=""
fi


# ------------------------------------------------------------
# Python virtual environment
# ------------------------------------------------------------

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: Python virtual environment does not exist:"
    echo "  $VENV_DIR"
    echo
    if command -v uv >/dev/null 2>&1; then
        echo "Create it and install vLLM with:"
        printf '  uv venv %q --python 3.12' "$VENV_DIR"
        echo
        printf '  uv pip install --python %q vllm==0.17.0 --torch-backend=cu128' "$VENV_DIR/bin/python"
        echo
    else
        echo "Install uv, then create a Python 3.12 virtual environment here."
    fi
    exit 1
fi

# Some activation scripts reference unset variables, so relax nounset briefly.
set +u
source "$VENV_DIR/bin/activate"
set -u

echo "Python virtual environment:"
echo "  $VIRTUAL_ENV"
echo


# ------------------------------------------------------------
# Dependency checks
# ------------------------------------------------------------

for cmd in tmux nvidia-smi curl python awk; do
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: '$cmd' was not found in PATH."
        exit 1
    fi
done

if ! command -v vllm >/dev/null 2>&1; then
    echo "ERROR: vLLM was not found in $VENV_DIR."
    echo
    if command -v uv >/dev/null 2>&1; then
        echo "Install a current Qwen3.5-compatible build with:"
        printf '  uv pip install --python %q vllm==0.17.0 --torch-backend=cu128' "$VENV_DIR/bin/python"
        echo
    fi
    exit 1
fi


# Resolve vLLM to its absolute path so tmux windows do not need to reactivate
# the virtual environment.
VLLM_BIN="$(command -v vllm)"

VLLM_VERSION="$(python -c 'from importlib.metadata import version; print(version("vllm"))')"

echo "Using vLLM:"
echo "  $VLLM_BIN"
echo
echo "vLLM version:"
echo "  $VLLM_VERSION"
echo

if ! python -c 'import sys; from packaging.version import Version; raise SystemExit(Version(sys.argv[1]) < Version("0.17.0"))' "$VLLM_VERSION"
then
    echo "ERROR: Qwen3.8-27B requires vLLM >= 0.17.0; found $VLLM_VERSION."
    exit 1
fi


# Qwen3.8-27B is dense, so it registers as Qwen3_5ForConditionalGeneration --
# the MoE architecture the 397B launcher checks for is a different class.
if ! python -c \
    'from vllm.model_executor.models import ModelRegistry; raise SystemExit("Qwen3_5ForConditionalGeneration" not in ModelRegistry.get_supported_archs())' \
    >/dev/null 2>&1; then

    echo "ERROR: vLLM $VLLM_VERSION does not register Qwen3_5ForConditionalGeneration."
    echo
    echo "Install a current stable or nightly vLLM build in '$VENV_DIR'."
    exit 1
fi


# ------------------------------------------------------------
# Resolve and validate the GPU selection
# ------------------------------------------------------------

AVAILABLE_GPUS="$(
    nvidia-smi \
        --query-gpu=index \
        --format=csv,noheader \
    | wc -l \
    | tr -d ' '
)"

GPU_LIST=()

if [[ "$GPU_SPEC_IS_LIST" == "true" ]]; then

    IFS=', ' read -r -a GPU_LIST <<< "$GPU_SPEC"

else

    NUM_GPUS="${GPU_SPEC:-8}"

    if ! [[ "$NUM_GPUS" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: GPU count must be a positive integer, got '$NUM_GPUS'."
        echo
        echo "To select specific GPUs use a list instead:"
        echo "  $0 --gpus 1,3"
        exit 1
    fi

    if (( NUM_GPUS > AVAILABLE_GPUS )); then
        echo "ERROR:"
        echo "  Requested GPUs: $NUM_GPUS"
        echo "  Available GPUs: $AVAILABLE_GPUS"
        exit 1
    fi

    for (( i = 0; i < NUM_GPUS; i++ )); do
        GPU_LIST+=( "$i" )
    done

fi


if (( ${#GPU_LIST[@]} == 0 )); then
    echo "ERROR: No GPUs selected."
    exit 1
fi


for GPU in "${GPU_LIST[@]}"; do

    if ! [[ "$GPU" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "ERROR: Invalid GPU index: '$GPU'"
        echo "  Expected something like: --gpus 1,3"
        exit 1
    fi

    if (( GPU >= AVAILABLE_GPUS )); then
        echo "ERROR: GPU $GPU does not exist."
        echo "  This host has $AVAILABLE_GPUS GPU(s): 0-$((AVAILABLE_GPUS - 1))"
        exit 1
    fi

done


DUPLICATE_GPUS="$(printf '%s\n' "${GPU_LIST[@]}" | sort | uniq -d | tr '\n' ' ')"

if [[ -n "${DUPLICATE_GPUS// /}" ]]; then
    echo "ERROR: GPU listed more than once: ${DUPLICATE_GPUS}"
    echo "  Each GPU belongs to exactly one replica, so indices must be unique."
    exit 1
fi


NUM_GPUS="${#GPU_LIST[@]}"


# ------------------------------------------------------------
# Resolve parallelism and split GPUs into full replica groups
# ------------------------------------------------------------

if [[ "$TP_SIZE_EXPLICIT" == "false" ]]; then
    TP_SIZE=1
fi

if [[ "$PP_SIZE_EXPLICIT" == "false" ]]; then
    PP_SIZE=1
fi


if ! [[ "$TP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --tp must be a positive integer, got '$TP_SIZE'."
    exit 1
fi

if ! [[ "$PP_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --pp must be a positive integer, got '$PP_SIZE'."
    exit 1
fi


# Pipeline parallelism partitions whole layers, but tensor parallelism must
# divide every head count evenly. Gated attention has 24 query heads; the
# gated DeltaNet layers have 16 linear key heads and 48 linear value heads.
# The common divisors are 1, 2, 4, and 8.
if (( 24 % TP_SIZE != 0 || 16 % TP_SIZE != 0 || 48 % TP_SIZE != 0 )); then
    echo "ERROR: --tp $TP_SIZE is incompatible with Qwen3.8-27B's head counts."
    echo
    echo "  Query heads (gated attention): 24"
    echo "  Linear key heads (DeltaNet):   16"
    echo "  Linear value heads (DeltaNet): 48"
    echo
    echo "Use 1, 2, 4, or 8."
    exit 1
fi


REPLICA_SIZE=$(( TP_SIZE * PP_SIZE ))

if (( REPLICA_SIZE > NUM_GPUS )); then
    echo "ERROR:"
    echo "  Tensor-parallel size: $TP_SIZE"
    echo "  Pipeline-parallel size: $PP_SIZE"
    echo "  GPUs per replica:     $REPLICA_SIZE"
    echo "  Selected GPUs:        $NUM_GPUS"
    echo
    echo "A replica cannot span more GPUs than were selected."
    exit 1
fi

if (( NUM_GPUS % REPLICA_SIZE != 0 )); then
    echo "ERROR:"
    echo "  Selected GPUs:        $NUM_GPUS"
    echo "  Tensor-parallel size: $TP_SIZE"
    echo "  Pipeline-parallel size: $PP_SIZE"
    echo "  GPUs per replica:     $REPLICA_SIZE"
    echo
    echo "The GPU count must be a multiple of TP * PP so every replica is full."
    exit 1
fi

NUM_REPLICAS=$(( NUM_GPUS / REPLICA_SIZE ))


# ------------------------------------------------------------
# Checkpoint precision and memory fit
# ------------------------------------------------------------

# Weight footprints measured from the checkpoints in the shared cache:
# 52 GiB for BF16, 29 GiB for the official FP8 repository.
if [[ "$MODEL" == *[Ff][Pp]8* || -n "$QUANTIZATION" ]]; then
    CHECKPOINT_PRECISION="fp8"
    WEIGHTS_MIB=30720
else
    CHECKPOINT_PRECISION="bf16"
    WEIGHTS_MIB=55296
fi

# Below this much free memory a replica cannot hold the vision encoder, the
# DeltaNet state blocks, and a usable KV cache on top of the weights.
MIN_WORKING_MIB=6144

REPLICA_MEMORY_MIB=0

for GPU in "${GPU_LIST[@]:0:$REPLICA_SIZE}"; do
    GPU_MEMORY_MIB="$(nvidia-smi -i "$GPU" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d ' ')"
    REPLICA_MEMORY_MIB=$(( REPLICA_MEMORY_MIB + GPU_MEMORY_MIB ))
done

REPLICA_BUDGET_MIB="$(
    awk \
        -v total="$REPLICA_MEMORY_MIB" \
        -v util="$GPU_MEM_UTIL" \
        'BEGIN { printf "%d", total * util }'
)"

if (( REPLICA_BUDGET_MIB < WEIGHTS_MIB + MIN_WORKING_MIB )); then

    echo "ERROR: the $CHECKPOINT_PRECISION checkpoint cannot fit in this replica."
    echo
    echo "  GPUs per replica:       $REPLICA_SIZE"
    echo "  Replica memory:         ${REPLICA_MEMORY_MIB} MiB"
    echo "  Usable at ${GPU_MEM_UTIL}:         ${REPLICA_BUDGET_MIB} MiB"
    echo "  Weights:                ${WEIGHTS_MIB} MiB"
    echo "  Minimum working memory: ${MIN_WORKING_MIB} MiB"
    echo

    if [[ "$CHECKPOINT_PRECISION" == "bf16" ]]; then
        echo "Serve the official FP8 checkpoint instead:"
        echo "  QWEN38_MODEL=Qwen/Qwen3.8-27B-FP8 $0"
        echo
    fi

    echo "Or shard each replica over more GPUs, e.g. --tp 2."
    exit 1
fi


# BF16 weights leave roughly 18 GiB per H100 for KV pages, DeltaNet state, and
# the vision encoder; FP8 leaves closer to 43 GiB. Start concurrency where the
# checkpoint can pay for it and raise it only after measuring under load.
if [[ "$MAX_NUM_SEQS_EXPLICIT" == "false" ]]; then

    if [[ "$CHECKPOINT_PRECISION" == "fp8" ]]; then
        MAX_NUM_SEQS=16
    else
        MAX_NUM_SEQS=8
    fi

fi


if (( PP_SIZE > 1 )); then

    if [[ "$GPU_MEM_UTIL_EXPLICIT" == "false" ]]; then
        GPU_MEM_UTIL=0.95
    fi

    if [[ "$MAX_NUM_SEQS_EXPLICIT" == "false" ]]; then
        MAX_NUM_SEQS=8
    fi

fi

if [[ "$ENFORCE_EAGER" == "auto" ]]; then
    if (( PP_SIZE > 1 )); then
        ENFORCE_EAGER=true
    else
        ENFORCE_EAGER=false
    fi
fi


# ------------------------------------------------------------
# Pick a tmux server that can actually read the weights
#
# tmux windows are children of the tmux *server*, not of this shell. A server
# started before the storage group was granted -- or from a shell where it was
# not active -- hands vLLM credentials that cannot reach the NFS cache, and
# transformers reports that as a bogus "another user is downloading the same
# model" error. Re-acquiring the group inside the window would fix the read and
# break the kill: a pane whose child is sg survives `tmux kill-session` and
# keeps its GPU. So the group is acquired one level up, by starting a private
# tmux server for this session.
# ------------------------------------------------------------

TMUX_SOCKET_ARGS=()
TMUX_LABEL="tmux"
START_SERVER_UNDER_GROUP=false

tmux_server_can_read_cache() {

    local probe_session="${SESSION}-cache-probe"
    local probe_file="${LOG_DIR}/.tmux-group-probe"
    local i

    rm -f "$probe_file"

    # Starts a server if none is running, in which case it inherits this
    # shell's credentials -- which the checks above already proved sufficient.
    if ! tmux new-session -d -s "$probe_session" \
        "if [ -r $(printf '%q' "$HF_HOME") ]; then echo ok; else echo no; fi > $(printf '%q' "$probe_file")" \
        2>/dev/null; then
        return 1
    fi

    for (( i = 0; i < 50; i++ )); do
        [[ -s "$probe_file" ]] && break
        sleep 0.1
    done

    tmux kill-session -t "=$probe_session" 2>/dev/null || true

    [[ "$(cat "$probe_file" 2>/dev/null)" == "ok" ]]
}


if ! tmux_server_can_read_cache; then

    if [[ -z "$HF_HOME_GROUP" ]]; then
        echo "ERROR: the running tmux server cannot read the Hugging Face cache:"
        echo "  $HF_HOME"
        echo
        echo "Its owning group could not be determined, so this launcher cannot"
        echo "start a server that can. Restart the tmux server from a shell that"
        echo "can read the cache, then run this script again."
        exit 1
    fi

    if ! command -v sg >/dev/null 2>&1; then
        echo "ERROR: the running tmux server cannot read the Hugging Face cache:"
        echo "  $HF_HOME"
        echo
        echo "'sg' is not available to start one under the ${HF_HOME_GROUP} group."
        exit 1
    fi

    TMUX_SOCKET_ARGS=( -L "$SESSION" )
    TMUX_LABEL="tmux -L $SESSION"

    if [[ "$(id -gn)" != "$HF_HOME_GROUP" ]]; then
        START_SERVER_UNDER_GROUP=true
    fi

    echo "NOTE: the running tmux server cannot read"
    echo "  $HF_HOME"
    echo
    echo "Using a private tmux server on socket '${SESSION}' instead, started"
    echo "under the ${HF_HOME_GROUP} group. Attach and kill it with"
    echo "'${TMUX_LABEL} ...', not plain tmux."
    echo
fi


# ------------------------------------------------------------
# Prevent a duplicate tmux session
# ------------------------------------------------------------

# The '=' prefix forces an exact name match. Without it tmux resolves the
# target by prefix, so an unrelated 'qwen38-smoke' session would look like
# this one and block the launch.
if tmux "${TMUX_SOCKET_ARGS[@]}" has-session -t "=$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists."
    echo
    echo "Attach:"
    echo "  ${TMUX_LABEL} attach -t $SESSION"
    echo
    echo "Stop:"
    echo "  ${TMUX_LABEL} kill-session -t $SESSION"
    exit 1
fi


# ------------------------------------------------------------
# Print configuration
# ------------------------------------------------------------

echo "============================================================"
echo " Starting persistent Qwen3.8 servers"
echo "============================================================"
echo
echo "Weight checkpoint:"
echo "  $MODEL ($CHECKPOINT_PRECISION)"
echo
echo "Served model name:"
echo "  $SERVED_MODEL_NAME"
echo
echo "Online quantization:"
echo "  ${QUANTIZATION:-checkpoint-defined}"
echo
echo "GPUs:"
echo "  $NUM_GPUS (${GPU_LIST[*]})"
echo
echo "Tensor-parallel size:"
echo "  $TP_SIZE"
echo
echo "Pipeline-parallel size:"
echo "  $PP_SIZE"
echo
echo "GPUs per replica:"
echo "  $REPLICA_SIZE"
echo
echo "Replicas:"
echo "  $NUM_REPLICAS"
echo
echo "Replica -> GPUs -> port:"
for (( r = 0; r < NUM_REPLICAS; r++ )); do

    REPLICA_GPUS=( "${GPU_LIST[@]:$(( r * REPLICA_SIZE )):$REPLICA_SIZE}" )

    printf '  %-9s %-12s %s\n' \
        "$r" \
        "$(IFS=,; echo "${REPLICA_GPUS[*]}")" \
        "$((BASE_PORT + r))"
done
echo
echo "Max model length:"
echo "  $MAX_MODEL_LEN"
echo
echo "Max num seqs:"
echo "  $MAX_NUM_SEQS"
echo
echo "GPU memory utilization:"
echo "  $GPU_MEM_UTIL"
echo
echo "Enforce eager execution:"
echo "  $ENFORCE_EAGER"
echo
echo "Multimodal encoder TP mode:"
if (( TP_SIZE > 1 )); then
    echo "  $MM_ENCODER_TP_MODE"
else
    echo "  not applicable (TP 1)"
fi
echo
echo "Media IO kwargs:"
echo "  ${MEDIA_IO_KWARGS:-vLLM default (32 frames per clip)}"
echo
echo "Reasoning parser:"
echo "  qwen3"
echo
echo "tmux session:"
echo "  $SESSION"
echo
echo "Logs:"
echo "  $LOG_DIR"
echo
echo "============================================================"
echo


# ------------------------------------------------------------
# Launch one server per parallel replica
#
# Each replica gets a generated runner script. This avoids nested shell
# quoting bugs and explicitly exports the environment that a long-running
# tmux server would otherwise fail to inherit from this shell.
# ------------------------------------------------------------

for (( r = 0; r < NUM_REPLICAS; r++ )); do

    REPLICA_GPUS=( "${GPU_LIST[@]:$(( r * REPLICA_SIZE )):$REPLICA_SIZE}" )
    GPU_CSV="$(IFS=,; echo "${REPLICA_GPUS[*]}")"

    PORT=$((BASE_PORT + r))
    WINDOW="r${r}-gpu$(IFS=-; echo "${REPLICA_GPUS[*]}")"
    LOG_FILE="${LOG_DIR}/${WINDOW}.log"
    RUNNER="$(mktemp "${LOG_DIR}/${WINDOW}.XXXXXX.sh")"

    echo "Starting:"
    echo "  replica $r  GPUs $GPU_CSV -> port $PORT"
    echo "  log: $LOG_FILE"

    CMD=(
        "$VLLM_BIN" serve "$MODEL"
        --served-model-name "$SERVED_MODEL_NAME"
        --host 127.0.0.1
        --port "$PORT"
        --tensor-parallel-size "$TP_SIZE"
        --pipeline-parallel-size "$PP_SIZE"
        --reasoning-parser qwen3
        --trust-remote-code
        --enable-prefix-caching
        --mm-processor-cache-type shm
        --max-model-len "$MAX_MODEL_LEN"
        --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEM_UTIL"
    )

    # Vision encoder sharding only exists when a replica spans several GPUs.
    if (( TP_SIZE > 1 )); then
        CMD+=( --mm-encoder-tp-mode "$MM_ENCODER_TP_MODE" )
    fi

    if [[ -n "$MEDIA_IO_KWARGS" ]]; then
        CMD+=( --media-io-kwargs "$MEDIA_IO_KWARGS" )
    fi

    if [[ -n "$QUANTIZATION" ]]; then
        CMD+=( --quantization "$QUANTIZATION" )
    fi

    if [[ "$ENFORCE_EAGER" == "true" ]]; then
        CMD+=( --enforce-eager )
    fi

    {
        printf '#!/usr/bin/env bash\n'
        printf '# Generated by start_qwen38.sh on %s\n' "$(date +%Y-%m-%dT%H:%M:%S)"
        printf 'set -o pipefail\n'
        printf 'umask 077\n\n'
        printf '%s\n\n' 'trap '\''rm -f -- "$0"'\'' EXIT'
        printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$GPU_CSV"
        printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
        printf 'export PATH=%q\n' "$PATH"
        printf 'export TMPDIR=%q\n' "$TMPDIR"
        printf 'export MAX_JOBS=%q\n' "$MAX_JOBS"
        printf 'export HF_HOME=%q\n' "$HF_HOME"
        printf 'export HF_HUB_CACHE=%q\n' "$HF_HUB_CACHE"
        printf 'export HUGGINGFACE_HUB_CACHE=%q\n' "$HUGGINGFACE_HUB_CACHE"
        printf 'export HF_ASSETS_CACHE=%q\n' "$HF_ASSETS_CACHE"
        printf 'export HF_XET_CACHE=%q\n' "$HF_XET_CACHE"
        printf 'export HF_XET_HIGH_PERFORMANCE=%q\n' "$HF_XET_HIGH_PERFORMANCE"

        # DeepGEMM is an optional FP8 kernel path; the 397B launcher skips it
        # on this host and this launcher keeps that environment identical.
        printf 'export VLLM_DEEP_GEMM_WARMUP=skip\n'
        printf 'export VLLM_USE_DEEP_GEMM=0\n'

        if [[ -n "${HF_TOKEN:-}" ]]; then
            printf 'export HF_TOKEN=%q\n' "$HF_TOKEN"
        fi

        printf '\nCMD=('
        printf ' %q' "${CMD[@]}"
        printf ' )\n\n'
        printf '"${CMD[@]}" 2>&1 | tee -a %q\n' "$LOG_FILE"
        printf 'exit "${PIPESTATUS[0]}"\n'

    } > "$RUNNER"

    chmod 700 "$RUNNER"

    # tmux treats its final argument as a shell command string. Quote the
    # runner path before handing it over so custom log paths may contain spaces.
    printf -v TMUX_CMD 'exec %q' "$RUNNER"

    if (( r == 0 )); then

        # Only the first call starts the server, so only it needs the group.
        if [[ "$START_SERVER_UNDER_GROUP" == "true" ]]; then

            printf -v START_CMD '%q ' \
                tmux "${TMUX_SOCKET_ARGS[@]}" \
                new-session -d -s "$SESSION" -n "$WINDOW" "$TMUX_CMD"

            if ! sg "$HF_HOME_GROUP" -c "$START_CMD"; then
                rm -f -- "$RUNNER"
                echo "ERROR: could not start tmux session '$SESSION'." >&2
                exit 1
            fi

        else

            if ! tmux "${TMUX_SOCKET_ARGS[@]}" new-session \
                -d \
                -s "$SESSION" \
                -n "$WINDOW" \
                "$TMUX_CMD"
            then
                rm -f -- "$RUNNER"
                echo "ERROR: could not start tmux session '$SESSION'." >&2
                exit 1
            fi

        fi

    else

        if ! tmux "${TMUX_SOCKET_ARGS[@]}" new-window \
            -t "=$SESSION" \
            -n "$WINDOW" \
            "$TMUX_CMD"
        then
            rm -f -- "$RUNNER"
            echo "ERROR: could not start tmux window '$WINDOW'." >&2
            exit 1
        fi

    fi

    # Keep failed windows visible so their logs and exit statuses can be
    # inspected after a long model-loading attempt.
    tmux "${TMUX_SOCKET_ARGS[@]}" set-option \
        -w \
        -t "=${SESSION}:${WINDOW}" \
        remain-on-exit on

done


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

echo
echo "============================================================"
echo " Servers launched"
echo "============================================================"
echo

for (( r = 0; r < NUM_REPLICAS; r++ )); do
    REPLICA_GPUS=( "${GPU_LIST[@]:$(( r * REPLICA_SIZE )):$REPLICA_SIZE}" )

    echo "Replica $r (GPUs $(IFS=,; echo "${REPLICA_GPUS[*]}")):"
    echo "  http://127.0.0.1:$((BASE_PORT + r))/v1"
done

echo

if [[ "$CHECKPOINT_PRECISION" == "fp8" ]]; then
    echo "The first start may download ~29 GiB of FP8 weights, then load and compile."
else
    echo "The first start may download ~52 GiB of BF16 weights, then load and compile."
fi

echo "Every replica loads its own copy, so a cold start is I/O bound on the cache."
echo "Watch the tmux logs during the cold start."
echo

if (( PP_SIZE > 1 )); then
    echo "NOTE: pipeline parallelism starts conservatively with eager execution"
    echo "and max-num-seqs 8. Confirm GPU headroom before raising concurrency."
    echo
fi

echo "Run annotation with matching model and backend settings:"
echo
echo "  LEROBOT_OPENAI_SEND_MM_KWARGS=1 \\"
echo "  VLLM_MODEL=${SERVED_MODEL_NAME} \\"
echo "  VLLM_NUM_BACKENDS=${NUM_REPLICAS} \\"
echo "  VLLM_BASE_PORT=${BASE_PORT} \\"
echo "  VLLM_TMUX_SESSION=${SESSION} \\"

if (( NUM_REPLICAS == 1 )); then
    echo "  VLM_PORT=${BASE_PORT} \\"
else
    echo "  VLM_PORT=9000 \\"
fi

echo "    ./run_annotate.sh SOURCE DEST"
echo
echo "LEROBOT_OPENAI_SEND_MM_KWARGS=1 is not optional for video prompts: without"
echo "it the client silently drops fps/num_frames and the server's lifted frame"
echo "cap does nothing."
echo
echo "If you set --vlm.reasoning_effort, only 'low' and 'medium' work here:"
echo "vLLM ${VLLM_VERSION} accepts low|medium|high and this chat template accepts"
echo "xhigh|medium|low, so 'high' and 'xhigh' each 400 at the other end."
echo "Omit it for the template's xhigh default."
echo

if (( NUM_REPLICAS > 1 )); then
    echo "run_annotate.sh talks to one base URL, so ${NUM_REPLICAS} replicas need the nginx"
    echo "load balancer on :9000 with upstreams covering ports"
    echo "${BASE_PORT}-$((BASE_PORT + NUM_REPLICAS - 1)) only -- or one annotation run per port."
    echo
fi

echo "Attach:"
echo
echo "  tmux attach -t $SESSION"
echo
echo "Monitor GPUs:"
echo
echo "  watch -n 1 nvidia-smi"
echo
echo "Check tmux windows:"
echo
echo "  ${TMUX_LABEL} list-windows -t $SESSION"
echo
echo "Stop all servers:"
echo
echo "  tmux kill-session -t $SESSION"
echo
