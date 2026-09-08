#!/usr/bin/env bash
set -euo pipefail
umask 077

# ============================================================
# Docker-based vLLM launcher for Qwen3.8-Flash-Next
#
# This is deliberately NOT a flag on start_qwen38.sh. That script serves
# Qwen3.8-27B, a dense Qwen3_5ForConditionalGeneration model, from the pip
# venv in .venv-vllm. Flash-Next is a different architecture entirely --
# Qwen4ExpForConditionalGeneration (model_type qwen4_exp) -- which vLLM 0.17
# does not have in its registry at all. The vLLM recipe is explicit that
# "PyPI installation is not supported for this recipe": the arch ships only in
# the dedicated vllm/vllm-openai:qwen38-flash-next image. Hence a container.
#
# Sizing, for 80 GiB cards:
#
#   BF16 checkpoint  335.3 GiB  -> needs 5+ GPUs for weights alone
#   FP8  checkpoint  172.8 GiB  -> 43.2 GiB/GPU at TP4
#
# So FP8 is not a fallback on a 4-GPU tray, it is the only option. BF16 does
# not fit in 4 x 80 GiB (320 GiB) even before KV cache.
#
# Why TEP and not plain TP: the FP8 checkpoint is block-quantized 128 wide,
# and moe_intermediate_size is 640. Sharding that dim by TP gives 160 (TP4) or
# 80 (TP8), neither a multiple of 128. Expert parallelism keeps each expert
# whole and distributes experts instead (512 / 4 = 128 per GPU), which is why
# the recipe calls plain TP8 incompatible and prescribes TEP on Hopper along
# with the Triton MoE backend.
#
# Why PLE CPU offload is on by default: 51B of the 176B total is an N-gram
# embedding lookup table with almost no per-token compute. Offloading it to
# host RAM cuts GPU-resident weights to roughly 122 GiB (~30.5 GiB/GPU),
# leaving ~41 GiB per card for KV cache and the Gated DeltaNet recurrent
# state. On a 4-GPU tray that headroom is what makes multimodal batches safe.
# Needs ~51 GB of host RAM.
#
# Usage:
#   ./start_qwen38_flashnext.sh                 # GPUs 4-7, TEP4, port 8000
#   ./start_qwen38_flashnext.sh --gpus 0,1,2,3
#   ./start_qwen38_flashnext.sh --port 8100
#   ./start_qwen38_flashnext.sh --bf16          # refuses unless >=5 GPUs
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
IMAGE="${FLASHNEXT_IMAGE:-vllm/vllm-openai:qwen38-flash-next}"

MODEL="${FLASHNEXT_MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}"
SERVED_MODEL_NAME="${FLASHNEXT_SERVED_MODEL_NAME:-Qwen/Qwen3.8-Flash-Next}"

# Default to the first four visible GPUs; override for the local topology.
GPUS="${FLASHNEXT_GPUS:-0,1,2,3}"

PORT="${FLASHNEXT_PORT:-8000}"

CONTAINER="${FLASHNEXT_CONTAINER:-qwen38-flashnext}"

# Native context is 262144. That reserves a large KV pool; 65536 is enough for
# annotation prompts and leaves room for multimodal batches. Beyond 262144
# needs YaRN, not just a bigger number here.
MAX_MODEL_LEN="${FLASHNEXT_MAX_MODEL_LEN:-65536}"

GPU_MEM_UTIL="${FLASHNEXT_GPU_MEM_UTIL:-0.90}"

# The recipe is specific: dropping below 256 triggers a mamba-cache capacity
# error at startup rather than merely reducing throughput.
MAX_NUM_SEQS="${FLASHNEXT_MAX_NUM_SEQS:-256}"

# Keeps the 51B N-gram table in host RAM.
PLE_CPU_OFFLOAD="${FLASHNEXT_PLE_CPU_OFFLOAD:-1}"

# Triton is the MoE backend the recipe prescribes for Hopper.
MOE_BACKEND="${FLASHNEXT_MOE_BACKEND:-triton}"

# Lifts vLLM's 32-frame-per-clip cap so the pipeline's per-request fps and
# num_frames actually take effect. Pairs with LEROBOT_OPENAI_SEND_MM_KWARGS=1
# on the client. Both fail silently when missing.
MEDIA_IO_KWARGS_DEFAULT='{"video": {"num_frames": -1}}'
MEDIA_IO_KWARGS="${FLASHNEXT_MEDIA_IO_KWARGS-$MEDIA_IO_KWARGS_DEFAULT}"

# Splits thinking into reasoning_content so it cannot leak into the JSON the
# annotation pipeline parses out of content.
REASONING_PARSER="${FLASHNEXT_REASONING_PARSER:-qwen3}"

# Set FLASHNEXT_MODEL_PATH to serve a pre-staged local checkpoint. When it is
# unset, the container resolves MODEL from the Hugging Face cache mounted at
# /hf instead of assuming a machine-specific filesystem layout.
MODEL_PATH="${FLASHNEXT_MODEL_PATH:-}"

LOG_DIR="${FLASHNEXT_LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/lerobot-align/qwen38-flashnext}"

USE_BF16=false
EXTRA_ARGS=()


# ------------------------------------------------------------
# Arguments
# ------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpus)      GPUS="$2"; shift 2 ;;
        --port)      PORT="$2"; shift 2 ;;
        --model)     MODEL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --container) CONTAINER="$2"; shift 2 ;;
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --bf16)      USE_BF16=true; shift ;;
        --no-ple-offload) PLE_CPU_OFFLOAD=0; shift ;;
        --help|-h)   sed -n '3,52p' "$0"; exit 0 ;;
        --)          shift; EXTRA_ARGS=( "$@" ); break ;;
        *)           echo "ERROR: unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ "$USE_BF16" == true ]]; then
    MODEL="Qwen/Qwen3.8-Flash-Next"
    MODEL_PATH=""
fi

# A staged directory is mounted at /model and served from there; otherwise the
# container resolves MODEL from the Hub into its own cache.
if [[ -n "$MODEL_PATH" ]]; then
    if [[ ! -d "$MODEL_PATH" ]]; then
        echo "ERROR: model directory not found: $MODEL_PATH" >&2
        echo "  Stage the checkpoint there, or pass --model-path / --model." >&2
        exit 1
    fi
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        echo "ERROR: $MODEL_PATH has no config.json; staging looks incomplete." >&2
        exit 1
    fi

    # A half-copied directory still has config.json and still measures a
    # plausible size, so the fit check below would pass and the load would die
    # partway through. Compare against the shard index instead.
    #
    # A missing index is itself a sign of incomplete staging, not a reason to
    # skip the check: `cp` writes model-00001-of-000NN.safetensors long before
    # model.safetensors.index.json, because '-' sorts ahead of '.'. Treating
    # absence as "nothing to verify" is exactly backwards.
    INDEX="$MODEL_PATH/model.safetensors.index.json"
    if [[ -f "$INDEX" ]]; then
        MISSING=$(python3 - "$INDEX" "$MODEL_PATH" <<'PYEOF'
import json, os, sys
index, root = sys.argv[1], sys.argv[2]
with open(index) as fh:
    shards = set(json.load(fh)["weight_map"].values())
print(sum(not os.path.exists(os.path.join(root, s)) for s in shards))
PYEOF
        )
        if [[ "$MISSING" != "0" ]]; then
            echo "ERROR: $MISSING shard(s) named in the index are missing from $MODEL_PATH." >&2
            echo "  Staging is incomplete -- wait for the copy to finish." >&2
            exit 1
        fi
    elif [[ ! -f "$MODEL_PATH/model.safetensors" ]]; then
        echo "ERROR: $MODEL_PATH has neither model.safetensors.index.json nor" >&2
        echo "       model.safetensors, so the checkpoint is not fully staged." >&2
        echo "  Wait for the copy to finish, or point --model-path elsewhere." >&2
        exit 1
    fi

    MODEL_REF="/model"
else
    MODEL_REF="$MODEL"
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
NUM_GPUS="${#GPU_LIST[@]}"
TP_SIZE="$NUM_GPUS"


# ------------------------------------------------------------
# Preflight
# ------------------------------------------------------------
if ! docker info >/dev/null 2>&1; then
    echo "ERROR: docker is not usable by $(id -un)." >&2
    exit 1
fi

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "ERROR: image not present locally: $IMAGE" >&2
    echo "  docker pull $IMAGE" >&2
    exit 1
fi

# Weight footprint check. These are the measured checkpoint sizes, not
# estimates; going over means the load dies partway through.
# Size from the actual checkpoint, which for a staged dir means measuring it
# rather than trusting the name.
if [[ -n "$MODEL_PATH" ]]; then
    WEIGHTS_GIB=$(du -sBG --apparent-size "$MODEL_PATH" 2>/dev/null | cut -dG -f1)
    if [[ -z "$WEIGHTS_GIB" ]]; then
        echo "ERROR: could not size $MODEL_PATH" >&2
        exit 1
    fi
elif [[ "$MODEL" == *"-FP8" ]]; then
    WEIGHTS_GIB=173
else
    WEIGHTS_GIB=336
fi

# The offloaded N-gram table never lands on the GPU.
if [[ "$PLE_CPU_OFFLOAD" == "1" ]]; then
    GPU_WEIGHTS_GIB=$(( WEIGHTS_GIB - 51 ))
else
    GPU_WEIGHTS_GIB="$WEIGHTS_GIB"
fi

PER_GPU_GIB=$(( (GPU_WEIGHTS_GIB + NUM_GPUS - 1) / NUM_GPUS ))

# Read the smallest card in the set rather than assuming a uniform tray.
MIN_TOTAL_MIB=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits \
    | awk -F', ' -v want=",$GPUS," '
        index(want, ","$1",") { if (min == "" || $2 < min) min = $2 }
        END { print (min == "" ? 0 : min) }')

if (( MIN_TOTAL_MIB == 0 )); then
    echo "ERROR: could not read memory for GPUs: $GPUS" >&2
    exit 1
fi

BUDGET_GIB=$(awk -v mib="$MIN_TOTAL_MIB" -v util="$GPU_MEM_UTIL" \
    'BEGIN { printf "%d", (mib / 1024) * util }')

echo "Memory plan:"
echo "  checkpoint            ${WEIGHTS_GIB} GiB"
if [[ "$PLE_CPU_OFFLOAD" == "1" ]]; then
    echo "  N-gram table offload  -51 GiB to host RAM"
fi
echo "  GPU-resident weights  ${GPU_WEIGHTS_GIB} GiB over ${NUM_GPUS} GPU(s)"
echo "  per GPU               ${PER_GPU_GIB} GiB of ${BUDGET_GIB} GiB budget"

if (( PER_GPU_GIB >= BUDGET_GIB )); then
    echo >&2
    echo "ERROR: weights do not fit." >&2
    echo "  ${PER_GPU_GIB} GiB/GPU needed, ${BUDGET_GIB} GiB available at util ${GPU_MEM_UTIL}." >&2
    if [[ "$MODEL" != *"-FP8" ]]; then
        echo "  The BF16 checkpoint needs more GPUs; use the FP8 checkpoint (default)." >&2
    else
        echo "  Add GPUs with --gpus, or keep N-gram offload enabled." >&2
    fi
    exit 1
fi

# Leave a floor for KV cache, activations and the Gated DeltaNet recurrent
# state. Under this the server starts and then OOMs on the first real batch.
HEADROOM_GIB=$(( BUDGET_GIB - PER_GPU_GIB ))
if (( HEADROOM_GIB < 12 )); then
    echo "  WARNING: only ${HEADROOM_GIB} GiB/GPU left for KV cache and GDN state." >&2
    echo "           Lower --max-model-len or add GPUs if startup OOMs." >&2
fi

if [[ "$PLE_CPU_OFFLOAD" == "1" ]]; then
    HOST_AVAIL_GIB=$(free -g | awk '/^Mem:/ { print $7 }')
    if (( HOST_AVAIL_GIB < 60 )); then
        echo "  WARNING: ${HOST_AVAIL_GIB} GiB host RAM available; offload wants ~51 GiB + headroom." >&2
    fi
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo >&2
    echo "ERROR: container '$CONTAINER' already exists." >&2
    echo "  docker logs -f $CONTAINER      # watch it" >&2
    echo "  docker rm -f $CONTAINER        # replace it" >&2
    exit 1
fi

if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
    echo "ERROR: port ${PORT} is already in use." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"


# ------------------------------------------------------------
# Launch
# ------------------------------------------------------------
SERVE_ARGS=(
    "$MODEL_REF"
    --served-model-name "$SERVED_MODEL_NAME"
    --host 0.0.0.0
    --port "$PORT"
    --tensor-parallel-size "$TP_SIZE"
    --enable-expert-parallel
    --moe-backend "$MOE_BACKEND"
    --gpu-memory-utilization "$GPU_MEM_UTIL"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --enable-prefix-caching
    --no-enable-flashinfer-autotune
    --reasoning-parser "$REASONING_PARSER"
    --trust-remote-code
)

if [[ -n "$MEDIA_IO_KWARGS" ]]; then
    SERVE_ARGS+=( --media-io-kwargs "$MEDIA_IO_KWARGS" )
fi

SERVE_ARGS+=( "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" )

MOUNT_ARGS=()
if [[ -n "$MODEL_PATH" ]]; then
    MOUNT_ARGS=( -v "${MODEL_PATH}:/model:ro" )
else
    MOUNT_ARGS=( -v "${FLASHNEXT_HF_HOME:-$HOME/.cache/huggingface}:/hf" -e HF_HOME=/hf )
fi

echo
echo "Launching:"
echo "  image      $IMAGE"
if [[ -n "$MODEL_PATH" ]]; then
    echo "  model      $MODEL_PATH (mounted at /model)"
else
    echo "  model      $MODEL"
fi
echo "  GPUs       $GPUS (TEP${TP_SIZE}, MoE backend ${MOE_BACKEND})"
echo "  port       $PORT"
echo "  context    $MAX_MODEL_LEN"
echo "  container  $CONTAINER"
echo

docker run -d \
    --name "$CONTAINER" \
    --gpus "\"device=${GPUS}\"" \
    --ipc host \
    --shm-size 32g \
    -p "127.0.0.1:${PORT}:${PORT}" \
    ${MOUNT_ARGS[@]+"${MOUNT_ARGS[@]}"} \
    -e VLLM_PLE_CPU_OFFLOAD="$PLE_CPU_OFFLOAD" \
    -e VLLM_LOGGING_LEVEL=INFO \
    "$IMAGE" \
    "${SERVE_ARGS[@]}"

echo "Started. Weight load for a 173 GiB checkpoint takes several minutes."
echo
echo "  docker logs -f $CONTAINER"
echo "  curl -s http://127.0.0.1:${PORT}/v1/models"
echo "  docker rm -f $CONTAINER"
