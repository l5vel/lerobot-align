#!/usr/bin/env bash
# Import existing human subtask spans, then generate plan/memory/task
# augmentations for each dataset listed in a text file.
set -uo pipefail
umask 077

usage() {
    cat <<'EOF'
Usage: scripts/run_import_batch.sh REPO_LIST_FILE

Required environment:
  DST_OWNER          Hugging Face namespace for generated datasets

Each destination name is the flattened source ID plus a 10-character source-ID
hash (for example, org/data -> $DST_OWNER/org-data-<hash>). This prevents two
organizations' same-named datasets from sharing work, log, or done-marker paths.

Optional environment:
  SOURCE_REVISION    Source revision to resolve (default: main)
  ANNOTATE_VENV      Virtual environment (default: .venv)
  WORK_ROOT          Writable staging root
  LOG_DIR            Log directory
  KEEP_WORK          Keep successful staging trees: 1 or 0 (default: 1)
  PUSH_PRIVATE       Upload private datasets: true or false (default: false)
  ALLOW_VERSION_TAG_MOVE
                     Explicitly allow updating an existing destination's
                     LeRobot version tag: true or false (default: false)
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 1 ]]; then
    usage >&2
    exit 2
fi

REPO_LIST="$1"
[[ -f "$REPO_LIST" ]] || { echo "ERROR: repository list not found: $REPO_LIST" >&2; exit 2; }

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="${ANNOTATE_VENV:-$REPO_ROOT/.venv}"
ANNOTATE="$VENV_DIR/bin/lerobot-align"
PY="$VENV_DIR/bin/python"

export HF_HOME="${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HUB_CACHE}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

WORK_ROOT="${WORK_ROOT:-${XDG_CACHE_HOME:-$HOME/.cache}/lerobot-align/work}"
LOG_DIR="${LOG_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/lerobot-align/import-batch}"
DST_OWNER="${DST_OWNER:-}"
SOURCE_REVISION="${SOURCE_REVISION:-main}"
KEEP_WORK="${KEEP_WORK:-1}"
PUSH_PRIVATE="${PUSH_PRIVATE:-false}"
ALLOW_VERSION_TAG_MOVE="${ALLOW_VERSION_TAG_MOVE:-false}"

VLM_MODEL="${VLM_MODEL:-Qwen/Qwen3.8-27B}"
VLM_API_BASE="${VLM_API_BASE:-http://127.0.0.1:8000/v1}"
EPISODE_PARALLELISM="${EPISODE_PARALLELISM:-8}"
CLIENT_CONCURRENCY="${CLIENT_CONCURRENCY:-8}"

valid_component() {
    local value="$1"
    [[ "$value" =~ ^[A-Za-z0-9_]([A-Za-z0-9._-]*[A-Za-z0-9_])?$ ]] \
        && [[ "$value" != *".."* ]] \
        && [[ "$value" != *"--"* ]] \
        && [[ "$value" != *.git ]] \
        && (( ${#value} <= 96 ))
}

valid_repo_id() {
    local value="$1" owner name
    [[ "$value" == */* && "$value" != */*/* ]] || return 1
    owner="${value%%/*}"
    name="${value#*/}"
    valid_component "$owner" && valid_component "$name" && (( ${#value} <= 96 ))
}

valid_component "$DST_OWNER" || {
    echo "ERROR: set DST_OWNER to a valid Hugging Face namespace." >&2
    exit 2
}
MAX_KEY_LEN=$((95 - ${#DST_OWNER}))
(( MAX_KEY_LEN >= 12 )) || { echo "ERROR: DST_OWNER is too long to form a valid destination ID." >&2; exit 2; }
[[ "$KEEP_WORK" == "0" || "$KEEP_WORK" == "1" ]] || { echo "ERROR: KEEP_WORK must be 0 or 1." >&2; exit 2; }
[[ "$PUSH_PRIVATE" == "true" || "$PUSH_PRIVATE" == "false" ]] || { echo "ERROR: PUSH_PRIVATE must be true or false." >&2; exit 2; }
[[ "$ALLOW_VERSION_TAG_MOVE" == "true" || "$ALLOW_VERSION_TAG_MOVE" == "false" ]] || {
    echo "ERROR: ALLOW_VERSION_TAG_MOVE must be true or false." >&2
    exit 2
}

for command_name in curl flock realpath sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || { echo "ERROR: '$command_name' is required." >&2; exit 1; }
done
[[ -x "$ANNOTATE" ]] || { echo "ERROR: missing executable: $ANNOTATE" >&2; exit 1; }
[[ -x "$PY" ]] || { echo "ERROR: missing Python interpreter: $PY" >&2; exit 1; }

if ! PACKAGE_VERSION="$("$PY" -c 'from importlib.metadata import version; print(version("lerobot-align"))')"; then
    echo "ERROR: cannot determine the installed lerobot-align version in $VENV_DIR." >&2
    exit 1
fi
SCRIPT_SHA="$(sha256sum "$SCRIPT_DIR/run_import_batch.sh" | cut -d' ' -f1)"
RECIPE_FINGERPRINT="$(
    printf '%s\n' \
        'mode=import' \
        "package=$PACKAGE_VERSION" \
        "script=$SCRIPT_SHA" \
        "model=$VLM_MODEL" \
        "api_base=$VLM_API_BASE" \
        "episode_parallelism=$EPISODE_PARALLELISM" \
        "client_concurrency=$CLIENT_CONCURRENCY" \
        "private=$PUSH_PRIVATE" \
        "allow_version_tag_move=$ALLOW_VERSION_TAG_MOVE" \
        | sha256sum | cut -d' ' -f1
)"
[[ "$RECIPE_FINGERPRINT" =~ ^[0-9a-f]{64}$ ]] || {
    echo "ERROR: could not fingerprint this annotation recipe." >&2
    exit 1
}

if ! mkdir -p -- "$WORK_ROOT" "$LOG_DIR"; then
    echo "ERROR: could not create work/log directories." >&2
    exit 1
fi
if ! WORK_ROOT_CANON="$(realpath -e -- "$WORK_ROOT")" \
    || [[ -z "$WORK_ROOT_CANON" || "$WORK_ROOT_CANON" != /* ]]
then
    echo "ERROR: could not resolve WORK_ROOT to an absolute path: $WORK_ROOT" >&2
    exit 2
fi
WORK_ROOT="$WORK_ROOT_CANON"
HOME_REAL="$(realpath -m -- "$HOME")"
if [[ "$WORK_ROOT" == "/" || "$WORK_ROOT" == "$HOME_REAL" ]]; then
    echo "ERROR: refusing unsafe WORK_ROOT: $WORK_ROOT" >&2
    exit 2
fi

# Snapshot downloads, in-place annotation, logs, markers, and Hub destinations
# are deterministic for a given list. Serialize batch runners sharing this root
# so a duplicate or different-mode invocation cannot race those resources.
if ! exec {BATCH_LOCK_FD}<"$WORK_ROOT" || ! flock -n "$BATCH_LOCK_FD"; then
    echo "ERROR: another lerobot-align batch is already using WORK_ROOT: $WORK_ROOT" >&2
    exit 1
fi

safe_remove_tree() {
    local target resolved
    target="$1"
    resolved="$(realpath -m -- "$target")"
    if [[ "$resolved" == "/" || "$resolved" == "$HOME_REAL" || "$resolved" != "$WORK_ROOT/"* ]]; then
        echo "ERROR: refusing to remove path outside WORK_ROOT: $resolved" >&2
        return 1
    fi
    rm -rf -- "$resolved"
}

resolve_revision() {
    "$PY" - "$1" "$SOURCE_REVISION" <<'PY'
import sys
from huggingface_hub import HfApi

info = HfApi().dataset_info(repo_id=sys.argv[1], revision=sys.argv[2])
if not info.sha:
    raise RuntimeError("Hub response did not include a commit SHA")
print(info.sha)
PY
}

download_snapshot() {
    "$PY" - "$1" "$2" "$3" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=sys.argv[1],
    repo_type="dataset",
    revision=sys.argv[2],
    local_dir=sys.argv[3],
    max_workers=8,
)
PY
}

read_version() {
    "$PY" - "$1" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.load(stream)["codebase_version"])
PY
}

served="$(curl -fsS --max-time 10 "$VLM_API_BASE/models" 2>/dev/null \
    | "$PY" -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])' 2>/dev/null || true)"
if [[ "$served" != "$VLM_MODEL" ]]; then
    echo "ERROR: $VLM_API_BASE serves '${served:-<unreachable>}', expected '$VLM_MODEL'." >&2
    exit 1
fi
echo "VLM ready: $served at $VLM_API_BASE"

TOTAL=0
OK=0
FAILED=()

while IFS= read -r REPO || [[ -n "$REPO" ]]; do
    REPO="${REPO%$'\r'}"
    [[ -z "$REPO" || "$REPO" == \#* ]] && continue
    TOTAL=$((TOTAL + 1))

    if ! valid_repo_id "$REPO"; then
        echo "ERROR: invalid repository ID on item $TOTAL: $REPO" >&2
        FAILED+=("item-$TOTAL:invalid-repo")
        continue
    fi

    HASH="$(printf '%s' "$REPO" | sha256sum | cut -c1-10)"
    PREFIX="${REPO//\//-}"
    TRUNCATED_PREFIX="${PREFIX:0:$((MAX_KEY_LEN - 11))}"
    # Avoid creating a forbidden "--" when truncation happens immediately
    # after an interior hyphen in a long, otherwise valid source ID.
    TRUNCATED_PREFIX="${TRUNCATED_PREFIX%-}"
    KEY="${TRUNCATED_PREFIX}-${HASH}"
    DST="$DST_OWNER/$KEY"
    WORK="$WORK_ROOT/$KEY.import"
    LOG="$LOG_DIR/$KEY.log"
    DONE="$WORK_ROOT/$KEY.import.done"

    valid_repo_id "$DST" || { echo "ERROR: generated invalid destination ID: $DST" >&2; exit 2; }

    if [[ "$DST" == "$REPO" ]]; then
        echo "ERROR: destination would overwrite source repository: $REPO" >&2
        FAILED+=("$KEY:source-equals-destination")
        continue
    fi

    echo
    echo "============================================================"
    echo " [$TOTAL] $REPO@$SOURCE_REVISION -> $DST"
    echo "============================================================"

    if ! SHA="$(resolve_revision "$REPO")"; then
        echo "  ERROR: could not resolve source revision" >&2
        FAILED+=("$KEY:resolve")
        continue
    fi

    if [[ -f "$DONE" ]] \
        && grep -Fxq "repo=$REPO" "$DONE" \
        && grep -Fxq "revision=$SHA" "$DONE" \
        && grep -Fxq "dst=$DST" "$DONE" \
        && grep -Fxq "mode=import" "$DONE" \
        && grep -Fxq "recipe=$RECIPE_FINGERPRINT" "$DONE"
    then
        echo "  already completed from immutable revision $SHA"
        OK=$((OK + 1))
        continue
    fi

    safe_remove_tree "$WORK" || exit 2
    echo "  downloading immutable revision $SHA"
    if ! download_snapshot "$REPO" "$SHA" "$WORK"; then
        echo "  ERROR: download failed" >&2
        FAILED+=("$KEY:download")
        safe_remove_tree "$WORK" || true
        continue
    fi
    if [[ ! -d "$WORK/data" || ! -f "$WORK/meta/info.json" ]]; then
        echo "  ERROR: downloaded snapshot is incomplete" >&2
        FAILED+=("$KEY:incomplete-snapshot")
        safe_remove_tree "$WORK" || true
        continue
    fi
    # snapshot_download(local_dir=...) stores transfer metadata here; it is
    # local cache state, not dataset content, and must not be uploaded.
    safe_remove_tree "$WORK/.cache" || exit 2

    if ! VERSION="$(read_version "$WORK/meta/info.json")"; then
        echo "  ERROR: cannot read codebase_version" >&2
        FAILED+=("$KEY:version")
        continue
    fi
    echo "  codebase_version: $VERSION"
    if [[ ! "$VERSION" =~ ^v3\.[0-9]+([.][0-9]+)*$ ]]; then
        echo "  ERROR: unsupported dataset codebase_version: $VERSION" >&2
        echo "  Convert the source to LeRobot v3.x (for v2.1, use LeRobot's v2.1-to-v3.0 converter) and retry." >&2
        FAILED+=("$KEY:unsupported-version")
        continue
    fi

    if ! : >"$LOG"; then
        echo "  ERROR: cannot create log: $LOG" >&2
        FAILED+=("$KEY:log")
        continue
    fi
    if ! chmod 600 "$LOG"; then
        echo "  ERROR: cannot secure log permissions: $LOG" >&2
        FAILED+=("$KEY:log-permissions")
        continue
    fi
    echo "  annotating (log: $LOG)"
    START=$SECONDS
    "$ANNOTATE" \
        --root="$WORK" \
        --new_repo_id="$DST" \
        --push_to_hub=true \
        --push_private="$PUSH_PRIVATE" \
        --allow_version_tag_move="$ALLOW_VERSION_TAG_MOVE" \
        --push_commit_message="Import human subtasks and generate plan/memory/task augmentation" \
        --job.target=local \
        --vlm.model_id="$VLM_MODEL" \
        --vlm.api_base="$VLM_API_BASE" \
        --vlm.api_key=EMPTY \
        --vlm.auto_serve=false \
        '--vlm.chat_template_kwargs={"enable_thinking": false}' \
        --vlm.max_new_tokens=4096 \
        --vlm.client_concurrency="$CLIENT_CONCURRENCY" \
        --executor.episode_parallelism="$EPISODE_PARALLELISM" \
        --plan.enabled=true \
        --plan.subtask_import=lerobot_annotations \
        --plan.emit_plan=true \
        --plan.emit_memory=true \
        --plan.n_task_rephrasings=10 \
        --plan.derive_task_from_video=if_short \
        --interjections.enabled=false \
        --vqa.enabled=false \
        >>"$LOG" 2>&1
    STATUS=$?
    ELAPSED=$((SECONDS - START))

    if (( STATUS != 0 )); then
        echo "  FAILED (exit $STATUS) after ${ELAPSED}s; see $LOG" >&2
        tail -20 "$LOG" >&2
        FAILED+=("$KEY:exit$STATUS")
        continue
    fi

    VLINE="$(grep -a "validator:" "$LOG" | tail -1 || true)"
    IMPORTED="$(grep -ac "imported .* subtask(s) from 'lerobot_annotations'" "$LOG" || true)"
    GENERATED="$(grep -ac "no recorded subtasks found\|found nothing to import" "$LOG" || true)"
    echo "  done in ${ELAPSED}s | ${VLINE:-validator output missing}"
    echo "  imported episodes: $IMPORTED | episodes without imported labels: $GENERATED"

    if [[ ! "$VLINE" =~ (^|[[:space:]])errors=0($|[[:space:]]) ]]; then
        echo "  FAILED: validator did not report errors=0" >&2
        FAILED+=("$KEY:validator")
        continue
    fi

    if ! DONE_TMP="$(mktemp "$WORK_ROOT/.$KEY.import.done.XXXXXX")"; then
        echo "  FAILED: could not create completion marker" >&2
        FAILED+=("$KEY:done-marker")
        continue
    fi
    if ! printf 'repo=%s\nrevision=%s\ndst=%s\nelapsed=%s\nmode=import\npackage=%s\nrecipe=%s\nimported_episodes=%s\n%s\n' \
        "$REPO" "$SHA" "$DST" "$ELAPSED" "$PACKAGE_VERSION" \
        "$RECIPE_FINGERPRINT" "$IMPORTED" "$VLINE" >"$DONE_TMP" \
        || ! mv -f -- "$DONE_TMP" "$DONE"
    then
        echo "  FAILED: could not publish completion marker" >&2
        rm -f -- "$DONE_TMP"
        FAILED+=("$KEY:done-marker")
        continue
    fi
    OK=$((OK + 1))
    echo "  pushed: https://huggingface.co/datasets/$DST"

    if [[ "$KEEP_WORK" == "0" ]]; then
        safe_remove_tree "$WORK" || exit 2
    fi
done <"$REPO_LIST"

echo
echo "============================================================"
echo " batch complete: $OK/$TOTAL succeeded"
if (( ${#FAILED[@]} > 0 )); then
    echo " failed: ${FAILED[*]}"
fi
echo "============================================================"

(( ${#FAILED[@]} == 0 ))
