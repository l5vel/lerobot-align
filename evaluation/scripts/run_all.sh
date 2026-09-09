#!/usr/bin/env bash
# Reproduce the whole evaluation from a bare checkout, given a serving endpoint.
#
# Stages are idempotent and resumable: every one checks for its own output and
# skips if the cached result is still valid, so an interrupted sweep is resumed
# by re-running this script with the same arguments.
#
#   ./run_all.sh prepare     download, strip, extract ground truth, make splits
#   ./run_all.sh preflight   integrity assertions -- MUST pass before any GPU work
#   ./run_all.sh serve       start vLLM replicas on the allowed GPUs
#   ./run_all.sh falsify     E1: align_defaults must equal baseline_upstream
#   ./run_all.sh pilot       E2: small run, hand-inspected
#   ./run_all.sh full        E3-E8: the full sweep
#   ./run_all.sh analyse     aggregate saved scores + gates + tables (CPU only)
#
# STUDY 2 -- fixed-label alignment (alignment_plan.md). The caller supplies the
# ordered label list and the VLM decides only WHEN each label happens. Upstream
# has no such mode at all (§5.3), so these stages compare CONFIGURATIONS OF THIS
# TOOL against each other and against model-free timing templates.
#
#   ./run_all.sh prepare-labels     per-episode label files from the ground truth
#   ./run_all.sh floors             the three model-free arms of §5.1 (CPU only)
#   ./run_all.sh calibrate          fit timing offsets per component, SEED only
#   ./run_all.sh align-full         the VLM alignment arms over the eval split
#   ./run_all.sh align-repeats      extra passes on one component: decoding noise
#   ./run_all.sh analyse-alignment  score + report the alignment study (CPU only)
#
# The alignment stages deliberately do NOT consult the C1 gate. C1 is an
# equivalence check between the two tools in GENERATION mode; upstream cannot do
# fixed-label alignment at all, so no C1 verdict can be certified for these arms
# and requiring one would block them on a result about a different experiment.
#
# THE MODEL-FREE SMOKE PATH. `prepare-labels` + `floors` + `analyse-alignment`
# exercises the entire data path -- labels, predictions, scoring, aggregation,
# report -- with no VLM call, no GPU and no replica, because the floors never
# open a video. Two environment variables make that a first-class run rather
# than something to be improvised:
#
#   ALIGN_DATASETS='ds-a ds-b'   restrict the STUDY 2 stages to these components.
#                                Does not touch $DATASETS, so study 1 and
#                                `stage_prepare`'s 16-component assertion are
#                                unaffected. An unknown name is fatal.
#   ALIGN_ALLOW_INCOMPLETE=1     tolerate aggregate.py's family-shortfall exit
#                                (3, and ONLY 3) and pass --allow-incomplete to
#                                the report generator, which stamps its output
#                                INCOMPLETE DRAFT -- NOT A RESULT and still exits
#                                non-zero. Without it the stage fails closed, as
#                                it must: a run of the floors alone computes 0 of
#                                the 24 pre-registered tests, since contrasts 7
#                                and 8 have no VLM numerator.
#
#   ALIGN_DATASETS='u850-bag-place-03-BC-v30 base4-clean-table-01-BC-FV-v30' \
#   ALIGN_ALLOW_INCOMPLETE=1 ./run_all.sh prepare-labels && ... floors && \
#       ... analyse-alignment
#
# GPU POLICY: GPU 4 is excluded everywhere in this file and must stay excluded.
# The allowed set is declared once, here, and never recomputed.

set -euo pipefail

# Historical fixed-label stages used a different Corpus A protocol. Keep them
# available for explicit reproduction only; new A/B studies share one driver.
case "${1:-}" in
    prepare-labels|floors|calibrate|align-full|align-repeats|analyse-alignment)
        if [[ "${ALIGN_LEGACY_PROTOCOL:-0}" != "1" ]]; then
            echo "Use prepare_alignment_study.py and evaluate_alignment_study.py for both corpora." >&2
            echo "See evaluation/calibration_protocol.md. Historical reproduction only: ALIGN_LEGACY_PROTOCOL=1." >&2
            exit 2
        fi
        ;;
esac

ALLOWED_GPUS="${ALLOWED_GPUS:-0,1,2,3,5,6,7}"
FORBIDDEN_GPU=4

# Normalise whitespace before validating whole numeric tokens. Refuse aliases,
# malformed lists and GPU 04 as well as GPU 4; never pass unchecked input on.
ALLOWED_GPUS="${ALLOWED_GPUS//[[:space:]]/}"
if [[ ! "$ALLOWED_GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "FATAL: invalid ALLOWED_GPUS='$ALLOWED_GPUS'" >&2
    exit 1
fi
IFS=',' read -ra checked_gpus <<< "$ALLOWED_GPUS"
for gpu in "${checked_gpus[@]}"; do
    if (( 10#$gpu == FORBIDDEN_GPU )); then
        echo "FATAL: GPU $FORBIDDEN_GPU appears in ALLOWED_GPUS='$ALLOWED_GPUS'" >&2
        exit 1
    fi
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_DIR="$(dirname "$HERE")"
REPO_DIR="$(dirname "$EVAL_DIR")"
PYTHON="${PYTHON:-$REPO_DIR/.venv/bin/python}"

STORAGE="${EVAL_STORAGE:-$REPO_DIR/.cache/lerobot-align-eval}"
GT_DIR="$STORAGE/ground_truth"
SPLIT_DIR="$STORAGE/splits"
WORK_DIR="$STORAGE/work"
PRED_DIR="$STORAGE/predictions"
RESULT_DIR="$EVAL_DIR/results"

# ---------------------------------------------------------------------------
# Study 2: fixed-label alignment. Paths, and the interfaces this script assumes
# of the scripts and configs it drives. They are written down because they are
# the seam between agents: a mismatch here fails loudly at the top of a stage
# rather than three hours into a sweep.
#
#   make_label_files.py     --gt-dir --splits-dir --out-dir [--datasets ...]
#                           corpus-wide; writes <out-dir>/<dataset>__<source>.json
#                           for source in {oracle, modal}. --verify re-reads them
#                           through the TOOL's loader and asserts every eval
#                           episode resolves to its own non-empty list.
#   align_floors.py         --gt --split --labels (files or directories)
#                           --label-source --out-dir [--self-test --summary-out]
#                           writes <out-dir>/<dataset>__<arm>.json for
#                           floor_uniform, floor_median_ratio,
#                           floor_cumulative_prior, in run_arm.py's record shape.
#   score_alignment.py      --predictions-dir --gt-dir --splits-dir --out
#                           index-based matching (D10). Emits exactly these
#                           metric names: placed_fraction, b_hit@1, b_hit@3,
#                           b_hit@5, boundary_mae_placed, boundary_mae_norm,
#                           macro_temporal_iou, calibration_applied. It resolves
#                           each arm's labels from the arm's OWN recorded
#                           subtasks_path, so nothing here has to tell it twice.
#   aggregate.py            --scores --out --matcher any --contrasts
#                           --arms-config [--require-true FIELD]
#                           --contrasts switches it into contrast-list mode, which
#                           is what makes Holm correct ONE family (D13).
#   make_alignment_report.py --results --out; every input defaults to
#                           <results>/alignment_{all,modal,gates,calibration}.json,
#                           so the stages write the files and the report finds them.
#   configs/alignment_arms.yaml      arms.yaml's schema. Every VLM arm declares
#                           plan.subtasks_path: '{subtasks_path}'; the calibrated
#                           ones also plan.subtask_align_calibration_path:
#                           '{calibration_path}'. Model-free floors carry
#                           tool: reference so run_arm.py and arms_matching skip
#                           them. supervision is labels_only or labels_plus_seed.
#   configs/alignment_contrasts.yaml the §8 family, so Holm runs in ONE pass.
ALIGN_ARMS_CONFIG="${ALIGN_ARMS_CONFIG:-$EVAL_DIR/configs/alignment_arms.yaml}"
ALIGN_CONTRASTS_CONFIG="${ALIGN_CONTRASTS_CONFIG:-$EVAL_DIR/configs/alignment_contrasts.yaml}"

# oracle_list IS the study (§4.2): every episode keeps its own label list and
# loses only its boundaries. modal_list is the same plumbing pointed at a
# different question and is out of scope here, so it is switchable rather than
# wired in.
ALIGN_LABEL_SOURCE="${ALIGN_LABEL_SOURCE:-oracle}"
LABEL_DIR="${ALIGN_LABEL_DIR:-$STORAGE/labels}"
CAL_DIR="${ALIGN_CALIBRATION_DIR:-$STORAGE/calibration}"
# Alignment predictions are kept OUT of $PRED_DIR. Both scorers glob a whole
# directory, so one study's spans landing in the other's scorer would be a
# silent cross-study contamination -- and the two studies do not even share a
# metric set. The label source is in the path for the same reason: switching it
# must not overwrite predictions made against different labels.
ALIGN_PRED_DIR="${ALIGN_PREDICTIONS_DIR:-$STORAGE/predictions_alignment/$ALIGN_LABEL_SOURCE}"
ALIGN_REPEAT_DIR="${ALIGN_REPEAT_PREDICTIONS_DIR:-$STORAGE/predictions_alignment/${ALIGN_LABEL_SOURCE}_repeats}"
# §12.8: temperature is 0.2 and the request carries no seed, so the server
# samples freely and repeats are mandatory, not optional. Two extra passes on
# one component sizes the decoding-noise band the null claims are read against.
ALIGN_REPEATS="${ALIGN_REPEATS:-2}"
ALIGN_REPEAT_DATASET="${ALIGN_REPEAT_DATASET:-u850-fridge-drink-01-FV-v30}"

# The fit gate's thresholds are passed EXPLICITLY and then overridden with
# --force (§7.2). Letting lerobot-align-fit refuse would drop all four
# clean-table components -- 25% of the corpus -- and leave claim A3 conditioned
# on the population where calibration already works, which is precisely the
# selection error §2.1b identifies in the prior sweep. The verdict is recorded
# instead of obeyed, and becomes the prediction claim A6 tests. Stating the
# thresholds here rather than inheriting them keeps the recorded verdict
# meaningful if the tool's defaults ever move.
ALIGN_FIT_MIN_GAIN="${ALIGN_FIT_MIN_GAIN:-3.0}"
ALIGN_FIT_MIN_FIRST_MAE="${ALIGN_FIT_MIN_FIRST_MAE:-1.5}"
# §7.4.8 requires fit_episode_count >= 5. Measured over Corpus A the pinned
# cohorts are 5..10 episodes, so this bound is live on two components
# (mobile-door-01 and fridge-drink-03, 5 each), not decorative.
ALIGN_FIT_MIN_COHORT="${ALIGN_FIT_MIN_COHORT:-5}"

# Which arms consume which per-component file, read from the arm definitions
# rather than guessed from arm names. Populated by load_arm_requirements; empty
# for the generation stages, which is what keeps dispatch() unchanged for them.
declare -A ARM_NEEDS_SUBTASKS=()
declare -A ARM_NEEDS_CALIBRATION=()

# Which arms config a job came from, and where its predictions land. run_arm.py
# defaults to configs/arms.yaml and would exit "unknown arm" on the first
# alignment job, so the file is always passed explicitly.
ARMS_CONFIG="${ARMS_CONFIG:-$EVAL_DIR/configs/arms.yaml}"
DISPATCH_PRED_DIR="$PRED_DIR"

BASE_PORT="${QWEN38_BASE_PORT:-8000}"
MODEL_ID="${QWEN38_SERVED_MODEL_NAME:-Qwen/Qwen3.8-27B}"

# Corpus A: the 16 source components that carry the imported reference layer.
# Do not substitute the 10 merged L5vel repositories: they combine imported
# reference and generated components and do not publish the historical source
# annotation sidecars used by this driver. See the immutable source manifest.
SOURCE_DIR="${EVAL_SOURCE_DIR:-$REPO_DIR/.cache/corpus-a-sources}"

DATASETS=(
    base4-clean-table-01-BC-FV-v30
    base4-clean-table-02-BC-v30
    base4-clean-table-03-BC-FV-v30
    base4-clean-table-04-BC-FV-v30
    base4-mobile-door-01-BC-FV-v30
    base4-mobile-door-02-BC-FV-v30
    base4-mobile-door-04-BC-FV-v30
    u850-bag-place-01-FV-v30
    u850-bag-place-02-FV-v30
    u850-bag-place-03-BC-v30
    u850-bag-place-04-BC-v30
    u850-fridge-drink-01-FV-v30
    u850-fridge-drink-02-v30
    u850-fridge-drink-03-v30
    u850-fridge-drink-04-v30
    u850-fridge-drink-05-v30
)
EXPECTED_COMPONENTS=16

# ALIGN_DATASETS narrows the STUDY 2 stages to a subset of the components above,
# space separated. It exists for the model-free smoke path -- `prepare-labels`,
# `floors`, `analyse-alignment` with no VLM arm run -- which exercises the whole
# data path on two components in seconds and needs no GPU at all.
#
# It deliberately does NOT touch $DATASETS, so `stage_prepare`'s
# EXPECTED_COMPONENTS=16 assertion and every study 1 stage still see the full
# corpus. A subset is a property of one alignment run, never of the corpus.
# Names are validated against the ground truth by the scripts themselves
# (`--datasets` is fatal on an unknown name), so a typo cannot quietly score a
# smaller population.
ALIGN_DATASETS_LIST=()
if [[ -n "${ALIGN_DATASETS:-}" ]]; then
    read -r -a ALIGN_DATASETS_LIST <<< "$ALIGN_DATASETS"
fi

# Emit `--datasets a b c` when a subset is asked for, and nothing at all when it
# is not, so the default command line is byte-identical to what it was before.
align_dataset_flags() {
    if (( ${#ALIGN_DATASETS_LIST[@]} )); then
        printf '%s\n' --datasets "${ALIGN_DATASETS_LIST[@]}"
    fi
}

# The components an alignment stage should act on: the subset when one is set,
# the whole corpus otherwise.
align_datasets() {
    if (( ${#ALIGN_DATASETS_LIST[@]} )); then
        printf '%s\n' "${ALIGN_DATASETS_LIST[@]}"
    else
        printf '%s\n' "${DATASETS[@]}"
    fi
}

# Arm names are derived from arms.yaml, never hardcoded: the factorial renames
# every arm to <tool>__<profile>__<camera>, and a stale list here would ask for
# arms that no longer exist.
arms_matching() {  # $1 = optional regex filter
    "$PYTHON" - "$ARMS_CONFIG" "${1:-}" <<'PYARMS'
import re, sys, yaml
spec = yaml.safe_load(open(sys.argv[1]))
pattern = re.compile(sys.argv[2])
names = [a['name'] for a in spec['arms']
         if a['tool'] != 'reference' and pattern.search(a['name'])]
if not names:
    raise SystemExit(f"FATAL: ARM_FILTER={sys.argv[2]!r} matches no arms")
print('\n'.join(names))
PYARMS
}

# The cell used for the C1 gate. One cell is enough to establish that the two
# tools agree when the new features are off; running the gate in every cell
# would cost 5x for the same conclusion.
E1_PROFILE="${E1_PROFILE:-wrap}"
E1_CAMERA="${E1_CAMERA:-wrist}"
E1_BASELINE="baseline_upstream__${E1_PROFILE}__${E1_CAMERA}"
E1_ARM="align_defaults__${E1_PROFILE}__${E1_CAMERA}"
E1_REPEATS="${E1_REPEATS:-3}"
# One gate file PER CELL. A single c1_gate.json meant certifying a second cell
# silently destroyed the first cell's verdict, and left `full` in another cell
# reading a verdict that was never about it. The unqualified name is kept as a
# symlink to the primary cell so the report and older invocations still resolve.
C1_GATE="$RESULT_DIR/c1_gate__${E1_PROFILE}__${E1_CAMERA}.json"
C1_PRIMARY_PROFILE="${C1_PRIMARY_PROFILE:-wrap}"
C1_PRIMARY_CAMERA="${C1_PRIMARY_CAMERA:-wrist}"

stage_prepare() {
    mkdir -p "$GT_DIR" "$SPLIT_DIR" "$WORK_DIR" "$PRED_DIR" "$RESULT_DIR"
    local accepted=0
    for dataset in "${DATASETS[@]}"; do
        local source="$SOURCE_DIR/$dataset"
        if [[ ! -d "$source" ]]; then
            echo "FATAL: $source not found. Corpus A is local-only; set EVAL_SOURCE_DIR." >&2
            exit 1
        fi
        if [[ ! -f "$GT_DIR/$dataset.json" ]]; then
            # prepare_dataset.py exits 2 when provenance fails. Under `set -e`
            # that would abort silently mid-sweep, so the refusal is caught and
            # reported as the hard error it is.
            if ! "$PYTHON" "$HERE/prepare_dataset.py" "$source" "$WORK_DIR/_probe_$dataset" \
                    --gt-out "$GT_DIR/$dataset.json" --dataset-name "$dataset" --overwrite; then
                echo "FATAL: $dataset failed the human-provenance gate." >&2
                exit 1
            fi
            rm -rf "$WORK_DIR/_probe_$dataset"
        fi
        [[ -f "$SPLIT_DIR/$dataset.json" ]] || "$PYTHON" "$HERE/make_splits.py" \
            --gt "$GT_DIR/$dataset.json" --out "$SPLIT_DIR/$dataset.json"
        accepted=$((accepted + 1))
    done
    if (( accepted != EXPECTED_COMPONENTS )); then
        echo "FATAL: prepared $accepted components, expected $EXPECTED_COMPONENTS" >&2
        exit 1
    fi
    "$PYTHON" "$HERE/calibrate_matcher.py" --gt-dir "$GT_DIR" \
        --out "$RESULT_DIR/matcher_calibration.json" --backend embedding
    echo "[prepare] $accepted components ready; model-free floor:"
    for dataset in "${DATASETS[@]}"; do
        "$PYTHON" "$HERE/reference_arms.py" --gt "$GT_DIR/$dataset.json" \
            --split "$SPLIT_DIR/$dataset.json" --out-dir "$PRED_DIR" >/dev/null
    done
}

stage_preflight() {
    "$PYTHON" "$HERE/preflight.py" --gt-dir "$GT_DIR" --splits-dir "$SPLIT_DIR"
}

stage_serve() {
    echo "[serve] launching replicas on GPUs $ALLOWED_GPUS (GPU $FORBIDDEN_GPU excluded)"
    "$REPO_DIR/scripts/start_qwen38.sh" --gpus "$ALLOWED_GPUS"
}

# One replica port per allowed GPU, in the order the replicas were started.
replica_ports() {
    local -a gpu_list=()
    local index
    IFS=',' read -ra gpu_list <<< "$ALLOWED_GPUS"
    for index in "${!gpu_list[@]}"; do printf '%s\n' "$((BASE_PORT + index))"; done
}

# A replica must be held by ONE job at a time. The original round-robin
# (`ports[index % n_ports]`) assigned by position, not availability: when an
# early job outlasted a later one, its port was handed out again while it was
# still serving. That silently shared a replica, which contaminates both the
# per-job wall clock and the per-job token counters read from that replica. A
# FIFO used as a token pool makes the port itself the lease: a job cannot start
# until it has one, and returns it on exit.
#
# Extracted from dispatch() verbatim so that the calibration fits, which are
# also GPU jobs whose runtime we care about, lease replicas under exactly this
# discipline instead of growing a second, subtly different scheduler.
open_replica_pool() {  # $@ = the ports to seed the pool with
    local pool port
    pool="$(mktemp -u)"
    mkfifo "$pool"
    exec 9<>"$pool"
    rm -f "$pool"
    for port in "$@"; do printf '%s\n' "$port" >&9; done
}

close_replica_pool() {
    exec 9>&-
}

# Distribute (dataset, arm) jobs across replicas, one job per replica at a time.
# Parallelism comes from independent processes, never from threads inside one,
# so a crash costs one job and the timings stay clean.
dispatch() {
    local -a jobs=("$@")
    local -a ports=()
    mapfile -t ports < <(replica_ports)
    local -a pids=()
    local failed=0
    local port
    open_replica_pool "${ports[@]}"
    # `set -e` would abort the whole sweep on the first non-zero `wait -n`,
    # losing every completed job and never printing the summary. Failures are
    # counted and reported instead: one arm crashing on one component is data
    # (it becomes that arm's failure rate), not a reason to discard the run.
    set +e
    for job in "${jobs[@]}"; do
        IFS='|' read -r dataset arm repeat <<< "$job"
        repeat="${repeat:-0}"
        # Blocks until a replica is free; that read IS the lease.
        read -r port <&9
        local tag="${dataset}__${arm}"
        (( repeat > 0 )) && tag="${tag}__r${repeat}"
        # The per-component files a fixed-label arm needs, offered only to the
        # arms whose own flags declare them (ARM_NEEDS_*, filled in by
        # load_arm_requirements). Both directions matter: without them no
        # fixed-label arm and neither calibrated arm can launch at all (D2),
        # and a file passed to an arm that does not consume it is worse than
        # useless -- run_arm.py refuses a label file no flag reads, because the
        # arm would then GENERATE labels while the cache key and the provenance
        # both claimed the supplied list was in use (§7.4.6).
        local -a supplied=()
        if [[ -n "${ARM_NEEDS_SUBTASKS[$arm]:-}" ]]; then
            supplied+=(--subtasks-path "$(align_label_file "$dataset")")
        fi
        if [[ -n "${ARM_NEEDS_CALIBRATION[$arm]:-}" ]]; then
            supplied+=(--calibration-path "$(align_calibration_file "$arm" "$dataset")")
        fi
        local -a command=(
            "$PYTHON" "$HERE/run_arm.py"
            --dataset "$dataset" --source-root "$SOURCE_DIR/$dataset"
            --arm "$arm" --arms-config "$ARMS_CONFIG"
            --work-dir "$WORK_DIR" --repeat "$repeat"
            --out "$DISPATCH_PRED_DIR/${tag}.json"
            --episodes "$SPLIT_DIR/$dataset.json"
            --base-url "http://127.0.0.1:${port}/v1"
            --model-id "$MODEL_ID" --python "$PYTHON"
            ${supplied[@]+"${supplied[@]}"}
        )
        (
            if [[ "${DISPATCH_DRY_RUN:-0}" == "1" ]]; then
                # Print the job instead of running it, while still taking and
                # returning the lease, so the argument construction -- which
                # per-component files an arm is handed -- can be inspected
                # without spending GPU time or perturbing the pool.
                local rendered
                printf -v rendered ' %q' "${command[@]}"
                printf '[dry-run]%s\n' "$rendered"
                status=0
            else
                "${command[@]}"
                status=$?
            fi
            # Return the lease no matter how the job ended, or the pool drains
            # and the sweep deadlocks.
            printf '%s\n' "$port" >&9
            exit $status
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
    close_replica_pool
    set -e
    echo "[dispatch] ${#jobs[@]} jobs, $failed failed"
    if (( failed > 0 )); then
        echo "[dispatch] NOTE: failures are retained as per-arm failure rates, not dropped."
    fi
}

stage_falsify() {
    # Repeats are the point: neither tool seeds its generation call, so the
    # same arm run twice differs. Without the resulting noise band, a null C1
    # is uninterpretable and so is every C2/C3 effect size.
    local -a jobs=()
    for dataset in base4-mobile-door-01-BC-FV-v30 u850-fridge-drink-01-FV-v30 u850-bag-place-01-FV-v30; do
        for r in $(seq 1 "$E1_REPEATS"); do
            jobs+=("$dataset|$E1_BASELINE|$r" "$dataset|$E1_ARM|$r")
        done
    done
    dispatch "${jobs[@]}"
    local threshold
    threshold="$("$PYTHON" -c "
import json; print(json.load(open('$RESULT_DIR/matcher_calibration.json'))['selected_threshold'])
")"
    "$PYTHON" "$HERE/score_runs.py" --predictions-dir "$PRED_DIR" --gt-dir "$GT_DIR" \
        --splits-dir "$SPLIT_DIR" --out "$RESULT_DIR/e1_scores.jsonl" \
        --embedding-threshold "$threshold"
    # Decides C1 by equivalence testing against the measured noise band, and
    # returns non-zero unless it concludes EQUIVALENT. `full` must not run
    # until this passes.
    "$PYTHON" "$HERE/c1_gate.py" --scores "$RESULT_DIR/e1_scores.jsonl" \
        --out "$C1_GATE" --model-id "$MODEL_ID" --threshold "$threshold" \
        --baseline "$E1_BASELINE" --arm "$E1_ARM"
    # The report quotes ONE gate, and it must be the primary cell's -- not
    # whichever cell was certified most recently.
    if [[ "$E1_PROFILE" == "$C1_PRIMARY_PROFILE" && "$E1_CAMERA" == "$C1_PRIMARY_CAMERA" ]]; then
        cp -f "$C1_GATE" "$RESULT_DIR/c1_gate.json"
    fi
}

stage_pilot() {
    local -a jobs=()
    local -a arms=()
    local selected
    selected="$(arms_matching "__${E1_PROFILE}__${E1_CAMERA}$")" || return 1
    mapfile -t arms <<< "$selected"
    for dataset in u850-fridge-drink-01-FV-v30 base4-clean-table-01-BC-FV-v30; do
        for arm in "${arms[@]}"; do jobs+=("$dataset|$arm"); done
    done
    dispatch "${jobs[@]}"
}

# The C1 gate only gates anything if something reads it. `full` spends hours of
# GPU, so it refuses to start unless C1 has actually been decided and passed.
require_c1_pass() {
    if [[ "${SKIP_C1_GATE:-0}" == "1" ]]; then
        echo "WARNING: SKIP_C1_GATE=1 -- running without a C1 verdict. Any difference" >&2
        echo "         this produces may be an uncontrolled confound, not a mechanism." >&2
        return 0
    fi
    if [[ ! -f "$C1_GATE" ]]; then
        echo "FATAL: no $C1_GATE. Run './run_all.sh falsify' first" >&2
        echo "       with E1_PROFILE=$E1_PROFILE E1_CAMERA=$E1_CAMERA." >&2
        echo "       C1 establishes that the two tools agree when the new features are" >&2
        echo "       off. Without it, every downstream number is uninterpretable." >&2
        exit 1
    fi
    local gate verdict
    gate="$("$PYTHON" -c "
import json; d=json.load(open('$C1_GATE')); print(d['gate'])")"
    verdict="$("$PYTHON" -c "
import json; d=json.load(open('$C1_GATE')); print(d['verdict'])")"
    # A PASS certified against different arm flags, different tool source, or a
    # different model is not a PASS for this sweep. Verify it describes the
    # configuration we are about to run, not merely that it says the right word.
    local threshold_arg=()
    if [[ -f "$RESULT_DIR/matcher_calibration.json" ]]; then
        threshold_arg=(--threshold "$("$PYTHON" -c "
import json; print(json.load(open('$RESULT_DIR/matcher_calibration.json'))['selected_threshold'])")")
    fi
    # The fingerprint records WHICH arms the verdict was certified for. In the
    # factorial those are cell-qualified names, so the check must be given the
    # same pair -- otherwise it compares against the bare defaults and reports a
    # spurious staleness for a verdict that is in fact current.
    if ! "$PYTHON" "$HERE/gate_fingerprint.py" --check "$C1_GATE" \
            --model-id "$MODEL_ID" --baseline "$E1_BASELINE" --arm "$E1_ARM" \
            "${threshold_arg[@]+"${threshold_arg[@]}"}"; then
        echo "FATAL: the C1 verdict is stale -- it was certified against a different" >&2
        echo "       configuration than the one about to run (see the diff above)." >&2
        echo "       Re-run './run_all.sh falsify' to re-certify." >&2
        exit 1
    fi
    if [[ "$gate" != "PASS" ]]; then
        echo "FATAL: C1 gate is $gate (verdict $verdict); refusing to spend GPU time." >&2
        if [[ "$verdict" == "DIFFERENT" ]]; then
            echo "       Two tools sharing prompts, defaults and model disagree. Find the" >&2
            echo "       confound before running anything else." >&2
        else
            echo "       Too imprecise to conclude equivalence. This is NOT a pass: add" >&2
            echo "       repeats or components, or reduce failure rates, and re-run" >&2
            echo "       './run_all.sh falsify'." >&2
        fi
        echo "       Override with SKIP_C1_GATE=1 only if you intend to report an" >&2
        echo "       uncontrolled comparison and say so." >&2
        exit 1
    fi
    echo "[full] C1 gate: $verdict -- proceeding."
}

stage_full() {
    local -a arms=()
    local selected
    selected="$(arms_matching "${ARM_FILTER:-}")" || return 1
    mapfile -t arms <<< "$selected"
    require_c1_pass
    echo "[full] ${#arms[@]} arms x ${#DATASETS[@]} components"
    local -a jobs=()
    for dataset in "${DATASETS[@]}"; do
        for arm in "${arms[@]}"; do jobs+=("$dataset|$arm"); done
    done
    dispatch "${jobs[@]}"
}

stage_analyse() {
    # Analysis must not rerun annotation or silently overwrite saved scores.
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/regenerate_reports.py" \
        --results "$RESULT_DIR" --analysis "$EVAL_DIR/analysis"
}


# ===========================================================================
# Study 2: fixed-label alignment (alignment_plan.md)
#
# The caller supplies the ordered label list; the VLM decides only WHEN each
# label happens. Nothing below reads a C1 verdict: C1 compares the two tools in
# GENERATION mode, and upstream has no fixed-label mode to be equivalent to.
# ===========================================================================

# The shipped calibration path is `lerobot-align-eval-batch` then
# `lerobot-align-fit` (§7.2). Prefer the console scripts installed beside
# $PYTHON; fall back to `python -m` so a checkout that was never installed in
# editable mode still runs exactly the same code rather than failing obscurely.
ALIGN_BIN_DIR="$(dirname "$PYTHON")"
if [[ -x "$ALIGN_BIN_DIR/lerobot-align-eval-batch" ]]; then
    ALIGN_EVAL_BATCH=("$ALIGN_BIN_DIR/lerobot-align-eval-batch")
else
    ALIGN_EVAL_BATCH=("$PYTHON" -m lerobot_align.diagnostics.eval_align_batch)
fi
if [[ -x "$ALIGN_BIN_DIR/lerobot-align-fit" ]]; then
    ALIGN_FIT=("$ALIGN_BIN_DIR/lerobot-align-fit")
else
    ALIGN_FIT=("$PYTHON" -m lerobot_align.diagnostics.fit_align_calibration)
fi

# make_label_files.py writes one flat directory: <dataset>__<source>.json.
align_label_file() {  # $1 = dataset
    printf '%s\n' "$LABEL_DIR/$1__$ALIGN_LABEL_SOURCE.json"
}

# One calibration per (arm, component), never one per component. §7.2a requires
# the fit to be made under the SCORING arm's own regime, so the single-view and
# the stacked arm cannot share a file however similar they look.
align_calibration_file() {  # $1 = arm, $2 = dataset
    printf '%s\n' "$CAL_DIR/$1/$2.json"
}

# Read from the arm definitions which arms consume a label file and which
# consume a calibration, so dispatch() offers each arm exactly what its own
# flags declare. Names are not evidence: an arm called *_cal that forgot the
# placeholder would otherwise be dispatched with a calibration it never applies.
load_arm_requirements() {
    local arm wants_subtasks wants_calibration
    ARM_NEEDS_SUBTASKS=()
    ARM_NEEDS_CALIBRATION=()
    while IFS=$'\t' read -r arm wants_subtasks wants_calibration; do
        [[ -n "$arm" ]] || continue
        if [[ "$wants_subtasks" == "1" ]]; then ARM_NEEDS_SUBTASKS["$arm"]=1; fi
        if [[ "$wants_calibration" == "1" ]]; then ARM_NEEDS_CALIBRATION["$arm"]=1; fi
    done < <("$PYTHON" - "$ARMS_CONFIG" <<'PYREQ'
import sys, yaml
spec = yaml.safe_load(open(sys.argv[1], encoding="utf-8"))
for arm in spec["arms"]:
    rendered = " ".join(str(v) for v in (arm.get("flags") or {}).values())
    print(arm["name"], int("{subtasks_path}" in rendered),
          int("{calibration_path}" in rendered), sep="\t")
PYREQ
    )
}

# Point the shared machinery at the alignment study. Every alignment stage calls
# this first: it is the only place that switches the arms config, the
# predictions directory and the per-arm file requirements together, so they
# cannot drift apart.
use_alignment_configuration() {
    if [[ ! -f "$ALIGN_ARMS_CONFIG" ]]; then
        echo "FATAL: $ALIGN_ARMS_CONFIG not found." >&2
        echo "       configs/arms.yaml contains ZERO alignment arms, so run_arm.py would" >&2
        echo "       exit 'unknown arm' on the first job (alignment_plan.md §10.2)." >&2
        exit 1
    fi
    ARMS_CONFIG="$ALIGN_ARMS_CONFIG"
    DISPATCH_PRED_DIR="$ALIGN_PRED_DIR"
    load_arm_requirements
}

# Stage A gates 1 and 2 (§7.4), before any GPU time is spent.
#
# Delegated to make_label_files.py --verify rather than re-implemented here,
# because that check re-reads the files through the TOOL's own loader -- and the
# tool's reading is the only one that decides whether an episode gets aligned or
# silently generated. Gate 2 is the one that would otherwise be invisible:
# _load_subtasks_file returns [] for an episode absent from an object-keyed file
# with no "default", and _subtask_spans then generates instead of aligning. That
# arm still produces spans, still scores, and still gets reported -- as a
# fixed-label result it never was.
require_label_coverage() {
    require_scripts make_label_files.py
    local -a components=()
    mapfile -t components < <(align_datasets)
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/make_label_files.py" \
        --gt-dir "$GT_DIR" --splits-dir "$SPLIT_DIR" --out-dir "$LABEL_DIR" \
        --datasets "${components[@]}" --verify
}

# Pin the fit cohort explicitly (D7), and record it.
#
# lerobot-align-fit picks its own cohort from whatever rows it is handed, and it
# is the only supported way in; it has no --episodes flag. So the cohort is
# pinned by controlling the input: eval_align_batch is asked for exactly these
# episodes, they all carry one label list, and the fitter's selector becomes a
# no-op instead of a hidden choice. This also enforces THIS STUDY'S tie-break
# (§4.2: most frequent, then the LONGER list) rather than inheriting whatever
# rule the tool happens to use -- on fridge-drink-03 the seed episodes split 5/5
# between a 5-label and a 6-label convention, and the two rules agree there
# today only by lexicographic accident.
#
# Prints the cohort on stdout, one line, space separated; writes its provenance
# to $2; every diagnostic goes to stderr so stdout stays parseable.
align_pin_cohort() {  # $1 = dataset, $2 = cohort JSON to write
    CUDA_VISIBLE_DEVICES='' "$PYTHON" - \
        "$1" "$GT_DIR/$1.json" "$SPLIT_DIR/$1.json" \
        "$SOURCE_DIR/$1/meta/lerobot_annotations.json" "$2" "$ALIGN_FIT_MIN_COHORT" <<'PYCOHORT'
import json
import sys
from collections import Counter
from pathlib import Path

dataset, gt_path, split_path, meta_path, out_path, min_cohort = sys.argv[1:7]
min_cohort = int(min_cohort)

truth_payload = json.loads(Path(gt_path).read_text(encoding="utf-8"))
episodes = {int(key): spans for key, spans in truth_payload["episodes"].items()}
split = json.loads(Path(split_path).read_text(encoding="utf-8"))
seed = [int(episode) for episode in split["seed"]]
overlap = sorted(set(seed) & {int(e) for e in split["eval"]})
if overlap:
    # §7.4.4. Fitting on an episode that is later scored would leak its
    # boundaries into the arm that is supposed to predict them.
    raise SystemExit(f"{dataset}: seed and eval episodes overlap: {overlap}")

lists = {
    episode: tuple(str(span["text"]) for span in episodes[episode])
    for episode in seed
    if episodes.get(episode)
}
if not lists:
    raise SystemExit(f"{dataset}: no seed episode carries ground truth")

counts = Counter(lists.values())
# Most frequent wins; ties go to the LONGER list (§4.2); the final key only
# removes the remaining arbitrariness so the cohort is reproducible.
chosen = max(counts, key=lambda labels: (counts[labels], len(labels), labels))
cohort = sorted(episode for episode in lists if lists[episode] == chosen)
dropped = sorted(episode for episode in lists if lists[episode] != chosen)

if len(cohort) < min_cohort:
    raise SystemExit(
        f"{dataset}: pinned fit cohort is only {len(cohort)} episode(s) "
        f"({cohort}) out of {len(lists)} seed episodes across {len(counts)} distinct "
        f"label lists; §7.4.8 requires at least {min_cohort}. Fitting on fewer skips "
        "the tool's own weight selection and produces a calibration nobody can defend."
    )

# The fitter reads labels from meta/lerobot_annotations.json while the cohort is
# chosen from the extracted ground truth. They are two readings of the same
# annotations and they must agree, or the fitter would re-split the cohort we
# just pinned.
meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
meta_episodes = meta.get("episodes") or {}
disagree = []
for episode in cohort:
    spans = (meta_episodes.get(str(episode)) or {}).get("subtasks") or []
    if tuple(str(span["label"]) for span in spans) != chosen:
        disagree.append(episode)
if disagree:
    raise SystemExit(
        f"{dataset}: meta/lerobot_annotations.json disagrees with the extracted ground "
        f"truth for episodes {disagree}; the pinned cohort would not survive the fitter."
    )

Path(out_path).parent.mkdir(parents=True, exist_ok=True)
Path(out_path).write_text(
    json.dumps(
        {
            "dataset": dataset,
            "seed_episodes": seed,
            "distinct_seed_label_lists": len(counts),
            "tie_break": "most frequent, then longest, then lexicographic (§4.2)",
            "cohort": cohort,
            "dropped_from_cohort": dropped,
            "labels": list(chosen),
        },
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
print(
    f"[calibrate] {dataset}: cohort {len(cohort)}/{len(seed)} seed episodes, "
    f"{len(counts)} distinct label list(s), dropped {dropped}",
    file=sys.stderr,
)
print(" ".join(str(episode) for episode in cohort))
PYCOHORT
}

# Derive the fitting command's flags FROM THE ARM DEFINITION (§7.2a).
#
# The predictions a calibration is fit on come from eval_align_batch, which
# carries its own defaults -- temperature 0.0 against the arms' 0.2, frame width
# 336 against PlanConfig's 224, fps 3.0, 300 frames. Writing those numbers down
# a second time here is how a fit ends up made under one regime and applied
# under another; deriving them means the two cannot disagree, and any field the
# driver cannot carry is an error rather than a silent default.
#
# Prints one argv token per line; writes the full regime to $2 so preflight and
# the report can check the fit against the arm field by field.
align_fit_flags() {  # $1 = arm, $2 = regime JSON to write
    CUDA_VISIBLE_DEVICES='' "$PYTHON" - "$ARMS_CONFIG" "$1" "$2" <<'PYFIT'
import json
import sys
from pathlib import Path

import yaml

config_path, arm_name, regime_path = sys.argv[1], sys.argv[2], sys.argv[3]
spec = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
arms = {arm["name"]: arm for arm in spec["arms"]}
if arm_name not in arms:
    raise SystemExit(f"unknown arm {arm_name!r} in {config_path}")
flags = dict(arms[arm_name].get("flags") or {})


def declared(key: str):
    if key not in flags:
        raise SystemExit(
            f"{arm_name}: {key} is not declared in {config_path}. The fitting command is "
            "derived from the arm definition (§7.2a) precisely so that a calibration is "
            "never fit under one regime and applied under another; an undeclared field "
            "would sit at the tool default while fitting and at the arm's value while "
            "scoring, and nothing downstream could tell."
        )
    return flags[key]


def camera_keys(value) -> list[str]:
    items = value if isinstance(value, (list, tuple)) else str(value).strip().strip("[]()").split(",")
    keys = [str(item).strip().strip("'\"") for item in items]
    keys = [key for key in keys if key]
    if not keys:
        raise SystemExit(f"{arm_name}: plan.subtask_align_camera_keys is empty")
    return keys


frame_format = str(declared("plan.subtask_align_frame_format"))
sampling = str(declared("plan.subtask_align_sampling"))
# Single-view arms declare only vlm.camera_key: with subtask_align_camera_keys
# empty the tool's resolver returns [None], meaning "the frame provider's own
# camera", which IS vlm.camera_key. eval_align_batch resolves an omitted
# --camera from the dataset's own metadata, which need not be the camera the arm
# declares, so the resolution is made explicit here rather than left to two
# defaults that agree by luck.
raw_cameras = flags.get("plan.subtask_align_camera_keys")
if raw_cameras is None or not str(raw_cameras).strip().strip("[]()").strip():
    cameras = [str(declared("vlm.camera_key"))]
else:
    cameras = camera_keys(raw_cameras)
fps = float(declared("plan.frames_per_second"))
max_frames = int(declared("plan.max_frames_per_prompt"))
temperature = float(declared("vlm.temperature"))

# contact_sheet_frame_width is NOT only a contact-sheet setting: the stacked
# video path resizes each view with it before compositing (to_stacked_view_frames),
# and eval_align_batch defaults to 336 where PlanConfig defaults to 224. On a
# stacked arm an undeclared width is therefore a real difference in the pixels
# the model sees, not a cosmetic one.
width = flags.get("plan.contact_sheet_frame_width")
if width is None and (frame_format == "contact_sheet" or len(cameras) > 1):
    raise SystemExit(
        f"{arm_name}: plan.contact_sheet_frame_width must be declared for a "
        f"{frame_format} arm with {len(cameras)} camera(s) -- it sizes the frames this "
        "regime actually renders, and the fitting driver defaults to a different value."
    )

# eval_align_batch builds its own VlmConfig with these hard-coded. An arm that
# disagrees cannot be fit under its own regime at all.
if "vlm.max_new_tokens" in flags and int(flags["vlm.max_new_tokens"]) != 4096:
    raise SystemExit(
        f"{arm_name}: vlm.max_new_tokens={flags['vlm.max_new_tokens']} but eval_align_batch "
        "hard-codes 4096, so the fit could not use the arm's budget."
    )
thinking = flags.get("vlm.chat_template_kwargs")
if thinking is not None:
    parsed = json.loads(thinking) if isinstance(thinking, str) else dict(thinking)
    if parsed.get("enable_thinking", False):
        raise SystemExit(
            f"{arm_name}: eval_align_batch hard-codes enable_thinking=false; this arm asks "
            "for true, so fit and score would decode differently."
        )

# Settings eval_align_batch's PlanConfig does not carry: they would sit at the
# tool default while fitting and at the arm's value while scoring.
uncarried = []
if frame_format == "contact_sheet":
    uncarried += [
        key
        for key in (
            "plan.contact_sheet_columns",
            "plan.contact_sheet_frames_per_sheet",
            "plan.contact_sheet_quality",
        )
        if key in flags
    ]
if sampling == "motion_stratified" and "plan.subtask_align_motion_feature_keys" in flags:
    uncarried.append("plan.subtask_align_motion_feature_keys")
if uncarried:
    raise SystemExit(
        f"{arm_name}: eval_align_batch cannot carry {uncarried} into the fit, so this arm "
        "cannot be calibrated under its own regime. Fit it with a driver that can, or drop "
        "the setting from the arm."
    )

# §7.2a's one deliberate exception, recorded rather than hidden: the fitting
# driver hard-sets subtask_align_min_fraction=0.0 (D8) so that a partly unplaced
# episode still yields a row to fit on, while the scored arm keeps its own value.
min_fraction = flags.get("plan.subtask_align_min_fraction")
min_fraction_source = "arm"
if min_fraction is None:
    min_fraction, min_fraction_source = 0.5, "PlanConfig default"
    print(
        f"note: {arm_name} does not declare plan.subtask_align_min_fraction; recording the "
        "tool default 0.5 as the scoring value",
        file=sys.stderr,
    )
min_fraction = float(min_fraction)
if not 0.0 <= min_fraction <= 1.0:
    raise SystemExit(f"{arm_name}: plan.subtask_align_min_fraction={min_fraction} is outside [0, 1]")

args = ["--formats", frame_format, "--sampling", sampling, "--fps", f"{fps}",
        "--max-frames", f"{max_frames}", "--temperature", f"{temperature}"]
for camera in cameras:
    args += ["--camera", camera]
if width is not None:
    args += ["--frame-width", f"{int(width)}"]

Path(regime_path).parent.mkdir(parents=True, exist_ok=True)
Path(regime_path).write_text(
    json.dumps(
        {
            "arm": arm_name,
            "arms_config": config_path,
            "frame_format": frame_format,
            "camera_keys": cameras,
            "sampling": sampling,
            "frames_per_second": fps,
            "max_frames_per_prompt": max_frames,
            "temperature": temperature,
            "contact_sheet_frame_width": None if width is None else int(width),
            "max_new_tokens": 4096,
            "enable_thinking": False,
            "seed": flags.get("seed"),
            # The arm-level seed reaches only the interjections and VQA modules,
            # both disabled in every arm, and neither frame sampler is random --
            # so it cannot change an alignment fit. It is recorded for provenance
            # rather than passed: eval_align_batch has no seed flag to pass it to,
            # and decoding is unseeded server side either way (§12.8).
            "seed_note": "provenance only: unreachable from the alignment path",
            # eval_align_batch does not set subtask_video_fallback, so it keeps
            # the tool default "contact_sheet" while the video arms declare
            # "error". The difference cannot be passed in; fit_one_component
            # detects it in the run log instead.
            "subtask_video_fallback_score": flags.get(
                "plan.subtask_video_fallback", "contact_sheet"
            ),
            "subtask_video_fallback_fit": "contact_sheet",
            "min_fraction_fit": 0.0,
            "min_fraction_score": min_fraction,
            "min_fraction_score_source": min_fraction_source,
            # §7.2a's one deliberate fit/serve difference, recorded rather than
            # hidden -- including whether it bites for this arm at all.
            "min_fraction_differs": min_fraction != 0.0,
            "min_fraction_note": (
                "eval_align_batch hard-sets subtask_align_min_fraction=0.0 (D8) so a partly "
                "unplaced episode still yields a fitting row, while the scored arm keeps its "
                "own value; min_fraction_differs says whether the two disagree here."
            ),
            "processor_fps": None,
            "processor_fps_note": (
                "not passed: the production CLI cannot send mm_processor_kwargs on the "
                "alignment path (§9), so a --processor-fps fit would be a regime the "
                "scored arm can never reproduce."
            ),
            "fit_command": "lerobot-align-eval-batch " + " ".join(args),
        },
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
print("\n".join(args))
PYFIT
}

# Record the fit gate's verdict instead of obeying it (§7.2), and run the Stage
# B checks that need the written calibration (§7.4.8).
align_record_gate() {  # $1 = arm, $2 = dataset, $3 = calibration, $4 = cohort, $5 = regime, $6 = gate out
    CUDA_VISIBLE_DEVICES='' "$PYTHON" - "$1" "$2" "$3" "$4" "$5" "$6" \
        "$ALIGN_FIT_MIN_GAIN" "$ALIGN_FIT_MIN_FIRST_MAE" "$ALIGN_FIT_MIN_COHORT" <<'PYGATE'
import json
import sys
from pathlib import Path

arm, dataset, calibration_path, cohort_path, regime_path, out_path = sys.argv[1:7]
min_gain, min_first_mae, min_cohort = float(sys.argv[7]), float(sys.argv[8]), int(sys.argv[9])

calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
cohort = json.loads(Path(cohort_path).read_text(encoding="utf-8"))
regime = json.loads(Path(regime_path).read_text(encoding="utf-8"))

labels = calibration.get("labels") or []
if not labels:
    # §3.3: applies_to is `not self.labels or tuple(labels) == self.labels`, so a
    # calibration with no recorded labels applies to EVERY label list silently.
    raise SystemExit(
        f"{dataset} {arm}: {calibration_path} records no non-empty 'labels' key, so it would "
        "apply to every label list including ones it was never fit for (§7.4.8)."
    )
if list(labels) != list(cohort["labels"]):
    raise SystemExit(
        f"{dataset} {arm}: the calibration's labels do not match the pinned cohort's; the "
        "fitter re-chose its own cohort and the calibration describes a different convention."
    )
fit_on = [int(e) for e in calibration.get("fit_on_episodes") or []]
outside = sorted(set(fit_on) - set(int(e) for e in cohort["cohort"]))
if outside:
    raise SystemExit(f"{dataset} {arm}: fitted on episodes outside the pinned cohort: {outside}")
count = int(calibration.get("fit_episode_count") or 0)
if count < min_cohort:
    raise SystemExit(
        f"{dataset} {arm}: fit_episode_count={count} is below the required {min_cohort} "
        "(§7.4.8); episodes were lost between the pinned cohort and the fit."
    )

first_mae = calibration.get("uncalibrated_first_boundary_mae")
gain = calibration.get("heldout_gain_points")
reasons = []
if first_mae is not None and first_mae < min_first_mae:
    reasons.append(f"first-boundary MAE {first_mae:.2f}s < {min_first_mae:.2f}s")
if gain is not None and gain < min_gain:
    reasons.append(f"held-out gain {gain:+.2f} < {min_gain:+.2f} points")

Path(out_path).write_text(
    json.dumps(
        {
            "dataset": dataset,
            "arm": arm,
            # The tool's own recommendation, kept as a PREDICTION to be tested
            # (claim A6) rather than a filter: obeying it would remove all four
            # clean-table components and condition A3 on the population where
            # calibration already works (§7.2).
            "gate_verdict": "RECOMMENDED" if not reasons else "NOT_RECOMMENDED",
            "gate_reasons": reasons,
            "forced": True,
            "min_gain_threshold": min_gain,
            "min_first_boundary_mae_threshold": min_first_mae,
            "uncalibrated_first_boundary_mae": first_mae,
            "heldout_gain_points": gain,
            "fit_episode_count": count,
            "fit_on_episodes": fit_on,
            "pinned_cohort": cohort["cohort"],
            "dropped_from_cohort": cohort["dropped_from_cohort"],
            "distinct_seed_label_lists": cohort["distinct_seed_label_lists"],
            "labels": list(labels),
            "duration_weight": calibration.get("duration_weight"),
            # §7.4.12: recorded so the floor's normalisation and the solver's
            # prior can be read against each other.
            "segment_fractions_sum": (
                None
                if not calibration.get("segment_fractions")
                else round(sum(calibration["segment_fractions"]), 6)
            ),
            "regime": regime,
            "calibration_path": calibration_path,
        },
        indent=1,
    )
    + "\n",
    encoding="utf-8",
)
verdict = "NOT RECOMMENDED" if reasons else "recommended"
print(f"[calibrate] {dataset} {arm}: gate {verdict}"
      + (f" ({'; '.join(reasons)})" if reasons else "")
      + f", fit on {count} episode(s), forced")
PYGATE
}

# Fit one component's calibration under one arm's regime, on SEED episodes only.
fit_one_component() {  # $1 = arm, $2 = dataset, $3 = replica port
    local arm="$1" dataset="$2" port="$3"
    local out_dir="$CAL_DIR/$arm"
    local calibration="$out_dir/$dataset.json"
    local cohort_file="$out_dir/$dataset.cohort.json"
    local regime_file="$out_dir/$dataset.regime.json"
    local gate_file="$out_dir/$dataset.gate.json"
    local predictions="$out_dir/$dataset.fit_predictions.json"
    local log="$out_dir/$dataset.fit.log"
    mkdir -p "$out_dir"
    if [[ -f "$calibration" && -f "$gate_file" && -f "$cohort_file" && -f "$regime_file" ]]; then
        echo "[calibrate] cached: $dataset $arm"
        return 0
    fi

    local cohort_line
    cohort_line="$(align_pin_cohort "$dataset" "$cohort_file")" || return 1
    local -a cohort=()
    read -ra cohort <<< "$cohort_line"

    local flags_text
    flags_text="$(align_fit_flags "$arm" "$regime_file")" || return 1
    local -a fit_flags=()
    mapfile -t fit_flags <<< "$flags_text"
    # The format to fit on comes from the arm's regime too: lerobot-align-fit
    # selects rows by format, and fitting the wrong one silently fits nothing.
    local frame_format
    frame_format="$("$PYTHON" -c "
import json, sys; print(json.load(open(sys.argv[1]))['frame_format'])" "$regime_file")"

    if [[ "${DISPATCH_DRY_RUN:-0}" == "1" ]]; then
        # The cohort and the regime above are CPU-only and already written, so a
        # dry run shows the actual derived fit command -- which is the one thing
        # a human should read before spending 320 calls on a regime that has to
        # match the scored arm field for field (§7.2a).
        local rendered_batch rendered_fit
        printf -v rendered_batch ' %q' "${ALIGN_EVAL_BATCH[@]}" "$SOURCE_DIR/$dataset" \
            --episodes "${cohort[@]}" "${fit_flags[@]}" \
            --model "$MODEL_ID" --api-base "http://127.0.0.1:${port}/v1" --out "$predictions"
        printf -v rendered_fit ' %q' "${ALIGN_FIT[@]}" "$SOURCE_DIR/$dataset" "$predictions" \
            --out "$calibration" --format "$frame_format" \
            --min-gain "$ALIGN_FIT_MIN_GAIN" \
            --min-first-boundary-mae "$ALIGN_FIT_MIN_FIRST_MAE" --force
        printf '[dry-run] fit %s %s:%s\n[dry-run] fit %s %s:%s\n' \
            "$dataset" "$arm" "$rendered_batch" "$dataset" "$arm" "$rendered_fit"
        return 0
    fi

    # eval_align_batch decodes video on the CPU and calls the replica over HTTP;
    # it never needs a device of its own. Blanking the device list makes that a
    # guarantee rather than an expectation -- GPU 4 is not merely absent from
    # the allow-list here, it is unreachable.
    #
    # The pristine source root is safe to read directly: unlike the production
    # CLI, this driver rewrites no parquet and writes only --out.
    if ! CUDA_VISIBLE_DEVICES='' "${ALIGN_EVAL_BATCH[@]}" "$SOURCE_DIR/$dataset" \
            --episodes "${cohort[@]}" \
            "${fit_flags[@]}" \
            --model "$MODEL_ID" --api-base "http://127.0.0.1:${port}/v1" \
            --out "$predictions" >"$log" 2>&1; then
        echo "FATAL: eval-batch failed for $dataset $arm; see $log" >&2
        return 1
    fi

    # The one regime difference that cannot be passed in, so it is detected
    # instead. eval_align_batch keeps plan.subtask_video_fallback at the tool
    # default "contact_sheet" while the video arms declare "error", and every
    # fallback path logs "using contact sheets" before it sends the alternate
    # request. A calibration fit on a mixture of video and contact-sheet
    # boundaries is not a video calibration, whatever the file says.
    if [[ "$frame_format" == "video" ]] && grep -q "using contact sheets" "$log"; then
        echo "FATAL: $dataset $arm: the fit fell back to contact sheets on at least one" >&2
        echo "       episode (see $log), so it would mix formats. eval_align_batch cannot" >&2
        echo "       carry plan.subtask_video_fallback=error into the fit (§7.2a)." >&2
        return 1
    fi

    # --force everywhere, by policy (§7.2). The thresholds are still passed so
    # that the verdict recorded next to the calibration is a verdict against a
    # STATED rule rather than against whatever the tool's defaults become.
    if ! CUDA_VISIBLE_DEVICES='' "${ALIGN_FIT[@]}" "$SOURCE_DIR/$dataset" "$predictions" \
            --out "$calibration" --format "$frame_format" \
            --min-gain "$ALIGN_FIT_MIN_GAIN" \
            --min-first-boundary-mae "$ALIGN_FIT_MIN_FIRST_MAE" \
            --force >>"$log" 2>&1; then
        echo "FATAL: fit failed for $dataset $arm; see $log" >&2
        return 1
    fi

    align_record_gate "$arm" "$dataset" "$calibration" "$cohort_file" "$regime_file" "$gate_file"
}

# Same lease discipline as dispatch(): one replica per fit, held for its whole
# duration. A failed fit is NOT treated as data the way a failed arm is -- a
# missing calibration silently removes that component from the calibrated arm,
# which is the corpus-shrinking failure §7.2 exists to prevent -- so the stage
# stops instead of reporting a smaller population.
dispatch_fits() {
    local -a jobs=("$@")
    local -a ports=()
    mapfile -t ports < <(replica_ports)
    local -a pids=()
    local failed=0
    local port
    open_replica_pool "${ports[@]}"
    set +e
    for job in "${jobs[@]}"; do
        IFS='|' read -r arm dataset <<< "$job"
        read -r port <&9
        (
            fit_one_component "$arm" "$dataset" "$port"
            status=$?
            printf '%s\n' "$port" >&9
            exit $status
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
    close_replica_pool
    set -e
    echo "[calibrate] ${#jobs[@]} fits, $failed failed"
    if (( failed > 0 )); then
        echo "FATAL: $failed calibration fit(s) failed. Every component must be calibrated" >&2
        echo "       or claim A3 is conditioned on the components where fitting happened" >&2
        echo "       to work (alignment_plan.md §7.2)." >&2
        exit 1
    fi
}

# Gather the per-component gate verdicts into the record the report reads. This
# is the evidence claim A6 is tested against: does the shipped gate fire exactly
# where calibration hurts?
#
# Keyed by "<component>__<arm>" because each calibrated arm is fit under its own
# regime (§7.2a) and the two fits earn their own verdicts; collapsing them onto
# the component would report one arm's gate as though it were both.
align_collect_gates() {
    CUDA_VISIBLE_DEVICES='' "$PYTHON" - "$CAL_DIR" "$RESULT_DIR/alignment_calibration.json" <<'PYGATES'
import json
import sys
from pathlib import Path

cal_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
paths = sorted(cal_dir.glob("*/*.gate.json"))
if not paths:
    # An empty index is not a result, and overwriting a real one with it -- which
    # a dry run pointed at another CAL_DIR would otherwise do -- destroys the
    # evidence A6 is tested against.
    print(f"WARNING: no gate verdicts under {cal_dir}; leaving {out_path} untouched", file=sys.stderr)
    raise SystemExit(0)

records = {}
for path in paths:
    row = json.loads(path.read_text(encoding="utf-8"))
    records[f"{row['dataset']}__{row['arm']}"] = {
        # The two field names make_alignment_report.py renders, beside the full
        # record so nothing has to be recomputed from the calibration files.
        "gate": row["gate_verdict"],
        "obeyed": "no (--force, plan §7.2)",
        **row,
    }
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(records, indent=1) + "\n", encoding="utf-8")
recommended = sum(1 for row in records.values() if row["gate"] == "RECOMMENDED")
print(f"[calibrate] {len(records)} gate verdict(s) -> {out_path} "
      f"({recommended} recommended, {len(records) - recommended} forced against the gate)")
PYGATES
}

# Every per-component file a job will be handed must exist BEFORE the sweep
# starts. run_arm.py content-hashes both files and exits if one is missing, so
# without this check a missing calibration costs one failed job per component
# and the calibrated arm quietly ends up with a smaller population than the arm
# it is contrasted against.
require_alignment_inputs() {  # $@ = arms
    local arm dataset path missing=0
    for dataset in "${DATASETS[@]}"; do
        for arm in "$@"; do
            if [[ -n "${ARM_NEEDS_SUBTASKS[$arm]:-}" ]]; then
                path="$(align_label_file "$dataset")"
                if [[ ! -f "$path" ]]; then
                    echo "missing label file: $path" >&2
                    missing=$((missing + 1))
                fi
            fi
            if [[ -n "${ARM_NEEDS_CALIBRATION[$arm]:-}" ]]; then
                path="$(align_calibration_file "$arm" "$dataset")"
                if [[ ! -f "$path" ]]; then
                    echo "missing calibration: $path" >&2
                    missing=$((missing + 1))
                fi
            fi
        done
    done
    if (( missing > 0 )); then
        echo "FATAL: $missing supplied file(s) missing. Run './run_all.sh prepare-labels'" >&2
        echo "       and './run_all.sh calibrate' first." >&2
        exit 1
    fi
}

# The alignment stages drive scripts that live beside this one and are written
# by other hands. Naming the missing ones up front beats a bare "can't open
# file" from python once a stage has already started doing work.
require_scripts() {
    local script
    local -a missing=()
    for script in "$@"; do
        [[ -f "$HERE/$script" ]] || missing+=("$script")
    done
    if (( ${#missing[@]} > 0 )); then
        echo "FATAL: missing evaluation script(s): ${missing[*]}" >&2
        echo "       They are part of this study's harness; see alignment_plan.md §10.2." >&2
        exit 1
    fi
}

stage_prepare_labels() {
    require_scripts make_label_files.py
    mkdir -p "$LABEL_DIR" "$RESULT_DIR"
    # One corpus-wide pass, not sixteen: the modal population and the 507-episode
    # reconciliation of §2.1b are corpus-level facts, and the summary that
    # carries them only means anything whole.
    local -a components=()
    mapfile -t components < <(align_datasets)
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/make_label_files.py" \
        --gt-dir "$GT_DIR" --splits-dir "$SPLIT_DIR" --out-dir "$LABEL_DIR" \
        --datasets "${components[@]}" \
        --summary "$RESULT_DIR/alignment_labels_summary.json"
    # Regenerating a label file changes what the arms were given. run_arm.py
    # hashes the file's CONTENT (D1), so predictions made against the previous
    # labels are correctly seen as stale -- but they are only re-run when a
    # sweep is re-run, so re-run one after regenerating labels.
    require_label_coverage
}

stage_floors() {
    require_scripts align_floors.py
    mkdir -p "$ALIGN_PRED_DIR" "$RESULT_DIR"
    require_label_coverage
    # No VLM call, no GPU, no replica: these arms never look at a frame. They
    # are the denominator of claim A5 -- six configurations compared only with
    # each other establish which is best, not whether any of them is good.
    #
    # Gate §7.4.11 runs FIRST. The seeded floors must be model-free, which is
    # checked by feeding them two opposite synthetic model answers and requiring
    # identical output. This is not a formality: §5.1 measured the obvious
    # "priors-only DP" floor still moving 0.70s with the prior weighted 100,000x,
    # i.e. a floor that silently contained the model it was meant to bound.
    local -a subset=()
    mapfile -t subset < <(align_dataset_flags)
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/align_floors.py" \
        --gt "$GT_DIR" --split "$SPLIT_DIR" --labels "$LABEL_DIR" \
        "${subset[@]}" \
        --label-source "$ALIGN_LABEL_SOURCE" --out-dir "$ALIGN_PRED_DIR" --self-test
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/align_floors.py" \
        --gt "$GT_DIR" --split "$SPLIT_DIR" --labels "$LABEL_DIR" \
        "${subset[@]}" \
        --label-source "$ALIGN_LABEL_SOURCE" --out-dir "$ALIGN_PRED_DIR" \
        --summary-out "$RESULT_DIR/alignment_floors_summary.json"
}

stage_calibrate() {
    use_alignment_configuration
    if (( ${#ARM_NEEDS_CALIBRATION[@]} == 0 )); then
        echo "FATAL: no arm in $ARMS_CONFIG declares a {calibration_path} flag, so there" >&2
        echo "       is nothing to fit and claim A3 has no numerator." >&2
        exit 1
    fi
    local -a arms=()
    mapfile -t arms < <(printf '%s\n' "${!ARM_NEEDS_CALIBRATION[@]}" | sort)
    echo "[calibrate] ${#arms[@]} calibrated arm(s) x ${#DATASETS[@]} components, seed episodes only"
    local -a jobs=()
    local arm dataset
    for arm in "${arms[@]}"; do
        for dataset in "${DATASETS[@]}"; do jobs+=("$arm|$dataset"); done
    done
    dispatch_fits "${jobs[@]}"
    align_collect_gates
}

stage_align_full() {
    use_alignment_configuration
    local -a arms=()
    local selected
    selected="$(arms_matching "${ALIGN_ARM_FILTER:-}")" || return 1
    mapfile -t arms <<< "$selected"
    require_label_coverage
    require_alignment_inputs "${arms[@]}"
    mkdir -p "$ALIGN_PRED_DIR"
    echo "[align-full] ${#arms[@]} arms x ${#DATASETS[@]} components -> $ALIGN_PRED_DIR"
    local -a jobs=()
    local dataset arm
    for dataset in "${DATASETS[@]}"; do
        for arm in "${arms[@]}"; do jobs+=("$dataset|$arm"); done
    done
    dispatch "${jobs[@]}"
}

stage_align_repeats() {
    use_alignment_configuration
    # Repeats land in their own directory. They measure the decoding-noise band
    # (§12.8), and a repeat swept into the headline population would weight one
    # component two or three times over the other fifteen.
    DISPATCH_PRED_DIR="$ALIGN_REPEAT_DIR"
    local -a arms=()
    local selected
    selected="$(arms_matching "${ALIGN_ARM_FILTER:-}")" || return 1
    mapfile -t arms <<< "$selected"
    require_alignment_inputs "${arms[@]}"
    mkdir -p "$ALIGN_REPEAT_DIR"
    echo "[align-repeats] ${#arms[@]} arms x $ALIGN_REPEATS extra pass(es) on $ALIGN_REPEAT_DATASET"
    local -a jobs=()
    local arm repeat
    for arm in "${arms[@]}"; do
        for repeat in $(seq 1 "$ALIGN_REPEATS"); do
            jobs+=("$ALIGN_REPEAT_DATASET|$arm|$repeat")
        done
    done
    dispatch "${jobs[@]}"
}

# aggregate.py, with the ONE tolerated failure mode spelled out.
#
# Exit 3 is "the Holm family computed is not the one registered" (its
# EXIT_FAMILY_SHORTFALL). That is exactly what a deliberately partial run
# produces: the model-free floors alone have no VLM numerator for contrasts 7 and
# 8, so 0 of the 24 pre-registered tests exist. It is ALSO what a real run with a
# broken family produces, which is why tolerating it requires saying so out loud
# via ALIGN_ALLOW_INCOMPLETE, and why the tolerance is pinned to that one code:
# any other non-zero status -- a crash, an unreadable scores file, a config the
# loader rejects -- still aborts the stage under `set -e`.
#
# Nothing about the statistics is relaxed. aggregate.py writes the same JSON with
# the same `holm_family_note`, Holm still corrects only the tests that exist, and
# make_alignment_report.py still refuses to render the family without its own
# --allow-incomplete and still stamps the output as not a result.
align_aggregate() {
    local status=0
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/aggregate.py" "$@" || status=$?
    if (( status == 0 )); then
        return 0
    fi
    if (( status == 3 )) && [[ -n "${ALIGN_ALLOW_INCOMPLETE:-}" ]]; then
        echo "[analyse-alignment] aggregate.py reported an INCOMPLETE Holm family (exit 3);" >&2
        echo "    tolerated because ALIGN_ALLOW_INCOMPLETE is set. This run is NOT the" >&2
        echo "    pre-registered family and its p-values are not the ones the" >&2
        echo "    pre-registration licensed (alignment_plan.md §8)." >&2
        return 0
    fi
    return "$status"
}

stage_analyse_alignment() {
    require_scripts score_alignment.py make_alignment_report.py aggregate.py
    use_alignment_configuration
    if [[ ! -f "$ALIGN_CONTRASTS_CONFIG" ]]; then
        echo "FATAL: $ALIGN_CONTRASTS_CONFIG not found. The §8 contrast family has to be" >&2
        echo "       data, or Holm corrects six small families instead of one (D13)." >&2
        exit 1
    fi
    mkdir -p "$RESULT_DIR" "$EVAL_DIR/analysis"
    # The floors are produced by a separate, cheap, GPU-free stage, which makes
    # them easy to forget -- and without them contrasts 7 and 8 have no
    # denominator, so claim A5 quietly disappears rather than failing.
    if ! compgen -G "$ALIGN_PRED_DIR/*__floor_*.json" >/dev/null; then
        echo "[analyse-alignment] WARNING: no model-free floor predictions in" >&2
        echo "    $ALIGN_PRED_DIR. Contrasts 7 and 8 (claim A5) cannot be computed;" >&2
        echo "    run './run_all.sh floors' -- it costs no GPU time." >&2
    fi
    # CPU only, and it must not rerun annotation: scoring reads saved spans.
    local -a subset=()
    mapfile -t subset < <(align_dataset_flags)
    CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/score_alignment.py" \
        --predictions-dir "$ALIGN_PRED_DIR" --gt-dir "$GT_DIR" \
        "${subset[@]}" \
        --splits-dir "$SPLIT_DIR" --out "$RESULT_DIR/alignment_scores.jsonl"
    # The repeats are scored by the same code into their own file, so the
    # decoding-noise band and the headline population never share a row.
    if compgen -G "$ALIGN_REPEAT_DIR/*.json" >/dev/null; then
        CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/score_alignment.py" \
            --predictions-dir "$ALIGN_REPEAT_DIR" --gt-dir "$GT_DIR" \
            "${subset[@]}" \
            --splits-dir "$SPLIT_DIR" --out "$RESULT_DIR/alignment_repeat_scores.jsonl"
    else
        echo "[analyse-alignment] WARNING: no repeat predictions in $ALIGN_REPEAT_DIR." >&2
        echo "    Temperature is 0.2 and the request carries no seed, so every null and" >&2
        echo "    every equivalence claim needs the noise band from './run_all.sh" >&2
        echo "    align-repeats' before it can be read (alignment_plan.md §12.8)." >&2
    fi
    # Aggregation is what turns per-episode rows into the pre-registered family.
    # --matcher any: alignment rows carry no matcher, because labels are supplied
    # and spans are matched by returned index (D10). --contrasts is what makes
    # Holm correct ONE family of (contrast x metric) tests instead of six small
    # ones (D13), and it scopes the equalisation gate per pair (D12).
    align_aggregate \
        --scores "$RESULT_DIR/alignment_scores.jsonl" \
        --out "$RESULT_DIR/alignment_all.json" \
        --matcher any --contrasts "$ALIGN_CONTRASTS_CONFIG" \
        --arms-config "$ARMS_CONFIG"
    # The same family restricted to the subpopulation where the treatment is
    # uniform (§4.2): under per-episode label lists a "calibrated" arm is
    # calibrated only where the label tuple matches the fit, so contrasts 3 and 4
    # are a blended treatment over the whole population. The restriction needs a
    # per-episode flag that does NOT depend on which arm produced the row --
    # `calibration_applied` is false for the uncalibrated baseline by
    # construction, so filtering on it would delete the other half of every
    # calibration contrast. Run it only when such a field exists; say so plainly
    # when it does not, rather than reporting a restricted number that is really
    # a one-armed one.
    local modal_field="${ALIGN_MODAL_FIELD:-labels_match_calibration}"
    if CUDA_VISIBLE_DEVICES='' "$PYTHON" -c "
import json, sys
row = json.loads(open(sys.argv[1], encoding='utf-8').readline())
raise SystemExit(0 if sys.argv[2] in row else 1)" \
            "$RESULT_DIR/alignment_scores.jsonl" "$modal_field"; then
        align_aggregate \
            --scores "$RESULT_DIR/alignment_scores.jsonl" \
            --out "$RESULT_DIR/alignment_modal.json" \
            --matcher any --contrasts "$ALIGN_CONTRASTS_CONFIG" \
            --arms-config "$ARMS_CONFIG" \
            --require-true "$modal_field"
    else
        echo "[analyse-alignment] NOTE: score rows carry no arm-independent '$modal_field'" >&2
        echo "    field, so the modal-subpopulation aggregate (§4.2) is not computed and the" >&2
        echo "    report will say so. Set ALIGN_MODAL_FIELD once score_alignment.py emits one." >&2
    fi
    align_collect_gates
    # Every input this reads defaults to <results>/<name>, so the report is
    # assembled from the files the stages above wrote rather than from a
    # command line that could name a different set.
    local -a draft=()
    if [[ -n "${ALIGN_ALLOW_INCOMPLETE:-}" ]]; then
        draft=(--allow-incomplete)
    fi
    # make_alignment_report.py exits non-zero for a marked draft BY DESIGN, and
    # that status is the point of the flag, so it is reported rather than
    # inherited by the stage. Without ALIGN_ALLOW_INCOMPLETE nothing changes: the
    # report refuses, `set -e` aborts, and the stage fails.
    if ! CUDA_VISIBLE_DEVICES='' "$PYTHON" "$HERE/make_alignment_report.py" \
        --results "$RESULT_DIR" "${draft[@]}" \
        --out "$EVAL_DIR/analysis/alignment_report.md"; then
        if [[ -n "${ALIGN_ALLOW_INCOMPLETE:-}" ]]; then
            echo "[analyse-alignment] report written as a MARKED DRAFT (see banner above)." >&2
        else
            return 1
        fi
    fi
}

case "${1:-}" in
    prepare)   stage_prepare ;;
    preflight) stage_preflight ;;
    serve)     stage_serve ;;
    falsify)   stage_falsify ;;
    pilot)     stage_pilot ;;
    full)      stage_full ;;
    analyse)   stage_analyse ;;
    # Study 2: fixed-label alignment. No C1 verdict is consulted anywhere below.
    prepare-labels)    stage_prepare_labels ;;
    floors)            stage_floors ;;
    calibrate)         stage_calibrate ;;
    align-full)        stage_align_full ;;
    align-repeats)     stage_align_repeats ;;
    analyse-alignment) stage_analyse_alignment ;;
    # The usage text is the whole leading comment block. A fixed line range
    # stopped listing stages as soon as the header grew past it, which is how a
    # stage gets added and never discovered.
    *) awk 'NR > 1 && /^#/ { print; next } NR > 1 { exit }' "${BASH_SOURCE[0]}"; exit 1 ;;
esac
