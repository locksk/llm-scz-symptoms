#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    printf '%s\n' \
        "Usage: $0 --scheme NAME --model-name MODEL [OPTIONS]" \
        "       $0 --smoke [--config FILE]" \
        "" \
        "Options: --config FILE --data-dir DIR --checkpoint-root DIR --output-dir DIR" \
        "         --dry-run --overwrite"
}

CONFIG_FILE="${MENTAL_LLM_CONFIG:-${REPOSITORY_ROOT}/configs/portable/binary.json}"
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--smoke" ]]; then
    shift
    CONFIG_FILE="${REPOSITORY_ROOT}/configs/smoke/binary.json"
    if [[ "${1:-}" == "--config" ]]; then CONFIG_FILE="${2:?Missing value}"; shift 2; fi
    [[ "$#" -eq 0 ]] || { usage >&2; exit 2; }
    WORKSPACE="$(mktemp -d)"
    trap 'rm -rf -- "${WORKSPACE}"' EXIT
    "${REPOSITORY_ROOT}/scripts/prepare_data.sh" all \
        --config "${CONFIG_FILE}" \
        --source "${REPOSITORY_ROOT}/data/samples/source.synthetic.csv" \
        --strata "${REPOSITORY_ROOT}/data/samples/strata.synthetic.csv" \
        --prepared-dir "${WORKSPACE}/prepared" --staged-dir "${WORKSPACE}/staged"
    MODEL="$("${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-models | sed -n '1p')"
    "${PYTHON_BIN}" -m mental_llm.binary.inference \
        --config "${CONFIG_FILE}" --scheme Scheme_2_S1_ValTest \
        --model-name "${MODEL}" --data-dir "${WORKSPACE}/staged/Scheme_2_S1_ValTest" \
        --checkpoint-root "${WORKSPACE}/checkpoints/binary" \
        --output-dir "${WORKSPACE}/outputs" --dry-run
    printf 'Synthetic binary inference configuration smoke test passed.\n'
    exit 0
fi

SCHEME=""
MODEL_NAME=""
DATA_DIR=""
CHECKPOINT_ROOT=""
OUTPUT_DIR=""
DRY_RUN=0
OVERWRITE=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="${2:?Missing value for --config}"; shift 2 ;;
        --scheme) SCHEME="${2:?Missing value for --scheme}"; shift 2 ;;
        --model-name) MODEL_NAME="${2:?Missing value for --model-name}"; shift 2 ;;
        --data-dir) DATA_DIR="${2:?Missing value for --data-dir}"; shift 2 ;;
        --checkpoint-root) CHECKPOINT_ROOT="${2:?Missing value for --checkpoint-root}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?Missing value for --output-dir}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$SCHEME" && -n "$MODEL_NAME" ]] || { usage >&2; exit 2; }
if [[ -z "$DATA_DIR" || -z "$CHECKPOINT_ROOT" || -z "$OUTPUT_DIR" ]]; then
    [[ -n "${MENTAL_LLM_DATA_ROOT:-}" ]] || {
        printf 'Set MENTAL_LLM_DATA_ROOT or pass all private paths explicitly.\n' >&2
        exit 2
    }
    SAFE_MODEL="${MODEL_NAME//\//_}"
    DATA_DIR="${DATA_DIR:-${MENTAL_LLM_DATA_ROOT}/staged/${SCHEME}}"
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MENTAL_LLM_DATA_ROOT}/checkpoints/binary}"
    OUTPUT_DIR="${OUTPUT_DIR:-${MENTAL_LLM_DATA_ROOT}/outputs/binary/${SCHEME}/${SAFE_MODEL}}"
fi
EXTRA_ARGS=()
[[ "$DRY_RUN" -eq 0 ]] || EXTRA_ARGS+=(--dry-run)
[[ "$OVERWRITE" -eq 0 ]] || EXTRA_ARGS+=(--overwrite)
"${PYTHON_BIN}" -m mental_llm.binary.inference \
    --config "${CONFIG_FILE}" --scheme "${SCHEME}" --model-name "${MODEL_NAME}" \
    --data-dir "${DATA_DIR}" --checkpoint-root "${CHECKPOINT_ROOT}" \
    --output-dir "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}"

