#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    printf '%s\n' \
        "Usage: $0 --scheme NAME --model-name MODEL (--label LABEL|--all-labels) [OPTIONS]" \
        "       $0 --smoke [--config FILE]" \
        "" \
        "Options: --config FILE --data-dir DIR --checkpoint-root DIR --dry-run --overwrite" \
        "Defaults use MENTAL_LLM_DATA_ROOT for private staged data and checkpoints."
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
    LABEL="$("${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-labels | sed -n '1p')"
    MODEL="$("${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-models | sed -n '1p')"
    "${PYTHON_BIN}" -m mental_llm.binary.train \
        --config "${CONFIG_FILE}" --scheme Scheme_1_S1_Train \
        --model-name "${MODEL}" --label "${LABEL}" \
        --data-dir "${WORKSPACE}/staged/Scheme_1_S1_Train" \
        --checkpoint-root "${WORKSPACE}/checkpoints/binary" --dry-run
    printf 'Synthetic binary training configuration smoke test passed.\n'
    exit 0
fi

SCHEME=""
MODEL_NAME=""
LABEL=""
ALL_LABELS=0
DATA_DIR=""
CHECKPOINT_ROOT=""
DRY_RUN=0
OVERWRITE=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="${2:?Missing value for --config}"; shift 2 ;;
        --scheme) SCHEME="${2:?Missing value for --scheme}"; shift 2 ;;
        --model-name) MODEL_NAME="${2:?Missing value for --model-name}"; shift 2 ;;
        --label) LABEL="${2:?Missing value for --label}"; shift 2 ;;
        --all-labels) ALL_LABELS=1; shift ;;
        --data-dir) DATA_DIR="${2:?Missing value for --data-dir}"; shift 2 ;;
        --checkpoint-root) CHECKPOINT_ROOT="${2:?Missing value for --checkpoint-root}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$SCHEME" && -n "$MODEL_NAME" ]] || { usage >&2; exit 2; }
if [[ "$ALL_LABELS" -eq 1 && -n "$LABEL" ]] || [[ "$ALL_LABELS" -eq 0 && -z "$LABEL" ]]; then
    usage >&2
    exit 2
fi
if [[ -z "$DATA_DIR" || -z "$CHECKPOINT_ROOT" ]]; then
    [[ -n "${MENTAL_LLM_DATA_ROOT:-}" ]] || {
        printf 'Set MENTAL_LLM_DATA_ROOT or pass --data-dir and --checkpoint-root.\n' >&2
        exit 2
    }
    DATA_DIR="${DATA_DIR:-${MENTAL_LLM_DATA_ROOT}/staged/${SCHEME}}"
    CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${MENTAL_LLM_DATA_ROOT}/checkpoints/binary}"
fi

if [[ "$ALL_LABELS" -eq 1 ]]; then
    mapfile -t LABELS < <(
        "${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-labels
    )
else
    LABELS=("${LABEL}")
fi
EXTRA_ARGS=()
[[ "$DRY_RUN" -eq 0 ]] || EXTRA_ARGS+=(--dry-run)
[[ "$OVERWRITE" -eq 0 ]] || EXTRA_ARGS+=(--overwrite)
for CURRENT_LABEL in "${LABELS[@]}"; do
    "${PYTHON_BIN}" -m mental_llm.binary.train \
        --config "${CONFIG_FILE}" --scheme "${SCHEME}" \
        --model-name "${MODEL_NAME}" --label "${CURRENT_LABEL}" \
        --data-dir "${DATA_DIR}" --checkpoint-root "${CHECKPOINT_ROOT}" \
        "${EXTRA_ARGS[@]}"
done

