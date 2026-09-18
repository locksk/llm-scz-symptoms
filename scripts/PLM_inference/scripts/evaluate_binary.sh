#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    printf '%s\n' \
        "Usage: $0 [--config FILE] [--evaluation-root DIR] [--output-dir DIR] [--dry-run]" \
        "       $0 --smoke [--config FILE]" \
        "" \
        "Defaults use MENTAL_LLM_DATA_ROOT/outputs/binary and reports/binary."
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
    "${PYTHON_BIN}" -m mental_llm.reporting.tables \
        --config "${CONFIG_FILE}" --evaluation-root "${WORKSPACE}/outputs" \
        --output-dir "${WORKSPACE}/reports" --dry-run
    printf 'Synthetic binary reporting configuration smoke test passed.\n'
    exit 0
fi

EVALUATION_ROOT=""
OUTPUT_DIR=""
DRY_RUN=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="${2:?Missing value for --config}"; shift 2 ;;
        --evaluation-root) EVALUATION_ROOT="${2:?Missing value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?Missing value}"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
if [[ -z "$EVALUATION_ROOT" || -z "$OUTPUT_DIR" ]]; then
    [[ -n "${MENTAL_LLM_DATA_ROOT:-}" ]] || {
        printf 'Set MENTAL_LLM_DATA_ROOT or pass both report paths explicitly.\n' >&2
        exit 2
    }
    EVALUATION_ROOT="${EVALUATION_ROOT:-${MENTAL_LLM_DATA_ROOT}/outputs/binary}"
    OUTPUT_DIR="${OUTPUT_DIR:-${MENTAL_LLM_DATA_ROOT}/reports/binary}"
fi
EXTRA_ARGS=()
[[ "$DRY_RUN" -eq 0 ]] || EXTRA_ARGS+=(--dry-run)
"${PYTHON_BIN}" -m mental_llm.reporting.tables \
    --config "${CONFIG_FILE}" --evaluation-root "${EVALUATION_ROOT}" \
    --output-dir "${OUTPUT_DIR}" "${EXTRA_ARGS[@]}"

