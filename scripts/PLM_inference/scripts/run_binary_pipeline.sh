#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    printf '%s\n' \
        "Usage: $0 --source FILE --strata FILE --data-root DIR [--config FILE]" \
        "       $0 --smoke [--config FILE]" \
        "" \
        "Prepare and stage both schemes, train each label independently, run" \
        "validation/test inference, aggregate predictions, and build final tables."
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi
if [[ "${1:-}" == "--smoke" ]]; then
    shift
    CONFIG_FILE="${REPOSITORY_ROOT}/configs/smoke/binary.json"
    if [[ "${1:-}" == "--config" ]]; then CONFIG_FILE="${2:?Missing value}"; shift 2; fi
    [[ "$#" -eq 0 ]] || { usage >&2; exit 2; }
    "${REPOSITORY_ROOT}/scripts/prepare_data.sh" --smoke --config "${CONFIG_FILE}"
    "${REPOSITORY_ROOT}/scripts/train_binary.sh" --smoke --config "${CONFIG_FILE}"
    "${REPOSITORY_ROOT}/scripts/infer_binary.sh" --smoke --config "${CONFIG_FILE}"
    "${REPOSITORY_ROOT}/scripts/evaluate_binary.sh" --smoke --config "${CONFIG_FILE}"
    printf 'Offline synthetic binary pipeline smoke test passed.\n'
    exit 0
fi

CONFIG_FILE="${MENTAL_LLM_CONFIG:-${REPOSITORY_ROOT}/configs/portable/binary.json}"
SOURCE_FILE=""
STRATA_FILE=""
DATA_ROOT="${MENTAL_LLM_DATA_ROOT:-}"
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="${2:?Missing value for --config}"; shift 2 ;;
        --source) SOURCE_FILE="${2:?Missing value for --source}"; shift 2 ;;
        --strata) STRATA_FILE="${2:?Missing value for --strata}"; shift 2 ;;
        --data-root) DATA_ROOT="${2:?Missing value for --data-root}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ -n "$SOURCE_FILE" && -n "$STRATA_FILE" && -n "$DATA_ROOT" ]] || {
    usage >&2
    exit 2
}
export MENTAL_LLM_DATA_ROOT="$DATA_ROOT"

"${REPOSITORY_ROOT}/scripts/prepare_data.sh" all \
    --config "${CONFIG_FILE}" --source "${SOURCE_FILE}" --strata "${STRATA_FILE}" \
    --prepared-dir "${DATA_ROOT}/prepared" --staged-dir "${DATA_ROOT}/staged"

mapfile -t SCHEMES < <(
    "${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-schemes
)
mapfile -t MODELS < <(
    "${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-models
)
for SCHEME in "${SCHEMES[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        "${REPOSITORY_ROOT}/scripts/train_binary.sh" \
            --config "${CONFIG_FILE}" --scheme "${SCHEME}" \
            --model-name "${MODEL}" --all-labels
        "${REPOSITORY_ROOT}/scripts/infer_binary.sh" \
            --config "${CONFIG_FILE}" --scheme "${SCHEME}" --model-name "${MODEL}"
    done
done
"${REPOSITORY_ROOT}/scripts/evaluate_binary.sh" --config "${CONFIG_FILE}"

