#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${REPOSITORY_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    printf '%s\n' \
        "Usage: $0 prepare --source FILE --strata FILE --output-dir DIR [OPTIONS]" \
        "       $0 stage --prepared-dir DIR --staged-dir DIR [OPTIONS]" \
        "       $0 all --source FILE --strata FILE --prepared-dir DIR --staged-dir DIR [OPTIONS]" \
        "       $0 --smoke [--config FILE]" \
        "" \
        "Options: --config FILE --overwrite --validation-only"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ "${1:-}" == "--smoke" ]]; then
    shift
    CONFIG_FILE="${REPOSITORY_ROOT}/configs/smoke/binary.json"
    if [[ "${1:-}" == "--config" ]]; then
        CONFIG_FILE="${2:?Missing value for --config}"
        shift 2
    fi
    [[ "$#" -eq 0 ]] || { usage >&2; exit 2; }
    WORKSPACE="$(mktemp -d)"
    trap 'rm -rf -- "${WORKSPACE}"' EXIT
    mapfile -t LABELS < <(
        "${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-labels
    )
    "${PYTHON_BIN}" -m mental_llm.data.group_split \
        --source "${REPOSITORY_ROOT}/data/samples/source.synthetic.csv" \
        --strata "${REPOSITORY_ROOT}/data/samples/strata.synthetic.csv" \
        --output-dir "${WORKSPACE}/prepared" --label-cols "${LABELS[@]}" --seed 42
    "${PYTHON_BIN}" -m mental_llm.data.stage \
        --prepared-dir "${WORKSPACE}/prepared" --staged-dir "${WORKSPACE}/staged" \
        --label-cols "${LABELS[@]}"
    printf 'Synthetic preparation and staging smoke test passed.\n'
    exit 0
fi

ACTION="${1:-}"
[[ "$ACTION" == "prepare" || "$ACTION" == "stage" || "$ACTION" == "all" ]] || {
    usage >&2
    exit 2
}
shift

CONFIG_FILE="${MENTAL_LLM_CONFIG:-${REPOSITORY_ROOT}/configs/portable/binary.json}"
SOURCE_FILE=""
STRATA_FILE=""
OUTPUT_DIR=""
PREPARED_DIR=""
STAGED_DIR=""
OVERWRITE=0
VALIDATION_ONLY=0
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --config) CONFIG_FILE="${2:?Missing value for --config}"; shift 2 ;;
        --source) SOURCE_FILE="${2:?Missing value for --source}"; shift 2 ;;
        --strata) STRATA_FILE="${2:?Missing value for --strata}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?Missing value for --output-dir}"; shift 2 ;;
        --prepared-dir) PREPARED_DIR="${2:?Missing value for --prepared-dir}"; shift 2 ;;
        --staged-dir) STAGED_DIR="${2:?Missing value for --staged-dir}"; shift 2 ;;
        --overwrite) OVERWRITE=1; shift ;;
        --validation-only) VALIDATION_ONLY=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done
mapfile -t LABELS < <(
    "${PYTHON_BIN}" -m mental_llm.config --config "${CONFIG_FILE}" --print-labels
)
EXTRA_ARGS=()
[[ "$OVERWRITE" -eq 1 ]] && EXTRA_ARGS+=(--overwrite)

if [[ "$ACTION" == "prepare" ]]; then
    [[ -n "$SOURCE_FILE" && -n "$STRATA_FILE" && -n "$OUTPUT_DIR" ]] || {
        usage >&2; exit 2;
    }
    [[ "$VALIDATION_ONLY" -eq 0 ]] || EXTRA_ARGS+=(--validation-only)
    "${PYTHON_BIN}" -m mental_llm.data.group_split \
        --source "${SOURCE_FILE}" --strata "${STRATA_FILE}" \
        --output-dir "${OUTPUT_DIR}" --label-cols "${LABELS[@]}" --seed 42 \
        "${EXTRA_ARGS[@]}"
elif [[ "$ACTION" == "stage" ]]; then
    [[ -n "$PREPARED_DIR" && -n "$STAGED_DIR" ]] || { usage >&2; exit 2; }
    "${PYTHON_BIN}" -m mental_llm.data.stage \
        --prepared-dir "${PREPARED_DIR}" --staged-dir "${STAGED_DIR}" \
        --label-cols "${LABELS[@]}" "${EXTRA_ARGS[@]}"
else
    [[ -n "$SOURCE_FILE" && -n "$STRATA_FILE" && -n "$PREPARED_DIR" && -n "$STAGED_DIR" ]] || {
        usage >&2; exit 2;
    }
    [[ "$VALIDATION_ONLY" -eq 0 ]] || {
        printf 'The all action cannot stage a validation-only preparation.\n' >&2
        exit 2
    }
    "${PYTHON_BIN}" -m mental_llm.data.group_split \
        --source "${SOURCE_FILE}" --strata "${STRATA_FILE}" \
        --output-dir "${PREPARED_DIR}" --label-cols "${LABELS[@]}" --seed 42 \
        "${EXTRA_ARGS[@]}"
    "${PYTHON_BIN}" -m mental_llm.data.stage \
        --prepared-dir "${PREPARED_DIR}" --staged-dir "${STAGED_DIR}" \
        --label-cols "${LABELS[@]}" "${EXTRA_ARGS[@]}"
fi

