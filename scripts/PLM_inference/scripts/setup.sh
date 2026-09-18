#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

usage() {
    printf '%s\n' \
        "Usage: $0 [--dev]" \
        "       $0 --smoke [--config FILE]" \
        "" \
        "Create a local virtual environment and install release dependencies."
}

INSTALL_DEV=0
SMOKE=0
CONFIG_FILE="${REPOSITORY_ROOT}/configs/smoke/binary.json"
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --dev) INSTALL_DEV=1; shift ;;
        --smoke) SMOKE=1; shift ;;
        --config) CONFIG_FILE="${2:?Missing value for --config}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

if [[ "$SMOKE" -eq 1 ]]; then
    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${REPOSITORY_ROOT}/src" python3 \
        -m mental_llm.config --config "${CONFIG_FILE}" --print-labels >/dev/null
    printf 'Setup smoke check passed without installing packages.\n'
    exit 0
fi

python3 -m venv "${REPOSITORY_ROOT}/.venv"
"${REPOSITORY_ROOT}/.venv/bin/python" -m pip install --upgrade pip
"${REPOSITORY_ROOT}/.venv/bin/python" -m pip install \
    -r "${REPOSITORY_ROOT}/requirements.txt"
"${REPOSITORY_ROOT}/.venv/bin/python" -m pip install --no-deps --editable \
    "${REPOSITORY_ROOT}"
if [[ "$INSTALL_DEV" -eq 1 ]]; then
    "${REPOSITORY_ROOT}/.venv/bin/python" -m pip install \
        -r "${REPOSITORY_ROOT}/requirements-dev.txt"
fi

