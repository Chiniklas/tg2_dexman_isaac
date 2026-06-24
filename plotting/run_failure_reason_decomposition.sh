#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/plots}"
SCALAR_CACHE="${3:-}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dexsafedagger-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/dexsafedagger-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}"

echo "[1/2] Drawing per-object failure-reason curves into: ${OUTPUT_DIR}"
SCALAR_CACHE_ARGS=()
if [[ -n "${SCALAR_CACHE}" ]]; then
  SCALAR_CACHE_ARGS=(--scalar-cache "${SCALAR_CACHE}")
fi
python -u "${SCRIPT_DIR}/plot_multi_obj_curve.py" \
  --config "${CONFIG_PATH}" \
  "${SCALAR_CACHE_ARGS[@]}" \
  --output "${OUTPUT_DIR}"

echo "[2/2] Building failure-reason grid..."
python -u "${SCRIPT_DIR}/concat_pics.py" \
  --config "${CONFIG_PATH}" \
  --input "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/object_metric_grid.png"

echo "Failure-reason figure: ${OUTPUT_DIR}/object_metric_grid.png"
