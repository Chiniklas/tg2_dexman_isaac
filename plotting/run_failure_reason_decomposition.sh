#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/plots}"
RAW_DATA_DIR="${3:-${OUTPUT_DIR}/raw_data}"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dexsafedagger-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/dexsafedagger-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OUTPUT_DIR}"

echo "[1/2] Drawing per-object failure-reason curves into: ${OUTPUT_DIR}"
python -u "${SCRIPT_DIR}/plot_multi_obj_curve.py" \
  --config "${CONFIG_PATH}" \
  --raw-data-dir "${RAW_DATA_DIR}" \
  --output "${OUTPUT_DIR}"

echo "[2/2] Building failure-reason grid..."
python -u "${SCRIPT_DIR}/concat_pics.py" \
  --config "${CONFIG_PATH}" \
  --input "${OUTPUT_DIR}" \
  --output "${OUTPUT_DIR}/object_metric_grid.png"

echo "Failure-reason figure: ${OUTPUT_DIR}/object_metric_grid.png"
