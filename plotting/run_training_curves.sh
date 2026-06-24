#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
PLOTS_DIR="${2:-${SCRIPT_DIR}/plots}"
SCALAR_CACHE="${3:-}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/dexsafedagger-matplotlib}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/dexsafedagger-cache}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${PLOTS_DIR}"

echo "[1/2] Drawing training curves into: ${PLOTS_DIR}"
SCALAR_CACHE_ARGS=()
if [[ -n "${SCALAR_CACHE}" ]]; then
  SCALAR_CACHE_ARGS=(--scalar-cache "${SCALAR_CACHE}")
fi
python -u "${SCRIPT_DIR}/plot_multi_training_curve.py" \
  --config "${CONFIG_PATH}" \
  "${SCALAR_CACHE_ARGS[@]}" \
  --output "${PLOTS_DIR}/comparison.png"

echo "[2/2] Concatenating training curves..."
python -u "${SCRIPT_DIR}/concat_training_curves.py" \
  --input "${PLOTS_DIR}" \
  --output "${PLOTS_DIR}/training_curves_concat.png"

echo "Training curve figure: ${PLOTS_DIR}/training_curves_concat.png"
