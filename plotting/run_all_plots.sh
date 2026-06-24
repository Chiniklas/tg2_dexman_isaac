#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${SCRIPT_DIR}/plots/${TIMESTAMP}"
SCALAR_CACHE="$(mktemp /tmp/dexsafedagger-plot-scalars.XXXXXX.pkl)"
trap 'rm -f "${SCALAR_CACHE}"' EXIT

mkdir -p "${OUTPUT_DIR}"

echo "Plot configuration: ${CONFIG_PATH}"
echo "Timestamped output: ${OUTPUT_DIR}"
echo
echo "=== Shared TensorBoard Load ==="
python -u "${SCRIPT_DIR}/prepare_scalar_cache.py" \
  --config "${CONFIG_PATH}" \
  --output "${SCALAR_CACHE}"

echo
echo "=== Training Curves ==="
"${SCRIPT_DIR}/run_training_curves.sh" \
  "${CONFIG_PATH}" \
  "${OUTPUT_DIR}" \
  "${SCALAR_CACHE}"

echo
echo "=== Failure Reason Decomposition ==="
"${SCRIPT_DIR}/run_failure_reason_decomposition.sh" \
  "${CONFIG_PATH}" \
  "${OUTPUT_DIR}" \
  "${SCALAR_CACHE}"

echo
echo "All plots saved to: ${OUTPUT_DIR}"
