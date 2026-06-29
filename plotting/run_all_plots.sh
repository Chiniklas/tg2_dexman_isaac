#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${1:-${SCRIPT_DIR}/config.yaml}"
TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${SCRIPT_DIR}/plots/${TIMESTAMP}"
PAPER_FIGURES_DIR="${PAPER_FIGURES_DIR:-${SCRIPT_DIR}/../paper/RAL_dexsafedagger/figures}"
SCALAR_CACHE="$(mktemp /tmp/dexsafedagger-plot-scalars.XXXXXX.pkl)"
trap 'rm -f "${SCALAR_CACHE}"' EXIT

mkdir -p "${OUTPUT_DIR}" "${PAPER_FIGURES_DIR}"

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
cp "${OUTPUT_DIR}/training_curves_concat.png" \
  "${OUTPUT_DIR}/ablation_comparison.png"

echo
echo "=== Failure Reason Decomposition ==="
"${SCRIPT_DIR}/run_failure_reason_decomposition.sh" \
  "${CONFIG_PATH}" \
  "${OUTPUT_DIR}" \
  "${SCALAR_CACHE}"

echo
echo "=== Final Failure Reason Figure ==="
python -u "${SCRIPT_DIR}/concat_failure_mode_figure.py" \
  --sources "${SCRIPT_DIR}/sources" \
  --grid "${OUTPUT_DIR}/object_metric_grid.png" \
  --output "${OUTPUT_DIR}/failure_mode.png" \
  --pdf-output "${OUTPUT_DIR}/failure_mode.pdf"

echo
echo "=== Publishing Paper Figure Names ==="
cp "${OUTPUT_DIR}/ablation_comparison.png" \
  "${PAPER_FIGURES_DIR}/ablation_comparison.png"
cp "${OUTPUT_DIR}/object_metric_grid.png" \
  "${PAPER_FIGURES_DIR}/object_metric_grid.png"
cp "${OUTPUT_DIR}/failure_mode.png" \
  "${PAPER_FIGURES_DIR}/failure_mode.png"
cp "${OUTPUT_DIR}/failure_mode.pdf" \
  "${PAPER_FIGURES_DIR}/failure_mode.pdf"
echo "Paper figures updated in: ${PAPER_FIGURES_DIR}"

echo
echo "All plots saved to: ${OUTPUT_DIR}"
