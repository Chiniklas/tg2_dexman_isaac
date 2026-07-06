#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/config.yaml"
CONFIG_PATH_SET=false
SAVE_RAW_DATA=true

usage() {
  cat <<EOF
Usage: $(basename "$0") [CONFIG_PATH] [OPTIONS]

Options:
  --config PATH                 Plot configuration YAML.
  --save-raw-data BOOL         Keep compact plotting data in the timestamped
                               output directory (true by default).
  --no-save-raw-data           Alias for --save-raw-data false.
  -h, --help                   Show this help.
EOF
}

parse_bool() {
  case "${1,,}" in
    true|1|yes|on) echo true ;;
    false|0|no|off) echo false ;;
    *)
      echo "Invalid boolean value: $1" >&2
      return 2
      ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a path." >&2; exit 2; }
      CONFIG_PATH="$2"
      CONFIG_PATH_SET=true
      shift 2
      ;;
    --save-raw-data|--save_raw_data)
      [[ $# -ge 2 ]] || { echo "--save-raw-data requires true or false." >&2; exit 2; }
      SAVE_RAW_DATA="$(parse_bool "$2")"
      shift 2
      ;;
    --save-raw-data=*|--save_raw_data=*)
      SAVE_RAW_DATA="$(parse_bool "${1#*=}")"
      shift
      ;;
    --no-save-raw-data|--no_save_raw_data)
      SAVE_RAW_DATA=false
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ "${CONFIG_PATH_SET}" == true ]]; then
        echo "Only one configuration path may be provided." >&2
        exit 2
      fi
      CONFIG_PATH="$1"
      CONFIG_PATH_SET=true
      shift
      ;;
  esac
done

if [[ $# -gt 0 ]]; then
  echo "Unexpected arguments: $*" >&2
  exit 2
fi

TIMESTAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUTPUT_DIR="${SCRIPT_DIR}/plots/${TIMESTAMP}"
PAPER_FIGURES_DIR="${PAPER_FIGURES_DIR:-${SCRIPT_DIR}/../paper/RAL_dexsafedagger/figures}"

if [[ "${SAVE_RAW_DATA}" == true ]]; then
  RAW_DATA_DIR="${OUTPUT_DIR}/raw_data"
else
  RAW_DATA_DIR="$(mktemp -d /tmp/dexsafedagger-plot-data.XXXXXX)"
  trap 'rm -rf "${RAW_DATA_DIR}"' EXIT
fi

mkdir -p "${OUTPUT_DIR}" "${PAPER_FIGURES_DIR}" "${RAW_DATA_DIR}"

echo "Plot configuration: ${CONFIG_PATH}"
echo "Timestamped output: ${OUTPUT_DIR}"
echo "Save compact raw data: ${SAVE_RAW_DATA}"
echo
echo "=== Exporting Compact Ablation Plot Data ==="
python -u "${SCRIPT_DIR}/export_ablation_raw_data.py" \
  --config "${CONFIG_PATH}" \
  --output "${RAW_DATA_DIR}"

echo
echo "=== Training Curves ==="
"${SCRIPT_DIR}/run_training_curves.sh" \
  "${CONFIG_PATH}" \
  "${OUTPUT_DIR}" \
  "${RAW_DATA_DIR}"
cp "${OUTPUT_DIR}/training_curves_concat.png" \
  "${OUTPUT_DIR}/ablation_comparison.png"

echo
echo "=== Failure Reason Decomposition ==="
"${SCRIPT_DIR}/run_failure_reason_decomposition.sh" \
  "${CONFIG_PATH}" \
  "${OUTPUT_DIR}" \
  "${RAW_DATA_DIR}"

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
if [[ "${SAVE_RAW_DATA}" == true ]]; then
  echo "Compact raw data saved to: ${RAW_DATA_DIR}"
else
  echo "Compact raw data was temporary and has been removed."
fi
