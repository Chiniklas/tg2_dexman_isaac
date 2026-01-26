#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Configurable settings (override via environment variables).
TASK="${TASK:-dextrah_tg2_inspirehand}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-16}"
HEADLESS="${HEADLESS:---headless}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-64}"
CV_MINIBATCH_SIZE="${CV_MINIBATCH_SIZE:-64}"
LEARNING_RATE="${LEARNING_RATE:-0.0001}"
HORIZON_LENGTH="${HORIZON_LENGTH:-16}"
MINI_EPOCHS="${MINI_EPOCHS:-4}"
MULTI_GPU="${MULTI_GPU:-False}"
WANDB_ACTIVATE="${WANDB_ACTIVATE:-False}"
SUCCESS_FOR_ADR="${SUCCESS_FOR_ADR:-0.4}"
USE_CUDA_GRAPH="${USE_CUDA_GRAPH:-False}"
MULTI_OBJECTS_ROOT="${MULTI_OBJECTS_ROOT:-${ROOT_DIR}/assets/multi_objects}"
MULTI_OBJECTS_DIR="${MULTI_OBJECTS_ROOT}/USD"
SINGLE_DIR_NAME="${SINGLE_DIR_NAME:-_single_object}"

MULTI_USD_DIR="${MULTI_OBJECTS_DIR}"
SINGLE_DIR="${ROOT_DIR}/assets/${SINGLE_DIR_NAME}"
SINGLE_USD_DIR="${SINGLE_DIR}/USD"

if [[ ! -d "${MULTI_USD_DIR}" ]]; then
  echo "Missing multi_objects USD dir: ${MULTI_USD_DIR}" >&2
  exit 1
fi

mkdir -p "${SINGLE_USD_DIR}"

for obj_dir in "${MULTI_USD_DIR}"/*; do
  [[ -d "${obj_dir}" ]] || continue
  obj_name="$(basename "${obj_dir}")"

  rm -rf "${SINGLE_USD_DIR:?}"/*
  ln -s "${obj_dir}" "${SINGLE_USD_DIR}/${obj_name}"

  echo "=== Training object: ${obj_name} ==="
  python train.py \
    --task="${TASK}" \
    --seed "${SEED}" \
    --num_envs "${NUM_ENVS}" \
    ${HEADLESS} \
    agent.params.config.full_experiment_name="$(date +%Y%m%d_%H%M%S)_${obj_name}" \
    agent.params.config.minibatch_size="${MINIBATCH_SIZE}" \
    agent.params.config.central_value_config.minibatch_size="${CV_MINIBATCH_SIZE}" \
    agent.params.config.learning_rate="${LEARNING_RATE}" \
    agent.params.config.horizon_length="${HORIZON_LENGTH}" \
    agent.params.config.mini_epochs="${MINI_EPOCHS}" \
    agent.params.config.multi_gpu="${MULTI_GPU}" \
    agent.wandb_activate="${WANDB_ACTIVATE}" \
    env.success_for_adr="${SUCCESS_FOR_ADR}" \
    env.objects_dir="${SINGLE_DIR_NAME}" \
    env.use_cuda_graph="${USE_CUDA_GRAPH}"

done
