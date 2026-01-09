#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="pybullet_test"

# Ensure the conda shell functions are available
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH." >&2
  exit 1
fi

eval "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -n "${ENV_NAME}" python=3.11
conda activate "${ENV_NAME}"

# Install pybullet (conda-forge preferred)
conda install -y -c conda-forge pybullet numpy || {
  pip install pybullet numpy
}

echo "Conda env '${ENV_NAME}' created and pybullet installed."
