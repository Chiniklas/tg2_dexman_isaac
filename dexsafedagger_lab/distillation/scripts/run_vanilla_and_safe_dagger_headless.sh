#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="${SCRIPT_DIR}/run_distillation_safedagger.py"
MAX_ITERS="${MAX_ITERS:-100000}"
VARIANT="${VARIANT:-dexsafedagger}" # vanilla_dagger | vanilla_safedagger | dexsafedagger | dexsafedaggerUltra | all

COMMON_ARGS=(
  --task=dexsafedagger_tg2_inspirehand
  --num_envs 32
  --headless
  --enable_cameras
  --teacher multi_object_distillation
  --eval_every 2500
  --eval_num_episodes 3
  --max_iterations "${MAX_ITERS}"
  env.distillation=True
  env.simulate_stereo=True
  env.objects_dir=distill_multi_objects
  env.enable_adr=False
)

run_variant() {
  local variant="$1"
  shift
  echo "Running ${variant} in headless mode..."
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --variant "${variant}" \
    "$@"
}

case "$VARIANT" in
  vanilla_dagger|vanilla_safedagger|dexsafedagger|dexsafedaggerUltra)
    run_variant "$VARIANT" "$@"
    ;;
  all)
    run_variant vanilla_dagger "$@"
    run_variant vanilla_safedagger "$@"
    run_variant dexsafedagger "$@"
    ;;
  *)
    echo "Invalid VARIANT='$VARIANT'. Expected: vanilla_dagger | vanilla_safedagger | dexsafedagger | dexsafedaggerUltra | all"
    exit 1
    ;;
esac

echo "Completed VARIANT=$VARIANT."
