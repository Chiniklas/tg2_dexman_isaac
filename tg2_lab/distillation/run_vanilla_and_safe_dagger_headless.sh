#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/chizhang/projects/dextrah/tg2_dexman_isaac/tg2_lab/distillation/run_distillation_safedagger.py"
MAX_ITERS="${MAX_ITERS:-100000}"
MODE="${MODE:-both}" # dagger | safedagger | both

COMMON_ARGS=(
  --pipeline safedagger
  --task=dextrah_tg2_inspirehand
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

run_dagger() {
  echo "Running vanilla DAgger (unsafe_mode=none) in headless mode..."
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --unsafe_mode none \
    "$@"
}

run_safedagger() {
  echo "Running vanilla SafeDAgger (unsafe_mode=l2) in headless mode..."
  python "$SCRIPT" \
    "${COMMON_ARGS[@]}" \
    --unsafe_mode l2 \
    "$@"
}

case "$MODE" in
  dagger)
    run_dagger "$@"
    ;;
  safedagger)
    run_safedagger "$@"
    ;;
  both)
    run_dagger "$@"
    run_safedagger "$@"
    ;;
  *)
    echo "Invalid MODE='$MODE'. Expected: dagger | safedagger | both"
    exit 1
    ;;
esac

echo "Completed MODE=$MODE."
