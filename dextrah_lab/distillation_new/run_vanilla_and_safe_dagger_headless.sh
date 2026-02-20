#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/distillation_new/run_distillation_safedagger.py"
MAX_ITERS="${MAX_ITERS:-100000}"

COMMON_ARGS=(
  --task=dextrah_tg2_inspirehand
  --num_envs 32
  --headless
  --enable_cameras
  --teacher multi_object_distillation
  --imitation_target action_distribution
  --loss_type l2
  --eval_every 2500
  --eval_num_episodes 3
  --max_iterations "${MAX_ITERS}"
  env.distillation=True
  env.simulate_stereo=True
  env.objects_dir=distill_multi_objects
  env.enable_adr=False
)

echo "[1/2] Running vanilla DAgger (unsafe_mode=none) in headless mode..."
python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --unsafe_mode none

echo "[2/2] Running vanilla SafeDAgger (unsafe_mode=l2) in headless mode..."
python "$SCRIPT" \
  "${COMMON_ARGS[@]}" \
  --unsafe_mode l2

echo "Both jobs completed successfully."
