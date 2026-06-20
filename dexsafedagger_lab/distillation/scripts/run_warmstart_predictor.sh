#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DISTILL_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd -- "${DISTILL_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/run_distillation_safedagger.py"

SMOKE="${SMOKE:-1}"
LEGACY_FULL_SCALE="${LEGACY_FULL_SCALE:-0}"
MAX_SAFE_NUM_ENVS="${MAX_SAFE_NUM_ENVS:-16}"
ALLOW_LARGE_NUM_ENVS="${ALLOW_LARGE_NUM_ENVS:-0}"

TASK="${TASK:-dexsafedagger_tg2_inspirehand}"
VARIANT="${VARIANT:-dexsafedagger}"
TEACHER="${TEACHER:-multi_object_distillation}"
OBJECTS_DIR="${OBJECTS_DIR:-distill_multi_objects}"
ENABLE_CAMERAS="${ENABLE_CAMERAS:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  NUM_ENVS="${NUM_ENVS:-4}"
  WARM_START_COLLECT_STEPS="${WARM_START_COLLECT_STEPS:-50}"
  WARM_START_PREDICTOR_TRAIN_STEPS="${WARM_START_PREDICTOR_TRAIN_STEPS:-10}"
else
  if [[ "${LEGACY_FULL_SCALE}" == "1" ]]; then
    NUM_ENVS="${NUM_ENVS:-32}"
    WARM_START_COLLECT_STEPS="${WARM_START_COLLECT_STEPS:-600}"
    WARM_START_PREDICTOR_TRAIN_STEPS="${WARM_START_PREDICTOR_TRAIN_STEPS:-8000}"
  else
    NUM_ENVS="${NUM_ENVS:-16}"
    WARM_START_COLLECT_STEPS="${WARM_START_COLLECT_STEPS:-200}"
    WARM_START_PREDICTOR_TRAIN_STEPS="${WARM_START_PREDICTOR_TRAIN_STEPS:-1000}"
  fi
fi

MATCH_PREDICTOR_BUFFER_SIZE="${MATCH_PREDICTOR_BUFFER_SIZE:-0}"

print_setting() {
  printf '[warmstart+predictor]   %-28s %s\n' "$1" "$2"
}

echo "[warmstart+predictor] Launch summary"
print_setting "repo" "${REPO_ROOT}"
print_setting "small_scale_mode" "${SMOKE}"
print_setting "num_envs" "${NUM_ENVS}"
print_setting "max_safe_num_envs" "${MAX_SAFE_NUM_ENVS}"
print_setting "collect_steps" "${WARM_START_COLLECT_STEPS}"
print_setting "predictor_updates" "${WARM_START_PREDICTOR_TRAIN_STEPS}"
print_setting "match_predictor_buffer" "${MATCH_PREDICTOR_BUFFER_SIZE}"
print_setting "enable_cameras" "${ENABLE_CAMERAS}"

if (( NUM_ENVS > MAX_SAFE_NUM_ENVS )) && [[ "${ALLOW_LARGE_NUM_ENVS}" != "1" ]]; then
  cat >&2 <<EOF
[warmstart+predictor] Refusing to launch with NUM_ENVS=${NUM_ENVS}.
[warmstart+predictor] Use a smaller NUM_ENVS, or explicitly allow it:
[warmstart+predictor]
[warmstart+predictor]   ALLOW_LARGE_NUM_ENVS=1 NUM_ENVS=${NUM_ENVS} ./scripts/run_warmstart_predictor.sh
[warmstart+predictor]
[warmstart+predictor] For the previous full-scale defaults, also set LEGACY_FULL_SCALE=1.
EOF
  exit 2
fi

echo "[warmstart+predictor] Running warm-start collection and predictor fitting only."

warmstart_cmd=(
  python "${TRAIN_SCRIPT}"
  --pipeline warmstart
  --variant "${VARIANT}"
  --task "${TASK}"
  --num_envs "${NUM_ENVS}"
  --headless
  --teacher "${TEACHER}"
  --eval_every 0
  --eval_num_episodes 0
  --max_iterations 1
  --warm_start_collect_steps "${WARM_START_COLLECT_STEPS}"
  --warm_start_predictor_train_steps "${WARM_START_PREDICTOR_TRAIN_STEPS}"
  env.distillation=True
  env.objects_dir="${OBJECTS_DIR}"
  env.enable_adr=False
)

if [[ "${ENABLE_CAMERAS}" == "1" ]]; then
  warmstart_cmd+=(--enable_cameras env.simulate_stereo=True)
fi
if [[ "${MATCH_PREDICTOR_BUFFER_SIZE}" != "1" ]]; then
  warmstart_cmd+=(--warm_start_no_match_predictor_buffer_size)
fi

warmstart_cmd+=("$@")

(
  cd "${REPO_ROOT}"
  "${warmstart_cmd[@]}"
)

echo "[warmstart+predictor] Done. No VLM dataset export or VLM API benchmark was run."
