# DexSafeDagger Distillation Pipeline

This folder contains the stereo student distillation pipeline for
`dexsafedagger_tg2_inspirehand`. The main entrypoint is
`scripts/run_distillation_safedagger.py`, which builds an Isaac Lab environment,
loads teacher and student RL-Games configs, and runs `SafeDagger` from
`core/distillation_safedagger.py`.

## Quick Start

Run the paper's full DexSafeDagger variant from this folder:

```bash
cd /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation

python scripts/run_distillation_safedagger.py \
  --variant dexsafedagger \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --headless \
  --enable_cameras \
  --teacher multi_object_distillation \
  --eval_every 2500 \
  --eval_num_episodes 3 \
  --max_iterations 100000 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

The helper script `run_vanilla_and_safe_dagger_headless.sh` can select any
registered variant:

```bash
VARIANT=dexsafedagger MAX_ITERS=100000 ./scripts/run_vanilla_and_safe_dagger_headless.sh
```

`VARIANT` can be `vanilla_dagger`, `vanilla_safedagger`,
`dexsafedagger`, experimental `dexsafedaggerUltra`, or `all`.

## What The Pipeline Does

1. Launches Isaac Lab with the requested task and Hydra overrides.
2. Requires stereo simulation (`env.simulate_stereo=True`) and uses the
   stereo transformer student policy config:
   `dexsafedagger_lab/tasks/tg2_inspirehand/agents/rl_games_ppo_stereo_transformer.yaml`.
3. Loads the teacher policy config from:
   `dexsafedagger_lab/tasks/tg2_inspirehand/agents/rl_games_ppo_lstm_cfg.yaml`.
4. Resolves teacher checkpoints under `pretrained_ckpts/`. For example,
   `--teacher multi_object_distillation` resolves to:
   `pretrained_ckpts/multi_object_distillation`.
5. Builds the student and teacher networks through RL-Games model builders.
6. Runs one of the supported variants:
   `vanilla_dagger`, `vanilla_safedagger`, `dexsafedagger`, or the
   experimental `dexsafedaggerUltra` threshold-advisor ablation.
7. Periodically evaluates the student policy and writes TensorBoard summaries.
8. Saves student checkpoints and run metadata under a timestamped `runs/`
   directory.

## Directory Layout

- `scripts/`: training and replay entrypoints.
- `core/`: SafeDAgger loop and warm-start orchestration.
- `models/`: RL-Games stereo transformer and teacher network builders.
- `safety/`: success-value critic and optional VLM threshold advisor.
- `utils/`: shared losses, eval metrics, augmentations, and data recording.
- `eval/`: standalone student evaluation.

## Distillation Variants

`--variant` selects the paper terminology directly:

- `vanilla_dagger`: student rollout with teacher labels. No intervention gate.
- `vanilla_safedagger`: teacher takeover when student/teacher action
  disagreement crosses `unsafe_l2_threshold`.
- `dexsafedagger`: the full method from the paper. It runs warm-start first,
  then online SafeDAgger with the OR gate combining action disagreement and the
  learned success-value critic.
- `vc_dexsafedagger`: VC-DexSafeDAgger, combining the SafeDAgger disagreement
  gate, success-value critic, and active VLM threshold-tuner pipeline.
- `dexsafedaggerUltra`: legacy alias for `vc_dexsafedagger`. It keeps the same
  `failure_predictor` arbitration as DexSafeDagger and only enables the VLM
  threshold advisor. The VLM recommends smoothed/clamped values for
  `unsafe_l2_threshold` and the predictor success threshold; it does not replace
  the OR gate or directly decide teacher takeover.
  Whenever the VLM advisor is enabled, the success-value critic is enabled by
  default so predicted-success statistics are included in advisor inputs.

The VLM advisor maintains a moderate runtime visual buffer by default: up to 64
downscaled frames, capturing 2 representative frames every 20 online steps and
attaching at most 6 images to each advisor request. Samples are selected from
high teacher-student disagreement, low predicted success, and unsafe-triggered
states, while the VLM still only returns threshold recommendations.
During warm-start, VLM-enabled runs also seed this buffer with compact images
from unsafe warm-start terminal states, so the first online advisor call can
see prior unsafe visual examples instead of starting from an empty buffer.

The default distillation settings live in the `params.distillation` section of
`rl_games_ppo_stereo_transformer.yaml`. CLI flags override the most common
values, including `variant`, `eval_every`, warm-start collection steps, and
success-critic warm-start checkpoint path.

## Checkpoints And Outputs

Each run creates:

```text
dexsafedagger_lab/distillation/runs/dexsafedagger-tg2-inspirehand-<variant>_<day-hour-min-sec>/
```

Inside that folder:

- `nn/`: student checkpoints.
- `summaries/`: TensorBoard logs.
- `params/`: saved environment config, student/teacher config snapshots,
  resolved DAgger config, CLI args, and Hydra overrides.
- `final_eval_metrics.json`: final exported eval metrics when an inline eval
  snapshot exists.

During online training, intermediate checkpoints are saved every 5000
iterations as `nn/dexsafedagger_student_<iter>_iters.pth`. At the end of a
SafeDAgger or `both` run, the final student checkpoint is saved as
`nn/dexsafedagger_student_safe_dagger.pth`. Depending on the RL-Games save helper,
the file may appear on disk with an extra extension:
`dexsafedagger_student_safe_dagger.pth.pth`.

## Teachers And Multi-Object Runs

For multi-object distillation, pass a directory of teacher checkpoints:

```bash
--teacher multi_object_distillation
```

The code resolves this relative to the repo root:

```text
pretrained_ckpts/multi_object_distillation/
```

Each object directory should contain a teacher checkpoint such as
`dexsafedagger_lstm.pth`. If object names and checkpoint folder names do not match,
provide an explicit object-to-teacher mapping with `--teacher_object_map`.
The value can be a JSON/YAML file path or inline JSON.

Example:

```bash
python scripts/run_distillation_safedagger.py \
  --variant dexsafedagger \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --headless \
  --enable_cameras \
  --teacher multi_object_distillation \
  --teacher_object_map /path/to/object_teacher_map.yaml \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects
```

## Resuming Or Warm-Starting A Student

Pass `--student` to initialize the student from an existing checkpoint. Absolute
paths are used directly. Relative paths are resolved under `pretrained_ckpts/`.

```bash
python scripts/run_distillation_safedagger.py \
  --variant vanilla_safedagger \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --headless \
  --enable_cameras \
  --teacher multi_object_distillation \
  --student path_or_name_of_student_checkpoint.pth \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects
```

## Evaluation

Inline evaluation is controlled by:

```bash
--eval_every 2500
--eval_num_episodes 3
--eval_objects_dir optional_eval_objects_folder
```

For standalone student evaluation, use `eval/eval_student.py`:

```bash
python eval/eval_student.py \
  --headless \
  --enable_cameras \
  --task dexsafedagger_tg2_inspirehand \
  --checkpoint /absolute/path/to/student_checkpoint.pth \
  --objects_dir distill_multi_objects \
  --num_envs 32 \
  --num_episodes 3 \
  env.distillation=True \
  env.enable_adr=False
```

## Useful Files

- `scripts/run_distillation_safedagger.py`: CLI, Isaac app launch, config assembly, run
  directory creation, final checkpoint/export.
- `scripts/run_warmstart_predictor.sh`: warm-start data collection plus failure
  predictor fitting only; no VLM export or VLM API calls.
- `core/distillation_safedagger.py`: student/teacher model setup, warm start,
  online intervention loop, unsafe checks, eval loop, checkpoint save/load.
- `core/distill_warm_start.py`: warm-start rollout collection and offline bootstrap
  support.
- `models/a2c_stereo_transformer.py`: stereo transformer RL-Games network builder.
- `safety/success_value_critic.py`: learned Bellman success-value critic.
- `safety/vlm_threshold_advisor.py`: optional VLM advisor that recommends
  smoothed/clamped L2 and predicted-success thresholds while leaving existing
  arbitration logic in place.
- `eval/eval_student.py`: standalone student checkpoint evaluation.
- `scripts/replay.py`: student policy replay/recording utility.
