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
   experimental `dexsafedaggerUltra` scaffold.
7. Periodically evaluates the student policy and writes TensorBoard summaries.
8. Saves student checkpoints and run metadata under a timestamped `runs/`
   directory.

## Directory Layout

- `scripts/`: training and replay entrypoints.
- `core/`: SafeDAgger loop and warm-start orchestration.
- `models/`: RL-Games stereo transformer and teacher network builders.
- `safety/`: failure predictor and experimental VLM intervention scaffold.
- `utils/`: shared losses, eval metrics, augmentations, and data recording.
- `eval/`: standalone student evaluation.

## Distillation Variants

`--variant` selects the paper terminology directly:

- `vanilla_dagger`: student rollout with teacher labels. No intervention gate.
- `vanilla_safedagger`: teacher takeover when student/teacher action
  disagreement crosses `unsafe_l2_threshold`.
- `dexsafedagger`: the full method from the paper. It runs warm-start first,
  then online SafeDAgger with the OR gate combining action disagreement and the
  learned critic-style risk predictor.
- `dexsafedaggerUltra`: brainstorming-stage ablation scaffold. The intended
  idea is to replace fixed-threshold intervention decisions with VLM-predicted
  teacher intervention points from visual/context observations. This is not a
  runnable method yet; selecting it enables `unsafe_mode=vlm_intervention`,
  which currently raises until the VLM backend, prompting, frame extraction, and
  temporal smoothing policy are implemented.

For `dexsafedaggerUltra`, the starter structure is:

- `safety/vlm_intervention.py`: future VLM intervention planner.
- `params.distillation.vlm_intervention`: provider/model/prompt and smoothing
  config stub.
- `SafeDagger.check_unsafe(..., unsafe_mode="vlm_intervention")`: placeholder
  branch where the VLM unsafe mask will replace threshold-based intervention.

The default distillation settings live in the `params.distillation` section of
`rl_games_ppo_stereo_transformer.yaml`. CLI flags override the most common
values, including `variant`, `eval_every`, warm-start collection steps, and
failure predictor warm-start checkpoint path.

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
- `core/distillation_safedagger.py`: student/teacher model setup, warm start,
  online intervention loop, unsafe checks, eval loop, checkpoint save/load.
- `core/distill_warm_start.py`: warm-start rollout collection and offline bootstrap
  support.
- `models/a2c_stereo_transformer.py`: stereo transformer RL-Games network builder.
- `safety/failure_predictor.py`: learned critic-style intervention/risk model.
- `safety/vlm_intervention.py`: scaffold for the experimental DexSafeDaggerUltra VLM
  intervention gate.
- `eval/eval_student.py`: standalone student checkpoint evaluation.
- `scripts/replay.py`: student policy replay/recording utility.
