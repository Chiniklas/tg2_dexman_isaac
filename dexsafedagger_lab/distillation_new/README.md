# DexSafeDagger Distillation Pipeline

This folder contains the stereo student distillation pipeline for
`dexsafedagger_tg2_inspirehand`. The main entrypoint is
`run_distillation_safedagger.py`, which builds an Isaac Lab environment,
loads teacher and student RL-Games configs, and runs `SafeDagger` from
`distillation_safedagger.py`.

## Quick Start

Run a headless multi-object SafeDAgger distillation job from this folder:

```bash
cd /home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac/dexsafedagger_lab/distillation_new

python run_distillation_safedagger.py \
  --pipeline safedagger \
  --task dexsafedagger_tg2_inspirehand \
  --num_envs 32 \
  --headless \
  --enable_cameras \
  --teacher multi_object_distillation \
  --unsafe_mode l2 \
  --eval_every 2500 \
  --eval_num_episodes 3 \
  --max_iterations 100000 \
  env.distillation=True \
  env.simulate_stereo=True \
  env.objects_dir=distill_multi_objects \
  env.enable_adr=False
```

The helper script `run_vanilla_and_safe_dagger_headless.sh` can also run
standard DAgger, SafeDAgger, or both:

```bash
MODE=safedagger MAX_ITERS=100000 ./run_vanilla_and_safe_dagger_headless.sh
```

`MODE` can be `dagger`, `safedagger`, or `both`.

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
6. Runs one of the configured training stages:
   `warmstart`, `safedagger`, or `both`.
7. Periodically evaluates the student policy and writes TensorBoard summaries.
8. Saves student checkpoints and run metadata under a timestamped `runs/`
   directory.

## Pipeline Modes

- `--pipeline warmstart`: collects teacher-driven rollouts and performs the
  offline bootstrap stage. The final checkpoint is saved as
  `dexsafedagger_student_after_warmstart.pth`.
- `--pipeline safedagger`: runs online intervention training. The student acts
  by default, and the teacher takes over when the unsafe gate fires.
- `--pipeline both`: runs warm start first, then continues into the online
  SafeDAgger stage.

## Unsafe Modes

`--unsafe_mode` controls when the teacher intervenes:

- `none`: vanilla DAgger-style training without unsafe intervention gating.
- `l2`: teacher intervenes when per-env student/teacher action disagreement
  crosses `unsafe_l2_threshold`.
- `ood`: enables the OOD classifier configured in the student YAML.
- `failure_predictor`: enables the learned state-action risk predictor.

The default distillation settings live in the `params.distillation` section of
`rl_games_ppo_stereo_transformer.yaml`. CLI flags override the most common
values, including `unsafe_mode`, `eval_every`, warm-start collection steps, and
failure predictor warm-start checkpoint path.

## Checkpoints And Outputs

Each run creates:

```text
dexsafedagger_lab/distillation_new/runs/dexsafedagger-tg2-inspirehand-<pipeline>_<day-hour-min-sec>/
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
python run_distillation_safedagger.py \
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
python run_distillation_safedagger.py \
  --pipeline safedagger \
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

For standalone student evaluation, use `eval_student.py`:

```bash
python eval_student.py \
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

- `run_distillation_safedagger.py`: CLI, Isaac app launch, config assembly, run
  directory creation, final checkpoint/export.
- `distillation_safedagger.py`: student/teacher model setup, warm start,
  online intervention loop, unsafe checks, eval loop, checkpoint save/load.
- `distill_warm_start.py`: warm-start rollout collection and offline bootstrap
  support.
- `a2c_stereo_transformer.py`: stereo transformer RL-Games network builder.
- `failure_predictor.py` and `failure_predictor_success_label.py`: learned
  intervention/risk models.
- `ood_classifier.py`: OOD-based intervention model.
- `eval_student.py`: standalone student checkpoint evaluation.
- `replay.py`: student policy replay/recording utility.
