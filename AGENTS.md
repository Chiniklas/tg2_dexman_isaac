# Repository Guide for Coding Agents

## Scope

This repository implements DexSafeDagger for TG2 + Inspire Hand in Isaac Lab.
Treat this file as durable project context. Keep experiment-specific commands
and temporary observations in `note.txt`, not here.

## Environment

- Repository root:
  `/home/chi-zhang/projects/dexsafedagger/tg2_dexman_isaac`
- Conda environment: `dexsafedagger`
- Main Python package: `dexsafedagger_lab`
- Isaac Sim / Isaac Lab jobs are GPU-heavy. Do not launch full training or
  evaluation unless the user explicitly requests it.
- Generated training outputs live under `dexsafedagger_lab/distillation/runs/`
  and should not be edited as source code.

## Important Entry Points

- Teacher training and replay:
  `dexsafedagger_lab/rl_games/`
- Distillation CLI:
  `dexsafedagger_lab/distillation/scripts/run_distillation_safedagger.py`
- Bash ablation launcher:
  `dexsafedagger_lab/distillation/scripts/run_vanilla_and_safe_dagger_headless.sh`
- Predictor-only warm start:
  `dexsafedagger_lab/distillation/scripts/run_warmstart_predictor.sh`
- Main distillation loop:
  `dexsafedagger_lab/distillation/core/distillation_safedagger.py`
- Warm-start implementation:
  `dexsafedagger_lab/distillation/core/distill_warm_start.py`
- Success critic:
  `dexsafedagger_lab/distillation/safety/success_value_critic.py`
- TG2 environment:
  `dexsafedagger_lab/tasks/tg2_inspirehand/dexsafedagger_tg2_inspirehand_env.py`
- Student/distillation configuration:
  `dexsafedagger_lab/tasks/tg2_inspirehand/agents/rl_games_ppo_stereo_transformer.yaml`
- Common commands and experiment runbook: `note.txt`

## Distillation Variants

- `vanilla_dagger`: no safety intervention gate.
- `vanilla_safedagger`: teacher intervention from student/teacher action L2.
- `dexsafedagger`: L2 gate OR learned low-success gate, with predictor warm
  start.
- `vc_dexsafedagger`: DexSafeDagger plus active VLM threshold tuning.
- `dexsafedaggerUltra`: legacy alias for `vc_dexsafedagger`.

The bash launcher's `VARIANT=all` runs the first three core ablations
sequentially. It intentionally excludes the VLM variant.

## Success-Value Critic

`SuccessValueCritic` is a twin-Q Bellman critic, not a one-step classifier.

- Input observation key is forced to `predictor_transition` during warm start
  and online distillation.
- `predictor_transition` contains noisy robot/hand proprioception and noisy
  object pose.
- The action input is the action actually executed in the environment.
- Replay entries contain `(s, a, s_next, a_next, success, done)`.
- The next action is linked SARSA-style when the following environment step
  arrives.
- Success is the hold-gated `info["lift_success"]` signal.
- `horizon_steps` controls direct positive back-labeling near success.
- `gamma` controls Bellman propagation to earlier transitions.
- Twin target critics use the minimum prediction conservatively.
- Intervention occurs when predicted success is below `success_threshold`.
- In `failure_predictor` mode, the final unsafe mask is:
  `low_success OR teacher_student_l2`.
- After warm-start fitting, `success_threshold` is calibrated to the configured
  low quantile (currently 10th percentile) of critic predictions over safe
  successful teacher rollouts. Episodes containing `out_of_reach` are excluded.
- The calibrated threshold is stored in the critic checkpoint and synchronized
  into the VLM advisor when that checkpoint is reloaded.
- Predictor intervention is delayed for 2,000 online steps. Once active, the
  run fails fast if beta remains at least 0.95 for 500 consecutive steps.

Current checked-in tuning uses `gamma: 0.95` and `horizon_steps: 1`.

The saved June 2026 teacher warm-start datasets reach their first hold-gated
lift success at roughly 74--77 environment steps on average (about 1.25 s at
60 Hz). This is useful context when changing `gamma`.

## Configuration Notes

- Distillation predictor settings originate in the `params.distillation`
  section of `rl_games_ppo_stereo_transformer.yaml`.
- `run_distillation_safedagger.py` loads that YAML and merges its
  `failure_predictor` dictionary into the runtime `dagger_config`.
- The selected variant then controls `unsafe_mode` and whether the predictor
  and VLM advisor are enabled.
- Do not assume arbitrary Hydra command-line overrides modify the student YAML;
  confirm the CLI/config plumbing before documenting an override.
- Runtime-resolved configuration is saved under each run's `params/dagger.yaml`.

## Tests and Lightweight Verification

Run the focused success-critic tests from the repository root:

```bash
pytest -q dexsafedagger_lab/distillation/tests/test_success_value_critic.py
```

For shell changes:

```bash
bash -n dexsafedagger_lab/distillation/scripts/run_vanilla_and_safe_dagger_headless.sh
bash -n dexsafedagger_lab/distillation/scripts/run_warmstart_predictor.sh
```

Before handing off changes:

```bash
git diff --check
git status --short
```

## Editing Practices

- Preserve unrelated user changes in a dirty worktree.
- Prefer targeted changes over broad refactors in the simulator/training path.
- Do not modify checkpoints, run artifacts, generated images, or TensorBoard
  event files.
- When changing success-critic semantics, update focused tests and distinguish
  direct labels (`horizon_steps`) from Bellman propagation (`gamma`).
- Keep `AGENTS.md` concise and stable; update it when architecture or workflow
  facts change.
