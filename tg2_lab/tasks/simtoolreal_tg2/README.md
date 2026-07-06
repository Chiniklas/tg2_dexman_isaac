# SimToolReal TG2

This folder contains the active TG2 teacher-training task for this branch.

For grasp-failure diagnosis and reference parity, read
[`REFERENCE_COMPARISON.md`](REFERENCE_COMPARISON.md) before changing PPO/SAPO
hyperparameters.

Contents:

- task registration
- task environment/config code
- SAPO RL-Games YAML recipes

The root entrypoints are:

- `tg2_lab/train_rl_games.py`
- `tg2_lab/play_rl_games.py`

Train with the vendored SAPO RL-Games fork:

```bash
python tg2_lab/train_rl_games.py \
  --task simtoolreal_tg2 \
  --agent_cfg rl_games_sapo_cfg.yaml \
  --num_envs 1536 \
  --headless
```

Small visual smoke test:

```bash
python tg2_lab/train_rl_games.py \
  --task simtoolreal_tg2 \
  --agent_cfg rl_games_sapo_cfg.yaml \
  --num_envs 6 \
  --max_iterations 1 \
  --visualize_keypoints \
  --visualize_fingertips \
  agent.params.config.expl_coef_block_size=6 \
  agent.params.config.minibatch_size=96 \
  agent.params.config.central_value_config.minibatch_size=96
```

Replay a checkpoint:

```bash
python tg2_lab/play_rl_games.py \
  --task simtoolreal_tg2 \
  --checkpoint tg2_lab/tasks/simtoolreal_tg2/logs/<run>/nn/<checkpoint>.pth
```

The SAPO package used by these scripts is the local copy at
`tg2_lab/rl_games`.

The active `rl_games_sapo_cfg.yaml` is temporarily configured as a
deterministic grasp-discovery baseline: observation/action delays, object-state
noise, joint-velocity observation noise, and external object disturbances are
disabled. Restore them incrementally only after grasp-and-lift is reliable.

Current checked-in object support:

- default: `claw_hammer` (`assets/dextoolbench_usd/claw_hammer/claw_hammer.usd`)
- optional: `cube` (`assets/primitives/USD/small_8_cuboid/small_8_cuboid.usd`)

## Scripted Physical Closure Test

Before another long RL run, verify that the current hand commands, mimic
mapping, gains, collision geometry, and friction can hold the hammer without a
policy:

```bash
python tg2_lab/tasks/simtoolreal_tg2/tests/test_scripted_hammer_closure.py
```

For a headless pass/fail run:

```bash
python tg2_lab/tasks/simtoolreal_tg2/tests/test_scripted_hammer_closure.py --headless
```

The script verifies that the configured arm pose is palm-up and places the
hammer in the open hand exactly once. The hammer then remains fully dynamic,
with gravity and collision active, while the hand settles, closes, preloads,
and waves. It checks sustained palm-relative drift, drop along the palm support
axis, environment termination, and hand joint tracking.

The default initialization reproduces the demonstrated USD pose in the robot
root frame: translation `(0.4518, -0.32205, 0.09672)` and RPY `(0, 0, 90)`
degrees. The arm pose comes from the palm-up Physics Inspector pose recorded on
2026-07-06. The default closed-hand command reproduces the
demonstrated grasp: four finger joints at `63` degrees, thumb opposition at
`59.2` degrees, and thumb flexion at `14.1` degrees. Tune the object pose visually if
needed:

This configuration passed the complete scripted test on 2026-07-06: 300/300
hold-and-wave steps, `0.0002 m` maximum palm-relative drift, and no termination.

```bash
python tg2_lab/tasks/simtoolreal_tg2/tests/test_scripted_hammer_closure.py \
  --fixture-mode robot \
  --object-local-offset 0.4518,-0.32205,0.09672 \
  --object-local-rpy-deg 0,0,90 \
  --debug-fingertips \
  --debug-grasp-bounding-box
```

Use `--fixture-mode training` to test the checked-in training placement, or
`--fixture-mode palm` with explicit palm-relative values. Exit status is `0`
for pass and `2` for a completed physical wave/hold failure.
