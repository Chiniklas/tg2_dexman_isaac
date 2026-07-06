# SimToolReal TG2

This folder contains the active TG2 teacher-training task for this branch.

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

Current checked-in object support:

- default: `claw_hammer` (`assets/dextoolbench_usd/claw_hammer/claw_hammer.usd`)
- optional: `cube` (`assets/primitives/USD/small_8_cuboid/small_8_cuboid.usd`)
