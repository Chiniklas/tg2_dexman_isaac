# SimToolReal TG2

This folder contains the active TG2 teacher-training task for this branch.

Contents:

- task registration
- task environment/config code
- SAPO RL-Games YAML recipes

The root entrypoints are:

- `dextrah_lab/train_rl_games.py`
- `dextrah_lab/play_rl_games.py`

Train with the vendored SAPO RL-Games fork:

```bash
conda run -n dexsafedagger python dextrah_lab/train_rl_games.py \
  --task simtoolreal_tg2 \
  --agent_cfg rl_games_sapo_cfg.yaml \
  --num_envs 1024 \
  --headless
```

Replay a checkpoint:

```bash
conda run -n dexsafedagger python dextrah_lab/play_rl_games.py \
  --task simtoolreal_tg2 \
  --checkpoint dextrah_lab/tasks/simtoolreal_tg2/logs/<run>/nn/<checkpoint>.pth
```

The SAPO package used by these scripts is the local copy at
`dextrah_lab/rl_games`.

Current checked-in object support:

- `cube` (`assets/primitives/USD/small_8_cuboid/small_8_cuboid.usd`)
