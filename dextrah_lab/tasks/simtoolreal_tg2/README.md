# SimToolReal TG2

This task follows the SimToolReal layout: task code and RL-Games agent recipes
live under this folder, while the training and replay entrypoints live at the
`dextrah_lab` package root.

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
