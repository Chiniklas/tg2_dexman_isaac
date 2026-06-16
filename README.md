# TG2 SimToolReal SAPO Branch

This branch is an Isaac Lab training workspace centered on a SimToolReal-style
teacher policy for the TG2 InspireHand asset.

What is in scope here:

- SAPO-style RL-Games teacher training
- TG2 SimToolReal task implementation
- Local vendored SAPO `rl_games`
- Replay of trained teacher checkpoints in Isaac Sim

This branch now uses the SimToolReal-style root entrypoints and TG2 task as the
active training path, while older Dextrah assets, deployment code, plotting,
stored checkpoints, and local reference material are still present in the tree.

## Layout

- `dextrah_lab/train_rl_games.py`: root training entrypoint
- `dextrah_lab/play_rl_games.py`: root replay entrypoint
- `dextrah_lab/tasks/simtoolreal_tg2`: TG2 task code and SAPO YAML configs
- `dextrah_lab/rl_games`: vendored SAPO RL-Games project
- `dextrah_lab/assets/tg2_inspirehand`: TG2 robot USD/config
- `dextrah_lab/assets/test_object`: checked-in SimToolReal object USDs
- `reference/simtoolreal_isaacsim`: local reference snapshot used for alignment

Logs and saved params are written under:

- `dextrah_lab/tasks/simtoolreal_tg2/logs`
- `dextrah_lab/tasks/simtoolreal_tg2/outputs`

## Environment

This branch assumes Isaac Sim and Isaac Lab are already installed in the target
Conda environment, typically `dexsafedagger`.

```bash
python -m pip install -e .
```

## Training

Train the TG2 SimToolReal task with the default SAPO config:

```bash
conda run -n dexsafedagger python dextrah_lab/train_rl_games.py \
  --task simtoolreal_tg2 \
  --agent_cfg rl_games_sapo_cfg.yaml \
  --num_envs 1536 \
  --headless
```

This matches the reference SAPO six-block setup:

```text
num_envs / expl_coef_block_size = 1536 / 256 = 6
```

The TG2 SAPO config already carries `expl_coef_block_size: 256`, so the short
command above is the reference-shaped command for this branch. Using `1024`
would only create 4 exploration groups.

Pretrain-like variant:

```bash
conda run -n dexsafedagger python dextrah_lab/train_rl_games.py \
  --task simtoolreal_tg2_pretrain_like \
  --agent_cfg rl_games_sapo_pretrain_like_cfg.yaml \
  --num_envs 1536 \
  --headless
```

Small smoke test:

```bash
conda run -n dexsafedagger python dextrah_lab/train_rl_games.py \
  --task simtoolreal_tg2 \
  --agent_cfg rl_games_sapo_cfg.yaml \
  --num_envs 6 \
  --max_iterations 1 \
  --headless
```

Use the 6-env command only as a startup/sanity smoke test. It does not preserve
the reference six-block SAPO exploration shape, so checkpoints from this tiny
run should not be treated as replay-compatible training outputs.

## Replay

Replay a trained checkpoint:

```bash
conda run -n dexsafedagger python dextrah_lab/play_rl_games.py \
  --task simtoolreal_tg2 \
  --checkpoint dextrah_lab/tasks/simtoolreal_tg2/logs/<run>/nn/<checkpoint>.pth
```

Optional object override:

```bash
conda run -n dexsafedagger python dextrah_lab/play_rl_games.py \
  --task simtoolreal_tg2 \
  --object 1wdf56lx \
  --checkpoint dextrah_lab/tasks/simtoolreal_tg2/logs/<run>/nn/<checkpoint>.pth
```

## Tiangong2Pro MuJoCo Asset Prep

The `dextrah_lab/assets/tiangong2pro` scaffold now keeps only:

- `urdf/tiangong2.0_pro_with_hands.urdf`
- `xml/tiangong2.0_pro_with_hands.xml`

After `python -m pip install -e .`, you can open the standalone XML directly in
the MuJoCo viewer by loading:

`dextrah_lab/assets/tiangong2pro/xml/tiangong2.0_pro_with_hands.xml`

If `mujoco` is missing in the environment, editable install now pulls it in as
part of this branch's package dependencies.

Simple local scene preview:

```bash
python dextrah_lab/deployment_ros2/mujoco/scene_loader.py
```

## Current Status

- SimToolReal-style TG2 task is implemented and registered
- root train/play scripts match the reference project structure
- SAPO RL-Games is vendored locally under `dextrah_lab/rl_games`
- default checked-in object support includes `cube` and `1wdf56lx`
- Tiangong2Pro MuJoCo asset scaffold is checked in as one URDF plus one standalone XML

Not yet validated in this restructuring pass:

- a fresh full training run after the repo reshuffle
- multi-object expansion beyond the checked-in test object set
