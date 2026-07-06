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

- `tg2_lab/train_rl_games.py`: root training entrypoint
- `tg2_lab/play_rl_games.py`: root replay entrypoint
- `tg2_lab/tasks/simtoolreal_tg2`: TG2 task code and SAPO YAML configs
- `tg2_lab/rl_games`: vendored SAPO RL-Games project
- `tg2_lab/assets/tg2_inspirehand`: TG2 robot USD/config
- `tg2_lab/assets/test_object`: checked-in SimToolReal object USDs
- `reference/simtoolreal_isaacsim`: local reference snapshot used for alignment

Logs and saved params are written under:

- `tg2_lab/tasks/simtoolreal_tg2/logs`
- `tg2_lab/tasks/simtoolreal_tg2/outputs`

## Environment

This branch assumes Isaac Sim and Isaac Lab are already installed and that the
target Conda environment is already active.

```bash
python -m pip install -e .
```

## Training

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

Use this for a quick startup/layout/reward-geometry sanity check. The object
spawn range, vertical goal range, sparse scene spacing, and table-facing layout
are already in the checked-in config. The exploration block size is overridden
only because this smoke run uses 6 envs instead of the 1536-env training shape.

Full 1536-env training run:

```bash
python tg2_lab/train_rl_games.py \
  --task simtoolreal_tg2 \
  --agent_cfg rl_games_sapo_cfg.yaml \
  --num_envs 1536 \
  --headless
```

Pretrain-like variant:

```bash
python tg2_lab/train_rl_games.py \
  --task simtoolreal_tg2_pretrain_like \
  --agent_cfg rl_games_sapo_pretrain_like_cfg.yaml \
  --num_envs 1536 \
  --headless
```

## Replay

Replay a trained checkpoint:

```bash
python tg2_lab/play_rl_games.py \
  --task simtoolreal_tg2 \
  --checkpoint tg2_lab/tasks/simtoolreal_tg2/logs/<run>/nn/<checkpoint>.pth
```

Optional object override:

```bash
python tg2_lab/play_rl_games.py \
  --task simtoolreal_tg2 \
  --object 1wdf56lx \
  --checkpoint tg2_lab/tasks/simtoolreal_tg2/logs/<run>/nn/<checkpoint>.pth
```

## Tiangong2Pro MuJoCo Asset Prep

The `tg2_lab/assets/tiangong2pro` scaffold now keeps only:

- `urdf/tiangong2.0_pro_with_hands.urdf`
- `xml/tiangong2.0_pro_with_hands.xml`

After `python -m pip install -e .`, you can open the standalone XML directly in
the MuJoCo viewer by loading:

`tg2_lab/assets/tiangong2pro/xml/tiangong2.0_pro_with_hands.xml`

If `mujoco` is missing in the environment, editable install now pulls it in as
part of this branch's package dependencies.

Simple local scene preview:

```bash
python tg2_lab/deployment/mujoco/scene_loader.py
```

Claw-hammer replay smoke scene:

```bash
python tg2_lab/deployment/mujoco/test_single_object_policy_replay.py --headless --steps 240
```

## Current Status

- SimToolReal-style TG2 task is implemented and registered
- root train/play scripts match the reference project structure
- SAPO RL-Games is vendored locally under `tg2_lab/rl_games`
- default TG2 SimToolReal object is DexToolBench `claw_hammer`; checked-in object assets also include `cube` and `1wdf56lx`
- Tiangong2Pro MuJoCo asset scaffold is checked in as one URDF plus one standalone XML

## Internal Notes

Component references used by the current TG2 SimToolReal task:

- Scene creation, parallel environment layout, and TG2 table-facing visual setup take reference from `tg2_lab/tasks/tg2_inspirehand`.
- Robot asset/config conventions take reference from `reference/simtoolreal_isaacsim/simtoolreal_lab/tasks/simtoolreal_sharpa` and its `KUKA_SHARPA_CFG`; TG2 now uses a fixed-head USD reimport with robot self-collision enabled while relying on the converted asset's URDF joint limits.
- Reward terms, keypoint reward structure, goal/object state bookkeeping, and table contact force handling take reference from `simtoolreal_sharpa`.
- SAPO RL-Games training config, asymmetric observation setup, runner wiring, and block-size constraints take reference from the `simtoolreal_sharpa` agent configs.
- Object and goal object spawning, visual debug overlays, and goal-object collision disabling follow the SimToolReal task pattern, with TG2-specific reachability defaults for object spawn center and vertical goal height.
- Local object assets combine the SimToolReal checked-in cube/test object path with DexToolBench `claw_hammer` copied from the reference snapshot.

## Future TG2 Tuning Notes

These are notes for later TG2-InspireHand SimToolReal trials. They are not
currently applied as config changes.

- Start by tuning the task-robot interface before changing SAPO/PPO hyperparameters. The copied KUKA-SHARPA training config is less likely to be the first blocker than reachability, action scaling, gains, and reward geometry.
- Validate the initial arm pose, `object_spawn_center`, `object_spawn_xy_range`, and `goal_height_above_object_range` together. The object and vertical goal should sit in a comfortable TG2 reachable region before widening randomization.
- If the arm is jittery or explores too violently, try smaller `dof_speed_scale` and larger `arm_moving_average`; a first candidate sweep is `dof_speed_scale: 0.5-1.0`, `arm_moving_average: 0.2-0.4`, and `hand_moving_average: 0.15-0.25`.
- If grasping is unstable, inspect actuator gains before changing RL settings. The TG2 asset currently uses SHARPA-like stiffness/damping, which may not be dynamically matched to this robot.
- Keep early reset randomization narrow until the policy can reliably reach, touch, close around, and lift the active object. A possible first easy setting is `object_spawn_xy_range: 0.03` and `goal_height_above_object_range: [0.20, 0.30]`.
- Use fingertip and keypoint debug visualization to check whether the reward geometry actually encourages TG2 fingertips toward useful grasp locations.
- Self-collision is enabled to match the reference SHARPA task more closely. If grasp learning becomes unstable, inspect collision pairs and asset contact geometry before changing SAPO/PPO hyperparameters.

## TODO

Reference behavior gaps to resolve against
`reference/simtoolreal_isaacsim/simtoolreal_lab/tasks/simtoolreal_sharpa`:

1. Restore reference-style 3D target-volume goal sampling, including first-goal random target-volume sampling and post-success `delta` / `coin_flip` goal updates.
2. Restore reference reset randomization for object xy/z pose, object rotation, robot DOF position, robot DOF velocity, and table height.
3. Replace the current TG2 reachability-only `object_spawn_center`, `object_spawn_xy_range`, and vertical `goal_height_above_object_range` path with reference-equivalent behavior, or keep it behind an explicit curriculum/easy-mode flag.
4. Restore pretrain-like object scale randomization: SHARPA pretrain-like uses `object_scale_noise_multiplier_range: [0.9, 1.1]`.
5. Validate TG2 asset self-collision parity. SHARPA enables robot self-collision; TG2 now matches that setting with a fixed-head USD reimport.
6. Port SHARPA regression tests for table-force smoothing, DexToolBench object loading, keypoint geometry, and fixed-size keypoint reward behavior.
7. Add full multi-DexToolBench support only after the single `claw_hammer` path is stable.

Not yet validated in this restructuring pass:

- a fresh full training run after the repo reshuffle
- MuJoCo claw-hammer replay against an exported policy, beyond the local XML smoke scene
