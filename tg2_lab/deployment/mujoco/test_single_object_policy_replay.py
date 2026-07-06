from __future__ import annotations

# Run from repo root:
# python tg2_lab/deployment/mujoco/test_single_object_policy_replay.py
#
# Headless smoke check:
# python tg2_lab/deployment/mujoco/test_single_object_policy_replay.py --headless --steps 240

import argparse
import copy
import math
import sys
import time
import types
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEXTRAH_ROOT = REPO_ROOT / "tg2_lab"
VENDORED_RL_GAMES = DEXTRAH_ROOT / "rl_games"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if VENDORED_RL_GAMES.is_dir() and str(VENDORED_RL_GAMES) not in sys.path:
    sys.path.insert(0, str(VENDORED_RL_GAMES))

DEFAULT_SCENE_XML = (
    REPO_ROOT
    / "tg2_lab"
    / "assets"
    / "tiangong2pro"
    / "mujoco_scene"
    / "claw_hammer_scene.xml"
)
INIT_QPOS = {
    "head_yaw_joint": 0.0,
    "head_pitch_joint": 0.0,
    "head_roll_joint": 0.0,
    "shoulder_pitch_r_joint": -1.570796,
    "shoulder_roll_r_joint": -0.523599,
    "shoulder_yaw_r_joint": 1.108284,
    "elbow_pitch_r_joint": -1.275836,
    "elbow_yaw_r_joint": 0.089012,
    "wrist_pitch_r_joint": -0.027925,
    "wrist_roll_r_joint": -0.048869,
    "index_joint_0": 0.0,
    "index_joint_1": 0.0,
    "middle_joint_0": 0.0,
    "middle_joint_1": 0.0,
    "ring_joint_0": 0.0,
    "ring_joint_1": 0.0,
    "little_joint_0": 0.0,
    "little_joint_1": 0.0,
    "thumb_joint_0": 0.4,
    "thumb_joint_1": 0.1,
    "thumb_joint_2": 0.2,
    "thumb_joint_3": 0.4,
}
POLICY_JOINT_NAMES = [
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "elbow_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
    "index_joint_0",
    "middle_joint_0",
    "ring_joint_0",
    "little_joint_0",
    "thumb_joint_0",
    "thumb_joint_1",
]
FINGERTIP_BODIES = [
    "index_link_1",
    "middle_link_1",
    "ring_link_1",
    "little_link_1",
    "thumb_link_3",
]
FINGERTIP_LOCAL_OFFSETS = {
    "index_link_1": (0.0, 0.038, 0.003),
    "middle_link_1": (0.0, 0.040, 0.003),
    "ring_link_1": (0.0, 0.038, 0.003),
    "little_link_1": (0.0, 0.032, 0.003),
    "thumb_link_3": (-0.018, 0.016, 0.003),
}
KEYPOINT_SITE_NAMES = [f"hammer_keypoint_{idx}" for idx in range(4)]
GOAL_KEYPOINT_SITE_NAMES = [f"goal_keypoint_{idx}" for idx in range(4)]
DEFAULT_OBJECT_SCALES = (2.5, 0.5625, 0.375)
OBS_SIZE = 92
ACTION_SIZE = 13
DEFAULT_VIEWER_LOOKAT = (0.0, 0.55, 0.78)
DEFAULT_VIEWER_DISTANCE = 1.75
DEFAULT_VIEWER_AZIMUTH = -90.0
DEFAULT_VIEWER_ELEVATION = -18.0


class _DummyRlGamesEnv:
    num_envs = 1

    def set_env_state(self, env_state):
        return None


def _require_mujoco():
    try:
        import mujoco  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("MuJoCo is required for this replay smoke test.") from exc
    return mujoco


def _maybe_import_viewer():
    try:
        import mujoco.viewer  # type: ignore
    except ModuleNotFoundError:
        return None
    return mujoco.viewer


def _torch_load(path: Path, device: str) -> Any:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Torch is required when --checkpoint is provided.") from exc

    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _yaml_load(path: Path) -> dict:
    import yaml

    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_gym_module():
    try:
        import gymnasium as gym

        sys.modules.setdefault("gym", gym)
        return gym
    except ModuleNotFoundError:
        pass

    try:
        import gym

        return gym
    except ModuleNotFoundError:
        pass

    import numpy as np

    class Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.dtype = dtype
            if shape is None:
                shape = np.shape(low) if np.shape(low) else np.shape(high)
            self.shape = tuple(shape)
            self.low = np.full(self.shape, low, dtype=dtype)
            self.high = np.full(self.shape, high, dtype=dtype)

    class Discrete:
        def __init__(self, n):
            self.n = int(n)
            self.shape = ()

    class Tuple:
        def __init__(self, spaces):
            self.spaces = tuple(spaces)
            self.shape = tuple(space.shape for space in self.spaces)

    class Dict:
        def __init__(self, spaces):
            self.spaces = dict(spaces)

    class Env:
        pass

    class Wrapper(Env):
        def __init__(self, env=None):
            self.env = env

    class RewardWrapper(Wrapper):
        pass

    class ObservationWrapper(Wrapper):
        pass

    class ActionWrapper(Wrapper):
        pass

    def _register(*args, **kwargs):
        return None

    spaces = types.ModuleType("gym.spaces")
    spaces.Box = Box
    spaces.Discrete = Discrete
    spaces.Tuple = Tuple
    spaces.Dict = Dict
    spaces.dict = types.SimpleNamespace(Dict=Dict)

    wrappers = types.ModuleType("gym.wrappers")
    wrappers.FlattenObservation = Wrapper
    wrappers.FilterObservation = Wrapper

    gym = types.ModuleType("gym")
    gym.spaces = spaces
    gym.envs = types.SimpleNamespace(register=_register)
    gym.Env = Env
    gym.Wrapper = Wrapper
    gym.RewardWrapper = RewardWrapper
    gym.ObservationWrapper = ObservationWrapper
    gym.ActionWrapper = ActionWrapper
    gym.make = lambda *args, **kwargs: None
    gym.wrappers = wrappers
    sys.modules["gym"] = gym
    sys.modules["gym.spaces"] = spaces
    sys.modules["gym.wrappers"] = wrappers
    return gym


def _checkpoint_params_dir(checkpoint_path: Path) -> Path | None:
    for parent in checkpoint_path.resolve().parents:
        params_dir = parent / "params"
        if (params_dir / "agent.yaml").is_file():
            return params_dir
    return None


def _load_agent_cfg(checkpoint_path: Path) -> dict:
    params_dir = _checkpoint_params_dir(checkpoint_path)
    if params_dir is not None:
        return _yaml_load(params_dir / "agent.yaml")
    cfg_path = (
        REPO_ROOT
        / "tg2_lab"
        / "tasks"
        / "simtoolreal_tg2"
        / "agents"
        / "rl_games_sapo_cfg.yaml"
    )
    return _yaml_load(cfg_path)


def _checkpoint_coef_id_count(checkpoint_path: Path, device: str) -> int | None:
    checkpoint = _torch_load(checkpoint_path, device)
    if isinstance(checkpoint, dict) and 0 in checkpoint:
        checkpoint = checkpoint[0]
    model_state = checkpoint.get("model", {}) if isinstance(checkpoint, dict) else {}
    for name in ("a2c_network.extra_params", "a2c_network.sigma"):
        weight = model_state.get(name)
        if weight is not None and getattr(weight, "ndim", 0) >= 2:
            return int(weight.shape[0])
    return None


def _make_policy_player(checkpoint_path: Path, device: str):
    gym = _ensure_gym_module()
    import torch
    from rl_games.algos_torch import torch_ext
    from rl_games.common.player import BasePlayer
    from rl_games.torch_runner import Runner

    agent_cfg = _load_agent_cfg(checkpoint_path)
    agent_cfg = copy.deepcopy(agent_cfg)
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = str(checkpoint_path)
    agent_cfg["params"]["config"]["device"] = device
    agent_cfg["params"]["config"]["device_name"] = device
    agent_cfg["params"]["config"]["num_actors"] = 1
    agent_cfg["params"]["config"]["vec_env"] = _DummyRlGamesEnv()
    agent_cfg["params"]["config"]["env_info"] = {
        "action_space": gym.spaces.Box(low=-1.0, high=1.0, shape=(ACTION_SIZE,), dtype=float),
        "observation_space": gym.spaces.Box(low=-math.inf, high=math.inf, shape=(OBS_SIZE,), dtype=float),
        "state_space": gym.spaces.Box(low=-math.inf, high=math.inf, shape=(114,), dtype=float),
        "value_size": 1,
    }
    coef_id_count = _checkpoint_coef_id_count(checkpoint_path, device)
    if coef_id_count is not None:
        agent_cfg["params"]["config"].setdefault("player", {})["coef_id_count"] = coef_id_count

    runner = Runner()
    runner.load(agent_cfg)
    player: BasePlayer = runner.create_player()
    checkpoint = torch_ext.load_checkpoint(str(checkpoint_path))
    if isinstance(checkpoint, dict) and 0 in checkpoint:
        checkpoint = checkpoint[0]
    player.model.load_state_dict(checkpoint["model"])
    if player.normalize_input and "running_mean_std" in checkpoint:
        player.model.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
    player.loaded_checkpoint = str(checkpoint_path)
    player.reset()
    player.has_batch_dimension = True
    player.batch_size = 1
    return player


def _summarize_checkpoint(path: Path, payload: Any) -> None:
    print(f"Loaded checkpoint: {path}")
    if isinstance(payload, dict):
        keys = list(payload.keys())
        print(f"  type: dict, keys: {keys[:16]}{' ...' if len(keys) > 16 else ''}")
        if 0 in payload and isinstance(payload[0], dict):
            payload = payload[0]
            keys = list(payload.keys())
            print(f"  rank-0 keys: {keys[:16]}{' ...' if len(keys) > 16 else ''}")
        for key in ("model", "state_dict", "running_mean_std", "optimizer", "epoch", "frame"):
            if key in payload:
                value = payload[key]
                if isinstance(value, dict):
                    print(f"  {key}: dict[{len(value)}]")
                else:
                    print(f"  {key}: {type(value).__name__}")
    else:
        print(f"  type: {type(payload).__name__}")


def _collect_checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    if args.checkpoint:
        paths.extend(Path(path).expanduser().resolve() for path in args.checkpoint)
    if args.checkpoint_dir:
        checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
        paths.extend(sorted(checkpoint_dir.glob("*.pth")))
        paths.extend(sorted(checkpoint_dir.glob("*.pt")))

    seen: set[Path] = set()
    unique_paths: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def _set_joint_qpos(mujoco, model, data, joint_name: str, value: float) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_addr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_addr] = value


def _get_joint_qpos(mujoco, model, data, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Joint not found: {joint_name}")
    return float(data.qpos[model.jnt_qposadr[joint_id]])


def _get_joint_qvel(mujoco, model, data, joint_name: str) -> float:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Joint not found: {joint_name}")
    return float(data.qvel[model.jnt_dofadr[joint_id]])


def _joint_range(mujoco, model, joint_name: str) -> tuple[float, float]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise KeyError(f"Joint not found: {joint_name}")
    lo, hi = model.jnt_range[joint_id]
    return float(lo), float(hi)


def _name_to_id(mujoco, model, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise KeyError(f"MuJoCo object not found: {name}")
    return obj_id


def _set_actuator_ctrl_from_qpos(mujoco, model, data) -> None:
    for actuator_id in range(model.nu):
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        if not actuator_name:
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, actuator_name)
        if joint_id < 0:
            lo, hi = model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = 0.5 * (lo + hi)
            continue
        qpos_addr = model.jnt_qposadr[joint_id]
        data.ctrl[actuator_id] = data.qpos[qpos_addr]


def _reset_scene(mujoco, model, data) -> None:
    mujoco.mj_resetData(model, data)
    for joint_name, value in INIT_QPOS.items():
        _set_joint_qpos(mujoco, model, data, joint_name, value)
    _set_actuator_ctrl_from_qpos(mujoco, model, data)
    mujoco.mj_forward(model, data)


def _quat_wxyz_to_xyzw(quat) -> list[float]:
    return [float(quat[1]), float(quat[2]), float(quat[3]), float(quat[0])]


def _unscale(value, lower, upper):
    return (2.0 * value - upper - lower) / (upper - lower)


def _scale(value, lower, upper):
    return 0.5 * (value + 1.0) * (upper - lower) + lower


def _rotate_by_xmat(xmat, local_vec):
    import numpy as np

    return np.asarray(xmat, dtype=float).reshape(3, 3) @ np.asarray(local_vec, dtype=float)


class MujocoPolicyAdapter:
    def __init__(self, mujoco, model, data, checkpoint_path: Path, device: str):
        import numpy as np

        self.mujoco = mujoco
        self.model = model
        self.data = data
        self.player = _make_policy_player(checkpoint_path, device)
        self.device = device
        self.joint_limits = np.asarray([_joint_range(mujoco, model, name) for name in POLICY_JOINT_NAMES], dtype=float)
        self.prev_action_targets = np.asarray(
            [_get_joint_qpos(mujoco, model, data, name) for name in POLICY_JOINT_NAMES],
            dtype=float,
        )
        self.actuator_ids = [
            _name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in POLICY_JOINT_NAMES
        ]
        self.palm_site_id = _name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, "hand_tcp")
        self.object_body_id = _name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "claw_hammer")
        self.goal_body_id = _name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, "goal_claw_hammer")
        self.fingertip_body_ids = [
            _name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in FINGERTIP_BODIES
        ]
        self.keypoint_site_ids = [
            _name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in KEYPOINT_SITE_NAMES
        ]
        self.goal_keypoint_site_ids = [
            _name_to_id(mujoco, model, mujoco.mjtObj.mjOBJ_SITE, name)
            for name in GOAL_KEYPOINT_SITE_NAMES
        ]
        agent_cfg = _load_agent_cfg(checkpoint_path)
        env_cfg = agent_cfg.get("env_cfg", {})
        self.clip_observations = float(agent_cfg.get("params", {}).get("env", {}).get("clip_observations", 10.0))
        self.dof_speed_scale = float(env_cfg.get("dof_speed_scale", 1.5))
        self.hand_moving_average = float(env_cfg.get("hand_moving_average", 0.1))
        self.arm_moving_average = float(env_cfg.get("arm_moving_average", 0.1))
        self.dt = float(env_cfg.get("sim_dt", model.opt.timestep))
        self.next_policy_time = 0.0
        self.object_scales = np.asarray(DEFAULT_OBJECT_SCALES, dtype=float)

    def observation(self):
        import numpy as np
        import torch

        joint_pos = np.asarray(
            [_get_joint_qpos(self.mujoco, self.model, self.data, name) for name in POLICY_JOINT_NAMES],
            dtype=float,
        )
        joint_vel = np.asarray(
            [_get_joint_qvel(self.mujoco, self.model, self.data, name) for name in POLICY_JOINT_NAMES],
            dtype=float,
        )
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        joint_pos_obs = _unscale(joint_pos, lower, upper)
        palm_pos = np.asarray(self.data.site_xpos[self.palm_site_id], dtype=float)
        palm_rot_xyzw = _quat_wxyz_to_xyzw(self.data.xquat[self.model.site_bodyid[self.palm_site_id]])
        object_rot_xyzw = _quat_wxyz_to_xyzw(self.data.xquat[self.object_body_id])

        fingertip_pos = []
        for body_name, body_id in zip(FINGERTIP_BODIES, self.fingertip_body_ids, strict=True):
            pos = np.asarray(self.data.xpos[body_id], dtype=float)
            pos = pos + _rotate_by_xmat(self.data.xmat[body_id], FINGERTIP_LOCAL_OFFSETS[body_name])
            fingertip_pos.append(pos)
        fingertip_rel_palm = (np.asarray(fingertip_pos, dtype=float) - palm_pos).reshape(-1)

        object_keypoints = np.asarray([self.data.site_xpos[idx] for idx in self.keypoint_site_ids], dtype=float)
        goal_keypoints = np.asarray([self.data.site_xpos[idx] for idx in self.goal_keypoint_site_ids], dtype=float)
        keypoints_rel_palm = (object_keypoints - palm_pos).reshape(-1)
        keypoints_rel_goal = (object_keypoints - goal_keypoints).reshape(-1)

        obs = np.concatenate(
            (
                joint_pos_obs,
                joint_vel,
                self.prev_action_targets,
                palm_pos,
                np.asarray(palm_rot_xyzw, dtype=float),
                np.asarray(object_rot_xyzw, dtype=float),
                fingertip_rel_palm,
                keypoints_rel_palm,
                keypoints_rel_goal,
                self.object_scales,
            )
        )
        if obs.shape[0] != OBS_SIZE:
            raise RuntimeError(f"Expected {OBS_SIZE} observations, got {obs.shape[0]}.")
        obs = np.clip(obs, -self.clip_observations, self.clip_observations)
        tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.player.device).unsqueeze(0)
        intr_reward_coef_embd = getattr(self.player, "intr_reward_coef_embd", None)
        if intr_reward_coef_embd is not None:
            tensor = torch.cat([tensor, intr_reward_coef_embd[0:1]], dim=1)
        return tensor

    def compute_targets(self, actions):
        import numpy as np

        actions = np.clip(actions, -1.0, 1.0)
        lower = self.joint_limits[:, 0]
        upper = self.joint_limits[:, 1]
        targets = self.prev_action_targets.copy()
        targets[7:] = _scale(actions[7:], lower[7:], upper[7:])
        targets[7:] = (
            self.hand_moving_average * targets[7:]
            + (1.0 - self.hand_moving_average) * self.prev_action_targets[7:]
        )
        targets[7:] = np.clip(targets[7:], lower[7:], upper[7:])
        targets[:7] = self.prev_action_targets[:7] + self.dof_speed_scale * self.dt * actions[:7]
        targets[:7] = np.clip(targets[:7], lower[:7], upper[:7])
        targets[:7] = (
            self.arm_moving_average * targets[:7]
            + (1.0 - self.arm_moving_average) * self.prev_action_targets[:7]
        )
        return targets

    def step_policy(self) -> tuple[float, float]:
        import torch

        with torch.inference_mode():
            action = self.player.get_action(self.observation(), is_deterministic=True)
        action_np = action.detach().cpu().numpy().reshape(-1)
        targets = self.compute_targets(action_np)
        for actuator_id, target in zip(self.actuator_ids, targets, strict=True):
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = min(max(float(target), float(lo)), float(hi))
        self.prev_action_targets = targets
        return float(action_np.min()), float(action_np.max())

    def maybe_step_policy(self) -> tuple[float, float] | None:
        if self.data.time + 1.0e-12 < self.next_policy_time:
            return None
        action_range = self.step_policy()
        self.next_policy_time += self.dt
        return action_range


def _print_model_summary(mujoco, model, scene_xml: Path, print_names: bool) -> None:
    print(f"Loaded scene XML: {scene_xml}")
    print(f"  bodies:    {model.nbody}")
    print(f"  joints:    {model.njnt}")
    print(f"  qpos:      {model.nq}")
    print(f"  actuators: {model.nu}")
    print(f"  geoms:     {model.ngeom}")
    print(f"  dt:        {model.opt.timestep}")
    print(f"  gravity:   {model.opt.gravity}")
    if not print_names:
        return
    print("  actuator names:")
    for actuator_id in range(model.nu):
        print(f"    {actuator_id:02d}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)}")
    print("  body names:")
    for body_id in range(model.nbody):
        print(f"    {body_id:02d}: {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)}")


def _step_headless(mujoco, model, data, steps: int, policy: MujocoPolicyAdapter | None) -> None:
    for _ in range(steps):
        if policy is not None:
            policy.maybe_step_policy()
        mujoco.mj_step(model, data)


def _configure_viewer_camera(mujoco, viewer) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat[:] = DEFAULT_VIEWER_LOOKAT
    viewer.cam.distance = DEFAULT_VIEWER_DISTANCE
    viewer.cam.azimuth = DEFAULT_VIEWER_AZIMUTH
    viewer.cam.elevation = DEFAULT_VIEWER_ELEVATION


def _step_viewer_until_closed(
    mujoco,
    model,
    data,
    policy: MujocoPolicyAdapter | None,
    print_policy_every: int,
    set_viewer_camera: bool,
) -> int:
    viewer_mod = _maybe_import_viewer()
    if viewer_mod is None:
        raise ModuleNotFoundError("mujoco.viewer is required for visualization. Use --headless for a smoke check.")

    total_steps = 0
    with viewer_mod.launch_passive(model, data) as viewer:
        if set_viewer_camera:
            _configure_viewer_camera(mujoco, viewer)
        try:
            while viewer.is_running():
                action_range = None
                if policy is not None:
                    action_range = policy.maybe_step_policy()
                    if action_range is not None and print_policy_every > 0 and total_steps % print_policy_every == 0:
                        action_min, action_max = action_range
                        print(f"  policy action min/max: {action_min:.4f} / {action_max:.4f}", flush=True)
                mujoco.mj_step(model, data)
                viewer.sync()
                total_steps += 1
                time.sleep(model.opt.timestep)
        except KeyboardInterrupt:
            pass
    return total_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quick MuJoCo scene/checkpoint validation for the TG2 claw-hammer task."
    )
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE_XML, help="MuJoCo scene XML to load.")
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        help="Checkpoint file to load. Can be passed multiple times.",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory containing .pth/.pt checkpoints.")
    parser.add_argument("--device", default="cpu", help="Torch device used for checkpoint loading.")
    parser.add_argument(
        "--steps",
        type=int,
        default=240,
        help="Number of MuJoCo steps for --headless smoke checks. Ignored by the viewer replay.",
    )
    parser.add_argument("--dt", type=float, default=None, help="Optional MuJoCo timestep override.")
    parser.add_argument("--headless", action="store_true", help="Run once without opening the MuJoCo viewer.")
    parser.add_argument("--print-names", action="store_true", help="Print actuator and body names after loading.")
    parser.add_argument(
        "--no-policy-control",
        action="store_true",
        help="Only load/summarize checkpoints; do not drive MuJoCo controls from the policy.",
    )
    parser.add_argument(
        "--print-policy-every",
        type=int,
        default=300,
        help="Viewer-mode interval for printing policy action min/max. Use 0 to disable.",
    )
    parser.add_argument(
        "--no-set-viewer-camera",
        action="store_true",
        help="Use MuJoCo's default free camera instead of the front-facing TG2 replay camera.",
    )
    args = parser.parse_args()

    scene_xml = args.scene_xml.expanduser().resolve()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Scene XML not found: {scene_xml}")

    checkpoint_paths = _collect_checkpoint_paths(args)
    for checkpoint_path in checkpoint_paths:
        _summarize_checkpoint(checkpoint_path, _torch_load(checkpoint_path, args.device))

    mujoco = _require_mujoco()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    if args.dt is not None:
        model.opt.timestep = float(args.dt)

    _reset_scene(mujoco, model, data)
    _print_model_summary(mujoco, model, scene_xml, args.print_names)
    policy = None
    if checkpoint_paths and not args.no_policy_control:
        policy = MujocoPolicyAdapter(mujoco, model, data, checkpoint_paths[0], args.device)
        print(f"Policy control enabled from checkpoint: {checkpoint_paths[0]}")

    if args.headless:
        _step_headless(mujoco, model, data, args.steps, policy)
        total_steps = args.steps
    else:
        total_steps = _step_viewer_until_closed(
            mujoco,
            model,
            data,
            policy,
            args.print_policy_every,
            set_viewer_camera=not args.no_set_viewer_camera,
        )

    print(f"Stepped {total_steps} MuJoCo steps.")
    print(f"  time:      {data.time:.4f}")
    print(f"  ctrl min/max: {data.ctrl.min():.4f} / {data.ctrl.max():.4f}")


if __name__ == "__main__":
    main()
