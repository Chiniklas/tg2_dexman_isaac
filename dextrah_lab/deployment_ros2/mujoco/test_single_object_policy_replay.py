from __future__ import annotations

# Run from repo root:
# python dextrah_lab/deployment_ros2/mujoco/test_single_object_policy_replay.py
#
# Headless smoke check:
# python dextrah_lab/deployment_ros2/mujoco/test_single_object_policy_replay.py --headless --steps 240

import argparse
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SCENE_XML = (
    REPO_ROOT
    / "dextrah_lab"
    / "assets"
    / "tiangong2pro"
    / "mujoco_scene"
    / "single_cube_scene.xml"
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


def _step_headless(mujoco, model, data, steps: int) -> None:
    for _ in range(steps):
        mujoco.mj_step(model, data)


def _step_viewer_until_closed(mujoco, model, data) -> int:
    viewer_mod = _maybe_import_viewer()
    if viewer_mod is None:
        raise ModuleNotFoundError("mujoco.viewer is required for visualization. Use --headless for a smoke check.")

    total_steps = 0
    with viewer_mod.launch_passive(model, data) as viewer:
        try:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
                total_steps += 1
                time.sleep(model.opt.timestep)
        except KeyboardInterrupt:
            pass
    return total_steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick MuJoCo scene/checkpoint validation for the TG2 single-cube task.")
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
    args = parser.parse_args()

    scene_xml = args.scene_xml.expanduser().resolve()
    if not scene_xml.is_file():
        raise FileNotFoundError(f"Scene XML not found: {scene_xml}")

    for checkpoint_path in _collect_checkpoint_paths(args):
        _summarize_checkpoint(checkpoint_path, _torch_load(checkpoint_path, args.device))

    mujoco = _require_mujoco()
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    if args.dt is not None:
        model.opt.timestep = float(args.dt)

    _reset_scene(mujoco, model, data)
    _print_model_summary(mujoco, model, scene_xml, args.print_names)

    if args.headless:
        _step_headless(mujoco, model, data, args.steps)
        total_steps = args.steps
    else:
        total_steps = _step_viewer_until_closed(mujoco, model, data)

    print(f"Stepped {total_steps} MuJoCo steps.")
    print(f"  time:      {data.time:.4f}")
    print(f"  ctrl min/max: {data.ctrl.min():.4f} / {data.ctrl.max():.4f}")


if __name__ == "__main__":
    main()
