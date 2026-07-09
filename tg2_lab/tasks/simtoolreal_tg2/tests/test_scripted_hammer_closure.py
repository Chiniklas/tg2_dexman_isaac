"""Deterministic physical closure-and-hold test for TG2 + InspireHand.

This is not an RL evaluation. It isolates the robot asset, mimic mapping,
actuator gains, collision geometry, and friction used by ``simtoolreal_tg2``.

The default arm pose is verified to be palm-up, then the hammer is placed once
in the open hand. From that point onward the hammer remains fully dynamic under
gravity and contact while the hand settles, closes, preloads, and waves through
a scripted joint-space motion. The test passes only if the closed hand keeps
the hammer near its initial pose relative to the moving palm.

Visual diagnostic:

    python tg2_lab/tasks/simtoolreal_tg2/tests/test_scripted_hammer_closure.py

Headless pass/fail diagnostic:

    python tg2_lab/tasks/simtoolreal_tg2/tests/test_scripted_hammer_closure.py --headless

Use ``--object-local-offset`` and ``--object-local-rpy-deg`` to calibrate the
hammer pose in the frame selected by ``--fixture-mode``. The demonstrated USD
pose uses the robot root; palm-relative and training placements remain
available for comparison.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import traceback
from typing import Sequence


# Direct execution sets sys.path[0] to this nested tests directory. Add the
# repository root so the checked-in ``tg2_lab`` package remains importable
# without requiring callers to set PYTHONPATH manually.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OPEN_HAND_ACTIONS = (-1.0, -1.0, -1.0, -1.0, -1.0, -1.0)
# Successful manually inspected hammer grasp (2026-07-06): the four finger
# joints are at 63 deg, thumb opposition at 59.2 deg, and thumb flexion at
# 14.1 deg. Converted from joint limits to the policy's normalized [-1, 1]
# action space. Driving both thumb joints to +1 over-closes the thumb and can
# push the handle out of the hand.
DEFAULT_CLOSED_HAND_ACTIONS = (
    1.0,
    1.0,
    1.0,
    1.0,
    0.6294,
    -0.0156,
)
DEFAULT_WAVE_ARM_AMPLITUDES = (0.0, 0.35, 0.0, 0.25, 0.0, 0.25, 0.60)
DEFAULT_WAVE_ARM_PHASES_DEG = (0.0, 0.0, 0.0, 180.0, 0.0, 90.0, 0.0)
# The TG2 palm's outward/support normal is local -X.  This was verified against
# the palm-up pose in Isaac Sim's Physics Inspector (2026-07-06).
PALM_SUPPORT_AXIS = (-1.0, 0.0, 0.0)
PALM_UP_ARM_JOINT_POS = {
    "shoulder_pitch_r_joint": math.radians(0.0),
    # Physics Inspector displayed 0 deg, just outside this joint's -5.73 deg
    # upper limit.  Use the valid boundary while preserving the observed pose.
    "shoulder_roll_r_joint": math.radians(-5.73),
    "shoulder_yaw_r_joint": math.radians(0.0),
    "elbow_pitch_r_joint": math.radians(-88.1),
    "elbow_yaw_r_joint": math.radians(-93.1),
    "wrist_pitch_r_joint": math.radians(0.0),
    "wrist_roll_r_joint": math.radians(0.0),
}


@dataclass(frozen=True)
class HoldResult:
    passed: bool
    completed_steps: int
    requested_steps: int
    hold_ratio: float
    max_position_drift_m: float
    max_vertical_drop_m: float
    max_hand_tracking_error_rad: float
    terminated: bool
    truncated: bool


@dataclass(frozen=True)
class ContactReport:
    filtered_contact: bool
    raw_contact: bool
    filtered_force_sum: float
    raw_force_sum: float
    reward_gate_count: int
    per_body_lines: tuple[str, ...]


def _parse_floats(raw: str, count: int, option_name: str) -> tuple[float, ...]:
    try:
        values = tuple(float(item.strip()) for item in raw.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{option_name} must contain comma-separated numbers") from exc
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{option_name} requires {count} values, got {len(values)}")
    return values


def _phase_steps(seconds: float, control_hz: float) -> int:
    return max(0, int(round(seconds * control_hz)))


def _linear_hand_action(
    open_actions: Sequence[float], closed_actions: Sequence[float], fraction: float
) -> tuple[float, ...]:
    fraction = min(max(float(fraction), 0.0), 1.0)
    return tuple(start + fraction * (end - start) for start, end in zip(open_actions, closed_actions))


def _evaluate_hold(
    *,
    completed_steps: int,
    requested_steps: int,
    within_threshold_steps: int,
    max_position_drift_m: float,
    max_vertical_drop_m: float,
    max_hand_tracking_error_rad: float,
    terminated: bool,
    truncated: bool,
    required_hold_ratio: float,
    max_drift_m: float,
    max_drop_m: float,
    max_hand_error_rad: float,
) -> HoldResult:
    hold_ratio = within_threshold_steps / max(1, requested_steps)
    passed = (
        completed_steps == requested_steps
        and not terminated
        and not truncated
        and hold_ratio >= required_hold_ratio
        and max_position_drift_m <= max_drift_m
        and max_vertical_drop_m <= max_drop_m
        and max_hand_tracking_error_rad <= max_hand_error_rad
    )
    return HoldResult(
        passed=passed,
        completed_steps=completed_steps,
        requested_steps=requested_steps,
        hold_ratio=hold_ratio,
        max_position_drift_m=max_position_drift_m,
        max_vertical_drop_m=max_vertical_drop_m,
        max_hand_tracking_error_rad=max_hand_tracking_error_rad,
        terminated=terminated,
        truncated=truncated,
    )


def _make_env_cfg(args_cli, total_steps: int):
    from tg2_lab.tasks.simtoolreal_tg2.simtoolreal_tg2_env_cfg import SimToolRealTg2EnvCfg

    cfg = SimToolRealTg2EnvCfg()
    cfg.scene.num_envs = 1
    cfg.scene.replicate_physics = False
    cfg.sim.device = args_cli.device or os.environ.get("SIMTOOLREAL_TG2_TEST_DEVICE", "cuda:0")
    cfg.episode_length_s = max(20.0, (total_steps + 120) * cfg.sim_dt)
    initial_joint_pos = dict(cfg.robot_cfg.init_state.joint_pos)
    initial_joint_pos.update(PALM_UP_ARM_JOINT_POS)
    cfg.robot_cfg = cfg.robot_cfg.replace(
        init_state=cfg.robot_cfg.init_state.replace(joint_pos=initial_joint_pos)
    )

    # Deterministic fixture setup. This test should diagnose grasp physics, not
    # robustness to randomization or delayed state/action delivery.
    cfg.object_spawn_xy_range = 0.0
    cfg.object_spawn_z_range = 0.0
    cfg.randomize_object_rotation = False
    cfg.table_reset_z_range = 0.0
    cfg.reset_dof_pos_noise_fingers = 0.0
    cfg.reset_dof_pos_noise_arm = 0.0
    cfg.reset_dof_vel_noise = 0.0
    cfg.use_obs_delay = False
    cfg.obs_delay_max = 1
    cfg.use_action_delay = False
    cfg.action_delay_max = 1
    cfg.use_object_state_delay_noise = False
    cfg.object_state_delay_max = 1
    cfg.object_state_xyz_noise_std = 0.0
    cfg.object_state_rotation_noise_degrees = 0.0
    cfg.joint_velocity_obs_noise_std = 0.0
    cfg.force_scale = 0.0
    cfg.force_prob_range = (0.0, 0.0)
    cfg.torque_scale = 0.0
    cfg.torque_prob_range = (0.0, 0.0)
    cfg.debug_keypoints = args_cli.debug_keypoints
    cfg.debug_grasp_bounding_box = args_cli.debug_grasp_bounding_box
    cfg.debug_fingertips = args_cli.debug_fingertips
    return cfg


def _make_policy_action(env, hand_actions: Sequence[float], arm_actions: Sequence[float] | None = None):
    import torch

    action = torch.zeros((1, env.num_actions), device=env.device)
    if arm_actions is not None:
        action[:, :7] = torch.tensor(arm_actions, device=env.device)
    action[:, 7:] = torch.tensor(hand_actions, device=env.device)
    return action


def _wave_arm_action(args_cli, elapsed_sec: float) -> tuple[float, ...]:
    omega_t = 2.0 * math.pi * args_cli.wave_frequency_hz * elapsed_sec
    return tuple(
        amplitude * math.sin(omega_t + math.radians(phase_deg))
        for amplitude, phase_deg in zip(args_cli.wave_arm_amplitudes, args_cli.wave_arm_phases_deg)
    )


def _initial_object_pose(env, args_cli):
    import torch
    from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

    env._compute_intermediate_values()
    if args_cli.fixture_mode == "training":
        return env.object.data.root_pos_w[0].clone(), env.object.data.root_quat_w[0].clone()

    local_offset = torch.tensor(args_cli.object_local_offset, device=env.device, dtype=env.palm_pos.dtype)
    if args_cli.fixture_mode == "robot":
        reference_pos_w = env.robot.data.root_pos_w[0]
        reference_rot_w = env.robot.data.root_quat_w[0]
    else:
        reference_pos_w = env.palm_pos[0] + env.scene.env_origins[0]
        reference_rot_w = env.palm_rot[0]
    initial_pos_w = reference_pos_w + quat_apply(
        reference_rot_w.unsqueeze(0), local_offset.unsqueeze(0)
    )[0]
    roll, pitch, yaw = (math.radians(value) for value in args_cli.object_local_rpy_deg)
    local_rot = quat_from_euler_xyz(
        torch.tensor([roll], device=env.device),
        torch.tensor([pitch], device=env.device),
        torch.tensor([yaw], device=env.device),
    )[0]
    initial_rot_w = quat_mul(reference_rot_w.unsqueeze(0), local_rot.unsqueeze(0))[0]
    return initial_pos_w, initial_rot_w


def _place_object_once(env, position_w, rotation_w) -> None:
    """Set the initial hammer pose and velocity; never constrain it afterward."""
    root_state = env.object.data.root_state_w.clone()
    root_state[0, 0:3] = position_w
    root_state[0, 3:7] = rotation_w
    root_state[0, 7:13] = 0.0
    env.object.write_root_state_to_sim(root_state)


def _lower_table(env, distance_m: float) -> None:
    table_state = env.table.data.root_state_w.clone()
    table_state[:, 2] -= float(distance_m)
    table_state[:, 7:13] = 0.0
    env.table.write_root_state_to_sim(table_state)


def _relative_object_position(env):
    from isaaclab.utils.math import quat_apply_inverse

    env._compute_intermediate_values()
    object_from_palm_w = env.object_pos - env.palm_pos
    return quat_apply_inverse(env.palm_rot, object_from_palm_w)[0]


def _palm_up_score(env) -> float:
    import torch
    from isaaclab.utils.math import quat_apply

    env._compute_intermediate_values()
    local_support_axis = torch.tensor(PALM_SUPPORT_AXIS, device=env.device, dtype=env.palm_pos.dtype)
    support_axis_w = quat_apply(env.palm_rot, local_support_axis.unsqueeze(0).expand(env.num_envs, -1))
    return float(support_axis_w[0, 2].item())


def _max_hand_tracking_error(env) -> float:
    import torch

    desired = env.dof_pos_targets[:, env.target_dof_indices[7:]]
    observed = env.robot.data.joint_pos[:, env.target_dof_indices[7:]]
    if desired.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(desired - observed)).item())


def _contact_report(env) -> ContactReport:
    import torch

    # Refresh kinematic/contact buffers without touching the reward bookkeeping.
    env._compute_intermediate_values()

    threshold = float(env.cfg.lift_contact_force_threshold)
    body_names = tuple(getattr(env.cfg, "lift_contact_body_names", ()))
    filtered_forces = getattr(env, "hand_object_contact_forces", None)
    raw_forces = getattr(env, "hand_raw_contact_forces", None)

    if filtered_forces is None or filtered_forces.numel() == 0:
        return ContactReport(
            filtered_contact=False,
            raw_contact=False,
            filtered_force_sum=0.0,
            raw_force_sum=0.0,
            reward_gate_count=0,
            per_body_lines=("no hand-object contact sensors configured",),
        )

    filtered = filtered_forces[0].detach()
    raw = raw_forces[0].detach() if raw_forces is not None and raw_forces.numel() > 0 else torch.zeros_like(filtered)
    filtered_contact_mask = filtered > threshold
    raw_contact_mask = raw > threshold
    # Recompute the same non-mutating gate used by the per-finger bonus:
    # each distal finger link gets credit when its filtered object-contact
    # force exceeds the configured threshold.  The palm/contact body at index 0
    # is for lift gating and is not counted as a per-finger bonus.
    distal_contact_mask = filtered_contact_mask[1:]
    reward_gate_count = int(distal_contact_mask.sum().item())

    lines = []
    for idx, body_name in enumerate(body_names):
        gate_text = ""
        if idx > 0:
            gate_text = f" reward_gate={bool(filtered_contact_mask[idx].item())}"
        lines.append(
            f"{body_name}: raw={float(raw[idx].item()):.3f}N "
            f"filtered={float(filtered[idx].item()):.3f}N "
            f"raw_hit={bool(raw_contact_mask[idx].item())} "
            f"object_hit={bool(filtered_contact_mask[idx].item())}{gate_text}"
        )

    return ContactReport(
        filtered_contact=bool(filtered_contact_mask.any().item()),
        raw_contact=bool(raw_contact_mask.any().item()),
        filtered_force_sum=float(filtered.sum().item()),
        raw_force_sum=float(raw.sum().item()),
        reward_gate_count=reward_gate_count,
        per_body_lines=tuple(lines),
    )


def _format_contact_summary(env) -> str:
    report = _contact_report(env)
    return (
        f"contact_raw={report.raw_contact} raw_force={report.raw_force_sum:.3f}N "
        f"contact_obj={report.filtered_contact} obj_force={report.filtered_force_sum:.3f}N "
        f"reward_gate_count={report.reward_gate_count}"
    )


def _print_contact_report(env, label: str) -> None:
    report = _contact_report(env)
    print(
        f"[CONTACT] {label}: raw_contact={report.raw_contact} "
        f"raw_force_sum={report.raw_force_sum:.3f}N "
        f"object_contact={report.filtered_contact} "
        f"object_force_sum={report.filtered_force_sum:.3f}N "
        f"reward_gate_count={report.reward_gate_count}",
        flush=True,
    )
    for line in report.per_body_lines:
        print(f"  {line}", flush=True)


def _step(env, action):
    _obs, _reward, terminated, truncated, _extras = env.step(action)
    return bool(terminated[0].item()), bool(truncated[0].item())


def _run_phase(env, *, name: str, steps: int, action_fn, print_every: int) -> None:
    if steps <= 0:
        return
    print(f"[PHASE] {name}: {steps} steps", flush=True)
    _print_contact_report(env, f"{name}:start")
    for step in range(steps):
        terminated, truncated = _step(env, action_fn(step, steps))
        if terminated or truncated:
            raise RuntimeError(f"Environment terminated during phase '{name}' at step {step + 1}")
        if print_every > 0 and ((step + 1) % print_every == 0 or step + 1 == steps):
            print(
                f"  step={step + 1}/{steps} hand_error={_max_hand_tracking_error(env):.4f} rad "
                f"{_format_contact_summary(env)}",
                flush=True,
            )
    _print_contact_report(env, f"{name}:end")


def _run_hold_and_wave_phase(env, args_cli, closed_actions, hold_steps: int, wave_steps: int) -> HoldResult:
    import torch

    requested_steps = hold_steps + wave_steps
    print(f"[PHASE] table-lowered-static-hold: {hold_steps} steps", flush=True)
    _print_contact_report(env, "hold:start")
    baseline_relative_pos = _relative_object_position(env).clone()
    support_axis = torch.tensor(PALM_SUPPORT_AXIS, device=env.device, dtype=baseline_relative_pos.dtype)
    baseline_support_height = float(torch.dot(baseline_relative_pos, support_axis).item())
    max_drift = 0.0
    max_drop = 0.0
    max_hand_error = 0.0
    within_threshold_steps = 0
    completed_steps = 0
    terminated = False
    truncated = False

    for step in range(requested_steps):
        if step == hold_steps and wave_steps > 0:
            print(f"[PHASE] wave-arm-while-holding: {wave_steps} steps", flush=True)
            _print_contact_report(env, "wave:start")
        if step < hold_steps:
            action = _make_policy_action(env, closed_actions)
            phase_step = step + 1
            phase_total = hold_steps
            phase_name = "hold"
        else:
            wave_step = step - hold_steps
            arm_actions = _wave_arm_action(args_cli, wave_step / 60.0)
            action = _make_policy_action(env, closed_actions, arm_actions)
            phase_step = wave_step + 1
            phase_total = wave_steps
            phase_name = "wave"

        terminated, truncated = _step(env, action)
        completed_steps = step + 1
        if terminated or truncated:
            break

        relative_pos = _relative_object_position(env)
        drift = float(torch.linalg.vector_norm(relative_pos - baseline_relative_pos).item())
        support_height = float(torch.dot(relative_pos, support_axis).item())
        vertical_drop = max(0.0, baseline_support_height - support_height)
        hand_error = _max_hand_tracking_error(env)
        max_drift = max(max_drift, drift)
        max_drop = max(max_drop, vertical_drop)
        max_hand_error = max(max_hand_error, hand_error)
        if drift <= args_cli.max_drift_m and vertical_drop <= args_cli.max_drop_m:
            within_threshold_steps += 1

        if args_cli.print_every > 0 and (
            phase_step % args_cli.print_every == 0 or phase_step == phase_total
        ):
            print(
                f"  {phase_name}_step={phase_step}/{phase_total} drift={drift:.4f}m "
                f"drop={vertical_drop:.4f}m hand_error={hand_error:.4f}rad "
                f"{_format_contact_summary(env)}",
                flush=True,
            )

    _print_contact_report(env, "hold-wave:end")
    return _evaluate_hold(
        completed_steps=completed_steps,
        requested_steps=requested_steps,
        within_threshold_steps=within_threshold_steps,
        max_position_drift_m=max_drift,
        max_vertical_drop_m=max_drop,
        max_hand_tracking_error_rad=max_hand_error,
        terminated=terminated,
        truncated=truncated,
        required_hold_ratio=args_cli.required_hold_ratio,
        max_drift_m=args_cli.max_drift_m,
        max_drop_m=args_cli.max_drop_m,
        max_hand_error_rad=args_cli.max_hand_error_rad,
    )


def _print_result(result: HoldResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[RESULT] {status}", flush=True)
    print(f"  completed hold: {result.completed_steps}/{result.requested_steps} steps", flush=True)
    print(f"  hold ratio: {result.hold_ratio:.3f}", flush=True)
    print(f"  max palm-relative drift: {result.max_position_drift_m:.4f} m", flush=True)
    print(f"  max palm-support-axis drop: {result.max_vertical_drop_m:.4f} m", flush=True)
    print(f"  max hand tracking error: {result.max_hand_tracking_error_rad:.4f} rad", flush=True)
    print(f"  terminated/truncated: {result.terminated}/{result.truncated}", flush=True)


def _run_with_app(args_cli) -> bool:
    from isaaclab.app import AppLauncher

    control_hz = 60.0
    settle_steps = _phase_steps(args_cli.settle_sec, control_hz)
    close_steps = _phase_steps(args_cli.close_sec, control_hz)
    preload_steps = _phase_steps(args_cli.preload_sec, control_hz)
    hold_steps = _phase_steps(args_cli.hold_sec, control_hz)
    wave_steps = _phase_steps(args_cli.wave_sec, control_hz)
    total_steps = settle_steps + close_steps + preload_steps + hold_steps + wave_steps

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app
    env = None
    try:
        print("[SETUP] Importing TG2 environment...", flush=True)
        from tg2_lab.tasks.simtoolreal_tg2.simtoolreal_tg2_env import SimToolRealTg2Env

        print("[SETUP] Building deterministic test configuration...", flush=True)
        cfg = _make_env_cfg(args_cli, total_steps)
        print("[SETUP] Creating simulation environment...", flush=True)
        env = SimToolRealTg2Env(cfg)
        print("[SETUP] Resetting simulation environment...", flush=True)
        env.reset()

        palm_up_score = _palm_up_score(env)
        print(
            f"[PALM] support_axis={PALM_SUPPORT_AXIS} world_up_alignment={palm_up_score:.3f}",
            flush=True,
        )
        if palm_up_score < args_cli.min_palm_up_score:
            raise RuntimeError(
                f"Configured arm pose is not palm-up: score={palm_up_score:.3f} "
                f"< threshold={args_cli.min_palm_up_score:.3f}"
            )

        open_actions = args_cli.open_hand_actions
        closed_actions = args_cli.closed_hand_actions
        initial_object_pose = _initial_object_pose(env, args_cli)
        _place_object_once(env, *initial_object_pose)
        print(
            f"[INITIAL POSE] mode={args_cli.fixture_mode} dynamic=true "
            f"position_w={tuple(round(float(v), 4) for v in initial_object_pose[0].tolist())}",
            flush=True,
        )

        open_action = _make_policy_action(env, open_actions)
        closed_action = _make_policy_action(env, closed_actions)
        _run_phase(
            env,
            name="open-and-settle",
            steps=settle_steps,
            action_fn=lambda _step, _total: open_action,
            print_every=args_cli.print_every,
        )
        _run_phase(
            env,
            name="gradual-close",
            steps=close_steps,
            action_fn=lambda step, total: _make_policy_action(
                env, _linear_hand_action(open_actions, closed_actions, (step + 1) / max(1, total))
            ),
            print_every=args_cli.print_every,
        )
        _run_phase(
            env,
            name="closed-preload",
            steps=preload_steps,
            action_fn=lambda _step, _total: closed_action,
            print_every=args_cli.print_every,
        )

        _lower_table(env, args_cli.table_drop_m)
        result = _run_hold_and_wave_phase(env, args_cli, closed_actions, hold_steps, wave_steps)
        _print_result(result)
        return result.passed
    except BaseException as exc:
        # Print before SimulationApp.close(): Kit may terminate the process
        # during shutdown, which would otherwise hide the original exception.
        print(f"[ERROR] Setup/run failed with {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        simulation_app.close()


def _build_parser() -> argparse.ArgumentParser:
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Scripted TG2 hammer closure and gravity-hold diagnostic.")
    parser.add_argument("--fixture-mode", choices=("training", "robot", "palm"), default="robot")
    parser.add_argument(
        "--object-local-offset",
        type=lambda raw: _parse_floats(raw, 3, "--object-local-offset"),
        default=(0.4518, -0.32205, 0.09672),
        metavar="X,Y,Z",
        help="Hammer position in the robot or palm frame selected by --fixture-mode.",
    )
    parser.add_argument(
        "--object-local-rpy-deg",
        type=lambda raw: _parse_floats(raw, 3, "--object-local-rpy-deg"),
        default=(0.0, 0.0, 90.0),
        metavar="R,P,Y",
        help="Hammer orientation relative to the selected reference frame, in degrees.",
    )
    parser.add_argument(
        "--open-hand-actions",
        type=lambda raw: _parse_floats(raw, 6, "--open-hand-actions"),
        default=DEFAULT_OPEN_HAND_ACTIONS,
        metavar="A0,...,A5",
    )
    parser.add_argument(
        "--closed-hand-actions",
        type=lambda raw: _parse_floats(raw, 6, "--closed-hand-actions"),
        default=DEFAULT_CLOSED_HAND_ACTIONS,
        metavar="A0,...,A5",
    )
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--close-sec", type=float, default=3.0)
    parser.add_argument("--preload-sec", type=float, default=1.0)
    parser.add_argument("--hold-sec", type=float, default=1.0)
    parser.add_argument("--wave-sec", type=float, default=4.0)
    parser.add_argument("--wave-frequency-hz", type=float, default=0.5)
    parser.add_argument(
        "--wave-arm-amplitudes",
        type=lambda raw: _parse_floats(raw, 7, "--wave-arm-amplitudes"),
        default=DEFAULT_WAVE_ARM_AMPLITUDES,
        metavar="A0,...,A6",
        help="Normalized sinusoidal arm-action amplitudes during the wave phase.",
    )
    parser.add_argument(
        "--wave-arm-phases-deg",
        type=lambda raw: _parse_floats(raw, 7, "--wave-arm-phases-deg"),
        default=DEFAULT_WAVE_ARM_PHASES_DEG,
        metavar="P0,...,P6",
    )
    parser.add_argument("--min-palm-up-score", type=float, default=0.75)
    parser.add_argument("--table-drop-m", type=float, default=0.50)
    parser.add_argument("--max-drift-m", type=float, default=0.05)
    parser.add_argument("--max-drop-m", type=float, default=0.03)
    parser.add_argument("--max-hand-error-rad", type=float, default=0.35)
    parser.add_argument("--required-hold-ratio", type=float, default=0.90)
    parser.add_argument("--print-every", type=int, default=30)
    parser.add_argument("--debug-keypoints", action="store_true")
    parser.add_argument("--debug-grasp-bounding-box", action="store_true")
    parser.add_argument("--debug-fingertips", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    return parser


def _validate_args(args_cli) -> None:
    for name in ("settle_sec", "close_sec", "preload_sec", "hold_sec", "wave_sec", "table_drop_m"):
        if getattr(args_cli, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args_cli.hold_sec <= 0.0:
        raise ValueError("--hold-sec must be positive")
    if args_cli.wave_frequency_hz <= 0.0:
        raise ValueError("--wave-frequency-hz must be positive")
    if not -1.0 <= args_cli.min_palm_up_score <= 1.0:
        raise ValueError("--min-palm-up-score must be in [-1, 1]")
    if not 0.0 <= args_cli.required_hold_ratio <= 1.0:
        raise ValueError("--required-hold-ratio must be in [0, 1]")
    for name in ("max_drift_m", "max_drop_m", "max_hand_error_rad"):
        if getattr(args_cli, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")


def test_linear_hand_action_endpoints() -> None:
    assert _linear_hand_action((0.0,) * 6, (1.0,) * 6, 0.0) == (0.0,) * 6
    assert _linear_hand_action((0.0,) * 6, (1.0,) * 6, 1.0) == (1.0,) * 6


def test_wave_arm_action_uses_configured_amplitude_and_phase() -> None:
    args = argparse.Namespace(
        wave_frequency_hz=0.5,
        wave_arm_amplitudes=(1.0,) + (0.0,) * 6,
        wave_arm_phases_deg=(90.0,) + (0.0,) * 6,
    )
    action = _wave_arm_action(args, elapsed_sec=0.0)
    assert math.isclose(action[0], 1.0)
    assert action[1:] == (0.0,) * 6


def test_hold_evaluation_requires_sustained_bounded_hold() -> None:
    result = _evaluate_hold(
        completed_steps=180,
        requested_steps=180,
        within_threshold_steps=180,
        max_position_drift_m=0.01,
        max_vertical_drop_m=0.005,
        max_hand_tracking_error_rad=0.1,
        terminated=False,
        truncated=False,
        required_hold_ratio=0.9,
        max_drift_m=0.05,
        max_drop_m=0.03,
        max_hand_error_rad=0.35,
    )
    assert result.passed


def main() -> None:
    parser = _build_parser()
    args_cli = parser.parse_args()
    _validate_args(args_cli)
    try:
        passed = _run_with_app(args_cli)
    except BaseException as exc:
        print(f"[ERROR] Scripted closure test failed with {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
