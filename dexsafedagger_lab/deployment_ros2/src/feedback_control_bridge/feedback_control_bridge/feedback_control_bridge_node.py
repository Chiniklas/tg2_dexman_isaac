"""Standalone ROS 2 bridge between generic JointState commands and Tiangong control IO.

This node is intended to be the robot-facing bridge layer. Upstream nodes can stay
in standard ROS message space and publish ``sensor_msgs/msg/JointState`` commands,
while this node converts those commands into the Tiangong-specific arm and hand
interfaces expected by the robot-side stack.

High-level behavior
-------------------
1. Subscribe to an arm command topic carrying ``JointState`` messages.
2. Convert the commanded arm joints into ``bodyctrl_msgs/msg/CmdSetMotorPosition``
   and publish them to the robot arm command topic, typically ``/arm/cmd_pos``.
3. Optionally subscribe to a hand command ``JointState`` topic and translate those
   values into the robot hand topic or ``bodyctrl_msgs/srv/SetAngleFlexible``
   service, depending on configuration.
4. Subscribe to live robot arm state from ``/arm/status`` and hand state from the
   hand state topic, then republish a merged ``/joint_states`` stream for RViz,
   robot_state_publisher, logging, or higher-level monitoring.

Control-domain model
--------------------
The bridge supports a small set of control domains that decide which joint groups
are active:

- ``right_arm``: right arm only
- ``left_arm``: left arm only
- ``right_full``: right arm + right hand
- ``left_full``: left arm + left hand
- ``upper_body``: placeholder for future arm + hand + head support
- ``full_body``: placeholder for future whole-body support including legs

At the moment, only the arm-only and arm+hand domains are implemented. The
placeholder domains are accepted so callers can already target the eventual API,
but they currently fall back to ``right_full`` behavior and emit a warning at
startup.

Interface assumptions
---------------------
- Arm commands are expressed as named joints in radians.
- Arm status arrives as ``bodyctrl_msgs/msg/MotorStatusMsg`` where motor IDs must
  match the joint-to-motor maps defined below.
- Hand commands are converted by normalizing selected hand joints into the
  documented robot hand percentage / ratio representation.
- The bridge does not perform planning, smoothing, collision checking, or
  synchronization across subsystems. It is a translation layer only.

Why this file looks the way it does
-----------------------------------
- Joint maps are explicit so the bridge behavior is deterministic and easy to
  audit against robot firmware expectations.
- The bridge caches the latest commanded or measured values so it can continue to
  publish coherent ``/joint_states`` messages even when some inputs are partial.
- Left and right hand interfaces are kept configurable, but the defaults are
  automatically adjusted for ``left_full`` so users do not need to override every
  hand topic manually.
"""

from __future__ import annotations

import ast
import math
import threading
from typing import Any, Dict, Iterable, List, Tuple, Union

import rclpy
from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, SetMotorPosition, WaistMotorStatus
from bodyctrl_msgs.srv import SetAngleFlexible
from rclpy.node import Node
from sensor_msgs.msg import JointState


LEFT_ARM_MAP: List[Tuple[int, str]] = [
    (11, "shoulder_pitch_l_joint"),
    (12, "shoulder_roll_l_joint"),
    (13, "shoulder_yaw_l_joint"),
    (14, "elbow_pitch_l_joint"),
    (15, "elbow_yaw_l_joint"),
    (16, "wrist_pitch_l_joint"),
    (17, "wrist_roll_l_joint"),
]

RIGHT_ARM_MAP: List[Tuple[int, str]] = [
    (21, "shoulder_pitch_r_joint"),
    (22, "shoulder_roll_r_joint"),
    (23, "shoulder_yaw_r_joint"),
    (24, "elbow_pitch_r_joint"),
    (25, "elbow_yaw_r_joint"),
    (26, "wrist_pitch_r_joint"),
    (27, "wrist_roll_r_joint"),
]

LEFT_LEG_MAP: List[Tuple[int, str]] = [
    (51, "hip_roll_l_joint"),
    (52, "hip_pitch_l_joint"),
    (53, "hip_yaw_l_joint"),
    (54, "knee_pitch_l_joint"),
    (55, "ankle_pitch_l_joint"),
    (56, "ankle_roll_l_joint"),
]

RIGHT_LEG_MAP: List[Tuple[int, str]] = [
    (61, "hip_roll_r_joint"),
    (62, "hip_pitch_r_joint"),
    (63, "hip_yaw_r_joint"),
    (64, "knee_pitch_r_joint"),
    (65, "ankle_pitch_r_joint"),
    (66, "ankle_roll_r_joint"),
]

HEAD_MAP: List[Tuple[int, str]] = [
    (1, "head_yaw_joint"),
    (2, "head_pitch_joint"),
    (3, "head_roll_joint"),
]

WAIST_STATE_JOINT = "body_yaw_joint"

LEFT_HAND_SERVICE_MAP: List[Tuple[str, str, float, float]] = [
    ("left_little_1_joint", "1", 0.0, 1.1),
    ("left_ring_1_joint", "2", 0.0, 1.1),
    ("left_middle_1_joint", "3", 0.0, 1.1),
    ("left_index_1_joint", "4", 0.0, 1.1),
    ("left_thumb_2_joint", "5", 0.0, 0.5),
    ("left_thumb_1_joint", "6", 0.3, 1.2),
]

RIGHT_HAND_SERVICE_MAP: List[Tuple[str, str, float, float]] = [
    ("right_little_1_joint", "1", 0.0, 1.1),
    ("right_ring_1_joint", "2", 0.0, 1.1),
    ("right_middle_1_joint", "3", 0.0, 1.1),
    ("right_index_1_joint", "4", 0.0, 1.1),
    ("right_thumb_2_joint", "5", 0.0, 0.5),
    ("right_thumb_1_joint", "6", 0.3, 1.2),
]

RPM_PER_RAD_PER_SEC = 60.0 / (2.0 * math.pi)
RAD_PER_SEC_PER_RPM = (2.0 * math.pi) / 60.0

DEFAULT_RIGHT_HAND_INPUT_TOPIC = "/dummy_control/right_hand_joint_states"
DEFAULT_LEFT_HAND_INPUT_TOPIC = "/dummy_control/left_hand_joint_states"
DEFAULT_RIGHT_HAND_COMMAND_TOPIC = "/inspire_hand/ctrl/right_hand"
DEFAULT_LEFT_HAND_COMMAND_TOPIC = "/inspire_hand/ctrl/left_hand"
DEFAULT_RIGHT_HAND_STATE_TOPIC = "/inspire_hand/state/right_hand"
DEFAULT_LEFT_HAND_STATE_TOPIC = "/inspire_hand/state/left_hand"
DEFAULT_RIGHT_HAND_SERVICE_NAME = "/inspire_hand/set_angle_flexible/right_hand"
DEFAULT_LEFT_HAND_SERVICE_NAME = "/inspire_hand/set_angle_flexible/left_hand"

CONTROL_DOMAIN_ALIASES = {
    "left": "left_arm",
    "right": "right_arm",
    "right_arm_hand": "right_full",
}

PLACEHOLDER_CONTROL_DOMAINS = {"upper_body", "full_body"}


def _normalize_control_domain(raw_value: str) -> str:
    control_domain = raw_value.strip().lower()
    if not control_domain:
        return "right_full"
    return CONTROL_DOMAIN_ALIASES.get(control_domain, control_domain)


def _resolve_default_hand_interface(
    configured_value: str,
    control_domain: str,
    right_default: str,
    left_default: str,
) -> str:
    if control_domain == "left_full" and configured_value == right_default:
        return left_default
    return configured_value

class FeedbackControlBridge(Node):
    def __init__(self) -> None:
        super().__init__("feedback_control_bridge")

        self.declare_parameter("control_domain", "")
        self.declare_parameter("arm_side", "right_full")
        self.declare_parameter("command_topic", "/arm/cmd_pos")
        self.declare_parameter("input_joint_state_topic", "/arm/command_joint_states")
        self.declare_parameter("input_hand_joint_state_topic", DEFAULT_RIGHT_HAND_INPUT_TOPIC)
        self.declare_parameter("hand_command_topic", DEFAULT_RIGHT_HAND_COMMAND_TOPIC)
        self.declare_parameter("hand_command_interface", "service")
        self.declare_parameter("hand_position_scale", 1.0)
        self.declare_parameter("hand_state_topic", DEFAULT_RIGHT_HAND_STATE_TOPIC)
        self.declare_parameter("status_topic", "/arm/status")
        self.declare_parameter("head_status_topic", "/head/status")
        self.declare_parameter("leg_status_topic", "/leg/status")
        self.declare_parameter("waist_status_topic", "/waist/status")
        self.declare_parameter("use_status", True)
        self.declare_parameter("publish_joint_states", True)
        self.declare_parameter("mirror_commanded_joint_states", True)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("joint_state_frame_id", "")
        self.declare_parameter("current_limit", 5.0)
        self.declare_parameter("default_velocity_rpm", 5.0)
        self.declare_parameter("min_velocity_rpm", 0.1)
        self.declare_parameter("velocity_rpm", "")
        self.declare_parameter(
            "hand_service_name",
            DEFAULT_RIGHT_HAND_SERVICE_NAME,
        )
        self.declare_parameter("hand_service_wait_sec", 0.3)

        raw_control_domain = str(self.get_parameter("control_domain").value)
        if not raw_control_domain.strip():
            raw_control_domain = str(self.get_parameter("arm_side").value)
        control_domain = _normalize_control_domain(raw_control_domain)
        if control_domain not in {
            "left_arm",
            "right_arm",
            "left_full",
            "right_full",
            "upper_body",
            "full_body",
        }:
            raise ValueError(
                "feedback_control_bridge: control_domain must be one of "
                "'right_arm', 'left_arm', 'right_full', 'left_full', "
                "'upper_body', 'full_body'"
            )

        self._placeholder_domain = control_domain in PLACEHOLDER_CONTROL_DOMAINS
        if control_domain == "left_arm":
            arm_mapping = LEFT_ARM_MAP
            self._hand_enabled = False
            self._hand_joint_order: List[str] = []
            self._hand_service_map = []
        elif control_domain == "right_arm":
            arm_mapping = RIGHT_ARM_MAP
            self._hand_enabled = False
            self._hand_joint_order = []
            self._hand_service_map = []
        elif control_domain == "left_full":
            arm_mapping = LEFT_ARM_MAP
            self._hand_enabled = True
            self._hand_joint_order = [joint for joint, _, _, _ in LEFT_HAND_SERVICE_MAP]
            self._hand_service_map = LEFT_HAND_SERVICE_MAP
        else:
            arm_mapping = RIGHT_ARM_MAP
            self._hand_enabled = True
            self._hand_joint_order = [joint for joint, _, _, _ in RIGHT_HAND_SERVICE_MAP]
            self._hand_service_map = RIGHT_HAND_SERVICE_MAP

        self._arm_joint_order: List[str] = [joint for _, joint in arm_mapping]
        self._command_joint_order: List[str] = list(self._arm_joint_order) + list(self._hand_joint_order)
        self._state_joint_order: List[str] = (
            [joint for _, joint in LEFT_ARM_MAP]
            + [joint for _, joint in RIGHT_ARM_MAP]
            + list(self._hand_joint_order)
            + [WAIST_STATE_JOINT]
            + [joint for _, joint in HEAD_MAP]
            + [joint for _, joint in LEFT_LEG_MAP]
            + [joint for _, joint in RIGHT_LEG_MAP]
        )
        self._id_by_joint: Dict[str, int] = {joint: motor_id for motor_id, joint in arm_mapping}
        self._arm_state_joint_by_id: Dict[int, str] = {
            motor_id: joint for motor_id, joint in (LEFT_ARM_MAP + RIGHT_ARM_MAP)
        }
        self._hand_limits_by_joint: Dict[str, Tuple[float, float]] = {
            joint: (lower, upper) for joint, _, lower, upper in self._hand_service_map
        }
        self._head_state_joint_by_id: Dict[int, str] = {
            motor_id: joint for motor_id, joint in HEAD_MAP
        }
        self._leg_state_joint_by_id: Dict[int, str] = {
            motor_id: joint for motor_id, joint in (LEFT_LEG_MAP + RIGHT_LEG_MAP)
        }

        self._command_topic = str(self.get_parameter("command_topic").value)
        self._input_joint_state_topic = str(self.get_parameter("input_joint_state_topic").value)
        self._input_hand_joint_state_topic = _resolve_default_hand_interface(
            str(self.get_parameter("input_hand_joint_state_topic").value),
            control_domain,
            DEFAULT_RIGHT_HAND_INPUT_TOPIC,
            DEFAULT_LEFT_HAND_INPUT_TOPIC,
        )
        self._hand_command_topic = _resolve_default_hand_interface(
            str(self.get_parameter("hand_command_topic").value),
            control_domain,
            DEFAULT_RIGHT_HAND_COMMAND_TOPIC,
            DEFAULT_LEFT_HAND_COMMAND_TOPIC,
        )
        self._hand_command_interface = str(
            self.get_parameter("hand_command_interface").value
        ).strip().lower()
        self._hand_position_scale = float(self.get_parameter("hand_position_scale").value)
        self._hand_state_topic = _resolve_default_hand_interface(
            str(self.get_parameter("hand_state_topic").value),
            control_domain,
            DEFAULT_RIGHT_HAND_STATE_TOPIC,
            DEFAULT_LEFT_HAND_STATE_TOPIC,
        )
        self._status_topic = str(self.get_parameter("status_topic").value)
        self._head_status_topic = str(self.get_parameter("head_status_topic").value)
        self._leg_status_topic = str(self.get_parameter("leg_status_topic").value)
        self._waist_status_topic = str(self.get_parameter("waist_status_topic").value)
        self._use_status = bool(self.get_parameter("use_status").value)
        self._publish_joint_states = bool(self.get_parameter("publish_joint_states").value)
        self._mirror_commanded_joint_states = bool(
            self.get_parameter("mirror_commanded_joint_states").value
        )
        self._joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self._joint_state_frame = str(self.get_parameter("joint_state_frame_id").value)
        self._current_limit = float(self.get_parameter("current_limit").value)
        self._default_velocity_rpm = float(self.get_parameter("default_velocity_rpm").value)
        self._min_velocity_rpm = float(self.get_parameter("min_velocity_rpm").value)
        self._velocity_override = self._parse_velocity_override(
            self.get_parameter("velocity_rpm").value
        )
        self._hand_service_name = _resolve_default_hand_interface(
            str(self.get_parameter("hand_service_name").value),
            control_domain,
            DEFAULT_RIGHT_HAND_SERVICE_NAME,
            DEFAULT_LEFT_HAND_SERVICE_NAME,
        )
        self._hand_service_wait_sec = float(self.get_parameter("hand_service_wait_sec").value)

        self._cmd_pub = self.create_publisher(
            CmdSetMotorPosition,
            self._command_topic,
            10,
        )
        self._joint_state_pub = None
        if self._publish_joint_states:
            self._joint_state_pub = self.create_publisher(JointState, self._joint_state_topic, 5)

        self._last_positions: Dict[str, float] = {}
        self._last_velocities: Dict[str, float] = {}
        self._last_currents: Dict[str, float] = {}
        self._last_warn_times: Dict[str, float] = {}
        for joint_name, _, lower, _ in self._hand_service_map:
            self._last_positions[joint_name] = lower

        self._lock = threading.RLock()
        self._status_sub = None
        self._head_status_sub = None
        self._leg_status_sub = None
        self._waist_status_sub = None
        if self._use_status or self._publish_joint_states:
            self._status_sub = self.create_subscription(
                MotorStatusMsg,
                self._status_topic,
                self._status_cb,
                10,
            )
            self._head_status_sub = self.create_subscription(
                MotorStatusMsg,
                self._head_status_topic,
                self._head_status_cb,
                10,
            )
            self._leg_status_sub = self.create_subscription(
                MotorStatusMsg,
                self._leg_status_topic,
                self._leg_status_cb,
                10,
            )
            self._waist_status_sub = self.create_subscription(
                WaistMotorStatus,
                self._waist_status_topic,
                self._waist_status_cb,
                10,
            )
        else:
            self.get_logger().warn(
                "feedback_control_bridge: use_status:=false and publish_joint_states:=false; "
                f"feedback will mirror commanded values instead of subscribing to "
                f"{self._status_topic}, {self._head_status_topic}, {self._leg_status_topic}, "
                f"and {self._waist_status_topic}"
            )

        self._command_sub = self.create_subscription(
            JointState,
            self._input_joint_state_topic,
            self._command_cb,
            10,
        )

        self._hand_command_sub = None
        self._hand_state_sub = None
        self._hand_client = None
        self._hand_topic_pub = None
        self._pending_hand_futures: List[Any] = []
        if self._hand_enabled:
            self._hand_command_sub = self.create_subscription(
                JointState,
                self._input_hand_joint_state_topic,
                self._hand_command_cb,
                10,
            )
            if self._hand_command_interface == "service":
                self._hand_client = self.create_client(
                    SetAngleFlexible,
                    self._hand_service_name,
                )
                if not self._hand_client.wait_for_service(
                    timeout_sec=max(0.0, self._hand_service_wait_sec)
                ):
                    self.get_logger().warn(
                        f"feedback_control_bridge: hand service unavailable at startup: {self._hand_service_name}"
                    )
            elif self._hand_command_interface == "topic":
                self._hand_topic_pub = self.create_publisher(
                    JointState,
                    self._hand_command_topic,
                    10,
                )
            else:
                raise ValueError(
                    "feedback_control_bridge: hand_command_interface must be 'topic' or 'service'"
                )
            if self._use_status or self._publish_joint_states:
                self._hand_state_sub = self.create_subscription(
                    JointState,
                    self._hand_state_topic,
                    self._hand_state_cb,
                    10,
                )

        if self._placeholder_domain:
            self.get_logger().warn(
                "feedback_control_bridge: control_domain='%s' is a placeholder; "
                "head and leg bridging are not implemented yet, so this mode currently behaves "
                "like 'right_full'." % control_domain
            )

        self.get_logger().info(
            "feedback_control_bridge: direct command mode "
            f"({self._input_joint_state_topic} -> {self._command_topic}), "
            f"status_topic={self._status_topic}, "
            f"head_status_topic={self._head_status_topic}, "
            f"leg_status_topic={self._leg_status_topic}, "
            f"waist_status_topic={self._waist_status_topic}, "
            f"control_domain={control_domain}, command_joints={len(self._command_joint_order)}, "
            f"state_joints={len(self._state_joint_order)}, "
            f"hand_input={self._input_hand_joint_state_topic if self._hand_enabled else '<disabled>'}, "
            f"hand_output={self._hand_command_topic if self._hand_enabled and self._hand_command_interface == 'topic' else '<service>' if self._hand_enabled else '<disabled>'}, "
            f"hand_interface={self._hand_command_interface if self._hand_enabled else '<disabled>'}, "
            f"hand_state={self._hand_state_topic if self._hand_enabled else '<disabled>'}, "
            f"hand_service={self._hand_service_name if self._hand_enabled else '<disabled>'}"
        )

    def _status_cb(self, msg: Any) -> None:
        for motor_state in msg.status:
            motor_id = int(motor_state.name)
            joint_name = self._arm_state_joint_by_id.get(motor_id)
            if joint_name:
                self._last_positions[joint_name] = float(motor_state.pos)
                self._last_velocities[joint_name] = float(motor_state.speed)
                self._last_currents[joint_name] = float(motor_state.current)

        self._publish_joint_state(self._stamp_or_now(msg.header.stamp))

    def _leg_status_cb(self, msg: Any) -> None:
        for motor_state in msg.status:
            motor_id = int(motor_state.name)
            joint_name = self._leg_state_joint_by_id.get(motor_id)
            if joint_name:
                self._last_positions[joint_name] = float(motor_state.pos)
                self._last_velocities[joint_name] = float(motor_state.speed)
                self._last_currents[joint_name] = float(motor_state.current)

        self._publish_joint_state(self._stamp_or_now(msg.header.stamp))

    def _head_status_cb(self, msg: Any) -> None:
        for motor_state in msg.status:
            motor_id = int(motor_state.name)
            joint_name = self._head_state_joint_by_id.get(motor_id)
            if joint_name:
                self._last_positions[joint_name] = float(motor_state.pos)
                self._last_velocities[joint_name] = float(motor_state.speed)
                self._last_currents[joint_name] = float(motor_state.current)

        self._publish_joint_state(self._stamp_or_now(msg.header.stamp))

    def _waist_status_cb(self, msg: WaistMotorStatus) -> None:
        self._last_positions[WAIST_STATE_JOINT] = float(msg.pos)
        self._last_velocities[WAIST_STATE_JOINT] = float(msg.vel)
        self._last_currents[WAIST_STATE_JOINT] = float(msg.cur)
        self._publish_joint_state(self._stamp_or_now(msg.header.stamp))

    def _publish_joint_state(self, stamp: Any) -> None:
        if not self._publish_joint_states or self._joint_state_pub is None:
            return

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.header.frame_id = self._joint_state_frame
        joint_state.name = list(self._state_joint_order)
        joint_state.position = [self._last_positions.get(name, 0.0) for name in self._state_joint_order]
        joint_state.velocity = [self._last_velocities.get(name, 0.0) for name in self._state_joint_order]
        joint_state.effort = [self._last_currents.get(name, 0.0) for name in self._state_joint_order]
        self._joint_state_pub.publish(joint_state)

    def _hand_state_cb(self, msg: JointState) -> None:
        if not self._hand_enabled:
            return

        joint_index = self._resolve_hand_joint_index(msg)
        positions = list(msg.position) if msg.position else []
        velocities = list(msg.velocity) if msg.velocity else []
        efforts = list(msg.effort) if msg.effort else []

        for joint_name in self._hand_joint_order:
            src_idx = joint_index.get(joint_name)
            if src_idx is None:
                continue
            if src_idx < len(positions):
                self._last_positions[joint_name] = self._hand_feedback_to_visual_position(
                    joint_name,
                    float(positions[src_idx]),
                )
            if src_idx < len(velocities):
                self._last_velocities[joint_name] = float(velocities[src_idx])
            if src_idx < len(efforts):
                self._last_currents[joint_name] = float(efforts[src_idx])

        self._publish_joint_state(self._stamp_or_now(msg.header.stamp))

    def _parse_velocity_override(
        self,
        raw: Union[str, int, float, List[float], Tuple[float, ...], None],
    ) -> Dict[str, float]:
        if raw in (None, ""):
            return {}

        parsed: Union[float, List[float], None] = None
        if isinstance(raw, (int, float)):
            parsed = float(raw)
        elif isinstance(raw, str):
            try:
                value = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                self.get_logger().warn(f"feedback_control_bridge: could not parse velocity_rpm '{raw}'")
                value = None
            if isinstance(value, (int, float)):
                parsed = float(value)
            elif isinstance(value, (list, tuple)):
                try:
                    parsed = [float(item) for item in value]
                except (TypeError, ValueError):
                    parsed = None
        elif isinstance(raw, (list, tuple)):
            try:
                parsed = [float(item) for item in raw]
            except (TypeError, ValueError):
                parsed = None

        if parsed is None:
            self.get_logger().warn(
                "feedback_control_bridge: invalid velocity_rpm parameter; ignoring override"
            )
            return {}

        if isinstance(parsed, list):
            if len(parsed) != len(self._arm_joint_order):
                self.get_logger().warn(
                    "feedback_control_bridge: velocity_rpm length mismatch; ignoring override"
                )
                return {}
            return {joint: value for joint, value in zip(self._arm_joint_order, parsed)}

        return {joint: parsed for joint in self._arm_joint_order}

    def _resolve_joint_index(self, msg: JointState) -> Dict[str, int]:
        if msg.name:
            return {name: idx for idx, name in enumerate(msg.name)}
        return {name: idx for idx, name in enumerate(self._command_joint_order)}

    def _resolve_hand_joint_index(self, msg: JointState) -> Dict[str, int]:
        if msg.name:
            return {name: idx for idx, name in enumerate(msg.name)}
        return {name: idx for idx, name in enumerate(self._hand_joint_order)}

    def _command_from_joint_state(self, msg: JointState) -> Any:
        cmd = CmdSetMotorPosition()
        cmd.header.stamp = self._stamp_or_now(msg.header.stamp)
        cmd.cmds = []

        joint_index = self._resolve_joint_index(msg)
        positions = list(msg.position) if msg.position else []
        velocities = list(msg.velocity) if msg.velocity else []

        for joint_name in self._arm_joint_order:
            motor_id = self._id_by_joint[joint_name]
            entry = SetMotorPosition()
            entry.name = motor_id
            src_idx = joint_index.get(joint_name)

            if src_idx is not None and src_idx < len(positions):
                entry.pos = float(positions[src_idx])
            else:
                entry.pos = self._last_positions.get(joint_name, 0.0)

            override_spd = self._velocity_override.get(joint_name)
            if override_spd is not None:
                entry.spd = float(override_spd)
                velocity_rad = float(override_spd) * RAD_PER_SEC_PER_RPM
            elif src_idx is not None and src_idx < len(velocities):
                velocity_rad = float(velocities[src_idx])
                entry.spd = velocity_rad * RPM_PER_RAD_PER_SEC
            else:
                entry.spd = self._default_velocity_rpm
                velocity_rad = self._default_velocity_rpm * RAD_PER_SEC_PER_RPM

            if abs(entry.spd) < self._min_velocity_rpm:
                entry.spd = self._default_velocity_rpm
                velocity_rad = self._default_velocity_rpm * RAD_PER_SEC_PER_RPM

            entry.cur = self._current_limit
            self._last_positions[joint_name] = float(entry.pos)
            self._last_velocities[joint_name] = float(velocity_rad)
            self._last_currents[joint_name] = float(entry.cur)
            cmd.cmds.append(entry)

        return cmd

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _hand_feedback_to_visual_position(self, joint_name: str, value: float) -> float:
        limits = self._hand_limits_by_joint.get(joint_name)
        if limits is None:
            return value

        lower, upper = limits
        span = max(1e-6, float(upper - lower))
        ratio = float(value)
        if ratio > 1.5:
            ratio /= 100.0
        ratio = self._clamp01(ratio)
        return float(upper - ratio * span)

    def _hand_visual_position_to_open_ratio(self, joint_name: str, value: float) -> float:
        limits = self._hand_limits_by_joint.get(joint_name)
        if limits is None:
            return self._clamp01(value)

        lower, upper = limits
        span = max(1e-6, float(upper - lower))
        closed_ratio = self._clamp01((float(value) - lower) / span)
        return float(1.0 - closed_ratio)

    def _command_hand_from_joint_state(self, msg: JointState) -> bool:
        if not self._hand_enabled:
            return False

        if self._hand_command_interface == "topic":
            return self._publish_hand_topic_command(msg)
        if self._hand_client is None:
            return False

        if not self._hand_client.service_is_ready():
            if not self._hand_client.wait_for_service(timeout_sec=max(0.0, self._hand_service_wait_sec)):
                self._warn_throttled(
                    "hand_service_unavailable",
                    2.0,
                    f"feedback_control_bridge: hand service unavailable: {self._hand_service_name}",
                )
                return False

        joint_index = self._resolve_joint_index(msg)
        positions = list(msg.position) if msg.position else []
        req = SetAngleFlexible.Request()
        req.name = []
        req.angle_ratio = []

        for joint_name, finger_name, lower, upper in self._hand_service_map:
            src_idx = joint_index.get(joint_name)
            if src_idx is None or src_idx >= len(positions):
                continue
            pos = float(positions[src_idx])
            self._last_positions[joint_name] = pos
            req.name.append(str(finger_name))
            req.angle_ratio.append(self._hand_visual_position_to_open_ratio(joint_name, pos))

        if not req.name:
            return False

        future = self._hand_client.call_async(req)
        self._pending_hand_futures.append(future)
        future.add_done_callback(self._hand_response_cb)
        return True

    def _publish_hand_topic_command(self, msg: JointState) -> bool:
        if self._hand_topic_pub is None:
            return False

        joint_index = self._resolve_joint_index(msg)
        positions = list(msg.position) if msg.position else []
        out = JointState()
        out.header.stamp = self._stamp_or_now(msg.header.stamp)
        out.name = []
        out.position = []

        for joint_name, finger_name, lower, upper in self._hand_service_map:
            src_idx = joint_index.get(joint_name)
            if src_idx is None or src_idx >= len(positions):
                continue
            pos = float(positions[src_idx])
            self._last_positions[joint_name] = pos
            out.name.append(str(finger_name))
            out.position.append(
                self._hand_visual_position_to_open_ratio(joint_name, pos) * self._hand_position_scale
            )

        if not out.name:
            return False

        self._hand_topic_pub.publish(out)
        return True

    def _hand_response_cb(self, future: Any) -> None:
        try:
            self._pending_hand_futures.remove(future)
        except ValueError:
            pass

        try:
            response = future.result()
        except Exception as exc:  # noqa: BLE001
            self._warn_throttled(
                "hand_service_failed",
                1.0,
                f"feedback_control_bridge: hand service call failed: {exc}",
            )
            return

        accepted = bool(getattr(response, "angle_accepted", True))
        if not accepted:
            self._warn_throttled(
                "hand_service_rejected",
                1.0,
                "feedback_control_bridge: hand service rejected angle request",
            )

    def _command_cb(self, msg: JointState) -> None:
        with self._lock:
            cmd = self._command_from_joint_state(msg)
            if cmd.cmds:
                self._cmd_pub.publish(cmd)
            else:
                self.get_logger().warn(
                    f"feedback_control_bridge: empty arm command, skipping publish to {self._command_topic}"
                )

            if self._hand_enabled:
                self._command_hand_from_joint_state(msg)

            if self._mirror_commanded_joint_states:
                self._publish_joint_state(self._stamp_or_now(msg.header.stamp))

    def _hand_command_cb(self, msg: JointState) -> None:
        with self._lock:
            if self._hand_enabled:
                self._command_hand_from_joint_state(msg)
                if self._mirror_commanded_joint_states:
                    self._publish_joint_state(self._stamp_or_now(msg.header.stamp))

    def _warn_throttled(self, key: str, period_sec: float, message: str) -> None:
        now_sec = self.get_clock().now().nanoseconds / 1e9
        last_sec = self._last_warn_times.get(key, float("-inf"))
        if now_sec - last_sec >= period_sec:
            self._last_warn_times[key] = now_sec
            self.get_logger().warn(message)

    def _stamp_or_now(self, stamp: Any) -> Any:
        if getattr(stamp, "sec", 0) == 0 and getattr(stamp, "nanosec", 0) == 0:
            return self.get_clock().now().to_msg()
        return stamp


def main(args: Iterable[str] | None = None) -> int:
    rclpy.init(args=args)
    try:
        node = FeedbackControlBridge()
    except Exception as exc:  # noqa: BLE001
        rclpy.logging.get_logger("feedback_control_bridge").error(str(exc))
        if rclpy.ok():
            rclpy.shutdown()
        return 1

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0
