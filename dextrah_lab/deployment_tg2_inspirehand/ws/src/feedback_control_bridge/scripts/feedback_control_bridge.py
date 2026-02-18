#!/usr/bin/env python3
"""Direct command + joint-state feedback bridge for Tiangong arm/hand.

This node bypasses MoveIt and accepts direct joint commands (sensor_msgs/JointState)
on a configurable topic. It converts the command into bodyctrl_msgs/CmdSetMotorPosition
messages for the robot and optionally republishes /arm/status as /joint_states.
"""

from __future__ import annotations

import ast
import math
import threading
from typing import Dict, Iterable, List, Tuple, Union

import rospy
from sensor_msgs.msg import JointState
from bodyctrl_msgs.msg import CmdSetMotorPosition, MotorStatusMsg, SetMotorPosition
from bodyctrl_msgs.srv import set_angle_flexible, set_angle_flexibleRequest

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
RIGHT_HAND_SERVICE_MAP: List[Tuple[str, str, float, float]] = [
    ("little_joint_0", "1", 0.0, 1.1),
    ("ring_joint_0", "2", 0.0, 1.1),
    ("middle_joint_0", "3", 0.0, 1.1),
    ("index_joint_0", "4", 0.0, 1.1),
    ("thumb_joint_1", "5", 0.0, 0.5),
    ("thumb_joint_0", "6", 0.3, 1.2),
]

RPM_PER_RAD_PER_SEC = 60.0 / (2.0 * math.pi)
RAD_PER_SEC_PER_RPM = (2.0 * math.pi) / 60.0


class FeedbackControlBridge:
    """Send direct joint commands while republishing live joint states."""

    def __init__(self) -> None:
        arm_side = rospy.get_param("~arm_side", "right_full").strip().lower()
        if arm_side not in {"left", "right", "right_full", "right_arm_hand"}:
            rospy.logerr(
                "feedback_control_bridge: arm_side must be one of "
                "'left', 'right', 'right_full', 'right_arm_hand'"
            )
            raise ValueError("Invalid arm_side parameter")

        if arm_side == "left":
            arm_mapping = LEFT_ARM_MAP
            self._hand_enabled = False
            self._hand_joint_order: List[str] = []
        elif arm_side in {"right_full", "right_arm_hand"}:
            arm_mapping = RIGHT_ARM_MAP
            self._hand_enabled = True
            self._hand_joint_order = [joint_name for joint_name, _, _, _ in RIGHT_HAND_SERVICE_MAP]
        else:
            arm_mapping = RIGHT_ARM_MAP
            self._hand_enabled = False
            self._hand_joint_order = []
        self._arm_joint_order: List[str] = [joint for _, joint in arm_mapping]
        self._joint_order: List[str] = list(self._arm_joint_order) + list(self._hand_joint_order)
        self._id_by_joint: Dict[str, int] = {joint: motor_id for motor_id, joint in arm_mapping}
        self._joint_by_id: Dict[int, str] = {motor_id: joint for motor_id, joint in arm_mapping}

        self._command_topic = rospy.get_param("~command_topic", "/arm/cmd_pos")
        self._input_joint_state_topic = rospy.get_param(
            "~input_joint_state_topic", "/arm/command_joint_states"
        )
        self._input_hand_joint_state_topic = rospy.get_param(
            "~input_hand_joint_state_topic", "/hand/command_joint_states"
        )
        self._status_topic = rospy.get_param("~status_topic", "/arm/status")
        self._use_status = rospy.get_param("~use_status", True)
        self._publish_joint_states = rospy.get_param("~publish_joint_states", True)
        self._joint_state_topic = rospy.get_param("~joint_state_topic", "/joint_states")
        self._joint_state_frame = rospy.get_param("~joint_state_frame_id", "")
        self._current_limit = float(rospy.get_param("~current_limit", 5.0))
        self._default_velocity_rpm = float(rospy.get_param("~default_velocity_rpm", 10.0))
        self._min_velocity_rpm = float(rospy.get_param("~min_velocity_rpm", 0.1))
        self._velocity_override = self._parse_velocity_override(
            rospy.get_param("~velocity_rpm", None)
        )
        self._hand_service_name = rospy.get_param(
            "~hand_service_name", "/inspire_hand/set_angle_flexible/right_hand"
        )
        self._hand_service_wait_sec = float(rospy.get_param("~hand_service_wait_sec", 0.3))
        self._hand_service_proxy = None

        self._cmd_pub = rospy.Publisher(self._command_topic, CmdSetMotorPosition, queue_size=10)
        self._joint_state_pub = None
        if self._publish_joint_states:
            self._joint_state_pub = rospy.Publisher(self._joint_state_topic, JointState, queue_size=5)

        self._last_positions: Dict[str, float] = {}
        self._last_velocities: Dict[str, float] = {}
        self._last_currents: Dict[str, float] = {}
        for joint_name, _, lower, _ in RIGHT_HAND_SERVICE_MAP:
            self._last_positions[joint_name] = lower

        needs_status = self._use_status or self._publish_joint_states
        if needs_status:
            self._status_sub = rospy.Subscriber(
                self._status_topic,
                MotorStatusMsg,
                self._status_cb,
                queue_size=10,
            )
        else:
            self._status_sub = None
            rospy.logwarn(
                "feedback_control_bridge: use_status:=false and publish_joint_states:=false -> "
                "no subscription to %s; feedback will mirror commanded values",
                self._status_topic,
            )

        self._lock = threading.RLock()
        self._command_sub = rospy.Subscriber(
            self._input_joint_state_topic,
            JointState,
            self._command_cb,
            queue_size=10,
        )
        self._hand_command_sub = None
        if self._hand_enabled:
            self._hand_command_sub = rospy.Subscriber(
                self._input_hand_joint_state_topic,
                JointState,
                self._hand_command_cb,
                queue_size=10,
            )
            try:
                rospy.wait_for_service(self._hand_service_name, timeout=max(0.0, self._hand_service_wait_sec))
                self._hand_service_proxy = rospy.ServiceProxy(
                    self._hand_service_name,
                    set_angle_flexible,
                    persistent=True,
                )
            except (rospy.ROSException, rospy.ROSInterruptException):
                rospy.logwarn(
                    "feedback_control_bridge: hand service unavailable at startup: %s",
                    self._hand_service_name,
                )

        rospy.loginfo(
            "feedback_control_bridge: direct command mode (%s -> %s), arm_side=%s, joints=%d, hand_input=%s, hand_service=%s",
            self._input_joint_state_topic,
            self._command_topic,
            arm_side,
            len(self._joint_order),
            self._input_hand_joint_state_topic if self._hand_enabled else "<disabled>",
            self._hand_service_name if self._hand_enabled else "<disabled>",
        )

    # --------------------------------------------------------------- status
    def _status_cb(self, msg: MotorStatusMsg) -> None:
        for motor_state in msg.status:
            motor_id = int(motor_state.name)
            joint_name = self._joint_by_id.get(motor_id)
            if joint_name:
                self._last_positions[joint_name] = motor_state.pos
                self._last_velocities[joint_name] = motor_state.speed
                self._last_currents[joint_name] = motor_state.current

        self._publish_joint_state(msg.header.stamp if msg.header else rospy.Time.now())

    def _publish_joint_state(self, stamp: rospy.Time) -> None:
        if not self._publish_joint_states or self._joint_state_pub is None:
            return

        joint_state = JointState()
        joint_state.header.stamp = stamp
        joint_state.header.frame_id = self._joint_state_frame
        joint_state.name = list(self._joint_order)
        joint_state.position = [
            self._last_positions.get(name, float("nan")) for name in self._joint_order
        ]
        joint_state.velocity = [
            self._last_velocities.get(name, float("nan")) for name in self._joint_order
        ]
        joint_state.effort = [
            self._last_currents.get(name, float("nan")) for name in self._joint_order
        ]
        self._joint_state_pub.publish(joint_state)

    # --------------------------------------------------------------- helpers
    def _parse_velocity_override(
        self, raw: Union[str, int, float, List[float], None]
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
                rospy.logwarn("feedback_control_bridge: could not parse velocity_rpm '%s'", raw)
                value = None
            if isinstance(value, (int, float)):
                parsed = float(value)
            elif isinstance(value, (list, tuple)):
                try:
                    parsed = [float(v) for v in value]
                except (TypeError, ValueError):
                    parsed = None
        elif isinstance(raw, (list, tuple)):
            try:
                parsed = [float(v) for v in raw]
            except (TypeError, ValueError):
                parsed = None

        joint_names = list(self._arm_joint_order)
        if parsed is None:
            rospy.logwarn("feedback_control_bridge: invalid velocity_rpm parameter; ignoring override")
            return {}

        if isinstance(parsed, list):
            if len(parsed) != len(joint_names):
                rospy.logwarn(
                    "feedback_control_bridge: velocity_rpm length %d mismatch joint count %d; ignoring override",
                    len(parsed),
                    len(joint_names),
                )
                return {}
            return {joint: value for joint, value in zip(joint_names, parsed)}

        return {joint: parsed for joint in joint_names}
    def _resolve_joint_index(self, msg: JointState) -> Dict[str, int]:
        if msg.name:
            return {name: idx for idx, name in enumerate(msg.name)}
        return {name: idx for idx, name in enumerate(self._joint_order)}

    def _command_from_joint_state(self, msg: JointState) -> CmdSetMotorPosition:
        cmd = CmdSetMotorPosition()
        cmd.header.stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
        cmd.cmds = []

        joint_index = self._resolve_joint_index(msg)
        positions = list(msg.position) if msg.position else []
        velocities = list(msg.velocity) if msg.velocity else None

        for joint_name in self._arm_joint_order:
            motor_id = self._id_by_joint[joint_name]
            entry = SetMotorPosition()
            entry.name = motor_id
            src_idx = joint_index.get(joint_name)

            if src_idx is not None and src_idx < len(positions):
                entry.pos = float(positions[src_idx])
            else:
                entry.pos = self._last_positions.get(joint_name, 0.0)

            override_spd = self._velocity_override.get(joint_name) if self._velocity_override else None
            if override_spd is not None:
                entry.spd = override_spd
                velocity_rad = override_spd * RAD_PER_SEC_PER_RPM
            elif velocities and src_idx is not None and src_idx < len(velocities):
                entry.spd = float(velocities[src_idx]) * RPM_PER_RAD_PER_SEC
                velocity_rad = float(velocities[src_idx])
            else:
                entry.spd = self._default_velocity_rpm
                velocity_rad = self._default_velocity_rpm * RAD_PER_SEC_PER_RPM

            if abs(entry.spd) < self._min_velocity_rpm:
                entry.spd = self._default_velocity_rpm
                velocity_rad = self._default_velocity_rpm * RAD_PER_SEC_PER_RPM

            entry.cur = self._current_limit
            self._last_positions[joint_name] = entry.pos
            self._last_velocities[joint_name] = velocity_rad
            self._last_currents[joint_name] = entry.cur
            cmd.cmds.append(entry)

        return cmd

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    def _command_hand_from_joint_state(self, msg: JointState) -> bool:
        if not self._hand_enabled:
            return False

        joint_index = self._resolve_joint_index(msg)
        positions = list(msg.position) if msg.position else []
        req = set_angle_flexibleRequest()
        req.name = []
        req.angleRatio = []

        for joint_name, finger_name, lower, upper in RIGHT_HAND_SERVICE_MAP:
            src_idx = joint_index.get(joint_name)
            # Partial update semantics: only joints present in the incoming
            # message are forwarded to the hand service.
            if src_idx is None or src_idx >= len(positions):
                continue
            pos = float(positions[src_idx])
            self._last_positions[joint_name] = pos
            span = max(1e-6, float(upper - lower))
            ratio = self._clamp01((pos - lower) / span)
            req.name.append(str(finger_name))
            req.angleRatio.append(float(ratio))

        if not req.name:
            return False

        if self._hand_service_proxy is None:
            try:
                rospy.wait_for_service(self._hand_service_name, timeout=max(0.0, self._hand_service_wait_sec))
                self._hand_service_proxy = rospy.ServiceProxy(
                    self._hand_service_name,
                    set_angle_flexible,
                    persistent=True,
                )
            except (rospy.ROSException, rospy.ROSInterruptException):
                rospy.logwarn_throttle(
                    2.0,
                    "feedback_control_bridge: hand service unavailable: %s",
                    self._hand_service_name,
                )
                return False

        try:
            resp = self._hand_service_proxy(req)
            accepted = bool(getattr(resp, "angle_accepted", True))
            if not accepted:
                rospy.logwarn_throttle(1.0, "feedback_control_bridge: hand service rejected angle request")
            return accepted
        except rospy.ServiceException as exc:
            self._hand_service_proxy = None
            rospy.logwarn_throttle(1.0, "feedback_control_bridge: hand service call failed: %s", str(exc))
            return False

    # -------------------------------------------------------------- command
    def _command_cb(self, msg: JointState) -> None:
        with self._lock:
            cmd = self._command_from_joint_state(msg)
            if cmd.cmds:
                self._cmd_pub.publish(cmd)
            else:
                rospy.logwarn("feedback_control_bridge: empty arm command, skipping /arm/cmd_pos publish")

            if self._hand_enabled:
                self._command_hand_from_joint_state(msg)

            stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
            self._publish_joint_state(stamp)

    def _hand_command_cb(self, msg: JointState) -> None:
        with self._lock:
            if self._hand_enabled:
                self._command_hand_from_joint_state(msg)
                stamp = msg.header.stamp if msg.header.stamp else rospy.Time.now()
                self._publish_joint_state(stamp)


def main(argv: Iterable[str]) -> int:
    rospy.init_node("feedback_control_bridge")
    FeedbackControlBridge()
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main([]))
