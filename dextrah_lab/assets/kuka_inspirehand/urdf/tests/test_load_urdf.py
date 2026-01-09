"""Simple loader to verify the iiwa7 URDF in PyBullet."""

import argparse
import os
import tempfile
import time

import pybullet as p
import pybullet_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI (otherwise headless).")
    parser.add_argument("--run_seconds", type=float, default=-1.0, help="How long to keep the sim alive; -1 for indefinitely.")
    parser.add_argument("--realtime", action="store_true", help="Enable PyBullet real-time simulation (handy with sliders).")
    parser.add_argument("--axis_len", type=float, default=0.1, help="Length (m) of debug axes drawn at the palm link.")
    args = parser.parse_args()

    cid = p.connect(p.GUI if args.gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")

    # Hardcoded URDF path.
    urdf_path = "/home/chizhang/projects/dextrah/tg2_dexman_isaac/dextrah_lab/assets/kuka_inspirehand/urdf/kuka_inspirehand_test.urdf"
    # urdf_path = "/home/chizhang/projects/FABRICS/src/fabrics_sim/models/robots/urdf/kuka_allegro/kuka_allegro.urdf"

    print(f"Loading URDF: {urdf_path}")
    robot_id = p.loadURDF(urdf_path, useFixedBase=True)

    print(f"Loaded robot id={robot_id}, plane id={plane_id}")

    # Build a joint name->index map for convenience.
    num_joints = p.getNumJoints(robot_id)
    name_to_index = {}
    for j in range(num_joints):
        ji = p.getJointInfo(robot_id, j)
        name_to_index[ji[1].decode("utf-8")] = j

    # Mimic finger joints using gear constraints (PyBullet ignores URDF <mimic> tags).
    mimic_relations = {
        "index_joint_1": ("index_joint_0", -1.54545),
        "middle_joint_1": ("middle_joint_0", -1.54545),
        "ring_joint_1": ("ring_joint_0", -1.54545),
        "little_joint_1": ("little_joint_0", -1.54545),
        "thumb_joint_2": ("thumb_joint_1", -1.0),
        "thumb_joint_3": ("thumb_joint_2", -2.4),
    }
    for follower, (driver, ratio) in mimic_relations.items():
        if follower not in name_to_index or driver not in name_to_index:
            print(f"Skipping mimic: missing joint(s) {driver}->{follower}")
            continue
        constraint_id = p.createConstraint(
            robot_id,
            name_to_index[driver],
            robot_id,
            name_to_index[follower],
            jointType=p.JOINT_GEAR,
            jointAxis=[0, 0, -1],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
        )
        # If motion is reversed, flip the sign of ratio.
        p.changeConstraint(constraint_id, gearRatio=ratio, maxForce=500.0)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=name_to_index[follower],
            controlMode=p.VELOCITY_CONTROL,
            force=0.0,
        )

    # Locate palm link index
    palm_link_idx = None
    for j in range(num_joints):
        ji = p.getJointInfo(robot_id, j)
        link_name = ji[12].decode("utf-8")
        if link_name == "palm":
            palm_link_idx = j
            break
    if palm_link_idx is None:
        raise RuntimeError("Could not find palm link in URDF.")
    print(f"Palm link index: {palm_link_idx}")

    def draw_palm_axes():
        """Draw debug axes at the palm link and print the +Z direction in world frame."""
        state = p.getLinkState(robot_id, palm_link_idx, computeForwardKinematics=True)
        pos = state[4]  # world position of link frame
        orn = state[5]  # world orientation of link frame
        rot = p.getMatrixFromQuaternion(orn)
        # Rotation matrix rows
        x_axis = rot[0], rot[1], rot[2]
        y_axis = rot[3], rot[4], rot[5]
        z_axis = rot[6], rot[7], rot[8]
        # print(f"Palm +Z in world: {z_axis}")
        # Clear previous axes
        for item in draw_palm_axes.debug_items:
            p.removeUserDebugItem(item)
        draw_palm_axes.debug_items = [
            p.addUserDebugLine(pos, [pos[i] + args.axis_len * x_axis[i] for i in range(3)], [1, 0, 0], 2),
            p.addUserDebugLine(pos, [pos[i] + args.axis_len * y_axis[i] for i in range(3)], [0, 1, 0], 2),
            p.addUserDebugLine(pos, [pos[i] + args.axis_len * z_axis[i] for i in range(3)], [0, 0, 1], 2),
        ]

    draw_palm_axes.debug_items = []
    draw_palm_axes()

    # Build GUI sliders for joint control when in GUI mode.
    slider_ids = []
    mimic_joints = set(mimic_relations.keys())
    if args.gui:
        for j in range(num_joints):
            ji = p.getJointInfo(robot_id, j)
            name = ji[1].decode("utf-8")
            if name in mimic_joints:
                continue
            lower, upper = ji[8], ji[9]
            # Default a reasonable mid-position; fall back to 0 if limits are invalid.
            default = 0.0
            if lower < upper:
                default = 0.5 * (lower + upper)
            slider_ids.append(
                (j, p.addUserDebugParameter(f"{j}: {name}", lower, upper if upper > lower else 1.0, default))
            )

    if args.realtime and args.gui:
        p.setRealTimeSimulation(1)

    # Step and keep the sim alive.
    if args.run_seconds < 0:
        # Run indefinitely until interrupted
        while True:
            if args.gui:
                # Update joints from sliders
                for j, sid in slider_ids:
                    target = p.readUserDebugParameter(sid)
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=j,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target,
                        force=500.0,
                    )
                draw_palm_axes()
            p.stepSimulation()
            if args.gui:
                time.sleep(1.0 / 240.0)
    else:
        steps = int(args.run_seconds * 240)
        for _ in range(steps):
            if args.gui:
                for j, sid in slider_ids:
                    target = p.readUserDebugParameter(sid)
                    p.setJointMotorControl2(
                        bodyUniqueId=robot_id,
                        jointIndex=j,
                        controlMode=p.POSITION_CONTROL,
                        targetPosition=target,
                        force=500.0,
                    )
                draw_palm_axes()
            p.stepSimulation()
            if args.gui:
                time.sleep(1.0 / 240.0)

    p.disconnect(cid)


if __name__ == "__main__":
    main()
