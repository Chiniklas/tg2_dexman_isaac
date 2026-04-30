from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("control_domain", default_value="right_full"),
        DeclareLaunchArgument("arm_side", default_value=""),
        DeclareLaunchArgument("command_topic", default_value="/arm/cmd_pos"),
        DeclareLaunchArgument("status_topic", default_value="/arm/status"),
        DeclareLaunchArgument("head_status_topic", default_value="/head/status"),
        DeclareLaunchArgument("leg_status_topic", default_value="/leg/status"),
        DeclareLaunchArgument("waist_status_topic", default_value="/waist/status"),
        DeclareLaunchArgument("use_status", default_value="true"),
        DeclareLaunchArgument("publish_joint_states", default_value="true"),
        DeclareLaunchArgument("joint_state_topic", default_value="/joint_states"),
        DeclareLaunchArgument("joint_state_frame_id", default_value=""),
        DeclareLaunchArgument("input_joint_state_topic", default_value="/arm/command_joint_states"),
        DeclareLaunchArgument(
            "input_hand_joint_state_topic",
            default_value="/dummy_control/right_hand_joint_states",
        ),
        DeclareLaunchArgument(
            "hand_command_topic",
            default_value="/inspire_hand/ctrl/right_hand",
        ),
        DeclareLaunchArgument(
            "hand_command_interface",
            default_value="service",
        ),
        DeclareLaunchArgument(
            "hand_position_scale",
            default_value="1.0",
        ),
        DeclareLaunchArgument(
            "hand_state_topic",
            default_value="/inspire_hand/state/right_hand",
        ),
        DeclareLaunchArgument(
            "hand_service_name",
            default_value="/inspire_hand/set_angle_flexible/right_hand",
        ),
        DeclareLaunchArgument("hand_service_wait_sec", default_value="0.3"),
        DeclareLaunchArgument("velocity_rpm", default_value=""),
        DeclareLaunchArgument("min_velocity_rpm", default_value="0.1"),
        DeclareLaunchArgument("current_limit", default_value="5.0"),
        DeclareLaunchArgument("default_velocity_rpm", default_value="5.0"),
    ]

    node = Node(
        package="feedback_control_bridge",
        executable="feedback_control_bridge",
        name="feedback_control_bridge",
        output="screen",
        parameters=[
            {
                "control_domain": LaunchConfiguration("control_domain"),
                "arm_side": LaunchConfiguration("arm_side"),
                "command_topic": LaunchConfiguration("command_topic"),
                "status_topic": LaunchConfiguration("status_topic"),
                "head_status_topic": LaunchConfiguration("head_status_topic"),
                "leg_status_topic": LaunchConfiguration("leg_status_topic"),
                "waist_status_topic": LaunchConfiguration("waist_status_topic"),
                "use_status": LaunchConfiguration("use_status"),
                "publish_joint_states": LaunchConfiguration("publish_joint_states"),
                "joint_state_topic": LaunchConfiguration("joint_state_topic"),
                "joint_state_frame_id": LaunchConfiguration("joint_state_frame_id"),
                "input_joint_state_topic": LaunchConfiguration("input_joint_state_topic"),
                "input_hand_joint_state_topic": LaunchConfiguration("input_hand_joint_state_topic"),
                "hand_command_topic": LaunchConfiguration("hand_command_topic"),
                "hand_command_interface": LaunchConfiguration("hand_command_interface"),
                "hand_position_scale": LaunchConfiguration("hand_position_scale"),
                "hand_state_topic": LaunchConfiguration("hand_state_topic"),
                "hand_service_name": LaunchConfiguration("hand_service_name"),
                "hand_service_wait_sec": LaunchConfiguration("hand_service_wait_sec"),
                "velocity_rpm": LaunchConfiguration("velocity_rpm"),
                "min_velocity_rpm": LaunchConfiguration("min_velocity_rpm"),
                "current_limit": LaunchConfiguration("current_limit"),
                "default_velocity_rpm": LaunchConfiguration("default_velocity_rpm"),
            }
        ],
    )

    return LaunchDescription(arguments + [node])
