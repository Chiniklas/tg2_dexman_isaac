from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("repo_root", default_value=""),
        DeclareLaunchArgument("cfg_path", default_value=""),
        DeclareLaunchArgument("checkpoint_path", default_value=""),
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("left_topic", default_value="/stereo/left/image_raw"),
        DeclareLaunchArgument("right_topic", default_value="/stereo/right/image_raw"),
        DeclareLaunchArgument("proprio_topic", default_value="/policy/proprio"),
        DeclareLaunchArgument("action_topic", default_value="/policy/action"),
        DeclareLaunchArgument("joint_command_topic", default_value="/arm/command_joint_states"),
        DeclareLaunchArgument("num_proprio_obs", default_value="159"),
        DeclareLaunchArgument("num_actions", default_value="11"),
        DeclareLaunchArgument("image_width", default_value="320"),
        DeclareLaunchArgument("image_height", default_value="240"),
        DeclareLaunchArgument("deterministic", default_value="true"),
        DeclareLaunchArgument("rate", default_value="20.0"),
        DeclareLaunchArgument("action_scale", default_value="1.0"),
        DeclareLaunchArgument("publish_joint_state", default_value="true"),
    ]

    node = Node(
        package="policy_inference_stereo_transformer",
        executable="policy_inference_stereo_transformer_node",
        name="policy_inference_stereo_transformer",
        output="screen",
        parameters=[
            {
                "repo_root": LaunchConfiguration("repo_root"),
                "cfg_path": LaunchConfiguration("cfg_path"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "device": LaunchConfiguration("device"),
                "left_topic": LaunchConfiguration("left_topic"),
                "right_topic": LaunchConfiguration("right_topic"),
                "proprio_topic": LaunchConfiguration("proprio_topic"),
                "action_topic": LaunchConfiguration("action_topic"),
                "joint_command_topic": LaunchConfiguration("joint_command_topic"),
                "num_proprio_obs": ParameterValue(LaunchConfiguration("num_proprio_obs"), value_type=int),
                "num_actions": ParameterValue(LaunchConfiguration("num_actions"), value_type=int),
                "image_width": ParameterValue(LaunchConfiguration("image_width"), value_type=int),
                "image_height": ParameterValue(LaunchConfiguration("image_height"), value_type=int),
                "deterministic": ParameterValue(LaunchConfiguration("deterministic"), value_type=bool),
                "rate": ParameterValue(LaunchConfiguration("rate"), value_type=float),
                "action_scale": ParameterValue(LaunchConfiguration("action_scale"), value_type=float),
                "publish_joint_state": ParameterValue(LaunchConfiguration("publish_joint_state"), value_type=bool),
                "joint_action_indices": [0, 1, 2, 3, 4, 5, 6],
                "joint_names": [
                    "shoulder_pitch_r_joint",
                    "shoulder_roll_r_joint",
                    "shoulder_yaw_r_joint",
                    "elbow_pitch_r_joint",
                    "elbow_yaw_r_joint",
                    "wrist_pitch_r_joint",
                    "wrist_roll_r_joint",
                ],
            }
        ],
    )

    return LaunchDescription(arguments + [node])
