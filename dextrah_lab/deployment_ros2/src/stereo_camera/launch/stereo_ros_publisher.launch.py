from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    arguments = [
        DeclareLaunchArgument("left_config", default_value="ov9732_L"),
        DeclareLaunchArgument("right_config", default_value="ov9732_R"),
        DeclareLaunchArgument("device_left", default_value=""),
        DeclareLaunchArgument("device_right", default_value=""),
        DeclareLaunchArgument("left_topic", default_value="/stereo/left/image_raw"),
        DeclareLaunchArgument("right_topic", default_value="/stereo/right/image_raw"),
        DeclareLaunchArgument("left_frame", default_value="stereo_left"),
        DeclareLaunchArgument("right_frame", default_value="stereo_right"),
        DeclareLaunchArgument("flip", default_value="both"),
        DeclareLaunchArgument("max_fails", default_value="5"),
        DeclareLaunchArgument("reconnect_wait", default_value="1.0"),
        DeclareLaunchArgument("rate", default_value="0.0"),
        DeclareLaunchArgument("width", default_value="320"),
        DeclareLaunchArgument("height", default_value="240"),
        DeclareLaunchArgument("fps", default_value="0"),
    ]

    node = Node(
        package="stereo_camera",
        executable="stereo_ros_publisher",
        name="stereo_ros_publisher",
        output="screen",
        parameters=[
            {
                "left_config": LaunchConfiguration("left_config"),
                "right_config": LaunchConfiguration("right_config"),
                "device_left": LaunchConfiguration("device_left"),
                "device_right": LaunchConfiguration("device_right"),
                "left_topic": LaunchConfiguration("left_topic"),
                "right_topic": LaunchConfiguration("right_topic"),
                "left_frame": LaunchConfiguration("left_frame"),
                "right_frame": LaunchConfiguration("right_frame"),
                "flip": LaunchConfiguration("flip"),
                "max_fails": LaunchConfiguration("max_fails"),
                "reconnect_wait": LaunchConfiguration("reconnect_wait"),
                "rate": LaunchConfiguration("rate"),
                "width": LaunchConfiguration("width"),
                "height": LaunchConfiguration("height"),
                "fps": LaunchConfiguration("fps"),
            }
        ],
    )

    return LaunchDescription(arguments + [node])
