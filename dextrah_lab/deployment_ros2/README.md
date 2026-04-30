# deployment_ros2

ROS 2 workspace for Tiangong deployment packages in this repository.

## Layout

- `src/bodyctrl_msgs`: custom messages and services
- `src/feedback_control_bridge`: ROS 2 bridge node

## Usage

From this directory:

```bash
colcon build
source install/setup.bash
```
