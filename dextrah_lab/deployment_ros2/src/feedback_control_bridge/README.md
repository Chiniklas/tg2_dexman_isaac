# feedback_control_bridge

ROS 2 port of the ROS 1 direct feedback control bridge.

It does three things:

- subscribes to `sensor_msgs/msg/JointState` arm commands
- converts them to `bodyctrl_msgs/CmdSetMotorPosition`
- subscribes to arm feedback from `/arm/status` as `bodyctrl_msgs/msg/MotorStatusMsg`
- subscribes to hand feedback from `/inspire_hand/state/right_hand` as `sensor_msgs/msg/JointState`
- optionally mirrors that feedback to `/joint_states`

The bridge control domain can be selected as:

- `right_arm`
- `left_arm`
- `right_full`
- `left_full`
- `upper_body`
- `full_body`

`upper_body` and `full_body` are currently placeholders. They are accepted by the
bridge, but today they still behave like `right_full` because head and leg
bridging are not implemented yet.

For compatibility, the older `arm_side` values `right`, `left`, and
`right_arm_hand` are still accepted as aliases.

If the selected control domain enables a hand, the bridge can also translate
hand joint targets into the `bodyctrl_msgs` hand service.

This workspace carries the `bodyctrl_msgs` interface package under
`src/bodyctrl_msgs`, sourced from the current robot firmware workspace.

Default runtime topics:

- arm command output: `/arm/cmd_pos`
- arm status input: `/arm/status`
- bridge input command: `/arm/command_joint_states`
- hand command input: `/inspire_hand/ctrl/right_hand`
- hand status input: `/inspire_hand/state/right_hand`
- hand service output: `/inspire_hand/set_angle_flexible/right_hand`

For `left_full`, the bridge automatically switches the default hand interfaces to
the left-hand names when you leave the launch defaults unchanged:

- hand command input: `/inspire_hand/ctrl/left_hand`
- hand status input: `/inspire_hand/state/left_hand`
- hand service output: `/inspire_hand/set_angle_flexible/left_hand`
