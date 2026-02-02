cmake_minimum_required(VERSION 3.0.2)
project(feedback_control_bridge)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  bodyctrl_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/feedback_control_bridge.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
