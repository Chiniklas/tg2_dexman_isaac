cmake_minimum_required(VERSION 3.0.2)
project(policy_inference_stereo_transformer)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/policy_inference_stereo_transformer_node.py
  scripts/policy_runtime.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
