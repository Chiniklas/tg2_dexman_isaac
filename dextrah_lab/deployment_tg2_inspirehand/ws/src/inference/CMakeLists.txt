cmake_minimum_required(VERSION 3.0.2)
project(inference)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  policy_inference_stereo.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
