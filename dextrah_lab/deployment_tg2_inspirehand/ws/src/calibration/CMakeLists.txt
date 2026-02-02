cmake_minimum_required(VERSION 3.0.2)
project(calibration)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  tf2_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  camera_calibration.py
  tg2_inspirehand_random_targets.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
