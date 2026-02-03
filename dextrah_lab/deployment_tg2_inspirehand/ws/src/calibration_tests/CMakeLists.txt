cmake_minimum_required(VERSION 3.0.2)
project(calibration_tests)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  tf2_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  tests/test_april_tag_detection.py
  tests/test_execute_targets.py
  tests/test_stereo_camera_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
