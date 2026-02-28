cmake_minimum_required(VERSION 3.0.2)
project(calibration)

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
  sensor_msgs
  std_msgs
  tf2_ros
  tf2_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  april_tag_detector.py
  camera_calibration.py
  tests/test_april_tag_detection.py
  tests/test_execute_targets.py
  tests/test_stereo_camera_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
