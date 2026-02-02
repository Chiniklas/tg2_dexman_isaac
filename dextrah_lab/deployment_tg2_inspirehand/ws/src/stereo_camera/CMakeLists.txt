cmake_minimum_required(VERSION 3.0.2)
project(stereo_camera)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/stereo_ros_publisher.py
  tests/camera_calibration.py
  tests/capture_single_stream.py
  tests/capture_stereo_stream.py
  tests/capture_two_stream.py
  tests/depth_plausibility.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY stereo_camera tests launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
