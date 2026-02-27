cmake_minimum_required(VERSION 3.0.2)
project(inference_offline)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  policy_inference_offline.py
  tests/student_traj_recorder.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
