cmake_minimum_required(VERSION 3.8)
project(bodyctrl_msgs)

find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

file(GLOB msg_files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "msg/*.msg")
list(SORT msg_files)

set(srv_files
  "srv/GetAngleAct.srv"
  "srv/GetError.srv"
  "srv/GetForceAct.srv"
  "srv/GetStatus.srv"
  "srv/JointSetZero.srv"
  "srv/MotorInit.srv"
  "srv/MotorResetPosition.srv"
  "srv/MotorStart.srv"
  "srv/MotorStop.srv"
  "srv/Movement.srv"
  "srv/PlanEefLine.srv"
  "srv/PlanEefRelLine.srv"
  "srv/PlanJointTraj.srv"
  "srv/ResetMotorZeroOffset.srv"
  "srv/SetAngle.srv"
  "srv/SetAngleFlexible.srv"
  "srv/SetClearError.srv"
  "srv/SetForce.srv"
  "srv/SetGestureForceCalibration.srv"
  "srv/SetMotorZeroOffset.srv"
  "srv/SetSpeed.srv"
  "srv/VersionUpgrade.srv"
  "srv/XSensImuInit.srv"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES
    builtin_interfaces
    geometry_msgs
    std_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
