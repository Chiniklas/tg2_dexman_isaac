cmake_minimum_required(VERSION 3.0.2)
project(bodyctrl_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
  geometry_msgs
)

add_message_files(
  FILES
    CmdMotorCtrl.msg
    CmdSetMotorCurTor.msg
    CmdSetMotorDistance.msg
    CmdSetMotorPosition.msg
    CmdSetMotorSpeed.msg
    CmdSetTsHandCtrl.msg
    CmdSetTsHandPosition.msg
    CmdSetWaistMotorPos.msg
    Euler.msg
    Exception.msg
    ExceptionArray.msg
    Imu.msg
    MotorCtrl.msg
    MotorName.msg
    MotorStatus.msg
    MotorStatus1.msg
    MotorStatusMsg.msg
    MotorStatusMsg1.msg
    MotorStatusMsg2.msg
    NodeState.msg
    PowerBatteryStatus.msg
    PowerBoardCtrl.msg
    PowerBoardKeyStatus.msg
    PowerLightCtrl.msg
    PowerStatus.msg
    SbusData.msg
    SetMotorCurTor.msg
    SetMotorDistance.msg
    SetMotorPosition.msg
    SetMotorSpeed.msg
    SetTsHandCtrl.msg
    SetTsHandCtrlItem.msg
    SetTsHandPosition.msg
    Sri.msg
    SriName.msg
    TsHandName.msg
    TsHandStatus.msg
    TsHandStatusMsg.msg
    WaistMotorStatus.msg
)

add_service_files(
  FILES
    get_angle_act.srv
    get_error.srv
    get_force_act.srv
    get_status.srv
    joint_set_zero.srv
    MotorInit.srv
    MotorStart.srv
    MotorStop.srv
    ResetMotorZeroOffset.srv
    set_angle.srv
    set_angle_flexible.srv
    set_clear_error.srv
    set_force.srv
    set_speed.srv
    SetMotorZeroOffset.srv
    XSensImuInit.srv
)

generate_messages(
  DEPENDENCIES
    std_msgs
    geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS message_runtime std_msgs geometry_msgs
)
