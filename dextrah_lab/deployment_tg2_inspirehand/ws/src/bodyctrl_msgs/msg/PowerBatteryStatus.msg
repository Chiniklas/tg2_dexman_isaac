# @brief 电源板电压、电流、电量、温度等信息

std_msgs/Header header

# battery status information
int32 battery_installed  # 0x00都不存在，0x01小电池存在，0x02大电池存在，0x03大小电池都存在
int32 battery_working    # 0x01大电池，0x10小电池

float32 master_battery_voltage # 电池电压信息
float32 master_battery_current # 电池电流信息
float32 master_battery_power   # 电池电量

float32 little_battery_voltage # 小电池电压信息
float32 little_battery_current # 小电池电流信息
float32 little_battery_power   # 小电池电量

int8 PG12A # 00低，01高
int8 PG12B # 00低，01高
int8 PG12C # 00低，01高
int8 PG12D # 00低，01高
int8 PG5CD # 00低，01高
int8 PG5AB # 00低，01高

int8 PGRDC2    # 00低，01高
int8 PGRDC1    # 00低，01高
int8 PGHREAD   # 00低，01高
int8 PGBUTTON2 # 00低，01高
