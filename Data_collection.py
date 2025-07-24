from scipy.interpolate import interp1d
import pandas as pd
import csv

df_vehicle_angular_velocity_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_vehicle_angular_velocity_0.csv")##时间戳、角速度
df_airspeed_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_airspeed_0.csv")##空速、温度记录
df_airspeed_validated_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_airspeed_validated_0.csv")##地速
df_differential_pressure_0 = pd.read_csv("C:\\Users\\LWJ\Desktop\\fly11.27\\fly11.27_differential_pressure_0.csv")## 压差？
df_estimator_local_position_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_estimator_local_position_0.csv")## 当地位置、速度数据（线位移） 1499
df_estimator_wind_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_estimator_wind_0.csv")## 风速
df_fw_virtual_attitude_setpoint_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_vehicle_attitude_setpoint_0.csv")## 欧拉角度
df_tecs_status_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_tecs_status_0.csv")## 总能耗率
df_vehicle_acceleration_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_vehicle_acceleration_0.csv")## 三轴加速度（线加速度)
df_vehicle_air_data_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_vehicle_air_data_0.csv")##盐度计、大气压强
df_vehicle_angular_acceleration_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_vehicle_acceleration_0.csv")##角加速度
df_battery_status_0 = pd.read_csv("C:\\Users\\LWJ\\Desktop\\fly11.27\\fly11.27_battery_status_0.csv")##电池的电流、电压

timestamp = df_vehicle_angular_velocity_0["timestamp"]##选取所有数据集中最长的timestamp作为基础(因为范围是差不多的，只是采样频率有差异)
timestamp.iloc[-1] = timestamp.iloc[-1]
##以下三个为角速度数据
xyz_0 = df_vehicle_angular_velocity_0["xyz[0]"]
xyz_1 = df_vehicle_angular_velocity_0["xyz[1]"]
xyz_2 = df_vehicle_angular_velocity_0["xyz[2]"]
##空速、温度记录
timestamp_airspeed_0 = df_airspeed_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_airspeed_0[0] = timestamp[0]
timestamp_airspeed_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
true_airspeed_m_s = df_airspeed_0["true_airspeed_m_s"]##记录原始数据
air_temperature_celsius = df_airspeed_0["air_temperature_celsius"]
f_true_airspeed_m_s = interp1d(timestamp_airspeed_0,true_airspeed_m_s,kind='linear')
f_air_temperature_celsius = interp1d(timestamp_airspeed_0,air_temperature_celsius,kind='linear')##插值函数
Final_true_airspeed_m_s = f_true_airspeed_m_s(timestamp)
Final_air_temperature_celsius = f_air_temperature_celsius(timestamp)##插值完成
##地速
timestamp_airspeed_validated_0 = df_airspeed_validated_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_airspeed_validated_0[0] = timestamp[0]
timestamp_airspeed_validated_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
true_ground_minus_wind_m_s = df_airspeed_validated_0["true_ground_minus_wind_m_s"]##记录原始数据
f_true_ground_minus_wind_m_s = interp1d(timestamp_airspeed_validated_0,true_ground_minus_wind_m_s,kind='linear')##插值函数
Final_true_ground_minus_wind_m_s = f_true_ground_minus_wind_m_s(timestamp)##插值完成
##压差
timestamp_differential_pressure_0 = df_differential_pressure_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_differential_pressure_0[0] = timestamp[0]
timestamp_differential_pressure_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
differential_pressure_pa = df_differential_pressure_0["differential_pressure_pa"]##记录原始数据
f_differential_pressure_pa = interp1d(timestamp_differential_pressure_0,differential_pressure_pa,kind='linear')##插值函数
Final_differential_pressure_pa = f_differential_pressure_pa(timestamp)##插值完成
##当地位置、速度数据（线位移）
timestamp_estimator_local_position_0 = df_estimator_local_position_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_estimator_local_position_0[0] = timestamp[0]
timestamp_estimator_local_position_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
x = df_estimator_local_position_0["x"]##记录原始数据
y = df_estimator_local_position_0["y"]
z = df_estimator_local_position_0["z"]
vx = df_estimator_local_position_0["vx"]
vy = df_estimator_local_position_0["vy"]
vz = df_estimator_local_position_0["vz"]
f_x = interp1d(timestamp_estimator_local_position_0,x,kind='linear')##插值函数
f_y = interp1d(timestamp_estimator_local_position_0,y,kind='linear')
f_z = interp1d(timestamp_estimator_local_position_0,z,kind='linear')
f_vx = interp1d(timestamp_estimator_local_position_0,vx,kind='linear')
f_vy = interp1d(timestamp_estimator_local_position_0,vy,kind='linear')
f_vz = interp1d(timestamp_estimator_local_position_0,vz,kind='linear')
Final_x = f_x(timestamp)##插值完成
Final_y = f_y(timestamp)
Final_z = f_z(timestamp)
Final_vx = f_vx(timestamp)
Final_vy = f_vy(timestamp)
Final_vz = f_vz(timestamp)
##风速
timestamp_estimator_wind_0 = df_estimator_wind_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_estimator_wind_0[0] = timestamp[0]
timestamp_estimator_wind_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
windspeed_north = df_estimator_wind_0["windspeed_north"]##记录原始数据
windspeed_east = df_estimator_wind_0["windspeed_east"]
f_windspeed_north = interp1d(timestamp_estimator_wind_0,windspeed_north,kind='linear')##插值函数
f_windspeed_east = interp1d(timestamp_estimator_wind_0,windspeed_east,kind='linear')
Final_windspeed_north = f_windspeed_north(timestamp)##插值完成
Final_windspeed_east = f_windspeed_east(timestamp)
##欧拉角度
timestamp_fw_virtual_attitude_setpoint_0 = df_fw_virtual_attitude_setpoint_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_fw_virtual_attitude_setpoint_0[0] = timestamp[0]
timestamp_fw_virtual_attitude_setpoint_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
roll_body = df_fw_virtual_attitude_setpoint_0["roll_body"]##记录原始数据
pitch_body = df_fw_virtual_attitude_setpoint_0["pitch_body"]
yaw_body = df_fw_virtual_attitude_setpoint_0["yaw_body"]
f_roll_body = interp1d(timestamp_fw_virtual_attitude_setpoint_0,roll_body,kind='linear')##插值函数
f_pitch_body = interp1d(timestamp_fw_virtual_attitude_setpoint_0,pitch_body,kind='linear')
f_yaw_body = interp1d(timestamp_fw_virtual_attitude_setpoint_0,yaw_body,kind='linear')
Final_roll_body = f_roll_body(timestamp)##插值完成
Final_pitch_body = f_pitch_body(timestamp)
Final_yaw_body = f_yaw_body(timestamp)
##总能耗率
timestamp_tecs_status_0 = df_tecs_status_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_tecs_status_0[0] = timestamp[0]
timestamp_tecs_status_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
total_energy_rate = df_tecs_status_0["total_energy_rate"]##记录原始数据
f_total_energy_rate = interp1d(timestamp_tecs_status_0,total_energy_rate,kind='linear')##插值函数
Final_total_energy_rate = f_total_energy_rate(timestamp)*100##插值完成
##三轴加速度
timestamp_vehicle_acceleration_0 = df_vehicle_acceleration_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_vehicle_acceleration_0[0] = timestamp[0]
timestamp_vehicle_acceleration_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
ax = df_vehicle_acceleration_0["xyz[0]"]##记录原始数据
ay = df_vehicle_acceleration_0["xyz[1]"]
az = df_vehicle_acceleration_0["xyz[2]"]
f_ax = interp1d(timestamp_vehicle_acceleration_0,ax,kind='linear')##插值函数
f_ay = interp1d(timestamp_vehicle_acceleration_0,ay,kind='linear')
f_az = interp1d(timestamp_vehicle_acceleration_0,az,kind='linear')
Final_ax = f_ax(timestamp)##插值完成
Final_ay = f_ay(timestamp)
Final_az = f_az(timestamp)
##盐度计、大气压强
timestamp_vehicle_air_data_0 = df_vehicle_air_data_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_vehicle_air_data_0[0] = timestamp[0]
timestamp_vehicle_air_data_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
baro_alt_meter = df_vehicle_air_data_0["baro_alt_meter"]##记录原始数据
baro_pressure_pa = df_vehicle_air_data_0["baro_pressure_pa"]
f_baro_alt_meter = interp1d(timestamp_vehicle_air_data_0,baro_alt_meter,kind='linear')##插值函数
f_baro_pressure_pa = interp1d(timestamp_vehicle_air_data_0,baro_pressure_pa,kind='linear')
Final_baro_alt_mete = f_baro_alt_meter(timestamp)##插值完成
Final_baro_pressure_pa = f_baro_pressure_pa(timestamp)
##角加速度
timestamp_vehicle_angular_acceleration_0 = df_vehicle_angular_acceleration_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_vehicle_angular_acceleration_0[0] = timestamp[0]
timestamp_vehicle_angular_acceleration_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
q1 = df_vehicle_angular_acceleration_0["xyz[0]"]##记录原始数据
q2 = df_vehicle_angular_acceleration_0["xyz[1]"]
q3 = df_vehicle_angular_acceleration_0["xyz[2]"]
f_q1 = interp1d(timestamp_vehicle_angular_acceleration_0,q1,kind='linear')##插值函数
f_q2 = interp1d(timestamp_vehicle_angular_acceleration_0,q2,kind='linear')
f_q3 = interp1d(timestamp_vehicle_angular_acceleration_0,q3,kind='linear')
Final_q1 = f_q1(timestamp)##插值完成
Final_q2 = f_q2(timestamp)
Final_q3 = f_q3(timestamp)

##电池的电流、电压（相乘得到功率）
timestamp_battery_status_0 = df_battery_status_0["timestamp"]##记录该数据集的timestamp（用于插值）
timestamp_battery_status_0[0] = timestamp[0]
timestamp_battery_status_0.iloc[-1] = timestamp.iloc[-1]##调整timestamp的范围，防止超出插值范围
voltage_filtered_v = df_battery_status_0["voltage_filtered_v"]##记录原始数据
current_filtered_a = df_battery_status_0["current_filtered_a"]
f_voltage_filtered_v = interp1d(timestamp_battery_status_0,voltage_filtered_v,kind='linear')##插值函数
f_current_filtered_a = interp1d(timestamp_battery_status_0,current_filtered_a,kind='linear')
Final_voltage_filtered_v = f_voltage_filtered_v(timestamp)##插值完成
Final_current_filtered_a = f_current_filtered_a(timestamp)
power = Final_voltage_filtered_v*Final_current_filtered_a


##数据存储
dataframe = pd.DataFrame({'timestamp':timestamp,'xyz_0':xyz_0,'xyz_1':xyz_1,'xyz_2':xyz_2,
'Final_true_airspeed_m_s':Final_true_airspeed_m_s,'Final_air_temperature_celsius':Final_air_temperature_celsius,
'Final_true_ground_minus_wind_m_s':Final_true_ground_minus_wind_m_s,
'Final_differential_pressure_pa':Final_differential_pressure_pa,
'Final_x':Final_x,'Final_y':Final_y,'Final_z':Final_z,'Final_vx':Final_vx,'Final_vy':Final_vy,'Final_vz':Final_vz,
'Final_windspeed_north':Final_windspeed_north,'Final_windspeed_east':Final_windspeed_east,
'Final_roll_body':Final_roll_body,'Final_pitch_body':Final_pitch_body,'Final_yaw_body':Final_yaw_body,
'Final_total_energy_rate':Final_total_energy_rate,
'Final_ax':Final_ax,'Final_ay':Final_ay,'Final_az':Final_az,
'Final_baro_alt_mete':Final_baro_alt_mete,'Final_baro_pressure_pa':Final_baro_pressure_pa,
'Final_q1':Final_q1,'Final_q2':Final_q2,'Final_q3':Final_q3,
'power':power
})
dataframe = dataframe.dropna(axis=1)
dataframe.to_csv(r"C:\\Users\\LWJ\\Desktop\\27.csv",sep=',')