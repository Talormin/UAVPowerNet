import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Dropout, BatchNormalization, Conv1D, Flatten
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据集构建和处理
print("开始构建数据集")
data_path = r'C:\\Users\\LWJ\\Desktop\\Dataset'
df = pd.read_csv(data_path + '\\XiAn.csv')
target_index = 19

# 滑动窗口的设置
M = 40  # 输入窗口长度
N = 15   # 预测窗口长度

x_data = []
y_data = []

# 按滑动窗口生成数据
for i in range(len(df) - M - N + 1):
    # 输入窗口 M
    input_window = df.iloc[i:i + M].values
    # 预测窗口 N
    output_window = df.iloc[i + M:i + M + N, target_index].values

    x_data.append(input_window)
    y_data.append(output_window)

x_data = np.array(x_data)
y_data = np.array(y_data)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=20, shuffle=True
)

# 全局归一化
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x_train_all = np.concatenate(x_train, axis=0)
y_train_all = np.concatenate(y_train, axis=0)

scaler_x.fit(x_train_all)
scaler_y.fit(y_train_all.reshape(-1, 1))

# 对训练集进行归一化
for i in range(len(x_train)):
    x_train[i] = scaler_x.transform(x_train[i])

for i in range(len(y_train)):
    y_train[i] = scaler_y.transform(y_train[i].reshape(-1, 1)).flatten()

# 对测试集进行归一化
for i in range(len(x_test)):
    x_test[i] = scaler_x.transform(x_test[i])

for i in range(len(y_test)):
    y_test[i] = scaler_y.transform(y_test[i].reshape(-1, 1)).flatten()

# 输出训练集总量
total_train_samples = len(x_train)
print(f"训练集总量: {total_train_samples}")

# 输出测试集总量
total_test_samples = len(x_test)
print(f"测试集总量: {total_test_samples}")

# **CNN + LSTM 模型**
def model_cnn_lstm(input_len_dynamic, timestep_dynamic, output_len, dropout=0.2, cnn_filters=64, lstm_units=64):
    input_dynamic = Input(shape=(timestep_dynamic, input_len_dynamic))

    # CNN 层提取局部特征
    x = Conv1D(filters=cnn_filters, kernel_size=3, activation="relu", padding="same")(input_dynamic)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    # LSTM 层学习时间序列依赖关系
    x = LSTM(lstm_units, return_sequences=False)(x)

    # 全连接层
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)

    outputs = Dense(output_len)(x)

    model = Model(inputs=input_dynamic, outputs=outputs)
    return model

# 构建模型
model = model_cnn_lstm(input_len_dynamic=x_train.shape[2], timestep_dynamic=x_train.shape[1], output_len=y_train.shape[1])

# 编译模型
model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))
history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test), verbose=1)

# 预测测试集
y_pred = model.predict(x_test)
y_pred_inverse = scaler_y.inverse_transform(y_pred)  # 反归一化预测结果
y_test_inverse = scaler_y.inverse_transform(y_test)  # 反归一化真实值

# 预测训练集
y_pred_train = model.predict(x_train)
y_pred_train_inverse = scaler_y.inverse_transform(y_pred_train)  # 反归一化预测结果
y_train_inverse = scaler_y.inverse_transform(y_train)  # 反归一化真实值

# 评估函数
def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    """ 评估模型训练效果 """
    Result = []

    # Test Set Evaluation
    y_p = y_p_test.flatten()
    y_t = y_t_test.flatten()

    non_zero_mask = np.abs(y_t) > 1e-6
    y_p = y_p[non_zero_mask]
    y_t = y_t[non_zero_mask]

    SSR = np.sum((y_p - y_t) ** 2)
    SST = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
    mape = np.mean(np.abs((y_p - y_t) / y_t)) * 100
    Result.extend([mae, rmse, r2, mape])

    # Train Set Evaluation
    y_p = y_p_train.flatten()
    y_t = y_t_train.flatten()

    non_zero_mask = np.abs(y_t) > 1e-6
    y_p = y_p[non_zero_mask]
    y_t = y_t[non_zero_mask]

    SSR = np.sum((y_p - y_t) ** 2)
    SST = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
    mape = np.mean(np.abs((y_p - y_t) / y_t)) * 100
    Result.extend([mae, rmse, r2, mape])

    return np.array(Result).reshape(1, -1)

# 评估模型
evaluation_results = Evaluate(
    y_p_test=y_pred_inverse,  # 测试集预测结果
    y_t_test=y_test_inverse,  # 测试集真实值
    y_p_train=y_pred_train_inverse,  # 训练集预测结果
    y_t_train=y_train_inverse  # 训练集真实值
)

# 输出评估结果
print("\n模型评估结果：")
print("Test MAE:", evaluation_results[0][0])
print("Test RMSE:", evaluation_results[0][1])
print("Test R2:", evaluation_results[0][2])
print("Test MAPE:", evaluation_results[0][3])
print("Train MAE:", evaluation_results[0][4])
print("Train RMSE:", evaluation_results[0][5])
print("Train R2:", evaluation_results[0][6])
print("Train MAPE:", evaluation_results[0][7])
