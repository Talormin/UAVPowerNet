import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras.models import  Model, load_model
from keras.layers import LSTM, GRU, Dense, Input, Reshape, Concatenate, Dropout, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention, BatchNormalization, Add, Activation, Conv1D, AveragePooling1D, Flatten
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError
from keras.initializers import TruncatedNormal
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pywt  # 小波变换
import optuna

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):##该函数用于将多个输入值转变为一组输出值，即多注意力机制
    # Normalization and Attention
    x = LayerNormalization(epsilon=1e-6)(inputs)##标准化层 
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x, x)##多头注意力机制
    res = x + inputs##残差操作，防止退化

    # Feed Forward Part（全连接层）
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)##一维卷积
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)##一维卷积
    return x + res##残差操作，防止退化

def model_dynamic(##搭建模型
    input_len_dynamic,
    timestep_dynamic,
    output_len,
    head_size=16,   
    num_heads=2,
    num_transformer_blocks=4,
    ff_dim=16,
    dropout = 0.1,
    lstm_units = 64
):
    input_dynamic = Input(shape=(timestep_dynamic,input_len_dynamic))##初始化深度学习网络输入层的tensor(张量)
    
    x = input_dynamic
    for _ in range(num_transformer_blocks):##定义循环次数（特征提取次数？）
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_dynamic = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    output_dynamic = Reshape((-1, 1))(output_dynamic)

    output_dynamic = LSTM(units=lstm_units, activation='tanh')(output_dynamic)

    hid_dynamic = Dense(units=64, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(output_dynamic)##全连接层 
    
    hid_dynamic = BatchNormalization()(hid_dynamic)##批标准化
    hid_dynamic = Dense(units=32, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(hid_dynamic)##全连接层 
    hid_dynamic = Dropout(dropout)(hid_dynamic)##Dropout可以通过在训练神经网络期间随机丢弃单元来防止过拟合
    outputs = Dense(units=output_len)(hid_dynamic)##全连接层 
    
    model = Model([input_dynamic], outputs)## inputs与outputs一定是Layer调用输出的张量
    return model

# 定义用于调参的目标函数
def objective(trial, X_train, X_val, y_train, y_val):
    # 从 Optuna 中采样超参数
    head_size = trial.suggest_int('head_size', 8, 64)
    num_heads = trial.suggest_int('num_heads', 1, 8)
    num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 1, 4)
    ff_dim = trial.suggest_int('ff_dim', 16, 128)
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    epoch =  trial.suggest_int('epoch', 10, 200)
    batch_size =  trial.suggest_int('batch_size', 512,1024)
    learn_rate = trial.suggest_float('learn_rate', 0.0001, 0.1)

    # 创建模型
    input_dynamic = Input(shape=(20, 19))  # 输入形状是20个时间步，每个时间步25个特征

    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # 池化
    output_dynamic = GlobalAveragePooling1D(data_format="channels_first")(x)
    
    # LSTM层
    output_dynamic = Reshape((-1, 1))(output_dynamic)
    output_dynamic = LSTM(units=lstm_units, activation='sigmoid')(output_dynamic)
    
    # 其他层
    hid_dynamic = BatchNormalization()(output_dynamic)
    hid_dynamic = Dense(units=32, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(hid_dynamic)
    hid_dynamic = Dropout(0.1)(hid_dynamic)
    outputs = Dense(units=20)(hid_dynamic)  # 输出功率预测

    model = Model([input_dynamic], outputs)
    model.compile(loss="mse", optimizer=Adam(learning_rate = learn_rate))##模型训练参数
    
    # 训练模型并返回验证损失（或使用交叉验证）
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size, validation_data=(X_val, y_val), verbose=0)
    loss = model.evaluate(X_val, y_val, verbose=0)
    
    return loss  # 返回验证损失作为优化目标

def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    
    Result = []
    
    # test
    y_p = y_p_test
    y_t = y_t_test
    
    plt.figure()
    plt.scatter(y_p, y_t, c='blue', s=5, alpha=0.8)
    x_equal = [min(y_t), max(y_t)]
    plt.plot(x_equal, x_equal, color='orange')
    plt.show()
    
    SSR = np.sum((y_p-y_t)**2)
    SST = np.sum((y_t-np.mean(y_t))**2)
    r2 = 1-SSR/SST
    mae = np.sum(np.absolute(y_p-y_t))/len(y_t)
    mae_mean = mae/np.absolute(np.mean(y_t))
    rmse = (np.sum((y_p-y_t)**2)/len(y_t))**0.5

    y_p = y_p[np.absolute(y_t)>0.0001]
    y_t = y_t[np.absolute(y_t)>0.0001]
    
    mape = np.sum(np.absolute((y_p-y_t)/y_t))/len(y_t)
    Result.append(mae)
    Result.append(rmse)
    Result.append(r2)
    Result.append(mape)
    Result.append(mae_mean)
    
    # train
    y_p = y_p_train
    y_t = y_t_train

    SSR = np.sum((y_p-y_t)**2)
    SST = np.sum((y_t-np.mean(y_t))**2)
    r2 = 1-SSR/SST
    mae = np.sum(np.absolute(y_p-y_t))/len(y_t)
    mae_mean = mae/np.absolute(np.mean(y_t))
    rmse = (np.sum((y_p-y_t)**2)/len(y_t))**0.5
    
    y_p = y_p[np.absolute(y_t)>0.0001]
    y_t = y_t[np.absolute(y_t)>0.0001]
    mape = np.sum(np.absolute((y_p-y_t)/y_t))/len(y_t)
    Result.append(mae)
    Result.append(rmse)
    Result.append(r2)
    Result.append(mape)
    Result.append(mae_mean)

    Result = np.array(Result).reshape(1,-1)
    return Result


print("开始构建数据集")
time_start = time.time()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
data_path = r'C:\Users\LWJ\Desktop'
model_path = r'C:\Users\LWJ\Desktop'

df = pd.read_csv(data_path + '\\27.csv')
feature = [ 'xyz_0', 'xyz_1', 'xyz_2', 'Final_true_airspeed_m_s',
            'Final_air_temperature_celsius', 'Final_true_ground_minus_wind_m_s', 'Final_differential_pressure_pa', 'Final_x', 'Final_y',
            'Final_z', 'Final_vx', 'Final_vy', 'Final_vz', 'Final_windspeed_north', 
            'Final_windspeed_east', 'Final_roll_body', 'Final_pitch_body', 'Final_yaw_body', 'Final_total_energy_rate', 
            'Final_ax', 'Final_ay','Final_az', 'Final_baro_alt_mete', 'Final_baro_pressure_pa', 'Final_q1', 
            'Final_q2','Final_q3','Final_voltage_filtered_v','Final_current_average_a']##定义29类数据(但是df中还记录了编号列，因此总共有30列)

seg_total = np.array(range(len(df['xyz_0'])))##构建训练集与测试集
seg_train, seg_test = train_test_split(seg_total, test_size=0.2, random_state = 20, shuffle = True)##构建训练集与测试集（这里是将序号进行分类）
x_train = list()##构建数组用于存储测试集与训练集的数据
x_test = list()
y_train = list()
y_test = list()

Scaler_x = MinMaxScaler(feature_range=(0,1))##Scaler_x记录了这种归一化的规范，用于统一归一化的方式和还原数据
Scaler_xt = MinMaxScaler(feature_range=(0,1))##Scaler_x记录了这种归一化的规范，用于统一归一化的方式和还原数据(测试集)
Scaler_y = MinMaxScaler(feature_range=(0,1))##归一化函数的初始化
Scaler_yt = MinMaxScaler(feature_range=(0,1))##归一化函数的初始化（测试集）

dd = df.values

Step = 20##步长
kk_test = 0
kk_train = 0
x_train1 = list()
x_test1 = list()
y_train1 = list()
y_test1 = list()
y_a = list()
y_s = list()
target_index = 19
for i in range(0,len(dd)):##数据集分类(并按步长分组)
    if i in seg_train:
        x_train1.append(dd[i])
        y_train1.append(dd[i][target_index])##y数据
        kk_train = kk_train+1
    if i in seg_test:
        x_test1.append(dd[i])
        y_test1.append(dd[i][target_index])
        kk_test = kk_test+1
    if kk_train > (Step-1):
        x_train1 = np.array(x_train1)
        y_train1 = np.array(y_train1)
        x_train1 = Scaler_x.fit_transform(x_train1)
        kk_train = 0
        x_train.append(x_train1)##按步长分组
        y_train.append(y_train1)
        x_train1 = []
        y_train1 = []
    if kk_test > (Step-1):
        x_test1 = np.array(x_test1)
        y_test1 = np.array(y_test1)
        x_test1 = Scaler_xt.fit_transform(x_test1)
        kk_test = 0
        x_test.append(x_test1)
        y_test.append(y_test1)
        x_test1 = []
        y_test1 = []
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = Scaler_y.fit_transform(y_train)
y_test = Scaler_yt.fit_transform(y_test)

trans_train = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]##预测19
x_test = x_test[:,:,trans_train]
x_train = x_train[:,:,trans_train]
time_end = time.time()
print("数据集构建完成,用时",time_end-time_start)

print('开始训练模型')
# Transformer-MLP
# positional embedding（位置编码，用于标记每个数据的位置信息）
timeStep = 20 ##一次输入20组数据
x_dynamic_position = np.arange(timeStep)/timeStep ##生成20组数据对应的编号
x_position = np.expand_dims(x_dynamic_position,0).repeat(len(trans_train),axis=0).T##维度增加
x_train_position = np.expand_dims(x_position,0).repeat(x_train.shape[0],axis=0)##一个步长内，每组数据的每一位都要加上对应的编号，例如第一组所有数据都加0.05，第二组都加0.1...
x_test_position = np.expand_dims(x_position,0).repeat(x_test.shape[0],axis=0)
x_train_dynamic = x_train + x_train_position
x_test_dynamic = x_test + x_test_position
for i in range(len(y_train)):
    for j in range(timeStep):
        y_train[i][j] =  y_train[i][j] + j/timeStep
for i in range(len(y_test)):
    for j in range(timeStep):
        y_test[i][j] =  y_test[i][j] + j/timeStep


# 使用 Optuna 进行超参数优化
study = optuna.create_study(direction='minimize')  # 目标是最小化损失
study.optimize(lambda trial: objective(trial, x_train_dynamic, x_test_dynamic, y_train, y_test), n_trials=100)  # 尝试 50 次不同的超参数组合

# 输出最佳超参数
best_params = study.best_trial.params
print("Best hyperparameters:", best_params)

# 使用最佳超参数训练最终模型
best_model = model_dynamic(
    input_len_dynamic=19,
    timestep_dynamic=20,
    output_len=20,
    head_size=best_params['head_size'],
    num_heads=best_params['num_heads'],
    num_transformer_blocks=best_params['num_transformer_blocks'],
    ff_dim=best_params['ff_dim'],
    dropout = best_params['dropout'],
    lstm_units = best_params['lstm_units']
)

best_model.compile(loss="mse", optimizer=Adam(learning_rate = best_params['learn_rate']))##模型训练参数
history = best_model.fit([x_train_dynamic], y_train, epochs=best_params['epoch'], batch_size=best_params['batch_size'], validation_data=([x_test_dynamic], y_test))

history_train = history.history##对学习效果进行分析
plt.figure()
plt.plot(history_train['loss'][1:], label='训练集')
plt.plot(history_train['val_loss'][1:], label='验证集')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()
plt.figure()
plt.plot(history_train['loss'][100:], label='训练集')
plt.plot(history_train['val_loss'][100:], label='验证集')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()

best_model.save(model_path + '\\transformer_' + '.h5')
# model = load_model(model_path + '\\transformer_' + str(Distance) + '_' + str(timeStep) + '_' + str(timeStep_range) + '.h5') #---------------------加载模型

time_end = time.time()
print('模型训练完成，用时(s)',time_end-time_start)

##测试集进行预测，并进行正确率验证
y_predict_test = best_model.predict([x_test_dynamic])
y_predict_test = np.array(y_predict_test)
y_test = np.array(y_test)
y_p_test = Scaler_yt.inverse_transform(y_predict_test)
y_t_test = Scaler_yt.inverse_transform(y_test)
y_predict_train = best_model.predict([x_train_dynamic])
y_p_train = Scaler_y.inverse_transform(y_predict_train)
y_t_train = Scaler_y.inverse_transform(y_train)

y_p_test = y_p_test.reshape(-1)
y_t_test = y_t_test.reshape(-1)
y_p_train = y_p_train.reshape(-1)
y_t_train = y_t_train.reshape(-1)

a_Result = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)##正确率检验

print('训练集样本数为%d' % y_train.shape[0])
print('测试集样本数为%d' % y_test.shape[0])
print('学习效果(')
print('指标:','平均绝对误差','均方根误差','决定系数','平均绝对百分比误差','平均绝对误差的平均值')
print('测试集：',a_Result[0][0],a_Result[0][1],a_Result[0][2],a_Result[0][3],a_Result[0][4])
print('训练集：',a_Result[0][5],a_Result[0][6],a_Result[0][7],a_Result[0][8],a_Result[0][9])

def visualize_results(history, y_true, y_pred):
    """
    可视化模型效果
    """
    residuals = y_true - y_pred  # 残差计算

    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 2. 真实值 vs 预测值
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100], label='True Values', color='blue', linestyle='-', linewidth=2)
    plt.plot(y_pred[:100], label='Predicted Values', color='orange', linestyle='--', linewidth=2)
    plt.title('True vs Predicted (Test Set)', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 3. 残差分布图
    plt.figure(figsize=(10, 6))
    plt.hist(residuals.flatten(), bins=30, color='purple', alpha=0.7)
    plt.title('Residual Distribution', fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

    # 4. 残差 vs 真实值
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.6, color='orange', label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2) 
    plt.title('Residuals vs True Values', fontsize=16)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

visualize_results(history, y_t_test.flatten(), y_p_test.flatten())