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
    num_transformer_blocks=2,
    ff_dim=16
):
    input_dynamic = Input(shape=(timestep_dynamic,input_len_dynamic))##初始化深度学习网络输入层的tensor(张量)
    
    x = input_dynamic
    for _ in range(num_transformer_blocks):##定义循环次数（特征提取次数？）
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_dynamic = GlobalAveragePooling1D(data_format="channels_first")(x)##池化
    
    hid_dynamic = BatchNormalization()(output_dynamic)##批标准化
    hid_dynamic = Dense(units=32, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(hid_dynamic)##全连接层 
    hid_dynamic = Dropout(0.1)(hid_dynamic)##Dropout可以通过在训练神经网络期间随机丢弃单元来防止过拟合
    outputs = Dense(units=output_len)(hid_dynamic)##全连接层 
    
    model = Model([input_dynamic], outputs)## inputs与outputs一定是Layer调用输出的张量
    return model

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
Scaler_y = MinMaxScaler(feature_range=(0,1))##归一化函数的初始化

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
for i in range(0,len(dd)):##数据集分类(并按步长分组)
    if i in seg_train:
        x_train1.append(dd[i])
        y_train1.append(dd[i][25])##LSTM的y数据
        kk_train = kk_train+1
    if i in seg_test:
        x_test1.append(dd[i])
        y_test1.append(dd[i][25])
        kk_test = kk_test+1
    if kk_train > 19:
        x_train1 = np.array(x_train1)
        y_train1 = np.array(y_train1)
        x_train1 = Scaler_x.fit_transform(x_train1)
        kk_train = 0
        x_train.append(x_train1)##按步长分组
        y_train.append(y_train1)
        x_train1 = []
        y_train1 = []
    if kk_test > 19:
        x_test1 = np.array(x_test1)
        y_test1 = np.array(y_test1)
        x_test1 = Scaler_x.fit_transform(x_test1)
        kk_test = 0
        x_test.append(x_test1)
        y_test.append(y_test1)
        x_test1 = []
        y_test1 = []
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)
y_train2 = []
y_test2 = []
for i in range(0,len(y_train)):
    y_train2.append(np.mean(y_train[i]))
for i in range(0,len(y_test)):
    y_test2.append(np.mean(y_test[i]))
y_train2 = np.array(y_train2)
y_test2 = np.array(y_test2)
y_train2 = Scaler_y.fit_transform(y_train2.reshape(-1,1))
y_test2 = Scaler_y.fit_transform(y_test2.reshape(-1,1))
trans_train = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]##预测25）
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

model = model_dynamic(
    input_len_dynamic=len(trans_train),
    timestep_dynamic=timeStep,
    output_len=1,
    )

model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=[RootMeanSquaredError()])##模型训练参数
# callbacks = [EarlyStopping(patience=30, restore_best_weights=True)]
history = model.fit(##模型训练
    [x_train_dynamic], y_train2,
    batch_size=4096, epochs=100,
    validation_data=([x_test], y_test2))
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

model.save(model_path + '\\transformer_' + '.h5')
# model = load_model(model_path + '\\transformer_' + str(Distance) + '_' + str(timeStep) + '_' + str(timeStep_range) + '.h5') #---------------------加载模型

time_end = time.time()
print('模型训练完成，用时(s)',time_end-time_start)

##测试集进行预测，并进行正确率验证
y_predict_test = model.predict([x_test_dynamic])
y_p_test = Scaler_y.inverse_transform(y_predict_test.reshape(-1,1))
y_t_test = Scaler_y.inverse_transform(y_test2.reshape(-1,1))
y_predict_train = model.predict([x_train_dynamic])
y_p_train = Scaler_y.inverse_transform(y_predict_train.reshape(-1,1))
y_t_train = Scaler_y.inverse_transform(y_train2.reshape(-1,1))

a_Result = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)##正确率检验
a_data_predict = np.concatenate((y_p_test, y_t_test), axis=1)

print('训练集样本数为%d' % y_train2.shape[0])
print('测试集样本数为%d' % y_test2.shape[0])

print('学习效果')
print('指标:','平均绝对误差','均方根误差','决定系数','平均绝对百分比误差','平均绝对误差的平均值')
print('测试集：',a_Result[0][0],a_Result[0][1],a_Result[0][2],a_Result[0][3],a_Result[0][4])
print('训练集：',a_Result[0][5],a_Result[0][6],a_Result[0][7],a_Result[0][8],a_Result[0][9])