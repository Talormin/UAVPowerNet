##
##该代码首先将数据集按步长20分组，在每个步长内对各个信号进行小波变换（纵向，即对每类数据进行小波变换），之后将得到的特征输入transformer模型中以预测一个步长内的平均功率。
##
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
import pywt
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
    input_len_w1,
    input_len_w2,
    input_len_w3,
    input_len_raw,
    timestep_w1,
    timestep_w2,
    timestep_w3,
    timestep_raw,
    output_len,
    head_size=16,   
    num_heads=2,
    num_transformer_blocks=2,
    ff_dim=16,
    lstm_units = 64,
    dropout = 0.1,


):
    input_w1 = Input(shape=(timestep_w1,input_len_w1))##初始化深度学习网络输入层的tensor(张量)
    input_w2 = Input(shape=(timestep_w2,input_len_w2))##初始化深度学习网络输入层的tensor(张量)
    input_w3 = Input(shape=(timestep_w3,input_len_w3))##初始化深度学习网络输入层的tensor(张量)
    input_raw = Input(shape=(timestep_raw,input_len_raw))##初始化深度学习网络输入层的tensor(张量)

    x = input_w1
    for _ in range(num_transformer_blocks):##基本频率
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_w1 = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    x = input_w2
    for _ in range(num_transformer_blocks):##低频
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_w2 = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    x = input_w3
    for _ in range(num_transformer_blocks):##高频
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_w3 = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    x = input_raw##原始数据
    for _ in range(num_transformer_blocks):##定义循环次数（特征提取次数？）
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_raw = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    output_w =  Concatenate(axis=1)([output_w1, output_w2,output_w3,output_raw])

    output_w = Reshape((-1, 1))(output_w)
    output_w = LSTM(units=lstm_units, activation='tanh')(output_w)
    
    hid_dynamic = BatchNormalization()(output_w)##批标准化
    hid_dynamic = Dense(units=64, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(hid_dynamic)##全连接层 
    hid_dynamic = Dropout(dropout)(hid_dynamic)##Dropout可以通过在训练神经网络期间随机丢弃单元来防止过拟合
    outputs = Dense(units=output_len)(hid_dynamic)##全连接层 
    
    model = Model([input_w1,input_w2,input_w3,input_raw], outputs)## inputs与outputs一定是Layer调用输出的张量
    return model

def objective(trial,x_train_w1, x_train_w2, x_train_w3, x_train,y_train,x_test_w1, x_test_w2, x_test_w3, x_test,y_test):##调参模型
    # 从 Optuna 中采样超参数
    head_size = trial.suggest_int('head_size', 8, 64)
    num_heads = trial.suggest_int('num_heads', 1, 8)
    num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 1, 4)
    ff_dim = trial.suggest_int('ff_dim', 16, 128)
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    epochs =  trial.suggest_int('epoch', 10, 200)
    batch_size =  trial.suggest_int('batch_size', 512,1024)
    learn_rate =  trial.suggest_float('learn_rate', 0.00001,0.1)

    step = 20
    input_len_w1 = len(x_train_w1[0][0])
    input_len_w2 = len(x_train_w2[0][0])
    input_len_w3 = len(x_train_w3[0][0])
    input_len_raw = len(x_train[0][0])
    timestep_w1 = step
    timestep_w2 = step
    timestep_w3 = step
    timestep_raw = step

    input_w1 = Input(shape=(timestep_w1,input_len_w1))##初始化深度学习网络输入层的tensor(张量)
    input_w2 = Input(shape=(timestep_w2,input_len_w2))##初始化深度学习网络输入层的tensor(张量)
    input_w3 = Input(shape=(timestep_w3,input_len_w3))##初始化深度学习网络输入层的tensor(张量)
    input_raw = Input(shape=(timestep_raw,input_len_raw))##初始化深度学习网络输入层的tensor(张量)

    x = input_w1
    for _ in range(num_transformer_blocks):##基本频率
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_w1 = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    x = input_w2
    for _ in range(num_transformer_blocks):##低频
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_w2 = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    x = input_w3
    for _ in range(num_transformer_blocks):##高频
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_w3 = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    x = input_raw##原始数据
    for _ in range(num_transformer_blocks):##定义循环次数（特征提取次数？）
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    output_raw = GlobalAveragePooling1D(data_format="channels_first")(x)##池化

    output_w =  Concatenate(axis=1)([output_w1, output_w2,output_w3,output_raw])

    output_w = Reshape((-1, 1))(output_w)
    output_w = LSTM(units=lstm_units, activation='tanh')(output_w)
    
    hid_dynamic = BatchNormalization()(output_w)##批标准化
    hid_dynamic = Dense(units=64, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(hid_dynamic)##全连接层 
    hid_dynamic = Dropout(dropout)(hid_dynamic)##Dropout可以通过在训练神经网络期间随机丢弃单元来防止过拟合
    outputs = Dense(units=1)(hid_dynamic)##全连接层 
    
    model = Model([input_w1,input_w2,input_w3,input_raw], outputs)## inputs与outputs一定是Layer调用输出的张量

    model.compile(loss='mse',optimizer=Adam(learning_rate=learn_rate))
    # 训练模型并返回验证损失（或使用交叉验证）
    model.fit([x_train_w1,x_train_w2,x_train_w3,x_train], y_train, epochs=epochs, batch_size=batch_size, validation_data=([x_test_w1, x_test_w2, x_test_w3, x_test],y_test), verbose=0)
    loss = model.evaluate([x_test_w1, x_test_w2, x_test_w3, x_test],y_test, verbose=0)
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

# 2. 小波变换：对时间序列进行小波分解
def wavelet_transform(data, wavelet='db4', level=2):##输出为一个二维数组，包含3个分量，第一个分量为近似系数（Approximation coefficients）：信号经过多层分解后，剩余的低频部分，包含了信号的大部分能量。后面两个依次为低频高频分量
    """这里输入是一个长度为20的二维数组
    对时间序列数据进行小波变换
    :param data: 输入的时间序列数据
    :param wavelet: 使用的小波函数
    :param level: 分解的层数
    :return: 小波分解的系数
    """
    w1 = []
    w2 = []
    w3 = []
    data = np.array(data).T##由于初始数据集的同类信号是在一列上，转置后将他们编导同一行，方便小波变换
    for i in range(20):
        coeffs = pywt.wavedec(data[i], wavelet, level=level)  # 小波分解
        w1.append(coeffs[0])##基本形状
        w2.append(coeffs[1])##最低频率
        w3.append(coeffs[2])##最高频率
    return w1,w2,w3

def number(data):##为输入数据编码,输入数据的格式为[nan,20,nan]
    for i in range(len(data)):
        for j in range(len(data[i])):
            index = np.array(range(0,len(data[i][j])))/20
            data[i][j] = data[i][j] + index
    return data

print("开始构建数据集")
time_start = time.time()
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
data_path = r'C:\Users\LWJ\Desktop'
model_path = r'C:\Users\LWJ\Desktop'

df = pd.read_csv(data_path + '\\27.csv')

seg_total = np.array(range(len(df['xyz_0'])))##构建训练集与测试集
seg_train, seg_test = train_test_split(seg_total, test_size=0.2, random_state = 20, shuffle = True)##构建训练集与测试集（这里是将序号进行分类）

Scaler_x = MinMaxScaler(feature_range=(0,1))##Scaler_x记录了这种归一化的规范，用于统一归一化的方式和还原数据
Scaler_y = MinMaxScaler(feature_range=(0,1))##归一化函数的初始化

dd = df.values

Step = 20##步长
x_train_w1 = list()##构建数组用于存储测试集与训练集的数据
x_train_w2 = list()##构建数组用于存储测试集与训练集的数据
x_train_w3 = list()##构建数组用于存储测试集与训练集的数据
x_test_w1 = list()
x_test_w2 = list()
x_test_w3 = list()
y_train = list()
y_test = list()
kk_test = 0
kk_train = 0
x_train1 = list()
x_train = list()
x_test1 = list()
x_test = list()
y_train1 = list()
y_test1 = list()
feature = np.array(range(25))
for i in range(0,len(dd)):##数据集分类(并按步长分组)
    if i in seg_train:
        x_train1.append(dd[i])
        y_train1.append(dd[i][25])##LSTM的y数据
        kk_train = kk_train+1
    if i in seg_test:
        x_test1.append(dd[i])
        y_test1.append(dd[i][25])
        kk_test = kk_test+1
    if kk_train > (Step-1):
        x_train1 = np.array(x_train1)
        y_train1 = np.array(y_train1)
        x_train1 = Scaler_x.fit_transform(x_train1[:,feature])##小波变换
        kk_train = 0
        w1,w2,w3 = wavelet_transform(x_train1)##训练集分量
        x_train_w1.append(w1)
        x_train_w2.append(w2)
        x_train_w3.append(w3)
        x_train.append(x_train1)
        y_train.append(y_train1)
        x_train1 = []
        y_train1 = []
    if kk_test > (Step-1):
        x_test1 = np.array(x_test1)
        y_test1 = np.array(y_test1)
        x_test1 = Scaler_x.fit_transform(x_test1[:,feature])
        kk_test = 0
        w1,w2,w3 = wavelet_transform(x_test1)##测试集分量
        x_test_w1.append(w1)
        x_test_w2.append(w2)
        x_test_w3.append(w3)
        x_test.append(x_test1)
        y_test.append(y_test1)
        x_test1 = []
        y_test1 = []
x_train_w1 = np.array(x_train_w1)
x_train_w2 = np.array(x_train_w2)
x_train_w3 = np.array(x_train_w3)
x_test_w1 = np.array(x_test_w1)
x_test_w2 = np.array(x_test_w2)
x_test_w3 = np.array(x_test_w3)
y_train = np.array(y_train)
y_test = np.array(y_test)
x_train = np.array(x_train)
x_test = np.array(x_test)
y_train2 = []
y_test2 = []
for i in range(0,len(y_train)):
    y_train2.append(np.mean(y_train[i]))
for i in range(0,len(y_test)):
    y_test2.append(np.mean(y_test[i]))
y_train2 = np.array(y_train2)
y_test2 = np.array(y_test2)##进行lstm的数据集
y_train2 = Scaler_y.fit_transform(y_train2.reshape(-1,1))
y_test2 = Scaler_y.fit_transform(y_test2.reshape(-1,1))

time_end = time.time()
print("数据集构建完成,用时",time_end-time_start)

##输入分三个数组，每个数组对应了小波分解后的一个分量，y为一维数组，为20组数据功率的平均值

# Transformer-MLP
# positional embedding（位置编码，用于标记每个数据的位置信息）
time_start = time.time()

step = 20
x_train_w1 = number(x_train_w1)
x_train_w2 = number(x_train_w2)
x_train_w3 = number(x_train_w3)
x_test_w1 = number(x_test_w1)
x_test_w2 = number(x_test_w2)
x_test_w3 = number(x_test_w3)
x_dynamic_position = np.arange(step)/step
x_position = np.expand_dims(x_dynamic_position,0).repeat(len(feature),axis=0).T##维度增加
x_train_position = np.expand_dims(x_position,0).repeat(x_train.shape[0],axis=0)
x_test_position = np.expand_dims(x_position,0).repeat(x_test.shape[0],axis=0)
x_train = x_train + x_train_position
x_test = x_test + x_test_position

# 使用 Optuna 进行超参数优化
study = optuna.create_study(direction='minimize')  # 目标是最小化损失
study.optimize(lambda trial: objective(trial, x_train_w1, x_train_w2,x_train_w3,x_train, y_train2,x_test_w1, x_test_w2,x_test_w3,x_test, y_test2), n_trials=200)  # 尝试 50 次不同的超参数组合

# 输出最佳超参数
best_params = study.best_trial.params
print("Best hyperparameters:", best_params)

##模型训练部分#
print('开始训练模型')
model = model_dynamic(
    input_len_w1 = len(x_train_w1[0][0]),
    input_len_w2 = len(x_train_w2[0][0]),
    input_len_w3 = len(x_train_w3[0][0]),
    input_len_raw = len(x_train[0][0]),
    timestep_w1 = step,
    timestep_w2 = step,
    timestep_w3 = step,
    timestep_raw = step,
    output_len=1,
    head_size=best_params['head_size'],   
    num_heads=best_params['num_heads'],
    num_transformer_blocks=best_params['num_transformer_blocks'],
    ff_dim=best_params['ff_dim'],
    lstm_units = best_params['lstm_units'],
    dropout = best_params['dropout']
    )

model.compile(loss="mse", optimizer=Adam(learning_rate=best_params['learn_rate']), metrics=[RootMeanSquaredError()])##模型训练参数
# callbacks = [EarlyStopping(patience=30, restore_best_weights=True)]
history = model.fit(##模型训练
    [x_train_w1,x_train_w2,x_train_w3,x_train], y_train2,
    batch_size=best_params['batch_size'], epochs=best_params['epoch'],
    validation_data=([x_test_w1,x_test_w2,x_test_w3,x_test], y_test2))
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
y_predict_test = model.predict([x_test_w1,x_test_w2,x_test_w3,x_test])
y_p_test = Scaler_y.inverse_transform(y_predict_test.reshape(-1,1))
y_t_test = Scaler_y.inverse_transform(y_test2.reshape(-1,1))
y_predict_train = model.predict([x_train_w1,x_train_w2,x_train_w3,x_train])
y_p_train = Scaler_y.inverse_transform(y_predict_train.reshape(-1,1))
y_t_train = Scaler_y.inverse_transform(y_train2.reshape(-1,1))

a_Result = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)##正确率检验
a_data_predict = np.concatenate((y_p_test, y_t_test), axis=1)

print('训练集样本数为%d' % y_train2.shape[0])
print('测试集样本数为%d' % y_test2.shape[0])

print('训练集样本数为%d' % y_train.shape[0])
print('测试集样本数为%d' % y_test.shape[0])
print('学习效果(')
print('指标:','平均绝对误差','均方根误差','决定系数','平均绝对百分比误差','平均绝对误差的平均值')
print('测试集：',a_Result[0][0],a_Result[0][1],a_Result[0][2],a_Result[0][3],a_Result[0][4])
print('训练集：',a_Result[0][5],a_Result[0][6],a_Result[0][7],a_Result[0][8],a_Result[0][9])