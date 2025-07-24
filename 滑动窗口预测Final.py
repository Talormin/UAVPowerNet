import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Reshape, Dropout, LayerNormalization, GlobalAveragePooling1D, BatchNormalization, Conv1D, MultiHeadAttention
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
from keras.initializers import TruncatedNormal
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    """ Transformer Encoder 模块 """
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout, 
                           kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x, x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    return x + res

def model_dynamic(input_len_dynamic, timestep_dynamic, output_len, head_size=16, num_heads=2, 
                  num_transformer_blocks=2, ff_dim=16, dropout=0.1, lstm_units=64):
    input_dynamic = Input(shape=(timestep_dynamic, input_len_dynamic))
    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)   
    
    output_dynamic = GlobalAveragePooling1D(data_format="channels_first")(x)
    output_dynamic = Reshape((-1, 1))(output_dynamic)
    output_dynamic = LSTM(units=lstm_units, activation='tanh')(output_dynamic)

    hid_dynamic = Dense(units=64, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(output_dynamic)
    hid_dynamic = BatchNormalization()(hid_dynamic)
    hid_dynamic = Dense(units=32, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(hid_dynamic)
    hid_dynamic = Dropout(dropout)(hid_dynamic)
    outputs = Dense(units=output_len)(hid_dynamic)

    model = Model([input_dynamic], outputs)
    return model

def visualize_results_combined(results_combined):
    """
    可视化所有模型的总体分析
    """
    residuals = results_combined["all_y_true"] - results_combined["all_y_pred"]

    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["loss"], label='Training Loss', color='blue', linewidth=2)
    plt.plot(results_combined["val_loss"], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Overall Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 真实值 vs 预测值
    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["all_y_true"][:100], label='True Values', color='blue', linestyle='-', linewidth=2)
    plt.plot(results_combined["all_y_pred"][:100], label='Predicted Values', color='orange', linestyle='--', linewidth=2)
    plt.title('Overall True vs Predicted', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 残差分布图
    plt.figure(figsize=(10, 6))
    plt.hist(residuals.flatten(), bins=30, color='purple', alpha=0.7)
    plt.title('Overall Residual Distribution', fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

    # 4. 残差 vs 真实值
    plt.figure(figsize=(10, 6))
    plt.scatter(results_combined["all_y_true"], residuals, alpha=0.6, color='orange', label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2) 
    plt.title('Overall Residuals vs True Values', fontsize=16)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    """ 评估模型训练效果 """
    Result = []

    # Test Set Evaluation
    y_p = y_p_test
    y_t = y_t_test

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
    y_p = y_p_train
    y_t = y_t_train

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

def bayesian_optimization(x_train_mode, y_train_mode, x_test_mode, y_test_mode):
    """ 贝叶斯优化超参数搜索 """

    def objective(trial):
        lstm_units = trial.suggest_int("lstm_units", 32, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        epoch = trial.suggest_int("epoch", 10, 50)

        model = model_dynamic(input_len_dynamic=x_train_mode.shape[2], timestep_dynamic=x_train_mode.shape[1],
                              output_len=y_train_mode.shape[1], dropout=dropout, lstm_units=lstm_units)

        model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
        model.fit(x_train_mode, y_train_mode, epochs=epoch, batch_size=128, verbose=0)
        loss = model.evaluate(x_test_mode, y_test_mode, verbose=0)
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=25)  #######调参轮数########
    return study.best_params

def train_flight_mode_classifier(df):
    """ 训练随机森林飞行阶段分类器 """
    features = df.iloc[:, :20]  # 使用前 20 列作为特征
    labels = df.iloc[:, 20]  # 第 21 列是飞行阶段

    # 训练测试划分
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # 训练随机森林分类器
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(x_train, y_train)

    # 评估分类器
    accuracy = rf_model.score(x_test, y_test)
    print(f"随机森林飞行阶段分类器训练完成，测试集准确率: {accuracy:.4f}")

    return rf_model


def determine_flight_mode_rf(model, window_data):
    """ 使用随机森林模型预测飞行阶段，并返回占比最多的类别 """
    predictions = model.predict(window_data)
    most_common_phase = Counter(predictions).most_common(1)[0][0]
    return most_common_phase


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    """ Transformer Encoder 模块 """
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout,
                           kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x, x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    return x + res


def model_dynamic(input_len_dynamic, timestep_dynamic, output_len, head_size=16, num_heads=2,
                  num_transformer_blocks=2, ff_dim=16, dropout=0.1, lstm_units=64):
    input_dynamic = Input(shape=(timestep_dynamic, input_len_dynamic))
    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)

    output_dynamic = GlobalAveragePooling1D(data_format="channels_first")(x)
    output_dynamic = Reshape((-1, 1))(output_dynamic)
    output_dynamic = LSTM(units=lstm_units, activation='tanh')(output_dynamic)

    hid_dynamic = Dense(units=64, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(output_dynamic)
    hid_dynamic = BatchNormalization()(hid_dynamic)
    hid_dynamic = Dense(units=32, activation='relu', kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(hid_dynamic)
    hid_dynamic = Dropout(dropout)(hid_dynamic)
    outputs = Dense(units=output_len)(hid_dynamic)

    model = Model([input_dynamic], outputs)
    return model


# 数据集构建和处理
print("开始构建数据集")
data_path = r'C:\\Users\\LWJ\\Desktop\\Dataset'
df = pd.read_csv(data_path + '\\XiAn.csv')

# 训练随机森林模型
rf_classifier = train_flight_mode_classifier(df)

target_index = 19
M = 40  # 输入窗口长度
N = 8   # 预测窗口长度

x_data = []
y_data = []
modes = []

# 按滑动窗口生成数据
for i in range(len(df) - M - N + 1):
    # 输入窗口 M
    input_window = df.iloc[i:i + M, :20].values  # 取前 20 列作为输入
    # 预测窗口 N
    output_window = df.iloc[i + M:i + M + N, target_index].values
    # 预测飞行模式
    flight_mode = determine_flight_mode_rf(rf_classifier, input_window)

    x_data.append(input_window)
    y_data.append(output_window)
    modes.append(flight_mode)

x_data = np.array(x_data)
y_data = np.array(y_data)
modes = np.array(modes)

# 划分训练集和测试集
x_train, x_test, y_train, y_test, mode_train, mode_test = train_test_split(
    x_data, y_data, modes, test_size=0.2, random_state=20, shuffle=True
)

# 归一化
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x_train_all = np.concatenate(x_train, axis=0)
y_train_all = np.concatenate(y_train, axis=0)

scaler_x.fit(x_train_all)
scaler_y.fit(y_train_all.reshape(-1, 1))

state_train_data = {mode: ([], []) for mode in np.unique(modes)}
state_test_data = {mode: ([], []) for mode in np.unique(modes)}

# 对训练集按飞行模式分类
for x_group, y_group, mode in zip(x_train, y_train, mode_train):
    x_group_scaled = scaler_x.transform(x_group)
    y_group_scaled = scaler_y.transform(y_group.reshape(-1, 1)).flatten()
    state_train_data[mode][0].append(x_group_scaled)
    state_train_data[mode][1].append(y_group_scaled)

# 对测试集按飞行模式分类
for x_group, y_group, mode in zip(x_test, y_test, mode_test):
    x_group_scaled = scaler_x.transform(x_group)
    y_group_scaled = scaler_y.transform(y_group.reshape(-1, 1)).flatten()
    state_test_data[mode][0].append(x_group_scaled)
    state_test_data[mode][1].append(y_group_scaled)

# 转换为 NumPy 数组
for mode in np.unique(modes):
    state_train_data[mode] = (np.array(state_train_data[mode][0]), np.array(state_train_data[mode][1]))
    state_test_data[mode] = (np.array(state_test_data[mode][0]), np.array(state_test_data[mode][1]))

print(f"数据处理完成，共 {len(x_data)} 组数据")



# 模型训练与评估
results_combined = {"all_y_true": [], "all_y_pred": [], "loss": [], "val_loss": []}
evaluation_results = {}

for mode in np.unique(modes):
    print(f"开始训练飞行状态模型: {mode}")
    x_train_mode, y_train_mode = state_train_data[mode]
    x_test_mode, y_test_mode = state_test_data[mode]

    best_params = bayesian_optimization(x_train_mode, y_train_mode, x_test_mode, y_test_mode)
    model = model_dynamic(input_len_dynamic=x_train_mode.shape[2], timestep_dynamic=x_train_mode.shape[1],
                          output_len=y_train_mode.shape[1], dropout=best_params['dropout'], lstm_units=best_params['lstm_units'])

    model.compile(loss="mse", optimizer=Adam(learning_rate=best_params['learning_rate']))
    history = model.fit(x_train_mode, y_train_mode, epochs=best_params['epoch'], batch_size=128, validation_data=(x_test_mode, y_test_mode), verbose=1)

    y_pred_mode = model.predict(x_test_mode)
    y_pred_mode_inverse = scaler_y.inverse_transform(y_pred_mode)
    y_test_mode_inverse = scaler_y.inverse_transform(y_test_mode)

    # 训练集预测
    y_pred_train_mode = model.predict(x_train_mode)
    y_pred_train_mode_inverse = scaler_y.inverse_transform(y_pred_train_mode)
    y_train_mode_inverse = scaler_y.inverse_transform(y_train_mode)

    # 记录每个模型的结果
    results_combined["all_y_true"].extend(y_test_mode_inverse.flatten())
    results_combined["all_y_pred"].extend(y_pred_mode_inverse.flatten())
    results_combined["loss"].extend(history.history['loss'])
    results_combined["val_loss"].extend(history.history['val_loss'])

    # 模型评估
    result = Evaluate(
        y_p_test=y_pred_mode_inverse.flatten(), 
        y_t_test=y_test_mode_inverse.flatten(),
        y_p_train=y_pred_train_mode_inverse.flatten(), 
        y_t_train=y_train_mode_inverse.flatten()
    )
    evaluation_results[mode] = result.flatten()
    print(f"模型 {mode} 评估结果：", result)

# 将评估结果转换为 DataFrame 并输出
evaluation_df = pd.DataFrame.from_dict(
    evaluation_results, 
    orient='index', 
    columns=['Test MAE', 'Test RMSE', 'Test R2', 'Test MAPE', 'Train MAE', 'Train RMSE', 'Train R2', 'Train MAPE']
)
print("\n各模型评估结果：")
print(evaluation_df)

# 计算各个误差指标的平均值
average_results = evaluation_df.mean(axis=0)
print("\n三个模型各个误差指标的平均值：")
print(average_results)

# 统一可视化所有模型的结果
results_combined["all_y_true"] = np.array(results_combined["all_y_true"])
results_combined["all_y_pred"] = np.array(results_combined["all_y_pred"])
visualize_results_combined(results_combined)

