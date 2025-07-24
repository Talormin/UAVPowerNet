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
import os

# Transformer Encoder 模块
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x, x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu", kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1, kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    return x + res

# 动态模型定义
def model_dynamic(input_len_dynamic, timestep_dynamic, output_len, head_size=16, num_heads=2, num_transformer_blocks=2, ff_dim=16, dropout=0.1, lstm_units=64):
    input_dynamic = Input(shape=(timestep_dynamic, input_len_dynamic))
    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=dropout)
    
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

# 飞行阶段分类
def determine_flight_mode(velocity_data):
    z_velocity = velocity_data[:, 2]
    z_velocity_mean = np.mean(z_velocity)

    if z_velocity_mean > 0.1:
        return "ascending"
    elif z_velocity_mean < -0.1:
        return "descending"
    else:
        return "other"

# 贝叶斯优化
def bayesian_optimization(x_train, y_train, x_test, y_test):
    def objective(trial):
        lstm_units = trial.suggest_int("lstm_units", 32, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        epochs = trial.suggest_int("epochs", 10, 50)
                                                                                                                            
        model = model_dynamic(input_len_dynamic=x_train.shape[2], timestep_dynamic=x_train.shape[1], output_len=y_train.shape[1], dropout=dropout, lstm_units=lstm_units)
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
        model.fit(x_train, y_train, epochs=epochs, batch_size=128, verbose=0)
        loss = model.evaluate(x_test, y_test, verbose=0)
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)
    return study.best_params

# 数据预处理
data_path = r'C:\Users\LWJ\Desktop'
df = pd.read_csv(data_path + '\\27.csv')
target_index = 19

M, N = 10, 5
x_data, y_data, modes = [], [], []

for i in range(len(df) - M - N + 1):
    input_window = df.iloc[i:i + M].values
    output_window = df.iloc[i + M:i + M + N, target_index].values
    flight_mode = determine_flight_mode(df.iloc[i:i + M, 7:10].values)

    x_data.append(input_window)
    y_data.append(output_window)
    modes.append(flight_mode)

x_data, y_data, modes = np.array(x_data), np.array(y_data), np.array(modes)
x_train, x_test, y_train, y_test, mode_train, mode_test = train_test_split(x_data, y_data, modes, test_size=0.2, random_state=20, shuffle=True)

scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
scaler_x.fit(np.concatenate(x_train, axis=0))
scaler_y.fit(np.concatenate(y_train, axis=0).reshape(-1, 1))

state_train_data = {mode: ([], []) for mode in np.unique(modes)}
state_test_data = {mode: ([], []) for mode in np.unique(modes)}

for x_group, y_group, mode in zip(x_train, y_train, mode_train):
    state_train_data[mode][0].append(scaler_x.transform(x_group))
    state_train_data[mode][1].append(scaler_y.transform(y_group.reshape(-1, 1)).flatten())

for x_group, y_group, mode in zip(x_test, y_test, mode_test):
    state_test_data[mode][0].append(scaler_x.transform(x_group))
    state_test_data[mode][1].append(scaler_y.transform(y_group.reshape(-1, 1)).flatten())

for mode in np.unique(modes):
    state_train_data[mode] = (np.array(state_train_data[mode][0]), np.array(state_train_data[mode][1]))
    state_test_data[mode] = (np.array(state_test_data[mode][0]), np.array(state_test_data[mode][1]))

# 模型训练
output_csv = os.path.join(os.path.expanduser("~"), "Desktop", "model_losses.csv")
csv_data = []

for mode in np.unique(modes):
    x_train_mode, y_train_mode = state_train_data[mode]
    x_test_mode, y_test_mode = state_test_data[mode]

    best_params = bayesian_optimization(x_train_mode, y_train_mode, x_test_mode, y_test_mode)
    model = model_dynamic(input_len_dynamic=x_train_mode.shape[2], timestep_dynamic=x_train_mode.shape[1], output_len=y_train_mode.shape[1],dropout=best_params['dropout'],lstm_units=best_params['lstm_units'])
    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']), loss='mse')
    history = model.fit(x_train_mode, y_train_mode, epochs=best_params['epochs'], batch_size=128, validation_data=(x_test_mode, y_test_mode), verbose=1)

    csv_data.append({
        "mode": mode,
        "train_loss": history.history['loss'],
        "val_loss": history.history['val_loss']
    })

    model.save(os.path.join(os.path.expanduser("~"), f"Desktop/{mode}_model.h5"))

loss_df = pd.DataFrame(csv_data)
loss_df.to_csv(output_csv, index=False)
print(f"训练完成，损失记录保存至 {output_csv}")
