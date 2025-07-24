import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Input, Reshape, Dropout, LayerNormalization, GlobalAveragePooling1D, BatchNormalization
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def build_model(
    input_len,
    time_steps,
    output_len,
    lstm_units=64,
    dropout=0.1
):
    input_layer = Input(shape=(time_steps, input_len))
    x = GlobalAveragePooling1D(data_format="channels_first")(input_layer)
    x = Reshape((-1, 1))(x)
    x = LSTM(units=lstm_units, activation='tanh')(x)
    x = Dense(units=64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dense(units=32, activation='relu')(x)
    x = Dropout(dropout)(x)
    output = Dense(units=output_len)(x)
    model = Model(inputs=input_layer, outputs=output)
    return model

def evaluate_model(y_pred_test, y_true_test, y_pred_train, y_true_train):
    results = []

    # --- Evaluation on test set ---
    y_p = y_pred_test
    y_t = y_true_test

    plt.figure()
    plt.scatter(y_p, y_t, c='blue', s=5, alpha=0.8)
    x_equal = [min(y_t), max(y_t)]
    plt.plot(x_equal, x_equal, color='orange')
    plt.title('Predicted vs True (Test Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.grid(True)
    plt.show()

    SSR = np.sum((y_p - y_t)**2)
    SST = np.sum((y_t - np.mean(y_t))**2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    mae_mean = mae / np.abs(np.mean(y_t))
    rmse = np.sqrt(np.mean((y_p - y_t)**2))
    non_zero_mask = np.abs(y_t) > 1e-4
    mape = np.mean(np.abs((y_p[non_zero_mask] - y_t[non_zero_mask]) / y_t[non_zero_mask]))

    results.extend([mae, rmse, r2, mape, mae_mean])

    # --- Evaluation on train set ---
    y_p = y_pred_train
    y_t = y_true_train

    SSR = np.sum((y_p - y_t)**2)
    SST = np.sum((y_t - np.mean(y_t))**2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    mae_mean = mae / np.abs(np.mean(y_t))
    rmse = np.sqrt(np.mean((y_p - y_t)**2))
    non_zero_mask = np.abs(y_t) > 1e-4
    mape = np.mean(np.abs((y_p[non_zero_mask] - y_t[non_zero_mask]) / y_t[non_zero_mask]))

    results.extend([mae, rmse, r2, mape, mae_mean])
    return np.array(results).reshape(1, -1)

# ----------------- Data Preparation --------------------

print("Building dataset...")
start_time = time.time()

data_path = 'YOUR_DATA_PATH'
model_path = 'YOUR_MODEL_PATH'

df = pd.read_csv(data_path + '/27.csv')

features = [
    'xyz_0', 'xyz_1', 'xyz_2', 'Final_true_airspeed_m_s', 'Final_air_temperature_celsius',
    'Final_true_ground_minus_wind_m_s', 'Final_differential_pressure_pa', 'Final_x', 'Final_y',
    'Final_z', 'Final_vx', 'Final_vy', 'Final_vz', 'Final_windspeed_north', 'Final_windspeed_east',
    'Final_roll_body', 'Final_pitch_body', 'Final_yaw_body', 'Final_total_energy_rate', 'Final_ax',
    'Final_ay', 'Final_az', 'Final_baro_alt_mete', 'Final_baro_pressure_pa', 'Final_q1', 'Final_q2',
    'Final_q3', 'Final_voltage_filtered_v', 'Final_current_average_a'
]

target_index = 19  # Index in raw data, not feature index

total_indices = np.array(range(len(df)))
train_idx, test_idx = train_test_split(total_indices, test_size=0.2, random_state=20, shuffle=True)

x_train, x_test, y_train, y_test = [], [], [], []
scaler_x = MinMaxScaler()
scaler_xt = MinMaxScaler()
scaler_y = MinMaxScaler()
scaler_yt = MinMaxScaler()

step = 20
x_temp_train, x_temp_test = [], []
y_temp_train, y_temp_test = [], []
counter_train = counter_test = 0

data_array = df.values

for i in range(len(data_array)):
    if i in train_idx:
        x_temp_train.append(data_array[i])
        y_temp_train.append(data_array[i][target_index])
        counter_train += 1
        if counter_train >= step:
            x_temp_train = np.array(x_temp_train)
            y_temp_train = np.array(y_temp_train)
            x_temp_train = scaler_x.fit_transform(x_temp_train)
            x_train.append(x_temp_train)
            y_train.append(y_temp_train)
            x_temp_train, y_temp_train = [], []
            counter_train = 0
    elif i in test_idx:
        x_temp_test.append(data_array[i])
        y_temp_test.append(data_array[i][target_index])
        counter_test += 1
        if counter_test >= step:
            x_temp_test = np.array(x_temp_test)
            y_temp_test = np.array(y_temp_test)
            x_temp_test = scaler_xt.fit_transform(x_temp_test)
            x_test.append(x_temp_test)
            y_test.append(y_temp_test)
            x_temp_test, y_temp_test = [], []
            counter_test = 0

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = np.array(y_train), np.array(y_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_yt.fit_transform(y_test)

print("Finished building dataset in %.2f seconds." % (time.time() - start_time))

def visualize_results(history, y_true, y_pred):
    residuals = y_true - y_pred

    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.show()

    # 2. True vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100], label='True Values', color='blue', linestyle='-')
    plt.plot(y_pred[:100], label='Predicted Values', color='orange', linestyle='--')
    plt.title('True vs Predicted', fontsize=16)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. Residual Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # 4. Residuals vs True
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.6, color='orange')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

# Call the visualizer
visualize_results(history, y_true_test_inv, y_pred_test_inv)

