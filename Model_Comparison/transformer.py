import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import LSTM, GRU, Dense, Input, Reshape, Concatenate, Dropout, LayerNormalization, \
    GlobalAveragePooling1D, MultiHeadAttention, BatchNormalization, Add, Activation, \
    Conv1D, AveragePooling1D, Flatten
from keras.callbacks import EarlyStopping
from keras.metrics import RootMeanSquaredError
from keras.initializers import TruncatedNormal
from keras.utils import plot_model
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pywt  # Wavelet Transform
import optuna


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    """
    Transformer encoder block
    """
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout,
                           kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x, x)
    res = x + inputs  # Residual connection to prevent degradation

    # Feed Forward part (fully connected)
    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu",
               kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1,
               kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    return x + res  # Residual connection


def model_dynamic_without_lstm(input_len_dynamic, timestep_dynamic, output_len,
                               head_size=16, num_heads=2, num_transformer_blocks=4,
                               ff_dim=16, dropout=0.1):
    """
    Transformer-only dynamic model (LSTM removed)
    """
    input_dynamic = Input(shape=(timestep_dynamic, input_len_dynamic))

    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=dropout)

    output_dynamic = GlobalAveragePooling1D(data_format="channels_first")(x)

    hid_dynamic = Dense(units=64, activation='relu',
                        kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(output_dynamic)
    hid_dynamic = BatchNormalization()(hid_dynamic)
    hid_dynamic = Dense(units=32, activation='relu',
                        kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(hid_dynamic)
    hid_dynamic = Dropout(dropout)(hid_dynamic)

    outputs = Dense(units=output_len)(hid_dynamic)

    return Model([input_dynamic], outputs)
def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    """
    Evaluate model performance on test and train sets
    """
    Result = []

    # Evaluation on test set
    y_p = y_p_test
    y_t = y_t_test

    plt.figure()
    plt.scatter(y_p, y_t, c='blue', s=5, alpha=0.8)
    x_equal = [min(y_t), max(y_t)]
    plt.plot(x_equal, x_equal, color='orange')
    plt.show()

    SSR = np.sum((y_p - y_t) ** 2)
    SST = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    mae_mean = mae / np.abs(np.mean(y_t))
    rmse = np.sqrt(np.mean((y_p - y_t) ** 2))

    y_p = y_p[np.abs(y_t) > 0.0001]
    y_t = y_t[np.abs(y_t) > 0.0001]
    mape = np.mean(np.abs((y_p - y_t) / y_t))

    Result.extend([mae, rmse, r2, mape, mae_mean])

    # Evaluation on train set
    y_p = y_p_train
    y_t = y_t_train

    SSR = np.sum((y_p - y_t) ** 2)
    SST = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    mae_mean = mae / np.abs(np.mean(y_t))
    rmse = np.sqrt(np.mean((y_p - y_t) ** 2))

    y_p = y_p[np.abs(y_t) > 0.0001]
    y_t = y_t[np.abs(y_t) > 0.0001]
    mape = np.mean(np.abs((y_p - y_t) / y_t))

    Result.extend([mae, rmse, r2, mape, mae_mean])

    return np.array(Result).reshape(1, -1)

# ------------------------------
# Dataset construction
# ------------------------------

print("Start building dataset")
time_start = time.time()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

data_path = r'YOUR_DATA_PATH'
model_path = r'YOUR_MODEL_PATH'

df = pd.read_csv(data_path + '/27.csv')

feature = ['xyz_0', 'xyz_1', 'xyz_2', 'Final_true_airspeed_m_s',
           'Final_air_temperature_celsius', 'Final_true_ground_minus_wind_m_s',
           'Final_differential_pressure_pa', 'Final_x', 'Final_y', 'Final_z',
           'Final_vx', 'Final_vy', 'Final_vz', 'Final_windspeed_north',
           'Final_windspeed_east', 'Final_roll_body', 'Final_pitch_body',
           'Final_yaw_body', 'Final_total_energy_rate', 'Final_ax', 'Final_ay',
           'Final_az', 'Final_baro_alt_mete', 'Final_baro_pressure_pa',
           'Final_q1', 'Final_q2', 'Final_q3', 'Final_voltage_filtered_v',
           'Final_current_average_a']

seg_total = np.arange(len(df['xyz_0']))
seg_train, seg_test = train_test_split(seg_total, test_size=0.2, random_state=20, shuffle=True)

x_train, x_test, y_train, y_test = [], [], [], []
Scaler_x = MinMaxScaler()
Scaler_xt = MinMaxScaler()
Scaler_y = MinMaxScaler()
Scaler_yt = MinMaxScaler()

dd = df.values
Step = 20
kk_test = kk_train = 0
x_train1, y_train1, x_test1, y_test1 = [], [], [], []
target_index = 19

for i in range(len(dd)):
    if i in seg_train:
        x_train1.append(dd[i])
        y_train1.append(dd[i][target_index])
        kk_train += 1
    if i in seg_test:
        x_test1.append(dd[i])
        y_test1.append(dd[i][target_index])
        kk_test += 1
    if kk_train >= Step:
        x_train.append(Scaler_x.fit_transform(np.array(x_train1)))
        y_train.append(np.array(y_train1))
        x_train1, y_train1, kk_train = [], [], 0
    if kk_test >= Step:
        x_test.append(Scaler_xt.fit_transform(np.array(x_test1)))
        y_test.append(np.array(y_test1))
        x_test1, y_test1, kk_test = [], [], 0

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

y_train = Scaler_y.fit_transform(y_train)
y_test = Scaler_yt.fit_transform(y_test)

trans_train = list(range(19))  # Use first 19 features
x_train = x_train[:, :, trans_train]
x_test = x_test[:, :, trans_train]

print("Dataset built. Time:", time.time() - time_start)

print("Start training model")
# Positional encoding
timeStep = 20
x_dynamic_position = np.arange(timeStep) / timeStep
x_position = np.expand_dims(x_dynamic_position, 0).repeat(19, axis=0).T
x_train_position = np.expand_dims(x_position, 0).repeat(x_train.shape[0], axis=0)
x_test_position = np.expand_dims(x_position, 0).repeat(x_test.shape[0], axis=0)

x_train_dynamic = x_train + x_train_position
x_test_dynamic = x_test + x_test_position

# Add position encoding to y (time-sensitive target)
for i in range(len(y_train)):
    for j in range(timeStep):
        y_train[i][j] += j / timeStep
for i in range(len(y_test)):
    for j in range(timeStep):
        y_test[i][j] += j / timeStep

# Initialize and compile model
best_model = model_dynamic_without_lstm(
    input_len_dynamic=19,
    timestep_dynamic=20,
    output_len=20
)

best_model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=0.001),
    metrics=[RootMeanSquaredError()]
)

history = best_model.fit(
    x_train_dynamic, y_train,
    epochs=100,
    batch_size=1024,
    validation_data=(x_test_dynamic, y_test)
)

# Plot training and validation loss
history_train = history.history
plt.figure()
plt.plot(history_train['loss'][1:], label='Training')
plt.plot(history_train['val_loss'][1:], label='Validation')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()

plt.figure()
plt.plot(history_train['loss'][100:], label='Training (Zoomed)')
plt.plot(history_train['val_loss'][100:], label='Validation (Zoomed)')
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Save model
best_model.save(model_path + '/transformer_model.h5')

print("Model training completed. Time elapsed:", time.time() - time_start)

# Predict
y_predict_test = best_model.predict(x_test_dynamic)
y_predict_train = best_model.predict(x_train_dynamic)

y_p_test = Scaler_yt.inverse_transform(y_predict_test).reshape(-1)
y_t_test = Scaler_yt.inverse_transform(y_test).reshape(-1)
y_p_train = Scaler_y.inverse_transform(y_predict_train).reshape(-1)
y_t_train = Scaler_y.inverse_transform(y_train).reshape(-1)

# Evaluate
a_Result = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)

print(f"Train samples: {y_train.shape[0]}")
print(f"Test samples: {y_test.shape[0]}")
print("Evaluation metrics:")
print("Metric:", 'MAE', 'RMSE', 'R2', 'MAPE', 'MAE/Mean')
print("Test:", a_Result[0][0], a_Result[0][1], a_Result[0][2], a_Result[0][3], a_Result[0][4])
print("Train:", a_Result[0][5], a_Result[0][6], a_Result[0][7], a_Result[0][8], a_Result[0][9])

# Visualization
def visualize_results(history, y_true, y_pred):
    residuals = y_true - y_pred

    # Loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # True vs Predicted
    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100], label='True Values', color='blue')
    plt.plot(y_pred[:100], label='Predicted Values', color='orange')
    plt.title('True vs Predicted (Test Set)')
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Residual histogram
    plt.figure(figsize=(10, 6))
    plt.hist(residuals.flatten(), bins=30, color='purple', alpha=0.7)
    plt.title('Residual Distribution')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Residuals vs True
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.6, color='orange')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals vs True Values')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.show()

visualize_results(history, y_t_test.flatten(), y_p_test.flatten())

