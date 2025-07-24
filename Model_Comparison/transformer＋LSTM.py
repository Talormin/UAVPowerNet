import pandas as pd
import numpy as np
import time
import tensorflow as tf
from keras.models import Model, load_model
from keras.layers import LSTM, Dense, Input, Reshape, Dropout, LayerNormalization, \
                         GlobalAveragePooling1D, MultiHeadAttention, BatchNormalization, Conv1D
from keras.callbacks import EarlyStopping
from keras.initializers import TruncatedNormal
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout,
                           kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x, x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu",
               kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1,
               kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    return x + res

def model_dynamic(input_len_dynamic, timestep_dynamic, output_len,
                  head_size=16, num_heads=2, num_transformer_blocks=4,
                  ff_dim=16, dropout=0.1, lstm_units=64):
    input_dynamic = Input(shape=(timestep_dynamic, input_len_dynamic))
    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout=0.2)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Reshape((-1, 1))(x)
    x = LSTM(units=lstm_units, activation='tanh')(x)
    x = Dense(units=64, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(x)
    x = BatchNormalization()(x)
    x = Dense(units=32, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0., stddev=0.001))(x)
    x = Dropout(dropout)(x)
    outputs = Dense(units=output_len)(x)
    return Model([input_dynamic], outputs)

def objective(trial, X_train, X_val, y_train, y_val):
    head_size = trial.suggest_int('head_size', 8, 64)
    num_heads = trial.suggest_int('num_heads', 1, 8)
    num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 1, 4)
    ff_dim = trial.suggest_int('ff_dim', 16, 128)
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    dropout = trial.suggest_float('dropout', 0.2, 0.5)
    epoch = trial.suggest_int('epoch', 10, 200)
    batch_size = trial.suggest_int('batch_size', 512, 1024)
    learn_rate = trial.suggest_float('learn_rate', 0.0001, 0.1)

    input_dynamic = Input(shape=(20, 19))
    x = input_dynamic
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = GlobalAveragePooling1D(data_format="channels_first")(x)
    x = Reshape((-1, 1))(x)
    x = LSTM(units=lstm_units, activation='sigmoid')(x)
    x = BatchNormalization()(x)
    x = Dense(units=32, activation='relu',
              kernel_initializer=TruncatedNormal(mean=0., stddev=0.01))(x)
    x = Dropout(0.1)(x)
    outputs = Dense(units=20)(x)

    model = Model([input_dynamic], outputs)
    model.compile(loss="mse", optimizer=Adam(learning_rate=learn_rate))
    model.fit(X_train, y_train, epochs=epoch, batch_size=batch_size,
              validation_data=(X_val, y_val), verbose=0)
    return model.evaluate(X_val, y_val, verbose=0)

def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    Result = []
    for y_p, y_t in [(y_p_test, y_t_test), (y_p_train, y_t_train)]:
        SSR = np.sum((y_p - y_t) ** 2)
        SST = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - SSR / SST
        mae = np.mean(np.abs(y_p - y_t))
        mae_mean = mae / np.abs(np.mean(y_t))
        rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
        mask = np.abs(y_t) > 1e-4
        mape = np.mean(np.abs((y_p[mask] - y_t[mask]) / y_t[mask]))
        Result.extend([mae, rmse, r2, mape, mae_mean])
    return np.array(Result).reshape(1, -1)

print("Start building dataset...")
time_start = time.time()
data_path = r'YOUR_DATA_PATH'
model_path = r'YOUR_MODEL_PATH'

df = pd.read_csv(data_path + '/27.csv')
feature = ['xyz_0', 'xyz_1', 'xyz_2', 'Final_true_airspeed_m_s', 'Final_air_temperature_celsius',
           'Final_true_ground_minus_wind_m_s', 'Final_differential_pressure_pa', 'Final_x', 'Final_y',
           'Final_z', 'Final_vx', 'Final_vy', 'Final_vz', 'Final_windspeed_north', 'Final_windspeed_east',
           'Final_roll_body', 'Final_pitch_body', 'Final_yaw_body', 'Final_total_energy_rate',
           'Final_ax', 'Final_ay', 'Final_az', 'Final_baro_alt_mete', 'Final_baro_pressure_pa',
           'Final_q1', 'Final_q2', 'Final_q3', 'Final_voltage_filtered_v', 'Final_current_average_a']

seg_total = np.arange(len(df['xyz_0']))
seg_train, seg_test = train_test_split(seg_total, test_size=0.2, random_state=20, shuffle=True)
Scaler_x = MinMaxScaler()
Scaler_xt = MinMaxScaler()
Scaler_y = MinMaxScaler()
Scaler_yt = MinMaxScaler()

dd = df.values
Step = 20
x_train, y_train, x_test, y_test = [], [], [], []
x_train1, y_train1, x_test1, y_test1 = [], [], [], []
kk_train = kk_test = 0
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

x_train, x_test = np.array(x_train), np.array(x_test)
y_train, y_test = Scaler_y.fit_transform(np.array(y_train)), Scaler_yt.fit_transform(np.array(y_test))

trans_train = list(range(19))
x_train = x_train[:, :, trans_train]
x_test = x_test[:, :, trans_train]

x_position = np.expand_dims(np.arange(Step) / Step, 0).repeat(19, axis=0).T
x_train_position = np.expand_dims(x_position, 0).repeat(x_train.shape[0], axis=0)
x_test_position = np.expand_dims(x_position, 0).repeat(x_test.shape[0], axis=0)

x_train_dynamic = x_train + x_train_position
x_test_dynamic = x_test + x_test_position

for i in range(len(y_train)):
    y_train[i] += np.arange(Step) / Step
for i in range(len(y_test)):
    y_test[i] += np.arange(Step) / Step

print("Dataset ready. Time used:", time.time() - time_start)

print("Start training...")
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, x_train_dynamic, x_test_dynamic, y_train, y_test), n_trials=100)

best_params = study.best_trial.params
print("Best hyperparameters:", best_params)

best_model = model_dynamic(19, 20, 20, **{k: best_params[k] for k in best_params if k in [
    'head_size', 'num_heads', 'num_transformer_blocks', 'ff_dim', 'dropout', 'lstm_units']})
best_model.compile(loss="mse", optimizer=Adam(learning_rate=best_params['learn_rate']))
history = best_model.fit(x_train_dynamic, y_train, epochs=best_params['epoch'],
                         batch_size=best_params['batch_size'], validation_data=(x_test_dynamic, y_test))

best_model.save(model_path + '/transformer_model.h5')

y_p_test = Scaler_yt.inverse_transform(best_model.predict(x_test_dynamic)).reshape(-1)
y_t_test = Scaler_yt.inverse_transform(y_test).reshape(-1)
y_p_train = Scaler_y.inverse_transform(best_model.predict(x_train_dynamic)).reshape(-1)
y_t_train = Scaler_y.inverse_transform(y_train).reshape(-1)

a_Result = Evaluate(y_p_test, y_t_test, y_p_train, y_t_train)

print('Train Samples:', y_train.shape[0])
print('Test Samples:', y_test.shape[0])
print('Performance (MAE, RMSE, R2, MAPE, MAE-MEAN):')
print('Test:', a_Result[0][:5])
print('Train:', a_Result[0][5:])

def visualize_results(history, y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100], label='True')
    plt.plot(y_pred[:100], label='Predicted')
    plt.title('Prediction on Test Set')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30)
    plt.title('Residual Distribution')
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs True Values')
    plt.grid()
    plt.show()

visualize_results(history, y_t_test, y_p_test)
