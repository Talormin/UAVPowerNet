import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import LSTM, Dense, Input, Reshape, Dropout, LayerNormalization, \
    GlobalAveragePooling1D, BatchNormalization, Conv1D, MultiHeadAttention
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
from keras.initializers import TruncatedNormal
from tcn import TCN


def model_dynamic(input_len_dynamic, timestep_dynamic, output_len, dropout=0.1, tcn_filters=64, tcn_kernel_size=3,
                  tcn_layers=3):
    input_dynamic = Input(shape=(timestep_dynamic, input_len_dynamic))
    x = input_dynamic

    # TCN Layers
    x = TCN(nb_filters=tcn_filters, kernel_size=tcn_kernel_size, nb_stacks=tcn_layers,
            dilations=[2 ** i for i in range(tcn_layers)], padding='causal', dropout_rate=dropout)(x)

    # Fully connected layers after TCN
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout)(x)
    outputs = Dense(output_len)(x)

    model = Model([input_dynamic], outputs)
    return model


def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    """ Evaluate model performance on test and train sets """
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


# Dataset preparation
print("Start building dataset")
data_path = r'YOUR_DATA_PATH'
df = pd.read_csv(data_path + '/XiAn.csv')
target_index = 19

# Sliding window settings
M = 40  # input window size
N = 5  # prediction window size

x_data = []
y_data = []

# Generate samples by sliding window
for i in range(len(df) - M - N + 1):
    input_window = df.iloc[i:i + M].values
    output_window = df.iloc[i + M:i + M + N, target_index].values
    x_data.append(input_window)
    y_data.append(output_window)

x_data = np.array(x_data)
y_data = np.array(y_data)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=20, shuffle=True
)

# Normalization
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x_train_all = np.concatenate(x_train, axis=0)
y_train_all = np.concatenate(y_train, axis=0)

scaler_x.fit(x_train_all)
scaler_y.fit(y_train_all.reshape(-1, 1))

# Normalize train set
for i in range(len(x_train)):
    x_train[i] = scaler_x.transform(x_train[i])
for i in range(len(y_train)):
    y_train[i] = scaler_y.transform(y_train[i].reshape(-1, 1)).flatten()

# Normalize test set
for i in range(len(x_test)):
    x_test[i] = scaler_x.transform(x_test[i])
for i in range(len(y_test)):
    y_test[i] = scaler_y.transform(y_test[i].reshape(-1, 1)).flatten()

# Output sample sizes
print(f"Total train samples: {len(x_train)}")
print(f"Total test samples: {len(x_test)}")

# Model training and evaluation
results_combined = {"all_y_true": [], "all_y_pred": [], "loss": [], "val_loss": []}
evaluation_results = {}

model = model_dynamic(input_len_dynamic=x_train.shape[2], timestep_dynamic=x_train.shape[1],
                      output_len=y_train.shape[1])
model.compile(loss="mse", optimizer=Adam(learning_rate=0.001))

history = model.fit(x_train, y_train, epochs=100, batch_size=128, validation_data=(x_test, y_test), verbose=1)

# Predict on test set
y_pred = model.predict(x_test)
y_pred_inverse = scaler_y.inverse_transform(y_pred)
y_test_inverse = scaler_y.inverse_transform(y_test)

# Predict on train set
y_pred_train = model.predict(x_train)
y_pred_train_inverse = scaler_y.inverse_transform(y_pred_train)
y_train_inverse = scaler_y.inverse_transform(y_train)

# Record results
results_combined["all_y_true"].extend(y_test_inverse.flatten())
results_combined["all_y_pred"].extend(y_pred_inverse.flatten())
results_combined["loss"].extend(history.history['loss'])
results_combined["val_loss"].extend(history.history['val_loss'])

# Evaluation
result = Evaluate(
    y_p_test=y_pred_inverse.flatten(),
    y_t_test=y_test_inverse.flatten(),
    y_p_train=y_pred_train_inverse.flatten(),
    y_t_train=y_train_inverse.flatten()
)

# Output evaluation results
print("Model evaluation result:", result)
