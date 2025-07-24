import pandas as pd
import numpy as np
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.optimizers import Adam
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load and preprocess data
print("Loading and transforming data...")
data_path = 'YOUR_DATA_PATH/XiAn.csv'  # <-- Replace with actual path
df = pd.read_csv(data_path)
target_index = 19

# Apply wavelet denoising to each column
def wavelet_denoise(signal, wavelet='db4', level=2):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    threshold = np.std(coeffs[-level]) * np.sqrt(2 * np.log(len(signal)))
    denoised_coeffs = [pywt.threshold(c, threshold, mode='soft') if i > 0 else c for i, c in enumerate(coeffs)]
    return pywt.waverec(denoised_coeffs, wavelet)[:len(signal)]

df_wavelet = df.copy()
for col in df.columns:
    df_wavelet[col] = wavelet_denoise(df[col])

# Sliding window setup
M, N = 40, 5
x_data, y_data = [], []
for i in range(len(df_wavelet) - M - N + 1):
    x_data.append(df_wavelet.iloc[i:i + M].values)
    y_data.append(df_wavelet.iloc[i + M:i + M + N, target_index].values)
x_data = np.array(x_data)
y_data = np.array(y_data)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42, shuffle=True)

# Normalize
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_train_reshaped = x_train.reshape(-1, x_train.shape[2])
x_test_reshaped = x_test.reshape(-1, x_test.shape[2])
scaler_x.fit(x_train_reshaped)
scaler_y.fit(y_train.reshape(-1, 1))
x_train = np.array([scaler_x.transform(x) for x in x_train])
x_test = np.array([scaler_x.transform(x) for x in x_test])
y_train = np.array([scaler_y.transform(y.reshape(-1, 1)).flatten() for y in y_train])
y_test = np.array([scaler_y.transform(y.reshape(-1, 1)).flatten() for y in y_test])

# 2. LSTM feature extraction model
def create_lstm_feature_model(input_shape, output_len):
    inp = Input(shape=input_shape)
    x = LSTM(64, return_sequences=False)(inp)
    x = Dense(64, activation='relu')(x)
    out = Dense(output_len)(x)
    return Model(inp, out)

lstm_model = create_lstm_feature_model((M, x_train.shape[2]), N)
lstm_model.compile(optimizer=Adam(0.001), loss='mse')
lstm_model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# Extract features from LSTM
train_features = lstm_model.predict(x_train)
test_features = lstm_model.predict(x_test)

# 3. Train XGBoost on extracted features
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.05)
xgb_model.fit(train_features, y_train)

# Predict and evaluate
y_pred_train = xgb_model.predict(train_features)
y_pred_test = xgb_model.predict(test_features)

# Inverse transform
y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Evaluation metrics
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

mae, rmse, r2 = evaluate(y_test_inv, y_pred_test_inv)
print(f"\nWav-LSTM-XGBoost Evaluation:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv[:100], label='True', linewidth=2)
plt.plot(y_pred_test_inv[:100], label='Predicted', linestyle='--', linewidth=2)
plt.legend()
plt.title("True vs Predicted (Test Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.grid()
plt.tight_layout()
plt.show()
