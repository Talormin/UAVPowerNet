import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Dataset preparation
print("Start building dataset")
data_path = r'YOUR_DATA_PATH'
df = pd.read_csv(data_path + '/XiAn.csv')
target_index = 19

# Sliding window settings
M = 40  # input window length
N = 5   # prediction window length

x_data = []
y_data = []

# Generate samples using sliding window
for i in range(len(df) - M - N + 1):
    input_window = df.iloc[i:i + M].values
    output_window = df.iloc[i + M:i + M + N, target_index].values
    x_data.append(input_window)
    y_data.append(output_window)

x_data = np.array(x_data)
y_data = np.array(y_data)

# Train/test split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=20, shuffle=True
)

# Global normalization
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
print(f"Total training samples: {len(x_train)}")
print(f"Total test samples: {len(x_test)}")

# Flatten input data for Random Forest
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Use only the first step of the target sequence
y_train_flat = y_train[:, 0]
y_test_flat = y_test[:, 0]

# Train Random Forest model
print("Training Random Forest model...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(x_train_flat, y_train_flat)

# Predict on test set
y_pred = rf_model.predict(x_test_flat)
y_pred_inverse = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test_inverse = scaler_y.inverse_transform(y_test_flat.reshape(-1, 1))

# Predict on train set
y_pred_train = rf_model.predict(x_train_flat)
y_pred_train_inverse = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
y_train_inverse = scaler_y.inverse_transform(y_train_flat.reshape(-1, 1))

# Evaluation function
def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    """ Evaluate model performance """
    Result = []

    # Test set evaluation
    y_p = y_p_test.flatten()
    y_t = y_t_test.flatten()
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

    # Train set evaluation
    y_p = y_p_train.flatten()
    y_t = y_t_train.flatten()
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

# Evaluate model
evaluation_results = Evaluate(
    y_p_test=y_pred_inverse,
    y_t_test=y_test_inverse,
    y_p_train=y_pred_train_inverse,
    y_t_train=y_train_inverse
)

# Output evaluation results
print("\nModel evaluation results:")
print("Test MAE:", evaluation_results[0][0])
print("Test RMSE:", evaluation_results[0][1])
print("Test R2:", evaluation_results[0][2])
print("Test MAPE:", evaluation_results[0][3])
print("Train MAE:", evaluation_results[0][4])
print("Train RMSE:", evaluation_results[0][5])
print("Train R2:", evaluation_results[0][6])
print("Train MAPE:", evaluation_results[0][7])
