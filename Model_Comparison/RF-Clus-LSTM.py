import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 1. Load dataset
print("Loading data...")
data_path = 'YOUR_DATA_PATH/XiAn.csv'  # Replace with actual path
df = pd.read_csv(data_path)
target_index = 19  # Target column index

# 2. Feature selection using Random Forest
print("Running Random Forest for feature selection...")
X_rf = df.drop(df.columns[target_index], axis=1).values
y_rf = df.iloc[:, target_index].values
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_rf, y_rf)
importances = rf.feature_importances_
selected_indices = np.argsort(importances)[-15:]  # Select top 15 important features
df_selected = df.iloc[:, selected_indices]

# 3. Add clustering label using KMeans
print("Clustering with KMeans...")
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(df_selected)
df_selected['Cluster'] = cluster_labels  # Append clustering results

# 4. Create sequences using sliding window
M, N = 40, 5  # Input and output sequence lengths
x_data, y_data = [], []
values = df_selected.values
target_series = df.iloc[:, target_index].values

for i in range(len(values) - M - N + 1):
    x_window = values[i:i + M]
    y_window = target_series[i + M:i + M + N]
    x_data.append(x_window)
    y_data.append(y_window)

x_data = np.array(x_data)
y_data = np.array(y_data)

# 5. Normalize data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_flat = x_data.reshape(-1, x_data.shape[2])
scaler_x.fit(x_flat)
x_data = np.array([scaler_x.transform(x) for x in x_data])
scaler_y.fit(y_data.reshape(-1, 1))
y_data = scaler_y.transform(y_data)

# 6. Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, shuffle=True
)

# 7. Define LSTM model
def build_lstm(input_shape, output_len):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(output_len))
    model.compile(optimizer='adam', loss='mse')
    return model

print("Training LSTM model...")
model = build_lstm(input_shape=(M, x_data.shape[2]), output_len=N)
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# 8. Evaluate the model
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)

y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

mae, rmse, r2 = evaluate(y_test_inv, y_pred_test_inv)
print(f"\nRF-Clus-LSTM Evaluation:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# 9. Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv[:100], label='True')
plt.plot(y_pred_test_inv[:100], label='Predicted')
plt.title("RF-Clus-LSTM: True vs Predicted")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
