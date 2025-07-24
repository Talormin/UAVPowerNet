import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, Flatten, Add, Activation
from keras.optimizers import Adam

# 1. Load and preprocess the dataset
print("Loading dataset...")
data_path = 'YOUR_DATA_PATH/XiAn.csv'  # Replace with actual path
df = pd.read_csv(data_path)
target_index = 19

# Create sequences using sliding window
M, N = 40, 5  # Input length M, prediction horizon N
x_data, y_data = [], []

for i in range(len(df) - M - N + 1):
    x_window = df.iloc[i:i + M].values
    y_window = df.iloc[i + M:i + M + N, target_index].values
    x_data.append(x_window)
    y_data.append(y_window)

x_data = np.array(x_data)
y_data = np.array(y_data)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42, shuffle=True
)

# Normalize features and targets
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

# 2. Define a TCN block
def tcn_block(x, filters, kernel_size, dilation_rate):
    conv1 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
    drop1 = Dropout(0.2)(conv1)
    conv2 = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(drop1)
    drop2 = Dropout(0.2)(conv2)
    if x.shape[-1] != filters:
        x = Conv1D(filters, 1, padding='same')(x)
    out = Add()([drop2, x])
    return Activation('relu')(out)

# 3. Build the LR-TCN model
def build_lr_tcn_model(input_shape, output_len):
    inputs = Input(shape=input_shape)
    x = tcn_block(inputs, filters=64, kernel_size=3, dilation_rate=1)
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=2)
    x = tcn_block(x, filters=64, kernel_size=3, dilation_rate=4)
    x = Flatten()(x)
    outputs = Dense(output_len)(x)  # Linear regression head
    return Model(inputs, outputs)

model = build_lr_tcn_model(input_shape=(M, x_train.shape[2]), output_len=N)
model.compile(optimizer=Adam(0.001), loss='mse')
model.summary()

# 4. Train the model
history = model.fit(x_train, y_train, epochs=50, batch_size=64, validation_data=(x_test, y_test), verbose=1)

# 5. Predict and inverse-transform
y_pred_test = model.predict(x_test)
y_pred_train = model.predict(x_train)
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, 1))
y_pred_test_inv = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
y_train_inv = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_pred_train_inv = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))

# 6. Evaluation function
def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

mae, rmse, r2 = evaluate(y_test_inv, y_pred_test_inv)
print(f"\nLR-TCN Evaluation Results:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# 7. Visualization
plt.figure(figsize=(10, 5))
plt.plot(y_test_inv[:100], label='True')
plt.plot(y_pred_test_inv[:100], label='Predicted')
plt.legend()
plt.title("LR-TCN: True vs Predicted")
plt.grid()
plt.tight_layout()
plt.show()
