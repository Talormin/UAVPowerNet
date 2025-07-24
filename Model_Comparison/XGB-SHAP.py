import pandas as pd
import numpy as np
import xgboost as xgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. Load and preprocess data
print("Loading dataset...")
data_path = 'YOUR_DATA_PATH/XiAn.csv'  # <-- Replace with your path
df = pd.read_csv(data_path)
target_index = 19  # Column index of the target variable

# Sliding window parameters
M, N = 40, 5  # Input window length, prediction horizon

x_data, y_data = [], []
for i in range(len(df) - M - N + 1):
    x_window = df.iloc[i:i + M].values.flatten()
    y_window = df.iloc[i + M:i + M + N, target_index].values
    x_data.append(x_window)
    y_data.append(y_window)

x_data = np.array(x_data)
y_data = np.array(y_data)

# Normalize features and targets
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
x_data = scaler_x.fit_transform(x_data)
y_data = scaler_y.fit_transform(y_data)

# Split into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 2. Train XGBoost model
print("Training XGBoost model...")
model = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.2,
    objective='reg:squarederror',
    random_state=42
)
model.fit(x_train, y_train)

# 3. Evaluate model
y_pred = model.predict(x_test)
y_pred_inv = scaler_y.inverse_transform(y_pred)
y_test_inv = scaler_y.inverse_transform(y_test)

mae = mean_absolute_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
r2 = r2_score(y_test_inv, y_pred_inv)

print(f"XGB-SHAP Evaluation:\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# 4. SHAP explanation
print("Computing SHAP values...")
explainer = shap.Explainer(model)
shap_values = explainer(x_test)

# Global feature importance plot
shap.summary_plot(shap_values, x_test, feature_names=[f'f{i}' for i in range(x_test.shape[1])])

# Local explanation example (first test sample)
shap.plots.waterfall(shap_values[0])
