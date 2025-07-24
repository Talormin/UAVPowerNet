import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train, scaler_y):
    """Evaluate model performance"""
    Result = []

    y_p_test = scaler_y.inverse_transform(y_p_test.reshape(-1, 1)).flatten()
    y_t_test = scaler_y.inverse_transform(y_t_test.reshape(-1, 1)).flatten()

    y_p_train = scaler_y.inverse_transform(y_p_train.reshape(-1, 1)).flatten()
    y_t_train = scaler_y.inverse_transform(y_t_train.reshape(-1, 1)).flatten()

    # Test Set
    non_zero_mask = np.abs(y_t_test) > 1e-6
    y_p = y_p_test[non_zero_mask]
    y_t = y_t_test[non_zero_mask]

    SSR = np.sum((y_p - y_t) ** 2)
    SST = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
    mape = np.mean(np.abs((y_p - y_t) / y_t)) * 100
    Result.extend([mae, rmse, r2, mape])

    # Train Set
    non_zero_mask = np.abs(y_t_train) > 1e-6
    y_p = y_p_train[non_zero_mask]
    y_t = y_t_train[non_zero_mask]

    SSR = np.sum((y_p - y_t) ** 2)
    SST = np.sum((y_t - np.mean(y_t)) ** 2)
    r2 = 1 - SSR / SST
    mae = np.mean(np.abs(y_p - y_t))
    rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
    mape = np.mean(np.abs((y_p - y_t) / y_t)) * 100
    Result.extend([mae, rmse, r2, mape])

    return np.array(Result).reshape(1, -1)

# Load and process dataset
print("Starting to construct dataset")
data_path = r'YOUR_DATA_PATH'  # <-- Replace this with your actual path
df = pd.read_csv(data_path + '/27.csv')
target_index = 19

Step = 3
dd = df.values
x_data, y_data = [], []

for i in range(0, len(dd) - Step + 1, Step):
    group = dd[i:i + Step, :]
    x_data.append(group)
    y_data.append(group[:, target_index])

x_data = np.array(x_data)
y_data = np.array(y_data)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=20, shuffle=True)

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_train_all = np.concatenate(x_train, axis=0)
y_train_all = np.concatenate(y_train, axis=0)

scaler_x.fit(x_train_all)
scaler_y.fit(y_train_all.reshape(-1, 1))

state_train_data = ([], [])
state_test_data = ([], [])

for x_group, y_group in zip(x_train, y_train):
    state_train_data[0].append(scaler_x.transform(x_group))
    state_train_data[1].append(scaler_y.transform(y_group.reshape(-1, 1)).flatten())

for x_group, y_group in zip(x_test, y_test):
    state_test_data[0].append(scaler_x.transform(x_group))
    state_test_data[1].append(scaler_y.transform(y_group.reshape(-1, 1)).flatten())

state_train_data = (np.array(state_train_data[0]), np.array(state_train_data[1]))
state_test_data = (np.array(state_test_data[0]), np.array(state_test_data[1]))

print(f"Total training samples: {state_train_data[0].shape[0]}")

results_combined = {"all_y_true": [], "all_y_pred": [], "loss": [], "val_loss": []}
evaluation_results = {}

def bayesian_optimization(x_train_mode, y_train_mode, x_test_mode, y_test_mode):
    """Bayesian optimization for hyperparameter tuning"""
    def objective(trial):
        model = xgb.XGBRegressor(
            n_estimators=trial.suggest_int("n_estimators", 100, 1000),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-2),
            gamma=trial.suggest_float("gamma", 0, 1)
        )
        model.fit(x_train_mode.reshape(x_train_mode.shape[0], -1), y_train_mode)
        preds = model.predict(x_test_mode.reshape(x_test_mode.shape[0], -1))
        return mean_squared_error(y_test_mode, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8)
    return study.best_params

best_params = bayesian_optimization(*state_train_data, *state_test_data)

model = xgb.XGBRegressor(**best_params)
model.fit(state_train_data[0].reshape(state_train_data[0].shape[0], -1), state_train_data[1])

y_pred_test = model.predict(state_test_data[0].reshape(state_test_data[0].shape[0], -1))
y_pred_test_inverse = scaler_y.inverse_transform(y_pred_test.reshape(-1, 1))
y_test_inverse = scaler_y.inverse_transform(state_test_data[1])

y_pred_train = model.predict(state_train_data[0].reshape(state_train_data[0].shape[0], -1))
y_pred_train_inverse = scaler_y.inverse_transform(y_pred_train.reshape(-1, 1))
y_train_inverse = scaler_y.inverse_transform(state_train_data[1])

results_combined["all_y_true"].extend(y_test_inverse.flatten())
results_combined["all_y_pred"].extend(y_pred_test_inverse.flatten())

result = Evaluate(y_pred_test_inverse.flatten(), y_test_inverse.flatten(),
                  y_pred_train_inverse.flatten(), y_train_inverse.flatten(), scaler_y)
evaluation_results["Overall"] = result.flatten()

print("Evaluation result:", result)

evaluation_df = pd.DataFrame.from_dict(
    evaluation_results,
    orient='index',
    columns=['Test MAE', 'Test RMSE', 'Test R2', 'Test MAPE', 'Train MAE', 'Train RMSE', 'Train R2', 'Train MAPE']
)

print("\nEvaluation results of all models:")
print(evaluation_df)

print("\nAverage metrics across all models:")
print(evaluation_df.mean())

def visualize_results_combined(results_combined):
    residuals = results_combined["all_y_true"] - results_combined["all_y_pred"]

    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["loss"], label='Training Loss')
    plt.plot(results_combined["val_loss"], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["all_y_true"][:100], label='True')
    plt.plot(results_combined["all_y_pred"][:100], label='Predicted')
    plt.title('True vs Predicted')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals.flatten(), bins=30)
    plt.title('Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Count')
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(results_combined["all_y_true"], residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs True')
    plt.xlabel('True')
    plt.ylabel('Residual')
    plt.grid()
    plt.show()

results_combined["all_y_true"] = np.array(results_combined["all_y_true"])
results_combined["all_y_pred"] = np.array(results_combined["all_y_pred"])
visualize_results_combined(results_combined)
