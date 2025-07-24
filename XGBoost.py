import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train, scaler_y):
    """ 评估模型训练效果 """
    Result = []

    # 对预测值和真实值进行逆归一化
    y_p_test = scaler_y.inverse_transform(y_p_test.reshape(-1, 1)).flatten()
    y_t_test = scaler_y.inverse_transform(y_t_test.reshape(-1, 1)).flatten()

    y_p_train = scaler_y.inverse_transform(y_p_train.reshape(-1, 1)).flatten()
    y_t_train = scaler_y.inverse_transform(y_t_train.reshape(-1, 1)).flatten()

    # Test Set Evaluation
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

    # Train Set Evaluation
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

# 数据集构建和处理
print("开始构建数据集")
data_path = r'C:\Users\LWJ\Desktop'
df = pd.read_csv(data_path + '\\27.csv')
target_index = 19

Step = 3  # 设定步长
dd = df.values

x_data = []
y_data = []
for i in range(0, len(dd) - Step + 1, Step):
    group = dd[i:i + Step, :]  # 获取滑动窗口数据
    x_data.append(group)  # x 数据
    y_data.append(group[:, target_index])  # y 数据（目标值）

x_data = np.array(x_data)
y_data = np.array(y_data)

# 训练集和测试集划分
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=20, shuffle=True)

# 全局归一化
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))

x_train_all = np.concatenate(x_train, axis=0)
y_train_all = np.concatenate(y_train, axis=0)

scaler_x.fit(x_train_all)
scaler_y.fit(y_train_all.reshape(-1, 1))

state_train_data = ([], [])
state_test_data = ([], [])

# 归一化训练集
for x_group, y_group in zip(x_train, y_train):
    x_group_scaled = scaler_x.transform(x_group)
    y_group_scaled = scaler_y.transform(y_group.reshape(-1, 1)).flatten()
    state_train_data[0].append(x_group_scaled)
    state_train_data[1].append(y_group_scaled)

# 归一化测试集
for x_group, y_group in zip(x_test, y_test):
    x_group_scaled = scaler_x.transform(x_group)
    y_group_scaled = scaler_y.transform(y_group.reshape(-1, 1)).flatten()
    state_test_data[0].append(x_group_scaled)
    state_test_data[1].append(y_group_scaled)

# 转换为 NumPy 数组
state_train_data = (np.array(state_train_data[0]), np.array(state_train_data[1]))
state_test_data = (np.array(state_test_data[0]), np.array(state_test_data[1]))

# 输出训练集总量
total_train_samples = state_train_data[0].shape[0]
print(f"训练集总量: {total_train_samples}")

# XGBoost模型训练
results_combined = {"all_y_true": [], "all_y_pred": [], "loss": [], "val_loss": []}
evaluation_results = {}

# XGBoost模型训练与评估
def bayesian_optimization(x_train_mode, y_train_mode, x_test_mode, y_test_mode):
    """ 贝叶斯优化超参数搜索 """
    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        gamma = trial.suggest_float("gamma", 0, 1)

        model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                  learning_rate=learning_rate, gamma=gamma)
        
        model.fit(x_train_mode.reshape(x_train_mode.shape[0], -1), y_train_mode)
        loss = mean_squared_error(y_test_mode, model.predict(x_test_mode.reshape(x_test_mode.shape[0], -1)))
        return loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=8)  # 调整搜索轮数
    return study.best_params

# 模型训练与评估
best_params = bayesian_optimization(state_train_data[0], state_train_data[1], state_test_data[0], state_test_data[1])
model = xgb.XGBRegressor(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'], 
                         learning_rate=best_params['learning_rate'], gamma=best_params['gamma'])

# 训练模型
model.fit(state_train_data[0].reshape(state_train_data[0].shape[0], -1), state_train_data[1])

# 预测
y_pred_mode = model.predict(state_test_data[0].reshape(state_test_data[0].shape[0], -1))
y_pred_mode_inverse = scaler_y.inverse_transform(y_pred_mode.reshape(-1, 1))
y_test_mode_inverse = scaler_y.inverse_transform(state_test_data[1])

# 训练集预测
y_pred_train_mode = model.predict(state_train_data[0].reshape(state_train_data[0].shape[0], -1))
y_pred_train_mode_inverse = scaler_y.inverse_transform(y_pred_train_mode.reshape(-1, 1))
y_train_mode_inverse = scaler_y.inverse_transform(state_train_data[1])

# 记录每个模型的结果
results_combined["all_y_true"].extend(y_test_mode_inverse.flatten())
results_combined["all_y_pred"].extend(y_pred_mode_inverse.flatten())

# 评估结果
result = Evaluate(y_pred_mode_inverse.flatten(), y_test_mode_inverse.flatten(), y_pred_train_mode_inverse.flatten(), y_train_mode_inverse.flatten(), scaler_y)
evaluation_results["Overall"] = result.flatten()
print(f"评估结果：", result)

evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index', columns=['Test MAE', 'Test RMSE', 'Test R2', 'Test MAPE', 'Train MAE', 'Train RMSE', 'Train R2', 'Train MAPE'])
print("\n各模型评估结果：")
print(evaluation_df)

# 计算各个误差指标的平均值
average_results = evaluation_df.mean(axis=0)
print("\n三个模型各个误差指标的平均值：")
print(average_results)

def visualize_results_combined(results_combined):
    """
    可视化所有模型的总体分析
    """
    residuals = results_combined["all_y_true"] - results_combined["all_y_pred"]

    # 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["loss"], label='Training Loss', color='blue', linewidth=2)
    plt.plot(results_combined["val_loss"], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Overall Loss Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=14)
    plt.ylabel('MSE Loss', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 真实值 vs 预测值
    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["all_y_true"][:100], label='True Values', color='blue', linestyle='-', linewidth=2)
    plt.plot(results_combined["all_y_pred"][:100], label='Predicted Values', color='orange', linestyle='--', linewidth=2)
    plt.title('Overall True vs Predicted', fontsize=16)
    plt.xlabel('Sample Index', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

    # 残差分布图
    plt.figure(figsize=(10, 6))
    plt.hist(residuals.flatten(), bins=30, color='purple', alpha=0.7)
    plt.title('Overall Residual Distribution', fontsize=16)
    plt.xlabel('Residuals', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.grid(True)
    plt.show()

    # 4. 残差 vs 真实值
    plt.figure(figsize=(10, 6))
    plt.scatter(results_combined["all_y_true"], residuals, alpha=0.6, color='orange', label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2) 
    plt.title('Overall Residuals vs True Values', fontsize=16)
    plt.xlabel('True Values', fontsize=14)
    plt.ylabel('Residuals', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

# 统一可视化所有模型的结果
results_combined["all_y_true"] = np.array(results_combined["all_y_true"])
results_combined["all_y_pred"] = np.array(results_combined["all_y_pred"])
visualize_results_combined(results_combined)
