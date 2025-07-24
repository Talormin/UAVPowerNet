import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from boa import BOA  # 假设你已经安装了 BOA 库

# 数据集构建和处理
print("开始构建数据集")
data_path = r'C:\\Users\\LWJ\\Desktop\\Dataset'
df = pd.read_csv(data_path + '\\XiAn.csv')
target_index = 19

# 滑动窗口的设置
M = 40  # 输入窗口长度
N = 5   # 预测窗口长度

x_data = []
y_data = []

# 按滑动窗口生成数据
for i in range(len(df) - M - N + 1):
    input_window = df.iloc[i:i + M].values
    output_window = df.iloc[i + M:i + M + N, target_index].values

    x_data.append(input_window)
    y_data.append(output_window)

x_data = np.array(x_data)
y_data = np.array(y_data)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_data.reshape(x_data.shape[0], -1),  # XGBoost 需要二维数据 (样本数, 特征数)
    y_data, 
    test_size=0.2, random_state=20, shuffle=True
)

# 归一化处理
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

# 目标函数：BOA 进行 XGBoost 超参数优化
def objective(params):
    params = {
        'max_depth': int(params[0]),
        'learning_rate': params[1],
        'n_estimators': int(params[2]),
        'subsample': params[3],
        'colsample_bytree': params[4],
        'gamma': params[5]
    }
    
    model = xgb.XGBRegressor(**params, objective='reg:squarederror')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    return mean_squared_error(y_test, y_pred)  # 目标是最小化 MSE

# BOA 参数边界
bounds = [
    (3, 10),        # max_depth
    (0.01, 0.3),    # learning_rate
    (50, 500),      # n_estimators
    (0.5, 1.0),     # subsample
    (0.5, 1.0),     # colsample_bytree
    (0, 5)          # gamma
]

# 运行 BOA 进行优化
boa = BOA(obj_func=objective, bounds=bounds, num_agents=10, max_iter=20)
best_params = boa.run()

# 训练最终 XGBoost 模型
best_params = {
    'max_depth': int(best_params[0]),
    'learning_rate': best_params[1],
    'n_estimators': int(best_params[2]),
    'subsample': best_params[3],
    'colsample_bytree': best_params[4],
    'gamma': best_params[5]
}

final_model = xgb.XGBRegressor(**best_params, objective='reg:squarederror')
final_model.fit(x_train, y_train)
y_pred = final_model.predict(x_test)

# 反归一化预测值
y_pred = scaler_y.inverse_transform(y_pred.reshape(-1, 1))
y_test = scaler_y.inverse_transform(y_test)

# 计算评估指标
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"BOA+XGBoost 评估结果：\nMAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
