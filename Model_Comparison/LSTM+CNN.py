import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import Input, Reshape, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import optuna
from keras.initializers import TruncatedNormal


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout):
    """ Transformer Encoder module """
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


def model_dynamic(input_len_dynamic, timestep_dynamic, output_len, head_size=16, num_heads=2,
                  num_transformer_blocks=2, ff_dim=16, dropout=0.1, lstm_units=64):
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

    model = Model([input_dynamic], outputs)
    return model


def visualize_results(history, y_true, y_pred):
    """ Visualize model performance """
    residuals = y_true - y_pred

    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    plt.title('Loss Curve');
    plt.xlabel('Epochs');
    plt.ylabel('MSE Loss');
    plt.legend();
    plt.grid();
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y_true[:100], label='True Values', color='blue')
    plt.plot(y_pred[:100], label='Predicted Values', color='orange', linestyle='--')
    plt.title('True vs Predicted');
    plt.xlabel('Sample Index');
    plt.ylabel('Value');
    plt.legend();
    plt.grid();
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals.flatten(), bins=30, color='purple', alpha=0.7)
    plt.title('Residual Distribution');
    plt.xlabel('Residuals');
    plt.ylabel('Frequency');
    plt.grid();
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, residuals, alpha=0.6, color='orange')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Residuals vs True Values');
    plt.xlabel('True Values');
    plt.ylabel('Residuals');
    plt.grid();
    plt.show()


def Evaluate(y_p_test, y_t_test, y_p_train, y_t_train):
    """ Evaluate model performance """
    Result = []
    for y_p, y_t in [(y_p_test, y_t_test), (y_p_train, y_t_train)]:
        mask = np.abs(y_t) > 1e-6
        y_p = y_p[mask];
        y_t = y_t[mask]
        SSR = np.sum((y_p - y_t) ** 2)
        SST = np.sum((y_t - np.mean(y_t)) ** 2)
        r2 = 1 - SSR / SST
        mae = np.mean(np.abs(y_p - y_t))
        rmse = np.sqrt(np.mean((y_p - y_t) ** 2))
        mape = np.mean(np.abs((y_p - y_t) / y_t)) * 100
        Result.extend([mae, rmse, r2, mape])
    return np.array(Result).reshape(1, -1)


def bayesian_optimization(x_train_mode, y_train_mode, x_test_mode, y_test_mode):
    """ Bayesian hyperparameter optimization """

    def objective(trial):
        lstm_units = trial.suggest_int("lstm_units", 32, 128)
        dropout = trial.suggest_float("dropout", 0.1, 0.5)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2)
        epoch = trial.suggest_int("epoch", 10, 50)

        model = model_dynamic(
            input_len_dynamic=x_train_mode.shape[2],
            timestep_dynamic=x_train_mode.shape[1],
            output_len=y_train_mode.shape[1],
            dropout=dropout, lstm_units=lstm_units
        )

        model.compile(loss="mse", optimizer=Adam(learning_rate=learning_rate))
        model.fit(x_train_mode, y_train_mode, epochs=epoch, batch_size=128, verbose=0)
        return model.evaluate(x_test_mode, y_test_mode, verbose=0)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    return study.best_params


# Load and preprocess data
print("Start building dataset")
data_path = r'YOUR_DATA_PATH'
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

# Model training and evaluation
results_combined = {"all_y_true": [], "all_y_pred": [], "loss": [], "val_loss": []}
evaluation_results = {}

model = model_dynamic(
    input_len_dynamic=state_train_data[0].shape[2],
    timestep_dynamic=state_train_data[0].shape[1],
    output_len=state_train_data[1].shape[1],
    dropout=0.2, lstm_units=64
)

best_params = bayesian_optimization(state_train_data[0], state_train_data[1], state_test_data[0], state_test_data[1])

model.compile(loss="mse", optimizer=Adam(learning_rate=best_params['learning_rate']))
history = model.fit(state_train_data[0], state_train_data[1],
                    epochs=best_params['epoch'], batch_size=128,
                    validation_data=(state_test_data[0], state_test_data[1]), verbose=1)

# Predictions and evaluation
y_pred_mode = model.predict(state_test_data[0])
y_pred_mode_inverse = scaler_y.inverse_transform(y_pred_mode)
y_test_mode_inverse = scaler_y.inverse_transform(state_test_data[1])

y_pred_train_mode = model.predict(state_train_data[0])
y_pred_train_mode_inverse = scaler_y.inverse_transform(y_pred_train_mode)
y_train_mode_inverse = scaler_y.inverse_transform(state_train_data[1])

results_combined["all_y_true"].extend(y_test_mode_inverse.flatten())
results_combined["all_y_pred"].extend(y_pred_mode_inverse.flatten())
results_combined["loss"].extend(history.history['loss'])
results_combined["val_loss"].extend(history.history['val_loss'])

result = Evaluate(y_pred_mode_inverse.flatten(), y_test_mode_inverse.flatten(),
                  y_pred_train_mode_inverse.flatten(), y_train_mode_inverse.flatten())
evaluation_results["Overall"] = result.flatten()

evaluation_df = pd.DataFrame.from_dict(evaluation_results, orient='index',
                                       columns=['Test MAE', 'Test RMSE', 'Test R2', 'Test MAPE',
                                                'Train MAE', 'Train RMSE', 'Train R2', 'Train MAPE'])

print("\nEvaluation Results:")
print(evaluation_df)
print("\nAverage of Metrics:")
print(evaluation_df.mean(axis=0))


def visualize_results_combined(results_combined):
    """ Visualize final results """
    residuals = results_combined["all_y_true"] - results_combined["all_y_pred"]

    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["loss"], label='Training Loss')
    plt.plot(results_combined["val_loss"], label='Validation Loss')
    plt.title('Loss Curve');
    plt.xlabel('Epochs');
    plt.ylabel('Loss');
    plt.legend();
    plt.grid();
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(results_combined["all_y_true"][:100], label='True')
    plt.plot(results_combined["all_y_pred"][:100], label='Predicted', linestyle='--')
    plt.title('True vs Predicted');
    plt.xlabel('Index');
    plt.ylabel('Value');
    plt.legend();
    plt.grid();
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=30, color='purple', alpha=0.7)
    plt.title('Residual Distribution');
    plt.grid();
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.scatter(results_combined["all_y_true"], residuals, alpha=0.6, color='orange')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals vs True');
    plt.xlabel('True');
    plt.ylabel('Residuals');
    plt.grid();
    plt.show()


results_combined["all_y_true"] = np.array(results_combined["all_y_true"])
results_combined["all_y_pred"] = np.array(results_combined["all_y_pred"])
visualize_results_combined(results_combined)
