import optuna
from tensorflow.keras.optimizers import Adam
from typing import Dict, Any
from config.hyperparameters import ModelParams, TrainingParams
from core.model_architecture import FlightModelBuilder


class ModelTrainer:
    """Handles model training and hyperparameter optimization"""

    def __init__(self, model_params: ModelParams, train_params: TrainingParams):
        self.model_params = model_params
        self.train_params = train_params
        self.model_builder = FlightModelBuilder()

    def optimize_hyperparameters(self, x_train: np.ndarray, y_train: np.ndarray,
                                 x_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Perform Bayesian optimization to find best hyperparameters"""

        def objective(trial):
            params = {
                'lstm_units': trial.suggest_int("lstm_units", *self.model_params.lstm_units),
                'dropout': trial.suggest_float("dropout", *self.model_params.dropout_range),
                'learning_rate': trial.suggest_float("learning_rate", *self.train_params.learning_rate_range),
                'epochs': trial.suggest_int("epochs", *self.train_params.epochs_range)
            }

            model = self.model_builder.build_model(
                input_shape=(x_train.shape[1], x_train.shape[2]),
                output_len=y_train.shape[1],
                params=self.model_params
            )

            model.compile(
                loss="mse",
                optimizer=Adam(learning_rate=params['learning_rate'])
            )

            history = model.fit(
                x_train, y_train,
                epochs=params['epochs'],
                batch_size=self.train_params.batch_size,
                validation_data=(x_val, y_val),
                verbose=0
            )

            return history.history['val_loss'][-1]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=self.train_params.n_trials)
        return study.best_params

    def train_model(self, x_train: np.ndarray, y_train: np.ndarray,
                    x_val: np.ndarray, y_val: np.ndarray,
                    best_params: Dict[str, Any]) -> Any:
        """Train final model with optimized hyperparameters"""
        # Create model with best params
        final_model = self.model_builder.build_model(
            input_shape=(x_train.shape[1], x_train.shape[2]),
            output_len=y_train.shape[1],
            params=self.model_params
        )

        final_model.compile(
            loss="mse",
            optimizer=Adam(learning_rate=best_params['learning_rate'])
        )

        # Train model
        history = final_model.fit(
            x_train, y_train,
            epochs=best_params['epochs'],
            batch_size=self.train_params.batch_size,
            validation_data=(x_val, y_val),
            verbose=1
        )

        return final_model, history