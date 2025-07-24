import numpy as np
import pandas as pd
from typing import Dict, Any


class ModelEvaluator:
    """Handles model evaluation and performance metrics"""

    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate various regression metrics"""
        non_zero_mask = np.abs(y_true) > 1e-6
        y_true = y_true[non_zero_mask]
        y_pred = y_pred[non_zero_mask]

        metrics = {}
        residuals = y_true - y_pred

        # Basic metrics
        metrics['MAE'] = np.mean(np.abs(residuals))
        metrics['RMSE'] = np.sqrt(np.mean(residuals ** 2))
        metrics['MAPE'] = np.mean(np.abs(residuals / y_true)) * 100

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        metrics['R2'] = 1 - (ss_res / ss_tot)

        return metrics

    def evaluate_model(self, model: Any, x_test: np.ndarray, y_test: np.ndarray,
                       x_train: np.ndarray, y_train: np.ndarray,
                       scaler_y: Any) -> Dict[str, Dict[str, float]]:
        """Evaluate model on both training and test sets"""
        # Test set predictions
        y_pred_test = model.predict(x_test)
        y_pred_test_inv = scaler_y.inverse_transform(y_pred_test)
        y_test_inv = scaler_y.inverse_transform(y_test)

        # Training set predictions
        y_pred_train = model.predict(x_train)
        y_pred_train_inv = scaler_y.inverse_transform(y_pred_train)
        y_train_inv = scaler_y.inverse_transform(y_train)

        # Calculate metrics
        test_metrics = self.calculate_metrics(y_test_inv.flatten(), y_pred_test_inv.flatten())
        train_metrics = self.calculate_metrics(y_train_inv.flatten(), y_pred_train_inv.flatten())

        return {
            'test': test_metrics,
            'train': train_metrics
        }

    def aggregate_results(self, all_results: Dict[Any, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
        """Combine results from all flight modes into a dataframe"""
        results_list = []

        for mode, metrics in all_results.items():
            row = {'Flight Mode': mode}
            for dataset in ['train', 'test']:
                for metric, value in metrics[dataset].items():
                    row[f"{dataset}_{metric}"] = value
            results_list.append(row)

        return pd.DataFrame(results_list)