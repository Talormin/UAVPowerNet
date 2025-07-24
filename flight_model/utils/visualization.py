import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any


class ResultVisualizer:
    """Handles visualization of model results and metrics"""

    @staticmethod
    def plot_loss_curves(history: Any) -> None:
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
        plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        plt.title('Model Training Loss Curves', fontsize=14)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 100) -> None:
        """Plot true vs predicted values"""
        plt.figure(figsize=(12, 6))
        plt.plot(y_true[:n_samples], label='True Values', linewidth=2)
        plt.plot(y_pred[:n_samples], label='Predicted Values', linestyle='--', linewidth=2)
        plt.title('True vs Predicted Values', fontsize=14)
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Value', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Plot residual analysis"""
        residuals = y_true - y_pred

        plt.figure(figsize=(15, 5))

        # Residual distribution
        plt.subplot(1, 3, 1)
        plt.hist(residuals, bins=30, color='blue', alpha=0.7)
        plt.title('Residual Distribution', fontsize=12)
        plt.xlabel('Residual Value', fontsize=10)
        plt.ylabel('Frequency', fontsize=10)
        plt.grid(True)

        # Residuals vs Predicted
        plt.subplot(1, 3, 2)
        plt.scatter(y_pred, residuals, alpha=0.6, color='green')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residuals vs Predicted', fontsize=12)
        plt.xlabel('Predicted Values', fontsize=10)
        plt.ylabel('Residuals', fontsize=10)
        plt.grid(True)

        # Residuals over time
        plt.subplot(1, 3, 3)
        plt.plot(residuals[:500], color='purple', alpha=0.7)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residuals over Time', fontsize=12)
        plt.xlabel('Sample Index', fontsize=10)
        plt.ylabel('Residual Value', fontsize=10)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_combined_results(self, results: Dict[str, np.ndarray]) -> None:
        """Visualize combined results from all flight modes"""
        # Loss curves
        self.plot_loss_curves({'history': {
            'loss': results['loss'],
            'val_loss': results['val_loss']
        }})

        # Predictions
        self.plot_predictions(results['all_y_true'], results['all_y_pred'])

        # Residuals
        self.plot_residuals(results['all_y_true'], results['all_y_pred'])