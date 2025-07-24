from dataclasses import dataclass
from typing import Tuple

@dataclass
class ModelParams:
    """Configuration for model architecture parameters"""
    transformer_blocks: int = 2
    head_size: int = 16
    num_heads: int = 2
    ff_dim: int = 16
    lstm_units: Tuple[int, int] = (32, 128)  # Range for hyperparameter tuning
    dropout_range: Tuple[float, float] = (0.1, 0.5)
    dense_units: Tuple[int, int, int] = (64, 32, 1)  # Units for each dense layer

@dataclass
class TrainingParams:
    """Configuration for training parameters"""
    learning_rate_range: Tuple[float, float] = (1e-4, 1e-2)
    batch_size: int = 128
    epochs_range: Tuple[int, int] = (10, 50)
    test_size: float = 0.2
    random_state: int = 42
    n_trials: int = 25  # Number of optimization trials

@dataclass
class DataParams:
    """Configuration for data processing"""
    window_size: int = 40
    prediction_window: int = 8
    target_index: int = 19
    feature_columns: int = 20  # Number of feature columns to use
    scaling_method: str = 'minmax'  # 'minmax', 'standard', or 'robust'