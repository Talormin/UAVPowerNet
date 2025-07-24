import pandas as pd
from scipy.interpolate import interp1d
from typing import Dict, Tuple
import numpy as np


class DataProcessor:
    def __init__(self, base_timestamp: pd.Series):
        """Initialize with base timestamp for interpolation"""
        self.base_timestamp = base_timestamp

    def interpolate_data(self, df: pd.DataFrame, columns: list) -> Dict[str, np.ndarray]:
        """
        Interpolate specified columns to match base timestamp

        Args:
            df: Source DataFrame containing data to interpolate
            columns: List of column names to interpolate

        Returns:
            Dictionary with interpolated values for each column
        """
        results = {}
        timestamp = df["timestamp"].copy()
        timestamp.iloc[0] = self.base_timestamp.iloc[0]
        timestamp.iloc[-1] = self.base_timestamp.iloc[-1]

        for col in columns:
            f = interp1d(timestamp, df[col], kind='linear', bounds_error=False)
            results[col] = f(self.base_timestamp)

        return results

    def calculate_power(self, voltage: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Calculate power from voltage and current measurements"""
        return voltage * current