import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, medfilt
from typing import List, Dict


class DataCleaningEngine:
    """Core data processing and transformation logic"""

    @staticmethod
    def apply_adaptive_filter(df: pd.DataFrame,
                              columns: List[str],
                              base_window: int = 7) -> pd.DataFrame:
        """
        Apply context-aware filtering based on data characteristics

        Args:
            df: Input DataFrame
            columns: Columns to process
            base_window: Initial filter window size

        Returns:
            Filtered DataFrame
        """
        df_clean = df.copy()
        for col in columns:
            noise_level = np.std(df[col].diff().dropna())
            dynamic_window = base_window + int(noise_level * 10)
            dynamic_window = max(3, min(dynamic_window, 21))

            if len(df) > dynamic_window:
                df_clean[col] = savgol_filter(
                    df[col],
                    window_length=dynamic_window,
                    polyorder=2,
                    mode='nearest'
                )
            else:
                df_clean[col] = medfilt(df[col], kernel_size=3)

        return df_clean

    @staticmethod
    def handle_missing_values(df: pd.DataFrame,
                              strategy: str = 'contextual') -> pd.DataFrame:
        """
        Advanced missing data imputation

        Args:
            df: Input DataFrame with missing values
            strategy: Imputation methodology

        Returns:
            DataFrame with imputed values
        """
        if strategy == 'contextual':
            # Use surrounding values for interpolation
            return df.interpolate(method='time', limit_direction='both')
        else:
            # Fallback to linear interpolation
            return df.interpolate()