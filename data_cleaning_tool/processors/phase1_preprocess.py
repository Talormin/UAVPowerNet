import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List


class DataPreprocessor:
    """Initial data preparation and quality checks"""

    @staticmethod
    def detect_anomalous_values(df: pd.DataFrame,
                                sensitivity: float = 3.5) -> Dict[str, List[int]]:
        """
        Identify statistical outliers using modified Z-score method

        Args:
            df: Input DataFrame
            sensitivity: Threshold for outlier detection

        Returns:
            Dictionary of outlier indices per column
        """
        outliers = {}
        median_abs_dev = lambda x: np.median(np.abs(x - np.median(x)))

        for col in df.select_dtypes(include=np.number).columns:
            median = np.median(df[col])
            mad = median_abs_dev(df[col])
            modified_z = 0.6745 * (df[col] - median) / mad
            outliers[col] = np.where(np.abs(modified_z) > sensitivity)[0].tolist()

        return outliers

    @staticmethod
    def normalize_timestamps(df: pd.DataFrame,
                             time_col: str = 'timestamp') -> pd.DataFrame:
        """
        Standardize timestamp formatting and frequency

        Args:
            df: Input DataFrame containing time series data
            time_col: Name of timestamp column

        Returns:
            DataFrame with normalized timestamps
        """
        df = df.copy()
        df[time_col] = pd.to_datetime(df[time_col])
        df.set_index(time_col, inplace=True)
        return df.asfreq('1S').reset_index()