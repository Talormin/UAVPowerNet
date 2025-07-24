import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from config.hyperparameters import DataParams, TrainingParams
from typing import Dict, Tuple, Any


class FlightDataProcessor:
    """Handles all data processing and preparation for flight mode modeling"""

    def __init__(self, config: DataParams, train_config: TrainingParams):
        self.config = config
        self.train_config = train_config
        self.scaler_x = MinMaxScaler()
        self.scaler_y = MinMaxScaler()

    def load_and_prepare_data(self, data_path: str) -> Dict[str, Any]:
        """Main method to load and prepare all data"""
        df = pd.read_csv(data_path)

        # Train flight mode classifier
        rf_classifier = self._train_flight_classifier(df)

        # Generate windowed datasets
        x_data, y_data, modes = self._generate_windowed_data(df, rf_classifier)

        # Split and scale data
        return self._prepare_datasets(x_data, y_data, modes)

    def _train_flight_classifier(self, df: pd.DataFrame) -> RandomForestClassifier:
        """Train random forest classifier for flight mode detection"""
        features = df.iloc[:, :self.config.feature_columns]
        labels = df.iloc[:, self.config.feature_columns]  # Assuming flight mode is next column

        x_train, x_test, y_train, y_test = train_test_split(
            features, labels,
            test_size=self.train_config.test_size,
            random_state=self.train_config.random_state,
            stratify=labels
        )

        model = RandomForestClassifier(n_estimators=100, random_state=self.train_config.random_state)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        print(f"Flight mode classifier trained with accuracy: {accuracy:.4f}")
        return model

    def _generate_windowed_data(self, df: pd.DataFrame, classifier: RandomForestClassifier) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray]:
        """Generate windowed time series data with flight mode labels"""
        x_data, y_data, modes = [], [], []

        for i in range(len(df) - self.config.window_size - self.config.prediction_window + 1):
            input_window = df.iloc[i:i + self.config.window_size, :self.config.feature_columns].values
            output_window = df.iloc[
                            i + self.config.window_size:i + self.config.window_size + self.config.prediction_window,
                            self.config.target_index].values
            flight_mode = self._determine_flight_mode(classifier, input_window)

            x_data.append(input_window)
            y_data.append(output_window)
            modes.append(flight_mode)

        return np.array(x_data), np.array(y_data), np.array(modes)

    def _determine_flight_mode(self, model: RandomForestClassifier, window_data: np.ndarray) -> int:
        """Predict flight mode for a window of data"""
        predictions = model.predict(window_data)
        return Counter(predictions).most_common(1)[0][0]

    def _prepare_datasets(self, x_data: np.ndarray, y_data: np.ndarray, modes: np.ndarray) -> Dict[str, Any]:
        """Split and scale datasets by flight mode"""
        # Split into train/test
        x_train, x_test, y_train, y_test, mode_train, mode_test = train_test_split(
            x_data, y_data, modes,
            test_size=self.train_config.test_size,
            random_state=self.train_config.random_state,
            shuffle=True
        )

        # Scale data
        x_train_all = np.concatenate(x_train, axis=0)
        y_train_all = np.concatenate(y_train, axis=0)

        self.scaler_x.fit(x_train_all)
        self.scaler_y.fit(y_train_all.reshape(-1, 1))

        # Organize by flight mode
        train_data = {mode: ([], []) for mode in np.unique(modes)}
        test_data = {mode: ([], []) for mode in np.unique(modes)}

        for x, y, mode in zip(x_train, y_train, mode_train):
            x_scaled = self.scaler_x.transform(x)
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
            train_data[mode][0].append(x_scaled)
            train_data[mode][1].append(y_scaled)

        for x, y, mode in zip(x_test, y_test, mode_test):
            x_scaled = self.scaler_x.transform(x)
            y_scaled = self.scaler_y.transform(y.reshape(-1, 1)).flatten()
            test_data[mode][0].append(x_scaled)
            test_data[mode][1].append(y_scaled)

        # Convert to numpy arrays
        for mode in train_data:
            train_data[mode] = (np.array(train_data[mode][0]), np.array(train_data[mode][1]))
            test_data[mode] = (np.array(test_data[mode][0]), np.array(test_data[mode][1]))

        return {
            'train_data': train_data,
            'test_data': test_data,
            'scalers': (self.scaler_x, self.scaler_y)
        }