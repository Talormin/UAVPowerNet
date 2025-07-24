import pandas as pd
from config_loader import ConfigLoader
from data_processor import DataProcessor
import os


class FlightDataProcessor:
    def __init__(self):
        """Initialize flight data processor with configuration"""
        self.config = ConfigLoader().load_config()
        self.validate_data_paths()

    def validate_data_paths(self):
        """Verify all required data files exist before processing"""
        for file_info in self.config.values():
            if not os.path.exists(file_info['path']):
                raise FileNotFoundError(f"Data file not found: {file_info['path']}")

    def process_all_data(self):
        """Main method to execute complete data processing pipeline"""
        # Load base dataset (angular velocity)
        angular_velocity_df = self._load_data('vehicle_angular_velocity')
        base_timestamp = angular_velocity_df["timestamp"]
        processor = DataProcessor(base_timestamp)

        # Process each dataset
        processed_data = {
            'angular_velocity': self._process_angular_velocity(angular_velocity_df),
            'airspeed': self._process_dataset('airspeed', ['true_airspeed_m_s', 'air_temperature_celsius']),
            # Other datasets would be processed here...
        }

        # Combine and save results
        combined_df = self._combine_data(processed_data)
        self._save_data(combined_df)

    def _load_data(self, dataset_name: str) -> pd.DataFrame:
        """Load specified dataset from CSV file"""
        file_info = self.config[dataset_name]
        return pd.read_csv(file_info['path'])

    def _process_dataset(self, dataset_name: str, columns: list) -> dict:
        """Standard processing for most datasets"""
        df = self._load_data(dataset_name)
        processor = DataProcessor(self._load_data('vehicle_angular_velocity')['timestamp'])
        return processor.interpolate_data(df, columns)

    def _process_angular_velocity(self, df: pd.DataFrame) -> dict:
        """Special processing for angular velocity data"""
        return {
            'xyz_0': df["xyz[0]"],
            'xyz_1': df["xyz[1]"],
            'xyz_2': df["xyz[2]"]
        }

    def _combine_data(self, processed_data: dict) -> pd.DataFrame:
        """Combine all processed data into single DataFrame"""
        # Implementation would combine all processed datasets
        pass

    def _save_data(self, df: pd.DataFrame):
        """Save final processed data to CSV file"""
        output_path = os.getenv('OUTPUT_PATH', './output/processed_flight_data.csv')
        df.to_csv(output_path, index=False)


if __name__ == "__main__":
    processor = FlightDataProcessor()
    processor.process_all_data()