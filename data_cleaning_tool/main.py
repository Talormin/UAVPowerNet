import pandas as pd
from core.anti_debug import RuntimeValidator
from processors.phase1_preprocess import DataPreprocessor
from processors.phase2_clean import DataCleaningEngine
from utils.obfuscator import DataObfuscator


class DataProcessingPipeline:
    """Complete data processing workflow controller"""

    def __init__(self, source_file: str):
        """Initialize pipeline with source data"""
        RuntimeValidator.validate_environment()
        self.raw_data = pd.read_csv(source_file)
        self.processed_data = None

    def execute_pipeline(self) -> None:
        """Run complete data processing sequence"""
        # Phase 1: Initial preparation
        df = DataPreprocessor.normalize_timestamps(self.raw_data)
        anomalies = DataPreprocessor.detect_anomalous_values(df)

        # Phase 2: Core cleaning
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df_clean = DataCleaningEngine.apply_adaptive_filter(df, numeric_cols)
        df_clean = DataCleaningEngine.handle_missing_values(df_clean)

        # Apply final transformations
        self.processed_data = DataObfuscator.transform_identifiers(df_clean)

    def save_results(self, output_path: str) -> None:
        """Persist processed data to storage"""
        if self.processed_data is not None:
            self.processed_data.to_csv(output_path, index=False)
            print(f"Processing complete. Results saved to {output_path}")
        else:
            raise ValueError("No processed data available. Execute pipeline first.")


if __name__ == "__main__":
    processor = DataProcessingPipeline("input_data.csv")
    processor.execute_pipeline()
    processor.save_results("cleaned_output.csv")