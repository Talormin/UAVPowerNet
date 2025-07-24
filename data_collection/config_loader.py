import os
from dotenv import load_dotenv
import json


class ConfigLoader:
    def __init__(self):
        """Initialize configuration loader and load environment variables"""
        load_dotenv()  # Load environment variables from .env file
        self.config_path = os.getenv('DATA_CONFIG_PATH', './config/data_sources.json')
        self._validate_config()

    def _validate_config(self):
        """Check if configuration file exists"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(
                f"Configuration file not found at {self.config_path}"
            )

    def load_config(self):
        """Load and return data sources configuration"""
        with open(self.config_path) as f:
            config = json.load(f)
        return config.get('data_sources', {})