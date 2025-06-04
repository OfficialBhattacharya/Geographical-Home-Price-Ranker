import yaml
import os
from pathlib import Path

class ConfigLoader:
    """Utility class to load and manage configuration from YAML file"""
    
    def __init__(self, config_file='config.yaml'):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Configuration file '{self.config_file}' not found. Please ensure it exists in the project root directory.")
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration file: {e}")
        except Exception as e:
            raise Exception(f"Error reading configuration file: {e}")
    
    def get_api_key(self):
        """Get FRED API key from config"""
        try:
            return self.config['api']['fred_api_key']
        except KeyError:
            raise KeyError("FRED API key not found in configuration. Please set 'api.fred_api_key' in config.yaml")
    
    def get_path(self, path_name):
        """Get a specific path from configuration"""
        try:
            path = self.config['paths'][path_name]
            # Convert to Path object and create directory if it doesn't exist
            path_obj = Path(path)
            if path_name.endswith('_dir'):
                path_obj.mkdir(parents=True, exist_ok=True)
            return str(path_obj)
        except KeyError:
            raise KeyError(f"Path '{path_name}' not found in configuration")
    
    def get_raw_data_dir(self):
        """Get the raw data directory path"""
        return self.get_path('raw_data_dir')
    
    def get_processed_data_dir(self):
        """Get the processed data directory path"""
        return self.get_path('processed_data_dir')
    
    def get_project_root(self):
        """Get the project root directory path"""
        return self.get_path('project_root')
    
    def get_fred_series_file(self):
        """Get the full path to the FRED series file"""
        filename = self.config['paths']['fred_series_file']
        return os.path.join(self.get_project_root(), filename)
    
    def get_unified_data_path(self):
        """Get the full path to the unified data file"""
        filename = self.config['paths']['unified_data_file']
        return os.path.join(self.get_processed_data_dir(), filename)
    
    def get_processing_setting(self, setting_name):
        """Get a processing setting"""
        try:
            return self.config['processing'][setting_name]
        except KeyError:
            raise KeyError(f"Processing setting '{setting_name}' not found in configuration")
    
    def get_start_date(self):
        """Get the start date for processing"""
        return self.get_processing_setting('start_date')
    
    def get_default_end_date(self):
        """Get the default end date for processing"""
        return self.get_processing_setting('default_end_date')
    
    def get_frequency_thresholds(self):
        """Get frequency detection thresholds"""
        return self.get_processing_setting('frequency_thresholds')
    
    def get_error_handling_setting(self, setting_name):
        """Get an error handling setting"""
        try:
            return self.config['error_handling'][setting_name]
        except KeyError:
            raise KeyError(f"Error handling setting '{setting_name}' not found in configuration")
    
    def get_file_setting(self, setting_name):
        """Get a file setting"""
        try:
            return self.config['files'][setting_name]
        except KeyError:
            raise KeyError(f"File setting '{setting_name}' not found in configuration")
    
    def should_continue_on_error(self):
        """Check if processing should continue on errors"""
        return self.get_error_handling_setting('continue_on_error')
    
    def get_max_retries(self):
        """Get maximum number of retries for API calls"""
        return self.get_error_handling_setting('max_retries')
    
    def get_api_timeout(self):
        """Get API timeout in seconds"""
        return self.get_error_handling_setting('api_timeout')
    
    def create_output_filename(self, base_name, end_date_str=None):
        """Create an output filename with date suffix"""
        if end_date_str:
            # Convert date string to filename format (YYYYMMDD)
            from datetime import datetime
            date_obj = datetime.strptime(end_date_str, '%Y-%m-%d')
            date_suffix = date_obj.strftime('%Y%m%d')
            name_parts = base_name.split('.')
            if len(name_parts) > 1:
                # Insert date before file extension
                return f"{'.'.join(name_parts[:-1])}_{date_suffix}.{name_parts[-1]}"
            else:
                return f"{base_name}_{date_suffix}"
        return base_name
    
    def get_full_config(self):
        """Get the complete configuration dictionary"""
        return self.config.copy()
    
    def validate_config(self):
        """Validate that all required configuration sections exist"""
        required_sections = ['api', 'paths', 'processing', 'files', 'error_handling']
        required_api_keys = ['fred_api_key']
        required_paths = ['project_root', 'raw_data_dir', 'processed_data_dir', 'fred_series_file', 'unified_data_file']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Required configuration section '{section}' is missing")
        
        for key in required_api_keys:
            if key not in self.config['api']:
                raise ValueError(f"Required API configuration '{key}' is missing")
        
        for path in required_paths:
            if path not in self.config['paths']:
                raise ValueError(f"Required path configuration '{path}' is missing")
        
        # Validate API key is not empty
        if not self.config['api']['fred_api_key'].strip():
            raise ValueError("FRED API key cannot be empty. Please set your API key in config.yaml")
        
        print("[OK] Configuration validation passed")
        return True

# Convenience function to get a configured instance
def get_config():
    """Get a configured instance of ConfigLoader"""
    return ConfigLoader()

# Test the configuration when module is run directly
if __name__ == "__main__":
    try:
        config = get_config()
        config.validate_config()
        print("Configuration loaded successfully!")
        print(f"Project root: {config.get_project_root()}")
        print(f"Raw data directory: {config.get_raw_data_dir()}")
        print(f"Processed data directory: {config.get_processed_data_dir()}")
        print(f"API key: {config.get_api_key()[:8]}...{config.get_api_key()[-4:]}")
    except Exception as e:
        print(f"Configuration error: {e}") 