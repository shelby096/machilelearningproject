import pandas as pd
import os
import joblib
from typing import Any, Dict
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DataLoader:
    """Class for loading and saving processed data."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.data_config = self.config_loader.get_data_config()
    
    def save_processed_data(self, df: pd.DataFrame, filename: str, 
                           data_type: str = 'processed') -> str:
        """
        Save processed data to specified directory.
        
        Args:
            df: DataFrame to save
            filename: Name of the file
            data_type: Type of data ('interim' or 'processed')
        
        Returns:
            Path where data was saved
        """
        try:
            if data_type == 'interim':
                save_path = self.data_config['data']['interim_data_path']
            else:
                save_path = self.data_config['data']['processed_data_path']
            
            # Create directory if it doesn't exist
            os.makedirs(save_path, exist_ok=True)
            
            # Full file path
            file_path = os.path.join(save_path, filename)
            
            # Save based on file extension
            if filename.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif filename.endswith('.parquet'):
                df.to_parquet(file_path, index=False)
            elif filename.endswith('.xlsx'):
                df.to_excel(file_path, index=False)
            else:
                # Default to CSV
                file_path = file_path + '.csv'
                df.to_csv(file_path, index=False)
            
            logger.info(f"Data saved successfully to: {file_path}")
            logger.info(f"Saved {len(df)} rows and {len(df.columns)} columns")
            
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            raise
    
    def load_processed_data(self, filename: str, data_type: str = 'processed') -> pd.DataFrame:
        """
        Load processed data from specified directory.
        
        Args:
            filename: Name of the file to load
            data_type: Type of data ('interim' or 'processed')
        
        Returns:
            Loaded DataFrame
        """
        try:
            if data_type == 'interim':
                load_path = self.data_config['data']['interim_data_path']
            else:
                load_path = self.data_config['data']['processed_data_path']
            
            file_path = os.path.join(load_path, filename)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Load based on file extension
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            elif filename.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            else:
                # Try CSV as default
                df = pd.read_csv(file_path)
            
            logger.info(f"Data loaded successfully from: {file_path}")
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def save_preprocessors(self, preprocessors: Dict[str, Any], 
                          filename: str = 'preprocessors.joblib') -> str:
        """
        Save preprocessing objects (scalers, encoders, etc.).
        
        Args:
            preprocessors: Dictionary containing preprocessing objects
            filename: Name of the file to save
        
        Returns:
            Path where preprocessors were saved
        """
        try:
            models_path = "models"
            os.makedirs(models_path, exist_ok=True)
            
            file_path = os.path.join(models_path, filename)
            joblib.dump(preprocessors, file_path)
            
            logger.info(f"Preprocessors saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving preprocessors: {str(e)}")
            raise
    
    def load_preprocessors(self, filename: str = 'preprocessors.joblib') -> Dict[str, Any]:
        """
        Load preprocessing objects.
        
        Args:
            filename: Name of the file to load
        
        Returns:
            Dictionary containing preprocessing objects
        """
        try:
            models_path = "models"
            file_path = os.path.join(models_path, filename)
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Preprocessors file not found: {file_path}")
            
            preprocessors = joblib.load(file_path)
            logger.info(f"Preprocessors loaded from: {file_path}")
            
            return preprocessors
            
        except Exception as e:
            logger.error(f"Error loading preprocessors: {str(e)}")
            raise
    
    def save_data_summary(self, summary: Dict[str, Any], 
                         filename: str = 'data_summary.json') -> str:
        """
        Save data summary information.
        
        Args:
            summary: Dictionary containing data summary
            filename: Name of the file to save
        
        Returns:
            Path where summary was saved
        """
        try:
            import json
            
            reports_path = "reports"
            os.makedirs(reports_path, exist_ok=True)
            
            file_path = os.path.join(reports_path, filename)
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'tolist'):
                    return obj.tolist()
                elif hasattr(obj, 'item'):
                    return obj.item()
                elif isinstance(obj, dict):
                    return {key: convert_numpy_types(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                return obj
            
            summary_serializable = convert_numpy_types(summary)
            
            with open(file_path, 'w') as f:
                json.dump(summary_serializable, f, indent=2)
            
            logger.info(f"Data summary saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error saving data summary: {str(e)}")
            raise
    
    def create_data_backup(self, df: pd.DataFrame, backup_name: str) -> str:
        """
        Create a backup of the current dataset.
        
        Args:
            df: DataFrame to backup
            backup_name: Name for the backup
        
        Returns:
            Path where backup was saved
        """
        try:
            from datetime import datetime
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{backup_name}_{timestamp}.csv"
            
            backup_path = os.path.join("data", "backups")
            os.makedirs(backup_path, exist_ok=True)
            
            file_path = os.path.join(backup_path, filename)
            df.to_csv(file_path, index=False)
            
            logger.info(f"Data backup created: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error creating backup: {str(e)}")
            raise
