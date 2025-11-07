import pandas as pd
import os
import sys
from typing import Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DataExtractor:
    """Class for extracting data from various sources."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.data_config = self.config_loader.get_data_config()
    
    def extract_csv_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Extract data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
        
        Returns:
            DataFrame containing the data
        """
        try:
            logger.info(f"Extracting data from CSV: {file_path}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_csv(file_path, **kwargs)
            logger.info(f"Successfully extracted {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting CSV data: {str(e)}")
            raise
    
    def extract_excel_data(self, file_path: str, sheet_name: str = 0, **kwargs) -> pd.DataFrame:
        """
        Extract data from Excel file.
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name or index
            **kwargs: Additional arguments for pd.read_excel
        
        Returns:
            DataFrame containing the data
        """
        try:
            logger.info(f"Extracting data from Excel: {file_path}, Sheet: {sheet_name}")
            
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            df = pd.read_excel(file_path, sheet_name=sheet_name, **kwargs)
            logger.info(f"Successfully extracted {len(df)} rows and {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting Excel data: {str(e)}")
            raise
    
    def extract_diet_data(self) -> pd.DataFrame:
        """
        Extract the main diet recommendation dataset.
        
        Returns:
            DataFrame containing the diet data
        """
        raw_data_path = self.data_config['data']['raw_data_path']
        logger.info(f"Extracting diet data from: {raw_data_path}")
        
        return self.extract_csv_data(raw_data_path)
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get basic information about the dataset.
        
        Args:
            df: DataFrame to analyze
        
        Returns:
            Dictionary containing dataset information
        """
        info = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicate_rows': df.duplicated().sum()
        }
        
        logger.info(f"Dataset info: Shape {info['shape']}, Missing values: {sum(info['missing_values'].values())}")
        
        return info
    
    def validate_required_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that required columns are present in the dataset.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            True if all required columns are present
        """
        required_columns = self.data_config['data']['validation']['required_columns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        logger.info("All required columns are present")
        return True
