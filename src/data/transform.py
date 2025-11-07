import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Dict, List, Tuple, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DataTransformer:
    """Class for transforming and cleaning data."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.data_config = self.config_loader.get_data_config()
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with missing values handled
        """
        logger.info("Handling missing values...")
        df_clean = df.copy()
        
        # Get column types from config
        numeric_cols = self.data_config['data']['validation']['numeric_columns']
        categorical_cols = self.data_config['data']['validation']['categorical_columns']
        
        # Handle numeric columns
        numeric_cols_in_df = [col for col in numeric_cols if col in df_clean.columns]
        if numeric_cols_in_df:
            imputer_numeric = SimpleImputer(strategy='median')
            df_clean[numeric_cols_in_df] = imputer_numeric.fit_transform(df_clean[numeric_cols_in_df])
            self.imputers['numeric'] = imputer_numeric
        
        # Handle categorical columns
        categorical_cols_in_df = [col for col in categorical_cols if col in df_clean.columns]
        if categorical_cols_in_df:
            imputer_categorical = SimpleImputer(strategy='most_frequent')
            df_clean[categorical_cols_in_df] = imputer_categorical.fit_transform(df_clean[categorical_cols_in_df])
            self.imputers['categorical'] = imputer_categorical
        
        logger.info(f"Missing values handled. Remaining missing values: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def remove_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Remove outliers from numeric columns.
        
        Args:
            df: Input DataFrame
            method: Method for outlier detection ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame with outliers removed
        """
        logger.info(f"Removing outliers using {method} method...")
        df_clean = df.copy()
        
        numeric_cols = self.data_config['data']['validation']['numeric_columns']
        numeric_cols_in_df = [col for col in numeric_cols if col in df_clean.columns]
        
        initial_rows = len(df_clean)
        
        for col in numeric_cols_in_df:
            if method == 'iqr':
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
            
            elif method == 'zscore':
                z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                df_clean = df_clean[z_scores <= threshold]
        
        removed_rows = initial_rows - len(df_clean)
        logger.info(f"Removed {removed_rows} outlier rows ({removed_rows/initial_rows*100:.2f}%)")
        
        return df_clean
    
    def encode_categorical_variables(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Encode categorical variables.
        
        Args:
            df: Input DataFrame
            target_column: Target column name (for label encoding)
        
        Returns:
            DataFrame with encoded categorical variables
        """
        logger.info("Encoding categorical variables...")
        df_encoded = df.copy()
        
        categorical_cols = self.data_config['data']['validation']['categorical_columns']
        categorical_cols_in_df = [col for col in categorical_cols if col in df_encoded.columns]
        
        for col in categorical_cols_in_df:
            if col == target_column:
                # Use Label Encoder for target variable
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.encoders[col] = le
                logger.info(f"Label encoded target column: {col}")
            else:
                # Use One-Hot Encoding for other categorical variables
                unique_values = df_encoded[col].nunique()
                if unique_values <= 10:  # One-hot encode if <= 10 unique values
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                    logger.info(f"One-hot encoded column: {col} ({unique_values} categories)")
                else:
                    # Use Label Encoding for high cardinality categorical variables
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                    logger.info(f"Label encoded high-cardinality column: {col} ({unique_values} categories)")
        
        return df_encoded
    
    def scale_numeric_features(self, df: pd.DataFrame, target_column: str = None) -> pd.DataFrame:
        """
        Scale numeric features.
        
        Args:
            df: Input DataFrame
            target_column: Target column to exclude from scaling
        
        Returns:
            DataFrame with scaled numeric features
        """
        logger.info("Scaling numeric features...")
        df_scaled = df.copy()
        
        numeric_cols = [col for col in df_scaled.select_dtypes(include=[np.number]).columns 
                       if col != target_column]
        
        if numeric_cols:
            scaler = StandardScaler()
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            self.scalers['standard'] = scaler
            logger.info(f"Scaled {len(numeric_cols)} numeric columns")
        
        return df_scaled
    
    def create_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Creating engineered features...")
        df_engineered = df.copy()
        
        # BMI Categories
        if 'BMI' in df_engineered.columns:
            df_engineered['BMI_Category'] = pd.cut(
                df_engineered['BMI'], 
                bins=[0, 18.5, 25, 30, float('inf')], 
                labels=['Underweight', 'Normal', 'Overweight', 'Obese']
            )
            logger.info("Created BMI categories")
        
        # Age Groups
        if 'Age' in df_engineered.columns:
            df_engineered['Age_Group'] = pd.cut(
                df_engineered['Age'], 
                bins=[0, 30, 45, 60, float('inf')], 
                labels=['Young', 'Middle_Age', 'Senior', 'Elderly']
            )
            logger.info("Created age groups")
        
        # Health Risk Score
        risk_factors = []
        if 'Chronic_Disease' in df_engineered.columns:
            risk_factors.append((df_engineered['Chronic_Disease'] != 'None').astype(int))
        if 'Smoking_Habit' in df_engineered.columns:
            risk_factors.append((df_engineered['Smoking_Habit'] == 'Yes').astype(int))
        if 'Alcohol_Consumption' in df_engineered.columns:
            risk_factors.append((df_engineered['Alcohol_Consumption'] == 'Yes').astype(int))
        
        if risk_factors:
            df_engineered['Health_Risk_Score'] = sum(risk_factors)
            logger.info("Created health risk score")
        
        # Exercise Level
        if 'Exercise_Frequency' in df_engineered.columns:
            df_engineered['Exercise_Level'] = pd.cut(
                df_engineered['Exercise_Frequency'], 
                bins=[-1, 0, 2, 4, float('inf')], 
                labels=['None', 'Low', 'Moderate', 'High']
            )
            logger.info("Created exercise level categories")
        
        return df_engineered
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names after transformation."""
        return list(self.feature_names) if hasattr(self, 'feature_names') else []
    
    def transform_pipeline(self, df: pd.DataFrame, target_column: str = None, 
                          remove_outliers: bool = True) -> pd.DataFrame:
        """
        Complete transformation pipeline.
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            remove_outliers: Whether to remove outliers
        
        Returns:
            Fully transformed DataFrame
        """
        logger.info("Starting data transformation pipeline...")
        
        # Step 1: Handle missing values
        df_transformed = self.handle_missing_values(df)
        
        # Step 2: Create engineered features
        df_transformed = self.create_feature_engineering(df_transformed)
        
        # Step 3: Remove outliers (optional)
        if remove_outliers:
            df_transformed = self.remove_outliers(df_transformed)
        
        # Step 4: Encode categorical variables
        df_transformed = self.encode_categorical_variables(df_transformed, target_column)
        
        # Step 5: Scale numeric features
        df_transformed = self.scale_numeric_features(df_transformed, target_column)
        
        logger.info(f"Transformation pipeline completed. Final shape: {df_transformed.shape}")
        return df_transformed
