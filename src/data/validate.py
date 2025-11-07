import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class DataValidator:
    """Class for validating data quality and integrity."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.data_config = self.config_loader.get_data_config()
        self.validation_results = {}
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate dataset schema against configuration.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary containing validation results
        """
        logger.info("Validating dataset schema...")
        
        results = {
            'schema_valid': True,
            'issues': [],
            'warnings': []
        }
        
        # Check required columns
        required_columns = self.data_config['data']['validation']['required_columns']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            results['schema_valid'] = False
            results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check numeric columns
        numeric_columns = self.data_config['data']['validation']['numeric_columns']
        for col in numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    results['warnings'].append(f"Column {col} should be numeric but has type {df[col].dtype}")
        
        # Check categorical columns
        categorical_columns = self.data_config['data']['validation']['categorical_columns']
        for col in categorical_columns:
            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 50:
                    results['warnings'].append(f"Column {col} appears to be continuous but marked as categorical")
        
        logger.info(f"Schema validation completed. Valid: {results['schema_valid']}")
        return results
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality metrics.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary containing data quality results
        """
        logger.info("Validating data quality...")
        
        results = {
            'quality_score': 0.0,
            'issues': [],
            'metrics': {}
        }
        
        # Calculate quality metrics
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        results['metrics'] = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_percentage': (missing_cells / total_cells) * 100,
            'duplicate_percentage': (duplicate_rows / len(df)) * 100,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Quality checks
        quality_score = 100.0
        
        # Missing values penalty
        if results['metrics']['missing_percentage'] > 10:
            quality_score -= 20
            results['issues'].append(f"High missing values: {results['metrics']['missing_percentage']:.2f}%")
        elif results['metrics']['missing_percentage'] > 5:
            quality_score -= 10
            results['issues'].append(f"Moderate missing values: {results['metrics']['missing_percentage']:.2f}%")
        
        # Duplicate rows penalty
        if results['metrics']['duplicate_percentage'] > 5:
            quality_score -= 15
            results['issues'].append(f"High duplicate rows: {results['metrics']['duplicate_percentage']:.2f}%")
        elif results['metrics']['duplicate_percentage'] > 1:
            quality_score -= 5
            results['issues'].append(f"Some duplicate rows: {results['metrics']['duplicate_percentage']:.2f}%")
        
        results['quality_score'] = max(0, quality_score)
        
        logger.info(f"Data quality score: {results['quality_score']:.2f}/100")
        return results
    
    def validate_target_distribution(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """
        Validate target variable distribution.
        
        Args:
            df: DataFrame containing target variable
            target_column: Name of target column
        
        Returns:
            Dictionary containing target distribution analysis
        """
        logger.info(f"Validating target distribution for: {target_column}")
        
        if target_column not in df.columns:
            return {'error': f"Target column {target_column} not found"}
        
        target_series = df[target_column]
        value_counts = target_series.value_counts()
        
        results = {
            'unique_values': target_series.nunique(),
            'value_counts': value_counts.to_dict(),
            'distribution_balance': {},
            'issues': []
        }
        
        # Check class balance
        if results['unique_values'] > 1:
            min_class_percentage = (value_counts.min() / len(target_series)) * 100
            max_class_percentage = (value_counts.max() / len(target_series)) * 100
            
            results['distribution_balance'] = {
                'min_class_percentage': min_class_percentage,
                'max_class_percentage': max_class_percentage,
                'balance_ratio': value_counts.min() / value_counts.max()
            }
            
            # Imbalance warnings
            if min_class_percentage < 5:
                results['issues'].append(f"Severe class imbalance: smallest class is {min_class_percentage:.2f}%")
            elif min_class_percentage < 10:
                results['issues'].append(f"Moderate class imbalance: smallest class is {min_class_percentage:.2f}%")
        
        logger.info(f"Target has {results['unique_values']} unique values")
        return results
    
    def validate_feature_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature distributions for anomalies.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Dictionary containing feature distribution analysis
        """
        logger.info("Validating feature distributions...")
        
        results = {
            'numeric_features': {},
            'categorical_features': {},
            'issues': []
        }
        
        # Analyze numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            col_stats = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'skewness': df[col].skew(),
                'kurtosis': df[col].kurtosis(),
                'zeros_percentage': (df[col] == 0).sum() / len(df) * 100,
                'outliers_percentage': self._calculate_outlier_percentage(df[col])
            }
            
            # Check for issues
            if abs(col_stats['skewness']) > 2:
                results['issues'].append(f"High skewness in {col}: {col_stats['skewness']:.2f}")
            
            if col_stats['outliers_percentage'] > 10:
                results['issues'].append(f"High outliers in {col}: {col_stats['outliers_percentage']:.2f}%")
            
            results['numeric_features'][col] = col_stats
        
        # Analyze categorical features
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            col_stats = {
                'unique_values': df[col].nunique(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'most_frequent_percentage': (df[col].value_counts().iloc[0] / len(df)) * 100,
                'cardinality': df[col].nunique() / len(df)
            }
            
            # Check for issues
            if col_stats['cardinality'] > 0.9:
                results['issues'].append(f"Very high cardinality in {col}: {col_stats['unique_values']} unique values")
            
            if col_stats['most_frequent_percentage'] > 90:
                results['issues'].append(f"Dominant category in {col}: {col_stats['most_frequent_percentage']:.2f}%")
            
            results['categorical_features'][col] = col_stats
        
        logger.info(f"Feature distribution validation completed. Found {len(results['issues'])} issues")
        return results
    
    def _calculate_outlier_percentage(self, series: pd.Series) -> float:
        """Calculate percentage of outliers using IQR method."""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = (series < lower_bound) | (series > upper_bound)
        return (outliers.sum() / len(series)) * 100
    
    def generate_validation_report(self, df: pd.DataFrame, target_column: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Args:
            df: DataFrame to validate
            target_column: Target column name (optional)
        
        Returns:
            Complete validation report
        """
        logger.info("Generating comprehensive validation report...")
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'dataset_shape': df.shape,
            'schema_validation': self.validate_schema(df),
            'quality_validation': self.validate_data_quality(df),
            'feature_validation': self.validate_feature_distributions(df)
        }
        
        if target_column:
            report['target_validation'] = self.validate_target_distribution(df, target_column)
        
        # Overall validation status
        all_issues = []
        all_issues.extend(report['schema_validation'].get('issues', []))
        all_issues.extend(report['quality_validation'].get('issues', []))
        all_issues.extend(report['feature_validation'].get('issues', []))
        
        if target_column and 'target_validation' in report:
            all_issues.extend(report['target_validation'].get('issues', []))
        
        report['overall_status'] = {
            'validation_passed': len(all_issues) == 0,
            'total_issues': len(all_issues),
            'all_issues': all_issues
        }
        
        logger.info(f"Validation report generated. Status: {'PASSED' if report['overall_status']['validation_passed'] else 'FAILED'}")
        return report
