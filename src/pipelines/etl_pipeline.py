import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.extract import DataExtractor
from data.transform import DataTransformer
from data.load import DataLoader
from data.validate import DataValidator
from features.build_features import FeatureBuilder
from utils.logger import get_logger
from utils.config_loader import ConfigLoader

logger = get_logger(__name__)


class ETLPipeline:
    """Complete ETL pipeline for diet recommendation data."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.data_config = self.config_loader.get_data_config()
        self.train_config = self.config_loader.get_train_config()
        
        # Initialize components
        self.extractor = DataExtractor(self.config_loader)
        self.transformer = DataTransformer(self.config_loader)
        self.loader = DataLoader(self.config_loader)
        self.validator = DataValidator(self.config_loader)
        self.feature_builder = FeatureBuilder(self.config_loader)
        
        # Pipeline state
        self.pipeline_results = {}
        self.execution_log = []
    
    def log_step(self, step_name: str, status: str, details: str = "") -> None:
        """Log pipeline step execution."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step_name,
            'status': status,
            'details': details
        }
        self.execution_log.append(log_entry)
        logger.info(f"Step '{step_name}': {status} - {details}")
    
    def extract_data(self) -> pd.DataFrame:
        """
        Extract raw data from source.
        
        Returns:
            Raw DataFrame
        """
        try:
            self.log_step("Data Extraction", "STARTED")
            
            # Extract main dataset
            df_raw = self.extractor.extract_diet_data()
            
            # Validate required columns
            if not self.extractor.validate_required_columns(df_raw):
                raise ValueError("Required columns missing from dataset")
            
            # Get basic data info
            data_info = self.extractor.get_data_info(df_raw)
            self.pipeline_results['raw_data_info'] = data_info
            
            # Save raw data backup
            backup_path = self.loader.create_data_backup(df_raw, "raw_data")
            
            self.log_step("Data Extraction", "COMPLETED", 
                         f"Extracted {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
            
            return df_raw
            
        except Exception as e:
            self.log_step("Data Extraction", "FAILED", str(e))
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and integrity.
        
        Args:
            df: DataFrame to validate
        
        Returns:
            Validation report
        """
        try:
            self.log_step("Data Validation", "STARTED")
            
            target_column = self.train_config['training']['processed_data_path'].split('/')[-1].replace('.csv', '')
            target_column = 'Recommended_Meal_Plan'  # From our dataset
            
            validation_report = self.validator.generate_validation_report(df, target_column)
            self.pipeline_results['validation_report'] = validation_report
            
            # Save validation report
            report_path = self.loader.save_data_summary(validation_report, "validation_report.json")
            
            status = "PASSED" if validation_report['overall_status']['validation_passed'] else "WARNING"
            details = f"Issues found: {validation_report['overall_status']['total_issues']}"
            
            self.log_step("Data Validation", status, details)
            
            return validation_report
            
        except Exception as e:
            self.log_step("Data Validation", "FAILED", str(e))
            raise
    
    def transform_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform and clean the data.
        
        Args:
            df: Raw DataFrame
        
        Returns:
            Transformed DataFrame
        """
        try:
            self.log_step("Data Transformation", "STARTED")
            
            target_column = 'Recommended_Meal_Plan'
            
            # Apply transformation pipeline
            df_transformed = self.transformer.transform_pipeline(
                df, 
                target_column=target_column,
                remove_outliers=self.train_config['training']['preprocessing']['remove_outliers']
            )
            
            # Save interim data
            interim_path = self.loader.save_processed_data(
                df_transformed, 
                "interim_diet_data.csv", 
                data_type='interim'
            )
            
            # Save transformation artifacts
            preprocessors = {
                'scalers': self.transformer.scalers,
                'encoders': self.transformer.encoders,
                'imputers': self.transformer.imputers
            }
            
            preprocessor_path = self.loader.save_preprocessors(preprocessors)
            self.pipeline_results['preprocessor_path'] = preprocessor_path
            
            self.log_step("Data Transformation", "COMPLETED", 
                         f"Shape: {df_transformed.shape}")
            
            return df_transformed
            
        except Exception as e:
            self.log_step("Data Transformation", "FAILED", str(e))
            raise
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features.
        
        Args:
            df: Transformed DataFrame
        
        Returns:
            DataFrame with engineered features
        """
        try:
            self.log_step("Feature Engineering", "STARTED")
            
            # Build all features
            df_features = self.feature_builder.build_all_features(df)
            
            # Feature selection (optional)
            target_column = 'Recommended_Meal_Plan'
            if target_column in df_features.columns:
                # Prepare data for feature selection
                X = df_features.drop(columns=[target_column])
                y = df_features[target_column]
                
                # Select top features
                selected_features = self.feature_builder.select_features(X, y, method='mutual_info', k=25)
                
                # Keep selected features + target
                df_final = df_features[selected_features + [target_column]].copy()
                
                # Save feature importance report
                feature_report = self.feature_builder.get_feature_importance_report()
                self.pipeline_results['feature_report'] = feature_report
                
                report_path = self.loader.save_data_summary(feature_report, "feature_importance.json")
            else:
                df_final = df_features
            
            self.log_step("Feature Engineering", "COMPLETED", 
                         f"Final features: {df_final.shape[1]-1}")
            
            return df_final
            
        except Exception as e:
            self.log_step("Feature Engineering", "FAILED", str(e))
            raise
    
    def load_processed_data(self, df: pd.DataFrame) -> str:
        """
        Save final processed data.
        
        Args:
            df: Final processed DataFrame
        
        Returns:
            Path to saved data
        """
        try:
            self.log_step("Data Loading", "STARTED")
            
            # Save processed data
            processed_path = self.loader.save_processed_data(
                df, 
                "processed_diet_data.csv", 
                data_type='processed'
            )
            
            # Save final data summary
            final_info = {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'target_distribution': df['Recommended_Meal_Plan'].value_counts().to_dict() if 'Recommended_Meal_Plan' in df.columns else {}
            }
            
            summary_path = self.loader.save_data_summary(final_info, "processed_data_summary.json")
            self.pipeline_results['processed_data_path'] = processed_path
            self.pipeline_results['final_data_info'] = final_info
            
            self.log_step("Data Loading", "COMPLETED", f"Saved to: {processed_path}")
            
            return processed_path
            
        except Exception as e:
            self.log_step("Data Loading", "FAILED", str(e))
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """
        Execute the complete ETL pipeline.
        
        Returns:
            Dictionary containing pipeline results and metrics
        """
        try:
            start_time = datetime.now()
            logger.info("Starting ETL Pipeline execution...")
            
            # Step 1: Extract Data
            df_raw = self.extract_data()
            
            # Step 2: Validate Data
            validation_report = self.validate_data(df_raw)
            
            # Step 3: Transform Data
            df_transformed = self.transform_data(df_raw)
            
            # Step 4: Engineer Features
            df_features = self.engineer_features(df_transformed)
            
            # Step 5: Load Processed Data
            processed_path = self.load_processed_data(df_features)
            
            # Pipeline completion
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Compile final results
            pipeline_summary = {
                'status': 'SUCCESS',
                'execution_time_seconds': execution_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'processed_data_path': processed_path,
                'pipeline_results': self.pipeline_results,
                'execution_log': self.execution_log,
                'final_data_shape': df_features.shape,
                'data_quality_score': validation_report['quality_validation']['quality_score']
            }
            
            # Save pipeline summary
            summary_path = self.loader.save_data_summary(pipeline_summary, "etl_pipeline_summary.json")
            
            logger.info(f"ETL Pipeline completed successfully in {execution_time:.2f} seconds")
            logger.info(f"Final dataset shape: {df_features.shape}")
            logger.info(f"Data quality score: {validation_report['quality_validation']['quality_score']:.2f}/100")
            
            return pipeline_summary
            
        except Exception as e:
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds() if 'start_time' in locals() else 0
            
            error_summary = {
                'status': 'FAILED',
                'error': str(e),
                'execution_time_seconds': execution_time,
                'execution_log': self.execution_log
            }
            
            logger.error(f"ETL Pipeline failed: {str(e)}")
            return error_summary
    
    def run_pipeline_step(self, step_name: str, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Run a specific pipeline step.
        
        Args:
            step_name: Name of the step to run
            df: Input DataFrame (if required)
        
        Returns:
            Output DataFrame
        """
        step_functions = {
            'extract': self.extract_data,
            'validate': lambda df: self.validate_data(df),
            'transform': lambda df: self.transform_data(df),
            'engineer': lambda df: self.engineer_features(df),
            'load': lambda df: self.load_processed_data(df)
        }
        
        if step_name not in step_functions:
            raise ValueError(f"Unknown step: {step_name}")
        
        if step_name == 'extract':
            return step_functions[step_name]()
        else:
            if df is None:
                raise ValueError(f"DataFrame required for step: {step_name}")
            return step_functions[step_name](df)
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline execution status.
        
        Returns:
            Dictionary containing pipeline status
        """
        return {
            'execution_log': self.execution_log,
            'pipeline_results': self.pipeline_results,
            'steps_completed': len(self.execution_log),
            'last_step': self.execution_log[-1] if self.execution_log else None
        }


def main():
    """Main function to run ETL pipeline."""
    try:
        # Initialize and run pipeline
        pipeline = ETLPipeline()
        results = pipeline.run_full_pipeline()
        
        print("\n" + "="*50)
        print("ETL PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Execution Time: {results['execution_time_seconds']:.2f} seconds")
        print(f"Final Data Shape: {results['final_data_shape']}")
        if 'data_quality_score' in results:
            print(f"Data Quality Score: {results['data_quality_score']:.2f}/100")
        print(f"Processed Data Path: {results['processed_data_path']}")
        print("="*50)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
