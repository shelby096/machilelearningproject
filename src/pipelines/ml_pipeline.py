import pandas as pd
import os
import sys
from datetime import datetime
from typing import Dict, Any, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.train_model import ModelTrainer
from models.evaluate_model import ModelEvaluator
from data.load import DataLoader
from utils.logger import get_logger
from utils.config_loader import ConfigLoader
from utils.metrics import save_metrics_to_json

logger = get_logger(__name__)


class MLPipeline:
    """Complete machine learning pipeline for diet recommendation."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.train_config = self.config_loader.get_train_config()
        
        # Initialize components
        self.trainer = ModelTrainer(self.config_loader)
        self.evaluator = ModelEvaluator()
        self.loader = DataLoader(self.config_loader)
        
        # Pipeline state
        self.pipeline_results = {}
        self.execution_log = []
        self.trained_models = {}
        self.evaluation_results = {}
    
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
    
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load processed data for training.
        
        Returns:
            Processed DataFrame ready for training
        """
        try:
            self.log_step("Data Loading", "STARTED")
            
            # Load processed data
            processed_data_path = self.train_config['training']['processed_data_path']
            
            # Try to load from processed directory
            try:
                df = self.loader.load_processed_data("processed_diet_data.csv", data_type='processed')
            except FileNotFoundError:
                # Fallback: try to load from the specified path
                if os.path.exists(processed_data_path):
                    df = pd.read_csv(processed_data_path)
                else:
                    raise FileNotFoundError(f"Processed data not found at {processed_data_path}")
            
            self.log_step("Data Loading", "COMPLETED", f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
            return df
            
        except Exception as e:
            self.log_step("Data Loading", "FAILED", str(e))
            raise
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for model training.
        
        Args:
            df: Processed DataFrame
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        try:
            self.log_step("Data Preparation", "STARTED")
            
            # Prepare data using trainer
            X_train, X_test, y_train, y_test = self.trainer.prepare_data(df)
            
            # Store data info
            data_info = {
                'total_samples': len(df),
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'features_count': X_train.shape[1],
                'target_classes': y_train.nunique(),
                'class_distribution': y_train.value_counts().to_dict()
            }
            
            self.pipeline_results['data_info'] = data_info
            
            self.log_step("Data Preparation", "COMPLETED", 
                         f"Train: {X_train.shape}, Test: {X_test.shape}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            self.log_step("Data Preparation", "FAILED", str(e))
            raise
    
    def train_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                    y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        
        Returns:
            Dictionary containing training results for all models
        """
        try:
            self.log_step("Model Training", "STARTED")
            
            # Train all models
            training_results = self.trainer.train_all_models(X_train, X_test, y_train, y_test)
            
            # Select best model
            best_model, best_model_name = self.trainer.select_best_model()
            
            # Store results
            self.trained_models = self.trainer.models
            self.pipeline_results['training_results'] = training_results
            self.pipeline_results['best_model'] = best_model_name
            
            # Save models
            if self.train_config['training']['save_models']:
                saved_paths = self.trainer.save_models()
                self.pipeline_results['saved_model_paths'] = saved_paths
            
            models_trained = len([r for r in training_results.values() if 'error' not in r])
            self.log_step("Model Training", "COMPLETED", 
                         f"Trained {models_trained} models, Best: {best_model_name}")
            
            return training_results
            
        except Exception as e:
            self.log_step("Model Training", "FAILED", str(e))
            raise
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary containing evaluation results for all models
        """
        try:
            self.log_step("Model Evaluation", "STARTED")
            
            evaluation_results = {}
            
            # Evaluate each trained model
            for model_name, model in self.trained_models.items():
                try:
                    eval_result = self.evaluator.evaluate_single_model(
                        model, X_test, y_test, model_name
                    )
                    evaluation_results[model_name] = eval_result
                    
                    # Generate evaluation plots
                    self.evaluator.generate_evaluation_plots(
                        model, X_test, y_test, model_name
                    )
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {str(e)}")
                    evaluation_results[model_name] = {'error': str(e)}
            
            # Generate comparison analysis
            comparison_results = self.evaluator.compare_models(evaluation_results)
            
            # Generate comparison plots
            self.evaluator.generate_model_comparison_plots(evaluation_results)
            
            # Create comprehensive evaluation report
            report_path = self.evaluator.create_evaluation_report(evaluation_results)
            
            # Store results
            self.evaluation_results = evaluation_results
            self.pipeline_results['evaluation_results'] = evaluation_results
            self.pipeline_results['comparison_results'] = comparison_results
            self.pipeline_results['evaluation_report_path'] = report_path
            
            successful_evaluations = len([r for r in evaluation_results.values() if 'error' not in r])
            self.log_step("Model Evaluation", "COMPLETED", 
                         f"Evaluated {successful_evaluations} models")
            
            return evaluation_results
            
        except Exception as e:
            self.log_step("Model Evaluation", "FAILED", str(e))
            raise
    
    def generate_model_artifacts(self) -> Dict[str, str]:
        """
        Generate additional model artifacts and reports.
        
        Returns:
            Dictionary mapping artifact types to file paths
        """
        try:
            self.log_step("Artifact Generation", "STARTED")
            
            artifacts = {}
            
            # Save training summary
            training_summary = self.trainer.get_training_summary()
            training_summary_path = os.path.join("reports", "training_summary.json")
            save_metrics_to_json(training_summary, training_summary_path)
            artifacts['training_summary'] = training_summary_path
            
            # Save evaluation results
            evaluation_summary_path = os.path.join("reports", "evaluation_summary.json")
            save_metrics_to_json(self.evaluation_results, evaluation_summary_path)
            artifacts['evaluation_summary'] = evaluation_summary_path
            
            # Save model comparison
            if 'comparison_results' in self.pipeline_results:
                comparison_path = os.path.join("reports", "model_comparison.json")
                save_metrics_to_json(self.pipeline_results['comparison_results'], comparison_path)
                artifacts['model_comparison'] = comparison_path
            
            # Generate model cards (documentation)
            model_cards_path = self._generate_model_cards()
            if model_cards_path:
                artifacts['model_cards'] = model_cards_path
            
            self.pipeline_results['artifacts'] = artifacts
            
            self.log_step("Artifact Generation", "COMPLETED", 
                         f"Generated {len(artifacts)} artifacts")
            
            return artifacts
            
        except Exception as e:
            self.log_step("Artifact Generation", "FAILED", str(e))
            raise
    
    def _generate_model_cards(self) -> str:
        """
        Generate model cards (documentation) for all trained models.
        
        Returns:
            Path to model cards file
        """
        try:
            model_cards_path = os.path.join("reports", "model_cards.md")
            os.makedirs("reports", exist_ok=True)
            
            with open(model_cards_path, 'w') as f:
                f.write("# Diet Recommendation Model Cards\n\n")
                f.write("This document provides detailed information about all trained models.\n\n")
                
                for model_name in self.trained_models.keys():
                    f.write(f"## {model_name.replace('_', ' ').title()}\n\n")
                    
                    # Model description
                    model_descriptions = {
                        'random_forest': 'Random Forest is an ensemble method that uses multiple decision trees.',
                        'xgboost': 'XGBoost is a gradient boosting framework optimized for speed and performance.',
                        'lightgbm': 'LightGBM is a gradient boosting framework that uses tree-based learning.',
                        'logistic_regression': 'Logistic Regression is a linear model for classification problems.'
                    }
                    
                    f.write(f"**Description:** {model_descriptions.get(model_name, 'Advanced machine learning model')}\n\n")
                    
                    # Performance metrics
                    if model_name in self.evaluation_results and 'error' not in self.evaluation_results[model_name]:
                        results = self.evaluation_results[model_name]
                        f.write("**Performance Metrics:**\n")
                        f.write(f"- Accuracy: {results.get('accuracy', 0):.4f}\n")
                        f.write(f"- Precision: {results.get('precision', 0):.4f}\n")
                        f.write(f"- Recall: {results.get('recall', 0):.4f}\n")
                        f.write(f"- F1 Score: {results.get('f1', 0):.4f}\n")
                        if 'roc_auc' in results:
                            f.write(f"- ROC AUC: {results.get('roc_auc', 0):.4f}\n")
                        f.write("\n")
                    
                    # Use cases
                    f.write("**Recommended Use Cases:**\n")
                    if model_name == 'random_forest':
                        f.write("- Good for interpretability and handling mixed data types\n")
                        f.write("- Robust to overfitting\n")
                    elif model_name == 'xgboost':
                        f.write("- High performance on structured data\n")
                        f.write("- Good for competitions and production systems\n")
                    elif model_name == 'lightgbm':
                        f.write("- Fast training and prediction\n")
                        f.write("- Memory efficient\n")
                    elif model_name == 'logistic_regression':
                        f.write("- Fast and simple baseline model\n")
                        f.write("- Good for linear relationships\n")
                    
                    f.write("\n")
                    
                    # Model configuration
                    if model_name in self.trainer.model_config['models']:
                        config = self.trainer.model_config['models'][model_name]
                        f.write("**Model Configuration:**\n")
                        for param, value in config.items():
                            f.write(f"- {param}: {value}\n")
                        f.write("\n")
                    
                    f.write("---\n\n")
            
            return model_cards_path
            
        except Exception as e:
            logger.error(f"Error generating model cards: {str(e)}")
            return None
    
    def run_full_pipeline(self, data_path: str = None) -> Dict[str, Any]:
        """
        Execute the complete ML pipeline.
        
        Args:
            data_path: Optional path to processed data
        
        Returns:
            Dictionary containing pipeline results and metrics
        """
        try:
            start_time = datetime.now()
            logger.info("Starting ML Pipeline execution...")
            
            # Step 1: Load processed data
            if data_path:
                df = pd.read_csv(data_path)
                self.log_step("Data Loading", "COMPLETED", f"Loaded from {data_path}")
            else:
                df = self.load_processed_data()
            
            # Step 2: Prepare training data
            X_train, X_test, y_train, y_test = self.prepare_training_data(df)
            
            # Step 3: Train models
            training_results = self.train_models(X_train, X_test, y_train, y_test)
            
            # Step 4: Evaluate models
            evaluation_results = self.evaluate_models(X_test, y_test)
            
            # Step 5: Generate artifacts
            artifacts = self.generate_model_artifacts()
            
            # Pipeline completion
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            # Compile final results
            pipeline_summary = {
                'status': 'SUCCESS',
                'execution_time_seconds': execution_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'models_trained': len(self.trained_models),
                'best_model': self.pipeline_results.get('best_model'),
                'pipeline_results': self.pipeline_results,
                'execution_log': self.execution_log
            }
            
            # Save pipeline summary
            summary_path = os.path.join("reports", "ml_pipeline_summary.json")
            save_metrics_to_json(pipeline_summary, summary_path)
            
            logger.info(f"ML Pipeline completed successfully in {execution_time:.2f} seconds")
            logger.info(f"Models trained: {len(self.trained_models)}")
            logger.info(f"Best model: {self.pipeline_results.get('best_model')}")
            
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
            
            logger.error(f"ML Pipeline failed: {str(e)}")
            return error_summary
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline execution status.
        
        Returns:
            Dictionary containing pipeline status
        """
        return {
            'execution_log': self.execution_log,
            'pipeline_results': self.pipeline_results,
            'models_trained': len(self.trained_models),
            'steps_completed': len(self.execution_log),
            'last_step': self.execution_log[-1] if self.execution_log else None
        }
    
    def load_best_model(self) -> Any:
        """
        Load the best performing model.
        
        Returns:
            Best trained model
        """
        if not self.pipeline_results.get('best_model'):
            raise ValueError("No best model available. Run the pipeline first.")
        
        best_model_name = self.pipeline_results['best_model']
        
        if best_model_name in self.trained_models:
            return self.trained_models[best_model_name]
        else:
            # Try to load from saved models
            model_path = os.path.join("models", f"{best_model_name}.joblib")
            if os.path.exists(model_path):
                import joblib
                return joblib.load(model_path)
            else:
                raise FileNotFoundError(f"Best model not found: {best_model_name}")


def main():
    """Main function to run ML pipeline."""
    try:
        # Initialize and run pipeline
        pipeline = MLPipeline()
        results = pipeline.run_full_pipeline()
        
        print("\n" + "="*50)
        print("ML PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"Status: {results['status']}")
        print(f"Execution Time: {results['execution_time_seconds']:.2f} seconds")
        print(f"Models Trained: {results['models_trained']}")
        print(f"Best Model: {results['best_model']}")
        print("="*50)
        
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
