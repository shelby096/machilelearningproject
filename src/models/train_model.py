import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
from typing import Dict, Any, Tuple, List
from src.utils.logger import get_logger
from src.utils.config_loader import ConfigLoader
from src.utils.metrics import calculate_classification_metrics, save_metrics_to_json

logger = get_logger(__name__)


class ModelTrainer:
    """Class for training machine learning models for diet recommendation."""
    
    def __init__(self, config_loader: ConfigLoader = None):
        self.config_loader = config_loader or ConfigLoader()
        self.model_config = self.config_loader.get_model_config()
        self.train_config = self.config_loader.get_train_config()
        self.models = {}
        self.model_results = {}
        self.best_model = None
        self.best_model_name = None
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare data for training by splitting features and target.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Preparing data for training...")
        
        target_column = self.model_config['models']['target_column']
        features_to_exclude = self.model_config['models']['features_to_exclude']
        
        # Separate features and target
        X = df.drop(columns=[target_column] + features_to_exclude, errors='ignore')
        y = df[target_column]
        
        # Split data
        test_size = self.model_config['evaluation']['test_size']
        random_state = self.model_config['evaluation']['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize all models with their configurations.
        
        Returns:
            Dictionary of initialized models
        """
        logger.info("Initializing models...")
        
        models = {}
        
        # Random Forest
        rf_config = self.model_config['models']['random_forest']
        models['random_forest'] = RandomForestClassifier(**rf_config)
        
        # XGBoost
        xgb_config = self.model_config['models']['xgboost']
        models['xgboost'] = xgb.XGBClassifier(**xgb_config)
        
        # LightGBM
        lgb_config = self.model_config['models']['lightgbm']
        models['lightgbm'] = lgb.LGBMClassifier(**lgb_config)
        
        # Logistic Regression
        lr_config = self.model_config['models']['logistic_regression']
        models['logistic_regression'] = LogisticRegression(**lr_config)
        
        logger.info(f"Initialized {len(models)} models")
        return models
    
    def train_single_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                          model_name: str) -> Any:
        """
        Train a single model.
        
        Args:
            model: Model instance to train
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
        
        Returns:
            Trained model
        """
        logger.info(f"Training {model_name}...")
        
        try:
            model.fit(X_train, y_train)
            logger.info(f"{model_name} training completed successfully")
            return model
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            raise
    
    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        try:
            # Predictions
            y_pred = model.predict(X_test)
            
            # Get prediction probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            
            # Calculate metrics
            metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            
            # Add model-specific information
            metrics['model_name'] = model_name
            metrics['test_samples'] = len(y_test)
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_names = X_test.columns.tolist()
                importance_dict = dict(zip(feature_names, model.feature_importances_))
                metrics['feature_importance'] = dict(sorted(importance_dict.items(), 
                                                          key=lambda x: x[1], reverse=True)[:10])
            
            logger.info(f"{model_name} evaluation completed - Accuracy: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            raise
    
    def perform_cross_validation(self, model, X: pd.DataFrame, y: pd.Series, 
                               model_name: str) -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to validate
            X: Features
            y: Target
            model_name: Name of the model
        
        Returns:
            Dictionary containing CV results
        """
        logger.info(f"Performing cross-validation for {model_name}...")
        
        cv_folds = self.model_config['evaluation']['cv_folds']
        scoring_metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        cv_results = {}
        
        for metric in scoring_metrics:
            try:
                scores = cross_val_score(model, X, y, cv=cv_folds, scoring=metric)
                cv_results[f'{metric}_mean'] = scores.mean()
                cv_results[f'{metric}_std'] = scores.std()
            except Exception as e:
                logger.warning(f"Could not calculate {metric} for {model_name}: {str(e)}")
                cv_results[f'{metric}_mean'] = 0.0
                cv_results[f'{metric}_std'] = 0.0
        
        logger.info(f"Cross-validation completed for {model_name}")
        return cv_results
    
    def hyperparameter_tuning(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                            model_name: str) -> Any:
        """
        Perform hyperparameter tuning for a model.
        
        Args:
            model: Model to tune
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
        
        Returns:
            Best model after tuning
        """
        logger.info(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define parameter grids for different models
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['liblinear', 'lbfgs']
            }
        }
        
        if model_name not in param_grids:
            logger.warning(f"No parameter grid defined for {model_name}, skipping tuning")
            return model
        
        try:
            scoring_metric = self.model_config['model_selection']['scoring_metric']
            cv_folds = self.train_config['hyperparameter_tuning']['cv_folds']
            
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=cv_folds, scoring=scoring_metric, 
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            return grid_search.best_estimator_
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            return model
    
    def train_all_models(self, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                        y_train: pd.Series, y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate all models.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
        
        Returns:
            Dictionary containing results for all models
        """
        logger.info("Training all models...")
        
        models = self.initialize_models()
        results = {}
        
        for model_name, model in models.items():
            try:
                logger.info(f"Processing {model_name}...")
                
                # Hyperparameter tuning (if enabled)
                if self.train_config['hyperparameter_tuning']['enabled']:
                    model = self.hyperparameter_tuning(model, X_train, y_train, model_name)
                
                # Train model
                trained_model = self.train_single_model(model, X_train, y_train, model_name)
                
                # Evaluate model
                evaluation_results = self.evaluate_model(trained_model, X_test, y_test, model_name)
                
                # Cross-validation
                if self.model_config['model_selection']['use_cross_validation']:
                    cv_results = self.perform_cross_validation(model, X_train, y_train, model_name)
                    evaluation_results['cross_validation'] = cv_results
                
                # Store results
                results[model_name] = evaluation_results
                self.models[model_name] = trained_model
                
                logger.info(f"{model_name} processing completed")
                
            except Exception as e:
                logger.error(f"Failed to process {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        self.model_results = results
        logger.info("All models training completed")
        return results
    
    def select_best_model(self) -> Tuple[Any, str]:
        """
        Select the best performing model based on the scoring metric.
        
        Returns:
            Tuple of (best_model, best_model_name)
        """
        logger.info("Selecting best model...")
        
        scoring_metric = self.model_config['model_selection']['scoring_metric']
        best_score = -1
        best_model_name = None
        
        for model_name, results in self.model_results.items():
            if 'error' in results:
                continue
            
            # Get the appropriate metric
            if scoring_metric == 'f1_weighted':
                score = results.get('f1', 0)
            elif scoring_metric == 'accuracy':
                score = results.get('accuracy', 0)
            else:
                score = results.get(scoring_metric, 0)
            
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        if best_model_name:
            self.best_model = self.models[best_model_name]
            self.best_model_name = best_model_name
            logger.info(f"Best model: {best_model_name} with {scoring_metric}: {best_score:.4f}")
        else:
            logger.error("No valid model found")
        
        return self.best_model, self.best_model_name
    
    def save_models(self, save_path: str = "models") -> Dict[str, str]:
        """
        Save all trained models.
        
        Args:
            save_path: Directory to save models
        
        Returns:
            Dictionary mapping model names to file paths
        """
        logger.info("Saving trained models...")
        
        os.makedirs(save_path, exist_ok=True)
        saved_paths = {}
        
        for model_name, model in self.models.items():
            try:
                file_path = os.path.join(save_path, f"{model_name}.joblib")
                joblib.dump(model, file_path)
                saved_paths[model_name] = file_path
                logger.info(f"Saved {model_name} to {file_path}")
            except Exception as e:
                logger.error(f"Error saving {model_name}: {str(e)}")
        
        # Save model results
        results_path = os.path.join(save_path, "model_results.json")
        save_metrics_to_json(self.model_results, results_path)
        
        logger.info("Model saving completed")
        return saved_paths
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive training summary.
        
        Returns:
            Dictionary containing training summary
        """
        return {
            'models_trained': list(self.models.keys()),
            'best_model': self.best_model_name,
            'model_results': self.model_results,
            'training_config': self.train_config,
            'model_config': self.model_config
        }
