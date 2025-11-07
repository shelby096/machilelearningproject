import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import LabelBinarizer
import joblib
import os
from typing import Dict, Any, List, Tuple
from src.utils.logger import get_logger
from src.utils.metrics import (
    calculate_classification_metrics, plot_confusion_matrix,
    calculate_model_performance_summary, save_metrics_to_json
)

logger = get_logger(__name__)


class ModelEvaluator:
    """Class for comprehensive model evaluation and analysis."""
    
    def __init__(self):
        self.evaluation_results = {}
        self.comparison_results = {}
    
    def load_model(self, model_path: str) -> Any:
        """
        Load a trained model from file.
        
        Args:
            model_path: Path to the saved model
        
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            raise
    
    def evaluate_single_model(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                            model_name: str) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
        
        Returns:
            Dictionary containing detailed evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = {'model_name': model_name}
        
        try:
            # Basic predictions
            y_pred = model.predict(X_test)
            results['predictions'] = y_pred.tolist()
            
            # Prediction probabilities (if available)
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
                results['prediction_probabilities'] = y_pred_proba.tolist()
            else:
                y_pred_proba = None
            
            # Basic metrics
            basic_metrics = calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            results.update(basic_metrics)
            
            # Detailed classification report
            class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            results['classification_report'] = class_report
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            results['confusion_matrix'] = cm.tolist()
            
            # Feature importance (if available)
            if hasattr(model, 'feature_importances_'):
                feature_names = X_test.columns.tolist()
                importance_scores = model.feature_importances_
                feature_importance = dict(zip(feature_names, importance_scores))
                results['feature_importance'] = dict(sorted(feature_importance.items(), 
                                                          key=lambda x: x[1], reverse=True))
            
            # Model-specific metrics
            if hasattr(model, 'score'):
                results['model_score'] = model.score(X_test, y_test)
            
            logger.info(f"{model_name} evaluation completed successfully")
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def compare_models(self, models_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models and generate comparison report.
        
        Args:
            models_results: Dictionary containing results for multiple models
        
        Returns:
            Dictionary containing model comparison analysis
        """
        logger.info("Comparing models...")
        
        comparison = {
            'model_count': len(models_results),
            'metrics_comparison': {},
            'ranking': {},
            'best_model_per_metric': {}
        }
        
        # Extract metrics for comparison
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, results in models_results.items():
                if 'error' not in results and metric in results:
                    metric_values[model_name] = results[metric]
            
            if metric_values:
                comparison['metrics_comparison'][metric] = metric_values
                # Find best model for this metric
                best_model = max(metric_values, key=metric_values.get)
                comparison['best_model_per_metric'][metric] = {
                    'model': best_model,
                    'score': metric_values[best_model]
                }
        
        # Overall ranking based on F1 score
        if 'f1' in comparison['metrics_comparison']:
            f1_scores = comparison['metrics_comparison']['f1']
            ranking = sorted(f1_scores.items(), key=lambda x: x[1], reverse=True)
            comparison['ranking']['by_f1_score'] = ranking
        
        # Create performance summary DataFrame
        performance_df = calculate_model_performance_summary(models_results)
        comparison['performance_summary'] = performance_df.to_dict()
        
        logger.info("Model comparison completed")
        return comparison
    
    def generate_evaluation_plots(self, model, X_test: pd.DataFrame, y_test: pd.Series, 
                                model_name: str, save_path: str = "reports/figures") -> Dict[str, str]:
        """
        Generate evaluation plots for a model.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            save_path: Directory to save plots
        
        Returns:
            Dictionary mapping plot types to file paths
        """
        logger.info(f"Generating evaluation plots for {model_name}...")
        
        os.makedirs(save_path, exist_ok=True)
        plot_paths = {}
        
        try:
            y_pred = model.predict(X_test)
            
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            cm = confusion_matrix(y_test, y_pred)
            class_names = sorted(y_test.unique())
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            cm_path = os.path.join(save_path, f'{model_name}_confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths['confusion_matrix'] = cm_path
            
            # Feature Importance (if available)
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                feature_names = X_test.columns
                importance_scores = model.feature_importances_
                
                # Get top 15 features
                indices = np.argsort(importance_scores)[-15:]
                plt.barh(range(len(indices)), importance_scores[indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 15 Feature Importances - {model_name}')
                plt.tight_layout()
                
                fi_path = os.path.join(save_path, f'{model_name}_feature_importance.png')
                plt.savefig(fi_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['feature_importance'] = fi_path
            
            # ROC Curves (for multi-class)
            if hasattr(model, 'predict_proba') and len(np.unique(y_test)) > 2:
                plt.figure(figsize=(10, 8))
                y_pred_proba = model.predict_proba(X_test)
                
                # Binarize the output
                lb = LabelBinarizer()
                y_test_bin = lb.fit_transform(y_test)
                
                # Compute ROC curve for each class
                for i, class_name in enumerate(lb.classes_):
                    if y_test_bin.shape[1] > 1:
                        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
                    else:
                        fpr, tpr, _ = roc_curve(y_test_bin, y_pred_proba[:, 1])
                    
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')
                
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curves - {model_name}')
                plt.legend(loc="lower right")
                
                roc_path = os.path.join(save_path, f'{model_name}_roc_curves.png')
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['roc_curves'] = roc_path
            
            logger.info(f"Evaluation plots generated for {model_name}")
            
        except Exception as e:
            logger.error(f"Error generating plots for {model_name}: {str(e)}")
        
        return plot_paths
    
    def generate_model_comparison_plots(self, models_results: Dict[str, Dict[str, Any]], 
                                      save_path: str = "reports/figures") -> Dict[str, str]:
        """
        Generate comparison plots for multiple models.
        
        Args:
            models_results: Dictionary containing results for multiple models
            save_path: Directory to save plots
        
        Returns:
            Dictionary mapping plot types to file paths
        """
        logger.info("Generating model comparison plots...")
        
        os.makedirs(save_path, exist_ok=True)
        plot_paths = {}
        
        try:
            # Metrics comparison bar plot
            metrics = ['accuracy', 'precision', 'recall', 'f1']
            model_names = []
            metric_values = {metric: [] for metric in metrics}
            
            for model_name, results in models_results.items():
                if 'error' not in results:
                    model_names.append(model_name)
                    for metric in metrics:
                        metric_values[metric].append(results.get(metric, 0))
            
            if model_names:
                fig, axes = plt.subplots(2, 2, figsize=(15, 12))
                axes = axes.ravel()
                
                for i, metric in enumerate(metrics):
                    axes[i].bar(model_names, metric_values[metric])
                    axes[i].set_title(f'{metric.capitalize()} Comparison')
                    axes[i].set_ylabel(metric.capitalize())
                    axes[i].tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                comp_path = os.path.join(save_path, 'model_comparison.png')
                plt.savefig(comp_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths['model_comparison'] = comp_path
            
            logger.info("Model comparison plots generated")
            
        except Exception as e:
            logger.error(f"Error generating comparison plots: {str(e)}")
        
        return plot_paths
    
    def create_evaluation_report(self, models_results: Dict[str, Dict[str, Any]], 
                               save_path: str = "reports") -> str:
        """
        Create comprehensive evaluation report.
        
        Args:
            models_results: Dictionary containing results for multiple models
            save_path: Directory to save report
        
        Returns:
            Path to the generated report
        """
        logger.info("Creating evaluation report...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Generate comparison analysis
        comparison = self.compare_models(models_results)
        
        # Save detailed results
        results_path = os.path.join(save_path, "detailed_evaluation_results.json")
        save_metrics_to_json(models_results, results_path)
        
        # Save comparison results
        comparison_path = os.path.join(save_path, "model_comparison.json")
        save_metrics_to_json(comparison, comparison_path)
        
        # Create summary report
        summary_path = os.path.join(save_path, "evaluation_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("DIET RECOMMENDATION MODEL EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Models Evaluated: {comparison['model_count']}\n\n")
            
            # Best model per metric
            f.write("BEST MODEL PER METRIC:\n")
            f.write("-" * 25 + "\n")
            for metric, info in comparison['best_model_per_metric'].items():
                f.write(f"{metric.upper()}: {info['model']} ({info['score']:.4f})\n")
            
            f.write("\n")
            
            # Overall ranking
            if 'by_f1_score' in comparison['ranking']:
                f.write("OVERALL RANKING (by F1 Score):\n")
                f.write("-" * 30 + "\n")
                for i, (model, score) in enumerate(comparison['ranking']['by_f1_score'], 1):
                    f.write(f"{i}. {model}: {score:.4f}\n")
            
            f.write("\n")
            
            # Detailed metrics for each model
            f.write("DETAILED METRICS:\n")
            f.write("-" * 17 + "\n")
            for model_name, results in models_results.items():
                if 'error' not in results:
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"  Accuracy:  {results.get('accuracy', 0):.4f}\n")
                    f.write(f"  Precision: {results.get('precision', 0):.4f}\n")
                    f.write(f"  Recall:    {results.get('recall', 0):.4f}\n")
                    f.write(f"  F1 Score:  {results.get('f1', 0):.4f}\n")
                    if 'roc_auc' in results:
                        f.write(f"  ROC AUC:   {results.get('roc_auc', 0):.4f}\n")
        
        logger.info(f"Evaluation report created: {summary_path}")
        return summary_path
    
    def evaluate_models_from_directory(self, models_dir: str, X_test: pd.DataFrame, 
                                     y_test: pd.Series) -> Dict[str, Dict[str, Any]]:
        """
        Load and evaluate all models from a directory.
        
        Args:
            models_dir: Directory containing saved models
            X_test: Test features
            y_test: Test target
        
        Returns:
            Dictionary containing evaluation results for all models
        """
        logger.info(f"Evaluating models from directory: {models_dir}")
        
        results = {}
        
        # Find all model files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        
        for model_file in model_files:
            model_name = model_file.replace('.joblib', '')
            model_path = os.path.join(models_dir, model_file)
            
            try:
                # Load and evaluate model
                model = self.load_model(model_path)
                evaluation_result = self.evaluate_single_model(model, X_test, y_test, model_name)
                results[model_name] = evaluation_result
                
                # Generate plots
                self.generate_evaluation_plots(model, X_test, y_test, model_name)
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Generate comparison plots
        self.generate_model_comparison_plots(results)
        
        # Create evaluation report
        self.create_evaluation_report(results)
        
        logger.info("Model evaluation completed")
        return results
