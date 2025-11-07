import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelBinarizer
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: np.ndarray = None, 
                                   average: str = 'weighted') -> Dict[str, float]:
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional)
        average: Averaging method for multi-class metrics
    
    Returns:
        Dictionary containing various metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)
    
    # ROC AUC for multi-class (if probabilities are provided)
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) > 2:
                # Multi-class ROC AUC
                lb = LabelBinarizer()
                y_true_bin = lb.fit_transform(y_true)
                if y_true_bin.shape[1] == 1:
                    y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
                metrics['roc_auc'] = roc_auc_score(y_true_bin, y_pred_proba, average=average)
            else:
                # Binary ROC AUC
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
        except Exception as e:
            metrics['roc_auc'] = np.nan
    
    return metrics


def get_classification_report_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    """
    Get detailed classification report as dictionary.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Classification report as dictionary
    """
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str] = None, 
                         figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        figsize: Figure size
    
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    return fig


def calculate_model_performance_summary(models_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Create a summary DataFrame of model performance.
    
    Args:
        models_results: Dictionary with model names as keys and metrics as values
    
    Returns:
        DataFrame with model performance summary
    """
    df = pd.DataFrame(models_results).T
    df = df.round(4)
    
    # Add ranking based on F1 score
    if 'f1' in df.columns:
        df['f1_rank'] = df['f1'].rank(ascending=False)
    
    return df.sort_values('f1', ascending=False) if 'f1' in df.columns else df


def save_metrics_to_json(metrics: Dict[str, Any], filepath: str) -> None:
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Metrics dictionary
        filepath: Path to save JSON file
    """
    import json
    import os
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        return obj
    
    metrics_serializable = convert_numpy_types(metrics)
    
    with open(filepath, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
