"""
Evaluation metrics for the MLCQ benchmark.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    def __init__(self, average_method: str = 'weighted'):
        """
        Initialize metrics calculator.
        
        Args:
            average_method: Method for averaging multi-class metrics
        """
        self.average_method = average_method
    
    def calculate_binary_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                y_pred_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Calculate binary classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            
        Returns:
            Dictionary of metric values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        return metrics
    
    def calculate_multiclass_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                   y_pred_proba: Optional[np.ndarray] = None,
                                   labels: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate multi-class classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            labels: Class labels (optional)
            
        Returns:
            Dictionary of metric values
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=self.average_method),
            'recall': recall_score(y_true, y_pred, average=self.average_method),
            'f1': f1_score(y_true, y_pred, average=self.average_method),
        }
        
        if y_pred_proba is not None and len(np.unique(y_true)) > 2:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, 
                                                  multi_class='ovr', average=self.average_method)
            except ValueError:
                logger.warning("Could not calculate ROC AUC for multiclass")
        
        return metrics
    
    def calculate_per_class_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  labels: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            
        Returns:
            Dictionary of per-class metrics
        """
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1 = f1_score(y_true, y_pred, average=None)
        
        per_class_metrics = {}
        for i, label in enumerate(labels):
            if i < len(precision):
                per_class_metrics[label] = {
                    'precision': precision[i],
                    'recall': recall[i],
                    'f1': f1[i]
                }
        
        return per_class_metrics
    
    def calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix
        """
        return confusion_matrix(y_true, y_pred)
    
    def get_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                labels: Optional[List[str]] = None) -> str:
        """
        Get detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels (optional)
            
        Returns:
            Classification report string
        """
        return classification_report(y_true, y_pred, target_names=labels)


class CrossValidationEvaluator:
    """Perform cross-validation evaluation."""
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize cross-validation evaluator.
        
        Args:
            cv_folds: Number of CV folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    def evaluate_model(self, model, X: np.ndarray, y: np.ndarray, 
                      scoring: str = 'accuracy') -> Dict[str, float]:
        """
        Evaluate model using cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            scoring: Scoring method
            
        Returns:
            Dictionary with CV scores
        """
        scores = cross_val_score(model, X, y, cv=self.cv, scoring=scoring)
        
        return {
            'cv_scores': scores,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_min': scores.min(),
            'cv_max': scores.max()
        }
    
    def evaluate_multiple_metrics(self, model, X: np.ndarray, y: np.ndarray, 
                                metrics: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model using multiple metrics.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            metrics: List of metrics to evaluate
            
        Returns:
            Dictionary with results for each metric
        """
        results = {}
        
        for metric in metrics:
            try:
                result = self.evaluate_model(model, X, y, scoring=metric)
                results[metric] = result
            except Exception as e:
                logger.error(f"Error evaluating metric {metric}: {e}")
                results[metric] = {'error': str(e)}
        
        return results


class ModelComparator:
    """Compare multiple models."""
    
    def __init__(self, metrics_calculator: MetricsCalculator):
        """
        Initialize model comparator.
        
        Args:
            metrics_calculator: Metrics calculator instance
        """
        self.metrics_calculator = metrics_calculator
        self.results = {}
    
    def add_model_results(self, model_name: str, y_true: np.ndarray, y_pred: np.ndarray,
                         y_pred_proba: Optional[np.ndarray] = None, 
                         labels: Optional[List[str]] = None, 
                         cv_results: Optional[Dict] = None) -> None:
        """
        Add results for a model.
        
        Args:
            model_name: Name of the model
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            labels: Class labels (optional)
            cv_results: Cross-validation results (optional)
        """
        is_binary = len(np.unique(y_true)) == 2
        
        if is_binary:
            metrics = self.metrics_calculator.calculate_binary_metrics(
                y_true, y_pred, y_pred_proba)
        else:
            metrics = self.metrics_calculator.calculate_multiclass_metrics(
                y_true, y_pred, y_pred_proba, labels)
        
        per_class_metrics = None
        if labels:
            per_class_metrics = self.metrics_calculator.calculate_per_class_metrics(
                y_true, y_pred, labels)
        
        self.results[model_name] = {
            'metrics': metrics,
            'per_class_metrics': per_class_metrics,
            'confusion_matrix': self.metrics_calculator.calculate_confusion_matrix(y_true, y_pred),
            'classification_report': self.metrics_calculator.get_classification_report(y_true, y_pred, labels),
            'cv_results': cv_results
        }
    
    def get_comparison_table(self, metrics: List[str]) -> pd.DataFrame:
        """
        Get comparison table of models.
        
        Args:
            metrics: List of metrics to include
            
        Returns:
            DataFrame with model comparison
        """
        comparison_data = []
        
        for model_name, results in self.results.items():
            row = {'Model': model_name}
            
            for metric in metrics:
                if metric in results['metrics']:
                    row[metric] = results['metrics'][metric]
                else:
                    row[metric] = np.nan
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric: str = 'f1') -> Tuple[str, float]:
        """
        Get the best performing model.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, metric_value)
        """
        best_model = None
        best_score = -np.inf
        
        for model_name, results in self.results.items():
            if metric in results['metrics']:
                score = results['metrics'][metric]
                if score > best_score:
                    best_score = score
                    best_model = model_name
        
        return best_model, best_score
    
    def save_results(self, output_path: str) -> None:
        """
        Save comparison results to file.
        
        Args:
            output_path: Path to save results
        """
        serializable_results = {}
        
        for model_name, results in self.results.items():
            serializable_results[model_name] = {
                'metrics': results['metrics'],
                'per_class_metrics': results['per_class_metrics'],
                'confusion_matrix': results['confusion_matrix'].tolist(),
                'classification_report': results['classification_report'],
                'cv_results': results['cv_results']
            }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Saved comparison results to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")


class PlotGenerator:
    """Generate evaluation plots."""
    
    def __init__(self, output_dir: str):
        """
        Initialize plot generator.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str], 
                            model_name: str, save: bool = True) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            labels: Class labels
            model_name: Name of the model
            save: Whether to save the plot
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{model_name}_confusion_matrix.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str, save: bool = True) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            save: Whether to save the plot
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'{model_name}_roc_curve.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            metric: str = 'f1', save: bool = True) -> None:
        """
        Plot model comparison.
        
        Args:
            comparison_df: DataFrame with model comparison
            metric: Metric to plot
            save: Whether to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        models = comparison_df['Model']
        scores = comparison_df[metric]
        
        bars = plt.bar(models, scores, color='skyblue', edgecolor='navy', alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel(metric.upper())
        plt.title(f'Model Comparison - {metric.upper()}')
        plt.xticks(rotation=45)
        
        for bar, score in zip(bars, scores):
            if not np.isnan(score):
                plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(self.output_dir / f'model_comparison_{metric}.png', 
                       dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Example usage of evaluation metrics."""
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    y_pred_proba = np.array([0.1, 0.8, 0.6, 0.9, 0.7, 0.2, 0.4, 0.3])
    
    metrics_calc = MetricsCalculator()
    
    metrics = metrics_calc.calculate_binary_metrics(y_true, y_pred, y_pred_proba)
    print("Binary metrics:", metrics)
    
    comparator = ModelComparator(metrics_calc)
    
    comparator.add_model_results('Example Model', y_true, y_pred, y_pred_proba)
    
    comparison_df = comparator.get_comparison_table(['accuracy', 'precision', 'recall', 'f1'])
    print("\nComparison table:")
    print(comparison_df)
    
    plot_gen = PlotGenerator('results/plots')
    
    cm = metrics_calc.calculate_confusion_matrix(y_true, y_pred)
    plot_gen.plot_confusion_matrix(cm, ['Clean', 'Smelly'], 'Example Model')


if __name__ == "__main__":
    main()