# In src/evaluate.py
from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    A comprehensive model evaluation class for regression tasks.
    Provides metrics calculation, visualization, and result export capabilities.
    
    Args:
        model: Trained model with a predict() method
        X_test: Test features (numpy array or pandas DataFrame)
        y_test: True target values (numpy array or pandas Series)
        feature_names: List of feature names. If None, will be generated if X is a DataFrame
        model_name: Name of the model for display purposes
    """
    
    def __init__(
        self,
        model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        feature_names: Optional[List[str]] = None,
        model_name: str = "Model"
    ):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.feature_names = self._get_feature_names(feature_names)
        self.y_pred = self._get_predictions()
        self.metrics: Dict[str, float] = {}
        self.feature_importance: Optional[pd.Series] = None
        
    def _get_feature_names(self, feature_names: Optional[List[str]]) -> List[str]:
        """Extract or generate feature names from input data."""
        if feature_names is not None:
            return feature_names
        if hasattr(self.X_test, 'columns'):
            return list(self.X_test.columns)
        return [f"feature_{i}" for i in range(self.X_test.shape[1])]
    
    def _get_predictions(self) -> np.ndarray:
        """Generate predictions from the model."""
        try:
            return self.model.predict(self.X_test)
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if len(self.y_test) == 0 or len(self.y_pred) == 0:
            raise ValueError("Test data or predictions are empty")
            
        self.metrics = {
            # Error metrics
            'mse': mean_squared_error(self.y_test, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred)),
            'mae': mean_absolute_error(self.y_test, self.y_pred),
            'mape': mean_absolute_percentage_error(self.y_test, self.y_pred) * 100,
            'max_error': max_error(self.y_test, self.y_pred),
            
            # R-squared metrics
            'r2': r2_score(self.y_test, self.y_pred),
            'adj_r2': 1 - (1 - r2_score(self.y_test, self.y_pred)) * (len(self.y_test) - 1) / 
                     (len(self.y_test) - self.X_test.shape[1] - 1),
            'explained_variance': explained_variance_score(self.y_test, self.y_pred),
            
            # Additional useful metrics
            'mean_absolute_percentage': mean_absolute_percentage_error(self.y_test, self.y_pred) * 100,
            'median_absolute_error': np.median(np.abs(self.y_test - self.y_pred)),
            'mean_absolute_deviation': np.mean(np.abs(self.y_test - np.mean(self.y_test))),
        }
        return self.metrics
    
    def get_feature_importance(self) -> Optional[pd.Series]:
        """
        Extract feature importance from the model if available.
        
        Returns:
            Pandas Series with feature importance or None if not available
        """
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.Series(
                self.model.feature_importances_,
                index=self.feature_names
            ).sort_values(ascending=False)
        elif hasattr(self.model, 'coef_'):
            self.feature_importance = pd.Series(
                np.abs(self.model.coef_),
                index=self.feature_names
            ).sort_values(ascending=False)
        else:
            logger.warning("Model does not have feature_importances_ or coef_ attribute")
            return None
            
        return self.feature_importance
    
    def plot_actual_vs_predicted(
        self, 
        figsize: Tuple[int, int] = (10, 8),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot actual vs predicted values.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: If provided, save the plot to this path
            
        Returns:
            Matplotlib Figure object
        """
        plt.figure(figsize=figsize)
        
        # Create scatter plot
        plt.scatter(
            self.y_test, 
            self.y_pred,
            alpha=0.5,
            label='Predictions'
        )
        
        # Add a reference line
        max_val = max(self.y_test.max(), self.y_pred.max())
        min_val = min(self.y_test.min(), self.y_pred.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Values - {self.model_name}')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Plot saved to {save_path}")
            
        return plt.gcf()
    
    def plot_residuals(
        self,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot residuals vs predicted values.
        
        Args:
            figsize: Figure size as (width, height)
            save_path: If provided, save the plot to this path
            
        Returns:
            Matplotlib Figure object
        """
        residuals = self.y_test - self.y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Residuals vs Predicted
        ax1.scatter(self.y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        ax1.grid(True)
        
        # Histogram of residuals
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_title('Distribution of Residuals')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Residual plot saved to {save_path}")
            
        return fig
    
    def plot_feature_importance(
        self,
        top_n: int = 20,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[str] = None
    ) -> Optional[plt.Figure]:
        """
        Plot feature importance if available.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size as (width, height)
            save_path: If provided, save the plot to this path
            
        Returns:
            Matplotlib Figure object or None if feature importance not available
        """
        if self.feature_importance is None:
            self.get_feature_importance()
            
        if self.feature_importance is None:
            return None
            
        # Get top N features
        top_features = self.feature_importance.head(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(
            x=top_features.values,
            y=top_features.index,
            palette='viridis'
        )
        
        plt.title(f'Top {top_n} Feature Importance - {self.model_name}')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            logger.info(f"Feature importance plot saved to {save_path}")
            
        return plt.gcf()
    
    def save_metrics(self, filepath: str) -> None:
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            filepath: Path to save the metrics
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        metrics_serializable = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in self.metrics.items()
        }
        
        with open(path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
            
        logger.info(f"Metrics saved to {path}")
    
    def generate_report(
        self,
        output_dir: str = "reports",
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report with metrics and plots.
        
        Args:
            output_dir: Directory to save the report
            model_name: Optional model name for the report
            
        Returns:
            Dictionary containing all evaluation results
        """
        if model_name:
            self.model_name = model_name
            
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate metrics if not already done
        if not self.metrics:
            self.calculate_metrics()
            
        # Generate plots
        plots = {
            'actual_vs_predicted': str(output_path / f"{self.model_name}_actual_vs_predicted.png"),
            'residuals': str(output_path / f"{self.model_name}_residuals.png"),
            'feature_importance': str(output_path / f"{self.model_name}_feature_importance.png")
        }
        
        # Create plots
        self.plot_actual_vs_predicted(save_path=plots['actual_vs_predicted'])
        self.plot_residuals(save_path=plots['residuals'])
        
        # Only create feature importance plot if available
        if self.get_feature_importance() is not None:
            self.plot_feature_importance(save_path=plots['feature_importance'])
        else:
            del plots['feature_importance']
        
        # Prepare report
        report = {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'plots': plots,
            'feature_importance': (
                self.feature_importance.to_dict() 
                if self.feature_importance is not None 
                else None
            )
        }
        
        # Save report as JSON
        report_path = output_path / f"{self.model_name}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        logger.info(f"Evaluation report saved to {report_path}")
        return report

# Helper function for quick evaluation
def evaluate_model(
    model: Any,
    X_test: Union[np.ndarray, pd.DataFrame],
    y_test: Union[np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    model_name: str = "Model",
    output_dir: str = "reports"
) -> Dict[str, Any]:
    """
    Quick evaluation of a model with default settings.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: True target values
        feature_names: Optional list of feature names
        model_name: Name of the model
        output_dir: Directory to save the report
        
    Returns:
        Dictionary containing evaluation results
    """
    evaluator = ModelEvaluator(
        model=model,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        model_name=model_name
    )
    
    # Calculate metrics
    metrics = evaluator.calculate_metrics()
    
    # Generate report
    report = evaluator.generate_report(output_dir=output_dir)
    
    return report