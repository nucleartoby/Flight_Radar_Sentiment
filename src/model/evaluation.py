import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import logging

class ModelEvaluator:

    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Predictions vs Actual"):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Oil Price')
        plt.ylabel('Predicted Oil Price')
        plt.title(f'{title} - Scatter Plot')
        
        plt.subplot(2, 2, 2) # Time series plot
        plt.plot(y_true, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Oil Price')
        plt.title(f'{title} - Time Series')
        plt.legend()
        
        plt.subplot(2, 2, 3)
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Oil Price')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        
        plt.subplot(2, 2, 4)
        plt.hist(residuals, bins=30, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        
        plt.tight_layout()
        plt.show()

    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                              feature_names: List[str], top_n: int = 15):
        if feature_importance is None:
            self.logger.warning("No feature importance available")
            return
        
        indices = np.argsort(feature_importance)[::-1][:top_n] # top features only guh
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_importance, y=top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        plt.show()
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'max_error': np.max(np.abs(y_true - y_pred)),
            'mean_error': np.mean(y_true - y_pred)
        }
        
        return metrics
    
    def print_evaluation_report(self, metrics: Dict[str, float]):
        print("Model Evaluation Report")
        print(f"Root Mean Square Error: ${metrics['rmse']:.4f}")
        print(f"Mean Absolute Error: ${metrics['mae']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        print(f"Maximum Error: ${metrics['max_error']:.4f}")
        print(f"Mean Error: ${metrics['mean_error']:.4f}")