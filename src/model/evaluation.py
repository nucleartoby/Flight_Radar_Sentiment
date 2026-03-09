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
