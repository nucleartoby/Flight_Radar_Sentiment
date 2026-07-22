import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
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

        plt.subplot(2, 2, 2)  # Time series plot
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


    def plot_feature_importance(self, feature_importance: np.ndarray, feature_names: List[str], top_n: int = 15):
        if feature_importance is None:

            return

        indices = np.argsort(feature_importance)[::-1][:top_n]  # top features only guh
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
            'mean_error': np.mean(y_true - y_pred)}

        return metrics


    def print_evaluation_report(self, metrics: Dict[str, float]):
        print("Model Evaluation Report")
        print(f"Root Mean Square Error: ${metrics['rmse']:.4f}")
        print(f"Mean Absolute Error: ${metrics['mae']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")
        print(f"Mean Absolute Percentage Error: {metrics['mape']:.2f}%")
        print(f"Maximum Error: ${metrics['max_error']:.4f}")
        print(f"Mean Error: ${metrics['mean_error']:.4f}")


    def calculate_hit_rate(self, y_true_ret: np.ndarray, y_pred_ret: np.ndarray) -> Dict[str, float]:
        y_true_ret = np.asarray(y_true_ret)
        y_pred_ret = np.asarray(y_pred_ret)

        true_sign = np.sign(y_true_ret)
        pred_sign = np.sign(y_pred_ret)

        # Exclude days where either side is exactly zero (no directional call)
        mask = (true_sign != 0) & (pred_sign != 0)
        n_total = mask.sum()

        if n_total == 0:
            return {'hit_rate': np.nan, 'n_days': 0, 'n_correct': 0}

        n_correct = int((true_sign[mask] == pred_sign[mask]).sum())
        hit_rate = n_correct / n_total

        return {'hit_rate': hit_rate, 'n_days': int(n_total), 'n_correct': n_correct,}


    def backtest_strategy(self, y_true_ret: np.ndarray, y_pred_ret: np.ndarray, transaction_cost_bps: float = 2.0,periods_per_year: int = 252) -> Dict[str, float]:
        y_true_ret = np.asarray(y_true_ret, dtype=float)
        y_pred_ret = np.asarray(y_pred_ret, dtype=float)

        position = np.sign(y_pred_ret)
        strategy_ret = position * y_true_ret

        # Transaction costs applied whenever position changes vs. previous period
        position_change = np.abs(np.diff(np.concatenate([[0], position])))
        cost = position_change * (transaction_cost_bps / 10000.0)
        strategy_ret_net = strategy_ret - cost

        cumulative_ret = np.cumsum(strategy_ret_net)
        cumulative_pnl_multiplier = np.exp(cumulative_ret)

        mean_ret = strategy_ret_net.mean()
        std_ret = strategy_ret_net.std(ddof=1) if len(strategy_ret_net) > 1 else np.nan
        sharpe = (mean_ret / std_ret) * np.sqrt(periods_per_year) if std_ret and std_ret > 0 else np.nan

        running_max = np.maximum.accumulate(cumulative_pnl_multiplier)
        drawdown = (cumulative_pnl_multiplier - running_max) / running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else np.nan

        buy_hold_cum_ret = np.cumsum(y_true_ret)
        buy_hold_final = np.exp(buy_hold_cum_ret[-1]) - 1 if len(buy_hold_cum_ret) > 0 else np.nan

        return {
            'total_return_pct': (cumulative_pnl_multiplier[-1] - 1) * 100 if len(cumulative_pnl_multiplier) else np.nan,
            'annualized_sharpe': sharpe,
            'max_drawdown_pct': max_drawdown * 100 if not np.isnan(max_drawdown) else np.nan,
            'mean_daily_return_pct': mean_ret * 100,
            'volatility_daily_pct': std_ret * 100 if std_ret is not None else np.nan,
            'n_trades': int(position_change.sum()),
            'buy_hold_return_pct': buy_hold_final * 100 if not np.isnan(buy_hold_final) else np.nan,
            'cumulative_pnl_series': cumulative_pnl_multiplier,}

    
    def print_trading_report(self, hit_rate_metrics: Dict[str, float],backtest_metrics: Dict[str, float], label: str = "Model"):
        print(f"\n{'='*60}")
        print(f" Trading Diagnostics — {label}")
        print(f"{'='*60}")
        print(f"Hit rate (directional accuracy): {hit_rate_metrics['hit_rate']*100:.2f}% "
              f"({hit_rate_metrics['n_correct']}/{hit_rate_metrics['n_days']} days)")
        print(f"Annualized Sharpe ratio: {backtest_metrics['annualized_sharpe']:.3f}")
        print(f"Total backtested return: {backtest_metrics['total_return_pct']:.2f}%")
        print(f"Buy-and-hold return (same window): {backtest_metrics['buy_hold_return_pct']:.2f}%")
        print(f"Max drawdown: {backtest_metrics['max_drawdown_pct']:.2f}%")
        print(f"Mean daily return: {backtest_metrics['mean_daily_return_pct']:.4f}%  "
              f"| Daily volatility: {backtest_metrics['volatility_daily_pct']:.4f}%")
        print(f"Number of position changes (trades): {backtest_metrics['n_trades']}")
        print(f"{'='*60}\n")


    def compare_full_vs_baseline(self,y_true_ret: np.ndarray,y_pred_full: np.ndarray, y_pred_baseline: np.ndarray, full_label: str = "Full model (flight + oil)",
                                  baseline_label: str = "Oil-only baseline", transaction_cost_bps: float = 2.0, periods_per_year: int = 252,
                                  plot: bool = True) -> pd.DataFrame:
        
        full_hit = self.calculate_hit_rate(y_true_ret, y_pred_full)
        base_hit = self.calculate_hit_rate(y_true_ret, y_pred_baseline)

        full_bt = self.backtest_strategy(y_true_ret, y_pred_full,
                                          transaction_cost_bps, periods_per_year)
        base_bt = self.backtest_strategy(y_true_ret, y_pred_baseline,
                                          transaction_cost_bps, periods_per_year)

        self.print_trading_report(full_hit, full_bt, label=full_label)
        self.print_trading_report(base_hit, base_bt, label=baseline_label)

        comparison = pd.DataFrame({
            'metric': ['hit_rate_pct', 'annualized_sharpe', 'total_return_pct',
                       'max_drawdown_pct', 'buy_hold_return_pct', 'n_trades'],
            full_label: [
                full_hit['hit_rate'] * 100,
                full_bt['annualized_sharpe'],
                full_bt['total_return_pct'],
                full_bt['max_drawdown_pct'],
                full_bt['buy_hold_return_pct'],
                full_bt['n_trades'],],

            baseline_label: [
                base_hit['hit_rate'] * 100,
                base_bt['annualized_sharpe'],
                base_bt['total_return_pct'],
                base_bt['max_drawdown_pct'],
                base_bt['buy_hold_return_pct'],
                base_bt['n_trades'],],})
        
        comparison['edge (full - baseline)'] = comparison[full_label] - comparison[baseline_label]

        print(comparison.to_string(index=False))
        if plot:
            self._plot_pnl_comparison(full_bt['cumulative_pnl_series'], base_bt['cumulative_pnl_series'], full_label, baseline_label)

        return comparison


    def _plot_pnl_comparison(self, full_pnl: np.ndarray, base_pnl: np.ndarray, full_label: str, baseline_label: str):
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(full_pnl, label=full_label, linewidth=1.5, color='crimson')
        ax.plot(base_pnl, label=baseline_label, linewidth=1.5, color='steelblue', linestyle='--')
        ax.axhline(1.0, color='grey', linewidth=0.8, linestyle=':')
        ax.set_xlabel('Trading day')
        ax.set_ylabel('Cumulative PnL multiplier (1.0 = breakeven)')
        ax.set_title('Backtested cumulative PnL — full model vs. oil-only baseline')
        ax.legend()
        plt.tight_layout()
        plt.savefig('data/processed/diagnostics/pnl_comparison.png', dpi=150)
        plt.show()
