# Cross-correlation and Granger causality diagnostics between military flight
# activity signals (military_count_24h, weighted_strategic_activity) and
# Brent oil futures (bz_price) returns, at lags 0-10 days.

## Run after the main

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger("diagnostics")


MAX_LAG = 10
SIGNAL_COLS = ["military_count_24h", "weighted_strategic_activity"]
PRICE_COL = "bz_price"
DATA_PATH = "data/processed/combined_features.csv"
OUT_DIR = Path("data/processed/diagnostics")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.sort_index()

    return df


def build_return_series(df: pd.DataFrame, price_col: str = PRICE_COL) -> pd.Series:
    ret = np.log(df[price_col]).diff()
    ret.name = f"{price_col}_ret_1d"

    return ret


def check_stationarity(series: pd.Series, name: str) -> Dict[str, float]:
    clean = series.dropna()
    stat, pval, *_ = adfuller(clean, autolag="AIC")
    result = {"adf_stat": stat, "p_value": pval, "stationary": pval < 0.05}
    logger.info(
        f"ADF test [{name}]: stat={stat:.4f}, p={pval:.4f}, "
        f"stationary={'YES' if result['stationary'] else 'NO'}")
    
    return result


def cross_correlation(signal: pd.Series, target: pd.Series, max_lag: int = MAX_LAG) -> pd.DataFrame:
    aligned = pd.concat([signal, target], axis=1).dropna()
    aligned.columns = ["signal", "target"]

    records = []
    for lag in range(-max_lag, max_lag + 1):
        shifted_signal = aligned["signal"].shift(lag)
        pair = pd.concat([shifted_signal, aligned["target"]], axis=1).dropna()
        if len(pair) < 10:
            corr, pval = np.nan, np.nan
        else:
            corr, pval = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        records.append({"lag": lag, "correlation": corr, "p_value": pval, "n_obs": len(pair)})

    return pd.DataFrame(records)


def granger_causality(signal: pd.Series, target: pd.Series, max_lag: int = MAX_LAG, signal_name: str = "signal") -> pd.DataFrame:
    aligned = pd.concat([target, signal], axis=1).dropna()
    aligned.columns = ["target", "signal"]

    records = []
    results = grangercausalitytests(aligned[["target", "signal"]], maxlag=max_lag, verbose=False)
    for lag, res in results.items():
        ftest = res[0]["ssr_ftest"]
        records.append({
            "lag": lag,
            "f_stat": ftest[0],
            "p_value": ftest[1],
            "significant_5pct": ftest[1] < 0.05,})

    return pd.DataFrame(records)


def plot_cross_correlation(ccf_df: pd.DataFrame, signal_name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["crimson" if p < 0.05 else "steelblue" for p in ccf_df["p_value"].fillna(1)]
    ax.bar(ccf_df["lag"], ccf_df["correlation"], color=colors, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Lag (days) — positive = signal leads oil return")
    ax.set_ylabel("Pearson correlation")
    ax.set_title(f"Cross-correlation: {signal_name} vs bz_price return\n(red bars = p < 0.05)")
    plt.tight_layout()
    out_path = out_dir / f"ccf_{signal_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved cross-correlation plot -> {out_path}")


def plot_granger_pvalues(granger_df: pd.DataFrame, signal_name: str, out_dir: Path):
    if granger_df.empty:

        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["crimson" if s else "steelblue" for s in granger_df["significant_5pct"]]
    ax.bar(granger_df["lag"], granger_df["p_value"], color=colors, width=0.6)
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="p = 0.05")
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Granger test p-value")
    ax.set_title(f"Granger causality: {signal_name} -> bz_price return\n(red bars = significant at 5%)")
    ax.legend()
    plt.tight_layout()
    out_path = out_dir / f"granger_{signal_name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Granger p-value plot -> {out_path}")


def run_diagnostics(df: pd.DataFrame, signal_cols: List[str] = SIGNAL_COLS,price_col: str = PRICE_COL, max_lag: int = MAX_LAG,out_dir: Path = OUT_DIR) -> Dict[str, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    target_ret = build_return_series(df, price_col)
    check_stationarity(target_ret, f"{price_col}_ret_1d")

    all_results = {}
    for col in signal_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' not found in dataframe — skipping.")
            continue

        logger.info(f"\n{'='*70}\nDiagnostics for: {col}\n{'='*70}")
        signal = df[col].astype(float)
        check_stationarity(signal, col)

        ccf_df = cross_correlation(signal, target_ret, max_lag)
        ccf_df.to_csv(out_dir / f"ccf_{col}.csv", index=False)
        plot_cross_correlation(ccf_df, col, out_dir)

        best_row = ccf_df.loc[ccf_df["correlation"].abs().idxmax()]
        logger.info(
            f"Strongest correlation for {col}: lag={int(best_row['lag'])}, "
            f"r={best_row['correlation']:.4f}, p={best_row['p_value']:.4f}")


        granger_df = granger_causality(signal, target_ret, max_lag, signal_name=col)
        granger_df.to_csv(out_dir / f"granger_{col}.csv", index=False)
        plot_granger_pvalues(granger_df, col, out_dir)

        sig_lags = granger_df[granger_df["significant_5pct"]]["lag"].tolist()
        if sig_lags:
            logger.info(f"Granger-significant lags (p<0.05) for {col}: {sig_lags}")
        else:
            logger.info(f"No Granger-significant lags found for {col} up to lag {max_lag}.")

        all_results[col] = {"cross_correlation": ccf_df, "granger": granger_df}

    return all_results


def summarize(all_results: Dict[str, Dict[str, pd.DataFrame]], out_dir: Path = OUT_DIR):
    rows = []
    for col, res in all_results.items():
        ccf = res["cross_correlation"]
        granger = res["granger"]
        best = ccf.loc[ccf["correlation"].abs().idxmax()]
        n_sig_granger = int(granger["significant_5pct"].sum()) if not granger.empty else 0
        rows.append({
            "signal": col,
            "best_lag": int(best["lag"]),
            "best_correlation": round(best["correlation"], 4),
            "best_corr_pvalue": round(best["p_value"], 4),
            "n_granger_significant_lags": n_sig_granger,})
        
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "diagnostic_summary.csv", index=False)
    logger.info(f"\n{summary_df.to_string(index=False)}")
    return summary_df


def main():
    logger.info("Loading combined feature data...")
    df = load_data()
    logger.info(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

    all_results = run_diagnostics(df)
    summarize(all_results)
    logger.info(f"Diagnostics complete. Outputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
