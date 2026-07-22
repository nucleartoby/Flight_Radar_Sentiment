# runs Granger causality battery separately for each geopolitical event type
# Need satelite data to identify parked vs active aircraft
# Run after main

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests, adfuller


MAX_LAG = 15
PRICE_COL = "bz_price"
DATA_PATH = "data/processed/combined_features.csv"
OUT_DIR = Path("data/processed/diagnostics/segmented")


# Event category continuous signal column binary flag column
EVENT_CATEGORIES = {
    "strike": {"signal": "strike_zscore", "flag": "event_strike_flag"},
    "troop_buildup": {"signal": "troop_buildup_ratio", "flag": "event_troop_buildup_flag"},
    "base_surge": {"signal": "base_surge_zscore", "flag": "event_base_surge_flag"},}


MIN_EVENT_DAYS = 10  # minimum days flag must be active before trusted


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.sort_index()


def build_return_series(df: pd.DataFrame, price_col: str = PRICE_COL) -> pd.Series:
    ret = np.log(df[price_col]).diff() # type: ignore
    ret.name = f"{price_col}_ret_1d"
    return ret


def check_stationarity(series: pd.Series, name: str) -> Dict[str, float]:
    clean = series.dropna()
    if clean.nunique() < 2 or len(clean) < 10:
        return {"adf_stat": np.nan, "p_value": np.nan, "stationary": None} # type: ignore

    stat, pval, *_ = adfuller(clean, autolag="AIC")
    return {"adf_stat": stat, "p_value": pval, "stationary": pval < 0.05} # type: ignore


def cross_correlation(signal: pd.Series, target: pd.Series, max_lag: int = MAX_LAG) -> pd.DataFrame:
    aligned = pd.concat([signal, target], axis=1).dropna()
    aligned.columns = ["signal", "target"]

    records = []
    for lag in range(-max_lag, max_lag + 1):
        shifted_signal = aligned["signal"].shift(lag)
        pair = pd.concat([shifted_signal, aligned["target"]], axis=1).dropna()

        if len(pair) < 10 or pair.iloc[:, 0].nunique() < 2:
            corr, pval = np.nan, np.nan
        else:
            corr, pval = stats.pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])

        records.append({"lag": lag, "correlation": corr, "p_value": pval, "n_obs": len(pair),})

    return pd.DataFrame(records)


def granger_causality(
    signal: pd.Series,
    target: pd.Series,
    max_lag: int = MAX_LAG,
    signal_name: str = "signal",) -> pd.DataFrame:
    aligned = pd.concat([target, signal], axis=1).dropna()
    aligned.columns = ["target", "signal"]

    records = []
    if aligned["signal"].nunique() < 2 or len(aligned) < max_lag * 3:
        return pd.DataFrame(records)

    results = grangercausalitytests(aligned[["target", "signal"]], maxlag=max_lag, verbose=False)

    for lag, res in results.items():
        ftest = res[0]["ssr_ftest"]
        records.append({"lag": lag,"f_stat": ftest[0],"p_value": ftest[1],"significant_5pct": ftest[1] < 0.05,})
    return pd.DataFrame(records)


def plot_cross_correlation(ccf_df: pd.DataFrame, name: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["crimson" if p < 0.05 else "steelblue" for p in ccf_df["p_value"].fillna(1)]
    ax.bar(ccf_df["lag"], ccf_df["correlation"], color=colors, width=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(ccf_df["lag"])
    ax.set_xlabel("Lag (days) — positive = signal leads oil return")
    ax.set_ylabel("Pearson correlation")
    ax.set_title(f"Cross-correlation: {name} vs bz_price return (red = p<0.05)")
    plt.tight_layout()
    out_path = out_dir / f"ccf_{name}.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_event_category(
    df: pd.DataFrame,
    category: str,
    cols: Dict[str, str],
    target_ret: pd.Series,
    out_dir: Path,) -> Dict[str, pd.DataFrame]:
    signal_col, flag_col = cols["signal"], cols["flag"]

    if flag_col not in df.columns:
        return {}

    n_active_days = int(df[flag_col].sum())

    check_stationarity(df[signal_col], signal_col)

    # Continuous signal cross-correlation
    ccf_df = cross_correlation(df[signal_col], target_ret)
    ccf_df.to_csv(out_dir / f"ccf_{category}.csv", index=False)
    plot_cross_correlation(ccf_df, category, out_dir)

    granger_df = granger_causality(df[signal_col], target_ret, signal_name=category)
    granger_df.to_csv(out_dir / f"granger_{category}.csv", index=False)

    # oil return distribution ON vs OFF event 
    conditional_rows = []
    for lag in range(0, MAX_LAG + 1):
        flag_shifted = df[flag_col].shift(lag)
        pair = pd.concat([flag_shifted, target_ret], axis=1).dropna()
        pair.columns = ["flag", "ret"]
        on_event = pair.loc[pair["flag"] == 1, "ret"]
        off_event = pair.loc[pair["flag"] == 0, "ret"]

        if len(on_event) >= 5 and len(off_event) >= 5:
            tstat, pval = stats.ttest_ind(on_event, off_event, equal_var=False)
        else:
            tstat, pval = np.nan, np.nan

        conditional_rows.append({
            "lag": lag,
            "mean_ret_on_event": on_event.mean() if len(on_event) else np.nan,
            "mean_ret_off_event": off_event.mean() if len(off_event) else np.nan,
            "n_on_event": len(on_event),
            "n_off_event": len(off_event),
            "t_stat": tstat,
            "p_value": pval,})

    conditional_df = pd.DataFrame(conditional_rows)
    conditional_df.to_csv(out_dir / f"conditional_return_{category}.csv", index=False)

    return {"cross_correlation": ccf_df, "granger": granger_df, "conditional": conditional_df, "n_active_days": n_active_days,} #type: ignore


def summarize(all_results: Dict[str, Dict], out_dir: Path):
    rows = []
    for category, res in all_results.items():
        if not res:
            continue

        ccf = res["cross_correlation"]
        granger = res["granger"]
        conditional = res["conditional"]

        valid_ccf = ccf.dropna(subset=["correlation"])
        if valid_ccf.empty:
            best_lag, best_corr, best_p = np.nan, np.nan, np.nan
        else:
            best_row = valid_ccf.loc[valid_ccf["correlation"].abs().idxmax()]
            best_lag, best_corr, best_p = best_row["lag"], best_row["correlation"], best_row["p_value"]

        n_granger_sig = int(granger["significant_5pct"].sum()) if not granger.empty else 0

        sig_conditional = conditional[conditional["p_value"] < 0.05]
        conditional_sign = np.nan
        if not sig_conditional.empty:
            row0 = sig_conditional.iloc[0]
            conditional_sign = (
                "positive" if row0["mean_ret_on_event"] > row0["mean_ret_off_event"] else "negative")

        rows.append({
            "event_category": category,
            "n_active_days": res["n_active_days"],
            "best_lag": best_lag,
            "best_correlation": round(best_corr, 4) if pd.notna(best_corr) else np.nan,
            "best_corr_pvalue": round(best_p, 4) if pd.notna(best_p) else np.nan,
            "n_granger_significant_lags": n_granger_sig,
            "n_significant_conditional_lags": len(sig_conditional),
            "conditional_effect_direction": conditional_sign,
            "reliable": res["n_active_days"] >= MIN_EVENT_DAYS,})

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "segmented_summary.csv", index=False)
    return summary_df


def plot_category_comparison(summary_df: pd.DataFrame, out_dir: Path):
    plot_df = summary_df.dropna(subset=["best_correlation"])
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["crimson" if r else "lightgrey" for r in plot_df["reliable"]]
    ax.bar(plot_df["event_category"], plot_df["best_correlation"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Strongest lagged correlation with bz_price return")
    ax.set_title("Event-type correlation with oil returns\n" f"(grey = fewer than {MIN_EVENT_DAYS} active days, exploratory only)")
    plt.tight_layout()
    out_path = out_dir / "category_comparison.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()

    target_ret = build_return_series(df)
    check_stationarity(target_ret, f"{PRICE_COL}_ret_1d")

    all_results = {}
    for category, cols in EVENT_CATEGORIES.items():
        all_results[category] = analyze_event_category(df, category, cols, target_ret, OUT_DIR)

    summary_df = summarize(all_results, OUT_DIR)
    plot_category_comparison(summary_df, OUT_DIR)


if __name__ == "__main__":
    main()
