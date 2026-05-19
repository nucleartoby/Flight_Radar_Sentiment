import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from src.data.opensky_api import OpenSkyAPICollector
from src.data.oil_price import OilPriceCollector
from src.data.base_monitor import BaseMonitor
from src.feature_engineering.features import FeatureEngineer
from src.model.model import OilPricePredictor
from src.model.evaluation import ModelEvaluator
from src.model.train import DataLoader
from config.settings import Config


MIN_TRAINING_ROWS = 30


def setup_logging() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("pipeline")


def create_directories():
    for path in ["data/raw/flight_data","data/raw/oil_prices","data/processed","models","logs",]:
        Path(path).mkdir(parents=True, exist_ok=True)



def collect_flights(logger: logging.Logger):
    logger.info("Connecting to OpenSky Network…")
    collector = OpenSkyAPICollector()

    status = collector.get_api_status()
    auth_label = status.get("auth_method", "anonymous")
    logger.info(f"OpenSky status: {status.get('api_accessible')} ({auth_label})")

    if not status.get("api_accessible"):
        logger.warning("OpenSky API not reachable, skipping live collection.")
        collector.close()
        return [], 0

    all_flights = collector.get_middle_east_flights()
    logger.info(f"Retrieved {len(all_flights)} total flights in region.")

    monitor = BaseMonitor()

    for flight in all_flights:
        near, base_name = monitor.is_near_base(flight.get("latitude", 0), flight.get("longitude", 0))
        flight["near_base"] = near
        flight["base_name"] = base_name if near else "None"
        flight["is_military"] = collector._is_military_flight(flight)
        flight["total_flights_in_snapshot"] = len(all_flights)

    military = [f for f in all_flights if f.get("is_military")]
    logger.info(f"Military / government: {len(military)} of {len(all_flights)} "f"({100 * len(military) / max(len(all_flights), 1):.1f}%)")

    collector.close()
    return military, len(all_flights)


def save_flight_snapshot(military: list, total: int, logger: logging.Logger):
    if not military:
        logger.info(f"No military aircraft to save (total in region: {total}).")
        return
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"data/raw/flight_data/flights_{ts}.csv"
    pd.DataFrame(military).to_csv(path, index=False)
    logger.info(f"Saved {len(military)} military records {path}")


def collect_oil_prices(logger: logging.Logger) -> pd.DataFrame:
    logger.info("Fetching historical oil prices (365 days)…")
    collector = OilPriceCollector()
    df = collector.fetch_historical_data(days=365)

    if df.empty:
        logger.warning("No oil price data returned.")
        return df

    ts = datetime.now().strftime("%Y%m%d")
    path = f"data/raw/oil_prices/oil_prices_{ts}.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved oil price data ({len(df)} rows) -> {path}")
    return df


def load_accumulated_data(logger: logging.Logger):
    loader = DataLoader()

    oil_data = pd.DataFrame()
    oil_data = loader.load_oil_data()
    logger.info(f"Loaded oil data: {len(oil_data)} rows.")

    flight_data = pd.DataFrame()
    flight_data = loader.load_flight_data()
    logger.info(f"Loaded flight data: {len(flight_data)} rows across "
    f"{flight_data['timestamp'].dt.date.nunique()} days.")

    return oil_data, flight_data


def build_features(flight_data: pd.DataFrame,oil_data: pd.DataFrame,logger: logging.Logger):
    fe = FeatureEngineer()

    flight_features = pd.DataFrame()
    if not flight_data.empty:
        logger.info("Engineering flight features…")
        flight_features = fe.create_flight_features(flight_data)
        uncertainty = fe.build_uncertainty_index(flight_features)
        flight_features["uncertainty_index"] = uncertainty
        logger.info(f"Uncertainty index — mean: {uncertainty.mean():.1f}, "f"max: {uncertainty.max():.1f}, min: {uncertainty.min():.1f}")
        flight_features.to_csv("data/processed/flight_features.csv")

    oil_features = pd.DataFrame()
    if not oil_data.empty:
        logger.info("Engineering oil features…")
        oil_features = fe.create_oil_features(oil_data)
        oil_features.to_csv("data/processed/oil_features.csv")

    combined = pd.DataFrame()
    if not flight_features.empty and not oil_features.empty:
        combined = fe.combine_features(flight_features, oil_features)
        combined.to_csv("data/processed/combined_features.csv")
        logger.info(f"Combined feature matrix: {combined.shape}")

    return flight_features, oil_features, combined


def train_and_evaluate(combined: pd.DataFrame, logger: logging.Logger):
    if combined.shape[0] < MIN_TRAINING_ROWS:
        logger.warning(f"Only {combined.shape[0]} rows of merged data — need at least "
            f"{MIN_TRAINING_ROWS} to train. Keep running the collector daily "
            "to accumulate enough history.")
        return

    predictor = OilPricePredictor()
    X, y, df = predictor.prepare_data(combined)

    if len(X) < MIN_TRAINING_ROWS:
        logger.warning("Not enough rows after preparing data. Skipping training.")
        return

    logger.info(f"Training on {len(X)} samples…")
    results = predictor.train_models(X, y, df)

    evaluator = ModelEvaluator()
    for _, res in results.items():
        metrics = evaluator.calculate_metrics(
            y[int(len(y) * Config.TRAIN_TEST_SPLIT):],
            res["model"].predict(
                predictor.scaler.transform(X[int(len(X) * Config.TRAIN_TEST_SPLIT):])
            ) if predictor.best_model_name != "xgboost" else
            res["model"].predict(
                __import__("xgboost").DMatrix(
                    predictor.scaler.transform(X[int(len(X) * Config.TRAIN_TEST_SPLIT):])
                )
            ),
        )
        evaluator.print_evaluation_report(metrics)

    model_path = f"models/oil_predictor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    predictor.save_model(model_path)
    logger.info(f"Best model ({predictor.best_model_name}) saved -> {model_path}")

    return predictor, results


def plot_uncertainty_index(flight_features: pd.DataFrame, logger: logging.Logger):
    if flight_features.empty or "uncertainty_index" not in flight_features.columns:
        logger.info("No uncertainty index to plot.")
        return

    _, ax = plt.subplots(figsize=(14, 5))
    idx = flight_features.index
    values = flight_features["uncertainty_index"]

    ax.fill_between(idx, values, alpha=0.25, color="crimson")
    ax.plot(idx, values, color="crimson", linewidth=1.2, label="Uncertainty Index")
    ax.axhline(values.mean(), color="grey", linestyle="--", linewidth=0.8,label=f"Mean ({values.mean():.1f})")

    ax.set_title("Geopolitical Uncertainty Index — Military Aviation Activity (Middle East)")
    ax.set_ylabel("Index (0–100)")
    ax.set_xlabel("Date")
    ax.legend()
    plt.tight_layout()

    out_path = "data/processed/uncertainty_index.png"
    plt.savefig(out_path, dpi=150)
    logger.info(f"Uncertainty index chart saved -> {out_path}")
    plt.show()


def print_military_aircraft(flights: list):
    military = [f for f in flights if f.get("is_military")]

    divider = "=" * 80
    print(f"\n{divider}")
    print(f"  MILITARY / GOVERNMENT AIRCRAFT IN MIDDLE EAST REGION")
    print(divider)

    if not military:
        print(f"  No military or government aircraft identified in this snapshot.")
        print(f"  ({len(flights)} total civilian aircraft tracked in region)")
    else:
        print(f"  {'CALLSIGN':<12} {'ICAO24':<8} {'COUNTRY':<22} {'ALT (m)':<10} "
              f"{'SPD (kt)':<10} {'NEAR BASE':<28} {'REASON'}")
        print("-" * 80)

        for f in sorted(military, key=lambda x: x.get("callsign") or ""):
            callsign = (f.get("callsign") or "N/A").strip()[:11]
            icao24 = (f.get("icao24")   or "N/A")[:7]
            country = (f.get("origin_country") or "N/A")[:21]
            alt = f.get("altitude")
            spd = f.get("speed")
            base = f.get("base_name") or "None"
            reason = (f.get("military_reason") or "Pattern match")[:30]

            alt_str = f"{int(alt):,}" if alt is not None else "---"
            spd_str = f"{int(spd)}"   if spd is not None else "---"

            print(f"  {callsign:<12} {icao24:<8} {country:<22} {alt_str:<10} "
                  f"{spd_str:<10} {base:<28} {reason}")

        print(f"\n  {len(military)} military/government  |  {len(flights)} total in region")

    print(f"{divider}\n")


def main():
    logger = setup_logging()
    create_directories()
    logger.info("=" * 60)
    logger.info("Flight Radar Sentiment — pipeline start")
    logger.info("=" * 60)

    # 1. Live collection
    flights, total_in_region = collect_flights(logger)
    save_flight_snapshot(flights, total_in_region, logger)

    # 2. Oil prices
    collect_oil_prices(logger)

    # 3. Load everything saved so far
    oil_data, flight_data = load_accumulated_data(logger)

    # 4. Features
    flight_features, _, combined = build_features(
        flight_data, oil_data, logger
    )

    # 5. Train (only if enough data)
    if not combined.empty:
        train_and_evaluate(combined, logger)

    # 6. Plot the index
    plot_uncertainty_index(flight_features, logger)

    # 7. Print military aircraft summary
    print_military_aircraft(flights)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
