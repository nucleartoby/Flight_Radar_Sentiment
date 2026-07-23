import time
import logging
import argparse
from pathlib import Path

from config.settings import Config
from src.data.flightradar_api import FlightRadarCollector
from src.data.base_monitor import BaseMonitor
from src.tracking import db
from src.tracking.snapshot import collect_snapshot
from src.tracking.tracker import FlightTracker
from src.tracking.oil_intraday import OilIntradayCollector


def setup_logging() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        handlers=[
            logging.FileHandler(Config.TRACKING_LOG_FILE),
            logging.StreamHandler(),],)
    return logging.getLogger("tracker.run")


def main():
    parser = argparse.ArgumentParser(description="Military flight lifecycle tracker")
    parser.add_argument("--minutes", type=float, default=None,
                        help="Stop after this many minutes (default: run forever)")
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Flight lifecycle tracker starting")
    logger.info("=" * 60)

    conn = db.connect()
    db.init_schema(conn)

    collector = FlightRadarCollector()
    monitor = BaseMonitor()
    tracker = FlightTracker(conn)
    oil = OilIntradayCollector(conn)

    # Prime oil so the very first opened events have a start price to as-of match.
    oil.refresh()

    deadline = time.time() + args.minutes * 60 if args.minutes else None
    last_oil = time.time()

    while True:
        cycle_start = time.time()
        now = int(cycle_start)

        snapshot = collect_snapshot(collector, monitor)
        counts = tracker.update(snapshot, now)
        logger.info(
            f"cycle: tracked={counts['tracked']} opened={counts['opened']} "
            f"closed={counts['closed']} active={counts['active']}")

        if time.time() - last_oil >= Config.OIL_REFRESH_SEC:
            oil.refresh()
            last_oil = time.time()

        if deadline and time.time() >= deadline:
            logger.info("Reached --minutes deadline, shutting down.")
            break

        elapsed = time.time() - cycle_start
        time.sleep(max(0, Config.POLL_INTERVAL_SEC - elapsed))

    tracker.close_all(int(time.time()))
    collector.close()
    conn.close()
    logger.info("Tracker stopped.")


if __name__ == "__main__":
    main()