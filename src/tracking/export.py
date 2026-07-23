import logging
from pathlib import Path

import pandas as pd

from config.settings import Config
from src.tracking import db

OUT_PATH = "data/processed/flight_events.csv"


def export(conn=None, out_path: str = OUT_PATH) -> pd.DataFrame:
    own = conn is None
    conn = conn or db.connect()
    try:
        df = pd.read_sql_query(
            "SELECT * FROM flight_events WHERE status = 'closed' ORDER BY first_seen_ts",
            conn,)
    finally:
        if own:
            conn.close()

    for col in ("first_seen_ts", "last_seen_ts"):
        if col in df.columns:
            df[col.replace("_ts", "_utc")] = pd.to_datetime(df[col], unit="s", utc=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logging.getLogger("export").info(f"Exported {len(df)} closed events -> {out_path}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d = export()
    print(f"Wrote {len(d)} closed flight events to {OUT_PATH}")
