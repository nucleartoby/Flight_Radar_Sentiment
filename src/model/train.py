import os
import pandas as pd
from pathlib import Path
from typing import Tuple


class DataLoader:

    def load_oil_data(self, data_dir: str = "data/raw/oil_prices") -> pd.DataFrame:
        oil_files = list(Path(data_dir).glob("oil_prices_*.csv"))
        if not oil_files:
            raise FileNotFoundError(f"No oil price CSVs found in {data_dir}")

        latest = max(oil_files, key=os.path.getctime)
        df = pd.read_csv(latest)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], utc=True).dt.tz_localize(None)
        return df

    def load_flight_data(self, data_dir: str = "data/raw/flight_data") -> pd.DataFrame:
        flight_files = list(Path(data_dir).glob("flights_*.csv"))
        if not flight_files:
            raise FileNotFoundError(f"No flight CSVs found in {data_dir}")

        frames = []
        for path in flight_files:
            df = pd.read_csv(path)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            frames.append(df)

        return pd.concat(frames, ignore_index=True).sort_values('timestamp').reset_index(drop=True)

    def load(self, oil_dir: str = "data/raw/oil_prices",
             flight_dir: str = "data/raw/flight_data") -> Tuple[pd.DataFrame, pd.DataFrame]:
        oil_data = self.load_oil_data(oil_dir)
        flight_data = self.load_flight_data(flight_dir)
        return oil_data, flight_data
