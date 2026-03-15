import os
import pandas as pd
from pathlib import Path


def load_data():
    oil_files = List(Path("data/raw/oil_prices").glob("oil_prices_*.csv"))
    if not oil_files:
        raise FileNotFoundError("No oil price data found")
    
    latest_oil_file = max(oil_files, key=os.path.getctime)
    oil_data = pd.read_csv(latest_oil_file)
    oil_data['Timestamp'] = pd.to_datetime(oil_data['Timestamp'])

    flight_files = List(Path("data/raw/flight_data")).glob("flights_*.csv")
    if not flight_files:
        raise FileNotFoundError("No flight data found")
    
    flight_data_list = []
    for flight in flight_files:
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        flight_data_list.append(df)

    flight_data = pd.concat(flight_data_list, ignore_index=True)

    return oil_data, flight_data
