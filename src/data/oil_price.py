import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List
from config.settings import Config

class OilPriceCollector:
    def __init__(self):
        self.symbols = Config.OIL_SYMBOLS
        self.logger = logging.getLogger(__name__)
    
    def fetch_current_prices(self) -> Dict[str, float]:
        prices = {}
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d")
            if not data.empty:
                prices[symbol] = data['Close'].iloc[-1]
                self.logger.info(f"Fetched {symbol}: ${prices[symbol]:.2f}")
                
        return prices

    def fetch_historical_data(self, days: int = 30) -> pd.DataFrame:
        end_data = datetime.now()
        start_date = end_data - timedelta(days=days)

        data_frames = []
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            hist_data = ticker.history(start=start_date, end=end_date)
            hist_data['Symbol'] = symbol
            hist_data['Timestamp'] = hist_data.index
            data_frames.append(hist_data)

        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        
        return pd.DataFrame()