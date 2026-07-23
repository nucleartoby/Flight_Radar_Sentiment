import logging
from typing import List

import pandas as pd
import yfinance as yf

from config.settings import Config
from src.tracking import db


class OilIntradayCollector:
    def __init__(self, conn, symbols: List[str] = None):
        self.conn = conn
        self.symbols = symbols or Config.OIL_SYMBOLS
        self.logger = logging.getLogger("oil_intraday")

    def refresh(self) -> int:
        total = 0
        for symbol in self.symbols:
            hist = yf.Ticker(symbol).history(
                period=Config.OIL_INTRADAY_PERIOD,
                interval=Config.OIL_INTRADAY_INTERVAL,)

            if hist is None or hist.empty:
                self.logger.warning(f"No intraday bars returned for {symbol}.")
                continue

            rows = self._to_rows(symbol, hist)
            db.upsert_oil(self.conn, rows)
            total += len(rows)
            self.logger.info(f"Oil {symbol}: upserted {len(rows)} hourly bars.")

        return total

    @staticmethod
    def _to_rows(symbol: str, hist: pd.DataFrame) -> List[dict]:
        rows = []
        for idx, r in hist.iterrows():
            # idx is a tz-aware Timestamp; store as UTC unix seconds.
            ts = int(pd.Timestamp(idx).tz_convert("UTC").timestamp())
            rows.append({
                "symbol": symbol,
                "ts": ts,
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": float(r.get("Volume", 0) or 0),})
        return rows