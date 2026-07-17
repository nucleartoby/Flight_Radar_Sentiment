import pandas as pd
import numpy as np


class FeatureEngineer:

    def create_flight_features(self, flight_data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()

        flight_data['hour'] = pd.to_datetime(flight_data['timestamp']).dt.hour
        flight_data['day_of_week'] = pd.to_datetime(flight_data['timestamp']).dt.dayofweek
        flight_data['is_weekend'] = flight_data['day_of_week'].isin([5, 6]).astype(int)

        time_windows = ['1h', '6h', '12h', '24h']

        for window in time_windows:
            grouped = flight_data.groupby(pd.Grouper(key='timestamp', freq=window))

            features[f'flight_count_{window}'] = grouped.size()
            features[f'unique_bases_{window}'] = grouped['base_name'].nunique()
            features[f'avg_altitude_{window}'] = grouped['altitude'].mean()
            features[f'avg_speed_{window}'] = grouped['speed'].mean()

            military_flights = flight_data[flight_data['is_military'] == True]
            mil_grouped = military_flights.groupby(pd.Grouper(key='timestamp', freq=window))

            features[f'military_count_{window}'] = mil_grouped.size()
            features[f'military_ratio_{window}'] = (mil_grouped.size() / grouped.size()).fillna(0)

        base_activity = flight_data.groupby(['timestamp', 'base_name']).size().unstack(fill_value=0)
        for base in base_activity.columns:
            features[f'{base.lower().replace(" ", "_")}_activity'] = base_activity[base]

        importance_weights = {
            'Al Udeid Air Base': 1.0,
            'NSA Bahrain (5th Fleet)': 1.0,
            'Al Dhafra Air Base': 0.8,
            'Al Asad Air Base': 0.8,
            'Camp Arifjan': 0.8,
            'Al Tanf Garrison': 0.7,
            'Ali Al Salem Air Base': 0.6,
            'Camp Buehring': 0.6,
            'Al Harir (Erbil) Air Base': 0.5}

        weighted_activity = pd.Series(0.0, index=base_activity.index)
        for base, weight in importance_weights.items():
            if base in base_activity.columns:
                weighted_activity += base_activity[base] * weight

        features['weighted_strategic_activity'] = weighted_activity

        return features.fillna(0)

    def create_oil_features(self, oil_data: pd.DataFrame) -> pd.DataFrame:
        frames = []

        for symbol in ['BZ=F', 'CL=F']:
            symbol_data = oil_data[oil_data['Symbol'] == symbol].copy()
            if symbol_data.empty:
                continue

            symbol_data = symbol_data.set_index('Timestamp').sort_index()
            symbol_data.index = (pd.to_datetime(symbol_data.index, utc=True).normalize().tz_localize(None))
            symbol_clean = symbol.replace('=F', '').lower()
            close = symbol_data['Close'].astype(float)

            s = pd.DataFrame(index=symbol_data.index)
            s[f'{symbol_clean}_price'] = close
            s[f'{symbol_clean}_volume'] = symbol_data['Volume'].astype(float)
            s[f'{symbol_clean}_sma_5'] = close.rolling(5).mean()
            s[f'{symbol_clean}_sma_20'] = close.rolling(20).mean()
            s[f'{symbol_clean}_rsi'] = self._calculate_rsi(close)
            s[f'{symbol_clean}_volatility'] = close.rolling(10).std()
            s[f'{symbol_clean}_pct_change_1d'] = close.pct_change(1)
            s[f'{symbol_clean}_pct_change_5d'] = close.pct_change(5)
            frames.append(s)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, axis=1).ffill().fillna(0)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        prices_float = prices.astype(float)
        delta = prices_float.diff()
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    def build_uncertainty_index(self, flight_features: pd.DataFrame) -> pd.Series:
        df = flight_features.copy()

        mil_ratio = df.get('military_ratio_24h', pd.Series(0.0, index=df.index)).clip(0, 1)

        weighted = df.get('weighted_strategic_activity', pd.Series(0.0, index=df.index)).astype(float)
        rolling_max = weighted.rolling(30, min_periods=1).max().replace(0, np.nan)
        weighted_norm = (weighted / rolling_max).clip(0, 1).fillna(0)

        breadth = df.get('unique_bases_24h', pd.Series(0.0, index=df.index)).astype(float)
        breadth_norm = (breadth / 9).clip(0, 1)

        total_mil = df.get('military_count_24h', pd.Series(0.0, index=df.index)).astype(float)
        rolling_mean = total_mil.rolling(30, min_periods=1).mean()
        rolling_std = total_mil.rolling(30, min_periods=1).std().fillna(1).replace(0, 1)
        z_score = (total_mil - rolling_mean) / rolling_std
        anomaly_norm = ((z_score.clip(-3, 3) + 3) / 6)

        composite = (0.30 * mil_ratio + 0.35 * weighted_norm + 0.15 * breadth_norm + 0.20 * anomaly_norm)

        return (composite * 100).clip(0, 100).rename('uncertainty_index')

    def combine_features(self, flight_features: pd.DataFrame, oil_features: pd.DataFrame) -> pd.DataFrame:
        ff = flight_features.copy()
        of = oil_features.copy()
        ff.index = pd.to_datetime(ff.index).normalize().tz_localize(None)
        of.index = pd.to_datetime(of.index).normalize().tz_localize(None)

        combined = pd.merge(ff, of, left_index=True, right_index=True, how='inner')

        combined['flight_oil_interaction'] = (combined['military_count_24h'] * combined['bz_volatility'])

        lag_periods = [1, 2, 3, 5]
        for period in lag_periods:
            combined[f'military_activity_lag_{period}'] = (combined['military_count_24h'].shift(period))

        return combined.fillna(0)
