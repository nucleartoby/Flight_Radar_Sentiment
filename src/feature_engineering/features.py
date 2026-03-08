import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FeatureEngineer:

    def create_flight_features(self, flight_data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame()

        flight_data['hour'] = pd.to_datetime(flight_data['timestamp']).dt.hour
        flight_data['day_of_week'] = pd.to_datetime(flight_data['timestamp']).dt.dayofweek
        flight_data['is_weekend'] = flight_data['day_of_week'].isin([5, 6]).astype(int)

        time_windows = ['1H', '6H', '12H', '24H']
        
        for window in time_windows:
            grouped = flight_data.groupby(pd.Grouper(key='timestamp', freq=window))
            
            features[f'flight_count_{window}'] = grouped.size()
            features[f'unique_bases_{window}'] = grouped['base_name'].nunique()
            features[f'avg_altitude_{window}'] = grouped['altitude'].mean()
            features[f'avg_speed_{window}'] = grouped['speed'].mean()

            military_flights = flight_data[flight_data['is_military'] == True]
            mil_grouped = military_flights.groupby(pd.Grouper(key='timestamp', freq=window))
            
            features[f'military_count_{window}'] = mil_grouped.size()
            features[f'military_ratio_{window}'] = (
                mil_grouped.size() / grouped.size()
            ).fillna(0)

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
            'Al Harir (Erbil) Air Base': 0.5
        }
        
        weighted_activity = 0
        for base, weight in importance_weights.items():
            if base in base_activity.columns:
                weighted_activity += base_activity[base] * weight
        
        features['weighted_strategic_activity'] = weighted_activity
        
        return features.fillna(0)
    
    def create_oil_features(self, oil_data: pd.DataFrame) -> pd.DataFrame:
        features = pd.DataFrame(index=oil_data.index)
        
        for symbol in ['BZ=F', 'CL=F']:
            symbol_data = oil_data[oil_data['Symbol'] == symbol]
            if symbol_data.empty:
                continue
                
            symbol_clean = symbol.replace('=F', '').lower()

            features[f'{symbol_clean}_price'] = symbol_data['Close']
            features[f'{symbol_clean}_volume'] = symbol_data['Volume']

            features[f'{symbol_clean}_sma_5'] = symbol_data['Close'].rolling(5).mean()
            features[f'{symbol_clean}_sma_20'] = symbol_data['Close'].rolling(20).mean()
            features[f'{symbol_clean}_rsi'] = self._calculate_rsi(symbol_data['Close'])

            features[f'{symbol_clean}_volatility'] = symbol_data['Close'].rolling(10).std()

            features[f'{symbol_clean}_pct_change_1d'] = symbol_data['Close'].pct_change(1)
            features[f'{symbol_clean}_pct_change_5d'] = symbol_data['Close'].pct_change(5)
            
        return features.fillna(method='ffill').fillna(0)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def combine_features(self, flight_features: pd.DataFrame, oil_features: pd.DataFrame) -> pd.DataFrame:
        combined = pd.merge(flight_features, oil_features, 
                           left_index=True, right_index=True, how='inner')

        combined['flight_oil_interaction'] = (
            combined['military_count_24H'] * combined['bz_volatility']
        )
        
        lag_periods = [1, 2, 3, 5]
        for period in lag_periods:
            combined[f'military_activity_lag_{period}'] = (
                combined['military_count_24H'].shift(period)
            )
        
        return combined.fillna(0)
    