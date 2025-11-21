import pandas as pd
import numpy as np

class Preprocessor:
    def __init__(self):
        pass
        
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans data: handles missing values, removes extreme spikes.
        """
        df = df.copy()
        
        # Forward fill missing values
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        
        # Remove extreme spikes (e.g., > 20% move in 1 min is likely bad data for most assets)
        # Using a simple return threshold
        df['pct_change'] = df['close'].pct_change()
        mask = df['pct_change'].abs() < 0.20
        df = df[mask].copy()
        df.drop(columns=['pct_change'], inplace=True)
        
        return df

    def resample_data(self, df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
        """
        Resamples data to target frequency.
        """
        # Assuming df is already indexed by timestamp
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        resampled = df.resample(freq).agg(agg_dict)
        resampled.dropna(inplace=True) # Drop empty bins
        return resampled
