import pandas as pd
import numpy as np
from scipy.stats import entropy, skew, kurtosis

class FeatureEngineer:
    def __init__(self):
        pass
        
    def compute_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Price Action Features
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
        
        # Rolling Volatility (multiple windows)
        for window in [15, 60, 240]: # 15m, 1h, 4h
            df[f'vol_{window}'] = df['log_ret'].rolling(window=window).std()
            
        # Vol of Vol
        df['vol_of_vol'] = df['vol_60'].rolling(window=60).std()
        
        # Return Z-scores (short term)
        df['ret_zscore'] = (df['log_ret'] - df['log_ret'].rolling(60).mean()) / df['log_ret'].rolling(60).std()
        
        # 2. Microstructure Features
        # Price Velocity (change in close per min)
        df['velocity'] = df['close'].diff()
        
        # Acceleration
        df['acceleration'] = df['velocity'].diff()
        
        # Rolling Intra-bar Range
        df['range'] = df['high'] - df['low']
        df['rolling_range_mean'] = df['range'].rolling(60).mean()
        
        # Rolling Drawdown (from rolling max)
        rolling_max = df['close'].rolling(window=240).max()
        df['drawdown'] = (df['close'] - rolling_max) / rolling_max
        
        # Trend Strength (ADX-like proxy or simple MA slope)
        # Using slope of 60-period MA
        ma_60 = df['close'].rolling(60).mean()
        df['trend_strength'] = ma_60.diff()
        
        # 3. Statistical Features
        # Rolling Entropy (of returns, discretized)
        # Simplified: rolling skew/kurtosis
        df['skew_60'] = df['log_ret'].rolling(60).apply(lambda x: skew(x, nan_policy='omit'), raw=True)
        df['kurt_60'] = df['log_ret'].rolling(60).apply(lambda x: kurtosis(x, nan_policy='omit'), raw=True)
        
        # Drop NaNs created by rolling windows
        df.dropna(inplace=True)
        
        return df
