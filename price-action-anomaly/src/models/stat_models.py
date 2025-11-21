import pandas as pd
import numpy as np

class StatisticalDetector:
    def __init__(self):
        pass
        
    def detect_zscore_anomalies(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """
        Detects anomalies based on Z-score of returns.
        """
        scores = df['ret_zscore'].abs()
        anomalies = scores > threshold
        return pd.DataFrame({'score': scores, 'is_anomaly': anomalies}, index=df.index)
        
    def detect_bollinger_anomalies(self, df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
        """
        Detects anomalies based on Bollinger Band breakouts/squeezes.
        """
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Breakout
        breakout = (df['close'] > upper_band) | (df['close'] < lower_band)
        
        # Band Width (for squeeze detection - not strictly an anomaly but a regime)
        bandwidth = (upper_band - lower_band) / rolling_mean
        
        # Score could be distance from band
        score = np.abs(df['close'] - rolling_mean) / rolling_std
        
        return pd.DataFrame({'score': score, 'is_anomaly': breakout}, index=df.index)
        
    def detect_cusum_anomalies(self, df: pd.DataFrame, threshold: float = 5.0, drift: float = 0.001) -> pd.DataFrame:
        """
        Detects structural breaks using CUSUM on returns.
        """
        # Simple tabular CUSUM
        s = df['log_ret'].values
        n = len(s)
        g_plus = np.zeros(n)
        g_minus = np.zeros(n)
        
        for i in range(1, n):
            g_plus[i] = max(0, g_plus[i-1] + s[i] - drift)
            g_minus[i] = max(0, g_minus[i-1] - s[i] - drift)
            
        score = np.maximum(g_plus, g_minus)
        anomalies = score > threshold
        
        return pd.DataFrame({'score': score, 'is_anomaly': anomalies}, index=df.index)
