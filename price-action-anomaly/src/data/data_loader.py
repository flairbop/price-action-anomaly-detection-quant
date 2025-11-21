import numpy as np
import pandas as pd
from typing import Optional, Tuple, List

class DataLoader:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

    def generate_synthetic_data(self, 
                                n_days: int = 365, 
                                freq: str = '1min',
                                seed: int = 42) -> pd.DataFrame:
        """
        Generates synthetic price data with GBM, regimes, and jumps.
        """
        np.random.seed(seed)
        
        # Time index
        dates = pd.date_range(start='2023-01-01', periods=n_days*24*60, freq=freq)
        n = len(dates)
        
        # Parameters
        mu = 0.0001  # Drift
        sigma_low = 0.0005 # Low vol regime
        sigma_high = 0.0020 # High vol regime
        
        # Regime switching (Markov Chain-like)
        regimes = np.zeros(n)
        current_regime = 0
        prob_switch = 0.001
        
        for i in range(1, n):
            if np.random.random() < prob_switch:
                current_regime = 1 - current_regime
            regimes[i] = current_regime
            
        # Volatility vector
        sigma = np.where(regimes == 0, sigma_low, sigma_high)
        
        # Jumps (Poisson)
        jump_prob = 0.0005
        jumps = np.random.normal(0, 0.01, n) * (np.random.random(n) < jump_prob)
        
        # Returns
        returns = np.random.normal(mu, sigma, n) + jumps
        
        # Intraday seasonality (higher vol at open/close)
        # Simplified: just amplify returns based on time of day
        # Assuming 24h market for simplicity or crypto
        time_of_day = dates.hour * 60 + dates.minute
        seasonality = 1.0 + 0.5 * np.sin(2 * np.pi * time_of_day / (24*60))
        returns = returns * seasonality
        
        # Price
        price = 100 * np.exp(np.cumsum(returns))
        
        # Create OHLCV (synthetic)
        # High/Low derived from volatility
        high = price * (1 + np.abs(np.random.normal(0, sigma/2, n)))
        low = price * (1 - np.abs(np.random.normal(0, sigma/2, n)))
        open_p = price * (1 + np.random.normal(0, sigma/4, n))
        close_p = price # Use calculated price as close
        
        # Ensure High >= Low, High >= Open, High >= Close, Low <= Open, Low <= Close
        high = np.maximum(high, np.maximum(open_p, close_p))
        low = np.minimum(low, np.minimum(open_p, close_p))
        
        volume = np.abs(np.random.normal(1000, 500, n)) * (1 + regimes) # More volume in high vol
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_p,
            'high': high,
            'low': low,
            'close': close_p,
            'volume': volume
        })
        
        df.set_index('timestamp', inplace=True)
        return df

    def load_data(self, filepath: Optional[str] = None) -> pd.DataFrame:
        if filepath:
            # Load real data logic here
            pass
        else:
            print("No data file provided. Generating synthetic data...")
            return self.generate_synthetic_data()
