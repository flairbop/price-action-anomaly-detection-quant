import pandas as pd
import numpy as np

class Backtester:
    def __init__(self):
        pass
        
    def run_backtest(self, df: pd.DataFrame, anomaly_col: str, strategy: str = 'fade') -> dict:
        """
        Simple backtest: 
        - Fade: Sell if anomaly & price went up, Buy if anomaly & price went down.
        - Follow: Buy if anomaly & price went up, Sell if anomaly & price went down.
        """
        df = df.copy()
        df['signal'] = 0
        
        # Determine direction of anomaly (did price go up or down to cause it?)
        # Using 1-period return sign
        df['direction'] = np.sign(df['log_ret'])
        
        if strategy == 'fade':
            # If price went up (1), we sell (-1). If price went down (-1), we buy (1).
            df.loc[df[anomaly_col], 'signal'] = -df.loc[df[anomaly_col], 'direction']
        elif strategy == 'follow':
            df.loc[df[anomaly_col], 'signal'] = df.loc[df[anomaly_col], 'direction']
            
        # Hold for k periods (e.g., 5 mins)
        # Simple vectorization: shift signal and apply to returns
        # For simplicity, let's assume we hold for 1 period (next period return)
        df['strategy_ret'] = df['signal'].shift(1) * df['log_ret']
        
        # Equity curve
        df['equity'] = (1 + df['strategy_ret'].fillna(0)).cumprod()
        
        # Metrics
        total_ret = df['equity'].iloc[-1] - 1
        sharpe = df['strategy_ret'].mean() / df['strategy_ret'].std() * np.sqrt(252*24*60) # Annualized
        
        # Max Drawdown
        roll_max = df['equity'].cummax()
        drawdown = (df['equity'] - roll_max) / roll_max
        max_dd = drawdown.min()
        
        return {
            'total_return': total_ret,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'equity_curve': df['equity']
        }
