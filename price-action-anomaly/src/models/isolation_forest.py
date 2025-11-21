import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class IsolationForestDetector:
    def __init__(self, contamination: float = 0.01):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
        
    def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        # -1 for outliers, 1 for inliers
        preds = self.model.fit_predict(X_scaled)
        
        # Decision function: lower is more abnormal. We invert it for "anomaly score"
        # The lower, the more abnormal. Negative values are outliers.
        # Let's make score positive for anomalies.
        raw_scores = self.model.decision_function(X_scaled)
        anomaly_scores = -raw_scores # Higher is more anomalous
        
        is_anomaly = preds == -1
        
        return pd.DataFrame({'score': anomaly_scores, 'is_anomaly': is_anomaly}, index=df.index)
