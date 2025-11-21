import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

class LOFDetector:
    def __init__(self, contamination: float = 0.01, n_neighbors: int = 20):
        self.model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        self.scaler = StandardScaler()
        
    def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        X = df[feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        
        preds = self.model.fit_predict(X_scaled)
        
        # Negative outlier factor: larger is better (inliers). 
        # We want higher score = anomaly.
        # negative_outlier_factor_ is approx -1 for inliers, very small negative for outliers.
        # Actually, sklearn docs: "The higher, the more normal."
        # So we negate it.
        anomaly_scores = -self.model.negative_outlier_factor_
        
        is_anomaly = preds == -1
        
        return pd.DataFrame({'score': anomaly_scores, 'is_anomaly': is_anomaly}, index=df.index)
