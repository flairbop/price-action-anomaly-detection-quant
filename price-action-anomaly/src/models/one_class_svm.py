import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

class OneClassSVMDetector:
    def __init__(self, nu: float = 0.01):
        self.model = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        self.scaler = StandardScaler()
        
    def fit_predict(self, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
        # OCSVM scales poorly with N, so we might need to subsample for training if N is huge
        # For this demo, we assume N is manageable or we train on a window
        X = df[feature_cols].values
        
        # Train on first 50% or just train on all (unsupervised)
        # Standard approach: fit on all
        X_scaled = self.scaler.fit_transform(X)
        
        preds = self.model.fit_predict(X_scaled)
        
        # Decision function
        raw_scores = self.model.decision_function(X_scaled)
        anomaly_scores = -raw_scores # Higher is more anomalous
        
        is_anomaly = preds == -1
        
        return pd.DataFrame({'score': anomaly_scores, 'is_anomaly': is_anomaly}, index=df.index)
