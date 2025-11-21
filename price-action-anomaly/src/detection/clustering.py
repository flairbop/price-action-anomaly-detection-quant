import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
try:
    import hdbscan
except ImportError:
    hdbscan = None

class AnomalyClusterer:
    def __init__(self):
        pass
        
    def cluster_anomalies(self, df: pd.DataFrame, anomaly_mask: pd.Series, feature_cols: list) -> pd.DataFrame:
        """
        Clusters the detected anomalies to identify types.
        """
        # Filter only anomalies
        anomalies = df[anomaly_mask].copy()
        
        if anomalies.empty:
            return pd.DataFrame()
            
        X = anomalies[feature_cols].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Choose clustering algo
        if hdbscan is not None:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=5)
            labels = clusterer.fit_predict(X_scaled)
        else:
            # Fallback
            clusterer = DBSCAN(eps=0.5, min_samples=5)
            labels = clusterer.fit_predict(X_scaled)
            
        anomalies['cluster'] = labels
        return anomalies

    def label_clusters(self, anomalies: pd.DataFrame) -> dict:
        """
        Auto-labels clusters based on feature centroids.
        """
        if 'cluster' not in anomalies.columns:
            return {}
            
        cluster_labels = {}
        unique_clusters = anomalies['cluster'].unique()
        
        for c in unique_clusters:
            if c == -1:
                cluster_labels[c] = "Noise"
                continue
                
            subset = anomalies[anomalies['cluster'] == c]
            
            # Heuristics
            avg_vol = subset['vol_60'].mean()
            avg_trend = subset['trend_strength'].abs().mean()
            avg_velocity = subset['velocity'].abs().mean()
            avg_drawdown = subset['drawdown'].mean() # usually negative
            
            # Simple logic tree
            label = "Unknown Anomaly"
            if avg_vol > 0.002: # High vol
                if avg_velocity > 0.5:
                    label = "Volatility Shock"
                else:
                    label = "Liquidity Collapse"
            elif avg_trend > 0.5:
                label = "Trend Breakout"
            elif avg_drawdown < -0.05:
                label = "Crash / Correction"
            else:
                label = "Microstructure Irregularity"
                
            cluster_labels[c] = label
            
        return cluster_labels
