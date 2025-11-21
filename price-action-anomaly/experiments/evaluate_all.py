import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from data.data_loader import DataLoader
from data.preprocess import Preprocessor
from data.feature_engineering import FeatureEngineer
from models.stat_models import StatisticalDetector
from models.isolation_forest import IsolationForestDetector
from models.lstm_autoencoder import LSTMAutoencoderDetector
from detection.clustering import AnomalyClusterer
from pipelines.backtest import Backtester
from utils.plotting import plot_anomalies

def main():
    print("Starting End-to-End Price Action Anomaly Detection...")
    
    # 1. Data Loading
    loader = DataLoader()
    df = loader.generate_synthetic_data(n_days=30)
    print(f"Generated data: {df.shape}")
    
    # 2. Preprocessing
    preprocessor = Preprocessor()
    df = preprocessor.clean_data(df)
    # df = preprocessor.resample_data(df) # Already 1min
    
    # 3. Feature Engineering
    fe = FeatureEngineer()
    df = fe.compute_features(df)
    feature_cols = [c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
    print(f"Features computed: {len(feature_cols)}")
    
    # 4. Statistical Detection
    stat_detector = StatisticalDetector()
    zscore_res = stat_detector.detect_zscore_anomalies(df)
    df['anomaly_zscore'] = zscore_res['is_anomaly']
    print(f"Z-Score Anomalies: {df['anomaly_zscore'].sum()}")
    
    # 5. ML Detection (Isolation Forest)
    iso_detector = IsolationForestDetector(contamination=0.01)
    iso_res = iso_detector.fit_predict(df, feature_cols)
    df['anomaly_iso'] = iso_res['is_anomaly']
    print(f"Isolation Forest Anomalies: {df['anomaly_iso'].sum()}")
    
    # 6. Deep Learning Detection (LSTM AE)
    lstm_detector = LSTMAutoencoderDetector(epochs=5) # Low epochs for demo speed
    lstm_res = lstm_detector.fit_predict(df, feature_cols)
    df['anomaly_lstm'] = lstm_res['is_anomaly']
    print(f"LSTM AE Anomalies: {df['anomaly_lstm'].sum()}")
    
    # 7. Clustering
    clusterer = AnomalyClusterer()
    # Cluster LSTM anomalies
    clustered_df = clusterer.cluster_anomalies(df, df['anomaly_lstm'], feature_cols)
    if not clustered_df.empty:
        labels = clusterer.label_clusters(clustered_df)
        print("Anomaly Clusters:", labels)
    
    # 8. Backtesting
    backtester = Backtester()
    res = backtester.run_backtest(df, 'anomaly_lstm', strategy='fade')
    print("Backtest Results (Fade Strategy on LSTM Anomalies):")
    print(res)
    
    # 9. Save Results
    df.to_csv("data/processed/final_results.csv")
    print("Results saved to data/processed/final_results.csv")
    
    print("Mission Complete.")

if __name__ == "__main__":
    main()
