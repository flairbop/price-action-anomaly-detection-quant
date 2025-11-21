import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_anomalies(df: pd.DataFrame, anomaly_col: str, title: str = "Anomalies"):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df['close'], label='Price', alpha=0.6)
    
    anomalies = df[df[anomaly_col]]
    plt.scatter(anomalies.index, anomalies['close'], color='red', label='Anomaly', s=20)
    
    plt.title(title)
    plt.legend()
    plt.show()

def plot_heatmap(anomaly_matrix: pd.DataFrame, title: str = "Multi-Asset Anomaly Heatmap"):
    plt.figure(figsize=(12, 8))
    sns.heatmap(anomaly_matrix.T, cmap='viridis', cbar_kws={'label': 'Anomaly Score'})
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Asset")
    plt.show()
