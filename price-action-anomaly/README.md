# Price Action Anomaly Detection

An advanced, end-to-end Price Action Anomaly Detection Project.

## Overview
This system autonomously:
1. Loads/Generates price data.
2. Engineers advanced features.
3. Detects anomalies using Statistical, ML, and Deep Learning methods.
4. Classifies anomalies with context awareness.
5. Visualizes results with multi-asset heatmaps.
6. Backtests trading strategies.

## Structure
- `data/`: Data storage.
- `notebooks/`: Jupyter notebooks for exploration and analysis.
- `src/`: Source code for data processing, models, detection, and pipelines.
- `experiments/`: Scripts to run experiments.

## Quick Start

### 1. Setup Environment
Ensure you are in the project root directory:
```bash
cd price-action-anomaly
```

Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
The main entry point runs data generation, training, detection, and backtesting in one go:
```bash
python experiments/evaluate_all.py
```

### 3. Explore Results
- **Console Output**: Summaries of detected anomalies and backtest performance.
- **Data**: Processed data and results are saved to `data/processed/final_results.csv`.
- **Visualizations**: Plots are generated during execution (and can be saved if configured).

## Project Modules
- **Data**: `src/data` handles synthetic generation (GBM, Regimes) and feature engineering.
- **Models**: `src/models` contains Statistical (Z-score), ML (IsolationForest), and DL (LSTM AE) detectors.
- **Detection**: `src/detection` handles clustering and labeling of anomalies.
- **Backtest**: `src/pipelines/backtest.py` evaluates trading strategies on detected anomalies.
