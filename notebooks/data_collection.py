 

import os
import argparse
import logging
from datetime import datetime
import pandas as pd
import yfinance as yf

# ── Configuration ──────────────────────────────────────────────────────────────

RAW_DATA_DIR = "data/raw"
COMBINED_CSV   = "data/raw/combined_price_data.csv"

# ── Functions ──────────────────────────────────────────────────────────────────

def fetch_and_save(ticker: str, start: str, end: str, interval: str):
    """
    Fetch OHLCV for a single ticker and save to CSV.
    """
    logging.info(f"Fetching {ticker} from {start} to {end} at interval {interval} …")
    df = yf.download(
        tickers=ticker,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
        actions=False
    )
    if df.empty:
        logging.warning(f"No data for {ticker}; skipping.")
        return None

    # Ensure directory exists
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    out_path = os.path.join(RAW_DATA_DIR, f"{ticker}.csv")
    df.to_csv(out_path)
    logging.info(f"Saved {len(df)} rows to {out_path}")
    # Add ticker column for combined
    df = df.assign(Ticker=ticker)
    return df.reset_index()


def main(tickers, start, end, interval):
    """
    Main orchestration: fetch each ticker, then produce combined CSV.
    """
    all_dfs = []
    for t in tickers:
        df = fetch_and_save(t.strip(), start, end, interval)
        if df is not None:
            all_dfs.append(df)

    if all_dfs:
        combined = pd.concat(all_dfs, axis=0, ignore_index=True)
        combined.to_csv(COMBINED_CSV, index=False)
        logging.info(f"Wrote combined file with {len(combined)} rows to {COMBINED_CSV}")
    else:
        logging.error("No data fetched; combined CSV not created.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    p = argparse.ArgumentParser(description="Fetch historical OHLCV for tickers via yfinance.")
    p.add_argument(
        "--tickers", "-t",
        type=str,
        required=True,
        help="Comma-separated list of ticker symbols, e.g. AAPL,MSFT,SPY"
    )
    p.add_argument(
        "--start", "-s",
        type=str,
        default="2015-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--end", "-e",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)"
    )
    p.add_argument(
        "--interval", "-i",
        type=str,
        default="1d",
        choices=["1d","1h","30m","15m","5m"],
        help="Data frequency"
    )

    args = p.parse_args()
    ticker_list = args.tickers.split(",")
    main(ticker_list, args.start, args.end, args.interval)
