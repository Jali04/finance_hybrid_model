import yfinance as yf
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# DATA LOADER SCRIPT
# =============================================================================
# This script handles the acquisition of historical market data.
# It is designed to strictly avoid look-ahead bias by using adjusted prices
# directly from the source (yfinance) and ensuring clear chronological order.
#
# Key features:
# 1. Download data via yfinance with `auto_adjust=True` to handle splits/dividends.
# 2. Save data to a raw storage format (Parquet) for efficient subsequent loading.
# 3. Download extra history (starting from 2010) to ensure sufficient warmup
#    periods for technical indicators (e.g., 200-day moving averages) before
#    the official training start in 2019.
# =============================================================================

def download_data(ticker="^GSPC", start_date="2010-01-01", end_date=None, save_dir="data/raw"):
    """
    Downloads historical market data for a given ticker.

    Args:
        ticker (str): The symbol to download (default: S&P 500 index '^GSPC').
        start_date (str): Start date string (YYYY-MM-DD). We default to 2010 to have
                          ample buffer for calculating lagging indicators (like MA200)
                          before the 2019 training start.
        end_date (str): End date string (YYYY-MM-DD). If None, downloads to current date.
        save_dir (str): Directory to save the downloaded file.

    Returns:
        pd.DataFrame: The downloaded and processed DataFrame.
    """
    
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"[{datetime.now()}] Starting download for ticker: {ticker}")
    print(f"[{datetime.now()}] Time range: {start_date} -> {end_date if end_date else 'Now'}")
    
    # -------------------------------------------------------------------------
    # FETCH DATA
    # -------------------------------------------------------------------------
    # We use `auto_adjust=True`.
    # Why?
    # Financial data often contain stock splits and dividend payments.
    # - Without adjustment, a 2:1 split would look like a 50% price crash.
    # - Dividends cause artificial price drops on the ex-dividend date.
    # `auto_adjust=True` ensures all OHLC (Open, High, Low, Close) prices are
    # backward-adjusted. This creates a smooth price series representing the 
    # true economic value history, preventing the model from learning false
    # volatility signatures from corporate actions.
    # -------------------------------------------------------------------------
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    
    # Check if data was actually downloaded
    if df.empty:
        print(f"ERROR: No data found for {ticker}. Please check the symbol or internet connection.")
        return None

    # -------------------------------------------------------------------------
    # DATA INSPECTION & CLEANING
    # -------------------------------------------------------------------------
    # yfinance returns a DataFrame with a DateTime index.
    # We ensure the index is sorted chronologically. This is critical for
    # time-series splitting later.
    df = df.sort_index()
    
    # Handle MultiIndex columns (sometimes yfinance returns (Price, Ticker) as columns)
    # We flatten it to just (Price) if that happens, or ensure standard naming.
    if isinstance(df.columns, pd.MultiIndex):
        # Drop the "Ticker" level if it exists, keeping just OHLCV
        df.columns = df.columns.get_level_values(0)
    
    # Standardize column names to lowercase/consistent format if needed
    # (Optional, but good practice). Here we expect: Open, High, Low, Close, Volume
    print(f"[{datetime.now()}] Download complete. Shape: {df.shape}")
    print(f"[{datetime.now()}] Columns: {df.columns.tolist()}")
    print(f"[{datetime.now()}] Date Range: {df.index.min()} to {df.index.max()}")

    # -------------------------------------------------------------------------
    # SAVE RAW DATA
    # -------------------------------------------------------------------------
    # We save as Parquet.
    # Why?
    # 1. Faster I/O than CSV.
    # 2. Preserves data types (Index remains DatetimeIndex, floats remain floats).
    # 3. Smaller file size (compression).
    # -------------------------------------------------------------------------
    filename = f"{ticker.replace('^', '')}_raw.parquet"
    filepath = os.path.join(save_dir, filename)
    
    try:
        df.to_parquet(filepath)
        print(f"[{datetime.now()}] Successfully saved data to: {filepath}")
    except Exception as e:
        print(f"ERROR: Failed to save file. Reason: {e}")

    return df

if __name__ == "__main__":
    # Example usage:
    # Download S&P 500 (^GSPC) data.
    # We purposefully start early (2010) to have history for initial indicators 
    # when the Training set (2019) begins.
    data = download_data(ticker="^GSPC", start_date="2010-01-01")
    
    # Optional: Display first few rows to verify structure
    if data is not None:
        print("\n--- Head of Data ---")
        print(data.head())
        print("\n--- Tail of Data ---")
        print(data.tail())
