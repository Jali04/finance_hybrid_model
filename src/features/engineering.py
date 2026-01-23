import pandas as pd
import numpy as np

# =============================================================================
# FEATURE ENGINEERING MODULE
# =============================================================================
# This module implements the feature creation for the financial forecasting model.
# Strict causal feature calculation is enforced: features at time 't' depend
# ONLY on data up to time 't'.
# =============================================================================

class FeatureEngineer:
    def __init__(self):
        # We define standard parameters here to ensure consistency.
        # Rationale for choices is provided in the specific methods.
        self.rsi_window = 14
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.bb_window = 20
        self.bb_std = 2

    def calculate_log_returns(self, df, price_col='Close'):
        """
        Calculates Logarithmic Returns.
        
        Why Log Returns?
        1. Additivity: Log returns over time can be summed (R_total = r1 + r2 ...).
        2. Stationarity: Prices are non-stationary (they drift), returns are 
           usually closer to stationary, which is better for Neural Networks.
        """
        # We calculate returns of the Close price.
        # r_t = ln(P_t / P_{t-1})
        df['Log_Return'] = np.log(df[price_col] / df[price_col].shift(1))
        return df

    def calculate_rsi(self, df, window=14, price_col='Close'):
        """
        Calculates Relative Strength Index (RSI).
        
        Parameter Choice: Window = 14
        Rationale: Introduced by J. Welles Wilder in 1978. 
        - 14 days is the industry standard "sweet spot". 
        - A shorter window (e.g., 7) is too noisy and triggers false signals.
        - A longer window (e.g., 28) is too lagging and misses recent momentum shifts.
        """
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        # Handle division by zero if loss is 0 (unlikely in long series but possible)
        rs = rs.replace([np.inf, -np.inf], np.nan) 
        
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def calculate_macd(self, df, fast=12, slow=26, signal=9, price_col='Close'):
        """
        Calculates MACD (Moving Average Convergence Divergence).
        
        Parameter Choice: (12, 26, 9)
        Rationale: Gerald Appel's standard set.
        - 12/26 days: Approximates 2 weeks vs 1 month (in trading days).
          Captures the relationship between short-term momentum and medium-term trend.
        - 9 day signal: Smoothes the MACD line to generate cleaner crossover signals.
        """
        # EMA: Exponential Moving Average gives more weight to recent prices
        ema_fast = df[price_col].ewm(span=fast, adjust=False).mean()
        ema_slow = df[price_col].ewm(span=slow, adjust=False).mean()
        
        df['MACD_Line'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD_Line'].ewm(span=signal, adjust=False).mean()
        # Histogram shows the strength of the trend (divergence between line and signal)
        df['MACD_Hist'] = df['MACD_Line'] - df['MACD_Signal']
        return df

    def calculate_bollinger_bands(self, df, window=20, num_std=2, price_col='Close'):
        """
        Calculates Bollinger Bands.
        
        Parameter Choice: (20, 2)
        Rationale: John Bollinger's standard set.
        - Window 20: Corresponds to a typical trading month (approx 20-22 days). 
          Acts as the intermediate trend baseline.
        - 2 Std Devs: Statistically, ~95% of price action happens within 2 standard 
          deviations in a normal distribution. Prices outside are considered 
          outliers (overbought/oversold) or volatility breakouts.
        
        We calculate 'BB_Width' as a measure of Market Uncertainty/Volatility.
        """
        sma = df[price_col].rolling(window=window).mean()
        std = df[price_col].rolling(window=window).std()
        
        df['BB_Upper'] = sma + (std * num_std)
        df['BB_Lower'] = sma - (std * num_std)
        
        # Feature: Relative position within the bands (0 to 1 ideally, can go outside)
        # (Price - Lower) / (Upper - Lower)
        df['BB_Position'] = (df[price_col] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Feature: Band Width (normalized by price to be comparable over years)
        # Indicates Volatility (High width = High Uncertainty)
        df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / sma
        
        return df

    def add_all_features(self, df):
        """Pipeline to add all features and drop NaNs created by lagging windows."""
        df = df.copy()
        
        df = self.calculate_log_returns(df)
        df = self.calculate_rsi(df, window=self.rsi_window)
        df = self.calculate_macd(df, fast=self.macd_fast, slow=self.macd_slow, signal=self.macd_signal)
        df = self.calculate_bollinger_bands(df, window=self.bb_window, num_std=self.bb_std)
        
        # Drop strictly the rows that contain NaNs due to initial warmup
        # (e.g. first 26 days for MACD, or 20 for BB)
        # We print how many rows are dropped to keep user informed.
        initial_len = len(df)
        df_dropped = df.dropna()
        dropped_count = initial_len - len(df_dropped)
        
        print(f"Feature Engineering: Dropped {dropped_count} rows due to warmup (NaNs).")
        
        return df_dropped

if __name__ == "__main__":
    # Test block
    try:
        raw_path = "data/raw/GSPC_raw.parquet"
        df = pd.read_parquet(raw_path)
        
        engineer = FeatureEngineer()
        df_features = engineer.add_all_features(df)
        
        print("\n--- Feature Engineering Verification ---")
        print("Columns created:", df_features.columns.tolist())
        print("Last 5 rows:")
        print(df_features[['Close', 'Log_Return', 'RSI', 'MACD_Hist', 'BB_Width']].tail())
        
        # Verify no NaNs left
        if df_features.isnull().sum().sum() == 0:
            print("SUCCESS: No NaNs remaining in dataset.")
        else:
            print("WARNING: Dataset still contains NaNs.")
            
    except Exception as e:
        print(f"Test run failed: {e}")
