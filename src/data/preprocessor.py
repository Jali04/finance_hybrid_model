import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# =============================================================================
# DATA PREPROCESSING MODULE
# =============================================================================
# This module maps to the "Look-Ahead Bias Avoidance" requirements of Phase 1.
# It handles:
# 1. Chronological Splitting: Dividing data into fixed time periods.
# 2. Causal Scaling: Learning scaling parameters ONLY from the Training set.
# =============================================================================

class DataProcessor:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.scaler = StandardScaler()
        self.is_fitted = False

    def split_data(self, df):
        """
        Splits the dataframe into Train, Validation, and Test sets based on strict
        year boundaries.
        
        Ranges:
        - Train: 2019 up to 2023 (inclusive)
        - Val:   2024
        - Test:  2025
        
        Args:
            df (pd.DataFrame): The dataframe with a DatetimeIndex.
            
        Returns:
            tuple: (train_df, val_df, test_df)
        """
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        print("--- Splitting Data ---")
        # Train: 2019-01-01 to 2023-12-31
        # strict mask 
        train_mask = (df.index.year >= 2019) & (df.index.year <= 2023)
        
        # Val: 2024-01-01 to 2024-12-31
        val_mask = (df.index.year == 2024)
        
        # Test: 2025-01-01 to 2025-12-31 (and beyond if available)
        test_mask = (df.index.year == 2025)

        train_df = df.loc[train_mask].copy()
        val_df = df.loc[val_mask].copy()
        test_df = df.loc[test_mask].copy()

        # LOGGING for verification
        # We print these to confirm no overlap.
        print(f"Train Set: {train_df.index.min()} to {train_df.index.max()} (n={len(train_df)})")
        print(f"Val Set:   {val_df.index.min()} to {val_df.index.max()} (n={len(val_df)})")
        print(f"Test Set:  {test_df.index.min()} to {test_df.index.max()} (n={len(test_df)})")

        return train_df, val_df, test_df

    def fit_scaler(self, train_df, columns=None):
        """
        Fits the scaler standardizing features by removing the mean and scaling to unit variance.
        CRITICAL: This must ONLY be called on the Training data.
        
        Args:
            train_df (pd.DataFrame): Training data.
            columns (list): List of column names to scale. If None, scales all.
        """
        if columns is None:
            columns = train_df.columns
        
        print("--- Fitting Scaler ---")
        print("Using Training Data stats for Mean and Std.")
        self.scaler.fit(train_df[columns])
        self.is_fitted = True
        self.columns_to_scale = columns

    def transform_data(self, df):
        """
        Applies the learned scaling to a dataframe.
        Crucially, this uses the fixed params from fit_scaler (Train stats).
        Out-of-bound values in Val/Test are preserved (no clipping), reflecting 
        real regime shifts.
        
        Args:
            df (pd.DataFrame): Data to transform.
            
        Returns:
            pd.DataFrame: Scaled dataframe (with original index/columns preserved).
        """
        if not self.is_fitted:
            raise ValueError("Scaler has not been fitted regarding strict causality rules! Call fit_scaler on Train data first.")
        
        if df.empty:
            return df

        # We keep the dataframe structure
        scaled_values = self.scaler.transform(df[self.columns_to_scale])
        df_scaled = df.copy()
        df_scaled[self.columns_to_scale] = scaled_values
        
        return df_scaled
    
    def save_scaler(self, filename="scaler.save"):
        """Persist the scaler and column metadata."""
        path = os.path.join(self.data_dir, filename)
        joblib.dump({
            'scaler': self.scaler,
            'columns': self.columns_to_scale
        }, path)
        print(f"Scaler saved to {path}")

    def load_scaler(self, filename="scaler.save"):
        path = os.path.join(self.data_dir, filename)
        if os.path.exists(path):
            checkpoint = joblib.load(path)
            # Support both old (just scaler) and new (dict) formats for robustness
            if isinstance(checkpoint, dict) and 'scaler' in checkpoint:
                self.scaler = checkpoint['scaler']
                self.columns_to_scale = checkpoint['columns']
            else:
                self.scaler = checkpoint
                # Fallback: if we loaded just a scaler, we can't easily recover columns 
                # without external knowledge. This path might fail transform_data.
                # But for now, we assume the new format will be used.
            
            self.is_fitted = True
            print(f"Scaler loaded from {path}")
        else:
            print(f"Warning: Scaler not found at {path}")

# Example usage/Testing block
if __name__ == "__main__":
    # Load raw data to test the splitting logic
    try:
        raw_path = "data/raw/GSPC_raw.parquet"
        df = pd.read_parquet(raw_path)
        
        processor = DataProcessor()
        train, val, test = processor.split_data(df)
        
        # Test Scaling (just on 'Close' as dummy example)
        # In Phase 2 we will scale actual features, not price usually (except for non-stationary inputs if used)
        processor.fit_scaler(train, columns=['Close'])
        
        train_scaled = processor.transform_data(train)
        val_scaled = processor.transform_data(val)
        
        print("\n--- Scaling Verification ---")
        print(f"Train Mean (should be ~0): {train_scaled['Close'].mean():.4f}")
        print(f"Train Std (should be ~1): {train_scaled['Close'].std():.4f}")
        print(f"Val Mean (unbiased): {val_scaled['Close'].mean():.4f}")
        
    except Exception as e:
        print(f"Test run failed: {e}")
