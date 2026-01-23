import numpy as np
import pandas as pd

# =============================================================================
# SLIDING WINDOW GENERATOR
# =============================================================================
# This module implements Phase 3: Converting time-series data into fixed-size
# sequences for the Neural Network.
#
# Key Concepts:
# 1. Input (X): A generic window of past 'T' days.
# 2. Target (y): The "Next Day Return". We specifically use Log-Return as defined
#    in Phase 2.
# 3. Boundary Handling:
#    - For TRAINING, we only use data completely within the Training set.
#    - For VALIDATION/TEST, we usually need the ending of the previous set to 
#      form the first few windows. This is allowed because the "Input" acts as 
#      context (past), while the "Target" (prediction) remains strictly within 
#      the evaluation period.
# =============================================================================

class WindowGenerator:
    def __init__(self, window_size=60, target_col='Log_Return'):
        self.window_size = window_size
        self.target_col = target_col

    def create_sequences(self, df, purpose="train"):
        """
        Creates (X, y) pairs from the dataframe.
        
        Args:
            df (pd.DataFrame): The dataframe containing features and target.
                               Must be sorted chronologically.
            purpose (str): 'train', 'val', or 'test'. Used for logging.
            
        Returns:
            X (np.array): Shape (Num_Samples, Window_Size, Num_Features)
            y (np.array): Shape (Num_Samples, ) - The target values.
            indices (pd.Index): The timestamps corresponding to the TARGET y.
        """
        data = df.values
        target_idx = df.columns.get_loc(self.target_col)
        
        X_list = []
        y_list = []
        date_list = []
        
        # We need at least (window_size + 1) rows to create one sample
        # (window_size for X, +1 for the next day target y)
        if len(df) <= self.window_size:
            print(f"[{purpose}] WARNING: Not enough data to create a single sequence.")
            return np.array([]), np.array([]), []

        # Loop explanation:
        # i starts at `window_size`.
        # We take X = data[i - window : i] -> The past window
        # We take y = data[i, target_idx] -> Current day (which is next day relative to window end)
        # Ideally, we want to predict Return at T+1 given T.
        # So if our data is [Day1, Day2, ... Day60, Day61]
        # X = [Day1 ... Day60]
        # y = Log_Return of Day61
        
        # Iterating through the array
        # `i` is the index of the TARGET
        for i in range(self.window_size, len(data)):
            # Extract window
            window = data[i - self.window_size : i]
            
            # Extract target
            target = data[i, target_idx]
            
            X_list.append(window)
            y_list.append(target)
            date_list.append(df.index[i])
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        print(f"[{purpose}] Created {len(X)} sequences. X Shape: {X.shape}, y Shape: {y.shape}")
        
        return X, y, pd.DatetimeIndex(date_list)

if __name__ == "__main__":
    # Internal Verification Test
    print("\n--- Window Generator Verification ---")
    
    # Create dummy data: 100 days
    dates = pd.date_range(start="2023-01-01", periods=100, freq='D')
    df = pd.DataFrame({
        'Feature1': np.arange(100),
        'Log_Return': np.random.randn(100)
    }, index=dates)
    
    gen = WindowGenerator(window_size=10)
    X, y, idx = gen.create_sequences(df, purpose="test_run")
    
    # Verification
    # If we have 100 days. Window 10.
    # First sequence: X=[0..9], y=Day 10 (index 10).
    # Last sequence: y=Day 99.
    # Total samples = 100 - 10 = 90.
    
    print(f"Expected Samples: 90. Actual: {len(X)}")
    print(f"First Target Date: {idx[0]} (Should be Jan 11)")
    print(f"X[0] last value (Feature1): {X[0, -1, 0]} (Should be 9)")
    
    if len(X) == 90 and X[0, -1, 0] == 9:
        print("SUCCESS: Sliding Window logic correct.")
    else:
        print("FAILURE: Dimensions mismatch.")
