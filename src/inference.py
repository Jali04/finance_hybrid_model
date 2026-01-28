import torch
import pandas as pd
import numpy as np
import os
import sys
import yfinance as yf
from datetime import datetime, timedelta

# Allow imports from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.preprocessor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.model.hybrid import LSTMTransformerHybrid

class InferenceEngine:
    def __init__(self, model_path="best_model.pth", scaler_path="scaler.save", config=None):
        self.config = config or {
            'window_size': 60,
            'hidden_dim': 64,
            'heads': 4,
            'trans_layers': 2
        }
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.processor = DataProcessor(data_dir="data") # Scaler is inside data/
        self.engineer = FeatureEngineer()
        
        # Load Resources
        self._load_model(model_path)
        self.scaler = self._load_scaler(scaler_path)
        
    def _load_model(self, path):
        # We need input_dim. It's usually known or fixed (14 features).
        # ideally we saved this in config, but we can hardcode or infer.
        # Let's assume 14 features for now as we saw in training.
        # Or better: init lazily or generic.
        self.model = LSTMTransformerHybrid(
            input_dim=15, # Updated to 15 to include VIX_Close
            lstm_hidden_dim=self.config['hidden_dim'],
            trans_heads=self.config['heads'],
            trans_layers=self.config['trans_layers']
        ).to(self.device)
        
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            print("Model loaded.")
        else:
            raise FileNotFoundError(f"Model not found at {path}")

    def _load_scaler(self, filename):
        # The DataProcessor load_scaler method sets state internally
        # We assume the scaler file is in 'data/' usually
        # But evaluator used just "scaler.save". Let's try both.
        try:
            self.processor.load_scaler(filename)
        except:
             # Try adding data/ prefix if faulty
             self.processor.load_scaler(os.path.join("data", filename))
        return self.processor

    def predict_next_day(self, ticker="^GSPC"):
        """
        Fetches live data, processes it, and returns prediction for tomorrow.
        """
        # 1. Fetch enough history to cover window + warmup
        # Window 60 + Warmup 26 (MACD) ~ 100 days
        start_date = (datetime.now() - timedelta(days=200)).strftime('%Y-%m-%d')
        
        print(f"Fetching live data for {ticker} since {start_date}...")
        df = yf.download(ticker, start=start_date, progress=False, auto_adjust=True)
        
        # Fetch VIX for the same period
        print(f"Fetching live VIX data since {start_date}...")
        vix_df = yf.download("^VIX", start=start_date, progress=False, auto_adjust=True)
        
        if df.empty or vix_df.empty:
            raise ValueError("No data downloaded for ticker or VIX.")
            
        # Flatten MultiIndex if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
            
        # Merge VIX
        df = self.engineer.add_external_data(df, vix_df, prefix='VIX')
            
        # 2. Feature Engineering
        df = self.engineer.add_all_features(df)
        
        # 3. Get the very last window_size rows
        # We need exactly the last 60 days of VALID features
        if len(df) < self.config['window_size']:
             raise ValueError(f"Not enough data after feature engineering. Need {self.config['window_size']} rows.")
             
        last_window_df = df.iloc[-self.config['window_size']:]
        last_date = last_window_df.index[-1]
        
        # 4. Scaling
        # Transform using the FIXED training scaler
        scaled_window_df = self.processor.transform_data(last_window_df)
        
        # 5. Model Inference
        X = scaled_window_df.values # Shape (60, 14)
        X_tensor = torch.FloatTensor(X).unsqueeze(0).to(self.device) # (1, 60, 14)
        
        with torch.no_grad():
            pred_scaled = self.model(X_tensor).item()
            
        # 6. Interpret Prediction
        # Scale back? 
        # y_real = (y_scaled * scale) + mean
        # We need Log_Return index.
        cols = scaled_window_df.columns.tolist()
        if 'Log_Return' in cols:
            idx = cols.index('Log_Return')
            mean = self.processor.scaler.mean_[idx]
            scale = self.processor.scaler.scale_[idx]
            pred_real = (pred_scaled * scale) + mean
        else:
            pred_real = pred_scaled # Fallback
            
        return {
            'last_date': last_date,
            'prediction_scaled': pred_scaled,
            'prediction_log_return': pred_real,
            'direction': "UP" if pred_real > 0 else "DOWN",
            'confidence': abs(pred_scaled) # Simplified confidence proxy
        }

if __name__ == "__main__":
    # Test
    engine = InferenceEngine(scaler_path="scaler.save")
    result = engine.predict_next_day()
    print("\n--- Live Prediction ---")
    print(result)
