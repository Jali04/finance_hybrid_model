import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import joblib
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# Allow running this script directly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.preprocessor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.data.window_generator import WindowGenerator
from src.model.hybrid import LSTMTransformerHybrid

# =============================================================================
# EVALUATION SCRIPT
# =============================================================================
# Phase 6: Evaluation & Baselines
#
# Metrics:
# 1. MSE/MAE: Standard regression errors.
# 2. Directional Accuracy (DA): The percentage of time the model correctly 
#    predicts the Sign of the return (Up vs Down). This is often more important
#    than exact magnitude in finance.
#
# Process:
# - Load Data -> Feature Engineering -> Split -> Scale.
# - IMPORTANT: We must 'Denormalize' predictions to check DA correctly, 
#   because the Scaler centers data (Mean=0 becomes new 0). Real 0 return might
#   be at -0.05 or +0.05 in scaled space depending on the training mean.
# =============================================================================

class Evaluator:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.processor = DataProcessor()
        self.engineer = FeatureEngineer()
        self.generator = WindowGenerator(window_size=config['window_size'])
        
    def load_data_and_model(self):
        print("--- Loading Data & Model ---")
        # 1. Data Load
        raw_path = "data/raw/GSPC_raw.parquet"
        df = pd.read_parquet(raw_path)
        
        # Load VIX Data (Fix for KeyError: VIX_Close)
        vix_path = "data/raw/VIX_raw.parquet"
        if os.path.exists(vix_path):
            print("--- Merging VIX data ---")
            vix_df = pd.read_parquet(vix_path)
            df = self.engineer.add_external_data(df, vix_df, prefix='VIX')
            
        df = self.engineer.add_all_features(df)
        _, _, test_df = self.processor.split_data(df)
        
        # 2. Scaler Load
        # DataProcessor defaults data_dir="data", so we just pass filename
        self.processor.load_scaler("scaler.save")
        # Transform Test Data
        test_scaled = self.processor.transform_data(test_df)
        
        # 3. Create Windows
        X_test, y_test_scaled, test_dates = self.generator.create_sequences(test_scaled, "test")
        
        # 4. Model Load
        # We need input_dim from X
        input_dim = X_test.shape[2]
        self.model = LSTMTransformerHybrid(
            input_dim=input_dim,
            lstm_hidden_dim=self.config['hidden_dim'],
            trans_heads=self.config['heads'],
            trans_layers=self.config['trans_layers']
        ).to(self.device)
        
        try:
            self.model.load_state_dict(torch.load("best_model.pth"))
            # print("Model loaded successfully.")
        except FileNotFoundError:
            print("ERROR: 'best_model.pth' not found. Train the model first.")
            sys.exit(1)
            
        self.test_loader = self._create_dataloader(X_test, y_test_scaled)
        
        # Store for analysis
        self.test_dates = test_dates
        self.y_test_scaled = y_test_scaled
        
        # We need to retrieve the 'Log_Return' scaling params to denormalize
        # The scaler stores mean_ and scale_ for all columns.
        # We need to find the index of 'Log_Return'.
        # We can re-derive columns from the dataframe logic
        self.col_names = test_scaled.columns.tolist()
        if 'Log_Return' in self.col_names:
            idx = self.col_names.index('Log_Return')
            self.ret_mean = self.processor.scaler.mean_[idx]
            self.ret_scale = self.processor.scaler.scale_[idx]
            # print(f"Log_Return Params -> Mean: {self.ret_mean:.6f}, Scale: {self.ret_scale:.6f}")
        else:
            print("WARNING: 'Log_Return' not found in columns. Cannot denormalize accurately.")
            self.ret_mean = 0
            self.ret_scale = 1

    def _create_dataloader(self, X, y):
        tensor_x = torch.Tensor(X)
        tensor_y = torch.Tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=False)

    def evaluate(self):
        print("--- Running Inference ---")
        self.model.eval()
        preds = []
        targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch = X_batch.to(self.device)
                output = self.model(X_batch)
                preds.extend(output.cpu().numpy())
                targets.extend(y_batch.numpy())
                
        preds = np.array(preds)
        targets = np.array(targets)
        
        # --- Metrics Calculation ---
        mse = mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)
        
        print(f"\n[Test Set Performance 2025]")
        print(f"MSE (Scaled): {mse:.6f}")
        print(f"MAE (Scaled): {mae:.6f}")
        
        # --- Denormalization & Directional Accuracy ---
        # y_real = (y_scaled * scale) + mean
        preds_real = (preds * self.ret_scale) + self.ret_mean
        targets_real = (targets * self.ret_scale) + self.ret_mean
        
        # Directional Accuracy: Sign(Pred) == Sign(True)
        # We use strict > 0.
        pred_dirs = np.sign(preds_real)
        target_dirs = np.sign(targets_real)
        
        # Remove flat 0s if any (rare in float), or count 0 as no-change
        # Ideally we just check consistency
        da = accuracy_score(target_dirs, pred_dirs)
        print(f"Directional Accuracy: {da*100:.2f}%")
        
        # --- Baselines ---
        # 1. Naive Baseline: Predict 0 return (always same price)
        # In scaled space, 0 return = (0 - mean) / scale
        naive_pred_scaled = (0 - self.ret_mean) / self.ret_scale
        naive_preds = np.full_like(preds, naive_pred_scaled)
        
        naive_mse = mean_squared_error(targets, naive_preds)
        print(f"\n[Baselines]")
        print(f"Naive Model MSE: {naive_mse:.6f}")
        if mse < naive_mse:
            print(">> Model OUTPERFORMS Naive Baseline (MSE).")
        else:
            print(">> Model UNDERPERFORMS Naive Baseline (MSE).")
            
        # Statistical Baseline: Always predict UP (if market is generally bullish)
        # or Always predict DOWN.
        # Let's verify DA of "Always Buy"
        always_buy_da = accuracy_score(target_dirs, np.ones_like(pred_dirs))
        print(f"Naive 'Always Buy' DA: {always_buy_da*100:.2f}%")
        
        if da > always_buy_da:
            print(">> Model OUTPERFORMS 'Always Buy' (DA).")
        else:
            print(">> Model UNDERPERFORMS 'Always Buy' (DA).")

if __name__ == "__main__":
    # Same config as training roughly
    config = {
        'window_size': 60,
        'batch_size': 32,
        'hidden_dim': 64,
        'heads': 4,
        'trans_layers': 2,
    }
    
    evaluator = Evaluator(config)
    evaluator.load_data_and_model()
    evaluator.evaluate()
