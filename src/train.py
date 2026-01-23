import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import os
import joblib
import sys

# Allow running this script directly from anywhere (adds project root to path)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.preprocessor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.data.window_generator import WindowGenerator
from src.model.hybrid import LSTMTransformerHybrid

# =============================================================================
# TRAINING SCRIPT
# =============================================================================
# Phase 5: Training and Validation
# This script orchestrates the data flow and model optimization.
# 
# Key Features:
# - Early Stopping: Monitors Validation Loss (2024 data) to prevent overfitting.
# - Model Checkpointing: Saves the model with the best validation performance.
# - Strict Data Separation: Ensures Train/Val/Test are treated independently.
# =============================================================================

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self.processor = DataProcessor()
        self.engineer = FeatureEngineer()
        self.generator = WindowGenerator(window_size=config['window_size'])
        
    def prepare_data(self):
        print("--- 1. Loading & Preprocessing ---")
        # Load Raw Data
        raw_path = "data/raw/GSPC_raw.parquet"
        if not os.path.exists(raw_path):
            raise FileNotFoundError(f"Data not found at {raw_path}. Run loader first.")
            
        df = pd.read_parquet(raw_path)
        
        # Feature Engineering (Calculate Indicators)
        # We do this BEFORE splitting because indicators need some history (warmup)
        # BUT we must ensure the split is done strictly by DATE after features are ready.
        # The 'FeatureEngineer' class handles the "causal" nature (past looking only).
        df = self.engineer.add_all_features(df)
        
        # Split Data
        # Train: 2019-2023, Val: 2024, Test: 2025
        train_df, val_df, test_df = self.processor.split_data(df)
        
        # Scaling
        # Fit ONLY on Train
        # We assume we want to scale all numeric features except maybe 'Log_Return' if it's the target?
        # Actually, scaling inputs is good. Scaling Target is optional but often helps.
        # Let's scale ALL features.
        feature_cols = [c for c in train_df.columns if c not in ['Date', 'Target']] 
        # Note: 'Log_Return' is both an input feature (past) and the basis for the target.
        
        self.processor.fit_scaler(train_df, columns=feature_cols)
        self.processor.save_scaler()
        
        train_scaled = self.processor.transform_data(train_df)
        val_scaled   = self.processor.transform_data(val_df)
        test_scaled  = self.processor.transform_data(test_df)
        
        # Generate Windows (X, y)
        print("--- 2. Generating Sequences ---")
        X_train, y_train, _ = self.generator.create_sequences(train_scaled, "train")
        X_val, y_val, _     = self.generator.create_sequences(val_scaled, "val")
        
        # Convert to Tensor
        self.train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        self.val_loader   = self._create_dataloader(X_val, y_val, shuffle=False)
        
        # Store input dim for model init
        self.input_dim = X_train.shape[2]
        
    def _create_dataloader(self, X, y, shuffle=False):
        if len(X) == 0:
            return None
        tensor_x = torch.Tensor(X)
        tensor_y = torch.Tensor(y)
        dataset = TensorDataset(tensor_x, tensor_y)
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=shuffle)

    def train(self):
        print("--- 3. Initializing Model ---")
        model = LSTMTransformerHybrid(
            input_dim=self.input_dim,
            lstm_hidden_dim=self.config['hidden_dim'],
            lstm_layers=self.config['lstm_layers'],
            trans_heads=self.config['heads'],
            trans_layers=self.config['trans_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=self.config['lr'], weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("--- 4. Starting Training Loop ---")
        for epoch in range(self.config['epochs']):
            model.train()
            train_loss = 0.0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                output = model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(self.train_loader.dataset)
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device)
                    pred = model(X_val)
                    loss = criterion(pred, y_val)
                    val_loss += loss.item() * X_val.size(0)
            
            val_loss /= len(self.val_loader.dataset)
            
            # Scheduler Step
            prev_lr = optimizer.param_groups[0]['lr']
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Logging
            print(f"Epoch {epoch+1}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {current_lr:.2e}")
            
            # Early Stopping & Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), "best_model.pth")
                # print("  -> Saved best model.")
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early Stopping triggered at epoch {epoch+1}")
                    break
        
        print(f"Training Complete. Best Val Loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    # Configuration
    config = {
        'window_size': 60,
        'batch_size': 32,
        'hidden_dim': 64,
        'lstm_layers': 1,
        'heads': 4,
        'trans_layers': 2,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 50,      # Maximum epochs
        'patience': 10     # Early stopping patience
    }
    
    trainer = Trainer(config)
    
    try:
        trainer.prepare_data()
        trainer.train()
    except Exception as e:
        print(f"An error occurred: {e}")
