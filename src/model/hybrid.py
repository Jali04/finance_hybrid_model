import torch
import torch.nn as nn
import math

# =============================================================================
# HYBRID MODEL ARCHITECTURE
# =============================================================================
# This module implements Phase 4: The LSTM-Transformer Hybrid.
#
# Design Rationale:
# 1. LSTM Layer: Acts as the first processor to capture local sequential 
#    dependencies and provide a strong inductive bias for time-series data.
#    It outputs a sequence of hidden states.
# 2. Transformer Encoder: Takes the LSTM outputs and applies Self-Attention
#    to find global correlations across the entire window (e.g., day 1 vs day 60).
# 3. Pooling: Aggregates the sequence information into a single fixed vector.
#    We use Global Average Pooling here to consider information from all time steps roughly equally.
# 4. Regressor Head: Maps the latent vector to the final scalar prediction.
# =============================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        # Create constant positional encoding matrix with values from sin and cos
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch, d_model)
        return x + self.pe[:x.size(0), :]

class LSTMTransformerHybrid(nn.Module):
    def __init__(self, input_dim, lstm_hidden_dim=64, lstm_layers=1, 
                 trans_heads=4, trans_layers=2, dropout=0.1):
        """
        Args:
            input_dim (int): Number of input features.
            lstm_hidden_dim (int): Hidden size of the LSTM.
            lstm_layers (int): Number of LSTM layers.
            trans_heads (int): Number of attention heads in Transformer.
            trans_layers (int): Number of Transformer Encoder layers.
            dropout (float): Dropout rate.
        """
        super(LSTMTransformerHybrid, self).__init__()
        
        self.model_type = 'LSTM-Transformer'
        
        # --- 1. LSTM Block ---
        # "batch_first=True" creates input shape (Batch, Seq, Feature)
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=lstm_hidden_dim, 
                            num_layers=lstm_layers, 
                            batch_first=True,
                            dropout=(dropout if lstm_layers > 1 else 0))
        
        # --- 2. Transformer Block ---
        # The Transformer needs d_model. We reuse lstm_hidden_dim as d_model.
        self.pos_encoder = PositionalEncoding(lstm_hidden_dim)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=lstm_hidden_dim, 
                                                    nhead=trans_heads, 
                                                    dim_feedforward=lstm_hidden_dim*4, 
                                                    dropout=dropout,
                                                    batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=trans_layers)
        
        # --- 3. Output Head ---
        self.dropout_layer = nn.Dropout(dropout)
        
        # Project from hidden dim to scalar output
        self.fc = nn.Linear(lstm_hidden_dim, 1)

    def forward(self, x):
        """
        Forward pass.
        Args:
           x: Input tensor of shape (Batch, Seq_Len, Input_Dim)
        """
        # 1. LSTM
        # output shape: (Batch, Seq_Len, Hidden_Dim)
        lstm_out, _ = self.lstm(x)
        
        # 2. Positional Encoding
        # PosEncoding expects (Seq, Batch, Feature) usually if batch_first=False
        # But we set batch_first=True for layers. However,standard PE implementation 
        # often assumes (Seq, Batch, feature) for easy slicing.
        # Let's adjust:
        # Transpose to (Seq, Batch, Dim) for PE
        lstm_out = lstm_out.transpose(0, 1)
        lstm_out = self.pos_encoder(lstm_out)
        # Transpose back to (Batch, Seq, Dim) for Transformer (since we used batch_first=True)
        lstm_out = lstm_out.transpose(0, 1)
        
        # 3. Transformer
        # Masking: We usually don't need a causal mask here because we are looking at a 
        # complete PAST window to predict ONE future point. The window itself is already 
        # strictly past data. The transformer is allowed to look at "future" tokens 
        # WITHIN the window (e.g. Day 59 looking at Day 60) because Day 60 is still 
        # in the past relative to the prediction Target (Day 61).
        trans_out = self.transformer_encoder(lstm_out)
        
        # 4. Pooling
        # Strategy: Global Average Pooling over the Sequence Dimension (dim=1)
        # This condenses the whole window's information.
        # trans_out shape: (Batch, Seq, Hidden)
        pooled_out = torch.mean(trans_out, dim=1) 
        
        # 5. Final Prediction
        prediction = self.fc(self.dropout_layer(pooled_out))
        
        # Squeeze to shape (Batch,)
        return prediction.squeeze(-1)

if __name__ == "__main__":
    # Internal Verification
    # create dummy input (Batch=32, Window=60, Features=14)
    dummy_input = torch.randn(32, 60, 14)
    model = LSTMTransformerHybrid(input_dim=14, lstm_hidden_dim=32, trans_heads=4)
    
    output = model(dummy_input)
    
    print("\n--- Model Verification ---")
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}")
    
    if output.shape == (32,):
        print("SUCCESS: Output shape matches (Batch_Size,).")
    else:
        print("FAILURE: Output shape incorrect.")
