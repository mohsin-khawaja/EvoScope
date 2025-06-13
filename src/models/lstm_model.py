import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

class LSTMPricePredictor(nn.Module):
    """
    LSTM model for stock price prediction and feature extraction
    """
    
    def __init__(
        self,
        input_size: int = 10,  # Number of features (OHLCV + technical indicators)
        hidden_size: int = 128,
        num_layers: int = 2,
        output_size: int = 1,  # Predict next price
        dropout: float = 0.2,
        sequence_length: int = 60  # Look back 60 time steps
    ):
        super(LSTMPricePredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Attention mechanism for better feature extraction
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            prediction: Price prediction
            features: Extracted features for RL agent
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention to LSTM outputs
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step output
        last_output = attn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Extract features for RL agent
        features = self.dropout(last_output)
        
        # Prediction head
        x = self.relu(self.fc1(features))
        x = self.dropout(x)
        prediction = self.fc2(x)
        
        return prediction, features
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features for RL agent without prediction
        """
        _, features = self.forward(x)
        return features


class TechnicalIndicatorLSTM(nn.Module):
    """
    Specialized LSTM for technical indicator analysis
    """
    
    def __init__(
        self,
        input_size: int = 20,  # More technical indicators
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = 5,  # Multiple technical signals
        sequence_length: int = 30
    ):
        super(TechnicalIndicatorLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for technical indicator signals
        
        Returns:
            signals: Technical trading signals (0-1 range)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        signals = self.sigmoid(self.output_layer(last_output))
        return signals


def create_lstm_model(config: dict) -> LSTMPricePredictor:
    """
    Factory function to create LSTM model with configuration
    """
    return LSTMPricePredictor(
        input_size=config.get('input_size', 10),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 2),
        output_size=config.get('output_size', 1),
        dropout=config.get('dropout', 0.2),
        sequence_length=config.get('sequence_length', 60)
    ) 