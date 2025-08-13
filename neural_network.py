import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple

class TemporalConvolutionalModule(nn.Module):
    """Temporal Convolutional Module (TCM) for processing sequential data"""
    
    def __init__(self, input_channels: int, output_channels: int, kernel_size: int = 3, 
                 dilation: int = 1, dropout: float = 0.1):
        super(TemporalConvolutionalModule, self).__init__()
        
        # Temporal convolution with causal padding
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(input_channels, output_channels, kernel_size, 
                             padding=padding, dilation=dilation)
        
        # Normalization and regularization
        self.norm = nn.BatchNorm1d(output_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None
        
    def forward(self, x):
        # x shape: (batch_size, channels, sequence_length)
        residual = x
        
        # Apply temporal convolution
        out = self.conv(x)
        
        # Remove future information (causal)
        if out.size(2) > x.size(2):
            out = out[:, :, :x.size(2)]
            
        # Normalization and activation
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        if self.residual is not None:
            residual = self.residual(residual)
            
        return out + residual

class CNNFeatureExtractor(nn.Module):
    """CNN for extracting spatial features from market data"""
    
    def __init__(self, input_features: int = 7, hidden_channels: int = 64):
        super(CNNFeatureExtractor, self).__init__()
        
        # 1D CNN layers for feature extraction
        self.conv1 = nn.Conv1d(input_features, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels * 2)
        self.bn3 = nn.BatchNorm1d(hidden_channels)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        # Transpose for conv1d: (batch_size, features, sequence_length)
        x = x.transpose(1, 2)
        
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        return x  # Shape: (batch_size, hidden_channels, sequence_length)

class PPONetwork(nn.Module):
    """PPO Actor-Critic Network with CNN-TCM architecture"""
    
    def __init__(self, input_features: int = 7, sequence_length: int = 96, 
                 hidden_dim: int = 256, action_dim: int = 3):
        super(PPONetwork, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_features = input_features
        
        # CNN Feature Extractor
        self.cnn = CNNFeatureExtractor(input_features, hidden_channels=64)
        
        # Temporal Convolutional Modules
        self.tcm1 = TemporalConvolutionalModule(64, 128, kernel_size=3, dilation=1)
        self.tcm2 = TemporalConvolutionalModule(128, 128, kernel_size=3, dilation=2)
        self.tcm3 = TemporalConvolutionalModule(128, 64, kernel_size=3, dilation=4)
        
        # Global pooling to aggregate temporal information
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers
        self.fc_shared = nn.Sequential(
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, features)
        batch_size = x.size(0)
        
        # CNN feature extraction
        cnn_features = self.cnn(x)  # (batch_size, 64, sequence_length)
        
        # Temporal convolutional processing
        tcm_out = self.tcm1(cnn_features)
        tcm_out = self.tcm2(tcm_out)
        tcm_out = self.tcm3(tcm_out)
        
        # Global pooling to get fixed-size representation
        pooled = self.global_pool(tcm_out)  # (batch_size, 64, 1)
        pooled = pooled.squeeze(-1)  # (batch_size, 64)
        
        # Shared fully connected layers
        shared_features = self.fc_shared(pooled)
        
        # Actor and Critic outputs
        action_logits = self.actor(shared_features)
        state_value = self.critic(shared_features)
        
        return action_logits, state_value
    
    def get_action_and_value(self, x, action=None):
        """Get action probabilities and state value"""
        action_logits, value = self.forward(x)
        
        # Create action distribution
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value
    
    def get_value(self, x):
        """Get state value only"""
        _, value = self.forward(x)
        return value

class GridTradingFeatureProcessor:
    """Process raw market data into features suitable for grid trading"""
    
    @staticmethod
    def normalize_features(data):
        """Normalize features for better training stability"""
        # Price normalization (using log returns)
        close_prices = data[:, :, 0]  # Close price is first feature
        log_returns = torch.log(close_prices[:, 1:] / close_prices[:, :-1])
        
        # Pad the first timestep
        log_returns = torch.cat([torch.zeros_like(log_returns[:, :1]), log_returns], dim=1)
        
        # Replace close price with log returns
        normalized_data = data.clone()
        normalized_data[:, :, 0] = log_returns
        
        # Volume normalization (log transform)
        volumes = data[:, :, 1]
        normalized_data[:, :, 1] = torch.log(volumes + 1e-8)
        
        # VWAP difference normalization
        vwap_diff = data[:, :, 3]
        normalized_data[:, :, 3] = vwap_diff / close_prices
        
        # RSI is already normalized (0-100), scale to (0-1)
        rsi = data[:, :, 5]
        normalized_data[:, :, 5] = rsi / 100.0
        
        return normalized_data
    
    @staticmethod
    def calculate_grid_levels(current_price, grid_spacing=0.0005):
        """Calculate grid levels based on current price"""
        # Grid spacing of 0.05% as specified
        upper_grid = current_price * (1 + grid_spacing)
        lower_grid = current_price * (1 - grid_spacing)
        
        return upper_grid, lower_grid
    
    @staticmethod
    def calculate_position_features(current_price, entry_price, position_type):
        """Calculate position-related features"""
        if entry_price is None:
            return 0.0, 0.0  # No position
            
        pnl_pct = (current_price - entry_price) / entry_price
        
        if position_type == 'long':
            unrealized_pnl = pnl_pct
        elif position_type == 'short':
            unrealized_pnl = -pnl_pct
        else:
            unrealized_pnl = 0.0
            
        return unrealized_pnl, abs(pnl_pct)

def test_network():
    """Test the neural network architecture"""
    # Test parameters
    batch_size = 32
    sequence_length = 96
    input_features = 7
    
    # Create test data
    test_input = torch.randn(batch_size, sequence_length, input_features)
    
    # Initialize network
    network = PPONetwork(input_features, sequence_length)
    
    # Test forward pass
    print("Testing PPO Network...")
    print(f"Input shape: {test_input.shape}")
    
    # Test action and value prediction
    action, log_prob, entropy, value = network.get_action_and_value(test_input)
    
    print(f"Action shape: {action.shape}")
    print(f"Log prob shape: {log_prob.shape}")
    print(f"Entropy shape: {entropy.shape}")
    print(f"Value shape: {value.shape}")
    
    # Test value-only prediction
    value_only = network.get_value(test_input)
    print(f"Value-only shape: {value_only.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\nNetwork test completed successfully!")

if __name__ == "__main__":
    test_network()