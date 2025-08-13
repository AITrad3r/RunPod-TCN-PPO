#!/usr/bin/env python3
"""
Configuration file for PPO Grid Trading System

This file contains all configurable parameters for the trading system,
including neural network architecture, PPO hyperparameters, trading parameters,
and system settings.
"""

import os
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class NetworkConfig:
    """Neural network architecture configuration"""
    # Input/Output dimensions
    input_features: int = 10  # 7 market features + 3 position features
    sequence_length: int = 96  # 4 days of hourly data
    action_dim: int = 3  # HOLD, BUY, SELL
    
    # CNN parameters
    cnn_channels: list = None  # [32, 64, 128]
    cnn_kernel_sizes: list = None  # [3, 3, 3]
    
    # TCM parameters
    tcm_channels: int = 128
    tcm_kernel_size: int = 3
    tcm_dropout: float = 0.1
    
    # Dense layers
    hidden_dim: int = 256
    dropout_rate: float = 0.2
    
    def __post_init__(self):
        if self.cnn_channels is None:
            self.cnn_channels = [32, 64, 128]
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]

@dataclass
class PPOConfig:
    """PPO algorithm configuration"""
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    
    # PPO specific
    clip_epsilon: float = 0.2  # PPO clipping parameter
    entropy_coef: float = 0.01  # Entropy bonus coefficient
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training parameters
    n_steps: int = 1024  # Steps per update
    batch_size: int = 64  # Minibatch size
    n_epochs: int = 10  # PPO epochs per update
    max_episode_steps: int = 500  # Max steps per episode
    
    # Checkpointing
    save_freq: int = 100  # Save model every N episodes
    eval_freq: int = 50  # Evaluate every N episodes
    
    # Early stopping
    patience: int = 200  # Episodes without improvement before stopping
    min_delta: float = 0.001  # Minimum improvement threshold

@dataclass
class TradingConfig:
    """Trading environment configuration"""
    # Grid parameters
    initial_balance: float = 10000.0  # Starting balance in USDT
    grid_levels: int = 10  # Number of grid levels
    grid_spacing: float = 0.0005  # 0.05% spacing between grid levels
    position_size_pct: float = 0.1  # 10% of balance per position
    
    # Risk management
    max_position_size: float = 0.5  # Max 50% of balance in positions
    stop_loss_pct: float = 0.05  # 5% stop loss
    take_profit_pct: float = 0.02  # 2% take profit
    
    # Transaction costs
    maker_fee: float = 0.001  # 0.1% maker fee
    taker_fee: float = 0.001  # 0.1% taker fee
    slippage: float = 0.0005  # 0.05% slippage
    
    # Technical indicators
    rsi_period: int = 14
    vwap_period: int = 20
    
    # Reward function weights
    profit_weight: float = 1.0
    sharpe_weight: float = 0.5
    drawdown_weight: float = -0.3
    trade_frequency_weight: float = 0.1
    risk_adjusted_weight: float = 0.2

@dataclass
class DataConfig:
    """Data configuration"""
    # Database
    db_path: str = "bitcoin_data.db"
    symbol: str = "BTCUSDT"
    
    # Data splits
    train_split: float = 0.6  # 60% for training
    val_split: float = 0.2    # 20% for validation
    test_split: float = 0.2   # 20% for testing
    
    # Preprocessing
    normalize_features: bool = True
    feature_window: int = 96  # 4 days of hourly data
    
    # Features to use
    price_features: list = None  # ['open', 'high', 'low', 'close']
    volume_features: list = None  # ['volume']
    technical_features: list = None  # ['vwap', 'rsi']
    
    def __post_init__(self):
        if self.price_features is None:
            self.price_features = ['open', 'high', 'low', 'close']
        if self.volume_features is None:
            self.volume_features = ['volume']
        if self.technical_features is None:
            self.technical_features = ['vwap', 'rsi']

@dataclass
class SystemConfig:
    """System configuration"""
    # Paths
    model_dir: str = "models"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_console: bool = True
    
    # Device
    device: str = "auto"  # 'auto', 'cpu', 'cuda'
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Performance
    num_workers: int = 4  # For data loading
    pin_memory: bool = True

class Config:
    """Main configuration class that combines all configs"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        # Initialize default configurations
        self.network = NetworkConfig()
        self.ppo = PPOConfig()
        self.trading = TradingConfig()
        self.data = DataConfig()
        self.system = SystemConfig()
        
        # Override with provided config
        if config_dict:
            self.update_from_dict(config_dict)
            
        # Create necessary directories
        self._create_directories()
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_dict.items():
            if hasattr(self, section) and isinstance(values, dict):
                config_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(config_obj, key):
                        setattr(config_obj, key, value)
    
    def _create_directories(self):
        """Create necessary directories"""
        dirs = [self.system.model_dir, self.system.log_dir, self.system.results_dir]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'network': self.network.__dict__,
            'ppo': self.ppo.__dict__,
            'trading': self.trading.__dict__,
            'data': self.data.__dict__,
            'system': self.system.__dict__
        }
    
    def save(self, filepath: str):
        """Save configuration to JSON file"""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'Config':
        """Load configuration from JSON file"""
        import json
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls(config_dict)
    
    def validate(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Validate network config
        if self.network.input_features <= 0:
            errors.append("input_features must be positive")
        if self.network.sequence_length <= 0:
            errors.append("sequence_length must be positive")
        if self.network.action_dim <= 0:
            errors.append("action_dim must be positive")
            
        # Validate PPO config
        if not 0 < self.ppo.learning_rate < 1:
            errors.append("learning_rate must be between 0 and 1")
        if not 0 < self.ppo.gamma <= 1:
            errors.append("gamma must be between 0 and 1")
        if not 0 < self.ppo.clip_epsilon < 1:
            errors.append("clip_epsilon must be between 0 and 1")
            
        # Validate trading config
        if self.trading.initial_balance <= 0:
            errors.append("initial_balance must be positive")
        if self.trading.grid_levels <= 0:
            errors.append("grid_levels must be positive")
        if not 0 < self.trading.position_size_pct <= 1:
            errors.append("position_size_pct must be between 0 and 1")
            
        # Validate data config
        splits = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(splits - 1.0) > 1e-6:
            errors.append("Data splits must sum to 1.0")
            
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True

# Default configuration instance
default_config = Config()

# Preset configurations for different scenarios
PRESETS = {
    'development': {
        'ppo': {
            'n_steps': 256,
            'batch_size': 32,
            'max_episode_steps': 100,
            'save_freq': 10,
            'eval_freq': 5
        },
        'trading': {
            'initial_balance': 1000.0,
            'grid_levels': 5
        }
    },
    
    'production': {
        'ppo': {
            'n_steps': 2048,
            'batch_size': 128,
            'max_episode_steps': 1000,
            'save_freq': 50,
            'eval_freq': 25
        },
        'trading': {
            'initial_balance': 10000.0,
            'grid_levels': 20
        },
        'network': {
            'hidden_dim': 512
        }
    },
    
    'conservative': {
        'trading': {
            'position_size_pct': 0.05,
            'max_position_size': 0.3,
            'stop_loss_pct': 0.03,
            'take_profit_pct': 0.015
        },
        'ppo': {
            'entropy_coef': 0.005  # Less exploration
        }
    },
    
    'aggressive': {
        'trading': {
            'position_size_pct': 0.15,
            'max_position_size': 0.7,
            'stop_loss_pct': 0.08,
            'take_profit_pct': 0.03
        },
        'ppo': {
            'entropy_coef': 0.02  # More exploration
        }
    }
}

def get_config(preset: str = 'default') -> Config:
    """Get configuration with optional preset"""
    if preset == 'default':
        return Config()
    elif preset in PRESETS:
        return Config(PRESETS[preset])
    else:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")

if __name__ == "__main__":
    # Test configuration
    config = get_config('development')
    print("Configuration validation:", config.validate())
    
    # Save example configuration
    config.save('example_config.json')
    print("Example configuration saved to example_config.json")