-## 🧪 Pipeline to Deployment

This project supports two standardized pipelines, both optimizing for the highest normalized composite score [0,1]. Model selection is based on out-of-sample (OOS) performance and promotion rules below.

### 1) Short Training and Testing (30 days)
- Train: 15 days
- Out-of-Sample (OOS): 8 days
- Backtest: 7 days (most recent)
- Objective: Train for highest composite score. OOS composite must exceed training composite. For deployment promotion, 7-day backtest composite should exceed both training and OOS composites.

### 2) Full Training and Testing (4 years)
- Train: 2 years
- Out-of-Sample (OOS): 1 year
- Backtest: 1 year (most recent)
- Objective: Train for highest composite score. OOS composite must exceed training composite. For deployment promotion, 1-year backtest composite should exceed both training and OOS composites.

### Model Selection and Promotion
- During training, checkpoint the model periodically.
- Select the best checkpoint by highest OOS composite.
- Promote to deployment only if the 7-day backtest composite > training and OOS composites.

### Deployment Targets
- Binance trading integration (spot/futures as configured) and Telegram notifications.
- Risk/Reward target: example $10 TP vs $4 SL (~2.5:1) aligns with grid rules (TP 0.25%, SL 0.10%).

### Auto Trade Management (planned)
- Manage an isolated margin account and dynamic position sizing.
- Example policy: if cumulative profit reaches $X, increase trade size to $Y.
- Safety constraints: max position size, max daily loss, and drawdown guard.

Note: The composite score is normalized [0,1] and used for OOS selection and deployment decisions; the environment’s per-step rewards remain PnL-based for stable PPO learning.

# PPO Grid-to-Grid Trading System
### Grid Parameters (updated)
- **Grid Spacing**: 0.05% per grid level (0.0005)
- **Take Profit (TP)**: 0.25% from entry (5 grid levels)
- **Stop Loss (SL)**: 0.10% from entry (2 grid levels)
- These rules apply symmetrically to long and short positions and are enforced in `trading_environment.py`.

A sophisticated reinforcement learning trading system that implements Proximal Policy Optimization (PPO) with CNN-TCM (Temporal Convolutional Module) architecture for automated grid trading on cryptocurrency markets.

## 🚀 Features

- **Advanced Neural Architecture**: CNN-TCM hybrid network for temporal pattern recognition
- **PPO Algorithm**: State-of-the-art reinforcement learning with GAE (Generalized Advantage Estimation)
- **Grid Trading Strategy**: Automated buy/sell grid system with dynamic position management
- **Technical Indicators**: VWAP and RSI integration for enhanced market analysis
- **Comprehensive Evaluation**: Multi-metric scoring system including Sharpe ratio, drawdown, and risk-adjusted returns
- **Real-time Data**: Binance API integration with local database caching
- **Backtesting**: Historical performance evaluation with detailed metrics
- **Configurable**: Extensive configuration system with presets for different trading styles

## 📁 Project Structure

```
Trae TCN PPO/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .env                        # Environment variables (API keys)
├── config.py                   # Configuration management
├── data_downloader.py          # Historical data download from Binance
├── data_query.py              # Database query and data processing
├── neural_network.py          # CNN-TCM neural network architecture
├── trading_environment.py     # Grid trading environment
├── ppo_algorithm.py           # PPO implementation
├── train_ppo_grid_trading.py  # Main training script
├── bitcoin_data.db            # SQLite database (created after data download)
├── models/                    # Saved model checkpoints
├── logs/                      # Training logs
└── results/                   # Evaluation results
```

## 🛠️ Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   - Copy `.env` file and add your Binance API credentials
   - Update `API.txt` with your actual API keys if needed

4. **Download historical data**:
   ```bash
   python data_downloader.py
   ```
   This will download 5 years of hourly Bitcoin data from Binance.

## 🎯 Quick Start

### 1. Test System Components
```bash
python train_ppo_grid_trading.py test
```

### 2. Train the Model
```bash
# Basic training
python train_ppo_grid_trading.py train --episodes 1000

# Training with custom parameters
python train_ppo_grid_trading.py train \
    --episodes 2000 \
    --learning-rate 0.0003 \
    --batch-size 128 \
    --hidden-dim 512 \
    --backtest
```

### 3. Evaluate Trained Model
```bash
python train_ppo_grid_trading.py evaluate --model-path models/best_model.pth
```

## 🧠 Neural Network Architecture

The system uses a hybrid CNN-TCM architecture:

### CNN Feature Extractor
- **Input**: 96-hour sequences with 10 features per timestep
- **Layers**: 3 convolutional layers (32→64→128 channels)
- **Purpose**: Extract spatial patterns from market data

### Temporal Convolutional Module (TCM)
- **Architecture**: Dilated convolutions with residual connections
- **Features**: Causal convolutions, dropout regularization
- **Purpose**: Capture long-term temporal dependencies

### Policy and Value Networks
- **Shared Backbone**: CNN-TCM feature extraction
- **Policy Head**: Outputs action probabilities (HOLD, BUY, SELL)
- **Value Head**: Estimates state values for PPO training

## 📊 Trading Environment

### State Space (10 features)
1. **Market Features (7)**:
   - OHLC prices (normalized)
   - Volume
   - VWAP (Volume Weighted Average Price)
   - RSI (Relative Strength Index)

2. **Position Features (3)**:
   - Current position size
   - Unrealized P&L
   - Grid level indicator

### Action Space
- **0**: BUY - Execute buy orders at current grid level
- **1**: SELL - Execute sell orders at current grid level
- **2**: HOLD - Maintain current positions

### Reward Function
 The environment provides per-step rewards primarily from realized PnL on exits, small unrealized PnL shaping while holding, and a minor penalty for invalid actions. Additionally, an episode-level composite score is computed for evaluation and model selection.

### Composite Score (normalized [0,1])
Weighted combination with trade frequency target and activity requirement:

```
base_composite_score = (
    0.28 * sortino_normalized +
    0.22 * calmar_normalized +
    0.20 * profit_factor_normalized +
    0.15 * win_rate_normalized +
    0.10 * max_drawdown_inverse +
    0.05 * trade_freq_normalized
)

# Activity penalty if trades < 4 in the episode
if total_trades < 4:
    base_composite_score *= (total_trades / 4.0)

# Final normalized score in [0,1]
final_composite_score = clamp(base_composite_score, 0.0, 1.0)
```

Normalization notes:
- `sortino_normalized = min(sortino / 2.0, 1.0)`
- `calmar_normalized  = min(calmar / 3.0, 1.0)`
- `profit_factor_normalized = min(profit_factor / 2.0, 1.0)`
- `win_rate_normalized = win_rate` (0–1)
- `max_drawdown_inverse = 1 / (1 + |max_drawdown|)`
- `trade_freq_normalized = min(trades_per_day / 4.0, 1.0)`

Notes:
- The PPO agent trains on per-step rewards; the composite score is used for evaluation/selection.
- Target trade frequency is 4/day or more; fewer than 4 trades reduces the score.
- **Profit**: Raw trading profits
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Risk management
- **Trade Frequency**: Activity level
- **Risk-Adjusted Returns**: Overall performance

## ⚙️ Configuration

The system uses a comprehensive configuration system with presets:

### Available Presets
- **development**: Fast training for testing
- **production**: Full-scale training parameters
- **conservative**: Risk-averse trading settings
- **aggressive**: High-risk, high-reward settings

### Custom Configuration
```python
from config import Config, get_config

# Use preset
config = get_config('production')

# Or create custom
config = Config({
    'ppo': {
        'learning_rate': 0.0001,
        'batch_size': 256
    },
    'trading': {
        'initial_balance': 50000,
        'grid_levels': 15
    }
})
```

## 📈 Training Process

### Data Splits
- **Training**: 60% of historical data
- **Validation**: 20% for hyperparameter tuning
- **Testing**: 20% for final evaluation

### Training Loop
1. **Experience Collection**: Agent interacts with environment
2. **Advantage Estimation**: GAE computation
3. **Policy Update**: PPO clipped objective
4. **Value Function Update**: MSE loss
5. **Evaluation**: Periodic validation on held-out data

### Early Stopping
- Monitors composite score on validation set
- Stops training if no improvement for 200 episodes
- Saves best model based on validation performance

## 📊 Evaluation Metrics

### Trading Performance
- **Total Return**: Cumulative profit/loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade**: Mean profit per trade

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **VaR (Value at Risk)**: Potential losses at confidence level
- **Calmar Ratio**: Return/Maximum Drawdown
- **Sortino Ratio**: Downside risk-adjusted returns

### Operational Metrics
- **Trade Frequency**: Number of trades per period
- **Position Utilization**: Average capital deployment
- **Grid Efficiency**: Grid level utilization

## 🔧 Advanced Usage

### Custom Neural Network
```python
from neural_network import PPONetwork
from config import NetworkConfig

# Custom architecture
net_config = NetworkConfig(
    hidden_dim=1024,
    cnn_channels=[64, 128, 256],
    tcm_dropout=0.2
)

network = PPONetwork(
    input_features=net_config.input_features,
    sequence_length=net_config.sequence_length,
    config=net_config
)
```

### Custom Trading Environment
```python
from trading_environment import GridTradingEnvironment
from config import TradingConfig

# Custom trading parameters
trading_config = TradingConfig(
    initial_balance=100000,
    grid_levels=25,
    position_size_pct=0.08,
    stop_loss_pct=0.04
)

env = GridTradingEnvironment(data_query, config=trading_config)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size: `--batch-size 32`
   - Reduce hidden dimension: `--hidden-dim 128`
   - Use CPU: Set device to 'cpu' in config

2. **Training Instability**:
   - Lower learning rate: `--learning-rate 0.0001`
   - Increase gradient clipping: `--max-grad-norm 1.0`
   - Reduce entropy coefficient

3. **Poor Performance**:
   - Increase training episodes
   - Adjust reward function weights
   - Try different presets (conservative/aggressive)

4. **Data Issues**:
   - Re-run data downloader
   - Check database integrity
   - Verify date ranges

### Logging
Training logs are saved to `logs/` directory with timestamps.
Use `--log-level DEBUG` for detailed information.

## 📚 References

- **PPO Paper**: [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- **GAE Paper**: [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)
- **TCN Paper**: [An Empirical Evaluation of Generic Convolutional and Recurrent Networks](https://arxiv.org/abs/1803.01271)
- **Grid Trading**: [Algorithmic Trading Strategies](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## ⚠️ Disclaimer

**This software is for educational and research purposes only. Trading cryptocurrencies involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Use at your own risk.**

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Happy Trading! 🚀📈**