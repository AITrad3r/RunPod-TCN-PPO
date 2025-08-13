## 🧪 Pipeline to Deployment

Two standardized pipelines optimize for the highest normalized composite score [0,1]. Selection and promotion are based on out-of-sample (OOS) performance.

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
- Checkpoint periodically during training.
- Select best checkpoint by highest OOS composite score.
- Promote to deployment only if backtest composite > training and OOS composites.

# PPO Grid-to-Grid Trading System with CNN-TCM and VWAP/RSI

A reinforcement learning-based grid trading system using **Proximal Policy Optimization (PPO)**. The agent learns to open and exit buy/sell positions at predefined grid levels using CNN and TCM for spatiotemporal pattern recognition. Training is guided by a composite performance score that optimizes both profitability and trading quality.

---

## 📊 System Overview

- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **Neural Network**: CNN + Temporal Convolutional Module (TCM)
- **Indicators Used**: VWAP, RSI(14)
- **Timeframe**: 60-minute candles
- **Trading Style**: Grid-to-Grid with predefined spacing and R\:R logic
- **Reward Function**: Per-step rewards (realized PnL on exits, small unrealized PnL shaping, invalid action penalty). Episode-level composite score (normalized [0,1]) is used for evaluation and model selection.

---

## ⚙️ Grid System Logic

- **Grid Spacing**: 0.05% of price (0.0005)
- **Risk**/**Reward** Ratio: 2.5:1

### Buy Logic:

- Entry: At grid level
- Exit: 5 grids above entry (0.25%)
- Stop Loss: 2 grids below entry (0.10%)

### Sell Logic:

- Entry: At grid level
- Exit: 5 grids below entry (0.25%)
- Stop Loss: 2 grids above entry (0.10%)

---

## 📊 State Space (Inputs)

### Observation Window:

- Last 96 hourly candles

### Features Per Timestep:

- Close price
- Volume
- VWAP
- Price - VWAP
- Above VWAP (binary)
- RSI(14)
- RSI 50 crossover (binary)

### Input Shape:

```
[96 x 7] tensor
```

---

## 🤖 Action Space (Discrete)

| ID | Action                              |
| -- | ----------------------------------- |
| 0  | Buy at Grid Level, Exit Grid Above  |
| 1  | Sell at Grid Level, Exit Grid Below |
| 2  | Hold                                |

Constraints:

- Only one trade open at a time
- Entry and exit are treated as a single atomic action
- Exit actions valid only if position is open

Constraints:

- Only one trade open at a time
- Entry and exit are treated as a single atomic action
- Exit actions valid only if position is open

---

## 🧠 Neural Network Architecture

```plaintext
Input: [96 x 7]
↓
CNN Layers (feature maps)
↓
TCM Layers (temporal dependencies)
↓
Fully Connected Layers
↓            ↓
Actor Head     Critic Head
(Logits)       (State Value)
```

---

## 🧰 Reward & Training Objective

### Final Composite Score (normalized [0,1], used for evaluation/model selection)

```python
base_composite_score = (
    0.28 * sortino_normalized +     # Downside risk-adjusted returns
    0.22 * calmar_normalized +      # Return per unit of max drawdown
    0.20 * profit_factor_normalized +  # Profitability efficiency
    0.15 * win_rate_normalized +    # Trading accuracy
    0.10 * max_drawdown_inverse +   # Risk management
    0.05 * trade_freq_normalized    # Target 4 trades/day
)

# Activity penalty if episode has fewer than 4 trades
if total_trades < 4:
    base_composite_score *= (total_trades / 4.0)

# Final normalized score in [0, 1]
final_composite_score = clamp(base_composite_score, 0.0, 1.0)
```

### Metric Normalization Examples:

- `sortino_normalized = min(sortino / 2.0, 1.0)`
- `calmar_normalized  = min(calmar / 3.0, 1.0)`
- `profit_factor_normalized = min(profit_factor / 2.0, 1.0)`
- `win_rate_normalized = win_rate`  # 0..1
- `max_drawdown_inverse = 1 / (1 + abs(drawdown_pct))`
- `trade_freq_normalized = min(trades_per_day / 4.0, 1.0)`

### Reward Shaping Options:

- Add small reward per timestep for unrealized PnL gain
- Penalize large drawdown or invalid trades

---

## 📊 PPO Training Loop Summary

1. Run environment for each episode
2. Collect metrics: profit, drawdown, win rate, trades/day
3. Compute `final_composite_score`
4. Use score to adjust PPO reward or model selection
5. Update PPO agent using clipped loss and GAE

---

## 🧪 Training, Validation & Backtesting

### Data Split Strategy

- **Training Period**: 2 years of historical 60-minute data
- **Out-of-Sample Validation**: 1 year
- **Backtest Period**: 1 year (most recent)

### Validation Criteria

- Backtest performance **must exceed** training and out-of-sample in:
  - Final Composite Score
  - Net Profit
  - Sharpe/Sortino
  - Max Drawdown (lower is better)

### Model Selection

- Checkpoints saved every N episodes
- Best model selected based on out-of-sample composite score
- Final model re-evaluated on 1-year backtest

---

## ✅ Evaluation Metrics

- Net Profit
- Sortino Ratio
- Calmar Ratio
- Profit Factor
- Win Rate
- Max Drawdown
- Trade Frequency
- Final Composite Score

---

## 📆 Resulting Agent Behavior

- Takes high-probability trades at grid edges
- Exits with strong reward/risk ratio
- Times entries using VWAP and RSI signals
- Maintains high trade frequency with risk control

---

## 🚀 Optional Enhancements

- Use multi-timeframe VWAP (e.g. daily + hourly)
- Mask invalid actions dynamically
- Live backtesting and deployment via Binance API
- Save checkpoints based on highest `final_composite_score`

---

## 🔹 Conclusion

This PPO-driven grid-to-grid trading system integrates technical indicators (VWAP, RSI) with temporal modeling via CNN-TCM. It learns to make efficient trading decisions by optimizing for a robust composite score that balances profitability, drawdown, frequency, and accuracy. Ideal for algorithmic intraday strategies focused on high-quality, high-frequency decision making.

