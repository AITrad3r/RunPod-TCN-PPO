import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import sqlite3
from data_query import DataQuery

class ActionType(Enum):
    BUY = 0
    SELL = 1
    HOLD = 2

@dataclass
class Position:
    """Represents a trading position"""
    entry_price: float
    entry_time: int
    position_type: str  # 'long' or 'short'
    grid_level: float
    stop_loss: float
    take_profit: float
    quantity: float = 1.0
    entry_index: int = 0
    
class TradingMetrics:
    """Calculate trading performance metrics"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.trades = []
        self.equity_curve = []
        self.drawdowns = []
        self.peak_equity = 0
        
    def add_trade(self, entry_price: float, exit_price: float, 
                  position_type: str, quantity: float = 1.0,
                  entry_time: str = None, exit_time: str = None):
        """Add a completed trade"""
        if position_type == 'long':
            pnl = (exit_price - entry_price) * quantity
            return_pct = (exit_price - entry_price) / entry_price
        else:  # short
            pnl = (entry_price - exit_price) * quantity
            return_pct = (entry_price - exit_price) / entry_price
            
        trade = {
            'entry_price': entry_price,
            'exit_price': exit_price,
            'position_type': position_type,
            'pnl': pnl,
            'return_pct': return_pct,
            'quantity': quantity
        }
        if entry_time is not None:
            trade['entry_time'] = entry_time
        if exit_time is not None:
            trade['exit_time'] = exit_time
        
        self.trades.append(trade)
        return pnl
        
    def update_equity(self, current_equity: float):
        """Update equity curve and drawdown"""
        self.equity_curve.append(current_equity)
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
            
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.drawdowns.append(drawdown)
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive trading metrics"""
        if not self.trades:
            return self._empty_metrics()
            
        trades_df = pd.DataFrame(self.trades)
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
        
        # Risk metrics
        returns = trades_df['return_pct'].values
        
        if len(returns) > 1:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            
            # Sortino ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            sortino_ratio = np.mean(returns) / downside_std if downside_std > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # System Quality Number (SQN): sqrt(n) * mean(R) / std(R)
        # Use trade return_pct as R multiple proxy. Defined only when >=2 trades and std>0
        if len(returns) >= 2 and np.std(returns) > 0:
            sqn = (np.mean(returns) / np.std(returns)) * np.sqrt(len(returns))
        else:
            sqn = 0.0
            
        # Drawdown metrics
        max_drawdown = max(self.drawdowns) if self.drawdowns else 0
        
        # Calmar ratio
        annual_return = np.mean(returns) * 24 * 365 if returns.size > 0 else 0  # Hourly to annual
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0
        
        # Trade frequency (trades per day)
        trade_frequency = total_trades / (len(self.equity_curve) / 24) if len(self.equity_curve) > 24 else 0
        
        return {
            'net_profit': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'trade_frequency': trade_frequency,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sqn': sqn,
        }
        
    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics when no trades"""
        return {
            'net_profit': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'trade_frequency': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
        
    def calculate_composite_score(self, target_profit: float = 1000.0) -> float:
        """Calculate the composite score as specified in the PPO spec"""
        metrics = self.calculate_metrics()
        total_trades = metrics.get('total_trades', 0)
        min_trades_required = 4
        
        # Normalize metrics
        sortino_normalized = min(max(metrics['sortino_ratio'], 0.0) / 2.0, 1.0)  # Cap at 2.0
        calmar_normalized = min(max(metrics['calmar_ratio'], 0.0) / 3.0, 1.0)    # Cap at 3.0
        profit_factor_normalized = min(max(metrics['profit_factor'], 0.0) / 2.0, 1.0)  # Cap at 2.0
        win_rate_normalized = max(min(metrics['win_rate'], 1.0), 0.0)
        max_drawdown_inverse = 1 / (1 + abs(metrics['max_drawdown']))
        
        # Trade frequency normalization (target: 4 trades per day or more)
        expected_max_trades_per_day = 4.0
        trade_freq_normalized = min(max(metrics['trade_frequency'], 0.0) / expected_max_trades_per_day, 1.0)
        
        # SQN normalization (cap at 3.0)
        sqn_normalized = min(max(metrics.get('sqn', 0.0), 0.0) / 3.0, 1.0)
        
        # Base composite score (Option B weights)
        base_composite_score = (
            0.24 * sortino_normalized +
            0.20 * calmar_normalized +
            0.18 * profit_factor_normalized +
            0.16 * win_rate_normalized +
            0.07 * max_drawdown_inverse +
            0.10 * trade_freq_normalized +
            0.05 * sqn_normalized
        )
        
        # Episode activity penalty (very low activity episodes)
        if total_trades < min_trades_required:
            activity_penalty = max(0.0, total_trades / float(min_trades_required))
            base_composite_score *= activity_penalty
        
        # User constraints penalties (hinge-style)
        # 1) Minimum trades/day >= 3
        min_trades_per_day = 3.0
        tf = float(metrics.get('trade_frequency', 0.0))
        penalty_trades = 1.0 if tf >= min_trades_per_day else max(0.0, tf / min_trades_per_day)
        
        # 2) Win rate >= 50%
        wr = float(metrics.get('win_rate', 0.0))
        if wr >= 0.5:
            penalty_wr = 1.0
        else:
            # Quadratic penalty to discourage very low win rates
            penalty_wr = max(0.0, (wr / 0.5) ** 2)
        
        penalized_score = base_composite_score * penalty_trades * penalty_wr
        
        # Final normalized composite score in [0, 1]
        final_composite_score = max(0.0, min(1.0, penalized_score))
        return final_composite_score

    def calculate_composite_score_pct(self) -> float:
        """Return composite score scaled to percentage [0, 100].

        Keeps the existing normalized score (0-1) intact for internal logic and
        reward shaping stability, but provides a human-friendly percentage for
        reporting.
        """
        base = self.calculate_composite_score()
        return float(round(base * 100.0, 4))

class GridTradingEnvironment:
    """Grid Trading Environment for PPO training"""
    
    def __init__(self, data_query: DataQuery, window_size: int = 96, 
                 grid_spacing: float = 0.0003, risk_reward_ratio: float = 2.5,
                 initial_balance: float = 10000.0,
                 hold_time_penalty_hours: int = 2,
                 hold_time_penalty: float = -0.005,
                 optimize_for_composite: bool = False,
                 per_trade_risk_dollars: float = 4.0,
                 fixed_dollar_rr: bool = True,
                 flat_hold_penalty: float = -0.05,
                 open_hold_penalty: float = -0.05,
                 min_bb_width: float = 0.01,
                 low_vol_penalty: float = -0.02,
                 use_vwap_filter: bool = False,
                 vwap_longs_only: bool = False):
        
        self.data_query = data_query
        self.window_size = window_size
        self.grid_spacing = grid_spacing
        self.rr = risk_reward_ratio
        self.initial_balance = initial_balance
        self.hold_time_penalty_hours = hold_time_penalty_hours
        self.hold_time_penalty = hold_time_penalty
        self.optimize_for_composite = optimize_for_composite
        # Trend filter config
        self.use_vwap_filter = bool(use_vwap_filter)
        self.vwap_longs_only = bool(vwap_longs_only)
        # Fixed-dollar RR config (risk per trade in $; reward implied by rr)
        self.per_trade_risk_dollars = float(per_trade_risk_dollars)
        self.fixed_dollar_rr = bool(fixed_dollar_rr)
        # Strong holding penalties
        self.flat_hold_penalty = float(flat_hold_penalty)
        self.open_hold_penalty = float(open_hold_penalty)
        # Volatility floor (Bollinger band width normalized by SMA). Set <=0 to disable.
        self.min_bb_width = float(min_bb_width)
        self.low_vol_penalty = float(low_vol_penalty)
        # Enforce exits strictly at SL/TP only
        self.require_sl_tp_close: bool = True
        
        # Load and prepare data
        self.data = None
        self.current_step = 0
        self.max_steps = 0
        # Anchor price for multiplicative grid levels (first close in loaded data)
        self.anchor_price: Optional[float] = None
        
        # Trading state
        self.balance = initial_balance
        self.current_position: Optional[Position] = None
        self.metrics = TradingMetrics()
        
        # Episode tracking
        self.episode_start_balance = initial_balance
        
    def load_data(self, start_date: str = None, end_date: str = None):
        """Load trading data"""
        df = self.data_query.get_data(start_date=start_date, end_date=end_date)
        df = self.data_query.calculate_indicators(df)
        
        # Select features for the model
        feature_columns = [
            'close_price', 'volume', 'vwap', 'price_minus_vwap',
            'above_vwap', 'rsi', 'rsi_50_cross'
        ]
        
        extra_cols = []
        if 'bb_pctb' in df.columns:
            extra_cols.append('bb_pctb')
        if 'bb_width' in df.columns:
            extra_cols.append('bb_width')
        
        # Remove NaN values ONLY from the required columns used by the model/timekeeping
        rows_before = len(df)
        required_cols = feature_columns + ['datetime', 'timestamp']
        required_cols = [c for c in required_cols if c in df.columns]
        df = df.dropna(subset=required_cols)
        rows_after = len(df)
        dropped = rows_before - rows_after
        if dropped > 0:
            print(f"Dropped {dropped} rows due to NaNs in required columns: {required_cols}")
        # If everything dropped (e.g., indicator windows too long), set empty data
        if len(df) == 0:
            cols = ['close_price', 'volume', 'vwap', 'price_minus_vwap',
                    'above_vwap', 'rsi', 'rsi_50_cross', 'datetime', 'timestamp']
            self.data = pd.DataFrame(columns=cols)
            self.anchor_price = None
            self.max_steps = 0
            print("Loaded 0 data points, 0 training steps available")
            return
        self.data = df[feature_columns + ['datetime', 'timestamp'] + extra_cols].copy()
        # Define anchor price for grid ladder as the first available close
        if len(self.data) > 0:
            self.anchor_price = float(self.data.iloc[0]['close_price'])
        else:
            self.anchor_price = None
        # Ensure non-negative max_steps for short slices
        self.max_steps = max(0, len(self.data) - 1)
        
        print(f"Loaded {len(self.data)} data points, {self.max_steps} training steps available")
        
    def reset(self, random_start: bool = True) -> np.ndarray:
        """Reset environment for new episode"""
        if random_start and self.max_steps > 1000:
            # Start from random position (but leave enough data for full episode)
            max_start = self.max_steps - 1000
            self.current_step = np.random.randint(0, max_start)
        else:
            self.current_step = 0
            
        # Reset trading state
        self.balance = self.initial_balance
        self.episode_start_balance = self.initial_balance
        self.current_position = None
        self.metrics.reset()
        
        return self._get_observation()
        
    def _get_observation(self) -> np.ndarray:
        """Get current observation window"""
        # Determine slice bounds
        if len(self.data) == 0:
            return np.zeros((self.window_size, 10), dtype=np.float32)
        start_idx = max(0, min(self.current_step, len(self.data) - 1))
        end_idx = min(len(self.data), start_idx + self.window_size)
        
        # Get feature data (prefer Bollinger %B when present)
        use_bb = ('bb_pctb' in self.data.columns) and (not self.data['bb_pctb'].isna().all())
        feature_columns = [
            'close_price', 'volume', 'vwap', 'price_minus_vwap',
            'above_vwap', 'rsi', 'bb_pctb' if use_bb else 'rsi_50_cross'
        ]
        
        obs_data = self.data.iloc[start_idx:end_idx][feature_columns].values
        obs_len = obs_data.shape[0]
        
        # Add position information as additional features
        position_features = self._get_position_features()
        
        # Expand position features to match sequence length
        position_features_expanded = np.tile(position_features, (obs_len, 1))

        # If slice shorter than window_size, pad at the beginning to match window_size
        if obs_len < self.window_size:
            pad_rows = self.window_size - obs_len
            # Pad using the first available row (or zeros if none)
            if obs_len > 0:
                pad_block = np.repeat(obs_data[:1, :], pad_rows, axis=0)
                pad_pos = np.repeat(position_features_expanded[:1, :], pad_rows, axis=0)
            else:
                pad_block = np.zeros((pad_rows, len(feature_columns)), dtype=obs_data.dtype)
                pad_pos = np.zeros((pad_rows, 3), dtype=np.float32)
            obs_data = np.concatenate([pad_block, obs_data], axis=0)
            position_features_expanded = np.concatenate([pad_pos, position_features_expanded], axis=0)
        
        # Concatenate observations with position features
        observation = np.concatenate([obs_data, position_features_expanded], axis=1)

        return observation.astype(np.float32)
        
    def _get_position_features(self) -> np.ndarray:
        """Get position-related features"""
        if self.current_position is None:
            return np.array([0.0, 0.0, 0.0])  # No position
            
        idx = min(self.current_step, len(self.data) - 1)
        current_price = self.data.iloc[idx]['close_price']
        
        # Calculate unrealized PnL
        if self.current_position.position_type == 'long':
            unrealized_pnl = (current_price - self.current_position.entry_price) / self.current_position.entry_price
        else:  # short
            unrealized_pnl = (self.current_position.entry_price - current_price) / self.current_position.entry_price
            
        # Position type (1 for long, -1 for short, 0 for none)
        position_type = 1.0 if self.current_position.position_type == 'long' else -1.0
        
        # Time in position (normalized)
        time_in_position = min((self.current_step - self.current_position.entry_time) / 24.0, 1.0)
        
        return np.array([position_type, unrealized_pnl, time_in_position])
        
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute one step in the environment"""
        # Guard: if no data available, terminate immediately to avoid iloc errors
        if self.data is None or len(self.data) == 0:
            zero_obs = np.zeros((self.window_size, 10), dtype=np.float32)
            info = {'balance': self.balance, 'current_price': 0.0, 'position': False, 'step': self.current_step}
            return zero_obs, 0.0, True, info
        idx = min(self.current_step + self.window_size, len(self.data) - 1)
        current_price = self.data.iloc[idx]['close_price']
        
        # Execute action
        reward = self._execute_action(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done (one step before exceeding max window start)
        done = self.current_step >= self.max_steps
        
        # Get next observation
        next_obs = self._get_observation() if not done else np.zeros_like(self._get_observation())
        
        # Update metrics
        self.metrics.update_equity(self.balance)
        
        # Composite optimization: preserve shaping rewards; at terminal step use composite as reward
        if self.optimize_for_composite and done:
            metrics = self.get_episode_metrics()
            reward = float(metrics.get('composite_score', 0.0))
        
        # Prepare info dict
        info = {
            'balance': self.balance,
            'current_price': current_price,
            'position': self.current_position is not None,
            'step': self.current_step
        }
        
        return next_obs, reward, done, info
        
    def _execute_action(self, action: int, current_price: float) -> float:
        """Execute trading action and return reward"""
        # If position is open, ignore the action and only check SL/TP
        if self.current_position is not None:
            return self._execute_hold(current_price)

        # Flat: only BUY/SELL allowed; HOLD discouraged strongly
        if action == ActionType.BUY.value:
            # Volatility floor gate (if enabled and bb_width present)
            idx = min(self.current_step + self.window_size, len(self.data) - 1)
            if self.min_bb_width > 0 and 'bb_width' in self.data.columns:
                bw = self.data.iloc[idx].get('bb_width', np.nan)
                if not np.isnan(bw) and bw < self.min_bb_width:
                    return self.low_vol_penalty

            # VWAP trend filter: allow BUY only if price > VWAP (when enabled)
            if self.use_vwap_filter and 'vwap' in self.data.columns:
                vwap_val = self.data.iloc[idx].get('vwap', np.nan)
                if not np.isnan(vwap_val):
                    if not (current_price > float(vwap_val)):
                        # Reject BUY when below/at VWAP
                        return self.flat_hold_penalty
            return self._execute_buy(current_price)
        elif action == ActionType.SELL.value:
            # Volatility floor gate (if enabled and bb_width present)
            idx = min(self.current_step + self.window_size, len(self.data) - 1)
            if self.min_bb_width > 0 and 'bb_width' in self.data.columns:
                bw = self.data.iloc[idx].get('bb_width', np.nan)
                if not np.isnan(bw) and bw < self.min_bb_width:
                    return self.low_vol_penalty

            # VWAP trend filter: allow SELL only if price < VWAP (unless longs-only)
            if self.use_vwap_filter and 'vwap' in self.data.columns:
                if self.vwap_longs_only:
                    # Block all SELLs in longs-only mode
                    return self.flat_hold_penalty
                vwap_val = self.data.iloc[idx].get('vwap', np.nan)
                if not np.isnan(vwap_val):
                    if not (current_price < float(vwap_val)):
                        # Reject SELL when above/at VWAP
                        return self.flat_hold_penalty
            return self._execute_sell(current_price)
        else:
            # Discourage HOLD when flat
            return self.flat_hold_penalty
        
    # -------- Grid helper utilities (strict grid ladder) --------
    def _grid_factor(self) -> float:
        """Multiplicative factor between adjacent grid levels."""
        return 1.0 + float(self.grid_spacing)

    def _grid_index_for_price(self, price: float) -> int:
        """Get nearest integer grid index for a given price using the anchor."""
        if not self.anchor_price or self.anchor_price <= 0:
            # Fallback: treat current price as anchor index 0
            return 0
        import math
        k = math.log(max(price, 1e-12) / self.anchor_price) / math.log(self._grid_factor())
        return int(round(k))

    def _price_from_grid_index(self, k: int) -> float:
        """Compute price at grid index k from the anchor price."""
        if not self.anchor_price or self.anchor_price <= 0:
            return 0.0
        return float(self.anchor_price) * (self._grid_factor() ** int(k))

    def _round_to_grid(self, price: float) -> Tuple[float, int]:
        """Round a raw price to the nearest grid level and return (grid_price, grid_index)."""
        k = self._grid_index_for_price(price)
        p = self._price_from_grid_index(k)
        return p, k

    def _execute_buy(self, current_price: float) -> float:
        """Execute buy action (long position). Entry and exits at grid levels only."""
        if self.current_position is not None:
            return -0.01  # invalid action penalty
        entry_price, entry_grid_idx = self._round_to_grid(current_price)
        # Use fixed risk grids for SL
        risk_grids = 2
        sl_grid_idx = entry_grid_idx - risk_grids
        stop_loss = self._price_from_grid_index(sl_grid_idx)
        # Compute TP from exact dollar RR target and snap to nearest grid
        tp_target = entry_price + self.rr * (entry_price - stop_loss)
        tp_grid_idx = self._grid_index_for_price(tp_target)
        take_profit = self._price_from_grid_index(tp_grid_idx)
        qty = 1.0
        if self.fixed_dollar_rr:
            risk_diff = max(1e-12, entry_price - stop_loss)
            qty = self.per_trade_risk_dollars / risk_diff
        self.current_position = Position(
            entry_price=entry_price,
            entry_time=self.current_step,
            position_type='long',
            grid_level=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=qty,
            entry_index=min(self.current_step + self.window_size, len(self.data) - 1)
        )
        return 0.0

    def _execute_sell(self, current_price: float) -> float:
        """Execute sell action (short position). Entry and exits at grid levels only."""
        if self.current_position is not None:
            return -0.01
        entry_price, entry_grid_idx = self._round_to_grid(current_price)
        # Use fixed risk grids for SL
        risk_grids = 2
        sl_grid_idx = entry_grid_idx + risk_grids
        stop_loss = self._price_from_grid_index(sl_grid_idx)
        # Compute TP from exact dollar RR target and snap to nearest grid
        tp_target = entry_price - self.rr * (stop_loss - entry_price)
        tp_grid_idx = self._grid_index_for_price(tp_target)
        take_profit = self._price_from_grid_index(tp_grid_idx)
        qty = 1.0
        if self.fixed_dollar_rr:
            risk_diff = max(1e-12, stop_loss - entry_price)
            qty = self.per_trade_risk_dollars / risk_diff
        self.current_position = Position(
            entry_price=entry_price,
            entry_time=self.current_step,
            position_type='short',
            grid_level=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=qty,
            entry_index=min(self.current_step + self.window_size, len(self.data) - 1)
        )
        return 0.0

    def _execute_hold(self, current_price: float) -> float:
        """When a position is open, only close at SL/TP. No shaping rewards/penalties."""
        if self.current_position is None:
            return 0.0
        position = self.current_position
        if position.position_type == 'long':
            if current_price <= position.stop_loss:
                return self._close_position(position.stop_loss)
            if current_price >= position.take_profit:
                return self._close_position(position.take_profit)
        else:
            if current_price >= position.stop_loss:
                return self._close_position(position.stop_loss)
            if current_price <= position.take_profit:
                return self._close_position(position.take_profit)
        # Apply strong penalty while holding to encourage faster resolution via SL/TP
        return self.open_hold_penalty
        
    def _close_position(self, exit_price: float) -> float:
        """Close current position and calculate reward"""
        if self.current_position is None:
            return 0.0
            
        position = self.current_position
        
        # Calculate PnL
        # Map indices to datetimes for entry and exit
        entry_idx = min(position.entry_index, len(self.data) - 1)
        exit_idx = min(self.current_step + self.window_size, len(self.data) - 1)
        entry_dt = str(self.data.iloc[entry_idx]['datetime']) if 'datetime' in self.data.columns else None
        exit_dt = str(self.data.iloc[exit_idx]['datetime']) if 'datetime' in self.data.columns else None

        pnl = self.metrics.add_trade(
            entry_price=position.entry_price,
            exit_price=exit_price,
            position_type=position.position_type,
            quantity=position.quantity,
            entry_time=entry_dt,
            exit_time=exit_dt
        )
        
        # Update balance
        self.balance += pnl
        
        # Calculate reward based on PnL percentage
        pnl_pct = pnl / position.entry_price
        reward = pnl_pct * 10  # Scale reward
        
        # Clear position
        self.current_position = None
        
        return reward
        
    def get_episode_metrics(self) -> Dict[str, float]:
        """Get metrics for the completed episode"""
        # If strict SL/TP-only exits, do not force close open positions at episode end
        if self.current_position is not None and not getattr(self, 'require_sl_tp_close', True):
            idx = min(self.current_step, len(self.data) - 1)
            current_price = self.data.iloc[idx]['close_price']
            self._close_position(current_price)
            
        metrics = self.metrics.calculate_metrics()
        metrics['episode_return'] = (self.balance - self.episode_start_balance) / self.episode_start_balance
        metrics['final_balance'] = self.balance
        metrics['composite_score'] = self.metrics.calculate_composite_score()
        metrics['composite_score_pct'] = self.metrics.calculate_composite_score_pct()
        
        return metrics

def test_environment():
    """Test the trading environment"""
    print("Testing Grid Trading Environment...")
    
    # Initialize data query and environment
    data_query = DataQuery()
    env = GridTradingEnvironment(data_query)
    
    # Load recent data for testing
    env.load_data()
    
    # Test reset
    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    # Test a few steps
    total_reward = 0
    for i in range(10):
        action = np.random.randint(0, 3)  # Random action
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Balance=${info['balance']:.2f}")
        
        if done:
            break
            
    # Get episode metrics
    metrics = env.get_episode_metrics()
    print(f"\nEpisode completed:")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Final balance: ${metrics['final_balance']:.2f}")
    print(f"Episode return: {metrics['episode_return']:.2%}")
    print(f"Composite score: {metrics['composite_score']:.4f}")
    
    print("\nEnvironment test completed successfully!")

if __name__ == "__main__":
    test_environment()