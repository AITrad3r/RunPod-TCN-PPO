import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import random
from dataclasses import dataclass
from neural_network import PPONetwork, GridTradingFeatureProcessor
from trading_environment import GridTradingEnvironment
from data_query import DataQuery
import os
from datetime import datetime
import json

@dataclass
class PPOConfig:
    """PPO hyperparameters configuration"""
    # Network architecture
    input_features: int = 10  # 7 market features + 3 position features
    sequence_length: int = 96
    hidden_dim: int = 256
    action_dim: int = 3
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda
    clip_epsilon: float = 0.2  # PPO clip parameter
    entropy_coef: float = 0.01  # Entropy coefficient
    value_coef: float = 0.5  # Value loss coefficient
    max_grad_norm: float = 0.5  # Gradient clipping
    
    # Training parameters
    n_steps: int = 2048  # Steps per update
    batch_size: int = 64
    n_epochs: int = 10  # PPO epochs per update
    n_envs: int = 1  # Number of parallel environments
    
    # Episode parameters
    max_episode_steps: int = 1000
    target_composite_score: float = 0.8
    optimize_for_composite: bool = False
    
    # Checkpointing
    save_freq: int = 100  # Save every N episodes
    eval_freq: int = 50   # Evaluate every N episodes

class PPOBuffer:
    """Experience buffer for PPO"""
    
    def __init__(self, size: int, obs_shape: Tuple, device: torch.device):
        self.size = size
        self.device = device
        
        # Buffers
        self.observations = torch.zeros((size,) + obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros(size, dtype=torch.long, device=device)
        self.rewards = torch.zeros(size, dtype=torch.float32, device=device)
        self.values = torch.zeros(size, dtype=torch.float32, device=device)
        self.log_probs = torch.zeros(size, dtype=torch.float32, device=device)
        self.dones = torch.zeros(size, dtype=torch.bool, device=device)
        
        # GAE buffers
        self.advantages = torch.zeros(size, dtype=torch.float32, device=device)
        self.returns = torch.zeros(size, dtype=torch.float32, device=device)
        
        self.ptr = 0
        self.full = False
        
    def store(self, obs: np.ndarray, action: int, reward: float, 
              value: float, log_prob: float, done: bool):
        """Store experience"""
        self.observations[self.ptr] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.actions[self.ptr] = torch.as_tensor(action, dtype=torch.long, device=self.device)
        self.rewards[self.ptr] = torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self.values[self.ptr] = torch.as_tensor(value, dtype=torch.float32, device=self.device).squeeze()
        self.log_probs[self.ptr] = torch.as_tensor(log_prob, dtype=torch.float32, device=self.device)
        self.dones[self.ptr] = torch.as_tensor(done, dtype=torch.bool, device=self.device)
        
        self.ptr = (self.ptr + 1) % self.size
        if self.ptr == 0:
            self.full = True
            
    def compute_gae(self, next_value: float, gamma: float, gae_lambda: float):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0
        
        # Work backwards through the buffer
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - float(self.dones[t])
                next_value_t = self.values[t + 1]
                
            delta = self.rewards[t] + gamma * next_value_t * next_non_terminal - self.values[t]
            advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            
        self.advantages = advantages
        self.returns = advantages + self.values
        
    def get_batch(self, batch_size: int):
        """Get random batch for training"""
        indices = torch.randperm(self.size, device=self.device)[:batch_size]
        
        return {
            'observations': self.observations[indices],
            'actions': self.actions[indices],
            'old_log_probs': self.log_probs[indices],
            'advantages': self.advantages[indices],
            'returns': self.returns[indices],
            'values': self.values[indices]
        }
        
    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.full = False

class PPOAgent:
    """PPO Agent for Grid Trading"""
    
    def __init__(self, config: PPOConfig, device: torch.device = None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize network
        self.network = PPONetwork(
            input_features=config.input_features,
            sequence_length=config.sequence_length,
            hidden_dim=config.hidden_dim,
            action_dim=config.action_dim
        ).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=config.learning_rate)
        
        # Experience buffer
        obs_shape = (config.sequence_length, config.input_features)
        self.buffer = PPOBuffer(config.n_steps, obs_shape, self.device)
        
        # Feature processor
        self.feature_processor = GridTradingFeatureProcessor()
        
        # Training metrics
        self.episode_rewards = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.episode_composite_scores = deque(maxlen=100)
        
        # Best model tracking
        self.best_composite_score = -float('inf')
        self.best_model_path = None
        
    def select_action(self, observation: np.ndarray, deterministic: bool = False):
        """Select action using the policy network"""
        with torch.no_grad():
            obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
            
            if deterministic:
                action_logits, value = self.network(obs_tensor)
                action = torch.argmax(action_logits, dim=-1)
                log_prob = torch.zeros(1, device=self.device)
            else:
                action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
                
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0]
        
    def update(self):
        """Update the policy using PPO"""
        # Compute advantages using GAE
        with torch.no_grad():
            # Get value of the last observation
            last_obs = self.buffer.observations[self.buffer.ptr - 1].unsqueeze(0)
            next_value = self.network.get_value(last_obs).cpu().numpy()[0]
            
        self.buffer.compute_gae(next_value, self.config.gamma, self.config.gae_lambda)
        
        # Normalize advantages
        advantages = self.buffer.advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        self.buffer.advantages = advantages
        
        # Training loop
        total_loss = 0
        policy_losses = []
        value_losses = []
        entropy_losses = []
        
        for epoch in range(self.config.n_epochs):
            batch = self.buffer.get_batch(self.config.batch_size)
            
            # Forward pass
            action, log_prob, entropy, value = self.network.get_action_and_value(
                batch['observations'], batch['actions']
            )
            
            # Policy loss (PPO clipped objective)
            ratio = torch.exp(log_prob - batch['old_log_probs'])
            surr1 = ratio * batch['advantages']
            surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 
                              1 + self.config.clip_epsilon) * batch['advantages']
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = nn.MSELoss()(value.squeeze(), batch['returns'])
            
            # Entropy loss (for exploration)
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = (policy_loss + 
                   self.config.value_coef * value_loss + 
                   self.config.entropy_coef * entropy_loss)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.max_grad_norm)
            self.optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropy_losses.append(entropy_loss.item())
            
        # Clear buffer
        self.buffer.clear()
        
        return {
            'total_loss': total_loss / self.config.n_epochs,
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses)
        }
        
    def train_episode(self, env: GridTradingEnvironment) -> Dict[str, float]:
        """Train for one episode"""
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(self.config.max_episode_steps):
            # Select action
            action, log_prob, value = self.select_action(obs)
            
            # Take step in environment
            next_obs, reward, done, info = env.step(action)
            
            # Store experience
            self.buffer.store(obs, action, reward, value, log_prob, done)
            
            # Update tracking
            episode_reward += reward
            episode_length += 1
            obs = next_obs
            
            if done:
                break
                
        # Get episode metrics
        episode_metrics = env.get_episode_metrics()
        
        # Store episode statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_composite_scores.append(episode_metrics['composite_score'])
        
        return episode_metrics
        
    def evaluate(self, env: GridTradingEnvironment, n_episodes: int = 5, deterministic: bool = True, epsilon_explore: float = 0.0) -> Dict[str, float]:
        """Evaluate the agent"""
        eval_rewards = []
        eval_composite_scores = []
        eval_metrics = []
        action_counts = {0: 0, 1: 0, 2: 0}  # BUY, SELL, HOLD
        
        for _ in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(self.config.max_episode_steps):
                action, _, _ = self.select_action(obs, deterministic=deterministic)
                if isinstance(action, np.ndarray):
                    a = int(action)
                else:
                    a = int(action)
                # Epsilon-greedy diagnostic exploration
                if epsilon_explore > 0.0 and random.random() < epsilon_explore:
                    if env.current_position is None:
                        # Force an entry action randomly between BUY(0)/SELL(1)
                        a = random.choice([0, 1])
                    else:
                        # Prefer HOLD to allow TP/SL logic to work naturally
                        a = 2
                if a in action_counts:
                    action_counts[a] += 1
                obs, reward, done, info = env.step(a)
                episode_reward += reward
                
                if done:
                    break
                    
            # Get episode metrics
            episode_metrics = env.get_episode_metrics()
            # Print per-trade details (up to 5)
            trades = getattr(env.metrics, 'trades', [])
            if trades:
                print("\nTrades (up to 5):")
                for i, t in enumerate(trades[:5], start=1):
                    et = t.get('entry_time', '')
                    xt = t.get('exit_time', '')
                    print(
                        f"Trade {i}: {t.get('position_type','?')} "
                        f"entry={t['entry_price']:.2f} ({et}) "
                        f"exit={t['exit_price']:.2f} ({xt}) "
                        f"pnl={t['pnl']:.2f} ({t['return_pct']*100:.2f}%)"
                    )
            eval_rewards.append(episode_reward)
            eval_composite_scores.append(episode_metrics['composite_score'])
            eval_metrics.append(episode_metrics)
            
        # Aggregate results (robust to missing keys across episodes)
        avg_metrics = {}
        if eval_metrics:
            all_keys = set()
            for m in eval_metrics:
                try:
                    all_keys.update(m.keys())
                except Exception:
                    pass
            for key in sorted(all_keys):
                try:
                    avg_metrics[f'eval_{key}'] = float(np.mean([m.get(key, 0.0) for m in eval_metrics]))
                except Exception:
                    # Fallback if any value is non-numeric
                    avg_metrics[f'eval_{key}'] = 0.0
        
        if eval_rewards:
            avg_metrics['eval_reward'] = float(np.mean(eval_rewards))
        if eval_composite_scores:
            avg_metrics['eval_composite_score'] = float(np.mean(eval_composite_scores))
        
        # Diagnostics: action distribution and trades
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            buy_p = action_counts[0] / total_actions
            sell_p = action_counts[1] / total_actions
            hold_p = action_counts[2] / total_actions
            print(f"\n[Diag] Action counts: BUY={action_counts[0]}, SELL={action_counts[1]}, HOLD={action_counts[2]} (HOLD%={hold_p:.2%})")
        # Average trades across episodes
        if eval_metrics:
            avg_trades = np.mean([m.get('total_trades', 0) for m in eval_metrics])
            print(f"[Diag] Avg trades per eval episode: {avg_trades:.2f}")
            any_position_opened = any(m.get('total_trades', 0) > 0 for m in eval_metrics)
            print(f"[Diag] Any position opened: {any_position_opened}")
        
        return avg_metrics
        
    def save_model(self, filepath: str, episode: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'metrics': metrics,
            'best_composite_score': self.best_composite_score
        }
        
        torch.save(checkpoint, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath: str):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)

        # If checkpoint contains a saved config, rebuild the network to match its shapes
        chk_cfg = checkpoint.get('config', None)
        if isinstance(chk_cfg, dict):
            input_features = chk_cfg.get('input_features', self.config.input_features)
            sequence_length = chk_cfg.get('sequence_length', self.config.sequence_length)
            hidden_dim = chk_cfg.get('hidden_dim', self.config.hidden_dim)
            action_dim = chk_cfg.get('action_dim', self.config.action_dim)

            # Recreate network with saved dimensions
            self.network = PPONetwork(
                input_features=input_features,
                sequence_length=sequence_length,
                hidden_dim=hidden_dim,
                action_dim=action_dim
            ).to(self.device)
            # Also refresh optimizer to point to new params
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.config.learning_rate)

        # Load weights (strict by default; fallback to non-strict if keys have minor diffs)
        try:
            self.network.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except Exception as e:
            print(f"Warning: strict load failed ({e}); trying non-strict...")
            self.network.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Load optimizer state if available
        opt_state = checkpoint.get('optimizer_state_dict')
        if opt_state is not None:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception as e:
                print(f"Warning: could not load optimizer state: {e}")

        self.best_composite_score = checkpoint.get('best_composite_score', -float('inf'))
        
        print(f"Model loaded from {filepath}")
        return checkpoint.get('episode', -1), checkpoint.get('metrics', {})
        
    def get_training_stats(self) -> Dict[str, float]:
        """Get current training statistics"""
        if not self.episode_rewards:
            return {}
            
        return {
            'avg_reward': np.mean(self.episode_rewards),
            'avg_episode_length': np.mean(self.episode_lengths),
            'avg_composite_score': np.mean(self.episode_composite_scores),
            'best_composite_score': self.best_composite_score
        }

class PPOTrainer:
    """PPO Training Manager"""
    
    def __init__(self, config: PPOConfig, data_query: DataQuery, 
                 train_start_date: str = None, train_end_date: str = None,
                 val_start_date: str = None, val_end_date: str = None,
                 env_kwargs: Optional[dict] = None):
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize agent
        self.agent = PPOAgent(config, self.device)
        
        # Extra environment configuration
        self.env_kwargs = env_kwargs or {}

        # Initialize environments
        self.train_env = GridTradingEnvironment(
            data_query,
            optimize_for_composite=self.config.optimize_for_composite,
            **self.env_kwargs
        )
        self.val_env = GridTradingEnvironment(
            data_query,
            optimize_for_composite=self.config.optimize_for_composite,
            **self.env_kwargs
        )
        
        # Load data
        print("Loading training data...")
        self.train_env.load_data(train_start_date, train_end_date)
        
        print("Loading validation data...")
        self.val_env.load_data(val_start_date, val_end_date)
        
        # Training tracking
        self.training_log = []
        self.episode_count = 0
        
        # Create checkpoints directory
        os.makedirs('checkpoints', exist_ok=True)
        
    def train(self, n_episodes: int):
        """Main training loop"""
        print(f"Starting PPO training for {n_episodes} episodes...")
        print(f"Device: {self.device}")
        
        for episode in range(n_episodes):
            self.episode_count += 1
            
            # Train episode
            episode_metrics = self.agent.train_episode(self.train_env)
            
            # Update policy every n_steps
            if self.agent.buffer.full:
                update_info = self.agent.update()
                episode_metrics.update(update_info)
                
            # Evaluation
            if episode % self.config.eval_freq == 0:
                eval_metrics = self.agent.evaluate(self.val_env)
                episode_metrics.update(eval_metrics)
                
                # Check for best model
                current_score = eval_metrics['eval_composite_score']
                if current_score > self.agent.best_composite_score:
                    self.agent.best_composite_score = current_score
                    best_model_path = f'checkpoints/best_model_episode_{episode}.pt'
                    self.agent.save_model(best_model_path, episode, eval_metrics)
                    self.agent.best_model_path = best_model_path
                    
            # Regular checkpointing
            if episode % self.config.save_freq == 0:
                checkpoint_path = f'checkpoints/checkpoint_episode_{episode}.pt'
                self.agent.save_model(checkpoint_path, episode, episode_metrics)
                
            # Logging
            self.training_log.append({
                'episode': episode,
                'timestamp': datetime.now().isoformat(),
                **episode_metrics
            })
            
            # Print progress
            if episode % 10 == 0:
                stats = self.agent.get_training_stats()
                print(f"Episode {episode:4d} | "
                      f"Reward: {stats.get('avg_reward', 0):.3f} | "
                      f"Composite: {stats.get('avg_composite_score', 0):.3f} | "
                      f"Best: {stats.get('best_composite_score', 0):.3f}")
                      
        print("Training completed!")
        
        # Save final model
        final_model_path = f'checkpoints/final_model_episode_{n_episodes}.pt'
        final_metrics = self.agent.get_training_stats()
        self.agent.save_model(final_model_path, n_episodes, final_metrics)
        
        # Save training log
        with open('training_log.json', 'w') as f:
            json.dump(self.training_log, f, indent=2)
            
        return self.agent.best_model_path
        
    def backtest(self, model_path: str = None, test_start_date: str = None, 
                 test_end_date: str = None, stochastic_eval: bool = False,
                 eval_episodes: int = 1, epsilon_explore: float = 0.0) -> Dict[str, float]:
        """Backtest the trained model"""
        if model_path:
            self.agent.load_model(model_path)
            
        # Create test environment
        test_env = GridTradingEnvironment(self.train_env.data_query, **(self.env_kwargs or {}))
        test_env.load_data(test_start_date, test_end_date)
        
        print("Running backtest...")
        backtest_results = self.agent.evaluate(
            test_env,
            n_episodes=eval_episodes,
            deterministic=(not stochastic_eval),
            epsilon_explore=epsilon_explore
        )
        # Attach first 5 trades (if any) for reporting
        try:
            trades = getattr(test_env.metrics, 'trades', [])
            if trades:
                backtest_results['trades'] = trades[:5]
                print("\nFirst 5 trades:")
                for i, t in enumerate(trades[:5], start=1):
                    et = t.get('entry_time', '')
                    xt = t.get('exit_time', '')
                    print(
                        f"Trade {i}: {t.get('position_type','?')} "
                        f"entry={t['entry_price']:.2f} ({et}) "
                        f"exit={t['exit_price']:.2f} ({xt}) "
                        f"pnl={t['pnl']:.2f} ({t['return_pct']*100:.2f}%)"
                    )
                # Export full trades list to results file
                os.makedirs('results', exist_ok=True)
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                trades_path = os.path.join('results', f"backtest_trades_{ts}.json")
                with open(trades_path, 'w') as f:
                    json.dump(trades, f, indent=2)
                backtest_results['trades_file'] = trades_path
        except Exception as _:
            pass

        # Export equity curve and drawdowns with timestamps for plotting/reporting
        try:
            eq_curve = list(getattr(test_env.metrics, 'equity_curve', []) or [])
            dd_curve = list(getattr(test_env.metrics, 'drawdowns', []) or [])
            if eq_curve:
                # Attempt to align timestamps from loaded data rows corresponding to equity updates
                # Each env.step() appends one equity point; mapping starts at window_size index
                dts = None
                try:
                    start_idx = int(getattr(test_env, 'window_size', 96))
                    end_idx = min(start_idx + len(eq_curve), len(getattr(test_env, 'data', [])))
                    if hasattr(test_env, 'data') and 'datetime' in test_env.data.columns and end_idx > start_idx:
                        dts = test_env.data.iloc[start_idx:end_idx]['datetime'].astype(str).tolist()
                except Exception:
                    dts = None

                # Save full time series to results file
                os.makedirs('results', exist_ok=True)
                ts2 = datetime.now().strftime('%Y%m%d_%H%M%S')
                equity_path = os.path.join('results', f"backtest_equity_{ts2}.json")
                payload = {
                    'datetime': dts,
                    'equity': eq_curve,
                    'drawdown': dd_curve
                }
                with open(equity_path, 'w') as f:
                    json.dump(payload, f, indent=2)
                # Include short previews directly in results for quick inspection
                backtest_results['equity_file'] = equity_path
                backtest_results['equity_points'] = len(eq_curve)
                if dts:
                    backtest_results['equity_start'] = dts[0]
                    backtest_results['equity_end'] = dts[-1]
        except Exception:
            pass
        
        print("\n=== Backtest Results ===")
        for key, value in backtest_results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value}")
        
        # Concise results report
        win_rate = backtest_results.get('eval_win_rate', None)
        comp = backtest_results.get('eval_composite_score', None)
        trades_per_day = backtest_results.get('eval_trade_frequency', None)
        if win_rate is not None or comp is not None or trades_per_day is not None:
            print("\n--- Results Report ---")
            if win_rate is not None:
                print(f"Win Rate: {win_rate:.4f}")
            if comp is not None:
                print(f"Composite Score: {comp:.4f}")
            if trades_per_day is not None:
                print(f"Trades/Day: {trades_per_day:.4f}")

        # Full composite breakdown
        try:
            import numpy as _np
            s = float(backtest_results.get('eval_sortino_ratio', 0.0))
            c = float(backtest_results.get('eval_calmar_ratio', 0.0))
            pf = float(backtest_results.get('eval_profit_factor', 0.0))
            wr = float(backtest_results.get('eval_win_rate', 0.0))
            mdd = float(backtest_results.get('eval_max_drawdown', 0.0))
            tf = float(backtest_results.get('eval_trade_frequency', 0.0))
            tt = float(backtest_results.get('eval_total_trades', 0.0))

            sortino_norm = min(s / 2.0, 1.0)
            calmar_norm = min(c / 3.0, 1.0)
            pf_capped = pf if _np.isfinite(pf) else 2.0
            pf_norm = min(pf_capped / 2.0, 1.0)
            win_rate_norm = min(max(wr, 0.0), 1.0)
            mdd_inv = 1.0 / (1.0 + abs(mdd))
            tf_norm = min(tf / 4.0, 1.0)

            base_score = (
                0.28 * sortino_norm +
                0.22 * calmar_norm +
                0.20 * pf_norm +
                0.15 * win_rate_norm +
                0.10 * mdd_inv +
                0.05 * tf_norm
            )
            activity_penalty = min(1.0, tt / 4.0)
            final_score = max(0.0, min(1.0, base_score * activity_penalty))

            print("\n--- Composite Breakdown ---")
            # Table header
            print(f"{'Metric':<16} {'Raw':>10} {'Normalized':>12} {'Weight':>8} {'Contribution':>14}")
            print("-" * 64)
            # Rows
            print(f"{'Sortino':<16} {s:>10.4f} {sortino_norm:>12.4f} {0.28:>8.2f} {(0.28*sortino_norm):>14.4f}")
            print(f"{'Calmar':<16} {c:>10.4f} {calmar_norm:>12.4f} {0.22:>8.2f} {(0.22*calmar_norm):>14.4f}")
            print(f"{'Profit Factor':<16} {pf if _np.isfinite(pf) else float('inf'):>10} {pf_norm:>12.4f} {0.20:>8.2f} {(0.20*pf_norm):>14.4f}")
            # SQN row (not part of composite weighting)
            sqn = float(backtest_results.get('eval_sqn', backtest_results.get('sqn', 0.0)))
            print(f"{'SQN':<16} {sqn:>10.4f} {'-':>12} {0.00:>8.2f} {0.0:>14.4f}")
            print(f"{'Win Rate':<16} {wr:>10.4f} {win_rate_norm:>12.4f} {0.15:>8.2f} {(0.15*win_rate_norm):>14.4f}")
            print(f"{'MaxDD Inv':<16} {mdd:>10.4f} {mdd_inv:>12.4f} {0.10:>8.2f} {(0.10*mdd_inv):>14.4f}")
            print(f"{'Trades/Day':<16} {tf:>10.4f} {tf_norm:>12.4f} {0.05:>8.2f} {(0.05*tf_norm):>14.4f}")
            print("-" * 64)
            print(f"{'Base composite':<16} {'':>10} {'':>12} {'':>8} {base_score:>14.4f}")
            print(f"{'Activity penalty':<16} {'':>10} {'':>12} {'':>8} {activity_penalty:>14.4f}")
            print(f"{'Final composite':<16} {'':>10} {'':>12} {'':>8} {final_score:>14.4f}")
            # Equity summary
            eq_final = float(backtest_results.get('eval_final_balance', backtest_results.get('final_balance', 0.0)))
            ep_ret = float(backtest_results.get('eval_episode_return', backtest_results.get('episode_return', 0.0)))
            eq_start = eq_final / (1.0 + ep_ret) if (1.0 + ep_ret) != 0 else 0.0
            print("-" * 64)
            print(f"{'Equity Start':<16} {eq_start:>10.2f}")
            print(f"{'Equity Final':<16} {eq_final:>10.2f}")
            print(f"{'Episode Return':<16} {ep_ret*100:>9.2f}%")
        except Exception as e:
            print(f"\nComposite breakdown unavailable: {e}")
                
        return backtest_results

def main():
    """Example training script"""
    # Configuration
    config = PPOConfig(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        max_episode_steps=500
    )
    
    # Initialize data query
    data_query = DataQuery()
    
    # Define data splits (adjust dates based on your data)
    train_start = "2020-08-01 00:00:00"
    train_end = "2022-08-01 00:00:00"
    val_start = "2022-08-01 00:00:00"
    val_end = "2023-08-01 00:00:00"
    test_start = "2023-08-01 00:00:00"
    test_end = "2024-08-01 00:00:00"
    
    # Initialize trainer
    trainer = PPOTrainer(
        config=config,
        data_query=data_query,
        train_start_date=train_start,
        train_end_date=train_end,
        val_start_date=val_start,
        val_end_date=val_end
    )
    
    # Train the model
    best_model_path = trainer.train(n_episodes=1000)
    
    # Backtest
    backtest_results = trainer.backtest(
        model_path=best_model_path,
        test_start_date=test_start,
        test_end_date=test_end
    )
    
    print(f"\nBest model saved at: {best_model_path}")
    print(f"Final composite score: {backtest_results['eval_composite_score']:.4f}")

if __name__ == "__main__":
    main()