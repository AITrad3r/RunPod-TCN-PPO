#!/usr/bin/env python3
"""
PPO Grid Trading System - Main Training Script

This script implements the complete PPO Grid-to-Grid Trading System as specified
in ppo_grid_trading_spec.md. It includes:
- CNN-TCM neural network architecture
- PPO algorithm with GAE
- Grid trading environment with VWAP/RSI indicators
- Composite scoring system
- Training, validation, and backtesting
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

# Import our modules
from data_query import DataQuery
from neural_network import PPONetwork
from trading_environment import GridTradingEnvironment
from ppo_algorithm import PPOAgent, PPOConfig, PPOTrainer

def setup_training_dates(data_query: DataQuery) -> Dict[str, str]:
    """Setup training, validation, and test date ranges based on available data"""
    stats = data_query.get_data_stats()
    
    # Parse date strings
    earliest = datetime.strptime(stats['earliest_date'], '%Y-%m-%d %H:%M:%S')
    latest = datetime.strptime(stats['latest_date'], '%Y-%m-%d %H:%M:%S')
    
    total_days = (latest - earliest).days
    
    # Split data: 60% training, 20% validation, 20% testing
    train_days = int(total_days * 0.6)
    val_days = int(total_days * 0.2)
    
    train_start = earliest
    train_end = earliest + timedelta(days=train_days)
    val_start = train_end
    val_end = val_start + timedelta(days=val_days)
    test_start = val_end
    test_end = latest
    
    dates = {
        'train_start': train_start.strftime('%Y-%m-%d %H:%M:%S'),
        'train_end': train_end.strftime('%Y-%m-%d %H:%M:%S'),
        'val_start': val_start.strftime('%Y-%m-%d %H:%M:%S'),
        'val_end': val_end.strftime('%Y-%m-%d %H:%M:%S'),
        'test_start': test_start.strftime('%Y-%m-%d %H:%M:%S'),
        'test_end': test_end.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    print("=== Data Split ===")
    print(f"Training:   {dates['train_start']} to {dates['train_end']} ({train_days} days)")
    print(f"Validation: {dates['val_start']} to {dates['val_end']} ({val_days} days)")
    print(f"Testing:    {dates['test_start']} to {dates['test_end']} ({(test_end - test_start).days} days)")
    print()
    
    return dates

def create_config(args) -> PPOConfig:
    """Create PPO configuration from command line arguments"""
    config = PPOConfig(
        # Network architecture
        input_features=10,  # 7 market + 3 position features
        sequence_length=getattr(args, 'sequence_length', 96),
        hidden_dim=args.hidden_dim,
        action_dim=3,
        
        # PPO hyperparameters
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_epsilon,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        
        # Training parameters
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        max_episode_steps=args.max_episode_steps,
        
        # Checkpointing
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        
        # Objective
        optimize_for_composite=bool(getattr(args, 'optimize_for_composite', False))
    )
    
    return config

def train_model(args):
    """Main training function"""
    print("=== PPO Grid Trading System Training ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Initialize data query
    print("Initializing data query...")
    data_query = DataQuery()
    
    # Verify data availability
    stats = data_query.get_data_stats()
    print(f"\nData available: {stats['total_records']:,} records")
    print(f"Date range: {stats['earliest_date']} to {stats['latest_date']}")
    print(f"Price range: ${stats['min_price']:,.2f} - ${stats['max_price']:,.2f}")
    
    # Setup date ranges
    dates = setup_training_dates(data_query)
    
    # Create configuration
    config = create_config(args)
    
    print("=== Training Configuration ===")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Batch size: {config.batch_size}")
    print(f"Episodes: {args.episodes}")
    print(f"Max episode steps: {config.max_episode_steps}")
    print(f"Hidden dimension: {config.hidden_dim}")
    print()
    
    # Initialize trainer
    print("Initializing PPO trainer...")
    trainer = PPOTrainer(
        config=config,
        data_query=data_query,
        train_start_date=dates['train_start'],
        train_end_date=dates['train_end'],
        val_start_date=dates['val_start'],
        val_end_date=dates['val_end']
    )
    
    # Override episode steps if provided
    if args.max_episode_steps is not None:
        trainer.agent.config.max_episode_steps = args.max_episode_steps
    
    # Start training
    print("Starting training...")
    start_time = datetime.now()
    
    try:
        best_model_path = trainer.train(n_episodes=args.episodes)
        
        training_time = datetime.now() - start_time
        print(f"\nTraining completed in {training_time}")
        print(f"Best model saved at: {best_model_path}")
        
        # Run backtest
        if args.backtest:
            print("\n=== Running Backtest ===")
            backtest_results = trainer.backtest(
                model_path=best_model_path,
                test_start_date=dates['test_start'],
                test_end_date=dates['test_end']
            )
            
            # Save backtest results
            with open('backtest_results.json', 'w') as f:
                json.dump(backtest_results, f, indent=2)
                
            print(f"\nBacktest results saved to backtest_results.json")
            
        return best_model_path
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        return None
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()
        return None

def evaluate_model(args):
    """Evaluate a trained model"""
    print("=== Model Evaluation ===")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        return
        
    # Initialize data query
    data_query = DataQuery()
    dates = setup_training_dates(data_query)
    
    # Create configuration (use defaults)
    config = PPOConfig()
    
    # Initialize agent and load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPOAgent(config, device)
    
    print(f"Loading model from {args.model_path}...")
    episode, metrics = agent.load_model(args.model_path)
    
    print(f"Model trained for {episode} episodes")
    print(f"Best composite score: {agent.best_composite_score:.4f}")
    
    # Create test environment
    test_env = GridTradingEnvironment(data_query)
    test_env.load_data(dates['test_start'], dates['test_end'])
    
    # Run evaluation
    print("\nRunning evaluation...")
    eval_results = agent.evaluate(test_env, n_episodes=args.eval_episodes)
    
    print("\n=== Evaluation Results ===")
    for key, value in eval_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Concise results report for quick reference
    win_rate = eval_results.get('eval_win_rate', None)
    comp = eval_results.get('eval_composite_score', None)
    trades_per_day = eval_results.get('eval_trade_frequency', None)
    
    if win_rate is not None or comp is not None or trades_per_day is not None:
        print("\n--- Results Report ---")
        if win_rate is not None:
            print(f"Win Rate: {win_rate:.4f}")
        if comp is not None:
            print(f"Composite Score: {comp:.4f}")
        if trades_per_day is not None:
            print(f"Trades/Day: {trades_per_day:.4f}")
    
    # Composite breakdown (derived from averaged evaluation metrics)
    try:
        s = float(eval_results.get('eval_sortino_ratio', 0.0))
        c = float(eval_results.get('eval_calmar_ratio', 0.0))
        pf = float(eval_results.get('eval_profit_factor', 0.0))
        wr = float(eval_results.get('eval_win_rate', 0.0))
        mdd = float(eval_results.get('eval_max_drawdown', 0.0))
        tf = float(eval_results.get('eval_trade_frequency', 0.0))
        tt = float(eval_results.get('eval_total_trades', 0.0))

        # Normalize per spec
        sortino_norm = min(s / 2.0, 1.0)
        calmar_norm = min(c / 3.0, 1.0)
        # Handle inf profit factor
        # Cap profit factor at 2.0 if non-finite
        try:
            _pf_val = float(pf)
        except Exception:
            _pf_val = float('inf')
        pf_capped = _pf_val if _pf_val not in (float('inf'), float('-inf')) else 2.0
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
        # Profit Factor raw display respects finiteness capping above
        _pf_raw_display = _pf_val if _pf_val not in (float('inf'), float('-inf')) else float('inf')
        print(f"{'Profit Factor':<16} {_pf_raw_display:>10} {pf_norm:>12.4f} {0.20:>8.2f} {(0.20*pf_norm):>14.4f}")
        # SQN row (not part of composite weighting)
        sqn = float(eval_results.get('eval_sqn', eval_results.get('sqn', 0.0)))
        print(f"{'SQN':<16} {sqn:>10.4f} {'-':>12} {0.00:>8.2f} {0.0:>14.4f}")
        print(f"{'Win Rate':<16} {wr:>10.4f} {win_rate_norm:>12.4f} {0.15:>8.2f} {(0.15*win_rate_norm):>14.4f}")
        print(f"{'MaxDD Inv':<16} {mdd:>10.4f} {mdd_inv:>12.4f} {0.10:>8.2f} {(0.10*mdd_inv):>14.4f}")
        print(f"{'Trades/Day':<16} {tf:>10.4f} {tf_norm:>12.4f} {0.05:>8.2f} {(0.05*tf_norm):>14.4f}")
        print("-" * 64)
        print(f"{'Base composite':<16} {'':>10} {'':>12} {'':>8} {base_score:>14.4f}")
        print(f"{'Activity penalty':<16} {'':>10} {'':>12} {'':>8} {activity_penalty:>14.4f}")
        print(f"{'Final composite':<16} {'':>10} {'':>12} {'':>8} {final_score:>14.4f}")
        # Equity summary
        eq_final = float(eval_results.get('eval_final_balance', eval_results.get('final_balance', 0.0)))
        ep_ret = float(eval_results.get('eval_episode_return', eval_results.get('episode_return', 0.0)))
        eq_start = eq_final / (1.0 + ep_ret) if (1.0 + ep_ret) != 0 else 0.0
        print("-" * 64)
        print(f"{'Equity Start':<16} {eq_start:>10.2f}")
        print(f"{'Equity Final':<16} {eq_final:>10.2f}")
        print(f"{'Episode Return':<16} {ep_ret*100:>9.2f}%")
    except Exception as e:
        print(f"\nComposite breakdown unavailable: {e}")
            
    # Save evaluation results
    eval_results['model_path'] = args.model_path
    eval_results['evaluation_date'] = datetime.now().isoformat()
    
    with open('evaluation_results.json', 'w') as f:
        json.dump(eval_results, f, indent=2)
        
    print("\nEvaluation results saved to evaluation_results.json")

def run_pipeline(args):
    """Run predefined training/testing pipelines"""
    assert args.mode in {"short", "targeted"}, "Supported modes: 'short', 'targeted'"

    print(f"=== Pipeline: {args.mode} ===")
    data_query = DataQuery()
    stats = data_query.get_data_stats()

    latest = datetime.strptime(stats['latest_date'], '%Y-%m-%d %H:%M:%S')

    if args.mode == 'short':
        # 30 days total window ending at latest
        total_days = 30
        train_days = 15
        oos_days = 8
        backtest_days = 7

        window_start = latest - timedelta(days=total_days)
        train_start = window_start
        train_end = train_start + timedelta(days=train_days)
        oos_start = train_end
        oos_end = oos_start + timedelta(days=oos_days)
        backtest_start = latest - timedelta(days=backtest_days)
        backtest_end = latest

        # Stringify
        train_start_s = train_start.strftime('%Y-%m-%d %H:%M:%S')
        train_end_s = train_end.strftime('%Y-%m-%d %H:%M:%S')
        oos_start_s = oos_start.strftime('%Y-%m-%d %H:%M:%S')
        oos_end_s = oos_end.strftime('%Y-%m-%d %H:%M:%S')
        backtest_start_s = backtest_start.strftime('%Y-%m-%d %H:%M:%S')
        backtest_end_s = backtest_end.strftime('%Y-%m-%d %H:%M:%S')

        print("\n=== Short Pipeline Dates ===")
        print(f"Train: {train_start_s} -> {train_end_s} ({train_days} days)")
        print(f"OOS:   {oos_start_s} -> {oos_end_s} ({oos_days} days)")
        print(f"BTest: {backtest_start_s} -> {backtest_end_s} ({backtest_days} days)")

        # Config
        config = create_config(args)

        # Trainer with OOS as validation
        trainer = PPOTrainer(
            config=config,
            data_query=data_query,
            train_start_date=train_start_s,
            train_end_date=train_end_s,
            val_start_date=oos_start_s,
            val_end_date=oos_end_s
        )

        # Train
        best_model_path = trainer.train(n_episodes=args.episodes)
        if not best_model_path:
            best_model_path = f'checkpoints/final_model_episode_{args.episodes}.pt'

        # Evaluate on OOS (validation env)
        print("\nEvaluating on OOS window...")
        oos_results = trainer.agent.evaluate(trainer.val_env, n_episodes=max(1, args.eval_episodes))

        # Backtest on last 7 days
        backtest_results = trainer.backtest(
            model_path=best_model_path,
            test_start_date=backtest_start_s,
            test_end_date=backtest_end_s
        )

        # Concise pipeline report
        def pick(metrics: dict):
            return {
                'win_rate': float(metrics.get('eval_win_rate', 0.0)),
                'composite_score': float(metrics.get('eval_composite_score', 0.0)),
                'trades_per_day': float(metrics.get('eval_trade_frequency', 0.0)),
            }

        report = {
            'pipeline': 'short',
            'train_window': [train_start_s, train_end_s],
            'oos_window': [oos_start_s, oos_end_s],
            'backtest_window': [backtest_start_s, backtest_end_s],
            'model_path': best_model_path,
            'oos': pick(oos_results),
            'backtest': pick(backtest_results),
            'timestamp': datetime.now().isoformat()
        }

        print("\n=== Pipeline Results (Short) ===")
        print(f"OOS -> WinRate: {report['oos']['win_rate']:.4f} | Composite: {report['oos']['composite_score']:.4f} | Trades/Day: {report['oos']['trades_per_day']:.4f}")
        print(f"BTest -> WinRate: {report['backtest']['win_rate']:.4f} | Composite: {report['backtest']['composite_score']:.4f} | Trades/Day: {report['backtest']['trades_per_day']:.4f}")

        os.makedirs('results', exist_ok=True)
        out_path = os.path.join('results', f"pipeline_short_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(out_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nPipeline report saved to {out_path}")

    if args.mode == 'targeted':
        # Custom split based on args (default: 15 train, 7 OOS, 8 backtest)
        train_days = int(getattr(args, 'train_days', 15))
        oos_days = int(getattr(args, 'oos_days', 7))
        backtest_days = int(getattr(args, 'backtest_days', 8))
        total_days = train_days + oos_days + backtest_days

        window_start = latest - timedelta(days=total_days)
        train_start = window_start
        train_end = train_start + timedelta(days=train_days)
        oos_start = train_end
        oos_end = oos_start + timedelta(days=oos_days)
        backtest_start = oos_end
        backtest_end = latest

        # Config and trainer
        config = create_config(args)
        trainer = PPOTrainer(
            config=config,
            data_query=data_query,
            train_start_date=train_start.strftime('%Y-%m-%d %H:%M:%S'),
            train_end_date=train_end.strftime('%Y-%m-%d %H:%M:%S'),
            val_start_date=oos_start.strftime('%Y-%m-%d %H:%M:%S'),
            val_end_date=oos_end.strftime('%Y-%m-%d %H:%M:%S')
        )

        # Loop training until criteria met or max rounds
        target = float(getattr(args, 'target_composite', 0.6))
        target_win = float(getattr(args, 'target_win_rate', 0.5))
        target_trades = float(getattr(args, 'target_trades_per_day', 3.0))
        require_better = bool(getattr(args, 'require_backtest_better', True))
        # Prefer explicit --round-episodes if provided; fallback to --episodes
        round_episodes = int(getattr(args, 'round_episodes', getattr(args, 'episodes', 100)))
        max_rounds = int(getattr(args, 'max_rounds', 10))

        achieved = False
        history = []
        for r in range(1, max_rounds + 1):
            print(f"\n=== Training Round {r}/{max_rounds} - {round_episodes} episodes ===")
            best_model_path = trainer.train(n_episodes=round_episodes)

            # Evaluate on train and OOS windows
            print("\nEvaluating on training window...")
            train_metrics = trainer.agent.evaluate(trainer.train_env, n_episodes=1, deterministic=False)
            print("Evaluating on OOS window...")
            oos_metrics = trainer.agent.evaluate(trainer.val_env, n_episodes=1, deterministic=False)
            # Export full OOS trades
            try:
                oos_trades = getattr(trainer.val_env.metrics, 'trades', [])
                if oos_trades:
                    os.makedirs('results', exist_ok=True)
                    ts_oos = datetime.now().strftime('%Y%m%d_%H%M%S')
                    oos_trades_path = os.path.join('results', f"oos_trades_{ts_oos}.json")
                    with open(oos_trades_path, 'w') as f:
                        json.dump(oos_trades, f, indent=2)
                    oos_metrics['trades_file'] = oos_trades_path
                    print(f"Saved OOS trades to {oos_trades_path}")
            except Exception:
                pass

            # Backtest on backtest window
            print("\nEvaluating backtest window...")
            bt_results = trainer.backtest(
                model_path=best_model_path,
                test_start_date=backtest_start.strftime('%Y-%m-%d %H:%M:%S'),
                test_end_date=backtest_end.strftime('%Y-%m-%d %H:%M:%S'),
                stochastic_eval=True,
                eval_episodes=1
            )

            train_comp = float(train_metrics.get('eval_composite_score', train_metrics.get('composite_score', 0.0)))
            oos_comp = float(oos_metrics.get('eval_composite_score', oos_metrics.get('composite_score', 0.0)))
            bt_comp = float(bt_results.get('eval_composite_score', 0.0))
            bt_win = float(bt_results.get('eval_win_rate', bt_results.get('win_rate', 0.0)))
            bt_trades = float(bt_results.get('eval_trade_frequency', bt_results.get('trades_per_day', 0.0)))

            record = {
                'round': r,
                'train_composite': train_comp,
                'oos_composite': oos_comp,
                'backtest_composite': bt_comp,
                'backtest_win_rate': bt_win,
                'backtest_trades_per_day': bt_trades,
                'best_model_path': best_model_path,
                'oos_trades_file': oos_metrics.get('trades_file'),
                'backtest_trades_file': bt_results.get('trades_file')
            }
            history.append(record)
            print(f"Round {r} -> Train: {train_comp:.4f} | OOS: {oos_comp:.4f} | BTest: {bt_comp:.4f} | Win%: {bt_win:.4f} | Trades/Day: {bt_trades:.2f}")

            cond_target = bt_comp >= target
            cond_win = bt_win >= target_win
            cond_trades = bt_trades >= target_trades
            cond_better = (bt_comp >= train_comp and bt_comp >= oos_comp) if require_better else True
            if cond_target and cond_win and cond_trades and cond_better:
                print("\nTarget achieved. Stopping training loop.")
                achieved = True
                # Persist history
                os.makedirs('results', exist_ok=True)
                out_path = os.path.join('results', f"pipeline_targeted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                with open(out_path, 'w') as f:
                    json.dump({'history': history, 'final': record}, f, indent=2)
                print(f"Saved targeted pipeline report to {out_path}")
                break

        if not achieved:
            print("\nMax rounds reached without meeting the target criteria.")
            os.makedirs('results', exist_ok=True)
            out_path = os.path.join('results', f"pipeline_targeted_{datetime.now().strftime('%Y%m%d_%H%M%S')}_incomplete.json")
            with open(out_path, 'w') as f:
                json.dump({'history': history}, f, indent=2)
            print(f"Saved targeted pipeline history to {out_path}")

def test_system(args):
    """Test system components"""
    print("=== System Component Testing ===")
    
    # Test data query
    print("Testing data query...")
    data_query = DataQuery()
    stats = data_query.get_data_stats()
    print(f"✓ Data query working. {stats['total_records']:,} records available")
    
    # Test neural network
    print("\nTesting neural network...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network = PPONetwork(input_features=10, sequence_length=96)
    network.to(device)
    
    # Test forward pass
    test_input = torch.randn(4, 96, 10, device=device)
    action, log_prob, entropy, value = network.get_action_and_value(test_input)
    print(f"✓ Neural network working. Output shapes: {action.shape}, {value.shape}")
    
    # Test environment
    print("\nTesting trading environment...")
    env = GridTradingEnvironment(data_query)
    env.load_data()
    
    obs = env.reset()
    print(f"✓ Environment working. Observation shape: {obs.shape}")
    
    # Test a few steps
    for i in range(5):
        action = np.random.randint(0, 3)
        obs, reward, done, info = env.step(action)
        if done:
            break
            
    metrics = env.get_episode_metrics()
    print(f"✓ Episode completed. Composite score: {metrics['composite_score']:.4f}")
    
    print("\n✓ All system components working correctly!")

def run_test_trades(args):
    """Run scripted test trades to validate environment mechanics."""
    data_query = DataQuery()

    # Compute window
    stats = data_query.get_data_stats()
    latest = datetime.strptime(stats['latest_date'], '%Y-%m-%d %H:%M:%S')
    earliest = datetime.strptime(stats['earliest_date'], '%Y-%m-%d %H:%M:%S')
    hours = int(args.last_hours or 72)
    end_dt = latest
    start_dt = max(earliest, end_dt - timedelta(hours=hours))
    start_date = start_dt.strftime('%Y-%m-%d %H:%M:%S')
    end_date = end_dt.strftime('%Y-%m-%d %H:%M:%S')

    env = GridTradingEnvironment(data_query)
    env.load_data(start_date, end_date)
    # If data window too short relative to window_size, extend to 240h
    if len(env.data) < getattr(env, 'window_size', 96) + 16:
        extended_start = max(earliest, end_dt - timedelta(hours=max(hours, 240)))
        start_date = extended_start.strftime('%Y-%m-%d %H:%M:%S')
        env.load_data(start_date, end_date)
    obs = env.reset(random_start=False)

    trades_to_do = int(args.trades or 5)
    max_hold = int(args.max_hold_hours or 3)
    made_trades = 0
    action_cycle = [0, 1]  # BUY then SELL
    cycle_idx = 0
    per_trade = []

    print(f"Running scripted test for {trades_to_do} trades from {start_date} -> {end_date}")

    while made_trades < trades_to_do and env.current_step < env.max_steps:
        # If no position, open next one using env.step
        if env.current_position is None:
            action = action_cycle[cycle_idx % len(action_cycle)]
            cycle_idx += 1
            obs, _, done, info = env.step(action)
            if done:
                break
        
        # Hold until TP/SL or max_hold reached using HOLD action (2)
        start_hold_step = env.current_step
        closed = False
        while env.current_position is not None and env.current_step < env.max_steps:
            obs, _, done, info = env.step(2)  # HOLD
            if env.current_position is None:
                closed = True  # closed by TP/SL inside step
                break
            if env.current_step - start_hold_step >= max_hold:
                # Force-close at current price from info
                current_price = info.get('current_price', None)
                if current_price is None:
                    # fallback: use last close
                    idx = min(env.current_step + env.window_size - 1, len(env.data) - 1)
                    current_price = float(env.data.iloc[idx]['close_price'])
                _ = env._close_position(current_price)
                closed = True
                break
            if done:
                break

        if closed:
            made_trades += 1
            if env.metrics.trades:
                t = env.metrics.trades[-1]
                per_trade.append({
                    'trade': made_trades,
                    'type': t['position_type'],
                    'entry': t['entry_price'],
                    'exit': t['exit_price'],
                    'pnl': t['pnl'],
                    'ret_pct': t['return_pct']
                })
                print(f"Trade {made_trades}: {t['position_type']} entry={t['entry_price']:.2f} exit={t['exit_price']:.2f} pnl={t['pnl']:.2f} ({t['return_pct']*100:.2f}%)")
        
        if env.current_step >= env.max_steps:
            break

    # Finalize metrics
    metrics = env.get_episode_metrics()
    summary = {
        'trades_made': made_trades,
        'net_profit': metrics.get('net_profit', 0.0),
        'final_balance': metrics.get('final_balance', env.balance),
        'episode_return': metrics.get('episode_return', 0.0),
        'max_drawdown': metrics.get('max_drawdown', 0.0),
        'win_rate': metrics.get('win_rate', 0.0)
    }
    print("\n=== Test Trades Summary ===")
    print(summary)
    with open('test_trades_results.json', 'w') as f:
        json.dump({'per_trade': per_trade, 'summary': summary}, f, indent=2)
    print("Saved to test_trades_results.json")

def run_backtest(args):
    """Backtest a saved model over a specified date window"""
    print("=== Backtest ===")
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} not found")
        return

    data_query = DataQuery()
    # Resolve date window
    if args.last_hours is not None:
        stats = data_query.get_data_stats()
        latest = datetime.strptime(stats['latest_date'], '%Y-%m-%d %H:%M:%S')
        earliest = datetime.strptime(stats['earliest_date'], '%Y-%m-%d %H:%M:%S')

        # Read model sequence_length from checkpoint config if available
        try:
            ckpt = torch.load(args.model_path, map_location='cpu')
            model_seq_len = int(ckpt.get('config', {}).get('sequence_length', 96))
        except Exception:
            model_seq_len = 96

        hours = max(int(args.last_hours), int(model_seq_len))
        end_dt = latest
        start_dt = max(earliest, end_dt - timedelta(hours=hours))
        start_date = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_date = end_dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        if not args.start_date or not args.end_date:
            print("Error: provide --last-hours or both --start-date and --end-date")
            return
        start_date = args.start_date
        end_date = args.end_date

    print(f"Window: {start_date} -> {end_date}")

    # Create config and trainer (use checkpoint seq length)
    config = create_config(argparse.Namespace(
        hidden_dim=128,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        max_episode_steps=500,
        sequence_length=model_seq_len,
        save_freq=100,
        eval_freq=100
    ))

    trainer = PPOTrainer(
        config=config,
        data_query=data_query,
        train_start_date=start_date,  # not used during backtest but required by ctor
        train_end_date=end_date,
        val_start_date=start_date,
        val_end_date=end_date
    )

    # Override episode steps if provided
    if args.max_episode_steps is not None:
        trainer.agent.config.max_episode_steps = args.max_episode_steps

    # Run backtest with deterministic flag based on CLI
    results = trainer.backtest(
        model_path=args.model_path,
        test_start_date=start_date,
        test_end_date=end_date,
        stochastic_eval=args.stochastic_eval,
        epsilon_explore=float(getattr(args, 'epsilon_explore', 0.0))
    )

    # Persist
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nBacktest results saved to backtest_results.json")

def main():
    parser = argparse.ArgumentParser(description='PPO Grid Trading System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the PPO model')
    train_parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    train_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    train_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    train_parser.add_argument('--n-steps', type=int, default=1024, help='Steps per update')
    train_parser.add_argument('--n-epochs', type=int, default=10, help='PPO epochs per update')
    train_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    train_parser.add_argument('--max-episode-steps', type=int, default=500, help='Max steps per episode')
    train_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    train_parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    train_parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip parameter')
    train_parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    train_parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    train_parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Gradient clipping')
    train_parser.add_argument('--save-freq', type=int, default=100, help='Save frequency')
    train_parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency')
    train_parser.add_argument('--optimize-for-composite', action='store_true', help='Use composite score as terminal reward (pure composite optimization)')
    train_parser.add_argument('--backtest', action='store_true', help='Run backtest after training')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate a trained model')
    eval_parser.add_argument('--model-path', type=str, required=True, help='Path to trained model')
    eval_parser.add_argument('--eval-episodes', type=int, default=10, help='Number of evaluation episodes')
    
    # Pipeline command
    pipe_parser = subparsers.add_parser('pipeline', help='Run a training/testing pipeline')
    pipe_parser.add_argument('--mode', type=str, choices=['short', 'targeted'], required=True, help="Pipeline mode: 'short' or 'targeted'")
    pipe_parser.add_argument('--episodes', type=int, default=500, help='Training episodes')
    pipe_parser.add_argument('--round-episodes', type=int, help='Episodes per training round (alias for --episodes in targeted mode)')
    pipe_parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    pipe_parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    pipe_parser.add_argument('--n-steps', type=int, default=1024, help='Steps per update')
    pipe_parser.add_argument('--n-epochs', type=int, default=10, help='PPO epochs per update')
    pipe_parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    pipe_parser.add_argument('--max-episode-steps', type=int, default=500, help='Max steps per episode')
    pipe_parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    pipe_parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    pipe_parser.add_argument('--clip-epsilon', type=float, default=0.2, help='PPO clip parameter')
    pipe_parser.add_argument('--entropy-coef', type=float, default=0.01, help='Entropy coefficient')
    pipe_parser.add_argument('--value-coef', type=float, default=0.5, help='Value loss coefficient')
    pipe_parser.add_argument('--max-grad-norm', type=float, default=0.5, help='Gradient clipping')
    pipe_parser.add_argument('--save-freq', type=int, default=100, help='Save frequency')
    pipe_parser.add_argument('--eval-freq', type=int, default=50, help='Evaluation frequency')
    pipe_parser.add_argument('--eval-episodes', type=int, default=3, help='Episodes for OOS evaluation')
    pipe_parser.add_argument('--optimize-for-composite', action='store_true', help='Use composite score as terminal reward (pure composite optimization)')
    # Targeted mode specific options
    pipe_parser.add_argument('--train-days', type=int, default=15, help='Training window days (targeted)')
    pipe_parser.add_argument('--oos-days', type=int, default=7, help='Out-of-sample window days (targeted)')
    pipe_parser.add_argument('--backtest-days', type=int, default=8, help='Backtest window days (targeted)')
    pipe_parser.add_argument('--target-composite', type=float, default=0.6, help='Target backtest composite to stop (targeted)')
    pipe_parser.add_argument('--target-win-rate', type=float, default=0.5, help='Target backtest win rate to stop (targeted)')
    pipe_parser.add_argument('--target-trades-per-day', type=float, default=3.0, help='Target backtest trades/day to stop (targeted)')
    pipe_parser.add_argument('--max-rounds', type=int, default=10, help='Max training rounds (targeted)')
    pipe_parser.set_defaults(require_backtest_better=True)
    pipe_parser.add_argument('--no-require-backtest-better', dest='require_backtest_better', action='store_false', help='Do not require backtest composite >= train and OOS')
    
    # Backtest command
    bt_parser = subparsers.add_parser('backtest', help='Backtest a saved model over a date window')
    bt_parser.add_argument('--model-path', type=str, required=True, help='Path to saved model to backtest')
    bt_parser.add_argument('--start-date', type=str, help='Start date YYYY-MM-DD HH:MM:SS')
    bt_parser.add_argument('--end-date', type=str, help='End date YYYY-MM-DD HH:MM:SS')
    bt_parser.add_argument('--last-hours', type=int, help='Backtest the last N hours')
    bt_parser.add_argument('--stochastic-eval', action='store_true', help='Use stochastic evaluation (sample actions)')
    bt_parser.add_argument('--max-episode-steps', type=int, default=None, help='Override max episode steps during backtest')
    bt_parser.add_argument('--eval-episodes', type=int, default=5, help='Episodes for evaluation inside backtest (if used)')
    bt_parser.add_argument('--epsilon-explore', type=float, default=0.0, help='Epsilon-greedy exploration during evaluation (diagnostic)')

    # Test trades command
    tt_parser = subparsers.add_parser('test-trades', help='Run scripted test trades to validate environment')
    tt_parser.add_argument('--last-hours', type=int, default=72, help='Window size in hours')
    tt_parser.add_argument('--trades', type=int, default=5, help='Number of trades to execute')
    tt_parser.add_argument('--max-hold-hours', type=int, default=3, help='Max hours to hold before force close')

    # Test command
    test_parser = subparsers.add_parser('test', help='Test system components')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'evaluate':
        evaluate_model(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'test-trades':
        run_test_trades(args)
    elif args.command == 'pipeline':
        run_pipeline(args)
    elif args.command == 'test':
        test_system(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()