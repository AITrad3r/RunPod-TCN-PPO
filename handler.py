import os
import json
import argparse
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List

import runpod
import torch

# Import training entrypoints
import train_ppo_grid_trading as pipeline


def _bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def build_args(payload: Dict[str, Any]) -> argparse.Namespace:
    """
    Build argparse-like args from Serverless JSON payload.
    Supports 'pipeline' (short/targeted) and 'backtest' commands.
    """
    command = payload.get("command", "pipeline")

    common = {
        'learning_rate': float(payload.get('learning_rate', 3e-4)),
        'batch_size': int(payload.get('batch_size', 64)),
        'n_steps': int(payload.get('n_steps', 1024)),
        'n_epochs': int(payload.get('n_epochs', 10)),
        'hidden_dim': int(payload.get('hidden_dim', 256)),
        'max_episode_steps': int(payload.get('max_episode_steps', 500)),
        'gamma': float(payload.get('gamma', 0.99)),
        'gae_lambda': float(payload.get('gae_lambda', 0.95)),
        'clip_epsilon': float(payload.get('clip_epsilon', 0.2)),
        'entropy_coef': float(payload.get('entropy_coef', 0.01)),
        'value_coef': float(payload.get('value_coef', 0.5)),
        'max_grad_norm': float(payload.get('max_grad_norm', 0.5)),
        'save_freq': int(payload.get('save_freq', 100)),
        'eval_freq': int(payload.get('eval_freq', 50)),
        'optimize_for_composite': _bool(payload.get('optimize_for_composite', False)),
        'use_vwap_filter': _bool(payload.get('use_vwap_filter', False)),
        'vwap_longs_only': _bool(payload.get('vwap_longs_only', False)),
        # Indicators
        'vwap_window': payload.get('vwap_window', None),
        'rsi_period': payload.get('rsi_period', None),
        'use_bbands': _bool(payload.get('use_bbands', False)),
        'bb_window': payload.get('bb_window', None),
        'bb_k': payload.get('bb_k', None),
        # Ensemble
        'ensemble_size': int(payload.get('ensemble_size', 1)),
        'ensemble_agg': payload.get('ensemble_agg', 'avg_logits'),
        'ensemble_weights': payload.get('ensemble_weights', None),
        'tune_ensemble_weights': _bool(payload.get('tune_ensemble_weights', False)),
        'tune_ensemble_trials': int(payload.get('tune_ensemble_trials', 50)),
        'tune_agg': _bool(payload.get('tune_agg', True)),
        # Tuning
        'tune_hparams': _bool(payload.get('tune_hparams', False)),
        'tune_trials': int(payload.get('tune_trials', 20)),
        'tune_trial_episodes': int(payload.get('tune_trial_episodes', 100)),
        'seed_base': int(payload.get('seed_base', 42)),
    }

    if command == 'pipeline':
        mode = payload.get('mode', 'short')
        args = {
            **common,
            'command': 'pipeline',
            'mode': mode,
            'episodes': int(payload.get('episodes', 500)),
            'eval_episodes': int(payload.get('eval_episodes', 3)),
            'require_cuda': _bool(payload.get('require_cuda', False)),
        }
        if mode == 'targeted':
            args.update({
                'train_days': int(payload.get('train_days', 15)),
                'oos_days': int(payload.get('oos_days', 7)),
                'backtest_days': int(payload.get('backtest_days', 8)),
                'target_composite': float(payload.get('target_composite', 0.6)),
                'target_win_rate': float(payload.get('target_win_rate', 0.5)),
                'target_trades_per_day': float(payload.get('target_trades_per_day', 3.0)),
                'max_rounds': int(payload.get('max_rounds', 10)),
                'round_episodes': int(payload.get('round_episodes', payload.get('episodes', 100))),
                'require_backtest_better': _bool(payload.get('require_backtest_better', True)),
            })
        return argparse.Namespace(**args)

    if command == 'backtest':
        args = {
            'command': 'backtest',
            'model_path': payload['model_path'],
            'start_date': payload.get('start_date'),
            'end_date': payload.get('end_date'),
            'last_hours': payload.get('last_hours'),
            'stochastic_eval': _bool(payload.get('stochastic_eval', False)),
            'max_episode_steps': payload.get('max_episode_steps'),
            'eval_episodes': int(payload.get('eval_episodes', 5)),
            'epsilon_explore': float(payload.get('epsilon_explore', 0.0)),
            'use_vwap_filter': _bool(payload.get('use_vwap_filter', False)),
            'vwap_longs_only': _bool(payload.get('vwap_longs_only', False)),
            'require_cuda': _bool(payload.get('require_cuda', False)),
        }
        return argparse.Namespace(**args)

    # Fallback: train
    return argparse.Namespace(**{
        **common,
        'command': 'train',
        'episodes': int(payload.get('episodes', 1000)),
        'backtest': _bool(payload.get('backtest', True)),
        'require_cuda': _bool(payload.get('require_cuda', False)),
    })


def torch_summary() -> Dict[str, Any]:
    try:
        has = torch.cuda.is_available()
        cnt = torch.cuda.device_count()
        name = torch.cuda.get_device_name(0) if has and cnt > 0 else 'N/A'
        return {
            'torch_version': torch.__version__,
            'cuda_available': bool(has),
            'device_count': int(cnt),
            'gpu_name': name,
        }
    except Exception as e:
        return {'error': str(e)}


def persist_artifacts(out_dirs: List[str]) -> Dict[str, Any]:
    """Copy artifacts from out_dirs into ARTIFACTS_DIR if set. Returns dict of saved files."""
    dest_root = os.environ.get('ARTIFACTS_DIR')  # mount a volume to this path in RunPod
    saved: Dict[str, List[str]] = {}
    if not dest_root:
        return {'saved': saved, 'note': 'ARTIFACTS_DIR not set; skipping persistence'}
    job_id = os.environ.get('RUNPOD_JOB_ID', uuid.uuid4().hex[:8])
    ts = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    run_folder = os.path.join(dest_root, f"job_{job_id}_{ts}")
    try:
        os.makedirs(run_folder, exist_ok=True)
        for d in out_dirs:
            if not os.path.isdir(d):
                continue
            target = os.path.join(run_folder, os.path.basename(d))
            os.makedirs(target, exist_ok=True)
            for root, _, files in os.walk(d):
                for fn in files:
                    src = os.path.join(root, fn)
                    rel = os.path.relpath(src, d)
                    dst = os.path.join(target, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    try:
                        shutil.copy2(src, dst)
                        saved.setdefault(d, []).append(dst)
                    except Exception:
                        pass
        return {'saved': saved, 'artifact_dir': run_folder}
    except Exception as e:
        return {'saved': saved, 'error': str(e)}


def handler(event):
    """RunPod Serverless entrypoint.
    event['input'] holds JSON parameters.
    """
    try:
        payload = event.get('input', {}) or {}
        args = build_args(payload)

        # Health ping
        if payload.get('command') == 'ping':
            summary = torch_summary()
            return {'status': 'ok', 'pong': True, 'torch': summary}

        os.makedirs('results', exist_ok=True)
        os.makedirs('checkpoints', exist_ok=True)

        # Fail fast if CUDA is required
        if getattr(args, 'require_cuda', False):
            ts = torch_summary()
            if not ts.get('cuda_available', False):
                return {'status': 'error', 'message': 'CUDA not available but require_cuda=True', 'torch': ts}

        if getattr(args, 'command', 'pipeline') == 'pipeline':
            pipeline.run_pipeline(args)
            # Return most recent pipeline_* file path
            files = sorted([f for f in os.listdir('results') if f.startswith('pipeline_')], reverse=True)
            out = files[0] if files else None
            persist = persist_artifacts(['results', 'checkpoints'])
            return { 'status': 'ok', 'report_file': f"results/{out}" if out else None, 'artifacts': persist }
        elif args.command == 'backtest':
            pipeline.run_backtest(args)
            persist = persist_artifacts(['results', 'checkpoints'])
            return { 'status': 'ok', 'backtest_file': 'backtest_results.json', 'artifacts': persist }
        elif args.command == 'train':
            pipeline.train_model(args)
            persist = persist_artifacts(['results', 'checkpoints'])
            return { 'status': 'ok', 'training_log': 'training_log.json', 'artifacts': persist }
        else:
            return { 'status': 'error', 'message': f"Unknown command: {args.command}" }
    except Exception as e:
        return { 'status': 'error', 'message': str(e) }


if __name__ == "__main__":
    # If LOCAL_TEST=1, run a quick local invocation; otherwise start serverless.
    if os.environ.get("LOCAL_TEST", "0") == "1":
        test_event = {
            'input': {
                'command': 'pipeline',
                'mode': 'short',
                'episodes': 10,
                'eval_episodes': 1,
                'use_vwap_filter': False,
                'vwap_longs_only': False
            }
        }
        print(handler(test_event))
    else:
        runpod.serverless.start({
            "handler": handler
        })
