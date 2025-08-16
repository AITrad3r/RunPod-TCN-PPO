import os
import json
from datetime import datetime
from typing import Optional

import numpy as np

from data_query import DataQuery
from ppo_algorithm import PPOConfig, PPOTrainer


def generate_report(model_path: str,
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    window: Optional[str] = None,
                    use_vwap_filter: bool = False,
                    vwap_longs_only: bool = False,
                    max_episode_steps: Optional[int] = None,
                    out_html: Optional[str] = None):
    """
    Generate a standalone HTML report with:
      - Summary metrics (win rate, composite, trades/day, PF, Sortino, Calmar, MDD, SQN)
      - Trade-by-trade table (time, type, entry, exit, pnl, return)
      - Equity and drawdown curves
    Args:
      model_path: checkpoint .pt to evaluate
      start_date/end_date: window to evaluate. If not provided and window=='oos' or 'backtest', attempt to read from the latest pipeline_* file.
      window: optional hint ('oos' or 'backtest') for title
    """
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    # Resolve output path
    os.makedirs('results', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_html = out_html or os.path.join('results', f'report_{ts}.html')

    # Build minimal config (values won't affect evaluation policy)
    # We will load sequence_length from checkpoint inside PPOTrainer.init load
    cfg = PPOConfig(
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        max_episode_steps=500,
    )

    dq = DataQuery()

    # If no dates provided, try to guess the last 7d
    if not start_date or not end_date:
        stats = dq.get_data_stats()
        latest = datetime.strptime(stats['latest_date'], '%Y-%m-%d %H:%M:%S')
        earliest = datetime.strptime(stats['earliest_date'], '%Y-%m-%d %H:%M:%S')
        end_dt = latest
        start_dt = max(earliest, end_dt - (end_dt - earliest) / 10)  # fallback: last 10% of data span
        start_date = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        end_date = end_dt.strftime('%Y-%m-%d %H:%M:%S')

    env_kwargs = {
        'use_vwap_filter': bool(use_vwap_filter),
        'vwap_longs_only': bool(vwap_longs_only),
    }

    trainer = PPOTrainer(
        config=cfg,
        data_query=dq,
        train_start_date=start_date,
        train_end_date=end_date,
        val_start_date=start_date,
        val_end_date=end_date,
        env_kwargs=env_kwargs,
    )

    # Override episode steps if requested
    if max_episode_steps is not None:
        trainer.agent.config.max_episode_steps = int(max_episode_steps)

    # Run deterministic single-episode evaluation
    eval_start_ts = datetime.now()
    res = trainer.backtest(
        model_path=model_path,
        test_start_date=start_date,
        test_end_date=end_date,
        stochastic_eval=False,
        eval_episodes=1,
        epsilon_explore=0.0,
    )
    eval_end_ts = datetime.now()

    # Collect series from results or reconstruct from env
    equity_file = res.get('equity_file')
    if equity_file and os.path.exists(equity_file):
        with open(equity_file, 'r') as f:
            eq_payload = json.load(f)
        dts = eq_payload.get('datetime')
        equity = eq_payload.get('equity')
        drawdown = eq_payload.get('drawdown')
    else:
        # Fallback: attempt to read from the validation env (last run)
        # Not guaranteed; leave empty if not available
        dts, equity, drawdown = None, None, None

    trades_file = res.get('trades_file')
    trades = []
    if trades_file and os.path.exists(trades_file):
        with open(trades_file, 'r') as f:
            trades = json.load(f)

    # Summary metrics (keys from evaluate/backtest)
    def g(k, dflt=0.0):
        return res.get(k, dflt)

    summary = {
        'Window': f"{start_date} → {end_date}",
        'Win Rate': f"{g('eval_win_rate', 0.0):.2%}",
        'Composite': f"{g('eval_composite_score', 0.0)*100:.2f}%",
        'Trades/Day': f"{g('eval_trade_frequency', g('trades_per_day', 0.0)):.3f}",
        'Profit Factor': f"{g('eval_profit_factor', 0.0):.3f}",
        'Sortino': f"{g('eval_sortino_ratio', 0.0):.3f}",
        'Calmar': f"{g('eval_calmar_ratio', 0.0):.3f}",
        'Max Drawdown': f"{g('eval_max_drawdown', 0.0):.2%}",
        'SQN': f"{g('eval_sqn', 0.0):.3f}",
        'Total Trades': f"{int(g('eval_total_trades', 0))}",
        'Final Balance': f"{g('eval_final_balance', g('final_balance', 0.0)):.2f}",
    }

    # Settings overview sources
    # DataQuery indicator settings
    dq_settings = {
        'Indicators': 'VWAP, RSI' + (', Bollinger Bands' if getattr(dq, 'use_bbands', False) else ''),
        'VWAP Window': getattr(dq, 'vwap_window', None),
        'RSI Period': getattr(dq, 'rsi_period', None),
        'BBands Enabled': getattr(dq, 'use_bbands', False),
        'BB Window': getattr(dq, 'bb_window', None) if getattr(dq, 'use_bbands', False) else '—',
        'BB K': getattr(dq, 'bb_k', None) if getattr(dq, 'use_bbands', False) else '—',
    }

    # Environment and algo settings (from trainer.train_env)
    env = trainer.train_env
    env_settings = {
        'Window Size (obs)': getattr(env, 'window_size', None),
        'Grid Spacing': getattr(env, 'grid_spacing', None),
        'Risk-Reward (RR)': getattr(env, 'rr', None),
        'Initial Balance': getattr(env, 'initial_balance', None),
        'Fixed $ Risk': getattr(env, 'fixed_dollar_rr', None),
        'Risk $/Trade': getattr(env, 'per_trade_risk_dollars', None),
        'Hold Penalty (flat)': getattr(env, 'flat_hold_penalty', None),
        'Hold Penalty (open)': getattr(env, 'open_hold_penalty', None),
        'Volatility Floor (min bb_width)': getattr(env, 'min_bb_width', None),
        'Low Vol Penalty': getattr(env, 'low_vol_penalty', None),
        'VWAP Filter Enabled': getattr(env, 'use_vwap_filter', None),
        'VWAP Longs Only': getattr(env, 'vwap_longs_only', None),
        'SL/TP Only Exit': getattr(env, 'require_sl_tp_close', None),
        'Optimize For Composite (train)': getattr(env, 'optimize_for_composite', None),
    }

    # PPO hyperparameters
    cfg_used = trainer.agent.config
    ppo_settings = {
        'Max Episode Steps': getattr(cfg_used, 'max_episode_steps', None),
        'Learning Rate': getattr(cfg_used, 'learning_rate', None),
        'Gamma': getattr(cfg_used, 'gamma', None),
        'GAE Lambda': getattr(cfg_used, 'gae_lambda', None),
        'Clip Epsilon': getattr(cfg_used, 'clip_epsilon', None),
        'Entropy Coef': getattr(cfg_used, 'entropy_coef', None),
        'Value Coef': getattr(cfg_used, 'value_coef', None),
        'Max Grad Norm': getattr(cfg_used, 'max_grad_norm', None),
        'n_steps': getattr(cfg_used, 'n_steps', None),
        'batch_size': getattr(cfg_used, 'batch_size', None),
        'n_epochs': getattr(cfg_used, 'n_epochs', None),
    }

    # Composite score overview (from TradingMetrics.calculate_composite_score Option B)
    composite_overview = {
        'Weights': 'Sortino 0.24, Calmar 0.20, ProfitFactor 0.18, WinRate 0.16, 1/(1+MDD) 0.07, Trades/Day 0.10, SQN 0.05',
        'Norm Caps': 'Sortino≤2, Calmar≤3, PF≤2, SQN≤3; WinRate∈[0,1] ; Drawdown uses 1/(1+|MDD|)',
        'Activity Penalty': 'If total_trades < 4, scale by total_trades/4',
        'Constraints': 'Min Trades/Day≥3 (linear hinge), WinRate≥50% (quadratic hinge)',
        'Composite Scale': 'Reported as % in Summary (0–100%)',
    }

    # Windows and durations
    try:
        sd = datetime.strptime(start_date, '%Y-%m-%d %H:%M:%S') if isinstance(start_date, str) else start_date
        ed = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S') if isinstance(end_date, str) else end_date
        dur_hours = (ed - sd).total_seconds() / 3600.0 if (sd and ed) else None
    except Exception:
        dur_hours = None
    durations = {
        'Evaluation Window': summary['Window'],
        'Window Duration (hours)': f"{dur_hours:.2f}" if dur_hours is not None else '—',
        'Eval Runtime (seconds)': f"{(eval_end_ts - eval_start_ts).total_seconds():.3f}",
        'Report Generated': ts,
        'Mode': window or 'evaluation',
    }

    # Build HTML
    title = f"PPO Report - {window.upper() if window else 'Evaluation'}"

    def html_table_from_dict(d: dict) -> str:
        rows = ''.join([f"<tr><th style='text-align:left;padding:6px 10px'>{k}</th><td style='padding:6px 10px'>{v}</td></tr>" for k, v in d.items()])
        return f"<table style='border-collapse:collapse;border:1px solid #ddd'>{rows}</table>"

    def html_trades_table(trs: list) -> str:
        if not trs:
            return "<p>No trades recorded.</p>"
        headers = ["entry_time", "exit_time", "position_type", "entry_price", "exit_price", "pnl", "return_pct"]
        thead = ''.join([f"<th style='text-align:left;padding:6px 10px'>{h}</th>" for h in headers])
        rows = []
        for t in trs:
            row = []
            for h in headers:
                v = t.get(h, '')
                if h == 'return_pct' and isinstance(v, (int, float)):
                    v = f"{v*100:.2f}%"
                elif isinstance(v, float):
                    # price/pnl formatting
                    v = f"{v:.4f}"
                row.append(f"<td style='padding:6px 10px;border-top:1px solid #f0f0f0'>{v}</td>")
            rows.append(f"<tr>{''.join(row)}</tr>")
        return f"<table style='border-collapse:collapse;border:1px solid #ddd'><thead><tr>{thead}</tr></thead><tbody>{''.join(rows)}</tbody></table>"

    # Plot data payloads
    dts = dts or []
    equity = equity or []
    drawdown = drawdown or []

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    h1 {{ margin-bottom: 0; }}
    .section {{ margin-top: 24px; }}
    .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
    @media (min-width: 980px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
    .card {{ padding: 16px; border: 1px solid #eaeaea; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
    table {{ width: 100%; font-size: 14px; }}
    th, td {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <p><strong>Model:</strong> {os.path.basename(model_path)} | <strong>Window:</strong> {summary['Window']}</p>

  <div class="grid">
    <div class="card">
      <h2>Summary Metrics</h2>
      {html_table_from_dict(summary)}
    </div>
    <div class="card">
      <h2>Equity & Drawdown</h2>
      <div id="equity_chart" style="height:360px"></div>
      <div id="dd_chart" style="height:240px;margin-top:12px"></div>
    </div>
  </div>

  <div class="section card">
    <h2>Settings Overview</h2>
    <div class="grid">
      <div class="card">
        <h3>Data & Indicators</h3>
        {html_table_from_dict(dq_settings)}
      </div>
      <div class="card">
        <h3>Environment & Algo</h3>
        {html_table_from_dict(env_settings)}
      </div>
    </div>
    <div class="grid" style="margin-top:12px">
      <div class="card">
        <h3>Composite Metric</h3>
        {html_table_from_dict(composite_overview)}
      </div>
      <div class="card">
        <h3>PPO Hyperparameters</h3>
        {html_table_from_dict(ppo_settings)}
      </div>
    </div>
    <div class="card" style="margin-top:12px">
      <h3>Durations</h3>
      {html_table_from_dict(durations)}
    </div>
  </div>

  <div class="section card">
    <h2>Trade-by-Trade</h2>
    {html_trades_table(trades)}
  </div>

  <script>
    const dts = {json.dumps(dts)};
    const equity = {json.dumps(equity)};
    const drawdown = {json.dumps(drawdown)};

    if (equity && equity.length > 0) {{
      const eqTrace = {{ x: dts.length === equity.length ? dts : Array.from({{length: equity.length}}, (_, i) => i), y: equity, mode: 'lines', name: 'Equity' }};
      Plotly.newPlot('equity_chart', [eqTrace], {{ margin: {{ t: 30 }}, xaxis: {{ title: 'Time' }}, yaxis: {{ title: 'Balance' }}, showlegend: false }});
    }} else {{
      document.getElementById('equity_chart').innerHTML = '<p>No equity data available.</p>';
    }}

    if (drawdown && drawdown.length > 0) {{
      const ddTrace = {{ x: dts.length === drawdown.length ? dts : Array.from({{length: drawdown.length}}, (_, i) => i), y: drawdown.map(v => v * 100), mode: 'lines', name: 'Drawdown (%)' }};
      Plotly.newPlot('dd_chart', [ddTrace], {{ margin: {{ t: 30 }}, xaxis: {{ title: 'Time' }}, yaxis: {{ title: 'Drawdown %' }}, showlegend: false }});
    }} else {{
      document.getElementById('dd_chart').innerHTML = '<p>No drawdown data available.</p>';
    }}
  </script>
</body>
</html>
"""

    with open(out_html, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Report written to {out_html}")
    return out_html


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description='Generate HTML report for PPO evaluation window')
    p.add_argument('--model-path', required=True)
    p.add_argument('--start-date')
    p.add_argument('--end-date')
    p.add_argument('--window', choices=['oos','backtest'])
    p.add_argument('--use-vwap-filter', action='store_true')
    p.add_argument('--vwap-longs-only', action='store_true')
    p.add_argument('--max-episode-steps', type=int)
    p.add_argument('--out')
    args = p.parse_args()

    generate_report(
        model_path=args.model_path,
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        use_vwap_filter=args.use_vwap_filter,
        vwap_longs_only=args.vwap_longs_only,
        max_episode_steps=args.max_episode_steps,
        out_html=args.out,
    )
