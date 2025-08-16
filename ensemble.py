import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
from ppo_algorithm import PPOAgent
from trading_environment import GridTradingEnvironment

class EnsemblePPOAgent:
    """Ensemble of PPO agents with aggregation for action selection and evaluation."""
    def __init__(self, agents: List[PPOAgent], agg: str = 'avg_logits', weights: List[float] = None):
        assert len(agents) > 0, "Ensemble must contain at least one agent"
        assert agg in { 'avg_logits', 'avg_probs', 'majority' }
        self.agents = agents
        self.agg = agg
        # Weights per member (normalized); default equal
        if weights is None:
            self.weights = [1.0 / len(agents)] * len(agents)
        else:
            assert len(weights) == len(agents), "weights length must match number of agents"
            s = float(sum(max(0.0, w) for w in weights)) or 1.0
            self.weights = [max(0.0, w) / s for w in weights]
        # Use first agent's config/device as reference
        self.config = agents[0].config
        self.device = agents[0].device

    def set_weights(self, weights: List[float]):
        assert len(weights) == len(self.agents), "weights length must match number of agents"
        s = float(sum(max(0.0, w) for w in weights)) or 1.0
        self.weights = [max(0.0, w) / s for w in weights]

    def _select_action_ensemble(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Aggregate member policies to select a single action."""
        # Convert obs once
        obs_tensor = torch.from_numpy(observation).unsqueeze(0).to(self.device)
        logits_list = []
        probs_list = []
        argmax_list = []

        with torch.no_grad():
            for a in self.agents:
                # Forward pass to get logits and value
                action_logits, _ = a.network(obs_tensor)
                logits_list.append(action_logits)
                probs = F.softmax(action_logits, dim=-1)
                probs_list.append(probs)
                argmax_list.append(int(torch.argmax(probs, dim=-1).cpu().numpy()[0]))

        if self.agg == 'avg_logits':
            # Weighted average of logits
            stacked = torch.stack(logits_list, dim=0)  # (N, B, A)
            w = torch.tensor(self.weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1, 1)
            avg_logits = torch.sum(stacked * w, dim=0)
            if deterministic:
                action = torch.argmax(avg_logits, dim=-1)
            else:
                dist = torch.distributions.Categorical(F.softmax(avg_logits, dim=-1))
                action = dist.sample()
            return int(action.cpu().numpy()[0])
        elif self.agg == 'avg_probs':
            # Weighted average of probabilities
            stacked = torch.stack(probs_list, dim=0)
            w = torch.tensor(self.weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1, 1)
            avg_probs = torch.sum(stacked * w, dim=0)
            if deterministic:
                action = torch.argmax(avg_probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(avg_probs)
                action = dist.sample()
            return int(action.cpu().numpy()[0])
        else:  # majority vote
            # Weighted majority vote; on tie fallback to weighted avg_probs
            counts = np.zeros(self.config.action_dim, dtype=float)
            for idx, a_id in enumerate(argmax_list):
                counts[a_id] += float(self.weights[idx])
            if (counts == counts.max()).sum() == 1:
                return int(np.argmax(counts))
            stacked = torch.stack(probs_list, dim=0)
            w = torch.tensor(self.weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1, 1)
            avg_probs = torch.sum(stacked * w, dim=0)
            action = torch.argmax(avg_probs, dim=-1)
            return int(action.cpu().numpy()[0])

    def evaluate(self, env: GridTradingEnvironment, n_episodes: int = 5, deterministic: bool = True) -> Dict[str, float]:
        """Evaluate the ensemble on an environment. Aggregates actions per step."""
        eval_rewards = []
        eval_composite_scores = []
        eval_metrics = []
        action_counts = {0: 0, 1: 0, 2: 0}

        for _ in range(n_episodes):
            obs = env.reset()
            episode_reward = 0.0

            for step in range(self.config.max_episode_steps):
                a = self._select_action_ensemble(obs, deterministic=deterministic)
                if a in action_counts:
                    action_counts[a] += 1
                obs, reward, done, info = env.step(a)
                episode_reward += reward
                if done:
                    break

            # Episode metrics
            m = env.get_episode_metrics()
            eval_rewards.append(episode_reward)
            eval_composite_scores.append(m.get('composite_score', 0.0))
            eval_metrics.append(m)

        # Aggregate robustly
        avg_metrics: Dict[str, float] = {}
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
                    avg_metrics[f'eval_{key}'] = 0.0
        if eval_rewards:
            avg_metrics['eval_reward'] = float(np.mean(eval_rewards))
        if eval_composite_scores:
            avg_metrics['eval_composite_score'] = float(np.mean(eval_composite_scores))

        # Diagnostics
        total = sum(action_counts.values())
        if total > 0:
            hold_p = action_counts[2] / total
            print(f"\n[Ensemble Diag] Action counts: BUY={action_counts[0]}, SELL={action_counts[1]}, HOLD={action_counts[2]} (HOLD%={hold_p:.2%})")
        if eval_metrics:
            avg_tr = np.mean([m.get('total_trades', 0) for m in eval_metrics])
            any_pos = any(m.get('total_trades', 0) > 0 for m in eval_metrics)
            print(f"[Ensemble Diag] Avg trades/episode: {avg_tr:.2f} | Any position opened: {any_pos}")

        return avg_metrics

    @staticmethod
    def from_model_paths(model_paths: List[str], agg: str = 'avg_logits', weights: List[float] = None) -> 'EnsemblePPOAgent':
        agents: List[PPOAgent] = []
        ref_device = None
        ref_config = None
        for mp in model_paths:
            # Load each agent with its saved config
            a = PPOAgent(config=None)
            episode, metrics = a.load_model(mp)
            agents.append(a)
            if ref_device is None:
                ref_device = a.device
                ref_config = a.config
        return EnsemblePPOAgent(agents, agg=agg, weights=weights)
