"""
CTDE PPO Training - Warehouse Multi-Agent Coordination
Honours Project: Deep Reinforcement Learning for Multi-Agent Coordination
in Mixed Human-Robot Warehouse Environments
Mohammed Al-Khudhair

Architecture: Centralised Training with Decentralised Execution (CTDE)
Algorithm:    Proximal Policy Optimisation (PPO)
Reference:    Schulman et al. (2017). Proximal Policy Optimisation Algorithms.

Each robot has:
    - Actor network:  local observation (26-dim) -> action logits (9-dim)
    - Critic network: global state (24-dim) -> value estimate (scalar)

The critic sees the full global state during training for stable value
estimation. At execution time only the actor is used with local observations.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from collections import defaultdict
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warehouse_env import WarehouseEnv

# ── Training hyperparameters ──────────────────────────────────────
TOTAL_TIMESTEPS   = 500_000
ROLLOUT_STEPS     = 512
N_EPOCHS          = 4
MINIBATCH_SIZE    = 128
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
CLIP_COEF         = 0.2
ENT_COEF          = 0.01
VF_COEF           = 0.5
MAX_GRAD_NORM     = 0.5
LR                = 3e-4
LR_MIN            = 1e-5
SAVE_INTERVAL     = 50_000
LOG_INTERVAL      = 2_048
MAX_STEPS_PER_EP  = 2_000

N_ROBOTS   = WarehouseEnv.N_ROBOTS
OBS_DIM    = WarehouseEnv.OBS_DIM
GLOBAL_DIM = WarehouseEnv.GLOBAL_DIM
ACT_DIM    = 9

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'training_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── Neural network definitions ────────────────────────────────────
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorNetwork(nn.Module):
    """
    Decentralised actor: takes local observation, outputs action distribution.
    Used during both training and execution.
    """
    def __init__(self, obs_dim: int = OBS_DIM, act_dim: int = ACT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def get_action_and_logprob(self, obs: torch.Tensor):
        logits = self.forward(obs)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()

    def get_logprob_and_entropy(self, obs: torch.Tensor,
                                 action: torch.Tensor):
        logits = self.forward(obs)
        dist   = Categorical(logits=logits)
        return dist.log_prob(action), dist.entropy()


class CriticNetwork(nn.Module):
    """
    Centralised critic: takes global state, outputs value estimate.
    Only used during training.
    """
    def __init__(self, global_dim: int = GLOBAL_DIM):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(global_dim, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 128)),
            nn.Tanh(),
            layer_init(nn.Linear(128, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.net(global_state).squeeze(-1)


class CTDEAgent:
    """
    CTDE agent wrapping N actor networks and one shared centralised critic.
    """
    def __init__(self, n_robots: int = N_ROBOTS,
                 obs_dim: int = OBS_DIM,
                 global_dim: int = GLOBAL_DIM,
                 act_dim: int = ACT_DIM,
                 lr: float = LR):
        self.n_robots = n_robots
        self.actors   = nn.ModuleList([
            ActorNetwork(obs_dim, act_dim) for _ in range(n_robots)
        ]).to(DEVICE)
        self.critic   = CriticNetwork(global_dim).to(DEVICE)
        self.optimizer = optim.Adam(
            list(self.actors.parameters()) +
            list(self.critic.parameters()),
            lr=lr, eps=1e-5,
        )

    def get_actions(self, obs_dict: dict):
        actions, logprobs, entropies = {}, {}, {}
        for i in range(self.n_robots):
            k   = f'robot_{i}'
            obs = torch.tensor(obs_dict[k], dtype=torch.float32,
                               device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                act, lp, ent = self.actors[i].get_action_and_logprob(obs)
            actions[k]  = int(act.item())
            logprobs[k] = float(lp.item())
            entropies[k]= float(ent.item())
        return actions, logprobs, entropies

    def get_value(self, global_state: np.ndarray) -> float:
        gs = torch.tensor(global_state, dtype=torch.float32,
                          device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            val = self.critic(gs)
        return float(val.item())

    def save(self, path: str):
        torch.save({
            'actors' : self.actors.state_dict(),
            'critic' : self.critic.state_dict(),
        }, path)
        print(f"  Model saved -> {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=DEVICE)
        self.actors.load_state_dict(ckpt['actors'])
        self.critic.load_state_dict(ckpt['critic'])
        print(f"  Model loaded <- {path}")


# ── Rollout buffer ────────────────────────────────────────────────
class RolloutBuffer:
    def __init__(self, rollout_steps: int, n_robots: int,
                 obs_dim: int, global_dim: int):
        self.T          = rollout_steps
        self.n          = n_robots
        self.obs        = np.zeros((rollout_steps, n_robots, obs_dim),
                                   dtype=np.float32)
        self.global_st  = np.zeros((rollout_steps, global_dim),
                                   dtype=np.float32)
        self.actions    = np.zeros((rollout_steps, n_robots), dtype=np.int64)
        self.logprobs   = np.zeros((rollout_steps, n_robots), dtype=np.float32)
        self.rewards    = np.zeros((rollout_steps, n_robots), dtype=np.float32)
        self.dones      = np.zeros(rollout_steps, dtype=np.float32)
        self.values     = np.zeros(rollout_steps, dtype=np.float32)
        self.advantages = np.zeros((rollout_steps, n_robots), dtype=np.float32)
        self.returns    = np.zeros((rollout_steps, n_robots), dtype=np.float32)
        self.ptr        = 0

    def add(self, obs_dict, global_state, actions, logprobs,
            rewards, done, value):
        for i in range(self.n):
            self.obs[self.ptr, i]      = obs_dict[f'robot_{i}']
            self.actions[self.ptr, i]  = actions[f'robot_{i}']
            self.logprobs[self.ptr, i] = logprobs[f'robot_{i}']
            self.rewards[self.ptr, i]  = rewards[f'robot_{i}']
        self.global_st[self.ptr] = global_state
        self.dones[self.ptr]     = float(done)
        self.values[self.ptr]    = value
        self.ptr += 1

    def compute_gae(self, last_value: float, last_done: bool):
        last_gae = np.zeros(self.n, dtype=np.float32)
        for t in reversed(range(self.T)):
            if t == self.T - 1:
                next_non_term = 1.0 - float(last_done)
                next_val      = last_value
            else:
                next_non_term = 1.0 - self.dones[t+1]
                next_val      = self.values[t+1]
            delta           = (self.rewards[t]
                               + GAMMA * next_val * next_non_term
                               - self.values[t])
            last_gae        = (delta
                               + GAMMA * GAE_LAMBDA * next_non_term * last_gae)
            self.advantages[t] = last_gae
            self.returns[t]    = last_gae + self.values[t]

    def get_batches(self, minibatch_size: int):
        indices = np.random.permutation(self.T)
        for start in range(0, self.T, minibatch_size):
            idx = indices[start:start+minibatch_size]
            yield (
                torch.tensor(self.obs[idx],       device=DEVICE),
                torch.tensor(self.global_st[idx], device=DEVICE),
                torch.tensor(self.actions[idx],   device=DEVICE),
                torch.tensor(self.logprobs[idx],  device=DEVICE),
                torch.tensor(self.advantages[idx],device=DEVICE),
                torch.tensor(self.returns[idx],   device=DEVICE),
            )

    def reset(self):
        self.ptr = 0


# ── PPO update ────────────────────────────────────────────────────
def ppo_update(agent: CTDEAgent, buffer: RolloutBuffer,
               update_idx: int, total_updates: int):
    frac = 1.0 - update_idx / total_updates
    lr   = max(LR_MIN, LR * frac)
    for pg in agent.optimizer.param_groups:
        pg['lr'] = lr

    pg_losses, vf_losses, ent_losses, clip_fracs = [], [], [], []

    for _ in range(N_EPOCHS):
        for (obs_b, gs_b, act_b, old_lp_b, adv_b, ret_b) in \
                buffer.get_batches(MINIBATCH_SIZE):

            # Normalise advantages per minibatch
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            # Centralised value estimate
            new_values = agent.critic(gs_b)
            vf_loss    = 0.5 * ((new_values.unsqueeze(-1).expand_as(ret_b)
                                 - ret_b) ** 2).mean()

            # Decentralised policy losses
            pg_loss_total  = torch.tensor(0.0, device=DEVICE)
            ent_loss_total = torch.tensor(0.0, device=DEVICE)

            for i in range(N_ROBOTS):
                new_lp, ent = agent.actors[i].get_logprob_and_entropy(
                    obs_b[:, i, :], act_b[:, i])
                log_ratio  = new_lp - old_lp_b[:, i]
                ratio      = log_ratio.exp()

                with torch.no_grad():
                    clip_fracs.append(
                        ((ratio - 1.0).abs() > CLIP_COEF).float().mean().item())

                adv_i        = adv_b[:, i]
                pg_loss1     = -adv_i * ratio
                pg_loss2     = -adv_i * torch.clamp(
                                    ratio, 1-CLIP_COEF, 1+CLIP_COEF)
                pg_loss_total  += torch.max(pg_loss1, pg_loss2).mean()
                ent_loss_total += ent.mean()

            pg_loss_total  /= N_ROBOTS
            ent_loss_total /= N_ROBOTS

            loss = (pg_loss_total
                    + VF_COEF * vf_loss
                    - ENT_COEF * ent_loss_total)

            agent.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(agent.actors.parameters()) +
                list(agent.critic.parameters()),
                MAX_GRAD_NORM)
            agent.optimizer.step()

            pg_losses.append(pg_loss_total.item())
            vf_losses.append(vf_loss.item())
            ent_losses.append(ent_loss_total.item())

    return {
        'pg_loss'   : np.mean(pg_losses),
        'vf_loss'   : np.mean(vf_losses),
        'ent_loss'  : np.mean(ent_losses),
        'clip_frac' : np.mean(clip_fracs),
        'lr'        : lr,
    }


# ── Training loop ─────────────────────────────────────────────────
def train():
    print("\nCTDE PPO Training - Warehouse Coordination")
    print("=" * 50)
    print(f"Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"Rollout steps   : {ROLLOUT_STEPS}")
    print(f"N epochs        : {N_EPOCHS}")
    print(f"Minibatch size  : {MINIBATCH_SIZE}")
    print(f"Output dir      : {OUTPUT_DIR}")
    print("=" * 50)

    env    = WarehouseEnv(max_steps=MAX_STEPS_PER_EP)
    agent  = CTDEAgent()
    buffer = RolloutBuffer(ROLLOUT_STEPS, N_ROBOTS, OBS_DIM, GLOBAL_DIM)

    obs, info     = env.reset()
    global_state  = info['global_state']
    done          = False

    total_steps    = 0
    update_count   = 0
    total_updates  = TOTAL_TIMESTEPS // ROLLOUT_STEPS
    episode_count  = 0
    ep_rewards     = defaultdict(float)
    ep_deliveries  = 0
    ep_steps       = 0

    metrics_log = []
    start_time  = time.time()

    print(f"\nStarting training...")

    while total_steps < TOTAL_TIMESTEPS:
        buffer.reset()

        # ── Collect rollout ───────────────────────────────────────
        for t in range(ROLLOUT_STEPS):
            actions, logprobs, _ = agent.get_actions(obs)
            value                = agent.get_value(global_state)

            next_obs, rewards, terminated, truncated, info = env.step(actions)
            next_global = info['global_state']
            done        = terminated or truncated

            buffer.add(obs, global_state, actions, logprobs,
                       rewards, done, value)

            for k in rewards:
                ep_rewards[k] += rewards[k]
            ep_deliveries  = info['deliveries']
            ep_steps       += 1
            total_steps    += 1

            obs          = next_obs
            global_state = next_global

            if done:
                episode_count += 1
                mean_reward = np.mean([ep_rewards[f'robot_{i}']
                                       for i in range(N_ROBOTS)])
                metrics_log.append({
                    'episode'    : episode_count,
                    'steps'      : total_steps,
                    'mean_reward': round(mean_reward, 3),
                    'deliveries' : ep_deliveries,
                    'collisions' : info['collisions'],
                    'safety_evts': info['safety_events'],
                    'wall_hits'  : info['wall_hits'],
                })
                obs, info    = env.reset()
                global_state = info['global_state']
                ep_rewards   = defaultdict(float)
                ep_deliveries= 0
                ep_steps     = 0
                done         = False

        # ── Compute GAE ───────────────────────────────────────────
        last_val  = agent.get_value(global_state)
        buffer.compute_gae(last_val, done)

        # ── PPO update ────────────────────────────────────────────
        update_count += 1
        stats = ppo_update(agent, buffer, update_count, total_updates)

        # ── Logging ───────────────────────────────────────────────
        if total_steps % LOG_INTERVAL < ROLLOUT_STEPS:
            elapsed  = time.time() - start_time
            sps      = total_steps / elapsed
            pct      = total_steps / TOTAL_TIMESTEPS * 100
            recent   = metrics_log[-10:] if len(metrics_log) >= 10 \
                       else metrics_log
            mean_r   = np.mean([m['mean_reward'] for m in recent]) \
                       if recent else 0.0
            mean_del = np.mean([m['deliveries']  for m in recent]) \
                       if recent else 0.0
            print(f"  [{pct:5.1f}%] steps={total_steps:,}  "
                  f"eps={episode_count}  "
                  f"reward={mean_r:7.2f}  "
                  f"del={mean_del:.1f}  "
                  f"pg={stats['pg_loss']:.4f}  "
                  f"vf={stats['vf_loss']:.4f}  "
                  f"lr={stats['lr']:.6f}  "
                  f"sps={sps:.0f}")

        # ── Checkpoint ────────────────────────────────────────────
        if total_steps % SAVE_INTERVAL < ROLLOUT_STEPS:
            ckpt_path = os.path.join(OUTPUT_DIR,
                                     f'checkpoint_{total_steps}.pt')
            agent.save(ckpt_path)

    # ── Final save ────────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, 'model_final.pt')
    agent.save(final_path)

    log_path = os.path.join(OUTPUT_DIR, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)
    print(f"\nTraining log saved -> {log_path}")

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Total episodes  : {episode_count}")
    print(f"Final model     : {final_path}")

    env.close()
    return agent, metrics_log


# ── Plot training curves ──────────────────────────────────────────
def plot_training(metrics_log: list, save_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    steps     = [m['steps']      for m in metrics_log]
    rewards   = [m['mean_reward']for m in metrics_log]
    deliveries= [m['deliveries'] for m in metrics_log]
    collisions= [m['collisions'] for m in metrics_log]

    def smooth(vals, w=20):
        if len(vals) < w:
            return vals
        return [np.mean(vals[max(0,i-w):i+1]) for i in range(len(vals))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor('#060E1A')

    titles  = ['Mean Reward per Episode',
               'Deliveries per Episode',
               'Collisions per Episode']
    data    = [rewards, deliveries, collisions]
    colors  = ['#3498DB', '#2ECC71', '#E74C3C']

    for ax, title, d, col in zip(axes, titles, data, colors):
        ax.set_facecolor('#0D1428')
        ax.plot(steps, d, color=col, alpha=0.25, lw=0.8)
        ax.plot(steps, smooth(d), color=col, lw=2.0, label='Smoothed')
        ax.set_title(title, color='#AAC4E0', fontsize=11)
        ax.set_xlabel('Timesteps', color='#8090A8')
        ax.tick_params(colors='#8090A8')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1A3050')
        ax.grid(alpha=0.12, color='#2A4060')

    fig.suptitle(
        'CTDE PPO Training - Warehouse Multi-Agent Coordination\n'
        'Mohammed Al-Khudhair  |  Honours Project  |  '
        'Schulman et al. (2017)',
        color='#AAC4E0', fontsize=11, y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Training curves saved -> {save_path}")


# ── Entry point ───────────────────────────────────────────────────
if __name__ == '__main__':
    agent, log = train()

    if log:
        plot_path = os.path.join(OUTPUT_DIR, 'training_curves.png')
        plot_training(log, plot_path)

    print("\nNext step: run warehouse_demo_trained.py to visualise the policy.")
