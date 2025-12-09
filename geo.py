import time
import math
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


# ===============================
#  Config
# ===============================

ENV_ID = "Pendulum-v1"

MAX_ENV_STEPS = 100_000
RANDOM_STEPS = 5_000

TRAIN_EVERY = 1_000        # env steps between training phases
TRAIN_ITERS = 200          # gradient steps per training phase
BATCH_SIZE = 32
SEQ_LEN = 50

LATENT_DIM = 32
HIDDEN_DIM = 128

GAMMA = 0.99

# CEM
CEM_HORIZON = 30
CEM_SAMPLES = 256
CEM_ITERS = 4
CEM_ELITE_FRAC = 0.1
CEM_INIT_STD = 1.0

# Loss weights
ALPHA_GEO = 1.0
ALPHA_BISIM = 0.1
REWARD_LOSS_WEIGHT = 1.0
LATENT_NORM_WEIGHT = 1e-3
BISIM_CLIP_VALUE = 10.0
LATENT_STD_TARGET = 0.5
LATENT_STD_WEIGHT = 1e-2

# Runtime budget (seconds)
MAX_RUNTIME = 60 * 30  # ~30 minutes


# ===============================
#  Utilities
# ===============================

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ===============================
#  Replay Buffer
# ===============================

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)

        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done: bool):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rews[self.idx] = reward
        self.dones[self.idx] = float(done)
        self.next_obs[self.idx] = next_obs

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @dataclass
    class Batch:
        obs: np.ndarray        # [B, T, obs_dim]
        actions: np.ndarray    # [B, T, act_dim]
        rews: np.ndarray       # [B, T]
        dones: np.ndarray      # [B, T]

    def sample_sequences(self, batch_size: int, seq_len: int):
        assert self.size > seq_len + 1, "Not enough data to sample sequences."
        obs_seq = []
        act_seq = []
        rew_seq = []
        done_seq = []

        max_start = self.size - seq_len - 1
        for _ in range(batch_size):
            start = np.random.randint(0, max_start)
            end = start + seq_len

            obs_seq.append(self.obs[start:end])
            act_seq.append(self.actions[start:end])
            rew_seq.append(self.rews[start:end])
            done_seq.append(self.dones[start:end])

        obs_seq = np.stack(obs_seq, axis=0)          # [B, T, obs_dim]
        act_seq = np.stack(act_seq, axis=0)          # [B, T, act_dim]
        rew_seq = np.stack(rew_seq, axis=0)          # [B, T]
        done_seq = np.stack(done_seq, axis=0)        # [B, T]

        return ReplayBuffer.Batch(obs_seq, act_seq, rew_seq, done_seq)


# ===============================
#  Models
# ===============================

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=HIDDEN_DIM, num_layers=2, act=nn.ReLU):
        super().__init__()
        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim=LATENT_DIM):
        super().__init__()
        self.net = MLP(obs_dim, latent_dim)

    def forward(self, obs):
        # obs: [B, T, obs_dim] or [B, obs_dim]
        if obs.dim() == 3:
            B, T, D = obs.shape
            x = obs.reshape(B * T, D)
            z = self.net(x)
            return torch.tanh(z).reshape(B, T, -1)
        else:  # [B, D]
            return torch.tanh(self.net(obs))


class DynamicsModel(nn.Module):
    """
    Deterministic latent dynamics: z_{t+1} = f([z_t, a_t])
    """
    def __init__(self, latent_dim=LATENT_DIM, act_dim=1):
        super().__init__()
        self.net = MLP(latent_dim + act_dim, latent_dim)

    def forward(self, z, a):
        # z: [*, latent_dim], a: [*, act_dim]
        x = torch.cat([z, a], dim=-1)
        return self.net(x)


class RewardModel(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, act_dim=1):
        super().__init__()
        self.net = MLP(latent_dim + act_dim, 1)

    def forward(self, z, a):
        x = torch.cat([z, a], dim=-1)
        return self.net(x)  # [*, 1]


# ===============================
#  Rollout utilities
# ===============================

def rollout_dynamics(dynamics: DynamicsModel, z0: torch.Tensor, act_seq: torch.Tensor):
    """
    Roll out dynamics given initial latent z0 and action sequence.

    z0: [B, latent_dim]
    act_seq: [B, T, act_dim]
    return: [B, T, latent_dim]
    """
    B, T, A = act_seq.shape
    z = z0
    zs = []
    for t in range(T):
        a_t = act_seq[:, t, :]
        z = dynamics(z, a_t)
        zs.append(z)
    return torch.stack(zs, dim=1)


def compute_discounted_returns(rews: torch.Tensor, dones: torch.Tensor, gamma: float = GAMMA):
    """
    rews: [B, T]
    dones: [B, T]
    returns: [B, T]
    """
    B, T = rews.shape
    returns = torch.zeros_like(rews)
    running = torch.zeros(B, device=rews.device)
    for t in reversed(range(T)):
        running = rews[:, t] + gamma * running * (1.0 - dones[:, t])
        returns[:, t] = running
    return returns


# ===============================
#  Geometric + bisimulation losses
# ===============================

def world_model_loss(batch: ReplayBuffer.Batch,
                     encoder: Encoder,
                     dynamics: DynamicsModel,
                     reward_model: RewardModel,
                     device,
                     gamma: float = GAMMA):
    """
    batch: obs, actions, rews, dones as numpy arrays
    """

    obs = torch.tensor(batch.obs, dtype=torch.float32, device=device)              # [B, T, obs_dim]
    actions = torch.tensor(batch.actions, dtype=torch.float32, device=device)      # [B, T, act_dim]
    rews = torch.tensor(batch.rews, dtype=torch.float32, device=device)            # [B, T]
    dones = torch.tensor(batch.dones, dtype=torch.float32, device=device)          # [B, T]

    B, T, _ = obs.shape

    # Encode observations to latent
    z = encoder(obs)                      # [B, T, latent_dim]
    latent_reg = (z.pow(2).mean()) if LATENT_NORM_WEIGHT > 0 else torch.tensor(0.0, device=device)
    # Encourage non-collapsed latents
    latent_std = z.std(dim=(0, 1))
    latent_std_loss = (latent_std - LATENT_STD_TARGET).abs().mean()
    z_t = z[:, :-1, :]                   # [B, T-1, latent_dim]
    z_tp1_target = z[:, 1:, :]           # [B, T-1, latent_dim]
    a_t = actions[:, :-1, :]             # [B, T-1, act_dim]
    r_t = rews[:, :-1]                   # [B, T-1]
    d_t = dones[:, :-1]                  # [B, T-1]

    # One-step prediction
    z_tp1_pred = dynamics(z_t.reshape(-1, LATENT_DIM),
                          a_t.reshape(-1, a_t.size(-1)))
    z_tp1_pred = z_tp1_pred.reshape(B, T - 1, LATENT_DIM)

    r_pred = reward_model(z_t.reshape(-1, LATENT_DIM),
                          a_t.reshape(-1, a_t.size(-1)))
    r_pred = r_pred.view(B, T - 1)

    # One-step losses
    mse = nn.MSELoss()
    L_dyn = mse(z_tp1_pred, z_tp1_target.detach())
    L_r = mse(r_pred, r_t)

    # Geodesic rollout alignment (multi-step)
    z0 = z[:, 0, :]                        # [B, latent_dim]
    a_seq = actions[:, :-1, :]            # [B, T-1, act_dim]
    z_roll = rollout_dynamics(dynamics, z0, a_seq)  # [B, T-1, latent_dim]
    z_true_seq = z[:, 1:, :]

    # For now, Euclidean distance as a proxy for latent geodesic
    geo_dist = torch.sqrt(torch.clamp((z_roll - z_true_seq.detach()).pow(2).sum(-1), min=1e-8))  # [B, T-1]
    geo_dist = torch.clamp(geo_dist, max=BISIM_CLIP_VALUE)
    L_geo = (geo_dist ** 2).mean()

    # Bisimulation-inspired metric shaping
    # Compute discounted returns as approximate values
    returns = compute_discounted_returns(rews, dones, gamma=gamma)  # [B, T]
    z_flat = z.reshape(-1, LATENT_DIM)
    v_flat = returns.reshape(-1)

    num_pairs = min(1024, z_flat.size(0))
    idx1 = torch.randint(0, z_flat.size(0), (num_pairs,), device=device)
    idx2 = torch.randint(0, z_flat.size(0), (num_pairs,), device=device)

    z1, z2 = z_flat[idx1], z_flat[idx2]
    v1, v2 = v_flat[idx1], v_flat[idx2]
    value_diff = (v1 - v2).abs()

    value_diff = torch.clamp(value_diff, max=BISIM_CLIP_VALUE)
    dist = torch.sqrt(torch.clamp((z1 - z2).pow(2).sum(-1), min=1e-8))
    dist = torch.clamp(dist, max=BISIM_CLIP_VALUE)
    # Regress distance toward value difference (could scale/clip if needed)
    L_bisim = ((dist - value_diff.detach()) ** 2).mean()

    loss = L_dyn + REWARD_LOSS_WEIGHT * L_r + ALPHA_GEO * L_geo + ALPHA_BISIM * L_bisim
    loss = loss + LATENT_NORM_WEIGHT * latent_reg + LATENT_STD_WEIGHT * latent_std_loss

    logs = {
        "L_dyn": L_dyn.item(),
        "L_r": L_r.item(),
        "L_geo": L_geo.item(),
        "L_bisim": L_bisim.item(),
        "L_latent_reg": latent_reg.item(),
        "L_latent_std": latent_std_loss.item(),
        "geo_mean_dist": geo_dist.mean().item(),
    }

    return loss, logs


# ===============================
#  CEM Planner (continuous actions)
# ===============================

def evaluate_action_sequences(z0: torch.Tensor,
                              act_seqs: torch.Tensor,
                              dynamics: DynamicsModel,
                              reward_model: RewardModel,
                              gamma: float = GAMMA):
    """
    z0: [1, latent_dim]
    act_seqs: [N, H, act_dim]
    return: [N] returns
    """
    N, H, A = act_seqs.shape
    device = z0.device
    z = z0.expand(N, -1)  # [N, latent_dim]
    total_reward = torch.zeros(N, device=device)
    discount = 1.0

    for t in range(H):
        a_t = act_seqs[:, t, :]
        r_t = reward_model(z, a_t).squeeze(-1)
        total_reward = total_reward + discount * r_t
        discount *= gamma
        z = dynamics(z, a_t)

    return total_reward


def cem_plan(z0: torch.Tensor,
             dynamics: DynamicsModel,
             reward_model: RewardModel,
             action_space,
             horizon: int = CEM_HORIZON,
             num_samples: int = CEM_SAMPLES,
             num_iters: int = CEM_ITERS,
             elite_frac: float = CEM_ELITE_FRAC,
             init_std: float = CEM_INIT_STD,
             gamma: float = GAMMA,
             device=None):
    """
    Standard CEM over action sequences.
    Returns a single action (numpy array).
    """
    if device is None:
        device = z0.device

    act_dim = action_space.shape[0]
    mean = torch.zeros(horizon, act_dim, device=device)
    std = torch.ones(horizon, act_dim, device=device) * init_std

    low = torch.as_tensor(action_space.low, device=device)
    high = torch.as_tensor(action_space.high, device=device)

    k = max(1, int(elite_frac * num_samples))

    for _ in range(num_iters):
        eps = torch.randn(num_samples, horizon, act_dim, device=device)
        actions = mean.unsqueeze(0) + std.unsqueeze(0) * eps
        actions = torch.clamp(actions, low, high)

        with torch.no_grad():
            returns = evaluate_action_sequences(z0, actions, dynamics, reward_model, gamma=gamma)

        values, idx = returns.topk(k)
        elites = actions[idx]  # [k, H, act_dim]

        mean = elites.mean(dim=0)
        std = elites.std(dim=0) + 1e-6

    best_action = mean[0]  # first time step
    return best_action.cpu().numpy()


# ===============================
#  Training + Evaluation
# ===============================

def evaluate_env(env_id,
                 encoder: Encoder,
                 dynamics: DynamicsModel,
                 reward_model: RewardModel,
                 horizon: int,
                 episodes: int = 5,
                 device=None):
    device = device or get_device()
    env = gym.make(env_id)
    returns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)  # [1, obs_dim]
            with torch.no_grad():
                z0 = encoder(obs_t)  # [1, latent_dim]
                action = cem_plan(z0, dynamics, reward_model, env.action_space,
                                  horizon=horizon, device=device)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_ret += reward

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns))


def main():
    set_seed(0)
    device = get_device()
    print("Using device:", device)

    env = gym.make(ENV_ID)
    obs, _ = env.reset()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    encoder = Encoder(obs_dim, LATENT_DIM).to(device)
    dynamics = DynamicsModel(LATENT_DIM, act_dim).to(device)
    reward_model = RewardModel(LATENT_DIM, act_dim).to(device)

    world_model_params = list(encoder.parameters()) + \
                         list(dynamics.parameters()) + \
                         list(reward_model.parameters())
    optimizer = torch.optim.Adam(world_model_params, lr=1e-4, weight_decay=1e-4)

    replay = ReplayBuffer(capacity=100_000, obs_dim=obs_dim, act_dim=act_dim)

    writer = SummaryWriter(log_dir="runs/planet_geom_cem")

    global_step = 0
    episode_idx = 0
    episode_return = 0.0
    episode_len = 0

    start_time = time.time()

    while global_step < MAX_ENV_STEPS and (time.time() - start_time) < MAX_RUNTIME:
        # Acting
        if global_step < RANDOM_STEPS:
            action = env.action_space.sample()
        else:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                z0 = encoder(obs_t)
                action = cem_plan(z0, dynamics, reward_model, env.action_space,
                                  horizon=CEM_HORIZON, device=device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay.add(np.asarray(obs, dtype=np.float32),
                   np.asarray(action, dtype=np.float32),
                   float(reward),
                   np.asarray(next_obs, dtype=np.float32),
                   done)

        episode_return += reward
        episode_len += 1
        global_step += 1
        obs = next_obs

        if done:
            writer.add_scalar("charts/episode_return", episode_return, global_step)
            writer.add_scalar("charts/episode_length", episode_len, global_step)
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0
            episode_idx += 1

        # Training
        if global_step >= RANDOM_STEPS and global_step % TRAIN_EVERY == 0:
            print(f"\n[Train] step={global_step}, buffer_size={replay.size}")
            for it in range(TRAIN_ITERS):
                batch = replay.sample_sequences(BATCH_SIZE, SEQ_LEN)
                loss, logs = world_model_loss(batch, encoder, dynamics, reward_model, device=device)

                if not torch.isfinite(loss):
                    print(f"  iter {it+1}/{TRAIN_ITERS}, loss is non-finite, skipping step")
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(world_model_params, 10.0)
                optimizer.step()

                if (it + 1) % 50 == 0:
                    print(f"  iter {it+1}/{TRAIN_ITERS}, loss={loss.item():.4f}")

            for k, v in logs.items():
                writer.add_scalar(f"loss/{k}", v, global_step)

            # Quick evaluation at two horizons
            print("[Eval] Running evaluation...")
            ret_H10 = evaluate_env(ENV_ID, encoder, dynamics, reward_model,
                                   horizon=10, episodes=3, device=device)
            ret_H30 = evaluate_env(ENV_ID, encoder, dynamics, reward_model,
                                   horizon=30, episodes=3, device=device)

            writer.add_scalar("eval/return_H10", ret_H10, global_step)
            writer.add_scalar("eval/return_H30", ret_H30, global_step)

            print(f"  Eval H=10: {ret_H10:.2f}, H=30: {ret_H30:.2f}")

    env.close()
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
