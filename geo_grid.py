import time
import math
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


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
RECON_LOSS_WEIGHT = 1.0
IMG_DOWNSAMPLE = 3  # simple stride downsample of rendered frames

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
#  Experiment toggles
# ===============================

# If False, disables geometric + bisim losses (baseline).
USE_GEOMETRIC_LOSSES = True
MAKE_GEOMETRY_PLOT = True
RUN_BASE_NAME = "planet_cem"


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


def downsample_image(img: np.ndarray, factor: int = IMG_DOWNSAMPLE):
    if factor <= 1:
        return img
    return img[::factor, ::factor]


class PixelObsWrapper(gym.Wrapper):
    """
    Replace state observations with rendered RGB frames (downsampled).
    """
    def __init__(self, env_id: str, downsample: int = IMG_DOWNSAMPLE):
        env = gym.make(env_id, render_mode="rgb_array")
        super().__init__(env)
        self.downsample = downsample
        # Probe to set observation space
        obs, _ = self.env.reset()
        img = downsample_image(self.env.render(), self.downsample)
        self.obs_shape = img.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.obs_shape, dtype=np.uint8
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        img = downsample_image(self.env.render(), self.downsample)
        return img, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        img = downsample_image(self.env.render(), self.downsample)
        return img, reward, terminated, truncated, info


# ===============================
#  Replay Buffer
# ===============================

class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, act_dim: int):
        """
        obs_shape: (H, W, C) pixel observations stored as uint8 for memory efficiency.
        """
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)

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
        """
        Sample sequences from the flat buffer.
        Sequences may cross episode boundaries, but downstream losses
        mask out transitions after dones.
        """
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


class ConvEncoder(nn.Module):
    """
    Lightweight CNN encoder for pixel observations (C, H, W) -> latent.
    Uses three stride-2 conv blocks to downsample.
    """
    def __init__(self, obs_shape, latent_dim=LATENT_DIM):
        super().__init__()
        c, h, w = obs_shape
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        # determine flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            out = self.conv(dummy)
            self.conv_out_shape = out.shape[1:]  # (C, H, W)
            conv_out_size = out.view(1, -1).size(1)
        self.fc = nn.Linear(conv_out_size, latent_dim)

    def forward(self, obs):
        """
        obs: [B, T, C, H, W] or [B, C, H, W]
        """
        if obs.dim() == 5:
            B, T, C, H, W = obs.shape
            x = obs.reshape(B * T, C, H, W)
            x = x.contiguous()
            z = self.fc(self.conv(x).view(B * T, -1))
            return torch.tanh(z).view(B, T, -1)
        else:
            obs = obs.contiguous()
            z = self.fc(self.conv(obs).view(obs.size(0), -1))
            return torch.tanh(z)


class ConvDecoder(nn.Module):
    """
    Lightweight CNN decoder: latent -> reconstructed pixels in [0,1].
    Mirrors ConvEncoder with transposed convolutions.
    """
    def __init__(self, latent_dim: int, conv_out_shape):
        super().__init__()
        c, h, w = conv_out_shape
        self.h = h
        self.w = w
        self.fc = nn.Linear(latent_dim, c * h * w)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        """
        z: [B, T, latent_dim] or [B, latent_dim]
        """
        if z.dim() == 3:
            B, T, D = z.shape
            x = z.reshape(B * T, D)
        else:
            B, D = z.shape
            x = z
        x = x.contiguous()
        x = self.fc(x)
        x = x.view(x.size(0), -1, self.h, self.w)
        x = self.deconv(x)
        x = x.contiguous()
        if z.dim() == 3:
            return x.view(B, T, 3, x.size(-2), x.size(-1)).contiguous()
        else:
            return x.contiguous()


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

def pearsonr_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute Pearson correlation between 1D tensors x, y.
    Returns a scalar tensor.
    """
    x = x - x.mean()
    y = y - y.mean()
    denom = (x.norm() * y.norm()).clamp_min(1e-8)
    return (x * y).sum() / denom


def world_model_loss(batch: ReplayBuffer.Batch,
                     encoder: ConvEncoder,
                     decoder: ConvDecoder,
                     dynamics: DynamicsModel,
                     reward_model: RewardModel,
                     device,
                     gamma: float = GAMMA,
                     alpha_geo: float = ALPHA_GEO,
                     alpha_bisim: float = ALPHA_BISIM,
                     return_latents: bool = False):
    """
    batch: obs, actions, rews, dones as numpy arrays

    Returns:
      loss: scalar tensor
      logs: dict of scalar metrics
      extras: optional tensors for visualization (latent histograms)
    """

    obs = torch.tensor(batch.obs, dtype=torch.float32, device=device) / 255.0      # [B, T, H, W, C]
    obs = obs.permute(0, 1, 4, 2, 3)                                               # [B, T, C, H, W]
    actions = torch.tensor(batch.actions, dtype=torch.float32, device=device)      # [B, T, act_dim]
    rews = torch.tensor(batch.rews, dtype=torch.float32, device=device)            # [B, T]
    dones = torch.tensor(batch.dones, dtype=torch.float32, device=device)          # [B, T]

    B, T, C, H, W = obs.shape

    # Encode observations to latent
    z = encoder(obs)                      # [B, T, latent_dim]
    latent_reg = (z.pow(2).mean()) if LATENT_NORM_WEIGHT > 0 else torch.tensor(0.0, device=device)

    # Encourage non-collapsed latents (per-dimension std close to target)
    latent_std = z.std(dim=(0, 1))
    latent_std_loss = (latent_std - LATENT_STD_TARGET).abs().mean()

    # Prepare one-step tensors (T-1 transitions)
    z_t = z[:, :-1, :]                   # [B, T-1, latent_dim]
    z_tp1_target = z[:, 1:, :]           # [B, T-1, latent_dim]
    a_t = actions[:, :-1, :]             # [B, T-1, act_dim]
    r_t = rews[:, :-1]                   # [B, T-1]
    d_t = dones[:, :-1]                  # [B, T-1]

    # Mask to ignore transitions after episode is done (no cross-episode dynamics)
    mask = (1.0 - d_t)                   # 1 where transition is valid, 0 after done

    # One-step prediction: dynamics and reward
    z_tp1_pred = dynamics(z_t.reshape(-1, LATENT_DIM),
                          a_t.reshape(-1, a_t.size(-1)))
    z_tp1_pred = z_tp1_pred.reshape(B, T - 1, LATENT_DIM)

    r_pred = reward_model(z_t.reshape(-1, LATENT_DIM),
                          a_t.reshape(-1, a_t.size(-1)))
    r_pred = r_pred.view(B, T - 1)

    # Reconstruction loss (decoder)
    recon = decoder(z)  # [B, T, 3, H_dec, W_dec] or [B, 3, H_dec, W_dec]
    if recon.dim() == 5:
        recon = recon.view(B * T, 3, recon.size(-2), recon.size(-1))
    target = obs.reshape(B * T, 3, H, W)
    # Adjust recon size to match target if needed
    if recon.shape[-2:] != target.shape[-2:]:
        recon = F.interpolate(recon, size=target.shape[-2:], mode="bilinear", align_corners=False)
    mse = nn.MSELoss()
    L_recon = mse(recon, target)

    # One-step dynamics loss with masking over valid transitions
    dyn_err = (z_tp1_pred - z_tp1_target.detach()).pow(2).sum(-1)  # [B, T-1]
    L_dyn = (dyn_err * mask).sum() / mask.sum().clamp_min(1.0)

    # Reward loss (no masking needed: last reward before done is still valid)
    L_r = mse(r_pred, r_t)

    # Geodesic rollout alignment (multi-step) with mask
    z0 = z[:, 0, :]                        # [B, latent_dim]
    a_seq = actions[:, :-1, :]             # [B, T-1, act_dim]
    z_roll = rollout_dynamics(dynamics, z0, a_seq)  # [B, T-1, latent_dim]
    z_true_seq = z[:, 1:, :]               # [B, T-1, latent_dim]

    # Euclidean distance as proxy for latent geodesic
    geo_dist = torch.sqrt(torch.clamp((z_roll - z_true_seq.detach()).pow(2).sum(-1), min=1e-8))  # [B, T-1]
    #geo_dist = torch.clamp(geo_dist, max=BISIM_CLIP_VALUE)
    geo_err = (geo_dist ** 2) * mask
    L_geo = geo_err.sum() / mask.sum().clamp_min(1.0)

    # Bisimulation-inspired metric shaping
    # Compute discounted returns as approximate values
    returns = compute_discounted_returns(rews, dones, gamma=gamma)  # [B, T]
    z_flat = z.reshape(-1, LATENT_DIM)                              # [B*T, latent_dim]
    v_flat = returns.reshape(-1)                                    # [B*T]

    num_pairs = min(1024, z_flat.size(0))
    idx1 = torch.randint(0, z_flat.size(0), (num_pairs,), device=device)
    idx2 = torch.randint(0, z_flat.size(0), (num_pairs,), device=device)

    z1, z2 = z_flat[idx1], z_flat[idx2]
    v1, v2 = v_flat[idx1], v_flat[idx2]
    value_diff = (v1 - v2).abs()

    #value_diff = torch.clamp(value_diff, max=BISIM_CLIP_VALUE)
    dist = torch.sqrt(torch.clamp((z1 - z2).pow(2).sum(-1), min=1e-8))
    #dist = torch.clamp(dist, max=BISIM_CLIP_VALUE)
    # Regress distance toward value difference
    L_bisim = ((dist - value_diff.detach()) ** 2).mean()

    logs = {}

    # --- Extra metrics: correlation & return-prediction error ---
    with torch.no_grad():
        # Pearson correlation between latent distance and value difference
        valid = torch.isfinite(dist) & torch.isfinite(value_diff)
        if valid.sum() >= 2:
            corr = pearsonr_torch(dist[valid], value_diff[valid]).item()
        else:
            corr = float("nan")
        logs["corr_latent_dist_value_diff"] = corr

        # Return prediction using multi-step model rollout vs true returns
        # Predicted rewards along rollout
        r_roll = reward_model(z_roll.reshape(-1, LATENT_DIM),
                              a_seq.reshape(-1, a_seq.size(-1)))
        r_roll = r_roll.view(B, T - 1)
        # Predicted returns for each time step (T-1), using same dones mask
        pred_returns = compute_discounted_returns(r_roll, d_t, gamma=gamma)  # [B, T-1]
        true_returns = returns[:, :-1]                                      # [B, T-1]

        ret_abs_err = (pred_returns - true_returns).abs() * mask
        ret_abs_err_mean = ret_abs_err.sum() / mask.sum().clamp_min(1.0)
        logs["ret_pred_abs_error"] = ret_abs_err_mean.item()

        # Geometric mean distance over valid transitions
        geo_mean = (geo_dist * mask).sum() / mask.sum().clamp_min(1.0)
        logs["geo_mean_dist"] = geo_mean.item()

    # Total loss with weights
    loss = L_dyn + REWARD_LOSS_WEIGHT * L_r
    loss = loss + alpha_geo * L_geo + alpha_bisim * L_bisim
    loss = loss + LATENT_NORM_WEIGHT * latent_reg + LATENT_STD_WEIGHT * latent_std_loss
    loss = loss + RECON_LOSS_WEIGHT * L_recon

    # Collect scalar logs (loss components)
    logs.update({
        "L_dyn": L_dyn.item(),
        "L_r": L_r.item(),
        "L_geo": L_geo.item(),
        "L_bisim": L_bisim.item(),
        "L_latent_reg": latent_reg.item(),
        "L_latent_std": latent_std_loss.item(),
        "L_recon": L_recon.item(),
    })

    # Optional extras for histograms
    extras = {}
    if return_latents:
        extras["z_flat"] = z.detach().reshape(-1, LATENT_DIM)
        extras["latent_std_vec"] = latent_std.detach()

    return loss, logs, extras


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
#  Geometry visualization
# ===============================

def visualize_geometry(batch: ReplayBuffer.Batch,
                       encoder: ConvEncoder,
                       device,
                       gamma: float,
                       save_path: str,
                       num_bins: int = 30,
                       clip_to_bisim: bool = True):
    """
    Make a scatter plot + binned profile of latent distance vs value difference.

    If clip_to_bisim=True, we clip |Δvalue| to BISIM_CLIP_VALUE, matching the
    scale used in training for L_bisim. This makes the relationship much clearer.
    """

    obs = torch.tensor(batch.obs, dtype=torch.float32, device=device) / 255.0  # [B, T, H, W, C]
    obs = obs.permute(0, 1, 4, 2, 3)  # [B, T, C, H, W]
    rews = torch.tensor(batch.rews, dtype=torch.float32, device=device)  # [B, T]
    dones = torch.tensor(batch.dones, dtype=torch.float32, device=device)# [B, T]

    with torch.no_grad():
        z = encoder(obs)  # [B, T, latent_dim]
        returns = compute_discounted_returns(rews, dones, gamma=gamma)  # [B, T]

        z_flat = z.reshape(-1, LATENT_DIM)
        v_flat = returns.reshape(-1)

        # Sample random pairs for scatter + profile
        num_pairs = min(5000, z_flat.size(0))
        idx1 = torch.randint(0, z_flat.size(0), (num_pairs,), device=device)
        idx2 = torch.randint(0, z_flat.size(0), (num_pairs,), device=device)

        z1, z2 = z_flat[idx1], z_flat[idx2]
        v1, v2 = v_flat[idx1], v_flat[idx2]

        dist = torch.sqrt(torch.clamp((z1 - z2).pow(2).sum(-1), min=1e-8)).cpu().numpy()
        val_diff = (v1 - v2).abs().cpu().numpy()

    """
    if clip_to_bisim:
        val_diff = np.minimum(val_diff, BISIM_CLIP_VALUE)
    """
    # Build binned profile: mean |Δvalue| per distance bin
    if dist.max() > 0:
        bins = np.linspace(0.0, dist.max(), num_bins + 1)
    else:
        bins = np.linspace(0.0, 1.0, num_bins + 1)

    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_means = np.zeros(num_bins)
    bin_stds = np.zeros(num_bins)

    for i in range(num_bins):
        mask = (dist >= bins[i]) & (dist < bins[i+1])
        if mask.sum() > 0:
            vals = val_diff[mask]
            bin_means[i] = vals.mean()
            bin_stds[i] = vals.std()
        else:
            bin_means[i] = np.nan
            bin_stds[i] = np.nan

    # Plot: scatter + profile
    plt.figure(figsize=(7, 6))
    plt.scatter(dist, val_diff, alpha=0.05, s=5, label="pairs")
    plt.xlabel("Latent distance ||z_i - z_j||")
    plt.ylabel(f"Value difference |G_i - G_j| )"
               if clip_to_bisim else "Value difference |G_i - G_j|")

    # Overlay line for bins
    valid_bins = ~np.isnan(bin_means)
    if valid_bins.sum() > 1:
        plt.plot(bin_centers[valid_bins], bin_means[valid_bins],
                 linewidth=2.0, label="bin-avg |Δvalue|")
        # optional shaded band (±1 std)
        lower = bin_means[valid_bins] - bin_stds[valid_bins]
        upper = bin_means[valid_bins] + bin_stds[valid_bins]
        plt.fill_between(bin_centers[valid_bins], lower, upper, alpha=0.2)

    plt.title("Latent geometry vs value differences")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ===============================
#  Training + Evaluation
# ===============================

def evaluate_env(env_id,
                 encoder: ConvEncoder,
                 dynamics: DynamicsModel,
                 reward_model: RewardModel,
                 horizon: int,
                 episodes: int = 5,
                 device=None):
    device = device or get_device()
    env = PixelObsWrapper(env_id, downsample=IMG_DOWNSAMPLE)
    returns = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
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

    env = PixelObsWrapper(ENV_ID, downsample=IMG_DOWNSAMPLE)
    obs, _ = env.reset()
    obs_shape = env.observation_space.shape  # (H, W, C)
    # Convert to (C, H, W) for model shapes
    c, h, w = obs_shape[2], obs_shape[0], obs_shape[1]
    act_dim = env.action_space.shape[0]

    encoder = ConvEncoder((c, h, w), LATENT_DIM).to(device)
    decoder = ConvDecoder(LATENT_DIM, encoder.conv_out_shape).to(device)
    dynamics = DynamicsModel(LATENT_DIM, act_dim).to(device)
    reward_model = RewardModel(LATENT_DIM, act_dim).to(device)

    # Choose weights depending on baseline vs geometric run
    if USE_GEOMETRIC_LOSSES:
        alpha_geo = ALPHA_GEO
        alpha_bisim = ALPHA_BISIM
        run_name = f"{RUN_BASE_NAME}_geom"
    else:
        alpha_geo = 0.0
        alpha_bisim = 0.0
        run_name = f"{RUN_BASE_NAME}_baseline"

    world_model_params = list(encoder.parameters()) + \
                         list(decoder.parameters()) + \
                         list(dynamics.parameters()) + \
                         list(reward_model.parameters())
    optimizer = torch.optim.Adam(world_model_params, lr=1e-4, weight_decay=1e-4)

    replay = ReplayBuffer(capacity=100_000, obs_shape=obs_shape, act_dim=act_dim)

    writer = SummaryWriter(log_dir=f"runs/{run_name}")

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
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device) / 255.0
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
            with torch.no_grad():
                z0 = encoder(obs_t)
                action = cem_plan(z0, dynamics, reward_model, env.action_space,
                                  horizon=CEM_HORIZON, device=device)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        replay.add(np.asarray(obs, dtype=np.uint8),
                   np.asarray(action, dtype=np.float32),
                   float(reward),
                   np.asarray(next_obs, dtype=np.uint8),
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
            extras = {}
            for it in range(TRAIN_ITERS):
                batch = replay.sample_sequences(BATCH_SIZE, SEQ_LEN)
                return_latents = (it == TRAIN_ITERS - 1)
                loss, logs, extras = world_model_loss(
                    batch,
                    encoder,
                    decoder,
                    dynamics,
                    reward_model,
                    device=device,
                    gamma=GAMMA,
                    alpha_geo=alpha_geo,
                    alpha_bisim=alpha_bisim,
                    return_latents=return_latents,
                )

                if not torch.isfinite(loss):
                    print(f"  iter {it+1}/{TRAIN_ITERS}, loss is non-finite, skipping step")
                    continue

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(world_model_params, 10.0)
                optimizer.step()

                if (it + 1) % 50 == 0:
                    print(f"  iter {it+1}/{TRAIN_ITERS}, loss={loss.item():.4f}")

            # Log scalars
            for k, v in logs.items():
                if k.startswith("corr_") or k.startswith("ret_pred_") or k == "geo_mean_dist":
                    writer.add_scalar(f"metrics/{k}", v, global_step)
                else:
                    writer.add_scalar(f"loss/{k}", v, global_step)

            # Histograms of latent norms/stds (geometry sanity check)
            if "z_flat" in extras and "latent_std_vec" in extras:
                z_flat = extras["z_flat"]
                latent_std_vec = extras["latent_std_vec"]

                z_norms = z_flat.norm(dim=-1).cpu().numpy()
                writer.add_histogram("latent/norms", z_norms, global_step)
                writer.add_histogram("latent/std_per_dim", latent_std_vec.cpu().numpy(), global_step)

            # Quick evaluation at two horizons
            print("[Eval] Running evaluation...")
            ret_H10 = evaluate_env(ENV_ID, encoder, dynamics, reward_model,
                                   horizon=10, episodes=3, device=device)
            ret_H30 = evaluate_env(ENV_ID, encoder, dynamics, reward_model,
                                   horizon=30, episodes=3, device=device)

            writer.add_scalar("eval/return_H10", ret_H10, global_step)
            writer.add_scalar("eval/return_H30", ret_H30, global_step)

            print(f"  Eval H=10: {ret_H10:.2f}, H=30: {ret_H30:.2f}")

    # Final geometry visualization: latent distance vs value difference
    if MAKE_GEOMETRY_PLOT and replay.size > SEQ_LEN + 1:
        batch_vis = replay.sample_sequences(batch_size=64, seq_len=SEQ_LEN)
        save_path = f"{run_name}_geom_scatter.png"
        visualize_geometry(batch_vis, encoder, device, GAMMA, save_path)
        print(f"Saved geometry visualization to {save_path}")

    env.close()
    writer.close()
    print("Training finished.")


if __name__ == "__main__":
    main()
