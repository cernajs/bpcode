#!/usr/bin/env python3
"""
PlaNet-style RSSM + Pendulum environment + geometry-aware planning with
decoder-Jacobian geometry (GELATO-lite).

Pipeline:
1) Collect offline data with random policy in the Pendulum environment.
2) Train a minimal PlaNet-like RSSM world model on that data.
3) Encode training sequences into latent space; build a kNN graph.
4) During planning:
   - For each candidate action sequence, simulate latent trajectory
     using the learned prior (imagination).
   - Add penalty proportional to decoder Jacobian metric (trace/logdet)
     as a proxy for local uncertainty / off-manifoldness.
   - Compare planning with/without geometry penalty.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gymnasium as gym
from PIL import Image


# ============================================================
# 0. Pendulum Environment (Classic Control)
# ============================================================

class PixelPendulumEnv:
    """
    Gymnasium Pendulum-v1 with pixel observations.
    - Renders to RGB, resizes to 64x64, flattens to a vector in [0,1].
    """

    def __init__(self, size=64):
        self.size = size
        self.env = gym.make("Pendulum-v1", render_mode="rgb_array")
        self.env.reset()
        self.action_dim = 1
        self.last_state_lowdim = None
        sample, _ = self._get_pixels(return_state=True)
        self.obs_dim = sample.size

    def _get_pixels(self, return_state=False):
        frame = self.env.render()
        img = Image.fromarray(frame).resize((self.size, self.size))
        arr = np.asarray(img).astype(np.float32) / 255.0  # (H, W, 3)
        flat = arr.reshape(-1)
        return (flat, self.last_state_lowdim) if return_state else flat

    def reset(self):
        obs_lowdim, _ = self.env.reset()
        self.last_state_lowdim = obs_lowdim
        pixels, _ = self._get_pixels(return_state=True)
        return pixels

    def step(self, action):
        a = np.clip(action, -2.0, 2.0)
        obs_lowdim, reward, terminated, truncated, _ = self.env.step(a)
        self.last_state_lowdim = obs_lowdim
        done = terminated or truncated
        obs = self._get_pixels()
        return obs, float(reward), done, {}

    def set_state(self, obs):
        # obs assumed to be lowdim [cos(theta), sin(theta), theta_dot]
        cos_theta, sin_theta, theta_dot = obs
        theta = np.arctan2(sin_theta, cos_theta)
        self.env.unwrapped.state = np.array([theta, theta_dot], dtype=np.float32)
        self.last_state_lowdim = np.array([cos_theta, sin_theta, theta_dot], dtype=np.float32)

    @property
    def max_torque(self):
        return 2.0


# ============================================================
# 1. Simple PlaNet-style RSSM
# ============================================================

def mlp(in_dim, out_dim, hidden_dim=128):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class SimplePlaNetRSSM(nn.Module):
    """
    Simplified PlaNet/Dreamer-style RSSM:

    h_t : deterministic hidden state (GRU)
    z_t : stochastic latent (Gaussian)
    prior p(z_t | h_t)
    posterior q(z_t | h_t, o_t)
    decoder p(o_t | h_t, z_t) via MLP (MSE loss)
    reward head r_t(h_t, z_t) via MLP (MSE loss)
    """

    def __init__(self, obs_dim, action_dim,
                 latent_dim=16, hidden_dim=128,
                 num_decoders=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_decoders = num_decoders

        # Deterministic part
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior p(z_t | h_t)
        self.prior_net = mlp(hidden_dim, 2 * latent_dim, hidden_dim=128)

        # Posterior q(z_t | h_t, o_t)
        self.post_net = mlp(hidden_dim + obs_dim, 2 * latent_dim, hidden_dim=128)

        # Observation decoder p(o_t | h_t, z_t) ~ N(μ, Σ)
        self.dec_obs_list = nn.ModuleList([
            mlp(hidden_dim + latent_dim, 2 * obs_dim, hidden_dim=128)
            for _ in range(num_decoders)
        ])

        # Reward decoder r_t(h_t, z_t)
        self.dec_rew = mlp(hidden_dim + latent_dim, 1, hidden_dim=128)

    def init_state(self, batch_size):
        h = torch.zeros(batch_size, self.hidden_dim)
        z = torch.zeros(batch_size, self.latent_dim)
        return h, z

    def _split_mean_logstd(self, x):
        mean, log_std = torch.chunk(x, 2, dim=-1)
        log_std = torch.clamp(log_std, -5.0, 2.0)
        return mean, log_std

    def prior(self, h):
        stats = self.prior_net(h)
        return self._split_mean_logstd(stats)

    def posterior(self, h, obs):
        inp = torch.cat([h, obs], dim=-1)
        stats = self.post_net(inp)
        return self._split_mean_logstd(stats)

    def sample(self, mean, log_std):
        eps = torch.randn_like(mean)
        return mean + eps * torch.exp(log_std)

    def decode_obs_mean_logstd(self, h, z, decoder_idx=None):
        """
        If decoder_idx is None, returns the average mean/log_std across decoders.
        """
        xs = torch.cat([h, z], dim=-1)
        outs = []
        decoders = self.dec_obs_list if decoder_idx is None else [self.dec_obs_list[decoder_idx]]
        for dec in decoders:
            out = dec(xs)
            mean, log_std = torch.chunk(out, 2, dim=-1)
            log_std = torch.clamp(log_std, -5.0, 2.0)
            outs.append((mean, log_std))
        mean_stack = torch.stack([m for m, _ in outs], dim=0)
        logstd_stack = torch.stack([l for _, l in outs], dim=0)
        mean_avg = mean_stack.mean(dim=0)
        logstd_avg = logstd_stack.mean(dim=0)
        return mean_avg, logstd_avg

    def decode_rew(self, h, z):
        return self.dec_rew(torch.cat([h, z], dim=-1))

    def gru_step(self, h, z, a):
        inp = torch.cat([z, a], dim=-1)
        return self.gru(inp, h)

    def imagine_step(self, h, z, a):
        """
        One-step imagination:
        h_{t+1} = GRU(h_t, [z_t, a_t])
        z_{t+1} = mean of prior p(z_{t+1} | h_{t+1})
        """
        h_next = self.gru_step(h, z, a)
        prior_mean, prior_log_std = self.prior(h_next)
        z_next = prior_mean  # deterministic mean for planning
        return h_next, z_next

    def encode_observation(self, obs_np):
        """
        Encode a single observation into (h, z) using the posterior with zero h.
        """
        self.eval()
        with torch.no_grad():
            obs = torch.from_numpy(obs_np).float().unsqueeze(0)  # (1, obs_dim)
            h = torch.zeros(1, self.hidden_dim)
            post_mean, post_log_std = self.posterior(h, obs)
            z = post_mean  # use mean for encoding
        return h, z


# Helper: Gaussian NLL for reconstruction
def gaussian_nll(mean, log_std, target):
    """Diagonal Gaussian negative log-likelihood."""
    var_inv = torch.exp(-2.0 * log_std)
    nll = 0.5 * ((target - mean) ** 2 * var_inv + 2.0 * log_std + np.log(2 * np.pi))
    return nll.mean()


# Geometry: decoder Jacobian-based score u(z)
def latent_geometry_score(model, h, z, alpha=0.5, metric="trace", eps=1e-5):
    """
    Compute a local geometry score at latent z using decoder Jacobians:
        G(z) = alpha * J_mu^T J_mu + (1 - alpha) * J_logstd^T J_logstd
    Score options:
        - "trace": trace(G)
        - "logdet": logdet(G + eps I)
    """
    assert z.dim() == 2 and z.size(0) == 1, "Expect z shape (1, latent_dim)"

    z = z.detach().clone().requires_grad_(True)
    h = h.detach()  # treated as constant wrt z

    def jacobian_for_decoder(dec_idx):
        def dec_mu(z_in):
            mu, _ = model.decode_obs_mean_logstd(h, z_in, decoder_idx=dec_idx)
            return mu

        def dec_logstd(z_in):
            _, log_std = model.decode_obs_mean_logstd(h, z_in, decoder_idx=dec_idx)
            return log_std

        J_mu_full = torch.autograd.functional.jacobian(dec_mu, z, create_graph=False)
        J_logstd_full = torch.autograd.functional.jacobian(dec_logstd, z, create_graph=False)
        J_mu = J_mu_full.squeeze(0).squeeze(1)
        J_logstd = J_logstd_full.squeeze(0).squeeze(1)
        return J_mu, J_logstd

    G_accum = None
    for dec_idx in range(model.num_decoders):
        J_mu, J_logstd = jacobian_for_decoder(dec_idx)
        G_local = alpha * (J_mu.T @ J_mu) + (1.0 - alpha) * (J_logstd.T @ J_logstd)
        G_accum = G_local if G_accum is None else G_accum + G_local
    G = G_accum / float(model.num_decoders)

    if metric == "trace":
        score = torch.trace(G)
    elif metric == "logdet":
        score = torch.logdet(G + eps * torch.eye(G.size(0)))
    else:
        raise ValueError(f"Unknown metric {metric}")

    return float(score.detach().cpu())


# ============================================================
# 2. Dataset collection
# ============================================================

def collect_dataset(env, n_episodes=500, T=50, policy='random'):
    """
    Collect trajectories with specified policy.
    
    policy: 'random' - uniform random actions
            'noisy_swing' - attempts swing-up with noise (better coverage)

    Returns:
        obs: (N, T+1, obs_dim)
        actions: (N, T, action_dim)
        rewards: (N, T)
    """
    obs_dim = env.obs_dim
    act_dim = env.action_dim

    obs = np.zeros((n_episodes, T + 1, obs_dim), dtype=np.float32)
    actions = np.zeros((n_episodes, T, act_dim), dtype=np.float32)
    rewards = np.zeros((n_episodes, T), dtype=np.float32)

    for ep in range(n_episodes):
        s = env.reset()
        obs[ep, 0] = s
        
        for t in range(T):
            if policy == 'random':
                a = np.random.uniform(-env.max_torque, env.max_torque, size=(act_dim,))
            elif policy == 'noisy_swing':
                # Simple energy-based swing with noise
                theta_dot = s[2]
                # Try to add energy when moving in the right direction
                a_base = np.sign(theta_dot) * env.max_torque * 0.5
                a = np.array([a_base + np.random.uniform(-1.0, 1.0)])
                a = np.clip(a, -env.max_torque, env.max_torque)
            elif policy == 'wide':
                # Larger exploratory actions to generate OOD coverage
                a = np.random.uniform(-1.5 * env.max_torque,
                                      1.5 * env.max_torque,
                                      size=(act_dim,))
                a = np.clip(a, -2.0 * env.max_torque, 2.0 * env.max_torque)
            else:
                a = np.random.uniform(-env.max_torque, env.max_torque, size=(act_dim,))
            
            actions[ep, t] = a
            s, r, done, _ = env.step(a)
            obs[ep, t + 1] = s
            rewards[ep, t] = r

    return obs, actions, rewards


# Encode dataset latents with posterior means
def encode_dataset_latents(model, obs, actions):
    """
    obs: (N, T+1, obs_dim)
    actions: (N, T, act_dim)
    Returns flattened latents (N*T, latent_dim)
    """
    model.eval()
    device = next(model.parameters()).device

    N, T_plus_1, obs_dim = obs.shape
    T = T_plus_1 - 1
    _, T_act, act_dim = actions.shape
    assert T_act == T

    obs_torch = torch.from_numpy(obs).float().to(device)
    act_torch = torch.from_numpy(actions).float().to(device)

    latents = []
    with torch.no_grad():
        for ep in range(N):
            h, z = model.init_state(batch_size=1)
            h = h.to(device)
            z = z.to(device)
            a_prev = torch.zeros(1, act_dim, device=device)

            for t in range(T):
                o_t = obs_torch[ep, t:t+1, :]
                a_t = act_torch[ep, t:t+1, :]

                post_mean, post_log_std = model.posterior(h, o_t)
                z_mean = post_mean

                latents.append(z_mean.squeeze(0).cpu().numpy())

                h = model.gru_step(h, z_mean, a_prev)
                a_prev = a_t

    return np.stack(latents, axis=0)


# Simple PCA projection helper
def compute_pca_projection(Z, k=2):
    Z_mean = Z.mean(axis=0, keepdims=True)
    Z_center = Z - Z_mean
    U, S, Vt = np.linalg.svd(Z_center, full_matrices=False)
    comps = Vt[:k]

    def project(X):
        return (X - Z_mean) @ comps.T

    return project


# Visualization of dataset latents and imagined vs actual latent rollouts
def visualize_latent_rollout(model, env, obs, actions,
                             action_seq, filename, title="latent_rollout",
                             start_pixels=None, start_state_lowdim=None):
    Z_data = encode_dataset_latents(model, obs, actions)
    project = compute_pca_projection(Z_data, k=2)

    # Start state for visualization
    if start_pixels is None or start_state_lowdim is None:
        # fallback random
        obs0 = env.reset()
        s0_lowdim = env.last_state_lowdim
        s0_pixels = obs0
    else:
        s0_pixels = start_pixels
        s0_lowdim = start_state_lowdim

    env_sim = PixelPendulumEnv(size=env.size)
    env_sim.set_state(s0_lowdim)

    h, z = model.encode_observation(s0_pixels)
    z_imag = [z.squeeze(0).cpu().numpy()]
    for a in action_seq:
        a_t = torch.from_numpy(a).float().unsqueeze(0)
        h, z = model.imagine_step(h, z, a_t)
        z_imag.append(z.squeeze(0).detach().cpu().numpy())
    z_imag = np.stack(z_imag, axis=0)

    z_true = []
    env_sim.set_state(s0_lowdim)
    for a in action_seq:
        s_next, _, done, _ = env_sim.step(a)
        h_enc, z_enc = model.encode_observation(s_next)
        z_true.append(z_enc.squeeze(0).cpu().numpy())
        if done:
            break
    z_true = np.stack(z_true, axis=0)

    Z_proj = project(Z_data)
    imag_proj = project(z_imag)
    true_proj = project(z_true)

    plt.figure(figsize=(7, 6))
    plt.scatter(Z_proj[:, 0], Z_proj[:, 1], s=6, alpha=0.08, color="#4c72b0", label="dataset latents")

    # Color by step index
    steps_imag = np.arange(len(imag_proj))
    steps_true = np.arange(len(true_proj))
    cmap_imag = plt.cm.Oranges
    cmap_true = plt.cm.Greens

    plt.plot(imag_proj[:, 0], imag_proj[:, 1], '-', color=cmap_imag(0.7), lw=2, alpha=0.9, label="imagined")
    plt.scatter(imag_proj[:, 0], imag_proj[:, 1], c=steps_imag, cmap=cmap_imag, s=20, alpha=0.9, edgecolors='k', linewidths=0.3)

    plt.plot(true_proj[:, 0], true_proj[:, 1], '-', color=cmap_true(0.7), lw=2, alpha=0.9, label="actual (encoded)")
    plt.scatter(true_proj[:, 0], true_proj[:, 1], c=steps_true, cmap=cmap_true, s=20, alpha=0.9, edgecolors='k', linewidths=0.3)

    # Start/end markers
    plt.scatter(imag_proj[0, 0], imag_proj[0, 1], marker='*', color='red', s=80, label="start")
    plt.scatter(imag_proj[-1, 0], imag_proj[-1, 1], marker='X', color='black', s=60, label="imag end")
    plt.scatter(true_proj[-1, 0], true_proj[-1, 1], marker='P', color='purple', s=60, label="actual end")

    plt.title(title)
    plt.legend(framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    return filename


# ============================================================
# 3. Training
# ============================================================

def kl_gaussian(mean_q, log_std_q, mean_p, log_std_p):
    """
    KL[ N(mean_q, std_q^2) || N(mean_p, std_p^2) ] for diagonal Gaussians.
    """
    var_q = torch.exp(2.0 * log_std_q)
    var_p = torch.exp(2.0 * log_std_p)
    term1 = 2.0 * (log_std_p - log_std_q)
    term2 = (var_q + (mean_q - mean_p) ** 2) / (var_p + 1e-8)
    kl = 0.5 * torch.sum(term1 + term2 - 1.0, dim=-1)
    return kl  # shape (batch,)


def train_world_model(model, obs, actions, rewards,
                      num_epochs=30, batch_size=32,
                      learning_rate=1e-3, beta_kl=0.1,
                      rew_scale=1.0, bisim_beta=0.0, bisim_gamma=0.99):
    """
    Train the RSSM on offline data.
    """
    device = torch.device("cpu")
    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    N, T_plus_1, obs_dim = obs.shape
    T = T_plus_1 - 1
    _, T_act, act_dim = actions.shape
    assert T_act == T

    obs_torch = torch.from_numpy(obs).float().to(device)
    act_torch = torch.from_numpy(actions).float().to(device)
    rew_torch = torch.from_numpy(rewards).float().to(device)

    n_batches = max(1, N // batch_size)

    for epoch in range(num_epochs):
        perm = torch.randperm(N)
        epoch_loss = 0.0

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            if idx.numel() == 0:
                continue

            batch_obs = obs_torch[idx]        # (B, T+1, obs_dim)
            batch_act = act_torch[idx]        # (B, T, act_dim)
            batch_rew = rew_torch[idx]        # (B, T)

            B = batch_obs.size(0)
            h, z = model.init_state(B)
            h = h.to(device)
            z = z.to(device)
            a_prev = torch.zeros(B, act_dim, device=device)

            recon_loss = 0.0
            rew_loss = 0.0
            kl_loss = 0.0
            bisim_loss = 0.0

            for t in range(T):
                o_t = batch_obs[:, t, :]                 # (B, obs_dim)
                r_t = batch_rew[:, t].unsqueeze(-1)      # (B, 1)
                a_t = batch_act[:, t, :]                 # (B, act_dim)

                # Prior & Posterior
                prior_mean, prior_log_std = model.prior(h)
                post_mean, post_log_std = model.posterior(h, o_t)

                # Sample latent from posterior
                z_post = model.sample(post_mean, post_log_std)

                # Decode observation & reward (average over decoder ensemble)
                dec_recon = 0.0
                for dec_idx in range(model.num_decoders):
                    o_mean, o_log_std = model.decode_obs_mean_logstd(h, z_post, decoder_idx=dec_idx)
                    dec_recon = dec_recon + gaussian_nll(o_mean, o_log_std, o_t)
                dec_recon = dec_recon / float(model.num_decoders)

                r_hat = model.decode_rew(h, z_post)

                recon_loss += dec_recon
                rew_loss += ((r_hat - r_t) ** 2).mean()

                # KL q(z|h,o) || p(z|h)
                kl = kl_gaussian(post_mean, post_log_std,
                                 prior_mean, prior_log_std).mean()
                kl_loss += kl

                # Bisimulation-style consistency (optional)
                if bisim_beta > 0.0 and B > 1:
                    i = torch.randint(0, B, (1,)).item()
                    j = torch.randint(0, B, (1,)).item()
                    while j == i and B > 1:
                        j = torch.randint(0, B, (1,)).item()
                    z_i = z_post[i]
                    z_j = z_post[j]
                    o_i = o_t[i]
                    o_j = o_t[j]
                    r_i = r_t[i]
                    r_j = r_t[j]
                    # target distance: observation + reward difference
                    obs_dist = torch.norm(o_i - o_j, p=2)
                    rew_dist = torch.abs(r_i - r_j)
                    target = obs_dist + bisim_gamma * rew_dist
                    latent_dist = torch.norm(z_i - z_j, p=2)
                    bisim_loss += torch.abs(latent_dist - target)

                # Update deterministic state
                h = model.gru_step(h, z_post, a_prev)
                a_prev = a_t

            # Normalize losses by T
            recon_loss = recon_loss / T
            rew_loss = rew_loss / T
            kl_loss = kl_loss / T

            loss = recon_loss + rew_scale * rew_loss + beta_kl * kl_loss
            if bisim_beta > 0.0:
                loss = loss + bisim_beta * (bisim_loss / max(1, T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / max(1, n_batches):.4f}")

    model.eval()


# ============================================================
# 4. Geometry diagnostics: does u(z) track model error?
# ============================================================

def evaluate_geometry_error_correlation(model, env,
                                        n_eval=20, T=40,
                                        policy='noisy_swing',
                                        geometry_alpha=0.5,
                                        geometry_metric="trace"):
    """
    Evaluate correlation between geometry score u(z) and decoder prediction error.
    """
    print("\\n[3] Geometry vs. prediction error (held-out rollouts)")
    obs_eval, act_eval, _ = collect_dataset(env, n_episodes=n_eval, T=T, policy=policy)
    device = next(model.parameters()).device

    scores = []
    errors = []

    model.eval()
    for ep in range(n_eval):
        h, z = model.init_state(batch_size=1)
        h = h.to(device)
        z = z.to(device)
        a_prev = torch.zeros(1, env.action_dim, device=device)

        for t in range(T):
            o_t = torch.from_numpy(obs_eval[ep, t:t+1, :]).float().to(device)
            a_t = torch.from_numpy(act_eval[ep, t:t+1, :]).float().to(device)

            post_mean, post_log_std = model.posterior(h, o_t)
            z_post = post_mean  # deterministic mean for eval

            # Geometry score at z_post
            score = latent_geometry_score(model, h, z_post,
                                          alpha=geometry_alpha,
                                          metric=geometry_metric)

            # Decoder prediction error on current observation
            o_mean, o_log_std = model.decode_obs_mean_logstd(h, z_post)
            err = ((o_mean - o_t) ** 2).mean().item()

            scores.append(score)
            errors.append(err)

            # rollout deterministic state
            h = model.gru_step(h, z_post, a_prev)
            a_prev = a_t

    scores_np = np.array(scores)
    errors_np = np.array(errors)
    corr = np.corrcoef(scores_np, errors_np)[0, 1]

    print(f"    Samples: {len(scores)}")
    print(f"    Geometry score (mean ± std): {scores_np.mean():.3f} ± {scores_np.std():.3f}")
    print(f"    Prediction error (mean ± std): {errors_np.mean():.3f} ± {errors_np.std():.3f}")
    print(f"    Pearson corr(score, error) = {corr:.3f}")

    return corr, scores_np, errors_np


# ============================================================
# 5. Planning with geometry penalty (CEM + random shooting)
# ============================================================

def rollout_sequence_in_model(model, h0, z0, action_seq,
                              lam=0.0,
                              geometry_alpha=0.5,
                              geometry_metric="trace",
                              geometry_window=1,
                              penalty_threshold=None):
    """
    Roll out one action sequence in the learned model's latent space.

    model: SimplePlaNetRSSM
    h0, z0: initial hidden & latent states (torch tensors, shape (1, *))
    action_seq: (H, action_dim) numpy array
    lam: geometry penalty weight
    penalty_threshold: if not None, reject if max_step_penalty > threshold.

    Returns:
        objective: reward - lam * penalty (or -inf if rejected)
        total_reward
        total_penalty
    """
    device = next(model.parameters()).device
    h = h0.clone().to(device)
    z = z0.clone().to(device)

    total_reward = 0.0
    total_penalty = 0.0
    max_step_penalty = 0.0
    recent_penalties = []

    model.eval()
    with torch.no_grad():
        for a_np in action_seq:
            a = torch.from_numpy(a_np).float().unsqueeze(0).to(device)  # (1, act_dim)
            # Imagine next latent
            h, z = model.imagine_step(h, z, a)
            # Predicted reward
            r_hat = model.decode_rew(h, z)
            total_reward += float(r_hat.item())

            if lam > 0.0:
                p = latent_geometry_score(model, h, z,
                                          alpha=geometry_alpha,
                                          metric=geometry_metric)
                recent_penalties.append(p)
                if len(recent_penalties) > geometry_window:
                    recent_penalties.pop(0)
                p_avg = float(np.mean(recent_penalties))
                total_penalty += p_avg
                max_step_penalty = max(max_step_penalty, p_avg)

            if penalty_threshold is not None and max_step_penalty > penalty_threshold:
                # Reject "fantasy" trajectories that go far off-manifold
                return -np.inf, total_reward, total_penalty

    objective = total_reward - lam * total_penalty
    return objective, total_reward, total_penalty


def cem_planner(model, h0, z0,
                horizon=25,
                num_candidates=500,
                num_elites=50,
                num_iterations=5,
                action_dim=1,
                action_low=-2.0,
                action_high=2.0,
                lam=0.0,
                geometry_alpha=0.5,
                geometry_metric="trace",
                geometry_window=1,
                penalty_threshold=None,
                seed=0):
    """
    Cross-Entropy Method (CEM) planner for better optimization.
    """
    rng = np.random.default_rng(seed)
    
    # Initialize mean and std for action sequence
    mean = np.zeros((horizon, action_dim))
    std = np.ones((horizon, action_dim)) * (action_high - action_low) / 2.0
    
    best_obj_overall = -np.inf
    best_seq_overall = None
    best_stats_overall = None
    
    for iteration in range(num_iterations):
        # Sample candidates from current distribution
        candidates = []
        objectives = []
        stats_list = []
        
        for _ in range(num_candidates):
            action_seq = rng.normal(mean, std)
            action_seq = np.clip(action_seq, action_low, action_high)
            
            obj, rew, pen = rollout_sequence_in_model(
                model, h0, z0, action_seq,
                lam=lam,
                geometry_alpha=geometry_alpha,
                geometry_metric=geometry_metric,
                geometry_window=geometry_window,
                penalty_threshold=penalty_threshold,
            )
            
            candidates.append(action_seq)
            objectives.append(obj)
            stats_list.append((rew, pen))
        
        # Select elites
        objectives = np.array(objectives)
        elite_indices = np.argsort(objectives)[-num_elites:]
        elite_seqs = [candidates[i] for i in elite_indices]
        
        # Update mean and std from elites
        elite_stack = np.stack(elite_seqs, axis=0)  # (num_elites, horizon, action_dim)
        mean = np.mean(elite_stack, axis=0)
        std = np.std(elite_stack, axis=0) + 0.01  # small epsilon to prevent collapse
        
        # Track best
        best_idx = elite_indices[-1]
        if objectives[best_idx] > best_obj_overall:
            best_obj_overall = objectives[best_idx]
            best_seq_overall = candidates[best_idx]
            best_stats_overall = stats_list[best_idx]
    
    return best_seq_overall, best_obj_overall, best_stats_overall


def random_shooting_planner(model, h0, z0,
                            horizon=30,
                            num_candidates=1024,
                            action_dim=1,
                            action_low=-2.0,
                            action_high=2.0,
                            lam=0.0,
                            geometry_alpha=0.5,
                            geometry_metric="trace",
                            geometry_window=1,
                            penalty_threshold=None,
                            seed=0,
                            debug_penalty=False):
    """
    Simple random shooting in latent space.
    """
    rng = np.random.default_rng(seed)
    best_obj = -np.inf
    best_seq = None
    best_stats = None
    
    all_penalties = []
    all_rewards = []

    for i in range(num_candidates):
        action_seq = rng.uniform(
            low=action_low,
            high=action_high,
            size=(horizon, action_dim),
        )

        obj, rew, pen = rollout_sequence_in_model(
            model, h0, z0, action_seq,
            lam=lam,
            geometry_alpha=geometry_alpha,
            geometry_metric=geometry_metric,
            geometry_window=geometry_window,
            penalty_threshold=penalty_threshold,
        )
        
        all_rewards.append(rew)
        if lam > 0.0:
            all_penalties.append(pen)
        
        if obj > best_obj:
            best_obj = obj
            best_seq = action_seq
            best_stats = (rew, pen)
    
    if debug_penalty and lam > 0.0 and len(all_penalties) > 0:
        print(f"  Penalty distribution: min={min(all_penalties):.2f}, "
              f"max={max(all_penalties):.2f}, mean={np.mean(all_penalties):.2f}, "
              f"std={np.std(all_penalties):.2f}")
        print(f"  Reward distribution: min={min(all_rewards):.2f}, "
              f"max={max(all_rewards):.2f}, mean={np.mean(all_rewards):.2f}")

    return best_seq, best_obj, best_stats


def eval_open_loop_in_env(env, obs0_lowdim, action_seq):
    """
    Execute an action sequence open-loop in the real environment.
    """
    env.set_state(obs0_lowdim)

    total_reward = 0.0
    for a in action_seq:
        _, r, done, _ = env.step(a)
        total_reward += r
        if done:
            break

    return total_reward


# ============================================================
# 6. Main experiment
# ============================================================

def run_planning_comparison(model, env, s0, h0, z0, 
                            horizon, num_candidates, lam, seed_base,
                            use_cem=False, debug=False,
                            geometry_alpha=0.5,
                            geometry_metric="trace",
                            geometry_window=1):
    """
    Run planning with and without geometry penalty, return results.
    """
    planner = cem_planner if use_cem else random_shooting_planner
    planner_kwargs = dict(
        horizon=horizon,
        action_dim=env.action_dim,
        action_low=-env.max_torque,
        action_high=env.max_torque,
    )
    
    if use_cem:
        planner_kwargs['num_candidates'] = num_candidates
        planner_kwargs['num_elites'] = max(10, num_candidates // 10)
        planner_kwargs['num_iterations'] = 5
    else:
        planner_kwargs['num_candidates'] = num_candidates
        planner_kwargs['debug_penalty'] = debug
    
    # Without geometry
    seq_no_geom, obj_no_geom, (rew_no_geom, pen_no_geom) = planner(
        model, h0, z0,
        lam=0.0,
        geometry_alpha=geometry_alpha,
        geometry_metric=geometry_metric,
        geometry_window=geometry_window,
        seed=seed_base,
        **planner_kwargs,
    )
    real_return_no_geom = eval_open_loop_in_env(env, s0, seq_no_geom)
    
    # With geometry
    seq_geom, obj_geom, (rew_geom, pen_geom) = planner(
        model, h0, z0,
        lam=lam,
        geometry_alpha=geometry_alpha,
        geometry_metric=geometry_metric,
        geometry_window=geometry_window,
        seed=seed_base + 1000,  # Different seed to allow different selection
        **planner_kwargs,
    )
    real_return_geom = eval_open_loop_in_env(env, s0, seq_geom)
    
    return {
        'no_geom': {
            'model_obj': obj_no_geom,
            'model_rew': rew_no_geom,
            'real_return': real_return_no_geom,
        },
        'with_geom': {
            'model_obj': obj_geom,
            'model_rew': rew_geom,
            'model_pen': pen_geom,
            'real_return': real_return_geom,
        }
    }


def run_variant(name, bisim_beta, obs, actions, rewards,
                planner_cfg, vis_horizon=40):
    """
    Train a model with given bisim_beta, run one planning rollout, and visualize latent trajectories.
    """
    np.random.seed(planner_cfg.get("seed", 0))
    torch.manual_seed(planner_cfg.get("seed", 0))

    env_eval = PixelPendulumEnv(size=64)

    model = SimplePlaNetRSSM(obs_dim=env_eval.obs_dim,
                             action_dim=env_eval.action_dim,
                             latent_dim=16,
                             hidden_dim=128,
                             num_decoders=3)
    train_world_model(model, obs, actions, rewards,
                      num_epochs=25, batch_size=32,
                      learning_rate=1e-3, beta_kl=0.1, rew_scale=1.0,
                      bisim_beta=bisim_beta, bisim_gamma=0.99)

    s0_pixels = env_eval.reset()
    s0_lowdim = env_eval.last_state_lowdim
    h0, z0 = model.encode_observation(s0_pixels)

    seq, obj, (rew_model, pen_model) = random_shooting_planner(
        model, h0, z0,
        horizon=planner_cfg["horizon"],
        num_candidates=planner_cfg["num_candidates"],
        action_dim=env_eval.action_dim,
        action_low=-env_eval.max_torque,
        action_high=env_eval.max_torque,
        lam=planner_cfg["lam"],
        geometry_alpha=planner_cfg["geometry_alpha"],
        geometry_metric=planner_cfg["geometry_metric"],
        seed=planner_cfg.get("seed", 0),
        debug_penalty=True,
    )
    real_return = eval_open_loop_in_env(env_eval, s0_lowdim, seq)

    # Visualization sequence: use planned seq (truncated to vis_horizon)
    seq_vis = seq[:vis_horizon]
    plot_file = visualize_latent_rollout(
        model, env_eval, obs, actions,
        seq_vis,
        filename=f"latent_rollout_{name}.png",
        title=f"{name} (bisim_beta={bisim_beta})",
        start_pixels=s0_pixels,
        start_state_lowdim=s0_lowdim,
    )

    return {
        "name": name,
        "bisim_beta": bisim_beta,
        "model_obj": obj,
        "model_rew": rew_model,
        "model_pen": pen_model,
        "real_return": real_return,
        "plot_file": plot_file,
    }


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Gymnasium Pendulum Pixels: Bisim vs Non-bisim + Latent Viz")
    print("=" * 60)

    env_data = PixelPendulumEnv(size=64)

    # 1) Collect offline data once
    print("\n[1] Collecting dataset with random policy...")
    n_episodes = 100
    T = 40
    obs, actions, rewards = collect_dataset(env_data, n_episodes=n_episodes, T=T, policy='random')
    print(f"    Dataset: {n_episodes} episodes × {T} steps")

    planner_cfg = {
        "horizon": 50,
        "num_candidates": 1200,
        "lam": 0.15,
        "geometry_alpha": 0.6,
        "geometry_metric": "trace",
        "geometry_window": 3,
        "seed": 123,
    }

    # 2) Run two variants: no bisim vs bisim
    variants = [
        ("no_bisim", 0.0),
        ("bisim", 0.05),
    ]
    results = []
    for name, beta in variants:
        print(f"\n[Variant: {name}] Training with bisim_beta={beta}")
        res = run_variant(name, beta, obs, actions, rewards,
                          planner_cfg=planner_cfg, vis_horizon=40)
        results.append(res)
        print(f"    Model obj={res['model_obj']:.2f}, model_rew={res['model_rew']:.2f}, "
              f"penalty={res['model_pen']:.2f}, real_return={res['real_return']:.2f}")
        print(f"    Saved latent plot: {res['plot_file']}")

    # Summary
    print("\n" + "=" * 60)
    print("Single-run summary (bisim vs no-bisim)")
    print("=" * 60)
    for res in results:
        print(f"  {res['name']}: real_return={res['real_return']:.2f}, "
              f"model_rew={res['model_rew']:.2f}, plot={res['plot_file']}")


if __name__ == "__main__":
    main()
