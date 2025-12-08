#!/usr/bin/env python3
"""
PlaNet-style RSSM + Pendulum environment + geometry-aware planning.

Pipeline:
1) Collect offline data with random policy in the Pendulum environment.
2) Train a minimal PlaNet-like RSSM world model on that data.
3) Encode training sequences into latent space; build a kNN graph.
4) During planning:
   - For each candidate action sequence, simulate latent trajectory
     using the learned prior (imagination).
   - Add penalty proportional to "deviation from data manifold" in latent
     space (measured via the kNN graph).
   - Compare planning with/without geometry penalty.

The Pendulum environment is nonlinear (sin/cos dynamics) and harder to model,
making it ideal for testing geometry-aware planning that prevents model exploitation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 0. Pendulum Environment (Classic Control)
# ============================================================

class PendulumEnv:
    """
    Classic inverted pendulum (swing-up task).
    
    State: [cos(θ), sin(θ), θ_dot]
    Action: torque in [-2, 2]
    
    Goal: Swing up and balance at the top (θ=0, pointing up).
    
    Dynamics are nonlinear due to sin(θ) term, making this harder
    to model accurately than simple linear systems.
    """

    def __init__(self, dt=0.05, max_speed=8.0, max_torque=2.0):
        self.dt = dt
        self.max_speed = max_speed
        self.max_torque = max_torque
        self.g = 10.0  # gravity
        self.m = 1.0   # mass
        self.l = 1.0   # length
        
        self.obs_dim = 3   # [cos(θ), sin(θ), θ_dot]
        self.action_dim = 1
        self.reset()

    def reset(self, theta=None, theta_dot=None):
        """Reset to random state (typically hanging down with some noise)."""
        if theta is None:
            # Start near bottom (hanging down = π) with small perturbation
            self.theta = np.random.uniform(np.pi - 0.5, np.pi + 0.5)
        else:
            self.theta = theta
        
        if theta_dot is None:
            self.theta_dot = np.random.uniform(-0.5, 0.5)
        else:
            self.theta_dot = theta_dot
            
        self.t = 0
        return self._get_obs()

    def _get_obs(self):
        """Observation: [cos(θ), sin(θ), θ_dot]"""
        return np.array([
            np.cos(self.theta),
            np.sin(self.theta),
            self.theta_dot
        ], dtype=np.float32)

    def _angle_normalize(self, x):
        """Normalize angle to [-π, π]"""
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def step(self, action):
        """
        Apply torque and simulate one timestep.
        
        Dynamics: θ'' = (3g/2l) * sin(θ) + (3/ml²) * u
        """
        u = float(np.clip(action[0], -self.max_torque, self.max_torque))
        
        theta = self.theta
        theta_dot = self.theta_dot
        
        # Pendulum dynamics
        # θ'' = (3g/2l) * sin(θ) + (3/ml²) * u
        theta_acc = (3.0 * self.g / (2.0 * self.l)) * np.sin(theta) + \
                    (3.0 / (self.m * self.l ** 2)) * u
        
        # Euler integration
        theta_dot_new = theta_dot + theta_acc * self.dt
        theta_dot_new = np.clip(theta_dot_new, -self.max_speed, self.max_speed)
        theta_new = theta + theta_dot_new * self.dt
        
        self.theta = theta_new
        self.theta_dot = theta_dot_new
        self.t += 1
        
        # Reward: penalize angle from upright, angular velocity, and torque
        # θ=0 is upright, θ=π is hanging down
        angle_cost = self._angle_normalize(self.theta) ** 2
        velocity_cost = 0.1 * (self.theta_dot ** 2)
        torque_cost = 0.001 * (u ** 2)
        
        reward = -(angle_cost + velocity_cost + torque_cost)
        
        done = self.t >= 200
        
        return self._get_obs(), float(reward), done, {}
    
    def set_state(self, obs):
        """Set state from observation [cos(θ), sin(θ), θ_dot]"""
        cos_theta, sin_theta, theta_dot = obs[0], obs[1], obs[2]
        self.theta = np.arctan2(sin_theta, cos_theta)
        self.theta_dot = theta_dot
        self.t = 0


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
                 latent_dim=16, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Deterministic part
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        # Prior p(z_t | h_t)
        self.prior_net = mlp(hidden_dim, 2 * latent_dim, hidden_dim=128)

        # Posterior q(z_t | h_t, o_t)
        self.post_net = mlp(hidden_dim + obs_dim, 2 * latent_dim, hidden_dim=128)

        # Observation decoder p(o_t | h_t, z_t) ~ N(μ, I)
        self.dec_obs = mlp(hidden_dim + latent_dim, obs_dim, hidden_dim=128)

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

    def decode_obs(self, h, z):
        return self.dec_obs(torch.cat([h, z], dim=-1))

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
            else:
                a = np.random.uniform(-env.max_torque, env.max_torque, size=(act_dim,))
            
            actions[ep, t] = a
            s, r, done, _ = env.step(a)
            obs[ep, t + 1] = s
            rewards[ep, t] = r

    return obs, actions, rewards


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
                      rew_scale=1.0):
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

            for t in range(T):
                o_t = batch_obs[:, t, :]                 # (B, obs_dim)
                r_t = batch_rew[:, t].unsqueeze(-1)      # (B, 1)
                a_t = batch_act[:, t, :]                 # (B, act_dim)

                # Prior & Posterior
                prior_mean, prior_log_std = model.prior(h)
                post_mean, post_log_std = model.posterior(h, o_t)

                # Sample latent from posterior
                z_post = model.sample(post_mean, post_log_std)

                # Decode observation & reward
                o_hat = model.decode_obs(h, z_post)
                r_hat = model.decode_rew(h, z_post)

                recon_loss += ((o_hat - o_t) ** 2).mean()
                rew_loss += ((r_hat - r_t) ** 2).mean()

                # KL q(z|h,o) || p(z|h)
                kl = kl_gaussian(post_mean, post_log_std,
                                 prior_mean, prior_log_std).mean()
                kl_loss += kl

                # Update deterministic state
                h = model.gru_step(h, z_post, a_prev)
                a_prev = a_t

            # Normalize losses by T
            recon_loss = recon_loss / T
            rew_loss = rew_loss / T
            kl_loss = kl_loss / T

            loss = recon_loss + rew_scale * rew_loss + beta_kl * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss / max(1, n_batches):.4f}")

    model.eval()


# ============================================================
# 4. Latent graph and manifold penalty
# ============================================================

def build_latent_graph_from_latents(Z, k_neighbors=15):
    """
    Z: (N, latent_dim) array of latent codes from training data.

    Build a kNN graph and a simple density proxy (local_scale).
    """
    N, d = Z.shape
    # pairwise distance matrix (O(N^2), fine for small N)
    diff = Z[:, None, :] - Z[None, :, :]  # (N, N, d)
    dist_matrix = np.linalg.norm(diff, axis=-1)  # (N, N)
    np.fill_diagonal(dist_matrix, np.inf)

    knn_indices = np.argsort(dist_matrix, axis=1)[:, :k_neighbors]
    knn_dists = np.take_along_axis(dist_matrix, knn_indices, axis=1)

    local_scale = knn_dists.mean(axis=1)  # (N,)

    graph = {
        "Z": Z,
        "knn_indices": knn_indices,
        "knn_dists": knn_dists,
        "local_scale": local_scale,
        "global_scale": np.median(local_scale),  # for normalization
    }
    return graph


def encode_dataset_latents(model, obs, actions):
    """
    Run the posterior over the dataset and collect latent means z_t.

    obs: (N, T+1, obs_dim)
    actions: (N, T, act_dim)

    Returns:
        Z_all: (N*T, latent_dim) flattened array of means z_t
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
                o_t = obs_torch[ep, t:t+1, :]   # (1, obs_dim)
                a_t = act_torch[ep, t:t+1, :]   # (1, act_dim)

                post_mean, post_log_std = model.posterior(h, o_t)
                z_mean = post_mean              # don't sample; take mean

                latents.append(z_mean.squeeze(0).cpu().numpy())

                # update deterministic state
                h = model.gru_step(h, z_mean, a_prev)
                a_prev = a_t

    Z_all = np.stack(latents, axis=0)  # (N*T, latent_dim)
    return Z_all


def manifold_penalty(z, graph, k_query=5, eps=1e-8):
    """
    Simple off-manifold penalty:

    - Find k_query nearest latents Z_i in Euclidean distance.
    - Compare distance to local_scale[i] (mean neighbor distance of node i).
    - Average the normalized distances as penalty.

    z: (latent_dim,) numpy array
    """
    Z = graph["Z"]
    local_scale = graph["local_scale"]

    dists = np.linalg.norm(Z - z[None, :], axis=1)
    idx_sorted = np.argsort(dists)
    neighbors = idx_sorted[:k_query]

    penalties = []
    for i in neighbors:
        d_euc = dists[i]
        scale = local_scale[i] + eps
        penalties.append(d_euc / scale)

    return float(np.mean(penalties))


# ============================================================
# 5. Planning with geometry penalty (CEM + random shooting)
# ============================================================

def rollout_sequence_in_model(model, h0, z0, action_seq,
                              graph=None, lam=0.0,
                              penalty_threshold=None):
    """
    Roll out one action sequence in the learned model's latent space.

    model: SimplePlaNetRSSM
    h0, z0: initial hidden & latent states (torch tensors, shape (1, *))
    action_seq: (H, action_dim) numpy array
    graph: latent graph (or None)
    lam: penalty weight
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

    model.eval()
    with torch.no_grad():
        for a_np in action_seq:
            a = torch.from_numpy(a_np).float().unsqueeze(0).to(device)  # (1, act_dim)
            # Imagine next latent
            h, z = model.imagine_step(h, z, a)
            # Predicted reward
            r_hat = model.decode_rew(h, z)
            total_reward += float(r_hat.item())

            if graph is not None:
                z_np = z.squeeze(0).cpu().numpy()
                p = manifold_penalty(z_np, graph)
                total_penalty += p
                max_step_penalty = max(max_step_penalty, p)

            if penalty_threshold is not None and max_step_penalty > penalty_threshold:
                # Reject "fantasy" trajectories that go far off-manifold
                return -np.inf, total_reward, total_penalty

    objective = total_reward - lam * total_penalty
    return objective, total_reward, total_penalty


def cem_planner(model, h0, z0,
                graph=None,
                horizon=25,
                num_candidates=500,
                num_elites=50,
                num_iterations=5,
                action_dim=1,
                action_low=-2.0,
                action_high=2.0,
                lam=0.0,
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
                graph=graph,
                lam=lam,
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
                            graph=None,
                            horizon=30,
                            num_candidates=1024,
                            action_dim=1,
                            action_low=-2.0,
                            action_high=2.0,
                            lam=0.0,
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
            graph=graph,
            lam=lam,
            penalty_threshold=penalty_threshold,
        )
        
        all_rewards.append(rew)
        if graph is not None:
            all_penalties.append(pen)
        
        if obj > best_obj:
            best_obj = obj
            best_seq = action_seq
            best_stats = (rew, pen)
    
    if debug_penalty and graph is not None and len(all_penalties) > 0:
        print(f"  Penalty distribution: min={min(all_penalties):.2f}, "
              f"max={max(all_penalties):.2f}, mean={np.mean(all_penalties):.2f}, "
              f"std={np.std(all_penalties):.2f}")
        print(f"  Reward distribution: min={min(all_rewards):.2f}, "
              f"max={max(all_rewards):.2f}, mean={np.mean(all_rewards):.2f}")

    return best_seq, best_obj, best_stats


def eval_open_loop_in_env(env, obs0, action_seq):
    """
    Execute an action sequence open-loop in the real environment.
    """
    env.set_state(obs0)

    total_reward = 0.0
    for a in action_seq:
        s, r, done, _ = env.step(a)
        total_reward += r
        if done:
            break

    return total_reward


# ============================================================
# 6. Main experiment
# ============================================================

def run_planning_comparison(model, env, graph, s0, h0, z0, 
                            horizon, num_candidates, lam, seed_base,
                            use_cem=False, debug=False):
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
        graph=None,
        lam=0.0,
        seed=seed_base,
        **planner_kwargs,
    )
    real_return_no_geom = eval_open_loop_in_env(env, s0, seq_no_geom)
    
    # With geometry
    seq_geom, obj_geom, (rew_geom, pen_geom) = planner(
        model, h0, z0,
        graph=graph,
        lam=lam,
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


def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Pendulum Environment: Geometry-Aware Planning Experiment")
    print("=" * 60)
    
    env = PendulumEnv(dt=0.05)

    # 1) Collect offline data (limited to create model uncertainty)
    print("\n[1] Collecting dataset with random policy...")
    n_episodes = 100  # Limited data to create model uncertainty
    T = 40
    obs, actions, rewards = collect_dataset(env, n_episodes=n_episodes, T=T, policy='random')
    print(f"    Dataset: {n_episodes} episodes × {T} steps")

    # 2) Train PlaNet-style RSSM
    print("\n[2] Training world model...")
    model = SimplePlaNetRSSM(obs_dim=env.obs_dim,
                             action_dim=env.action_dim,
                             latent_dim=16,
                             hidden_dim=128)
    train_world_model(model, obs, actions, rewards,
                      num_epochs=30, batch_size=32,
                      learning_rate=1e-3, beta_kl=0.1, rew_scale=1.0)

    # 3) Encode dataset into latent space and build graph
    print("\n[3] Building latent manifold graph...")
    Z_all = encode_dataset_latents(model, obs, actions)
    graph = build_latent_graph_from_latents(Z_all, k_neighbors=15)
    print(f"    Latent points: {Z_all.shape[0]}, global scale: {graph['global_scale']:.3f}")

    # 4) Multi-trial evaluation
    print("\n[4] Running planning comparison (multiple trials)...")
    
    num_trials = 10
    horizon = 30
    num_candidates = 1000
    lam = 0.3  # Geometry penalty weight
    
    results_no_geom = []
    results_with_geom = []
    
    for trial in range(num_trials):
        # Random initial state (hanging down with perturbation)
        s0 = env.reset()
        h0, z0 = model.encode_observation(s0)
        
        results = run_planning_comparison(
            model, env, graph, s0, h0, z0,
            horizon=horizon,
            num_candidates=num_candidates,
            lam=lam,
            seed_base=trial * 100,
            use_cem=False,
            debug=(trial == 0)  # Debug first trial only
        )
        
        results_no_geom.append(results['no_geom']['real_return'])
        results_with_geom.append(results['with_geom']['real_return'])
        
        if trial == 0:
            print(f"\n    Trial {trial+1} details:")
            print(f"      No geometry:   model_rew={results['no_geom']['model_rew']:.2f}, "
                  f"real={results['no_geom']['real_return']:.2f}")
            print(f"      With geometry: model_rew={results['with_geom']['model_rew']:.2f}, "
                  f"penalty={results['with_geom']['model_pen']:.2f}, "
                  f"real={results['with_geom']['real_return']:.2f}")
    
    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nSettings: horizon={horizon}, candidates={num_candidates}, λ={lam}")
    print(f"Trials: {num_trials}")
    
    mean_no_geom = np.mean(results_no_geom)
    std_no_geom = np.std(results_no_geom)
    mean_with_geom = np.mean(results_with_geom)
    std_with_geom = np.std(results_with_geom)
    
    print(f"\nReal Environment Returns:")
    print(f"  Without geometry: {mean_no_geom:.2f} ± {std_no_geom:.2f}")
    print(f"  With geometry:    {mean_with_geom:.2f} ± {std_with_geom:.2f}")
    
    # Model exploitation metric
    exploitation_gap = mean_with_geom - mean_no_geom
    print(f"\nGeometry benefit: {exploitation_gap:+.2f}")
    
    if exploitation_gap > 0:
        print("  → Geometry penalty HELPS (prevents model exploitation)")
    else:
        print("  → Geometry penalty does not help (model is accurate enough)")
    
    # Additional analysis: model vs reality gap
    print("\n" + "-" * 40)
    print("Per-trial breakdown:")
    print("-" * 40)
    for i, (no_g, with_g) in enumerate(zip(results_no_geom, results_with_geom)):
        diff = with_g - no_g
        marker = "✓" if diff > 0 else "✗"
        print(f"  Trial {i+1}: no_geom={no_g:.2f}, with_geom={with_g:.2f}, diff={diff:+.2f} {marker}")


if __name__ == "__main__":
    main()
