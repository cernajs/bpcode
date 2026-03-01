#!/usr/bin/env python3
"""
Comprehensive latent-geometry evaluation for Dreamer world models on maze envs.

Pipeline per maze layout:
  1. Train vanilla Dreamer world model  (encoder + RSSM + decoder + reward + actor-critic)
  2. Collect (position, h, s) tuples from trained model rollouts
  3. Post-hoc train GeoEncoder g(h,s) on frozen representations
  4. Probe analysis      — linear / MLP probes  position decoding  R²
  5. Distance analysis   — Pearson / Spearman of latent dist vs geodesic dist
  6. Neighbourhood analysis — kNN overlap between latent and geodesic spaces
  7. Visualizations      — PCA embeddings, scatter plots, summary comparison

Usage:
    python maze_geometry_test.py                          # all 5 layouts, default settings
    python maze_geometry_test.py --mazes corridor loop    # just those two
    python maze_geometry_test.py --quick                  # fast sanity-check run
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from maze_env import MAZE_LAYOUTS, PointMazeEnv, generate_random_maze
from models import (
    RSSM,
    Actor,
    ContinueModel,
    ConvDecoder,
    ConvEncoder,
    RewardModel,
    ValueModel,
)
from geom_head import GeoEncoder, temporal_reachability_loss
from utils import ReplayBuffer, bottle, get_device, preprocess_img, set_seed

# ===================================================================
#  Config
# ===================================================================

@dataclass
class TrainCfg:
    seed_episodes: int = 5
    max_episodes: int = 100
    collect_interval: int = 50
    train_steps: int = 30
    batch_size: int = 32
    seq_len: int = 30
    img_size: int = 64
    bit_depth: int = 5
    embed_dim: int = 1024
    stoch_dim: int = 30
    deter_dim: int = 200
    hidden_dim: int = 200
    actor_hidden_dim: int = 400
    value_hidden_dim: int = 400
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    value_lr: float = 8e-5
    adam_eps: float = 1e-5
    grad_clip: float = 100.0
    gamma: float = 0.99
    lambda_: float = 0.95
    kl_weight: float = 1.0
    kl_free_nats: float = 3.0
    imagination_horizon: int = 15
    actor_entropy_scale: float = 3e-3
    expl_amount: float = 0.3
    replay_capacity: int = 100_000
    action_repeat: int = 1
    geo_dim: int = 32
    geo_hidden: int = 256
    geo_lr: float = 3e-4
    geo_epochs: int = 300
    geo_batch: int = 64
    geo_window: int = 30
    geo_pos_k: int = 3
    geo_neg_k: int = 8
    geo_margin: float = 0.6
    collect_episodes: int = 60
    n_probe_epochs: int = 1500
    n_pairs: int = 8000
    knn_k: int = 10


# ===================================================================
#  1. Training
# ===================================================================

def _compute_lambda_returns(rewards, values, discounts, lambda_=0.95):
    B, H = rewards.shape
    last = values[:, -1]
    out = torch.zeros_like(rewards)
    nv = values[:, 1:]
    for t in reversed(range(H)):
        bootstrap = (1.0 - lambda_) * nv[:, t] + lambda_ * last
        last = rewards[:, t] + discounts[:, t] * bootstrap
        out[:, t] = last
    return out


def train_world_model(env: PointMazeEnv, cfg: TrainCfg, device: torch.device):
    """Train full DreamerV1 on *env* and return model dict."""
    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    encoder = ConvEncoder(embedding_size=cfg.embed_dim, in_channels=C).to(device)
    decoder = ConvDecoder(cfg.deter_dim, cfg.stoch_dim, embedding_size=cfg.embed_dim, out_channels=C).to(device)
    rssm = RSSM(cfg.stoch_dim, cfg.deter_dim, act_dim, cfg.embed_dim, cfg.hidden_dim).to(device)
    reward_model = RewardModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device)
    cont_model = ContinueModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device)
    actor = Actor(cfg.deter_dim, cfg.stoch_dim, act_dim, cfg.actor_hidden_dim).to(device)
    value_model = ValueModel(cfg.deter_dim, cfg.stoch_dim, cfg.value_hidden_dim).to(device)

    world_params = (
        list(encoder.parameters()) + list(decoder.parameters()) +
        list(rssm.parameters()) + list(reward_model.parameters()) +
        list(cont_model.parameters())
    )
    model_opt = torch.optim.Adam(world_params, lr=cfg.model_lr, eps=cfg.adam_eps)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.adam_eps)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=cfg.value_lr, eps=cfg.adam_eps)

    replay = ReplayBuffer(cfg.replay_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    free_nats = torch.ones(1, device=device) * cfg.kl_free_nats

    # Seed buffer
    for _ in range(cfg.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            a = env.action_space.sample()
            nobs, r, t, tr, _ = env.step(a, repeat=cfg.action_repeat)
            done = bool(t or tr)
            replay.add(np.ascontiguousarray(obs, np.uint8), a.astype(np.float32), r, None, done)
            obs = nobs

    total_steps = 0
    t0 = time.time()

    for ep in range(cfg.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=cfg.bit_depth)
        with torch.no_grad():
            e0 = encoder(obs_t)
            h, s = rssm.get_init_state(e0)

        while not done:
            encoder.eval(); rssm.eval(); actor.eval()
            with torch.no_grad():
                a_t, _ = actor.get_action(h, s, deterministic=False)
                if cfg.expl_amount > 0:
                    a_t = torch.clamp(a_t + cfg.expl_amount * torch.randn_like(a_t), -1, 1)
                a_np = a_t.squeeze(0).cpu().numpy().astype(np.float32)

            nobs, r, t, tr, _ = env.step(a_np, repeat=cfg.action_repeat)
            done = bool(t or tr)
            ep_ret += float(r)
            replay.add(np.ascontiguousarray(obs, np.uint8), a_np, r, None, done)
            obs = nobs
            total_steps += 1

            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=cfg.bit_depth)
            with torch.no_grad():
                e = encoder(obs_t)
                act_t = torch.tensor(a_np, dtype=torch.float32, device=device).unsqueeze(0)
                h, s, _, _ = rssm.observe_step(e, act_t, h, s, sample=False)

            if total_steps % cfg.collect_interval == 0 and replay.size > (cfg.seq_len + 2):
                encoder.train(); decoder.train(); rssm.train()
                reward_model.train(); cont_model.train(); actor.train(); value_model.train()

                for _ in range(cfg.train_steps):
                    batch = replay.sample_sequences(cfg.batch_size, cfg.seq_len + 1)
                    obs_seq = torch.as_tensor(batch.obs, device=device).float()
                    act_seq = torch.as_tensor(batch.actions, device=device)
                    rew_seq = torch.as_tensor(batch.rews, device=device)
                    done_seq = torch.as_tensor(batch.dones, device=device)
                    B, T1 = rew_seq.shape
                    T = T1 - 1

                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                    preprocess_img(x, depth=cfg.bit_depth)
                    e_t = bottle(encoder, x)

                    h_t, s_t = rssm.get_init_state(e_t[:, 0])
                    h_list, s_list, pri_list, post_list = [], [], [], []
                    for ti in range(T):
                        h_t = rssm.deterministic_state_fwd(h_t, s_t, act_seq[:, ti])
                        pri_list.append(rssm.state_prior(h_t))
                        post_list.append(rssm.state_posterior(h_t, e_t[:, ti + 1]))
                        pm, ps = post_list[-1]
                        s_t = pm + torch.randn_like(ps) * ps
                        h_list.append(h_t)
                        s_list.append(s_t)

                    from torch.distributions import Normal
                    from torch.distributions.kl import kl_divergence

                    h_seq = torch.stack(h_list, 1)
                    s_seq = torch.stack(s_list, 1)
                    pri_m = torch.stack([p[0] for p in pri_list], 0)
                    pri_s = torch.stack([p[1] for p in pri_list], 0)
                    pos_m = torch.stack([p[0] for p in post_list], 0)
                    pos_s = torch.stack([p[1] for p in post_list], 0)

                    kl_lhs = kl_divergence(Normal(pos_m.detach(), pos_s.detach()), Normal(pri_m, pri_s))
                    kl_rhs = kl_divergence(Normal(pos_m, pos_s), Normal(pri_m.detach(), pri_s.detach()))
                    kld = torch.max((0.5 * (kl_lhs + kl_rhs)).sum(-1), free_nats).mean()

                    recon = bottle(decoder, h_seq, s_seq)
                    rec_loss = F.mse_loss(recon, x[:, 1:T + 1], reduction="none").sum((2, 3, 4)).mean()
                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_loss = F.mse_loss(rew_pred, rew_seq[:, :T])
                    cont_logits = bottle(cont_model, h_seq, s_seq)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, (1.0 - done_seq[:, :T]).clamp(0, 1))

                    model_loss = rec_loss + cfg.kl_weight * kld + rew_loss + cont_loss
                    model_opt.zero_grad(set_to_none=True)
                    model_loss.backward()
                    nn.utils.clip_grad_norm_(world_params, cfg.grad_clip)
                    model_opt.step()

                    # ---- actor-critic (imagination) ----
                    with torch.no_grad():
                        Dh, Ds = h_seq.size(-1), s_seq.size(-1)
                        h0 = h_seq.reshape(-1, Dh).detach()
                        s0 = s_seq.reshape(-1, Ds).detach()

                    h_im, s_im = h0, s0
                    h_im_l, s_im_l = [h0], [s0]
                    for _ in range(cfg.imagination_horizon):
                        a_im, _ = actor.get_action(h_im, s_im, deterministic=False)
                        h_im = rssm.deterministic_state_fwd(h_im, s_im, a_im)
                        s_im = rssm.state_prior(h_im, sample=True)
                        h_im_l.append(h_im); s_im_l.append(s_im)
                    h_imag = torch.stack(h_im_l, 1)
                    s_imag = torch.stack(s_im_l, 1)
                    r_im = bottle(reward_model, h_imag[:, 1:], s_imag[:, 1:])
                    cl_im = bottle(cont_model, h_imag[:, 1:], s_imag[:, 1:])
                    disc = cfg.gamma * torch.sigmoid(cl_im).clamp(0, 1)

                    with torch.no_grad():
                        v_tgt = bottle(value_model, h_imag, s_imag)
                        lam_tgt = _compute_lambda_returns(r_im.detach(), v_tgt, disc.detach(), cfg.lambda_)
                        w_tgt = torch.cumprod(torch.cat([torch.ones(h0.size(0), 1, device=device), disc.detach()], 1), 1)[:, :-1]

                    v_pred = bottle(value_model, h_imag.detach(), s_imag.detach())
                    value_loss = ((v_pred[:, :-1] - lam_tgt) ** 2 * w_tgt).mean()
                    value_opt.zero_grad(set_to_none=True)
                    value_loss.backward()
                    nn.utils.clip_grad_norm_(value_model.parameters(), cfg.grad_clip)
                    value_opt.step()

                    mean, std = actor.forward(h_imag[:, :-1].detach(), s_imag[:, :-1].detach())
                    from torch.distributions import Normal as Norm
                    entropy = Norm(mean, std).entropy().sum(-1).mean()
                    v_for_act = bottle(value_model, h_imag, s_imag)
                    lam_act = _compute_lambda_returns(r_im, v_for_act, disc, cfg.lambda_)
                    w_act = torch.cumprod(torch.cat([torch.ones(h0.size(0), 1, device=device), disc.detach()], 1), 1)[:, :-1]
                    actor_loss = -(w_act.detach() * lam_act).mean() - cfg.actor_entropy_scale * entropy
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), cfg.grad_clip)
                    actor_opt.step()

        if (ep + 1) % max(1, cfg.max_episodes // 5) == 0:
            print(f"    ep {ep+1}/{cfg.max_episodes}  steps={total_steps}  ret={ep_ret:.2f}  ({time.time()-t0:.0f}s)")

    models = dict(encoder=encoder, decoder=decoder, rssm=rssm,
                  reward_model=reward_model, cont_model=cont_model,
                  actor=actor, value_model=value_model)
    for m in models.values():
        m.eval()
    return models


# ===================================================================
#  2. Data collection
# ===================================================================

@torch.no_grad()
def collect_data(env: PointMazeEnv, models: dict, cfg: TrainCfg, device: torch.device):
    """Run trained model, collect (pos, h, s) per step, grouped by episode."""
    encoder, rssm, actor = models["encoder"], models["rssm"], models["actor"]
    encoder.eval(); rssm.eval(); actor.eval()

    all_pos, all_h, all_s = [], [], []
    traj_pos, traj_h, traj_s = [], [], []

    for _ in range(cfg.collect_episodes):
        obs, _ = env.reset()
        done = False
        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=cfg.bit_depth)
        e = encoder(obs_t)
        h, s = rssm.get_init_state(e)

        ep_pos, ep_h, ep_s = [env.agent_pos.copy()], [h.squeeze(0).cpu().numpy()], [s.squeeze(0).cpu().numpy()]

        while not done:
            a_t, _ = actor.get_action(h, s, deterministic=False)
            a_t = torch.clamp(a_t + 0.1 * torch.randn_like(a_t), -1, 1)
            a_np = a_t.squeeze(0).cpu().numpy().astype(np.float32)
            obs, _, t, tr, _ = env.step(a_np, repeat=cfg.action_repeat)
            done = bool(t or tr)

            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=cfg.bit_depth)
            e = encoder(obs_t)
            act_t = torch.tensor(a_np, dtype=torch.float32, device=device).unsqueeze(0)
            h, s, _, _ = rssm.observe_step(e, act_t, h, s, sample=False)

            ep_pos.append(env.agent_pos.copy())
            ep_h.append(h.squeeze(0).cpu().numpy())
            ep_s.append(s.squeeze(0).cpu().numpy())

        all_pos.extend(ep_pos)
        all_h.extend(ep_h)
        all_s.extend(ep_s)
        traj_pos.append(np.array(ep_pos))
        traj_h.append(np.array(ep_h))
        traj_s.append(np.array(ep_s))

    data = {
        "pos": np.array(all_pos, dtype=np.float32),
        "h": np.array(all_h, dtype=np.float32),
        "s": np.array(all_s, dtype=np.float32),
        "traj_h": traj_h,
        "traj_s": traj_s,
    }
    print(f"    Collected {len(data['pos'])} data points from {cfg.collect_episodes} episodes")
    return data


# ===================================================================
#  3. Post-hoc GeoEncoder training
# ===================================================================

def train_geo_encoder(data: dict, cfg: TrainCfg, device: torch.device):
    """Train GeoEncoder on frozen (h,s) trajectories with temporal contrastive loss."""
    geo = GeoEncoder(cfg.deter_dim, cfg.stoch_dim, geo_dim=cfg.geo_dim, hidden_dim=cfg.geo_hidden).to(device)
    opt = torch.optim.Adam(geo.parameters(), lr=cfg.geo_lr)

    trajs_h = data["traj_h"]
    trajs_s = data["traj_s"]
    valid = [(h, s) for h, s in zip(trajs_h, trajs_s) if len(h) >= cfg.geo_window]
    if len(valid) < 4:
        print("    WARNING: not enough long trajectories for GeoEncoder training")
        return geo

    for epoch in range(cfg.geo_epochs):
        idx = np.random.choice(len(valid), size=cfg.geo_batch, replace=True)
        batch_h, batch_s = [], []
        for i in idx:
            h_traj, s_traj = valid[i]
            start = np.random.randint(0, len(h_traj) - cfg.geo_window + 1)
            batch_h.append(h_traj[start:start + cfg.geo_window])
            batch_s.append(s_traj[start:start + cfg.geo_window])

        bh = torch.tensor(np.array(batch_h), dtype=torch.float32, device=device)
        bs = torch.tensor(np.array(batch_s), dtype=torch.float32, device=device)
        g = geo(bh, bs)

        loss = temporal_reachability_loss(g, pos_k=cfg.geo_pos_k, neg_k=cfg.geo_neg_k, margin=cfg.geo_margin)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, cfg.geo_epochs // 3) == 0:
            print(f"    GeoEncoder epoch {epoch+1}/{cfg.geo_epochs}  loss={loss.item():.4f}")

    geo.eval()
    return geo


# ===================================================================
#  4. Probing — position decoding R²
# ===================================================================

class _LinearProbe(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc = nn.Linear(in_dim, 2)

    def forward(self, x):
        return self.fc(x)


class _MLPProbe(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                                 nn.Linear(hid, hid), nn.ReLU(),
                                 nn.Linear(hid, 2))

    def forward(self, x):
        return self.net(x)


def _train_probe(probe, X_train, Y_train, X_test, Y_test, device, epochs=1500, lr=3e-3):
    probe = probe.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    xtr = torch.tensor(X_train, dtype=torch.float32, device=device)
    ytr = torch.tensor(Y_train, dtype=torch.float32, device=device)
    xte = torch.tensor(X_test, dtype=torch.float32, device=device)
    yte = torch.tensor(Y_test, dtype=torch.float32, device=device)

    for _ in range(epochs):
        pred = probe(xtr)
        loss = F.mse_loss(pred, ytr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred_te = probe(xte)
        mse = (pred_te - yte).pow(2).mean().item()
        var = yte.var(dim=0).sum().item()
        r2 = 1.0 - mse / max(var, 1e-8)
    return r2


def run_probes(data: dict, geo_encoder: nn.Module, cfg: TrainCfg, device: torch.device):
    """Train linear + MLP probes on h, s, (h,s), g  ->  (x,y)."""
    pos = data["pos"]
    h = data["h"]
    s = data["s"]
    hs = np.concatenate([h, s], axis=-1)

    with torch.no_grad():
        ht = torch.tensor(h, dtype=torch.float32, device=device)
        st = torch.tensor(s, dtype=torch.float32, device=device)
        g = geo_encoder(ht, st).cpu().numpy()

    N = len(pos)
    idx = np.random.permutation(N)
    split = int(0.8 * N)
    tr, te = idx[:split], idx[split:]

    features = {"h": h, "s": s, "h+s": hs, "g(h,s)": g}
    results = {}
    for name, feat in features.items():
        r2_lin = _train_probe(_LinearProbe(feat.shape[1]), feat[tr], pos[tr], feat[te], pos[te],
                              device, epochs=cfg.n_probe_epochs)
        r2_mlp = _train_probe(_MLPProbe(feat.shape[1]), feat[tr], pos[tr], feat[te], pos[te],
                              device, epochs=cfg.n_probe_epochs)
        results[name] = {"linear_R2": round(r2_lin, 4), "mlp_R2": round(r2_mlp, 4)}
        print(f"    Probe {name:8s}  linear R²={r2_lin:.4f}  MLP R²={r2_mlp:.4f}")
    return results


# ===================================================================
#  5. Distance analysis
# ===================================================================

def run_distance_analysis(data: dict, geo_encoder: nn.Module, geodesic: "GeodesicComputer",
                          cfg: TrainCfg, device: torch.device):
    """Pearson / Spearman of latent distance vs geodesic distance."""
    pos = data["pos"]
    h = data["h"]
    s = data["s"]
    hs = np.concatenate([h, s], axis=-1)

    with torch.no_grad():
        ht = torch.tensor(h, dtype=torch.float32, device=device)
        st = torch.tensor(s, dtype=torch.float32, device=device)
        g = geo_encoder(ht, st).cpu().numpy()

    N = len(pos)
    n_pairs = min(cfg.n_pairs, N * (N - 1) // 2)
    i1 = np.random.randint(0, N, n_pairs)
    i2 = np.random.randint(0, N, n_pairs)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]

    geo_d = np.array([geodesic.distance(pos[a], pos[b]) for a, b in zip(i1, i2)], dtype=np.float32)
    valid = np.isfinite(geo_d) & (geo_d > 0)
    i1, i2, geo_d = i1[valid], i2[valid], geo_d[valid]

    features = {"h": h, "s": s, "h+s": hs, "g(h,s)": g}
    results = {}
    for name, feat in features.items():
        lat_d = np.linalg.norm(feat[i1] - feat[i2], axis=1)
        pr, pp = sp_stats.pearsonr(lat_d, geo_d)
        sr, sp = sp_stats.spearmanr(lat_d, geo_d)
        results[name] = {
            "pearson_r": round(float(pr), 4), "pearson_p": float(pp),
            "spearman_rho": round(float(sr), 4), "spearman_p": float(sp),
        }
        print(f"    Distance {name:8s}  pearson={pr:.4f}  spearman={sr:.4f}")
    return results, (i1, i2, geo_d, features)


# ===================================================================
#  6. Neighbourhood analysis — kNN overlap
# ===================================================================

def run_knn_analysis(data: dict, geo_encoder: nn.Module, geodesic: "GeodesicComputer",
                     cfg: TrainCfg, device: torch.device):
    """Fraction of k-nearest-neighbours shared between latent and geodesic spaces."""
    pos = data["pos"]
    h, s = data["h"], data["s"]
    hs = np.concatenate([h, s], axis=-1)

    with torch.no_grad():
        ht = torch.tensor(h, dtype=torch.float32, device=device)
        st = torch.tensor(s, dtype=torch.float32, device=device)
        g = geo_encoder(ht, st).cpu().numpy()

    # Sub-sample for tractability
    N = min(800, len(pos))
    idx = np.random.choice(len(pos), N, replace=False)
    pos_sub = pos[idx]

    geo_dm = geodesic.pairwise_distances(pos_sub)
    np.fill_diagonal(geo_dm, np.inf)
    geo_knn = np.argsort(geo_dm, axis=1)[:, :cfg.knn_k]

    features = {"h": h[idx], "s": s[idx], "h+s": hs[idx], "g(h,s)": g[idx]}
    results = {}
    for name, feat in features.items():
        dm = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=-1)
        np.fill_diagonal(dm, np.inf)
        lat_knn = np.argsort(dm, axis=1)[:, :cfg.knn_k]
        overlap = np.mean([len(set(lat_knn[i]) & set(geo_knn[i])) / cfg.knn_k for i in range(N)])
        results[name] = round(float(overlap), 4)
        print(f"    kNN overlap {name:8s}  = {overlap:.4f}")
    return results


# ===================================================================
#  7. Visualisation
# ===================================================================

def _pca_2d(X):
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def generate_plots(maze_name: str, data: dict, probe_res: dict, dist_res: dict,
                   dist_raw: tuple, knn_res: dict, geo_encoder: nn.Module,
                   env: PointMazeEnv, cfg: TrainCfg, device: torch.device, out_dir: str):
    """Generate all per-maze figures."""
    os.makedirs(out_dir, exist_ok=True)
    pos = data["pos"]
    h, s = data["h"], data["s"]
    hs = np.concatenate([h, s], axis=-1)

    with torch.no_grad():
        ht = torch.tensor(h, dtype=torch.float32, device=device)
        st = torch.tensor(s, dtype=torch.float32, device=device)
        g = geo_encoder(ht, st).cpu().numpy()

    # --- Figure 1: PCA embeddings coloured by x, y ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for col, (name, feat) in enumerate([("h", h), ("s", s), ("h+s", hs), ("g(h,s)", g)]):
        z2d = _pca_2d(feat)
        for row, (coord_name, coord_idx) in enumerate([("x", 0), ("y", 1)]):
            ax = axes[row, col]
            sc = ax.scatter(z2d[:, 0], z2d[:, 1], c=pos[:, coord_idx], cmap="viridis", s=4, alpha=0.5)
            plt.colorbar(sc, ax=ax, fraction=0.046)
            ax.set_title(f"{name} coloured by {coord_name}")
            ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.suptitle(f"Latent PCA — {maze_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pca_embeddings.png"), dpi=120)
    plt.close(fig)

    # --- Figure 2: Distance scatter (latent vs geodesic) ---
    i1, i2, geo_d, features = dist_raw
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for col, (name, feat) in enumerate(features.items()):
        lat_d = np.linalg.norm(feat[i1] - feat[i2], axis=1)
        ax = axes[col]
        ax.scatter(geo_d, lat_d, s=2, alpha=0.15)
        pr = dist_res[name]["pearson_r"]
        sr = dist_res[name]["spearman_rho"]
        ax.set_xlabel("Geodesic distance")
        ax.set_ylabel("Latent distance")
        ax.set_title(f"{name}\nr={pr:.3f}  ρ={sr:.3f}")
    fig.suptitle(f"Latent vs geodesic distance — {maze_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "distance_scatter.png"), dpi=120)
    plt.close(fig)

    # --- Figure 3: Probe R² bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(probe_res.keys())
    lin_r2 = [probe_res[n]["linear_R2"] for n in names]
    mlp_r2 = [probe_res[n]["mlp_R2"] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.18, lin_r2, 0.35, label="Linear", color="#5B9BD5")
    ax.bar(x + 0.18, mlp_r2, 0.35, label="MLP", color="#ED7D31")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("R²"); ax.set_title(f"Position Decoding — {maze_name}")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "probe_r2.png"), dpi=120)
    plt.close(fig)

    # --- Figure 4: kNN overlap bar chart ---
    fig, ax = plt.subplots(figsize=(7, 5))
    names_k = list(knn_res.keys())
    vals = [knn_res[n] for n in names_k]
    ax.bar(names_k, vals, color="#70AD47")
    ax.set_ylabel(f"kNN overlap (k={cfg.knn_k})")
    ax.set_title(f"Neighbourhood Preservation — {maze_name}")
    ax.set_ylim(0, 1.05); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "knn_overlap.png"), dpi=120)
    plt.close(fig)

    # --- Figure 5: Position heatmap (predicted vs true) ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for col, (name, feat) in enumerate([("h", h), ("s", s), ("h+s", hs), ("g(h,s)", g)]):
        probe = _MLPProbe(feat.shape[1]).to(device)
        N = len(pos)
        idx_all = np.random.permutation(N)
        spl = int(0.8 * N)
        tr_i, te_i = idx_all[:spl], idx_all[spl:]
        _train_probe(probe, feat[tr_i], pos[tr_i], feat[te_i], pos[te_i], device, epochs=cfg.n_probe_epochs)
        with torch.no_grad():
            pred = probe(torch.tensor(feat[te_i], dtype=torch.float32, device=device)).cpu().numpy()
        ax = axes[col]
        err = np.linalg.norm(pred - pos[te_i], axis=1)
        sc = ax.scatter(pos[te_i, 0], pos[te_i, 1], c=err, cmap="hot_r", s=8, alpha=0.6, vmin=0)
        plt.colorbar(sc, ax=ax, label="Error")
        ax.set_title(f"{name} — position error")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        ax.set_aspect("equal")
    fig.suptitle(f"Probe Prediction Error — {maze_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "position_error_map.png"), dpi=120)
    plt.close(fig)


def generate_summary_plots(all_results: dict, out_dir: str):
    """Cross-maze comparison figures."""
    os.makedirs(out_dir, exist_ok=True)
    mazes = list(all_results.keys())
    feat_names = ["h", "s", "h+s", "g(h,s)"]

    # --- Summary 1: MLP R² across mazes ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(mazes))
    w = 0.18
    for i, fn in enumerate(feat_names):
        vals = [all_results[m]["probes"].get(fn, {}).get("mlp_R2", 0) for m in mazes]
        ax.bar(x + (i - 1.5) * w, vals, w, label=fn)
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("MLP Probe R²"); ax.set_title("Position Decoding R² by Maze and Feature")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_probe_r2.png"), dpi=150)
    plt.close(fig)

    # --- Summary 2: Spearman rho across mazes ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, fn in enumerate(feat_names):
        vals = [all_results[m]["distances"].get(fn, {}).get("spearman_rho", 0) for m in mazes]
        ax.bar(x + (i - 1.5) * w, vals, w, label=fn)
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("Spearman ρ (latent vs geodesic)")
    ax.set_title("Distance Preservation by Maze and Feature")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_spearman.png"), dpi=150)
    plt.close(fig)

    # --- Summary 3: kNN overlap across mazes ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, fn in enumerate(feat_names):
        vals = [all_results[m]["knn"].get(fn, 0) for m in mazes]
        ax.bar(x + (i - 1.5) * w, vals, w, label=fn)
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("kNN Overlap"); ax.set_title("Neighbourhood Preservation by Maze and Feature")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_knn.png"), dpi=150)
    plt.close(fig)


# ===================================================================
#  8. Per-maze pipeline
# ===================================================================

def run_single_maze(maze_name: str, cfg: TrainCfg, device: torch.device, out_root: str):
    print(f"\n{'='*60}")
    print(f"  Maze: {maze_name}")
    print(f"{'='*60}")

    if maze_name.startswith("random_"):
        seed = int(maze_name.split("_")[1]) if "_" in maze_name else 42
        lines = generate_random_maze(11, 11, wall_density=0.18, seed=seed)
        env = PointMazeEnv(layout=lines, img_size=(cfg.img_size, cfg.img_size))
    else:
        env = PointMazeEnv(layout=maze_name, img_size=(cfg.img_size, cfg.img_size))

    print(f"  Grid: {env.grid_h}×{env.grid_w}  free cells: {env.geodesic.n_free}")
    out_dir = os.path.join(out_root, maze_name)
    os.makedirs(out_dir, exist_ok=True)

    print("\n  [1/5] Training Dreamer world model ...")
    set_seed(0)
    models = train_world_model(env, cfg, device)

    print("\n  [2/5] Collecting position-latent data ...")
    data = collect_data(env, models, cfg, device)

    print("\n  [3/5] Training GeoEncoder (post-hoc) ...")
    geo_enc = train_geo_encoder(data, cfg, device)

    print("\n  [4/5] Running analyses ...")
    probe_res = run_probes(data, geo_enc, cfg, device)
    dist_res, dist_raw = run_distance_analysis(data, geo_enc, env.geodesic, cfg, device)
    knn_res = run_knn_analysis(data, geo_enc, env.geodesic, cfg, device)

    print("\n  [5/5] Generating plots ...")
    generate_plots(maze_name, data, probe_res, dist_res, dist_raw, knn_res,
                   geo_enc, env, cfg, device, out_dir)

    env.close()

    results = {"probes": probe_res, "distances": dist_res, "knn": knn_res}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ===================================================================
#  9. Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Maze latent-geometry evaluation")
    p.add_argument("--mazes", nargs="+", default=list(MAZE_LAYOUTS.keys()),
                   help="Which maze layouts to test")
    p.add_argument("--output_dir", default="maze_geometry_results")
    p.add_argument("--quick", action="store_true", help="Fast sanity-check run")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)

    cfg = TrainCfg()
    if args.quick:
        cfg.max_episodes = 30
        cfg.train_steps = 15
        cfg.collect_interval = 40
        cfg.collect_episodes = 20
        cfg.batch_size = 16
        cfg.seq_len = 20
        cfg.geo_epochs = 80
        cfg.geo_batch = 32
        cfg.geo_window = 18
        cfg.n_probe_epochs = 600
        cfg.n_pairs = 3000
        cfg.replay_capacity = 40_000

    print(f"Device: {device}")
    print(f"Mazes:  {args.mazes}")
    print(f"Quick:  {args.quick}\n")

    all_results = {}
    for maze_name in args.mazes:
        results = run_single_maze(maze_name, cfg, device, args.output_dir)
        all_results[maze_name] = results

    summary_dir = os.path.join(args.output_dir, "summary")
    generate_summary_plots(all_results, summary_dir)

    with open(os.path.join(summary_dir, "all_metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Print summary table
    print(f"\n{'='*80}")
    print("  SUMMARY")
    print(f"{'='*80}")
    header = f"{'Maze':12s} | {'Feature':8s} | {'Lin R²':8s} | {'MLP R²':8s} | {'Pearson':8s} | {'Spearman':8s} | {'kNN':6s}"
    print(header)
    print("-" * len(header))
    for maze in args.mazes:
        r = all_results[maze]
        for fn in ["h", "s", "h+s", "g(h,s)"]:
            pr = r["probes"].get(fn, {})
            dr = r["distances"].get(fn, {})
            knn = r["knn"].get(fn, 0)
            print(f"{maze:12s} | {fn:8s} | {pr.get('linear_R2',0):8.4f} | {pr.get('mlp_R2',0):8.4f} | "
                  f"{dr.get('pearson_r',0):8.4f} | {dr.get('spearman_rho',0):8.4f} | {knn:6.4f}")
        print("-" * len(header))

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
