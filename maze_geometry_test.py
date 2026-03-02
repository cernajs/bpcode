#!/usr/bin/env python3
"""
Comprehensive latent-geometry evaluation for Dreamer world models on maze envs.

Pipeline per maze layout × seed:
  1. Train vanilla Dreamer world model  (encoder + RSSM + decoder + reward + actor-critic)
  2. Collect (position, h, s, encoder_emb, raw_obs) tuples from trained model rollouts
  3. Post-hoc train GeoEncoder g(h,s) with revisit-aware temporal contrastive loss
  4. Probe analysis      — linear / MLP probes  position decoding  R²
                           (episode-wise train/test split to avoid trajectory interpolation)
  5. Distance analysis   — Pearson / Spearman of latent dist vs geodesic dist
  6. Neighbourhood analysis — kNN overlap (with chance baseline) + trustworthiness & continuity
  7. Visualizations      — PCA embeddings, scatter plots, summary comparison

Baselines evaluated alongside RSSM (h, s, h+s):
  - encoder_e:  raw ConvEncoder output (before RSSM)
  - pix_pca64:  64-dim PCA of flattened raw pixels

Multi-seed support:
  Runs ≥5 seeds by default and reports mean±std for all metrics.

Usage:
    python maze_geometry_test.py                                   # all 5 layouts, 5 seeds
    python maze_geometry_test.py --mazes corridor loop             # just those two
    python maze_geometry_test.py --quick                           # fast sanity-check run
    python maze_geometry_test.py --seeds 0 1 2 3 4 5 6 7 8 9      # 10 seeds
    python maze_geometry_test.py --quick --seeds 0                 # single seed quick test
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from maze_env import MAZE_LAYOUTS, GeodesicComputer, PointMazeEnv, generate_random_maze
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
    # Optional: geodesic-supervised GeoEncoder variant
    geo_sup_epochs: int = 200
    geo_sup_batch: int = 256
    geo_sup_candidates: int = 64
    geo_sup_margin: float = 0.2
    geo_sup_uniformity_weight: float = 0.1
    geo_sup_uniformity_t: float = 2.0
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
    """Run trained model, collect (pos, h, s, encoder_emb, raw_obs) per step, grouped by episode."""
    encoder, rssm, actor = models["encoder"], models["rssm"], models["actor"]
    encoder.eval(); rssm.eval(); actor.eval()

    all_pos, all_h, all_s, all_e, all_obs = [], [], [], [], []
    traj_pos, traj_h, traj_s = [], [], []
    ep_ids = []

    for ep_idx in range(cfg.collect_episodes):
        obs, _ = env.reset()
        done = False
        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=cfg.bit_depth)
        e = encoder(obs_t)
        h, s = rssm.get_init_state(e)

        ep_pos = [env.agent_pos.copy()]
        ep_h = [h.squeeze(0).cpu().numpy()]
        ep_s = [s.squeeze(0).cpu().numpy()]
        ep_e = [e.squeeze(0).cpu().numpy()]
        ep_obs = [obs.astype(np.float32).flatten() / 255.0]

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
            ep_e.append(e.squeeze(0).cpu().numpy())
            ep_obs.append(obs.astype(np.float32).flatten() / 255.0)

        n_ep = len(ep_pos)
        all_pos.extend(ep_pos)
        all_h.extend(ep_h)
        all_s.extend(ep_s)
        all_e.extend(ep_e)
        all_obs.extend(ep_obs)
        ep_ids.extend([ep_idx] * n_ep)
        traj_pos.append(np.array(ep_pos))
        traj_h.append(np.array(ep_h))
        traj_s.append(np.array(ep_s))

    data = {
        "pos": np.array(all_pos, dtype=np.float32),
        "h": np.array(all_h, dtype=np.float32),
        "s": np.array(all_s, dtype=np.float32),
        "encoder_emb": np.array(all_e, dtype=np.float32),
        "raw_obs": np.array(all_obs, dtype=np.float32),
        "episode_ids": np.array(ep_ids, dtype=np.int64),
        "traj_h": traj_h,
        "traj_s": traj_s,
        "traj_pos": traj_pos,
    }
    print(f"    Collected {len(data['pos'])} data points from {cfg.collect_episodes} episodes")
    return data


# ===================================================================
#  3. Post-hoc GeoEncoder training
# ===================================================================

def _revisit_aware_temporal_loss(
    g_seq: torch.Tensor,
    cell_seq: torch.Tensor,
    pos_k: int = 3,
    neg_k: int = 12,
    margin: float = 0.6,
    uniformity_weight: float = 0.1,
    uniformity_t: float = 2.0,
) -> torch.Tensor:
    """Temporal contrastive loss that filters false negatives from revisits.

    cell_seq: [B, T] int tensor of geodesic cell indices per timestep.
    A "negative" at time i+neg_k is skipped (resampled) if it shares the
    same cell as the anchor at time i, avoiding the revisit problem.
    """
    B, T, D = g_seq.shape
    device = g_seq.device

    if T <= neg_k or pos_k < 1 or neg_k <= pos_k:
        return g_seq.new_zeros(())

    max_i = T - neg_k - 1
    if max_i < 1:
        max_i = T - neg_k

    b = torch.arange(B, device=device)
    i = torch.randint(0, max_i, (B,), device=device)
    pos_delta = torch.randint(1, pos_k + 1, (B,), device=device)
    j_pos = i + pos_delta
    j_neg1 = i + neg_k
    j_neg2 = (i + neg_k + 1).clamp(max=T - 1)

    anchor_cell = cell_seq[b, i]
    neg1_cell = cell_seq[b, j_neg1]
    neg2_cell = cell_seq[b, j_neg2]

    # Resample negatives that revisit the anchor's cell:
    # try shifting the negative further away, else use a cross-batch negative.
    for _ in range(3):
        revisit1 = (neg1_cell == anchor_cell)
        if not revisit1.any():
            break
        shift = torch.randint(neg_k, T, (B,), device=device).clamp(max=T - 1)
        j_neg1 = torch.where(revisit1, shift, j_neg1)
        neg1_cell = cell_seq[b, j_neg1]

    for _ in range(3):
        revisit2 = (neg2_cell == anchor_cell)
        if not revisit2.any():
            break
        shift = torch.randint(neg_k, T, (B,), device=device).clamp(max=T - 1)
        j_neg2 = torch.where(revisit2, shift, j_neg2)
        neg2_cell = cell_seq[b, j_neg2]

    g_i = g_seq[b, i]
    g_pos = g_seq[b, j_pos]
    g_neg1 = g_seq[b, j_neg1]
    g_neg2 = g_seq[b, j_neg2]

    d_pos = torch.norm(g_i - g_pos, dim=-1)
    d_neg1 = torch.norm(g_i - g_neg1, dim=-1)
    d_neg2 = torch.norm(g_i - g_neg2, dim=-1)

    loss_rank = (
        F.relu(d_pos + margin - d_neg1).mean()
        + F.relu(d_pos + margin - d_neg2).mean()
    ) * 0.5

    g_flat = g_seq.reshape(B * T, D)
    max_uni = min(B * T, 256)
    if B * T > max_uni:
        idx = torch.randperm(B * T, device=device)[:max_uni]
        g_uni = g_flat[idx]
    else:
        g_uni = g_flat
    sq = torch.cdist(g_uni, g_uni).pow(2)
    mask = torch.eye(g_uni.size(0), device=device, dtype=torch.bool)
    sq = sq.masked_fill(mask, 1e9)
    loss_uni = torch.logsumexp(-uniformity_t * sq, dim=1).mean()

    g_cross = g_seq.roll(1, dims=0)[b, i]
    d_cross = torch.norm(g_i - g_cross, dim=-1)
    loss_cross = F.relu(d_pos + margin - d_cross).mean()

    return loss_rank + uniformity_weight * loss_uni + 0.5 * loss_cross


def train_geo_encoder(data: dict, cfg: TrainCfg, device: torch.device, geodesic: "GeodesicComputer"):
    """Train GeoEncoder on frozen (h,s) trajectories with revisit-aware temporal contrastive loss."""
    geo = GeoEncoder(cfg.deter_dim, cfg.stoch_dim, geo_dim=cfg.geo_dim, hidden_dim=cfg.geo_hidden).to(device)
    opt = torch.optim.Adam(geo.parameters(), lr=cfg.geo_lr)

    trajs_h = data["traj_h"]
    trajs_s = data["traj_s"]
    trajs_pos = data.get("traj_pos", [None] * len(trajs_h))

    valid = []
    for h, s, p in zip(trajs_h, trajs_s, trajs_pos):
        if len(h) >= cfg.geo_window:
            cell_idx = None
            if p is not None:
                cell_idx = np.array(
                    [geodesic.cell_to_idx[geodesic.pos_to_cell(float(pt[0]), float(pt[1]))]
                     for pt in p],
                    dtype=np.int64,
                )
            valid.append((h, s, cell_idx))

    if len(valid) < 4:
        print("    WARNING: not enough long trajectories for GeoEncoder training")
        return geo

    has_cells = all(v[2] is not None for v in valid)

    for epoch in range(cfg.geo_epochs):
        idx = np.random.choice(len(valid), size=cfg.geo_batch, replace=True)
        batch_h, batch_s, batch_cells = [], [], []
        for i in idx:
            h_traj, s_traj, c_traj = valid[i]
            start = np.random.randint(0, len(h_traj) - cfg.geo_window + 1)
            batch_h.append(h_traj[start:start + cfg.geo_window])
            batch_s.append(s_traj[start:start + cfg.geo_window])
            if c_traj is not None:
                batch_cells.append(c_traj[start:start + cfg.geo_window])

        bh = torch.tensor(np.array(batch_h), dtype=torch.float32, device=device)
        bs = torch.tensor(np.array(batch_s), dtype=torch.float32, device=device)
        g = geo(bh, bs)

        if has_cells and batch_cells:
            bc = torch.tensor(np.array(batch_cells), dtype=torch.long, device=device)
            loss = _revisit_aware_temporal_loss(
                g, bc, pos_k=cfg.geo_pos_k, neg_k=cfg.geo_neg_k, margin=cfg.geo_margin,
            )
        else:
            loss = temporal_reachability_loss(
                g, pos_k=cfg.geo_pos_k, neg_k=cfg.geo_neg_k, margin=cfg.geo_margin,
            )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, cfg.geo_epochs // 3) == 0:
            print(f"    GeoEncoder epoch {epoch+1}/{cfg.geo_epochs}  loss={loss.item():.4f}")

    geo.eval()
    return geo


# ===================================================================
#  3b. Optional: geodesic-supervised GeoEncoder training
# ===================================================================

def _positions_to_cell_indices(geodesic: "GeodesicComputer", pos: np.ndarray) -> np.ndarray:
    """Map continuous (x,y) positions to free-cell indices used by geodesic.dist_matrix."""
    # geodesic.pos_to_cell expects (x,y) but returns (row,col)
    cells = [geodesic.pos_to_cell(float(p[0]), float(p[1])) for p in pos]
    return np.asarray([geodesic.cell_to_idx[c] for c in cells], dtype=np.int64)


def _uniformity_loss_sphere(g: torch.Tensor, t: float = 2.0, max_uni: int = 256) -> torch.Tensor:
    """Wang & Isola-style uniformity: logsumexp(-t * ||g_i - g_j||^2)."""
    g_flat = g.reshape(-1, g.shape[-1])
    n = g_flat.size(0)
    if n <= 2:
        return g.new_zeros(())
    if n > max_uni:
        idx = torch.randperm(n, device=g.device)[:max_uni]
        g_flat = g_flat[idx]
    sq = torch.cdist(g_flat, g_flat).pow(2)
    mask = torch.eye(sq.size(0), device=g.device, dtype=torch.bool)
    sq = sq.masked_fill(mask, 1e9)
    return torch.logsumexp(-t * sq, dim=1).mean()


def train_geo_encoder_geodesic(
    data: dict,
    cfg: TrainCfg,
    device: torch.device,
    geodesic: "GeodesicComputer",
):
    """Train GeoEncoder so ||g_i - g_j|| preserves geodesic ordering (triplet ranking)."""
    geo = GeoEncoder(cfg.deter_dim, cfg.stoch_dim, geo_dim=cfg.geo_dim, hidden_dim=cfg.geo_hidden).to(device)
    opt = torch.optim.Adam(geo.parameters(), lr=cfg.geo_lr)

    pos = data["pos"]
    h = data["h"]
    s = data["s"]
    N = len(pos)
    if N < 512:
        print("    WARNING: not enough points for geodesic-supervised GeoEncoder; skipping")
        geo.eval()
        return geo

    cell_idx = _positions_to_cell_indices(geodesic, pos)
    dist_mat = geodesic.dist_matrix  # numpy array [n_free, n_free]

    for epoch in range(cfg.geo_sup_epochs):
        # anchors and candidate pool
        B = min(cfg.geo_sup_batch, N)
        anchors = np.random.randint(0, N, size=B)
        cand = np.random.randint(0, N, size=(B, cfg.geo_sup_candidates))

        a_cell = cell_idx[anchors]  # [B]
        c_cell = cell_idx[cand]     # [B, C]
        d_geo = dist_mat[a_cell[:, None], c_cell]  # [B, C]

        # avoid selecting self as positive
        same = cand == anchors[:, None]
        d_geo = np.where(same, np.inf, d_geo)

        # positives: nearest geodesic; negatives: farthest geodesic
        pos_j = cand[np.arange(B), np.argmin(d_geo, axis=1)]
        neg_j = cand[np.arange(B), np.argmax(d_geo, axis=1)]

        bh = torch.tensor(h[anchors], dtype=torch.float32, device=device)
        bs = torch.tensor(s[anchors], dtype=torch.float32, device=device)
        ph = torch.tensor(h[pos_j], dtype=torch.float32, device=device)
        ps = torch.tensor(s[pos_j], dtype=torch.float32, device=device)
        nh = torch.tensor(h[neg_j], dtype=torch.float32, device=device)
        ns = torch.tensor(s[neg_j], dtype=torch.float32, device=device)

        g_a = geo(bh, bs)
        g_p = geo(ph, ps)
        g_n = geo(nh, ns)

        d_pos = torch.norm(g_a - g_p, dim=-1)
        d_neg = torch.norm(g_a - g_n, dim=-1)
        loss_rank = F.relu(d_pos + cfg.geo_sup_margin - d_neg).mean()
        loss_uni = _uniformity_loss_sphere(g_a, t=cfg.geo_sup_uniformity_t)
        loss = loss_rank + cfg.geo_sup_uniformity_weight * loss_uni

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, cfg.geo_sup_epochs // 3) == 0:
            print(
                f"    GeoSup epoch {epoch+1}/{cfg.geo_sup_epochs}  "
                f"rank={loss_rank.item():.4f}  uni={loss_uni.item():.4f}  total={loss.item():.4f}"
            )

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


def _raw_pixel_pca(raw_obs: np.ndarray, n_components: int = 64) -> np.ndarray:
    """PCA baseline: reduce flattened raw pixels to n_components dims."""
    X = raw_obs - raw_obs.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:n_components].T


def _build_feature_dict(
    data: dict,
    device: torch.device,
    geo_temporal: Optional[nn.Module] = None,
    geo_geodesic: Optional[nn.Module] = None,
) -> Dict[str, np.ndarray]:
    """Compute all feature representations to evaluate, including baselines."""
    h = data["h"]
    s = data["s"]
    feats: Dict[str, np.ndarray] = {"h": h, "s": s, "h+s": np.concatenate([h, s], axis=-1)}

    if "encoder_emb" in data:
        feats["encoder_e"] = data["encoder_emb"]

    if "raw_obs" in data:
        feats["pix_pca64"] = _raw_pixel_pca(data["raw_obs"], n_components=64)

    if geo_temporal is not None:
        with torch.no_grad():
            ht = torch.tensor(h, dtype=torch.float32, device=device)
            st = torch.tensor(s, dtype=torch.float32, device=device)
            feats["g(h,s)"] = geo_temporal(ht, st).cpu().numpy()
    if geo_geodesic is not None:
        with torch.no_grad():
            ht = torch.tensor(h, dtype=torch.float32, device=device)
            st = torch.tensor(s, dtype=torch.float32, device=device)
            feats["g_geo(h,s)"] = geo_geodesic(ht, st).cpu().numpy()
    return feats


def run_probes(
    pos: np.ndarray,
    features: Dict[str, np.ndarray],
    cfg: TrainCfg,
    device: torch.device,
    episode_ids: Optional[np.ndarray] = None,
):
    """Train linear + MLP probes on each feature -> (x,y).

    Uses episode-wise split when episode_ids is provided: train on some
    episodes, test on held-out episodes.  This prevents "trajectory
    interpolation" from inflating R² scores.
    """
    N = len(pos)

    if episode_ids is not None:
        unique_eps = np.unique(episode_ids)
        np.random.shuffle(unique_eps)
        split_ep = int(0.8 * len(unique_eps))
        train_eps = set(unique_eps[:split_ep].tolist())
        tr = np.array([i for i in range(N) if episode_ids[i] in train_eps])
        te = np.array([i for i in range(N) if episode_ids[i] not in train_eps])
        print(f"    Episode-wise split: {split_ep} train eps, {len(unique_eps)-split_ep} test eps  "
              f"({len(tr)} train pts, {len(te)} test pts)")
    else:
        idx = np.random.permutation(N)
        split = int(0.8 * N)
        tr, te = idx[:split], idx[split:]

    results = {}
    for name, feat in features.items():
        r2_lin = _train_probe(_LinearProbe(feat.shape[1]), feat[tr], pos[tr], feat[te], pos[te],
                              device, epochs=cfg.n_probe_epochs)
        r2_mlp = _train_probe(_MLPProbe(feat.shape[1]), feat[tr], pos[tr], feat[te], pos[te],
                              device, epochs=cfg.n_probe_epochs)
        results[name] = {"linear_R2": round(r2_lin, 4), "mlp_R2": round(r2_mlp, 4)}
        print(f"    Probe {name:12s}  linear R²={r2_lin:.4f}  MLP R²={r2_mlp:.4f}")
    return results


# ===================================================================
#  5. Distance analysis
# ===================================================================

def run_distance_analysis(pos: np.ndarray, features: Dict[str, np.ndarray], geodesic: "GeodesicComputer",
                          cfg: TrainCfg):
    """Pearson / Spearman of latent distance vs geodesic distance."""
    N = len(pos)
    n_pairs = min(cfg.n_pairs, N * (N - 1) // 2)
    i1 = np.random.randint(0, N, n_pairs)
    i2 = np.random.randint(0, N, n_pairs)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]

    geo_d = np.array([geodesic.distance(pos[a], pos[b]) for a, b in zip(i1, i2)], dtype=np.float32)
    valid = np.isfinite(geo_d) & (geo_d > 0)
    i1, i2, geo_d = i1[valid], i2[valid], geo_d[valid]

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

def run_knn_analysis(pos: np.ndarray, features: Dict[str, np.ndarray], geodesic: "GeodesicComputer",
                     cfg: TrainCfg):
    """Fraction of k-nearest-neighbours shared between latent and geodesic spaces.

    Also reports the chance baseline: k / (N-1).
    """
    N = min(800, len(pos))
    idx = np.random.choice(len(pos), N, replace=False)
    pos_sub = pos[idx]

    geo_dm = geodesic.pairwise_distances(pos_sub)
    np.fill_diagonal(geo_dm, np.inf)
    geo_knn = np.argsort(geo_dm, axis=1)[:, :cfg.knn_k]

    chance = cfg.knn_k / (N - 1)

    results = {}
    for name, feat_full in features.items():
        feat = feat_full[idx]
        dm = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=-1)
        np.fill_diagonal(dm, np.inf)
        lat_knn = np.argsort(dm, axis=1)[:, :cfg.knn_k]
        overlap = np.mean([len(set(lat_knn[i]) & set(geo_knn[i])) / cfg.knn_k for i in range(N)])
        ratio_vs_chance = overlap / max(chance, 1e-9)
        results[name] = round(float(overlap), 4)
        print(f"    kNN overlap {name:12s}  = {overlap:.4f}  ({ratio_vs_chance:.1f}× chance={chance:.4f})")

    results["_chance_baseline"] = round(float(chance), 6)
    return results


# ===================================================================
#  6b. Trustworthiness & Continuity (local manifold quality)
# ===================================================================

def _rank_matrix(dm: np.ndarray) -> np.ndarray:
    """Return rank matrix: rank_ij = rank of j among i's neighbours (0-indexed)."""
    return np.argsort(np.argsort(dm, axis=1), axis=1)


def run_trustworthiness_continuity(
    pos: np.ndarray,
    features: Dict[str, np.ndarray],
    geodesic: "GeodesicComputer",
    cfg: TrainCfg,
    k: int = 10,
):
    """Trustworthiness (T) and Continuity (C) metrics (Venna & Kaski, 2006).

    T measures whether latent neighbours are also true (geodesic) neighbours.
    C measures whether true neighbours are also latent neighbours.
    Both in [0, 1]; 1 = perfect.
    """
    N = min(800, len(pos))
    idx = np.random.choice(len(pos), N, replace=False)
    pos_sub = pos[idx]

    geo_dm = geodesic.pairwise_distances(pos_sub)
    np.fill_diagonal(geo_dm, np.inf)
    geo_ranks = _rank_matrix(geo_dm)

    results = {}
    norm_factor = N * k * (2 * N - 3 * k - 1)
    if norm_factor <= 0:
        norm_factor = 1.0

    for name, feat_full in features.items():
        feat = feat_full[idx]
        lat_dm = np.linalg.norm(feat[:, None, :] - feat[None, :, :], axis=-1)
        np.fill_diagonal(lat_dm, np.inf)
        lat_ranks = _rank_matrix(lat_dm)

        lat_knn = np.argsort(lat_dm, axis=1)[:, :k]
        geo_knn = np.argsort(geo_dm, axis=1)[:, :k]

        # Trustworthiness: penalty for latent-neighbours not in geo top-k
        trust_sum = 0.0
        for i in range(N):
            lat_set = set(lat_knn[i])
            geo_set = set(geo_knn[i])
            intruders = lat_set - geo_set
            for j in intruders:
                trust_sum += geo_ranks[i, j] - k
        T = 1.0 - (2.0 / norm_factor) * trust_sum

        # Continuity: penalty for geo-neighbours missing from latent top-k
        cont_sum = 0.0
        for i in range(N):
            lat_set = set(lat_knn[i])
            geo_set = set(geo_knn[i])
            missing = geo_set - lat_set
            for j in missing:
                cont_sum += lat_ranks[i, j] - k
        C = 1.0 - (2.0 / norm_factor) * cont_sum

        results[name] = {"trustworthiness": round(float(T), 4), "continuity": round(float(C), 4)}
        print(f"    T&C {name:12s}  trust={T:.4f}  cont={C:.4f}")

    return results


# ===================================================================
#  7. Visualisation
# ===================================================================

def _pca_2d(X):
    X = X - X.mean(0)
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    return X @ Vt[:2].T


def generate_plots(
    maze_name: str,
    pos: np.ndarray,
    features: Dict[str, np.ndarray],
    probe_res: dict,
    dist_res: dict,
    dist_raw: tuple,
    knn_res: dict,
    cfg: TrainCfg,
    device: torch.device,
    out_dir: str,
):
    """Generate all per-maze figures."""
    os.makedirs(out_dir, exist_ok=True)
    feat_items = list(features.items())
    n_feat = len(feat_items)

    # --- Figure 1: PCA embeddings coloured by x, y ---
    fig, axes = plt.subplots(2, n_feat, figsize=(5 * n_feat, 10))
    if n_feat == 1:
        axes = np.asarray(axes).reshape(2, 1)
    for col, (name, feat) in enumerate(feat_items):
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
    fig, axes = plt.subplots(1, len(features), figsize=(5 * len(features), 5))
    if len(features) == 1:
        axes = [axes]
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

    # --- Figure 4: kNN overlap bar chart with chance baseline ---
    fig, ax = plt.subplots(figsize=(7, 5))
    names_k = [n for n in knn_res if not n.startswith("_")]
    vals = [knn_res[n] for n in names_k]
    chance = knn_res.get("_chance_baseline", 0)
    ax.bar(names_k, vals, color="#70AD47")
    if chance > 0:
        ax.axhline(chance, color="red", linestyle="--", linewidth=1.2, label=f"chance={chance:.4f}")
        ax.legend()
    ax.set_ylabel(f"kNN overlap (k={cfg.knn_k})")
    ax.set_title(f"Neighbourhood Preservation — {maze_name}")
    ax.set_ylim(0, max(1.05, max(vals) * 1.15 if vals else 1.05)); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "knn_overlap.png"), dpi=120)
    plt.close(fig)

    # --- Figure 5: Position heatmap (predicted vs true) ---
    fig, axes = plt.subplots(1, n_feat, figsize=(5 * n_feat, 5))
    if n_feat == 1:
        axes = [axes]
    for col, (name, feat) in enumerate(feat_items):
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


# ===================================================================
#  8. Per-maze pipeline (single seed)
# ===================================================================

def run_single_maze(maze_name: str, cfg: TrainCfg, device: torch.device, out_root: str,
                    seed: int = 0):
    """Full pipeline for one maze + one seed. Returns results dict."""
    print(f"\n{'='*60}")
    print(f"  Maze: {maze_name}  (seed={seed})")
    print(f"{'='*60}")

    set_seed(seed)

    if maze_name.startswith("random_"):
        maze_seed = int(maze_name.split("_")[1]) if "_" in maze_name else 42
        lines = generate_random_maze(11, 11, wall_density=0.18, seed=maze_seed)
        env = PointMazeEnv(layout=lines, img_size=(cfg.img_size, cfg.img_size))
    else:
        env = PointMazeEnv(layout=maze_name, img_size=(cfg.img_size, cfg.img_size))

    print(f"  Grid: {env.grid_h}×{env.grid_w}  free cells: {env.geodesic.n_free}")
    out_dir = os.path.join(out_root, maze_name, f"seed{seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n  [1/5] Training Dreamer world model ...")
    models = train_world_model(env, cfg, device)

    print("\n  [2/5] Collecting position-latent data ...")
    data = collect_data(env, models, cfg, device)

    print("\n  [3/5] Training GeoEncoder (post-hoc, temporal) ...")
    geo_temporal = train_geo_encoder(data, cfg, device, env.geodesic)

    print("\n  [4/5] Running analyses ...")
    geo_geo = None
    if cfg.geo_sup_epochs > 0 and getattr(cfg, "_do_geo_supervised", False):
        print("    Training GeoEncoder (geodesic-supervised) ...")
        geo_geo = train_geo_encoder_geodesic(data, cfg, device, env.geodesic)

    pos = data["pos"]
    episode_ids = data.get("episode_ids", None)
    feat_dict = _build_feature_dict(data, device, geo_temporal=geo_temporal, geo_geodesic=geo_geo)

    probe_res = run_probes(pos, feat_dict, cfg, device, episode_ids=episode_ids)
    dist_res, dist_raw = run_distance_analysis(pos, feat_dict, env.geodesic, cfg)
    knn_res = run_knn_analysis(pos, feat_dict, env.geodesic, cfg)
    tc_res = run_trustworthiness_continuity(pos, feat_dict, env.geodesic, cfg, k=cfg.knn_k)

    print("\n  [5/5] Generating plots ...")
    generate_plots(maze_name, pos, feat_dict, probe_res, dist_res, dist_raw, knn_res,
                   cfg, device, out_dir)

    env.close()

    results = {"probes": probe_res, "distances": dist_res, "knn": knn_res, "trust_cont": tc_res}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results


# ===================================================================
#  8b. Multi-seed aggregation
# ===================================================================

def _deep_scalar_paths(d: dict, prefix: str = "") -> List[Tuple[str, float]]:
    """Recursively extract (dotted_key, scalar_value) pairs from nested dict."""
    out = []
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.extend(_deep_scalar_paths(v, key))
        elif isinstance(v, (int, float)):
            out.append((key, float(v)))
    return out


def _aggregate_seed_results(seed_results: List[dict]) -> dict:
    """Given a list of per-seed result dicts, compute mean±std for every scalar."""
    from collections import defaultdict
    buckets: Dict[str, List[float]] = defaultdict(list)
    for sr in seed_results:
        for key, val in _deep_scalar_paths(sr):
            buckets[key].append(val)

    agg = {}
    for key, vals in buckets.items():
        arr = np.array(vals)
        agg[key] = {"mean": round(float(arr.mean()), 4), "std": round(float(arr.std()), 4),
                     "values": [round(v, 4) for v in vals]}
    return agg


def _get_agg(agg: dict, dotted_key: str, stat: str = "mean") -> float:
    """Helper: agg['probes.h.mlp_R2']['mean']."""
    entry = agg.get(dotted_key, {})
    if isinstance(entry, dict):
        return entry.get(stat, 0.0)
    return 0.0


def _fmt_mean_std(agg: dict, dotted_key: str) -> str:
    m = _get_agg(agg, dotted_key, "mean")
    s = _get_agg(agg, dotted_key, "std")
    return f"{m:.4f}±{s:.4f}"


def generate_summary_plots_multiseed(all_agg: dict, out_dir: str):
    """Cross-maze comparison with error bars from multi-seed aggregation.

    all_agg: {maze_name: aggregated_dict} where aggregated_dict has keys
    like 'probes.h.mlp_R2' -> {mean, std, values}.
    """
    os.makedirs(out_dir, exist_ok=True)
    mazes = list(all_agg.keys())

    feat_set = set()
    for m in mazes:
        for key in all_agg[m]:
            parts = key.split(".")
            if parts[0] == "probes" and len(parts) >= 2:
                feat_set.add(parts[1])
    base_order = ["h", "s", "h+s", "encoder_e", "pix_pca64"]
    extra = sorted([f for f in feat_set if f not in base_order])
    feat_names = [f for f in base_order if f in feat_set] + extra
    n_feat = max(1, len(feat_names))

    x = np.arange(len(mazes))
    w = min(0.14, 0.80 / n_feat)

    # --- Summary 1: MLP R² ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, fn in enumerate(feat_names):
        means = [_get_agg(all_agg[m], f"probes.{fn}.mlp_R2") for m in mazes]
        stds = [_get_agg(all_agg[m], f"probes.{fn}.mlp_R2", "std") for m in mazes]
        ax.bar(x + (i - (n_feat - 1) / 2) * w, means, w, yerr=stds, capsize=3, label=fn)
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("MLP Probe R² (mean±std)"); ax.set_title("Position Decoding R²")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_probe_r2.png"), dpi=150)
    plt.close(fig)

    # --- Summary 2: Spearman ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, fn in enumerate(feat_names):
        means = [_get_agg(all_agg[m], f"distances.{fn}.spearman_rho") for m in mazes]
        stds = [_get_agg(all_agg[m], f"distances.{fn}.spearman_rho", "std") for m in mazes]
        ax.bar(x + (i - (n_feat - 1) / 2) * w, means, w, yerr=stds, capsize=3, label=fn)
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("Spearman ρ (mean±std)")
    ax.set_title("Distance Preservation (latent vs geodesic)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_spearman.png"), dpi=150)
    plt.close(fig)

    # --- Summary 3: kNN overlap ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, fn in enumerate(feat_names):
        means = [_get_agg(all_agg[m], f"knn.{fn}") for m in mazes]
        stds = [_get_agg(all_agg[m], f"knn.{fn}", "std") for m in mazes]
        ax.bar(x + (i - (n_feat - 1) / 2) * w, means, w, yerr=stds, capsize=3, label=fn)
    chance_vals = [_get_agg(all_agg[m], "knn._chance_baseline") for m in mazes]
    if any(c > 0 for c in chance_vals):
        ax.axhline(np.mean(chance_vals), color="red", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("kNN Overlap (mean±std)"); ax.set_title("Neighbourhood Preservation")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_knn.png"), dpi=150)
    plt.close(fig)

    # --- Summary 4: Trustworthiness ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, fn in enumerate(feat_names):
        means = [_get_agg(all_agg[m], f"trust_cont.{fn}.trustworthiness") for m in mazes]
        stds = [_get_agg(all_agg[m], f"trust_cont.{fn}.trustworthiness", "std") for m in mazes]
        ax.bar(x + (i - (n_feat - 1) / 2) * w, means, w, yerr=stds, capsize=3, label=fn)
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("Trustworthiness (mean±std)"); ax.set_title("Manifold Trustworthiness (k=knn_k)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_trustworthiness.png"), dpi=150)
    plt.close(fig)

    # --- Summary 5: Continuity ---
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, fn in enumerate(feat_names):
        means = [_get_agg(all_agg[m], f"trust_cont.{fn}.continuity") for m in mazes]
        stds = [_get_agg(all_agg[m], f"trust_cont.{fn}.continuity", "std") for m in mazes]
        ax.bar(x + (i - (n_feat - 1) / 2) * w, means, w, yerr=stds, capsize=3, label=fn)
    ax.set_xticks(x); ax.set_xticklabels(mazes, rotation=15)
    ax.set_ylabel("Continuity (mean±std)"); ax.set_title("Manifold Continuity (k=knn_k)")
    ax.set_ylim(0, 1.05); ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "summary_continuity.png"), dpi=150)
    plt.close(fig)


# ===================================================================
#  9. Main
# ===================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Maze latent-geometry evaluation")
    p.add_argument("--mazes", nargs="+", default=list(MAZE_LAYOUTS.keys()),
                   help="Which maze layouts to test")
    p.add_argument("--output_dir", default="maze_geometry_results")
    p.add_argument("--quick", action="store_true", help="Fast sanity-check run")
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                   help="Random seeds to run (default: 5 seeds)")
    p.add_argument(
        "--geo_supervised",
        action="store_true",
        help="Also train a GeoEncoder supervised by ground-truth geodesic distances",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()

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
        cfg.geo_sup_epochs = 80
        cfg.geo_sup_batch = 192
        cfg.geo_sup_candidates = 48

    cfg._do_geo_supervised = bool(args.geo_supervised)

    n_seeds = len(args.seeds)
    print(f"Device: {device}")
    print(f"Mazes:  {args.mazes}")
    print(f"Seeds:  {args.seeds}  ({n_seeds} seeds)")
    print(f"Quick:  {args.quick}\n")

    # {maze -> [per_seed_results]}
    all_seed_results: Dict[str, List[dict]] = {m: [] for m in args.mazes}

    for seed in args.seeds:
        for maze_name in args.mazes:
            results = run_single_maze(maze_name, cfg, device, args.output_dir, seed=seed)
            all_seed_results[maze_name].append(results)

    # Aggregate across seeds
    all_agg: Dict[str, dict] = {}
    for maze_name in args.mazes:
        all_agg[maze_name] = _aggregate_seed_results(all_seed_results[maze_name])

    summary_dir = os.path.join(args.output_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)

    generate_summary_plots_multiseed(all_agg, summary_dir)

    with open(os.path.join(summary_dir, "all_metrics_aggregated.json"), "w") as f:
        json.dump(all_agg, f, indent=2)
    with open(os.path.join(summary_dir, "all_metrics_per_seed.json"), "w") as f:
        json.dump({m: sr for m, sr in zip(args.mazes, [all_seed_results[m] for m in args.mazes])}, f, indent=2)

    # ---- Print summary table (mean±std) ----
    print(f"\n{'='*120}")
    print(f"  SUMMARY  ({n_seeds} seeds: {args.seeds})")
    print(f"{'='*120}")
    header = (f"{'Maze':12s} | {'Feature':12s} | {'Lin R²':14s} | {'MLP R²':14s} | "
              f"{'Pearson':14s} | {'Spearman':14s} | {'kNN':14s} | {'Trust':14s} | {'Cont':14s}")
    print(header)
    print("-" * len(header))
    for maze in args.mazes:
        agg = all_agg[maze]
        feat_set = set()
        for key in agg:
            parts = key.split(".")
            if parts[0] == "probes" and len(parts) >= 2:
                feat_set.add(parts[1])
        base = ["h", "s", "h+s", "encoder_e", "pix_pca64"]
        extra = sorted([f for f in feat_set if f not in base])
        feat_names = [f for f in base if f in feat_set] + extra
        for fn in feat_names:
            lin = _fmt_mean_std(agg, f"probes.{fn}.linear_R2")
            mlp = _fmt_mean_std(agg, f"probes.{fn}.mlp_R2")
            pr = _fmt_mean_std(agg, f"distances.{fn}.pearson_r")
            sr = _fmt_mean_std(agg, f"distances.{fn}.spearman_rho")
            knn = _fmt_mean_std(agg, f"knn.{fn}")
            tr = _fmt_mean_std(agg, f"trust_cont.{fn}.trustworthiness")
            co = _fmt_mean_std(agg, f"trust_cont.{fn}.continuity")
            print(f"{maze:12s} | {fn:12s} | {lin:14s} | {mlp:14s} | {pr:14s} | {sr:14s} | {knn:14s} | {tr:14s} | {co:14s}")
        chance = _get_agg(agg, "knn._chance_baseline")
        if chance > 0:
            print(f"{'':12s} | {'(chance)':12s} | {'':14s} | {'':14s} | {'':14s} | {'':14s} | {chance:14.6f} | {'':14s} | {'':14s}")
        print("-" * len(header))

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
