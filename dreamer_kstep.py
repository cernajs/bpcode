#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from maze_geometry_test import _positions_to_cell_indices
from models import RSSM, Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel, ValueModel
from pointmaze_large_topo_v2 import PointMazeLargeDiverseGRWrapper
from utils import ReplayBuffer, bottle, get_device, no_param_grads, preprocess_img, set_seed


# =====================================================================
# Env wrapper (unchanged from dreamer_kstep.py)
# =====================================================================


class PointMazeLargeDreamerWrapper(PointMazeLargeDiverseGRWrapper):
    def __init__(self, img_size=64, *, reset_mode="fixed_start",
                 fixed_start_cell=None, start_cells=None):
        super().__init__(img_size=img_size)
        self.reset_mode = str(reset_mode)
        self.fixed_start_cell = fixed_start_cell
        self.start_cells = list(start_cells) if start_cells else []
        if self.reset_mode == "fixed_start" and self.fixed_start_cell is None:
            self.fixed_start_cell = tuple(self._free_cells[0]) if self._free_cells else (1, 1)
        if self.reset_mode == "start_subset" and not self.start_cells:
            if self._free_cells:
                self.start_cells = [tuple(self._free_cells[i]) for i in range(min(3, len(self._free_cells)))]
            else:
                self.start_cells = [(1, 1)]

    def _sample_goal_cell(self, sr, sc):
        cells = self._free_cells
        if not cells:
            return np.array([sr, sc], dtype=np.int64)
        candidates = [c for c in cells if c != (sr, sc)]
        if not candidates:
            return np.array([sr, sc], dtype=np.int64)
        gr, gc = candidates[int(np.random.randint(0, len(candidates)))]
        return np.array([gr, gc], dtype=np.int64)

    def reset(self, **kwargs):
        if self.reset_mode == "random":
            return super().reset(**kwargs)
        opts = dict(kwargs.get("options") or {})
        if self.reset_mode == "fixed_start":
            r, c = self.fixed_start_cell
        elif self.reset_mode == "start_subset":
            r, c = self.start_cells[int(np.random.randint(0, len(self.start_cells)))]
        else:
            raise ValueError(f"Unknown reset_mode {self.reset_mode!r}")
        opts["reset_cell"] = np.array([r, c], dtype=np.int64)
        opts["goal_cell"] = self._sample_goal_cell(int(r), int(c))
        kwargs = dict(kwargs)
        kwargs["options"] = opts
        obs_dict, info = self._env.reset(**kwargs)
        self._update_agent_pos(obs_dict)
        frame = self._resize_frame(self._env.render()).astype(np.uint8)
        return frame, info


# =====================================================================
# Geometry Head with anti-collapse
# =====================================================================


class GeoHead(nn.Module):
    """Projects Dreamer latent (h, s) -> compact geometric embedding.

    Trained with temporal contrastive loss. Anti-collapse built in:
    - L2-normalized output (lives on hypersphere, prevents scale collapse)
    - LayerNorm between hidden layers (stabilizes training)
    - Trained with separate optimizer (doesn't corrupt world model)
    """

    def __init__(self, h_dim: int, s_dim: int, geo_dim: int = 32, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(h_dim + s_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, geo_dim),
        )
        self.geo_dim = geo_dim

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, s], dim=-1)
        g = self.net(x)
        return F.normalize(g, dim=-1)


def geo_temporal_contrastive_loss(
    g_seq: torch.Tensor,
    done_seq: torch.Tensor | None = None,
    *,
    pos_k: int = 4,
    n_neg: int = 64,
    temperature: float = 0.07,
    max_anchors: int = 256,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InfoNCE on geo-head outputs. Separate from k-step loss on raw z."""
    B, T, D = g_seq.shape
    if T < 2:
        return g_seq.new_zeros(()), {"geo_nce/loss": 0.0, "geo_nce/n_pairs": 0.0}

    flat = g_seq.reshape(B * T, D)
    pairs = []
    for b in range(B):
        for t in range(T - 1):
            max_delta = min(pos_k, T - 1 - t)
            if max_delta < 1:
                continue
            valid = []
            for d in range(1, max_delta + 1):
                if done_seq is not None and torch.any(done_seq[b, t:t + d] > 0.5):
                    break
                valid.append(d)
            if not valid:
                continue
            d = valid[int(torch.randint(0, len(valid), (1,)).item())]
            pairs.append((b * T + t, b * T + t + d))

    if not pairs:
        return g_seq.new_zeros(()), {"geo_nce/loss": 0.0, "geo_nce/n_pairs": 0.0}

    if len(pairs) > max_anchors:
        perm = torch.randperm(len(pairs))[:max_anchors].tolist()
        pairs = [pairs[i] for i in perm]

    anc_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=g_seq.device)
    pos_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=g_seq.device)
    anchors = flat[anc_idx]
    positives = flat[pos_idx]

    neg_idx = torch.randint(0, flat.size(0), (len(pairs), n_neg), device=g_seq.device)
    negatives = flat[neg_idx]

    pos_logits = (anchors * positives).sum(-1, keepdim=True) / temperature
    neg_logits = torch.einsum("bd,bnd->bn", anchors, negatives) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    targets = torch.zeros(len(pairs), dtype=torch.long, device=g_seq.device)
    loss = F.cross_entropy(logits, targets)

    return loss, {
        "geo_nce/loss": float(loss.item()),
        "geo_nce/n_pairs": float(len(pairs)),
        "geo_nce/pos_sim": float((anchors * positives).sum(-1).mean().item()),
        "geo_nce/neg_sim": float(torch.einsum("bd,bnd->bn", anchors, negatives).mean().item()),
    }


def geo_anticollapse_loss(
    g: torch.Tensor,
    var_target: float = 1.0,
    var_weight: float = 5.0,
    unif_weight: float = 1.0,
    cov_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """VICReg-style anti-collapse on geo-head embeddings.

    1. Variance hinge: per-dim std should exceed var_target
    2. Uniformity: log-sum-exp repulsion on hypersphere (Wang & Isola 2020)
    3. Covariance: decorrelate dimensions
    """
    N, D = g.shape
    if N < 4:
        return g.new_zeros(()), {
            "geo_ac/var": 0.0, "geo_ac/unif": 0.0, "geo_ac/cov": 0.0,
            "geo_ac/mean_std": 0.0, "geo_ac/min_std": 0.0, "geo_ac/eff_rank": 0.0,
        }

    std = g.std(dim=0)
    var_loss = F.relu(var_target - std).mean()

    sq_dist = torch.cdist(g, g).pow(2)
    mask = ~torch.eye(N, device=g.device, dtype=torch.bool)
    unif_loss = torch.logsumexp(-2.0 * sq_dist[mask].view(N, N - 1), dim=1).mean()

    g_c = g - g.mean(dim=0, keepdim=True)
    cov = (g_c.T @ g_c) / max(N - 1, 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag / max(D * (D - 1), 1)

    total = var_weight * var_loss + unif_weight * unif_loss + cov_weight * cov_loss

    try:
        s = torch.linalg.svdvals(g_c)
        s_norm = s / s.sum().clamp(min=1e-8)
        eff_rank = float(torch.exp(-(s_norm * torch.log(s_norm + 1e-10)).sum()).item())
    except Exception:
        eff_rank = 0.0

    return total, {
        "geo_ac/var": float(var_loss.item()),
        "geo_ac/unif": float(unif_loss.item()),
        "geo_ac/cov": float(cov_loss.item()),
        "geo_ac/total": float(total.item()),
        "geo_ac/mean_std": float(std.mean().item()),
        "geo_ac/min_std": float(std.min().item()),
        "geo_ac/eff_rank": eff_rank,
    }


# =====================================================================
# Embedding Memory Bank
# =====================================================================


class GeoMemoryBank:
    """Fixed-size FIFO bank of geo-embeddings for intrinsic reward computation."""

    def __init__(self, capacity: int, geo_dim: int, device: torch.device):
        self.capacity = capacity
        self.embeddings = torch.zeros(capacity, geo_dim, device=device)
        self.cell_ids = -torch.ones(capacity, dtype=torch.long, device=device)
        self.ptr = 0
        self.full = False

    @property
    def size(self) -> int:
        return self.capacity if self.full else self.ptr

    def add(self, g: torch.Tensor, cells: torch.Tensor | None = None) -> None:
        N = g.size(0)
        if N == 0:
            return
        g = g.detach()
        if N >= self.capacity:
            self.embeddings[:] = g[-self.capacity:]
            if cells is not None:
                self.cell_ids[:] = cells[-self.capacity:].detach()
            self.ptr = 0
            self.full = True
            return
        end = self.ptr + N
        if end <= self.capacity:
            self.embeddings[self.ptr:end] = g
            if cells is not None:
                self.cell_ids[self.ptr:end] = cells.detach()
            self.ptr = end
        else:
            first = self.capacity - self.ptr
            self.embeddings[self.ptr:] = g[:first]
            self.embeddings[:N - first] = g[first:]
            if cells is not None:
                self.cell_ids[self.ptr:] = cells[:first].detach()
                self.cell_ids[:N - first] = cells[first:].detach()
            self.ptr = N - first
        if end >= self.capacity:
            self.full = True

    def get_all(self) -> torch.Tensor:
        return self.embeddings if self.full else self.embeddings[:self.ptr]

    def get_cell_counts(self) -> dict[int, int]:
        valid = self.cell_ids[:self.size]
        counts: dict[int, int] = {}
        for c in valid.cpu().tolist():
            if c >= 0:
                counts[c] = counts.get(c, 0) + 1
        return counts


# =====================================================================
# Intrinsic Reward
# =====================================================================


@torch.no_grad()
def compute_geo_intrinsic_reward(
    geo_head: GeoHead,
    h_imag: torch.Tensor,
    s_imag: torch.Tensor,
    memory: GeoMemoryBank,
    *,
    knn_k: int = 8,
    frontier_weight: float = 0.5,
    distortion_weight: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Intrinsic reward for imagined trajectories.

    reward = distortion (knn dist in geo-space) + frontier (inverse visit density).
    Returns [B, H] rewards (excluding t=0 start state).
    """
    B, Hp1, _ = h_imag.shape
    H = Hp1 - 1
    if memory.size < knn_k + 1 or H < 1:
        return torch.zeros(B, max(H, 1), device=h_imag.device)

    g_imag = geo_head(h_imag[:, 1:].reshape(B * H, -1), s_imag[:, 1:].reshape(B * H, -1))
    g_mem = memory.get_all()

    if distortion_weight > 0.0:
        dists = torch.cdist(g_imag, g_mem)
        knn_k_actual = min(knn_k, dists.size(1))
        knn_dists, knn_idx = torch.topk(dists, knn_k_actual, dim=1, largest=False)
        distortion = knn_dists.mean(dim=1)
    else:
        distortion = torch.zeros_like(g_imag[:, 0])

    cell_counts = memory.get_cell_counts()
    nn_cells = memory.cell_ids[knn_idx[:, 0]]
    if frontier_weight > 0.0:
        frontier = torch.tensor(
            [1.0 / math.sqrt(cell_counts.get(int(c), 0) + 1.0) for c in nn_cells.cpu().tolist()],
            device=h_imag.device, dtype=torch.float32,
        )
    else:
        frontier = torch.zeros_like(g_imag[:, 0])

    reward = distortion_weight * distortion + frontier_weight * frontier
    reward = reward.reshape(B, H)

    if normalize and reward.numel() > 1:
        reward = (reward - reward.mean()) / (reward.std() + 1e-8)

    return reward


# =====================================================================
# Original losses/helpers (unchanged from dreamer_kstep.py)
# =====================================================================


def kstep_reachability_nce_loss(
    z_seq, done_seq=None, *, k_max=5, n_neg=64, temperature=0.1, max_anchors=256,
):
    B, T, D = z_seq.shape
    if T < 2:
        zero = z_seq.new_zeros(())
        return zero, {"kstep/loss": 0.0, "kstep/n_pairs": 0.0, "kstep/pos_sim": 0.0, "kstep/neg_sim": 0.0}
    z = F.normalize(z_seq, dim=-1)
    flat = z.reshape(B * T, D)
    pairs = []
    for b in range(B):
        for t in range(T - 1):
            max_delta = min(int(k_max), T - 1 - t)
            if max_delta < 1:
                continue
            valid_deltas = []
            for d in range(1, max_delta + 1):
                if done_seq is not None and torch.any(done_seq[b, t:t + d] > 0.5):
                    break
                valid_deltas.append(d)
            if not valid_deltas:
                continue
            d = valid_deltas[int(torch.randint(0, len(valid_deltas), (1,), device=z_seq.device).item())]
            pairs.append((b * T + t, b * T + t + d))
    if not pairs:
        return z_seq.new_zeros(()), {"kstep/loss": 0.0, "kstep/n_pairs": 0.0, "kstep/pos_sim": 0.0, "kstep/neg_sim": 0.0}
    if len(pairs) > max_anchors:
        perm = torch.randperm(len(pairs), device=z_seq.device)[:max_anchors].tolist()
        pairs = [pairs[i] for i in perm]
    anchor_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=z_seq.device)
    pos_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=z_seq.device)
    anchors = flat[anchor_idx]
    positives = flat[pos_idx]
    neg_idx = torch.randint(0, flat.size(0), (anchors.size(0), n_neg), device=z_seq.device)
    negatives = flat[neg_idx]
    pos_logits = (anchors * positives).sum(-1, keepdim=True) / max(float(temperature), 1e-6)
    neg_logits = torch.einsum("bd,bnd->bn", anchors, negatives) / max(float(temperature), 1e-6)
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    targets = torch.zeros(anchors.size(0), dtype=torch.long, device=z_seq.device)
    loss = F.cross_entropy(logits, targets)
    return loss, {
        "kstep/loss": float(loss.item()), "kstep/n_pairs": float(anchors.size(0)),
        "kstep/pos_sim": float((anchors * positives).sum(-1).mean().item()),
        "kstep/neg_sim": float(torch.einsum("bd,bnd->bn", anchors, negatives).mean().item()),
    }


def rssm_latent(h, s):
    return torch.cat([h, s], dim=-1)


@torch.no_grad()
def compute_latent_diagnostics(z, prefix="latent", max_samples=2048):
    N = z.shape[0]
    idx = np.random.choice(N, size=min(max_samples, N), replace=False)
    z = z[idx]
    dm = torch.cdist(z, z)
    triu_mask = torch.triu(torch.ones_like(dm, dtype=torch.bool), diagonal=1)
    dists = dm[triu_mask]
    per_dim_std = z.std(dim=0)
    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / max(z.shape[0] - 1, 1)
    diag = cov.diagonal()
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    off_diag_norm = (off_diag_sq / max(cov.shape[0] ** 2 - cov.shape[0], 1)).sqrt()
    try:
        s = torch.linalg.svdvals(z_c)
        s_norm = s / s.sum().clamp(min=1e-8)
        entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
        eff_rank = torch.exp(entropy)
    except Exception:
        eff_rank = torch.tensor(0.0)
    return {
        f"{prefix}/mean_pairwise_dist": float(dists.mean().item()),
        f"{prefix}/std_pairwise_dist": float(dists.std().item()),
        f"{prefix}/min_pairwise_dist": float(dists.min().item()),
        f"{prefix}/median_pairwise_dist": float(dists.median().item()),
        f"{prefix}/mean_per_dim_std": float(per_dim_std.mean().item()),
        f"{prefix}/min_per_dim_std": float(per_dim_std.min().item()),
        f"{prefix}/max_per_dim_std": float(per_dim_std.max().item()),
        f"{prefix}/num_dead_dims": float((per_dim_std < 0.01).sum().item()),
        f"{prefix}/off_diag_cov": float(off_diag_norm.item()),
        f"{prefix}/effective_rank": float(eff_rank.item()),
        f"{prefix}/mean_norm": float(z.norm(dim=-1).mean().item()),
    }


@torch.no_grad()
def compute_hs_geodesic_correlation(encoder, rssm, obs_buffer, pos_buffer, geodesic,
                                     device, bit_depth, n_pairs=2000):
    from scipy.stats import spearmanr
    encoder.eval(); rssm.eval()
    N = len(pos_buffer)
    if N < 20:
        return {"geo_hs/spearman": 0.0, "geo_hs/n_pairs": 0}
    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    ii = rng.integers(0, N, size=n_pairs)
    jj = rng.integers(0, N, size=n_pairs)
    valid = ii != jj; ii, jj = ii[valid], jj[valid]
    def encode_batch(indices):
        latents = []
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            batch_idx = indices[start:end]
            obs_batch = torch.tensor(obs_buffer[batch_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            preprocess_img(obs_batch, depth=bit_depth)
            e_batch = encoder(obs_batch)
            h_batch, s_batch = rssm.get_init_state(e_batch)
            latents.append(rssm_latent(h_batch, s_batch).cpu().numpy())
        return np.concatenate(latents, axis=0)
    z_i = encode_batch(ii); z_j = encode_batch(jj)
    d_hs = np.linalg.norm(z_i - z_j, axis=-1)
    cell_i = _positions_to_cell_indices(geodesic, pos_buffer[ii])
    cell_j = _positions_to_cell_indices(geodesic, pos_buffer[jj])
    d_geo = np.array([float(geodesic.dist_matrix[int(ci), int(cj)]) for ci, cj in zip(cell_i, cell_j)], dtype=np.float32)
    finite = np.isfinite(d_geo) & (d_geo > 0) & np.isfinite(d_hs) & (d_hs > 0)
    if finite.sum() < 10:
        return {"geo_hs/spearman": 0.0, "geo_hs/n_pairs": 0}
    rho = spearmanr(d_hs[finite], d_geo[finite]).correlation
    if rho is None or not np.isfinite(rho): rho = 0.0
    return {"geo_hs/spearman": float(rho), "geo_hs/n_pairs": int(finite.sum()),
            "geo_hs/mean_latent_dist": float(np.mean(d_hs[finite])), "geo_hs/mean_geo_dist": float(np.mean(d_geo[finite]))}


@torch.no_grad()
def compute_geohead_geodesic_correlation(
    geo_head, encoder, rssm, obs_buffer, pos_buffer, geodesic, device, bit_depth, n_pairs=2000,
):
    """THE KEY METRIC: geo-head embedding distance vs oracle geodesic distance.

    If this stays high across training, the self-reinforcing loop is working.
    """
    from scipy.stats import spearmanr
    encoder.eval(); rssm.eval(); geo_head.eval()
    N = len(pos_buffer)
    if N < 20:
        return {"geo_head/spearman": 0.0, "geo_head/n_pairs": 0}
    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    ii = rng.integers(0, N, size=n_pairs)
    jj = rng.integers(0, N, size=n_pairs)
    valid = ii != jj; ii, jj = ii[valid], jj[valid]
    def encode_batch(indices):
        latents = []
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            batch_idx = indices[start:end]
            obs_batch = torch.tensor(obs_buffer[batch_idx], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
            preprocess_img(obs_batch, depth=bit_depth)
            e_batch = encoder(obs_batch)
            h_batch, s_batch = rssm.get_init_state(e_batch)
            latents.append(geo_head(h_batch, s_batch).cpu().numpy())
        return np.concatenate(latents, axis=0)
    g_i = encode_batch(ii); g_j = encode_batch(jj)
    d_g = np.linalg.norm(g_i - g_j, axis=-1)
    cell_i = _positions_to_cell_indices(geodesic, pos_buffer[ii])
    cell_j = _positions_to_cell_indices(geodesic, pos_buffer[jj])
    d_geo = np.array([float(geodesic.dist_matrix[int(ci), int(cj)]) for ci, cj in zip(cell_i, cell_j)], dtype=np.float32)
    finite = np.isfinite(d_geo) & (d_geo > 0) & np.isfinite(d_g) & (d_g > 0)
    if finite.sum() < 10:
        return {"geo_head/spearman": 0.0, "geo_head/n_pairs": 0}
    rho = spearmanr(d_g[finite], d_geo[finite]).correlation
    if rho is None or not np.isfinite(rho): rho = 0.0
    return {"geo_head/spearman": float(rho), "geo_head/n_pairs": int(finite.sum()),
            "geo_head/mean_embed_dist": float(np.mean(d_g[finite])), "geo_head/mean_geo_dist": float(np.mean(d_geo[finite]))}


def compute_lambda_returns(rewards, values, discounts, lambda_=0.95):
    if not torch.is_tensor(discounts):
        discounts = torch.full_like(rewards, float(discounts))
    else:
        if discounts.dim() == 1:
            discounts = discounts.unsqueeze(0).expand_as(rewards)
        discounts = discounts.to(dtype=rewards.dtype, device=rewards.device)
    B, H = rewards.shape
    next_values = values[:, 1:]
    last = values[:, -1]
    out = torch.zeros_like(rewards)
    for t in reversed(range(H)):
        bootstrap = (1.0 - lambda_) * next_values[:, t] + lambda_ * last
        last = rewards[:, t] + discounts[:, t] * bootstrap
        out[:, t] = last
    return out


def compute_discount_weights(discounts):
    B, H = discounts.shape
    ones = torch.ones((B, 1), device=discounts.device, dtype=discounts.dtype)
    return torch.cumprod(torch.cat([ones, discounts], dim=1), dim=1)[:, :-1]


def _bridge_crossing_count(geodesic, pos_seq):
    from maze_geometry_test import _adj_from_distmat, _find_bridges
    adj = _adj_from_distmat(geodesic.dist_matrix)
    bridges = set(_find_bridges(adj))
    if len(pos_seq) < 2: return 0
    cells = _positions_to_cell_indices(geodesic, pos_seq)
    return sum(1 for a, b in zip(cells[:-1], cells[1:])
               if ((int(a), int(b)) if int(a) < int(b) else (int(b), int(a))) in bridges)


@torch.no_grad()
def run_periodic_eval(env, encoder, rssm, actor, device, bit_depth, n_episodes, geodesic, action_repeat):
    encoder.eval(); rssm.eval(); actor.eval()
    rets, lens, uniques, bridges, successes = [], [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret, ep_steps = 0.0, 0
        ep_success = False
        cells = set()
        pos_traj = []
        c0 = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
        cells.add(c0); pos_traj.append(env.agent_pos.copy())
        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=bit_depth)
        e0 = encoder(obs_t)
        h_state, s_state = rssm.get_init_state(e0)
        while not done:
            action_t, _ = actor.get_action(h_state, s_state, deterministic=True)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, r, term, trunc, info = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            ep_ret += float(r); ep_steps += 1
            if info.get("success", False) or info.get("is_success", False): ep_success = True
            c = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
            cells.add(c); pos_traj.append(env.agent_pos.copy())
            obs_t = torch.tensor(np.ascontiguousarray(next_obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=bit_depth)
            e = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(e, act_t, h_state, s_state, sample=False)
        rets.append(ep_ret); lens.append(ep_steps); uniques.append(float(len(cells)))
        successes.append(1.0 if ep_success else 0.0)
        bridges.append(float(_bridge_crossing_count(geodesic, np.stack(pos_traj))) if len(pos_traj) >= 2 else 0.0)
    return {
        "return_mean": float(np.mean(rets)), "return_std": float(np.std(rets)),
        "length_mean": float(np.mean(lens)), "unique_cells_mean": float(np.mean(uniques)),
        "bridge_crossings_mean": float(np.mean(bridges)), "success_rate": float(np.mean(successes)),
    }


# =====================================================================
# CLI
# =====================================================================


def build_parser():
    p = argparse.ArgumentParser(description="Dreamer + K-step InfoNCE + Online Geo Head + Intrinsic Exploration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--max_episodes", type=int, default=400)
    p.add_argument("--seed_episodes", type=int, default=5)
    p.add_argument("--collect_interval", type=int, default=50)
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--replay_capacity", type=int, default=100_000)
    p.add_argument("--model_lr", type=float, default=6e-4)
    p.add_argument("--actor_lr", type=float, default=8e-5)
    p.add_argument("--value_lr", type=float, default=8e-5)
    p.add_argument("--adam_eps", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=100.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda_", type=float, default=0.95)
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--actor_hidden_dim", type=int, default=400)
    p.add_argument("--value_hidden_dim", type=int, default=400)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--kl_free_nats", type=float, default=3.0)
    p.add_argument("--imagination_horizon", type=int, default=15)
    p.add_argument("--imagination_starts", type=int, default=8)
    p.add_argument("--expl_amount", type=float, default=0.3)
    p.add_argument("--expl_decay", type=float, default=0.0)
    p.add_argument("--expl_min", type=float, default=0.0)
    p.add_argument("--actor_entropy_scale", type=float, default=1e-3)
    p.add_argument("--reset_mode", type=str, default="fixed_start",
                   choices=["fixed_start", "start_subset", "random"])
    p.add_argument("--fixed_start_row", type=int, default=-1)
    p.add_argument("--fixed_start_col", type=int, default=-1)
    p.add_argument("--start_subset", type=str, default="")
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--eval_episodes", type=int, default=5)

    # --- Geo mode ---
    p.add_argument("--geo_mode", type=str, default="kstep_explore",
                   choices=["baseline", "kstep", "kstep_explore"])

    # K-step InfoNCE on raw z (shapes RSSM latent)
    p.add_argument("--kstep_weight", type=float, default=0.1)
    p.add_argument("--kstep_max_k", type=int, default=5)
    p.add_argument("--kstep_negatives", type=int, default=64)
    p.add_argument("--kstep_temperature", type=float, default=0.1)
    p.add_argument("--kstep_max_anchors", type=int, default=256)
    p.add_argument("--kstep_min_steps", type=int, default=50_000)

    # Geo head
    p.add_argument("--geo_dim", type=int, default=32)
    p.add_argument("--geo_head_hidden", type=int, default=256)
    p.add_argument("--geo_head_lr", type=float, default=3e-4)
    p.add_argument("--geo_nce_temperature", type=float, default=0.07)
    p.add_argument("--geo_nce_pos_k", type=int, default=4)
    p.add_argument("--geo_nce_weight", type=float, default=1.0)
    p.add_argument("--geo_anticollapse_weight", type=float, default=0.5)
    p.add_argument("--geo_var_weight", type=float, default=5.0)
    p.add_argument("--geo_unif_weight", type=float, default=1.0)
    p.add_argument("--geo_cov_weight", type=float, default=1.0)
    p.add_argument("--geo_head_min_steps", type=int, default=30_000)

    # Intrinsic reward
    p.add_argument("--geo_intrinsic_weight", type=float, default=0.3)
    p.add_argument("--geo_intrinsic_min_steps", type=int, default=60_000)
    p.add_argument("--geo_intrinsic_warmup", type=int, default=10_000)
    p.add_argument("--geo_memory_size", type=int, default=8192)
    p.add_argument("--geo_intrinsic_knn_k", type=int, default=8)
    p.add_argument("--geo_frontier_weight", type=float, default=0.0)
    p.add_argument("--geo_distortion_weight", type=float, default=1.0)

    # Diagnostics
    p.add_argument("--diag_interval", type=int, default=40)
    p.add_argument("--geo_corr_interval", type=int, default=80)

    p.add_argument("--wm_path", type=str, default="")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="kstep_explore")
    p.add_argument("--save_interval", type=int, default=100)
    return p


# =====================================================================
# Main training loop
# =====================================================================


def main(args):
    set_seed(args.seed)
    device = get_device()

    use_kstep = args.geo_mode in ("kstep", "kstep_explore")
    use_geo_explore = args.geo_mode == "kstep_explore"

    print(f"Device: {device}")
    print(f"Geo mode: {args.geo_mode}")
    print(f"  K-step InfoNCE on z: {use_kstep} (weight={args.kstep_weight})")
    print(f"  Geo head + intrinsic explore: {use_geo_explore}")
    if use_geo_explore:
        print(f"    geo_dim={args.geo_dim}  lr={args.geo_head_lr}")
        print(f"    intrinsic_weight={args.geo_intrinsic_weight}  activates at step {args.geo_intrinsic_min_steps}")
        print(f"    anti-collapse: var_w={args.geo_var_weight} unif_w={args.geo_unif_weight} cov_w={args.geo_cov_weight}")

    # --- Warmup schedule ---
    # Phase 1 (0 -> kstep_min_steps):       WM only
    # Phase 2 (kstep_min_steps -> geo_head_min_steps): WM + kstep on raw z
    # Phase 3 (geo_head_min_steps -> geo_intrinsic_min_steps): + geo head trains
    # Phase 4 (geo_intrinsic_min_steps ->):  + intrinsic reward feeds actor
    print(f"\n  Schedule:")
    print(f"    Phase 1: WM only (steps 0-{args.kstep_min_steps})")
    if use_kstep:
        print(f"    Phase 2: + k-step InfoNCE on z (steps {args.kstep_min_steps}+)")
    if use_geo_explore:
        print(f"    Phase 3: + geo head training (steps {args.geo_head_min_steps}+)")
        print(f"    Phase 4: + intrinsic reward (steps {args.geo_intrinsic_min_steps}+)")

    start_cells = None
    if args.reset_mode == "start_subset" and args.start_subset.strip():
        start_cells = []
        for part in args.start_subset.split(";"):
            part = part.strip()
            if not part: continue
            r, c = part.split(",")
            start_cells.append((int(r.strip()), int(c.strip())))

    fixed_cell = None
    if args.reset_mode == "fixed_start":
        if args.fixed_start_row >= 0 and args.fixed_start_col >= 0:
            fixed_cell = (args.fixed_start_row, args.fixed_start_col)

    env = PointMazeLargeDreamerWrapper(
        img_size=args.img_size, reset_mode=args.reset_mode,
        fixed_start_cell=fixed_cell, start_cells=start_cells,
    )
    geodesic = env.geodesic
    print(f"  Maze: free_cells={geodesic.n_free}  reset={args.reset_mode}")

    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    action_repeat = 1
    effective_gamma = args.gamma ** action_repeat

    encoder = ConvEncoder(embedding_size=args.embed_dim, in_channels=C).to(device)
    decoder = ConvDecoder(args.deter_dim, args.stoch_dim, embedding_size=args.embed_dim, out_channels=C).to(device)
    rssm = RSSM(args.stoch_dim, args.deter_dim, act_dim, args.embed_dim, args.hidden_dim).to(device)
    reward_model = RewardModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(device)
    cont_model = ContinueModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(device)

    if args.wm_path:
        ckpt = torch.load(args.wm_path, map_location=device)
        for k in ("encoder", "decoder", "rssm", "reward_model", "cont_model"):
            if k in ckpt:
                locals()[k].load_state_dict(ckpt[k])
        print(f"  Loaded WM from {args.wm_path}")

    actor = Actor(args.deter_dim, args.stoch_dim, act_dim, args.actor_hidden_dim).to(device)
    value_model = ValueModel(args.deter_dim, args.stoch_dim, args.value_hidden_dim).to(device)

    world_params = (list(encoder.parameters()) + list(decoder.parameters())
                    + list(rssm.parameters()) + list(reward_model.parameters())
                    + list(cont_model.parameters()))
    model_opt = torch.optim.Adam(world_params, lr=args.model_lr, eps=args.adam_eps)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=args.adam_eps)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=args.value_lr, eps=args.adam_eps)

    # Geo head: SEPARATE optimizer to avoid corrupting world model gradients
    geo_head = None
    geo_opt = None
    geo_memory = None
    if use_geo_explore:
        geo_head = GeoHead(args.deter_dim, args.stoch_dim,
                           geo_dim=args.geo_dim, hidden_dim=args.geo_head_hidden).to(device)
        geo_opt = torch.optim.Adam(geo_head.parameters(), lr=args.geo_head_lr, eps=args.adam_eps)
        geo_memory = GeoMemoryBank(args.geo_memory_size, args.geo_dim, device)

    replay = ReplayBuffer(args.replay_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    tag = f"{args.run_name}_seed{args.seed}" if args.run_name else f"{args.geo_mode}_seed{args.seed}"
    writer = SummaryWriter(f"{args.log_dir}/{tag}")
    writer.add_text("hyperparameters", str(vars(args)), 0)
    out_dir = os.path.join(args.log_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    obs_geo_buffer: list[np.ndarray] = []
    pos_geo_buffer: list[np.ndarray] = []
    GEO_BUFFER_MAX = 8000

    total_steps = 0
    expl_amount = args.expl_amount

    print(f"  Seeding replay with {args.seed_episodes} episodes ...")
    for _ in range(args.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, r, term, trunc, _ = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            replay.add(obs=np.ascontiguousarray(obs, np.uint8), action=action.astype(np.float32),
                       reward=float(r), next_obs=np.ascontiguousarray(next_obs, np.uint8), done=done)
            obs = next_obs
            total_steps += 1

    first_goal_step = None
    cumulative_successes = 0
    print(f"\n  Training {args.max_episodes} episodes ...")

    for episode in range(args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret, ep_steps = 0.0, 0
        ep_success = False
        ep_cells: set[int] = set()
        ep_pos_traj: list[np.ndarray] = []
        c0 = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
        ep_cells.add(c0); ep_pos_traj.append(env.agent_pos.copy())

        with torch.no_grad():
            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=args.bit_depth)
            e0 = encoder(obs_t)
            h_state, s_state = rssm.get_init_state(e0)

        while not done:
            encoder.eval(); rssm.eval(); actor.eval()
            if geo_head is not None: geo_head.eval()
            with torch.no_grad():
                action_t, _ = actor.get_action(h_state, s_state, deterministic=False)
                if expl_amount > 0:
                    action_t = torch.clamp(action_t + expl_amount * torch.randn_like(action_t), -1.0, 1.0)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            next_obs, r, term, trunc, step_info = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            replay.add(obs=np.ascontiguousarray(obs, np.uint8), action=action,
                       reward=float(r), next_obs=np.ascontiguousarray(next_obs, np.uint8), done=done)
            obs = next_obs
            ep_ret += float(r); ep_steps += 1; total_steps += 1
            if step_info.get("success", False) or step_info.get("is_success", False):
                ep_success = True

            c = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
            ep_cells.add(c); ep_pos_traj.append(env.agent_pos.copy())

            if len(obs_geo_buffer) < GEO_BUFFER_MAX:
                obs_geo_buffer.append(np.ascontiguousarray(obs, np.uint8))
                pos_geo_buffer.append(env.agent_pos.copy())

            with torch.no_grad():
                obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                preprocess_img(obs_t, depth=args.bit_depth)
                e = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(e, act_t, h_state, s_state, sample=False)

            # ================================================================
            # TRAINING
            # ================================================================
            if total_steps % args.collect_interval == 0 and replay.size > (args.seq_len + 2):
                encoder.train(); decoder.train(); rssm.train()
                reward_model.train(); cont_model.train()
                actor.train(); value_model.train()
                if geo_head is not None: geo_head.train()

                sum_rec = sum_kld = sum_rew = sum_cont = sum_model = 0.0
                sum_actor = sum_value = sum_imag_r = sum_intrinsic_r = 0.0
                sum_kstep = sum_geo_nce = sum_geo_ac = 0.0
                kstep_info_accum: dict[str, float] = {}
                geo_info_accum: dict[str, float] = {}
                intrinsic_active = False

                for _ in range(args.train_steps):
                    batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)
                    obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                    act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)
                    rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)
                    done_seq = torch.tensor(batch.dones, dtype=torch.float32, device=device)

                    B, T1 = rew_seq.shape; T = T1 - 1
                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                    preprocess_img(x, depth=args.bit_depth)
                    e_t = bottle(encoder, x)

                    h_t, s_t = rssm.get_init_state(e_t[:, 0])
                    states, priors, posts, s_samples = [], [], [], []
                    for t in range(T):
                        h_t = rssm.deterministic_state_fwd(h_t, s_t, act_seq[:, t])
                        states.append(h_t)
                        priors.append(rssm.state_prior(h_t))
                        posts.append(rssm.state_posterior(h_t, e_t[:, t + 1]))
                        pm, ps = posts[-1]
                        s_t = pm + torch.randn_like(ps) * ps
                        s_samples.append(s_t)

                    h_seq = torch.stack(states, dim=1)
                    s_seq = torch.stack(s_samples, dim=1)
                    prior_m = torch.stack([p[0] for p in priors], dim=0)
                    prior_s = torch.stack([p[1] for p in priors], dim=0)
                    post_m = torch.stack([p[0] for p in posts], dim=0)
                    post_s = torch.stack([p[1] for p in posts], dim=0)

                    recon = bottle(decoder, h_seq, s_seq)
                    target = x[:, 1:T + 1]
                    rec_loss = F.mse_loss(recon, target, reduction="none").sum((2, 3, 4)).mean()
                    kld = torch.max(kl_divergence(Normal(post_m, post_s), Normal(prior_m, prior_s)).sum(-1), free_nats).mean()
                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_loss = F.mse_loss(rew_pred, rew_seq[:, :T])
                    cont_logits = bottle(cont_model, h_seq, s_seq)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, (1.0 - done_seq[:, :T]).clamp(0, 1))

                    model_loss = rec_loss + args.kl_weight * kld + rew_loss + cont_loss

                    # ---- K-step InfoNCE on raw z ----
                    l_kstep = torch.tensor(0.0, device=device)
                    if use_kstep and total_steps >= args.kstep_min_steps:
                        z_seq = rssm_latent(h_seq, s_seq)
                        l_kstep, kstep_info = kstep_reachability_nce_loss(
                            z_seq, done_seq=done_seq[:, :T],
                            k_max=args.kstep_max_k, n_neg=args.kstep_negatives,
                            temperature=args.kstep_temperature, max_anchors=args.kstep_max_anchors)
                        ramp = min(1.0, (total_steps - args.kstep_min_steps) / 10_000.0)
                        model_loss = model_loss + args.kstep_weight * ramp * l_kstep
                        for k, v in kstep_info.items():
                            kstep_info_accum[k] = kstep_info_accum.get(k, 0.0) + v

                    model_opt.zero_grad(set_to_none=True)
                    model_loss.backward()
                    torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip)
                    model_opt.step()

                    sum_rec += float(rec_loss.item())
                    sum_kld += float(kld.item())
                    sum_rew += float(rew_loss.item())
                    sum_cont += float(cont_loss.item())
                    sum_model += float(model_loss.item())
                    sum_kstep += float(l_kstep.item())

                    # ============================================================
                    # GEO HEAD (separate optimizer, detached from WM)
                    # ============================================================
                    if use_geo_explore and total_steps >= args.geo_head_min_steps:
                        h_det = h_seq.detach()
                        s_det = s_seq.detach()
                        g_seq = geo_head(h_det, s_det)

                        l_geo_nce, geo_nce_info = geo_temporal_contrastive_loss(
                            g_seq, done_seq=done_seq[:, :T],
                            pos_k=args.geo_nce_pos_k, n_neg=args.kstep_negatives,
                            temperature=args.geo_nce_temperature, max_anchors=args.kstep_max_anchors)

                        g_flat = g_seq.reshape(-1, args.geo_dim)
                        g_sub = g_flat[torch.randperm(g_flat.size(0), device=device)[:512]] if g_flat.size(0) > 512 else g_flat
                        l_ac, ac_info = geo_anticollapse_loss(
                            g_sub, var_weight=args.geo_var_weight,
                            unif_weight=args.geo_unif_weight, cov_weight=args.geo_cov_weight)

                        geo_loss = args.geo_nce_weight * l_geo_nce + args.geo_anticollapse_weight * l_ac
                        geo_opt.zero_grad(set_to_none=True)
                        geo_loss.backward()
                        torch.nn.utils.clip_grad_norm_(geo_head.parameters(), args.grad_clip)
                        geo_opt.step()

                        sum_geo_nce += float(l_geo_nce.item())
                        sum_geo_ac += float(l_ac.item())
                        for k, v in geo_nce_info.items():
                            geo_info_accum[k] = geo_info_accum.get(k, 0.0) + v
                        for k, v in ac_info.items():
                            geo_info_accum[k] = geo_info_accum.get(k, 0.0) + v

                        with torch.no_grad():
                            geo_memory.add(g_flat.detach())

                    # ============================================================
                    # ACTOR-CRITIC with intrinsic reward
                    # ============================================================
                    B_seq, T_seq, Dh = h_seq.shape; Ds = s_seq.size(-1)
                    if args.imagination_starts and 0 < args.imagination_starts < T_seq:
                        K = args.imagination_starts
                        t_idx = torch.randint(0, T_seq, (B_seq, K), device=device)
                        h_start = h_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Dh)).reshape(-1, Dh).detach()
                        s_start = s_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Ds)).reshape(-1, Ds).detach()
                    else:
                        h_start = h_seq.reshape(-1, Dh).detach()
                        s_start = s_seq.reshape(-1, Ds).detach()

                    with no_param_grads(rssm), no_param_grads(reward_model), no_param_grads(cont_model):
                        h_im_list, s_im_list = [h_start], [s_start]
                        for _ in range(args.imagination_horizon):
                            a_im, _ = actor.get_action(h_im_list[-1], s_im_list[-1], deterministic=False)
                            h_next = rssm.deterministic_state_fwd(h_im_list[-1], s_im_list[-1], a_im)
                            s_next = rssm.state_prior(h_next, sample=True)
                            h_im_list.append(h_next)
                            s_im_list.append(s_next)
                        h_imag = torch.stack(h_im_list, dim=1)
                        s_imag = torch.stack(s_im_list, dim=1)
                        rewards_imag = bottle(reward_model, h_imag[:, 1:], s_imag[:, 1:])
                        cont_logits_imag = bottle(cont_model, h_imag[:, 1:], s_imag[:, 1:])
                        pcont_imag = torch.sigmoid(cont_logits_imag).clamp(0.0, 1.0)
                        discounts_imag = effective_gamma * pcont_imag

                    # ---- Intrinsic reward ----
                    intrinsic_reward = torch.zeros_like(rewards_imag)
                    intrinsic_active = (
                        use_geo_explore and geo_head is not None and geo_memory is not None
                        and geo_memory.size >= 64 and total_steps >= args.geo_intrinsic_min_steps
                    )
                    if intrinsic_active:
                        with torch.no_grad(), no_param_grads(geo_head):
                            intrinsic_reward = compute_geo_intrinsic_reward(
                                geo_head, h_imag, s_imag, geo_memory,
                                knn_k=args.geo_intrinsic_knn_k,
                                frontier_weight=args.geo_frontier_weight,
                                distortion_weight=args.geo_distortion_weight)
                        ramp = min(1.0, (total_steps - args.geo_intrinsic_min_steps) / max(args.geo_intrinsic_warmup, 1))
                        intrinsic_reward = ramp * args.geo_intrinsic_weight * intrinsic_reward

                    rewards_total = rewards_imag + intrinsic_reward
                    sum_imag_r += float(rewards_imag.mean().item())
                    sum_intrinsic_r += float(intrinsic_reward.mean().item())

                    with torch.no_grad():
                        values_tgt = bottle(value_model, h_imag, s_imag)
                        lambda_ret = compute_lambda_returns(rewards_total.detach(), values_tgt, discounts_imag.detach(), lambda_=args.lambda_)
                        w_val = compute_discount_weights(discounts_imag.detach())

                    values_pred = bottle(value_model, h_imag.detach(), s_imag.detach())
                    value_loss = ((values_pred[:, :-1] - lambda_ret) ** 2 * w_val).mean()
                    value_opt.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip)
                    value_opt.step()
                    sum_value += float(value_loss.item())

                    mean_a, std_a = actor.forward(h_imag[:, :-1].detach(), s_imag[:, :-1].detach())
                    noise_a = torch.randn_like(mean_a)
                    raw_a = mean_a + std_a * noise_a
                    entropy = (Normal(mean_a, std_a).entropy() + torch.log(1 - torch.tanh(raw_a).pow(2) + 1e-6)).sum(-1).mean()

                    with no_param_grads(value_model):
                        values_for_actor = bottle(value_model, h_imag, s_imag)
                    w_actor = compute_discount_weights(discounts_imag.detach())
                    lambda_actor = compute_lambda_returns(rewards_total, values_for_actor, discounts_imag, lambda_=args.lambda_)
                    actor_loss = -(w_actor.detach() * lambda_actor).mean() - args.actor_entropy_scale * entropy
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    actor_opt.step()
                    sum_actor += float(actor_loss.item())

                # ---- Logging ----
                n_ts = float(args.train_steps)
                writer.add_scalar("loss/reconstruction", sum_rec / n_ts, total_steps)
                writer.add_scalar("loss/kl", sum_kld / n_ts, total_steps)
                writer.add_scalar("loss/reward_pred", sum_rew / n_ts, total_steps)
                writer.add_scalar("loss/continue", sum_cont / n_ts, total_steps)
                writer.add_scalar("loss/model_total", sum_model / n_ts, total_steps)
                writer.add_scalar("loss/actor", sum_actor / n_ts, total_steps)
                writer.add_scalar("loss/value", sum_value / n_ts, total_steps)
                writer.add_scalar("imag/reward_mean", sum_imag_r / n_ts, total_steps)
                writer.add_scalar("imag/intrinsic_reward_mean", sum_intrinsic_r / n_ts, total_steps)
                writer.add_scalar("train/exploration_noise", expl_amount, total_steps)
                if use_kstep:
                    writer.add_scalar("loss/kstep_total", sum_kstep / n_ts, total_steps)
                    for k, v in kstep_info_accum.items():
                        writer.add_scalar(f"loss/{k}", v / n_ts, total_steps)
                if use_geo_explore:
                    writer.add_scalar("loss/geo_nce", sum_geo_nce / n_ts, total_steps)
                    writer.add_scalar("loss/geo_anticollapse", sum_geo_ac / n_ts, total_steps)
                    writer.add_scalar("geo/intrinsic_active", 1.0 if intrinsic_active else 0.0, total_steps)
                    if geo_memory is not None:
                        writer.add_scalar("geo/memory_size", float(geo_memory.size), total_steps)
                    for k, v in geo_info_accum.items():
                        writer.add_scalar(f"geo/{k}", v / n_ts, total_steps)

        if args.expl_decay > 0:
            expl_amount = max(args.expl_min, expl_amount - args.expl_decay)

        if ep_success:
            cumulative_successes += 1
            if first_goal_step is None:
                first_goal_step = total_steps
                writer.add_scalar("eval/first_goal_env_step", float(first_goal_step), 0)

        writer.add_scalar("train/episode_return", ep_ret, episode)
        writer.add_scalar("episode/return_env_step", ep_ret, total_steps)
        writer.add_scalar("train/episode_success", 1.0 if ep_success else 0.0, episode)
        writer.add_scalar("train/success_rate", float(cumulative_successes) / float(episode + 1), episode)
        writer.add_scalar("eval/unique_cells_episode", float(len(ep_cells)), episode)
        if len(ep_pos_traj) >= 2:
            writer.add_scalar("eval/bridge_crossings_episode",
                              float(_bridge_crossing_count(geodesic, np.stack(ep_pos_traj))), episode)

        print(f"  Ep {episode+1}/{args.max_episodes}  ret={ep_ret:.2f}  steps={ep_steps}  "
              f"cells={len(ep_cells)}  total={total_steps}  success={ep_success}")

        # ---- Diagnostics ----
        if args.diag_interval > 0 and (episode + 1) % args.diag_interval == 0:
            if replay.size > 256:
                diag_batch = replay.sample_sequences(min(64, args.batch_size), 1)
                diag_obs = torch.tensor(diag_batch.obs[:, 0], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                preprocess_img(diag_obs, depth=args.bit_depth)
                with torch.no_grad():
                    diag_e = encoder(diag_obs)
                    diag_h, diag_s = rssm.get_init_state(diag_e)
                    diag_z = rssm_latent(diag_h, diag_s)
                diags = compute_latent_diagnostics(diag_z, prefix="latent_hs")
                for k, v in diags.items():
                    writer.add_scalar(k, v, total_steps)
                print(f"    [diag h+s] eff_rank={diags['latent_hs/effective_rank']:.1f}  "
                      f"dead={diags['latent_hs/num_dead_dims']:.0f}")

                if use_geo_explore and geo_head is not None and total_steps >= args.geo_head_min_steps:
                    with torch.no_grad():
                        diag_g = geo_head(diag_h, diag_s)
                    gd = compute_latent_diagnostics(diag_g, prefix="geo_head")
                    for k, v in gd.items():
                        writer.add_scalar(k, v, total_steps)
                    print(f"    [diag geo] eff_rank={gd['geo_head/effective_rank']:.1f}  "
                          f"min_std={gd['geo_head/min_per_dim_std']:.4f}  dead={gd['geo_head/num_dead_dims']:.0f}")

        # ---- Geodesic correlation ----
        if args.geo_corr_interval > 0 and (episode + 1) % args.geo_corr_interval == 0:
            if len(obs_geo_buffer) >= 100:
                obs_arr = np.stack(obs_geo_buffer, axis=0)
                pos_arr = np.stack(pos_geo_buffer, axis=0)
                gc = compute_hs_geodesic_correlation(encoder, rssm, obs_arr, pos_arr, geodesic, device, args.bit_depth)
                for k, v in gc.items():
                    writer.add_scalar(k, v, total_steps)
                print(f"    [geo h+s] ρ={gc['geo_hs/spearman']:.4f}")

                if use_geo_explore and geo_head is not None and total_steps >= args.geo_head_min_steps:
                    ghc = compute_geohead_geodesic_correlation(
                        geo_head, encoder, rssm, obs_arr, pos_arr, geodesic, device, args.bit_depth)
                    for k, v in ghc.items():
                        writer.add_scalar(k, v, total_steps)
                    print(f"    [geo head] ρ={ghc['geo_head/spearman']:.4f}  *** KEY METRIC ***")

        # ---- Eval ----
        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            ev = run_periodic_eval(env, encoder, rssm, actor, device, args.bit_depth,
                                   args.eval_episodes, geodesic, action_repeat)
            for k, v in ev.items():
                writer.add_scalar(f"eval/{k}", v, total_steps)
            print(f"    [eval] ret={ev['return_mean']:.2f}  cells={ev['unique_cells_mean']:.1f}  "
                  f"bridges={ev['bridge_crossings_mean']:.1f}  success={ev['success_rate']:.2f}")

        # ---- Checkpoint ----
        if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
            ckpt_data = {
                "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
                "rssm": rssm.state_dict(), "reward_model": reward_model.state_dict(),
                "cont_model": cont_model.state_dict(), "actor": actor.state_dict(),
                "value_model": value_model.state_dict(),
                "episode": episode + 1, "total_steps": total_steps, "geo_mode": args.geo_mode,
            }
            if geo_head is not None:
                ckpt_data["geo_head"] = geo_head.state_dict()
            torch.save(ckpt_data, os.path.join(out_dir, f"checkpoint_ep{episode+1}.pt"))

    # Final save
    ckpt_data = {
        "encoder": encoder.state_dict(), "decoder": decoder.state_dict(),
        "rssm": rssm.state_dict(), "reward_model": reward_model.state_dict(),
        "cont_model": cont_model.state_dict(), "actor": actor.state_dict(),
        "value_model": value_model.state_dict(),
        "episode": args.max_episodes, "total_steps": total_steps, "geo_mode": args.geo_mode,
    }
    if geo_head is not None:
        ckpt_data["geo_head"] = geo_head.state_dict()
    torch.save(ckpt_data, os.path.join(out_dir, "world_model_final.pt"))

    env.close(); writer.close()
    print(f"\nDone. Mode={args.geo_mode}  total_steps={total_steps}")


if __name__ == "__main__":
    main(build_parser().parse_args())