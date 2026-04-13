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
# Env: Dreamer-style reset (fixed / subset spawn), not uniform random start
# =====================================================================


class PointMazeLargeDreamerWrapper(PointMazeLargeDiverseGRWrapper):
    """Large maze with controllable spawn. Default: fixed start cell, random goal elsewhere.

    The base ``PointMazeLargeDiverseGRWrapper`` randomizes both start and goal every episode
    for coverage; that is closer to an exploration analysis setup than a typical Dreamer run.
    """

    def __init__(
        self,
        img_size: int = 64,
        *,
        reset_mode: str = "fixed_start",
        fixed_start_cell: tuple[int, int] | None = None,
        start_cells: list[tuple[int, int]] | None = None,
    ):
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

    def _sample_goal_cell(self, sr: int, sc: int) -> np.ndarray:
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
            r, c = self.fixed_start_cell  # type: ignore[misc]
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
# Geometry head (decoupled from world model for k-step)
# =====================================================================


class GeoHead(nn.Module):
    """Projects Dreamer latent (h, s) -> normalized geometric embedding.

    Trained with a separate optimizer; gradients do not flow into RSSM/encoder,
    so k-step contrastive loss does not corrupt dynamics.
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

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, s], dim=-1)
        g = self.net(x)
        return F.normalize(g, dim=-1)


# =====================================================================
# Geometric losses
# =====================================================================

def kstep_nce_with_bank(
    g_seq: torch.Tensor,
    bank: torch.Tensor,
    done_seq: torch.Tensor | None = None,
    *,
    k_max: int = 5,
    n_neg: int = 256,
    temperature: float = 0.07,
    max_anchors: int = 256,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InfoNCE with memory-bank negatives (reduces false negatives vs batch sampling).

    ``g_seq`` should already be L2-normalized per vector (e.g. GeoHead output).
    """
    B, T, D = g_seq.shape
    if T < 2:
        z = g_seq.new_zeros(())
        return z, {
            "kstep/loss": 0.0,
            "kstep/n_pairs": 0.0,
            "kstep/pos_sim": 0.0,
            "kstep/neg_sim": 0.0,
            "kstep/gap": 0.0,
        }

    flat = g_seq.reshape(B * T, D)
    pairs: list[tuple[int, int]] = []
    max_anchors = max(1, int(max_anchors))
    for b in range(B):
        for t in range(T - 1):
            max_delta = min(int(k_max), T - 1 - t)
            if max_delta < 1:
                continue
            valid: list[int] = []
            for d in range(1, max_delta + 1):
                if done_seq is not None:
                    seg = done_seq[b, t : t + d]
                    if torch.any(seg > 0.5):
                        break
                valid.append(d)
            if not valid:
                continue
            d = valid[int(torch.randint(0, len(valid), (1,), device=g_seq.device).item())]
            pairs.append((b * T + t, b * T + t + d))

    if not pairs:
        z = g_seq.new_zeros(())
        return z, {
            "kstep/loss": 0.0,
            "kstep/n_pairs": 0.0,
            "kstep/pos_sim": 0.0,
            "kstep/neg_sim": 0.0,
            "kstep/gap": 0.0,
        }

    if len(pairs) > max_anchors:
        perm = torch.randperm(len(pairs), device=g_seq.device)[:max_anchors]
        pairs = [pairs[i] for i in perm.tolist()]

    anc_idx = torch.tensor([p[0] for p in pairs], dtype=torch.long, device=g_seq.device)
    pos_idx = torch.tensor([p[1] for p in pairs], dtype=torch.long, device=g_seq.device)
    anchors = flat[anc_idx]
    positives = flat[pos_idx]

    n_neg = max(1, int(n_neg))
    neg_idx = torch.randint(0, bank.size(0), (len(pairs), n_neg), device=g_seq.device)
    negatives = bank[neg_idx]

    temp = max(float(temperature), 1e-6)
    pos_logits = (anchors * positives).sum(-1, keepdim=True) / temp
    neg_logits = torch.einsum("bd,bnd->bn", anchors, negatives) / temp
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    loss = F.cross_entropy(logits, torch.zeros(len(pairs), dtype=torch.long, device=g_seq.device))

    pos_sim = (anchors * positives).sum(-1)
    neg_mean = neg_logits.mean()
    return loss, {
        "kstep/loss": float(loss.item()),
        "kstep/n_pairs": float(len(pairs)),
        "kstep/pos_sim": float(pos_sim.mean().item()),
        "kstep/neg_sim": float(neg_mean.item()),
        "kstep/gap": float((pos_sim.mean() - neg_mean).item()),
    }


def uniformity_loss(g: torch.Tensor) -> torch.Tensor:
    """Encourages spread on the hypersphere (Wang & Isola, 2020)."""
    if g.size(0) < 2:
        return g.new_zeros(())
    d = torch.pdist(g, p=2).pow(2)
    return torch.log(torch.exp(-2.0 * d).mean() + 1e-6)


def rssm_latent(h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Concatenate deterministic and stochastic Dreamer state into one latent."""
    return torch.cat([h, s], dim=-1)


# =====================================================================
# Geometric diagnostics (logged to TensorBoard)
# =====================================================================


@torch.no_grad()
def compute_latent_diagnostics(
    z: torch.Tensor,
    prefix: str = "latent",
    max_samples: int = 2048,
) -> dict[str, float]:
    """Compute latent space health metrics from precomputed embeddings."""
    N = z.shape[0]
    idx = np.random.choice(N, size=min(max_samples, N), replace=False)
    z = z[idx]

    # Pairwise L2 distances
    dm = torch.cdist(z, z)
    triu_mask = torch.triu(torch.ones_like(dm, dtype=torch.bool), diagonal=1)
    dists = dm[triu_mask]

    # Per-dimension statistics
    per_dim_std = z.std(dim=0)

    # Covariance off-diagonal magnitude
    z_c = z - z.mean(dim=0)
    cov = (z_c.T @ z_c) / max(z.shape[0] - 1, 1)
    diag = cov.diagonal()
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    off_diag_norm = (off_diag_sq / max(cov.shape[0] ** 2 - cov.shape[0], 1)).sqrt()

    # Effective rank approximation via singular values
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
def compute_hs_geodesic_correlation(
    encoder: ConvEncoder,
    rssm: RSSM,
    obs_buffer,
    pos_buffer: np.ndarray,
    geodesic,
    device: torch.device,
    bit_depth: int,
    n_pairs: int = 2000,
) -> dict[str, float]:
    """Spearman correlation: Dreamer h+s latent distance vs geodesic distance."""
    from scipy.stats import spearmanr

    encoder.eval(); rssm.eval()
    N = len(pos_buffer)
    if N < 20:
        return {"geo_hs/spearman": 0.0, "geo_hs/n_pairs": 0}

    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    ii = rng.integers(0, N, size=n_pairs)
    jj = rng.integers(0, N, size=n_pairs)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]

    # Encode observations into Dreamer state latents z=[h,s]
    def encode_batch(indices):
        latents = []
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            batch_idx = indices[start:end]
            obs_batch = torch.tensor(
                obs_buffer[batch_idx], dtype=torch.float32, device=device
            ).permute(0, 3, 1, 2)
            preprocess_img(obs_batch, depth=bit_depth)
            e_batch = encoder(obs_batch)
            h_batch, s_batch = rssm.get_init_state(e_batch)
            z_batch = rssm_latent(h_batch, s_batch)
            latents.append(z_batch.cpu().numpy())
        return np.concatenate(latents, axis=0)

    z_i = encode_batch(ii)
    z_j = encode_batch(jj)
    d_hs = np.linalg.norm(z_i - z_j, axis=-1)

    cell_i = _positions_to_cell_indices(geodesic, pos_buffer[ii])
    cell_j = _positions_to_cell_indices(geodesic, pos_buffer[jj])
    d_geo = np.array([
        float(geodesic.dist_matrix[int(ci), int(cj)])
        for ci, cj in zip(cell_i, cell_j)
    ], dtype=np.float32)

    finite = np.isfinite(d_geo) & (d_geo > 0) & np.isfinite(d_hs) & (d_hs > 0)
    if finite.sum() < 10:
        return {"geo_hs/spearman": 0.0, "geo_hs/n_pairs": 0}

    rho = spearmanr(d_hs[finite], d_geo[finite]).correlation
    if rho is None or not np.isfinite(rho):
        rho = 0.0

    return {
        "geo_hs/spearman": float(rho),
        "geo_hs/n_pairs": int(finite.sum()),
        "geo_hs/mean_latent_dist": float(np.mean(d_hs[finite])),
        "geo_hs/mean_geo_dist": float(np.mean(d_geo[finite])),
    }


@torch.no_grad()
def compute_geohead_geodesic_correlation(
    geo_head: GeoHead,
    encoder: ConvEncoder,
    rssm: RSSM,
    obs_buffer,
    pos_buffer: np.ndarray,
    geodesic,
    device: torch.device,
    bit_depth: int,
    n_pairs: int = 2000,
) -> dict[str, float]:
    """Spearman correlation: GeoHead embedding distance vs geodesic distance."""
    from scipy.stats import spearmanr

    geo_head.eval()
    encoder.eval()
    rssm.eval()
    N = len(pos_buffer)
    if N < 20:
        return {"geo_head/spearman": 0.0, "geo_head/n_pairs": 0}

    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    ii = rng.integers(0, N, size=n_pairs)
    jj = rng.integers(0, N, size=n_pairs)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]

    def encode_batch(indices):
        outs = []
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            batch_idx = indices[start:end]
            obs_batch = torch.tensor(
                obs_buffer[batch_idx], dtype=torch.float32, device=device
            ).permute(0, 3, 1, 2)
            preprocess_img(obs_batch, depth=bit_depth)
            e_batch = encoder(obs_batch)
            h_batch, s_batch = rssm.get_init_state(e_batch)
            g_batch = geo_head(h_batch, s_batch)
            outs.append(g_batch.cpu().numpy())
        return np.concatenate(outs, axis=0)

    g_i = encode_batch(ii)
    g_j = encode_batch(jj)
    d_g = np.linalg.norm(g_i - g_j, axis=-1)

    cell_i = _positions_to_cell_indices(geodesic, pos_buffer[ii])
    cell_j = _positions_to_cell_indices(geodesic, pos_buffer[jj])
    d_geo = np.array([
        float(geodesic.dist_matrix[int(ci), int(cj)])
        for ci, cj in zip(cell_i, cell_j)
    ], dtype=np.float32)

    finite = np.isfinite(d_geo) & (d_geo > 0) & np.isfinite(d_g) & (d_g > 0)
    if finite.sum() < 10:
        return {"geo_head/spearman": 0.0, "geo_head/n_pairs": 0}

    rho = spearmanr(d_g[finite], d_geo[finite]).correlation
    if rho is None or not np.isfinite(rho):
        rho = 0.0

    return {
        "geo_head/spearman": float(rho),
        "geo_head/n_pairs": int(finite.sum()),
    }


# =====================================================================
# Training helpers
# =====================================================================


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
    w = torch.cumprod(torch.cat([ones, discounts], dim=1), dim=1)[:, :-1]
    return w


def _bridge_crossing_count(geodesic, pos_seq):
    from maze_geometry_test import _adj_from_distmat, _find_bridges
    adj = _adj_from_distmat(geodesic.dist_matrix)
    bridges = set(_find_bridges(adj))
    if len(pos_seq) < 2:
        return 0
    cells = _positions_to_cell_indices(geodesic, pos_seq)
    n_cross = 0
    for a, b in zip(cells[:-1], cells[1:]):
        u, v = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        if (u, v) in bridges:
            n_cross += 1
    return n_cross


# =====================================================================
# Intrinsic reward helpers (mirrored from evaluator)
# =====================================================================


INTRINSIC_PRESETS: dict[str, tuple[float, float, float]] = {
    "baseline": (0.0, 0.0, 0.0),
    "f_only": (1.0, 0.0, 0.0),
    "f_d": (1.0, 0.15, 0.0),
    "f_d_b": (1.0, 0.15, 0.05),
}


def _online_disagreement_step(
    ep_z: list[np.ndarray],
    episodic_g: list[np.ndarray],
    z_np: np.ndarray,
    g_t: np.ndarray,
    ema_scale: float,
    max_landmarks: int = 4,
) -> tuple[float, float]:
    """Mean |d_geo - s * d_raw| to a few prior landmarks; EMA scale s matches raw vs geo distances."""
    if len(ep_z) < 1:
        return 0.0, ema_scale
    n_prev = len(ep_z)
    n_lm = max(1, min(int(max_landmarks), n_prev))
    idx = np.unique(np.linspace(0, n_prev - 1, num=n_lm, dtype=int))
    drs: list[float] = []
    dgs: list[float] = []
    for j in idx.tolist():
        dr = float(np.linalg.norm(z_np - ep_z[int(j)]))
        dg = float(np.linalg.norm(g_t - episodic_g[int(j)]))
        drs.append(dr)
        dgs.append(dg)
    ratios = [dgs[i] / (drs[i] + 1e-6) for i in range(len(drs)) if drs[i] > 1e-6]
    if ratios:
        ema_scale = 0.92 * ema_scale + 0.08 * float(np.mean(ratios))
    Dv = float(np.mean([abs(dgs[i] - ema_scale * drs[i]) for i in range(len(drs))]))
    return Dv, ema_scale


class _EMANormalizer:
    """Running EMA mean/std normalizer for a scalar signal."""

    def __init__(self, alpha: float = 0.01):
        self._alpha = alpha
        self._mean = 0.0
        self._var = 1.0
        self._count = 0

    def normalize(self, x: float) -> float:
        self._count += 1
        if self._count == 1:
            self._mean = x
            self._var = 1.0
            return 0.0
        self._mean = (1 - self._alpha) * self._mean + self._alpha * x
        self._var = (1 - self._alpha) * self._var + self._alpha * (x - self._mean) ** 2
        std = max(math.sqrt(self._var), 1e-6)
        return (x - self._mean) / std


def _oracle_bridge_edges(geodesic) -> set[tuple[int, int]]:
    """Precompute bridge edges for the maze graph (oracle topology)."""
    from maze_geometry_test import _adj_from_distmat, _find_bridges
    adj = _adj_from_distmat(geodesic.dist_matrix)
    bridges = _find_bridges(adj)
    edge_set: set[tuple[int, int]] = set()
    for u, v in bridges:
        a, b = (int(u), int(v)) if int(u) < int(v) else (int(v), int(u))
        edge_set.add((a, b))
    return edge_set


# =====================================================================
# Periodic evaluation
# =====================================================================


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
        cells.add(c0)
        pos_traj.append(env.agent_pos.copy())

        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=bit_depth)
        e0 = encoder(obs_t)
        h_state, s_state = rssm.get_init_state(e0)

        while not done:
            action_t, _ = actor.get_action(h_state, s_state, deterministic=True)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, r, term, trunc, info = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            ep_ret += float(r)
            ep_steps += 1
            if info.get("success", False) or info.get("is_success", False):
                ep_success = True
            c = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
            cells.add(c)
            pos_traj.append(env.agent_pos.copy())
            obs_t = torch.tensor(np.ascontiguousarray(next_obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=bit_depth)
            e = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(e, act_t, h_state, s_state, sample=False)

        rets.append(ep_ret)
        lens.append(ep_steps)
        uniques.append(float(len(cells)))
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
    p = argparse.ArgumentParser(description="Dreamer + geometric regularization ablations on PointMaze_Large")
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
    p.add_argument(
        "--reset_mode",
        type=str,
        default="fixed_start",
        choices=["fixed_start", "start_subset", "random"],
        help="fixed_start / start_subset: fixed spawn + random goal (typical Dreamer); "
        "random: uniform start+goal on all free cells (old behavior).",
    )
    p.add_argument(
        "--fixed_start_row",
        type=int,
        default=-1,
        help="With fixed_start: spawn row; -1 uses first free cell from geodesic.",
    )
    p.add_argument(
        "--fixed_start_col",
        type=int,
        default=-1,
        help="With fixed_start: spawn col; -1 uses first free cell from geodesic.",
    )
    p.add_argument(
        "--start_subset",
        type=str,
        default="",
        help="For start_subset: semicolon-separated row,col pairs, e.g. '1,1;5,3'. Empty = first 3 free cells.",
    )
    p.add_argument("--eval_interval", type=int, default=20)
    p.add_argument("--eval_episodes", type=int, default=5)

    # Geometric regularization
    p.add_argument("--geo_mode", type=str, default="baseline",
                   choices=["baseline", "kstep"],
                   help="baseline | kstep (InfoNCE reachability)")
    p.add_argument("--kstep_weight", type=float, default=0.1,
                   help="Weight of k-step InfoNCE on GeoHead (separate optimizer)")
    p.add_argument("--kstep_max_k", type=int, default=5,
                   help="Positive window size for k-step reachability")
    p.add_argument("--kstep_negatives", type=int, default=256,
                   help="Negatives per anchor (sampled from geo memory bank)")
    p.add_argument("--kstep_temperature", type=float, default=0.07,
                   help="InfoNCE temperature for k-step reachability")
    p.add_argument("--kstep_max_anchors", type=int, default=256,
                   help="Max anchor-positive pairs per batch for k-step reachability")
    p.add_argument("--kstep_min_steps", type=int, default=50_000,
                   help="Env steps before applying k-step GeoHead loss (WM trains without it until then)")
    p.add_argument("--kstep_unif_weight", type=float, default=0.5,
                   help="Weight of uniformity loss on GeoHead embeddings")
    p.add_argument("--geo_dim", type=int, default=32, help="GeoHead output dimension")
    p.add_argument("--geo_bank_size", type=int, default=16384,
                   help="Memory bank size for k-step negatives")
    p.add_argument("--geo_lr", type=float, default=3e-4, help="GeoHead AdamW learning rate")

    # Intrinsic reward during training
    p.add_argument("--intrinsic_ablation", type=str, default="baseline",
                   choices=["baseline", "f_only", "f_d", "f_d_b"],
                   help="Intrinsic reward preset: baseline (none), "
                   "f_only (frontier only), f_d (frontier+disagreement), "
                   "f_d_b (frontier+disagreement+bridge)")
    p.add_argument("--int_lambda_f", type=float, default=1.0,
                   help="Weight on frontier novelty F (overridden by preset)")
    p.add_argument("--int_lambda_d", type=float, default=0.15,
                   help="Weight on raw-vs-geo disagreement D (overridden by preset)")
    p.add_argument("--int_lambda_b", type=float, default=0.05,
                   help="Weight on bridge crossing B (overridden by preset)")
    p.add_argument("--intrinsic_scale", type=float, default=0.05,
                   help="Global scale beta: r_store = r_ext + beta * r_int")
    p.add_argument("--intrinsic_decay_on_success", type=float, default=0.25,
                   help="Multiply intrinsic_scale by this factor after first goal success "
                   "(0 = kill intrinsic entirely; 1 = no decay)")
    p.add_argument("--intrinsic_normalize", action="store_true",
                   help="EMA-normalize F and D before mixing (makes lambda ratios "
                   "reflect effective weight regardless of raw magnitude)")

    # Diagnostics
    p.add_argument("--diag_interval", type=int, default=40,
                   help="Run h+s latent diagnostics every N training episodes")
    p.add_argument("--geo_corr_interval", type=int, default=80,
                   help="Run geodesic correlation eval every N episodes (expensive)")

    p.add_argument("--wm_path", type=str, default="")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="georeg")
    p.add_argument("--save_interval", type=int, default=100,
                   help="Save checkpoint every N episodes")
    return p


# =====================================================================
# Main training loop
# =====================================================================


def main(args):
    set_seed(args.seed)
    device = get_device()

    use_kstep = args.geo_mode == "kstep"

    use_intrinsic = args.intrinsic_ablation != "baseline"
    int_lf, int_ld, int_lb = INTRINSIC_PRESETS[args.intrinsic_ablation]
    intrinsic_beta = float(args.intrinsic_scale) if use_intrinsic else 0.0

    print(f"Device: {device} Geo mode: {args.geo_mode}")
    print(f"  K-step InfoNCE: {use_kstep} (weight={args.kstep_weight})")
    if use_intrinsic:
        print(f"  Intrinsic: {args.intrinsic_ablation}  λ_f={int_lf} λ_d={int_ld} λ_b={int_lb}  β={intrinsic_beta}")
        print(f"  Intrinsic decay_on_success={args.intrinsic_decay_on_success}  normalize={args.intrinsic_normalize}")
        if int_ld > 0 and not use_kstep:
            print("  WARNING: intrinsic D requires geo_mode=kstep (geo_head). D will be 0.")
    if use_kstep:
        print(
            f"  K-step: GeoHead (dim={args.geo_dim}) + bank={args.geo_bank_size}, "
            f"temp={args.kstep_temperature}, separate AdamW; starts after {args.kstep_min_steps} env steps"
        )

    start_cells: list[tuple[int, int]] | None = None
    if args.reset_mode == "start_subset" and args.start_subset.strip():
        start_cells = []
        for part in args.start_subset.split(";"):
            part = part.strip()
            if not part:
                continue
            r, c = part.split(",")
            start_cells.append((int(r.strip()), int(c.strip())))

    fixed_cell: tuple[int, int] | None = None
    if args.reset_mode == "fixed_start":
        if args.fixed_start_row >= 0 and args.fixed_start_col >= 0:
            fixed_cell = (args.fixed_start_row, args.fixed_start_col)

    env = PointMazeLargeDreamerWrapper(
        img_size=args.img_size,
        reset_mode=args.reset_mode,
        fixed_start_cell=fixed_cell,
        start_cells=start_cells,
    )
    geodesic = env.geodesic
    print(f"  Maze: PointMaze_Large  grid={env.grid_h}x{env.grid_w}  free_cells={geodesic.n_free}")
    print(f"  Reset: {args.reset_mode}", end="")
    if args.reset_mode == "fixed_start":
        print(f"  spawn={env.fixed_start_cell}  (random goal on other free cells)")
    elif args.reset_mode == "start_subset":
        print(f"  |start_subset|={len(env.start_cells)}")
    else:
        print("  (uniform random start+goal)")

    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    action_repeat = 1
    effective_gamma = args.gamma ** action_repeat

    encoder = ConvEncoder(embedding_size=args.embed_dim, in_channels=C).to(device)
    decoder = ConvDecoder(args.deter_dim, args.stoch_dim, embedding_size=args.embed_dim, out_channels=C).to(device)
    rssm = RSSM(args.stoch_dim, args.deter_dim, act_dim, args.embed_dim, args.hidden_dim).to(device)
    reward_model = RewardModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(device)
    cont_model = ContinueModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(device)

    ckpt = None
    if args.wm_path:
        ckpt = torch.load(args.wm_path, map_location=device)
        for k in ("encoder", "decoder", "rssm", "reward_model", "cont_model"):
            if k in ckpt:
                locals()[k].load_state_dict(ckpt[k])
        print(f"  Loaded world model from {args.wm_path}")

    actor = Actor(args.deter_dim, args.stoch_dim, act_dim, args.actor_hidden_dim).to(device)
    value_model = ValueModel(args.deter_dim, args.stoch_dim, args.value_hidden_dim).to(device)

    geo_head: GeoHead | None = None
    geo_opt = None
    bank: torch.Tensor | None = None
    bank_ptr = 0
    if use_kstep:
        geo_head = GeoHead(args.deter_dim, args.stoch_dim, geo_dim=args.geo_dim).to(device)
        geo_opt = torch.optim.AdamW(geo_head.parameters(), lr=args.geo_lr, weight_decay=1e-5)
        bank = F.normalize(
            torch.randn(args.geo_bank_size, args.geo_dim, device=device), dim=-1
        )
        if ckpt is not None:
            if "geo_head" in ckpt:
                geo_head.load_state_dict(ckpt["geo_head"])
            if "bank" in ckpt:
                bank = ckpt["bank"].to(device)
            if "bank_ptr" in ckpt:
                bank_ptr = int(ckpt["bank_ptr"])

    world_params = (
        list(encoder.parameters()) + list(decoder.parameters())
        + list(rssm.parameters()) + list(reward_model.parameters())
        + list(cont_model.parameters())
    )
    model_opt = torch.optim.Adam(world_params, lr=args.model_lr, eps=args.adam_eps)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=args.adam_eps)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=args.value_lr, eps=args.adam_eps)

    replay = ReplayBuffer(args.replay_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    tag = f"{args.geo_mode}_seed{args.seed}"
    if args.run_name:
        tag = f"{args.run_name}_seed{args.seed}"
    writer = SummaryWriter(f"{args.log_dir}/{tag}")
    writer.add_text("hyperparameters", str(vars(args)), 0)

    out_dir = os.path.join(args.log_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # Observation + position buffers for geodesic correlation eval
    obs_geo_buffer: list[np.ndarray] = []
    pos_geo_buffer: list[np.ndarray] = []
    GEO_BUFFER_MAX = 8000

    int_visit_count: dict[int, int] = defaultdict(int)
    int_bridge_edges: set[tuple[int, int]] = set()
    if use_intrinsic and int_lb > 0:
        int_bridge_edges = _oracle_bridge_edges(geodesic)
        print(f"  Bridge topology: {len(int_bridge_edges)} bridge edges (oracle, edge-crossing only)")
    int_norm_F: _EMANormalizer | None = None
    int_norm_D: _EMANormalizer | None = None
    if use_intrinsic and args.intrinsic_normalize:
        int_norm_F = _EMANormalizer(alpha=0.01)
        int_norm_D = _EMANormalizer(alpha=0.01)

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
    print(f"\n  Training {args.max_episodes} episodes (geo_mode={args.geo_mode})")

    for episode in range(args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret, ep_steps = 0.0, 0
        ep_success = False
        ep_cells: set[int] = set()
        ep_pos_traj: list[np.ndarray] = []
        c0 = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
        ep_cells.add(c0)
        ep_pos_traj.append(env.agent_pos.copy())

        with torch.no_grad():
            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=args.bit_depth)
            e0 = encoder(obs_t)
            h_state, s_state = rssm.get_init_state(e0)

        ep_z_int: list[np.ndarray] = []
        ep_g_int: list[np.ndarray] = []
        ep_used_bridge_edges: set[tuple[int, int]] = set()
        ep_ema_scale = 1.0
        ep_sum_int = ep_sum_F = ep_sum_D = ep_sum_B = 0.0

        while not done:
            encoder.eval(); rssm.eval(); actor.eval()
            with torch.no_grad():
                action_t, _ = actor.get_action(h_state, s_state, deterministic=False)
                if expl_amount > 0:
                    action_t = action_t + expl_amount * torch.randn_like(action_t)
                    action_t = torch.clamp(action_t, -1.0, 1.0)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            # --- Intrinsic: F (frontier) and D (disagreement) before env step ---
            Fv = Dv = 0.0
            if use_intrinsic and total_steps >= args.kstep_min_steps:
                cell_pre = int(_positions_to_cell_indices(
                    geodesic, env.agent_pos.reshape(1, -1))[0])
                vc = int_visit_count[cell_pre]
                Fv = 1.0 / math.sqrt(vc + 1.0)
                int_visit_count[cell_pre] = vc + 1

                # D is gated: only active after geo head has trained past kstep_min_steps,
                # then ramps in over 10k steps to avoid noisy random-head disagreement.
                geo_ready = (
                    int_ld > 0
                    and geo_head is not None
                    #and total_steps >= args.kstep_min_steps
                )
                if geo_ready:
                    with torch.no_grad():
                        h_np = h_state.squeeze(0).cpu().numpy().astype(np.float32)
                        s_np = s_state.squeeze(0).cpu().numpy().astype(np.float32)
                        z_np = np.concatenate([h_np, s_np], axis=-1)
                        g_t = geo_head(h_state, s_state).squeeze(0).cpu().numpy().astype(np.float32)
                        if len(ep_z_int) >= 1:
                            Dv, ep_ema_scale = _online_disagreement_step(
                                ep_z_int, ep_g_int, z_np, g_t, ep_ema_scale)
                        ep_z_int.append(z_np.copy())
                        ep_g_int.append(g_t.copy())
                    ramp_denom = 10_000.0
                    d_ramp = min(1.0, (total_steps - args.kstep_min_steps) / ramp_denom)
                    Dv *= d_ramp

            next_obs, r, term, trunc, step_info = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)

            # --- Intrinsic: B (bridge edge crossing) after env step ---
            Bv = 0.0
            if use_intrinsic and int_lb > 0:
                cell_post = int(_positions_to_cell_indices(
                    geodesic, env.agent_pos.reshape(1, -1))[0])
                if cell_pre != cell_post:
                    edge = (min(cell_pre, cell_post), max(cell_pre, cell_post))
                    if edge in int_bridge_edges and edge not in ep_used_bridge_edges:
                        Bv = 1.0
                        ep_used_bridge_edges.add(edge)

            # Optional EMA normalization before mixing
            Fv_mix, Dv_mix = Fv, Dv
            if int_norm_F is not None and Fv != 0.0:
                Fv_mix = max(0.0, int_norm_F.normalize(Fv))
            if int_norm_D is not None and Dv != 0.0:
                Dv_mix = max(0.0, int_norm_D.normalize(Dv))

            r_int = int_lf * Fv_mix + int_ld * Dv_mix + int_lb * Bv
            r_store = float(r) + intrinsic_beta * r_int
            ep_sum_int += r_int
            ep_sum_F += Fv
            ep_sum_D += Dv
            ep_sum_B += Bv

            replay.add(
                obs=np.ascontiguousarray(obs, np.uint8),
                action=action,
                reward=r_store,
                next_obs=np.ascontiguousarray(next_obs, np.uint8),
                done=done,
            )
            obs = next_obs
            ep_ret += float(r)
            ep_steps += 1
            total_steps += 1
            if step_info.get("success", False) or step_info.get("is_success", False):
                ep_success = True

            c = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
            ep_cells.add(c)
            ep_pos_traj.append(env.agent_pos.copy())

            # Store for geodesic correlation eval
            if len(obs_geo_buffer) < GEO_BUFFER_MAX:
                obs_geo_buffer.append(np.ascontiguousarray(obs, np.uint8))
                pos_geo_buffer.append(env.agent_pos.copy())

            with torch.no_grad():
                obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                preprocess_img(obs_t, depth=args.bit_depth)
                e = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(e, act_t, h_state, s_state, sample=False)

            # ---- Training steps ----
            if total_steps % args.collect_interval == 0 and replay.size > (args.seq_len + 2):
                encoder.train(); decoder.train(); rssm.train()
                reward_model.train(); cont_model.train()
                actor.train(); value_model.train()
                if geo_head is not None:
                    geo_head.train()

                sum_rec = sum_kld = sum_rew = sum_cont = sum_model = 0.0
                sum_actor = sum_value = sum_imag_r = 0.0
                sum_kstep = 0.0
                sum_kstep_unif = sum_geo_total = 0.0
                sum_kstep_kmax = 0.0
                kstep_info_accum: dict[str, float] = {}

                for _ in range(args.train_steps):
                    batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)
                    obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                    act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)
                    rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)
                    done_seq = torch.tensor(batch.dones, dtype=torch.float32, device=device)

                    B, T1 = rew_seq.shape
                    T = T1 - 1
                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                    preprocess_img(x, depth=args.bit_depth)

                    e_t = bottle(encoder, x)  # [B, T+1, embed_dim]

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

                    prior_dist = Normal(prior_m, prior_s)
                    post_dist = Normal(post_m, post_s)

                    recon = bottle(decoder, h_seq, s_seq)
                    target = x[:, 1:T + 1]
                    rec_loss = F.mse_loss(recon, target, reduction="none").sum((2, 3, 4)).mean()
                    kld = torch.max(kl_divergence(post_dist, prior_dist).sum(-1), free_nats).mean()
                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_loss = F.mse_loss(rew_pred, rew_seq[:, :T])
                    cont_logits = bottle(cont_model, h_seq, s_seq)
                    cont_target = (1.0 - done_seq[:, :T]).clamp(0.0, 1.0)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

                    model_loss = rec_loss + args.kl_weight * kld + rew_loss + cont_loss

                    model_opt.zero_grad(set_to_none=True)
                    model_loss.backward()
                    torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip)
                    model_opt.step()

                    l_kstep = torch.tensor(0.0, device=device)
                    l_kstep_unif = torch.tensor(0.0, device=device)
                    l_geo_total = torch.tensor(0.0, device=device)
                    k_curriculum_kmax = float(args.kstep_max_k)
                    if (
                        use_kstep
                        and geo_head is not None
                        and geo_opt is not None
                        and bank is not None
                        and total_steps >= args.kstep_min_steps
                    ):
                        g_seq = geo_head(h_seq.detach(), s_seq.detach())
                        k_curriculum_kmax = float(min(args.kstep_max_k + total_steps // 50_000, 16))
                        l_nce, kstep_info = kstep_nce_with_bank(
                            g_seq,
                            bank,
                            done_seq[:, :T],
                            k_max=int(k_curriculum_kmax),
                            n_neg=args.kstep_negatives,
                            temperature=args.kstep_temperature,
                            max_anchors=args.kstep_max_anchors,
                        )
                        l_unif = uniformity_loss(g_seq.reshape(-1, args.geo_dim))
                        ramp_denom = 10_000
                        if total_steps < args.kstep_min_steps + ramp_denom:
                            kstep_w = args.kstep_weight * (total_steps - args.kstep_min_steps) / ramp_denom
                        else:
                            kstep_w = args.kstep_weight
                        kstep_w = max(0.0, float(kstep_w))
                        l_geo_total = kstep_w * l_nce + args.kstep_unif_weight * l_unif
                        geo_opt.zero_grad(set_to_none=True)
                        l_geo_total.backward()
                        torch.nn.utils.clip_grad_norm_(geo_head.parameters(), args.grad_clip)
                        geo_opt.step()
                        with torch.no_grad():
                            g_mean = g_seq.mean(dim=1)
                            n = g_mean.size(0)
                            if bank_ptr + n <= bank.size(0):
                                bank[bank_ptr : bank_ptr + n] = g_mean
                            else:
                                remain = bank.size(0) - bank_ptr
                                bank[bank_ptr:] = g_mean[:remain]
                                bank[: n - remain] = g_mean[remain:]
                            bank_ptr = (bank_ptr + n) % bank.size(0)
                        l_kstep = l_nce
                        l_kstep_unif = l_unif
                        for k, v in kstep_info.items():
                            kstep_info_accum[k] = kstep_info_accum.get(k, 0.0) + v

                    sum_rec += float(rec_loss.item())
                    sum_kld += float(kld.item())
                    sum_rew += float(rew_loss.item())
                    sum_cont += float(cont_loss.item())
                    sum_model += float(model_loss.item())
                    sum_kstep += float(l_kstep.item())
                    sum_kstep_unif += float(l_kstep_unif.item())
                    sum_geo_total += float(l_geo_total.item())
                    sum_kstep_kmax += k_curriculum_kmax

                    # ---- Actor-critic (imagination) ----
                    B_seq, T_seq, Dh = h_seq.shape
                    Ds = s_seq.size(-1)
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

                        rewards_total = rewards_imag
                        sum_imag_r += float(rewards_total.mean().item())

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

                # ---- TensorBoard logging ----
                n_ts = float(args.train_steps)
                writer.add_scalar("loss/reconstruction", sum_rec / n_ts, total_steps)
                writer.add_scalar("loss/kl", sum_kld / n_ts, total_steps)
                writer.add_scalar("loss/reward_pred", sum_rew / n_ts, total_steps)
                writer.add_scalar("loss/continue", sum_cont / n_ts, total_steps)
                writer.add_scalar("loss/model_total", sum_model / n_ts, total_steps)
                writer.add_scalar("loss/actor", sum_actor / n_ts, total_steps)
                writer.add_scalar("loss/value", sum_value / n_ts, total_steps)
                writer.add_scalar("imag/reward_mean", sum_imag_r / n_ts, total_steps)
                writer.add_scalar("train/exploration_noise", expl_amount, total_steps)
                if use_kstep:
                    writer.add_scalar(
                        "train/kstep_loss_enabled",
                        1.0 if total_steps >= args.kstep_min_steps else 0.0,
                        total_steps,
                    )

                if use_kstep:
                    writer.add_scalar("loss/kstep_nce", sum_kstep / n_ts, total_steps)
                    writer.add_scalar("loss/kstep_unif", sum_kstep_unif / n_ts, total_steps)
                    writer.add_scalar("loss/geo_total", sum_geo_total / n_ts, total_steps)
                    writer.add_scalar("kstep/k_max", sum_kstep_kmax / n_ts, total_steps)
                    for k, v in kstep_info_accum.items():
                        writer.add_scalar(f"loss/{k}", v / n_ts, total_steps)

        if args.expl_decay > 0:
            expl_amount = max(args.expl_min, expl_amount - args.expl_decay)

        if ep_success:
            cumulative_successes += 1
            if first_goal_step is None:
                first_goal_step = total_steps
                writer.add_scalar("eval/first_goal_env_step", float(first_goal_step), 0)
            if use_intrinsic and args.intrinsic_decay_on_success < 1.0:
                old_beta = intrinsic_beta
                intrinsic_beta *= args.intrinsic_decay_on_success
                intrinsic_beta = max(0.005, intrinsic_beta)
                if abs(old_beta - intrinsic_beta) > 1e-8:
                    print(f"    [intrinsic decay] β {old_beta:.6f} → {intrinsic_beta:.6f} (success #{cumulative_successes})")
                writer.add_scalar("intrinsic/beta", intrinsic_beta, total_steps)

        writer.add_scalar("train/episode_return", ep_ret, episode)
        writer.add_scalar("episode/return_env_step", ep_ret, total_steps)
        writer.add_scalar("train/episode_success", 1.0 if ep_success else 0.0, episode)
        writer.add_scalar("train/success_rate", float(cumulative_successes) / float(episode + 1), episode)
        writer.add_scalar("eval/unique_cells_episode", float(len(ep_cells)), episode)
        if len(ep_pos_traj) >= 2:
            writer.add_scalar("eval/bridge_crossings_episode",
                              float(_bridge_crossing_count(geodesic, np.stack(ep_pos_traj))), episode)

        if use_intrinsic and ep_steps > 0:
            writer.add_scalar("intrinsic/r_int_mean", ep_sum_int / ep_steps, total_steps)
            writer.add_scalar("intrinsic/F_mean", ep_sum_F / ep_steps, total_steps)
            writer.add_scalar("intrinsic/D_mean", ep_sum_D / ep_steps, total_steps)
            writer.add_scalar("intrinsic/B_mean", ep_sum_B / ep_steps, total_steps)
            writer.add_scalar("intrinsic/r_store_bonus", intrinsic_beta * ep_sum_int / ep_steps, total_steps)
            writer.add_scalar("intrinsic/beta", intrinsic_beta, total_steps)
            writer.add_scalar("intrinsic/D_gated", 1.0 if total_steps >= args.kstep_min_steps else 0.0, total_steps)

        int_str = ""
        if use_intrinsic and ep_steps > 0:
            int_str = f"  r_int={ep_sum_int/ep_steps:.4f}(F={ep_sum_F/ep_steps:.3f} D={ep_sum_D/ep_steps:.3f} B={ep_sum_B/ep_steps:.3f})"
        print(f"  Ep {episode+1}/{args.max_episodes}  ret={ep_ret:.2f}  steps={ep_steps}  "
              f"cells={len(ep_cells)}  total={total_steps}  success={ep_success}{int_str}")

        # ---- h+s latent diagnostics ----
        if args.diag_interval > 0 and (episode + 1) % args.diag_interval == 0:
            # Grab a batch of recent observations for diagnostics
            if replay.size > 256:
                diag_batch = replay.sample_sequences(min(64, args.batch_size), 1)
                diag_obs = torch.tensor(diag_batch.obs[:, 0], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                preprocess_img(diag_obs, depth=args.bit_depth)
                diag_e = encoder(diag_obs)
                diag_h, diag_s = rssm.get_init_state(diag_e)
                diag_z = rssm_latent(diag_h, diag_s)
                diags = compute_latent_diagnostics(diag_z, prefix="latent_hs")
                for k, v in diags.items():
                    writer.add_scalar(k, v, total_steps)
                print(f"    [diag h+s] mean_d={diags['latent_hs/mean_pairwise_dist']:.4f}  "
                      f"eff_rank={diags['latent_hs/effective_rank']:.1f}  "
                      f"mean_std={diags['latent_hs/mean_per_dim_std']:.4f}  "
                      f"dead_dims={diags['latent_hs/num_dead_dims']:.0f}")

        # ---- Geodesic correlation in h+s latent ----
        if args.geo_corr_interval > 0 and (episode + 1) % args.geo_corr_interval == 0:
            if len(obs_geo_buffer) >= 100:
                obs_arr = np.stack(obs_geo_buffer, axis=0)
                pos_arr = np.stack(pos_geo_buffer, axis=0)
                geo_corr = compute_hs_geodesic_correlation(
                    encoder, rssm, obs_arr, pos_arr, geodesic, device, args.bit_depth, n_pairs=2000)
                for k, v in geo_corr.items():
                    writer.add_scalar(k, v, total_steps)
                print(f"    [geo h+s] ρ={geo_corr['geo_hs/spearman']:.4f}  "
                      f"mean_d={geo_corr.get('geo_hs/mean_latent_dist', 0):.4f}")
                if geo_head is not None:
                    ghc = compute_geohead_geodesic_correlation(
                        geo_head,
                        encoder,
                        rssm,
                        obs_arr,
                        pos_arr,
                        geodesic,
                        device,
                        args.bit_depth,
                        n_pairs=2000,
                    )
                    for k, v in ghc.items():
                        writer.add_scalar(k, v, total_steps)
                    print(f"    [geo head] ρ={ghc.get('geo_head/spearman', 0):.4f}")

        # ---- Periodic eval ----
        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            ev = run_periodic_eval(env, encoder, rssm, actor, device, args.bit_depth,
                                   args.eval_episodes, geodesic, action_repeat)
            for k, v in ev.items():
                writer.add_scalar(f"eval/{k}", v, total_steps)
            print(f"    [eval] ret={ev['return_mean']:.2f}±{ev['return_std']:.2f}  "
                  f"cells={ev['unique_cells_mean']:.1f}  bridges={ev['bridge_crossings_mean']:.1f}  "
                  f"success={ev['success_rate']:.2f}")

        # ---- Save checkpoint ----
        if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(out_dir, f"checkpoint_ep{episode+1}.pt")
            ckpt_d = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "rssm": rssm.state_dict(),
                "reward_model": reward_model.state_dict(),
                "cont_model": cont_model.state_dict(),
                "actor": actor.state_dict(),
                "value_model": value_model.state_dict(),
                "episode": episode + 1,
                "total_steps": total_steps,
                "geo_mode": args.geo_mode,
            }
            if geo_head is not None and bank is not None:
                ckpt_d["geo_head"] = geo_head.state_dict()
                ckpt_d["bank"] = bank.detach().cpu()
                ckpt_d["bank_ptr"] = bank_ptr
            torch.save(ckpt_d, ckpt_path)

    # ---- Final save ----
    final_ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "rssm": rssm.state_dict(),
        "reward_model": reward_model.state_dict(),
        "cont_model": cont_model.state_dict(),
        "actor": actor.state_dict(),
        "value_model": value_model.state_dict(),
        "episode": args.max_episodes,
        "total_steps": total_steps,
        "geo_mode": args.geo_mode,
    }
    if geo_head is not None and bank is not None:
        final_ckpt["geo_head"] = geo_head.state_dict()
        final_ckpt["bank"] = bank.detach().cpu()
        final_ckpt["bank_ptr"] = bank_ptr
    torch.save(final_ckpt, os.path.join(out_dir, "world_model_final.pt"))

    env.close()
    writer.close()
    print(f"\nDone. Mode={args.geo_mode}  total_steps={total_steps}")


if __name__ == "__main__":
    main(build_parser().parse_args())
