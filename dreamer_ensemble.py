#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict, deque

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


class GeoDisagreementEnsemble(nn.Module):
    """Ensemble of predictors for next-step geo embedding g_{t+1}.

    Each member outputs a raw D-vector (not L2-normalized) so disagreement
    (variance across members) is not capped by the unit-sphere geometry. Training
    still uses MSE to ``g_tgt`` from GeoHead (already normalized).

    Optional **randomized prior functions** (Osband et al., 2018): each member is
    ``m_k(x) + β·p_k(x)`` where ``p_k`` is a fixed randomly initialized MLP
    (never trained), preserving cross-member diversity as data grows.
    """

    def __init__(
        self,
        h_dim: int,
        s_dim: int,
        geo_dim: int,
        hidden_dim: int = 256,
        ensemble_size: int = 5,
        *,
        prior_scale: float = 0.0,
        prior_hidden_dim: int | None = None,
    ):
        super().__init__()
        in_dim = h_dim + s_dim
        self.ensemble_size = int(ensemble_size)
        self.geo_dim = int(geo_dim)
        self.register_buffer("prior_scale", torch.tensor(float(prior_scale)))
        ph = int(prior_hidden_dim) if prior_hidden_dim is not None else int(hidden_dim)
        self.members = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ELU(),
                nn.Linear(hidden_dim, geo_dim),
            )
            for _ in range(self.ensemble_size)
        ])
        self.priors: nn.ModuleList | None
        if float(prior_scale) > 0.0:
            self.priors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(in_dim, ph),
                        nn.ELU(),
                        nn.Linear(ph, geo_dim),
                    )
                    for _ in range(self.ensemble_size)
                ]
            )
            for pr in self.priors:
                for p in pr.parameters():
                    p.requires_grad_(False)
        else:
            self.priors = None

    def _preds_list(self, x: torch.Tensor) -> list[torch.Tensor]:
        beta = self.prior_scale
        if self.priors is not None:
            return [m(x) + beta * pr(x) for m, pr in zip(self.members, self.priors)]
        return [m(x) for m in self.members]

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, s], dim=-1)
        return torch.stack(self._preds_list(x), dim=0)

    def bootstrap_training_loss(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        g_tgt: torch.Tensor,
        *,
        keep_prob: float,
    ) -> torch.Tensor:
        """Per-member MSE to g_tgt on an independent random row mask (bootstrap), then mean over members."""
        x = torch.cat([h, s], dim=-1)
        n = h.size(0)
        device = h.device
        p = float(keep_prob)
        losses: list[torch.Tensor] = []
        for pred in self._preds_list(x):
            mask = torch.rand(n, device=device) < p
            if int(mask.sum()) > 0:
                loss_k = F.mse_loss(pred[mask], g_tgt[mask])
            else:
                loss_k = F.mse_loss(pred, g_tgt)
            losses.append(loss_k)
        return torch.stack(losses).mean()

    @staticmethod
    def disagreement(preds: torch.Tensor) -> torch.Tensor:
        """Return per-sample ensemble variance scalar.

        preds: [K, ..., D]
        returns: [...]
        """
        mean = preds.mean(dim=0)
        return (preds - mean).pow(2).mean(dim=0).sum(dim=-1)


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
    """InfoNCE with memory-bank negatives.

    Negatives are mined by similarity to each anchor (hard-negative mining),
    with random subsampling from a top-similarity pool to reduce false negatives.
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
    bank_n = int(bank.size(0))
    n_neg_eff = min(n_neg, bank_n)
    # Mine hard negatives by maximum cosine similarity to each anchor.
    # For efficiency, score a candidate subset when bank is very large.
    cand = min(bank_n, max(n_neg_eff * 8, 2048))
    if cand < bank_n:
        cand_idx = torch.randperm(bank_n, device=g_seq.device)[:cand]
        bank_cand = bank[cand_idx]
    else:
        cand_idx = None
        bank_cand = bank
    sim = anchors @ bank_cand.T  # [N_pairs, cand]
    # Middle-ground mining: sample from a moderately hard top-similarity pool.
    topk_pool = min(cand, max(n_neg_eff * 4, n_neg_eff))
    _, pool_idx = torch.topk(sim, k=topk_pool, dim=1, largest=True)
    rand_pick = torch.randint(0, topk_pool, (sim.size(0), n_neg_eff), device=g_seq.device)
    hard_idx = pool_idx.gather(1, rand_pick)
    if cand_idx is not None:
        neg_idx = cand_idx[hard_idx]
    else:
        neg_idx = hard_idx
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


def knn_novelty_from_memory(
    e_t: np.ndarray,
    memory: list[np.ndarray],
    *,
    k: int,
    clip_min: float,
    clip_max: float,
    use_density_count: bool,
    density_radius: float,
) -> float:
    """NGU-style episodic novelty over geometric embedding memory."""
    if not memory:
        return float(clip_max)
    mem = np.asarray(memory, dtype=np.float32)
    d = np.linalg.norm(mem - e_t.reshape(1, -1), axis=1)
    if use_density_count:
        cnt = int(np.sum(d <= max(float(density_radius), 1e-6)))
        return float(1.0 / math.sqrt(max(cnt, 1)))
    kk = max(1, min(int(k), len(d)))
    dk = float(np.partition(d, kk - 1)[kk - 1])
    lo = float(clip_min)
    hi = max(float(clip_max), lo + 1e-6)
    return float(np.clip(dk, lo, hi))


@torch.no_grad()
def refresh_embed_memory_from_replay(
    *,
    replay: ReplayBuffer,
    encoder: ConvEncoder,
    rssm: RSSM,
    geo_head_ema: GeoHead,
    device: torch.device,
    bit_depth: int,
    target_size: int,
    batch_size: int,
) -> list[np.ndarray]:
    """Recompute embedding memory from replay with current EMA head."""
    n_target = max(0, int(target_size))
    if n_target <= 0 or replay.size < 2:
        return []
    out: list[np.ndarray] = []
    bs = max(8, int(batch_size))
    while len(out) < n_target:
        cur_bs = min(bs, n_target - len(out))
        batch = replay.sample_sequences(cur_bs, 1)
        obs0 = torch.tensor(batch.obs[:, 0], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        preprocess_img(obs0, depth=bit_depth)
        e0 = encoder(obs0)
        h0, s0 = rssm.get_init_state(e0)
        g0 = geo_head_ema(h0, s0).detach().cpu().numpy().astype(np.float32)
        out.extend([g0[i].copy() for i in range(g0.shape[0])])
    return out[:n_target]


def sample_sequences_with_priority(
    replay: ReplayBuffer,
    priorities: np.ndarray,
    *,
    batch_size: int,
    seq_len: int,
    alpha: float,
    beta_is: float,
    candidate_multiplier: int,
) -> tuple[ReplayBuffer.Batch, np.ndarray]:
    """Crude PER-style sequence sampler over replay start indices."""
    if replay.size <= seq_len + 1:
        batch = replay.sample_sequences(batch_size, seq_len)
        return batch, np.ones((batch_size,), dtype=np.float32)

    n_cand = max(int(batch_size) * max(int(candidate_multiplier), 2), int(batch_size) * 8)
    candidates = np.random.randint(0, replay.size - seq_len, size=n_cand)
    if replay.full:
        wrap_mask = (candidates < replay.idx) & (replay.idx < candidates + seq_len)
        candidates = candidates[~wrap_mask]

    valid: list[int] = []
    for start in candidates.tolist():
        end = int(start) + int(seq_len)
        if np.any(replay.dones[int(start) : end - 1]):
            continue
        valid.append(int(start))
        if len(valid) >= max(batch_size * 4, batch_size):
            break
    if not valid:
        batch = replay.sample_sequences(batch_size, seq_len)
        return batch, np.ones((batch_size,), dtype=np.float32)

    v = np.asarray(valid, dtype=np.int64)
    p = np.maximum(priorities[v], 1e-8).astype(np.float64)
    p = np.power(p, max(float(alpha), 0.0))
    den = float(p.sum())
    if not np.isfinite(den) or den <= 0:
        prob = np.ones_like(p, dtype=np.float64) / float(len(p))
    else:
        prob = p / den

    chosen_local = np.random.choice(len(v), size=int(batch_size), replace=True, p=prob)
    starts = v[chosen_local]

    seq_idx = starts[:, None] + np.arange(seq_len)[None, :]
    batch = ReplayBuffer.Batch(
        replay.obs[seq_idx],
        replay.actions[seq_idx],
        replay.rews[seq_idx],
        replay.dones[seq_idx],
    )

    # Importance weights: w_i = (N * P_i)^(-beta), normalized by max.
    p_i = prob[chosen_local]
    n_eff = max(len(v), 1)
    w = np.power(n_eff * np.maximum(p_i, 1e-12), -max(float(beta_is), 0.0)).astype(np.float32)
    w /= max(float(np.max(w)), 1e-8)
    return batch, w


def _graph_frontier_nodes(
    adj: list[list[int]],
    visit_count: list[int],
    node_disag: list[float],
    *,
    reliable_min_visits: int,
    top_quantile: float,
    disagreement_weight: float,
) -> list[int]:
    n = len(adj)
    if n == 0:
        return []
    deg = np.asarray([max(len(a), 1) for a in adj], dtype=np.float32)
    inv_density = 1.0 / np.sqrt(deg)
    v = np.asarray(visit_count, dtype=np.float32)
    reliable = v >= max(1, int(reliable_min_visits))
    d = np.maximum(np.asarray(node_disag, dtype=np.float32), 0.0)
    if d.size and np.any(np.isfinite(d)):
        den = float(np.nanpercentile(d, 95)) + 1e-6
        d = np.clip(d / den, 0.0, 1.0)
    raw = (inv_density + float(disagreement_weight) * d) * reliable.astype(np.float32)
    valid = reliable & np.isfinite(raw)
    if not np.any(valid):
        return []
    q = float(np.clip(top_quantile, 10.0, 99.5))
    thr = float(np.percentile(raw[valid], q))
    out = np.where(valid & (raw >= thr))[0].tolist()
    if not out:
        out = [int(np.argmax(raw))]
    return out


def _graph_distance_to_frontier(
    adj: list[list[int]],
    start_idx: int,
    frontier_nodes: list[int],
) -> float:
    if start_idx < 0 or start_idx >= len(adj) or not frontier_nodes:
        return float("inf")
    frontier_set = set(int(x) for x in frontier_nodes)
    if start_idx in frontier_set:
        return 0.0
    q: deque[tuple[int, int]] = deque([(int(start_idx), 0)])
    seen = {int(start_idx)}
    while q:
        u, d = q.popleft()
        for v in adj[u]:
            vv = int(v)
            if vv in seen:
                continue
            if vv in frontier_set:
                return float(d + 1)
            seen.add(vv)
            q.append((vv, d + 1))
    return float("inf")


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


GEO_HEAD_FREEZE_AFTER_STEPS = float("inf")
GEO_HEAD_EMA_DECAY = 0.995

INTRINSIC_PRESETS = {
    "baseline": (0.0, 0.0, 0.0),
    "knn_only": (1.0, 0.0, 0.0),
    "dvar_only": (0.0, 0.2, 0.0),
    "knn_dvar": (1.0, 0.2, 0.0),
    "knn_dvar_frontier": (1.0, 0.2, 0.1),
}

def intrinsic_beta_linear_schedule(
    total_steps: int,
    *,
    intrinsic_scale: float,
    kstep_min_steps: int,
    explore_period: int,
    decay_period: int,
) -> float:
    """β = 0 until kstep_min_steps, then intrinsic_scale for explore_period env steps, then linear decay to 0.

    k0 = kstep_min_steps. Plateau is [k0, k0 + explore_period); decay is
    [k0 + explore_period, k0 + explore_period + decay_period].
    """
    k0 = int(kstep_min_steps)
    plateau_end = k0 + int(explore_period)
    dper = int(max(decay_period, 0))
    decay_end = plateau_end + dper
    s = float(intrinsic_scale)
    if total_steps < k0:
        return 0.0
    return s
    """
    if total_steps < plateau_end:
        return s
    if dper <= 0 or total_steps >= decay_end:
        return 0.0
    t = total_steps - plateau_end
    return s * max(0.0, 1.0 - t / float(dper))
    """


def _trapezoid_auc(xs: list[float], ys: list[float]) -> float:
    """Area under y(x) with x = env steps (trapezoid rule)."""
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    s = 0.0
    for i in range(len(xs) - 1):
        s += (xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) * 0.5
    return float(s)


def _first_step_at_least(xs: list[float], ys: list[float], thr: float) -> float | None:
    for x, y in zip(xs, ys):
        if y >= thr:
            return float(x)
    return None


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


def _build_kstep_disag_targets_from_seq(
    h_seq: torch.Tensor,
    s_seq: torch.Tensor,
    done_seq: torch.Tensor | None,
    *,
    k_min: int,
    k_max: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build (h_t, s_t) -> (h_{t+k}, s_{t+k}) pairs with k ~ Uniform(k_min, k_max).

    Targets are sampled from real replay sequences (same sampled batch), with terminal
    boundaries respected when done_seq is provided.
    """
    B, T, Dh = h_seq.shape
    Ds = int(s_seq.size(-1))
    device = h_seq.device
    kmn = max(1, int(k_min))
    kmx = max(kmn, int(k_max))
    if T < (kmn + 1):
        z_h = torch.empty((0, Dh), device=device, dtype=h_seq.dtype)
        z_s = torch.empty((0, Ds), device=device, dtype=s_seq.dtype)
        return z_h, z_s, z_h, z_s, {"n_pairs": 0.0, "k_mean": 0.0, "k_min": float(kmn), "k_max": float(kmx)}

    h_src_list: list[torch.Tensor] = []
    s_src_list: list[torch.Tensor] = []
    h_tgt_list: list[torch.Tensor] = []
    s_tgt_list: list[torch.Tensor] = []
    ks: list[float] = []

    done_np = done_seq.detach().cpu().numpy() if done_seq is not None else None
    for b in range(B):
        for t in range(T):
            max_delta = min(kmx, T - 1 - t)
            if max_delta < kmn:
                continue
            valid_k: list[int] = []
            for d in range(kmn, max_delta + 1):
                if done_np is not None and bool(done_np[b, t : t + d].max() > 0.5):
                    continue
                valid_k.append(d)
            if not valid_k:
                continue
            k = int(valid_k[np.random.randint(0, len(valid_k))])
            j = t + k
            h_src_list.append(h_seq[b, t])
            s_src_list.append(s_seq[b, t])
            h_tgt_list.append(h_seq[b, j])
            s_tgt_list.append(s_seq[b, j])
            ks.append(float(k))

    if not h_src_list:
        z_h = torch.empty((0, Dh), device=device, dtype=h_seq.dtype)
        z_s = torch.empty((0, Ds), device=device, dtype=s_seq.dtype)
        return z_h, z_s, z_h, z_s, {"n_pairs": 0.0, "k_mean": 0.0, "k_min": float(kmn), "k_max": float(kmx)}

    h_src = torch.stack(h_src_list, dim=0).detach().reshape(-1, Dh)
    s_src = torch.stack(s_src_list, dim=0).detach().reshape(-1, Ds)
    h_tgt = torch.stack(h_tgt_list, dim=0).detach().reshape(-1, Dh)
    s_tgt = torch.stack(s_tgt_list, dim=0).detach().reshape(-1, Ds)
    return h_src, s_src, h_tgt, s_tgt, {
        "n_pairs": float(len(ks)),
        "k_mean": float(np.mean(np.asarray(ks, dtype=np.float32))),
        "k_min": float(kmn),
        "k_max": float(kmx),
    }


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
    p.add_argument("--expl_novelty_gain", type=float, default=0.0,
                   help="Scale exploration noise by (1 + gain * r_int) at each env step")
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
                   help="Negatives per anchor (hard-mined as farthest samples from geo memory bank)")
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
    p.add_argument(
        "--geo_bank_prefill_batches",
        type=int,
        default=256,
        help="Replay-driven bank prefill batches (random-walk seed data) before k-step contrastive updates",
    )
    p.add_argument("--geo_lr", type=float, default=3e-4, help="GeoHead AdamW learning rate")
    p.add_argument("--disag_ensemble_size", type=int, default=5,
                   help="Number of ensemble members for Option-A geo disagreement")
    p.add_argument("--disag_hidden_dim", type=int, default=256,
                   help="Hidden size for each geo disagreement predictor")
    p.add_argument("--disag_lr", type=float, default=3e-4,
                   help="Adam learning rate for geo disagreement ensemble")
    p.add_argument("--disag_target_noise", type=float, default=0.0,
                   help="Optional Gaussian noise added to geo targets during ensemble training")
    p.add_argument("--disag_reward_clip", type=float, default=5.0,
                   help="Clip for intrinsic disagreement reward before beta scaling")
    p.add_argument(
        "--disag_bootstrap_prob",
        type=float,
        default=0.7,
        help="Per-ensemble-member row mask: train MSE on random fraction ~p of batch rows (bootstrap diversity)",
    )
    p.add_argument(
        "--disag_kstep_min_k",
        type=int,
        default=5,
        help="Min k for replay-based multistep disagreement targets g_{t+k}.",
    )
    p.add_argument(
        "--disag_kstep_max_k",
        type=int,
        default=0,
        help="Max k for replay-based multistep disagreement; <=0 uses --kstep_max_k.",
    )
    p.add_argument(
        "--disag_reward_source",
        type=str,
        default="one_step",
        choices=["one_step", "kstep"],
        help="Which disagreement signal drives intrinsic reward D: one_step (Dv_mix) or kstep (Dv_kstep_mix).",
    )
    p.add_argument(
        "--disag_imag_ema_alpha",
        type=float,
        default=0.01,
        help="EMA alpha for batch mean/std of imag disagreement (0 = off, use raw clamped disag).",
    )
    p.add_argument(
        "--disag_prior_scale",
        type=float,
        default=1.0,
        help="Osband et al. 2018 randomized priors: pred_k = m_k(x)+β·p_k(x) with fixed p_k; 0 disables",
    )
    p.add_argument(
        "--disag_prior_hidden_dim",
        type=int,
        default=0,
        help="Hidden width of each fixed prior MLP; 0 means match --disag_hidden_dim",
    )

    # Intrinsic reward during training
    p.add_argument("--intrinsic_ablation", type=str, default="baseline",
                   choices=["baseline", "knn_only", "dvar_only", "knn_dvar", "knn_dvar_frontier"],
                   help="Intrinsic preset: baseline; knn_only; dvar_only; knn_dvar; knn_dvar_frontier.")
    p.add_argument("--int_lambda_f", type=float, default=1.0,
                   help="Weight on frontier novelty F (overridden by preset)")
    p.add_argument("--int_lambda_d", type=float, default=0.15,
                   help="Weight on raw-vs-geo disagreement D (overridden by preset)")
    p.add_argument("--lambda_knn", type=float, default=None,
                   help="Explicit weight on kNN novelty F (defaults to preset λ_f)")
    p.add_argument("--lambda_dvar", type=float, default=None,
                   help="Explicit weight on one-step disagreement D (defaults to preset λ_d)")
    p.add_argument("--lambda_frontier", type=float, default=None,
                   help="Explicit weight on replay-graph frontier bonus (defaults to preset value)")
    p.add_argument("--intrinsic_scale", type=float, default=0.25,
                   help="Global scale beta: imagination r += beta * r_int (sweep with λs for ~0.01–0.05 r_store_bonus)")
    p.add_argument(
        "--explore_period",
        type=int,
        default=50_000,
        help="Env steps after kstep_min_steps to keep beta=intrinsic_scale before linear decay",
    )
    p.add_argument(
        "--decay_period",
        type=int,
        default=100_000,
        help="Env steps to linearly decay beta from intrinsic_scale to 0 (after explore_period)",
    )
    p.add_argument("--intrinsic_normalize", action="store_true",
                   help="EMA-normalize F and D before mixing (makes lambda ratios "
                   "reflect effective weight regardless of raw magnitude)")
    p.add_argument("--novelty_knn_k", type=int, default=10,
                   help="k for kNN novelty in contrastive embedding memory")
    p.add_argument("--novelty_clip_min", type=float, default=0.0,
                   help="Lower clip for kNN novelty")
    p.add_argument("--novelty_clip_max", type=float, default=5.0,
                   help="Upper clip for kNN novelty")
    p.add_argument("--novelty_use_density_count", action="store_true",
                   help="Use 1/sqrt(count-in-radius) novelty variant")
    p.add_argument("--novelty_density_radius", type=float, default=0.25,
                   help="Radius for density-count novelty variant")
    p.add_argument("--novelty_memory_refresh_interval", type=int, default=15000,
                   help="Steps between replay-memory embedding refreshes; <=0 disables")
    p.add_argument("--novelty_memory_refresh_samples", type=int, default=4096,
                   help="How many replay states to re-embed on each memory refresh")
    p.add_argument("--frontier_memory_size", type=int, default=50000,
                   help="Max replay-graph nodes kept for frontier scoring")
    p.add_argument("--frontier_knn_k", type=int, default=6,
                   help="Optional kNN edges per new node in embedding space (0 disables)")
    p.add_argument("--frontier_knn_threshold", type=float, default=0.6,
                   help="Distance threshold for adding embedding-space kNN edges")
    p.add_argument("--frontier_reliable_min_visits", type=int, default=2,
                   help="Min node visits to be eligible as frontier")
    p.add_argument("--frontier_top_quantile", type=float, default=80.0,
                   help="Top quantile of frontier raw score to keep as frontier nodes")
    p.add_argument("--frontier_tau", type=float, default=3.0,
                   help="Tau in exp(-d_graph/tau) frontier reward")
    p.add_argument("--frontier_disagreement_weight", type=float, default=0.5,
                   help="Weight on one-step disagreement in frontier-node score")
    p.add_argument("--prioritized_replay_enable", action="store_true",
                   help="Use novelty-prioritized replay sampling for training sequences")
    p.add_argument("--prioritized_replay_alpha", type=float, default=0.6,
                   help="Priority exponent alpha in p_i=(beta*r_int+eps)^alpha")
    p.add_argument("--prioritized_replay_beta_is", type=float, default=0.4,
                   help="Importance-sampling exponent beta for PER correction")
    p.add_argument("--prioritized_replay_eps", type=float, default=1e-4,
                   help="Epsilon floor in priority p_i=(beta*r_int+eps)^alpha")
    p.add_argument("--prioritized_replay_candidate_multiplier", type=int, default=32,
                   help="Number of candidate starts = multiplier * batch_size before weighted resampling")

    # Diagnostics
    p.add_argument("--diag_interval", type=int, default=40,
                   help="Run h+s latent diagnostics every N training episodes")
    p.add_argument("--geo_corr_interval", type=int, default=80,
                   help="Run geodesic correlation eval every N episodes (expensive)")
    p.add_argument("--summary_unique_cells_thr", type=float, default=12.0,
                   help="Threshold for summary/first_step_unique_cells_mean")
    p.add_argument("--summary_bridge_thr", type=float, default=1.0,
                   help="Threshold for summary/first_step_bridge_crossings_mean")
    p.add_argument("--summary_success_thr", type=float, default=0.4,
                   help="Threshold for summary/first_step_success_rate")

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

    if args.intrinsic_ablation in ["knn_only", "dvar_only", "knn_dvar", "knn_dvar_frontier"] and args.geo_mode != "kstep":
        raise ValueError("Geo intrinsic variants require --geo_mode kstep")

    use_intrinsic = args.intrinsic_ablation != "baseline"
    int_lf, int_ld, int_lf_frontier = INTRINSIC_PRESETS[args.intrinsic_ablation]
    lambda_knn = float(args.lambda_knn) if args.lambda_knn is not None else float(int_lf)
    lambda_dvar = float(args.lambda_dvar) if args.lambda_dvar is not None else float(int_ld)
    lambda_frontier = float(args.lambda_frontier) if args.lambda_frontier is not None else float(int_lf_frontier)
    intrinsic_beta = float(args.intrinsic_scale) if use_intrinsic else 0.0
    use_geo_head = (args.geo_mode == "kstep") or (use_intrinsic and (lambda_dvar > 0 or lambda_knn > 0 or lambda_frontier > 0))
    use_geo_disagreement = use_intrinsic and lambda_dvar > 0

    print(f"Device: {device} Geo mode: {args.geo_mode}")
    print(f"  K-step InfoNCE on GeoHead: {use_geo_head} (weight={args.kstep_weight})")
    if use_intrinsic:
        print(
            f"  Intrinsic: {args.intrinsic_ablation}  λ_f={int_lf} λ_d={int_ld} λ_frontier={int_lf_frontier}  "
            f"β_scale={args.intrinsic_scale}"
        )
        print(
            f"  Intrinsic mix override: λ_knn={lambda_knn} λ_dvar={lambda_dvar} "
            f"λ_frontier={lambda_frontier}"
        )
        print(
            f"  Intrinsic β schedule: β=0 until step {args.kstep_min_steps}, then β={args.intrinsic_scale} "
            f"for {args.explore_period} steps, then linear decay to 0 over {args.decay_period} env steps"
        )
        print(f"  Intrinsic normalize={args.intrinsic_normalize}")
    if use_geo_head:
        print(
            f"  GeoHead: dim={args.geo_dim}  bank={args.geo_bank_size}, "
            f"temp={args.kstep_temperature}, separate AdamW; trains from step 0 "
            f"(intrinsic D still gated until {args.kstep_min_steps} env steps)"
        )
    if use_geo_disagreement:
        print(
            f"  Geo disagreement ensemble: K={args.disag_ensemble_size} hidden={args.disag_hidden_dim} "
            f"lr={args.disag_lr} clip={args.disag_reward_clip} bootstrap_p={args.disag_bootstrap_prob} "
            f"imag_ema_alpha={args.disag_imag_ema_alpha} "
            f"prior_β={args.disag_prior_scale} prior_h={args.disag_prior_hidden_dim or args.disag_hidden_dim}"
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
    geo_head_ema: GeoHead | None = None
    geo_opt = None
    bank: torch.Tensor | None = None
    bank_ptr = 0
    bank_loaded_from_ckpt = False
    if use_geo_head:
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
                bank_loaded_from_ckpt = True
            if "bank_ptr" in ckpt:
                bank_ptr = int(ckpt["bank_ptr"])
        geo_head_ema = GeoHead(args.deter_dim, args.stoch_dim, geo_dim=args.geo_dim).to(device)
        if ckpt is not None and "geo_head_ema" in ckpt:
            geo_head_ema.load_state_dict(ckpt["geo_head_ema"])
        else:
            geo_head_ema.load_state_dict(geo_head.state_dict())
        for p in geo_head_ema.parameters():
            p.requires_grad_(False)
        geo_head_ema.eval()

    geo_disag: GeoDisagreementEnsemble | None = None
    geo_disag_opt = None
    geo_disag_kstep: GeoDisagreementEnsemble | None = None
    geo_disag_kstep_opt = None
    if use_geo_disagreement:
        prior_h = int(args.disag_prior_hidden_dim) if int(args.disag_prior_hidden_dim) > 0 else None
        geo_disag = GeoDisagreementEnsemble(
            args.deter_dim,
            args.stoch_dim,
            args.geo_dim,
            hidden_dim=args.disag_hidden_dim,
            ensemble_size=args.disag_ensemble_size,
            prior_scale=float(args.disag_prior_scale),
            prior_hidden_dim=prior_h,
        ).to(device)
        geo_disag_trainable = [p for p in geo_disag.parameters() if p.requires_grad]
        geo_disag_opt = torch.optim.Adam(geo_disag_trainable, lr=args.disag_lr, eps=args.adam_eps)
        if ckpt is not None and "geo_disag" in ckpt:
            inc = geo_disag.load_state_dict(ckpt["geo_disag"], strict=False)
            if inc.missing_keys:
                print(f"    [geo_disag] checkpoint missing {len(inc.missing_keys)} keys (re-init those modules)")
            if inc.unexpected_keys:
                print(f"    [geo_disag] checkpoint unexpected {len(inc.unexpected_keys)} keys (ignored)")

    world_params = (
        list(encoder.parameters()) + list(decoder.parameters())
        + list(rssm.parameters()) + list(reward_model.parameters())
        + list(cont_model.parameters())
    )
    model_opt = torch.optim.Adam(world_params, lr=args.model_lr, eps=args.adam_eps)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=args.adam_eps)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=args.value_lr, eps=args.adam_eps)

    replay = ReplayBuffer(args.replay_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    replay_priority = np.ones((args.replay_capacity,), dtype=np.float32)
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

    int_norm_F: _EMANormalizer | None = None
    int_norm_D: _EMANormalizer | None = None
    int_norm_D_kstep: _EMANormalizer | None = None
    if use_intrinsic and args.intrinsic_normalize:
        int_norm_F = _EMANormalizer(alpha=0.01)
        int_norm_D = _EMANormalizer(alpha=0.01)
        int_norm_D_kstep = _EMANormalizer(alpha=0.01)

    # E_replay over contrastive embeddings and explicit replay-graph state.
    embed_memory: list[np.ndarray] = []
    graph_adj: list[list[int]] = []
    graph_visit_count: list[int] = []
    graph_node_disag: list[float] = []
    prev_graph_idx = -1

    total_steps = 0
    geo_head_frozen = False
    expl_amount = args.expl_amount
    disag_imag_ema_mean: float | None = None
    disag_imag_ema_std: float = 1.0

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

    if (
        geo_head is not None
        and bank is not None
        and not bank_loaded_from_ckpt
        and replay.size > 3
        and int(args.geo_bank_prefill_batches) > 0
    ):
        print("  Prefilling geo bank from random-walk replay ...")
        encoder_prev, rssm_prev, geo_prev = encoder.training, rssm.training, geo_head.training
        encoder.eval(); rssm.eval(); geo_head.eval()
        filled = 0
        bs = max(8, min(int(args.batch_size), 256, replay.size - 2))
        seq_len_prefill = 2
        with torch.no_grad():
            for _ in range(int(args.geo_bank_prefill_batches)):
                if filled >= bank.size(0):
                    break
                batch = replay.sample_sequences(bs, seq_len_prefill)
                obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)
                x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                preprocess_img(x, depth=args.bit_depth)
                e_t = bottle(encoder, x)
                h_t, s_t = rssm.get_init_state(e_t[:, 0])
                h_t = rssm.deterministic_state_fwd(h_t, s_t, act_seq[:, 0])
                pm, ps = rssm.state_posterior(h_t, e_t[:, 1])
                s_t = pm
                g = geo_head(h_t, s_t)
                n = min(g.size(0), bank.size(0) - filled)
                bank[filled : filled + n] = g[:n]
                filled += n
        if filled < bank.size(0):
            bank[filled:] = F.normalize(torch.randn_like(bank[filled:]), dim=-1)
        bank_ptr = 0
        if encoder_prev:
            encoder.train()
        if rssm_prev:
            rssm.train()
        if geo_prev and not geo_head_frozen:
            geo_head.train()
        print(f"    geo bank prefill: {filled}/{bank.size(0)} from replay, {bank.size(0)-filled} random fallback")

    first_goal_step = None
    cumulative_successes = 0
    eval_snapshots: list[dict[str, float]] = []
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

        ep_sum_int = ep_sum_F = ep_sum_D = ep_sum_D_kstep = 0.0
        ep_sum_knn_novel = ep_sum_frontier = 0.0
        prev_graph_idx = -1

        while not done:
            encoder.eval(); rssm.eval(); actor.eval()
            if (
                geo_head_ema is not None
                and int(args.novelty_memory_refresh_interval) > 0
                and total_steps > 0
                and (total_steps % int(args.novelty_memory_refresh_interval) == 0)
                and replay.size > 32
            ):
                refreshed = refresh_embed_memory_from_replay(
                    replay=replay,
                    encoder=encoder,
                    rssm=rssm,
                    geo_head_ema=geo_head_ema,
                    device=device,
                    bit_depth=args.bit_depth,
                    target_size=min(int(args.frontier_memory_size), int(args.novelty_memory_refresh_samples)),
                    batch_size=min(int(args.batch_size), 256),
                )
                embed_memory = refreshed
                graph_adj = [[] for _ in range(len(embed_memory))]
                graph_visit_count = [1 for _ in range(len(embed_memory))]
                graph_node_disag = [0.0 for _ in range(len(embed_memory))]
                prev_graph_idx = -1
            with torch.no_grad():
                action_t, _ = actor.get_action(h_state, s_state, deterministic=False)

            # --- Intrinsic: F (frontier) and D (disagreement) before env step ---
            Fv = Dv = Dv_kstep = 0.0
            knn_novelty = frontier_bonus = 0.0
            intrinsic_needs_geo_delay = geo_head is not None
            intrinsic_mix_ready = (not intrinsic_needs_geo_delay) or (
                total_steps >= args.kstep_min_steps
            )

            # D: ensemble disagreement over next-step geo prediction in g-space.
            if use_intrinsic and intrinsic_mix_ready:
                geo_ready = lambda_dvar > 0 and geo_head is not None and geo_disag is not None
                if geo_ready:
                    with torch.no_grad():
                        preds = geo_disag(h_state, s_state)
                        Dv = float(geo_disag.disagreement(preds).squeeze(0).item())
                    ramp_denom = 10_000.0
                    d_ramp = min(1.0, (total_steps - args.kstep_min_steps) / ramp_denom)
                    Dv *= d_ramp

            # Keep embedding memory updated even before intrinsic reward turns on.
            if use_intrinsic and geo_head_ema is not None:
                with torch.no_grad():
                    e_curr = geo_head_ema(h_state, s_state).squeeze(0).detach().cpu().numpy().astype(np.float32)
                if intrinsic_mix_ready and len(embed_memory) > int(args.novelty_knn_k):
                    knn_novelty = knn_novelty_from_memory(
                        e_curr,
                        embed_memory,
                        k=args.novelty_knn_k,
                        clip_min=args.novelty_clip_min,
                        clip_max=args.novelty_clip_max,
                        use_density_count=bool(args.novelty_use_density_count),
                        density_radius=args.novelty_density_radius,
                    )
                    Fv = knn_novelty
                else:
                    knn_novelty = 0.0

                if len(embed_memory) < int(args.frontier_memory_size):
                    idx_new = len(embed_memory)
                    embed_memory.append(e_curr.copy())
                    graph_adj.append([])
                    graph_visit_count.append(1)
                    graph_node_disag.append(float(Dv))
                    if prev_graph_idx >= 0 and prev_graph_idx < idx_new:
                        graph_adj[prev_graph_idx].append(idx_new)
                        graph_adj[idx_new].append(prev_graph_idx)
                    if int(args.frontier_knn_k) > 0 and idx_new > 0:
                        prev = np.asarray(embed_memory[:-1], dtype=np.float32)
                        d_prev = np.linalg.norm(prev - e_curr.reshape(1, -1), axis=1)
                        order = np.argsort(d_prev)[: int(args.frontier_knn_k)]
                        for j in order.tolist():
                            if float(d_prev[j]) <= float(args.frontier_knn_threshold):
                                jj = int(j)
                                graph_adj[idx_new].append(jj)
                                graph_adj[jj].append(idx_new)
                    if graph_adj[idx_new]:
                        graph_adj[idx_new] = sorted(set(int(x) for x in graph_adj[idx_new]))
                    for nb in graph_adj[idx_new]:
                        graph_adj[int(nb)] = sorted(set(int(x) for x in graph_adj[int(nb)]))
                    prev_graph_idx = idx_new
                elif embed_memory:
                    mem = np.asarray(embed_memory, dtype=np.float32)
                    j = int(np.argmin(np.linalg.norm(mem - e_curr.reshape(1, -1), axis=1)))
                    graph_visit_count[j] += 1
                    graph_node_disag[j] = 0.9 * graph_node_disag[j] + 0.1 * float(Dv)
                    if prev_graph_idx >= 0 and prev_graph_idx != j:
                        graph_adj[prev_graph_idx].append(j)
                        graph_adj[j].append(prev_graph_idx)
                        graph_adj[prev_graph_idx] = sorted(set(int(x) for x in graph_adj[prev_graph_idx]))
                        graph_adj[j] = sorted(set(int(x) for x in graph_adj[j]))
                    prev_graph_idx = j

                if intrinsic_mix_ready:
                    frontier_nodes = _graph_frontier_nodes(
                        graph_adj,
                        graph_visit_count,
                        graph_node_disag,
                        reliable_min_visits=args.frontier_reliable_min_visits,
                        top_quantile=args.frontier_top_quantile,
                        disagreement_weight=args.frontier_disagreement_weight,
                    )
                    d_graph = _graph_distance_to_frontier(graph_adj, prev_graph_idx, frontier_nodes)
                    tau = max(float(args.frontier_tau), 1e-6)
                    frontier_bonus = 0.0 if not np.isfinite(d_graph) else float(np.exp(-d_graph / tau))
                else:
                    frontier_bonus = 0.0

            next_obs, r, term, trunc, step_info = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)

            # Optional EMA normalization before mixing
            Fv_mix, Dv_mix, Dv_kstep_mix = Fv, Dv, Dv_kstep
            if int_norm_F is not None and Fv != 0.0:
                Fv_mix = max(0.0, int_norm_F.normalize(Fv))
            if int_norm_D is not None and Dv != 0.0:
                Dv_mix = max(0.0, int_norm_D.normalize(Dv))
            if int_norm_D_kstep is not None and Dv_kstep != 0.0:
                Dv_kstep_mix = max(0.0, int_norm_D_kstep.normalize(Dv_kstep))

            # Keep disagreement term as one-step prediction-error variance.
            d_for_reward = Dv_mix
            r_int = (
                lambda_knn * Fv_mix
                + lambda_dvar * d_for_reward
                + lambda_frontier * frontier_bonus
            )

            local_noise = float(expl_amount)
            if expl_amount > 0:
                nov = float(knn_novelty)
                local_noise = float(expl_amount) * (1.0 + float(args.expl_novelty_gain) * nov)
                local_noise = max(0.0, local_noise)
                action_t = action_t + local_noise * torch.randn_like(action_t)
                action_t = torch.clamp(action_t, -1.0, 1.0)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            #r_store = float(r) + intrinsic_beta * r_int
            ep_sum_int += r_int
            ep_sum_F += Fv
            ep_sum_D += Dv
            ep_sum_D_kstep += Dv_kstep
            ep_sum_knn_novel += knn_novelty
            ep_sum_frontier += frontier_bonus

            writer.add_scalar("intrinsic/component_D_var_ens_kstep_norm", Dv_kstep_mix, total_steps)
            writer.add_scalar("intrinsic/component_knn_novelty", knn_novelty, total_steps)
            writer.add_scalar("intrinsic/component_replay_frontier", frontier_bonus, total_steps)
            writer.add_scalar("train/exploration_noise_local", local_noise, total_steps)

            # train value model with combined reward so we dont get ood in imagination
            #r_combined = r + intrinsic_beta * r_int
            r_combined = float(r)
            replay.add(
                obs=np.ascontiguousarray(obs, np.uint8),
                action=action,
                reward=r_combined,
                next_obs=np.ascontiguousarray(next_obs, np.uint8),
                done=done,
            )
            idx_added = (replay.idx - 1) % replay.capacity
            pri = float(knn_novelty) + float(args.prioritized_replay_eps)
            replay_priority[idx_added] = max(pri, args.prioritized_replay_eps) ** max(args.prioritized_replay_alpha, 0.0)
            obs = next_obs
            ep_ret += float(r)
            ep_steps += 1
            total_steps += 1
            if use_intrinsic:
                intrinsic_beta = intrinsic_beta_linear_schedule(
                    total_steps,
                    intrinsic_scale=args.intrinsic_scale,
                    kstep_min_steps=args.kstep_min_steps,
                    explore_period=args.explore_period,
                    decay_period=args.decay_period,
                )
            if (
                not geo_head_frozen
                and geo_head is not None
                and total_steps > GEO_HEAD_FREEZE_AFTER_STEPS
            ):
                for p in geo_head.parameters():
                    p.requires_grad = False
                geo_head_frozen = True
                print(f"    [geo_head] frozen (requires_grad=False) at total_steps={total_steps}")
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
                    if geo_head_frozen:
                        geo_head.eval()
                    else:
                        geo_head.train()
                if geo_head_ema is not None:
                    geo_head_ema.eval()
                if geo_disag is not None:
                    geo_disag.train()
                if geo_disag_kstep is not None:
                    geo_disag_kstep.train()

                sum_rec = sum_kld = sum_rew = sum_cont = sum_model = 0.0
                sum_actor = sum_value = sum_imag_r = 0.0
                sum_kstep = 0.0
                sum_kstep_unif = sum_geo_total = 0.0
                sum_kstep_action_reg = 0.0
                sum_kstep_kmax = 0.0
                sum_disag_loss = 0.0
                sum_disag_kstep_loss = 0.0
                sum_disag_reward = 0.0
                sum_disag_kstep_pairs = 0.0
                sum_disag_kstep_kmean = 0.0
                sum_geo_ema_absdiff = 0.0
                kstep_info_accum: dict[str, float] = {}
                sum_replay_is_w = 0.0

                for _ in range(args.train_steps):
                    if bool(args.prioritized_replay_enable):
                        batch, is_w_np = sample_sequences_with_priority(
                            replay,
                            replay_priority,
                            batch_size=args.batch_size,
                            seq_len=args.seq_len + 1,
                            alpha=args.prioritized_replay_alpha,
                            beta_is=args.prioritized_replay_beta_is,
                            candidate_multiplier=args.prioritized_replay_candidate_multiplier,
                        )
                    else:
                        batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)
                        is_w_np = np.ones((args.batch_size,), dtype=np.float32)
                    obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                    act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)
                    rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)
                    done_seq = torch.tensor(batch.dones, dtype=torch.float32, device=device)
                    is_w = torch.tensor(is_w_np, dtype=torch.float32, device=device)

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
                    _, _, Dh = h_seq.shape
                    Ds = s_seq.size(-1)
                    prior_m = torch.stack([p[0] for p in priors], dim=0)
                    prior_s = torch.stack([p[1] for p in priors], dim=0)
                    post_m = torch.stack([p[0] for p in posts], dim=0)
                    post_s = torch.stack([p[1] for p in posts], dim=0)

                    prior_dist = Normal(prior_m, prior_s)
                    post_dist = Normal(post_m, post_s)

                    recon = bottle(decoder, h_seq, s_seq)
                    target = x[:, 1:T + 1]
                    rec_bt = F.mse_loss(recon, target, reduction="none").sum((2, 3, 4))
                    rec_b = rec_bt.mean(dim=1)
                    rec_loss = (rec_b * is_w).sum() / (is_w.sum() + 1e-8)
                    kld_tb = torch.max(kl_divergence(post_dist, prior_dist).sum(-1), free_nats)
                    kld_b = kld_tb.transpose(0, 1).mean(dim=1)
                    kld = (kld_b * is_w).sum() / (is_w.sum() + 1e-8)
                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_bt = F.mse_loss(rew_pred, rew_seq[:, :T], reduction="none")
                    rew_b = rew_bt.mean(dim=1)
                    rew_loss = (rew_b * is_w).sum() / (is_w.sum() + 1e-8)
                    cont_logits = bottle(cont_model, h_seq, s_seq)
                    cont_target = (1.0 - done_seq[:, :T]).clamp(0.0, 1.0)
                    cont_bt = F.binary_cross_entropy_with_logits(cont_logits, cont_target, reduction="none")
                    cont_b = cont_bt.mean(dim=1)
                    cont_loss = (cont_b * is_w).sum() / (is_w.sum() + 1e-8)

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
                        geo_head is not None
                        and geo_opt is not None
                        and bank is not None
                        and not geo_head_frozen
                    ):
                        g_seq = geo_head(h_seq.detach(), s_seq.detach())
                        #k_curriculum_kmax = float(min(args.kstep_max_k + total_steps // 50_000, 16))
                        k_curriculum_kmax = float(args.kstep_max_k)
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
                        if total_steps < ramp_denom:
                            kstep_w = args.kstep_weight * float(total_steps) / float(ramp_denom)
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

                    if (
                        geo_head is not None
                        and geo_head_ema is not None
                        and geo_disag is not None
                        and geo_disag_opt is not None
                    ):
                        h_curr = h_seq[:, :-1].detach().reshape(-1, Dh)
                        s_curr = s_seq[:, :-1].detach().reshape(-1, Ds)
                        with torch.no_grad():
                            g_tgt = geo_head_ema(h_seq[:, 1:].detach(), s_seq[:, 1:].detach()).reshape(
                                -1, args.geo_dim
                            )
                            if args.disag_target_noise > 0:
                                g_tgt = F.normalize(
                                    g_tgt + args.disag_target_noise * torch.randn_like(g_tgt),
                                    dim=-1,
                                )
                        if h_curr.numel() > 0:
                            disag_loss = geo_disag.bootstrap_training_loss(
                                h_curr,
                                s_curr,
                                g_tgt,
                                keep_prob=args.disag_bootstrap_prob,
                            )
                            geo_disag_opt.zero_grad(set_to_none=True)
                            disag_loss.backward()
                            torch.nn.utils.clip_grad_norm_(geo_disag_trainable, args.grad_clip)
                            geo_disag_opt.step()
                        else:
                            disag_loss = torch.tensor(0.0, device=device)
                    else:
                        disag_loss = torch.tensor(0.0, device=device)

                    if (
                        geo_head is not None
                        and geo_head_ema is not None
                        and geo_disag_kstep is not None
                        and geo_disag_kstep_opt is not None
                    ):
                        kstep_target_max = int(args.disag_kstep_max_k) if int(args.disag_kstep_max_k) > 0 else int(args.kstep_max_k)
                        h_curr_k, s_curr_k, h_tgt_k, s_tgt_k, k_meta = _build_kstep_disag_targets_from_seq(
                            h_seq.detach(),
                            s_seq.detach(),
                            done_seq[:, :T].detach(),
                            k_min=int(args.disag_kstep_min_k),
                            k_max=kstep_target_max,
                        )
                        if h_curr_k.numel() > 0:
                            with torch.no_grad():
                                g_tgt_k = geo_head_ema(h_tgt_k, s_tgt_k)
                                if args.disag_target_noise > 0:
                                    g_tgt_k = F.normalize(
                                        g_tgt_k + args.disag_target_noise * torch.randn_like(g_tgt_k),
                                        dim=-1,
                                    )
                            disag_kstep_loss = geo_disag_kstep.bootstrap_training_loss(
                                h_curr_k,
                                s_curr_k,
                                g_tgt_k,
                                keep_prob=args.disag_bootstrap_prob,
                            )
                            geo_disag_kstep_opt.zero_grad(set_to_none=True)
                            disag_kstep_loss.backward()
                            torch.nn.utils.clip_grad_norm_(geo_disag_kstep_trainable, args.grad_clip)
                            geo_disag_kstep_opt.step()
                        else:
                            disag_kstep_loss = torch.tensor(0.0, device=device)
                        sum_disag_kstep_pairs += float(k_meta.get("n_pairs", 0.0))
                        sum_disag_kstep_kmean += float(k_meta.get("k_mean", 0.0))
                    else:
                        disag_kstep_loss = torch.tensor(0.0, device=device)

                    sum_rec += float(rec_loss.item())
                    sum_kld += float(kld.item())
                    sum_rew += float(rew_loss.item())
                    sum_cont += float(cont_loss.item())
                    sum_model += float(model_loss.item())
                    sum_kstep += float(l_kstep.item())
                    sum_kstep_unif += float(l_kstep_unif.item())
                    sum_geo_total += float(l_geo_total.item())
                    sum_kstep_kmax += k_curriculum_kmax
                    sum_disag_loss += float(disag_loss.item())
                    sum_disag_kstep_loss += float(disag_kstep_loss.item())
                    sum_replay_is_w += float(is_w.mean().item())

                    # ---- Actor-critic (imagination) ----
                    B_seq, T_seq = h_seq.shape[0], h_seq.shape[1]
                    if args.imagination_starts and 0 < args.imagination_starts < T_seq:
                        K = args.imagination_starts
                        t_idx = torch.randint(0, T_seq, (B_seq, K), device=device)
                        h_start = h_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Dh)).reshape(-1, Dh).detach()
                        s_start = s_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Ds)).reshape(-1, Ds).detach()
                    else:
                        h_start = h_seq.reshape(-1, Dh).detach()
                        s_start = s_seq.reshape(-1, Ds).detach()

                    with no_param_grads(rssm), no_param_grads(reward_model), no_param_grads(cont_model):
                        h_im_list, s_im_list, a_im_list = [h_start], [s_start], []
                        for _ in range(args.imagination_horizon):
                            a_im, _ = actor.get_action(h_im_list[-1], s_im_list[-1], deterministic=False)
                            a_im_list.append(a_im)
                            h_next = rssm.deterministic_state_fwd(h_im_list[-1], s_im_list[-1], a_im)
                            s_next = rssm.state_prior(h_next, sample=True)
                            h_im_list.append(h_next)
                            s_im_list.append(s_next)
                        h_imag = torch.stack(h_im_list, dim=1)
                        s_imag = torch.stack(s_im_list, dim=1)
                        a_imag = torch.stack(a_im_list, dim=1)

                        rewards_imag = bottle(reward_model, h_imag[:, 1:], s_imag[:, 1:])
                        cont_logits_imag = bottle(cont_model, h_imag[:, 1:], s_imag[:, 1:])
                        pcont_imag = torch.sigmoid(cont_logits_imag).clamp(0.0, 1.0)
                        discounts_imag = effective_gamma * pcont_imag

                        r_knn_imag = torch.zeros_like(rewards_imag)
                        with torch.no_grad():
                            if geo_head is not None and geo_disag is not None and lambda_dvar > 0:
                                preds_imag = geo_disag(
                                    h_imag[:, :-1].reshape(-1, Dh),
                                    s_imag[:, :-1].reshape(-1, Ds),
                                )
                                disag_imag = geo_disag.disagreement(preds_imag).reshape(h_imag.size(0), -1)
                                if float(args.disag_imag_ema_alpha) > 0.0:
                                    bm = float(disag_imag.mean().detach().item())
                                    bs = float(
                                        disag_imag.std(unbiased=False).clamp(min=1e-6).detach().item()
                                    )
                                    if disag_imag_ema_mean is None:
                                        disag_imag_ema_mean = bm
                                        disag_imag_ema_std = max(bs, 1e-6)
                                    em_m = disag_imag_ema_mean
                                    em_s = max(disag_imag_ema_std, 1e-6)
                                    #z = (disag_imag - em_m) / (em_s + 1e-6)
                                    #z = (disag_imag - em_m) / max(em_s, 0.05)
                                    z = torch.clamp(disag_imag - em_m, 0.0)
                                    r_dvar_imag = torch.clamp(z, 0.0, float(args.disag_reward_clip))
                                    a_ema = float(args.disag_imag_ema_alpha)
                                    disag_imag_ema_mean = (1.0 - a_ema) * disag_imag_ema_mean + a_ema * bm
                                    disag_imag_ema_std = (1.0 - a_ema) * disag_imag_ema_std + a_ema * max(bs, 1e-6)
                                else:
                                    r_dvar_imag = torch.clamp(
                                        disag_imag, 0.0, float(args.disag_reward_clip)
                                    )
                            else:
                                r_dvar_imag = torch.zeros_like(rewards_imag)

                            use_knn_in_imag = False
                            if (
                                geo_head_ema is not None
                                and lambda_knn > 0
                                and len(embed_memory) > int(args.novelty_knn_k)
                                and use_knn_in_imag
                            ):
                                g_imag = geo_head_ema(
                                    h_imag[:, 1:].reshape(-1, Dh),
                                    s_imag[:, 1:].reshape(-1, Ds),
                                )
                                mem = torch.tensor(
                                    np.asarray(embed_memory, dtype=np.float32),
                                    device=device,
                                    dtype=g_imag.dtype,
                                )
                                d = torch.cdist(g_imag, mem)
                                k = min(int(args.novelty_knn_k), int(mem.size(0)))
                                kth = torch.topk(d, k=k, dim=1, largest=False).values[:, -1]
                                kth = torch.clamp(
                                    kth,
                                    float(args.novelty_clip_min),
                                    float(args.novelty_clip_max),
                                )
                                r_knn_imag = kth.reshape(h_imag.size(0), -1)

                            r_int_imag = lambda_knn * r_knn_imag + lambda_dvar * r_dvar_imag

                        rewards_total = rewards_imag + intrinsic_beta * r_int_imag
                        sum_imag_r += float(rewards_total.mean().item())
                        sum_disag_reward += float(r_int_imag.mean().item())

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

                    if geo_head is not None and geo_head_ema is not None:
                        with torch.no_grad():
                            g_on = geo_head(h_seq.detach(), s_seq.detach())
                            g_em = geo_head_ema(h_seq.detach(), s_seq.detach())
                            sum_geo_ema_absdiff += float((g_on - g_em).abs().mean().item())
                            beta = 1.0 - GEO_HEAD_EMA_DECAY
                            for p_ema, p in zip(geo_head_ema.parameters(), geo_head.parameters()):
                                p_ema.data.mul_(GEO_HEAD_EMA_DECAY).add_(p.data, alpha=beta)

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
                if geo_head is not None:
                    writer.add_scalar(
                        "train/kstep_loss_enabled",
                        1.0 if not geo_head_frozen else 0.0,
                        total_steps,
                    )

                if geo_head is not None:
                    writer.add_scalar("loss/kstep_nce", sum_kstep / n_ts, total_steps)
                    writer.add_scalar("loss/kstep_unif", sum_kstep_unif / n_ts, total_steps)
                    writer.add_scalar("loss/kstep_action_reg", sum_kstep_action_reg / n_ts, total_steps)
                    writer.add_scalar("loss/geo_total", sum_geo_total / n_ts, total_steps)
                    writer.add_scalar("kstep/k_max", sum_kstep_kmax / n_ts, total_steps)
                    for k, v in kstep_info_accum.items():
                        writer.add_scalar(f"loss/{k}", v / n_ts, total_steps)
                if geo_head_ema is not None:
                    writer.add_scalar(
                        "geo_head/mean_abs_online_minus_ema",
                        sum_geo_ema_absdiff / n_ts,
                        total_steps,
                    )
                if geo_disag is not None:
                    writer.add_scalar("loss/disag_pred", sum_disag_loss / n_ts, total_steps)
                    writer.add_scalar("imag/disag_reward_mean", sum_disag_reward / n_ts, total_steps)
                    if float(args.disag_imag_ema_alpha) > 0.0 and disag_imag_ema_mean is not None:
                        writer.add_scalar("imag/disag_ema_mean", disag_imag_ema_mean, total_steps)
                        writer.add_scalar("imag/disag_ema_std", disag_imag_ema_std, total_steps)
                if geo_disag_kstep is not None:
                    writer.add_scalar("loss/disag_pred_kstep", sum_disag_kstep_loss / n_ts, total_steps)
                    writer.add_scalar("intrinsic/disag_kstep_pairs", sum_disag_kstep_pairs / n_ts, total_steps)
                    writer.add_scalar("intrinsic/disag_kstep_kmean", sum_disag_kstep_kmean / n_ts, total_steps)
                if bool(args.prioritized_replay_enable):
                    writer.add_scalar("replay/is_weight_mean", sum_replay_is_w / n_ts, total_steps)

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

        if use_intrinsic and ep_steps > 0:
            writer.add_scalar("intrinsic/r_int_mean", ep_sum_int / ep_steps, total_steps)
            writer.add_scalar("intrinsic/F_mean", ep_sum_F / ep_steps, total_steps)
            writer.add_scalar("intrinsic/D_mean", ep_sum_D / ep_steps, total_steps)
            writer.add_scalar("intrinsic/D_kstep_mean", ep_sum_D_kstep / ep_steps, total_steps)
            writer.add_scalar("intrinsic/knn_novelty_mean", ep_sum_knn_novel / ep_steps, total_steps)
            writer.add_scalar("intrinsic/replay_frontier_mean", ep_sum_frontier / ep_steps, total_steps)
            writer.add_scalar("intrinsic/r_store_bonus", intrinsic_beta * ep_sum_int / ep_steps, total_steps)
            writer.add_scalar("intrinsic/beta", intrinsic_beta, total_steps)
            writer.add_scalar(
                "intrinsic/disag_reward_source_is_kstep",
                0.0,
                total_steps,
            )
            writer.add_scalar(
                "intrinsic/D_gated",
                1.0 if (geo_head is None or total_steps >= args.kstep_min_steps) else 0.0,
                total_steps,
            )

        int_str = ""
        if use_intrinsic and ep_steps > 0:
            int_str = (
                f"  r_int={ep_sum_int/ep_steps:.4f}(F={ep_sum_F/ep_steps:.3f} "
                f"D={ep_sum_D/ep_steps:.3f} Dk={ep_sum_D_kstep/ep_steps:.3f} "
                f"N={ep_sum_knn_novel/ep_steps:.3f} Fr={ep_sum_frontier/ep_steps:.3f} "
                f"Dsrc=one_step)"
            )
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
            snap = {"env_step": float(total_steps), **{k: float(v) for k, v in ev.items()}}
            eval_snapshots.append(snap)
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
                if geo_head_ema is not None:
                    ckpt_d["geo_head_ema"] = geo_head_ema.state_dict()
                ckpt_d["bank"] = bank.detach().cpu()
                ckpt_d["bank_ptr"] = bank_ptr
            if geo_disag is not None:
                ckpt_d["geo_disag"] = geo_disag.state_dict()
            if geo_disag_kstep is not None:
                ckpt_d["geo_disag_kstep"] = geo_disag_kstep.state_dict()
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
        if geo_head_ema is not None:
            final_ckpt["geo_head_ema"] = geo_head_ema.state_dict()
        final_ckpt["bank"] = bank.detach().cpu()
        final_ckpt["bank_ptr"] = bank_ptr
    if geo_disag is not None:
        final_ckpt["geo_disag"] = geo_disag.state_dict()
    if geo_disag_kstep is not None:
        final_ckpt["geo_disag_kstep"] = geo_disag_kstep.state_dict()
    torch.save(final_ckpt, os.path.join(out_dir, "world_model_final.pt"))

    # ---- Run-level summary (AUC / max / first crossing on periodic eval curve) ----
    if eval_snapshots:
        xs = [s["env_step"] for s in eval_snapshots]
        uc = [s["unique_cells_mean"] for s in eval_snapshots]
        br = [s["bridge_crossings_mean"] for s in eval_snapshots]
        sr = [s["success_rate"] for s in eval_snapshots]
        ret_m = [s["return_mean"] for s in eval_snapshots]

        def _log_summary(prefix: str, xss: list[float], y_uc: list[float], y_br: list[float], y_sr: list[float]):
            gstep = int(xss[-1]) if xss else 0
            writer.add_scalar(f"{prefix}/auc_unique_cells_mean", _trapezoid_auc(xss, y_uc), gstep)
            writer.add_scalar(f"{prefix}/auc_bridge_crossings_mean", _trapezoid_auc(xss, y_br), gstep)
            writer.add_scalar(f"{prefix}/auc_success_rate", _trapezoid_auc(xss, y_sr), gstep)
            writer.add_scalar(f"{prefix}/max_unique_cells_mean", max(y_uc), gstep)
            writer.add_scalar(f"{prefix}/max_bridge_crossings_mean", max(y_br), gstep)
            writer.add_scalar(f"{prefix}/max_success_rate", max(y_sr), gstep)
            uc1 = _first_step_at_least(xss, y_uc, args.summary_unique_cells_thr)
            br1 = _first_step_at_least(xss, y_br, args.summary_bridge_thr)
            sr1 = _first_step_at_least(xss, y_sr, args.summary_success_thr)
            if uc1 is not None:
                writer.add_scalar(f"{prefix}/first_step_unique_cells_ge_thr", uc1, gstep)
            if br1 is not None:
                writer.add_scalar(f"{prefix}/first_step_bridge_ge_thr", br1, gstep)
            if sr1 is not None:
                writer.add_scalar(f"{prefix}/first_step_success_ge_thr", sr1, gstep)

        _log_summary("summary", xs, uc, br, sr)

        k0 = int(args.kstep_min_steps)
        if k0 > 0:
            filt = [s for s in eval_snapshots if s["env_step"] >= k0]
            if len(filt) >= 2:
                xf = [s["env_step"] for s in filt]
                _log_summary(
                    "summary_after_kstep",
                    xf,
                    [s["unique_cells_mean"] for s in filt],
                    [s["bridge_crossings_mean"] for s in filt],
                    [s["success_rate"] for s in filt],
                )

        writer.add_scalar("summary/auc_return_mean", _trapezoid_auc(xs, ret_m), int(xs[-1]))
        writer.add_scalar("summary/max_return_mean", max(ret_m), int(xs[-1]))

        print(
            "\n  [summary eval] "
            f"AUC cells={_trapezoid_auc(xs, uc):.0f}  bridges={_trapezoid_auc(xs, br):.0f}  "
            f"AUC success={_trapezoid_auc(xs, sr):.2f}  "
            f"max cells={max(uc):.1f}  max bridges={max(br):.1f}  max success={max(sr):.2f}"
        )
        uc_x = _first_step_at_least(xs, uc, args.summary_unique_cells_thr)
        br_x = _first_step_at_least(xs, br, args.summary_bridge_thr)
        if uc_x is not None or br_x is not None:
            print(
                f"  [summary first-cross @ thr] cells>={args.summary_unique_cells_thr}: {uc_x}  "
                f"bridges>={args.summary_bridge_thr}: {br_x}"
            )

    env.close()
    writer.close()
    print(f"\nDone. Mode={args.geo_mode}  total_steps={total_steps}")


if __name__ == "__main__":
    main(build_parser().parse_args())
