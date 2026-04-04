#!/usr/bin/env python3
"""
Latent Metric Correction v2 — redesigned to avoid the death spiral.

Key changes from v1:
  1. Projection head (SimCLR-style) — encoder never collapses
  2. L2-based triplet ranking loss — preserves distance structure
  3. Frozen graph — no iterative rebuild death spiral
  4. Frozen encoder bottom layers — prevents feature destruction
  5. Three distance-target modes:
       --mode graph       : graph BFS shortest paths (oracle-free)
       --mode oracle      : true geodesic distances (upper bound)
       --mode pos_boot    : position-predicted geodesic (practical)

Usage:
  python test_lmc_v2.py --wm_path world_model.pt --mode oracle
  python test_lmc_v2.py --wm_path world_model.pt --mode graph
  python test_lmc_v2.py --wm_path world_model.pt --mode pos_boot
"""

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass

import cv2
import gymnasium as gym
import gymnasium_robotics  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maze_env import GeodesicComputer
from maze_geometry_test import (
    TrainCfg,
    collect_data,
    compute_sanity_metrics,
    _positions_to_cell_indices,
)
from utils import get_device, set_seed, preprocess_img
from models import RSSM, ContinueModel, ConvDecoder, ConvEncoder, RewardModel

from pointmaze_gr_geometry_test_topo import (
    _build_mixed_replay_graph,
    _mixed_graph_undirected_edges,
    _all_pairs_shortest_paths_bfs,
    _connected_components_from_adj,
    _spearman_rho_numpy,
    _find_bridges,
    _components_without_bridges,
    _adj_from_distmat,
)

from pointmaze_large_topo_v2 import (
    PointMazeLargeDiverseGRWrapper,
    make_pointmaze_large_gr_geodesic,
)


# =====================================================================
# Config
# =====================================================================

@dataclass
class LMCv2Cfg:
    seed: int = 0
    wm_path: str = "world_model.pt"
    output_dir: str = "lmc_v2_results"
    quick: bool = False

    # distance target mode: "graph" | "oracle" | "pos_boot"
    mode: str = "oracle"

    # data collection
    collect_episodes: int = 60

    # graph building (for mode="graph" and initial kNN)
    graph_max: int = 1800
    knn_k: int = 10

    # triplet sampling
    close_max_hops: int = 3       # graph-close: d_G ≤ this
    far_min_hops: int = 8         # graph-far: d_G ≥ this
    # for oracle/pos_boot modes: geodesic thresholds
    close_max_geodesic: float = 2.5
    far_min_geodesic: float = 6.0

    # training
    total_steps: int = 3000
    batch_triplets: int = 256
    margin: float = 1.0
    proj_dim: int = 128
    encoder_lr: float = 1e-5
    proj_lr: float = 3e-4
    weight_decay: float = 1e-5
    recon_lambda: float = 5.0     # reconstruction weight (high to prevent collapse)
    rank_lambda: float = 1.0      # ranking loss weight
    recon_batch: int = 32
    grad_clip: float = 5.0

    # encoder freezing: freeze all layers whose name doesn't match these patterns
    # empty list = train all layers (with low LR)
    encoder_trainable_patterns: tuple = ("fc", "linear", "ln")  # typical last-layer names
    freeze_encoder_frac: float = 0.7  # fallback: freeze first 70% of params by count

    # evaluation
    eval_interval: int = 200
    eval_n_pairs: int = 2000
    eval_n_triplets: int = 5000

    # position bootstrap (mode="pos_boot")
    pos_probe_hidden: int = 256
    pos_probe_epochs: int = 500
    pos_probe_lr: float = 3e-4
    pos_probe_batch: int = 256

    # room discovery (evaluation only)
    room_knn_k: int = 8
    room_n_bridges_frac: float = 0.15


# =====================================================================
# Projection head
# =====================================================================

class ProjectionHead(nn.Module):
    """Maps encoder embeddings to a lower-dim space for metric learning.

    Key: outputs are RAW vectors (no L2 normalization).
    This preserves distance structure — unlike cosine-based approaches.
    """

    def __init__(self, embed_dim: int, proj_dim: int = 128):
        super().__init__()
        mid = max(proj_dim, 256)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, mid),
            nn.BatchNorm1d(mid),
            nn.ELU(),
            nn.Linear(mid, mid),
            nn.BatchNorm1d(mid),
            nn.ELU(),
            nn.Linear(mid, proj_dim),
        )
        # small init so projections start near zero — stable training
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.3)
                nn.init.zeros_(m.bias)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


# =====================================================================
# Position probe (for pos_boot mode)
# =====================================================================

class PositionProbe(nn.Module):
    """MLP: encoder embedding → (x, y) position prediction."""

    def __init__(self, embed_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


def train_position_probe(
    encoder_emb: np.ndarray,
    positions: np.ndarray,
    cfg: LMCv2Cfg,
    device: torch.device,
) -> PositionProbe:
    """Train MLP to predict (x,y) from frozen encoder embeddings."""
    N = len(encoder_emb)
    embed_dim = encoder_emb.shape[1]

    probe = PositionProbe(embed_dim, cfg.pos_probe_hidden).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.pos_probe_lr)

    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    p_t = torch.tensor(positions, dtype=torch.float32, device=device)

    rng = np.random.default_rng(cfg.seed + 777)
    n_val = max(1, int(0.15 * N))
    perm = rng.permutation(N)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    best_val = float("inf")
    best_state = None

    for epoch in range(cfg.pos_probe_epochs):
        probe.train()
        idx = rng.choice(train_idx, size=min(cfg.pos_probe_batch, len(train_idx)),
                         replace=False)
        pred = probe(e_t[idx])
        loss = F.mse_loss(pred, p_t[idx])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, cfg.pos_probe_epochs // 5) == 0:
            probe.eval()
            with torch.no_grad():
                val_pred = probe(e_t[val_idx])
                val_loss = F.mse_loss(val_pred, p_t[val_idx]).item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu().clone() for k, v in probe.state_dict().items()}
            print(f"    pos_probe epoch {epoch+1}/{cfg.pos_probe_epochs}  "
                  f"train={loss.item():.4f}  val={val_loss:.4f}")

    if best_state is not None:
        probe.load_state_dict(best_state)
    probe.eval()

    # Report R²
    with torch.no_grad():
        all_pred = probe(e_t).cpu().numpy()
    residuals = positions - all_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((positions - positions.mean(axis=0)) ** 2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    print(f"    Position probe: R² = {r2:.4f}  best_val_mse = {best_val:.4f}")

    return probe


# =====================================================================
# Distance target computation
# =====================================================================

def compute_pairwise_distances_oracle(
    pos: np.ndarray,
    geodesic: GeodesicComputer,
) -> np.ndarray:
    """Compute oracle geodesic distance for all (cell_i, cell_j) pairs.

    Returns: float32 matrix [n_free, n_free] of geodesic distances.
    """
    return geodesic.dist_matrix.astype(np.float32)


def compute_pairwise_distances_pos_boot(
    encoder_emb: np.ndarray,
    pos_probe: PositionProbe,
    geodesic: GeodesicComputer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict positions, then look up geodesic distances via cell mapping.

    Returns: (predicted_positions [N,2], cell_indices [N])
    """
    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred_pos = pos_probe(e_t).cpu().numpy().astype(np.float32)
    cell_idx = _positions_to_cell_indices(geodesic, pred_pos)
    return pred_pos, cell_idx


# =====================================================================
# Triplet sampling
# =====================================================================

def sample_triplets_graph(
    d_graph: np.ndarray,
    idx_global: np.ndarray,
    rng: np.random.Generator,
    close_max: int = 3,
    far_min: int = 8,
    n_triplets: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample (anchor, close, far) triplets from graph BFS distances.

    Returns: (anchor_global, close_global, far_global,
              d_close_arr, d_far_arr)  — global indices + distance values.
    """
    m = d_graph.shape[0]
    anchors, closes, fars = [], [], []
    d_close_list, d_far_list = [], []

    attempts = 0
    max_attempts = n_triplets * 10

    while len(anchors) < n_triplets and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, m))

        row = d_graph[i]
        close_cands = np.where((row > 0) & (row <= close_max))[0]
        far_cands = np.where(row >= far_min)[0]

        if len(close_cands) == 0 or len(far_cands) == 0:
            continue

        j = int(close_cands[rng.integers(len(close_cands))])
        k = int(far_cands[rng.integers(len(far_cands))])

        anchors.append(int(idx_global[i]))
        closes.append(int(idx_global[j]))
        fars.append(int(idx_global[k]))
        d_close_list.append(float(row[j]))
        d_far_list.append(float(row[k]))

    if not anchors:
        return (np.zeros(0, dtype=np.int64),) * 3 + (np.zeros(0, dtype=np.float32),) * 2

    return (
        np.array(anchors, dtype=np.int64),
        np.array(closes, dtype=np.int64),
        np.array(fars, dtype=np.int64),
        np.array(d_close_list, dtype=np.float32),
        np.array(d_far_list, dtype=np.float32),
    )


def sample_triplets_geodesic(
    cell_indices: np.ndarray,
    dist_matrix: np.ndarray,
    rng: np.random.Generator,
    close_max: float = 2.5,
    far_min: float = 6.0,
    n_triplets: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample triplets from geodesic distances (oracle or position-bootstrapped).

    cell_indices: [N] — cell index for each replay state.
    dist_matrix: [n_free, n_free] — pairwise geodesic distances between cells.

    Returns: (anchor_idx, close_idx, far_idx, d_close, d_far)
        All indices are into the replay buffer (not cell indices).
    """
    N = len(cell_indices)
    anchors, closes, fars = [], [], []
    d_close_list, d_far_list = [], []

    # Precompute: for each cell, which replay indices belong to it
    n_free = dist_matrix.shape[0]
    cell_to_replay: dict[int, list[int]] = {}
    for i, c in enumerate(cell_indices.tolist()):
        cell_to_replay.setdefault(int(c), []).append(i)

    # For each cell, find close and far cells
    cell_close: dict[int, list[int]] = {}
    cell_far: dict[int, list[int]] = {}
    for c in range(n_free):
        row = dist_matrix[c]
        finite = np.isfinite(row) & (row > 0)
        close_cells = np.where(finite & (row <= close_max))[0]
        far_cells = np.where(finite & (row >= far_min))[0]
        cell_close[c] = close_cells.tolist()
        cell_far[c] = far_cells.tolist()

    attempts = 0
    max_attempts = n_triplets * 10
    while len(anchors) < n_triplets and attempts < max_attempts:
        attempts += 1

        # Pick random anchor state
        a_idx = int(rng.integers(0, N))
        a_cell = int(cell_indices[a_idx])

        cc = cell_close.get(a_cell, [])
        fc = cell_far.get(a_cell, [])
        if not cc or not fc:
            continue

        # Pick a close cell and a far cell
        j_cell = int(cc[rng.integers(len(cc))])
        k_cell = int(fc[rng.integers(len(fc))])

        # Pick a replay state from each cell
        j_pool = cell_to_replay.get(j_cell, [])
        k_pool = cell_to_replay.get(k_cell, [])
        if not j_pool or not k_pool:
            continue

        j_idx = int(j_pool[rng.integers(len(j_pool))])
        k_idx = int(k_pool[rng.integers(len(k_pool))])

        anchors.append(a_idx)
        closes.append(j_idx)
        fars.append(k_idx)
        d_close_list.append(float(dist_matrix[a_cell, j_cell]))
        d_far_list.append(float(dist_matrix[a_cell, k_cell]))

    if not anchors:
        return (np.zeros(0, dtype=np.int64),) * 3 + (np.zeros(0, dtype=np.float32),) * 2

    return (
        np.array(anchors, dtype=np.int64),
        np.array(closes, dtype=np.int64),
        np.array(fars, dtype=np.int64),
        np.array(d_close_list, dtype=np.float32),
        np.array(d_far_list, dtype=np.float32),
    )


# =====================================================================
# Ranking loss (replaces InfoNCE)
# =====================================================================

def triplet_ranking_loss(
    encoder: ConvEncoder,
    proj_head: ProjectionHead,
    obs_tensor: torch.Tensor,
    anchor_idx: torch.Tensor,
    close_idx: torch.Tensor,
    far_idx: torch.Tensor,
    d_close: torch.Tensor,
    d_far: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """L2-based triplet ranking loss with distance-scaled margin.

    Encourages: d_proj(anchor, close) + margin < d_proj(anchor, far)
    Margin scales with log(d_far / d_close) — pairs with larger
    geodesic ratio get stricter enforcement.
    """
    # Gather unique indices to avoid redundant encoder forward passes
    all_idx = torch.cat([anchor_idx, close_idx, far_idx])
    unique_idx, inverse = torch.unique(all_idx, return_inverse=True)

    # Forward through encoder + projection head
    emb = encoder(obs_tensor[unique_idx])
    proj = proj_head(emb)

    n = len(anchor_idx)
    a_proj = proj[inverse[:n]]
    c_proj = proj[inverse[n:2*n]]
    f_proj = proj[inverse[2*n:]]

    # L2 distances in projection space
    d_close_proj = torch.norm(a_proj - c_proj, dim=-1)
    d_far_proj = torch.norm(a_proj - f_proj, dim=-1)

    # Scaled margin: stricter for pairs with large geodesic ratio
    ratio = d_far / torch.clamp(d_close, min=0.5)
    scaled_margin = margin * torch.log1p(ratio)

    # Triplet loss: want d_close_proj + margin < d_far_proj
    loss = F.relu(d_close_proj - d_far_proj + scaled_margin)

    return loss.mean()


# =====================================================================
# Reconstruction loss (through original decoder)
# =====================================================================

def reconstruction_and_variance_loss(
    encoder: ConvEncoder,
    obs_tensor: torch.Tensor,
    batch_idx: np.ndarray,
    device: torch.device,
    recon_head: nn.Module | None = None,
    variance_weight: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combined reconstruction + embedding variance regularization.

    Returns (L_recon, L_var) separately so they can be weighted independently.

    L_var penalizes low variance in encoder outputs — the most direct
    defence against the representation collapse observed in v1.
    When embeddings collapse, var → 0 and L_var → large.
    """
    obs_batch = obs_tensor[batch_idx]
    e = encoder(obs_batch)

    # Reconstruction through direct MLP head
    if recon_head is not None:
        recon = recon_head(e)
        l_recon = F.mse_loss(recon, obs_batch)
    else:
        l_recon = torch.tensor(0.0, device=device)

    # Variance regularization: -log(var + eps) per dimension, averaged
    # This diverges as var → 0, creating a strong anti-collapse gradient.
    # When var is healthy (~0.5+), it contributes near-zero loss.
    per_dim_var = torch.var(e, dim=0)  # [embed_dim]
    l_var = -torch.log(per_dim_var + 1e-4).mean()

    return l_recon, l_var


class DirectReconHead(nn.Module):
    """Lightweight MLP: embed_dim -> pixels."""

    def __init__(self, embed_dim: int, obs_channels: int = 3, img_size: int = 64):
        super().__init__()
        out_dim = obs_channels * img_size * img_size
        self.obs_shape = (obs_channels, img_size, img_size)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.ELU(),
            nn.Linear(512, out_dim),
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e).view(e.shape[0], *self.obs_shape)


# =====================================================================
# Encoder embedding computation
# =====================================================================

@torch.no_grad()
def recompute_embeddings(
    encoder: ConvEncoder,
    obs_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Re-encode all observations with current encoder weights."""
    encoder.eval()
    N = obs_tensor.shape[0]
    all_emb = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        emb = encoder(obs_tensor[start:end])
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


# =====================================================================
# Evaluation
# =====================================================================

def evaluate_encoder_geometry(
    encoder_emb: np.ndarray,
    pos: np.ndarray,
    geodesic: GeodesicComputer,
    n_pairs: int = 2000,
    rng: np.random.Generator = None,
) -> dict:
    """Spearman correlation between encoder L2 distances and geodesic distances."""
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(encoder_emb)
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    ii = rng.choice(N, size=n_pairs)
    jj = rng.choice(N, size=n_pairs)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]

    d_enc = np.linalg.norm(encoder_emb[ii] - encoder_emb[jj], axis=-1)

    cell_i = _positions_to_cell_indices(geodesic, pos[ii])
    cell_j = _positions_to_cell_indices(geodesic, pos[jj])
    d_geo = np.array([
        float(geodesic.dist_matrix[int(ci), int(cj)])
        for ci, cj in zip(cell_i, cell_j)
    ], dtype=np.float32)

    finite = np.isfinite(d_geo) & (d_geo > 0)
    if finite.sum() < 10:
        return {"spearman_enc_vs_geodesic": 0.0, "n_pairs": 0,
                "mean_enc_dist": 0.0, "std_enc_dist": 0.0}

    d_enc_f = d_enc[finite].astype(np.float64)
    d_geo_f = d_geo[finite].astype(np.float64)

    rho = _spearman_rho_numpy(d_enc_f, d_geo_f)
    return {
        "spearman_enc_vs_geodesic": float(rho),
        "n_pairs": int(finite.sum()),
        "mean_enc_dist": float(np.mean(d_enc_f)),
        "std_enc_dist": float(np.std(d_enc_f)),
        "mean_geo_dist": float(np.mean(d_geo_f)),
    }


def evaluate_projection_geometry(
    encoder: ConvEncoder,
    proj_head: ProjectionHead,
    obs_tensor: torch.Tensor,
    pos: np.ndarray,
    geodesic: GeodesicComputer,
    device: torch.device,
    n_pairs: int = 2000,
    rng: np.random.Generator = None,
) -> dict:
    """Spearman correlation for projection head outputs vs geodesic."""
    if rng is None:
        rng = np.random.default_rng(0)

    encoder.eval()
    proj_head.eval()
    N = obs_tensor.shape[0]

    # Compute all projections
    all_proj = []
    with torch.no_grad():
        for start in range(0, N, 256):
            end = min(start + 256, N)
            e = encoder(obs_tensor[start:end])
            p = proj_head(e)
            all_proj.append(p.cpu().numpy())
    proj_emb = np.concatenate(all_proj, axis=0).astype(np.float32)

    n_pairs = min(n_pairs, N * (N - 1) // 2)
    ii = rng.choice(N, size=n_pairs)
    jj = rng.choice(N, size=n_pairs)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]

    d_proj = np.linalg.norm(proj_emb[ii] - proj_emb[jj], axis=-1)
    cell_i = _positions_to_cell_indices(geodesic, pos[ii])
    cell_j = _positions_to_cell_indices(geodesic, pos[jj])
    d_geo = np.array([
        float(geodesic.dist_matrix[int(ci), int(cj)])
        for ci, cj in zip(cell_i, cell_j)
    ], dtype=np.float32)

    finite = np.isfinite(d_geo) & (d_geo > 0)
    if finite.sum() < 10:
        return {"spearman_proj_vs_geodesic": 0.0}

    rho = _spearman_rho_numpy(
        d_proj[finite].astype(np.float64),
        d_geo[finite].astype(np.float64),
    )
    return {
        "spearman_proj_vs_geodesic": float(rho),
        "mean_proj_dist": float(np.mean(d_proj[finite])),
        "std_proj_dist": float(np.std(d_proj[finite])),
    }


def evaluate_room_discovery(
    encoder_emb: np.ndarray,
    pos: np.ndarray,
    geodesic: GeodesicComputer,
    knn_k: int = 8,
) -> dict:
    """Discover rooms from encoder kNN graph, compare to oracle rooms."""
    cell_idx = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free

    # Cell-level mean embeddings
    D = encoder_emb.shape[1]
    cell_feats = np.zeros((n_free, D), dtype=np.float32)
    counts = np.zeros(n_free, dtype=np.int64)
    for feat, c in zip(encoder_emb, cell_idx):
        cell_feats[int(c)] += feat
        counts[int(c)] += 1
    valid = counts > 0
    if valid.sum() < 4:
        return {"n_rooms_latent": 0, "n_rooms_oracle": 0, "pair_agreement": 0.0}

    idx_cells = np.where(valid)[0]
    Fv = cell_feats[idx_cells] / counts[idx_cells, None]
    n_cells = len(idx_cells)

    k = min(knn_k, n_cells - 1)
    dm = np.linalg.norm(Fv[:, None, :] - Fv[None, :, :], axis=-1)
    np.fill_diagonal(dm, np.inf)
    knn = np.argsort(dm, axis=1)[:, :k]

    adj_lat = [[] for _ in range(n_cells)]
    edges_set = set()
    for i in range(n_cells):
        for j_local in knn[i]:
            j = int(j_local)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in edges_set:
                edges_set.add((a, b))
                adj_lat[i].append(j)
                adj_lat[j].append(i)

    bridges_lat = _find_bridges(adj_lat)
    comp_lat, n_rooms_lat = _components_without_bridges(adj_lat, bridges_lat)

    # Oracle rooms
    adj_oracle = _adj_from_distmat(geodesic.dist_matrix)
    bridges_oracle = _find_bridges(adj_oracle)
    comp_oracle, n_rooms_oracle = _components_without_bridges(adj_oracle, bridges_oracle)

    # Pairwise agreement
    comp_lat_full = np.full(n_free, -1, dtype=np.int64)
    for loc, cg in enumerate(idx_cells):
        comp_lat_full[int(cg)] = comp_lat[loc]

    agree = total = 0
    for i in range(len(idx_cells)):
        for j in range(i + 1, len(idx_cells)):
            ci, cj = idx_cells[i], idx_cells[j]
            same_oracle = comp_oracle[ci] == comp_oracle[cj]
            same_latent = comp_lat_full[ci] == comp_lat_full[cj]
            if same_oracle == same_latent:
                agree += 1
            total += 1

    return {
        "n_rooms_latent": int(n_rooms_lat),
        "n_rooms_oracle": int(n_rooms_oracle),
        "n_bridges_latent": int(len(bridges_lat)),
        "n_bridges_oracle": int(len(bridges_oracle)),
        "pair_agreement": float(agree / max(total, 1)),
        "n_cells_visited": int(n_cells),
    }


# =====================================================================
# Encoder freezing
# =====================================================================

def freeze_encoder_bottom(encoder: ConvEncoder, cfg: LMCv2Cfg):
    """Freeze bottom layers of encoder, keep top layers trainable.

    Tries to match named parameters against trainable_patterns.
    Falls back to freezing first freeze_encoder_frac of params by order.
    """
    all_params = list(encoder.named_parameters())
    n_total = len(all_params)

    # Try pattern matching first
    matched = []
    for name, p in all_params:
        name_lower = name.lower()
        if any(pat in name_lower for pat in cfg.encoder_trainable_patterns):
            matched.append(name)

    if matched:
        n_frozen = 0
        for name, p in all_params:
            if name not in matched:
                p.requires_grad_(False)
                n_frozen += 1
            else:
                p.requires_grad_(True)
        print(f"    Encoder: {n_frozen}/{n_total} param groups frozen (pattern match)")
        print(f"    Trainable: {matched}")
    else:
        # Fallback: freeze first N% by order
        n_freeze = int(cfg.freeze_encoder_frac * n_total)
        for i, (name, p) in enumerate(all_params):
            if i < n_freeze:
                p.requires_grad_(False)
            else:
                p.requires_grad_(True)
        print(f"    Encoder: {n_freeze}/{n_total} param groups frozen "
              f"(first {cfg.freeze_encoder_frac:.0%})")

    n_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in encoder.parameters())
    print(f"    Trainable params: {n_trainable:,} / {n_all:,} "
          f"({n_trainable/n_all:.1%})")


# =====================================================================
# Plotting
# =====================================================================

def plot_training_curves(
    history: list[dict], out_path: str, mode: str = "?",
):
    """Plot key metrics across training."""
    if not history:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        ax.text(
            0.5,
            0.5,
            "No evaluation checkpoints recorded.\n"
            "If training ran, every step may have skipped the optimizer\n"
            "(len(anchors) < 4 after triplet sampling).",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            wrap=True,
        )
        ax.axis("off")
        fig.suptitle(f"LMC v2 training — mode: {mode}", fontsize=14)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    steps = [h["step"] for h in history]
    rho_enc = [h.get("spearman_enc_vs_geodesic", 0.0) for h in history]
    rho_proj = [h.get("spearman_proj_vs_geodesic", 0.0) for h in history]
    n_rooms = [h.get("n_rooms_latent", 0) for h in history]
    rank_loss = [
        float(h["rank_loss"])
        if h.get("rank_loss") is not None
        else float("nan")
        for h in history
    ]
    mean_enc_dist = [h.get("mean_enc_dist", 0.0) for h in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(steps, rho_enc, "s-", color="darkorange", markersize=4)
    axes[0, 0].set_ylabel("Spearman ρ (encoder vs geodesic)")
    axes[0, 0].set_title("Encoder distance correlation")
    axes[0, 0].axhline(0, color="gray", linestyle=":", linewidth=0.5)

    axes[0, 1].plot(steps, rho_proj, "o-", color="teal", markersize=4)
    axes[0, 1].set_ylabel("Spearman ρ (projection vs geodesic)")
    axes[0, 1].set_title("Projection distance correlation")
    axes[0, 1].axhline(0, color="gray", linestyle=":", linewidth=0.5)

    axes[0, 2].plot(steps, n_rooms, "D-", color="purple", markersize=4)
    axes[0, 2].set_ylabel("Rooms discovered")
    axes[0, 2].set_title("Room count (evaluation only)")

    axes[1, 0].plot(steps, rank_loss, "^-", color="crimson", markersize=3)
    axes[1, 0].set_ylabel("Ranking loss")
    axes[1, 0].set_title("Triplet ranking loss")

    axes[1, 1].plot(steps, mean_enc_dist, "v-", color="navy", markersize=3)
    axes[1, 1].set_ylabel("Mean encoder L2 dist")
    axes[1, 1].set_title("Encoder dist (collapse check)")

    pair_agree = [h.get("pair_agreement", 0.0) for h in history]
    axes[1, 2].plot(steps, pair_agree, "s-", color="forestgreen", markersize=4)
    axes[1, 2].set_ylabel("Pair agreement with oracle")
    axes[1, 2].set_title("Room pair agreement")

    for ax in axes.flat:
        ax.set_xlabel("Training step")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"LMC v2 training — mode: {history[0].get('mode', mode)}",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =====================================================================
# Main training loop
# =====================================================================

def run_lmc_v2(
    models: dict,
    env: PointMazeLargeDiverseGRWrapper,
    cfg_train: TrainCfg,
    cfg: LMCv2Cfg,
    device: torch.device,
):
    """Main LMC v2 training: frozen graph, projection head, ranking loss."""
    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    encoder = models["encoder"]
    decoder = models["decoder"]
    rssm = models["rssm"]
    geodesic = env.geodesic

    rng = np.random.default_rng(cfg.seed)

    # ---- Collect replay data ----
    print("\n  [1] Collecting replay data ...")
    cfg_train.collect_episodes = cfg.collect_episodes
    data = collect_data(env, models, cfg_train, device)
    pos = data["pos"]
    episode_ids = data["episode_ids"]
    raw_obs = data["raw_obs"]
    N = len(pos)
    print(f"    {N} replay states, {cfg.collect_episodes} episodes")

    # Build observation tensor
    img_size = cfg_train.img_size
    obs_images = raw_obs.reshape(N, img_size, img_size, 3)
    obs_tensor = torch.tensor(obs_images, dtype=torch.float32,
                              device=device).permute(0, 3, 1, 2)
    preprocess_img(obs_tensor, depth=cfg_train.bit_depth)

    # ---- Initial encoder embeddings (frozen) ----
    encoder.eval()
    encoder_emb_frozen = recompute_embeddings(encoder, obs_tensor, device)
    embed_dim = encoder_emb_frozen.shape[1]

    # ---- Baseline evaluation ----
    eval_baseline = evaluate_encoder_geometry(
        encoder_emb_frozen, pos, geodesic,
        n_pairs=cfg.eval_n_pairs, rng=rng,
    )
    rooms_baseline = evaluate_room_discovery(
        encoder_emb_frozen, pos, geodesic, knn_k=cfg.room_knn_k,
    )
    print(f"    Baseline: ρ = {eval_baseline['spearman_enc_vs_geodesic']:.4f}  "
          f"mean_d = {eval_baseline['mean_enc_dist']:.4f}  "
          f"rooms = {rooms_baseline['n_rooms_latent']}/{rooms_baseline['n_rooms_oracle']}")

    # ---- Build distance targets depending on mode ----
    cell_indices_true = _positions_to_cell_indices(geodesic, pos)

    if cfg.mode == "oracle":
        print("\n  [2] Mode: ORACLE — using true geodesic distances")
        dist_matrix = geodesic.dist_matrix.astype(np.float32)
        cell_indices_for_sampling = cell_indices_true
        sample_fn = lambda rng, n: sample_triplets_geodesic(
            cell_indices_for_sampling, dist_matrix, rng,
            close_max=cfg.close_max_geodesic,
            far_min=cfg.far_min_geodesic,
            n_triplets=n,
        )

    elif cfg.mode == "pos_boot":
        print("\n  [2] Mode: POSITION BOOTSTRAP — training position probe ...")
        pos_probe = train_position_probe(
            encoder_emb_frozen, pos, cfg, device,
        )
        _, cell_indices_pred = compute_pairwise_distances_pos_boot(
            encoder_emb_frozen, pos_probe, geodesic, device,
        )
        dist_matrix = geodesic.dist_matrix.astype(np.float32)
        cell_indices_for_sampling = cell_indices_pred
        sample_fn = lambda rng, n: sample_triplets_geodesic(
            cell_indices_for_sampling, dist_matrix, rng,
            close_max=cfg.close_max_geodesic,
            far_min=cfg.far_min_geodesic,
            n_triplets=n,
        )
        # Report how noisy the predicted cells are
        match_frac = float(np.mean(cell_indices_pred == cell_indices_true))
        print(f"    Cell prediction accuracy: {match_frac:.2%} "
              f"(exact cell match)")

    elif cfg.mode == "graph":
        print("\n  [2] Mode: GRAPH — using BFS shortest path distances")
        print("    Building frozen replay graph ...")
        idx_global, g2l, adj_list, edges, n_nodes, n_edges = \
            build_replay_graph(
                encoder_emb_frozen, episode_ids,
                n_graph_max=cfg.graph_max, k_knn=cfg.knn_k,
            )
        print(f"    Graph: {n_nodes} nodes, {n_edges} edges (frozen)")
        print("    Computing all-pairs BFS ...")
        d_graph = _all_pairs_shortest_paths_bfs(adj_list)
        sample_fn = lambda rng, n: sample_triplets_graph(
            d_graph, idx_global, rng,
            close_max=cfg.close_max_hops,
            far_min=cfg.far_min_hops,
            n_triplets=n,
        )
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    # ---- Verify we can sample triplets ----
    test_triplets = sample_fn(rng, 100)
    if len(test_triplets[0]) < 10:
        print(f"    WARNING: only {len(test_triplets[0])} triplets sampled "
              f"from test batch. Check thresholds.")
        # Try relaxing thresholds
        if cfg.mode == "graph":
            cfg.close_max_hops = min(cfg.close_max_hops + 2, 6)
            cfg.far_min_hops = max(cfg.far_min_hops - 2, 4)
            print(f"    Relaxed: close_max={cfg.close_max_hops}, "
                  f"far_min={cfg.far_min_hops}")
        else:
            cfg.close_max_geodesic *= 1.5
            cfg.far_min_geodesic *= 0.75
            print(f"    Relaxed: close_max={cfg.close_max_geodesic:.1f}, "
                  f"far_min={cfg.far_min_geodesic:.1f}")

    # ---- Freeze encoder bottom layers ----
    print("\n  [3] Freezing encoder bottom layers ...")
    freeze_encoder_bottom(encoder, cfg)

    # ---- Freeze decoder + RSSM (not used in this version) ----
    decoder.eval()
    rssm.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    for p in rssm.parameters():
        p.requires_grad_(False)

    # ---- Initialize projection head + reconstruction head ----
    proj_head = ProjectionHead(embed_dim, cfg.proj_dim).to(device)
    recon_head = DirectReconHead(embed_dim, obs_channels=3,
                                 img_size=cfg_train.img_size).to(device)

    # ---- Optimizer with differential LR ----
    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam([
        {"params": encoder_params, "lr": cfg.encoder_lr},
        {"params": proj_head.parameters(), "lr": cfg.proj_lr},
        {"params": recon_head.parameters(), "lr": cfg.proj_lr},
    ], weight_decay=cfg.weight_decay)

    print(f"\n  [4] Training for {cfg.total_steps} steps (mode={cfg.mode}) ...")
    history = []
    running_rank_loss = 0.0
    running_recon_loss = 0.0

    for step in range(1, cfg.total_steps + 1):
        encoder.train()
        proj_head.train()
        optimizer.zero_grad(set_to_none=True)

        # ---- Sample triplets ----
        anchors, closes, fars, d_close, d_far = sample_fn(
            rng, cfg.batch_triplets,
        )
        trained_this_step = len(anchors) >= 4
        if trained_this_step:
            a_t = torch.tensor(anchors, dtype=torch.long, device=device)
            c_t = torch.tensor(closes, dtype=torch.long, device=device)
            f_t = torch.tensor(fars, dtype=torch.long, device=device)
            dc_t = torch.tensor(d_close, dtype=torch.float32, device=device)
            df_t = torch.tensor(d_far, dtype=torch.float32, device=device)

            l_rank = triplet_ranking_loss(
                encoder, proj_head, obs_tensor,
                a_t, c_t, f_t, dc_t, df_t,
                margin=cfg.margin,
            )

            recon_idx = rng.choice(N, size=min(cfg.recon_batch, N), replace=False)
            l_recon, l_var = reconstruction_and_variance_loss(
                encoder, obs_tensor, recon_idx, device,
                recon_head=recon_head,
                variance_weight=1.0,
            )

            loss = (cfg.rank_lambda * l_rank
                    + cfg.recon_lambda * l_recon
                    + 0.5 * l_var)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder_params) + list(proj_head.parameters())
                + list(recon_head.parameters()),
                cfg.grad_clip,
            )
            optimizer.step()

            running_rank_loss += float(l_rank.item())
            running_recon_loss += float(l_recon.item())

            if step % max(1, cfg.total_steps // 20) == 0:
                avg_rank = running_rank_loss / max(step, 1)
                print(f"    step {step:5d}/{cfg.total_steps}  "
                      f"L_rank={l_rank.item():.4f}  "
                      f"L_recon={l_recon.item():.4f}  "
                      f"L_var={l_var.item():.4f}  "
                      f"L_total={loss.item():.4f}  "
                      f"(avg_rank={avg_rank:.4f})")

        # ---- Periodic evaluation (always; do not skip when triplets missing) ----
        if step % cfg.eval_interval == 0 or step == cfg.total_steps:
            encoder.eval()
            proj_head.eval()

            encoder_emb_current = recompute_embeddings(
                encoder, obs_tensor, device,
            )

            eval_enc = evaluate_encoder_geometry(
                encoder_emb_current, pos, geodesic,
                n_pairs=cfg.eval_n_pairs, rng=rng,
            )
            eval_proj = evaluate_projection_geometry(
                encoder, proj_head, obs_tensor, pos, geodesic,
                device, n_pairs=cfg.eval_n_pairs, rng=rng,
            )
            rooms = evaluate_room_discovery(
                encoder_emb_current, pos, geodesic,
                knn_k=cfg.room_knn_k,
            )

            record = {
                "step": step,
                "mode": cfg.mode,
                "triplet_ok": trained_this_step,
                "rank_loss": (
                    float(l_rank.item()) if trained_this_step else None
                ),
                "recon_loss": (
                    float(l_recon.item()) if trained_this_step else None
                ),
                **eval_enc,
                **eval_proj,
                **rooms,
            }
            history.append(record)

            rho_e = eval_enc["spearman_enc_vs_geodesic"]
            rho_p = eval_proj["spearman_proj_vs_geodesic"]
            nr = rooms["n_rooms_latent"]
            pa = rooms["pair_agreement"]
            skip_note = "" if trained_this_step else "  [no triplet step]"
            print(f"    ── eval step {step}: "
                  f"ρ_enc={rho_e:.4f}  ρ_proj={rho_p:.4f}  "
                  f"rooms={nr}/{rooms['n_rooms_oracle']}  "
                  f"pair_agree={pa:.3f}  "
                  f"mean_d_enc={eval_enc['mean_enc_dist']:.4f}"
                  f"{skip_note}")

    # ---- Final evaluation ----
    print("\n  [5] Final evaluation ...")
    encoder.eval()
    proj_head.eval()
    encoder_emb_final = recompute_embeddings(encoder, obs_tensor, device)

    eval_final = evaluate_encoder_geometry(
        encoder_emb_final, pos, geodesic,
        n_pairs=cfg.eval_n_pairs, rng=rng,
    )
    eval_proj_final = evaluate_projection_geometry(
        encoder, proj_head, obs_tensor, pos, geodesic,
        device, n_pairs=cfg.eval_n_pairs, rng=rng,
    )
    rooms_final = evaluate_room_discovery(
        encoder_emb_final, pos, geodesic, knn_k=cfg.room_knn_k,
    )

    # ---- Save everything ----
    plot_training_curves(
        history, os.path.join(out_dir, "training_curves.png"), mode=cfg.mode,
    )

    summary = {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "total_steps": cfg.total_steps,
        "baseline": {**eval_baseline, **rooms_baseline},
        "final_encoder": {**eval_final},
        "final_projection": {**eval_proj_final},
        "final_rooms": {**rooms_final},
        "history": history,
        "config": {
            k: v for k, v in cfg.__dict__.items()
            if not k.startswith("_")
        },
    }

    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)

    # Save checkpoint
    ckpt = {
        "encoder": encoder.state_dict(),
        "proj_head": proj_head.state_dict(),
        "decoder": decoder.state_dict(),
        "rssm": rssm.state_dict(),
    }
    torch.save(ckpt, os.path.join(out_dir, "checkpoint.pt"))

    print(f"\n  ── SUMMARY ({cfg.mode}) ──")
    print(f"  Baseline:  ρ_enc={eval_baseline['spearman_enc_vs_geodesic']:.4f}  "
          f"rooms={rooms_baseline['n_rooms_latent']}/{rooms_baseline['n_rooms_oracle']}  "
          f"pair_agree={rooms_baseline['pair_agreement']:.3f}")
    print(f"  Final enc: ρ_enc={eval_final['spearman_enc_vs_geodesic']:.4f}  "
          f"mean_d={eval_final['mean_enc_dist']:.4f}")
    print(f"  Final proj:ρ_proj={eval_proj_final['spearman_proj_vs_geodesic']:.4f}")
    print(f"  Final rooms: {rooms_final['n_rooms_latent']}/{rooms_final['n_rooms_oracle']}  "
          f"pair_agree={rooms_final['pair_agreement']:.3f}")

    return summary


def build_replay_graph(encoder_emb, episode_ids, n_graph_max=1800, k_knn=10):
    """Build mixed graph. Reused from v1."""
    idx_global, g2l, adj_list = _build_mixed_replay_graph(
        encoder_emb, episode_ids,
        n_graph_max=n_graph_max, k_knn=k_knn,
    )
    edges = _mixed_graph_undirected_edges(adj_list)
    return idx_global, g2l, adj_list, edges, len(adj_list), len(edges)


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="LMC v2: Latent Metric Correction via projection head + ranking loss",
    )
    p.add_argument("--wm_path", type=str, default="world_model.pt")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="lmc_v2_results")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--mode", type=str, default="oracle",
                   choices=["graph", "oracle", "pos_boot"],
                   help="Distance target: graph (BFS), oracle (true geodesic), "
                        "pos_boot (position-predicted geodesic)")
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--collect_episodes", type=int, default=60)
    p.add_argument("--margin", type=float, default=1.0)
    p.add_argument("--rank_lambda", type=float, default=1.0)
    p.add_argument("--recon_lambda", type=float, default=5.0)
    p.add_argument("--encoder_lr", type=float, default=1e-5)
    p.add_argument("--proj_lr", type=float, default=3e-4)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--freeze_frac", type=float, default=0.7,
                   help="Fraction of encoder params to freeze (from bottom)")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)

    cfg_train = TrainCfg()
    cfg = LMCv2Cfg(
        seed=args.seed,
        wm_path=args.wm_path,
        output_dir=args.output_dir,
        quick=args.quick,
        mode=args.mode,
        total_steps=args.total_steps,
        collect_episodes=args.collect_episodes,
        margin=args.margin,
        rank_lambda=args.rank_lambda,
        recon_lambda=args.recon_lambda,
        encoder_lr=args.encoder_lr,
        proj_lr=args.proj_lr,
        eval_interval=args.eval_interval,
        freeze_encoder_frac=args.freeze_frac,
    )

    if args.quick:
        cfg.total_steps = 500
        cfg.collect_episodes = 20
        cfg.eval_interval = 100
        cfg.eval_n_pairs = 500
        cfg.pos_probe_epochs = 100
        cfg_train.collect_episodes = 20

    # Append mode to output dir
    cfg.output_dir = os.path.join(args.output_dir, cfg.mode)

    print(f"Device: {device}")
    print(f"Mode: {cfg.mode}")
    print(f"World model: {args.wm_path}")
    print(f"Total steps: {cfg.total_steps}")
    print(f"Output: {cfg.output_dir}")

    # ---- Load world model ----
    print("\n  Loading world model ...")
    assert os.path.exists(args.wm_path), f"Not found: {args.wm_path}"
    checkpoint = torch.load(args.wm_path, weights_only=False, map_location=device)

    maze_name = "PointMaze_Large_Diverse_GR-v3"
    env = PointMazeLargeDiverseGRWrapper(env_name=maze_name, img_size=cfg_train.img_size)
    act_dim = env.action_space.shape[0]

    encoder = ConvEncoder(cfg_train.embed_dim).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    decoder = ConvDecoder(cfg_train.deter_dim, cfg_train.stoch_dim,
                          embedding_size=cfg_train.embed_dim).to(device)
    decoder.load_state_dict(checkpoint["decoder"])
    rssm = RSSM(cfg_train.stoch_dim, cfg_train.deter_dim, act_dim,
                cfg_train.embed_dim, cfg_train.hidden_dim).to(device)
    rssm.load_state_dict(checkpoint["rssm"])
    reward_model = RewardModel(cfg_train.deter_dim, cfg_train.stoch_dim,
                               cfg_train.hidden_dim).to(device)
    reward_model.load_state_dict(checkpoint["reward_model"])
    cont_model = ContinueModel(cfg_train.deter_dim, cfg_train.stoch_dim,
                               cfg_train.hidden_dim).to(device)
    cont_model.load_state_dict(checkpoint["cont_model"])

    models = {
        "encoder": encoder,
        "decoder": decoder,
        "rssm": rssm,
        "reward_model": reward_model,
        "cont_model": cont_model,
    }

    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  Maze: {maze_name}  free_cells={env.geodesic.n_free}")

    # ---- Run ----
    summary = run_lmc_v2(models, env, cfg_train, cfg, device)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()