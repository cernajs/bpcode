#!/usr/bin/env python3

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass, field

import copy

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
from utils import get_device, set_seed, preprocess_img, bottle
from models import RSSM, ContinueModel, ConvDecoder, ConvEncoder, RewardModel

from pointmaze_gr_geometry_test_topo import (
    _build_mixed_replay_graph,
    _mixed_graph_undirected_edges,
    _all_pairs_shortest_paths_bfs,
    _connected_components_from_adj,
    _build_temporal_positive_table,
    _spearman_rho_numpy,
)

from pointmaze_large_topo_v2 import (
    PointMazeLargeDiverseGRWrapper,
    make_pointmaze_large_gr_geodesic,
)


# =====================================================================
# Direct reconstruction head (encoder collapse prevention)
# =====================================================================

class DirectReconHead(nn.Module):
    """Lightweight MLP: embed_dim → pixels, bypassing RSSM entirely.

    The RSSM path (encoder → get_init_state → decoder) provides near-zero
    gradient back into the encoder because h is zero-initialised and the
    decoder learns to ignore e.  A direct e → MLP → obs head keeps the
    encoder gradient healthy regardless of RSSM dynamics.
    """

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
# Config
# =====================================================================

@dataclass
class BarrierTrainingCfg:
    seed: int = 0
    wm_path: str = "world_model.pt"
    output_dir: str = "barrier_iterative_results"
    quick: bool = False

    # data collection
    collect_episodes: int = 60

    # graph building
    graph_max: int = 1800
    knn_k: int = 10

    # barrier score & partitioning
    barrier_threshold_frac: float = 0.15
    min_room_size: int = 5

    # contrastive correction
    geo_lr: float = 1e-4
    geo_lambda: float = 0.5
    geo_temperature: float = 0.07
    geo_train_steps: int = 200
    geo_batch: int = 256
    pos_k: int = 3
    n_hard_neg: int = 32

    # Negative mining: bootstrap with graph-distance negatives when ρ is low,
    # then switch to different-room + close-in-encoder once ρ crosses threshold.
    neg_rho_switch_threshold: float = 0.1
    neg_min_graph_hops: int = 15  # BFS hops strictly greater than this (i.e. > 15)

    # reconstruction regularization (direct MLP head)
    recon_lambda: float = 1.0
    recon_batch: int = 16

    # EMA encoder (BYOL-style collapse prevention)
    ema_decay: float = 0.99

    # warm-start: use temporal-only edges for first N iterations to let
    # the encoder build metric structure before kNN shortcuts are trusted
    warm_start_iters: int = 3

    # iteration loop
    max_iterations: int = 10
    convergence_patience: int = 2

    # evaluation
    eval_n_pairs: int = 2000


# =====================================================================
# Phase 1: Build mixed replay graph
# =====================================================================

def build_replay_graph(encoder_emb: np.ndarray, episode_ids: np.ndarray,
                       n_graph_max: int = 1800, k_knn: int = 10):
    """Build mixed graph with temporal + kNN edges. Returns local indexing."""
    idx_global, g2l, adj_list = _build_mixed_replay_graph(
        encoder_emb, episode_ids,
        n_graph_max=n_graph_max, k_knn=k_knn,
    )
    edges = _mixed_graph_undirected_edges(adj_list)
    n_nodes = len(adj_list)
    n_edges = len(edges)
    return idx_global, g2l, adj_list, edges, n_nodes, n_edges


# =====================================================================
# Phase 2: Graph shortest paths
# =====================================================================

def compute_graph_distances(adj_list: list[list[int]]) -> np.ndarray:
    """All-pairs shortest paths via BFS. Returns int32 matrix, -1 = unreachable."""
    return _all_pairs_shortest_paths_bfs(adj_list)


# =====================================================================
# Phase 3: Barrier score computation
# =====================================================================

def compute_barrier_scores(encoder_emb: np.ndarray, idx_global: np.ndarray,
                           adj_list: list[list[int]], d_graph: np.ndarray
                           ) -> np.ndarray:
    """Barrier ratio: d_graph(i,j) / max(d_enc(i,j), eps) for all node pairs."""
    emb_sub = encoder_emb[idx_global]
    d_enc = np.linalg.norm(
        emb_sub[:, None, :] - emb_sub[None, :, :], axis=-1
    ).astype(np.float64)

    d_g = d_graph.astype(np.float64)
    # Unreachable pairs (d_g < 0) must NOT be treated as zero-barrier neighbors.
    # Setting them to 0 inflates the safe-pair count, raises the percentile
    # threshold, and lets real barriers slip through. Mask them out instead.
    reachable = d_g >= 0
    barrier = np.zeros_like(d_g)
    barrier[reachable] = d_g[reachable] / np.maximum(d_enc[reachable], 1e-8)
    np.fill_diagonal(barrier, 0.0)
    return barrier.astype(np.float32)


def compute_per_edge_barrier_scores(edges: np.ndarray,
                                    barrier_matrix: np.ndarray) -> np.ndarray:
    """Extract barrier score for each undirected edge."""
    if len(edges) == 0:
        return np.zeros(0, dtype=np.float32)
    scores = np.array(
        [barrier_matrix[int(e[0]), int(e[1])] for e in edges],
        dtype=np.float32,
    )
    return scores


# =====================================================================
# Phase 4: Barrier-aware graph partitioning
# =====================================================================

def partition_into_rooms(adj_list: list[list[int]], edges: np.ndarray,
                         barrier_matrix: np.ndarray,
                         threshold_frac: float = 0.15,
                         min_room_size: int = 5) -> np.ndarray:
    """Remove top barrier edges, return connected-component room labels."""
    m = len(adj_list)
    if len(edges) == 0:
        return np.zeros(m, dtype=np.int64)

    edge_scores = compute_per_edge_barrier_scores(edges, barrier_matrix)
    threshold = float(np.percentile(edge_scores, 100.0 * (1.0 - threshold_frac)))

    filtered_adj: list[list[int]] = [[] for _ in range(m)]
    for e, s in zip(edges, edge_scores):
        if float(s) < threshold:
            u, v = int(e[0]), int(e[1])
            filtered_adj[u].append(v)
            filtered_adj[v].append(u)

    room_labels = _connected_components_from_adj(filtered_adj)

    # Merge tiny rooms into nearest large room
    unique, counts = np.unique(room_labels, return_counts=True)
    large_rooms = set(unique[counts >= min_room_size].tolist())
    if large_rooms:
        for i in range(m):
            if int(room_labels[i]) not in large_rooms:
                for nbr in adj_list[i]:
                    if int(room_labels[int(nbr)]) in large_rooms:
                        room_labels[i] = room_labels[int(nbr)]
                        break

    return room_labels


# =====================================================================
# Phase 5: Hard positive/negative mining + InfoNCE loss
# =====================================================================

def mine_pairs(encoder_emb: np.ndarray, room_labels: np.ndarray,
               episode_ids: np.ndarray, idx_global: np.ndarray,
               d_graph: np.ndarray,
               g2l: dict[int, int],
               spearman_rho: float,
               rho_switch_threshold: float = 0.1,
               min_graph_hops: int = 15,
               pos_k: int = 3, n_hard_neg: int = 32,
               rng: np.random.Generator = None):
    """Mine positive pairs (same room + temporally close) and hard negatives.

    When *spearman_rho* < *rho_switch_threshold*, negatives are bootstrapped from
    graph BFS distance: nodes farther than *min_graph_hops* hops on the replay
    graph are treated as reliable negatives regardless of room labels (breaks the
    chicken-and-egg when only one room is discovered).  Once ρ is high enough,
    negatives are *different room* + close in encoder space as before.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(encoder_emb)
    m = len(idx_global)
    use_graph_neg = float(spearman_rho) < float(rho_switch_threshold)

    # Global room labels for all replay states (-1 = not in graph)
    global_room = np.full(N, -1, dtype=np.int64)
    for loc, g in enumerate(idx_global.tolist()):
        global_room[int(g)] = int(room_labels[loc])

    # Positive pairs: same room + temporally close
    pos_table = _build_temporal_positive_table(episode_ids, pos_k=pos_k)
    anchors, positives = [], []
    for i in range(N):
        if global_room[i] < 0 or len(pos_table[i]) == 0:
            continue
        for j in pos_table[i].tolist():
            if global_room[int(j)] == global_room[i]:
                anchors.append(i)
                positives.append(int(j))

    if len(anchors) == 0:
        return None, None, None, use_graph_neg

    anchors = np.array(anchors, dtype=np.int64)
    positives = np.array(positives, dtype=np.int64)

    # Hard negatives: (bootstrap) far in graph BFS, or (mature) different room;
    # both modes then prefer states close in encoder space (hard negatives).
    graph_indices = idx_global.copy()
    emb_graph = encoder_emb[graph_indices]
    # Sample random pairs to estimate a representative pairwise median distance.
    _rng_idx = rng.choice(len(emb_graph),
                          size=(min(500, len(emb_graph)), 2), replace=True)
    median_dist = float(np.median(
        np.linalg.norm(emb_graph[_rng_idx[:, 0]] - emb_graph[_rng_idx[:, 1]],
                       axis=-1)))
    median_dist = max(median_dist, 1e-8)

    unique_anchors = np.unique(anchors)
    hard_neg_map: dict[int, np.ndarray] = {}

    sample_anchors = unique_anchors
    if len(sample_anchors) > 2000:
        sample_anchors = rng.choice(sample_anchors, size=2000, replace=False)

    for a in sample_anchors.tolist():
        a = int(a)
        a_room = int(global_room[a])
        if a_room < 0:
            continue
        la = g2l.get(a)
        if la is None:
            continue

        dists = np.linalg.norm(encoder_emb[graph_indices] - encoder_emb[a], axis=-1)

        if use_graph_neg:
            dg = d_graph[int(la)].astype(np.int64)
            # Unreachable (-1) excluded; require strictly more than min_graph_hops
            cand_mask = (dg > int(min_graph_hops)) & (dg >= 0)
        else:
            cand_mask = np.array([
                int(global_room[int(g)]) != a_room and int(global_room[int(g)]) >= 0
                for g in graph_indices
            ], dtype=bool)

        close_mask = dists < 2.0 * median_dist
        valid = cand_mask & close_mask
        valid_idx = graph_indices[valid]

        if len(valid_idx) == 0:
            pool = graph_indices[cand_mask]
            if len(pool) > 0:
                d_pool = np.linalg.norm(encoder_emb[pool] - encoder_emb[a], axis=-1)
                k = min(n_hard_neg, len(pool))
                top_k = np.argpartition(d_pool, kth=k - 1)[:k]
                valid_idx = pool[top_k]
            else:
                continue

        if len(valid_idx) > n_hard_neg:
            d_valid = np.linalg.norm(encoder_emb[valid_idx] - encoder_emb[a], axis=-1)
            top_k = np.argpartition(d_valid, kth=n_hard_neg - 1)[:n_hard_neg]
            valid_idx = valid_idx[top_k]

        hard_neg_map[a] = valid_idx

    return anchors, positives, hard_neg_map, use_graph_neg


def infonce_loss(encoder: ConvEncoder, obs_tensor: torch.Tensor,
                 anchor_idx: np.ndarray, pos_idx: np.ndarray,
                 neg_idx_map: dict[int, np.ndarray],
                 batch_size: int, temperature: float,
                 device: torch.device, rng: np.random.Generator,
                 cfg: TrainCfg,
                 ema_encoder: ConvEncoder | None = None):
    """Compute InfoNCE geo-correction loss on a mini-batch.

    When *ema_encoder* is supplied (BYOL-style), positive embeddings are
    produced by the EMA encoder under torch.no_grad(), so gradients flow
    only through anchors and hard-negatives.  This prevents the EMA target
    from collapsing along with the online encoder.
    """
    n_pairs = len(anchor_idx)
    batch_idx = rng.choice(n_pairs, size=min(batch_size, n_pairs), replace=False)

    batch_anchors = anchor_idx[batch_idx]
    batch_pos = pos_idx[batch_idx]

    # ---- anchor + negative indices → trainable encoder ----
    an_neg_indices: set[int] = set()
    for a in batch_anchors.tolist():
        an_neg_indices.add(int(a))
        if int(a) in neg_idx_map:
            for n in neg_idx_map[int(a)].tolist():
                an_neg_indices.add(int(n))
    an_neg_list = sorted(an_neg_indices)
    an_neg_to_local = {g: i for i, g in enumerate(an_neg_list)}

    an_neg_emb = encoder(obs_tensor[an_neg_list])

    # ---- positive indices → EMA encoder (no grad) or online encoder ----
    pos_global_list = sorted(set(int(p) for p in batch_pos.tolist()))
    pos_to_local = {g: i for i, g in enumerate(pos_global_list)}

    if ema_encoder is not None:
        with torch.no_grad():
            pos_emb_all = ema_encoder(obs_tensor[pos_global_list])
    else:
        pos_emb_all = encoder(obs_tensor[pos_global_list])

    total_loss = torch.tensor(0.0, device=device)
    count = 0
    for bi in range(len(batch_anchors)):
        a = int(batch_anchors[bi])
        p = int(batch_pos[bi])
        if a not in neg_idx_map or len(neg_idx_map[a]) == 0:
            continue

        a_emb = an_neg_emb[an_neg_to_local[a]]
        p_emb = pos_emb_all[pos_to_local[p]]

        neg_globals = neg_idx_map[a]
        n_embs = an_neg_emb[[an_neg_to_local[int(n)] for n in neg_globals.tolist()
                               if int(n) in an_neg_to_local]]

        if len(n_embs) == 0:
            continue

        pos_sim = F.cosine_similarity(a_emb.unsqueeze(0), p_emb.unsqueeze(0)) / temperature
        neg_sim = F.cosine_similarity(a_emb.unsqueeze(0).expand_as(n_embs), n_embs) / temperature

        logits = torch.cat([pos_sim, neg_sim], dim=0)
        labels = torch.zeros(1, dtype=torch.long, device=device)
        total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), labels)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / count


# =====================================================================
# Reconstruction regularization loss (direct MLP, no RSSM)
# =====================================================================

def direct_reconstruction_loss(encoder: ConvEncoder,
                                recon_head: DirectReconHead,
                                obs_tensor: torch.Tensor,
                                batch_size: int, device: torch.device,
                                rng: np.random.Generator) -> torch.Tensor:
    """Pixel reconstruction via encoder → DirectReconHead → obs.

    Avoids the zero-initialised RSSM h vector that causes near-zero
    gradient flow back into the encoder from the full decoder path.
    """
    N = obs_tensor.shape[0]
    idx = rng.choice(N, size=min(batch_size, N), replace=False)
    obs_batch = obs_tensor[idx]
    e = encoder(obs_batch)
    recon = recon_head(e)
    return F.mse_loss(recon, obs_batch)


# =====================================================================
# Re-embed: recompute encoder embeddings after fine-tuning
# =====================================================================

@torch.no_grad()
def recompute_embeddings(encoder: ConvEncoder, obs_tensor: torch.Tensor,
                         device: torch.device, batch_size: int = 256
                         ) -> np.ndarray:
    """Re-encode all observations with updated encoder weights."""
    encoder.eval()
    N = obs_tensor.shape[0]
    all_emb = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        emb = encoder(obs_tensor[start:end])
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


# =====================================================================
# Convergence check
# =====================================================================

def rooms_converged(rooms_new: np.ndarray, rooms_prev: np.ndarray | None,
                    tolerance: float = 0.05) -> bool:
    """Check if room partitioning has stabilized (adjusted Rand index proxy)."""
    if rooms_prev is None:
        return False
    if len(rooms_new) != len(rooms_prev):
        return False

    n = len(rooms_new)
    n_same_new = 0
    n_same_prev = 0
    n_agree = 0
    total = 0
    sample = min(n, 2000)
    rng = np.random.default_rng(42)
    pairs = rng.choice(n, size=(sample, 2), replace=True)
    for i, j in pairs:
        i, j = int(i), int(j)
        if i == j:
            continue
        total += 1
        same_n = rooms_new[i] == rooms_new[j]
        same_p = rooms_prev[i] == rooms_prev[j]
        if same_n:
            n_same_new += 1
        if same_p:
            n_same_prev += 1
        if same_n == same_p:
            n_agree += 1

    if total == 0:
        return True
    agreement = n_agree / total
    return agreement > (1.0 - tolerance)


# =====================================================================
# Evaluation metrics
# =====================================================================

def evaluate_encoder_geometry(encoder_emb: np.ndarray, pos: np.ndarray,
                              geodesic: GeodesicComputer,
                              n_pairs: int = 2000,
                              rng: np.random.Generator = None) -> dict:
    """Evaluate how well encoder distances correlate with true geodesic distances."""
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
        return {"spearman": 0.0, "n_pairs": 0}

    d_enc_f = d_enc[finite].astype(np.float64)
    d_geo_f = d_geo[finite].astype(np.float64)

    rho = _spearman_rho_numpy(d_enc_f, d_geo_f)
    return {
        "spearman_enc_vs_geodesic": float(rho),
        "n_pairs": int(finite.sum()),
        "mean_enc_dist": float(np.mean(d_enc_f)),
        "mean_geo_dist": float(np.mean(d_geo_f)),
    }


# =====================================================================
# Visualization
# =====================================================================

def plot_rooms_on_maze(geodesic: GeodesicComputer, pos: np.ndarray,
                       room_labels_global: np.ndarray, out_path: str,
                       title: str = "Discovered rooms"):
    """Scatter plot of agent positions colored by room assignment."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    grid = geodesic.grid
    h, w = len(grid), len(grid[0])
    wall_img = np.ones((h, w, 3), dtype=np.float32) * 0.9
    for r in range(h):
        for c in range(w):
            if grid[r][c] == "1":
                wall_img[r, c] = [0.2, 0.2, 0.2]
    ax.imshow(wall_img, origin="upper", extent=(-0.5, w - 0.5, h - 0.5, -0.5))

    valid = room_labels_global >= 0
    n_rooms = int(room_labels_global[valid].max()) + 1 if valid.any() else 0
    cmap = plt.cm.get_cmap("tab20", max(n_rooms, 1))

    if valid.any():
        sc = ax.scatter(pos[valid, 0], pos[valid, 1], c=room_labels_global[valid],
                        cmap=cmap, s=4, alpha=0.5, vmin=0, vmax=max(n_rooms - 1, 1))
        plt.colorbar(sc, ax=ax, label="Room ID")

    ax.set_title(f"{title} ({n_rooms} rooms)")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_barrier_distribution(edge_scores: np.ndarray, threshold: float,
                              out_path: str, iteration: int = 0):
    """Histogram of per-edge barrier scores with threshold line."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.hist(edge_scores, bins=80, color="steelblue", alpha=0.7, edgecolor="none")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=2,
               label=f"Threshold={threshold:.2f}")
    ax.set_xlabel("Barrier score (d_graph / d_enc)")
    ax.set_ylabel("Edge count")
    ax.set_title(f"Barrier score distribution (iter {iteration})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(history: list[dict], out_path: str):
    """Plot key metrics across iterations."""
    iters = [h["iteration"] for h in history]
    n_rooms = [h["n_rooms"] for h in history]
    spearman = [h.get("spearman_enc_vs_geodesic", 0.0) for h in history]
    geo_loss = [h.get("mean_geo_loss", 0.0) for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(iters, n_rooms, "o-", color="teal")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Rooms discovered")
    axes[0].set_title("Room count convergence")

    axes[1].plot(iters, spearman, "s-", color="darkorange")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Spearman ρ (enc vs geodesic)")
    axes[1].set_title("Encoder-geodesic correlation")

    axes[2].plot(iters, geo_loss, "^-", color="crimson")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("L_geo (InfoNCE)")
    axes[2].set_title("Contrastive correction loss")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =====================================================================
# Phase 6: Full iterative training loop
# =====================================================================

def run_iterative_barrier_training(
    models: dict,
    env: PointMazeLargeDiverseGRWrapper,
    cfg_train: TrainCfg,
    cfg_barrier: BarrierTrainingCfg,
    device: torch.device,
):
    """Main bootstrap loop: graph -> barriers -> rooms -> correct -> repeat."""
    out_dir = cfg_barrier.output_dir
    os.makedirs(out_dir, exist_ok=True)

    encoder = models["encoder"]
    decoder = models["decoder"]
    rssm = models["rssm"]
    geodesic = env.geodesic

    rng = np.random.default_rng(cfg_barrier.seed)

    # ---- Collect replay data ----
    print("\n  [Collect] Gathering replay data ...")
    cfg_train.collect_episodes = cfg_barrier.collect_episodes
    data = collect_data(env, models, cfg_train, device)
    pos = data["pos"]
    episode_ids = data["episode_ids"]
    raw_obs = data["raw_obs"]  # [N, H*W*C] flattened, /255
    N = len(pos)
    print(f"    {N} replay states from {cfg_barrier.collect_episodes} episodes")

    # Reconstruct image tensors for encoder forward passes: [N, C, H, W]
    img_size = cfg_train.img_size
    obs_images = raw_obs.reshape(N, img_size, img_size, 3)
    obs_tensor = torch.tensor(obs_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    preprocess_img(obs_tensor, depth=cfg_train.bit_depth)

    # Initial encoder embeddings
    encoder_emb = recompute_embeddings(encoder, obs_tensor, device)

    # Initial evaluation
    eval0 = evaluate_encoder_geometry(encoder_emb, pos, geodesic,
                                      n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
    print(f"    Initial encoder-geodesic Spearman ρ = {eval0['spearman_enc_vs_geodesic']:.4f}")
    # ρ from last eval drives negative mining (graph bootstrap vs room-based).
    current_rho = float(eval0["spearman_enc_vs_geodesic"])

    # ---- EMA encoder (BYOL-style — positives come from the EMA copy) ----
    ema_encoder = copy.deepcopy(encoder)
    for _p in ema_encoder.parameters():
        _p.requires_grad_(False)
    ema_encoder.eval()
    print(f"    EMA encoder initialised (decay={cfg_barrier.ema_decay})")

    # ---- Direct reconstruction head (no RSSM gradient path) ----
    recon_head = DirectReconHead(
        embed_dim=cfg_train.embed_dim,
        obs_channels=3,
        img_size=cfg_train.img_size,
    ).to(device)

    # Joint optimizer: encoder + recon head
    encoder_opt = torch.optim.Adam(
        list(encoder.parameters()) + list(recon_head.parameters()),
        lr=cfg_barrier.geo_lr,
    )

    history: list[dict] = []
    prev_rooms = None
    patience_counter = 0

    for iteration in range(cfg_barrier.max_iterations):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration + 1}/{cfg_barrier.max_iterations}")
        print(f"{'='*60}")

        # Phase 1: Build replay graph
        # Warm-start: use temporal edges only for the first N iterations so the
        # barrier detector has ground-truth topology even when the encoder is flat.
        # kNN edges are re-introduced once the encoder gains some metric structure.
        effective_knn_k = (0 if iteration < cfg_barrier.warm_start_iters
                           else cfg_barrier.knn_k)
        print("  [Phase 1] Building mixed replay graph "
              f"({'temporal-only warm-start' if effective_knn_k == 0 else f'kNN k={effective_knn_k}'}) ...")
        idx_global, g2l, adj_list, edges, n_nodes, n_edges = build_replay_graph(
            encoder_emb, episode_ids,
            n_graph_max=cfg_barrier.graph_max,
            k_knn=effective_knn_k,
        )
        print(f"    Graph: {n_nodes} nodes, {n_edges} edges")

        if n_nodes < 16:
            print("    Graph too small, stopping.")
            break

        # Phase 2: Compute graph shortest paths
        print("  [Phase 2] Computing all-pairs shortest paths ...")
        d_graph = compute_graph_distances(adj_list)

        # Phase 3: Compute barrier scores
        print("  [Phase 3] Computing barrier scores ...")
        barrier_matrix = compute_barrier_scores(encoder_emb, idx_global, adj_list, d_graph)
        edge_scores = compute_per_edge_barrier_scores(edges, barrier_matrix)

        barrier_thresh = float(np.percentile(edge_scores,
                                             100.0 * (1.0 - cfg_barrier.barrier_threshold_frac)))
        n_high = int(np.sum(edge_scores >= barrier_thresh))
        print(f"    Barrier threshold={barrier_thresh:.2f}, "
              f"high-barrier edges={n_high}/{len(edge_scores)}")

        plot_barrier_distribution(
            edge_scores, barrier_thresh,
            os.path.join(out_dir, f"barrier_dist_iter{iteration}.png"),
            iteration=iteration,
        )

        # Phase 4: Partition into rooms
        print("  [Phase 4] Partitioning into candidate rooms ...")
        room_labels = partition_into_rooms(
            adj_list, edges, barrier_matrix,
            threshold_frac=cfg_barrier.barrier_threshold_frac,
            min_room_size=cfg_barrier.min_room_size,
        )
        n_rooms = int(room_labels.max()) + 1
        room_sizes = [int(np.sum(room_labels == r)) for r in range(n_rooms)]
        print(f"    Discovered {n_rooms} rooms: sizes={room_sizes}")

        # Map room labels to global space for visualization
        global_room = np.full(N, -1, dtype=np.int64)
        for loc, g in enumerate(idx_global.tolist()):
            global_room[int(g)] = int(room_labels[loc])

        plot_rooms_on_maze(
            geodesic, pos, global_room,
            os.path.join(out_dir, f"rooms_iter{iteration}.png"),
            title=f"Rooms (iter {iteration})",
        )

        # Convergence check
        converged = rooms_converged(room_labels, prev_rooms)
        if converged:
            patience_counter += 1
            print(f"    Rooms converged ({patience_counter}/{cfg_barrier.convergence_patience})")
            if patience_counter >= cfg_barrier.convergence_patience:
                print("    Converged! Stopping iteration.")
                eval_final = evaluate_encoder_geometry(
                    encoder_emb, pos, geodesic,
                    n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
                history.append({
                    "iteration": iteration,
                    "n_rooms": n_rooms,
                    "room_sizes": room_sizes,
                    "n_high_barrier_edges": n_high,
                    "barrier_threshold": barrier_thresh,
                    "converged": True,
                    **eval_final,
                })
                break
        else:
            patience_counter = 0
        prev_rooms = room_labels.copy()

        # Phase 5: Mine pairs and train encoder
        print("  [Phase 5] Mining hard positives/negatives ...")
        anchors, positives, hard_neg_map, used_graph_negs = mine_pairs(
            encoder_emb, room_labels, episode_ids, idx_global,
            d_graph=d_graph,
            g2l=g2l,
            spearman_rho=current_rho,
            rho_switch_threshold=cfg_barrier.neg_rho_switch_threshold,
            min_graph_hops=cfg_barrier.neg_min_graph_hops,
            pos_k=cfg_barrier.pos_k,
            n_hard_neg=cfg_barrier.n_hard_neg,
            rng=rng,
        )
        neg_mode = (
            f"graph-dist >{cfg_barrier.neg_min_graph_hops} hops (ρ={current_rho:.4f} < "
            f"{cfg_barrier.neg_rho_switch_threshold})"
            if used_graph_negs
            else f"different-room (ρ={current_rho:.4f} ≥ {cfg_barrier.neg_rho_switch_threshold})"
        )
        print(f"    Negative mode: {neg_mode}")

        if anchors is None or len(anchors) == 0:
            print("    No valid pairs found, skipping training step.")
            eval_iter = evaluate_encoder_geometry(
                encoder_emb, pos, geodesic,
                n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
            current_rho = float(eval_iter["spearman_enc_vs_geodesic"])
            history.append({
                "iteration": iteration,
                "n_rooms": n_rooms,
                "room_sizes": room_sizes,
                "n_high_barrier_edges": n_high,
                "barrier_threshold": barrier_thresh,
                "mean_geo_loss": 0.0,
                "converged": False,
                "neg_mining_graph_bootstrap": used_graph_negs,
                **eval_iter,
            })
            continue

        n_anchors_with_neg = sum(1 for a in np.unique(anchors) if int(a) in hard_neg_map)
        print(f"    {len(anchors)} positive pairs, "
              f"{n_anchors_with_neg} anchors with hard negatives")

        print(f"  [Phase 5] Training encoder ({cfg_barrier.geo_train_steps} steps) ...")
        encoder.train()
        geo_losses = []

        for step in range(cfg_barrier.geo_train_steps):
            encoder_opt.zero_grad(set_to_none=True)

            # L_geo: InfoNCE contrastive correction; EMA encoder produces positives
            l_geo = infonce_loss(
                encoder, obs_tensor,
                anchors, positives, hard_neg_map,
                batch_size=cfg_barrier.geo_batch,
                temperature=cfg_barrier.geo_temperature,
                device=device, rng=rng, cfg=cfg_train,
                ema_encoder=ema_encoder,
            )

            # L_recon: direct encoder → MLP → obs (no RSSM, healthy gradient)
            l_recon = direct_reconstruction_loss(
                encoder, recon_head, obs_tensor,
                batch_size=cfg_barrier.recon_batch,
                device=device, rng=rng,
            )

            loss = cfg_barrier.recon_lambda * l_recon + cfg_barrier.geo_lambda * l_geo
            loss.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 10.0)
            encoder_opt.step()

            # EMA update: θ_ema ← decay·θ_ema + (1−decay)·θ
            with torch.no_grad():
                for p_ema, p_online in zip(ema_encoder.parameters(),
                                           encoder.parameters()):
                    p_ema.data.mul_(cfg_barrier.ema_decay).add_(
                        p_online.data, alpha=1.0 - cfg_barrier.ema_decay)

            geo_losses.append(float(l_geo.item()))

            if (step + 1) % max(1, cfg_barrier.geo_train_steps // 5) == 0:
                print(f"      step {step+1}/{cfg_barrier.geo_train_steps}  "
                      f"L_geo={l_geo.item():.4f}  L_recon={l_recon.item():.4f}  "
                      f"L_total={loss.item():.4f}")

        mean_geo_loss = float(np.mean(geo_losses)) if geo_losses else 0.0

        # Phase 6: Re-embed
        print("  [Phase 6] Recomputing encoder embeddings ...")
        encoder_emb = recompute_embeddings(encoder, obs_tensor, device)

        # Evaluate
        eval_iter = evaluate_encoder_geometry(
            encoder_emb, pos, geodesic,
            n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
        current_rho = float(eval_iter["spearman_enc_vs_geodesic"])
        dt = time.time() - t0
        print(f"    Spearman ρ (enc vs geodesic) = {eval_iter['spearman_enc_vs_geodesic']:.4f}  "
              f"({dt:.1f}s)")

        history.append({
            "iteration": iteration,
            "n_rooms": n_rooms,
            "room_sizes": room_sizes,
            "n_high_barrier_edges": n_high,
            "barrier_threshold": barrier_thresh,
            "mean_geo_loss": mean_geo_loss,
            "elapsed_s": dt,
            "converged": False,
            "neg_mining_graph_bootstrap": used_graph_negs,
            **eval_iter,
        })

    # ---- Final summary ----
    plot_convergence(history, os.path.join(out_dir, "convergence.png"))

    # Save final checkpoint
    final_ckpt = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "rssm": rssm.state_dict(),
        "reward_model": models["reward_model"].state_dict(),
        "cont_model": models["cont_model"].state_dict(),
    }
    torch.save(final_ckpt, os.path.join(out_dir, "world_model_corrected.pt"))

    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2, default=_json_default)

    print(f"\n  Results saved to {out_dir}/")
    if history:
        h0 = history[0]
        hf = history[-1]
        print(f"    ρ: {eval0['spearman_enc_vs_geodesic']:.4f} (initial) -> "
              f"{hf.get('spearman_enc_vs_geodesic', 0.0):.4f} (final)")
        print(f"    Rooms: {h0['n_rooms']} (iter 0) -> {hf['n_rooms']} (final)")

    return history


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    return str(obj)


# =====================================================================
# CLI entry point
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Iterative barrier-aware encoder training")
    p.add_argument("--wm_path", type=str, default="world_model.pt",
                   help="Path to world model checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="barrier_iterative_results")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--max_iter", type=int, default=10)
    p.add_argument("--collect_episodes", type=int, default=60)
    p.add_argument("--graph_max", type=int, default=1800)
    p.add_argument("--knn_k", type=int, default=10)
    p.add_argument("--barrier_frac", type=float, default=0.15)
    p.add_argument("--geo_lr", type=float, default=1e-4)
    p.add_argument("--geo_lambda", type=float, default=0.5)
    p.add_argument("--geo_steps", type=int, default=200)
    p.add_argument("--geo_batch", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.07)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    set_seed(args.seed)

    cfg_train = TrainCfg()

    cfg_barrier = BarrierTrainingCfg(
        seed=args.seed,
        wm_path=args.wm_path,
        output_dir=args.output_dir,
        quick=args.quick,
        collect_episodes=args.collect_episodes,
        graph_max=args.graph_max,
        knn_k=args.knn_k,
        barrier_threshold_frac=args.barrier_frac,
        geo_lr=args.geo_lr,
        geo_lambda=args.geo_lambda,
        geo_train_steps=args.geo_steps,
        geo_batch=args.geo_batch,
        geo_temperature=args.temperature,
        max_iterations=args.max_iter,
    )

    if args.quick:
        cfg_barrier.collect_episodes = 20
        cfg_barrier.geo_train_steps = 50
        cfg_barrier.max_iterations = 3
        cfg_barrier.graph_max = 1200
        cfg_barrier.eval_n_pairs = 500
        cfg_train.collect_episodes = 20

    print(f"Device: {device}")
    print(f"World model: {args.wm_path}")
    print(f"Quick mode: {args.quick}")
    print(f"Max iterations: {cfg_barrier.max_iterations}")

    # ---- Load world model ----
    print("\n  Loading world model checkpoint ...")
    assert os.path.exists(args.wm_path), f"Checkpoint not found: {args.wm_path}"
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

    print(f"  Loaded: encoder({sum(p.numel() for p in encoder.parameters())} params), "
          f"decoder, rssm, reward, continue")
    print(f"  Maze: {maze_name}  grid={env.grid_h}x{env.grid_w}  "
          f"free_cells={env.geodesic.n_free}")

    # ---- Run iterative training ----
    history = run_iterative_barrier_training(
        models, env, cfg_train, cfg_barrier, device)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
