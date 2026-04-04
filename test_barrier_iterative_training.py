#!/usr/bin/env python3
"""Barrier-aware replay-graph training: projector head, detour+routed edge scores, topology metrics."""

import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import comb

from maze_env import GeodesicComputer
from maze_geometry_test import (
    TrainCfg,
    collect_data,
    _positions_to_cell_indices,
    _adj_from_distmat,
    _find_bridges,
    _components_without_bridges,
)
from utils import get_device, set_seed, preprocess_img
from models import RSSM, ContinueModel, ConvDecoder, ConvEncoder, RewardModel

from pointmaze_gr_geometry_test_topo import (
    _build_mixed_replay_graph,
    _mixed_graph_undirected_edges,
    _all_pairs_shortest_paths_bfs,
    _connected_components_from_adj,
    _build_temporal_positive_table,
    _spearman_rho_numpy,
    _temporal_only_adj_local,
    _edge_betweenness_brandes_undirected,
    _kmeans_numpy,
)

from pointmaze_large_topo_v2 import PointMazeLargeDiverseGRWrapper


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
    detour_max_edges: int = 4000
    route_sample_pairs: int = 1200
    route_d_min: int = 4
    score_w_detour: float = 0.55
    score_w_route: float = 0.45
    unreachable_cap_mul: int = 2

    # projector head (train before / instead of full encoder fine-tune)
    projector_dim: int = 128
    projector_hidden: int = 256
    projector_lr: float = 3e-4
    projector_warmup_steps: int = 80
    projector_steps: int = 200
    finetune_encoder: bool = False
    finetune_encoder_steps: int = 0
    finetune_encoder_lr: float = 5e-6

    # contrastive correction (on projector features; optional encoder FT)
    geo_lambda: float = 0.5
    geo_temperature: float = 0.07
    geo_batch: int = 256
    pos_k: int = 3
    n_hard_neg: int = 32

    # reconstruction regularization (only when finetune_encoder)
    recon_lambda: float = 1.0
    recon_batch: int = 16

    # iteration loop
    max_iterations: int = 10
    convergence_patience: int = 2

    # evaluation
    eval_n_pairs: int = 2000
    topo_top_frac: float = 0.15


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
# Phase 3: Edge scores (detour + routed mismatch)
# =====================================================================

def bfs_dist_without_edge(
    adj_list: list[list[int]],
    n: int,
    src: int,
    dst: int,
    skip_u: int,
    skip_v: int,
) -> int:
    """Shortest hop distance src→dst when undirected edge (skip_u, skip_v) is forbidden."""
    dist = np.full(n, -1, dtype=np.int32)
    dist[src] = 0
    q: deque[int] = deque([src])
    while q:
        x = int(q.popleft())
        dx = int(dist[x])
        for y in adj_list[x]:
            y = int(y)
            if (x == skip_u and y == skip_v) or (x == skip_v and y == skip_u):
                continue
            if dist[y] < 0:
                dist[y] = dx + 1
                if y == dst:
                    return int(dist[y])
                q.append(y)
    return -1


def bfs_shortest_path_edges(
    adj_list: list[list[int]], n: int, src: int, dst: int
) -> list[tuple[int, int]]:
    """Undirected edges (min,max) on one BFS shortest path src→dst; empty if unreachable."""
    parent = np.full(n, -1, dtype=np.int32)
    dist = np.full(n, -1, dtype=np.int32)
    dist[src] = 0
    q: deque[int] = deque([src])
    while q:
        x = int(q.popleft())
        dx = int(dist[x])
        for y in adj_list[x]:
            y = int(y)
            if dist[y] < 0:
                dist[y] = dx + 1
                parent[y] = x
                if y == dst:
                    q.clear()
                    break
                q.append(y)
    if dist[dst] < 0:
        return []
    path_edges: list[tuple[int, int]] = []
    cur = int(dst)
    while cur != src:
        p = int(parent[cur])
        a, b = (p, cur) if p < cur else (cur, p)
        path_edges.append((a, b))
        cur = p
    return path_edges


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = float(np.mean(x))
    s = float(np.std(x))
    if s < 1e-12:
        return np.zeros_like(x, dtype=np.float64)
    return (x - m) / s


def compute_detour_edge_scores(
    edges: np.ndarray,
    adj_list: list[list[int]],
    emb_sub: np.ndarray,
    rng: np.random.Generator,
    max_eval_edges: int = 4000,
    unreachable_cap_mul: int = 2,
) -> np.ndarray:
    """score_e ≈ d_{G\\e}(u,v) / d_enc(u,v); bridges get large detour / small d_enc."""
    m = len(adj_list)
    n_e = len(edges)
    scores = np.zeros(n_e, dtype=np.float64)
    if n_e == 0:
        return scores.astype(np.float32)
    cap = int(max(2, unreachable_cap_mul * m))
    order = np.arange(n_e)
    if n_e > max_eval_edges:
        order = rng.permutation(n_e)[:max_eval_edges]
    mask = np.zeros(n_e, dtype=bool)
    for ei in order.tolist():
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        raw = bfs_dist_without_edge(adj_list, m, u, v, u, v)
        if raw < 0:
            raw = cap
        d_enc = float(np.linalg.norm(emb_sub[u] - emb_sub[v])) + 1e-8
        scores[ei] = raw / d_enc
        mask[ei] = True
    if not np.all(mask):
        fill = float(np.median(scores[mask])) if mask.any() else 1.0
        scores[~mask] = fill
    return scores.astype(np.float32)


def compute_routed_mismatch_edge_scores(
    edges: np.ndarray,
    adj_list: list[list[int]],
    d_graph: np.ndarray,
    emb_sub: np.ndarray,
    rng: np.random.Generator,
    n_samples: int = 1200,
    d_min: int = 4,
) -> np.ndarray:
    """Accumulate graph-vs-encoder mismatch along shortest paths for distant pairs."""
    m = len(adj_list)
    n_e = len(edges)
    accum = np.zeros(n_e, dtype=np.float64)
    counts = np.zeros(n_e, dtype=np.int64)
    if n_e == 0 or m < 4:
        return accum.astype(np.float32)

    edge_to_i: dict[tuple[int, int], int] = {}
    for i in range(n_e):
        a, b = int(edges[i, 0]), int(edges[i, 1])
        edge_to_i[(a, b) if a < b else (b, a)] = i

    # Candidate pairs with long graph distance
    ii, jj = np.where((d_graph >= d_min) & (d_graph >= 0))
    if len(ii) == 0:
        return accum.astype(np.float32)
    pool = np.stack([ii, jj], axis=1)
    pool = pool[pool[:, 0] < pool[:, 1]]
    if len(pool) == 0:
        return accum.astype(np.float32)
    take = min(int(n_samples), len(pool))
    sel = rng.choice(len(pool), size=take, replace=len(pool) < take)
    for t in sel.tolist():
        i, j = int(pool[t, 0]), int(pool[t, 1])
        dg = int(d_graph[i, j])
        de = float(np.linalg.norm(emb_sub[i] - emb_sub[j])) + 1e-8
        mismatch = float(dg) / de
        for a, b in bfs_shortest_path_edges(adj_list, m, i, j):
            key = (a, b) if a < b else (b, a)
            ei = edge_to_i.get(key, None)
            if ei is not None:
                accum[ei] += mismatch
                counts[ei] += 1

    out = np.zeros(n_e, dtype=np.float64)
    nz = counts > 0
    out[nz] = accum[nz] / counts[nz].astype(np.float64)
    if (~nz).any() and nz.any():
        out[~nz] = float(np.median(out[nz]))
    return out.astype(np.float32)


def compute_combined_edge_barrier_scores(
    edges: np.ndarray,
    adj_list: list[list[int]],
    d_graph: np.ndarray,
    emb_sub: np.ndarray,
    rng: np.random.Generator,
    *,
    detour_max_edges: int = 4000,
    unreachable_cap_mul: int = 2,
    route_sample_pairs: int = 1200,
    route_d_min: int = 4,
    w_detour: float = 0.55,
    w_route: float = 0.45,
) -> np.ndarray:
    d_detour = compute_detour_edge_scores(
        edges, adj_list, emb_sub, rng,
        max_eval_edges=detour_max_edges,
        unreachable_cap_mul=unreachable_cap_mul,
    )
    d_route = compute_routed_mismatch_edge_scores(
        edges, adj_list, d_graph, emb_sub, rng,
        n_samples=route_sample_pairs,
        d_min=route_d_min,
    )
    z_d = _zscore_1d(d_detour.astype(np.float64))
    z_r = _zscore_1d(d_route.astype(np.float64))
    wsum = max(w_detour + w_route, 1e-8)
    combined = (w_detour * z_d + w_route * z_r) / wsum
    return combined.astype(np.float32)


# =====================================================================
# Phase 4: Barrier-aware graph partitioning + escape hatch
# =====================================================================

def partition_into_rooms(
    adj_list: list[list[int]],
    edges: np.ndarray,
    edge_scores: np.ndarray,
    threshold_frac: float = 0.15,
    min_room_size: int = 5,
) -> np.ndarray:
    """Remove top barrier-score edges; return connected-component room labels."""
    m = len(adj_list)
    if len(edges) == 0:
        return np.zeros(m, dtype=np.int64)

    threshold = float(np.percentile(edge_scores, 100.0 * (1.0 - threshold_frac)))

    filtered_adj: list[list[int]] = [[] for _ in range(m)]
    for e, s in zip(edges, edge_scores):
        if float(s) < threshold:
            u, v = int(e[0]), int(e[1])
            filtered_adj[u].append(v)
            filtered_adj[v].append(u)

    room_labels = _connected_components_from_adj(filtered_adj)

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


def fiedler_bisection_two_way(adj_list: list[list[int]], rng: np.random.Generator) -> np.ndarray:
    """Second eigenvector of normalized Laplacian, split at median → two pseudo-rooms."""
    m = len(adj_list)
    if m < 4:
        return np.zeros(m, dtype=np.int64)
    a = np.zeros((m, m), dtype=np.float64)
    for i, nei in enumerate(adj_list):
        for j in nei:
            jj = int(j)
            if jj != i:
                a[i, jj] = 1.0
    deg = a.sum(axis=1)
    inv_sqrt = np.zeros(m, dtype=np.float64)
    nz = deg > 1e-12
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    d_inv = np.diag(inv_sqrt)
    ln = np.eye(m, dtype=np.float64) - d_inv @ a @ d_inv
    try:
        evals, evecs = np.linalg.eigh(ln)
    except np.linalg.LinAlgError:
        return np.zeros(m, dtype=np.int64)
    f = evecs[:, 1].astype(np.float64)
    med = float(np.median(f))
    return (f > med).astype(np.int64)


def split_by_high_betweenness_cut(adj_list: list[list[int]], rng: np.random.Generator) -> np.ndarray | None:
    """Remove single highest-betweenness edge and return two components if disconnected."""
    eb = _edge_betweenness_brandes_undirected(adj_list)
    if not eb:
        return None
    # pick max edge
    best_e, best_s = None, -1.0
    for (a, b), s in eb.items():
        if float(s) > best_s:
            best_s, best_e = float(s), (int(a), int(b))
    if best_e is None:
        return None
    u, v = best_e[0], best_e[1]
    m = len(adj_list)
    filt: list[list[int]] = [[] for _ in range(m)]
    for i in range(m):
        for j in adj_list[i]:
            j = int(j)
            if {i, j} == {u, v}:
                continue
            filt[i].append(j)
    comp = _connected_components_from_adj(filt)
    if int(comp.max()) + 1 >= 2:
        return comp
    return None


def apply_escape_hatch_singleton(
    room_labels: np.ndarray,
    adj_list: list[list[int]],
    idx_global: np.ndarray,
    episode_ids: np.ndarray,
    emb_sub: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, str]:
    """If pruning yields one room, force ≥2 groups from graph/geometry structure."""
    n_rooms = int(room_labels.max()) + 1
    if n_rooms >= 2:
        return room_labels, "none"

    lab = fiedler_bisection_two_way(adj_list, rng)
    if len(np.unique(lab)) >= 2:
        return lab.astype(np.int64), "fiedler_bisection"

    adj_t = _temporal_only_adj_local(idx_global, episode_ids, adj_list)
    comp_t = _connected_components_from_adj(adj_t)
    if int(comp_t.max()) + 1 >= 2:
        return comp_t.astype(np.int64), "temporal_components"

    cut = split_by_high_betweenness_cut(adj_list, rng)
    if cut is not None:
        return cut.astype(np.int64), "betweenness_cut"

    # k-means k=2 on embeddings (last resort)
    if len(emb_sub) >= 8:
        km = _kmeans_numpy(emb_sub.astype(np.float64), k=2, rng=rng, n_iter=30)
        if len(np.unique(km)) >= 2:
            return km.astype(np.int64), "kmeans_emb_k2"

    return room_labels, "failed"


# =====================================================================
# Projector head (trained before optional encoder fine-tune)
# =====================================================================

class GeoProjector(nn.Module):
    """Small MLP on frozen encoder features; outputs L2-normalized z for contrastive loss."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1, eps=1e-8)


# =====================================================================
# Phase 5: Hard positive/negative mining + InfoNCE loss
# =====================================================================

def mine_pairs(encoder_emb: np.ndarray, room_labels: np.ndarray,
               episode_ids: np.ndarray, idx_global: np.ndarray,
               pos_k: int = 3, n_hard_neg: int = 32,
               rng: np.random.Generator = None):
    """Mine positive pairs (same room + temporally close) and hard negatives
    (different room + close in encoder space)."""
    if rng is None:
        rng = np.random.default_rng(0)

    N = len(encoder_emb)
    m = len(idx_global)

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
        return None, None, None

    anchors = np.array(anchors, dtype=np.int64)
    positives = np.array(positives, dtype=np.int64)

    # Hard negatives: different room but close in encoder space
    graph_indices = idx_global.copy()
    emb_graph = encoder_emb[graph_indices]
    median_dist = float(np.median(np.linalg.norm(
        emb_graph[:1] - emb_graph[1:], axis=-1)))

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

        dists = np.linalg.norm(encoder_emb[graph_indices] - encoder_emb[a], axis=-1)
        # Candidates: different room, within 2x median distance
        cand_mask = np.array([
            int(global_room[int(g)]) != a_room and int(global_room[int(g)]) >= 0
            for g in graph_indices
        ], dtype=bool)
        close_mask = dists < 2.0 * median_dist
        valid = cand_mask & close_mask
        valid_idx = graph_indices[valid]

        if len(valid_idx) == 0:
            diff_room = graph_indices[cand_mask]
            if len(diff_room) > 0:
                d_diff = np.linalg.norm(encoder_emb[diff_room] - encoder_emb[a], axis=-1)
                k = min(n_hard_neg, len(diff_room))
                top_k = np.argpartition(d_diff, kth=k - 1)[:k]
                valid_idx = diff_room[top_k]
            else:
                continue

        if len(valid_idx) > n_hard_neg:
            d_valid = np.linalg.norm(encoder_emb[valid_idx] - encoder_emb[a], axis=-1)
            top_k = np.argpartition(d_valid, kth=n_hard_neg - 1)[:n_hard_neg]
            valid_idx = valid_idx[top_k]

        hard_neg_map[a] = valid_idx

    return anchors, positives, hard_neg_map


def infonce_projector_loss(
    encoder: ConvEncoder,
    projector: GeoProjector,
    obs_tensor: torch.Tensor,
    anchor_idx: np.ndarray,
    pos_idx: np.ndarray,
    neg_idx_map: dict[int, np.ndarray],
    batch_size: int,
    temperature: float,
    device: torch.device,
    rng: np.random.Generator,
    *,
    grad_encoder: bool,
) -> torch.Tensor:
    """InfoNCE on L2-normalized projector features; encoder grads optional."""
    n_pairs = len(anchor_idx)
    batch_idx = rng.choice(n_pairs, size=min(batch_size, n_pairs), replace=False)

    batch_anchors = anchor_idx[batch_idx]
    batch_pos = pos_idx[batch_idx]

    all_indices: set[int] = set()
    for a in batch_anchors.tolist():
        all_indices.add(int(a))
        if int(a) in neg_idx_map:
            for n in neg_idx_map[int(a)].tolist():
                all_indices.add(int(n))
    for p in batch_pos.tolist():
        all_indices.add(int(p))
    sorted_idx = sorted(all_indices)
    idx_to_local = {g: i for i, g in enumerate(sorted_idx)}

    obs_batch = obs_tensor[sorted_idx].contiguous()

    if grad_encoder:
        e = encoder(obs_batch)
        z = projector(e)
    else:
        with torch.no_grad():
            e = encoder(obs_batch)
        z = projector(e)

    total_loss = torch.tensor(0.0, device=device)
    count = 0
    for bi in range(len(batch_anchors)):
        a = int(batch_anchors[bi])
        p = int(batch_pos[bi])
        if a not in neg_idx_map or len(neg_idx_map[a]) == 0:
            continue

        a_z = z[idx_to_local[a]]
        p_z = z[idx_to_local[p]]

        neg_globals = neg_idx_map[a]
        n_z = z[[idx_to_local[int(n)] for n in neg_globals.tolist()
                 if int(n) in idx_to_local]]

        if len(n_z) == 0:
            continue

        pos_sim = (a_z * p_z).sum() / temperature
        neg_sim = (a_z.unsqueeze(0) * n_z).sum(dim=-1) / temperature

        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim], dim=0)
        labels = torch.zeros(1, dtype=torch.long, device=device)
        total_loss = total_loss + F.cross_entropy(logits.unsqueeze(0), labels)
        count += 1

    if count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    return total_loss / count


# =====================================================================
# Reconstruction regularization loss
# =====================================================================

def reconstruction_loss(encoder: ConvEncoder, decoder: ConvDecoder,
                        rssm: RSSM, obs_tensor: torch.Tensor,
                        batch_size: int, device: torch.device,
                        rng: np.random.Generator, cfg: TrainCfg):
    """Pixel reconstruction loss to prevent encoder from forgetting."""
    N = obs_tensor.shape[0]
    idx = rng.choice(N, size=min(batch_size, N), replace=False)
    obs_batch = obs_tensor[idx]
    e = encoder(obs_batch)
    h, s = rssm.get_init_state(e)
    recon = decoder(h, s)
    target = obs_batch
    return F.mse_loss(recon, target)


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
        emb = encoder(obs_tensor[start:end].contiguous())
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0).astype(np.float32)


@torch.no_grad()
def recompute_latent_repr(
    encoder: ConvEncoder,
    projector: GeoProjector,
    obs_tensor: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """L2-normalized projector features for every replay row."""
    encoder.eval()
    projector.eval()
    N = obs_tensor.shape[0]
    out: list[np.ndarray] = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = obs_tensor[start:end].contiguous()
        z = projector(encoder(x))
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def adjusted_rand_index(yt: np.ndarray, yp: np.ndarray) -> float:
    """Adjusted Rand index (oracle-free labels vs predicted); needs scipy.special.comb."""
    yt = np.asarray(yt, dtype=np.int64)
    yp = np.asarray(yp, dtype=np.int64)
    n = len(yt)
    if n < 2:
        return 0.0
    classes = np.unique(yt)
    clusters = np.unique(yp)
    ct = np.zeros((len(classes), len(clusters)), dtype=np.int64)
    for i, c in enumerate(classes):
        for j, k in enumerate(clusters):
            ct[i, j] = int(np.sum((yt == c) & (yp == k)))
    nijs = ct.ravel()
    ni = ct.sum(axis=1)
    nj = ct.sum(axis=0)
    tn = float(comb(nijs, 2, exact=False).sum())
    sum_comb_c = float(comb(ni, 2, exact=False).sum())
    sum_comb_k = float(comb(nj, 2, exact=False).sum())
    cn = float(comb(n, 2, exact=False))
    if cn < 1e-12:
        return 0.0
    prod_comb = sum_comb_c * sum_comb_k / cn
    mean_comb = 0.5 * (sum_comb_c + sum_comb_k)
    denom = mean_comb - prod_comb
    if abs(denom) < 1e-12:
        return 0.0
    return float((tn - prod_comb) / denom)


def warmup_temporal_projector(
    encoder: ConvEncoder,
    projector: GeoProjector,
    obs_tensor: torch.Tensor,
    episode_ids: np.ndarray,
    device: torch.device,
    rng: np.random.Generator,
    n_steps: int,
    lr: float,
    temperature: float,
    batch: int = 128,
    n_neg: int = 24,
) -> None:
    """Projector-only InfoNCE using consecutive same-episode indices as positives."""
    if n_steps <= 0:
        return
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    projector.train()
    opt = torch.optim.Adam(projector.parameters(), lr=lr)
    ep = np.asarray(episode_ids, dtype=np.int64)
    N = int(len(ep))
    if N < 4:
        for p in encoder.parameters():
            p.requires_grad = True
        return
    for _ in range(n_steps):
        anchors = []
        positives = []
        for _ in range(batch * 4):
            i = int(rng.integers(0, N - 1))
            if ep[i] == ep[i + 1]:
                anchors.append(i)
                positives.append(i + 1)
            if len(anchors) >= batch:
                break
        if len(anchors) < 8:
            continue
        anchors = np.asarray(anchors[:batch], dtype=np.int64)
        positives = np.asarray(positives[:batch], dtype=np.int64)
        neg = rng.integers(0, N, size=(len(anchors), n_neg), dtype=np.int64)
        for r in range(neg.shape[0]):
            for c in range(neg.shape[1]):
                if neg[r, c] == anchors[r] or neg[r, c] == positives[r]:
                    neg[r, c] = (neg[r, c] + 1) % N

        idx_set: set[int] = set()
        for a, p in zip(anchors.tolist(), positives.tolist()):
            idx_set.add(int(a))
            idx_set.add(int(p))
        for row in neg.tolist():
            for j in row:
                idx_set.add(int(j))
        idx_list = sorted(idx_set)
        local = {g: i for i, g in enumerate(idx_list)}
        obs_b = obs_tensor[idx_list].contiguous()
        with torch.no_grad():
            e_all = encoder(obs_b)
        z_all = projector(e_all)
        opt.zero_grad(set_to_none=True)
        total = torch.tensor(0.0, device=device)
        for bi in range(len(anchors)):
            a = local[int(anchors[bi])]
            p = local[int(positives[bi])]
            a_z = z_all[a]
            p_z = z_all[p]
            n_idx = [local[int(x)] for x in neg[bi].tolist()]
            n_z = z_all[n_idx]
            pos_sim = (a_z * p_z).sum() / temperature
            neg_sim = (a_z.unsqueeze(0) * n_z).sum(dim=-1) / temperature
            logits = torch.cat([pos_sim.unsqueeze(0), neg_sim], dim=0)
            labels = torch.zeros(1, dtype=torch.long, device=device)
            total = total + F.cross_entropy(logits.unsqueeze(0), labels)
        (total / len(anchors)).backward()
        opt.step()
    for p in encoder.parameters():
        p.requires_grad = True


# =====================================================================
# Convergence check
# =====================================================================

def rooms_converged(rooms_new: np.ndarray, rooms_prev: np.ndarray | None,
                    tolerance: float = 0.05) -> bool:
    """Check if room partitioning has stabilized (pairwise co-assignment agreement)."""
    if rooms_prev is None:
        return False
    if len(rooms_new) != len(rooms_prev):
        return False

    n_rooms = int(rooms_new.max()) + 1
    if n_rooms < 2:
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


def oracle_cell_rooms(geodesic: GeodesicComputer) -> tuple[np.ndarray, set[tuple[int, int]]]:
    """Room label per free cell (components after deleting maze bridges) + bridge edge set."""
    adj = _adj_from_distmat(geodesic.dist_matrix)
    bridges = _find_bridges(adj)
    comp, _ = _components_without_bridges(adj, bridges)
    return comp.astype(np.int64), bridges


def evaluate_topology_metrics(
    pos: np.ndarray,
    geodesic: GeodesicComputer,
    idx_global: np.ndarray,
    edges: np.ndarray,
    edge_scores: np.ndarray,
    pred_room_global: np.ndarray,
    top_frac: float = 0.15,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Room–oracle agreement, cut-edge enrichment, bridge-adjacent hit rate on top-scored edges."""
    if rng is None:
        rng = np.random.default_rng(0)

    oracle_rooms, bridge_set = oracle_cell_rooms(geodesic)
    cells = _positions_to_cell_indices(geodesic, pos).astype(np.int64)
    oracle_for_replay = oracle_rooms[cells]

    valid = pred_room_global >= 0
    if valid.sum() >= 4:
        ari = adjusted_rand_index(oracle_for_replay[valid], pred_room_global[valid])
    else:
        ari = 0.0

    def _cell_pair(gu: int, gv: int) -> tuple[int, int]:
        cu, cv = int(cells[gu]), int(cells[gv])
        return (cu, cv) if cu < cv else (cv, cu)

    n_e = len(edges)
    room_cut = np.zeros(n_e, dtype=bool)
    bridge_hit = np.zeros(n_e, dtype=bool)
    for ei in range(n_e):
        u, v = int(edges[ei, 0]), int(edges[ei, 1])
        gu, gv = int(idx_global[u]), int(idx_global[v])
        cu, cv = int(cells[gu]), int(cells[gv])
        room_cut[ei] = oracle_rooms[cu] != oracle_rooms[cv]
        bridge_hit[ei] = _cell_pair(gu, gv) in bridge_set

    if n_e > 0:
        thr = float(np.percentile(edge_scores, 100.0 * (1.0 - top_frac)))
        top_mask = edge_scores >= thr
        rand_mask = np.zeros(n_e, dtype=bool)
        rand_mask[rng.choice(n_e, size=min(n_e, max(1, int(top_mask.sum()))), replace=False)] = True
        p_top = float(room_cut[top_mask].mean()) if top_mask.any() else 0.0
        p_rand = float(room_cut[rand_mask].mean()) if rand_mask.any() else 0.0
        cut_enrich = p_top / max(p_rand, 1e-6)
        b_top = float(bridge_hit[top_mask].mean()) if top_mask.any() else 0.0
        b_rand = float(bridge_hit[rand_mask].mean()) if rand_mask.any() else 0.0
        bridge_enrich = b_top / max(b_rand, 1e-6)
    else:
        p_top = p_rand = cut_enrich = b_top = b_rand = bridge_enrich = 0.0

    n_oracle_rooms = int(oracle_rooms.max()) + 1
    n_pred = int(pred_room_global[valid].max()) + 1 if valid.any() else 0

    return {
        "adjusted_rand_vs_oracle_rooms": float(ari),
        "n_oracle_rooms": float(n_oracle_rooms),
        "n_pred_rooms_on_graph": float(n_pred),
        "cut_edge_rate_top_frac": float(p_top),
        "cut_edge_rate_random": float(p_rand),
        "cut_edge_enrichment": float(cut_enrich),
        "bridge_hit_rate_top_frac": float(b_top),
        "bridge_hit_rate_random": float(b_rand),
        "bridge_hit_enrichment": float(bridge_enrich),
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
    ax.set_xlabel("Combined edge score (z-detour + z-routed mismatch)")
    ax.set_ylabel("Edge count")
    ax.set_title(f"Edge score distribution (iter {iteration})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_convergence(history: list[dict], out_path: str):
    """Topology-focused metrics across iterations."""
    iters = [h["iteration"] for h in history]
    n_rooms = [h["n_rooms"] for h in history]
    ari = [h.get("adjusted_rand_vs_oracle_rooms", 0.0) for h in history]
    cut_e = [h.get("cut_edge_enrichment", 0.0) for h in history]
    bridge_e = [h.get("bridge_hit_enrichment", 0.0) for h in history]
    geo_loss = [h.get("mean_geo_loss", 0.0) for h in history]

    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))

    axes[0].plot(iters, n_rooms, "o-", color="teal")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Predicted rooms")
    axes[0].set_title("Room count")

    axes[1].plot(iters, ari, "s-", color="darkgreen")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("ARI vs oracle rooms")
    axes[1].set_title("Room agreement")

    axes[2].plot(iters, cut_e, "^-", color="coral")
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Enrichment")
    axes[2].set_title("Cut-edge enrichment")

    axes[3].plot(iters, bridge_e, "d-", color="purple")
    axes[3].set_xlabel("Iteration")
    axes[3].set_ylabel("Enrichment")
    axes[3].set_title("Bridge-hit enrichment")

    axes[4].plot(iters, geo_loss, "v-", color="crimson")
    axes[4].set_xlabel("Iteration")
    axes[4].set_ylabel("L_geo")
    axes[4].set_title("InfoNCE (projector)")

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
    """Bootstrap loop on projector features: detour+routed scores → rooms → InfoNCE → repeat."""
    out_dir = cfg_barrier.output_dir
    os.makedirs(out_dir, exist_ok=True)

    encoder = models["encoder"]
    decoder = models["decoder"]
    rssm = models["rssm"]
    geodesic = env.geodesic

    rng = np.random.default_rng(cfg_barrier.seed)

    projector = GeoProjector(
        int(cfg_train.embed_dim),
        int(cfg_barrier.projector_hidden),
        int(cfg_barrier.projector_dim),
    ).to(device)

    # ---- Collect replay data ----
    print("\n  [Collect] Gathering replay data ...")
    cfg_train.collect_episodes = cfg_barrier.collect_episodes
    data = collect_data(env, models, cfg_train, device)
    pos = data["pos"]
    episode_ids = data["episode_ids"]
    raw_obs = data["raw_obs"]
    N = len(pos)
    print(f"    {N} replay states from {cfg_barrier.collect_episodes} episodes")

    img_size = cfg_train.img_size
    obs_images = raw_obs.reshape(N, img_size, img_size, 3)
    obs_tensor = torch.tensor(obs_images, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    preprocess_img(obs_tensor, depth=cfg_train.bit_depth)

    print("  [Warmup] Projector-only temporal InfoNCE ...")
    warmup_temporal_projector(
        encoder, projector, obs_tensor, episode_ids, device, rng,
        n_steps=int(cfg_barrier.projector_warmup_steps),
        lr=float(cfg_barrier.projector_lr),
        temperature=float(cfg_barrier.geo_temperature),
    )

    latent_np = recompute_latent_repr(encoder, projector, obs_tensor, device)
    encoder_emb = recompute_embeddings(encoder, obs_tensor, device)

    eval0_enc = evaluate_encoder_geometry(
        encoder_emb, pos, geodesic, n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
    print(f"    Initial encoder–geodesic Spearman ρ = {eval0_enc['spearman_enc_vs_geodesic']:.4f} (diagnostic)")

    projector_opt = torch.optim.Adam(projector.parameters(), lr=float(cfg_barrier.projector_lr))
    encoder_opt = None
    if cfg_barrier.finetune_encoder or int(cfg_barrier.finetune_encoder_steps) > 0:
        encoder_opt = torch.optim.Adam(encoder.parameters(), lr=float(cfg_barrier.finetune_encoder_lr))

    history: list[dict] = []
    prev_rooms: np.ndarray | None = None
    patience_counter = 0

    for iteration in range(cfg_barrier.max_iterations):
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration + 1}/{cfg_barrier.max_iterations}")
        print(f"{'='*60}")

        print("  [Phase 1] Building mixed replay graph (projector features) ...")
        idx_global, g2l, adj_list, edges, n_nodes, n_edges = build_replay_graph(
            latent_np, episode_ids,
            n_graph_max=cfg_barrier.graph_max,
            k_knn=cfg_barrier.knn_k,
        )
        emb_sub = latent_np[idx_global]
        print(f"    Graph: {n_nodes} nodes, {n_edges} edges")

        if n_nodes < 16:
            print("    Graph too small, stopping.")
            break

        print("  [Phase 2] All-pairs shortest paths ...")
        d_graph = compute_graph_distances(adj_list)

        print("  [Phase 3] Edge scores (detour without edge + routed mismatch) ...")
        edge_scores = compute_combined_edge_barrier_scores(
            edges, adj_list, d_graph, emb_sub, rng,
            detour_max_edges=int(cfg_barrier.detour_max_edges),
            unreachable_cap_mul=int(cfg_barrier.unreachable_cap_mul),
            route_sample_pairs=int(cfg_barrier.route_sample_pairs),
            route_d_min=int(cfg_barrier.route_d_min),
            w_detour=float(cfg_barrier.score_w_detour),
            w_route=float(cfg_barrier.score_w_route),
        )

        barrier_thresh = float(np.percentile(
            edge_scores, 100.0 * (1.0 - cfg_barrier.barrier_threshold_frac)))
        n_high = int(np.sum(edge_scores >= barrier_thresh))
        print(f"    Prune threshold={barrier_thresh:.3f}, high-score edges={n_high}/{len(edge_scores)}")

        plot_barrier_distribution(
            edge_scores, barrier_thresh,
            os.path.join(out_dir, f"barrier_dist_iter{iteration}.png"),
            iteration=iteration,
        )

        print("  [Phase 4] Partition + escape hatch if single room ...")
        room_labels = partition_into_rooms(
            adj_list, edges, edge_scores,
            threshold_frac=cfg_barrier.barrier_threshold_frac,
            min_room_size=cfg_barrier.min_room_size,
        )
        n_rooms_pre = int(room_labels.max()) + 1
        room_labels, hatch = apply_escape_hatch_singleton(
            room_labels, adj_list, idx_global, episode_ids, emb_sub, rng)
        n_rooms = int(room_labels.max()) + 1
        room_sizes = [int(np.sum(room_labels == r)) for r in range(n_rooms)]
        print(f"    Rooms after prune: {n_rooms_pre}; after hatch ({hatch}): {n_rooms} — sizes={room_sizes}")

        global_room = np.full(N, -1, dtype=np.int64)
        for loc, g in enumerate(idx_global.tolist()):
            global_room[int(g)] = int(room_labels[loc])

        topo = evaluate_topology_metrics(
            pos, geodesic, idx_global, edges, edge_scores, global_room,
            top_frac=float(cfg_barrier.topo_top_frac), rng=rng,
        )
        print(f"    ARI vs oracle rooms={topo['adjusted_rand_vs_oracle_rooms']:.4f}  "
              f"cut-enrich={topo['cut_edge_enrichment']:.3f}  "
              f"bridge-enrich={topo['bridge_hit_enrichment']:.3f}")

        plot_rooms_on_maze(
            geodesic, pos, global_room,
            os.path.join(out_dir, f"rooms_iter{iteration}.png"),
            title=f"Rooms (iter {iteration}, hatch={hatch})",
        )

        converged = rooms_converged(room_labels, prev_rooms)
        if converged:
            patience_counter += 1
            print(f"    Partition stable ({patience_counter}/{cfg_barrier.convergence_patience})")
            if patience_counter >= cfg_barrier.convergence_patience:
                print("    Stopping (partition convergence).")
                encoder_emb = recompute_embeddings(encoder, obs_tensor, device)
                eval_final = evaluate_encoder_geometry(
                    encoder_emb, pos, geodesic, n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
                history.append({
                    "iteration": iteration,
                    "n_rooms": n_rooms,
                    "room_sizes": room_sizes,
                    "escape_hatch": hatch,
                    "n_high_barrier_edges": n_high,
                    "barrier_threshold": barrier_thresh,
                    "converged": True,
                    **topo,
                    **eval_final,
                })
                break
        else:
            patience_counter = 0
        prev_rooms = room_labels.copy()

        print("  [Phase 5] Mine pairs (projector latent space) ...")
        anchors, positives, hard_neg_map = mine_pairs(
            latent_np, room_labels, episode_ids, idx_global,
            pos_k=cfg_barrier.pos_k,
            n_hard_neg=cfg_barrier.n_hard_neg,
            rng=rng,
        )

        if anchors is None or len(anchors) == 0:
            print("    No valid pairs; skipping training.")
            encoder_emb = recompute_embeddings(encoder, obs_tensor, device)
            eval_iter = evaluate_encoder_geometry(
                encoder_emb, pos, geodesic, n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
            history.append({
                "iteration": iteration,
                "n_rooms": n_rooms,
                "room_sizes": room_sizes,
                "escape_hatch": hatch,
                "n_high_barrier_edges": n_high,
                "barrier_threshold": barrier_thresh,
                "mean_geo_loss": 0.0,
                "converged": False,
                **topo,
                **eval_iter,
            })
            latent_np = recompute_latent_repr(encoder, projector, obs_tensor, device)
            continue

        n_anchors_with_neg = sum(1 for a in np.unique(anchors) if int(a) in hard_neg_map)
        print(f"    {len(anchors)} positives, {n_anchors_with_neg} anchors with hard negatives")

        print(f"  [Phase 5b] Train projector ({cfg_barrier.projector_steps} steps, encoder frozen) ...")
        projector.train()
        encoder.eval()
        geo_losses: list[float] = []

        for step in range(int(cfg_barrier.projector_steps)):
            projector_opt.zero_grad(set_to_none=True)
            l_geo = infonce_projector_loss(
                encoder, projector, obs_tensor,
                anchors, positives, hard_neg_map,
                batch_size=int(cfg_barrier.geo_batch),
                temperature=float(cfg_barrier.geo_temperature),
                device=device, rng=rng,
                grad_encoder=False,
            )
            (cfg_barrier.geo_lambda * l_geo).backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 10.0)
            projector_opt.step()
            geo_losses.append(float(l_geo.item()))
            if (step + 1) % max(1, int(cfg_barrier.projector_steps) // 5) == 0:
                print(f"      proj step {step+1}/{cfg_barrier.projector_steps}  L_geo={l_geo.item():.4f}")

        n_ft = int(cfg_barrier.finetune_encoder_steps)
        if cfg_barrier.finetune_encoder and encoder_opt is not None and n_ft > 0:
            print(f"  [Phase 5c] Optional encoder FT ({n_ft} steps) ...")
            encoder.train()
            projector.train()
            for step in range(n_ft):
                encoder_opt.zero_grad(set_to_none=True)
                projector_opt.zero_grad(set_to_none=True)
                l_geo = infonce_projector_loss(
                    encoder, projector, obs_tensor,
                    anchors, positives, hard_neg_map,
                    batch_size=int(cfg_barrier.geo_batch),
                    temperature=float(cfg_barrier.geo_temperature),
                    device=device, rng=rng,
                    grad_encoder=True,
                )
                l_recon = reconstruction_loss(
                    encoder, decoder, rssm, obs_tensor,
                    batch_size=int(cfg_barrier.recon_batch),
                    device=device, rng=rng, cfg=cfg_train,
                )
                loss = cfg_barrier.recon_lambda * l_recon + cfg_barrier.geo_lambda * l_geo
                loss.backward()
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0)
                torch.nn.utils.clip_grad_norm_(projector.parameters(), 5.0)
                encoder_opt.step()
                projector_opt.step()
                if (step + 1) % max(1, n_ft // 4) == 0:
                    print(f"      enc step {step+1}/{n_ft}  L_total={loss.item():.4f}")

        mean_geo_loss = float(np.mean(geo_losses)) if geo_losses else 0.0

        print("  [Phase 6] Refresh latent representation ...")
        latent_np = recompute_latent_repr(encoder, projector, obs_tensor, device)
        encoder_emb = recompute_embeddings(encoder, obs_tensor, device)
        eval_iter = evaluate_encoder_geometry(
            encoder_emb, pos, geodesic, n_pairs=cfg_barrier.eval_n_pairs, rng=rng)
        dt = time.time() - t0
        print(f"    ARI={topo['adjusted_rand_vs_oracle_rooms']:.4f}  "
              f"ρ(enc,geo)={eval_iter['spearman_enc_vs_geodesic']:.4f}  ({dt:.1f}s)")

        history.append({
            "iteration": iteration,
            "n_rooms": n_rooms,
            "room_sizes": room_sizes,
            "escape_hatch": hatch,
            "n_high_barrier_edges": n_high,
            "barrier_threshold": barrier_thresh,
            "mean_geo_loss": mean_geo_loss,
            "elapsed_s": dt,
            "converged": False,
            **topo,
            **eval_iter,
        })

    plot_convergence(history, os.path.join(out_dir, "convergence.png"))

    final_ckpt = {
        "encoder": encoder.state_dict(),
        "projector": projector.state_dict(),
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
        hf = history[-1]
        print(f"    ARI (oracle rooms): {hf.get('adjusted_rand_vs_oracle_rooms', 0.0):.4f}")
        print(f"    Cut-edge enrichment: {hf.get('cut_edge_enrichment', 0.0):.3f}")
        print(f"    Rooms: {hf.get('n_rooms', 0)}")

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
    p.add_argument("--geo_lambda", type=float, default=0.5)
    p.add_argument("--projector_steps", type=int, default=200,
                   help="Projector InfoNCE steps per iteration (alias: was --geo_steps)")
    p.add_argument("--geo_batch", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--finetune_encoder", action="store_true",
                   help="After projector steps, fine-tune encoder (+ recon) for --ft_steps")
    p.add_argument("--ft_steps", type=int, default=0,
                   help="Encoder fine-tune steps (requires --finetune_encoder)")
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
        geo_lambda=args.geo_lambda,
        projector_steps=args.projector_steps,
        geo_batch=args.geo_batch,
        geo_temperature=args.temperature,
        max_iterations=args.max_iter,
        finetune_encoder=bool(args.finetune_encoder),
        finetune_encoder_steps=int(args.ft_steps),
    )

    if args.quick:
        cfg_barrier.collect_episodes = 20
        cfg_barrier.projector_steps = 40
        cfg_barrier.projector_warmup_steps = 30
        cfg_barrier.max_iterations = 3
        cfg_barrier.graph_max = 1200
        cfg_barrier.detour_max_edges = 800
        cfg_barrier.route_sample_pairs = 400
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
