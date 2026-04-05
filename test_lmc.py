#!/usr/bin/env python3

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import spearmanr

from models import RSSM, Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel
from utils import get_device, preprocess_img, set_seed
from pointmaze_large_topo_v2 import PointMazeLargeDiverseGRWrapper
from maze_geometry_test import TrainCfg, _positions_to_cell_indices


# =====================================================================
# Config
# =====================================================================

@dataclass
class Cfg:
    seed: int = 0
    wm_path: str = "world_model.pt"
    output_dir: str = "lmc_v3_results"
    mode: str = "replay"   # replay | pos_boot | oracle
    quick: bool = False

    collect_episodes: int = 60
    total_steps: int = 3000
    batch_triplets: int = 256
    margin: float = 1.0
    head_feature: str = "h+s"  # encoder_e | h | s | h+s
    head_dim: int = 128
    head_hidden: int = 256
    head_lr: float = 3e-4
    weight_decay: float = 1e-5
    grad_clip: float = 5.0

    # variance regularizer on head output: hinge on per-dim std
    var_lambda: float = 0.05
    target_std: float = 1.0

    # replay-graph options
    graph_max: int = 1800
    knn_k: int = 8
    replay_knn_radius_mult: float = 1.25
    replay_far_quantile: float = 0.70
    replay_close_quantile: float = 0.20

    # geodesic teacher options (oracle / pos_boot)
    close_max_geodesic: float = 2.5
    far_min_geodesic: float = 6.0

    # adaptive fallback if thresholds are too strict
    near_k_min: int = 8
    far_k_min: int = 16
    max_triplet_attempt_mult: int = 40
    min_triplets_ok: int = 32

    # position bootstrap
    pos_probe_hidden: int = 256
    pos_probe_epochs: int = 500
    pos_probe_lr: float = 3e-4
    pos_probe_batch: int = 256

    # evaluation
    eval_interval: int = 200
    eval_n_pairs: int = 2000
    room_knn_k: int = 8


# =====================================================================
# Helpers
# =====================================================================


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple, set)):
        return list(obj)
    return str(obj)


def _spearman_rho_numpy(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or len(y) < 3:
        return 0.0
    rho = spearmanr(x, y).correlation
    if rho is None or not np.isfinite(rho):
        return 0.0
    return float(rho)


def _adj_from_distmat(dist_matrix: np.ndarray) -> List[List[int]]:
    n = dist_matrix.shape[0]
    adj = [[] for _ in range(n)]
    finite = np.isfinite(dist_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if finite[i, j] and dist_matrix[i, j] > 0:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _find_bridges(adj: List[List[int]]) -> set[Tuple[int, int]]:
    n = len(adj)
    tin = [-1] * n
    low = [-1] * n
    timer = 0
    bridges: set[Tuple[int, int]] = set()

    def dfs(v: int, p: int) -> None:
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        for to in adj[v]:
            if to == p:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v)
                low[v] = min(low[v], low[to])
                if low[to] > tin[v]:
                    a, b = (v, to) if v < to else (to, v)
                    bridges.add((a, b))

    for v in range(n):
        if tin[v] == -1:
            dfs(v, -1)
    return bridges


def _components_without_bridges(adj: List[List[int]], bridges: set[Tuple[int, int]]) -> Tuple[np.ndarray, int]:
    n = len(adj)
    comp = np.full(n, -1, dtype=np.int64)
    cid = 0
    for s in range(n):
        if comp[s] != -1:
            continue
        stack = [s]
        comp[s] = cid
        while stack:
            v = stack.pop()
            for to in adj[v]:
                a, b = (v, to) if v < to else (to, v)
                if (a, b) in bridges:
                    continue
                if comp[to] == -1:
                    comp[to] = cid
                    stack.append(to)
        cid += 1
    return comp, cid


def compute_local_time_indices(episode_ids: np.ndarray) -> np.ndarray:
    out = np.zeros_like(episode_ids, dtype=np.int64)
    prev = None
    t = 0
    for i, ep in enumerate(episode_ids.tolist()):
        if prev is None or ep != prev:
            t = 0
        else:
            t += 1
        out[i] = t
        prev = ep
    return out


# =====================================================================
# Replay data collection
# =====================================================================

@torch.no_grad()
def collect_replay_data(env, models: dict, cfg_train: TrainCfg, device: torch.device, n_episodes: int):
    encoder = models["encoder"]
    rssm = models["rssm"]
    actor = models.get("actor", None)
    encoder.eval()
    rssm.eval()
    if actor is not None:
        actor.eval()

    all_pos, all_h, all_s, all_e, all_obs, ep_ids = [], [], [], [], [], []

    for ep_idx in range(n_episodes):
        obs, _ = env.reset()
        done = False

        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=cfg_train.bit_depth)
        e = encoder(obs_t)
        h, s = rssm.get_init_state(e)

        all_pos.append(env.agent_pos.copy())
        all_h.append(h.squeeze(0).cpu().numpy())
        all_s.append(s.squeeze(0).cpu().numpy())
        all_e.append(e.squeeze(0).cpu().numpy())
        all_obs.append(obs.astype(np.float32).flatten() / 255.0)
        ep_ids.append(ep_idx)

        while not done:
            if actor is not None:
                a_t, _ = actor.get_action(h, s, deterministic=False)
                a_t = torch.clamp(a_t + 0.1 * torch.randn_like(a_t), -1, 1)
                a_np = a_t.squeeze(0).cpu().numpy().astype(np.float32)
            else:
                a_np = env.action_space.sample().astype(np.float32)

            obs, _, t, tr, _ = env.step(a_np, repeat=cfg_train.action_repeat)
            done = bool(t or tr)

            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=cfg_train.bit_depth)
            e = encoder(obs_t)
            act_t = torch.tensor(a_np, dtype=torch.float32, device=device).unsqueeze(0)
            h, s, _, _ = rssm.observe_step(e, act_t, h, s, sample=False)

            all_pos.append(env.agent_pos.copy())
            all_h.append(h.squeeze(0).cpu().numpy())
            all_s.append(s.squeeze(0).cpu().numpy())
            all_e.append(e.squeeze(0).cpu().numpy())
            all_obs.append(obs.astype(np.float32).flatten() / 255.0)
            ep_ids.append(ep_idx)

    data = {
        "pos": np.asarray(all_pos, dtype=np.float32),
        "h": np.asarray(all_h, dtype=np.float32),
        "s": np.asarray(all_s, dtype=np.float32),
        "encoder_e": np.asarray(all_e, dtype=np.float32),
        "raw_obs": np.asarray(all_obs, dtype=np.float32),
        "episode_ids": np.asarray(ep_ids, dtype=np.int64),
    }
    print(f"    Collected {len(data['pos'])} data points from {n_episodes} episodes")
    return data


# =====================================================================
# Models
# =====================================================================

class LatentMetricHead(nn.Module):
    def __init__(self, in_dim: int, head_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, head_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ELU(),
            nn.Linear(hidden, hidden),
            nn.ELU(),
            nn.Linear(hidden, 2),
        )
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(2))
        self.register_buffer("y_std", torch.ones(2))
        self.feature_name = "unknown"
        self.best_r2 = float("-inf")

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std.clamp_min(1e-6)

    def _denormalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std.clamp_min(1e-6) + self.y_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(self._normalize_x(x))
        return self._denormalize_y(z)


# =====================================================================
# Position bootstrap
# =====================================================================


def train_position_probe(feature_bank: Dict[str, np.ndarray], positions: np.ndarray, cfg: Cfg, device: torch.device):
    rng = np.random.default_rng(cfg.seed + 777)
    best_probe = None
    best_name = None
    best_r2 = float("-inf")

    for name in ["h", "h+s", "encoder_e", "s"]:
        x_np = feature_bank[name].astype(np.float32)
        y_np = positions.astype(np.float32)
        N, in_dim = x_np.shape
        probe = PositionProbe(in_dim, cfg.pos_probe_hidden).to(device)
        probe.x_mean.copy_(torch.tensor(x_np.mean(0), device=device))
        probe.x_std.copy_(torch.tensor(x_np.std(0) + 1e-6, device=device))
        probe.y_mean.copy_(torch.tensor(y_np.mean(0), device=device))
        probe.y_std.copy_(torch.tensor(y_np.std(0) + 1e-6, device=device))
        opt = torch.optim.Adam(probe.parameters(), lr=cfg.pos_probe_lr)

        e_t = torch.tensor(x_np, dtype=torch.float32, device=device)
        p_t = torch.tensor(y_np, dtype=torch.float32, device=device)
        n_val = max(1, int(0.15 * N))
        perm = rng.permutation(N)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]

        best_val = float("inf")
        best_state = None
        for ep in range(cfg.pos_probe_epochs):
            idx = rng.choice(tr_idx, size=min(cfg.pos_probe_batch, len(tr_idx)), replace=False)
            pred = probe(e_t[idx])
            loss = F.mse_loss(pred, p_t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (ep + 1) % max(1, cfg.pos_probe_epochs // 5) == 0:
                with torch.no_grad():
                    val_pred = probe(e_t[val_idx])
                    val_loss = F.mse_loss(val_pred, p_t[val_idx]).item()
                if val_loss < best_val:
                    best_val = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in probe.state_dict().items()}
                print(f"    pos_probe[{name}] epoch {ep+1}/{cfg.pos_probe_epochs}  train={loss.item():.4f}  val={val_loss:.4f}")

        if best_state is not None:
            probe.load_state_dict(best_state)
        with torch.no_grad():
            all_pred = probe(e_t).cpu().numpy()
        ss_res = float(np.sum((positions - all_pred) ** 2))
        ss_tot = float(np.sum((positions - positions.mean(axis=0)) ** 2))
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
        probe.feature_name = name
        probe.best_r2 = r2
        print(f"    Position probe[{name}]: R² = {r2:.4f}  best_val_mse = {best_val:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_probe = probe
            best_name = name

    assert best_probe is not None and best_name is not None
    print(f"    Using pos_boot feature: {best_name}  (best R²={best_r2:.4f})")
    return best_probe.eval(), best_name


# =====================================================================
# Replay graph teacher
# =====================================================================


def _standardize_features(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def select_graph_nodes(episode_ids: np.ndarray, graph_max: int) -> np.ndarray:
    N = len(episode_ids)
    if N <= graph_max:
        return np.arange(N, dtype=np.int64)
    idx_sel = []
    start = 0
    unique_eps, counts = np.unique(episode_ids, return_counts=True)
    total = int(counts.sum())
    remaining_budget = graph_max
    for ep, cnt in zip(unique_eps.tolist(), counts.tolist()):
        alloc = max(2, int(round(graph_max * (cnt / total))))
        alloc = min(alloc, cnt)
        ep_idx = np.arange(start, start + cnt, dtype=np.int64)
        if alloc >= cnt:
            chosen = ep_idx
        else:
            chosen = np.linspace(0, cnt - 1, num=alloc, dtype=np.int64)
            chosen = ep_idx[chosen]
        idx_sel.extend(chosen.tolist())
        start += cnt
    idx_sel = np.unique(np.asarray(idx_sel, dtype=np.int64))
    if len(idx_sel) > graph_max:
        keep = np.linspace(0, len(idx_sel) - 1, num=graph_max, dtype=np.int64)
        idx_sel = idx_sel[keep]
    return idx_sel


def build_replay_graph_teacher(features: np.ndarray, episode_ids: np.ndarray, cfg: Cfg):
    idx_global = select_graph_nodes(episode_ids, cfg.graph_max)
    X = _standardize_features(features[idx_global].astype(np.float32))
    ep_sel = episode_ids[idx_global].astype(np.int64)
    t_all = compute_local_time_indices(episode_ids)
    t_sel = t_all[idx_global].astype(np.int64)
    M = len(idx_global)

    # Full pairwise distances on selected nodes.
    dm = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1).astype(np.float32)
    np.fill_diagonal(dm, np.inf)

    # Weighted undirected graph.
    adj = lil_matrix((M, M), dtype=np.float32)

    # Temporal edges with actual time gaps after subsampling.
    temporal_feat_d = []
    for i in range(M - 1):
        if ep_sel[i] == ep_sel[i + 1]:
            dt = float(max(1, int(t_sel[i + 1] - t_sel[i])))
            adj[i, i + 1] = dt
            adj[i + 1, i] = dt
            temporal_feat_d.append(float(dm[i, i + 1]))

    if temporal_feat_d:
        local_scale = float(np.median(temporal_feat_d))
        radius = float(np.quantile(np.asarray(temporal_feat_d), 0.90) * cfg.replay_knn_radius_mult)
    else:
        finite = dm[np.isfinite(dm)]
        local_scale = float(np.median(finite)) if finite.size else 1.0
        radius = float(np.quantile(finite, 0.10)) if finite.size else 1.0
    local_scale = max(local_scale, 1e-6)
    radius = max(radius, 1e-6)

    # Conservative cross-episode mutual kNN edges only.
    k = min(cfg.knn_k, M - 1)
    if k >= 1:
        knn = np.argsort(dm, axis=1)[:, :k]
        for i in range(M):
            for j in knn[i]:
                j = int(j)
                if ep_sel[i] == ep_sel[j]:
                    continue
                # mutual kNN
                if i not in knn[j]:
                    continue
                d = float(dm[i, j])
                if d > radius:
                    continue
                w = max(1.0, d / local_scale)
                if adj[i, j] == 0 or w < float(adj[i, j]):
                    adj[i, j] = w
                    adj[j, i] = w

    dist = shortest_path(adj.tocsr(), directed=False, unweighted=False).astype(np.float32)
    finite = dist[np.isfinite(dist) & (dist > 0)]
    if finite.size == 0:
        close_thr = 1.0
        far_thr = 2.0
    else:
        close_thr = float(np.quantile(finite, cfg.replay_close_quantile))
        far_thr = float(np.quantile(finite, cfg.replay_far_quantile))
        if far_thr <= close_thr:
            far_thr = close_thr + 1.0

    n_temporal = int((adj.count_nonzero() // 2))
    print(f"    Replay-graph nodes={M}  edges={n_temporal}  local_scale={local_scale:.4f}  radius={radius:.4f}")
    if finite.size:
        print(f"    Replay shortest-path stats: min={finite.min():.2f}  median={np.median(finite):.2f}  max={finite.max():.2f}")
        print(f"    Replay thresholds: close≤{close_thr:.2f}  far≥{far_thr:.2f}")

    return idx_global, dist, close_thr, far_thr


# =====================================================================
# Triplet sampling
# =====================================================================


def _empty_triplets():
    z = np.zeros((0,), dtype=np.int64)
    f = np.zeros((0,), dtype=np.float32)
    return z, z.copy(), z.copy(), f, f.copy()


def _select_near_far_candidates(d_row: np.ndarray, rng: np.random.Generator, close_max: float, far_min: float,
                                near_k_min: int, far_k_min: int):
    valid = np.isfinite(d_row) & (d_row > 0)
    if valid.sum() < 2:
        return None
    near = np.where(valid & (d_row <= close_max))[0]
    far = np.where(valid & (d_row >= far_min))[0]
    order = np.argsort(d_row)
    order = order[np.isfinite(d_row[order]) & (d_row[order] > 0)]
    if len(near) == 0:
        near = order[:min(len(order), near_k_min)]
    if len(far) == 0:
        far = order[max(0, len(order) - far_k_min):]
    if len(near) == 0 or len(far) == 0:
        return None
    j = int(near[rng.integers(len(near))])
    far = far[far != j]
    if len(far) == 0:
        return None
    k = int(far[rng.integers(len(far))])
    return j, k, float(d_row[j]), float(d_row[k])


def sample_triplets_from_graph(dist_graph: np.ndarray, idx_global: np.ndarray, rng: np.random.Generator, n_triplets: int,
                               close_max: float, far_min: float, cfg: Cfg):
    M = dist_graph.shape[0]
    anchors, closes, fars = [], [], []
    d_close, d_far = [], []
    attempts = 0
    max_attempts = max(n_triplets * cfg.max_triplet_attempt_mult, n_triplets + 1)
    while len(anchors) < n_triplets and attempts < max_attempts:
        attempts += 1
        i = int(rng.integers(0, M))
        picked = _select_near_far_candidates(dist_graph[i], rng, close_max, far_min, cfg.near_k_min, cfg.far_k_min)
        if picked is None:
            continue
        j, k, dj, dk = picked
        anchors.append(int(idx_global[i]))
        closes.append(int(idx_global[j]))
        fars.append(int(idx_global[k]))
        d_close.append(dj)
        d_far.append(dk)
    if not anchors:
        return _empty_triplets()
    return (np.asarray(anchors, np.int64), np.asarray(closes, np.int64), np.asarray(fars, np.int64),
            np.asarray(d_close, np.float32), np.asarray(d_far, np.float32))


def sample_triplets_geodesic(cell_indices: np.ndarray, dist_matrix: np.ndarray, rng: np.random.Generator,
                             n_triplets: int, close_max: float, far_min: float, cfg: Cfg):
    N = len(cell_indices)
    cell_to_replay: Dict[int, List[int]] = {}
    for i, c in enumerate(cell_indices.tolist()):
        cell_to_replay.setdefault(int(c), []).append(i)
    anchors, closes, fars = [], [], []
    d_close, d_far = [], []
    attempts = 0
    max_attempts = max(n_triplets * cfg.max_triplet_attempt_mult, n_triplets + 1)
    while len(anchors) < n_triplets and attempts < max_attempts:
        attempts += 1
        a_idx = int(rng.integers(0, N))
        a_cell = int(cell_indices[a_idx])
        picked = _select_near_far_candidates(dist_matrix[a_cell], rng, close_max, far_min, cfg.near_k_min, cfg.far_k_min)
        if picked is None:
            continue
        jc, kc, dj, dk = picked
        jpool = cell_to_replay.get(int(jc), [])
        kpool = cell_to_replay.get(int(kc), [])
        if not jpool or not kpool:
            continue
        j_idx = int(jpool[rng.integers(len(jpool))])
        k_idx = int(kpool[rng.integers(len(kpool))])
        anchors.append(a_idx)
        closes.append(j_idx)
        fars.append(k_idx)
        d_close.append(dj)
        d_far.append(dk)
    if not anchors:
        return _empty_triplets()
    return (np.asarray(anchors, np.int64), np.asarray(closes, np.int64), np.asarray(fars, np.int64),
            np.asarray(d_close, np.float32), np.asarray(d_far, np.float32))


# =====================================================================
# Training and evaluation
# =====================================================================


def head_variance_loss(z: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    std = torch.sqrt(torch.var(z, dim=0) + 1e-4)
    return F.relu(target_std - std).mean()


@torch.no_grad()
def compute_head_embeddings(head: LatentMetricHead, features: np.ndarray, device: torch.device, batch_size: int = 512) -> np.ndarray:
    head.eval()
    out = []
    for st in range(0, len(features), batch_size):
        en = min(st + batch_size, len(features))
        x = torch.tensor(features[st:en], dtype=torch.float32, device=device)
        z = head(x)
        out.append(z.cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32)


def evaluate_embedding_geometry(emb: np.ndarray, pos: np.ndarray, geodesic, n_pairs: int, rng: np.random.Generator):
    N = len(emb)
    ii = rng.integers(0, N, size=n_pairs)
    jj = rng.integers(0, N, size=n_pairs)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]
    d_lat = np.linalg.norm(emb[ii] - emb[jj], axis=-1)
    ci = _positions_to_cell_indices(geodesic, pos[ii])
    cj = _positions_to_cell_indices(geodesic, pos[jj])
    d_geo = np.asarray([geodesic.dist_matrix[int(a), int(b)] for a, b in zip(ci, cj)], dtype=np.float32)
    finite = np.isfinite(d_geo) & (d_geo > 0)
    if finite.sum() < 10:
        return {"spearman": 0.0, "mean_dist": 0.0}
    return {
        "spearman": _spearman_rho_numpy(d_lat[finite].astype(np.float64), d_geo[finite].astype(np.float64)),
        "mean_dist": float(np.mean(d_lat[finite]))
    }


def evaluate_room_discovery(emb: np.ndarray, pos: np.ndarray, geodesic, knn_k: int):
    cell_idx = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free
    D = emb.shape[1]
    cell_feats = np.zeros((n_free, D), dtype=np.float32)
    counts = np.zeros(n_free, dtype=np.int64)
    for feat, c in zip(emb, cell_idx):
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
    seen = set()
    for i in range(n_cells):
        for j_local in knn[i]:
            j = int(j_local)
            a, b = (i, j) if i < j else (j, i)
            if (a, b) not in seen:
                seen.add((a, b))
                adj_lat[i].append(j)
                adj_lat[j].append(i)
    bridges_lat = _find_bridges(adj_lat)
    comp_lat, n_rooms_lat = _components_without_bridges(adj_lat, bridges_lat)
    adj_oracle = _adj_from_distmat(geodesic.dist_matrix)
    bridges_oracle = _find_bridges(adj_oracle)
    comp_oracle, n_rooms_oracle = _components_without_bridges(adj_oracle, bridges_oracle)
    comp_lat_full = np.full(n_free, -1, dtype=np.int64)
    for loc, cg in enumerate(idx_cells):
        comp_lat_full[int(cg)] = comp_lat[loc]
    agree = 0
    total = 0
    for i in range(len(idx_cells)):
        for j in range(i + 1, len(idx_cells)):
            ci, cj = idx_cells[i], idx_cells[j]
            same_oracle = comp_oracle[ci] == comp_oracle[cj]
            same_latent = comp_lat_full[ci] == comp_lat_full[cj]
            agree += int(same_oracle == same_latent)
            total += 1
    return {
        "n_rooms_latent": int(n_rooms_lat),
        "n_rooms_oracle": int(n_rooms_oracle),
        "pair_agreement": float(agree / max(total, 1)),
    }


def plot_history(history: List[dict], out_path: str, mode: str):
    if not history:
        return
    steps = [h["step"] for h in history]
    rho_head = [h["rho_head"] for h in history]
    rank = [h["rank_loss"] for h in history]
    rooms = [h["rooms_head"] for h in history]
    pair = [h["pair_head"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].plot(steps, rho_head)
    axes[0, 0].set_title("Head vs geodesic Spearman")
    axes[0, 1].plot(steps, rank)
    axes[0, 1].set_title("Head ranking loss")
    axes[1, 0].plot(steps, rooms)
    axes[1, 0].set_title("Discovered rooms")
    axes[1, 1].plot(steps, pair)
    axes[1, 1].set_title("Room pair agreement")
    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("step")
    fig.suptitle(f"Replay-graph latent head — mode={mode}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =====================================================================
# Main
# =====================================================================


def run(cfg: Cfg):
    device = get_device()
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    cfg_train = TrainCfg()
    if cfg.quick:
        cfg.total_steps = 500
        cfg.collect_episodes = 20
        cfg.eval_interval = 100
        cfg.eval_n_pairs = 500
        cfg.pos_probe_epochs = 120

    os.makedirs(cfg.output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Mode: {cfg.mode}")
    print("Train target: latent_head")
    print(f"World model: {cfg.wm_path}")
    print(f"Total steps: {cfg.total_steps}")
    print(f"Output: {cfg.output_dir}")

    print("\n  Loading world model ...")
    ckpt = torch.load(cfg.wm_path, map_location=device, weights_only=False)
    env = PointMazeLargeDiverseGRWrapper(env_name="PointMaze_Large_Diverse_GR-v3", img_size=cfg_train.img_size)
    act_dim = env.action_space.shape[0]

    encoder = ConvEncoder(cfg_train.embed_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    decoder = ConvDecoder(cfg_train.deter_dim, cfg_train.stoch_dim, embedding_size=cfg_train.embed_dim).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    rssm = RSSM(cfg_train.stoch_dim, cfg_train.deter_dim, act_dim, cfg_train.embed_dim, cfg_train.hidden_dim).to(device)
    rssm.load_state_dict(ckpt["rssm"])
    reward_model = RewardModel(cfg_train.deter_dim, cfg_train.stoch_dim, cfg_train.hidden_dim).to(device)
    if "reward_model" in ckpt:
        reward_model.load_state_dict(ckpt["reward_model"])
    cont_model = ContinueModel(cfg_train.deter_dim, cfg_train.stoch_dim, cfg_train.hidden_dim).to(device)
    if "cont_model" in ckpt:
        cont_model.load_state_dict(ckpt["cont_model"])
    actor = None
    if "actor" in ckpt:
        actor = Actor(cfg_train.deter_dim, cfg_train.stoch_dim, act_dim, cfg_train.actor_hidden_dim).to(device)
        actor.load_state_dict(ckpt["actor"])

    models = {
        "encoder": encoder,
        "decoder": decoder,
        "rssm": rssm,
        "reward_model": reward_model,
        "cont_model": cont_model,
        "actor": actor,
    }
    print(f"  Encoder: {sum(p.numel() for p in encoder.parameters()):,} params")
    print(f"  Maze: PointMaze_Large_Diverse_GR-v3  free_cells={env.geodesic.n_free}")

    print("\n  [1] Collecting replay data ...")
    data = collect_replay_data(env, models, cfg_train, device, cfg.collect_episodes)
    pos = data["pos"]
    h = data["h"]
    s = data["s"]
    encoder_e = data["encoder_e"]
    episode_ids = data["episode_ids"]
    raw_obs = data["raw_obs"]
    N = len(pos)
    print(f"    {N} replay states, {cfg.collect_episodes} episodes")

    feature_bank = {
        "encoder_e": encoder_e.astype(np.float32),
        "h": h.astype(np.float32),
        "s": s.astype(np.float32),
        "h+s": np.concatenate([h, s], axis=-1).astype(np.float32),
    }
    base_feature = feature_bank[cfg.head_feature]

    # Baseline on frozen encoder_e for reference.
    baseline = evaluate_embedding_geometry(encoder_e, pos, env.geodesic, cfg.eval_n_pairs, rng)
    baseline_rooms = evaluate_room_discovery(encoder_e, pos, env.geodesic, cfg.room_knn_k)
    print(f"    Baseline: ρ = {baseline['spearman']:.4f}  mean_d = {baseline['mean_dist']:.4f}  rooms = {baseline_rooms['n_rooms_latent']}/{baseline_rooms['n_rooms_oracle']}")

    cell_idx_true = _positions_to_cell_indices(env.geodesic, pos)

    if cfg.mode == "oracle":
        print("\n  [2] Mode: ORACLE — using true geodesic distances")
        dist_matrix = env.geodesic.dist_matrix.astype(np.float32)
        sample_fn = lambda rr, n: sample_triplets_geodesic(cell_idx_true, dist_matrix, rr, n, cfg.close_max_geodesic, cfg.far_min_geodesic, cfg)
    elif cfg.mode == "pos_boot":
        print("\n  [2] Mode: POSITION BOOTSTRAP — training position probe ...")
        probe, feat_name = train_position_probe(feature_bank, pos, cfg, device)
        with torch.no_grad():
            pred_pos = probe(torch.tensor(feature_bank[feat_name], dtype=torch.float32, device=device)).cpu().numpy().astype(np.float32)
        cell_idx_pred = _positions_to_cell_indices(env.geodesic, pred_pos)
        dist_matrix = env.geodesic.dist_matrix.astype(np.float32)
        sample_fn = lambda rr, n: sample_triplets_geodesic(cell_idx_pred, dist_matrix, rr, n, cfg.close_max_geodesic, cfg.far_min_geodesic, cfg)
        cell_acc = float(np.mean(cell_idx_pred == cell_idx_true))
        pos_err = float(np.mean(np.linalg.norm(pred_pos - pos, axis=-1)))
        print(f"    Cell prediction accuracy: {cell_acc:.2%} (exact cell match)")
        print(f"    Mean position error: {pos_err:.4f}")
    elif cfg.mode == "replay":
        print("\n  [2] Mode: REPLAY — using replay-graph shortest paths (oracle-free)")
        idx_global, d_graph, close_thr, far_thr = build_replay_graph_teacher(base_feature, episode_ids, cfg)
        sample_fn = lambda rr, n: sample_triplets_from_graph(d_graph, idx_global, rr, n, close_thr, far_thr, cfg)
    else:
        raise ValueError(f"Unknown mode: {cfg.mode}")

    test_trip = sample_fn(rng, max(100, cfg.batch_triplets // 2))
    print(f"    Test triplets available: {len(test_trip[0])}")

    print("\n  [3] Auxiliary latent-head mode ...")
    print(f"    Frozen backbone; training separate head on feature: {cfg.head_feature}")
    print(f"    Feature dim: {base_feature.shape[1]}  head dim: {cfg.head_dim}")

    head = LatentMetricHead(base_feature.shape[1], cfg.head_dim, cfg.head_hidden).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.head_lr, weight_decay=cfg.weight_decay)
    base_feature_t = torch.tensor(base_feature, dtype=torch.float32, device=device)

    history = []
    running_rank = 0.0
    print(f"\n  [4] Training latent head for {cfg.total_steps} steps (mode={cfg.mode}) ...")
    for step in range(1, cfg.total_steps + 1):
        anchors, closes, fars, d_close, d_far = sample_fn(rng, cfg.batch_triplets)
        trained = len(anchors) >= 4
        if trained:
            a = torch.tensor(anchors, dtype=torch.long, device=device)
            c = torch.tensor(closes, dtype=torch.long, device=device)
            f = torch.tensor(fars, dtype=torch.long, device=device)
            dc = torch.tensor(d_close, dtype=torch.float32, device=device)
            df = torch.tensor(d_far, dtype=torch.float32, device=device)
            all_idx = torch.cat([a, c, f])
            unique_idx, inv = torch.unique(all_idx, return_inverse=True)
            z = head(base_feature_t[unique_idx])
            n = len(a)
            za = z[inv[:n]]
            zc = z[inv[n:2*n]]
            zf = z[inv[2*n:]]
            ratio = df / torch.clamp(dc, min=0.5)
            scaled_margin = cfg.margin * torch.log1p(ratio)
            dpos = torch.norm(za - zc, dim=-1)
            dneg = torch.norm(za - zf, dim=-1)
            l_rank = F.relu(dpos - dneg + scaled_margin).mean()
            l_var = head_variance_loss(z, target_std=cfg.target_std)
            loss = l_rank + cfg.var_lambda * l_var
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), cfg.grad_clip)
            opt.step()
            running_rank += float(l_rank.item())

            if step % max(1, cfg.total_steps // 20) == 0:
                avg_rank = running_rank / step
                print(f"    step {step:5d}/{cfg.total_steps}  L_rank_head={l_rank.item():.4f}  L_var={l_var.item():.4f}  L_total={loss.item():.4f}  (avg_rank={avg_rank:.4f})")

        if step % cfg.eval_interval == 0 or step == cfg.total_steps:
            head_emb = compute_head_embeddings(head, base_feature, device)
            enc_eval = evaluate_embedding_geometry(encoder_e, pos, env.geodesic, cfg.eval_n_pairs, rng)
            head_eval = evaluate_embedding_geometry(head_emb, pos, env.geodesic, cfg.eval_n_pairs, rng)
            enc_rooms = evaluate_room_discovery(encoder_e, pos, env.geodesic, cfg.room_knn_k)
            head_rooms = evaluate_room_discovery(head_emb, pos, env.geodesic, cfg.room_knn_k)
            history.append({
                "step": step,
                "rho_enc": enc_eval["spearman"],
                "rho_head": head_eval["spearman"],
                "rank_loss": float(l_rank.item()) if trained else float("nan"),
                "rooms_enc": enc_rooms["n_rooms_latent"],
                "rooms_head": head_rooms["n_rooms_latent"],
                "pair_enc": enc_rooms["pair_agreement"],
                "pair_head": head_rooms["pair_agreement"],
            })
            note = "" if trained else "  [no triplet step]"
            print(
                f"    ── eval step {step}: ρ_enc={enc_eval['spearman']:.4f}  ρ_head={head_eval['spearman']:.4f}  "
                f"rooms_enc={enc_rooms['n_rooms_latent']}/{enc_rooms['n_rooms_oracle']}  "
                f"rooms_head={head_rooms['n_rooms_latent']}/{head_rooms['n_rooms_oracle']}  "
                f"pair_agree_enc={enc_rooms['pair_agreement']:.3f}  "
                f"pair_agree_head={head_rooms['pair_agreement']:.3f}  "
                f"mean_d_enc={enc_eval['mean_dist']:.4f}{note}"
            )

    print("\n  [5] Final evaluation ...")
    head_emb = compute_head_embeddings(head, base_feature, device)
    enc_eval = evaluate_embedding_geometry(encoder_e, pos, env.geodesic, cfg.eval_n_pairs, rng)
    head_eval = evaluate_embedding_geometry(head_emb, pos, env.geodesic, cfg.eval_n_pairs, rng)
    enc_rooms = evaluate_room_discovery(encoder_e, pos, env.geodesic, cfg.room_knn_k)
    head_rooms = evaluate_room_discovery(head_emb, pos, env.geodesic, cfg.room_knn_k)

    plot_history(history, os.path.join(cfg.output_dir, "training_curves.png"), cfg.mode)
    summary = {
        "mode": cfg.mode,
        "seed": cfg.seed,
        "head_feature": cfg.head_feature,
        "baseline": {**baseline, **baseline_rooms},
        "final_encoder": {**enc_eval, **enc_rooms},
        "final_head": {**head_eval, **head_rooms},
        "history": history,
        "config": {k: v for k, v in cfg.__dict__.items()},
    }
    with open(os.path.join(cfg.output_dir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    torch.save({"head": head.state_dict(), "head_feature": cfg.head_feature}, os.path.join(cfg.output_dir, "checkpoint.pt"))

    print(f"\n  ── SUMMARY ({cfg.mode}, latent_head) ──")
    print(f"  Baseline:  ρ_enc={baseline['spearman']:.4f}  rooms={baseline_rooms['n_rooms_latent']}/{baseline_rooms['n_rooms_oracle']}  pair_agree={baseline_rooms['pair_agreement']:.3f}")
    print(f"  Final enc: ρ_enc={enc_eval['spearman']:.4f}  mean_d={enc_eval['mean_dist']:.4f}")
    print(f"  Final head:ρ_head={head_eval['spearman']:.4f}")
    print(f"  Final rooms(enc): {enc_rooms['n_rooms_latent']}/{enc_rooms['n_rooms_oracle']}  pair_agree={enc_rooms['pair_agreement']:.3f}")
    print(f"  Final rooms(head): {head_rooms['n_rooms_latent']}/{head_rooms['n_rooms_oracle']}  pair_agree={head_rooms['pair_agreement']:.3f}")
    env.close()


# =====================================================================
# CLI
# =====================================================================


def parse_args():
    p = argparse.ArgumentParser(description="Replay-graph latent-head metric learning")
    p.add_argument("--wm_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output_dir", type=str, default="lmc_v3_results")
    p.add_argument("--mode", type=str, default="replay", choices=["replay", "pos_boot", "oracle"])
    p.add_argument("--quick", action="store_true")
    p.add_argument("--total_steps", type=int, default=3000)
    p.add_argument("--collect_episodes", type=int, default=60)
    p.add_argument("--head_feature", type=str, default="h+s", choices=["encoder_e", "h", "s", "h+s"])
    p.add_argument("--graph_max", type=int, default=1800)
    p.add_argument("--knn_k", type=int, default=8)
    p.add_argument("--margin", type=float, default=1.0)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Cfg(
        seed=args.seed,
        wm_path=args.wm_path,
        output_dir=os.path.join(args.output_dir, f"{args.mode}_latent_head"),
        mode=args.mode,
        quick=args.quick,
        total_steps=args.total_steps,
        collect_episodes=args.collect_episodes,
        head_feature=args.head_feature,
        graph_max=args.graph_max,
        knn_k=args.knn_k,
        margin=args.margin,
    )
    run(cfg)
    print("\nDone.")


if __name__ == "__main__":
    main()
