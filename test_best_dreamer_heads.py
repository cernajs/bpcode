#!/usr/bin/env python3
"""
python test_best_dreamer_heads.py \
  --run_dir runs/dreamer_vicreg_baseline_seed0 \
  --checkpoints world_model_final.pt \
  --collect_episodes 40 \
  --state_stride 2 \
  --graph_knn_k 6
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import types
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.stats import pearsonr, rankdata, spearmanr


from dreamer_vics import PointMazeLargeDreamerWrapper
from geom_head import GeoEncoder, temporal_reachability_loss  
from models import Actor, ConvEncoder, RSSM  
from utils import get_device, preprocess_img, set_seed  


# -----------------------------------------------------------------------------
# Small modules
# -----------------------------------------------------------------------------


def rssm_latent(h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return torch.cat([h, s], dim=-1)


class PairTopologyHead(nn.Module):
    """Pair classifier on raw Dreamer latents for topology-only diagnostics."""

    def __init__(self, z_dim: int, hidden_dim: int = 256):
        super().__init__()
        in_dim = z_dim * 4  # z_i, z_j, |diff|, product
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_i, z_j, (z_i - z_j).abs(), z_i * z_j], dim=-1)
        return self.net(x).squeeze(-1)


@dataclass
class RolloutData:
    obs: np.ndarray
    pos: np.ndarray
    cell_idx: np.ndarray
    episode_id: np.ndarray
    t_in_ep: np.ndarray
    h: np.ndarray
    s: np.ndarray
    z: np.ndarray
    enc_e: np.ndarray
    reward: np.ndarray
    success: np.ndarray


# -----------------------------------------------------------------------------
# Graph helpers
# -----------------------------------------------------------------------------


def _adj_from_weighted_edges(n_nodes: int, edges: list[tuple[int, int, float]]) -> list[list[int]]:
    adj = [[] for _ in range(n_nodes)]
    for u, v, _ in edges:
        if u == v:
            continue
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _connected_components(adj: list[list[int]]) -> tuple[np.ndarray, list[list[int]]]:
    n = len(adj)
    comp = -np.ones(n, dtype=np.int64)
    groups: list[list[int]] = []
    cid = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        q: deque[int] = deque([i])
        comp[i] = cid
        group: list[int] = []
        while q:
            u = q.popleft()
            group.append(u)
            for v in adj[u]:
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        groups.append(group)
        cid += 1
    return comp, groups


def _find_bridges(adj: list[list[int]]) -> list[tuple[int, int]]:
    n = len(adj)
    tin = [-1] * n
    low = [-1] * n
    timer = 0
    bridges: list[tuple[int, int]] = []

    sys.setrecursionlimit(max(10_000, n + 100))

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
                    bridges.append((a, b))

    for i in range(n):
        if tin[i] == -1:
            dfs(i, -1)
    return bridges


def _find_articulation_points(adj: list[list[int]]) -> list[int]:
    n = len(adj)
    tin = [-1] * n
    low = [-1] * n
    timer = 0
    out: set[int] = set()

    sys.setrecursionlimit(max(10_000, n + 100))

    def dfs(v: int, p: int) -> None:
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        children = 0
        for to in adj[v]:
            if to == p:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
            else:
                dfs(to, v)
                low[v] = min(low[v], low[to])
                if p != -1 and low[to] >= tin[v]:
                    out.add(v)
                children += 1
        if p == -1 and children > 1:
            out.add(v)

    for i in range(n):
        if tin[i] == -1:
            dfs(i, -1)
    return sorted(out)


def _components_without_bridges(
    adj: list[list[int]], bridges: list[tuple[int, int]]
) -> tuple[np.ndarray, int]:
    n = len(adj)
    bset = set(bridges)
    comp = -np.ones(n, dtype=np.int64)
    cid = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        q: deque[int] = deque([i])
        comp[i] = cid
        while q:
            u = q.popleft()
            for v in adj[u]:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in bset:
                    continue
                if comp[v] == -1:
                    comp[v] = cid
                    q.append(v)
        cid += 1
    return comp, cid


def _adj_from_geodesic_distmat(dist_mat: np.ndarray, edge_eps: float = 1.45) -> list[list[int]]:
    n = dist_mat.shape[0]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_mat[i, j])
            if np.isfinite(d) and 0.0 < d <= edge_eps:
                adj[i].append(j)
                adj[j].append(i)
    return adj


def _bfs_shortest_path(adj: list[list[int]], src: int, dst: int) -> list[int] | None:
    if src == dst:
        return [src]
    parent: dict[int, int | None] = {src: None}
    q: deque[int] = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if dst not in parent:
        return None
    path: list[int] = []
    cur: int | None = dst
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


# -----------------------------------------------------------------------------
# Stats helpers
# -----------------------------------------------------------------------------


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 4:
        return {"pearson": 0.0, "spearman": 0.0, "n": int(mask.sum())}
    try:
        pr = pearsonr(x[mask], y[mask])[0]
    except Exception:
        pr = 0.0
    try:
        sr = spearmanr(x[mask], y[mask]).correlation
    except Exception:
        sr = 0.0
    pr = 0.0 if pr is None or not np.isfinite(pr) else float(pr)
    sr = 0.0 if sr is None or not np.isfinite(sr) else float(sr)
    return {"pearson": pr, "spearman": sr, "n": int(mask.sum())}


def _fit_scale(d_embed: np.ndarray, d_target: np.ndarray, eps: float = 1e-8) -> float:
    mask = np.isfinite(d_embed) & np.isfinite(d_target) & (d_target > 0)
    if mask.sum() < 4:
        return 1.0
    num = float((d_embed[mask] * d_target[mask]).sum())
    den = float((d_target[mask] * d_target[mask]).sum()) + eps
    return max(num / den, eps)


def _mae_after_scale(d_embed: np.ndarray, d_target: np.ndarray, scale: float) -> float:
    mask = np.isfinite(d_embed) & np.isfinite(d_target)
    if mask.sum() < 1:
        return 0.0
    pred = d_embed[mask] / max(scale, 1e-8)
    return float(np.mean(np.abs(pred - d_target[mask])))


BIN_SPECS: list[tuple[str, Callable[[np.ndarray], np.ndarray]]] = [
    ("short", lambda d: (d > 0) & (d <= 3)),
    ("mid", lambda d: (d > 3) & (d <= 8)),
    ("long", lambda d: d > 8),
]


def _distance_bin_report(d_target: np.ndarray, d_embed: np.ndarray, scale: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, fn in BIN_SPECS:
        mask = fn(d_target) & np.isfinite(d_target) & np.isfinite(d_embed)
        if mask.sum() < 4:
            out[name] = {"n": int(mask.sum())}
            continue
        corr = _safe_corr(d_target[mask], d_embed[mask] / max(scale, 1e-8))
        out[name] = {
            "n": int(mask.sum()),
            "pearson": corr["pearson"],
            "spearman": corr["spearman"],
            "mae": _mae_after_scale(d_embed[mask], d_target[mask], scale),
            "mean_pred": float(np.mean(d_embed[mask] / max(scale, 1e-8))),
            "mean_target": float(np.mean(d_target[mask])),
        }
    return out


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    mask = np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = rankdata(y_score)
    sum_pos = float(ranks[y_true == 1].sum())
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def _balanced_accuracy(y_true: np.ndarray, y_score: np.ndarray, thresh: float = 0.0) -> float:
    pred = (y_score >= thresh).astype(np.int64)
    y_true = y_true.astype(np.int64)
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return 0.0
    tpr = float((pred[pos] == 1).mean())
    tnr = float((pred[neg] == 0).mean())
    return 0.5 * (tpr + tnr)


def _knn_overlap_from_dist(dist_a: np.ndarray, dist_b: np.ndarray, k: int = 10) -> float:
    n = dist_a.shape[0]
    if n <= 2:
        return 0.0
    k = min(k, n - 1)
    overlap = 0.0
    valid = 0
    for i in range(n):
        da = dist_a[i].copy()
        db = dist_b[i].copy()
        da[i] = np.inf
        db[i] = np.inf
        if not np.isfinite(da).any() or not np.isfinite(db).any():
            continue
        ia = np.argsort(da)[:k]
        ib = np.argsort(db)[:k]
        overlap += len(set(ia.tolist()) & set(ib.tolist())) / float(k)
        valid += 1
    return float(overlap / max(valid, 1))


# -----------------------------------------------------------------------------
# Loading / collection
# -----------------------------------------------------------------------------


def load_models(
    ckpt_path: str,
    device: torch.device,
    obs_channels: int,
    act_dim: int,
    embed_dim: int,
    stoch_dim: int,
    deter_dim: int,
    hidden_dim: int,
    actor_hidden_dim: int,
) -> tuple[ConvEncoder, RSSM, Actor | None]:
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    encoder = ConvEncoder(embedding_size=embed_dim, in_channels=obs_channels).to(device)
    rssm = RSSM(stoch_dim, deter_dim, act_dim, embed_dim, hidden_dim).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    rssm.load_state_dict(ckpt["rssm"])
    encoder.eval()
    rssm.eval()

    actor = None
    if "actor" in ckpt:
        actor = Actor(deter_dim, stoch_dim, act_dim, actor_hidden_dim).to(device)
        actor.load_state_dict(ckpt["actor"])
        actor.eval()
    return encoder, rssm, actor


@torch.no_grad()
def _encode_reset_obs(
    obs_uint8: np.ndarray,
    encoder: ConvEncoder,
    rssm: RSSM,
    device: torch.device,
    bit_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = torch.tensor(np.ascontiguousarray(obs_uint8), dtype=torch.float32, device=device)
    if x.dim() == 3:
        x = x.permute(2, 0, 1).unsqueeze(0)
    preprocess_img(x, depth=bit_depth)
    e = encoder(x)
    h, s = rssm.get_init_state(e, mean=True)
    z = rssm_latent(h, s)
    return (
        e.squeeze(0).detach().cpu().numpy(),
        h.squeeze(0).detach().cpu().numpy(),
        s.squeeze(0).detach().cpu().numpy(),
        z.squeeze(0).detach().cpu().numpy(),
    )


def _pick_goal_cell(geodesic, r: int, c: int) -> tuple[int, int]:
    for gr, gc in geodesic.idx_to_cell:
        if (gr, gc) != (r, c):
            return int(gr), int(gc)
    return int(r), int(c)


def collect_obs_at_cells(env: PointMazeLargeDreamerWrapper, cell_indices: list[int], geodesic) -> dict[int, np.ndarray]:
    cache: dict[int, np.ndarray] = {}
    for idx in cell_indices:
        r, c = geodesic.idx_to_cell[idx]
        gr, gc = _pick_goal_cell(geodesic, int(r), int(c))
        obs_dict, _ = env._env.reset(
            options={
                "reset_cell": np.array([r, c], dtype=np.int64),
                "goal_cell": np.array([gr, gc], dtype=np.int64),
            }
        )
        env._update_agent_pos(obs_dict)
        frame = env._resize_frame(env._env.render()).astype(np.uint8)
        cache[int(idx)] = np.ascontiguousarray(frame, dtype=np.uint8)
    return cache


@torch.no_grad()
def collect_rollout_data(
    env: PointMazeLargeDreamerWrapper,
    encoder: ConvEncoder,
    rssm: RSSM,
    actor: Actor | None,
    device: torch.device,
    bit_depth: int,
    collect_episodes: int,
    state_stride: int,
    max_nodes: int,
    deterministic_policy: bool,
    expl_noise: float,
    random_policy: bool,
) -> RolloutData:
    geodesic = env.geodesic

    obs_list: list[np.ndarray] = []
    pos_list: list[np.ndarray] = []
    cell_list: list[int] = []
    ep_list: list[int] = []
    t_list: list[int] = []
    h_list: list[np.ndarray] = []
    s_list: list[np.ndarray] = []
    z_list: list[np.ndarray] = []
    e_list: list[np.ndarray] = []
    r_list: list[float] = []
    succ_list: list[float] = []

    total_kept = 0

    for ep in range(int(collect_episodes)):
        obs, _ = env.reset()
        x = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(x, depth=bit_depth)
        e = encoder(x)
        h, s = rssm.get_init_state(e, mean=True)

        done = False
        t_env = 0
        while not done:
            keep = (t_env % max(1, state_stride) == 0)
            if keep:
                pos = env.agent_pos.copy().astype(np.float32)
                cell = int(geodesic.cell_to_idx[geodesic.pos_to_cell(float(pos[0]), float(pos[1]))])
                obs_list.append(np.ascontiguousarray(obs, dtype=np.uint8))
                pos_list.append(pos)
                cell_list.append(cell)
                ep_list.append(ep)
                t_list.append(t_env)
                h_np = h.squeeze(0).detach().cpu().numpy().astype(np.float32)
                s_np = s.squeeze(0).detach().cpu().numpy().astype(np.float32)
                e_np = e.squeeze(0).detach().cpu().numpy().astype(np.float32)
                z_np = np.concatenate([h_np, s_np], axis=-1).astype(np.float32)
                h_list.append(h_np)
                s_list.append(s_np)
                e_list.append(e_np)
                z_list.append(z_np)
                r_list.append(0.0)
                succ_list.append(0.0)
                total_kept += 1
                if total_kept >= max_nodes:
                    done = True
                    break

            if random_policy or actor is None:
                action = env.action_space.sample().astype(np.float32)
                a_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            else:
                a_t, _ = actor.get_action(h, s, deterministic=deterministic_policy)
                if expl_noise > 0:
                    a_t = torch.clamp(a_t + expl_noise * torch.randn_like(a_t), -1.0, 1.0)
                action = a_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

            next_obs, r, term, trunc, info = env.step(action, repeat=1)
            done = bool(term or trunc)
            t_env += 1

            x = torch.tensor(np.ascontiguousarray(next_obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(x, depth=bit_depth)
            e = encoder(x)
            h, s, _, _ = rssm.observe_step(e, a_t, h, s, sample=False)
            obs = next_obs

            if keep and total_kept > 0:
                r_list[-1] = float(r)
                succ_list[-1] = 1.0 if info.get("success", False) or info.get("is_success", False) else 0.0

        if total_kept >= max_nodes:
            break

    if not obs_list:
        raise RuntimeError("No rollout states collected.")

    return RolloutData(
        obs=np.stack(obs_list, axis=0),
        pos=np.stack(pos_list, axis=0),
        cell_idx=np.asarray(cell_list, dtype=np.int64),
        episode_id=np.asarray(ep_list, dtype=np.int64),
        t_in_ep=np.asarray(t_list, dtype=np.int64),
        h=np.stack(h_list, axis=0),
        s=np.stack(s_list, axis=0),
        z=np.stack(z_list, axis=0),
        enc_e=np.stack(e_list, axis=0),
        reward=np.asarray(r_list, dtype=np.float32),
        success=np.asarray(succ_list, dtype=np.float32),
    )


def _concat_rollout_data(chunks: list[RolloutData]) -> RolloutData:
    if not chunks:
        raise RuntimeError("No rollout chunks to concatenate.")
    obs = []
    pos = []
    cell_idx = []
    episode_id = []
    t_in_ep = []
    h = []
    s = []
    z = []
    enc_e = []
    reward = []
    success = []
    ep_offset = 0
    for chunk in chunks:
        obs.append(chunk.obs)
        pos.append(chunk.pos)
        cell_idx.append(chunk.cell_idx)
        episode_id.append(chunk.episode_id + ep_offset)
        t_in_ep.append(chunk.t_in_ep)
        h.append(chunk.h)
        s.append(chunk.s)
        z.append(chunk.z)
        enc_e.append(chunk.enc_e)
        reward.append(chunk.reward)
        success.append(chunk.success)
        if chunk.episode_id.size:
            ep_offset += int(chunk.episode_id.max()) + 1
    return RolloutData(
        obs=np.concatenate(obs, axis=0),
        pos=np.concatenate(pos, axis=0),
        cell_idx=np.concatenate(cell_idx, axis=0),
        episode_id=np.concatenate(episode_id, axis=0),
        t_in_ep=np.concatenate(t_in_ep, axis=0),
        h=np.concatenate(h, axis=0),
        s=np.concatenate(s, axis=0),
        z=np.concatenate(z, axis=0),
        enc_e=np.concatenate(enc_e, axis=0),
        reward=np.concatenate(reward, axis=0),
        success=np.concatenate(success, axis=0),
    )


def _teacher_start_subset(env: PointMazeLargeDreamerWrapper, size: int) -> list[tuple[int, int]]:
    size = max(1, min(int(size), len(env._free_cells)))
    return [tuple(env._free_cells[i]) for i in range(size)]


def _set_env_mode(
    env: PointMazeLargeDreamerWrapper,
    mode: str,
    subset_size: int,
) -> dict[str, Any]:
    old = {
        "reset_mode": env.reset_mode,
        "fixed_start_cell": env.fixed_start_cell,
        "start_cells": list(env.start_cells),
    }
    if mode == "random_reset":
        env.reset_mode = "random"
    elif mode == "start_subset":
        env.reset_mode = "start_subset"
        env.start_cells = _teacher_start_subset(env, subset_size)
    elif mode == "fixed_start":
        env.reset_mode = "fixed_start"
    else:
        raise ValueError(f"Unknown teacher collection mode {mode!r}")
    return old


def _restore_env_mode(env: PointMazeLargeDreamerWrapper, old: dict[str, Any]) -> None:
    env.reset_mode = old["reset_mode"]
    env.fixed_start_cell = old["fixed_start_cell"]
    env.start_cells = list(old["start_cells"])


@torch.no_grad()
def collect_teacher_data_with_quotas(
    env: PointMazeLargeDreamerWrapper,
    encoder: ConvEncoder,
    rssm: RSSM,
    actor: Actor | None,
    device: torch.device,
    bit_depth: int,
    teacher_collect_episodes: int,
    state_stride: int,
    max_nodes: int,
    deterministic_policy: bool,
    expl_noise: float,
    teacher_high_noise: float,
    teacher_random_fraction: float,
    teacher_subset_fraction: float,
    teacher_start_subset_size: int,
) -> tuple[RolloutData, dict[str, int]]:
    frac_random = float(np.clip(teacher_random_fraction, 0.0, 1.0))
    frac_subset = float(np.clip(teacher_subset_fraction, 0.0, 1.0 - frac_random))
    frac_fixed = max(0.0, 1.0 - frac_random - frac_subset)

    quotas = {
        "random_reset": int(round(max_nodes * frac_random)),
        "start_subset": int(round(max_nodes * frac_subset)),
    }
    quotas["fixed_start"] = max(0, int(max_nodes) - quotas["random_reset"] - quotas["start_subset"])

    mode_cfg = {
        "random_reset": {
            "episodes": max(1, int(round(teacher_collect_episodes * max(frac_random, 0.1)))),
            "noise": 0.0,
            "random_policy": True,
        },
        "start_subset": {
            "episodes": max(1, int(round(teacher_collect_episodes * max(frac_subset, 0.1)))),
            "noise": float(teacher_high_noise),
            "random_policy": actor is None,
        },
        "fixed_start": {
            "episodes": max(1, int(round(teacher_collect_episodes * max(frac_fixed, 0.1)))),
            "noise": float(expl_noise),
            "random_policy": actor is None,
        },
    }

    chunks: list[RolloutData] = []
    mode_hist: dict[str, int] = {}
    for mode in ["random_reset", "start_subset", "fixed_start"]:
        quota = int(quotas[mode])
        if quota <= 0:
            continue
        old = _set_env_mode(env, mode, teacher_start_subset_size)
        try:
            chunk = collect_rollout_data(
                env=env,
                encoder=encoder,
                rssm=rssm,
                actor=actor,
                device=device,
                bit_depth=bit_depth,
                collect_episodes=mode_cfg[mode]["episodes"],
                state_stride=state_stride,
                max_nodes=quota,
                deterministic_policy=deterministic_policy,
                expl_noise=float(mode_cfg[mode]["noise"]),
                random_policy=bool(mode_cfg[mode]["random_policy"]),
            )
        finally:
            _restore_env_mode(env, old)
        chunks.append(chunk)
        mode_hist[mode] = int(chunk.z.shape[0])

    total = sum(mode_hist.values())
    if total < max_nodes and chunks:
        # top up from fixed-start / noisy actor if rounding left slack.
        rem = int(max_nodes - total)
        old = _set_env_mode(env, "fixed_start", teacher_start_subset_size)
        try:
            chunk = collect_rollout_data(
                env=env,
                encoder=encoder,
                rssm=rssm,
                actor=actor,
                device=device,
                bit_depth=bit_depth,
                collect_episodes=max(1, teacher_collect_episodes // 3),
                state_stride=state_stride,
                max_nodes=rem,
                deterministic_policy=deterministic_policy,
                expl_noise=float(expl_noise),
                random_policy=actor is None,
            )
        finally:
            _restore_env_mode(env, old)
        chunks.append(chunk)
        mode_hist["fixed_start"] = mode_hist.get("fixed_start", 0) + int(chunk.z.shape[0])

    return _concat_rollout_data(chunks), mode_hist


# -----------------------------------------------------------------------------
# Replay graph construction
# -----------------------------------------------------------------------------


def _standardize_features(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


@dataclass
class ReplayGraph:
    node_indices: np.ndarray
    z_basis: np.ndarray
    edges: list[tuple[int, int, float]]
    dist_mat: np.ndarray
    adj: list[list[int]]
    comp_id: np.ndarray
    giant_nodes_local: np.ndarray
    giant_nodes_global: np.ndarray
    bridges: list[tuple[int, int]]
    articulation: list[int]
    rooms_wo_bridges: np.ndarray
    n_rooms_wo_bridges: int
    graph_stats: dict[str, Any]
    edge_confidence: dict[tuple[int, int], float]
    node_confidence: np.ndarray
    views: list[str]


def build_replay_graph(
    data: RolloutData,
    basis_name: str,
    graph_knn_k: int,
    graph_knn_weight: float,
    graph_temporal_weight: float,
    graph_knn_max_percentile: float,
    graph_same_ep_gap: int,
    max_graph_nodes: int,
    graph_views: list[str] | None = None,
    graph_min_view_votes: int = 1,
) -> ReplayGraph:
    rng = np.random.default_rng(0)
    n_total = data.z.shape[0]
    if n_total > max_graph_nodes:
        keep = np.sort(rng.choice(n_total, size=max_graph_nodes, replace=False))
    else:
        keep = np.arange(n_total, dtype=np.int64)

    views = list(graph_views) if graph_views else [basis_name]
    view_basis: dict[str, np.ndarray] = {}
    for view in views:
        if view == "encoder":
            basis = data.enc_e[keep]
        elif view == "h":
            basis = data.h[keep]
        elif view == "s":
            basis = data.s[keep]
        else:
            basis = data.z[keep]
        view_basis[view] = _standardize_features(basis.astype(np.float32))

    ep = data.episode_id[keep]
    tt = data.t_in_ep[keep]
    n = len(keep)

    edge_weight: dict[tuple[int, int], float] = {}
    edge_conf: dict[tuple[int, int], float] = {}
    edge_views: dict[tuple[int, int], set[str]] = defaultdict(set)
    temporal_edges = 0

    def _register(u: int, v: int, w: float, conf: float, tag: str) -> None:
        if u == v:
            return
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        prev = edge_weight.get((a, b), None)
        if prev is None or w < prev:
            edge_weight[(a, b)] = float(w)
        edge_conf[(a, b)] = max(float(conf), edge_conf.get((a, b), 0.0))
        edge_views[(a, b)].add(str(tag))

    for e_id in np.unique(ep):
        loc = np.where(ep == e_id)[0]
        if len(loc) < 2:
            continue
        order = np.argsort(tt[loc])
        loc = loc[order]
        for a, b in zip(loc[:-1], loc[1:]):
            dt = int(tt[b] - tt[a])
            w = float(max(1, dt) * graph_temporal_weight)
            _register(int(a), int(b), w, 1.0, "temporal")
            temporal_edges += 1

    per_view_knn: dict[str, int] = {}
    for view, basis_std in view_basis.items():
        knn_count = 0
        if graph_knn_k <= 0 or n < 4:
            per_view_knn[view] = 0
            continue
        dmat = np.linalg.norm(basis_std[:, None, :] - basis_std[None, :, :], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        if graph_same_ep_gap > 0:
            same_ep = ep[:, None] == ep[None, :]
            gap = np.abs(tt[:, None] - tt[None, :])
            dmat[same_ep & (gap <= graph_same_ep_gap)] = np.inf
        nbrs = np.argsort(dmat, axis=1)[:, :graph_knn_k]
        mutual: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        cand_d: list[float] = []
        for i in range(n):
            for j in nbrs[i]:
                if not np.isfinite(dmat[i, j]):
                    continue
                if i in nbrs[j]:
                    a, b = (int(i), int(j)) if i < j else (int(j), int(i))
                    if (a, b) not in seen:
                        seen.add((a, b))
                        dij = float(dmat[a, b])
                        cand_d.append(dij)
                        mutual.append((a, b, dij))
        if not cand_d:
            per_view_knn[view] = 0
            continue
        max_d = float(np.percentile(np.asarray(cand_d, dtype=np.float32), graph_knn_max_percentile))
        for a, b, dij in mutual:
            if dij > max_d:
                continue
            sim = 1.0 - min(max(dij / max(max_d, 1e-6), 0.0), 1.0)
            _register(a, b, float(graph_knn_weight * (2.0 - sim)), 0.5 + 0.5 * sim, view)
            knn_count += 1
        per_view_knn[view] = int(knn_count)

    edges: list[tuple[int, int, float]] = []
    for (a, b), w in edge_weight.items():
        votes = len([x for x in edge_views[(a, b)] if x != "temporal"])
        if "temporal" in edge_views[(a, b)] or votes >= max(1, int(graph_min_view_votes)):
            if "temporal" not in edge_views[(a, b)]:
                vote_frac = votes / max(len(views), 1)
                conf = max(edge_conf[(a, b)], vote_frac)
                w = float(w * (1.5 - 0.5 * conf))
                edge_conf[(a, b)] = float(conf)
            edges.append((a, b, float(w)))

    mat = lil_matrix((n, n), dtype=np.float32)
    for u, v, w in edges:
        if u == v:
            continue
        prev = mat[u, v]
        if prev == 0 or float(prev) > float(w):
            mat[u, v] = float(w)
            mat[v, u] = float(w)

    dist = shortest_path(mat.tocsr(), method="D", directed=False, unweighted=False)
    adj = _adj_from_weighted_edges(n, edges)
    comp_id, groups = _connected_components(adj)
    giant = max(groups, key=len) if groups else []
    giant_local = np.asarray(sorted(giant), dtype=np.int64)
    giant_global = keep[giant_local] if len(giant_local) else np.zeros((0,), dtype=np.int64)

    bridges = _find_bridges(adj)
    articulation = _find_articulation_points(adj)
    rooms, n_rooms = _components_without_bridges(adj, bridges)

    node_conf = np.zeros((n,), dtype=np.float32)
    deg = np.zeros((n,), dtype=np.int64)
    for (a, b), c in edge_conf.items():
        if (a, b) not in {(min(u,v), max(u,v)) for u,v,_ in edges}:
            continue
        node_conf[a] += float(c)
        node_conf[b] += float(c)
        deg[a] += 1
        deg[b] += 1
    node_conf = np.divide(node_conf, np.maximum(deg, 1), dtype=np.float32)

    stats = {
        "n_nodes_total": int(n),
        "n_edges_total": int(mat.nnz // 2),
        "n_temporal_edges": int(temporal_edges),
        "n_knn_edges_by_view": {k: int(v) for k, v in per_view_knn.items()},
        "n_components": int(len(groups)),
        "giant_component_size": int(len(giant_local)),
        "giant_component_fraction": float(len(giant_local) / max(n, 1)),
        "n_bridges": int(len(bridges)),
        "n_articulation": int(len(articulation)),
        "n_rooms_wo_bridges": int(n_rooms),
        "views": views,
        "edge_conf_mean": float(np.mean(list(edge_conf.values()))) if edge_conf else 0.0,
        "edge_conf_p90": float(np.percentile(list(edge_conf.values()), 90)) if edge_conf else 0.0,
    }
    return ReplayGraph(
        node_indices=keep,
        z_basis=view_basis.get(views[0], np.zeros((n, 1), dtype=np.float32)),
        edges=edges,
        dist_mat=np.asarray(dist, dtype=np.float32),
        adj=adj,
        comp_id=comp_id,
        giant_nodes_local=giant_local,
        giant_nodes_global=giant_global,
        bridges=bridges,
        articulation=articulation,
        rooms_wo_bridges=rooms,
        n_rooms_wo_bridges=n_rooms,
        graph_stats=stats,
        edge_confidence={k: float(v) for k, v in edge_conf.items()},
        node_confidence=node_conf.astype(np.float32),
        views=views,
    )


# -----------------------------------------------------------------------------
# Pair / sequence sampling
# -----------------------------------------------------------------------------


def _sample_pairs_by_bins(
    dist_mat: np.ndarray,
    node_ids: np.ndarray,
    n_pairs: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    local_nodes = np.asarray(node_ids, dtype=np.int64)
    if len(local_nodes) < 2:
        raise RuntimeError("Not enough nodes to sample pairs.")

    bins: dict[str, list[tuple[int, int, float]]] = {"short": [], "mid": [], "long": []}
    max_scan = min(80_000, len(local_nodes) * len(local_nodes))
    for _ in range(max_scan):
        i = int(rng.choice(local_nodes))
        j = int(rng.choice(local_nodes))
        if i == j:
            continue
        d = float(dist_mat[i, j])
        if not np.isfinite(d) or d <= 0:
            continue
        if d <= 3:
            bins["short"].append((i, j, d))
        elif d <= 8:
            bins["mid"].append((i, j, d))
        else:
            bins["long"].append((i, j, d))
        if sum(len(v) for v in bins.values()) >= max(4 * n_pairs, 2000):
            break

    out: list[tuple[int, int, float]] = []
    per_bin = max(1, n_pairs // 3)
    for name in ["short", "mid", "long"]:
        pool = bins[name]
        if not pool:
            continue
        choose = min(per_bin, len(pool))
        idx = rng.choice(len(pool), size=choose, replace=False)
        out.extend([pool[k] for k in idx])
    if len(out) < n_pairs:
        flat = bins["short"] + bins["mid"] + bins["long"]
        if not flat:
            raise RuntimeError("Could not sample any finite-distance pairs.")
        need = min(n_pairs - len(out), len(flat))
        idx = rng.choice(len(flat), size=need, replace=False)
        out.extend([flat[k] for k in idx])

    ii = np.asarray([x[0] for x in out], dtype=np.int64)
    jj = np.asarray([x[1] for x in out], dtype=np.int64)
    dd = np.asarray([x[2] for x in out], dtype=np.float32)
    return ii, jj, dd


def _episode_sequences(global_ids: np.ndarray, episode_id: np.ndarray, t_in_ep: np.ndarray) -> list[np.ndarray]:
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for gid in global_ids.tolist():
        groups[int(episode_id[gid])].append((int(t_in_ep[gid]), int(gid)))
    out: list[np.ndarray] = []
    for _, arr in groups.items():
        arr.sort(key=lambda x: x[0])
        out.append(np.asarray([x[1] for x in arr], dtype=np.int64))
    return out


# -----------------------------------------------------------------------------
# Training candidate heads
# -----------------------------------------------------------------------------


def train_replay_metric_head(
    h: np.ndarray,
    s: np.ndarray,
    replay_dist: np.ndarray,
    node_ids: np.ndarray,
    episode_id: np.ndarray,
    t_in_ep: np.ndarray,
    geo_dim: int,
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_pairs: int,
    device: torch.device,
    seed: int,
    canon_weight: float = 0.0,
    canon_batch_pairs: int = 128,
    canon_pos_radius: float = 3.0,
    canon_min_gap: int = 8,
) -> tuple[GeoEncoder, float]:
    rng = np.random.default_rng(seed)
    model = GeoEncoder(h.shape[1], s.shape[1], geo_dim=geo_dim, hidden_dim=hidden_dim).to(device)
    scale = torch.nn.Parameter(torch.tensor(0.1, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [scale], lr=lr)

    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    ep = np.asarray(episode_id, dtype=np.int64)
    tt = np.asarray(t_in_ep, dtype=np.int64)

    pos_cands: dict[int, np.ndarray] = {}
    if canon_weight > 0:
        for i in node_ids.tolist():
            mask = np.isfinite(replay_dist[i]) & (replay_dist[i] > 0) & (replay_dist[i] <= float(canon_pos_radius))
            mask[i] = False
            ctx = (ep != ep[i]) | (np.abs(tt - tt[i]) >= int(canon_min_gap))
            cand = np.where(mask & ctx)[0]
            if cand.size:
                pos_cands[int(i)] = cand.astype(np.int64)

    for _ in range(int(epochs)):
        ii, jj, d_np = _sample_pairs_by_bins(replay_dist, node_ids, batch_pairs, int(rng.integers(1 << 31)))
        h_i = h_t[ii]
        s_i = s_t[ii]
        h_j = h_t[jj]
        s_j = s_t[jj]
        g_i = model(h_i, s_i)
        g_j = model(h_j, s_j)
        d_lat = torch.norm(g_i - g_j, dim=-1)
        d_t = torch.tensor(d_np, dtype=torch.float32, device=device)
        w = 1.0 / torch.sqrt(d_t + 1.0)
        loss_metric = (w * (d_lat - scale.clamp(min=1e-4) * d_t).pow(2)).mean()

        g_pair = torch.cat([g_i, g_j], dim=0)
        sq = torch.cdist(g_pair, g_pair).pow(2)
        mask_eye = torch.eye(sq.size(0), device=device, dtype=torch.bool)
        sq = sq.masked_fill(mask_eye, 1e9)
        loss_uni = torch.logsumexp(-2.0 * sq, dim=1).mean()

        loss_canon = torch.zeros((), device=device)
        if canon_weight > 0 and pos_cands:
            anchors = rng.choice(np.asarray(list(pos_cands.keys()), dtype=np.int64), size=min(canon_batch_pairs, len(pos_cands)), replace=False)
            pos_idx = []
            for a in anchors.tolist():
                cand = pos_cands[int(a)]
                pos_idx.append(int(cand[int(rng.integers(len(cand)))]))
            pos_idx = np.asarray(pos_idx, dtype=np.int64)
            g_a = model(h_t[anchors], s_t[anchors])
            g_p = model(h_t[pos_idx], s_t[pos_idx])
            d_ap = torch.norm(g_a - g_p, dim=-1)
            d_teacher = torch.tensor(replay_dist[anchors, pos_idx], dtype=torch.float32, device=device)
            loss_canon = (d_ap - scale.clamp(min=1e-4) * d_teacher).abs().mean() + 0.5 * d_ap.mean()

        total = loss_metric + 0.05 * loss_uni + float(canon_weight) * loss_canon

        opt.zero_grad(set_to_none=True)
        total.backward()
        opt.step()

    model.eval()
    return model, float(scale.detach().cpu().item())


def train_temporal_head(
    h: np.ndarray,
    s: np.ndarray,
    episode_id: np.ndarray,
    t_in_ep: np.ndarray,
    node_ids: np.ndarray,
    geo_dim: int,
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    seed: int,
) -> GeoEncoder:
    rng = np.random.default_rng(seed)
    model = GeoEncoder(h.shape[1], s.shape[1], geo_dim=geo_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    sequences = _episode_sequences(node_ids, episode_id, t_in_ep)
    sequences = [seq for seq in sequences if len(seq) >= max(4, seq_len)]
    if not sequences:
        raise RuntimeError("No sufficiently long sequences for temporal head training.")

    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)

    for _ in range(int(epochs)):
        batch_seqs: list[np.ndarray] = []
        for _ in range(int(batch_size)):
            seq = sequences[int(rng.integers(len(sequences)))]
            if len(seq) == seq_len:
                batch_seqs.append(seq)
            else:
                start = int(rng.integers(0, len(seq) - seq_len + 1))
                batch_seqs.append(seq[start : start + seq_len])
        idx = np.stack(batch_seqs, axis=0)
        h_seq = h_t[idx]
        s_seq = s_t[idx]
        g_seq = model(h_seq, s_seq)
        loss = temporal_reachability_loss(g_seq, pos_k=3, neg_k=min(12, seq_len - 2), margin=0.6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    return model


def train_topology_pair_head(
    z: np.ndarray,
    same_room_labels: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> PairTopologyHead:
    rng = np.random.default_rng(seed)
    model = PairTopologyHead(z.shape[1], hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    z_t = torch.tensor(z, dtype=torch.float32, device=device)
    y = torch.tensor(same_room_labels, dtype=torch.float32, device=device)
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)

    pos_idx = np.where(same_room_labels == 1)[0]
    neg_idx = np.where(same_room_labels == 0)[0]
    if len(pos_idx) < 8 or len(neg_idx) < 8:
        raise RuntimeError("Not enough positive/negative topology pairs.")

    for _ in range(int(epochs)):
        half = max(1, batch_size // 2)
        sel_pos = rng.choice(pos_idx, size=min(half, len(pos_idx)), replace=len(pos_idx) < half)
        sel_neg = rng.choice(neg_idx, size=min(half, len(neg_idx)), replace=len(neg_idx) < half)
        sel = np.concatenate([sel_pos, sel_neg], axis=0)
        rng.shuffle(sel)

        zi = z_t[pair_i[sel]]
        zj = z_t[pair_j[sel]]
        target = y[sel]
        logits = model(zi, zj)
        loss = F.binary_cross_entropy_with_logits(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    return model


# -----------------------------------------------------------------------------
# Evaluation helpers
# -----------------------------------------------------------------------------


def _aggregate_by_cell(values: np.ndarray, cell_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uniq = np.unique(cell_idx)
    means = []
    counts = []
    for c in uniq.tolist():
        mask = cell_idx == c
        means.append(values[mask].mean(axis=0))
        counts.append(int(mask.sum()))
    return uniq.astype(np.int64), np.stack(means, axis=0).astype(np.float32), np.asarray(counts, dtype=np.int64)


@torch.no_grad()
def _embed_with_head(
    name: str,
    h: np.ndarray,
    s: np.ndarray,
    z: np.ndarray,
    head: GeoEncoder | None,
    device: torch.device,
) -> np.ndarray:
    if name == "raw_hs" or head is None:
        return z.astype(np.float32)
    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    g = head(h_t, s_t).detach().cpu().numpy().astype(np.float32)
    return g


@torch.no_grad()
def _embed_reset_cells(
    visited_cells: np.ndarray,
    env: PointMazeLargeDreamerWrapper,
    encoder: ConvEncoder,
    rssm: RSSM,
    device: torch.device,
    bit_depth: int,
    head_name: str,
    head: GeoEncoder | None,
) -> dict[int, np.ndarray]:
    obs_cache = collect_obs_at_cells(env, visited_cells.tolist(), env.geodesic)
    out: dict[int, np.ndarray] = {}
    for cell in visited_cells.tolist():
        e_np, h_np, s_np, z_np = _encode_reset_obs(obs_cache[cell], encoder, rssm, device, bit_depth)
        if head_name == "raw_hs" or head is None:
            out[int(cell)] = z_np.astype(np.float32)
        else:
            h_t = torch.tensor(h_np[None], dtype=torch.float32, device=device)
            s_t = torch.tensor(s_np[None], dtype=torch.float32, device=device)
            out[int(cell)] = head(h_t, s_t).squeeze(0).detach().cpu().numpy().astype(np.float32)
    return out


def _within_cell_variance(emb: np.ndarray, cell_idx: np.ndarray) -> float:
    vals = []
    for c in np.unique(cell_idx).tolist():
        x = emb[cell_idx == c]
        if len(x) < 2:
            continue
        mu = x.mean(axis=0, keepdims=True)
        vals.append(float(np.mean(np.linalg.norm(x - mu, axis=-1))))
    return float(np.mean(vals)) if vals else 0.0


def _rollout_reset_gap(
    cell_ids: np.ndarray,
    cell_means: np.ndarray,
    reset_map: dict[int, np.ndarray],
) -> float:
    vals = []
    for cid, mu in zip(cell_ids.tolist(), cell_means):
        if int(cid) not in reset_map:
            continue
        vals.append(float(np.linalg.norm(mu - reset_map[int(cid)])))
    return float(np.mean(vals)) if vals else 0.0


@dataclass
class OracleTopo:
    adj: list[list[int]]
    bridges: list[tuple[int, int]]
    rooms: np.ndarray
    n_rooms: int


def _make_oracle_topology(geodesic) -> OracleTopo:
    dist_mat = np.asarray(geodesic.dist_matrix, dtype=np.float32)
    adj = _adj_from_geodesic_distmat(dist_mat)
    bridges = _find_bridges(adj)
    rooms, n_rooms = _components_without_bridges(adj, bridges)
    return OracleTopo(adj=adj, bridges=bridges, rooms=rooms, n_rooms=n_rooms)


def _evaluate_embedding(
    name: str,
    emb_nodes: np.ndarray,
    replay_graph: ReplayGraph,
    data: RolloutData,
    geodesic,
    oracle_topo: OracleTopo,
    scale_for_distance: float,
    pair_eval_i_local: np.ndarray,
    pair_eval_j_local: np.ndarray,
    replay_eval_d: np.ndarray,
    device: torch.device,
    env: PointMazeLargeDreamerWrapper,
    encoder: ConvEncoder,
    rssm: RSSM,
    bit_depth: int,
    head: GeoEncoder | None,
) -> dict[str, Any]:
    node_global = replay_graph.node_indices
    emb = emb_nodes.astype(np.float32)
    d_embed = np.linalg.norm(emb[pair_eval_i_local] - emb[pair_eval_j_local], axis=-1)
    replay_corr = _safe_corr(replay_eval_d, d_embed / max(scale_for_distance, 1e-8))
    replay_bins = _distance_bin_report(replay_eval_d, d_embed, scale_for_distance)

    # Local neighborhood overlap wrt replay graph on giant component subset.
    gc = replay_graph.giant_nodes_local
    if len(gc) >= 8:
        sub = gc[: min(len(gc), 256)]
        dist_embed = np.linalg.norm(emb[sub][:, None, :] - emb[sub][None, :, :], axis=-1) / max(scale_for_distance, 1e-8)
        dist_replay = replay_graph.dist_mat[np.ix_(sub, sub)]
        replay_knn = _knn_overlap_from_dist(dist_embed, dist_replay, k=10)
    else:
        replay_knn = 0.0

    # Temporal edge vs far pair separation.
    edge_pairs = []
    for u, v, _ in replay_graph.edges:
        if u in gc and v in gc:
            edge_pairs.append((u, v))
    far_i, far_j, far_d = _sample_pairs_by_bins(replay_graph.dist_mat, gc, max(128, len(edge_pairs)), seed=11)
    far_mask = far_d > 8
    far_i = far_i[far_mask]
    far_j = far_j[far_mask]
    n_cmp = min(len(edge_pairs), len(far_i))
    if n_cmp > 0:
        pos = np.asarray([np.linalg.norm(emb[u] - emb[v]) for (u, v) in edge_pairs[:n_cmp]], dtype=np.float32)
        neg = np.asarray([np.linalg.norm(emb[int(i)] - emb[int(j)]) for i, j in zip(far_i[:n_cmp], far_j[:n_cmp])], dtype=np.float32)
        y = np.concatenate([np.ones_like(pos), np.zeros_like(neg)]).astype(np.int64)
        score = np.concatenate([-pos, -neg])  # smaller distance => more likely edge
        edge_far_auc = _binary_auc(y, score)
        edge_mean = float(pos.mean())
        far_mean = float(neg.mean())
    else:
        edge_far_auc = 0.0
        edge_mean = 0.0
        far_mean = 0.0

    # Aggregate by oracle cell for geodesic diagnostics.
    cell_ids, cell_means, cell_counts = _aggregate_by_cell(emb, data.cell_idx[node_global])
    visited_mask = cell_ids
    if len(cell_ids) >= 4:
        rng = np.random.default_rng(123)
        n_pairs = min(1000, len(cell_ids) * max(len(cell_ids) - 1, 1))
        ii = rng.integers(0, len(cell_ids), size=n_pairs)
        jj = rng.integers(0, len(cell_ids), size=n_pairs)
        ok = ii != jj
        ii = ii[ok]
        jj = jj[ok]
        geo_d = geodesic.dist_matrix[np.ix_(cell_ids[ii], cell_ids[jj])].astype(np.float32).diagonal() if False else None
        geo_d = np.asarray([float(geodesic.dist_matrix[int(cell_ids[i]), int(cell_ids[j])]) for i, j in zip(ii, jj)], dtype=np.float32)
        d_cell = np.linalg.norm(cell_means[ii] - cell_means[jj], axis=-1)
        geo_scale = _fit_scale(d_cell, geo_d)
        geo_corr = _safe_corr(geo_d, d_cell / max(geo_scale, 1e-8))
        geo_bins = _distance_bin_report(geo_d, d_cell, geo_scale)

        dist_embed_cell = np.linalg.norm(cell_means[:, None, :] - cell_means[None, :, :], axis=-1) / max(geo_scale, 1e-8)
        dist_geo_cell = geodesic.dist_matrix[np.ix_(cell_ids, cell_ids)].astype(np.float32)
        geo_knn = _knn_overlap_from_dist(dist_embed_cell, dist_geo_cell, k=min(10, len(cell_ids) - 1))
    else:
        geo_corr = {"pearson": 0.0, "spearman": 0.0, "n": 0}
        geo_bins = {}
        geo_knn = 0.0
        geo_scale = 1.0

    # Bridge-edge separation on oracle cell graph.
    bridge_pos = []
    bridge_neg = []
    c_to_loc = {int(c): i for i, c in enumerate(cell_ids.tolist())}
    bset = set(oracle_topo.bridges)
    for u_or, v_or in bset:
        if u_or in c_to_loc and v_or in c_to_loc:
            u = c_to_loc[u_or]
            v = c_to_loc[v_or]
            bridge_pos.append(float(np.linalg.norm(cell_means[u] - cell_means[v])))
    for u_or in cell_ids.tolist():
        for v_or in oracle_topo.adj[int(u_or)]:
            if int(u_or) >= int(v_or):
                continue
            a, b = (int(u_or), int(v_or)) if int(u_or) < int(v_or) else (int(v_or), int(u_or))
            if (a, b) in bset:
                continue
            if a in c_to_loc and b in c_to_loc:
                u = c_to_loc[a]
                v = c_to_loc[b]
                bridge_neg.append(float(np.linalg.norm(cell_means[u] - cell_means[v])))
    n_cmp = min(len(bridge_pos), len(bridge_neg))
    if n_cmp > 0:
        y = np.concatenate([np.ones(n_cmp), np.zeros(n_cmp)]).astype(np.int64)
        s = np.concatenate([np.asarray(bridge_pos[:n_cmp]), np.asarray(bridge_neg[:n_cmp])])
        bridge_auc = _binary_auc(y, s)
        bridge_gap = float(np.mean(bridge_pos[:n_cmp]) - np.mean(bridge_neg[:n_cmp]))
    else:
        bridge_auc = 0.0
        bridge_gap = 0.0

    # Path/chord ratio on oracle geodesic paths using cell means.
    adj_oracle = oracle_topo.adj
    rng = np.random.default_rng(7)
    path_ratios = []
    if len(cell_ids) >= 4:
        far_pairs = 0
        tries = 0
        while far_pairs < 12 and tries < 10_000:
            tries += 1
            a = int(rng.choice(cell_ids))
            b = int(rng.choice(cell_ids))
            if a == b:
                continue
            if float(geodesic.dist_matrix[a, b]) < 8:
                continue
            path = _bfs_shortest_path(adj_oracle, a, b)
            if not path:
                continue
            if any((p not in c_to_loc) for p in path):
                continue
            zs = np.stack([cell_means[c_to_loc[p]] for p in path], axis=0)
            chord = float(np.linalg.norm(zs[-1] - zs[0]))
            plen = float(np.linalg.norm(np.diff(zs, axis=0), axis=-1).sum())
            if chord > 1e-8:
                path_ratios.append(plen / chord)
                far_pairs += 1
    path_ratio_mean = float(np.mean(path_ratios)) if path_ratios else 0.0

    # Context sensitivity diagnostics.
    reset_map = _embed_reset_cells(cell_ids, env, encoder, rssm, device, bit_depth, name, head)
    within_cell = _within_cell_variance(emb, data.cell_idx[node_global])
    rr_gap = _rollout_reset_gap(cell_ids, cell_means, reset_map)

    return {
        "replay_distance": {
            **replay_corr,
            "mae": _mae_after_scale(d_embed, replay_eval_d, scale_for_distance),
            "scale": float(scale_for_distance),
            "bins": replay_bins,
        },
        "oracle_geodesic": {
            **geo_corr,
            "scale": float(geo_scale),
            "bins": geo_bins,
        },
        "knn_overlap": {
            "replay": float(replay_knn),
            "oracle_geodesic": float(geo_knn),
        },
        "edge_vs_far": {
            "auc": float(edge_far_auc),
            "mean_edge_dist": float(edge_mean),
            "mean_far_dist": float(far_mean),
        },
        "bridge_separation": {
            "auc": float(bridge_auc),
            "gap_mean": float(bridge_gap),
        },
        "path_chord_ratio_mean": float(path_ratio_mean),
        "context": {
            "within_cell_var_mean": float(within_cell),
            "rollout_reset_gap_mean": float(rr_gap),
            "n_cells_eval": int(len(cell_ids)),
            "mean_cell_count": float(np.mean(cell_counts)) if len(cell_counts) else 0.0,
        },
    }


@torch.no_grad()
def _evaluate_replay_potential_and_imagination(
    replay_head: GeoEncoder,
    replay_scale: float,
    replay_graph: ReplayGraph,
    data: RolloutData,
    env: PointMazeLargeDreamerWrapper,
    encoder: ConvEncoder,
    rssm: RSSM,
    actor: Actor | None,
    device: torch.device,
    bit_depth: int,
    imagination_starts: int,
    imagination_horizon: int,
    lambda_front: float,
    lambda_dist: float,
    lambda_off: float,
    ngu_weight: float,
    bonus_conf_power: float,
    use_disagreement_bonus: bool = False,
    lambda_disagree: float = 0.0,
) -> dict[str, Any]:
    gc_global = replay_graph.giant_nodes_global
    if len(gc_global) < 8:
        return {}

    h_gc = torch.tensor(data.h[gc_global], dtype=torch.float32, device=device)
    s_gc = torch.tensor(data.s[gc_global], dtype=torch.float32, device=device)
    g_gc = replay_head(h_gc, s_gc).detach().cpu().numpy().astype(np.float32)

    gc_local = replay_graph.giant_nodes_local
    dist_replay = replay_graph.dist_mat[np.ix_(gc_local, gc_local)].astype(np.float32)
    rng = np.random.default_rng(0)
    landmarks = [int(rng.integers(len(gc_global)))]
    while len(landmarks) < min(16, len(gc_global)):
        d_min = np.min(dist_replay[:, landmarks], axis=1)
        cand = int(np.argmax(d_min))
        if cand in landmarks:
            break
        landmarks.append(cand)
    landmarks = np.asarray(landmarks, dtype=np.int64)
    d_lat_lm = np.linalg.norm(g_gc[:, None, :] - g_gc[landmarks][None, :, :], axis=-1) / max(replay_scale, 1e-8)
    d_rep_lm = dist_replay[:, landmarks]
    distortion = np.mean(np.abs(d_lat_lm - d_rep_lm), axis=1)

    node_conf = replay_graph.node_confidence[gc_local].astype(np.float32)
    conf = np.power(np.clip(node_conf, 0.0, 1.0), float(max(bonus_conf_power, 1e-6)))

    visit_counts = defaultdict(int)
    for c in data.cell_idx[gc_global].tolist():
        visit_counts[int(c)] += 1
    frontier = np.asarray([1.0 / math.sqrt(visit_counts[int(c)] + 1.0) for c in data.cell_idx[gc_global]], dtype=np.float32)

    # NGU-style novelty proxy: k-th NN radius in replay-head space.
    dm = np.linalg.norm(g_gc[:, None, :] - g_gc[None, :, :], axis=-1)
    np.fill_diagonal(dm, np.inf)
    k = min(10, max(1, len(gc_global) - 1))
    ngu = np.sort(dm, axis=1)[:, k - 1] / max(replay_scale, 1e-8)
    if np.isfinite(ngu).any():
        ngu = (ngu - np.nanmean(ngu)) / (np.nanstd(ngu) + 1e-6)
    ngu = np.nan_to_num(ngu, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    disagree = np.zeros_like(frontier)
    if use_disagreement_bonus and lambda_disagree > 0:
        raw_gc = data.z[gc_global].astype(np.float32)
        d_raw_lm = np.linalg.norm(raw_gc[:, None, :] - raw_gc[landmarks][None, :, :], axis=-1)
        raw_scale = _fit_scale(d_raw_lm.reshape(-1), d_rep_lm.reshape(-1))
        disagree = np.mean(np.abs(d_raw_lm / max(raw_scale, 1e-8) - d_rep_lm), axis=1).astype(np.float32)
        if np.isfinite(disagree).any():
            disagree = (disagree - np.nanmean(disagree)) / (np.nanstd(disagree) + 1e-6)
        disagree = np.nan_to_num(disagree, nan=0.0, posinf=0.0, neginf=0.0)

    frontier_conf = frontier * conf
    node_bonus = float(ngu_weight) * ngu + float(lambda_front) * frontier_conf + float(lambda_dist) * (distortion * conf)
    if use_disagreement_bonus and lambda_disagree > 0:
        node_bonus = node_bonus + float(lambda_disagree) * disagree * conf

    oracle_topo = _make_oracle_topology(env.geodesic)
    bridge_cells: set[int] = set()
    for u, v in oracle_topo.bridges:
        bridge_cells.add(int(u))
        bridge_cells.add(int(v))
    y_bridge = np.asarray([1 if int(c) in bridge_cells else 0 for c in data.cell_idx[gc_global]], dtype=np.int64)

    out = {
        "potential": {
            "frontier_mean": float(np.mean(frontier)),
            "confidence_mean": float(np.mean(conf)),
            "frontier_conf_mean": float(np.mean(frontier_conf)),
            "ngu_mean": float(np.mean(ngu)),
            "distortion_mean": float(np.mean(distortion)),
            "bonus_mean": float(np.mean(node_bonus)),
            "bonus_bridge_auc": float(_binary_auc(y_bridge, node_bonus)),
            "frontier_conf_bridge_auc": float(_binary_auc(y_bridge, frontier_conf)),
            "distortion_bridge_auc": float(_binary_auc(y_bridge, distortion)),
            "confidence_bridge_auc": float(_binary_auc(y_bridge, conf)),
        }
    }
    if use_disagreement_bonus and lambda_disagree > 0:
        out["potential"]["disagreement_mean"] = float(np.mean(disagree))
        out["potential"]["disagreement_bridge_auc"] = float(_binary_auc(y_bridge, disagree))

    if actor is None:
        return out

    starts = np.random.default_rng(123).choice(len(gc_global), size=min(imagination_starts, len(gc_global)), replace=False)
    h = torch.tensor(data.h[gc_global[starts]], dtype=torch.float32, device=device)
    s = torch.tensor(data.s[gc_global[starts]], dtype=torch.float32, device=device)

    h_seq = [h]
    s_seq = [s]
    for _ in range(int(imagination_horizon)):
        a, _ = actor.get_action(h_seq[-1], s_seq[-1], deterministic=False)
        h_next = rssm.deterministic_state_fwd(h_seq[-1], s_seq[-1], a)
        s_next = rssm.state_prior(h_next, sample=True)
        h_seq.append(h_next)
        s_seq.append(s_next)
    h_im = torch.stack(h_seq, dim=1)
    s_im = torch.stack(s_seq, dim=1)

    g_im = replay_head(h_im.reshape(-1, h_im.size(-1)), s_im.reshape(-1, s_im.size(-1))).detach().cpu().numpy().astype(np.float32)
    dm_im = np.linalg.norm(g_im[:, None, :] - g_gc[None, :, :], axis=-1)
    nn_idx = np.argmin(dm_im, axis=1)
    nn_dist = dm_im[np.arange(dm_im.shape[0]), nn_idx]
    nn_idx = nn_idx.reshape(h_im.size(0), h_im.size(1))
    nn_dist = nn_dist.reshape(h_im.size(0), h_im.size(1))
    bonus_path = node_bonus[nn_idx]
    bonus_gain = bonus_path[:, 1:] - bonus_path[:, :-1]
    frontier_gain = frontier_conf[nn_idx[:, 1:]] - frontier_conf[nn_idx[:, :-1]]
    ngu_gain = ngu[nn_idx[:, 1:]] - ngu[nn_idx[:, :-1]]
    disagree_gain = disagree[nn_idx[:, 1:]] - disagree[nn_idx[:, :-1]] if (use_disagreement_bonus and lambda_disagree > 0) else None
    replay_jump = []
    for b in range(nn_idx.shape[0]):
        for t in range(nn_idx.shape[1] - 1):
            replay_jump.append(float(dist_replay[nn_idx[b, t], nn_idx[b, t + 1]]))

    out["imagination"] = {
        "off_manifold_mean": float(np.mean(nn_dist[:, 1:])),
        "off_manifold_p90": float(np.percentile(nn_dist[:, 1:], 90)),
        "bonus_gain_mean": float(np.mean(bonus_gain)),
        "bonus_gain_pos_frac": float(np.mean(bonus_gain > 0)),
        "frontier_gain_mean": float(np.mean(frontier_gain)),
        "ngu_gain_mean": float(np.mean(ngu_gain)),
        "replay_nn_step_mean": float(np.mean(replay_jump)) if replay_jump else 0.0,
        "intrinsic_reward_mean": float(np.mean(bonus_gain)),
    }
    if disagree_gain is not None:
        out["imagination"]["disagreement_gain_mean"] = float(np.mean(disagree_gain))
    return out


def _evaluate_topology_head(
    topo_head: PairTopologyHead,
    z_nodes: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    replay_same_room: np.ndarray,
    oracle_same_room: np.ndarray,
    oracle_geo_d: np.ndarray,
    device: torch.device,
) -> dict[str, Any]:
    with torch.no_grad():
        zi = torch.tensor(z_nodes[pair_i], dtype=torch.float32, device=device)
        zj = torch.tensor(z_nodes[pair_j], dtype=torch.float32, device=device)
        logits = topo_head(zi, zj).detach().cpu().numpy().astype(np.float32)

    out = {
        "replay_same_room": {
            "auc": float(_binary_auc(replay_same_room, logits)),
            "balanced_acc": float(_balanced_accuracy(replay_same_room, logits, thresh=0.0)),
        },
        "oracle_same_room": {
            "auc": float(_binary_auc(oracle_same_room, logits)),
            "balanced_acc": float(_balanced_accuracy(oracle_same_room, logits, thresh=0.0)),
        },
        "bins": {},
    }
    for name, fn in BIN_SPECS:
        mask = fn(oracle_geo_d)
        if mask.sum() < 8:
            out["bins"][name] = {"n": int(mask.sum())}
            continue
        out["bins"][name] = {
            "n": int(mask.sum()),
            "replay_auc": float(_binary_auc(replay_same_room[mask], logits[mask])),
            "oracle_auc": float(_binary_auc(oracle_same_room[mask], logits[mask])),
            "oracle_bal_acc": float(_balanced_accuracy(oracle_same_room[mask], logits[mask], thresh=0.0)),
        }
    return out


# -----------------------------------------------------------------------------
# Main per-checkpoint pipeline
# -----------------------------------------------------------------------------


def _compute_teacher_quality(data: RolloutData, replay_graph: ReplayGraph, geodesic) -> dict[str, Any]:
    oracle_topo = _make_oracle_topology(geodesic)
    node_global = replay_graph.node_indices
    cell_nodes = data.cell_idx[node_global]
    visited_cells = np.unique(cell_nodes)
    coverage = float(len(visited_cells) / max(int(geodesic.n_free), 1))

    # room/bottleneck coverage diagnostics
    oracle_rooms = oracle_topo.rooms[visited_cells]
    bridge_cells: set[int] = set()
    for u, v in oracle_topo.bridges:
        bridge_cells.add(int(u))
        bridge_cells.add(int(v))
    visited_bridge = sum(1 for c in visited_cells.tolist() if int(c) in bridge_cells)
    total_bridge = max(len(bridge_cells), 1)

    cell_ids, cell_means, _ = _aggregate_by_cell(data.z[node_global], cell_nodes)
    if len(cell_ids) >= 4:
        rng = np.random.default_rng(12345)
        n_pairs = min(1200, len(cell_ids) * max(len(cell_ids) - 1, 1))
        ii = rng.integers(0, len(cell_ids), size=n_pairs)
        jj = rng.integers(0, len(cell_ids), size=n_pairs)
        ok = ii != jj
        ii = ii[ok]
        jj = jj[ok]
        geo_d = np.asarray([float(geodesic.dist_matrix[int(cell_ids[i]), int(cell_ids[j])]) for i, j in zip(ii, jj)], dtype=np.float32)
        # replay distance via closest teacher node per cell mean in first graph view basis
        basis_all = data.enc_e[node_global] if "encoder" in replay_graph.views else data.z[node_global]
        basis_cell = []
        for c in cell_ids.tolist():
            basis_cell.append(basis_all[cell_nodes == c].mean(axis=0))
        basis_cell = _standardize_features(np.stack(basis_cell, axis=0).astype(np.float32))
        basis_graph = _standardize_features(basis_all.astype(np.float32))
        dm = np.linalg.norm(basis_cell[:, None, :] - basis_graph[None, :, :], axis=-1)
        nn = np.argmin(dm, axis=1)
        rep_d = replay_graph.dist_mat[np.ix_(nn[ii], nn[jj])].astype(np.float32).diagonal() if False else np.asarray([float(replay_graph.dist_mat[int(nn[i]), int(nn[j])]) for i, j in zip(ii, jj)], dtype=np.float32)
        corr = _safe_corr(geo_d, rep_d)
        corr["mean_replay"] = float(np.mean(rep_d[np.isfinite(rep_d)])) if np.isfinite(rep_d).any() else 0.0
        corr["mean_geodesic"] = float(np.mean(geo_d[np.isfinite(geo_d)])) if np.isfinite(geo_d).any() else 0.0
    else:
        corr = {"pearson": 0.0, "spearman": 0.0, "n": 0, "mean_replay": 0.0, "mean_geodesic": 0.0}

    return {
        "coverage": coverage,
        "visited_cells": int(len(visited_cells)),
        "giant_component_fraction": float(replay_graph.graph_stats.get("giant_component_fraction", 0.0)),
        "geo_vs_replay": corr,
        "oracle_room_coverage": float(len(np.unique(oracle_rooms)) / max(int(oracle_topo.n_rooms), 1)) if len(visited_cells) else 0.0,
        "oracle_bridge_cells_total": int(len(bridge_cells)),
        "oracle_bridge_cells_visited": int(visited_bridge),
        "oracle_bridge_coverage": float(visited_bridge / total_bridge),
    }


def evaluate_checkpoint(args: argparse.Namespace, ckpt_path: str) -> dict[str, Any]:
    set_seed(args.seed)
    device = get_device()

    env = PointMazeLargeDreamerWrapper(img_size=args.img_size, reset_mode=args.reset_mode)
    geodesic = env.geodesic
    H, W, C = env.observation_space.shape
    act_dim = int(env.action_space.shape[0])

    encoder, rssm, actor = load_models(
        ckpt_path=ckpt_path,
        device=device,
        obs_channels=C,
        act_dim=act_dim,
        embed_dim=args.embed_dim,
        stoch_dim=args.stoch_dim,
        deter_dim=args.deter_dim,
        hidden_dim=args.hidden_dim,
        actor_hidden_dim=args.actor_hidden_dim,
    )

    teacher_data, mode_hist = collect_teacher_data_with_quotas(
        env=env,
        encoder=encoder,
        rssm=rssm,
        actor=actor,
        device=device,
        bit_depth=args.bit_depth,
        teacher_collect_episodes=args.teacher_collect_episodes,
        state_stride=args.state_stride,
        max_nodes=args.teacher_max_nodes,
        deterministic_policy=args.policy_deterministic,
        expl_noise=args.expl_noise,
        teacher_high_noise=args.teacher_high_noise,
        teacher_random_fraction=args.teacher_random_fraction,
        teacher_subset_fraction=args.teacher_subset_fraction,
        teacher_start_subset_size=args.teacher_start_subset_size,
    )

    replay_graph = build_replay_graph(
        data=teacher_data,
        basis_name=args.graph_knn_basis,
        graph_knn_k=args.graph_knn_k,
        graph_knn_weight=args.graph_knn_weight,
        graph_temporal_weight=args.graph_temporal_weight,
        graph_knn_max_percentile=args.graph_knn_max_percentile,
        graph_same_ep_gap=args.graph_same_ep_gap,
        max_graph_nodes=args.max_graph_nodes,
        graph_views=[x.strip() for x in args.graph_views.split(",") if x.strip()],
        graph_min_view_votes=args.graph_min_view_votes,
    )

    teacher_quality = _compute_teacher_quality(teacher_data, replay_graph, geodesic)
    teacher_quality["thresholds"] = {
        "min_coverage": float(args.teacher_min_coverage),
        "min_giant_component_fraction": float(args.teacher_min_giant_fraction),
        "min_geo_replay_spearman": float(args.teacher_min_geo_replay_spearman),
    }
    teacher_quality["passes"] = {
        "coverage": bool(teacher_quality["coverage"] >= args.teacher_min_coverage),
        "giant_component_fraction": bool(teacher_quality["giant_component_fraction"] >= args.teacher_min_giant_fraction),
        "geo_vs_replay": bool(teacher_quality["geo_vs_replay"]["spearman"] >= args.teacher_min_geo_replay_spearman),
    }
    teacher_quality["topology_ok"] = bool(all(teacher_quality["passes"].values()))

    oracle_topo = _make_oracle_topology(geodesic)
    gc = replay_graph.giant_nodes_local
    if len(gc) < 32:
        raise RuntimeError("Replay graph giant component too small for meaningful training/eval.")

    pair_train_i, pair_train_j, replay_train_d = _sample_pairs_by_bins(
        replay_graph.dist_mat, gc, args.train_pairs, seed=args.seed + 17
    )
    pair_eval_i, pair_eval_j, replay_eval_d = _sample_pairs_by_bins(
        replay_graph.dist_mat, gc, args.eval_pairs, seed=args.seed + 29
    )

    local_ep = teacher_data.episode_id[replay_graph.node_indices]
    local_t = teacher_data.t_in_ep[replay_graph.node_indices]
    replay_head, replay_scale = train_replay_metric_head(
        h=teacher_data.h[replay_graph.node_indices],
        s=teacher_data.s[replay_graph.node_indices],
        replay_dist=replay_graph.dist_mat,
        node_ids=gc,
        episode_id=local_ep,
        t_in_ep=local_t,
        geo_dim=args.geo_dim,
        hidden_dim=args.geo_hidden,
        lr=args.geo_lr,
        epochs=args.replay_head_epochs,
        batch_pairs=args.replay_head_batch_pairs,
        device=device,
        seed=args.seed + 101,
        canon_weight=args.canon_weight,
        canon_batch_pairs=args.canon_batch_pairs,
        canon_pos_radius=args.canon_pos_radius,
        canon_min_gap=args.canon_min_gap,
    )

    temp_head = train_temporal_head(
        h=teacher_data.h,
        s=teacher_data.s,
        episode_id=teacher_data.episode_id,
        t_in_ep=teacher_data.t_in_ep,
        node_ids=replay_graph.giant_nodes_global,
        geo_dim=args.geo_dim,
        hidden_dim=args.geo_hidden,
        lr=args.geo_lr,
        epochs=args.temp_head_epochs,
        batch_size=args.temp_head_batch_size,
        seq_len=args.temp_head_seq_len,
        device=device,
        seed=args.seed + 202,
    )

    topo_pair_i, topo_pair_j, topo_d = _sample_pairs_by_bins(
        replay_graph.dist_mat, gc, args.topo_eval_pairs, seed=args.seed + 303
    )
    replay_room = replay_graph.rooms_wo_bridges
    replay_same_room = (replay_room[topo_pair_i] == replay_room[topo_pair_j]).astype(np.int64)

    global_nodes = replay_graph.node_indices
    cell_of_local = teacher_data.cell_idx[global_nodes]
    oracle_same_room = (oracle_topo.rooms[cell_of_local[topo_pair_i]] == oracle_topo.rooms[cell_of_local[topo_pair_j]]).astype(np.int64)
    oracle_geo_d = np.asarray([
        float(geodesic.dist_matrix[int(cell_of_local[i]), int(cell_of_local[j])])
        for i, j in zip(topo_pair_i, topo_pair_j)
    ], dtype=np.float32)

    topo_eval: dict[str, Any]
    if not teacher_quality["topology_ok"]:
        topo_eval = {
            "skipped": True,
            "reason": "Teacher quality below threshold; topology conclusions skipped.",
            "teacher_quality": teacher_quality,
        }
    elif replay_same_room.min() != replay_same_room.max():
        topo_head = train_topology_pair_head(
            z=teacher_data.z[global_nodes],
            same_room_labels=replay_same_room,
            pair_i=topo_pair_i,
            pair_j=topo_pair_j,
            hidden_dim=args.topo_hidden,
            lr=args.topo_lr,
            epochs=args.topo_epochs,
            batch_size=args.topo_batch_size,
            device=device,
            seed=args.seed + 404,
        )
        topo_eval = _evaluate_topology_head(
            topo_head=topo_head,
            z_nodes=teacher_data.z[global_nodes],
            pair_i=topo_pair_i,
            pair_j=topo_pair_j,
            replay_same_room=replay_same_room,
            oracle_same_room=oracle_same_room,
            oracle_geo_d=oracle_geo_d,
            device=device,
        )
    else:
        topo_eval = {"warning": "Replay graph produced only one room after bridge removal; topology head skipped."}

    raw_emb = teacher_data.z[global_nodes]
    replay_emb = _embed_with_head("replay_metric", teacher_data.h[global_nodes], teacher_data.s[global_nodes], teacher_data.z[global_nodes], replay_head, device)
    temp_emb = _embed_with_head("temporal_metric", teacher_data.h[global_nodes], teacher_data.s[global_nodes], teacher_data.z[global_nodes], temp_head, device)

    temp_scale = _fit_scale(
        np.linalg.norm(temp_emb[pair_train_i] - temp_emb[pair_train_j], axis=-1), replay_train_d
    )
    raw_scale = _fit_scale(
        np.linalg.norm(raw_emb[pair_train_i] - raw_emb[pair_train_j], axis=-1), replay_train_d
    )

    raw_eval = _evaluate_embedding(
        name="raw_hs",
        emb_nodes=raw_emb,
        replay_graph=replay_graph,
        data=teacher_data,
        geodesic=geodesic,
        oracle_topo=oracle_topo,
        scale_for_distance=raw_scale,
        pair_eval_i_local=pair_eval_i,
        pair_eval_j_local=pair_eval_j,
        replay_eval_d=replay_eval_d,
        device=device,
        env=env,
        encoder=encoder,
        rssm=rssm,
        bit_depth=args.bit_depth,
        head=None,
    )
    replay_eval = _evaluate_embedding(
        name="replay_metric",
        emb_nodes=replay_emb,
        replay_graph=replay_graph,
        data=teacher_data,
        geodesic=geodesic,
        oracle_topo=oracle_topo,
        scale_for_distance=replay_scale,
        pair_eval_i_local=pair_eval_i,
        pair_eval_j_local=pair_eval_j,
        replay_eval_d=replay_eval_d,
        device=device,
        env=env,
        encoder=encoder,
        rssm=rssm,
        bit_depth=args.bit_depth,
        head=replay_head,
    )
    replay_extra = _evaluate_replay_potential_and_imagination(
        replay_head=replay_head,
        replay_scale=replay_scale,
        replay_graph=replay_graph,
        data=teacher_data,
        env=env,
        encoder=encoder,
        rssm=rssm,
        actor=actor,
        device=device,
        bit_depth=args.bit_depth,
        imagination_starts=args.imagination_starts,
        imagination_horizon=args.imagination_horizon,
        lambda_front=args.lambda_front,
        lambda_dist=args.lambda_dist,
        lambda_off=args.lambda_off,
        ngu_weight=args.ngu_weight,
        bonus_conf_power=args.bonus_conf_power,
        use_disagreement_bonus=args.enable_disagreement_bonus,
        lambda_disagree=args.lambda_disagree,
    )
    replay_eval.update(replay_extra)

    temp_eval = _evaluate_embedding(
        name="temporal_metric",
        emb_nodes=temp_emb,
        replay_graph=replay_graph,
        data=teacher_data,
        geodesic=geodesic,
        oracle_topo=oracle_topo,
        scale_for_distance=temp_scale,
        pair_eval_i_local=pair_eval_i,
        pair_eval_j_local=pair_eval_j,
        replay_eval_d=replay_eval_d,
        device=device,
        env=env,
        encoder=encoder,
        rssm=rssm,
        bit_depth=args.bit_depth,
        head=temp_head,
    )

    visited_cells = int(len(np.unique(teacher_data.cell_idx[replay_graph.node_indices])))
    summary = {
        "file": os.path.basename(ckpt_path),
        "sanity": {
            "collection_mode_hist": {k: int(v) for k, v in mode_hist.items()},
            "teacher_quality": teacher_quality,
            "n_rollout_nodes": int(teacher_data.z.shape[0]),
            "n_graph_nodes": int(len(replay_graph.node_indices)),
            "n_free_cells_total": int(geodesic.n_free),
            "n_cells_visited": int(visited_cells),
            "coverage": float(visited_cells / max(int(geodesic.n_free), 1)),
            "graph": replay_graph.graph_stats,
            "mean_reward": float(np.mean(teacher_data.reward)),
            "success_rate": float(np.mean(teacher_data.success)),
        },
        "raw_hs": raw_eval,
        "replay_metric_head": replay_eval,
        "temporal_reachability_head": temp_eval,
        "topology_pair_head": topo_eval,
    }
    env.close()
    return summary


# -----------------------------------------------------------------------------
# CLI / output
# -----------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Offline evaluation of replay-derived heads for dreamer_vics checkpoints")
    p.add_argument("--run_dir", type=str, default=".")
    p.add_argument("--checkpoints", nargs="*", type=str, default=["world_model_final.pt"])
    p.add_argument("--out_dir", type=str, default="")

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--reset_mode", type=str, default="fixed_start")

    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--actor_hidden_dim", type=int, default=400)

    p.add_argument("--collect_episodes", type=int, default=40)
    p.add_argument("--state_stride", type=int, default=2)
    p.add_argument("--max_nodes", type=int, default=2500)
    p.add_argument("--teacher_collect_episodes", type=int, default=80)
    p.add_argument("--teacher_max_nodes", type=int, default=3500)
    p.add_argument("--teacher_high_noise", type=float, default=0.45)
    p.add_argument("--teacher_random_fraction", type=float, default=0.4)
    p.add_argument("--teacher_subset_fraction", type=float, default=0.35)
    p.add_argument("--teacher_start_subset_size", type=int, default=6)
    p.add_argument("--teacher_min_coverage", type=float, default=0.45)
    p.add_argument("--teacher_min_giant_fraction", type=float, default=0.3)
    p.add_argument("--teacher_min_geo_replay_spearman", type=float, default=0.45)
    p.add_argument("--teacher_quality_pairs", type=int, default=1200)
    p.add_argument("--policy_deterministic", action="store_true")
    p.add_argument("--expl_noise", type=float, default=0.15)
    p.add_argument("--random_policy", action="store_true")

    p.add_argument("--max_graph_nodes", type=int, default=1800)
    p.add_argument("--graph_knn_basis", type=str, default="encoder", choices=["encoder", "z", "h", "s"])
    p.add_argument("--graph_views", type=str, default="encoder,z")
    p.add_argument("--graph_min_view_votes", type=int, default=1)
    p.add_argument("--graph_knn_k", type=int, default=6)
    p.add_argument("--graph_knn_weight", type=float, default=1.0)
    p.add_argument("--graph_temporal_weight", type=float, default=1.0)
    p.add_argument("--graph_knn_max_percentile", type=float, default=80.0)
    p.add_argument("--graph_same_ep_gap", type=int, default=2)

    p.add_argument("--train_pairs", type=int, default=1500)
    p.add_argument("--eval_pairs", type=int, default=1500)

    p.add_argument("--geo_dim", type=int, default=32)
    p.add_argument("--geo_hidden", type=int, default=256)
    p.add_argument("--geo_lr", type=float, default=3e-4)

    p.add_argument("--replay_head_epochs", type=int, default=250)
    p.add_argument("--replay_head_batch_pairs", type=int, default=512)
    p.add_argument("--canon_weight", type=float, default=0.25)
    p.add_argument("--canon_batch_pairs", type=int, default=128)
    p.add_argument("--canon_pos_radius", type=float, default=3.0)
    p.add_argument("--canon_min_gap", type=int, default=8)

    p.add_argument("--temp_head_epochs", type=int, default=250)
    p.add_argument("--temp_head_batch_size", type=int, default=32)
    p.add_argument("--temp_head_seq_len", type=int, default=12)

    p.add_argument("--topo_epochs", type=int, default=250)
    p.add_argument("--topo_batch_size", type=int, default=256)
    p.add_argument("--topo_hidden", type=int, default=256)
    p.add_argument("--topo_lr", type=float, default=3e-4)
    p.add_argument("--topo_eval_pairs", type=int, default=1800)

    p.add_argument("--imagination_starts", type=int, default=32)
    p.add_argument("--imagination_horizon", type=int, default=15)
    p.add_argument("--lambda_front", type=float, default=0.5)
    p.add_argument("--lambda_dist", type=float, default=0.0)
    p.add_argument("--lambda_off", type=float, default=0.0)
    p.add_argument("--ngu_weight", type=float, default=1.0)
    p.add_argument("--bonus_conf_power", type=float, default=1.0)
    p.add_argument("--enable_disagreement_bonus", action="store_true")
    p.add_argument("--lambda_disagree", type=float, default=0.0)
    return p


def _make_summary_plots(results: list[dict[str, Any]], out_dir: str) -> None:
    if not results:
        return
    names = [r["file"] for r in results]
    x = np.arange(len(names))

    def vals(path: list[str]) -> list[float]:
        out = []
        for r in results:
            cur: Any = r
            ok = True
            for k in path:
                if isinstance(cur, dict) and k in cur:
                    cur = cur[k]
                else:
                    ok = False
                    break
            out.append(float(cur) if ok and np.isfinite(cur) else 0.0)
        return out

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=130)
    axes[0].bar(x - 0.25, vals(["raw_hs", "oracle_geodesic", "spearman"]), width=0.25, label="raw_hs")
    axes[0].bar(x, vals(["replay_metric_head", "oracle_geodesic", "spearman"]), width=0.25, label="g_replay")
    axes[0].bar(x + 0.25, vals(["temporal_reachability_head", "oracle_geodesic", "spearman"]), width=0.25, label="g_temp")
    axes[0].set_title("Oracle geodesic Spearman")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=25, ha="right")
    axes[0].legend(fontsize=8)

    axes[1].bar(x - 0.25, vals(["raw_hs", "replay_distance", "spearman"]), width=0.25, label="raw_hs")
    axes[1].bar(x, vals(["replay_metric_head", "replay_distance", "spearman"]), width=0.25, label="g_replay")
    axes[1].bar(x + 0.25, vals(["temporal_reachability_head", "replay_distance", "spearman"]), width=0.25, label="g_temp")
    axes[1].set_title("Replay distance Spearman")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=25, ha="right")

    axes[2].bar(x, vals(["topology_pair_head", "oracle_same_room", "auc"]), width=0.4)
    axes[2].set_title("Topology head: oracle room AUC")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_heads.png"))
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    out_dir = args.out_dir or os.path.join(args.run_dir, "replay_head_eval")
    os.makedirs(out_dir, exist_ok=True)

    ckpt_paths = []
    for name in args.checkpoints:
        path = name if os.path.isabs(name) else os.path.join(args.run_dir, name)
        if os.path.isfile(path):
            ckpt_paths.append(path)
        else:
            print(f"[skip] missing checkpoint: {path}")
    if not ckpt_paths:
        raise SystemExit("No valid checkpoints found.")

    all_results = {"meta": vars(args), "checkpoints": []}
    for ckpt in ckpt_paths:
        print(f"\n=== Evaluating {os.path.basename(ckpt)} ===")
        res = evaluate_checkpoint(args, ckpt)
        all_results["checkpoints"].append(res)
        print(json.dumps({
            "file": res["file"],
            "coverage": res["sanity"]["coverage"],
            "teacher_geo_vs_replay": res["sanity"]["teacher_quality"]["geo_vs_replay"]["spearman"],
            "teacher_topology_ok": res["sanity"]["teacher_quality"]["topology_ok"],
            "raw_geo_spearman": res["raw_hs"]["oracle_geodesic"]["spearman"],
            "replay_geo_spearman": res["replay_metric_head"]["oracle_geodesic"]["spearman"],
            "temp_geo_spearman": res["temporal_reachability_head"]["oracle_geodesic"]["spearman"],
            "topo_oracle_auc": res["topology_pair_head"].get("oracle_same_room", {}).get("auc", 0.0),
        }, indent=2))

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    _make_summary_plots(all_results["checkpoints"], out_dir)

    print(f"\nWrote metrics and plots to {out_dir}")

if __name__ == "__main__":
    main()
