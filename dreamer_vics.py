#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

from maze_geometry_test import _positions_to_cell_indices
from models import RSSM, Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel, ValueModel
from geom_head import GeoEncoder, temporal_reachability_loss
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
# Geometric losses
# =====================================================================


def vicreg_loss(
    z: torch.Tensor,
    z_pos: torch.Tensor,
    var_weight: float = 25.0,
    inv_weight: float = 25.0,
    cov_weight: float = 1.0,
    target_std: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """VICReg loss on latent states.

    z:     [B, D] — anchor latents (e.g. z_t = [h_t, s_t])
    z_pos: [B, D] — positive latents (e.g. z_{t+1} from same episode)

    Returns (total_loss, {component scalars for logging}).

    Variance: per-dim std of z should be >= target_std.
              Uses hinge: ReLU(target - std).
    Invariance: MSE between z and z_pos (temporal neighbors should be close).
    Covariance: off-diagonal of cov(z) should be zero (decorrelation).
    """
    B, D = z.shape

    # Variance: hinge on per-dimension std
    z_centered = z - z.mean(dim=0)
    std_z = torch.sqrt(z_centered.var(dim=0) + 1e-4)
    l_var_z = F.relu(target_std - std_z).mean()

    z_pos_centered = z_pos - z_pos.mean(dim=0)
    std_zp = torch.sqrt(z_pos_centered.var(dim=0) + 1e-4)
    l_var_zp = F.relu(target_std - std_zp).mean()

    l_var = 0.5 * (l_var_z + l_var_zp)

    # Invariance: temporal neighbors should map nearby
    l_inv = F.mse_loss(z, z_pos)

    # Covariance: off-diagonal elements of covariance matrix → 0
    cov_z = (z_centered.T @ z_centered) / max(B - 1, 1)
    off_diag = cov_z.pow(2).sum() - cov_z.diagonal().pow(2).sum()
    l_cov_z = off_diag / D

    cov_zp = (z_pos_centered.T @ z_pos_centered) / max(B - 1, 1)
    off_diag_p = cov_zp.pow(2).sum() - cov_zp.diagonal().pow(2).sum()
    l_cov_zp = off_diag_p / D

    l_cov = 0.5 * (l_cov_z + l_cov_zp)

    total = var_weight * l_var + inv_weight * l_inv + cov_weight * l_cov
    info = {
        "vicreg/variance": float(l_var.item()),
        "vicreg/invariance": float(l_inv.item()),
        "vicreg/covariance": float(l_cov.item()),
        "vicreg/total": float(total.item()),
        "vicreg/mean_std": float(std_z.mean().item()),
        "vicreg/min_std": float(std_z.min().item()),
    }
    return total, info


def straightness_loss(
    z_seq: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Trajectory straightness regularization.

    z_seq: [B, T, D] — sequence of encoder embeddings from same episode.
    mask:  [B, T] — 1.0 where valid (same episode), 0.0 at episode boundaries.

    For each sequence, the linear interpolation between z_0 and z_{T-1}
    should pass through the intermediate z_t. Enforces that temporal
    trajectories are geodesics (straight lines) in latent space.
    """
    B, T, D = z_seq.shape
    if T < 3:
        return torch.tensor(0.0, device=z_seq.device), {"straight/loss": 0.0}

    z_start = z_seq[:, 0:1, :]              # [B, 1, D]
    z_end = z_seq[:, -1:, :]                # [B, 1, D]

    # Alpha: fractional position along trajectory [0, 1]
    alpha = torch.linspace(0.0, 1.0, T, device=z_seq.device)  # [T]
    alpha = alpha.view(1, T, 1)             # [1, T, 1]

    # Interpolated trajectory
    z_interp = z_start + alpha * (z_end - z_start)  # [B, T, D]

    # Deviation from straight line
    deviation = (z_seq - z_interp).pow(2).sum(dim=-1)  # [B, T]

    # Skip endpoints (they're exact by construction), focus on middle
    deviation = deviation[:, 1:-1]  # [B, T-2]

    if mask is not None:
        m = mask[:, 1:-1]
        if m.sum() < 1:
            return torch.tensor(0.0, device=z_seq.device), {"straight/loss": 0.0}
        l_straight = (deviation * m).sum() / m.sum().clamp(min=1)
    else:
        l_straight = deviation.mean()

    # Normalize by trajectory length to be scale-invariant
    traj_length = (z_end - z_start).pow(2).sum(dim=-1).mean().clamp(min=1e-6)
    l_straight_norm = l_straight / traj_length

    info = {
        "straight/loss": float(l_straight.item()),
        "straight/loss_normalized": float(l_straight_norm.item()),
        "straight/traj_length": float(traj_length.item()),
        "straight/mean_deviation": float(deviation.mean().item()),
    }
    return l_straight_norm, info


def rssm_latent(h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """Concatenate deterministic and stochastic Dreamer state into one latent."""
    return torch.cat([h, s], dim=-1)


def latent_coverage_bonus(
    z_current: torch.Tensor,
    z_memory: torch.Tensor,
    k: int = 5,
) -> torch.Tensor:
    """Intrinsic reward: distance to k-th nearest neighbor in latent memory.

    z_current: [B, D] — current step embeddings
    z_memory:  [M, D] — memory bank of past embeddings

    Returns: [B] intrinsic reward (higher = more novel region).
    """
    # Pairwise distances
    dists = torch.cdist(z_current, z_memory)  # [B, M]
    k_actual = min(k, dists.shape[1] - 1)
    if k_actual < 1:
        return torch.zeros(z_current.shape[0], device=z_current.device)

    # k-th nearest neighbor distance
    knn_dists, _ = torch.topk(dists, k_actual + 1, dim=1, largest=False)
    bonus = knn_dists[:, -1]  # k-th distance (skip self if self is in memory)
    return bonus


# =====================================================================
#  Teacher replay-graph + structured exploration module
# =====================================================================


def _standardize_features(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-6
    return (x - mu) / sd


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    try:
        from scipy.stats import spearmanr
        rho = spearmanr(x, y).correlation
        if rho is None or not np.isfinite(rho):
            return 0.0
        return float(rho)
    except Exception:
        return 0.0


def _set_env_mode(env: PointMazeLargeDreamerWrapper, mode: str, subset_size: int):
    old = (env.reset_mode, env.fixed_start_cell, list(env.start_cells))
    if mode == "random_reset":
        env.reset_mode = "random"
    elif mode == "start_subset":
        env.reset_mode = "start_subset"
        if not env.start_cells:
            cells = [tuple(env._free_cells[i]) for i in range(min(max(1, subset_size), len(env._free_cells)))]
            env.start_cells = cells
    elif mode == "fixed_start":
        env.reset_mode = "fixed_start"
    else:
        raise ValueError(f"Unknown teacher mode {mode!r}")
    return old


def _restore_env_mode(env: PointMazeLargeDreamerWrapper, old):
    env.reset_mode, env.fixed_start_cell, start_cells = old
    env.start_cells = list(start_cells)


@torch.no_grad()
def collect_teacher_data_with_quotas(
    env: PointMazeLargeDreamerWrapper,
    encoder: ConvEncoder,
    rssm: RSSM,
    actor: Actor | None,
    device: torch.device,
    bit_depth: int,
    teacher_collect_episodes: int,
    max_nodes: int,
    teacher_high_noise: float,
    teacher_random_fraction: float,
    teacher_subset_fraction: float,
    teacher_start_subset_size: int,
    state_stride: int,
):
    frac_random = float(np.clip(teacher_random_fraction, 0.0, 1.0))
    frac_subset = float(np.clip(teacher_subset_fraction, 0.0, 1.0 - frac_random))
    frac_fixed = max(0.0, 1.0 - frac_random - frac_subset)
    quotas = {
        "random_reset": max(1, int(round(max_nodes * max(frac_random, 1e-6)))) if frac_random > 0 else 0,
        "start_subset": max(1, int(round(max_nodes * max(frac_subset, 1e-6)))) if frac_subset > 0 else 0,
        "fixed_start": max(1, max_nodes - (max(1, int(round(max_nodes * max(frac_random, 1e-6)))) if frac_random > 0 else 0) - (max(1, int(round(max_nodes * max(frac_subset, 1e-6)))) if frac_subset > 0 else 0)),
    }
    if frac_random == 0: quotas["random_reset"] = 0
    if frac_subset == 0: quotas["start_subset"] = 0
    if frac_fixed == 0: quotas["fixed_start"] = 0

    obs_list=[]; pos_list=[]; cell_list=[]; ep_list=[]; t_list=[]; h_list=[]; s_list=[]; z_list=[]; e_list=[]; r_list=[]; succ_list=[]
    mode_hist = Counter()
    geodesic = env.geodesic
    ep_ctr = 0
    total_kept = 0
    modes = [m for m in ["random_reset", "start_subset", "fixed_start"] if quotas[m] > 0]
    if not modes:
        modes = ["fixed_start"]
        quotas["fixed_start"] = max_nodes

    for mode in modes:
        old = _set_env_mode(env, mode, teacher_start_subset_size)
        kept_mode = 0
        max_ep_mode = max(1, int(round(teacher_collect_episodes * (quotas[mode] / max(max_nodes, 1)))))
        for _ in range(max_ep_mode):
            if kept_mode >= quotas[mode] or total_kept >= max_nodes:
                break
            obs, _ = env.reset()
            x = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
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
                    ep_list.append(ep_ctr)
                    t_list.append(t_env)
                    h_np = h.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    s_np = s.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    e_np = e.squeeze(0).detach().cpu().numpy().astype(np.float32)
                    z_np = np.concatenate([h_np, s_np], axis=-1).astype(np.float32)
                    h_list.append(h_np); s_list.append(s_np); e_list.append(e_np); z_list.append(z_np)
                    r_list.append(0.0); succ_list.append(0.0)
                    total_kept += 1; kept_mode += 1; mode_hist[mode] += 1
                    if total_kept >= max_nodes or kept_mode >= quotas[mode]:
                        done = True
                        break

                if actor is None:
                    action = env.action_space.sample().astype(np.float32)
                    a_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                else:
                    a_t, _ = actor.get_action(h, s, deterministic=False)
                    noise = teacher_high_noise if mode != "fixed_start" else max(0.15, teacher_high_noise * 0.5)
                    if noise > 0:
                        a_t = torch.clamp(a_t + noise * torch.randn_like(a_t), -1.0, 1.0)
                    action = a_t.squeeze(0).detach().cpu().numpy().astype(np.float32)

                next_obs, r, term, trunc, info = env.step(action, repeat=1)
                done = bool(term or trunc)
                t_env += 1
                x = torch.tensor(np.ascontiguousarray(next_obs), dtype=torch.float32, device=device).permute(2,0,1).unsqueeze(0)
                preprocess_img(x, depth=bit_depth)
                e = encoder(x)
                h, s, _, _ = rssm.observe_step(e, a_t, h, s, sample=False)
                obs = next_obs
                if keep and total_kept > 0:
                    r_list[-1] = float(r)
                    succ_list[-1] = 1.0 if info.get("success", False) or info.get("is_success", False) else 0.0
            ep_ctr += 1
        _restore_env_mode(env, old)

    if not obs_list:
        raise RuntimeError("No teacher states collected.")

    return {
        "obs": np.stack(obs_list, axis=0),
        "pos": np.stack(pos_list, axis=0),
        "cell_idx": np.asarray(cell_list, dtype=np.int64),
        "episode_id": np.asarray(ep_list, dtype=np.int64),
        "t_in_ep": np.asarray(t_list, dtype=np.int64),
        "h": np.stack(h_list, axis=0),
        "s": np.stack(s_list, axis=0),
        "z": np.stack(z_list, axis=0),
        "enc_e": np.stack(e_list, axis=0),
        "reward": np.asarray(r_list, dtype=np.float32),
        "success": np.asarray(succ_list, dtype=np.float32),
    }, dict(mode_hist)


def _connected_components(adj):
    n = len(adj)
    comp = -np.ones(n, dtype=np.int64)
    groups = []
    cid = 0
    for i in range(n):
        if comp[i] != -1:
            continue
        q = deque([i])
        comp[i] = cid
        group = []
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


def build_multiview_replay_graph(data, graph_views, graph_knn_k, graph_knn_weight, graph_temporal_weight, graph_knn_max_percentile, graph_same_ep_gap, max_graph_nodes, graph_min_view_votes):
    rng = np.random.default_rng(0)
    n_total = data["z"].shape[0]
    keep = np.sort(rng.choice(n_total, size=max_graph_nodes, replace=False)) if n_total > max_graph_nodes else np.arange(n_total, dtype=np.int64)
    ep = data["episode_id"][keep]
    tt = data["t_in_ep"][keep]
    n = len(keep)
    edge_w = {}
    temporal_edges = 0
    for e_id in np.unique(ep):
        loc = np.where(ep == e_id)[0]
        if len(loc) < 2:
            continue
        order = np.argsort(tt[loc]); loc = loc[order]
        for a,b in zip(loc[:-1], loc[1:]):
            dt = int(tt[b] - tt[a])
            w = float(max(1, dt) * graph_temporal_weight)
            key = (int(min(a,b)), int(max(a,b)))
            edge_w[key] = min(edge_w.get(key, 1e9), w)
            temporal_edges += 1

    view_votes = defaultdict(int)
    view_list = list(graph_views) if graph_views else ["encoder", "z"]
    for view in view_list:
        if view == "encoder": basis = data["enc_e"][keep]
        elif view == "h": basis = data["h"][keep]
        elif view == "s": basis = data["s"][keep]
        else: basis = data["z"][keep]
        basis = _standardize_features(basis.astype(np.float32))
        if n < 4:
            continue
        dmat = np.linalg.norm(basis[:,None,:] - basis[None,:,:], axis=-1)
        np.fill_diagonal(dmat, np.inf)
        if graph_same_ep_gap > 0:
            same_ep = ep[:,None] == ep[None,:]
            gap = np.abs(tt[:,None] - tt[None,:])
            dmat[same_ep & (gap <= graph_same_ep_gap)] = np.inf
        nbrs = np.argsort(dmat, axis=1)[:, :graph_knn_k]
        cand = []
        seen = set()
        cand_d = []
        for i in range(n):
            for j in nbrs[i]:
                if not np.isfinite(dmat[i,j]):
                    continue
                if i in nbrs[j]:
                    a,b = (int(i),int(j)) if i<j else (int(j),int(i))
                    if (a,b) not in seen:
                        seen.add((a,b)); cand.append((a,b,float(dmat[a,b]))); cand_d.append(float(dmat[a,b]))
        if cand_d:
            max_d = float(np.percentile(np.asarray(cand_d, dtype=np.float32), graph_knn_max_percentile))
            for a,b,dij in cand:
                if dij <= max_d:
                    view_votes[(a,b)] += 1

    knn_edges = 0
    vote_strength = np.zeros(n, dtype=np.float32)
    for (a,b), votes in view_votes.items():
        if votes >= max(1, int(graph_min_view_votes)):
            edge_w[(a,b)] = min(edge_w.get((a,b), 1e9), float(graph_knn_weight))
            knn_edges += 1
            vote_strength[a] += votes / max(len(view_list),1)
            vote_strength[b] += votes / max(len(view_list),1)

    mat = lil_matrix((n,n), dtype=np.float32)
    adj = [[] for _ in range(n)]
    for (u,v),w in edge_w.items():
        mat[u,v] = float(w); mat[v,u] = float(w)
        adj[u].append(v); adj[v].append(u)
    dist = shortest_path(mat.tocsr(), method="D", directed=False, unweighted=False)
    comp_id, groups = _connected_components(adj)
    giant = max(groups, key=len) if groups else []
    giant_local = np.asarray(sorted(giant), dtype=np.int64)
    giant_global = keep[giant_local] if len(giant_local) else np.zeros((0,), dtype=np.int64)
    deg = np.asarray([len(a) for a in adj], dtype=np.float32)
    node_conf = vote_strength / np.maximum(deg, 1.0)
    stats = {
        "n_nodes_total": int(n),
        "n_edges_total": int(mat.nnz // 2),
        "n_temporal_edges": int(temporal_edges),
        "n_knn_edges": int(knn_edges),
        "n_components": int(len(groups)),
        "giant_component_size": int(len(giant_local)),
        "giant_component_fraction": float(len(giant_local) / max(n,1)),
    }
    return {
        "node_indices": keep,
        "dist_mat": np.asarray(dist, dtype=np.float32),
        "adj": adj,
        "comp_id": comp_id,
        "giant_nodes_local": giant_local,
        "giant_nodes_global": giant_global,
        "node_conf": node_conf.astype(np.float32),
        "stats": stats,
    }


def _sample_pairs_by_bins(dist_mat, node_ids, n_pairs, seed):
    rng = np.random.default_rng(seed)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    bins = {"short": [], "mid": [], "long": []}
    max_scan = min(80000, len(node_ids) * max(len(node_ids),1))
    for _ in range(max_scan):
        i = int(rng.choice(node_ids)); j = int(rng.choice(node_ids))
        if i == j:
            continue
        d = float(dist_mat[i,j])
        if not np.isfinite(d) or d <= 0:
            continue
        if d <= 3: bins["short"].append((i,j,d))
        elif d <= 8: bins["mid"].append((i,j,d))
        else: bins["long"].append((i,j,d))
        if sum(len(v) for v in bins.values()) >= max(4*n_pairs, 2000):
            break
    out = []
    per_bin = max(1, n_pairs // 3)
    for name in ["short","mid","long"]:
        pool = bins[name]
        if not pool: continue
        choose = min(per_bin, len(pool))
        idx = rng.choice(len(pool), size=choose, replace=False)
        out.extend([pool[k] for k in idx])
    flat = bins["short"] + bins["mid"] + bins["long"]
    if len(out) < n_pairs and flat:
        need = min(n_pairs - len(out), len(flat))
        idx = rng.choice(len(flat), size=need, replace=False)
        out.extend([flat[k] for k in idx])
    ii = np.asarray([x[0] for x in out], dtype=np.int64)
    jj = np.asarray([x[1] for x in out], dtype=np.int64)
    dd = np.asarray([x[2] for x in out], dtype=np.float32)
    return ii,jj,dd


def _episode_sequences(node_ids, episode_id, t_in_ep):
    groups = defaultdict(list)
    for gid in np.asarray(node_ids, dtype=np.int64).tolist():
        groups[int(episode_id[gid])].append((int(t_in_ep[gid]), int(gid)))
    out = []
    for _, arr in groups.items():
        arr.sort(key=lambda x: x[0])
        out.append(np.asarray([x[1] for x in arr], dtype=np.int64))
    return out


def train_replay_metric_head(h, s, replay_dist, node_ids, geo_dim, hidden_dim, lr, epochs, batch_pairs, device, seed, canon_weight=0.0):
    rng = np.random.default_rng(seed)
    model = GeoEncoder(h.shape[1], s.shape[1], geo_dim=geo_dim, hidden_dim=hidden_dim).to(device)
    scale = torch.nn.Parameter(torch.tensor(0.1, device=device))
    opt = torch.optim.Adam(list(model.parameters()) + [scale], lr=lr)
    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    for _ in range(int(epochs)):
        ii,jj,d_np = _sample_pairs_by_bins(replay_dist, node_ids, batch_pairs, int(rng.integers(1<<31)))
        if len(ii) < 8:
            continue
        g_i = model(h_t[ii], s_t[ii])
        g_j = model(h_t[jj], s_t[jj])
        d_lat = torch.norm(g_i-g_j, dim=-1)
        d_t = torch.tensor(d_np, dtype=torch.float32, device=device)
        w = 1.0 / torch.sqrt(d_t + 1.0)
        loss_metric = (w * (d_lat - scale.clamp(min=1e-4) * d_t).pow(2)).mean()
        g_pair = torch.cat([g_i, g_j], dim=0)
        sq = torch.cdist(g_pair, g_pair).pow(2)
        sq = sq.masked_fill(torch.eye(sq.size(0), device=device, dtype=torch.bool), 1e9)
        loss_uni = torch.logsumexp(-2.0 * sq, dim=1).mean()
        loss = loss_metric + 0.05 * loss_uni
        if canon_weight > 0:
            pos_idx = jj
            d_ap = torch.norm(g_i - model(h_t[pos_idx], s_t[pos_idx]), dim=-1)
            loss = loss + float(canon_weight) * d_ap.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model, float(scale.detach().cpu().item())


def train_temporal_head(h, s, episode_id, t_in_ep, node_ids, geo_dim, hidden_dim, lr, epochs, batch_size, seq_len, device, seed):
    rng = np.random.default_rng(seed)
    model = GeoEncoder(h.shape[1], s.shape[1], geo_dim=geo_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sequences = [seq for seq in _episode_sequences(node_ids, episode_id, t_in_ep) if len(seq) >= max(4, seq_len)]
    if not sequences:
        return model
    h_t = torch.tensor(h, dtype=torch.float32, device=device)
    s_t = torch.tensor(s, dtype=torch.float32, device=device)
    for _ in range(int(epochs)):
        batch_seqs = []
        for _ in range(int(batch_size)):
            seq = sequences[int(rng.integers(len(sequences)))]
            start = 0 if len(seq) == seq_len else int(rng.integers(0, len(seq) - seq_len + 1))
            batch_seqs.append(seq[start:start+seq_len])
        idx = np.stack(batch_seqs, axis=0)
        g_seq = model(h_t[idx], s_t[idx])
        loss = temporal_reachability_loss(g_seq, pos_k=3, neg_k=min(12, seq_len - 2), margin=0.6)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    return model


class TeacherStructModule:
    def __init__(self, deter_dim, stoch_dim, device, args):
        self.device = device
        self.args = args
        self.ready = False
        self.last_update_step = -10**9
        self.replay_head = None
        self.temporal_head = None
        self.replay_scale = 1.0
        self.node_replay = None
        self.node_temp = None
        self.node_phi = None
        self.node_conf = None
        self.graph = None
        self.teacher_data = None
        self.mode_hist = {}
        self.teacher_quality = {}

    def maybe_refresh(self, total_steps, env, encoder, rssm, actor, device, bit_depth):
        if total_steps - self.last_update_step < self.args.struct_update_interval:
            return False
        encoder.eval(); rssm.eval()
        data, mode_hist = collect_teacher_data_with_quotas(
            env, encoder, rssm, actor, device, bit_depth,
            teacher_collect_episodes=self.args.teacher_collect_episodes,
            max_nodes=self.args.teacher_max_nodes,
            teacher_high_noise=self.args.teacher_high_noise,
            teacher_random_fraction=self.args.teacher_random_fraction,
            teacher_subset_fraction=self.args.teacher_subset_fraction,
            teacher_start_subset_size=self.args.teacher_start_subset_size,
            state_stride=self.args.struct_state_stride,
        )
        graph = build_multiview_replay_graph(
            data,
            graph_views=[x.strip() for x in self.args.graph_views.split(",") if x.strip()],
            graph_knn_k=self.args.graph_knn_k,
            graph_knn_weight=self.args.graph_knn_weight,
            graph_temporal_weight=self.args.graph_temporal_weight,
            graph_knn_max_percentile=self.args.graph_knn_max_percentile,
            graph_same_ep_gap=self.args.graph_same_ep_gap,
            max_graph_nodes=self.args.max_graph_nodes,
            graph_min_view_votes=self.args.graph_min_view_votes,
        )
        gc = graph["giant_nodes_local"]
        if len(gc) < 32:
            self.ready = False
            self.teacher_quality = {"coverage": float(len(np.unique(data["cell_idx"])) / max(int(env.geodesic.n_free),1)), "giant_component_fraction": float(graph["stats"]["giant_component_fraction"]), "geo_vs_replay": {"spearman": 0.0}, "topology_ok": False}
            self.last_update_step = total_steps
            return False
        self.replay_head, self.replay_scale = train_replay_metric_head(
            h=data["h"][graph["node_indices"]],
            s=data["s"][graph["node_indices"]],
            replay_dist=graph["dist_mat"],
            node_ids=gc,
            geo_dim=self.args.struct_geo_dim,
            hidden_dim=self.args.struct_geo_hidden,
            lr=self.args.struct_geo_lr,
            epochs=self.args.struct_replay_head_epochs,
            batch_pairs=self.args.struct_replay_head_batch_pairs,
            device=self.device,
            seed=self.args.seed + int(total_steps),
            canon_weight=self.args.struct_canon_weight,
        )
        self.temporal_head = train_temporal_head(
            h=data["h"], s=data["s"], episode_id=data["episode_id"], t_in_ep=data["t_in_ep"],
            node_ids=graph["giant_nodes_global"], geo_dim=self.args.struct_geo_dim, hidden_dim=self.args.struct_geo_hidden,
            lr=self.args.struct_geo_lr, epochs=self.args.struct_temp_head_epochs, batch_size=self.args.struct_temp_head_batch_size,
            seq_len=self.args.struct_temp_seq_len, device=self.device, seed=self.args.seed + int(total_steps) + 7,
        )
        with torch.no_grad():
            h_t = torch.tensor(data["h"][graph["node_indices"]], dtype=torch.float32, device=self.device)
            s_t = torch.tensor(data["s"][graph["node_indices"]], dtype=torch.float32, device=self.device)
            self.node_replay = self.replay_head(h_t, s_t).detach().cpu().numpy().astype(np.float32)
            self.node_temp = self.temporal_head(h_t, s_t).detach().cpu().numpy().astype(np.float32)
        conf = graph["node_conf"].copy().astype(np.float32)
        if conf.max() > 0:
            conf = conf / max(conf.max(), 1e-6)
        cell_counts = defaultdict(int)
        for c in data["cell_idx"][graph["node_indices"]].tolist():
            cell_counts[int(c)] += 1
        frontier = np.asarray([1.0 / np.sqrt(cell_counts[int(c)] + 1.0) for c in data["cell_idx"][graph["node_indices"]]], dtype=np.float32)
        # disagreement by landmark profiles on giant component
        gcl = graph["giant_nodes_local"]
        g_rep = self.node_replay[gcl]
        g_tmp = self.node_temp[gcl]
        rng = np.random.default_rng(self.args.seed)
        lm = [int(rng.integers(len(gcl)))]
        while len(lm) < min(self.args.struct_n_landmarks, len(gcl)):
            dmin = np.min(np.linalg.norm(g_rep[:,None,:] - g_rep[np.asarray(lm)][None,:,:], axis=-1), axis=1)
            cand = int(np.argmax(dmin))
            if cand in lm: break
            lm.append(cand)
        lm = np.asarray(lm, dtype=np.int64)
        rep_prof = np.linalg.norm(self.node_replay[:,None,:] - self.node_replay[gcl[lm]][None,:,:], axis=-1)
        tmp_prof = np.linalg.norm(self.node_temp[:,None,:] - self.node_temp[gcl[lm]][None,:,:], axis=-1)
        rep_prof = (rep_prof - rep_prof.mean(axis=1, keepdims=True)) / (rep_prof.std(axis=1, keepdims=True) + 1e-6)
        tmp_prof = (tmp_prof - tmp_prof.mean(axis=1, keepdims=True)) / (tmp_prof.std(axis=1, keepdims=True) + 1e-6)
        disagree = np.mean(np.abs(rep_prof - tmp_prof), axis=1).astype(np.float32)
        conf_pow = np.power(np.clip(conf, 0.0, 1.0), float(max(self.args.struct_conf_power, 1e-6)))
        self.node_phi = conf_pow * (self.args.struct_lambda_front * frontier + self.args.struct_lambda_disagree * disagree)
        self.node_conf = conf
        self.graph = graph
        self.teacher_data = data
        self.mode_hist = mode_hist
        # diagnostics
        coverage = float(len(np.unique(data["cell_idx"])) / max(int(env.geodesic.n_free), 1))
        ii,jj,rd = _sample_pairs_by_bins(graph["dist_mat"], gcl, min(1200, max(200, len(gcl)*2)), seed=self.args.seed + 99)
        cells = data["cell_idx"][graph["node_indices"]]
        gd = np.asarray([float(env.geodesic.dist_matrix[int(cells[i]), int(cells[j])]) for i,j in zip(ii,jj)], dtype=np.float32)
        finite = np.isfinite(gd) & np.isfinite(rd) & (gd > 0) & (rd > 0)
        rho = _safe_spearman(gd[finite], rd[finite]) if finite.sum() >= 10 else 0.0
        self.teacher_quality = {
            "coverage": coverage,
            "visited_cells": int(len(np.unique(data["cell_idx"]))),
            "giant_component_fraction": float(graph["stats"]["giant_component_fraction"]),
            "geo_vs_replay": {"spearman": float(rho), "n": int(finite.sum())},
            "topology_ok": bool(coverage >= self.args.teacher_min_coverage and graph["stats"]["giant_component_fraction"] >= self.args.teacher_min_giant_fraction and rho >= self.args.teacher_min_geo_replay_spearman),
        }
        self.ready = True
        self.last_update_step = total_steps
        return True

    @torch.no_grad()
    def intrinsic_reward_for_imagination(self, h_imag, s_imag):
        if not self.ready or self.node_replay is None:
            B = h_imag.size(0)
            H = h_imag.size(1) - 1
            return torch.zeros((B, H), dtype=torch.float32, device=h_imag.device), {}
        B, T1, Dh = h_imag.shape
        Ds = s_imag.size(-1)
        g_rep = self.replay_head(h_imag.reshape(-1, Dh), s_imag.reshape(-1, Ds)).detach().cpu().numpy().astype(np.float32)
        dm = np.linalg.norm(g_rep[:,None,:] - self.node_replay[None,:,:], axis=-1)
        nn_idx = np.argmin(dm, axis=1)
        nn_dist = dm[np.arange(dm.shape[0]), nn_idx].reshape(B, T1)
        nn_idx = nn_idx.reshape(B, T1)
        phi = self.node_phi[nn_idx]
        conf = self.node_conf[nn_idx]
        phi_gain = phi[:,1:] - phi[:,:-1]
        conf_next = conf[:,1:]
        r = conf_next * phi_gain - float(self.args.struct_off_weight) * nn_dist[:,1:]
        stats = {
            "off_manifold_mean": float(np.mean(nn_dist[:,1:])),
            "phi_gain_mean": float(np.mean(phi_gain)),
            "phi_gain_pos_frac": float(np.mean(phi_gain > 0)),
            "conf_mean": float(np.mean(conf_next)),
        }
        return torch.tensor(r, dtype=torch.float32, device=h_imag.device), stats



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
                   choices=["baseline", "vicreg", "straight", "georeg", "georeg_cov", "struct_explore"],
                   help="baseline | vicreg | straight | georeg (both) | georeg_cov (both + coverage) | struct_explore")
    p.add_argument("--vicreg_weight", type=float, default=0.1,
                   help="Weight of VICReg loss added to world model loss")
    p.add_argument("--vicreg_var_w", type=float, default=25.0)
    p.add_argument("--vicreg_inv_w", type=float, default=25.0)
    p.add_argument("--vicreg_cov_w", type=float, default=1.0)
    p.add_argument("--vicreg_target_std", type=float, default=1.0)
    p.add_argument("--straight_weight", type=float, default=0.5,
                   help="Weight of straightness loss added to world model loss")
    p.add_argument("--coverage_weight", type=float, default=0.01,
                   help="Weight of latent coverage intrinsic reward")
    p.add_argument("--coverage_memory_size", type=int, default=4096)
    p.add_argument("--coverage_knn_k", type=int, default=5)



    # Structured exploration via teacher graph + replay/temporal heads
    p.add_argument("--struct_geo_weight", type=float, default=0.1,
                   help="Weight of structured graph bonus in imagination")
    p.add_argument("--struct_ngu_weight", type=float, default=0.05,
                   help="Weight of NGU-style novelty bonus alongside structured bonus")
    p.add_argument("--struct_update_interval", type=int, default=5000,
                   help="Refresh teacher graph and heads every N env steps")
    p.add_argument("--teacher_collect_episodes", type=int, default=80)
    p.add_argument("--teacher_max_nodes", type=int, default=3500)
    p.add_argument("--teacher_high_noise", type=float, default=0.45)
    p.add_argument("--teacher_random_fraction", type=float, default=0.4)
    p.add_argument("--teacher_subset_fraction", type=float, default=0.3)
    p.add_argument("--teacher_start_subset_size", type=int, default=6)
    p.add_argument("--teacher_min_coverage", type=float, default=0.45)
    p.add_argument("--teacher_min_giant_fraction", type=float, default=0.3)
    p.add_argument("--teacher_min_geo_replay_spearman", type=float, default=0.45)
    p.add_argument("--struct_state_stride", type=int, default=2)
    p.add_argument("--graph_views", type=str, default="encoder,z")
    p.add_argument("--graph_min_view_votes", type=int, default=1)
    p.add_argument("--graph_knn_k", type=int, default=6)
    p.add_argument("--graph_knn_weight", type=float, default=1.0)
    p.add_argument("--graph_temporal_weight", type=float, default=1.0)
    p.add_argument("--graph_knn_max_percentile", type=float, default=80.0)
    p.add_argument("--graph_same_ep_gap", type=int, default=2)
    p.add_argument("--max_graph_nodes", type=int, default=1800)
    p.add_argument("--struct_geo_dim", type=int, default=32)
    p.add_argument("--struct_geo_hidden", type=int, default=256)
    p.add_argument("--struct_geo_lr", type=float, default=3e-4)
    p.add_argument("--struct_replay_head_epochs", type=int, default=200)
    p.add_argument("--struct_replay_head_batch_pairs", type=int, default=512)
    p.add_argument("--struct_temp_head_epochs", type=int, default=200)
    p.add_argument("--struct_temp_head_batch_size", type=int, default=32)
    p.add_argument("--struct_temp_seq_len", type=int, default=12)
    p.add_argument("--struct_canon_weight", type=float, default=0.25)
    p.add_argument("--struct_lambda_front", type=float, default=1.0)
    p.add_argument("--struct_lambda_disagree", type=float, default=0.5)
    p.add_argument("--struct_conf_power", type=float, default=1.0)
    p.add_argument("--struct_off_weight", type=float, default=0.0)
    p.add_argument("--struct_n_landmarks", type=int, default=16)

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

    use_vicreg = args.geo_mode in ("vicreg", "georeg", "georeg_cov")
    use_straight = args.geo_mode in ("straight", "georeg", "georeg_cov")
    use_coverage = args.geo_mode == "georeg_cov"
    use_struct = args.geo_mode == "struct_explore"

    print(f"Device: {device}")
    print(f"Geo mode: {args.geo_mode}")
    print(f"  VICReg: {use_vicreg} (weight={args.vicreg_weight})")
    print(f"  Straightness: {use_straight} (weight={args.straight_weight})")
    print(f"  Coverage bonus: {use_coverage} (weight={args.coverage_weight})")
    print(f"  Structured explore: {use_struct} (geo={args.struct_geo_weight}, ngu={args.struct_ngu_weight})")

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

    if args.wm_path:
        ckpt = torch.load(args.wm_path, map_location=device)
        for k in ("encoder", "decoder", "rssm", "reward_model", "cont_model"):
            if k in ckpt:
                locals()[k].load_state_dict(ckpt[k])
        print(f"  Loaded world model from {args.wm_path}")

    actor = Actor(args.deter_dim, args.stoch_dim, act_dim, args.actor_hidden_dim).to(device)
    value_model = ValueModel(args.deter_dim, args.stoch_dim, args.value_hidden_dim).to(device)

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

    # Coverage memory bank (for novelty bonuses)
    coverage_memory: torch.Tensor | None = None
    coverage_ptr = 0

    struct_module = TeacherStructModule(args.deter_dim, args.stoch_dim, device, args) if use_struct else None

    # Observation + position buffers for geodesic correlation eval
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

        while not done:
            encoder.eval(); rssm.eval(); actor.eval()
            with torch.no_grad():
                action_t, _ = actor.get_action(h_state, s_state, deterministic=False)
                if expl_amount > 0:
                    action_t = action_t + expl_amount * torch.randn_like(action_t)
                    action_t = torch.clamp(action_t, -1.0, 1.0)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            next_obs, r, term, trunc, step_info = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            replay.add(obs=np.ascontiguousarray(obs, np.uint8), action=action,
                       reward=float(r), next_obs=np.ascontiguousarray(next_obs, np.uint8), done=done)
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
                if use_struct and struct_module is not None:
                    try:
                        refreshed = struct_module.maybe_refresh(total_steps, env, encoder, rssm, actor, device, args.bit_depth)
                        if refreshed:
                            tq = struct_module.teacher_quality
                            print(f"    [struct refresh] coverage={tq.get('coverage', 0):.2f} giant={tq.get('giant_component_fraction', 0):.2f} geo_vs_replay={tq.get('geo_vs_replay', {}).get('spearman', 0):.3f}")
                    except Exception as e:
                        print(f"    [struct refresh failed] {e}")
                encoder.train(); decoder.train(); rssm.train()
                reward_model.train(); cont_model.train()
                actor.train(); value_model.train()

                sum_rec = sum_kld = sum_rew = sum_cont = sum_model = 0.0
                sum_actor = sum_value = sum_imag_r = 0.0
                sum_vicreg = sum_straight = sum_coverage = 0.0
                sum_struct = sum_ngu = 0.0
                vicreg_info_accum: dict[str, float] = {}
                straight_info_accum: dict[str, float] = {}
                struct_stats_accum: dict[str, float] = {}

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

                    # ---- VICReg loss on Dreamer state z=[h,s] ----
                    l_vicreg = torch.tensor(0.0, device=device)
                    if use_vicreg:
                        z_seq = rssm_latent(h_seq, s_seq)
                        z_anchor = z_seq[:, :-1].reshape(-1, z_seq.shape[-1])
                        z_positive = z_seq[:, 1:].reshape(-1, z_seq.shape[-1])
                        # Subsample to keep cost bounded
                        n_vic = min(512, z_anchor.shape[0])
                        vic_idx = torch.randperm(z_anchor.shape[0], device=device)[:n_vic]
                        l_vicreg, vic_info = vicreg_loss(
                            z_anchor[vic_idx], z_positive[vic_idx],
                            var_weight=args.vicreg_var_w,
                            inv_weight=args.vicreg_inv_w,
                            cov_weight=args.vicreg_cov_w,
                            target_std=args.vicreg_target_std,
                        )
                        model_loss = model_loss + args.vicreg_weight * l_vicreg
                        vic_info = {k.replace("vicreg/", "vicreg_hs/"): v for k, v in vic_info.items()}
                        for k, v in vic_info.items():
                            vicreg_info_accum[k] = vicreg_info_accum.get(k, 0.0) + v

                    # ---- Straightness loss ----
                    l_straight = torch.tensor(0.0, device=device)
                    if use_straight:
                        # Build mask: 1 where same episode (no done between steps)
                        mask = (1.0 - done_seq[:, :T]).cumprod(dim=1)
                        # Use encoder embeddings as the "trajectory" in latent space
                        # e_t is [B, T+1, D], take the subsequence [B, T, D]
                        e_traj = e_t[:, :T]
                        l_straight, str_info = straightness_loss(e_traj, mask)
                        model_loss = model_loss + args.straight_weight * l_straight
                        for k, v in str_info.items():
                            straight_info_accum[k] = straight_info_accum.get(k, 0.0) + v

                    model_opt.zero_grad(set_to_none=True)
                    model_loss.backward()
                    torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip)
                    model_opt.step()

                    sum_rec += float(rec_loss.item())
                    sum_kld += float(kld.item())
                    sum_rew += float(rew_loss.item())
                    sum_cont += float(cont_loss.item())
                    sum_model += float(model_loss.item())
                    sum_vicreg += float(l_vicreg.item())
                    sum_straight += float(l_straight.item())

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

                    # ---- Structured / novelty intrinsic rewards ----
                    rewards_total = rewards_imag
                    z_imag_flat = rssm_latent(h_imag[:, 1:].reshape(-1, Dh),
                                              s_imag[:, 1:].reshape(-1, Ds))
                    cov_bonus = None
                    if use_coverage or use_struct:
                        with torch.no_grad():
                            buf_len = (
                                min(len(z_imag_flat), args.coverage_memory_size)
                                if coverage_memory is None
                                else coverage_memory.shape[0]
                            )
                            if coverage_memory is None:
                                coverage_memory = z_imag_flat[:buf_len].detach().clone()
                                coverage_ptr = buf_len
                            cov_bonus = latent_coverage_bonus(
                                z_imag_flat.detach(), coverage_memory, k=args.coverage_knn_k
                            ).reshape(rewards_imag.shape)
                            cov_bonus = cov_bonus / (cov_bonus.std().clamp(min=1e-4))
                            n_new = min(len(z_imag_flat), buf_len)
                            new_states = z_imag_flat.detach()[:n_new]
                            for i in range(n_new):
                                coverage_memory[coverage_ptr % buf_len] = new_states[i]
                                coverage_ptr += 1
                    if use_struct and struct_module is not None and struct_module.ready:
                        struct_bonus, struct_stats = struct_module.intrinsic_reward_for_imagination(h_imag, s_imag)
                        rewards_total = rewards_total + args.struct_geo_weight * struct_bonus
                        sum_struct += float(struct_bonus.mean().item())
                        for k, v in struct_stats.items():
                            struct_stats_accum[k] = struct_stats_accum.get(k, 0.0) + float(v)
                        if cov_bonus is not None and args.struct_ngu_weight > 0:
                            rewards_total = rewards_total + args.struct_ngu_weight * cov_bonus
                            sum_ngu += float(cov_bonus.mean().item())
                    elif use_coverage and cov_bonus is not None:
                        rewards_total = rewards_total + args.coverage_weight * cov_bonus
                        sum_coverage += float(cov_bonus.mean().item())

                    sum_imag_r += float(rewards_imag.mean().item())

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

                if use_vicreg:
                    writer.add_scalar("loss/vicreg_total", sum_vicreg / n_ts, total_steps)
                    writer.add_scalar("loss/vicreg_weighted", args.vicreg_weight * sum_vicreg / n_ts, total_steps)
                    for k, v in vicreg_info_accum.items():
                        writer.add_scalar(f"loss/{k}", v / n_ts, total_steps)
                if use_straight:
                    writer.add_scalar("loss/straight_total", sum_straight / n_ts, total_steps)
                    writer.add_scalar("loss/straight_weighted", args.straight_weight * sum_straight / n_ts, total_steps)
                    for k, v in straight_info_accum.items():
                        writer.add_scalar(f"loss/{k}", v / n_ts, total_steps)
                if use_coverage:
                    writer.add_scalar("loss/coverage_bonus_mean", sum_coverage / n_ts, total_steps)
                if use_struct:
                    writer.add_scalar("loss/struct_bonus_mean", sum_struct / n_ts, total_steps)
                    writer.add_scalar("loss/struct_ngu_bonus_mean", sum_ngu / n_ts, total_steps)
                    if struct_module is not None and struct_module.ready:
                        tq = struct_module.teacher_quality
                        writer.add_scalar("struct/teacher_coverage", float(tq.get("coverage", 0.0)), total_steps)
                        writer.add_scalar("struct/teacher_giant_fraction", float(tq.get("giant_component_fraction", 0.0)), total_steps)
                        writer.add_scalar("struct/teacher_geo_vs_replay_spearman", float(tq.get("geo_vs_replay", {}).get("spearman", 0.0)), total_steps)
                        for k, v in struct_stats_accum.items():
                            writer.add_scalar(f"struct/{k}", v / n_ts, total_steps)

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
              f"cells={len(ep_cells)}  total={total_steps}  episode_success={ep_success}")

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
            torch.save({
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
            }, ckpt_path)

    # ---- Final save ----
    torch.save({
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
    }, os.path.join(out_dir, "world_model_final.pt"))

    env.close()
    writer.close()
    print(f"\nDone. Mode={args.geo_mode}  total_steps={total_steps}")


if __name__ == "__main__":
    main(build_parser().parse_args())
