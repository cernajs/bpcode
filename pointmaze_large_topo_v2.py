#!/usr/bin/env python3

import argparse
import json
import os
import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import gymnasium as gym
import gymnasium_robotics  # type: ignore
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from maze_env import GeodesicComputer
from maze_geometry_test import (
    TrainCfg,
    train_world_model,
    collect_data,
    train_geo_encoder,
    train_geo_encoder_geodesic,
    _build_feature_dict,
    _latent_indexed_l2,
    run_probes,
    run_distance_analysis,
    run_knn_analysis,
    run_trustworthiness_continuity,
    generate_plots,
    compute_sanity_metrics,
    _positions_to_cell_indices,
    _adj_from_distmat,
    _find_bridges,
    _components_without_bridges,
)
from utils import get_device, set_seed
from models import (
    RSSM,
    ContinueModel,
    ConvDecoder,
    ConvEncoder,
    RewardModel,
)
from geom_head import GeoEncoder

from pointmaze_gr_geometry_test_topo import (
    TemporalTopoHead,
    ReplayContrastiveHead,
    ReplayGraphDiscreteHead,
    ReplayNodeScoreHead,
    ReplayEdgeTopoClassifier,
    _mixed_graph_undirected_edges,
    _sample_nonedge_pairs_local,
    _oracle_free_replay_step_distances,
    _make_temporal_dist_mlp,
    _build_mixed_replay_graph,
    _build_temporal_positive_table,
    _predict_pair_ensemble_variance_batch,
    _node_betweenness_brandes_undirected,
    _edge_betweenness_brandes_undirected,
    _edge_betweenness_push_to_nodes,
    _articulation_points_undirected,
    _cell_replay_adjacency,
    _zscore_safe,
    _zscore_1d_nonconst,
    _all_pairs_shortest_paths_bfs,
    _connected_components_from_adj,
    _temporal_only_adj_local,
    _kmeans_numpy,
    _spectral_community_labels,
    _spearman_rho_numpy,
    _compute_encoder_temporal_local_global_disagreement,
    _compute_encoder_temporal_barrier_scores,
    plot_temporal_dist_error_map,
    plot_temporal_dist_error_scatter,
    plot_encoder_temporal_barrier_heatmap,
    plot_encoder_temporal_local_global_disagreement_heatmap,
    plot_laplacian_kmeans_grid,
)

# ---------------------------------------------------------------------------
# Geodesic for PointMaze_Large_Diverse_GR-v3 
# ---------------------------------------------------------------------------

LARGE_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, "C", 0, 0, 0, 1, "C", 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, "C", 0, 1, 0, 0, "C", 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, "C", 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, "C", 0, "C", 1, 0, "C", 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]


def make_pointmaze_large_gr_geodesic() -> GeodesicComputer:
    grid = []
    for row in LARGE_MAZE_DIVERSE_GR:
        grid_row = []
        for cell in row:
            grid_row.append("1" if cell == 1 else "0")
        grid.append(grid_row)
    return GeodesicComputer(grid)


# ---------------------------------------------------------------------------
# Canonical oracle bridge computation (Fix 9: single consistent source)
# ---------------------------------------------------------------------------

def _canonical_oracle_bridges(geodesic: GeodesicComputer) -> tuple[set, list]:
    """Single authoritative bridge set from the geodesic distance matrix.

    Returns (bridge_cell_set, bridge_edge_list) where bridge_cell_set contains
    all cell indices that are endpoints of at least one bridge edge.
    """
    adj_oracle = _adj_from_distmat(geodesic.dist_matrix)
    bridges = _find_bridges(adj_oracle)
    bridge_cells: set[int] = set()
    for u, v in bridges:
        bridge_cells.add(int(u))
        bridge_cells.add(int(v))
    return bridge_cells, bridges


# ---------------------------------------------------------------------------
# Wrapper: Gymnasium-Robotics PointMaze_Large -> Dreamer-compatible pixel env
# ---------------------------------------------------------------------------

class PointMazeLargeDiverseGRWrapper:
    def __init__(self, env_name: str = "PointMaze_Large_Diverse_GR-v3", img_size: int = 64):
        gym.register_envs(gymnasium_robotics)
        self._env = gym.make(env_name, render_mode="rgb_array")
        self.img_size = int(img_size)

        # Geodesic / free cells before first reset so we can randomize start+goal like the medium test.
        self.geodesic = make_pointmaze_large_gr_geodesic()
        self.grid_h = len(self.geodesic.grid)
        self.grid_w = len(self.geodesic.grid[0])
        # Free cells (row, col) for random reset — ensures coverage across the maze.
        self._free_cells = list(self.geodesic.idx_to_cell)

        obs_dict, _ = self._env.reset(**self._random_reset_kwargs())
        frame = self._env.render()
        assert isinstance(frame, np.ndarray) and frame.ndim == 3

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8
        )
        self.action_space = self._env.action_space

        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._update_agent_pos(obs_dict)

    @property
    def agent_pos(self) -> np.ndarray:
        return self._agent_pos.copy()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[0] == self.img_size and frame.shape[1] == self.img_size:
            return frame
        return cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

    def _world_to_grid(self, xy):
        x, y = float(xy[0]), float(xy[1])
        gx = x + (self.grid_w / 2 - 0.5)
        gy = (self.grid_h / 2 - 0.5) - y
        return np.array([gx, gy], dtype=np.float32)

    def _update_agent_pos(self, obs_dict):
        ag = obs_dict.get("achieved_goal", None)
        if ag is None:
            obs = obs_dict.get("observation", None)
            if obs is not None and len(obs) >= 2:
                ag = obs[:2]
        if ag is not None:
            self._agent_pos = self._world_to_grid(ag[:2])

    def _random_reset_kwargs(self) -> dict:
        """Options for gymnasium reset: random start + goal on free cells (PointMaze API)."""
        if not self._free_cells:
            return {}
        n = len(self._free_cells)
        i0 = int(np.random.randint(0, n))
        i1 = int(np.random.randint(0, n))
        if n > 1 and i1 == i0:
            i1 = (i0 + 1) % n
        r0, c0 = self._free_cells[i0]
        r1, c1 = self._free_cells[i1]
        return {
            "options": {
                "reset_cell": np.array([r0, c0], dtype=np.int64),
                "goal_cell": np.array([r1, c1], dtype=np.int64),
            }
        }

    def reset(self, **kwargs):
        # Start agent at a random free cell each episode for full maze coverage (+ random goal).
        if self._free_cells:
            options = kwargs.get("options") or {}
            options = dict(options)
            rnd = self._random_reset_kwargs().get("options") or {}
            options["reset_cell"] = rnd["reset_cell"]
            options["goal_cell"] = rnd["goal_cell"]
            kwargs = dict(kwargs)
            kwargs["options"] = options
        obs_dict, info = self._env.reset(**kwargs)
        self._update_agent_pos(obs_dict)
        frame = self._resize_frame(self._env.render()).astype(np.uint8)
        return frame, info

    def step(self, action, repeat: int = 1):
        total_reward = 0.0
        terminated = truncated = False
        info = {}
        for _ in range(int(repeat)):
            obs_dict, r, t, tr, info = self._env.step(action)
            self._update_agent_pos(obs_dict)
            total_reward += float(r)
            terminated, truncated = bool(t), bool(tr)
            if terminated or truncated:
                break
        frame = self._resize_frame(self._env.render()).astype(np.uint8)
        return frame, total_reward, terminated, truncated, info

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------

@dataclass
class PointMazeLargeRunCfg:
    seed: int = 0
    output_dir: str = "pointmaze_large_gr_results"
    quick: bool = False
    geo_supervised: bool = False
    wm_path: str = ""
    # existing heads
    replay_topology: bool = True
    replay_topology_dim: int = 2
    replay_topology_hidden: int = 256
    replay_topology_epochs: int = 1000
    replay_topology_batch_pairs: int = 512
    replay_topology_pair_pool: int = 8000
    replay_topology_val_frac: float = 0.15
    replay_topology_n_ensemble: int = 5
    replay_laplacian: bool = False
    replay_laplacian_dim: int = 3
    replay_laplacian_graph_max: int = 1800
    replay_laplacian_knn_k: int = 10
    replay_cont: bool = False
    replay_cont_dim: int = 16
    replay_cont_hidden: int = 256
    replay_cont_epochs: int = 700
    replay_cont_batch: int = 512
    replay_cont_val_frac: float = 0.15
    replay_cont_pos_k: int = 3
    replay_cont_neg_k: int = 32
    replay_cont_temp: float = 0.10
    replay_graph: bool = False
    replay_graph_disc_classes: int = 64
    replay_graph_hidden: int = 256
    replay_graph_epochs: int = 400
    replay_graph_batch_edges: int = 512
    replay_graph_graph_max: int = 1800
    replay_graph_knn_k: int = 10
    replay_graph_val_frac: float = 0.1
    replay_graph_lambda_smooth: float = 1.0
    replay_graph_lambda_neg: float = 0.15
    replay_graph_lambda_edge: float = 0.5
    replay_node_score: bool = True
    replay_node_score_hidden: int = 256
    replay_node_score_epochs: int = 300
    replay_node_score_batch: int = 1024
    replay_node_score_val_frac: float = 0.15
    replay_node_score_alpha: float = 1.0
    replay_node_score_beta: float = 0.0
    replay_node_score_gamma: float = 1.0
    replay_node_score_delta: float = 1.0
    replay_edge_topo: bool = True
    replay_edge_topo_hidden: int = 256
    replay_edge_topo_epochs: int = 300
    replay_edge_topo_batch: int = 1024
    replay_edge_topo_val_frac: float = 0.15
    replay_edge_topo_bottleneck_frac: float = 0.1
    # --- optional heads (off unless CLI enables) ---
    replay_sr: bool = False
    replay_sr_dim: int = 8
    replay_sr_hidden: int = 256
    replay_sr_epochs: int = 600
    replay_sr_batch: int = 512
    replay_sr_val_frac: float = 0.15
    replay_sr_gamma: float = 0.95
    replay_hit: bool = False
    replay_hit_dim: int = 8
    replay_hit_hidden: int = 256
    replay_hit_epochs: int = 600
    replay_hit_batch: int = 512
    replay_hit_val_frac: float = 0.15
    replay_hit_n_walks: int = 200
    replay_hit_walk_len: int = 300
    replay_fiedler: bool = False
    replay_fiedler_dim: int = 16
    replay_fiedler_hidden: int = 256
    replay_fiedler_epochs: int = 700
    replay_fiedler_batch: int = 512
    replay_fiedler_val_frac: float = 0.15
    replay_fiedler_lambda_fiedler: float = 1.0
    replay_fiedler_pos_k: int = 3
    replay_fiedler_neg_k: int = 32
    replay_fiedler_temp: float = 0.10


# =====================================================================
# NEW HEAD 1: g_sr(e) — Successor Feature Head
# =====================================================================

class SuccessorFeatureHead(nn.Module):
    """Learns M(e) satisfying M(e_t) ≈ φ(e_t) + γ·M(e_{t+1}) via Bellman."""

    def __init__(self, embed_dim: int, sr_dim: int, hidden_dim: int):
        super().__init__()
        self.sr_dim = sr_dim
        self.phi_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, sr_dim),
        )
        self.m_net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, sr_dim),
        )
        for net in (self.phi_net, self.m_net):
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=0.5)
                    nn.init.zeros_(m.bias)

    def phi(self, e: torch.Tensor) -> torch.Tensor:
        return self.phi_net(e)

    def m_value(self, e: torch.Tensor) -> torch.Tensor:
        return self.m_net(e)

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        return self.m_value(e)


def train_replay_sr_head(
    data: dict,
    cfg: PointMazeLargeRunCfg,
    device: torch.device,
    seed: int,
):
    """Train successor feature head via Bellman regression on temporal chains."""
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or len(ep_ids) != len(encoder_emb):
        return None, None, None
    n = len(encoder_emb)
    if n < 16:
        return None, None, None

    same_ep = (ep_ids[:-1] == ep_ids[1:])
    trans_idx = np.where(same_ep)[0].astype(np.int64)
    if len(trans_idx) < 16:
        return None, None, None

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(trans_idx))
    n_val = max(1, int(round(cfg.replay_sr_val_frac * len(trans_idx))))
    train_trans = trans_idx[perm[n_val:]]
    val_trans = trans_idx[perm[:n_val]]

    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    gamma = float(cfg.replay_sr_gamma)
    model = SuccessorFeatureHead(
        embed_dim=int(encoder_emb.shape[1]),
        sr_dim=int(cfg.replay_sr_dim),
        hidden_dim=int(cfg.replay_sr_hidden),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = int(max(2, cfg.replay_sr_epochs))
    batch = int(min(max(32, cfg.replay_sr_batch), len(train_trans)))

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        idx = rng.choice(train_trans, size=batch, replace=len(train_trans) < batch)
        idx_t = torch.tensor(idx, dtype=torch.long, device=device)
        idx_next = idx_t + 1

        phi_t = model.phi(e_t[idx_t])
        m_t = model.m_value(e_t[idx_t])
        with torch.no_grad():
            m_next = model.m_value(e_t[idx_next])
        target = phi_t.detach() + gamma * m_next
        loss = F.mse_loss(m_t, target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            vi = torch.tensor(val_trans, dtype=torch.long, device=device)
            phi_v = model.phi(e_t[vi])
            m_v = model.m_value(e_t[vi])
            m_vn = model.m_value(e_t[vi + 1])
            vl = F.mse_loss(m_v, phi_v + gamma * m_vn)
            if float(vl.item()) < best_val:
                best_val = float(vl.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, epochs // 6) == 0:
            print(f"    g_sr epoch {epoch+1}/{epochs}  train={loss.item():.4f}  val={vl.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_all = model.encode(e_t).cpu().numpy().astype(np.float32)

    meta = {
        "n_transitions": int(len(trans_idx)),
        "n_train": int(len(train_trans)),
        "n_val": int(len(val_trans)),
        "epochs": epochs,
        "sr_dim": int(cfg.replay_sr_dim),
        "gamma": gamma,
        "best_val_bellman_mse": float(best_val),
        "feat_dict_key": "g_sr(e)",
    }
    return model, z_all, meta


# =====================================================================
# NEW HEAD 2: g_hit(e) — Hitting-Time Head
# =====================================================================

class HittingTimeHead(nn.Module):
    """Predict expected hitting times from encoder embeddings."""

    def __init__(self, embed_dim: int, hit_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.z_head = nn.Linear(hidden_dim, hit_dim)
        self.pair_mlp = nn.Sequential(
            nn.Linear(2 * hit_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.orthogonal_(self.z_head.weight, gain=0.1)
        nn.init.zeros_(self.z_head.bias)

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        return self.z_head(self.trunk(e))

    def predict_pair(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        z_i, z_j = self.encode(e_i), self.encode(e_j)
        return self.pair_mlp(torch.cat([z_i, z_j], dim=-1)).squeeze(-1)


def _compute_hitting_times_mc(
    adj_list: list[list[int]],
    rng: np.random.Generator,
    n_walks: int = 200,
    walk_len: int = 300,
    n_sources: int = 64,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Monte Carlo random-walk hitting times on an undirected graph.

    Returns (src_arr, dst_arr, ht_arr) of sampled mean hitting times.
    Hitting time h(s,t) = E[first passage from s to t under uniform random walk].
    """
    m = len(adj_list)
    if m < 4:
        return None
    degrees = np.array([len(adj_list[i]) for i in range(m)], dtype=np.int64)
    valid = np.where(degrees > 0)[0]
    if len(valid) < 4:
        return None

    sources = rng.choice(valid, size=min(n_sources, len(valid)), replace=False)
    targets = rng.choice(valid, size=min(n_sources, len(valid)), replace=False)

    src_list, dst_list, ht_list = [], [], []
    for s in sources:
        s = int(s)
        if degrees[s] == 0:
            continue
        for t in targets:
            t = int(t)
            if s == t or degrees[t] == 0:
                continue
            hits = []
            for _ in range(n_walks):
                cur = s
                for step in range(1, walk_len + 1):
                    nei = adj_list[cur]
                    if not nei:
                        break
                    cur = int(nei[rng.integers(0, len(nei))])
                    if cur == t:
                        hits.append(step)
                        break
            if len(hits) >= max(1, n_walks // 10):
                src_list.append(s)
                dst_list.append(t)
                ht_list.append(float(np.mean(hits)))

    if not ht_list:
        return None
    return (
        np.array(src_list, dtype=np.int64),
        np.array(dst_list, dtype=np.int64),
        np.array(ht_list, dtype=np.float32),
    )


def train_replay_hit_head(
    data: dict,
    cfg: PointMazeLargeRunCfg,
    device: torch.device,
    seed: int,
):
    """Train hitting-time regression head on MC-estimated random-walk times."""
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or len(ep_ids) != len(encoder_emb):
        return None, None, None

    rng = np.random.default_rng(int(seed))
    idx_global, _, adj_list = _build_mixed_replay_graph(
        encoder_emb, ep_ids,
        n_graph_max=int(cfg.replay_graph_graph_max),
        k_knn=int(cfg.replay_graph_knn_k),
    )
    m = len(adj_list)
    if m < 16:
        return None, None, None

    print(f"    g_hit: computing MC hitting times ({cfg.replay_hit_n_walks} walks, "
          f"len {cfg.replay_hit_walk_len}) on {m}-node mixed graph ...")
    ht_data = _compute_hitting_times_mc(
        adj_list, rng,
        n_walks=int(cfg.replay_hit_n_walks),
        walk_len=int(cfg.replay_hit_walk_len),
        n_sources=min(64, m),
    )
    if ht_data is None:
        print("    g_hit: insufficient hitting time pairs.")
        return None, None, None

    src_loc, dst_loc, ht_vals = ht_data
    src_glob = idx_global[src_loc]
    dst_glob = idx_global[dst_loc]
    n_pairs = len(ht_vals)
    print(f"    g_hit: {n_pairs} hitting-time pairs (mean ht={np.mean(ht_vals):.1f})")
    if n_pairs < 16:
        return None, None, None

    perm = rng.permutation(n_pairs)
    n_val = max(1, int(round(cfg.replay_hit_val_frac * n_pairs)))
    tr_mask = np.ones(n_pairs, dtype=bool)
    tr_mask[perm[:n_val]] = False

    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    log_ht = np.log1p(ht_vals).astype(np.float32)
    ht_t = torch.tensor(log_ht, dtype=torch.float32, device=device)

    model = HittingTimeHead(
        embed_dim=int(encoder_emb.shape[1]),
        hit_dim=int(cfg.replay_hit_dim),
        hidden_dim=int(cfg.replay_hit_hidden),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = int(max(2, cfg.replay_hit_epochs))
    batch = int(min(max(32, cfg.replay_hit_batch), int(np.sum(tr_mask))))

    tr_idx = np.where(tr_mask)[0]
    val_idx = np.where(~tr_mask)[0]

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        bi = rng.choice(tr_idx, size=batch, replace=len(tr_idx) < batch)
        si = torch.tensor(src_glob[bi], dtype=torch.long, device=device)
        di = torch.tensor(dst_glob[bi], dtype=torch.long, device=device)
        pred = model.predict_pair(e_t[si], e_t[di])
        loss = F.mse_loss(pred, ht_t[bi])

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            sv = torch.tensor(src_glob[val_idx], dtype=torch.long, device=device)
            dv = torch.tensor(dst_glob[val_idx], dtype=torch.long, device=device)
            vl = F.mse_loss(model.predict_pair(e_t[sv], e_t[dv]), ht_t[val_idx])
            if float(vl.item()) < best_val:
                best_val = float(vl.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, epochs // 6) == 0:
            print(f"    g_hit epoch {epoch+1}/{epochs}  train={loss.item():.4f}  val={vl.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_all = model.encode(e_t).cpu().numpy().astype(np.float32)

    var_targets = float(np.var(log_ht[val_idx])) if len(val_idx) > 1 else 1e-8
    r2 = 1.0 - best_val / max(var_targets, 1e-8)

    meta = {
        "n_pairs": n_pairs, "n_train": int(np.sum(tr_mask)), "n_val": int(np.sum(~tr_mask)),
        "epochs": epochs, "hit_dim": int(cfg.replay_hit_dim),
        "best_val_mse_log_ht": float(best_val), "val_r2": float(r2),
        "mc_n_walks": int(cfg.replay_hit_n_walks), "mc_walk_len": int(cfg.replay_hit_walk_len),
        "feat_dict_key": "g_hit(e)",
        "asymmetry_note": "hitting times are inherently asymmetric; h(s,t) != h(t,s)",
    }
    return model, z_all, meta


# =====================================================================
# NEW HEAD 3: g_fiedler(e) — Fiedler-Guided Contrastive Head
# =====================================================================

class FiedlerContrastiveHead(nn.Module):
    """InfoNCE contrastive embedding with auxiliary Fiedler vector regression."""

    def __init__(self, embed_dim: int, fiedler_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
        )
        self.z_head = nn.Linear(hidden_dim, fiedler_dim)
        self.fiedler_head = nn.Linear(hidden_dim, 1)
        nn.init.orthogonal_(self.z_head.weight, gain=0.1)
        nn.init.zeros_(self.z_head.bias)

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        return self.z_head(self.trunk(e))

    def encode_unit(self, e: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.encode(e), dim=-1, eps=1e-8)

    def predict_fiedler(self, e: torch.Tensor) -> torch.Tensor:
        return self.fiedler_head(self.trunk(e)).squeeze(-1)


def _compute_fiedler_vector(adj_list: list[list[int]], idx_global: np.ndarray) -> Optional[np.ndarray]:
    """2nd smallest eigenvector of the normalized Laplacian (Fiedler vector)."""
    m = len(adj_list)
    if m < 8:
        return None
    a = np.zeros((m, m), dtype=np.float64)
    for i, nei in enumerate(adj_list):
        for j in nei:
            if int(j) != i:
                a[i, int(j)] = 1.0
    deg = a.sum(axis=1)
    inv_sqrt = np.zeros(m, dtype=np.float64)
    nz = deg > 1e-12
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    d_inv = np.diag(inv_sqrt)
    ln = np.eye(m) - d_inv @ a @ d_inv
    try:
        evals, evecs = np.linalg.eigh(ln)
    except np.linalg.LinAlgError:
        return None
    if m < 3:
        return None
    return evecs[:, 1].astype(np.float32)


def train_replay_fiedler_head(
    data: dict,
    cfg: PointMazeLargeRunCfg,
    device: torch.device,
    seed: int,
):
    """Train Fiedler-guided contrastive head: InfoNCE + Fiedler vector regression."""
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or len(ep_ids) != len(encoder_emb):
        return None, None, None
    n = len(encoder_emb)
    if n < 16:
        return None, None, None

    rng = np.random.default_rng(int(seed))
    idx_global, g2l, adj_list = _build_mixed_replay_graph(
        encoder_emb, ep_ids,
        n_graph_max=int(cfg.replay_graph_graph_max),
        k_knn=int(cfg.replay_graph_knn_k),
    )
    fiedler_local = _compute_fiedler_vector(adj_list, idx_global)
    if fiedler_local is None:
        print("    g_fiedler: could not compute Fiedler vector.")
        return None, None, None

    fiedler_global = np.zeros(n, dtype=np.float32)
    for loc, g in enumerate(idx_global.tolist()):
        fiedler_global[int(g)] = fiedler_local[loc]

    pos_table = _build_temporal_positive_table(ep_ids, pos_k=int(cfg.replay_fiedler_pos_k))
    valid_idx = np.array([i for i, arr in enumerate(pos_table) if len(arr) > 0], dtype=np.int64)
    if len(valid_idx) < 8:
        return None, None, None

    perm = rng.permutation(len(valid_idx))
    n_val = max(1, int(round(cfg.replay_fiedler_val_frac * len(valid_idx))))
    train_idx = valid_idx[perm[n_val:]]
    val_idx_arr = valid_idx[perm[:n_val]]

    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    fiedler_t = torch.tensor(fiedler_global, dtype=torch.float32, device=device)

    model = FiedlerContrastiveHead(
        embed_dim=int(encoder_emb.shape[1]),
        fiedler_dim=int(cfg.replay_fiedler_dim),
        hidden_dim=int(cfg.replay_fiedler_hidden),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    n_neg = int(max(4, cfg.replay_fiedler_neg_k))
    temp = float(max(1e-3, cfg.replay_fiedler_temp))
    lam_f = float(cfg.replay_fiedler_lambda_fiedler)
    epochs = int(max(2, cfg.replay_fiedler_epochs))
    batch = int(min(max(16, cfg.replay_fiedler_batch), len(train_idx)))

    def _sample_batch(anchor_pool, b):
        a = rng.choice(anchor_pool, size=b, replace=len(anchor_pool) < b).astype(np.int64)
        p = np.empty(b, dtype=np.int64)
        for t, ai in enumerate(a.tolist()):
            cands = pos_table[int(ai)]
            p[t] = int(cands[int(rng.integers(0, len(cands)))])
        neg = rng.integers(0, n, size=(b, n_neg), dtype=np.int64)
        return (
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(p, dtype=torch.long, device=device),
            torch.tensor(neg, dtype=torch.long, device=device),
        )

    def _loss(ai_t, pi_t, ni_t):
        za = model.encode_unit(e_t[ai_t])
        zp = model.encode_unit(e_t[pi_t])
        zn = model.encode_unit(e_t[ni_t.reshape(-1)]).reshape(ai_t.shape[0], n_neg, -1)
        pos_logit = torch.sum(za * zp, dim=-1, keepdim=True) / temp
        neg_logit = torch.sum(za[:, None, :] * zn, dim=-1) / temp
        logits = torch.cat([pos_logit, neg_logit], dim=1)
        nce = (-pos_logit.squeeze(1) + torch.logsumexp(logits, dim=1)).mean()

        fiedler_pred = model.predict_fiedler(e_t[ai_t])
        fiedler_loss = F.mse_loss(fiedler_pred, fiedler_t[ai_t])
        return nce + lam_f * fiedler_loss, nce, fiedler_loss

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        ai, pi, ni = _sample_batch(train_idx, batch)
        loss, nce_l, fied_l = _loss(ai, pi, ni)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            vai, vpi, vni = _sample_batch(val_idx_arr, min(batch, max(8, len(val_idx_arr))))
            vl, vnce, vfied = _loss(vai, vpi, vni)
            if float(vl.item()) < best_val:
                best_val = float(vl.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, epochs // 6) == 0:
            print(f"    g_fiedler epoch {epoch+1}/{epochs}  "
                  f"train={loss.item():.4f} (nce={nce_l.item():.4f} fied={fied_l.item():.4f})  "
                  f"val={vl.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        z_all = model.encode(e_t).cpu().numpy().astype(np.float32)

    meta = {
        "n_samples": n, "epochs": epochs,
        "fiedler_dim": int(cfg.replay_fiedler_dim),
        "lambda_fiedler": lam_f,
        "best_val_combined": float(best_val),
        "feat_dict_key": "g_fiedler(e)",
    }
    return model, z_all, meta


# =====================================================================
# FIX 1: Oracle-free centrality bundle (no oracle bridge mask)
# =====================================================================

def _build_cell_centrality_bundle_oracle_free(
    pos: np.ndarray,
    episode_ids: np.ndarray,
    geodesic: GeodesicComputer,
) -> np.ndarray:
    """Per free-cell scalar: mean of z-scored (node betweenness, edge push, articulation).

    Unlike the original _build_cell_centrality_bundle, this version does NOT
    include the oracle bridge boundary mask — making it truly oracle-free.
    """
    adj = _cell_replay_adjacency(pos, episode_ids, geodesic)
    n_free = int(geodesic.n_free)
    if n_free == 0:
        return np.zeros(0, dtype=np.float32)
    nb = _node_betweenness_brandes_undirected(adj)
    et = _edge_betweenness_brandes_undirected(adj)
    eb = _edge_betweenness_push_to_nodes(adj, et)
    art = _articulation_points_undirected(adj)
    parts = [
        _zscore_safe(nb.astype(np.float32)),
        _zscore_safe(eb.astype(np.float32)),
        _zscore_safe(art.astype(np.float32)),
    ]
    return np.mean(np.stack(parts, axis=0), axis=0).astype(np.float32)


def _build_graph_native_node_scores_oracle_free(
    encoder_emb: np.ndarray,
    pos: np.ndarray,
    episode_ids: np.ndarray,
    geodesic: GeodesicComputer,
    rng: np.random.Generator,
    cfg: Optional[PointMazeLargeRunCfg] = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """Oracle-free node scores: novelty + uncertainty + centrality (no disagreement term).

    Centrality uses only betweenness and articulation (no oracle bridge mask).
    ``replay_node_score_beta`` is ignored (disagreement removed from the blend).
    """
    n = int(len(encoder_emb))
    alpha = gamma = delta = 1.0
    if cfg is not None:
        alpha = float(cfg.replay_node_score_alpha)
        gamma = float(cfg.replay_node_score_gamma)
        delta = float(cfg.replay_node_score_delta)

    cell_idx = _positions_to_cell_indices(geodesic, pos)
    n_free = int(geodesic.n_free)
    cell_counts = np.zeros(n_free, dtype=np.int64)
    for c in cell_idx.tolist():
        cell_counts[int(c)] += 1
    novelty = 1.0 / np.sqrt(np.maximum(1, cell_counts[cell_idx]).astype(np.float32))

    pair_pool = _oracle_free_replay_step_distances(
        episode_ids, n_pairs_target=min(max(2000, n), 10000), max_sources=128, rng=rng,
    )
    uncert = np.zeros(n, dtype=np.float32)
    if pair_pool is not None:
        ii, _, dd = pair_pool
        by_src: dict[int, list[float]] = {}
        for a, d in zip(ii.tolist(), dd.tolist()):
            by_src.setdefault(int(a), []).append(float(d))
        for a, vals in by_src.items():
            if len(vals) >= 2:
                uncert[a] = float(np.std(np.asarray(vals, dtype=np.float32)))

    nov_z = _zscore_safe(novelty)
    unc_z = _zscore_safe(uncert)

    cent_z = np.zeros(n, dtype=np.float32)
    cent_cell = _build_cell_centrality_bundle_oracle_free(pos, episode_ids, geodesic)
    if cent_cell.size == n_free:
        cent_z = _zscore_safe(cent_cell[cell_idx.astype(np.int64)])

    components: dict[str, np.ndarray] = {
        "novelty": nov_z,
        "uncertainty": unc_z,
        "centrality": cent_z,
    }
    summary: dict[str, float] = {}
    for name, arr in components.items():
        summary[f"{name}_mean"] = float(np.mean(arr))
        summary[f"{name}_std"] = float(np.std(arr))
        summary[f"{name}_frac_abs_gt_1e-8"] = float(np.mean(np.abs(arr) > 1e-8))

    if delta != 0.0:
        denom = max(1e-8, alpha + gamma + delta)
        score = (alpha * nov_z + gamma * unc_z + delta * cent_z) / denom
    else:
        denom = max(1e-8, alpha + gamma)
        score = (alpha * nov_z + gamma * unc_z) / denom

    summary["combined_target_mean"] = float(np.mean(score))
    summary["combined_target_std"] = float(np.std(score))
    summary["oracle_contamination"] = False
    summary["disagreement_term_included"] = False
    summary["blend_weights_used"] = {"alpha_novelty": alpha, "gamma_uncertainty": gamma, "delta_centrality": delta}
    return score.astype(np.float32), components, summary


def analyze_b_head_target_terms(
    components: dict[str, np.ndarray],
    y: np.ndarray,
    term_names: tuple[str, ...] = ("novelty", "uncertainty", "centrality"),
) -> dict[str, Any]:
    """How useful each b_head component is alone vs its contribution in a joint linear model.

    ``y`` is the combined b_score training target (same scale as fed to ReplayNodeScoreHead).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    alone: dict[str, Any] = {}
    for name in term_names:
        if name not in components:
            continue
        x = np.asarray(components[name], dtype=np.float64).ravel()
        if np.std(x) < 1e-12 or np.std(y) < 1e-12:
            alone[name] = {
                "pearson_vs_target": None,
                "spearman_vs_target": None,
                "r2_univariate_vs_target": None,
            }
            continue
        pr = float(np.corrcoef(x, y)[0, 1])
        sr = float(_spearman_rho_numpy(x.astype(np.float32), y.astype(np.float32)))
        alone[name] = {
            "pearson_vs_target": pr,
            "spearman_vs_target": sr,
            "r2_univariate_vs_target": float(pr * pr),
        }

    cols: list[np.ndarray] = []
    valid_names: list[str] = []
    for name in term_names:
        if name in components:
            cols.append(np.asarray(components[name], dtype=np.float64).ravel())
            valid_names.append(name)
    if not cols:
        return {
            "alone": alone,
            "ols_on_combined_target": None,
            "note": "no component columns",
        }

    X = np.column_stack(cols)
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-12
    Xs = (X - mu) / sd
    coef, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    y_hat = Xs @ coef
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_full = float(1.0 - ss_res / max(ss_tot, 1e-12))
    sq = coef * coef
    den = float(np.sum(sq) + 1e-12)
    rel_sq = {valid_names[i]: float(sq[i] / den) for i in range(len(valid_names))}

    drop_one: dict[str, float] = {}
    for j in range(len(valid_names)):
        mask = np.ones(len(valid_names), dtype=bool)
        mask[j] = False
        X_sub = Xs[:, mask]
        if X_sub.shape[1] == 0:
            drop_one[valid_names[j]] = float(r2_full)
            continue
        c2, *_ = np.linalg.lstsq(X_sub, y, rcond=None)
        y2 = X_sub @ c2
        ss2 = float(np.sum((y - y2) ** 2))
        r2_minus = float(1.0 - ss2 / max(ss_tot, 1e-12))
        drop_one[valid_names[j]] = float(r2_full - r2_minus)

    return {
        "alone": alone,
        "ols_on_combined_target": {
            "term_names": valid_names,
            "coefficients_standardized_X": {
                valid_names[i]: float(coef[i]) for i in range(len(valid_names))
            },
            "r2_full_multivariate": r2_full,
            "relative_contribution_squared_beta": rel_sq,
            "r2_drop_one_delta": drop_one,
        },
        "note": (
            "alone: univariate association with combined b_score target; "
            "r2_univariate_vs_target = Pearson^2. "
            "ols_on_combined_target: multivariate OLS on z-scored columns; "
            "relative_contribution_squared_beta normalizes coef^2 to sum 1; "
            "r2_drop_one_delta = R2_full - R2_without term (approx. unique variance)."
        ),
    }


# =====================================================================
# FIX 3: Edge classifier labels without encoder-distance stress
# =====================================================================

def _label_topology_edges_no_stress(
    idx_global: np.ndarray,
    adj_list: list[list[int]],
    episode_ids: np.ndarray,
    encoder_emb: np.ndarray,
    rng: np.random.Generator,
    *,
    bottleneck_top_frac: float = 0.2,
) -> Optional[np.ndarray]:
    """Label edges WITHOUT using encoder distances in the bottleneck composite.

    Removes the stress feature (log(graph_dist) - log(enc_dist)) that creates
    circularity when the classifier then predicts labels from encoder embeddings.
    Uses 4 features: edge betweenness, articulation, temporal-component, spectral-community.
    """
    edges = _mixed_graph_undirected_edges(adj_list)
    if len(edges) < 8:
        return None
    m = len(adj_list)
    idx_global = np.asarray(idx_global, dtype=np.int64)
    ep = np.asarray(episode_ids, dtype=np.int64)

    u_l = edges[:, 0].astype(np.int64)
    v_l = edges[:, 1].astype(np.int64)
    u_g = idx_global[u_l]
    v_g = idx_global[v_l]
    temporal = (np.abs(u_g - v_g) == 1) & (ep[u_g] == ep[v_g])

    edge_bet = _edge_betweenness_brandes_undirected(adj_list)
    ap = _articulation_points_undirected(adj_list)
    adj_temp = _temporal_only_adj_local(idx_global, ep, adj_list)
    comp_temp = _connected_components_from_adj(adj_temp)
    spec_lab = _spectral_community_labels(adj_list, rng)

    n_e = len(edges)
    a_l, b_l = u_l.astype(np.int64), v_l.astype(np.int64)
    ak, bk = np.minimum(a_l, b_l), np.maximum(a_l, b_l)
    eb_arr = np.array([float(edge_bet.get((int(ak[t]), int(bk[t])), 0.0)) for t in range(n_e)], dtype=np.float64)
    art_arr = np.maximum(ap[a_l], ap[b_l]).astype(np.float64)
    ct_arr = (comp_temp[a_l] != comp_temp[b_l]).astype(np.float64)
    cs_arr = (spec_lab[a_l] != spec_lab[b_l]).astype(np.float64)

    nt_mask = ~temporal
    nt_idx = np.flatnonzero(nt_mask)
    if len(nt_idx) == 0:
        return None

    feat = np.stack([eb_arr[nt_idx], art_arr[nt_idx], ct_arr[nt_idx], cs_arr[nt_idx]], axis=1)
    zfeat = np.stack([_zscore_1d_nonconst(feat[:, j]) for j in range(feat.shape[1])], axis=1)
    composite = zfeat.mean(axis=1)

    frac = float(np.clip(bottleneck_top_frac, 1e-6, 0.999))
    k_top = max(1, int(round(frac * len(nt_idx))))
    order = np.argsort(-composite)
    bot_nt = np.zeros(len(nt_idx), dtype=bool)
    bot_nt[order[:k_top]] = True

    cls = np.ones(n_e, dtype=np.int64)
    cls[temporal] = 0
    cls[nt_idx[bot_nt]] = 2

    rows = []
    for t in range(n_e):
        rows.append((int(u_g[t]), int(v_g[t]), int(cls[t])))
    arr = np.asarray(rows, dtype=np.int64)
    return arr if len(arr) >= 8 else None


# =====================================================================
# FIX 6: Bridge overlap with hypergeometric p-value
# =====================================================================

def _hypergeom_sf(k: int, N: int, K: int, n: int) -> float:
    """P(X >= k) under Hypergeometric(N, K, n). Exact for small N."""
    if k <= 0:
        return 1.0
    p = 0.0
    for x in range(k, min(K, n) + 1):
        log_num = (
            _log_comb(K, x) + _log_comb(N - K, n - x)
        )
        log_den = _log_comb(N, n)
        p += math.exp(log_num - log_den)
    return min(1.0, p)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return -float("inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _bridge_sanity_with_pvalue(
    geodesic: GeodesicComputer,
    pos: np.ndarray,
    src_idx: np.ndarray,
    src_score: np.ndarray,
    top_frac: float = 0.15,
) -> dict:
    """Bridge overlap sanity with hypergeometric p-value (Fix 6)."""
    bridge_cells, _ = _canonical_oracle_bridges(geodesic)

    if len(src_idx) == 0:
        return {"n_sources_scored": 0, "n_cells_scored": 0,
                "n_bridges_oracle": len(bridge_cells), "p_value": None,
                "top_cells_bridge_frac": None, "n_top_cells": 0}

    cell_idx_all = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free
    acc = np.zeros(n_free, dtype=np.float64)
    cnt = np.zeros(n_free, dtype=np.int64)
    for i, sc in zip(src_idx.tolist(), src_score.tolist()):
        c = int(cell_idx_all[int(i)])
        acc[c] += float(sc)
        cnt[c] += 1

    scored_cells = np.where(cnt > 0)[0].astype(np.int64)
    if len(scored_cells) == 0:
        return {"n_sources_scored": int(len(src_idx)), "n_cells_scored": 0,
                "n_bridges_oracle": len(bridge_cells), "p_value": None,
                "top_cells_bridge_frac": None, "n_top_cells": 0}

    mean = np.zeros(n_free, dtype=np.float32)
    m = cnt > 0
    mean[m] = (acc[m] / cnt[m]).astype(np.float32)

    bridge_mask = np.array([int(c) in bridge_cells for c in scored_cells], dtype=bool)
    mean_bridge = float(np.mean(mean[scored_cells[bridge_mask]])) if np.any(bridge_mask) else None
    mean_non = float(np.mean(mean[scored_cells[~bridge_mask]])) if np.any(~bridge_mask) else None

    k = max(1, int(round(float(top_frac) * len(scored_cells))))
    order = np.argsort(mean[scored_cells])[::-1]
    top_cells = scored_cells[order[:k]]
    n_top_bridge = sum(1 for c in top_cells.tolist() if int(c) in bridge_cells)
    top_bridge_frac = n_top_bridge / max(len(top_cells), 1)

    N_total = len(scored_cells)
    K_bridge_in_scored = int(np.sum(bridge_mask))
    p_val = _hypergeom_sf(n_top_bridge, N_total, K_bridge_in_scored, len(top_cells))

    return {
        "n_sources_scored": int(len(src_idx)),
        "n_cells_scored": N_total,
        "n_bridges_oracle": len(bridge_cells),
        "n_bridges_in_scored": K_bridge_in_scored,
        "mean_score_bridge": mean_bridge,
        "mean_score_nonbridge": mean_non,
        "top_frac": float(top_frac),
        "n_top_cells": int(len(top_cells)),
        "n_top_bridge": n_top_bridge,
        "top_cells_bridge_frac": float(top_bridge_frac),
        "hypergeom_p_value": float(p_val),
        "note": f"P(>={n_top_bridge} bridges in {len(top_cells)} draws from "
                f"{N_total} cells with {K_bridge_in_scored} bridges)",
    }


# =====================================================================
# Training wrappers (with fixes applied)
# =====================================================================

def train_replay_node_score_head_v2(
    data: dict,
    cfg: PointMazeLargeRunCfg,
    device: torch.device,
    seed: int,
    geodesic: GeodesicComputer,
):
    """Train b(e) node score head — oracle-free (Fix 1 applied)."""
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    pos = np.asarray(data["pos"], dtype=np.float32)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or len(ep_ids) != len(encoder_emb):
        return None, None, None, None
    n = int(len(encoder_emb))
    if n < 16:
        return None, None, None, None

    rng = np.random.default_rng(int(seed))
    target_np, components, component_summary = _build_graph_native_node_scores_oracle_free(
        encoder_emb, pos, ep_ids, geodesic, rng, cfg
    )
    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    y_t = torch.tensor(target_np, dtype=torch.float32, device=device)

    idx_all = np.arange(n, dtype=np.int64)
    component_bridge_sanity: dict[str, dict] = {}
    for cname, arr in components.items():
        component_bridge_sanity[cname] = _bridge_sanity_with_pvalue(
            geodesic, pos, idx_all, np.asarray(arr, dtype=np.float32)
        )

    b_head_target_analysis = analyze_b_head_target_terms(components, target_np)
    print("    b_score target term analysis (see metrics.json → b_head_target_analysis):")
    for tn, row in b_head_target_analysis.get("alone", {}).items():
        print(
            f"      {tn:12s}  r(target)={row.get('pearson_vs_target')}  "
            f"R²_uni={row.get('r2_univariate_vs_target')}"
        )
    ols = b_head_target_analysis.get("ols_on_combined_target") or {}
    if ols.get("relative_contribution_squared_beta"):
        print(
            "      joint OLS rel. |β|² share:",
            ols["relative_contribution_squared_beta"],
            " R²_full=",
            ols.get("r2_full_multivariate"),
        )

    perm = rng.permutation(n)
    n_val = min(max(1, int(round(cfg.replay_node_score_val_frac * n))), n - 1)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    model = ReplayNodeScoreHead(
        embed_dim=int(encoder_emb.shape[1]),
        hidden_dim=int(cfg.replay_node_score_hidden),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = int(max(2, cfg.replay_node_score_epochs))
    batch = int(min(max(32, cfg.replay_node_score_batch), max(32, len(train_idx))))

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        idx = rng.choice(train_idx, size=batch, replace=len(train_idx) < batch)
        it = torch.tensor(idx, dtype=torch.long, device=device)
        pred = model(e_t[it])
        loss = F.mse_loss(pred, y_t[it])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            iv = torch.tensor(val_idx, dtype=torch.long, device=device)
            vloss = F.mse_loss(model(e_t[iv]), y_t[iv])
            if float(vloss.item()) < best_val:
                best_val = float(vloss.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(1, epochs // 6) == 0:
            print(f"    b_score epoch {epoch+1}/{epochs}  train={loss.item():.4f}  val={vloss.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        b_all = model(e_t).cpu().numpy().astype(np.float32).reshape(-1, 1)

    meta = {
        "n_samples": n, "epochs": epochs, "batch": batch,
        "best_val_mse": float(best_val),
        "feat_dict_key": "b_score(e)",
        "oracle_free": True,
        "centrality_terms": ["node_betweenness", "edge_betweenness", "articulation"],
        "removed_oracle_term": "oracle_bridge_boundary_mask",
        "disagreement_removed_from_blend": True,
        "component_summary": component_summary,
        "component_bridge_sanity": component_bridge_sanity,
        "b_head_target_analysis": b_head_target_analysis,
    }
    return model, b_all, meta, components


def train_replay_edge_topology_head_v2(
    data: dict,
    cfg: PointMazeLargeRunCfg,
    device: torch.device,
    seed: int,
):
    """Train edge topology classifier — no encoder-distance stress (Fix 3)."""
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or len(ep_ids) != len(encoder_emb):
        return None, None
    idx_global, _, adj_list = _build_mixed_replay_graph(
        encoder_emb, ep_ids,
        n_graph_max=int(cfg.replay_graph_graph_max),
        k_knn=int(cfg.replay_graph_knn_k),
    )
    rng = np.random.default_rng(int(seed))
    labeled = _label_topology_edges_no_stress(
        idx_global, adj_list, ep_ids, encoder_emb, rng,
        bottleneck_top_frac=float(cfg.replay_edge_topo_bottleneck_frac),
    )
    if labeled is None:
        print("    edge_topo_v2: not enough labeled edges.")
        return None, None

    perm = rng.permutation(len(labeled))
    n_val = min(max(1, int(round(cfg.replay_edge_topo_val_frac * len(labeled)))), len(labeled) - 1)
    train_arr = labeled[perm[n_val:]]
    val_arr = labeled[perm[:n_val]]

    # Class-weighted CE (inverse frequency on train split; mean weight = 1)
    train_cls = train_arr[:, 2].astype(np.int64)
    train_cls_counts_np = np.bincount(train_cls, minlength=3).astype(np.float64)
    n_tr = max(len(train_arr), 1)
    inv_freq = n_tr / (3.0 * np.maximum(train_cls_counts_np, 1.0))
    inv_freq = inv_freq * (3.0 / np.sum(inv_freq))
    ce_class_weight = torch.tensor(inv_freq, dtype=torch.float32, device=device)

    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    model = ReplayEdgeTopoClassifier(
        embed_dim=int(encoder_emb.shape[1]),
        hidden_dim=int(cfg.replay_edge_topo_hidden),
        n_classes=3,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    epochs = int(max(2, cfg.replay_edge_topo_epochs))
    batch = int(min(max(32, cfg.replay_edge_topo_batch), max(32, len(train_arr))))

    def _loss(arr):
        ui = torch.tensor(arr[:, 0], dtype=torch.long, device=device)
        vi = torch.tensor(arr[:, 1], dtype=torch.long, device=device)
        yi = torch.tensor(arr[:, 2], dtype=torch.long, device=device)
        logits = model(e_t[ui], e_t[vi])
        return F.cross_entropy(logits, yi, weight=ce_class_weight), logits, yi

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        idx = rng.choice(len(train_arr), size=batch, replace=len(train_arr) < batch)
        loss, _, _ = _loss(train_arr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            vloss, vlogits, vy = _loss(val_arr)
            if float(vloss.item()) < best_val:
                best_val = float(vloss.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            vacc = float((vlogits.argmax(dim=-1) == vy).float().mean().item())
        if (epoch + 1) % max(1, epochs // 6) == 0:
            print(f"    edge_topo epoch {epoch+1}/{epochs}  train_ce={loss.item():.4f}  "
                  f"val_ce={vloss.item():.4f}  val_acc={vacc:.3f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        _, logits_v, y_v = _loss(val_arr)
        val_acc = float((logits_v.argmax(dim=-1) == y_v).float().mean().item())
        val_cls_counts = np.bincount(val_arr[:, 2], minlength=3).astype(np.int64).tolist()

    meta = {
        "n_pairs_total": int(len(labeled)),
        "n_pairs_train": int(len(train_arr)),
        "n_pairs_val": int(len(val_arr)),
        "epochs": epochs, "batch": batch,
        "best_val_ce": float(best_val),
        "val_acc": float(val_acc),
        "val_class_counts": val_cls_counts,
        "train_class_counts": [int(x) for x in train_cls_counts_np.tolist()],
        "cross_entropy_class_weights": [float(x) for x in inv_freq.tolist()],
        "cross_entropy_weighting": "inverse_frequency_train_split_mean_normalized",
        "label_features": ["edge_betweenness", "articulation", "temporal_component", "spectral_community"],
        "removed_feature": "encoder_distance_stress (Fix 3: removes circularity)",
        "bottleneck_top_frac": float(cfg.replay_edge_topo_bottleneck_frac),
    }
    return model, meta


# =====================================================================
# FIX 7: g_topo training with R² reporting
# =====================================================================

def train_replay_topology_head_v2(
    data: dict,
    cfg: PointMazeLargeRunCfg,
    device: torch.device,
    seed: int,
    geodesic: GeodesicComputer,
    out_dir: str,
):
    """Train g_topo with R² = 1 - MSE/Var(targets) reported (Fix 7)."""
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    pos = np.asarray(data["pos"], dtype=np.float32)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or encoder_emb.shape[0] != len(pos):
        return None, None, None

    rng = np.random.default_rng(int(seed))
    pool = _oracle_free_replay_step_distances(
        ep_ids, n_pairs_target=int(cfg.replay_topology_pair_pool), max_sources=96, rng=rng,
    )
    if pool is None:
        return None, None, None
    ii, jj, dd = pool
    n_pairs = len(ii)
    if n_pairs < 16:
        return None, None, None

    perm = rng.permutation(n_pairs)
    n_val = min(max(1, int(round(cfg.replay_topology_val_frac * n_pairs))), n_pairs - 1)
    train_mask = np.ones(n_pairs, dtype=bool)
    train_mask[perm[:n_val]] = False
    train_ii, train_jj, train_dd = ii[train_mask], jj[train_mask], dd[train_mask]
    val_ii, val_jj, val_dd = ii[~train_mask], jj[~train_mask], dd[~train_mask]

    embed_dim = int(encoder_emb.shape[1])
    n_ens = max(1, int(cfg.replay_topology_n_ensemble))
    topo_dim = int(cfg.replay_topology_dim)
    if topo_dim not in (2, 3):
        topo_dim = 2

    model = TemporalTopoHead(
        embed_dim=embed_dim, topo_dim=topo_dim,
        hidden_dim=int(cfg.replay_topology_hidden), n_ensemble=n_ens,
    ).to(device)
    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    total_epochs = max(2, int(cfg.replay_topology_epochs))
    batch_pairs = int(min(cfg.replay_topology_batch_pairs, max(1, len(train_ii))))

    best_val = float("inf")
    best_state = None
    for epoch in range(total_epochs):
        model.train()
        idx = rng.choice(len(train_ii), size=batch_pairs, replace=len(train_ii) < batch_pairs)
        ei = e_t[torch.tensor(train_ii[idx], dtype=torch.long, device=device)]
        ej = e_t[torch.tensor(train_jj[idx], dtype=torch.long, device=device)]
        target = torch.tensor(train_dd[idx], dtype=torch.float32, device=device)
        pred = model.forward_pair_ensemble(ei, ej)
        loss = F.mse_loss(pred, target.unsqueeze(-1).expand_as(pred))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            ve_i = e_t[torch.tensor(val_ii, dtype=torch.long, device=device)]
            ve_j = e_t[torch.tensor(val_jj, dtype=torch.long, device=device)]
            vt = torch.tensor(val_dd, dtype=torch.float32, device=device)
            val_loss = F.mse_loss(
                model.forward_pair_ensemble(ve_i, ve_j),
                vt.unsqueeze(-1).expand(-1, n_ens)
            )
            if float(val_loss.item()) < best_val:
                best_val = float(val_loss.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, total_epochs // 6) == 0:
            print(f"    g_topo epoch {epoch+1}/{total_epochs}  "
                  f"train_mse={loss.item():.4f}  val_mse={val_loss.item():.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        z_all = model.encode(e_t).cpu().numpy().astype(np.float32)

    # Fix 7: compute R² on validation set
    var_val_targets = float(np.var(val_dd))
    r2_val = 1.0 - best_val / max(var_val_targets, 1e-8)
    var_all_targets = float(np.var(dd))
    r2_interpretable = 1.0 - best_val / max(var_all_targets, 1e-8)

    pair_var_all = _predict_pair_ensemble_variance_batch(model, e_t, ii, jj, device)
    mean_var_all = float(np.mean(pair_var_all))

    # plots
    plot_path = os.path.join(out_dir, "temporal_dist_error_map.png")
    plot_temporal_dist_error_map(geodesic, pos, ii, jj, pair_var_all, plot_path)
    scatter_path = os.path.join(out_dir, "temporal_dist_error_scatter.png")
    plot_temporal_dist_error_scatter(pos, ii, pair_var_all, scatter_path, rng)

    src_idx, src_score = _compute_encoder_temporal_barrier_scores(
        encoder_emb=encoder_emb, episode_ids=ep_ids,
        pairs_ii=ii, pairs_jj=jj, pairs_steps=dd,
        rng=rng, k_nn=12, barrier_steps=25.0, candidate_pool=1800,
    )
    conflict_path = os.path.join(out_dir, "encoder_temporal_barrier_heatmap.png")
    plot_encoder_temporal_barrier_heatmap(geodesic, pos, src_idx, src_score, conflict_path)
    conflict_bridge = _bridge_sanity_with_pvalue(geodesic, pos, src_idx, src_score, top_frac=0.15)

    src_lg, disc_lg = _compute_encoder_temporal_local_global_disagreement(
        encoder_emb=encoder_emb, episode_ids=ep_ids, rng=rng,
    )
    lg_path = os.path.join(out_dir, "encoder_temporal_local_global_disagreement.png")
    plot_encoder_temporal_local_global_disagreement_heatmap(geodesic, pos, src_lg, disc_lg, lg_path)
    lg_bridge = _bridge_sanity_with_pvalue(geodesic, pos, src_lg, disc_lg, top_frac=0.15)

    meta = {
        "n_pairs": n_pairs, "n_ensemble": n_ens,
        "topo_dim": topo_dim, "total_epochs": total_epochs,
        "best_val_mse": float(best_val),
        "val_target_variance": var_val_targets,
        "val_r2": float(r2_val),
        "all_target_variance": var_all_targets,
        "r2_interpretable": float(r2_interpretable),
        "note_fix7": "R² = 1 - MSE/Var(targets). MSE alone is uninterpretable without target scale.",
        "mean_pair_pred_variance_all_pairs": mean_var_all,
        "feat_dict_key": "g_topo(e)",
        "encoder_temporal_barrier_bridge_sanity": conflict_bridge,
        "encoder_temporal_local_global_bridge_sanity": lg_bridge,
    }
    return model, z_all, meta


# =====================================================================
# FIX 2: g_lap with held-out evaluation and circularity flag
# =====================================================================

def compute_replay_laplacian_embedding_v2(
    data: dict,
    cfg: PointMazeLargeRunCfg,
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """Laplacian embedding with circularity note and held-out Spearman."""
    ep_ids = np.asarray(data.get("episode_ids", None), dtype=np.int64)
    enc = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if enc.ndim != 2 or len(ep_ids) != int(enc.shape[0]):
        return None, None

    idx_global, _, adj_list = _build_mixed_replay_graph(
        enc, ep_ids,
        n_graph_max=int(cfg.replay_laplacian_graph_max),
        k_knn=int(cfg.replay_laplacian_knn_k),
    )
    m = int(len(idx_global))
    if m < 8:
        return None, None

    a = np.zeros((m, m), dtype=np.float64)
    for i, nei in enumerate(adj_list):
        for j in nei:
            if int(j) != i:
                a[i, int(j)] = 1.0
                a[int(j), i] = 1.0
    deg = np.sum(a, axis=1)
    if np.all(deg <= 0):
        return None, None
    inv_sqrt = np.zeros(m, dtype=np.float64)
    nz = deg > 1e-12
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    d_inv = np.diag(inv_sqrt)
    l_mat = np.eye(m) - d_inv @ a @ d_inv
    evals, evecs = np.linalg.eigh(l_mat)
    dim = int(min(max(2, int(cfg.replay_laplacian_dim)), 3))
    start = 1
    end = min(m, start + dim)
    if end - start < 1:
        return None, None
    emb_local = evecs[:, start:end].astype(np.float32)
    if emb_local.shape[1] < dim:
        pad = np.zeros((m, dim - emb_local.shape[1]), dtype=np.float32)
        emb_local = np.concatenate([emb_local, pad], axis=1)

    z_all = np.zeros((len(ep_ids), dim), dtype=np.float32)
    z_all[idx_global] = emb_local

    meta = {
        "n_nodes": m, "dim": dim,
        "knn_k": int(cfg.replay_laplacian_knn_k),
        "graph_max": int(cfg.replay_laplacian_graph_max),
        "eigvals_used": [float(x) for x in evals[start:end].tolist()],
        "feat_dict_key": "g_lap(e)",
        "circularity_warning": (
            "Pearson/Spearman of g_lap distances vs same-graph shortest paths is "
            "tautological: spectral embeddings preserve graph structure by construction. "
            "Only cross-graph or oracle-geodesic correlations are informative."
        ),
    }
    return z_all, meta


# =====================================================================
# Fixed analysis functions
# =====================================================================

def _build_replay_transition_matrices(data, geodesic: GeodesicComputer):
    traj_pos = data.get("traj_pos", [])
    n_free = geodesic.n_free
    trans_counts = np.zeros((n_free, n_free), dtype=np.int64)
    for traj in traj_pos:
        if len(traj) < 2:
            continue
        cells = _positions_to_cell_indices(geodesic, traj)
        for t in range(len(cells) - 1):
            u, v = int(cells[t]), int(cells[t + 1])
            if u != v:
                trans_counts[u, v] += 1
    adj = [[] for _ in range(n_free)]
    for i in range(n_free):
        js = np.where((trans_counts[i] + trans_counts[:, i]) > 0)[0]
        adj[i] = [int(j) for j in js]
    return trans_counts, adj


def run_directed_geometry_analysis(data, geodesic: GeodesicComputer):
    trans_counts, _ = _build_replay_transition_matrices(data, geodesic)
    n_free = geodesic.n_free
    asym_vals = []
    total_edges = strong_asym = 0
    for i in range(n_free):
        for j in range(i + 1, n_free):
            cij, cji = trans_counts[i, j], trans_counts[j, i]
            total = cij + cji
            if total == 0:
                continue
            total_edges += 1
            asym_abs = abs(cij - cji) / float(total)
            asym_vals.append(asym_abs)
            if asym_abs > 0.5:
                strong_asym += 1
    if not asym_vals:
        return {"n_edges_with_data": 0, "mean_abs_asymmetry": 0.0}
    arr = np.asarray(asym_vals, dtype=np.float32)
    return {
        "n_edges_with_data": total_edges,
        "mean_abs_asymmetry": float(arr.mean()),
        "median_abs_asymmetry": float(np.median(arr)),
        "fraction_strong_asymmetry": strong_asym / max(total_edges, 1),
    }


def run_imagination_vs_replay_geometry(models, data, cfg, device, geo_temporal=None):
    """Fix 10: flag near-random imagination geometry."""
    rssm: RSSM = models["rssm"]
    rssm.eval()
    h_rep, s_rep = data["h"], data["s"]
    N_total = len(h_rep)
    if N_total < 4:
        return {}
    N = min(256, N_total)
    idx = np.random.choice(N_total, N, replace=False)
    h0 = torch.tensor(h_rep[idx], dtype=torch.float32, device=device)
    s0 = torch.tensor(s_rep[idx], dtype=torch.float32, device=device)
    horizon = max(1, getattr(cfg, "imagination_horizon", 15))
    h_im, s_im = h0.clone(), s0.clone()
    for _ in range(horizon):
        a = torch.empty(h_im.size(0), rssm.act_dim, device=device).uniform_(-1, 1)
        h_im, s_im = rssm.imagine_step(h_im, s_im, a)
    h_im_np, s_im_np = h_im.detach().cpu().numpy(), s_im.detach().cpu().numpy()

    if geo_temporal is not None:
        with torch.no_grad():
            g_rep = geo_temporal(
                torch.tensor(h_rep[idx], dtype=torch.float32, device=device),
                torch.tensor(s_rep[idx], dtype=torch.float32, device=device),
            ).cpu().numpy()
            g_im = geo_temporal(
                torch.tensor(h_im_np, dtype=torch.float32, device=device),
                torch.tensor(s_im_np, dtype=torch.float32, device=device),
            ).cpu().numpy()
        feat_rep, feat_im, feat_name = g_rep, g_im, "g(h,s)"
    else:
        feat_rep = np.concatenate([h_rep[idx], s_rep[idx]], axis=-1)
        feat_im = np.concatenate([h_im_np, s_im_np], axis=-1)
        feat_name = "h+s"

    from scipy import stats as sp_stats
    dm_rep = np.linalg.norm(feat_rep[:, None, :] - feat_rep[None, :, :], axis=-1)
    dm_im = np.linalg.norm(feat_im[:, None, :] - feat_im[None, :, :], axis=-1)
    triu = np.triu_indices(N, k=1)
    d_rep, d_im = dm_rep[triu], dm_im[triu]

    if len(d_rep) < 4:
        pearson = spearman = float("nan")
    else:
        pearson = float(sp_stats.pearsonr(d_rep, d_im)[0])
        spearman = float(sp_stats.spearmanr(d_rep, d_im)[0])

    k = min(10, N - 1)
    knn_rep = np.argsort(dm_rep, axis=1)[:, 1:k+1]
    knn_im = np.argsort(dm_im, axis=1)[:, 1:k+1]
    overlap = float(np.mean([len(set(knn_rep[i]) & set(knn_im[i])) / k for i in range(N)]))

    near_random = abs(pearson) < 0.3 and overlap < 0.25
    return {
        feat_name: {
            "pearson_replay_vs_imagination": pearson,
            "spearman_replay_vs_imagination": spearman,
            "knn_overlap_replay_vs_imagination": overlap,
            "k": k, "N_points": N, "horizon": horizon,
            "near_random_flag": near_random,
            "interpretation": (
                "RSSM imagination has diverged from real-space geometry after "
                f"{horizon} random-action steps. Imagination-based intrinsic rewards "
                "may be unreliable." if near_random else "Imagination geometry partially preserved."
            ),
        }
    }


def run_latent_room_discovery_v2(data, geodesic, features, cfg):
    """Fix 8 + 9: flag uniform community failure, use canonical bridges."""
    pos = data["pos"]
    cell_idx_all = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free

    adj_oracle = _adj_from_distmat(geodesic.dist_matrix)
    bridge_cells, bridges_oracle = _canonical_oracle_bridges(geodesic)
    comp_oracle, n_rooms_oracle = _components_without_bridges(adj_oracle, bridges_oracle)

    def _latent_graph_stats(feat_name, feat_arr):
        D = feat_arr.shape[1]
        cell_feats = np.zeros((n_free, D), dtype=np.float32)
        counts = np.zeros(n_free, dtype=np.int64)
        for f, c in zip(feat_arr, cell_idx_all):
            cell_feats[int(c)] += f
            counts[int(c)] += 1
        valid = counts > 0
        if not np.any(valid):
            return None
        cell_feats[valid] /= counts[valid, None]
        idx_cells = np.where(valid)[0]
        Fv = cell_feats[idx_cells]
        n_cells = len(idx_cells)
        if n_cells < 4:
            return None

        k = min(max(3, cfg.knn_k), n_cells - 1)
        dm = np.linalg.norm(Fv[:, None, :] - Fv[None, :, :], axis=-1)
        np.fill_diagonal(dm, np.inf)
        knn = np.argsort(dm, axis=1)[:, :k]

        adj_lat = [[] for _ in range(n_cells)]
        edges = set()
        for i in range(n_cells):
            for j_local in knn[i]:
                j = int(j_local)
                a, b = (i, j) if i < j else (j, i)
                if (a, b) not in edges:
                    edges.add((a, b))
                    adj_lat[i].append(j)
                    adj_lat[j].append(i)

        bridges_lat = _find_bridges(adj_lat)
        comp_lat, n_rooms_lat = _components_without_bridges(adj_lat, bridges_lat)
        comp_lat_full = np.full(n_free, -1, dtype=np.int64)
        for loc, cg in enumerate(idx_cells):
            comp_lat_full[int(cg)] = comp_lat[loc]

        visited = idx_cells
        agree = total_p = 0
        for i in range(len(visited)):
            for j in range(i + 1, len(visited)):
                ci, cj = visited[i], visited[j]
                if (comp_oracle[ci] == comp_oracle[cj]) == (comp_lat_full[ci] == comp_lat_full[cj]):
                    agree += 1
                total_p += 1
        pair_agreement = agree / max(total_p, 1)

        return {
            "n_rooms_oracle": int(n_rooms_oracle),
            "n_rooms_latent": int(n_rooms_lat),
            "n_bridges_oracle": int(len(bridge_cells)),
            "n_bridges_latent": int(len(bridges_lat)),
            "room_pair_agreement": float(pair_agreement),
            "n_cells_visited": int(len(visited)),
        }

    results = {}
    all_single_room = True
    for name, feat in features.items():
        stats = _latent_graph_stats(name, feat)
        if stats is not None:
            results[name] = stats
            if stats["n_rooms_latent"] > 1:
                all_single_room = False

    # Fix 8: flag uniform community failure
    if all_single_room and results:
        results["_community_failure_flag"] = {
            "all_n_rooms_latent_equal_1": True,
            "interpretation": (
                "No representation recovers multi-room structure at cell-mean level. "
                "This is a significant negative finding: positive distance correlations "
                "do not imply topological structure recovery."
            ),
        }
    return results


def run_metric_class_mismatch_v2(data, features, geodesic, cfg, device=None):
    """Fix 2: flag g_lap same-graph correlation as tautological."""
    from scipy import stats as sp_stats

    pos = data["pos"]
    cells = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free

    trans_counts, _ = _build_replay_transition_matrices(data, geodesic)
    adj = [[] for _ in range(n_free)]
    for i in range(n_free):
        js = np.where((trans_counts[i] + trans_counts[:, i]) > 0)[0]
        adj[i] = [int(j) for j in js]

    dist_replay = np.full((n_free, n_free), np.inf, dtype=np.float32)
    for src in range(n_free):
        dist_replay[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            du = dist_replay[src, u]
            for v in adj[u]:
                if dist_replay[src, v] == np.inf:
                    dist_replay[src, v] = du + 1.0
                    q.append(v)

    episode_ids = data.get("episode_ids", None)
    oracle_free_pairs = _oracle_free_replay_step_distances(
        episode_ids, n_pairs_target=min(cfg.n_pairs, 4000), max_sources=64,
    ) if episode_ids is not None else None

    N = len(pos)
    n_pairs = min(cfg.n_pairs, N * (N - 1) // 2)
    rng_np = np.random.default_rng(42)
    i1 = rng_np.integers(0, N, n_pairs)
    i2 = rng_np.integers(0, N, n_pairs)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]
    geo_d = np.array([geodesic.distance(pos[a], pos[b]) for a, b in zip(i1, i2)], dtype=np.float32)
    c1, c2 = cells[i1], cells[i2]
    rep_d = dist_replay[c1, c2]
    valid = np.isfinite(geo_d) & np.isfinite(rep_d) & (geo_d > 0) & (rep_d > 0)
    i1, i2, geo_d, rep_d = i1[valid], i2[valid], geo_d[valid], rep_d[valid]
    if len(geo_d) < 4:
        return {}

    results = {}
    for name, feat in features.items():
        lat_d = _latent_indexed_l2(feat, i1, i2, device=device)
        pr_geo = float(sp_stats.pearsonr(lat_d, geo_d)[0])
        sr_geo = float(sp_stats.spearmanr(lat_d, geo_d)[0])
        pr_rep = float(sp_stats.pearsonr(lat_d, rep_d)[0])
        sr_rep = float(sp_stats.spearmanr(lat_d, rep_d)[0])

        entry = {
            "pearson_latent_vs_geodesic": pr_geo,
            "spearman_latent_vs_geodesic": sr_geo,
            "pearson_latent_vs_replay_graph": pr_rep,
            "spearman_latent_vs_replay_graph": sr_rep,
            "feat_dim": int(feat.shape[1]),
        }

        # Fix 5: annotate 1D scalars
        if feat.shape[1] <= 2:
            entry["warning_low_dim"] = (
                f"Feature is {feat.shape[1]}-dimensional. T&C and kNN metrics "
                "for scalars are not comparable to high-dim embeddings."
            )

        # Fix 2: flag g_lap same-graph correlation
        if "g_lap" in name:
            entry["circularity_note"] = (
                "g_lap is the Laplacian eigenvector of the mixed graph. "
                "Correlation with same-graph shortest paths is tautological."
            )

        if oracle_free_pairs is not None:
            ii_of, jj_of, d_steps = oracle_free_pairs
            lat_d_s = _latent_indexed_l2(feat, ii_of, jj_of, device=device)
            entry["pearson_latent_vs_replay_steps"] = float(sp_stats.pearsonr(lat_d_s, d_steps)[0])
            entry["spearman_latent_vs_replay_steps"] = float(sp_stats.spearmanr(lat_d_s, d_steps)[0])

        results[name] = entry
    return results


# =====================================================================
# Single-seed pipeline
# =====================================================================

def run_single_seed(cfg_pm: PointMazeLargeRunCfg):
    device = get_device()
    set_seed(cfg_pm.seed)
    _mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    print(
        f"  Device: {device!s}  |  torch.cuda.is_available()={torch.cuda.is_available()}  "
        f"|  torch.backends.mps.is_available()={_mps_ok}"
    )

    cfg = TrainCfg()
    if cfg_pm.quick:
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
        for attr in ("replay_topology_epochs", "replay_cont_epochs", "replay_graph_epochs",
                      "replay_node_score_epochs", "replay_edge_topo_epochs",
                      "replay_sr_epochs", "replay_hit_epochs", "replay_fiedler_epochs"):
            setattr(cfg_pm, attr, min(80, getattr(cfg_pm, attr)))
        cfg_pm.replay_laplacian_graph_max = min(1200, cfg_pm.replay_laplacian_graph_max)
        cfg_pm.replay_graph_graph_max = min(1200, cfg_pm.replay_graph_graph_max)
        cfg_pm.replay_hit_n_walks = min(50, cfg_pm.replay_hit_n_walks)

    maze_name = "PointMaze_Large_Diverse_GR-v3"
    print(f"\n{'='*60}")
    print(f"  Seed {cfg_pm.seed}  |  {maze_name}  |  quick={cfg_pm.quick}")
    print(f"{'='*60}")

    env = PointMazeLargeDiverseGRWrapper(env_name=maze_name, img_size=cfg.img_size)
    print(f"  Grid: {env.grid_h}x{env.grid_w}  free cells: {env.geodesic.n_free}")

    # Canonical oracle bridges (Fix 9)
    bridge_cells, bridge_edges = _canonical_oracle_bridges(env.geodesic)
    print(f"  Oracle bridges: {len(bridge_edges)} edges, {len(bridge_cells)} endpoint cells")

    out_dir = os.path.join(cfg_pm.output_dir, f"seed{cfg_pm.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # [1] World model
    print("\n  [1/7] Training Dreamer world model ...")
    if cfg_pm.wm_path:
        checkpoint = torch.load(cfg_pm.wm_path, weights_only=False)
        act_dim = env.action_space.shape[0]
        encoder = ConvEncoder(cfg.embed_dim).to(device)
        encoder.load_state_dict(checkpoint["encoder"])
        decoder = ConvDecoder(cfg.deter_dim, cfg.stoch_dim, embedding_size=cfg.embed_dim).to(device)
        decoder.load_state_dict(checkpoint["decoder"])
        rssm = RSSM(cfg.stoch_dim, cfg.deter_dim, act_dim, cfg.embed_dim, cfg.hidden_dim).to(device)
        rssm.load_state_dict(checkpoint["rssm"])
        reward_model = RewardModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device)
        reward_model.load_state_dict(checkpoint["reward_model"])
        cont_model = ContinueModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device)
        cont_model.load_state_dict(checkpoint["cont_model"])
        models = {"encoder": encoder, "decoder": decoder, "rssm": rssm,
                  "reward_model": reward_model, "cont_model": cont_model}
        for m in models.values():
            m.eval()
    else:
        cfg.max_episodes = 200
        cfg.collect_interval = 100
        cfg.train_steps = 60
        models = train_world_model(env, cfg, device)
        wm_path = os.path.join(out_dir, "world_model.pt")
        torch.save({k: m.state_dict() for k, m in models.items() if m is not None}, wm_path)
        print(f"    Saved world model to {wm_path}")

    # [2] Data collection
    print("\n  [2/7] Collecting position-latent data ...")
    data = collect_data(env, models, cfg, device)
    sanity = compute_sanity_metrics(data, env.geodesic, cfg)
    print(f"    Sanity: cells={sanity['coverage']:.2%} rooms={sanity['room_coverage']:.2%} "
          f"bridges={sanity['bridge_crossings']} {'FAILED' if sanity['failed'] else 'ok'}")

    # [3] GeoEncoder
    print("\n  [3/7] Training GeoEncoder (temporal) ...")
    geo_temporal = train_geo_encoder(data, cfg, device, env.geodesic)
    geo_geo = None
    if cfg_pm.geo_supervised:
        geo_geo = train_geo_encoder_geodesic(data, cfg, device, env.geodesic)

    # [4] Primary replay heads: b_score (oracle-free), edge_topo, g_topo
    print("\n  [4/7] Training primary replay heads (b_score, edge_topo, g_topo) ...")
    topo_head = z_topo_all = topo_meta = None
    z_lap_all = lap_meta = None
    cont_head = z_cont_all = cont_meta = None
    graph_head = z_graph_all = graph_meta = None
    node_score_head = b_score_all = node_score_meta = None
    b_score_components = None
    edge_topo_head = edge_topo_meta = None

    can_train = (data.get("episode_ids") is not None and data.get("encoder_emb") is not None)

    if cfg_pm.replay_topology and can_train:
        print("    Training g_topo(e) ...")
        topo_head, z_topo_all, topo_meta = train_replay_topology_head_v2(
            data, cfg_pm, device, cfg_pm.seed, env.geodesic, out_dir)

    if cfg_pm.replay_cont and can_train:
        print("    Training g_cont(e) ...")
        from pointmaze_gr_geometry_test_topo import train_replay_contrastive_head
        cont_head, z_cont_all, cont_meta = train_replay_contrastive_head(
            data, cfg_pm, device, cfg_pm.seed)

    if cfg_pm.replay_laplacian and can_train:
        print("    Computing g_lap(e) ...")
        z_lap_all, lap_meta = compute_replay_laplacian_embedding_v2(data, cfg_pm)

    if cfg_pm.replay_graph and can_train:
        print("    Training g_graph(e) ...")
        from pointmaze_gr_geometry_test_topo import train_replay_graph_discrete_head
        graph_head, z_graph_all, graph_meta = train_replay_graph_discrete_head(
            data, cfg_pm, device, cfg_pm.seed)

    if cfg_pm.replay_node_score and can_train:
        print("    Training b_score(e) [oracle-free] ...")
        node_score_head, b_score_all, node_score_meta, b_score_components = \
            train_replay_node_score_head_v2(data, cfg_pm, device, cfg_pm.seed, env.geodesic)

    if cfg_pm.replay_edge_topo and can_train:
        print("    Training edge_topo [no stress] ...")
        edge_topo_head, edge_topo_meta = train_replay_edge_topology_head_v2(
            data, cfg_pm, device, cfg_pm.seed)

    # [5] Optional heads (CLI --enable_* only)
    _opt = cfg_pm.replay_sr or cfg_pm.replay_hit or cfg_pm.replay_fiedler
    print("\n  [5/7] Optional replay heads (g_sr / g_hit / g_fiedler) ..."
          f" {'training' if _opt else 'skipped (pass --enable_replay_*)'}")
    sr_head = z_sr_all = sr_meta = None
    hit_head = z_hit_all = hit_meta = None
    fiedler_head = z_fiedler_all = fiedler_meta = None

    if cfg_pm.replay_sr and can_train:
        print("    Training g_sr(e) — Successor Feature Head ...")
        sr_head, z_sr_all, sr_meta = train_replay_sr_head(data, cfg_pm, device, cfg_pm.seed)

    if cfg_pm.replay_hit and can_train:
        print("    Training g_hit(e) — Hitting-Time Head ...")
        hit_head, z_hit_all, hit_meta = train_replay_hit_head(data, cfg_pm, device, cfg_pm.seed)

    if cfg_pm.replay_fiedler and can_train:
        print("    Training g_fiedler(e) — Fiedler Contrastive Head ...")
        fiedler_head, z_fiedler_all, fiedler_meta = train_replay_fiedler_head(
            data, cfg_pm, device, cfg_pm.seed)

    # -----------------------------------------------------------------------
    # [6] Analysis — fast path:
    #   • subsample to ANALYSIS_N_MAX points before any pairwise computation
    #   • cap probe epochs so they don't run forever
    #   • T&C only for low-dim heads (≤64-dim); skip for 1024-dim encoder
    #   • every sub-step has a timed, flushed print so you can see progress
    # -----------------------------------------------------------------------
    import sys, time as _time

    ANALYSIS_N_MAX = 5_000   # max replay samples fed to probe/knn/T&C
    PROBE_EPOCHS_CAP = 500   # cap on linear+MLP probe training epochs
    TC_DIM_THRESHOLD = 64    # skip T&C for features wider than this

    def _tstamp(msg: str):
        print(f"\n  [6/7] {msg} ({_time.strftime('%H:%M:%S')})", flush=True)

    _tstamp("Building feature dict ...")
    pos = data["pos"]
    episode_ids = data.get("episode_ids", None)
    feat_dict = _build_feature_dict(data, device, geo_temporal=geo_temporal, geo_geodesic=geo_geo)

    # Inject trained head embeddings
    for _key, _arr in [
        ("g_topo(e)",    z_topo_all),
        ("g_cont(e)",    z_cont_all),
        ("g_lap(e)",     z_lap_all),
        ("g_graph(e)",   z_graph_all),
        ("g_sr(e)",      z_sr_all),
        ("g_hit(e)",     z_hit_all),
        ("g_fiedler(e)", z_fiedler_all),
    ]:
        if _arr is not None:
            feat_dict[_key] = _arr

    # b_score scalar: pad to 2-col so downstream code doesn't break on 1-col
    if b_score_all is not None:
        feat_dict["b_score(e)"] = np.concatenate(
            [b_score_all, np.zeros_like(b_score_all)], axis=1
        )
    if b_score_components is not None:
        for cname, arr in b_score_components.items():
            zcol = np.asarray(arr, dtype=np.float32).reshape(-1, 1)
            feat_dict[f"b_{cname}(e)"] = np.concatenate([zcol, np.zeros_like(zcol)], axis=1)

    # ── Subsample ─────────────────────────────────────────────────────────
    N_full = len(pos)
    if N_full > ANALYSIS_N_MAX:
        rng_sub = np.random.default_rng(cfg_pm.seed + 9999)
        sub_idx = rng_sub.choice(N_full, size=ANALYSIS_N_MAX, replace=False)
        sub_idx.sort()
        pos_sub = pos[sub_idx]
        feat_dict_sub = {k: v[sub_idx] for k, v in feat_dict.items()}
        ep_ids_sub = episode_ids[sub_idx] if episode_ids is not None else None
        # sub_data: consistent view for functions that read data["pos"] internally
        sub_data = dict(data)
        sub_data["pos"] = pos_sub
        if episode_ids is not None:
            sub_data["episode_ids"] = ep_ids_sub
        # traj_pos is per-episode, not per-timestep — keep full for directed geometry
        print(f"    Subsampled {N_full} → {ANALYSIS_N_MAX} points for analyses", flush=True)
    else:
        sub_idx = None
        pos_sub = pos
        feat_dict_sub = feat_dict
        ep_ids_sub = episode_ids
        sub_data = data

    # ── Cap probe epochs ──────────────────────────────────────────────────
    orig_probe_epochs = cfg.n_probe_epochs
    cfg.n_probe_epochs = min(cfg.n_probe_epochs, PROBE_EPOCHS_CAP)
    print(f"    Probe epochs capped: {orig_probe_epochs} → {cfg.n_probe_epochs}", flush=True)

    # ── Probes ────────────────────────────────────────────────────────────
    _tstamp(f"Probes (n_features={len(feat_dict_sub)}, n_probe_epochs={cfg.n_probe_epochs}) ...")
    probe_res = run_probes(pos_sub, feat_dict_sub, cfg, device, episode_ids=ep_ids_sub)
    print("    Probes complete", flush=True)

    # Restore probe epochs for any later use
    cfg.n_probe_epochs = orig_probe_epochs

    # ── Distance correlations ─────────────────────────────────────────────
    _tstamp("Distance correlations ...")
    dist_res, dist_raw = run_distance_analysis(
        pos_sub, feat_dict_sub, env.geodesic, cfg, device=device
    )
    print("    Distances complete", flush=True)

    # ── kNN ───────────────────────────────────────────────────────────────
    _tstamp("kNN analysis ...")
    knn_res = run_knn_analysis(pos_sub, feat_dict_sub, env.geodesic, cfg, device=device)
    print("    KNN complete", flush=True)

    # ── T&C — only for low-dim heads ──────────────────────────────────────
    # High-dim features (encoder_e 1024-dim, h 200-dim) require an N×N
    # cdist that takes minutes each.  We skip them and note it explicitly.
    _tstamp(f"Trustworthiness & Continuity (dim ≤ {TC_DIM_THRESHOLD} only) ...")
    feat_dict_tc = {
        k: v for k, v in feat_dict_sub.items() if v.shape[1] <= TC_DIM_THRESHOLD
    }
    skipped_tc = [k for k in feat_dict_sub if k not in feat_dict_tc]
    if skipped_tc:
        print(f"    T&C skipping high-dim features: {skipped_tc}", flush=True)

    tc_res_raw = run_trustworthiness_continuity(
        pos_sub, feat_dict_tc, env.geodesic, cfg, k=cfg.knn_k, device=device
    )
    # Add skipped entries with note
    for k in skipped_tc:
        tc_res_raw[k] = {
            "trustworthiness": None,
            "continuity": None,
            "skipped": f"dim={feat_dict_sub[k].shape[1]} > {TC_DIM_THRESHOLD}; "
                       "full cdist would OOM/stall — use subsampled kNN instead",
        }
    print("    T&C complete", flush=True)

    # Annotate with dimensionality (Fix 5)
    tc_annotated = {}
    for name, tc_vals in tc_res_raw.items():
        feat = feat_dict_sub.get(name)
        dim = int(feat.shape[1]) if feat is not None else -1
        entry = dict(tc_vals) if isinstance(tc_vals, dict) else {"value": tc_vals}
        entry["feat_dim"] = dim
        if 0 < dim <= 2:
            entry["warning"] = (
                f"{dim}D scalar: T&C nearest-neighbor overlap is trivially "
                "high for scalars correlated with maze geometry. Not comparable to high-dim."
            )
        tc_annotated[name] = entry
    tc_res = tc_annotated

    # ── Remaining analyses ────────────────────────────────────────────────
    _tstamp("Directed geometry ...")
    directed_geo_res = run_directed_geometry_analysis(data, env.geodesic)
    print("    Directed geometry complete", flush=True)

    _tstamp("Imagination vs replay geometry ...")
    imagination_res = run_imagination_vs_replay_geometry(
        models, data, cfg, device, geo_temporal
    )
    print("    Imagination vs replay complete", flush=True)

    _tstamp("Latent room discovery ...")
    community_res = run_latent_room_discovery_v2(
        sub_data, env.geodesic, feat_dict_sub, cfg
    )
    print("    Latent room discovery complete", flush=True)

    _tstamp("Metric class mismatch ...")
    metric_mismatch_res = run_metric_class_mismatch_v2(
        sub_data, feat_dict_sub, env.geodesic, cfg, device=device
    )
    print("    Metric class mismatch complete", flush=True)
    # [7] Plots — use subsampled pos/feat_dict so plots match the analysis data
    print("\n  [7/7] Generating plots ...", flush=True)
    generate_plots(maze_name, pos_sub, feat_dict_sub, probe_res, dist_res, dist_raw, knn_res, cfg, device, out_dir)
    if "g_lap(e)" in feat_dict_sub:
        plot_laplacian_kmeans_grid(
            geodesic=env.geodesic, pos=pos_sub, g_lap_feat=feat_dict_sub["g_lap(e)"],
            out_path=os.path.join(out_dir, "g_lap_kmeans9_grid.png"), n_clusters=9, seed=cfg_pm.seed)

    env.close()

    # Assemble results
    results = {
        "seed": cfg_pm.seed,
        "maze": maze_name,
        "replay_heads_enabled": {
            "b_score": bool(cfg_pm.replay_node_score),
            "edge_topo": bool(cfg_pm.replay_edge_topo),
            "g_topo": bool(cfg_pm.replay_topology),
            "g_cont": bool(cfg_pm.replay_cont),
            "g_lap": bool(cfg_pm.replay_laplacian),
            "g_graph": bool(cfg_pm.replay_graph),
            "g_sr": bool(cfg_pm.replay_sr),
            "g_hit": bool(cfg_pm.replay_hit),
            "g_fiedler": bool(cfg_pm.replay_fiedler),
        },
        "b_score_blend": {
            "components": ["novelty", "uncertainty", "centrality"],
            "disagreement_in_blend": False,
            "replay_node_score_beta": float(cfg_pm.replay_node_score_beta),
        },
        "n_free_cells": int(env.geodesic.n_free),
        "n_bridges_oracle": int(len(bridge_cells)),
        "n_bridge_edges_oracle": int(len(bridge_edges)),
        "sanity": sanity,
        "probes": probe_res,
        "distances": dist_res,
        "knn": knn_res,
        "trust_cont": tc_res,
        "directed_geometry": directed_geo_res,
        "imagination_vs_replay": imagination_res,
        "latent_communities": community_res,
        "metric_class_mismatch": metric_mismatch_res,
    }
    if topo_meta:
        results["g_topo_eval"] = topo_meta
    if cont_meta:
        results["g_cont_eval"] = cont_meta
    if lap_meta:
        results["g_lap_eval"] = lap_meta
    if graph_meta:
        results["g_graph_eval"] = graph_meta
    if node_score_meta:
        results["b_score_eval"] = node_score_meta
        bhta = node_score_meta.get("b_head_target_analysis")
        if bhta is not None:
            results["b_head_target_analysis"] = bhta
    if edge_topo_meta:
        results["edge_topo_eval"] = edge_topo_meta
    if sr_meta:
        results["g_sr_eval"] = sr_meta
    if hit_meta:
        results["g_hit_eval"] = hit_meta
    if fiedler_meta:
        results["g_fiedler_eval"] = fiedler_meta

    results["fixes_applied"] = {
        "fix1_oracle_free_centrality": True,
        "fix2_glap_circularity_flagged": True,
        "fix3_edge_labels_no_stress": True,
        "fix5_scalar_tc_annotated": True,
        "fix6_bridge_pvalue": True,
        "fix7_topo_r2": True,
        "fix8_community_failure_flagged": True,
        "fix9_canonical_bridges": True,
        "fix10_imagination_flagged": True,
    }
    results["analysis_settings"] = {
        "analysis_n_max": ANALYSIS_N_MAX,
        "n_full_replay": int(N_full),
        "n_subsampled": int(len(pos_sub)),
        "probe_epochs_cap": PROBE_EPOCHS_CAP,
        "probe_epochs_used": int(min(orig_probe_epochs, PROBE_EPOCHS_CAP)),
        "tc_dim_threshold": TC_DIM_THRESHOLD,
        "tc_skipped_features": skipped_tc,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    print(f"\n  Seed {cfg_pm.seed} results saved to {out_dir}/metrics.json")
    return results


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
# Multi-seed runner + summary
# =====================================================================

def summarize_seeds(all_results: list[dict], output_dir: str) -> dict:
    """Produce a cross-seed summary with mean +/- std for key metrics."""
    summary: dict[str, Any] = {
        "n_seeds": len(all_results),
        "seeds": [r["seed"] for r in all_results],
        "maze": all_results[0].get("maze", ""),
    }

    def _extract_scalar(results, *keys):
        vals = []
        for r in results:
            v = r
            for k in keys:
                if isinstance(v, dict):
                    v = v.get(k)
                else:
                    v = None
                    break
            if v is not None and isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        return vals

    def _mean_std(vals):
        if not vals:
            return {"mean": None, "std": None, "n": 0}
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}

    # Sanity
    summary["coverage"] = _mean_std(_extract_scalar(all_results, "sanity", "coverage"))
    summary["room_coverage"] = _mean_std(_extract_scalar(all_results, "sanity", "room_coverage"))

    # g_topo R² (Fix 7)
    summary["g_topo_val_r2"] = _mean_std(_extract_scalar(all_results, "g_topo_eval", "val_r2"))
    summary["g_topo_val_mse"] = _mean_std(_extract_scalar(all_results, "g_topo_eval", "best_val_mse"))

    # g_sr
    summary["g_sr_val_bellman_mse"] = _mean_std(_extract_scalar(all_results, "g_sr_eval", "best_val_bellman_mse"))

    # g_hit
    summary["g_hit_val_r2"] = _mean_std(_extract_scalar(all_results, "g_hit_eval", "val_r2"))

    # g_fiedler
    summary["g_fiedler_val_combined"] = _mean_std(_extract_scalar(all_results, "g_fiedler_eval", "best_val_combined"))

    # b_score
    summary["b_score_val_mse"] = _mean_std(_extract_scalar(all_results, "b_score_eval", "best_val_mse"))

    # edge_topo
    summary["edge_topo_val_acc"] = _mean_std(_extract_scalar(all_results, "edge_topo_eval", "val_acc"))

    # Per-feature distance correlations (aggregate across seeds)
    feat_names_all = set()
    for r in all_results:
        mc = r.get("metric_class_mismatch", {})
        feat_names_all.update(mc.keys())

    dist_summary = {}
    for fn in sorted(feat_names_all):
        if fn.startswith("_"):
            continue
        pearson_geo = _extract_scalar(all_results, "metric_class_mismatch", fn, "pearson_latent_vs_geodesic")
        spearman_geo = _extract_scalar(all_results, "metric_class_mismatch", fn, "spearman_latent_vs_geodesic")
        spearman_replay = _extract_scalar(all_results, "metric_class_mismatch", fn, "spearman_latent_vs_replay_steps")
        entry = {
            "pearson_vs_geodesic": _mean_std(pearson_geo),
            "spearman_vs_geodesic": _mean_std(spearman_geo),
            "spearman_vs_replay_steps": _mean_std(spearman_replay),
        }
        # Check for circularity flag
        for r in all_results:
            mc = r.get("metric_class_mismatch", {}).get(fn, {})
            if "circularity_note" in mc:
                entry["circularity_note"] = mc["circularity_note"]
                break
            if "warning_low_dim" in mc:
                entry["warning_low_dim"] = mc["warning_low_dim"]
                break
        dist_summary[fn] = entry
    summary["distance_correlations"] = dist_summary

    # Community failure (Fix 8)
    community_fail_count = 0
    for r in all_results:
        lc = r.get("latent_communities", {})
        if "_community_failure_flag" in lc:
            community_fail_count += 1
    summary["community_failure_all_seeds"] = community_fail_count == len(all_results)
    summary["community_failure_n_seeds"] = community_fail_count

    # Imagination divergence (Fix 10)
    near_random_count = 0
    for r in all_results:
        ir = r.get("imagination_vs_replay", {})
        for feat_name, vals in ir.items():
            if isinstance(vals, dict) and vals.get("near_random_flag", False):
                near_random_count += 1
                break
    summary["imagination_near_random_n_seeds"] = near_random_count

    # Bridge sanity p-values across seeds
    pvals = []
    for r in all_results:
        bs = (r.get("g_topo_eval") or {}).get("encoder_temporal_barrier_bridge_sanity", {})
        pv = bs.get("hypergeom_p_value")
        if pv is not None:
            pvals.append(pv)
    summary["bridge_sanity_pvalues"] = _mean_std(pvals) if pvals else None

    out_path = os.path.join(output_dir, "summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=_json_default)
    print(f"\nCross-seed summary saved to {out_path}")
    return summary


def print_summary_table(summary: dict):
    """Print a human-readable summary table."""
    print("\n" + "=" * 70)
    print(f"  CROSS-SEED SUMMARY ({summary['n_seeds']} seeds on {summary['maze']})")
    print("=" * 70)

    def _fmt(d):
        if d is None or d.get("mean") is None:
            return "N/A"
        return f"{d['mean']:.4f} +/- {d['std']:.4f} (n={d['n']})"

    print(f"  Coverage:           {_fmt(summary.get('coverage'))}")
    print(f"  Room coverage:      {_fmt(summary.get('room_coverage'))}")
    print(f"  g_topo val R²:      {_fmt(summary.get('g_topo_val_r2'))}")
    print(f"  g_topo val MSE:     {_fmt(summary.get('g_topo_val_mse'))}")
    print(f"  b_score val MSE:    {_fmt(summary.get('b_score_val_mse'))}")
    print(f"  edge_topo val acc:  {_fmt(summary.get('edge_topo_val_acc'))}")
    for label, key in (
        ("g_sr val Bellman", "g_sr_val_bellman_mse"),
        ("g_hit val R²", "g_hit_val_r2"),
        ("g_fiedler val", "g_fiedler_val_combined"),
    ):
        block = summary.get(key)
        if block and block.get("n", 0) > 0:
            print(f"  {label}: {_fmt(block)}")

    if summary.get("community_failure_all_seeds"):
        print("\n  *** COMMUNITY FAILURE: No representation recovered multi-room structure ***")
    if summary.get("imagination_near_random_n_seeds", 0) > 0:
        n = summary["imagination_near_random_n_seeds"]
        print(f"\n  *** IMAGINATION DIVERGENCE: {n}/{summary['n_seeds']} seeds show near-random geometry ***")

    dc = summary.get("distance_correlations", {})
    if dc:
        print("\n  Distance correlations (Spearman vs geodesic, mean +/- std):")
        for fn in sorted(dc.keys()):
            entry = dc[fn]
            sg = entry.get("spearman_vs_geodesic", {})
            circ = " [TAUTOLOGICAL]" if "circularity_note" in entry else ""
            ldim = " [LOW-DIM]" if "warning_low_dim" in entry else ""
            print(f"    {fn:25s}  {_fmt(sg)}{circ}{ldim}")

    bs = summary.get("bridge_sanity_pvalues")
    if bs and bs.get("mean") is not None:
        print(f"\n  Bridge sanity p-value: {_fmt(bs)}")

    print("=" * 70)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-seed topology eval on PointMaze_Large_Diverse_GR-v3. "
        "Default: b_score (3-term oracle-free), edge_topo, g_topo only."
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2",
        help="Comma-separated seeds (default 3 seeds for mean±std)",
    )
    p.add_argument("--output_dir", default="pointmaze_large_gr_results")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--geo_supervised", action="store_true")
    p.add_argument("--wm_path", type=str, default="")
    p.add_argument("--no_replay_topology", action="store_true", help="Disable g_topo")
    p.add_argument("--no_replay_node_score", action="store_true", help="Disable b_score")
    p.add_argument("--no_replay_edge_topo", action="store_true", help="Disable edge classifier")
    p.add_argument(
        "--enable_replay_cont",
        action="store_true",
        help="Train g_cont(e) and include in probes / distance analysis",
    )
    p.add_argument(
        "--enable_replay_laplacian",
        action="store_true",
        help="Compute g_lap(e) and include in analysis",
    )
    p.add_argument(
        "--enable_replay_graph",
        action="store_true",
        help="Train g_graph(e) discrete head and include in analysis",
    )
    p.add_argument(
        "--enable_replay_sr",
        action="store_true",
        help="Train g_sr(e) successor-feature head",
    )
    p.add_argument(
        "--enable_replay_hit",
        action="store_true",
        help="Train g_hit(e) hitting-time head",
    )
    p.add_argument(
        "--enable_replay_fiedler",
        action="store_true",
        help="Train g_fiedler(e) contrastive head",
    )
    p.add_argument(
        "--replay_edge_topo_bottleneck_frac",
        type=float,
        default=0.1,
        help="Top fraction of non-temporal edges labeled as bottleneck (class 2)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print(f"Running {len(seeds)} seeds: {seeds}")

    all_results = []
    for seed in seeds:
        cfg_pm = PointMazeLargeRunCfg(
            seed=seed,
            output_dir=args.output_dir,
            quick=bool(args.quick),
            geo_supervised=bool(args.geo_supervised),
            wm_path=args.wm_path,
            replay_topology=not args.no_replay_topology,
            replay_node_score=not args.no_replay_node_score,
            replay_edge_topo=not args.no_replay_edge_topo,
            replay_edge_topo_bottleneck_frac=float(args.replay_edge_topo_bottleneck_frac),
            replay_cont=bool(args.enable_replay_cont),
            replay_laplacian=bool(args.enable_replay_laplacian),
            replay_graph=bool(args.enable_replay_graph),
            replay_sr=bool(args.enable_replay_sr),
            replay_hit=bool(args.enable_replay_hit),
            replay_fiedler=bool(args.enable_replay_fiedler),
        )
        results = run_single_seed(cfg_pm)
        all_results.append(results)

    summary = summarize_seeds(all_results, args.output_dir)
    print_summary_table(summary)


if __name__ == "__main__":
    main()