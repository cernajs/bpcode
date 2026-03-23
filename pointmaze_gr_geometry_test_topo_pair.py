#!/usr/bin/env python3

import argparse
import os
from dataclasses import dataclass
from typing import Tuple, Dict, Optional

import torch.nn as nn
import torch.nn.functional as F

import cv2
import gymnasium as gym
import gymnasium_robotics  # type: ignore
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
from scipy.sparse.linalg import eigsh

from maze_env import GeodesicComputer
from maze_geometry_test import (
    TrainCfg,
    train_world_model,
    collect_data,
    train_geo_encoder,
    train_geo_encoder_geodesic,
    _build_feature_dict,
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


# ---------------------------------------------------------------------------
# Geodesic for PointMaze_Medium_Diverse_GR-v3 (published layout)
# ---------------------------------------------------------------------------


MEDIUM_MAZE_DIVERSE_GR = [
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, "C", 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, "C", 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, "C", 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, "C", 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
]


def make_pointmaze_medium_gr_geodesic() -> GeodesicComputer:
    """Convert MEDIUM_MAZE_DIVERSE_GR to a binary grid and build GeodesicComputer.

    We treat '0' and 'C' cells as free; '1' as walls, matching the docs.
    """
    grid = []
    for row in MEDIUM_MAZE_DIVERSE_GR:
        grid_row = []
        for cell in row:
            if cell == 1:
                grid_row.append("1")
            else:
                # 0 or 'C' => free
                grid_row.append("0")
        grid.append(grid_row)
    return GeodesicComputer(grid)


# ---------------------------------------------------------------------------
# Wrapper: Gymnasium-Robotics PointMaze -> Dreamer-compatible pixel env
# ---------------------------------------------------------------------------


class PointMazeMediumDiverseGRWrapper:
    """Wraps PointMaze_Medium_Diverse_GR-v3 to match PointMazeEnv interface.

    - Observations: RGB arrays from env.render(), shape (H, W, 3), uint8.
    - Actions: same Box(-1,1,(2,)) as the base env.
    - reset(): returns (obs, info)
    - step(a, repeat): repeats underlying env.step(a) 'repeat' times, summing
      rewards and returning the final rendered frame plus (terminated,truncated).
    - agent_pos: uses the achieved_goal (x,y) from the env's observation dict.
    - geodesic: GeodesicComputer built from MEDIUM_MAZE_DIVERSE_GR.
    """

    def __init__(self, env_name: str = "PointMaze_Medium_Diverse_GR-v3", img_size: int = 64):
        gym.register_envs(gymnasium_robotics)
        # Use rgb_array render mode so env.render() returns frames.
        self._env = gym.make(
            env_name,
            render_mode="rgb_array",
        )
        self.img_size = int(img_size)
        # One dummy reset/render to infer image shape; we ignore the dict obs.
        obs_dict, _ = self._env.reset()
        frame = self._env.render()
        assert isinstance(frame, np.ndarray) and frame.ndim == 3

        # Observation space is always (img_size, img_size, 3) to match TrainCfg.
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self.img_size, self.img_size, 3), dtype=np.uint8
        )
        self.action_space = self._env.action_space

        # Geodesic on the published medium-diverse-GR layout.
        self.geodesic = make_pointmaze_medium_gr_geodesic()
        self.grid_h = len(self.geodesic.grid)
        self.grid_w = len(self.geodesic.grid[0])
        # Free cells (row, col) for random reset — ensures coverage across the maze.
        self._free_cells = list(self.geodesic.idx_to_cell)

        self._agent_pos = np.zeros(2, dtype=np.float32)
        self._update_agent_pos(obs_dict)

    # ----- public helpers -----

    @property
    def agent_pos(self) -> np.ndarray:
        return self._agent_pos.copy()

    def _resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize raw mujoco frame to (img_size, img_size)."""
        if frame.shape[0] == self.img_size and frame.shape[1] == self.img_size:
            return frame
        return cv2.resize(frame, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

    def _world_to_grid(self, xy):
        x, y = float(xy[0]), float(xy[1])

        # likely mapping for 8x8 published maze
        gx = x + (self.grid_w / 2 - 0.5)
        gy = (self.grid_h / 2 - 0.5) - y

        return np.array([gx, gy], dtype=np.float32)

    def _update_agent_pos(self, obs_dict):
        # obs_dict["achieved_goal"] is shape (2,) with (x,y) in MuJoCo coords.
        ag = obs_dict.get("achieved_goal", None)
        if ag is None:
            # Fallback: use first two entries of "observation" (x,y).
            obs = obs_dict.get("observation", None)
            if obs is not None and len(obs) >= 2:
                ag = obs[:2]
        if ag is not None:
            self._agent_pos = self._world_to_grid(ag[:2])

    # ----- gym-like interface -----

    def reset(self, **kwargs):
        # Start agent at a random free cell each episode for full maze coverage.
        if self._free_cells:
            r, c = self._free_cells[np.random.randint(0, len(self._free_cells))]
            options = kwargs.get("options") or {}
            options = dict(options)
            options["reset_cell"] = np.array([r, c], dtype=np.int64)
            kwargs = dict(kwargs)
            kwargs["options"] = options
        obs_dict, info = self._env.reset(**kwargs)
        self._update_agent_pos(obs_dict)
        frame = self._env.render()
        frame = self._resize_frame(frame).astype(np.uint8)
        return frame, info

    def step(self, action, repeat: int = 1):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(int(repeat)):
            obs_dict, r, t, tr, info = self._env.step(action)
            self._update_agent_pos(obs_dict)
            total_reward += float(r)
            terminated = bool(t)
            truncated = bool(tr)
            if terminated or truncated:
                break

        frame = self._env.render()
        frame = self._resize_frame(frame).astype(np.uint8)
        return frame, total_reward, terminated, truncated, info

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# Single-seed pipeline (mirrors run_single_maze)
# ---------------------------------------------------------------------------


@dataclass
class PointMazeRunCfg:
    seed: int = 0
    output_dir: str = "pointmaze_gr_results"
    quick: bool = False
    geo_supervised: bool = False
    wm_path: str = ""
    replay_topology: bool = True
    replay_topology_dim: int = 16
    replay_topology_hidden: int = 256
    replay_topology_spec_dim: int = 8
    replay_topology_epochs: int = 200
    replay_topology_batch_nodes: int = 1024
    replay_graph_max_nodes: int = 2500
    replay_graph_knn: int = 4
    replay_graph_quantile: float = 0.10
    replay_topology_same_region_weight: float = 1.0
    replay_topology_bridge_node_weight: float = 0.75
    replay_topology_neighbor_weight: float = 0.50
    replay_topology_bridge_edge_weight: float = 0.50
    replay_topology_cut_weight: float = 0.25
    replay_topology_spec_weight: float = 0.75
    replay_topology_margin_weight: float = 0.25
    replay_topology_train_frac: float = 0.70
    replay_topology_val_frac: float = 0.15
    replay_topology_test_frac: float = 0.15
    replay_topology_eval_repeats: int = 24


def run_single_pointmaze(cfg_pm: PointMazeRunCfg):
    device = get_device()
    set_seed(cfg_pm.seed)

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
        cfg_pm.replay_topology_epochs = min(80, int(cfg_pm.replay_topology_epochs))

    maze_name = "PointMaze_Medium_Diverse_GR-v3"
    print(f"Device: {device}")
    print(f"Maze:   {maze_name}")
    print(f"Seed:   {cfg_pm.seed}")
    print(f"Quick:  {cfg_pm.quick}")

    env = PointMazeMediumDiverseGRWrapper(
        env_name=maze_name,
        img_size=cfg.img_size)
    print(f"  Grid (medium-diverse-GR): {env.grid_h}×{env.grid_w}  free cells: {env.geodesic.n_free}")

    out_dir = os.path.join(cfg_pm.output_dir, f"seed{cfg_pm.seed}")
    os.makedirs(out_dir, exist_ok=True)

    print("\n  [1/5] Training Dreamer world model ...")
    if cfg_pm.wm_path:
        print(f"    Resuming from world model checkpoint at {cfg_pm.wm_path}")
        checkpoint = torch.load(cfg_pm.wm_path)
        act_dim = env.action_space.shape[0]
        
        encoder = ConvEncoder(cfg.embed_dim).to(device)
        encoder.load_state_dict(checkpoint["encoder"]),

        decoder = ConvDecoder(cfg.deter_dim, cfg.stoch_dim, embedding_size=cfg.embed_dim).to(device)
        decoder.load_state_dict(checkpoint["decoder"])

        rssm = RSSM(cfg.stoch_dim, cfg.deter_dim, act_dim, cfg.embed_dim, cfg.hidden_dim).to(device)
        rssm.load_state_dict(checkpoint["rssm"])

        reward_model = RewardModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device)
        reward_model.load_state_dict(checkpoint["reward_model"])

        cont_model = ContinueModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device)
        cont_model.load_state_dict(checkpoint["cont_model"])

        models = {
            "encoder": encoder,
            "decoder": decoder,
            "rssm": rssm,
            "reward_model": reward_model,
            "cont_model": cont_model,
        }

        models["encoder"].eval()
        models["decoder"].eval()
        models["rssm"].eval()
        models["reward_model"].eval()
        models["cont_model"].eval()
    else:
        models = train_world_model(env, cfg, device)

        wm_path = os.path.join(out_dir, "world_model.pt")
        checkpoint = {
            "encoder": models["encoder"].state_dict(),
            "decoder": models["decoder"].state_dict(),
            "rssm": models["rssm"].state_dict(),
            "reward_model": models["reward_model"].state_dict(),
            "cont_model": models["cont_model"].state_dict(),
        }
        torch.save(checkpoint, wm_path)
        print(f"    World model saved to {wm_path}")

    print("\n  [2/5] Collecting position-latent data ...")
    data = collect_data(env, models, cfg, device)

    sanity = compute_sanity_metrics(data, env.geodesic, cfg)
    print(
        f"    Sanity: cells={sanity['coverage']:.2%} "
        f"({sanity['n_cells_visited']}/{sanity['n_free']})  "
        f"rooms={sanity['room_coverage']:.2%} "
        f"({sanity['n_rooms_visited']}/{sanity['n_rooms_total']})  "
        f"bridges={sanity['bridge_crossings']}  "
        f"var_h={sanity['var_h']:.2e}  var_encoder_e={sanity['var_encoder_e']:.2e}  "
        f"{'FAILED' if sanity['failed'] else 'ok'}"
    )

    print("\n  [3/5] Training GeoEncoder (post-hoc, temporal) ...")
    geo_temporal = train_geo_encoder(data, cfg, device, env.geodesic)

    geo_geo = None
    if cfg_pm.geo_supervised and cfg.geo_sup_epochs > 0:
        print("    Training GeoEncoder (geodesic-supervised) ...")
        geo_geo = train_geo_encoder_geodesic(data, cfg, device, env.geodesic)

    topo_head = None
    z_topo_all = None
    topo_meta = None
    if bool(cfg_pm.replay_topology):
        if data.get("episode_ids", None) is None:
            print("    Skipping replay-topology head: episode_ids missing from replay data.")
        elif data.get("encoder_emb", None) is None:
            print("    Skipping replay-topology head: encoder_emb missing from replay data.")
        else:
            print("    Training replay-topology head (pairwise same-region / bridge / neighbor / cut; oracle-free) ...")
            topo_head, z_topo_all, topo_meta = train_replay_topology_head(
                data=data,
                cfg_pm=cfg_pm,
                device=device,
                seed=int(cfg_pm.seed),
            )

    print("\n  [4/5] Running analyses ...")
    pos = data["pos"]
    episode_ids = data.get("episode_ids", None)
    feat_dict = _build_feature_dict(
        data, device, geo_temporal=geo_temporal, geo_geodesic=geo_geo
    )
    if z_topo_all is not None:
        feat_dict["g_topo(h,s)"] = z_topo_all

    probe_res = run_probes(pos, feat_dict, cfg, device, episode_ids=episode_ids)
    dist_res, dist_raw = run_distance_analysis(pos, feat_dict, env.geodesic, cfg)
    knn_res = run_knn_analysis(pos, feat_dict, env.geodesic, cfg)
    tc_res = run_trustworthiness_continuity(pos, feat_dict, env.geodesic, cfg, k=cfg.knn_k)

    # Directed / controllability geometry from replay transitions
    directed_geo_res = run_directed_geometry_analysis(data, env.geodesic)
    # Geometry of imagination vs replay (uses world model only)
    imagination_res = run_imagination_vs_replay_geometry(
        models, data, cfg, device, geo_temporal=geo_temporal
    )
    # Community / room structure in latent replay graphs
    community_res = run_latent_room_discovery(
        data, env.geodesic, feat_dict, cfg
    )
    # Metric class mismatch: replay-graph vs oracle geodesic
    metric_mismatch_res = run_metric_class_mismatch(
        data, feat_dict, env.geodesic, cfg
    )

    print("\n  [5/5] Generating plots ...")
    generate_plots(
        maze_name,
        pos,
        feat_dict,
        probe_res,
        dist_res,
        dist_raw,
        knn_res,
        cfg,
        device,
        out_dir,
    )

    env.close()

    results = {
        "sanity": sanity,
        "probes": probe_res,
        "distances": dist_res,
        "knn": knn_res,
        "trust_cont": tc_res,
        "directed_geometry": directed_geo_res,
        "imagination_vs_replay": imagination_res,
        "latent_communities": community_res,
        "metric_class_mismatch": metric_mismatch_res,
        "replay_topology_eval": run_replay_topology_eval(topo_meta),
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        import json

        json.dump(results, f, indent=2)

    print(f"\nResults saved to {cfg_pm.output_dir}/")
    return results


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0, help="Random seed (single run)")
    p.add_argument(
        "--output_dir",
        default="pointmaze_gr_results",
        help="Output directory for metrics and figures",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="Fast sanity-check run (reduced episodes/epochs)",
    )
    p.add_argument(
        "--geo_supervised",
        action="store_true",
        help="Also train a GeoEncoder supervised by ground-truth geodesic distances",
    )
    p.add_argument(
        "--wm_path",
        type=str,
        default="",
        help="Path to world model checkpoint to resume from",
    )
    p.add_argument(
        "--no_replay_topology",
        action="store_true",
        help="Disable the oracle-free replay-topology head",
    )
    p.add_argument("--replay_topology_train_frac", type=float, default=0.70, help="Episode fraction for replay-topology training split")
    p.add_argument("--replay_topology_val_frac", type=float, default=0.15, help="Episode fraction for replay-topology validation split")
    p.add_argument("--replay_topology_test_frac", type=float, default=0.15, help="Episode fraction for replay-topology test split")
    p.add_argument("--replay_topology_eval_repeats", type=int, default=24, help="Number of repeated pair-eval batches per split")
    return p.parse_args()


def main():
    args = parse_args()
    cfg_pm = PointMazeRunCfg(
        seed=args.seed,
        output_dir=args.output_dir,
        quick=bool(args.quick),
        geo_supervised=bool(args.geo_supervised),
        wm_path=args.wm_path,
        replay_topology=not bool(args.no_replay_topology),
        replay_topology_train_frac=float(args.replay_topology_train_frac),
        replay_topology_val_frac=float(args.replay_topology_val_frac),
        replay_topology_test_frac=float(args.replay_topology_test_frac),
        replay_topology_eval_repeats=int(args.replay_topology_eval_repeats),
    )
    run_single_pointmaze(cfg_pm)



def _subsample_indices(mask: np.ndarray, max_nodes: int):
    idx = np.flatnonzero(mask)
    if len(idx) <= int(max_nodes):
        return idx.astype(np.int64)
    keep = np.linspace(0, len(idx) - 1, num=int(max_nodes), dtype=int)
    return idx[keep].astype(np.int64)


def _pairwise_l2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    aa = (a * a).sum(axis=1, keepdims=True)
    d2 = np.maximum(aa + aa.T - 2.0 * (a @ a.T), 0.0)
    return np.sqrt(d2, dtype=np.float32)


def _build_replay_node_graph(
    local_feat: np.ndarray,
    ep_ids: np.ndarray,
    orig_idx: np.ndarray,
    knn_k: int = 4,
    cross_quantile: float = 0.10,
):
    n = int(len(orig_idx))
    adj = [set() for _ in range(n)]

    for i in range(n - 1):
        if int(ep_ids[i]) == int(ep_ids[i + 1]):
            adj[i].add(i + 1)
            adj[i + 1].add(i)

    if n <= 2:
        return [sorted(list(x)) for x in adj]

    d = _pairwise_l2(local_feat)
    np.fill_diagonal(d, np.inf)
    finite_vals = d[np.isfinite(d)]
    gate = float(np.quantile(finite_vals, float(cross_quantile))) if len(finite_vals) else float("inf")
    k_use = int(min(max(1, int(knn_k)), n - 1))
    nn_idx = np.argsort(d, axis=1)[:, :k_use]
    nn_sets = [set(map(int, row.tolist())) for row in nn_idx]

    for i in range(n):
        for j in nn_idx[i]:
            j = int(j)
            if float(d[i, j]) > gate:
                continue
            if i in nn_sets[j]:
                adj[i].add(j)
                adj[j].add(i)

    return [sorted(list(x)) for x in adj]


def _adj_list_to_csr(adj, n: int) -> csr_matrix:
    rows, cols = [], []
    for i, js in enumerate(adj):
        for j in js:
            rows.append(i)
            cols.append(int(j))
    if not rows:
        return csr_matrix((n, n), dtype=np.int8)
    data = np.ones(len(rows), dtype=np.int8)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _unique_undirected_edges(adj):
    edges = []
    for i, js in enumerate(adj):
        for j in js:
            j = int(j)
            if i < j:
                edges.append((int(i), int(j)))
    if not edges:
        return np.zeros((0, 2), dtype=np.int64)
    return np.asarray(edges, dtype=np.int64)


def _spectral_coords_from_adj(adj, dim: int) -> np.ndarray:
    n = int(len(adj))
    dim = int(max(1, dim))
    if n <= 2:
        return np.zeros((n, dim), dtype=np.float32)
    A = _adj_list_to_csr(adj, n).astype(np.float32)
    if A.nnz == 0:
        return np.zeros((n, dim), dtype=np.float32)
    L = csgraph_laplacian(A, normed=True)
    k = int(min(max(2, dim + 1), n - 1))
    try:
        vals, vecs = eigsh(L, k=k, which="SM")
        order = np.argsort(vals)
        vecs = vecs[:, order]
        vecs = vecs[:, 1 : 1 + dim]
    except Exception:
        dense = L.toarray() if hasattr(L, "toarray") else np.asarray(L)
        vals, vecs = np.linalg.eigh(dense)
        order = np.argsort(vals)
        vecs = vecs[:, order]
        vecs = vecs[:, 1 : 1 + dim]
    if vecs.shape[1] < dim:
        pad = np.zeros((n, dim - vecs.shape[1]), dtype=np.float32)
        vecs = np.concatenate([vecs.astype(np.float32), pad], axis=1)
    vecs = vecs.astype(np.float32)
    vecs -= vecs.mean(axis=0, keepdims=True)
    vecs /= np.maximum(vecs.std(axis=0, keepdims=True), 1e-6)
    return vecs


class ReplayTopologyHead(nn.Module):
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        topo_dim: int,
        hidden_dim: int,
        spec_dim: int,
    ):
        super().__init__()
        in_dim = int(deter_dim) + int(stoch_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.z_head = nn.Linear(hidden_dim, topo_dim)
        self.bridge_head = nn.Linear(hidden_dim, 1)
        self.spec_head = nn.Linear(hidden_dim, int(spec_dim))

        pair_hidden = max(64, hidden_dim // 2)
        self.pair_net = nn.Sequential(
            nn.Linear(2 * int(topo_dim), pair_hidden),
            nn.ELU(),
            nn.Linear(pair_hidden, pair_hidden),
            nn.ELU(),
        )
        self.same_region_head = nn.Linear(pair_hidden, 1)
        self.neighbor_head = nn.Linear(pair_hidden, 1)
        self.bridge_edge_head = nn.Linear(pair_hidden, 1)
        self.cut_head = nn.Linear(pair_hidden, 1)

        nn.init.orthogonal_(self.z_head.weight, gain=0.1)
        nn.init.zeros_(self.z_head.bias)

    def _trunk_out(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, s], dim=-1)
        return self.trunk(x)

    def encode(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        base = self._trunk_out(h, s)
        z = self.z_head(base)
        return F.normalize(z, dim=-1)

    def node_outputs(self, h: torch.Tensor, s: torch.Tensor):
        base = self._trunk_out(h, s)
        z = F.normalize(self.z_head(base), dim=-1)
        return {
            "z": z,
            "bridge_logit": self.bridge_head(base).squeeze(-1),
            "spec_pred": self.spec_head(base),
        }

    def pair_outputs_from_z(self, zi: torch.Tensor, zj: torch.Tensor):
        phi = torch.cat([torch.abs(zi - zj), zi * zj], dim=-1)
        hid = self.pair_net(phi)
        return {
            "same_region_logit": self.same_region_head(hid).squeeze(-1),
            "neighbor_logit": self.neighbor_head(hid).squeeze(-1),
            "bridge_edge_logit": self.bridge_edge_head(hid).squeeze(-1),
            "cut_logit": self.cut_head(hid).squeeze(-1),
        }

    def forward(self, h: torch.Tensor, s: torch.Tensor):
        return self.node_outputs(h, s)


def _build_replay_topology_teacher(
    local_feat: np.ndarray,
    ep_ids: np.ndarray,
    orig_idx: np.ndarray,
    knn_k: int,
    cross_quantile: float,
    spec_dim: int,
):
    adj = _build_replay_node_graph(
        local_feat=local_feat,
        ep_ids=ep_ids,
        orig_idx=orig_idx,
        knn_k=int(knn_k),
        cross_quantile=float(cross_quantile),
    )
    bridges = _find_bridges(adj)
    room_ids, n_rooms = _components_without_bridges(adj, bridges)
    room_ids = np.asarray(room_ids, dtype=np.int64)

    bridge_score = np.zeros(len(adj), dtype=np.float32)
    deg = np.array([max(1, len(js)) for js in adj], dtype=np.float32)
    for u, v in bridges:
        bridge_score[int(u)] += 1.0
        bridge_score[int(v)] += 1.0
    bridge_score = np.clip(bridge_score / deg, 0.0, 1.0)

    spec = _spectral_coords_from_adj(adj, dim=int(spec_dim))
    edges = _unique_undirected_edges(adj)

    bridge_set = {tuple(sorted((int(u), int(v)))) for (u, v) in bridges}
    if len(edges):
        edge_bridge_mask = np.array(
            [1.0 if tuple(sorted((int(u), int(v)))) in bridge_set else 0.0 for u, v in edges],
            dtype=np.float32,
        )
    else:
        edge_bridge_mask = np.zeros((0,), dtype=np.float32)

    if spec.shape[1] > 0:
        cut_side = (spec[:, 0] >= 0.0).astype(np.int64)
    else:
        cut_side = np.zeros((len(adj),), dtype=np.int64)

    return {
        "adj": adj,
        "bridges": bridges,
        "room_ids": room_ids,
        "n_rooms": int(n_rooms),
        "bridge_score": bridge_score,
        "spec": spec,
        "edges": edges,
        "edge_bridge_mask": edge_bridge_mask,
        "cut_side": cut_side,
    }


def _sample_distinct_random_pairs(
    rng: np.random.Generator,
    allowed_idx: np.ndarray,
    batch_size: int,
    pair_is_valid=None,
):
    allowed_idx = np.asarray(allowed_idx, dtype=np.int64)
    if len(allowed_idx) < 2:
        ii = np.zeros((0,), dtype=np.int64)
        jj = np.zeros((0,), dtype=np.int64)
        return ii, jj

    ii = np.empty((int(batch_size),), dtype=np.int64)
    jj = np.empty((int(batch_size),), dtype=np.int64)
    filled = 0
    max_tries = max(1000, int(batch_size) * 20)
    tries = 0
    while filled < int(batch_size) and tries < max_tries:
        tries += 1
        a = int(allowed_idx[rng.integers(0, len(allowed_idx))])
        b = int(allowed_idx[rng.integers(0, len(allowed_idx))])
        if a == b:
            continue
        if pair_is_valid is not None and not bool(pair_is_valid(a, b)):
            continue
        ii[filled] = a
        jj[filled] = b
        filled += 1
    return ii[:filled], jj[:filled]


def _balanced_bce_from_logits(pos_logits: torch.Tensor, neg_logits: torch.Tensor):
    logits = torch.cat([pos_logits, neg_logits], dim=0)
    targets = torch.cat(
        [
            torch.ones_like(pos_logits),
            torch.zeros_like(neg_logits),
        ],
        dim=0,
    )
    return F.binary_cross_entropy_with_logits(logits, targets)


def _shortest_path_matrix_from_adj(adj):
    n = int(len(adj))
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        q = [int(src)]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[src, u]
            for v in adj[u]:
                v = int(v)
                if dist[src, v] == np.inf:
                    dist[src, v] = du + 1.0
                    q.append(v)
    return dist


def _binary_auc_roc(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    pos_ranks = ranks[y_true == 1].sum()
    auc = (pos_ranks - (n_pos * (n_pos + 1) / 2.0)) / float(max(1, n_pos * n_neg))
    return float(auc)


def _binary_average_precision(y_true: np.ndarray, y_score: np.ndarray):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = int((y_true == 1).sum())
    if n_pos == 0:
        return None
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = 0
    fp = 0
    precisions = []
    recalls = []
    for y in y_sorted:
        if y == 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / float(max(1, tp + fp)))
        recalls.append(tp / float(n_pos))
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precisions, recalls):
        ap += p * max(0.0, r - prev_recall)
        prev_recall = r
    return float(ap)


def _summary_stats(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not vals:
        return {"mean": None, "std": None, "ci95": None, "n": 0}
    arr = np.asarray(vals, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    ci95 = float(1.96 * std / max(1.0, np.sqrt(len(arr))))
    return {
        "mean": float(arr.mean()),
        "std": std,
        "ci95": ci95,
        "n": int(len(arr)),
    }


def train_replay_topology_head(
    data: dict,
    cfg_pm: PointMazeRunCfg,
    device: torch.device,
    seed: int,
):
    h = np.asarray(data["h"], dtype=np.float32)
    s = np.asarray(data["s"], dtype=np.float32)
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    enc = np.asarray(data["encoder_emb"], dtype=np.float32)

    keep_idx = _subsample_indices(np.ones(len(h), dtype=bool), int(cfg_pm.replay_graph_max_nodes))
    h_sub = h[keep_idx]
    s_sub = s[keep_idx]
    ep_sub = ep_ids[keep_idx]
    enc_sub = enc[keep_idx]

    n = int(len(keep_idx))
    if n < 8:
        return None, None, None

    rng = np.random.default_rng(int(seed))
    uniq_eps = np.unique(ep_sub)
    rng.shuffle(uniq_eps)
    n_eps = int(len(uniq_eps))
    frac_sum = max(
        1e-6,
        float(cfg_pm.replay_topology_train_frac)
        + float(cfg_pm.replay_topology_val_frac)
        + float(cfg_pm.replay_topology_test_frac),
    )
    f_train = float(cfg_pm.replay_topology_train_frac) / frac_sum
    f_val = float(cfg_pm.replay_topology_val_frac) / frac_sum
    f_test = float(cfg_pm.replay_topology_test_frac) / frac_sum
    n_train_eps = max(1, int(round(f_train * n_eps)))
    n_val_eps = max(1, int(round(f_val * n_eps)))
    n_test_eps = max(1, int(round(f_test * n_eps)))
    if n_train_eps + n_val_eps + n_test_eps > n_eps:
        overflow = n_train_eps + n_val_eps + n_test_eps - n_eps
        n_train_eps = max(1, n_train_eps - overflow)
    if n_train_eps + n_val_eps >= n_eps:
        n_val_eps = max(1, n_eps - n_train_eps - 1)
    n_test_eps = max(1, n_eps - n_train_eps - n_val_eps)

    eps_train = set(map(int, uniq_eps[:n_train_eps].tolist()))
    eps_val = set(map(int, uniq_eps[n_train_eps : n_train_eps + n_val_eps].tolist()))
    eps_test = set(map(int, uniq_eps[n_train_eps + n_val_eps :].tolist()))
    if not eps_test:
        eps_test = set(map(int, uniq_eps[-1:].tolist()))
        eps_val = set(map(int, uniq_eps[n_train_eps : -1].tolist()))

    split_global = {
        "train": np.where(np.array([int(e) in eps_train for e in ep_sub], dtype=bool))[0].astype(np.int64),
        "val": np.where(np.array([int(e) in eps_val for e in ep_sub], dtype=bool))[0].astype(np.int64),
        "test": np.where(np.array([int(e) in eps_test for e in ep_sub], dtype=bool))[0].astype(np.int64),
    }
    if min(len(split_global["train"]), len(split_global["val"]), len(split_global["test"])) < 8:
        return None, None, None

    model = ReplayTopologyHead(
        deter_dim=int(h.shape[1]),
        stoch_dim=int(s.shape[1]),
        topo_dim=int(cfg_pm.replay_topology_dim),
        hidden_dim=int(cfg_pm.replay_topology_hidden),
        spec_dim=int(cfg_pm.replay_topology_spec_dim),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    h_t = torch.tensor(h_sub, dtype=torch.float32, device=device)
    s_t = torch.tensor(s_sub, dtype=torch.float32, device=device)

    split_ctx: Dict[str, dict] = {}
    for split_name, g_idx in split_global.items():
        feat_split = enc_sub[g_idx]
        ep_split = ep_sub[g_idx]
        teacher = _build_replay_topology_teacher(
            local_feat=feat_split,
            ep_ids=ep_split,
            orig_idx=keep_idx[g_idx],
            knn_k=int(cfg_pm.replay_graph_knn),
            cross_quantile=float(cfg_pm.replay_graph_quantile),
            spec_dim=int(cfg_pm.replay_topology_spec_dim),
        )
        room_ids = np.asarray(teacher["room_ids"], dtype=np.int64)
        cut_side = np.asarray(teacher["cut_side"], dtype=np.int64)
        adj = teacher["adj"]
        adj_sets = [set(map(int, js)) for js in adj]
        edges = np.asarray(teacher["edges"], dtype=np.int64)
        edge_bridge_mask = np.asarray(teacher["edge_bridge_mask"], dtype=np.float32)
        bridge_edges = edges[edge_bridge_mask > 0.5]
        non_bridge_edges = edges[edge_bridge_mask <= 0.5]
        bridge_score = np.asarray(teacher["bridge_score"], dtype=np.float32)
        spec = np.asarray(teacher["spec"], dtype=np.float32)

        groups = {}
        for i_local, rid in enumerate(room_ids.tolist()):
            groups.setdefault(int(rid), []).append(int(i_local))
        groups = {rid: np.asarray(nodes, dtype=np.int64) for rid, nodes in groups.items()}
        groups_nt = {rid: nodes for rid, nodes in groups.items() if len(nodes) >= 2}
        sp_dist = _shortest_path_matrix_from_adj(adj)

        split_ctx[split_name] = {
            "global_idx": g_idx.astype(np.int64),
            "teacher": teacher,
            "room_ids": room_ids,
            "cut_side": cut_side,
            "adj_sets": adj_sets,
            "bridge_edges": bridge_edges.astype(np.int64),
            "non_bridge_edges": non_bridge_edges.astype(np.int64),
            "bridge_score": bridge_score,
            "spec": spec,
            "groups_nt": groups_nt,
            "sp_dist": sp_dist,
            "bridge_t": torch.tensor(bridge_score, dtype=torch.float32, device=device),
            "spec_t": torch.tensor(spec, dtype=torch.float32, device=device),
        }

    def _sample_same_region_pairs(batch_size: int, split: str):
        ctx = split_ctx[split]
        allowed = np.arange(len(ctx["global_idx"]), dtype=np.int64)
        groups = ctx["groups_nt"]

        pos_i, pos_j = [], []
        if groups:
            room_keys = list(groups.keys())
            for _ in range(int(batch_size)):
                rid = int(room_keys[rng.integers(0, len(room_keys))])
                nodes = groups[rid]
                if len(nodes) < 2:
                    continue
                take = rng.choice(len(nodes), size=2, replace=False)
                pos_i.append(int(nodes[take[0]]))
                pos_j.append(int(nodes[take[1]]))
        pos_i = np.asarray(pos_i, dtype=np.int64)
        pos_j = np.asarray(pos_j, dtype=np.int64)

        room_ids = ctx["room_ids"]
        sp_dist = ctx["sp_dist"]

        def _valid_neg(a, b):
            if room_ids[int(a)] == room_ids[int(b)]:
                return False
            d = float(sp_dist[int(a), int(b)])
            return np.isfinite(d) and d <= 4.0

        neg_i, neg_j = _sample_distinct_random_pairs(
            rng=rng,
            allowed_idx=allowed,
            batch_size=max(1, len(pos_i)),
            pair_is_valid=_valid_neg,
        )
        return pos_i, pos_j, neg_i, neg_j

    def _sample_neighbor_pairs(batch_size: int, split: str):
        ctx = split_ctx[split]
        allowed = np.arange(len(ctx["global_idx"]), dtype=np.int64)
        edge_split = ctx["bridge_edges"] if len(ctx["bridge_edges"]) else np.asarray(ctx["teacher"]["edges"], dtype=np.int64)
        m = min(int(batch_size), len(edge_split))
        if m > 0:
            take = rng.choice(len(edge_split), size=m, replace=False)
            pos_i = edge_split[take, 0].astype(np.int64)
            pos_j = edge_split[take, 1].astype(np.int64)
        else:
            pos_i = np.zeros((0,), dtype=np.int64)
            pos_j = np.zeros((0,), dtype=np.int64)

        adj_sets = ctx["adj_sets"]
        sp_dist = ctx["sp_dist"]

        def _valid_neg(a, b):
            if int(b) in adj_sets[int(a)]:
                return False
            d = float(sp_dist[int(a), int(b)])
            return np.isfinite(d) and abs(d - 2.0) < 1e-6

        neg_i, neg_j = _sample_distinct_random_pairs(
            rng=rng,
            allowed_idx=allowed,
            batch_size=max(1, len(pos_i)),
            pair_is_valid=_valid_neg,
        )
        return pos_i, pos_j, neg_i, neg_j

    def _sample_bridge_edge_pairs(batch_size: int, split: str):
        ctx = split_ctx[split]
        pos_edges = ctx["bridge_edges"]
        neg_edges = ctx["non_bridge_edges"]
        m = min(int(batch_size), len(pos_edges), max(1, len(neg_edges)))
        if m <= 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )
        pos_take = rng.choice(len(pos_edges), size=m, replace=False)
        neg_take = rng.choice(len(neg_edges), size=m, replace=False)
        pos_i = pos_edges[pos_take, 0].astype(np.int64)
        pos_j = pos_edges[pos_take, 1].astype(np.int64)
        neg_i = neg_edges[neg_take, 0].astype(np.int64)
        neg_j = neg_edges[neg_take, 1].astype(np.int64)
        return pos_i, pos_j, neg_i, neg_j

    def _sample_cut_pairs(batch_size: int, split: str):
        ctx = split_ctx[split]
        allowed = np.arange(len(ctx["global_idx"]), dtype=np.int64)
        left = allowed[ctx["cut_side"][allowed] == 0]
        right = allowed[ctx["cut_side"][allowed] == 1]
        if len(left) == 0 or len(right) == 0:
            return (
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int64),
            )

        half = max(1, int(batch_size) // 2)
        same_i, same_j = [], []
        for _ in range(half):
            bucket = left if rng.random() < 0.5 else right
            if len(bucket) < 2:
                continue
            take = rng.choice(len(bucket), size=2, replace=False)
            same_i.append(int(bucket[take[0]]))
            same_j.append(int(bucket[take[1]]))
        same_i = np.asarray(same_i, dtype=np.int64)
        same_j = np.asarray(same_j, dtype=np.int64)

        opp_i = left[rng.integers(0, len(left), size=max(1, len(same_i)))]
        opp_j = right[rng.integers(0, len(right), size=max(1, len(same_i)))]
        return same_i, same_j, opp_i.astype(np.int64), opp_j.astype(np.int64)

    def _gather_pair_logits(z_all: torch.Tensor, split: str, ii: np.ndarray, jj: np.ndarray, key: str):
        if len(ii) == 0 or len(jj) == 0:
            return torch.zeros((0,), dtype=torch.float32, device=device)
        gidx = split_ctx[split]["global_idx"]
        out = model.pair_outputs_from_z(z_all[gidx[ii]], z_all[gidx[jj]])
        return out[key]

    def _compute_pair_task_losses(z_all: torch.Tensor, split: str, pair_batch: int):
        pos_i_sr, pos_j_sr, neg_i_sr, neg_j_sr = _sample_same_region_pairs(pair_batch, split)
        pos_i_nb, pos_j_nb, neg_i_nb, neg_j_nb = _sample_neighbor_pairs(pair_batch, split)
        pos_i_be, pos_j_be, neg_i_be, neg_j_be = _sample_bridge_edge_pairs(pair_batch, split)
        pos_i_ct, pos_j_ct, neg_i_ct, neg_j_ct = _sample_cut_pairs(pair_batch, split)

        losses = {}
        logits_pos = _gather_pair_logits(z_all, split, pos_i_sr, pos_j_sr, "same_region_logit")
        logits_neg = _gather_pair_logits(z_all, split, neg_i_sr, neg_j_sr, "same_region_logit")
        losses["same_region"] = _balanced_bce_from_logits(logits_pos, logits_neg) if len(logits_pos) and len(logits_neg) else torch.tensor(0.0, device=device)

        logits_pos = _gather_pair_logits(z_all, split, pos_i_nb, pos_j_nb, "neighbor_logit")
        logits_neg = _gather_pair_logits(z_all, split, neg_i_nb, neg_j_nb, "neighbor_logit")
        losses["neighbor"] = _balanced_bce_from_logits(logits_pos, logits_neg) if len(logits_pos) and len(logits_neg) else torch.tensor(0.0, device=device)

        logits_pos = _gather_pair_logits(z_all, split, pos_i_be, pos_j_be, "bridge_edge_logit")
        logits_neg = _gather_pair_logits(z_all, split, neg_i_be, neg_j_be, "bridge_edge_logit")
        losses["bridge_edge"] = _balanced_bce_from_logits(logits_pos, logits_neg) if len(logits_pos) and len(logits_neg) else torch.tensor(0.0, device=device)

        logits_pos = _gather_pair_logits(z_all, split, pos_i_ct, pos_j_ct, "cut_logit")
        logits_neg = _gather_pair_logits(z_all, split, neg_i_ct, neg_j_ct, "cut_logit")
        losses["cut"] = _balanced_bce_from_logits(logits_pos, logits_neg) if len(logits_pos) and len(logits_neg) else torch.tensor(0.0, device=device)

        if len(pos_i_nb) and len(neg_i_nb):
            gidx = split_ctx[split]["global_idx"]
            m = int(min(len(pos_i_nb), len(neg_i_nb)))
            if m <= 0:
                losses["margin"] = torch.tensor(0.0, device=device)
                return losses
            # Hard-negative filters can yield imbalanced counts; align pair counts.
            pos_take = np.arange(m, dtype=np.int64)
            neg_take = np.arange(m, dtype=np.int64)
            d_pos = torch.norm(
                z_all[gidx[pos_i_nb[pos_take]]] - z_all[gidx[pos_j_nb[pos_take]]],
                dim=-1,
            )
            d_neg = torch.norm(
                z_all[gidx[neg_i_nb[neg_take]]] - z_all[gidx[neg_j_nb[neg_take]]],
                dim=-1,
            )
            losses["margin"] = F.relu(d_pos + 0.25 - d_neg).mean()
        else:
            losses["margin"] = torch.tensor(0.0, device=device)

        return losses

    best_val = float("inf")
    best_state = None
    train_n = len(split_ctx["train"]["global_idx"])
    val_n = len(split_ctx["val"]["global_idx"])
    batch_nodes = int(min(cfg_pm.replay_topology_batch_nodes, train_n))
    pair_batch = int(min(max(128, batch_nodes // 2), max(32, train_n)))

    for epoch in range(int(cfg_pm.replay_topology_epochs)):
        model.train()
        train_g = split_ctx["train"]["global_idx"]
        node_batch_local = rng.choice(train_n, size=batch_nodes, replace=False) if batch_nodes < train_n else np.arange(train_n, dtype=np.int64)
        node_batch = train_g[node_batch_local]
        node_batch_t = torch.tensor(node_batch, dtype=torch.long, device=device)
        out = model(h_t[node_batch_t], s_t[node_batch_t])

        train_bridge_t = split_ctx["train"]["bridge_t"][torch.tensor(node_batch_local, dtype=torch.long, device=device)]
        train_spec_t = split_ctx["train"]["spec_t"][torch.tensor(node_batch_local, dtype=torch.long, device=device)]
        loss_bridge_node = F.binary_cross_entropy_with_logits(out["bridge_logit"], train_bridge_t)
        loss_spec = F.mse_loss(out["spec_pred"], train_spec_t)

        z_all = model.encode(h_t, s_t)
        pair_losses = _compute_pair_task_losses(z_all, split="train", pair_batch=pair_batch)

        loss = (
            float(cfg_pm.replay_topology_same_region_weight) * pair_losses["same_region"]
            + float(cfg_pm.replay_topology_bridge_node_weight) * loss_bridge_node
            + float(cfg_pm.replay_topology_neighbor_weight) * pair_losses["neighbor"]
            + float(cfg_pm.replay_topology_bridge_edge_weight) * pair_losses["bridge_edge"]
            + float(cfg_pm.replay_topology_cut_weight) * pair_losses["cut"]
            + float(cfg_pm.replay_topology_spec_weight) * loss_spec
            + float(cfg_pm.replay_topology_margin_weight) * pair_losses["margin"]
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        with torch.no_grad():
            model.eval()
            val_g = split_ctx["val"]["global_idx"]
            val_idx_t = torch.tensor(val_g, dtype=torch.long, device=device)
            val_out = model(h_t[val_idx_t], s_t[val_idx_t])
            val_bridge_node = F.binary_cross_entropy_with_logits(val_out["bridge_logit"], split_ctx["val"]["bridge_t"])
            val_spec = F.mse_loss(val_out["spec_pred"], split_ctx["val"]["spec_t"])

            z_all_val = model.encode(h_t, s_t)
            val_pair_losses = _compute_pair_task_losses(z_all_val, split="val", pair_batch=min(pair_batch, max(16, val_n)))
            val_loss = (
                float(cfg_pm.replay_topology_same_region_weight) * val_pair_losses["same_region"]
                + float(cfg_pm.replay_topology_bridge_node_weight) * val_bridge_node
                + float(cfg_pm.replay_topology_neighbor_weight) * val_pair_losses["neighbor"]
                + float(cfg_pm.replay_topology_bridge_edge_weight) * val_pair_losses["bridge_edge"]
                + float(cfg_pm.replay_topology_cut_weight) * val_pair_losses["cut"]
                + float(cfg_pm.replay_topology_spec_weight) * val_spec
                + float(cfg_pm.replay_topology_margin_weight) * val_pair_losses["margin"]
            )
            if float(val_loss.item()) < best_val:
                best_val = float(val_loss.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, int(cfg_pm.replay_topology_epochs) // 6) == 0:
            print(
                f"    g_topo epoch {epoch + 1}/{int(cfg_pm.replay_topology_epochs)}  "
                f"same={pair_losses['same_region'].item():.4f}  "
                f"bridge_node={loss_bridge_node.item():.4f}  "
                f"neighbor={pair_losses['neighbor'].item():.4f}  "
                f"bridge_edge={pair_losses['bridge_edge'].item():.4f}  "
                f"cut={pair_losses['cut'].item():.4f}  "
                f"spec={loss_spec.item():.4f}  margin={pair_losses['margin'].item():.4f}  "
                f"val={float(val_loss.item()):.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        h_sub_t = torch.tensor(h_sub, dtype=torch.float32, device=device)
        s_sub_t = torch.tensor(s_sub, dtype=torch.float32, device=device)
        out_sub = model(h_sub_t, s_sub_t)
        z_sub = out_sub["z"].cpu().numpy().astype(np.float32)
        bridge_prob_sub = torch.sigmoid(out_sub["bridge_logit"]).cpu().numpy().astype(np.float32)
        spec_pred_sub = out_sub["spec_pred"].cpu().numpy().astype(np.float32)

        h_all_t = torch.tensor(h, dtype=torch.float32, device=device)
        s_all_t = torch.tensor(s, dtype=torch.float32, device=device)
        out_all = model(h_all_t, s_all_t)
        z_all = out_all["z"].cpu().numpy().astype(np.float32)
        bridge_prob_all = torch.sigmoid(out_all["bridge_logit"]).cpu().numpy().astype(np.float32)

    def _pair_task_eval(split: str, key: str, sampler):
        repeats = int(max(4, cfg_pm.replay_topology_eval_repeats))
        bal_accs, aucs, aps = [], [], []
        for _ in range(repeats):
            pair_eval = max(64, min(512, len(split_ctx[split]["global_idx"])))
            pos_i, pos_j, neg_i, neg_j = sampler(pair_eval, split)
            if len(pos_i) == 0 or len(neg_i) == 0:
                continue
            gidx = split_ctx[split]["global_idx"]
            with torch.no_grad():
                z_t = torch.tensor(z_sub, dtype=torch.float32, device=device)
                logits_pos = model.pair_outputs_from_z(z_t[gidx[pos_i]], z_t[gidx[pos_j]])[key]
                logits_neg = model.pair_outputs_from_z(z_t[gidx[neg_i]], z_t[gidx[neg_j]])[key]
                prob_pos = torch.sigmoid(logits_pos).cpu().numpy()
                prob_neg = torch.sigmoid(logits_neg).cpu().numpy()
            pred_pos = (prob_pos >= 0.5).astype(np.float32)
            pred_neg = (prob_neg >= 0.5).astype(np.float32)
            bal_acc = 0.5 * (float((pred_pos == 1.0).mean()) + float((pred_neg == 0.0).mean()))
            y_true = np.concatenate([np.ones_like(prob_pos, dtype=np.int32), np.zeros_like(prob_neg, dtype=np.int32)], axis=0)
            y_score = np.concatenate([prob_pos, prob_neg], axis=0)
            bal_accs.append(bal_acc)
            aucs.append(_binary_auc_roc(y_true, y_score))
            aps.append(_binary_average_precision(y_true, y_score))
        return {
            "bal_acc": _summary_stats(bal_accs),
            "auroc": _summary_stats(aucs),
            "auprc": _summary_stats(aps),
        }

    def _bridge_node_eval(split: str):
        ctx = split_ctx[split]
        gidx = ctx["global_idx"]
        prob = bridge_prob_sub[gidx]
        true_cont = ctx["bridge_score"]
        y_true = (true_cont > 0.0).astype(np.int32)
        pred = (prob >= 0.5).astype(np.int32)
        pos_mask = y_true == 1
        neg_mask = y_true == 0
        if pos_mask.any() and neg_mask.any():
            bal_acc = 0.5 * (
                float((pred[pos_mask] == 1).mean()) +
                float((pred[neg_mask] == 0).mean())
            )
        else:
            bal_acc = None
        thresholds = np.linspace(0.05, 0.95, 19, dtype=np.float32)
        sweep = []
        for thr in thresholds:
            p = (prob >= float(thr)).astype(np.int32)
            if pos_mask.any() and neg_mask.any():
                ba = 0.5 * (float((p[pos_mask] == 1).mean()) + float((p[neg_mask] == 0).mean()))
            else:
                ba = None
            sweep.append({"thr": float(thr), "bal_acc": ba})
        return {
            "bal_acc@0.5": bal_acc,
            "auroc": _binary_auc_roc(y_true, prob),
            "auprc": _binary_average_precision(y_true, prob),
            "threshold_sweep": sweep,
            "n_pos": int(pos_mask.sum()),
            "n_neg": int(neg_mask.sum()),
        }

    pair_eval = {
        "same_region": {s: _pair_task_eval(s, "same_region_logit", _sample_same_region_pairs) for s in ("train", "val", "test")},
        "neighbor": {s: _pair_task_eval(s, "neighbor_logit", _sample_neighbor_pairs) for s in ("train", "val", "test")},
        "bridge_edge": {s: _pair_task_eval(s, "bridge_edge_logit", _sample_bridge_edge_pairs) for s in ("train", "val", "test")},
        "cut": {s: _pair_task_eval(s, "cut_logit", _sample_cut_pairs) for s in ("train", "val", "test")},
    }
    bridge_eval = {s: _bridge_node_eval(s) for s in ("train", "val", "test")}

    # Sanity: label permutation and latent ablation (val split, same-region task).
    val_sr = _pair_task_eval("val", "same_region_logit", _sample_same_region_pairs)
    perm_acc = None
    z_perm = z_sub.copy()
    rng.shuffle(z_perm, axis=0)
    pos_i, pos_j, neg_i, neg_j = _sample_same_region_pairs(256, "val")
    if len(pos_i) and len(neg_i):
        gidx = split_ctx["val"]["global_idx"]
        with torch.no_grad():
            z_t = torch.tensor(z_perm, dtype=torch.float32, device=device)
            lp = model.pair_outputs_from_z(z_t[gidx[pos_i]], z_t[gidx[pos_j]])["same_region_logit"]
            ln = model.pair_outputs_from_z(z_t[gidx[neg_i]], z_t[gidx[neg_j]])["same_region_logit"]
            pp = (torch.sigmoid(lp) >= 0.5).float().cpu().numpy()
            pn = (torch.sigmoid(ln) >= 0.5).float().cpu().numpy()
        perm_acc = 0.5 * (float((pp == 1.0).mean()) + float((pn == 0.0).mean()))

    meta = {
        "subset_idx": keep_idx.astype(np.int64),
        "z_subset": z_sub,
        "bridge_prob_subset": bridge_prob_sub,
        "bridge_prob_all": bridge_prob_all,
        "spec_pred_subset": spec_pred_sub,
        "split_sizes": {k: int(len(v["global_idx"])) for k, v in split_ctx.items()},
        "teacher_rooms_by_split": {k: int(v["teacher"]["n_rooms"]) for k, v in split_ctx.items()},
        "teacher_bridges_by_split": {k: int(len(v["teacher"]["bridges"])) for k, v in split_ctx.items()},
        "bridge_node_eval": bridge_eval,
        "pair_eval": pair_eval,
        "sanity_checks": {
            "val_same_region_bal_acc": val_sr["bal_acc"]["mean"] if val_sr["bal_acc"]["n"] > 0 else None,
            "val_same_region_bal_acc_permuted_latent": perm_acc,
        },
        "best_val_loss": float(best_val),
    }
    return model, z_all, meta


def run_replay_topology_eval(topo_meta: dict):
    if topo_meta is None:
        return None
    return {
        "split_sizes": topo_meta["split_sizes"],
        "teacher_rooms_by_split": topo_meta["teacher_rooms_by_split"],
        "teacher_bridges_by_split": topo_meta["teacher_bridges_by_split"],
        "bridge_node_eval": topo_meta["bridge_node_eval"],
        "pair_eval": topo_meta["pair_eval"],
        "sanity_checks": topo_meta["sanity_checks"],
        "best_val_loss": float(topo_meta["best_val_loss"]),
    }

def _build_replay_transition_matrices(data, geodesic: GeodesicComputer):
    """Build directed transition counts and undirected adjacency from replay trajectories."""
    traj_pos = data.get("traj_pos", [])
    n_free = geodesic.n_free
    trans_counts = np.zeros((n_free, n_free), dtype=np.int64)

    for traj in traj_pos:
        if len(traj) < 2:
            continue
        cells = _positions_to_cell_indices(geodesic, traj)
        for t in range(len(cells) - 1):
            u = int(cells[t])
            v = int(cells[t + 1])
            if u == v:
                continue
            trans_counts[u, v] += 1

    # Undirected adjacency for the replay graph (used for rooms/loops)
    adj = [[] for _ in range(n_free)]
    for i in range(n_free):
        js = np.where((trans_counts[i] + trans_counts[:, i]) > 0)[0]
        adj[i] = [int(j) for j in js]
    return trans_counts, adj


def run_directed_geometry_analysis(data, geodesic: GeodesicComputer):
    """D. Directed geometry / controllability geometry from replay transitions.

    We approximate controllability asymmetry by comparing transition counts
    trans(i->j) vs trans(j->i) over neighbouring free cells.
    """
    trans_counts, _ = _build_replay_transition_matrices(data, geodesic)
    n_free = geodesic.n_free

    asym_vals = []
    total_edges = 0
    strong_asym_edges = 0

    for i in range(n_free):
        for j in range(i + 1, n_free):
            cij = trans_counts[i, j]
            cji = trans_counts[j, i]
            total = cij + cji
            if total == 0:
                continue
            total_edges += 1
            asym = (cij - cji) / float(total)
            asym_abs = abs(asym)
            asym_vals.append(asym_abs)
            if asym_abs > 0.5:
                strong_asym_edges += 1

    if not asym_vals:
        return {
            "n_edges_with_data": 0,
            "mean_abs_asymmetry": 0.0,
            "median_abs_asymmetry": 0.0,
            "fraction_strong_asymmetry": 0.0,
        }

    asym_vals = np.asarray(asym_vals, dtype=np.float32)
    frac_strong = strong_asym_edges / max(total_edges, 1)
    return {
        "n_edges_with_data": int(total_edges),
        "mean_abs_asymmetry": float(asym_vals.mean()),
        "median_abs_asymmetry": float(np.median(asym_vals)),
        "fraction_strong_asymmetry": float(frac_strong),
    }


def run_imagination_vs_replay_geometry(
    models: dict,
    data: dict,
    cfg: TrainCfg,
    device: torch.device,
    geo_temporal=None,
):
    """E. Geometry of imagination vs replay.

    We:
      1) Sample starting (h,s) states from replay.
      2) Roll the RSSM forward in imagination using random actions.
      3) Compare neighbour / distance structure between replay and imagined
         g(h,s) (or h+s if GeoEncoder is absent).
    """
    rssm: RSSM = models["rssm"]
    rssm.eval()

    h_rep = data["h"]
    s_rep = data["s"]
    N_total = len(h_rep)
    if N_total < 4:
        return {}

    N = min(256, N_total)
    idx = np.random.choice(N_total, N, replace=False)

    h0 = torch.tensor(h_rep[idx], dtype=torch.float32, device=device)
    s0 = torch.tensor(s_rep[idx], dtype=torch.float32, device=device)

    act_dim = rssm.act_dim
    horizon = max(1, getattr(cfg, "imagination_horizon", 15))

    h_im = h0.clone()
    s_im = s0.clone()
    for _ in range(horizon):
        a = torch.empty(h_im.size(0), act_dim, device=device).uniform_(-1.0, 1.0)
        h_im, s_im = rssm.imagine_step(h_im, s_im, a)

    h_im_np = h_im.detach().cpu().numpy()
    s_im_np = s_im.detach().cpu().numpy()

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
        feat_rep = g_rep
        feat_im = g_im
        feat_name = "g(h,s)"
    else:
        feat_rep = np.concatenate([h_rep[idx], s_rep[idx]], axis=-1)
        feat_im = np.concatenate([h_im_np, s_im_np], axis=-1)
        feat_name = "h+s"

    # Pairwise distance geometry comparison between replay and imagination
    from scipy import stats as sp_stats

    dm_rep = np.linalg.norm(feat_rep[:, None, :] - feat_rep[None, :, :], axis=-1)
    dm_im = np.linalg.norm(feat_im[:, None, :] - feat_im[None, :, :], axis=-1)

    triu_idx = np.triu_indices(N, k=1)
    d_rep = dm_rep[triu_idx]
    d_im = dm_im[triu_idx]

    if len(d_rep) < 4:
        spearman = np.nan
        pearson = np.nan
    else:
        pr, _ = sp_stats.pearsonr(d_rep, d_im)
        sr, _ = sp_stats.spearmanr(d_rep, d_im)
        pearson = float(pr)
        spearman = float(sr)

    # Neighbour overlap between replay and imagination geometries
    k = min(10, N - 1)
    knn_rep = np.argsort(dm_rep, axis=1)[:, 1 : k + 1]
    knn_im = np.argsort(dm_im, axis=1)[:, 1 : k + 1]
    overlap = np.mean(
        [len(set(knn_rep[i]) & set(knn_im[i])) / float(k) for i in range(N)]
    )

    return {
        feat_name: {
            "pearson_replay_vs_imagination": float(pearson),
            "spearman_replay_vs_imagination": float(spearman),
            "knn_overlap_replay_vs_imagination": float(overlap),
            "k": int(k),
            "N_points": int(N),
            "horizon": int(horizon),
        }
    }


def run_latent_room_discovery(
    data: dict,
    geodesic: GeodesicComputer,
    features: dict,
    cfg: TrainCfg,
):
    """F. Community / room discovery in latent space.

    We:
      1) Aggregate latent embeddings per geodesic cell (mean over visits).
      2) Build a kNN graph in latent space.
      3) Run the same bridge/room decomposition oracle used on the geodesic
         graph and compare counts & agreement.
    """
    pos = data["pos"]
    cell_idx_all = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free

    # Oracle rooms/bridges/loops from geodesic graph
    dist_mat = geodesic.dist_matrix
    adj_oracle = _adj_from_distmat(dist_mat)
    bridges_oracle = _find_bridges(adj_oracle)
    comp_oracle, n_rooms_oracle = _components_without_bridges(adj_oracle, bridges_oracle)

    # Helper: component/loop stats for a latent kNN graph on cells
    def _latent_graph_stats(feat_name, feat_arr):
        # Aggregate per cell: mean embedding over all samples visiting that cell
        D = feat_arr.shape[1]
        cell_feats = np.zeros((n_free, D), dtype=np.float32)
        counts = np.zeros(n_free, dtype=np.int64)
        for f, c in zip(feat_arr, cell_idx_all):
            c_int = int(c)
            cell_feats[c_int] += f
            counts[c_int] += 1
        valid = counts > 0
        if not np.any(valid):
            return None
        cell_feats[valid] /= counts[valid, None]

        # Restrict to visited cells for stability
        idx_cells = np.where(valid)[0]
        Fv = cell_feats[idx_cells]
        n_cells = len(idx_cells)
        if n_cells < 4:
            return None

        # Build symmetric kNN graph in latent space
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

        # Map latent components back to global cell indices
        comp_lat_full = np.full(n_free, -1, dtype=np.int64)
        for local_idx, cell_global in enumerate(idx_cells):
            comp_lat_full[int(cell_global)] = comp_lat[local_idx]

        # Loops via Euler characteristic: E - N + C
        n_edges_lat = sum(len(v) for v in adj_lat) // 2
        n_loops_lat = max(n_edges_lat - n_cells + len(set(comp_lat)), 0)

        # Room-agreement over visited cells (same-room vs different-room consistency)
        visited = idx_cells
        agree = 0
        total_pairs = 0
        for i in range(len(visited)):
            for j in range(i + 1, len(visited)):
                ci = visited[i]
                cj = visited[j]
                same_oracle = comp_oracle[ci] == comp_oracle[cj]
                same_lat = comp_lat_full[ci] == comp_lat_full[cj]
                if same_lat == same_oracle:
                    agree += 1
                total_pairs += 1
        pair_agreement = agree / max(total_pairs, 1)

        return {
            "n_rooms_oracle": int(n_rooms_oracle),
            "n_rooms_latent": int(n_rooms_lat),
            "n_bridges_oracle": int(len(bridges_oracle)),
            "n_bridges_latent": int(len(bridges_lat)),
            "n_loops_latent": int(n_loops_lat),
            "room_pair_agreement": float(pair_agreement),
            "n_cells_visited": int(len(visited)),
        }

    results = {}
    for name, feat in features.items():
        stats = _latent_graph_stats(name, feat)
        if stats is not None:
            results[name] = stats
    return results


def run_metric_class_mismatch(
    data: dict,
    features: dict,
    geodesic: GeodesicComputer,
    cfg: TrainCfg,
):
    """G. Metric class mismatch: oracle geodesic vs replay-graph shortest paths.

    We:
      1) Build a replay transition graph over free cells and compute its
         shortest-path distances.
      2) Compare latent Euclidean distances against both:
         (a) oracle geodesic
         (b) replay-graph distances
    """
    from scipy import stats as sp_stats

    pos = data["pos"]
    cells = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free

    # Replay graph distances over cells
    trans_counts, adj_replay = _build_replay_transition_matrices(data, geodesic)

    # Build undirected adjacency (edge exists if either direction observed)
    adj = [[] for _ in range(n_free)]
    for i in range(n_free):
        js = np.where((trans_counts[i] + trans_counts[:, i]) > 0)[0]
        adj[i] = [int(j) for j in js]

    # BFS from each cell to get shortest-path distances in replay graph
    dist_replay = np.full((n_free, n_free), np.inf, dtype=np.float32)
    for src in range(n_free):
        dist_replay[src, src] = 0.0
        q = [src]
        while q:
            u = q.pop(0)
            du = dist_replay[src, u]
            for v in adj[u]:
                if dist_replay[src, v] == np.inf:
                    dist_replay[src, v] = du + 1.0
                    q.append(v)

    # ------------------------------------------------------------------
    # Oracle-free replay graph metric (timestep graph, directed)
    # ------------------------------------------------------------------
    episode_ids = data.get("episode_ids", None)

    def _oracle_free_replay_step_distances(n_pairs_target: int = 4000, max_sources: int = 64):
        """Sample directed shortest-path distances on the replay transition graph over timesteps.

        Nodes are individual replay samples i=0..N-1.
        Directed edges follow the behavior policy transitions: i -> i+1 when both
        belong to the same episode (episode_ids[i] == episode_ids[i+1]).

        Returns (ii, jj, d_steps) where d_steps[k] is #steps from ii[k] to jj[k] (finite).
        """
        if episode_ids is None:
            return None
        ep = np.asarray(episode_ids, dtype=np.int64)
        Np = len(ep)
        if Np < 4:
            return None

        # Build forward adjacency list (each node has at most 1 outgoing edge)
        nxt = np.full(Np, -1, dtype=np.int64)
        same = ep[:-1] == ep[1:]
        nxt[:-1][same] = np.arange(1, Np, dtype=np.int64)[same]

        rng = np.random.default_rng(0)
        sources = rng.choice(Np, size=min(max_sources, Np), replace=False)
        ii_list, jj_list, dd_list = [], [], []

        # For each source, BFS is trivial due to outdegree<=1: follow nxt chain.
        for src in sources:
            # Build distances along the chain until episode ends
            dist = {}
            cur = int(src)
            d = 0
            while cur != -1 and cur not in dist:
                dist[cur] = d
                cur = int(nxt[cur])
                d += 1
                if d > 10_000:  # safety
                    break

            if len(dist) < 3:
                continue

            # Sample targets from reachable nodes (excluding src)
            nodes = np.array(list(dist.keys()), dtype=np.int64)
            if len(nodes) <= 1:
                continue
            nodes = nodes[nodes != src]
            m = min(len(nodes), max(4, n_pairs_target // max(1, len(sources))))
            tgt = rng.choice(nodes, size=m, replace=False)

            ii_list.append(np.full(m, src, dtype=np.int64))
            jj_list.append(tgt)
            dd_list.append(np.array([dist[int(t)] for t in tgt], dtype=np.float32))

            if sum(len(x) for x in dd_list) >= n_pairs_target:
                break

        if not dd_list:
            return None
        ii = np.concatenate(ii_list, axis=0)
        jj = np.concatenate(jj_list, axis=0)
        dd = np.concatenate(dd_list, axis=0)
        # keep positive distances only
        keep = dd > 0
        if int(np.sum(keep)) < 4:
            return None
        return ii[keep], jj[keep], dd[keep]

    oracle_free_pairs = _oracle_free_replay_step_distances(
        n_pairs_target=min(cfg.n_pairs, 4000),
        max_sources=64,
    )

    def _oracle_free_mixed_graph_step_distances(
        feat: np.ndarray,
        n_pairs_target: int = 4000,
        max_sources: int = 64,
        n_graph_max: int = 1600,
        k_knn: int = 10,
    ):
        """Oracle-free branching graph over replay timesteps: temporal + latent-kNN edges.

        - Nodes: replay samples (timesteps) restricted to a subset of episodes for tractability.
        - Edges:
            (1) temporal neighbours within the same episode (undirected)
            (2) kNN neighbours in the provided latent feature (undirected)
        - Metric: unweighted shortest-path length in this graph ("steps to reach").

        Returns (ii_global, jj_global, d_steps) where indices refer to the original
        replay arrays, and d_steps are finite positive integers (float32).
        """
        if episode_ids is None:
            return None
        ep = np.asarray(episode_ids, dtype=np.int64)
        Np = len(ep)
        if Np < 8:
            return None

        # Choose a subset that preserves temporal adjacency: take whole episodes until limit.
        uniq = np.unique(ep)
        chosen = []
        total = 0
        for e in uniq:
            idx_e = np.where(ep == e)[0]
            if total + len(idx_e) > n_graph_max and total > 0:
                break
            chosen.append(idx_e)
            total += len(idx_e)
            if total >= n_graph_max:
                break
        if not chosen or total < 8:
            return None

        idx_global = np.concatenate(chosen, axis=0)
        idx_global.sort()
        M = len(idx_global)
        if M < 8:
            return None

        # Local indexing for graph ops
        g2l = {int(g): i for i, g in enumerate(idx_global.tolist())}

        adj_m = [set() for _ in range(M)]

        # (1) Temporal edges within chosen subset (undirected)
        for g in idx_global:
            g_int = int(g)
            g_next = g_int + 1
            if g_next < Np and ep[g_next] == ep[g_int] and g_next in g2l:
                u = g2l[g_int]
                v = g2l[g_next]
                adj_m[u].add(v)
                adj_m[v].add(u)

        # (2) kNN edges in feature space (undirected, within subset)
        k_use = int(min(max(2, k_knn), M - 1))
        X = feat[idx_global]
        dm = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
        np.fill_diagonal(dm, np.inf)
        knn = np.argsort(dm, axis=1)[:, :k_use]
        for i in range(M):
            for j in knn[i]:
                j = int(j)
                adj_m[i].add(j)
                adj_m[j].add(i)

        # Convert to lists for faster iteration
        adj_list = [list(s) for s in adj_m]

        rng = np.random.default_rng(0)
        sources = rng.choice(M, size=min(max_sources, M), replace=False)
        ii_list, jj_list, dd_list = [], [], []

        for src in sources:
            dist = np.full(M, -1, dtype=np.int32)
            dist[src] = 0
            q = [int(src)]
            qi = 0
            while qi < len(q):
                u = q[qi]
                qi += 1
                du = dist[u]
                for v in adj_list[u]:
                    if dist[v] == -1:
                        dist[v] = du + 1
                        q.append(int(v))

            reachable = np.where(dist > 0)[0]
            if len(reachable) < 4:
                continue
            m = min(
                len(reachable),
                max(8, n_pairs_target // max(1, len(sources))),
            )
            tgt = rng.choice(reachable, size=m, replace=False)

            ii_list.append(np.full(m, src, dtype=np.int64))
            jj_list.append(tgt.astype(np.int64))
            dd_list.append(dist[tgt].astype(np.float32))

            if sum(len(x) for x in dd_list) >= n_pairs_target:
                break

        if not dd_list:
            return None

        ii_l = np.concatenate(ii_list, axis=0)
        jj_l = np.concatenate(jj_list, axis=0)
        dd = np.concatenate(dd_list, axis=0)

        keep = dd > 0
        if int(np.sum(keep)) < 4:
            return None

        # Map local indices back to global replay indices
        ii_g = idx_global[ii_l[keep]]
        jj_g = idx_global[jj_l[keep]]
        return ii_g.astype(np.int64), jj_g.astype(np.int64), dd[keep].astype(np.float32)

    N = len(pos)
    n_pairs = min(cfg.n_pairs, N * (N - 1) // 2)
    i1 = np.random.randint(0, N, n_pairs)
    i2 = np.random.randint(0, N, n_pairs)
    mask = i1 != i2
    i1, i2 = i1[mask], i2[mask]

    # Oracle geodesic distances
    geo_d = np.array(
        [geodesic.distance(pos[a], pos[b]) for a, b in zip(i1, i2)],
        dtype=np.float32,
    )

    # Replay-graph distances between underlying cells
    c1 = cells[i1]
    c2 = cells[i2]
    rep_d = dist_replay[c1, c2]

    # Keep only pairs where both distances are finite and > 0
    valid = np.isfinite(geo_d) & np.isfinite(rep_d) & (geo_d > 0) & (rep_d > 0)
    i1, i2, geo_d, rep_d = i1[valid], i2[valid], geo_d[valid], rep_d[valid]

    if len(geo_d) < 4:
        return {}

    results = {}
    for name, feat in features.items():
        lat_d = np.linalg.norm(feat[i1] - feat[i2], axis=1)

        pr_geo, _ = sp_stats.pearsonr(lat_d, geo_d)
        sr_geo, _ = sp_stats.spearmanr(lat_d, geo_d)
        pr_rep, _ = sp_stats.pearsonr(lat_d, rep_d)
        sr_rep, _ = sp_stats.spearmanr(lat_d, rep_d)
        pr_geo_rep, _ = sp_stats.pearsonr(geo_d, rep_d)
        sr_geo_rep, _ = sp_stats.spearmanr(geo_d, rep_d)

        results[name] = {
            "pearson_latent_vs_geodesic": float(pr_geo),
            "spearman_latent_vs_geodesic": float(sr_geo),
            "pearson_latent_vs_replay_graph": float(pr_rep),
            "spearman_latent_vs_replay_graph": float(sr_rep),
            "pearson_geodesic_vs_replay_graph": float(pr_geo_rep),
            "spearman_geodesic_vs_replay_graph": float(sr_geo_rep),
        }

        # Oracle-free: compare latent distances to replay "steps-to-reach" distances
        if oracle_free_pairs is not None:
            ii_of, jj_of, d_steps = oracle_free_pairs
            lat_d_steps = np.linalg.norm(feat[ii_of] - feat[jj_of], axis=1)
            pr_of, _ = sp_stats.pearsonr(lat_d_steps, d_steps)
            sr_of, _ = sp_stats.spearmanr(lat_d_steps, d_steps)
            results[name]["pearson_latent_vs_replay_steps"] = float(pr_of)
            results[name]["spearman_latent_vs_replay_steps"] = float(sr_of)
            results[name]["n_pairs_replay_steps"] = int(len(d_steps))
        else:
            results[name]["pearson_latent_vs_replay_steps"] = None
            results[name]["spearman_latent_vs_replay_steps"] = None
            results[name]["n_pairs_replay_steps"] = 0

        # Oracle-free branching graph: temporal + latent-kNN edges, shortest-path steps
        mixed_pairs = _oracle_free_mixed_graph_step_distances(
            feat,
            n_pairs_target=min(cfg.n_pairs, 4000),
            max_sources=64,
            n_graph_max=1600,
            k_knn=int(getattr(cfg, "knn_k", 10)),
        )
        if mixed_pairs is not None:
            ii_m, jj_m, d_m = mixed_pairs
            lat_d_m = np.linalg.norm(feat[ii_m] - feat[jj_m], axis=1)
            pr_m, _ = sp_stats.pearsonr(lat_d_m, d_m)
            sr_m, _ = sp_stats.spearmanr(lat_d_m, d_m)
            results[name]["pearson_latent_vs_replay_mixed_steps"] = float(pr_m)
            results[name]["spearman_latent_vs_replay_mixed_steps"] = float(sr_m)
            results[name]["n_pairs_replay_mixed_steps"] = int(len(d_m))
            results[name]["replay_mixed_graph_n_nodes"] = int(len(np.unique(np.concatenate([ii_m, jj_m]))))
            results[name]["replay_mixed_graph_knn_k"] = int(getattr(cfg, "knn_k", 10))
        else:
            results[name]["pearson_latent_vs_replay_mixed_steps"] = None
            results[name]["spearman_latent_vs_replay_mixed_steps"] = None
            results[name]["n_pairs_replay_mixed_steps"] = 0
    return results


if __name__ == "__main__":
    main()