#!/usr/bin/env python3
"""
Latent-geometry evaluation on Gymnasium-Robotics PointMaze_Medium_Diverse_GR-v3.

This mirrors the pipeline in maze_geometry_test.py but swaps the custom
PointMazeEnv for the mujoco-based PointMaze_Medium_Diverse_GR-v3 env.

We:
  1. Wrap the Gymnasium-Robotics env to expose a Pixel-observation interface
     compatible with the existing Dreamer training code.
  2. Build a GeodesicComputer from the published MEDIUM_MAZE_DIVERSE_GR layout.
  3. Reuse the world-model training, GeoEncoder training, and analysis utilities
     from maze_geometry_test.py for a single-seed run.

Usage (single seed):
    python pointmaze_gr_geometry_test.py --seed 0
    python pointmaze_gr_geometry_test.py --seed 0 --quick
"""

import argparse
import os
from dataclasses import dataclass
from collections import deque
from typing import Optional, Tuple

import cv2
import gymnasium as gym
import gymnasium_robotics  # type: ignore
import numpy as np
import torch
from geom_head import GeoEncoder

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
    Actor,
    ContinueModel,
    ConvDecoder,
    ConvEncoder,
    RewardModel,
)


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
    replay_geo: bool = True
    replay_geo_dim: int = 32
    replay_geo_hidden: int = 256
    replay_geo_epochs: int = 300
    replay_geo_batch_pairs: int = 2048
    replay_graph_max_nodes: int = 2500
    replay_graph_knn: int = 4
    replay_graph_quantile: float = 0.10
    # Imagination-validated kNN (replaces encoder-only kNN / cycle filter)
    replay_graph_imagination_max_steps: int = 20
    replay_graph_imagination_h_thresh: float = 0.0  # 0 => auto from temporal ||Δh||
    replay_imagination_pair_batch: int = 512
    # Iterative bootstrapping: round 0 = temporal edges only; round k adds kNN in g_{k-1} + imagination filter
    replay_bootstrap_rounds: int = 3
    # Optional: geodesic correlation logging each round (uses maze layout — evaluation only)
    replay_bootstrap_log_geodesic_corr: bool = True
    replay_bootstrap_geo_pairs: int = 8000


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
        cfg_pm.replay_bootstrap_rounds = min(2, int(cfg_pm.replay_bootstrap_rounds))
        cfg_pm.replay_graph_imagination_max_steps = min(12, int(cfg_pm.replay_graph_imagination_max_steps))
        cfg_pm.replay_geo_epochs = min(80, int(cfg_pm.replay_geo_epochs))

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
        device = get_device()
        checkpoint = torch.load(cfg_pm.wm_path, map_location=device)
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

    # Imagination rollouts for graph validation use Dreamer-style Actor (mean action if deterministic).
    # maze_geometry_test does not train an actor; checkpoints may omit "actor" → randomly initialized.
    act_dim = env.action_space.shape[0]
    actor = Actor(
        cfg.deter_dim,
        cfg.stoch_dim,
        act_dim,
        cfg.hidden_dim,
    ).to(device)
    if "actor" in checkpoint:
        actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    models["actor"] = actor

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

    geo_replay = None
    replay_graph_eval = None
    if bool(cfg_pm.replay_geo):
        if data.get("episode_ids", None) is None:
            print("    Skipping replay-graph GeoEncoder: episode_ids missing from replay data.")
        elif data.get("encoder_emb", None) is None:
            print("    Skipping replay-graph GeoEncoder: encoder_emb missing from replay data.")
        else:
            print("    Training GeoEncoder (replay-graph, held-out episodes; oracle-free) ...")
            geo_replay, g_replay_all, replay_meta = train_geo_encoder_replay_graph(
                data,
                cfg_pm,
                device,
                seed=int(cfg_pm.seed),
                models=models,
                cfg=cfg,
                geodesic=env.geodesic,
                pos=data["pos"],
            )

    print("\n  [4/5] Running analyses ...")
    pos = data["pos"]
    episode_ids = data.get("episode_ids", None)
    feat_dict = _build_feature_dict(
        data, device, geo_temporal=geo_temporal, geo_geodesic=geo_geo
    )
    if geo_replay is not None:
        feat_dict["g_replay(h,s)"] = g_replay_all

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
    if geo_replay is not None:
        replay_graph_eval = run_replay_graph_eval(feat_dict, replay_meta, seed=int(cfg_pm.seed))

    topology_recovery_res = run_topology_recovery_eval(
        env.geodesic, data["pos"], feat_dict, cfg, seed=int(cfg_pm.seed)
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
        "replay_graph_eval": replay_graph_eval,
        "topology_recovery": topology_recovery_res,
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
        "--replay_bootstrap_rounds",
        type=int,
        default=3,
        help="Iterative replay-graph rounds (0th = temporal only; later add imagination kNN in g)",
    )
    p.add_argument(
        "--replay_imagination_max_steps",
        type=int,
        default=20,
        help="RSSM imagination horizon for validating kNN edges",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg_pm = PointMazeRunCfg(
        seed=args.seed,
        output_dir=args.output_dir,
        quick=bool(args.quick),
        geo_supervised=bool(args.geo_supervised),
        wm_path=args.wm_path,
        replay_bootstrap_rounds=int(args.replay_bootstrap_rounds),
        replay_graph_imagination_max_steps=int(args.replay_imagination_max_steps),
    )
    run_single_pointmaze(cfg_pm)


def _episode_split_ids(ep_ids: np.ndarray, seed: int, train_frac: float = 0.8):
    uniq = np.unique(ep_ids)
    rng = np.random.default_rng(int(seed))
    uniq = rng.permutation(uniq)
    n_train = max(1, int(round(float(train_frac) * len(uniq))))
    train_eps = set(int(x) for x in uniq[:n_train])
    test_eps = set(int(x) for x in uniq[n_train:])
    train_mask = np.array([int(e) in train_eps for e in ep_ids], dtype=bool)
    test_mask = ~train_mask
    return train_mask, test_mask, train_eps, test_eps


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
    """
    Oracle-free replay graph:
      - temporal edges between successive retained nodes in the same episode
      - sparse mutual-kNN edges in a local chart space (default: encoder_emb)
    """
    n = int(len(orig_idx))
    adj = [set() for _ in range(n)]

    # temporal edges on kept points:
    # connect successive retained nodes within the same episode, even if we subsampled
    for i in range(n - 1):
        if int(ep_ids[i]) == int(ep_ids[i + 1]):
            adj[i].add(i + 1)
            adj[i + 1].add(i)

    # mutual-kNN edges in local feature space
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
            # mutual kNN only
            if i in nn_sets[j]:
                adj[i].add(j)
                adj[j].add(i)

    return [sorted(list(x)) for x in adj]

from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import shortest_path


def _auto_imagination_h_threshold(h_nodes: np.ndarray, ep_ids: np.ndarray) -> float:
    """Scale-free closeness in h-space from typical one-step temporal moves."""
    d_temp = []
    for i in range(len(h_nodes) - 1):
        if int(ep_ids[i]) == int(ep_ids[i + 1]):
            d_temp.append(float(np.linalg.norm(h_nodes[i] - h_nodes[i + 1])))
    if not d_temp:
        return 1.0
    med = float(np.median(d_temp))
    return float(max(med * 2.0, 1e-6))


@torch.no_grad()
def _batched_imagination_reach_steps(
    rssm: RSSM,
    actor: Actor,
    h_src: torch.Tensor,
    s_src: torch.Tensor,
    h_tgt: torch.Tensor,
    max_steps: int,
    h_close_thresh: float,
    device: torch.device,
) -> torch.Tensor:
    """Minimum steps until ||h_rollout - h_tgt|| < thresh (per row). Unreachable -> max_steps + 1."""
    B = int(h_src.shape[0])
    h = h_src.clone()
    s = s_src.clone()
    done = torch.zeros(B, dtype=torch.bool, device=device)
    out = torch.full((B,), float(max_steps + 1), device=device, dtype=torch.float32)
    thr = torch.as_tensor(h_close_thresh, device=device, dtype=h.dtype)
    for t in range(int(max_steps)):
        #a, _ = actor.get_action(h, s, deterministic=True)
        a = torch.empty(h.size(0), rssm.act_dim, device=device).uniform_(-1.0, 1.0)
        h = rssm.deterministic_state_fwd(h, s, a)
        m, _ = rssm.state_prior(h_src, sample=False)
        s = m
        h = h_src.clone()
        dist = torch.norm(h - h_tgt, dim=-1)
        newly = (~done) & (dist < thr)
        out = torch.where(newly, torch.full_like(out, float(t + 1)), out)
        done = done | newly
        if bool(done.all().item()):
            break
    return out


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


def _build_and_solve_replay_graph(
    local_feat,
    ep_ids,
    orig_idx,
    knn_k=4,
    cross_quantile=0.05,
    *,
    include_chart_knn=True,
):
    n = len(orig_idx)
    adj = lil_matrix((n, n), dtype=np.float32)

    for i in range(n - 1):
        if ep_ids[i] == ep_ids[i + 1]:
            dt = float(orig_idx[i + 1] - orig_idx[i])
            adj[i, i + 1] = dt
            adj[i + 1, i] = dt

    if include_chart_knn and n > 2:
        d = _pairwise_l2(local_feat)
        np.fill_diagonal(d, np.inf)
        finite_vals = d[np.isfinite(d)]
        gate = float(np.quantile(finite_vals, cross_quantile))
        k_use = min(max(1, knn_k), n - 1)
        nn_idx = np.argsort(d, axis=1)[:, :k_use]
        nn_sets = [set(map(int, row)) for row in nn_idx]
        for i in range(n):
            for j in nn_idx[i]:
                j = int(j)
                if d[i, j] > gate:
                    continue
                if i in nn_sets[j]:   # mutual kNN only
                    adj[i, j] = float(d[i, j])
                    adj[j, i] = float(d[i, j])

    return shortest_path(csgraph=adj, directed=False, unweighted=False)


def _all_pairs_shortest_paths(adj):
    n = len(adj)
    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        q = deque([int(src)])
        while q:
            u = int(q.popleft())
            du = float(dist[src, u])
            for v in adj[u]:
                v = int(v)
                if not np.isfinite(dist[src, v]):
                    dist[src, v] = du + 1.0
                    q.append(v)
    return dist


def _replay_geo_corr_vs_geodesic(
    geo: GeoEncoder,
    h_t: torch.Tensor,
    s_t: torch.Tensor,
    pos: np.ndarray,
    train_idx: np.ndarray,
    geodesic: GeodesicComputer,
    n_pairs: int,
    seed: int,
    device: torch.device,
):
    """Pearson/Spearman between ||g_i-g_j|| and geodesic(cell_i, cell_j) on train nodes (eval / logging)."""
    from scipy import stats as sp_stats

    rng = np.random.default_rng(int(seed))
    n = int(len(train_idx))
    if n < 4:
        return {"pearson": float("nan"), "spearman": float("nan"), "n_pairs": 0}
    cells_all = _positions_to_cell_indices(geodesic, pos)
    cells = cells_all[train_idx.astype(np.int64)]
    ii = rng.integers(0, n, size=int(n_pairs))
    jj = rng.integers(0, n, size=int(n_pairs))
    same = ii == jj
    if np.any(same):
        jj[same] = (jj[same] + 1) % n
    d_geo = np.array(
        [float(geodesic.dist_matrix[int(cells[i]), int(cells[j])]) for i, j in zip(ii, jj)],
        dtype=np.float32,
    )
    valid = np.isfinite(d_geo) & (d_geo > 0)
    if int(valid.sum()) < 32:
        return {"pearson": float("nan"), "spearman": float("nan"), "n_pairs": int(valid.sum())}
    ii = ii[valid]
    jj = jj[valid]
    d_geo = d_geo[valid]
    geo.eval()
    with torch.no_grad():
        gi = geo(h_t[ii], s_t[ii])
        gj = geo(h_t[jj], s_t[jj])
        d_lat = torch.norm(gi - gj, dim=-1).cpu().numpy().astype(np.float32)
    if len(d_lat) < 8:
        return {"pearson": float("nan"), "spearman": float("nan"), "n_pairs": len(d_lat)}
    pr = float(np.corrcoef(d_lat, d_geo)[0, 1])
    sr = float(sp_stats.spearmanr(d_lat, d_geo).correlation)
    return {"pearson": pr, "spearman": sr, "n_pairs": int(len(d_lat))}


def train_geo_encoder_replay_graph(
    data: dict,
    cfg_pm: PointMazeRunCfg,
    device: torch.device,
    seed: int,
    models: dict,
    cfg: TrainCfg,
    geodesic: GeodesicComputer,
    pos: np.ndarray,
):
    """
    Train g_replay(h,s) on TRAIN episodes only with iterative bootstrapping.

    Round 0: temporal edges only → Dijkstra teacher.
    Later rounds: temporal + mutual kNN in g_{k-1} chart, imagination-validated.
    """
    rssm: RSSM = models["rssm"]
    actor: Actor = models["actor"]

    h = np.asarray(data["h"], dtype=np.float32)
    s = np.asarray(data["s"], dtype=np.float32)
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)

    train_mask, test_mask, train_eps, test_eps = _episode_split_ids(ep_ids, seed=int(seed), train_frac=0.8)
    train_idx = _subsample_indices(train_mask, int(cfg_pm.replay_graph_max_nodes))
    test_idx = _subsample_indices(test_mask, int(cfg_pm.replay_graph_max_nodes))

    h_train = h[train_idx]
    s_train = s[train_idx]
    ep_train = ep_ids[train_idx]

    h_test = h[test_idx]
    s_test = s[test_idx]
    ep_test = ep_ids[test_idx]

    geo = GeoEncoder(
        int(h.shape[1]),
        int(s.shape[1]),
        geo_dim=int(cfg_pm.replay_geo_dim),
        hidden_dim=int(cfg_pm.replay_geo_hidden),
    ).to(device)
    opt = torch.optim.Adam(geo.parameters(), lr=3e-4)

    h_t = torch.tensor(h_train, dtype=torch.float32, device=device)
    s_t = torch.tensor(s_train, dtype=torch.float32, device=device)
    h_test_t = torch.tensor(h_test, dtype=torch.float32, device=device)
    s_test_t = torch.tensor(s_test, dtype=torch.float32, device=device)

    n = int(len(h_train))
    rng = np.random.default_rng(int(seed))
    n_rounds = max(1, int(cfg_pm.replay_bootstrap_rounds))
    epochs_per_round = max(1, int(np.ceil(float(cfg_pm.replay_geo_epochs) / float(n_rounds))))

    bootstrap_round_stats = []
    g_chart_train: Optional[np.ndarray] = None

    for br in range(n_rounds):
        include_knn = br > 0
        chart_train = g_chart_train if include_knn else np.zeros((n, 1), dtype=np.float32)

        train_dist = _build_and_solve_replay_graph(
            chart_train,
            ep_train,
            train_idx,
            knn_k=int(cfg_pm.replay_graph_knn),
            cross_quantile=float(cfg_pm.replay_graph_quantile),
            include_chart_knn=include_knn,
        )

        print(
            f"    Bootstrap round {br + 1}/{n_rounds}: "
            f"{'temporal + imagination kNN in g' if include_knn else 'temporal only'}"
        )

        for epoch in range(epochs_per_round):
            if n < 4:
                break
            ii = rng.integers(0, n, size=int(cfg_pm.replay_geo_batch_pairs))
            jj = rng.integers(0, n, size=int(cfg_pm.replay_geo_batch_pairs))
            same = ii == jj
            if np.any(same):
                jj[same] = (jj[same] + 1) % n

            d_teacher = train_dist[ii, jj]
            valid = np.isfinite(d_teacher) & (d_teacher > 0)
            if int(valid.sum()) < 64:
                continue

            ii_v = ii[valid]
            jj_v = jj[valid]
            d_teacher_v = d_teacher[valid].astype(np.float32)

            gi = geo(h_t[ii_v], s_t[ii_v])
            gj = geo(h_t[jj_v], s_t[jj_v])
            d_lat = torch.norm(gi - gj, dim=-1)

            d_teacher_t = torch.tensor(d_teacher_v, dtype=torch.float32, device=device)
            w = 1.0 / torch.sqrt(d_teacher_t + 1.0)
            scale = d_teacher_t.mean().clamp_min(1e-3)
            loss = (w * (d_lat - d_teacher_t / scale) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            global_e = br * epochs_per_round + epoch + 1
            total_e = n_rounds * epochs_per_round
            if global_e % max(1, total_e // 6) == 0:
                print(
                    f"    g_replay round {br + 1} epoch {epoch + 1}/{epochs_per_round}  "
                    f"loss={loss.item():.4f}  train_nodes={n} test_nodes={len(test_idx)}"
                )

        geo.eval()
        with torch.no_grad():
            g_chart_train = geo(h_t, s_t).cpu().numpy().astype(np.float32)

        rd = {
            "round": int(br),
            "include_knn": bool(include_knn),
        }
        if bool(cfg_pm.replay_bootstrap_log_geodesic_corr):
            rd["g_vs_geodesic"] = _replay_geo_corr_vs_geodesic(
                geo,
                h_t,
                s_t,
                pos,
                train_idx,
                geodesic,
                int(cfg_pm.replay_bootstrap_geo_pairs),
                seed=int(seed) + br * 10007,
                device=device,
            )
        bootstrap_round_stats.append(rd)

    geo.eval()
    with torch.no_grad():
        g_chart_test = geo(h_test_t, s_test_t).cpu().numpy().astype(np.float32)

    test_dist = _build_and_solve_replay_graph(
        g_chart_test,
        ep_test,
        test_idx,
        knn_k=int(cfg_pm.replay_graph_knn),
        cross_quantile=float(cfg_pm.replay_graph_quantile),
        include_chart_knn=bool(n_rounds > 1),
    )

    with torch.no_grad():
        g_all = geo(
            torch.tensor(h, dtype=torch.float32, device=device),
            torch.tensor(s, dtype=torch.float32, device=device),
        ).cpu().numpy().astype(np.float32)

    meta = {
        "train_idx": train_idx.astype(np.int64),
        "test_idx": test_idx.astype(np.int64),
        "train_dist": train_dist,
        "test_dist": test_dist,
        "train_eps": sorted(list(train_eps)),
        "test_eps": sorted(list(test_eps)),
        "bootstrap_round_stats": bootstrap_round_stats,
        "replay_bootstrap_rounds": int(n_rounds),
    }
    return geo, g_all, meta


def _sample_dist_pairs(dist_mat: np.ndarray, n_pairs: int = 20000, seed: int = 0):
    rng = np.random.default_rng(int(seed))
    n = int(dist_mat.shape[0])
    ii = rng.integers(0, n, size=int(n_pairs))
    jj = rng.integers(0, n, size=int(n_pairs))
    d = dist_mat[ii, jj]
    valid = np.isfinite(d) & (d > 0) & (ii != jj)
    return ii[valid].astype(np.int64), jj[valid].astype(np.int64), d[valid].astype(np.float32)


def run_replay_graph_eval(
    feat_dict: dict,
    meta: dict,
    seed: int,
):
    """Evaluate all features on held-out replay-graph shortest-path distances."""
    from scipy import stats as sp_stats

    test_idx = meta["test_idx"]
    test_dist = meta["test_dist"]

    ii, jj, d_teacher = _sample_dist_pairs(test_dist, n_pairs=30000, seed=int(seed))
    out = {}
    for name, feat in feat_dict.items():
        f = np.asarray(feat, dtype=np.float32)[test_idx]
        d_lat = np.linalg.norm(f[ii] - f[jj], axis=1)
        if len(d_lat) < 16:
            continue
        pear = float(np.corrcoef(d_lat, d_teacher)[0, 1])
        spear = float(sp_stats.spearmanr(d_lat, d_teacher).correlation)
        out[name] = {"pearson": pear, "spearman": spear}
    return out


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


def run_topology_recovery_eval(
    geodesic: GeodesicComputer,
    pos: np.ndarray,
    features: dict,
    cfg: TrainCfg,
    seed: int = 0,
):
    """Threshold graphs in latent space vs oracle room/bridge structure (evaluation).

    For each representation, aggregate embeddings per free cell, sweep distance thresholds,
    build graphs { (i,j) : d(i,j) < τ }, and compare connected components to oracle rooms
    and bridge counts from the geodesic graph.
    """
    from scipy.sparse.csgraph import connected_components

    cell_idx_all = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free
    dist_mat = geodesic.dist_matrix
    adj_oracle = _adj_from_distmat(dist_mat)
    bridges_oracle = _find_bridges(adj_oracle)
    comp_oracle, n_rooms_oracle = _components_without_bridges(adj_oracle, bridges_oracle)
    n_bridges_o = int(len(bridges_oracle))

    def _aggregate(feat_arr: np.ndarray):
        D = int(feat_arr.shape[1])
        cell_feats = np.zeros((n_free, D), dtype=np.float32)
        counts = np.zeros(n_free, dtype=np.int64)
        for f, c in zip(feat_arr, cell_idx_all):
            ci = int(c)
            cell_feats[ci] += f
            counts[ci] += 1
        valid = counts > 0
        if not np.any(valid):
            return None
        vis = np.where(valid)[0]
        fv = cell_feats[vis] / counts[vis, None]
        return vis, fv

    out_all = {}
    for name, feat in features.items():
        f = np.asarray(feat, dtype=np.float32)
        ag = _aggregate(f)
        if ag is None:
            continue
        vis, fv = ag
        nv = int(len(vis))
        if nv < 4:
            continue
        dm = np.linalg.norm(fv[:, None, :] - fv[None, :, :], axis=-1)
        triu_i, triu_j = np.triu_indices(nv, k=1)
        flat = dm[triu_i, triu_j]
        flat = flat[np.isfinite(flat)]
        if len(flat) < 16:
            continue

        sweep = []
        best_bridge = None
        best_room = None
        for q in np.linspace(0.02, 0.48, 24):
            thresh = float(np.quantile(flat, float(q)))
            adj_lat = [[] for _ in range(nv)]
            for i in range(nv):
                for j in range(i + 1, nv):
                    if dm[i, j] < thresh:
                        adj_lat[i].append(j)
                        adj_lat[j].append(i)
            bridges_l = _find_bridges(adj_lat)
            n_bl = int(len(bridges_l))
            n_comp, labels = connected_components(_adj_list_to_csr(adj_lat, nv), directed=False)

            agree = 0
            tot = 0
            for a in range(nv):
                for b in range(a + 1, nv):
                    ga, gb = int(vis[a]), int(vis[b])
                    same_o = comp_oracle[ga] == comp_oracle[gb]
                    same_l = labels[a] == labels[b]
                    if same_l == same_o:
                        agree += 1
                    tot += 1
            p_agree = float(agree / max(tot, 1))

            rec = {
                "quantile": float(q),
                "thresh": thresh,
                "n_bridges_latent": n_bl,
                "n_components": int(n_comp),
                "room_pair_agreement": p_agree,
            }
            sweep.append(rec)
            err_b = abs(n_bl - n_bridges_o)
            if best_bridge is None or err_b < best_bridge["bridge_count_abs_error"]:
                best_bridge = {
                    "quantile": float(q),
                    "thresh": thresh,
                    "n_bridges_latent": n_bl,
                    "n_components": int(n_comp),
                    "room_pair_agreement": p_agree,
                    "bridge_count_abs_error": int(err_b),
                }
            if best_room is None or p_agree > best_room["room_pair_agreement"]:
                best_room = {
                    "quantile": float(q),
                    "thresh": thresh,
                    "n_bridges_latent": n_bl,
                    "n_components": int(n_comp),
                    "room_pair_agreement": p_agree,
                    "bridge_count_abs_error": int(err_b),
                }

        out_all[name] = {
            "n_bridges_oracle": n_bridges_o,
            "n_rooms_oracle": int(n_rooms_oracle),
            "best_by_bridge_match": best_bridge,
            "best_by_room_agreement": best_room,
            "threshold_sweep_head": sweep[:6],
            "threshold_sweep_tail": sweep[-4:],
            "seed": int(seed),
        }
    return out_all


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