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
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

import cv2
import gymnasium as gym
import gymnasium_robotics  # type: ignore
import numpy as np
import torch

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
    replay_topology_dim: int = 2
    replay_topology_hidden: int = 256
    replay_topology_epochs: int = 1000
    replay_topology_batch_pairs: int = 512
    replay_topology_pair_pool: int = 8000
    replay_topology_val_frac: float = 0.15
    replay_topology_n_ensemble: int = 5
    replay_laplacian: bool = True
    replay_laplacian_dim: int = 3
    replay_laplacian_graph_max: int = 1800
    replay_laplacian_knn_k: int = 10
    replay_cont: bool = True
    replay_cont_dim: int = 16
    replay_cont_hidden: int = 256
    replay_cont_epochs: int = 700
    replay_cont_batch: int = 512
    replay_cont_val_frac: float = 0.15
    replay_cont_pos_k: int = 3
    replay_cont_neg_k: int = 32
    replay_cont_temp: float = 0.10


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
        cfg_pm.replay_cont_epochs = min(80, int(cfg_pm.replay_cont_epochs))
        cfg_pm.replay_cont_batch = min(256, int(cfg_pm.replay_cont_batch))
        cfg_pm.replay_laplacian_graph_max = min(1200, int(cfg_pm.replay_laplacian_graph_max))

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
    z_lap_all = None
    lap_meta = None
    cont_head = None
    z_cont_all = None
    cont_meta = None
    if bool(cfg_pm.replay_topology):
        if data.get("episode_ids", None) is None:
            print("    Skipping replay-topology head: episode_ids missing from replay data.")
        elif data.get("encoder_emb", None) is None:
            print("    Skipping replay-topology head: encoder_emb missing from replay data.")
        else:
            print(
                "    Training g_topo(e) with step-count regression (MSE) and strict low-dim bottleneck ..."
            )
            topo_head, z_topo_all, topo_meta = train_replay_topology_head(
                data=data,
                cfg_pm=cfg_pm,
                device=device,
                seed=int(cfg_pm.seed),
                geodesic=env.geodesic,
                out_dir=out_dir,
            )
    if bool(cfg_pm.replay_cont):
        if data.get("episode_ids", None) is None:
            print("    Skipping replay-contrastive head: episode_ids missing from replay data.")
        elif data.get("encoder_emb", None) is None:
            print("    Skipping replay-contrastive head: encoder_emb missing from replay data.")
        else:
            print(
                "    Training g_cont(e) with replay-graph InfoNCE (temporal positives, random negatives) ..."
            )
            cont_head, z_cont_all, cont_meta = train_replay_contrastive_head(
                data=data,
                cfg_pm=cfg_pm,
                device=device,
                seed=int(cfg_pm.seed),
            )
    if bool(cfg_pm.replay_laplacian):
        if data.get("episode_ids", None) is None:
            print("    Skipping replay-laplacian embedding: episode_ids missing from replay data.")
        elif data.get("encoder_emb", None) is None:
            print("    Skipping replay-laplacian embedding: encoder_emb missing from replay data.")
        else:
            print("    Computing replay-mixed-graph Laplacian eigenvector embedding (post-hoc) ...")
            z_lap_all, lap_meta = compute_replay_laplacian_embedding(data, cfg_pm)

    print("\n  [4/5] Running analyses ...")
    pos = data["pos"]
    episode_ids = data.get("episode_ids", None)
    feat_dict = _build_feature_dict(
        data, device, geo_temporal=geo_temporal, geo_geodesic=geo_geo
    )
    if z_topo_all is not None:
        feat_dict["g_topo(e)"] = z_topo_all
    if z_cont_all is not None:
        feat_dict["g_cont(e)"] = z_cont_all
    if z_lap_all is not None:
        feat_dict["g_lap(e)"] = z_lap_all

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
        "replay_contrastive_eval": run_replay_contrastive_eval(cont_meta),
        "replay_laplacian_eval": run_replay_laplacian_eval(lap_meta),
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
    p.add_argument(
        "--no_replay_cont",
        action="store_true",
        help="Disable the replay-contrastive spectral head",
    )
    p.add_argument(
        "--no_replay_laplacian",
        action="store_true",
        help="Disable post-hoc replay-graph Laplacian embedding",
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
        replay_topology=not bool(args.no_replay_topology),
        replay_cont=not bool(args.no_replay_cont),
        replay_laplacian=not bool(args.no_replay_laplacian),
    )
    run_single_pointmaze(cfg_pm)



def _oracle_free_replay_step_distances(
    episode_ids: np.ndarray,
    n_pairs_target: int = 4000,
    max_sources: int = 64,
    rng: Optional[np.random.Generator] = None,
):
    """Directed shortest-path distances on the replay transition graph over timesteps.

    Nodes are replay indices i=0..N-1. Edges follow the behavior policy:
    i -> i+1 when both belong to the same episode.

    Returns (ii, jj, d_steps) where d_steps[k] is the number of steps from ii[k] to jj[k].
    """
    rng = rng or np.random.default_rng(0)
    ep = np.asarray(episode_ids, dtype=np.int64)
    Np = len(ep)
    if Np < 4:
        return None

    nxt = np.full(Np, -1, dtype=np.int64)
    same = ep[:-1] == ep[1:]
    nxt[:-1][same] = np.arange(1, Np, dtype=np.int64)[same]

    sources = rng.choice(Np, size=min(max_sources, Np), replace=False)
    ii_list, jj_list, dd_list = [], [], []

    for src in sources:
        dist = {}
        cur = int(src)
        d = 0
        while cur != -1 and cur not in dist:
            dist[cur] = d
            cur = int(nxt[cur])
            d += 1
            if d > 10_000:
                break

        # Need at least one successor on the chain (strictly positive d_steps).
        if len(dist) < 2:
            continue

        nodes = np.array(list(dist.keys()), dtype=np.int64)
        nodes = nodes[nodes != src]
        if len(nodes) <= 0:
            continue
        m = min(
            len(nodes),
            max(4, n_pairs_target // max(1, len(sources))),
        )
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
    keep = dd > 0
    if int(np.sum(keep)) < 1:
        return None
    return ii[keep], jj[keep], dd[keep]


def _make_temporal_dist_mlp(pair_in: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(pair_in, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ELU(),
        nn.Linear(hidden_dim, 1),
    )


class TemporalTopoHead(nn.Module):
    """Trunk + z on encoder_e only; K independent MLPs predict temporal step distance (no RSSM h/s)."""

    def __init__(
        self,
        embed_dim: int,
        topo_dim: int,
        hidden_dim: int,
        n_ensemble: int = 5,
    ):
        super().__init__()
        in_dim = int(embed_dim)
        self.n_ensemble = int(n_ensemble)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.z_head = nn.Linear(hidden_dim, topo_dim)
        pair_in = 2 * int(topo_dim)
        self.temporal_dist_heads = nn.ModuleList(
            [_make_temporal_dist_mlp(pair_in, hidden_dim) for _ in range(self.n_ensemble)]
        )
        nn.init.orthogonal_(self.z_head.weight, gain=0.1)
        nn.init.zeros_(self.z_head.bias)
        for head in self.temporal_dist_heads:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    nn.init.zeros_(m.bias)

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        z = self.z_head(self.trunk(e))
        return z

    def forward_pair_ensemble(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        """Stacked predictions [B, K] from K independent heads on concat(z_i, z_j)."""
        z_i = self.encode(e_i)
        z_j = self.encode(e_j)
        x = torch.cat([z_i, z_j], dim=-1)
        outs = [h(x).squeeze(-1) for h in self.temporal_dist_heads]
        return torch.stack(outs, dim=-1)

    def forward_pair(self, e_i: torch.Tensor, e_j: torch.Tensor) -> torch.Tensor:
        """Mean ensemble prediction [B] (convenience)."""
        return self.forward_pair_ensemble(e_i, e_j).mean(dim=-1)


class ReplayContrastiveHead(nn.Module):
    """Continuous head g_cont(e): projection on encoder features for spectral-like embedding."""

    def __init__(self, embed_dim: int, cont_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = int(embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.z_head = nn.Linear(hidden_dim, cont_dim)
        nn.init.orthogonal_(self.z_head.weight, gain=0.1)
        nn.init.zeros_(self.z_head.bias)

    def encode(self, e: torch.Tensor) -> torch.Tensor:
        return self.z_head(self.trunk(e))

    def encode_unit(self, e: torch.Tensor) -> torch.Tensor:
        z = self.encode(e)
        return F.normalize(z, dim=-1, eps=1e-8)


def _build_temporal_positive_table(
    episode_ids: np.ndarray,
    pos_k: int,
) -> list[np.ndarray]:
    """For each replay index i, cache indices j within <=pos_k temporal steps in episode."""
    ep = np.asarray(episode_ids, dtype=np.int64)
    n = int(len(ep))
    k = int(max(1, pos_k))
    table: list[np.ndarray] = []
    for i in range(n):
        cands = []
        e_i = int(ep[i])
        for d in range(1, k + 1):
            j = i + d
            if j < n and int(ep[j]) == e_i:
                cands.append(j)
            j = i - d
            if j >= 0 and int(ep[j]) == e_i:
                cands.append(j)
        if cands:
            table.append(np.asarray(cands, dtype=np.int64))
        else:
            table.append(np.zeros((0,), dtype=np.int64))
    return table


def _predict_pair_ensemble_variance_batch(
    model: TemporalTopoHead,
    e_t: torch.Tensor,
    ii: np.ndarray,
    jj: np.ndarray,
    device: torch.device,
    batch: int = 2048,
) -> np.ndarray:
    """Per-pair variance across the K ensemble predictions (numpy)."""
    model.eval()
    n = len(ii)
    var_out = np.empty(n, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, n, batch):
            end = min(n, start + batch)
            i_t = torch.tensor(ii[start:end], dtype=torch.long, device=device)
            j_t = torch.tensor(jj[start:end], dtype=torch.long, device=device)
            pred = model.forward_pair_ensemble(e_t[i_t], e_t[j_t])
            var_out[start:end] = pred.var(dim=-1, unbiased=False).cpu().numpy().astype(np.float32)
    return var_out


def plot_temporal_dist_error_map(
    geodesic: GeodesicComputer,
    pos: np.ndarray,
    ii: np.ndarray,
    jj: np.ndarray,
    pair_var: np.ndarray,
    out_path: str,
    title: str = "Temporal distance ensemble: variance of predictions (aggregated per cell)",
):
    """Heatmap of mean per-pair prediction variance over maze cells (from pair endpoints)."""
    cell_idx = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free
    acc = np.zeros(n_free, dtype=np.float64)
    cnt = np.zeros(n_free, dtype=np.int64)
    for a, b, e in zip(ii, jj, pair_var):
        ca = int(cell_idx[int(a)])
        cb = int(cell_idx[int(b)])
        acc[ca] += float(e)
        acc[cb] += float(e)
        cnt[ca] += 1
        cnt[cb] += 1
    mean = np.zeros(n_free, dtype=np.float32)
    m = cnt > 0
    mean[m] = (acc[m] / cnt[m]).astype(np.float32)

    gh, gw = len(geodesic.grid), len(geodesic.grid[0])
    grid = np.full((gh, gw), np.nan, dtype=np.float32)
    for idx, (r, c) in enumerate(geodesic.idx_to_cell):
        if idx < len(mean):
            v = float(mean[idx])
            if cnt[idx] > 0:
                grid[int(r), int(c)] = v

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(grid, aspect="equal", origin="upper", cmap="magma")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mean Var(pred_ensemble)")
    ax.set_title(title)
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_temporal_dist_error_scatter(
    pos: np.ndarray,
    ii: np.ndarray,
    pair_var: np.ndarray,
    out_path: str,
    rng: np.random.Generator,
    max_points: int = 6000,
    title: str = "Temporal distance ensemble: Var(pred) at pair sources (x, y)",
):
    """Scatter of (x,y) at pair anchors i, colored by variance across K ensemble preds."""
    n = len(ii)
    if n == 0:
        return
    if n > max_points:
        sel = rng.choice(n, size=max_points, replace=False)
    else:
        sel = np.arange(n, dtype=np.int64)
    x = pos[ii[sel], 0]
    y = pos[ii[sel], 1]
    c = pair_var[sel]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    sc = ax.scatter(x, y, c=c, cmap="magma", s=12, alpha=0.55, edgecolors="none")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Var(pred_ensemble)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (grid coords)")
    ax.set_ylabel("y (grid coords)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _compute_encoder_temporal_barrier_scores(
    encoder_emb: np.ndarray,
    episode_ids: np.ndarray,
    pairs_ii: np.ndarray,
    pairs_jj: np.ndarray,
    pairs_steps: np.ndarray,
    *,
    rng: np.random.Generator,
    k_nn: int = 12,
    barrier_steps: float = 25.0,
    candidate_pool: int = 1800,
    max_sources: int = 1500,
    min_known: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Score sources by how often encoder-near neighbors have large temporal steps.

    Uses oracle-free step distances only for pairs present in (pairs_ii, pairs_jj, pairs_steps).
    For each source i, we take encoder kNN neighbors among a candidate subset and count the
    fraction whose (i->j) step distance is known and >= barrier_steps.

    Returns (src_idx, src_score) arrays aligned (float32 scores in [0,1]).
    """
    enc = np.asarray(encoder_emb, dtype=np.float32)
    ep = np.asarray(episode_ids, dtype=np.int64)
    ii = np.asarray(pairs_ii, dtype=np.int64)
    jj = np.asarray(pairs_jj, dtype=np.int64)
    dd = np.asarray(pairs_steps, dtype=np.float32)

    N = int(enc.shape[0])
    if N < 8 or len(ii) < 8:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    # Sources: restrict to those that appear as ii in oracle-free pairs.
    src_all = np.unique(ii)
    if len(src_all) > int(max_sources):
        src = rng.choice(src_all, size=int(max_sources), replace=False).astype(np.int64)
    else:
        src = src_all.astype(np.int64)

    # Candidate set for approximate kNN (keeps compute reasonable for large encoder dims).
    M = int(min(N, max(64, int(candidate_pool))))
    cand = rng.choice(N, size=M, replace=False).astype(np.int64)
    # Ensure all sources are included
    if len(src) > 0:
        cand = np.unique(np.concatenate([cand, src], axis=0)).astype(np.int64)
    M = int(len(cand))

    # Map oracle-free pairs to per-source dict: (j -> steps). Keep the smallest steps if repeated.
    by_src: dict[int, dict[int, float]] = {}
    for a, b, d in zip(ii.tolist(), jj.tolist(), dd.tolist()):
        d0 = float(d)
        if d0 <= 0:
            continue
        if a not in by_src:
            by_src[a] = {b: d0}
        else:
            prev = by_src[a].get(b, None)
            if prev is None or d0 < prev:
                by_src[a][b] = d0

    k_use = int(max(3, min(int(k_nn), M - 1)))
    scores = np.zeros(len(src), dtype=np.float32)
    valid_src = np.zeros(len(src), dtype=bool)

    # Pre-extract candidate embeddings for speed
    enc_c = enc[cand]  # [M, D]

    for idx_s, s_idx in enumerate(src.tolist()):
        s = int(s_idx)
        neigh_steps = by_src.get(s, None)
        if not neigh_steps:
            continue

        # Distances from source to candidate subset
        v = enc[s]
        d = np.linalg.norm(enc_c - v[None, :], axis=1)
        # Exclude self if present in cand
        d[cand == s] = np.inf
        nn_local = np.argpartition(d, kth=k_use - 1)[:k_use]
        nn_global = cand[nn_local]

        # Count "near in encoder but far in temporal steps"
        known = 0
        far = 0
        for jg in nn_global.tolist():
            j = int(jg)
            # Optional: keep same-episode comparisons to reduce cross-episode noise.
            if int(ep[j]) != int(ep[s]):
                continue
            dstep = neigh_steps.get(j, None)
            if dstep is None:
                continue
            known += 1
            if float(dstep) >= float(barrier_steps):
                far += 1

        if known >= int(min_known):
            scores[idx_s] = float(far) / float(max(known, 1))
            valid_src[idx_s] = True

    src_v = src[valid_src]
    score_v = scores[valid_src].astype(np.float32)
    return src_v.astype(np.int64), score_v


def plot_encoder_temporal_barrier_heatmap(
    geodesic: GeodesicComputer,
    pos: np.ndarray,
    src_idx: np.ndarray,
    src_score: np.ndarray,
    out_path: str,
    title: str = "Encoder-near / temporal-far conflicts (source-only)",
):
    """Heatmap of mean conflict score per cell, assigning score to source cell only."""
    if len(src_idx) == 0:
        return
    cell_idx_all = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free
    acc = np.zeros(n_free, dtype=np.float64)
    cnt = np.zeros(n_free, dtype=np.int64)
    for i, sc in zip(src_idx.tolist(), src_score.tolist()):
        c = int(cell_idx_all[int(i)])
        acc[c] += float(sc)
        cnt[c] += 1
    mean = np.zeros(n_free, dtype=np.float32)
    m = cnt > 0
    mean[m] = (acc[m] / cnt[m]).astype(np.float32)

    gh, gw = len(geodesic.grid), len(geodesic.grid[0])
    grid = np.full((gh, gw), np.nan, dtype=np.float32)
    for idx, (r, c) in enumerate(geodesic.idx_to_cell):
        if cnt[idx] > 0:
            grid[int(r), int(c)] = float(mean[idx])

    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(grid, aspect="equal", origin="upper", cmap="viridis", vmin=0.0, vmax=1.0)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="P(temporal steps ≥ barrier | encoder-kNN)")
    ax.set_title(title)
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _bridge_sanity_for_conflict_scores(
    geodesic: GeodesicComputer,
    pos: np.ndarray,
    src_idx: np.ndarray,
    src_score: np.ndarray,
    top_frac: float = 0.15,
) -> dict:
    """Sanity: do high-conflict cells overlap oracle bridge cells?"""
    if len(src_idx) == 0:
        return {
            "n_sources_scored": 0,
            "n_cells_scored": 0,
            "n_bridges_oracle": int(0),
            "mean_score_bridge": None,
            "mean_score_nonbridge": None,
            "top_frac": float(top_frac),
            "top_cells_bridge_frac": None,
        }

    # Oracle bridge cells from geodesic adjacency
    dist_mat = geodesic.dist_matrix
    adj_oracle = _adj_from_distmat(dist_mat)
    bridges = _find_bridges(adj_oracle)
    bridge_cells: set[int] = set()
    for u, v in bridges:
        bridge_cells.add(int(u))
        bridge_cells.add(int(v))

    cell_idx_all = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free
    acc = np.zeros(n_free, dtype=np.float64)
    cnt = np.zeros(n_free, dtype=np.int64)
    for i, sc in zip(src_idx.tolist(), src_score.tolist()):
        c = int(cell_idx_all[int(i)])
        acc[c] += float(sc)
        cnt[c] += 1
    m = cnt > 0
    mean = np.zeros(n_free, dtype=np.float32)
    mean[m] = (acc[m] / cnt[m]).astype(np.float32)

    scored_cells = np.where(m)[0].astype(np.int64)
    if len(scored_cells) == 0:
        return {
            "n_sources_scored": int(len(src_idx)),
            "n_cells_scored": 0,
            "n_bridges_oracle": int(len(bridge_cells)),
            "mean_score_bridge": None,
            "mean_score_nonbridge": None,
            "top_frac": float(top_frac),
            "top_cells_bridge_frac": None,
        }

    bridge_mask = np.array([int(c) in bridge_cells for c in scored_cells], dtype=bool)
    mean_bridge = float(np.mean(mean[scored_cells[bridge_mask]])) if np.any(bridge_mask) else None
    mean_non = float(np.mean(mean[scored_cells[~bridge_mask]])) if np.any(~bridge_mask) else None

    # Top cells overlap
    k = max(1, int(round(float(top_frac) * len(scored_cells))))
    order = np.argsort(mean[scored_cells])[::-1]
    top_cells = scored_cells[order[:k]]
    top_bridge_frac = float(np.mean([int(c) in bridge_cells for c in top_cells.tolist()])) if len(top_cells) else None

    return {
        "n_sources_scored": int(len(src_idx)),
        "n_cells_scored": int(len(scored_cells)),
        "n_bridges_oracle": int(len(bridge_cells)),
        "mean_score_bridge": mean_bridge,
        "mean_score_nonbridge": mean_non,
        "top_frac": float(top_frac),
        "top_cells_bridge_frac": top_bridge_frac,
    }


def _build_mixed_replay_graph(
    feat: np.ndarray,
    episode_ids: np.ndarray,
    *,
    n_graph_max: int = 1800,
    k_knn: int = 10,
) -> tuple[np.ndarray, dict[int, int], list[list[int]]]:
    """Mixed replay graph over timestep indices: temporal edges + latent-kNN edges.

    Graph construction matches run_metric_class_mismatch:
      (1) temporal undirected edges between consecutive timesteps in same episode
      (2) undirected kNN edges in latent feature space (within the selected subset)
    """
    ep = np.asarray(episode_ids, dtype=np.int64)
    X = np.asarray(feat, dtype=np.float32)
    N = len(ep)
    if N < 2:
        return np.zeros((0,), dtype=np.int64), {}, []

    uniq = np.unique(ep)
    chosen = []
    total = 0
    for e in uniq:
        idx_e = np.where(ep == e)[0]
        if total + len(idx_e) > int(n_graph_max) and total > 0:
            break
        chosen.append(idx_e)
        total += len(idx_e)
        if total >= int(n_graph_max):
            break
    if not chosen:
        return np.zeros((0,), dtype=np.int64), {}, []

    idx_global = np.concatenate(chosen, axis=0).astype(np.int64)
    idx_global.sort()
    M = len(idx_global)
    if M < 2:
        return np.zeros((0,), dtype=np.int64), {}, []

    g2l = {int(g): i for i, g in enumerate(idx_global.tolist())}
    adj_m = [set() for _ in range(M)]

    # Temporal undirected edges
    for g in idx_global.tolist():
        g_next = int(g) + 1
        if g_next < N and ep[g_next] == ep[int(g)] and g_next in g2l:
            u = g2l[int(g)]
            v = g2l[g_next]
            adj_m[u].add(v)
            adj_m[v].add(u)

    # Latent-kNN undirected edges
    if M > 1:
        k_use = int(min(max(2, int(k_knn)), M - 1))
        Xs = X[idx_global]
        dm = np.linalg.norm(Xs[:, None, :] - Xs[None, :, :], axis=-1)
        np.fill_diagonal(dm, np.inf)
        knn = np.argsort(dm, axis=1)[:, :k_use]
        for i in range(M):
            for j in knn[i]:
                j = int(j)
                adj_m[i].add(j)
                adj_m[j].add(i)

    adj_list = [list(s) for s in adj_m]
    return idx_global, g2l, adj_list


def compute_replay_laplacian_embedding(
    data: dict,
    cfg_pm: PointMazeRunCfg,
) -> tuple[Optional[np.ndarray], Optional[dict]]:
    """Post-hoc spectral embedding from normalized replay mixed-graph Laplacian."""
    ep_ids = np.asarray(data.get("episode_ids", None), dtype=np.int64)
    enc = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if enc.ndim != 2 or len(ep_ids) != int(enc.shape[0]):
        return None, None

    idx_global, _, adj_list = _build_mixed_replay_graph(
        enc,
        ep_ids,
        n_graph_max=int(cfg_pm.replay_laplacian_graph_max),
        k_knn=int(cfg_pm.replay_laplacian_knn_k),
    )
    m = int(len(idx_global))
    if m < 8:
        return None, None

    # Build unweighted symmetric adjacency.
    a = np.zeros((m, m), dtype=np.float64)
    for i, nei in enumerate(adj_list):
        for j in nei:
            jj = int(j)
            if jj == i:
                continue
            a[i, jj] = 1.0
            a[jj, i] = 1.0

    deg = np.sum(a, axis=1)
    if np.all(deg <= 0):
        return None, None

    inv_sqrt = np.zeros(m, dtype=np.float64)
    nz = deg > 1e-12
    inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    d_inv = np.diag(inv_sqrt)
    l = np.eye(m, dtype=np.float64) - d_inv @ a @ d_inv

    evals, evecs = np.linalg.eigh(l)
    dim = int(min(max(2, int(cfg_pm.replay_laplacian_dim)), 3))
    # Skip the first trivial eigenvector.
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
        "n_nodes": int(m),
        "dim": int(dim),
        "knn_k": int(cfg_pm.replay_laplacian_knn_k),
        "graph_max": int(cfg_pm.replay_laplacian_graph_max),
        "eigvals_used": [float(x) for x in evals[start:end].tolist()],
        "feat_dict_key": "g_lap(e)",
        "objective": "normalized_graph_laplacian_eigenvectors_posthoc",
    }
    return z_all, meta


def _spearman_rho_numpy(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation of rank-transformed x, y (Spearman rho). Returns in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        return 1.0
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    sx = float(np.sqrt(np.sum(rx**2)))
    sy = float(np.sqrt(np.sum(ry**2)))
    if sx < 1e-12 or sy < 1e-12:
        return 1.0
    return float(np.dot(rx, ry) / (sx * sy))


def _compute_encoder_temporal_local_global_disagreement(
    encoder_emb: np.ndarray,
    episode_ids: np.ndarray,
    *,
    rng: np.random.Generator,
    k_nn: int = 12,
    candidate_pool: int = 1800,
    max_sources: int = 2000,
    min_labeled_neighbors: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-source disagreement between local encoder distances and mixed-graph shortest paths.

    For each source s:
      - pick encoder-kNN neighbors in a candidate subset
      - run BFS on the mixed replay graph (temporal + latent-kNN edges)
      - compute Spearman(d_encoder(s,j), d_mixed_sp(s,j)) on those same neighbors
      - score = clip(1 - rho, 0, 1)
    """
    enc = np.asarray(encoder_emb, dtype=np.float32)
    N = int(enc.shape[0])
    if N < 8:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    idx_global, g2l, adj_list = _build_mixed_replay_graph(
        enc,
        episode_ids,
        n_graph_max=int(candidate_pool),
        k_knn=int(k_nn),
    )
    if len(idx_global) < 8:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    n_src = min(int(max_sources), len(idx_global))
    src = rng.choice(idx_global, size=n_src, replace=False).astype(np.int64)

    M = int(min(len(idx_global), max(64, int(candidate_pool))))
    cand = rng.choice(idx_global, size=M, replace=False).astype(np.int64)
    if len(src) > 0:
        cand = np.unique(np.concatenate([cand, src], axis=0)).astype(np.int64)
    M = int(len(cand))

    k_use = int(max(3, min(int(k_nn), M - 1)))
    enc_c = enc[cand]

    out_idx: list[int] = []
    out_score: list[float] = []

    for s in src.tolist():
        s = int(s)
        s_local = g2l.get(s, None)
        if s_local is None:
            continue

        # BFS shortest paths from source on mixed graph
        dist = np.full(len(idx_global), -1, dtype=np.int32)
        dist[s_local] = 0
        q = [int(s_local)]
        qi = 0
        while qi < len(q):
            u = q[qi]
            qi += 1
            du = dist[u]
            for v in adj_list[u]:
                if dist[v] == -1:
                    dist[v] = du + 1
                    q.append(int(v))

        v = enc[s]
        d_enc_all = np.linalg.norm(enc_c - v[None, :], axis=1)
        d_enc_all[cand == s] = np.inf
        nn_local = np.argpartition(d_enc_all, kth=k_use - 1)[:k_use]
        nn_global = cand[nn_local]

        d_enc_list: list[float] = []
        d_temp_list: list[float] = []
        for jg in nn_global.tolist():
            j = int(jg)
            if j == s:
                continue
            j_local = g2l.get(j, None)
            if j_local is None:
                continue
            dstep = int(dist[j_local])
            if dstep <= 0:
                continue
            d_enc_list.append(float(np.linalg.norm(enc[s] - enc[j])))
            d_temp_list.append(float(dstep))

        if len(d_enc_list) < int(min_labeled_neighbors):
            continue

        d_enc_arr = np.asarray(d_enc_list, dtype=np.float64)
        d_temp_arr = np.asarray(d_temp_list, dtype=np.float64)
        rho = _spearman_rho_numpy(d_enc_arr, d_temp_arr)
        disagree = float(np.clip(1.0 - rho, 0.0, 1.0))
        out_idx.append(s)
        out_score.append(disagree)

    if not out_idx:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    return np.asarray(out_idx, dtype=np.int64), np.asarray(out_score, dtype=np.float32)


def plot_encoder_temporal_local_global_disagreement_heatmap(
    geodesic: GeodesicComputer,
    pos: np.ndarray,
    src_idx: np.ndarray,
    disagreement: np.ndarray,
    out_path: str,
    title: str = "Exploration: encoder-local vs replay-temporal geometry (high = disagree)",
):
    """Full maze heatmap: mean disagreement score per free cell (sources only)."""
    if len(src_idx) == 0:
        return
    cell_idx_all = _positions_to_cell_indices(geodesic, pos)
    n_free = geodesic.n_free
    acc = np.zeros(n_free, dtype=np.float64)
    cnt = np.zeros(n_free, dtype=np.int64)
    for i, sc in zip(src_idx.tolist(), disagreement.tolist()):
        c = int(cell_idx_all[int(i)])
        acc[c] += float(sc)
        cnt[c] += 1
    mean = np.zeros(n_free, dtype=np.float32)
    m = cnt > 0
    mean[m] = (acc[m] / cnt[m]).astype(np.float32)

    gh, gw = len(geodesic.grid), len(geodesic.grid[0])
    grid = np.full((gh, gw), np.nan, dtype=np.float32)
    for idx, (r, c) in enumerate(geodesic.idx_to_cell):
        if cnt[idx] > 0:
            grid[int(r), int(c)] = float(mean[idx])

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, aspect="equal", origin="upper", cmap="plasma", vmin=0.0, vmax=1.0)
    plt.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        label="1 − Spearman(d_enc kNN, d_replay graph SP hops)",
    )
    ax.set_title(title)
    ax.set_xlabel("grid col")
    ax.set_ylabel("grid row")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def train_replay_topology_head(
    data: dict,
    cfg_pm: PointMazeRunCfg,
    device: torch.device,
    seed: int,
    geodesic: GeodesicComputer,
    out_dir: str,
):
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    pos = np.asarray(data["pos"], dtype=np.float32)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or encoder_emb.shape[0] != len(pos):
        print("    temporal_dist_head: encoder_emb shape mismatch vs pos.")
        return None, None, None

    rng = np.random.default_rng(int(seed))
    pool_n = int(cfg_pm.replay_topology_pair_pool)
    pool = _oracle_free_replay_step_distances(
        ep_ids,
        n_pairs_target=pool_n,
        max_sources=96,
        rng=rng,
    )
    if pool is None:
        print("    temporal_dist_head: could not build replay step-distance pairs.")
        return None, None, None

    ii, jj, dd = pool
    n_pairs = len(ii)
    if n_pairs < 16:
        print("    temporal_dist_head: too few pairs.")
        return None, None, None

    perm = rng.permutation(n_pairs)
    n_val = min(max(1, int(round(float(cfg_pm.replay_topology_val_frac) * n_pairs))), n_pairs - 1)
    train_mask = np.ones(n_pairs, dtype=bool)
    train_mask[perm[:n_val]] = False

    train_ii, train_jj, train_dd = ii[train_mask], jj[train_mask], dd[train_mask]
    val_ii, val_jj, val_dd = ii[~train_mask], jj[~train_mask], dd[~train_mask]

    embed_dim = int(encoder_emb.shape[1])
    n_ens = max(1, int(cfg_pm.replay_topology_n_ensemble))
    topo_dim = int(cfg_pm.replay_topology_dim)
    # Enforce low-dimensional Euclidean bottleneck to avoid high-D folding.
    if topo_dim not in (2, 3):
        topo_dim = 2
        cfg_pm.replay_topology_dim = 2
    model = TemporalTopoHead(
        embed_dim=embed_dim,
        topo_dim=topo_dim,
        hidden_dim=int(cfg_pm.replay_topology_hidden),
        n_ensemble=n_ens,
    ).to(device)

    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    n_train = len(train_ii)
    batch_pairs = int(min(cfg_pm.replay_topology_batch_pairs, max(1, n_train)))
    total_epochs = max(2, int(cfg_pm.replay_topology_epochs))
    best_val = float("inf")
    best_state = None

    print(
        f"    temporal_dist: {total_epochs} epochs (pure step-count MSE) "
        f"with g_topo(e) bottleneck dim={topo_dim}, ensemble K={n_ens}"
    )

    def _val_mse_ensemble(ve_i, ve_j, vt):
        pv = model.forward_pair_ensemble(ve_i, ve_j)
        return F.mse_loss(pv, vt.unsqueeze(-1).expand_as(pv))

    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    for epoch in range(total_epochs):
        model.train()
        idx = rng.choice(n_train, size=batch_pairs, replace=n_train < batch_pairs)
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
            val_loss = _val_mse_ensemble(ve_i, ve_j, vt)
            if float(val_loss.item()) < best_val:
                best_val = float(val_loss.item())
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, total_epochs // 6) == 0:
            print(
                f"    temporal_dist epoch {epoch + 1}/{total_epochs} (phase=step-mse)  "
                f"train_mse={float(loss.item()):.4f}  val_mse={float(val_loss.item()):.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        z_all = model.encode(e_t).cpu().numpy().astype(np.float32)

    pair_var_all = _predict_pair_ensemble_variance_batch(model, e_t, ii, jj, device)
    mean_var_all = float(np.mean(pair_var_all))
    mean_var_val = float(
        np.mean(_predict_pair_ensemble_variance_batch(model, e_t, val_ii, val_jj, device))
    )

    plot_path = os.path.join(out_dir, "temporal_dist_error_map.png")
    plot_temporal_dist_error_map(
        geodesic,
        pos,
        ii,
        jj,
        pair_var_all,
        plot_path,
        title=f"Temporal distance ensemble (K={n_ens}): mean Var(pred) per maze cell (pair endpoints)",
    )
    scatter_path = os.path.join(out_dir, "temporal_dist_error_scatter.png")
    plot_temporal_dist_error_scatter(
        pos,
        ii,
        pair_var_all,
        scatter_path,
        rng,
        title=f"Temporal distance ensemble (K={n_ens}): Var(pred) at pair source (x, y)",
    )

    # ------------------------------------------------------------------
    # Encoder-near / temporal-far conflict heatmap (closer to "wormhole" concept)
    # ------------------------------------------------------------------
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    src_idx, src_score = _compute_encoder_temporal_barrier_scores(
        encoder_emb=encoder_emb,
        episode_ids=ep_ids,
        pairs_ii=ii,
        pairs_jj=jj,
        pairs_steps=dd,
        rng=rng,
        k_nn=12,
        barrier_steps=25.0,
        candidate_pool=1800,
        max_sources=1500,
        min_known=2,
    )
    conflict_map_path = os.path.join(out_dir, "encoder_temporal_barrier_heatmap.png")
    plot_encoder_temporal_barrier_heatmap(
        geodesic,
        pos,
        src_idx,
        src_score,
        conflict_map_path,
        title="Encoder-kNN but temporally far (source-only)",
    )
    conflict_bridge_sanity = _bridge_sanity_for_conflict_scores(
        geodesic,
        pos,
        src_idx,
        src_score,
        top_frac=0.15,
    )

    # ------------------------------------------------------------------
    # Encoder-local vs replay-temporal geometry (exploration: disagree most)
    # ------------------------------------------------------------------
    src_lg, disc_lg = _compute_encoder_temporal_local_global_disagreement(
        encoder_emb=encoder_emb,
        episode_ids=ep_ids,
        rng=rng,
        k_nn=12,
        candidate_pool=1800,
        max_sources=2000,
        min_labeled_neighbors=4,
    )
    local_global_path = os.path.join(out_dir, "encoder_temporal_local_global_disagreement.png")
    plot_encoder_temporal_local_global_disagreement_heatmap(
        geodesic,
        pos,
        src_lg,
        disc_lg,
        local_global_path,
        title="Exploration: encoder kNN vs replay-graph shortest-path hops (high = disagree)",
    )
    local_global_bridge_sanity = _bridge_sanity_for_conflict_scores(
        geodesic,
        pos,
        src_lg,
        disc_lg,
        top_frac=0.15,
    )
    mean_lg_disagree = float(np.mean(disc_lg)) if len(disc_lg) else 0.0

    meta = {
        "n_pairs": int(n_pairs),
        "n_pairs_train": int(len(train_ii)),
        "n_pairs_val": int(len(val_ii)),
        "n_ensemble": int(n_ens),
        "total_epochs": int(total_epochs),
        "topo_dim": int(topo_dim),
        "best_val_rank_loss": None,
        "mean_pair_pred_variance_all_pairs": mean_var_all,
        "val_mean_pair_pred_variance": mean_var_val,
        "best_val_mse": float(best_val),
        "input": "encoder_e_only",
        "embed_dim": embed_dim,
        "feat_dict_key": "g_topo(e)",
        "plot_metric": "ensemble_pred_variance",
        "error_map_path": plot_path,
        "error_scatter_path": scatter_path,
        "encoder_temporal_barrier_heatmap_path": conflict_map_path,
        "encoder_temporal_barrier_bridge_sanity": conflict_bridge_sanity,
        "encoder_temporal_local_global_disagreement_path": local_global_path,
        "mean_local_global_disagreement": mean_lg_disagree,
        "encoder_temporal_local_global_bridge_sanity": local_global_bridge_sanity,
    }
    return model, z_all, meta


def train_replay_contrastive_head(
    data: dict,
    cfg_pm: PointMazeRunCfg,
    device: torch.device,
    seed: int,
):
    ep_ids = np.asarray(data["episode_ids"], dtype=np.int64)
    encoder_emb = np.asarray(data.get("encoder_emb", None), dtype=np.float32)
    if encoder_emb.ndim != 2 or encoder_emb.shape[0] != len(ep_ids):
        print("    replay_contrastive_head: encoder_emb shape mismatch vs episode_ids.")
        return None, None, None

    rng = np.random.default_rng(int(seed))
    n = int(len(ep_ids))
    if n < 16:
        print("    replay_contrastive_head: too few replay samples.")
        return None, None, None

    pos_table = _build_temporal_positive_table(ep_ids, pos_k=int(cfg_pm.replay_cont_pos_k))
    valid_idx = np.asarray([i for i, arr in enumerate(pos_table) if len(arr) > 0], dtype=np.int64)
    if len(valid_idx) < 8:
        print("    replay_contrastive_head: too few anchors with temporal positives.")
        return None, None, None

    perm = rng.permutation(len(valid_idx))
    n_val = min(
        max(1, int(round(float(cfg_pm.replay_cont_val_frac) * len(valid_idx)))),
        len(valid_idx) - 1,
    )
    train_idx = valid_idx[perm[n_val:]]
    val_idx = valid_idx[perm[:n_val]]

    e_t = torch.tensor(encoder_emb, dtype=torch.float32, device=device)
    embed_dim = int(encoder_emb.shape[1])
    model = ReplayContrastiveHead(
        embed_dim=embed_dim,
        cont_dim=int(cfg_pm.replay_cont_dim),
        hidden_dim=int(cfg_pm.replay_cont_hidden),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)

    batch = int(min(max(16, cfg_pm.replay_cont_batch), max(16, len(train_idx))))
    n_neg = int(max(4, cfg_pm.replay_cont_neg_k))
    temp = float(max(1e-3, cfg_pm.replay_cont_temp))
    epochs = int(max(2, cfg_pm.replay_cont_epochs))

    def _sample_batch(anchor_pool: np.ndarray, b: int):
        a = rng.choice(anchor_pool, size=b, replace=len(anchor_pool) < b).astype(np.int64)
        p = np.empty(b, dtype=np.int64)
        for t, ai in enumerate(a.tolist()):
            cands = pos_table[int(ai)]
            p[t] = int(cands[int(rng.integers(0, len(cands)))])
        neg = rng.integers(0, n, size=(b, n_neg), dtype=np.int64)
        # Avoid exact anchor/positive collisions in negatives.
        for t in range(b):
            same = (neg[t] == a[t]) | (neg[t] == p[t])
            if np.any(same):
                repl = rng.integers(0, n, size=int(np.sum(same)), dtype=np.int64)
                neg[t, same] = repl
        return (
            torch.tensor(a, dtype=torch.long, device=device),
            torch.tensor(p, dtype=torch.long, device=device),
            torch.tensor(neg, dtype=torch.long, device=device),
        )

    def _info_nce(ai_t: torch.Tensor, pi_t: torch.Tensor, ni_t: torch.Tensor):
        za = model.encode_unit(e_t[ai_t])  # [B, D]
        zp = model.encode_unit(e_t[pi_t])  # [B, D]
        zn = model.encode_unit(e_t[ni_t.reshape(-1)]).reshape(ai_t.shape[0], n_neg, -1)  # [B,K,D]
        pos_logit = torch.sum(za * zp, dim=-1, keepdim=True) / temp  # [B,1]
        neg_logit = torch.sum(za[:, None, :] * zn, dim=-1) / temp  # [B,K]
        logits = torch.cat([pos_logit, neg_logit], dim=1)  # [B,1+K]
        loss = -pos_logit.squeeze(1) + torch.logsumexp(logits, dim=1)
        return loss.mean(), float(pos_logit.mean().item()), float(neg_logit.mean().item())

    @torch.no_grad()
    def _val_loss():
        model.eval()
        ai_t, pi_t, ni_t = _sample_batch(val_idx, b=min(batch, max(8, len(val_idx))))
        l, pl, nl = _info_nce(ai_t, pi_t, ni_t)
        return float(l.item()), pl, nl

    best_val = float("inf")
    best_state = None
    for epoch in range(epochs):
        model.train()
        ai_t, pi_t, ni_t = _sample_batch(train_idx, b=batch)
        loss, pos_m, neg_m = _info_nce(ai_t, pi_t, ni_t)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        val_loss, val_pos_m, val_neg_m = _val_loss()
        if val_loss < best_val:
            best_val = float(val_loss)
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        if (epoch + 1) % max(1, epochs // 6) == 0:
            print(
                f"    replay_cont epoch {epoch + 1}/{epochs}  "
                f"train_infonce={float(loss.item()):.4f}  val_infonce={val_loss:.4f}  "
                f"train_pos_logit={pos_m:.3f} train_neg_logit={neg_m:.3f}  "
                f"val_pos_logit={val_pos_m:.3f} val_neg_logit={val_neg_m:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        z_all = model.encode(e_t).cpu().numpy().astype(np.float32)

    meta = {
        "n_samples": int(n),
        "n_anchor_train": int(len(train_idx)),
        "n_anchor_val": int(len(val_idx)),
        "epochs": int(epochs),
        "batch": int(batch),
        "n_neg": int(n_neg),
        "pos_k": int(cfg_pm.replay_cont_pos_k),
        "temperature": float(temp),
        "best_val_infonce": float(best_val),
        "input": "encoder_e_only",
        "embed_dim": int(embed_dim),
        "feat_dict_key": "g_cont(e)",
        "objective": "InfoNCE_temporal_pos_random_neg",
    }
    return model, z_all, meta


def run_replay_topology_eval(topo_meta: dict):
    if topo_meta is None:
        return None
    return {
        "n_pairs": int(topo_meta.get("n_pairs", 0)),
        "n_ensemble": int(topo_meta.get("n_ensemble", 0)),
        "best_val_rank_loss": topo_meta.get("best_val_rank_loss", None),
        "mean_pair_pred_variance_all_pairs": float(
            topo_meta.get("mean_pair_pred_variance_all_pairs", 0.0)
        ),
        "val_mean_pair_pred_variance": float(topo_meta.get("val_mean_pair_pred_variance", 0.0)),
        "best_val_mse": float(topo_meta.get("best_val_mse", 0.0)),
        "input": str(topo_meta.get("input", "")),
        "embed_dim": int(topo_meta.get("embed_dim", 0)),
        "feat_dict_key": str(topo_meta.get("feat_dict_key", "")),
        "plot_metric": str(topo_meta.get("plot_metric", "")),
        "error_map_path": str(topo_meta.get("error_map_path", "")),
        "error_scatter_path": str(topo_meta.get("error_scatter_path", "")),
        "encoder_temporal_barrier_heatmap_path": str(
            topo_meta.get("encoder_temporal_barrier_heatmap_path", "")
        ),
        "encoder_temporal_barrier_bridge_sanity": topo_meta.get(
            "encoder_temporal_barrier_bridge_sanity", None
        ),
        "encoder_temporal_local_global_disagreement_path": str(
            topo_meta.get("encoder_temporal_local_global_disagreement_path", "")
        ),
        "mean_local_global_disagreement": float(
            topo_meta.get("mean_local_global_disagreement", 0.0)
        ),
        "encoder_temporal_local_global_bridge_sanity": topo_meta.get(
            "encoder_temporal_local_global_bridge_sanity", None
        ),
    }


def run_replay_contrastive_eval(cont_meta: dict):
    if cont_meta is None:
        return None
    return {
        "n_samples": int(cont_meta.get("n_samples", 0)),
        "n_anchor_train": int(cont_meta.get("n_anchor_train", 0)),
        "n_anchor_val": int(cont_meta.get("n_anchor_val", 0)),
        "epochs": int(cont_meta.get("epochs", 0)),
        "batch": int(cont_meta.get("batch", 0)),
        "n_neg": int(cont_meta.get("n_neg", 0)),
        "pos_k": int(cont_meta.get("pos_k", 0)),
        "temperature": float(cont_meta.get("temperature", 0.0)),
        "best_val_infonce": float(cont_meta.get("best_val_infonce", 0.0)),
        "input": str(cont_meta.get("input", "")),
        "embed_dim": int(cont_meta.get("embed_dim", 0)),
        "feat_dict_key": str(cont_meta.get("feat_dict_key", "")),
        "objective": str(cont_meta.get("objective", "")),
    }


def run_replay_laplacian_eval(lap_meta: dict):
    if lap_meta is None:
        return None
    return {
        "n_nodes": int(lap_meta.get("n_nodes", 0)),
        "dim": int(lap_meta.get("dim", 0)),
        "knn_k": int(lap_meta.get("knn_k", 0)),
        "graph_max": int(lap_meta.get("graph_max", 0)),
        "eigvals_used": lap_meta.get("eigvals_used", []),
        "feat_dict_key": str(lap_meta.get("feat_dict_key", "")),
        "objective": str(lap_meta.get("objective", "")),
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