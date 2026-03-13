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
from typing import Tuple

import cv2
import gymnasium as gym
import gymnasium_robotics  # type: ignore
import numpy as np
import torch

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
)
from utils import get_device, set_seed
from models import (
    RSSM,
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

    def _update_agent_pos(self, obs_dict):
        # obs_dict["achieved_goal"] is shape (2,) with (x,y) in MuJoCo coords.
        ag = obs_dict.get("achieved_goal", None)
        if ag is None:
            # Fallback: use first two entries of "observation" (x,y).
            obs = obs_dict.get("observation", None)
            if obs is not None and len(obs) >= 2:
                ag = obs[:2]
        if ag is not None:
            self._agent_pos = np.asarray(ag[:2], dtype=np.float32)

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

    def step(self, action, repeat: int = 10):
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
        models = {
            "encoder": ConvEncoder(cfg.embed_dim).to(device).load_state_dict(checkpoint["encoder"]),
            "decoder": ConvDecoder(cfg.deter_dim, cfg.stoch_dim, embedding_size=cfg.embed_dim).to(device).load_state_dict(checkpoint["decoder"]),
            "rssm": RSSM(cfg.stoch_dim, cfg.deter_dim, act_dim, cfg.embed_dim, cfg.hidden_dim).to(device).load_state_dict(checkpoint["rssm"]),
            "reward_model": RewardModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device).load_state_dict(checkpoint["reward_model"]),
            "cont_model": ContinueModel(cfg.deter_dim, cfg.stoch_dim, cfg.hidden_dim).to(device).load_state_dict(checkpoint["cont_model"]),
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

    print("\n  [4/5] Running analyses ...")
    pos = data["pos"]
    episode_ids = data.get("episode_ids", None)
    feat_dict = _build_feature_dict(
        data, device, geo_temporal=geo_temporal, geo_geodesic=geo_geo
    )

    probe_res = run_probes(pos, feat_dict, cfg, device, episode_ids=episode_ids)
    dist_res, dist_raw = run_distance_analysis(pos, feat_dict, env.geodesic, cfg)
    knn_res = run_knn_analysis(pos, feat_dict, env.geodesic, cfg)
    tc_res = run_trustworthiness_continuity(pos, feat_dict, env.geodesic, cfg, k=cfg.knn_k)

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
    return p.parse_args()


def main():
    args = parse_args()
    cfg_pm = PointMazeRunCfg(
        seed=args.seed,
        output_dir=args.output_dir,
        quick=bool(args.quick),
        geo_supervised=bool(args.geo_supervised),
    )
    run_single_pointmaze(cfg_pm)


if __name__ == "__main__":
    main()

