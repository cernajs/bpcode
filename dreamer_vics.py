#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from maze_geometry_test import _positions_to_cell_indices
from models import RSSM, Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel, ValueModel
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
    """VICReg loss on encoder embeddings.

    z:     [B, D] — anchor embeddings (e.g. e_t)
    z_pos: [B, D] — positive embeddings (e.g. e_{t+1} from same episode)

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
# Geometric diagnostics (logged to TensorBoard)
# =====================================================================


@torch.no_grad()
def compute_encoder_diagnostics(
    encoder: ConvEncoder,
    obs_tensor: torch.Tensor,
    device: torch.device,
    max_samples: int = 2048,
) -> dict[str, float]:
    """Compute latent space health metrics from a batch of observations."""
    encoder.eval()
    N = obs_tensor.shape[0]
    idx = np.random.choice(N, size=min(max_samples, N), replace=False)
    e = encoder(obs_tensor[idx])

    # Pairwise L2 distances
    dm = torch.cdist(e, e)
    triu_mask = torch.triu(torch.ones_like(dm, dtype=torch.bool), diagonal=1)
    dists = dm[triu_mask]

    # Per-dimension statistics
    per_dim_std = e.std(dim=0)
    per_dim_mean = e.mean(dim=0)

    # Covariance off-diagonal magnitude
    e_c = e - e.mean(dim=0)
    cov = (e_c.T @ e_c) / max(e.shape[0] - 1, 1)
    diag = cov.diagonal()
    off_diag_sq = cov.pow(2).sum() - diag.pow(2).sum()
    off_diag_norm = (off_diag_sq / max(cov.shape[0] ** 2 - cov.shape[0], 1)).sqrt()

    # Effective rank approximation via singular values
    try:
        s = torch.linalg.svdvals(e_c)
        s_norm = s / s.sum().clamp(min=1e-8)
        entropy = -(s_norm * torch.log(s_norm + 1e-10)).sum()
        eff_rank = torch.exp(entropy)
    except Exception:
        eff_rank = torch.tensor(0.0)

    return {
        "encoder/mean_pairwise_dist": float(dists.mean().item()),
        "encoder/std_pairwise_dist": float(dists.std().item()),
        "encoder/min_pairwise_dist": float(dists.min().item()),
        "encoder/median_pairwise_dist": float(dists.median().item()),
        "encoder/mean_per_dim_std": float(per_dim_std.mean().item()),
        "encoder/min_per_dim_std": float(per_dim_std.min().item()),
        "encoder/max_per_dim_std": float(per_dim_std.max().item()),
        "encoder/num_dead_dims": float((per_dim_std < 0.01).sum().item()),
        "encoder/off_diag_cov": float(off_diag_norm.item()),
        "encoder/effective_rank": float(eff_rank.item()),
        "encoder/mean_norm": float(e.norm(dim=-1).mean().item()),
    }


@torch.no_grad()
def compute_geodesic_correlation(
    encoder: ConvEncoder,
    obs_buffer,
    pos_buffer: np.ndarray,
    geodesic,
    device: torch.device,
    bit_depth: int,
    n_pairs: int = 2000,
) -> dict[str, float]:
    """Spearman correlation: encoder L2 distance vs geodesic distance."""
    from scipy.stats import spearmanr

    encoder.eval()
    N = len(pos_buffer)
    if N < 20:
        return {"geo/spearman_enc": 0.0, "geo/n_pairs": 0}

    rng = np.random.default_rng(42)
    n_pairs = min(n_pairs, N * (N - 1) // 2)
    ii = rng.integers(0, N, size=n_pairs)
    jj = rng.integers(0, N, size=n_pairs)
    valid = ii != jj
    ii, jj = ii[valid], jj[valid]

    # Encode observations
    def encode_batch(indices):
        embs = []
        for start in range(0, len(indices), 256):
            end = min(start + 256, len(indices))
            batch_idx = indices[start:end]
            obs_batch = torch.tensor(
                obs_buffer[batch_idx], dtype=torch.float32, device=device
            ).permute(0, 3, 1, 2)
            preprocess_img(obs_batch, depth=bit_depth)
            embs.append(encoder(obs_batch).cpu().numpy())
        return np.concatenate(embs, axis=0)

    e_i = encode_batch(ii)
    e_j = encode_batch(jj)
    d_enc = np.linalg.norm(e_i - e_j, axis=-1)

    cell_i = _positions_to_cell_indices(geodesic, pos_buffer[ii])
    cell_j = _positions_to_cell_indices(geodesic, pos_buffer[jj])
    d_geo = np.array([
        float(geodesic.dist_matrix[int(ci), int(cj)])
        for ci, cj in zip(cell_i, cell_j)
    ], dtype=np.float32)

    finite = np.isfinite(d_geo) & (d_geo > 0) & np.isfinite(d_enc) & (d_enc > 0)
    if finite.sum() < 10:
        return {"geo/spearman_enc": 0.0, "geo/n_pairs": 0}

    rho = spearmanr(d_enc[finite], d_geo[finite]).correlation
    if rho is None or not np.isfinite(rho):
        rho = 0.0

    return {
        "geo/spearman_enc": float(rho),
        "geo/n_pairs": int(finite.sum()),
        "geo/mean_enc_dist": float(np.mean(d_enc[finite])),
        "geo/mean_geo_dist": float(np.mean(d_geo[finite])),
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
                   choices=["baseline", "vicreg", "straight", "georeg", "georeg_cov"],
                   help="baseline | vicreg | straight | georeg (both) | georeg_cov (both + coverage)")
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

    # Diagnostics
    p.add_argument("--diag_interval", type=int, default=40,
                   help="Run encoder diagnostics every N training episodes")
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

    print(f"Device: {device}")
    print(f"Geo mode: {args.geo_mode}")
    print(f"  VICReg: {use_vicreg} (weight={args.vicreg_weight})")
    print(f"  Straightness: {use_straight} (weight={args.straight_weight})")
    print(f"  Coverage bonus: {use_coverage} (weight={args.coverage_weight})")

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

    # Coverage memory bank (for georeg_cov mode)
    coverage_memory: torch.Tensor | None = None
    coverage_ptr = 0

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
                encoder.train(); decoder.train(); rssm.train()
                reward_model.train(); cont_model.train()
                actor.train(); value_model.train()

                sum_rec = sum_kld = sum_rew = sum_cont = sum_model = 0.0
                sum_actor = sum_value = sum_imag_r = 0.0
                sum_vicreg = sum_straight = sum_coverage = 0.0
                vicreg_info_accum: dict[str, float] = {}
                straight_info_accum: dict[str, float] = {}

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

                    # ---- VICReg loss ----
                    l_vicreg = torch.tensor(0.0, device=device)
                    if use_vicreg:
                        # Temporal pairs: e_t and e_{t+1}
                        e_anchor = e_t[:, :-1].reshape(-1, e_t.shape[-1])
                        e_positive = e_t[:, 1:].reshape(-1, e_t.shape[-1])
                        # Subsample to keep cost bounded
                        n_vic = min(512, e_anchor.shape[0])
                        vic_idx = torch.randperm(e_anchor.shape[0], device=device)[:n_vic]
                        l_vicreg, vic_info = vicreg_loss(
                            e_anchor[vic_idx], e_positive[vic_idx],
                            var_weight=args.vicreg_var_w,
                            inv_weight=args.vicreg_inv_w,
                            cov_weight=args.vicreg_cov_w,
                            target_std=args.vicreg_target_std,
                        )
                        model_loss = model_loss + args.vicreg_weight * l_vicreg
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

                    # ---- Coverage intrinsic reward ----
                    if use_coverage:
                        with torch.no_grad():
                            # Encode imagined states for coverage bonus
                            e_imag_flat = torch.cat([h_imag[:, 1:].reshape(-1, Dh),
                                                      s_imag[:, 1:].reshape(-1, Ds)], dim=-1)
                            if coverage_memory is None:
                                coverage_memory = e_imag_flat[:args.coverage_memory_size].clone()
                                coverage_ptr = min(len(e_imag_flat), args.coverage_memory_size)
                            cov_bonus = latent_coverage_bonus(
                                e_imag_flat, coverage_memory, k=args.coverage_knn_k
                            ).reshape(rewards_imag.shape)
                            # Normalize to comparable scale with extrinsic reward
                            cov_bonus = cov_bonus / (cov_bonus.std().clamp(min=1e-4))
                            # Update memory with recent states
                            n_new = min(len(e_imag_flat), args.coverage_memory_size)
                            new_states = e_imag_flat[:n_new]
                            for i in range(n_new):
                                coverage_memory[coverage_ptr % args.coverage_memory_size] = new_states[i]
                                coverage_ptr += 1
                        rewards_total = rewards_imag + args.coverage_weight * cov_bonus
                        sum_coverage += float(cov_bonus.mean().item())
                    else:
                        rewards_total = rewards_imag

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

        # ---- Encoder diagnostics ----
        if args.diag_interval > 0 and (episode + 1) % args.diag_interval == 0:
            # Grab a batch of recent observations for diagnostics
            if replay.size > 256:
                diag_batch = replay.sample_sequences(min(64, args.batch_size), 1)
                diag_obs = torch.tensor(diag_batch.obs[:, 0], dtype=torch.float32, device=device).permute(0, 3, 1, 2)
                preprocess_img(diag_obs, depth=args.bit_depth)
                diags = compute_encoder_diagnostics(encoder, diag_obs, device)
                for k, v in diags.items():
                    writer.add_scalar(k, v, total_steps)
                print(f"    [diag] mean_d={diags['encoder/mean_pairwise_dist']:.4f}  "
                      f"eff_rank={diags['encoder/effective_rank']:.1f}  "
                      f"mean_std={diags['encoder/mean_per_dim_std']:.4f}  "
                      f"dead_dims={diags['encoder/num_dead_dims']:.0f}")

        # ---- Geodesic correlation ----
        if args.geo_corr_interval > 0 and (episode + 1) % args.geo_corr_interval == 0:
            if len(obs_geo_buffer) >= 100:
                obs_arr = np.stack(obs_geo_buffer, axis=0)
                pos_arr = np.stack(pos_geo_buffer, axis=0)
                geo_corr = compute_geodesic_correlation(
                    encoder, obs_arr, pos_arr, geodesic, device, args.bit_depth, n_pairs=2000)
                for k, v in geo_corr.items():
                    writer.add_scalar(k, v, total_steps)
                print(f"    [geo] ρ_enc={geo_corr['geo/spearman_enc']:.4f}  "
                      f"mean_d_enc={geo_corr.get('geo/mean_enc_dist', 0):.4f}")

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