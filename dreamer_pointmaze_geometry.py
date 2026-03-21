import argparse
import os
from typing import List

import numpy as np
import torch
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from models import RSSM, Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel, ValueModel
from geom_head import GeoEncoder
from utils import ReplayBuffer, bottle, get_device, preprocess_img, set_seed
from pointmaze_gr_geometry_test import PointMazeMediumDiverseGRWrapper
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path


# =============================================================================
#  Geometry-aware Dreamer for PointMaze
# =============================================================================
#
# This variant implements the "geometry-mismatch frontier" idea:
#   - Latents know local state (good decoding), but Euclidean distance does not
#     match maze geodesics.
#   - Replay graph shortest paths are a good, oracle-free geometry teacher.
#   - Imagination drifts away from replay geometry, so geometric exploration
#     should be driven by replay geometry but gated by on-manifold proximity.
#
# Concretely, we:
#   1) Train a geometry head g_replay(h,s) on replay so ||g(v)-g(u)|| matches
#      replay-graph shortest-path distance between timesteps v,u.
#   2) Maintain per-node distortion scores δ(v): mismatch between latent metric
#      and replay-graph distances to a small landmark set.
#   3) Define a potential φ(v) that combines distortion, novelty (1/sqrt(N+1)),
#      and optionally a simple bottleneck proxy.
#   4) During imagination, map imagined states z_t to nearest replay nodes v_t
#      in g-space and define intrinsic reward
#          r_geo_t = φ(v_{t+1}) - φ(v_t) - λ_off * m(z_{t+1}),
#      where m penalizes off-replay-manifold distance in g-space.
#
# This file is intentionally self-contained and PointMaze-specific.
#


def _pairwise_l2(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    aa = (a * a).sum(axis=1, keepdims=True)
    d2 = np.maximum(aa + aa.T - 2.0 * (a @ a.T), 0.0)
    return np.sqrt(d2, dtype=np.float32)


class ReplayGeometryModule:
    """Maintain replay graph geometry + distortion / frontier scores."""

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        geo_dim: int = 32,
        geo_hidden: int = 256,
        device: torch.device | None = None,
        lambda_dist: float = 1.0,
        lambda_front: float = 0.5,
        lambda_off: float = 1.0,
        n_landmarks: int = 16,
        max_nodes: int = 2500,
    ):
        self.device = device or torch.device("cpu")
        self.geo_head = GeoEncoder(deter_dim, stoch_dim, geo_dim=geo_dim, hidden_dim=geo_hidden).to(
            self.device
        )
        self.opt = torch.optim.Adam(self.geo_head.parameters(), lr=3e-4)

        self.lambda_dist = float(lambda_dist)
        self.lambda_front = float(lambda_front)
        self.lambda_off = float(lambda_off)
        self.n_landmarks = int(n_landmarks)
        self.max_nodes = int(max_nodes)

        self._h_nodes: List[np.ndarray] = []
        self._s_nodes: List[np.ndarray] = []
        self._e_nodes: List[np.ndarray] = []
        self._episode_ids: List[int] = []
        self._step_indices: List[float] = []
        self._current_step: int = 0

        self._g_nodes: np.ndarray | None = None
        self._dist_replay: np.ndarray | None = None
        self._phi: np.ndarray | None = None
        self._visit_counts: np.ndarray | None = None

        # TensorBoard diagnostics (updated each train phase when geometry is on)
        self.tb_last_graph_n_nodes: int = 0
        self.tb_last_dist_finite_frac: float | None = None
        self.tb_last_geo_train_loss: float | None = None
        self.tb_last_geo_train_steps: int = 0
        self.tb_last_delta_mean: float | None = None
        self.tb_last_phi_mean: float | None = None
        self.tb_last_phi_std: float | None = None

    def geo_intrinsic_ready(self) -> bool:
        return self._g_nodes is not None and self._phi is not None

    def add_batch_trajectories(
        self,
        traj_h: List[np.ndarray],
        traj_s: List[np.ndarray],
        traj_e: List[np.ndarray],
        traj_ep_ids: List[int],
    ):
        """Append trajectory steps; FIFO-evict oldest rows when over max_nodes."""
        for ep_h, ep_s, ep_e, ep_id in zip(traj_h, traj_s, traj_e, traj_ep_ids):
            T = len(ep_h)
            if T < 2:
                continue
            for t in range(T):
                if len(self._h_nodes) >= self.max_nodes:
                    self._h_nodes.pop(0)
                    self._s_nodes.pop(0)
                    self._e_nodes.pop(0)
                    self._episode_ids.pop(0)
                    self._step_indices.pop(0)
                self._h_nodes.append(ep_h[t])
                self._s_nodes.append(ep_s[t])
                self._e_nodes.append(ep_e[t])
                self._episode_ids.append(ep_id)
                self._step_indices.append(float(self._current_step))
                self._current_step += 1

    def rebuild_graph_and_distances(self, knn_k: int = 4, cross_quantile: float = 0.05):
        """Cycle-consistent mutual-kNN + weighted temporal edges; cache all-pairs shortest paths."""
        n = len(self._h_nodes)
        if n < 4:
            self._dist_replay = None
            self.tb_last_graph_n_nodes = n
            self.tb_last_dist_finite_frac = None
            return

        local_feat = np.asarray(self._e_nodes, dtype=np.float32)
        ep_ids = np.asarray(self._episode_ids, dtype=np.int64)
        orig_idx = np.asarray(self._step_indices, dtype=np.float64)

        adj = lil_matrix((n, n), dtype=np.float32)

        for i in range(n - 1):
            if ep_ids[i] == ep_ids[i + 1]:
                dt = float(orig_idx[i + 1] - orig_idx[i])
                adj[i, i + 1] = dt
                adj[i + 1, i] = dt

        if n > 2:
            d = _pairwise_l2(local_feat)
            np.fill_diagonal(d, np.inf)

            finite_vals = d[np.isfinite(d)]
            gate = (
                float(np.quantile(finite_vals, float(cross_quantile)))
                if len(finite_vals)
                else float("inf")
            )
            k_use = int(min(max(1, int(knn_k)), n - 1))
            nn_idx = np.argsort(d, axis=1)[:, :k_use]
            nn_sets = [set(map(int, row)) for row in nn_idx]

            for i in range(n):
                for j in nn_idx[i]:
                    j = int(j)
                    if float(d[i, j]) > gate:
                        continue
                    if i not in nn_sets[j]:
                        continue
                    if (
                        i < n - 1
                        and j < n - 1
                        and ep_ids[i] == ep_ids[i + 1]
                        and ep_ids[j] == ep_ids[j + 1]
                    ):
                        d_next = float(d[i + 1, j + 1])
                        d_prev = (
                            float(d[i + 1, j - 1])
                            if j > 0 and ep_ids[j] == ep_ids[j - 1]
                            else float("inf")
                        )
                        if d_next <= gate or d_prev <= gate:
                            adj[i, j] = 1.0
                            adj[j, i] = 1.0

        self._dist_replay = shortest_path(csgraph=adj, directed=False, unweighted=False)
        self.tb_last_graph_n_nodes = n
        self.tb_last_dist_finite_frac = float(np.isfinite(self._dist_replay).mean())

    def train_geo_head(self, n_steps: int = 200, batch_pairs: int = 2048):
        """Train g(h,s) against cached Dijkstra distances from rebuild_graph_and_distances."""
        self.tb_last_geo_train_loss = None
        self.tb_last_geo_train_steps = 0
        if self._dist_replay is None:
            return
        h_np = np.asarray(self._h_nodes, dtype=np.float32)
        s_np = np.asarray(self._s_nodes, dtype=np.float32)
        N = h_np.shape[0]
        if N < 4:
            return

        h_t = torch.tensor(h_np, dtype=torch.float32, device=self.device)
        s_t = torch.tensor(s_np, dtype=torch.float32, device=self.device)
        dist_mat = self._dist_replay

        loss_accum: List[float] = []
        for _ in range(n_steps):
            ii = np.random.randint(0, N, size=batch_pairs)
            jj = np.random.randint(0, N, size=batch_pairs)
            same = ii == jj
            jj[same] = (jj[same] + 1) % max(N, 1)
            d_np = dist_mat[ii, jj]
            valid = np.isfinite(d_np) & (d_np > 0)
            if valid.sum() < 32:
                continue
            ii = ii[valid]
            jj = jj[valid]
            d_np = d_np[valid].astype(np.float32)

            gi = self.geo_head(h_t[ii], s_t[ii])
            gj = self.geo_head(h_t[jj], s_t[jj])
            d_lat = torch.norm(gi - gj, dim=-1)

            d_teacher = torch.tensor(d_np, device=self.device)
            w = 1.0 / torch.sqrt(d_teacher + 1.0)
            scale = (d_teacher.detach().mean().clamp_min(1e-3)).item()
            loss = (w * (d_lat - (d_teacher / scale)) ** 2).mean()

            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            self.opt.step()
            loss_accum.append(float(loss.detach().item()))

        self.tb_last_geo_train_steps = len(loss_accum)
        self.tb_last_geo_train_loss = (
            float(np.mean(loss_accum)) if loss_accum else None
        )

        with torch.no_grad():
            self._g_nodes = self.geo_head(h_t, s_t).detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    #  Distortion / frontier potential
    # ------------------------------------------------------------------ #

    def build_potential(self):
        """Compute per-node potential φ(v) from distortion + frontier."""
        self.tb_last_delta_mean = None
        self.tb_last_phi_mean = None
        self.tb_last_phi_std = None
        if self._g_nodes is None or self._dist_replay is None:
            return
        g = self._g_nodes
        N = g.shape[0]
        if N < 4:
            return

        # Landmarks: spread across graph via simple farthest-point heuristic
        rng = np.random.default_rng(0)
        landmarks = [rng.integers(0, N)]
        while len(landmarks) < min(self.n_landmarks, N):
            # pick node with largest min-distance to existing landmarks in graph metric
            d_min = np.min(self._dist_replay[:, landmarks], axis=1)
            cand = int(np.argmax(d_min))
            if cand in landmarks:
                break
            landmarks.append(cand)
        landmarks = np.asarray(landmarks, dtype=np.int64)

        # Distortion δ(v): average absolute error between ||g(v)-g(ℓ)|| and scaled d_replay(v,ℓ)
        d_geo = np.asarray(self._dist_replay[:, landmarks], dtype=np.float64)  # [N, L]
        with torch.no_grad():
            gv = torch.tensor(g, device=self.device)
            gl = torch.tensor(g[landmarks], device=self.device)
            d_lat = torch.cdist(gv, gl).cpu().numpy()

        # Finite scale only (graph distances can be inf / disconnected; mean can be inf → inf/inf in divide)
        finite_mask = np.isfinite(d_geo) & (d_geo > 0)
        if finite_mask.any():
            scale = max(float(np.mean(d_geo[finite_mask])), 1e-3)
        else:
            scale = 1.0
        d_geo_scaled = np.full_like(d_geo, np.nan, dtype=np.float64)
        np.divide(d_geo, scale, out=d_geo_scaled, where=finite_mask)
        delta = np.nanmean(np.abs(d_lat - d_geo_scaled), axis=1)
        delta = np.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)

        # Frontier term: unvisited nodes should have higher weight
        #if self._visit_counts is None or len(self._visit_counts) != N:
        self._visit_counts = np.zeros(N, dtype=np.int64)
        front = 1.0 / np.sqrt(self._visit_counts.astype(np.float32) + 1.0)

        phi = self.lambda_dist * delta + self.lambda_front * front
        self._phi = phi.astype(np.float32)
        self.tb_last_delta_mean = float(np.mean(delta))
        self.tb_last_phi_mean = float(np.mean(self._phi))
        self.tb_last_phi_std = float(np.std(self._phi))

    # ------------------------------------------------------------------ #
    #  Intrinsic reward for imagination
    # ------------------------------------------------------------------ #

    def intrinsic_reward_for_imagination(
        self,
        h_imag: torch.Tensor,
        s_imag: torch.Tensor,
    ) -> torch.Tensor:
        """Compute r_geo over imagined rollout.

        h_imag, s_imag: [B_imag, H+1, D]
        Returns: r_geo [B_imag, H] to be added to imagined rewards.
        """
        if self._g_nodes is None or self._phi is None:
            return torch.zeros((h_imag.size(0), h_imag.size(1) - 1), device=h_imag.device)

        B, H1, _ = h_imag.shape
        H = H1 - 1

        with torch.no_grad():
            g_im = self.geo_head(
                h_imag.reshape(-1, h_imag.size(-1)),
                s_imag.reshape(-1, s_imag.size(-1)),
            )  # [B*H1, G]
            g_im_np = g_im.cpu().numpy().astype(np.float32)

        g_nodes = self._g_nodes  # [N, G]
        phi = self._phi  # [N]

        # kNN in g-space: nearest replay node for each imagined state
        # (we flatten [B,H1] then reshape back)
        N = g_nodes.shape[0]
        # For moderate N, dense cdist is fine
        dm = np.linalg.norm(g_im_np[:, None, :] - g_nodes[None, :, :], axis=-1)  # [B*H1, N]
        nn_idx = np.argmin(dm, axis=1).astype(np.int64)  # [B*H1]
        nn_dist = dm[np.arange(dm.shape[0]), nn_idx]  # off-manifold distance

        v_idx = nn_idx.reshape(B, H1)  # nearest replay node index per (b,t)
        d_off = nn_dist.reshape(B, H1)

        # Update visit counts lightly (only for on-policy states at t=0)
        if self._visit_counts is not None:
            for b in range(B):
                self._visit_counts[v_idx[b, 0]] += 1

        # r_geo_t = φ(v_{t+1}) - φ(v_t) - λ_off * m(z_{t+1})
        v_t = v_idx[:, :-1]
        v_tp1 = v_idx[:, 1:]
        phi_t = phi[v_t]
        phi_tp1 = phi[v_tp1]
        m_off = d_off[:, 1:]

        margin = 0.5
        m_off_penalty = np.maximum(m_off - margin, 0.0)
        gamma = 0.99 
        geo_scale = 0.02

        r_geo_raw = (gamma * phi_tp1 - phi_t) - self.lambda_off * m_off_penalty
        r_geo = r_geo_raw * geo_scale
        
        return torch.tensor(r_geo, dtype=torch.float32, device=h_imag.device)


# =============================================================================
#  PointMaze Dreamer with Geometry-Mismatch Intrinsic Reward
# =============================================================================


def build_parser():
    p = argparse.ArgumentParser(description="Dreamer for PointMaze with geometry-mismatch exploration")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)

    # Training
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=30)
    p.add_argument("--max_episodes", type=int, default=400)
    p.add_argument("--seed_episodes", type=int, default=5)
    p.add_argument("--collect_interval", type=int, default=50)
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--replay_capacity", type=int, default=100_000)

    # Optimization
    p.add_argument("--model_lr", type=float, default=6e-4)
    p.add_argument("--actor_lr", type=float, default=8e-5)
    p.add_argument("--value_lr", type=float, default=8e-5)
    p.add_argument("--adam_eps", type=float, default=1e-5)
    p.add_argument("--grad_clip", type=float, default=100.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda_", type=float, default=0.95)

    # Model sizes
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--actor_hidden_dim", type=int, default=400)
    p.add_argument("--value_hidden_dim", type=int, default=400)

    # KL / continuation
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--kl_free_nats", type=float, default=3.0)

    # Imagination
    p.add_argument("--imagination_horizon", type=int, default=15)
    p.add_argument("--imagination_starts", type=int, default=8)

    # Exploration
    p.add_argument("--expl_amount", type=float, default=0.3)
    p.add_argument("--expl_decay", type=float, default=0.0)
    p.add_argument("--expl_min", type=float, default=0.0)

    # Geometry intrinsic reward weights
    p.add_argument("--geo_lambda_dist", type=float, default=1.0)
    p.add_argument("--geo_lambda_front", type=float, default=0.5)
    p.add_argument("--geo_lambda_off", type=float, default=1.0)
    p.add_argument("--geo_max_nodes", type=int, default=2500)
    p.add_argument("--geo_knn_k", type=int, default=4)
    p.add_argument("--geo_cross_quantile", type=float, default=0.05)

    p.add_argument(
        "--baseline",
        action="store_true",
        help="Standard Dreamer: no replay-graph geometry, geo head, or intrinsic r_geo (saves compute vs default).",
    )

    # Logging / output
    p.add_argument(
        "--wm_path",
        type=str,
        default="",
        help="Optional path to world_model.pt (encoder/decoder/rssm/reward/cont); actor/critic still random init",
    )
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="pointmaze_geo")
    p.add_argument("--eval_episodes", type=int, default=5)
    p.add_argument("--eval_interval", type=int, default=20)

    return p


def log_training_phase_tensorboard(
    writer: SummaryWriter,
    global_step: int,
    *,
    replay_size: int,
    expl_amount: float,
    baseline: bool,
    geo_mod: ReplayGeometryModule | None,
    wm: dict[str, float],
    grad: dict[str, float],
    imag: dict[str, float],
    policy: dict[str, float],
) -> None:
    """One summary point per env interaction step when a training phase runs."""
    writer.add_scalar("replay/size", replay_size, global_step)
    writer.add_scalar("train/exploration_noise", expl_amount, global_step)

    for k, v in wm.items():
        writer.add_scalar(f"wm/{k}", v, global_step)
    for k, v in grad.items():
        writer.add_scalar(f"grad/{k}", v, global_step)
    for k, v in imag.items():
        writer.add_scalar(f"imag/{k}", v, global_step)
    for k, v in policy.items():
        writer.add_scalar(f"policy/{k}", v, global_step)

    if baseline:
        writer.add_scalar("geo/disabled", 1.0, global_step)
        return

    assert geo_mod is not None
    writer.add_scalar("geo/disabled", 0.0, global_step)
    writer.add_scalar("geo/n_nodes", float(geo_mod.tb_last_graph_n_nodes), global_step)
    if geo_mod.tb_last_dist_finite_frac is not None:
        writer.add_scalar(
            "geo/dist_matrix_finite_frac", geo_mod.tb_last_dist_finite_frac, global_step
        )
    if geo_mod.tb_last_geo_train_loss is not None:
        writer.add_scalar("geo/head_train_loss", geo_mod.tb_last_geo_train_loss, global_step)
    writer.add_scalar(
        "geo/head_train_opt_steps", float(geo_mod.tb_last_geo_train_steps), global_step
    )
    if geo_mod.tb_last_phi_mean is not None:
        writer.add_scalar("geo/phi_mean", geo_mod.tb_last_phi_mean, global_step)
        writer.add_scalar("geo/phi_std", float(geo_mod.tb_last_phi_std or 0.0), global_step)
    if geo_mod.tb_last_delta_mean is not None:
        writer.add_scalar("geo/delta_mean", geo_mod.tb_last_delta_mean, global_step)
    writer.add_scalar(
        "geo/intrinsic_ready", 1.0 if geo_mod.geo_intrinsic_ready() else 0.0, global_step
    )


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    env = PointMazeMediumDiverseGRWrapper(
        env_name="PointMaze_Medium_Diverse_GR-v3", img_size=args.img_size
    )
    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    action_repeat = 1
    effective_gamma = args.gamma ** action_repeat
    print(
        f"Maze: PointMaze_Medium_Diverse_GR-v3  img={H}x{W}  act_dim={act_dim}  gamma_eff={effective_gamma:.6f}"
    )
    if args.baseline:
        print("Mode: baseline Dreamer (geometry module disabled)")

    # Models
    encoder = ConvEncoder(embedding_size=args.embed_dim, in_channels=C).to(device)
    decoder = ConvDecoder(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        embedding_size=args.embed_dim,
        out_channels=C,
    ).to(device)
    rssm = RSSM(
        stoch_dim=args.stoch_dim,
        deter_dim=args.deter_dim,
        act_dim=act_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    reward_model = RewardModel(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    cont_model = ContinueModel(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    if args.wm_path:
        wm_path = os.path.expanduser(args.wm_path)
        if not os.path.isfile(wm_path):
            raise FileNotFoundError(f"--wm_path not found: {wm_path}")
        ckpt = torch.load(wm_path, map_location=device)
        for key in ("encoder", "decoder", "rssm", "reward_model", "cont_model"):
            if key not in ckpt:
                raise KeyError(f"Checkpoint missing '{key}'; expected keys like pointmaze_gr_geometry_test world_model.pt")
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        rssm.load_state_dict(ckpt["rssm"])
        reward_model.load_state_dict(ckpt["reward_model"])
        cont_model.load_state_dict(ckpt["cont_model"])
        print(f"Loaded world model weights from {wm_path}")

    actor = Actor(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        act_dim=act_dim,
        hidden_dim=args.actor_hidden_dim,
    ).to(device)
    value_model = ValueModel(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        hidden_dim=args.value_hidden_dim,
    ).to(device)

    world_params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(rssm.parameters())
        + list(reward_model.parameters())
        + list(cont_model.parameters())
    )
    model_opt = torch.optim.Adam(world_params, lr=args.model_lr, eps=args.adam_eps)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=args.adam_eps)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=args.value_lr, eps=args.adam_eps)

    replay = ReplayBuffer(args.replay_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    geo_mod = None
    if not args.baseline:
        geo_mod = ReplayGeometryModule(
            deter_dim=args.deter_dim,
            stoch_dim=args.stoch_dim,
            geo_dim=32,
            geo_hidden=256,
            device=device,
            lambda_dist=args.geo_lambda_dist,
            lambda_front=args.geo_lambda_front,
            lambda_off=args.geo_lambda_off,
            max_nodes=args.geo_max_nodes,
        )

    writer = SummaryWriter(f"{args.log_dir}/{args.run_name}_seed{args.seed}")
    writer.add_text("hyperparameters", str(vars(args)), 0)
    writer.add_scalar("config/baseline_mode", 1.0 if args.baseline else 0.0, 0)

    total_steps = 0
    expl_amount = args.expl_amount

    # ----------------- Seed replay with random episodes -----------------
    print(f"Seeding replay buffer with {args.seed_episodes} episodes...")
    for ep in range(args.seed_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = env.action_space.sample()
            next_obs, r, term, trunc, _ = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            replay.add(
                obs=np.ascontiguousarray(obs, dtype=np.uint8),
                action=action.astype(np.float32),
                reward=float(r),
                next_obs=np.ascontiguousarray(next_obs, dtype=np.uint8),
                done=done,
            )
            obs = next_obs
            ep_ret += float(r)
            total_steps += 1
        print(f"  seed ep {ep+1}/{args.seed_episodes}: return={ep_ret:.2f}")

    # ----------------- Main training loop -----------------
    print(f"\nStarting training for {args.max_episodes} episodes...")
    for episode in range(args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_steps = 0

        # Init posterior
        with torch.no_grad():
            obs_t = torch.tensor(
                np.ascontiguousarray(obs), dtype=torch.float32, device=device
            ).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=args.bit_depth)
            e0 = encoder(obs_t)
            h_state, s_state = rssm.get_init_state(e0)

        while not done:
            # Actor
            encoder.eval()
            rssm.eval()
            actor.eval()
            with torch.no_grad():
                action_t, _ = actor.get_action(h_state, s_state, deterministic=False)
                if expl_amount > 0:
                    action_t = action_t + expl_amount * torch.randn_like(action_t)
                    action_t = torch.clamp(action_t, -1.0, 1.0)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            # Env step
            next_obs, r, term, trunc, _ = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            replay.add(
                obs=np.ascontiguousarray(obs, dtype=np.uint8),
                action=action,
                reward=float(r),
                next_obs=np.ascontiguousarray(next_obs, dtype=np.uint8),
                done=done,
            )
            obs = next_obs
            ep_ret += float(r)
            ep_steps += 1
            total_steps += 1

            # Update posterior
            with torch.no_grad():
                obs_t = torch.tensor(
                    np.ascontiguousarray(obs), dtype=torch.float32, device=device
                ).permute(2, 0, 1).unsqueeze(0)
                preprocess_img(obs_t, depth=args.bit_depth)
                e = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(e, act_t, h_state, s_state, sample=False)

            # ----------------- Training -----------------
            if total_steps % args.collect_interval == 0 and replay.size > (args.seq_len + 2):
                encoder.train()
                decoder.train()
                rssm.train()
                reward_model.train()
                cont_model.train()
                actor.train()
                value_model.train()

                sum_rec = sum_kld = sum_rew = sum_cont = sum_model = 0.0
                sum_actor = sum_value = 0.0
                sum_weighted_ret = 0.0
                gn_model: list[float] = []
                gn_actor: list[float] = []
                gn_value: list[float] = []
                imag_r_mean: list[float] = []
                imag_r_std: list[float] = []
                imag_geo_mean: list[float] = []
                imag_geo_std: list[float] = []
                imag_geo_abs_mean: list[float] = []
                imag_geo_ratio: list[float] = []

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

                    # World model
                    e_t = bottle(encoder, x)
                    h_t, s_t = rssm.get_init_state(e_t[:, 0])

                    states = []
                    priors = []
                    posts = []
                    s_samples = []
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
                    target = x[:, 1 : T + 1]
                    rec_loss = (
                        F.mse_loss(recon, target, reduction="none").sum((2, 3, 4)).mean()
                    )

                    kld = torch.max(
                        kl_divergence(post_dist, prior_dist).sum(-1), free_nats
                    ).mean()

                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_target = rew_seq[:, :T]
                    rew_loss = F.mse_loss(rew_pred, rew_target)

                    cont_logits = bottle(cont_model, h_seq, s_seq)
                    cont_target = (1.0 - done_seq[:, :T]).clamp(0.0, 1.0)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

                    model_loss = rec_loss + args.kl_weight * kld + rew_loss + cont_loss

                    model_opt.zero_grad(set_to_none=True)
                    model_loss.backward()
                    gn_m = torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip)
                    gn_model.append(float(gn_m))
                    model_opt.step()

                    # ------------- Geometry module update (offline) -------------
                    # Posterior (h,s) + encoder embeddings e for cycle-consistent graph
                    if geo_mod is not None and np.random.rand() < 0.1:
                        h_np = h_seq.detach().cpu().numpy()
                        s_np = s_seq.detach().cpu().numpy()
                        e_np = e_t[:, 1 : T + 1].detach().cpu().numpy()
                        traj_h = [h_np[b] for b in range(B)]
                        traj_s = [s_np[b] for b in range(B)]
                        traj_e = [e_np[b] for b in range(B)]
                        traj_ids = [episode * B + b for b in range(B)]
                        geo_mod.add_batch_trajectories(traj_h, traj_s, traj_e, traj_ids)

                    # ------------- Imagination for actor/critic -------------
                    # Start imagination from posterior states
                    B_seq, T_seq, Dh = h_seq.shape
                    Ds = s_seq.size(-1)
                    if (
                        args.imagination_starts
                        and args.imagination_starts > 0
                        and args.imagination_starts < T_seq
                    ):
                        K = args.imagination_starts
                        t_idx = torch.randint(0, T_seq, (B_seq, K), device=device)
                        h_start = (
                            h_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Dh))
                            .reshape(-1, Dh)
                            .detach()
                        )
                        s_start = (
                            s_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Ds))
                            .reshape(-1, Ds)
                            .detach()
                        )
                    else:
                        h_start = h_seq.reshape(-1, Dh).detach()
                        s_start = s_seq.reshape(-1, Ds).detach()

                    h_im_list = [h_start]
                    s_im_list = [s_start]
                    for _ in range(args.imagination_horizon):
                        a_im, _ = actor.get_action(h_im_list[-1], s_im_list[-1], deterministic=False)
                        h_next = rssm.deterministic_state_fwd(
                            h_im_list[-1], s_im_list[-1], a_im
                        )
                        s_next = rssm.state_prior(h_next, sample=True)
                        h_im_list.append(h_next)
                        s_im_list.append(s_next)
                    h_imag = torch.stack(h_im_list, dim=1)
                    s_imag = torch.stack(s_im_list, dim=1)

                    # Predict extrinsic reward / discount
                    rewards_imag = bottle(reward_model, h_imag[:, :-1], s_imag[:, :-1])
                    cont_logits_imag = bottle(cont_model, h_imag[:, 1:], s_imag[:, 1:])
                    pcont_imag = torch.sigmoid(cont_logits_imag).clamp(0.0, 1.0)
                    discounts_imag = effective_gamma * pcont_imag

                    # Intrinsic geometry reward (skipped in --baseline)
                    if geo_mod is None:
                        r_geo = torch.zeros_like(rewards_imag)
                    else:
                        r_geo = geo_mod.intrinsic_reward_for_imagination(h_imag, s_imag)
                    rewards_total = rewards_imag + r_geo

                    # Critic target (λ-returns) with detached targets
                    with torch.no_grad():
                        values_all = bottle(value_model, h_imag, s_imag)  # [B_imag, H+1]
                        lambda_ret = compute_lambda_returns(
                            rewards_total, values_all, discounts_imag, lambda_=args.lambda_
                        )
                        w_val = compute_discount_weights(discounts_imag)

                    values_pred = bottle(
                        value_model, h_imag.detach(), s_imag.detach()
                    )  # [B_imag, H+1]
                    value_loss = ((values_pred[:, :-1] - lambda_ret) ** 2 * w_val).mean()
                    value_opt.zero_grad(set_to_none=True)
                    value_loss.backward()
                    gn_v = torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip)
                    gn_value.append(float(gn_v))
                    value_opt.step()

                    # Actor loss: maximize geometry-augmented returns
                    with torch.no_grad():
                        v_det = bottle(value_model, h_imag, s_imag)
                    lambda_actor = compute_lambda_returns(
                        rewards_total, v_det, discounts_imag, lambda_=args.lambda_
                    )
                    w_actor = compute_discount_weights(discounts_imag)
                    actor_loss = -(w_actor * lambda_actor).mean()
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    gn_a = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    gn_actor.append(float(gn_a))
                    actor_opt.step()

                    sum_rec += float(rec_loss.item())
                    sum_kld += float(kld.item())
                    sum_rew += float(rew_loss.item())
                    sum_cont += float(cont_loss.item())
                    sum_model += float(model_loss.item())
                    sum_actor += float(actor_loss.item())
                    sum_value += float(value_loss.item())
                    sum_weighted_ret += float((w_actor * lambda_actor).mean().item())
                    imag_r_mean.append(float(rewards_imag.mean().item()))
                    imag_r_std.append(float(rewards_imag.std().item()))
                    imag_geo_mean.append(float(r_geo.mean().item()))
                    imag_geo_std.append(float(r_geo.std().item()))
                    imag_geo_abs_mean.append(float(r_geo.abs().mean().item()))
                    denom = float(rewards_imag.abs().mean().item()) + 1e-8
                    imag_geo_ratio.append(float(r_geo.abs().mean().item()) / denom)

                n_ts = float(args.train_steps)
                wm_avg = {
                    "reconstruction": sum_rec / n_ts,
                    "kl": sum_kld / n_ts,
                    "reward_pred": sum_rew / n_ts,
                    "continue": sum_cont / n_ts,
                    "total": sum_model / n_ts,
                    "kl_weighted": args.kl_weight * sum_kld / n_ts,
                }
                grad_avg = {
                    "world_model": float(np.mean(gn_model)) if gn_model else 0.0,
                    "actor": float(np.mean(gn_actor)) if gn_actor else 0.0,
                    "value": float(np.mean(gn_value)) if gn_value else 0.0,
                }
                imag_avg = {
                    "reward_mean": float(np.mean(imag_r_mean)),
                    "reward_std": float(np.mean(imag_r_std)),
                    "r_geo_mean": float(np.mean(imag_geo_mean)),
                    "r_geo_std": float(np.mean(imag_geo_std)),
                    "r_geo_abs_mean": float(np.mean(imag_geo_abs_mean)),
                    "r_geo_over_abs_extrinsic": float(np.mean(imag_geo_ratio)),
                }
                policy_avg = {
                    "actor_loss": sum_actor / n_ts,
                    "value_loss": sum_value / n_ts,
                    "mean_weighted_return": sum_weighted_ret / n_ts,
                }

                if geo_mod is not None:
                    geo_mod.rebuild_graph_and_distances(
                        knn_k=args.geo_knn_k, cross_quantile=args.geo_cross_quantile
                    )
                    geo_mod.train_geo_head(n_steps=20, batch_pairs=1024)
                    geo_mod.build_potential()

                log_training_phase_tensorboard(
                    writer,
                    total_steps,
                    replay_size=replay.size,
                    expl_amount=expl_amount,
                    baseline=args.baseline,
                    geo_mod=geo_mod,
                    wm=wm_avg,
                    grad=grad_avg,
                    imag=imag_avg,
                    policy=policy_avg,
                )

        # end of episode
        if args.expl_decay > 0:
            expl_amount = max(args.expl_min, expl_amount - args.expl_decay)

        writer.add_scalar("train/episode_return", ep_ret, episode)
        writer.add_scalar("episode/return_env_step", ep_ret, total_steps)
        writer.add_scalar("train/episode_steps", ep_steps, episode)
        writer.add_scalar("episode/length_env_step", ep_steps, total_steps)
        writer.add_scalar("train/total_steps", total_steps, episode)
        print(
            f"Episode {episode+1}/{args.max_episodes}  return={ep_ret:.2f}  steps={ep_steps}  total_steps={total_steps}"
        )

    env.close()
    writer.close()


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


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)

