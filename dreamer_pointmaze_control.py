#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from maze_geometry_test import _positions_to_cell_indices, _adj_from_distmat, _find_bridges
from models import RSSM, Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel, ValueModel
from pointmaze_gr_geometry_test import PointMazeMediumDiverseGRWrapper
from pointmaze_gr_geometry_test_topo import (
    ReplayEdgeTopoClassifier,
    ReplayNodeScoreHead,
    _articulation_points_undirected,
    _build_mixed_replay_graph,
    _compute_encoder_temporal_local_global_disagreement,
    _edge_betweenness_brandes_undirected,
    _edge_betweenness_push_to_nodes,
    _kmeans_numpy,
    _label_topology_edges,
    _node_betweenness_brandes_undirected,
    _oracle_free_replay_step_distances,
    _zscore_safe,
)
from utils import ReplayBuffer, bottle, get_device, no_param_grads, preprocess_img, set_seed


# -----------------------------------------------------------------------------
# Oracle-free b(v): novelty (cluster visits) + disagreement + uncertainty + replay graph centrality
# -----------------------------------------------------------------------------


def _replay_graph_centrality_per_global(
    encoder_emb: np.ndarray,
    episode_ids: np.ndarray,
    rng: np.random.Generator,
    n_graph_max: int,
    k_knn: int,
) -> np.ndarray:
    """Mean z-scored (node betweenness, edge push, articulation) on mixed replay graph."""
    n = len(encoder_emb)
    out = np.zeros(n, dtype=np.float32)
    idx_global, _, adj_list = _build_mixed_replay_graph(
        encoder_emb,
        episode_ids,
        n_graph_max=int(n_graph_max),
        k_knn=int(k_knn),
    )
    m = len(adj_list)
    if m < 4:
        return out

    nb = _node_betweenness_brandes_undirected(adj_list)
    et = _edge_betweenness_brandes_undirected(adj_list)
    eb = _edge_betweenness_push_to_nodes(adj_list, et)
    art = _articulation_points_undirected(adj_list)
    bundle = np.mean(
        np.stack(
            [
                _zscore_safe(nb.astype(np.float32)),
                _zscore_safe(eb.astype(np.float32)),
                _zscore_safe(art.astype(np.float32)),
            ],
            axis=0,
        ),
        axis=0,
    ).astype(np.float32)
    for loc, g in enumerate(idx_global.tolist()):
        out[int(g)] = float(bundle[loc])
    return out


def build_oracle_free_b_targets(
    encoder_emb: np.ndarray,
    episode_ids: np.ndarray,
    rng: np.random.Generator,
    *,
    n_graph_max: int,
    k_knn: int,
    intrinsic_mode: str,
    novelty_k: int = 48,
) -> np.ndarray:
    """Compose b(v) with terms from replay only; no oracle maze / bridge mask."""
    n = int(len(encoder_emb))
    X = np.asarray(encoder_emb, dtype=np.float32)
    ep = np.asarray(episode_ids, dtype=np.int64)

    alpha = beta = gamma = delta = 1.0
    if intrinsic_mode == "novelty":
        alpha, beta, gamma, delta = 1.0, 0.0, 0.0, 0.0
    elif intrinsic_mode == "node":
        alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0

    # 1) Novelty: cluster visitation (not oracle cells)
    k = int(max(2, min(novelty_k, max(2, n // 8))))
    lab = _kmeans_numpy(X, k=k, rng=rng, n_iter=20)
    counts = np.bincount(lab, minlength=k).astype(np.int64)
    novelty = 1.0 / np.sqrt(np.maximum(1, counts[lab]).astype(np.float32))

    # 2) disagreement
    src_lg, disc_lg = _compute_encoder_temporal_local_global_disagreement(
        encoder_emb=X,
        episode_ids=ep,
        rng=rng,
        k_nn=12,
        candidate_pool=min(1800, max(256, n)),
        max_sources=min(2000, max(32, n)),
        min_labeled_neighbors=4,
    )
    disc_all = np.zeros(n, dtype=np.float32)
    if len(src_lg) > 0:
        disc_all[src_lg.astype(np.int64)] = disc_lg.astype(np.float32)

    # 3) uncertainty: per-source std of replay-step distances
    uncert = np.zeros(n, dtype=np.float32)
    if intrinsic_mode != "novelty":
        pair_pool = _oracle_free_replay_step_distances(
            ep,
            n_pairs_target=min(max(2000, n), 10000),
            max_sources=128,
            rng=rng,
        )
        if pair_pool is not None:
            ii, _, dd = pair_pool
            by_src: dict[int, list[float]] = {}
            for a, d in zip(ii.tolist(), dd.tolist()):
                by_src.setdefault(int(a), []).append(float(d))
            for a, vals in by_src.items():
                if len(vals) >= 2:
                    uncert[a] = float(np.std(np.asarray(vals, dtype=np.float32)))

    # 4) centrality on replay graph
    cent = np.zeros(n, dtype=np.float32)
    if intrinsic_mode != "novelty" and delta != 0.0:
        cent = _replay_graph_centrality_per_global(X, ep, rng, n_graph_max, k_knn)

    nov_z = _zscore_safe(novelty)
    disc_z = _zscore_safe(disc_all)
    unc_z = _zscore_safe(uncert)
    cent_z = _zscore_safe(cent)

    denom = max(1e-8, alpha + beta + gamma + delta)
    score = (alpha * nov_z + beta * disc_z + gamma * unc_z + delta * cent_z) / denom
    return score.astype(np.float32)


# -----------------------------------------------------------------------------
# Env wrapper: fixed start / small start subsets (control regime)
# -----------------------------------------------------------------------------


class PointMazeControlWrapper(PointMazeMediumDiverseGRWrapper):
    """reset_mode: random_cell (analysis) | fixed_start | start_subset."""

    def __init__(
        self,
        env_name: str = "PointMaze_Medium_Diverse_GR-v3",
        img_size: int = 64,
        *,
        reset_mode: str = "random_cell",
        fixed_start_cell: tuple[int, int] | None = None,
        start_cells: list[tuple[int, int]] | None = None,
    ):
        super().__init__(env_name=env_name, img_size=img_size)
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

    def reset(self, **kwargs):
        opts = dict(kwargs.get("options") or {})
        kwargs = dict(kwargs)

        if self.reset_mode == "random_cell":
            if self._free_cells:
                r, c = self._free_cells[np.random.randint(0, len(self._free_cells))]
                opts["reset_cell"] = np.array([r, c], dtype=np.int64)
        elif self.reset_mode == "fixed_start":
            r, c = self.fixed_start_cell  # type: ignore[misc]
            opts["reset_cell"] = np.array([r, c], dtype=np.int64)
        elif self.reset_mode == "start_subset":
            r, c = self.start_cells[np.random.randint(0, len(self.start_cells))]
            opts["reset_cell"] = np.array([r, c], dtype=np.int64)
        else:
            raise ValueError(f"Unknown reset_mode {self.reset_mode}")

        kwargs["options"] = opts
        obs_dict, info = self._env.reset(**kwargs)
        self._update_agent_pos(obs_dict)
        frame = self._env.render()
        frame = self._resize_frame(frame).astype(np.uint8)
        return frame, info


# -----------------------------------------------------------------------------
# Distillation heads: latent b(h,s) and edge logits from (h,s, h',s')
# -----------------------------------------------------------------------------


class NodeScoreDistillHead(nn.Module):
    def __init__(self, deter_dim: int, stoch_dim: int, hidden_dim: int):
        super().__init__()
        d = int(deter_dim) + int(stoch_dim)
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, s], dim=-1)
        return self.net(x).squeeze(-1)


class EdgeTopoDistillHead(nn.Module):
    def __init__(self, deter_dim: int, stoch_dim: int, hidden_dim: int, n_classes: int = 3):
        super().__init__()
        d = 2 * (int(deter_dim) + int(stoch_dim))
        self.net = nn.Sequential(
            nn.Linear(d, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, int(n_classes)),
        )

    def forward(self, h_i, s_i, h_j, s_j) -> torch.Tensor:
        return self.net(torch.cat([h_i, s_i, h_j, s_j], dim=-1))


# -----------------------------------------------------------------------------
# Topology module: buffers, refit encoder heads, distill, imagination bonuses
# -----------------------------------------------------------------------------


class TopologyControlModule:
    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        embed_dim: int,
        device: torch.device,
        *,
        max_nodes: int,
        graph_max: int,
        knn_k: int,
        intrinsic_mode: str,
        novelty_k: int,
        hidden: int = 256,
        distill_steps: int = 200,
        edge_batch: int = 256,
        node_epochs: int = 40,
        edge_epochs: int = 40,
        bottleneck_frac: float = 0.2,
    ):
        self.device = device
        self.deter_dim = int(deter_dim)
        self.stoch_dim = int(stoch_dim)
        self.embed_dim = int(embed_dim)
        self.max_nodes = int(max_nodes)
        self.graph_max = int(graph_max)
        self.knn_k = int(knn_k)
        self.intrinsic_mode = intrinsic_mode
        self.novelty_k = int(novelty_k)
        self.hidden = int(hidden)
        self.distill_steps = int(distill_steps)
        self.edge_batch = int(edge_batch)
        self.node_epochs = int(node_epochs)
        self.edge_epochs = int(edge_epochs)
        self.bottleneck_frac = float(bottleneck_frac)

        self.b_head_e = ReplayNodeScoreHead(self.embed_dim, self.hidden).to(device)
        self.edge_head_e = ReplayEdgeTopoClassifier(self.embed_dim, self.hidden, n_classes=3).to(device)
        self.b_head_hs = NodeScoreDistillHead(self.deter_dim, self.stoch_dim, self.hidden).to(device)
        self.edge_head_hs = EdgeTopoDistillHead(self.deter_dim, self.stoch_dim, self.hidden, n_classes=3).to(
            device
        )

        self.opt_b_e = torch.optim.Adam(self.b_head_e.parameters(), lr=3e-4)
        self.opt_edge_e = torch.optim.Adam(self.edge_head_e.parameters(), lr=3e-4)
        self.opt_b_hs = torch.optim.Adam(self.b_head_hs.parameters(), lr=3e-4)
        self.opt_edge_hs = torch.optim.Adam(self.edge_head_hs.parameters(), lr=3e-4)

        self._h_nodes: List[np.ndarray] = []
        self._s_nodes: List[np.ndarray] = []
        self._e_nodes: List[np.ndarray] = []
        self._episode_ids: List[int] = []
        self._next_traj_id = 0

        self._ready_node = False
        self._ready_edge = False
        self.tb_last_node_loss: float | None = None
        self.tb_last_edge_loss: float | None = None

    def add_batch_trajectories(
        self,
        traj_h: List[np.ndarray],
        traj_s: List[np.ndarray],
        traj_e: List[np.ndarray],
    ):
        for ep_h, ep_s, ep_e in zip(traj_h, traj_s, traj_e):
            T = len(ep_h)
            if T < 2:
                continue
            ep_id = self._next_traj_id
            self._next_traj_id += 1
            for t in range(T):
                if len(self._h_nodes) >= self.max_nodes:
                    self._h_nodes.pop(0)
                    self._s_nodes.pop(0)
                    self._e_nodes.pop(0)
                    self._episode_ids.pop(0)
                self._h_nodes.append(ep_h[t])
                self._s_nodes.append(ep_s[t])
                self._e_nodes.append(ep_e[t])
                self._episode_ids.append(ep_id)

    def refit(self, seed: int) -> None:
        rng = np.random.default_rng(int(seed))
        n = len(self._episode_ids)
        if n < 64:
            self._ready_node = False
            self._ready_edge = False
            return

        self.tb_last_node_loss = None
        self.tb_last_edge_loss = None

        enc = np.asarray(self._e_nodes, dtype=np.float32)
        ep = np.asarray(self._episode_ids, dtype=np.int64)
        h_np = np.asarray(self._h_nodes, dtype=np.float32)
        s_np = np.asarray(self._s_nodes, dtype=np.float32)

        # --- targets (node) ---
        if self.intrinsic_mode in ("novelty", "node", "both"):
            y_b = build_oracle_free_b_targets(
                enc,
                ep,
                rng,
                n_graph_max=self.graph_max,
                k_knn=self.knn_k,
                intrinsic_mode="novelty" if self.intrinsic_mode == "novelty" else "node",
                novelty_k=self.novelty_k,
            )
            e_t = torch.tensor(enc, dtype=torch.float32, device=self.device)
            y_t = torch.tensor(y_b, dtype=torch.float32, device=self.device)
            perm = rng.permutation(n)
            n_val = max(1, int(round(0.12 * n)))
            tr = perm[n_val:]
            va = perm[:n_val]

            self.b_head_e.train()
            best_loss = float("inf")
            best_state = None
            losses: list[float] = []
            for _ in range(self.node_epochs):
                idx = rng.choice(tr, size=min(512, len(tr)), replace=len(tr) < 512)
                it = torch.tensor(idx, dtype=torch.long, device=self.device)
                pred = self.b_head_e(e_t[it])
                loss = F.mse_loss(pred, y_t[it])
                self.opt_b_e.zero_grad(set_to_none=True)
                loss.backward()
                self.opt_b_e.step()
                losses.append(float(loss.item()))
                with torch.no_grad():
                    iv = torch.tensor(va, dtype=torch.long, device=self.device)
                    vloss = F.mse_loss(self.b_head_e(e_t[iv]), y_t[iv])
                    if float(vloss.item()) < best_loss:
                        best_loss = float(vloss.item())
                        best_state = {k: v.detach().cpu() for k, v in self.b_head_e.state_dict().items()}
            if best_state is not None:
                self.b_head_e.load_state_dict(best_state)
            self.b_head_e.eval()
            self.tb_last_node_loss = float(np.mean(losses)) if losses else None

            with torch.no_grad():
                b_t = self.b_head_e(e_t).detach()

            self.b_head_hs.train()
            for _ in range(self.distill_steps):
                idx = rng.integers(0, n, size=min(512, n))
                hi = torch.tensor(h_np[idx], dtype=torch.float32, device=self.device)
                si = torch.tensor(s_np[idx], dtype=torch.float32, device=self.device)
                pred = self.b_head_hs(hi, si)
                loss = F.mse_loss(pred, b_t[idx])
                self.opt_b_hs.zero_grad(set_to_none=True)
                loss.backward()
                self.opt_b_hs.step()
            self.b_head_hs.eval()

        self._ready_node = self.intrinsic_mode in ("novelty", "node", "both")

        # --- edge classifier + distill ---
        edge_ok = False
        if self.intrinsic_mode in ("edge", "both"):
            idx_global, _, adj_list = _build_mixed_replay_graph(
                enc, ep, n_graph_max=self.graph_max, k_knn=self.knn_k
            )
            labeled = _label_topology_edges(
                idx_global,
                adj_list,
                ep,
                enc,
                rng,
                bottleneck_top_frac=self.bottleneck_frac,
            )
            if labeled is not None and len(labeled) >= 16:
                train_perm = rng.permutation(len(labeled))
                n_val_e = max(1, int(round(0.12 * len(labeled))))
                tr_e = labeled[train_perm[n_val_e:]]
                va_e = labeled[train_perm[:n_val_e]]

                va_u = torch.tensor(va_e[:, 0], dtype=torch.long, device=self.device)
                va_v = torch.tensor(va_e[:, 1], dtype=torch.long, device=self.device)
                va_y = torch.tensor(va_e[:, 2], dtype=torch.long, device=self.device)
                e_t = torch.tensor(enc, dtype=torch.float32, device=self.device)

                self.edge_head_e.train()
                best_ce = float("inf")
                best_edge_state = None
                losses_e: list[float] = []
                for _ in range(self.edge_epochs):
                    sel = rng.choice(len(tr_e), size=min(self.edge_batch, len(tr_e)), replace=len(tr_e) < self.edge_batch)
                    uu = torch.tensor(tr_e[sel, 0], dtype=torch.long, device=self.device)
                    vv = torch.tensor(tr_e[sel, 1], dtype=torch.long, device=self.device)
                    yy = torch.tensor(tr_e[sel, 2], dtype=torch.long, device=self.device)
                    logits = self.edge_head_e(e_t[uu], e_t[vv])
                    loss_e = F.cross_entropy(logits, yy)
                    self.opt_edge_e.zero_grad(set_to_none=True)
                    loss_e.backward()
                    self.opt_edge_e.step()
                    losses_e.append(float(loss_e.item()))
                    with torch.no_grad():
                        vlogits = self.edge_head_e(e_t[va_u], e_t[va_v])
                        vloss = F.cross_entropy(vlogits, va_y)
                        if float(vloss.item()) < best_ce:
                            best_ce = float(vloss.item())
                            best_edge_state = {k: v.detach().cpu() for k, v in self.edge_head_e.state_dict().items()}
                if best_edge_state is not None:
                    self.edge_head_e.load_state_dict(best_edge_state)
                self.edge_head_e.eval()
                self.tb_last_edge_loss = float(np.mean(losses_e)) if losses_e else None

                h_t = torch.tensor(h_np, dtype=torch.float32, device=self.device)
                s_t = torch.tensor(s_np, dtype=torch.float32, device=self.device)

                self.edge_head_hs.train()
                for _ in range(self.distill_steps):
                    sel = rng.choice(len(tr_e), size=min(self.edge_batch, len(tr_e)), replace=len(tr_e) < self.edge_batch)
                    uu = torch.tensor(tr_e[sel, 0], dtype=torch.long, device=self.device)
                    vv = torch.tensor(tr_e[sel, 1], dtype=torch.long, device=self.device)
                    yy = torch.tensor(tr_e[sel, 2], dtype=torch.long, device=self.device)
                    logits = self.edge_head_hs(h_t[uu], s_t[uu], h_t[vv], s_t[vv])
                    loss_e = F.cross_entropy(logits, yy)
                    self.opt_edge_hs.zero_grad(set_to_none=True)
                    loss_e.backward()
                    self.opt_edge_hs.step()
                self.edge_head_hs.eval()
                edge_ok = True

        self._ready_edge = edge_ok

    def imagination_intrinsic(
        self,
        h_imag: torch.Tensor,
        s_imag: torch.Tensor,
        lambda_b: float,
        lambda_e: float,
    ) -> torch.Tensor:
        """r_int[b,t] for t in 0..H-1 matching transitions (t,t+1)."""
        B, Hp1, Dh = h_imag.shape
        Ds = s_imag.size(-1)
        H = Hp1 - 1
        device = h_imag.device
        if not self._ready_node and not self._ready_edge:
            return torch.zeros((B, H), device=device, dtype=torch.float32)

        self.b_head_hs.eval()
        self.edge_head_hs.eval()
        with torch.no_grad():
            r = torch.zeros((B, H), device=device, dtype=torch.float32)
            if (
                self._ready_node
                and self.intrinsic_mode in ("novelty", "node", "both")
                and lambda_b > 0
            ):
                b = self.b_head_hs(h_imag, s_imag)
                r = r + float(lambda_b) * (b[:, 1:] - b[:, :-1])
            if self._ready_edge and self.intrinsic_mode in ("edge", "both") and lambda_e > 0:
                h_i = h_imag[:, :-1].reshape(-1, Dh)
                s_i = s_imag[:, :-1].reshape(-1, Ds)
                h_j = h_imag[:, 1:].reshape(-1, Dh)
                s_j = s_imag[:, 1:].reshape(-1, Ds)
                logits = self.edge_head_hs(h_i, s_i, h_j, s_j)
                p_bot = F.softmax(logits, dim=-1)[:, 2].view(B, H)
                r = r + float(lambda_e) * p_bot
        return r


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


def _bridge_crossing_count(geodesic, pos_seq: np.ndarray) -> int:
    """Evaluation-only: count transitions across oracle grid bridges (doorways)."""
    dist_mat = geodesic.dist_matrix
    adj = _adj_from_distmat(dist_mat)
    bridges = _find_bridges(adj)
    bridge_set = set(bridges)
    if len(pos_seq) < 2:
        return 0
    cells = _positions_to_cell_indices(geodesic, pos_seq)
    n_cross = 0
    for a, b in zip(cells[:-1], cells[1:]):
        u, v = (int(a), int(b)) if int(a) < int(b) else (int(b), int(a))
        if (u, v) in bridge_set:
            n_cross += 1
    return n_cross


@torch.no_grad()
def run_periodic_eval(
    env: PointMazeControlWrapper,
    encoder: torch.nn.Module,
    rssm: torch.nn.Module,
    actor: torch.nn.Module,
    device: torch.device,
    bit_depth: int,
    n_episodes: int,
    geodesic,
    action_repeat: int,
) -> dict[str, float]:
    """Deterministic actor rollouts (no exploration noise). Same reset policy as training."""
    encoder.eval()
    rssm.eval()
    actor.eval()
    rets: list[float] = []
    lens: list[int] = []
    uniques: list[float] = []
    bridges: list[float] = []
    for _ in range(int(n_episodes)):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_steps = 0
        cells: set[int] = set()
        pos_traj: list[np.ndarray] = []
        c0 = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
        cells.add(c0)
        pos_traj.append(env.agent_pos.copy())
        obs_t = (
            torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        preprocess_img(obs_t, depth=bit_depth)
        e0 = encoder(obs_t)
        h_state, s_state = rssm.get_init_state(e0)
        while not done:
            action_t, _ = actor.get_action(h_state, s_state, deterministic=True)
            action = action_t.squeeze(0).cpu().numpy().astype(np.float32)
            next_obs, r, term, trunc, _ = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            obs = next_obs
            ep_ret += float(r)
            ep_steps += 1
            c = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
            cells.add(c)
            pos_traj.append(env.agent_pos.copy())
            obs_t = (
                torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            preprocess_img(obs_t, depth=bit_depth)
            e = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(e, act_t, h_state, s_state, sample=False)
        rets.append(ep_ret)
        lens.append(ep_steps)
        uniques.append(float(len(cells)))
        if len(pos_traj) >= 2:
            bridges.append(float(_bridge_crossing_count(geodesic, np.stack(pos_traj, axis=0))))
        else:
            bridges.append(0.0)
    return {
        "return_mean": float(np.mean(rets)),
        "return_std": float(np.std(rets)),
        "length_mean": float(np.mean(lens)),
        "unique_cells_mean": float(np.mean(uniques)),
        "bridge_crossings_mean": float(np.mean(bridges)),
    }


def build_parser():
    p = argparse.ArgumentParser(
        description="Dreamer PointMaze control: fixed starts + oracle-free topology shaping"
    )
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
        choices=["random_cell", "fixed_start", "start_subset"],
        help="Control: fixed_start / start_subset; analysis: random_cell.",
    )
    p.add_argument("--fixed_start_row", type=int, default=1)
    p.add_argument("--fixed_start_col", type=int, default=1)
    p.add_argument(
        "--start_subset",
        type=str,
        default="",
        help="Semicolon-separated row,col pairs for start_subset, e.g. '1,1;5,3'. Empty uses first 3 free cells.",
    )

    p.add_argument(
        "--intrinsic",
        type=str,
        default="none",
        choices=["none", "novelty", "node", "edge", "both"],
    )
    p.add_argument("--lambda_b", type=float, default=0.1, help="Scale for Δb in imagination.")
    p.add_argument("--lambda_e", type=float, default=0.1, help="Scale for bottleneck edge prob.")

    p.add_argument("--topo_buffer_max", type=int, default=2500)
    p.add_argument("--graph_max", type=int, default=1800)
    p.add_argument("--graph_knn", type=int, default=10)
    p.add_argument("--novelty_k", type=int, default=48)
    p.add_argument("--topo_distill_steps", type=int, default=200)
    p.add_argument("--topo_node_epochs", type=int, default=40)
    p.add_argument("--topo_edge_epochs", type=int, default=40)
    p.add_argument("--topo_edge_batch", type=int, default=256)
    p.add_argument("--bottleneck_frac", type=float, default=0.2)

    p.add_argument("--wm_path", type=str, default="")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="pointmaze_control")
    p.add_argument("--log_coverage", action="store_true", help="Log unique cells per episode (eval-oracle).")
    p.add_argument(
        "--eval_interval",
        type=int,
        default=20,
        help="Run deterministic eval every N training episodes (0 = disabled).",
    )
    p.add_argument(
        "--eval_episodes",
        type=int,
        default=5,
        help="Number of rollouts per eval phase.",
    )

    return p


def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    start_cells: list[tuple[int, int]] | None = None
    if args.reset_mode == "start_subset" and args.start_subset.strip():
        start_cells = []
        for part in args.start_subset.split(";"):
            part = part.strip()
            if not part:
                continue
            r, c = part.split(",")
            start_cells.append((int(r.strip()), int(c.strip())))

    fixed_cell = (args.fixed_start_row, args.fixed_start_col) if args.reset_mode == "fixed_start" else None

    env = PointMazeControlWrapper(
        img_size=args.img_size,
        reset_mode=args.reset_mode,
        fixed_start_cell=fixed_cell,
        start_cells=start_cells,
    )
    geodesic = env.geodesic

    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    action_repeat = 1
    effective_gamma = args.gamma**action_repeat

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
        ckpt = torch.load(wm_path, map_location=device)
        for key in ("encoder", "decoder", "rssm", "reward_model", "cont_model"):
            if key not in ckpt:
                raise KeyError(f"Checkpoint missing '{key}'")
        encoder.load_state_dict(ckpt["encoder"])
        decoder.load_state_dict(ckpt["decoder"])
        rssm.load_state_dict(ckpt["rssm"])
        reward_model.load_state_dict(ckpt["reward_model"])
        cont_model.load_state_dict(ckpt["cont_model"])
        print(f"Loaded world model from {wm_path}")

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

    topo_mod: TopologyControlModule | None = None
    if args.intrinsic != "none":
        topo_mod = TopologyControlModule(
            deter_dim=args.deter_dim,
            stoch_dim=args.stoch_dim,
            embed_dim=args.embed_dim,
            device=device,
            max_nodes=args.topo_buffer_max,
            graph_max=args.graph_max,
            knn_k=args.graph_knn,
            intrinsic_mode=args.intrinsic,
            novelty_k=args.novelty_k,
            distill_steps=args.topo_distill_steps,
            edge_batch=args.topo_edge_batch,
            node_epochs=args.topo_node_epochs,
            edge_epochs=args.topo_edge_epochs,
            bottleneck_frac=args.bottleneck_frac,
        )

    writer = SummaryWriter(f"{args.log_dir}/{args.run_name}_intrinsic_{args.intrinsic}_seed{args.seed}")
    writer.add_text("hyperparameters", str(vars(args)), 0)

    total_steps = 0
    expl_amount = args.expl_amount

    print(f"Seeding replay with {args.seed_episodes} episodes...")
    for ep in range(args.seed_episodes):
        obs, _ = env.reset()
        done = False
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
            total_steps += 1

    print(f"Training {args.max_episodes} episodes  intrinsic={args.intrinsic}  reset={args.reset_mode}")

    for episode in range(args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        ep_steps = 0
        ep_cells: set[int] = set()
        ep_pos_traj: list[np.ndarray] = []
        if args.log_coverage:
            c0 = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
            ep_cells.add(c0)
            ep_pos_traj.append(env.agent_pos.copy())

        ep_h_list: list[np.ndarray] = []
        ep_s_list: list[np.ndarray] = []
        ep_e_list: list[np.ndarray] = []

        with torch.no_grad():
            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(
                2, 0, 1
            ).unsqueeze(0)
            preprocess_img(obs_t, depth=args.bit_depth)
            e0 = encoder(obs_t)
            h_state, s_state = rssm.get_init_state(e0)
            ep_h_list.append(h_state.squeeze(0).cpu().numpy())
            ep_s_list.append(s_state.squeeze(0).cpu().numpy())
            ep_e_list.append(e0.squeeze(0).cpu().numpy())

        while not done:
            encoder.eval()
            rssm.eval()
            actor.eval()
            with torch.no_grad():
                action_t, _ = actor.get_action(h_state, s_state, deterministic=False)
                if expl_amount > 0:
                    action_t = action_t + expl_amount * torch.randn_like(action_t)
                    action_t = torch.clamp(action_t, -1.0, 1.0)
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

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

            if args.log_coverage:
                c = int(_positions_to_cell_indices(geodesic, env.agent_pos.reshape(1, -1))[0])
                ep_cells.add(c)
                ep_pos_traj.append(env.agent_pos.copy())

            with torch.no_grad():
                obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(
                    2, 0, 1
                ).unsqueeze(0)
                preprocess_img(obs_t, depth=args.bit_depth)
                e = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(e, act_t, h_state, s_state, sample=False)
                ep_h_list.append(h_state.squeeze(0).cpu().numpy())
                ep_s_list.append(s_state.squeeze(0).cpu().numpy())
                ep_e_list.append(e.squeeze(0).cpu().numpy())

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
                sum_imag_r = sum_imag_r_int = 0.0
                gn_model: list[float] = []
                gn_actor: list[float] = []
                gn_value: list[float] = []

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
                    rec_loss = F.mse_loss(recon, target, reduction="none").sum((2, 3, 4)).mean()
                    kld = torch.max(kl_divergence(post_dist, prior_dist).sum(-1), free_nats).mean()
                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_loss = F.mse_loss(rew_pred, rew_seq[:, :T])
                    cont_logits = bottle(cont_model, h_seq, s_seq)
                    cont_target = (1.0 - done_seq[:, :T]).clamp(0.0, 1.0)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

                    model_loss = rec_loss + args.kl_weight * kld + rew_loss + cont_loss
                    model_opt.zero_grad(set_to_none=True)
                    model_loss.backward()
                    gn_m = torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip)
                    gn_model.append(float(gn_m))
                    model_opt.step()
                    sum_rec += float(rec_loss.item())
                    sum_kld += float(kld.item())
                    sum_rew += float(rew_loss.item())
                    sum_cont += float(cont_loss.item())
                    sum_model += float(model_loss.item())

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

                    with (
                        no_param_grads(rssm),
                        no_param_grads(reward_model),
                        no_param_grads(cont_model),
                    ):
                        h_im_list = [h_start]
                        s_im_list = [s_start]
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

                    if topo_mod is None:
                        r_int = torch.zeros_like(rewards_imag)
                    else:
                        r_int = topo_mod.imagination_intrinsic(h_imag, s_imag, args.lambda_b, args.lambda_e)

                    rewards_total = rewards_imag + r_int
                    sum_imag_r += float(rewards_imag.mean().item())
                    sum_imag_r_int += float(r_int.mean().item())

                    with torch.no_grad():
                        values_tgt = bottle(value_model, h_imag, s_imag)
                        lambda_ret = compute_lambda_returns(
                            rewards_total.detach(), values_tgt, discounts_imag.detach(), lambda_=args.lambda_
                        )
                        w_val = compute_discount_weights(discounts_imag.detach())

                    values_pred = bottle(value_model, h_imag.detach(), s_imag.detach())
                    value_loss = ((values_pred[:, :-1] - lambda_ret) ** 2 * w_val).mean()
                    value_opt.zero_grad(set_to_none=True)
                    value_loss.backward()
                    gn_v = torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip)
                    gn_value.append(float(gn_v))
                    value_opt.step()
                    sum_value += float(value_loss.item())

                    mean_a, std_a = actor.forward(h_imag[:, :-1].detach(), s_imag[:, :-1].detach())
                    noise_a = torch.randn_like(mean_a)
                    raw_a = mean_a + std_a * noise_a
                    entropy = (
                        Normal(mean_a, std_a).entropy()
                        + torch.log(1 - torch.tanh(raw_a).pow(2) + 1e-6)
                    ).sum(dim=-1).mean()

                    with no_param_grads(value_model):
                        values_for_actor = bottle(value_model, h_imag, s_imag)
                    w_actor = compute_discount_weights(discounts_imag.detach())
                    lambda_actor = compute_lambda_returns(
                        rewards_total, values_for_actor, discounts_imag, lambda_=args.lambda_
                    )
                    actor_loss = -(w_actor.detach() * lambda_actor).mean() - args.actor_entropy_scale * entropy
                    actor_opt.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    gn_a = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip)
                    gn_actor.append(float(gn_a))
                    actor_opt.step()
                    sum_actor += float(actor_loss.item())

                if topo_mod is not None:
                    topo_mod.refit(seed=args.seed + episode)

                n_ts = float(args.train_steps)
                writer.add_scalar("replay/size", replay.size, total_steps)
                writer.add_scalar("train/exploration_noise", expl_amount, total_steps)
                writer.add_scalar("loss/reconstruction", sum_rec / n_ts, total_steps)
                writer.add_scalar("loss/kl", sum_kld / n_ts, total_steps)
                writer.add_scalar("loss/reward_pred", sum_rew / n_ts, total_steps)
                writer.add_scalar("loss/continue", sum_cont / n_ts, total_steps)
                writer.add_scalar("loss/model_total", sum_model / n_ts, total_steps)
                writer.add_scalar("loss/kl_weighted", args.kl_weight * sum_kld / n_ts, total_steps)
                writer.add_scalar("loss/actor", sum_actor / n_ts, total_steps)
                writer.add_scalar("loss/value", sum_value / n_ts, total_steps)
                writer.add_scalar("grad/world_model", float(np.mean(gn_model)) if gn_model else 0.0, total_steps)
                writer.add_scalar("grad/actor", float(np.mean(gn_actor)) if gn_actor else 0.0, total_steps)
                writer.add_scalar("grad/value", float(np.mean(gn_value)) if gn_value else 0.0, total_steps)
                writer.add_scalar("imag/reward_mean", sum_imag_r / n_ts, total_steps)
                writer.add_scalar("imag/r_int_mean", sum_imag_r_int / n_ts, total_steps)
                denom = abs(sum_imag_r / n_ts) + 1e-8
                writer.add_scalar("imag/r_int_over_abs_extrinsic", (sum_imag_r_int / n_ts) / denom, total_steps)
                if topo_mod:
                    writer.add_scalar("topo/ready_node", 1.0 if topo_mod._ready_node else 0.0, total_steps)
                    writer.add_scalar("topo/ready_edge", 1.0 if topo_mod._ready_edge else 0.0, total_steps)
                if topo_mod and topo_mod.tb_last_node_loss is not None:
                    writer.add_scalar("topo/node_train_mse", topo_mod.tb_last_node_loss, total_steps)
                if topo_mod and topo_mod.tb_last_edge_loss is not None:
                    writer.add_scalar("topo/edge_train_ce", topo_mod.tb_last_edge_loss, total_steps)

        if topo_mod is not None and len(ep_h_list) >= 2:
            topo_mod.add_batch_trajectories(
                [np.stack(ep_h_list, axis=0)],
                [np.stack(ep_s_list, axis=0)],
                [np.stack(ep_e_list, axis=0)],
            )

        if args.expl_decay > 0:
            expl_amount = max(args.expl_min, expl_amount - args.expl_decay)

        writer.add_scalar("train/episode_return", ep_ret, episode)
        writer.add_scalar("episode/return_env_step", ep_ret, total_steps)
        if args.log_coverage:
            writer.add_scalar("eval/unique_cells_episode", float(len(ep_cells)), episode)
            if len(ep_pos_traj) >= 2:
                writer.add_scalar(
                    "eval/bridge_crossings_episode",
                    float(_bridge_crossing_count(geodesic, np.stack(ep_pos_traj, axis=0))),
                    episode,
                )

        print(
            f"Episode {episode+1}/{args.max_episodes}  return={ep_ret:.2f}  steps={ep_steps}  total_steps={total_steps}"
        )

        if args.eval_interval > 0 and (episode + 1) % args.eval_interval == 0:
            ev = run_periodic_eval(
                env,
                encoder,
                rssm,
                actor,
                device,
                args.bit_depth,
                args.eval_episodes,
                geodesic,
                action_repeat,
            )
            writer.add_scalar("eval/return_mean", ev["return_mean"], total_steps)
            writer.add_scalar("eval/return_std", ev["return_std"], total_steps)
            writer.add_scalar("eval/length_mean", ev["length_mean"], total_steps)
            writer.add_scalar("eval/unique_cells_mean", ev["unique_cells_mean"], total_steps)
            writer.add_scalar("eval/bridge_crossings_mean", ev["bridge_crossings_mean"], total_steps)
            print(
                f"  [eval] return={ev['return_mean']:.2f}±{ev['return_std']:.2f}  "
                f"len={ev['length_mean']:.1f}  unique_cells={ev['unique_cells_mean']:.1f}  "
                f"bridges={ev['bridge_crossings_mean']:.2f}"
            )

    env.close()
    writer.close()


if __name__ == "__main__":
    main(build_parser().parse_args())