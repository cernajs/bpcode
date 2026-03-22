#!/usr/bin/env python3
"""
Presentation-quality figures for Dreamer + GeoEncoder on maze_env / custom_maze.

Use after training (in the same process or after load_maze_dreamer_checkpoint):

  from maze_dreamer_visuals import (
      rollout_collect_latents,
      figure_maze_hs_pca,
      figure_geo_vs_geodesic,
      figure_imagination_geo_guidance,
      save_all_maze_geo_figures,
      save_maze_dreamer_checkpoint,
      load_maze_dreamer_checkpoint,
  )

Requires matplotlib (see requirements.txt).
"""

from __future__ import annotations

import json
import os

# utils may import mujoco; egl is often invalid on macOS — fall back like dreamer_geo_old
_mgl = os.environ.get("MUJOCO_GL", "")
if not _mgl or _mgl.lower() == "egl":
    os.environ["MUJOCO_GL"] = "glfw"
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from utils import ENV_ACTION_REPEAT, preprocess_img, make_env


def _get_goal_latent(
    env,
    encoder: nn.Module,
    rssm: nn.Module,
    device: torch.device,
    bit_depth: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not hasattr(env, "get_goal_observation"):
        return None
    goal_obs = env.get_goal_observation()
    obs_t = (
        torch.tensor(np.ascontiguousarray(goal_obs), dtype=torch.float32, device=device)
        .permute(2, 0, 1)
        .unsqueeze(0)
    )
    preprocess_img(obs_t, depth=bit_depth)
    with torch.no_grad():
        e = encoder(obs_t)
        h_init, s_init = rssm.get_init_state(e)
        a_dummy = torch.zeros((1, env.action_space.shape[0]), device=device)
        h_goal = rssm.deterministic_state_fwd(h_init, s_init, a_dummy)
        post_m, _ = rssm.state_posterior(h_goal, e)
        s_goal = post_m
    return h_goal, s_goal


def _pca_xy(X: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """First two principal components of X (N, D), centered."""
    if X.shape[0] < 3:
        return np.zeros((X.shape[0], 2), dtype=np.float32)
    Xf = X.astype(np.float64)
    Xc = Xf - Xf.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    proj = Xc @ vt[:2].T
    return proj.astype(np.float32)


def _r2_xy_linear(X: np.ndarray, pos: np.ndarray) -> Tuple[float, float]:
    """R² for each coordinate of pos (N,2) predicted by linear model of X (+ bias)."""
    n, d = X.shape
    if n < d + 2:
        return float("nan"), float("nan")
    A = np.concatenate([np.ones((n, 1), dtype=np.float64), X.astype(np.float64)], axis=1)
    out = []
    for k in range(2):
        coef, _, _, _ = np.linalg.lstsq(A, pos[:, k].astype(np.float64), rcond=None)
        pred = A @ coef
        ss_res = float(np.sum((pos[:, k] - pred) ** 2))
        ss_tot = float(np.sum((pos[:, k] - np.mean(pos[:, k])) ** 2)) + 1e-12
        out.append(1.0 - ss_res / ss_tot)
    return float(out[0]), float(out[1])


def _grid_rgba(grid: List[List[str]]) -> np.ndarray:
    h, w = len(grid), len(grid[0])
    img = np.ones((h, w, 4), dtype=np.float32)
    img[:, :, :3] = 0.96
    img[:, :, 3] = 1.0
    for r in range(h):
        for c in range(w):
            if grid[r][c] == "1":
                img[r, c, :3] = 0.18
    return img


@torch.no_grad()
def rollout_collect_latents(
    env,
    encoder: nn.Module,
    rssm: nn.Module,
    actor: nn.Module,
    device: torch.device,
    bit_depth: int,
    num_episodes: int = 12,
    deterministic: bool = True,
    action_repeat: int = 1,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Run the actor in the environment and record (x,y), h, s at each step.
    """
    encoder.eval()
    rssm.eval()
    actor.eval()
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    pos_list, h_list, s_list = [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        obs_t = (
            torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        preprocess_img(obs_t, depth=bit_depth)
        e = encoder(obs_t)
        h, s = rssm.get_init_state(e)
        if hasattr(env, "agent_pos"):
            pos_list.append(np.asarray(env.agent_pos, dtype=np.float32).copy())
            h_list.append(h.squeeze(0).cpu().numpy().copy())
            s_list.append(s.squeeze(0).cpu().numpy().copy())

        while not done:
            action, _ = actor.get_action(h, s, deterministic=deterministic)
            a_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            obs, _, term, trunc, _ = env.step(a_np, repeat=action_repeat)
            done = bool(term or trunc)

            obs_t = (
                torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            preprocess_img(obs_t, depth=bit_depth)
            e = encoder(obs_t)
            act_t = torch.tensor(a_np, dtype=torch.float32, device=device).unsqueeze(0)
            h, s, _, _ = rssm.observe_step(e, act_t, h, s, sample=False)
            if hasattr(env, "agent_pos"):
                pos_list.append(np.asarray(env.agent_pos, dtype=np.float32).copy())
                h_list.append(h.squeeze(0).cpu().numpy().copy())
                s_list.append(s.squeeze(0).cpu().numpy().copy())

    pos = np.asarray(pos_list, dtype=np.float32)
    h_arr = np.asarray(h_list, dtype=np.float32)
    s_arr = np.asarray(s_list, dtype=np.float32)
    return {"pos": pos, "h": h_arr, "s": s_arr}


def figure_maze_hs_pca(
    data: Dict[str, np.ndarray],
    grid: List[List[str]],
    out_path: Optional[str] = None,
    title: str = "Latent vs maze position",
    max_scatter: int = 4000,
    seed: int = 0,
):
    """
    Three panels: maze positions (colored by geodesic distance rank) and PCA of h / s with the same colors.
    """
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    pos, h, s = data["pos"], data["h"], data["s"]
    n = min(len(pos), len(h), len(s))
    pos, h, s = pos[:n], h[:n], s[:n]

    d_geo_arr = data.get("d_goal_geo")
    if d_geo_arr is not None:
        d_geo_arr = np.asarray(d_geo_arr, dtype=np.float32)[:n]
        colors = d_geo_arr.copy()
        cbar_label = "Geodesic dist. to goal"
    else:
        colors = None

    if n > max_scatter:
        idx = rng.choice(n, size=max_scatter, replace=False)
        pos, h, s = pos[idx], h[idx], s[idx]
        if colors is not None:
            colors = colors[idx]

    if colors is None:
        # fallback: Euclidean to approximate
        gh, gw = len(grid), len(grid[0])
        goal_xy = np.array([(gw - 1) * 0.5, (gh - 1) * 0.5], dtype=np.float32)
        colors = np.linalg.norm(pos - goal_xy, axis=1)
        cbar_label = "Euclidean dist. to maze center (proxy)"

    ph = _pca_xy(h, rng)
    ps = _pca_xy(s, rng)
    r2_h = _r2_xy_linear(h, pos)
    r2_s = _r2_xy_linear(s, pos)
    r2_hs = _r2_xy_linear(np.concatenate([h, s], axis=1), pos)

    H, W = len(grid), len(grid[0])
    rgba = _grid_rgba(grid)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    for ax, xy, name, r2 in zip(
        axes,
        [pos, ph, ps],
        ["Maze (x, y)", "PCA of h", "PCA of s"],
        [None, r2_h, r2_s],
    ):
        if name.startswith("Maze"):
            ax.imshow(rgba, origin="upper", extent=(0, W, H, 0), aspect="equal", zorder=0)
            sc = ax.scatter(
                pos[:, 0],
                pos[:, 1],
                c=colors,
                s=6,
                alpha=0.85,
                cmap="viridis",
                linewidths=0,
                zorder=2,
            )
            ax.set_xlim(0, W)
            ax.set_ylim(H, 0)
        else:
            sc = ax.scatter(xy[:, 0], xy[:, 1], c=colors, s=6, alpha=0.85, cmap="viridis", linewidths=0)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{name}\n$R^2_x={r2[0]:.2f}$, $R^2_y={r2[1]:.2f}$", fontsize=10)
        if name.startswith("Maze"):
            ax.set_title(f"{name}\n(linear $R^2$ h: {r2_h[0]:.2f}/{r2_h[1]:.2f}, s: {r2_s[0]:.2f}/{r2_s[1]:.2f})", fontsize=9)
            ax.set_xlabel("x (cell)")
            ax.set_ylabel("y (cell)")

    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.72, pad=0.02)
    cbar.set_label(cbar_label)

    note = (
        f"Concat(h,s) linear $R^2$: x={r2_hs[0]:.2f}, y={r2_hs[1]:.2f} — "
        "higher ⇒ easier linear decode of position from latent."
    )
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, style="italic")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
    return fig


def figure_maze_hs_pca_with_geodesic_color(
    data: Dict[str, np.ndarray],
    env,
    out_path: Optional[str] = None,
    title: str = "Maze and latent PCA (colored by geodesic distance to goal)",
    max_scatter: int = 4000,
    seed: int = 0,
):
    """Same as figure_maze_hs_pca but colors points by true geodesic distance to goal."""
    grid = env.grid
    pos = data["pos"]
    n0 = len(pos)
    d_geo = np.zeros(n0, dtype=np.float32)
    if hasattr(env, "geodesic") and hasattr(env, "goal_pos"):
        gpos = env.goal_pos
        for i in range(n0):
            d_geo[i] = env.geodesic.distance(pos[i], gpos)
    enriched = {**data, "d_goal_geo": d_geo}
    return figure_maze_hs_pca(enriched, grid, out_path=out_path, title=title, max_scatter=max_scatter, seed=seed)


@torch.no_grad()
def figure_geo_vs_geodesic(
    env,
    encoder: nn.Module,
    rssm: nn.Module,
    geo: nn.Module,
    data: Dict[str, np.ndarray],
    device: torch.device,
    bit_depth: int,
    out_path: Optional[str] = None,
    title: str = "GeoEncoder aligns with maze geodesic",
    max_points: int = 6000,
    seed: int = 0,
):
    """
    Left: scatter of geodesic distance vs chord distance ||g - g_goal|| (both to goal).
    Right: 2D PCA of g colored by geodesic distance (goal in red).
    """
    import matplotlib.pyplot as plt

    if not hasattr(env, "geodesic") or not hasattr(env, "goal_pos"):
        raise ValueError("env needs geodesic and goal_pos")

    rng = np.random.default_rng(seed)
    pos, h, s = data["pos"], data["h"], data["s"]
    n = min(len(pos), len(h), len(s))
    pos, h, s = pos[:n], h[:n], s[:n]
    if n > max_points:
        sel = rng.choice(n, size=max_points, replace=False)
        pos, h, s = pos[sel], h[sel], s[sel]

    gpos = env.goal_pos
    d_maze = np.array([env.geodesic.distance(p, gpos) for p in pos], dtype=np.float32)

    ht = torch.tensor(h, device=device)
    st = torch.tensor(s, device=device)
    g = geo(ht, st).cpu().numpy()

    gl = _get_goal_latent(env, encoder, rssm, device, bit_depth)
    if gl is None:
        raise RuntimeError("Could not encode goal observation")
    h_g, s_g = gl
    g_goal_row = geo(h_g, s_g).cpu().numpy().reshape(-1)
    d_lat = np.linalg.norm(g - g_goal_row[np.newaxis, :], axis=1)

    g_c = g - g.mean(0, keepdims=True)
    _, _, vt = np.linalg.svd(g_c, full_matrices=False)
    basis = vt[:2].T
    p2 = g_c @ basis
    g_goal_c = g_goal_row - g.mean(0)
    p_goal = g_goal_c @ basis

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    fig.suptitle(title, fontsize=13)

    ax0.scatter(d_maze, d_lat, c=d_maze, s=8, alpha=0.5, cmap="magma")
    ax0.set_xlabel("Geodesic distance to goal (maze)")
    ax0.set_ylabel(r"$\|g(z) - g(z_{\mathrm{goal}})\|$ (GeoEncoder)")
    lim = max(float(d_maze.max()), float(d_lat.max()), 0.1)
    ax0.plot([0, lim], [0, lim], "k--", alpha=0.35, label="y = x (scale-agnostic)")
    ax0.legend(fontsize=8)
    ax0.set_aspect("auto")

    sc = ax1.scatter(p2[:, 0], p2[:, 1], c=d_maze, s=10, alpha=0.75, cmap="viridis")
    ax1.scatter([p_goal[0]], [p_goal[1]], c="red", s=120, marker="*", zorder=5, label=r"$g_{\mathrm{goal}}$")
    ax1.set_xlabel("PCA 1 of $g(h,s)$")
    ax1.set_ylabel("PCA 2 of $g(h,s)$")
    ax1.legend(loc="upper right")
    ax1.set_aspect("equal", adjustable="box")
    fig.colorbar(sc, ax=ax1, shrink=0.8, label="Geodesic to goal")

    with np.errstate(invalid="ignore"):
        rho = float(np.corrcoef(d_maze, d_lat)[0, 1])
    if not np.isfinite(rho):
        rho = float("nan")
    fig.text(0.5, 0.02, f"Pearson ρ(d_maze, d_lat) = {rho:.3f}", ha="center", fontsize=10)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
    return fig


@torch.no_grad()
def figure_imagination_geo_guidance(
    env,
    encoder: nn.Module,
    rssm: nn.Module,
    actor: nn.Module,
    geo: Optional[nn.Module],
    device: torch.device,
    bit_depth: int,
    imagination_horizon: int = 15,
    num_starts: int = 4,
    geo_plan_weight: float = 0.15,
    out_path: Optional[str] = None,
    title: str = "Imagined rollouts: GeoEncoder distance shapes actor targets",
    seed: int = 0,
):
    """
    For several start states: prior imagination (as in Dreamer), plot
    latent chord distance to goal vs time, and the implied plan_only penalty
    increment (proportional to d_t along the imagined trajectory).
    """
    import matplotlib.pyplot as plt

    rssm.eval()
    actor.eval()
    encoder.eval()
    if geo is not None:
        geo.eval()

    gl = _get_goal_latent(env, encoder, rssm, device, bit_depth)
    if gl is None or geo is None:
        raise ValueError("Need goal encoding and GeoEncoder for this figure")

    h_g, s_g = gl
    g_goal = geo(h_g, s_g)

    traces_d: List[np.ndarray] = []
    traces_penalty: List[np.ndarray] = []

    for i in range(num_starts):
        try:
            obs, _ = env.reset(seed=seed + i)
        except TypeError:
            obs, _ = env.reset()
        obs_t = (
            torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        preprocess_img(obs_t, depth=bit_depth)
        e = encoder(obs_t)
        h, s = rssm.get_init_state(e)

        d_path = []
        h_i, s_i = h, s
        for _t in range(imagination_horizon + 1):
            g_i = geo(h_i, s_i)
            d = torch.norm(g_i - g_goal, dim=-1).item()
            d_path.append(d)
            if _t == imagination_horizon:
                break
            a_i, _ = actor.get_action(h_i, s_i, deterministic=True)
            h_i, s_i = rssm.imagine_step(h_i, s_i, a_i)

        d_arr = np.array(d_path, dtype=np.float32)
        traces_d.append(d_arr)
        # plan_only: imagined reward has -w * d_t for t>=1 (aligned with dreamer_geo_old)
        pen = -geo_plan_weight * d_arr[1:]
        traces_penalty.append(pen)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    fig.suptitle(title, fontsize=12)

    t = np.arange(imagination_horizon + 1)
    for i, d_arr in enumerate(traces_d):
        ax0.plot(t, d_arr, alpha=0.85, marker="o", markersize=3, label=f"start {i+1}")
    ax0.set_xlabel("Imagined step")
    ax0.set_ylabel(r"$\|g - g_{\mathrm{goal}}\|$")
    ax0.legend(fontsize=7, ncol=2, loc="upper right")
    ax0.set_title("Distance along imagined trajectory")

    tp = np.arange(1, imagination_horizon + 1)
    for i, pen in enumerate(traces_penalty):
        ax1.plot(tp, pen, marker="s", markersize=4, alpha=0.85, label=f"start {i+1}")
    ax1.set_xlabel("Imagined step (reward index)")
    ax1.set_ylabel("Geo penalty term (plan_only)")
    ax1.set_title(rf"Contribution $-\lambda\, d_t$ to imagined reward ($\lambda$={geo_plan_weight})")
    ax1.legend(fontsize=7, ncol=2)
    ax1.axhline(0.0, color="k", lw=0.5, alpha=0.4)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
    return fig


def save_all_maze_geo_figures(
    env,
    encoder: nn.Module,
    rssm: nn.Module,
    actor: nn.Module,
    geo: Optional[nn.Module],
    device: torch.device,
    bit_depth: int,
    out_dir: str,
    env_id: str = "maze",
    rollout_episodes: int = 16,
    action_repeat: int = 1,
    imagination_horizon: int = 15,
    geo_plan_weight: float = 0.15,
    seed: int = 0,
) -> Dict[str, str]:
    """
    Collect latents and write three figures into out_dir. Returns paths.
    """
    if action_repeat <= 0 and hasattr(env, "unwrapped"):
        pass
    ar = action_repeat

    data = rollout_collect_latents(
        env,
        encoder,
        rssm,
        actor,
        device,
        bit_depth,
        num_episodes=rollout_episodes,
        deterministic=True,
        action_repeat=ar,
        seed=seed,
    )
    paths = {}
    p0 = os.path.join(out_dir, f"{env_id}_maze_hs_pca.png")
    figure_maze_hs_pca_with_geodesic_color(
        data, env, out_path=p0, title=f"{env_id}: maze vs PCA(h), PCA(s) (color = geodesic to goal)"
    )
    paths["maze_hs_pca"] = p0

    if geo is not None:
        p1 = os.path.join(out_dir, f"{env_id}_geo_vs_geodesic.png")
        figure_geo_vs_geodesic(
            env, encoder, rssm, geo, data, device, bit_depth, out_path=p1, seed=seed
        )
        paths["geo_vs_geodesic"] = p1

        p2 = os.path.join(out_dir, f"{env_id}_imagination_geo_guidance.png")
        figure_imagination_geo_guidance(
            env,
            encoder,
            rssm,
            actor,
            geo,
            device,
            bit_depth,
            imagination_horizon=imagination_horizon,
            geo_plan_weight=geo_plan_weight,
            out_path=p2,
            seed=seed + 1,
        )
        paths["imagination_geo"] = p2

    return paths


def save_maze_dreamer_checkpoint(
    path: str,
    encoder: nn.Module,
    rssm: nn.Module,
    actor: nn.Module,
    value_model: nn.Module,
    reward_model: nn.Module,
    cont_model: nn.Module,
    decoder: nn.Module,
    geo: Optional[nn.Module],
    meta: Dict[str, Any],
) -> None:
    """Save weights + small JSON meta (hyperparameters, env_id, etc.)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "encoder": encoder.state_dict(),
        "rssm": rssm.state_dict(),
        "actor": actor.state_dict(),
        "value_model": value_model.state_dict(),
        "reward_model": reward_model.state_dict(),
        "cont_model": cont_model.state_dict(),
        "decoder": decoder.state_dict(),
    }
    if geo is not None:
        payload["geo"] = geo.state_dict()
    torch.save(payload, path)
    meta_path = path.replace(".pt", "_meta.json").replace(".pth", "_meta.json")
    if meta_path == path:
        meta_path = path + "_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_maze_dreamer_checkpoint(
    path: str,
    device: torch.device,
    meta: Dict[str, Any],
    observation_shape: Tuple[int, int, int],
    act_dim: int,
) -> Dict[str, nn.Module]:
    """
    Rebuild modules from meta (must contain keys matching dreamer_geo_old.parse_args fields:
    embed_dim, stoch_dim, deter_dim, hidden_dim, actor_hidden_dim, value_hidden_dim,
    geom_hidden_dim, geo_dim).
    """
    from models import Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel, RSSM, ValueModel
    from geom_head import GeoEncoder

    H, W, C = observation_shape
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location=device)

    enc = ConvEncoder(embedding_size=meta["embed_dim"], in_channels=C).to(device)
    dec = ConvDecoder(
        meta["deter_dim"], meta["stoch_dim"], embedding_size=meta["embed_dim"], out_channels=C
    ).to(device)
    rssm = RSSM(
        meta["stoch_dim"],
        meta["deter_dim"],
        act_dim,
        meta["embed_dim"],
        meta["hidden_dim"],
    ).to(device)
    rew = RewardModel(meta["deter_dim"], meta["stoch_dim"], meta["hidden_dim"]).to(device)
    cont = ContinueModel(meta["deter_dim"], meta["stoch_dim"], meta["hidden_dim"]).to(device)
    actor = Actor(meta["deter_dim"], meta["stoch_dim"], act_dim, meta["actor_hidden_dim"]).to(device)
    val = ValueModel(meta["deter_dim"], meta["stoch_dim"], meta["value_hidden_dim"]).to(device)

    enc.load_state_dict(ckpt["encoder"])
    dec.load_state_dict(ckpt["decoder"])
    rssm.load_state_dict(ckpt["rssm"])
    rew.load_state_dict(ckpt["reward_model"])
    cont.load_state_dict(ckpt["cont_model"])
    actor.load_state_dict(ckpt["actor"])
    val.load_state_dict(ckpt["value_model"])

    geo = None
    if "geo" in ckpt and ckpt["geo"] is not None:
        geo = GeoEncoder(
            meta["deter_dim"],
            meta["stoch_dim"],
            geo_dim=meta.get("geo_dim", 32),
            hidden_dim=meta.get("geom_hidden_dim", 256),
        ).to(device)
        geo.load_state_dict(ckpt["geo"])

    out = {
        "encoder": enc,
        "decoder": dec,
        "rssm": rssm,
        "reward_model": rew,
        "cont_model": cont,
        "actor": actor,
        "value_model": val,
        "geo": geo,
    }
    for m in out.values():
        if m is not None:
            m.eval()
    return out


def visualize_from_checkpoint_example():
    """Example CLI-style usage (run as script with a .pt path)."""
    import argparse

    p = argparse.ArgumentParser(description="Load maze Dreamer checkpoint and save figures")
    p.add_argument("checkpoint", type=str)
    p.add_argument("--meta_json", type=str, default=None, help="Path to *_meta.json (default: infer)")
    p.add_argument("--out_dir", type=str, default="maze_viz_out")
    p.add_argument("--env_id", type=str, default="custom_maze:two_room")
    p.add_argument("--img_size", type=int, default=128)
    args = p.parse_args()

    meta_path = args.meta_json
    if meta_path is None:
        cand = args.checkpoint.replace(".pt", "_meta.json")
        meta_path = cand if os.path.isfile(cand) else args.checkpoint + "_meta.json"
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bit_depth = int(meta.get("bit_depth", 5))
    _crop = args.img_size // 2
    _ego = (_crop, _crop) if _crop >= 48 else None
    env = make_env(args.env_id, img_size=(args.img_size, args.img_size), num_stack=1, egocentric_crop_size=_ego)
    obs, _ = env.reset()
    H, W, C = obs.shape
    act_dim = env.action_space.shape[0]
    ar = int(meta.get("action_repeat", 0))
    if ar <= 0:
        ar = ENV_ACTION_REPEAT.get(args.env_id, 2)

    nets = load_maze_dreamer_checkpoint(
        args.checkpoint, device, meta, observation_shape=(H, W, C), act_dim=act_dim
    )
    paths = save_all_maze_geo_figures(
        env,
        nets["encoder"],
        nets["rssm"],
        nets["actor"],
        nets["geo"],
        device,
        bit_depth,
        args.out_dir,
        env_id=args.env_id.replace(":", "_"),
        action_repeat=ar,
        geo_plan_weight=float(meta.get("geo_plan_weight", 0.15)),
    )
    print("Wrote:", paths)
    env.close()


if __name__ == "__main__":
    visualize_from_checkpoint_example()
