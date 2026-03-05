#!/usr/bin/env python3
"""
Dreamer with geodesic-trained g_geo (GeodesicComputer + train_geo_encoder_geodesic).
No dyn_pb_reg_weight / dyn_pb_bisim_weight.

Three variants (run as multiple variants in one script):
  1. plan_only: Train Dreamer, freeze WM, train g_geo on replay; use g_geo only at
     planning time: minimize ||g_geo(z_t) - g_geo(z_goal)|| (goal from env).
  2. shaping: Same g_geo; reward shaping r_t' = r_t + alpha*(d_geo(t-1) - d_geo(t)).
  3. aux_backprop: Warm up WM, train g_geo, then unfreeze encoder+RSSM and add
     small auxiliary loss from g_geo (representation shaper).
"""

import os
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

torch.set_float32_matmul_precision("high")

from models import (
    RSSM,
    Actor,
    ContinueModel,
    ConvDecoder,
    ConvEncoder,
    RewardModel,
    ValueModel,
)
from utils import (
    ENV_ACTION_REPEAT,
    ReplayBuffer,
    bottle,
    get_device,
    make_env,
    no_param_grads,
    preprocess_img,
    set_seed,
)
from geom_head import GeoEncoder


# ===============================
#  Geodesic-supervised g_geo training (from maze_geometry_test)
# ===============================


def _positions_to_cell_indices(geodesic, pos: np.ndarray) -> np.ndarray:
    """Map (x,y) positions to free-cell indices for geodesic.dist_matrix."""
    cells = [geodesic.pos_to_cell(float(p[0]), float(p[1])) for p in pos]
    return np.asarray([geodesic.cell_to_idx[c] for c in cells], dtype=np.int64)


def _uniformity_loss_sphere(g: torch.Tensor, t: float = 2.0, max_uni: int = 256) -> torch.Tensor:
    """Wang & Isola-style uniformity on sphere."""
    g_flat = g.reshape(-1, g.shape[-1])
    n = g_flat.size(0)
    if n <= 2:
        return g.new_zeros(())
    if n > max_uni:
        idx = torch.randperm(n, device=g.device)[:max_uni]
        g_flat = g_flat[idx]
    sq = torch.cdist(g_flat, g_flat).pow(2)
    mask = torch.eye(sq.size(0), device=g.device, dtype=torch.bool)
    sq = sq.masked_fill(mask, 1e9)
    return torch.logsumexp(-t * sq, dim=1).mean()


def train_geo_encoder_geodesic(
    data: dict,
    deter_dim: int,
    stoch_dim: int,
    geo_dim: int,
    geo_hidden: int,
    geo_lr: float,
    geo_sup_epochs: int,
    geo_sup_batch: int,
    geo_sup_stress_eps: float,
    geo_sup_stress_weight: float,
    geo_sup_uniformity_weight: float,
    geo_sup_uniformity_t: float,
    device: torch.device,
    geodesic,
) -> GeoEncoder:
    """Train GeoEncoder so ||g_i - g_j|| matches geodesic distance (pairwise stress + uniformity)."""
    geo = GeoEncoder(deter_dim, stoch_dim, geo_dim=geo_dim, hidden_dim=geo_hidden).to(device)
    scale = torch.nn.Parameter(torch.tensor(0.1, device=device))
    opt = torch.optim.Adam(list(geo.parameters()) + [scale], lr=geo_lr)

    pos = data["pos"]
    h = data["h"]
    s = data["s"]
    N = len(pos)
    if N < 512:
        geo.eval()
        return geo

    cell_idx = _positions_to_cell_indices(geodesic, pos)
    dist_mat = geodesic.dist_matrix

    for epoch in range(geo_sup_epochs):
        B = min(geo_sup_batch, N)
        ii = np.random.randint(0, N, size=B)
        jj = np.random.randint(0, N, size=B)
        same = ii == jj
        jj[same] = (jj[same] + 1) % N
        jj = np.clip(jj, 0, N - 1)

        c_i = cell_idx[ii]
        c_j = cell_idx[jj]
        d_geo_np = dist_mat[c_i, c_j].astype(np.float32)
        valid = np.isfinite(d_geo_np) & (d_geo_np > 0)
        if valid.sum() < 4:
            continue
        ii, jj, d_geo_np = ii[valid], jj[valid], d_geo_np[valid]
        B = len(ii)

        d_geo_t = torch.tensor(d_geo_np, device=device, dtype=torch.float32)
        weight = 1.0 / (d_geo_t + geo_sup_stress_eps)

        h_i = torch.tensor(h[ii], dtype=torch.float32, device=device)
        s_i = torch.tensor(s[ii], dtype=torch.float32, device=device)
        h_j = torch.tensor(h[jj], dtype=torch.float32, device=device)
        s_j = torch.tensor(s[jj], dtype=torch.float32, device=device)

        g_i = geo(h_i, s_i)
        g_j = geo(h_j, s_j)
        d_lat = torch.norm(g_i - g_j, dim=-1)

        scale_d_geo = scale.clamp(min=1e-4) * d_geo_t
        stress = (weight * (d_lat - scale_d_geo).pow(2)).mean()
        loss_uni = _uniformity_loss_sphere(torch.stack([g_i, g_j], 1), t=geo_sup_uniformity_t)
        loss = geo_sup_stress_weight * stress + geo_sup_uniformity_weight * loss_uni

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (epoch + 1) % max(1, geo_sup_epochs // 3) == 0:
            print(
                f"    GeoSup epoch {epoch+1}/{geo_sup_epochs}  "
                f"stress={stress.item():.4f}  scale={scale.item():.4f}  total={loss.item():.4f}"
            )

    geo.eval()
    return geo


# ===============================
#  Helpers
# ===============================


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    B, H = rewards.shape
    next_values = values[:, 1:]
    last = values[:, -1]
    out = torch.zeros_like(rewards)
    for t in reversed(range(H)):
        bootstrap = (1.0 - lambda_) * next_values[:, t] + lambda_ * last
        last = rewards[:, t] + discounts[:, t] * bootstrap
        out[:, t] = last
    return out


def compute_discount_weights(discounts: torch.Tensor) -> torch.Tensor:
    B, H = discounts.shape
    ones = torch.ones((B, 1), device=discounts.device, dtype=discounts.dtype)
    return torch.cumprod(torch.cat([ones, discounts], dim=1), dim=1)[:, :-1]


def get_goal_latent(
    env,
    encoder: nn.Module,
    rssm: nn.Module,
    device: torch.device,
    bit_depth: int,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Encode goal observation -> (h_goal, s_goal). Returns None if env has no goal obs."""
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
        h, s = rssm.get_init_state(e)
    return h, s


# ===============================
#  Evaluation
# ===============================


@torch.no_grad()
def evaluate_actor_policy(
    env,
    img_size: int,
    encoder: nn.Module,
    rssm: nn.Module,
    actor: nn.Module,
    episodes: int,
    seed: int,
    device: torch.device,
    bit_depth: int,
    action_repeat: int,
) -> Tuple[float, float]:
    returns: List[float] = []
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    encoder.eval()
    rssm.eval()
    actor.eval()

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        obs_t = (
            torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        preprocess_img(obs_t, depth=bit_depth)
        e = encoder(obs_t)
        h, s = rssm.get_init_state(e)

        while not done:
            action, _ = actor.get_action(h, s, deterministic=True)
            a_np = action.squeeze(0).cpu().numpy().astype(np.float32)

            obs, total_reward, term, trunc, _ = env.step(a_np, repeat=action_repeat)
            done = bool(term or trunc)
            ep_ret += float(total_reward)

            obs_t = (
                torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            preprocess_img(obs_t, depth=bit_depth)
            e = encoder(obs_t)
            h, s, _, _ = rssm.observe_step(e, action, h, s, sample=False)

        returns.append(ep_ret)

    return float(np.mean(returns)), float(np.std(returns))


# ===============================
#  Training
# ===============================


@dataclass
class VariantCfg:
    name: str
    # "plan_only" | "shaping" | "aux_backprop"
    geo_variant: str = "plan_only"
    geo_plan_weight: float = 0.1
    geo_shaping_alpha: float = 0.3
    geo_aux_weight: float = 0.05


def run_one_seed(args, cfg: VariantCfg, seed: int) -> Dict[str, float]:
    set_seed(seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    env = make_env(args.env_id, img_size=(args.img_size, args.img_size), num_stack=1)
    obs, _ = env.reset()
    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    action_repeat = (
        args.action_repeat if args.action_repeat > 0 else ENV_ACTION_REPEAT.get(args.env_id, 2)
    )
    effective_gamma = args.gamma ** action_repeat

    # Maze: geodesic + goal for geo variants
    geodesic = getattr(env, "geodesic", None)
    is_maze = geodesic is not None and hasattr(env, "agent_pos")

    # Models
    encoder = ConvEncoder(embedding_size=args.embed_dim, in_channels=C).to(device)
    decoder = ConvDecoder(
        args.deter_dim, args.stoch_dim, embedding_size=args.embed_dim, out_channels=C
    ).to(device)
    rssm = RSSM(
        args.stoch_dim, args.deter_dim, act_dim, args.embed_dim, args.hidden_dim
    ).to(device)
    reward_model = RewardModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(device)
    cont_model = ContinueModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(device)
    actor = Actor(args.deter_dim, args.stoch_dim, act_dim, args.actor_hidden_dim).to(device)
    value_model = ValueModel(args.deter_dim, args.stoch_dim, args.value_hidden_dim).to(device)

    geo = None
    if is_maze and cfg.geo_variant in ("plan_only", "shaping", "aux_backprop"):
        geo = GeoEncoder(
            args.deter_dim,
            args.stoch_dim,
            geo_dim=args.geo_dim,
            hidden_dim=args.geom_hidden_dim,
        ).to(device)

    # Optims
    world_params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(rssm.parameters())
        + list(reward_model.parameters())
        + list(cont_model.parameters())
    )
    model_optim = torch.optim.Adam(world_params, lr=args.model_lr, eps=args.adam_eps)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=args.adam_eps)
    value_optim = torch.optim.Adam(
        value_model.parameters(), lr=args.value_lr, eps=args.adam_eps
    )
    replay = ReplayBuffer(args.replay_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    # Geo data: (pos, h, s) for geodesic-supervised g_geo training (maze only)
    geo_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    geo_data_max_points = getattr(args, "geo_data_max_points", 50_000)

    run_name = f"{args.env_id}_{cfg.name}_seed{seed}"
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
    writer.add_text("hyperparameters", str(vars(args)), 0)
    writer.add_text("variant", str(cfg), 0)

    # Phasing: V1/V2: warmup -> freeze WM -> train g_geo once -> plan with g_geo.
    #           V3: warmup -> train g_geo once -> aux_backprop (encoder+rssm only).
    geo_phase = "warmup"
    wm_frozen = False
    geo_trained_once = False

    # Seed buffer
    for _ in range(args.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, total_reward, term, trunc, _ = env.step(action, repeat=action_repeat)
            done = bool(term or trunc)
            replay.add(
                obs=np.ascontiguousarray(obs, np.uint8),
                action=np.asarray(action, np.float32),
                reward=total_reward,
                next_obs=None,
                done=done,
            )
            obs = next_obs

    total_steps = 0
    expl = args.expl_amount
    t_start = time.time()
    train_updates = 0

    for ep in range(args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0
        traj_pos, traj_h, traj_s = [], [], []

        obs_t = (
            torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        preprocess_img(obs_t, depth=args.bit_depth)
        with torch.no_grad():
            e0 = encoder(obs_t)
            h, s = rssm.get_init_state(e0)

        while not done:
            encoder.eval()
            rssm.eval()
            actor.eval()
            with torch.no_grad():
                a_t, _ = actor.get_action(h, s, deterministic=False)
                if expl > 0:
                    a_t = torch.clamp(a_t + expl * torch.randn_like(a_t), -1.0, 1.0)
                a_np = a_t.squeeze(0).cpu().numpy().astype(np.float32)

            next_obs, total_reward, term, trunc, _ = env.step(a_np, repeat=action_repeat)
            done = bool(term or trunc)
            ep_return += float(total_reward)
            ep_steps += 1
            replay.add(
                obs=np.ascontiguousarray(obs, np.uint8),
                action=a_np,
                reward=total_reward,
                next_obs=None,
                done=done,
            )
            obs = next_obs
            total_steps += 1

            if is_maze and geo is not None:
                pos = env.agent_pos
                traj_pos.append(pos.copy())
                traj_h.append(h.squeeze(0).cpu().numpy().copy())
                traj_s.append(s.squeeze(0).cpu().numpy().copy())

            obs_t = (
                torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            preprocess_img(obs_t, depth=args.bit_depth)
            with torch.no_grad():
                e = encoder(obs_t)
                act_t = torch.tensor(a_np, dtype=torch.float32, device=device).unsqueeze(0)
                h, s, _, _ = rssm.observe_step(e, act_t, h, s, sample=False)

            if total_steps % args.collect_interval == 0 and replay.size > (args.seq_len + 2):
                encoder.train()
                decoder.train()
                rssm.train()
                reward_model.train()
                cont_model.train()
                actor.train()
                value_model.train()
                if geo is not None:
                    geo.train()

                for _ in range(args.train_steps):
                    batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)
                    obs_seq = torch.as_tensor(batch.obs, device=device).float()
                    act_seq = torch.as_tensor(batch.actions, device=device)
                    rew_seq = torch.as_tensor(batch.rews, device=device)
                    done_seq = torch.as_tensor(batch.dones, device=device)

                    B, T1 = rew_seq.shape
                    T = T1 - 1

                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                    preprocess_img(x, depth=args.bit_depth)

                    e_t = bottle(encoder, x)
                    h_t, s_t = rssm.get_init_state(e_t[:, 0])
                    h_list, s_list = [], []
                    pri_list, post_list = [], []

                    for t in range(T):
                        h_t = rssm.deterministic_state_fwd(h_t, s_t, act_seq[:, t])
                        pri_list.append(rssm.state_prior(h_t))
                        post_list.append(rssm.state_posterior(h_t, e_t[:, t + 1]))
                        pm, ps = post_list[-1]
                        s_t = pm + torch.randn_like(ps) * ps
                        h_list.append(h_t)
                        s_list.append(s_t)

                    h_seq = torch.stack(h_list, dim=1)
                    s_seq = torch.stack(s_list, dim=1)
                    pri_m = torch.stack([p[0] for p in pri_list], dim=0)
                    pri_s = torch.stack([p[1] for p in pri_list], dim=0)
                    pos_m = torch.stack([p[0] for p in post_list], dim=0)
                    pos_s = torch.stack([p[1] for p in post_list], dim=0)

                    kl_lhs = kl_divergence(
                        Normal(pos_m.detach(), pos_s.detach()), Normal(pri_m, pri_s)
                    )
                    kl_rhs = kl_divergence(
                        Normal(pos_m, pos_s), Normal(pri_m.detach(), pri_s.detach())
                    )
                    kld = 0.5 * (kl_lhs + kl_rhs)
                    kld = torch.max(kld.sum(-1), free_nats).mean()

                    recon = bottle(decoder, h_seq, s_seq)
                    rec_loss = (
                        F.mse_loss(recon, x[:, 1 : T + 1], reduction="none")
                        .sum((2, 3, 4))
                        .mean()
                    )
                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_loss = F.mse_loss(rew_pred, rew_seq[:, :T])
                    cont_logits = bottle(cont_model, h_seq, s_seq)
                    cont_target = (1.0 - done_seq[:, :T]).clamp(0.0, 1.0)
                    cont_loss = F.binary_cross_entropy_with_logits(cont_logits, cont_target)

                    geo_aux_loss = torch.zeros((), device=device)
                    if (
                        geo is not None
                        and cfg.geo_variant == "aux_backprop"
                        and geo_phase == "aux_backprop"
                    ):
                        g_real = bottle(geo, h_seq, s_seq)
                        g_flat = g_real.reshape(-1, g_real.size(-1))
                        if g_flat.size(0) > 1:
                            d_lat = torch.cdist(g_flat, g_flat)
                            mask = torch.eye(d_lat.size(0), device=device, dtype=torch.bool)
                            d_lat = d_lat.masked_fill(mask, 1e9)
                            geo_aux_loss = -d_lat.min(dim=1).values.mean()

                    model_loss = (
                        rec_loss
                        + args.kl_weight * kld
                        + rew_loss
                        + args.cont_weight * cont_loss
                        + (
                            cfg.geo_aux_weight * geo_aux_loss
                            if cfg.geo_variant == "aux_backprop" and geo_phase == "aux_backprop"
                            else 0.0
                        )
                    )

                    model_optim.zero_grad(set_to_none=True)
                    model_loss.backward()

                    if cfg.geo_variant == "aux_backprop" and geo_phase == "aux_backprop":
                        for p in (
                            list(decoder.parameters())
                            + list(reward_model.parameters())
                            + list(cont_model.parameters())
                        ):
                            if p.grad is not None:
                                p.grad.zero_()

                    if not wm_frozen:
                        torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip_norm)
                        model_optim.step()

                    # ---------- Train g_geo once after warmup (geodesic-supervised) ----------
                    if (
                        geo is not None
                        and is_maze
                        and not geo_trained_once
                        and total_steps >= args.geo_warmup_steps
                    ):
                        total_pts = sum(len(p) for p, _, _ in geo_data)
                        if total_pts >= 512:
                            if cfg.geo_variant in ("plan_only", "shaping"):
                                wm_frozen = True
                            geo_phase = "geo_train"
                            all_pos = np.concatenate([p for p, _, _ in geo_data], axis=0)
                            all_h = np.concatenate([_h for _, _h, _ in geo_data], axis=0)
                            all_s = np.concatenate([_s for _, _, _s in geo_data], axis=0)
                            if len(all_pos) > geo_data_max_points:
                                idx = np.random.choice(
                                    len(all_pos), geo_data_max_points, replace=False
                                )
                                all_pos = all_pos[idx]
                                all_h = all_h[idx]
                                all_s = all_s[idx]
                            data = {"pos": all_pos, "h": all_h, "s": all_s}
                            geo = train_geo_encoder_geodesic(
                                data,
                                args.deter_dim,
                                args.stoch_dim,
                                args.geo_dim,
                                args.geom_hidden_dim,
                                args.geo_lr,
                                args.geo_sup_epochs,
                                args.geo_sup_batch,
                                args.geo_sup_stress_eps,
                                args.geo_sup_stress_weight,
                                args.geo_sup_uniformity_weight,
                                args.geo_sup_uniformity_t,
                                device,
                                geodesic,
                            )
                            geo_trained_once = True
                            if cfg.geo_variant == "aux_backprop":
                                geo_phase = "aux_backprop"
                                wm_frozen = False
                            geo_data.clear()

                    # ---------- Actor-Critic (imagination) ----------
                    with torch.no_grad():
                        Dh, Ds = h_seq.size(-1), s_seq.size(-1)
                        if 0 < args.imagination_starts < T:
                            K = args.imagination_starts
                            t_idx = torch.randint(0, T, (B, K), device=device)
                            h0 = h_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Dh)).reshape(-1, Dh).detach()
                            s0 = s_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Ds)).reshape(-1, Ds).detach()
                        else:
                            h0 = h_seq.reshape(-1, Dh).detach()
                            s0 = s_seq.reshape(-1, Ds).detach()

                    with no_param_grads(rssm), no_param_grads(reward_model), no_param_grads(cont_model):
                        h_im, s_im = h0, s0
                        h_im_list = [h0]
                        s_im_list = [s0]
                        for _ in range(args.imagination_horizon):
                            a_im, _ = actor.get_action(h_im, s_im, deterministic=False)
                            h_im = rssm.deterministic_state_fwd(h_im, s_im, a_im)
                            s_im = rssm.state_prior(h_im, sample=True)
                            h_im_list.append(h_im)
                            s_im_list.append(s_im)
                        h_imag = torch.stack(h_im_list, dim=1)
                        s_imag = torch.stack(s_im_list, dim=1)

                    rewards_im = bottle(reward_model, h_imag[:, 1:], s_imag[:, 1:])
                    cont_logits_im = bottle(cont_model, h_imag[:, 1:], s_imag[:, 1:])
                    pcont = torch.sigmoid(cont_logits_im).clamp(0.0, 1.0)
                    discounts = effective_gamma * pcont

                    if geo is not None and is_maze:
                        goal_latent = get_goal_latent(env, encoder, rssm, device, args.bit_depth)
                        if goal_latent is not None:
                            h_goal, s_goal = goal_latent
                            with no_param_grads(geo):
                                g_imag = bottle(geo, h_imag, s_imag)
                                g_goal = geo(h_goal.expand(h_imag.size(0), -1), s_goal.expand(s_imag.size(0), -1))
                            d_geo = torch.norm(g_imag - g_goal.unsqueeze(1), dim=-1)
                            d_geo_t = d_geo[:, 1:]
                            if cfg.geo_variant == "plan_only":
                                rewards_im = rewards_im - cfg.geo_plan_weight * d_geo_t
                            elif cfg.geo_variant == "shaping":
                                d_geo_prev = d_geo[:, :-1]
                                progress = d_geo_prev - d_geo_t
                                rewards_im = rewards_im + cfg.geo_shaping_alpha * progress
                            else:
                                rewards_im = rewards_im - cfg.geo_plan_weight * d_geo_t

                    with torch.no_grad():
                        values_tgt = bottle(value_model, h_imag, s_imag)
                        lam_ret_tgt = compute_lambda_returns(
                            rewards_im.detach(), values_tgt, discounts.detach(), lambda_=args.lambda_
                        )
                        w_tgt = compute_discount_weights(discounts.detach())

                    v_pred = bottle(value_model, h_imag.detach(), s_imag.detach())
                    value_loss = ((v_pred[:, :-1] - lam_ret_tgt) ** 2 * w_tgt).mean()
                    value_optim.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm)
                    value_optim.step()

                    mean, std = actor.forward(h_imag[:, :-1].detach(), s_imag[:, :-1].detach())
                    noise = torch.randn_like(mean)
                    raw = mean + std * noise
                    entropy = (
                        Normal(mean, std).entropy()
                        + torch.log(1 - torch.tanh(raw).pow(2) + 1e-6)
                    ).sum(dim=-1).mean()
                    with no_param_grads(value_model):
                        values_for_actor = bottle(value_model, h_imag, s_imag)
                    w_actor = compute_discount_weights(discounts.detach())
                    lam_ret_actor = compute_lambda_returns(
                        rewards_im, values_for_actor, discounts, lambda_=args.lambda_
                    )
                    actor_loss = -(w_actor.detach() * lam_ret_actor).mean() - args.actor_entropy_scale * entropy

                    actor_optim.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip_norm)
                    actor_optim.step()

                    train_updates += 1
                    if args.train_log_interval > 0 and train_updates % args.train_log_interval == 0:
                        writer.add_scalar("loss/model_total", model_loss.item(), total_steps)
                        writer.add_scalar("loss/reconstruction", rec_loss.item(), total_steps)
                        writer.add_scalar("loss/kl_divergence", kld.item(), total_steps)
                        writer.add_scalar("loss/reward", rew_loss.item(), total_steps)
                        writer.add_scalar("loss/continue", cont_loss.item(), total_steps)
                        writer.add_scalar("loss/actor", actor_loss.item(), total_steps)
                        writer.add_scalar("loss/value", value_loss.item(), total_steps)
                        writer.add_scalar("train/exploration", expl, total_steps)

            if args.expl_decay > 0:
                expl = max(args.expl_min, expl - args.expl_decay)

        if is_maze and geo is not None and len(traj_pos) > 0:
            geo_data.append(
                (np.array(traj_pos), np.array(traj_h), np.array(traj_s))
            )
            while sum(len(p) for p, _, _ in geo_data) > geo_data_max_points and len(geo_data) > 1:
                geo_data.pop(0)

        writer.add_scalar("train/episode_return", ep_return, ep)
        writer.add_scalar("train/episode_steps", ep_steps, ep)
        writer.add_scalar("train/total_steps", total_steps, ep)

        if (ep + 1) % max(1, args.max_episodes // 5) == 0:
            dt = time.time() - t_start
            print(
                f"    [{cfg.name} | seed {seed}] ep {ep + 1}/{args.max_episodes}  "
                f"steps={total_steps}  buf={replay.size}  ({dt:.0f}s)"
            )

        if args.eval_interval > 0 and (ep + 1) % args.eval_interval == 0:
            eval_mean, eval_std = evaluate_actor_policy(
                env=env,
                img_size=args.img_size,
                encoder=encoder,
                rssm=rssm,
                actor=actor,
                episodes=args.eval_episodes,
                seed=seed + 10000,
                device=device,
                bit_depth=args.bit_depth,
                action_repeat=action_repeat,
            )
            print(f"    [Eval @ ep {ep + 1}] mean={eval_mean:.2f} std={eval_std:.2f}", flush=True)
            writer.add_scalar("eval/mean_return", eval_mean, ep + 1)
            writer.add_scalar("eval/std_return", eval_std, ep + 1)

    env.close()

    mean_ret, std_ret = evaluate_actor_policy(
        env=make_env(args.env_id, img_size=(args.img_size, args.img_size), num_stack=1),
        img_size=args.img_size,
        encoder=encoder,
        rssm=rssm,
        actor=actor,
        episodes=args.eval_episodes,
        seed=seed + 10_000,
        device=device,
        bit_depth=args.bit_depth,
        action_repeat=action_repeat,
    )
    writer.add_scalar("eval/mean_return", mean_ret, args.max_episodes)
    writer.add_scalar("eval/std_return", std_ret, args.max_episodes)
    writer.close()

    return {
        "seed": float(seed),
        "eval_mean": float(mean_ret),
        "eval_std": float(std_ret),
        "total_steps": float(total_steps),
    }


def parse_args():
    p = argparse.ArgumentParser(description="Dreamer + geodesic g_geo (plan_only / shaping / aux_backprop)")
    p.add_argument("--env_id", type=str, default="custom_maze:corridor")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--quick", action="store_true")

    p.add_argument("--seed_episodes", type=int, default=5)
    p.add_argument("--max_episodes", type=int, default=100)
    p.add_argument("--collect_interval", type=int, default=50)
    p.add_argument("--train_steps", type=int, default=30)
    p.add_argument("--action_repeat", type=int, default=0)
    p.add_argument("--replay_capacity", type=int, default=200_000)

    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=50)

    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--actor_hidden_dim", type=int, default=400)
    p.add_argument("--value_hidden_dim", type=int, default=400)
    p.add_argument("--geom_hidden_dim", type=int, default=256)

    p.add_argument("--model_lr", type=float, default=6e-4)
    p.add_argument("--actor_lr", type=float, default=8e-5)
    p.add_argument("--value_lr", type=float, default=8e-5)
    p.add_argument("--adam_eps", type=float, default=1e-5)
    p.add_argument("--grad_clip_norm", type=float, default=100.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda_", type=float, default=0.95)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--kl_free_nats", type=float, default=3.0)
    p.add_argument("--cont_weight", type=float, default=1.0)
    p.add_argument("--imagination_horizon", type=int, default=15)
    p.add_argument("--imagination_starts", type=int, default=8)
    p.add_argument("--actor_entropy_scale", type=float, default=3e-3)

    p.add_argument("--geo_dim", type=int, default=32)
    p.add_argument("--geo_lr", type=float, default=3e-4)
    p.add_argument("--geo_warmup_steps", type=int, default=5_000)
    p.add_argument("--geo_data_max_points", type=int, default=50_000)

    p.add_argument("--geo_sup_epochs", type=int, default=200)
    p.add_argument("--geo_sup_batch", type=int, default=256)
    p.add_argument("--geo_sup_stress_eps", type=float, default=0.5)
    p.add_argument("--geo_sup_stress_weight", type=float, default=1.0)
    p.add_argument("--geo_sup_uniformity_weight", type=float, default=0.1)
    p.add_argument("--geo_sup_uniformity_t", type=float, default=2.0)

    p.add_argument("--expl_amount", type=float, default=0.3)
    p.add_argument("--expl_decay", type=float, default=0.0)
    p.add_argument("--expl_min", type=float, default=0.0)

    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--train_log_interval", type=int, default=20)
    p.add_argument("--log_dir", type=str, default="runs")

    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.seeds = args.seeds[:2]
        args.max_episodes = 20
        args.train_steps = 25
        args.collect_interval = 80
        args.eval_episodes = 5
        args.replay_capacity = 80_000
        args.batch_size = 32
        args.seq_len = 32
        args.imagination_starts = 4

    variants: List[VariantCfg] = [
        VariantCfg(name="geo_plan_only", geo_variant="plan_only", geo_plan_weight=0.15),
        VariantCfg(name="geo_shaping", geo_variant="shaping", geo_shaping_alpha=0.3),
        VariantCfg(name="geo_aux_backprop", geo_variant="aux_backprop", geo_aux_weight=0.05),
    ]

    all_results: Dict[str, List[Dict[str, float]]] = {v.name: [] for v in variants}

    print("\nDreamer + geodesic g_geo (plan_only / shaping / aux_backprop)")
    print(f"Env: {args.env_id} | seeds={args.seeds} | quick={args.quick}")
    for v in variants:
        print(f"  - {v}")
    print("")

    for v in variants:
        for seed in args.seeds:
            print(f"== {v.name} (seed={seed}) ==")
            out = run_one_seed(args, v, seed)
            all_results[v.name].append(out)
            print(f"  eval: {out['eval_mean']:.1f} ± {out['eval_std']:.1f}\n")

    print("\n=== Summary (across seeds) ===")
    for v in variants:
        rets = [r["eval_mean"] for r in all_results[v.name]]
        if rets:
            print(f"{v.name:20s}  mean={np.mean(rets):7.2f}  std={np.std(rets):7.2f}  n={len(rets)}")


if __name__ == "__main__":
    main()
