#!/usr/bin/env python3
"""
Notes
-----
* Geometry is anchored to encoder features via a learnable head phi(s)->e.
* The Jacobian regularizer is a directional (JVP) approximation encouraging
  local near-isometry so that L2 in s has a chance to approximate manifold
  distances without expensive kNN geodesics.
* The actor gets a small smoothness penalty in the same feature space to
  actually *use* the geometry during imagination.
"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"


def _f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence

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

# ===============================
#  Small helper modules
# ===============================


class FeatureHead(nn.Module):
    """phi(s) -> e  (geometry anchor into encoder feature space)."""

    def __init__(self, stoch_dim: int, embed_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(stoch_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discounts: torch.Tensor,
    lambda_: float = 0.95,
) -> torch.Tensor:
    """Dreamer-style lambda returns.

    rewards:   [B, H]
    values:    [B, H+1]
    discounts: [B, H]
    returns:   [B, H]
    """
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
    """Cumulative product weights: w_0=1, w_t=prod_{i<t} discounts_i."""
    B, H = discounts.shape
    ones = torch.ones((B, 1), device=discounts.device, dtype=discounts.dtype)
    return torch.cumprod(torch.cat([ones, discounts], dim=1), dim=1)[:, :-1]


# ===============================
#  Regularizers
# ===============================


def l2_bisim_loss(
    z: torch.Tensor, z_next: torch.Tensor, r: torch.Tensor, gamma: float = 0.99
) -> torch.Tensor:
    """L2 bisimulation on flattened pairs.

    z, z_next: [N, Dz]
    r:         [N]
    """
    N = z.shape[0]
    perm = torch.randperm(N, device=z.device)
    z2 = z[perm].detach()
    dz = torch.norm(z - z2, p=2, dim=1)
    with torch.no_grad():
        dnext = torch.norm(z_next - z_next[perm], p=2, dim=1)
        dr = (r - r[perm]).abs()
        target = dr + gamma * dnext
        scale = target.abs().mean().clamp_min(1e-6)
    return F.mse_loss(dz / scale, target / scale)


def jacobian_isometry_jvp(
    phi: nn.Module, s: torch.Tensor, num_projections: int = 4
) -> torch.Tensor:
    """Directional JVP estimate encouraging ||Jv||^2 ~= ||v||^2.

    Uses Rademacher v for stability. Returns scalar.
    """
    s = s.detach().requires_grad_(True)
    loss = 0.0
    for _ in range(num_projections):
        v = (torch.randint(0, 2, s.shape, device=s.device) * 2 - 1).to(s.dtype)
        _, jv = torch.autograd.functional.jvp(
            lambda ss: phi(ss), (s,), (v,), create_graph=True
        )
        jv2 = jv.pow(2).mean(dim=1)
        v2 = v.pow(2).mean(dim=1)
        loss = loss + (jv2 - v2).pow(2).mean()
    return loss / float(num_projections)


# ===============================
#  Evaluation
# ===============================


@torch.no_grad()
def evaluate_actor_policy(
    env_id: str,
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
    env = make_env(env_id, img_size=(img_size, img_size), num_stack=1)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    encoder.eval()
    rssm.eval()
    actor.eval()
    returns: List[float] = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        obs_t = torch.tensor(
            np.ascontiguousarray(obs), dtype=torch.float32, device=device
        )
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=bit_depth)
        e = encoder(obs_t)
        h, s = rssm.get_init_state(e)

        while not done:
            action, _ = actor.get_action(h, s, deterministic=True)
            a_np = action.squeeze(0).cpu().numpy().astype(np.float32)

            """
            total_reward = 0.0
            for _ in range(action_repeat):
                obs, r, term, trunc, _ = env.step(a_np)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)
            ep_ret += total_reward
            """

            obs, total_reward, term, trunc, _ = env.step(a_np, repeat=action_repeat)
            done = bool(term or trunc)
            ep_ret += float(total_reward)

            obs_t = torch.tensor(
                np.ascontiguousarray(obs), dtype=torch.float32, device=device
            )
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=bit_depth)
            e = encoder(obs_t)
            h, s, _, _ = rssm.observe_step(e, action, h, s, sample=False)

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ===============================
#  Training
# ===============================


@dataclass
class VariantCfg:
    name: str
    # representation shaping
    l2_bisim_weight: float = 0.0
    # geometry head
    geom_head_weight: float = 0.0
    geom_iso_weight: float = 0.0
    geom_iso_projections: int = 2
    # actor term
    actor_geom_smooth_weight: float = 0.0


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
        args.action_repeat
        if args.action_repeat > 0
        else ENV_ACTION_REPEAT.get(args.env_id, 2)
    )
    effective_gamma = args.gamma**action_repeat

    # Models
    encoder = ConvEncoder(embedding_size=args.embed_dim, in_channels=C).to(device)
    decoder = ConvDecoder(
        args.deter_dim, args.stoch_dim, embedding_size=args.embed_dim, out_channels=C
    ).to(device)
    rssm = RSSM(
        args.stoch_dim, args.deter_dim, act_dim, args.embed_dim, args.hidden_dim
    ).to(device)
    reward_model = RewardModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(
        device
    )
    cont_model = ContinueModel(args.deter_dim, args.stoch_dim, args.hidden_dim).to(
        device
    )
    actor = Actor(args.deter_dim, args.stoch_dim, act_dim, args.actor_hidden_dim).to(
        device
    )
    value_model = ValueModel(args.deter_dim, args.stoch_dim, args.value_hidden_dim).to(
        device
    )

    encoder = torch.compile(encoder)
    decoder = torch.compile(decoder)
    rssm = torch.compile(rssm)
    actor = torch.compile(actor)
    value_model = torch.compile(value_model)

    phi = None
    if (
        cfg.geom_head_weight > 0
        or cfg.geom_iso_weight > 0
        or cfg.actor_geom_smooth_weight > 0
    ):
        phi = FeatureHead(
            args.stoch_dim, args.embed_dim, hidden_dim=args.geom_hidden_dim
        ).to(device)

    # Optims
    world_params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(rssm.parameters())
        + list(reward_model.parameters())
        + list(cont_model.parameters())
        + ([] if phi is None else list(phi.parameters()))
    )
    model_optim = torch.optim.Adam(world_params, lr=args.model_lr, eps=args.adam_eps)
    actor_optim = torch.optim.Adam(
        actor.parameters(), lr=args.actor_lr, eps=args.adam_eps
    )
    value_optim = torch.optim.Adam(
        value_model.parameters(), lr=args.value_lr, eps=args.adam_eps
    )

    replay = ReplayBuffer(args.replay_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    # Seed buffer with random policy
    for _ in range(args.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            """
            total_reward = 0.0
            for _ in range(action_repeat):
                next_obs, r, term, trunc, _ = env.step(action)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)
            """

            next_obs, total_reward, term, trunc, _ = env.step(
                action, repeat=action_repeat
            )
            done = bool(term or trunc)

            replay.add(
                obs=np.ascontiguousarray(obs, np.uint8),
                action=np.asarray(action, np.float32),
                reward=total_reward,
                next_obs=None,  # not needed
                done=done,
            )
            obs = next_obs

    total_steps = 0
    expl = args.expl_amount
    t_start = time.time()
    train_updates = 0

    # Online episodes
    for ep in range(args.max_episodes):
        obs, _ = env.reset()
        done = False
        # init belief
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
            # action
            encoder.eval()
            rssm.eval()
            actor.eval()
            with torch.no_grad():
                a_t, _ = actor.get_action(h, s, deterministic=False)
                if expl > 0:
                    a_t = torch.clamp(a_t + expl * torch.randn_like(a_t), -1.0, 1.0)
                a_np = a_t.squeeze(0).cpu().numpy().astype(np.float32)

            # env step
            """
            total_reward = 0.0
            for _ in range(action_repeat):
                next_obs, r, term, trunc, _ = env.step(a_np)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)
            """
            next_obs, total_reward, term, trunc, _ = env.step(
                a_np, repeat=action_repeat
            )
            done = bool(term or trunc)
            replay.add(
                obs=np.ascontiguousarray(obs, np.uint8),
                action=a_np,
                reward=total_reward,
                next_obs=None,
                done=done,
            )
            obs = next_obs
            total_steps += 1

            # belief update
            obs_t = (
                torch.tensor(
                    np.ascontiguousarray(obs), dtype=torch.float32, device=device
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            preprocess_img(obs_t, depth=args.bit_depth)
            with torch.no_grad():
                e = encoder(obs_t)
                act_t = torch.tensor(
                    a_np, dtype=torch.float32, device=device
                ).unsqueeze(0)
                h, s, _, _ = rssm.observe_step(e, act_t, h, s, sample=False)

            # training bursts
            if total_steps % args.collect_interval == 0 and replay.size > (
                args.seq_len + 2
            ):
                encoder.train()
                decoder.train()
                rssm.train()
                reward_model.train()
                cont_model.train()
                actor.train()
                value_model.train()
                if phi is not None:
                    phi.train()

                for _ in range(args.train_steps):
                    batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)
                    obs_seq = torch.as_tensor(
                        batch.obs, device=device
                    ).float()  # [B, T+1, H, W, C]
                    act_seq = torch.as_tensor(
                        batch.actions, device=device
                    )  # [B, T+1, A]
                    rew_seq = torch.as_tensor(batch.rews, device=device)  # [B, T+1]
                    done_seq = torch.as_tensor(batch.dones, device=device)  # [B, T+1]

                    B, T1 = rew_seq.shape
                    T = T1 - 1

                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()  # [B, T+1, C, H, W]
                    preprocess_img(x, depth=args.bit_depth)

                    # ------------------- World model -------------------
                    e_t = bottle(encoder, x)  # [B, T+1, E]
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

                    h_seq = torch.stack(h_list, dim=1)  # [B, T, Dh]
                    s_seq = torch.stack(s_list, dim=1)  # [B, T, Ds]

                    pri_m = torch.stack([p[0] for p in pri_list], dim=0)  # [T, B, Ds]
                    pri_s = torch.stack([p[1] for p in pri_list], dim=0)
                    pos_m = torch.stack([p[0] for p in post_list], dim=0)
                    pos_s = torch.stack([p[1] for p in post_list], dim=0)
                    kld = torch.max(
                        kl_divergence(Normal(pos_m, pos_s), Normal(pri_m, pri_s)).sum(
                            -1
                        ),
                        free_nats,
                    ).mean()

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
                    cont_loss = F.binary_cross_entropy_with_logits(
                        cont_logits, cont_target
                    )

                    # L2 bisim (local)
                    bisim_val = torch.zeros((), device=device)
                    if cfg.l2_bisim_weight > 0 and T > 2:
                        z = s_seq[:, :-1]
                        zn = s_seq[:, 1:]
                        r = rew_seq[:, : T - 1]
                        z_ln = F.layer_norm(z, (z.size(-1),))
                        zn_ln = F.layer_norm(zn, (zn.size(-1),))
                        bisim_val = l2_bisim_loss(
                            z_ln.reshape(-1, z_ln.size(-1)),
                            zn_ln.reshape(-1, zn_ln.size(-1)),
                            r.reshape(-1),
                            gamma=args.gamma,
                        )

                    # Geometry head: match phi(s) to encoder features
                    geom_fit = torch.zeros((), device=device)
                    geom_iso = torch.zeros((), device=device)
                    if phi is not None and (
                        cfg.geom_head_weight > 0 or cfg.geom_iso_weight > 0
                    ):
                        # Target: encoder features at t+1 to align with posterior at t
                        # (both correspond to the same observation index)
                        e_target = (
                            e_t[:, 1 : T + 1].detach().reshape(-1, args.embed_dim)
                        )
                        s_flat = s_seq.reshape(-1, args.stoch_dim)
                        e_pred = phi(s_flat)
                        # Normalize (helps when encoder has LN/VICReg-style scale)
                        e_pred_n = F.layer_norm(e_pred, (e_pred.size(-1),))
                        e_tgt_n = F.layer_norm(e_target, (e_target.size(-1),))
                        geom_fit = F.mse_loss(e_pred_n, e_tgt_n)
                        if cfg.geom_iso_weight > 0:
                            geom_iso = jacobian_isometry_jvp(
                                phi, s_flat, num_projections=cfg.geom_iso_projections
                            )

                    model_loss = (
                        rec_loss
                        + args.kl_weight * kld
                        + rew_loss
                        + args.cont_weight * cont_loss
                        + cfg.l2_bisim_weight * bisim_val
                        + cfg.geom_head_weight * geom_fit
                        + cfg.geom_iso_weight * geom_iso
                    )

                    model_optim.zero_grad(set_to_none=True)
                    model_loss.backward()
                    torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip_norm)
                    model_optim.step()

                    # ------------------- Actor-Critic (imagination) -------------------
                    # Choose imagination starts
                    with torch.no_grad():
                        Dh = h_seq.size(-1)
                        Ds = s_seq.size(-1)
                        if 0 < args.imagination_starts < T:
                            K = args.imagination_starts
                            t_idx = torch.randint(0, T, (B, K), device=device)
                            h0 = (
                                h_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Dh))
                                .reshape(-1, Dh)
                                .detach()
                            )
                            s0 = (
                                s_seq.gather(1, t_idx.unsqueeze(-1).expand(-1, -1, Ds))
                                .reshape(-1, Ds)
                                .detach()
                            )
                        else:
                            h0 = h_seq.reshape(-1, Dh).detach()
                            s0 = s_seq.reshape(-1, Ds).detach()

                    with (
                        no_param_grads(rssm),
                        no_param_grads(reward_model),
                        no_param_grads(cont_model),
                    ):
                        # Rollout imagination
                        h_im_list = [h0]
                        s_im_list = [s0]
                        logp_list = []
                        h_im, s_im = h0, s0
                        for _ in range(args.imagination_horizon):
                            a_im, logp = actor.get_action(
                                h_im, s_im, deterministic=False
                            )
                            logp_list.append(logp)
                            h_im = rssm.deterministic_state_fwd(h_im, s_im, a_im)
                            s_im = rssm.state_prior(h_im, sample=True)
                            h_im_list.append(h_im)
                            s_im_list.append(s_im)
                        h_imag = torch.stack(h_im_list, dim=1)  # [B_im, H+1, Dh]
                        s_imag = torch.stack(s_im_list, dim=1)  # [B_im, H+1, Ds]

                        # Rewards & discounts
                        # rewards_im = bottle(reward_model, h_imag[:, :-1], s_imag[:, :-1])
                        rewards_im = bottle(reward_model, h_imag[:, 1:], s_imag[:, 1:])
                        cont_logits_im = bottle(
                            cont_model, h_imag[:, 1:], s_imag[:, 1:]
                        )
                        pcont = torch.sigmoid(cont_logits_im).clamp(0.0, 1.0)
                        discounts = effective_gamma * pcont

                    # Value targets (no grad). Important: value targets should NOT
                    # backprop into the imagination rollout.
                    with torch.no_grad():
                        values_tgt = bottle(value_model, h_imag, s_imag)  # [B_im, H+1]
                        lam_ret_tgt = compute_lambda_returns(
                            rewards_im.detach(),
                            values_tgt,
                            discounts.detach(),
                            lambda_=args.lambda_,
                        )  # [B_im, H]
                        w_tgt = compute_discount_weights(
                            discounts.detach()
                        )  # [B_im, H]

                    # Value loss (predict the target returns)
                    v_pred = bottle(value_model, h_imag.detach(), s_imag.detach())
                    value_loss = ((v_pred[:, :-1] - lam_ret_tgt) ** 2 * w_tgt).mean()
                    value_optim.zero_grad(set_to_none=True)
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        value_model.parameters(), args.grad_clip_norm
                    )
                    value_optim.step()

                    # Actor loss
                    actor_entropy_scale = args.actor_entropy_scale
                    dist = actor.get_dist(
                        h_imag[:, :-1].detach(), s_imag[:, :-1].detach()
                    )
                    entropy = dist.entropy().sum(dim=-1).mean()
                    # Actor objective: maximize imagined returns.
                    # Gradients should flow through rewards/discounts/dynamics, but not
                    # through the value model.
                    """
                    with torch.no_grad():
                        values_for_actor = bottle(
                            value_model, h_imag, s_imag
                        )  # detached from graph
                        w_actor = compute_discount_weights(discounts)  # just weights
                    """
                    with no_param_grads(value_model):
                        values_for_actor = bottle(
                            value_model, h_imag, s_imag
                        )  # gradients flow to h/s
                    w_actor = compute_discount_weights(discounts.detach())
                    lam_ret_actor = compute_lambda_returns(
                        rewards_im, values_for_actor, discounts, lambda_=args.lambda_
                    )
                    actor_loss = (
                        -(w_actor.detach() * lam_ret_actor).mean()
                        - actor_entropy_scale * entropy
                    )

                    # Geometry-aware actor smoothness (feature space)
                    if phi is not None and cfg.actor_geom_smooth_weight > 0:
                        with no_param_grads(phi):
                            f_prev = phi(
                                s_imag[:, :-1].reshape(-1, args.stoch_dim)
                            ).view(s_imag.size(0), -1, args.embed_dim)
                            f_next = phi(
                                s_imag[:, 1:].reshape(-1, args.stoch_dim)
                            ).view(s_imag.size(0), -1, args.embed_dim)
                        smooth = (f_next - f_prev).pow(2).mean()
                        actor_loss = actor_loss + cfg.actor_geom_smooth_weight * smooth

                    actor_optim.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), args.grad_clip_norm
                    )
                    actor_optim.step()

                    train_updates += 1

                    if args.train_log_interval > 0 and (
                        train_updates % args.train_log_interval == 0
                    ):
                        # raw KL diagnostic (useful to see if you're always clamped by free_nats)
                        raw_kl = kl_divergence(
                            Normal(pos_m, pos_s), Normal(pri_m, pri_s)
                        ).sum(-1)  # [T, B]
                        kl_raw_mean = raw_kl.mean().item()
                        kl_clamp_frac = (raw_kl < free_nats).float().mean().item()

                        # geometry diagnostics (cheap-ish)
                        iso_ratio = None
                        if (phi is not None) and (cfg.geom_iso_weight > 0):
                            s_flat_dbg = s_seq.reshape(-1, args.stoch_dim).detach()
                            # subsample to keep JVP diagnostic cheap (up to ~256 points)
                            step = max(1, s_flat_dbg.shape[0] // 256)
                            s_sub = s_flat_dbg[::step].requires_grad_(True)
                            v = (
                                torch.randint(0, 2, s_sub.shape, device=s_sub.device)
                                * 2
                                - 1
                            ).to(s_sub.dtype)
                            _, jv = torch.autograd.functional.jvp(
                                lambda ss: phi(ss), (s_sub,), (v,), create_graph=False
                            )
                            iso_ratio = (
                                (
                                    jv.pow(2).mean(dim=1)
                                    / v.pow(2).mean(dim=1).clamp_min(1e-8)
                                )
                                .mean()
                                .item()
                            )

                        head_cos = None
                        if (phi is not None) and (cfg.geom_head_weight > 0):
                            # reuse the same alignment you train on
                            e_target = (
                                e_t[:, 1 : T + 1].detach().reshape(-1, args.embed_dim)
                            )
                            s_flat = s_seq.reshape(-1, args.stoch_dim).detach()
                            e_pred = phi(s_flat).detach()
                            e_pred_n = F.layer_norm(e_pred, (e_pred.size(-1),))
                            e_tgt_n = F.layer_norm(e_target, (e_target.size(-1),))
                            head_cos = (
                                F.cosine_similarity(e_pred_n, e_tgt_n, dim=-1)
                                .mean()
                                .item()
                            )

                        smooth_val = None
                        if (phi is not None) and (cfg.actor_geom_smooth_weight > 0):
                            smooth_val = smooth.detach().item()

                        print(
                            f"[{cfg.name} | seed {seed}] upd={train_updates:6d} env={total_steps:7d} "
                            f"rec={_f(rec_loss)} kl={_f(kld)} (raw={kl_raw_mean:.3f}, clamp={kl_clamp_frac:.2f}) "
                            f"rew={_f(rew_loss)} cont={_f(cont_loss)} bis={_f(bisim_val, 3)} "
                            f"gfit={_f(geom_fit)} isoL={_f(geom_iso)} "
                            f"isoR={(f'{iso_ratio:.2f}' if iso_ratio is not None else 'NA')} "
                            f"cos={(f'{head_cos:.2f}' if head_cos is not None else 'NA')} "
                            f"sm={(f'{smooth_val:.2e}' if smooth_val is not None else 'NA')} "
                            f"act={_f(actor_loss)} val={_f(value_loss)} ent={_f(entropy)}",
                            flush=True,
                        )

        # (optional) exploration decay
        if args.expl_decay > 0:
            expl = max(args.expl_min, expl - args.expl_decay)

        if (ep + 1) % max(1, args.max_episodes // 5) == 0:
            dt = time.time() - t_start
            print(
                f"    [{cfg.name} | seed {seed}] ep {ep + 1}/{args.max_episodes}  "
                f"steps={total_steps}  buf={replay.size}  ({dt:.0f}s)"
            )

        if args.eval_interval > 0 and (ep + 1) % args.eval_interval == 0:
            eval_mean, eval_std = evaluate_actor_policy(
                env_id=args.env_id,
                img_size=args.img_size,
                encoder=encoder,
                rssm=rssm,
                actor=actor,
                episodes=args.eval_episodes,
                seed=seed + 10000,  # Keep seed fixed for consistent validation set
                device=device,
                bit_depth=args.bit_depth,
                action_repeat=action_repeat,
            )
            print(
                f"    [Eval @ ep {ep + 1}] mean={eval_mean:.2f} std={eval_std:.2f}",
                flush=True,
            )

    env.close()

    mean_ret, std_ret = evaluate_actor_policy(
        env_id=args.env_id,
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

    return {
        "seed": float(seed),
        "eval_mean": float(mean_ret),
        "eval_std": float(std_ret),
        "total_steps": float(total_steps),
    }


# ===============================
#  CLI
# ===============================


def parse_args():
    p = argparse.ArgumentParser(
        description="Dreamer ablation: baseline vs L2 bisim vs geometry-aware"
    )
    p.add_argument("--env_id", type=str, default="cartpole-swingup")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])

    p.add_argument("--quick", action="store_true")

    # Training schedule
    p.add_argument("--seed_episodes", type=int, default=5)
    p.add_argument("--max_episodes", type=int, default=50)
    p.add_argument("--collect_interval", type=int, default=100)
    p.add_argument("--train_steps", type=int, default=50)
    p.add_argument("--action_repeat", type=int, default=0, help="0 = env default")
    p.add_argument("--replay_capacity", type=int, default=200_000)

    # Batch / sequences
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=50)

    # Architecture
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--actor_hidden_dim", type=int, default=400)
    p.add_argument("--value_hidden_dim", type=int, default=400)
    p.add_argument("--geom_hidden_dim", type=int, default=256)

    # Loss / RL
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

    # Exploration
    p.add_argument("--expl_amount", type=float, default=0.3)
    p.add_argument("--expl_decay", type=float, default=0.0)
    p.add_argument("--expl_min", type=float, default=0.0)

    # Evaluation
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument(
        "--eval_interval", type=int, default=50, help="Run eval every N episodes"
    )

    p.add_argument(
        "--train_log_interval",
        type=int,
        default=20,
        help="print every N gradient updates",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        # Keep this modest; dm_control pixels can still be slow.
        args.seeds = args.seeds[:2]
        args.max_episodes = 20
        args.train_steps = 25
        args.collect_interval = 80
        args.eval_episodes = 5
        args.replay_capacity = 80_000
        args.batch_size = 32
        args.seq_len = 32
        args.imagination_starts = 4

    """
    variants: List[VariantCfg] = [
        VariantCfg(name="dreamer"),
        VariantCfg(name="dreamer+l2bisim", l2_bisim_weight=0.03),
        VariantCfg(
            name="dreamer+geometry",
            l2_bisim_weight=0.03,
            geom_head_weight=1.0,
            geom_iso_weight=0.05,
            geom_iso_projections=2,
            actor_geom_smooth_weight=1e-3,
        ),
    ]
    """

    variants: List[VariantCfg] = [
        # 0) baseline
        VariantCfg(name="dreamer"),
        # 1) representation shaping only
        VariantCfg(name="dreamer+l2bisim", l2_bisim_weight=0.03),
        # 2) geometry head ONLY (no isometry, no actor smoothness)
        VariantCfg(
            name="dreamer+geom_head",
            l2_bisim_weight=0.03,
            geom_head_weight=1.0,
            geom_iso_weight=0.0,
            actor_geom_smooth_weight=0.0,
        ),
        # 3) isometry ONLY (IMPORTANT: give it a *tiny* head fit so phi doesn't drift arbitrarily)
        VariantCfg(
            name="dreamer+iso_only",
            l2_bisim_weight=0.03,
            geom_head_weight=0.05,  # tiny anchor
            geom_iso_weight=0.05,
            geom_iso_projections=2,
            actor_geom_smooth_weight=0.0,
        ),
        # 4) full
        VariantCfg(
            name="dreamer+geometry",
            l2_bisim_weight=0.03,
            geom_head_weight=1.0,
            geom_iso_weight=0.05,
            geom_iso_projections=2,
            actor_geom_smooth_weight=1e-3,
        ),
    ]

    all_results: Dict[str, List[Dict[str, float]]] = {v.name: [] for v in variants}

    print("\nDreamer regularization ablation")
    print(f"Env: {args.env_id} | seeds={args.seeds} | quick={args.quick}")
    print("Variants:")
    for v in variants:
        print(f"  - {v}")
    print("")

    for v in variants:
        for seed in args.seeds:
            print(f"== {v.name} (seed={seed}) ==")
            out = run_one_seed(args, v, seed)
            all_results[v.name].append(out)
            print(f"  eval: {out['eval_mean']:.1f} ± {out['eval_std']:.1f}\n")

    # Aggregate
    print("\n=== Summary (across seeds) ===")
    for v in variants:
        rets = [r["eval_mean"] for r in all_results[v.name]]
        if len(rets) == 0:
            continue
        print(
            f"{v.name:16s}  mean={np.mean(rets):7.2f}  std={np.std(rets):7.2f}  n={len(rets)}"
        )


if __name__ == "__main__":
    main()
