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
from geom_head import GeoEncoder, temporal_reachability_loss, geo_step_penalty, min_bank_dist

# ===============================
#  Small helper modules
# ===============================


def approx_knn_stats(x: torch.Tensor, bank_sub: torch.Tensor) -> tuple[float, float]:
    """
    Approximate kNN distance of x to bank_sub using min over cdist rows.
    x: [N, Ds], bank_sub: [M, Ds]
    returns: (mean, p90)
    """
    d = torch.cdist(x, bank_sub).min(dim=1).values  # [N]
    return d.mean().item(), torch.quantile(d, 0.90).item()


def subsample_rows(x, max_rows):
    """x: [N, ...] -> returns <=max_rows rows"""
    N = x.shape[0]
    if max_rows is None or max_rows <= 0 or N <= max_rows:
        return x
    idx = torch.randperm(N, device=x.device)[:max_rows]
    return x[idx]


def knn_gate_weights(s_prev, bank_sub, tau, max_gate_pts=512):
    """
    s_prev: [N, Ds], bank_sub: [M, Ds], tau: scalar tensor/float
    Returns w: [N] with nonzero entries only for sampled points.
    """
    N = s_prev.shape[0]
    device = s_prev.device
    w = torch.zeros(N, device=device)

    # print(s_prev.device, bank_sub.device)

    if N == 0:
        return w

    if max_gate_pts is not None and max_gate_pts > 0 and N > max_gate_pts:
        idx = torch.randperm(N, device=device)[:max_gate_pts]
        s_sub = s_prev[idx]
        d = torch.cdist(s_sub, bank_sub).min(dim=1).values
        w[idx] = torch.relu(d - tau)
    else:
        d = torch.cdist(s_prev, bank_sub).min(dim=1).values
        w = torch.relu(d - tau)

    return w


class ScalarEMA:
    def __init__(self, init=0.0, beta=0.99):
        self.v = float(init)
        self.beta = beta
        self.inited = False

    def update(self, x: float) -> float:
        x = float(x)
        if not self.inited:
            self.v = x
            self.inited = True
        else:
            self.v = self.beta * self.v + (1 - self.beta) * x
        return self.v


class LatentBank:
    def __init__(self, capacity: int, ds: int, device="cpu"):
        self.capacity = capacity
        self.ds = ds
        self.device = device
        self.buf = torch.empty((capacity, ds), dtype=torch.float32, device=device)
        self.n = 0
        self.i = 0

    def add(self, x: torch.Tensor):
        # x: [N, Ds] (cpu float32 preferred)
        x = x.detach().to(self.device, dtype=torch.float32)
        if x.ndim != 2 or x.size(1) != self.ds:
            raise ValueError("LatentBank.add: wrong shape")
        for row in x:
            if self.n < self.capacity:
                # FIFO: add at next free slot
                self.buf[self.i].copy_(row)
                self.i = (self.i + 1) % self.capacity
                self.n += 1
            else:
                # Buffer full: evict the bank point closest to this new point
                d = torch.cdist(row.unsqueeze(0), self.buf[:self.n])  # [1, n]
                evict_idx = d.squeeze(0).argmin().item()
                self.buf[evict_idx].copy_(row)
            

    def sample(self, m: int, device: torch.device) -> torch.Tensor:
        if self.n == 0:
            return None
        idx = torch.randint(0, self.n, (m,), device=torch.device("cpu"))
        out = self.buf[idx].to(device)
        return out


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


def directional_pullback_len_jvp(
    phi: nn.Module, s: torch.Tensor, ds: torch.Tensor
) -> torch.Tensor:
    """
    Returns per-sample length: || J_phi(s) ds ||^2.
    s, ds: [N, Ds]
    output: [N]
    """
    s = s.detach().requires_grad_(True)
    ds = ds.detach()
    _, jv = torch.autograd.functional.jvp(
        lambda ss: phi(ss), (s,), (ds,), create_graph=True
    )
    return jv.pow(2).mean(dim=1)  # mean over feature dims (stable scaling)


def _rademacher_like(x: torch.Tensor) -> torch.Tensor:
    return (torch.randint(0, 2, x.shape, device=x.device) * 2 - 1).to(x.dtype)


def rssm_transition_mean(rssm: nn.Module, h: torch.Tensor, s: torch.Tensor, a: torch.Tensor):
    """Transition map f(h, s, a) -> (h', s'_prior_mean)."""
    h_next = rssm.deterministic_state_fwd(h, s, a)
    s_mean, _ = rssm.state_prior(h_next)
    return h_next, s_mean


def dynamics_pullback_trace_jvp(
    rssm: nn.Module,
    h: torch.Tensor,
    s: torch.Tensor,
    a: torch.Tensor,
    num_projections: int = 4,
) -> torch.Tensor:
    """Hutchinson estimate of tr(J_f^T J_f) for f(h, s, a) = (h', s'_prior_mean)."""
    h = h.detach().requires_grad_(True)
    s = s.detach().requires_grad_(True)
    a = a.detach().requires_grad_(True)
    loss = 0.0
    for _ in range(num_projections):
        v_h = _rademacher_like(h)
        v_s = _rademacher_like(s)
        v_a = _rademacher_like(a)
        _, (jv_h, jv_s) = torch.autograd.functional.jvp(
            lambda hh, ss, aa: rssm_transition_mean(rssm, hh, ss, aa),
            (h, s, a),
            (v_h, v_s, v_a),
            create_graph=True,
        )
        trace_est = jv_h.pow(2).sum(dim=1) + jv_s.pow(2).sum(dim=1)
        loss = loss + trace_est.mean()
    return loss / float(num_projections)


def dynamics_pullback_len_jvp(
    rssm: nn.Module,
    h: torch.Tensor,
    s: torch.Tensor,
    a: torch.Tensor,
    dh: torch.Tensor,
    ds: torch.Tensor,
    da: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Per-sample squared length ||J_f(h,s,a)*(dh,ds,da)||^2 (mean over dims)."""
    h = h.detach().requires_grad_(True)
    s = s.detach().requires_grad_(True)
    a = a.detach().requires_grad_(True)
    dh = dh.detach()
    ds = ds.detach()
    da = da.detach()
    _, (jv_h, jv_s) = torch.autograd.functional.jvp(
        lambda hh, ss, aa: rssm_transition_mean(rssm, hh, ss, aa),
        (h, s, a),
        (dh, ds, da),
        create_graph=create_graph,
    )
    return jv_h.pow(2).mean(dim=1) + jv_s.pow(2).mean(dim=1)


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
    # dynamics pullback
    dyn_pb_reg_weight: float = 0.0
    dyn_pb_reg_projections: int = 2
    dyn_pb_bisim_weight: float = 0.0
    # geometry head
    geom_head_weight: float = 0.0
    geom_iso_weight: float = 0.0
    geom_iso_projections: int = 2
    # actor term
    actor_geom_smooth_weight: float = 0.0
    actor_knn_gate: bool = False

    # GELATO fields
    geo_aux_weight: float = 0.0
    geo_reward_bonus_weight: float = 0.0
    geo_plan_penalty_weight: float = 0.0
    geo_plan_penalty: bool = False

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

    """
    encoder = torch.compile(encoder)
    decoder = torch.compile(decoder)
    rssm = torch.compile(rssm)
    actor = torch.compile(actor)
    value_model = torch.compile(value_model)
    """
    rssm_jvp = getattr(rssm, "_orig_mod", rssm)
    

    run_name = f"{args.env_id}_{cfg.name}_seed{seed}"
    writer = SummaryWriter(log_dir=os.path.join(args.log_dir, run_name))
    writer.add_text("hyperparameters", str(vars(args)), 0)
    writer.add_text("variant", str(cfg), 0)

    phi = None
    if (
        cfg.geom_head_weight > 0
        or cfg.geom_iso_weight > 0
        or cfg.actor_geom_smooth_weight > 0
    ):
        phi = FeatureHead(
            args.stoch_dim, args.embed_dim, hidden_dim=args.geom_hidden_dim
        ).to(device)

    geo = None
    geo_bank = None

    if (
        cfg.geo_aux_weight > 0
        or cfg.geo_reward_bonus_weight > 0
        or cfg.geo_plan_penalty_weight > 0
    ):
        geo = GeoEncoder(
            args.deter_dim,
            args.stoch_dim,
            geo_dim=args.geo_dim,
            hidden_dim=args.geom_hidden_dim,
        ).to(device)
        geo_bank = LatentBank(
            capacity=args.geo_bank_capacity,
            ds=args.geo_dim,
            device="cpu",
        )

    geo_optim = None
    if geo is not None:
        geo_optim = torch.optim.Adam(
            geo.parameters(), lr=args.geo_lr, eps=args.adam_eps
        )

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

    eta_ema = ScalarEMA(beta=0.99)
    tau_ema = ScalarEMA(beta=0.99)

    latent_bank = LatentBank(capacity=50_000, ds=args.stoch_dim, device="cpu")

    # Seed buffer with random policy
    for _ in range(args.seed_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()

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
        ep_return = 0.0
        ep_steps = 0
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
            next_obs, total_reward, term, trunc, _ = env.step(
                a_np, repeat=action_repeat
            )
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
                    h_prev_list, s_prev_list, a_prev_list = [], [], []
                    pri_list, post_list = [], []

                    for t in range(T):
                        h_prev_list.append(h_t)
                        s_prev_list.append(s_t)
                        a_prev_list.append(act_seq[:, t])
                        h_t = rssm.deterministic_state_fwd(h_t, s_t, act_seq[:, t])
                        pri_list.append(rssm.state_prior(h_t))
                        post_list.append(rssm.state_posterior(h_t, e_t[:, t + 1]))
                        pm, ps = post_list[-1]
                        s_t = pm + torch.randn_like(ps) * ps
                        h_list.append(h_t)
                        s_list.append(s_t)

                    h_seq = torch.stack(h_list, dim=1)  # [B, T, Dh]
                    s_seq = torch.stack(s_list, dim=1)  # [B, T, Ds]
                    h_prev_seq = torch.stack(h_prev_list, dim=1)  # [B, T, Dh]
                    s_prev_seq = torch.stack(s_prev_list, dim=1)  # [B, T, Ds]
                    a_prev_seq = torch.stack(a_prev_list, dim=1)  # [B, T, A]

                    pri_m = torch.stack([p[0] for p in pri_list], dim=0)  # [T, B, Ds]
                    pri_s = torch.stack([p[1] for p in pri_list], dim=0)
                    pos_m = torch.stack([p[0] for p in post_list], dim=0)
                    pos_s = torch.stack([p[1] for p in post_list], dim=0)
                    """
                    kld = torch.max(
                        kl_divergence(Normal(pos_m, pos_s), Normal(pri_m, pri_s)).sum(
                            -1
                        ),
                        free_nats,
                    ).mean()
                    """
                    # stop grad on each dist as in dreamerv2
                    kl_lhs = kl_divergence(Normal(pos_m.detach(), pos_s.detach()), Normal(pri_m, pri_s))
                    kl_rhs = kl_divergence(Normal(pos_m, pos_s), Normal(pri_m.detach(), pri_s.detach()))
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
                    cont_loss = F.binary_cross_entropy_with_logits(
                        cont_logits, cont_target
                    )

                    geo_aux_loss = torch.zeros((), device=device)
                    if geo is not None and total_steps >= args.geo_learn_after_steps:
                        g_real = bottle(geo, h_seq.detach(), s_seq.detach())  # [B, T, Dg]
                        geo_aux_loss = temporal_reachability_loss(
                            g_real,
                            pos_k=args.geo_pos_k,
                            neg_k=args.geo_neg_k,
                            margin=args.geo_margin,
                        )

                    dyn_pb_reg = torch.zeros((), device=device)
                    dyn_pb_bisim = torch.zeros((), device=device)
                    dyn_pb_dz_mean = None
                    dyn_pb_dnext_mean = None
                    dyn_pb_dz_std = None
                    dyn_pb_target_mean = None
                    dyn_pb_target_std = None
                    dyn_pb_corr = None
                    dyn_pb_scale = None
                    dyn_pb_ratio = None
                    if cfg.dyn_pb_reg_weight > 0:
                        h_flat = h_prev_seq.reshape(-1, h_prev_seq.size(-1))
                        s_flat = s_prev_seq.reshape(-1, s_prev_seq.size(-1))
                        a_flat = a_prev_seq.reshape(-1, a_prev_seq.size(-1))
                        if (
                            args.dyn_pb_max_pts > 0
                            and h_flat.size(0) > args.dyn_pb_max_pts
                        ):
                            idx = torch.randperm(h_flat.size(0), device=device)[
                                : args.dyn_pb_max_pts
                            ]
                            h_flat = h_flat[idx]
                            s_flat = s_flat[idx]
                            a_flat = a_flat[idx]
                        dyn_pb_reg = dynamics_pullback_trace_jvp(
                            rssm_jvp,
                            h_flat,
                            s_flat,
                            a_flat,
                            num_projections=cfg.dyn_pb_reg_projections,
                        )

                    if cfg.dyn_pb_bisim_weight > 0 and T > 1:
                        h_cur = h_prev_seq[:, :-1].reshape(-1, h_prev_seq.size(-1))
                        s_cur = s_prev_seq[:, :-1].reshape(-1, s_prev_seq.size(-1))
                        a_cur = a_prev_seq[:, :-1].reshape(-1, a_prev_seq.size(-1))
                        r_cur = rew_seq[:, : T - 1].reshape(-1)
                        h_next_cur = h_prev_seq[:, 1:].reshape(
                            -1, h_prev_seq.size(-1)
                        )
                        s_next_cur = s_prev_seq[:, 1:].reshape(
                            -1, s_prev_seq.size(-1)
                        )
                        a_next_cur = a_prev_seq[:, 1:].reshape(
                            -1, a_prev_seq.size(-1)
                        )

                        N = h_cur.size(0)
                        if args.dyn_pb_max_pts > 0 and N > args.dyn_pb_max_pts:
                            idx = torch.randperm(N, device=device)[
                                : args.dyn_pb_max_pts
                            ]
                            h_cur = h_cur[idx]
                            s_cur = s_cur[idx]
                            a_cur = a_cur[idx]
                            r_cur = r_cur[idx]
                            h_next_cur = h_next_cur[idx]
                            s_next_cur = s_next_cur[idx]
                            a_next_cur = a_next_cur[idx]
                            N = h_cur.size(0)

                        perm = torch.randperm(N, device=device)
                        h2 = h_cur[perm].detach()
                        s2 = s_cur[perm].detach()
                        a2 = a_cur[perm].detach()
                        dh = h_cur - h2
                        ds = s_cur - s2
                        da = a_cur - a2
                        dz2 = dynamics_pullback_len_jvp(
                            rssm_jvp, h_cur, s_cur, a_cur, dh, ds, da, create_graph=True
                        )
                        dz = torch.sqrt(dz2 + 1e-8)

                        h2n = h_next_cur[perm].detach()
                        s2n = s_next_cur[perm].detach()
                        a2n = a_next_cur[perm].detach()
                        dhn = h_next_cur - h2n
                        dsn = s_next_cur - s2n
                        dan = a_next_cur - a2n
                        dnext2 = dynamics_pullback_len_jvp(
                            rssm_jvp,
                            h_next_cur,
                            s_next_cur,
                            a_next_cur,
                            dhn,
                            dsn,
                            dan,
                            create_graph=False,
                        )
                        dnext = torch.sqrt(dnext2 + 1e-8)

                        with torch.no_grad():
                            dr = (r_cur - r_cur[perm]).abs()
                            target = dr + args.gamma * dnext
                            scale = target.abs().mean().clamp_min(1e-6)
                        dyn_pb_bisim = F.mse_loss(dz / scale, target / scale)
                        dyn_pb_dz_mean = dz.mean().item()
                        dyn_pb_dnext_mean = dnext.mean().item()
                        with torch.no_grad():
                            dyn_pb_dz_std = dz.std().item()
                            dyn_pb_target_mean = target.mean().item()
                            dyn_pb_target_std = target.std().item()
                            dyn_pb_scale = scale.item()
                            denom = dz.std() * target.std() + 1e-8
                            dyn_pb_corr = (
                                ((dz - dz.mean()) * (target - target.mean())).mean()
                                / denom
                            ).item()
                            dyn_pb_ratio = (
                                dz.mean() / target.mean().clamp_min(1e-8)
                            ).item()

                    model_loss = (
                        rec_loss
                        + args.kl_weight * kld
                        + rew_loss
                        + args.cont_weight * cont_loss
                        + cfg.dyn_pb_reg_weight * dyn_pb_reg
                        + cfg.dyn_pb_bisim_weight * dyn_pb_bisim
                        #+ cfg.geo_aux_weight * geo_aux_loss
                    )

                    model_optim.zero_grad(set_to_none=True)
                    model_loss.backward()
                    torch.nn.utils.clip_grad_norm_(world_params, args.grad_clip_norm)
                    model_optim.step()

                    if geo is not None and geo_optim is not None and cfg.geo_aux_weight > 0:
                        geo_aux_loss_val = geo_aux_loss
                        if geo_aux_loss_val.requires_grad:
                            geo_optim.zero_grad(set_to_none=True)
                            (cfg.geo_aux_weight * geo_aux_loss_val).backward()
                            torch.nn.utils.clip_grad_norm_(geo.parameters(), args.grad_clip_norm)
                            geo_optim.step()

                    if geo is not None:
                        with torch.no_grad():
                            g_bank_add = bottle(geo, h_seq.detach(), s_seq.detach()).reshape(-1, args.geo_dim)
                            g_bank_add = subsample_rows(g_bank_add, args.geo_bank_sample)
                            geo_bank.add(g_bank_add.cpu())

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

                    rewards_im_raw = rewards_im
                    rewards_total = rewards_im
                    geo_frontier_bonus = torch.zeros_like(rewards_im)
                    geo_jump_pen = torch.zeros((), device=device)

                    if geo is not None:
                        with no_param_grads(geo):
                            # freeze geo weights, but keep gradient to h_imag / s_imag
                            # before no_grad killed grad that shape actor
                            g_imag = bottle(geo, h_imag, s_imag)

                        # reward shaping starts after geometry has had time to learn
                        if total_steps >= args.geo_reward_after_steps and geo_bank is not None and geo_bank.n > 0:
                            bank_sub = geo_bank.sample(args.geo_bank_sample, device)
                            d_novel = min_bank_dist(g_imag[:, 1:], bank_sub)  # [B_im, H]

                            geo_frontier_bonus = torch.relu(d_novel - args.geo_frontier_tau)
                            if args.geo_frontier_cap > 0:
                                geo_frontier_bonus = geo_frontier_bonus.clamp(max=args.geo_frontier_cap)

                            rewards_total = rewards_total + cfg.geo_reward_bonus_weight * geo_frontier_bonus

                            rewards_im = rewards_total

                        # planning constraint (defined here, applied later)
                        if total_steps >= args.geo_plan_after_steps and cfg.geo_plan_penalty:
                            geo_jump_pen = geo_step_penalty(g_imag, args.geo_step_radius)

                    # Value targets (no grad). Important: value targets should NOT
                    # backprop into the imagination rollout.
                    with torch.no_grad():
                        values_tgt = bottle(value_model, h_imag, s_imag)  # [B_im, H+1]
                        lam_ret_tgt = compute_lambda_returns(
                            rewards_im_raw.detach(),
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
                    """
                    dist = actor.get_dist(
                        h_imag[:, :-1].detach(), s_imag[:, :-1].detach()
                    )
                    entropy = dist.entropy().sum(dim=-1).mean()
                    """
                    # use post tanh entropy
                    mean, std = actor.forward(h_imag[:, :-1].detach(), s_imag[:, :-1].detach())
                    noise = torch.randn_like(mean)
                    raw = mean + std * noise
                    entropy = (Normal(mean, std).entropy() 
                            + torch.log(1 - torch.tanh(raw).pow(2) + 1e-6)
                            ).sum(dim=-1).mean()
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
                        + cfg.geo_plan_penalty_weight * geo_jump_pen
                    )

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
                        writer.add_scalar(
                            "loss/model_total", model_loss.item(), total_steps
                        )
                        writer.add_scalar(
                            "loss/reconstruction", rec_loss.item(), total_steps
                        )
                        writer.add_scalar(
                            "loss/kl_divergence", kld.item(), total_steps
                        )
                        writer.add_scalar("loss/reward", rew_loss.item(), total_steps)
                        writer.add_scalar(
                            "loss/continue", cont_loss.item(), total_steps
                        )
                        writer.add_scalar(
                            "loss/dyn_pb_reg", dyn_pb_reg.item(), total_steps
                        )
                        writer.add_scalar(
                            "loss/dyn_pb_bisim", dyn_pb_bisim.item(), total_steps
                        )
                        writer.add_scalar("loss/actor", actor_loss.item(), total_steps)
                        writer.add_scalar("loss/value", value_loss.item(), total_steps)
                        writer.add_scalar("train/exploration", expl, total_steps)
                        writer.add_scalar(
                            "geom/dynpb_trace", dyn_pb_reg.item(), total_steps
                        )
                        writer.add_scalar(
                            "geom/dynpb_bisim_loss", dyn_pb_bisim.item(), total_steps
                        )

                        if geo is not None:
                            writer.add_scalar("loss/geo_aux", geo_aux_loss.item(), total_steps)
                            writer.add_scalar("train/geo_frontier_bonus", geo_frontier_bonus.mean().item(), total_steps)
                            writer.add_scalar("loss/geo_jump_pen", geo_jump_pen.item(), total_steps)

                            from geom_head import geo_embedding_stats
                            with torch.no_grad():
                                g_log = bottle(geo, h_seq.detach(), s_seq.detach())
                            stats = geo_embedding_stats(g_log)
                            for k, v in stats.items():
                                writer.add_scalar(k, v, total_steps)

                        if dyn_pb_dz_mean is not None:
                            writer.add_scalar(
                                "train/dyn_pb_dz_mean",
                                dyn_pb_dz_mean,
                                total_steps,
                            )
                            writer.add_scalar(
                                "geom/dynpb_dz_mean", dyn_pb_dz_mean, total_steps
                            )
                        if dyn_pb_dnext_mean is not None:
                            writer.add_scalar(
                                "train/dyn_pb_dnext_mean",
                                dyn_pb_dnext_mean,
                                total_steps,
                            )
                            writer.add_scalar(
                                "geom/dynpb_dnext_mean", dyn_pb_dnext_mean, total_steps
                            )
                        if dyn_pb_target_mean is not None:
                            writer.add_scalar(
                                "geom/dynpb_target_mean",
                                dyn_pb_target_mean,
                                total_steps,
                            )
                        if dyn_pb_target_std is not None:
                            writer.add_scalar(
                                "geom/dynpb_target_std",
                                dyn_pb_target_std,
                                total_steps,
                            )
                        if dyn_pb_dz_std is not None:
                            writer.add_scalar(
                                "geom/dynpb_dz_std", dyn_pb_dz_std, total_steps
                            )
                        if dyn_pb_corr is not None:
                            writer.add_scalar(
                                "geom/dynpb_corr", dyn_pb_corr, total_steps
                            )
                        if dyn_pb_scale is not None:
                            writer.add_scalar(
                                "geom/dynpb_scale", dyn_pb_scale, total_steps
                            )
                        if dyn_pb_ratio is not None:
                            writer.add_scalar(
                                "geom/dynpb_ratio", dyn_pb_ratio, total_steps
                            )

                        with torch.no_grad():
                            Dh = h_seq.size(-1)
                            Ds = s_seq.size(-1)
                            h_flat = h_seq.reshape(-1, Dh)
                            s_flat = s_seq.reshape(-1, Ds)
                            h_norm = h_flat.norm(dim=1)
                            s_norm = s_flat.norm(dim=1)
                            writer.add_scalar(
                                "geom/h_norm_mean", h_norm.mean().item(), total_steps
                            )
                            writer.add_scalar(
                                "geom/h_norm_std", h_norm.std().item(), total_steps
                            )
                            writer.add_scalar(
                                "geom/s_norm_mean", s_norm.mean().item(), total_steps
                            )
                            writer.add_scalar(
                                "geom/s_norm_std", s_norm.std().item(), total_steps
                            )
                            h_var = h_flat.var(dim=0, unbiased=False)
                            s_var = s_flat.var(dim=0, unbiased=False)
                            writer.add_scalar(
                                "geom/h_var_mean", h_var.mean().item(), total_steps
                            )
                            writer.add_scalar(
                                "geom/h_var_min", h_var.min().item(), total_steps
                            )
                            writer.add_scalar(
                                "geom/s_var_mean", s_var.mean().item(), total_steps
                            )
                            writer.add_scalar(
                                "geom/s_var_min", s_var.min().item(), total_steps
                            )
                            if T > 1:
                                dh = h_seq[:, 1:] - h_seq[:, :-1]
                                ds = s_seq[:, 1:] - s_seq[:, :-1]
                                step_len = torch.sqrt(
                                    dh.pow(2).mean(dim=-1) + ds.pow(2).mean(dim=-1)
                                )
                                writer.add_scalar(
                                    "geom/step_len_mean",
                                    step_len.mean().item(),
                                    total_steps,
                                )
                                writer.add_scalar(
                                    "geom/step_len_std",
                                    step_len.std().item(),
                                    total_steps,
                                )

                            max_pts = args.geom_log_max_pts
                            n_pts = min(max_pts, s_flat.size(0))
                            if n_pts >= 2:
                                idx = torch.randperm(s_flat.size(0), device=device)[
                                    :n_pts
                                ]
                                s_sub = s_flat[idx]
                                dists = torch.cdist(s_sub, s_sub)
                                dists.fill_diagonal_(float("inf"))
                                nn = dists.min(dim=1).values
                                writer.add_scalar(
                                    "geom/s_nn_mean",
                                    nn.mean().item(),
                                    total_steps,
                                )
                                writer.add_scalar(
                                    "geom/s_nn_p90",
                                    torch.quantile(nn, 0.90).item(),
                                    total_steps,
                                )
                                pdist_mean = dists[torch.isfinite(dists)].mean().item()
                                writer.add_scalar(
                                    "geom/s_pdist_mean", pdist_mean, total_steps
                                )

                        max_pts = args.geom_log_max_pts
                        h_prev_flat = h_prev_seq.reshape(-1, h_prev_seq.size(-1))
                        s_prev_flat = s_prev_seq.reshape(-1, s_prev_seq.size(-1))
                        a_prev_flat = a_prev_seq.reshape(-1, a_prev_seq.size(-1))
                        n_pts = min(max_pts, h_prev_flat.size(0))
                        if n_pts > 0:
                            idx = torch.randperm(h_prev_flat.size(0), device=device)[
                                :n_pts
                            ]
                            h_sub = h_prev_flat[idx]
                            s_sub = s_prev_flat[idx]
                            a_sub = a_prev_flat[idx]
                            v_h = _rademacher_like(h_sub)
                            v_s = _rademacher_like(s_sub)
                            v_a = _rademacher_like(a_sub)
                            jv2 = dynamics_pullback_len_jvp(
                                rssm_jvp,
                                h_sub,
                                s_sub,
                                a_sub,
                                v_h,
                                v_s,
                                v_a,
                                create_graph=False,
                            )
                            v2 = (
                                v_h.pow(2).mean(dim=1)
                                + v_s.pow(2).mean(dim=1)
                                + v_a.pow(2).mean(dim=1)
                            )
                            iso_ratio = (jv2 / v2.clamp_min(1e-8)).mean().item()
                            writer.add_scalar(
                                "geom/dynpb_iso_ratio", iso_ratio, total_steps
                            )


        # (optional) exploration decay
        if args.expl_decay > 0:
            expl = max(args.expl_min, expl - args.expl_decay)

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
            writer.add_scalar("eval/mean_return", eval_mean, ep + 1)
            writer.add_scalar("eval/std_return", eval_std, ep + 1)

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
    writer.add_scalar("eval/mean_return", mean_ret, args.max_episodes)
    writer.add_scalar("eval/std_return", std_ret, args.max_episodes)
    writer.close()

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
    p.add_argument("--dyn_pb_reg_weight", type=float, default=1e-3)
    p.add_argument("--dyn_pb_reg_projections", type=int, default=2)
    p.add_argument("--dyn_pb_bisim_weight", type=float, default=0.03)
    p.add_argument("--dyn_pb_max_pts", type=int, default=256)
    p.add_argument("--geom_log_max_pts", type=int, default=256)

    p.add_argument("--geom_iso_max_pts", type=int, default=256)
    p.add_argument(
        "--geom_iso_every", type=int, default=1
    )  # run iso JVP every N updates
    p.add_argument("--eta_max_pts", type=int, default=256)
    p.add_argument("--knn_gate_max_pts", type=int, default=512)
    p.add_argument("--geom_pen_cap", type=float, default=0.0)  # 0 disables cap

    # Exploration
    p.add_argument("--expl_amount", type=float, default=0.3)
    p.add_argument("--expl_decay", type=float, default=0.0)
    p.add_argument("--expl_min", type=float, default=0.0)


    # Geometry embedding
    p.add_argument("--geo_dim", type=int, default=32)
    p.add_argument("--geo_pos_k", type=int, default=3)
    #p.add_argument("--geo_neg_k", type=int, default=12)
    #p.add_argument("--geo_margin", type=float, default=0.25)

    # Bank / frontier bonus
    p.add_argument("--geo_bank_capacity", type=int, default=500_000)
    p.add_argument("--geo_bank_sample", type=int, default=512)
    p.add_argument("--geo_frontier_tau", type=float, default=0.10)
    p.add_argument("--geo_frontier_cap", type=float, default=1.0)

    # Planning constraint
    p.add_argument("--geo_step_radius", type=float, default=0.35)

    p.add_argument("--geo_lr", type=float, default=3e-4,
               help="Separate lr for GeoEncoder (higher than model_lr)")
    p.add_argument("--geo_margin", type=float, default=0.6,   # was 0.25
                help="Ranking margin; 0.6 works on unit sphere")
    p.add_argument("--geo_neg_k", type=int, default=8,        # was 12
                help="Steps apart to count as negative — reduce if seq_len=50")
    p.add_argument("--geo_uniformity_weight", type=float, default=0.1)

    # Three activation phases (in env steps)
    p.add_argument("--geo_learn_after_steps", type=int, default=2_000)
    p.add_argument("--geo_reward_after_steps", type=int, default=10_000)
    p.add_argument("--geo_plan_after_steps", type=int, default=20_000)
    p.add_argument("--geo_plan_penalty", action="store_true", default=False)

    # Evaluation
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument(
        "--eval_interval", type=int, default=50, help="Run eval every N episodes"
    )

    p.add_argument(
        "--train_log_interval",
        type=int,
        default=20,
        help="log every N gradient updates",
    )
    p.add_argument("--log_dir", type=str, default="runs")

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

    variants: List[VariantCfg] = [
        #VariantCfg(name="dreamer"),
        VariantCfg(
            name="dreamer+gelato",
            geo_aux_weight=0.50,
            geo_reward_bonus_weight=0.03,
            geo_plan_penalty_weight=1e-3,
            geo_plan_penalty=False,
        ),
        VariantCfg(
            name="dreamer+gelato+plan",
            geo_aux_weight=0.50,
            geo_reward_bonus_weight=0.03,
            geo_plan_penalty_weight=1e-3,
            geo_plan_penalty=True,
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
            if v.name == "dreamer+l2bisim" and seed == 42:
                continue
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
