#!/usr/bin/env python3
"""
Dreamer + Semantic Pullback Geometry (JVP-only)
================================================

This training script is a *new* Dreamer ablation runner that aligns with our
discussion:

Core idea ("true geometry")
--------------------------
Define a stable semantic feature space via a VICReg-trained teacher ψ(o) with an
EMA target. Then learn a semantic head F(z) = F([h,s]) that predicts teacher
features. The induced pullback metric is:

    G(z) = J_F(z)^T J_F(z)

We never build G explicitly. We use *directional* JVPs to compute:
    length(z, dz) = || J_F(z) dz ||^2

We stabilize the metric with a conditioning-band regularizer on Hutchinson trace
estimates (Exp 6 style), and we make the policy "use" geometry by adding a
hinged, ramped, optionally kNN-gated penalty on semantic pullback lengths in
imagination (Phase 1 Exp 1/2 style).


"""

import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw"

import argparse
import copy
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
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
#  Formatting
# ===============================


def _f(x, n=4):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return str(x)


# ===============================
#  Small state helpers
# ===============================

import math


class ScalarEMA:
    def __init__(self, beta=0.99, min_std=1e-4):
        self.mean = 0.0
        self.var = 0.0
        self.beta = beta
        self.inited = False
        self.min_std = min_std

    def update(self, x: float) -> tuple[float, float]:
        x = float(x)
        if not self.inited:
            self.mean = x
            self.var = 0.0
            self.inited = True
        else:
            diff = x - self.mean
            self.mean = self.mean + (1 - self.beta) * diff
            self.var = self.beta * (self.var + (1 - self.beta) * diff**2)
        return self.mean, self.std

    @property
    def std(self) -> float:
        return max(math.sqrt(self.var), self.min_std)


import threading


class PrefetchSampler:
    """Samples the next batch in a background thread while GPU processes current batch."""

    def __init__(self, replay, batch_size, seq_len, device):
        self.replay = replay
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.device = device
        self._batch = None
        self._thread = None
        self._prefetch()

    def _fetch(self):
        batch = self.replay.sample_sequences(self.batch_size, self.seq_len)
        self._batch = (
            torch.as_tensor(batch.obs)
            .pin_memory()
            .to(self.device, non_blocking=True)
            .float(),
            torch.as_tensor(batch.actions)
            .pin_memory()
            .to(self.device, non_blocking=True),
            torch.as_tensor(batch.rews).pin_memory().to(self.device, non_blocking=True),
            torch.as_tensor(batch.dones)
            .pin_memory()
            .to(self.device, non_blocking=True),
        )

    def _prefetch(self):
        self._thread = threading.Thread(target=self._fetch, daemon=True)
        self._thread.start()

    def next(self):
        self._thread.join()  # wait for current prefetch to finish
        batch = self._batch
        self._prefetch()  # immediately start fetching next batch
        return batch


class LatentBank:
    """CPU ring buffer for latent support (used for kNN gating / OOD diagnostics)."""

    def __init__(self, capacity: int, ds: int, device="cpu"):
        self.capacity = int(capacity)
        self.ds = int(ds)
        self.device = device
        self.buf = torch.empty((capacity, ds), dtype=torch.float32, device=device)
        self.n = 0
        self.i = 0

    def add(self, x: torch.Tensor):
        x = x.detach().to(self.device, dtype=torch.float32)
        if x.ndim != 2 or x.size(1) != self.ds:
            raise ValueError("LatentBank.add: wrong shape")
        n = x.shape[0]
        end = self.i + n
        if end <= self.capacity:
            self.buf[self.i : end].copy_(x)
        else:
            # wrap around
            first = self.capacity - self.i
            self.buf[self.i :].copy_(x[:first])
            self.buf[: end - self.capacity].copy_(x[first:])
        self.i = end % self.capacity
        self.n = min(self.n + n, self.capacity)

    def sample(self, m: int, device: torch.device) -> Optional[torch.Tensor]:
        if self.n == 0:
            return None
        idx = torch.randint(0, self.n, (m,), device=torch.device("cpu"))
        return self.buf[idx].to(device)


def subsample_rows(x: torch.Tensor, max_rows: int) -> torch.Tensor:
    N = x.shape[0]
    if max_rows is None or max_rows <= 0 or N <= max_rows:
        return x
    idx = torch.randperm(N, device=x.device)[:max_rows]
    return x[idx]


def subsample_together(max_rows: int, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Subsample multiple tensors along dim0 using the same indices."""
    if len(xs) == 0:
        return tuple()
    N = xs[0].shape[0]
    for x in xs[1:]:
        if x.shape[0] != N:
            raise ValueError("subsample_together: tensors must share dim0")
    if max_rows is None or max_rows <= 0 or N <= max_rows:
        return xs
    idx = torch.randperm(N, device=xs[0].device)[:max_rows]
    return tuple(x[idx] for x in xs)


def approx_knn_stats(x: torch.Tensor, bank_sub: torch.Tensor) -> tuple[float, float]:
    with autocast(device_type="cuda", enabled=False):
        d = torch.cdist(x, bank_sub).min(dim=1).values
    return d.mean().item(), torch.quantile(d, 0.90).item()


# ===============================
#  VICReg teacher (ψ) + augmentations
# ===============================


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Random shift augmentation (pad+crop). x: [N,C,H,W]"""
    if pad <= 0:
        return x
    N, C, H, W = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    # sample shifts
    max_y = 2 * pad
    max_x = 2 * pad
    offs_y = torch.randint(0, max_y + 1, (N,), device=x.device)
    offs_x = torch.randint(0, max_x + 1, (N,), device=x.device)
    out = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    for i in range(N):
        y = int(offs_y[i])
        x0 = int(offs_x[i])
        out[i] = x_pad[i, :, y : y + H, x0 : x0 + W]
    return out


def phi_augment(x: torch.Tensor, pad: int = 4, noise_std: float = 0.01) -> torch.Tensor:
    """Light augmentations for VICReg teacher."""
    y = random_shift(x, pad=pad)
    if noise_std and noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return y


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    inv_w: float = 25.0,
    var_w: float = 25.0,
    cov_w: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Standard VICReg (invariance + variance + covariance)."""
    # invariance
    inv = F.mse_loss(z1, z2)

    # variance
    def var_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    var = var_term(z1) + var_term(z2)

    # covariance
    def cov_term(z):
        z = z - z.mean(dim=0)
        N = z.size(0)
        cov = (z.T @ z) / max(1, N - 1)
        return off_diagonal(cov).pow(2).mean()

    cov = cov_term(z1) + cov_term(z2)

    return inv_w * inv + var_w * var + cov_w * cov


@torch.no_grad()
def ema_update(target: nn.Module, online: nn.Module, tau: float):
    for p_t, p in zip(target.parameters(), online.parameters()):
        p_t.data.mul_(tau).add_(p.data, alpha=(1.0 - tau))


class TeacherNet(nn.Module):
    """Image teacher network used for VICReg; returns embedding vectors."""

    def __init__(self, in_channels: int, embed_dim: int, proj_dim: int):
        super().__init__()
        self.enc = ConvEncoder(embedding_size=embed_dim, in_channels=in_channels)
        # projector (helps VICReg)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ELU(),
            nn.Linear(proj_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N,C,H,W]
        e = self.enc(x)
        z = self.proj(e)
        z = F.layer_norm(z, (z.size(-1),))
        return z


# ===============================
#  Semantic head F(z) and pullback primitives
# ===============================


class SemanticHead(nn.Module):
    """F(z) where z = [h,s] concatenation. Output dim matches teacher proj_dim."""

    def __init__(
        self, deter_dim: int, stoch_dim: int, out_dim: int, hidden_dim: int = 512
    ):
        super().__init__()
        in_dim = deter_dim + stoch_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        y = self.net(z)
        return F.layer_norm(y, (y.size(-1),))


def directional_pullback_len_jvp(
    Fnet: nn.Module,
    z: torch.Tensor,
    dz: torch.Tensor,
    *,
    create_graph: bool,
    reduce: str = "mean",  # "mean" or "sum"
) -> torch.Tensor:
    with autocast(device_type="cuda", enabled=False):
        """Returns per-sample || J_F(z) dz ||^2. z,dz: [N,Dz]."""
        # z = z.detach().requires_grad_(True) caller needs actor grad
        dz = dz.detach()
        _, jv = torch.autograd.functional.jvp(
            lambda zz: Fnet(zz), (z,), (dz,), create_graph=create_graph
        )
        if reduce == "sum":
            return jv.pow(2).sum(dim=1)
        return jv.pow(2).mean(dim=1)


def hutch_trace_hat(
    Fnet: nn.Module,
    z: torch.Tensor,
    num_projections: int = 1,
    create_graph: bool = False,
) -> torch.Tensor:
    """Hutchinson trace estimator for tr(J^T J) via Rademacher probes. Returns [N]."""
    vals = []
    for _ in range(int(num_projections)):
        v = torch.empty_like(z).bernoulli_(0.5).mul_(2.0).add_(-1.0)  # ±1
        val = directional_pullback_len_jvp(
            Fnet, z, v, create_graph=create_graph, reduce="sum"
        )
        vals.append(val)
    return torch.stack(vals, dim=0).mean(dim=0)


# ===============================
#  kNN gate helpers
# ===============================


def knn_gate_weights(
    s_prev: torch.Tensor,
    bank_sub: torch.Tensor,
    tau: float,
    *,
    kind: str = "sigmoid",  # "sigmoid" or "relu"
    k: float = 8.0,
    max_gate_pts: int = 512,
) -> torch.Tensor:
    """
    Compute gating weights w for a subset of points (to reduce cdist cost).
    Returns w: [N], with zeros for unsampled indices when N > max_gate_pts.
    """
    N = s_prev.shape[0]
    device = s_prev.device
    w = torch.zeros(N, device=device)
    if N == 0:
        return w

    def gate(d):
        if kind == "relu":
            return torch.relu(d - float(tau))
        if kind == "sigmoid":
            return torch.sigmoid(float(k) * (d - float(tau)))
        raise ValueError(f"Unknown gate kind: {kind}")

    if max_gate_pts is not None and max_gate_pts > 0 and N > max_gate_pts:
        idx = torch.randperm(N, device=device)[:max_gate_pts]
        s_sub = s_prev[idx]
        with autocast(device_type="cuda", enabled=False):
            d = torch.cdist(s_sub, bank_sub).min(dim=1).values
        w[idx] = gate(d)
    else:
        with autocast(device_type="cuda", enabled=False):
            d = torch.cdist(s_prev, bank_sub).min(dim=1).values
        w = gate(d)

    return w


# ===============================
#  Dreamer returns helpers
# ===============================


def compute_lambda_returns(rewards, values, discounts, lambda_=0.95):
    """rewards: [B,H], values:[B,H+1], discounts:[B,H] -> returns:[B,H]"""
    B, H = rewards.shape
    returns = torch.zeros_like(rewards)
    next_value = values[:, -1]
    for t in reversed(range(H)):
        next_value = rewards[:, t] + discounts[:, t] * (
            (1 - lambda_) * values[:, t + 1] + lambda_ * next_value
        )
        returns[:, t] = next_value
    return returns


def compute_discount_weights(discounts):
    """discounts: [B,H] -> weights [B,H]"""
    B, H = discounts.shape
    w = torch.ones_like(discounts)
    for t in range(1, H):
        w[:, t] = w[:, t - 1] * discounts[:, t - 1]
    return w


def l2_bisim_loss(z, zn, r, gamma=0.99):
    """
    Local bisim style target:
        ||z - z'|| ≈ |r-r'| + γ ||zn - zn'||
    Here we use random pairing by shifting batch indices (cheap).
    """
    B = z.size(0)
    perm = torch.randperm(B, device=z.device)
    z2 = z[perm]
    zn2 = zn[perm]
    r2 = r[perm]
    dz = torch.norm(z - z2.detach(), p=2, dim=1)
    with torch.no_grad():
        dnext = torch.norm(zn - zn2, p=2, dim=1)
        dr = torch.abs(r - r2)
        target = dr + gamma * dnext
    return F.mse_loss(dz, target)


def beta_schedule(
    total_steps: int, ramp_start: int, ramp_dur: int, off_step: Optional[int]
) -> float:
    up = (total_steps - ramp_start) / max(1, ramp_dur)
    up = float(np.clip(up, 0.0, 1.0))
    if off_step is None:
        return up
    if total_steps >= off_step:
        return 0.0
    off_start = max(
        ramp_start + ramp_dur, off_step - 100_000
    )  # default 100k decay window
    if total_steps <= off_start:
        return up
    down = 1.0 - (total_steps - off_start) / max(1, off_step - off_start)
    down = float(np.clip(down, 0.0, 1.0))
    return up * down


# ===============================
#  Ablation config
# ===============================


@dataclass
class VariantCfg:
    name: str

    # baseline regularizers
    l2_bisim_weight: float = 0.0

    # semantic teacher + head (representation shaping)
    use_teacher: bool = False
    teacher_proj_dim: int = 256
    semantic_fit_weight: float = 0.0  # align F(z) to teacher ψ(o)

    # metric conditioning band (Exp 6 style)
    semantic_cond_weight: float = 0.0
    cond_proj: int = 1
    cond_lo_mult: float = 0.5
    cond_hi_mult: float = 2.0

    # actor geometry usage (Exp 1/2 style)
    actor_geom_weight: float = 0.0
    actor_ramp_start: int = 30_000
    actor_ramp_dur: int = 30_000
    actor_off_step: Optional[int] = None

    # gating (Exp 2)
    use_gate: bool = False
    gate_kind: str = "sigmoid"  # "sigmoid" or "relu"
    gate_k: float = 8.0
    tau_quantile: float = 0.70

    # caps / compute
    geom_pen_cap: float = 0.0  # 0 disables cap


# ===============================
#  Evaluation
# ===============================


@torch.no_grad()
def evaluate_actor_policy(
    env, encoder, rssm, actor, args, action_repeat: int, episodes: int = 5
):
    device = next(actor.parameters()).device
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        obs_t = (
            torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        preprocess_img(obs_t, depth=args.bit_depth)
        e0 = encoder(obs_t)
        h, s = rssm.get_init_state(e0)
        total = 0.0
        while not done:
            a, _ = actor.get_action(h, s, deterministic=True)
            a_np = a.squeeze(0).cpu().numpy().astype(np.float32)
            obs, r, term, trunc, _ = env.step(a_np, repeat=action_repeat)
            done = bool(term or trunc)
            total += float(r)
            obs_t = (
                torch.tensor(
                    np.ascontiguousarray(obs), dtype=torch.float32, device=device
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            preprocess_img(obs_t, depth=args.bit_depth)
            e = encoder(obs_t)
            h = rssm.deterministic_state_fwd(h, s, a)
            pm, ps = rssm.state_posterior(h, e)
            s = pm  # use mean for eval belief update
        returns.append(total)
    return float(np.mean(returns)), float(np.std(returns))


# ===============================
#  Main training for one seed
# ===============================


def run_one_seed(args, cfg: VariantCfg, seed: int) -> Dict[str, float]:
    set_seed(seed)
    device = get_device()
    amp_enabled = device.type == "cuda"
    amp_dtype = (
        torch.bfloat16
        if (amp_enabled and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    scaler = torch.amp.GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    def amp():
        return autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled)

    def backward_and_step(loss, optimizer, parameters):
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm)
            optimizer.step()

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

    # ---------------- Models ----------------
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

    # Optional semantic teacher + semantic head
    teacher = teacher_tgt = None
    sem_head = None
    teacher_optim = None

    if (
        cfg.use_teacher
        or cfg.semantic_fit_weight > 0
        or cfg.semantic_cond_weight > 0
        or cfg.actor_geom_weight > 0
    ):
        cfg.use_teacher = True  # force teacher on if any semantic geometry is used
        teacher = TeacherNet(
            in_channels=C,
            embed_dim=args.teacher_embed_dim,
            proj_dim=cfg.teacher_proj_dim,
        ).to(device)
        teacher_tgt = copy.deepcopy(teacher).to(device)
        teacher_tgt.eval()
        for p in teacher_tgt.parameters():
            p.requires_grad_(False)
        teacher_optim = torch.optim.Adam(
            teacher.parameters(), lr=args.teacher_lr, eps=args.adam_eps
        )

        sem_head = SemanticHead(
            args.deter_dim,
            args.stoch_dim,
            out_dim=cfg.teacher_proj_dim,
            hidden_dim=args.semantic_hidden_dim,
        ).to(device)

        sem_head_tgt = copy.deepcopy(sem_head).to(device)
        sem_head_tgt.eval()
        for p in sem_head_tgt.parameters():
            p.requires_grad_(False)

        sem_head_optim = torch.optim.Adam(
            sem_head.parameters(), lr=args.model_lr, eps=args.adam_eps
        )

    if args.compile and hasattr(torch, "compile"):
        encoder = torch.compile(encoder)
        decoder = torch.compile(decoder)
        rssm = torch.compile(rssm)
        actor = torch.compile(actor)
        value_model = torch.compile(value_model)
        # (Teacher and sem_head are small; compilation usually not worth it.)

    # ---------------- Optims ----------------
    world_params = (
        list(encoder.parameters())
        + list(decoder.parameters())
        + list(rssm.parameters())
        + list(reward_model.parameters())
        + list(cont_model.parameters())
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

    # EMAs for hinge/gate/conditioning
    eta_ema = ScalarEMA(beta=0.99)
    tau_ema = ScalarEMA(beta=0.99)
    trace_ema = ScalarEMA(beta=0.99)

    latent_bank = LatentBank(
        capacity=args.bank_capacity, ds=args.stoch_dim, device="cpu"
    )

    # ---------------- Seed buffer ----------------
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
                reward=float(total_reward),
                next_obs=None,
                done=done,
            )
            obs = next_obs

    total_steps = 0
    expl = args.expl_amount
    t_start = time.time()
    train_updates = 0

    # ---------------- Online episodes ----------------
    for ep in range(args.max_episodes):
        obs, _ = env.reset()
        done = False

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
            with torch.no_grad(), amp():
                a_t, _ = actor.get_action(h, s, deterministic=False)
                if expl > 0:
                    a_t = torch.clamp(a_t + expl * torch.randn_like(a_t), -1.0, 1.0)
                a_np = a_t.squeeze(0).cpu().numpy().astype(np.float32)

            # step env
            next_obs, total_reward, term, trunc, _ = env.step(
                a_np, repeat=action_repeat
            )
            done = bool(term or trunc)

            # add to replay
            replay.add(
                obs=np.ascontiguousarray(obs, np.uint8),
                action=np.asarray(a_np, np.float32),
                reward=float(total_reward),
                next_obs=None,
                done=done,
            )

            # belief update
            obs = next_obs
            obs_t = (
                torch.tensor(
                    np.ascontiguousarray(obs), dtype=torch.float32, device=device
                )
                .permute(2, 0, 1)
                .unsqueeze(0)
            )
            preprocess_img(obs_t, depth=args.bit_depth)
            with torch.no_grad(), amp():
                e = encoder(obs_t)
                h = rssm.deterministic_state_fwd(h, s, a_t)
                pm, ps = rssm.state_posterior(h, e)
                s = pm + torch.randn_like(ps) * ps

            total_steps += action_repeat

            # ---------------- Training ----------------
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
                if teacher is not None:
                    teacher.train()
                if sem_head is not None:
                    sem_head.train()

                prefetcher = PrefetchSampler(
                    replay, args.batch_size, args.seq_len + 1, device
                )

                for _ in range(args.train_steps):
                    obs_seq, act_seq, rew_seq, done_seq = prefetcher.next()
                    """
                    batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)
                    obs_seq = torch.as_tensor(
                        batch.obs, device=device
                    ).float()  # [B,T+1,H,W,C]
                    act_seq = torch.as_tensor(batch.actions, device=device)  # [B,T+1,A]
                    rew_seq = torch.as_tensor(batch.rews, device=device)  # [B,T+1]
                    done_seq = torch.as_tensor(batch.dones, device=device)  # [B,T+1]
                    """

                    B, T1 = rew_seq.shape
                    T = T1 - 1

                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()  # [B,T+1,C,H,W]
                    preprocess_img(x, depth=args.bit_depth)

                    # --- teacher VICReg update (representation space ψ) ---
                    vic_loss_val = torch.zeros((), device=device)
                    if teacher is not None and (
                        train_updates % max(1, args.teacher_every) == 0
                    ):
                        imgs = x[:, 1 : T + 1].reshape(-1, C, H, W)
                        imgs = subsample_rows(imgs, args.teacher_batch)
                        x1 = phi_augment(
                            imgs,
                            pad=args.teacher_aug_pad,
                            noise_std=args.teacher_aug_noise,
                        )
                        x2 = phi_augment(
                            imgs,
                            pad=args.teacher_aug_pad,
                            noise_std=args.teacher_aug_noise,
                        )
                        with amp():
                            z1 = teacher(x1)
                            z2 = teacher(x2)
                            vic_loss_val = vicreg_loss(
                                z1,
                                z2,
                                inv_w=args.vic_inv_w,
                                var_w=args.vic_var_w,
                                cov_w=args.vic_cov_w,
                            )
                        teacher_optim.zero_grad(set_to_none=True)
                        """
                        vic_loss_val.backward()
                        torch.nn.utils.clip_grad_norm_(
                            teacher.parameters(), args.grad_clip_norm
                        )
                        teacher_optim.step()
                        """
                        if scaler.is_enabled():
                            scaler.scale(vic_loss_val).backward()
                            scaler.unscale_(teacher_optim)
                            torch.nn.utils.clip_grad_norm_(
                                teacher.parameters(), args.grad_clip_norm
                            )
                            scaler.step(teacher_optim)
                            scaler.update()
                        else:
                            vic_loss_val.backward()
                            torch.nn.utils.clip_grad_norm_(
                                teacher.parameters(), args.grad_clip_norm
                            )
                            teacher_optim.step()
                        ema_update(teacher_tgt, teacher, tau=args.teacher_ema_tau)

                    # ------------------- World model -------------------
                    with amp():
                        e_t = bottle(encoder, x)  # [B,T+1,E]
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

                        h_seq = torch.stack(h_list, dim=1)  # [B,T,Dh]
                        s_seq = torch.stack(s_list, dim=1)  # [B,T,Ds]

                        pri_m = torch.stack([p[0] for p in pri_list], dim=0)
                        pri_s = torch.stack([p[1] for p in pri_list], dim=0)
                        pos_m = torch.stack([p[0] for p in post_list], dim=0)
                        pos_s = torch.stack([p[1] for p in post_list], dim=0)

                        with autocast(device_type="cuda", enabled=False):
                            kld = torch.max(
                                kl_divergence(
                                    Normal(pos_m, pos_s), Normal(pri_m, pri_s)
                                ).sum(-1),
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

                        # L2 bisim (optional baseline)
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

                        # Semantic fit: F([h,s]) matches teacher ψ(o)
                        sem_fit = torch.zeros((), device=device)
                        sem_cos = torch.zeros((), device=device)
                        if sem_head is not None and cfg.semantic_fit_weight > 0:
                            with torch.no_grad():
                                imgs = x[:, 1 : T + 1].reshape(-1, C, H, W)
                                tgt = teacher_tgt(imgs)  # [B*T, D]
                            z_flat = torch.cat([h_seq, s_seq], dim=-1).reshape(
                                -1, args.deter_dim + args.stoch_dim
                            )
                            pred = sem_head(z_flat)
                            sem_fit = F.mse_loss(pred, tgt)
                            sem_cos = F.cosine_similarity(pred, tgt, dim=-1).mean()

                    # Metric conditioning band on trace(G)
                    sem_cond = torch.zeros((), device=device)
                    trace_mean = torch.zeros((), device=device)
                    if sem_head is not None and cfg.semantic_cond_weight > 0:
                        z_flat = torch.cat([h_seq, s_seq], dim=-1).reshape(
                            -1, args.deter_dim + args.stoch_dim
                        )
                        z_sub = subsample_rows(z_flat, args.cond_max_pts)
                        # conditioning regularizes the online head's geometry
                        tr = hutch_trace_hat(
                            sem_head,
                            z_sub,
                            num_projections=cfg.cond_proj,
                            create_graph=True,
                        )
                        tr_mean = tr.mean()
                        trace_mean = tr_mean.detach()
                        # Update EMA (no grad)
                        tr_ema_val, _ = trace_ema.update(float(tr_mean.detach().item()))
                        lo = cfg.cond_lo_mult * tr_ema_val
                        hi = cfg.cond_hi_mult * tr_ema_val
                        sem_cond = (F.relu(lo - tr) + F.relu(tr - hi)).mean()

                    model_loss = (
                        rec_loss
                        + args.kl_weight * kld
                        + rew_loss
                        + args.cont_weight * cont_loss
                        + cfg.l2_bisim_weight * bisim_val
                    )

                    model_optim.zero_grad(set_to_none=True)
                    backward_and_step(model_loss, model_optim, world_params)

                    if sem_head is not None:
                        sem_loss = (cfg.semantic_fit_weight * sem_fit) + (
                            cfg.semantic_cond_weight * sem_cond
                        )
                        if sem_loss.item() > 0:
                            sem_head_optim.zero_grad(set_to_none=True)
                            backward_and_step(
                                sem_loss, sem_head_optim, sem_head.parameters()
                            )
                            sht = 0.99
                            ema_update(sem_head_tgt, sem_head, tau=sht)

                    # --- bank update ---
                    with torch.no_grad():
                        s_flat_cpu = (
                            s_seq.reshape(-1, args.stoch_dim).detach().to("cpu")
                        )
                        if s_flat_cpu.size(0) > args.bank_add_max_pts:
                            idx = torch.randint(
                                0, s_flat_cpu.size(0), (args.bank_add_max_pts,)
                            )
                            latent_bank.add(s_flat_cpu[idx])
                        else:
                            latent_bank.add(s_flat_cpu)

                    # --- η update from posterior semantic lengths ---
                    eta = 0.0
                    if sem_head is not None and cfg.actor_geom_weight > 0:
                        with torch.no_grad():
                            h_prev = h_seq[:, :-1].reshape(-1, args.deter_dim)
                            s_prev = s_seq[:, :-1].reshape(-1, args.stoch_dim)
                            h_next = h_seq[:, 1:].reshape(-1, args.deter_dim)
                            s_next = s_seq[:, 1:].reshape(-1, args.stoch_dim)
                            z_prev = torch.cat([h_prev, s_prev], dim=-1)
                            z_next = torch.cat([h_next, s_next], dim=-1)
                            dz = z_next - z_prev
                            z_prev, dz = subsample_together(
                                args.eta_max_pts, z_prev, dz
                            )
                        with no_param_grads(sem_head_tgt):
                            len_sub = (
                                directional_pullback_len_jvp(
                                    sem_head_tgt,
                                    z_prev,
                                    dz,
                                    create_graph=False,
                                    reduce="mean",
                                )
                                .mean()
                                .item()
                            )
                        eta, _ = eta_ema.update(len_sub)
                    else:
                        eta = eta_ema.mean if eta_ema.inited else 0.0

                    # --- τ update from posterior kNN distances ---
                    tau = tau_ema.mean if tau_ema.inited else 0.0
                    if cfg.use_gate and latent_bank.n > 1000:
                        with torch.no_grad():
                            s_post = s_seq.reshape(-1, args.stoch_dim)
                            s_post = subsample_rows(s_post, args.tau_max_pts)
                            bank_sub = latent_bank.sample(
                                m=args.bank_query_m, device=device
                            )
                            if bank_sub is not None:
                                with autocast(device_type="cuda", enabled=False):
                                    d = torch.cdist(s_post, bank_sub).min(dim=1).values
                                tau_batch = torch.quantile(
                                    d, float(cfg.tau_quantile)
                                ).item()
                                tau, _ = tau_ema.update(tau_batch)

                    # ------------------- Actor-Critic (imagination) -------------------
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
                        with amp():
                            h_im_list = [h0]
                            s_im_list = [s0]
                            h_im, s_im = h0, s0
                            for _ in range(args.imagination_horizon):
                                a_im, _ = actor.get_action(
                                    h_im, s_im, deterministic=False
                                )
                                h_im = rssm.deterministic_state_fwd(h_im, s_im, a_im)
                                s_im = rssm.state_prior(h_im, sample=True)
                                h_im_list.append(h_im)
                                s_im_list.append(s_im)

                            h_imag = torch.stack(h_im_list, dim=1)  # [B_im,H+1,Dh]
                            s_imag = torch.stack(s_im_list, dim=1)  # [B_im,H+1,Ds]

                            rewards_im = bottle(
                                reward_model, h_imag[:, 1:], s_imag[:, 1:]
                            )  # [B_im,H]
                            cont_logits_im = bottle(
                                cont_model, h_imag[:, 1:], s_imag[:, 1:]
                            )
                            pcont = torch.sigmoid(cont_logits_im).clamp(0.0, 1.0)
                            discounts = effective_gamma * pcont

                    with torch.no_grad():
                        values_tgt = bottle(value_model, h_imag, s_imag)
                        lam_ret_tgt = compute_lambda_returns(
                            rewards_im.detach(),
                            values_tgt,
                            discounts.detach(),
                            lambda_=args.lambda_,
                        )
                        w_tgt = compute_discount_weights(discounts.detach())

                    with amp():
                        v_pred = bottle(value_model, h_imag.detach(), s_imag.detach())
                    value_loss = ((v_pred[:, :-1] - lam_ret_tgt) ** 2 * w_tgt).mean()
                    value_optim.zero_grad(set_to_none=True)
                    """
                    value_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        value_model.parameters(), args.grad_clip_norm
                    )
                    value_optim.step()
                    """
                    backward_and_step(value_loss, value_optim, value_model.parameters())

                    with amp():
                        # Actor loss
                        dist = actor.get_dist(
                            h_imag[:, :-1].detach(), s_imag[:, :-1].detach()
                        )
                        entropy = dist.entropy().sum(dim=-1).mean()

                    with no_param_grads(value_model), amp():
                        values_for_actor = bottle(value_model, h_imag, s_imag)
                    w_actor = compute_discount_weights(discounts.detach())
                    lam_ret_actor = compute_lambda_returns(
                        rewards_im, values_for_actor, discounts, lambda_=args.lambda_
                    )
                    actor_loss = (
                        -(w_actor.detach() * lam_ret_actor).mean()
                        - args.actor_entropy_scale * entropy
                    )

                    # Geometry usage in actor objective (true semantic pullback)
                    pen = torch.zeros((), device=device)
                    gate_frac = 0.0
                    beta_eff = 0.0

                    if sem_head is not None and cfg.actor_geom_weight > 0:
                        sched = beta_schedule(
                            total_steps,
                            cfg.actor_ramp_start,
                            cfg.actor_ramp_dur,
                            cfg.actor_off_step,
                        )
                        beta_eff = cfg.actor_geom_weight * sched
                        if beta_eff > 0:
                            h_prev = h_imag[:, :-1].reshape(-1, args.deter_dim)
                            s_prev = s_imag[:, :-1].reshape(-1, args.stoch_dim)
                            h_next = h_imag[:, 1:].reshape(-1, args.deter_dim)
                            s_next = s_imag[:, 1:].reshape(-1, args.stoch_dim)
                            z_prev = torch.cat([h_prev, s_prev], dim=-1)
                            z_next = torch.cat([h_next, s_next], dim=-1)
                            dz = z_next - z_prev

                            # subsample for compute
                            z_prev, dz, s_prev_sub = subsample_together(
                                args.actor_geom_max_pts, z_prev, dz, s_prev
                            )

                            with no_param_grads(sem_head_tgt):
                                length = directional_pullback_len_jvp(
                                    sem_head_tgt,
                                    z_prev,
                                    dz,
                                    create_graph=True,
                                    reduce="mean",
                                )
                                # Defer penalty until EMA has stabilized (e.g., 100 updates)
                                if train_updates < 100:
                                    pen_standardized = torch.zeros_like(length)
                                else:
                                    # eta_ema.std now safely uses the max(..., min_std) floor
                                    pen_standardized = (length - eta_ema.mean) / (
                                        eta_ema.std + 1e-6
                                    )
                                length_h = torch.relu(pen_standardized)

                            # gate on support distance in s-space
                            if cfg.use_gate and latent_bank.n > 1000:
                                bank_sub = latent_bank.sample(
                                    m=args.bank_query_m, device=device
                                )
                                if bank_sub is not None:
                                    with torch.no_grad():
                                        w = knn_gate_weights(
                                            s_prev_sub.detach(),
                                            bank_sub,
                                            tau,
                                            kind=cfg.gate_kind,
                                            k=cfg.gate_k,
                                            max_gate_pts=args.knn_gate_max_pts,
                                        )
                                    # interpret gfrac for sigmoid as w>0.5
                                    if cfg.gate_kind == "sigmoid":
                                        gate_frac = (w > 0.5).float().mean().item()
                                    else:
                                        gate_frac = (w > 0).float().mean().item()
                                    pen = (w * length_h).mean()
                                else:
                                    pen = length_h.mean()
                                    gate_frac = 0.0
                            else:
                                pen = length_h.mean()
                                gate_frac = 0.0

                            if cfg.geom_pen_cap and cfg.geom_pen_cap > 0:
                                pen = torch.clamp(pen, max=float(cfg.geom_pen_cap))
                            actor_loss = actor_loss + beta_eff * pen

                    actor_optim.zero_grad(set_to_none=True)
                    """
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), args.grad_clip_norm
                    )
                    actor_optim.step()
                    """
                    backward_and_step(actor_loss, actor_optim, actor.parameters())

                    train_updates += 1

                    # ------------------- Diagnostics -------------------
                    if args.train_log_interval > 0 and (
                        train_updates % args.train_log_interval == 0
                    ):
                        d_post_mean = d_post_p90 = None
                        d_imag_mean = d_imag_p90 = None
                        ood_ratio = None
                        if latent_bank.n > 1000:
                            with torch.no_grad():
                                bank_sub = latent_bank.sample(
                                    m=args.bank_query_m, device=device
                                )
                                if bank_sub is not None:
                                    s_post = subsample_rows(
                                        s_seq.reshape(-1, args.stoch_dim).detach(), 256
                                    )
                                    s_im = subsample_rows(
                                        s_imag[:, :-1]
                                        .reshape(-1, args.stoch_dim)
                                        .detach(),
                                        256,
                                    )
                                    d_post_mean, d_post_p90 = approx_knn_stats(
                                        s_post, bank_sub
                                    )
                                    d_imag_mean, d_imag_p90 = approx_knn_stats(
                                        s_im, bank_sub
                                    )
                                    ood_ratio = d_imag_mean / (d_post_mean + 1e-8)

                        # semantic length diagnostics (posterior vs imagination)
                        len_post = len_rand = None
                        discriminability_ratio = 0.0
                        if sem_head is not None:
                            with torch.no_grad():
                                # posterior (consistent pairing)
                                hp = h_seq[:, :-1].reshape(-1, args.deter_dim)
                                sp = s_seq[:, :-1].reshape(-1, args.stoch_dim)
                                hp2 = h_seq[:, 1:].reshape(-1, args.deter_dim)
                                sp2 = s_seq[:, 1:].reshape(-1, args.stoch_dim)
                                zp = torch.cat([hp, sp], dim=-1)
                                dzp = torch.cat([hp2, sp2], dim=-1) - zp

                                # NEW: Random pairs (OOD proxy)
                                idx = torch.randperm(zp.shape[0])
                                z_rand = torch.cat([hp, sp[idx]], dim=-1)
                                dz_rand = z_rand - zp

                                zp_sub, dzp_sub, dz_rand_sub = subsample_together(
                                    128, zp, dzp, dz_rand
                                )

                                with no_param_grads(sem_head_tgt):
                                    len_post = (
                                        directional_pullback_len_jvp(
                                            sem_head_tgt,
                                            zp_sub,
                                            dzp_sub,
                                            create_graph=False,
                                            reduce="mean",
                                        )
                                        .mean()
                                        .item()
                                    )

                                    len_rand = (
                                        directional_pullback_len_jvp(
                                            sem_head_tgt,
                                            zp_sub,
                                            dz_rand_sub,
                                            create_graph=False,
                                            reduce="mean",
                                        )
                                        .mean()
                                        .item()
                                    )

                                discriminability_ratio = len_rand / (len_post + 1e-8)

                        msg = (
                            f"[{cfg.name} | seed {seed}] upd {train_updates:6d}  steps={total_steps:7d}  "
                            f"rec={_f(rec_loss)} kl={_f(kld)} rew={_f(rew_loss)} cont={_f(cont_loss)} "
                            f"bisim={_f(bisim_val)} "
                            f"vic={_f(vic_loss_val)} semfit={_f(sem_fit)} cos={_f(sem_cos)} "
                            f"cond={_f(sem_cond)} tr={_f(trace_mean)} "
                            f"eta={_f(eta)} tau={_f(tau)} "
                            f"gbeta={_f(beta_eff, 5)} gpen={_f(pen)} gfrac={_f(gate_frac, 4)} "
                            f"dpost={(f'{d_post_mean:.3f}' if d_post_mean is not None else 'NA')}/{(f'{d_post_p90:.3f}' if d_post_p90 is not None else 'NA')} "
                            f"dimag={(f'{d_imag_mean:.3f}' if d_imag_mean is not None else 'NA')}/{(f'{d_imag_p90:.3f}' if d_imag_p90 is not None else 'NA')} "
                            f"ood={(f'{ood_ratio:.2f}' if ood_ratio is not None else 'NA')} "
                            f"discriminability_ratio={(f'{discriminability_ratio:.2f}' if discriminability_ratio is not None else 'NA')} "
                            f"lpost={(f'{len_post:.3f}' if len_post is not None else 'NA')} lim={(f'{len_rand:.3f}' if len_rand is not None else 'NA')}"
                        )
                        print(msg)

                # expl decay (per collect)
                if args.expl_decay > 0:
                    expl = max(args.expl_min, expl - args.expl_decay)

        # ---------------- Evaluation ----------------
        if (ep + 1) % args.eval_interval == 0:
            encoder.eval()
            rssm.eval()
            actor.eval()
            mean_r, std_r = evaluate_actor_policy(
                env,
                encoder,
                rssm,
                actor,
                args,
                action_repeat,
                episodes=args.eval_episodes,
            )
            elapsed = time.time() - t_start
            print(
                f"[{cfg.name} | seed {seed}] Eval ep {ep + 1}/{args.max_episodes} steps={total_steps}  "
                f"return={mean_r:.2f}±{std_r:.2f}  buf={replay.size}  ({elapsed:.0f}s)"
            )

    # final eval
    encoder.eval()
    rssm.eval()
    actor.eval()
    mean_r, std_r = evaluate_actor_policy(
        env, encoder, rssm, actor, args, action_repeat, episodes=args.eval_episodes
    )
    return {"return_mean": mean_r, "return_std": std_r}


# ===============================
#  CLI
# ===============================


def parse_args():
    p = argparse.ArgumentParser(
        description="Dreamer with semantic pullback geometry (VICReg teacher + JVP)"
    )
    p.add_argument("--env_id", type=str, default="cartpole-swingup")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1])

    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for main models (encoder/decoder/rssm/actor/value).",
    )

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

    # Teacher + semantic head
    p.add_argument(
        "--teacher_embed_dim",
        type=int,
        default=512,
        help="ConvEncoder embedding dim inside teacher net.",
    )
    p.add_argument("--teacher_lr", type=float, default=1e-4)
    p.add_argument(
        "--teacher_every",
        type=int,
        default=2,
        help="Update teacher every N model updates.",
    )
    p.add_argument(
        "--teacher_batch", type=int, default=256, help="Frames per VICReg update."
    )
    p.add_argument("--teacher_ema_tau", type=float, default=0.995)
    p.add_argument("--teacher_aug_pad", type=int, default=4)
    p.add_argument("--teacher_aug_noise", type=float, default=0.01)
    p.add_argument("--vic_inv_w", type=float, default=25.0)
    p.add_argument("--vic_var_w", type=float, default=25.0)
    p.add_argument("--vic_cov_w", type=float, default=1.0)
    p.add_argument("--semantic_hidden_dim", type=int, default=512)

    # Conditioning / geometry compute limits
    p.add_argument("--cond_max_pts", type=int, default=256)
    p.add_argument("--eta_max_pts", type=int, default=256)
    p.add_argument("--actor_geom_max_pts", type=int, default=512)

    # Bank / gating
    p.add_argument("--bank_capacity", type=int, default=50_000)
    p.add_argument("--bank_add_max_pts", type=int, default=512)
    p.add_argument("--bank_query_m", type=int, default=2048)
    p.add_argument("--tau_max_pts", type=int, default=256)
    p.add_argument("--knn_gate_max_pts", type=int, default=512)

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

    # Evaluation / logs
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=50)
    p.add_argument("--train_log_interval", type=int, default=20)

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

    print("Dreamer semantic pullback ablation")
    print(f"Env: {args.env_id} | seeds={args.seeds} | quick={args.quick}")

    # Default ablations ladder (A0 -> A4)
    variants: List[VariantCfg] = [
        VariantCfg(name="dreamer", l2_bisim_weight=0.0, use_teacher=False),
        VariantCfg(
            name="teacher_only",
            l2_bisim_weight=0.0,
            use_teacher=True,
            teacher_proj_dim=256,
            semantic_fit_weight=1.0,
            semantic_cond_weight=0.0,
            actor_geom_weight=0.0,
        ),
        VariantCfg(
            name="semantic_cond_only",
            l2_bisim_weight=0.0,
            use_teacher=True,
            teacher_proj_dim=256,
            semantic_fit_weight=1.0,
            semantic_cond_weight=0.05,
            cond_proj=1,
            actor_geom_weight=0.0,
        ),
        VariantCfg(
            name="semantic_actor_geom_gate",
            l2_bisim_weight=0.0,
            use_teacher=True,
            teacher_proj_dim=256,
            semantic_fit_weight=1.0,
            semantic_cond_weight=0.05,
            cond_proj=1,
            actor_geom_weight=3e-4,
            actor_off_step=200_000,
            use_gate=True,
            gate_kind="sigmoid",
            gate_k=8.0,
            tau_quantile=0.70,
            geom_pen_cap=0.0,
        ),
    ]

    # Run
    for cfg in variants:
        print(f"\n== {cfg.name} ==")
        all_stats = []
        for seed in args.seeds:
            stats = run_one_seed(args, cfg, seed)
            all_stats.append(stats["return_mean"])
        print(
            f"[{cfg.name}] mean_return_over_seeds = {np.mean(all_stats):.2f} ± {np.std(all_stats):.2f}"
        )


if __name__ == "__main__":
    main()
