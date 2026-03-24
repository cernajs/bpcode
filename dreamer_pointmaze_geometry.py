import argparse
import os
from typing import List, Optional

import numpy as np
import torch
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from models import RSSM, Actor, ContinueModel, ConvDecoder, ConvEncoder, RewardModel, ValueModel
from utils import ReplayBuffer, bottle, get_device, preprocess_img, set_seed
from pointmaze_gr_geometry_test import PointMazeMediumDiverseGRWrapper


# =============================================================================
#  PointMaze Dreamer + temporal-distance ensemble intrinsic reward
# =============================================================================
#
# Train an ensemble of temporal-distance heads on oracle-free replay step
# distances (same-episode forward chain). During imagination,
#   r_int ∝ Var_k T_k(h_t, s_t, h_ref, s_ref)
# over random replay references (metric mismatch / bottlenecks → higher disagreement).
#
# With --baseline, the world model + actor run without this module.
# =============================================================================


def _oracle_free_replay_step_distances(
    episode_ids: np.ndarray,
    n_pairs_target: int = 4000,
    max_sources: int = 64,
    rng: Optional[np.random.Generator] = None,
):
    """Same-episode forward chain i→i+1; shortest path along chain gives step counts."""
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


class TemporalDistHead(nn.Module):
    """Predicts replay step distance from concat(h_i, s_i, h_j, s_j) (matches pointmaze_gr_geometry_test_topo)."""

    def __init__(self, deter_dim: int, stoch_dim: int, hidden_dim: int):
        super().__init__()
        in_dim = 2 * (int(deter_dim) + int(stoch_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_i: torch.Tensor, s_i: torch.Tensor, h_j: torch.Tensor, s_j: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h_i, s_i, h_j, s_j], dim=-1)
        return self.net(x).squeeze(-1)


class TemporalDistEnsemble(nn.Module):
    """K independent distance heads; intrinsic signal = Var(pred_1,...,pred_K)."""

    def __init__(self, deter_dim: int, stoch_dim: int, hidden_dim: int, n_heads: int):
        super().__init__()
        self.n_heads = int(max(1, n_heads))
        self.heads = nn.ModuleList(
            [TemporalDistHead(deter_dim, stoch_dim, hidden_dim) for _ in range(self.n_heads)]
        )

    def forward_all(self, h_i: torch.Tensor, s_i: torch.Tensor, h_j: torch.Tensor, s_j: torch.Tensor) -> torch.Tensor:
        """Returns [K, B] predictions."""
        return torch.stack([head(h_i, s_i, h_j, s_j) for head in self.heads], dim=0)


class TemporalIntrinsicModule:
    """FIFO (h,s) replay buffer + temporal-distance ensemble; intrinsic reward = Var_k predictions."""

    def __init__(
        self,
        deter_dim: int,
        stoch_dim: int,
        device: torch.device | None = None,
        max_nodes: int = 2500,
        temporal_hidden: int = 256,
        n_temporal_heads: int = 5,
        n_temporal_ref: int = 8,
    ):
        self.device = device or torch.device("cpu")
        self.max_nodes = int(max_nodes)

        self.temporal_hidden = int(temporal_hidden)
        self.n_temporal_heads = int(max(1, n_temporal_heads))
        self.temporal_ensemble = TemporalDistEnsemble(
            deter_dim, stoch_dim, self.temporal_hidden, self.n_temporal_heads
        ).to(self.device)
        self.temporal_opt = torch.optim.Adam(self.temporal_ensemble.parameters(), lr=3e-4)
        self.n_temporal_ref = int(max(1, n_temporal_ref))
        self._temporal_trained_once = False
        self.tb_last_temporal_loss: float | None = None
        self.tb_last_temporal_train_steps: int = 0

        self._h_nodes: List[np.ndarray] = []
        self._s_nodes: List[np.ndarray] = []
        self._episode_ids: List[int] = []

    def temporal_intrinsic_ready(self) -> bool:
        return self._temporal_trained_once and len(self._h_nodes) >= 8

    def add_batch_trajectories(
        self,
        traj_h: List[np.ndarray],
        traj_s: List[np.ndarray],
        traj_ep_ids: List[int],
    ):
        """Append trajectory steps; FIFO-evict oldest rows when over max_nodes."""
        for ep_h, ep_s, ep_id in zip(traj_h, traj_s, traj_ep_ids):
            T = len(ep_h)
            if T < 2:
                continue
            for t in range(T):
                if len(self._h_nodes) >= self.max_nodes:
                    self._h_nodes.pop(0)
                    self._s_nodes.pop(0)
                    self._episode_ids.pop(0)
                self._h_nodes.append(ep_h[t])
                self._s_nodes.append(ep_s[t])
                self._episode_ids.append(ep_id)

    def train_temporal_ensemble(
        self,
        n_steps: int = 40,
        batch_pairs: int = 512,
        pair_pool: int = 6000,
        seed: int = 0,
    ):
        """Supervised training: each head predicts oracle-free replay step distance."""
        self.tb_last_temporal_loss = None
        self.tb_last_temporal_train_steps = 0
        ep = np.asarray(self._episode_ids, dtype=np.int64)
        if len(ep) < 8:
            return

        rng = np.random.default_rng(int(seed))
        pool = _oracle_free_replay_step_distances(
            ep,
            n_pairs_target=int(pair_pool),
            max_sources=96,
            rng=rng,
        )
        if pool is None:
            return
        ii, jj, dd = pool
        n_pairs = len(ii)
        if n_pairs < 16:
            return

        perm = rng.permutation(n_pairs)
        n_val = max(1, int(round(0.12 * n_pairs)))
        train_mask = np.ones(n_pairs, dtype=bool)
        train_mask[perm[:n_val]] = False
        tr_i, tr_j, tr_d = ii[train_mask], jj[train_mask], dd[train_mask]
        va_i, va_j, va_d = ii[~train_mask], jj[~train_mask], dd[~train_mask]

        h_np = np.asarray(self._h_nodes, dtype=np.float32)
        s_np = np.asarray(self._s_nodes, dtype=np.float32)
        h_t = torch.tensor(h_np, dtype=torch.float32, device=self.device)
        s_t = torch.tensor(s_np, dtype=torch.float32, device=self.device)

        n_tr = len(tr_i)
        batch_pairs = min(int(batch_pairs), max(1, n_tr))
        losses: List[float] = []
        best_val = float("inf")
        best_state = None

        self.temporal_ensemble.train()
        for step in range(n_steps):
            idx = rng.integers(0, n_tr, size=batch_pairs)
            hi = h_t[tr_i[idx]]
            si = s_t[tr_i[idx]]
            hj = h_t[tr_j[idx]]
            sj = s_t[tr_j[idx]]
            target = torch.tensor(tr_d[idx], dtype=torch.float32, device=self.device)

            preds = self.temporal_ensemble.forward_all(hi, si, hj, sj)
            loss = F.mse_loss(preds, target.unsqueeze(0).expand_as(preds))

            self.temporal_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.temporal_opt.step()
            losses.append(float(loss.item()))

            with torch.no_grad():
                pv = self.temporal_ensemble.forward_all(
                    h_t[va_i], s_t[va_i], h_t[va_j], s_t[va_j]
                )
                tv = torch.tensor(va_d, dtype=torch.float32, device=self.device)
                val_loss = F.mse_loss(pv, tv.unsqueeze(0).expand_as(pv))
                if float(val_loss.item()) < best_val:
                    best_val = float(val_loss.item())
                    best_state = {k: v.detach().cpu() for k, v in self.temporal_ensemble.state_dict().items()}

        if best_state is not None:
            self.temporal_ensemble.load_state_dict(best_state)
        self.temporal_ensemble.eval()
        self._temporal_trained_once = True
        self.tb_last_temporal_train_steps = n_steps
        self.tb_last_temporal_loss = float(np.mean(losses)) if losses else None

    def temporal_intrinsic_reward(
        self,
        h_imag: torch.Tensor,
        s_imag: torch.Tensor,
    ) -> torch.Tensor:
        """r_int[b,t] = mean_r Var_k T_k(h_t,s_t,h_ref,s_ref): ensemble disagreement (no true d at imagine time)."""
        if not self.temporal_intrinsic_ready():
            return torch.zeros(
                (h_imag.size(0), h_imag.size(1) - 1),
                device=h_imag.device,
                dtype=torch.float32,
            )

        B, H1, Dh = h_imag.shape
        Ds = s_imag.size(-1)
        H = H1 - 1
        device = h_imag.device

        h_buf = torch.tensor(
            np.asarray(self._h_nodes, dtype=np.float32), device=device, dtype=torch.float32
        )
        s_buf = torch.tensor(
            np.asarray(self._s_nodes, dtype=np.float32), device=device, dtype=torch.float32
        )
        N = h_buf.size(0)
        R = self.n_temporal_ref

        hb = h_imag[:, :-1].reshape(-1, Dh)
        sb = s_imag[:, :-1].reshape(-1, Ds)
        BH = hb.size(0)

        with torch.no_grad():
            self.temporal_ensemble.eval()
            idx = torch.randint(0, N, (BH, R), device=device)
            hi = hb.unsqueeze(1).expand(-1, R, -1).reshape(BH * R, Dh)
            si = sb.unsqueeze(1).expand(-1, R, -1).reshape(BH * R, Ds)
            hj = h_buf[idx].reshape(BH * R, Dh)
            sj = s_buf[idx].reshape(BH * R, Ds)
            preds = self.temporal_ensemble.forward_all(hi, si, hj, sj)
            preds = preds.view(self.n_temporal_heads, BH, R)
            var_r = preds.var(dim=0, unbiased=False)
            r_flat = var_r.mean(dim=1)
        return r_flat.view(B, H)


# =============================================================================
#  CLI, logging, training loop
# =============================================================================


def build_parser():
    p = argparse.ArgumentParser(
        description="Dreamer for PointMaze with temporal-distance ensemble intrinsic reward"
    )
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

    p.add_argument(
        "--temporal_buffer_max",
        type=int,
        default=2500,
        help="Max (h,s) points stored for temporal pair training and reference sampling.",
    )
    p.add_argument(
        "--temporal_scale",
        type=float,
        default=0.05,
        help="Scale for Var_k T_k intrinsic reward during imagination.",
    )
    p.add_argument("--temporal_heads", type=int, default=5)
    p.add_argument(
        "--temporal_ref",
        type=int,
        default=8,
        help="Replay references per imagined state when computing ensemble variance.",
    )
    p.add_argument("--temporal_train_steps", type=int, default=40)
    p.add_argument("--temporal_batch_pairs", type=int, default=512)
    p.add_argument("--temporal_pair_pool", type=int, default=6000)

    p.add_argument(
        "--baseline",
        action="store_true",
        help="Train world model + actor without temporal ensemble intrinsic reward.",
    )

    # Logging / output
    p.add_argument(
        "--wm_path",
        type=str,
        default="",
        help="Optional path to world_model.pt (encoder/decoder/rssm/reward/cont); actor/critic still random init",
    )
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--run_name", type=str, default="pointmaze_temporal")
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
    temporal_mod: TemporalIntrinsicModule | None,
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
        writer.add_scalar("temporal/disabled", 1.0, global_step)
        return

    assert temporal_mod is not None
    writer.add_scalar("temporal/disabled", 0.0, global_step)
    writer.add_scalar("temporal/buffer_size", float(len(temporal_mod._h_nodes)), global_step)
    writer.add_scalar(
        "temporal/intrinsic_ready",
        1.0 if temporal_mod.temporal_intrinsic_ready() else 0.0,
        global_step,
    )
    if temporal_mod.tb_last_temporal_loss is not None:
        writer.add_scalar(
            "temporal/train_loss", float(temporal_mod.tb_last_temporal_loss), global_step
        )
    writer.add_scalar(
        "temporal/train_steps", float(temporal_mod.tb_last_temporal_train_steps), global_step
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
        print("Mode: baseline Dreamer (no temporal intrinsic reward)")

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

    temporal_mod = None
    if not args.baseline:
        temporal_mod = TemporalIntrinsicModule(
            deter_dim=args.deter_dim,
            stoch_dim=args.stoch_dim,
            device=device,
            max_nodes=args.temporal_buffer_max,
            temporal_hidden=256,
            n_temporal_heads=args.temporal_heads,
            n_temporal_ref=args.temporal_ref,
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
                imag_r_int_mean: list[float] = []
                imag_r_int_std: list[float] = []
                imag_r_int_abs_mean: list[float] = []
                imag_r_int_ratio: list[float] = []

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

                    # ------------- FIFO (h,s) buffer for temporal pair training -------------
                    if temporal_mod is not None and np.random.rand() < 0.1:
                        h_np = h_seq.detach().cpu().numpy()
                        s_np = s_seq.detach().cpu().numpy()
                        traj_h = [h_np[b] for b in range(B)]
                        traj_s = [s_np[b] for b in range(B)]
                        traj_ids = [episode * B + b for b in range(B)]
                        temporal_mod.add_batch_trajectories(traj_h, traj_s, traj_ids)

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

                    if temporal_mod is None:
                        r_int = torch.zeros_like(rewards_imag)
                    else:
                        r_int = temporal_mod.temporal_intrinsic_reward(h_imag, s_imag) * float(
                            args.temporal_scale
                        )
                    rewards_total = rewards_imag + r_int

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

                    # Actor loss: maximize reward-augmented returns
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
                    imag_r_int_mean.append(float(r_int.mean().item()))
                    imag_r_int_std.append(float(r_int.std().item()))
                    imag_r_int_abs_mean.append(float(r_int.abs().mean().item()))
                    denom = float(rewards_imag.abs().mean().item()) + 1e-8
                    imag_r_int_ratio.append(float(r_int.abs().mean().item()) / denom)

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
                    "r_int_mean": float(np.mean(imag_r_int_mean)),
                    "r_int_std": float(np.mean(imag_r_int_std)),
                    "r_int_abs_mean": float(np.mean(imag_r_int_abs_mean)),
                    "r_int_over_abs_extrinsic": float(np.mean(imag_r_int_ratio)),
                }
                policy_avg = {
                    "actor_loss": sum_actor / n_ts,
                    "value_loss": sum_value / n_ts,
                    "mean_weighted_return": sum_weighted_ret / n_ts,
                }

                if temporal_mod is not None:
                    temporal_mod.train_temporal_ensemble(
                        n_steps=int(args.temporal_train_steps),
                        batch_pairs=int(args.temporal_batch_pairs),
                        pair_pool=int(args.temporal_pair_pool),
                        seed=int(args.seed),
                    )

                log_training_phase_tensorboard(
                    writer,
                    total_steps,
                    replay_size=replay.size,
                    expl_amount=expl_amount,
                    baseline=args.baseline,
                    temporal_mod=temporal_mod,
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

