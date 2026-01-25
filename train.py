import argparse
import copy
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from utils import (
    ReplayBuffer,
    get_device,
    no_param_grads,
    set_seed,
    preprocess_img,
    bottle,
    random_masking,
    make_env,
    ENV_ACTION_REPEAT,
    log_visualizations,
    log_imagination_rollout,
    log_reward_prediction,
    log_latent_reward_structure,
    log_action_conditioned_prediction,
    log_latent_dynamics,
)
from models import ConvEncoder, ConvDecoder, RSSM, RewardModel, FeatureDecoder
from vicreg_net import PhiNet, ema_update, phi_augment


# ===============================
# Planning (CEM)
# ===============================

@torch.no_grad()
def cem_plan_action_rssm(
    rssm,
    reward_model,
    h_t,
    s_t,
    act_low,
    act_high,
    horizon=12,
    candidates=1000,
    iters=10,
    top_k=100,
    device=None,
    explore=False,
    explore_noise=0.3,
):
    """
    CEM planner using RSSM prior rollout and reward model.
    """
    device = device or h_t.device
    act_dim = act_low.shape[0]

    act_low_t = torch.tensor(act_low, device=device, dtype=torch.float32)
    act_high_t = torch.tensor(act_high, device=device, dtype=torch.float32)

    mu = torch.zeros(horizon, act_dim, device=device)
    stddev = torch.ones(horizon, act_dim, device=device)

    for _ in range(iters):
        actions = Normal(mu, stddev).sample((candidates,))  # [N, H, A]
        actions = torch.clamp(actions, act_low_t, act_high_t)

        rwds = torch.zeros(candidates, device=device)
        h = h_t.expand(candidates, -1).clone()
        s = s_t.expand(candidates, -1).clone()

        for t in range(horizon):
            a_t = actions[:, t]
            h = rssm.deterministic_state_fwd(h, s, a_t)
            s = rssm.state_prior(h, sample=True)
            rwds += reward_model(h, s)

        _, k = torch.topk(rwds, top_k, dim=0, largest=True, sorted=False)
        elite_actions = actions[k]
        mu = elite_actions.mean(dim=0)
        stddev = elite_actions.std(dim=0, unbiased=False)

    action = mu[0].clone()
    if explore:
        action = action + torch.randn_like(action) * explore_noise

    action = torch.clamp(action, act_low_t, act_high_t)
    return action.cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate_planner_rssm(
    env_id,
    img_size,
    encoder,
    rssm,
    reward_model,
    plan_kwargs,
    episodes=10,
    seed=0,
    device="cpu",
    bit_depth=5,
    action_repeat=1,
):
    env = make_env(env_id, img_size=(img_size, img_size), num_stack=1)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    encoder.eval()
    rssm.eval()
    reward_model.eval()

    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0

        obs_t = obs_to_tensor(obs, device, bit_depth)
        enc = encoder(obs_t)
        h_state, s_state = rssm.get_init_state(enc)

        while not done:
            action = cem_plan_action_rssm(
                rssm=rssm,
                reward_model=reward_model,
                h_t=h_state,
                s_t=s_state,
                device=device,
                explore=False,
                **plan_kwargs,
            )

            total_reward = 0.0
            for _ in range(action_repeat):
                obs, r, term, trunc, _ = env.step(action)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)
            ep_ret += total_reward

            obs_t = obs_to_tensor(obs, device, bit_depth)
            enc = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(enc, act_t, h_state, s_state, sample=False)

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ===============================
# Args / Config
# ===============================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="cheetah-run")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--seq_len", type=int, default=50)
    p.add_argument("--max_episodes", type=int, default=1000)
    p.add_argument("--seed_episodes", type=int, default=5)
    p.add_argument("--collect_interval", type=int, default=100)
    p.add_argument("--train_steps", type=int, default=100)
    p.add_argument("--action_repeat", type=int, default=0)

    p.add_argument("--replay_buff_capacity", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--adam_eps", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1000.0)
    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--kl_free_nats", type=float, default=3.0)

    p.add_argument("--pb_curvature_weight", type=float, default=0.0)
    p.add_argument("--pb_curvature_projections", type=int, default=2)
    p.add_argument("--pb_detach_features", action="store_true")

    p.add_argument("--pullback_bisim", action="store_true")
    p.add_argument("--geo_knn_k", type=int, default=8)
    p.add_argument("--geo_time_subsample", type=int, default=16)
    p.add_argument("--bisimulation_weight", type=float, default=0.0)
    p.add_argument("--bisimulation_warmup", type=int, default=0)
    p.add_argument("--bisimulation_ramp", type=int, default=0)

    p.add_argument("--reward_scale", type=float, default=1.0)

    p.add_argument("--use_feature_decoder", action="store_true")
    p.add_argument("--feature_dim", type=int, default=128)
    p.add_argument("--mask_ratio", type=float, default=0.5)
    p.add_argument("--mask_ratio_final", type=float, default=0.35)
    p.add_argument("--mask_ramp_steps", type=int, default=80000)
    p.add_argument("--mask_patch_size", type=int, default=8)
    p.add_argument("--feature_bank", action="store_true")
    p.add_argument("--aux_pixel_weight", type=float, default=0.0)
    p.add_argument("--feature_rec_scale", type=float, default=10.0)

    p.add_argument("--phi_lr", type=float, default=1e-4)
    p.add_argument("--phi_update_every", type=int, default=4)
    p.add_argument("--phi_tau", type=float, default=0.9995)
    p.add_argument("--phi_freeze_steps", type=int, default=0)
    p.add_argument("--phi_warmup_steps", type=int, default=0)

    p.add_argument("--plan_horizon", type=int, default=12)
    p.add_argument("--plan_candidates", type=int, default=1000)
    p.add_argument("--plan_iters", type=int, default=10)
    p.add_argument("--plan_top_k", type=int, default=100)
    p.add_argument("--explore_noise", type=float, default=0.3)

    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=10)

    p.add_argument("--run_name", type=str, default="train")
    p.add_argument("--log_dir", type=str, default="runs")
    p.add_argument("--save_interval", type=int, default=0)
    p.add_argument("--load_checkpoint", type=str, default="")
    p.add_argument("--visualize_interval", type=int, default=0)

    return p


def make_run_name(args):
    parts = [args.env_id]
    if args.use_feature_decoder:
        parts.append(f"featdec{args.feature_dim}")
        if args.mask_ramp_steps > 0:
            parts.append(f"mask_ramp{args.mask_ratio_final}")
        else:
            parts.append(f"mask{args.mask_ratio}")
        if args.aux_pixel_weight > 0:
            parts.append(f"auxpix{args.aux_pixel_weight}")

    if args.bisimulation_weight > 0:
        parts.append(f"bisim{args.bisimulation_weight}")
    else:
        parts.append("no_bisim")

    if args.pb_curvature_weight > 0:
        parts.append(f"jacobreg{args.pb_curvature_weight}")
    else:
        parts.append("no_jacobreg")

    if args.pullback_bisim:
        parts.append("pullback_bisim")

    if args.phi_lr != 1e-3:
        parts.append(f"philr{args.phi_lr}")

    parts.append(f"seed{args.seed}")
    return "_".join(parts)


# ===============================
# Geometry helpers (pullback / geodesic)
# ===============================

def pullback_distance_full(decoder, phi_target, h1, s1, h2, s2, create_graph=True, is_feature_decoder=False):
    """
    Computes || J_{h,s} g(h1, s1) · ([h1, s1] - [h2, s2]) ||_2.
    """
    delta_h = (h1 - h2)
    delta_s = (s1 - s2)

    if is_feature_decoder:
        def decoder_wrapper(h, s):
            f = decoder(h, s)
            f = F.layer_norm(f, (f.size(-1),), eps=1e-3)
            return f

        with no_param_grads(decoder):
            _, Jdelta = torch.autograd.functional.jvp(
                decoder_wrapper, (h1, s1), (delta_h, delta_s), create_graph=create_graph
            )
    else:
        def decoder_wrapper(h, s):
            x = decoder(h, s)
            f = phi_target(x)
            f = F.layer_norm(f, (f.size(-1),), eps=1e-3)
            return f

        with no_param_grads(decoder), no_param_grads(phi_target):
            _, Jdelta = torch.autograd.functional.jvp(
                decoder_wrapper, (h1, s1), (delta_h, delta_s), create_graph=create_graph
            )

    Jdelta = Jdelta.reshape(Jdelta.size(0), -1)
    return torch.sqrt((Jdelta * Jdelta).mean(dim=1) + 1e-8)


def floyd_warshall_minplus(W):
    D = W
    B = D.size(0)
    for k in range(B):
        trough = D[:, k:k + 1] + D[k:k + 1, :]
        D = torch.minimum(D, trough)
    return D


def geodesic_pb_knn_slice(
    decoder,
    phi_target,
    h_t,
    z_t,
    targets,
    k=3,
    create_graph=True,
    inf=1e9,
    is_feature_decoder=False,
):
    B = z_t.size(0)
    device = z_t.device

    with torch.no_grad():
        dist2 = torch.cdist(z_t, z_t, p=2).pow(2)
        dist2.fill_diagonal_(float("inf"))
        knn = dist2.topk(k, largest=False).indices  # [B, k]

    src = torch.arange(B, device=device).repeat_interleave(k)
    dst = knn.reshape(-1)

    src2 = torch.cat([src, dst], dim=0)
    dst2 = torch.cat([dst, src], dim=0)

    w2 = pullback_distance_full(
        decoder,
        phi_target=phi_target,
        h1=h_t[src2],
        s1=z_t[src2],
        h2=h_t[dst2].detach(),
        s2=z_t[dst2].detach(),
        create_graph=create_graph,
        is_feature_decoder=is_feature_decoder,
    )

    W = torch.full((B, B), inf, device=device, dtype=torch.float32)
    W.fill_diagonal_(0.0)
    W[src2, dst2] = w2.float()
    W = torch.minimum(W, W.T)

    D = floyd_warshall_minplus(W)
    dz_t = D[torch.arange(B, device=device), targets]
    return dz_t, w2.detach()


def action_targets(a_t, topm=3):
    dist = torch.cdist(a_t, a_t)
    dist.fill_diagonal_(float("inf"))
    nn = dist.topk(topm, largest=False).indices
    pick = torch.randint(0, topm, (a_t.size(0),), device=a_t.device)
    return nn[torch.arange(a_t.size(0), device=a_t.device), pick]


def geodesic_pb_knn(
    decoder,
    phi_target,
    h,
    z,
    targets,
    k=10,
    time_subsample=None,
    t_idx=None,
    create_graph=True,
    return_mask=False,
    is_feature_decoder=False,
):
    B, T, _ = z.shape
    dz = torch.zeros((B, T), device=z.device, dtype=z.dtype)
    mask = torch.zeros((T,), device=z.device, dtype=torch.bool)
    edge_weights_list = []

    if t_idx is None:
        t_idx = (
            torch.arange(T, device=z.device)
            if (time_subsample is None or time_subsample >= T)
            else torch.randint(0, T, (time_subsample,), device=z.device)
        )

    for t in t_idx.tolist():
        tgt_t = targets if targets.ndim == 1 else targets[:, t]
        dz_t, w_t = geodesic_pb_knn_slice(
            decoder,
            phi_target,
            h[:, t],
            z[:, t],
            tgt_t,
            k=k,
            create_graph=create_graph,
            is_feature_decoder=is_feature_decoder,
        )
        dz[:, t] = dz_t
        mask[t] = True
        edge_weights_list.append(w_t)

    edge_weights = torch.cat(edge_weights_list) if edge_weights_list else torch.tensor([], device=z.device)

    if return_mask:
        return dz.reshape(-1), mask.repeat(B), t_idx, edge_weights
    return dz.reshape(-1), edge_weights


# ===============================
# Build / Helpers
# ===============================

@dataclass
class Models:
    encoder: torch.nn.Module
    decoder: torch.nn.Module
    vis_decoder: torch.nn.Module
    rssm: torch.nn.Module
    reward_model: torch.nn.Module
    phi_net: torch.nn.Module
    phi_net_target: torch.nn.Module


@dataclass
class Optimizers:
    world: torch.optim.Optimizer
    vis_decoder: torch.optim.Optimizer | None
    phi: torch.optim.Optimizer


def obs_to_tensor(obs, device, bit_depth):
    obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
    obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
    preprocess_img(obs_t, depth=bit_depth)
    return obs_t


def compute_mask_ratio(args, total_steps):
    if args.mask_ramp_steps > 0:
        return min(args.mask_ratio_final, args.mask_ratio_final * total_steps / args.mask_ramp_steps)
    return args.mask_ratio


def bisim_weight_schedule(total_steps, warmup, ramp, target):
    if warmup > 0 and total_steps < warmup:
        return 0.0
    if ramp > 0:
        t = min(1.0, (total_steps - warmup) / float(ramp))
        return target * t
    return target


def build_models(args, obs_channels, act_dim, device):
    encoder = ConvEncoder(embedding_size=args.embed_dim, in_channels=obs_channels).to(device)

    vis_decoder = ConvDecoder(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        embedding_size=args.embed_dim,
        out_channels=obs_channels,
    ).to(device)

    if args.use_feature_decoder:
        decoder = FeatureDecoder(
            state_size=args.deter_dim,
            latent_size=args.stoch_dim,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            activation_function="elu",
        ).to(device)
        print(f"Using FeatureDecoder (feature_dim={args.feature_dim})")
    else:
        decoder = vis_decoder
        print("Using ConvDecoder (pixel reconstruction)")

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

    phi_out_dim = args.feature_dim if args.use_feature_decoder else args.embed_dim
    phi_net = PhiNet(in_channels=obs_channels, hidden_channels=args.hidden_dim, out_dim=phi_out_dim).to(device)
    phi_net_target = copy.deepcopy(phi_net)
    phi_net_target.eval()
    for param in phi_net_target.parameters():
        param.requires_grad = False
    ema_update(phi_net_target, phi_net, 0.999)

    return Models(
        encoder=encoder,
        decoder=decoder,
        vis_decoder=vis_decoder,
        rssm=rssm,
        reward_model=reward_model,
        phi_net=phi_net,
        phi_net_target=phi_net_target,
    )


def build_optimizers(args, models):
    params = (
        list(models.encoder.parameters())
        + list(models.decoder.parameters())
        + list(models.rssm.parameters())
        + list(models.reward_model.parameters())
    )
    if args.use_feature_decoder and args.aux_pixel_weight > 0:
        params = params + list(models.vis_decoder.parameters())

    world_optim = torch.optim.Adam(params, lr=args.lr, eps=args.adam_eps)

    if args.use_feature_decoder and args.aux_pixel_weight == 0:
        vis_decoder_optim = torch.optim.Adam(models.vis_decoder.parameters(), lr=args.lr, eps=args.adam_eps)
    else:
        vis_decoder_optim = None

    phi_optim = torch.optim.Adam(list(models.phi_net.parameters()), lr=args.phi_lr, eps=args.adam_eps)

    return Optimizers(world=world_optim, vis_decoder=vis_decoder_optim, phi=phi_optim)


def init_feature_bank(args, device):
    if not args.feature_bank:
        return None, 0
    bank_capacity = 256
    feature_dim = args.feature_dim if args.use_feature_decoder else args.embed_dim
    feature_bank = torch.randn(bank_capacity, feature_dim, device=device)
    return feature_bank, 0


def load_checkpoint_if_needed(args, models, device):
    if not args.load_checkpoint:
        return 0, 0

    print(f"Loading checkpoint from: {args.load_checkpoint}")
    checkpoint = torch.load(args.load_checkpoint, map_location=device)
    models.encoder.load_state_dict(checkpoint["encoder"])
    models.decoder.load_state_dict(checkpoint["decoder"])
    models.rssm.load_state_dict(checkpoint["rssm"])
    models.reward_model.load_state_dict(checkpoint["reward_model"])
    if args.use_feature_decoder:
        if "vis_decoder" in checkpoint:
            models.vis_decoder.load_state_dict(checkpoint["vis_decoder"])
        if "phi_net" in checkpoint:
            models.phi_net.load_state_dict(checkpoint["phi_net"])
            ema_update(models.phi_net_target, models.phi_net, 0.0)

    total_steps = checkpoint.get("total_steps", 0)
    start_episode = checkpoint.get("episode", 0)
    print(f"Resumed from episode {start_episode}, total_steps {total_steps}")
    return total_steps, start_episode


def seed_replay_buffer(env, replay, action_repeat, seed_episodes, total_steps):
    print(f"Seeding replay buffer with {seed_episodes} random episodes...")
    for seed_ep in range(seed_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action = env.action_space.sample()
            total_reward = 0.0
            for _ in range(action_repeat):
                next_obs, r, term, trunc, _ = env.step(action)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)

            replay.add(
                obs=np.ascontiguousarray(obs, dtype=np.uint8),
                action=np.asarray(action, dtype=np.float32),
                reward=total_reward,
                next_obs=np.ascontiguousarray(next_obs, dtype=np.uint8),
                done=done,
            )
            ep_return += total_reward
            total_steps += 1
            obs = next_obs

        print(
            f"  Seed episode {seed_ep + 1}/{seed_episodes}: return = {ep_return:.2f}, buffer size = {replay.size}"
        )
    return total_steps


def log_initial_visualization(args, writer, replay, models, device):
    if replay.size <= (args.seq_len + 2) or args.visualize_interval <= 0:
        return

    print("Generating initial visualization (step 0)...")
    viz_batch = replay.sample_sequences(min(args.batch_size, 8), args.seq_len + 1)
    viz_obs = torch.tensor(viz_batch.obs, dtype=torch.float32, device=device)
    viz_act = torch.tensor(viz_batch.actions, dtype=torch.float32, device=device)
    viz_rew = torch.tensor(viz_batch.rews, dtype=torch.float32, device=device)

    viz_x = viz_obs.permute(0, 1, 4, 2, 3).contiguous()
    preprocess_img(viz_x, depth=args.bit_depth)

    with torch.no_grad():
        viz_e = bottle(models.encoder, viz_x)
        viz_h_init, viz_s_init = models.rssm.get_init_state(viz_e[:, 0])

        viz_h_list, viz_s_list = [], []
        h_curr, s_curr = viz_h_init, viz_s_init
        viz_T = viz_x.shape[1] - 1

        for t in range(viz_T):
            h_curr = models.rssm.deterministic_state_fwd(h_curr, s_curr, viz_act[:, t])
            viz_h_list.append(h_curr)
            post_mean, _ = models.rssm.state_posterior(h_curr, viz_e[:, t + 1])
            s_curr = post_mean
            viz_s_list.append(s_curr)

        viz_h_seq = torch.stack(viz_h_list, dim=1)
        viz_s_seq = torch.stack(viz_s_list, dim=1)

    pixel_decoder = models.vis_decoder if args.use_feature_decoder else models.decoder
    log_visualizations(
        writer=writer,
        step=0,
        encoder=models.encoder,
        decoder=pixel_decoder,
        rssm=models.rssm,
        obs_batch=viz_x[:, 1:viz_T + 1],
        act_batch=viz_act[:, :viz_T],
        h_seq=viz_h_seq,
        s_seq=viz_s_seq,
        bit_depth=args.bit_depth,
        knn_k=args.geo_knn_k,
        device=device,
    )

    log_imagination_rollout(
        writer=writer,
        step=0,
        encoder=models.encoder,
        decoder=pixel_decoder,
        rssm=models.rssm,
        obs_batch=viz_x,
        act_batch=viz_act,
        bit_depth=args.bit_depth,
        imagination_horizon=min(15, viz_T),
        device=device,
    )
    log_reward_prediction(
        writer=writer,
        step=0,
        encoder=models.encoder,
        rssm=models.rssm,
        reward_model=models.reward_model,
        obs_batch=viz_x,
        act_batch=viz_act,
        rew_batch=viz_rew,
        device=device,
    )
    log_latent_reward_structure(
        writer=writer,
        step=0,
        h_seq=viz_h_seq,
        s_seq=viz_s_seq,
        rew_batch=viz_rew[:, :viz_T],
        device=device,
    )
    log_action_conditioned_prediction(
        writer=writer,
        step=0,
        encoder=models.encoder,
        decoder=pixel_decoder,
        rssm=models.rssm,
        obs_batch=viz_x,
        act_batch=viz_act,
        act_dim=viz_act.shape[-1],
        horizon=min(5, viz_T),
        device=device,
    )
    log_latent_dynamics(
        writer=writer,
        step=0,
        h_seq=viz_h_seq,
        s_seq=viz_s_seq,
        act_batch=viz_act[:, :viz_T],
        device=device,
    )


def train_world_model(
    args,
    models,
    optimizers,
    replay,
    writer,
    total_steps,
    feature_bank,
    bank_ptr,
    target_scale_ema,
    phi_update_counter,
    device,
):
    models.encoder.train()
    models.decoder.train()
    models.rssm.train()
    models.reward_model.train()

    loss_accum = {"total": 0.0, "rec": 0.0, "kl": 0.0, "rew": 0.0, "pb": 0.0, "bisim": 0.0}
    bisim_stats_accum = {
        "dz_mean": 0.0,
        "dz_std": 0.0,
        "target_mean": 0.0,
        "target_std": 0.0,
        "abs_err_mean": 0.0,
        "rel_err": 0.0,
        "corr": 0.0,
    }
    grad_norm_accum = {"total": 0.0, "encoder": 0.0, "decoder": 0.0, "rssm": 0.0, "reward": 0.0, "bisim_weighted": 0.0}
    n_bisim_computations = 0

    feat_align_weight = 0.1
    encoder_params = list(models.encoder.parameters())
    decoder_params = list(models.decoder.parameters())
    rssm_params = list(models.rssm.parameters())
    reward_params = list(models.reward_model.parameters())

    for _ in range(args.train_steps):
        batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)

        obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
        act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)
        rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)

        B, T1 = rew_seq.shape
        T = T1 - 1

        x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
        preprocess_img(x, depth=args.bit_depth)

        if args.use_feature_decoder:
            x_flat = x.view(-1, x.size(2), x.size(3), x.size(4))
            with torch.no_grad():
                target_features = models.phi_net_target(x_flat)
                target_features = F.layer_norm(target_features, (target_features.size(-1),), eps=1e-3)
                target_features = target_features.view(B, T1, -1)

            mask_ratio = compute_mask_ratio(args, total_steps)
            x_masked = random_masking(x, mask_ratio=mask_ratio, patch_size=args.mask_patch_size)
            e_t = bottle(models.encoder, x_masked)
        else:
            e_t = bottle(models.encoder, x)

        h_t, s_t = models.rssm.get_init_state(e_t[:, 0])

        states = []
        priors = []
        posteriors = []
        posterior_samples = []

        for t in range(T):
            h_t = models.rssm.deterministic_state_fwd(h_t, s_t, act_seq[:, t])
            states.append(h_t)

            priors.append(models.rssm.state_prior(h_t))
            posteriors.append(models.rssm.state_posterior(h_t, e_t[:, t + 1]))

            post_mean, post_std = posteriors[-1]
            s_t = post_mean + torch.randn_like(post_std) * post_std
            posterior_samples.append(s_t)

        h_seq = torch.stack(states, dim=1)
        s_seq = torch.stack(posterior_samples, dim=1)

        prior_means = torch.stack([p[0] for p in priors], dim=0)
        prior_stds = torch.stack([p[1] for p in priors], dim=0)
        post_means = torch.stack([p[0] for p in posteriors], dim=0)
        post_stds = torch.stack([p[1] for p in posteriors], dim=0)

        prior_dist = Normal(prior_means, prior_stds)
        posterior_dist = Normal(post_means, post_stds)

        if args.use_feature_decoder:
            pred_features = bottle(models.decoder, h_seq, s_seq)
            feat_target_seq = target_features[:, 1:T + 1]
            pred_features = F.layer_norm(pred_features, (pred_features.size(-1),), eps=1e-3)
            pred_norm = F.normalize(pred_features, dim=-1)
            target_norm = F.normalize(feat_target_seq, dim=-1)
            cosine_sim = (pred_norm * target_norm).sum(dim=-1)
            raw_rec_loss = (1.0 - cosine_sim).mean() * args.feature_rec_scale

            if args.phi_warmup_steps > 0 and total_steps < args.phi_warmup_steps:
                phi_warmup_factor = total_steps / args.phi_warmup_steps
                rec_loss = raw_rec_loss * phi_warmup_factor
            else:
                rec_loss = raw_rec_loss

            feat_loss = torch.zeros((), device=device)
            feat_target = target_features[:, 1:T + 1].reshape(-1, args.feature_dim)

            if args.aux_pixel_weight > 0:
                pix_recon = bottle(models.vis_decoder, h_seq, s_seq)
                pix_target = x[:, 1:T + 1]
                aux_pix_loss = F.mse_loss(pix_recon, pix_target, reduction="none").sum((2, 3, 4)).mean()
            else:
                aux_pix_loss = torch.zeros((), device=device)
                if optimizers.vis_decoder is not None:
                    target_pixels = x[:, 1:T + 1]
                    vis_recon = bottle(models.vis_decoder, h_seq.detach(), s_seq.detach())
                    vis_loss = F.mse_loss(vis_recon, target_pixels, reduction="none").sum((2, 3, 4)).mean()
                    optimizers.vis_decoder.zero_grad()
                    vis_loss.backward()
                    torch.nn.utils.clip_grad_norm_(models.vis_decoder.parameters(), 10.0)
                    optimizers.vis_decoder.step()
        else:
            recon = bottle(models.decoder, h_seq, s_seq)
            target = x[:, 1:T + 1]
            rec_loss = F.mse_loss(recon, target, reduction="none").sum((2, 3, 4)).mean()

            recon_flat = recon.reshape(-1, x.size(2), x.size(3), x.size(4))
            target_flat = target.reshape(-1, x.size(2), x.size(3), x.size(4))
            with torch.no_grad():
                _ = models.phi_net_target(target_flat)
            feat_loss = torch.zeros((), device=device)
            aux_pix_loss = torch.zeros((), device=device)

        kld_loss = torch.max(kl_divergence(posterior_dist, prior_dist).sum(-1), torch.ones(1, device=device) * args.kl_free_nats).mean()
        rew_pred = bottle(models.reward_model, h_seq, s_seq)
        rew_target = rew_seq[:, :T]
        rew_loss = F.mse_loss(rew_pred, rew_target) * args.reward_scale

        pb_loss = torch.zeros((), device=device)
        if args.pb_curvature_weight > 0.0:
            pass

        bisim_weight = bisim_weight_schedule(
            total_steps,
            warmup=args.bisimulation_warmup,
            ramp=args.bisimulation_ramp,
            target=args.bisimulation_weight,
        )

        bisim_loss_val = torch.zeros((), device=device)
        if T > 2:
            h = h_seq[:, :-1]
            hn = h_seq[:, 1:]
            z = s_seq[:, :-1]
            zn = s_seq[:, 1:]
            r = rew_seq[:, :T - 1]
            a = act_seq[:, :T - 1]

            if not args.pullback_bisim:
                z = F.layer_norm(z, (z.size(-1),))
                zn = F.layer_norm(zn, (zn.size(-1),))

            Bm, Tm1, _ = z.shape
            t_idx = torch.arange(Tm1, device=device)
            targets = torch.arange(Bm, device=device, dtype=torch.long)[:, None].expand(Bm, Tm1).clone()
            for t in t_idx.tolist():
                targets[:, t] = action_targets(a[:, t], topm=3)

            needs_grad = bisim_weight > 0.0
            if args.pullback_bisim:
                dz, dz_mask, t_idx, edge_weights = geodesic_pb_knn(
                    decoder=models.decoder,
                    phi_target=models.phi_net_target,
                    h=h,
                    z=z,
                    targets=targets,
                    k=args.geo_knn_k,
                    time_subsample=args.geo_time_subsample,
                    create_graph=needs_grad,
                    return_mask=True,
                    is_feature_decoder=args.use_feature_decoder,
                )
            else:
                t_grid = torch.arange(Tm1, device=device)[None, :].expand(Bm, Tm1)
                z2 = z[targets, t_grid].detach()
                dz = torch.norm(z - z2, p=2, dim=-1).reshape(-1)
                dz_mask = torch.zeros((Bm * Tm1,), device=device, dtype=torch.bool)
                dz_mask[(t_grid.reshape(-1).unsqueeze(0) == t_idx.reshape(-1).unsqueeze(1)).any(dim=0)] = True
                edge_weights = torch.tensor([], device=device)

            with torch.no_grad():
                t_grid = torch.arange(Tm1, device=device)[None, :].expand(Bm, Tm1)
                r2 = r[targets, t_grid]
                dr = (r - r2).abs().reshape(-1)

                if args.pullback_bisim:
                    dnext, _, _, _ = geodesic_pb_knn(
                        decoder=models.decoder,
                        phi_target=models.phi_net_target,
                        h=hn,
                        z=zn,
                        targets=targets,
                        k=args.geo_knn_k,
                        time_subsample=args.geo_time_subsample,
                        t_idx=t_idx,
                        create_graph=False,
                        return_mask=True,
                        is_feature_decoder=args.use_feature_decoder,
                    )
                else:
                    n2 = zn[targets, t_grid]
                    dnext = torch.norm(zn - n2, p=2, dim=-1).reshape(-1)

                raw_target = dr + args.gamma * dnext
                batch_scale = raw_target.mean()
                target_scale_ema = 0.99 * target_scale_ema + 0.01 * batch_scale.item()
                safe_scale = max(target_scale_ema, 1e-4)
                normalized_target = raw_target / safe_scale

            mask = dz_mask.bool()
            dz_m = dz[mask]
            tgt_m = normalized_target[mask]
            bisim_loss_val = F.mse_loss(dz_m, tgt_m)

            if args.bisimulation_weight > 0.0:
                with torch.no_grad():
                    dz_detached = dz.detach()[mask]
                    target_detached = normalized_target.detach()[mask]
                    dz_mean = dz_detached.mean().item()
                    dz_std = dz_detached.std().item()
                    target_mean = target_detached.mean().item()
                    target_std = target_detached.std().item()
                    abs_err = (dz_detached - target_detached).abs()
                    abs_err_mean = abs_err.mean().item()
                    rel_err = abs_err_mean / (target_mean + 1e-6)
                    dz_centered = dz_detached - dz_detached.mean()
                    target_centered = target_detached - target_detached.mean()
                    numerator = (dz_centered * target_centered).mean()
                    dz_var = dz_centered.pow(2).mean()
                    target_var = target_centered.pow(2).mean()
                    corr = (numerator / (dz_var.sqrt() * target_var.sqrt() + 1e-8)).item()
                    edge_weight_mean = edge_weights.mean().item() if len(edge_weights) > 0 else 0.0
                    edge_weight_std = edge_weights.std().item() if len(edge_weights) > 0 else 0.0
                    feat_norm_mean = feat_target.norm(dim=-1).mean().item()
                    feat_std = feat_target.std(dim=-1).mean().item()

                    bisim_stats_accum["dz_mean"] += dz_mean
                    bisim_stats_accum["dz_std"] += dz_std
                    bisim_stats_accum["target_mean"] += target_mean
                    bisim_stats_accum["target_std"] += target_std
                    bisim_stats_accum["abs_err_mean"] += abs_err_mean
                    bisim_stats_accum["rel_err"] += rel_err
                    bisim_stats_accum["corr"] += corr
                    bisim_stats_accum["edge_weight_mean"] = bisim_stats_accum.get("edge_weight_mean", 0.0) + edge_weight_mean
                    bisim_stats_accum["edge_weight_std"] = bisim_stats_accum.get("edge_weight_std", 0.0) + edge_weight_std
                    bisim_stats_accum["feat_norm_mean"] = bisim_stats_accum.get("feat_norm_mean", 0.0) + feat_norm_mean
                    bisim_stats_accum["feat_std"] = bisim_stats_accum.get("feat_std", 0.0) + feat_std
                    bisim_stats_accum["target_scale_ema"] = bisim_stats_accum.get("target_scale_ema", 0.0) + target_scale_ema
                    n_bisim_computations += 1

        total = (
            rec_loss
            + args.kl_weight * kld_loss
            + rew_loss
            + bisim_weight * bisim_loss_val
            + feat_align_weight * feat_loss
            + args.aux_pixel_weight * aux_pix_loss
        )

        optimizers.world.zero_grad()
        total.backward()

        bisim_grad_norm = bisim_weight * bisim_loss_val.item() if bisim_weight > 0 else 0.0
        with torch.no_grad():
            total_grad_norm = torch.nn.utils.clip_grad_norm_(list(optimizers.world.param_groups[0]["params"]), float("inf"), norm_type=2)
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder_params, float("inf"), norm_type=2) if encoder_params else torch.tensor(0.0)
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder_params, float("inf"), norm_type=2) if decoder_params else torch.tensor(0.0)
            rssm_grad_norm = torch.nn.utils.clip_grad_norm_(rssm_params, float("inf"), norm_type=2) if rssm_params else torch.tensor(0.0)
            reward_grad_norm = torch.nn.utils.clip_grad_norm_(reward_params, float("inf"), norm_type=2) if reward_params else torch.tensor(0.0)
            grad_norm_accum["total"] += total_grad_norm.item()
            grad_norm_accum["encoder"] += encoder_grad_norm.item()
            grad_norm_accum["decoder"] += decoder_grad_norm.item()
            grad_norm_accum["rssm"] += rssm_grad_norm.item()
            grad_norm_accum["reward"] += reward_grad_norm.item()
            grad_norm_accum["bisim_weighted"] += bisim_grad_norm if isinstance(bisim_grad_norm, (int, float)) else bisim_grad_norm.item()

        if args.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(optimizers.world.param_groups[0]["params"], args.grad_clip_norm, norm_type=2)

        optimizers.world.step()

        # VICReg / PhiNet update
        phi_update_counter += 1
        do_phi_update = (phi_update_counter % args.phi_update_every == 0)
        if args.phi_freeze_steps > 0:
            do_phi_update = do_phi_update and (total_steps < args.phi_freeze_steps)

        if do_phi_update:
            if args.use_feature_decoder:
                real = x[:, 1:T + 1].reshape(-1, x.size(2), x.size(3), x.size(4)).detach()
            else:
                real = target.reshape(-1, x.size(2), x.size(3), x.size(4)).detach()

            models.phi_net.train()
            x1 = phi_augment(real)
            x2 = phi_augment(real)
            z1 = models.phi_net(x1)
            z2 = models.phi_net(x2)

            if args.feature_bank and feature_bank is not None:
                with torch.no_grad():
                    z_out = z1.detach()
                    batch_len = z_out.shape[0]
                    bank_capacity = feature_bank.shape[0]
                    indices = torch.arange(bank_ptr, bank_ptr + batch_len, device=device) % bank_capacity
                    feature_bank[indices] = z_out
                    bank_ptr = (bank_ptr + batch_len) % bank_capacity

            sim_loss = F.mse_loss(z1, z2)
            if args.feature_bank and feature_bank is not None:
                z_all = torch.cat([z1, feature_bank.detach()], dim=0)
            else:
                z_all = z1

            eps = 1e-4
            std_z = torch.sqrt(z_all.var(dim=0) + eps)
            var_loss = F.relu(1.0 - std_z).mean()

            z_all_centered = z_all - z_all.mean(dim=0)
            cov_z = (z_all_centered.T @ z_all_centered) / (z_all.shape[0] - 1)
            n_dim = cov_z.shape[0]
            off_diag_mask = ~torch.eye(n_dim, dtype=torch.bool, device=device)
            cov_loss = cov_z[off_diag_mask].pow(2).mean()

            phi_loss = 25.0 * sim_loss + 25.0 * var_loss + 1.0 * cov_loss
            optimizers.phi.zero_grad()
            phi_loss.backward()
            torch.nn.utils.clip_grad_norm_(models.phi_net.parameters(), 10.0)
            optimizers.phi.step()

            ema_update(models.phi_net_target, models.phi_net, args.phi_tau)
            models.phi_net_target.eval()

            with torch.no_grad():
                writer.add_scalar("phi/loss", phi_loss.item(), total_steps)
                writer.add_scalar("phi/sim_loss", sim_loss.item(), total_steps)
                writer.add_scalar("phi/var_loss", var_loss.item(), total_steps)
                writer.add_scalar("phi/cov_loss", cov_loss.item(), total_steps)
                writer.add_scalar("phi/z1_mean", z1.mean().item(), total_steps)
                writer.add_scalar("phi/z1_std", z1.std().item(), total_steps)
                writer.add_scalar("phi/z2_mean", z2.mean().item(), total_steps)
                writer.add_scalar("phi/z2_std", z2.std().item(), total_steps)
                writer.add_scalar("phi/z1_z2_corr", (z1 * z2).mean().item(), total_steps)
                writer.add_scalar("phi/z1_z2_cov", (z1 * z2).var().item(), total_steps)
                writer.add_scalar("phi/bank_std_mean", std_z.mean().item(), total_steps)

                target_feat_sample = models.phi_net_target(real[:min(64, real.shape[0])])
                writer.add_scalar("phi/target_feat_mean", target_feat_sample.mean().item(), total_steps)
                writer.add_scalar("phi/target_feat_std", target_feat_sample.std().item(), total_steps)
                writer.add_scalar("phi/target_feat_norm", target_feat_sample.norm(dim=-1).mean().item(), total_steps)

        loss_accum["total"] += total.item()
        loss_accum["rec"] += rec_loss.item()
        loss_accum["kl"] += kld_loss.item()
        loss_accum["rew"] += rew_loss.item()
        loss_accum["pb"] += pb_loss.item()
        loss_accum["bisim"] += bisim_loss_val.item()
        loss_accum["bisim_weighted"] = loss_accum.get("bisim_weighted", 0.0) + bisim_weight * bisim_loss_val.item()
        loss_accum["aux_pix"] = loss_accum.get("aux_pix", 0.0) + aux_pix_loss.item()

    n_updates = args.train_steps
    writer.add_scalar("loss/total", loss_accum["total"] / n_updates, total_steps)
    writer.add_scalar("loss/reconstruction", loss_accum["rec"] / n_updates, total_steps)
    writer.add_scalar("loss/kl_divergence", loss_accum["kl"] / n_updates, total_steps)
    writer.add_scalar("loss/reward", loss_accum["rew"] / n_updates, total_steps)
    writer.add_scalar("loss/pullback_curvature", loss_accum["pb"] / n_updates, total_steps)
    writer.add_scalar("loss/bisimulation", loss_accum["bisim"] / n_updates, total_steps)
    writer.add_scalar("loss/bisimulation_weighted", loss_accum.get("bisim_weighted", 0.0) / n_updates, total_steps)

    if args.use_feature_decoder:
        writer.add_scalar("loss/feature_rec", loss_accum["rec"] / n_updates, total_steps)
        writer.add_scalar("loss/aux_pixel", loss_accum.get("aux_pix", 0.0) / n_updates, total_steps)
        writer.add_scalar("config/mask_ratio", compute_mask_ratio(args, total_steps), total_steps)
        writer.add_scalar("config/feature_rec_scale", args.feature_rec_scale, total_steps)
        if args.phi_warmup_steps > 0:
            phi_warmup_factor = min(1.0, total_steps / args.phi_warmup_steps)
            writer.add_scalar("config/phi_warmup_factor", phi_warmup_factor, total_steps)

    writer.add_scalar("loss/bisim_weight", bisim_weight, total_steps)

    if args.pullback_bisim and n_bisim_computations > 0:
        n_bisim_updates = n_bisim_computations
        writer.add_scalar("bisim_stats/dz_mean", bisim_stats_accum["dz_mean"] / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/dz_std", bisim_stats_accum["dz_std"] / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/target_mean", bisim_stats_accum["target_mean"] / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/target_std", bisim_stats_accum["target_std"] / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/abs_err_mean", bisim_stats_accum["abs_err_mean"] / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/rel_err", bisim_stats_accum["rel_err"] / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/corr", bisim_stats_accum["corr"] / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/edge_weight_mean", bisim_stats_accum.get("edge_weight_mean", 0.0) / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/edge_weight_std", bisim_stats_accum.get("edge_weight_std", 0.0) / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/feat_norm_mean", bisim_stats_accum.get("feat_norm_mean", 0.0) / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/feat_std", bisim_stats_accum.get("feat_std", 0.0) / n_bisim_updates, total_steps)
        writer.add_scalar("bisim_stats/target_scale_ema", bisim_stats_accum.get("target_scale_ema", 0.0) / n_bisim_updates, total_steps)

    writer.add_scalar("grad_norm/total", grad_norm_accum["total"] / n_updates, total_steps)
    writer.add_scalar("grad_norm/encoder", grad_norm_accum["encoder"] / n_updates, total_steps)
    writer.add_scalar("grad_norm/decoder", grad_norm_accum["decoder"] / n_updates, total_steps)
    writer.add_scalar("grad_norm/rssm", grad_norm_accum["rssm"] / n_updates, total_steps)
    writer.add_scalar("grad_norm/reward", grad_norm_accum["reward"] / n_updates, total_steps)
    writer.add_scalar("grad_norm/bisim_weighted", grad_norm_accum["bisim_weighted"] / n_updates, total_steps)

    if grad_norm_accum["total"] > 0:
        bisim_ratio = grad_norm_accum["bisim_weighted"] / grad_norm_accum["total"]
        writer.add_scalar("grad_norm/bisim_ratio", bisim_ratio, total_steps)
    if grad_norm_accum["decoder"] > 0:
        decoder_bisim_ratio = grad_norm_accum["bisim_weighted"] / grad_norm_accum["decoder"]
        writer.add_scalar("grad_norm/bisim_to_decoder_ratio", decoder_bisim_ratio, total_steps)

    return feature_bank, bank_ptr, target_scale_ema, phi_update_counter


# ===============================
# Main
# ===============================

def main(args):
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    run_name = make_run_name(args) if args.run_name == "train" else args.run_name
    writer = SummaryWriter(log_dir=f"{args.log_dir}/{run_name}")
    print(f"TensorBoard run name: {run_name}")
    writer.add_text("hyperparameters", str(vars(args)), 0)

    env = make_env(args.env_id, img_size=(args.img_size, args.img_size), num_stack=1)
    obs, _ = env.reset()

    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    action_repeat = args.action_repeat if args.action_repeat > 0 else ENV_ACTION_REPEAT.get(args.env_id, 2)
    print(f"Environment: {args.env_id}, Action repeat: {action_repeat}")

    models = build_models(args, obs_channels=C, act_dim=act_dim, device=device)
    optimizers = build_optimizers(args, models)
    replay = ReplayBuffer(args.replay_buff_capacity, obs_shape=(H, W, C), act_dim=act_dim)

    feature_bank, bank_ptr = init_feature_bank(args, device)
    target_scale_ema = 1.0
    phi_update_counter = 0

    total_steps, start_episode = load_checkpoint_if_needed(args, models, device)

    total_steps = seed_replay_buffer(env, replay, action_repeat, args.seed_episodes, total_steps)
    log_initial_visualization(args, writer, replay, models, device)

    if start_episode > 0:
        print(f"\nResuming training from episode {start_episode} to {args.max_episodes}...")
    else:
        print(f"\nStarting training for {args.max_episodes} episodes...")

    for episode in range(start_episode, args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0

        with torch.no_grad():
            obs_t = obs_to_tensor(obs, device, args.bit_depth)
            enc = models.encoder(obs_t)
            h_state, s_state = models.rssm.get_init_state(enc)

        while not done:
            models.encoder.eval()
            models.rssm.eval()
            models.reward_model.eval()
            action = cem_plan_action_rssm(
                rssm=models.rssm,
                reward_model=models.reward_model,
                h_t=h_state,
                s_t=s_state,
                act_low=act_low,
                act_high=act_high,
                horizon=args.plan_horizon,
                candidates=args.plan_candidates,
                iters=args.plan_iters,
                top_k=args.plan_top_k,
                device=device,
                explore=True,
                explore_noise=args.explore_noise,
            )
            action = np.asarray(action, dtype=np.float32).reshape(act_dim,)

            total_reward = 0.0
            for _ in range(action_repeat):
                next_obs, r, term, trunc, _ = env.step(action)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)

            replay.add(
                obs=np.ascontiguousarray(obs, dtype=np.uint8),
                action=action,
                reward=total_reward,
                next_obs=np.ascontiguousarray(next_obs, dtype=np.uint8),
                done=done,
            )

            ep_return += total_reward
            ep_steps += 1
            total_steps += 1
            obs = next_obs

            with torch.no_grad():
                obs_t = obs_to_tensor(obs, device, args.bit_depth)
                enc = models.encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = models.rssm.observe_step(enc, act_t, h_state, s_state, sample=False)

            if total_steps % args.collect_interval == 0 and replay.size > (args.seq_len + 2):
                (
                    feature_bank,
                    bank_ptr,
                    target_scale_ema,
                    phi_update_counter,
                ) = train_world_model(
                    args,
                    models,
                    optimizers,
                    replay,
                    writer,
                    total_steps,
                    feature_bank,
                    bank_ptr,
                    target_scale_ema,
                    phi_update_counter,
                    device,
                )

                if args.visualize_interval > 0 and total_steps % args.visualize_interval == 0:
                    log_initial_visualization(args, writer, replay, models, device)

        writer.add_scalar("train/episode_return", ep_return, episode)
        writer.add_scalar("train/episode_steps", ep_steps, episode)
        writer.add_scalar("train/total_steps", total_steps, episode)
        print(
            f"Episode {episode + 1}/{args.max_episodes}: return = {ep_return:.2f}, steps = {ep_steps}, total_steps = {total_steps}"
        )

        if (episode + 1) % args.eval_interval == 0:
            plan_kwargs = dict(
                act_low=act_low,
                act_high=act_high,
                horizon=args.plan_horizon,
                candidates=args.plan_candidates,
                iters=args.plan_iters,
                top_k=args.plan_top_k,
            )
            mean_ret, std_ret = evaluate_planner_rssm(
                env_id=args.env_id,
                img_size=args.img_size,
                encoder=models.encoder,
                rssm=models.rssm,
                reward_model=models.reward_model,
                plan_kwargs=plan_kwargs,
                episodes=args.eval_episodes,
                seed=args.seed + 100,
                device=device,
                bit_depth=args.bit_depth,
                action_repeat=action_repeat,
            )
            print(f"  [Eval] return: {mean_ret:.2f} ± {std_ret:.2f}")
            writer.add_scalar("eval/mean_return", mean_ret, episode)
            writer.add_scalar("eval/std_return", std_ret, episode)

        if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
            checkpoint_dir = f"{args.log_dir}/{run_name}"
            checkpoint_path = f"{checkpoint_dir}/model_ep{episode + 1}.pt"
            print(f"  Saving checkpoint to: {checkpoint_path}")
            checkpoint_dict = {
                "encoder": models.encoder.state_dict(),
                "decoder": models.decoder.state_dict(),
                "rssm": models.rssm.state_dict(),
                "reward_model": models.reward_model.state_dict(),
                "args": vars(args),
                "total_steps": total_steps,
                "episode": episode + 1,
            }
            if args.use_feature_decoder:
                checkpoint_dict["vis_decoder"] = models.vis_decoder.state_dict()
                checkpoint_dict["phi_net"] = models.phi_net.state_dict()
            torch.save(checkpoint_dict, checkpoint_path)

    env.close()
    writer.close()

    checkpoint_dir = f"{args.log_dir}/{run_name}"
    checkpoint_path = f"{checkpoint_dir}/model_final.pt"
    print(f"\nSaving final model checkpoint to: {checkpoint_path}")

    checkpoint_dict = {
        "encoder": models.encoder.state_dict(),
        "decoder": models.decoder.state_dict(),
        "rssm": models.rssm.state_dict(),
        "reward_model": models.reward_model.state_dict(),
        "args": vars(args),
        "total_steps": total_steps,
        "episode": args.max_episodes,
    }
    if args.use_feature_decoder:
        checkpoint_dict["vis_decoder"] = models.vis_decoder.state_dict()
        checkpoint_dict["phi_net"] = models.phi_net.state_dict()
    torch.save(checkpoint_dict, checkpoint_path)
    print(f"Training complete! TensorBoard logs and checkpoint saved to: {args.log_dir}/{run_name}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
