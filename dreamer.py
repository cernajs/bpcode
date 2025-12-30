import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast
from torch.amp import GradScaler

from utils import (PixelObsWrapper, DMControlWrapper, ReplayBuffer, get_device, set_seed, 
                   preprocess_img, bottle, make_env, ENV_ACTION_REPEAT)

from models import ConvEncoder, ConvDecoder, RSSM, RewardModel, Actor, ValueModel


# ===============================
#  Losses / Regularizers
# ===============================

def compute_pullback_curvature_loss(decoder, h, s, num_projections=4, detach_features=True):
    """
    Hutchinson estimate of || J^T J - I ||_F^2 for decoder Jacobian wrt latent.
    """
    if detach_features:
        h = h.detach().requires_grad_(True)
        s = s.detach().requires_grad_(True)
    else:
        h = h.requires_grad_(True)
        s = s.requires_grad_(True)

    loss = 0.0
    for _ in range(num_projections):
        u_h = (torch.randint(0, 2, h.shape, device=h.device) * 2 - 1).to(h.dtype)
        u_s = (torch.randint(0, 2, s.shape, device=s.device) * 2 - 1).to(s.dtype)

        recon, (Ju_h, Ju_s) = torch.autograd.functional.jvp(
            lambda hh, ss: decoder(hh, ss), (h, s), (u_h, u_s), create_graph=True
        )

        Gu_h = torch.autograd.grad(
            outputs=recon, inputs=h, grad_outputs=Ju_h if isinstance(Ju_h, torch.Tensor) else recon, 
            create_graph=True, retain_graph=True
        )[0]
        Gu_s = torch.autograd.grad(
            outputs=recon, inputs=s, grad_outputs=Ju_s if isinstance(Ju_s, torch.Tensor) else recon,
            create_graph=True
        )[0]

        Au_h = Gu_h - u_h
        Au_s = Gu_s - u_s
        loss = loss + (Au_h.pow(2).sum(dim=1)).mean() + (Au_s.pow(2).sum(dim=1)).mean()

    return loss / float(num_projections)


def bisimulation_loss(z1, z2, r1, r2, next_z1, next_z2, gamma=0.99):
    dz = torch.norm(z1 - z2.detach(), p=2, dim=1)
    with torch.no_grad():
        dnext = torch.norm(next_z1 - next_z2, p=2, dim=1)
        dr = torch.abs(r1 - r2).view(-1)
        target = dr + gamma * dnext
    return F.mse_loss(dz, target)


# ===============================
#  Pullback Geodesic Bisimulation
# ===============================

def pullback_distance_full(decoder, h1, s1, h2, s2, create_graph=True):
    """
    Computes || J_{h,s} g(h1, s1) · ([h1, s1] - [h2, s2]) ||_2
    """
    delta_h = (h1 - h2)
    delta_s = (s1 - s2)
    
    def decoder_wrapper(h, s):
        return decoder(h, s)
    
    _, Jdelta = torch.autograd.functional.jvp(
        decoder_wrapper,
        (h1, s1),
        (delta_h, delta_s),
        create_graph=create_graph,
    )
    Jdelta = Jdelta.reshape(Jdelta.size(0), -1)
    return torch.sqrt((Jdelta * Jdelta).mean(dim=1) + 1e-8)


def softmin(x, trough, tau=0.1):
    x = torch.clamp(x, max=1e6)
    trough = torch.clamp(trough, max=1e6)
    return -tau * torch.logsumexp(torch.stack([-x/tau, -trough/tau], dim=0), dim=0)


def floyd_warshall_minplus(W):
    """Differentiable all-pairs shortest paths."""
    D = W
    B = D.size(0)
    for k in range(B):
        trough = D[:, k:k+1] + D[k:k+1, :]
        D = softmin(D, trough)
    return D


def geodesic_pb_knn_slice(decoder, h_t, z_t, targets, k=3, create_graph=True, inf=1e9):
    """Compute geodesic distances via kNN graph in a single time slice."""
    B = z_t.size(0)
    device = z_t.device
    dtype = z_t.dtype

    with torch.no_grad():
        dist2 = torch.cdist(z_t, z_t, p=2).pow(2)
        dist2.fill_diagonal_(float("inf"))
        knn = dist2.topk(k, largest=False).indices

    src = torch.arange(B, device=device).repeat_interleave(k)
    dst = knn.reshape(-1)

    w = pullback_distance_full(
        decoder,
        h1=h_t[src],
        s1=z_t[src],
        h2=h_t[dst].detach(),
        s2=z_t[dst].detach(),
        create_graph=create_graph
    )

    W = torch.full((B, B), inf, device=device, dtype=dtype)
    W.fill_diagonal_(0.0)
    W[src, dst] = w
    W = torch.minimum(W, W.T)
    D = floyd_warshall_minplus(W)
    
    dz_t = D[torch.arange(B, device=device), targets]
    return dz_t


def geodesic_pb_knn(decoder, h, z, perm, k=10, time_subsample=None, t_idx=None, 
                    create_graph=True, return_mask=False):
    """Compute geodesic distances across time with optional subsampling."""
    B, Tm1, _ = z.shape
    device = z.device

    if t_idx is None:
        if time_subsample is None or time_subsample >= Tm1:
            t_idx = torch.arange(Tm1, device=device)
        else:
            t_idx = torch.randint(0, Tm1, (time_subsample,), device=device)

    dz = torch.zeros((B, Tm1), device=device, dtype=z.dtype)
    mask = torch.zeros((Tm1,), device=device, dtype=torch.bool)
    
    for t in t_idx.tolist():
        dz[:, t] = geodesic_pb_knn_slice(
            decoder=decoder,
            h_t=h[:, t],
            z_t=z[:, t],
            targets=perm,
            k=k,
            create_graph=create_graph
        )
        mask[t] = True

    if return_mask:
        return dz.reshape(-1), mask.repeat(B), t_idx
    return dz.reshape(-1)


# ===============================
#  DreamerV1 Imagination & Returns
# ===============================

def imagine_ahead(rssm, actor, h_init, s_init, horizon=15):
    """
    Rollout imagined trajectories using the actor policy.
    Returns states and actions for value learning.
    """
    B = h_init.size(0)
    device = h_init.device
    
    h_list = [h_init]
    s_list = [s_init]
    a_list = []
    
    h, s = h_init, s_init
    
    for _ in range(horizon):
        # Get action from actor
        with torch.no_grad():
            action, _ = actor.get_action(h, s, deterministic=False)
        
        # Transition dynamics (prior, no observation)
        h = rssm.deterministic_state_fwd(h, s, action)
        s = rssm.state_prior(h, sample=True)
        
        h_list.append(h)
        s_list.append(s)
        a_list.append(action)
    
    # Stack: [H+1, B, D] -> [B, H+1, D]
    h_imag = torch.stack(h_list, dim=1)
    s_imag = torch.stack(s_list, dim=1)
    a_imag = torch.stack(a_list, dim=1)
    
    return h_imag, s_imag, a_imag


def compute_lambda_returns(rewards, values, gamma=0.99, lambda_=0.95):
    """
    Compute λ-returns for value learning.
    rewards: [B, H] - predicted rewards
    values: [B, H+1] - predicted values (includes bootstrap)
    Returns: [B, H] λ-returns
    """
    B, H = rewards.shape
    device = rewards.device
    
    # Bootstrap value
    next_values = values[:, 1:]  # [B, H]
    
    # Compute λ-returns backwards
    lambda_returns = torch.zeros_like(rewards)
    last_lambda = values[:, -1]  # Bootstrap
    
    for t in reversed(range(H)):
        lambda_returns[:, t] = rewards[:, t] + gamma * (
            (1 - lambda_) * next_values[:, t] + lambda_ * last_lambda
        )
        last_lambda = lambda_returns[:, t]
    
    return lambda_returns


# ===============================
#  Evaluation
# ===============================

@torch.no_grad()
def evaluate_policy(
    env_id, img_size, encoder, rssm, actor,
    episodes=10, seed=0, device="cpu", bit_depth=5, action_repeat=1
):
    """Evaluate learned policy on environment."""
    env = make_env(env_id, img_size=(img_size, img_size), num_stack=1)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    encoder.eval()
    rssm.eval()
    actor.eval()

    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        
        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
        obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=bit_depth)
        enc = encoder(obs_t)
        h_state, s_state = rssm.get_init_state(enc)

        while not done:
            action, _ = actor.get_action(h_state, s_state, deterministic=True)
            action_np = action.squeeze(0).cpu().numpy().astype(np.float32)
            
            total_reward = 0.0
            for _ in range(action_repeat):
                obs, r, term, trunc, _ = env.step(action_np)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)
            ep_ret += total_reward
            
            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=bit_depth)
            enc = encoder(obs_t)
            act_t = action
            h_state, s_state, _, _ = rssm.observe_step(enc, act_t, h_state, s_state, sample=False)

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ===============================
#  Args
# ===============================

def build_parser():
    p = argparse.ArgumentParser()
    
    # Environment
    p.add_argument("--env_id", type=str, default="cheetah-run")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    
    # Training
    p.add_argument("--batch_size", type=int, default=50, help="DreamerV1: 50")
    p.add_argument("--seq_len", type=int, default=50, help="DreamerV1: 50")
    p.add_argument("--max_episodes", type=int, default=1000)
    p.add_argument("--seed_episodes", type=int, default=5)
    p.add_argument("--collect_interval", type=int, default=100)
    p.add_argument("--train_steps", type=int, default=100)
    p.add_argument("--action_repeat", type=int, default=0)
    p.add_argument("--replay_buff_capacity", type=int, default=1_000_000)
    
    # Optimization
    p.add_argument("--model_lr", type=float, default=6e-4, help="DreamerV1: 6e-4")
    p.add_argument("--actor_lr", type=float, default=8e-5, help="DreamerV1: 8e-5")
    p.add_argument("--value_lr", type=float, default=8e-5, help="DreamerV1: 8e-5")
    p.add_argument("--adam_eps", type=float, default=1e-5)
    p.add_argument("--grad_clip_norm", type=float, default=100.0, help="DreamerV1: 100")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lambda_", type=float, default=0.95, help="DreamerV1 λ-returns")
    
    # Mixed precision for RTX 5080
    p.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")
    
    # Model architecture
    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--hidden_dim", type=int, default=200)
    p.add_argument("--actor_hidden_dim", type=int, default=400, help="DreamerV1: 400")
    p.add_argument("--value_hidden_dim", type=int, default=400, help="DreamerV1: 400")
    
    # World model losses
    p.add_argument("--kl_weight", type=float, default=1.0)
    p.add_argument("--kl_free_nats", type=float, default=3.0)
    
    # DreamerV1 imagination
    p.add_argument("--imagination_horizon", type=int, default=15, help="DreamerV1: 15")
    
    # Exploration
    p.add_argument("--expl_amount", type=float, default=0.3, help="Action noise during collection")
    p.add_argument("--expl_decay", type=float, default=0.0, help="Exploration decay rate")
    p.add_argument("--expl_min", type=float, default=0.0)
    
    # Actor-critic stability
    p.add_argument("--actor_warmup_steps", type=int, default=0, help="Steps before actor training starts")
    p.add_argument("--advantage_clip", type=float, default=10.0, help="Clip advantages before normalization")
    
    # Optional regularizers (your novelty)
    p.add_argument("--pb_curvature_weight", type=float, default=0.0)
    p.add_argument("--pb_curvature_projections", type=int, default=2)
    p.add_argument("--pb_detach_features", action="store_true")
    
    p.add_argument("--pullback_bisim", action="store_true")
    p.add_argument("--bisimulation_weight", type=float, default=0.0)
    p.add_argument("--bisimulation_warmup", type=int, default=50_000)
    p.add_argument("--bisimulation_ramp", type=int, default=100_000)
    p.add_argument("--geo_knn_k", type=int, default=3)
    p.add_argument("--geo_time_subsample", type=int, default=8)
    
    # Evaluation
    p.add_argument("--eval_episodes", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=10)
    
    # Logging
    p.add_argument("--run_name", type=str, default="dreamerv1")
    p.add_argument("--log_dir", type=str, default="runs")

    return p


def make_run_name(args):
    """Generate a descriptive run name."""
    parts = [args.env_id, "dreamerv1"]
    
    if args.bisimulation_weight > 0:
        parts.append(f"bisim{args.bisimulation_weight}")
    if args.pullback_bisim:
        parts.append("pullback_bisim")
    if args.pb_curvature_weight > 0:
        parts.append(f"jacobreg{args.pb_curvature_weight}")
    if args.use_amp:
        parts.append("amp")
    
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


# ===============================
#  Main Training Loop
# ===============================

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Check for CUDA and set up AMP
    use_amp = args.use_amp and device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None
    print(f"Mixed precision training: {use_amp}")
    
    # Run name
    if args.run_name == "dreamerv1":
        run_name = make_run_name(args)
    else:
        run_name = args.run_name
    
    writer = SummaryWriter(log_dir=f"{args.log_dir}/{run_name}")
    print(f"TensorBoard run name: {run_name}")
    writer.add_text("hyperparameters", str(vars(args)), 0)

    # Create environment
    env = make_env(args.env_id, img_size=(args.img_size, args.img_size), num_stack=1)
    obs, _ = env.reset()

    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    action_repeat = args.action_repeat if args.action_repeat > 0 else ENV_ACTION_REPEAT.get(args.env_id, 2)
    print(f"Environment: {args.env_id}, Action repeat: {action_repeat}")

    # ========================================
    # Models - DreamerV1 Architecture
    # ========================================
    encoder = ConvEncoder(embedding_size=args.embed_dim, in_channels=C).to(device)
    decoder = ConvDecoder(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        embedding_size=args.embed_dim,
        out_channels=C
    ).to(device)
    rssm = RSSM(
        stoch_dim=args.stoch_dim,
        deter_dim=args.deter_dim,
        act_dim=act_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    reward_model = RewardModel(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        hidden_dim=args.hidden_dim
    ).to(device)
    
    # Actor-Critic (DreamerV1)
    actor = Actor(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        act_dim=act_dim,
        hidden_dim=args.actor_hidden_dim
    ).to(device)
    value_model = ValueModel(
        state_size=args.deter_dim,
        latent_size=args.stoch_dim,
        hidden_dim=args.value_hidden_dim
    ).to(device)

    # Separate optimizers (DreamerV1 style)
    world_model_params = (
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(rssm.parameters()) + 
        list(reward_model.parameters())
    )
    model_optim = torch.optim.Adam(world_model_params, lr=args.model_lr, eps=args.adam_eps)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr, eps=args.adam_eps)
    value_optim = torch.optim.Adam(value_model.parameters(), lr=args.value_lr, eps=args.adam_eps)

    replay = ReplayBuffer(args.replay_buff_capacity, obs_shape=(H, W, C), act_dim=act_dim)
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    total_steps = 0
    expl_amount = args.expl_amount
    
    # ========================================
    # Phase 1: Seed buffer with random episodes
    # ========================================
    print(f"Seeding replay buffer with {args.seed_episodes} random episodes...")
    for seed_ep in range(args.seed_episodes):
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
                done=done
            )
            ep_return += total_reward
            total_steps += 1
            obs = next_obs
        
        print(f"  Seed episode {seed_ep + 1}/{args.seed_episodes}: return = {ep_return:.2f}")
    
    # ========================================
    # Phase 2: Main DreamerV1 Training Loop
    # ========================================
    print(f"\nStarting DreamerV1 training for {args.max_episodes} episodes...")
    
    for episode in range(args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0
        
        # Initialize belief state
        with torch.no_grad():
            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
            obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=args.bit_depth)
            enc = encoder(obs_t)
            h_state, s_state = rssm.get_init_state(enc)
        
        while not done:
            # -------- Action Selection with Actor --------
            encoder.eval()
            rssm.eval()
            actor.eval()
            
            with torch.no_grad():
                action_t, _ = actor.get_action(h_state, s_state, deterministic=False)
                
                # Add exploration noise
                if expl_amount > 0:
                    action_t = action_t + expl_amount * torch.randn_like(action_t)
                    action_t = torch.clamp(action_t, -1.0, 1.0)
                
                action = action_t.squeeze(0).cpu().numpy().astype(np.float32)

            # -------- Environment step --------
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
                done=done
            )

            ep_return += total_reward
            ep_steps += 1
            total_steps += 1
            obs = next_obs

            # Update belief state
            with torch.no_grad():
                obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device)
                obs_t = obs_t.permute(2, 0, 1).unsqueeze(0)
                preprocess_img(obs_t, depth=args.bit_depth)
                enc = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(enc, act_t, h_state, s_state, sample=False)

            # -------- Training --------
            if total_steps % args.collect_interval == 0 and replay.size > (args.seq_len + 2):
                encoder.train()
                decoder.train()
                rssm.train()
                reward_model.train()
                actor.train()
                value_model.train()
                
                loss_accum = {
                    "model_total": 0.0, "rec": 0.0, "kl": 0.0, "rew": 0.0,
                    "pb": 0.0, "bisim": 0.0, "actor": 0.0, "value": 0.0,
                    "adv_mean": 0.0, "adv_std": 0.0, "adv_min": 0.0, "adv_max": 0.0
                }
                
                for upd in range(args.train_steps):
                    # Sample sequences
                    batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)
                    
                    obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                    act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)
                    rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)

                    B, T1 = rew_seq.shape
                    T = T1 - 1

                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                    preprocess_img(x, depth=args.bit_depth)

                    # ========================================
                    # World Model Training
                    # ========================================
                    with autocast(device_type='cuda', enabled=use_amp):
                        e_t = bottle(encoder, x)
                        h_t_train, s_t_train = rssm.get_init_state(e_t[:, 0])

                        states = []
                        priors = []
                        posteriors = []
                        posterior_samples = []

                        for t in range(T):
                            h_t_train = rssm.deterministic_state_fwd(h_t_train, s_t_train, act_seq[:, t])
                            states.append(h_t_train)
                            priors.append(rssm.state_prior(h_t_train))
                            posteriors.append(rssm.state_posterior(h_t_train, e_t[:, t + 1]))
                            post_mean, post_std = posteriors[-1]
                            s_t_train = post_mean + torch.randn_like(post_std) * post_std
                            posterior_samples.append(s_t_train)

                        h_seq = torch.stack(states, dim=1)
                        s_seq = torch.stack(posterior_samples, dim=1)

                        prior_means = torch.stack([p[0] for p in priors], dim=0)
                        prior_stds = torch.stack([p[1] for p in priors], dim=0)
                        post_means = torch.stack([p[0] for p in posteriors], dim=0)
                        post_stds = torch.stack([p[1] for p in posteriors], dim=0)

                        prior_dist = Normal(prior_means, prior_stds)
                        posterior_dist = Normal(post_means, post_stds)

                        # Reconstruction loss
                        recon = bottle(decoder, h_seq, s_seq)
                        target = x[:, 1:T+1]
                        rec_loss = F.mse_loss(recon, target, reduction='none').sum((2, 3, 4)).mean()

                        # KL loss with free nats
                        kld_loss = torch.max(
                            kl_divergence(posterior_dist, prior_dist).sum(-1),
                            free_nats
                        ).mean()

                        # Reward loss
                        rew_pred = bottle(reward_model, h_seq, s_seq)
                        rew_target = rew_seq[:, :T]
                        rew_loss = F.mse_loss(rew_pred, rew_target)

                        # Optional pullback curvature regularization
                        pb_loss = torch.zeros((), device=device)
                        if args.pb_curvature_weight > 0.0:
                            h_flat = h_seq.reshape(-1, args.deter_dim)
                            s_flat = s_seq.reshape(-1, args.stoch_dim)
                            pb_loss = compute_pullback_curvature_loss(
                                decoder=decoder,
                                h=h_flat,
                                s=s_flat,
                                num_projections=args.pb_curvature_projections,
                                detach_features=args.pb_detach_features
                            )

                        # Optional bisimulation loss
                        bisim_loss_val = torch.zeros((), device=device)
                        if args.bisimulation_weight > 0.0 and T > 2:
                            h = h_seq[:, :-1]
                            hn = h_seq[:, 1:]
                            z = s_seq[:, :-1]
                            zn = s_seq[:, 1:]
                            r = rew_seq[:, :T-1]

                            if not args.pullback_bisim:
                                z = F.layer_norm(z, (z.size(-1),))
                                zn = F.layer_norm(zn, (zn.size(-1),))

                            perm = torch.randperm(B, device=device)

                            z1 = z.reshape(-1, z.size(-1))
                            z2 = z[perm].reshape(-1, z.size(-1))
                            n1 = zn.reshape(-1, zn.size(-1))
                            n2 = zn[perm].reshape(-1, zn.size(-1))
                            r1 = r.reshape(-1)
                            r2 = r[perm].reshape(-1)

                            if args.pullback_bisim:
                                dz, dz_mask, t_idx = geodesic_pb_knn(
                                    decoder=decoder, h=h, z=z, perm=perm,
                                    k=args.geo_knn_k,
                                    time_subsample=args.geo_time_subsample,
                                    create_graph=True, return_mask=True
                                )
                            else:
                                dz = torch.norm(z1 - z2.detach(), p=2, dim=1)

                            with torch.no_grad():
                                if args.pullback_bisim:
                                    dnext, _, _ = geodesic_pb_knn(
                                        decoder=decoder, h=hn, z=zn, perm=perm,
                                        k=args.geo_knn_k,
                                        time_subsample=args.geo_time_subsample,
                                        t_idx=t_idx, create_graph=False, return_mask=True
                                    )
                                else:
                                    dnext = torch.norm(n1 - n2, p=2, dim=1)
                                dr = (r1 - r2).abs()
                                target_bisim = dr + args.gamma * dnext

                            if args.pullback_bisim:
                                mask = dz_mask.bool()
                                target_masked = target_bisim[mask]
                                dz_masked = dz[mask]
                                scale = target_masked.detach().abs().mean().clamp_min(1e-6)
                                bisim_loss_val = F.mse_loss(dz_masked / scale, target_masked / scale)
                            else:
                                scale = target_bisim.detach().abs().mean().clamp_min(1e-6)
                                bisim_loss_val = F.mse_loss(dz / scale, target_bisim / scale)

                        # Bisim warmup schedule
                        def bisim_lambda(ts, warmup=50_000, ramp=100_000, tgt=0.03):
                            if ts < warmup:
                                return 0.0
                            return tgt * min(1.0, (ts - warmup) / float(ramp))

                        bisim_weight = bisim_lambda(
                            total_steps,
                            warmup=args.bisimulation_warmup,
                            ramp=args.bisimulation_ramp,
                            tgt=args.bisimulation_weight
                        )

                        model_loss = (
                            rec_loss +
                            args.kl_weight * kld_loss +
                            rew_loss +
                            args.pb_curvature_weight * pb_loss +
                            bisim_weight * bisim_loss_val
                        )

                    # World model backward pass
                    model_optim.zero_grad()
                    if use_amp:
                        scaler.scale(model_loss).backward()
                        scaler.unscale_(model_optim)
                        torch.nn.utils.clip_grad_norm_(world_model_params, args.grad_clip_norm)
                        scaler.step(model_optim)
                        scaler.update()
                    else:
                        model_loss.backward()
                        torch.nn.utils.clip_grad_norm_(world_model_params, args.grad_clip_norm)
                        model_optim.step()

                    # ========================================
                    # Actor-Critic Training (Imagination)
                    # ========================================
                    with autocast(device_type='cuda', enabled=use_amp):
                        with torch.no_grad():
                            # Start imagination from posterior states
                            h_start = h_seq[:, -1].detach()
                            s_start = s_seq[:, -1].detach()

                        # Imagination rollout
                        h_imag_list = [h_start]
                        s_imag_list = [s_start]
                        a_imag_list = []
                        log_prob_list = []
                        
                        h_im, s_im = h_start, s_start
                        for _ in range(args.imagination_horizon):
                            action_im, log_prob = actor.get_action(h_im, s_im, deterministic=False)
                            a_imag_list.append(action_im)
                            log_prob_list.append(log_prob)
                            
                            h_im = rssm.deterministic_state_fwd(h_im, s_im, action_im)
                            s_im = rssm.state_prior(h_im, sample=True)
                            
                            h_imag_list.append(h_im)
                            s_imag_list.append(s_im)

                        h_imag = torch.stack(h_imag_list, dim=1)  # [B, H+1, D]
                        s_imag = torch.stack(s_imag_list, dim=1)  # [B, H+1, D]
                        log_probs = torch.stack(log_prob_list, dim=1)  # [B, H]

                        # Predict rewards and values
                        with torch.no_grad():
                            rewards_imag = bottle(reward_model, h_imag[:, :-1], s_imag[:, :-1])  # [B, H]
                        
                        values_imag = bottle(value_model, h_imag, s_imag)  # [B, H+1]
                        
                        # Store values BEFORE update for stable advantage computation
                        with torch.no_grad():
                            values_for_advantages = values_imag.detach().clone()

                        # Compute λ-returns
                        with torch.no_grad():
                            lambda_returns = compute_lambda_returns(
                                rewards_imag, values_imag.detach(),
                                gamma=args.gamma, lambda_=args.lambda_
                            )

                        # Value loss
                        value_loss = F.mse_loss(values_imag[:, :-1], lambda_returns)

                    # Value optimizer step
                    value_optim.zero_grad()
                    if use_amp:
                        scaler.scale(value_loss).backward(retain_graph=True)
                        scaler.unscale_(value_optim)
                        torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm)
                        scaler.step(value_optim)
                        # Don't update scaler here - we still need to do actor backward
                    else:
                        value_loss.backward(retain_graph=True)
                        torch.nn.utils.clip_grad_norm_(value_model.parameters(), args.grad_clip_norm)
                        value_optim.step()

                    # Actor loss: maximize imagined returns
                    # Use advantage = λ-returns - baseline (use values from BEFORE update for stability)
                    # Skip actor training during warmup period
                    if total_steps >= args.actor_warmup_steps:
                        with autocast(device_type='cuda', enabled=use_amp):
                            with torch.no_grad():
                                # Use stored values for stable advantages
                                advantages = lambda_returns - values_for_advantages[:, :-1]
                                # Clip advantages before normalization to prevent explosion
                                advantages = torch.clamp(advantages, -args.advantage_clip, args.advantage_clip)
                                # Normalize advantages with larger epsilon for stability
                                adv_std = advantages.std()
                                if adv_std > 1e-4:  # Only normalize if there's meaningful variance
                                    advantages = (advantages - advantages.mean()) / (adv_std + 1e-5)
                                else:
                                    advantages = advantages - advantages.mean()
                            
                            actor_loss = -(log_probs * advantages).mean()

                        actor_optim.zero_grad()
                        if use_amp:
                            scaler.scale(actor_loss).backward()
                            scaler.unscale_(actor_optim)
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip_norm)
                            scaler.step(actor_optim)
                            scaler.update()  # Update scaler after all backward passes
                        else:
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip_norm)
                            actor_optim.step()
                    else:
                        # During warmup, still update scaler but skip actor training
                        actor_loss = torch.zeros((), device=device)
                        if use_amp:
                            scaler.update()

                    # Accumulate losses
                    loss_accum["model_total"] += model_loss.item()
                    loss_accum["rec"] += rec_loss.item()
                    loss_accum["kl"] += kld_loss.item()
                    loss_accum["rew"] += rew_loss.item()
                    loss_accum["pb"] += pb_loss.item()
                    loss_accum["bisim"] += bisim_loss_val.item()
                    loss_accum["actor"] += actor_loss.item()
                    loss_accum["value"] += value_loss.item()
                    
                    # Log advantage statistics for debugging
                    with torch.no_grad():
                        raw_adv = lambda_returns - values_for_advantages[:, :-1]
                        loss_accum["adv_mean"] += raw_adv.mean().item()
                        loss_accum["adv_std"] += raw_adv.std().item()
                        loss_accum["adv_min"] += raw_adv.min().item()
                        loss_accum["adv_max"] += raw_adv.max().item()

                # Log losses
                n_updates = args.train_steps
                writer.add_scalar("loss/model_total", loss_accum["model_total"] / n_updates, total_steps)
                writer.add_scalar("loss/reconstruction", loss_accum["rec"] / n_updates, total_steps)
                writer.add_scalar("loss/kl_divergence", loss_accum["kl"] / n_updates, total_steps)
                writer.add_scalar("loss/reward", loss_accum["rew"] / n_updates, total_steps)
                writer.add_scalar("loss/pullback_curvature", loss_accum["pb"] / n_updates, total_steps)
                writer.add_scalar("loss/bisimulation", loss_accum["bisim"] / n_updates, total_steps)
                writer.add_scalar("loss/actor", loss_accum["actor"] / n_updates, total_steps)
                writer.add_scalar("loss/value", loss_accum["value"] / n_updates, total_steps)
                writer.add_scalar("train/exploration", expl_amount, total_steps)
                
                # Log advantage statistics
                writer.add_scalar("actor/advantage_mean", loss_accum["adv_mean"] / n_updates, total_steps)
                writer.add_scalar("actor/advantage_std", loss_accum["adv_std"] / n_updates, total_steps)
                writer.add_scalar("actor/advantage_min", loss_accum["adv_min"] / n_updates, total_steps)
                writer.add_scalar("actor/advantage_max", loss_accum["adv_max"] / n_updates, total_steps)

        # -------- End of episode --------
        # Decay exploration
        if args.expl_decay > 0:
            expl_amount = max(args.expl_min, expl_amount - args.expl_decay)
        
        writer.add_scalar("train/episode_return", ep_return, episode)
        writer.add_scalar("train/episode_steps", ep_steps, episode)
        writer.add_scalar("train/total_steps", total_steps, episode)
        
        print(f"Episode {episode + 1}/{args.max_episodes}: return = {ep_return:.2f}, steps = {ep_steps}, total = {total_steps}")
        
        # -------- Evaluation --------
        if (episode + 1) % args.eval_interval == 0:
            mean_ret, std_ret = evaluate_policy(
                env_id=args.env_id,
                img_size=args.img_size,
                encoder=encoder,
                rssm=rssm,
                actor=actor,
                episodes=args.eval_episodes,
                seed=args.seed + 100,
                device=device,
                bit_depth=args.bit_depth,
                action_repeat=action_repeat,
            )
            print(f"  [Eval] return: {mean_ret:.2f} ± {std_ret:.2f}")
            
            writer.add_scalar("eval/mean_return", mean_ret, episode)
            writer.add_scalar("eval/std_return", std_ret, episode)

    env.close()
    writer.close()
    print(f"\nTraining complete! TensorBoard logs saved to: {args.log_dir}/{run_name}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
