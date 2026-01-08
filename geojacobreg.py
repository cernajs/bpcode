import argparse
import copy
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from utils import (PixelObsWrapper, DMControlWrapper, ReplayBuffer, get_device, set_seed, 
                   preprocess_img, bottle, make_env, ENV_ACTION_REPEAT)

from models import ConvEncoder, ConvDecoder, RSSM, RewardModel
from vicreg_net import PhiNet, ema_update, phi_augment, vicreg_loss

# ===============================
#  Losses / Regularizers
# ===============================


def bisimulation_loss(z1, z2, r1, r2, next_z1, next_z2, gamma=0.99):
    dz = torch.norm(z1 - z2.detach(), p=2, dim=1)              # [B]
    with torch.no_grad():
        dnext = torch.norm(next_z1 - next_z2, p=2, dim=1) # [B]
        dr = torch.abs(r1 - r2).view(-1)                  # [B]
        target = dr + gamma * dnext
    return F.mse_loss(dz, target)


# ===============================
#  CEM Planner
# ===============================

@torch.no_grad()
def cem_plan_action_rssm(
    rssm, reward_model,
    h_t, s_t,
    act_low, act_high,
    horizon=12, candidates=1000, iters=10,
    top_k=100,
    device=None,
    explore=False,
    explore_noise=0.3,
):
    """
    Key features:
    - No alpha momentum blending (like reference)
    - Reward model takes (h, s) not (z, a)
    - Direct top-k selection
    - Exploration noise when explore=True (0.3 * randn like reference)
    """
    device = device or h_t.device
    act_dim = act_low.shape[0]
    
    act_low_t = torch.tensor(act_low, device=device, dtype=torch.float32)
    act_high_t = torch.tensor(act_high, device=device, dtype=torch.float32)
    
    # Initialize mean and std
    mu = torch.zeros(horizon, act_dim, device=device)
    stddev = torch.ones(horizon, act_dim, device=device)
    
    for _ in range(iters):
        # Sample action sequences
        actions = Normal(mu, stddev).sample((candidates,))  # [N, H, A]
        actions = torch.clamp(actions, act_low_t, act_high_t)
        
        # Rollout with prior dynamics
        rwds = torch.zeros(candidates, device=device)
        h = h_t.expand(candidates, -1).clone()
        s = s_t.expand(candidates, -1).clone()
        
        for t in range(horizon):
            a_t = actions[:, t]
            h = rssm.deterministic_state_fwd(h, s, a_t)
            s = rssm.state_prior(h, sample=True)
            rwds += reward_model(h, s)
        
        # Select top-k
        _, k = torch.topk(rwds, top_k, dim=0, largest=True, sorted=False)
        elite_actions = actions[k]
        
        # Update distribution (no alpha blending)
        mu = elite_actions.mean(dim=0)
        stddev = elite_actions.std(dim=0, unbiased=False)
    
    # Get first action
    action = mu[0].clone()
    
    # Add exploration noise (like reference: self.a += torch.randn_like(self.a)*0.3)
    if explore:
        action = action + torch.randn_like(action) * explore_noise
    
    # Clamp to action bounds
    action = torch.clamp(action, act_low_t, act_high_t)
    
    return action.cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate_planner_rssm(
    env_id, img_size,
    encoder, rssm, reward_model,
    plan_kwargs,
    episodes=10, seed=0, device="cpu",
    bit_depth=5,
    action_repeat=1,
):
    """Evaluate the planner on the environment."""
    env = make_env(env_id, img_size=(img_size, img_size), num_stack=1)
    try:
        env.reset(seed=seed)
    except TypeError:
        pass

    encoder.eval(); rssm.eval(); reward_model.eval()

    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        
        # Initialize state
        obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=bit_depth)
        enc = encoder(obs_t)
        h_state, s_state = rssm.get_init_state(enc)

        while not done:
            # No exploration noise during evaluation
            action = cem_plan_action_rssm(
                rssm=rssm,
                reward_model=reward_model,
                h_t=h_state,
                s_t=s_state,
                device=device,
                explore=False,
                **plan_kwargs
            )
            
            # Action repeat
            total_reward = 0.0
            for _ in range(action_repeat):
                obs, r, term, trunc, _ = env.step(action)
                total_reward += float(r)
                if term or trunc:
                    break
            done = bool(term or trunc)
            ep_ret += total_reward
            
            # Update state with observation
            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=bit_depth)
            enc = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(enc, act_t, h_state, s_state, sample=False)

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ===============================
#  Args
# ===============================

def build_parser():
    p = argparse.ArgumentParser()
    # Environment - dm_control standard benchmarks: cheetah-run, reacher-easy, ball_in_cup-catch, 
    # finger-spin, cartpole-swingup, walker-walk
    p.add_argument("--env_id", type=str, default="cheetah-run", 
                   help="dm_control: cheetah-run, reacher-easy, etc. or gymnasium: Pendulum-v1")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)

    p.add_argument("--seed", type=int, default=0)

    # Reference PlaNet training settings
    p.add_argument("--batch_size", type=int, default=50, help="Reference: 50")
    p.add_argument("--seq_len", type=int, default=50, help="Reference: 50 (chunk length)")

    p.add_argument("--max_episodes", type=int, default=1000, help="Reference ~1000 episodes")
    p.add_argument("--seed_episodes", type=int, default=5, help="Reference: 5 random episodes to seed buffer")
    p.add_argument("--collect_interval", type=int, default=100, help="Reference: train every 100 steps")
    p.add_argument("--train_steps", type=int, default=100, help="Reference: 100 gradient steps per collect")
    p.add_argument("--action_repeat", type=int, default=0, help="0 = use env default from ENV_ACTION_REPEAT")

    p.add_argument("--replay_buff_capacity", type=int, default=1_000_000, help="Reference: 1M transitions")

    p.add_argument("--lr", type=float, default=1e-3, help="Reference: 1e-3")
    p.add_argument("--adam_eps", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=1000.0)

    p.add_argument("--gamma", type=float, default=0.99)

    # Model architecture (reference PlaNet)
    p.add_argument("--embed_dim", type=int, default=1024, help="Reference: 1024")
    p.add_argument("--stoch_dim", type=int, default=30, help="Reference: 30")
    p.add_argument("--deter_dim", type=int, default=200, help="Reference: 200")
    p.add_argument("--hidden_dim", type=int, default=200, help="Reference: 200")
    p.add_argument("--kl_weight", type=float, default=1.0, help="Reference (beta): 1.0")
    p.add_argument("--kl_free_nats", type=float, default=3.0, help="Reference: 3.0")

    # Optional regularizers
    p.add_argument("--pb_curvature_weight", type=float, default=0.0, help="Jacobian/pullback regularizer weight")
    p.add_argument("--pb_curvature_projections", type=int, default=2)
    p.add_argument("--pb_detach_features", action="store_true")

    p.add_argument("--pullback_bisim", action="store_true")
    p.add_argument("--geo_knn_k", type=int, default=8, help="kNN degree for geodesic computation")
    p.add_argument("--geo_time_subsample", type=int, default=16, help="Number of time steps to subsample for geodesic computation")

    p.add_argument("--bisimulation_weight", type=float, default=0.0, help="Bisimulation loss weight")
    p.add_argument("--bisimulation_warmup", type=int, default=0, help="Bisimulation warmup steps (0=no warmup)")
    p.add_argument("--bisimulation_ramp", type=int, default=0, help="Bisimulation ramp steps (0=no ramp)")
    
    # Loss balancing (PlaNet-style)
    p.add_argument("--reward_scale", type=float, default=1.0, help="Scale factor for reward loss")

    # CEM planner (reference defaults)
    p.add_argument("--plan_horizon", type=int, default=12, help="Reference: 12")
    p.add_argument("--plan_candidates", type=int, default=1000, help="Reference: 1000")
    p.add_argument("--plan_iters", type=int, default=10, help="Reference: 10")
    p.add_argument("--plan_top_k", type=int, default=100, help="Reference: 100")
    p.add_argument("--explore_noise", type=float, default=0.3, help="Reference: 0.3")

    # Evaluation
    p.add_argument("--eval_episodes", type=int, default=10, help="Episodes per evaluation")
    p.add_argument("--eval_interval", type=int, default=10, help="Evaluate every N episodes")

    # TensorBoard logging
    p.add_argument("--run_name", type=str, default="geojacobreg", help="Name for TensorBoard run")
    p.add_argument("--log_dir", type=str, default="runs", help="Directory for TensorBoard logs")
    
    # Checkpointing
    p.add_argument("--save_interval", type=int, default=0, help="Save checkpoint every N episodes (0=only final)")
    p.add_argument("--load_checkpoint", type=str, default="", help="Path to checkpoint to resume from")

    return p


# ===============================
#  Main Training Loop
# ===============================

def make_run_name(args):
    """Generate a descriptive run name based on experimental configuration."""
    parts = [args.env_id]
    
    # Bisimulation on/off
    if args.bisimulation_weight > 0:
        parts.append(f"bisim{args.bisimulation_weight}")
    else:
        parts.append("no_bisim")
    
    # Jacobian/pullback curvature regularization on/off
    if args.pb_curvature_weight > 0:
        parts.append(f"jacobreg{args.pb_curvature_weight}")
    else:
        parts.append("no_jacobreg")

    if args.pullback_bisim:
        parts.append("pullback_bisim_knn_h_s_grad")

    parts.append(f"phi_align")
    
    # Add seed for reproducibility tracking
    parts.append(f"seed{args.seed}")
    
    return "_".join(parts)

def pullback_distance_s(decoder, phi_target, h, s1, s2, create_graph=True):
    """
    Computes || J_s g(h, s1) · (s1 - s2) ||_2  (pullback metric at (h, s1))
    h:  [N, Dh] treated as constant
    s1: [N, Ds] point where Jacobian is evaluated
    s2: [N, Ds] reference point (can be detached)
    """
    delta = (s1 - s2)  # [N, Ds]

    def decoder_wrapper(s):
        x = decoder(h, s)
        return phi_target(x)

    # JVP wrt s only, holding h fixed
    _, Jdelta = torch.autograd.functional.jvp(
        decoder_wrapper,
        (s1,),
        (delta,),
        create_graph=create_graph,
    )
    
    return torch.norm(Jdelta, dim=1) / math.sqrt(Jdelta.size(1))

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
    # return torch.norm(Jdelta, dim=1)
    return torch.sqrt((Jdelta * Jdelta).mean(dim=1) + 1e-8)

# softmin(x, x) = -tau*log(2) < 0
def softmin(x, trough, tau=0.1):
    x = torch.clamp(x, max=1e6)
    trough = torch.clamp(trough, max=1e6)
    return -tau * torch.logsumexp(torch.stack([-x/tau, -trough/tau], dim=0), dim=0)

def floyd_warshall_minplus(W):
    """
    W: [B, B] directed edge weights with large INF for missing edges and 0 on diag.
    Returns all-pairs shortest path distances D: [B, B].
    """
    D = W
    B = D.size(0)
    for k in range(B):
        trough = D[:, k:k+1] + D[k:k+1, :]
        D = torch.minimum(D, trough)
    return D

def geodesic_pb_knn_slice(decoder, phi_target, h_t, z_t, targets, k=3, create_graph=True, inf=1e9):
    """
    h_t: [B, Dh]
    z_t: [B, Ds]
    targets: [B] indices in [0..B-1] (your perm)
    k: number of neighbors per node in the kNN graph
    create_graph: True for dz, False for dnext
    Returns: dz_t: [B] geodesic approx from i -> targets[i]
    """
    B = z_t.size(0)
    device = z_t.device
    dtype = z_t.dtype

    # --- Build kNN graph by cheap Euclidean distance in latent (locality) ---
    with torch.no_grad():
        # squared distances [B,B]
        dist2 = torch.cdist(z_t, z_t, p=2).pow(2)
        dist2.fill_diagonal_(float("inf"))
        knn = dist2.topk(k, largest=False).indices  # [B,k]

    # edges: i -> knn[i, j]
    src = torch.arange(B, device=device).repeat_interleave(k)       # [B*k]
    dst = knn.reshape(-1)                                          # [B*k]

    # --- Edge weights: pullback length at source (asymmetric: detach destination) ---
    w = pullback_distance_s(
        decoder,
        phi_target=phi_target,
        h=h_t[src],
        s1=z_t[src],
        s2=z_t[dst].detach(),
        create_graph=create_graph
    )  # [B*k]

    # w = pullback_distance_full(
    #     decoder,
    #     h1=h_t[src],
    #     s1=z_t[src],
    #     h2=h_t[dst].detach(),
    #     s2=z_t[dst].detach(),
    #     create_graph=create_graph
    # )

    # --- Assemble adjacency matrix W ---
    W = torch.full((B, B), inf, device=device, dtype=dtype)
    W.fill_diagonal_(0.0)
    W[src, dst] = w

    # Make graph undirected by symmetrizing (take minimum of both directions)
    W = torch.minimum(W, W.T)

    # --- All-pairs shortest paths ---
    D = floyd_warshall_minplus(W)  # [B,B]

    # distance i -> targets[i]
    dz_t = D[torch.arange(B, device=device), targets]
    return dz_t

def geodesic_pb_knn(decoder, phi_target, h, z, perm, k=10, time_subsample=None, t_idx=None, create_graph=True, return_mask=False):
    """
    h: [B, T, Dh]  (here you’ll pass h = h_seq[:, :-1])
    z: [B, T, Ds]  (here z = s_seq[:, :-1])
    perm: [B] permutation used for pairing within each time slice
    k: kNN degree
    time_subsample: None or int number of time indices to sample
    t_idx: optional pre-chosen time indices (overrides sampling)
    return_mask: return boolean mask over sampled time steps
    create_graph: True for dz, False for dnext
    Returns: dz_flat: [B*T], (optional) mask_flat: [B*T] bool, (optional) t_idx used
    """
    B, Tm1, _ = z.shape
    device = z.device

    if t_idx is None:
        if time_subsample is None or time_subsample >= Tm1:
            t_idx = torch.arange(Tm1, device=device)
        else:
            t_idx = torch.randint(0, Tm1, (time_subsample,), device=device)

    # Compute per-slice and scatter back into [B,Tm1]
    dz = torch.zeros((B, Tm1), device=device, dtype=z.dtype)
    mask = torch.zeros((Tm1,), device=device, dtype=torch.bool)
    for t in t_idx.tolist():
        dz[:, t] = geodesic_pb_knn_slice(
            decoder=decoder,
            phi_target=phi_target,
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

def main(args):
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    # Generate run name if not explicitly provided (i.e., using default)
    if args.run_name == "geojacobreg":
        run_name = make_run_name(args)
    else:
        run_name = args.run_name
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"{args.log_dir}/{run_name}")
    print(f"TensorBoard run name: {run_name}")
    
    # Log hyperparameters
    hparams = vars(args)
    writer.add_text("hyperparameters", str(hparams), 0)

    # Create environment (supports both dm_control and gymnasium)
    env = make_env(args.env_id, img_size=(args.img_size, args.img_size), num_stack=1)
    obs, _ = env.reset()

    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    # Get action repeat (use env default if not specified)
    action_repeat = args.action_repeat if args.action_repeat > 0 else ENV_ACTION_REPEAT.get(args.env_id, 2)
    print(f"Environment: {args.env_id}, Action repeat: {action_repeat}")

    # Models matching reference architecture
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

    phi_net = PhiNet(in_channels=C, hidden_channels=args.hidden_dim, out_dim=args.embed_dim).to(device)
    phi_net_target = copy.deepcopy(phi_net)
    phi_net_target.eval()
    for param in phi_net_target.parameters():
        param.requires_grad = False
    ema_update(phi_net_target, phi_net, 0.999)

    # Single optimizer for all parameters
    params = (
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(rssm.parameters()) + 
        list(reward_model.parameters())
    )
    optim = torch.optim.Adam(params, lr=args.lr, eps=args.adam_eps)
    phi_net_params = list(phi_net.parameters())
    phi_optim = torch.optim.Adam(phi_net_params, lr=args.lr, eps=args.adam_eps)
    phi_tau = 0.995
    feat_align_weight = 0.1
    
    # Store parameter groups for gradient norm computation
    encoder_params = list(encoder.parameters())
    decoder_params = list(decoder.parameters())
    rssm_params = list(rssm.parameters())
    reward_params = list(reward_model.parameters())
    # phi_net_params = list(phi_net.parameters())

    replay = ReplayBuffer(args.replay_buff_capacity, obs_shape=(H, W, C), act_dim=act_dim)

    # Free nats tensor
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    total_steps = 0
    start_episode = 0
    
    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        encoder.load_state_dict(checkpoint['encoder'])
        decoder.load_state_dict(checkpoint['decoder'])
        rssm.load_state_dict(checkpoint['rssm'])
        reward_model.load_state_dict(checkpoint['reward_model'])
        total_steps = checkpoint.get('total_steps', 0)
        start_episode = checkpoint.get('episode', 0)
        print(f"Resumed from episode {start_episode}, total_steps {total_steps}")
    
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
            
            # Action repeat
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
        
        print(f"  Seed episode {seed_ep + 1}/{args.seed_episodes}: return = {ep_return:.2f}, buffer size = {replay.size}")
    
    # ========================================
    # Phase 2: Main training loop
    # ========================================
    if start_episode > 0:
        print(f"\nResuming training from episode {start_episode} to {args.max_episodes}...")
    else:
        print(f"\nStarting training for {args.max_episodes} episodes...")
    
    for episode in range(start_episode, args.max_episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0
        
        # Initialize belief state for this episode
        with torch.no_grad():
            obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=args.bit_depth)
            enc = encoder(obs_t)
            h_state, s_state = rssm.get_init_state(enc)
        
        while not done:
            # -------- Planning with CEM --------
            encoder.eval(); rssm.eval(); reward_model.eval()
            action = cem_plan_action_rssm(
                rssm=rssm,
                reward_model=reward_model,
                h_t=h_state,
                s_t=s_state,
                act_low=act_low,
                act_high=act_high,
                horizon=args.plan_horizon,
                candidates=args.plan_candidates,
                iters=args.plan_iters,
                top_k=args.plan_top_k,
                device=device,
                explore=True,  # Exploration noise during training
                explore_noise=args.explore_noise,
            )
            action = np.asarray(action, dtype=np.float32).reshape(act_dim,)

            # -------- Environment step with action repeat --------
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
                obs_t = torch.tensor(np.ascontiguousarray(obs), dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                preprocess_img(obs_t, depth=args.bit_depth)
                enc = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(enc, act_t, h_state, s_state, sample=False)

            # -------- Train every collect_interval steps --------
            if total_steps % args.collect_interval == 0 and replay.size > (args.seq_len + 2):
                encoder.train(); decoder.train(); rssm.train(); reward_model.train()
                
                loss_accum = {"total": 0.0, "rec": 0.0, "kl": 0.0, "rew": 0.0, "pb": 0.0, "bisim": 0.0}
                bisim_stats_accum = {
                    "dz_mean": 0.0, "dz_std": 0.0,
                    "target_mean": 0.0, "target_std": 0.0,
                    "abs_err_mean": 0.0, "rel_err": 0.0, "corr": 0.0
                }
                grad_norm_accum = {
                    "total": 0.0, "encoder": 0.0, "decoder": 0.0, 
                    "rssm": 0.0, "reward": 0.0, "bisim_weighted": 0.0
                }
                n_bisim_computations = 0

                for upd in range(args.train_steps):
                    # Sample sequences
                    batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)

                    # Convert to tensors - [B, T+1, H, W, C]
                    obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                    act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)
                    rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)

                    B, T1 = rew_seq.shape
                    T = T1 - 1

                    # Convert to [B, T+1, C, H, W] and preprocess
                    x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                    preprocess_img(x, depth=args.bit_depth)

                    # Encode all observations
                    e_t = bottle(encoder, x)

                    # Initialize state from first observation
                    h_t_train, s_t_train = rssm.get_init_state(e_t[:, 0])

                    # Process sequence
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

                    # Build distributions for KL
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
                    # rec_loss = F.mse_loss(recon, target)

                    # Compute feature alignment loss WITHOUT phi gradients
                    # Clone phi_net temporarily to get features without gradient tracking
                    recon_flat = recon.reshape(-1, C, H, W)
                    target_flat = target.reshape(-1, C, H, W)
                    
                    with torch.no_grad():
                        feat_target = phi_net_target(target_flat)
                    
                    # Use phi_net_target for both (it has no gradients anyway)
                    # This guides decoder but doesn't train phi
                    feat_recon = phi_net_target(recon_flat) 
                    feat_loss = torch.zeros((), device=device)  # Disabled for now to avoid conflicts

                    kld_loss = torch.max(
                        kl_divergence(posterior_dist, prior_dist).sum(-1),
                        free_nats
                    ).mean()

                    # KL loss with free nats (mean over latent dim, then over batch/time)
                    # kl_per_latent = kl_divergence(posterior_dist, prior_dist)  # [T, B, stoch_dim]
                    # kl_per_sample = kl_per_latent.mean(-1)  # mean over stoch_dim -> [T, B]
                    # free_nats_per_dim = args.kl_free_nats / args.stoch_dim  # ~0.1 per dim
                    # kld_loss = torch.max(kl_per_sample, torch.tensor(free_nats_per_dim, device=device)).mean()

                    # Reward loss (scaled to balance with reconstruction)
                    rew_pred = bottle(reward_model, h_seq, s_seq)
                    rew_target = rew_seq[:, :T]
                    rew_loss = F.mse_loss(rew_pred, rew_target) * args.reward_scale

                    pb_loss = torch.zeros((), device=device)
                    if args.pb_curvature_weight > 0.0:
                        # TODO: test later change geometry of manifold
                        pass

                    # Bisimulation weight schedule (optional warmup/ramp)
                    def bisim_lambda(total_steps, warmup=0, ramp=0, target=1.0):
                        if warmup > 0 and total_steps < warmup:
                            return 0.0
                        if ramp > 0:
                            t = min(1.0, (total_steps - warmup) / float(ramp))
                            return target * t
                        return target  # No ramp: use full weight immediately

                    bisim_weight = bisim_lambda(total_steps,
                                               warmup=args.bisimulation_warmup,
                                               ramp=args.bisimulation_ramp,
                                               target=args.bisimulation_weight)


                    # Always compute bisimulation loss for monitoring (even if weight=0)
                    bisim_loss_val = torch.zeros((), device=device)
                    if T > 2:
                        # use only stochastic state
                        h = h_seq[:, :-1]        # [B, T-1, Dh]
                        hn = h_seq[:, 1:]        # [B, T-1, Dh]
                        z = s_seq[:, :-1]        # [B, T-1, Ds]
                        zn = s_seq[:, 1:]        # [B, T-1, Ds]
                        r  = rew_seq[:, :T-1]    # [B, T-1]

                        # layernorm on last dim if not args.pullback_bisim
                        # then we would change geometry of decoder
                        if not args.pullback_bisim:
                            z  = F.layer_norm(z,  (z.size(-1),))
                            zn = F.layer_norm(zn, (zn.size(-1),))

                        perm = torch.randperm(B, device=device)

                        h1  = h.reshape(-1, h.size(-1))
                        h2  = hn.reshape(-1, hn.size(-1))
                        z1  = z.reshape(-1, z.size(-1))
                        z2  = z[perm].reshape(-1, z.size(-1))
                        n1  = zn.reshape(-1, zn.size(-1))
                        n2  = zn[perm].reshape(-1, zn.size(-1))
                        r1  = r.reshape(-1)
                        r2  = r[perm].reshape(-1)

                        # asymmetric dz (only create_graph when actually optimizing)
                        needs_grad = bisim_weight > 0.0
                        if args.pullback_bisim:
                            dz, dz_mask, t_idx = geodesic_pb_knn(
                                decoder=decoder,
                                phi_target=phi_net_target,
                                h=h,
                                z=z,
                                perm=perm,
                                k=args.geo_knn_k,
                                time_subsample=args.geo_time_subsample,
                                create_graph=needs_grad,
                                return_mask=True
                            )
                        else:
                            if needs_grad:
                                dz = torch.norm(z1 - z2.detach(), p=2, dim=1)
                            else:
                                with torch.no_grad():
                                    dz = torch.norm(z1 - z2, p=2, dim=1)

                        with torch.no_grad():
                            if args.pullback_bisim:
                                dnext, _, _ = geodesic_pb_knn(
                                    decoder=decoder,
                                    phi_target=phi_net_target,
                                    h=hn,
                                    z=zn,
                                    perm=perm,
                                    k=args.geo_knn_k,
                                    time_subsample=args.geo_time_subsample,
                                    t_idx=t_idx,  # reuse the same time sampling for target
                                    create_graph=False,
                                    return_mask=True
                                )
                            else:
                                dnext = torch.norm(n1 - n2, p=2, dim=1)
                            dr = (r1 - r2).abs()
                            target = dr + args.gamma * dnext

                        if args.pullback_bisim:
                            # Mask to only sampled time steps; keep scale comparable
                            mask = dz_mask.bool()
                            target_masked = target[mask]
                            dz_masked = dz[mask]
                            scale = target_masked.detach().abs().mean().clamp_min(1e-6)
                            bisim_loss_val = F.mse_loss(dz_masked, target_masked) / scale
                        else:
                            scale = target.detach().abs().mean().clamp_min(1e-6)
                            bisim_loss_val = F.mse_loss(dz, target) / scale
                        
                        # Compute statistics for pullback bisim logging
                        if args.pullback_bisim:
                            with torch.no_grad():
                                mask = dz_mask.bool()
                                dz_detached = dz.detach()[mask]
                                target_detached = target.detach()[mask]
                                
                                dz_mean = dz_detached.mean().item()
                                dz_std = dz_detached.std().item()
                                target_mean = target_detached.mean().item()
                                target_std = target_detached.std().item()
                                
                                abs_err = (dz_detached - target_detached).abs()
                                abs_err_mean = abs_err.mean().item()
                                
                                rel_err = abs_err_mean / (target_mean + 1e-6)
                                
                                # Pearson correlation
                                dz_centered = dz_detached - dz_detached.mean()
                                target_centered = target_detached - target_detached.mean()
                                numerator = (dz_centered * target_centered).mean()
                                dz_var = dz_centered.pow(2).mean()
                                target_var = target_centered.pow(2).mean()
                                corr = (numerator / (dz_var.sqrt() * target_var.sqrt() + 1e-8)).item()
                                
                                bisim_stats_accum["dz_mean"] += dz_mean
                                bisim_stats_accum["dz_std"] += dz_std
                                bisim_stats_accum["target_mean"] += target_mean
                                bisim_stats_accum["target_std"] += target_std
                                bisim_stats_accum["abs_err_mean"] += abs_err_mean
                                bisim_stats_accum["rel_err"] += rel_err
                                bisim_stats_accum["corr"] += corr
                                n_bisim_computations += 1

                    # Total loss (including bisimulation when weight > 0)
                    total = (
                        rec_loss 
                        + args.kl_weight * kld_loss 
                        + rew_loss
                        + bisim_weight * bisim_loss_val
                        + feat_align_weight * feat_loss
                    )

                    total_loss_value = total.item()

                    optim.zero_grad()
                    total.backward()
                    
                    # Compute bisim gradient norm for logging (approximate)
                    bisim_grad_norm = bisim_weight * bisim_loss_val.item() if bisim_weight > 0 else 0.0
                    
                    # Compute gradient norms for each component (before clipping)
                    with torch.no_grad():
                        # now params.grad is from total loss (total + bisim)
                        total_grad_norm = torch.nn.utils.clip_grad_norm_(params, float('inf'), norm_type=2)
                        encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder_params, float('inf'), norm_type=2) if encoder_params else torch.tensor(0.0)
                        decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder_params, float('inf'), norm_type=2) if decoder_params else torch.tensor(0.0)
                        rssm_grad_norm = torch.nn.utils.clip_grad_norm_(rssm_params, float('inf'), norm_type=2) if rssm_params else torch.tensor(0.0)
                        reward_grad_norm = torch.nn.utils.clip_grad_norm_(reward_params, float('inf'), norm_type=2) if reward_params else torch.tensor(0.0)
                        
                        grad_norm_accum["total"] += total_grad_norm.item()
                        grad_norm_accum["encoder"] += encoder_grad_norm.item()
                        grad_norm_accum["decoder"] += decoder_grad_norm.item()
                        grad_norm_accum["rssm"] += rssm_grad_norm.item()
                        grad_norm_accum["reward"] += reward_grad_norm.item()
                        grad_norm_accum["bisim_weighted"] += bisim_grad_norm if isinstance(bisim_grad_norm, (int, float)) else bisim_grad_norm.item()
                    
                    if args.grad_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm, norm_type=2)

                    optim.step()

                    # Train phi network separately (after main optimizer to avoid inplace conflicts)
                    # Reuse target_flat from earlier (already reshaped)
                    real = target_flat.detach()
                    phi_net.train()
                    x1 = phi_augment(real)
                    x2 = phi_augment(real)
                    z1 = phi_net(x1)
                    z2 = phi_net(x2)
                    phi_loss = vicreg_loss(z1, z2)

                    phi_optim.zero_grad()
                    phi_loss.backward()
                    phi_optim.step()
                    ema_update(phi_net_target, phi_net, phi_tau)
                    phi_net_target.eval()

                    with torch.no_grad():
                        writer.add_scalar("phi/loss", phi_loss.item(), total_steps)
                        writer.add_scalar("phi/z1_mean", z1.mean().item(), total_steps)
                        writer.add_scalar("phi/z1_std", z1.std().item(), total_steps)
                        writer.add_scalar("phi/z2_mean", z2.mean().item(), total_steps)
                        writer.add_scalar("phi/z2_std", z2.std().item(), total_steps)
                        writer.add_scalar("phi/z1_z2_corr", (z1 * z2).mean().item(), total_steps)
                        writer.add_scalar("phi/z1_z2_cov", (z1 * z2).var().item(), total_steps)

                    # Accumulate losses for logging
                    loss_accum["total"] += total_loss_value
                    loss_accum["rec"] += rec_loss.item()
                    loss_accum["kl"] += kld_loss.item()
                    loss_accum["rew"] += rew_loss.item()
                    loss_accum["pb"] += pb_loss.item()
                    loss_accum["bisim"] += bisim_loss_val.item()
                    loss_accum["bisim_weighted"] = loss_accum.get("bisim_weighted", 0.0) + bisim_weight * bisim_loss_val.item()

                # Log average losses to TensorBoard
                n_updates = args.train_steps
                writer.add_scalar("loss/total", loss_accum["total"] / n_updates, total_steps)
                writer.add_scalar("loss/reconstruction", loss_accum["rec"] / n_updates, total_steps)
                writer.add_scalar("loss/kl_divergence", loss_accum["kl"] / n_updates, total_steps)
                writer.add_scalar("loss/reward", loss_accum["rew"] / n_updates, total_steps)
                writer.add_scalar("loss/pullback_curvature", loss_accum["pb"] / n_updates, total_steps)
                writer.add_scalar("loss/bisimulation", loss_accum["bisim"] / n_updates, total_steps)
                writer.add_scalar("loss/bisimulation_weighted", loss_accum.get("bisim_weighted", 0.0) / n_updates, total_steps)
                
                # Log effective bisim weight (useful to see warmup/ramp progress)
                writer.add_scalar("loss/bisim_weight", bisim_weight, total_steps)
                
                # Log pullback bisim statistics if computed
                if args.pullback_bisim and n_bisim_computations > 0:
                    n_bisim_updates = n_bisim_computations
                    writer.add_scalar("bisim_stats/dz_mean", bisim_stats_accum["dz_mean"] / n_bisim_updates, total_steps)
                    writer.add_scalar("bisim_stats/dz_std", bisim_stats_accum["dz_std"] / n_bisim_updates, total_steps)
                    writer.add_scalar("bisim_stats/target_mean", bisim_stats_accum["target_mean"] / n_bisim_updates, total_steps)
                    writer.add_scalar("bisim_stats/target_std", bisim_stats_accum["target_std"] / n_bisim_updates, total_steps)
                    writer.add_scalar("bisim_stats/abs_err_mean", bisim_stats_accum["abs_err_mean"] / n_bisim_updates, total_steps)
                    writer.add_scalar("bisim_stats/rel_err", bisim_stats_accum["rel_err"] / n_bisim_updates, total_steps)
                    writer.add_scalar("bisim_stats/corr", bisim_stats_accum["corr"] / n_bisim_updates, total_steps)
                
                # Log gradient norms
                n_updates = args.train_steps
                writer.add_scalar("grad_norm/total", grad_norm_accum["total"] / n_updates, total_steps)
                writer.add_scalar("grad_norm/encoder", grad_norm_accum["encoder"] / n_updates, total_steps)
                writer.add_scalar("grad_norm/decoder", grad_norm_accum["decoder"] / n_updates, total_steps)
                writer.add_scalar("grad_norm/rssm", grad_norm_accum["rssm"] / n_updates, total_steps)
                writer.add_scalar("grad_norm/reward", grad_norm_accum["reward"] / n_updates, total_steps)
                writer.add_scalar("grad_norm/bisim_weighted", grad_norm_accum["bisim_weighted"] / n_updates, total_steps)
                
                # Log bisim gradient contribution ratio
                if grad_norm_accum["total"] > 0:
                    bisim_ratio = grad_norm_accum["bisim_weighted"] / grad_norm_accum["total"]
                    writer.add_scalar("grad_norm/bisim_ratio", bisim_ratio, total_steps)
                
                # Log bisim gradient contribution to decoder (since bisim affects decoder most)
                if grad_norm_accum["decoder"] > 0:
                    decoder_bisim_ratio = grad_norm_accum["bisim_weighted"] / grad_norm_accum["decoder"]
                    writer.add_scalar("grad_norm/bisim_to_decoder_ratio", decoder_bisim_ratio, total_steps)
        
        # -------- End of episode --------
        writer.add_scalar("train/episode_return", ep_return, episode)
        writer.add_scalar("train/episode_steps", ep_steps, episode)
        writer.add_scalar("train/total_steps", total_steps, episode)
        
        print(f"Episode {episode + 1}/{args.max_episodes}: return = {ep_return:.2f}, steps = {ep_steps}, total_steps = {total_steps}")
        
        # -------- Evaluation --------
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
                encoder=encoder,
                rssm=rssm,
                reward_model=reward_model,
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
        
        # Periodic checkpoint saving
        if args.save_interval > 0 and (episode + 1) % args.save_interval == 0:
            checkpoint_dir = f"{args.log_dir}/{run_name}"
            checkpoint_path = f"{checkpoint_dir}/model_ep{episode + 1}.pt"
            print(f"  Saving checkpoint to: {checkpoint_path}")
            torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'rssm': rssm.state_dict(),
                'reward_model': reward_model.state_dict(),
                'args': vars(args),
                'total_steps': total_steps,
                'episode': episode + 1,
            }, checkpoint_path)

    env.close()
    writer.close()
    
    # Save final model checkpoint
    checkpoint_dir = f"{args.log_dir}/{run_name}"
    checkpoint_path = f"{checkpoint_dir}/model_final.pt"
    print(f"\nSaving final model checkpoint to: {checkpoint_path}")
    
    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'rssm': rssm.state_dict(),
        'reward_model': reward_model.state_dict(),
        'args': vars(args),
        'total_steps': total_steps,
        'episode': args.max_episodes,
    }, checkpoint_path)
    
    print(f"Training complete! TensorBoard logs and checkpoint saved to: {args.log_dir}/{run_name}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
