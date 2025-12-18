import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.utils.tensorboard import SummaryWriter

from utils import PixelObsWrapper, ReplayBuffer, get_device, set_seed, preprocess_img, bottle

from models import ConvEncoder, ConvDecoder, RSSM, RewardModel

# ===============================
#  Losses / Regularizers
# ===============================

def compute_pullback_curvature_loss(decoder, h, s, num_projections=4, detach_features=True):
    """
    Hutchinson estimate of || J^T J - I ||_F^2 for decoder Jacobian wrt latent.
    Expensive: keep projections small.
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
    dz = torch.norm(z1 - z2, p=2, dim=1)              # [B]
    dnext = torch.norm(next_z1 - next_z2, p=2, dim=1) # [B]
    dr = torch.abs(r1 - r2).view(-1)                  # [B]
    target = dr + gamma * dnext
    return F.mse_loss(dz, target)


# ===============================
#  CEM Planner (reference implementation)
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
        
        # Update distribution (no alpha blending like reference)
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
):
    env = PixelObsWrapper(env_id, img_size=(img_size, img_size), num_stack=1)
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
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
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
                explore=False,  # No exploration during eval
                **plan_kwargs
            )
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += float(r)
            
            # Update state with observation
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                preprocess_img(obs_t, depth=bit_depth)
                enc = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(enc, act_t, h_state, s_state, sample=False)
            
            done = bool(term or trunc)

        returns.append(ep_ret)

    env.close()
    return float(np.mean(returns)), float(np.std(returns))


# ===============================
#  Args
# ===============================

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--env_id", type=str, default="Pendulum-v1")
    p.add_argument("--img_size", type=int, default=64)
    p.add_argument("--bit_depth", type=int, default=5)

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=50)

    p.add_argument("--max_env_steps", type=int, default=100_000)
    p.add_argument("--start_steps", type=int, default=5_000)

    p.add_argument("--update_every", type=int, default=1_000)
    p.add_argument("--updates_per_step", type=int, default=150)
    p.add_argument("--action_repeat", type=int, default=2)

    p.add_argument("--replay_buff_capacity", type=int, default=200_000)

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

    # optional regs
    p.add_argument("--pb_curvature_weight", type=float, default=0.0)
    p.add_argument("--pb_curvature_projections", type=int, default=2)
    p.add_argument("--pb_detach_features", action="store_true")

    p.add_argument("--bisimulation_weight", type=float, default=0.0)

    # planning (reference defaults)
    p.add_argument("--plan_horizon", type=int, default=12)  # shorter horizon for better accuracy
    p.add_argument("--plan_candidates", type=int, default=1000)
    p.add_argument("--plan_iters", type=int, default=10)
    p.add_argument("--plan_top_k", type=int, default=100)
    p.add_argument("--plan_every", type=int, default=1)
    p.add_argument("--explore_noise", type=float, default=0.3)  # exploration noise (reference uses 0.3)

    # TensorBoard logging
    p.add_argument("--run_name", type=str, default="geojacobreg", help="Name for TensorBoard run")
    p.add_argument("--log_dir", type=str, default="runs", help="Directory for TensorBoard logs")

    return p


# ===============================
#  Main Training Loop
# ===============================

def main(args):
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=f"{args.log_dir}/{args.run_name}")
    
    # Log hyperparameters
    hparams = vars(args)
    writer.add_text("hyperparameters", str(hparams), 0)

    # Use num_stack=1 like reference (single frame, not stacked)
    env = PixelObsWrapper(args.env_id, img_size=(args.img_size, args.img_size), num_stack=1)
    obs, _ = env.reset()

    H, W, C = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

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

    # Single optimizer for all parameters (like reference)
    params = (
        list(encoder.parameters()) + 
        list(decoder.parameters()) + 
        list(rssm.parameters()) + 
        list(reward_model.parameters())
    )
    optim = torch.optim.Adam(params, lr=args.lr, eps=args.adam_eps)

    replay = ReplayBuffer(args.replay_buff_capacity, obs_shape=(H, W, C), act_dim=act_dim)

    # Free nats tensor
    free_nats = torch.ones(1, device=device) * args.kl_free_nats

    step = 0
    episode_return = 0.0
    episode_count = 0
    
    # Initialize belief state
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
        preprocess_img(obs_t, depth=args.bit_depth)
        enc = encoder(obs_t)
        h_state, s_state = rssm.get_init_state(enc)
        a_prev = torch.zeros(1, act_dim, device=device)

    while step < args.max_env_steps:
        # -------- act --------
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            if step % args.plan_every == 0:
                # Use explore=True during training to add noise (like reference)
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
                    explore=True,  # Add exploration noise during training
                    explore_noise=args.explore_noise,
                )
            else:
                action = env.action_space.sample()

        action = np.asarray(action, dtype=np.float32).reshape(act_dim,)

        # -------- env step with action repeat --------
        total_reward = 0.0
        terminated = False
        truncated = False
        next_obs = obs
        for _ in range(args.action_repeat):
            next_obs, r_step, terminated, truncated, _ = env.step(action)
            total_reward += float(r_step)
            if terminated or truncated:
                break
        done = bool(terminated or truncated)

        replay.add(
            obs=np.asarray(obs, dtype=np.uint8),
            action=np.asarray(action, dtype=np.float32),
            reward=total_reward,
            next_obs=np.asarray(next_obs, dtype=np.uint8),
            done=done
        )

        episode_return += total_reward
        step += 1
        obs = next_obs

        # Update belief state with new observation
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=args.bit_depth)
            enc = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(enc, act_t, h_state, s_state, sample=False)

        if done:
            episode_count += 1
            # Log episode metrics to TensorBoard
            writer.add_scalar("train/episode_return", episode_return, step)
            writer.add_scalar("train/episode_count", episode_count, step)
            
            obs, _ = env.reset()
            episode_return = 0.0
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)
                preprocess_img(obs_t, depth=args.bit_depth)
                enc = encoder(obs_t)
                h_state, s_state = rssm.get_init_state(enc)
                a_prev = torch.zeros(1, act_dim, device=device)

        # -------- train --------
        if step >= args.start_steps and step % args.update_every == 0 and replay.size > (args.seq_len + 2):
            encoder.train(); decoder.train(); rssm.train(); reward_model.train()
            
            # Accumulators for logging average losses over updates
            loss_accum = {"total": 0.0, "rec": 0.0, "kl": 0.0, "rew": 0.0, "pb": 0.0, "bisim": 0.0}

            for upd in range(args.updates_per_step):
                # Sample sequences: obs[0:T], actions[0:T-1], rewards[0:T-1]
                batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)

                # Convert to tensors - [B, T+1, H, W, C]
                obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device)
                act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)  # [B, T+1, A]
                rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)     # [B, T+1]

                B, T1 = rew_seq.shape
                T = T1 - 1

                # Convert to [B, T+1, C, H, W] and preprocess
                x = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                preprocess_img(x, depth=args.bit_depth)

                # Encode all observations [B, T+1, E]
                e_t = bottle(encoder, x)  # [B, T+1, embed_dim]

                # Initialize state from first observation
                h_t, s_t = rssm.get_init_state(e_t[:, 0])

                # Process sequence (reference training loop)
                states = []
                priors = []
                posteriors = []
                posterior_samples = []

                for t in range(T):
                    # Deterministic state forward
                    h_t = rssm.deterministic_state_fwd(h_t, s_t, act_seq[:, t])
                    states.append(h_t)
                    
                    # Prior and posterior (use next embedding e_t[t+1])
                    priors.append(rssm.state_prior(h_t))
                    posteriors.append(rssm.state_posterior(h_t, e_t[:, t + 1]))
                    
                    # Sample from posterior
                    post_mean, post_std = posteriors[-1]
                    s_t = post_mean + torch.randn_like(post_std) * post_std
                    posterior_samples.append(s_t)

                # Stack results
                h_seq = torch.stack(states, dim=1)  # [B, T, deter_dim]
                s_seq = torch.stack(posterior_samples, dim=1)  # [B, T, stoch_dim]

                # Build distributions for KL
                prior_means = torch.stack([p[0] for p in priors], dim=0)  # [T, B, stoch_dim]
                prior_stds = torch.stack([p[1] for p in priors], dim=0)
                post_means = torch.stack([p[0] for p in posteriors], dim=0)
                post_stds = torch.stack([p[1] for p in posteriors], dim=0)

                prior_dist = Normal(prior_means, prior_stds)
                posterior_dist = Normal(post_means, post_stds)

                # Reconstruction loss (MSE like reference)
                # Decode from (h, s) and compare to x[1:T+1]
                recon = bottle(decoder, h_seq, s_seq)  # [B, T, C, H, W]
                target = x[:, 1:T+1]  # [B, T, C, H, W]
                rec_loss = F.mse_loss(recon, target, reduction='none').sum((2, 3, 4)).mean()

                # KL loss with free nats (like reference)
                kld_loss = torch.max(
                    kl_divergence(posterior_dist, prior_dist).sum(-1),
                    free_nats
                ).mean()

                # Reward loss (simple MSE like reference)
                rew_pred = bottle(reward_model, h_seq, s_seq)  # [B, T]
                rew_target = rew_seq[:, :T]  # rewards for transitions 0 to T-1
                rew_loss = F.mse_loss(rew_pred, rew_target)

                # Optional regularizers
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

                bisim_loss_val = torch.zeros((), device=device)
                if args.bisimulation_weight > 0.0 and T > 1:
                    # Flatten states for bisimulation
                    z = torch.cat([h_seq, s_seq], dim=-1)  # [B, T, deter+stoch]
                    z_tm1 = z[:, :-1].reshape(-1, z.size(-1))
                    z_tp1 = z[:, 1:].reshape(-1, z.size(-1))
                    rew_flat = rew_seq[:, :T-1].reshape(-1)
                    
                    n = z_tm1.size(0)
                    if n >= 2:
                        idx1 = torch.randint(0, n, (n,), device=device)
                        idx2 = torch.randint(0, n, (n,), device=device)
                        bisim_loss_val = bisimulation_loss(
                            z_tm1[idx1], z_tm1[idx2],
                            rew_flat[idx1], rew_flat[idx2],
                            z_tp1[idx1], z_tp1[idx2],
                            gamma=args.gamma
                        )

                # Total loss (like reference: beta*kl + rec + rew)
                total = (
                    args.kl_weight * kld_loss 
                    + rec_loss 
                    + rew_loss
                    + args.pb_curvature_weight * pb_loss
                    + args.bisimulation_weight * bisim_loss_val
                )

                optim.zero_grad()
                total.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm, norm_type=2)
                optim.step()

                # Accumulate losses for logging
                loss_accum["total"] += total.item()
                loss_accum["rec"] += rec_loss.item()
                loss_accum["kl"] += kld_loss.item()
                loss_accum["rew"] += rew_loss.item()
                loss_accum["pb"] += pb_loss.item()
                loss_accum["bisim"] += bisim_loss_val.item()

                if (upd + 1) % 10 == 0:
                    print(
                        f"[step={step}] upd={upd+1}/{args.updates_per_step} "
                        f"loss={total.item():.4f} rec={rec_loss.item():.4f} "
                        f"kl={kld_loss.item():.4f} r={rew_loss.item():.4f} "
                        f"pb={pb_loss.item():.4f} bisim={bisim_loss_val.item():.4f}"
                    )
            
            # Log average losses to TensorBoard
            n_updates = args.updates_per_step
            writer.add_scalar("loss/total", loss_accum["total"] / n_updates, step)
            writer.add_scalar("loss/reconstruction", loss_accum["rec"] / n_updates, step)
            writer.add_scalar("loss/kl_divergence", loss_accum["kl"] / n_updates, step)
            writer.add_scalar("loss/reward", loss_accum["rew"] / n_updates, step)
            writer.add_scalar("loss/pullback_curvature", loss_accum["pb"] / n_updates, step)
            writer.add_scalar("loss/bisimulation", loss_accum["bisim"] / n_updates, step)

        # -------- eval --------
        if step % 5_000 == 0:
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
                episodes=10,
                seed=args.seed + 100,
                device=device,
                bit_depth=args.bit_depth,
            )
            print(f"[step={step}] Eval return: {mean_ret:.2f} ± {std_ret:.2f}")
            
            # Log evaluation metrics to TensorBoard
            writer.add_scalar("eval/mean_return", mean_ret, step)
            writer.add_scalar("eval/std_return", std_ret, step)

    env.close()
    writer.close()
    print(f"TensorBoard logs saved to: {args.log_dir}/{args.run_name}")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
