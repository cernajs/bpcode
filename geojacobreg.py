import argparse
import numpy as np
import torch
import torch.nn.functional as F

from utils import PixelObsWrapper, ReplayBuffer, get_device, set_seed

from models import ConvEncoder, ConvDecoder, RSSM, RewardModel

# ===============================
#  Losses / Regularizers
# ===============================

def compute_pullback_curvature_loss(decoder, features, num_projections=4, detach_features=True):
    """
    Hutchinson estimate of || J^T J - I ||_F^2 for decoder Jacobian wrt latent.
    Expensive: keep projections small.
    """
    x = features.detach().requires_grad_(True) if detach_features else features.requires_grad_(True)

    loss = 0.0
    for _ in range(num_projections):
        u = (torch.randint(0, 2, x.shape, device=x.device) * 2 - 1).to(x.dtype)

        recon, Ju = torch.autograd.functional.jvp(
            lambda z: decoder(z), (x,), (u,), create_graph=True
        )

        Gu = torch.autograd.grad(
            outputs=recon, inputs=x, grad_outputs=Ju, create_graph=True
        )[0]

        Au = Gu - u
        loss = loss + (Au.pow(2).sum(dim=1)).mean()

    return loss / float(num_projections)


def bisimulation_loss(z1, z2, r1, r2, next_z1, next_z2, gamma=0.99):
    dz = torch.norm(z1 - z2, p=2, dim=1)              # [B]
    dnext = torch.norm(next_z1 - next_z2, p=2, dim=1) # [B]
    dr = torch.abs(r1 - r2).view(-1)                  # [B]
    target = dr + gamma * dnext
    return F.mse_loss(dz, target)


# ===============================
#  CEM Planner
# ===============================

def _rssm_prior_stats(rssm, h):
    # expects rssm.prior(h) -> (mean, std)
    if hasattr(rssm, "prior"):
        return rssm.prior(h)

    # fallback: rssm.prior_net(h) -> [*, 2*stoch_dim]
    stats = rssm.prior_net(h)
    mean, log_std = torch.chunk(stats, 2, dim=-1)
    std = F.softplus(log_std) + 1e-4
    return mean, std


@torch.no_grad()
def _rssm_prior_step(rssm, h, s, a, use_mean=True):
    # expects rssm.gru to be GRUCell
    h = rssm.gru(torch.cat([s, a], dim=-1), h)
    m, st = _rssm_prior_stats(rssm, h)
    if use_mean:
        s = m
    else:
        s = m + torch.randn_like(st) * st
    return h, s


@torch.no_grad()
def cem_plan_action_rssm(
    obs, encoder, rssm, reward_model,
    act_low, act_high,
    horizon=12, candidates=512, iters=5,
    elite_frac=0.1, alpha=0.25,
    init_std=1.0, min_std=0.05,
    gamma=0.99,
    use_mean_rollout=True,
    device=None,
    start_h=None,
    start_s=None,
):
    """
    obs: (H,W,Cstack) uint8
    returns: np.ndarray [act_dim]
    """
    device = device or next(encoder.parameters()).device
    act_dim = act_low.shape[0]

    if start_h is None or start_s is None:
        # encode current obs -> embed
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        obs_t = obs_t.permute(0, 3, 1, 2).contiguous()  # [1,C,H,W]
        e0 = encoder(obs_t)  # [1,E]
        h0, s0 = rssm.init_state(e0)
    else:
        h0, s0 = start_h, start_s

    # CEM distribution over action sequences
    mean = torch.zeros(horizon, act_dim, device=device)
    std = torch.ones(horizon, act_dim, device=device) * init_std

    act_low_t = torch.tensor(act_low, device=device, dtype=torch.float32)
    act_high_t = torch.tensor(act_high, device=device, dtype=torch.float32)
    elite_n = max(1, int(candidates * elite_frac))

    for _ in range(iters):
        eps = torch.randn(candidates, horizon, act_dim, device=device)
        actions = mean.unsqueeze(0) + eps * std.unsqueeze(0)
        actions = torch.max(torch.min(actions, act_high_t), act_low_t)

        h = h0.expand(candidates, -1).contiguous()
        s = s0.expand(candidates, -1).contiguous()

        ret = torch.zeros(candidates, device=device)
        disc = 1.0

        for t in range(horizon):
            a_t = actions[:, t, :]
            z = torch.cat([h, s], dim=-1)  # [N, deter+stoch]
            r_t = reward_model(z, a_t).view(-1)
            ret = ret + disc * r_t
            h, s = _rssm_prior_step(rssm, h, s, a_t, use_mean=use_mean_rollout)
            disc *= gamma

        elites = actions[ret.topk(elite_n, largest=True).indices]
        new_mean = elites.mean(dim=0)
        new_std = elites.std(dim=0).clamp(min=min_std)

        mean = alpha * mean + (1 - alpha) * new_mean
        std = alpha * std + (1 - alpha) * new_std

    a0 = torch.max(torch.min(mean[0], act_high_t), act_low_t)
    return a0.cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate_planner_rssm(
    env_id, img_size, num_stack,
    encoder, rssm, reward_model,
    plan_kwargs,
    episodes=10, seed=0, device="cpu",
):
    env = PixelObsWrapper(env_id, img_size=(img_size, img_size), num_stack=num_stack)
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
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            obs_t = obs_t.permute(0, 3, 1, 2).contiguous()
            init_embed = encoder(obs_t)
            h_state, s_state = rssm.init_state(init_embed)

        while not done:
            action = cem_plan_action_rssm(
                obs=obs,
                encoder=encoder,
                rssm=rssm,
                reward_model=reward_model,
                device=device,
                start_h=h_state,
                start_s=s_state,
                **plan_kwargs
            )
            obs, r, term, trunc, _ = env.step(action)
            ep_ret += float(r)
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                obs_t = obs_t.permute(0, 3, 1, 2).contiguous()
                embed_t = encoder(obs_t)
                act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
                h_state, s_state, _, _ = rssm.observe_step(embed_t, act_t, h_state, s_state)
                h_state = h_state.detach()
                s_state = s_state.detach()
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
    p.add_argument("--num_stack", type=int, default=3)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=128)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seq_len", type=int, default=50)

    p.add_argument("--max_env_steps", type=int, default=100_000)
    p.add_argument("--start_steps", type=int, default=5_000)

    p.add_argument("--update_every", type=int, default=1_000)
    p.add_argument("--updates_per_step", type=int, default=200)

    p.add_argument("--replay_buff_capacity", type=int, default=200_000)

    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip_norm", type=float, default=10.0)

    p.add_argument("--gamma", type=float, default=0.99)

    p.add_argument("--embed_dim", type=int, default=1024)
    p.add_argument("--stoch_dim", type=int, default=30)
    p.add_argument("--deter_dim", type=int, default=200)
    p.add_argument("--kl_weight", type=float, default=1.0)

    # optional regs
    p.add_argument("--pb_curvature_weight", type=float, default=0.0)
    p.add_argument("--pb_curvature_projections", type=int, default=2)
    p.add_argument("--pb_detach_features", action="store_true")

    p.add_argument("--bisimulation_weight", type=float, default=0.0)

    # planning
    p.add_argument("--plan_horizon", type=int, default=12)
    p.add_argument("--plan_candidates", type=int, default=512)
    p.add_argument("--plan_iters", type=int, default=5)
    p.add_argument("--plan_elite_frac", type=float, default=0.1)
    p.add_argument("--plan_alpha", type=float, default=0.25)
    p.add_argument("--plan_init_std", type=float, default=1.0)
    p.add_argument("--plan_min_std", type=float, default=0.05)
    p.add_argument("--plan_every", type=int, default=1)

    return p


# ===============================
#  Main
# ===============================

def main(args):
    set_seed(args.seed)
    device = get_device()
    print("Using device:", device)

    env = PixelObsWrapper(args.env_id, img_size=(args.img_size, args.img_size), num_stack=args.num_stack)
    obs, _ = env.reset()

    # obs is (H, W, Cstack)
    H, W, Cstack = env.observation_space.shape
    act_dim = env.action_space.shape[0]
    act_low = env.action_space.low
    act_high = env.action_space.high

    encoder = ConvEncoder((Cstack, H, W), latent_dim=args.embed_dim).to(device)

    state_dim = args.deter_dim + args.stoch_dim
    decoder = ConvDecoder(state_dim, encoder.conv_out_shape, out_channels=Cstack).to(device)
    reward_model = RewardModel(state_dim, act_dim, hidden_dim=args.hidden_dim).to(device)

    rssm = RSSM(
        stoch_dim=args.stoch_dim,
        deter_dim=args.deter_dim,
        act_dim=act_dim,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim
    ).to(device)

    params = list(encoder.parameters()) + list(decoder.parameters()) + list(rssm.parameters()) + list(reward_model.parameters())
    optim = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    replay = ReplayBuffer(args.replay_buff_capacity, obs_shape=(H, W, Cstack), act_dim=act_dim)

    step = 0
    episode_return = 0.0
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
        obs_t = obs_t.permute(0, 3, 1, 2).contiguous()
        init_embed = encoder(obs_t)
        h_state, s_state = rssm.init_state(init_embed)

    while step < args.max_env_steps:
        # -------- act --------
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            if step % args.plan_every == 0:
                action = cem_plan_action_rssm(
                    obs=obs,
                    encoder=encoder,
                    rssm=rssm,
                    reward_model=reward_model,
                    act_low=act_low,
                    act_high=act_high,
                    horizon=args.plan_horizon,
                    candidates=args.plan_candidates,
                    iters=args.plan_iters,
                    elite_frac=args.plan_elite_frac,
                    alpha=args.plan_alpha,
                    init_std=args.plan_init_std,
                    min_std=args.plan_min_std,
                    gamma=args.gamma,
                    use_mean_rollout=True,
                    device=device,
                    start_h=h_state,
                    start_s=s_state,
                )
            else:
                action = env.action_space.sample()

        action = np.asarray(action, dtype=np.float32).reshape(act_dim,)

        # -------- env step --------
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)

        replay.add(
            obs=np.asarray(obs, dtype=np.uint8),
            action=np.asarray(action, dtype=np.float32),
            reward=float(reward),
            next_obs=np.asarray(next_obs, dtype=np.uint8),
            done=done
        )

        episode_return += float(reward)
        step += 1
        obs = next_obs

        # filter update with latest observation/action
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
            obs_t = obs_t.permute(0, 3, 1, 2).contiguous()
            embed_t = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(embed_t, act_t, h_state, s_state)
            h_state = h_state.detach()
            s_state = s_state.detach()

        if done:
            obs, _ = env.reset()
            episode_return = 0.0
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                obs_t = obs_t.permute(0, 3, 1, 2).contiguous()
                init_embed = encoder(obs_t)
                h_state, s_state = rssm.init_state(init_embed)

        # -------- train --------
        if step >= args.start_steps and step % args.update_every == 0 and replay.size > (args.seq_len + 2):
            encoder.train(); decoder.train(); rssm.train(); reward_model.train()

            for upd in range(args.updates_per_step):
                # IMPORTANT: sample seq_len+1 so we can build (t -> t+1) transitions by shifting
                batch = replay.sample_sequences(args.batch_size, args.seq_len + 1)

                obs_seq = torch.tensor(batch.obs, dtype=torch.float32, device=device) / 255.0      # [B,T+1,H,W,C]
                act_seq = torch.tensor(batch.actions, dtype=torch.float32, device=device)          # [B,T+1,A]
                rew_seq = torch.tensor(batch.rews, dtype=torch.float32, device=device)             # [B,T+1]
                done_seq = torch.tensor(batch.dones, dtype=torch.float32, device=device)           # [B,T+1]

                # build mask over T steps (exclude final bootstrap obs)
                B, T1 = done_seq.shape  # T1 = seq_len+1
                T = T1 - 1
                mask_full = torch.ones(B, T1, device=device)
                mask_full[:, 1:] = torch.cumprod(1.0 - done_seq[:, :-1], dim=1)
                mask = mask_full[:, :-1]  # [B,T]

                # obs to [B,T1,C,H,W]
                obs_chw_full = obs_seq.permute(0, 1, 4, 2, 3).contiguous()
                obs_chw = obs_chw_full[:, :-1]  # drop last bootstrap obs -> [B,T,C,H,W]

                # encode embeddings [B,T,E]
                embeds_full = encoder(obs_chw_full)  # ConvEncoder supports [B,T,C,H,W]
                embeds = embeds_full[:, :-1]         # [B,T,E]

                # shift actions to match embeds: use first T actions
                act_in = act_seq[:, :-1]  # [B,T,A]

                rssm_out = rssm.observe(embeds, act_in)  # h,s + prior/post stats
                h = rssm_out["h"]; s = rssm_out["s"]
                state = torch.cat([h, s], dim=-1)  # [B,T,Dh+Ds]

                recon = decoder(state)  # [B,T,C,H,W]
                recon_loss = ((recon - obs_chw) ** 2).mean(dim=(2,3,4))          # [B,T]
                recon_loss = (recon_loss * mask).sum() / mask.sum().clamp_min(1.0)

                # reward loss (use rewards for the same T transitions)
                state_flat = state.reshape(B*T, -1)
                act_flat = act_in.reshape(B*T, -1)
                pred_rew = reward_model(state_flat, act_flat).view(B, T)
                rew_t = rew_seq[:, :-1]  # [B,T]
                rew_loss = ((pred_rew - rew_t) ** 2)
                rew_loss = (rew_loss * mask).sum() / mask.sum().clamp_min(1.0)

                # KL(q||p) per step (diag Gaussians)
                qm, qs = rssm_out["post_mean"], rssm_out["post_std"]
                pm, ps = rssm_out["prior_mean"], rssm_out["prior_std"]

                # KL for diagonal Gaussians
                kl = (
                    torch.log(ps/qs)
                    + (qs**2 + (qm-pm)**2) / (2.0 * ps**2)
                    - 0.5
                ).sum(dim=-1)  # [B,T]
                kl_loss = (kl * mask).sum() / mask.sum().clamp_min(1.0)

                # flatten state/reward for optional regularizers
                z_tm1 = state[:, :-1].reshape(-1, state.size(-1))           # [B*(T-1), Dz]
                z_tp1 = state[:, 1:].reshape(-1, state.size(-1))            # [B*(T-1), Dz]
                rew_flat = rew_t[:, :-1].reshape(-1)                        # [B*(T-1)]
                mask_flat = mask[:, :-1].reshape(-1)                        # [B*(T-1)]

                pb_loss = torch.zeros((), device=device)
                if args.pb_curvature_weight > 0.0:
                    pb_loss = compute_pullback_curvature_loss(
                        decoder=decoder,
                        features=state_flat,
                        num_projections=args.pb_curvature_projections,
                        detach_features=args.pb_detach_features
                    )

                bisim_loss_val = torch.zeros((), device=device)
                if args.bisimulation_weight > 0.0:
                    valid = (mask_flat > 0.5).nonzero(as_tuple=False).view(-1)
                    if valid.numel() >= 2:
                        n = min(valid.numel(), B * (T-1))
                        idx1 = valid[torch.randint(0, valid.numel(), (n,), device=device)]
                        idx2 = valid[torch.randint(0, valid.numel(), (n,), device=device)]
                        bisim_loss_val = bisimulation_loss(
                            z_tm1[idx1], z_tm1[idx2],
                            rew_flat[idx1], rew_flat[idx2],
                            z_tp1[idx1], z_tp1[idx2],
                            gamma=args.gamma
                        )

                total = (
                    recon_loss
                    + rew_loss
                    + args.kl_weight * kl_loss
                    + args.pb_curvature_weight * pb_loss
                    + args.bisimulation_weight * bisim_loss_val
                )

                optim.zero_grad(set_to_none=True)
                total.backward()
                if args.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip_norm)
                optim.step()

                if (upd + 1) % 10 == 0:
                    print(
                        f"[step={step}] upd={upd+1}/{args.updates_per_step} "
                        f"loss={total.item():.4f} recon={recon_loss.item():.4f} "
                        f"kl={kl_loss.item():.4f} r={rew_loss.item():.4f} "
                        f"pb={pb_loss.item():.4f} bisim={bisim_loss_val.item():.4f}"
                    )

        # -------- eval --------
        if step % 5_000 == 0:
            plan_kwargs = dict(
                act_low=act_low,
                act_high=act_high,
                horizon=args.plan_horizon,
                candidates=args.plan_candidates,
                iters=args.plan_iters,
                elite_frac=args.plan_elite_frac,
                alpha=args.plan_alpha,
                init_std=args.plan_init_std,
                min_std=args.plan_min_std,
                gamma=args.gamma,
            )
            mean_ret, std_ret = evaluate_planner_rssm(
                env_id=args.env_id,
                img_size=args.img_size,
                num_stack=args.num_stack,
                encoder=encoder,
                rssm=rssm,
                reward_model=reward_model,
                plan_kwargs=plan_kwargs,
                episodes=10,
                seed=args.seed + 100,
                device=device,
            )
            print(f"[step={step}] Eval return: {mean_ret:.2f} ± {std_ret:.2f}")

    env.close()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
