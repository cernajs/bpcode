import argparse
import numpy as np
import torch
from utils import (
    make_env, get_device, set_seed, preprocess_img, 
    ENV_ACTION_REPEAT
)
from models import ConvEncoder, ConvDecoder, RSSM, RewardModel
from geojacobreg import cem_plan_action_rssm, evaluate_planner_rssm


class NoisyBackgroundWrapper:
    """
    Wrapper that adds random noise to the background of observations.
    This tests if bisimulation helps ignore irrelevant background noise.
    Works with both DMControlWrapper and gymnasium environments.
    """
    def __init__(self, env, noise_strength=0.3, noise_type='uniform'):
        """
        Args:
            env: The environment to wrap (DMControlWrapper or gym.Env)
            noise_strength: Strength of noise (0.0 to 1.0)
            noise_type: 'uniform' or 'gaussian'
        """
        self.env = env
        self.noise_strength = noise_strength
        self.noise_type = noise_type
        
        # Forward action_space and observation_space
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        
    def _add_noise(self, obs):
        """Add noise to observation background."""
        obs = obs.astype(np.float32)
        
        if self.noise_type == 'uniform':
            noise = np.random.uniform(
                -self.noise_strength * 255,
                self.noise_strength * 255,
                size=obs.shape
            ).astype(np.float32)
        elif self.noise_type == 'gaussian':
            noise = np.random.normal(
                0,
                self.noise_strength * 255 / 3,  # 3-sigma rule
                size=obs.shape
            ).astype(np.float32)
        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")
        
        noisy_obs = obs + noise
        noisy_obs = np.clip(noisy_obs, 0, 255).astype(np.uint8)
        return noisy_obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._add_noise(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._add_noise(obs), reward, terminated, truncated, info
    
    def close(self):
        """Forward close() to wrapped environment."""
        if hasattr(self.env, 'close'):
            self.env.close()


def make_noisy_env(env_id: str, img_size=(64, 64), num_stack=1, 
                   noise_strength=0.3, noise_type='uniform'):
    """Create environment with noisy background."""
    env = make_env(env_id, img_size=img_size, num_stack=num_stack)
    return NoisyBackgroundWrapper(env, noise_strength=noise_strength, noise_type=noise_type)


def load_checkpoint(checkpoint_path, device):
    """Load checkpoint and return models and args."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get args from checkpoint
    args_dict = checkpoint.get('args', {})
    
    # Default args if not in checkpoint
    embed_dim = args_dict.get('embed_dim', 1024)
    stoch_dim = args_dict.get('stoch_dim', 30)
    deter_dim = args_dict.get('deter_dim', 200)
    hidden_dim = args_dict.get('hidden_dim', 200)
    
    # Get action space info (will be set from env)
    env_id = args_dict.get('env_id', 'cheetah-run')
    img_size = args_dict.get('img_size', 64)
    
    # Create a temporary env to get action space
    temp_env = make_env(env_id, img_size=(img_size, img_size), num_stack=1)
    act_dim = temp_env.action_space.shape[0]
    temp_env.close()
    
    # Create models
    encoder = ConvEncoder(embedding_size=embed_dim, in_channels=3).to(device)
    decoder = ConvDecoder(
        state_size=deter_dim,
        latent_size=stoch_dim,
        embedding_size=embed_dim,
        out_channels=3
    ).to(device)
    rssm = RSSM(
        stoch_dim=stoch_dim,
        deter_dim=deter_dim,
        act_dim=act_dim,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim
    ).to(device)
    reward_model = RewardModel(
        state_size=deter_dim,
        latent_size=stoch_dim,
        hidden_dim=hidden_dim
    ).to(device)
    
    # Load state dicts
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    rssm.load_state_dict(checkpoint['rssm'])
    reward_model.load_state_dict(checkpoint['reward_model'])
    
    print(f"Loaded checkpoint from episode {checkpoint.get('episode', 'unknown')}, "
          f"total_steps {checkpoint.get('total_steps', 'unknown')}")
    
    return encoder, decoder, rssm, reward_model, args_dict


def evaluate_with_noise(encoder, rssm, reward_model, env_id, img_size,
                        plan_kwargs, noise_strength=0.3, noise_type='uniform',
                        episodes=10, seed=0, device="cpu", bit_depth=5,
                        action_repeat=1):
    """Evaluate planner on environment with noisy background."""
    # Create noisy environment
    env = make_noisy_env(
        env_id, 
        img_size=(img_size, img_size), 
        num_stack=1,
        noise_strength=noise_strength,
        noise_type=noise_type
    )
    
    try:
        env.reset(seed=seed)
    except TypeError:
        pass
    
    encoder.eval()
    rssm.eval()
    reward_model.eval()
    
    returns = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        
        # Initialize state
        obs_t = torch.tensor(
            np.ascontiguousarray(obs), 
            dtype=torch.float32, 
            device=device
        ).permute(2, 0, 1).unsqueeze(0)
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
            obs_t = torch.tensor(
                np.ascontiguousarray(obs), 
                dtype=torch.float32, 
                device=device
            ).permute(2, 0, 1).unsqueeze(0)
            preprocess_img(obs_t, depth=bit_depth)
            enc = encoder(obs_t)
            act_t = torch.tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
            h_state, s_state, _, _ = rssm.observe_step(
                enc, act_t, h_state, s_state, sample=False
            )
        
        returns.append(ep_ret)
        print(f"  Episode {ep+1}/{episodes}: return = {ep_ret:.2f}")
    
    env.close()
    return float(np.mean(returns)), float(np.std(returns))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate learned world model on dm_control with noisy background"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to checkpoint file (.pt)")
    parser.add_argument("--env_id", type=str, default=None,
                       help="Environment ID (overrides checkpoint if provided)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0,
                       help="Random seed")
    
    # Noise parameters
    parser.add_argument("--noise_strength", type=float, default=0.3,
                       help="Strength of background noise (0.0 to 1.0)")
    parser.add_argument("--noise_type", type=str, default="uniform",
                       choices=["uniform", "gaussian"],
                       help="Type of noise to add")
    
    # Evaluation settings
    parser.add_argument("--eval_clean", action="store_true",
                       help="Also evaluate on clean environment for comparison")
    parser.add_argument("--img_size", type=int, default=None,
                       help="Image size (overrides checkpoint if provided)")
    parser.add_argument("--bit_depth", type=int, default=5,
                       help="Bit depth for image preprocessing")
    
    # CEM planner settings
    parser.add_argument("--plan_horizon", type=int, default=12,
                       help="Planning horizon")
    parser.add_argument("--plan_candidates", type=int, default=1000,
                       help="Number of CEM candidates")
    parser.add_argument("--plan_iters", type=int, default=10,
                       help="Number of CEM iterations")
    parser.add_argument("--plan_top_k", type=int, default=100,
                       help="Top-k elite actions in CEM")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Load checkpoint
    encoder, decoder, rssm, reward_model, checkpoint_args = load_checkpoint(
        args.checkpoint, device
    )
    
    # Get environment settings
    env_id = args.env_id or checkpoint_args.get('env_id', 'cheetah-run')
    img_size = args.img_size or checkpoint_args.get('img_size', 64)
    action_repeat = checkpoint_args.get('action_repeat', 0)
    if action_repeat == 0:
        action_repeat = ENV_ACTION_REPEAT.get(env_id, 2)
    
    print(f"\nEnvironment: {env_id}")
    print(f"Image size: {img_size}")
    print(f"Action repeat: {action_repeat}")
    print(f"Noise: {args.noise_type} with strength {args.noise_strength}")
    
    # Get action space
    temp_env = make_env(env_id, img_size=(img_size, img_size), num_stack=1)
    act_low = temp_env.action_space.low
    act_high = temp_env.action_space.high
    temp_env.close()
    
    # CEM planner kwargs
    plan_kwargs = dict(
        act_low=act_low,
        act_high=act_high,
        horizon=args.plan_horizon,
        candidates=args.plan_candidates,
        iters=args.plan_iters,
        top_k=args.plan_top_k,
    )
    
    # Evaluate on noisy environment
    print(f"\n{'='*60}")
    print("Evaluating on NOISY environment")
    print(f"{'='*60}")
    mean_ret_noisy, std_ret_noisy = evaluate_with_noise(
        encoder=encoder,
        rssm=rssm,
        reward_model=reward_model,
        env_id=env_id,
        img_size=img_size,
        plan_kwargs=plan_kwargs,
        noise_strength=args.noise_strength,
        noise_type=args.noise_type,
        episodes=args.episodes,
        seed=args.seed,
        device=device,
        bit_depth=args.bit_depth,
        action_repeat=action_repeat,
    )
    print(f"\nNoisy Environment Results:")
    print(f"  Mean return: {mean_ret_noisy:.2f} ± {std_ret_noisy:.2f}")
    
    # Optionally evaluate on clean environment for comparison
    if args.eval_clean:
        print(f"\n{'='*60}")
        print("Evaluating on CLEAN environment (for comparison)")
        print(f"{'='*60}")
        mean_ret_clean, std_ret_clean = evaluate_planner_rssm(
            env_id=env_id,
            img_size=img_size,
            encoder=encoder,
            rssm=rssm,
            reward_model=reward_model,
            plan_kwargs=plan_kwargs,
            episodes=args.episodes,
            seed=args.seed,
            device=device,
            bit_depth=args.bit_depth,
            action_repeat=action_repeat,
        )
        print(f"\nClean Environment Results:")
        print(f"  Mean return: {mean_ret_clean:.2f} ± {std_ret_clean:.2f}")
        
        print(f"\n{'='*60}")
        print("Comparison:")
        print(f"{'='*60}")
        print(f"Clean:   {mean_ret_clean:.2f} ± {std_ret_clean:.2f}")
        print(f"Noisy:   {mean_ret_noisy:.2f} ± {std_ret_noisy:.2f}")
        degradation = ((mean_ret_clean - mean_ret_noisy) / (abs(mean_ret_clean) + 1e-8)) * 100
        print(f"Degradation: {degradation:.1f}%")
        
        if abs(degradation) < 10:
            print("\n✓ Bisimulation appears to be working! Model is robust to noise.")
        else:
            print("\n✗ Model performance degrades significantly with noise.")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
