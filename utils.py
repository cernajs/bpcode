import os

# Configure dm_control to use software rendering (for headless servers)
# Must be set BEFORE importing dm_control
# Options: 'egl' (GPU headless), 'osmesa' (software), 'glfw' (requires display)
if "MUJOCO_GL" not in os.environ:
    # Try EGL first (GPU-accelerated), fall back to osmesa if DISPLAY not set
    if "DISPLAY" not in os.environ:
        os.environ["MUJOCO_GL"] = "egl"  # or 'osmesa' if EGL fails
    # If you get gladLoadGL errors, run: export MUJOCO_GL=osmesa

import collections
import random
from dataclasses import dataclass

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "ReplayBuffer",
    "get_device",
    "set_seed",
    "preprocess_img",
    "postprocess_img",
    "bottle",
    "random_masking",
    "mask_spatiotemporal",
    "make_env",
    "ENV_ACTION_REPEAT",
    "log_visualizations",
    "log_imagination_rollout",
    "log_reward_prediction",
    "log_latent_reward_structure",
    "log_action_conditioned_prediction",
    "log_latent_dynamics",
]

# Reference PlaNet action repeat settings per environment
ENV_ACTION_REPEAT = {
    "cartpole-swingup": 8,
    "reacher-easy": 4,
    "cheetah-run": 4,
    "finger-spin": 2,
    "ball_in_cup-catch": 4,
    "walker-walk": 2,
    # Gymnasium fallbacks
    "Pendulum-v1": 2,
    "MountainCarContinuous-v0": 4,
}


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def preprocess_img(image, depth=5):
    """
    Preprocesses an observation inplace.
    From float32 Tensor [0, 255] to [-0.5, 0.5]
    Also adds some noise to the observations.
    """
    image.div_(2 ** (8 - depth)).floor_().div_(2**depth).sub_(0.5)
    image.add_(torch.randn_like(image).div_(2**depth)).clamp_(-0.5, 0.5)


from contextlib import contextmanager


@contextmanager
def no_param_grads(module):
    req = [p.requires_grad for p in module.parameters()]
    try:
        for p in module.parameters():
            p.requires_grad_(False)
        yield
    finally:
        for p, r in zip(module.parameters(), req):
            p.requires_grad_(r)


def postprocess_img(image, depth=5):
    """
    Postprocess an image observation for storage.
    From float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """
    image = np.floor((image + 0.5) * 2**depth)
    return np.clip(image * 2 ** (8 - depth), 0, 2**8 - 1).astype(np.uint8)


def bottle(func, *tensors):
    """
    Evaluates a func that operates in N x D with inputs of shape N x T x D.
    Handles both single tensor outputs and tuple outputs (e.g., FeatureDecoder2).
    """
    n, t = tensors[0].shape[:2]
    out = func(*[x.reshape(n * t, *x.shape[2:]) for x in tensors])

    # Handle tuple outputs (e.g., FeatureDecoder2 returns (mu, log_sigma))
    if isinstance(out, tuple):
        return tuple(o.reshape(n, t, *o.shape[1:]) for o in out)
    return out.reshape(n, t, *out.shape[1:])


def random_masking(imgs, mask_ratio=0.5, patch_size=8):
    """
    Randomly zeros out patches of the image (Spatio-Temporal Masking).

    This forces the model to rely on the stochastic state to "fill in the blanks,"
    preventing posterior collapse where s becomes unused.

    Args:
        imgs: [B, T, C, H, W] or [B, C, H, W] tensor of images
        mask_ratio: Fraction of patches to zero out (0.5 = 50% masked)
        patch_size: Size of square patches to mask (default 8x8)

    Returns:
        Masked images with same shape as input
    """
    is_sequence = imgs.dim() == 5
    if is_sequence:
        B, T, C, H, W = imgs.shape
        x = imgs.view(B * T, C, H, W)
    else:
        x = imgs
        B_flat, C, H, W = x.shape

    # Calculate number of patches
    h_patches = H // patch_size
    w_patches = W // patch_size
    num_patches = h_patches * w_patches
    num_masked = int(mask_ratio * num_patches)

    # Create mask [B*T, num_patches]
    mask = torch.ones(x.shape[0], num_patches, device=x.device)

    # Randomly select indices to zero out
    noise = torch.rand(x.shape[0], num_patches, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_mask = ids_shuffle[:, :num_masked]

    mask.scatter_(1, ids_mask, 0)

    # Reshape mask back to image
    mask = mask.view(x.shape[0], 1, h_patches, w_patches)
    # Upsample mask to pixel level (nearest neighbor)
    mask = F.interpolate(mask, size=(H, W), mode="nearest")

    x_masked = x * mask

    if is_sequence:
        x_masked = x_masked.view(B, T, C, H, W)

    return x_masked


def mask_spatiotemporal(video, mask_ratio=0.5, cube_t=2, cube_hw=8, mode="zero"):
    """
    Spatio-temporal cube masking (2405.06263 style).

    Samples 3D cubes (temporal x spatial) and masks them until ~mask_ratio of
    the video is masked. This is more aggressive than per-frame patch masking.

    Args:
        video: [B, T, C, H, W] video tensor
        mask_ratio: target fraction of pixels to mask (0.5 = 50%)
        cube_t: temporal size of cubes (default 2 frames)
        cube_hw: spatial size of cubes (default 8x8 pixels)
        mode: 'zero' (set to 0) or 'noise' (fill with random noise)

    Returns:
        masked_video: [B, T, C, H, W] with cubes masked
        mask_metadata: dict with mask positions (optional, for debugging)
    """
    B, T, C, H, W = video.shape
    device = video.device

    masked_video = video.clone()

    # Calculate how many pixels to mask in total
    total_pixels = B * T * C * H * W
    target_masked = int(mask_ratio * total_pixels)

    # Cube dimensions
    t_cubes = max(1, T // cube_t)
    h_cubes = max(1, H // cube_hw)
    w_cubes = max(1, W // cube_hw)

    pixels_per_cube = cube_t * cube_hw * cube_hw * C
    n_cubes_to_mask = max(1, target_masked // (pixels_per_cube * B))

    # For each batch element, sample cubes to mask
    for b in range(B):
        masked_count = 0
        attempts = 0
        max_attempts = n_cubes_to_mask * 3  # avoid infinite loops

        while masked_count < target_masked / B and attempts < max_attempts:
            # Sample random cube position
            t_start = torch.randint(
                0, max(1, T - cube_t + 1), (1,), device=device
            ).item()
            h_start = torch.randint(
                0, max(1, H - cube_hw + 1), (1,), device=device
            ).item()
            w_start = torch.randint(
                0, max(1, W - cube_hw + 1), (1,), device=device
            ).item()

            t_end = min(t_start + cube_t, T)
            h_end = min(h_start + cube_hw, H)
            w_end = min(w_start + cube_hw, W)

            # Apply mask
            if mode == "zero":
                masked_video[b, t_start:t_end, :, h_start:h_end, w_start:w_end] = 0.0
            elif mode == "noise":
                noise = (
                    torch.randn(
                        t_end - t_start,
                        C,
                        h_end - h_start,
                        w_end - w_start,
                        device=device,
                        dtype=video.dtype,
                    )
                    * 0.1
                )  # small noise
                masked_video[b, t_start:t_end, :, h_start:h_end, w_start:w_end] = noise

            masked_count += (
                (t_end - t_start) * C * (h_end - h_start) * (w_end - w_start)
            )
            attempts += 1

    return masked_video


class PixelObsWrapper(gym.Wrapper):
    """Wrapper for Gymnasium environments to provide pixel observations."""

    def __init__(self, env_id: str, img_size=(64, 64), num_stack=1):
        env = gym.make(env_id, render_mode="rgb_array")
        super().__init__(env)
        self.img_size = img_size
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        self.env.reset()
        img = self._get_obs()

        # Shape: (H, W, C * Stack) or (H, W, C) if num_stack=1
        obs_shape = (img.shape[0], img.shape[1], img.shape[2] * num_stack)

        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def _get_obs(self):
        img = self.env.render()
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        return img  # Returns (H, W, C)

    def _get_stacked_obs(self):
        if self.num_stack == 1:
            return list(self.frames)[0]
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        img = self._get_obs()
        for _ in range(self.num_stack):
            self.frames.append(img)
        return self._get_stacked_obs(), info

    """
    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        img = self._get_obs()
        self.frames.append(img)
        return self._get_stacked_obs(), reward, terminated, truncated, info
    """

    def step(self, action, repeat: int = 1):
        """Frame-skip style step.

        Advance the underlying env `repeat` times using the same action,
        sum rewards, and render only once at the end.
        """
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(int(repeat)):
            _, r, terminated, truncated, info = self.env.step(action)
            total_reward += float(r)
            if terminated or truncated:
                break

        img = self._get_obs()  # single render at end
        self.frames.append(img)
        return self._get_stacked_obs(), total_reward, terminated, truncated, info


def _configure_dm_control_rendering():
    """Try to configure dm_control rendering backend for headless servers."""
    # Already configured via environment variable at top of file
    # This function provides additional fallback logic if needed
    pass


class DMControlWrapper:
    """
    Wrapper for DeepMind Control Suite environments.
    Standard benchmark for PlaNet: cheetah-run, reacher-easy, ball_in_cup-catch, finger-spin, cartpole-swingup, walker-walk
    """

    def __init__(self, domain_task: str, img_size=(64, 64), num_stack=1):
        try:
            from dm_control import suite
        except ImportError:
            raise ImportError(
                "dm_control not installed. Install with: pip install dm_control\n"
                "For headless servers, you may also need: apt-get install libosmesa6-dev libgl1-mesa-glx libglfw3"
            )

        # Parse "domain-task" format (e.g., "cheetah-run")
        domain, task = domain_task.split("-", 1)
        self.env = suite.load(domain, task)

        self.img_size = img_size
        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        # Get action spec
        action_spec = self.env.action_spec()
        self.action_space = gym.spaces.Box(
            low=action_spec.minimum.astype(np.float32),
            high=action_spec.maximum.astype(np.float32),
            dtype=np.float32,
        )

        # Observation space (pixel-based)
        obs_shape = (img_size[0], img_size[1], 3 * num_stack)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8
        )

    def _get_obs(self):
        img = self.env.physics.render(
            height=self.img_size[0], width=self.img_size[1], camera_id=0
        )
        return img  # Returns (H, W, C) uint8

    def _get_stacked_obs(self):
        if self.num_stack == 1:
            return list(self.frames)[0]
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        time_step = self.env.reset()
        img = self._get_obs()
        for _ in range(self.num_stack):
            self.frames.append(img)
        return self._get_stacked_obs(), {}

    """
    def step(self, action):
        time_step = self.env.step(action)
        img = self._get_obs()
        self.frames.append(img)

        reward = time_step.reward or 0.0
        terminated = time_step.last()
        truncated = False

        return self._get_stacked_obs(), reward, terminated, truncated, {}
    """

    def step(self, action, repeat: int = 1):
        """Frame-skip style step for dm_control.

        Advance `repeat` physics steps with the same action, sum rewards,
        stop early on termination, then render once.
        """
        total_reward = 0.0
        terminated = False
        truncated = False  # dm_control suite doesn't expose truncation here

        for _ in range(int(repeat)):
            time_step = self.env.step(action)
            total_reward += float(time_step.reward or 0.0)
            terminated = bool(time_step.last())
            if terminated:
                break

        img = self._get_obs()  # single render at end
        self.frames.append(img)

        return self._get_stacked_obs(), total_reward, terminated, truncated, {}

    def close(self):
        pass


def make_env(env_id: str, img_size=(64, 64), num_stack=1):
    """
    Create environment - supports both dm_control (domain-task format) and gymnasium.

    dm_control envs: "cheetah-run", "reacher-easy", "ball_in_cup-catch", "finger-spin", "cartpole-swingup", "walker-walk"
    gymnasium envs: "Pendulum-v1", "MountainCarContinuous-v0", etc.
    """
    # Check if it's a dm_control env (contains hyphen but no version suffix like -v1)
    if "-" in env_id and not any(env_id.endswith(f"-v{i}") for i in range(10)):
        return DMControlWrapper(env_id, img_size=img_size, num_stack=num_stack)
    else:
        return PixelObsWrapper(env_id, img_size=img_size, num_stack=num_stack)


# ===============================
#  Replay Buffer
# ===============================
# currently buggy, if we get more than self.capacity we wrap around doing teleportation
class ReplayBuffer:
    def __init__(self, capacity: int, obs_shape, act_dim: int):
        """
        obs_shape: (H, W, C) pixel observations stored as uint8 for memory efficiency.
        """
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)

        self.idx = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done: bool):
        self.obs[self.idx] = obs
        self.actions[self.idx] = action
        self.rews[self.idx] = reward
        self.dones[self.idx] = float(done)
        self.next_obs[self.idx] = next_obs

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    @dataclass
    class Batch:
        obs: np.ndarray  # [B, T, H, W, C]
        actions: np.ndarray  # [B, T, act_dim]
        rews: np.ndarray  # [B, T]
        dones: np.ndarray  # [B, T]

    def sample_sequences(self, batch_size: int, seq_len: int):
        """
        Sample sequences consistent with episode boundaries and buffer pointers.
        Fixes the 'teleportation' issue by rejecting invalid indices.
        """
        assert self.size > seq_len + 1, "Not enough data to sample sequences."

        obs_seq = []
        act_seq = []
        rew_seq = []
        done_seq = []

        # We need to find 'batch_size' valid starting indices
        count = 0
        while count < batch_size:
            # 1. Sample a random start index
            # We subtract seq_len to ensure the slice fits in the array
            start = np.random.randint(0, self.size - seq_len)
            end = start + seq_len

            # 2. Check for Buffer Overwrite (Circular Buffer "Head")
            # If the buffer is full, self.idx is the split between oldest and newest data.
            # We cannot sample a sequence that crosses this boundary.
            if self.size == self.capacity:
                # If the interval [start, end] contains the write head 'self.idx'
                if start < self.idx < end:
                    continue

            # 3. Check for Episode Boundaries
            # If any step (except the last one) is 'done', the sequence contains a reset.
            # We want transitions: s_0->s_1, ..., s_{T-1}->s_T.
            # If s_k is terminal, the transition s_k -> s_{k+1} is invalid (teleportation).
            if np.any(self.dones[start : end - 1]):
                continue

            # If valid, append to batch
            obs_seq.append(self.obs[start:end])
            act_seq.append(self.actions[start:end])
            rew_seq.append(self.rews[start:end])
            done_seq.append(self.dones[start:end])
            count += 1

        obs_seq = np.stack(obs_seq, axis=0)  # [B, T, H, W, C]
        act_seq = np.stack(act_seq, axis=0)  # [B, T, act_dim]
        rew_seq = np.stack(rew_seq, axis=0)  # [B, T]
        done_seq = np.stack(done_seq, axis=0)  # [B, T]

        return ReplayBuffer.Batch(obs_seq, act_seq, rew_seq, done_seq)


# ===============================
#  Visualization Utilities
# ===============================


def log_visualizations(
    writer,
    step,
    encoder,
    decoder,
    rssm,
    obs_batch,  # [B, T, C, H, W] preprocessed observations
    act_batch,  # [B, T, act_dim] actions
    h_seq,  # [B, T, deter_dim] deterministic states
    s_seq,  # [B, T, stoch_dim] stochastic states
    bit_depth=5,
    knn_k=8,
    device="cpu",
):
    """
    Log visualizations to TensorBoard:
    1. Reconstruction comparison (original vs reconstructed)
    2. Latent state heatmap (h and s concatenated)
    3. kNN graph connectivity in latent space (2D PCA projection)

    Args:
        writer: TensorBoard SummaryWriter
        step: Current training step
        encoder, decoder, rssm: Model components
        obs_batch: Preprocessed observations [B, T, C, H, W]
        act_batch: Actions [B, T, act_dim]
        h_seq: Deterministic states [B, T, deter_dim]
        s_seq: Stochastic states [B, T, stoch_dim]
        bit_depth: Bit depth for image preprocessing
        knn_k: Number of neighbors for kNN graph
        device: Torch device
    """
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import io

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from PIL import Image
    from sklearn.decomposition import PCA

    encoder.eval()
    decoder.eval()
    rssm.eval()

    with torch.no_grad():
        B, T = obs_batch.shape[:2]

        # ===== 1. Reconstruction Comparison =====
        # Take first batch, multiple time steps
        n_show = min(8, T)
        time_indices = np.linspace(0, T - 1, n_show, dtype=int)

        fig, axes = plt.subplots(2, n_show, figsize=(2 * n_show, 4))
        if n_show == 1:
            axes = axes.reshape(2, 1)

        for i, t in enumerate(time_indices):
            # Original observation (convert back to displayable)
            orig = obs_batch[0, t].cpu()  # [C, H, W]
            orig_disp = (orig + 0.5).clamp(0, 1)  # [-0.5, 0.5] -> [0, 1]

            # Reconstruction
            h_t = h_seq[0:1, t]  # [1, deter_dim]
            s_t = s_seq[0:1, t]  # [1, stoch_dim]
            recon = decoder(h_t, s_t)  # [1, C, H, W]
            recon_disp = (recon[0].cpu() + 0.5).clamp(0, 1)

            # Plot original
            if orig_disp.shape[0] == 3:
                axes[0, i].imshow(orig_disp.permute(1, 2, 0).numpy())
            else:
                axes[0, i].imshow(orig_disp[0].numpy(), cmap="gray")
            axes[0, i].set_title(f"t={t}", fontsize=8)
            axes[0, i].axis("off")

            # Plot reconstruction
            if recon_disp.shape[0] == 3:
                axes[1, i].imshow(recon_disp.permute(1, 2, 0).numpy())
            else:
                axes[1, i].imshow(recon_disp[0].numpy(), cmap="gray")
            axes[1, i].axis("off")

        axes[0, 0].set_ylabel("Original", fontsize=10)
        axes[1, 0].set_ylabel("Recon", fontsize=10)
        plt.suptitle(f"Reconstruction @ step {step}", fontsize=12)
        plt.tight_layout()

        # Convert to tensor for TensorBoard
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/reconstruction", img_tensor, step)
        plt.close(fig)

        # ===== 2. Latent State Heatmap =====
        # Show h and s as heatmaps for first batch item
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Deterministic state h
        h_np = h_seq[0].cpu().numpy()  # [T, deter_dim]
        im0 = axes[0].imshow(h_np.T, aspect="auto", cmap="viridis")
        axes[0].set_xlabel("Time step")
        axes[0].set_ylabel("Dimension")
        axes[0].set_title(f"Deterministic state h (deter_dim={h_np.shape[1]})")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        # Stochastic state s
        s_np = s_seq[0].cpu().numpy()  # [T, stoch_dim]
        im1 = axes[1].imshow(s_np.T, aspect="auto", cmap="plasma")
        axes[1].set_xlabel("Time step")
        axes[1].set_ylabel("Dimension")
        axes[1].set_title(f"Stochastic state s (stoch_dim={s_np.shape[1]})")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        plt.suptitle(f"Latent States @ step {step}", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/latent_states", img_tensor, step)
        plt.close(fig)

        # ===== 3. kNN Graph in Latent Space =====
        # Combine h and s, flatten across batch and time
        z_combined = torch.cat([h_seq, s_seq], dim=-1)  # [B, T, deter+stoch]
        z_flat = z_combined.reshape(-1, z_combined.shape[-1]).cpu().numpy()  # [B*T, D]

        # Time labels for coloring
        time_labels = np.tile(np.arange(T), B)
        batch_labels = np.repeat(np.arange(B), T)

        # PCA to 2D
        if z_flat.shape[0] > 2:
            pca = PCA(n_components=2)
            z_2d = pca.fit_transform(z_flat)
        else:
            z_2d = (
                z_flat[:, :2]
                if z_flat.shape[1] >= 2
                else np.zeros((z_flat.shape[0], 2))
            )

        # Build kNN graph
        z_tensor = torch.tensor(z_flat, device="cpu")
        dist_matrix = torch.cdist(z_tensor, z_tensor, p=2)
        dist_matrix.fill_diagonal_(float("inf"))
        knn_indices = dist_matrix.topk(knn_k, largest=False).indices.numpy()  # [N, k]

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: kNN edges colored by whether they connect same batch
        ax = axes[0]
        # Draw edges first (behind points)
        n_points = z_2d.shape[0]
        same_batch_edges = 0
        cross_batch_edges = 0
        same_time_edges = 0

        for i in range(n_points):
            for j in knn_indices[i]:
                same_batch = batch_labels[i] == batch_labels[j]
                if same_batch:
                    ax.plot(
                        [z_2d[i, 0], z_2d[j, 0]],
                        [z_2d[i, 1], z_2d[j, 1]],
                        "b-",
                        alpha=0.1,
                        linewidth=0.5,
                    )
                    same_batch_edges += 1
                else:
                    ax.plot(
                        [z_2d[i, 0], z_2d[j, 0]],
                        [z_2d[i, 1], z_2d[j, 1]],
                        "r-",
                        alpha=0.2,
                        linewidth=0.5,
                    )
                    cross_batch_edges += 1
                if time_labels[i] == time_labels[j]:
                    same_time_edges += 1

        # Draw points colored by time
        scatter = ax.scatter(
            z_2d[:, 0],
            z_2d[:, 1],
            c=time_labels,
            cmap="coolwarm",
            s=20,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.3,
        )
        plt.colorbar(scatter, ax=ax, label="Time step")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"kNN Graph (k={knn_k})\nBlue=same batch, Red=cross batch")

        # Right: Statistics and connectivity info
        ax = axes[1]
        ax.axis("off")

        total_edges = n_points * knn_k
        stats_text = f"""kNN Graph Statistics @ step {step}

Total points: {n_points} ({B} batches × {T} timesteps)
k (neighbors): {knn_k}
Total edges: {total_edges}

Edge breakdown:
  Same-batch edges: {same_batch_edges} ({100 * same_batch_edges / total_edges:.1f}%)
  Cross-batch edges: {cross_batch_edges} ({100 * cross_batch_edges / total_edges:.1f}%)
  Same-time edges: {same_time_edges} ({100 * same_time_edges / total_edges:.1f}%)

Latent dimensions:
  h (deterministic): {h_seq.shape[-1]}
  s (stochastic): {s_seq.shape[-1]}
  combined: {z_combined.shape[-1]}

PCA variance explained:
  PC1: {pca.explained_variance_ratio_[0] * 100:.1f}%
  PC2: {pca.explained_variance_ratio_[1] * 100:.1f}%
  Total: {sum(pca.explained_variance_ratio_) * 100:.1f}%
"""
        ax.text(
            0.1,
            0.9,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.suptitle(f"kNN Connectivity Analysis @ step {step}", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/knn_graph", img_tensor, step)
        plt.close(fig)

        # Log scalar metrics for kNN connectivity
        writer.add_scalar(
            "viz_stats/cross_batch_edge_ratio", cross_batch_edges / total_edges, step
        )
        writer.add_scalar(
            "viz_stats/same_time_edge_ratio", same_time_edges / total_edges, step
        )
        writer.add_scalar(
            "viz_stats/pca_variance_explained", sum(pca.explained_variance_ratio_), step
        )


def log_imagination_rollout(
    writer,
    step,
    encoder,
    decoder,
    rssm,
    obs_batch,  # [B, T, C, H, W] preprocessed observations
    act_batch,  # [B, T, act_dim] actions
    bit_depth=5,
    imagination_horizon=15,
    device="cpu",
):
    """
    Visualize open-loop imagination: what the model predicts without new observations.
    Shows real trajectory vs imagined trajectory to assess dynamics quality.
    """
    import matplotlib

    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from PIL import Image

    encoder.eval()
    decoder.eval()
    rssm.eval()

    with torch.no_grad():
        B, T, C, H, W = obs_batch.shape
        horizon = min(imagination_horizon, T - 1)

        # Encode first observation to get initial state
        e_0 = encoder(obs_batch[:, 0])  # [B, embed_dim]
        h, s = rssm.get_init_state(e_0)

        # Real trajectory (with observations)
        real_h_list, real_s_list = [h], [s]
        for t in range(horizon):
            h = rssm.deterministic_state_fwd(h, s, act_batch[:, t])
            e_t = encoder(obs_batch[:, t + 1])
            post_mean, post_std = rssm.state_posterior(h, e_t)
            s = post_mean
            real_h_list.append(h)
            real_s_list.append(s)

        # Imagined trajectory (open-loop, using prior only)
        h, s = rssm.get_init_state(e_0)
        imag_h_list, imag_s_list = [h], [s]
        for t in range(horizon):
            h = rssm.deterministic_state_fwd(h, s, act_batch[:, t])
            prior_mean, prior_std = rssm.state_prior(h)
            s = prior_mean  # Use prior mean (no observation)
            imag_h_list.append(h)
            imag_s_list.append(s)

        # Decode both trajectories
        real_h = torch.stack(real_h_list[1:], dim=1)  # [B, horizon, deter]
        real_s = torch.stack(real_s_list[1:], dim=1)  # [B, horizon, stoch]
        imag_h = torch.stack(imag_h_list[1:], dim=1)
        imag_s = torch.stack(imag_s_list[1:], dim=1)

        real_recon = bottle(decoder, real_h, real_s)  # [B, horizon, C, H, W]
        imag_recon = bottle(decoder, imag_h, imag_s)

        # Visualize first batch item
        n_show = min(8, horizon)
        time_indices = np.linspace(0, horizon - 1, n_show, dtype=int)

        fig, axes = plt.subplots(3, n_show, figsize=(2 * n_show, 6))

        for i, t in enumerate(time_indices):
            # Ground truth
            gt = (obs_batch[0, t + 1].cpu() + 0.5).clamp(0, 1)
            if gt.shape[0] == 3:
                axes[0, i].imshow(gt.permute(1, 2, 0).numpy())
            else:
                axes[0, i].imshow(gt[0].numpy(), cmap="gray")
            axes[0, i].set_title(f"t={t + 1}", fontsize=8)
            axes[0, i].axis("off")

            # Real reconstruction (with observations)
            real = (real_recon[0, t].cpu() + 0.5).clamp(0, 1)
            if real.shape[0] == 3:
                axes[1, i].imshow(real.permute(1, 2, 0).numpy())
            else:
                axes[1, i].imshow(real[0].numpy(), cmap="gray")
            axes[1, i].axis("off")

            # Imagined (open-loop)
            imag = (imag_recon[0, t].cpu() + 0.5).clamp(0, 1)
            if imag.shape[0] == 3:
                axes[2, i].imshow(imag.permute(1, 2, 0).numpy())
            else:
                axes[2, i].imshow(imag[0].numpy(), cmap="gray")
            axes[2, i].axis("off")

        axes[0, 0].set_ylabel("Ground Truth", fontsize=9)
        axes[1, 0].set_ylabel("Real (w/ obs)", fontsize=9)
        axes[2, 0].set_ylabel("Imagined", fontsize=9)
        plt.suptitle(f"Imagination Rollout @ step {step}", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/imagination_rollout", img_tensor, step)
        plt.close(fig)

        # Compute and log imagination error over time
        mse_per_step = ((imag_recon[0] - obs_batch[0, 1 : horizon + 1]) ** 2).mean(
            dim=(1, 2, 3)
        )

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(
            range(1, horizon + 1),
            mse_per_step.cpu().numpy(),
            "b-o",
            label="Imagination MSE",
        )
        ax.set_xlabel("Steps into future")
        ax.set_ylabel("MSE")
        ax.set_title(f"Imagination Error Growth @ step {step}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/imagination_error", img_tensor, step)
        plt.close(fig)

        # Log scalar: average imagination error
        writer.add_scalar(
            "viz_stats/imagination_mse_mean", mse_per_step.mean().item(), step
        )
        writer.add_scalar(
            "viz_stats/imagination_mse_final", mse_per_step[-1].item(), step
        )


def log_reward_prediction(
    writer,
    step,
    encoder,
    rssm,
    reward_model,
    obs_batch,  # [B, T, C, H, W]
    act_batch,  # [B, T, act_dim]
    rew_batch,  # [B, T] actual rewards
    device="cpu",
):
    """
    Visualize reward prediction accuracy: predicted vs actual rewards.
    """
    import matplotlib

    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from PIL import Image

    encoder.eval()
    rssm.eval()
    reward_model.eval()

    with torch.no_grad():
        B, T = rew_batch.shape
        T = T - 1  # We predict T-1 rewards

        # Compute latent states
        e_t = bottle(encoder, obs_batch)
        h, s = rssm.get_init_state(e_t[:, 0])

        h_list, s_list = [], []
        for t in range(T):
            h = rssm.deterministic_state_fwd(h, s, act_batch[:, t])
            h_list.append(h)
            post_mean, post_std = rssm.state_posterior(h, e_t[:, t + 1])
            s = post_mean
            s_list.append(s)

        h_seq = torch.stack(h_list, dim=1)
        s_seq = torch.stack(s_list, dim=1)

        # Predict rewards
        pred_rew = bottle(reward_model, h_seq, s_seq)  # [B, T]
        actual_rew = rew_batch[:, :T]

        # Flatten for scatter plot
        pred_flat = pred_rew.reshape(-1).cpu().numpy()
        actual_flat = actual_rew.reshape(-1).cpu().numpy()

        # Scatter plot: predicted vs actual
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Scatter plot
        ax = axes[0]
        ax.scatter(actual_flat, pred_flat, alpha=0.3, s=10)
        min_val = min(actual_flat.min(), pred_flat.min())
        max_val = max(actual_flat.max(), pred_flat.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect")
        ax.set_xlabel("Actual Reward")
        ax.set_ylabel("Predicted Reward")
        ax.set_title("Reward Prediction Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Time series for first batch
        ax = axes[1]
        ax.plot(range(T), actual_rew[0].cpu().numpy(), "b-", label="Actual", alpha=0.8)
        ax.plot(
            range(T), pred_rew[0].cpu().numpy(), "r--", label="Predicted", alpha=0.8
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Reward")
        ax.set_title("Reward Over Time (Batch 0)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Error histogram
        ax = axes[2]
        errors = pred_flat - actual_flat
        ax.hist(errors, bins=50, alpha=0.7, edgecolor="black")
        ax.axvline(0, color="r", linestyle="--", label="Zero error")
        ax.set_xlabel("Prediction Error")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Error Distribution (μ={errors.mean():.3f}, σ={errors.std():.3f})"
        )
        ax.legend()

        plt.suptitle(f"Reward Prediction @ step {step}", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/reward_prediction", img_tensor, step)
        plt.close(fig)

        # Log metrics
        mse = ((pred_rew - actual_rew) ** 2).mean().item()
        correlation = np.corrcoef(pred_flat, actual_flat)[0, 1]
        writer.add_scalar("viz_stats/reward_mse", mse, step)
        writer.add_scalar("viz_stats/reward_correlation", correlation, step)


def log_latent_reward_structure(
    writer,
    step,
    h_seq,  # [B, T, deter_dim]
    s_seq,  # [B, T, stoch_dim]
    rew_batch,  # [B, T]
    device="cpu",
):
    """
    Visualize if latent space has bisimulation structure:
    - States with similar rewards should be close
    - Plot latent distance vs reward difference
    """
    import matplotlib

    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from PIL import Image
    from sklearn.decomposition import PCA

    with torch.no_grad():
        B, T, _ = h_seq.shape

        # Combine h and s
        z = torch.cat([h_seq, s_seq], dim=-1)  # [B, T, D]
        z_flat = z.reshape(-1, z.shape[-1])  # [B*T, D]
        rew_flat = rew_batch[:, :T].reshape(-1)  # [B*T]

        N = z_flat.shape[0]

        # Sample pairs (full pairwise is expensive)
        n_pairs = min(5000, N * (N - 1) // 2)
        idx1 = torch.randint(0, N, (n_pairs,), device=z_flat.device)
        idx2 = torch.randint(0, N, (n_pairs,), device=z_flat.device)
        # Avoid self-pairs
        mask = idx1 != idx2
        idx1, idx2 = idx1[mask], idx2[mask]

        # Compute distances
        z_dist = torch.norm(z_flat[idx1] - z_flat[idx2], p=2, dim=1).cpu().numpy()
        rew_diff = torch.abs(rew_flat[idx1] - rew_flat[idx2]).cpu().numpy()

        # PCA for visualization
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_flat.cpu().numpy())
        rew_np = rew_flat.cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # 1. Latent distance vs reward difference (bisimulation check)
        ax = axes[0]
        ax.scatter(rew_diff, z_dist, alpha=0.1, s=5)
        ax.set_xlabel("|Reward Difference|")
        ax.set_ylabel("Latent Distance")
        ax.set_title("Bisimulation Structure\n(Should correlate if working)")
        ax.grid(True, alpha=0.3)

        # Add correlation
        corr = np.corrcoef(rew_diff, z_dist)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"Corr: {corr:.3f}",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        # 2. Latent space colored by reward
        ax = axes[1]
        scatter = ax.scatter(
            z_2d[:, 0], z_2d[:, 1], c=rew_np, cmap="RdYlGn", s=10, alpha=0.5
        )
        plt.colorbar(scatter, ax=ax, label="Reward")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Latent Space by Reward\n(Similar rewards should cluster)")

        # 3. Reward distribution
        ax = axes[2]
        ax.hist(rew_np, bins=50, alpha=0.7, edgecolor="black")
        ax.set_xlabel("Reward")
        ax.set_ylabel("Count")
        ax.set_title(
            f"Reward Distribution\n(μ={rew_np.mean():.2f}, σ={rew_np.std():.2f})"
        )
        ax.axvline(rew_np.mean(), color="r", linestyle="--", label="Mean")
        ax.legend()

        plt.suptitle(f"Latent-Reward Structure @ step {step}", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/latent_reward_structure", img_tensor, step)
        plt.close(fig)

        # Log bisimulation correlation
        writer.add_scalar("viz_stats/bisim_correlation", corr, step)


def log_action_conditioned_prediction(
    writer,
    step,
    encoder,
    decoder,
    rssm,
    obs_batch,  # [B, T, C, H, W]
    act_batch,  # [B, T, act_dim]
    act_dim,
    horizon=5,
    device="cpu",
):
    """
    Show different futures for different actions from the same initial state.
    Helps verify the model is action-sensitive.
    """
    import matplotlib

    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from PIL import Image

    encoder.eval()
    decoder.eval()
    rssm.eval()

    with torch.no_grad():
        # Use first observation as starting point
        e_0 = encoder(obs_batch[0:1, 0])  # [1, embed_dim]
        h_init, s_init = rssm.get_init_state(e_0)

        # Create different action sequences
        n_actions = 5
        action_sequences = []
        labels = []

        # 1. Original actions
        action_sequences.append(act_batch[0:1, :horizon])
        labels.append("Original")

        # 2. Zero actions
        action_sequences.append(torch.zeros(1, horizon, act_dim, device=device))
        labels.append("Zero")

        # 3. Positive max actions
        action_sequences.append(torch.ones(1, horizon, act_dim, device=device))
        labels.append("+1 All")

        # 4. Negative max actions
        action_sequences.append(-torch.ones(1, horizon, act_dim, device=device))
        labels.append("-1 All")

        # 5. Random actions
        action_sequences.append(
            torch.randn(1, horizon, act_dim, device=device).clamp(-1, 1)
        )
        labels.append("Random")

        # Roll out each action sequence
        predictions = []
        for actions in action_sequences:
            h, s = h_init.clone(), s_init.clone()
            frames = []
            for t in range(horizon):
                h = rssm.deterministic_state_fwd(h, s, actions[:, t])
                prior_mean, _ = rssm.state_prior(h)
                s = prior_mean
                recon = decoder(h, s)
                frames.append(recon)
            predictions.append(torch.cat(frames, dim=0))  # [horizon, C, H, W]

        # Plot
        fig, axes = plt.subplots(
            n_actions + 1, horizon, figsize=(2 * horizon, 2 * (n_actions + 1))
        )

        # First row: ground truth
        for t in range(horizon):
            gt = (obs_batch[0, t + 1].cpu() + 0.5).clamp(0, 1)
            if gt.shape[0] == 3:
                axes[0, t].imshow(gt.permute(1, 2, 0).numpy())
            else:
                axes[0, t].imshow(gt[0].numpy(), cmap="gray")
            axes[0, t].axis("off")
            if t == 0:
                axes[0, t].set_ylabel("Ground Truth", fontsize=9)
            axes[0, t].set_title(f"t={t + 1}", fontsize=8)

        # Remaining rows: different action sequences
        for row, (pred, label) in enumerate(zip(predictions, labels), 1):
            for t in range(horizon):
                img = (pred[t].cpu() + 0.5).clamp(0, 1)
                if img.shape[0] == 3:
                    axes[row, t].imshow(img.permute(1, 2, 0).numpy())
                else:
                    axes[row, t].imshow(img[0].numpy(), cmap="gray")
                axes[row, t].axis("off")
                if t == 0:
                    axes[row, t].set_ylabel(label, fontsize=9)

        plt.suptitle(f"Action-Conditioned Predictions @ step {step}", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/action_conditioned", img_tensor, step)
        plt.close(fig)

        # Compute action sensitivity: how different are predictions for different actions?
        pred_stack = torch.stack(
            [p.reshape(-1) for p in predictions]
        )  # [n_actions, horizon*C*H*W]
        pairwise_dist = torch.cdist(pred_stack, pred_stack, p=2)
        action_sensitivity = pairwise_dist.mean().item()
        writer.add_scalar("viz_stats/action_sensitivity", action_sensitivity, step)


def log_latent_dynamics(
    writer,
    step,
    h_seq,  # [B, T, deter_dim]
    s_seq,  # [B, T, stoch_dim]
    act_batch,  # [B, T, act_dim]
    device="cpu",
):
    """
    Visualize latent dynamics: how states evolve and cluster.
    Shows trajectory structure in latent space.
    """
    import matplotlib

    matplotlib.use("Agg")
    import io

    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as TF
    from PIL import Image
    from sklearn.decomposition import PCA

    with torch.no_grad():
        B, T, _ = h_seq.shape

        # Combine h and s
        z = torch.cat([h_seq, s_seq], dim=-1).cpu().numpy()  # [B, T, D]

        # PCA on all points
        z_flat = z.reshape(-1, z.shape[-1])
        pca = PCA(n_components=2)
        z_2d = pca.fit_transform(z_flat).reshape(B, T, 2)

        # Action magnitude for coloring
        act_mag = torch.norm(act_batch, dim=-1).cpu().numpy()  # [B, T]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # 1. Trajectories colored by batch
        ax = axes[0]
        colors = plt.cm.tab10(np.linspace(0, 1, min(B, 10)))
        for b in range(min(B, 10)):
            ax.plot(
                z_2d[b, :, 0],
                z_2d[b, :, 1],
                "-",
                color=colors[b],
                alpha=0.7,
                linewidth=1,
            )
            ax.scatter(
                z_2d[b, 0, 0],
                z_2d[b, 0, 1],
                color=colors[b],
                s=50,
                marker="o",
                edgecolors="black",
            )  # Start
            ax.scatter(
                z_2d[b, -1, 0], z_2d[b, -1, 1], color=colors[b], s=50, marker="x"
            )  # End
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Latent Trajectories\n(○=start, ×=end)")
        ax.grid(True, alpha=0.3)

        # 2. All points colored by time
        ax = axes[1]
        time_colors = np.tile(np.arange(T), B)
        scatter = ax.scatter(
            z_2d.reshape(-1, 2)[:, 0],
            z_2d.reshape(-1, 2)[:, 1],
            c=time_colors,
            cmap="viridis",
            s=10,
            alpha=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Time step")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title("Latent Space by Time")
        ax.grid(True, alpha=0.3)

        # 3. Latent velocity (how much state changes per step)
        ax = axes[2]
        z_velocity = np.linalg.norm(np.diff(z, axis=1), axis=-1)  # [B, T-1]
        mean_vel = z_velocity.mean(axis=0)
        std_vel = z_velocity.std(axis=0)
        ax.fill_between(range(T - 1), mean_vel - std_vel, mean_vel + std_vel, alpha=0.3)
        ax.plot(range(T - 1), mean_vel, "b-", linewidth=2)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Latent Velocity (||Δz||)")
        ax.set_title("Latent Dynamics Speed")
        ax.grid(True, alpha=0.3)

        plt.suptitle(f"Latent Dynamics @ step {step}", fontsize=12)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
        buf.seek(0)
        img = Image.open(buf)
        img_tensor = TF.to_tensor(img)
        writer.add_image("viz/latent_dynamics", img_tensor, step)
        plt.close(fig)

        # Log metrics
        writer.add_scalar("viz_stats/latent_velocity_mean", z_velocity.mean(), step)
        writer.add_scalar("viz_stats/latent_spread", z_flat.std(), step)
