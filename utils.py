import torch
import random
import collections
import gymnasium as gym
import numpy as np
import cv2
from dataclasses import dataclass

__all__ = ['PixelObsWrapper', 'ReplayBuffer', 'get_device', 'set_seed', 
           'preprocess_img', 'postprocess_img', 'bottle']

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
    image.div_(2 ** (8 - depth)).floor_().div_(2 ** depth).sub_(0.5)
    image.add_(torch.randn_like(image).div_(2 ** depth)).clamp_(-0.5, 0.5)


def postprocess_img(image, depth=5):
    """
    Postprocess an image observation for storage.
    From float32 numpy array [-0.5, 0.5] to uint8 numpy array [0, 255])
    """
    image = np.floor((image + 0.5) * 2 ** depth)
    return np.clip(image * 2**(8 - depth), 0, 2**8 - 1).astype(np.uint8)


def bottle(func, *tensors):
    """
    Evaluates a func that operates in N x D with inputs of shape N x T x D.
    """
    n, t = tensors[0].shape[:2]
    out = func(*[x.view(n*t, *x.shape[2:]) for x in tensors])
    return out.view(n, t, *out.shape[1:])


class PixelObsWrapper(gym.Wrapper):
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

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        img = self._get_obs()
        self.frames.append(img)
        return self._get_stacked_obs(), reward, terminated, truncated, info


# ===============================
#  Replay Buffer
# ===============================

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
        obs: np.ndarray        # [B, T, H, W, C]
        actions: np.ndarray    # [B, T, act_dim]
        rews: np.ndarray       # [B, T]
        dones: np.ndarray      # [B, T]

    def sample_sequences(self, batch_size: int, seq_len: int):
        """
        Sample sequences from the flat buffer.
        Sequences may cross episode boundaries, but downstream losses
        mask out transitions after dones.
        """
        assert self.size > seq_len + 1, "Not enough data to sample sequences."
        obs_seq = []
        act_seq = []
        rew_seq = []
        done_seq = []

        max_start = self.size - seq_len - 1
        for _ in range(batch_size):
            start = np.random.randint(0, max_start)
            end = start + seq_len

            obs_seq.append(self.obs[start:end])
            act_seq.append(self.actions[start:end])
            rew_seq.append(self.rews[start:end])
            done_seq.append(self.dones[start:end])

        obs_seq = np.stack(obs_seq, axis=0)          # [B, T, H, W, C]
        act_seq = np.stack(act_seq, axis=0)          # [B, T, act_dim]
        rew_seq = np.stack(rew_seq, axis=0)          # [B, T]
        done_seq = np.stack(done_seq, axis=0)        # [B, T]

        return ReplayBuffer.Batch(obs_seq, act_seq, rew_seq, done_seq)
