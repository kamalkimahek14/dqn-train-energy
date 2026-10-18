# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Uniform experience replay in plain numpy, shared by the PyTorch and TensorFlow agents."""
import numpy as np


class ReplayBuffer:
    """Fixed-size ring buffer. When full, the oldest transition is overwritten."""

    def __init__(self, capacity: int, obs_dim: int, seed: int = 0):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)  # 1.0 = terminal: no bootstrap from next_obs
        self.idx = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done):
        i = self.idx
        self.obs[i], self.actions[i], self.rewards[i] = obs, action, reward
        self.next_obs[i], self.dones[i] = next_obs, float(done)
        self.idx = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        """Uniform sample with replacement; returns a dict of numpy arrays."""
        j = self.rng.integers(0, self.size, size=batch_size)
        return {"obs": self.obs[j], "actions": self.actions[j], "rewards": self.rewards[j],
                "next_obs": self.next_obs[j], "dones": self.dones[j]}

    def __len__(self):
        return self.size
