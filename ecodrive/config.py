# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Hyperparameters shared by both backends, plus the environment factory."""
from dataclasses import dataclass, replace

import gymnasium as gym

from .env import TrainEnv


@dataclass(frozen=True)
class DQNConfig:
    env_id: str = "train"            # "train" (this repo's TrainEnv) or "CartPole-v1"
    seed: int = 0
    total_steps: int = 150_000       # environment steps
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 5e-4
    hidden: tuple = (128, 128)
    learning_starts: int = 2_000     # fill the buffer with random play before the first update
    train_every: int = 1             # gradient step every N environment steps
    target_update_every: int = 1_000 # hard copy online -> target every N environment steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 60_000    # linear decay from eps_start to eps_end
    grad_clip: float = 10.0          # global gradient-norm clip
    eval_every: int = 10_000         # greedy check on validation episodes; the best weights are kept
    eval_episodes: int = 10


PRESETS = {
    "train": DQNConfig(train_every=2),  # one gradient step per two decisions keeps CPU time modest
    "CartPole-v1": DQNConfig(env_id="CartPole-v1", total_steps=50_000, buffer_size=50_000, lr=1e-3,
                             learning_starts=1_000, target_update_every=500, eps_decay_steps=15_000,
                             eval_every=5_000),
}

# Seed ranges never overlap: training draws tracks from the env's own stream,
# model selection uses VALIDATION_SEED + i, and evaluate.py reports on TEST_SEED + i.
VALIDATION_SEED = 1_000
TEST_SEED = 10_000


def get_config(env_id: str, **overrides) -> DQNConfig:
    return replace(PRESETS[env_id], **overrides)


def make_env(env_id: str) -> gym.Env:
    return TrainEnv() if env_id == "train" else gym.make(env_id)


def epsilon(cfg: DQNConfig, step: int) -> float:
    """Linearly decayed exploration rate."""
    frac = min(step / cfg.eps_decay_steps, 1.0)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)
