# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Double DQN in PyTorch. Same structure, line for line, as tf_dqn.py."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..config import DQNConfig


def build_q_net(obs_dim: int, n_actions: int, hidden: tuple) -> nn.Module:
    """MLP: observation -> one Q-value per action."""
    layers, size = [], obs_dim
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, n_actions))
    return nn.Sequential(*layers)


class TorchDQN:
    def __init__(self, obs_dim: int, n_actions: int, cfg: DQNConfig):
        torch.manual_seed(cfg.seed)
        self.cfg = cfg
        self.q = build_q_net(obs_dim, n_actions, cfg.hidden)          # online network, trained
        self.q_target = build_q_net(obs_dim, n_actions, cfg.hidden)   # target network, copied
        self.sync_target()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return self.q(torch.as_tensor(obs, dtype=torch.float32)).numpy()

    def greedy(self, obs: np.ndarray) -> int:
        return int(np.argmax(self.q_values(obs[None])[0]))

    def td_target(self, rewards, next_obs, dones) -> torch.Tensor:
        """Double DQN: the online net picks the next action, the target net scores it."""
        with torch.no_grad():
            next_a = self.q(next_obs).argmax(dim=1, keepdim=True)
            next_q = self.q_target(next_obs).gather(1, next_a).squeeze(1)
            return rewards + self.cfg.gamma * (1.0 - dones) * next_q

    def loss(self, batch: dict) -> torch.Tensor:
        b = {k: torch.as_tensor(v) for k, v in batch.items()}
        target = self.td_target(b["rewards"], b["next_obs"], b["dones"])
        q = self.q(b["obs"]).gather(1, b["actions"].unsqueeze(1)).squeeze(1)  # Q(s, a) of the taken action
        return F.smooth_l1_loss(q, target)  # Huber loss with delta = 1

    def update(self, batch: dict) -> float:
        """One gradient step on a replay batch; returns the loss."""
        loss = self.loss(batch)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.grad_clip)
        self.opt.step()
        return loss.item()

    def sync_target(self):
        self.q_target.load_state_dict(self.q.state_dict())

    def save(self, path: str):
        torch.save(self.q.state_dict(), path + ".pt")

    def load(self, path: str):
        self.q.load_state_dict(torch.load(path + ".pt"))
        self.sync_target()
