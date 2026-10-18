# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Slack predictor in PyTorch: a feed-forward net over the flattened history window. Same structure,
line for line, as tf_slack.py."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def build_net(n_in: int, hidden: tuple) -> nn.Module:
    """MLP: inputs -> one logit for 'a slack event happens in the next decision'."""
    layers, size = [], n_in
    for h in hidden:
        layers += [nn.Linear(size, h), nn.ReLU()]
        size = h
    layers.append(nn.Linear(size, 1))
    net = nn.Sequential(*layers)
    for layer in net:  # Keras Dense defaults, as in torch_dqn.py
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
    return net


class TorchSlackNet:
    def __init__(self, n_in: int, hidden: tuple, seed: int, lr: float):
        torch.manual_seed(seed)
        self.net = build_net(n_in, hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr, eps=1e-7)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return torch.sigmoid(self.net(torch.as_tensor(x, dtype=torch.float32))).squeeze(1).numpy()

    def loss(self, x, y, pos_weight: float) -> torch.Tensor:
        logits = self.net(torch.as_tensor(x, dtype=torch.float32)).squeeze(1)
        # Weighted cross-entropy: each positive counts pos_weight times, against the class imbalance.
        return F.binary_cross_entropy_with_logits(logits, torch.as_tensor(y, dtype=torch.float32),
                                                  pos_weight=torch.tensor(pos_weight))

    def eval_loss(self, x, y, pos_weight: float) -> float:
        with torch.no_grad():
            return self.loss(x, y, pos_weight).item()

    def update(self, x, y, pos_weight: float) -> float:
        loss = self.loss(x, y, pos_weight)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

    def get_weights(self) -> list:
        return [p.detach().numpy().copy() for p in self.net.state_dict().values()]

    def set_weights(self, weights: list):
        self.net.load_state_dict({k: torch.as_tensor(w) for k, w in zip(self.net.state_dict(), weights)})

    def save(self, path: str):
        torch.save(self.net.state_dict(), path + ".pt")

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path + ".pt"))
