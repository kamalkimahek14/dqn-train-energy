# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Backend switch for the slack predictor. Imports are lazy, as in ecodrive.agents."""

HIDDEN = (64, 64, 32)  # three hidden Dense layers plus the output: a 4-layer network


def make_net(backend: str, n_in: int, seed: int = 0, lr: float = 1e-3):
    if backend == "torch":
        from .torch_slack import TorchSlackNet
        return TorchSlackNet(n_in, HIDDEN, seed, lr)
    if backend == "tf":
        from .tf_slack import TFSlackNet
        return TFSlackNet(n_in, HIDDEN, seed, lr)
    raise ValueError(f"unknown backend {backend!r}, expected 'torch' or 'tf'")
