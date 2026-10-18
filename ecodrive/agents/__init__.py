# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Backend switch. Imports are lazy so a torch-only run never loads TensorFlow, and the reverse."""


def make_agent(backend: str, obs_dim: int, n_actions: int, cfg):
    if backend == "torch":
        from .torch_dqn import TorchDQN
        return TorchDQN(obs_dim, n_actions, cfg)
    if backend == "tf":
        from .tf_dqn import TFDQN
        return TFDQN(obs_dim, n_actions, cfg)
    raise ValueError(f"unknown backend {backend!r}, expected 'torch' or 'tf'")


def limit_threads(backend: str, n: int):
    """Cap CPU threads. Must run before the backend does any math (TensorFlow refuses afterwards)."""
    if backend == "torch":
        import torch
        torch.set_num_threads(n)
    else:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(n)
        tf.config.threading.set_inter_op_parallelism_threads(n)
