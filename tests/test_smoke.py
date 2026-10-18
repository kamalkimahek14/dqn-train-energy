# Copyright (c) 2026 Kamal Mehta. MIT License.
"""A few hundred real training steps per backend and task: everything runs end to end on CPU."""
import math
import os
import time

import pytest

from ecodrive.agents import make_agent
from ecodrive.config import get_config, make_env
from ecodrive.loop import train


@pytest.mark.parametrize("env_id", ["train", "CartPole-v1"])
@pytest.mark.parametrize("backend", ["torch", "tf"])
def test_short_training_run(backend, env_id, tmp_path):
    pytest.importorskip("torch" if backend == "torch" else "tensorflow")
    cfg = get_config(env_id, total_steps=400, learning_starts=100, buffer_size=1_000,
                     target_update_every=100, eval_every=400, eval_episodes=1)
    env = make_env(env_id)
    agent = make_agent(backend, env.observation_space.shape[0], int(env.action_space.n), cfg)
    start = time.perf_counter()
    result = train(agent, env, cfg, out_dir=str(tmp_path), log=lambda _: None)
    assert time.perf_counter() - start < 60
    assert math.isfinite(result["best_val_return"]) and result["episodes"]
    assert any(name.startswith("best") for name in os.listdir(tmp_path))
