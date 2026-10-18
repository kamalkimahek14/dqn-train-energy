# Copyright (c) 2026 Kamal Mehta. MIT License.
"""The training loop's schedule: when it updates, when it syncs the target, how epsilon decays."""
import pytest

from ecodrive.config import epsilon, get_config, make_env
from ecodrive.loop import train


class CountingAgent:
    """Stands in for a DQN agent and only counts calls."""

    def __init__(self, batch_size):
        self.batch_size, self.updates, self.syncs = batch_size, 0, 0

    def greedy(self, obs):
        return 0

    def update(self, batch):
        assert batch["obs"].shape == (self.batch_size, 4)
        self.updates += 1
        return 0.0

    def sync_target(self):
        self.syncs += 1

    def save(self, path):
        pass


def test_updates_start_after_warmup_and_target_syncs_on_schedule():
    cfg = get_config("CartPole-v1", total_steps=1_000, learning_starts=200, target_update_every=300,
                     eval_every=1_000, eval_episodes=1)
    agent = CountingAgent(cfg.batch_size)
    result = train(agent, make_env(cfg.env_id), cfg, log=lambda _: None)
    assert agent.updates == 1_000 - 200 + 1          # one update per step from step 200 on
    assert agent.syncs == 3                          # steps 300, 600 and 900
    assert len(result["evals"]) == 1 and result["episodes"]


def test_epsilon_decays_linearly_then_holds():
    cfg = get_config("train")
    assert epsilon(cfg, 0) == cfg.eps_start
    assert epsilon(cfg, cfg.eps_decay_steps // 2) == pytest.approx((cfg.eps_start + cfg.eps_end) / 2)
    assert epsilon(cfg, cfg.eps_decay_steps) == pytest.approx(cfg.eps_end)
    assert epsilon(cfg, 10 * cfg.eps_decay_steps) == pytest.approx(cfg.eps_end)
