# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Backend checks: the DQN loss against a numpy reference, one update step, and the target network."""
import numpy as np
import pytest

from ecodrive.agents import make_agent
from ecodrive.config import DQNConfig

OBS_DIM, N_ACTIONS, BATCH = 4, 3, 32
CFG = DQNConfig(hidden=(32, 32), lr=1e-3, gamma=0.9)


@pytest.fixture(params=["torch", "tf"])
def agent(request):
    pytest.importorskip("torch" if request.param == "torch" else "tensorflow")
    return make_agent(request.param, OBS_DIM, N_ACTIONS, CFG)


@pytest.fixture
def batch():
    rng = np.random.default_rng(0)
    return {"obs": rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
            "actions": rng.integers(0, N_ACTIONS, BATCH).astype(np.int64),
            "rewards": rng.normal(scale=2.0, size=BATCH).astype(np.float32),
            "next_obs": rng.normal(size=(BATCH, OBS_DIM)).astype(np.float32),
            "dones": (rng.random(BATCH) < 0.25).astype(np.float32)}


def weights(net):
    if hasattr(net, "state_dict"):  # torch
        return [p.detach().numpy().copy() for p in net.state_dict().values()]
    return [w.copy() for w in net.get_weights()]


def target_q_values(agent, obs):
    if hasattr(agent.q_target, "state_dict"):
        import torch
        with torch.no_grad():
            return agent.q_target(torch.as_tensor(obs)).numpy()
    return agent.q_target(obs).numpy()


def as_float(loss):
    return float(loss.detach()) if hasattr(loss, "detach") else float(loss)  # torch tensor or tf tensor


def same(a, b):
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def test_target_moves_only_on_sync(agent, batch):
    assert same(weights(agent.q), weights(agent.q_target))       # starts as a copy
    agent.update(batch)
    assert not same(weights(agent.q), weights(agent.q_target))   # the online net learned...
    frozen = weights(agent.q_target)
    agent.update(batch)
    assert same(frozen, weights(agent.q_target))                 # ...the target did not
    agent.sync_target()
    assert same(weights(agent.q), weights(agent.q_target))


def test_loss_matches_numpy_double_dqn_huber(agent, batch):
    agent.update(batch)  # make online and target differ, so Double DQN is not plain DQN
    rows = np.arange(BATCH)
    best_next = agent.q_values(batch["next_obs"]).argmax(axis=1)                 # online picks
    next_q = target_q_values(agent, batch["next_obs"])[rows, best_next]          # target scores
    y = batch["rewards"] + CFG.gamma * (1 - batch["dones"]) * next_q
    err = agent.q_values(batch["obs"])[rows, batch["actions"]] - y
    huber = np.where(np.abs(err) <= 1.0, 0.5 * err ** 2, np.abs(err) - 0.5).mean()
    assert as_float(agent.loss(batch)) == pytest.approx(huber, rel=1e-5)


def test_one_update_lowers_the_loss_on_its_batch(agent, batch):
    before = as_float(agent.loss(batch))
    after_update = agent.update(batch)
    assert after_update == pytest.approx(before, rel=1e-5)  # update() reports the pre-step loss
    assert as_float(agent.loss(batch)) < before


def test_save_and_load_round_trip(agent, batch, tmp_path):
    agent.update(batch)
    agent.save(str(tmp_path / "w"))
    expected = agent.q_values(batch["obs"])
    fresh = make_agent("torch" if hasattr(agent.q, "state_dict") else "tf", OBS_DIM, N_ACTIONS,
                       DQNConfig(hidden=(32, 32), seed=1))
    fresh.load(str(tmp_path / "w"))
    np.testing.assert_allclose(fresh.q_values(batch["obs"]), expected, rtol=1e-6)
    assert same(weights(fresh.q), weights(fresh.q_target))
