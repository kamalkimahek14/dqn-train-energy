# Copyright (c) 2026 Kamal Mehta. MIT License.
"""The slack predictor: inputs, splits, metrics, both backends, and the safety filter."""
import numpy as np
import pytest

from ecodrive.baseline import baseline_action
from ecodrive.coupled import N_FEATURES, WINDOW, CoupledTrainEnv
from ecodrive.env import N_ACTIONS
from ecodrive.predictor import make_net
from ecodrive.slack import (N_INPUTS, SlackFilter, SlackPredictor, binary_metrics, choose, collect,
                            epsilon_policy, fit, inputs, pick_threshold, roc_auc, to_xy)
from slack_predictor import split_seeds


def test_splits_share_no_track():
    s = {k: set(v) for k, v in split_seeds().items()}
    assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"])
    assert len(s["train"]) == 200 and len(s["val"]) == len(s["test"]) == 50


def test_inputs_layout():
    hist = np.random.default_rng(0).normal(size=(WINDOW, N_FEATURES)).astype(np.float32)
    hist[-1, 5] = 0.5                                 # last control level: notch 4
    x = inputs(hist, np.arange(N_ACTIONS))
    assert x.shape == (N_ACTIONS, N_INPUTS) and x.dtype == np.float32
    np.testing.assert_array_equal(x[:, :hist.size], np.tile(hist.reshape(-1), (N_ACTIONS, 1)))
    np.testing.assert_array_equal(x[:, hist.size:hist.size + N_ACTIONS], np.eye(N_ACTIONS))
    assert x[8, -2] == 1.0 and x[8, -1] == pytest.approx(0.5)     # notch 8: level 1, change +0.5
    assert x[11, -2] == -1.0 and x[11, -1] == pytest.approx(-1.5)  # brake 3: level -1, change -1.5


def test_choose_respects_the_threshold_and_never_comes_back_empty():
    rng = np.random.default_rng(0)
    for _ in range(500):
        q, risk, thr = rng.normal(size=12), rng.random(12), rng.random()
        a = choose(q, risk, thr)
        assert 0 <= a < 12
        if (risk < thr).any():
            assert risk[a] < thr and q[a] == q[risk < thr].max()
        else:
            assert a == int(np.argmin(risk))           # all risky: the least risky one
    assert choose(np.arange(12.0), np.zeros(12), 0.5) == 11      # nothing risky: plain argmax Q


def test_metrics():
    y = np.array([0, 0, 1, 1])
    assert roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert roc_auc(y, np.full(4, 0.5)) == 0.5
    m = binary_metrics(y, np.array([0.1, 0.6, 0.4, 0.9]), 0.5)
    assert m["confusion"] == {"tp": 1, "fp": 1, "fn": 1, "tn": 1} and m["f1"] == pytest.approx(0.5)
    score = np.linspace(0, 1, 100)
    yy = (np.random.default_rng(1).random(100) < score).astype(int)
    thr = pick_threshold(yy, score, 0.9)
    assert binary_metrics(yy, score, thr)["recall"] >= 0.9
    assert binary_metrics(yy, score, thr + 1e-9)["recall"] < 0.9  # the highest such threshold


@pytest.fixture(params=["torch", "tf"])
def net(request):
    pytest.importorskip("torch" if request.param == "torch" else "tensorflow")
    return make_net(request.param, N_INPUTS, seed=0)


def test_forward_shapes_and_weighted_loss(net):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, N_INPUTS)).astype(np.float32)
    y = (rng.random(16) < 0.3).astype(np.float32)
    p = net.predict_proba(x)
    assert p.shape == (16,) and ((p > 0) & (p < 1)).all()
    ref = -(4.0 * y * np.log(p) + (1 - y) * np.log(1 - p)).mean()  # positives weighted 4x
    assert net.eval_loss(x, y, 4.0) == pytest.approx(ref, rel=1e-4)
    before = net.eval_loss(x, y, 4.0)
    assert net.update(x, y, 4.0) == pytest.approx(before, rel=1e-5)  # update() reports the pre-step loss
    assert net.eval_loss(x, y, 4.0) < before


class FixedQ:
    def q_values(self, obs):
        return np.arange(12.0)[None]                  # always prefers action 11, then 10, ...


class RiskyAbove:
    """Flags every action above `limit` as certain to cause an event."""

    def __init__(self, limit):
        self.limit = limit

    def risk(self, x):
        return (x[:, WINDOW * N_FEATURES:WINDOW * N_FEATURES + N_ACTIONS].argmax(1) > self.limit).astype(float)


def test_filter_wraps_the_agent():
    env = CoupledTrainEnv()
    obs, _ = env.reset(seed=0)
    filt = SlackFilter(FixedQ(), RiskyAbove(6), env, threshold=0.5)
    assert filt.greedy(obs) == 6 and filt.overrides == 1
    assert SlackFilter(FixedQ(), RiskyAbove(99), env, 0.5).greedy(obs) == 11


@pytest.mark.parametrize("backend", ["torch", "tf"])
def test_smoke_dataset_fit_and_filtered_trip(backend):
    pytest.importorskip("torch" if backend == "torch" else "tensorflow")
    env = CoupledTrainEnv()
    rand = epsilon_policy(lambda obs: baseline_action(env), 1.0, np.random.default_rng(0))
    data = [collect(env, rand, seed) for seed in (0, 1)]
    x, y = to_xy({k: np.concatenate([d[k] for d in data]) for k in data[0]})
    assert x.shape == (len(y), N_INPUTS) and 0 < y.sum() < len(y)
    pred = fit(make_net(backend, N_INPUTS), x, y, x, y, epochs=2, log=lambda _: None)
    assert isinstance(pred, SlackPredictor) and len(pred.history) == 2

    class Baseline:  # a stand-in DQN: Q is highest for the baseline's choice
        def q_values(self, obs):
            q = np.zeros((1, 12))
            q[0, baseline_action(env)] = 1.0
            return q

    filt = SlackFilter(Baseline(), pred, env, threshold=0.5)
    obs, _ = env.reset(seed=2)
    done = False
    while not done:
        obs, _, done, _, info = env.step(filt.greedy(obs))
    assert info["t"] > 0 and np.isfinite(info["fuel"])
