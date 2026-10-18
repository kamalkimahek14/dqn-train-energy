# Copyright (c) 2026 Kamal Mehta. MIT License.
"""The coupled train: the coupler law, Newton's third law between groups, and steady pulling."""
import numpy as np
import pytest

from ecodrive import coupled
from ecodrive.baseline import baseline_action
from ecodrive.coupled import (F_EVENT, GAP, GROUP_LENGTH, K_COUPLER, N_FEATURES, N_GROUPS, WINDOW,
                              CoupledTrainEnv, coupler_force)
from ecodrive.env import TrainEnv


def test_deadband_transmits_nothing_inside_the_slack():
    d = np.array([-0.99, -0.5, 0.0, 0.5, 0.99]) * GAP
    np.testing.assert_array_equal(coupler_force(d, np.full(5, 0.3)), 0.0)    # even while closing fast


def test_spring_beyond_the_slack_and_signs():
    f = coupler_force(np.array([GAP + 0.01, -GAP - 0.01]), np.zeros(2))
    np.testing.assert_allclose(f, [K_COUPLER * 0.01, -K_COUPLER * 0.01])     # + draft, - buff
    # A stretched coupler that separates fast is pulled back to zero force, never into buff.
    assert coupler_force(np.array([GAP + 1e-4]), np.array([-1.0]))[0] == 0.0


def test_coupler_forces_balance(monkeypatch):
    """No resistance, no grade, no traction: the couplers only move momentum between groups."""
    monkeypatch.setattr(coupled, "DAVIS", (0.0, 0.0, 0.0))
    env = CoupledTrainEnv()
    env.reset(seed=0)
    env.track.grade[:] = 0.0
    env.vg[:] = 10.0
    env.s[1:] -= 0.2                                  # stretch the first coupler well past its slack
    momentum = env.vg.sum()
    env.step(0)
    assert env.step_draft > 1e5                       # the couplers did work...
    assert env.vg.sum() == pytest.approx(momentum, rel=1e-9)   # ...and the momentum is unchanged


def test_steady_pull_splits_by_the_mass_behind():
    """Settled, coupler j carries the traction share of the groups behind it: 3/4, 2/4, 1/4."""
    env = CoupledTrainEnv()
    env.reset(seed=0)
    env.track.grade[:] = 0.0
    for _ in range(6):
        env.step(3)
    behind = np.arange(N_GROUPS - 1, 0, -1)
    np.testing.assert_allclose(env.force / env.force[0], behind / behind[0], rtol=0.05)


def test_same_spaces_and_a_similar_trip_to_the_point_mass():
    point, lumped = TrainEnv(), CoupledTrainEnv()
    assert lumped.observation_space == point.observation_space and lumped.action_space == point.action_space
    out = []
    for env in (point, lumped):
        env.reset(seed=0)
        done = False
        while not done:
            _, _, done, _, info = env.step(baseline_action(env))
        out.append(info)
    assert out[1]["arrived"] and out[1]["t"] == pytest.approx(out[0]["t"], rel=0.05)
    assert out[1]["fuel"] == pytest.approx(out[0]["fuel"], rel=0.05)


def test_events_and_features():
    env = CoupledTrainEnv()
    env.reset(seed=0)
    assert env.slack_features().shape == (WINDOW, N_FEATURES)
    env.step(8)                                       # full notch from a standstill snatches the slack
    assert env.step_draft > F_EVENT and env.draft_events == 1
    feats = env.slack_features()
    assert feats.dtype == np.float32 and np.isfinite(feats).all()
    assert feats[-1, 5] == 1.0                        # newest row holds the last control level
    np.testing.assert_allclose(env.s[:-1] - env.s[1:], GROUP_LENGTH, atol=1.0)  # the train stays in one piece
