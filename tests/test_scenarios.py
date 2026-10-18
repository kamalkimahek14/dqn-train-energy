# Copyright (c) 2026 Kamal Mehta. MIT License.
"""The named slack scenarios: each one builds and runs, and every train integrates stably."""
import numpy as np
import pytest

from ecodrive.coupled import (C_COUPLER, GROUP_MASS, H, K_COUPLER, N_FEATURES, N_GROUPS, WINDOW,
                              CoupledTrainEnv)
from ecodrive.scenarios import POWER, SCENARIOS, SECTIONS, TRAINS


def test_the_scenario_list():
    assert 10 <= len(SCENARIOS) <= 20 and len({s.name for s in SCENARIOS}) == len(SCENARIOS)
    assert {(s.train, s.power, s.section) for s in SCENARIOS} == {(t, p, s) for t in TRAINS for p in POWER
                                                                   for s in SECTIONS}
    assert all(0 < s.grade <= 0.015 for s in SCENARIOS)


@pytest.mark.parametrize("sc", SCENARIOS, ids=lambda s: s.name)
def test_each_scenario_builds_and_runs_one_step(sc):
    env = sc.make_env()
    obs, _ = env.reset(seed=0)
    assert env.track.length == 6000 and env.track.grade_at(0.0) in (0.0, sc.grade)
    obs, reward, done, _, info = env.step(8)
    assert obs.shape == (11,) and np.isfinite(reward) and not done and info["x"] > 0
    assert env.slack_features().shape == (WINDOW, N_FEATURES) and np.isfinite(env.slack_features()).all()


def test_fixed_section_ignores_the_seed():
    env = SCENARIOS[0].make_env()
    env.reset(seed=0)
    first = env.track
    env.reset(seed=123)
    assert env.track is first


def test_mid_power_splits_the_pull():
    """Settled, with 2/3 of the traction at the head and 1/3 on group 2 the couplers carry 5/12, 2/12 and 3/12
    of it, against 9/12, 6/12 and 3/12 with all of it at the head."""
    forces = []
    for env in (CoupledTrainEnv(head_units=2 / 3, mid_units=1 / 3), CoupledTrainEnv()):
        env.reset(seed=0)
        env.track.grade[:] = 0.0
        for _ in range(6):
            env.step(3)
        forces.append(env.force)
    np.testing.assert_allclose(forces[0] / forces[1][0], [5 / 9, 2 / 9, 1 / 3], rtol=0.05)


@pytest.mark.parametrize("kw", [dict(group_mass=GROUP_MASS, k=K_COUPLER, c=C_COUPLER)] + list(TRAINS.values()),
                         ids=["default"] + list(TRAINS))
def test_the_coupler_integration_is_stable(kw):
    """The highest coupler mode of the 4-group chain, stepped like _integrate, must not grow."""
    lam = 2 + 2 * np.cos(np.pi / N_GROUPS)            # largest eigenvalue of the chain's stiffness pattern
    w2, z = lam * kw["k"] / kw["group_mass"], lam * kw["c"] / kw["group_mass"]
    step = np.array([[1 - w2 * H * H / 2, H - z * H * H / 2], [-w2 * H, 1 - z * H]])  # (stretch, rate) per H
    assert np.abs(np.linalg.eigvals(step)).max() < 0.98
