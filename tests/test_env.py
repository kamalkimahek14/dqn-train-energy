# Copyright (c) 2026 Kamal Mehta. MIT License.
import numpy as np
import pytest

from ecodrive.baseline import baseline_action
from ecodrive.env import DT, N_ACTIONS, TrainEnv, action_to_control


@pytest.fixture
def env():
    return TrainEnv()


def test_spaces_and_observation(env):
    obs, _ = env.reset(seed=0)
    assert env.action_space.n == N_ACTIONS == 12
    assert obs.shape == (10,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)


def test_action_mapping():
    assert action_to_control(0) == (0.0, 0.0)      # coast
    assert action_to_control(8) == (1.0, 0.0)      # full throttle
    assert action_to_control(11) == (0.0, 1.0)     # full brake
    assert action_to_control(9)[1] == pytest.approx(1 / 3)


def test_reset_is_deterministic(env):
    a, _ = env.reset(seed=3)
    limits = env.track.limit.copy()
    b, _ = env.reset(seed=3)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(limits, env.track.limit)
    env.reset(seed=4)
    assert not np.array_equal(limits, env.track.limit)


def test_physics_invariants_under_random_actions(env):
    rng = np.random.default_rng(0)
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            x, fuel, t = env.x, env.fuel, env.t
            obs, reward, done, _, info = env.step(int(rng.integers(N_ACTIONS)))
            assert env.v >= 0.0                      # a train does not roll backwards here
            assert env.x >= x                        # position never decreases
            assert env.fuel >= fuel                  # fuel is only ever spent
            assert 0.0 < env.t - t <= DT + 1e-9
            assert np.isfinite(reward) and env.observation_space.contains(obs)


def test_coasting_and_braking_use_no_fuel(env):
    env.reset(seed=0)
    for _ in range(5):
        env.step(8)                                  # build up speed
    for action in (0, 9, 11):
        fuel = env.fuel
        env.step(action)
        assert env.fuel == fuel


def test_more_throttle_burns_more_fuel_and_accelerates_more():
    results = []
    for notch in (4, 8):
        env = TrainEnv()
        env.reset(seed=0)
        env.step(notch)
        results.append((env.fuel, env.v))
    (fuel4, v4), (fuel8, v8) = results
    assert 0 < fuel4 < fuel8 and v4 < v8


def test_coasting_on_flat_track_slows_down(env):
    env.reset(seed=0)
    env.track.grade[:] = 0.0                         # flatten the track: only resistance acts
    env.v = 20.0
    env.step(0)
    assert env.v < 20.0


def test_baseline_arrives_on_time_without_violations(env):
    for seed in range(5):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, done, _, info = env.step(baseline_action(env))
        assert info["arrived"] and info["on_time"] and info["violations"] == 0
        assert env.x >= env.track.length


def test_standing_still_hits_the_hard_time_limit(env):
    env.reset(seed=0)
    done, steps = False, 0
    while not done:
        _, reward, done, _, info = env.step(0)       # coast from standstill: the train never moves
        steps += 1
    assert not info["arrived"] and env.t >= env.t_hard
    assert reward < -10.0                            # the whole track is left: big penalty
