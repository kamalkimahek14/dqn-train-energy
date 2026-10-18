# Copyright (c) 2026 Kamal Mehta. MIT License.
import numpy as np

from ecodrive.replay import ReplayBuffer


def fill(buf, n, start=0):
    for i in range(start, start + n):
        buf.add(np.full(3, i), i % 2, float(i), np.full(3, i + 1), i % 5 == 0)


def test_len_grows_then_caps_at_capacity():
    buf = ReplayBuffer(capacity=10, obs_dim=3)
    fill(buf, 4)
    assert len(buf) == 4
    fill(buf, 20, start=4)
    assert len(buf) == 10


def test_ring_overwrites_the_oldest():
    buf = ReplayBuffer(capacity=10, obs_dim=3)
    fill(buf, 13)                                     # items 0..12; 0, 1, 2 are overwritten
    assert sorted(buf.rewards.tolist()) == list(range(3, 13))


def test_sample_shapes_dtypes_and_consistency():
    buf = ReplayBuffer(capacity=100, obs_dim=3, seed=0)
    fill(buf, 50)
    b = buf.sample(32)
    assert b["obs"].shape == (32, 3) and b["next_obs"].shape == (32, 3)
    assert b["actions"].dtype == np.int64 and b["rewards"].dtype == np.float32
    np.testing.assert_array_equal(b["obs"][:, 0], b["rewards"])          # each row stays one transition
    np.testing.assert_array_equal(b["next_obs"][:, 0], b["rewards"] + 1)
    np.testing.assert_array_equal(b["dones"], (b["rewards"] % 5 == 0).astype(np.float32))


def test_sample_only_from_filled_slots_and_is_seeded():
    a, b = ReplayBuffer(1000, 3, seed=7), ReplayBuffer(1000, 3, seed=7)
    fill(a, 5)
    fill(b, 5)
    s = a.sample(200)
    assert s["rewards"].max() < 5                     # empty slots are never drawn
    np.testing.assert_array_equal(s["rewards"], b.sample(200)["rewards"])
