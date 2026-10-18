# Copyright (c) 2026 Kamal Mehta. MIT License.
"""A toy train on a 1-D track: drive from A to B on time, under the speed limit, with little fuel.

Everything here is textbook point-mass physics:
    m * dv/dt = F_traction - F_brake - R(v) - m * g * grade
    R(v) = A + B*v + C*v^2                  (Davis rolling + air resistance)
    F_traction = notch/8 * min(F_max, P_max / v)   (force limit at low speed, power limit above)
    fuel = SFC * traction energy             (fuel is proportional to the tractive power used)

Each episode draws a new random track (grade profile and speed-limit profile) from the seed,
so the agent has to read the track ahead from its observation instead of memorising one route.
"""
from __future__ import annotations

import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# ---- Train and physics constants (SI units) -------------------------------------------------
MASS = 1.0e6            # kg, a 1,000 t train
G = 9.81                # m/s^2
F_MAX = 300e3           # N, tractive-force limit (adhesion) at low speed
P_MAX = 3.0e6           # W, traction power limit at the wheel
B_MAX = 400e3           # N, braking force at the strongest brake level
DAVIS = (1.0e4, 300.0, 8.0)  # A [N], B [N/(m/s)], C [N/(m/s)^2]
SFC = 0.25              # litres of fuel per kWh of traction energy

N_NOTCHES = 8           # throttle notches 0..8 (0 = coast)
N_BRAKES = 3            # brake levels 1..3
N_ACTIONS = N_NOTCHES + 1 + N_BRAKES  # 12 discrete actions

# ---- Episode settings ------------------------------------------------------------------------
TRACK_LENGTH = 10_000.0  # m
DT = 10.0                # s per agent decision
SUBSTEPS = 10            # physics is integrated at DT / SUBSTEPS = 1 s
LIMITS_KMH = (60, 80, 100)
V_NORM = 30.0            # m/s, scale for speeds in the observation
EARLY, LATE, HARD = 1.05, 1.40, 1.70  # time window and hard stop, as multiples of the free-flow time

# ---- Reward weights --------------------------------------------------------------------------
W_FUEL = 0.1     # per litre
W_SPEED = 0.5    # per m/s above the limit (worst excess within the step)
W_JERK = 0.05    # per unit change of the control level u in [-1, 1]
W_TIME = 5.0     # per minute outside the arrival window
W_STALL = 10.0   # times the fraction of track left when the hard time limit hits


def action_to_control(action: int) -> tuple[float, float]:
    """Map a discrete action to (throttle fraction, brake fraction). 0..8 = notches, 9..11 = brakes."""
    if action <= N_NOTCHES:
        return action / N_NOTCHES, 0.0
    return 0.0, (action - N_NOTCHES) / N_BRAKES


def control_level(action: int) -> float:
    """One number per action: +1 = full throttle, 0 = coast, -1 = full brake. Used for the harshness penalty."""
    throttle, brake = action_to_control(action)
    return throttle - brake


def tractive_force_available(v: float) -> float:
    """Force at notch 8: constant up to the corner speed, then constant power (F = P / v)."""
    return min(F_MAX, P_MAX / max(v, 1.0))


def resistance(v: float) -> float:
    a, b, c = DAVIS
    return a + b * v + c * v * v


class Track:
    """Piecewise-constant grade and speed-limit profiles, generated from a numpy Generator."""

    def __init__(self, rng: np.random.Generator, length: float = TRACK_LENGTH):
        self.length = length
        self.grade_x, self.grade = self._segments(rng, 600, 1500, lambda: rng.uniform(-0.006, 0.006))
        limits = [k / 3.6 for k in LIMITS_KMH]
        last = [None]

        def next_limit():  # consecutive limits always differ, so every boundary is a real change
            v = rng.choice([l for l in limits if l != last[0]])
            last[0] = v
            return float(v)

        self.limit_x, self.limit = self._segments(rng, 1500, 3500, next_limit)
        # Free-flow time: driving every segment exactly at its limit. A lower bound on the trip time.
        ends = np.append(self.limit_x[1:], length)
        self.free_flow_time = float(np.sum((ends - self.limit_x) / self.limit))

    def _segments(self, rng, lo, hi, draw):
        starts, values, x = [], [], 0.0
        while x < self.length:
            starts.append(x)
            values.append(draw())
            x += rng.uniform(lo, hi)
        return np.array(starts), np.array(values)

    def grade_at(self, x: float) -> float:
        return float(self.grade[np.searchsorted(self.grade_x, x, side="right") - 1])

    def limit_at(self, x: float) -> float:
        return float(self.limit[np.searchsorted(self.limit_x, x, side="right") - 1])

    def next_limit_change(self, x: float) -> tuple[float, float]:
        """(position, new limit) of the next speed-limit change ahead, or (inf, current limit)."""
        i = np.searchsorted(self.limit_x, x, side="right")
        if i >= len(self.limit_x):
            return math.inf, self.limit_at(x)
        return float(self.limit_x[i]), float(self.limit[i])


class TrainEnv(gym.Env):
    """Gymnasium environment. Observation: 10 floats. Action: Discrete(12)."""

    metadata = {"render_modes": []}

    def __init__(self):
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(10,), dtype=np.float32)
        self.track: Track | None = None

    # -- Gymnasium API --------------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)          # seeds self.np_random
        self.track = Track(self.np_random)
        self.x, self.v, self.t = 0.0, 0.0, 0.0
        self.fuel = 0.0                   # litres used so far
        self.u_prev = 0.0                 # previous control level, for the harshness penalty
        self.violations = 0               # decisions during which the train exceeded the limit
        self.t_early = EARLY * self.track.free_flow_time
        self.t_late = LATE * self.track.free_flow_time
        self.t_hard = HARD * self.track.free_flow_time
        return self._obs(), self._info()

    def step(self, action: int):
        assert self.action_space.contains(action), action
        throttle, brake = action_to_control(int(action))
        u = throttle - brake
        dt = DT / SUBSTEPS
        fuel_step, over = 0.0, 0.0
        arrived = False

        for _ in range(SUBSTEPS):
            f_trac = throttle * tractive_force_available(self.v)
            f_net = f_trac - brake * B_MAX - resistance(self.v) - MASS * G * self.track.grade_at(self.x)
            v_new = max(0.0, self.v + f_net / MASS * dt)  # semi-implicit Euler; a stopped train does not roll back
            v_avg = 0.5 * (self.v + v_new)
            fuel_step += SFC * f_trac * v_avg * dt / 3.6e6  # J -> kWh -> litres
            self.x += v_avg * dt
            self.v = v_new
            self.t += dt
            over = max(over, self.v - self.track.limit_at(min(self.x, self.track.length)))
            if self.x >= self.track.length:
                arrived = True
                break

        self.fuel += fuel_step
        if over > 0.0:
            self.violations += 1

        reward = -W_FUEL * fuel_step - W_SPEED * max(over, 0.0) - W_JERK * abs(u - self.u_prev)
        self.u_prev = u

        # The deadline is part of the task and time is in the observation, so both endings are
        # true terminal states (terminated=True), not time-limit truncations.
        terminated = False
        if arrived:
            terminated = True
            minutes_off = (max(0.0, self.t - self.t_late) + max(0.0, self.t_early - self.t)) / 60.0
            reward -= W_TIME * minutes_off
        elif self.t >= self.t_hard:
            terminated = True
            remaining = (self.track.length - self.x) / self.track.length
            reward -= W_STALL * remaining + W_TIME * (self.t - self.t_late) / 60.0

        return self._obs(), float(reward), terminated, False, self._info(arrived)

    # -- Helpers --------------------------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        tr = self.track
        x = min(self.x, tr.length)
        change_x, next_limit = tr.next_limit_change(x)
        ahead = np.mean([tr.grade_at(min(x + d, tr.length - 1e-6)) for d in (200, 400, 600, 800, 1000)])
        remaining = tr.length - x
        time_left = max(self.t_late - self.t, DT)
        obs = np.array([
            self.v / V_NORM,
            tr.limit_at(x) / V_NORM,
            next_limit / V_NORM,
            min((change_x - x) / 2000.0, 1.0),
            tr.grade_at(x) * 100.0,
            ahead * 100.0,
            remaining / tr.length,
            self.t / self.t_late,
            remaining / time_left / V_NORM,   # average speed still needed to arrive by t_late
            self.u_prev,
        ], dtype=np.float32)
        return np.clip(obs, -2.0, 2.0)

    def _info(self, arrived: bool = False) -> dict:
        return {"x": self.x, "v": self.v, "t": self.t, "fuel": self.fuel, "violations": self.violations,
                "arrived": arrived, "on_time": arrived and self.t_early <= self.t <= self.t_late}
