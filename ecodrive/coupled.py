# Copyright (c) 2026 Kamal Mehta. MIT License.
"""The same train as env.py, split into car groups joined by couplers with slack.

Opt-in: TrainEnv is untouched, and every published number comes from it. CoupledTrainEnv keeps the
11-float observation and the 12 actions, so a DQN trained on TrainEnv can drive it unchanged.

Textbook lumped-mass model; every parameter below is invented for this repo, not taken from a real train:
    m_i dv_i/dt = F_trac (lead only) - F_brake,i - R_i(v_i) - m_i g grade(s_i) + F_{i-1} - F_i
    F_j = k (d_j - gap) + c dd_j/dt   if d_j > gap     (draft: the coupler is stretched)
          k (d_j + gap) + c dd_j/dt   if d_j < -gap    (buff: the coupler is compressed)
          0                           otherwise        (inside the slack, nothing is transmitted)
where d_j is how far coupler j is stretched from its middle position. Each group feels the grade under
its own position. Brakes reach group i i * BRAKE_DELAY seconds after the command, like an air brake
signal running down the train pipe, so the front slows first and the rear runs in.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from .env import B_MAX, DAVIS, DT, G, MASS, SFC, SUBSTEPS, V_NORM, TrainEnv, tractive_force_available

N_GROUPS = 4             # car groups; group 0 carries the locomotives
GROUP_MASS = MASS / N_GROUPS
GROUP_LENGTH = 200.0     # m between group centres, so the train is about 800 m long
K_COUPLER = 2.0e6        # N/m, spring stiffness of one lumped coupler
C_COUPLER = 1.5e5        # N s/m, damping while the coupler is engaged
GAP = 0.05               # m of free slack each way from the middle position (10 cm in total)
BRAKE_DELAY = 2          # s per group for a brake change to arrive
H = 0.1                  # s, internal step for the coupler dynamics (10 per 1 s physics substep)

F_EVENT = 300e3          # N: a coupler force beyond this, either way, is a slack event
F_REF = 300e3            # N, scale for forces in the predictor's features
WINDOW = 4               # decisions of history the slack predictor sees
N_FEATURES = 17          # per decision, see slack_features()


def coupler_force(d: np.ndarray, dd: np.ndarray) -> np.ndarray:
    """Force in each coupler (+ draft, - buff) from its stretch d (m) and stretch rate dd (m/s)."""
    spring = np.where(d > GAP, d - GAP, np.where(d < -GAP, d + GAP, 0.0))
    f = np.where(spring != 0.0, K_COUPLER * spring + C_COUPLER * dd, 0.0)
    # A damper cannot push a stretched coupler into buff (or the reverse) while it separates.
    return np.where(spring > 0, np.maximum(f, 0.0), np.where(spring < 0, np.minimum(f, 0.0), 0.0))


class CoupledTrainEnv(TrainEnv):
    """TrainEnv with N_GROUPS masses and slack couplers. Adds coupler forces and slack events to info."""

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.s = -GROUP_LENGTH * np.arange(N_GROUPS)       # group positions on the track, lead at 0
        self.vg = np.zeros(N_GROUPS)                        # group speeds
        self.brake_cmd = deque([0.0] * (BRAKE_DELAY * (N_GROUPS - 1) + 1), maxlen=BRAKE_DELAY * (N_GROUPS - 1) + 1)
        self.force = np.zeros(N_GROUPS - 1)                 # coupler forces now, + draft / - buff
        self.step_draft = self.step_buff = 0.0              # peak draft and buff over the last decision (N, >= 0)
        self.draft_events = self.buff_events = 0         # decisions with a draft / buff event
        self.peak_draft = self.peak_buff = 0.0
        self.force_trace = []                               # per 1 s substep: (t, x, peak draft, peak buff)
        obs, info = super().reset(seed=seed, options=options)
        self.history = deque([self._features()] * WINDOW, maxlen=WINDOW)
        return obs, self._info()

    def step(self, action: int):
        out = super().step(action)
        self.history.append(self._features())
        return out

    def slack_features(self) -> np.ndarray:
        """(WINDOW, N_FEATURES) float32: the state at the last WINDOW decisions, oldest first."""
        return np.array(self.history, dtype=np.float32)

    # -- physics --------------------------------------------------------------------------------
    def _advance(self, throttle: float, brake: float) -> tuple[float, float, bool]:
        self.step_draft = self.step_buff = 0.0
        out = self._integrate(throttle, brake)
        # Like the limit violations, events are counted per decision: one 10 s step with any coupler past
        # F_EVENT is one event. Counting threshold crossings instead would count each spring oscillation.
        self.draft_events += self.step_draft > F_EVENT
        self.buff_events += self.step_buff > F_EVENT
        self.peak_draft = max(self.peak_draft, self.step_draft)
        self.peak_buff = max(self.peak_buff, self.step_buff)
        return out

    def _integrate(self, throttle: float, brake: float) -> tuple[float, float, bool]:
        tr, m = self.track, GROUP_MASS
        a, b, c = DAVIS
        n_inner = round(DT / SUBSTEPS / H)
        fuel_step, over = 0.0, 0.0
        s, v = self.s, self.vg
        for _ in range(SUBSTEPS):
            self.brake_cmd.appendleft(brake)
            brakes = np.array([self.brake_cmd[i * BRAKE_DELAY] for i in range(N_GROUPS)]) * B_MAX / N_GROUPS
            # Grades change every 600 m or more, so a lookup once per second is enough.
            gravity = m * G * np.array([tr.grade_at(max(p, 0.0)) for p in s])
            sub_draft = np.zeros(N_GROUPS - 1)
            sub_buff = np.zeros(N_GROUPS - 1)
            for _ in range(n_inner):
                f_trac = throttle * tractive_force_available(v[0])
                f = coupler_force(s[:-1] - s[1:] - GROUP_LENGTH, v[:-1] - v[1:])
                net = -brakes - (a + b * v + c * v * v) / N_GROUPS - gravity
                net[0] += f_trac
                net[:-1] -= f
                net[1:] += f
                v_new = np.maximum(v + net / m * H, 0.0)                # a stopped group does not roll back
                v_avg = 0.5 * (v + v_new)
                fuel_step += SFC * f_trac * v_avg[0] * H / 3.6e6
                s = s + v_avg * H
                v = v_new
                sub_draft = np.maximum(sub_draft, f)
                sub_buff = np.maximum(sub_buff, -f)
            self.s, self.vg, self.force = s, v, f
            self.x, self.v = float(s[0]), float(v[0])
            self.t += DT / SUBSTEPS
            self.step_draft = max(self.step_draft, float(sub_draft.max()))
            self.step_buff = max(self.step_buff, float(sub_buff.max()))
            self.force_trace.append((self.t, self.x, float(sub_draft.max()), float(sub_buff.max())))
            over = max(over, self.v - tr.limit_at(min(self.x, tr.length)))
            if self.x >= tr.length:
                return fuel_step, over, True
        return fuel_step, over, False

    # -- predictor inputs and info --------------------------------------------------------------
    def _features(self) -> np.ndarray:
        tr = self.track
        x = min(self.x, tr.length - 1e-6)
        rear = max(float(self.s[-1]), 0.0)
        ahead = np.mean([tr.grade_at(min(x + d, tr.length - 1e-6)) for d in (200, 400, 600, 800, 1000)])
        d = self.s[:-1] - self.s[1:] - GROUP_LENGTH
        return np.concatenate([
            [self.v / V_NORM, tr.limit_at(x) / V_NORM, tr.grade_at(x) * 100, tr.grade_at(rear) * 100,
             ahead * 100, self.u_prev],
            self.force / F_REF,                    # coupler forces now
            [self.step_draft / F_REF, self.step_buff / F_REF],  # peaks over the last decision
            d / GAP,                               # where each coupler sits in its slack
            (self.vg[:-1] - self.vg[1:]),          # closing speed of each coupler, m/s
        ]).astype(np.float32)

    def _info(self, arrived: bool = False) -> dict:
        info = super()._info(arrived)
        if hasattr(self, "force"):
            info |= {"draft_events": self.draft_events, "buff_events": self.buff_events,
                     "peak_draft": self.peak_draft, "peak_buff": self.peak_buff,
                     "step_peak": max(self.step_draft, self.step_buff)}
        return info
