# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Named slack scenarios: a train type, a power setup and a fixed track section with one operating move.

Each scenario is one trip on a hand-drawn section, driven once without and once with the SlackFilter.
All trains and sections are invented for this repo. The group count stays at N_GROUPS = 4 because the
predictor's inputs are sized for 4 groups and 3 couplers; a train type changes the mass, the spacing
(train length), the couplers, the traction and the timetable instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coupled import CoupledTrainEnv
from .env import Track

SECTION_LENGTH = 6_000.0  # m


class FixedTrack(Track):
    """A Track drawn by hand: (start in m, grade) and (start in m, limit in km/h) pairs, first at 0."""

    def __init__(self, grades, limits, length: float = SECTION_LENGTH):
        self.length = length
        self.grade_x, self.grade = (np.array(c, dtype=float) for c in zip(*grades))
        self.limit_x, kmh = (np.array(c, dtype=float) for c in zip(*limits))
        self.limit = kmh / 3.6
        self.free_flow_time = self.free_flow_time_from(0.0)


# Train types: CoupledTrainEnv keyword arguments plus the traction, in units of the TrainEnv locomotive
# (300 kN at low speed); at 1.1 units a steady pull puts about 250 kN on the first coupler. The damper c
# keeps the damping ratio near the reference train's 0.15, which also keeps the 0.1 s integration stable
# (tests/test_scenarios.py). timetable stretches the arrival window for the slower trains.
TRAINS = {
    "heavy unit": dict(group_mass=500e3, group_length=200.0, k=3.0e6, c=2.6e5, gap=0.05, units=1.1,
                       timetable=1.3),   # 2,000 t, 800 m, stiff couplers
    "light mixed": dict(group_mass=150e3, group_length=200.0, k=1.5e6, c=1.0e5, gap=0.10, units=1.0,
                        timetable=1.1),  # 600 t, 800 m, softer gear with twice the slack
    "long": dict(group_mass=375e3, group_length=350.0, k=1.2e6, c=1.4e5, gap=0.12, units=1.1,
                 timetable=1.2),         # 1,500 t, 1.4 km, many couplers lumped into soft, loose ones
}
# Power setups: the share of the traction on the middle group (N_GROUPS // 2); the rest stays at the head.
POWER = {"head-end": 0.0, "head + mid": 1 / 3}


# Track sections, each a function of its grade g (as a fraction) so that one number sets how hard it is.
def hard_start(g: float) -> FixedTrack:
    """Standing start on an upgrade: the DQN opens the throttle with the slack bunched."""
    return FixedTrack([(0, g), (2000, 0.0)], [(0, 80), (2500, 100)])


def crest_sag(g: float) -> FixedTrack:
    """Up to a crest at 2.5 km, down into a sag at 3.7 km, up again: run-out at the crest, run-in in the sag."""
    return FixedTrack([(0, 0.0), (1000, g), (2500, -g), (3700, g), (5000, 0.0)], [(0, 80)])


def brake_release(g: float) -> FixedTrack:
    """A limit drop to 60 km/h on a descent, then the limit lifts on an upgrade: brake, release, power."""
    return FixedTrack([(0, 0.0), (1500, -g), (3500, g), (5000, 0.0)], [(0, 100), (3000, 60), (3600, 100)])


SECTIONS = {"hard start": hard_start, "crest then sag": crest_sag, "brake and release": brake_release}


@dataclass(frozen=True)
class Scenario:
    name: str
    train: str
    power: str
    section: str
    grade: float  # the section's grade, a fraction (0.006 = 0.6 %)

    def make_env(self) -> CoupledTrainEnv:
        kw = dict(TRAINS[self.train])
        units, mid = kw.pop("units"), POWER[self.power]
        return CoupledTrainEnv(**kw, head_units=units * (1 - mid), mid_units=units * mid,
                               track=SECTIONS[self.section](self.grade))


# The grade of each section, picked on the plain DQN (PyTorch, seed 0) only, never with the filter: from
# 0.1-1.5 % in 0.1 % steps, among the trips that arrive, the slack-event count nearest 2, then on time,
# then the grade nearest 0.6 %. Order: hard start, crest then sag, brake and release.
GRADES = {
    ("heavy unit", "head-end"): (0.006, 0.006, 0.002),
    ("heavy unit", "head + mid"): (0.004, 0.013, 0.006),
    ("light mixed", "head-end"): (0.004, 0.005, 0.006),
    ("light mixed", "head + mid"): (0.007, 0.005, 0.009),
    ("long", "head-end"): (0.006, 0.006, 0.005),
    ("long", "head + mid"): (0.001, 0.006, 0.005),
}
SCENARIOS = [Scenario(f"{t}, {p}, {s}", t, p, s, g)
             for (t, p), grades in GRADES.items() for s, g in zip(SECTIONS, grades)]
