# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Rule-based baseline: hold the speed limit, and brake early enough for a lower limit ahead.

This is what a careful driver with a cruise control does. It knows the train's physics, so it
picks the notch that holds the target speed on the current grade. It ignores fuel and the time
window, which is the gap the learned agent can exploit.
"""
import math

from .env import (B_MAX, DT, G, MASS, N_BRAKES, N_NOTCHES, TrainEnv, resistance,
                  tractive_force_available)

MARGIN = 0.5        # m/s below the limit, so small overshoots stay legal
PLAN_DECEL = 0.25   # m/s^2, comfortable braking rate used to plan for a lower limit ahead


def baseline_action(env: TrainEnv) -> int:
    track, x, v = env.track, env.x, env.v
    target = track.limit_at(x) - MARGIN

    # Start braking when the distance to a lower limit is about the braking distance plus two steps.
    change_x, next_limit = track.next_limit_change(x)
    if next_limit < target + MARGIN:
        v_next = next_limit - MARGIN
        brake_dist = max(v * v - v_next * v_next, 0.0) / (2 * PLAN_DECEL)
        if change_x - x <= brake_dist + 2 * v * DT:  # two steps of look-ahead: decisions are 10 s apart
            target = min(target, v_next)

    # Plan with the more downhill of the grade here and where this step ends, so a grade change
    # inside the 10 s step cannot push the train over the limit.
    grade = min(track.grade_at(x), track.grade_at(min(x + v * DT, track.length - 1e-6)))
    # Force that would reach the target speed in one decision step on this grade.
    f_needed = MASS * (target - v) / DT + resistance(v) + MASS * G * grade
    if f_needed >= 0:
        # Round down: sagging a little below the target on a climb is legal, overshooting is not.
        return min(N_NOTCHES, int(N_NOTCHES * f_needed / tractive_force_available(v)))
    if v > target + MARGIN:
        return N_NOTCHES + min(N_BRAKES, max(1, math.ceil(N_BRAKES * -f_needed / B_MAX)))
    # Coast, unless gravity alone would carry the train over the limit within this step.
    v_coast = v + DT * (-resistance(v) - MASS * G * grade) / MASS
    return N_NOTCHES + 1 if v_coast > target + MARGIN - 0.2 else 0
