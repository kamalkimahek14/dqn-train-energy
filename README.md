# Train eco-driving with Deep Q-Networks

A Double DQN agent drives a simulated 1,000 t train over a 10 km track. It has to arrive inside a
time window, stay under the speed limit and burn as little fuel as it can. I wrote the agent twice,
once in PyTorch and once in TensorFlow/Keras. Both sit on the same environment, replay buffer,
config and training loop, so the two files can be read side by side.

On 50 held-out test tracks, averaged over two training seeds, the PyTorch agent used **31.5 % less
fuel** than a rule-based driver that holds the speed limit and arrived on time on **99 %** of the
tracks. The TensorFlow agent saved 24.8 % and arrived on time on 95 %. Neither is perfect on the
speed limit yet; the numbers and the history are below.

![One test episode: speed against the limit, and the throttle notch per decision](assets/episode.gif)

*Test track 10003, PyTorch agent (seed 0). Top: speed against the limit (red), DQN in cyan, the
baseline in grey. Bottom: the notch or brake level picked at each 10 s decision.*

## Why this problem

Fuel-efficient driving is a sequential decision problem. A notch chosen now changes the speed, the
time left and the fuel for the rest of the trip, and the right choice depends on what lies ahead: a
climb, a descent, a lower limit, the deadline. "Hold the limit" is safe and punctual. It also spends
kinetic energy that the next lower limit throws away in the brakes, and it never uses the slack in
the timetable. Those are the two gaps a learned policy can go after.

## The environment

`ecodrive/env.py` is a point-mass train with textbook physics and a Gymnasium API:

```
m dv/dt = F_traction − F_brake − (A + B v + C v²) − m g · grade
F_traction = notch/8 · min(F_max, P_max / v)        fuel = SFC · ∫ F_traction · v dt
```

Each reset draws a new track from the seed: grades between −0.6 % and +0.6 % in 0.6–1.5 km pieces,
and limits of 60, 80 or 100 km/h in 1.5–3.5 km pieces. The agent decides every 10 s and the physics
runs at 1 s inside that step. A trip takes about 55–70 decisions.

| | |
|---|---|
| **State** (11 floats) | speed; current limit; next limit and the distance to it; grade here; mean grade over the next 1 km; distance left; elapsed time; average speed still needed; schedule slack; last control level |
| **Actions** (12) | throttle notch 0–8 (0 = coast) or brake level 1–3 |
| **Reward** per decision | −0.1 per litre of fuel, −5 per m/s over the limit, −0.05 × the change in control level (full brake = −1, full notch = +1) |
| **At arrival** | −5 per minute outside the window [1.05, 1.40] × free-flow time, and −10 more if late at all |
| **Hard stop** | at 1.70 × free-flow time: −10 × the share of track still ahead, plus the lateness terms |

Free-flow time is the trip at exactly the limit everywhere. Schedule slack is the time left before
the deadline minus the free-flow time of the track still ahead: once it goes negative, the train can
no longer make it.

The baseline (`ecodrive/baseline.py`) is a cruise control that knows the train's physics. It aims
0.5 m/s under the limit, picks the notch that balances resistance and grade, and starts braking two
decisions before it needs to for a lower limit. On all 50 test tracks it arrived on time with zero
limit violations, at 1.20 × free-flow time on average.

## The algorithm

![Agent-environment loop and the Double DQN update](assets/dqn_loop.svg)

Plain DQN with the usual stabilisers:

- **Experience replay**: a 100k ring buffer in numpy (`ecodrive/replay.py`), uniform batches of 64.
- **Target network**: a frozen copy, hard-synced every 2,000 environment steps.
- **Double DQN**: the online network picks the next action and the target network scores it. This
  takes out most of the max-operator overestimation.
- **Huber loss** (δ = 1) and a global gradient-norm clip of 10.
- **ε-greedy**, linear from 1.0 to 0.05 over 100k steps.
- **One seed** for tracks, exploration, replay sampling and weight init.

Both backends use the same settings: Adam at 5e-4 (ε = 1e-7), an MLP 11 → 128 → 128 → 12 with Glorot
initialisation, one gradient step every two decisions, 300k environment steps (about 4,000 trips).
Every 10k steps the greedy policy plays 30 validation tracks (seeds 1000–1029) and the best weights
are kept. Test tracks start at seed 10000 and only `evaluate.py` touches them.

## Results

`evaluate.py --seeds 0 1` on test seeds 10000–10049. Each cell is the mean over two training seeds,
with the per-seed range in brackets. Raw numbers are in `results/eval_train_*.json`.

![Learning curves](assets/learning_curves.png)

| 50 test tracks, 2 seeds | DQN, PyTorch | DQN, TensorFlow | Baseline |
|---|---|---|---|
| Fuel (L) | **33.7** [32.0, 35.4] | 37.0 [37.0, 37.1] | 49.2 |
| Fuel saving vs baseline | **31.5 %** [28.1, 35.0] | 24.8 % [24.7, 24.9] | |
| On time | **99 %** [98, 100] | 95 % [90, 100] | 100 % |
| Decisions over the limit, per trip | 0.31 [0.16, 0.46] | **0.10** [0.04, 0.16] | 0 |
| Trip time / free-flow time | 1.30 [1.29, 1.32] | 1.30 [1.27, 1.32] | 1.20 |
| Control changes, per trip | **24.2** | 26.9 | 30.3 |
| Episode return | −5.24 [−6.38, −4.10] | **−4.99** [−5.50, −4.48] | −5.92 |

Both agents found the same policy: accelerate less hard, cruise below the limit, coast into lower
limits, and spend some of the timetable slack. They use about 8 % more trip time than the baseline
(1.30 against 1.20 × free-flow), still well inside the window, and change notch less often.

What is not solved yet is the speed limit. The PyTorch agent goes over on 0.31 decisions per trip
on average. The TensorFlow agent gets to 0.10, but one of its seeds arrives late on 5 of the 50
tracks. The baseline never does either, because it plans its braking with the train's physics; the
agent has to learn that from the reward alone.

The track in the GIF is a typical one (28 % saving, on time, no overshoot). The last 1.5 km is the
weak part of most episodes: the agent switches quickly between brake 1 and low notches while it lines
up the final limit and the arrival.

### How I got here

I tuned in two passes, about 36 minutes and 47 minutes of CPU training. Every number below comes from
the same 50 test tracks.

**First pass** (150k steps, one seed, 10 validation tracks, no schedule-slack input, Adam at 5e-4,
target sync every 1,000 steps):

| Setting | Backend | Fuel saving | On time | Over the limit / trip |
|---|---|---|---|---|
| −0.5 per m/s over, −2 per minute late | PyTorch | 47.0 % | 84 % | 0.54 |
| Same, plus −1 flat per decision over and −5 per minute late | PyTorch | diverged, stopped at 140k | | |
| −0.5 per m/s over, −5 per minute late | PyTorch | 37.6 % | 62 % | 0.40 |
| −0.5 per m/s over, −5 per minute late | TensorFlow | 17.5 % | 36 % | 2.28 |

The agents were buying fuel with lateness. They aimed at the late edge of the window and, with 10 s
decisions, landed past it on 16 % to 64 % of the tracks.

**Second pass** (300k steps, two seeds, 30 validation tracks):

- **Round A**: added the schedule-slack input and the flat −10 for any late arrival, and raised the
  limit penalty to −2 per m/s. Lateness was fixed in one step: all four runs were on time on 98–100 %
  of the tracks. But two of the four runs diverged. Their TD loss climbed from about 0.1 to 2 and to 17,
  and the limit overshoots went up to 3.5 per trip.
- **Round B**: Adam at 3e-4, target sync every 2,000 steps, −5 per m/s over the limit. Stable, but
  most runs only started to learn after 200k steps, and on time fell to 54–96 %.
- **Round C**: back to 5e-4, keeping the slower target sync and the −5. TensorFlow reached the table
  above. PyTorch was clearly behind (78 % on time).
- **Round D**: the PyTorch and Keras defaults were not the same. `nn.Linear` starts from Kaiming-uniform
  weights with non-zero biases, Keras `Dense` from Glorot-uniform with zero biases, and Adam's ε is
  1e-8 against 1e-7. I matched the PyTorch agent to the Keras defaults and retrained only the PyTorch
  seeds. Their best validation returns went from −10.5 and −10.8 to −6.9 and −4.0, and on time on the
  test tracks went from 78 % to 99 %.

I picked the final setting by mean validation return, not by test results. I also considered scaling
the rewards down, but with Adam that mostly acts like raising the learning rate, which was the wrong
direction here.

### Sanity check: CartPole-v1

Same code and loop, 50k steps, Adam at 1e-3, target sync every 500 steps, one seed, 50 test episodes:

| | Mean return | Min |
|---|---|---|
| PyTorch | 304.8 ± 52.5 | 252 |
| TensorFlow | **500.0** ± 0.0 | 500 |

TensorFlow's best checkpoint holds the pole for the full 500 steps on every test episode. DQN on
CartPole swings a lot during training, as both curves show, and a single seed says little about which
backend is better here.

### What is next

- Treat the speed limit as a hard constraint: mask the actions that cannot keep the train under the
  next limit, using the same braking model the baseline uses. That should take the overshoots to zero.
- Shorter decisions (5 s) near limit changes and at the end of the trip, where the 10 s steps are too
  coarse.
- More seeds. Two per backend shows the spread but not a reliable ranking of the frameworks.
- A terminal stop at the platform, so the arrival is a real stopping problem.

## PyTorch vs TensorFlow

Same algorithm, same hyperparameters, same initialisation, same loop. Only
`ecodrive/agents/torch_dqn.py` and `ecodrive/agents/tf_dqn.py` differ, and they have the same methods
in the same order.

| | PyTorch | TensorFlow / Keras |
|---|---|---|
| Lines of code in the agent (no comments, docstrings or blank lines) | 54 | 52 |
| Network | `nn.Sequential` of `nn.Linear` + `nn.ReLU` | `keras.Sequential` of `Dense` |
| Initialisation | set explicitly to Glorot uniform, zero bias | Keras default (Glorot uniform, zero bias) |
| Q(s, a) of the taken action | `q.gather(1, a.unsqueeze(1))` | `tf.gather(q, a, batch_dims=1)` |
| Huber loss | `F.smooth_l1_loss` (β = 1) | `keras.losses.Huber(delta=1.0)` |
| Gradient step | `loss.backward()`, `clip_grad_norm_`, `opt.step()` | `GradientTape`, Adam with `global_clipnorm` |
| Speed-up | none needed, eager | `@tf.function` on the forward pass and the train step |
| Target sync | `load_state_dict(state_dict())` | `set_weights(get_weights())` |
| Wall time, train task, 300k steps | 506 s, 511 s | 585 s, 587 s |
| Wall time, CartPole, 50k steps | 145 s | 249 s |

The wall times are rough. Each process was capped at 4 threads on a 16-thread laptop CPU and ran at
idle priority while other work used the machine. The final TensorFlow runs shared the CPU with three
other training runs, and the final PyTorch runs with two. On a network this small the per-call overhead
dominates, not the arithmetic, so neither framework has a clear edge.

Versions: Python 3.13, PyTorch 2.14.1 (CPU), TensorFlow 2.21.0 (CPU) with Keras 3.15.1,
Gymnasium 1.4.0, NumPy 2.4.6.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # CPU-only wheels

python train.py --backend torch --seed 0              # or --backend tf; repeat with --seed 1
python evaluate.py --backend torch --seeds 0 1        # DQN vs baseline on 50 test tracks
python train.py --backend tf --env CartPole-v1        # sanity check
python evaluate.py --backend tf --env CartPole-v1
python plots.py                                       # learning curves and the GIF
```

`train.py` writes weights, a per-episode `log.csv` and `summary.json` to `runs/<env>-<backend>-s<seed>/`.
`evaluate.py` writes `results/eval_<env>_<backend>.json`; `--trace-track` picks the test track that
`plots.py` animates. `--threads` sets the CPU threads (default 4).

## Tests

`pytest -q` runs 28 tests in under a minute on a laptop CPU. GitHub Actions runs the same suite, with
both backends, on Python 3.11 (`.github/workflows/tests.yml`).

| File | What it checks |
|---|---|
| `tests/test_env.py` | speed never negative; position and fuel never decrease; coasting and braking burn no fuel; more notch, more fuel; resistance slows a coasting train; seeded resets repeat; free-flow time left shrinks to zero; the baseline is on time; standing still hits the hard stop |
| `tests/test_replay.py` | the size cap, overwriting the oldest entry, sample shapes and dtypes, rows staying whole, seeded sampling |
| `tests/test_agents.py` | per backend: the loss equals a numpy Double DQN + Huber reference; one update lowers the loss on its batch; the target network moves only on sync; save and load round-trip |
| `tests/test_loop.py` | updates start after warm-up, the target syncs on schedule, the ε schedule |
| `tests/test_smoke.py` | 400 real training steps per backend on both tasks, each well under 60 s |

## Layout

```
ecodrive/
  env.py            train environment: physics, random tracks, reward
  baseline.py       rule-based driver that holds the limit
  replay.py         replay buffer (shared)
  config.py         hyperparameters (shared), env factory, epsilon schedule
  loop.py           training loop (shared)
  agents/
    torch_dqn.py    Double DQN in PyTorch
    tf_dqn.py       Double DQN in TensorFlow/Keras
train.py            train one backend on one task
evaluate.py         DQN vs baseline on test tracks
plots.py            learning curves and the episode GIF
results/            evaluation numbers and the recorded test episode
assets/             README figures
tests/              pytest suite
```

## Author

Kamal Mehta, ML engineer working on reinforcement learning for train control.
[kamal-mehta.com](https://kamal-mehta.com)

## License

MIT. See [LICENSE](LICENSE).
