# Train eco-driving with Deep Q-Networks

A Double DQN agent drives a simulated 1,000 t train over a 10 km track. It has to arrive inside a
time window, stay under the speed limit and burn as little fuel as it can. I wrote the agent twice,
once in PyTorch and once in TensorFlow/Keras. Both sit on the same environment, replay buffer,
config and training loop, so the two files can be read side by side.

On 50 held-out test tracks the PyTorch agent used **37.6 % less fuel** than a rule-based driver that
holds the speed limit (30.7 L against 49.2 L). That saving is not free yet: the agent arrived on time
on 62 % of the tracks, against 100 % for the baseline. The details are below.

![One test episode: speed against the limit, and the throttle notch per decision](assets/episode.gif)

*Test track 10000, PyTorch agent. Top: speed against the limit (red), DQN in cyan, the baseline in
grey. Bottom: the notch or brake level picked at each 10 s decision.*

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
| **State** (10 floats) | speed; current limit; next limit and the distance to it; grade here; mean grade over the next 1 km; distance left; elapsed time; average speed still needed to arrive on time; last control level |
| **Actions** (12) | throttle notch 0–8 (0 = coast) or brake level 1–3 |
| **Reward** per decision | −0.1 per litre of fuel, −0.5 per m/s over the limit, −0.05 × the change in control level (full brake = −1, full notch = +1) |
| **At arrival** | −5 per minute outside the window [1.05, 1.40] × free-flow time |
| **Hard stop** | at 1.70 × free-flow time: −10 × the share of track still ahead, plus the lateness term |

Free-flow time is the trip at exactly the limit everywhere. No real run gets there, because the train
has to accelerate from a standstill.

The baseline (`ecodrive/baseline.py`) is a cruise control that knows the train's physics. It aims
0.5 m/s under the limit, picks the notch that balances resistance and grade, and starts braking two
decisions before it needs to for a lower limit. On all 50 test tracks it arrived on time with zero
limit violations, at 1.20 × free-flow time on average.

## The algorithm

![Agent-environment loop and the Double DQN update](assets/dqn_loop.svg)

Plain DQN with the usual stabilisers:

- **Experience replay**: a 100k ring buffer in numpy (`ecodrive/replay.py`), uniform batches of 64.
- **Target network**: a frozen copy, hard-synced every 1,000 environment steps.
- **Double DQN**: the online network picks the next action and the target network scores it. This
  takes out most of the max-operator overestimation.
- **Huber loss** (δ = 1) and a global gradient-norm clip of 10.
- **ε-greedy**, linear from 1.0 to 0.05 over 60k steps.
- **One seed** for tracks, exploration, replay sampling and weight init.

One gradient step every two decisions, Adam at 5e-4, an MLP 10 → 128 → 128 → 12, 150k environment
steps (about 2,000 trips). Every 10k steps the greedy policy plays 10 validation tracks (seeds
1000–1009) and the best weights are kept. Test tracks start at seed 10000 and are only touched by
`evaluate.py`.

## Results

All numbers come from `results/*.json`, produced by `evaluate.py` on test seeds 10000–10049.

![Learning curves](assets/learning_curves.png)

| 50 test tracks | DQN, PyTorch | DQN, TensorFlow | Baseline |
|---|---|---|---|
| Fuel (L) | **30.7** ± 7.0 | 40.6 ± 6.1 | 49.2 ± 6.0 |
| Fuel saving vs baseline | **37.6 %** | 17.5 % | |
| Tracks where DQN used less fuel | 50 / 50 | 48 / 50 | |
| Trip time (s) | 659 ± 48 | 694 ± 62 | 562 ± 37 |
| Trip time / free-flow time | 1.40 ± 0.07 | 1.48 ± 0.13 | 1.20 ± 0.04 |
| On time | 62 % | 36 % | **100 %** |
| Decisions over the limit, per trip | 0.40 | 2.28 | **0** |
| Control changes, per trip | **20.9** | 25.6 | 30.3 |
| Episode return | **−4.87** ± 2.22 | −10.29 ± 5.28 | −5.92 ± 0.60 |

The PyTorch agent beats the baseline on the reward it was trained on, and by a wide margin on fuel.
It found the obvious lever: cruise slower and use the timetable slack. It also coasts more and changes
notch a third less often. The problem is precision. It aims for the late edge of the window, and with
10 s decisions it lands past it on about four tracks in ten. The overshoots of the limit are small
(the reward charges by the m/s), but a real driver would get none.

The TensorFlow agent learned the same shape of policy and does not beat the baseline overall. Its
training curve tracks the PyTorch one closely. Its validation score peaked at step 100k and then
wandered, so model selection kept that early checkpoint. With only 10 validation tracks the score is
noisy, and I think that choice, more than the framework, explains most of the gap. A second seed for
each backend would settle it; I have not run one yet.

The GIF shows a flaw worth knowing about. Near 8.8 km the agent brakes almost to a stop before a limit
drop and then has to pull away again. It still finishes with less fuel than the baseline, but a human
driver would not do that.

### What I tried first

The table above is the third reward setting. The first one charged −2 per minute late. With it, the
PyTorch agent reached 26.1 L on the same 50 test tracks (47 % less than the baseline), but it was on
time on only 84 % of them and went over the limit on 0.54 decisions per trip. It was buying fuel with
lateness. For the second try I added a flat −1 for every decision over the limit and raised lateness
to −5 per minute. That training was unstable: validation return reached −9.3 at 100k steps and then
fell to −31, so I stopped it at 140k. The current setting keeps −5 per minute and drops the flat term.

### Sanity check: CartPole-v1

Same code and loop, 50k steps, lr 1e-3, target sync every 500 steps, 50 test episodes:

| | Mean return | Min |
|---|---|---|
| PyTorch | 345.4 ± 88.7 | 230 |
| TensorFlow | **500.0** ± 0.0 | 500 |

TensorFlow's best checkpoint holds the pole for the full 500 steps on every episode. PyTorch's training
return was still climbing at 50k (365 over the last 20 episodes). DQN on CartPole is known to swing a
lot, and both curves show it.

### What is next

- Train longer. Both train-task curves were still rising at 150k steps.
- Use more validation tracks (50 rather than 10), and three seeds per backend.
- Treat the deadline and the limit as hard constraints rather than soft penalties, for example by
  masking actions that cannot keep the train under the next limit.
- Add a terminal stop at the platform, so the arrival is a real stopping problem.

## PyTorch vs TensorFlow

Same algorithm, same hyperparameters, same loop. Only `ecodrive/agents/torch_dqn.py` and
`ecodrive/agents/tf_dqn.py` differ, and they have the same methods in the same order.

| | PyTorch | TensorFlow / Keras |
|---|---|---|
| Lines of code in the agent (no comments, docstrings or blank lines) | 49 | 52 |
| Network | `nn.Sequential` of `nn.Linear` + `nn.ReLU` | `keras.Sequential` of `Dense` |
| Q(s, a) of the taken action | `q.gather(1, a.unsqueeze(1))` | `tf.gather(q, a, batch_dims=1)` |
| Huber loss | `F.smooth_l1_loss` (β = 1) | `keras.losses.Huber(delta=1.0)` |
| Gradient step | `loss.backward()`, `clip_grad_norm_`, `opt.step()` | `GradientTape`, Adam with `global_clipnorm` |
| Speed-up | none needed, eager | `@tf.function` on the forward pass and the train step |
| Target sync | `load_state_dict(state_dict())` | `set_weights(get_weights())` |
| Wall time, train task, 150k steps | 602 s | 545 s |
| Wall time, CartPole, 50k steps | 305 s | 249 s |

The wall times are from one 16-thread laptop CPU, with each backend capped at 4 threads and run at
idle priority. The two backends trained at the same time, and the machine was busy with other work, so
read them as rough. In a quick run on its own, with nothing else in parallel, TensorFlow was also the
faster of the two: the compiled train step beats eager PyTorch on a network this small.

Versions: Python 3.13, PyTorch 2.14.1 (CPU), TensorFlow 2.21.0 (CPU) with Keras 3.15.1,
Gymnasium 1.4.0, NumPy 2.4.6.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # CPU-only wheels

python train.py --backend torch                       # or --backend tf
python evaluate.py --backend torch                    # DQN vs baseline on 50 test tracks
python train.py --backend tf --env CartPole-v1        # sanity check
python evaluate.py --backend tf --env CartPole-v1
python plots.py                                       # learning curves and the GIF
```

`train.py` writes weights, a per-episode `log.csv` and `summary.json` to `runs/<env>-<backend>/`.
`evaluate.py` writes `results/eval_<env>_<backend>.json`. `--threads` sets the CPU threads (default 4).

## Tests

`pytest -q` runs 27 tests in under a minute on a laptop CPU. GitHub Actions runs the same suite, with
both backends, on Python 3.11 (`.github/workflows/tests.yml`).

| File | What it checks |
|---|---|
| `tests/test_env.py` | speed never negative; position and fuel never decrease; coasting and braking burn no fuel; more notch, more fuel; resistance slows a coasting train; seeded resets repeat; the baseline is on time; standing still hits the hard stop |
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
