# Copyright (c) 2026 Kamal Mehta. MIT License.
"""The DQN training loop. It only calls agent.greedy / update / sync_target / save,
so the same loop drives the PyTorch and the TensorFlow agent."""
import os
import time

import numpy as np

from .config import VALIDATION_SEED, DQNConfig, epsilon, make_env
from .replay import ReplayBuffer


def run_episode(env, policy, seed=None) -> dict:
    """Play one episode with policy(obs) -> action; return the total reward, length and final info."""
    obs, info = env.reset(seed=seed)
    total, length, done = 0.0, 0, False
    while not done:
        obs, reward, terminated, truncated, info = env.step(policy(obs))
        total += reward
        length += 1
        done = terminated or truncated
    return {"return": total, "length": length, **info}


def train(agent, env, cfg: DQNConfig, out_dir: str | None = None, log=print) -> dict:
    rng = np.random.default_rng(cfg.seed)          # drives exploration only
    n_actions = env.action_space.n
    buffer = ReplayBuffer(cfg.buffer_size, env.observation_space.shape[0], seed=cfg.seed)
    val_env = make_env(cfg.env_id)
    episodes, evals, losses = [], [], []
    best = -np.inf

    obs, _ = env.reset(seed=cfg.seed)
    ep_return, ep_len = 0.0, 0
    start = time.perf_counter()
    for step in range(1, cfg.total_steps + 1):
        # 1) Act: epsilon-greedy on the online network.
        eps = epsilon(cfg, step)
        action = int(rng.integers(n_actions)) if rng.random() < eps else agent.greedy(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)

        # 2) Store. Only a true terminal state stops bootstrapping; a time-limit truncation does not.
        buffer.add(obs, action, reward, next_obs, terminated)
        obs, ep_return, ep_len = next_obs, ep_return + reward, ep_len + 1
        if terminated or truncated:
            episodes.append({"step": step, "return": ep_return, "length": ep_len, "epsilon": eps,
                             "fuel": info.get("fuel"), "on_time": info.get("on_time")})
            obs, _ = env.reset()
            ep_return, ep_len = 0.0, 0

        # 3) Learn from a random replay batch, and refresh the target network on a fixed schedule.
        if step >= cfg.learning_starts and step % cfg.train_every == 0:
            losses.append(agent.update(buffer.sample(cfg.batch_size)))
        if step % cfg.target_update_every == 0:
            agent.sync_target()

        # 4) Model selection: greedy play on validation episodes that training never sees.
        if step % cfg.eval_every == 0:
            score = float(np.mean([run_episode(val_env, agent.greedy, seed=VALIDATION_SEED + i)["return"]
                                   for i in range(cfg.eval_episodes)]))
            evals.append({"step": step, "val_return": score})
            if score > best:
                best = score
                if out_dir:
                    agent.save(os.path.join(out_dir, "best"))
            recent = np.mean([e["return"] for e in episodes[-20:]]) if episodes else float("nan")
            loss = np.mean(losses[-1000:]) if losses else float("nan")
            log(f"step {step:>7}  eps {eps:.2f}  train return {recent:8.2f}  val return {score:8.2f}"
                f"  loss {loss:.4f}  {time.perf_counter() - start:6.0f} s")

    if out_dir:
        agent.save(os.path.join(out_dir, "final"))
    return {"episodes": episodes, "evals": evals, "best_val_return": best,
            "wall_time_s": time.perf_counter() - start}
