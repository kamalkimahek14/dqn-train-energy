# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Evaluate a trained agent on held-out episodes.

    python evaluate.py --backend torch                  # DQN vs the rule-based baseline, 50 test tracks
    python evaluate.py --backend tf --env CartPole-v1   # mean return over 50 episodes

Writes results/eval_<env>_<backend>.json, and for the train task one recorded episode
(results/rollout_<backend>.json) that plots.py turns into the animation.
"""
import argparse
import json
import os

import numpy as np


def rollout(env, policy, seed):
    """One episode on test track `seed`; returns its metrics and a step-by-step trace."""
    obs, info = env.reset(seed=seed)
    trace = {"x": [0.0], "v": [0.0], "t": [0.0], "fuel": [0.0], "action": []}
    total, changes, prev, done = 0.0, 0, 0, False
    while not done:
        action = policy(obs)
        changes += action != prev
        prev = action
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        trace["x"].append(info["x"])
        trace["v"].append(info["v"])
        trace["t"].append(info["t"])
        trace["fuel"].append(info["fuel"])
        trace["action"].append(int(action))
        done = terminated or truncated
    metrics = {"return": total, "fuel_l": info["fuel"], "time_s": info["t"],
               "time_ratio": info["t"] / env.track.free_flow_time, "on_time": float(info["on_time"]),
               "violation_steps": info["violations"], "control_changes": changes}
    return metrics, trace


def summarize(rows):
    return {k: {"mean": float(np.mean([r[k] for r in rows])), "std": float(np.std([r[k] for r in rows]))}
            for k in rows[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["torch", "tf"], required=True)
    p.add_argument("--env", choices=["train", "CartPole-v1"], default="train")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--weights", default="best", choices=["best", "final"])
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from ecodrive.agents import limit_threads, make_agent
    from ecodrive.baseline import baseline_action
    from ecodrive.config import TEST_SEED, get_config, make_env
    from ecodrive.loop import run_episode

    limit_threads(args.backend, 4)
    cfg = get_config(args.env)
    env = make_env(args.env)
    agent = make_agent(args.backend, env.observation_space.shape[0], int(env.action_space.n), cfg)
    agent.load(os.path.join(args.runs, f"{args.env}-{args.backend}", args.weights))
    os.makedirs(args.results, exist_ok=True)
    seeds = [TEST_SEED + i for i in range(args.episodes)]

    if args.env == "CartPole-v1":
        returns = [run_episode(env, agent.greedy, seed=s)["return"] for s in seeds]
        out = {"episodes": args.episodes, "return_mean": float(np.mean(returns)),
               "return_std": float(np.std(returns)), "return_min": float(np.min(returns))}
        print(f"CartPole-v1 {args.backend}: return {out['return_mean']:.1f} +- {out['return_std']:.1f}"
              f" (min {out['return_min']:.0f}) over {args.episodes} episodes")
    else:
        dqn, base = [], []
        for s in seeds:
            m, dqn_trace = rollout(env, agent.greedy, s)
            dqn.append(m)
            m, base_trace = rollout(env, lambda obs: baseline_action(env), s)
            base.append(m)
            if s == seeds[0]:  # keep the first test track for the animation
                tr = env.track
                trace = {"seed": s, "length": tr.length, "limit_x": tr.limit_x.tolist(), "limit": tr.limit.tolist(),
                         "grade_x": tr.grade_x.tolist(), "grade": tr.grade.tolist(), "dqn": dqn_trace,
                         "baseline": base_trace}
                with open(os.path.join(args.results, f"rollout_{args.backend}.json"), "w") as f:
                    json.dump(trace, f)
        out = {"episodes": args.episodes, "dqn": summarize(dqn), "baseline": summarize(base),
               "fuel_saving_pct": 100 * (1 - np.mean([d["fuel_l"] for d in dqn]) / np.mean([b["fuel_l"] for b in base])),
               "dqn_less_fuel_share": float(np.mean([d["fuel_l"] < b["fuel_l"] for d, b in zip(dqn, base)]))}
        print(f"{'metric':<16}{'DQN (' + args.backend + ')':>18}{'baseline':>18}")
        for k in out["dqn"]:
            d, b = out["dqn"][k], out["baseline"][k]
            print(f"{k:<16}{d['mean']:>11.2f} +-{d['std']:5.2f}{b['mean']:>11.2f} +-{b['std']:5.2f}")
        print(f"fuel saving {out['fuel_saving_pct']:.1f} %, DQN used less fuel on "
              f"{100 * out['dqn_less_fuel_share']:.0f} % of {args.episodes} test tracks")

    with open(os.path.join(args.results, f"eval_{args.env}_{args.backend}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
