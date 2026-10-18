# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Evaluate a trained agent on held-out episodes.

    python evaluate.py --backend torch --seeds 0 1      # DQN vs the rule-based baseline, 50 test tracks
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


def across(per_seed: list[dict]) -> dict:
    """Mean and range over training seeds of each per-seed number."""
    return {k: {"mean": float(np.mean([d[k] for d in per_seed])), "min": float(np.min([d[k] for d in per_seed])),
                "max": float(np.max([d[k] for d in per_seed]))} for k in per_seed[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["torch", "tf"], required=True)
    p.add_argument("--env", choices=["train", "CartPole-v1"], default="train")
    p.add_argument("--seeds", type=int, nargs="+", default=[0], help="training seeds to evaluate")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--trace-track", type=int, default=0, help="test track (0-based) to record for plots.py")
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
    os.makedirs(args.results, exist_ok=True)
    tracks = [TEST_SEED + i for i in range(args.episodes)]

    def load(seed):
        agent = make_agent(args.backend, env.observation_space.shape[0], int(env.action_space.n), cfg)
        agent.load(os.path.join(args.runs, f"{args.env}-{args.backend}-s{seed}", args.weights))
        return agent

    if args.env == "CartPole-v1":
        per_seed = {}
        for seed in args.seeds:
            returns = [run_episode(env, load(seed).greedy, seed=t)["return"] for t in tracks]
            per_seed[seed] = {"return_mean": float(np.mean(returns)), "return_std": float(np.std(returns)),
                              "return_min": float(np.min(returns))}
            print(f"CartPole-v1 {args.backend} seed {seed}: return {per_seed[seed]['return_mean']:.1f}"
                  f" +- {per_seed[seed]['return_std']:.1f} (min {per_seed[seed]['return_min']:.0f})")
        out = {"episodes": args.episodes, "per_seed": per_seed, "across_seeds": across(list(per_seed.values()))}
    else:
        base, base_traces = [], {}
        for t in tracks:
            m, base_traces[t] = rollout(env, lambda obs: baseline_action(env), t)
            base.append(m)
        out = {"episodes": args.episodes, "baseline": summarize(base), "per_seed": {}}
        for seed in args.seeds:
            agent, dqn = load(seed), []
            for t in tracks:
                m, trace = rollout(env, agent.greedy, t)
                dqn.append(m)
                if seed == args.seeds[0] and t == tracks[args.trace_track]:
                    tr = env.track
                    with open(os.path.join(args.results, f"rollout_{args.backend}.json"), "w") as f:
                        json.dump({"track": t, "training_seed": seed, "length": tr.length,
                                   "limit_x": tr.limit_x.tolist(), "limit": tr.limit.tolist(),
                                   "grade_x": tr.grade_x.tolist(), "grade": tr.grade.tolist(),
                                   "dqn": trace, "baseline": base_traces[t]}, f)
            summary = summarize(dqn)
            summary["fuel_saving_pct"] = {"mean": 100 * (1 - summary["fuel_l"]["mean"] / out["baseline"]["fuel_l"]["mean"])}
            summary["less_fuel_share"] = {"mean": float(np.mean([d["fuel_l"] < b["fuel_l"] for d, b in zip(dqn, base)]))}
            out["per_seed"][seed] = summary
        out["across_seeds"] = across([{k: v["mean"] for k, v in d.items()} for d in out["per_seed"].values()])

        print(f"{'metric':<18}{'baseline':>10}" + "".join(f"{'seed ' + str(s):>10}" for s in args.seeds)
              + f"{'mean':>10}{'range':>18}")
        for k, a in out["across_seeds"].items():
            b = f"{out['baseline'][k]['mean']:10.2f}" if k in out["baseline"] else " " * 10
            print(f"{k:<18}{b}" + "".join(f"{out['per_seed'][s][k]['mean']:10.2f}" for s in args.seeds)
                  + f"{a['mean']:10.2f}   [{a['min']:6.2f}, {a['max']:6.2f}]")

    with open(os.path.join(args.results, f"eval_{args.env}_{args.backend}.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
