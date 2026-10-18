# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Train a Double DQN agent.

    python train.py --backend torch                   # the train-driving task
    python train.py --backend tf --env CartPole-v1    # sanity check on a classic task

Writes runs/<env>-<backend>-s<seed>/: best and final weights, log.csv (one row per episode), summary.json.
"""
import argparse
import csv
import json
import os
from dataclasses import asdict


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["torch", "tf"], required=True)
    p.add_argument("--env", choices=["train", "CartPole-v1"], default="train")
    p.add_argument("--steps", type=int, help="override the number of environment steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4, help="CPU threads for the backend")
    p.add_argument("--out", default="runs")
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))  # read when torch / TF load
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")          # hide TensorFlow's start-up info lines
    from ecodrive.agents import limit_threads, make_agent
    from ecodrive.config import get_config, make_env
    from ecodrive.loop import train

    limit_threads(args.backend, args.threads)
    overrides = {"seed": args.seed} | ({"total_steps": args.steps} if args.steps else {})
    cfg = get_config(args.env, **overrides)
    env = make_env(cfg.env_id)
    agent = make_agent(args.backend, env.observation_space.shape[0], int(env.action_space.n), cfg)

    out_dir = os.path.join(args.out, f"{args.env}-{args.backend}-s{args.seed}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"training {args.backend} DQN on {args.env} for {cfg.total_steps} steps -> {out_dir}")
    result = train(agent, env, cfg, out_dir=out_dir)

    with open(os.path.join(out_dir, "log.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["episodes"][0]))
        writer.writeheader()
        writer.writerows(result["episodes"])
    summary = {"backend": args.backend, "config": asdict(cfg), "wall_time_s": result["wall_time_s"],
               "best_val_return": result["best_val_return"], "evals": result["evals"]}
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"done in {result['wall_time_s']:.0f} s, best validation return {result['best_val_return']:.2f}")


if __name__ == "__main__":
    main()
