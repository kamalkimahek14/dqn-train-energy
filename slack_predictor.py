# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Slack predictor on the coupled train: build the dataset, train the predictor, run the A/B.

    python slack_predictor.py data                        # roll out six policies, label every decision
    python slack_predictor.py train --backend torch       # or --backend tf
    python slack_predictor.py ab --backend torch --seeds 0 1   # DQN without vs with the filter, 50 test tracks
    python slack_predictor.py scenarios --backend torch   # the same A/B on 18 named scenarios, DQN seed 0

`data` writes runs/slack-data.npz, `train` writes runs/slack-<backend>/ and results/slack_predictor_<backend>.json,
`ab` writes results/slack_ab_<backend>.json and one recorded test track (results/slack_trace_<backend>.json),
`scenarios` writes results/slack_scenarios_<backend>.json.
Every track is seeded, so the same tracks and the same seeds go into every arm of the A/B.
"""
import argparse
import json
import os

import numpy as np

N_TRAIN, N_VAL, N_TEST = 200, 50, 50
# (name, policy, epsilon): the DQNs as they drive, noisier versions of them, the baseline, and pure random.
POLICIES = [("dqn-s0", "dqn0", 0.0), ("dqn-s0-eps0.2", "dqn0", 0.2), ("dqn-s1-eps0.4", "dqn1", 0.4),
            ("baseline", "baseline", 0.0), ("baseline-eps0.3", "baseline", 0.3), ("random", "baseline", 1.0)]


def split_seeds():
    from ecodrive.config import SLACK_TRAIN_SEED, TEST_SEED, VALIDATION_SEED
    return {"train": [SLACK_TRAIN_SEED + i for i in range(N_TRAIN)],
            "val": [VALIDATION_SEED + i for i in range(N_VAL)],
            "test": [TEST_SEED + i for i in range(N_TEST)]}


def load_dqn(backend, seed, runs="runs"):
    from ecodrive.agents import make_agent
    from ecodrive.config import get_config
    agent = make_agent(backend, 11, 12, get_config("train"))
    agent.load(os.path.join(runs, f"train-{backend}-s{seed}", "best"))
    return agent


def cmd_data(args):
    from ecodrive.baseline import baseline_action
    from ecodrive.coupled import CoupledTrainEnv
    from ecodrive.slack import collect, epsilon_policy

    env = CoupledTrainEnv()
    base = {"dqn0": load_dqn("torch", 0, args.runs).greedy, "dqn1": load_dqn("torch", 1, args.runs).greedy,
            "baseline": lambda obs: baseline_action(env)}
    parts = []
    for split, seeds in split_seeds().items():
        for p, (name, key, eps) in enumerate(POLICIES):
            for seed in seeds:
                rng = np.random.default_rng([seed, p])  # exploration noise is seeded per track and policy
                d = collect(env, epsilon_policy(base[key], eps, rng), seed)
                d["policy"] = np.full(len(d["action"]), p)
                d["split"] = np.full(len(d["action"]), split)
                parts.append(d)
            rows = [d for d in parts if d["split"][0] == split and d["policy"][0] == p]
            n = sum(len(d["action"]) for d in rows)
            ev = sum(int((d["draft"] | d["buff"]).sum()) for d in rows)
            print(f"{split:<6}{name:<17}{n:>7} decisions  {ev:>5} with an event ({100 * ev / n:.1f} %)")
    data = {k: np.concatenate([d[k] for d in parts]) for k in parts[0]}
    np.savez_compressed(os.path.join(args.runs, "slack-data.npz"), **data)


def load_split(path, split):
    from ecodrive.slack import to_xy
    raw = np.load(path)
    keep = raw["split"] == split
    data = {k: raw[k][keep] for k in ("hist", "action", "draft", "buff", "track", "policy")}
    return (*to_xy(data), data)


def cmd_train(args):
    from ecodrive.predictor import HIDDEN, make_net
    from ecodrive.slack import N_INPUTS, binary_metrics, fit, pick_threshold

    path = os.path.join(args.runs, "slack-data.npz")
    (x, y, d_tr), (xv, yv, d_va), (xt, yt, d_te) = (load_split(path, s) for s in ("train", "val", "test"))
    assert not (set(d_tr["track"]) & set(d_va["track"]) or set(d_tr["track"]) & set(d_te["track"])
                or set(d_va["track"]) & set(d_te["track"])), "a track is in two splits"
    print(f"train {len(y)} rows ({y.mean():.1%} events), validation {len(yv)}, test {len(yt)}")
    pred = fit(make_net(args.backend, N_INPUTS, seed=args.seed), x, y, xv, yv, epochs=args.epochs, seed=args.seed)
    out_dir = os.path.join(args.runs, f"slack-{args.backend}")
    os.makedirs(out_dir, exist_ok=True)
    pred.save(os.path.join(out_dir, "best"))

    threshold = pick_threshold(yv, pred.risk(xv), args.min_recall)  # chosen on validation only
    result = {"backend": args.backend, "hidden": HIDDEN, "inputs": N_INPUTS, "epochs": args.epochs,
              "pos_weight": pred.pos_weight, "min_recall": args.min_recall, "threshold": threshold,
              "rows": {"train": len(y), "val": len(yv), "test": len(yt)},
              "tracks": {"train": len(set(d_tr["track"])), "val": len(set(d_va["track"])), "test": len(set(d_te["track"]))},
              "val": binary_metrics(yv, pred.risk(xv), threshold),
              "test": binary_metrics(yt, pred.risk(xt), threshold),
              "test_at_0.5": binary_metrics(yt, pred.risk(xt), 0.5),
              "history": pred.history}
    m = result["test"]
    print(f"threshold {threshold:.3f}  test precision {m['precision']:.3f} recall {m['recall']:.3f}"
          f" F1 {m['f1']:.3f} ROC-AUC {m['roc_auc']:.4f}  {m['confusion']}")
    with open(os.path.join(args.results, f"slack_predictor_{args.backend}.json"), "w") as f:
        json.dump(result, f, indent=2)


def trip(env, policy, seed):
    from evaluate import rollout
    m, trace = rollout(env, policy, seed)
    m |= {"draft_events": env.draft_events, "buff_events": env.buff_events,
          "peak_force_kn": max(env.peak_draft, env.peak_buff) / 1e3}
    trace["force"] = env.force_trace  # (t, x, peak draft, peak buff) per 1 s substep
    return m, trace


def cmd_ab(args):
    from ecodrive.baseline import baseline_action
    from ecodrive.coupled import CoupledTrainEnv, F_EVENT
    from ecodrive.slack import SlackFilter, load_predictor
    from evaluate import summarize

    with open(os.path.join(args.results, f"slack_predictor_{args.backend}.json")) as f:
        threshold = json.load(f)["threshold"]
    env = CoupledTrainEnv()
    predictor = load_predictor(args.backend, args.runs)
    tracks = split_seeds()["test"]
    trace_track = tracks[args.trace_track]

    def arm(policy_for_track):
        rows, traces = [], {}
        for t in tracks:
            policy, filt = policy_for_track()
            m, traces[t] = trip(env, policy, t)
            m["overrides"] = filt.overrides if filt else 0
            rows.append(m)
        out = summarize(rows)
        out["total_draft_events"] = int(sum(r["draft_events"] for r in rows))
        out["total_buff_events"] = int(sum(r["buff_events"] for r in rows))
        out["max_peak_force_kn"] = float(max(r["peak_force_kn"] for r in rows))
        out["trips_with_an_event"] = int(sum(r["draft_events"] + r["buff_events"] > 0 for r in rows))
        return out, traces

    result = {"threshold": threshold, "event_force_kn": F_EVENT / 1e3, "tracks": tracks, "per_seed": {}}
    result["baseline"], _ = arm(lambda: (lambda obs: baseline_action(env), None))
    record = {}
    for seed in args.seeds:
        agent = load_dqn(args.backend, seed, args.runs)

        def filtered():
            filt = SlackFilter(agent, predictor, env, threshold)
            return filt.greedy, filt

        plain, plain_traces = arm(lambda: (agent.greedy, None))
        safe, safe_traces = arm(filtered)
        result["per_seed"][seed] = {"dqn": plain, "dqn_filter": safe}
        if seed == args.seeds[0]:
            env.reset(seed=trace_track)
            record = {"track": trace_track, "training_seed": seed, "threshold": threshold,
                      "length": env.track.length, "limit_x": env.track.limit_x.tolist(),
                      "limit": env.track.limit.tolist(),
                      "event_force_kn": F_EVENT / 1e3,
                      "dqn": plain_traces[trace_track], "dqn_filter": safe_traces[trace_track]}
        for name, r in (("DQN", plain), ("DQN + filter", safe)):
            print(f"seed {seed} {name:<13} draft {r['draft_events']['mean']:.2f} buff {r['buff_events']['mean']:.2f}"
                  f" /trip  peak {r['peak_force_kn']['mean']:.0f} kN (max {r['max_peak_force_kn']:.0f})"
                  f"  fuel {r['fuel_l']['mean']:.1f} L  on time {r['on_time']['mean']:.0%}"
                  f"  over limit {r['violation_steps']['mean']:.2f}  overrides {r['overrides']['mean']:.1f}")
    b = result["baseline"]
    print(f"baseline             draft {b['draft_events']['mean']:.2f} buff {b['buff_events']['mean']:.2f}"
          f" /trip  peak {b['peak_force_kn']['mean']:.0f} kN  fuel {b['fuel_l']['mean']:.1f} L"
          f"  on time {b['on_time']['mean']:.0%}  over limit {b['violation_steps']['mean']:.2f}")
    with open(os.path.join(args.results, f"slack_ab_{args.backend}.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.results, f"slack_trace_{args.backend}.json"), "w") as f:
        json.dump(record, f)


def cmd_scenarios(args):
    from ecodrive.scenarios import SCENARIOS
    from ecodrive.slack import SlackFilter, load_predictor

    with open(os.path.join(args.results, f"slack_predictor_{args.backend}.json")) as f:
        threshold = json.load(f)["threshold"]
    predictor = load_predictor(args.backend, args.runs)
    seed = args.seeds[0]
    agent = load_dqn(args.backend, seed, args.runs)
    keys = ("draft_events", "buff_events", "peak_force_kn", "fuel_l", "time_s", "on_time", "violation_steps")
    rows = []
    for sc in SCENARIOS:
        env = sc.make_env()
        filt = SlackFilter(agent, predictor, env, threshold)
        row = {"name": sc.name, "train": sc.train, "power": sc.power, "section": sc.section, "grade": sc.grade}
        for arm, policy in (("dqn", agent.greedy), ("dqn_filter", filt.greedy)):
            m, _ = trip(env, policy, 0)  # the section is fixed, so the reset seed changes nothing
            row[arm] = {k: m[k] for k in keys} | {"arrived": env.x >= env.track.length,
                                                  "distance_m": min(env.x, env.track.length),
                                                  "window_s": [env.t_early, env.t_late]}
        row["dqn_filter"]["overrides"] = filt.overrides
        rows.append(row)
        a, b = row["dqn"], row["dqn_filter"]
        print(f"{sc.name:<44} events {a['draft_events']}+{a['buff_events']} -> {b['draft_events']}+{b['buff_events']}"
              f"  peak {a['peak_force_kn']:.0f} -> {b['peak_force_kn']:.0f} kN"
              f"  fuel {a['fuel_l']:.1f} -> {b['fuel_l']:.1f} L  time {a['time_s']:.0f} -> {b['time_s']:.0f} s"
              f"  km {a['distance_m'] / 1e3:.1f} -> {b['distance_m'] / 1e3:.1f}  overrides {b['overrides']}")
    total = {arm: sum(r[arm]["draft_events"] + r[arm]["buff_events"] for r in rows) for arm in ("dqn", "dqn_filter")}
    print(f"slack events over {len(rows)} scenarios: {total['dqn']} without the filter, {total['dqn_filter']} with it")
    with open(os.path.join(args.results, f"slack_scenarios_{args.backend}.json"), "w") as f:
        json.dump({"backend": args.backend, "training_seed": seed, "threshold": threshold, "scenarios": rows,
                   "total_events": total}, f, indent=2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["data", "train", "ab", "scenarios"])
    p.add_argument("--backend", choices=["torch", "tf"], default="torch")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1], help="DQN training seeds for the A/B")
    p.add_argument("--seed", type=int, default=0, help="predictor initialisation and shuffling seed")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--min-recall", type=float, default=0.9, help="validation recall the threshold must keep")
    p.add_argument("--trace-track", type=int, default=0, help="test track (0-based) to record for plots.py")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    args = p.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from ecodrive.agents import limit_threads
    limit_threads("torch" if args.step == "data" else args.backend, args.threads)
    os.makedirs(args.results, exist_ok=True)
    {"data": cmd_data, "train": cmd_train, "ab": cmd_ab, "scenarios": cmd_scenarios}[args.step](args)


if __name__ == "__main__":
    main()
