# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Draw the README figures from runs/ and results/.

    python plots.py                    # assets/learning_curves.png and assets/episode.gif
    python plots.py --backend tf       # animate the TensorFlow agent's recorded episode instead
"""
import argparse
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

BG, PANEL, FG, GRID = "#0d1117", "#161b22", "#e6edf3", "#30363d"
COLOR = {"torch": "#22d3ee", "tf": "#f59e0b", "baseline": "#9ca3af", "limit": "#f87171"}
NAME = {"torch": "PyTorch", "tf": "TensorFlow"}


def dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "axes.edgecolor": GRID, "axes.labelcolor": FG,
        "text.color": FG, "xtick.color": FG, "ytick.color": FG, "grid.color": GRID,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID, "font.size": 10})


def moving_average(y, k):
    k = max(1, min(k, len(y)))
    return np.convolve(y, np.ones(k) / k, mode="valid"), k


def learning_curves(runs, results, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for ax, env_id, title in zip(axes, ["train", "CartPole-v1"], ["Train driving task", "CartPole-v1 (sanity check)"]):
        for backend in ("torch", "tf"):
            for i, run in enumerate(sorted(glob.glob(os.path.join(runs, f"{env_id}-{backend}-s*")))):
                if not os.path.exists(os.path.join(run, "log.csv")):
                    continue
                with open(os.path.join(run, "log.csv")) as f:
                    rows = list(csv.DictReader(f))
                steps = np.array([int(r["step"]) for r in rows])
                avg, k = moving_average(np.array([float(r["return"]) for r in rows]), 50)
                seed = run.rsplit("-s", 1)[1]
                # First seed solid, later seeds dashed; validation points only for the first to keep it readable.
                ax.plot(steps[k - 1:], avg, color=COLOR[backend], lw=1.4, ls="-" if i == 0 else "--",
                        label=f"{NAME[backend]} seed {seed}: training episodes")
                if i == 0:
                    with open(os.path.join(run, "summary.json")) as f:
                        evals = json.load(f)["evals"]
                    ax.plot([e["step"] for e in evals], [e["val_return"] for e in evals], "o", ms=3.5,
                            color=COLOR[backend], markerfacecolor=PANEL,
                            label=f"{NAME[backend]} seed {seed}: validation (greedy)")
        base = os.path.join(results, "eval_train_torch.json")
        if env_id == "train" and os.path.exists(base):
            with open(base) as f:
                ax.axhline(json.load(f)["baseline"]["return"]["mean"], ls=":", lw=1.6, color=COLOR["baseline"],
                           label="rule-based baseline")
        ax.set(title=title, xlabel="environment steps", ylabel="episode return")
        ax.grid(alpha=0.6)
        ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


def animate_episode(rollout_path, backend, out):
    with open(rollout_path) as f:
        tr = json.load(f)
    dqn, base = tr["dqn"], tr["baseline"]
    xd, vd = np.array(dqn["x"]) / 1000, np.array(dqn["v"]) * 3.6        # km and km/h
    xb, vb = np.array(base["x"]) / 1000, np.array(base["v"]) * 3.6
    # Control level per decision: +notch/8 for throttle, -level/3 for brake.
    u = np.array([a / 8 if a <= 8 else -(a - 8) / 3 for a in dqn["action"]])
    ub = np.array([a / 8 if a <= 8 else -(a - 8) / 3 for a in base["action"]])
    n = len(u)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.8), sharex=True, gridspec_kw={"height_ratios": [3, 1.5]})
    edges = np.append(tr["limit_x"], tr["length"]) / 1000
    ax1.stairs(np.array(tr["limit"]) * 3.6, edges, baseline=None, color=COLOR["limit"], lw=1.6, label="speed limit")
    ax1.plot(xb, vb, color=COLOR["baseline"], lw=1.2, label="baseline: hold the limit")
    line, = ax1.plot([], [], color=COLOR[backend], lw=2.2, label=f"DQN ({NAME[backend]})")
    dot, = ax1.plot([], [], "o", color=COLOR[backend])
    ax1.set(ylim=(0, max(tr["limit"]) * 3.6 + 15), ylabel="speed (km/h)")
    ax1.legend(fontsize=8, loc="lower center", ncol=3)
    ax1.grid(alpha=0.6)
    text = ax1.text(0.01, 0.97, "", transform=ax1.transAxes, va="top", fontsize=9, family="monospace")

    ax2.stairs(ub, xb, color=COLOR["baseline"], lw=1, alpha=0.7)
    steps = ax2.stairs(u[:1], xd[:2], color=COLOR[backend], lw=1.8)
    ax2.set(ylim=(-1.15, 1.15), xlim=(0, tr["length"] / 1000), xlabel="position (km)")
    ax2.set_yticks([-1, 0, 1], ["brake 3", "coast", "notch 8"])
    ax2.grid(alpha=0.6)
    fig.tight_layout()

    fuel_base_at = lambda x: np.interp(x, xb, base["fuel"])  # baseline fuel at the same position

    def draw(i):
        k = min(i, n)
        line.set_data(xd[:k + 1], vd[:k + 1])
        dot.set_data(xd[k:k + 1], vd[k:k + 1])
        if k > 0:
            steps.set_data(u[:k], xd[:k + 1])
        text.set_text(f"t = {dqn['t'][k]:4.0f} s   fuel so far: DQN {dqn['fuel'][k]:5.1f} L"
                      f" | baseline {fuel_base_at(xd[k]):5.1f} L")
        return line, dot, steps, text

    anim = FuncAnimation(fig, draw, frames=n + 1 + 16, blit=False)  # 16 extra frames hold the end
    anim.save(out, writer=PillowWriter(fps=8), dpi=80)
    print("wrote", out, f"({os.path.getsize(out) / 1e6:.2f} MB)")


SLACK = {"dqn": "#d95926", "dqn_filter": "#3987e5", "draft": "#9085e9", "buff": "#199e70", "muted": "#8b949e"}


def slack_trace(path, out):
    """Speed and coupler force along one test track, DQN without and with the slack filter."""
    with open(path) as f:
        tr = json.load(f)
    thr = tr["event_force_kn"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
    edges = np.append(tr["limit_x"], tr["length"]) / 1000
    ax1.stairs(np.array(tr["limit"]) * 3.6, edges, baseline=None, color=COLOR["limit"], lw=1.4, label="speed limit")
    runs = {}
    for key, label in (("dqn", "DQN"), ("dqn_filter", "DQN + slack filter")):
        run = tr[key]
        ax1.plot(np.array(run["x"]) / 1000, np.array(run["v"]) * 3.6, color=SLACK[key], lw=1.6, label=label)
        t, x, draft, buff = (np.array(c) for c in zip(*run["force"]))
        runs[key] = (x / 1000, draft / 1e3, buff / 1e3)
        n = sum(1 for i in range(0, len(t), 10) if max(draft[i:i + 10].max(), buff[i:i + 10].max()) > thr * 1e3)
        signed = np.where(draft >= buff, draft, -buff) / 1e3  # the larger of the two, per second
        ax2.plot(x / 1000, signed, color=SLACK[key], lw=1.3,
                 label=f"{label}: {n} decision{'s' * (n != 1)} over the line")
    ax1.set(ylabel="speed (km/h)", ylim=(0, max(tr["limit"]) * 3.6 + 15))
    ax1.legend(fontsize=8, loc="lower center", ncol=3)
    ax1.grid(alpha=0.6)

    for sign in (1, -1):
        ax2.axhline(sign * thr, ls="--", lw=1.2, color=SLACK["muted"])
    ax2.text(0.05, thr + 12, f"event line: {thr:.0f} kN draft", color=SLACK["muted"], fontsize=8)
    ax2.text(0.05, -thr - 32, f"event line: {thr:.0f} kN buff", color=SLACK["muted"], fontsize=8)
    # Mark each stretch where the plain DQN crossed the line and the filtered run, at the same place, did not.
    xp, dp, bp = runs["dqn"]
    xf, df, bf = runs["dqn_filter"]
    over = np.flatnonzero(np.maximum(dp, bp) > thr)
    groups = np.split(over, np.flatnonzero(np.diff(xp[over]) > 0.3) + 1) if len(over) else []
    for g in groups:
        i = g[np.argmax(np.maximum(dp[g], bp[g]))]
        near = np.abs(xf - xp[i]) < 0.2
        if near.any() and np.maximum(df[near], bf[near]).max() <= thr:
            y = dp[i] if dp[i] >= bp[i] else -bp[i]
            ax2.annotate("prevented", (xp[i], y), xytext=(xp[i] + 0.3, y + np.sign(y) * 45), fontsize=8,
                         color=FG, arrowprops={"arrowstyle": "->", "color": FG, "lw": 0.8})
    ax2.set(ylabel="peak coupler force (kN)\n+ draft, − buff", xlabel="position of the lead locomotive (km)",
            xlim=(0, tr["length"] / 1000), ylim=(-thr - 140, max(dp.max(), df.max(), thr) + 90))
    ax2.legend(fontsize=8, loc="lower center")
    ax2.grid(alpha=0.6)
    fig.suptitle(f"Test track {tr['track']}, PyTorch DQN seed {tr['training_seed']} on the coupled train",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


def slack_events(results, out):
    """Slack events per trip on the 50 test tracks: baseline, then each DQN without and with the filter."""
    bars = []
    for backend in ("torch", "tf"):
        path = os.path.join(results, f"slack_ab_{backend}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            ab = json.load(f)
        if not bars:
            bars.append(("Baseline", ab["baseline"]["draft_events"]["mean"], ab["baseline"]["buff_events"]["mean"]))
        seeds = list(ab["per_seed"].values())
        for key, label in (("dqn", f"DQN\n{NAME[backend]}"), ("dqn_filter", f"DQN {NAME[backend]}\n+ slack filter")):
            bars.append((label, np.mean([s[key]["draft_events"]["mean"] for s in seeds]),
                         np.mean([s[key]["buff_events"]["mean"] for s in seeds])))
    labels, draft, buff = zip(*bars)
    x = np.arange(len(bars))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x, draft, width=0.6, color=SLACK["draft"], label="draft (stretch)", edgecolor=PANEL, linewidth=2)
    ax.bar(x, buff, width=0.6, bottom=draft, color=SLACK["buff"], label="buff (compression)",
           edgecolor=PANEL, linewidth=2)
    for xi, d, b in zip(x, draft, buff):
        ax.text(xi, d + b + 0.06, f"{d + b:.2f}", ha="center", fontsize=9, color=FG)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set(ylabel="slack events per trip", ylim=(0, max(np.add(draft, buff)) * 1.2 + 0.1))
    ax.set_title("Decisions with a coupler past the event line, mean over 50 test tracks and 2 DQN seeds",
                 fontsize=9, loc="left")
    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("wrote", out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["torch", "tf"], default="torch")
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--assets", default="assets")
    p.add_argument("--slack", action="store_true", help="only the slack-predictor figures")
    args = p.parse_args()
    os.makedirs(args.assets, exist_ok=True)
    dark_style()
    if args.slack:
        slack_trace(os.path.join(args.results, f"slack_trace_{args.backend}.json"),
                    os.path.join(args.assets, "slack_trace.png"))
        slack_events(args.results, os.path.join(args.assets, "slack_events.png"))
        return
    learning_curves(args.runs, args.results, os.path.join(args.assets, "learning_curves.png"))
    rollout = os.path.join(args.results, f"rollout_{args.backend}.json")
    if os.path.exists(rollout):
        animate_episode(rollout, args.backend, os.path.join(args.assets, "episode.gif"))


if __name__ == "__main__":
    main()
