# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Draw the README figures from runs/ and results/.

    python plots.py                    # assets/learning_curves.png and assets/episode.gif
    python plots.py --backend tf       # animate the TensorFlow agent's recorded episode instead
"""
import argparse
import csv
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
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, env_id, title in zip(axes, ["train", "CartPole-v1"], ["Train driving task", "CartPole-v1 (sanity check)"]):
        for backend in ("torch", "tf"):
            run = os.path.join(runs, f"{env_id}-{backend}")
            if not os.path.exists(os.path.join(run, "log.csv")):
                continue
            with open(os.path.join(run, "log.csv")) as f:
                rows = list(csv.DictReader(f))
            steps = np.array([int(r["step"]) for r in rows])
            avg, k = moving_average(np.array([float(r["return"]) for r in rows]), 50)
            ax.plot(steps[k - 1:], avg, color=COLOR[backend], lw=1.6, label=f"{NAME[backend]}: training episodes")
            with open(os.path.join(run, "summary.json")) as f:
                evals = json.load(f)["evals"]
            ax.plot([e["step"] for e in evals], [e["val_return"] for e in evals], "o", ms=4,
                    color=COLOR[backend], markerfacecolor=PANEL, label=f"{NAME[backend]}: validation (greedy)")
        base = os.path.join(results, "eval_train_torch.json")
        if env_id == "train" and os.path.exists(base):
            with open(base) as f:
                ax.axhline(json.load(f)["baseline"]["return"]["mean"], ls="--", color=COLOR["baseline"],
                           label="rule-based baseline")
        ax.set(title=title, xlabel="environment steps", ylabel="episode return")
        ax.grid(alpha=0.6)
        ax.legend(fontsize=8, loc="lower right")
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["torch", "tf"], default="torch")
    p.add_argument("--runs", default="runs")
    p.add_argument("--results", default="results")
    p.add_argument("--assets", default="assets")
    args = p.parse_args()
    os.makedirs(args.assets, exist_ok=True)
    dark_style()
    learning_curves(args.runs, args.results, os.path.join(args.assets, "learning_curves.png"))
    rollout = os.path.join(args.results, f"rollout_{args.backend}.json")
    if os.path.exists(rollout):
        animate_episode(rollout, args.backend, os.path.join(args.assets, "episode.gif"))


if __name__ == "__main__":
    main()
