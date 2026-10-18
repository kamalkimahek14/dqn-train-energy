# Copyright (c) 2026 Kamal Mehta. MIT License.
"""Slack predictor: the inputs it sees, the dataset, training, the metrics, and the safety filter.

The predictor answers one question per candidate action: if the train takes this action now, will a
coupler go past F_EVENT (draft or buff) during the next 10 s decision? The filter asks it about all 12
actions and lets the DQN pick the best one that the predictor does not flag.
"""
from __future__ import annotations

import os

import numpy as np

from .coupled import F_EVENT, N_FEATURES, WINDOW
from .env import N_ACTIONS, action_to_control

ACTION_U = np.array([t - b for t, b in map(action_to_control, range(N_ACTIONS))], dtype=np.float32)
N_INPUTS = WINDOW * N_FEATURES + N_ACTIONS + 2


def inputs(hist: np.ndarray, actions) -> np.ndarray:
    """Rows of [flattened history window, one-hot action, control level of the action, its change]."""
    actions = np.asarray(actions)
    u_prev = hist[-1, 5]  # the last control level, from the newest row of the window
    flat = np.broadcast_to(hist.reshape(-1), (len(actions), hist.size))
    return np.hstack([flat, np.eye(N_ACTIONS, dtype=np.float32)[actions],
                      ACTION_U[actions, None], ACTION_U[actions, None] - u_prev]).astype(np.float32)


# -- dataset ------------------------------------------------------------------------------------
def collect(env, policy, seed: int) -> dict:
    """Play one trip on track `seed`; one row per decision: the window, the action, and its outcome."""
    obs, _ = env.reset(seed=seed)
    hists, actions, draft, buff, done = [], [], [], [], False
    while not done:
        hists.append(env.slack_features())
        actions.append(policy(obs))
        obs, _, done, _, _ = env.step(actions[-1])
        draft.append(env.step_draft > F_EVENT)
        buff.append(env.step_buff > F_EVENT)
    return {"hist": np.array(hists), "action": np.array(actions), "draft": np.array(draft),
            "buff": np.array(buff), "track": np.full(len(actions), seed)}


def epsilon_policy(policy, eps: float, rng, n_actions: int = N_ACTIONS):
    """With probability eps a uniformly random action, otherwise the wrapped policy's choice."""
    return lambda obs: int(rng.integers(n_actions)) if rng.random() < eps else int(policy(obs))


def to_xy(data: dict) -> tuple[np.ndarray, np.ndarray]:
    x = np.concatenate([inputs(h, [a]) for h, a in zip(data["hist"], data["action"])])
    return x, (data["draft"] | data["buff"]).astype(np.float32)


# -- model --------------------------------------------------------------------------------------
class SlackPredictor:
    """A backend net plus the input standardisation fitted on the training split."""

    def __init__(self, net, mean: np.ndarray, std: np.ndarray):
        self.net, self.mean, self.std = net, mean.astype(np.float32), std.astype(np.float32)

    def risk(self, x: np.ndarray) -> np.ndarray:
        return self.net.predict_proba((x - self.mean) / self.std)

    def save(self, path: str):
        self.net.save(path)
        np.savez(path + ".norm.npz", mean=self.mean, std=self.std)

    @classmethod
    def load(cls, net, path: str) -> "SlackPredictor":
        net.load(path)
        norm = np.load(path + ".norm.npz")
        return cls(net, norm["mean"], norm["std"])


def fit(net, x, y, x_val, y_val, epochs: int = 30, batch: int = 256, seed: int = 0, log=print) -> SlackPredictor:
    """Weighted cross-entropy, Adam, shuffled minibatches; keeps the epoch with the lowest validation loss."""
    mean, std = x.mean(0), x.std(0) + 1e-6
    xs, xv = (x - mean) / std, (x_val - mean) / std
    pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))  # balances the two classes in the loss
    rng = np.random.default_rng(seed)
    best, best_w, history = np.inf, None, []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(xs))
        train_loss = np.mean([net.update(xs[i], y[i], pos_weight) for i in np.array_split(order, max(1, len(xs) // batch))])
        val_loss = net.eval_loss(xv, y_val, pos_weight)
        history.append({"epoch": epoch, "train_loss": float(train_loss), "val_loss": val_loss})
        if val_loss < best:
            best, best_w = val_loss, net.get_weights()
        log(f"epoch {epoch:>3}  train loss {train_loss:.4f}  val loss {val_loss:.4f}")
    net.set_weights(best_w)
    pred = SlackPredictor(net, mean, std)
    pred.history, pred.pos_weight = history, pos_weight
    return pred


# -- metrics ------------------------------------------------------------------------------------
def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    _, inv, counts = np.unique(score, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(counts) - (counts - 1) / 2)[inv]  # 1-based ranks, averaged over ties
    pos = y == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict:
    flag = score >= threshold
    tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
    fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"threshold": threshold, "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12), "roc_auc": roc_auc(y, score),
            "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn}, "positives": int(y.sum()), "n": len(y)}


def pick_threshold(y_val: np.ndarray, score_val: np.ndarray, min_recall: float = 0.9) -> float:
    """The highest threshold that still flags at least min_recall of the validation events."""
    pos = np.sort(score_val[y_val == 1])[::-1]
    return float(pos[int(np.ceil(min_recall * len(pos))) - 1])


# -- the second opinion -------------------------------------------------------------------------
def choose(q: np.ndarray, risk: np.ndarray, threshold: float) -> int:
    """Highest-Q action whose risk is under the threshold; if none is, the lowest-risk action."""
    safe = risk < threshold
    if safe.any():
        return int(np.argmax(np.where(safe, q, -np.inf)))
    return int(np.argmin(risk))


class SlackFilter:
    """Wraps a DQN agent: same greedy(obs) call, but every candidate action is scored by the predictor first.
    The agent's own code is not touched."""

    def __init__(self, agent, predictor: SlackPredictor, env, threshold: float):
        self.agent, self.predictor, self.env, self.threshold = agent, predictor, env, threshold
        self.overrides = 0  # decisions where the filter changed the DQN's choice

    def greedy(self, obs: np.ndarray) -> int:
        q = self.agent.q_values(obs[None])[0]
        risk = self.predictor.risk(inputs(self.env.slack_features(), np.arange(len(q))))
        action = choose(q, risk, self.threshold)
        self.overrides += action != int(np.argmax(q))
        return action


def load_predictor(backend: str, runs: str = "runs", which: str = "best") -> SlackPredictor:
    from .predictor import make_net
    return SlackPredictor.load(make_net(backend, N_INPUTS), os.path.join(runs, f"slack-{backend}", which))
