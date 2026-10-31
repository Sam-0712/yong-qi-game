"""序列编码器：用 (own, opp) 出招序列预测信念因子（把手写因子蒸馏进小网络）。

两种结构（可都试）：
  - CNN : 因果窗口（最后 W 步）→ 隐层 → 14 因子
  - GRU : 逐步循环 → 每步输出 14 因子

纯 numpy，参数小，可导出 JSON。监督目标 = BeliefTracker 在同序列上的因子。
之后可用对局结果做在线微调，替代手写因子（面向真人/未知风格）。
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Tuple

import numpy as np

from ..core.rules import MOVE_TO_IDX, MOVES
from .belief import BeliefTracker

FACTOR_KEYS = [
    "long_atk", "long_heavy", "long_def", "long_dodge", "long_exhale", "long_noqi",
    "short_atk", "attack_streak", "switch", "predictability", "mirror",
    "volatility", "escalation", "threat",
]
F = len(FACTOR_KEYS)
D_IN = 2 * len(MOVES)  # own one-hot + opp one-hot


def _onehot(move: str) -> np.ndarray:
    v = np.zeros(len(MOVES), dtype=np.float64)
    idx = MOVE_TO_IDX.get(move, 0)
    if idx:
        v[idx - 1] = 1.0
    return v


def encode_step(own: str, opp: str) -> np.ndarray:
    return np.concatenate([_onehot(own), _onehot(opp)])


def factors_vec(tracker: BeliefTracker) -> np.ndarray:
    f = tracker.factors()
    return np.array([f[k] for k in FACTOR_KEYS], dtype=np.float64)


def sequence_to_xy(own_seq: List[str], opp_seq: List[str],
                   tracker_kwargs: Dict | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """把一条对局序列转成 (X[T,18], Y[T,14])，Y 由 BeliefTracker 现算。"""
    tr = BeliefTracker(**(tracker_kwargs or {}))
    X = np.zeros((len(own_seq), D_IN))
    Y = np.zeros((len(own_seq), F))
    for t, (a, b) in enumerate(zip(own_seq, opp_seq)):
        tr.observe(a, b)
        X[t] = encode_step(a, b)
        Y[t] = factors_vec(tr)
    return X, Y


# ── Adam ──
class _Adam:
    def __init__(self, params: Dict[str, np.ndarray], lr=0.01):
        self.lr, self.b1, self.b2, self.eps, self.t = lr, 0.9, 0.999, 1e-8, 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


# ── CNN（因果窗口 MLP）──
class CNNEncoder:
    def __init__(self, window: int = 4, hidden: int = 64, seed: int = 0, lr: float = 0.01):
        rng = np.random.default_rng(seed)
        self.window, self.hidden, self.lr = window, hidden, lr
        self.params = {
            "W1": rng.normal(0, 0.1, (hidden, window * D_IN)),
            "b1": np.zeros(hidden),
            "W2": rng.normal(0, 0.1, (F, hidden)),
            "b2": np.zeros(F),
        }
        self.opt = _Adam(self.params, lr)

    def _windows(self, X: np.ndarray) -> Tuple[np.ndarray, int]:
        T = X.shape[0]
        pad = np.zeros((self.window - 1, D_IN))
        Xp = np.vstack([pad, X])
        W = np.zeros((T, self.window * D_IN))
        for t in range(T):
            W[t] = Xp[t:t + self.window].reshape(-1)
        return W, T

    def forward(self, W: np.ndarray):
        p = self.params
        h = np.maximum(0.0, W @ p["W1"].T + p["b1"])
        yhat = _sigmoid(h @ p["W2"].T + p["b2"])
        return h, yhat

    def fit_step(self, W, Y):
        p = self.params
        h, yhat = self.forward(W)
        diff = (yhat - Y) / len(Y)
        g = {"W2": diff.T @ h, "b2": diff.sum(0)}
        dh = (diff @ p["W2"]) * (h > 0)
        g["W1"] = dh.T @ W
        g["b1"] = dh.sum(0)
        self.opt.step(p, g)
        return float(((yhat - Y) ** 2).mean())

    def predict_seq(self, X):
        W, _ = self._windows(X)
        _, yhat = self.forward(W)
        return yhat

    def n_params(self):
        return sum(v.size for v in self.params.values())


# ── GRU ──
class GRUEncoder:
    def __init__(self, hidden: int = 24, seed: int = 0, lr: float = 0.01):
        rng = np.random.default_rng(seed)
        self.H, self.lr = hidden, lr

        def nz(*s):
            return rng.normal(0, 0.1, s)
        self.params = {
            "Wz": nz(hidden, D_IN), "Uz": nz(hidden, hidden), "bz": np.zeros(hidden),
            "Wr": nz(hidden, D_IN), "Ur": nz(hidden, hidden), "br": np.zeros(hidden),
            "Wn": nz(hidden, D_IN), "Un": nz(hidden, hidden), "bn": np.zeros(hidden),
            "Wo": nz(F, hidden), "bo": np.zeros(F),
        }
        self.opt = _Adam(self.params, lr)

    def forward(self, X):
        p, H = self.params, self.H
        T = X.shape[0]
        hs = np.zeros((T + 1, H))
        zs = np.zeros((T, H)); rs = np.zeros((T, H)); ns = np.zeros((T, H))
        for t in range(T):
            x = X[t]; hp = hs[t]
            z = _sigmoid(p["Wz"] @ x + p["Uz"] @ hp + p["bz"])
            r = _sigmoid(p["Wr"] @ x + p["Ur"] @ hp + p["br"])
            n = np.tanh(p["Wn"] @ x + p["Un"] @ (r * hp) + p["bn"])
            hs[t + 1] = (1 - z) * n + z * hp
            zs[t], rs[t], ns[t] = z, r, n
        yhat = _sigmoid(hs[1:] @ p["Wo"].T + p["bo"])
        return hs, zs, rs, ns, yhat

    def fit_step(self, X, Y):
        p, H = self.params, self.H
        hs, zs, rs, ns, yhat = self.forward(X)
        T = X.shape[0]
        diff = (yhat - Y) / T
        g = {k: np.zeros_like(v) for k, v in p.items()}
        g["Wo"] = diff.T @ hs[1:]; g["bo"] = diff.sum(0)
        dh_next = np.zeros(H)
        dh_prev = diff @ p["Wo"]  # dL/dh_t (from output)
        # backprop through time
        for t in reversed(range(T)):
            hp = hs[t]; z = zs[t]; r = rs[t]; n = ns[t]
            dh = dh_prev[t] + dh_next
            dn = dh * (1 - z)
            dz = dh * (hp - n)
            dhp = dh * z
            dan = dn * (1 - n * n)
            g["Wn"] += np.outer(dan, X[t]); g["Un"] += np.outer(dan, r * hp); g["bn"] += dan
            d_rp = (p["Un"].T @ dan) * hp
            dr = d_rp * r * (1 - r)
            g["Wr"] += np.outer(dr, X[t]); g["Ur"] += np.outer(dr, hp); g["br"] += dr
            dhp += (p["Ur"].T @ dr)
            daz = dz * z * (1 - z)
            g["Wz"] += np.outer(daz, X[t]); g["Uz"] += np.outer(daz, hp); g["bz"] += daz
            dhp += (p["Uz"].T @ daz)
            dh_next = dhp
        self.opt.step(p, g)
        return float(((yhat - Y) ** 2).mean())

    def predict_seq(self, X):
        _, _, _, _, yhat = self.forward(X)
        return yhat

    def n_params(self):
        return sum(v.size for v in self.params.values())


def rmse_per_factor(pred: np.ndarray, Y: np.ndarray) -> Dict[str, float]:
    err = np.sqrt(((pred - Y) ** 2).mean(0))
    return {k: float(err[i]) for i, k in enumerate(FACTOR_KEYS)}
