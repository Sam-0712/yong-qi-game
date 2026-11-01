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
from .belief import BeliefTracker, CATEGORIES as CATEGORIES_OF, CATEGORIES, CATS

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
                   tracker_kwargs: Dict | None = None,
                   max_len: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """把一条对局序列转成 (X[T,18], Y[T,14])，Y 由 BeliefTracker 现算。

    max_len：只保留最后 T 步的输入（在线 EncoderBelief 也只看最近 max_len 步），
    但 Y 仍按**完整历史**计算——这样训练目标与在线手写信念一致，不会因为
    截断输入而把早期历史的衰减计入丢失。
    """
    tr = BeliefTracker(**(tracker_kwargs or {}))
    X = np.zeros((len(own_seq), D_IN))
    Y = np.zeros((len(own_seq), F))
    for t, (a, b) in enumerate(zip(own_seq, opp_seq)):
        tr.observe(a, b)
        X[t] = encode_step(a, b)
        Y[t] = factors_vec(tr)
    if max_len is not None and len(own_seq) > max_len:
        X = X[-max_len:]
        Y = Y[-max_len:]
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
        self.arch = "cnn"
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

    def fit_step(self, W, Y, weight: float = 1.0):
        p = self.params
        h, yhat = self.forward(W)
        diff = (yhat - Y) / len(Y) * weight
        g = {"W2": diff.T @ h, "b2": diff.sum(0)}
        dh = (diff @ p["W2"]) * (h > 0)
        g["W1"] = dh.T @ W
        g["b1"] = dh.sum(0)
        self.opt.step(p, g)
        return float((((yhat - Y) ** 2).mean()) * weight)

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
        self.arch = "gru"
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

    def fit_step(self, X, Y, weight: float = 1.0):
        p, H = self.params, self.H
        hs, zs, rs, ns, yhat = self.forward(X)
        T = X.shape[0]
        diff = (yhat - Y) / T * weight
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


# ── 加载 & 推理适配（把编码器接到 BR 上）──

def load_encoder(path: str):
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)
    arch = blob.get("arch", "cnn")
    if arch == "cnn":
        m = CNNEncoder(window=int(blob["window"]), hidden=int(blob["hidden"]))
    else:
        m = GRUEncoder(hidden=int(blob["hidden"]))
    for k in m.params:
        m.params[k] = np.array(blob[k], dtype=np.float64)
    m.arch = arch
    return m


def predict_factors(model, own_hist: List[str], opp_hist: List[str]) -> Dict[str, float]:
    X = np.array([encode_step(a, b) for a, b in zip(own_hist, opp_hist)])
    arch = getattr(model, "arch", "cnn")
    if arch == "cnn":
        W, _ = model._windows(X)
        _, yhat = model.forward(W)
    elif arch == "moe":
        yhat = model.experts[model._route(opp_hist)].forward(X)[4]
    else:
        yhat = model.forward(X)[4]
    row = yhat[-1]
    return {k: float(row[i]) for i, k in enumerate(FACTOR_KEYS)}


# ── 风格路由 + MoE（多个小 GRU + 规则 router，按风格分派）──

def route_style(opp_hist: List[str], k: int = 3) -> int:
    """按最近 k 步对手动作的主导类别路由：0=defense 1=attack 2=heavy 3=mixed。"""
    recent = opp_hist[-k:]
    cats = [CATEGORIES.get(m, "attack") for m in recent]
    cnt = {"defense": cats.count("defense"),
           "attack": cats.count("attack"),
           "heavy": cats.count("heavy")}
    mx = max(cnt.values())
    if mx == 0:
        return 3
    tops = [c for c, v in cnt.items() if v == mx]
    if len(tops) != 1:
        return 3
    return {"defense": 0, "attack": 1, "heavy": 2}[tops[0]]


class MoEStyleEncoder:
    """按风格路由到 E 个「小 GRU 专家」；每个专家 n 很小，总参数可控。"""

    def __init__(self, n_experts: int = 4, hidden: int = 10, seed: int = 0, lr: float = 0.01):
        self.arch = "moe"
        self.n_experts = n_experts
        self.experts = [GRUEncoder(hidden=hidden, seed=seed + i, lr=lr)
                        for i in range(n_experts)]

    def n_params(self) -> int:
        return sum(e.n_params() for e in self.experts)

    def _route(self, opp_hist) -> int:
        s = route_style(opp_hist)
        return s if s < self.n_experts else self.n_experts - 1

    def fit_seq(self, X, Y, opp_hist):
        self.experts[self._route(opp_hist)].fit_step(X, Y)

    def predict_seq(self, X, opp_hist):
        return self.experts[self._route(opp_hist)].predict_seq(X)


class EncoderBelief:
    """用编码器替代手写信念，接口对齐 BeliefTracker（供 BR 使用）。"""

    def __init__(self, model, max_len: int = 24, smooth: float = 0.05,
                 tau: float = 1.0, floor: float = 0.0):
        self.model = model
        self.max_len = max_len
        self.smooth = smooth
        # 校准：tau>1 锐化(压低小概率残余)、floor 截断小概率噪声
        self.tau = tau
        self.floor = floor
        self.own: List[str] = []
        self.opp: List[str] = []
        self.last_own = None
        self.steps = 0
        # bigram 转移统计（对手上一招类别 -> 下一招类别，衰减计数）。
        # GRU 因子把序列压成 14 个标量后丢失了转移细节，而「交替出招」的
        # 边际分布与均匀无异，必须用转移概率才能看穿（实测 mirror 脚本 52%）。
        from .belief import CATS
        self._CATS = CATS
        self.bigram = {c: {} for c in CATS}
        self.bigram_tot = 0.0
        self.prev_cat = None

    def reset(self):
        self.own, self.opp = [], []
        self.last_own = None
        self.steps = 0
        self.bigram = {c: {} for c in self._CATS}
        self.bigram_tot = 0.0
        self.prev_cat = None

    def observe(self, own_move, opp_move):
        if opp_move is None:
            return
        self.own.append(own_move)
        self.opp.append(opp_move)
        if len(self.own) > self.max_len:
            self.own.pop(0); self.opp.pop(0)
        self.last_own = own_move
        self.steps += 1
        cat = CATEGORIES_OF.get(opp_move, "attack")
        if self.prev_cat is not None:
            d = 0.85
            for c in self.bigram:
                row = self.bigram[c]
                for k in list(row):
                    row[k] *= d
                    if row[k] < 1e-6:
                        del row[k]
            self.bigram_tot *= d
            row = self.bigram[self.prev_cat]
            row[cat] = row.get(cat, 0.0) + 1.0
            self.bigram_tot += 1.0
        self.prev_cat = cat

    def factors(self) -> Dict[str, float]:
        if not self.own:
            # 无历史：给均匀先验
            u = 1.0 / len(CATS)
            return {k: u for k in FACTOR_KEYS}
        return predict_factors(self.model, self.own, self.opp)

    def opp_weights(self, legal, last_own=None):
        f = self.factors()
        cat_long = {
            "attack": f["long_atk"], "heavy": f["long_heavy"],
            "defense": f["long_def"], "dodge": f["long_dodge"],
            "exhale": f["long_exhale"], "noqi": f["long_noqi"],
        }
        cat_short = dict(cat_long)
        cat_short["attack"] = f["short_atk"]
        ws = f["switch"]
        cat_w = {c: (1 - ws) * cat_long[c] + ws * cat_short[c] for c in CATS}
        # 校准：截断小概率残差 + 锐化，抑制"幻影攻击"导致的过度谨慎
        if self.floor > 0:
            cat_w = {c: max(0.0, cat_w[c] - self.floor) for c in CATS}
        if self.tau != 1.0:
            cat_w = {c: (cat_w[c] ** self.tau) for c in CATS}
        if f["attack_streak"] > 0:
            cat_w["attack"] *= (1 + 1.5 * f["attack_streak"])
            cat_w["heavy"] *= (1 + 1.0 * f["attack_streak"])
        w = {m: cat_w.get(CATEGORIES.get(m, "attack"), 0.0) + 1e-6 for m in legal}
        if last_own is not None and f["mirror"] > 0.3 and last_own in w:
            w[last_own] += f["mirror"]

        # bigram 转移通道（与手写 BeliefTracker 同款）：β 随样本量上升
        if self.bigram_tot > 1e-9 and self.prev_cat is not None:
            row = self.bigram[self.prev_cat]
            row_tot = sum(row.values())
            if row_tot > 1e-9:
                k = len(self._CATS)
                bg = {c: (row.get(c, 0.0) + 0.3) / (row_tot + 0.3 * k) for c in self._CATS}
                w_bg = {m: bg.get(CATEGORIES_OF.get(m, "attack"), 0.0) + 1e-6 for m in legal}
                s_bg = sum(w_bg.values())
                if s_bg > 1e-12:
                    w_bg = {m: v / s_bg for m, v in w_bg.items()}
                    beta = row_tot / (row_tot + 3.0)
                    w = {m: (1.0 - beta) * w.get(m, 0.0) + beta * w_bg.get(m, 0.0)
                         for m in legal}

        s = sum(w.values())
        w = {m: v / s for m, v in w.items()} if s > 0 else {m: 1.0 / len(legal) for m in legal}
        if self.smooth > 0:
            u = 1.0 / len(legal)
            w = {m: (1 - self.smooth) * w[m] + self.smooth * u for m in legal}
        return w
