"""小型对手模型（C 阶段的最小闭环）。

跨状态共享参数的 2 层 MLP，纯 numpy，参数 < 1k，可导出 JSON 供浏览器推理。
输入特征：状态 (qa, qp, mode) + 对手上一招 one-hot
输出    ：对手下一招分布（softmax over MOVES）

内存/算力友好：float32，单隐层，批量训练。仅在 learning 层依赖 numpy。
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

from ..core.rules import MOVES, MOVE_TO_IDX, get_valid_moves

_N = len(MOVES)          # 动作数
_FEAT = 3 + _N           # qa,qp,mode 归一 + 上一招 one-hot


def features(qa: int, qp: int, mode: int = 0,
             last_opp: Optional[str] = None) -> np.ndarray:
    x = np.zeros(_FEAT, dtype=np.float32)
    x[0] = qa / 5.0
    x[1] = qp / 5.0
    x[2] = float(mode)
    if last_opp is not None:
        idx = MOVE_TO_IDX.get(last_opp, 0)
        if idx:
            x[2 + idx] = 1.0
    return x


class OpponentModel:
    def __init__(self, hidden: int = 16, seed: int = 0, lr: float = 0.1):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        self.lr = lr
        self.W1 = rng.normal(0.0, 0.1, (hidden, _FEAT)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.normal(0.0, 0.1, (_N, hidden)).astype(np.float32)
        self.b2 = np.zeros(_N, dtype=np.float32)

    @property
    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def _forward(self, X: np.ndarray):
        H = np.tanh(X @ self.W1.T + self.b1)      # (B, h)
        logits = H @ self.W2.T + self.b2          # (B, N)
        logits -= logits.max(axis=1, keepdims=True)
        E = np.exp(logits)
        P = E / E.sum(axis=1, keepdims=True)
        return H, P

    def proba_batch(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[1]

    def predict_proba(self, qa, qp, mode=0, last_opp=None) -> np.ndarray:
        return self.proba_batch(features(qa, qp, mode, last_opp)[None, :])[0]

    def proba_masked(self, qa, qp, mode=0, last_opp=None) -> np.ndarray:
        """把非法招式概率清零并重归一，游戏内可直接采样。"""
        p = self.predict_proba(qa, qp, mode, last_opp).astype(np.float64).copy()
        legal = set(get_valid_moves(qa, mode=mode))
        for i, m in enumerate(MOVES):
            if m not in legal:
                p[i] = 0.0
        s = p.sum()
        return (p / s) if s > 0 else p

    def predict(self, qa, qp, mode=0, last_opp=None) -> str:
        return MOVES[int(np.argmax(self.proba_masked(qa, qp, mode, last_opp)))]

    def train_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        """交叉熵 + 手写反向传播。y 为 0..N-1 的动作下标。"""
        B = X.shape[0]
        H, P = self._forward(X)
        loss = float(-np.log(P[np.arange(B), y] + 1e-9).mean())

        dZ = P.copy()
        dZ[np.arange(B), y] -= 1.0
        dZ /= B
        gW2 = dZ.T @ H
        gb2 = dZ.sum(axis=0)
        dA = (dZ @ self.W2) * (1.0 - H * H)
        gW1 = dA.T @ X
        gb1 = dA.sum(axis=0)

        self.W1 -= self.lr * gW1
        self.b1 -= self.lr * gb1
        self.W2 -= self.lr * gW2
        self.b2 -= self.lr * gb2
        return loss

    def save(self, path: str):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        blob = {
            "hidden": self.hidden,
            "feat_dim": _FEAT,
            "moves": MOVES,
            "W1": self.W1.tolist(), "b1": self.b1.tolist(),
            "W2": self.W2.tolist(), "b2": self.b2.tolist(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(blob, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "OpponentModel":
        with open(path, "r", encoding="utf-8") as f:
            blob = json.load(f)
        m = cls(hidden=int(blob["hidden"]))
        m.W1 = np.array(blob["W1"], dtype=np.float32)
        m.b1 = np.array(blob["b1"], dtype=np.float32)
        m.W2 = np.array(blob["W2"], dtype=np.float32)
        m.b2 = np.array(blob["b2"], dtype=np.float32)
        return m
