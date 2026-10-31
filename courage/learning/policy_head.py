"""策略头：以信念因子(+气数)为输入，输出动作分布；用 REINFORCE 以胜负微调。

这是「结果微调」的一条路径：编码器(frozen) 产因子 → 策略头(可学) 选招 →
用对局胜负做策略梯度。可与解析 BR 对照。
"""

from __future__ import annotations

import numpy as np

from ..core.rules import MOVES
from .seq_encoder import FACTOR_KEYS, _Adam

NF = len(FACTOR_KEYS)
NQ = 6
D = NF + 2 * NQ
A = len(MOVES)


def featurize(factors: dict, qa: int, qp: int) -> np.ndarray:
    x = np.zeros(D)
    x[:NF] = [factors.get(k, 0.0) for k in FACTOR_KEYS]
    x[NF + qa] = 1.0
    x[NF + NQ + qp] = 1.0
    return x


class PolicyHead:
    def __init__(self, hidden: int = 32, seed: int = 0, lr: float = 0.02):
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0, 0.1, (hidden, D)), "b1": np.zeros(hidden),
            "W2": rng.normal(0, 0.1, (A, hidden)), "b2": np.zeros(A),
        }
        self.opt = _Adam(self.params, lr)

    def _forward(self, x, mask):
        p = self.params
        h = np.maximum(0.0, p["W1"] @ x + p["b1"])
        z = p["W2"] @ h + p["b2"]
        z = np.where(mask > 0, z, -1e9)
        z = z - z.max()
        e = np.exp(z)
        return h, e / e.sum()

    def act(self, x, mask, rng):
        _, prob = self._forward(x, mask)
        r = rng.random()
        cum = 0.0
        for i, pv in enumerate(prob):
            cum += pv
            if r <= cum:
                return i, prob
        return len(prob) - 1, prob

    def update(self, traj, advantage: float):
        """traj: list of (x, mask, a)。单条轨迹用同一 advantage（MC）。"""
        g = {k: np.zeros_like(v) for k, v in self.params.items()}
        for (x, mask, a) in traj:
            h, prob = self._forward(x, mask)
            d = -prob.copy()
            d[a] += 1.0                      # ∂logπ/∂logits
            g["W2"] += np.outer(d, h)
            g["b2"] += d
            dh = (self.params["W2"].T @ d) * (h > 0)
            g["W1"] += np.outer(dh, x)
            g["b1"] += dh
        for k in g:
            g[k] *= advantage
        self.opt.step(self.params, g)
