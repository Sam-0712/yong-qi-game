"""二维递推：在 (qa,qp) 网格上解「实时胜率面」W(s)。

为什么是二维：
- 只有我方 qa 上升（吐气）和对方 qp 下降（打中）两条路能改变局势；两者都在
  同一个格点上结算，所以必须同时在两个方向上递推。
- 每回合结算一次、胜负由结算决定 → 状态 (qa,qp) 上的 W 就是"实时胜率"，
  与历史无关（历史只决定 π 与 μ 这两组出招分布）。

方程（吸收链）：
    W(s) = Σ_a π(a|s) Σ_b μ(b|s) · [ 1 若 a 胜 b；0 若 a 负 b；W(s') 否则 ]
    s' = (next_qi(qa,a), next_qi(qp,b))
用 γ 阻尼解决"双方都不愿变"的闭合类（否则 I-P 奇异），最后按 (u+v) 归一，
得到"在会结束的对局里我方最终取胜的概率"。
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from courage.core.rules import get_valid_moves, judge, next_qi
from courage.solvers.equilibrium import QI_MAX

S = QI_MAX + 1
STATES = [(qa, qp) for qa in range(S) for qp in range(S)]
IDX = {st: i for i, st in enumerate(STATES)}


def solve_win_surface(pi: Dict[Tuple[int, int], Dict[str, float]],
                      mu: Dict[Tuple[int, int], Dict[str, float]],
                      damp: float = 0.9995) -> Tuple[np.ndarray, np.ndarray]:
    """返回 (W, progress)：W 为归一化胜率面；progress 为"能结束的概率"。"""
    n = len(STATES)
    P = np.zeros((n, n))
    win = np.zeros(n)
    lose = np.zeros(n)
    for si, (qa, qp) in enumerate(STATES):
        pa = pi.get((qa, qp)) or {}
        pb = mu.get((qa, qp)) or {}
        if not pa or not pb:
            continue
        for a, wa in pa.items():
            if wa <= 0:
                continue
            for b, wb in pb.items():
                if wb <= 0:
                    continue
                p = wa * wb
                w = judge(a, b)
                if w == 1:
                    win[si] += p
                elif w == -1:
                    lose[si] += p
                else:
                    j = IDX[(next_qi(qa, a), next_qi(qp, b))]
                    P[si, j] += p
    I = np.eye(n)
    # (I - damp·P) x = damp·r ，用阻尼避免闭合类导致的奇异
    u = np.linalg.solve(I - damp * P, damp * win)
    v = np.linalg.solve(I - damp * P, damp * lose)
    tot = u + v
    # tot 很小 = 这个状态大概率永远打不完（闭合类），此时 u/tot 是 0/0，标为 nan
    W = np.where(tot > 1e-3, u / np.maximum(tot, 1e-9), np.nan)
    return W.reshape(S, S), tot.reshape(S, S)


def surface(pi, mu) -> np.ndarray:
    W, _ = solve_win_surface(pi, mu)
    return W
