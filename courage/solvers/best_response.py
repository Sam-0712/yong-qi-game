"""对给定对手策略的最佳响应（单边 MDP 值迭代，最大化收益≈胜率）。

与均衡不同：这里把对手当成环境的一部分（其动作分布固定），
AI 只需对每个状态选使 Q 最大的动作。C 阶段用对手模型喂入 opp_policy。

    Q(s,a) = Σ_b π_opp(b|s)·[ J(a,b) 若终局, 否则 γ·V(s') ]
    V(s)   = max_a Q(s,a)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from ..core.state import MODE_STANDARD
from .equilibrium import Tables, build_tables


@dataclass
class BRResult:
    mode: int
    gamma: float
    policy: Dict[Tuple[int, int], Dict[str, float]]
    values: Dict[Tuple[int, int], float]
    meta: Dict[str, float] = field(default_factory=dict)


def _opp_matrix(T: Tables, opp_policy: Dict[Tuple[int, int], Dict[str, float]]
                ) -> np.ndarray:
    S, B = T.maskB.shape
    Y = np.zeros((S, B), dtype=np.float64)
    for si, (qa, qp) in enumerate(T.states):
        pol = opp_policy.get((qa, qp), {})
        vb = T.movesB[si]
        for j, b in enumerate(vb):
            Y[si, j] = max(float(pol.get(b, 0.0)), 0.0)
        s = Y[si].sum()
        if s <= 1e-12:
            Y[si, :len(vb)] = 1.0 / len(vb)
        else:
            Y[si] /= s
    return Y


def solve_best_response(mode: int,
                        opp_policy: Dict[Tuple[int, int], Dict[str, float]],
                        gamma: float = 0.95, max_sweeps: int = 2000,
                        tol: float = 1e-7) -> BRResult:
    T = build_tables(mode)
    S = len(T.states)
    Y = _opp_matrix(T, opp_policy)

    V = np.zeros(S)
    Q = np.zeros((S, T.maskA.shape[1]))
    sweeps = 0
    for sw in range(1, max_sweeps + 1):
        sweeps = sw
        M = np.where(T.term, T.pay, gamma * V[T.succ])
        Q = np.einsum("sab,sb->sa", M, Y)
        Qm = np.where(T.maskA > 0, Q, -1e9)
        Vn = Qm.max(1)
        d = float(np.abs(Vn - V).max())
        V = Vn
        if d < tol:
            break

    policy: Dict[Tuple[int, int], Dict[str, float]] = {}
    for si, (qa, qp) in enumerate(T.states):
        va = T.movesA[si]
        q = Q[si, :len(va)]
        best = q.max()
        idx = [i for i in range(len(va)) if q[i] >= best - 1e-9]
        policy[(qa, qp)] = {va[i]: 1.0 / len(idx) for i in idx}
    values = {(qa, qp): float(V[si]) for si, (qa, qp) in enumerate(T.states)}
    return BRResult(mode, gamma, policy, values, {"sweeps": sweeps})
