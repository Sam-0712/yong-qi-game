"""平稳随机博弈的均衡求解。

状态降为 (qa, qp)，回合不再入状态；靠折扣 γ<1 保证值迭代收敛（Shapley）。
内层用「批量对冲（Hedge / 指数权重）」同时求解每个状态的零和矩阵博弈，
一次向量化更新全部 36 个状态，避免 Python 层嵌套循环。

价值方程（零和、同时行动）：
    V(s) = Val( M_s ),  M_s[a,b] = J(a,b) 若终局, 否则 γ·V(s')
其中 a=AI 行、b=对手列，J 为 AI 视角收益。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..core.rules import get_valid_moves, judge, next_qi
from ..core.state import MODE_STANDARD, QI_MAX

QS = list(range(QI_MAX + 1))


def _states() -> List[Tuple[int, int]]:
    return [(qa, qp) for qa in QS for qp in QS]


def _sid(qa: int, qp: int) -> int:
    return qa * (QI_MAX + 1) + qp


@dataclass
class Tables:
    states: List[Tuple[int, int]]
    movesA: List[List[str]]
    movesB: List[List[str]]
    term: np.ndarray    # (S,A,B) 是否终局
    pay: np.ndarray     # (S,A,B) 终局收益（AI 视角）
    succ: np.ndarray    # (S,A,B) 后继状态下标
    maskA: np.ndarray   # (S,A) AI 合法动作
    maskB: np.ndarray   # (S,B) 对手合法动作


def build_tables(mode: int = MODE_STANDARD) -> Tables:
    states = _states()
    S = len(states)
    amax = bmax = max(len(get_valid_moves(q, mode=mode)) for q in QS)
    term = np.zeros((S, amax, bmax), dtype=bool)
    pay = np.zeros((S, amax, bmax), dtype=np.float64)
    succ = np.zeros((S, amax, bmax), dtype=np.int64)
    maskA = np.zeros((S, amax), dtype=np.float64)
    maskB = np.zeros((S, bmax), dtype=np.float64)
    movesA: List[List[str]] = []
    movesB: List[List[str]] = []

    for si, (qa, qp) in enumerate(states):
        va = get_valid_moves(qa, mode=mode)
        vb = get_valid_moves(qp, mode=mode)
        movesA.append(va)
        movesB.append(vb)
        maskA[si, :len(va)] = 1.0
        maskB[si, :len(vb)] = 1.0
        for i, a in enumerate(va):
            for j, b in enumerate(vb):
                w = judge(a, b)
                if w != 0:
                    term[si, i, j] = True
                    pay[si, i, j] = float(w)
                else:
                    succ[si, i, j] = _sid(next_qi(qa, a), next_qi(qp, b))
    return Tables(states, movesA, movesB, term, pay, succ, maskA, maskB)


def solve_matrices(M: np.ndarray, maskA: np.ndarray, maskB: np.ndarray,
                   iters: int = 3000, X0=None, Y0=None):
    """批量求解 (S,A,B) 的零和矩阵博弈（regret-matching+，线性平均）。

    返回 (值, 行平均策略, 列平均策略)。对每个状态独立做 CFR，一次向量化。
    """
    S, A, B = M.shape
    unifA = maskA / maskA.sum(1, keepdims=True)
    unifB = maskB / maskB.sum(1, keepdims=True)
    Rr = np.zeros((S, A))
    Rc = np.zeros((S, B))
    Sr = np.zeros((S, A))
    Sc = np.zeros((S, B))

    for t in range(1, iters + 1):
        pr = np.maximum(Rr, 0.0) * maskA
        sr = pr.sum(1, keepdims=True)
        X = np.where(sr > 0, pr / np.where(sr > 0, sr, 1.0), unifA)
        pc = np.maximum(Rc, 0.0) * maskB
        sc = pc.sum(1, keepdims=True)
        Y = np.where(sc > 0, pc / np.where(sc > 0, sc, 1.0), unifB)

        Ux = np.einsum("sab,sb->sa", M, Y)
        Uy = -np.einsum("sab,sa->sb", M, X)
        v = np.einsum("sa,sab,sb->s", X, M, Y)

        Rr = np.maximum(Rr + (Ux - v[:, None]) * maskA, 0.0)
        Rc = np.maximum(Rc + (Uy + v[:, None]) * maskB, 0.0)
        Sr += t * X
        Sc += t * Y

    Xb = Sr * maskA
    Xb /= Xb.sum(1, keepdims=True)
    Yb = Sc * maskB
    Yb /= Yb.sum(1, keepdims=True)
    V = np.einsum("sa,sab,sb->s", Xb, M, Yb)
    return V, Xb, Yb


def _exploitability(T: Tables, V: np.ndarray, gamma: float,
                    X: np.ndarray, Y: np.ndarray) -> float:
    M = np.where(T.term, T.pay, gamma * V[T.succ])
    row = np.einsum("sab,sb->sa", M, Y)
    col = np.einsum("sab,sa->sb", M, X)
    br_row = np.where(T.maskA > 0, row, -1e9).max(1)
    br_col = np.where(T.maskB > 0, col, 1e9).min(1)
    return float((br_row - br_col).max())


@dataclass
class EquilibriumResult:
    mode: int
    gamma: float
    values: Dict[Tuple[int, int], float]
    policy: Dict[Tuple[int, int], Dict[str, float]]
    meta: Dict[str, float] = field(default_factory=dict)


def solve_equilibrium(mode: int = MODE_STANDARD, gamma: float = 0.95,
                      iters: int = 800, max_sweeps: int = 400,
                      tol: float = 1e-3, verbose: bool = False) -> EquilibriumResult:
    T = build_tables(mode)
    S = len(T.states)
    V = np.zeros(S)
    X = Y = None
    sweeps = 0
    for sw in range(1, max_sweeps + 1):
        sweeps = sw
        M = np.where(T.term, T.pay, gamma * V[T.succ])
        Vn, X, Y = solve_matrices(M, T.maskA, T.maskB, iters=iters, X0=X, Y0=Y)
        d = float(np.abs(Vn - V).max())
        V = Vn
        if verbose and (sw == 1 or sw % 20 == 0):
            print(f"  sweep {sw:>3}  ΔV={d:.2e}  V(3,3)={V[_sid(3,3)]:+.4f}")
        if d < tol:
            break

    gap = _exploitability(T, V, gamma, X, Y)
    policy: Dict[Tuple[int, int], Dict[str, float]] = {}
    for si, (qa, qp) in enumerate(T.states):
        va = T.movesA[si]
        policy[(qa, qp)] = {va[i]: float(X[si, i]) for i in range(len(va))}
    values = {(qa, qp): float(V[si]) for si, (qa, qp) in enumerate(T.states)}
    return EquilibriumResult(mode, gamma, values, policy,
                             {"sweeps": sweeps, "gap": gap, "mode": mode, "gamma": gamma})
