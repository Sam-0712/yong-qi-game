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

from ..core.rules import next_qi
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


def _next_qi_matrix(T: Tables) -> np.ndarray:
    """(S, A) 每格：AI 出该招后的气数（用于气数储备约束）。"""
    S, A = T.maskA.shape
    out = np.zeros((S, A), dtype=np.int64)
    for si, (qa, _qp) in enumerate(T.states):
        va = T.movesA[si]
        for i, a in enumerate(va):
            out[si, i] = next_qi(qa, a)
    return out


def _allowed_actions(T: Tables, Y: np.ndarray, qi_floor: int,
                     floor_win: float) -> np.ndarray:
    """哪些动作允许。

    气数储备：默认不许把气打到 qi_floor 以下——除非这一手对当前信念的
    即时胜率 >= floor_win（值得用气换命）。这条约束编码了「节奏」：
    把气打空却只换来平局，等于把主动权白送，低气后无攻击手段会被蓄气
    套路拖死。
    """
    nqa = _next_qi_matrix(T)
    pwin = np.einsum("sab,sb->sa", (T.term & (T.pay > 0)).astype(np.float64), Y)
    allow = (nqa >= qi_floor) | (pwin >= floor_win)
    allow &= T.maskA > 0
    dead = ~allow.any(1)
    if dead.any():  # 兜底：全被禁时退回原始合法集
        allow[dead] = T.maskA[dead] > 0
    return allow


def solve_best_response(mode: int,
                        opp_policy: Dict[Tuple[int, int], Dict[str, float]],
                        gamma: float = 0.95, max_sweeps: int = 2000,
                        tol: float = 1e-7, step_cost: float = 0.0,
                        tables: Tables | None = None,
                        qi_floor: int = 0, floor_win: float = 0.5,
                        temp: float = 0.0) -> BRResult:
    """单边最佳响应。

    step_cost>0 ：每回合扣 c，惩罚无限平局、鼓励主动破局。
    qi_floor    ：气数储备下限（0=关闭），见 _allowed_actions。
    temp>0      ：对 Q 做 softmax（而非硬 argmax），消解"确定性可被脚本化"。
    """
    T = tables if tables is not None else build_tables(mode)
    S = len(T.states)
    Y = _opp_matrix(T, opp_policy)
    allow = _allowed_actions(T, Y, qi_floor, floor_win)

    V = np.zeros(S)
    Q = np.zeros((S, T.maskA.shape[1]))
    sweeps = 0
    for sw in range(1, max_sweeps + 1):
        sweeps = sw
        M = np.where(T.term, T.pay, gamma * V[T.succ] - step_cost)
        Q = np.einsum("sab,sb->sa", M, Y)
        Qm = np.where(allow, Q, -1e9)
        Vn = Qm.max(1)
        d = float(np.abs(Vn - V).max())
        V = Vn
        if d < tol:
            break

    policy: Dict[Tuple[int, int], Dict[str, float]] = {}
    for si, (qa, qp) in enumerate(T.states):
        va = T.movesA[si]
        n = len(va)
        q = np.where(allow[si, :n], Q[si, :n], -1e9)
        if temp > 0:
            w = np.exp((q - q.max()) / temp)
            w = w / w.sum()
            policy[(qa, qp)] = {va[i]: float(w[i]) for i in range(n) if w[i] > 1e-6}
        else:
            best = q.max()
            idx = [i for i in range(n) if q[i] >= best - 1e-9]
            policy[(qa, qp)] = {va[i]: 1.0 / len(idx) for i in idx}
    values = {(qa, qp): float(V[si]) for si, (qa, qp) in enumerate(T.states)}
    return BRResult(mode, gamma, policy, values, {"sweeps": sweeps})
