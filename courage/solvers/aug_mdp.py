"""精确值 oracle：把「对手真正依赖的相位」并进状态后再求最佳响应。

为什么需要这个模块
------------------
``best_response.solve_best_response`` 在 36 个 ``(qa,qp)`` 状态上求 BR，前提是
对手策略只是**当前气数**的函数。但真实打法经常依赖相位：

    低气交替   依赖对手自己上一招（吐气→馒头→吐气…）
    镜像/反跳  依赖我方上一招
    4 手/2 手  依赖回合数

对这些对手，「36 状态边际 BR」等价于假设对手每回合独立重抽，会**系统性低估**
可得价值（对手的周期结构本来是白送的）。评测要量化的正是这个差额。

做法：把对手策略真正读的相位 ``needs`` 拼进状态，得到 ``36 × 10^k``（k<=2）或
``36 × (t_max+1)`` 大小的增广 MDP——最多几百格，仍是毫秒级精确值迭代。
转移/判定本身不依赖历史（见 ``core/state.py``），所以增广只影响「对手出招分布」，
不会破坏 MDP 结构：对手一动，相位就确定性地前进一格。

产出
----
``solve_aug`` 返回 ``V``（对**任意**对手的可达价值上限）与 ``Q``。用它和智能体
实际决策的 Q 逐点比较，就得到「逐决策 regret」——比 80 局胜率精确、零噪声，
并且能定位到具体是哪一格把价值算错了。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.rules import IDX_TO_MOVE, MOVE_TO_IDX, MOVES, get_valid_moves, judge, next_qi
from ..core.state import MODE_STANDARD, QI_MAX

DEFAULT_GAMMA = 0.95
DEFAULT_T_MAX = 4          # t >= 4 时 4 手/2 手套路都已进入平稳阶段

LAST_MAX = len(MOVES) + 1  # 0=None, 1..9=MOVES
_LAST_ALIASES = {"own_last": "bl", "opp_last": "al"}


def normalize_needs(needs: Sequence[str]) -> Tuple[str, ...]:
    """把 styles 的相位名翻译成本模块的维度名（own_last→bl, opp_last→al）。"""
    out: List[str] = []
    for n in needs:
        k = _LAST_ALIASES.get(n, n)
        if k not in ("al", "bl", "t"):
            raise ValueError(f"未知相位: {n}")
        if k not in out:
            out.append(k)
    return tuple(out)


@dataclass
class AugTables:
    """增广 MDP 张量。状态元组固定为 (qa, qp, al, bl, t)。"""

    mode: int
    needs: Tuple[str, ...]
    t_max: int
    states: List[Tuple[int, int, int, int, int]]
    movesA: List[List[str]]
    movesB: List[List[str]]
    term: np.ndarray        # (S,A,B) bool
    pay: np.ndarray         # (S,A,B) 终局收益（AI 视角）
    succ: np.ndarray        # (S,A,B) 后继状态下标
    maskA: np.ndarray       # (S,A)
    maskB: np.ndarray       # (S,B)
    _idx: np.ndarray        # 投影索引表 (6,6,LAST_MAX,LAST_MAX,t_max+1)

    def index(self, qa: int, qp: int, al: int = 0, bl: int = 0, t: int = 0) -> int:
        if "al" not in self.needs:
            al = 0
        if "bl" not in self.needs:
            bl = 0
        if "t" not in self.needs:
            t = 0
        else:
            t = min(int(t), self.t_max)
        si = int(self._idx[qa, qp, al, bl, t])
        if si < 0:
            raise KeyError(f"状态不在增广空间: {(qa, qp, al, bl, t)}")
        return si


def build_aug(mode: int = MODE_STANDARD, needs: Sequence[str] = (),
              t_max: int = DEFAULT_T_MAX) -> AugTables:
    """按 needs 构造增广 MDP。needs=() 时与 ``equilibrium.build_tables`` 同构。"""
    needs = normalize_needs(needs)
    t_dim = (t_max + 1) if "t" in needs else 1
    al_vals = range(LAST_MAX) if "al" in needs else range(1)
    bl_vals = range(LAST_MAX) if "bl" in needs else range(1)
    t_vals = range(t_dim)

    qs = range(QI_MAX + 1)
    amax = max(len(get_valid_moves(q, mode=mode)) for q in qs)
    bmax = amax

    states: List[Tuple[int, int, int, int, int]] = []
    idx = np.full((QI_MAX + 1, QI_MAX + 1, LAST_MAX, LAST_MAX, t_dim), -1,
                  dtype=np.int64)
    for qa in qs:
        for qp in qs:
            for al in al_vals:
                for bl in bl_vals:
                    for t in t_vals:
                        idx[qa, qp, al, bl, t] = len(states)
                        states.append((qa, qp, al, bl, t))

    S = len(states)
    term = np.zeros((S, amax, bmax), dtype=bool)
    pay = np.zeros((S, amax, bmax), dtype=np.float64)
    succ = np.zeros((S, amax, bmax), dtype=np.int64)
    maskA = np.zeros((S, amax), dtype=np.float64)
    maskB = np.zeros((S, bmax), dtype=np.float64)
    movesA: List[List[str]] = []
    movesB: List[List[str]] = []

    for si, (qa, qp, al, bl, t) in enumerate(states):
        va = get_valid_moves(qa, mode=mode)
        vb = get_valid_moves(qp, mode=mode)
        movesA.append(va)
        movesB.append(vb)
        maskA[si, :len(va)] = 1.0
        maskB[si, :len(vb)] = 1.0
        for i, a in enumerate(va):
            nqa = next_qi(qa, a)
            nal = MOVE_TO_IDX[a] if "al" in needs else 0
            for j, b in enumerate(vb):
                w = judge(a, b)
                if w != 0:
                    term[si, i, j] = True
                    pay[si, i, j] = float(w)
                    continue
                nbl = MOVE_TO_IDX[b] if "bl" in needs else 0
                nt = min(t + 1, t_max) if "t" in needs else 0
                succ[si, i, j] = idx[nqa, next_qi(qp, b), nal, nbl, nt]

    return AugTables(mode, needs, t_max if "t" in needs else 0, states,
                     movesA, movesB, term, pay, succ, maskA, maskB, idx)


def opp_matrix(T: AugTables,
               opp_fn: Callable[..., Dict[str, float]]) -> np.ndarray:
    """把对手策略压成 (S,B) 矩阵。

    ``opp_fn(own, opp, t, own_last, opp_last)`` 使用**对手自己**的视角
    （own=对手气数），与 ``courage.bots.opponents.OpponentSpec.policy`` 同签名。
    """
    S, B = T.maskB.shape
    Y = np.zeros((S, B), dtype=np.float64)
    for si, (qa, qp, al, bl, t) in enumerate(T.states):
        vb = T.movesB[si]
        own_last = IDX_TO_MOVE.get(bl)
        opp_last = IDX_TO_MOVE.get(al)
        dist = opp_fn(qp, qa, t, own_last, opp_last) or {}
        for j, b in enumerate(vb):
            Y[si, j] = max(float(dist.get(b, 0.0)), 0.0)
        s = Y[si].sum()
        if s <= 1e-12:                       # 无支撑 → 均匀（与 BR 的实现一致）
            Y[si, :len(vb)] = 1.0 / len(vb)
        else:
            Y[si] /= s
    return Y


@dataclass
class AugSolution:
    tables: AugTables
    V: np.ndarray
    Y: np.ndarray
    gamma: float
    sweeps: int

    # ── 查值 ──
    def value(self, qa: int, qp: int, al: int = 0, bl: int = 0, t: int = 0) -> float:
        return float(self.V[self.tables.index(qa, qp, al, bl, t)])

    def q_row(self, qa: int, qp: int, al: int = 0, bl: int = 0,
              t: int = 0) -> np.ndarray:
        """该状态下我方每个动作的精确 Q（对齐 ``tables.movesA[si]``）。"""
        si = self.tables.index(qa, qp, al, bl, t)
        return self.q_row_at(si)

    def q_row_at(self, si: int) -> np.ndarray:
        T = self.tables
        M = np.where(T.term[si], T.pay[si],
                     self.gamma * self.V[T.succ[si]])
        return M @ self.Y[si]

    def q_matrix(self) -> np.ndarray:
        """全部状态的 Q，(S,A)。"""
        T = self.tables
        M = np.where(T.term, T.pay, self.gamma * self.V[T.succ])
        return np.einsum("sab,sb->sa", M, self.Y)

    def policy(self) -> List[Dict[str, float]]:
        """最优响应策略（每状态一分布，平手均分）。"""
        Q = self.q_matrix()
        out: List[Dict[str, float]] = []
        for si, va in enumerate(self.tables.movesA):
            n = len(va)
            q = np.where(self.tables.maskA[si, :n] > 0, Q[si, :n], -np.inf)
            best = q.max()
            idx = [i for i in range(n) if q[i] >= best - 1e-9]
            out.append({va[i]: 1.0 / len(idx) for i in idx})
        return out


def solve_aug(T: AugTables, opp_fn: Callable[..., Dict[str, float]],
              gamma: float = DEFAULT_GAMMA, max_sweeps: int = 4000,
              tol: float = 1e-10, Y: Optional[np.ndarray] = None) -> AugSolution:
    """增广状态上的精确最佳响应（单边 MDP 值迭代，无塑形项）。"""
    Y = opp_matrix(T, opp_fn) if Y is None else Y
    S = len(T.states)
    V = np.zeros(S)
    sweeps = 0
    for sw in range(1, max_sweeps + 1):
        sweeps = sw
        cont = gamma * V[T.succ]
        M = np.where(T.term, T.pay, cont)
        Q = np.einsum("sab,sb->sa", M, Y)
        Qm = np.where(T.maskA > 0, Q, -1e9)
        Vn = Qm.max(1)
        d = float(np.abs(Vn - V).max())
        V = Vn
        if d < tol:
            break
    return AugSolution(T, V, Y, gamma, sweeps)


def solve_style(name: str, mode: int = MODE_STANDARD, gamma: float = DEFAULT_GAMMA,
                t_max: int = DEFAULT_T_MAX) -> AugSolution:
    """便捷入口：直接对一个 ``courage.bots.opponents`` 风格求精确值。"""
    from ..bots.opponents import OPPONENTS
    spec = OPPONENTS[name]
    T = build_aug(mode, spec.needs, t_max)
    return solve_aug(T, spec.policy, gamma=gamma)


def oracle_for(spec, mode: int = MODE_STANDARD, gamma: float = DEFAULT_GAMMA,
               t_max: int = DEFAULT_T_MAX) -> AugSolution:
    T = build_aug(mode, spec.needs, t_max)
    return solve_aug(T, spec.policy, gamma=gamma)
