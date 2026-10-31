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
                        temp: float = 0.0,
                        death_penalty: float = 0.0,
                        kappa: float = 0.0,
                        zero_penalty: float = 0.0) -> BRResult:
    """单边最佳响应。

    step_cost     ：每回合扣 c，惩罚无限平局、鼓励主动破局。
    qi_floor      ：气数储备下限（0=关闭），见 _allowed_actions。
    temp>0        ：对 Q 做 softmax（而非硬 argmax），消解"确定性可被脚本化"。
    death_penalty ：风险厌恶。把"本回合立即输"的终局收益从 -1 放大到 -(1+ρ)，
                    而"少一口气/平局"不受影响。
    kappa>0       ：min-混合（鲁棒）。Q = (1-κ)·E_μ[V] + κ·min_{b∈supp(μ)} V。

                    为什么必须有：单边"对手模型"是**循环依赖**的——我出防御，
                    对手就用能破防的招；我出跳，对手就换雷切。按固定 μ 取期望
                    等于假设对手不会针对我，于是 BR 在低气局面会算出"V(0,3)=+0.69"
                    这种荒谬的乐观值（因为 μ 恰好把对手的招猜成了对我们有利的
                    那种），进而认为"把气打光也无所谓"，开局就出雷切自毁。

                    κ 与 μ 的置信度反向：看不清对手时更要按最坏情况定价。
    """
    T = tables if tables is not None else build_tables(mode)
    S = len(T.states)
    Y = _opp_matrix(T, opp_policy)
    allow = _allowed_actions(T, Y, qi_floor, floor_win)

    # 终局收益矩阵；death_penalty 放大失败分支
    pay = T.pay
    if death_penalty > 0:
        pay = pay - death_penalty * np.maximum(0.0, -pay)

    # 只在 μ 的支撑集上取最坏值：对手不会出的招不该影响定价
    support = (Y > 1e-6)[:, None, :]

    # 「归零」惩罚：把气打到 0 之后进入的状态，价值按 V-w 计。
    # 为什么：0 气没有攻击招，只能靠对手失误才爬得回来。而单边 μ 无法表达
    # "对手会针对我"（我出防御它就换破防招），于是 V(0,·) 被系统性高估
    # （实测 +0.69，而真实对局里从 0 气翻盘几乎是 0）。这一项把高估的部分
    # 扣回来，只影响**续接**分支，"本回合直接获胜"不受影响。
    if zero_penalty > 0:
        zero_mask = np.zeros((S, T.maskA.shape[1]), dtype=np.float64)
        for si, (qa, _qp) in enumerate(T.states):
            for i, a in enumerate(T.movesA[si]):
                if next_qi(qa, a) == 0:
                    zero_mask[si, i] = 1.0
    else:
        zero_mask = None

    V = np.zeros(S)
    Q = np.zeros((S, T.maskA.shape[1]))
    sweeps = 0
    for sw in range(1, max_sweeps + 1):
        sweeps = sw
        cont = gamma * V[T.succ] - step_cost
        if zero_mask is not None:
            cont = cont - zero_penalty * zero_mask[:, :, None]
        M = np.where(T.term, pay, cont)
        if kappa > 0:
            Msup = np.where(support, M, 1e9)
            Qworst = Msup.min(2)
            Q = (1.0 - kappa) * np.einsum("sab,sb->sa", M, Y) + kappa * Qworst
        else:
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
