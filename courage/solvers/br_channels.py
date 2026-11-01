"""多通道风险 BR：把「我分不清对手是谁」这件事定价进 Q 值。

为什么不用固定先验解决
----------------------
上一版用 ``mixture_model`` 的 ``null_weight``（"我可能什么都不懂"的先验）来兜底。
实测这是个跷跷板：``random`` 要它高、``strong``/风格 zoo 要它低，扫遍
``null ∈ [0.3,0.6]``、``eta ∈ [1,8]`` 都只是滑动，没有一方全胜——
因为一局只有 7~9 个观测，先验权重压不下去。

正确的杠杆不是「先验给多少」，而是**通道之间的分歧有多大**：

    Q(a) = (1-κ)·Σ_c w_c·Q_c(a)  +  κ·min_{c: w_c>ε} Q_c(a)

- 通道**一致**时 min 不起作用 ⇒ 风格集上零保费（拿回那 0.05/局）；
- 通道**分歧**时 min 生效 ⇒ 退化成接近 maximin，正好治 ``random`` 那种
  「几条自信的错答案」——在没有信息的地方保守，代价最小。

这也把 ``best_response.py`` 里那段「kappa 为什么必须有」的注释接上了：
κ 当初是拿来做「对手会针对我」的定价；现在有了更精确的载体——
**我分不清对手是谁**。

线性恒等式（保证 κ=0 时与旧实现逐位一致）
--------------------------------------
对混合分布 p = Σ_c w_c·p_c，有
    Q_p(a) = Σ_c w_c·(Σ_b p_c(b)·M(a,b)) = Σ_c w_c·Q_c(a)
所以 κ=0 的通道版 BR 与「先混合分布、再求 BR」**完全等价**，
只是多算了 K 组 Q，换来可以定量追踪的分歧量。

实测结论：这个机制**没有带来收益**，默认 κ=0（保留做实验）
--------------------------------------------------------
(1,1) 残局、40 局/对手，κ 从 0 扫到 0.4：

    κ      strong  random  defensive  exploit4  converter  alternate_low  风格zoo
    0.0    0.086   0.450   0.789      0.072      0.050      0.098          0.720
    0.2    0.086   0.526   0.743      0.050      0.050      0.098          0.714
    0.4    0.086   0.540   0.743      0.050      0.050      0.098          0.721

合计几乎不动（2.265 → 2.270），是在 `random`（变差）与 `defensive`/`exploit4`
（变好）之间对倒。更强的否决证据来自**分歧与损失的相关系数**：

    r(分歧, 本步 regret)：strong −0.37  random −0.01  defensive −0.17
                           exploit4 −0.29  converter −0.67  alternate_low −0.99

"分歧大 ⇒ 亏得多"这个前提在 7 个对手里 6 个**符号相反**。原因在
``tools/trace.py`` 里一目了然：真正错的地方，几条通道是**一起错**的
（都源自同一批 12 条脚本与同一份语料），分歧反而小；而分歧大的地方往往是
多招价值接近、怎么选都差不多的格子。

推论：**这是「共同偏差」而不是「认知不确定性」**。贝叶斯意义上的正确做法本来就是
按后验概率混合（κ=0）；min 混合只有在"对手会针对我的不确定性来选择自己是谁"时
才成立，本游戏不是那样。下一步该换的是**通道的加权依据**（用决策质量而不是
预测似然），而不是继续加风险系数。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.rules import next_qi
from .best_response import _allowed_actions, _opp_matrix
from .equilibrium import Tables, build_tables

Policy = Dict[Tuple[int, int], Dict[str, float]]


@dataclass
class ChannelBRResult:
    mode: int
    gamma: float
    kappa: float
    weights: Dict[str, float]
    policy: Policy                              # 实际使用的（κ 调整后）
    values: Dict[Tuple[int, int], float]
    q_risk: Policy                              # (1-κ)·均值 + κ·最坏
    q_mean: Policy                              # 纯风险中性
    q_lo: Policy                                # 最坏通道
    q_channels: Dict[str, Policy] = field(default_factory=dict)
    meta: Dict[str, float] = field(default_factory=dict)

    # ── 追踪用的派生量 ──
    def spread(self, qa: int, qp: int) -> Dict[str, float]:
        """每个动作上「均值 − 最坏」的差 = 通道分歧，∈ [0,2]（Q 的取值域是 [-1,1]）。"""
        m, lo = self.q_mean.get((qa, qp), {}), self.q_lo.get((qa, qp), {})
        return {a: float(m[a] - lo.get(a, m[a])) for a in m}

    def flip(self, qa: int, qp: int) -> bool:
        """κ 是否改变了 argmax（＝这次决策「分歧到了会改判」的程度）。"""
        m, r = self.q_mean.get((qa, qp), {}), self.q_risk.get((qa, qp), {})
        if not m or not r:
            return False
        return _argmax(m) != _argmax(r)


def _argmax(d: Dict[str, float]) -> str:
    return max(d, key=lambda k: d[k])


def solve_channel_br(
        mode: int,
        policies: Dict[str, Policy],
        weights: Optional[Dict[str, float]] = None,
        kappa: float = 0.0,
        gamma: float = 0.95,
        max_sweeps: int = 2000,
        tol: float = 1e-7,
        step_cost: float = 0.0,
        zero_penalty: float = 0.0,
        tables: Optional[Tables] = None,
        qi_floor: int = 0,
        floor_win: float = 1.0,
) -> ChannelBRResult:
    """在「一组对手模型通道」上求风险调整的最佳响应。

    ``policies`` : {通道名: 36 状态的对手分布}
    ``weights``  : 通道权重（None = 均匀）
    ``kappa``    : 0 = 风险中性（等价旧 BR）；1 = 纯 maximin over channels
    """
    if not policies:
        raise ValueError("至少需要一条通道")
    T = tables if tables is not None else build_tables(mode)
    S = len(T.states)
    names = list(policies)

    if weights is None:
        w = {c: 1.0 / len(names) for c in names}
    else:
        s = sum(max(0.0, weights.get(c, 0.0)) for c in names)
        w = ({c: max(0.0, weights.get(c, 0.0)) / s for c in names} if s > 1e-12
             else {c: 1.0 / len(names) for c in names})
    active = [c for c in names if w[c] > 1e-6] or names

    Ys = {c: _opp_matrix(T, policies[c]) for c in names}
    Ymean = sum(w[c] * Ys[c] for c in names)
    allow = _allowed_actions(T, Ymean, qi_floor, floor_win)

    pay = T.pay
    if zero_penalty > 0:
        zero_mask = np.zeros((S, T.maskA.shape[1]), dtype=np.float64)
        for si, (qa, _qp) in enumerate(T.states):
            for i, a in enumerate(T.movesA[si]):
                if next_qi(qa, a) == 0:
                    zero_mask[si, i] = 1.0
    else:
        zero_mask = None

    kappa = min(1.0, max(0.0, float(kappa)))
    V = np.zeros(S)
    Qmean = np.zeros((S, T.maskA.shape[1]))
    Qlo = np.zeros_like(Qmean)
    Qs: Dict[str, np.ndarray] = {c: np.zeros_like(Qmean) for c in names}
    sweeps = 0
    for sw in range(1, max_sweeps + 1):
        sweeps = sw
        cont = gamma * V[T.succ] - step_cost
        if zero_mask is not None:
            cont = cont - zero_penalty * zero_mask[:, :, None]
        M = np.where(T.term, pay, cont)
        Qs = {c: np.einsum("sab,sb->sa", M, Ys[c]) for c in names}
        Qmean = sum(w[c] * Qs[c] for c in names)
        # Q_lo 始终计算：κ=0 时不用它定价，但分歧量要能追踪（否则 κ=0 的 spread 恒为 0）
        Qlo = np.minimum.reduce([Qs[c] for c in active]) if len(active) > 1 else Qmean
        Q = ((1.0 - kappa) * Qmean + kappa * Qlo) if kappa > 0 else Qmean
        Qm = np.where(allow, Q, -1e9)
        Vn = Qm.max(1)
        d = float(np.abs(Vn - V).max())
        V = Vn
        if d < tol:
            break

    def _tab(M: np.ndarray) -> Policy:
        out: Policy = {}
        for si, (qa, qp) in enumerate(T.states):
            va = T.movesA[si]
            n = len(va)
            out[(qa, qp)] = {va[i]: float(M[si, i]) for i in range(n)}
        return out

    q_mean, q_lo = _tab(Qmean), _tab(Qlo)
    Qrisk = (1.0 - kappa) * Qmean + kappa * Qlo
    q_risk = _tab(Qrisk)
    q_channels = {c: _tab(Qs[c]) for c in names}

    policy: Policy = {}
    for si, (qa, qp) in enumerate(T.states):
        va = T.movesA[si]
        n = len(va)
        q = np.where(allow[si, :n], Qrisk[si, :n], -1e9)
        best = float(q.max())
        idx = [i for i in range(n) if q[i] >= best - 1e-9]
        policy[(qa, qp)] = {va[i]: 1.0 / len(idx) for i in idx}

    values = {(qa, qp): float(V[si]) for si, (qa, qp) in enumerate(T.states)}
    gaps = np.array([float(np.mean(Qmean[si, :len(T.movesA[si])]
                                - Qlo[si, :len(T.movesA[si])]))
                     for si in range(S)])
    return ChannelBRResult(
        mode=mode, gamma=gamma, kappa=kappa, weights=w, policy=policy,
        values=values, q_risk=q_risk, q_mean=q_mean, q_lo=q_lo,
        q_channels=q_channels,
        meta={"sweeps": sweeps, "channels": names,
              "mean_spread": float(gaps.mean()) if S else 0.0})
