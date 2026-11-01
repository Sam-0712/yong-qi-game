"""通道混合：用预测似然做指数加权（Hedge / 贝叶斯模型平均），替代硬切换。

为什么要把 `hybrid_model.HybridOpponentModel` 换掉
--------------------------------------------------
硬切换（按类型后验的 top/质量二选一）有三个已量化的问题（见 tools/bench.py）：

1. **比两条通道都差。** 对集合外的 ``strong`` 对手、低气残局：
   ``type`` 0.086、``context`` 0.252，而硬切换 **1.053** regret/局。
   原因是门控会随历史反复翻面，产出的策略既不等于 A 也不等于 B，
   再加上路径依赖，早一步走错会在后续状态下持续放大。

2. **判据测的不是可信度。** ``known_mass ≈ 1 - w_random`` 几乎是常数，
   ``top_known < 0.25`` 在均匀先验（1/9）下恒真——实际行为是「开局必用上下文，
   观测几手后必翻回类型」，与「我到底该信哪个」无关。

3. **可被反推。** 门控是历史的确定性函数，对手演几手就能把 AI 推到更弱的通道上。

替代方案
--------
两条通道每回合输出的是**同一个合法集上的分布**，因此是一个标准的专家聚合问题：
按各自的实际预测表现做指数加权，再**混合**（而不是选择）。

    log w_c ← decay · log w_c + eta · log p_c(b_obs)
    p(m)    = Σ_c w_c · p_c(m) / Σ_c w_c

``eta=1, decay=1`` 时它就是精确的贝叶斯模型平均（在「对手不变、通道互斥且穷尽」
的假设下）。``prior_mix`` 给每条通道留一个极小概率地板，防止一次巧合把某条通道
永久打死——这一点对非平稳对手很重要。

``uniform`` 作为一条**显式通道**加进来是有意的：面对真正随机的对手，正确答案就是
均匀分布，而硬切换从来没有这个选项（它只能在几个自信的错答案里挑）。
实测 ``random`` 上 ``type`` 0.83 / ``context`` 0.87 regret/局，而 ``hand``（近似均匀）
只有 0.42——缺的就是这条通道。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ..core.rules import get_valid_moves


class UniformChannel:
    """「我完全不知道」也是一条正当的假设。

    对随机型对手它是精确模型；对其它对手它给出最保守的 BR，正好是信息不足时
    该做的事。没有它，聚合只能在几条自信的错答案之间分配权重。
    """

    name = "uniform"
    steps = 0

    def reset(self) -> None:
        pass

    def observe(self, *a) -> None:
        pass

    def opp_weights(self, qa: int, qp: int, legal: List[str],
                    last_own: Optional[str] = None) -> Dict[str, float]:
        if not legal:
            return {}
        u = 1.0 / len(legal)
        return {m: u for m in legal}


class BeliefChannel:
    """把 ``BeliefTracker`` 适配成统一通道接口（它不接收 qa，是历史遗留问题）。"""

    name = "hand"

    def __init__(self, belief):
        self.belief = belief
        self.passive = True          # 由 agent 负责 observe，这里不重复更新

    def reset(self) -> None:
        pass

    def observe(self, *a) -> None:
        pass

    def opp_weights(self, qa: int, qp: int, legal: List[str],
                    last_own: Optional[str] = None) -> Dict[str, float]:
        return self.belief.opp_weights(legal, last_own=last_own)


class MixtureOpponentModel:
    """指数加权通道混合。接口与 ``OpponentTypeModel`` / ``ContextOpponentModel`` 对齐。"""

    def __init__(self, channels: Dict[str, object],
                 passive: Tuple[str, ...] = (),
                 eta: float = 1.0, decay: float = 1.0,
                 prior_mix: float = 0.02, p_floor: float = 0.01,
                 null_weight: float = 0.4, mode: int = 0):
        if not channels:
            raise ValueError("至少需要一条通道")
        self.channels = dict(channels)
        self.names = list(self.channels)
        self.passive = set(passive) | {k for k, v in self.channels.items()
                                       if getattr(v, "passive", False)}
        self.eta = float(eta)
        self.decay = min(1.0, max(0.0, float(decay)))
        self.prior_mix = min(1.0, max(0.0, float(prior_mix)))
        self.p_floor = max(1e-6, float(p_floor))
        self.mode = mode
        # 先验偏向「无结构」这条零假设：对手在被证明有结构之前，按均匀对待。
        # 这既是统计上的惯例（没有证据就不拒绝 null），也是决策上的保守选择：
        # 对均匀 BR 是最大化最坏情况，over-commit 到一条错通道的代价不对称地大。
        self.null_weight = min(1.0, max(0.0, float(null_weight)))
        self.prior = self._make_prior()
        self.logw: Dict[str, float] = {}
        self.prev_own: Optional[str] = None
        self.steps = 0
        self.ll: Dict[str, float] = {}
        self._reset_weights()

    def _make_prior(self) -> Dict[str, float]:
        k = len(self.names)
        has_null = "uniform" in self.names
        if not has_null or k == 1:
            return {c: 1.0 / k for c in self.names}
        rest = (1.0 - self.null_weight) / (k - 1)
        return {c: (self.null_weight if c == "uniform" else rest)
                for c in self.names}

    def _reset_weights(self) -> None:
        self.logw = {c: math.log(max(self.prior[c], 1e-9)) for c in self.names}
        self.ll = {c: 0.0 for c in self.names}

    # ── 生命周期 ──
    def reset(self) -> None:
        for c in self.channels.values():
            fn = getattr(c, "reset", None)
            if callable(fn):
                fn()
        self._reset_weights()
        self.prev_own = None
        self.steps = 0

    # ── 权重 ──
    def weights(self) -> Dict[str, float]:
        if not self.logw:
            return {}
        mx = max(self.logw.values())
        w = {c: math.exp(v - mx) for c, v in self.logw.items()}
        s = sum(w.values())
        w = {c: v / s for c, v in w.items()} if s > 1e-12 else \
            {c: 1.0 / len(w) for c in w}
        if self.prior_mix > 0:
            k = len(w)
            w = {c: (1.0 - self.prior_mix) * v + self.prior_mix / k
                 for c, v in w.items()}
        return w

    def _mix(self, dists: Dict[str, Dict[str, float]], w: Dict[str, float],
             legal: List[str]) -> Dict[str, float]:
        acc = {m: 0.0 for m in legal}
        for c, d in dists.items():
            wc = w.get(c, 0.0)
            if wc <= 1e-9:
                continue
            for m, v in d.items():
                if m in acc:
                    acc[m] += wc * v
        s = sum(acc.values())
        if s <= 1e-12:
            u = 1.0 / len(legal) if legal else 0.0
            return {m: u for m in legal}
        return {m: v / s for m, v in acc.items()}

    # ── 观测：先给每条通道打分，再更新它们 ──
    def observe(self, qa: int, qp: int, own_move: str, opp_move: str) -> None:
        legal = get_valid_moves(qp, mode=self.mode)
        if legal:
            for c, ch in self.channels.items():
                try:
                    d = ch.opp_weights(qa, qp, legal, last_own=self.prev_own)
                except TypeError:                       # 兼容旧签名
                    d = ch.opp_weights(legal, last_own=self.prev_own)
                p = max(float(d.get(opp_move, 0.0)), self.p_floor)
                self.ll[c] = self.ll.get(c, 0.0) - math.log(p)
                self.logw[c] = (self.decay * self.logw[c]
                                + self.eta * math.log(p))
            # 去掉公共漂移，防止数值上溢（权重只看相对值）
            shift = max(self.logw.values())
            if abs(shift) > 1e-12:
                self.logw = {c: v - shift for c, v in self.logw.items()}

        for c, ch in self.channels.items():
            if c in self.passive:
                continue
            fn = getattr(ch, "observe", None)
            if callable(fn):
                fn(qa, qp, own_move, opp_move)
        self.prev_own = own_move
        self.steps += 1

    # ── 推理 ──
    def _channel_dists(self, qa: int, qp: int, legal: List[str],
                       last_own: Optional[str]) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        for c, ch in self.channels.items():
            try:
                out[c] = ch.opp_weights(qa, qp, legal, last_own=last_own)
            except TypeError:
                out[c] = ch.opp_weights(legal, last_own=last_own)
        return out

    def opp_weights(self, qa: int, qp: int, legal: List[str],
                    last_own: Optional[str] = None) -> Dict[str, float]:
        if not legal:
            return {}
        lo = self.prev_own if last_own is None else last_own
        dists = self._channel_dists(qa, qp, legal, lo)
        return self._mix(dists, self.weights(), legal)

    def channel_policies(self, states, last_own: Optional[str] = None
                         ) -> Tuple[Dict[str, Dict], Dict[str, float]]:
        """把 36 个状态的对手模型按通道拆开，供 ``solvers/br_channels.py`` 使用。

        返回 ``({通道: {(qa,qp): {move: p}}}, 通道权重)``。风险定价需要**逐通道的
        Q**，所以这里不能在``opp_weights``里先混合——那样信息就没了。
        """
        lo = self.prev_own if last_own is None else last_own
        out: Dict[str, Dict] = {c: {} for c in self.names}
        for qa, qp in states:
            legal = get_valid_moves(qp, mode=self.mode)
            if not legal:
                continue
            dists = self._channel_dists(qa, qp, legal, lo)
            for c in self.names:
                out[c][(qa, qp)] = dists[c]
        return out, self.weights()

    # ── 诊断 / trace ──
    def breakdown(self, qa: int, qp: int, legal: List[str],
                  last_own: Optional[str] = None
                  ) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
        """返回 (通道权重, 各通道分布)，供 tools/trace.py 逐回合展开。"""
        lo = self.prev_own if last_own is None else last_own
        return self.weights(), self._channel_dists(qa, qp, legal, lo)

    def summary(self) -> str:
        w = self.weights()
        parts = []
        for c in self.names:
            avg = self.ll.get(c, 0.0) / self.steps if self.steps else 0.0
            parts.append(f"{c}:w={w[c]:.2f},ll={avg:.3f}")
        return "  ".join(parts)
