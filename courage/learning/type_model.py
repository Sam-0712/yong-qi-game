"""对手类型后验：用一组「脚本类型」的似然更新，替代固定常数的幽灵权重。

为什么需要（实测）：
- R1 很多脚本都出「防御」；R2 才分叉（4 手套路吐气 / 2 手套路馒头）。
- 决策发生在 R2 之前，单看一阶频率/短序列无法区分；固定 robust 幽灵
  只能在「4 手 80% / 2 手 16%」与「4 手 20% / 2 手 74%」之间跷跷板。
- 正确做法是维护类型后验：P(type | 已观测序列)，再按类型混合出 μ。

类型策略全部由规则/脚本定义，不含手调概率；后验用贝叶斯似然更新，
观测到不可能的动作时只降权不清零（保留对非平稳对手的适应性）。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..bots.opponents import EXPLOIT4 as _EXPLOIT4
from ..bots.opponents import OPPONENTS, TYPE_SET
from ..core.rules import get_valid_moves
from .threat import converter_policy  # noqa: F401  (保留导出，供旧脚本使用)

EXPLOIT4 = list(_EXPLOIT4)          # 兼容旧名字；定义在 bots/opponents.py
DEFAULT_TYPES = list(TYPE_SET)      # 类型集合与 style 库保持单一来源


def _uniform(legal: List[str]) -> Dict[str, float]:
    u = 1.0 / len(legal) if legal else 0.0
    return {m: u for m in legal}


def _norm(d: Dict[str, float], legal: List[str]) -> Dict[str, float]:
    p = {m: max(float(d.get(m, 0.0)), 0.0) for m in legal}
    s = sum(p.values())
    if s <= 1e-12:
        return _uniform(legal)
    return {m: v / s for m, v in p.items()}


class OpponentTypeModel:
    """类型后验 + 类型混合 μ。接口与 BeliefTracker 的 opp_weights 对齐。"""

    def __init__(self, mode: int = 0, types: Optional[List[str]] = None,
                 prior: Optional[Dict[str, float]] = None,
                 decay: float = 1.0, eps: float = 0.02,
                 smooth: float = 0.02, max_steps: int = 24):
        self.mode = mode
        self.types = list(types or DEFAULT_TYPES)
        self.decay = min(1.0, max(0.0, decay))
        self.eps = max(1e-6, eps)
        self.smooth = min(1.0, max(0.0, smooth))
        self.max_steps = max_steps
        p = prior or {}
        w = {t: max(0.0, float(p.get(t, 1.0))) for t in self.types}
        if sum(w.values()) <= 1e-12:
            w = {t: 1.0 for t in self.types}
        s = sum(w.values())
        self.prior = {t: w[t] / s for t in self.types}
        self.reset()

    def reset(self) -> None:
        self.w = dict(self.prior)
        self.steps = 0
        self.last_own: Optional[str] = None

    # ── 类型策略 ──
    def _policy(self, t: str, qa: int, qp: int,
                last_own: Optional[str]) -> Dict[str, float]:
        """类型 t 的剧本分布（**对手视角**：qp 是对手自己的气数）。

        剧本全部来自 ``courage/bots/opponents.py``——类型后验「假设的对手」与评测
        「实际跑的对手」必须是同一个对象，否则命中率/胜率测的是别的东西。
        """
        legalB = get_valid_moves(qp, mode=self.mode)
        if not legalB:
            return {}
        spec = OPPONENTS.get(t)
        if spec is None:
            return _uniform(legalB)
        return spec.policy(qp, qa, t=self.steps, own_last=None,
                           opp_last=last_own, mode=self.mode)

    # ── 观测更新 ──
    def observe(self, qa: int, qp: int, own_move: str, opp_move: str) -> None:
        lik = {}
        for t in self.types:
            p = self._policy(t, qa, qp, self.last_own)
            lik[t] = p.get(opp_move, 0.0)
        w = {}
        for t in self.types:
            w[t] = (self.w[t] ** self.decay) * (self.eps + lik[t])
        s = sum(w.values())
        if s <= 1e-12:
            w = dict(self.prior)
        else:
            w = {t: v / s for t, v in w.items()}
        if self.smooth > 0:
            w = {t: (1.0 - self.smooth) * w[t] + self.smooth * self.prior[t]
                 for t in self.types}
        self.w = w
        self.steps = min(self.steps + 1, self.max_steps)
        self.last_own = own_move

    # ── 当前 μ ──
    def opp_weights(self, qa: int, qp: int, legal: List[str],
                    last_own: Optional[str] = None) -> Dict[str, float]:
        if not legal:
            return {}
        lo = self.last_own if last_own is None else last_own
        w = {m: 0.0 for m in legal}
        for t, wt in self.w.items():
            if wt <= 0:
                continue
            p = self._policy(t, qa, qp, lo)
            for m, pv in p.items():
                if m in w:
                    w[m] += wt * pv
        w = _norm(w, legal)
        if self.smooth > 0:
            u = _uniform(legal)
            w = {m: (1.0 - self.smooth) * w[m] + self.smooth * u[m] for m in legal}
        return w

    def top_types(self, k: int = 3) -> List[tuple[str, float]]:
        return sorted(self.w.items(), key=lambda kv: -kv[1])[:k]
