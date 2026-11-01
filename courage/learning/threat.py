"""威胁模型：把「理性转化者」作为一个幽灵对手类型混入信念。

为什么需要（实测）：
- 纯 BR + 静态信念对「理性转化者」的胜率是 **0%**：对手只要一直安全出招
  （我方打不到他就蓄气、能一击必杀就下手、否则防守），等我们花光气再收割，
  就能把我们打穿。而它对只会一条路走到黑的 bot 是 100%。
- 均衡策略对同一对手是 ~48%（就是对称博弈的理论值），说明不是"打不过"，
  而是**打法可被针对**。

做法：把对手模型写成
    π(b|s) = (1-β)·π_信念(b|s) + β·π_转化者(b|s)
再用 BR 优化。β 越大越稳、越小越能剥削；β=0 就是旧行为。

π_转化者 的**定义**在 ``courage/bots/opponents.py``（与评测里的 ``converter`` 对手
同源——类型后验假设的剧本和评测实际跑的剧本必须是同一个对象）。
本模块只负责「幽灵威胁」这一层：把它按 β 混进信念。
"""

from __future__ import annotations

from typing import Dict, List

from ..bots.opponents import converter_policy, expectation  # noqa: F401  (对外再导出)

__all__ = ["converter_policy", "expectation", "blend_with_converter"]


def blend_with_converter(w: Dict[str, float], legal: List[str],
                         own_legal: List[str], beta: float) -> Dict[str, float]:
    """信念分布与「转化者」分布混合，返回归一化后的对手模型。"""
    beta = min(1.0, max(0.0, beta))
    if beta <= 0:
        return w
    conv = converter_policy(legal, own_legal)
    if not conv:
        return w
    p = {m: (1.0 - beta) * w.get(m, 0.0) + beta * conv.get(m, 0.0) for m in legal}
    s = sum(p.values())
    return {m: v / s for m, v in p.items()} if s > 1e-12 else dict(w)
