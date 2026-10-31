"""策略混合：均衡（安全网） ↔ 最佳响应（剥削）。

为什么需要这一层（实测结论）：
- 纯 BR 是**确定性 argmax**，且假设对手信念静止。对手只要"先守、等你花光
  气、再收割"，就能把 BR 打到 0%（开局矩阵里对理性转化者全 0.00）。
- 均衡策略不可被针对：同一对手下 58% 胜 / 42% 负（有来有回，不是被吃死）。
- 但因为均衡不剥削，对固定风格的 bot 只有 55~90%，远不如 BR 的 90~100%。

所以按「信念置信度」加权：
    开     局（无信息）→ 均衡为主（随机、各招可达、不可被脚本化）
    有信息之后        → 逐回合转向 BR（开始剥削）
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

Policy = Dict[Tuple[int, int], Dict[str, float]]


def confidence(steps: int, scale: float = 1.5) -> float:
    """λ = 转向 BR 的权重，随观测步数增长（0 → 均衡，1 → 纯 BR）。"""
    if steps <= 0:
        return 0.0
    return 1.0 - math.exp(-steps / max(scale, 1e-6))


def blend(eq: Optional[Dict[str, float]], br: Dict[str, float],
          lam: float) -> Dict[str, float]:
    """把均衡分布与 BR 分布按 λ 混合（只在两者都合法的招上）。"""
    lam = min(1.0, max(0.0, lam))
    if not eq or lam >= 1.0:
        return dict(br)
    keys = set(br) | set(eq)
    p = {m: (1.0 - lam) * eq.get(m, 0.0) + lam * br.get(m, 0.0) for m in keys}
    s = sum(p.values())
    if s <= 1e-12:
        return dict(br)
    return {m: v / s for m, v in p.items()}


def blend_policies(eq_pol: Optional[Policy], br_pol: Policy,
                   lam: float) -> Policy:
    out: Policy = {}
    for k, br in br_pol.items():
        out[k] = blend((eq_pol or {}).get(k), br, lam)
    return out
