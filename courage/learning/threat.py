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

π_转化者 完全由规则推出，不含任何手调概率：
    1. 若有一招能赢过我方**全部**合法招 → 用它（真正的必杀）；
    2. 否则若「吐气」不会被我方任何一招击败 → 蓄气（零风险，最划算）；
    3. 否则在"不会输"的招里挑防御/攻击最高的；
    4. 连安全的招都没有 → 硬着头皮出最高攻击。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..core.rules import MOVES_ATTR, judge


def converter_policy(opp_legal: List[str], own_legal: List[str]) -> Dict[str, float]:
    """理性转化者在我方合法招为 own_legal 时的出招分布。"""
    if not opp_legal:
        return {}
    kills = [b for b in opp_legal if all(judge(b, a) == 1 for a in own_legal)]
    if kills:
        best = min(kills, key=lambda m: MOVES_ATTR[m][0])
        return {best: 1.0}
    safe = [b for b in opp_legal
            if not any(judge(a, b) == 1 for a in own_legal)]
    if not safe:
        best = max(opp_legal, key=lambda m: MOVES_ATTR[m][1])
        return {best: 1.0}
    if "吐气" in safe:
        return {"吐气": 1.0}
    # 同为安全招时选「高防 + 低耗」：防御 > 雷切（雷切白花 3 气，且不增加安全）
    best = max(safe, key=lambda m: (MOVES_ATTR[m][2], -MOVES_ATTR[m][0]))
    return {best: 1.0}


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
