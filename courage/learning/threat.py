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


def _expectation(b: str, own_legal: List[str]) -> float:
    """假设我方均匀出招时，对手出 b 的平均收益（+1 胜 / 0 平 / -1 负）。"""
    if not own_legal:
        return 0.0
    return sum(judge(b, a) for a in own_legal) / len(own_legal)


def converter_policy(opp_legal: List[str], own_legal: List[str]) -> Dict[str, float]:
    """「会兑现优势」的对手模型。

    旧版要求"一招能赢过我方**全部**合法招"才算必杀，在低气局面会误判：
      (我方 0 气, 对手 4 气) 时，螺旋环能打穿吐气/防御，但打不穿「跳」
      （跳只输给雷切/咏春）。于是旧版认为"无必杀"→ 去蓄气了，
      完全掩盖了"我方 0 气正在被收割"的事实，导致 V(0,·) 被系统性高估。

    新版按**平均收益**排序：只要这一手能稳定打到我们（EV>0），就出它。
    这才是"对手会利用我的弱点"的正确刻画。
    """
    if not opp_legal:
        return {}
    # 1. 真必杀（打穿全部）——最省气的那一手
    kills = [b for b in opp_legal if all(judge(b, a) == 1 for a in own_legal)]
    if kills:
        best = min(kills, key=lambda m: MOVES_ATTR[m][0])
        return {best: 1.0}

    # 2. 能拿到正收益 → 同时保留两种"会兑现优势"的姿态：
    #    最省的一手（节流型：馒头/双馒，只花 1~2 气就能打穿低防招）与
    #    最强的一手（收割型：雷切/螺旋环，一口气打穿高防）。
    #    只取"期望最高"会偏向强招，于是幽灵以为对手会烧光自己的气，
    #    反而掩盖了"低气正在被低成本蚕食"的事实（实测：AI 因此在吐气上
    #    反复送死，被人用馒头打死 38/43 局）。
    evs = {b: _expectation(b, own_legal) for b in opp_legal}
    if max(evs.values()) > 1e-9:
        cands = [b for b in opp_legal if evs[b] > 1e-9]
        cheap = min(cands, key=lambda m: (MOVES_ATTR[m][0], -evs[m]))
        strong = max(cands, key=lambda m: (evs[m], -MOVES_ATTR[m][0]))
        if cheap == strong:
            return {cheap: 1.0}
        return {cheap: 0.5, strong: 0.5}

    # 3. 打不到我们（我方有全能格挡）→ 蓄气最划算，其次高防低耗
    if "吐气" in opp_legal:
        return {"吐气": 1.0}
    best = max(opp_legal, key=lambda m: (MOVES_ATTR[m][2], -MOVES_ATTR[m][0]))
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
