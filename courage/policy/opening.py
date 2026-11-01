"""开局簿：用大轮次模拟的对局胜率决定首招分布（3.0 思路的 5.x 形态）。

为什么不用解析 BR 定首招：
  BR 是**确定性 argmax**，在无历史时会把所有概率压到一招上。对手一旦
  发现（人类几局就能发现），就能用固定脚本吃掉这一招——实测「防御→
  吐气→吐气→咏春」对旧版 20/20 全胜。

做法（保留 3.0 的蒙特卡洛调优）：
  1. 对每个合法首招跑 K 局（后续走正式策略），得到该首招的得分率；
  2. Laplacian 平滑 + 温度软化 → 概率分布；
  3. 采样出招 ⇒ 每招都有可能、好招占多、且**不可被脚本化**；
  4. 跨局统计可对该分布做轻微偏置（对手开局偏好 / 我方对该对手的首招得分）。

首招分布本身也是可分析的：同一对手反复以某招开局，就是稳定的开局倾向。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.rules import MOVE_CATEGORY, MOVES_ATTR


def _norm(d: Dict[str, float]) -> Dict[str, float]:
    s = sum(max(v, 0.0) for v in d.values())
    if s <= 0:
        n = len(d) or 1
        return {k: 1.0 / n for k in d}
    return {k: max(v, 0.0) / s for k, v in d.items()}


@dataclass
class OpeningBook:
    """首招概率表。probs 是「基础分布」，采样时再叠加对手偏置与随机性。"""

    probs: Dict[str, float]
    meta: Dict[str, Any] = field(default_factory=dict)

    # ── 采样 ──
    def dist(self, legal: List[str],
             opp_open: Optional[Dict[str, float]] = None,
             opp_weight: float = 0.8, eps: float = 0.04) -> Dict[str, float]:
        """首招的实际概率分布（采样前的最后一步）。

        单独抽出来是为了让评测能拿到「这一手本来是多少概率」——regret 用期望值
        计算比用实际采样到的那一手更稳（去掉智能体自身的采样噪声）。
        """
        legal = [m for m in legal if m]
        if not legal:
            return {}
        p = {m: max(float(self.probs.get(m, 0.0)), 0.0) for m in legal}
        if sum(p.values()) <= 0:                     # 表里没有这些招 → 均匀
            p = {m: 1.0 for m in legal}
        p = _norm(p)
        if eps > 0:
            u = 1.0 / len(legal)
            p = {m: (1.0 - eps) * p[m] + eps * u for m in legal}

        if opp_open:
            q = _norm({m: float(opp_open.get(m, 0.0)) for m in legal})
            w = max(0.0, min(1.0, opp_weight))
            p = {m: (1.0 - w) * p[m] + w * q.get(m, 0.0) for m in legal}
            p = _norm(p)
        return p

    def sample(self, legal: List[str], rng: Optional[random.Random] = None,
               opp_open: Optional[Dict[str, float]] = None,
               opp_weight: float = 0.8, eps: float = 0.04) -> Optional[str]:
        """按分布采样一首招。

        eps         ：与均匀分布混合，保证「任何合法招都可能出现」。
        opp_open    ：跨局统计出的对手首招倾向；按首回合期望收益轻微偏置。
        opp_weight  ：偏置强度（0 关闭）。
        """
        rng = rng or random.Random()
        p = self.dist(legal, opp_open=opp_open, opp_weight=opp_weight, eps=eps)
        legal = list(p)
        if not legal:
            return None

        r = rng.random()
        cum = 0.0
        last = legal[-1]
        for m in legal:
            last = m
            cum += p[m]
            if r <= cum:
                return m
        return last

    def tilt_by_scores(self, scores: Dict[str, float], strength: float = 1.0,
                       eps: float = 0.04) -> "OpeningBook":
        """按「我方首招对某对手的实测得分」重排分布（跨局学习用）。

        得分率∈[0,1]，用 exp(λ·(score-0.5)) 做乘性偏置，再与均匀混合，
        保证不会退化成确定性（防脚本化的底线）。
        """
        p = dict(self.probs)
        for m, s in scores.items():
            if m in p:
                p[m] *= math.exp(strength * (float(s) - 0.5))
        legal = list(p)
        p = _norm(p)
        if eps > 0:
            u = 1.0 / len(legal)
            p = {m: (1.0 - eps) * p[m] + eps * u for m in legal}
        return OpeningBook(p, dict(self.meta))

    # ── 序列化 ──
    def to_dict(self) -> Dict[str, Any]:
        return {"probs": self.probs, "meta": self.meta}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "OpeningBook":
        return cls({k: float(v) for k, v in d.get("probs", {}).items()},
                   dict(d.get("meta", {})))


def scores_to_probs(scores: Dict[str, float], temp: float = 0.12,
                    eps: float = 0.04) -> Dict[str, float]:
    """得分率 → 首招概率（温度软化 + 均匀混合）。"""
    if not scores:
        return {}
    mx = max(scores.values())
    w = {m: math.exp((s - mx) / max(temp, 1e-6)) for m, s in scores.items()}
    p = _norm(w)
    if eps > 0:
        u = 1.0 / len(p)
        p = {m: (1.0 - eps) * v + eps * u for m, v in p.items()}
    return _norm(p)


DEFAULT_BOOK_NAME = "opening_mode0.json"

# 内置「安全开局」分布（无训练产物时的兜底）。
# 依据：3 气时没有任何招式能击穿「防御」(def3)，所以首招「防御」在本回合不可输；
# 其余质量只用于制造随机性（避免被脚本化），不是主选。
_SAFE_WEIGHTS = {
    "防御": 0.70, "双馒": 0.13, "雷切": 0.08,
    "吐气": 0.06, "馒头": 0.02, "跳": 0.01, "无气拨": 0.05,
}


def safe_book(legal: List[str], eps: float = 0.02) -> OpeningBook:
    """内置安全开局簿：按 legal 裁剪并重归一，保证每招都有极小概率。"""
    p = {m: _SAFE_WEIGHTS.get(m, 0.02) for m in legal}
    p = _norm(p)
    if eps > 0:
        u = 1.0 / len(p)
        p = {m: (1.0 - eps) * v + eps * u for m, v in p.items()}
    return OpeningBook(_norm(p), {"source": "builtin-safe"})


STYLE_OPENING = {
    # 风格 = 对「招式类别」的先验偏好；乘以 MC 得分再归一。
    # 只表达"这个风格想怎么打"，具体数值仍由大样本对局决定。
    "balanced":   {"attack": 1.0, "heavy": 1.0, "defense": 1.55, "dodge": 0.5, "exhale": 1.0, "noqi": 0.5},
    "aggressive": {"attack": 1.6, "heavy": 1.3, "defense": 0.85, "dodge": 0.4, "exhale": 0.8, "noqi": 0.5},
    "defensive":  {"attack": 0.6, "heavy": 0.6, "defense": 2.1, "dodge": 1.2, "exhale": 1.1, "noqi": 0.6},
    "gambler":    {"attack": 1.35, "heavy": 2.0, "defense": 0.75, "dodge": 0.6, "exhale": 0.7, "noqi": 0.5},
    "thinker":    {"attack": 1.05, "heavy": 0.9, "defense": 1.75, "dodge": 0.8, "exhale": 1.3, "noqi": 0.5},
}

_CAT_OF = MOVE_CATEGORY          # 定义在 core/rules.py，与因子/上下文/风格共用


def build_style_book(scores: Dict[str, float], style: str = "balanced",
                     temp: float = 0.12, eps: float = 0.04,
                     start_qi: int = 3, dump_cap: float = 0.20) -> OpeningBook:
    """MC 得分 × 风格先验 → 该风格的首招分布。

    scores 是"这一手在这个风格下续战"的实测得分率；先 softmax 软化，
    再乘风格对招式类别的偏好，最后与均匀分布混合（保证每招都可能）。

    dump_cap：**开局把气一次打光的招（如 3 气时的雷切）占比上限**。
    理由与风格无关：打光气后下一回合没有任何攻击手段（0 气只能吐气/防御/跳），
    等于主动交出节奏——实测这条分支必败。风格可以激进，但不该自杀。
    """
    if not scores:
        return uniform_book([])
    mx = max(scores.values())
    base = {m: math.exp((s - mx) / max(temp, 1e-6)) for m, s in scores.items()}
    w = STYLE_OPENING.get(style, {})
    p = {m: v * w.get(_CAT_OF.get(m, "attack"), 1.0) for m, v in base.items()}
    p = _norm(p)

    # 打光气的招限幅，溢出按比例分给其他招（mode=1 起始 0 气时全部 cost=0，不做限制）
    if dump_cap is not None and 0 < dump_cap < 1 and start_qi > 0:
        risky = [m for m in p
                 if MOVES_ATTR.get(m, [0])[0] >= start_qi and MOVES_ATTR.get(m, [0])[0] > 0]
        excess = sum(max(0.0, p[m] - dump_cap) for m in risky)
        if excess > 1e-9:
            for m in risky:
                p[m] = min(p[m], dump_cap)
            others = [m for m in p if m not in risky]
            tot = sum(p[m] for m in others)
            if tot > 1e-9:
                for m in others:
                    p[m] += excess * p[m] / tot

    p = _norm(p)
    if eps > 0:
        u = 1.0 / len(p)
        p = {m: (1.0 - eps) * v + eps * u for m, v in p.items()}
    return OpeningBook(_norm(p), {"style": style, "scores": dict(scores),
                                  "temp": temp, "eps": eps, "dump_cap": dump_cap})


def style_book_name(mode: int, style: str) -> str:
    return f"opening_mode{mode}_{style}.json"


def load_style_book(legal: List[str], mode: int, style: str) -> OpeningBook:
    """按风格加载开局簿；缺失则用内置安全分布（再按风格加权）。"""
    book = load_opening_book(style_book_name(mode, style))
    if book is None:
        base = safe_book(legal)
        w = STYLE_OPENING.get(style, {})
        p = {m: base.probs[m] * w.get(_CAT_OF.get(m, "attack"), 1.0) for m in legal}
        return OpeningBook(_norm(p), {"source": "builtin-safe", "style": style})
    trimmed = {m: book.probs.get(m, 0.0) for m in legal}
    if sum(trimmed.values()) <= 1e-9:
        return safe_book(legal)
    return OpeningBook(_norm(trimmed), dict(book.meta))


def load_or_safe(legal: List[str], name: str = DEFAULT_BOOK_NAME) -> OpeningBook:
    """优先用训练好的开局簿，缺失则用内置安全分布。"""
    book = load_opening_book(name)
    if book is None:
        return safe_book(legal)
    trimmed = {m: book.probs.get(m, 0.0) for m in legal}
    if sum(trimmed.values()) <= 1e-9:
        return safe_book(legal)
    return OpeningBook(_norm(trimmed), dict(book.meta))


def load_opening_book(name: str = DEFAULT_BOOK_NAME) -> Optional[OpeningBook]:
    from ..storage.artifacts import load_json
    blob = load_json(name)
    return OpeningBook.from_dict(blob) if blob else None


def uniform_book(moves: List[str]) -> OpeningBook:
    return OpeningBook({m: 1.0 / len(moves) for m in moves},
                       {"source": "uniform"})
