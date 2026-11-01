"""风格（style）：整体攻/防偏好——低维、与状态无关、可参数化。

先把两个概念分开（此前代码里混着叫"风格"，是真实的混乱源）：

====================  ======================================  ======================
概念                    含义                                      代码里长什么样
====================  ======================================  ======================
**风格 style**            整体攻击/防御（含跳、蓄气）的偏好比例。        5 个权重（本模块）
                        低维、无"思路"、只依赖当前状态。             ``StyleProfile``
                        真人更接近这一类：他有一个倾向，没有剧本。
**打法 pattern**          一个特定思路或局部定势，往往依赖相位：        ``bots/opponents.py``
                        第 1 手防御后转攻 / 低气吐气馒头交替 /       ``OpponentSpec.needs``
                        能收割就收割 / 复制你上一招。
====================  ======================================  ======================

这不是措辞问题，而是**能不能被学习**的区别：

- 风格是**参数估计**问题：几场之后就能估出「他大概 6 成攻击」，样本效率高，
  而且天然泛化到没见过的对手——它不要求对手属于某个已知集合。
- 打法是**模型选择/序列预测**问题：只能靠相位去索引（回合数、双方上一招），
  样本效率低，且只对见过的套路有效。

两者在评测里的角色也不同：打法库测「已知套路打不打得过」，
风格采样测「**没见过**的倾向打不打得过」——后者才是真人的分布外考题。

注意：本模块的 `profile_policy` 是**由权重推出**的招式分布（类别内均分），
不含任何"思路"。任何需要看历史/回合才能决定的东西，都不属于风格。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from ..core.rules import CATEGORY_NAMES, MOVE_CATEGORY, get_valid_moves

# 风格维度的顺序（noqi 只在无气拨模式有意义，采样时给 0）
STYLE_DIMS: List[str] = ["attack", "heavy", "defense", "dodge", "exhale"]


@dataclass(frozen=True)
class StyleProfile:
    """一个风格 = 五个类别的偏好权重（未归一化亦可，使用时归一）。"""

    attack: float = 1.0
    heavy: float = 1.0
    defense: float = 1.0
    dodge: float = 1.0
    exhale: float = 1.0

    # ── 基本操作 ──
    def as_dict(self) -> Dict[str, float]:
        return {d: float(getattr(self, d)) for d in STYLE_DIMS}

    def normalized(self) -> Dict[str, float]:
        d = {k: max(0.0, v) for k, v in self.as_dict().items()}
        s = sum(d.values())
        if s <= 1e-12:
            return {k: 1.0 / len(d) for k in d}
        return {k: v / s for k, v in d.items()}

    def tilted(self, factor: float = 1.0) -> "StyleProfile":
        """整体变锋利/变平：factor>1 更极端，<1 更接近均匀。"""
        n = self.normalized()
        p = {k: max(v, 1e-6) ** factor for k, v in n.items()}
        s = sum(p.values())
        return StyleProfile(**{k: p[k] / s for k in STYLE_DIMS})

    def label(self) -> str:
        """人读标签：取最强与最弱的两个维度。"""
        n = self.normalized()
        order = sorted(n.items(), key=lambda kv: -kv[1])
        return f"{order[0][0]}主/{order[-1][0]}弱"

    def key(self) -> str:
        return "atk%.2f-heavy%.2f-def%.2f-dodge%.2f-exh%.2f" % tuple(
            self.normalized()[d] for d in STYLE_DIMS)

    def to_dict(self) -> Dict[str, float]:
        return {"kind": "style", **self.as_dict()}


# 均匀风格 = 「没有偏好」，等价于招式层面接近均匀（但不完全，见 profile_policy）
BALANCED = StyleProfile()

# 几种典型风格，用于对照（不是"打法"：它们都不看历史）
CLASSIC: Dict[str, StyleProfile] = {
    "style_turtle": StyleProfile(attack=0.2, heavy=0.1, defense=6.0,
                                 dodge=2.0, exhale=1.0),
    "style_guard": StyleProfile(attack=1.0, heavy=0.5, defense=4.0,
                                dodge=2.0, exhale=1.0),
    "style_press": StyleProfile(attack=5.0, heavy=2.0, defense=1.0,
                                dodge=0.5, exhale=1.0),
    "style_heavy": StyleProfile(attack=2.0, heavy=4.0, defense=1.5,
                                dodge=0.5, exhale=1.0),
    "style_evasive": StyleProfile(attack=1.5, heavy=0.5, defense=2.0,
                                  dodge=4.0, exhale=1.5),
    "style_battery": StyleProfile(attack=1.0, heavy=1.0, defense=2.0,
                                  dodge=1.0, exhale=4.0),
}


def profile_policy(profile: StyleProfile, own: int, opp: int = 0,
                   mode: int = 0) -> Dict[str, float]:
    """风格 → 该状态下的出招分布。

    做法只有一步：**类别偏好按合法集摊开，类别内均分**。
        p(m) ∝ w[cat(m)] / #{合法招里同类的个数}
    这样一个「进攻 6 成」的风格就是真的 6 成进攻，而不会因为某一类里有 3 招、
    另一类只有 1 招而被合法集大小悄悄带偏。

    这里**没有**任何依赖历史/回合的分支——有分支的就不叫风格了。
    """
    legal = get_valid_moves(own, mode=mode)
    if not legal:
        return {}
    w = profile.normalized()
    count: Dict[str, int] = {}
    for m in legal:
        c = MOVE_CATEGORY.get(m, "attack")
        count[c] = count.get(c, 0) + 1
    p = {}
    for m in legal:
        c = MOVE_CATEGORY.get(m, "attack")
        p[m] = max(w.get(c, 0.0), 0.0) / max(count[c], 1)
    s = sum(p.values())
    if s <= 1e-12:                      # 偏好全落在当前不合法的类别上
        u = 1.0 / len(legal)
        return {m: u for m in legal}
    return {m: v / s for m, v in p.items()}


def sample_profile(rng: random.Random, alpha: float = 1.0,
                   sharpness: Optional[float] = None) -> StyleProfile:
    """从 Dirichlet(alpha) 采一个风格。

    ``alpha`` 越小越容易出现极端风格（纯防、纯攻）；越大越接近均匀。
    ``sharpness`` 不为 None 时对结果再做一次幂次锐化。
    """
    g = [rng.gammavariate(max(alpha, 1e-3), 1.0) for _ in STYLE_DIMS]
    s = sum(g) or 1.0
    prof = StyleProfile(**{d: g[i] / s for i, d in enumerate(STYLE_DIMS)})
    if sharpness is not None:
        prof = prof.tilted(sharpness)
    return prof


def sample_zoo(n: int, seed: int = 0, alpha: float = 1.0,
               sharpness: Optional[float] = None) -> List[StyleProfile]:
    """一批"没见过的风格"。用来做分布外泛化测验。"""
    rng = random.Random(seed)
    return [sample_profile(rng, alpha, sharpness) for _ in range(n)]


def style_spec_name(profile: StyleProfile) -> str:
    return "style:" + profile.key()


class StyleOpponent:
    """把 StyleProfile 包成 ``needs=()`` 的对手（供 aug_mdp 的精确 oracle 使用）。"""

    needs: Sequence[str] = ()
    kind = "style"
    ood = True

    def __init__(self, profile: StyleProfile, name: str = ""):
        self.profile = profile
        self.name = name or style_spec_name(profile)
        self.label = f"风格 {profile.label()}"
        self.note = ""

    def policy(self, own: int, opp: int, t: int = 0,
               own_last: Optional[str] = None, opp_last: Optional[str] = None,
               mode: int = 0) -> Dict[str, float]:
        return profile_policy(self.profile, own, opp, mode=mode)


class StyleOpponentAgent:
    """风格对手的对局对象（act/observe/reset）。"""

    def __init__(self, profile: StyleProfile, rng: Optional[random.Random] = None,
                 mode: int = 0):
        self.profile = profile
        self.rng = rng or random.Random()
        self.mode = mode

    def reset(self) -> None:
        pass

    def dist(self, own: int, opp: int) -> Dict[str, float]:
        return profile_policy(self.profile, own, opp, mode=self.mode)

    def act(self, own: int, opp: int) -> str:
        d = self.dist(own, opp)
        if not d:
            return "吐气"
        r = self.rng.random()
        cum = 0.0
        last = None
        for m, p in d.items():
            last = m
            cum += p
            if r <= cum:
                return m
        return last or "吐气"

    def observe(self, own: int, opp: int, own_move: str, opp_move: str) -> None:
        pass


def cats_of(move: str) -> str:
    return MOVE_CATEGORY.get(move, "attack")


__all__ = ["STYLE_DIMS", "CATEGORY_NAMES", "StyleProfile", "BALANCED", "CLASSIC",
           "profile_policy", "sample_profile", "sample_zoo", "StyleOpponent",
           "StyleOpponentAgent", "style_spec_name", "cats_of"]
