"""对手库：**打法/定势**（pattern）的唯一定义来源。

术语分界（这次重构的核心，见 ``bots/style_profiles.py`` 的对照表）：

- **风格 style**：整体攻/防偏好，低维、可参数化、只依赖当前状态 → 在
  ``style_profiles.py``。真人更像这一类。
- **打法 pattern**：一个特定思路或局部定势，通常依赖相位（回合数、自己上一招、
  我方上一招）→ 在本文件，用 ``needs`` 明确声明它到底读哪些相位。

``needs`` 不只是元数据：它决定精确 oracle（``solvers/aug_mdp.py``）的状态空间
要扩到多大。``()`` = 纯状态函数，36 格就够；``("t",)`` = 依赖回合数（4 手/2 手
套路）；``("own_last",)`` = 依赖对手自己上一招（低气交替）；``("opp_last",)`` =
依赖我方上一招（镜像/反跳）。**凡是 needs 非空的，都不可能被"风格参数"表达。**

为什么要有这个文件：同一批对手策略此前在三个地方各写了一份，且已经不一致——
  1. 语料生成   ``tools/gen_corpus.py``（曾有 _pressure/_alternate_low/_exploit4/_two/_converter）
  2. 类型后验库 ``courage/learning/type_model.py``（曾有 EXPLOIT4 + two/exploit4/converter）
  3. 评测对手   ``tools/win_policy.py``（曾有 _strong + 4手/2手/转化者）
类型后验「假设对手是剧本 X」和评测「实际跑的是剧本 Y」必须是同一个对象，
否则测出来的命中率/胜率测的是别的东西。

每个条目提供两个接口：
  ``make_opponent(name, seed)``  → ``act/observe/reset`` 的对局对象（采样出招）
  ``OPPONENTS[name].policy(...)`` → 给定相位返回 ``{招式: 概率}``（供精确 oracle 用）

⚠ 三类容易混淆的「转化者」：
  ``converter`` / ``two``(t>=2) : ``converter_policy``，**在**类型后验的集合内
  ``strong``                    : 旧 ``win_policy._strong``——要求「一招打穿全部合法招」
                                  才算必杀，比 converter_policy 更保守，**不在**类型
                                  集合内，属于「集合外压力测试对手」（``ood=True``）。
两者故意保留，因为一个是训练假设、一个是分布外考题。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from ..core.rules import MOVES_ATTR, get_valid_moves, judge
from .scripts import bots_by_name, sample_policy

# 注：这里的模块不能 import courage.learning 下的任何东西——learning/__init__ 会
# import type_model，而 type_model 又 import 本模块，会形成循环导入。
# 所以「理性转化者」的定义放在本文件，由 learning/threat.py 再导出。

# 4 手套路（用户打法），顺序即前四手。
EXPLOIT4: List[str] = ["防御", "吐气", "吐气", "咏春"]

# 「上一招」的索引取值：0 表示 None（对局开始），1..9 对应 MOVES。
LAST_MAX = 10

# 打法分类，只用于报表与文档，不参与计算
KIND_STYLE = "style"        # 纯全局偏好，理论上可被风格参数表达
KIND_PATTERN = "pattern"    # 有思路/定势，依赖相位
KIND_BASELINE = "baseline"  # 无结构


# ── 工具 ──

def _uniform(legal: List[str]) -> Dict[str, float]:
    u = 1.0 / len(legal) if legal else 0.0
    return {m: u for m in legal}


def _norm(d: Dict[str, float], legal: List[str]) -> Dict[str, float]:
    p = {m: max(float(d.get(m, 0.0)), 0.0) for m in legal}
    s = sum(p.values())
    return {m: v / s for m, v in p.items()} if s > 1e-12 else _uniform(legal)


def _dist_of(ret, legal: List[str]) -> Dict[str, float]:
    """脚本返回值 → 分布。dict 直接归一；单招若不合法则退回首个合法招。

    退回策略必须是**确定性**的：旧的语料生成器与旧的评测脚本一直是 ``legal[0]``，
    只有类型后验用均匀分布。三份拷贝里唯一的异类已在此统一——触发条件极罕见
    （仅当脚本要求的那一手因气数不足而不合法时）。
    """
    if isinstance(ret, dict):
        return _norm(ret, legal)
    if not legal:
        return {}
    if ret in legal:
        return {ret: 1.0}
    return {legal[0]: 1.0}


# ── 理性转化者 ──
#
# 为什么需要（实测）：纯 BR + 静态信念对「理性转化者」的胜率是 0%——对手只要一直
# 安全出招（我方打不到他就蓄气、能一击必杀就下手、否则防守），等我们花光气再收割，
# 就能把我们打穿。而它对只会一条路走到黑的 bot 是 100%。均衡策略对同一对手约 48%
# （对称博弈的理论值），说明不是"打不过"，而是**打法可被针对**。
#
# π_转化者 完全由规则推出，不含手调概率：
#   1. 若有一招能赢过我方**全部**合法招 → 用它（真正的必杀）；
#   2. 否则若「吐气」不会被我方任何一招击败 → 蓄气（零风险，最划算）；
#   3. 否则在"不会输"的招里挑防御/攻击最高的；
#   4. 连安全的招都没有 → 硬着头皮出最高攻击。

def expectation(b: str, own_legal: List[str]) -> float:
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
    evs = {b: expectation(b, own_legal) for b in opp_legal}
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


# ── 具体打法（pattern） ──

def pressure(own: int, opp: int, t: int = 0,
             own_last: Optional[str] = None, opp_last: Optional[str] = None) -> str:
    """分散小攻击压制：能双馒就双馒，否则馒头，再否则吐气。只看自己气数。"""
    legal = get_valid_moves(own, mode=0)
    if own >= 2 and "双馒" in legal:
        return "双馒"
    if own >= 1 and "馒头" in legal:
        return "馒头"
    if "吐气" in legal:
        return "吐气"
    return "防御"


def alternate_low(own: int, opp: int, t: int = 0,
                  own_last: Optional[str] = None, opp_last: Optional[str] = None) -> str:
    """低气交替：双方都 <=2 气时在「吐气/馒头」间交替（依赖自己上一招），否则龟缩。"""
    legal = get_valid_moves(own, mode=0)
    if own <= 2 and opp <= 2:
        if own_last == "吐气" and "馒头" in legal:
            return "馒头"
        if own_last == "馒头" and "吐气" in legal:
            return "吐气"
        return "吐气" if "吐气" in legal else "防御"
    return "防御" if "防御" in legal else ("跳" if "跳" in legal else "吐气")


def exploit4(own: int, opp: int, t: int = 0,
             own_last: Optional[str] = None, opp_last: Optional[str] = None) -> str:
    """4 手套路：前四手固定，之后满气咏春/否则吐气。依赖回合数。"""
    legal = get_valid_moves(own, mode=0)
    want = EXPLOIT4[t] if t < len(EXPLOIT4) else ("咏春" if own >= 5 else "吐气")
    return want if want in legal else "吐气"


def two(own: int, opp: int, t: int = 0,
        own_last: Optional[str] = None, opp_last: Optional[str] = None):
    """2 手套路：防御→馒头，之后转为理性转化者。"""
    if t == 0:
        return "防御"
    if t == 1:
        return "馒头" if "馒头" in get_valid_moves(own, mode=0) else "防御"
    return converter_policy(get_valid_moves(own, mode=0), get_valid_moves(opp, mode=0))


def converter(own: int, opp: int, t: int = 0,
              own_last: Optional[str] = None, opp_last: Optional[str] = None):
    """理性转化者（类型集内）：会趁对方无力时蓄气/收割。"""
    return converter_policy(get_valid_moves(own, mode=0), get_valid_moves(opp, mode=0))


def strong(own: int, opp: int, t: int = 0,
           own_last: Optional[str] = None, opp_last: Optional[str] = None) -> str:
    """旧 ``win_policy._strong``：更保守的转化者，**集合外**压力测试对手。

    与 ``converter_policy`` 的差别：只有「一招打穿我方全部合法招」才算必杀，
    否则宁可蓄气/高防。低气局面下它会低估对手的收割能力。
    """
    vb = get_valid_moves(own, mode=0)
    ours = get_valid_moves(opp, mode=0)
    kills = [b for b in vb if all(judge(b, a) == 1 for a in ours)]
    if kills:
        return min(kills, key=lambda m: MOVES_ATTR[m][0])
    safe = [b for b in vb if not any(judge(a, b) == 1 for a in ours)]
    if not safe:
        return max(vb, key=lambda m: MOVES_ATTR[m][1])
    if "吐气" in safe:
        return "吐气"
    return max(safe, key=lambda m: (MOVES_ATTR[m][2], MOVES_ATTR[m][1]))


# ── 注册表 ──

@dataclass(frozen=True)
class OpponentSpec:
    """一个对手打法：脚本函数 或 现成 bot，加上它依赖的相位与分类。"""

    name: str
    fn: Optional[Callable] = None
    bot: Optional[str] = None
    needs: Tuple[str, ...] = ()
    kind: str = KIND_PATTERN
    label: str = ""
    ood: bool = False           # True = 不在类型后验的类型集合内
    note: str = ""

    def policy(self, own: int, opp: int, t: int = 0,
               own_last: Optional[str] = None, opp_last: Optional[str] = None,
               mode: int = 0) -> Dict[str, float]:
        """给定相位返回该打法出招分布。

        own/opp 是**该对手自己**的视角（own=自己的气）。评测与 oracle 都走这里，
        因此「对手会怎么走」只有一处定义。
        """
        legal = get_valid_moves(own, mode=mode)
        if not legal:
            return {}
        if self.bot is not None:
            bot = bots_by_name[self.bot]
            return _norm(bot.policy(own, opp, mode=mode, last_opp=opp_last), legal)
        if self.fn is None:
            return _uniform(legal)
        return _dist_of(self.fn(own, opp, t, own_last, opp_last), legal)


OPPONENTS: Dict[str, OpponentSpec] = {
    # 纯全局偏好（可被风格参数近似）
    "turtle": OpponentSpec("turtle", bot="turtle", kind=KIND_STYLE, label="龟缩"),
    "defensive": OpponentSpec("defensive", bot="defensive", kind=KIND_STYLE,
                              label="防守"),
    "aggressive": OpponentSpec("aggressive", bot="aggressive", kind=KIND_STYLE,
                               label="猛攻"),
    # 有思路/定势
    "cycler": OpponentSpec("cycler", bot="cycler", kind=KIND_PATTERN, label="循环"),
    "mirror": OpponentSpec("mirror", bot="mirror", needs=("opp_last",),
                           label="镜像"),
    "anti_jump": OpponentSpec("anti_jump", bot="anti_jump", needs=("opp_last",),
                              label="反跳"),
    "two": OpponentSpec("two", fn=two, needs=("t",), label="用户2手"),
    "exploit4": OpponentSpec("exploit4", fn=exploit4, needs=("t",), label="用户4手"),
    "converter": OpponentSpec("converter", fn=converter, label="转化者(集合内)"),
    "pressure": OpponentSpec("pressure", fn=pressure, label="分散压制"),
    "alternate_low": OpponentSpec("alternate_low", fn=alternate_low,
                                  needs=("own_last",), label="低气交替"),
    # 无结构基线
    "random": OpponentSpec("random", bot="random", kind=KIND_BASELINE, label="随机"),
    # 集合外压力测试
    "strong": OpponentSpec("strong", fn=strong, label="转化者(集合外)", ood=True,
                           note="旧 win_policy._strong，要求打穿全部合法招才算必杀"),
}

# 兼容旧名字（此前叫 STYLES / StyleSpec / StyleAgent）
STYLES = OPPONENTS
StyleSpec = OpponentSpec

# 类型后验默认使用的那几个（与 type_model.DEFAULT_TYPES 保持一致）
TYPE_SET: Tuple[str, ...] = ("turtle", "defensive", "aggressive", "two",
                             "exploit4", "converter", "mirror", "cycler", "random")


def opponent(name: str) -> OpponentSpec:
    return OPPONENTS[name]


def policy_of(name: str, own: int, opp: int, t: int = 0,
              own_last: Optional[str] = None, opp_last: Optional[str] = None,
              mode: int = 0) -> Dict[str, float]:
    return OPPONENTS[name].policy(own, opp, t, own_last, opp_last, mode)


def names(include_ood: bool = True) -> List[str]:
    return [k for k, v in OPPONENTS.items() if include_ood or not v.ood]


def names_by_kind(kind: str) -> List[str]:
    return [k for k, v in OPPONENTS.items() if v.kind == kind]


# ── 对局对象 ──

class OpponentAgent:
    """把 OpponentSpec 包成 ``act/observe/reset`` 的对局对手。"""

    def __init__(self, spec: OpponentSpec, rng: Optional[random.Random] = None,
                 mode: int = 0):
        self.spec = spec
        self.rng = rng or random.Random()
        self.mode = mode
        self.t = 0
        self.own_last: Optional[str] = None
        self.opp_last: Optional[str] = None

    def reset(self) -> None:
        self.t = 0
        self.own_last = None
        self.opp_last = None

    def dist(self, own: int, opp: int) -> Dict[str, float]:
        return self.spec.policy(own, opp, self.t, self.own_last, self.opp_last,
                                self.mode)

    def act(self, own: int, opp: int) -> str:
        d = self.dist(own, opp)
        if not d:
            legal = get_valid_moves(own, mode=self.mode)
            return legal[0] if legal else "吐气"
        if len(d) == 1:                 # 确定性脚本：不消耗随机数，保持流不变
            return next(iter(d))
        return sample_policy(d, self.rng)

    def observe(self, own: int, opp: int, own_move: str, opp_move: str) -> None:
        self.own_last = own_move
        self.opp_last = opp_move
        self.t += 1

    def __repr__(self) -> str:
        return f"<OpponentAgent {self.spec.name}>"


StyleAgent = OpponentAgent        # 兼容旧名字


def make_opponent(name: str, seed: int = 0, mode: int = 0) -> OpponentAgent:
    return OpponentAgent(OPPONENTS[name], random.Random(seed), mode)


make_agent = make_opponent        # 兼容旧名字
