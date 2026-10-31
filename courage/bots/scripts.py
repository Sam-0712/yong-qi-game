"""脚本化对抗对手（有计划地生成对抗样本）。

真实人类数据稀缺，故用一组覆盖策略空间的 bot 作为训练/评估对手。
每个 bot 提供 ``policy(qa, qp, mode, last_opp) -> {move: prob}``，
既可确定性也可随机；``best_response`` 给出对其的解析最佳响应。

覆盖维度：
  随机 | 进攻 | 防守 | 龟缩 | 循环 | 反跳 | 镜像
"""

from __future__ import annotations

import random
from typing import Dict, Optional

from ..core.rules import MOVES_ATTR, HIT_JUMP, get_valid_moves, judge


def _norm(d: Dict[str, float]) -> Dict[str, float]:
    t = sum(d.values())
    return {k: v / t for k, v in d.items()} if t > 0 else d


def sample_policy(policy: Dict[str, float],
                  rng: Optional[random.Random] = None) -> str:
    rng = rng or random
    r = rng.random()
    cum = 0.0
    last = None
    for m, p in policy.items():
        last = m
        cum += p
        if r <= cum:
            return m
    return last if last is not None else "吐气"


class Bot:
    name = "bot"

    def policy(self, qa: int, qp: int, mode: int = 0,
               last_opp: Optional[str] = None) -> Dict[str, float]:
        raise NotImplementedError

    def act(self, qa: int, qp: int, mode: int = 0,
            last_opp: Optional[str] = None,
            rng: Optional[random.Random] = None) -> str:
        return sample_policy(self.policy(qa, qp, mode, last_opp), rng)


class RandomBot(Bot):
    name = "random"

    def policy(self, qa, qp, mode=0, last_opp=None):
        va = get_valid_moves(qa, mode=mode)
        return {m: 1.0 / len(va) for m in va}


class AggressiveBot(Bot):
    """0.75 概率用可及最高攻击，其余随机。"""
    name = "aggressive"

    def policy(self, qa, qp, mode=0, last_opp=None):
        va = get_valid_moves(qa, mode=mode)
        p = {m: 0.25 / len(va) for m in va}
        atk = [m for m in va if MOVES_ATTR[m][1] > 0]
        if atk:
            best = max(atk, key=lambda m: (MOVES_ATTR[m][1], MOVES_ATTR[m][2]))
            p[best] += 0.75
        return _norm(p)


class DefensiveBot(Bot):
    """偏防御/跳/吐气。"""
    name = "defensive"

    def policy(self, qa, qp, mode=0, last_opp=None):
        va = get_valid_moves(qa, mode=mode)
        p = {m: 0.1 / len(va) for m in va}
        for m, w in (("防御", 0.5), ("跳", 0.3), ("吐气", 0.1)):
            if m in p:
                p[m] += w
        return _norm(p)


class TurtleBot(Bot):
    """永远龟缩：能防御就防御，否则跳/吐气。"""
    name = "turtle"

    def policy(self, qa, qp, mode=0, last_opp=None):
        va = get_valid_moves(qa, mode=mode)
        for pref in ("防御", "跳", "吐气"):
            if pref in va:
                return {pref: 1.0}
        return {va[0]: 1.0}


class CyclerBot(Bot):
    """按 (qa+qp) 确定性循环出招。"""
    name = "cycler"

    def policy(self, qa, qp, mode=0, last_opp=None):
        va = get_valid_moves(qa, mode=mode)
        idx = (qa + qp) % len(va)
        return {va[idx]: 1.0}


class AntiJumpBot(Bot):
    """识别到对手上一招是「跳」就用来得及命中的招式，否则进攻。"""
    name = "anti_jump"

    def policy(self, qa, qp, mode=0, last_opp=None):
        va = get_valid_moves(qa, mode=mode)
        if last_opp == "跳":
            hits = [m for m in va if m in HIT_JUMP]
            if hits:
                best = max(hits, key=lambda m: MOVES_ATTR[m][1])
                return {best: 1.0}
        return AggressiveBot().policy(qa, qp, mode, last_opp)


class MirrorBot(Bot):
    """复制对手上一招（合法时），否则进攻。"""
    name = "mirror"

    def policy(self, qa, qp, mode=0, last_opp=None):
        va = get_valid_moves(qa, mode=mode)
        if last_opp in va:
            return {last_opp: 1.0}
        return AggressiveBot().policy(qa, qp, mode, last_opp)


ALL_BOTS = [
    RandomBot(), AggressiveBot(), DefensiveBot(), TurtleBot(),
    CyclerBot(), AntiJumpBot(), MirrorBot(),
]
bots_by_name = {b.name: b for b in ALL_BOTS}


def best_response(qa: int, qp: int, opp_policy: Dict[str, float],
                  mode: int = 0) -> str:
    """对给定对手分布的解析最佳响应（单步，未考虑未来气）。"""
    va = get_valid_moves(qa, mode=mode)

    def ev(a: str) -> float:
        return sum(p * judge(a, m) for m, p in opp_policy.items())

    return max(va, key=ev)
