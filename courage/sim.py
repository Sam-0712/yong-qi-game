"""对局模拟：给定两个「策略函数」，跑一局并返回结果与回合数。

策略函数签名：fn(qa, qp) -> {move: prob}（已在合法集上归一）。
返回：result ∈ {1: AI胜, -1: AI负, 0: 超时未决}, rounds。
"""

from __future__ import annotations

import random
from typing import Callable, Dict, Tuple

from .core.rules import judge, next_qi
from .core.state import MODE_STANDARD

PolicyFn = Callable[[int, int], Dict[str, float]]


def sample_dist(dist: Dict[str, float], rng: random.Random) -> str:
    r = rng.random()
    cum = 0.0
    last = None
    for m, p in dist.items():
        last = m
        cum += p
        if r <= cum:
            return m
    return last


def play_game(policy_ai: PolicyFn, policy_opp: PolicyFn,
              mode: int = MODE_STANDARD, cap: int = 200,
              rng: random.Random | None = None) -> Tuple[int, int]:
    rng = rng or random.Random()
    qa = qp = 3
    for rnd in range(1, cap + 1):
        a = sample_dist(policy_ai(qa, qp), rng)
        b = sample_dist(policy_opp(qa, qp), rng)
        w = judge(a, b)
        if w != 0:
            return w, rnd
        qa = next_qi(qa, a)
        qp = next_qi(qp, b)
    return 0, cap
