"""item 3：信息增益首招。

把「对手类型」当隐变量（此处用脚本 bot 集合）。对候选首招 a，计算它带来的
关于类型的互信息 I(type; response | a)。对**记忆型**对手(mirror/anti_jump)，
a 会改变其回应，因而有正信息增益；记忆无关对手则 I=0。选 I 最大者作首招，
之后转入信念 BR 剥削。

用法：python -m tools.opening
"""

from __future__ import annotations

import argparse
import math
import os
import random
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.agents import BotAgent, OnlineBRAgent, run_duel
from courage.bots.scripts import bots_by_name
from courage.core.rules import get_valid_moves
from courage.solvers.best_response import solve_best_response
from courage.solvers.equilibrium import build_tables
from courage.policy.settings import PlaySettings


def _entropy(ps):
    return -sum(p * math.log(p + 1e-12) for p in ps if p > 0)


def info_gain_table(mode=0):
    types = list(bots_by_name)
    qi = 3
    cands = get_valid_moves(qi, mode=mode)
    rows = {}
    Ht = math.log(len(types))  # 均匀先验熵
    for a in cands:
        # P(type, response | a)
        joint = {}
        for t in types:
            dist = bots_by_name[t].policy(qi, qi, mode=mode, last_opp=a)  # 对手自身3气、我方3气
            for b, p in dist.items():
                joint[(t, b)] = joint.get((t, b), 0.0) + p / len(types)
        # P(response)
        Pb = {}
        for (t, b), v in joint.items():
            Pb[b] = Pb.get(b, 0.0) + v
        # H(type | b)
        cond = 0.0
        for b, pb in Pb.items():
            sub = [v for (t, bb), v in joint.items() if bb == b]
            s = sum(sub) or 1.0
            cond += pb * _entropy([x / s for x in sub])
        rows[a] = max(0.0, Ht - cond)
    return rows


class OpeningAgent:
    def __init__(self, base, first_move):
        self.base = base
        self.first = first_move
        self.t = 0

    def act(self, own, opp):
        if self.t == 0:
            self.t += 1
            # 首招非法则回退到 base
            if self.first in get_valid_moves(own, mode=self.base.mode):
                return self.first
        self.t += 1
        return self.base.act(own, opp)

    def observe(self, *a):
        self.base.observe(*a)

    def reset(self):
        self.t = 0
        self.base.reset()


def br_first_move(mode=0):
    types = list(bots_by_name)
    opp = {}
    for qa in range(6):
        for qp in range(6):
            d = {}
            for t in types:
                for b, p in bots_by_name[t].policy(qp, qa, mode=mode, last_opp=None).items():
                    d[b] = d.get(b, 0.0) + p / len(types)
            opp[(qa, qp)] = d
    T = build_tables(mode)
    br = solve_best_response(mode, opp, gamma=0.95, max_sweeps=1500, tol=1e-6, tables=T)
    pol = br.policy.get((3, 3), {})
    return max(pol.items(), key=lambda kv: kv[1])[0] if pol else "防御"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--cap", type=int, default=50)
    args = ap.parse_args()

    table = info_gain_table(0)
    print("首招信息增益 I(type;response|a):")
    for a, ig in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"  {a:<8} {ig:.4f} nats")
    a_info = max(table.items(), key=lambda kv: kv[1])[0]
    a_br = br_first_move(0)
    print(f"\n信息增益首招 = {a_info}   信念BR首招 = {a_br}")

    opps = ["mirror", "anti_jump", "aggressive", "defensive"]
    print(f"\n{'对手':<12}{'info首招':>22}{'BR首招':>22}{'随机首招':>22}")
    for name in opps:
        row = []
        for first in (a_info, a_br, "跳"):
            base = OnlineBRAgent(PlaySettings(), mode=0, br_sweeps=150)
            ai = OpeningAgent(base, first)
            r = run_duel(ai, BotAgent(bots_by_name[name]), mode=0,
                         games=args.games, cap=args.cap)
            row.append(f"{r['wins']/r['games']:>10.0%}/{statistics.mean(r['rounds']):>4.1f}r")
        print(f"{name:<12}" + "".join(f"{c:>22}" for c in row))


if __name__ == "__main__":
    main()
