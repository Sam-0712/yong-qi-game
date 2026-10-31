"""实时胜率面 W(qa, qp)：二维递推（解析）+ 实测（蒙特卡洛）两种口径。

    python -m tools.winrate dp --opp converter          # 吸收链精确解
    python -m tools.winrate mc --opp defensive --games 300   # 实际对局分状态统计

为什么不只看我方气数：
- 每回合结算一次，胜负在结算时决定；因此"实时胜率"是状态 (qa,qp) 上的量，
  必须同时在"我方气数上升"和"对方气数下降"两个方向递推。
- 解析口径用吸收链：W(s) = Σπ(a|s)Σμ(b|s)[胜 1 / 负 0 / 否则 W(s')]；
  用阻尼 γ 处理"双方都不愿变气"的闭合类。
- 解析口径必须给策略加 ε 混合，否则策略是确定性 argmax，链条退化成 0/1。
  这会让对手带 10% 随机走子，绝对胜率偏高——所以另有 mc 口径做交叉验证。
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from typing import Dict

import numpy as np

from courage.core.rules import get_valid_moves, judge, next_qi, MOVES_ATTR
from courage.solvers.equilibrium import build_tables
from courage.solvers.best_response import solve_best_response
from courage.solvers.winrate import solve_win_surface
from courage.agents import OnlineBRAgent, BotAgent
from courage.bots.scripts import bots_by_name
from courage.learning.threat import converter_policy
from courage.policy.settings import PlaySettings


def _agent_policy(smooth: float) -> Dict:
    T = build_tables(0)
    a = OnlineBRAgent(settings=PlaySettings(1.0, "balanced"), robust=0.3)
    br = solve_best_response(0, a._opp_policy_from_belief(), gamma=0.95,
                             max_sweeps=1500, tol=1e-6, step_cost=0.03, tables=T)
    out = {}
    for st, d in br.policy.items():
        legal = get_valid_moves(st[0], mode=0)
        u = 1.0 / len(legal)
        out[st] = {m: (1 - smooth) * d.get(m, 0.0) + smooth * u for m in legal}
    return out


def _opp_policy(name: str, smooth: float) -> Dict:
    def raw(qa, qp):
        if name == "converter":
            return converter_policy(get_valid_moves(qp, mode=0),
                                    get_valid_moves(qa, mode=0))
        return bots_by_name[name].policy(qp, qa, mode=0)
    out = {}
    for qa in range(6):
        for qp in range(6):
            d = raw(qa, qp)
            legal = get_valid_moves(qp, mode=0)
            u = 1.0 / len(legal)
            out[(qa, qp)] = {m: (1 - smooth) * d.get(m, 0.0) + smooth * u
                             for m in legal}
    return out


def _print(W: np.ndarray, title: str) -> None:
    print(f"=== {title} ===")
    print("        qp=0   qp=1   qp=2   qp=3   qp=4   qp=5")
    for qa in range(6):
        cells = []
        for qp in range(6):
            v = W[qa, qp]
            cells.append("   — " if v != v else "%5.2f" % v)   # nan=闭合类，标 —
        print("qa=%d   " % qa + " ".join(cells))
    print()


def task_dp(args) -> None:
    pi = _agent_policy(args.smooth)
    mu = _opp_policy(args.opp, args.smooth)
    W, _ = solve_win_surface(pi, mu)
    _print(W, f"解析 W(qa,qp)  对手={args.opp}  ε={args.smooth}")


def task_mc(args) -> None:
    from courage.pipeline import PlayerConfig, CouragePlayer
    byl = defaultdict(lambda: [0, 0, 0])
    for s in range(args.games):
        p = CouragePlayer(PlayerConfig(belief_source="hand", style=args.style, seed=s))
        p.reset()
        opp = BotAgent(bots_by_name[args.opp], rng=random.Random(s))
        qa = qp = 3
        ds = []
        res = 0
        for _ in range(args.cap):
            ds.append(qa - qp)
            a = p.act(qa, qp)
            b = opp.act(qp, qa)
            w = judge(a, b)
            p.observe(qa, qp, a, b)
            if w:
                res = w
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        for d in ds:
            byl[d][0 if res == 1 else (1 if res == -1 else 2)] += 1
    print(f"=== 实测胜率 vs 气差 d=qa-qp  对手={args.opp}  {args.games} 局 ===")
    for d in range(-5, 6):
        w, l, dr = byl[d]
        tot = w + l
        if tot < 20:
            continue
        print(f"  d={d:+d}  胜率 {w/tot:.2f}   已决 {tot:5d}   和/超时 {dr:5d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dp")
    d.add_argument("--opp", default="converter")
    d.add_argument("--smooth", type=float, default=0.10)
    d.set_defaults(func=task_dp)

    m = sub.add_parser("mc")
    m.add_argument("--opp", default="defensive")
    m.add_argument("--games", type=int, default=300)
    m.add_argument("--cap", type=int, default=100)
    m.add_argument("--style", default="balanced")
    m.set_defaults(func=task_mc)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
