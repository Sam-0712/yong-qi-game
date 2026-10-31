"""在线信念-BR 对照：对战各脚本对手，并与固定均衡策略比较。

用法：
  python -m tools.online --all --games 150
  python -m tools.online --switch           # 演示「先攻后转守」的识别
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.agents import (BotAgent, OnlineBRAgent, ScriptedAgent,
                            TableAgent, run_duel)
from courage.bots.scripts import bots_by_name
from courage.core.rules import MOVES_ATTR, get_valid_moves, judge, next_qi
from courage.policy.settings import PlaySettings
from courage.storage.artifacts import load_policy_table


def _fmt(res) -> str:
    n = res["games"]
    r = res["rounds"]
    return (f"胜{res['wins']:>3} 负{res['losses']:>3} 超时{res['timeouts']:>3} | "
            f"胜率 {res['wins']/n:5.1%} | 回合 均值 {statistics.mean(r):4.1f} "
            f"中位 {statistics.median(r):3.0f}")


def make_switch_fn(mode, attack_rounds=3):
    def fn(own, opp, t):
        if t < attack_rounds:
            va = get_valid_moves(own, mode=mode)
            atk = [m for m in va if MOVES_ATTR[m][1] > 0]
            return max(atk, key=lambda m: MOVES_ATTR[m][1]) if atk else "吐气"
        return "防御"
    return fn


def run_all(args):
    eq_pol, _ = load_policy_table("policy_mode0.json")
    print(f"{'对手':<10}{'在线信念-BR':<44}{'固定均衡':<30}")
    for name in bots_by_name:
        opp1 = BotAgent(bots_by_name[name])
        online = OnlineBRAgent(PlaySettings(), args.mode, decay=args.decay,
                               alpha=args.alpha, smooth=args.smooth,
                               step_cost=args.step_cost, br_sweeps=args.br_sweeps)
        r_on = run_duel(online, opp1, args.mode, games=args.games, cap=args.cap)

        opp2 = BotAgent(bots_by_name[name])
        eq = TableAgent(eq_pol, PlaySettings(), args.mode) if eq_pol else None
        r_eq = run_duel(eq, opp2, args.mode, games=args.games, cap=args.cap) if eq else None
        eqs = _fmt(r_eq) if r_eq else "(无均衡表)"
        print(f"{name:<10}{_fmt(r_on):<44}{eqs:<30}")


def run_switch(args):
    mode = args.mode
    policy_eq, _ = load_policy_table("policy_mode0.json")

    for label, ai in [
        ("在线信念-BR", OnlineBRAgent(PlaySettings(), mode, decay=args.decay,
                                   alpha=args.alpha, smooth=args.smooth,
                                   step_cost=args.step_cost, br_sweeps=args.br_sweeps)),
        ("固定均衡", TableAgent(policy_eq, PlaySettings(), mode) if policy_eq else None),
    ]:
        if ai is None:
            continue
        opp = ScriptedAgent(make_switch_fn(mode, attack_rounds=3))
        print(f"\n=== {label} vs 先攻后守 (前3回合进攻→改防御) ===")
        print(f"{'回合':<4}{'AI气':<5}{'敌气':<5}{'AI出':<8}{'敌出':<8}"
              f"{'long_atk':<9}{'long_def':<9}{'switch':<7}{'> 信念解读'}")
        ai.reset(); opp.reset()
        qa = qp = 3
        for rnd in range(1, args.cap + 1):
            a = ai.act(qa, qp)
            b = opp.act(qp, qa)
            f = ai.belief.factors() if hasattr(ai, "belief") else None
            tag = ""
            if f is not None:
                tag = (f"L_atk={f['long_atk']:.2f} L_def={f['long_def']:.2f} "
                       f"sw={f['switch']:.2f}")
                extra = f"  {tag}  ->"
            else:
                extra = "  " + "-" * 20 + "  ->"
            fa = f"{f['long_atk']:<9.2f}{f['long_def']:<9.2f}{f['switch']:<7.2f}" if f else f"{'-':<9}{'-':<9}{'-':<7}"
            print(f"{rnd:<4}{qa:<5}{qp:<5}{a:<8}{b:<8}{fa}")
            ai.observe(qa, qp, a, b)
            opp.observe(qp, qa, b, a)
            if judge(a, b) != 0:
                print(f"  -> 第 {rnd} 回合 {'AI胜' if judge(a,b)==1 else 'AI负'}")
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 在线信念-BR")
    ap.add_argument("--all", action="store_true", help="对各脚本对手做对照")
    ap.add_argument("--switch", action="store_true", help="演示先攻后守")
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--decay", type=float, default=0.85)
    ap.add_argument("--alpha", type=float, default=0.20, help="Dirichlet 平滑伪计数")
    ap.add_argument("--smooth", type=float, default=0.05, help="对手分布与均匀混合比例")
    ap.add_argument("--step-cost", dest="step_cost", type=float, default=0.03,
                    help="每回合成本，惩罚无限平局")
    ap.add_argument("--br-sweeps", dest="br_sweeps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import random
    random.seed(args.seed)
    if args.switch:
        run_switch(args)
    if args.all or not args.switch:
        run_all(args)


if __name__ == "__main__":
    main()
