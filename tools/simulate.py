"""对局模拟器：观测胜率与对局长度（支持难度 / 风格）。

用法：
  python -m tools.simulate --ai equilibrium --opp turtle --games 300
  python -m tools.simulate --opp self --games 100          # 自对弈（看长度）
  python -m tools.simulate --style aggressive --difficulty 0.7 --opp aggressive
  python -m tools.simulate --policy-file policy_mode0.json --opp random
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.core.rules import get_valid_moves
from courage.policy.settings import PlaySettings, apply_settings
from courage.bots.scripts import bots_by_name
from courage.sim import play_game


def build_policy(args):
    if args.policy_file:
        from courage.storage.artifacts import load_policy_table
        pol, _ = load_policy_table(args.policy_file)
        if pol:
            return pol
        print(f"[!] 未找到 {args.policy_file}，改为现场求解")
    from courage.solvers.equilibrium import solve_equilibrium
    res = solve_equilibrium(mode=args.mode, gamma=args.gamma, iters=args.inner,
                            max_sweeps=args.sweeps, tol=args.tol)
    return res.policy


def make_ai(policy, settings, mode):
    def fn(qa, qp):
        legal = get_valid_moves(qa, mode=mode)
        base = policy.get((qa, qp)) or {m: 1.0 / len(legal) for m in legal}
        return apply_settings(base, legal, settings)
    return fn


def make_bot(bot, mode):
    # bot.policy(own_qi, opp_qi)：作为对手时，其自身气数是 qp
    return lambda qa, qp: bot.policy(qp, qa, mode=mode)


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 对局模拟")
    ap.add_argument("--ai", default="equilibrium", help="AI 策略来源（占位）")
    ap.add_argument("--policy-file", default="", help="从产物加载策略表，省去重算")
    ap.add_argument("--opp", default="turtle",
                    choices=list(bots_by_name) + ["self", "random"])
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--style", default="balanced")
    ap.add_argument("--difficulty", type=float, default=1.0)
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--inner", type=int, default=800)
    ap.add_argument("--sweeps", type=int, default=400)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    policy = build_policy(args)
    ai_fn = make_ai(policy, PlaySettings(args.difficulty, args.style), args.mode)

    if args.opp == "self":
        def opp_fn(qa, qp):
            legal = get_valid_moves(qp, mode=args.mode)
            base = policy.get((qp, qa)) or {m: 1.0 / len(legal) for m in legal}
            return apply_settings(base, legal, PlaySettings(1.0, "balanced"))
    else:
        opp_fn = make_bot(bots_by_name[args.opp], args.mode)

    rng = random.Random(args.seed)
    W = L = T = 0
    rounds = []
    for _ in range(args.games):
        r, rnd = play_game(ai_fn, opp_fn, mode=args.mode, cap=args.cap, rng=rng)
        rounds.append(rnd)
        if r == 1:
            W += 1
        elif r == -1:
            L += 1
        else:
            T += 1

    n = args.games
    print(f"AI(style={args.style}, 难度={args.difficulty}) vs {args.opp} | "
          f"games={n} mode={args.mode} cap={args.cap}")
    print(f"  胜 {W}  负 {L}  超时/未决 {T}  |  胜率={W/n:.1%}  不输率={(W+T)/n:.1%}")
    print(f"  回合: 均值={statistics.mean(rounds):.1f}  "
          f"中位={statistics.median(rounds):.0f}  "
          f"最短={min(rounds)} 最长={max(rounds)}")
    dec = [x for x in rounds if x < args.cap]
    if dec:
        print(f"  分出胜负局回合均值={statistics.mean(dec):.1f} (n={len(dec)})")


if __name__ == "__main__":
    main()
