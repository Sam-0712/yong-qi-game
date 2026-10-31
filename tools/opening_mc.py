"""开局簿构建与检验（蒙特卡洛调优首招分布）。

用法：
    python -m tools.opening_mc build --games 30 --save          # 训练并保存
    python -m tools.opening_mc show                             # 查看分布
    python -m tools.opening_mc check --lines 1296               # 脚本套路穷举检验

思路（沿用 3.0）：对每个首招跑大轮次模拟，用**胜率**决定其占比，再做平滑/温度
变成概率分布 → 采样。好处：首招随机（不可被脚本化）、各招都可能、好招占多。
"""

from __future__ import annotations

import argparse
import itertools
import random
import statistics
import sys
from typing import Callable, Dict, List, Optional

from courage.agents import BotAgent, OnlineBRAgent, ScriptedAgent, run_duel
from courage.bots.scripts import bots_by_name, sample_policy, MOVES_ATTR
from courage.core.rules import get_valid_moves, judge, next_qi
from courage.policy.opening import (OpeningBook, build_style_book,
                                   scores_to_probs, DEFAULT_BOOK_NAME)
from courage.storage.artifacts import save_json, load_json

QI0 = 3


# ── 脚本对手：覆盖「人类会用的套路」──
def _max_atk(own: int) -> str:
    legal = get_valid_moves(own)
    atk = [m for m in legal if MOVES_ATTR[m][1] > 0]
    return max(atk, key=lambda m: MOVES_ATTR[m][1]) if atk else "吐气"


def f_patient(own, opp, t):          # 蓄气型：攒到 5 气再一击
    if own >= 5:
        return "咏春"
    if own >= 4 and opp <= 3:
        return "螺旋环"
    return "吐气"


def f_guard_then_build(own, opp, t):  # 先守一手再蓄气（本次被发现的套路）
    if t == 0:
        return "防御"
    return f_patient(own, opp, t)


def f_rusher(own, opp, t):
    return _max_atk(own)


def f_punish_low(own, opp, t):        # 见对手低气就压上
    if opp <= 1 and own >= 3:
        return _max_atk(own)
    return f_patient(own, opp, t)


SCRIPTED = {
    "patient": f_patient,
    "guard_build": f_guard_then_build,
    "rusher": f_rusher,
    "punish_low": f_punish_low,
}


def opponent_pool(mode: int) -> Dict[str, object]:
    pool: Dict[str, object] = {}
    for name in ["turtle", "aggressive", "defensive", "random", "cycler", "mirror", "anti_jump"]:
        pool[name] = BotAgent(bots_by_name[name], mode=mode)
    for name, fn in SCRIPTED.items():
        pool[name] = ScriptedAgent(fn)
    return pool


def make_ai(mode: int, qi_floor: int = 0, floor_win: float = 1.0,
            style: str = "balanced", sweeps: int = 600) -> OnlineBRAgent:
    """用于 MC 估值的 AI：续战策略带上目标风格，这样「这一手对这个风格值多少」才是真值。"""
    from courage.policy.settings import PlaySettings
    return OnlineBRAgent(mode=mode, qi_floor=qi_floor, floor_win=floor_win,
                         settings=PlaySettings(1.0, style), br_sweeps=sweeps,
                         rng=random.Random(0))


def play_forced(ai, opp, first: str, cap: int = 100):
    """AI 首招强制为 first，其余走正式策略。返回 (result, rounds)。"""
    ai.reset()
    opp.reset()
    qa = qp = QI0
    forced = first
    for r in range(1, cap + 1):
        a = forced if forced is not None else ai.act(qa, qp)
        forced = None
        b = opp.act(qp, qa)
        w = judge(a, b)
        ai.observe(qa, qp, a, b)
        opp.observe(qp, qa, b, a)
        if w != 0:
            return w, r
        qa, qp = next_qi(qa, a), next_qi(qp, b)
    return 0, cap


STYLES = ["balanced", "aggressive", "defensive", "gambler", "thinker"]


def build(args):
    """style=all 时返回 {style: OpeningBook}，否则返回单个 OpeningBook。"""
    mode = args.mode
    styles = STYLES if args.style == "all" else [args.style]
    books = {}
    for style in styles:
        books[style] = build_one(args, mode, style)
        print()
    if len(books) == 1:
        return next(iter(books.values()))
    print("=== 五风格开局簿对比（首招占比 %）===")
    print("首招   " + "".join(f"{s:>12}" for s in styles))
    for m in get_valid_moves(QI0, mode=mode):
        row = f"{m:<6} " + "".join(f"{books[s].probs.get(m,0)*100:11.1f}%" for s in styles)
        print(row)
    return books


def build_one(args, mode: int, style: str) -> OpeningBook:
    pool = opponent_pool(mode)
    moves = get_valid_moves(QI0, mode=mode)
    print(f"=== 风格: {style} ===")
    scores: Dict[str, float] = {}
    detail: Dict[str, float] = {}
    print(f"首招候选: {moves}")
    print(f"{'首招':<6}{'得分率':>8}{'胜':>6}{'平':>6}{'负':>6}   分对手")
    for m in moves:
        w = d = l = 0
        per: List[str] = []
        for oname, opp in pool.items():
            ow = od = ol = 0
            ai = make_ai(mode, args.qi_floor, args.floor_win, style, args.sweeps)
            for _ in range(args.games):
                res, _r = play_forced(ai, opp, m, args.cap)
                if res == 1:
                    ow += 1
                elif res == -1:
                    ol += 1
                else:
                    od += 1
            n = max(1, ow + od + ol)
            sw = (ow + 0.5 * od) / n
            per.append(f"{oname}:{sw:.2f}")
            w, d, l = w + ow, d + od, l + ol
        n = max(1, w + d + l)
        scores[m] = (w + 0.5 * d) / n
        detail[m] = scores[m]
        print(f"{m:<6}{scores[m]:>8.3f}{w:>6}{d:>6}{l:>6}   {' '.join(per)}")

    # 得分（大样本对局）× 风格先验 → 该风格的首招分布
    book = build_style_book(scores, style, args.temp, args.eps)
    book.meta.update({
        "mode": mode, "style": style, "games_per_opp": args.games, "cap": args.cap,
        "qi_floor": args.qi_floor, "floor_win": args.floor_win,
    })
    print("\n首招分布:")
    for m, p in sorted(book.probs.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<5}{p*100:5.1f}%   (bar {'█' * int(round(p * 40))})")
    return book


# ── 套路穷举检验：人类能否用 ≤N 招的固定脚本稳定吃死 AI ──
def check(args) -> None:
    mode = args.mode
    book = None
    if args.book:
        blob = load_json(args.book)
        book = OpeningBook.from_dict(blob) if blob else None

    lines = list(itertools.product(
        ["吐气", "馒头", "双馒", "雷切", "防御", "跳"], repeat=args.depth))
    wins = 0
    examples: List[str] = []
    for seq in lines:
        ai = make_ai(mode, args.qi_floor, args.floor_win)
        if book is not None:
            ai.opening_book = book
        ai.reset()
        qa = qp = QI0
        human = list(seq)
        played = []
        for r in range(1, args.cap + 1):
            a = ai.act(qa, qp)
            if human:
                b = human.pop(0)
                if b not in get_valid_moves(qp, mode=mode):
                    b = "吐气"
            else:
                b = "咏春" if qp >= 5 else "吐气"
            played.append(f"{a}/{b}")
            w = judge(a, b)
            ai.observe(qa, qp, a, b)
            if w != 0:
                if w == -1:
                    wins += 1
                    if len(examples) < 3:
                        examples.append(" ".join(played))
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
    tot = len(lines)
    print(f"固定脚本穷举：深度 {args.depth} 共 {tot} 条，人类可胜 {wins} 条 "
          f"({wins / tot * 100:.1f}%)")
    for e in examples:
        print("   例:", e)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--mode", type=int, default=0)
        p.add_argument("--qi-floor", dest="qi_floor", type=int, default=1)
        p.add_argument("--floor-win", dest="floor_win", type=float, default=1.0)
        p.add_argument("--cap", type=int, default=100)

    b = sub.add_parser("build")
    b.add_argument("--games", type=int, default=30, help="每个对手每个首招的局数")
    b.add_argument("--temp", type=float, default=0.12)
    b.add_argument("--eps", type=float, default=0.04)
    b.add_argument("--style", default="balanced",
                   choices=STYLES + ["all"], help="风格（all=五风格各出一本）")
    b.add_argument("--sweeps", type=int, default=250, help="MC 续战时的 BR 迭代数")
    b.add_argument("--save", action="store_true")
    b.add_argument("--out", default="")
    common(b)
    b.set_defaults(func=lambda a: _do_build(a))

    s = sub.add_parser("show")
    s.add_argument("--book", default=DEFAULT_BOOK_NAME)
    s.set_defaults(func=lambda a: _do_show(a))

    c = sub.add_parser("check")
    c.add_argument("--depth", type=int, default=3)
    c.add_argument("--book", default="")
    common(c)
    c.set_defaults(func=check)

    args = ap.parse_args()
    args.func(args)


def _do_build(args) -> None:
    result = build(args)
    if not args.save:
        return
    if isinstance(result, dict):
        for st, b in result.items():
            name = f"opening_mode{args.mode}_{st}.json"
            save_json(name, b.to_dict())
            print(f"已保存: data/artifacts/{name}")
    else:
        name = args.out or f"opening_mode{args.mode}_{args.style}.json"
        save_json(name, result.to_dict())
        print(f"\n已保存: data/artifacts/{name}")


def _do_show(args) -> None:
    blob = load_json(args.book)
    if not blob:
        print(f"未找到 {args.book}，先运行 build --save")
        return
    book = OpeningBook.from_dict(blob)
    print("meta:", {k: v for k, v in book.meta.items() if k != "scores"})
    for m, p in sorted(book.probs.items(), key=lambda kv: -kv[1]):
        print(f"  {m:<5}{p*100:5.1f}%")


if __name__ == "__main__":
    main()
