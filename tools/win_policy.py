"""对照实验：BR + step_cost  vs  W 面策略迭代（无折扣、无每回合惩罚）。

用法：
    python -m tools.win_policy                       # 手写信念 + GRU 编码器，各 80 局
    python -m tools.win_policy --games 150 --belief both

对手集合：用户 4 手套路 / 理性转化者 / 龟缩 / 猛攻 / 防守 / 随机 / 镜像。
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.agents import BotAgent, OnlineBRAgent, run_duel
from courage.bots.scripts import bots_by_name
from courage.core.rules import get_valid_moves, judge, MOVES_ATTR, next_qi
from courage.learning.seq_encoder import EncoderBelief, load_encoder
from courage.learning.threat import converter_policy
from courage.policy.opening import load_style_book
from courage.policy.settings import PlaySettings
from courage.policy.winpolicy import make_win_agent

ENC = os.path.join(_ROOT, "data", "artifacts", "encoder_gru.json")
SLACK = 0.15
ITERS = 60


class Scripted:
    """按 (自方气, 对方气, 回合) 出招的脚本对手。"""

    def __init__(self, fn):
        self.fn = fn
        self.t = 0

    def reset(self):
        self.t = 0

    def act(self, own, opp):
        v = self.fn(own, opp, self.t)
        self.t += 1
        legal = get_valid_moves(own, mode=0)
        return v if v in legal else legal[0]

    def observe(self, *a):
        pass


def _strong(qp, qa):
    vb = get_valid_moves(qp, mode=0)
    ours = get_valid_moves(qa, mode=0)
    kills = [b for b in vb if all(judge(b, a) == 1 for a in ours)]
    if kills:
        return min(kills, key=lambda m: MOVES_ATTR[m][0])
    safe = [b for b in vb if not any(judge(a, b) == 1 for a in ours)]
    if not safe:
        return max(vb, key=lambda m: MOVES_ATTR[m][1])
    if "吐气" in safe:
        return "吐气"
    return max(safe, key=lambda m: (MOVES_ATTR[m][2], MOVES_ATTR[m][1]))


EXPLOIT = ["防御", "吐气", "吐气", "咏春"]


def opponents():
    return {
        "用户套路": Scripted(lambda qp, qa, t: EXPLOIT[t] if t < 4
                            else ("咏春" if qp >= 5 else "吐气")),
        "转化者": Scripted(lambda qp, qa, t: _strong(qp, qa)),
        "turtle": BotAgent(bots_by_name["turtle"]),
        "aggressive": BotAgent(bots_by_name["aggressive"]),
        "defensive": BotAgent(bots_by_name["defensive"]),
        "random": BotAgent(bots_by_name["random"]),
        "mirror": BotAgent(bots_by_name["mirror"]),
    }


def build(kind: str, belief: str, style: str = "balanced", robust: float = 0.3):
    book = load_style_book(get_valid_moves(3), 0, style)
    bl = None
    if belief == "encoder":
        if not os.path.exists(ENC):
            return None
        bl = EncoderBelief(load_encoder(ENC), tau=2.0, floor=0.0, smooth=0.05)
    kw = dict(settings=PlaySettings(1.0, style), mode=0, robust=robust,
              opening_book=book, belief_like=bl, rng=random.Random(0))
    if kind == "br":
        return OnlineBRAgent(**kw)
    return make_win_agent(OnlineBRAgent, win_iters=ITERS, win_slack=SLACK, **kw)


def run(agent, opp, games: int, cap: int = 100):
    res = run_duel(agent, opp, 0, games=games, cap=cap)
    n = res["games"]
    return res["wins"] / n * 100, statistics.mean(res["rounds"]), res["timeouts"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--belief", default="both", choices=["hand", "encoder", "both"])
    ap.add_argument("--style", default="balanced")
    ap.add_argument("--slack", type=float, default=0.15)
    ap.add_argument("--iters", type=int, default=60)
    args = ap.parse_args()
    global SLACK, ITERS
    SLACK = args.slack
    ITERS = args.iters

    beliefs = ["hand", "encoder"] if args.belief == "both" else [args.belief]
    opps = opponents()

    for belief in beliefs:
        print(f"\n=== 信念来源: {belief}  风格: {args.style}  {args.games} 局/对手 ===")
        print(f"{'对手':<10}{'BR+step_cost':>22}{'W 策略迭代':>22}")
        t0 = time.time()
        for name, opp in opps.items():
            row = f"{name:<10}"
            for kind in ("br", "win"):
                ag = build(kind, belief, args.style)
                if ag is None:
                    row += f"{'（缺编码器）':>22}"
                    continue
                w, rd, to = run(ag, opp, args.games)
                row += f"{w:6.0f}% /{rd:5.1f}R /{to:2d}T"
            print(row)
        print(f"  用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
