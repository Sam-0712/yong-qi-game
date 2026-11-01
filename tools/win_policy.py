"""对照实验：BR + step_cost  vs  W 面策略迭代（无折扣、无每回合惩罚）。

用法：
    python -m tools.win_policy                       # 手写信念 + GRU 编码器，各 80 局
    python -m tools.win_policy --games 150 --belief both
    python -m tools.win_policy --belief encoder --encoder-file data/artifacts/encoder_gru_h32.json --kind br

对手集合（定义见 courage/bots/opponents.py）：用户 2 手/4 手套路 / 理性转化者（集合内）
/ 转化者[外] / 分散压制 / 低气交替 / 龟缩 / 猛攻 / 防守 / 随机 / 镜像 / 循环 / 反跳。
评测默认确定性播种（AI 与对手每局各自固定随机流），cap=99 仅作兜底。

⚠ 本脚本只报胜率，样本量小的时候标准误很大（80 局约 ±4.5pp），
  做模型对比请用 tools/bench.py 的逐决策 regret。
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

from courage.agents import OnlineBRAgent, run_duel
from courage.bots.opponents import make_opponent
from courage.core.rules import get_valid_moves
from courage.learning.context_model import ContextOpponentModel
from courage.learning.hybrid_model import HybridOpponentModel
from courage.learning.seq_encoder import EncoderBelief, load_encoder
from courage.learning.type_model import OpponentTypeModel
from courage.policy.opening import load_style_book
from courage.policy.settings import PlaySettings
from courage.policy.winpolicy import make_win_agent

ENC = os.path.join(_ROOT, "data", "artifacts", "encoder_gru.json")
SLACK = 0.15
ITERS = 60

# (显示名, styles 里的风格名)。带 [外] 的是类型后验集合外的分布外考题。
OPPONENTS = [
    ("用户4手", "exploit4"),
    ("用户2手", "two"),
    ("转化者", "converter"),
    ("转化者[外]", "strong"),
    ("分散压制", "pressure"),
    ("低气交替", "alternate_low"),
    ("turtle", "turtle"),
    ("aggressive", "aggressive"),
    ("defensive", "defensive"),
    ("random", "random"),
    ("mirror", "mirror"),
    ("cycler", "cycler"),
    ("anti_jump", "anti_jump"),
]


def opponents():
    """评测对手表：脚本定义统一来自 courage/bots/opponents.py。"""
    return {label: make_opponent(name) for label, name in OPPONENTS}


def build(kind: str, belief: str, style: str = "balanced", robust: float = 0.3,
          enc_path: str = ENC, seed: int = 0, type_posterior: bool = False,
          type_blend: float = 1.0, type_decay: float = 1.0,
          type_eps: float = 0.02, context_path: str = ""):
    book = load_style_book(get_valid_moves(3), 0, style)
    bl = None
    if belief == "encoder":
        if not os.path.exists(enc_path):
            return None
        bl = EncoderBelief(load_encoder(enc_path), tau=2.0, floor=0.0, smooth=0.05)
    tm = (OpponentTypeModel(mode=0, decay=type_decay, eps=type_eps)
          if type_posterior else None)
    if context_path and os.path.exists(context_path):
        cm = ContextOpponentModel.load(context_path)
        tm = HybridOpponentModel(tm, cm) if tm is not None else cm
    kw = dict(settings=PlaySettings(1.0, style), mode=0, robust=robust,
              opening_book=book, belief_like=bl, type_model=tm,
              type_blend=type_blend, rng=random.Random(seed))
    if kind == "br":
        return OnlineBRAgent(**kw)
    return make_win_agent(OnlineBRAgent, win_iters=ITERS, win_slack=SLACK, **kw)


def run(agent, opp, games: int, cap: int = 99, seed: int = 0):
    res = run_duel(agent, opp, 0, games=games, cap=cap, seed=seed)
    n = res["games"]
    return (res["wins"] / n * 100, res["losses"] / n * 100,
            statistics.mean(res["rounds"]), res["timeouts"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=80)
    ap.add_argument("--belief", default="both", choices=["hand", "encoder", "both"])
    ap.add_argument("--style", default="balanced")
    ap.add_argument("--slack", type=float, default=0.15)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--encoder-file", dest="encoder_file", default=ENC)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--robust", type=float, default=0.3,
                    help="幽灵威胁权重（0=关闭）")
    ap.add_argument("--type-posterior", dest="type_posterior", action="store_true",
                    help="启用对手类型后验（4手/2手/转化者/镜像等）")
    ap.add_argument("--type-blend", dest="type_blend", type=float, default=1.0,
                    help="类型后验与序列信念的混合权重（1=纯类型）")
    ap.add_argument("--type-decay", dest="type_decay", type=float, default=1.0,
                    help="类型权重遗忘率（<1 更快遗忘）")
    ap.add_argument("--type-eps", dest="type_eps", type=float, default=0.02,
                    help="类型似然下限")
    ap.add_argument("--context-file", dest="context_file", default="",
                    help="用上下文对手模型（学习型）替代类型后验")
    ap.add_argument("--cap", type=int, default=99,
                    help="兜底回合上限（不作为产品结算上限）")
    ap.add_argument("--kind", default="both", choices=["br", "win", "both"])
    args = ap.parse_args()
    global SLACK, ITERS
    SLACK = args.slack
    ITERS = args.iters

    beliefs = ["hand", "encoder"] if args.belief == "both" else [args.belief]
    kinds = ("br", "win") if args.kind == "both" else (args.kind,)
    opps = opponents()

    for belief in beliefs:
        print(f"\n=== 信念来源: {belief}  风格: {args.style}  {args.games} 局/对手 ===")
        header = "".join(f"{k:>24}" for k in kinds)
        print(f"{'对手':<10}{header}")
        t0 = time.time()
        for name, opp in opps.items():
            row = f"{name:<10}"
            for kind in kinds:
                ag = build(kind, belief, args.style, robust=args.robust,
                           enc_path=args.encoder_file, seed=args.seed,
                           type_posterior=args.type_posterior,
                           type_blend=args.type_blend,
                           type_decay=args.type_decay,
                           type_eps=args.type_eps,
                           context_path=args.context_file)
                if ag is None:
                    row += f"{'（缺编码器）':>24}"
                    continue
                w, _l, rd, to = run(ag, opp, args.games, args.cap, args.seed)
                row += f"{w:6.0f}% /{rd:5.1f}R /{to:2d}T"
            print(row)
        print(f"  用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
