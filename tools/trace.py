"""逐回合展开决策过程：看 AI 到底「信了谁」，以及每一次决策亏在哪。

`tools/bench.py` 给的是聚合指标（regret/局、胜率）。但当一个数字变好或变坏时，
我们需要知道**是哪一步、哪条通道、哪个 Q 值**造成的——否则调参就是盲调。

本工具跑一局（或几局）并把每个回合拆开打印：

    对手 / 模型 / 起始气数 / 种子
    ───────────────────────────────
    R3  (1,1)  我上吐气 对上馒头  t=2
      通道权重   type 0.21  context 0.35  uniform 0.44
      类型后验   前3: converter .31 defensive .22 turtle .18
      通道预测
        type    : 防御 .50 咏春 .50
        context : 馒头 .61 吐气 .22 防御 .17
        uniform : 吐气 .33 馒头 .33 防御 .33
      混合      : 馒头 .44 防御 .30 吐气 .26
      Q         : 馒头 +0.930  防御 +0.902  吐气 -0.850
      BR        : 馒头 1.00
      实选      : 馒头    对手 吐气    → 我胜
      本步 regret 0.0000

用法
----
    python -m tools.trace --style strong --model mix --start 1,1
    python -m tools.trace --style random --model mix --game 3 --wide
    python -m tools.trace --style defensive --model hybrid     # 看硬切换的门控决定
    python -m tools.trace --style strong --model type --no-regret --max-rounds 6
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.bots.opponents import OPPONENTS, OpponentAgent
from courage.bots.style_profiles import StyleOpponent, sample_zoo
from courage.core.rules import (IDX_TO_MOVE, MOVE_TO_IDX, get_valid_moves, judge,
                                next_qi)
from courage.pipeline import build_player
from courage.solvers.aug_mdp import AugSolution, oracle_for
from tools.bench import model_config


# ── 排版 ──
def _w(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, n: int) -> str:
    return str(s) + " " * max(0, n - _w(str(s)))


def _top(dist: Dict[str, float], k: Optional[int] = None) -> str:
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    if k is not None:
        items = items[:k]
    return "  ".join(f"{m} {v:.2f}" for m, v in items if v > 1e-6)


def _seed_agent(agent, seed: int) -> None:
    target = getattr(agent, "agent", agent)
    if hasattr(target, "rng"):
        target.rng = random.Random(seed)


# ── 单局 ──
def run_game(ai, opp: StyleAgent, oracle: Optional[AugSolution], cap: int,
             start: Tuple[int, int], gs: int, verbose: bool, wide: bool,
             max_rounds: int) -> dict:
    inner = getattr(ai, "agent", ai)
    _seed_agent(ai, gs * 2 + 1)
    _seed_agent(opp, gs * 2 + 2)
    ai.reset()
    opp.reset()

    qa, qp = start
    al = bl = t = 0
    t_max = oracle.tables.t_max if oracle is not None else 0
    reg_sum, reg_n = 0.0, 0
    result, rounds = 0, cap

    if verbose:
        print(f"对手 {opp.spec.name} ({getattr(opp.spec, 'label', '')})  "
              f"种子 {gs}  起始 {qa},{qp}")
        print("  " + "─" * 78)

    for rnd in range(1, cap + 1):
        a = ai.act(qa, qp)
        info = getattr(inner, "last_info", None) or {}
        b = opp.act(qp, qa)

        # ── 真实价值（精确 oracle）──
        gap = None
        best_move = None
        qrow: Dict[str, float] = {}
        if oracle is not None:
            si = oracle.tables.index(qa, qp, al, bl, t)
            q = oracle.q_row_at(si)
            va = oracle.tables.movesA[si]
            qrow = {m: float(q[i]) for i, m in enumerate(va)}
            best = max(qrow.values())
            best_move = max(qrow, key=lambda m: qrow[m])
            dist = info.get("dist") or {a: 1.0}
            got = sum(p * qrow.get(m, best) for m, p in dist.items())
            gap = best - got
            reg_sum += gap
            reg_n += 1

        if verbose and rnd <= max_rounds:
            _print_round(rnd, qa, qp, al, bl, t, inner, info, qrow,
                         best_move, a, b, gap, wide)

        w = judge(a, b)
        ai.observe(qa, qp, a, b)
        opp.observe(qp, qa, b, a)
        if w != 0:
            result, rounds = w, rnd
            if verbose:
                tag = "我胜" if w == 1 else "我负"
                print(f"  → {tag}（第 {rnd} 回合）")
            break
        qa, qp = next_qi(qa, a), next_qi(qp, b)
        al, bl = MOVE_TO_IDX[a], MOVE_TO_IDX[b]
        t = min(t + 1, t_max)
    else:
        if verbose:
            print(f"  → 平局/超时（{cap} 回合上限）")

    return {"result": result, "rounds": rounds, "regret": reg_sum,
            "decisions": reg_n}


def _print_round(rnd: int, qa: int, qp: int, al: int, bl: int, t: int,
                 inner, info: dict, qrow: Dict[str, float],
                 best_move: Optional[str], a: str, b: str,
                 gap: Optional[float], wide: bool) -> None:
    def mv(i: int) -> str:
        return IDX_TO_MOVE.get(i) or "-"

    print(f"R{rnd:<2} ({qa},{qp})  我上{mv(al)} 对上{mv(bl)}  t={t}"
          + ("   [开局簿]" if info.get("opening") else ""))
    if info.get("loop_break"):
        print("      循环检测介入：强制偏离周期")

    model = getattr(inner, "type_model", None)
    legal = get_valid_moves(qp, mode=0)

    # 混合通道：权重 + 各通道预测
    if model is not None and hasattr(model, "breakdown"):
        weights, dists = model.breakdown(qa, qp, legal)
        line = "  ".join(f"{c} {w:.2f}" for c, w in weights.items())
        print(f"    通道权重   {line}")
        print("    通道预测（对手下一步）")
        for c, d in dists.items():
            print(f"      {_pad(c, 8)}: {_top(d, None if wide else 3)}")
        print(f"    对手混合  : {_top(model.opp_weights(qa, qp, legal),
                                  None if wide else 4)}")
        if getattr(model, "steps", 0):
            print(f"    累计 logloss  {model.summary()}")
    elif model is not None and hasattr(model, "_use_context"):
        # 硬切换（HybridOpponentModel）
        pick = "上下文模型" if model._use_context() else "类型后验"
        w = getattr(model.type_model, "w", None)
        top3 = ""
        if isinstance(w, dict) and w:
            top3 = "  ".join(f"{k} {v:.2f}" for k, v in
                             sorted(w.items(), key=lambda kv: -kv[1])[:3])
        print(f"    门控       → {pick}    类型后验前3: {top3}")
        print(f"    对手模型   : {_top(info.get('opp_policy', {}).get((qa, qp), {}),
                                  None if wide else 4)}")

    if qrow:
        qs = "  ".join(f"{m} {v:+.3f}" for m, v in
                       sorted(qrow.items(), key=lambda kv: -kv[1]))
        print(f"    Q(真值)    : {qs}")

    # 通道层面的 Q 与分歧：定量追踪「通道之间到底吵了多少」
    qm = info.get("q_mean") or {}
    if qm:
        top = sorted(qm.items(), key=lambda kv: -kv[1])
        top = top if wide else top[:3]
        print("    Q(通道均值): " + "  ".join(f"{m} {v:+.3f}" for m, v in top))
        sp = info.get("spread") or {}
        if sp:
            sps = sorted(sp.items(), key=lambda kv: -kv[1])
            sps = sps if wide else sps[:3]
            print("    Q 分歧     : " + "  ".join(f"{m} {v:.2f}" for m, v in sps))
        kappa = float(info.get("kappa") or 0.0)
        qr = info.get("q_risk") or {}
        if kappa > 0 and qr:
            top = sorted(qr.items(), key=lambda kv: -kv[1])
            top = top if wide else top[:3]
            print(f"    Q(κ={kappa:g})   : "
                  + "  ".join(f"{m} {v:+.3f}" for m, v in top)
                  + ("   ← 改判" if info.get("flip") else ""))
    pol = (info.get("br").policy.get((qa, qp)) if info.get("br") else None) or {}
    if pol:
        print(f"    BR         : {_top(pol, None if wide else 3)}"
              + (f"   argmax={best_move}" if best_move else ""))
    mine = info.get("dist") or {}
    if mine:
        print(f"    我方分布   : {_top(mine, None if wide else 4)}"
              + ("   （含难度/风格/噪声）" if len(mine) > 1 else ""))
    print(f"    实选       : {a}    对手 {b}" +
          ("" if gap is None else f"    本步 regret {gap:.4f}"))


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 单局决策过程追踪")
    ap.add_argument("--style", default="strong", help="对手打法（courage/bots/opponents.py）")
    ap.add_argument("--model", default="mix",
                    help="hand/type/context/hybrid/mix/mixnohand/uniform")
    ap.add_argument("--belief", default="hand", choices=["hand", "encoder"])
    ap.add_argument("--opening", default="off", choices=["safe", "off"])
    ap.add_argument("--start", default="1,1")
    ap.add_argument("--game", type=int, default=0, help="要展开第几局（0 起）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cap", type=int, default=99)
    ap.add_argument("--max-rounds", dest="max_rounds", type=int, default=99,
                    help="最多展开多少回合明细")
    ap.add_argument("--wide", action="store_true", help="不截断，打印全部招式")
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="通道分歧定价 κ（0=风险中性）")
    ap.add_argument("--zoo-index", dest="zoo_index", type=int, default=-1,
                    help=">=0 时改为展开第 N 个采样**风格**（与 bench --zoo 同一批）")
    ap.add_argument("--zoo-n", dest="zoo_n", type=int, default=12)
    ap.add_argument("--zoo-seed", dest="zoo_seed", type=int, default=1)
    ap.add_argument("--zoo-alpha", dest="zoo_alpha", type=float, default=1.0)
    ap.add_argument("--zoo-sharp", dest="zoo_sharp", type=float, default=2.0)
    ap.add_argument("--no-regret", dest="no_regret", action="store_true",
                    help="不算精确 oracle（更快，但没有 Q/regret）")
    args = ap.parse_args()

    if args.zoo_index < 0 and args.style not in OPPONENTS:
        raise SystemExit(f"未知对手 {args.style}；可选: {sorted(OPPONENTS)}")
    start = tuple(int(x) for x in args.start.split(","))
    if len(start) != 2:
        raise SystemExit("--start 需要两个整数，如 1,1")

    cfg = model_config(args.model, args.belief, args.opening, kappa=args.kappa)
    ai = build_player(cfg)
    if args.zoo_index >= 0:
        profs = sample_zoo(args.zoo_n, seed=args.zoo_seed, alpha=args.zoo_alpha,
                           sharpness=args.zoo_sharp)
        idx = min(max(args.zoo_index, 0), len(profs) - 1)
        spec = StyleOpponent(profs[idx], name=f"zoo{idx}")
    else:
        spec = OPPONENTS[args.style]
    opp = OpponentAgent(spec, random.Random(0))
    oracle = None if args.no_regret else oracle_for(spec)

    print(f"模型 {args.model}  门控 {cfg.gate}  "
          f"信念 {cfg.belief_source}  开局 {cfg.opening}")
    game_rng = random.Random(args.seed)
    for g in range(args.game + 1):
        gs = game_rng.randrange(1 << 30)
        verbose = (g == args.game)
        if not verbose:
            run_game(ai, opp, oracle, args.cap, start, gs, False, args.wide,
                     args.max_rounds)
            continue
        out = run_game(ai, opp, oracle, args.cap, start, gs, True, args.wide,
                       args.max_rounds)

    if oracle is not None:
        try:
            v = oracle.value(*_start_state(oracle, start))
            print(f"\nV*(起始 {start[0]},{start[1]}) = {v:+.4f}   "
                  f"（该对手下可达价值上限）")
        except KeyError:
            pass
    print(f"结果 {out['result']:+d}  回合 {out['rounds']}  "
          f"regret {out['regret']:.4f}  （{out['decisions']} 次决策）")
    model = getattr(getattr(ai, "agent", ai), "type_model", None)
    if model is not None and hasattr(model, "summary"):
        print(f"通道总结  {model.summary()}")


def _start_state(oracle: AugSolution, start: Tuple[int, int]):
    n = oracle.tables.needs
    return (start[0], start[1], 0, 0, 0 if "t" in n else 0)


if __name__ == "__main__":
    main()
