"""生成大规模混合对局语料（给「学习型对手模型」用）。

不是只跑随机局：纯随机只能覆盖合法性，学不到风格结构。这里按分布混合：
  - 均匀随机合法招（覆盖边界/残局）
  - 7 个脚本 bot
  - 固定打法生成器：防御、分散小攻击压制、低气交替
  - 固定套路：4 手、2 手、理性转化者
并支持随机起始气数（残局）。

用法：
  python -m tools.gen_corpus --games 1000 --out data/corpus/pilot.jsonl
  python -m tools.gen_corpus --games 20000 --out data/corpus/corpus_v1.jsonl

输出 JSONL：每局一条，含双方策略名、起始气数、完整序列、结果、回合数、seed。
消费端可从 p1 或 p2 视角取样本；训练/验证/测试按策略名切分，避免同分布泄漏。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.bots.opponents import OPPONENTS, make_opponent
from courage.core.rules import judge, next_qi


def make_agent(name: str, seed: int):
    """对手脚本统一来自 courage/bots/opponents.py。

    与 `learning/type_model.py`（类型后验假设的剧本）和 `tools/bench.py`
    （评测实际跑的剧本）共用同一份定义——此前三处各写一份且已不一致。
    """
    return make_opponent(name, seed)


POLICIES: Dict[str, float] = {
    "random": 0.15,
    "turtle": 0.07,
    "defensive": 0.10,
    "aggressive": 0.07,
    "cycler": 0.05,
    "anti_jump": 0.05,
    "mirror": 0.05,
    "pressure": 0.12,
    "alternate_low": 0.12,
    "exploit4": 0.07,
    "two": 0.08,
    "converter": 0.07,
}

_MISSING = [k for k in POLICIES if k not in OPPONENTS]
if _MISSING:
    raise RuntimeError(f"语料策略未在 courage/bots/opponents.py 里定义: {_MISSING}")


def choose_policy(rng: random.Random) -> str:
    r = rng.random() * sum(POLICIES.values())
    acc = 0.0
    last = "random"
    for name, w in POLICIES.items():
        last = name
        acc += w
        if r <= acc:
            return name
    return last


def play_one(game: int, args) -> dict:
    seed = (args.seed * 2_654_435_761 + game * 97 + 12_345) & 0x7FFFFFFF
    rng = random.Random(seed)
    n1, n2 = choose_policy(rng), choose_policy(rng)
    a1 = make_agent(n1, seed * 2 + 1)
    a2 = make_agent(n2, seed * 2 + 2)
    a1.reset(); a2.reset()

    if rng.random() < args.random_start_frac:
        qa, qp = rng.randint(0, 5), rng.randint(0, 5)
    else:
        qa = qp = 3

    own, opp, qis = [], [], []
    result, rounds = 0, args.steps
    for r in range(1, args.steps + 1):
        x = a1.act(qa, qp)
        y = a2.act(qp, qa)
        own.append(x); opp.append(y); qis.append([qa, qp])
        w = judge(x, y)
        a1.observe(qa, qp, x, y)
        a2.observe(qp, qa, y, x)
        if w != 0:
            result, rounds = w, r
            break
        qa, qp = next_qi(qa, x), next_qi(qp, y)
    return {"game": game, "seed": seed, "mode": args.mode,
            "p1": n1, "p2": n2, "start": [qis[0][0], qis[0][1]] if qis else [3, 3],
            "result": result, "rounds": rounds,
            "own_seq": own, "opp_seq": opp, "qi_seq": qis}


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 混合对局语料生成")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-start-frac", dest="random_start_frac",
                    type=float, default=0.3, help="随机起始气数（残局）比例")
    ap.add_argument("--out", default="data/corpus/corpus_v1.jsonl")
    args = ap.parse_args()

    out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stats = Counter()
    total_steps = 0
    with open(out, "w", encoding="utf-8") as f:
        for g in range(args.games):
            rec = play_one(g, args)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats[(rec["p1"], rec["p2"])] += 1
            total_steps += len(rec["own_seq"])
            if (g + 1) % 1000 == 0:
                print(f"  {g+1}/{args.games} games, {total_steps} steps")
    print(f"written {out}")
    print(f"games={args.games} steps={total_steps} avg_steps={total_steps/args.games:.1f}")
    print("top pairings:", stats.most_common(8))


if __name__ == "__main__":
    main()
