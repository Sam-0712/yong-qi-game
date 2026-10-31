"""评估工具：对手模型模仿精度。

用法：
  python -m tools.eval opponent --bot aggressive --samples 2000
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.core.rules import MOVES
from courage.bots.scripts import bots_by_name
from courage.storage.artifacts import artifact_path


def task_opponent(args) -> None:
    import numpy as np
    from courage.learning.opponent import OpponentModel
    from tools.train import gen_samples

    path = artifact_path(f"opponent_{args.bot}.json")
    if not os.path.exists(path):
        print(f"未找到模型: {path}")
        print(f"先运行: python -m tools.train opponent --bot {args.bot} --save")
        return

    model = OpponentModel.load(path)
    X, y = gen_samples(bots_by_name[args.bot], args.samples, args.mode, args.seed)

    # 合法动作掩码后的预测（游戏内真实可用）
    pred = np.empty(len(y), dtype=np.int64)
    for i in range(len(y)):
        qa = int(round(float(X[i, 0]) * 5))
        qp = int(round(float(X[i, 1]) * 5))
        mode = int(round(float(X[i, 2])))
        pred[i] = int(np.argmax(model.proba_masked(qa, qp, mode)))

    acc = float((pred == y).mean())
    print(f"bot={args.bot}  样本={len(y)}  掩码模仿精度={acc:.3f}")
    print("预测动作分布:", dict(collections.Counter(MOVES[i] for i in pred)))
    print("真实动作分布:", dict(collections.Counter(MOVES[i] for i in y)))


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 评估工具")
    sub = ap.add_subparsers(dest="task", required=True)

    op = sub.add_parser("opponent", help="评估对手模型模仿精度")
    op.add_argument("--bot", default="aggressive", choices=list(bots_by_name))
    op.add_argument("--mode", type=int, default=0)
    op.add_argument("--samples", type=int, default=2000)
    op.add_argument("--seed", type=int, default=12345)
    op.set_defaults(func=task_opponent)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
