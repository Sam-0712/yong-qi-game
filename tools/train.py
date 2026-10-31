"""训练工具。

当前阶段（工程化）：
  states    打印状态空间规模（验证抽象与压缩）
  opponent  用脚本 bot 合成样本训练小型对手模型（C 的最小闭环）

B 阶段的博弈求解器将在后续接入本文件。
用法：
  python -m tools.train states
  python -m tools.train opponent --bot aggressive --save --verbose
"""

from __future__ import annotations

import argparse
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.core.rules import MOVES, MOVE_TO_IDX
from courage.core.state import (QI_MAX, MODE_STANDARD, MODE_NO_QI,
                                iter_states)
from courage.bots.scripts import bots_by_name
from courage.storage.artifacts import artifact_path


def task_states(args) -> None:
    for mode, label in ((MODE_STANDARD, "standard"), (MODE_NO_QI, "no_qi")):
        n_all = len(list(iter_states(mode)))
        n_sym = len(list(iter_states(mode, symmetric_only=True)))
        n_fin = len(list(iter_states(mode, horizon=100)))
        print(f"[{label}] 平稳状态={n_all}  对称={n_sym}  有限期(H=100)={n_fin}")


def gen_samples(bot, n: int, mode: int, seed: int):
    """从脚本 bot 合成 (特征, 标签) 样本。"""
    import numpy as np
    from courage.learning.opponent import features

    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        qa = rng.randint(0, QI_MAX)
        qp = rng.randint(0, QI_MAX)
        last = rng.choice([None] + list(MOVES))
        mv = bot.act(qa, qp, mode=mode, last_opp=last, rng=rng)
        X.append(features(qa, qp, mode, last))
        y.append(MOVE_TO_IDX[mv] - 1)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def task_opponent(args) -> None:
    import numpy as np
    from courage.learning.opponent import OpponentModel

    bot = bots_by_name[args.bot]
    X, y = gen_samples(bot, args.samples, args.mode, args.seed)
    model = OpponentModel(hidden=args.hidden, seed=args.seed, lr=args.lr)
    print(f"对手模型: bot={args.bot}  参数={model.n_params}  "
          f"特征维={X.shape[1]}  样本={len(y)}")

    g = np.random.default_rng(args.seed)
    idx = np.arange(len(y))
    for ep in range(args.epochs):
        g.shuffle(idx)
        tot, nb = 0.0, 0
        for i in range(0, len(idx), args.batch):
            b = idx[i:i + args.batch]
            tot += model.train_batch(X[b], y[b])
            nb += 1
        if args.verbose and (ep + 1) % max(1, args.epochs // 5) == 0:
            acc = float((model.proba_batch(X).argmax(1) == y).mean())
            print(f"  epoch {ep + 1:>3}  loss={tot / nb:.4f}  acc={acc:.3f}")

    acc = float((model.proba_batch(X).argmax(1) == y).mean())
    print(f"最终训练精度: {acc:.3f}")
    if args.save:
        out = artifact_path(f"opponent_{args.bot}.json")
        model.save(out)
        print(f"已保存: {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 训练工具")
    sub = ap.add_subparsers(dest="task", required=True)

    sp = sub.add_parser("states", help="打印状态空间规模")
    sp.set_defaults(func=task_states)

    op = sub.add_parser("opponent", help="训练小型对手模型")
    op.add_argument("--bot", default="aggressive", choices=list(bots_by_name))
    op.add_argument("--mode", type=int, default=0)
    op.add_argument("--samples", type=int, default=4000)
    op.add_argument("--epochs", type=int, default=30)
    op.add_argument("--batch", type=int, default=64)
    op.add_argument("--hidden", type=int, default=16)
    op.add_argument("--lr", type=float, default=0.1)
    op.add_argument("--seed", type=int, default=0)
    op.add_argument("--save", action="store_true")
    op.add_argument("--verbose", action="store_true")
    op.set_defaults(func=task_opponent)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
