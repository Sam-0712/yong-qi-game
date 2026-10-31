"""因子蒸馏实验：系统比较 架构 × 窗口 × 数据量 × 输入消融。

关注点（序列 → 因子这一核心）：
  1. 学得动吗？训练/测试 RMSE（过拟合）
  2. 哪些因子最难？
  3. 架构对比：CNN(窗口) vs GRU
  4. 消融：输入「双方」vs「只看对手」

用法：python -m tools.distill_exp
"""

from __future__ import annotations

import argparse
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from courage.bots.scripts import ALL_BOTS, sample_policy
from courage.core.rules import get_valid_moves, judge, next_qi, MOVES
from courage.learning.seq_encoder import (CNNEncoder, GRUEncoder, FACTOR_KEYS,
                                          rmse_per_factor, sequence_to_xy, D_IN)

TK = {"alpha": 0.20, "smooth": 0.05}
NOWN = len(MOVES)


def gen(n, steps, mode, seed):
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        b1, b2 = rng.choice(ALL_BOTS), rng.choice(ALL_BOTS)
        qa = qp = 3
        own, opp = [], []
        for _t in range(steps):
            a = sample_policy(b1.policy(qa, qp, mode=mode, last_opp=opp[-1] if opp else None), rng)
            b = sample_policy(b2.policy(qp, qa, mode=mode, last_opp=own[-1] if own else None), rng)
            own.append(a); opp.append(b)
            if judge(a, b) != 0:
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        if len(own) >= 2:
            seqs.append((own, opp))
    return seqs


def build(seqs, opp_only):
    data = []
    for o, p in seqs:
        X, Y = sequence_to_xy(o, p, TK)
        if opp_only:
            X = X.copy(); X[:, :NOWN] = 0.0
        data.append((X, Y))
    return data


def eval_rmse(pred_fn, data):
    P = np.vstack([pred_fn(X) for X, _ in data])
    Y = np.vstack([Y for _, Y in data])
    per = rmse_per_factor(P, Y)
    return float(np.sqrt(((P - Y) ** 2).mean())), per


def run_cnn(data_tr, data_te, window, hidden, epochs, seed):
    m = CNNEncoder(window=window, hidden=hidden, seed=seed, lr=0.01)
    Ws, Ys = [], []
    for X, Y in data_tr:
        W, _ = m._windows(X); Ws.append(W); Ys.append(Y)
    Wc, Yc = np.vstack(Ws), np.vstack(Ys)
    idx = np.arange(len(Yc)); g = np.random.default_rng(seed)
    for _ in range(epochs):
        g.shuffle(idx)
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            m.fit_step(Wc[b], Yc[b])
    f = lambda X: m.forward(m._windows(X)[0])[1]
    return eval_rmse(f, data_tr), eval_rmse(f, data_te)


def run_gru(data_tr, data_te, hidden, epochs, seed):
    m = GRUEncoder(hidden=hidden, seed=seed, lr=0.01)
    order = list(range(len(data_tr)))
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(order)
        for i in order:
            X, Y = data_tr[i]
            m.fit_step(X, Y)
    f = lambda X: m.predict_seq(X)
    return eval_rmse(f, data_tr), eval_rmse(f, data_te)


def main():
    ap = argparse.ArgumentParser(description="因子蒸馏实验")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--mode", type=int, default=0)
    args = ap.parse_args()

    te = build(gen(150, args.steps, args.mode, seed=999), opp_only=False)
    print(f"{'config':<26}{'train':>8}{'test':>8}   最难因子(test top3)")
    print("-" * 70)

    def show(tag, tr, te_):
        (rtr, _), (rte, per) = tr, te_
        hardest = sorted(per.items(), key=lambda kv: -kv[1])[:3]
        hs = " ".join(f"{k}={v:.2f}" for k, v in hardest)
        print(f"{tag:<26}{rtr:>8.3f}{rte:>8.3f}   {hs}")

    # 1) 数据量学习曲线（GRU, 双方输入）
    for n in (100, 300, 600):
        tr = build(gen(n, args.steps, args.mode, seed=1), opp_only=False)
        show(f"GRU  n={n} both", *run_gru(tr, te, 24, args.epochs, seed=1))

    # 2) 架构/窗口（n=300）
    tr = build(gen(300, args.steps, args.mode, seed=1), opp_only=False)
    for w in (2, 4, 8):
        show(f"CNN  w={w} n=300", *run_cnn(tr, te, w, 64, args.epochs, seed=1))
    show("GRU  n=300 both(re)", *run_gru(tr, te, 24, args.epochs, seed=1))

    # 3) 消融：只看对手（n=300）
    tr_opp = build(gen(300, args.steps, args.mode, seed=1), opp_only=True)
    show("GRU  n=300 opp-only", *run_gru(tr_opp, te, 24, args.epochs, seed=1))
    show("CNN w=4 n=300 opp-only", *run_cnn(tr_opp, te, 4, 64, args.epochs, seed=1))


if __name__ == "__main__":
    main()
