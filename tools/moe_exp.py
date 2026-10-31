"""MoE 可行性实验：等参数/等算力下，多小专家 vs 单一大 GRU vs CNN。

结构：
  single-H   ：单个 GRU(hidden=H)
  cnn-w4     ：因果窗口 CNN
  moe-E-hs   ：按风格路由到 E 个 GRU(hidden=hs) 专家
指标：参数量、因子 RMSE(train/test)、以及编码器因子喂 BR(tau=2) 后的胜率/回合。

用法：python -m tools.moe_exp
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

import numpy as np

from courage.agents import BotAgent, OnlineBRAgent, run_duel
from courage.bots.scripts import ALL_BOTS, bots_by_name, sample_policy
from courage.core.rules import get_valid_moves, judge, next_qi
from courage.learning.seq_encoder import (CNNEncoder, GRUEncoder, MoEStyleEncoder,
                                          EncoderBelief, sequence_to_xy,
                                          rmse_per_factor, route_style)

TK = {"alpha": 0.20, "smooth": 0.05}


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


def prep(seqs):
    return [(o, p, *sequence_to_xy(o, p, TK)) for (o, p) in seqs]


def train_single(data, H, epochs, seed):
    m = GRUEncoder(hidden=H, seed=seed, lr=0.01)
    order = list(range(len(data)))
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(order)
        for i in order:
            _, _, X, Y = data[i]
            m.fit_step(X, Y)
    return m, lambda d: m.predict_seq(d[2])


def train_cnn(data, w, hidden, epochs, seed):
    m = CNNEncoder(window=w, hidden=hidden, seed=seed, lr=0.01)
    Ws, Ys = [], []
    for _, _, X, Y in data:
        W, _ = m._windows(X); Ws.append(W); Ys.append(Y)
    Wc, Yc = np.vstack(Ws), np.vstack(Ys)
    idx = np.arange(len(Yc)); g = np.random.default_rng(seed)
    for _ in range(epochs):
        g.shuffle(idx)
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            m.fit_step(Wc[b], Yc[b])
    return m, lambda d: m.forward(m._windows(d[2])[0])[1]


def train_moe(data, E, hs, epochs, seed):
    m = MoEStyleEncoder(n_experts=E, hidden=hs, seed=seed, lr=0.01)
    counts = [0] * E
    order = list(range(len(data)))
    for ep in range(epochs):
        random.Random(seed + ep).shuffle(order)
        for i in order:
            _, p, X, Y = data[i]
            e = m._route(p)
            m.experts[e].fit_step(X, Y)
            counts[e] += 1
    return m, (lambda d: m.predict_seq(d[2], d[1])), f"[{','.join(map(str, counts))}]"


def eval_rmse(pred_fn, data):
    P = np.vstack([pred_fn(d) for d in data])
    Y = np.vstack([d[3] for d in data])
    return float(np.sqrt(((P - Y) ** 2).mean())), rmse_per_factor(P, Y)


def decision_eval(model, opps, games, cap):
    out = {}
    for name in opps:
        ai = OnlineBRAgent(step_cost=0.03, br_sweeps=150,
                           belief_like=EncoderBelief(model, tau=2.0))
        out[name] = run_duel(ai, BotAgent(bots_by_name[name]), mode=0,
                             games=games, cap=cap)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--cap", type=int, default=60)
    args = ap.parse_args()

    tr = prep(gen(args.train, args.steps, 0, seed=1))
    te = prep(gen(args.test, args.steps, 0, seed=999))

    configs = []
    m1, f1 = train_single(tr, 28, args.epochs, 1); configs.append(("single GRU H=28", m1, f1, ""))
    m2, f2 = train_cnn(tr, 4, 64, args.epochs, 1); configs.append(("CNN w=4 h=64", m2, f2, ""))
    m3, f3, c3 = train_moe(tr, 4, 10, args.epochs, 1); configs.append(("MoE E=4 hs=10", m3, f3, c3))
    m4, f4, c4 = train_moe(tr, 2, 18, args.epochs, 1); configs.append(("MoE E=2 hs=18", m4, f4, c4))

    print(f"{'结构':<18}{'参数':>7}{'train':>8}{'test':>8}   最难(test)")
    print("-" * 64)
    for name, m, f, c in configs:
        rtr, _ = eval_rmse(f, tr)
        rte, per = eval_rmse(f, te)
        hard = sorted(per.items(), key=lambda kv: -kv[1])[:2]
        hs = " ".join(f"{k}={v:.2f}" for k, v in hard)
        extra = f"  专家样本{c}" if c else ""
        print(f"{name:<18}{m.n_params():>7}{rtr:>8.3f}{rte:>8.3f}   {hs}{extra}")

    # 决策对比：single vs MoE E=4
    print("\n决策(编码器因子→BR tau=2) 胜率/回合：")
    opps = ["turtle", "defensive", "aggressive", "random"]
    r_single = decision_eval(m1, opps, args.games, args.cap)
    r_moe = decision_eval(m3, opps, args.games, args.cap)
    print(f"{'对手':<10}{'single GRU':>20}{'MoE E4':>20}")
    for name in opps:
        a, b = r_single[name], r_moe[name]
        print(f"{name:<10}{a['wins']/a['games']:>10.0%}/{statistics.mean(a['rounds']):>4.1f}r"
              f"{b['wins']/b['games']:>12.0%}/{statistics.mean(b['rounds']):>4.1f}r")


if __name__ == "__main__":
    main()
