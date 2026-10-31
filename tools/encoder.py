"""训练序列编码器：用模拟对局序列蒸馏信念因子。

用法：
  python -m tools.encoder --arch both --seqs 400 --epochs 12
报告每个因子的 RMSE（越接近 0，网络越贴合手写因子）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from courage.bots.scripts import ALL_BOTS, sample_policy
from courage.core.rules import get_valid_moves, judge, next_qi
from courage.learning.seq_encoder import (CNNEncoder, GRUEncoder, FACTOR_KEYS,
                                          rmse_per_factor, sequence_to_xy)
from courage.storage.artifacts import artifact_path

TK = {"alpha": 0.20, "smooth": 0.05}  # 与在线信念一致


def gen_sequences(n, max_steps, mode, seed):
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        b1, b2 = rng.choice(ALL_BOTS), rng.choice(ALL_BOTS)
        qa = qp = 3
        own, opp = [], []
        for _t in range(max_steps):
            a = sample_policy(b1.policy(qa, qp, mode=mode, last_opp=opp[-1] if opp else None), rng)
            b = sample_policy(b2.policy(qp, qa, mode=mode, last_opp=own[-1] if own else None), rng)
            own.append(a); opp.append(b)
            if judge(a, b) != 0:
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        if len(own) >= 2:
            seqs.append((own, opp))
    return seqs


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 序列编码器训练")
    ap.add_argument("--arch", default="both", choices=["cnn", "gru", "both"])
    ap.add_argument("--seqs", type=int, default=400)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=64, help="CNN 隐层")
    ap.add_argument("--ghidden", type=int, default=24, help="GRU 隐层")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    seqs = gen_sequences(args.seqs, args.steps, args.mode, args.seed)
    steps_total = sum(len(s[0]) for s in seqs)
    print(f"序列 {len(seqs)} 条 / 步 {steps_total} 步 / 因子维 {len(FACTOR_KEYS)}")

    # 预生成 (X,Y)
    data = [sequence_to_xy(o, p, TK) for (o, p) in seqs]

    if args.arch in ("cnn", "both"):
        cnn = CNNEncoder(window=args.window, hidden=args.hidden, seed=args.seed, lr=args.lr)
        Ws, Ys = [], []
        for X, Y in data:
            W, _ = cnn._windows(X)
            Ws.append(W); Ys.append(Y)
        Wcat = np.vstack(Ws); Ycat = np.vstack(Ys)
        idx = np.arange(len(Ycat)); g = np.random.default_rng(args.seed)
        for ep in range(args.epochs):
            g.shuffle(idx)
            tot = 0.0
            for i in range(0, len(idx), 128):
                b = idx[i:i + 128]
                tot += cnn.fit_step(Wcat[b], Ycat[b])
            if (ep + 1) % max(1, args.epochs // 4) == 0:
                print(f"  [CNN] epoch {ep+1:>2}  mse={tot/(len(idx)//128+1):.5f}")
        pred = cnn.forward(Wcat)[1]
        per = rmse_per_factor(pred, Ycat)
        overall = float(np.sqrt(((pred - Ycat) ** 2).mean()))
        print(f"[CNN] 参数={cnn.n_params()}  总体 RMSE={overall:.4f}")
        print("  " + " ".join(f"{k}={per[k]:.3f}" for k in FACTOR_KEYS))
        if args.save:
            out = artifact_path("encoder_cnn.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"arch": "cnn", "window": args.window, "hidden": args.hidden,
                           **{k: v.tolist() for k, v in cnn.params.items()}}, f)
            print("  已保存:", out)

    if args.arch in ("gru", "both"):
        import json
        gru = GRUEncoder(hidden=args.ghidden, seed=args.seed, lr=args.lr)
        for ep in range(args.epochs):
            tot, n = 0.0, 0
            order = list(range(len(data)))
            random.Random(args.seed + ep).shuffle(order)
            for i in order:
                X, Y = data[i]
                tot += gru.fit_step(X, Y); n += 1
            if (ep + 1) % max(1, args.epochs // 4) == 0:
                print(f"  [GRU] epoch {ep+1:>2}  mse={tot/n:.5f}")
        P = np.vstack([gru.predict_seq(X) for X, _ in data])
        Yc = np.vstack([Y for _, Y in data])
        per = rmse_per_factor(P, Yc)
        overall = float(np.sqrt(((P - Yc) ** 2).mean()))
        print(f"[GRU] 参数={gru.n_params()}  总体 RMSE={overall:.4f}")
        print("  " + " ".join(f"{k}={per[k]:.3f}" for k in FACTOR_KEYS))
        if args.save:
            out = artifact_path("encoder_gru.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"arch": "gru", "hidden": args.ghidden,
                           **{k: v.tolist() for k, v in gru.params.items()}}, f)
            print("  已保存:", out)


if __name__ == "__main__":
    import json
    main()
