"""训练序列编码器：用（扩大的）随机对局 + 人造残局/镜像/重复序列蒸馏信念因子。

用法：
  python -m tools.encoder --arch gru --seqs 2000 --steps 24 --epochs 12 --ghidden 24 --save
  python -m tools.encoder --arch gru --seqs 2000 --ghidden 32 --tag gru_h32 --save
  python -m tools.encoder --arch both --seqs 600 --epochs 8

设计要点：
  - 随机起始气数（含残局）→ 覆盖低气/高气下的出招分布；
  - 人造模式（镜像/交替/重复/循环）→ 覆盖一阶频率看不见的序列结构；
  - 输入只保留最近 max_len 步、目标仍按完整历史计算 → 与在线 EncoderBelief 对齐；
  - 固定 train/val/test 切分，报告 test RMSE；选型不能只看训练 RMSE。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from courage.bots.scripts import ALL_BOTS, sample_policy
from courage.core.rules import get_valid_moves, judge, next_qi
from courage.learning.belief import CATEGORIES, CATS
from courage.learning.threat import converter_policy
from courage.learning.seq_encoder import (CNNEncoder, GRUEncoder, FACTOR_KEYS,
                                          rmse_per_factor, sequence_to_xy)
from courage.storage.artifacts import artifact_path

TK = {"alpha": 0.20, "smooth": 0.05}  # 与在线信念一致
CAT_OF = CATEGORIES


def _start_qi(rng: random.Random, random_start: float) -> tuple[int, int]:
    """随机起始气数：覆盖残局（低气）与中局；否则标准 3,3。"""
    if random_start > 0 and rng.random() < random_start:
        return rng.randint(0, 5), rng.randint(0, 5)
    return 3, 3


def gen_game_sequences(n: int, max_steps: int, mode: int, seed: int,
                       random_start: float = 0.5) -> list[tuple[list, list]]:
    """随机 bot 对局；一部分从随机气数（残局）开始。"""
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        b1, b2 = rng.choice(ALL_BOTS), rng.choice(ALL_BOTS)
        qa, qp = _start_qi(rng, random_start)
        own, opp = [], []
        for _t in range(max_steps):
            a = sample_policy(b1.policy(qa, qp, mode=mode,
                                        last_opp=opp[-1] if opp else None), rng)
            b = sample_policy(b2.policy(qp, qa, mode=mode,
                                        last_opp=own[-1] if own else None), rng)
            own.append(a); opp.append(b)
            if judge(a, b) != 0:
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        if len(own) >= 2:
            seqs.append((own, opp))
    return seqs


def _moves_in_cat(cat: str, legal: list[str]) -> list[str]:
    return [m for m in legal if CAT_OF.get(m, "attack") == cat]


def _pick_cat(rng: random.Random, cat: str, legal: list[str]) -> str:
    cand = _moves_in_cat(cat, legal)
    return rng.choice(cand) if cand else rng.choice(legal)


def gen_pattern_sequences(n: int, max_len: int, mode: int,
                          seed: int) -> list[tuple[list, list]]:
    """人造模式：镜像 / 两类交替 / 同招重复 / 循环。

    这些序列的一阶边际往往接近均匀，只有转移结构（bigram / 长程记忆）
    能区分；正是编码器与 online belief 需要补的能力。
    """
    rng = random.Random(seed)
    seqs = []
    kinds = ["mirror", "alternate", "repeat", "cycle",
             "exploit", "two", "converter"]
    alt_pairs = [("exhale", "attack"), ("defense", "attack"),
                 ("dodge", "attack"), ("exhale", "heavy")]
    exploit = ["防御", "吐气", "吐气", "咏春"]
    for _ in range(n):
        kind = rng.choice(kinds)
        if kind in ("exploit", "two"):
            qa, qp = 3, 3          # 套路从标准开局开始，否则招式不合法
        else:
            qa, qp = _start_qi(rng, 0.7)
        own, opp = [], []
        length = rng.randint(max(4, max_len // 2), max_len)
        run_cat = None
        run_left = 0
        for t in range(length):
            la = get_valid_moves(qa, mode=mode)
            lb = get_valid_moves(qp, mode=mode)
            a = rng.choice(la)

            if kind == "mirror":
                b = own[-1] if (t > 0 and own[-1] in lb) else rng.choice(lb)
            elif kind == "alternate":
                pair = alt_pairs[(t // 4) % len(alt_pairs)]
                b = _pick_cat(rng, pair[t % 2], lb)
            elif kind == "repeat":
                if run_left <= 0:
                    run_cat = rng.choice(CATS)
                    run_left = rng.randint(2, 6)
                b = _pick_cat(rng, run_cat, lb)
                run_left -= 1
            elif kind == "cycle":
                b = lb[(qa + qp + t) % len(lb)]
            elif kind == "exploit":
                want = exploit[t] if t < len(exploit) else (
                    "咏春" if qp >= 5 else "吐气")
                b = want if want in lb else rng.choice(lb)
            elif kind == "two":
                if t == 0:
                    b = "防御" if "防御" in lb else rng.choice(lb)
                elif t == 1:
                    b = "馒头" if "馒头" in lb else rng.choice(lb)
                else:
                    b = sample_policy(converter_policy(lb, la), rng)
            else:  # converter
                b = sample_policy(converter_policy(lb, la), rng)

            own.append(a); opp.append(b)
            if judge(a, b) != 0:
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        if len(own) >= 2:
            seqs.append((own, opp))
    return seqs


def gen_endgame_sequences(n: int, max_len: int, mode: int,
                          seed: int) -> list[tuple[list, list]]:
    """残局练习：从低气（0–2）开始，双方都按「理性转化者」出招。

    这样编码器能看到低气下真正最优的序列（而不是随机 bot 的均匀拖沓），
    减少它把「低气 = 乱跳/乱防」学成通用先验。
    """
    rng = random.Random(seed)
    seqs = []
    for _ in range(n):
        qa, qp = rng.randint(0, 2), rng.randint(0, 2)
        own, opp = [], []
        for _t in range(max_len):
            la = get_valid_moves(qa, mode=mode)
            lb = get_valid_moves(qp, mode=mode)
            a = sample_policy(converter_policy(la, lb), rng)
            b = sample_policy(converter_policy(lb, la), rng)
            own.append(a); opp.append(b)
            if judge(a, b) != 0:
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        if len(own) >= 2:
            seqs.append((own, opp))
    return seqs


def split_sequences(seqs, seed: int, train_frac: float = 0.8,
                    val_frac: float = 0.1):
    rng = random.Random(seed)
    seqs = list(seqs)
    rng.shuffle(seqs)
    n = len(seqs)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = seqs[:n_train]
    val = seqs[n_train:n_train + n_val]
    test = seqs[n_train + n_val:]
    return train, val, test


def to_xy(seqs, max_len: int):
    return [sequence_to_xy(o, p, TK, max_len=max_len) for (o, p) in seqs]


def eval_rmse(model, data) -> tuple[float, dict]:
    if not data:
        return 0.0, {}
    P = np.vstack([model.predict_seq(X) for X, _ in data])
    Y = np.vstack([Y for _, Y in data])
    per = rmse_per_factor(P, Y)
    return float(np.sqrt(((P - Y) ** 2).mean())), per


def train_gru(data, hidden: int, seed: int, epochs: int, lr: float):
    gru = GRUEncoder(hidden=hidden, seed=seed, lr=lr)
    for ep in range(epochs):
        order = list(range(len(data)))
        random.Random(seed * 1009 + ep).shuffle(order)
        tot = 0.0
        for i in order:
            X, Y = data[i]
            tot += gru.fit_step(X, Y)
        if (ep + 1) % max(1, epochs // 4) == 0:
            print(f"  [GRU] epoch {ep+1:>2}  mse={tot/max(1, len(order)):.5f}")
    return gru


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 序列编码器训练（扩大数据 + 残局/模式）")
    ap.add_argument("--arch", default="gru", choices=["cnn", "gru", "both"])
    ap.add_argument("--seqs", type=int, default=3000, help="总序列数")
    ap.add_argument("--steps", type=int, default=32, help="单局最大步数（可超过输入窗口）")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--window", type=int, default=4, help="CNN 窗口")
    ap.add_argument("--hidden", type=int, default=64, help="CNN 隐层")
    ap.add_argument("--ghidden", type=int, default=24, help="GRU 隐层（24–48 扫参）")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="模型初始化种子")
    ap.add_argument("--data-seed", dest="data_seed", type=int, default=None,
                    help="数据生成种子（缺省等于 --seed）")
    ap.add_argument("--random-start", dest="random_start", type=float, default=0.5,
                    help="随机起始气数（残局）比例")
    ap.add_argument("--pattern-frac", dest="pattern_frac", type=float, default=0.35,
                    help="人造模式序列占比")
    ap.add_argument("--endgame-frac", dest="endgame_frac", type=float, default=0.15,
                    help="低气残局（转化者最优序列）占比")
    ap.add_argument("--max-len", dest="max_len", type=int, default=24,
                    help="输入窗口（与在线 EncoderBelief 一致）")
    ap.add_argument("--train-frac", dest="train_frac", type=float, default=0.8)
    ap.add_argument("--val-frac", dest="val_frac", type=float, default=0.1)
    ap.add_argument("--tag", default="gru", help="产物名：encoder_<tag>.json")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    data_seed = args.seed if args.data_seed is None else args.data_seed
    n_pattern = int(args.seqs * max(0.0, min(1.0, args.pattern_frac)))
    n_end = int(args.seqs * max(0.0, min(1.0, args.endgame_frac)))
    n_game = max(0, args.seqs - n_pattern - n_end)
    games = gen_game_sequences(n_game, args.steps, args.mode, data_seed,
                               args.random_start)
    pats = gen_pattern_sequences(n_pattern, args.max_len, args.mode,
                                 data_seed + 7919)
    ends = gen_endgame_sequences(n_end, args.max_len, args.mode,
                                 data_seed + 104729)
    seqs = games + pats + ends
    train, val, test = split_sequences(seqs, data_seed + 17,
                                        args.train_frac, args.val_frac)
    print(f"序列 {len(seqs)} 条（对局 {len(games)} / 模式 {len(pats)} / "
          f"残局 {len(ends)}） train/val/test = "
          f"{len(train)}/{len(val)}/{len(test)}")
    print(f"步数合计 {sum(len(s[0]) for s in seqs)} / 窗口 {args.max_len} / "
          f"随机起始 {args.random_start:.0%}")

    data_tr = to_xy(train, args.max_len)
    data_va = to_xy(val, args.max_len)
    data_te = to_xy(test, args.max_len)

    if args.arch in ("cnn", "both"):
        cnn = CNNEncoder(window=args.window, hidden=args.hidden,
                         seed=args.seed, lr=args.lr)
        Ws, Ys = [], []
        for X, Y in data_tr:
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
                print(f"  [CNN] epoch {ep+1:>2}  mse={tot/max(1, len(idx)//128):.5f}")
        tr_rmse, per = eval_rmse(cnn, data_tr)
        va_rmse, _ = eval_rmse(cnn, data_va)
        te_rmse, _ = eval_rmse(cnn, data_te)
        print(f"[CNN] 参数={cnn.n_params()}  train={tr_rmse:.4f} "
              f"val={va_rmse:.4f} test={te_rmse:.4f}")
        print("  " + " ".join(f"{k}={per[k]:.3f}" for k in FACTOR_KEYS))
        if args.save:
            out = artifact_path(f"encoder_cnn_{args.tag}.json")
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"arch": "cnn", "window": args.window,
                           "hidden": args.hidden,
                           **{k: v.tolist() for k, v in cnn.params.items()}}, f)
            print("  已保存:", out)

    if args.arch in ("gru", "both"):
        gru = train_gru(data_tr, args.ghidden, args.seed, args.epochs, args.lr)
        tr_rmse, per = eval_rmse(gru, data_tr)
        va_rmse, _ = eval_rmse(gru, data_va)
        te_rmse, _ = eval_rmse(gru, data_te)
        print(f"[GRU] 参数={gru.n_params()}  train={tr_rmse:.4f} "
              f"val={va_rmse:.4f} test={te_rmse:.4f}")
        print("  " + " ".join(f"{k}={per[k]:.3f}" for k in FACTOR_KEYS))
        if args.save:
            out = artifact_path(f"encoder_{args.tag}.json")
            meta = {
                "hidden": args.ghidden, "seed": args.seed, "data_seed": data_seed,
                "seqs": len(seqs), "games": len(games), "patterns": len(pats),
                "steps": args.steps, "max_len": args.max_len,
                "epochs": args.epochs, "random_start": args.random_start,
                "pattern_frac": args.pattern_frac,
                "split": [len(train), len(val), len(test)],
                "train_rmse": tr_rmse, "val_rmse": va_rmse, "test_rmse": te_rmse,
                "created": int(time.time()),
            }
            with open(out, "w", encoding="utf-8") as f:
                json.dump({"arch": "gru", "hidden": args.ghidden,
                           **{k: v.tolist() for k, v in gru.params.items()},
                           "meta": meta}, f)
            print("  已保存:", out)


if __name__ == "__main__":
    main()
