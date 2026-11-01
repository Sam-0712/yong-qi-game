"""从混合对局语料训练上下文对手模型（纯计数，无手写风格规则）。

用法：
  python -m tools.train_context --corpus data/corpus/corpus_v1.jsonl \
      --holdout pressure,alternate_low --out data/artifacts/context_holdout.json
  python -m tools.train_context --corpus data/corpus/corpus_v1.jsonl \
      --train-all --out data/artifacts/context_all.json

报告留出风格的 logloss/accuracy；部署模型建议用 --train-all。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.core.rules import get_valid_moves
from courage.learning.context_model import (ContextOpponentModel, LEVELS,
                                            context_keys)


def _iter_samples(rec):
    own, opp, qis = rec["own_seq"], rec["opp_seq"], rec["qi_seq"]
    for t in range(len(own)):
        qa, qp = qis[t]
        yield (rec["p2"], qp, qa, own[:t], opp[:t], opp[t])
        yield (rec["p1"], qa, qp, opp[:t], own[:t], own[t])


def build_counts(records) -> dict:
    cnt = {lv: defaultdict(Counter) for lv in LEVELS}
    for rec in records:
        for _pol, qa, qp, own_hist, opp_hist, label in _iter_samples(rec):
            for lv, key in context_keys(own_hist, opp_hist, qa, qp):
                cnt[lv][key][label] += 1
    return {lv: {k: dict(v) for k, v in cnt[lv].items()} for lv in LEVELS}


def evaluate(records, counts, smooth: float):
    # p1 / p2 各自维护自己的观察历史，避免用对方视角的序列泄漏。
    n = 0
    ll = 0.0
    acc = 0
    uni = 0.0
    per = defaultdict(lambda: {"n": 0, "ll": 0.0, "acc": 0})
    for rec in records:
        m1 = ContextOpponentModel(counts=counts, smooth=smooth)
        m2 = ContextOpponentModel(counts=counts, smooth=smooth)
        for t in range(len(rec["own_seq"])):
            a, b = rec["own_seq"][t], rec["opp_seq"][t]
            qa, qp = rec["qi_seq"][t]
            legal1 = get_valid_moves(qp, mode=0)
            legal2 = get_valid_moves(qa, mode=0)
            w1 = m1.opp_weights(qa, qp, legal1)
            w2 = m2.opp_weights(qp, qa, legal2)
            p1 = max(w1.get(b, 0.0), 1e-12)
            p2 = max(w2.get(a, 0.0), 1e-12)
            ll -= math.log(p1) + math.log(p2)
            acc += (1 if max(legal1, key=lambda m: w1.get(m, 0.0)) == b else 0)
            acc += (1 if max(legal2, key=lambda m: w2.get(m, 0.0)) == a else 0)
            uni -= math.log(1.0 / len(legal1)) + math.log(1.0 / len(legal2))
            ps1 = per[rec["p2"]]; ps1["n"] += 1
            ps1["ll"] -= math.log(p1)
            ps1["acc"] += 1 if max(legal1, key=lambda m: w1.get(m, 0.0)) == b else 0
            ps2 = per[rec["p1"]]; ps2["n"] += 1
            ps2["ll"] -= math.log(p2)
            ps2["acc"] += 1 if max(legal2, key=lambda m: w2.get(m, 0.0)) == a else 0
            m1.observe(qa, qp, a, b)
            m2.observe(qp, qa, b, a)
            n += 2
    return {"n": n, "logloss": ll / n, "acc": acc / n, "uniform": uni / n,
            "per_policy": {k: {"n": v["n"], "logloss": v["ll"] / v["n"],
                              "acc": v["acc"] / v["n"]} for k, v in per.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 上下文对手模型训练")
    ap.add_argument("--corpus", default="data/corpus/corpus_v1.jsonl")
    ap.add_argument("--holdout", default="pressure,alternate_low")
    ap.add_argument("--train-all", dest="train_all", action="store_true")
    ap.add_argument("--smooth", type=float, default=0.5)
    ap.add_argument("--test-frac", dest="test_frac", type=float, default=0.2)
    ap.add_argument("--out", default="data/artifacts/context_opponent.json")
    args = ap.parse_args()

    path = args.corpus if os.path.isabs(args.corpus) else os.path.join(_ROOT, args.corpus)
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    holdout = set(x for x in args.holdout.split(",") if x)
    train, test = [], []
    if args.train_all:
        train = recs
    else:
        for i, r in enumerate(recs):
            if r["p1"] in holdout or r["p2"] in holdout:
                test.append(r)
            elif i % max(1, int(1.0 / max(args.test_frac, 1e-6))) == 0:
                test.append(r)
            else:
                train.append(r)

    counts = build_counts(train)
    metrics = evaluate(test, counts, args.smooth) if test else {}
    print(f"corpus={path} games={len(recs)} train={len(train)} test={len(test)}")
    if metrics:
        print(f"  n={metrics['n']} logloss={metrics['logloss']:.4f} "
              f"acc={metrics['acc']:.3f} uniform={metrics['uniform']:.4f}")
        for pol in sorted(holdout):
            ps = metrics["per_policy"].get(pol)
            if ps:
                print(f"  holdout {pol:<12} n={ps['n']:>6} "
                      f"logloss={ps['logloss']:.4f} acc={ps['acc']:.3f}")

    meta = {"corpus": os.path.relpath(path, _ROOT), "games": len(recs),
            "train": len(train), "test": len(test), "holdout": sorted(holdout),
            "train_all": args.train_all, "metrics": metrics}
    out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ContextOpponentModel(counts=counts, smooth=args.smooth,
                         meta=meta).save(out)
    print(f"written {out}  size={os.path.getsize(out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
