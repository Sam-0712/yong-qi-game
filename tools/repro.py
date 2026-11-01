"""一条命令：校验 / 重建 5.0 需要的语料与产物。

背景：``data/`` 被 .gitignore 忽略，换机器或清理一次就会让整条学习链路静默失效——
``pipeline.py`` 发现文件不存在时会自动回退（编码器缺失→手写信念，上下文模型缺失→
不构造 hybrid），**不报错、不告警**，于是默认配置实际跑的是「9 个手写剧本的类型后验」，
而你以为在跑学习型对手模型。本工具把这件事变成一条命令。

用法
----
    python -m tools.repro --check              # 只看现状（不写任何文件）
    python -m tools.repro --build              # 重建 corpus + context 两个产物
    python -m tools.repro --build --games 5000 # 缩小语料规模（快速冒烟）

注意
----
``--build`` 会覆盖 ``data/corpus/corpus_v1.jsonl`` 与 ``data/artifacts/context_*.json``。
编码器（``encoder_gru.json``）不在本工具的重建范围，它由 ``tools/encoder.py`` 训练。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

CORPUS = "data/corpus/corpus_v1.jsonl"
CONTEXT_HOLDOUT = "data/artifacts/context_holdout.json"
CONTEXT_ALL = "data/artifacts/context_all.json"
ENCODER = "data/artifacts/encoder_gru.json"
OPENING = "data/artifacts/opening_mode0_balanced.json"


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_ROOT, p)


REQUIRED: List[Tuple[str, str, bool]] = [
    (CORPUS, "混合对局语料", True),
    (CONTEXT_HOLDOUT, "上下文模型（留出评估版）", True),
    (CONTEXT_ALL, "上下文模型（部署版，train-all）", True),
    (ENCODER, "GRU 序列编码器（可选：缺了会回退手写信念）", False),
    (OPENING, "开局簿（可选：缺了用内置安全分布）", False),
]


def _size(p: str) -> str:
    n = os.path.getsize(p)
    return f"{n/1024/1024:.2f} MB" if n >= 1024 * 1024 else f"{n/1024:.0f} KB"


def check() -> bool:
    ok = True
    print(f"仓库根: {_ROOT}\n")
    print(f"{'状态':<6}{'文件':<44}{'大小':>10}  说明")
    print("-" * 96)
    for rel, desc, hard in REQUIRED:
        p = _abs(rel)
        if os.path.exists(p):
            print(f"{'OK':<6}{rel:<44}{_size(p):>10}  {desc}")
        else:
            print(f"{'缺失':<6}{rel:<44}{'-':>10}  {desc}")
            if hard:
                ok = False
    if not ok:
        print("\n以上「缺失」为必需项：先跑 python -m tools.repro --build")

    # 把产物自带的关键指标读出来，避免「文件存在但内容不对」
    for rel in (CONTEXT_HOLDOUT, CONTEXT_ALL):
        p = _abs(rel)
        if not os.path.exists(p):
            continue
        try:
            blob = json.load(open(p, encoding="utf-8"))
        except Exception as exc:                      # noqa: BLE001
            print(f"\n{rel} 读取失败: {exc}")
            continue
        meta = blob.get("meta", {})
        m = meta.get("metrics") or {}
        n_keys = {lv: len(v) for lv, v in (blob.get("levels") or {}).items()}
        print(f"\n{rel}")
        print(f"  corpus={meta.get('corpus')} games={meta.get('games')} "
              f"train={meta.get('train')} test={meta.get('test')} "
              f"holdout={meta.get('holdout')} train_all={meta.get('train_all')}")
        print(f"  键数={n_keys}")
        if m:
            print(f"  test logloss={m.get('logloss'):.4f} acc={m.get('acc'):.3f} "
                  f"uniform={m.get('uniform'):.4f}")
            for pol, ps in (m.get("per_policy") or {}).items():
                if pol in (meta.get("holdout") or []):
                    print(f"    holdout {pol:<14} logloss={ps['logloss']:.4f} "
                          f"acc={ps['acc']:.3f} n={ps['n']}")
    return ok


def _run(module: str, *args: str) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=_ROOT)
    if res.returncode != 0:
        raise SystemExit(f"{module} 失败（exit={res.returncode}）")


def build(games: int, seed: int) -> None:
    _run("tools.gen_corpus", "--games", str(games), "--seed", str(seed),
         "--out", CORPUS)
    # 留出版本：按策略名留出 pressure/alternate_low，用来看泛化
    _run("tools.train_context", "--corpus", CORPUS,
         "--holdout", "pressure,alternate_low", "--out", CONTEXT_HOLDOUT)
    # 部署版本：全量训练
    _run("tools.train_context", "--corpus", CORPUS, "--train-all", "--out", CONTEXT_ALL)


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 语料与产物校验/重建")
    ap.add_argument("--check", action="store_true", help="只校验现状")
    ap.add_argument("--build", action="store_true", help="重建 corpus + context 产物")
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.check and not args.build:
        args.check = True
    if args.build:
        build(args.games, args.seed)
        print()
    ok = check()
    if args.build and ok:
        print("\n产物齐全。下一步：python -m tools.bench --models type,context,hybrid")


if __name__ == "__main__":
    main()
