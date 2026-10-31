"""信念因子回归测试（python -m tests.test_belief）。

验证用户的关键例子：
  「连攻数手后转守」与「攻防交替」应被因子区分——
  前者变化点(switch)/长程防御(long_def)更强，而非只看上一招。
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from courage.learning.belief import BeliefTracker


def _feed(moves, own="馒头", decay=0.7):
    tr = BeliefTracker(decay=decay, alpha=0.05, smooth=0.0)
    for m in moves:
        tr.observe(own, m)
    return tr


def main() -> None:
    # A: 连攻三手后转守
    A = _feed(["雷切", "雷切", "雷切", "防御", "防御", "防御"])
    # B: 攻防交替
    B = _feed(["雷切", "防御", "雷切", "防御", "雷切", "防御"])

    fA, fB = A.factors(), B.factors()

    for v in fA.values():
        assert 0.0 <= v <= 1.0, v
    for v in fB.values():
        assert 0.0 <= v <= 1.0, v

    # A 在末段转入防御 → 长程防御更高
    assert fA["long_def"] > fB["long_def"], (fA["long_def"], fB["long_def"])
    # A 的短/长程分歧更大（变化点更明显）
    assert fA["switch"] > fB["switch"] + 0.05, (fA["switch"], fB["switch"])
    # B 更分散
    assert fB["volatility"] > fA["volatility"], (fB["volatility"], fA["volatility"])

    print(f"A(连攻转守): long_def={fA['long_def']:.2f} switch={fA['switch']:.2f} "
          f"vol={fA['volatility']:.2f}  -> {BeliefTracker.describe_style(fA)}")
    print(f"B(攻防交替): long_def={fB['long_def']:.2f} switch={fB['switch']:.2f} "
          f"vol={fB['volatility']:.2f}  -> {BeliefTracker.describe_style(fB)}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
