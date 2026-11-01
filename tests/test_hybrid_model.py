"""混合对手模型回归测试（python -m tests.test_hybrid_model）。

验证：
  - 类型后验「已知类型质量高」时用类型；
  - top 是 random / 已知类型质量低时切到上下文模型；
  - observe/reset 会同时更新两条通道。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.learning.hybrid_model import HybridOpponentModel


class FakeType:
    def __init__(self, w):
        self.w = w
        self.steps = 0

    def reset(self):
        self.steps = 0

    def observe(self, qa, qp, own, opp):
        self.steps += 1

    def opp_weights(self, qa, qp, legal, last_own=None):
        return {m: 1.0 / len(legal) if m == legal[0] else 0.0 for m in legal}

    def top_types(self, k=1):
        return sorted(self.w.items(), key=lambda kv: -kv[1])[:k]


class FakeCtx:
    def __init__(self):
        self.steps = 0

    def reset(self):
        self.steps = 0

    def observe(self, qa, qp, own, opp):
        self.steps += 1

    def opp_weights(self, qa, qp, legal, last_own=None):
        return {m: 1.0 if m == legal[-1] else 0.0 for m in legal}


def _top(legal, w):
    return max(legal, key=lambda m: w[m])


def main() -> None:
    legal = ["吐气", "馒头", "防御"]

    known = FakeType({"converter": 0.8, "random": 0.2})
    ctx = FakeCtx()
    h = HybridOpponentModel(known, ctx)
    w = h.opp_weights(2, 2, legal)
    assert _top(legal, w) == legal[0], w  # 用类型
    print("  type gate: known mass -> type")

    unknown = FakeType({"random": 0.9, "converter": 0.1})
    h2 = HybridOpponentModel(unknown, ctx)
    w2 = h2.opp_weights(2, 2, legal)
    assert _top(legal, w2) == legal[-1], w2  # 用上下文
    print("  type gate: random-dominant -> context")

    h.observe(2, 2, "防御", "吐气")
    assert known.steps == 1 and ctx.steps == 1
    h.reset()
    assert known.steps == 0 and ctx.steps == 0
    print("  observe/reset: both channels updated")
    print("ALL HYBRID TESTS PASSED")


if __name__ == "__main__":
    main()
