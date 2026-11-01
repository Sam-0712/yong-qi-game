"""对手类型后验回归测试（python -m tests.test_type_model）。

验证核心场景：R1 双方都可能出防御；R2 才分叉。
  - 只看到 R1 防御时，μ 必须同时保留「防御 / 攻击(馒头) / 吐气」三种可能；
  - 观察到 R2 吐气 → exploit4 权重上升；
  - 观察到 R2 馒头 → two 权重上升。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.core.rules import get_valid_moves
from courage.learning.type_model import OpponentTypeModel


def _top(tm, k=3):
    return [t for t, _ in tm.top_types(k)]


def test_r1_defense_keeps_three_branches() -> None:
    tm = OpponentTypeModel()
    tm.observe(3, 3, "防御", "防御")
    mu = tm.opp_weights(3, 3, get_valid_moves(3, mode=0), last_own="防御")
    assert mu["防御"] > 0.10, mu
    assert mu["馒头"] > 0.05, mu
    assert mu["吐气"] > 0.05, mu
    print(f"  R1 防御 -> top={_top(tm)}  "
          f"mu(防御/馒头/吐气)={mu['防御']:.2f}/{mu['馒头']:.2f}/{mu['吐气']:.2f}")


def test_r2_branch_updates_posterior() -> None:
    # 4 手：R1 防御 → R2 吐气
    a = OpponentTypeModel()
    a.observe(3, 3, "防御", "防御")
    a.observe(3, 3, "防御", "吐气")
    assert a.w["exploit4"] > a.w["two"], a.w
    print(f"  R2 吐气 -> exploit4={a.w['exploit4']:.2f} two={a.w['two']:.2f}")

    # 2 手：R1 防御 → R2 馒头
    b = OpponentTypeModel()
    b.observe(3, 3, "防御", "防御")
    b.observe(3, 3, "防御", "馒头")
    assert b.w["two"] > b.w["exploit4"], b.w
    print(f"  R2 馒头 -> two={b.w['two']:.2f} exploit4={b.w['exploit4']:.2f}")


def test_mu_sums_and_legal() -> None:
    tm = OpponentTypeModel()
    tm.observe(3, 3, "防御", "防御")
    legal = get_valid_moves(3, mode=0)
    mu = tm.opp_weights(3, 3, legal, last_own="防御")
    assert set(mu) == set(legal)
    assert abs(sum(mu.values()) - 1.0) < 1e-9, mu
    print("  mu normalized & legal-only: ok")


def test_pipeline_integration() -> None:
    from courage.agents import BotAgent, run_duel
    from courage.bots.scripts import bots_by_name
    from courage.pipeline import PlayerConfig, build_player

    p = build_player(PlayerConfig(type_posterior=True, robust=0.0, seed=1))
    res = run_duel(p, BotAgent(bots_by_name["turtle"]), mode=0,
                   games=3, cap=99, seed=1)
    assert res["games"] == 3
    print(f"  pipeline integration: vs turtle {res['wins']}W/{res['losses']}L/"
          f"{res['timeouts']}T")


def main() -> None:
    test_r1_defense_keeps_three_branches()
    test_r2_branch_updates_posterior()
    test_mu_sums_and_legal()
    test_pipeline_integration()
    print("ALL TYPE-MODEL TESTS PASSED")


if __name__ == "__main__":
    main()
