"""确定性 / 结算兜底 / 信念通道回归测试（python -m tests.test_determinism）。

P0 的三条底线：
  1. 同一 seed 重跑，逐局结果与回合完全一致；
  2. 不给 cap 时用兜底 99，而不是产品结算上限；
  3. Python 侧编码器信念（含 bigram → μ）可复现，且通道真的生效。
"""

from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.agents import BotAgent, run_duel
from courage.bots.scripts import bots_by_name
from courage.core.rules import get_valid_moves
from courage.learning.seq_encoder import EncoderBelief, load_encoder
from courage.pipeline import PlayerConfig, build_player

ENC = os.path.join(_ROOT, "data", "artifacts", "encoder_gru.json")


def _duel(seed: int, opp: str = "mirror", games: int = 20):
    ai = build_player(PlayerConfig(seed=seed))
    return run_duel(ai, BotAgent(bots_by_name[opp]), mode=0, games=games,
                    seed=seed)


def test_reproducible() -> None:
    a = _duel(123)
    b = _duel(123)
    assert a["results"] == b["results"], (a["results"], b["results"])
    assert a["rounds"] == b["rounds"], (a["rounds"], b["rounds"])
    assert a["seeds"] == b["seeds"]
    c = _duel(124)
    assert a["seeds"] != c["seeds"]
    print(f"  reproducible: results={a['results']}")


def test_timeout_fallback() -> None:
    # 双方都是防御：气数永不变化 → 必然走满兜底 cap。
    ai = BotAgent(bots_by_name["turtle"])
    opp = BotAgent(bots_by_name["turtle"])
    res = run_duel(ai, opp, mode=0, games=1, seed=7)
    assert res["timeouts"] == 1, res
    assert res["rounds"] == [99], res
    assert res["cap"] == 99, res
    print("  timeout fallback: 99 rounds")


def test_encoder_belief_reproducible() -> None:
    model = load_encoder(ENC)
    own = ["馒头", "防御", "馒头", "防御", "馒头", "防御"]
    opp = ["吐气", "馒头", "吐气", "馒头", "吐气", "馒头"]
    legal = get_valid_moves(3, mode=0)

    b1 = EncoderBelief(model, tau=2.0)
    b2 = EncoderBelief(model, tau=2.0)
    for a, o in zip(own, opp):
        b1.observe(a, o)
        b2.observe(a, o)
    w1 = b1.opp_weights(legal, last_own=own[-1])
    w2 = b2.opp_weights(legal, last_own=own[-1])
    assert w1 == w2, (w1, w2)
    assert b1.bigram_tot > 0, "bigram 通道未生效"
    print(f"  encoder mu: {', '.join(f'{k}={v:.3f}' for k, v in w1.items())}")

    # 换一段完全不同的历史，μ 必须变化（序列信息确实进入了决策）。
    b3 = EncoderBelief(model, tau=2.0)
    for _ in range(6):
        b3.observe("防御", "防御")
    w3 = b3.opp_weights(legal, last_own="防御")
    assert w1 != w3
    print("  bigram/history changes mu: yes")


def main() -> None:
    test_reproducible()
    test_timeout_fallback()
    test_encoder_belief_reproducible()
    print("ALL DETERMINISM TESTS PASSED")


if __name__ == "__main__":
    main()
