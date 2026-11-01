"""通道混合（指数加权）的正确性：权重怎么动、混合值对不对、装配是否接上。

这些断言守住的是「混合替代硬切换」这个改动的可信度：权重必须按预测似然走、
必须能回到先验、passive 通道不能被重复更新。
"""

from __future__ import annotations

import math
import os
import random
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.agents import OnlineBRAgent
from courage.core.rules import get_valid_moves
from courage.learning.mixture_model import (BeliefChannel, MixtureOpponentModel,
                                            UniformChannel)


class FixedChannel:
    """固定分布的假通道，用来验证聚合逻辑本身。"""

    def __init__(self, name: str, probs: dict):
        self.name = name
        self.probs = probs
        self.seen = 0

    def reset(self) -> None:
        self.seen = 0

    def observe(self, *a) -> None:
        self.seen += 1

    def opp_weights(self, qa, qp, legal, last_own=None):
        return {m: self.probs.get(m, 0.0) for m in legal}


class OracleChannel:
    """永远猜中实际出招——权重应该涨到接近 1。"""

    def __init__(self):
        self.target = None
        self.seen = 0

    def reset(self):
        self.target = None
        self.seen = 0

    def observe(self, qa, qp, own_move, opp_move):
        self.target = opp_move
        self.seen += 1

    def opp_weights(self, qa, qp, legal, last_own=None):
        legal = list(legal)
        p = {m: 0.0 for m in legal}
        if self.target in p:
            p[self.target] = 1.0
        else:
            p[legal[0]] = 1.0
        return p


LEGAL = get_valid_moves(3, mode=0)


def _mix(channels, **kw):
    kw.setdefault("prior_mix", 0.0)
    return MixtureOpponentModel(channels, **kw)


def test_weights_start_at_prior():
    m = _mix({"type": FixedChannel("type", {"吐气": 1.0}),
              "uniform": UniformChannel()}, null_weight=0.5)
    w = m.weights()
    assert w["uniform"] == pytest.approx(0.5)
    assert w["type"] == pytest.approx(0.5)


def test_prior_prefers_null_with_several_channels():
    m = _mix({"a": FixedChannel("a", {"吐气": 1.0}),
              "b": FixedChannel("b", {"馒头": 1.0}),
              "uniform": UniformChannel()}, null_weight=0.5)
    w = m.weights()
    assert w["uniform"] == pytest.approx(0.5)
    assert w["a"] == pytest.approx(0.25)
    assert w["b"] == pytest.approx(0.25)


def test_weights_learn_from_predictive_likelihood():
    oracle = OracleChannel()
    m = _mix({"oracle": oracle, "uniform": UniformChannel()}, null_weight=0.5)
    rng = random.Random(0)
    for _ in range(30):
        move = rng.choice(LEGAL)
        oracle.target = move
        m.observe(3, 3, "防御", move)
    w = m.weights()
    assert w["oracle"] > 0.95, w


def test_uniform_holds_against_pure_noise():
    """对手真随机时，「永远猜吐气」这种通道不该赢过均匀。"""
    stubborn = FixedChannel("stubborn", {"吐气": 1.0})
    m = _mix({"stubborn": stubborn, "uniform": UniformChannel()}, null_weight=0.5)
    rng = random.Random(1)
    for _ in range(200):
        m.observe(3, 3, "防御", rng.choice(LEGAL))
    w = m.weights()
    assert w["uniform"] > w["stubborn"], w


def test_mixture_is_weighted_average():
    a = FixedChannel("a", {"吐气": 1.0})
    b = FixedChannel("b", {"馒头": 1.0})
    m = _mix({"a": a, "b": b})
    m.logw["a"] = math.log(0.3)
    m.logw["b"] = math.log(0.7)
    w = m.weights()
    got = m.opp_weights(3, 3, LEGAL)
    assert got["吐气"] == pytest.approx(w["a"], abs=1e-9)
    assert got["馒头"] == pytest.approx(w["b"], abs=1e-9)
    assert sum(got.values()) == pytest.approx(1.0)


def test_passive_channel_is_not_updated_twice():
    passive = FixedChannel("hand", {"吐气": 1.0})
    active = FixedChannel("type", {"馒头": 1.0})
    m = _mix({"hand": passive, "type": active, "uniform": UniformChannel()},
             passive=("hand",))
    m.observe(3, 3, "防御", "吐气")
    assert passive.seen == 0, "passive 通道被重复 observe，会污染它自己的统计"
    assert active.seen == 1


def test_reset_restores_prior_weights():
    m = _mix({"a": FixedChannel("a", {"吐气": 1.0}),
              "uniform": UniformChannel()}, null_weight=0.5)
    m.observe(3, 3, "防御", "吐气")
    assert m.weights()["a"] != pytest.approx(0.5)
    m.reset()
    assert m.weights()["a"] == pytest.approx(0.5)
    assert m.steps == 0


def test_belief_channel_adapts_tracker_signature():
    from courage.learning.belief import BeliefTracker

    bt = BeliefTracker()
    ch = BeliefChannel(bt)
    out = ch.opp_weights(3, 3, LEGAL, last_own=None)
    assert set(out) == set(LEGAL)
    assert sum(out.values()) == pytest.approx(1.0)


def test_agent_installs_mixture_channels():
    """gate="mix" 时，agent 上的对手模型应被换成混合通道且能被 trace 展开。"""
    from courage.learning.type_model import OpponentTypeModel

    ag = OnlineBRAgent(gate="mix", type_model=OpponentTypeModel(mode=0))
    assert hasattr(ag.type_model, "breakdown")
    assert ag.type_model.names == ["type", "uniform"]
    w, dists = ag.type_model.breakdown(3, 3, get_valid_moves(3))
    assert set(w) == {"type", "uniform"}
    assert set(dists) == {"type", "uniform"}


def test_agent_mix_gate_is_deterministic_per_seed():
    """同一 seed ⇒ 同一串决策（混合通道不能引入额外的随机性）。"""
    from courage.learning.type_model import OpponentTypeModel

    def run(seed: int):
        ag = OnlineBRAgent(gate="mix", type_model=OpponentTypeModel(mode=0),
                           rng=random.Random(seed))
        qa = qp = 3
        out = []
        for _ in range(6):
            a = ag.act(qa, qp)
            b = "防御"
            out.append(a)
            ag.observe(qa, qp, a, b)
            qa, qp = max(0, qa - (1 if a in ("馒头", "双馒") else 0)), qp
        return out

    assert run(7) == run(7)
