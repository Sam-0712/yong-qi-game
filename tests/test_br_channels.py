"""多通道风险 BR 的代数性质。

最重要的一条是**线性恒等式**：κ=0 时「先按权重混合分布再求 BR」与
「逐通道求 Q 再加权」必须逐位一致。它保证：
  1. 通道版 BR 在 κ=0 时是旧实现的 drop-in（不会因为改成通道版而悄悄改变行为）；
  2. 后面所有 κ>0 的实验，都是在这个精确基线上做加法。
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.bots.opponents import OPPONENTS
from courage.solvers.best_response import solve_best_response
from courage.solvers.br_channels import solve_channel_br
from courage.solvers.equilibrium import build_tables

STATES = [(qa, qp) for qa in range(6) for qp in range(6)]
T = build_tables(0)


def _policy(name: str):
    spec = OPPONENTS[name]
    return {(qa, qp): spec.policy(qp, qa) for qa, qp in STATES}


def _mix_policies(pols, weights):
    out = {}
    for (qa, qp) in STATES:
        acc = {}
        for c, p in pols.items():
            for m, v in p[(qa, qp)].items():
                acc[m] = acc.get(m, 0.0) + weights[c] * v
        s = sum(acc.values())
        out[(qa, qp)] = {m: v / s for m, v in acc.items()} if s > 1e-12 else acc
    return out


def test_kappa_zero_equals_mixing_distributions_first():
    """κ=0 ⇒ Q 通道加权 == 对混合分布的 Q（线性恒等式）。"""
    pols = {"a": _policy("converter"), "b": _policy("pressure")}
    w = {"a": 0.3, "b": 0.7}
    res = solve_channel_br(0, pols, weights=w, kappa=0.0, max_sweeps=6000,
                           tol=1e-12, tables=T)
    ref = solve_best_response(0, _mix_policies(pols, w), max_sweeps=6000,
                              tol=1e-12, step_cost=0.0, tables=T)
    for st in STATES:
        assert res.values[st] == pytest.approx(ref.values[st], abs=1e-7), st
        for m, q in ref.q[st].items():
            assert res.q_mean[st][m] == pytest.approx(q, abs=1e-7), (st, m)


def test_single_channel_ignores_kappa():
    """只有一条通道时，min 就是它自己 ⇒ 任何 κ 都等价于普通 BR。"""
    pols = {"only": _policy("converter")}
    for kappa in (0.0, 0.5, 1.0):
        res = solve_channel_br(0, pols, weights={"only": 1.0}, kappa=kappa,
                               max_sweeps=6000, tol=1e-12, tables=T)
        ref = solve_best_response(0, pols["only"], max_sweeps=6000, tol=1e-12,
                                  step_cost=0.0, tables=T)
        for st in STATES:
            assert res.values[st] == pytest.approx(ref.values[st], abs=1e-7), st


def test_kappa_one_is_worst_case_over_channels():
    pols = {"a": _policy("converter"), "b": _policy("pressure")}
    w = {"a": 0.5, "b": 0.5}
    res = solve_channel_br(0, pols, weights=w, kappa=1.0, max_sweeps=6000,
                           tol=1e-12, tables=T)
    for st in STATES:
        for m, v in res.q_lo[st].items():
            assert res.q_risk[st][m] == pytest.approx(v, abs=1e-9), (st, m)


def test_spread_is_nonnegative_and_bounded():
    pols = {"a": _policy("converter"), "b": _policy("strong"),
            "c": _policy("pressure")}
    res = solve_channel_br(0, pols, kappa=0.0, max_sweeps=6000, tol=1e-12,
                           tables=T)
    for st in STATES:
        for m, sp in res.spread(*st).items():
            assert -1e-9 <= sp <= 2.0 + 1e-9, (st, m, sp)


def test_identical_channels_have_zero_spread_and_no_flip():
    p = _policy("converter")
    res = solve_channel_br(0, {"a": p, "b": p}, kappa=0.4, max_sweeps=6000,
                           tol=1e-12, tables=T)
    for st in STATES:
        assert all(abs(v) < 1e-9 for v in res.spread(*st).values()), st
        assert not res.flip(*st), st


def test_flip_flag_matches_argmax_comparison():
    pols = {"a": _policy("converter"), "b": _policy("defensive")}
    res = solve_channel_br(0, pols, weights={"a": 0.5, "b": 0.5}, kappa=0.5,
                           max_sweeps=6000, tol=1e-12, tables=T)
    for st in STATES:
        am = max(res.q_mean[st], key=lambda k: res.q_mean[st][k])
        ar = max(res.q_risk[st], key=lambda k: res.q_risk[st][k])
        assert res.flip(*st) == (am != ar), st


def test_zero_weights_fall_back_to_uniform_channel_weights():
    pols = {"a": _policy("converter"), "b": _policy("pressure")}
    res = solve_channel_br(0, pols, weights={"a": 0.0, "b": 0.0}, kappa=0.0,
                           max_sweeps=300, tables=T)
    assert res.weights["a"] == pytest.approx(0.5)
    assert res.weights["b"] == pytest.approx(0.5)


def test_policy_is_a_distribution_over_legal_moves():
    pols = {"a": _policy("converter"), "b": _policy("strong")}
    res = solve_channel_br(0, pols, kappa=0.3, max_sweeps=300, tables=T)
    for st in STATES:
        p = res.policy[st]
        assert abs(sum(p.values()) - 1.0) < 1e-9
        assert set(p) <= set(T.movesA[T.states.index(st)])
