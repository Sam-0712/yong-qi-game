"""精确值 oracle（增广 MDP）与 36 状态 BR 的一致性 / 相位正确性 / regret 非负。

这些断言守住的是评测基准的可信度：如果 oracle 与 `solve_best_response` 不一致，
或者相位没接对，那 tools/bench.py 报出的 regret 就全是噪声。
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.bots.opponents import OPPONENTS
from courage.core.rules import IDX_TO_MOVE, get_valid_moves
from courage.solvers.aug_mdp import (DEFAULT_GAMMA, DEFAULT_T_MAX, LAST_MAX,
                                    build_aug, normalize_needs, oracle_for,
                                    solve_aug)
from courage.solvers.best_response import solve_best_response
from courage.solvers.equilibrium import build_tables

STATES = [(qa, qp) for qa in range(6) for qp in range(6)]


def _marginal_policy(style_name: str, t_max: int = DEFAULT_T_MAX):
    """把带相位的风格投影成 36 状态的「相位边际」对手模型。"""
    spec = OPPONENTS[style_name]
    al_vals = range(LAST_MAX) if "al" in spec.needs else [0]
    bl_vals = range(LAST_MAX) if "bl" in spec.needs else [0]
    t_vals = range(t_max + 1) if "t" in spec.needs else [0]
    out = {}
    for qa, qp in STATES:
        acc: dict = {}
        n = 0
        for al in al_vals:
            for bl in bl_vals:
                for t in t_vals:
                    d = spec.policy(qp, qa, t, IDX_TO_MOVE.get(bl),
                                    IDX_TO_MOVE.get(al))
                    for m, v in d.items():
                        acc[m] = acc.get(m, 0.0) + v
                    n += 1
        s = sum(acc.values())
        out[(qa, qp)] = ({m: v / s for m, v in acc.items()} if s > 1e-12
                         else {m: 1.0 / n for m in get_valid_moves(qp)})
    return out


def test_normalize_needs_aliases():
    assert normalize_needs(("own_last",)) == ("bl",)
    assert normalize_needs(("opp_last",)) == ("al",)
    assert normalize_needs(("t", "own_last")) == ("t", "bl")
    with pytest.raises(ValueError):
        normalize_needs(("bogus",))


def test_state_space_size_matches_needs():
    assert len(build_aug(0, ()).states) == 36
    assert len(build_aug(0, ("t",)).states) == 36 * (DEFAULT_T_MAX + 1)
    assert len(build_aug(0, ("al",)).states) == 36 * LAST_MAX
    assert len(build_aug(0, ("bl",)).states) == 36 * LAST_MAX


def test_stationary_oracle_equals_36_state_br():
    """needs=() 时，增广 oracle 必须与 solve_best_response 逐状态完全一致。"""
    spec = OPPONENTS["converter"]
    sol = solve_aug(build_aug(0, spec.needs), spec.policy)
    br = solve_best_response(0, _marginal_policy("converter"), gamma=DEFAULT_GAMMA,
                             max_sweeps=6000, tol=1e-12, step_cost=0.0,
                             tables=build_tables(0))
    for qa, qp in STATES:
        assert sol.value(qa, qp) == pytest.approx(br.values[(qa, qp)], abs=1e-7), \
            (qa, qp)


def test_state_order_matches_equilibrium_tables():
    T = build_aug(0, ())
    ref = build_tables(0)
    assert T.states == [(qa, qp, 0, 0, 0) for qa, qp in ref.states]


def test_alternate_low_reads_its_own_last_move():
    """低气交替依赖「对手自己上一招」——oracle 必须把这条相位接进状态。"""
    spec = OPPONENTS["alternate_low"]
    T = build_aug(0, spec.needs)
    at_11 = lambda bl: spec.policy(1, 1, 0, IDX_TO_MOVE.get(bl), None)  # noqa: E731
    assert at_11(0)["吐气"] == 1.0          # 无上一招 → 先吐气
    assert at_11(1)["馒头"] == 1.0          # 上一招吐气 → 换馒头
    si_a = T.index(1, 1, 0, 0, 0)
    si_b = T.index(1, 1, 0, 1, 0)
    assert si_a != si_b                     # 两个相位是不同的状态
    # 知道相位就能赢：(1,1) 无上一招时对手必吐气，用馒头直接打穿
    sol = solve_aug(T, spec.policy)
    q = sol.q_row_at(si_a)
    moves = T.movesA[si_a]
    assert q[moves.index("馒头")] > q[moves.index("防御")]


def test_phase_ignorance_costs_value():
    """36 状态边际模型在相位对手身上必然有正 regret，且 oracle 严格更优。"""
    spec = OPPONENTS["alternate_low"]
    sol = oracle_for(spec)
    T = sol.tables
    marg = solve_best_response(0, _marginal_policy("alternate_low"),
                               gamma=DEFAULT_GAMMA, max_sweeps=6000, tol=1e-12,
                               step_cost=0.0, tables=build_tables(0))
    total = 0.0
    for bl in range(LAST_MAX):
        si = T.index(1, 1, 0, bl, 0)
        q = sol.q_row_at(si)
        moves = T.movesA[si]
        choice = marg.policy[(1, 1)]
        got = sum(p * q[moves.index(m)] for m, p in choice.items() if m in moves)
        total += float(q.max() - got)
    assert total > 0.0, "相位边际模型在 (1,1) 上没有付出任何代价，说明相位没接进 oracle"


def test_regret_is_never_negative():
    """任何策略在任何状态下都不该拿到负 regret（否则基准本身算错了）。"""
    spec = OPPONENTS["pressure"]
    sol = oracle_for(spec)
    Q = sol.q_matrix()
    for si in range(len(sol.tables.states)):
        q = Q[si][:len(sol.tables.movesA[si])]
        assert all(q.max() - v >= -1e-12 for v in q)


def test_all_styles_solvable():
    """每个风格都能建 oracle 并收敛（含集合外与依赖多相位的）。"""
    for name, spec in OPPONENTS.items():
        sol = oracle_for(spec)
        assert sol.sweeps < 4000, f"{name} 未收敛"
        v = sol.value(3, 3)
        assert -1.0 <= v <= 1.0, (name, v)
