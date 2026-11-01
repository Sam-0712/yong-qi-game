"""风格（StyleProfile）与「风格对手」的可信度：类别偏好真的被尊重、可复现、能进 oracle。

守住这件事的意义：风格是**参数族**（低维、可采样、只依赖状态），
和打法（离散脚本、依赖相位）不是一回事。如果 profile_policy 不真的尊重
类别权重，那"风格 zoo"就退化成又一个随机 bot，分布外测验就没有意义。
"""

from __future__ import annotations

import os
import random
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.bots.opponents import OPPONENTS, TYPE_SET, names_by_kind, KIND_STYLE
from courage.bots.style_profiles import (BALANCED, CLASSIC, STYLE_DIMS,
                                         StyleOpponent, StyleOpponentAgent,
                                         StyleProfile, profile_policy,
                                         sample_profile, sample_zoo)
from courage.core.rules import MOVE_CATEGORY, get_valid_moves
from courage.solvers.aug_mdp import oracle_for


def test_every_profile_dimension_is_a_real_category():
    assert set(STYLE_DIMS) <= set(MOVE_CATEGORY.values())


def test_pure_defense_profile_only_plays_defense():
    prof = StyleProfile(attack=0.0, heavy=0.0, defense=1.0, dodge=0.0, exhale=0.0)
    for q in range(6):
        p = profile_policy(prof, q)
        legal = get_valid_moves(q)
        assert set(p) == set(legal)
        nonzero = [m for m, v in p.items() if v > 1e-9]
        assert nonzero == ["防御"], (q, p)


def test_category_weights_are_not_distorted_by_legal_set_size():
    """「六成进攻」必须真的是六成——不能因为进攻有三招、防御只有一招就偏过去。"""
    prof = StyleProfile(attack=3.0, heavy=0.0, defense=2.0, dodge=0.0, exhale=0.0)
    p = profile_policy(prof, 3)
    by_cat = {}
    for m, v in p.items():
        by_cat[MOVE_CATEGORY[m]] = by_cat.get(MOVE_CATEGORY[m], 0.0) + v
    assert by_cat["attack"] == pytest.approx(0.6, abs=1e-9)
    assert by_cat["defense"] == pytest.approx(0.4, abs=1e-9)
    assert abs(sum(p.values()) - 1.0) < 1e-9


def test_profile_with_no_legal_category_falls_back_to_uniform():
    """0 气时进攻/重攻都不合法——不能因此输出空分布。"""
    prof = StyleProfile(attack=1.0, heavy=1.0, defense=0.0, dodge=0.0, exhale=0.0)
    p = profile_policy(prof, 0)
    legal = get_valid_moves(0)
    assert set(p) == set(legal)
    assert sum(p.values()) == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0 / len(legal)) for v in p.values())


def test_normalized_and_tilted_keep_distribution_valid():
    prof = StyleProfile(attack=5.0, heavy=0.1, defense=2.0, dodge=0.0, exhale=1.0)
    assert sum(prof.normalized().values()) == pytest.approx(1.0)
    sharp = prof.tilted(3.0)
    assert sum(sharp.normalized().values()) == pytest.approx(1.0)
    # 锐化后最大维度不应变小
    a = max(prof.normalized().values())
    b = max(sharp.normalized().values())
    assert b >= a - 1e-12


def test_sample_profile_is_reproducible_and_normalized():
    a = sample_profile(random.Random(42))
    b = sample_profile(random.Random(42))
    assert a == b
    assert sum(a.normalized().values()) == pytest.approx(1.0)
    assert all(v >= 0 for v in a.as_dict().values())


def test_sample_zoo_is_seeded_and_distinct():
    z1 = sample_zoo(5, seed=7)
    z2 = sample_zoo(5, seed=7)
    assert z1 == z2
    assert len({p.key() for p in z1}) > 1, "采样出的风格不该全一样"


def test_style_opponent_has_no_phase_dependency():
    """风格的定义要求「不看历史」：needs 必须为空，oracle 只需 36 格。"""
    spec = StyleOpponent(CLASSIC["style_guard"])
    assert tuple(spec.needs) == ()
    T = oracle_for(spec).tables
    assert len(T.states) == 36


def test_style_opponent_policy_matches_profile_policy():
    prof = CLASSIC["style_press"]
    spec = StyleOpponent(prof)
    for q in range(4):
        assert spec.policy(q, q, 0, "防御", "馒头") == profile_policy(prof, q, q)


def test_style_opponent_agent_acts_legally_and_deterministically():
    prof = CLASSIC["style_evasive"]
    a = StyleOpponentAgent(prof, random.Random(3))
    b = StyleOpponentAgent(prof, random.Random(3))
    seq_a = [a.act(3, 3) for _ in range(20)]
    seq_b = [b.act(3, 3) for _ in range(20)]
    assert seq_a == seq_b
    assert all(m in get_valid_moves(3) for m in seq_a)


def test_kind_taxonomy_is_populated():
    """打法库里要明确标出哪些是纯风格、哪些是有思路的定势。"""
    assert "turtle" in names_by_kind(KIND_STYLE)
    assert "exploit4" in OPPONENTS and OPPONENTS["exploit4"].kind != KIND_STYLE
    # 依赖相位的打法，needs 不能为空
    assert OPPONENTS["exploit4"].needs == ("t",)
    assert OPPONENTS["alternate_low"].needs == ("own_last",)
    assert OPPONENTS["mirror"].needs == ("opp_last",)
    # 类型后验的集合与打法库必须对齐
    assert set(TYPE_SET) <= set(OPPONENTS)


def test_balanced_profile_is_not_identical_to_uniform():
    """均匀风格 ≠ 均匀招式：类别内均分会按各类合法招数摊开，这点要显性化。"""
    p = profile_policy(BALANCED, 3)
    legal = get_valid_moves(3)
    assert abs(sum(p.values()) - 1.0) < 1e-9
    assert set(p) == set(legal)
    # 3 气时进攻类有 3 招、其余类各 1 招 ⇒ 单招概率必然不等
    assert max(p.values()) - min(p.values()) > 0.05
    assert not all(abs(v - 1.0 / len(legal)) < 1e-9 for v in p.values())
