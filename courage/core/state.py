"""最小信息集的状态抽象。

旧五元组 S = (qa, qp, la, lp, r) 中的「上一招」是冗余的：
  - get_valid_moves 不使用 last_move_idx
  - judge / transition 均不依赖历史
故游戏是马尔可夫博弈，真实状态只需 (qa, qp, mode[, r])。

此外：
  - 零和对称：交换双方有 V(qp,qa) = -V(qa,qp)；(q,q) 时 V = 0
  - 回合 r 只在折扣/终局截断处有意义；平稳解可令 r=None

本模块是 B/C 重构的目标状态表示；旧代码仍使用 5 元组（见 core.rules）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

QI_MAX = 5
MODE_STANDARD = 0   # 标准模式，无「无气拨」
MODE_NO_QI = 1      # 无气拨模式


@dataclass(frozen=True, slots=True)
class State:
    qa: int
    qp: int
    mode: int = MODE_STANDARD
    r: Optional[int] = None

    def key(self) -> Tuple[int, int, int, int]:
        return (self.qa, self.qp, self.mode, -1 if self.r is None else self.r)

    def is_symmetric(self) -> bool:
        return self.qa == self.qp

    def canon(self) -> Tuple["State", bool]:
        """返回 (规范态, 是否发生交换)。用于利用零和对称压缩价值表。"""
        if self.qa >= self.qp:
            return self, False
        return State(self.qp, self.qa, self.mode, self.r), True

    def to_legacy(self) -> Tuple[int, int, int, int, int]:
        """转旧五元组（上一招置 0），仅供与旧模块互操作。"""
        return (self.qa, self.qp, 0, 0, 0 if self.r is None else self.r)


def iter_states(mode: int = MODE_STANDARD,
                horizon: Optional[int] = None,
                symmetric_only: bool = False) -> Iterator[State]:
    """枚举状态。

    horizon=None：平稳解，r 不进状态（只枚举 qa、qp）。
    horizon=H   ：有限期，枚举 r ∈ [0, H)。
    symmetric_only：只枚举 qa>=qp（利用对称）。
    """
    rounds = [None] if horizon is None else list(range(horizon))
    for qa in range(QI_MAX + 1):
        for qp in range(QI_MAX + 1):
            if symmetric_only and qa < qp:
                continue
            for r in rounds:
                yield State(qa, qp, mode, r)


def init_state(mode: int = MODE_STANDARD, qi: int = 3,
               r: Optional[int] = 0) -> State:
    return State(qi, qi, mode, r)
