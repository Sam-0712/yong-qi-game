"""核心：游戏规则与状态抽象。"""

from .rules import (
    MOVES, MOVES_ATTR, MOVE_TO_IDX, IDX_TO_MOVE,
    HIT_JUMP, MAX_QI, MAX_ROUNDS,
    judge, transition, get_valid_moves, decode_state,
    state_hash, hash_to_state, cfr_lookup,
    build_state_graph, state_summary,
)
from .state import (
    State, QI_MAX, MODE_STANDARD, MODE_NO_QI,
    iter_states, init_state,
)

__all__ = [
    "MOVES", "MOVES_ATTR", "MOVE_TO_IDX", "IDX_TO_MOVE",
    "HIT_JUMP", "MAX_QI", "MAX_ROUNDS",
    "judge", "transition", "get_valid_moves", "decode_state",
    "state_hash", "hash_to_state", "cfr_lookup",
    "build_state_graph", "state_summary",
    "State", "QI_MAX", "MODE_STANDARD", "MODE_NO_QI",
    "iter_states", "init_state",
]
