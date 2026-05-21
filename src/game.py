"""
核心游戏引擎
"""

from typing import Tuple, List, Dict, Union

MOVES: List[str] = [
    "吐气",
    "馒头",
    "双馒",
    "雷切",
    "螺旋环",
    "咏春",
    "防御",
    "跳",
]

MOVES_ATTR: Dict[str, List[int]] = {
    "吐气":   [0, 0, 0],
    "馒头":   [1, 1, 1],
    "双馒":   [2, 2, 2],
    "雷切":   [3, 3, 3],
    "螺旋环": [4, 4, 4],
    "咏春":   [5, 5, 5],
    "防御":   [0, 0, 3],
    "跳":     [0, 0, 0],
}

HIT_JUMP: set = {"雷切", "咏春"}
MAX_QI: int = 5
MAX_ROUNDS: int = 15
MOVE_TO_IDX: Dict[str, int] = {m: i + 1 for i, m in enumerate(MOVES)}
IDX_TO_MOVE: Dict[int, str] = {i + 1: m for i, m in enumerate(MOVES)}
ATTACK_MOVES: set = {"馒头", "双馒", "雷切", "螺旋环", "咏春"}

State = Tuple[int, int, int, int, int]
Outcome = Union[State, Tuple[str, int]]


def state_hash(state: State) -> int:
    """状态 → 20 位整数哈希 (用于 DB 主键和快速查找)。"""
    qi_a, qi_p, la, lp, r = state
    return (qi_a << 17) | (qi_p << 14) | (la << 10) | (lp << 6) | r


def hash_to_state(h: int) -> State:
    return ((h >> 17) & 0x7, (h >> 14) & 0x7,
            (h >> 10) & 0xF, (h >> 6) & 0xF, h & 0x3F)


def get_valid_moves(qi: int, last_move_idx: int) -> List[str]:
    valid = []
    for name, attr in MOVES_ATTR.items():
        if attr[0] <= qi:
            valid.append(name)
    return valid


def judge(m1: str, m2: str) -> int:
    a1, a2 = MOVES_ATTR[m1], MOVES_ATTR[m2]

    win1 = False
    if a1[1] > 0:
        if m2 == "跳":
            if m1 in HIT_JUMP:
                win1 = True
        elif a1[1] > a2[2]:
            win1 = True

    win2 = False
    if a2[1] > 0:
        if m1 == "跳":
            if m2 in HIT_JUMP:
                win2 = True
        elif a2[1] > a1[2]:
            win2 = True

    if win1 and not win2:
        return 1
    if win2 and not win1:
        return -1
    return 0


def transition(state: State, m_ai: str, m_pl: str) -> Outcome:
    qi_ai, qi_pl, last_ai, last_pl, rnd = state
    winner = judge(m_ai, m_pl)
    if winner != 0:
        return ('terminal', winner)

    qa = max(0, min(qi_ai - MOVES_ATTR[m_ai][0] + (1 if m_ai == "吐气" else 0), MAX_QI))
    qp = max(0, min(qi_pl - MOVES_ATTR[m_pl][0] + (1 if m_pl == "吐气" else 0), MAX_QI))
    return (qa, qp, MOVE_TO_IDX[m_ai], MOVE_TO_IDX[m_pl], rnd + 1)


def decode_state(state: State) -> str:
    qi_a, qi_p, la, lp, r = state
    sa = IDX_TO_MOVE.get(la, "初始")
    sp = IDX_TO_MOVE.get(lp, "初始")
    return f"[R{r}] AI{qi_a}玩{qi_p} 上{sa}上{sp}"


# ── CFR 策略查找（轮次钳位）──

def cfr_lookup(cfr_dict: dict, state: State) -> dict:
    """
    从 CFR 策略表中查找当前状态的策略。
    若轮次 >= MAX_ROUNDS，钳位到 MAX_ROUNDS-1（训练截断处）。

    返回: {招式: 概率} 或 None（未找到时回退到均匀随机）。
    """
    qi_a, qi_p, la, lp, r = state
    if r >= MAX_ROUNDS:
        r = MAX_ROUNDS - 1
    key = f"{qi_a},{qi_p},{la},{lp},{r}"
    return cfr_dict.get(key)


# ── 状态图构建 ──

def build_state_graph(start_qi_ai: int = 3, start_qi_pl: int = 3):
    """BFS 构建所有可达状态的有向图。"""
    start = (start_qi_ai, start_qi_pl, 0, 0, 0)
    graph, queue, visited = {}, [start], {start}

    while queue:
        st = queue.pop(0)
        qi_a, qi_p, la, lp, r = st
        if r >= MAX_ROUNDS:
            graph[st] = {'valid_ai': [], 'valid_pl': [],
                         'transitions': {}, 'terminal_value': 0}
            continue

        va, vb = get_valid_moves(qi_a, la), get_valid_moves(qi_p, lp)
        trans = {}
        for ma in va:
            for mb in vb:
                out = transition(st, ma, mb)
                if not (out[0] == 'terminal') and out not in visited:
                    visited.add(out)
                    queue.append(out)
                trans[(ma, mb)] = out
        graph[st] = {'valid_ai': va, 'valid_pl': vb, 'transitions': trans}
    return graph, start


def state_summary(graph):
    n = len(graph)
    nt = sum(1 for info in graph.values()
             if info.get('terminal_value') is not None or len(info['transitions']) == 0)
    np = sum(len(info['transitions']) for info in graph.values())
    print(f"状态总数: {n}  终局: {nt}  招式对: {np}")
