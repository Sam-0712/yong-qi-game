"""
自对抗测试
"""

import sys, os, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.game import get_valid_moves, transition, judge, MOVES_ATTR, cfr_lookup
from src.adaptive import CombinedAdaptive, AdaptiveLayer, WithinGameAdaptive
from src.db import load_strategy

# 加载策略
CFR = load_strategy(mode=0)

# 对手
def bot_random(s):
    return random.choice(get_valid_moves(s[1], s[3]))

def bot_aggressive(s):
    va = get_valid_moves(s[1], s[3])
    atk = [m for m in va if MOVES_ATTR[m][1] > 0]
    return max(atk, key=lambda m: MOVES_ATTR[m][1]) if atk and random.random() < 0.7 else random.choice(va)

def bot_defensive(s):
    va = get_valid_moves(s[1], s[3])
    safe = [m for m in va if m in ("吐气", "防御")]
    return random.choice(safe) if safe and random.random() < 0.6 else random.choice(va)

BOTS = {"random": bot_random, "aggressive": bot_aggressive, "defensive": bot_defensive}

def sk(state):
    return f"{state[0]},{state[1]},{state[2]},{state[3]},{state[4]}"

def sample(probs):
    r = random.random()
    cum = 0
    for m, p in probs.items():
        cum += p
        if r <= cum:
            return m
    return list(probs)[-1]

def play_game(ai_mode, opponent, adaptive=None):
    state = (3, 3, 0, 0, 0)
    for _ in range(300):  # 安全阀，实际对局不应达到
        qi_a, qi_p, la, lp, r = state
        key = sk(state)
        va = get_valid_moves(qi_a, la)
        cfr_p = cfr_lookup(CFR, state) or {m: 1.0/len(va) for m in va}

        if ai_mode == "cfr_only":
            ai_move = sample(cfr_p)
        elif ai_mode == "cfr_cross":
            ai_move = sample(adaptive.get_adjusted_strategy(key, cfr_p, qi_a, qi_p, la, lp))
        elif ai_mode == "cfr_within":
            ai_move = sample(adaptive.get_adjusted_strategy(cfr_p, qi_a, qi_p, la, lp))
        else:
            ai_move = sample(adaptive.get_adjusted_strategy(key, cfr_p, qi_a, qi_p, la, lp))

        pl_move = BOTS[opponent](state)
        if adaptive is not None:
            if ai_mode == "cfr_within":
                adaptive.observe(pl_move)
            else:
                adaptive.observe(key, pl_move)

        result = judge(ai_move, pl_move)
        if result != 0:
            return result
        state = transition(state, ai_move, pl_move)
    return -1  # 安全阀触发 → 玩家胜

def run_test(ai_mode, opponent, n=300):
    adaptive = None
    if ai_mode == "cfr_cross":
        adaptive = AdaptiveLayer()
    elif ai_mode == "cfr_within":
        adaptive = WithinGameAdaptive()
    elif ai_mode == "cfr_combined":
        adaptive = CombinedAdaptive()

    w = l = d = 0
    for i in range(n):
        if ai_mode in ("cfr_within", "cfr_combined"):
            adaptive.reset()
        res = play_game(ai_mode, opponent, adaptive)
        if res == 1:    w += 1
        elif res == -1: l += 1
        else:           d += 1
    return w, l, d


print("=" * 56)
print("  自对抗: 纯CFR | 跨局 | 局内 | 融合")
print("=" * 56)

for opp in ["random", "aggressive", "defensive"]:
    print(f"\n── {opp} ──")
    base = run_test("cfr_only", opp)
    wr0 = base[0] / sum(base)
    print(f"  纯CFR:      {base[0]:>3}胜 {base[1]:>3}负 {base[2]:>3}平  {wr0:.1%}")

    for mode, label in [("cfr_cross", "跨局"), ("cfr_within", "局内"), ("cfr_combined", "融合")]:
        r = run_test(mode, opp)
        wr = r[0] / sum(r)
        print(f"  +{label}:  {r[0]:>3}胜 {r[1]:>3}负 {r[2]:>3}平  {wr:.1%}  Δ={wr-wr0:+.1%}")
