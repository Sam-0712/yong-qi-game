#!/usr/bin/env python3


import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.game import get_valid_moves, transition, judge, decode_state, IDX_TO_MOVE, MOVES, cfr_lookup
from src.adaptive import CombinedAdaptive
from src.personality import get_personality, apply_bias, PRESETS
from src.db import (
    load_strategy, has_strategy, save_observations, save_global_stats,
    load_observations, load_global_stats, create_session, update_session,
)

MOVE_HELP = {
    "吐  气":   "增加1口气", 
    "馒  头":   "攻击1，防御1，耗气1", 
    "双  馒":   "攻击2，防御2，耗气2",
    "雷  切":   "攻击3，防御3，耗气3，可击中【跳】", 
    "螺旋环":   "攻击4，防御4，耗气4",
    "咏  春":   "攻击5，防御5，耗气5，可击中【跳】", 
    "防  御":   "攻击0，防御3，耗气0",
    "  跳  ":   "攻击0，防御0，耗气0",
    "反  弹":   "攻击0，防御3，耗气0，可反弹【馒头】【双馒】【雷切】的伤害",
}


def main():
    # 模式选择
    mc = input("模式 (1=无反弹, 2=有反弹, 默认 1): ").strip()
    game_mode = 1 if mc == "2" else 0

    # 加载策略
    if has_strategy(game_mode):
        cfr = load_strategy(game_mode)
    else:
        print(f"[!] 数据库中无 mode={game_mode} 策略，请先运行 tests/validate.py")
        return

    # 性格
    names = list(PRESETS.keys())
    print("\n性格:")
    for i, n in enumerate(names, 1):
        print(f"  ({i}) {PRESETS[n].name}")
    pc = int(input("选择 (1-5, 默认 1): ").strip() or "1") - 1
    p = get_personality(names[max(0, min(pc, len(names)-1))])

    # 自适应层
    adp = CombinedAdaptive(
        cross_kw={"temperature": p.cross_temperature, "blend_rate": p.cross_blend_rate,
                   "confidence_k": p.cross_confidence_k},
        within_kw={"temperature": p.within_temperature, "max_blend": p.within_max_blend,
                    "prior_strength": p.within_prior},
    )
    # 加载历史
    for sk, mv in load_observations(game_mode).items():
        for a, c in mv.items():
            adp.cross.pc[sk][a] += c
            adp.cross.sv[sk] += c
    for a, c in load_global_stats(game_mode).items():
        adp.within.counts[a] += c
        adp.within.total += c

    sid = create_session(game_mode, p.name)
    _mode_label = "有反弹" if game_mode else "无反弹"
    print(f"\n{'='*50}")
    print(f"  勇气  —  {p.name}  ({_mode_label})")
    print(f"  输入招式名, h 帮助, q 退出")
    print(f"{'='*50}")

    def sk(state):
        return f"{state[0]},{state[1]},{state[2]},{state[3]},{state[4]}"

    import random
    def sample(probs):
        r, cum = random.random(), 0
        for m, p in probs.items():
            cum += p
            if r <= cum:
                return m
        return list(probs)[-1]

    total = hw = aw = dw = 0
    while True:
        print(f"\n第 {total+1} 局  |  你 {hw} : AI {aw}")
        adp.reset()
        state = (3, 3, 0, 0, 0)
        game_obs = []

        while True:
            qi_a, qi_p, la, lp, r = state
            key = sk(state)
            va, vb = get_valid_moves(qi_a, la), get_valid_moves(qi_p, lp)

            print(f"\n回合 {r+1}  AI气={qi_a}  你气={qi_p}")
            print(f"  你的招式: {', '.join(vb)}")

            # 轮次钳位：超出训练轮次则复用最后一轮策略
            ai_p = cfr_lookup(cfr, state) or {m: 1/len(va) for m in va}
            adj = apply_bias(adp.get_adjusted_strategy(key, ai_p, qi_a, qi_p, la, lp), p)
            ai_m = sample(adj)

            while True:
                cmd = input("  出招: ").strip()
                if cmd == "q":
                    print(f"\n最终: 你 {hw} - AI {aw}")
                    return
                if cmd == "h":
                    for m in vb:
                        print(f"    {m:8s}  {MOVE_HELP.get(m,'')}")
                    continue
                if cmd in vb:
                    break
                print(f"  无效。可选: {', '.join(vb)}")

            adp.observe(key, cmd)
            game_obs.append((key, cmd))
            res = judge(ai_m, cmd)
            print(f"  AI: {ai_m:6s}  你: {cmd:6s}  -> "
                  f"{['你胜','未分胜负','AI胜'][res+1]}")
            if res != 0:
                break
            state = transition(state, ai_m, cmd)

            # 安全阀
            if r > 100:
                print("\n  [!] 对局超过 100 回合，强制结束")
                res = -1
                break

        total += 1
        update_session(sid, res == -1, res == 1)
        if res == 1:   aw += 1
        elif res == -1: hw += 1
        else:          dw += 1

        # 持久化
        obs = {}
        for k, m in game_obs:
            obs.setdefault(k, {})
            obs[k][m] = obs[k].get(m, 0) + 1
        save_observations(obs, game_mode)
        gc = {m: sum(1 for _, x in game_obs if x == m) for m in set(x for _, x in game_obs)}
        save_global_stats(gc, game_mode)

        print(f"\n比分: 你 {hw} : AI {aw} : 平 {dw}")


if __name__ == "__main__":
    main()
