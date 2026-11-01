"""item 2：奖励加权因子目标（用胜负重采样实现 RWR）。

思路：先用「我方(均衡表) vs 机器人」生成带胜负的轨迹；训练编码器时按
exp(beta*result) 重采样，让「赢的轨迹」在因子回归里占更大权重。
评估看**决策**（编码器因子喂 BR 的胜率），而不是 RMSE。
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from courage.agents import BotAgent, OnlineBRAgent, TableAgent, run_duel
from courage.bots.scripts import bots_by_name, sample_policy, ALL_BOTS
from courage.core.rules import get_valid_moves, judge, next_qi
from courage.learning.seq_encoder import (GRUEncoder, EncoderBelief,
                                          sequence_to_xy, load_encoder)
from courage.storage.artifacts import load_policy_table

TK = {"alpha": 0.20, "smooth": 0.05}


def gen_outcomes(agent, bot, episodes, cap, mode, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(episodes):
        qa = qp = 3
        own, opp = [], []
        res = 0
        for _ in range(cap):
            a = agent.act(qa, qp)
            b = sample_policy(bot.policy(qp, qa, mode=mode,
                                         last_opp=own[-1] if own else None), rng)
            own.append(a); opp.append(b)
            w = judge(a, b)
            if w != 0:
                res = w
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        if len(own) >= 2:
            out.append((own, opp, res))
    return out


def train_gru(data, epochs, seed, weighted, beta, gh=24, lr=0.01):
    Xs = [sequence_to_xy(o, p, TK) for (o, p, _) in data]
    w = np.array([np.exp(beta * r) for (_, _, r) in data])
    w = w / w.sum()
    m = GRUEncoder(hidden=gh, seed=seed, lr=lr)
    rng = random.Random(seed)
    n = len(Xs)
    for ep in range(epochs):
        if weighted:
            idx = rng.choices(range(n), weights=w.tolist(), k=n)
        else:
            idx = list(range(n))
            rng.shuffle(idx)
        for i in idx:
            X, Y = Xs[i]
            m.fit_step(X, Y)
    return m


def eval_encoder(model, opp_names, games, cap):
    res = {}
    for name in opp_names:
        ai = OnlineBRAgent(step_cost=0.03, br_sweeps=200,
                           belief_like=EncoderBelief(model, tau=2.0))
        res[name] = run_duel(ai, BotAgent(bots_by_name[name]), mode=0,
                             games=games, cap=cap)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--cap", type=int, default=99,
                    help="兜底回合上限（不作为产品结算上限）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pol, _ = load_policy_table("policy_mode0.json")
    if not pol:
        from courage.solvers.equilibrium import solve_equilibrium
        pol = solve_equilibrium(mode=0, gamma=0.95, iters=3000, tol=1e-4).policy
    agent = TableAgent(pol, mode=0, rng=random.Random(args.seed))

    data = []
    rng = random.Random(args.seed)
    for _ in range(args.episodes // 60 + 1):
        name = rng.choice(list(bots_by_name))
        data += gen_outcomes(agent, bots_by_name[name], 60, args.cap, 0, args.seed + len(data))
    data = data[:args.episodes]
    wins = sum(1 for (_, _, r) in data if r == 1)
    print(f"轨迹 {len(data)} 条, 我方胜 {wins} 负 {sum(1 for (_,_,r) in data if r==-1)}")

    m_uni = train_gru(data, args.epochs, args.seed, weighted=False, beta=args.beta)
    m_rw = train_gru(data, args.epochs, args.seed, weighted=True, beta=args.beta)

    opps = ["turtle", "defensive", "aggressive", "random"]
    r_uni = eval_encoder(m_uni, opps, args.games, args.cap)
    r_rw = eval_encoder(m_rw, opps, args.games, args.cap)
    print(f"{'对手':<10}{'均匀蒸馏':>22}{'奖励加权':>22}")
    for name in opps:
        a, b = r_uni[name], r_rw[name]
        print(f"{name:<10}"
              f"{a['wins']/a['games']:>10.0%}/{statistics.mean(a['rounds']):>4.1f}r"
              f"{b['wins']/b['games']:>14.0%}/{statistics.mean(b['rounds']):>4.1f}r")


if __name__ == "__main__":
    main()
