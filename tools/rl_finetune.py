"""用胜负结果微调策略头（frozen 编码器产因子），并与解析 BR 对照。

用法：python -m tools.rl_finetune --encoder-file data/artifacts/encoder_gru.json \
        --episodes 400 --games 20
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

from courage.agents import BotAgent, OnlineBRAgent, run_duel
from courage.bots.scripts import bots_by_name, sample_policy
from courage.core.rules import MOVES, get_valid_moves, judge, next_qi
from courage.learning.policy_head import PolicyHead, featurize
from courage.learning.seq_encoder import EncoderBelief, load_encoder


def _mask(legal):
    return np.array([1.0 if m in legal else 0.0 for m in MOVES])


class HeadAgent:
    def __init__(self, head, model, mode=0, rng=None):
        self.head = head
        self.eb = EncoderBelief(model)
        self.mode = mode
        self.rng = rng or random.Random()

    def act(self, own, opp):
        f = self.eb.factors()
        x = featurize(f, own, opp)
        m = _mask(get_valid_moves(own, mode=self.mode))
        i, _ = self.head.act(x, m, self.rng)
        return MOVES[i]

    def observe(self, own, opp, om, pm):
        self.eb.observe(om, pm)

    def reset(self):
        self.eb.reset()


def train(head, model, episodes, cap, mode, seed, pool):
    rng = random.Random(seed)
    baseline, beta = 0.0, 0.05
    for _ep in range(episodes):
        eb = EncoderBelief(model)
        opp = bots_by_name[rng.choice(pool)]
        qa = qp = 3
        traj = []
        result = 0
        for _ in range(cap):
            x = featurize(eb.factors(), qa, qp)
            m = _mask(get_valid_moves(qa, mode=mode))
            i, _ = head.act(x, m, rng)
            a = MOVES[i]
            traj.append((x, m, i))
            b = sample_policy(opp.policy(qp, qa, mode=mode, last_opp=a), rng)
            w = judge(a, b)
            eb.observe(a, b)
            if w != 0:
                result = w
                break
            qa, qp = next_qi(qa, a), next_qi(qp, b)
        baseline = (1 - beta) * baseline + beta * result
        head.update(traj, result - baseline)


def line(tag, res):
    r = res["rounds"]
    print(f"  {tag:<16} 胜{res['wins']:>2} 负{res['losses']:>2} 超{res['timeouts']:>2}"
          f" | 胜率 {res['wins']/res['games']:5.1%} | 回合 {statistics.mean(r):4.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-file", dest="encoder_file",
                    default="data/artifacts/encoder_gru.json")
    ap.add_argument("--episodes", type=int, default=400)
    ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model = load_encoder(args.encoder_file)
    head = PolicyHead(seed=args.seed, lr=0.03)
    pool = ["random", "aggressive", "defensive", "turtle"]
    print(f"训练策略头: episodes={args.episodes} pool={pool}")
    train(head, model, args.episodes, args.cap, 0, args.seed, pool)

    evals = ["random", "aggressive", "defensive", "turtle"]
    print("\n[策略头(结果微调)] vs 各 bot")
    for name in evals:
        r = run_duel(HeadAgent(head, model, rng=random.Random(args.seed)),
                     BotAgent(bots_by_name[name]), games=args.games, cap=60)
        line(name, r)

    print("\n[解析 BR(编码器因子, step=0.15)] vs 各 bot（对照）")
    for name in evals:
        ai = OnlineBRAgent(step_cost=0.15, br_sweeps=300,
                           belief_like=EncoderBelief(model))
        r = run_duel(ai, BotAgent(bots_by_name[name]), games=args.games, cap=60)
        line(name, r)


if __name__ == "__main__":
    main()
