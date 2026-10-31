"""item 1：校准消融。用 τ(锐化) × floor(截断) 去噪，看能否在不调 step_cost 时救回
编码器信念对防守型对手的"过度谨慎/停滞"。

固定 step_cost=0.03（此前对 turtle 会全超时的设置），只变校准参数。
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.agents import BotAgent, OnlineBRAgent, run_duel
from courage.bots.scripts import bots_by_name
from courage.learning.seq_encoder import EncoderBelief, load_encoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder-file", dest="encoder_file",
                    default="data/artifacts/encoder_gru.json")
    ap.add_argument("--taus", default="1,2,3")
    ap.add_argument("--floors", default="0,0.02")
    ap.add_argument("--opps", default="turtle,defensive,random")
    ap.add_argument("--games", type=int, default=8)
    ap.add_argument("--cap", type=int, default=50)
    ap.add_argument("--br-sweeps", dest="br_sweeps", type=int, default=150)
    ap.add_argument("--step-cost", dest="step_cost", type=float, default=0.03)
    args = ap.parse_args()

    model = load_encoder(args.encoder_file)
    opps = args.opps.split(",")
    taus = [float(x) for x in args.taus.split(",")]
    floors = [float(x) for x in args.floors.split(",")]

    print(f"固定 step_cost={args.step_cost}  games={args.games} cap={args.cap}")
    print(f"{'tau':>4}{'floor':>7}  " + "".join(f"{o:>14}" for o in opps))
    print("-" * (11 + 14 * len(opps)))
    for tau in taus:
        for fl in floors:
            cells = []
            for o in opps:
                ai = OnlineBRAgent(step_cost=args.step_cost, br_sweeps=args.br_sweeps,
                                   belief_like=EncoderBelief(model, tau=tau, floor=fl))
                r = run_duel(ai, BotAgent(bots_by_name[o]), mode=0,
                             games=args.games, cap=args.cap)
                wr = r["wins"] / r["games"]
                if r["timeouts"]:
                    cells.append(f"{wr:5.0%}(T{r['timeouts']})")
                else:
                    cells.append(f"{wr:7.0%}/{statistics.mean(r['rounds']):4.1f}")
            print(f"{tau:>4.1f}{fl:>7.2f}  " + "".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
