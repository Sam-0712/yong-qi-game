"""
勇气 — CFR / DCFR 训练器
=============================
Tabular CFR/DCFR，按回合逆序遍历 DAG，数值索引加速。
"""

import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from .game import (
    build_state_graph, state_summary, decode_state,
    State, MOVES, MOVES_ATTR, MOVE_TO_IDX, IDX_TO_MOVE,
    MAX_QI, MAX_ROUNDS,
)


@dataclass
class _CompiledState:
    va: List[int]
    vb: List[int]
    na: int
    nb: int
    trans_winner: List[List[int]] = None
    trans_next: List[List[Optional[int]]] = None
    is_terminal: bool = False


def _compile(graph: Dict, start: State):
    def _int(s):
        return (s[0] << 17) | (s[1] << 14) | (s[2] << 10) | (s[3] << 6) | s[4]

    compiled, s2i = {}, {s: _int(s) for s in graph}
    for st, info in graph.items():
        sk = s2i[st]
        va, vb = info['valid_ai'], info['valid_pl']
        na, nb = len(va), len(vb)

        if 'terminal_value' in info or na == 0 or nb == 0:
            compiled[sk] = _CompiledState(
                va=[MOVE_TO_IDX[m] for m in va],
                vb=[MOVE_TO_IDX[m] for m in vb],
                na=na, nb=nb, is_terminal=True)
            continue

        tw = [[0]*nb for _ in range(na)]
        tn = [[None]*nb for _ in range(na)]
        for i, ma in enumerate(va):
            for j, mb in enumerate(vb):
                out = info['transitions'][(ma, mb)]
                if out[0] == 'terminal':
                    tw[i][j] = out[1]
                else:
                    tn[i][j] = s2i[out]
        compiled[sk] = _CompiledState(va=[MOVE_TO_IDX[m] for m in va],
                                      vb=[MOVE_TO_IDX[m] for m in vb],
                                      na=na, nb=nb, trans_winner=tw, trans_next=tn)
    return compiled, _int(start)


class CFRTrainer:
    """
    Tabular CFR/DCFR 训练器。

    参数:
        gamma:               折扣因子 (0,1]。
        dcfr_power_positive: 正遗憾折扣幂 (0=无折扣,1=线性,1.5=加速)。
        dcfr_power_negative: 负遗憾折扣幂 (推荐 0.0)。
        weight_strategy:     累积策略是否线性加权 (LCFR)。
    """

    def __init__(self, gamma=0.85, dcfr_power_positive=1.5,
                 dcfr_power_negative=0.0, weight_strategy=True):
        self.gamma = gamma
        self.dcfr_power_positive = dcfr_power_positive
        self.dcfr_power_negative = dcfr_power_negative
        self.weight_strategy = weight_strategy

        print("构建状态图...")
        raw, raw_start = build_state_graph()
        state_summary(raw)
        self.graph, self.start_key = _compile(raw, raw_start)

        self.sorted_keys = sorted(
            [k for k, cs in self.graph.items() if not cs.is_terminal],
            key=lambda k: k & 0x3F, reverse=True)
        self.discounts = [gamma ** r for r in range(MAX_ROUNDS)]

        self.regret_ai: Dict[int, List[float]] = {}
        self.regret_pl: Dict[int, List[float]] = {}
        self.cum_ai: Dict[int, List[float]] = {}
        self.cum_pl: Dict[int, List[float]] = {}
        for sk, cs in self.graph.items():
            if cs.is_terminal:
                continue
            self.regret_ai[sk] = [0.0] * cs.na
            self.regret_pl[sk] = [0.0] * cs.nb
            self.cum_ai[sk] = [0.0] * cs.na
            self.cum_pl[sk] = [0.0] * cs.nb

    @staticmethod
    def _rm(regrets, n):
        pos = [max(0, r) for r in regrets]
        t = sum(pos)
        if t > 1e-12:
            return [p / t for p in pos]
        return [1.0 / n] * n

    def _run_one_iteration(self, it: int) -> float:
        base = it / (it + 1.0)
        ap = base ** self.dcfr_power_positive if self.dcfr_power_positive else 1.0
        an = base ** self.dcfr_power_negative if self.dcfr_power_negative else 1.0
        nv, ds = {}, self.discounts

        for sk in self.sorted_keys:
            cs = self.graph[sk]
            ra, rb = self.regret_ai[sk], self.regret_pl[sk]
            ca, cb = self.cum_ai[sk], self.cum_pl[sk]
            na, nb = cs.na, cs.nb

            sa, sb = self._rm(ra, na), self._rm(rb, nb)
            tw, tn = cs.trans_winner, cs.trans_next
            disc = ds[sk & 0x3F]

            vn, cfa, cfb = 0.0, [0.0]*na, [0.0]*nb
            for i in range(na):
                sai, twi, tni = sa[i], tw[i], tn[i]
                cfi = 0.0
                for j in range(nb):
                    spj = sb[j]
                    w = twi[j]
                    v = disc * w if w else (0.0 if (tni[j] & 0x3F) >= MAX_ROUNDS else nv[tni[j]])
                    cfi += spj * v
                    cfb[j] += sai * v
                    vn += sai * spj * v
                cfa[i] = cfi
            nv[sk] = vn

            for i in range(na):
                d = cfa[i] - vn
                ra[i] += d * (ap if d >= 0 else an)
            for j in range(nb):
                d = vn - cfb[j]
                rb[j] += d * (ap if d >= 0 else an)

            w = it if self.weight_strategy else 1.0
            for i in range(na):
                ca[i] += sa[i] * w
            for j in range(nb):
                cb[j] += sb[j] * w

        return nv.get(self.start_key, 0.0)

    def train(self, iterations=1000, print_every=50, cb=None):
        print(f"\n训练: {iterations} iter, gamma={self.gamma}, "
              f"DCFR(p+={self.dcfr_power_positive},p-={self.dcfr_power_negative})")
        ts = time.time()
        for it in range(1, iterations + 1):
            rv = self._run_one_iteration(it)
            if it % print_every == 0 or it == 1:
                el = time.time() - ts
                print(f"  iter {it:>4}/{iterations}  root={rv:+.6f}  "
                      f"{it/el:.0f} iter/s  {el:.1f}s")
                if cb:
                    cb(it, rv, el)

    def compute_average_strategy_value(self) -> float:
        nv, ds = {}, self.discounts
        for sk in self.sorted_keys:
            cs = self.graph[sk]
            ca, cb = self.cum_ai[sk], self.cum_pl[sk]
            ta, tb = sum(ca) or 1.0, sum(cb) or 1.0
            tw, tn = cs.trans_winner, cs.trans_next
            dc = ds[sk & 0x3F]
            vn = 0.0
            for i in range(cs.na):
                sai = ca[i] / ta
                if sai < 1e-12:
                    continue
                twi, tni = tw[i], tn[i]
                for j in range(cs.nb):
                    spj = cb[j] / tb
                    if spj < 1e-12:
                        continue
                    w = twi[j]
                    v = dc * w if w else (0.0 if (tni[j] & 0x3F) >= MAX_ROUNDS else nv[tni[j]])
                    vn += sai * spj * v
            nv[sk] = vn
        return nv.get(self.start_key, 0.0)

    def get_average_strategy(self, player='ai'):
        cum_src = self.cum_ai if player == 'ai' else self.cum_pl
        result = {}
        for sk, cs in self.graph.items():
            if cs.is_terminal:
                continue
            st = ((sk >> 17) & 0x7, (sk >> 14) & 0x7,
                  (sk >> 10) & 0xF, (sk >> 6) & 0xF, sk & 0x3F)
            key = f"{st[0]},{st[1]},{st[2]},{st[3]},{st[4]}"
            moves = cs.va if player == 'ai' else cs.vb
            cum = cum_src[sk]
            total = sum(cum)
            if total > 1e-12:
                result[key] = {IDX_TO_MOVE[m]: cum[i] / total
                               for i, m in enumerate(moves)}
            else:
                inv = 1.0 / len(moves)
                result[key] = {IDX_TO_MOVE[m]: inv for m in moves}
        return result


