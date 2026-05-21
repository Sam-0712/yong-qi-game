"""
自适应层：跨局 / 局内 / 融合
"""

import math
from typing import Dict, List
from collections import defaultdict
from .game import judge, get_valid_moves


class AdaptiveLayer:
    """跨局自适应"""
    def __init__(self, temperature=0.3, blend_rate=0.4, confidence_k=8.0):
        self.t, self.br, self.ck = temperature, blend_rate, confidence_k
        self.pc = defaultdict(lambda: defaultdict(int))
        self.sv = defaultdict(int)

    def observe(self, key, move):
        self.pc[key][move] += 1
        self.sv[key] += 1

    def get_adjusted_strategy(self, key, cfr, qi_a, qi_p, la, lp):
        v = self.sv.get(key, 0)
        if v < 2:
            return dict(cfr)
        lam = self.br * v / (v + self.ck)
        br = self._br(key, qi_a, qi_p, la, lp) or {}
        adj = {a: (1 - lam) * cfr[a] + lam * br.get(a, 0) for a in cfr}
        t = sum(adj.values())
        return {a: adj[a] / t for a in adj} if t else dict(cfr)

    def _br(self, key, qi_a, qi_p, la, lp):
        va, vb = get_valid_moves(qi_a, la), get_valid_moves(qi_p, lp)
        if not va or not vb:
            return {}
        cnt, tc = self.pc.get(key, {}), sum(self.pc.get(key, {}).values())
        if tc < 1:
            return {}
        eps = 0.3
        emp = {m: (cnt.get(m, 0) + eps) / (tc + eps * len(vb)) for m in vb}
        et = sum(emp.values())
        if et:
            for m in emp:
                emp[m] /= et
        ev = [sum(emp[p] * judge(a, p) for p in emp) for a in va]
        mx = max(ev) if ev else 0
        ex = [math.exp((e - mx) / self.t) for e in ev]
        xt = sum(ex)
        return {a: ex[i] / xt for i, a in enumerate(va)} if xt else {}

    def reset(self):
        self.pc.clear()
        self.sv.clear()


class WithinGameAdaptive:
    """局内自适应"""
    def __init__(self, temperature=0.3, max_blend=0.5, prior_strength=6.0):
        self.t, self.mb, self.ps = temperature, max_blend, prior_strength
        self.counts, self.total = defaultdict(int), 0

    def observe(self, move):
        self.counts[move] += 1
        self.total += 1

    def get_adjusted_strategy(self, cfr, qi_a, qi_p, la, lp):
        if self.total < 2:
            return dict(cfr)
        va, vb = get_valid_moves(qi_a, la), get_valid_moves(qi_p, lp)
        if not va or not vb:
            return dict(cfr)
        post = {}
        for m in vb:
            c = self.counts.get(m, 0)
            post[m] = (self.ps / len(vb) + c) / (self.ps + self.total)
        pt = sum(post.values())
        if pt:
            for m in post:
                post[m] /= pt
        ev = [sum(post[p] * judge(a, p) for p in post) for a in va]
        mx = max(ev) if ev else 0
        ex = [math.exp((e - mx) / self.t) for e in ev]
        xt = sum(ex)
        br = {a: ex[i] / xt for i, a in enumerate(va)} if xt else {}
        conf = self.total / (self.total + self.ps)
        bl = self.mb * conf
        adj = {a: (1 - bl) * cfr[a] + bl * br.get(a, 0) for a in cfr}
        t = sum(adj.values())
        return {a: adj[a] / t for a in adj} if t else dict(cfr)

    def reset(self):
        self.counts.clear()
        self.total = 0

    def get_global_distribution(self):
        t = sum(self.counts.values())
        return {m: c / t for m, c in self.counts.items()} if t else {}


class CombinedAdaptive:
    """融合"""
    def __init__(self, cross_kw=None, within_kw=None):
        self.cross = AdaptiveLayer(**(cross_kw or {}))
        self.within = WithinGameAdaptive(**(within_kw or {}))

    def observe(self, key, move):
        self.cross.observe(key, move)
        self.within.observe(move)

    def get_adjusted_strategy(self, key, cfr, qi_a, qi_p, la, lp):
        mid = self.cross.get_adjusted_strategy(key, cfr, qi_a, qi_p, la, lp)
        return self.within.get_adjusted_strategy(mid, qi_a, qi_p, la, lp)

    def reset(self):
        self.within.reset()
