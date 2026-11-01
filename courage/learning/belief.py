"""信念因子：把「双方完整出招序列」压成 N 个 0~1 的连续因子。

设计要点：
  - 因子不是「风格标签」，是连续数值；风格只是对因子取值的解读。
  - 用衰减统计 + 变化点 + 双元可预测性 + 镜像倾向，让「先连攻后转守」
    与「攻防交替」产生不同取值。
  - 显式区分「轻攻(attack)」与「重攻(heavy: 螺旋环/咏春)」以及「蓄气(exhale)」，
    以便估计对手的 **升级/节奏威胁**（防御会随对手气数升高而变危险）。
  - 支持 Dirichlet 平滑(alpha) 与先验(prior)冷启动；状态可序列化以便 SQLite 持久化。

因子（均 ∈ [0,1]）：
  long_atk / long_heavy / long_def / long_dodge / long_exhale / long_noqi
  short_atk, attack_streak, switch, predictability, mirror, volatility,
  escalation = long_exhale + long_heavy    （蓄气+重攻的升级倾向）
  threat     = 0.3*exhale + 0.5*attack + 1.0*heavy （当前危险度加权）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ..core.rules import CATEGORY_NAMES, MOVE_CATEGORY

# 招式→类别只在一处定义（core/rules.py），这里仅为兼容旧导入名而再导出。
CATEGORIES: Dict[str, str] = MOVE_CATEGORY
CATS = list(CATEGORY_NAMES)


class BeliefTracker:
    def __init__(self, decay: float = 0.85, short_decay: float = 0.5,
                 max_run: int = 6, alpha: float = 0.20, smooth: float = 0.05,
                 prior: Optional[Dict[str, float]] = None, prior_strength: float = 3.0):
        self.d = decay
        self.ds = short_decay
        self.max_run = max_run
        self.alpha = alpha
        self.smooth = smooth
        self.prior = prior
        self.prior_strength = prior_strength
        self._reset_stats()
        self.last_own: Optional[str] = None
        self.last_opp: Optional[str] = None

    def _reset_stats(self) -> None:
        self.long = {c: 0.0 for c in CATS}
        self.long_tot = 0.0
        self.short = {c: 0.0 for c in CATS}
        self.short_tot = 0.0
        if self.prior:
            s = sum(max(v, 0.0) for v in self.prior.values()) or 1.0
            for c in CATS:
                v = self.prior_strength * max(self.prior.get(c, 0.0), 0.0) / s
                self.long[c] = v
                self.short[c] = v
            self.long_tot = self.prior_strength
            self.short_tot = self.prior_strength
        self.run = 0.0
        self.bigram: Dict[str, Dict[str, float]] = {c: {} for c in CATS}
        self.bigram_tot = 0.0
        self.prev_cat: Optional[str] = None
        self.mirror = 0.0
        self.mirror_tot = 0.0
        self.steps = 0

    def reset(self) -> None:
        self._reset_stats()
        self.last_own = None
        self.last_opp = None

    # ── 增量更新 ──
    def observe(self, own_move: Optional[str], opp_move: Optional[str]) -> None:
        cat = CATEGORIES.get(opp_move or "")
        if cat is None:
            self.last_own, self.last_opp = own_move, opp_move
            return
        self.long = {c: v * self.d for c, v in self.long.items()}
        self.long_tot *= self.d
        self.short = {c: v * self.ds for c, v in self.short.items()}
        self.short_tot *= self.ds
        self.long[cat] += 1.0
        self.long_tot += 1.0
        self.short[cat] += 1.0
        self.short_tot += 1.0
        self.run = min(self.run + 1.0, self.max_run) if cat in ("attack", "heavy") else 0.0
        if self.prev_cat is not None:
            for c in self.bigram:
                self.bigram[c] = {k: w * self.d for k, w in self.bigram[c].items()}
            self.bigram_tot *= self.d
            row = self.bigram[self.prev_cat]
            row[cat] = row.get(cat, 0.0) + 1.0
            self.bigram_tot += 1.0
        self.prev_cat = cat
        self.mirror *= self.d
        self.mirror_tot *= self.d
        self.mirror_tot += 1.0
        if own_move is not None and opp_move == own_move:
            self.mirror += 1.0
        self.steps += 1
        self.last_own, self.last_opp = own_move, opp_move

    # ── 因子 ──
    def _props(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        K = len(CATS)
        dl = self.long_tot + self.alpha * K
        ds = self.short_tot + self.alpha * K
        lp = {c: ((self.long[c] + self.alpha) / dl if dl > 1e-9 else 1.0 / K) for c in CATS}
        sp = {c: ((self.short[c] + self.alpha) / ds if ds > 1e-9 else 1.0 / K) for c in CATS}
        return lp, sp

    def factors(self) -> Dict[str, float]:
        lp, sp = self._props()
        streak = self.run / self.max_run if self.run > 0 else 0.0
        switch = abs(sp["attack"] - lp["attack"]) if self.short_tot > 1e-9 else 0.0
        if self.bigram_tot > 1e-9 and self.prev_cat is not None:
            row = self.bigram[self.prev_cat]
            s = sum(row.values())
            pred = (max(row.values()) / s) if s > 1e-9 else 0.0
        else:
            pred = 0.0
        mirror = self.mirror / self.mirror_tot if self.mirror_tot > 1e-9 else 0.0
        volatility = (1.0 - max(lp.values())) if self.long_tot > 1e-9 else 0.0
        escalation = min(1.0, lp["exhale"] + lp["heavy"])
        threat = min(1.0, 0.3 * lp["exhale"] + 0.5 * lp["attack"] + 1.0 * lp["heavy"])
        return {
            "long_atk": lp["attack"], "long_heavy": lp["heavy"],
            "long_def": lp["defense"], "long_dodge": lp["dodge"],
            "long_exhale": lp["exhale"], "long_noqi": lp["noqi"],
            "short_atk": sp["attack"], "attack_streak": streak,
            "switch": switch, "predictability": pred, "mirror": mirror,
            "volatility": volatility, "escalation": escalation, "threat": threat,
        }

    # ── 因子 → 对手分布 ──
    def opp_weights(self, legal: List[str], last_own: Optional[str] = None) -> Dict[str, float]:
        lp, sp = self._props()
        f = self.factors()
        w_short = f["switch"]
        cat_w = {c: (1.0 - w_short) * lp[c] + w_short * sp[c] for c in CATS}
        if f["attack_streak"] > 0:
            cat_w["attack"] *= (1.0 + 1.5 * f["attack_streak"])
            cat_w["heavy"] *= (1.0 + 1.0 * f["attack_streak"])
        w = {m: cat_w.get(CATEGORIES.get(m, "attack"), 0.0) + 1e-6 for m in legal}
        if last_own is not None and f["mirror"] > 0.3 and last_own in w:
            w[last_own] += f["mirror"]

        # ── bigram 通道：P(next_cat | 对手上一招的类别) ──
        # 一阶频率对「交替出招」（吐气/馒头/吐气/馒头…）完全失明——它的
        # 边际分布和均匀一样，所以 AI 学不到任何东西（实测镜像脚本 52% 卡死）。
        # 转移概率直接看穿这种周期：馒头之后多半是吐气（def0），出手就能打死。
        # β 随该转移行的样本量上升，样本不足时自动回退到无条件分布。
        if self.bigram_tot > 1e-9 and self.prev_cat is not None:
            row = self.bigram[self.prev_cat]
            row_tot = sum(row.values())
            if row_tot > 1e-9:
                k = len(CATS)
                bg = {c: (row.get(c, 0.0) + 0.3) / (row_tot + 0.3 * k) for c in CATS}
                w_bg = {m: bg.get(CATEGORIES.get(m, "attack"), 0.0) + 1e-6 for m in legal}
                s_bg = sum(w_bg.values())
                if s_bg > 1e-12:
                    w_bg = {m: v / s_bg for m, v in w_bg.items()}
                    beta = row_tot / (row_tot + 3.0)
                    w = {m: (1.0 - beta) * w.get(m, 0.0) + beta * w_bg.get(m, 0.0)
                         for m in legal}

        s = sum(w.values())
        w = {m: v / s for m, v in w.items()} if s > 0 else {m: 1.0 / len(legal) for m in legal}
        if self.smooth > 0:
            u = 1.0 / len(legal)
            w = {m: (1.0 - self.smooth) * w[m] + self.smooth * u for m in legal}
        return w

    # ── 序列化（SQLite 持久化用）──
    def to_dict(self) -> Dict[str, object]:
        return {
            "decay": self.d, "short_decay": self.ds, "max_run": self.max_run,
            "alpha": self.alpha, "smooth": self.smooth,
            "long": self.long, "long_tot": self.long_tot,
            "short": self.short, "short_tot": self.short_tot,
            "run": self.run, "bigram": self.bigram, "bigram_tot": self.bigram_tot,
            "prev_cat": self.prev_cat, "mirror": self.mirror,
            "mirror_tot": self.mirror_tot, "steps": self.steps,
            "last_own": self.last_own, "last_opp": self.last_opp,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "BeliefTracker":
        t = cls(decay=float(d.get("decay", 0.85)),
                short_decay=float(d.get("short_decay", 0.5)),
                max_run=int(d.get("max_run", 6)),
                alpha=float(d.get("alpha", 0.20)),
                smooth=float(d.get("smooth", 0.05)))
        t.long = {c: float(d["long"].get(c, 0.0)) for c in CATS}
        t.long_tot = float(d.get("long_tot", 0.0))
        t.short = {c: float(d["short"].get(c, 0.0)) for c in CATS}
        t.short_tot = float(d.get("short_tot", 0.0))
        t.run = float(d.get("run", 0.0))
        bg = d.get("bigram", {})
        t.bigram = {c: {k: float(v) for k, v in bg.get(c, {}).items()} for c in CATS}
        t.bigram_tot = float(d.get("bigram_tot", 0.0))
        t.prev_cat = d.get("prev_cat")
        t.mirror = float(d.get("mirror", 0.0))
        t.mirror_tot = float(d.get("mirror_tot", 0.0))
        t.steps = int(d.get("steps", 0))
        t.last_own = d.get("last_own")
        t.last_opp = d.get("last_opp")
        return t

    @staticmethod
    def describe_style(f: Dict[str, float]) -> str:
        if f["long_def"] > 0.45:
            return "防守型"
        if f["long_heavy"] > 0.30:
            return "重攻型"
        if f["long_atk"] > 0.45:
            return "激进型"
        if f["volatility"] > 0.6:
            return "多变型"
        return "混合型"
