"""信念因子：把「双方完整出招序列」压成 N 个 0~1 的连续因子。

设计要点（对齐用户构想）：
  - 因子不是「风格标签」，而是连续数值；「风格」只是对因子取值的形象化解读。
  - 只看上一招太浅；这里用**衰减统计 + 变化点 + 双元可预测性 + 镜像倾向**，
    让「先连攻几手后转守」与「攻防交替」产生不同的因子取值。
  - 因子供后续决策（BR 的对手分布）使用；定长、O(1) 增量更新、无 numpy。

因子（均 ∈ [0,1]）：
  long_atk / long_def / long_dodge / long_exhale / long_noqi  长程动作倾向
  short_atk                                                   短程攻击倾向
  attack_streak                                               当前连攻长度（归一）
  switch                                                      近期 vs 长程的分歧（变化点强度）
  predictability                                              双元转移的集中度
  mirror                                                      对手模仿「我上一招」的倾向
  volatility                                                  对手动作的分散程度
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

CATEGORIES: Dict[str, str] = {
    "馒头": "attack", "双馒": "attack", "雷切": "attack",
    "螺旋环": "attack", "咏春": "attack",
    "防御": "defense", "跳": "dodge", "吐气": "exhale", "无气拨": "noqi",
}
CATS = ["attack", "defense", "dodge", "exhale", "noqi"]


class BeliefTracker:
    def __init__(self, decay: float = 0.85, short_decay: float = 0.5,
                 max_run: int = 6, alpha: float = 0.25, smooth: float = 0.05):
        # alpha  : 每类动作的 Dirichlet 伪计数（越大越平滑、越不轻易下结论）
        # smooth : 最终对手分布与均匀分布的混合比例（防止过度自信）
        self.d = decay
        self.ds = short_decay
        self.max_run = max_run
        self.alpha = alpha
        self.smooth = smooth
        self._reset_stats()
        self.last_own: Optional[str] = None
        self.last_opp: Optional[str] = None

    def _reset_stats(self) -> None:
        self.long = {c: 0.0 for c in CATS}
        self.long_tot = 0.0
        self.short = {c: 0.0 for c in CATS}
        self.short_tot = 0.0
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

    # ── 增量更新（每回合调用一次）──
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

        self.run = min(self.run + 1.0, self.max_run) if cat == "attack" else 0.0

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

    # ── 因子读出 ──
    def _props(self) -> Tuple[Dict[str, float], Dict[str, float]]:
        K = len(CATS)
        den_l = self.long_tot + self.alpha * K
        den_s = self.short_tot + self.alpha * K
        lp = {c: ((self.long[c] + self.alpha) / den_l if den_l > 1e-9 else 1.0 / K)
              for c in CATS}
        sp = {c: ((self.short[c] + self.alpha) / den_s if den_s > 1e-9 else 1.0 / K)
              for c in CATS}
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
        return {
            "long_atk": lp["attack"], "long_def": lp["defense"],
            "long_dodge": lp["dodge"], "long_exhale": lp["exhale"],
            "long_noqi": lp["noqi"], "short_atk": sp["attack"],
            "attack_streak": streak, "switch": switch,
            "predictability": pred, "mirror": mirror, "volatility": volatility,
        }

    # ── 因子 → 对手在当前合法招式上的权重（供 BR 的对手分布）──
    def opp_weights(self, legal: List[str],
                    last_own: Optional[str] = None) -> Dict[str, float]:
        lp, sp = self._props()
        f = self.factors()
        w_short = f["switch"]  # 变化点越强，越信短程
        cat_w = {c: (1.0 - w_short) * lp[c] + w_short * sp[c] for c in CATS}
        if f["attack_streak"] > 0:
            cat_w["attack"] *= (1.0 + 1.5 * f["attack_streak"])

        w = {m: cat_w.get(CATEGORIES.get(m, "attack"), 0.0) + 1e-6 for m in legal}
        # 镜像倾向：对手爱复制我上一招
        if last_own is not None and f["mirror"] > 0.3 and last_own in w:
            w[last_own] += f["mirror"]
        s = sum(w.values())
        w = {m: v / s for m, v in w.items()} if s > 0 else {m: 1.0 / len(legal) for m in legal}
        # 与均匀混合，避免对噪声人类过度自信
        if self.smooth > 0:
            u = 1.0 / len(legal)
            w = {m: (1.0 - self.smooth) * w[m] + self.smooth * u for m in legal}
        return w

    # ── 「风格」只是对因子的解读（可选、纯展示，不参与决策）──
    @staticmethod
    def describe_style(f: Dict[str, float]) -> str:
        if f["long_def"] > 0.55:
            return "防守型"
        if f["long_atk"] > 0.55:
            return "激进型"
        if f["volatility"] > 0.6:
            return "多变型"
        if f["predictability"] > 0.7:
            return "模式化"
        return "混合型"
