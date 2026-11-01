"""上下文对手模型：从大规模对局语料学「对手下一步」，替代手写类型策略。

模型是分层 n-gram / 条件计数：
  full   : (对手最近两类, 己方最近一类, qa, qp)
  bigram : (对手最近一类, 己方最近一类, qa, qp)
  opp    : (对手最近一类, qp)
  qi     : (qa, qp)
  none   : 均匀
逐层回退；每层对合法招做 Laplace 平滑。它没有手写风格规则，完全由数据决定，
因此对 pressure / alternate_low 这类固定打法也能学到。

接口与 OpponentTypeModel 对齐：
  observe(qa, qp, own_move, opp_move)
  opp_weights(qa, qp, legal, last_own)
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional

from .belief import CATEGORIES

LEVELS = ["full", "bigram", "opp", "qi", "none"]


def _cat(move: Optional[str]) -> str:
    return CATEGORIES.get(move or "", "attack") if move else "-"


def context_keys(own_hist: List[str], opp_hist: List[str],
                 qa: int, qp: int) -> List[tuple[str, str]]:
    """训练与推理共用的上下文键；顺序即回退优先级。"""
    c_opp = _cat(opp_hist[-1]) if opp_hist else "-"
    c_own = _cat(own_hist[-1]) if own_hist else "-"
    c2 = ",".join(_cat(m) for m in opp_hist[-2:]) if opp_hist else "-"
    return [
        ("full", f"{c2}|{c_own}|{qa}|{qp}"),
        ("bigram", f"{c_opp}|{c_own}|{qa}|{qp}"),
        ("opp", f"{c_opp}|{qp}"),
        ("qi", f"{qa}|{qp}"),
        ("none", "-"),
    ]


class ContextOpponentModel:
    def __init__(self, counts: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
                 smooth: float = 0.5, meta: Optional[dict] = None):
        self.counts = counts or {lv: {} for lv in LEVELS}
        self.smooth = max(1e-6, smooth)
        self.meta = meta or {}
        self.own: List[str] = []
        self.opp: List[str] = []
        self.last_own = None
        self.steps = 0

    def reset(self) -> None:
        self.own = []
        self.opp = []
        self.last_own = None
        self.steps = 0

    def observe(self, qa: int, qp: int, own_move: str, opp_move: str) -> None:
        self.own.append(own_move)
        self.opp.append(opp_move)
        self.last_own = own_move
        self.steps += 1

    def opp_weights(self, qa: int, qp: int, legal: List[str],
                    last_own: Optional[str] = None) -> Dict[str, float]:
        if not legal:
            return {}
        for level, key in context_keys(self.own, self.opp, qa, qp):
            row = self.counts.get(level, {}).get(key)
            if not row:
                continue
            total = sum(max(0.0, row.get(m, 0.0)) for m in legal)
            if total <= 0:
                continue
            denom = total + self.smooth * len(legal)
            p = {m: (max(0.0, row.get(m, 0.0)) + self.smooth) / denom
                 for m in legal}
            s = sum(p.values())
            return {m: v / s for m, v in p.items()}
        u = 1.0 / len(legal)
        return {m: u for m in legal}

    # ── 序列化 ──
    def to_dict(self) -> dict:
        return {"levels": self.counts, "smooth": self.smooth, "meta": self.meta}

    @classmethod
    def from_dict(cls, blob: dict) -> "ContextOpponentModel":
        return cls(counts=blob.get("levels", {}),
                   smooth=float(blob.get("smooth", 0.5)),
                   meta=blob.get("meta", {}))

    @classmethod
    def load(cls, path: str) -> "ContextOpponentModel":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False)
