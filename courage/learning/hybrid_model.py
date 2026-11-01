"""单一对手模型接口：类型后验 + 上下文学习模型。

为什么需要混合：
- 类型后验对固定脚本（converter/4手/2手/mirror）很准，但对集合外风格/低气交替
  会退化到 random；
- 上下文学习模型能识别周期/风格（(1,1) 交替从 0/50 提升到 50/50），
  但对 converter 这类依赖完整合法集的策略偏弱。

切换规则（一个机制，不做多级路由）：
  类型后验的 top type 明确且不是 random/低置信 → 用类型后验；
  否则（top=random、最大权重 < min_type_weight、没有类型模型）→ 用上下文模型。
"""

from __future__ import annotations

from typing import Dict, List, Optional


class HybridOpponentModel:
    def __init__(self, type_model=None, context_model=None,
                 unknown_types=("random",), min_known_mass: float = 0.5,
                 min_top_known: float = 0.25):
        self.type_model = type_model
        self.context_model = context_model
        self.unknown_types = set(unknown_types)
        self.min_known_mass = float(min_known_mass)
        self.min_top_known = float(min_top_known)

    def reset(self) -> None:
        if self.type_model is not None:
            self.type_model.reset()
        if self.context_model is not None:
            self.context_model.reset()

    def observe(self, qa: int, qp: int, own_move: str, opp_move: str) -> None:
        if self.type_model is not None:
            self.type_model.observe(qa, qp, own_move, opp_move)
        if self.context_model is not None:
            self.context_model.observe(qa, qp, own_move, opp_move)

    def _use_context(self) -> bool:
        if self.context_model is None:
            return False
        if self.type_model is None:
            return True
        # 类型后验有全局权重 w：看「已知类型总质量」和「最强已知类型」，
        # 而不是只看 top1。否则 random 排第一时会误伤 converter/4手。
        w = getattr(self.type_model, "w", None)
        if isinstance(w, dict) and w:
            known = {t: v for t, v in w.items() if t not in self.unknown_types}
            known_mass = sum(known.values())
            top_known = max(known.values()) if known else 0.0
            return known_mass < self.min_known_mass or top_known < self.min_top_known
        # 没有权重接口时退回旧规则
        if hasattr(self.type_model, "top_types"):
            top = self.type_model.top_types(1)
            if top:
                name, wt = top[0]
                return name in self.unknown_types or wt < self.min_known_mass
        return False

    def opp_weights(self, qa: int, qp: int, legal: List[str],
                    last_own: Optional[str] = None) -> Dict[str, float]:
        if self._use_context():
            return self.context_model.opp_weights(qa, qp, legal, last_own)
        return self.type_model.opp_weights(qa, qp, legal, last_own)
