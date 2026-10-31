"""对局设置：难度与风格（均可调）。

设计目标：求解器给出「最优策略」后，由本层在运行时调整表现，二者解耦。

难度 difficulty ∈ [0,1]：
    1.0 = 完全按策略最优出手
    0.0 = 完全均匀随机（最弱）
    中间 = 以 d 的概率按策略、1-d 的概率随机（近似「失误率」旋钮）

风格 style：对策略做乘性偏置后重归一（不改求解器本身）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

# 招式 → 乘性权重（>1 鼓励，<1 抑制）
STYLE_BIAS: Dict[str, Dict[str, float]] = {
    "balanced": {},
    "aggressive": {
        "馒头": 1.3, "双馒": 1.3, "雷切": 1.3, "螺旋环": 1.25, "咏春": 1.25,
        "防御": 0.6, "跳": 0.7,
    },
    "defensive": {
        "防御": 1.5, "跳": 1.35, "吐气": 1.1,
        "馒头": 0.7, "双馒": 0.7, "雷切": 0.6, "螺旋环": 0.6, "咏春": 0.6,
    },
    "gambler": {
        "螺旋环": 1.5, "咏春": 1.6, "防御": 0.4, "跳": 0.5, "吐气": 0.6,
    },
    "thinker": {
        "防御": 1.2, "跳": 1.1, "咏春": 1.05, "无气拨": 1.0,
    },
}


@dataclass
class PlaySettings:
    difficulty: float = 1.0
    style: str = "balanced"
    # 额外随机噪声 ∈ [0,1]：在难度混合之上再叠一层均匀混合，
    # 主要用于打破低气/无气拨等长循环（如 mode1）。
    noise: float = 0.0

    def clamped_difficulty(self) -> float:
        return min(1.0, max(0.0, float(self.difficulty)))

    def clamped_noise(self) -> float:
        return min(1.0, max(0.0, float(self.noise)))


def apply_settings(policy: Dict[str, float], legal: List[str],
                   settings: PlaySettings) -> Dict[str, float]:
    """把求解器策略变成最终出手分布（风格偏置 + 难度混合 + 噪声）。"""
    bias = STYLE_BIAS.get(settings.style, {})
    p = {}
    for m in legal:
        p[m] = max(float(policy.get(m, 0.0)), 0.0) * bias.get(m, 1.0)
    s = sum(p.values())
    if s <= 1e-12:
        p = {m: 1.0 for m in legal}
        s = float(len(legal))
    p = {m: v / s for m, v in p.items()}

    u = 1.0 / len(legal)
    d = settings.clamped_difficulty()
    q = {m: d * p[m] + (1.0 - d) * u for m in legal}

    nz = settings.clamped_noise()
    if nz > 0.0:
        q = {m: (1.0 - nz) * q[m] + nz * u for m in legal}

    t = sum(q.values())
    return {m: v / t for m, v in q.items()}


# 各模式默认盘外随机噪声：无气拨模式天生易长循环，给更大噪声。
DEFAULT_NOISE_BY_MODE: Dict[int, float] = {0: 0.0, 1: 0.3}
