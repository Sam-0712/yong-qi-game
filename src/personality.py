"""
勇气 — AI 性格系统
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Personality:
    name: str = "平衡型"
    cross_temperature: float = 0.3
    cross_blend_rate: float = 0.4
    cross_confidence_k: float = 8.0
    within_temperature: float = 0.3
    within_max_blend: float = 0.5
    within_prior: float = 6.0
    dcfr_power_positive: float = 1.5
    gamma: float = 0.85
    attack_bias: float = 0.0
    defense_bias: float = 0.0


PRESETS: Dict[str, Personality] = {
    "balanced": Personality(name="平衡型"),
    "aggressive": Personality(
        name="激进型",
        cross_temperature=0.15, cross_blend_rate=0.5,
        within_temperature=0.15, within_max_blend=0.6,
        attack_bias=0.15, defense_bias=-0.1,
    ),
    "defensive": Personality(
        name="防守型",
        cross_temperature=0.5, cross_blend_rate=0.3,
        within_temperature=0.5, within_max_blend=0.4,
        within_prior=10.0, attack_bias=-0.15, defense_bias=0.15,
    ),
    "gambler": Personality(
        name="赌徒型",
        cross_temperature=0.05, cross_blend_rate=0.7,
        within_temperature=0.05, within_max_blend=0.8,
        within_prior=3.0, attack_bias=0.2,
    ),
    "thinker": Personality(
        name="深思型",
        dcfr_power_positive=1.5,
        cross_temperature=0.2, cross_blend_rate=0.3,
        within_temperature=0.2, within_max_blend=0.3,
        within_prior=12.0,
    ),
}


def get_personality(name: str = "balanced") -> Personality:
    return PRESETS.get(name, PRESETS["balanced"])


def apply_bias(strategy: Dict[str, float], p: Personality) -> Dict[str, float]:
    if not p.attack_bias and not p.defense_bias:
        return strategy
    b = dict(strategy)
    for m in b:
        if m in ("馒头", "双馒", "雷切", "螺旋环", "咏春"):
            b[m] *= (1.0 + p.attack_bias)
        if m in ("防御", "吐气"):
            b[m] *= (1.0 + p.defense_bias)
    t = sum(b.values())
    return {m: b[m] / t for m in b} if t else strategy
