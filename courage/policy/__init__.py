"""策略层：性格与自适应（旧版，B/C 阶段将被重构）。"""

from .personality import Personality, PRESETS, get_personality, apply_bias

__all__ = ["Personality", "PRESETS", "get_personality", "apply_bias"]
