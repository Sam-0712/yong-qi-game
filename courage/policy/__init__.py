"""策略层：性格 / 难度风格设置 / 自适应（旧版，B/C 阶段继续重构）。"""

from .personality import Personality, PRESETS, get_personality, apply_bias
from .settings import PlaySettings, STYLE_BIAS, apply_settings

__all__ = [
    "Personality", "PRESETS", "get_personality", "apply_bias",
    "PlaySettings", "STYLE_BIAS", "apply_settings",
]
