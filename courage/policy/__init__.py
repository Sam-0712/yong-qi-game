"""策略层：难度/风格/噪声（表现旋钮）+ 开局簿 + 混合工具。

「风格」在这里指**我方**输出分布的整体调制（``PlaySettings.style``），
与 ``bots/style_profiles.py`` 里「对手的整体攻防偏好」是同一个概念的两个用法。
"""

from .settings import PlaySettings, STYLE_BIAS, apply_settings

__all__ = ["PlaySettings", "STYLE_BIAS", "apply_settings"]
