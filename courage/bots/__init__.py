"""脚本化对抗对手：有计划地生成对抗样本。

真实人类数据稀缺，故用一组覆盖策略空间的 bot（随机/进攻/防守/龟缩/
循环/反跳/镜像）作为训练与评估对手。
"""

from .scripts import (
    Bot, ALL_BOTS, bots_by_name, sample_policy, best_response,
)

__all__ = ["Bot", "ALL_BOTS", "bots_by_name", "sample_policy", "best_response"]
