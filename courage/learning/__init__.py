"""学习层：信念因子 + 小型对手模型（C）。本层 numpy 仅用于 opponent 的 MLP。"""

from .belief import BeliefTracker
from .opponent import OpponentModel, features

__all__ = ["BeliefTracker", "OpponentModel", "features"]
