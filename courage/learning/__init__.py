"""学习层：信念因子 + 对手模型通道（类型后验 / 上下文 / 混合 / 序列编码器）。

通道可以叠加：``MixtureOpponentModel`` 把其中几条按预测似然指数加权后混合，
再由 ``solvers/br_channels.py`` 按通道分歧定价。
"""

from .belief import BeliefTracker
from .type_model import OpponentTypeModel
from .context_model import ContextOpponentModel
from .hybrid_model import HybridOpponentModel
from .mixture_model import MixtureOpponentModel, UniformChannel

__all__ = ["BeliefTracker", "OpponentTypeModel", "ContextOpponentModel",
           "HybridOpponentModel", "MixtureOpponentModel", "UniformChannel"]
