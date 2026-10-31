"""勇气 —— 小游戏《勇气》的算法与工程实现。

重构路线（B + C）：
  B  平稳随机博弈求解 + 胜率最大化（含破局激励），替代旧的非平稳 CFR
  C  小型、跨状态共享参数的对手模型（在线学习风格识别）

工程约定：
  - 仅 learning 层依赖 numpy；core/bots/storage/solvers 保持标准库可用
  - 产物（策略表/权重）落在 data/artifacts/，git 忽略，桌面与浏览器共用
"""

__version__ = "0.2.0"
