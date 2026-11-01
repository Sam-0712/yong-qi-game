"""勇气 —— 小游戏《勇气》的算法与工程实现。

5.0 决策链：
    序列/历史 → 对手模型通道（打法脚本 / 上下文 n-gram / 手写信念 / 无结构）
              → 通道混合（预测似然指数加权）
              → 精确 BR（36 状态值迭代；可对通道分歧定价）
              → 难度 / 我方风格 / 噪声（表现层）

工程约定：
  - 仅 learning / solvers 层依赖 numpy；core/bots/policy/storage 保持标准库可用
  - 产物落在 data/artifacts/（git 忽略）；用 python -m tools.repro 校验或重建
  - 评测入口统一走 python -m tools.bench（36 状态精确 oracle + 逐决策 regret）
"""

__version__ = "0.3.0"
