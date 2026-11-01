"""求解器：精确最佳响应 / 增广 MDP oracle / 均衡 / 多通道风险 BR。

- ``best_response.py``  36 状态上的精确 BR（单边 MDP 值迭代）
- ``aug_mdp.py``        把对手真正依赖的相位并入状态后的精确值 oracle（评测基准）
- ``br_channels.py``    多通道对手模型的风险 BR（通道分歧定价）
- ``equilibrium.py``    批量 Hedge 求纳什均衡（安全网 / 参照）
"""
