"""勇气 5.0 决策链——统一入口。

把定型的链路封装成一个可复用对象：
    序列 → (单 GRU) 信念因子 → 校准(τ/floor) → 解析 BR → 难度/风格/噪声
可选：冷启动先验、跨会话 SQLite 记忆、手写信念 或 编码器信念。

一个 PlayerConfig 决定一切；CouragePlayer 暴露 act/observe/reset 三个方法，
可直接用于对局循环（与 agents.run_duel 兼容）。
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Optional

from .agents import PRIORS, OnlineBRAgent
from .policy.settings import PlaySettings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_ROOT, path)


@dataclass
class PlayerConfig:
    mode: int = 0
    belief_source: str = "encoder"            # "encoder" 或 "hand"
    encoder_file: str = "data/artifacts/encoder_gru.json"
    tau: float = 2.0                          # 校准：锐化
    floor: float = 0.0                        # 校准：小概率截断
    gamma: float = 0.95
    step_cost: float = 0.03
    br_sweeps: int = 600
    # 表现旋钮
    difficulty: float = 1.0
    style: str = "balanced"
    noise: float = 0.0
    # 对手模型：类型后验（默认）或上下文学习模型（context_file 非空时优先）
    type_posterior: bool = True
    context_file: str = "data/artifacts/context_all.json"
    type_blend: float = 1.0                   # 1=纯类型后验，0=纯序列信念
    type_decay: float = 1.0                   # <1 时更快遗忘旧证据
    type_eps: float = 0.02                    # 观测似然下限（不把类型一票否决）
    type_smooth: float = 0.02
    # 通道装配："switch"=硬切换（旧）/ "mix"=按预测似然指数加权后混合
    gate: str = "mix"
    mix_eta: float = 1.0                      # 1=贝叶斯模型平均；<1 更保守
    mix_decay: float = 1.0                    # <1 更快遗忘旧证据（非平稳对手）
    mix_prior: float = 0.02                   # 权重地板，防一条通道被永久打死
    mix_null: float = 0.4                     # 「无结构」零假设（uniform 通道）的先验
    mix_hand: bool = False                    # 14 因子手写信念：实测 logloss 不如均匀，默认关
    channel_kappa: float = 0.0                # 通道分歧定价：0=风险中性，1=纯 maximin
    # 抗针对性（见 docs/勇气5.0算法修正-针对性漏洞.md）
    opening: str = "safe"                     # "safe"=开局簿采样 / "off"=BR 决定首招
    opening_book: str = "opening_mode0.json"  # 训练产物；缺失则用内置安全分布
    robust: float = 0.3                       # 幽灵威胁（理性转化者）权重，按样本量衰减但不归零
    zero_penalty: float = 0.15                # 归零惩罚：把气打到 0 的续接价值要打折
    loop_risk: float = 0.9                    # 破僵局时选「改气数」招的概率
    round_total: int = 15                     # 预期对局长度（轮数退火基准）
    noise0: float = 0.0                       # 起始随机比例（默认关：防死锁交给 anti_loop）
    urgency: float = 3.0                      # step_cost 在末段的放大倍数
    anti_loop: bool = True                    # 周期检测（局部死锁时主动偏离）
    qi_floor: int = 0                         # 气数下限（0=关）；实测开启会过度保守
    # 冷启动 & 记忆
    prior: str = "none"                       # PRIORS 键
    decay: float = 0.85
    alpha: float = 0.20
    smooth: float = 0.05
    persist: bool = False
    opp_id: str = ""
    seed: int = 0


def describe(cfg: PlayerConfig) -> str:
    return (f"[5.0] mode={cfg.mode} belief={cfg.belief_source} "
            f"tau={cfg.tau} step={cfg.step_cost} prior={cfg.prior} "
            f"type={'on' if cfg.type_posterior else 'off'} "
            f"style={cfg.style} diff={cfg.difficulty} noise={cfg.noise}")


class CouragePlayer:
    def __init__(self, cfg: Optional[PlayerConfig] = None):
        self.cfg = cfg or PlayerConfig()

        belief_like = None
        if self.cfg.belief_source == "encoder":
            path = _resolve(self.cfg.encoder_file)
            if os.path.exists(path):
                from .learning.seq_encoder import EncoderBelief, load_encoder
                model = load_encoder(path)
                belief_like = EncoderBelief(model, tau=self.cfg.tau,
                                            floor=self.cfg.floor,
                                            smooth=self.cfg.smooth)
            else:
                self.cfg.belief_source = "hand"  # 编码器缺失，回退

        type_model = None
        if self.cfg.type_posterior:
            from .learning.type_model import OpponentTypeModel
            type_model = OpponentTypeModel(
                mode=self.cfg.mode, decay=self.cfg.type_decay,
                eps=self.cfg.type_eps, smooth=self.cfg.type_smooth)
        context_model = None
        if self.cfg.context_file:
            path = _resolve(self.cfg.context_file)
            if os.path.exists(path):
                from .learning.context_model import ContextOpponentModel
                context_model = ContextOpponentModel.load(path)
        if type_model is not None and context_model is not None:
            from .learning.hybrid_model import HybridOpponentModel
            type_model = HybridOpponentModel(type_model, context_model)
        elif type_model is None and context_model is not None:
            type_model = context_model

        self.agent = OnlineBRAgent(
            settings=PlaySettings(self.cfg.difficulty, self.cfg.style, self.cfg.noise),
            mode=self.cfg.mode, gamma=self.cfg.gamma,
            decay=self.cfg.decay, alpha=self.cfg.alpha, smooth=self.cfg.smooth,
            step_cost=self.cfg.step_cost, br_sweeps=self.cfg.br_sweeps,
            persist=self.cfg.persist, prior=PRIORS.get(self.cfg.prior),
            qi_floor=self.cfg.qi_floor,
            # 类型/上下文模型已包含或学到的对手模型；再叠常数幽灵会重复计数。
            robust=(0.0 if type_model is not None else self.cfg.robust),
            zero_penalty=self.cfg.zero_penalty,
            round_total=self.cfg.round_total, noise0=self.cfg.noise0,
            urgency=self.cfg.urgency, anti_loop=self.cfg.anti_loop,
            loop_risk=self.cfg.loop_risk,
            opening_book=self._opening_book(),
            belief_like=belief_like, type_model=type_model,
            type_blend=self.cfg.type_blend, rng=random.Random(self.cfg.seed),
            gate=self.cfg.gate, mix_eta=self.cfg.mix_eta,
            mix_decay=self.cfg.mix_decay, mix_prior=self.cfg.mix_prior,
            mix_null=self.cfg.mix_null, mix_hand=self.cfg.mix_hand,
            channel_kappa=self.cfg.channel_kappa,
        )

        if self.cfg.persist and self.cfg.opp_id and self.cfg.belief_source == "hand":
            from .storage.belief_store import load_belief
            bt = load_belief(self.cfg.opp_id)
            if bt is not None:
                self.agent.belief = bt

    def _opening_book(self):
        """按风格取开局簿（五风格各一本）；opening='off' 时交给 BR。"""
        if self.cfg.opening == "off":
            return None
        from .core.rules import get_valid_moves
        from .policy.opening import load_style_book
        legal = get_valid_moves(0 if self.cfg.mode == 1 else 3, mode=self.cfg.mode)
        return load_style_book(legal, self.cfg.mode, self.cfg.style)

    # Agent 接口
    def act(self, own: int, opp: int) -> str:
        return self.agent.act(own, opp)

    def observe(self, own, opp, own_move, opp_move) -> None:
        self.agent.observe(own, opp, own_move, opp_move)

    def reset(self) -> None:
        self.agent.reset()

    def close(self) -> None:
        """持久化跨会话记忆（仅手写信念）。"""
        if self.cfg.persist and self.cfg.opp_id and self.cfg.belief_source == "hand":
            from .storage.belief_store import save_belief
            save_belief(self.cfg.opp_id, self.agent.belief)

    def __repr__(self) -> str:
        return describe(self.cfg)


def build_player(cfg: Optional[PlayerConfig] = None) -> CouragePlayer:
    return CouragePlayer(cfg)
