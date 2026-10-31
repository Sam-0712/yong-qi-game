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

        self.agent = OnlineBRAgent(
            settings=PlaySettings(self.cfg.difficulty, self.cfg.style, self.cfg.noise),
            mode=self.cfg.mode, gamma=self.cfg.gamma,
            decay=self.cfg.decay, alpha=self.cfg.alpha, smooth=self.cfg.smooth,
            step_cost=self.cfg.step_cost, br_sweeps=self.cfg.br_sweeps,
            persist=self.cfg.persist, prior=PRIORS.get(self.cfg.prior),
            belief_like=belief_like, rng=random.Random(self.cfg.seed),
        )

        if self.cfg.persist and self.cfg.opp_id and self.cfg.belief_source == "hand":
            from .storage.belief_store import load_belief
            bt = load_belief(self.cfg.opp_id)
            if bt is not None:
                self.agent.belief = bt

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
