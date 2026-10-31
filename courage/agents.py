"""对局智能体与擂台。

Agent 统一视角：act(own_qi, opp_qi) / observe(own_qi, opp_qi, own_move, opp_move)。
这样任一智能体都从「自己的气数在前」出发，避免对手视角错位。

- BotAgent      ：包装脚本 bot（无状态或有 last_opp）
- TableAgent    ：固定策略表（均衡/BR 产物）
- ScriptedAgent ：任意脚本函数（如「先攻后守」测试）
- OnlineBRAgent ：核心——用 BeliefTracker 把历史序列压成因子，
                  每回合把因子转成对手分布并重解 BR（在线剥削）
"""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Protocol, Tuple

from .bots.scripts import sample_policy
from .core.rules import judge, next_qi, get_valid_moves
from .core.state import MODE_STANDARD, QI_MAX
from .learning.belief import BeliefTracker
from .learning.threat import blend_with_converter
from .policy.mix import blend, confidence
from .policy.settings import PlaySettings, apply_settings
from .solvers.best_response import solve_best_response
from .solvers.equilibrium import build_tables


class Agent(Protocol):
    def act(self, own: int, opp: int) -> str: ...
    def observe(self, own: int, opp: int, own_move: str, opp_move: str) -> None: ...
    def reset(self) -> None: ...


# 冷启动先验：无历史时对对手类别的先验（可视为 2.0/3.0 蒙特卡洛首招的形式化）。
PRIORS: Dict[str, Optional[Dict[str, float]]] = {
    "none": None,
    "balanced": {"attack": 3, "heavy": 2, "defense": 3, "dodge": 1, "exhale": 1, "noqi": 0},
    "aggressive": {"attack": 5, "heavy": 3, "defense": 1, "dodge": 0.5, "exhale": 0.5, "noqi": 0},
    "defensive": {"attack": 1, "heavy": 0.5, "defense": 6, "dodge": 3, "exhale": 1, "noqi": 0},
    "heavy": {"attack": 2, "heavy": 5, "defense": 2, "dodge": 0.5, "exhale": 1, "noqi": 0},
}


class BotAgent:
    def __init__(self, bot, mode: int = MODE_STANDARD, rng: Optional[random.Random] = None):
        self.bot = bot
        self.mode = mode
        self.rng = rng or random.Random()
        self.last_opp: Optional[str] = None

    def act(self, own, opp):
        return self.bot.act(own, opp, mode=self.mode,
                            last_opp=self.last_opp, rng=self.rng)

    def observe(self, own, opp, own_move, opp_move):
        self.last_opp = opp_move

    def reset(self):
        self.last_opp = None


class TableAgent:
    def __init__(self, table, settings: Optional[PlaySettings] = None,
                 mode: int = MODE_STANDARD, rng: Optional[random.Random] = None):
        self.table = table
        self.settings = settings or PlaySettings()
        self.mode = mode
        self.rng = rng or random.Random()

    def act(self, own, opp):
        legal = get_valid_moves(own, mode=self.mode)
        base = self.table.get((own, opp)) or {m: 1.0 / len(legal) for m in legal}
        return sample_policy(apply_settings(base, legal, self.settings), self.rng)

    def observe(self, *a):
        pass

    def reset(self):
        pass


class ScriptedAgent:
    def __init__(self, fn: Callable[[int, int, int], str],
                 rng: Optional[random.Random] = None):
        self.fn = fn
        self.rng = rng or random.Random()
        self.t = 0

    def act(self, own, opp):
        mv = self.fn(own, opp, self.t)
        self.t += 1
        return mv

    def observe(self, *a):
        pass

    def reset(self):
        self.t = 0


class OnlineBRAgent:
    """在线信念 → 每回合重解 BR 的剥削型智能体。"""

    def __init__(self, settings: Optional[PlaySettings] = None,
                 mode: int = MODE_STANDARD, gamma: float = 0.95,
                 decay: float = 0.85, alpha: float = 0.20, smooth: float = 0.05,
                 step_cost: float = 0.03, br_sweeps: int = 600, persist: bool = False,
                 prior: Optional[Dict[str, float]] = None, prior_strength: float = 3.0,
                 qi_floor: int = 0, floor_win: float = 1.0, temp: float = 0.0,
                 opening_book=None,
                 opp_open_stats: Optional[Dict[str, float]] = None,
                 eq_policy=None, conf_scale: float = 1.5, robust: float = 0.0,
                 belief_like=None, rng: Optional[random.Random] = None):
        self.settings = settings or PlaySettings()
        self.mode = mode
        self.gamma = gamma
        self.step_cost = step_cost
        self.br_sweeps = br_sweeps
        self.qi_floor = qi_floor
        self.floor_win = floor_win
        self.temp = temp
        self.opening_book = opening_book
        self.opp_open_stats = opp_open_stats or {}
        self.eq_policy = eq_policy          # 安全网：均衡策略（None=纯 BR）
        self.conf_scale = conf_scale
        self.robust = robust
        self._last_opening = None
        self.rng = rng or random.Random()
        self.persist = persist
        if belief_like is not None:
            self.belief = belief_like
        else:
            self.belief = BeliefTracker(decay=decay, alpha=alpha, smooth=smooth,
                                        prior=prior, prior_strength=prior_strength)
        self.T = build_tables(mode)
        self._all = [(qa, qp) for qa in range(QI_MAX + 1) for qp in range(QI_MAX + 1)]
        self._last_br = None

    def _opp_policy_from_belief(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        pol = {}
        # 「看不清对手」时才启用幽灵威胁；看清了就放开剥削
        beta = 0.0
        if self.robust > 0:
            try:
                vol = float(self.belief.factors().get("volatility", 1.0))
            except Exception:
                vol = 1.0
            beta = self.robust * vol
        for (qa, qp) in self._all:
            legal = get_valid_moves(qp, mode=self.mode)
            w = self.belief.opp_weights(legal, last_own=self.belief.last_own)
            if beta > 0:
                # 幽灵类型：理性转化者（对手会趁我无力时蓄气/收割）
                w = blend_with_converter(w, legal, get_valid_moves(qa, mode=self.mode), beta)
            pol[(qa, qp)] = w
        return pol

    def opp_opening_pred(self) -> Optional[Dict[str, float]]:
        """跨局统计出的「对手首招倾向」（归一化）。无数据返回 None。

        首局/首招本身就是一个可分析的组：同一对手若反复以某招开局，
        说明其开局偏好稳定，可作为对其首招的预测。
        """
        s = sum(self.opp_open_stats.values())
        if s <= 0:
            return None
        return {m: v / s for m, v in self.opp_open_stats.items() if v > 0}

    def act(self, own, opp):
        # 开局：无任何历史时按「蒙特卡洛开局簿」采样，保证首招随机且各招可达
        if self.opening_book is not None and getattr(self.belief, "steps", 0) == 0:
            legal = get_valid_moves(own, mode=self.mode)
            m = self.opening_book.sample(legal, self.rng,
                                         opp_open=self.opp_opening_pred())
            if m is not None:
                self._last_opening = m
                return m
        opp_pol = self._opp_policy_from_belief()
        br = solve_best_response(self.mode, opp_pol, gamma=self.gamma,
                                 max_sweeps=self.br_sweeps, tol=1e-5,
                                 step_cost=self.step_cost, tables=self.T,
                                 qi_floor=self.qi_floor, floor_win=self.floor_win,
                                 temp=self.temp)
        legal = get_valid_moves(own, mode=self.mode)
        base = br.policy.get((own, opp)) or {m: 1.0 / len(legal) for m in legal}
        # 安全网：无信息时用均衡混合（随机、不可被脚本化），有信息后转向 BR 剥削
        if self.eq_policy is not None:
            lam = confidence(getattr(self.belief, "steps", 0), self.conf_scale)
            base = blend(self.eq_policy.get((own, opp)), base, lam)
        return sample_policy(apply_settings(base, legal, self.settings), self.rng)

    def observe(self, own, opp, own_move, opp_move):
        self.belief.observe(own_move, opp_move)

    def reset(self):
        # 跨局记忆：persist=True 时保留信念，仅每局由上层决定是否清零
        self._last_opening = None
        if not self.persist:
            self.belief.reset()


def run_duel(ai: Agent, opp: Agent, mode: int = MODE_STANDARD,
             games: int = 200, cap: int = 200,
             rng: Optional[random.Random] = None,
             reset_per_game: bool = True,
             recorder=None, opp_label: str = "?") -> Dict[str, object]:
    """跑多局。reset_per_game=False 可实现跨局记忆（不清空 AI 信念）。
    recorder 若提供（有 .log(**rec)），每局落盘双方完整序列。"""
    rng = rng or random.Random()
    wins = losses = timeouts = 0
    rounds: List[int] = []
    results: List[int] = []
    for g in range(games):
        opp.reset()
        if reset_per_game:
            ai.reset()
        qa = qp = 3
        result = 0
        ai_seq: List[str] = []
        opp_seq: List[str] = []
        qi_seq: List[list] = []
        for rnd in range(1, cap + 1):
            a = ai.act(qa, qp)
            b = opp.act(qp, qa)
            ai_seq.append(a)
            opp_seq.append(b)
            qi_seq.append([qa, qp])
            w = judge(a, b)
            ai.observe(qa, qp, a, b)
            opp.observe(qp, qa, b, a)
            if w != 0:
                result = w
                rounds.append(rnd)
                break
            qa = next_qi(qa, a)
            qp = next_qi(qp, b)
        else:
            result = 0
            rounds.append(cap)
        results.append(result)
        if recorder is not None:
            recorder.log(game=g, opp=opp_label, mode=mode, result=result,
                         rounds=rounds[-1], ai_seq=ai_seq, opp_seq=opp_seq,
                         qi_seq=qi_seq)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            timeouts += 1
    return {"wins": wins, "losses": losses, "timeouts": timeouts,
            "rounds": rounds, "results": results, "games": games}
