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
                 death_penalty: float = 0.0, kappa: float = 0.0,
                 zero_penalty: float = 0.0,
                 opening_book=None,
                 opp_open_stats: Optional[Dict[str, float]] = None,
                 eq_policy=None, conf_scale: float = 1.5, robust: float = 0.0,
                 eq_mix: Optional[float] = None,
                 round_total: int = 15, noise0: float = 0.0,
                 urgency: float = 3.0, anti_loop: bool = True, loop_risk: float = 0.9,
                 belief_like=None, type_model=None, type_blend: float = 0.0,
                 gate: str = "switch", mix_eta: float = 1.0,
                 mix_decay: float = 1.0, mix_prior: float = 0.02,
                 mix_null: float = 0.4, mix_hand: bool = False,
                 rng: Optional[random.Random] = None):
        self.settings = settings or PlaySettings()
        self.mode = mode
        self.gamma = gamma
        self.step_cost = step_cost
        self.br_sweeps = br_sweeps
        self.qi_floor = qi_floor
        self.floor_win = floor_win
        self.temp = temp
        self.death_penalty = death_penalty
        self.kappa = kappa
        self.zero_penalty = zero_penalty
        self.opening_book = opening_book
        self.opp_open_stats = opp_open_stats or {}
        # 类型后验：belief 之外的另一条对手模型通道；type_blend=0 时完全关闭。
        self.type_model = type_model
        self.type_blend = min(1.0, max(0.0, type_blend))
        self.eq_policy = eq_policy          # 安全网：均衡策略（None=纯 BR）
        self.conf_scale = conf_scale
        self.eq_mix = eq_mix
        self.robust = robust
        # ── 轮数退火（round annealing）──
        # 实测死锁：双方都按确定性 argmax 出招时会形成镜像循环
        #   (0,0) 吐气 → (1,1) 馒头 → (0,0) 吐气 → …   可以无限平局
        # 根因是"策略比重"恒定：前几轮信息量极少，π 却已经和后期一样"自信"，
        # 于是双方一拍即合同步循环；而 step_cost 恒定，也让"再来一回合"不痛。
        # 所以三个量都随进度 p = 已过回合/预期长度 变化：
        #   noise(p)：p=0 时 0.30（多随机、早探路）→ p=1 时 0（果断执行策略）
        #   step_cost(p)：0.03 → 0.03·(1+urgency)（越拖越亏，逼出破局手）
        self.round_total = max(1, int(round_total))
        self.noise0 = noise0
        self.urgency = urgency
        self.anti_loop = anti_loop
        self.loop_risk = loop_risk
        self._hist = []          # [(state, our_move)]，用于周期检测
        self._last_opening = None
        self._start_qi = 0 if mode == 1 else 3
        self.rng = rng or random.Random()
        self.persist = persist
        if belief_like is not None:
            self.belief = belief_like
        else:
            self.belief = BeliefTracker(decay=decay, alpha=alpha, smooth=smooth,
                                        prior=prior, prior_strength=prior_strength)
        # 对手模型装配方式：
        #   "switch" = 硬切换（HybridOpponentModel，旧行为，保留做 A/B）
        #   "mix"    = 按预测似然指数加权后混合（MixtureOpponentModel）
        self.gate = gate
        if gate == "mix":
            self._install_mixture(mix_eta, mix_decay, mix_prior, mix_null,
                                  mix_hand, mode)
        self.T = build_tables(mode)
        self._all = [(qa, qp) for qa in range(QI_MAX + 1) for qp in range(QI_MAX + 1)]
        self._last_br = None
        # 评测用：最近一次决策的完整信息（分布/选择/是否为开局）。
        # tools/bench.py 靠它算「逐决策 regret」——用分布算期望，去掉采样噪声。
        self.last_info: Optional[Dict[str, object]] = None

    def _install_mixture(self, eta: float, decay: float, prior_mix: float,
                         null_weight: float, include_hand: bool,
                         mode: int) -> None:
        """把 type / context / hand / uniform 收成一条指数加权混合通道。

        如果上层传进来的是 Hybrid（硬切换），这里把它**拆开**成两条独立通道——
        混合的前提是各通道独立给分，而不是让其中一条先做路由。
        """
        from .learning.mixture_model import (BeliefChannel, MixtureOpponentModel,
                                             UniformChannel)

        channels: Dict[str, object] = {}
        tm = self.type_model
        if tm is not None:
            inner_t = getattr(tm, "type_model", None)
            inner_c = getattr(tm, "context_model", None)
            if inner_t is not None or inner_c is not None:      # Hybrid
                if inner_t is not None:
                    channels["type"] = inner_t
                if inner_c is not None:
                    channels["context"] = inner_c
            else:
                channels["type"] = tm
        if include_hand:
            channels["hand"] = BeliefChannel(self.belief)
        channels["uniform"] = UniformChannel()
        self.type_model = MixtureOpponentModel(
            channels, eta=eta, decay=decay, prior_mix=prior_mix,
            null_weight=null_weight, mode=mode)
        self.type_blend = 1.0      # 混合通道已含 hand，外层的旧混合通道让位

    def _opp_policy_from_belief(self) -> Dict[Tuple[int, int], Dict[str, float]]:
        pol = {}
        # 幽灵威胁权重：按**观测样本量**衰减，但**永不归零**。
        # 旧版用 volatility 调制，等价于"看得越清越不设防"——于是对手只要先演
        # 几手温顺牌，AI 就把"它会兑现优势"这个假设完全丢掉，随即被反杀。
        # 现在：样本少 → 接近 robust（保守）；样本多 → 降到 1/4，保留最低限度的
        # 「对手可能变招」的假设。这才是对自适应对手的正确定价。
        beta = 0.0
        if self.robust > 0:
            n = float(getattr(self.belief, "steps", 0) or 0)
            conf = n / (n + 6.0)          # 0 → 0; 6 步 → 0.5; 60 步 → 0.91
            beta = self.robust * (1.0 - 0.75 * conf)
        for (qa, qp) in self._all:
            legal = get_valid_moves(qp, mode=self.mode)
            w = self.belief.opp_weights(legal, last_own=self.belief.last_own)
            if self.type_model is not None and self.type_blend > 0:
                wt = self.type_model.opp_weights(
                    qa, qp, legal, last_own=self.belief.last_own)
                w = blend(wt, w, 1.0 - self.type_blend)
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

    def _round_dyn(self) -> tuple:
        """轮数退火：返回 (noise, step_cost)，都随对局进度变化。

        p = 已过回合 / (预期回合数-1)，截断到 [0,1]。
        - noise：前几轮大（避免双方早早同步成镜像循环），末段收敛到 0（执行策略）。
        - step_cost：末段放大 (1+urgency) 倍（每多拖一回合代价更高，逼出破局手）。
        """
        r = float(getattr(self.belief, "steps", 0) or 0)
        p = min(1.0, max(0.0, r / max(self.round_total - 1, 1)))
        noise = self.noise0 * (1.0 - p)
        step = self.step_cost * (1.0 + self.urgency * p)
        return noise, step

    # ── 循环检测（直接防局部死锁，比全局随机精确）──
    #
    # 死锁实测发生在 (1,1)：双方都选 cost=0 的「防御/跳」，气数永远不变，
    # 于是局面精确复现。全局 noise 能打破它，但代价是把正确招也随机掉
    # （实测 (0,0) 该出吐气的定理级最优，被 noise 污染成 82%）。
    #
    # 正确做法是**只在检测到周期时**主动偏离：把"上一周期里在同一个状态
    # 已经用过的招"排除，优先选能改变气数的那一手。这样平时策略完全不变，
    # 只在真正开始打转时出手。
    def _detect_cycle(self, max_period: int = 5) -> int:
        """返回最近序列的周期长度；无周期返回 0。"""
        h = self._hist
        n = len(h)
        for p in range(1, max_period + 1):
            if n >= 2 * p and h[-p:] == h[-2 * p:-p]:
                return p
        return 0

    def _loop_break(self, own: int, opp: int, legal: List[str],
                    base: Dict[str, float]) -> Optional[str]:
        """若陷入周期，返回一个打破僵局的招；否则 None。"""
        if not self.anti_loop:
            return None
        p = self._detect_cycle()
        if p <= 0:
            return None
        used = {m for (s, m) in self._hist[-p:] if s == (own, opp)}
        alt = [m for m in legal if m not in used]
        if not alt:
            # 周期里已把这状态的所有招都用过了 → 换"能改气数"的招
            alt = [m for m in legal if next_qi(own, m) != own]
        if not alt:
            return None
        # 破僵局时在「冒险(改气数)」与「保守」之间按 loop_risk 概率二选一：
        # 总是冒险会被"固定第二手"的脚本按顺序吃掉（实测 2手套路 5/25 -> 20/10）。
        progress = [m for m in alt if next_qi(own, m) != own]
        safe = [m for m in alt if next_qi(own, m) == own]
        if progress and safe:
            pool = progress if self.rng.random() < self.loop_risk else safe
        else:
            pool = progress or safe or alt
        return max(pool, key=lambda m: base.get(m, 0.0))

    def _round_dyn_unused(self):
        pass

    def act(self, own, opp):
        # 开局：仅在真正的开局状态（双方起始气）按「开局簿」采样，保证首招随机且各招可达。
        # 注意不能只看 steps==0——若首回合发生在 (0,0)（无气拨模式）或异常状态，
        # 套用 (3,3) 的开局分布会把「防御」硬塞进 0 气局面（实测该处吐气才是唯一最优）。
        if (self.opening_book is not None and getattr(self.belief, "steps", 0) == 0
                and own == self._start_qi and opp == self._start_qi):
            legal = get_valid_moves(own, mode=self.mode)
            odist = self.opening_book.dist(legal, opp_open=self.opp_opening_pred())
            m = sample_policy(odist, self.rng) if odist else None
            if m is not None:
                self._last_opening = m
                self._hist.append(((own, opp), m))
                self.last_info = {"state": (own, opp), "dist": odist,
                                  "move": m, "opening": True}
                return m
        noise, step = self._round_dyn()
        opp_pol = self._opp_policy_from_belief()
        br = solve_best_response(self.mode, opp_pol, gamma=self.gamma,
                                 max_sweeps=self.br_sweeps, tol=1e-5,
                                 step_cost=step, tables=self.T,
                                 qi_floor=self.qi_floor, floor_win=self.floor_win,
                                 temp=self.temp, death_penalty=self.death_penalty,
                                 kappa=self.kappa, zero_penalty=self.zero_penalty)
        legal = get_valid_moves(own, mode=self.mode)
        base = br.policy.get((own, opp)) or {m: 1.0 / len(legal) for m in legal}
        # 安全网：无信息时用均衡混合（随机、不可被脚本化），有信息后转向 BR 剥削
        if self.eq_policy is not None:
            if self.eq_mix is not None:
                lam = self.eq_mix        # 固定混合比（消融用）
            else:
                lam = confidence(getattr(self.belief, "steps", 0), self.conf_scale)
            base = blend(self.eq_policy.get((own, opp)), base, lam)
        dist = apply_settings(base, legal,
                              PlaySettings(self.settings.difficulty,
                                           self.settings.style,
                                           max(self.settings.noise, noise)))
        # 循环检测：只在真的开始打转时偏离（平时完全不干预）
        forced = self._loop_break(own, opp, legal, dist)
        m = forced if forced is not None else sample_policy(dist, self.rng)
        self._hist.append(((own, opp), m))
        if len(self._hist) > 24:
            self._hist.pop(0)
        self.last_info = {"state": (own, opp), "dist": dist, "move": m,
                          "opening": False, "loop_break": forced is not None,
                          "step_cost": step, "br": br, "opp_policy": opp_pol}
        return m

    def observe(self, own, opp, own_move, opp_move):
        # 顺序有讲究：先让对手模型**在更新前**给各通道打分（混合模型要算预测似然），
        # 再更新手写信念。反过来会让 hand 通道用「刚看到的那一手」给自己打分。
        if self.type_model is not None:
            self.type_model.observe(own, opp, own_move, opp_move)
        self.belief.observe(own_move, opp_move)

    def reset(self):
        # 跨局记忆：persist=True 时保留信念，仅每局由上层决定是否清零
        self._last_opening = None
        self._hist = []
        self.last_info = None
        if self.type_model is not None:
            self.type_model.reset()
        if not self.persist:
            self.belief.reset()


def _seed_agent(agent, seed: int) -> None:
    """把每局的随机流固定下来；兼容 CouragePlayer 这类包装对象。"""
    target = getattr(agent, "agent", agent)
    if hasattr(target, "rng"):
        target.rng = random.Random(seed)


def run_duel(ai: Agent, opp: Agent, mode: int = MODE_STANDARD,
             games: int = 200, cap: int = 99,
             rng: Optional[random.Random] = None,
             seed: Optional[int] = None,
             reset_per_game: bool = True,
             recorder=None, opp_label: str = "?") -> Dict[str, object]:
    """跑多局。reset_per_game=False 可实现跨局记忆（不清空 AI 信念）。
    recorder 若提供（有 .log(**rec)），每局落盘双方完整序列。

    确定性：给出 seed（或 rng）后，每局都用独立的、可复现的随机流分别
    播种 AI 与对手；这样同配置重跑结果逐局一致，也便于 N 个模型做配对评测。
    cap=99 只是兜底（防死循环），不作为产品结算回合上限。
    """
    game_rng = random.Random(seed) if seed is not None else rng
    wins = losses = timeouts = 0
    rounds: List[int] = []
    results: List[int] = []
    game_seeds: List[int] = []
    for g in range(games):
        opp.reset()
        if reset_per_game:
            ai.reset()
        if game_rng is not None:
            gs = game_rng.randrange(1 << 30)
            game_seeds.append(gs)
            _seed_agent(ai, gs * 2 + 1)
            _seed_agent(opp, gs * 2 + 2)
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
            rec = dict(game=g, opp=opp_label, mode=mode, result=result,
                       rounds=rounds[-1], ai_seq=ai_seq, opp_seq=opp_seq,
                       qi_seq=qi_seq)
            if game_seeds:
                rec["seed"] = game_seeds[-1]
            recorder.log(**rec)
        if result == 1:
            wins += 1
        elif result == -1:
            losses += 1
        else:
            timeouts += 1
    return {"wins": wins, "losses": losses, "timeouts": timeouts,
            "rounds": rounds, "results": results, "games": games,
            "seeds": game_seeds, "cap": cap}
