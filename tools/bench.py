"""评测基准：逐决策 regret + 带置信区间的胜率。

为什么不用「80 局胜率」当验收标准
--------------------------------
1. 80 局的标准误约 4.5 个百分点，「80% vs 77% 不掉」完全在噪声里；
2. 胜率把三件事混在一起：对手模型准不准、开局分布好不好、破僵局机制有没有生效；
3. BR 只在乎 Q 的 argmax 结构，模型命中率 80% 也可能在决定性的那一格排反。

本工具的两把尺子
----------------
``regret``（主指标，零噪声）
    对每个对手，用 ``solvers/aug_mdp`` 在「对手真正依赖的相位」增广空间上求**精确
    价值上限** ``V*``，再在智能体每次决策的真实状态上比较::

        regret = max_a Q(s,a) - Σ_a π_agent(a)·Q(s,a)

    它是「这一次决策最多亏了多少」，可以直接累加、可以定位到具体状态。
    开局决策（走开局簿而非 BR）单独统计——那是**故意的安全成本**，
    和对手模型的误差不是一回事，混在一起会冤枉模型。

    ⚠ 两条解读上的坑，必须知道：
    1. regret 含一个**固定的塑形代价**（``step_cost``/``urgency``/``zero_penalty``
       让智能体优化的目标与真实价值不同，开局簿同理）。绝对值不是「模型误差」，
       只应在**同对手、跨模型**之间比较。
    2. **逐决策 regret 之和 ≠ V* − 实际收益。** 每次决策的基准都是「从当前状态
       用真值最优」，所以一步走错之后，后续状态又被重新记一次账（重复计数）。
       它可能超过 V*（现成的例子：``uniform`` 在风格 zoo 上 regret 1.578 > V* 0.725）。
       排序仍然可比，但不要把它读成「丢了多少胜率」。
``win``（辅指标）
    配对的确定性播种 + Wilson 95% 区间，只用来看方向，不用来分辨 3 个点。

覆盖：打法（pattern）与风格（style）是两类不同的考题
--------------------------------------------------
**打法**来自 ``courage/bots/opponents.py``：一个特定思路或定势（4 手套路、低气交替、
理性转化者…），依赖相位，用 ``needs`` 声明。含此前评测里没有入口的
``pressure``（分散压制）、``alternate_low``（低气交替）、以及 ``--start 1,1`` 的残局。
``strong`` 是集合外压力测试对手。缺点：13 条里有 9 条本来就在类型后验的剧本库内，
属于**类内测试**。

**风格**来自 ``--zoo N``：从 Dirichlet 采样的全局攻防偏好（``bots/style_profiles.py``）。
它没有剧本、不看历史，几乎不可能恰好等于某条脚本——这才是**分布外**考题，
也更接近真人（真人有一个倾向，但没有剧本）。

用法
----
    python -m tools.bench                                  # 4 个模型配置 × 全部打法
    python -m tools.bench --models hybrid,type --games 40
    python -m tools.bench --styles converter,pressure,alternate_low --start 1,1
    python -m tools.bench --zoo 12 --games 30 --start 1,1 --opening off
    python -m tools.bench --belief encoder --models hybrid   # 用 GRU 编码器当信念源
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.bots.opponents import OPPONENTS, OpponentAgent, names
from courage.bots.style_profiles import StyleOpponent, sample_zoo
from courage.core.rules import MOVE_TO_IDX, judge, next_qi
from courage.pipeline import PlayerConfig, build_player
from courage.solvers.aug_mdp import AugSolution, oracle_for

CONTEXT_FILE = "data/artifacts/context_all.json"
ENCODER_FILE = "data/artifacts/encoder_gru.json"


# ── 模型配置 ──

MIX_TUNE = {"eta": 1.0, "decay": 1.0, "prior": 0.02, "null": 0.4, "kappa": 0.0}


def model_config(name: str, belief: str = "hand",
                 opening: str = "safe",
                 kappa: Optional[float] = None) -> PlayerConfig:
    """只在对「对手模型通道」上做差异，其余保持生产默认。

    ``opening="off"`` 让 BR 决定首招——评测对手模型时应关掉开局簿，
    否则 (3,3) 开局永远被开局簿接管，模型一次决策机会都拿不到
    （vs converter/转化者这类一回合定胜负的对手尤其明显）。
    """
    base = dict(belief_source=belief, encoder_file=ENCODER_FILE, seed=0,
                opening=opening, gate="switch")
    if name == "hand":
        return PlayerConfig(type_posterior=False, context_file="", **base)
    if name == "type":
        return PlayerConfig(type_posterior=True, context_file="", **base)
    if name == "context":
        return PlayerConfig(type_posterior=False, context_file=CONTEXT_FILE, **base)
    if name == "hybrid":
        return PlayerConfig(type_posterior=True, context_file=CONTEXT_FILE, **base)
    mix_kw = dict(gate="mix", mix_eta=MIX_TUNE["eta"],
                  mix_decay=MIX_TUNE["decay"], mix_prior=MIX_TUNE["prior"],
                  mix_null=MIX_TUNE["null"],
                  channel_kappa=MIX_TUNE["kappa"] if kappa is None else float(kappa))
    if name == "mix":
        # 指数加权混合：type + context + hand + uniform 四条通道
        return PlayerConfig(type_posterior=True, context_file=CONTEXT_FILE,
                            mix_hand=True, **{**base, **mix_kw})
    if name == "uniform":
        # 基线：对手模型 = 均匀（等价于最大化最坏情况），只有 uniform 一条通道
        return PlayerConfig(type_posterior=False, context_file="", mix_hand=False,
                            **{**base, "gate": "mix"})
    if name == "mixnohand":
        # 消融：去掉手写信念通道，看它是否有独立贡献
        return PlayerConfig(type_posterior=True, context_file=CONTEXT_FILE,
                            mix_hand=False, **{**base, **mix_kw})
    raise SystemExit(f"未知模型配置: {name}（可选 hand/type/context/hybrid/mix/mixnohand）")


MODELS = ["hand", "type", "context", "hybrid", "mix"]


# ── 统计 ──

def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """二项比例的 Wilson 区间（比正态近似在小样本下可靠）。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


# ── 对局 ──

class Tally:
    """一个 (模型, 风格) 的累计统计。"""

    __slots__ = ("games", "wins", "losses", "timeouts", "rounds",
                 "reg_sum", "reg_n", "open_sum", "open_n", "reg_by_state",
                 "spread_sum", "spread_n", "flip_n",
                 "sx", "sy", "sxx", "syy", "sxy", "corr_n")

    def __init__(self) -> None:
        self.games = 0
        self.wins = 0
        self.losses = 0
        self.timeouts = 0
        self.rounds: List[int] = []
        self.reg_sum = 0.0
        self.reg_n = 0
        self.open_sum = 0.0
        self.open_n = 0
        self.reg_by_state: Dict[Tuple[int, int], float] = {}
        # 通道分歧追踪：spread = 该步各动作上「通道均值 Q − 最坏通道 Q」的均值
        self.spread_sum = 0.0
        self.spread_n = 0
        self.flip_n = 0                      # κ 改变了 argmax 的决策数
        self.sx = self.sy = 0.0
        self.sxx = self.syy = self.sxy = 0.0
        self.corr_n = 0

    def add_decision(self, spread: float, flip: bool, regret: float) -> None:
        self.spread_sum += spread
        self.spread_n += 1
        if flip:
            self.flip_n += 1
        self.sx += spread
        self.sy += regret
        self.sxx += spread * spread
        self.syy += regret * regret
        self.sxy += spread * regret
        self.corr_n += 1

    def add_result(self, result: int, rounds: int) -> None:
        self.games += 1
        self.rounds.append(rounds)
        if result == 1:
            self.wins += 1
        elif result == -1:
            self.losses += 1
        else:
            self.timeouts += 1

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def regret_per_decision(self) -> float:
        return self.reg_sum / self.reg_n if self.reg_n else 0.0

    @property
    def regret_per_game(self) -> float:
        """每局「留在桌上的价值」——跨模型可比的量。

        比 ``regret/决策`` 稳：不同模型的对局长度不同（赢越快的模型决策点越少
        且分布不同），按决策平均会被这个差异带偏。
        """
        return self.reg_sum / self.games if self.games else 0.0

    @property
    def opening_cost(self) -> float:
        return self.open_sum / self.open_n if self.open_n else 0.0

    # ── 通道分歧的定量追踪 ──
    @property
    def mean_spread(self) -> float:
        """平均分歧：通道之间对「这一手值多少」的意见差（Q 的取值域 [-1,1]，故 ∈[0,2]）。"""
        return self.spread_sum / self.spread_n if self.spread_n else 0.0

    @property
    def flip_rate(self) -> float:
        """κ 真的改了 argmax 的比例——分歧到了「会改判」的程度有多少。"""
        return self.flip_n / self.spread_n if self.spread_n else 0.0

    @property
    def corr_spread_regret(self) -> float:
        """分歧与本步 regret 的相关系数：验证「分歧大 ⇒ 亏得多」这个假设。"""
        n = self.corr_n
        if n < 3:
            return 0.0
        mx, my = self.sx / n, self.sy / n
        vx = self.sxx / n - mx * mx
        vy = self.syy / n - my * my
        if vx <= 1e-15 or vy <= 1e-15:
            return 0.0
        return (self.sxy / n - mx * my) / math.sqrt(vx * vy)

    def merge(self, other: "Tally") -> None:
        """把另一批对局的统计并进来（风格 zoo 要按"一批对手"聚合）。"""
        self.games += other.games
        self.wins += other.wins
        self.losses += other.losses
        self.timeouts += other.timeouts
        self.rounds.extend(other.rounds)
        self.reg_sum += other.reg_sum
        self.reg_n += other.reg_n
        self.open_sum += other.open_sum
        self.open_n += other.open_n
        for k, v in other.reg_by_state.items():
            self.reg_by_state[k] = self.reg_by_state.get(k, 0.0) + v
        self.spread_sum += other.spread_sum
        self.spread_n += other.spread_n
        self.flip_n += other.flip_n
        self.sx += other.sx
        self.sy += other.sy
        self.sxx += other.sxx
        self.syy += other.syy
        self.sxy += other.sxy
        self.corr_n += other.corr_n


def _seed_agent(agent, seed: int) -> None:
    target = getattr(agent, "agent", agent)
    if hasattr(target, "rng"):
        target.rng = random.Random(seed)


def play(ai, opp: OpponentAgent, oracle: Optional[AugSolution], cap: int,
         start: Tuple[int, int], seed: int, tally: Tally) -> None:
    """单局：与 ``agents.run_duel`` 语义一致，额外记录逐决策 regret。"""
    inner = getattr(ai, "agent", ai)
    ai.reset()
    opp.reset()
    qa, qp = start
    al = bl = t = 0
    t_max = oracle.tables.t_max if oracle is not None else 0
    result, rounds = 0, cap

    for rnd in range(1, cap + 1):
        a = ai.act(qa, qp)
        info = getattr(inner, "last_info", None)
        b = opp.act(qp, qa)

        if oracle is not None:
            si = oracle.tables.index(qa, qp, al, bl, t)
            q = oracle.q_row_at(si)
            va = oracle.tables.movesA[si]
            best = float(q.max())
            dist = (info or {}).get("dist")
            if dist:
                got = sum(float(p) * float(q[va.index(m)])
                          for m, p in dist.items() if m in va)
            else:
                got = float(q[va.index(a)])
            gap = best - got
            if (info or {}).get("opening"):
                tally.open_sum += gap
                tally.open_n += 1
            else:
                tally.reg_sum += gap
                tally.reg_n += 1
                k = (qa, qp)
                tally.reg_by_state[k] = tally.reg_by_state.get(k, 0.0) + gap
                sp = (info or {}).get("spread") or {}
                if sp:
                    tally.add_decision(sum(sp.values()) / len(sp),
                                       bool((info or {}).get("flip")), gap)

        w = judge(a, b)
        ai.observe(qa, qp, a, b)
        opp.observe(qp, qa, b, a)
        if w != 0:
            result, rounds = w, rnd
            break
        qa, qp = next_qi(qa, a), next_qi(qp, b)
        al, bl = MOVE_TO_IDX[a], MOVE_TO_IDX[b]
        t = min(t + 1, t_max)

    tally.add_result(result, rounds)


def bench_spec(cfg: PlayerConfig, spec, games: int, cap: int,
               start: Tuple[int, int], seed: int,
               oracle: Optional[AugSolution]) -> Tally:
    """对一个对手（打法 spec 或风格 spec）跑若干局。

    ``spec`` 只需有 ``policy(own, opp, t, own_last, opp_last, mode)`` 和 ``name``，
    所以 ``OpponentSpec``（打法）与 ``StyleOpponent``（风格）都能直接用。
    """
    ai = build_player(cfg)
    opp = OpponentAgent(spec, random.Random(0))
    tally = Tally()
    game_rng = random.Random(seed)
    for _ in range(games):
        gs = game_rng.randrange(1 << 30)
        _seed_agent(ai, gs * 2 + 1)
        _seed_agent(opp, gs * 2 + 2)
        play(ai, opp, oracle, cap, start, gs, tally)
    return tally


def bench_style(cfg: PlayerConfig, style_name: str, games: int, cap: int,
                start: Tuple[int, int], seed: int,
                oracle: Optional[AugSolution]) -> Tally:
    return bench_spec(cfg, OPPONENTS[style_name], games, cap, start, seed, oracle)


def bench_zoo(cfg: PlayerConfig, profiles, games: int, cap: int,
              start: Tuple[int, int], seed: int, with_regret: bool) -> Tuple[Tally, float]:
    """在**采样出来的风格**上跑——这才是分布外泛化测验。

    与 13 个手写打法的区别：那些写法里 9 个本来就在类型后验的类型集合内，
    属于类内测试。风格是参数族，采样出来的几乎不可能恰好等于某一条脚本，
    所以它测的是「对手有倾向但不在我的剧本库里」这种情况——也就是真人。
    """
    total = Tally()
    v_star = 0.0
    for i, prof in enumerate(profiles):
        spec = StyleOpponent(prof, name=f"zoo{i}")
        oracle = oracle_for(spec) if with_regret else None
        if oracle is not None:
            try:
                v_star += oracle.value(*_oracle_state(oracle, start))
            except KeyError:
                pass
        total.merge(bench_spec(cfg, spec, games, cap, start, seed + i * 7919, oracle))
    return total, (v_star / len(profiles) if profiles else 0.0)


# ── 报表 ──

def _phase_label(needs: Tuple[str, ...]) -> str:
    return "+".join({"al": "我上招", "bl": "对上招", "t": "回合"}[n] for n in needs) or "-"


def _row(model: str, name: str, label: str, needs: str, v_star: str,
         tally: Tally, ood: bool, seconds: float) -> dict:
    lo, hi = wilson(tally.wins, tally.games)
    return {
        "model": model,
        "style": name,
        "label": label,
        "needs": needs,
        "V_star": v_star,
        "ood": ood,
        "regret_per_decision": tally.regret_per_decision,
        "regret_per_game": tally.regret_per_game,
        "regret_total": tally.reg_sum,
        "regret_decisions": tally.reg_n,
        "opening_cost": tally.opening_cost,
        "win": tally.win_rate,
        "win_lo": lo,
        "win_hi": hi,
        "wins": tally.wins,
        "losses": tally.losses,
        "timeouts": tally.timeouts,
        "games": tally.games,
        "rounds": statistics.mean(tally.rounds) if tally.rounds else 0.0,
        "top_regret_state": _top_state(tally.reg_by_state),
        "mean_spread": tally.mean_spread,
        "flip_rate": tally.flip_rate,
        "corr_spread_regret": tally.corr_spread_regret,
        "spread_decisions": tally.spread_n,
        "seconds": round(seconds, 2),
    }


def rows_for(model: str, belief: str, style_list: List[str], games: int,
             cap: int, start: Tuple[int, int], seed: int,
             with_regret: bool, opening: str = "safe", zoo: int = 0,
             zoo_alpha: float = 1.0, zoo_sharp: float = 2.0) -> List[dict]:
    out = []
    cfg = model_config(model, belief, opening)
    for name in style_list:
        spec = OPPONENTS[name]
        oracle = oracle_for(spec) if with_regret else None
        t0 = time.time()
        tally = bench_style(cfg, name, games, cap, start, seed, oracle)
        star = ""
        if oracle is not None:
            try:
                star = f"{oracle.value(*_oracle_state(oracle, start)):.3f}"
            except KeyError:
                star = "?"
        out.append(_row(model, name, spec.label,
                        _phase_label(oracle.tables.needs) if oracle is not None else "-",
                        star, tally, spec.ood, time.time() - t0))

    if zoo > 0:
        profiles = sample_zoo(zoo, seed=seed, alpha=zoo_alpha, sharpness=zoo_sharp)
        t0 = time.time()
        tally, v_star = bench_zoo(cfg, profiles, games, cap, start, seed, with_regret)
        out.append(_row(model, "风格zoo", f"采样风格×{zoo}",
                        "-", f"{v_star:.3f}" if with_regret else "",
                        tally, True, time.time() - t0))
    return out


def _oracle_state(oracle: AugSolution, start: Tuple[int, int]) -> Tuple[int, ...]:
    """起始状态在增广空间里的坐标（t=0、无上一招）。"""
    n = oracle.tables.needs
    vals = [(start[0]), (start[1])]
    vals.append(0 if "al" in n else 0)
    vals.append(0 if "bl" in n else 0)
    vals.append(0 if "t" in n else 0)
    return tuple(vals)  # type: ignore[return-value]


def print_track(rows: List[dict], model: str, kappa: float) -> None:
    """通道分歧的定量追踪表。

    三个量：
      分歧   = 该步各动作上「通道均值 Q − 最坏通道 Q」的均值（Q∈[-1,1]，故 ∈[0,2]）
      改判%  = κ 真的改变了 argmax 的决策比例 —— 「分歧到了会改变选招」有多少
      r      = 分歧 与 本步 regret 的相关系数；若显著为正，说明「分歧大 ⇒ 亏得多」，
               那么按分歧定价就是对的（而不是拍脑袋加一个风险系数）
    """
    tracked = [r for r in rows if r["spread_decisions"] > 0]
    if not tracked:
        return
    print(f"\n--- 通道分歧追踪（{model}，κ={kappa:g}）---")
    cols = [("对手", 14, "left"), ("分歧", 7, "right"), ("改判%", 7, "right"),
            ("r(分歧,损失)", 13, "right"), ("分歧最大态", 12, "left")]
    head = "".join(_pad(n, w, a) for n, w, a in cols)
    print(head)
    print("-" * _w(head))
    for r in tracked:
        vals = [r["style"], f"{r['mean_spread']:.3f}",
                f"{r['flip_rate']*100:.0f}%",
                f"{r['corr_spread_regret']:+.2f}", r["top_regret_state"]]
        print("".join(_pad(v, w, a) for v, (_, w, a) in zip(vals, cols)))


def _top_state(reg_by_state: Dict[Tuple[int, int], float]):
    if not reg_by_state:
        return ""
    (qa, qp), v = max(reg_by_state.items(), key=lambda kv: kv[1])
    return f"{qa},{qp}:{v:.2f}"


def _w(s: str) -> int:
    """字符串显示宽度（CJK 记 2）。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in s)


def _pad(s: str, n: int, align: str = "left") -> str:
    gap = max(0, n - _w(str(s)))
    s = str(s)
    return (" " * gap + s) if align == "right" else (s + " " * gap)


def print_block(rows: List[dict], model: str, belief: str, games: int,
                start: Tuple[int, int], opening: str = "safe") -> None:
    print(f"\n=== 模型 {model:<8} 信念源 {belief:<7} 开局 {opening:<4} "
          f"{games} 局/风格 起始 {start[0]},{start[1]} ===")
    cols = [("风格", 14, "left"), ("标签", 16, "left"), ("相位", 9, "left"),
            ("V*", 6, "right"), ("regret/局", 10, "right"),
            ("regret/决策", 12, "right"), ("决策", 6, "right"),
            ("开局代价", 9, "right"), ("胜率", 7, "right"),
            ("95%CI", 12, "right"), ("回合", 6, "right"), ("超时", 5, "right")]
    head = "".join(_pad(n, w, a) for n, w, a in cols)
    print(head)
    print("-" * _w(head))

    for r in rows:
        ci = f"[{r['win_lo']*100:.0f},{r['win_hi']*100:.0f}]"
        vals = [r["style"], r["label"], r["needs"], r["V_star"],
                f"{r['regret_per_game']:.4f}",
                f"{r['regret_per_decision']:.4f}", str(r["regret_decisions"]),
                f"{r['opening_cost']:.4f}",
                f"{r['win']*100:.0f}%", ci,
                f"{r['rounds']:.1f}", str(r["timeouts"])]
        print("".join(_pad(v, w, a) for v, (_, w, a) in zip(vals, cols)))

    if rows:
        tot_r = sum(r["regret_total"] for r in rows)
        tot_n = sum(r["regret_decisions"] for r in rows)
        gg = sum(r["games"] for r in rows)
        gw = sum(r["wins"] for r in rows)
        lo, hi = wilson(gw, gg)
        vals = ["合计", "", "", "",
                f"{(tot_r/gg if gg else 0):.4f}",
                f"{(tot_r/tot_n if tot_n else 0):.4f}", str(tot_n), "",
                f"{gw/gg*100 if gg else 0:.0f}%",
                f"[{lo*100:.0f},{hi*100:.0f}]", "", ""]
        print("".join(_pad(v, w, a) for v, (_, w, a) in zip(vals, cols)))


def main() -> None:
    ap = argparse.ArgumentParser(description="勇气 · 决策级评测基准")
    ap.add_argument("--models", default=",".join(MODELS),
                    help="逗号分隔：hand/type/context/hybrid")
    ap.add_argument("--styles", default="",
                    help="逗号分隔的风格名（默认全部，含集合外）")
    ap.add_argument("--belief", default="hand", choices=["hand", "encoder"],
                    help="信念来源；encoder 需要 data/artifacts/encoder_gru.json")
    ap.add_argument("--games", type=int, default=40, help="每个风格的局数")
    ap.add_argument("--cap", type=int, default=99, help="兜底回合上限（不作结算上限）")
    ap.add_argument("--start", default="3,3", help="起始气数，如 1,1 表示低气残局")
    ap.add_argument("--opening", default="safe", choices=["safe", "off"],
                    help="safe=生产开局簿（首招不由 BR 决定）/ off=首招交给 BR（评模型用）")
    ap.add_argument("--mix-eta", dest="mix_eta", type=float, default=1.0,
                    help="指数加权的温度（1=贝叶斯模型平均，>1 更快分离）")
    ap.add_argument("--mix-decay", dest="mix_decay", type=float, default=1.0,
                    help="通道权重的遗忘率（<1 适应非平稳对手）")
    ap.add_argument("--mix-prior", dest="mix_prior", type=float, default=0.02,
                    help="通道权重地板，防止一条通道被永久打死")
    ap.add_argument("--mix-null", dest="mix_null", type=float, default=0.4,
                    help="uniform 通道（无结构零假设）的先验权重")
    ap.add_argument("--kappa", type=float, default=0.0,
                    help="通道分歧定价：Q=(1-κ)·E_c[Q_c]+κ·min_c Q_c（0=风险中性）")
    ap.add_argument("--zoo", type=int, default=0,
                    help="额外采样 N 个**风格**对手（分布外泛化测验）；0=关")
    ap.add_argument("--zoo-alpha", dest="zoo_alpha", type=float, default=1.0,
                    help="风格采样的 Dirichlet 浓度（越小越极端）")
    ap.add_argument("--zoo-sharp", dest="zoo_sharp", type=float, default=2.0,
                    help="采样后对风格做幂次锐化（1=不动）")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-regret", dest="no_regret", action="store_true",
                    help="跳过精确 oracle（只报胜率，胜率≈噪声大）")
    ap.add_argument("--json", default="", help="把结果另存为 JSON")
    args = ap.parse_args()

    style_list = ([s.strip() for s in args.styles.split(",") if s.strip()]
                  if args.styles else names(include_ood=True))
    unknown = [s for s in style_list if s not in OPPONENTS]
    if unknown:
        raise SystemExit(f"未知风格: {unknown}；可选: {sorted(OPPONENTS)}")
    start = tuple(int(x) for x in args.start.split(","))  # type: ignore[assignment]
    if len(start) != 2:
        raise SystemExit("--start 需要两个整数，如 3,3")

    MIX_TUNE.update(eta=args.mix_eta, decay=args.mix_decay,
                    prior=args.mix_prior, null=args.mix_null, kappa=args.kappa)
    if args.belief == "encoder":
        p = os.path.join(_ROOT, ENCODER_FILE)
        if not os.path.exists(p):
            raise SystemExit(f"缺编码器: {p}；先跑 tools/repro.py --build 或改用 --belief hand")

    t0 = time.time()
    all_rows: List[dict] = []
    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        rows = rows_for(model, args.belief, style_list, args.games, args.cap,
                        start, args.seed, with_regret=not args.no_regret,
                        opening=args.opening, zoo=args.zoo,
                        zoo_alpha=args.zoo_alpha, zoo_sharp=args.zoo_sharp)
        print_block(rows, model, args.belief, args.games, start, args.opening)
        print_track(rows, model, args.kappa)
        all_rows.extend(rows)

    row_names = style_list + (["风格zoo"] if args.zoo > 0 else [])
    # 跨模型的直接对比（同风格、同种子 ⇒ 可配对）
    if len(all_rows) and not args.no_regret:
        print("\n=== 对手模型通道对比（regret/局，越小越好）===")
        models = [m.strip() for m in args.models.split(",") if m.strip()]
        head = _pad("对手", 14) + "".join(_pad(m, 12, "right") for m in models)
        print(head)
        print("-" * _w(head))
        for name in row_names:
            cells = []
            ood = False
            for m in models:
                r = next((x for x in all_rows
                          if x["model"] == m and x["style"] == name), None)
                if r is None:
                    cells.append(_pad("-", 12, "right"))
                    continue
                ood = ood or bool(r.get("ood"))
                cells.append(_pad(f"{r['regret_per_game']:.3f}", 12, "right"))
            print(_pad(name + ("*" if ood else ""), 14) + "".join(cells))
        print("  带 * = 分布外：不在类型后验的剧本库内（strong / 风格zoo）")
        print("  风格zoo = 从 Dirichlet 采样的**风格**（全局攻防偏好），")
        print("  与 13 个手写**打法**不同，它测的是「对手有倾向但没有剧本」——更接近真人")
        zero_dec = [(r["model"], r["style"]) for r in all_rows
                    if r["regret_decisions"] == 0]
        if zero_dec:
            print(f"  ⚠ 这些组合没有任何 BR 决策（全局被开局簿接管）：{zero_dec[:4]}")

    print(f"\n用时 {time.time()-t0:.1f}s")

    if args.json:
        path = args.json if os.path.isabs(args.json) else os.path.join(_ROOT, args.json)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "rows": all_rows}, f,
                      ensure_ascii=False, indent=2)
        print(f"written {path}")


if __name__ == "__main__":
    main()
