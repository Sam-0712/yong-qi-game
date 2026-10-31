"""实验性模块：用「最大可达胜率 W」直接做决策（试图取代 step_cost）。

**结论（实测）：不能整体取代 step_cost，不建议作为默认策略。**
- 理论上：纯可达概率的 Bellman 方程 W = max(..., W) 有大量不动点——"原地防守"
  与最优值等价（自环），贪心会选出"永不前进"的次优策略。必须额外加"反拖延"
  约束（本文件的 slack）才能跑通，而 slack 又会在别的对手上反噬。
- 实测（50 局/对手，手写信念，slack=0.15）：套路 66%→100%、龟缩 100%/24R、
  防守 90%、但转化者 48%→14%、随机 78%→60%。实现细节（迭代数/是否热启动）
  会显著改变结果，说明该目标函数对策略过于敏感。
- 真正有价值的用法：把 W 当作**校准/评估**工具（开局、难度、风格），
  以及用 W(0,·) 低这一点来论证"0 气必败"，而不是替换每回合成本。

以下保留实现，供后续研究（例如"最大可达 + 最短期望时长"的字典序目标）。


思路：
- step_cost 原本只是"别无限平局"的粗糙代理——它每回合扣一个常数，但实测
  气数的价值是**阶跃**的（4 气破 def3、5 气破一切），线性惩罚既压不住长局、
  又会在低气时误伤。
- 胜率面 W(s) 本身就是"这个状态最终能赢的概率"，于是可以直接对它做策略迭代：
      W      = 解吸收链(π, μ)
      Q(s,a) = Σ_b μ(b|s)[ 1 若胜 | 0 若负 | W(s') 否则 ]
      π'     = argmax_a Q(s,a)
  没有折扣、没有 step_cost；也不会选"把自己拖进 0 气"的招（W(0,·) 本身低）。
- 计算量：W 只依赖 (qa,qp)，一次 36×36 线性求解 + 一轮贪心，比 600 次扫的 BR 更小。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..bots.scripts import sample_policy
from ..core.rules import get_valid_moves, judge, next_qi
from ..solvers.winrate import STATES, solve_win_surface
from .settings import apply_settings


def _q_from_W(W: np.ndarray, mu: Dict[Tuple[int, int], Dict[str, float]],
              qa: int, qp: int) -> Tuple[list, Dict[str, float]]:
    legal = get_valid_moves(qa, mode=0)
    dist = mu.get((qa, qp)) or {}
    q: Dict[str, float] = {}
    for a in legal:
        v = 0.0
        for b, pb in dist.items():
            if pb <= 0:
                continue
            w = judge(a, b)
            if w != 0:
                v += pb * (1.0 if w == 1 else 0.0)
            else:
                v += pb * W[next_qi(qa, a), next_qi(qp, b)]
        q[a] = v
    return legal, q


def solve_reachability(mu: Dict[Tuple[int, int], Dict[str, float]],
                       iters: int = 120, tol: float = 1e-9,
                       tables=None, W0: Optional[np.ndarray] = None) -> np.ndarray:
    """最大可达概率 W*（"最终能赢"的概率）——向量化值迭代。

        W(s) = max_a Σ_b μ(b|s) · [ 1 若胜 | 0 若负 | W(s') 否则 ]

    与"先解某策略的 W 再贪心"不同：自环（防御→同状态）不会让 W 上升，
    所以"反正本来就赢、不必动手"的假象不成立。
    初值 0（悲观），单调上升，收敛到最大可达概率；给 W0 可跨回合热启动。
    """
    from ..solvers.best_response import _opp_matrix
    from ..solvers.equilibrium import build_tables
    T = tables if tables is not None else build_tables(0)
    Y = _opp_matrix(T, mu)
    S = len(T.states)
    W = np.zeros(S) if W0 is None else np.asarray(W0, dtype=np.float64).reshape(S).copy()
    allow = T.maskA > 0
    for _ in range(iters):
        M = np.where(T.term, T.pay, W[T.succ])
        Q = np.einsum("sab,sb->sa", M, Y)
        Wn = np.where(allow, Q, -1e9).max(1)
        if float(np.abs(Wn - W).max()) < tol:
            W = Wn
            break
        W = Wn
    return W.reshape(6, 6)


def policy_from_W(W: np.ndarray, mu: Dict[Tuple[int, int], Dict[str, float]],
                  slack: float = 0.0
                  ) -> Dict[Tuple[int, int], Dict[str, float]]:
    """从 W 面取策略。

    slack > 0 时启用「反拖延」：若某个**会改变我方气数**的动作，其 Q 与最优值
    差距在 slack 内，则优先选它。原因：纯可达概率下"原地防守"与最优值等价
    （方程 W = max(..., W) 有大量不动点），贪心会选出"永不前进"的次优策略；
    要求"有进步就优先进步"可破掉这个退化。
    """
    pi: Dict[Tuple[int, int], Dict[str, float]] = {}
    for (qa, qp) in STATES:
        legal, q = _q_from_W(W, mu, qa, qp)
        if not legal:
            continue
        best = max(q.values())
        idx = [m for m in legal if q.get(m, 0.0) >= best - 1e-9]
        if slack > 0:
            prog = [m for m in legal if next_qi(qa, m) != qa
                    and q.get(m, 0.0) >= best - slack]
            if prog:
                idx = prog
        pi[(qa, qp)] = {m: 1.0 / len(idx) for m in idx}
    return pi


def solve_win_policy(mu: Dict[Tuple[int, int], Dict[str, float]],
                     pi0=None, iters: int = 400
                     ) -> Dict[Tuple[int, int], Dict[str, float]]:
    """兼容入口：直接返回按最大可达概率得到的策略（pi0 不再需要）。"""
    return policy_from_W(solve_reachability(mu, iters=iters), mu)


def make_win_agent(base_cls, *args, win_iters: int = 60, win_slack: float = 0.15, **kwargs):
    """把 OnlineBRAgent 换成「最大可达概率」版本（延迟继承，避免包间循环引用）。"""

    class WinRateAgent(base_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.win_iters = win_iters
            self.win_slack = win_slack

        def act(self, own, opp):
            if (self.opening_book is not None and getattr(self.belief, "steps", 0) == 0
                    and own == self._start_qi and opp == self._start_qi):
                legal = get_valid_moves(own, mode=self.mode)
                m = self.opening_book.sample(legal, self.rng,
                                             opp_open=self.opp_opening_pred())
                if m is not None:
                    self._last_opening = m
                    return m
            mu = self._opp_policy_from_belief()
            W = solve_reachability(mu, iters=self.win_iters, tables=self.T)
            pi = policy_from_W(W, mu, slack=self.win_slack)
            legal = get_valid_moves(own, mode=self.mode)
            base = pi.get((own, opp)) or {m: 1.0 / len(legal) for m in legal}
            return sample_policy(apply_settings(base, legal, self.settings), self.rng)

    return WinRateAgent(*args, **kwargs)
