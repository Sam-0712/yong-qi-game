#!/usr/bin/env python3
"""
勇气 — PyQt6 前端
===================
CFR 纳什均衡 + 自适应学习 + 性格系统。
"""

import sys, os, random
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QCheckBox, QGridLayout,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from src.game import (
    MOVES, MOVES_ATTR, IDX_TO_MOVE,
    judge, transition, get_valid_moves, MAX_QI, MAX_ROUNDS, cfr_lookup,
)
from src.db import (
    load_strategy, save_observations, save_global_stats,
    load_observations, load_global_stats, create_session, update_session,
)
from src.personality import PRESETS, apply_bias
from src.adaptive import CombinedAdaptive


# ── 招式说明 ──
MOVE_HELP = {
    "吐气":   "回1气",
    "馒头":   "攻1 防1 耗1",
    "双馒":   "攻2 防2 耗2",
    "雷切":   "攻3 防3 耗3 ·可中跳",
    "螺旋环": "攻4 防4 耗4",
    "咏春":   "攻5 防5 耗5 ·可中跳",
    "防御":   "攻0 防3 耗0",
    "跳":     "攻0 防0 耗0 ·闪避",
    "无气拨": "攻0 防0 耗0 ·只赢吐气",
}

# ── 招式按钮布局 (row, col) ──
MOVE_BUTTONS = [
    ("咏春",   0, 0), ("吐气",   0, 1),
    ("防御",   1, 0), ("跳",     1, 1),
    ("无气拨", 2, 0), ("馒头",   2, 1),
    ("双馒",   3, 0), ("雷切",   3, 1),
    ("螺旋环", 4, 0),
]

MOVE_QI_COST = {m: MOVES_ATTR[m][0] for m in MOVES if m != "无气拨"}

# ── 性格颜色 ──
PERSONALITY_COLORS = {
    "balanced":   QColor(255, 192, 204),
    "aggressive": QColor(144, 232, 144),
    "defensive":  QColor(249, 215, 112),
    "gambler":    QColor(220, 140, 80),
    "thinker":    QColor(140, 180, 220),
}


class CourageGame(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("勇气")
        self.setFixedSize(810, 540)

        # ── 游戏状态 ──
        self.qi_ai = 3
        self.qi_pl = 3
        self.last_ai = 0
        self.last_pl = 0
        self.round = 0
        self.game_started = False
        self.game_over = False
        self.consecutive_no_qi = 0

        # ── AI 性格 ──
        self.personalities = list(PRESETS.keys())
        self.personality_idx = 0

        # ── CFR ──
        self.cfr_strategy = {}
        self.cfr_loaded = self._load_cfr()

        # ── 自适应层 ──
        self.adaptive = None
        self._init_adaptive()

        # ── 比分 ──
        self.wins = 0
        self.losses = 0
        self.draws = 0
        self.session_id = None

        # ── 设置 ──
        self.enable_no_qi = False
        self.first_thunder = True
        self.disable_defense = False
        self.show_score = True

        self._build_ui()

    # ═══════════════════════════════════════════
    #  CFR & 自适应 加载
    # ═══════════════════════════════════════════

    def _load_cfr(self) -> bool:
        try:
            self.cfr_strategy = load_strategy(0)
            n = sum(len(v) for v in self.cfr_strategy.values())
            print(f"[CFR] 加载 {n} 条策略")
            return n > 0
        except Exception as e:
            print(f"[CFR] 加载失败: {e}")
            return False

    def _init_adaptive(self):
        p = PRESETS[self.personalities[self.personality_idx]]
        self.adaptive = CombinedAdaptive(
            cross_kw={"temperature":      p.cross_temperature,
                      "blend_rate":       p.cross_blend_rate,
                      "confidence_k":     p.cross_confidence_k},
            within_kw={"temperature":     p.within_temperature,
                       "max_blend":       p.within_max_blend,
                       "prior_strength":  p.within_prior},
        )
        for sk, mv in load_observations(0).items():
            for a, c in mv.items():
                self.adaptive.cross.pc[sk][a] += c
                self.adaptive.cross.sv[sk] += c
        for a, c in load_global_stats(0).items():
            self.adaptive.within.counts[a] += c
            self.adaptive.within.total += c

    # ═══════════════════════════════════════════
    #  界面构建
    # ═══════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)
        main.setContentsMargins(10, 10, 10, 10)
        main.setSpacing(10)

        # ── 左侧面板 ──
        left = QWidget()
        left.setFixedWidth(460)
        ll = QVBoxLayout(left)

        # 标题行
        hdr = QHBoxLayout()
        for t in ("出招", "气数", "比分"):
            lbl = QLabel(t)
            lbl.setFont(QFont("Microsoft YaHei", 16))
            lbl.setStyleSheet("color: #3344AA;")
            hdr.addWidget(lbl, 1)
        ll.addLayout(hdr)

        # AI 行
        ai_r = QHBoxLayout()
        self.lbl_ai_name = QLabel("平衡型")
        self.lbl_ai_name.setFont(QFont("Microsoft YaHei", 14))
        self.lbl_ai_name.setStyleSheet("color: rgb(255,192,204);")
        ai_r.addWidget(self.lbl_ai_name)

        self.lbl_ai_move = QLabel("")
        self.lbl_ai_move.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        self.lbl_ai_move.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ai_r.addWidget(self.lbl_ai_move)

        self.lbl_ai_qi = QLabel("3")
        self.lbl_ai_qi.setFont(QFont("Microsoft YaHei", 16))
        self.lbl_ai_qi.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ai_r.addWidget(self.lbl_ai_qi)

        self.lbl_ai_score = QLabel("0")
        self.lbl_ai_score.setFont(QFont("Microsoft YaHei", 12))
        self.lbl_ai_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ai_r.addWidget(self.lbl_ai_score)
        ll.addLayout(ai_r)

        # 玩家行
        pl_r = QHBoxLayout()
        self.lbl_pl_name = QLabel("玩家")
        self.lbl_pl_name.setFont(QFont("Microsoft YaHei", 14))
        self.lbl_pl_name.setStyleSheet("color: #330033;")
        pl_r.addWidget(self.lbl_pl_name)

        self.lbl_pl_move = QLabel("")
        self.lbl_pl_move.setFont(QFont("Microsoft YaHei", 18, QFont.Weight.Bold))
        self.lbl_pl_move.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_r.addWidget(self.lbl_pl_move)

        self.lbl_pl_qi = QLabel("3")
        self.lbl_pl_qi.setFont(QFont("Microsoft YaHei", 16))
        self.lbl_pl_qi.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_r.addWidget(self.lbl_pl_qi)

        self.lbl_pl_score = QLabel("0")
        self.lbl_pl_score.setFont(QFont("Microsoft YaHei", 12))
        self.lbl_pl_score.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl_r.addWidget(self.lbl_pl_score)
        ll.addLayout(pl_r)

        # 回合计数器
        self.lbl_round = QLabel("第 1 回合")
        self.lbl_round.setFont(QFont("Microsoft YaHei", 10))
        self.lbl_round.setStyleSheet("color: #888888;")
        ll.addWidget(self.lbl_round)

        # AI 对话
        self.lbl_ai_text = QLabel("CFR 策略已加载，点击「开始」进行游戏！")
        self.lbl_ai_text.setFont(QFont("Microsoft YaHei", 12))
        self.lbl_ai_text.setWordWrap(True)
        self.lbl_ai_text.setMinimumHeight(45)
        self.lbl_ai_text.setStyleSheet("padding: 4px;")
        ll.addWidget(self.lbl_ai_text)

        # 结果标签
        self.lbl_result = QLabel("")
        self.lbl_result.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setMinimumHeight(35)
        ll.addWidget(self.lbl_result)

        # 设置区
        sets = QWidget()
        sg = QHBoxLayout(sets)
        sg.setContentsMargins(0, 0, 0, 0)

        self.chk_no_qi = QCheckBox("无气拨模式")
        self.chk_no_qi.toggled.connect(lambda c: setattr(self, 'enable_no_qi', c))
        sg.addWidget(self.chk_no_qi)

        self.chk_thunder = QCheckBox("首招雷切")
        self.chk_thunder.setChecked(True)
        self.chk_thunder.toggled.connect(lambda c: setattr(self, 'first_thunder', c))
        sg.addWidget(self.chk_thunder)

        self.chk_no_def = QCheckBox("禁用防御")
        self.chk_no_def.toggled.connect(lambda c: setattr(self, 'disable_defense', c))
        sg.addWidget(self.chk_no_def)

        self.chk_score = QCheckBox("得分/胜率")
        self.chk_score.setChecked(True)
        self.chk_score.toggled.connect(lambda c: (setattr(self, 'show_score', c), self._update_scores()))
        sg.addWidget(self.chk_score)
        ll.addWidget(sets)

        # 底部按钮
        btns = QHBoxLayout()
        self.btn_start = QPushButton("开始！")
        self.btn_start.setFont(QFont("Microsoft YaHei", 22, QFont.Weight.Bold))
        self.btn_start.setMinimumHeight(60)
        self.btn_start.clicked.connect(self._on_start)
        btns.addWidget(self.btn_start)

        self.btn_person = QPushButton("平衡型")
        self.btn_person.setFont(QFont("Microsoft YaHei", 14))
        self.btn_person.setFixedWidth(130)
        self.btn_person.setMinimumHeight(60)
        self.btn_person.clicked.connect(self._on_personality_change)
        self.btn_person.setStyleSheet("background-color: rgb(255,192,204); color: black;")
        btns.addWidget(self.btn_person)
        ll.addLayout(btns)

        main.addWidget(left)

        # ── 右侧招式面板 ──
        right = QWidget()
        right.setFixedWidth(320)
        rg = QGridLayout()
        rg.setSpacing(8)

        self.move_buttons = {}
        for name, row, col in MOVE_BUTTONS:
            btn = QPushButton(name)
            btn.setFont(QFont("Microsoft YaHei", 16, QFont.Weight.Bold))
            btn.setMinimumHeight(44)
            btn.setToolTip(MOVE_HELP.get(name, ""))
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, n=name: self._on_player_move(n))
            self.move_buttons[name] = btn
            rg.addWidget(btn, row, col)

        right.setLayout(rg)
        main.addWidget(right)

    # ═══════════════════════════════════════════
    #  游戏控制
    # ═══════════════════════════════════════════

    def _on_start(self):
        """开始/重新开始"""
        start_qi = 0 if self.enable_no_qi else 3
        self.qi_ai = start_qi
        self.qi_pl = start_qi
        self.last_ai = 0
        self.last_pl = 0
        self.round = 0
        self.consecutive_no_qi = 0
        self.game_started = True
        self.game_over = False

        self.lbl_ai_move.setText("")
        self.lbl_pl_move.setText("")
        self.lbl_result.setText("")
        self.lbl_ai_qi.setText(str(start_qi))
        self.lbl_pl_qi.setText(str(start_qi))
        self.lbl_round.setText("第 1 回合")

        self.adaptive.reset()  # 局内记忆清零
        self._update_move_buttons()
        self.btn_start.setText("重新开始")

        self._game_obs = []

        if not self.session_id:
            self.session_id = create_session(0, PRESETS[self.personalities[self.personality_idx]].name)

        if self.enable_no_qi:
            self.lbl_ai_text.setText("无气拨之战开始！双方 0 气开局，最多连 3 次无气拨。")
        else:
            self.lbl_ai_text.setText("游戏开始！请选择招式。")

    def _on_player_move(self, move_name: str):
        if not self.game_started or self.game_over:
            return

        # 气数校验
        if move_name in MOVE_QI_COST and MOVE_QI_COST[move_name] > self.qi_pl:
            return
        if self.disable_defense and move_name == "防御":
            return
        if not self.first_thunder and self.round == 0 and move_name == "雷切":
            return
        if not self.enable_no_qi and move_name == "无气拨":
            return
        if move_name == "无气拨" and self.consecutive_no_qi >= 3:
            self.lbl_ai_text.setText("连续无气拨已达 3 次上限，换招吧！")
            return

        self.lbl_pl_move.setText(move_name)
        ai_move = self._ai_decide()
        self.lbl_ai_move.setText(ai_move)

        winner = judge(ai_move, move_name)

        # ── 自适应观测 ──
        state_key = f"{self.qi_ai},{self.qi_pl},{self.last_ai},{self.last_pl},{self.round}"
        self.adaptive.observe(state_key, move_name)
        self._game_obs.append((state_key, move_name))

        if winner != 0:
            self.game_over = True
            self._disable_all_moves()
            if winner == 1:
                self.wins += 1
                self.lbl_result.setText("玩家获胜！")
                self.lbl_result.setStyleSheet("color: #00AA00; font-weight: bold;")
                self.lbl_ai_text.setText("你赢了！太厉害了～")
            else:
                self.losses += 1
                self.lbl_result.setText("AI 获胜！")
                self.lbl_result.setStyleSheet("color: #CC0000; font-weight: bold;")
                self.lbl_ai_text.setText("我赢了！下次再挑战吧～")

            self._update_scores()
            update_session(self.session_id, winner == 1, winner == -1)
            self._persist_observations()
            self.btn_start.setText("重新开始")
            return

        # ── 平局 ──
        state = (self.qi_ai, self.qi_pl, self.last_ai, self.last_pl, self.round)
        outcome = transition(state, ai_move, move_name)

        # 安全阀：超过 100 回合强制结束
        if self.round >= 100:
            self.game_over = True
            self._disable_all_moves()
            self.losses += 1
            self.lbl_result.setText(f"超过 100 回合，强制结束！")
            self.lbl_result.setStyleSheet("color: #CC0000; font-weight: bold;")
            self.lbl_ai_text.setText("打太久了，算我赢吧～")
            self._update_scores()
            update_session(self.session_id, False, True)
            self._persist_observations()
            self.btn_start.setText("重新开始")
            return

        if isinstance(outcome, tuple) and outcome[0] != 'terminal':
            self.qi_ai, self.qi_pl, self.last_ai, self.last_pl, self.round = outcome

        self.lbl_ai_qi.setText(str(self.qi_ai))
        self.lbl_pl_qi.setText(str(self.qi_pl))
        self.lbl_round.setText(f"第 {self.round + 1} 回合")
        self.lbl_result.setText(f"平局，继续…")
        self.lbl_result.setStyleSheet("color: #666666;")

        if move_name == "无气拨":
            self.consecutive_no_qi += 1
        else:
            self.consecutive_no_qi = 0

        self._update_move_buttons()

    def _ai_decide(self) -> str:
        """AI 决策: CFR 查表 → 自适应调整 → 性格偏差 → 采样"""
        state = (self.qi_ai, self.qi_pl, self.last_ai, self.last_pl, self.round)
        key = f"{self.qi_ai},{self.qi_pl},{self.last_ai},{self.last_pl},{self.round}"

        # 获取合法招式
        valid = get_valid_moves(self.qi_ai, self.last_ai)
        if self.disable_defense and "防御" in valid:
            valid.remove("防御")
        if not self.first_thunder and self.round == 0 and "雷切" in valid:
            valid.remove("雷切")
        if not self.enable_no_qi and "无气拨" in valid:
            valid.remove("无气拨")
        if self.consecutive_no_qi >= 3 and "无气拨" in valid:
            valid.remove("无气拨")
        if not valid:
            valid = ["吐气"]

        # CFR 基线
        base = cfr_lookup(self.cfr_strategy, state) or {m: 1.0 / len(valid) for m in valid}

        # 自适应调整
        adjusted = self.adaptive.get_adjusted_strategy(
            key, base, self.qi_ai, self.qi_pl, self.last_ai, self.last_pl)

        # 性格偏差
        p = PRESETS[self.personalities[self.personality_idx]]
        final = apply_bias(adjusted, p)

        # 采样
        probs = {m: final.get(m, 0) for m in valid}
        total = sum(probs.values())
        if total < 1e-12:
            return random.choice(valid)
        r = random.random() * total
        cum = 0
        for m in valid:
            cum += probs[m]
            if r <= cum:
                return m
        return valid[-1]

    def _persist_observations(self):
        """持久化本局观测到 DB"""
        obs = defaultdict(lambda: defaultdict(int))
        gc = defaultdict(int)
        for k, m in getattr(self, '_game_obs', []):
            obs[k][m] += 1
            gc[m] += 1
        if obs:
            save_observations(dict(obs), 0)
        if gc:
            save_global_stats(dict(gc), 0)

    # ═══════════════════════════════════════════
    #  UI 更新
    # ═══════════════════════════════════════════

    def _update_move_buttons(self):
        if not self.game_started or self.game_over:
            for btn in self.move_buttons.values():
                btn.setEnabled(False)
            return
        for name, btn in self.move_buttons.items():
            if name == "无气拨":
                btn.setVisible(self.enable_no_qi)
                btn.setEnabled(self.consecutive_no_qi < 3)
                continue
            cost = MOVE_QI_COST.get(name, 999)
            ok = cost <= self.qi_pl
            if self.disable_defense and name == "防御":
                ok = False
            if not self.first_thunder and self.round == 0 and name == "雷切":
                ok = False
            btn.setEnabled(ok)

    def _disable_all_moves(self):
        for btn in self.move_buttons.values():
            btn.setEnabled(False)

    def _update_scores(self):
        if self.show_score:
            self.lbl_ai_score.setText(f"AI {self.losses}")
            self.lbl_pl_score.setText(f"玩家 {self.wins}")
        else:
            t = self.wins + self.losses + self.draws
            if t > 0:
                wr = self.wins / t * 100
                self.lbl_ai_score.setText(f"AI {100-wr:.0f}%")
                self.lbl_pl_score.setText(f"玩家 {wr:.0f}%")
            else:
                self.lbl_ai_score.setText("AI 0%")
                self.lbl_pl_score.setText("玩家 0%")

    # ═══════════════════════════════════════════
    #  性格切换
    # ═══════════════════════════════════════════

    def _on_personality_change(self):
        self.personality_idx = (self.personality_idx + 1) % len(self.personalities)
        p_key = self.personalities[self.personality_idx]
        p = PRESETS[p_key]
        c = PERSONALITY_COLORS.get(p_key, QColor(200, 200, 200))
        self.lbl_ai_name.setText(p.name)
        self.lbl_ai_name.setStyleSheet(f"color: rgb({c.red()},{c.green()},{c.blue()});")
        self.btn_person.setText(p.name)
        self.btn_person.setStyleSheet(
            f"background-color: rgb({c.red()},{c.green()},{c.blue()}); color: black;")


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei", 10))
    window = CourageGame()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
