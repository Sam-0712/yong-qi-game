"""
勇气 — SQLite 数据层 (哈希/整数化)
========================================
- 状态: 20 位整数哈希
- 招式: MOVE_TO_IDX 整数
- 概率: ROUND(.., 8)
"""

import sqlite3, os
from typing import Dict, Optional
from collections import defaultdict
from .game import MOVE_TO_IDX, IDX_TO_MOVE, state_hash, hash_to_state

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'courage.db')


def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    c = _conn().execute
    c("""
    CREATE TABLE IF NOT EXISTS strategies (
        mode        INTEGER NOT NULL,    -- 0=no_counter, 1=with_counter
        state_hash  INTEGER NOT NULL,
        action_id   INTEGER NOT NULL,
        probability REAL NOT NULL,
        PRIMARY KEY (mode, state_hash, action_id)
    )""")
    c("""
    CREATE TABLE IF NOT EXISTS sessions (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        mode        INTEGER NOT NULL,
        personality TEXT DEFAULT 'balanced',
        started_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        human_wins  INTEGER DEFAULT 0,
        ai_wins     INTEGER DEFAULT 0,
        draws       INTEGER DEFAULT 0
    )""")
    c("""
    CREATE TABLE IF NOT EXISTS player_observations (
        mode        INTEGER NOT NULL,
        state_hash  INTEGER NOT NULL,
        action_id   INTEGER NOT NULL,
        count       INTEGER DEFAULT 1,
        PRIMARY KEY (mode, state_hash, action_id)
    )""")
    c("""
    CREATE TABLE IF NOT EXISTS player_global_stats (
        mode       INTEGER NOT NULL,
        action_id  INTEGER NOT NULL,
        count      INTEGER DEFAULT 1,
        PRIMARY KEY (mode, action_id)
    )""")
    _conn().commit()


# ── 策略 ──

def save_strategy(strategy: Dict[str, Dict[str, float]], mode: int):
    """保存策略。strategy 是 {state_key: {action: prob}}。"""
    conn = _conn()
    conn.execute("DELETE FROM strategies WHERE mode = ?", (mode,))
    rows = []
    for sk, probs in strategy.items():
        h = state_hash(tuple(int(x) for x in sk.split(",")))
        for act, prob in probs.items():
            aid = MOVE_TO_IDX.get(act, 0)
            if aid:
                rows.append((mode, h, aid, round(prob, 8)))
    conn.executemany(
        "INSERT INTO strategies VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    print(f"  [DB] 策略已存: {len(rows)} 条 (mode={mode})")


def load_strategy(mode: int) -> Dict[str, Dict[str, float]]:
    conn = _conn()
    rows = conn.execute(
        "SELECT state_hash, action_id, probability FROM strategies WHERE mode=?",
        (mode,)
    ).fetchall()
    conn.close()
    if not rows:
        return {}
    result = defaultdict(dict)
    for r in rows:
        st = hash_to_state(r["state_hash"])
        key = f"{st[0]},{st[1]},{st[2]},{st[3]},{st[4]}"
        result[key][IDX_TO_MOVE[r["action_id"]]] = r["probability"]
    return dict(result)


def has_strategy(mode: int) -> bool:
    cnt = _conn().execute(
        "SELECT COUNT(*) FROM strategies WHERE mode=?", (mode,)
    ).fetchone()[0]
    return cnt > 0


# ── 观察 ──

def save_observations(obs: Dict[str, Dict[str, int]], mode: int):
    conn = _conn()
    for sk, moves in obs.items():
        h = state_hash(tuple(int(x) for x in sk.split(",")))
        for act, cnt in moves.items():
            aid = MOVE_TO_IDX.get(act, 0)
            if aid:
                conn.execute("""
                    INSERT INTO player_observations VALUES (?,?,?,?)
                    ON CONFLICT(mode,state_hash,action_id)
                    DO UPDATE SET count=count+?
                """, (mode, h, aid, cnt, cnt))
    conn.commit()
    conn.close()


def load_observations(mode: int) -> Dict[str, Dict[str, int]]:
    rows = _conn().execute(
        "SELECT state_hash, action_id, count FROM player_observations WHERE mode=?",
        (mode,)
    ).fetchall()
    result = defaultdict(lambda: defaultdict(int))
    for r in rows:
        st = hash_to_state(r["state_hash"])
        key = f"{st[0]},{st[1]},{st[2]},{st[3]},{st[4]}"
        result[key][IDX_TO_MOVE[r["action_id"]]] = r["count"]
    return dict(result)


def save_global_stats(counts: Dict[str, int], mode: int):
    conn = _conn()
    for act, cnt in counts.items():
        aid = MOVE_TO_IDX.get(act, 0)
        if aid:
            conn.execute("""
                INSERT INTO player_global_stats VALUES (?,?,?)
                ON CONFLICT(mode,action_id) DO UPDATE SET count=count+?
            """, (mode, aid, cnt, cnt))
    conn.commit()
    conn.close()


def load_global_stats(mode: int) -> Dict[str, int]:
    rows = _conn().execute(
        "SELECT action_id, count FROM player_global_stats WHERE mode=?",
        (mode,)
    ).fetchall()
    return {IDX_TO_MOVE[r["action_id"]]: r["count"] for r in rows}


def reset_observations(mode: int):
    conn = _conn()
    conn.execute("DELETE FROM player_observations WHERE mode=?", (mode,))
    conn.execute("DELETE FROM player_global_stats WHERE mode=?", (mode,))
    conn.commit()
    conn.close()


# ── 会话 ──

def create_session(mode: int, personality: str = "balanced") -> int:
    c = _conn().execute(
        "INSERT INTO sessions (mode, personality) VALUES (?,?)",
        (mode, personality))
    c.connection.commit()
    sid = c.lastrowid
    c.connection.close()
    return sid


def update_session(sid: int, human_win: bool, ai_win: bool):
    conn = _conn()
    if human_win:
        conn.execute("UPDATE sessions SET human_wins=human_wins+1 WHERE id=?", (sid,))
    elif ai_win:
        conn.execute("UPDATE sessions SET ai_wins=ai_wins+1 WHERE id=?", (sid,))
    else:
        conn.execute("UPDATE sessions SET draws=draws+1 WHERE id=?", (sid,))
    conn.commit()
    conn.close()


def get_session_stats(sid: int) -> Optional[dict]:
    row = _conn().execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone()
    return dict(row) if row else None


init_db()
