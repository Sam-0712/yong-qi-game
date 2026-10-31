"""跨会话信念持久化（SQLite）。

沿用旧版常用 SQLite 的做法：把每个对手的 BeliefTracker 状态存成一行 JSON，
换程序/换机后仍能"记得"对手（冷启动的长期方案之一）。
"""

from __future__ import annotations

import json
import os
import sqlite3

from ..learning.belief import BeliefTracker

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_PATH = os.path.join(_ROOT, "data", "belief.db")


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _init():
    c = _conn()
    c.execute("""
        CREATE TABLE IF NOT EXISTS beliefs (
            opp     TEXT PRIMARY KEY,
            data    TEXT NOT NULL,
            updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )""")
    c.commit()
    c.close()


def save_belief(opp: str, tracker: BeliefTracker) -> None:
    _init()
    c = _conn()
    c.execute("""
        INSERT INTO beliefs(opp, data) VALUES (?,?)
        ON CONFLICT(opp) DO UPDATE SET data=excluded.data, updated=CURRENT_TIMESTAMP
    """, (opp, json.dumps(tracker.to_dict(), ensure_ascii=False)))
    c.commit()
    c.close()


def load_belief(opp: str):
    if not os.path.exists(DB_PATH):
        return None
    _init()
    c = _conn()
    row = c.execute("SELECT data FROM beliefs WHERE opp=?", (opp,)).fetchone()
    c.close()
    return BeliefTracker.from_dict(json.loads(row["data"])) if row else None


def list_opponents():
    if not os.path.exists(DB_PATH):
        return []
    _init()
    c = _conn()
    rows = c.execute("SELECT opp, updated FROM beliefs ORDER BY updated DESC").fetchall()
    c.close()
    return [(r["opp"], r["updated"]) for r in rows]
