"""对局记录：把每局双方的完整出招序列追加到 JSONL（分析材料）。

真实人类数据稀缺，先把每局的 (ai_seq, opp_seq, qi_seq, 结果, 回合) 落盘，
后续可用于：估计因子、监督小序列编码器、复盘诊断。
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PATH = os.path.join(_ROOT, "data", "records", "episodes.jsonl")


class EpisodeRecorder:
    def __init__(self, path: Optional[str] = None):
        self.path = path or DEFAULT_PATH
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def log(self, **rec) -> None:
        rec.setdefault("ts", round(time.time(), 3))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def count(self) -> int:
        if not os.path.exists(self.path):
            return 0
        with open(self.path, encoding="utf-8") as f:
            return sum(1 for _ in f)
