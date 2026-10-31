"""5.0 冒烟自检：确认打包链路可用（深度校验留给小版本）。

python -m tools.selfcheck
"""

from __future__ import annotations

import os
import statistics
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.agents import BotAgent, run_duel
from courage.bots.scripts import bots_by_name
from courage.pipeline import PlayerConfig, build_player, describe


def main():
    cfg = PlayerConfig()
    # 若编码器缺失，build_player 会回退到 hand；这里显式提示
    enc = os.path.join(_ROOT, cfg.encoder_file)
    print(describe(cfg))
    print("编码器文件存在:", os.path.exists(enc))

    for name in ("turtle", "defensive", "aggressive"):
        p = build_player(PlayerConfig())
        r = run_duel(p, BotAgent(bots_by_name[name]), mode=0, games=6, cap=60)
        p.close()
        print(f"  vs {name:<10} 胜{r['wins']} 负{r['losses']} 超{r['timeouts']} "
              f"| 回合 {statistics.mean(r['rounds']):.1f}")
    print("OK")


if __name__ == "__main__":
    main()
