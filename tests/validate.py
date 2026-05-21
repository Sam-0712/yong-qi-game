"""
策略训练 + 验证
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cfr import CFRTrainer
from src.db import save_strategy
from src.game import get_valid_moves, decode_state


print("=" * 56)
print("  勇气  —  DCFR 策略训练")
print("=" * 56)

# 训练
trainer = CFRTrainer(gamma=0.85, dcfr_power_positive=1.5,
                     dcfr_power_negative=0.0, weight_strategy=True)
trainer.train(iterations=500, print_every=50)
avg = trainer.compute_average_strategy_value()
print(f"\n平均策略价值: {avg:+.6f}")

save_strategy(trainer.get_average_strategy('ai'), mode=0)  # no_counter
