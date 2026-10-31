"""对拍：Python GRU 前向 vs 浏览器端 JS 前向（web/engine.js + encoder_gru.js）。

作用：保证「序列 → 14 信念因子」在前后端一致。任何一侧改动（权重导出精度、
one-hot 编码、门顺序、history 截断）都会在这里暴露。

运行：python -m tests.test_encoder_js
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from courage.learning.seq_encoder import FACTOR_KEYS, load_encoder, predict_factors

ENC = os.path.join(_ROOT, "data", "artifacts", "encoder_gru.json")

CASES = {
    "exploit": (["雷切", "防御", "跳", "吐气"], ["防御", "吐气", "吐气", "咏春"]),
    "turtle": (["防御"] * 5, ["防御"] * 5),
    "mixed": (["馒头", "双馒", "雷切", "螺旋环", "咏春", "防御"],
              ["雷切", "防御", "馒头", "双馒", "吐气", "跳"]),
}

JS = r"""
const enc = require('./web/encoder_gru.js');
const E = require('./web/engine.js');
const cases = JSON.parse(process.argv[1]);
const out = {};
for (const k in cases) out[k] = E.gruFactors(enc, cases[k][0], cases[k][1]);
console.log(JSON.stringify({ keys: E.FACTOR_KEYS, vals: out }));
"""


def main() -> None:
    node = shutil.which("node")
    if node is None:
        print("SKIP: 未找到 node，跳过前后端对拍")
        return

    model = load_encoder(ENC)
    py = {k: [predict_factors(model, a, b)[f] for f in FACTOR_KEYS]
          for k, (a, b) in CASES.items()}

    raw = subprocess.run([node, "-e", JS, json.dumps(CASES, ensure_ascii=False)],
                         cwd=_ROOT, capture_output=True, text=True, check=True)
    js = json.loads(raw.stdout)
    assert js["keys"] == FACTOR_KEYS, "因子顺序不一致"

    worst = 0.0
    for k in py:
        d = max(abs(x - y) for x, y in zip(py[k], js["vals"][k]))
        worst = max(worst, d)
        print(f"  {k:<8} 最大|Δ| = {d:.5f}")

    # 权重导出保留 4 位小数，允许 2e-3 的数值差
    assert worst < 2e-3, f"前后端因子不一致: {worst:.5f}"
    print(f"ENCODER PARITY OK (max |Δ| = {worst:.5f})")


if __name__ == "__main__":
    main()
