"""把 GRU 权重导出成浏览器可用的 JS（web/encoder_gru.js）。"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data", "artifacts", "encoder_gru.json")
DST = os.path.join(ROOT, "web", "encoder_gru.js")


def fmt(v):
    if isinstance(v, list):
        return "[" + ",".join(fmt(x) for x in v) + "]"
    return repr(round(float(v), 4))


def main() -> None:
    d = json.load(open(SRC, encoding="utf-8"))
    lines = [
        "/* 勇气 5.0 · GRU 序列编码器权重（由 courage/learning/seq_encoder.py 训练导出）",
        f" * 结构: 单层 GRU(hidden={d['hidden']}) + sigmoid 输出 → 14 个信念因子",
        " * 输入: (own, opp) 出招序列的 one-hot(9+9)，逐步前向，取最后一步。",
        " */",
        "(function (g) {",
        "  const E = {",
        '    arch: "gru",',
        f"    hidden: {d['hidden']},",
    ]
    for k in ["Wz", "Uz", "bz", "Wr", "Ur", "br", "Wn", "Un", "bn", "Wo", "bo"]:
        lines.append(f"    {k}: {fmt(d[k])},")
    lines += [
        "  };",
        "  g.CourageEncoder = E;",
        "  if (typeof module !== 'undefined' && module.exports) module.exports = E;",
        "})(typeof window !== 'undefined' ? window : globalThis);",
        "",
    ]
    open(DST, "w", encoding="utf-8").write("\n".join(lines))
    print(f"written {DST}  {os.path.getsize(DST)/1024:.1f} KB")


if __name__ == "__main__":
    main()
