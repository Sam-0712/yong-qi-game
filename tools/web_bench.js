/* 浏览器引擎基准：GRU 编码器 vs 手写信念，各校准温度下的胜率。
 * 用法：node tools/web_bench.js [--games N]
 * 说明：只读 web/ 下的 engine.js 与 encoder_gru.js，不参与部署（Pages 只发布 ./web）。
 */
globalThis.CourageEncoder = require("../web/encoder_gru.js");
const E = require("../web/engine.js");

const N = (() => {
  const i = process.argv.indexOf("--games");
  return i > 0 ? parseInt(process.argv[i + 1], 10) : 60;
})();

// ── 对手 ──
const EX = ["防御", "吐气", "吐气", "咏春"];
const exploit = (qp, qa, r) => EX[r - 1] || (qp >= 5 ? "咏春" : "吐气");
const turtle = (qp) => (E.validMoves(qp, 0).includes("防御") ? "防御" : "跳");
const aggressive = (qp) => {
  const L = E.validMoves(qp, 0);
  const atk = L.filter((m) => E.ATTR[m][1] > 0);
  const best = atk.sort((a, b) => E.ATTR[b][1] - E.ATTR[a][1])[0];
  return Math.random() < 0.75 && best ? best : L[(Math.random() * L.length) | 0];
};
const defensive = (qp) => {
  const L = E.validMoves(qp, 0);
  const p = {};
  L.forEach((m) => { p[m] = 0.1 / L.length; });
  ["防御", "跳", "吐气"].forEach((m) => { if (L.includes(m)) p[m] += 0.5; });
  let s = 0;
  for (const m in p) s += p[m];
  let r = Math.random();
  for (const m in p) { r -= p[m] / s; if (r <= 0) return m; }
  return L[0];
};
const random = (qp) => {
  const L = E.validMoves(qp, 0);
  return L[(Math.random() * L.length) | 0];
};

function play(cfg, seed, opp) {
  const ai = new E.AI(cfg);
  ai.seed = seed;
  let qa = 3, qp = 3;
  for (let r = 1; r <= 60; r++) {
    const a = ai.act(qa, qp);
    const o = opp(qp, qa, r);
    const w = E.judge(a, o);
    ai.observe(a, o);
    if (w !== 0) return w === 1 ? 1 : 0;
    qa = E.nextQi(qa, a);
    qp = E.nextQi(qp, o);
  }
  return 0;
}

function bench(cfg, opp, n) {
  let w = 0;
  for (let s = 0; s < n; s++) w += play(cfg, (777 + s * 2654435761) >>> 0, opp);
  return ((w / n) * 100).toFixed(0) + "%";
}

const opps = { 套路: exploit, 龟缩: turtle, 猛攻: aggressive, 防守: defensive, 随机: random };
const beliefs = (process.argv.includes("--hand-only") ? ["hand"] : ["gru", "hand"]);
const taus = [1, 1.5, 2, 2.5];

console.log(`每格 ${N} 局`);
console.log("信念  tau  " + Object.keys(opps).map((k) => k.padEnd(6)).join(""));
for (const belief of beliefs) {
  for (const tau of taus) {
    const cfg = { mode: 0, belief, tau, seed: 1 };
    const row = Object.values(opps).map((o) => bench(cfg, o, N).padEnd(6)).join("");
    console.log(belief.padEnd(6) + String(tau).padEnd(5) + row);
  }
}
console.log("\n各风格（gru, tau=2）对套路胜率：");
for (const st of ["balanced", "aggressive", "defensive", "gambler", "thinker"]) {
  process.stdout.write(st + ":" + bench({ mode: 0, belief: "gru", tau: 2, style: st }, exploit, N) + "  ");
}
console.log();
