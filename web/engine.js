/* 勇气 5.0 —— 浏览器端引擎（纯逻辑，无依赖）。
 * 忠实移植 Python：规则 + 手写信念因子 + 校准 + 解析 BR + 难度/风格/噪声。
 * 同文件可在 Node 下 require 做对拍测试。 */
(function (global) {
  "use strict";

  const MOVES = ["吐气", "馒头", "双馒", "雷切", "螺旋环", "咏春", "防御", "跳", "无气拨"];
  const ATTR = { // [cost, atk, def]
    "吐气": [0, 0, 0], "馒头": [1, 1, 1], "双馒": [2, 2, 2], "雷切": [3, 3, 3],
    "螺旋环": [4, 4, 4], "咏春": [5, 5, 5], "防御": [0, 0, 3], "跳": [0, 0, 0],
    "无气拨": [0, 0, 0],
  };
  const HIT_JUMP = new Set(["雷切", "咏春"]);
  const MAX_QI = 5;
  const CATEGORIES = {
    "馒头": "attack", "双馒": "attack", "雷切": "attack",
    "螺旋环": "heavy", "咏春": "heavy",
    "防御": "defense", "跳": "dodge", "吐气": "exhale", "无气拨": "noqi",
  };
  const CATS = ["attack", "heavy", "defense", "dodge", "exhale", "noqi"];

  const STYLE_BIAS = {
    balanced: {},
    aggressive: { "馒头": 1.3, "双馒": 1.3, "雷切": 1.3, "螺旋环": 1.25, "咏春": 1.25, "防御": 0.6, "跳": 0.7 },
    defensive: { "防御": 1.5, "跳": 1.35, "吐气": 1.1, "馒头": 0.7, "双馒": 0.7, "雷切": 0.6, "螺旋环": 0.6, "咏春": 0.6 },
    gambler: { "螺旋环": 1.5, "咏春": 1.6, "防御": 0.4, "跳": 0.5, "吐气": 0.6 },
    thinker: { "防御": 1.2, "跳": 1.1, "咏春": 1.05 },
  };

  const PRIORS = {
    none: null,
    balanced: { attack: 3, heavy: 2, defense: 3, dodge: 1, exhale: 1, noqi: 0 },
    aggressive: { attack: 5, heavy: 3, defense: 1, dodge: 0.5, exhale: 0.5, noqi: 0 },
    defensive: { attack: 1, heavy: 0.5, defense: 6, dodge: 3, exhale: 1, noqi: 0 },
    heavy: { attack: 2, heavy: 5, defense: 2, dodge: 0.5, exhale: 1, noqi: 0 },
  };

  function judge(m1, m2) {
    const a1 = ATTR[m1], a2 = ATTR[m2];
    if (m1 === "无气拨") {
      if (m2 === "吐气") return 1;
      if (m2 === "防御" || m2 === "跳" || m2 === "无气拨") return 0;
      return -1;
    }
    if (m2 === "无气拨") {
      if (m1 === "吐气") return -1;
      if (m1 === "防御" || m1 === "跳") return 0;
      return 1;
    }
    let win1 = false, win2 = false;
    if (a1[1] > 0) {
      if (m2 === "跳") { if (HIT_JUMP.has(m1)) win1 = true; }
      else if (a1[1] > a2[2]) win1 = true;
    }
    if (a2[1] > 0) {
      if (m1 === "跳") { if (HIT_JUMP.has(m2)) win2 = true; }
      else if (a2[1] > a1[2]) win2 = true;
    }
    if (win1 && !win2) return 1;
    if (win2 && !win1) return -1;
    return 0;
  }

  function validMoves(qi, mode) {
    const out = [];
    for (const m of MOVES) {
      if (m === "无气拨" && mode === 0) continue;
      if (ATTR[m][0] <= qi) out.push(m);
    }
    return out;
  }

  function nextQi(qi, move) {
    const cost = ATTR[move][0];
    const gain = move === "吐气" ? 1 : 0;
    return Math.max(0, Math.min(qi - cost + gain, MAX_QI));
  }

  // ── 信念因子（简化：仅决策所需）──
  class Belief {
    constructor(opts = {}) {
      this.d = opts.decay ?? 0.85;
      this.ds = opts.short_decay ?? 0.5;
      this.maxRun = opts.max_run ?? 6;
      this.alpha = opts.alpha ?? 0.20;
      this.smooth = opts.smooth ?? 0.05;
      this.prior = opts.prior ?? null;
      this.priorStrength = opts.prior_strength ?? 3.0;
      this.reset();
    }
    reset() {
      this.long = {}; this.short = {}; CATS.forEach(c => { this.long[c] = 0; this.short[c] = 0; });
      this.longTot = 0; this.shortTot = 0; this.run = 0; this.mirror = 0; this.mirrorTot = 0;
      if (this.prior) {
        const s = Object.values(this.prior).reduce((a, b) => a + Math.max(b, 0), 0) || 1;
        CATS.forEach(c => {
          const v = this.priorStrength * Math.max(this.prior[c] || 0, 0) / s;
          this.long[c] = v; this.short[c] = v;
        });
        this.longTot = this.shortTot = this.priorStrength;
      }
    }
    observe(own, opp) {
      const cat = CATEGORIES[opp];
      if (!cat) return;
      for (const c of CATS) { this.long[c] *= this.d; this.short[c] *= this.ds; }
      this.longTot *= this.d; this.shortTot *= this.ds;
      this.long[cat] += 1; this.longTot += 1;
      this.short[cat] += 1; this.shortTot += 1;
      this.run = (cat === "attack" || cat === "heavy") ? Math.min(this.run + 1, this.maxRun) : 0;
      this.mirrorTot = this.mirrorTot * this.d + 1;
      this.mirror *= this.d;
      if (own != null && own === opp) this.mirror += 1;
    }
    _props() {
      const K = CATS.length, dl = this.longTot + this.alpha * K, ds = this.shortTot + this.alpha * K;
      const lp = {}, sp = {};
      for (const c of CATS) {
        lp[c] = dl > 1e-9 ? (this.long[c] + this.alpha) / dl : 1 / K;
        sp[c] = ds > 1e-9 ? (this.short[c] + this.alpha) / ds : 1 / K;
      }
      return [lp, sp];
    }
    oppWeights(legal, lastOwn, tau, floor) {
      const [lp, sp] = this._props();
      const switchS = this.shortTot > 1e-9 ? Math.abs(sp.attack - lp.attack) : 0;
      const streak = this.run > 0 ? this.run / this.maxRun : 0;
      const mirror = this.mirrorTot > 1e-9 ? this.mirror / this.mirrorTot : 0;
      let catw = {};
      for (const c of CATS) catw[c] = (1 - switchS) * lp[c] + switchS * (c === "attack" ? sp.attack : lp[c]);
      if (floor > 0) for (const c of CATS) catw[c] = Math.max(0, catw[c] - floor);
      if (tau !== 1) for (const c of CATS) catw[c] = Math.pow(Math.max(catw[c], 0), tau);
      if (streak > 0) { catw.attack *= (1 + 1.5 * streak); catw.heavy *= (1 + 1.0 * streak); }
      let w = {}, s = 0;
      for (const m of legal) { const v = (catw[CATEGORIES[m] || "attack"] || 0) + 1e-6; w[m] = v; s += v; }
      for (const m of legal) w[m] /= s;
      if (lastOwn != null && mirror > 0.3 && w[lastOwn] != null) w[lastOwn] += mirror;
      s = 0; for (const m of legal) s += w[m];
      for (const m of legal) w[m] /= s;
      if (this.smooth > 0) {
        const u = 1 / legal.length;
        for (const m of legal) w[m] = (1 - this.smooth) * w[m] + this.smooth * u;
      }
      return w;
    }
  }

  function sample(dist, rnd) {
    let r = rnd(), c = 0, last = null;
    for (const m in dist) { last = m; c += dist[m]; if (r <= c) return m; }
    return last;
  }

  const SID = (qa, qp) => qa * 6 + qp;

  class AI {
    constructor(cfg = {}) {
      this.cfg = Object.assign({
        mode: 0, gamma: 0.95, stepCost: 0.03, brSweeps: 80,
        tau: 1.0, floor: 0.0, difficulty: 1.0, style: "balanced", noise: 0.0,
        prior: "none", seed: 12345,
      }, cfg);
      this.belief = new Belief({ prior: PRIORS[this.cfg.prior] });
      this.seed = this.cfg.seed >>> 0;
      this.lastOwn = null;
    }
    _rnd() { // xorshift32 可复现
      let x = this.seed; x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
      this.seed = x; return x / 4294967296;
    }
    reset() { this.belief.reset(); this.lastOwn = null; }
    observe(own, opp) { this.belief.observe(own, opp); this.lastOwn = own; }

    _oppTable() {
      const mode = this.cfg.mode, table = [];
      for (let qa = 0; qa <= 5; qa++) for (let qp = 0; qp <= 5; qp++) {
        const legal = validMoves(qp, mode);
        table[SID(qa, qp)] = this.belief.oppWeights(legal, this.lastOwn, this.cfg.tau, this.cfg.floor);
      }
      return table;
    }

    _solveBR(table) {
      const mode = this.cfg.mode, g = this.cfg.gamma, sc = this.cfg.stepCost;
      const V = new Float64Array(36);
      const validA = [], validB = [];
      for (let qa = 0; qa <= 5; qa++) for (let qp = 0; qp <= 5; qp++) {
        validA[SID(qa, qp)] = validMoves(qa, mode);
        validB[SID(qa, qp)] = validMoves(qp, mode);
      }
      for (let sw = 0; sw < this.cfg.brSweeps; sw++) {
        let maxd = 0;
        for (let si = 0; si < 36; si++) {
          const qa = (si / 6) | 0, qp = si % 6;
          const A = validA[si], opp = table[si];
          let best = -1e9;
          for (const a of A) {
            let q = 0;
            for (const b of validB[si]) {
              const pb = opp[b] || 0; if (!pb) continue;
              const w = judge(a, b);
              q += pb * (w !== 0 ? w : g * V[SID(nextQi(qa, a), nextQi(qp, b))] - sc);
            }
            if (q > best) best = q;
          }
          const d = Math.abs(best - V[si]); if (d > maxd) maxd = d;
          V[si] = best;
        }
        if (maxd < 1e-6) break;
      }
      return V;
    }

    _policy(qa, qp, V, table) {
      const A = validMoves(qa, this.cfg.mode), B = validMoves(qp, this.cfg.mode);
      const qs = [];
      for (const a of A) {
        let q = 0;
        for (const b of B) {
          const pb = table[SID(qa, qp)][b] || 0; if (!pb) continue;
          const w = judge(a, b);
          q += pb * (w !== 0 ? w : this.cfg.gamma * V[SID(nextQi(qa, a), nextQi(qp, b))] - this.cfg.stepCost);
        }
        qs.push(q);
      }
      const mx = Math.max(...qs);
      let dist = {};
      const best = A.filter((_, i) => qs[i] >= mx - 1e-9);
      best.forEach(m => dist[m] = 1 / best.length);
      return dist; // BR 分布（确定性 argmax，平局均分）
    }

    act(qa, qp) {
      const table = this._oppTable();
      const V = this._solveBR(table);
      let dist = this._policy(qa, qp, V, table);

      // 风格偏置
      const bias = STYLE_BIAS[this.cfg.style] || {};
      let p = {}, s = 0;
      for (const m in dist) { const v = dist[m] * (bias[m] ?? 1); p[m] = v; s += v; }
      for (const m in p) p[m] /= s || 1;

      // 难度混合（向均匀）
      const d = Math.min(1, Math.max(0, this.cfg.difficulty));
      const legal = validMoves(qa, this.cfg.mode), u = 1 / legal.length;
      let q = {}, t = 0;
      for (const m of legal) { q[m] = d * (p[m] || 0) + (1 - d) * u; t += q[m]; }
      for (const m of legal) q[m] /= t;

      // 噪声（再叠均匀）
      const nz = Math.min(1, Math.max(0, this.cfg.noise));
      if (nz > 0) {
        let u2 = 1 / legal.length, r = {}, t2 = 0;
        for (const m of legal) { r[m] = (1 - nz) * q[m] + nz * u2; t2 += r[m]; }
        for (const m of legal) r[m] /= t2;
        q = r;
      }
      return sample(q, () => this._rnd());
    }
  }

  const api = { MOVES, ATTR, CATEGORIES, CATS, STYLE_BIAS, PRIORS, judge, validMoves, nextQi, Belief, AI, sample };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.CourageEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
