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

  // 五风格开局簿（tools/opening_mc.py 用大样本对局定价：风格先验 × 实测得分）。
  // 共同约束：3 气时「雷切」会把气一次打光（下一回合无攻击手段=交出节奏），占比限 20%。
  // balanced 的「防御 51% / 攻击 37%」是有意为之——平衡型应该是攻防大致均衡。
  const OPENING_BY_STYLE = {
    balanced: { "吐气": 0.1048, "馒头": 0.0519, "双馒": 0.1206, "雷切": 0.1983, "防御": 0.5108, "跳": 0.0137 },
    aggressive: { "吐气": 0.0842, "馒头": 0.1083, "双馒": 0.2472, "雷切": 0.1983, "防御": 0.3481, "跳": 0.0138 },
    defensive: { "吐气": 0.072, "馒头": 0.035, "双馒": 0.0913, "雷切": 0.1983, "防御": 0.5828, "跳": 0.0205 },
    gambler: { "吐气": 0.0722, "馒头": 0.1103, "双馒": 0.2178, "雷切": 0.1983, "防御": 0.3848, "跳": 0.0166 },
    thinker: { "吐气": 0.0848, "馒头": 0.0506, "双馒": 0.1174, "雷切": 0.1983, "防御": 0.5332, "跳": 0.0157 },
  };
  const SAFE_OPENING = { "防御": 0.70, "双馒": 0.13, "雷切": 0.08, "吐气": 0.06, "馒头": 0.02, "跳": 0.01, "无气拨": 0.05 };

  // 理性转化者：能必杀就必杀；否则能安全蓄气就蓄气；否则挑高防低耗。
  // 它是「聪明人类」的规则化模型，用来给决策补上"对手会趁我无力时收割"这一威胁。
  function converterPolicy(legalB, legalOwn) {
    if (!legalB.length) return null;
    // 1. 真必杀：能打穿我方全部合法招 → 用最省的
    const kills = legalB.filter((b) => legalOwn.every((a) => judge(b, a) === 1));
    if (kills.length) {
      const best = kills.reduce((x, y) => (ATTR[x][0] <= ATTR[y][0] ? x : y));
      return { [best]: 1 };
    }
    // 2. 有正收益 → 同时保留「节流」（最省的能赢招）和「收割」（最强的能赢招）。
    //    只取期望最高会偏向强招，让幽灵误以为对手会烧光自己的气，
    //    从而掩盖"我方低气正被低成本蚕食"这一事实。
    const ev = {};
    for (const b of legalB) {
      let sum = 0;
      for (const a of legalOwn) sum += judge(b, a);
      ev[b] = legalOwn.length ? sum / legalOwn.length : 0;
    }
    const bestEv = Math.max(...legalB.map((b) => ev[b]));
    if (bestEv > 1e-9) {
      const cands = legalB.filter((b) => ev[b] > 1e-9);
      const cheap = cands.reduce((x, y) =>
        (ATTR[x][0] - ev[x] <= ATTR[y][0] - ev[y] ? x : y));
      const strong = cands.reduce((x, y) =>
        (ev[x] - ATTR[x][0] >= ev[y] - ATTR[y][0] ? x : y));
      if (cheap === strong) return { [cheap]: 1 };
      return { [cheap]: 0.5, [strong]: 0.5 };
    }
    // 3. 打不到我们（我方有全能格挡）→ 蓄气最划算
    if (legalB.includes("吐气")) return { "吐气": 1 };
    const best = legalB.reduce((x, y) => (ATTR[x][2] >= ATTR[y][2] ? x : y));
    return { [best]: 1 };
  }

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
      this.steps = 0;
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
      this.steps += 1;
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

  // ── GRU 编码器信念（与 Python 的 EncoderBelief 对齐）──
  // 输入 (own, opp) 出招序列 → 14 个信念因子 → 对手动作分布。
  // 因子语义与手写信念一致，只是"算法"从统计式换成了学习式（3.4k 参数）。
  const FACTOR_KEYS = [
    "long_atk", "long_heavy", "long_def", "long_dodge", "long_exhale", "long_noqi",
    "short_atk", "attack_streak", "switch", "predictability", "mirror",
    "volatility", "escalation", "threat",
  ];
  const CAT_OF = {
    "馒头": "attack", "双馒": "attack", "雷切": "attack",
    "螺旋环": "heavy", "咏春": "heavy",
    "防御": "defense", "跳": "dodge", "吐气": "exhale", "无气拨": "noqi",
  };

  function _sig(x) { return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, x)))); }

  function mv(W, v) {           // 矩阵(行) × 向量
    const out = new Array(W.length);
    for (let i = 0; i < W.length; i++) {
      const row = W[i];
      let s = 0;
      for (let j = 0; j < row.length; j++) s += row[j] * v[j];
      out[i] = s;
    }
    return out;
  }

  // 逐步前向，返回最后一步的 14 个因子
  function gruLastStep(enc, ownSeq, oppSeq) {
    const H = enc.hidden, N = MOVES.length;
    let h = new Array(H).fill(0);
    let y = null;
    const T = Math.min(ownSeq.length, oppSeq.length);
    for (let t = 0; t < T; t++) {
      const x = new Array(2 * N).fill(0);
      const oi = MOVES.indexOf(ownSeq[t]), pi = MOVES.indexOf(oppSeq[t]);
      if (oi >= 0) x[oi] = 1;
      if (pi >= 0) x[N + pi] = 1;
      const zx = mv(enc.Wz, x), zh = mv(enc.Uz, h);
      const z = new Array(H);
      for (let i = 0; i < H; i++) z[i] = _sig(zx[i] + zh[i] + enc.bz[i]);
      const rx = mv(enc.Wr, x), rh0 = mv(enc.Ur, h);
      const rh = new Array(H);
      for (let i = 0; i < H; i++) rh[i] = h[i] * _sig(rx[i] + rh0[i] + enc.br[i]);
      const nx = mv(enc.Wn, x), nh = mv(enc.Un, rh);
      const n = new Array(H);
      for (let i = 0; i < H; i++) n[i] = Math.tanh(nx[i] + nh[i] + enc.bn[i]);
      h = h.map((hp, i) => (1 - z[i]) * n[i] + z[i] * hp);
      const yx = mv(enc.Wo, h), yh = new Array(enc.Wo.length);
      for (let i = 0; i < yx.length; i++) yh[i] = _sig(yx[i] + enc.bo[i]);
      y = yh;
    }
    return y || new Array(FACTOR_KEYS.length).fill(1 / 6);
  }

  class EncoderBelief {
    constructor(enc, opts = {}) {
      this.enc = enc;
      this.maxLen = opts.max_len ?? 24;
      this.smooth = opts.smooth ?? 0.05;
      this.tau = opts.tau ?? 1.0;
      this.floor = opts.floor ?? 0.0;
      this.own = []; this.opp = []; this.lastOwn = null; this.steps = 0;
    }
    reset() { this.own = []; this.opp = []; this.lastOwn = null; this.steps = 0; }
    observe(ownMove, oppMove) {
      if (!oppMove) return;
      this.own.push(ownMove); this.opp.push(oppMove);
      if (this.own.length > this.maxLen) { this.own.shift(); this.opp.shift(); }
      this.lastOwn = ownMove;
      this.steps += 1;
    }
    factors() {
      if (!this.own.length) {
        const u = 1 / 6, f = {};
        FACTOR_KEYS.forEach((k) => { f[k] = u; });
        return f;
      }
      const row = gruLastStep(this.enc, this.own, this.opp);
      const f = {};
      FACTOR_KEYS.forEach((k, i) => { f[k] = row[i]; });
      return f;
    }
    oppWeights(legal, lastOwn) {
      const f = this.factors();
      const cl = {
        attack: f.long_atk, heavy: f.long_heavy, defense: f.long_def,
        dodge: f.long_dodge, exhale: f.long_exhale, noqi: f.long_noqi,
      };
      const cs = Object.assign({}, cl, { attack: f.short_atk });
      const ws = f.switch;
      let cw = {};
      for (const c of CATS) cw[c] = (1 - ws) * cl[c] + ws * cs[c];
      if (this.floor > 0) for (const c of CATS) cw[c] = Math.max(0, cw[c] - this.floor);
      if (this.tau !== 1) for (const c of CATS) cw[c] = Math.pow(cw[c], this.tau);
      if (f.attack_streak > 0) {
        cw.attack *= 1 + 1.5 * f.attack_streak;
        cw.heavy *= 1 + 1.0 * f.attack_streak;
      }
      const w = {};
      let s = 0;
      for (const m of legal) { const v = (cw[CAT_OF[m]] || 0) + 1e-6; w[m] = v; s += v; }
      if (s > 0) for (const m of legal) w[m] /= s;
      const lo = lastOwn ?? this.lastOwn;
      if (lo != null && f.mirror > 0.3 && w[lo] != null) {
        w[lo] += f.mirror;
        let s2 = 0; for (const m of legal) s2 += w[m];
        for (const m of legal) w[m] /= s2;
      }
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
        prior: "none", seed: 12345, robust: 0.3, opening: true, belief: "auto",
        // noise0 默认 0：全局随机会把定理级正确的最优招一起随机掉
        // （(0,0) 该出吐气，被 noise 0.30 污染到 82%）。防死锁交给 antiLoop。
        roundTotal: 15, noise0: 0.0, urgency: 3.0, zeroPenalty: 0.4, antiLoop: true,
      }, cfg);
      const enc = (typeof globalThis !== "undefined" && globalThis.CourageEncoder) || null;
      const wantGru = this.cfg.belief === "gru"
        || (this.cfg.belief === "auto" && !!enc);
      this.beliefKind = (wantGru && enc) ? "gru" : "hand";
      this.belief = this.beliefKind === "gru"
        ? new EncoderBelief(enc, { tau: this.cfg.tau, smooth: 0.05 })
        : new Belief({ prior: PRIORS[this.cfg.prior] });
      this.seed = this.cfg.seed >>> 0;
      this.lastOwn = null;
    }
    _rnd() { // xorshift32 可复现
      let x = this.seed; x ^= x << 13; x >>>= 0; x ^= x >> 17; x ^= x << 5; x >>>= 0;
      this.seed = x; return x / 4294967296;
    }
    reset() { this.belief.reset(); this.lastOwn = null; this._hist = []; }
    observe(own, opp) { this.belief.observe(own, opp); this.lastOwn = own; }

    _oppTable() {
      const mode = this.cfg.mode, table = [];
      // 幽灵威胁权重：按观测样本量衰减，但**永不归零**。
      // 旧版用 (1-maxCat) 调制，等价于"看得越清越不设防"——对手只要先演几手
      // 温顺牌，AI 就把"它会兑现优势"这个假设整个丢掉，随即被反杀。
      let beta = 0;
      if (this.cfg.robust > 0) {
        const n = this.belief.steps || 0;
        const conf = n / (n + 6);           // 0 → 0; 6 步 → 0.5; 60 步 → 0.91
        beta = this.cfg.robust * (1 - 0.75 * conf);
      }
      for (let qa = 0; qa <= 5; qa++) for (let qp = 0; qp <= 5; qp++) {
        const legalB = validMoves(qp, mode);
        let w = this.belief.oppWeights(legalB, this.lastOwn, this.cfg.tau, this.cfg.floor);
        if (beta > 0) {
          const conv = converterPolicy(legalB, validMoves(qa, mode));
          if (conv) {
            const mixed = {};
            for (const m of legalB) mixed[m] = (1 - beta) * (w[m] || 0) + beta * (conv[m] || 0);
            const s = legalB.reduce((t, m) => t + mixed[m], 0);
            if (s > 1e-12) for (const m of legalB) mixed[m] /= s;
            w = mixed;
          }
        }
        table[SID(qa, qp)] = w;
      }
      return table;
    }

    _maxCat() {
      const b = this.belief;
      if (b._props) {
        const lp = b._props()[0];
        return Math.max(...CATS.map((c) => lp[c] || 0));
      }
      const f = b.factors ? b.factors() : null;
      if (f) {
        return Math.max(f.long_atk, f.long_heavy, f.long_def,
          f.long_dodge, f.long_exhale, f.long_noqi);
      }
      return 1 / 6;
    }

    // ── 循环检测：直接防局部死锁，比全局随机精确 ──
    // 实测死锁发生在 (1,1)：双方都出 cost=0 的「防御/跳」，气数永远不变，
    // 局面精确复现。全局 noise 也能打断，但代价是把正确招一起随机掉
    // （(0,0) 该出吐气，被 noise 污染到 82%）。所以只在"真的开始打转"时偏离。
    _detectCycle(maxP) {
      const h = this._hist || [];
      const n = h.length;
      const mp = maxP || 5;
      for (let p = 1; p <= mp; p++) {
        if (n < 2 * p) continue;
        let same = true;
        for (let i = 0; i < p; i++) {
          const a = h[n - p + i], b = h[n - 2 * p + i];
          if (a.s !== b.s || a.m !== b.m) { same = false; break; }
        }
        if (same) return p;
      }
      return 0;
    }

    _loopBreak(qa, qp, legal, dist) {
      if (!this.cfg.antiLoop) return null;
      const p = this._detectCycle();
      if (!p) return null;
      const h = this._hist, key = SID(qa, qp);
      const used = new Set();
      for (let i = h.length - p; i < h.length; i++) if (h[i].s === key) used.add(h[i].m);
      let alt = legal.filter((m) => !used.has(m));
      if (!alt.length) alt = legal.filter((m) => nextQi(qa, m) !== qa);
      if (!alt.length) return null;
      const prog = alt.filter((m) => nextQi(qa, m) !== qa);
      const pool = prog.length ? prog : alt;
      return pool.reduce((x, y) => ((dist[x] || 0) >= (dist[y] || 0) ? x : y));
    }

    // 轮数退火：前几轮多随机（避免双方早早同步成镜像循环），末段收敛执行策略；
    // 同时每回合成本随进度上升（越拖越亏，逼出破局手）。
    _roundDyn() {
      const r = this.belief.steps || 0;
      const total = Math.max(1, this.cfg.roundTotal || 15);
      const p = Math.min(1, Math.max(0, r / (total - 1)));
      return {
        noise: (this.cfg.noise0 || 0) * (1 - p),
        step: this.cfg.stepCost * (1 + (this.cfg.urgency || 0) * p),
        progress: p,
      };
    }

    _openingMove(legal) {
      const table = OPENING_BY_STYLE[this.cfg.style] || OPENING_BY_STYLE.balanced;
      const p = {};
      let s = 0;
      for (const m of legal) { p[m] = table[m] || SAFE_OPENING[m] || 0.02; s += p[m]; }
      let r = this._rnd() * s, acc = 0, last = legal[legal.length - 1];
      for (const m of legal) { last = m; acc += p[m]; if (r <= acc) return m; }
      return last;
    }

    _solveBR(table, stepCost) {
      const mode = this.cfg.mode, g = this.cfg.gamma;
      const sc = (stepCost === undefined ? this.cfg.stepCost : stepCost);
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
            const na = nextQi(qa, a);
            // 归零惩罚：把气打到 0 之后的续接价值要打折。
            // 0 气没有任何攻击招，只能等对手犯错才能爬回来；而单边对手模型
            // 无法表达"我出防御他就换破防招"，于是 V(0,·) 会被系统性高估
            // （Python 侧实测 +0.69，真实对局里翻盘几乎为 0），导致 AI 敢在第一手
            // 就把 3 气打光（雷切）然后被蓄气套路收割。这一项只影响续接分支，
            // "本回合直接获胜"不受影响。
            const zp = (this.cfg.zeroPenalty > 0 && na === 0) ? this.cfg.zeroPenalty : 0;
            let q = 0;
            for (const b of validB[si]) {
              const pb = opp[b] || 0; if (!pb) continue;
              const w = judge(a, b);
              q += pb * (w !== 0 ? w : g * V[SID(na, nextQi(qp, b))] - sc - zp);
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
      const legal0 = validMoves(qa, this.cfg.mode);
      // 首招：仅在真正的开局状态（双方起始气）按开局簿采样。
      // 不能只看 steps===0：0 气局面套用 (3,3) 的开局分布会把「防御」硬塞进去，
      // 而实测 (0,0) 的最优是唯一解「吐气」。
      const start = this.cfg.mode === 1 ? 0 : 3;
      if (this.cfg.opening && this.belief.steps === 0 && qa === start && qp === start) {
        const om = this._openingMove(legal0);
        if (!this._hist) this._hist = [];
        this._hist.push({ s: SID(qa, qp), m: om });
        return om;
      }
      const dyn = this._roundDyn();
      const table = this._oppTable();
      const V = this._solveBR(table, dyn.step);
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

      // 噪声（再叠均匀）：取「档位噪声」与「轮数退火噪声」的较大者
      const nz = Math.min(1, Math.max(0, Math.max(this.cfg.noise, dyn.noise)));
      if (nz > 0) {
        let u2 = 1 / legal.length, r = {}, t2 = 0;
        for (const m of legal) { r[m] = (1 - nz) * q[m] + nz * u2; t2 += r[m]; }
        for (const m of legal) r[m] /= t2;
        q = r;
      }
      const forced = this._loopBreak(qa, qp, legal, q);
      const m = forced !== null ? forced : sample(q, () => this._rnd());
      if (!this._hist) this._hist = [];
      this._hist.push({ s: SID(qa, qp), m });
      if (this._hist.length > 24) this._hist.shift();
      return m;
    }
  }

  const api = { MOVES, ATTR, CATEGORIES, CATS, STYLE_BIAS, PRIORS, FACTOR_KEYS,
    judge, validMoves, nextQi, Belief, EncoderBelief, gruFactors: gruLastStep, AI, sample };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  global.CourageEngine = api;
})(typeof window !== "undefined" ? window : globalThis);
