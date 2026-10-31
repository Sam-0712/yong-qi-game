/* 勇气 5.0 · 前端交互（依赖 engine.js 暴露的 window.CourageEngine） */
(function () {
  "use strict";
  const E = window.CourageEngine;
  const $ = (id) => document.getElementById(id);
  const STORE = "courage5:settings";

  const MOVE_HELP = {
    "吐气": "回 1 气", "馒头": "攻1 防1 耗1", "双馒": "攻2 防2 耗2",
    "雷切": "攻3 防3 耗3 ·中跳", "螺旋环": "攻4 防4 耗4", "咏春": "攻5 防5 耗5 ·中跳",
    "防御": "攻0 防3 耗0", "跳": "攻0 防0 ·闪避", "无气拨": "攻0 防0 ·只赢吐气",
  };
  const STYLE_NAME = { balanced: "平衡型", aggressive: "进攻型", defensive: "防守型", gambler: "赌徒型", thinker: "深思型" };

  const saved = (() => { try { return JSON.parse(localStorage.getItem(STORE)) || {}; } catch (e) { return {}; } })();
  const cfg = {
    mode: saved.mode ?? 0, tau: 2.0, stepCost: 0.03, brSweeps: 80,
    difficulty: saved.difficulty ?? 1.0, style: saved.style ?? "balanced",
    noise: (saved.mode === 1) ? 0.3 : 0.0, prior: "none",
    belief: "gru",   // 序列编码器（缺失时自动回退到手写信念）
  };

  let ai = new E.AI(cfg);
  let qa = 3, qp = 3, round = 0, started = false, over = false;
  // 回合上限：此前完全没有上限，实测能一路平到 40 回合。
  // 到上限后按「剩余气数」判定——气多是硬优势（0~5 气是阈值型价值，见设计文档）。
  const MAX_ROUNDS = 15;
  let score = { w: 0, l: 0 };

  // 主题
  const theme = saved.theme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  document.documentElement.setAttribute("data-theme", theme);
  $("themeBtn").textContent = theme === "dark" ? "☾" : "☀";
  $("themeBtn").onclick = () => {
    const t = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", t);
    $("themeBtn").textContent = t === "dark" ? "☾" : "☀";
    persist({ theme: t });
  };

  function persist(patch) { Object.assign(saved, patch); try { localStorage.setItem(STORE, JSON.stringify(saved)); } catch (e) {} }
  function syncAI() { ai.cfg.difficulty = cfg.difficulty; ai.cfg.style = cfg.style; ai.cfg.noise = cfg.noise; ai.cfg.mode = cfg.mode; }

  // 招式按钮
  function buildGrid() {
    const grid = $("moveGrid"); grid.innerHTML = "";
    for (const m of E.MOVES) {
      if (m === "无气拨" && cfg.mode === 0) continue;
      const b = document.createElement("button");
      b.dataset.move = m;
      b.innerHTML = `${m}<small>${MOVE_HELP[m] || ""}</small>`;
      b.onclick = () => play(m);
      grid.appendChild(b);
    }
  }

  function pips(el, n) {
    el.innerHTML = "";
    for (let i = 0; i < 5; i++) {
      const i2 = document.createElement("i");
      if (i < n) i2.className = "on";
      el.appendChild(i2);
    }
  }

  function render() {
    pips($("aiQi"), qa); pips($("plQi"), qp);
    $("aiStyle").textContent = STYLE_NAME[cfg.style] || "平衡型";
    $("score").textContent = `胜 ${score.w} · 负 ${score.l}`;
    $("round").className = "round";
    if (!started) $("round").textContent = "点击「开始对局」";
    else if (over) { /* 结果文字已在结算里写 */ }
    else $("round").textContent = `第 ${round + 1} 回合`;
    const legal = E.validMoves(qp, cfg.mode);
    document.querySelectorAll("#moveGrid button").forEach((b) => {
      b.disabled = !started || over || !legal.includes(b.dataset.move);
    });
  }

  function setMove(el, text) {
    el.textContent = text;
    el.classList.add("pop");
    setTimeout(() => el.classList.remove("pop"), 300);
  }

  function log(html) {
    const d = document.createElement("div");
    d.className = "l"; d.innerHTML = html;
    const box = $("log"); box.appendChild(d); box.scrollTop = box.scrollHeight;
  }

  function start() {
    cfg.mode = +$("mode").value;
    cfg.noise = cfg.mode === 1 ? 0.3 : 0.0;
    syncAI();
    ai = new E.AI({ mode: cfg.mode, tau: cfg.tau, stepCost: cfg.stepCost, brSweeps: cfg.brSweeps,
      difficulty: cfg.difficulty, style: cfg.style, noise: cfg.noise, prior: "none",
      belief: cfg.belief, seed: (Date.now() % 1e9) | 0 });
    const badge = $("beliefTag");
    if (badge) badge.textContent = ai.beliefKind === "gru" ? "GRU 编码器" : "手写信念";
    qa = qp = cfg.mode === 1 ? 0 : 3; round = 0; started = true; over = false;
    $("aiMove").textContent = "—"; $("plMove").textContent = "—";
    $("startBtn").textContent = "重新开始";
    buildGrid();
    log(`对局开始 · 双方 ${qa} 气${cfg.mode === 1 ? " · 无气拨模式" : ""}`);
    render();
  }

  function play(move) {
    if (!started || over) return;
    if (!E.validMoves(qp, cfg.mode).includes(move)) return;
    document.querySelectorAll("#moveGrid button").forEach((b) => (b.disabled = true));

    const aiMove = ai.act(qa, qp);
    const w = E.judge(aiMove, move);
    ai.observe(aiMove, move);
    setMove($("plMove"), move);
    setMove($("aiMove"), aiMove);

    setTimeout(() => {
      if (w !== 0) {
        over = true;
        const humanWin = w === -1;
        if (humanWin) { score.w++; $("round").textContent = "你赢了！"; $("round").className = "round win"; }
        else { score.l++; $("round").textContent = "AI 获胜"; $("round").className = "round lose"; }
        log(`第 ${round + 1} 回合 · 你出 <b>${move}</b> · AI 出 <b>${aiMove}</b> · <b>${humanWin ? "你胜" : "AI胜"}</b>`);
        render();
      } else {
        qa = E.nextQi(qa, aiMove); qp = E.nextQi(qp, move); round++;
        log(`第 ${round} 回合 · 你出 <b>${move}</b> · AI 出 <b>${aiMove}</b> · 平局继续`);
        if (round >= MAX_ROUNDS) {
          // 上限结算：气多者胜；气数相同判和
          over = true;
          const mark = "round";
          if (qa > qp) {
            score.l++; $("round").textContent = `AI 获胜（${MAX_ROUNDS} 回合上限，气数 ${qa}:${qp}）`;
            $("round").className = "round lose"; log(`达到 ${MAX_ROUNDS} 回合上限 · 气数 ${qa}:${qp} · <b>AI胜</b>`);
          } else if (qa < qp) {
            score.w++; $("round").textContent = `你赢了！（${MAX_ROUNDS} 回合上限，气数 ${qa}:${qp}）`;
            $("round").className = "round win"; log(`达到 ${MAX_ROUNDS} 回合上限 · 气数 ${qa}:${qp} · <b>你胜</b>`);
          } else {
            $("round").textContent = `和局（${MAX_ROUNDS} 回合上限，气数相同）`;
            $("round").className = mark; log(`达到 ${MAX_ROUNDS} 回合上限 · 气数相同 · <b>和局</b>`);
          }
        }
        render();
      }
    }, 320);
  }

  // 设置绑定
  $("startBtn").onclick = start;
  $("diff").value = cfg.difficulty;
  $("style").value = cfg.style;
  $("mode").value = String(cfg.mode);
  $("diff").oninput = (e) => { cfg.difficulty = +e.target.value; syncAI(); persist({ difficulty: cfg.difficulty }); };
  $("style").onchange = (e) => { cfg.style = e.target.value; syncAI(); render(); persist({ style: cfg.style }); };
  $("mode").onchange = (e) => { cfg.mode = +e.target.value; cfg.noise = cfg.mode === 1 ? 0.3 : 0; syncAI(); buildGrid(); persist({ mode: cfg.mode }); };

  buildGrid();
  render();
})();
