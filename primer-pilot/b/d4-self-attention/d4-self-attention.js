// D4 Self-attention. Figures: a sentence view of GPT-2 heads (click a token), query/key/value projections of a tiny embedding, a worked attention
// example with draggable queries and keys, the sqrt(d_k) scaling experiment, a heat map with a causal-mask switch, a leak experiment on a tiny model,
// and a grid of all 144 GPT-2 heads. The pure logic is exported with module.exports for node tests.
(function () {
  "use strict";
  // ---------- pure logic ----------
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function softmaxArr(z) { var m = Math.max.apply(null, z), e = z.map(function (x) { return Math.exp(x - m); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (x) { return x / s; }); }
  // One query against the first `vis` keys (the rest are masked). Returns raw dot products, scaled scores, weights (0 for masked), exp sum and output.
  function attend(q, K, V, vis) {
    var dk = q.length, raw = [], sc = [], i;
    for (i = 0; i < vis; i++) { raw.push(dot(q, K[i])); sc.push(raw[i] / Math.sqrt(dk)); }
    var w = softmaxArr(sc), Z = 0; sc.forEach(function (s) { Z += Math.exp(s); });
    var o = V[0].map(function () { return 0; });
    for (i = 0; i < vis; i++) for (var c = 0; c < o.length; c++) o[c] += w[i] * V[i][c];
    while (w.length < K.length) { w.push(0); raw.push(null); sc.push(null); }
    return { raw: raw, sc: sc, w: w, Z: Z, o: o };
  }
  function matvec(W, x) { return W.map(function (r) { return dot(r, x); }); }
  if (typeof module !== "undefined" && module.exports) { module.exports = { dot: dot, softmaxArr: softmaxArr, attend: attend, matvec: matvec }; }
  if (typeof document === "undefined") return;

  // ---------- helpers ----------
  var ATT = null, ALL = null, LEAK = null;
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function showTok(t) { return t.charAt(0) === " " ? "▁" + t.slice(1) : t; }
  function box(val, cap, cls) { return '<span class="t ' + (cls || "") + '"><b>' + val + "</b><i>" + cap + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function f1(x) { return Kit.fmt(x, 1); }
  function f2(x) { return Kit.fmt(x, 2); }
  function f3(x) { return Kit.fmt(x, 3); }
  function pc(x) { return (100 * x).toFixed(x >= 0.995 || x < 0.1 ? (x < 0.005 ? 1 : 0) : 0) + "%"; }
  function pcs(x) { return x < 0.005 ? "0%" : x > 0.995 ? "100%" : (100 * x).toFixed(0) + "%"; }
  function vec(v, d) { return "(" + v.map(function (x) { return Kit.fmt(x, d == null ? 1 : d); }).join(", ") + ")"; }
  function svgSize(svg, w, h) { svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.setAttribute("width", w); svg.setAttribute("height", h); svg.style.maxWidth = w + "px"; }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function svgText(parent, x, y, txt, attrs) { var t = Kit.el("text", Object.assign({ x: x, y: y }, attrs || {}), parent); t.textContent = txt; return t; }
  var MONO = "ui-monospace, Menlo, monospace";

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/attention.json"), Kit.data("data/allheads.json"), Kit.data("data/leak.json")]).then(function (r) {
      ATT = r[0]; ALL = r[1]; LEAK = r[2];
      sentenceFig(); qkvFig(); exampleFig(); scaleFig(); heatFig(); leakFig(); headsFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. sentence view ----------
  var HEADNOTE = {
    L4H11: "Layer 4, head 11 puts almost all its weight on the token just before.",
    L0H1: "Layer 0, head 1 puts most of its weight on the token itself. When the same token appeared earlier, that copy gets a share.",
    L5H5: "Layer 5, head 5 puts its weight on the first token for most tokens. When a token has appeared before, it looks at the token that came after the earlier copy.",
    L1H10: "Layer 1, head 10 has no sharp peak. It spreads weight over the nearest tokens."
  };
  var SENTLABEL = { s1: "Tag 4127 twice", s2: "He let it go", s3: "Sassoon Dock" };
  function sentenceFig() {
    var bar = Kit.$("sv-bar"), svg = Kit.$("sv-svg"), read = Kit.$("sv-read"), st = Kit.$("sv-status");
    var si = 0, hid = "L5H5", sel = 16, sbtn = null;
    sbtn = Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ATT.sentences.map(function (s) { return SENTLABEL[s.id]; }), 0, function (i) { si = i; sel = si === 0 ? 16 : ATT.sentences[si].tokens.length - 1; draw(); });
    var hsel = Kit.select(bar, "Head", ATT.heads.map(function (h) { return { value: h.id, label: h.id + ", " + h.name }; }), hid, function (v) { hid = v; draw(); });
    var nav = Kit.html("span", { class: "row-chips" }, bar);
    Kit.button(nav, "▲ Earlier token", function () { sel = Math.max(0, sel - 1); draw(); });
    Kit.button(nav, "Later token ▼", function () { sel = Math.min(ATT.sentences[si].tokens.length - 1, sel + 1); draw(); });
    Kit.button(nav, "Show ▁Tag again", function () { si = 0; sel = 16; sbtn.forEach(function (b, k) { b.setAttribute("aria-pressed", String(k === 0)); }); hid = "L5H5"; hsel.value = hid; draw(); });
    function draw() {
      var P = Kit.palette(), S = ATT.sentences[si], toks = S.tokens, n = toks.length, W = S.heads[hid].w;
      var maxlen = 0; toks.forEach(function (t) { maxlen = Math.max(maxlen, showTok(t).length); });
      var wid = Math.max(300, Kit.width(svg.parentNode) - 36), rowH = 25, top = 14, tokW = Math.min(112, 8 * maxlen + 10), idxW = 24;
      var x0 = idxW + tokW + 14, H = top + n * rowH + 6, maxB = Math.max(26, Math.min(70, (wid - x0) * 0.22)), barX = x0 + maxB + 18, barMax = Math.max(40, wid - barX - 46);
      svgSize(svg, wid, H); clear(svg);
      var row = W[sel], best = 0; for (var j = 0; j <= sel; j++) if (row[j] > row[best]) best = j;
      function yc(i) { return top + i * rowH + rowH / 2; }
      // arcs first (below the labels)
      for (j = 0; j < sel; j++) {
        var w = row[j]; if (w < 0.02) continue;
        var b = Math.min(maxB, 12 + 5 * (sel - j));
        Kit.el("path", { d: "M" + x0 + "," + yc(sel) + " C" + (x0 + b) + "," + yc(sel) + " " + (x0 + b) + "," + yc(j) + " " + x0 + "," + yc(j), fill: "none", stroke: P.coral, "stroke-width": (1 + 7 * w).toFixed(2), "stroke-opacity": (0.3 + 0.65 * w).toFixed(2), "stroke-linecap": "round" }, svg);
      }
      for (var i = 0; i < n; i++) {
        var g = Kit.el("g", { role: "button", tabindex: 0, "aria-label": "Token " + showTok(toks[i]) + ", position " + i + ". Activate to show where it looks.", style: "cursor:pointer" }, svg);
        var future = i > sel, wi = i <= sel ? row[i] : 0;
        Kit.el("rect", { x: 0, y: top + i * rowH + 1, width: x0 - 4, height: rowH - 2, rx: 4, fill: i === sel ? P.mark : "none", stroke: i === sel ? P.navy : "none", "stroke-width": 1.5 }, g);
        if (!future && i !== sel && wi > 0.01) Kit.el("rect", { x: idxW - 2, y: top + i * rowH + 3, width: tokW + 4, height: rowH - 6, rx: 4, fill: P.coral, "fill-opacity": (0.1 + 0.55 * wi).toFixed(2) }, g);
        if (i === 0) svgText(svg, barX, 7, "weight", { "font-size": 10.5, fill: P.muted });
        svgText(g, idxW - 4, yc(i) + 4, String(i), { "text-anchor": "end", "font-size": 11, fill: P.muted, "font-family": MONO });
        svgText(g, idxW + tokW, yc(i) + 4.5, showTok(toks[i]), { "text-anchor": "end", "font-size": 13.5, fill: future ? P.muted : P.ink, "font-family": MONO, opacity: future ? 0.5 : 1, "font-weight": i === sel ? 700 : 400 });
        Kit.el("circle", { cx: x0, cy: yc(i), r: i === sel ? 5 : 3.5, fill: i === sel ? P.coral : (future ? "none" : P.navy), stroke: future ? P.muted : "none", "stroke-width": 1, opacity: future ? 0.5 : 1 }, g);
        if (!future) {
          Kit.el("rect", { x: barX, y: yc(i) - 5, width: Math.max(0, wi * barMax), height: 10, rx: 2, fill: i === sel ? P.navy : P.coral, "fill-opacity": i === sel ? 0.55 : 0.85 }, g);
          if (wi >= 0.02) svgText(g, barX + wi * barMax + 5, yc(i) + 4, wi >= 0.995 ? "1.00" : wi.toFixed(2), { "font-size": 12, fill: i === best ? P.coral : P.ink2, "font-weight": i === best ? 700 : 400, "font-family": MONO });
        }
        (function (k) { g.addEventListener("click", function () { sel = k; draw(); }); g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = k; draw(); } }); })(i);
      }
      // readout: top three
      var order = []; for (j = 0; j <= sel; j++) order.push(j); order.sort(function (a, b) { return row[b] - row[a]; });
      var parts = order.slice(0, 3).filter(function (k) { return row[k] >= 0.02; }).map(function (k) { return "<b>" + esc(showTok(toks[k])) + "</b> (" + (k === sel ? "itself, " : "position " + k + ", ") + pcs(row[k]) + ")"; });
      read.innerHTML = "<span>Token: <b>" + esc(showTok(toks[sel])) + "</b> (position " + sel + ")</span><span>Reads most from: " + parts.join(", ") + "</span><span>Weights add up to <b>" + (row.slice(0, sel + 1).reduce(function (a, b) { return a + b; }, 0)).toFixed(2) + "</b></span>";
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "Head " + hid + " on the sentence “" + S.text + "”. The token " + showTok(toks[sel]) + " puts " + pcs(row[best]) + " of its weight on " + (best === sel ? "itself" : showTok(toks[best])) + ".");
      st.textContent = HEADNOTE[hid] + " This describes where the weights go on this sentence. It does not say what the head is for.";
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 2. queries, keys and values from one embedding ----------
  var QKV = {
    tokens: ["the", "young", "hammerhead", "it"],
    X: [[0.5, -1, 0], [1, 1, 0.5], [2, 0.5, -1], [1, -0.5, 1]],
    W: { q: [[1, 0, 1], [0, 1, 0]], k: [[0, 1, 0], [1, 0, -1]], v: [[1, -1, 0], [0, 1, 1]] },
    names: { q: "query", k: "key", v: "value" }, sym: { q: "q", k: "k", v: "v" }, cls: { q: "c-coral", k: "c-navy", v: "c-teal" },
    desc: { q: "what this token looks for", k: "the label this token offers", v: "what this token hands over" }
  };
  function qkvFig() {
    var bar = Kit.$("qk-bar"), ctl = Kit.$("qk-sliders"), work = Kit.$("qk-work"), tab = Kit.$("qk-table"), st = Kit.$("qk-status");
    var tk = 3, role = "q", X = QKV.X.map(function (r) { return r.slice(); }), sl = [];
    var tchips = Kit.chips(Kit.html("div", { class: "row-chips" }, bar), QKV.tokens, tk, function (i) { tk = i; sync(); draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Query", "Key", "Value"], 0, function (i) { role = ["q", "k", "v"][i]; draw(); });
    Kit.button(bar, "Reset", function () { X = QKV.X.map(function (r) { return r.slice(); }); sync(); draw(); });
    for (var c = 0; c < 3; c++) (function (c) { sl.push(Kit.slider(ctl, { label: "x" + (c + 1) + ", number " + (c + 1) + " of the embedding", min: -2, max: 2, step: 0.5, value: X[tk][c] }, function (v) { X[tk][c] = v; draw(); })); })(c);
    function sync() { for (var c = 0; c < 3; c++) sl[c].set(X[tk][c]); }
    function draw() {
      var x = X[tk], W = QKV.W[role], out = matvec(W, x), h = "";
      for (var r = 0; r < 2; r++) {
        var terms = [];
        for (var c = 0; c < 3; c++) terms.push(box(f2(W[r][c] * x[c]), Kit.fmt(W[r][c], 1) + " × " + Kit.fmt(x[c], 1)));
        h += '<div class="eqn">' + box(f2(out[r]), QKV.sym[role] + "<sub>" + (r + 1) + "</sub>, number " + (r + 1) + " of the " + QKV.names[role], QKV.cls[role]) + op("=") + terms[0] + op("+") + terms[1] + op("+") + terms[2] + "</div>";
      }
      work.innerHTML = h;
      work.setAttribute("aria-label", "Working for the " + QKV.names[role] + " of " + QKV.tokens[tk]);
      var t = '<table class="sv-table"><thead><tr><th>token</th><th>embedding x</th><th>query q</th><th>key k</th><th>value v</th></tr></thead><tbody>';
      QKV.tokens.forEach(function (name, i) {
        t += '<tr' + (i === tk ? ' class="on"' : "") + "><td>" + name + "</td><td>" + vec(X[i]) + "</td><td>" + vec(matvec(QKV.W.q, X[i]), 1) + "</td><td>" + vec(matvec(QKV.W.k, X[i]), 1) + "</td><td>" + vec(matvec(QKV.W.v, X[i]), 1) + "</td></tr>";
      });
      tab.innerHTML = t + "</tbody></table>";
      var mats = "W<sub>" + QKV.sym[role].toUpperCase() + "</sub> = " + vec(W[0], 1) + " and " + vec(W[1], 1) + " (two rows of three numbers).";
      st.innerHTML = "The " + QKV.names[role] + " is " + QKV.desc[role] + ". " + mats + " All numbers are made up.";
    }
    draw();
  }

  // ---------- 3. the worked example ----------
  var EX = {
    tokens: ["the", "young", "hammerhead", "it"],
    Q: [[0.6, -0.2], [0.4, 0.5], [0.8, 0.2], [1.2, 0.5]], K: [[0.3, -0.5], [0.8, 0.9], [1.4, 0.6], [0.5, 0.4]], V: [[0.2, 0.9], [-0.8, 0.5], [1.1, -0.2], [0.3, -0.7]]
  };
  function exampleFig() {
    var bar = Kit.$("ex-bar"), sq = Kit.$("ex-qk"), sv = Kit.$("ex-v"), tab = Kit.$("ex-table"), eq = Kit.$("ex-eq"), st = Kit.$("ex-status");
    var Q = EX.Q.map(function (r) { return r.slice(); }), K = EX.K.map(function (r) { return r.slice(); }), V = EX.V;
    var qi = 3, row = null, R1 = 1.6, R2 = 1.25, SZ = 340, M = 16;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["the", "young", "hammerhead", "it"].map(function (t) { return "Query: " + t; }), qi, function (i) { qi = i; row = null; update(); });
    Kit.button(bar, "Reset", function () { Q = EX.Q.map(function (r) { return r.slice(); }); K = EX.K.map(function (r) { return r.slice(); }); row = null; update(); });
    svgSize(sq, SZ, SZ); svgSize(sv, SZ, SZ);
    var qGrid = Kit.el("g", {}, sq), qDyn = Kit.el("g", {}, sq), qHan = Kit.el("g", {}, sq), vGrid = Kit.el("g", {}, sv), vDyn = Kit.el("g", {}, sv);
    function mapper(R) { var s = (SZ - 2 * M) / (2 * R); return { x: function (a) { return M + (a + R) * s; }, y: function (b) { return SZ - M - (b + R) * s; }, s: s, R: R }; }
    var m1 = mapper(R1), m2 = mapper(R2);
    function grid(g, m, P) {
      clear(g);
      for (var t = -Math.floor(m.R); t <= m.R + 1e-9; t += 1) {
        Kit.el("line", { x1: m.x(t), y1: m.y(-m.R), x2: m.x(t), y2: m.y(m.R), stroke: P.rule, "stroke-width": t === 0 ? 1.5 : 0.8 }, g);
        Kit.el("line", { x1: m.x(-m.R), y1: m.y(t), x2: m.x(m.R), y2: m.y(t), stroke: P.rule, "stroke-width": t === 0 ? 1.5 : 0.8 }, g);
        if (t === 0) { Kit.el("rect", { x: m.x(-m.R), y: m.y(m.R), width: 2 * m.R * m.s, height: 2 * m.R * m.s, fill: "none", stroke: P.rule, "stroke-width": 1 }, g); }
        if (t !== 0) { svgText(g, m.x(t), m.y(0) + 12, String(t), { "font-size": 10.5, fill: P.muted, "text-anchor": "middle" }); svgText(g, m.x(0) - 4, m.y(t) + 3.5, String(t), { "font-size": 10.5, fill: P.muted, "text-anchor": "end" }); }
      }
    }
    function arrow(g, m, x, y, col, wdt, dash, op_) {
      var x0 = m.x(0), y0 = m.y(0), x1 = m.x(x), y1 = m.y(y), L = Math.hypot(x1 - x0, y1 - y0);
      if (L < 2) return;
      var ux = (x1 - x0) / L, uy = (y1 - y0) / L, hs = Math.min(9, L * 0.4);
      var attrs = { x1: x0, y1: y0, x2: x1 - ux * hs * 0.6, y2: y1 - uy * hs * 0.6, stroke: col, "stroke-width": wdt, "stroke-linecap": "round", opacity: op_ == null ? 1 : op_ };
      if (dash) attrs["stroke-dasharray"] = "4 3";
      Kit.el("line", attrs, g);
      Kit.el("polygon", { points: x1 + "," + y1 + " " + (x1 - ux * hs - uy * hs * 0.45) + "," + (y1 - uy * hs + ux * hs * 0.45) + " " + (x1 - ux * hs + uy * hs * 0.45) + "," + (y1 - uy * hs - ux * hs * 0.45), fill: col, opacity: op_ == null ? 1 : op_ }, g);
    }
    // handles: 4 keys and the query, kept across redraws so dragging works
    var handles = [];
    function mkHandle(kind, idx) {
      var c = Kit.el("circle", { r: 8, "stroke-width": 2, style: "cursor:grab;touch-action:none", tabindex: 0, role: "slider" }, qHan);
      var drag = false;
      function setFrom(ev) {
        var rc = sq.getBoundingClientRect(), sc = SZ / rc.width;
        var px = (ev.clientX - rc.left) * sc, py = (ev.clientY - rc.top) * sc;
        var a = Math.round(((px - M) / m1.s - R1) * 10) / 10, b = Math.round((-(py - SZ + M) / m1.s - R1) * 10) / 10;
        a = Math.max(-R1, Math.min(R1, a)); b = Math.max(-R1, Math.min(R1, b));
        var tgt = kind === "q" ? Q[qi] : K[idx]; tgt[0] = a; tgt[1] = b; update();
      }
      c.addEventListener("pointerdown", function (e) { drag = true; c.setPointerCapture(e.pointerId); e.preventDefault(); });
      c.addEventListener("pointermove", function (e) { if (drag) setFrom(e); });
      c.addEventListener("pointerup", function () { drag = false; });
      c.addEventListener("pointercancel", function () { drag = false; });
      c.addEventListener("keydown", function (e) {
        var d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[e.key]; if (!d) return;
        e.preventDefault(); var st_ = e.shiftKey ? 0.5 : 0.1, tgt = kind === "q" ? Q[qi] : K[idx];
        tgt[0] = Math.max(-R1, Math.min(R1, Math.round((tgt[0] + d[0] * st_) * 10) / 10)); tgt[1] = Math.max(-R1, Math.min(R1, Math.round((tgt[1] + d[1] * st_) * 10) / 10)); update();
      });
      handles.push({ c: c, kind: kind, idx: idx });
    }
    for (var i = 0; i < 4; i++) mkHandle("k", i);
    mkHandle("q", 0);
    function update() {
      var P = Kit.palette(), vis = qi + 1, q = Q[qi], A = attend(q, K, V, vis);
      if (row === null || row >= vis) { row = 0; for (var j = 0; j < vis; j++) if (A.w[j] > A.w[row]) row = j; }
      // left plane
      clear(qDyn); grid(qGrid, m1, P);
      for (j = 0; j < 4; j++) {
        var masked = j >= vis;
        arrow(qDyn, m1, K[j][0], K[j][1], P.navy, j === row ? 3 : 2, masked, masked ? 0.35 : 1);
        svgText(qDyn, m1.x(K[j][0]), m1.y(K[j][1]) + (K[j][1] >= 0 ? -13 : 22), "k" + (j + 1) + " " + EX.tokens[j], { "font-size": 12, fill: P.navy, "text-anchor": "middle", opacity: masked ? 0.5 : 1, "paint-order": "stroke", stroke: P.paper, "stroke-width": 3 });
      }
      arrow(qDyn, m1, q[0], q[1], P.coral, 3.5);
      svgText(qDyn, m1.x(q[0]), m1.y(q[1]) + 24, "q of " + EX.tokens[qi], { "font-size": 12.5, fill: P.coral, "font-weight": 700, "text-anchor": "middle", "paint-order": "stroke", stroke: P.paper, "stroke-width": 3 });
      handles.forEach(function (h) {
        var p = h.kind === "q" ? q : K[h.idx], masked = h.kind === "k" && h.idx >= vis;
        h.c.setAttribute("cx", m1.x(p[0])); h.c.setAttribute("cy", m1.y(p[1]));
        h.c.setAttribute("fill", h.kind === "q" ? P.coral : P.navy); h.c.setAttribute("fill-opacity", masked ? 0.12 : 0.28); h.c.setAttribute("stroke", h.kind === "q" ? P.coral : P.navy);
        h.c.setAttribute("aria-label", "Move the tip of the " + (h.kind === "q" ? "query of " + EX.tokens[qi] : "key of " + EX.tokens[h.idx]) + ". Arrow keys move it by 0.1.");
        h.c.setAttribute("aria-valuetext", "(" + f1(p[0]) + ", " + f1(p[1]) + ")");
      });
      // right plane: values and the chain of pieces w_i v_i
      clear(vDyn); grid(vGrid, m2, P);
      var cx = 0, cy = 0;
      for (j = 0; j < vis; j++) {
        arrow(vDyn, m2, V[j][0], V[j][1], P.sand, 1.8, false, 0.9);
        var vx = m2.x(V[j][0]), right = V[j][0] >= 0 && vx < SZ - 96;
        svgText(vDyn, vx + (right ? 8 : -8), m2.y(V[j][1]) + (V[j][1] >= 0 ? -6 : 14), "v" + (j + 1) + " " + EX.tokens[j], { "font-size": 11.5, fill: P.ink2, "text-anchor": right ? "start" : "end", "paint-order": "stroke", stroke: P.paper, "stroke-width": 3 });
      }
      for (j = 0; j < vis; j++) {
        var nx = cx + A.w[j] * V[j][0], ny = cy + A.w[j] * V[j][1];
        Kit.el("line", { x1: m2.x(cx), y1: m2.y(cy), x2: m2.x(nx), y2: m2.y(ny), stroke: P.teal, "stroke-width": 2, "stroke-dasharray": "2 3", opacity: 0.9 }, vDyn);
        Kit.el("circle", { cx: m2.x(nx), cy: m2.y(ny), r: 3, fill: P.teal }, vDyn);
        cx = nx; cy = ny;
      }
      arrow(vDyn, m2, A.o[0], A.o[1], P.teal, 4);
      svgText(vDyn, m2.x(A.o[0]), m2.y(A.o[1]) + 26, "output o of " + EX.tokens[qi], { "font-size": 13, fill: P.teal, "font-weight": 700, "text-anchor": "middle", "paint-order": "stroke", stroke: P.paper, "stroke-width": 4 });
      sq.setAttribute("role", "group"); sq.setAttribute("aria-label", "Plane of queries and keys. The query of " + EX.tokens[qi] + " is " + vec(q) + ". Its weights: " + EX.tokens.slice(0, vis).map(function (t, k) { return t + " " + pcs(A.w[k]); }).join(", ") + ".");
      sv.setAttribute("role", "img"); sv.setAttribute("aria-label", "Plane of values. The output is the weighted sum " + vec(A.o, 2) + ".");
      // table
      var h = "";
      for (j = 0; j < 4; j++) {
        var vs = j < vis;
        h += '<div class="ex-row' + (j === row && vs ? " on" : "") + (vs ? "" : " off") + '" data-j="' + j + '"' + (vs ? ' tabindex="0" role="button" aria-label="Show the working for ' + EX.tokens[j] + '"' : "") + "><span class='ex-tok'>" + EX.tokens[j] + "</span><span>" + (vs ? f2(A.raw[j]) : "masked") + "</span><span>" + (vs ? f2(A.sc[j]) : "") + "</span><span class='ex-w'><span class='ex-bar'><i style='width:" + (100 * A.w[j]).toFixed(1) + "%'></i></span><b>" + (vs ? pcs(A.w[j]) : "0%") + "</b></span></div>";
      }
      tab.innerHTML = '<div class="ex-row ex-head"><span>token</span><span>q · k</span><span>÷ √2</span><span>weight</span></div>' + h;
      Array.prototype.forEach.call(tab.querySelectorAll(".ex-row[data-j]"), function (n) {
        function pick() { var k = Number(n.getAttribute("data-j")); if (k < vis) { row = k; update(); } }
        n.addEventListener("click", pick); n.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
      });
      // equation boxes for the chosen row
      var k = K[row], tok = EX.tokens[row];
      var e1 = box(f2(q[0] * k[0]), f1(q[0]) + " × " + f1(k[0]), "c-coral") + op("+") + box(f2(q[1] * k[1]), f1(q[1]) + " × " + f1(k[1]), "c-coral") + op("=") + box(f2(A.raw[row]), "q · k for " + tok, "c-coral") + op("÷") + box("1.41", "√2, since d<sub>k</sub> = 2", "c-sand") + op("=") + box(f2(A.sc[row]), "score s", "c-navy");
      var e2 = box(f2(Math.exp(A.sc[row])), "e<sup>s</sup>, the score exponentiated", "c-navy") + op("÷") + box(f2(A.Z), "sum of e<sup>s</sup> over the " + vis + " visible tokens", "c-sand") + op("=") + box(pcs(A.w[row]), "weight of " + tok, "c-teal");
      var e3 = "", terms = [];
      for (j = 0; j < vis; j++) terms.push(box(f2(A.w[j]) + " × " + vec(V[j]), "w × v of " + EX.tokens[j], "c-teal"));
      e3 = terms.join(op("+")) + op("=") + box(vec(A.o, 2), "output o of " + EX.tokens[qi], "c-teal");
      eq.innerHTML = '<div class="eqn" aria-label="Score and scaling for ' + tok + '">' + e1 + '</div><div class="eqn" aria-label="Softmax for ' + tok + '">' + e2 + '</div><div class="eqn" aria-label="The weighted sum of the values">' + e3 + "</div>";
      st.textContent = vis < 4 ? EX.tokens[qi] + " reads only itself and the tokens before it, so the later keys are dashed and get weight 0. The numbers are made up." : "Drag the coral tip (the query) or a navy tip (a key). Click a row of the table to see its working. The numbers are made up.";
    }
    update(); Kit.onTheme(update);
  }

  // ---------- 4. why divide by sqrt(d_k) ----------
  function scaleFig() {
    var ctl = Kit.$("sc-controls"), read = Kit.$("sc-read"), st = Kit.$("sc-status");
    var DK = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], ix = 6, scaled = true, cache = {}, NT = 300, NK = 8, sampleSeed = 0;
    function raw(dk) {
      if (cache[dk]) return cache[dk];
      var rnd = Kit.rng(4127 + dk);   // one fixed stream per key size, so the figure looks the same on every visit
      function gauss2() { var u = 0, v = 0; while (!u) u = rnd(); v = rnd(); var r = Math.sqrt(-2 * Math.log(u)); return [r * Math.cos(2 * Math.PI * v), r * Math.sin(2 * Math.PI * v)]; }
      var n = NT * NK, out = new Float64Array(n), q = new Float64Array(dk), k = new Float64Array(dk);
      for (var t = 0; t < NT; t++) {
        for (var a = 0; a < dk; a += 2) { var g = gauss2(); q[a] = g[0]; if (a + 1 < dk) q[a + 1] = g[1]; }
        for (var j = 0; j < NK; j++) {
          for (a = 0; a < dk; a += 2) { g = gauss2(); k[a] = g[0]; if (a + 1 < dk) k[a + 1] = g[1]; }
          var s = 0; for (a = 0; a < dk; a++) s += q[a] * k[a]; out[t * NK + j] = s;
        }
      }
      cache[dk] = out; return out;
    }
    function stats(dk, sc) {
      var r = raw(dk), f = sc ? 1 / Math.sqrt(dk) : 1, mean = 0, mx = 0, i, j, ssum = 0;
      for (i = 0; i < r.length; i++) { mean += r[i] * f; } mean /= r.length;
      for (i = 0; i < r.length; i++) ssum += Math.pow(r[i] * f - mean, 2);
      var sd = Math.sqrt(ssum / r.length);
      for (var t = 0; t < NT; t++) { var z = []; for (j = 0; j < NK; j++) z.push(r[t * NK + j] * f); mx += Math.max.apply(null, softmaxArr(z)); }
      return { sd: sd, maxw: mx / NT };
    }
    var sl = Kit.slider(ctl, { label: "Key size d_k", min: 0, max: DK.length - 1, step: 1, value: ix, show: function (v) { return String(DK[v]); } }, function (v) { ix = v; draw(); });
    var sw = Kit.html("label", { class: "chk" }, ctl); var cb = Kit.html("input", { type: "checkbox", checked: "checked" }, sw); Kit.html("span", {}, sw, " Divide the scores by √d_k");
    cb.addEventListener("change", function () { scaled = cb.checked; draw(); });
    var nb = Kit.html("span", { class: "row-chips" }, ctl); Kit.button(nb, "New sample of 8 keys", function () { sampleSeed++; draw(); });
    var lineDone = false;
    function tickDK() { return Kit.width(Kit.$("sc-line")) < 520 ? [1, 4, 16, 64, 256, 1024] : DK; }   // fewer ticks on a phone, so the labels stay level
    function draw() {
      var dk = DK[ix], f = scaled ? 1 / Math.sqrt(dk) : 1, r = raw(dk), S = stats(dk, scaled);
      var xs = []; for (var i = 0; i < r.length; i++) xs.push(r[i] * f);
      Kit.plot("sc-hist", function (P) { return [{ type: "histogram", x: xs, nbinsx: 40, marker: { color: scaled ? P.teal : P.coral, opacity: 0.75, line: { color: P.paper, width: 0.5 } }, hovertemplate: "scores %{x}: %{y} of " + xs.length + "<extra></extra>" }]; },
        { margin: { l: 46, r: 10, t: 22, b: 42 }, xaxis: { title: { text: scaled ? "score q·k / √d_k" : "score q·k", standoff: 8 }, automargin: true }, yaxis: { title: { text: "count of " + xs.length }, automargin: true },
          annotations: [{ x: 0.98, y: 0.96, xref: "paper", yref: "paper", xanchor: "right", yanchor: "top", showarrow: false, text: "spread " + S.sd.toFixed(2), font: { size: 14 } }] });
      // one sample of 8 keys: the first trial offset by sampleSeed
      var t0 = (sampleSeed * 7) % NT, z = []; for (i = 0; i < NK; i++) z.push(r[t0 * NK + i] * f);
      var w = softmaxArr(z);
      Kit.plot("sc-bars", function (P) { return [{ type: "bar", x: z.map(function (_, i) { return i + 1; }), y: w, marker: { color: w.map(function (v) { return v === Math.max.apply(null, w) ? P.coral : P.navy; }), opacity: 0.85 }, text: w.map(function (v) { return v >= 0.995 ? "100%" : (100 * v).toFixed(0) + "%"; }), textposition: "outside", cliponaxis: false, hovertemplate: "%{x}: weight %{y:.3f}<extra></extra>" }]; },
        { margin: { l: 44, r: 10, t: 22, b: 42 }, yaxis: { title: { text: "softmax weight" }, range: [0, 1.15], automargin: true }, xaxis: { title: { text: Kit.width(Kit.$("sc-bars")) < 300 ? "key number" : "key number (8 random keys, one query)", standoff: 8 }, dtick: 1, automargin: true } });
      if (!lineDone) {
        var us = DK.map(function (d) { return stats(d, false).maxw; }), ss = DK.map(function (d) { return stats(d, true).maxw; }); lineDone = true;
        Kit.plot("sc-line", function (P) { return [
          { type: "scatter", mode: "lines+markers", name: "no scaling", x: DK, y: us, line: { color: P.coral, width: 2 }, marker: { size: 6 }, hovertemplate: "d_k %{x}: %{y:.2f}<extra>no scaling</extra>" },
          { type: "scatter", mode: "lines+markers", name: "divided by √d_k", x: DK, y: ss, line: { color: P.teal, width: 2 }, marker: { size: 6 }, hovertemplate: "d_k %{x}: %{y:.2f}<extra>scaled</extra>" },
          { type: "scatter", mode: "markers", x: [dk], y: [stats(dk, scaled).maxw], marker: { size: 13, color: "rgba(0,0,0,0)", line: { color: P.ink, width: 2.5 } }, hoverinfo: "skip", showlegend: false }]; },
          { margin: { l: 52, r: 10, t: 26, b: 44 }, showlegend: true, legend: { orientation: "h", y: 1.2, x: 0 }, xaxis: { type: "log", title: { text: "key size d_k", standoff: 8 }, tickvals: tickDK(), ticktext: tickDK().map(String), tickangle: 0, automargin: true }, yaxis: { title: { text: "mean largest weight" }, range: [0, 1.05], automargin: true } });
        window.__scLine = { us: us, ss: ss };
      } else {
        var L = window.__scLine;
        Kit.plot("sc-line", function (P) { return [
          { type: "scatter", mode: "lines+markers", name: "no scaling", x: DK, y: L.us, line: { color: P.coral, width: 2 }, marker: { size: 6 }, hovertemplate: "d_k %{x}: %{y:.2f}<extra>no scaling</extra>" },
          { type: "scatter", mode: "lines+markers", name: "divided by √d_k", x: DK, y: L.ss, line: { color: P.teal, width: 2 }, marker: { size: 6 }, hovertemplate: "d_k %{x}: %{y:.2f}<extra>scaled</extra>" },
          { type: "scatter", mode: "markers", x: [dk], y: [scaled ? L.ss[ix] : L.us[ix]], marker: { size: 13, color: "rgba(0,0,0,0)", line: { color: P.ink, width: 2.5 } }, hoverinfo: "skip", showlegend: false }]; },
          { margin: { l: 52, r: 10, t: 26, b: 44 }, showlegend: true, legend: { orientation: "h", y: 1.2, x: 0 }, xaxis: { type: "log", title: { text: "key size d_k", standoff: 8 }, tickvals: tickDK(), ticktext: tickDK().map(String), tickangle: 0, automargin: true }, yaxis: { title: { text: "mean largest weight" }, range: [0, 1.05], automargin: true } });
      }
      var sd0 = stats(dk, false).sd;
      read.innerHTML = "<span>d<sub>k</sub> = <b>" + dk + "</b></span><span>Spread of the scores: <b>" + S.sd.toFixed(2) + "</b>" + (scaled ? " (" + sd0.toFixed(2) + " before dividing; √" + dk + " = " + Math.sqrt(dk).toFixed(2) + ")" : " (√" + dk + " = " + Math.sqrt(dk).toFixed(2) + ")") + "</span><span>Mean largest weight of 8: <b>" + (100 * S.maxw).toFixed(0) + "%</b></span><span>This sample: <b>" + pcs(Math.max.apply(null, w)) + "</b> on one key</span>";
      Kit.$("sc-hist").setAttribute("aria-label", "Histogram of " + xs.length + " dot products of random vectors with " + dk + " numbers" + (scaled ? ", divided by the square root of " + dk : "") + ". Their spread is " + S.sd.toFixed(2) + ".");
      Kit.$("sc-bars").setAttribute("aria-label", "Softmax weights of eight random keys. The largest is " + pcs(Math.max.apply(null, w)) + ".");
      Kit.$("sc-line").setAttribute("aria-label", "Mean largest softmax weight against key size. Without scaling it climbs toward 100%. With scaling it stays near 40%.");
      st.textContent = "The vectors are random numbers with mean 0 and variance 1, drawn in your browser (300 queries with 8 keys per query).";
    }
    draw();
  }

  // ---------- 5a. heat map with a causal mask ----------
  function heatFig() {
    var bar = Kit.$("hm-bar"), svg = Kit.$("hm-svg"), read = Kit.$("hm-read"), st = Kit.$("hm-status");
    var si = 0, hid = "L4H11", stage = 2, mask = true, cell = null;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ATT.sentences.map(function (s) { return SENTLABEL[s.id]; }), 0, function (i) { si = i; cell = null; draw(); });
    var hsel = Kit.select(bar, "Head", ATT.heads.map(function (h) { return { value: h.id, label: h.id + ", " + h.name }; }), hid, function (v) { hid = v; cell = null; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["1 Scores", "2 Mask", "3 Softmax"], stage, function (i) { stage = i; draw(); });
    var lab = Kit.html("label", { class: "chk" }, bar), cb = Kit.html("input", { type: "checkbox", checked: "checked" }, lab); Kit.html("span", {}, lab, " Causal mask");
    cb.addEventListener("change", function () { mask = cb.checked; draw(); });
    function draw() {
      var P = Kit.palette(), S = ATT.sentences[si], toks = S.tokens, n = toks.length, H = S.heads[hid];
      var wid = Math.max(300, Kit.width(svg.parentNode) - 36), maxlen = 0; toks.forEach(function (t) { maxlen = Math.max(maxlen, showTok(t).length); });
      var left = Math.min(96, 7.4 * maxlen + 26), right = mask ? 8 : 62;
      var cs = Math.min(26, Math.floor((wid - left - right) / n)), gw = cs * n, Wd = left + gw + right, upright = cs < 15;
      var topM = upright ? Math.min(90, 6.3 * maxlen + 24) : Math.min(86, 7 * maxlen * 0.75 + 30), Ht = topM + gw + 22;   // upright labels need a little more room above the grid
      svgSize(svg, Wd, Ht); clear(svg);
      var defs = Kit.el("defs", {}, svg), pat = Kit.el("pattern", { id: "hm-hatch", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" }, defs);
      Kit.el("rect", { width: 6, height: 6, fill: P.paper }, pat); Kit.el("line", { x1: 0, y1: 0, x2: 0, y2: 6, stroke: P.muted, "stroke-width": 1.6, opacity: 0.55 }, pat);
      var clip = 12, futs = [];
      for (var i = 0; i < n; i++) {
        var fsum = 0; for (var j = 0; j < n; j++) if (j > i) fsum += H.wn[i][j]; futs.push(fsum);
        svgText(svg, left - 6, topM + i * cs + cs / 2 + 4, showTok(toks[i]), { "text-anchor": "end", "font-size": cs < 17 ? 11 : 12.5, fill: P.ink, "font-family": MONO });
        var tx = left + i * cs + cs / 2, ty = topM - 6;
        svgText(svg, upright ? tx + 3.5 : tx, ty, showTok(toks[i]), { "font-size": upright ? 9.5 : (cs < 17 ? 11 : 12.5), fill: P.ink, "font-family": MONO, transform: "rotate(" + (upright ? -90 : -55) + " " + (upright ? tx + 3.5 : tx) + " " + ty + ")", "text-anchor": "start" });
      }
      svgText(svg, 4, topM - 6, "queries ↓", { "font-size": 11, fill: P.muted });
      for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
        var x = left + j * cs, y = topM + i * cs, masked = mask && j > i, fillc, fo, v;
        if (stage === 0 || (stage === 1 && !masked)) {
          v = H.s[i][j]; fillc = v >= 0 ? P.teal : P.coral; fo = Math.min(1, Math.abs(v) / clip);
        } else if (stage === 1 && masked) { fillc = "url(#hm-hatch)"; fo = 1; }
        else { v = mask ? H.w[i][j] : H.wn[i][j]; fillc = masked ? "none" : P.teal; fo = masked ? 0 : Math.pow(v, 0.75); }
        var r = Kit.el("rect", { x: x, y: y, width: cs - 1, height: cs - 1, fill: fillc, "fill-opacity": fo.toFixed(3), stroke: stage === 2 && masked ? "none" : P.rule, "stroke-width": 0.5, style: "cursor:pointer" }, svg);
        if (stage === 2 && !masked) r.setAttribute("stroke", P.rule);
        if (cell && cell[0] === i && cell[1] === j) { Kit.el("rect", { x: x - 1, y: y - 1, width: cs + 1, height: cs + 1, fill: "none", stroke: P.ink, "stroke-width": 2.2, "pointer-events": "none" }, svg); }
        (function (a, b) { r.addEventListener("click", function () { cell = [a, b]; draw(); }); })(i, j);
      }
      // diagonal guide
      Kit.el("line", { x1: left, y1: topM, x2: left + gw, y2: topM + gw, stroke: P.ink2, "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0.5, "pointer-events": "none" }, svg);
      // future-share bars when the mask is off
      if (!mask && stage === 2) {
        svgText(svg, left + gw + 6, topM - 6, "weight on later tokens", { "font-size": 10.5, fill: P.coral, transform: "rotate(-55 " + (left + gw + 6) + " " + (topM - 6) + ")" });
        for (i = 0; i < n; i++) { var bw = futs[i] * (right - 14); Kit.el("rect", { x: left + gw + 6, y: topM + i * cs + 2, width: Math.max(0, bw), height: cs - 5, fill: P.coral, "fill-opacity": 0.8 }, svg); if (futs[i] >= 0.1 && cs >= 13) svgText(svg, left + gw + 8 + bw, topM + i * cs + cs / 2 + 4, pcs(futs[i]), { "font-size": 10.5, fill: P.ink2 }); }
      }
      svgText(svg, left + gw / 2, topM + gw + 16, "keys: the token being read →", { "font-size": 11, fill: P.muted, "text-anchor": "middle" });
      var avgFut = futs.slice(0, n - 1).reduce(function (a, b) { return a + b; }, 0) / (n - 1);
      var rd = "<span>Sentence: <b>" + n + " tokens</b>, " + n * n + " scores, <b>" + (mask ? n * (n + 1) / 2 : n * n) + "</b> " + (mask ? "kept" : "used") + "</span>";
      if (stage === 2) rd += "<span>Mean weight on later tokens: <b>" + (mask ? "0%" : pcs(avgFut)) + "</b></span>";
      if (cell) {
        var ci = cell[0], cj = cell[1], mk = mask && cj > ci;
        rd += "<span>Cell: <b>" + esc(showTok(toks[ci])) + "</b> reading <b>" + esc(showTok(toks[cj])) + "</b>: score <b>" + f1(H.s[ci][cj]) + "</b>" + (mk ? ", masked, weight <b>0</b>" : ", weight <b>" + (stage === 2 || true ? (mask ? H.w[ci][cj] : H.wn[ci][cj]).toFixed(3) : "") + "</b>") + "</span>";
      } else rd += "<span>Click a cell to read its score and weight.</span>";
      read.innerHTML = rd;
      svg.setAttribute("role", "img");
      svg.setAttribute("aria-label", "Table of " + n + " by " + n + " " + ["scaled scores", "scores with the mask applied", "attention weights"][stage] + " for head " + hid + (mask ? ", with the causal mask, so cells above the diagonal are empty." : ", without a mask, so later tokens receive weight. On average " + pcs(avgFut) + " of a row's weight goes to later tokens."));
      st.textContent = [
        "Stage 1: all queries against all keys, q·k divided by √64 = 8. Teal is positive and coral is negative; colour saturates at ±12, and a click shows the true value.",
        mask ? "Stage 2: the causal mask replaces the scores above the diagonal with −∞ (hatched). Later tokens cannot be read." : "Stage 2: with the mask off no score is replaced, so this table is the same as stage 1.",
        mask ? "Stage 3: softmax runs along a row. The masked cells (hatched in stage 2) became e^(−∞) = 0, so all rows add up to 1 over the tokens they may read." : "Stage 3 without the mask: the same scores, softmax over the whole row. A token now gives weight to words that come after it. GPT-2 was trained with the mask, so this shows what would leak, not what the model would compute."
      ][stage];
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 5b. what goes wrong without the mask ----------
  function leakFig() {
    var bar = Kit.$("lk-bar"), read = Kit.$("lk-read"), st = Kit.$("lk-status");
    var show = 0, step = 400;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Both models", "Trained with the mask", "Trained without the mask"], 0, function (i) { show = i; draw(); });
    var sl = Kit.slider(bar, { label: "Training step", min: 0, max: 400, step: 20, value: 400 }, function (v) { step = v; draw(); });
    function draw() {
      var A = LEAK.mask, B = LEAK.nomask, xs = A.map(function (r) { return r[0]; }), k = step / 20;
      Kit.plot("lk-plot", function (P) {
        var tr = [];
        if (show !== 2) { tr.push({ type: "scatter", mode: "lines", name: "with mask: loss on training reports", x: xs, y: A.map(function (r) { return r[1]; }), line: { color: P.teal, width: 3 } }); tr.push({ type: "scatter", mode: "lines", name: "with mask: loss on new reports", x: xs, y: A.map(function (r) { return r[2]; }), line: { color: P.teal, width: 2, dash: "dot" } }); }
        if (show !== 1) { tr.push({ type: "scatter", mode: "lines", name: "without mask: loss on training reports", x: xs, y: B.map(function (r) { return r[1]; }), line: { color: P.coral, width: 3 } }); tr.push({ type: "scatter", mode: "lines", name: "without mask: loss on new reports, mask on", x: xs, y: B.map(function (r) { return r[2]; }), line: { color: P.coral, width: 2, dash: "dot" } }); }
        tr.push({ type: "scatter", mode: "lines", name: "guessing at random", x: [0, 400], y: [LEAK.uniform, LEAK.uniform], line: { color: P.muted, width: 1, dash: "dash" }, hoverinfo: "skip" });
        tr.push({ type: "scatter", mode: "markers", x: [step, step, step, step].slice(0, 4), y: [A[k][1], A[k][2], B[k][1], B[k][2]], marker: { size: 9, color: [P.teal, P.teal, P.coral, P.coral], line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip", showlegend: false });
        return tr;
      }, function (P) { var nar = Kit.width(Kit.$("lk-plot")) < 520; Kit.$("lk-plot").style.minHeight = nar ? "400px" : ""; return { margin: { l: 52, r: 10, t: 12, b: nar ? 150 : 104 }, showlegend: true, legend: { orientation: "h", yref: "container", yanchor: "bottom", y: 0.01, x: 0, font: { size: 12 } }, xaxis: { title: { text: "training step", standoff: 8 }, automargin: true }, yaxis: { title: { text: nar ? "loss, log axis" : "loss per word, log axis" }, automargin: true, type: "log", range: [Math.log10(0.03), Math.log10(7)], tickvals: [0.05, 0.1, 0.3, 1, 3], ticktext: ["0.05", "0.1", "0.3", "1", "3"] } }; });
      read.innerHTML = "<span>Step <b>" + step + "</b></span><span>With the mask: training <b>" + f2(A[k][1]) + "</b>, new reports <b>" + f2(A[k][2]) + "</b></span><span>Without the mask: training <b>" + f2(B[k][1]) + "</b>, new reports with the mask on <b>" + f2(B[k][2]) + "</b></span>";
      Kit.$("lk-plot").setAttribute("aria-label", "Loss curves of two tiny models. Without the mask the training loss falls to " + f2(B[B.length - 1][1]) + ", below the " + f2(A[A.length - 1][1]) + " of the masked model, but on new reports with the mask on it is " + f2(B[B.length - 1][2]) + " against " + f2(A[A.length - 1][2]) + ".");
      st.textContent = "Made-up reports from templates (D3), a model with one attention head, a vocabulary of 92 words, 400 training reports and 100 new ones. Loss in nats per word (D3).";
    }
    draw();
  }

  // ---------- 6. all 144 heads ----------
  function headsFig() {
    var bar = Kit.$("mh-bar"), svg = Kit.$("mh-grid"), pins = Kit.$("mh-pins"), read = Kit.$("mh-read"), st = Kit.$("mh-status");
    var measure = "first", pinned = [[5, 5], [4, 11], [1, 10]];
    var NAMES = { first: "weight on the first token", prev: "weight on the previous token", self: "weight on the token itself" };
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["First token", "Previous token", "Itself"], 0, function (i) { measure = ["first", "prev", "self"][i]; drawGrid(); });
    Kit.button(bar, "Pin the four heads of the sentence view", function () { pinned = [[4, 11], [0, 1], [5, 5], [1, 10]]; drawGrid(); drawPins(); });
    Kit.button(bar, "Clear pins", function () { pinned = []; drawGrid(); drawPins(); });
    var featured = ATT.heads.map(function (h) { return [h.layer, h.head]; });
    function isPinned(l, h) { return pinned.some(function (p) { return p[0] === l && p[1] === h; }); }
    function drawGrid() {
      var P = Kit.palette(), M = ALL.stats[measure], wid = Math.max(300, Kit.width(svg.parentNode) - 36), lab = 44, cs = Math.min(30, Math.floor((wid - lab - 6) / 12)), top = 40;
      svgSize(svg, lab + 12 * cs + 6, top + 12 * cs + 8); clear(svg);
      svgText(svg, lab, 14, "head 0 → 11", { "font-size": 11.5, fill: P.muted }); svgText(svg, 2, top - 6, "layer", { "font-size": 11.5, fill: P.muted });
      for (var l = 0; l < 12; l++) {
        svgText(svg, lab - 6, top + l * cs + cs / 2 + 4, String(l), { "text-anchor": "end", "font-size": 11.5, fill: P.muted });
        for (var h = 0; h < 12; h++) {
          var v = M[l][h], x = lab + h * cs, y = top + l * cs;
          var r = Kit.el("rect", { x: x, y: y, width: cs - 2, height: cs - 2, rx: 3, fill: P.teal, "fill-opacity": Math.min(1, v).toFixed(2), stroke: isPinned(l, h) ? P.ink : P.rule, "stroke-width": isPinned(l, h) ? 2.4 : 0.6, style: "cursor:pointer", tabindex: 0, role: "button", "aria-label": "Layer " + l + ", head " + h + ": " + NAMES[measure] + " " + pcs(v) + ". Activate to pin or unpin." }, svg);
          var isF = featured.some(function (f) { return f[0] === l && f[1] === h; });
          if (isF) Kit.el("circle", { cx: x + (cs - 2) / 2, cy: y + (cs - 2) / 2, r: 2.6, fill: P.coral, "pointer-events": "none" }, svg);
          (function (a, b) { function tg() { if (isPinned(a, b)) pinned = pinned.filter(function (p) { return !(p[0] === a && p[1] === b); }); else { pinned.push([a, b]); if (pinned.length > 4) pinned.shift(); } drawGrid(); drawPins(); }
            r.addEventListener("click", tg); r.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); tg(); } }); r.addEventListener("mouseenter", function () { read.innerHTML = "<span>Layer <b>" + a + "</b>, head <b>" + b + "</b>: " + NAMES[measure] + " <b>" + pcs(M[a][b]) + "</b></span>"; }); })(l, h);
        }
      }
      var cnt = 0; for (l = 0; l < 12; l++) for (h = 0; h < 12; h++) if (ALL.stats.first[l][h] > 0.8) cnt++;
      svg.setAttribute("role", "group"); svg.setAttribute("aria-label", "Grid of the 144 heads of GPT-2 small coloured by the " + NAMES[measure] + ". " + cnt + " heads put more than 80% of their weight on the first token.");
      read.innerHTML = "<span>Heads with more than 80% of their weight on the first token: <b>" + cnt + " of 144</b></span><span>Coral dots mark the four heads of the sentence view.</span>";
      st.textContent = "Averages over " + ALL.ntexts + " short texts (" + ALL.ntokens + " tokens, from the third token of a text). Click a head to pin it, up to four.";
    }
    function drawPins() {
      var P = Kit.palette(); clear(pins);
      if (!pinned.length) { Kit.html("p", { class: "note" }, pins, "No head pinned. Click squares in the grid."); return; }
      var n = ALL.tokens.length, wrap = Kit.html("div", { class: "pinwrap" }, pins), cs = 11;
      pinned.forEach(function (p) {
        var d = Kit.html("div", { class: "pin" }, wrap), id = "L" + p[0] + "H" + p[1], f = ATT.heads.filter(function (h) { return h.id === id; })[0];
        Kit.html("div", { class: "pin-t" }, d, id + (f ? ", " + f.name : ""));
        var sv_ = Kit.el("svg", { role: "img", "aria-label": "Attention table of head " + id + " on the Tag 4127 sentence" }, d); svgSize(sv_, 22 + n * cs, 14 + n * cs);
        var T = ALL.tri[p[0]][p[1]];
        for (var i = 0; i < n; i++) for (var j = 0; j <= i; j++) {
          var r = Kit.el("rect", { x: 20 + j * cs, y: 12 + i * cs, width: cs - 1, height: cs - 1, fill: P.teal, "fill-opacity": (Math.pow(T[i][j] / 100, 0.75)).toFixed(2) }, sv_);
          var tt = Kit.el("title", {}, r); tt.textContent = showTok(ALL.tokens[i]) + " reads " + showTok(ALL.tokens[j]) + ": " + T[i][j] + "%";
        }
        for (i = 0; i < n; i += 4) { svgText(sv_, 17, 12 + i * cs + cs - 2, String(i), { "text-anchor": "end", "font-size": 9, fill: P.muted }); svgText(sv_, 20 + i * cs + 1, 9, String(i), { "font-size": 9, fill: P.muted }); }
      });
      Kit.html("p", { class: "note" }, pins, "Positions: " + ALL.tokens.map(function (t, i) { return i + " " + showTok(t); }).join(" · ") + ". Rows are the reading token, columns the token read.");
    }
    drawGrid(); drawPins(); Kit.onTheme(function () { drawGrid(); drawPins(); }); window.addEventListener("resize", Kit.debounce(function () { drawGrid(); }, 200));
  }
})();
