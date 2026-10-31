// D5 The transformer. Figures: a token stepping through one block (a tiny block with made-up weights, computed in the page), the residual stream of
// GPT-2 with a logit-lens probe, sinusoidal and learned position vectors, RoPE as rotation of 2D pairs, a decoder-only and encoder-only comparison
// (GPT-2 and BERT on one sentence), and the generation loop with a key-value cache. The pure logic is exported with module.exports for node tests.
(function () {
  "use strict";
  // ---------- pure logic ----------
  function matmul(A, B) {
    var n = A.length, m = B[0].length, k = B.length, out = [], i, j, t;
    for (i = 0; i < n; i++) { var r = []; for (j = 0; j < m; j++) { var s = 0; for (t = 0; t < k; t++) s += A[i][t] * B[t][j]; r.push(s); } out.push(r); }
    return out;
  }
  function layerNorm(x, eps) { var d = x.length, mu = 0, v = 0, i; for (i = 0; i < d; i++) mu += x[i]; mu /= d; for (i = 0; i < d; i++) v += (x[i] - mu) * (x[i] - mu); v /= d; var s = Math.sqrt(v + (eps == null ? 1e-5 : eps)); return x.map(function (a) { return (a - mu) / s; }); }
  function gelu(x) { return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))); }
  function softmaxArr(z) { var m = Math.max.apply(null, z), e = z.map(function (x) { return Math.exp(x - m); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (x) { return x / s; }); }
  function addRows(A, B) { return A.map(function (r, i) { return r.map(function (v, j) { return v + B[i][j]; }); }); }
  function norm(v) { return Math.sqrt(v.reduce(function (a, b) { return a + b * b; }, 0)); }
  function cosine(a, b) { var d = 0; for (var i = 0; i < a.length; i++) d += a[i] * b[i]; return d / (norm(a) * norm(b)); }
  // The tiny block of block.json. mode "pre": layer norm before the sub-layers (GPT-2). mode "post": add, then norm (2017 paper). resid false drops the shortcuts.
  function blockRun(B, mode, resid) {
    var X = B.X, n = X.length, H = B.H, dk = B.dk;
    function attn(h) {
      var q = matmul(h, B.Wq), k = matmul(h, B.Wk), v = matmul(h, B.Wv), cat = h.map(function () { return []; }), ws = [];
      for (var hd = 0; hd < H; hd++) {
        var W = [];
        for (var i = 0; i < n; i++) {
          var sc = [];
          for (var j = 0; j <= i; j++) { var s = 0; for (var c = 0; c < dk; c++) s += q[i][hd * dk + c] * k[j][hd * dk + c]; sc.push(s / Math.sqrt(dk)); }
          var w = softmaxArr(sc); while (w.length < n) w.push(0); W.push(w);
          for (c = 0; c < dk; c++) { var o = 0; for (j = 0; j <= i; j++) o += w[j] * v[j][hd * dk + c]; cat[i].push(o); }
        }
        ws.push(W);
      }
      return { out: matmul(cat, B.Wo), w: ws };
    }
    function mlp(h) { return matmul(matmul(h, B.W1).map(function (r) { return r.map(gelu); }), B.W2); }
    var a, m, A;
    if (mode === "pre") {
      var h1 = X.map(function (r) { return layerNorm(r); }); A = attn(h1); a = A.out;
      var x1 = resid ? addRows(X, a) : a, h2 = x1.map(function (r) { return layerNorm(r); }); m = mlp(h2);
      var x2 = resid ? addRows(x1, m) : m;
      return { rows: { x: X, h1: h1, a: a, x1: x1, h2: h2, m: m, x2: x2 }, w: A.w };
    }
    A = attn(X); a = A.out;
    var s1 = resid ? addRows(X, a) : a, y1 = s1.map(function (r) { return layerNorm(r); }); m = mlp(y1);
    var s2 = resid ? addRows(y1, m) : m, y2 = s2.map(function (r) { return layerNorm(r); });
    return { rows: { x: X, a: a, s1: s1, y1: y1, m: m, s2: s2, y2: y2 }, w: A.w };
  }
  function sinPE(pos, d, base) { var v = []; for (var i = 0; i < d / 2; i++) { var w = Math.pow(base, -2 * i / d); v.push(Math.sin(pos * w), Math.cos(pos * w)); } return v; }
  function ropeRot(v, a) { var c = Math.cos(a), s = Math.sin(a); return [v[0] * c - v[1] * s, v[0] * s + v[1] * c]; }
  function kvNumbers(layers, d, n) { return 2 * layers * n * d; }
  if (typeof module !== "undefined" && module.exports) { module.exports = { matmul: matmul, layerNorm: layerNorm, gelu: gelu, blockRun: blockRun, sinPE: sinPE, ropeRot: ropeRot, kvNumbers: kvNumbers, cosine: cosine, norm: norm }; }
  if (typeof document === "undefined") return;

  // ---------- helpers ----------
  var BL = null, G2 = null, MD = null;
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function showTok(t) { t = t.replace(/\\n/g, "↵"); return t.charAt(0) === " " ? "▁" + t.slice(1) : t; }
  function box(val, cap, cls) { return '<span class="t ' + (cls || "") + '"><b>' + val + "</b><i>" + cap + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function f1(x) { return Kit.fmt(x, 1); }
  function f2(x) { return Kit.fmt(x, 2); }
  function nf(n) { return Number(n).toLocaleString("en-US"); }
  function sg(v) { return v < 0 ? "−" + (-v) : String(v); }   // whole numbers with a real minus sign
  function pcs(x) { return x < 0.0005 ? "0%" : x < 0.01 ? (100 * x).toFixed(1) + "%" : x > 0.995 ? "100%" : (100 * x).toFixed(0) + "%"; }
  function svgSize(svg, w, h) { svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.setAttribute("width", w); svg.setAttribute("height", h); svg.style.maxWidth = w + "px"; }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function svgText(parent, x, y, txt, attrs) { var t = Kit.el("text", Object.assign({ x: x, y: y }, attrs || {}), parent); t.textContent = txt; return t; }
  function cells(v, cls) {
    return v.map(function (x) {
      var a = Math.min(1, Math.abs(x) / 2.5), col = x >= 0 ? "var(--teal)" : "var(--coral)";
      return '<span class="vc' + (cls ? " " + cls : "") + '" style="background:color-mix(in srgb,' + col + " " + Math.round(8 + 62 * a) + '%, transparent)">' + Kit.fmt(x, 2) + "</span>";
    }).join("");
  }

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/block.json"), Kit.data("data/gpt2.json"), Kit.data("data/models.json")]).then(function (r) {
      BL = r[0]; G2 = r[1]; MD = r[2];
      blockFig(); streamFig(); positionFig(); ropeFig(); familyFig(); cacheFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. one block ----------
  var STEPS = {
    pre: [
      { id: "x", short: "Input", stream: true, name: "The stream enters the block" },
      { id: "h1", short: "Layer norm", stream: false, name: "Layer norm, for attention" },
      { id: "a", short: "Attention", stream: false, name: "Attention reads the earlier tokens" },
      { id: "x1", short: "Add", stream: true, name: "Add the correction to the stream" },
      { id: "h2", short: "Layer norm", stream: false, name: "Layer norm, for the MLP" },
      { id: "m", short: "MLP", stream: false, name: "The MLP works on this token alone" },
      { id: "x2", short: "Add", stream: true, name: "Add the second correction" }
    ],
    post: [
      { id: "x", short: "Input", stream: true, name: "The vector enters the block" },
      { id: "a", short: "Attention", stream: false, name: "Attention reads the earlier tokens" },
      { id: "s1", short: "Add", stream: true, name: "Add the correction" },
      { id: "y1", short: "Norm", stream: true, name: "Norm the sum" },
      { id: "m", short: "MLP", stream: false, name: "The MLP works on this token alone" },
      { id: "s2", short: "Add", stream: true, name: "Add the second correction" },
      { id: "y2", short: "Norm", stream: true, name: "Norm the sum" }
    ]
  };
  function blockFig() {
    var bar = Kit.$("bl-bar"), svg = Kit.$("bl-svg"), rowsEl = Kit.$("bl-rows"), ttl = Kit.$("bl-title"), txt = Kit.$("bl-text"), eq = Kit.$("bl-eq"), read = Kit.$("bl-read"), att = Kit.$("bl-att");
    var tok = 2, step = 0, mode = "pre", resid = true, timer = 0;
    var toks = BL.tokens;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), toks.map(function (t) { return "Follow: " + t; }), tok, function (i) { tok = i; draw(); });
    var bBack = Kit.button(bar, "◀ Back", function () { stop(); step = Math.max(0, step - 1); draw(); });
    var bNext = Kit.button(bar, "Step ▶", function () { stop(); step = Math.min(6, step + 1); draw(); });
    var bPlay = Kit.button(bar, "Play", function () { if (timer) { stop(); return; } if (step >= 6) step = 0; bPlay.textContent = "Pause"; draw(); timer = setInterval(function () { if (step >= 6) { stop(); return; } step++; draw(); }, 1300); });
    Kit.button(bar, "Reset", function () { stop(); step = 0; draw(); });
    var bar2 = Kit.$("bl-bar2");
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar2), ["Norm first (GPT-2)", "Norm after the add (2017 paper)"], 0, function (i) { mode = i === 0 ? "pre" : "post"; draw(); });
    var lab = Kit.html("label", { class: "chk" }, bar2), cb = Kit.html("input", { type: "checkbox", checked: "checked" }, lab); Kit.html("span", {}, lab, " Keep the shortcut (the add)");
    cb.addEventListener("change", function () { resid = cb.checked; draw(); });
    function stop() { if (timer) clearInterval(timer); timer = 0; bPlay.textContent = "Play"; }
    var TEXT = {
      pre: [
        "The token enters the block as a vector of 4 numbers (made up; GPT-2 small has 768). This vector is the stream. All the parts of the block read it or add to it.",
        "Layer norm rescales a <em>copy</em> of the stream to mean 0 and spread 1 (<a href=\"../b6-deep-networks/#norm\">B6</a>). The stream itself is untouched. The copy goes to attention.",
        "Attention (<a href=\"../d4-self-attention/#heads\">D4</a>) turns the normalised copies of the tokens up to this one into q, k and v, mixes the values with softmax weights, and returns a correction a. Two heads of 2 numbers, joined and mixed by W<sub>O</sub>.",
        "Add: the correction goes onto the stream, x<sub>1</sub> = x + a. The stream still contains x.",
        "A second layer norm, again on a copy. The stream is not changed.",
        "The MLP expands the copy to 8 numbers, applies GELU and shrinks it back to 4. It works on this token alone and reads no other token. Its output m is a second correction.",
        "Add again: x<sub>2</sub> = x<sub>1</sub> + m. This is the output of the block and the input of the next block."
      ],
      post: [
        "In the 2017 design the token enters as a vector and attention reads it directly, with no norm first.",
        "Attention returns a correction a from the tokens up to this one.",
        "Add: s = x + a.",
        "Norm: the sum is rescaled to mean 0 and spread 1. In this order the norm lies on the path of the stream, so all blocks rescale it.",
        "The MLP works on the normed vector of this token alone.",
        "Add: s = y<sub>1</sub> + m.",
        "A second norm. The result is the output of the block."
      ]
    };
    function drawDiagram(P, S, cur) {
      var W = 300, H = 470, sx = 52, bx = 168, bw = 112, bh = 28;
      svgSize(svg, W, H); clear(svg);
      var pre = mode === "pre";
      // geometry: elements 1..6 in order
      var el = pre ? [
        { t: "branch", y: 392, lab: "layer norm" }, { t: "branch", y: 330, lab: "attention" }, { t: "add", y: 262 },
        { t: "branch", y: 200, lab: "layer norm" }, { t: "branch", y: 138, lab: "MLP" }, { t: "add", y: 70 }
      ] : [
        { t: "branch", y: 372, lab: "attention" }, { t: "add", y: 306 }, { t: "norm", y: 258 },
        { t: "branch", y: 196, lab: "MLP" }, { t: "add", y: 130 }, { t: "norm", y: 82 }
      ];
      var yIn = 446, yOut = 24;
      var forks = pre ? [{ y: 424, to: 3 }, { y: 232, to: 6 }] : [{ y: 424, to: 2 }, { y: 232, to: 5 }];
      // stream line (grey), then progress (teal)
      Kit.el("line", { x1: sx, y1: yIn, x2: sx, y2: yOut + 8, stroke: P.rule, "stroke-width": 5, "stroke-linecap": "round" }, svg);
      // stream-position of the token at this step
      function posY(s) { // y of the stream marker after step s
        if (s === 0) return yIn;
        var e = el[s - 1]; if (e.t !== "branch") return e.y;
        // branch step: the stream marker waits at the fork point before it
        var f = forks.filter(function (f) { return f.to > s; })[0]; return f ? f.y : yIn;
      }
      var yMark = posY(cur);
      Kit.el("line", { x1: sx, y1: yIn, x2: sx, y2: yMark, stroke: P.teal, "stroke-width": 5, "stroke-linecap": "round" }, svg);
      // shortcut segments when the shortcut is off
      forks.forEach(function (f) { if (!resid) { Kit.el("line", { x1: sx, y1: f.y, x2: sx, y2: el[f.to - 1].y + 10, stroke: P.bg, "stroke-width": 7 }, svg); Kit.el("line", { x1: sx, y1: f.y, x2: sx, y2: el[f.to - 1].y + 10, stroke: P.coral, "stroke-width": 2, "stroke-dasharray": "3 4" }, svg); svgText(svg, sx - 10, (f.y + el[f.to - 1].y) / 2 + 4, "cut", { "font-size": 11, fill: P.coral, "text-anchor": "end" }); } });
      // branches
      forks.forEach(function (f) {
        var yEnd = el[f.to - 1].y, midx = bx + bw / 2 + 10;
        Kit.el("path", { d: "M" + sx + "," + f.y + " H" + (bx + bw / 2) + " V" + yEnd + " H" + (sx + 12), fill: "none", stroke: P.muted, "stroke-width": 1.6, opacity: 0.8 }, svg);
        Kit.el("polygon", { points: (sx + 12) + "," + (yEnd - 4) + " " + (sx + 12) + "," + (yEnd + 4) + " " + (sx + 5) + "," + yEnd, fill: P.muted }, svg);
      });
      // the stream marker sits under the shapes, so the labels stay readable
      Kit.el("circle", { cx: sx, cy: yMark, r: 8, fill: P.teal, stroke: P.paper, "stroke-width": 2 }, svg);
      el.forEach(function (e, i) {
        var k = i + 1, state = k < cur ? "done" : k === cur ? "now" : "next", on = state === "now";
        var fillc = on ? P.coral : state === "done" ? P.navy : P.paper, fo = on ? 0.28 : state === "done" ? 0.18 : 1;
        if (e.t === "branch") {
          Kit.el("rect", { x: bx, y: e.y - bh / 2, width: bw, height: bh, rx: 6, fill: fillc, "fill-opacity": fo, stroke: on ? P.coral : P.muted, "stroke-width": on ? 2.4 : 1.2 }, svg);
          svgText(svg, bx + bw / 2, e.y + 4.5, e.lab, { "text-anchor": "middle", "font-size": 13.5, fill: P.ink, "font-weight": on ? 700 : 400 });
        } else if (e.t === "add") {
          Kit.el("circle", { cx: sx, cy: e.y, r: 12, fill: fillc, "fill-opacity": fo, stroke: on ? P.coral : P.muted, "stroke-width": on ? 2.4 : 1.2 }, svg);
          svgText(svg, sx, e.y + 5.5, "+", { "text-anchor": "middle", "font-size": 18, fill: P.ink, "font-weight": 700 });
        } else {
          Kit.el("rect", { x: sx - 34, y: e.y - 13, width: 68, height: 26, rx: 6, fill: fillc, "fill-opacity": fo, stroke: on ? P.coral : P.muted, "stroke-width": on ? 2.4 : 1.2 }, svg);
          svgText(svg, sx, e.y + 4.5, "norm", { "text-anchor": "middle", "font-size": 13.5, fill: P.ink, "font-weight": on ? 700 : 400 });
        }
      });
      if (cur > 0 && el[cur - 1].t === "branch") {
        var e = el[cur - 1]; Kit.el("circle", { cx: bx - 12, cy: e.y, r: 6, fill: P.coral, stroke: P.paper, "stroke-width": 2 }, svg);
      }
      svgText(svg, sx + 14, yIn + 14, "token: " + toks[tok], { "font-size": 12.5, fill: P.ink2 });
      svgText(svg, sx + 14, yOut + 4, "to the next block", { "font-size": 12.5, fill: P.ink2 });
      svgText(svg, 10, 240, "the stream", { "font-size": 12, fill: P.teal, transform: "rotate(-90 10 240)", "text-anchor": "middle", "font-weight": 700 });
    }
    function draw() {
      var P = Kit.palette(), R = blockRun(BL, mode, resid), S = STEPS[mode], rs = R.rows;
      var X0 = BL.X[tok];
      drawDiagram(P, S, step);
      var head = '<div class="vr vh"><span></span><span>the 4 numbers</span><span>length</span></div>', h = head;
      for (var s = 0; s <= step; s++) {
        var st = S[s], v = rs[st.id][tok], label;
        var names = {
          pre: { x: "x", h1: "norm of x", a: "attention output a", x1: resid ? "x₁ = x + a" : "x₁ = a", h2: "norm of x₁", m: "MLP output m", x2: resid ? "x₂ = x₁ + m" : "x₂ = m" },
          post: { x: "x", a: "attention output a", s1: resid ? "s = x + a" : "s = a", y1: "y₁ = norm of s", m: "MLP output m", s2: resid ? "s = y₁ + m" : "s = m", y2: "y₂ = norm of s" }
        }[mode];
        var kind = st.stream ? "stream" : (st.id === "a" || st.id === "m" ? "corr" : "copy");
        h += '<div class="vr ' + kind + (s === step ? " now" : "") + '"><span class="vl">' + names[st.id] + "</span><span class=\"vcs\">" + cells(v) + "</span><span class=\"vn\">" + f2(norm(v)) + "</span></div>";
      }
      rowsEl.innerHTML = h;
      ttl.textContent = "Step " + step + " of 6: " + S[step].name;
      txt.innerHTML = TEXT[mode][step];
      // live equation
      var cur = rs[S[step].id][tok], e = "";
      var id = S[step].id;
      if (id === "h1" || id === "h2" || id === "y1" || id === "y2") {
        var src = mode === "pre" ? (id === "h1" ? rs.x[tok] : rs.x1[tok]) : (id === "y1" ? rs.s1[tok] : rs.s2[tok]);
        var mu = src.reduce(function (a, b) { return a + b; }, 0) / src.length, va = src.reduce(function (a, b) { return a + (b - mu) * (b - mu); }, 0) / src.length;
        e = box(f2(src[0]), "first number") + op("&minus;") + box(f2(mu), "mean of the 4", "c-sand") + op("&divide;") + box(f2(Math.sqrt(va + 1e-5)), "&radic;(variance + 10<sup>&minus;5</sup>), variance " + f2(va), "c-sand") + op("=") + box(f2(cur[0]), "first number, normed", "c-teal");
      } else if (id === "a") {
        var wv = R.w.map(function (W) { return W[tok]; });
        e = wv.map(function (w, hd) { return box(w.slice(0, tok + 1).map(function (x) { return pcs(x); }).join(" / "), "head " + (hd + 1) + " weights on " + toks.slice(0, tok + 1).join(" / "), "c-coral"); }).join(op("&bull;")) + op("&rarr;") + box(f2(norm(cur)), "length of a", "c-coral");
      } else if (id === "m") {
        e = box("4", "numbers in") + op("&rarr;") + box("8", "after W<sub>1</sub> and GELU", "c-sand") + op("&rarr;") + box("4", "after W<sub>2</sub>: the output m", "c-coral");
      } else if (id === "x1" || id === "x2" || id === "s1" || id === "s2") {
        var A = id === "x1" ? [rs.x[tok], rs.a[tok]] : id === "x2" ? [rs.x1[tok], rs.m[tok]] : id === "s1" ? [rs.x[tok], rs.a[tok]] : [rs.y1[tok], rs.m[tok]];
        e = resid ? box(f2(A[0][0]), "first number of the stream", "c-teal") + op("+") + box(f2(A[1][0]), "first number of the correction", "c-coral") + op("=") + box(f2(cur[0]), "first number of the sum", "c-teal") : box(f2(A[1][0]), "the correction replaces the stream", "c-coral") + op("&rarr;") + box(f2(cur[0]), "first number of the result", "c-coral");
      } else e = box(f2(norm(cur)), "length of the stream", "c-teal");
      eq.innerHTML = '<div class="eqn">' + e + "</div>";
      var out = mode === "pre" ? "x2" : "y2", aN = norm(rs.a[tok]), mN = norm(rs.m[tok]);
      read.innerHTML = "<span>Lengths: input <b>" + f2(norm(X0)) + "</b>, attention output <b>" + f2(aN) + "</b>, MLP output <b>" + f2(mN) + "</b>, block output <b>" + f2(norm(rs[out][tok])) + "</b></span>" + (resid ? "<span>The output is the input plus two corrections" + (mode === "post" ? ", normed" : "") + ".</span>" : "<span>With the shortcut cut, the input is not a term of the output.</span>");
      att.textContent = step >= (mode === "pre" ? 2 : 1) ? "Attention weights of " + toks[tok] + ": " + R.w.map(function (W, hd) { return "head " + (hd + 1) + " " + toks.slice(0, tok + 1).map(function (t, j) { return t + " " + pcs(W[tok][j]); }).join(", "); }).join("; ") + "." : "";
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Diagram of one transformer block, step " + step + " of 6: " + S[step].name + ". The shortcut is " + (resid ? "kept" : "cut") + ".");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 2. residual stream in GPT-2 ----------
  function streamFig() {
    var bar = Kit.$("rs-bar"), listEl = Kit.$("rs-list"), read = Kit.$("rs-read"), st = Kit.$("rs-status");
    var ci = 0, k = 9;
    Kit.select(bar, "Text", G2.contexts.map(function (c, i) { return { value: String(i), label: c.key === "capital" ? "Mumbai is the capital of ..." : c.key === "nets" ? "The fishermen at Sassoon Dock cast their ..." : "... We photographed it and put it ..." }; }), "0", function (v) { ci = Number(v); draw(); });
    var sl = Kit.slider(bar, { label: "Blocks read so far", min: 0, max: 12, step: 1, value: k }, function (v) { k = v; draw(); });
    function draw() {
      var C = G2.contexts[ci], L = C.layers, row = L[k];
      var xs = []; for (var i = 1; i <= 12; i++) xs.push(i);
      Kit.plot("rs-norms", function (P) { return [
        { type: "bar", name: "attention output", x: xs, y: L.slice(0, 12).map(function (r) { return r.attn; }), marker: { color: P.coral, opacity: 0.85 }, hovertemplate: "block %{x} attention output: %{y:.0f}<extra></extra>" },
        { type: "bar", name: "MLP output", x: xs, y: L.slice(0, 12).map(function (r) { return r.mlp; }), marker: { color: P.navy, opacity: 0.85 }, hovertemplate: "block %{x} MLP output: %{y:.0f}<extra></extra>" },
        { type: "scatter", mode: "lines+markers", name: "stream after the block", x: xs, y: L.slice(1).map(function (r) { return r.norm; }), line: { color: P.teal, width: 3 }, marker: { size: 6 }, hovertemplate: "after block %{x}: stream length %{y:.0f}<extra></extra>" }]; },
        function (P) { return { barmode: "group", margin: { l: 50, r: 10, t: 24, b: 44 }, showlegend: true, legend: { orientation: "h", y: 1.2, x: 0, font: { size: 12 } }, xaxis: { title: { text: "block number", standoff: 8 }, dtick: 1, automargin: true }, yaxis: { title: { text: "length of the vector" }, automargin: true },
          shapes: k >= 1 ? [{ type: "line", x0: k, x1: k, y0: 0, y1: 1, yref: "paper", line: { color: P.ink, width: 1.5, dash: "dot" } }] : [] }; });
      Kit.plot("rs-p", function (P) { return [
        { type: "scatter", mode: "lines+markers", x: L.map(function (r) { return r.layer; }), y: L.map(function (r) { return r.p_true; }), line: { color: P.coral, width: 2.5 }, marker: { size: 6 }, hovertemplate: "after %{x} blocks: " + "%{y:.3f}<extra></extra>" },
        { type: "scatter", mode: "markers", x: [k], y: [row.p_true], marker: { size: 13, color: "rgba(0,0,0,0)", line: { color: P.ink, width: 2.5 } }, hoverinfo: "skip" }]; },
        { margin: { l: 50, r: 10, t: 14, b: 44 }, xaxis: { title: { text: "blocks read so far", standoff: 8 }, dtick: 1, automargin: true }, yaxis: { title: { text: "p of the true token" }, range: [0, 1.02], automargin: true } });
      var h = "";
      row.top.forEach(function (t) { var isTrue = t[0] === C.true; h += '<div class="rs-row' + (isTrue ? " tr" : "") + '"><span class="rs-tok">' + esc(showTok(t[0])) + '</span><span class="rs-track"><i style="width:' + (100 * t[1]).toFixed(1) + '%"></i></span><b>' + pcs(t[1]) + "</b></div>"; });
      listEl.innerHTML = h;
      var trueTop = row.top.some(function (t) { return t[0] === C.true; });
      read.innerHTML = "<span>After <b>" + k + "</b> blocks the stream has length <b>" + row.norm.toFixed(0) + "</b></span><span>True next token <b>" + esc(showTok(C.true)) + "</b>: <b>" + pcs(row.p_true) + "</b>, rank <b>" + nf(row.rank_true) + "</b> of 50,257" + (trueTop ? " (in the list)" : "") + "</span>";
      Kit.$("rs-norms").setAttribute("aria-label", "Bars for the length of the attention output and of the MLP output of the 12 blocks, and a line for the length of the stream, for the text " + C.text);
      Kit.$("rs-p").setAttribute("aria-label", "The probability of the true next token after 0 to 12 blocks. It is " + pcs(row.p_true) + " after " + k + " blocks and " + pcs(L[12].p_true) + " at the end.");
      st.textContent = "GPT-2 small on the last token of the text, computed offline. The top five guesses use the logit lens: the final layer norm and the output layer applied to the stream after " + k + " blocks. Lengths are Euclidean lengths of 768-number vectors.";
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 3a. position vectors ----------
  function positionFig() {
    var bar = Kit.$("pe-bar"), read = Kit.$("pe-read"), st = Kit.$("pe-status");
    var mode = 0, p = 20, d = 32, NP = 64;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Sinusoidal (2017 paper)", "GPT-2 (learned table)"], 0, function (i) { mode = i; draw(); });
    Kit.slider(bar, { label: "Position p", min: 0, max: NP - 1, step: 1, value: p }, function (v) { p = v; draw(); });
    var dsl = Kit.slider(bar, { label: "Numbers per vector d (sinusoidal)", min: 0, max: 2, step: 1, value: 1, show: function (v) { return String([16, 32, 64][v]); } }, function (v) { d = [16, 32, 64][v]; draw(); });
    function cosv(a, b) { return cosine(a, b); }
    function draw() {
      var xs = [], i, j, S = G2.wpe.cos, simS = [], simG = [];
      for (i = 0; i < NP; i++) xs.push(i);
      var vp = sinPE(p, d, 10000);
      for (j = 0; j < NP; j++) { simS.push(cosv(vp, sinPE(j, d, 10000))); simG.push(S[p][j]); }
      if (mode === 0) {
        var Z = []; for (i = 0; i < NP; i++) Z.push(sinPE(i, d, 10000));
        Kit.plot("pe-map", function (P) { return [{ type: "heatmap", z: Z, x: Z[0].map(function (_, c) { return c; }), y: xs, zmin: -1, zmax: 1, colorscale: [[0, P.coral], [0.5, P.paper], [1, P.teal]], showscale: false, hovertemplate: "position %{y}, number %{x}: %{z:.2f}<extra></extra>" }]; },
          function (P) { return { margin: { l: 50, r: 10, t: 10, b: 44 }, xaxis: { title: { text: "number in the vector (" + d + ")", standoff: 8 }, automargin: true }, yaxis: { title: { text: "position" }, autorange: "reversed", automargin: true }, shapes: [{ type: "rect", x0: -0.5, x1: d - 0.5, y0: p - 0.5, y1: p + 0.5, line: { color: P.ink, width: 2 } }] }; });
      } else {
        Kit.plot("pe-map", function (P) { return [{ type: "heatmap", z: S, x: xs, y: xs, zmin: 0.3, zmax: 1, colorscale: [[0, P.paper], [1, P.teal]], showscale: false, hovertemplate: "positions %{y} and %{x}: cosine %{z:.2f}<extra></extra>" }]; },
          function (P) { return { margin: { l: 50, r: 10, t: 10, b: 44 }, xaxis: { title: { text: "position q", standoff: 8 }, automargin: true }, yaxis: { title: { text: "position p" }, autorange: "reversed", automargin: true }, shapes: [{ type: "rect", x0: -0.5, x1: NP - 0.5, y0: p - 0.5, y1: p + 0.5, line: { color: P.ink, width: 2 } }] }; });
      }
      Kit.plot("pe-sim", function (P) { return [
        { type: "scatter", mode: "lines", name: "sinusoidal, " + d + " numbers", x: xs, y: simS, line: { color: P.teal, width: mode === 0 ? 3 : 1.5 }, opacity: mode === 0 ? 1 : 0.55, hovertemplate: "q = %{x}: %{y:.2f}<extra>sinusoidal</extra>" },
        { type: "scatter", mode: "lines", name: "GPT-2, learned", x: xs, y: simG, line: { color: P.coral, width: mode === 1 ? 3 : 1.5 }, opacity: mode === 1 ? 1 : 0.55, hovertemplate: "q = %{x}: %{y:.2f}<extra>GPT-2</extra>" },
        { type: "scatter", mode: "markers", x: [p], y: [mode === 0 ? simS[p] : simG[p]], marker: { size: 11, color: "rgba(0,0,0,0)", line: { color: P.ink, width: 2.5 } }, hoverinfo: "skip", showlegend: false }]; },
        { margin: { l: 50, r: 10, t: Kit.width(Kit.$("pe-sim")) < 420 ? 56 : 22, b: 44 }, showlegend: true, legend: Kit.width(Kit.$("pe-sim")) < 420 ? { orientation: "h", yref: "container", yanchor: "top", y: 1, x: 0, font: { size: 12 } } : { orientation: "h", y: 1.18, x: 0, font: { size: 12 } }, xaxis: { title: { text: "other position q", standoff: 8 }, automargin: true }, yaxis: { title: { text: "cosine of p and q" }, range: [-0.2, 1.05], automargin: true } });
      var q1 = p + 1 < NP ? p + 1 : p - 1, q5 = p + 5 < NP ? p + 5 : p - 5;   // near the end of the table, compare with the positions before p
      read.innerHTML = "<span>Position <b>" + p + "</b></span><span>Cosine with position " + q1 + ": sinusoidal <b>" + f2(simS[q1]) + "</b>, GPT-2 <b>" + f2(simG[q1]) + "</b></span><span>With position " + q5 + ": <b>" + f2(simS[q5]) + "</b> and <b>" + f2(simG[q5]) + "</b></span>";
      Kit.$("pe-map").setAttribute("aria-label", mode === 0 ? "Heat map of the sinusoidal encoding: 64 positions by " + d + " numbers, in stripes of different wavelengths." : "Heat map of the cosine similarity between the first 64 learned position vectors of GPT-2: bright near the diagonal.");
      Kit.$("pe-sim").setAttribute("aria-label", "Cosine similarity of position " + p + " with the other positions, for the sinusoidal encoding and for GPT-2's learned table. Both fall as positions move apart.");
      st.textContent = "Sinusoidal vectors are computed in your browser. The GPT-2 values are the cosine similarities of the first 64 rows of its position table (1,024 rows of 768 numbers), computed offline. Position 0 behaves differently from the others in GPT-2.";
    }
    draw();
  }

  // ---------- 3b. RoPE ----------
  function ropeFig() {
    var bar = Kit.$("rp-bar"), box3 = Kit.$("rp-planes"), read = Kit.$("rp-read"), st = Kit.$("rp-status"), eqEl = Kit.$("rp-eq");
    var m = 3, n = 8, base = 100;
    var Q = [[1.0, 0.2], [0.8, -0.4], [0.5, 0.6]], K = [[0.9, 0.5], [0.4, 0.8], [-0.4, 0.7]];
    var sm = Kit.slider(bar, { label: "Position m of the query", min: 0, max: 40, step: 1, value: m }, function (v) { m = v; draw(); });
    var sn = Kit.slider(bar, { label: "Position n of the key", min: 0, max: 40, step: 1, value: n }, function (v) { n = v; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["base 10", "base 100", "base 10,000"], 1, function (i) { base = [10, 100, 10000][i]; draw(); });
    Kit.button(bar, "Move both by +5", function () { var s5 = Math.min(5, 40 - Math.max(m, n)); m += s5; n += s5; sm.set(m); sn.set(n); draw(); });   // the same shift for both, so the offset stays
    Kit.button(bar, "Move both by −5", function () { var s5 = Math.min(5, Math.min(m, n)); m -= s5; n -= s5; sm.set(m); sn.set(n); draw(); });
    function theta(i) { return Math.pow(base, -2 * i / 6); }
    function draw() {
      var P = Kit.palette(), S = 210, R = 1.3, sc = (S - 14) / (2 * R), narrow = Kit.width(box3) < 520, fs = narrow ? 24 : 12; clear(box3);   // labels are drawn larger on a phone, where the planes are scaled down
      var total = 0, contrib = [];
      for (var i = 0; i < 3; i++) {
        var qa = ropeRot(Q[i], m * theta(i)), ka = ropeRot(K[i], n * theta(i)), c = qa[0] * ka[0] + qa[1] * ka[1]; total += c; contrib.push(c);
        var wrap = Kit.html("div", { class: "rp-p" }, box3), svg = Kit.el("svg", { role: "img", "aria-label": "Pair " + (i + 1) + ": the query and the key, rotated" }, wrap); svgSize(svg, S, S);
        var X = function (a) { return S / 2 + a * sc; }, Y = function (b) { return S / 2 - b * sc; };
        Kit.el("circle", { cx: S / 2, cy: S / 2, r: R * sc * 0.8, fill: "none", stroke: P.rule, "stroke-width": 1 }, svg);
        Kit.el("line", { x1: 7, y1: S / 2, x2: S - 7, y2: S / 2, stroke: P.rule }, svg); Kit.el("line", { x1: S / 2, y1: 7, x2: S / 2, y2: S - 7, stroke: P.rule }, svg);
        [[Q[i], P.coral, 0.25, "q"], [K[i], P.navy, 0.25, "k"]].forEach(function (g) { Kit.el("line", { x1: S / 2, y1: S / 2, x2: X(g[0][0]), y2: Y(g[0][1]), stroke: g[1], "stroke-width": 1.5, "stroke-dasharray": "3 3", opacity: 0.5 }, svg); });
        function arrow(v, col, lab) { Kit.el("line", { x1: S / 2, y1: S / 2, x2: X(v[0]), y2: Y(v[1]), stroke: col, "stroke-width": 3.2, "stroke-linecap": "round" }, svg); Kit.el("circle", { cx: X(v[0]), cy: Y(v[1]), r: 4.5, fill: col }, svg); svgText(svg, X(v[0]) + (v[0] >= 0 ? 6 : -6), Y(v[1]) + (v[1] >= 0 ? -6 : (narrow ? 22 : 13)), lab, { "font-size": fs, fill: col, "font-weight": 700, "text-anchor": v[0] >= 0 ? "start" : "end" }); }
        arrow(ka, P.navy, "k"); arrow(qa, P.coral, "q");
        Kit.html("div", { class: "rp-t" }, wrap, "pair " + (i + 1) + ": " + (theta(i)).toFixed(3) + " rad per position");
        Kit.html("div", { class: "rp-c" }, wrap, "q·k = " + f2(c));
      }
      eqEl.innerHTML = '<div class="eqn">' + contrib.map(function (c, i) { return box(f2(c), "pair " + (i + 1), "c-" + ["coral", "navy", "teal"][i]); }).join(op("+")) + op("=") + box(f2(total), "q·k at offset n − m = " + sg(n - m), "c-sand") + "</div>";
      var xs = [], ys = []; for (var dlt = -40; dlt <= 40; dlt++) { xs.push(dlt); var t = 0; for (var j = 0; j < 3; j++) { var kr = ropeRot(K[j], dlt * theta(j)); t += Q[j][0] * kr[0] + Q[j][1] * kr[1]; } ys.push(t); }
      Kit.plot("rp-plot", function (P) { return [{ type: "scatter", mode: "lines", x: xs, y: ys, line: { color: P.sand, width: 2.5 }, hovertemplate: "offset %{x}: %{y:.2f}<extra></extra>" }, { type: "scatter", mode: "markers", x: [n - m], y: [total], marker: { size: 12, color: P.coral, line: { color: P.ink, width: 2 } }, hoverinfo: "skip" }]; },
        { margin: { l: 50, r: 10, t: 12, b: 44 }, xaxis: { title: { text: "offset n − m", standoff: 8 }, automargin: true }, yaxis: { title: { text: "q·k" }, automargin: true } });
      Kit.$("rp-plot").setAttribute("aria-label", "The dot product of the rotated query and key against the offset n minus m. The current point is at offset " + sg(n - m) + " with value " + f2(total) + ".");
      read.innerHTML = "<span>m = <b>" + m + "</b>, n = <b>" + n + "</b>, offset <b>" + sg(n - m) + "</b></span><span>q·k = <b>" + f2(total) + "</b></span>";
      st.textContent = "Three pairs of numbers (a vector of 6 numbers). The unrotated q and k are drawn dotted. The vectors are made up. Base 10 and 100 make the slow pairs visible; actual models use larger bases, 10,000 in the RoFormer paper.";
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 4. decoder-only against encoder-only ----------
  function familyFig() {
    var bar = Kit.$("fm-bar"), wordsEl = Kit.$("fm-words"), svg = Kit.$("fm-mask"), guess = Kit.$("fm-guess"), eq = Kit.$("fm-eq"), read = Kit.$("fm-read"), st = Kit.$("fm-status");
    var mode = 1, sel = 5, R = MD.rows, W = MD.words, n = W.length;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Decoder-only (GPT-2)", "Encoder-only (BERT)"], mode, function (i) { mode = i; draw(); });
    function draw() {
      var P = Kit.palette(), row = R[sel], g = mode === 0;
      var h = "";
      W.forEach(function (w, i) {
        var cls = "wd";
        if (i === sel) cls += " tgt"; else if (g && i > sel) cls += " hid";
        var txt = i === sel ? (g ? "?" : "[MASK]") : w;
        h += '<button type="button" class="' + cls + '" data-i="' + i + '"' + (i === 0 ? ' disabled title="no left context to predict from"' : "") + ' aria-pressed="' + (i === sel) + '" aria-label="' + (i === 0 ? "first word" : "Hide the word " + w) + '">' + esc(txt) + "</button>";
      });
      wordsEl.innerHTML = h;
      Array.prototype.forEach.call(wordsEl.querySelectorAll("button"), function (b) { b.addEventListener("click", function () { var i = Number(b.getAttribute("data-i")); if (i >= 1) { sel = i; draw(); } }); });
      // mask grid
      var cs = 22, off = 4; svgSize(svg, n * cs + 2 * off, n * cs + 2 * off); clear(svg);
      var rowUsed = g ? sel - 1 : sel;
      for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
        var vis = g ? j <= i : true;
        Kit.el("rect", { x: off + j * cs, y: off + i * cs, width: cs - 2, height: cs - 2, rx: 3, fill: vis ? P.teal : "none", "fill-opacity": vis ? (i === rowUsed ? 0.85 : 0.3) : 0, stroke: vis ? "none" : P.rule, "stroke-width": 1 }, svg);
      }
      Kit.el("rect", { x: off - 2, y: off + rowUsed * cs - 2, width: n * cs + 2, height: cs + 2, rx: 4, fill: "none", stroke: P.coral, "stroke-width": 2.2 }, svg);
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", (g ? "Triangular mask: a token reads itself and earlier tokens. " : "Full table: all tokens read all tokens. ") + "The framed row is the one that makes the guess for the hidden word.");
      // guesses
      var gs = g ? row.gpt : row.bert, truth = W[sel];
      var gh = "";
      gs.top.forEach(function (t) { var hit = t[0].toLowerCase() === truth.toLowerCase(); gh += '<div class="rs-row' + (hit ? " tr" : "") + '"><span class="rs-tok">' + esc(t[0]) + '</span><span class="rs-track"><i style="width:' + (100 * t[1]).toFixed(1) + '%"></i></span><b>' + pcs(t[1]) + "</b></div>"; });
      guess.innerHTML = gh;
      var loss = -Math.log(Math.max(gs.p_true, 1e-9));
      eq.innerHTML = '<div class="eqn">' + box(esc(truth), "the hidden word", "c-coral") + op("&rarr;") + box(pcs(gs.p_true), "probability the model gave it", "c-teal") + op("&rarr;") + box(loss.toFixed(2), "loss &minus;ln p, nats", "c-sand") + op("&bull;") + box("rank " + nf(gs.rank_true), "of the vocabulary", "c-navy") + "</div>";
      var seen = g ? W.slice(0, sel).join(" ") : W.map(function (w, i) { return i === sel ? "[MASK]" : w; }).join(" ");
      read.innerHTML = "<span>The model sees: <b>" + esc(seen) + "</b></span><span>It must guess: <b>" + esc(truth) + "</b></span>";
      st.textContent = g ? "Decoder-only: a token reads itself and the tokens before it. Training asks for the next word at all positions of a text, so all positions are training examples." : "Encoder-only: all tokens read all tokens. BERT is trained by hiding about 15% of the tokens and asking for them, so only the hidden positions give a loss. BERT here is the uncased base model, so the words are lower case.";
      st.textContent += " The predictions are from GPT-2 small and BERT base, computed offline.";
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 5. generation loop and key-value cache ----------
  function cacheFig() {
    var bar = Kit.$("gn-bar"), tokEl = Kit.$("gn-toks"), svg = Kit.$("gn-grid"), read = Kit.$("gn-read"), eq = Kit.$("gn-eq"), st = Kit.$("gn-status");
    var GN = G2.generate, all = GN.prompt_tokens.concat(GN.new_tokens), np = GN.prompt_tokens.length, N = all.length, step = 0, cached = true, timer = 0, bytes = 2;
    var NL = G2.config.n_layer, D = G2.config.n_embd;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["With the cache", "Without the cache"], 0, function (i) { cached = i === 0; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["32-bit numbers", "16-bit numbers"], 1, function (i) { bytes = i === 0 ? 4 : 2; draw(); });
    Kit.button(bar, "Next token", function () { stop(); if (step < N - np) step++; draw(); });
    var bp = Kit.button(bar, "Play", function () { if (timer) { stop(); return; } if (step >= N - np) step = 0; bp.textContent = "Pause"; draw(); timer = setInterval(function () { if (step >= N - np) { stop(); return; } step++; draw(); }, 1000); });
    Kit.button(bar, "Reset", function () { stop(); step = 0; draw(); });
    function stop() { if (timer) clearInterval(timer); timer = 0; bp.textContent = "Play"; }
    function work(upTo, useCache) { // token-passes up to step s: prefill np, then one new token per step
      var t = np; for (var s = 1; s <= upTo; s++) t += useCache ? 1 : np + s; return t;
    }
    function draw() {
      var P = Kit.palette(), len = np + step; // tokens in the context now
      var h = "";
      for (var i = 0; i < N; i++) {
        var cls = "tk " + (i < np ? "c0" : "c1"); if (i >= len) cls += " gone"; if (i === len - 1 && step > 0) cls += " newest";
        h += '<span class="' + cls + '">' + esc(showTok(all[i])) + "</span>";
      }
      tokEl.innerHTML = h;
      var cs = Math.min(26, Math.floor((Math.max(300, Kit.width(svg.parentNode) - 36) - 34) / N)), rh = 12, left = 34;
      svgSize(svg, left + cs * N + 6, 16 + NL * rh + 6); clear(svg);
      svgText(svg, 2, 11, "layer", { "font-size": 10.5, fill: P.muted });
      for (var l = 0; l < NL; l++) {
        if (l % 3 === 0 || l === NL - 1) svgText(svg, left - 6, 16 + l * rh + 9, String(l + 1), { "font-size": 10, fill: P.muted, "text-anchor": "end" });
        for (var j = 0; j < N; j++) {
          var inCache = j < len, fresh = cached ? (step === 0 ? j < np : j === len - 1) : inCache, fillc = !inCache ? "none" : fresh ? P.coral : P.navy, fo = !inCache ? 0 : fresh ? 0.85 : 0.35;
          Kit.el("rect", { x: left + j * cs, y: 16 + l * rh, width: cs - 2, height: rh - 2, rx: 2, fill: fillc, "fill-opacity": fo, stroke: inCache ? "none" : P.rule, "stroke-width": 0.6 }, svg);
        }
      }
      for (j = 0; j < N; j += 1) if (j % 2 === 0 || cs >= 20) svgText(svg, left + j * cs + (cs - 2) / 2, 11, String(j + 1), { "font-size": 9.5, fill: P.muted, "text-anchor": "middle" });
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Grid of the keys and values of " + len + " tokens in 12 layers. " + (cached ? "Only the newest token is computed in this step; the others are reused from the cache." : "All tokens are computed again in this step."));
      var per = kvNumbers(NL, D, 1), nowNum = per * len;
      var thisStep = cached ? (step === 0 ? np : 1) : len;
      read.innerHTML = "<span>Tokens in the context: <b>" + len + "</b></span><span>Computed in this step: <b>" + thisStep + "</b> token" + (thisStep === 1 ? "" : "s") + " (all 12 layers)</span><span>Total so far: <b>" + work(step, true) + "</b> with the cache, <b>" + work(step, false) + "</b> without</span>";
      eq.innerHTML = '<div class="eqn">' + box("2", "K and V", "c-sand") + op("&times;") + box(NL, "layers", "c-navy") + op("&times;") + box(len, "tokens now", "c-coral") + op("&times;") + box(D, "numbers per key or value", "c-teal") + op("=") + box(nf(nowNum), "numbers in the cache", "c-sand") + op("&times;") + box(bytes, "bytes per number", "c-navy") + op("=") + box(nf(nowNum * bytes), "bytes (" + (nowNum * bytes / 1e3).toFixed(0) + " KB)", "c-coral") + "</div>";
      st.textContent = step === 0 ? "Prefill: the prompt of " + np + " tokens is read in one pass, and the cache is filled for all of them." : "Decoding step " + step + ": the model chose " + showTok(all[len - 1]) + " (greedy, GPT-2 small, computed offline). " + (cached ? "It computes the key and value of that one token and attends over the cache." : "Without a cache it reads all " + len + " tokens again.");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
    // calculator
    var cb = Kit.$("kc-bar"), ceq = Kit.$("kc-eq"), cread = Kit.$("kc-read"), SZ = G2.kv.sizes, mi = 0, nn = 1024, by = 2;
    Kit.chips(Kit.html("div", { class: "row-chips" }, cb), SZ.map(function (s) { return s.name; }), 0, function (i) { mi = i; calc(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, cb), ["32-bit", "16-bit"], 1, function (i) { by = i === 0 ? 4 : 2; calc(); });
    Kit.slider(cb, { label: "Tokens in the context n", min: 1, max: 1024, step: 1, value: nn }, function (v) { nn = v; calc(); });
    function mb(x) { return x >= 1e9 ? (x / 1e9).toFixed(2) + " GB" : (x / 1e6).toFixed(1) + " MB"; }
    function calc() {
      var s = SZ[mi], num = kvNumbers(s.layers, s.d, nn), perTok = kvNumbers(s.layers, s.d, 1);
      ceq.innerHTML = '<div class="eqn">' + box("2", "K and V", "c-sand") + op("&times;") + box(s.layers, "layers", "c-navy") + op("&times;") + box(nf(nn), "tokens n", "c-coral") + op("&times;") + box(nf(s.d), "width d", "c-teal") + op("=") + box(nf(num), "numbers", "c-sand") + op("&times;") + box(by, "bytes per number", "c-navy") + op("=") + box(mb(num * by), nf(num * by) + " bytes", "c-coral") + "</div>";
      cread.innerHTML = "<span>" + s.name + " (" + s.params + " weights): <b>" + nf(perTok) + "</b> numbers per token, <b>" + nf(perTok * by) + "</b> bytes per token</span><span>For the full 1,024-token window: <b>" + mb(kvNumbers(s.layers, s.d, 1024) * by) + "</b></span>";
    }
    calc();
  }
})();
