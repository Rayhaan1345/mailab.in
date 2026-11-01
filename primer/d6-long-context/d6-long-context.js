// D6 Long context. Figures: compute and memory against length, the KV cache at real model sizes, a gallery of attention masks with live pair counts,
// how information travels through stacked windows, GPT-2 with its attention cut to a window, fixed-size memories (sum, decay, delta rule, window, full cache),
// test-time training read chunk by chunk, and a step-through of four routes. The pure logic is exported with module.exports for node tests.
(function () {
  "use strict";
  // ---------- pure logic ----------
  function nf(n) { return Number(n).toLocaleString("en-US"); }
  function human(b) {
    if (b >= 1e9) return (b / 1e9).toFixed(b >= 1e11 ? 0 : 1) + " GB";
    if (b >= 1e6) return (b / 1e6).toFixed(b >= 1e8 ? 0 : 1) + " MB";
    if (b >= 1e3) return (b / 1e3).toFixed(b >= 1e5 ? 0 : 1) + " KB";
    return Math.round(b) + " B";
  }
  function big(x) { // multiply-adds with a unit
    if (x >= 1e15) return (x / 1e15).toFixed(1) + " quadrillion";
    if (x >= 1e12) return (x / 1e12).toFixed(1) + " trillion";
    if (x >= 1e9) return (x / 1e9).toFixed(1) + " billion";
    if (x >= 1e6) return (x / 1e6).toFixed(1) + " million";
    return nf(Math.round(x));
  }
  // Shapes of the models used in the figures. pb = weights per block that act on one token at a time (projections and MLP), da = width of the attention
  // (query heads times head size), full = layers that keep a key and value for every token, nKV and dh as in the configuration files.
  var MODELS = {
    gpt2: { name: "GPT-2 small", L: 12, d: 768, da: 768, pb: 7087872, full: 12, nQ: 12, nKV: 12, dh: 64, params: 124439808, maxPos: 1024, linear: 0, state: 0 },
    q05: { name: "Qwen2.5-0.5B", L: 24, d: 896, da: 896, pb: 14909440, full: 24, nQ: 14, nKV: 2, dh: 64, params: 494032768, maxPos: 32768, linear: 0, state: 0 },
    q4: { name: "Qwen3-4B", L: 36, d: 2560, da: 4096, pb: 100925440, full: 36, nQ: 32, nKV: 8, dh: 128, params: 4.02e9, maxPos: 40960, linear: 0, state: 0 },
    q27: { name: "Qwen3.6-27B", L: 64, d: 5120, da: 6144, pb: 0, full: 16, nQ: 24, nKV: 4, dh: 256, params: 27e9, maxPos: 262144, linear: 48, state: 48 * 48 * 128 * 128 }
  };
  function attnMulAdds(n, da, causal) { return causal ? n * (n + 1) * da : 2 * n * n * da; }
  function crossover(pb, da) { return pb / da - 1; }
  function kvPerToken(m, bytes, mha) { return 2 * m.full * (mha ? m.nQ : m.nKV) * m.dh * bytes; }
  function kvTotal(m, n, bytes, mha) { return kvPerToken(m, bytes, mha) * n + (m.state ? m.state * bytes : 0); }
  function rng(seed) { var a = seed >>> 0; return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; var t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  function gauss(r) { var u = 0, v = 0; while (!u) u = r(); while (!v) v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

  // ----- masks. P = {n, w, l, g, r, causal}. Rows are queries i, columns are keys j. -----
  var PATTERNS = ["full", "window", "strided", "global", "bigbird"];
  function winRange(i, P) {
    if (P.causal) return [Math.max(0, i - P.w + 1), i];
    var h = Math.floor(P.w / 2); return [Math.max(0, i - h), Math.min(P.n - 1, i + h)];
  }
  function rowBase(pat, i, P) { // count of keys in the non-random part of row i
    var n = P.n, g = Math.min(P.g, n), r = winRange(i, P), win = r[1] - r[0] + 1;
    if (pat === "full") return P.causal ? i + 1 : n;
    if (pat === "window") return win;
    if (pat === "strided") return Math.min(i, P.l) + 1 + Math.max(0, Math.floor(i / P.l) - 1);
    // global (and the base of bigbird)
    if (P.causal) return win + Math.min(g, r[0]);
    if (i < g) return n;
    return win + g - Math.max(0, Math.min(g - 1, r[1]) - r[0] + 1);
  }
  function rowCount(pat, i, P) {
    var base = rowBase(pat, i, P);
    if (pat !== "bigbird") return base;
    var avail = P.causal ? i + 1 : P.n;
    return base + Math.max(0, Math.min(P.r, avail - base));
  }
  function pairsTotal(pat, n, P) {
    var Q = { n: n, w: P.w, l: P.l, g: P.g, r: P.r, causal: P.causal }, s = 0;
    for (var i = 0; i < n; i++) s += rowCount(pat, i, Q);
    return s;
  }
  // Classes of the keys in row i: "win", "glob", "str", "rnd", "full". Returns an object j -> class.
  function rowCells(pat, i, P) {
    var n = P.n, g = Math.min(P.g, n), out = {}, j, r;
    if (pat === "full") { for (j = 0; j <= (P.causal ? i : n - 1); j++) out[j] = "full"; return out; }
    if (pat === "strided") {
      for (j = Math.max(0, i - P.l); j <= i; j++) out[j] = "win";
      for (j = i; j >= 0; j -= P.l) if (out[j] == null) out[j] = "str";
      return out;
    }
    r = winRange(i, P);
    for (j = r[0]; j <= r[1]; j++) out[j] = "win";
    if (pat === "global" || pat === "bigbird") {
      if (!P.causal && i < g) { for (j = 0; j < n; j++) out[j] = "glob"; } else { for (j = 0; j < g; j++) if ((P.causal ? j <= i : true)) out[j] = "glob"; }
    }
    if (pat === "bigbird" && P.r > 0) {
      var avail = [], hi = P.causal ? i : n - 1;
      for (j = 0; j <= hi; j++) if (out[j] == null) avail.push(j);
      var rd = rng(7919 * (i + 1) + 13);
      for (var k = avail.length - 1; k > 0; k--) { var m = Math.floor(rd() * (k + 1)), t = avail[k]; avail[k] = avail[m]; avail[m] = t; }
      for (k = 0; k < Math.min(P.r, avail.length); k++) out[avail[k]] = "rnd";
    }
    return out;
  }
  function neighbours(pat, i, P) { return Object.keys(rowCells(pat, i, P)).map(Number); }
  // Which input tokens can influence token `top` after L layers? Returns sets[0..L], sets[L] = [top].
  function reachSets(pat, P, top, L) {
    var sets = [], cur = [top];
    sets[L] = cur;
    for (var l = L - 1; l >= 0; l--) {
      var seen = {}; cur.forEach(function (i) { neighbours(pat, i, P).forEach(function (j) { seen[j] = 1; }); });
      cur = Object.keys(seen).map(Number).sort(function (a, b) { return a - b; }); sets[l] = cur;
    }
    return sets;
  }

  // ----- fixed-size memories -----
  function unit(v) { var s = Math.sqrt(v.reduce(function (a, b) { return a + b * b; }, 0)); return v.map(function (x) { return x / s; }); }
  // The first d keys are orthonormal (a model whose keys are well separated); later keys are random unit vectors, because at most d vectors can be orthogonal.
  function memItems(n, d, seed, orth) {
    var r = rng(seed), K = [], V = [];
    for (var i = 0; i < n; i++) {
      var k = [], v = [], c;
      for (c = 0; c < d; c++) { k.push(gauss(r)); v.push(gauss(r)); }
      if (orth && i < d) K.forEach(function (p) { var dot = 0, q; for (q = 0; q < d; q++) dot += k[q] * p[q]; for (q = 0; q < d; q++) k[q] -= dot * p[q]; });
      K.push(unit(k)); V.push(unit(v));
    }
    return { K: K, V: V };
  }
  function zeros(d) { var S = []; for (var i = 0; i < d; i++) { S.push(new Array(d).fill(0)); } return S; }
  function matvec(S, k) { return S.map(function (row) { var s = 0; for (var c = 0; c < k.length; c++) s += row[c] * k[c]; return s; }); }
  function dist(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += (a[i] - b[i]) * (a[i] - b[i]); return Math.sqrt(s); }
  function softmaxRead(K, V, lo, hi, q, tau) {
    var m = -1e9, i, sc = [];
    for (i = lo; i <= hi; i++) { var s = 0; for (var c = 0; c < q.length; c++) s += q[c] * K[i][c]; sc.push(tau * s); if (tau * s > m) m = tau * s; }
    var e = sc.map(function (x) { return Math.exp(x - m); }), Z = e.reduce(function (a, b) { return a + b; }, 0), out = new Array(V[0].length).fill(0);
    for (i = lo; i <= hi; i++) for (c = 0; c < out.length; c++) out[c] += e[i - lo] / Z * V[i][c];
    return out;
  }
  // errors[i] = distance between the value read back with key i and the stored value i (all values have length 1).
  function memErrors(type, it, n, o) {
    var d = it.K[0].length, S = zeros(d), errs = [], i, c, c2;
    o = o || {};
    if (type === "full" || type === "window") {
      var lo = type === "window" ? Math.max(0, n - (o.w || 4)) : 0;
      for (i = 0; i < n; i++) errs.push(dist(softmaxRead(it.K, it.V, lo, n - 1, it.K[i], o.tau || 12), it.V[i]));
      return errs;
    }
    for (i = 0; i < n; i++) {
      var k = it.K[i], v = it.V[i];
      if (type === "decay") { for (c = 0; c < d; c++) for (c2 = 0; c2 < d; c2++) S[c][c2] = (o.gamma == null ? 0.85 : o.gamma) * S[c][c2] + v[c] * k[c2]; }
      else if (type === "delta") { var pr = matvec(S, k); for (c = 0; c < d; c++) for (c2 = 0; c2 < d; c2++) S[c][c2] += (o.eta == null ? 1 : o.eta) * (v[c] - pr[c]) * k[c2]; }
      else { for (c = 0; c < d; c++) for (c2 = 0; c2 < d; c2++) S[c][c2] += v[c] * k[c2]; }
    }
    for (i = 0; i < n; i++) errs.push(dist(matvec(S, it.K[i]), it.V[i]));
    return errs;
  }
  // Test-time training on chunks: S starts at zero. Loss "sq": 1/2 sum ||S k - v||^2 (TTT layers); loss "dot": - sum v . (S k) (In-Place TTT).
  // Returns the matrices after each chunk and the loss on the chunk before and after the step.
  function ttRun(it, n, o) {
    var d = it.K[0].length, S = zeros(d), mats = [zeros(d)], loss = [], c = o.c, eta = o.eta, i, a, b;
    function chunkLoss(M, lo, hi) {
      var s = 0;
      for (var t = lo; t < hi; t++) {
        var pr = matvec(M, it.K[t]);
        if (o.loss === "dot") { for (var q = 0; q < d; q++) s -= it.V[t][q] * pr[q]; }
        else { s += 0.5 * Math.pow(dist(pr, it.V[t]), 2); }
      }
      return s;
    }
    for (var lo = 0; lo + c <= n; lo += c) {
      var before = chunkLoss(S, lo, lo + c);
      for (var rep = 0; rep < (o.reps || 1); rep++) {
        var G = zeros(d);
        for (i = lo; i < lo + c; i++) {
          var pr = matvec(S, it.K[i]);
          for (a = 0; a < d; a++) for (b = 0; b < d; b++) G[a][b] += (o.loss === "dot" ? it.V[i][a] : it.V[i][a] - pr[a]) * it.K[i][b];
        }
        for (a = 0; a < d; a++) for (b = 0; b < d; b++) S[a][b] += eta * G[a][b];
      }
      loss.push([before, chunkLoss(S, lo, lo + c)]);
      mats.push(S.map(function (r) { return r.slice(); }));
    }
    return { mats: mats, loss: loss };
  }
  function readErrors(S, it, n) { var e = []; for (var i = 0; i < n; i++) e.push(dist(matvec(S, it.K[i]), it.V[i])); return e; }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { MODELS: MODELS, attnMulAdds: attnMulAdds, crossover: crossover, kvPerToken: kvPerToken, kvTotal: kvTotal, rowCount: rowCount, pairsTotal: pairsTotal, rowCells: rowCells,
      reachSets: reachSets, neighbours: neighbours, memItems: memItems, memErrors: memErrors, ttRun: ttRun, readErrors: readErrors, human: human, big: big, nf: nf };
  }
  if (typeof document === "undefined") return;

  // ---------- helpers ----------
  var TOK = null, WIN = null;
  function box(val, cap, cls) { return '<span class="t ' + (cls || "") + '"><b>' + val + "</b><i>" + cap + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function svgSize(svg, w, h) { svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.setAttribute("width", w); svg.setAttribute("height", h); svg.style.maxWidth = w + "px"; }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function svgText(parent, x, y, txt, attrs) { var t = Kit.el("text", Object.assign({ x: x, y: y }, attrs || {}), parent); t.textContent = txt; return t; }
  function rowChips(bar, labels, init, fn) { return Kit.chips(Kit.html("div", { class: "row-chips" }, bar), labels, init, fn); }
  function log10(x) { return Math.log(x) / Math.LN10; }
  // A slider on a log scale whose start value and button values are exact (the slider itself moves in steps of 0.01 in the exponent).
  function logSlider(ctl, label, lo, hi, n0, fn) {
    var forced = n0, s = Kit.slider(ctl, { label: label, min: lo, max: hi, step: 0.01, value: log10(n0), show: function (v) { return nf(forced != null && Math.abs(log10(forced) - v) < 0.006 ? forced : Math.round(Math.pow(10, v))); } }, function () { forced = null; fn(); });
    return { input: s.input, n: function () { return forced != null ? forced : Math.round(Math.pow(10, s.get())); }, setN: function (n) { forced = n; s.set(log10(n)); } };
  }

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/tokens.json"), Kit.data("data/gpt2win.json")]).then(function (r) {
      TOK = r[0]; WIN = r[1];
      costFig(); cacheFig(); maskFig(); reachFig(); cutFig(); memFig(); tttFig(); routesFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. cost against length ----------
  function costFig() {
    var mkeys = ["gpt2", "q4"], mi = 0, causal = true, bar = Kit.$("cs-bar"), ctl = Kit.$("cs-controls");
    var presets = [["a report", 33], ["GPT-2 window", 1024], ["500 reports", Math.round(500 * TOK.reports_mean_tokens.English.gpt2)], ["dive log as text", TOK.formats.sentence.gpt2], ["dive as photos", TOK.image_tokens_dive]];
    var sl = logSlider(ctl, "Length n, in tokens", 1.5, 6, 1024, draw);
    rowChips(bar, ["GPT-2 small sizes", "Qwen3-4B sizes"], 0, function (i) { mi = i; draw(); });
    rowChips(bar, ["Allowed pairs only (causal)", "All n by n pairs"], 0, function (i) { causal = i === 0; draw(); });
    var pb = Kit.html("div", { class: "row-chips" }, bar);
    presets.forEach(function (p) { Kit.button(pb, p[0], function () { sl.setN(p[1]); draw(); }); });
    var xs = [], i; for (i = 0; i <= 90; i++) xs.push(Math.pow(10, 1.5 + (4.5 * i) / 90));
    function draw() {
      var m = MODELS[mkeys[mi]], n = sl.n(), rest = n * m.pb, att = attnMulAdds(n, m.da, causal), nstar = causal ? crossover(m.pb, m.da) : m.pb / (2 * m.da);
      Kit.plot("cs-plot", function (P) {
        return [
          { x: xs, y: xs.map(function (x) { return x * m.pb; }), mode: "lines", name: "the rest of the layer", line: { color: P.navy, width: 3 }, hovertemplate: "n = %{x:,.0f}<br>%{y:.3s} multiply-adds<extra>the rest of the layer</extra>" },
          { x: xs, y: xs.map(function (x) { return attnMulAdds(x, m.da, causal); }), mode: "lines", name: "attention pairs", line: { color: P.coral, width: 3 }, hovertemplate: "n = %{x:,.0f}<br>%{y:.3s} multiply-adds<extra>attention pairs</extra>" },
          { x: [n, n], y: [rest, att], mode: "markers", marker: { color: [P.navy, P.coral], size: 11, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" },
          { x: [nstar], y: [nstar * m.pb], mode: "markers+text", text: ["equal at about " + nf(Math.round(nstar))], textposition: "top left", textfont: { color: P.ink2, size: 12.5 }, marker: { symbol: "diamond", color: P.sand, size: 11, line: { color: P.ink2, width: 1 } }, hoverinfo: "skip" }
        ];
      }, function (P) {
        return { xaxis: { type: "log", title: { text: "length n, in tokens" }, range: [1.5, 6] }, yaxis: { type: "log", title: { text: "multiply-adds in one layer" }, range: [7, 16] }, margin: { l: 64, r: 12, t: 12, b: 46 },
          annotations: [{ x: log10(5e5), y: log10(attnMulAdds(5e5, m.da, causal)), text: "attention pairs", showarrow: false, xanchor: "right", yanchor: "bottom", font: { color: P.coral, size: 13.5 } },
            { x: log10(40), y: log10(40 * m.pb) + 0.12, text: "the rest of the layer", showarrow: false, xanchor: "left", yanchor: "bottom", font: { color: P.navy, size: 13.5 } }] };
      });
      var pairs = causal ? n * (n + 1) / 2 : n * n, share = att / (att + rest);
      Kit.$("cs-read").innerHTML = "<span>Length: <b>" + nf(n) + "</b> tokens</span><span>Pairs per head: <b>" + nf(pairs) + "</b></span><span>Attention share of the layer's work: <b>" + (100 * share < 0.1 ? "under 0.1" : (100 * share).toFixed(share > 0.995 ? 2 : 1)) + "%</b></span><span>Double the length: attention <b>&times;4</b>, the rest <b>&times;2</b></span>";
      Kit.$("cs-eq").innerHTML = '<div class="eqn">' + box(nf(n), "tokens n", "c-sand") + op("&times;") + box(nf(m.pb), "weights in a block that act on one token", "c-navy") + op("=") + box(big(rest), "multiply-adds, the rest of the layer", "c-navy") + '</div><div class="eqn">' +
        box(nf(n), "n", "c-sand") + op("&times;") + box(nf(causal ? n + 1 : 2 * n), causal ? "n + 1" : "2n", "c-sand") + op("&times;") + box(nf(m.da), "attention width", "c-teal") + op("=") + box(big(att), "multiply-adds, attention pairs", "c-coral") + "</div>";
      Kit.$("cs-status").textContent = share > 0.5 ? "At this length attention is the larger part of a layer." : "At this length the weights that work on one token at a time are the larger part. Attention takes over at about " + nf(Math.round(nstar)) + " tokens.";
      Kit.$("cs-plot").setAttribute("aria-label", "Log-log chart of multiply-adds in one layer against length. The line for attention pairs rises twice as steeply as the line for the rest of the layer and crosses it at about " + nf(Math.round(nstar)) + " tokens for the " + m.name + " sizes.");
    }
    draw(); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 2. the KV cache at real sizes ----------
  function cacheFig() {
    var keys = ["gpt2", "q05", "q4", "q27"], mi = 2, bytes = 2, bar = Kit.$("kv-bar"), ctl = Kit.$("kv-controls");
    var sl = logSlider(ctl, "Length n, in tokens", 2, 6, 32768, draw);
    var bg = Kit.slider(ctl, { label: "Memory for the model and its cache", min: 4, max: 80, step: 1, value: 16, show: function (v) { return v + " GB"; } }, draw);
    rowChips(bar, keys.map(function (k) { return MODELS[k].name; }), mi, function (i) { mi = i; draw(); });
    rowChips(bar, ["16-bit numbers", "8-bit numbers"], 0, function (i) { bytes = i === 0 ? 2 : 1; draw(); });
    function draw() {
      var m = MODELS[keys[mi]], n = sl.n(), budget = bg.get() * 1e9, wbytes = m.params * 2;
      var cache = kvTotal(m, n, bytes, false), mha = kvTotal(m, n, bytes, true), per = kvPerToken(m, bytes, false);
      var fit = Math.max(0, Math.floor((budget - wbytes) / per));
      Kit.plot("kv-plot", function (P) {
        return [{ type: "bar", orientation: "h", y: ["cache, no sharing", "cache as built", "weights, 16-bit"], x: [mha / 1e9, cache / 1e9, wbytes / 1e9],
          marker: { color: [P.sand, P.coral, P.navy] }, text: [human(mha), human(cache), human(wbytes)], textposition: "outside", cliponaxis: false, hovertemplate: "%{y}: %{x:.3f} GB<extra></extra>" }];
      }, function (P) {
        var top = Math.max(mha / 1e9, wbytes / 1e9, bg.get()) * 1.6;
        return { xaxis: { title: { text: "memory, GB" }, range: [0, top] }, yaxis: { automargin: true, tickfont: { size: 12.5 } }, margin: { l: 110, r: 10, t: 10, b: 46 },
          shapes: [{ type: "line", xref: "x", yref: "paper", x0: bg.get(), x1: bg.get(), y0: 0, y1: 1, line: { color: P.teal, width: 2.5, dash: "dot" } }] };
      });
      var beyond = n > m.maxPos ? " Beyond the " + nf(m.maxPos) + " tokens of its configuration file the model needs position scaling (D5)." : "";
      Kit.$("kv-read").innerHTML = "<span>Per token: <b>" + nf(per) + "</b> bytes</span><span>Cache now: <b>" + human(cache) + "</b></span><span>Weights are <b>" + (wbytes / Math.max(cache, 1)).toFixed(wbytes > cache ? 1 : 2) + "</b> times the cache</span><span>Tokens that fit after the weights: <b>" + (wbytes >= budget ? "none" : nf(fit)) + "</b></span>";
      var eqs = box("2", "K and V", "c-sand") + op("&times;") + box(m.full, "layers with a full cache", "c-navy") + op("&times;") + box(m.nKV, "key-value heads", "c-teal") + op("&times;") + box(m.dh, "numbers per head", "c-teal") + op("&times;") + box(bytes, "bytes per number", "c-sand") + op("=") + box(nf(per), "bytes per token", "c-coral");
      Kit.$("kv-eq").innerHTML = '<div class="eqn">' + eqs + '</div><div class="eqn">' + box(nf(per), "bytes per token", "c-coral") + op("&times;") + box(nf(n), "tokens n", "c-sand") + op("=") + box(human(per * n), "bytes of cache", "c-coral") + (m.state ? op("+") + box(human(m.state * bytes), "fixed state of " + m.linear + " linear layers (estimate)", "c-sand") : "") + "</div>";
      Kit.$("kv-status").textContent = "The dotted line is the memory you chose. " + (m.linear ? "Only " + m.full + " of " + m.L + " layers keep a key and value for all tokens; the other " + m.linear + " keep a state of fixed size. " : "") + "Sharing keys and values between query heads shrinks the cache by " + (m.nQ / m.nKV).toFixed(m.nQ % m.nKV ? 1 : 0) + " times here." + beyond;
      Kit.$("kv-plot").setAttribute("aria-label", "Bars for " + m.name + " at " + nf(n) + " tokens: weights " + human(wbytes) + ", KV cache " + human(cache) + ", cache if no sharing " + human(mha) + ".");
    }
    draw(); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 3. the mask gallery ----------
  function maskFig() {
    var bar = Kit.$("mk-bar"), ctl = Kit.$("mk-controls"), cv = Kit.$("mk-canvas"), st = { pat: "window", causal: true, row: 40, sq: false };
    var NAMES = ["Full", "Sliding window", "Strided", "Global + window", "Random + window + global"];
    rowChips(bar, NAMES, 1, function (i) { st.pat = PATTERNS[i]; draw(); });
    rowChips(bar, ["Reads the past only", "Reads both ways"], 0, function (i) { st.causal = i === 0; draw(); });
    var sn = Kit.slider(ctl, { label: "Tokens n", min: 12, max: 96, step: 1, value: 64 }, draw);
    var sw = Kit.slider(ctl, { label: "Window width w", min: 2, max: 32, step: 1, value: 8 }, draw);
    var sl = Kit.slider(ctl, { label: "Stride l", min: 2, max: 16, step: 1, value: 8 }, draw);
    var sg = Kit.slider(ctl, { label: "Global tokens g", min: 1, max: 6, step: 1, value: 2 }, draw);
    var sr = Kit.slider(ctl, { label: "Random keys per row r", min: 0, max: 6, step: 1, value: 3 }, draw);
    var sq = Kit.slider(ctl, { label: "Show the row of token i", min: 0, max: 95, step: 1, value: 40, digits: 0 }, draw);
    rowChips(bar, ["Stride fixed", "Stride = square root of n"], 0, function (i) { st.sq = i === 1; draw(); });
    var COL = function (P) { return { full: P.teal, win: P.navy, glob: P.coral, str: P.sand, rnd: P.sand }; };
    function params() { return { n: sn.get(), w: sw.get(), l: sl.get(), g: sg.get(), r: sr.get(), causal: st.causal }; }
    function draw() {
      var P = Kit.palette(), Q = params(), n = Q.n, side = Math.min(Kit.width(cv.parentNode) - 8, 460), cell = side / n, dpr = window.devicePixelRatio || 1;
      if (st.pat === "strided") Q.causal = true;
      if (st.sq) { Q.l = Math.max(2, Math.round(Math.sqrt(n))); sl.set(Math.min(16, Q.l)); }
      var cz = Q.causal;
      sq.input.max = n - 1; if (sq.get() > n - 1) sq.set(n - 1); var row = sq.get();
      cv.width = Math.round(side * dpr); cv.height = Math.round(side * dpr); cv.style.width = side + "px"; cv.style.height = side + "px";
      var ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, side, side);
      var colors = COL(P); ctx.fillStyle = P.paper; ctx.fillRect(0, 0, side, side);
      var total = 0, i, j;
      for (i = 0; i < n; i++) {
        var rc = rowCells(st.pat, i, Q);
        for (var k in rc) { j = +k; ctx.fillStyle = colors[rc[k]]; ctx.globalAlpha = (i === row) ? 1 : 0.78; ctx.fillRect(j * cell, i * cell, Math.max(1, cell - (cell > 6 ? 0.8 : 0)), Math.max(1, cell - (cell > 6 ? 0.8 : 0))); total++; }
      }
      ctx.globalAlpha = 1; ctx.strokeStyle = P.coral; ctx.lineWidth = 2; ctx.strokeRect(1, row * cell, side - 2, cell);
      ctx.strokeStyle = P.rule; ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, side - 1, side - 1);
      var full = cz ? n * (n + 1) / 2 : n * n, rowN = rowCount(st.pat, row, Q), chk = pairsTotal(st.pat, n, Q);
      if (chk !== total) console.error("pair count mismatch", chk, total);
      var legend = { full: "computed pair", win: "window", glob: "global token", str: "l-th, 2l-th, ... token", rnd: "random key" }, used = {};
      for (i = 0; i < n; i++) { var rr = rowCells(st.pat, i, Q); for (var kk in rr) used[rr[kk]] = 1; }
      Kit.$("mk-legend").innerHTML = Object.keys(legend).filter(function (c) { return used[c]; }).map(function (c) { return '<span class="lg"><i style="background:' + colors[c] + '"></i>' + legend[c] + "</span>"; }).join("") + '<span class="lg"><i class="out" style="border-color:' + P.coral + '"></i>row of token ' + row + " (" + rowN + " keys)</span><span class=\"lg\"><i class=\"emp\"></i>skipped</span>";
      var ds = cz ? "causal" : "both ways";
      Kit.$("mk-read").innerHTML = "<span>Pairs computed: <b>" + nf(total) + "</b></span><span>Full (" + ds + "): <b>" + nf(full) + "</b></span><span>Share kept: <b>" + (100 * total / full).toFixed(total === full ? 0 : 1) + "%</b></span><span>Token " + row + " looks at <b>" + rowN + "</b> of " + (cz ? row + 1 : n) + " allowed tokens</span>";
      var wq = Q.w, h = Math.floor(wq / 2);
      var eq1 = st.pat === "window" ? (Q.causal ? box(nf(wq), "w", "c-navy") + op("&times;") + box(nf(n), "n", "c-sand") + op("&minus;") + box(nf(wq * (wq - 1) / 2), "w(w&minus;1)/2", "c-sand") + op("=") + box(nf(wq * n - wq * (wq - 1) / 2), "pairs", "c-teal")
        : box(nf(2 * h + 1), "2&lfloor;w/2&rfloor; + 1", "c-navy") + op("&times;") + box(nf(n), "n", "c-sand") + op("&minus;") + box(nf(h * (h + 1)), "h(h+1)", "c-sand") + op("=") + box(nf((2 * h + 1) * n - h * (h + 1)), "pairs", "c-teal"))
        : box(nf(total), "pairs computed", "c-teal") + op("of") + box(nf(full), "full", "c-sand");
      Kit.$("mk-eq").innerHTML = '<div class="eqn">' + eq1 + "</div>";
      cv.setAttribute("role", "img"); cv.setAttribute("aria-label", "Grid of " + n + " by " + n + " cells. Rows are tokens that look, columns are tokens looked at. Pattern: " + NAMES[PATTERNS.indexOf(st.pat)] + ", " + nf(total) + " of " + nf(full) + " pairs computed.");
      curves(P, Q);
      Kit.$("mk-status").textContent = (st.pat === "strided" ? "The strided pattern is defined for models that read left to right, so it ignores the reading switch. " : "") + (st.pat === "global" && st.causal ? "When tokens read only the past, a global token at the start is seen by all later tokens but sees no earlier token. This is the first-token effect of the next section. " : "") + (st.pat === "bigbird" ? "The random keys of a row are drawn from a fixed seed, so the same picture appears for the same sliders." : "");
    }
    var ns = [], q; for (q = 0; q <= 14; q++) ns.push(Math.round(Math.pow(10, 1.7 + 3.3 * q / 14)));
    function curves(P, Q) {
      var lq = Object.assign({}, Q); if (st.pat === "strided") lq.causal = true;
      function at(n) { var qq = Object.assign({}, lq); qq.n = n; if (st.sq) qq.l = Math.max(2, Math.round(Math.sqrt(n))); return qq; }
      var sel = ns.map(function (n) { return pairsTotal(st.pat, n, at(n)); });
      var fullc = ns.map(function (n) { return lq.causal ? n * (n + 1) / 2 : n * n; });
      var n1 = 10000, n2 = 20000, q1 = at(n1), q2 = at(n2);
      var grow = pairsTotal(st.pat, n2, q2) / pairsTotal(st.pat, n1, q1), growFull = (lq.causal ? n2 * (n2 + 1) / (n1 * (n1 + 1)) : 4);
      Kit.plot("mk-plot", function (P2) {
        return [
          { x: ns, y: fullc, mode: "lines", line: { color: P2.sand, width: 3, dash: "dot" }, name: "full", hovertemplate: "n = %{x:,}<br>%{y:.3s} pairs<extra>full</extra>" },
          { x: ns, y: sel, mode: "lines+markers", line: { color: P2.teal, width: 3 }, marker: { size: 5 }, name: "this pattern", hovertemplate: "n = %{x:,}<br>%{y:.3s} pairs<extra>this pattern</extra>" },
          { x: [Q.n], y: [pairsTotal(st.pat, Q.n, at(Q.n))], mode: "markers", marker: { color: P2.coral, size: 11, line: { color: P2.bg, width: 2 } }, hoverinfo: "skip" }
        ];
      }, function () { return { xaxis: { type: "log", title: { text: "tokens n" } }, yaxis: { type: "log", title: { text: "pairs computed" } }, margin: { l: 70, r: 12, t: 10, b: 54 } }; });
      Kit.$("mk-plot").setAttribute("aria-label", "Log-log chart of pairs computed against tokens for the full mask and the chosen pattern. Going from 10,000 to 20,000 tokens multiplies the full count by " + growFull.toFixed(2) + " and this pattern by " + grow.toFixed(2) + ".");
      Kit.$("mk-grow").innerHTML = "From 10,000 to 20,000 tokens the pairs grow <b>&times;" + growFull.toFixed(2) + "</b> for the full mask and <b>&times;" + grow.toFixed(2) + "</b> for this pattern.";
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 4. who can see whom ----------
  function reachFig() {
    var bar = Kit.$("rc-bar"), ctl = Kit.$("rc-controls"), svg = Kit.$("rc-svg"), N = 25, st = { pat: "window", causal: true }, top = 24;
    var NAMES = ["Window, reads the past", "Window, reads both ways", "Window + global token, both ways", "Strided, reads the past"];
    var MODE = [["window", true], ["window", false], ["global", false], ["strided", true]];
    rowChips(bar, NAMES, 0, function (i) { st.pat = MODE[i][0]; st.causal = MODE[i][1]; draw(); });
    var sw = Kit.slider(ctl, { label: "Window width w", min: 2, max: 7, step: 1, value: 3 }, draw);
    var sL = Kit.slider(ctl, { label: "Layers L", min: 1, max: 8, step: 1, value: 3 }, draw);
    var sq = Kit.slider(ctl, { label: "Token at the top", min: 0, max: N - 1, step: 1, value: 24 }, draw);
    function draw() {
      var P = Kit.palette(), L = sL.get(), Q = { n: N, w: sw.get(), l: 5, g: 1, r: 0, causal: st.causal }; top = sq.get();
      var sets = reachSets(st.pat, Q, top, L), W = Math.max(300, Kit.width(svg.parentNode) - 8), left = 44, dx = (W - left - 10) / (N - 1), dy = 46, H = 28 + L * dy + 30;
      svgSize(svg, W, H); clear(svg);
      var yOf = function (l) { return 18 + (L - l) * dy; }, xOf = function (i) { return left + i * dx; };
      var l, i, inSet = sets.map(function (s) { var o = {}; s.forEach(function (j) { o[j] = 1; }); return o; });
      for (l = 1; l <= L; l++) sets[l].forEach(function (i2) {
        neighbours(st.pat, i2, Q).forEach(function (j) { if (inSet[l - 1][j]) Kit.el("line", { x1: xOf(j), y1: yOf(l - 1), x2: xOf(i2), y2: yOf(l), stroke: P.teal, "stroke-opacity": 0.28, "stroke-width": 1 }, svg); });
      });
      for (l = 0; l <= L; l++) {
        svgText(svg, left - 10, yOf(l) + 4, l === 0 ? "input" : "layer " + l, { "font-size": 11.5, fill: P.muted, "text-anchor": "end" });
        for (i = 0; i < N; i++) {
          var on = inSet[l][i], isTop = l === L && i === top, fill = isTop ? P.coral : on ? P.teal : "none";
          Kit.el("circle", { cx: xOf(i), cy: yOf(l), r: isTop ? 6 : on ? 4.6 : 3, fill: fill, stroke: on || isTop ? "none" : P.muted, "stroke-width": 1, "fill-opacity": isTop ? 1 : 0.9 }, svg);
        }
      }
      for (i = 0; i < N; i += 4) svgText(svg, xOf(i), H - 8, String(i + 1), { "font-size": 10.5, fill: P.muted, "text-anchor": "middle" });
      svgText(svg, left - 10, H - 8, "token", { "font-size": 11.5, fill: P.muted, "text-anchor": "end" });
      var allowed = (st.causal || st.pat === "strided") ? top + 1 : N, cover = sets[0].length, need = -1;
      for (var k = 1; k <= 12; k++) { if (reachSets(st.pat, Q, top, k)[0].length === allowed) { need = k; break; } }
      Kit.$("rc-read").innerHTML = "<span>Input tokens that reach token " + (top + 1) + ": <b>" + cover + "</b> of " + allowed + " allowed</span><span>Layers needed to reach all of them: <b>" + (need < 0 ? "more than 12" : need) + "</b></span><span>Growth per layer: " + (st.pat === "window" && st.causal ? "<b>w &minus; 1 = " + (Q.w - 1) + "</b> tokens" : st.pat === "window" ? "<b>2&lfloor;w/2&rfloor; = " + 2 * Math.floor(Q.w / 2) + "</b> tokens" : st.pat === "global" ? "all, through the global token" : "<b>about &radic;n</b>") + "</span>";
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Stack of " + L + " layers over " + N + " tokens. Token " + (top + 1) + " at the top receives information from " + cover + " input tokens.");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 5. cutting GPT-2's attention ----------
  function cutFig() {
    var bar = Kit.$("gw-bar"), ctl = Kit.$("gw-controls"), svg = Kit.$("gw-grid"), WS = WIN.windows, st = { first: true }, sel = [5, 5];
    var sl = Kit.slider(ctl, { label: "Window w, in tokens (log steps)", min: 0, max: WS.length - 1, step: 1, value: 6, show: function (v) { return nf(WS[v]); }, digits: 0 }, draw);
    rowChips(bar, ["Window only", "Window + first token"], 1, function (i) { st.first = i === 1; draw(); });
    function draw() {
      var P = Kit.palette(), wi = sl.get(), w = WS[wi], sh = st.first ? WIN.share_sink : WIN.share, N = WIN.tokens, causalPairs = N * (N + 1) / 2;
      var kept = w >= N ? causalPairs : w * N - w * (w - 1) / 2 + (st.first ? Math.max(0, N - w) : 0);
      Kit.plot("gw-plot", function (P2) {
        return [
          { x: WS, y: WIN.loss_window, mode: "lines+markers", name: "window only", line: { color: P2.navy, width: 3 }, marker: { size: 6 }, hovertemplate: "w = %{x}<br>loss %{y:.2f}<extra>window only</extra>" },
          { x: WS, y: WIN.loss_window_first, mode: "lines+markers", name: "window + first token", line: { color: P2.teal, width: 3 }, marker: { size: 6 }, hovertemplate: "w = %{x}<br>loss %{y:.2f}<extra>window + first token</extra>" },
          { x: [WS[0], WS[WS.length - 1]], y: [WIN.loss_full, WIN.loss_full], mode: "lines", showlegend: false, line: { color: P2.sand, width: 2, dash: "dot" }, hoverinfo: "skip" },
          { x: [w], y: [st.first ? WIN.loss_window_first[wi] : WIN.loss_window[wi]], mode: "markers", showlegend: false, marker: { color: P2.coral, size: 13, line: { color: P2.bg, width: 2 } }, hoverinfo: "skip" }
        ];
      }, function (P2) {
        return { xaxis: { type: "log", title: { text: "window w, in tokens" }, tickvals: [1, 4, 16, 64, 256, 1024], ticktext: ["1", "4", "16", "64", "256", "1024"] }, yaxis: { type: "log", title: { text: "loss per token, nats" }, tickvals: [1, 2, 4, 8, 16], ticktext: ["1", "2", "4", "8", "16"] }, margin: { l: 58, r: 12, t: 34, b: 46 },
          showlegend: true, legend: { orientation: "h", x: 0, y: 1.14, font: { size: 13 } },
          annotations: [{ x: log10(WS[1]), y: log10(WIN.loss_full) + 0.04, text: "full attention " + WIN.loss_full.toFixed(2), showarrow: false, xanchor: "left", yanchor: "bottom", font: { color: P2.ink2, size: 12.5 } }] };
      });
      // grid of 12 x 12 heads
      var cs = Math.max(16, Math.min(30, Math.floor((Kit.width(svg.parentNode) - 60) / 12))), left = 44, top = 20;
      svgSize(svg, left + 12 * cs + 4, top + 12 * cs + 22); clear(svg);
      svgText(svg, left - 8, top - 7, "head", { "font-size": 11, fill: P.muted, "text-anchor": "end" });
      for (var h = 0; h < 12; h++) svgText(svg, left + h * cs + cs / 2, top - 7, String(h), { "font-size": 10.5, fill: P.muted, "text-anchor": "middle" });
      for (var ly = 0; ly < 12; ly++) {
        svgText(svg, left - 8, top + ly * cs + cs / 2 + 4, "L" + ly, { "font-size": 10.5, fill: P.muted, "text-anchor": "end" });
        for (h = 0; h < 12; h++) {
          var v = sh[ly][h][wi], g = Kit.el("g", {}, svg);
          Kit.el("rect", { x: left + h * cs, y: top + ly * cs, width: cs - 2, height: cs - 2, rx: 3, fill: P.teal, "fill-opacity": 0.08 + 0.85 * v, stroke: (ly === sel[0] && h === sel[1]) ? P.coral : P.rule, "stroke-width": (ly === sel[0] && h === sel[1]) ? 2.5 : 0.6, style: "cursor:pointer" }, g);
          var tt = Kit.el("title", {}, g); tt.textContent = "layer " + ly + ", head " + h + ": " + (100 * v).toFixed(0) + "% of its weight is inside the window";
          g.addEventListener("click", (function (a, b) { return function () { sel = [a, b]; draw(); }; })(ly, h));
        }
      }
      svgText(svg, left, top + 12 * cs + 15, "stronger colour = more of the head's weight is kept", { "font-size": 11.5, fill: P.muted });
      var all = 0; for (ly = 0; ly < 12; ly++) for (h = 0; h < 12; h++) all += sh[ly][h][wi]; all /= 144;
      var s0 = sh[sel[0]][sel[1]][wi], ft = WIN.first_token_share[sel[0]][sel[1]];
      Kit.$("gw-read").innerHTML = "<span>Pairs computed: <b>" + nf(kept) + "</b> of " + nf(causalPairs) + " (<b>" + (100 * kept / causalPairs).toFixed(1) + "%</b>)</span><span>Loss: <b>" + (st.first ? WIN.loss_window_first[wi] : WIN.loss_window[wi]).toFixed(2) + "</b> (full: " + WIN.loss_full.toFixed(2) + ")</span><span>Weight kept, mean of 144 heads: <b>" + (100 * all).toFixed(0) + "%</b></span><span>Head L" + sel[0] + "H" + sel[1] + ": <b>" + (100 * s0).toFixed(0) + "%</b> kept, " + (100 * ft).toFixed(0) + "% on the first token</span>";
      Kit.$("gw-plot").setAttribute("aria-label", "Loss of GPT-2 small on a 1,024-token logbook against the window size. With the window alone the loss is " + WIN.loss_window[6].toFixed(1) + " at 128 tokens, with the first token kept it is " + WIN.loss_window_first[6].toFixed(1) + ", and full attention gives " + WIN.loss_full.toFixed(2) + ".");
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Grid of the 144 heads of GPT-2 small, shaded by the share of attention weight inside the window.");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 6. fixed-size memories ----------
  var MEM_D = 16, MEM_N = 24, TT_D = 8, TT_N = 16;
  function errPlot(id, errs, n, P, hatchFrom) {
    var cols = errs.map(function (e) { return e < 0.25 ? P.teal : e < 0.6 ? P.sand : P.coral; });
    Kit.plot(id, [{ type: "bar", x: errs.map(function (_, i) { return i + 1; }), y: errs.map(function (e) { return Math.min(e, 1.6); }), marker: { color: cols }, hovertemplate: "item %{x}<br>error %{y:.2f}<extra></extra>" },
      { x: [0.4, n + 0.6], y: [0.25, 0.25], mode: "lines", line: { color: P.muted, width: 1.5, dash: "dot" }, hoverinfo: "skip" }],
    { xaxis: { title: { text: "item stored (1 = the oldest)" }, range: [0.4, MEM_N + 0.6], dtick: 2 }, yaxis: { title: { text: "error when read back" }, range: [0, 1.65] }, margin: { l: 60, r: 10, t: 24, b: 54 },
      annotations: [{ xref: "paper", yref: "paper", x: 1, y: 1.02, text: "dotted line: error 0.25, good enough", showarrow: false, xanchor: "right", yanchor: "bottom", font: { color: P.muted, size: 12 } }] });
  }
  function memFig() {
    var bar = Kit.$("mem-bar"), ctl = Kit.$("mem-controls"), items = memItems(MEM_N, MEM_D, 4127, true), types = ["full", "window", "sum", "decay", "delta"];
    var NAMES = ["Full cache", "Window of 4", "Sum (linear attention)", "Decay (state space)", "Delta rule (test-time training)"], ti = 2, timer = 0;
    rowChips(bar, NAMES, 2, function (i) { ti = i; draw(); });
    var sn = Kit.slider(ctl, { label: "Items stored n", min: 1, max: MEM_N, step: 1, value: 20 }, draw);
    var sg = Kit.slider(ctl, { label: "Decay factor per item", min: 0.5, max: 1, step: 0.01, value: 0.85 }, draw);
    var btn = Kit.html("div", { class: "row-chips" }, bar);
    Kit.button(btn, "Store next item", function () { stop(); if (sn.get() < MEM_N) sn.set(sn.get() + 1); draw(); });
    var bp = Kit.button(btn, "Play", function () { if (timer) { stop(); return; } if (sn.get() >= MEM_N) sn.set(1); bp.textContent = "Pause"; draw(); timer = setInterval(function () { if (sn.get() >= MEM_N) { stop(); return; } sn.set(sn.get() + 1); draw(); }, 650); });
    Kit.button(btn, "Reset", function () { stop(); sn.set(1); draw(); });
    function stop() { if (timer) clearInterval(timer); timer = 0; bp.textContent = "Play"; }
    function draw() {
      var n = sn.get(), type = types[ti], P = Kit.palette(), errs = memErrors(type, items, n, { w: 4, gamma: sg.get(), eta: 1, tau: 12 });
      sg.input.parentNode.style.opacity = type === "decay" ? 1 : 0.4; sg.input.disabled = type !== "decay";
      errPlot("mem-plot", errs, n, P);
      var good = errs.filter(function (e) { return e < 0.25; }).length, mean = errs.reduce(function (a, b) { return a + b; }, 0) / n;
      var kept = type === "full" ? 2 * n * MEM_D : type === "window" ? 2 * Math.min(n, 4) * MEM_D : MEM_D * MEM_D;
      Kit.$("mem-read").innerHTML = "<span>Items stored: <b>" + n + "</b></span><span>Numbers kept: <b>" + kept + "</b>" + (type === "full" ? " (2 &times; n &times; 16)" : type === "window" ? " (2 &times; 4 &times; 16)" : " (a 16 by 16 matrix)") + "</span><span>Read back well: <b>" + good + "</b> of " + n + "</span><span>Mean error: <b>" + mean.toFixed(2) + "</b></span>";
      Kit.$("mem-status").textContent = [
        "The cache keeps all keys and values and picks one with a sharp softmax, so the items come back almost without error. The price is 2 × n × 16 numbers, which grows with n.",
        "Only the last 4 items are kept. Older items cannot be read: the error is about 1, the size of the value itself.",
        "One 16 by 16 matrix stores a sum of outer products. While the keys are well separated, up to 16 items come back without error. Items beyond the 16th cannot have separate keys, so they add noise to all the others.",
        "A new item first shrinks the matrix by the decay factor. The newest items come back well and the oldest fade, even when there is space left.",
        "A write first reads what the matrix already predicts for the key and stores only the difference. Up to 16 items come back without error. After that the newest item is right and older ones are overwritten in part."][ti];
      Kit.$("mem-plot").setAttribute("aria-label", "Bars of the error when " + n + " stored items is read back from the " + NAMES[ti] + " memory. " + good + " are read back well.");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 7. test-time training, chunk by chunk ----------
  function tttFig() {
    var bar = Kit.$("tt-bar"), ctl = Kit.$("tt-controls"), svg = Kit.$("tt-svg"), N = TT_N, items = memItems(N, TT_D, 3305, true), st = { loss: "sq", step: 0, reps: 1 }, timer = 0;
    rowChips(bar, ["Squared error (TTT layers)", "Dot product (In-Place TTT)"], 0, function (i) { st.loss = i === 0 ? "sq" : "dot"; draw(); });
    rowChips(bar, ["Chunks read once", "Chunks read twice"], 0, function (i) { st.reps = i + 1; draw(); });
    var cs = Kit.slider(ctl, { label: "Chunk size c, in tokens", min: 0, max: 3, step: 1, value: 2, show: function (v) { return nf(Math.pow(2, v)); }, digits: 0 }, function () { st.step = 0; draw(); });
    var es = Kit.slider(ctl, { label: "Step size η", min: 0.05, max: 1.2, step: 0.05, value: 1 }, draw);
    var btn = Kit.html("div", { class: "row-chips" }, bar);
    Kit.button(btn, "Read next chunk", function () { stop(); next(); });
    var bp = Kit.button(btn, "Play", function () { if (timer) { stop(); return; } if (st.step >= N / chunk()) st.step = 0; bp.textContent = "Pause"; next(); timer = setInterval(function () { if (st.step >= N / chunk()) { stop(); return; } next(); }, 900); });
    Kit.button(btn, "Reset", function () { stop(); st.step = 0; draw(); });
    function chunk() { return Math.pow(2, cs.get()); }
    function stop() { if (timer) clearInterval(timer); timer = 0; bp.textContent = "Play"; }
    function next() { if (st.step < N / chunk()) st.step++; draw(); }
    function draw() {
      var P = Kit.palette(), c = chunk(), run = ttRun(items, N, { c: c, eta: es.get(), loss: st.loss, reps: st.reps }), k = Math.min(st.step, N / c), read = k * c;
      var W = Math.max(300, Kit.width(svg.parentNode) - 8), tw = Math.min(34, (W - 20) / N), H = 252, i, j;
      svgSize(svg, W, H); clear(svg);
      svgText(svg, 4, 14, "tokens of the text, read in chunks of " + c, { "font-size": 12.5, fill: P.muted });
      for (i = 0; i < N; i++) {
        var chk = Math.floor(i / c), done = i < read, cur = chk === k - 1 && k > 0, x = 4 + i * tw;
        Kit.el("rect", { x: x, y: 22, width: tw - 3, height: 24, rx: 3, fill: done ? (cur ? P.coral : P.navy) : "none", "fill-opacity": done ? (cur ? 0.9 : 0.4) : 1, stroke: done ? "none" : P.muted, "stroke-width": 1 }, svg);
        svgText(svg, x + (tw - 3) / 2, 38, String(i + 1), { "font-size": 11, fill: done ? P.bg : P.muted, "text-anchor": "middle" });
      }
      for (var q = 0; q < N / c; q++) {
        var x0 = 4 + q * c * tw, x1 = x0 + c * tw - 3, passed = q < k - 1 || (q === k - 1);
        Kit.el("path", { d: "M" + x0 + " 52 v5 H" + x1 + " v-5", fill: "none", stroke: passed ? P.coral : P.rule, "stroke-width": 1.3 }, svg);
        if (c * tw >= 28) svgText(svg, (x0 + x1) / 2, 74, q === 0 ? "read with W₀" : "with W" + String.fromCharCode(8320 + Math.min(q, 9)), { "font-size": 10.5, fill: passed ? P.ink2 : P.muted, "text-anchor": "middle" });
      }
      var S = run.mats[k] || run.mats[run.mats.length - 1], mx = 0;
      run.mats.forEach(function (M) { M.forEach(function (r) { r.forEach(function (v) { mx = Math.max(mx, Math.abs(v)); }); }); });
      mx = Math.max(mx, 0.5); var cell = 16, mx0 = 18, my0 = 122;
      svgText(svg, mx0, 108, "fast weight W (8 × 8 = 64 numbers)", { "font-size": 12.5, fill: P.ink2 });
      for (i = 0; i < TT_D; i++) for (j = 0; j < TT_D; j++) {
        var v = S[i][j], a = Math.min(1, Math.abs(v) / mx);
        Kit.el("rect", { x: mx0 + j * cell, y: my0 + i * cell, width: cell - 1.5, height: cell - 1.5, rx: 2, fill: v >= 0 ? P.teal : P.coral, "fill-opacity": 0.1 + 0.8 * a }, svg);
      }
      var gx = mx0 + TT_D * cell + 24;
      Kit.el("rect", { x: gx, y: my0, width: Math.max(100, Math.min(W - gx - 8, 180)), height: 8 * cell - 2, rx: 6, fill: "none", stroke: P.rule, "stroke-dasharray": "4 3" }, svg);
      svgText(svg, gx + 10, my0 + 22, "all other weights:", { "font-size": 12.5, fill: P.muted });
      svgText(svg, gx + 10, my0 + 40, "frozen", { "font-size": 12.5, fill: P.muted });
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "After " + k + " of " + N / c + " chunks the fast weight matrix W has been updated " + k + " times. All other weights stay fixed.");
      var errs = readErrors(S, items, N).slice(0, Math.max(read, 0)), pl = Kit.$("tt-plot");
      if (read > 0) {
        Kit.plot("tt-plot", function (P2) { return [{ type: "bar", x: errs.map(function (_, i2) { return i2 + 1; }), y: errs.map(function (e) { return Math.min(e, 1.6); }), marker: { color: errs.map(function (e) { return e < 0.25 ? P2.teal : e < 0.6 ? P2.sand : P2.coral; }) }, hovertemplate: "token %{x}<br>error %{y:.2f}<extra></extra>" }, { x: [0.4, N + 0.6], y: [0.25, 0.25], mode: "lines", line: { color: P2.muted, width: 1.5, dash: "dot" }, hoverinfo: "skip" }]; },
          { xaxis: { title: { text: "token read so far" }, range: [0.4, N + 0.6], dtick: 2 }, yaxis: { title: { text: "error, read back with W" }, range: [0, 1.65] }, margin: { l: 60, r: 10, t: 24, b: 54 },
            annotations: [{ xref: "paper", yref: "paper", x: 1, y: 1.02, text: "dotted line: error 0.25, good enough", showarrow: false, xanchor: "right", yanchor: "bottom", font: { color: P.muted, size: 12 } }] });
      } else {
        Kit.plot("tt-plot", [], { xaxis: { title: { text: "token read so far" }, range: [0.4, N + 0.6], dtick: 2 }, yaxis: { title: { text: "error, read back with W" }, range: [0, 1.65] }, margin: { l: 60, r: 10, t: 24, b: 54 },
          annotations: [{ xref: "paper", yref: "paper", x: 0.5, y: 0.5, text: "Press Read next chunk.<br>The bars show how well W gives back<br>the tokens it has read.", showarrow: false, font: { color: P.muted, size: 14 } }] });
      }
      var ls = k > 0 ? run.loss[k - 1] : null;
      Kit.$("tt-read").innerHTML = "<span>Chunks read: <b>" + k + "</b> of " + N / c + "</span><span>Updates done: <b>" + k + "</b>, one at the end of a chunk</span>" + (ls ? "<span>Loss on the last chunk: <b>" + ls[0].toFixed(2) + "</b> before, <b>" + ls[1].toFixed(2) + "</b> after the step</span>" : "") + (read ? "<span>Read back well: <b>" + errs.filter(function (e) { return e < 0.25; }).length + "</b> of " + read + "</span>" : "");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 8. four routes ----------
  function routesFig() {
    var bar = Kit.$("rt-bar"), svg = Kit.$("rt-svg"), tb = Kit.$("rt-table"), step = 0;
    var R = [
      { name: "Full attention", mem: "grows with n", work: "n pairs per new token", keeps: "all tokens", loses: "no information, but the cost grows" },
      { name: "Sparse attention", mem: "grows with n unless old keys are dropped; w + g if they are", work: "w + g (+ r) pairs per new token", keeps: "recent tokens, a few global ones", loses: "far tokens outside the pattern, unless trained with it" },
      { name: "Linear attention and state-space models", mem: "fixed: one matrix per head", work: "one matrix update per new token", keeps: "a blurred sum of the past", loses: "perfect recall when more is stored than the matrix can separate" },
      { name: "Test-time training", mem: "fixed: one weight matrix per updated layer", work: "an ordinary read, plus one update per chunk", keeps: "what the update rule wrote into W", loses: "what the update rule chose not to write; W resets for the next input" }
    ];
    Kit.button(bar, "Back", function () { step = Math.max(0, step - 1); draw(); });
    Kit.button(bar, "Next route", function () { step = Math.min(3, step + 1); draw(); });
    rowChips(bar, ["Full", "Sparse", "Linear / state-space", "Test-time training"], 0, function (i) { step = i; draw(); });
    var sn = logSlider(Kit.$("rt-controls"), "Length n, in tokens", 3, 6, 32768, draw);
    function draw() {
      var P = Kit.palette(), n = sn.n(), m = MODELS.q4, dkv = m.nKV * m.dh, w = 4096, g = 4;
      var W = Math.max(300, Kit.width(svg.parentNode) - 8), H = step >= 2 ? 172 : 138, N = 16, tw = Math.min(30, (W - 16) / N), i;
      svgSize(svg, W, H); clear(svg);
      svgText(svg, 4, 14, "the text (16 tokens shown) and what the route keeps", { "font-size": 12.5, fill: P.muted });
      for (i = 0; i < N; i++) Kit.el("rect", { x: 4 + i * tw, y: 22, width: tw - 3, height: 20, rx: 3, fill: P.navy, "fill-opacity": 0.35 }, svg);
      var y = 70, x;
      if (step === 0) { for (i = 0; i < N; i++) { Kit.el("rect", { x: 4 + i * tw, y: y, width: tw - 3, height: 22, rx: 3, fill: P.coral, "fill-opacity": 0.75 }, svg); Kit.el("rect", { x: 4 + i * tw, y: y + 26, width: tw - 3, height: 22, rx: 3, fill: P.teal, "fill-opacity": 0.75 }, svg); } }
      if (step === 1) { for (i = 0; i < N; i++) { var keep = i < 2 || i >= N - 5; Kit.el("rect", { x: 4 + i * tw, y: y, width: tw - 3, height: 22, rx: 3, fill: keep ? P.coral : "none", "fill-opacity": 0.75, stroke: keep ? "none" : P.muted, "stroke-dasharray": keep ? "0" : "3 2" }, svg); Kit.el("rect", { x: 4 + i * tw, y: y + 26, width: tw - 3, height: 22, rx: 3, fill: keep ? P.teal : "none", "fill-opacity": 0.75, stroke: keep ? "none" : P.muted, "stroke-dasharray": keep ? "0" : "3 2" }, svg); } }
      if (step >= 2) {
        var cell = 12, x0 = 8; for (i = 0; i < 8; i++) for (var j = 0; j < 8; j++) Kit.el("rect", { x: x0 + j * cell, y: y - 4 + i * cell, width: cell - 1.5, height: cell - 1.5, rx: 2, fill: ((i * 7 + j * 3) % 5 < 2 ? P.coral : P.teal), "fill-opacity": 0.25 + 0.12 * ((i + j * 2) % 5) }, svg);
        svgText(svg, x0 + 8 * cell + 14, y + 12, step === 2 ? "one matrix S, the same size at any length" : "one weight matrix W, updated after a chunk", { "font-size": 12, fill: P.ink2 });
        svgText(svg, x0 + 8 * cell + 14, y + 32, step === 2 ? "a token adds to it, and it may fade (decay)" : "the loss comes from the text, so no labels", { "font-size": 12, fill: P.ink2 });
        if (step === 3) { var c = 4; for (i = 0; i < N / c; i++) { var xa = 4 + i * c * tw, xb = xa + c * tw - 3; Kit.el("path", { d: "M" + xa + " 46 v5 H" + xb + " v-5", fill: "none", stroke: P.coral, "stroke-width": 1.3 }, svg); } }
      }
      var cacheNum = 2 * n * dkv, state = m.nKV * m.dh * m.dh, rows = [
        ["Full attention", "2 &times; n &times; " + nf(dkv) + " = " + nf(cacheNum), "n = " + nf(n)],
        ["Sparse (w = " + nf(w) + ", g = " + g + ")", "2 &times; (w + g) &times; " + nf(dkv) + " = " + nf(2 * Math.min(n, w + g) * dkv), nf(Math.min(n, w + g))],
        ["Linear or state-space", nf(state) + " (8 matrices of 128 by 128)", "one update of " + nf(state) + " numbers"],
        ["Test-time training", "24,903,680 per updated layer (the down projection of In-Place TTT in Qwen3-4B, kept beside the cache)", "one read; one update per chunk of c tokens"]];
      tb.innerHTML = "<table class=\"sv-table\"><thead><tr><th>Route</th><th>Memory per layer, numbers (Qwen3-4B sizes)</th><th>Work per new token</th></tr></thead><tbody>" + rows.map(function (r, k) { return "<tr" + (k === step ? ' class="on"' : "") + "><td>" + r[0] + "</td><td>" + r[1] + "</td><td>" + r[2] + "</td></tr>"; }).join("") + "</tbody></table>";
      var r = R[step];
      Kit.$("rt-text").innerHTML = "<b>" + r.name + ".</b> Memory: " + r.mem + ". Work: " + r.work + ". It keeps " + r.keeps + ". It loses " + r.loses + ".";
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Diagram of what the " + r.name + " route keeps in memory: " + r.keeps + ".");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }
})();
