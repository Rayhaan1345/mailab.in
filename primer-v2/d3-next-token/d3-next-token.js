// D3 Predicting the next token. Figures: next-token bars from GPT-2 (two instances: scores and probabilities, then temperature, top-k, top-p and
// Sample), a generation loop with a tiny word model, the context window, one training step on one example, and per-token loss with perplexity.
// The pure logic (items, distribution, sampling helpers) is exported with module.exports for node tests.
(function () {
  "use strict";
  // ---------- pure logic ----------
  // items: named tokens (one token per item) followed by tail groups (mult tokens with about the same score). z is a score relative to the top score.
  function buildItems(ctx) {
    var items = ctx.top.map(function (t) { return { tok: t[0], z: t[1], mult: 1 }; });
    var tl = ctx.tail, tail = [];
    tl.counts.forEach(function (c, i) { if (c > 0) tail.push({ tok: null, z: tl.lo + (i + 0.5) * tl.w, mult: c }); });
    tail.sort(function (a, b) { return b.z - a.z; });
    return items.concat(tail);
  }
  // Temperature, then top-k, then top-p, then renormalise. k = 0 and p >= 1 switch them off.
  // Returns {rows: [{tok, z, mult, base, kept, final}], Z, entropy, nkept, keptMass} where base and final are the mass of the item (all its tokens).
  function distribution(items, T, k, p) {
    var Z = 0, i, rows = items.map(function (it) { var e = Math.exp(it.z / T); Z += it.mult * e; return { tok: it.tok, z: it.z, mult: it.mult, e: e, base: 0, keep: 1, final: 0 }; });
    rows.forEach(function (r) { r.base = r.mult * r.e / Z; });
    var named = 0; rows.forEach(function (r) { if (r.tok !== null) named++; });
    if (k > 0) rows.forEach(function (r, j) { if (j >= k) r.keep = 0; });
    var mass = 0; rows.forEach(function (r) { mass += r.keep * r.base; });
    var keptAfterK = mass;
    if (p < 1) {
      var cum = 0;
      rows.forEach(function (r) {
        if (!r.keep) return;
        var q = r.keep * r.base / mass;
        if (cum >= p - 1e-12) { r.keep = 0; return; }
        if (cum + q > p && r.mult > 1) r.keep = Math.min(1, Math.ceil((p - cum) / (q / r.mult)) / r.mult);
        cum += q;
      });
    }
    var m2 = 0, nk = 0, H = 0; rows.forEach(function (r) { m2 += r.keep * r.base; nk += r.keep * r.mult; });
    rows.forEach(function (r) { r.final = r.keep * r.base / m2; if (r.final > 0) H -= r.final * Math.log(r.final / (r.keep * r.mult)); });
    return { rows: rows, Z: Z, entropy: H, nkept: nk, keptMass: m2, keptAfterK: keptAfterK };
  }
  function pickIndex(rows, u) { var c = 0; for (var i = 0; i < rows.length; i++) { c += rows[i].final; if (u < c) return i; } for (i = rows.length - 1; i >= 0; i--) if (rows[i].final > 0) return i; return 0; }
  function softmaxPlain(z, T) { var m = Math.max.apply(null, z), e = z.map(function (x) { return Math.exp((x - m) / T); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (x) { return x / s; }); }
  if (typeof module !== "undefined" && module.exports) { module.exports = { buildItems: buildItems, distribution: distribution, pickIndex: pickIndex, softmaxPlain: softmaxPlain }; }
  if (typeof document === "undefined") return;

  // ---------- helpers ----------
  var CTX = null, WIN = null, LOSS = null, TINY = null;
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function showTok(t) { t = t.replace(/\n/g, "\\n"); return t.charAt(0) === " " ? "▁" + t.slice(1) : (t === "" ? "(empty)" : t); }
  function pc(x) { return x >= 0.1 ? (100 * x).toFixed(0) + "%" : x >= 0.001 ? (100 * x).toFixed(1) + "%" : x > 0 ? "<0.1%" : "0%"; }
  function nf(n) { return Number(n).toLocaleString("en-US"); }
  function box(val, cap, cls) { return '<span class="t ' + (cls || "") + '"><b>' + val + "</b><i>" + cap + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function f2(x) { return Kit.fmt(x, 2); }
  function f3(x) { return Kit.fmt(x, 3); }
  var SHORT = ["Capital of a state", "Fishermen cast their", "Message m02: put it", "Message m01: about", "Message m03: so we", "Message m04: two milk", "Message m13: Hinglish", "An open sentence"];
  var rngMain = null;

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/contexts.json"), Kit.data("data/window.json"), Kit.data("data/loss.json"), Kit.data("data/tiny.json")]).then(function (r) {
      CTX = r[0]; WIN = r[1]; LOSS = r[2]; TINY = r[3]; rngMain = Kit.rng(4127);
      nextFig("nx", false); nextFig("sm", true); loopFig(); windowFig(); trainFig(); lossFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- next-token bars (basic: scores and probabilities; full: temperature, top-k, top-p, Sample) ----------
  function nextFig(id, full) {
    var root = Kit.$(id), bar = Kit.$(id + "-bar"), ctl = Kit.$(id + "-controls"), act = Kit.$(id + "-act"), prompt = Kit.$(id + "-prompt"), rowsBox = Kit.$(id + "-rows"), read = Kit.$(id + "-read"), eq = Kit.$(id + "-eqn"), st = Kit.$(id + "-status"), hist = Kit.$(id + "-hist");
    var ci = id === "nx" ? 2 : 3, sel = 0, draws = null, last = [], items = null, T = 1, K = 0, P = 1;
    Kit.select(bar, "Context", CTX.contexts.map(function (c, i) { return { value: String(i), label: SHORT[i] }; }), String(ci), function (v) { ci = Number(v); draws = null; last = []; sel = 0; build(); draw(); });
    var sT, sK, sP;
    if (full) {
      sT = Kit.slider(ctl, { label: "Temperature T", min: 0.1, max: 2, step: 0.05, value: 1 }, function (v) { T = v; draws = null; draw(); });
      sK = Kit.slider(ctl, { label: "Top-k (0 means off)", min: 0, max: 100, step: 1, value: 0, digits: 0, show: function (v) { return v === 0 ? "off" : String(v); } }, function (v) { K = v; draws = null; draw(); });
      sP = Kit.slider(ctl, { label: "Top-p (1 means off)", min: 0.05, max: 1, step: 0.01, value: 1, show: function (v) { return v >= 1 ? "off" : v.toFixed(2); } }, function (v) { P = v; draws = null; draw(); });
      Kit.button(act, "Sample one token", function () { sampleOne(); });
      Kit.button(act, "Draw 200", function () { drawMany(200); });
      Kit.button(act, "Reset", function () { T = 1; K = 0; P = 1; sT.set(1); sK.set(0); sP.set(1); draws = null; last = []; draw(); });
      var presets = Kit.html("div", { class: "row-chips" }, act);
      Kit.chips(presets, ["Greedy (k = 1)", "Plain (T = 1)", "Top-p 0.9", "Hot (T = 1.6)"], 1, function (i) {
        if (i === 0) { T = 1; K = 1; P = 1; } else if (i === 1) { T = 1; K = 0; P = 1; } else if (i === 2) { T = 1; K = 0; P = 0.9; } else { T = 1.6; K = 0; P = 1; }
        sT.set(T); sK.set(K); sP.set(P); draws = null; draw();
      });
    }
    function build() { items = buildItems(CTX.contexts[ci]); }
    function sampleOne() {
      var d = distribution(items, T, K, P), i = pickIndex(d.rows, rngMain()), r = d.rows[i], tok;
      if (r.tok !== null) tok = r.tok; else tok = tailToken(CTX.contexts[ci], T, r.z);
      last.unshift({ tok: tok, p: r.final / Math.max(1, r.keep * r.mult), tail: r.tok === null }); last = last.slice(0, 10);
      draws = null; draw(true);
    }
    function tailToken(c, Tm, z) {
      var s = c.tail.sample, w = s.map(function (t) { return Math.exp((t[1] - z) / Tm); }), tot = w.reduce(function (a, b) { return a + b; }, 0), u = rngMain() * tot, a = 0;
      for (var i = 0; i < s.length; i++) { a += w[i]; if (u < a) return s[i][0]; }
      return s[s.length - 1][0];
    }
    function drawMany(n) {
      var d = distribution(items, T, K, P), cnt = {}, other = 0;
      for (var j = 0; j < n; j++) { var i = pickIndex(d.rows, rngMain()), r = d.rows[i]; if (r.tok !== null && i < 14) cnt[i] = (cnt[i] || 0) + 1; else other++; }
      draws = { n: n, cnt: cnt, other: other }; draw();
    }
    function draw(flash) {
      var c = CTX.contexts[ci], d = distribution(items, T, K, P), rows = d.rows, shown = 14, h = "", maxv = 0, j;
      for (j = 0; j < shown; j++) maxv = Math.max(maxv, rows[j].base, rows[j].final);
      var restBase = 0, restFinal = 0; for (j = shown; j < rows.length; j++) { restBase += rows[j].base; restFinal += rows[j].final; }
      var scale = maxv || 1, overB = restBase > scale, overF = restFinal > scale;
      if (sel >= shown) sel = 0;
      prompt.innerHTML = esc(c.text) + '<span class="nxt">' + (last.length ? esc(last[0].tok.replace(/\n/g, "\\n")) : (c.true ? "" : "")) + "</span>";
      for (j = 0; j < shown; j++) {
        var r = rows[j], isTrue = c.true && r.tok === c.true.tok, cut = r.keep === 0, shownFinal = full ? r.final : r.base;
        h += '<div class="nx-row' + (j === sel ? " on" : "") + (cut ? " cut" : "") + (isTrue ? " tr" : "") + '" data-j="' + j + '" tabindex="0" role="button" aria-label="Show the working for ' + esc(showTok(r.tok)) + '">' +
          '<span class="nx-tok">' + esc(showTok(r.tok)) + (isTrue ? ' <em>true</em>' : "") + '</span><span class="nx-z">' + Kit.fmt(r.z, 2) + '</span><span class="nx-track"><span class="nx-base" style="width:' + (100 * r.base / scale).toFixed(2) + '%"></span>' +
          (full ? '<span class="nx-fin" style="width:' + (100 * r.final / scale).toFixed(2) + '%"></span>' : "") + (draws && draws.cnt[j] ? '<span class="nx-draw" style="width:' + (100 * draws.cnt[j] / draws.n / scale).toFixed(2) + '%"></span>' : "") + '</span><b>' + pc(full ? r.final : r.base) + "</b></div>";
      }
      h += '<div class="nx-row rest' + (restFinal === 0 && full ? " cut" : "") + '"><span class="nx-tok">all other ' + nf(rows.slice(shown).reduce(function (a, r) { return a + r.mult; }, 0)) + ' tokens</span><span class="nx-z"></span><span class="nx-track"><span class="nx-base' + (overB ? " over" : "") + '" style="width:' + (100 * Math.min(restBase, scale) / scale).toFixed(2) + '%"></span>' +
        (full ? '<span class="nx-fin' + (overF ? " over" : "") + '" style="width:' + (100 * Math.min(restFinal, scale) / scale).toFixed(2) + '%"></span>' : "") + (draws && draws.other ? '<span class="nx-draw" style="width:' + (100 * Math.min(draws.other / draws.n, scale) / scale).toFixed(2) + '%"></span>' : "") + "</span><b>" + pc(full ? restFinal : restBase) + "</b></div>";
      rowsBox.innerHTML = h;
      rowsBox.querySelectorAll(".nx-row[data-j]").forEach(function (n) { n.addEventListener("click", function () { sel = Number(n.dataset.j); draw(); }); n.addEventListener("keydown", function (e) { if (e.key === "Enter") { sel = Number(n.dataset.j); draw(); } }); });
      // readouts
      var top1 = full ? Math.max.apply(null, rows.map(function (r) { return r.final; })) : rows[0].base, tr = c.true;
      var trueRow = null; if (tr) rows.forEach(function (r) { if (r.tok === tr.tok) trueRow = r; });
      read.innerHTML = '<span>Entropy of the choice: <b>' + d.entropy.toFixed(2) + " nats</b></span><span>Effective number of choices (e<sup>entropy</sup>): <b>" + (Math.exp(d.entropy) < 100 ? Math.exp(d.entropy).toFixed(1) : nf(Math.round(Math.exp(d.entropy)))) + "</b></span><span>Top token: <b>" + pc(top1) + "</b></span>" +
        (full ? "<span>Tokens kept: <b>" + nf(Math.round(d.nkept)) + "</b></span>" : "") + (tr ? "<span>True next token " + esc(showTok(tr.tok)) + ": <b>" + (trueRow ? pc(full ? trueRow.final : trueRow.base) : "not shown") + "</b>, rank " + nf(tr.rank) + "</span>" : "");
      // working for the selected row
      var s = rows[sel], e = Math.exp(s.z / T);
      eq.innerHTML = box(esc(showTok(s.tok)), "the chosen token", "c-sand") + op(":") + box(f2(s.z) + " &divide; " + T.toFixed(2) + " = " + f2(s.z / T), "score divided by T", "c-navy") + op("&rarr;") + box(e.toFixed(3), "e to that power", "c-navy") + op("&divide;") + box(d.Z.toFixed(2), "sum over all " + nf(50257) + " tokens", "c-sand") + op("=") + box(pc(s.base), "probability p", "c-teal");
      eq.setAttribute("aria-label", "The probability of " + showTok(s.tok) + " is e to the power of its score divided by the temperature, divided by the sum over all tokens, which is " + pc(s.base));
      if (hist) {
        hist.innerHTML = last.length ? '<span class="side-h">Draws, newest first:</span> ' + last.map(function (l, i) { return '<span class="tk ' + (l.tail ? "unk" : "c" + (i % 3)) + '">' + esc(showTok(l.tok)) + "</span>"; }).join("") : '<span class="vnone">Press Sample one token.</span>';
      }
      var parts = [];
      if (full) {
        parts.push("Pale bars show the probability at this temperature; solid bars show it after top-k and top-p cuts and renormalising. A striped bar is longer than the box.");
        if (K > 0 || P < 1) parts.push("The cuts keep " + nf(Math.round(d.nkept)) + " tokens.");
        if (draws) parts.push("Outlines show how often 200 draws picked the tokens.");
      } else parts.push("Left numbers are the scores z, relative to the top one. Bars are probabilities from softmax. Click a row for its working.");
      st.textContent = parts.join(" ");
      if (full) cumPlot(d);
    }
    function cumPlot(d) {
      var n = 60, xs = [], cum = [], a = 0; for (var j = 0; j < n; j++) { a += d.rows[j].base; xs.push(j + 1); cum.push(a * 100); }
      var shapes = [];
      if (P < 1) shapes.push({ type: "line", x0: 1, x1: n, y0: P * 100, y1: P * 100, line: { color: "#c2563a", width: 1.5, dash: "dot" } });
      if (K > 0 && K <= n) shapes.push({ type: "line", x0: K + 0.5, x1: K + 0.5, y0: 0, y1: 100, line: { color: "#3b4f82", width: 1.5, dash: "dot" } });
      Kit.plot(id + "-cum", function (Pl) { return [{ type: "scatter", mode: "lines+markers", x: xs, y: cum, line: { color: Pl.teal, width: 2 }, marker: { size: 4, color: Pl.teal }, hovertemplate: "top %{x} tokens have %{y:.1f}% of the probability<extra></extra>" }]; },
        { margin: { l: 46, r: 10, t: 6, b: 44 }, xaxis: { title: { text: "tokens, most likely first", standoff: 8 } }, yaxis: { title: { text: "probability so far, %" }, range: [0, 102] }, shapes: shapes });
    }
    build(); draw();
    Kit.onTheme(function () { draw(); });
  }

  // ---------- the generation loop with a tiny word model ----------
  function loopFig() {
    var bar = Kit.$("lp-bar"), ctl = Kit.$("lp-controls"), act = Kit.$("lp-act"), txt = Kit.$("lp-text"), rowsBox = Kit.$("lp-rows"), read = Kit.$("lp-read"), st = Kit.$("lp-status"), logbox = Kit.$("lp-log");
    var prompts = [["(start of a report)", []], ["we found a", ["we", "found", "a"]], ["the water at", ["the", "water", "at"]], ["our gillnet caught", ["our", "gillnet", "caught"]]];
    var pi = 0, words = [], lps = [], ended = false, T = 1, K = 0, P = 1, timer = 0, rng = Kit.rng(99), lastPick = null, lastCtx = null, MAXW = 40;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), prompts.map(function (p) { return p[0]; }), 0, function (i) { pi = i; reset(); });
    var sT = Kit.slider(ctl, { label: "Temperature T", min: 0.1, max: 2, step: 0.05, value: 1 }, function (v) { T = v; draw(); });
    var sK = Kit.slider(ctl, { label: "Top-k (0 means off)", min: 0, max: 20, step: 1, value: 0, digits: 0, show: function (v) { return v === 0 ? "off" : String(v); } }, function (v) { K = v; draw(); });
    var sP = Kit.slider(ctl, { label: "Top-p (1 means off)", min: 0.05, max: 1, step: 0.01, value: 1, show: function (v) { return v >= 1 ? "off" : v.toFixed(2); } }, function (v) { P = v; draw(); });
    Kit.button(act, "Next token", function () { stop(); step(); });
    var runBtn = Kit.button(act, "Run", function () { if (timer) stop(); else run(); });
    Kit.button(act, "Reset", function () { stop(); reset(); });
    var seedBtn = Kit.button(act, "New random seed", function () { rng = Kit.rng(Math.floor(Math.random() * 1e9)); stop(); reset(); });
    function stop() { if (timer) { clearInterval(timer); timer = 0; runBtn.textContent = "Run"; } }
    function run() { if (ended) reset(); runBtn.textContent = "Pause"; timer = setInterval(function () { if (ended) { stop(); return; } step(); }, 650); }
    function ctxKey() { var w = ["<s>", "<s>"].concat(words); return w[w.length - 2] + "|" + w[w.length - 1]; }
    function lookup(key) {
      var c = TINY.contexts[key]; if (c) return c;
      var last = key.split("|")[1], agg = {}; Object.keys(TINY.contexts).forEach(function (k) { if (k.split("|")[1] === last) TINY.contexts[k].forEach(function (x) { agg[x[0]] = (agg[x[0]] || 0) + x[1]; }); });
      var out = Object.keys(agg).map(function (w) { return [w, agg[w]]; }).sort(function (a, b) { return b[1] - a[1]; }); return out.length ? out : [["</s>", 1]];
    }
    function dist(key) {
      var c = lookup(key), mx = c[0][1], items = c.map(function (x) { return { tok: x[0], z: Math.log(x[1] / mx), mult: 1, n: x[1] }; });
      return { c: c, d: distribution(items, T, K, P), items: items, total: c.reduce(function (a, x) { return a + x[1]; }, 0) };
    }
    function reset() { words = prompts[pi][1].slice(); lps = []; ended = false; lastPick = null; lastCtx = null; draw(); }
    function step() {
      if (ended) return;
      var key = ctxKey(), D = dist(key), i = pickIndex(D.d.rows, rng()), tok = D.items[i].tok;
      lastPick = i; lastCtx = key; lps.push({ w: tok, lp: Math.log(D.items[i].n / D.total) });
      if (tok === "</s>" || words.length >= MAXW) { ended = true; } else words.push(tok);
      draw();
    }
    function draw() {
      var stepped = lastCtx !== null, key = stepped ? lastCtx : ctxKey(), D = dist(key), rows = D.d.rows, shown = Math.min(8, rows.length), h = "", mx = 0, j, len = words.length;
      for (j = 0; j < shown; j++) mx = Math.max(mx, rows[j].base, rows[j].final);
      var parts = key.split("|"), ctxWords = parts.filter(function (w) { return w !== "<s>"; });
      var w0 = stepped && !ended ? len - 3 : len - 2, w1 = w0 + 1, html = [];
      words.forEach(function (w, i) { var c = (i === w0 || i === w1) ? "win" : ""; if (stepped && !ended && i === len - 1) c += " newword"; html.push(c ? '<span class="' + c + '">' + esc(w) + "</span>" : esc(w)); });
      txt.innerHTML = (len ? html.join(" ") : '<span class="vnone">(empty so far: the model sees only the start marker)</span>') + (ended ? ' <span class="endtok">[end]</span>' : "");
      for (j = 0; j < shown; j++) {
        var r = rows[j];
        h += '<div class="nx-row' + (r.keep === 0 ? " cut" : "") + (stepped && lastPick === j ? " tr" : "") + '"><span class="nx-tok">' + esc(r.tok === "</s>" ? "[end]" : r.tok) + '</span><span class="nx-z">' + D.c[j][1] + '</span><span class="nx-track"><span class="nx-base" style="width:' + (100 * r.base / mx).toFixed(1) + '%"></span><span class="nx-fin" style="width:' + (100 * r.final / mx).toFixed(1) + '%"></span></span><b>' + pc(r.final) + "</b></div>";
      }
      if (rows.length > shown) h += '<div class="nx-row rest"><span class="nx-tok">' + (rows.length - shown) + ' more words</span><span class="nx-z"></span><span class="nx-track"></span><b>' + pc(rows.slice(shown).reduce(function (a, r) { return a + r.final; }, 0)) + "</b></div>";
      rowsBox.innerHTML = h;
      Kit.setText("lp-ctx", ctxWords.length ? ctxWords.join(" ") : "(start)");
      var tl = lps.reduce(function (a, x) { return a + x.lp; }, 0);
      logbox.innerHTML = lps.length ? lps.slice(-10).map(function (x) { return '<span class="tk c0">' + esc(x.w === "</s>" ? "[end]" : x.w) + "<i>" + Kit.fmt(x.lp, 2) + "</i></span>"; }).join("") : '<span class="vnone">No tokens yet.</span>';
      read.innerHTML = "<span>Tokens made: <b>" + lps.length + "</b></span><span>Sum of ln p: <b>" + Kit.fmt(tl, 2) + "</b></span><span>Mean loss per token: <b>" + (lps.length ? (-tl / lps.length).toFixed(2) : "none yet") + "</b></span><span>Perplexity of the text: <b>" + (lps.length ? Math.exp(-tl / lps.length).toFixed(1) : "none yet") + "</b></span>";
      st.textContent = ended ? "The model chose the end marker, so the loop stops. Press Reset to start again." : stepped ? "The bars show the distribution the model used for the new coral word. The underlined words were its whole context. Press Next token to use the new last two words." : "The model sees only the last two words, its whole context window. Pale bars: counts turned into probabilities at this temperature. Solid bars: after the cuts.";
    }
    reset();
  }

  // ---------- the context window ----------
  function windowFig() {
    var toks = WIN.tokens, n = toks.length, w = n, chart = "win-plot", tbox = Kit.$("win-toks"), read = Kit.$("win-read"), st = Kit.$("win-status");
    var sl = Kit.slider(Kit.$("win-controls"), { label: "Tokens the model can see (the last w)", min: 1, max: n, step: 1, value: n, digits: 0 }, function (v) { w = v; draw(); });
    Kit.button(Kit.$("win-bar"), "Show all 46", function () { w = n; sl.set(n); draw(); });
    Kit.button(Kit.$("win-bar"), "Cut off the tag", function () { w = n - 5; sl.set(w); draw(); });
    function draw() {
      var h = "", row = WIN.rows[w - 1];
      toks.forEach(function (t, i) { var inw = i >= n - w; h += '<span class="tk ' + (inw ? "c" + (i % 3) : "gone") + '">' + esc(showTok(t)) + "</span>"; });
      tbox.innerHTML = h;
      read.innerHTML = "<span>Tokens seen: <b>" + w + " of " + n + "</b></span><span>Probability of &ldquo;&#9601;4&rdquo;, the start of 4127: <b>" + pc(row[1]) + "</b></span><span>Most likely next token: <b>" + esc(showTok(row[2])) + " (" + pc(row[3]) + ")</b></span>";
      st.textContent = w >= n - 4 ? "The model can see the tag, so \u201C\u25814\u201D becomes likely." : "The tag number is outside the window. The model cannot use what it cannot see, so it guesses: its top choice is " + showTok(row[2]) + ".";
      var xs = WIN.rows.map(function (r) { return r[0]; }), ys = WIN.rows.map(function (r) { return r[1] * 100; });
      Kit.plot(chart, function (P) { return [{ type: "scatter", mode: "lines+markers", x: xs, y: ys, line: { color: P.navy, width: 2 }, marker: { size: 5, color: P.navy }, hovertemplate: "%{x} tokens seen: %{y:.1f}%<extra></extra>" },
        { type: "scatter", mode: "markers", x: [w], y: [ys[w - 1]], marker: { size: 14, color: P.coral, line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip" }]; },
        { margin: { l: 50, r: 10, t: 8, b: 48 }, xaxis: { title: { text: "tokens the model can see (w)", standoff: 8 }, range: [0, n + 1] }, yaxis: { title: { text: "probability of ▁ 4, %" }, range: [0, 48] }, shapes: [{ type: "line", x0: n - 4 + 0.5, x1: n - 4 + 0.5, y0: 0, y1: 48, line: { color: "#8f98a2", width: 1.5, dash: "dot" } }], annotations: [{ x: n - 4 + 0.5, y: 46, text: "window reaches \u2581Tag", showarrow: false, xanchor: "right", font: { size: 11 } }] });
    }
    draw();
    Kit.onTheme(draw);
  }

  // ---------- one training step on one example ----------
  function trainFig() {
    var bar = Kit.$("tn-bar"), ctl = Kit.$("tn-controls"), act = Kit.$("tn-act"), rowsBox = Kit.$("tn-rows"), read = Kit.$("tn-read"), eq = Kit.$("tn-eqn"), st = Kit.$("tn-status");
    var withTrue = []; CTX.contexts.forEach(function (c, i) { if (c.true && c.true.rank <= 300) withTrue.push(i); });
    var ci = withTrue[2], eta = 1, items = null, steps = 0, hist = [], trueIdx = 0;
    Kit.select(bar, "Context", withTrue.map(function (i) { return { value: String(i), label: SHORT[i] }; }), String(ci), function (v) { ci = Number(v); reset(); });
    var sE = Kit.slider(ctl, { label: "Step size η", min: 0.1, max: 3, step: 0.1, value: 1 }, function (v) { eta = v; draw(); });
    Kit.button(act, "One training step", function () { upd(1); });
    Kit.button(act, "Ten steps", function () { upd(10); });
    Kit.button(act, "Reset", function () { reset(); });
    function reset() {
      var c = CTX.contexts[ci]; items = buildItems(c).map(function (x) { return { tok: x.tok, z: x.z, mult: x.mult }; });
      trueIdx = 0; items.forEach(function (x, i) { if (x.tok === c.true.tok) trueIdx = i; });
      steps = 0; hist = [ptrue()]; draw();
    }
    function probs() { var Z = 0; items.forEach(function (x) { x.e = Math.exp(x.z); Z += x.mult * x.e; }); return { Z: Z }; }
    function ptrue() { var d = probs(); return items[trueIdx].e / d.Z; }
    function upd(n) {
      for (var s = 0; s < n; s++) { var d = probs(); items.forEach(function (x, i) { var p = x.e / d.Z, g = p - (i === trueIdx ? 1 : 0); x.z -= eta * g; }); steps++; hist.push(ptrue()); }
      draw();
    }
    function draw() {
      var c = CTX.contexts[ci], d = probs(), ptr = items[trueIdx].e / d.Z, shown = Math.min(9, items.length), h = "", j, maxv = 0, restMass = 0;
      var order = items.map(function (x, i) { return i; }).slice(0, 300).sort(function (a, b) { return items[b].z - items[a].z; }).slice(0, shown);
      if (order.indexOf(trueIdx) < 0) order[order.length - 1] = trueIdx;
      order.forEach(function (i) { maxv = Math.max(maxv, items[i].e / d.Z); });
      var shownMass = 0;
      order.forEach(function (i) {
        var x = items[i], p = x.e / d.Z; shownMass += p;
        h += '<div class="nx-row' + (i === trueIdx ? " tr" : "") + '"><span class="nx-tok">' + esc(showTok(x.tok)) + (i === trueIdx ? " <em>true</em>" : "") + '</span><span class="nx-z">' + Kit.fmt(x.z, 2) + '</span><span class="nx-track"><span class="nx-fin" style="width:' + (100 * p / Math.max(maxv, 0.02)).toFixed(2) + '%"></span></span><b>' + pc(p) + "</b></div>";
      });
      rowsBox.innerHTML = h + '<div class="nx-row rest"><span class="nx-tok">all other tokens</span><span class="nx-z"></span><span class="nx-track"><span class="nx-base" style="width:' + Math.min(100, 100 * Math.max(0, 1 - shownMass) / Math.max(maxv, 0.02)).toFixed(2) + '%"></span></span><b>' + pc(Math.max(0, 1 - shownMass)) + "</b></div>";
      var loss = -Math.log(ptr), top = order.filter(function (i) { return i !== trueIdx; })[0], pt = items[top].e / d.Z, zt = items[trueIdx].z;
      read.innerHTML = "<span>Steps taken: <b>" + steps + "</b></span><span>Probability of the true token: <b>" + pc(ptr) + "</b></span><span>Loss &minus;ln p: <b>" + loss.toFixed(3) + "</b></span><span>Starting loss: <b>" + (-Math.log(hist[0])).toFixed(3) + "</b></span>";
      eq.innerHTML = box(esc(showTok(c.true.tok)), "the true token", "c-teal") + op(":") + box("p &minus; 1 = " + Kit.fmt(ptr - 1, 3), "slope of the loss at its score", "c-coral") + op("&rarr;") + box("z = " + Kit.fmt(zt, 2) + " &minus; " + eta.toFixed(1) + " &times; (" + Kit.fmt(ptr - 1, 3) + ") = " + Kit.fmt(zt - eta * (ptr - 1), 2), "next score", "c-teal") +
        op("|") + box(esc(showTok(items[top].tok)), "the strongest rival", "c-sand") + op(":") + box("p &minus; 0 = " + Kit.fmt(pt, 3), "slope at its score", "c-coral") + op("&rarr;") + box("z = " + Kit.fmt(items[top].z, 2) + " &minus; " + eta.toFixed(1) + " &times; " + Kit.fmt(pt, 3) + " = " + Kit.fmt(items[top].z - eta * pt, 2), "next score", "c-navy");
      eq.setAttribute("aria-label", "The slope of the loss at the true token's score is p minus 1, and at a wrong token's score it is p");
      st.textContent = "All scores move against their slopes: the true token's score goes up, and the other scores go down in proportion to their probability. In a full network the scores move only through the weights, so one step also changes other contexts.";
      var xs = hist.map(function (_, i) { return i; }), ys = hist.map(function (p) { return -Math.log(p); });
      Kit.plot("tn-loss", function (P) { return [{ type: "scatter", mode: "lines+markers", x: xs, y: ys, line: { color: P.coral, width: 2 }, marker: { size: 5, color: P.coral }, hovertemplate: "step %{x}: loss %{y:.3f}<extra></extra>" }]; },
        { margin: { l: 50, r: 10, t: 8, b: 44 }, xaxis: { title: { text: "training steps", standoff: 8 }, range: [0, Math.max(10, steps + 1)] }, yaxis: { title: { text: "loss, nats" }, rangemode: "tozero" } });
    }
    reset();
    Kit.onTheme(draw);
  }

  // ---------- per-token loss and perplexity ----------
  function lossFig() {
    var M = LOSS.messages, sel = 1, metric = 0, bar = Kit.$("ls-bar"), tbox = Kit.$("ls-toks"), read = Kit.$("ls-read"), eq = Kit.$("ls-eqn"), st = Kit.$("ls-status");
    Kit.select(bar, "Message", M.map(function (m, i) { return { value: String(i), label: m.id + " " + m.language }; }), String(sel), function (v) { sel = Number(v); draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Perplexity per token", "Total loss, nats", "Bits per character"], 0, function (i) { metric = i; draw(); });
    function draw() {
      Kit.$("ls-bar").querySelector("select").value = String(sel);
      var m = M[sel], h = "", tot = 0;
      m.tokens.forEach(function (t) { tot += t[1]; var a = Math.min(1, t[1] / 9); h += '<span class="tk lossc" title="loss ' + t[1].toFixed(2) + ' nats" style="background:color-mix(in srgb, var(--coral) ' + Math.round(6 + 80 * a) + '%, transparent)">' + esc(showTok(t[0])) + "<i>" + t[1].toFixed(1) + "</i></span>"; });
      tbox.innerHTML = h;
      var n = m.tokens.length, L = tot / n;
      read.innerHTML = "<span>Tokens: <b>" + n + "</b></span><span>Characters: <b>" + m.chars + "</b></span><span>Mean loss L: <b>" + L.toFixed(2) + " nats</b></span><span>Perplexity e<sup>L</sup>: <b>" + Math.exp(L).toFixed(1) + "</b></span><span>Total loss: <b>" + tot.toFixed(0) + " nats</b></span>";
      eq.innerHTML = box(tot.toFixed(0), "total of &minus;ln p over " + n + " tokens", "c-coral") + op("&divide;") + box(n, "tokens n", "c-navy") + op("=") + box(L.toFixed(2), "mean loss L", "c-sand") + op("&rarr;") + box("e<sup>" + L.toFixed(2) + "</sup> = " + Math.exp(L).toFixed(1), "perplexity", "c-teal");
      eq.setAttribute("aria-label", "Total loss " + tot.toFixed(0) + " nats over " + n + " tokens gives a mean loss of " + L.toFixed(2) + " and a perplexity of " + Math.exp(L).toFixed(1));
      st.textContent = "Tokens show their loss in nats under them, and darker means more surprised. All messages are invented. The Hindi, Marathi and Hinglish texts have not yet been checked by a native speaker.";
      var langs = ["English", "Hindi", "Marathi", "Hinglish"], names = ["Perplexity per token", "Total loss, nats", "Bits per character"];
      function val(x) { var t = x.tokens.reduce(function (a, b) { return a + b[1]; }, 0); return metric === 0 ? Math.exp(t / x.tokens.length) : metric === 1 ? t : t / Math.LN2 / x.chars; }
      Kit.plot("ls-plot", function (P) {
        return langs.map(function (L, li) {
          var ms = M.filter(function (x) { return x.language === L; });
          return { type: "bar", name: L, x: ms.map(function (x) { return x.id; }), y: ms.map(val), text: ms.map(function (x) { var v = val(x); return v >= 100 ? v.toFixed(0) : v.toFixed(1); }), textposition: "outside", cliponaxis: false,
            marker: { color: [P.navy, P.teal, P.coral, P.sand][li], line: { color: ms.map(function (x) { return x.id === m.id ? P.ink : "rgba(0,0,0,0)"; }), width: 2 }, opacity: ms.map(function (x) { return x.id === m.id ? 1 : 0.75; }) }, hovertemplate: "%{x} " + L + ": %{y:.1f}<extra></extra>" };
        });
      }, { barmode: "overlay", margin: { l: 56, r: 10, t: 30, b: 44 }, yaxis: { title: { text: names[metric] }, type: metric === 0 ? "log" : "linear", rangemode: metric === 0 ? "normal" : "tozero", tickvals: metric === 0 ? [5, 10, 20, 50, 100, 200, 500, 1000, 2000] : undefined, ticktext: metric === 0 ? ["5", "10", "20", "50", "100", "200", "500", "1,000", "2,000"] : undefined }, xaxis: { title: { text: "message", standoff: 6 } }, showlegend: true, legend: { orientation: "h", y: 1.16, x: 0 } });
    }
    draw();
    Kit.onTheme(draw);
  }
})();
