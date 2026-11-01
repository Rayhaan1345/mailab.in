// D2 Embeddings. Figures: one-hot against table lookup, "company" counts and cosine, a map of 50-number GloVe words,
// an analogy explorer, and a map of the 16 invented messages in a multilingual sentence embedding.
// The pure vector helpers are exported with module.exports for node tests.
(function () {
  "use strict";
  function dot(a, b) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  function norm(a) { return Math.sqrt(dot(a, a)); }
  function cosine(a, b) { return dot(a, b) / (norm(a) * norm(b)); }
  function add(a, b, c) { return a.map(function (x, i) { return x - b[i] + c[i]; }); } // a - b + c
  // Nearest words to a query vector among rows {w, v}; skip the words in `skip`.
  function nearest(q, rows, skip, n) {
    var out = [];
    rows.forEach(function (r) { if (skip && skip.indexOf(r.w) >= 0) return; out.push({ w: r.w, c: cosine(q, r.v) }); });
    out.sort(function (x, y) { return y.c - x.c; });
    return out.slice(0, n);
  }
  if (typeof module !== "undefined" && module.exports) { module.exports = { dot: dot, norm: norm, cosine: cosine, add: add, nearest: nearest }; }
  if (typeof document === "undefined") return;

  var W = null, SN = null;
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function f2(x) { return Kit.fmt(x, 2); }
  function nf(n) { return Number(n).toLocaleString("en-US"); }
  function hookClick(div, fn) { var tries = 0; (function go() { if (div.on) { div.on("plotly_click", fn); } else if (tries++ < 30) setTimeout(go, 150); })(); }
  function box(val, cap, cls) { return '<span class="t ' + (cls || "") + '"><b>' + val + "</b><i>" + cap + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/words.json"), Kit.data("data/sentences.json")]).then(function (r) {
      W = r[0]; SN = r[1];
      onehotFig(); companyFig(); mapFig(); analogyFig(); sentFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. one-hot against lookup ----------
  var TOY = ["shark", "whale", "dolphin", "boat", "net", "dock", "camera", "tide"];
  var TOYE = [[0.9, -0.3, 0.7, 0.1], [0.8, -0.2, 0.9, 0.0], [0.7, -0.4, 0.8, 0.2], [-0.5, 0.8, 0.1, 0.6], [-0.4, 0.6, -0.2, 0.7], [-0.6, 0.9, 0.2, 0.4], [0.1, 0.2, -0.9, -0.5], [-0.2, 0.3, 0.6, -0.7]];
  function onehotFig() {
    var bar = Kit.$("oh-bar"), grid = Kit.$("oh-grid"), eq = Kit.$("oh-eqn"), read = Kit.$("oh-read"), st = Kit.$("oh-status");
    var E = TOYE.map(function (r) { return r.slice(); }), sel = 2, col = 0, mode = 0, size = 0, nudged = false, G = [0.4, -0.2, 0.1, 0.3], ETA = 0.5;
    var cw = Kit.html("div", { class: "row-chips" }, bar);
    Kit.chips(cw, TOY, sel, function (i) { sel = i; draw(); });
    var b2 = Kit.html("div", { class: "row-chips" }, bar);
    Kit.chips(b2, ["Multiply by the one-hot vector", "Look up the row"], 0, function (i) { mode = i; draw(); });
    var b3 = Kit.html("div", { class: "row-chips" }, bar);
    Kit.chips(b3, ["Toy table, 8 words", "GPT-2, 50,257 tokens"], 0, function (i) { size = i; draw(); });
    Kit.button(bar, "Training nudge to this row", function () { if (!nudged) { for (var j = 0; j < 4; j++) E[sel][j] = Math.round((E[sel][j] - ETA * G[j]) * 100) / 100; nudged = true; nudgedRow = sel; draw(); } });
    Kit.button(bar, "Reset", function () { E = TOYE.map(function (r) { return r.slice(); }); nudged = false; nudgedRow = -1; draw(); });
    var nudgedRow = -1;
    function draw() {
      var h = '<div class="oh-col"><div class="oh-h">one-hot vector</div>';
      TOY.forEach(function (w, i) { h += '<div class="oh-c' + (i === sel ? " on" : "") + '"><span>' + (i === sel ? 1 : 0) + "</span><em>" + w + "</em></div>"; });
      h += '</div><div class="oh-times">&times;</div><div class="oh-col wide"><div class="oh-h">table E (8 rows, 4 numbers per row)</div>';
      TOY.forEach(function (w, i) {
        h += '<div class="oh-r' + (i === sel ? " on" : "") + (i === nudgedRow ? " nud" : "") + '">';
        for (var j = 0; j < 4; j++) h += "<span" + (mode === 0 && j === col && i === sel ? ' class="x"' : "") + ">" + Kit.fmt(E[i][j], 2) + "</span>";
        h += "</div>";
      });
      h += '</div><div class="oh-times">=</div><div class="oh-col"><div class="oh-h">embedding of ' + TOY[sel] + '</div><div class="oh-r on res">';
      for (var j = 0; j < 4; j++) h += '<span class="' + (j === col ? "x" : "") + '" data-j="' + j + '" tabindex="0" role="button" aria-label="Show the working for number ' + (j + 1) + '">' + Kit.fmt(E[sel][j], 2) + "</span>";
      h += "</div></div>";
      grid.innerHTML = h;
      grid.querySelectorAll(".res span").forEach(function (s) { s.addEventListener("click", function () { col = Number(s.dataset.j); draw(); }); });
      // the working for one number
      var terms = TOY.map(function (w, i) { return (i === sel ? 1 : 0) + " &times; " + Kit.fmt(E[i][col], 2); }).join(" + ");
      var e = mode === 0 ? box(terms, "number " + (col + 1) + " of the result: eight products, seven of them 0", "c-navy") + op("=") + box(Kit.fmt(E[sel][col], 2), "the table entry E[" + sel + "][" + col + "]", "c-teal")
        : box("E[" + sel + "]", "row " + sel + " of the table", "c-navy") + op("=") + box("(" + E[sel].map(function (x) { return Kit.fmt(x, 2); }).join(", ") + ")", "copied, no arithmetic", "c-teal");
      eq.innerHTML = e;
      var V = size ? 50257 : 8, d = size ? 768 : 4;
      read.innerHTML = "<span>Table size: <b>" + nf(V) + " &times; " + nf(d) + " = " + nf(V * d) + " numbers</b></span><span>Multiplying: <b>" + nf(V * d) + " multiplications</b></span><span>Looking up: <b>" + nf(d) + " copies</b></span>";
      st.textContent = nudged ? "Only the row of " + TOY[nudgedRow] + " changed. The other rows were not used, so they got no slope." : "Click a number of the result to see where it comes from. Press the training nudge to see which rows learning touches.";
    }
    draw();
  }

  // ---------- 2. company: co-occurrence counts and cosine ----------
  var COOC = {
    targets: ["shark", "whale", "dolphin", "boat", "trawler", "camera", "lens", "rain", "monsoon", "tide"],
    ctx: ["sea", "fin", "reef", "water", "swam", "boat", "dock", "net", "harbour", "tide", "camera", "lens", "rain", "wind", "big"],
    sentences: ["the shark swam past the boat in the sea", "a big shark has sharp teeth and a fin", "the shark circled the reef in the warm sea", "the whale swam in the deep sea",
      "a big whale has a large fin and blows water", "the whale rose near the reef in the sea", "the dolphin swam near the boat in the sea", "a dolphin has a fin and swims fast in the sea",
      "the dolphin played near the reef in the warm water", "the trawler left the dock at dawn with a net", "the boat returned to the dock with a full net",
      "the crew pulled the net onto the boat at the harbour", "the trawler and the boat waited in the harbour for the tide", "the trawler tied up at the dock in the harbour",
      "the camera has a lens and films the shark in the sea", "the camera on the robot films the reef under the water", "the lens of the camera fogged in the rain",
      "the lens and the camera need a dry battery", "rain and wind came with the monsoon", "the monsoon rain kept the boat at the dock", "the monsoon wind pushed water over the dock",
      "the tide came in at the harbour", "the tide went out and the dock was dry", "the tide moved the water over the reef"]
  };
  function companyFig() {
    var C = COOC, T = C.targets, X = C.ctx, M = T.map(function (t) {
      return X.map(function (c) { var n = 0; C.sentences.forEach(function (s) { var ws = s.split(" "); if (ws.indexOf(t) >= 0 && ws.indexOf(c) >= 0 && c !== t) n++; }); return n; });
    });
    var a = 0, b = 1, tbl = Kit.$("co-table"), eq = Kit.$("co-eqn"), st = Kit.$("co-status"), bar = Kit.$("co-bar");
    var cos = T.map(function (_, i) { return T.map(function (_, j) { return cosine(M[i], M[j]); }); });
    Kit.select(bar, "Word A", T.map(function (t, i) { return { value: String(i), label: t }; }), String(a), function (v) { a = Number(v); draw(); });
    Kit.select(bar, "Word B", T.map(function (t, i) { return { value: String(i), label: t }; }), String(b), function (v) { b = Number(v); draw(); });
    Kit.button(bar, "Show the sentences", function () { var d = Kit.$("co-sent"); d.hidden = !d.hidden; });
    Kit.$("co-sent").innerHTML = "<ol>" + C.sentences.map(function (s) { return "<li>" + esc(s) + "</li>"; }).join("") + "</ol>";
    function draw() {
      var h = '<table class="ct"><thead><tr><th></th>' + X.map(function (c) { return "<th><span>" + c + "</span></th>"; }).join("") + "</tr></thead><tbody>";
      T.forEach(function (t, i) {
        h += '<tr class="' + (i === a ? "ra" : i === b ? "rb" : "") + '" data-i="' + i + '"><th>' + t + "</th>" + M[i].map(function (n) { return '<td class="' + (n ? "n" + Math.min(n, 3) : "") + '">' + (n || "") + "</td>"; }).join("") + "</tr>";
      });
      tbl.innerHTML = h + "</tbody></table>";
      tbl.querySelectorAll("tr[data-i]").forEach(function (r) { r.addEventListener("click", function () { var i = Number(r.dataset.i); if (i !== a) { b = a; a = i; } draw(); }); });
      var d = dot(M[a], M[b]), na = norm(M[a]), nb = norm(M[b]);
      eq.innerHTML = box(d, T[a] + " &middot; " + T[b], "c-navy") + op("&divide;") + box(f2(na) + " &times; " + f2(nb), "lengths of the two rows", "c-sand") + op("=") + box(f2(d / (na * nb)), "cosine similarity", "c-teal");
      eq.setAttribute("aria-label", "The cosine similarity of the count rows of " + T[a] + " and " + T[b] + " is " + f2(d / (na * nb)));
      var best = T.map(function (t, j) { return { t: t, c: cos[a][j], j: j }; }).filter(function (x) { return x.j !== a; }).sort(function (x, y) { return y.c - x.c; });
      st.innerHTML = "Most similar to <b>" + T[a] + "</b> by its company: " + best.slice(0, 3).map(function (x) { return x.t + " (" + f2(x.c) + ")"; }).join(", ") + ". Click a row to pick a word.";
      Kit.plot("co-heat", function (P) {
        return [{ type: "heatmap", z: cos, x: T, y: T, zmin: 0, zmax: 1, colorscale: [[0, P.paper], [1, P.teal]], showscale: false, xgap: 1, ygap: 1,
          hovertemplate: "%{y} and %{x}: %{z:.2f}<extra></extra>", text: cos.map(function (r) { return r.map(function (v) { return v.toFixed(2); }); }), texttemplate: "%{text}", textfont: { size: 10, color: P.ink } }];
      }, { margin: { l: 70, r: 8, t: 8, b: 70 }, xaxis: { tickangle: -40, side: "bottom", automargin: true }, yaxis: { autorange: "reversed", automargin: true }, shapes: [
        { type: "rect", xref: "x", yref: "y", x0: b - 0.5, x1: b + 0.5, y0: a - 0.5, y1: a + 0.5, line: { color: "#c2563a", width: 3 } }] });
    }
    draw();
    hookClick(Kit.$("co-heat"), function (e) { var p = e.points[0]; a = T.indexOf(p.y); b = T.indexOf(p.x); draw(); });
    Kit.onTheme(draw);
  }

  // ---------- 3. the map ----------
  function wordRows() { return W.words.map(function (x) { return { w: x.w, v: x.v, g: x.g, x: x }; }); }
  function mapFig() {
    var rows = wordRows(), idx = {}; rows.forEach(function (r, i) { idx[r.w] = i; });
    var sel = idx["shark"], cmp = idx["whale"], proj = "tsne", bar = Kit.$("mp-bar"), strip = Kit.$("mp-strip"), nnb = Kit.$("mp-nn"), nna = Kit.$("mp-nnall"), eq = Kit.$("mp-eqn"), st = Kit.$("mp-status");
    var names = rows.map(function (r) { return r.w; }).slice().sort();
    var opts = names.map(function (w) { return { value: w, label: w }; });
    var selA = Kit.select(bar, "Word", opts, rows[sel].w, function (v) { sel = idx[v]; draw(); });
    var selB = Kit.select(bar, "Compare with", opts, rows[cmp].w, function (v) { cmp = idx[v]; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["t-SNE map", "PCA map"], 0, function (i) { proj = i ? "pca" : "tsne"; draw(); });
    var symbols = ["circle", "square", "diamond", "triangle-up", "cross"];
    function draw() {
      selA.value = rows[sel].w; selB.value = rows[cmp].w;
      var near = nearest(rows[sel].v, rows, [rows[sel].w], 5), nearIdx = near.map(function (n) { return idx[n.w]; });
      Kit.plot("mp-plot", function (P) {
        var cols = [P.navy, P.coral, P.teal, P.sand, P.ink2], tr = [];
        W.groups.forEach(function (g, gi) {
          var m = rows.filter(function (r) { return r.g === g; });
          tr.push({ type: "scatter", mode: "markers+text", name: g, x: m.map(function (r) { return r.x[proj][0]; }), y: m.map(function (r) { return r.x[proj][1]; }), text: m.map(function (r) { return r.w; }), textposition: "top center",
            textfont: { size: 11, color: P.ink2 }, marker: { size: 8, symbol: symbols[gi], color: cols[gi], line: { width: 0 } }, customdata: m.map(function (r) { return r.w; }), hovertemplate: "%{text}<extra>" + g + "</extra>" });
        });
        var lx = [], ly = [], s = rows[sel].x[proj];
        nearIdx.forEach(function (i) { lx.push(s[0], rows[i].x[proj][0], null); ly.push(s[1], rows[i].x[proj][1], null); });
        tr.push({ type: "scatter", mode: "lines", x: lx, y: ly, line: { color: P.coral, width: 1.5, dash: "dot" }, hoverinfo: "skip", showlegend: false });
        tr.push({ type: "scatter", mode: "markers", x: [s[0]], y: [s[1]], marker: { size: 20, color: "rgba(0,0,0,0)", line: { color: P.coral, width: 3 } }, hoverinfo: "skip", showlegend: false });
        return tr;
      }, { margin: { l: 8, r: 8, t: 8, b: 8 }, xaxis: { visible: false, range: [-1.12, 1.12] }, yaxis: { visible: false, range: [-1.12, 1.12] }, showlegend: true, legend: { orientation: "h", y: -0.02, font: { size: 12 } }, hovermode: "closest", dragmode: false },
        { displayModeBar: true, modeBarButtons: [["zoomIn2d", "zoomOut2d", "resetScale2d"]], displaylogo: false });
      // the 50 numbers
      var v = rows[sel].v, h = "";
      v.forEach(function (x) { var a = Math.min(1, Math.abs(x) / 1.6); h += '<span title="' + x + '" style="background:color-mix(in srgb, var(' + (x >= 0 ? "--teal" : "--coral") + ") " + Math.round(8 + 85 * a) + '%, transparent)"></span>'; });
      strip.innerHTML = h;
      Kit.setText("mp-word", rows[sel].w);
      nnb.innerHTML = near.map(function (n) { return box(n.w, f2(n.c), idx[n.w] !== undefined ? "c-teal" : ""); }).join("");
      nna.innerHTML = rows[sel].x.nn.slice(0, 6).map(function (n) { return box(n[0], f2(n[1]), "c-sand"); }).join("");
      var a = rows[sel].v, b = rows[cmp].v, d = dot(a, b), na = norm(a), nb = norm(b);
      eq.innerHTML = box(f2(d), rows[sel].w + " &middot; " + rows[cmp].w, "c-navy") + op("&divide;") + box(f2(na), "length of " + rows[sel].w, "c-sand") + op("&times;") + box(f2(nb), "length of " + rows[cmp].w, "c-sand") + op("=") + box(f2(d / (na * nb)), "cosine similarity", "c-teal");
      eq.setAttribute("aria-label", "Cosine similarity of " + rows[sel].w + " and " + rows[cmp].w + " is " + f2(d / (na * nb)));
      var near2d = rows.map(function (r, i) { return { i: i, d: Math.hypot(r.x[proj][0] - rows[sel].x[proj][0], r.x[proj][1] - rows[sel].x[proj][1]) }; }).filter(function (o) { return o.i !== sel; }).sort(function (p, q) { return p.d - q.d; })[0];
      st.innerHTML = "Nearest on this map: <b>" + rows[near2d.i].w + "</b>. Nearest in 50 numbers: <b>" + near[0].w + "</b>. " + (near2d.i === idx[near[0].w] ? "They agree for this word." : "They differ, which shows that the map is a flattened view.");
    }
    draw();
    hookClick(Kit.$("mp-plot"), function (e) { var p = e.points[0]; if (p && p.text && idx[p.text] !== undefined) { sel = idx[p.text]; draw(); } });
    Kit.onTheme(draw);
  }

  // ---------- 4. analogies ----------
  function analogyFig() {
    var all = W.words.map(function (x) { return { w: x.w, v: x.v }; }).concat(W.extra), idx = {}; all.forEach(function (r, i) { idx[r.w] = i; });
    var names = all.map(function (r) { return r.w; }).sort(), opts = names.map(function (w) { return { value: w, label: w }; });
    var A = "king", B = "man", C = "woman", excl = true, bar = Kit.$("an-bar"), res = Kit.$("an-res"), eq = Kit.$("an-eqn"), st = Kit.$("an-status"), full = Kit.$("an-full");
    var pre = Kit.html("div", { class: "row-chips" }, bar);
    Kit.chips(pre, W.presets.map(function (p) { return p.a + " &minus; " + p.b + " + " + p.c; }).map(function (s) { return s.replace(/&minus;/g, "−"); }), 0, function (i) { var p = W.presets[i]; A = p.a; B = p.b; C = p.c; selA.value = A; selB.value = B; selC.value = C; draw(i); });
    var ctl = Kit.$("an-controls");
    var selA = Kit.select(ctl, "A", opts, A, function (v) { A = v; draw(-1); });
    var selB = Kit.select(ctl, "minus B", opts, B, function (v) { B = v; draw(-1); });
    var selC = Kit.select(ctl, "plus C", opts, C, function (v) { C = v; draw(-1); });
    var cb = Kit.html("label", { class: "opts" }, ctl); var inp = Kit.html("input", { type: "checkbox", checked: "" }, cb); cb.appendChild(document.createTextNode(" leave out the three input words"));
    inp.addEventListener("change", function () { excl = inp.checked; draw(-1); });
    function draw(pi) {
      var q = add(all[idx[A]].v, all[idx[B]].v, all[idx[C]].v), top = nearest(q, all, excl ? [A, B, C] : [], 5), mx = Math.max.apply(null, top.map(function (t) { return t.c; }));
      var h = "";
      top.forEach(function (t, i) { h += '<div class="an-row' + (i === 0 ? " first" : "") + '"><span class="an-w">' + esc(t.w) + '</span><span class="an-track"><span style="width:' + Math.max(2, 100 * t.c).toFixed(1) + '%"></span></span><b>' + f2(t.c) + "</b></div>"; });
      res.innerHTML = h;
      eq.innerHTML = box(A, "start", "c-navy") + op("&minus;") + box(B, "take away", "c-coral") + op("+") + box(C, "add", "c-teal") + op("&asymp;") + box(top[0].w, "nearest of the " + all.length + " words here", "c-sand");
      var p = pi >= 0 ? W.presets[pi] : null;
      if (!p) { for (var k = 0; k < W.presets.length; k++) if (W.presets[k].a === A && W.presets[k].b === B && W.presets[k].c === C) p = W.presets[k]; }
      if (p) {
        full.innerHTML = '<p class="verdict v-' + p.verdict.split(" ")[0].replace(/[^a-z]/gi, "") + '"><b>' + esc(p.verdict) + ".</b> " + esc(p.note) + "</p><p>Among all 400,000 words, leaving out the inputs: " + p.top.slice(0, 5).map(function (t) { return "<b>" + esc(t[0]) + "</b> (" + f2(t[1]) + ")"; }).join(", ") + ". Without leaving them out: " + p.top_with_inputs.slice(0, 3).map(function (t) { return esc(t[0]) + " (" + f2(t[1]) + ")"; }).join(", ") + ".</p>";
      } else full.innerHTML = '<p>This is your own analogy. Only the ' + all.length + " words of this figure are searched, so the answer can be a poor substitute for the true nearest of all 400,000 words.</p>";
      st.textContent = excl ? "The input words are left out of the answer, as in standard analogy tests." : "The input words are allowed. The answer is often one of them.";
    }
    draw(0);
    // test-set accuracy by category
    var secs = W.analogy_test.sections.filter(function (s) { return s[0] !== "Total accuracy"; }).slice().sort(function (a, b) { return a[1] - b[1]; });
    var pretty = { "capital-common-countries": "capital of a common country", "capital-world": "capital of a country", "currency": "currency of a country", "city-in-state": "city in a US state", "family": "family (boy, girl)", "gram1-adjective-to-adverb": "adjective to adverb", "gram2-opposite": "opposite", "gram3-comparative": "comparative", "gram4-superlative": "superlative", "gram5-present-participle": "present participle", "gram6-nationality-adjective": "nationality adjective", "gram7-past-tense": "past tense", "gram8-plural": "plural noun", "gram9-plural-verbs": "plural verb" };
    Kit.plot("an-acc", function (P) {
      return [{ type: "bar", orientation: "h", x: secs.map(function (s) { return s[1] * 100; }), y: secs.map(function (s) { return pretty[s[0]] || s[0]; }), text: secs.map(function (s) { return Math.round(s[1] * 100) + "%"; }), textposition: "outside", cliponaxis: false,
        marker: { color: secs.map(function (s) { return s[0] === "city-in-state" ? P.coral : P.navy; }) }, hovertemplate: "%{y}: %{x:.0f}% right (of " + "%{customdata} questions)<extra></extra>", customdata: secs.map(function (s) { return s[2]; }) }];
    }, { margin: { l: 150, r: 34, t: 6, b: 56 }, xaxis: { title: { text: "answered right, %", standoff: 12 }, range: [0, 112] }, yaxis: { automargin: true, tickfont: { size: 12 } }, shapes: [{ type: "line", x0: W.analogy_test.overall * 100, x1: W.analogy_test.overall * 100, y0: -0.5, y1: secs.length - 0.5, line: { color: "#8f98a2", dash: "dot", width: 1.5 } }] });
    Kit.onTheme(function () { draw(-1); });
  }

  // ---------- 5. sentences in many languages ----------
  function sentFig() {
    var M = SN.messages, sel = 1, bar = Kit.$("sn-bar"), info = Kit.$("sn-info"), st = Kit.$("sn-status");
    var langs = ["English", "Hindi", "Marathi", "Hinglish"], sym = ["circle", "square", "diamond", "triangle-up"];
    Kit.select(bar, "Message", M.map(function (m, i) { return { value: String(i), label: m.id + " " + m.language }; }), String(sel), function (v) { sel = Number(v); draw(); });
    function draw() {
      Kit.$("sn-bar").querySelector("select").value = String(sel);
      Kit.plot("sn-plot", function (P) {
        var cols = [P.navy, P.coral, P.teal, P.sand], tr = [];
        langs.forEach(function (L, li) { tr.push({ type: "scatter", mode: "markers", name: L, x: [null], y: [null], marker: { size: 11, symbol: sym[li], color: P.ink2 }, hoverinfo: "skip" }); });
        ["Spadenose count", "Tag 4127", "Hammerhead", "Uran water"].forEach(function (t, ti) { tr.push({ type: "scatter", mode: "markers", name: t, x: [null], y: [null], marker: { size: 11, symbol: "circle", color: cols[ti] }, hoverinfo: "skip" }); });
        langs.forEach(function (L, li) {
          var m = M.filter(function (x) { return x.language === L; });
          tr.push({ type: "scatter", mode: "markers+text", name: L, showlegend: false, x: m.map(function (x) { return x.xy[0]; }), y: m.map(function (x) { return x.xy[1]; }), text: m.map(function (x) { return x.id; }), textposition: ["top center", "top right", "bottom left", "bottom center"][li], textfont: { size: 11, color: P.ink2 },
            marker: { size: 12, symbol: sym[li], color: m.map(function (x) { return cols[x.topic]; }), line: { color: P.ink2, width: 1 } }, hovertemplate: "%{text} " + L + "<extra></extra>" });
        });
        var s = M[sel], lx = [], ly = [], order = SN.cos[sel].map(function (c, i) { return { c: c, i: i }; }).filter(function (o) { return o.i !== sel; }).sort(function (a, b) { return b.c - a.c; }).slice(0, 3);
        order.forEach(function (o) { lx.push(s.xy[0], M[o.i].xy[0], null); ly.push(s.xy[1], M[o.i].xy[1], null); });
        tr.push({ type: "scatter", mode: "lines", x: lx, y: ly, line: { color: P.coral, width: 1.5, dash: "dot" }, hoverinfo: "skip", showlegend: false });
        return tr;
      }, { margin: { l: 8, r: 8, t: 8, b: 8 }, xaxis: { visible: false, range: [-1.2, 1.2] }, yaxis: { visible: false, range: [-1.25, 1.25] }, showlegend: true, legend: { orientation: "h", y: -0.02 }, hovermode: "closest", dragmode: false });
      var s = M[sel], order = SN.cos[sel].map(function (c, i) { return { c: c, i: i }; }).filter(function (o) { return o.i !== sel; }).sort(function (a, b) { return b.c - a.c; }).slice(0, 3);
      var h = '<p class="sn-msg"><b>' + s.id + " (" + s.language + "), topic: " + esc(SN.topics[s.topic]) + ".</b> " + esc(s.text) + "</p>" + (s.gloss ? '<p class="sn-gloss">English meaning: ' + esc(s.gloss) + "</p>" : "");
      h += '<div class="eqn">' + order.map(function (o) { var m = M[o.i]; return box(f2(o.c), m.id + " " + m.language + (m.topic === s.topic ? ", same topic" : ", other topic"), m.topic === s.topic ? "c-teal" : "c-coral"); }).join("") + "</div>";
      info.innerHTML = h;
      st.textContent = "Dotted lines join the message to its three most similar messages by cosine. Colour is the topic, symbol is the language.";
    }
    draw();
    hookClick(Kit.$("sn-plot"), function (e) { var p = e.points[0]; if (p && p.text) { sel = M.map(function (m) { return m.id; }).indexOf(p.text); draw(); } });
    Kit.onTheme(draw);
  }
})();
