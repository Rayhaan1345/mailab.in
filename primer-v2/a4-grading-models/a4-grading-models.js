// A4 Grading a model fairly: figures for the chapter. Shark records are made-up (not field data); the clinic data is the published Pima set.
(function () {
  "use strict";
  var IS_NODE = typeof module !== "undefined" && module.exports && typeof document === "undefined";

  // ---------- pure helpers (exported for node tests) ----------
  function sum(a) { var s = 0; for (var i = 0; i < a.length; i++) s += a[i]; return s; }
  function logit(p) { return Math.log(p / (1 - p)); }
  function sigm(z) { return 1 / (1 + Math.exp(-z)); }
  // Counts for the rule "flag when z >= t" (z is a log-odds score, y is 0 or 1).
  function count(z, y, t) {
    var tp = 0, fp = 0, tn = 0, fn = 0;
    for (var i = 0; i < z.length; i++) {
      var f = z[i] >= t;
      if (y[i]) { if (f) tp++; else fn++; } else { if (f) fp++; else tn++; }
    }
    return { tp: tp, fp: fp, tn: tn, fn: fn };
  }
  function rates(c) {
    var P = c.tp + c.fn, N = c.fp + c.tn, F = c.tp + c.fp, T = P + N, d = 2 * c.tp + c.fp + c.fn;
    return { P: P, N: N, F: F, T: T, acc: T ? (c.tp + c.tn) / T : null, prec: F ? c.tp / F : null, rec: P ? c.tp / P : null,
      spec: N ? c.tn / N : null, f1: d ? 2 * c.tp / d : null, fpr: N ? c.fp / N : null };
  }
  // ROC points from the highest threshold down; one point per distinct score.
  function roc(z, y) {
    var n = z.length, idx = [], i;
    for (i = 0; i < n; i++) idx.push(i);
    idx.sort(function (a, b) { return z[b] - z[a]; });
    var P = sum(y), N = n - P, tp = 0, fp = 0, fpr = [0], tpr = [0], thr = [Infinity], prec = [1];
    for (var k = 0; k < n; k++) {
      i = idx[k]; if (y[i]) tp++; else fp++;
      if (k === n - 1 || z[idx[k + 1]] !== z[i]) { fpr.push(fp / N); tpr.push(tp / P); thr.push(z[i]); prec.push(tp / (tp + fp)); }
    }
    return { fpr: fpr, tpr: tpr, thr: thr, prec: prec, P: P, N: N };
  }
  function trapz(x, y) { var a = 0; for (var i = 1; i < x.length; i++) a += (x[i] - x[i - 1]) * (y[i] + y[i - 1]) / 2; return a; }
  // Share of (positive, negative) pairs in which the positive scores higher; ties count half.
  function aucPairs(z, y) {
    var pos = [], neg = [], i, j, win = 0, tie = 0;
    for (i = 0; i < z.length; i++) (y[i] ? pos : neg).push(z[i]);
    for (i = 0; i < pos.length; i++) for (j = 0; j < neg.length; j++) { if (pos[i] > neg[j]) win++; else if (pos[i] === neg[j]) tie++; }
    var tot = pos.length * neg.length;
    return { wins: win, ties: tie, total: tot, auc: tot ? (win + tie / 2) / tot : null };
  }
  // 95% Wilson interval for k successes in n trials.
  function wilson(k, n) {
    if (!n) return [0, 1];
    var z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, c - h), Math.min(1, c + h)];
  }
  // Chebyshev polynomial fit by Householder QR (stable for degrees up to about 12).
  function cheb(u, M) { var r = [1, u]; for (var j = 2; j <= M; j++) r.push(2 * u * r[j - 1] - r[j - 2]); return r.slice(0, M + 1); }
  function lstsq(A, y) {
    var n = A.length, p = A[0].length, R = A.map(function (r) { return r.slice(); }), b = y.slice(), i, j, k;
    for (k = 0; k < p; k++) {
      var norm = 0; for (i = k; i < n; i++) norm += R[i][k] * R[i][k]; norm = Math.sqrt(norm);
      if (norm < 1e-14) continue;
      var s = R[k][k] > 0 ? -norm : norm, v = []; for (i = k; i < n; i++) v.push(R[i][k]); v[0] -= s;
      var vn = 0; for (i = 0; i < v.length; i++) vn += v[i] * v[i];
      if (vn < 1e-30) continue;
      for (j = k; j < p; j++) { var d = 0; for (i = k; i < n; i++) d += v[i - k] * R[i][j]; var f = 2 * d / vn; for (i = k; i < n; i++) R[i][j] -= f * v[i - k]; }
      var d2 = 0; for (i = k; i < n; i++) d2 += v[i - k] * b[i]; var f2 = 2 * d2 / vn; for (i = k; i < n; i++) b[i] -= f2 * v[i - k];
    }
    var w = new Array(p).fill(0);
    for (k = p - 1; k >= 0; k--) { var t = b[k]; for (j = k + 1; j < p; j++) t -= R[k][j] * w[j]; w[k] = Math.abs(R[k][k]) < 1e-12 ? 0 : t / R[k][k]; }
    return w;
  }
  function polyFit(xs, ys, M, lo, hi) {
    var u = function (x) { return (2 * x - (lo + hi)) / (hi - lo); };
    var w = lstsq(xs.map(function (x) { return cheb(u(x), M); }), ys);
    return function (x) { var r = cheb(u(x), M), t = 0; for (var j = 0; j <= M; j++) t += w[j] * r[j]; return t; };
  }
  function rmse(f, xs, ys) { var e = 0; for (var i = 0; i < xs.length; i++) { var d = f(xs[i]) - ys[i]; e += d * d; } return Math.sqrt(e / xs.length); }
  function rngOf(seed) { var a = seed >>> 0; return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; var t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }

  if (IS_NODE) { module.exports = { count: count, rates: rates, roc: roc, trapz: trapz, aucPairs: aucPairs, wilson: wilson, polyFit: polyFit, rmse: rmse, logit: logit, sigm: sigm, rngOf: rngOf }; return; }

  // ---------- small DOM helpers ----------
  var NAMES = ["spadenose", "milk", "bamboo", "hammerhead", "Ganges"];
  function box(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function op(s) { return '<span class="op">' + s + '</span>'; }
  function pc(v, d) { return v == null ? "n/a" : Kit.pct(v, d == null ? 1 : d); }
  function fix(v, d) { return v == null ? "n/a" : Kit.fmt(v, d); }
  function commas(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function theme(fn) { fn(); Kit.onTheme(fn); window.addEventListener("resize", Kit.debounce(fn)); }
  var CACHE = {};
  function load(path) { return CACHE[path] || (CACHE[path] = Kit.data(path)); }
  function sel(id) { return d3.select(Kit.$(id)); }

  var REG = [];
  document.addEventListener("DOMContentLoaded", function () {
    REG.forEach(function (f) {
      if (!Kit.$(f[0])) return;
      try { var r = f[1](); if (r && r.catch) r.catch(function (e) { console.error(f[0], e); fail(f[0], e); }); } catch (e) { console.error(f[0], e); fail(f[0], e); }
    });
  });
  function fail(id, e) { var n = Kit.$(id); if (n) { var p = document.createElement("p"); p.className = "status"; p.textContent = "This figure could not start: " + (e && e.message ? e.message : e); n.appendChild(p); } }

  // ---------- 1. polynomial degree ----------
  function polyFig() {
    return Promise.all([load("../data/sharks/catch.json"), load("data/perm.json")]).then(function (r) {
      var sp = r[0].records.filter(function (s) { return s.species === "spadenose"; }), perm = r[1].seeds;
      var L = sp.map(function (s) { return s.length_cm; }), W = sp.map(function (s) { return s.weight_kg; }), LO = 18, HI = 60, MMAX = 9;
      var seed = 2, n = 14, M = 3, fit = null;
      var sM = Kit.slider(Kit.$("pf-controls"), { label: "Degree M of the polynomial", min: 1, max: MMAX, step: 1, value: 3, digits: 0 }, function (v) { M = v; draw(); });
      var sN = Kit.slider(Kit.$("pf-controls"), { label: "Training sample size", min: 10, max: 60, step: 1, value: 14, digits: 0 }, function (v) { n = v; compute(); draw(); });
      Kit.button(Kit.$("pf-bar"), "New sample", function () { seed = seed % 12 + 1; compute(); draw(); });
      Kit.button(Kit.$("pf-bar"), "Back to the first sample", function () { seed = 2; n = 14; M = 3; sN.set(14); sM.set(3); compute(); draw(); });
      var tag = Kit.html("span", { class: "label" }, Kit.$("pf-bar"), "");
      function compute() {
        var idx = perm[String(seed)], tr = idx.slice(0, n), te = idx.slice(n), top = Math.min(MMAX, n - 1);
        var xt = tr.map(function (i) { return L[i]; }), yt = tr.map(function (i) { return W[i]; }), xe = te.map(function (i) { return L[i]; }), ye = te.map(function (i) { return W[i]; });
        var mean = sum(yt) / n, base = Math.sqrt(sum(ye.map(function (v) { return (v - mean) * (v - mean); })) / ye.length);
        var fs = [null], errs = [null];
        for (var m = 1; m <= top; m++) { var f = polyFit(xt, yt, m, LO, HI); fs.push(f); errs.push([rmse(f, xt, yt), rmse(f, xe, ye)]); }
        fit = { tr: tr, te: te, xt: xt, yt: yt, xe: xe, ye: ye, fs: fs, errs: errs, top: top, base: base };
        sM.input.max = top; if (M > top) { M = top; sM.set(M); }
      }
      function draw() {
        var f = fit.fs[M], xs = [], ys = [], ymax = 0, ymin = 0;
        for (var x = 17; x <= 61; x += 0.5) { xs.push(x); var y = f(x); ys.push(y); ymax = Math.max(ymax, y); ymin = Math.min(ymin, y); }
        var off = ymax > 1.05 || ymin < -0.25, extreme = Math.max(ymax, -ymin);
        Kit.plot("pf-fit", function (P) {
          return [
            { x: fit.xe, y: fit.ye, mode: "markers", marker: { color: P.coral, size: 5, opacity: 0.4 }, hoverinfo: "skip" },
            { x: fit.xt, y: fit.yt, mode: "markers", marker: { color: P.navy, size: 10, line: { color: P.paper, width: 1.5 } }, hovertemplate: "%{x:.1f} cm, %{y:.3f} kg<extra>training shark</extra>" },
            { x: xs, y: ys, mode: "lines", line: { color: P.teal, width: 3 }, hoverinfo: "skip" }
          ];
        }, function (P) {
          return { xaxis: { title: { text: "length, cm" }, range: [16, 62] }, yaxis: { title: { text: "weight, kg" }, range: [-0.25, 1.05] },
            annotations: [
              { xref: "paper", yref: "paper", x: 0.02, y: 0.98, xanchor: "left", yanchor: "top", showarrow: false, align: "left", text: "navy: " + n + " training sharks<br>coral: records the curve never saw<br>teal: polynomial of degree " + M, font: { size: 12.5, color: P.ink2 }, bgcolor: P.paper, borderpad: 3 }
            ].concat(off ? [{ xref: "paper", yref: "paper", x: 0.98, y: 0.04, xanchor: "right", yanchor: "bottom", showarrow: false, text: "the curve reaches " + (extreme >= 100 ? commas(extreme) : Kit.fmt(extreme, 1)) + " kg and leaves the chart", font: { size: 12.5, color: P.coral } }] : []) };
        });
        var degs = [], tr = [], te = [];
        for (var m = 1; m <= fit.top; m++) { degs.push(m); tr.push(fit.errs[m][0]); te.push(fit.errs[m][1]); }
        Kit.plot("pf-err", function (P) {
          return [
            { x: [0.6, MMAX + 0.4], y: [fit.base, fit.base], mode: "lines", line: { color: P.muted, width: 1.6, dash: "dot" }, hoverinfo: "skip" },
            { x: degs, y: tr, mode: "lines+markers", line: { color: P.navy, width: 2.5 }, marker: { color: P.navy, size: 7 }, hovertemplate: "degree %{x}: %{y:.3f} kg<extra>training error</extra>" },
            { x: degs, y: te, mode: "lines+markers", line: { color: P.coral, width: 2.5 }, marker: { color: P.coral, size: 7 }, hovertemplate: "degree %{x}: %{y:.3f} kg<extra>test error</extra>" },
            { x: [M, M], y: [fit.errs[M][0], fit.errs[M][1]], mode: "markers", marker: { color: P.teal, size: 13, symbol: "circle-open", line: { width: 3, color: P.teal } }, hoverinfo: "skip" }
          ];
        }, function (P) {
          return { xaxis: { title: { text: "degree M" }, range: [0.5, MMAX + 0.5], dtick: 1 }, yaxis: { title: { text: "error (RMSE), kg, log scale" }, type: "log", range: [-2, 3.3], tickvals: [0.01, 0.1, 1, 10, 100, 1000], ticktext: ["0.01", "0.1", "1", "10", "100", "1000"] },
            annotations: [
              { x: Math.min(fit.top, MMAX), y: Math.log10(te[te.length - 1]), xref: "x", yref: "y", text: "test error", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 13, color: P.coral }, yshift: 6 },
              { x: 1, y: Math.log10(tr[0]), xref: "x", yref: "y", text: "training error", showarrow: false, xanchor: "left", yanchor: "top", font: { size: 13, color: P.navy }, yshift: -8, xshift: 6 },
              { x: MMAX + 0.4, y: Math.log10(fit.base), xref: "x", yref: "y", text: "predict the mean weight", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 12, color: P.muted } }
            ] };
        });
        var e = fit.errs[M], ratio = e[1] / e[0], best = 1;
        for (var k = 2; k <= fit.top; k++) if (fit.errs[k][1] < fit.errs[best][1]) best = k;
        Kit.$("pf-eqn").innerHTML = box("c-sand", String(M), "degree M") + op("&rarr;") + box("c-navy", Kit.fmt(e[0], 3), "training RMSE, kg") + box("c-coral", e[1] >= 100 ? commas(e[1]) : Kit.fmt(e[1], e[1] < 1 ? 3 : 1), "test RMSE, kg") + op("&rArr;") + box("c-teal", ratio >= 100 ? commas(ratio) : Kit.fmt(ratio, 1), "test error &divide; training error");
        var verdict, bestErr = fit.errs[best][1];
        if (e[1] <= 1.15 * bestErr) verdict = "A good balance: the test error is within 15% of the lowest any degree gives for this sample.";
        else if (M > best) verdict = "Overfitting: the curve follows the " + n + " training sharks, and the test error is " + (ratio >= 100 ? commas(ratio) : Kit.fmt(ratio, 1)) + " times the training error.";
        else verdict = "Underfitting: the curve is too stiff for the trend, and a higher degree would do better on the test records.";
        Kit.setText("pf-status", verdict + " Lowest test error for this sample: degree " + best + ". Predicting the mean weight would give " + Kit.fmt(fit.base, 3) + " kg.");
        tag.textContent = "Sample " + seed + " of 12";
      }
      compute(); draw();
    });
  }
  REG.push(["pf", polyFig]);

  // ---------- shared: dots laid out in blocks (training, validation, test or folds) ----------
  function layoutBlocks(a, B, W, vertical) {
    var n = a.length, counts = new Array(B).fill(0), i, b;
    for (i = 0; i < n; i++) counts[a[i]]++;
    var g = W < 520 ? 8 : 14, head = vertical ? 24 : 44, res = { pos: new Array(n), blocks: [], counts: counts, s: 0, H: 0 };
    var s, usable;
    if (!vertical) {
      usable = W - (B - 1) * g; s = Math.max(6, Math.min(12.5, usable / 66));
      var x0 = 0, rowsMax = 0;
      for (b = 0; b < B; b++) {
        var wb = usable * counts[b] / n, cols = Math.max(1, Math.floor(wb / s)), rows = Math.ceil(counts[b] / cols);
        res.blocks.push({ x: x0, w: wb, y: 0, cols: cols, rows: rows, head: head });
        x0 += wb + g; rowsMax = Math.max(rowsMax, rows);
      }
      res.H = head + rowsMax * s + 10;
    } else {
      s = Math.max(6, Math.min(8, W / 50)); var y0 = 0;
      for (b = 0; b < B; b++) {
        var cols2 = Math.max(1, Math.floor(W / s)), rows2 = Math.ceil(counts[b] / cols2);
        res.blocks.push({ x: 0, w: W, y: y0, cols: cols2, rows: rows2, head: head });
        y0 += head + rows2 * s + 8;
      }
      res.H = y0;
    }
    res.s = s;
    var fill = new Array(B).fill(0);
    for (i = 0; i < n; i++) {
      b = a[i]; var k = fill[b]++, bl = res.blocks[b];
      res.pos[i] = { x: bl.x + (k % bl.cols + 0.5) * s, y: bl.y + bl.head + (Math.floor(k / bl.cols) + 0.5) * s };
    }
    return res;
  }
  function decode(str, base) { var a = new Array(str.length); for (var i = 0; i < str.length; i++) { var c = str.charCodeAt(i); a[i] = c >= 97 ? c - 87 : c >= 65 ? c - 55 : c - 48; } return a; }
  var MODEL_NAMES = { lazy: "Always says spadenose", logistic: "Logistic regression", tree3: "Decision tree, depth 3", treeFull: "Decision tree, no depth limit" };

  // ---------- 2. training, validation and test sets ----------
  function holdoutFig() {
    return Promise.all([load("data/splits.json"), load("data/scores.json")]).then(function (r) {
      var S = r[0], species = r[1].species, n = species.length, node = Kit.$("sp-svg"), svg = d3.select(node);
      var ri = 1, strat = false, si = 0, picked = null, opened = false, peek = false, cur = null, dots = [], colorOf = [];
      Kit.chips(Kit.$("sp-ratio"), S.ratios, 1, function (i) { ri = i; reshuffle(false); });
      Kit.chips(Kit.$("sp-strat"), ["Plain shuffle", "Stratified by species"], 0, function (i) { strat = i === 1; reshuffle(false); });
      Kit.button(Kit.$("sp-btns"), "Shuffle again", function () { si = (si + 1) % S.seeds.length; reshuffle(true); });
      var bPick = Kit.button(Kit.$("sp-bar2"), "Pick the best model on validation", function () { pickBest(); });
      var bOpen = Kit.button(Kit.$("sp-bar2"), "Open the test set, once", function () { opened = true; update(); });
      var bPeek = Kit.button(Kit.$("sp-bar2"), "Peek at all test scores", function () { peek = true; update(); });
      function row() { return S.holdout[strat ? "strat" : "plain"][ri][si]; }
      function pickBest() { var R = row(), best = null; S.models.forEach(function (m) { if (best === null || R.acc[m][1] > R.acc[best][1]) best = m; }); picked = best; update(); }
      function reshuffle(animate) { picked = null; opened = false; peek = false; build(animate); }
      function palette() { var P = Kit.palette(); return [P.navy, P.sand, P.coral]; }
      function build(animate) {
        var P = Kit.palette(), W = Kit.width(node), vertical = W < 520, a = decode(row().a), L = layoutBlocks(a, 3, W, vertical), cols = palette(), names = ["Training", "Validation", "Test"];
        var prev = cur; cur = { a: a, L: L, W: W };
        svg.attr("viewBox", "0 0 " + W + " " + L.H).attr("height", L.H);
        var firstTime = !dots.length || (prev && prev.W !== W) || !svg.select("circle").size();
        if (firstTime) {
          svg.selectAll("*").remove(); dots = [];
          for (var i = 0; i < n; i++) dots.push(svg.append("circle").attr("cx", L.pos[i].x).attr("cy", L.pos[i].y));
        }
        // headers
        svg.selectAll(".hd").remove();
        var gan = [0, 0, 0], ham = [0, 0, 0];
        a.forEach(function (b, i) { if (species[i] === 4) gan[b]++; if (species[i] === 3) ham[b]++; });
        L.blocks.forEach(function (bl, b) {
          var g = svg.append("g").attr("class", "hd");
          if (!vertical) {
            g.append("rect").attr("x", bl.x).attr("y", 0).attr("width", Math.max(2, bl.w)).attr("height", 3).attr("fill", cols[b]);
            g.append("text").attr("x", bl.x).attr("y", 18).attr("font-size", 14).attr("font-weight", 600).attr("fill", P.ink).text(names[b]);
            g.append("text").attr("x", bl.x).attr("y", 33).attr("font-size", 12.5).attr("fill", P.ink2).text(L.counts[b] + " records");
            g.append("text").attr("x", bl.x).attr("y", 33).attr("dx", bl.w > 150 ? 78 : 0).attr("dy", bl.w > 150 ? 0 : 0).attr("font-size", 12.5).attr("fill", P.coral).text(bl.w > 150 ? "Ganges: " + gan[b] : "").attr("opacity", bl.w > 150 ? 1 : 0);
            if (bl.w <= 150) g.append("text").attr("x", bl.x).attr("y", 44).attr("font-size", 12.5).attr("fill", P.coral).text("Ganges: " + gan[b]);
          } else {
            g.append("text").attr("x", 0).attr("y", bl.y + 16).attr("font-size", 13.5).attr("fill", P.ink).text(names[b] + ": " + L.counts[b] + " records, Ganges " + gan[b]);
          }
        });
        function paint(posFn) {
          for (var i = 0; i < n; i++) {
            var p = posFn(i), isG = species[i] === 4;
            dots[i].attr("cx", p.x).attr("cy", p.y).attr("r", isG ? L.s * 0.62 : L.s * 0.36).attr("fill", cols[a[i]]).attr("opacity", a[i] === 0 ? 0.7 : 0.95)
              .attr("stroke", isG ? P.ink : "none").attr("stroke-width", isG ? 2 : 0);
          }
        }
        if (animate && prev && !firstTime && !Kit.reduce) {
          var from = prev.L.pos;
          for (var j = 0; j < n; j++) dots[j].attr("fill", cols[a[j]]).attr("opacity", a[j] === 0 ? 0.7 : 0.95);
          Kit.animate(700, function (p) { var e = Kit.ease(p); paint(function (i) { return { x: from[i].x + (L.pos[i].x - from[i].x) * e, y: from[i].y + (L.pos[i].y - from[i].y) * e }; }); });
        } else paint(function (i) { return L.pos[i]; });
        update();
      }
      function update() {
        var R = row(), P = Kit.palette(), tb = Kit.$("sp-table"), html = '<thead><tr><th>Model</th><th>Training</th><th>Validation</th><th>Test</th></tr></thead><tbody>', maxv = -1;
        S.models.forEach(function (m) { maxv = Math.max(maxv, R.acc[m][1]); });
        S.models.forEach(function (m) {
          var A = R.acc[m], show = peek || (opened && picked === m);
          var gap = A[0] - A[1] > 0.1;
          html += '<tr' + (picked === m ? ' class="pick"' : '') + '><td>' + MODEL_NAMES[m] + '</td><td>' + pc(A[0]) + '</td><td' + (gap ? ' class="gap"' : '') + '>' + pc(A[1]) + '</td>' +
            (show ? '<td>' + pc(A[2]) + '</td>' : '<td class="locked">locked</td>') + '</tr>';
        });
        tb.innerHTML = html + '</tbody>';
        bPick.disabled = picked !== null; bOpen.disabled = picked === null || opened; bPeek.disabled = peek;
        var nt = R.a.split("2").length - 1, p = picked ? R.acc[picked][opened ? 2 : 1] : 0.8, se = Math.sqrt(p * (1 - p) / nt);
        Kit.$("sp-eqn").innerHTML = box("c-coral", String(nt), "test records") + op("&rarr;") + box("c-navy", pc(p), picked ? (opened ? "test accuracy of the pick" : "validation accuracy of the pick") : "a score of 80%, for example") + op("&rArr;") +
          box("c-teal", "&plusmn;" + Kit.fmt(1.96 * se * 100, 1) + " points", "95% interval, from SE = " + Kit.fmt(se * 100, 1) + " points");
        var all = S.holdout[strat ? "strat" : "plain"][ri].map(function (x) { return x.acc[picked || "logistic"][2]; }), lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
        var txt;
        if (peek) txt = "All test scores are visible. A model chosen with them has used the test set, so its score is no longer a fair estimate.";
        else if (opened) txt = "Report " + pc(R.acc[picked][2]) + " for " + MODEL_NAMES[picked].toLowerCase() + ". The test set has been used once. Over all " + S.seeds.length + " shuffles of this split the same model scored between " + pc(lo) + " and " + pc(hi) + " on its test set.";
        else if (picked) txt = MODEL_NAMES[picked] + " has the best validation accuracy (" + pc(R.acc[picked][1]) + "). Now open the test set, once.";
        else txt = "The test column is locked. Compare the models on training and validation first. The tree with no depth limit scores " + pc(R.acc.treeFull[0]) + " on its own training records.";
        Kit.setText("sp-status", txt);
      }
      theme(function () { build(false); });
    });
  }
  REG.push(["sp", holdoutFig]);

  // ---------- 3. k-fold cross-validation ----------
  function kfoldFig() {
    return Promise.all([load("data/splits.json"), load("data/scores.json")]).then(function (r) {
      var S = r[0], species = r[1].species, n = species.length, node = Kit.$("kf-svg"), svg = d3.select(node);
      var KS = ["5", "10"], ki = 0, strat = false, si = 0, model = "logistic", step = -1, cur = null, dots = [], timer = null;
      Kit.chips(Kit.$("kf-k"), ["5 folds", "10 folds"], 0, function (i) { ki = i; reset(true); });
      Kit.chips(Kit.$("kf-model"), S.models.map(function (m) { return MODEL_NAMES[m]; }), 1, function (i) { model = S.models[i]; draw(); });
      Kit.chips(Kit.$("kf-strat"), ["Plain shuffle", "Stratified by species"], 0, function (i) { strat = i === 1; reset(true); });
      var bStep = Kit.button(Kit.$("kf-bar"), "Step", function () { advance(); });
      var bPlay = Kit.button(Kit.$("kf-bar"), "Play", function () { if (timer) { stop(); return; } if (step >= K() - 1) { step = -1; draw(); } bPlay.textContent = "Pause"; timer = setInterval(function () { if (!advance()) stop(); }, 900); });
      Kit.button(Kit.$("kf-bar"), "Start over", function () { stop(); step = -1; draw(); });
      Kit.button(Kit.$("kf-bar"), "Shuffle again", function () { si = (si + 1) % S.seeds.length; reset(true); });
      function K() { return Number(KS[ki]); }
      function row() { return S.kfold[strat ? "strat" : "plain"][KS[ki]][si]; }
      function stop() { if (timer) clearInterval(timer); timer = null; bPlay.textContent = "Play"; }
      function advance() { if (step >= K() - 1) return false; step++; draw(); return step < K() - 1; }
      function reset(animate) { stop(); step = -1; build(animate); }
      function build(animate) {
        var W = Kit.width(node), a = decode(row().a), k = K(), pad = W < 520 ? 34 : 42, L = layoutBlocks(a, k, W - pad, false), prev = cur;
        cur = { a: a, L: L, W: W, pad: pad, k: k };
        var chartH = 118, H = L.H + 24 + chartH;
        svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H);
        var fresh = !dots.length || !prev || prev.W !== W || prev.k !== k || !svg.select("circle").size();
        if (fresh) { svg.selectAll("*").remove(); dots = []; svg.append("g").attr("class", "zone").attr("transform", "translate(" + pad + ",0)"); svg.append("g").attr("class", "chart"); var z = svg.select(".zone"); for (var i = 0; i < n; i++) dots.push(z.append("circle")); }
        if (animate && prev && !fresh && !Kit.reduce) {
          var from = prev.L.pos;
          Kit.animate(700, function (p) { var e = Kit.ease(p); place(function (i) { return { x: from[i].x + (L.pos[i].x - from[i].x) * e, y: from[i].y + (L.pos[i].y - from[i].y) * e }; }); });
        } else place(function (i) { return L.pos[i]; });
        draw();
      }
      function place(posFn) {
        var L = cur.L;
        for (var i = 0; i < n; i++) { var p = posFn(i), g = species[i] === 4; dots[i].attr("cx", p.x).attr("cy", p.y).attr("r", g ? L.s * 0.62 : L.s * 0.36).attr("stroke", g ? Kit.palette().ink : "none").attr("stroke-width", g ? 2 : 0); }
      }
      function draw() {
        if (!cur) return;
        var P = Kit.palette(), L = cur.L, k = cur.k, a = cur.a, R = row(), acc = R.acc[model], W = cur.W, pad = cur.pad;
        for (var i = 0; i < n; i++) {
          var c = step < 0 ? P.muted : (a[i] === step ? P.sand : P.navy);
          dots[i].attr("fill", c).attr("opacity", step < 0 ? 0.55 : (a[i] === step ? 1 : 0.7));
        }
        var z = svg.select(".zone"); z.selectAll(".hd").remove();
        L.blocks.forEach(function (bl, b) {
          var g = z.append("g").attr("class", "hd"), on = b === step, done = b <= step;
          g.append("rect").attr("x", bl.x).attr("y", 0).attr("width", Math.max(2, bl.w)).attr("height", 3).attr("fill", on ? P.sand : P.rule);
          g.append("text").attr("x", bl.x + bl.w / 2).attr("y", 20).attr("text-anchor", "middle").attr("font-size", k > 5 && W < 520 ? 11.5 : 13.5).attr("font-weight", on ? 700 : 500).attr("fill", on ? P.ink : P.ink2).text(k > 5 && W < 520 ? String(b + 1) : "Fold " + (b + 1));
          if (bl.w > 56) g.append("text").attr("x", bl.x + bl.w / 2).attr("y", 36).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", on ? P.sand : P.muted).text(on ? "validation" : (step < 0 ? "" : "training"));
        });
        // dot plot of the fold scores, aligned with the blocks
        var ch = svg.select(".chart"); ch.selectAll("*").remove();
        var y0 = L.H + 24, h = 96, ys = function (v) { return y0 + h - (v - 0.3) / 0.7 * h; };
        ch.append("text").attr("x", 0).attr("y", y0 - 8).attr("font-size", 12.5).attr("fill", P.muted).text("accuracy on the fold scored (axis starts at 30%)");
        [0.3, 0.5, 0.7, 0.9].forEach(function (v) {
          ch.append("line").attr("x1", pad - 4).attr("x2", W).attr("y1", ys(v)).attr("y2", ys(v)).attr("stroke", P.rule);
          ch.append("text").attr("x", pad - 8).attr("y", ys(v) + 4).attr("text-anchor", "end").attr("font-size", 11.5).attr("fill", P.muted).text(Math.round(v * 100) + "%");
        });
        var done = acc.slice(0, step + 1), mean = done.length ? sum(done) / done.length : null;
        if (mean !== null) ch.append("line").attr("x1", pad).attr("x2", W).attr("y1", ys(mean)).attr("y2", ys(mean)).attr("stroke", P.teal).attr("stroke-width", 2).attr("stroke-dasharray", "6 4");
        L.blocks.forEach(function (bl, b) {
          if (b > step) return;
          var cx = pad + bl.x + bl.w / 2;
          ch.append("line").attr("x1", cx).attr("x2", cx).attr("y1", ys(0.3)).attr("y2", ys(acc[b])).attr("stroke", P.rule).attr("stroke-width", 2);
          ch.append("circle").attr("cx", cx).attr("cy", ys(acc[b])).attr("r", b === step ? 7 : 5.5).attr("fill", b === step ? P.sand : P.teal).attr("stroke", P.ink).attr("stroke-width", b === step ? 2 : 0);
          if (bl.w > 44) ch.append("text").attr("x", cx).attr("y", ys(acc[b]) - 11).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", P.ink).text(Kit.fmt(acc[b] * 100, 1));
        });
        if (mean !== null) ch.append("text").attr("x", W - 2).attr("y", ys(mean) + 16).attr("text-anchor", "end").attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.teal).text("mean so far " + pc(mean));
        // equation boxes and status
        var sd = done.length > 1 ? Math.sqrt(sum(done.map(function (v) { return (v - mean) * (v - mean); })) / (done.length - 1)) : null, html = "";
        if (k === 5) {
          for (var b = 0; b < k; b++) { html += (b ? op("+") : "") + box(b <= step ? "c-sand" : "", b <= step ? Kit.fmt(acc[b], 3) : "?", "fold " + (b + 1)); }
          html += op("&divide;") + box("", String(k), "folds") + op("=") + box("c-teal", mean !== null && step === k - 1 ? Kit.fmt(mean, 3) : "?", "cross-validated accuracy");
        } else html = box("c-sand", String(step + 1), "folds scored so far") + op("&rarr;") + box("c-teal", mean !== null ? Kit.fmt(mean, 3) : "?", "mean of the scores so far");
        html += box("c-navy", sd !== null ? Kit.fmt(sd * 100, 1) : "?", "SD across folds, points");
        Kit.$("kf-eqn").innerHTML = html;
        var done_ = step === k - 1, tb = "";
        if (done_) {
          var rows = S.models.map(function (m) { var v = R.acc[m], mu = sum(v) / v.length, s = Math.sqrt(sum(v.map(function (x) { return (x - mu) * (x - mu); })) / (v.length - 1)); return { m: m, mu: mu, sd: s, lo: Math.min.apply(null, v), hi: Math.max.apply(null, v) }; });
          var best = rows.reduce(function (p, q) { return q.mu > p.mu ? q : p; });
          tb = '<thead><tr><th>Model</th><th>Mean accuracy</th><th>SD across folds</th><th>Lowest to highest fold</th></tr></thead><tbody>' + rows.map(function (x) { return '<tr' + (x.m === best.m ? ' class="pick"' : '') + '><td>' + MODEL_NAMES[x.m] + '</td><td>' + pc(x.mu) + '</td><td>' + Kit.fmt(x.sd * 100, 1) + ' points</td><td>' + pc(x.lo) + ' to ' + pc(x.hi) + '</td></tr>'; }).join("") + '</tbody>';
        }
        Kit.$("kf-table").innerHTML = tb; Kit.$("kf-table").style.display = done_ ? "" : "none";
        bStep.disabled = step >= k - 1;
        var nv = L.counts[Math.max(step, 0)];
        Kit.setText("kf-status", step < 0 ? "The " + n + " records are dealt into " + k + " folds of about " + Math.round(n / k) + " records. Press Step to score the first fold." :
          "Round " + (step + 1) + " of " + k + ": train " + MODEL_NAMES[model].toLowerCase() + " on the other " + (n - nv) + " records (navy), then score it on fold " + (step + 1) + " (sand, " + nv + " records): " + pc(acc[step]) + "." + (done_ ? " All records have now been scored once. Compare the means in the table." : ""));
      }
      theme(function () { build(false); });
    });
  }
  REG.push(["kf", kfoldFig]);

  // ---------- 4. leakage ----------
  function leakFig() {
    return load("data/leak.json").then(function (D) {
      var tab = 0, leaky = true, seed = 3;
      var TABS = [
        { name: "Scaling", cols: ["Scaler fitted on all rows", "Scaler fitted on training rows only"],
          steps: [["Scale the features (mean and SD)", [1, 1], [1, 0]], ["Fit the model", [1, 0], [1, 0]], ["Score the model", [0, 1], [0, 1]]],
          bars: [["Logistic regression", D.scaling.logistic], ["Decision tree, no depth limit", D.scaling.tree]],
          text: function () { return "Task: mature or not, from length, girth, depth, temperature and month, for 268 made-up spadenose sharks. Scores are the mean of " + D.reps + " random splits. The two pipelines agree to within 0.2 points, so for these models the scaling leak does no visible harm. The rule stays, because the size of the harm depends on the model and the data, and you cannot know in advance."; } },
        { name: "The same shark twice", cols: ["Frames split at random", "Whole sharks split"], frames: true,
          bars: [["Logistic regression", D.duplicates.logistic], ["Decision tree, no depth limit", D.duplicates.tree]],
          text: function () { return "The 268 made-up sharks have 4 frames apiece, with length, girth, depth and temperature re-measured with small errors. Mean of " + D.reps + " random splits. The unlimited tree can tell a shark from its depth, temperature and month, so a test frame of a shark it has already seen is easy: the leaky score is " + pc(D.duplicates.tree.leaky[0]) + " against " + pc(D.duplicates.tree.fair[0]) + " on new sharks. The logistic regression cannot memorise, so the leak does not fool it."; } },
        { name: "Peeking at labels", cols: ["Columns picked with all the labels", "Columns picked inside the training folds"],
          steps: [["Pick the 10 readings that match the labels best", [1, 1], [1, 0]], ["Fit the model", [1, 0], [1, 0]], ["Score the model", [0, 1], [0, 1]]],
          bars: [["Logistic regression", D.peeking]], line50: true,
          text: function () { return "The data are 80 made-up sharks, 200 filler readings with no link to anything, and labels set by coin flips. No reading can predict a coin flip, so a fair score is about 50%. Picking the readings with all the labels first, then cross-validating, reports " + pc(D.peeking.leaky[0]) + ". Mean of " + D.reps + " runs."; } }
      ];
      var tabs = Kit.$("lk-tabs"), tbtn = TABS.map(function (T, i) {
        var b = Kit.html("button", { type: "button", role: "tab", "aria-selected": String(i === 0) }, tabs, T.name);
        b.addEventListener("click", function () { tab = i; draw(); }); return b;
      });
      Kit.chips(Kit.$("lk-mode"), ["Leaky pipeline", "Fair pipeline"], 0, function (i) { leaky = i === 0; draw(); });
      Kit.button(Kit.$("lk-bar"), "Draw another split", function () { seed++; draw(); });
      function framesAssign() {
        var r = rngOf(seed * 7919 + 11), a = new Array(48).fill(0), i, j;
        if (leaky) { var idx = []; for (i = 0; i < 48; i++) idx.push(i); for (i = 47; i > 0; i--) { j = Math.floor(r() * (i + 1)); var t = idx[i]; idx[i] = idx[j]; idx[j] = t; } for (i = 0; i < 12; i++) a[idx[i]] = 1; }
        else { var sh = []; for (i = 0; i < 12; i++) sh.push(i); for (i = 11; i > 0; i--) { j = Math.floor(r() * (i + 1)); var u = sh[i]; sh[i] = sh[j]; sh[j] = u; } for (i = 0; i < 3; i++) for (j = 0; j < 4; j++) a[sh[i] * 4 + j] = 1; }
        return a;
      }
      function drawFrames() {
        var P = Kit.palette(), node = Kit.$("lk-svg"), W = Math.min(Kit.width(node), 520), svg = d3.select(node), a = framesAssign(), cell = 17, H = 14 + 12 * (cell + 4) + 8, both = 0, x0 = 74;
        svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).selectAll("*").remove();
        for (var s = 0; s < 12; s++) {
          var y = 8 + s * (cell + 4), hasT = 0, hasR = 0;
          for (var f = 0; f < 4; f++) { if (a[s * 4 + f]) hasT++; else hasR++; }
          var mixed = hasT > 0 && hasR > 0; if (mixed) both++;
          svg.append("text").attr("x", 0).attr("y", y + cell - 4).attr("font-size", 13).attr("fill", P.ink2).text("shark " + (s + 1));
          for (var f2 = 0; f2 < 4; f2++) svg.append("rect").attr("x", x0 + f2 * (cell + 4)).attr("y", y).attr("width", cell).attr("height", cell).attr("rx", 3).attr("fill", a[s * 4 + f2] ? P.coral : P.navy).attr("opacity", a[s * 4 + f2] ? 1 : 0.75);
          if (mixed) svg.append("text").attr("x", x0 + 4 * (cell + 4) + 6).attr("y", y + cell - 4).attr("font-size", 13).attr("font-weight", 600).attr("fill", P.coral).text("in both sets");
        }
        return both;
      }
      function draw() {
        var T = TABS[tab], P = Kit.palette();
        tbtn.forEach(function (b, i) { b.setAttribute("aria-selected", String(i === tab)); });
        var body = Kit.$("lk-viz"), info = "";
        Kit.$("lk-bar").style.display = T.frames ? "" : "none";
        if (T.frames) {
          body.innerHTML = '<svg id="lk-svg" role="img" aria-label="Twelve made-up sharks with four frames, coloured by whether the frame is in the training set (navy) or the test set (coral)."></svg><p class="mini-note" id="lk-count"></p>';
          var both = drawFrames();
          Kit.setText("lk-count", "Sharks with frames in both sets: " + both + " of 12. Navy: training frame. Coral: test frame.");
        } else {
          var h = '<table class="tbl" aria-label="Which rows the steps of the pipeline look at"><thead><tr><th>Step of the pipeline</th><th>Looks at training rows</th><th>Looks at test rows</th></tr></thead><tbody>';
          T.steps.forEach(function (st) {
            var v = leaky ? st[1] : st[2], leak = leaky && st[1][1] === 1 && st[2][1] === 0 && st[0] === T.steps[0][0];
            h += '<tr><td>' + st[0] + '</td><td>' + (v[0] ? "yes" : "no") + '</td><td' + (leak ? ' class="gap"' : '') + '>' + (v[1] ? "yes" : "no") + (leak ? " (the leak)" : "") + '</td></tr>';
          });
          body.innerHTML = h + '</tbody></table>';
        }
        var cols = [T.cols[0], T.cols[1]], html = "";
        [0, 1].forEach(function (c) {
          var on = (c === 0) === leaky;
          html += '<div class="col" style="' + (on ? "" : "opacity: 0.55") + '"><h4>' + cols[c] + '</h4>';
          T.bars.forEach(function (b) {
            var v = b[1][c === 0 ? "leaky" : "fair"][0];
            html += '<div class="vbar"><span>' + b[0] + '</span><div class="tr"><div class="fl" style="width: ' + (v * 100) + '%; background: ' + (c === 0 ? P.coral : P.teal) + '"></div>' + (T.line50 ? '<div style="position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: ' + P.ink + '"></div>' : '') + '</div><b>' + Kit.fmt(v * 100, 1) + '%</b></div>';
          });
          html += '</div>';
        });
        Kit.$("lk-bars").innerHTML = html;
        Kit.setText("lk-status", T.text() + (T.line50 ? " The black line marks 50%." : ""));
      }
      theme(draw);
    });
  }
  REG.push(["lk", leakFig]);

  // ---------- shared: a confusion matrix drawn as icons ----------
  // Cells are numbered TN 0, FP 1, FN 2, TP 3 (rows: true class, columns: model says), as in scikit-learn.
  var CELLS = [{ k: "TN", name: "true negatives", right: true }, { k: "FP", name: "false positives", right: false }, { k: "FN", name: "false negatives", right: false }, { k: "TP", name: "true positives", right: true }];
  var SCORE_DEF = {
    acc: { name: "accuracy", num: [0, 3], den: [0, 1, 2, 3] },
    prec: { name: "precision", num: [3], den: [1, 3] },
    rec: { name: "recall", num: [3], den: [2, 3] },
    spec: { name: "specificity", num: [0], den: [0, 1] },
    f1: { name: "F1", num: [3], den: [1, 2, 3] },
    fpr: { name: "false positive rate", num: [1], den: [0, 1] }
  };
  function ConfMatrix(node, D, words, onHover) {
    var svg = d3.select(node), dots = [], rects = [], counts = [], geom = null, state = null;
    function build() {
      var P = Kit.palette(), W = Kit.width(node), narrow = W < 520, lw = 22, top = 34, gap = 10, cw = (W - lw - gap) / 2;
      var s = Math.max(4, Math.min(8, (cw - 16) / 60)), ch = Math.max(104, 42 + Math.ceil(D.y.length / Math.max(1, Math.floor((cw - 16) / s))) * s + 12);
      geom = { cw: cw, ch: ch, s: s, lw: lw, top: top, gap: gap, narrow: narrow };
      var H = top + 2 * ch + gap + 4;
      svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).selectAll("*").remove();
      ["model says: not flagged", "model says: flagged"].forEach(function (t, c) {
        svg.append("text").attr("x", lw + c * (cw + gap) + cw / 2).attr("y", 20).attr("text-anchor", "middle").attr("font-size", narrow ? 12.5 : 14).attr("font-weight", 600).attr("fill", P.ink2).text(narrow ? t.replace("model says: ", "") : t);
      });
      [words.neg, words.pos].forEach(function (t, r) {
        var cy = top + r * (ch + gap) + ch / 2;
        svg.append("text").attr("transform", "translate(14," + cy + ") rotate(-90)").attr("text-anchor", "middle").attr("font-size", narrow ? 12 : 13.5).attr("font-weight", 600).attr("fill", P.ink2).text(t);
      });
      rects = []; counts = [];
      for (var c = 0; c < 4; c++) {
        var x = lw + (c % 2) * (cw + gap), y = top + Math.floor(c / 2) * (ch + gap), g = svg.append("g");
        rects.push(g.append("rect").attr("x", x).attr("y", y).attr("width", cw).attr("height", ch).attr("rx", 8).attr("fill", P.paper));
        counts.push(g.append("text").attr("x", x + 10).attr("y", y + 27).attr("font-size", 23).attr("font-weight", 700).attr("fill", CELLS[c].right ? P.teal : P.coral));
      }
      dots = [];
      for (var i = 0; i < D.y.length; i++) {
        var d = svg.append("circle").attr("r", s * 0.38).attr("fill", D.y[i] ? P.coral : P.navy).attr("cursor", "pointer");
        (function (i) { d.on("mouseenter", function () { if (onHover) onHover(i, state && state.cellOf[i]); }); })(i);
        dots.push(d);
      }
      if (state) update(state.t, state.hi);
    }
    function update(t, hi) {
      var P = Kit.palette(), g = geom, cellOf = new Array(D.y.length), fill = [0, 0, 0, 0], n = [0, 0, 0, 0], i;
      for (i = 0; i < D.y.length; i++) { var f = D.z[i] >= t ? 1 : 0, c = (D.y[i] ? 2 : 0) + f; cellOf[i] = c; n[c]++; }
      var cols = Math.max(1, Math.floor((g.cw - 16) / g.s));
      for (i = 0; i < D.y.length; i++) {
        var c2 = cellOf[i], k = fill[c2]++, x = g.lw + (c2 % 2) * (g.cw + g.gap) + 8, y = g.top + Math.floor(c2 / 2) * (g.ch + g.gap) + 42;
        var inv = !hi || SCORE_DEF[hi].num.indexOf(c2) >= 0 || SCORE_DEF[hi].den.indexOf(c2) >= 0;
        dots[i].attr("cx", x + (k % cols + 0.5) * g.s).attr("cy", y + (Math.floor(k / cols) + 0.5) * g.s).attr("opacity", inv ? 0.95 : 0.22);
      }
      for (var c3 = 0; c3 < 4; c3++) {
        var isNum = hi && SCORE_DEF[hi].num.indexOf(c3) >= 0, isDen = hi && SCORE_DEF[hi].den.indexOf(c3) >= 0 && !isNum, col = CELLS[c3].right ? P.teal : P.coral;
        rects[c3].attr("fill", isNum ? P.mark : P.paper).attr("stroke", isNum ? P.teal : (isDen ? P.ink : col)).attr("stroke-width", isNum ? 3.2 : (isDen ? 2 : 1.4)).attr("stroke-dasharray", isDen ? "6 4" : null).attr("opacity", (!hi || isNum || isDen) ? 1 : 0.6);
        counts[c3].text(n[c3] + (g.narrow ? "  " + CELLS[c3].k : "")).attr("fill", col);
      }
      svg.selectAll(".cap").remove();
      if (!g.narrow) for (var c4 = 0; c4 < 4; c4++) svg.append("text").attr("class", "cap").attr("x", g.lw + (c4 % 2) * (g.cw + g.gap) + 14 + 14 * String(n[c4]).length).attr("y", g.top + Math.floor(c4 / 2) * (g.ch + g.gap) + 26).attr("font-size", 13).attr("fill", P.ink2).text(CELLS[c4].k + ", " + CELLS[c4].name);
      state = { t: t, hi: hi, cellOf: cellOf, n: n };
      return { tn: n[0], fp: n[1], fn: n[2], tp: n[3] };
    }
    return { build: build, update: update, get: function () { return state; } };
  }
  function scoreBoxes(c, r) {
    return [["acc", r.acc], ["prec", r.prec], ["rec", r.rec], ["spec", r.spec], ["f1", r.f1]];
  }
  function fracBoxes(key, c) {
    var D = SCORE_DEF[key], r = rates(c), teal = "c-teal", num, den, val, nl, dl;
    if (key === "acc") { return box("c-teal", String(c.tp + c.tn), "TP + TN: right calls") + op("&divide;") + box("", String(r.T), "TP + FP + TN + FN: all records") + op("=") + box("c-sand", pc(r.acc), "accuracy"); }
    if (key === "prec") { return box("c-teal", String(c.tp), "TP: flagged and positive") + op("&divide;") + box("", String(r.F), "TP + FP: all flagged") + op("=") + box("c-sand", pc(r.prec), "precision"); }
    if (key === "rec") { return box("c-teal", String(c.tp), "TP: flagged and positive") + op("&divide;") + box("", String(r.P), "TP + FN: all positives") + op("=") + box("c-sand", pc(r.rec), "recall"); }
    if (key === "spec") { return box("c-teal", String(c.tn), "TN: not flagged and negative") + op("&divide;") + box("", String(r.N), "TN + FP: all negatives") + op("=") + box("c-sand", pc(r.spec), "specificity"); }
    if (key === "fpr") { return box("c-coral", String(c.fp), "FP: flagged and negative") + op("&divide;") + box("", String(r.N), "TN + FP: all negatives") + op("=") + box("c-sand", pc(r.fpr), "false positive rate"); }
    return box("c-teal", String(2 * c.tp), "2 &times; TP") + op("&divide;") + box("", String(2 * c.tp + c.fp + c.fn), "2 &times; TP + FP + FN") + op("=") + box("c-sand", pc(r.f1), "F1");
  }

  // ---------- 5. the confusion matrix and its scores ----------
  function matrixFig() {
    return load("data/scores.json").then(function (S) {
      var D = { z: S.hammer.z5, y: S.hammer.y }, TAU = 0.30, pick = "rec", node = Kit.$("cm-svg");
      var M = ConfMatrix(node, D, { neg: "not a hammerhead", pos: "hammerhead" }, function (i, c) {
        var sc = sigm(D.z[i]);
        Kit.setText("cm-status", "Tag " + S.ids[i] + ", " + NAMES[S.species[i]] + " shark, " + Kit.fmt(S.length[i], 1) + " cm, score " + Kit.fmt(sc, sc < 0.01 ? 3 : 2) + ": " + ["true negative (not flagged, not a hammerhead)", "false positive (flagged, but not a hammerhead)", "false negative (not flagged, but a hammerhead)", "true positive (flagged, and a hammerhead)"][c] + ".");
      });
      function draw() {
        var c = M.update(logit(TAU), pick), r = rates(c), P = Kit.palette();
        var keys = [["acc", "accuracy"], ["prec", "precision"], ["rec", "recall"], ["spec", "specificity"], ["f1", "F1"]], h = "";
        keys.forEach(function (k) { h += '<div role="button" tabindex="0" class="btn-like' + (k[0] === pick ? " on" : "") + '" data-k="' + k[0] + '" aria-pressed="' + (k[0] === pick) + '"><b>' + pc(r[k[0]]) + '</b><span>' + k[1] + '</span></div>'; });
        var sc = Kit.$("cm-scores"); sc.innerHTML = h;
        sc.querySelectorAll("div").forEach(function (d) { function go() { pick = d.getAttribute("data-k"); draw(); } d.addEventListener("click", go); d.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } }); });
        Kit.$("cm-eqn").innerHTML = fracBoxes(pick, c);
        Kit.setText("cm-thr", "The detector flags a record when its score is at least " + Kit.fmt(TAU, 2) + ". Scores come from a logistic regression on five measurements, out of fold (records are scored by a model that never saw them).");
      }
      theme(function () { M.build(); draw(); });
    });
  }
  REG.push(["cm", matrixFig]);

  // ---------- 5b. F1 against the plain average ----------
  function f1Fig() {
    var pr = 57 / 600, re = 1;
    var sP = Kit.slider(Kit.$("f1-controls"), { label: "Precision", min: 0.5, max: 100, step: 0.5, value: 9.5, show: function (v) { return Kit.fmt(v, 1) + "%"; } }, function (v) { pr = v / 100; draw(); });
    var sR = Kit.slider(Kit.$("f1-controls"), { label: "Recall", min: 0.5, max: 100, step: 0.5, value: 100, show: function (v) { return Kit.fmt(v, 1) + "%"; } }, function (v) { re = v / 100; draw(); });
    function set(p, r) { pr = p; re = r; sP.set(p * 100); sR.set(r * 100); draw(); }
    Kit.button(Kit.$("f1-bar"), "Flag all 600 records", function () { set(57 / 600, 1); });
    Kit.button(Kit.$("f1-bar"), "Flag only the two surest", function () { set(1, 2 / 57); });
    Kit.button(Kit.$("f1-bar"), "A balanced detector", function () { set(0.6, 0.5); });
    function draw() {
      var P = Kit.palette(), mean = (pr + re) / 2, f1 = 2 * pr * re / (pr + re), lo = Math.min(pr, re);
      function bar(name, v, col, note) { return '<div class="vbar"><span>' + name + '</span><div class="tr"><div class="fl" style="width: ' + (v * 100) + '%; background: ' + col + '"></div></div><b>' + Kit.fmt(v * 100, 1) + '%</b></div>'; }
      Kit.$("f1-bars").innerHTML = bar("plain average", mean, P.sand) + bar("F1", f1, P.teal) + bar("the weaker of the two", lo, P.muted);
      Kit.$("f1-eqn").innerHTML = op("2 &times;") + box("c-navy", Kit.fmt(pr, 3), "precision") + op("&times;") + box("c-coral", Kit.fmt(re, 3), "recall") + op("&divide; (") + box("c-navy", Kit.fmt(pr, 3), "precision") + op("+") + box("c-coral", Kit.fmt(re, 3), "recall") + op(") =") + box("c-teal", Kit.fmt(f1, 3), "F1");
      Kit.setText("f1-status", mean - f1 > 0.15 ? "The plain average is " + pc(mean) + " but F1 is " + pc(f1) + ". F1 stays close to the weaker of the two numbers, so one weak side cannot hide." : "When precision and recall are close, F1 and the plain average are close.");
    }
    draw();
    Kit.onTheme(draw);
  }
  REG.push(["f1", f1Fig]);

  // ---------- shared: ROC drawing ----------
  function rocPlot(id, curve, auc, marker, extra) {
    Kit.plot(id, function (P) {
      var tr = [
        { x: [0, 1], y: [0, 1], mode: "lines", line: { color: P.muted, width: 1.5, dash: "dash" }, hoverinfo: "skip" },
        { x: curve.fpr, y: curve.tpr, mode: "lines", line: { color: P.teal, width: 3, shape: "hv" }, fill: "tozeroy", fillcolor: P.mark, hoverinfo: "skip" },
        { x: [marker.fpr], y: [marker.tpr], mode: "markers", marker: { color: P.coral, size: 13, line: { color: P.paper, width: 2 } }, hovertemplate: "false positive rate %{x:.3f}<br>true positive rate %{y:.3f}<extra></extra>" }
      ];
      return extra && extra.traces ? tr.concat(extra.traces(P)) : tr;
    }, function (P) {
      return { xaxis: { title: { text: "false positive rate (1 minus specificity)" }, range: [0, 1], dtick: 0.25 }, yaxis: { title: { text: "true positive rate (recall)" }, range: [0, 1.02], dtick: 0.25 },
        margin: { l: 56, r: 12, t: 12, b: 48 },
        annotations: [
          { x: 0.97, y: 0.06, xref: "x", yref: "y", xanchor: "right", yanchor: "bottom", showarrow: false, text: "AUC " + Kit.fmt(auc, 3), font: { size: 15, color: P.ink }, bgcolor: P.paper, borderpad: 3 },
          { x: 0.5, y: 0.46, xref: "x", yref: "y", showarrow: false, text: "coin flip", textangle: -38, font: { size: 12.5, color: P.muted } }
        ] };
    });
  }

  // ---------- shared: the score strip with a draggable threshold ----------
  function Strip(node, D, X, onSet) {
    var svg = d3.select(node), dots = [], jitter = [], r = rngOf(31), st = { t: 0, ratio: 5 }, G = null, i;
    for (i = 0; i < D.z.length; i++) jitter.push(r());
    function build() {
      var P = Kit.palette(), W = Kit.width(node), mL = 12, mR = 12, laneH = 62;
      var yRib = 22 + 2 * laneH + 6 + 66, H = yRib + 56;
      G = { W: W, mL: mL, mR: mR, laneH: laneH, H: H, x: function (z) { return mL + (z - X[0]) / (X[1] - X[0]) * (W - mL - mR); }, zOf: function (px) { return X[0] + (px - mL) / (W - mL - mR) * (X[1] - X[0]); }, yAxis: 22 + 2 * laneH + 6, yRib: yRib };
      svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).style("touch-action", "pan-y").selectAll("*").remove();
      [["hammerheads (" + sum(D.y) + ")", 0], ["other sharks (" + (D.y.length - sum(D.y)) + ")", 1]].forEach(function (l) {
        svg.append("text").attr("x", mL + 2).attr("y", 22 + l[1] * laneH + 13).attr("font-size", 12.5).attr("fill", l[1] ? P.navy : P.coral).attr("font-weight", 600).text(l[0]);
        svg.append("line").attr("x1", mL).attr("x2", W - mR).attr("y1", 22 + l[1] * laneH + laneH / 2 + 4).attr("y2", 22 + l[1] * laneH + laneH / 2 + 4).attr("stroke", P.rule);
      });
      svg.append("rect").attr("class", "flagrect").attr("y", 18).attr("height", 2 * laneH + 6).attr("fill", P.mark);
      svg.append("line").attr("x1", mL).attr("x2", W - mR).attr("y1", G.yAxis).attr("y2", G.yAxis).attr("stroke", P.muted);
      [0.001, 0.01, 0.1, 0.5, 0.9, 0.99].forEach(function (p) {
        var z = logit(p); if (z < X[0] || z > X[1]) return;
        svg.append("line").attr("x1", G.x(z)).attr("x2", G.x(z)).attr("y1", G.yAxis).attr("y2", G.yAxis + 5).attr("stroke", P.muted);
        svg.append("text").attr("x", G.x(z)).attr("y", G.yAxis + 19).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.muted).text(p);
      });
      svg.append("text").attr("x", W - mR).attr("y", G.yAxis + 36).attr("text-anchor", "end").attr("font-size", 12.5).attr("fill", P.muted).text(W < 520 ? "score, spaced by log-odds" : "score (probability of hammerhead, spaced by log-odds)");
      dots = [];
      for (i = 0; i < D.z.length; i++) dots.push(svg.append("circle").attr("r", 2.9).attr("cx", G.x(D.z[i])).attr("cy", 22 + (D.y[i] ? 0 : 1) * laneH + 12 + 8 + jitter[i] * (laneH - 26)).attr("fill", D.y[i] ? P.coral : P.navy).attr("opacity", 0.7));
      svg.append("g").attr("class", "rib");
      svg.append("line").attr("class", "tline").attr("y1", 14).attr("y2", G.yAxis).attr("stroke", P.ink).attr("stroke-width", 2.5);
      svg.append("circle").attr("class", "thandle").attr("cy", 12).attr("r", 8).attr("fill", P.ink).attr("stroke", P.paper).attr("stroke-width", 2).attr("cursor", "ew-resize");
      svg.append("text").attr("class", "tlabel").attr("y", 17).attr("font-size", 13).attr("font-weight", 600).attr("fill", P.ink);
      var down = false;
      function at(ev) { var rect = node.getBoundingClientRect(), px = (ev.clientX - rect.left) * (W / rect.width); onSet(Math.max(X[0] + 0.2, Math.min(X[1] - 0.2, G.zOf(px)))); }
      svg.on("pointerdown", function (ev) { down = true; try { node.setPointerCapture(ev.pointerId); } catch (e) {} at(ev); });
      svg.on("pointermove", function (ev) { if (down) at(ev); });
      svg.on("pointerup pointercancel", function () { down = false; });
      if (st) update(st.t, st.costs, st.best);
    }
    // costs: {fn: [..], fp: [..]} grid arrays are drawn as a ribbon; best is the cheapest z
    function update(t, ribbon, best) {
      var P = Kit.palette(); st = { t: t, costs: ribbon, best: best };
      svg.select(".flagrect").attr("x", G.x(t)).attr("width", Math.max(0, G.W - G.mR - G.x(t)));
      svg.select(".tline").attr("x1", G.x(t)).attr("x2", G.x(t));
      svg.select(".thandle").attr("cx", G.x(t));
      var lab = Kit.fmt(sigm(t), sigm(t) < 0.01 ? 3 : 2);
      var tl = svg.select(".tlabel").text("flag if score is at least " + lab); var tx = G.x(t); tl.attr("text-anchor", tx < G.W / 2 ? "start" : "end").attr("x", tx < G.W / 2 ? tx + 15 : tx - 15);
      for (var k = 0; k < dots.length; k++) { var f = D.z[k] >= t, err = f !== !!D.y[k]; dots[k].attr("stroke", err ? P.ink : "none").attr("stroke-width", err ? 1.4 : 0).attr("opacity", err ? 1 : 0.7); }
      var rib = svg.select(".rib"); rib.selectAll("*").remove();
      if (ribbon) {
        var mx = Math.max.apply(null, ribbon.c), pts = ribbon.z.map(function (z, j) { return [G.x(z), G.yRib + 44 - 40 * ribbon.c[j] / mx]; });
        rib.append("text").attr("x", G.mL + 2).attr("y", G.yRib - 4).attr("font-size", 12.5).attr("fill", P.muted).text("total cost of the errors, by threshold (lower is better)");
        rib.append("path").attr("d", d3.line()(pts)).attr("fill", "none").attr("stroke", P.sand).attr("stroke-width", 2.5);
        var ci = 0; for (var q = 1; q < ribbon.z.length; q++) if (Math.abs(ribbon.z[q] - t) < Math.abs(ribbon.z[ci] - t)) ci = q;
        rib.append("circle").attr("cx", pts[ci][0]).attr("cy", pts[ci][1]).attr("r", 5).attr("fill", P.coral);
        if (best != null) rib.append("path").attr("d", "M" + G.x(best) + "," + (G.yRib + 40) + " l-6,10 l12,0 z").attr("fill", P.teal);
      }
    }
    return { build: build, update: update };
  }

  // ---------- 6. moving the threshold ----------
  function thresholdFig() {
    return load("data/scores.json").then(function (S) {
      var D = { z: S.hammer.z5, y: S.hammer.y }, curve = roc(D.z, D.y), AUC = trapz(curve.fpr, curve.tpr), t = logit(0.30), ratio = 5, pick = "rec";
      var Mx = ConfMatrix(Kit.$("th-svg"), D, { neg: "not a hammerhead", pos: "hammerhead" }, null);
      var St = Strip(Kit.$("th-strip"), D, [-9, 6.2], function (z) { t = z; sT.set(z); draw(); });
      var sT = Kit.slider(Kit.$("th-controls"), { label: "Threshold (as a probability)", min: -6.5, max: 4.5, step: 0.05, value: t, show: function (v) { var p = sigm(v); return Kit.fmt(p, p < 0.01 ? 3 : 2) + "  (log-odds " + Kit.fmt(v, 2) + ")"; } }, function (v) { t = v; draw(); });
      var sC = Kit.slider(Kit.$("th-cost"), { label: "A missed hammerhead costs this many false alarms", min: 1, max: 30, step: 1, value: ratio, digits: 0 }, function (v) { ratio = v; draw(); });
      Kit.button(Kit.$("th-bar"), "Jump to the cheapest threshold", function () { var b = cheapest(); t = b.z; sT.set(t); draw(); });
      [0.5, 0.3, 0.1].forEach(function (p) { Kit.button(Kit.$("th-bar"), "Threshold " + Kit.fmt(p, 2), function () { t = logit(p); sT.set(t); draw(); }); });
      function cheapest() {
        var best = null; for (var k = 1; k < curve.thr.length; k++) { var fp = curve.fpr[k] * curve.N, fn = curve.P - curve.tpr[k] * curve.P, cst = ratio * fn + fp; if (best === null || cst < best.cost) best = { z: curve.thr[k], cost: cst, fp: fp, fn: fn }; }
        return best;
      }
      function ribbon() {
        var z = [], c = []; for (var g = -9; g <= 6.2; g += 0.1) { var k = count(D.z, D.y, g); z.push(g); c.push(ratio * k.fn + k.fp); } return { z: z, c: c };
      }
      function draw() {
        var c = Mx.update(t, pick), r = rates(c), best = cheapest(), cc = count(D.z, D.y, t);
        St.update(t, ribbon(), best.z);
        rocPlot("th-roc", curve, AUC, { fpr: r.fpr, tpr: r.rec });
        var keys = [["acc", "accuracy"], ["prec", "precision"], ["rec", "recall"], ["spec", "specificity"], ["f1", "F1"]], h = "";
        keys.forEach(function (k) { h += '<div role="button" tabindex="0" class="btn-like' + (k[0] === pick ? " on" : "") + '" data-k="' + k[0] + '" aria-pressed="' + (k[0] === pick) + '"><b>' + pc(r[k[0]]) + '</b><span>' + k[1] + '</span></div>'; });
        var sc = Kit.$("th-scores"); sc.innerHTML = h;
        sc.querySelectorAll("div").forEach(function (d) { function go() { pick = d.getAttribute("data-k"); draw(); } d.addEventListener("click", go); d.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } }); });
        var cost = ratio * c.fn + c.fp, formula = 1 / (1 + ratio);
        Kit.$("th-eqn").innerHTML = box("c-coral", String(c.fn), "misses (FN)") + op("&times;") + box("c-sand", String(ratio), "cost of a miss") + op("+") + box("c-navy", String(c.fp), "false alarms (FP)") + op("&times; 1 =") + box("c-teal", String(cost), "total cost now");
        Kit.$("th-eqn2").innerHTML = box("c-teal", Kit.fmt(formula, 3), "threshold from the formula 1 &divide; (1 + " + ratio + ")") + box("c-sand", Kit.fmt(sigm(best.z), 3), "cheapest on these records (cost " + Math.round(best.cost) + ")");
        Kit.setText("th-status", "At threshold " + Kit.fmt(sigm(t), sigm(t) < 0.01 ? 3 : 2) + " the detector flags " + (c.tp + c.fp) + " records: " + c.tp + " hammerheads and " + c.fp + " other sharks. It misses " + c.fn + ".");
      }
      theme(function () { Mx.build(); St.build(); draw(); });
    });
  }
  REG.push(["th", thresholdFig]);

  // ---------- 7. ROC curve and AUC ----------
  function rocFig() {
    return load("data/scores.json").then(function (S) {
      var y = S.hammer.y, n = y.length, r0 = rngOf(2024), coin = [], perfect = [], i;
      for (i = 0; i < n; i++) { coin.push(Kit.gauss(r0)); perfect.push(y[i] ? 3 : -3); }
      var SC = [
        { name: "Five measurements", z: S.hammer.z5 },
        { name: "Length only", z: S.hammer.z1 },
        { name: "Coin flip", z: coin },
        { name: "Perfect", z: perfect },
        { name: "Upside down", z: S.hammer.z5.map(function (v) { return -v; }) }
      ];
      SC.forEach(function (s) { s.curve = roc(s.z, y); s.auc = trapz(s.curve.fpr, s.curve.tpr); s.pairs = aucPairs(s.z, y); s.lo = Math.min.apply(null, s.z) - 0.4; s.hi = Math.max.apply(null, s.z) + 0.4; });
      var cur = 0, t = logit(0.3), tally = { n: 0, w: 0 }, last = null, seed = 1, node = Kit.$("rc-line"), svg = d3.select(node);
      Kit.chips(Kit.$("rc-model"), SC.map(function (s) { return s.name; }), 0, function (k) { cur = k; tally = { n: 0, w: 0 }; last = null; clampT(); sT.set(t); draw(); });
      var sT = Kit.slider(Kit.$("rc-controls"), { label: "Threshold (log-odds of the score)", min: -9, max: 6.2, step: 0.05, value: t, show: function (v) { return Kit.fmt(v, 2); } }, function (v) { t = v; draw(); });
      function clampT() { var s = SC[cur]; t = Math.max(s.lo, Math.min(s.hi, t)); }
      Kit.button(Kit.$("rc-btns"), "Draw a pair", function () { drawPairs(1); });
      Kit.button(Kit.$("rc-btns"), "Draw 100 pairs", function () { drawPairs(100); });
      Kit.button(Kit.$("rc-btns"), "Check all pairs", function () { var p = SC[cur].pairs; tally = { n: p.total, w: p.wins + p.ties / 2, all: true }; last = null; draw(); });
      Kit.button(Kit.$("rc-btns"), "Clear", function () { tally = { n: 0, w: 0 }; last = null; draw(); });
      function drawPairs(k) {
        var s = SC[cur], pos = [], neg = [], r = rngOf(seed++ * 7919 + 3);
        for (i = 0; i < n; i++) (y[i] ? pos : neg).push(i);
        if (tally.all) tally = { n: 0, w: 0 };
        for (var j = 0; j < k; j++) {
          var a = pos[Math.floor(r() * pos.length)], b = neg[Math.floor(r() * neg.length)], d = s.z[a] - s.z[b];
          tally.n++; tally.w += d > 0 ? 1 : (d === 0 ? 0.5 : 0); last = [a, b];
        }
        draw();
      }
      function drawLine() {
        var P = Kit.palette(), s = SC[cur], W = Kit.width(node), H = 118, mL = 10, mR = 10, x = function (z) { return mL + (z - s.lo) / (s.hi - s.lo) * (W - mL - mR); }, raw = cur === 2 || cur === 3;
        svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).selectAll("*").remove();
        svg.append("line").attr("x1", mL).attr("x2", W - mR).attr("y1", 54).attr("y2", 54).attr("stroke", P.muted);
        [0.001, 0.01, 0.1, 0.5, 0.9, 0.99].forEach(function (p) { var z = logit(p); if (z < s.lo || z > s.hi || raw) return; svg.append("line").attr("x1", x(z)).attr("x2", x(z)).attr("y1", 54).attr("y2", 59).attr("stroke", P.muted); svg.append("text").attr("x", x(z)).attr("y", 98).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", P.muted).text(p); });
        svg.append("text").attr("x", W - mR).attr("y", 113).attr("text-anchor", "end").attr("font-size", 12).attr("fill", P.muted).text(raw ? "score (made-up scale)" : "score, probability scale");
        if (last) {
          var za = s.z[last[0]], zb = s.z[last[1]];
          svg.append("circle").attr("cx", x(zb)).attr("cy", 54).attr("r", 9).attr("fill", P.navy).attr("stroke", P.paper).attr("stroke-width", 2);
          svg.append("circle").attr("cx", x(za)).attr("cy", 54).attr("r", 9).attr("fill", P.coral).attr("stroke", P.paper).attr("stroke-width", 2).attr("transform", za === zb ? "translate(0,-14)" : null);
          svg.append("text").attr("x", Math.max(36, Math.min(W - 36, x(za)))).attr("y", 26).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 600).attr("fill", P.coral).text("hammerhead");
          svg.append("text").attr("x", Math.max(40, Math.min(W - 40, x(zb)))).attr("y", 82).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 600).attr("fill", P.navy).text("other shark");
        } else svg.append("text").attr("x", W / 2).attr("y", 36).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.muted).text("Draw a pair: one hammerhead and one other shark");
      }
      function draw() {
        var s = SC[cur], c = count(s.z, y, t), r = rates(c);
        rocPlot("rc-plot", s.curve, s.auc, { fpr: r.fpr, tpr: r.rec });
        drawLine();
        var P = s.pairs;
        Kit.$("rc-eqn").innerHTML = box("c-teal", commas(Math.floor(P.wins + P.ties / 2)) + ((P.ties % 2) ? ".5" : ""), "pairs where the hammerhead scores higher (ties count half)") + op("&divide;") + box("", commas(P.total), "all pairs: " + P.total / 543 + " &times; 543") + op("=") + box("c-sand", Kit.fmt(P.auc, 3), "AUC, the same as the area under the curve");
        var lastTxt = "";
        if (last) { var za = s.z[last[0]], zb = s.z[last[1]]; lastTxt = "This pair: the hammerhead scored " + (cur === 2 || cur === 3 ? Kit.fmt(za, 2) : Kit.fmt(sigm(za), 3)) + " and the other shark " + (cur === 2 || cur === 3 ? Kit.fmt(zb, 2) : Kit.fmt(sigm(zb), 3)) + ", so the hammerhead " + (za > zb ? "ranks higher." : (za === zb ? "ties." : "ranks lower.")) + " "; }
        Kit.setText("rc-tally", lastTxt + (tally.n ? (tally.all ? "All " + commas(tally.n) + " pairs: " : "Hammerhead higher in " + commas(tally.w) + " of " + commas(tally.n) + " pairs: ") + Kit.pct(tally.w / tally.n, 1) + "." : "No pairs drawn yet."));
        Kit.setText("rc-status", s.name + ": at this threshold the true positive rate is " + pc(r.rec) + " and the false positive rate is " + pc(r.fpr) + ". AUC " + Kit.fmt(s.auc, 3) + ".");
      }
      theme(function () { draw(); });
    });
  }
  REG.push(["rc", rocFig]);

  // ---------- 8. rare classes ----------
  function rareFig() {
    return load("data/scores.json").then(function (S) {
      var CL = [{ name: "Ganges shark", pl: "Ganges sharks", y: S.ganges.y, z: S.ganges.z5, lo: -8, hi: 1 }, { name: "hammerhead", pl: "hammerheads", y: S.hammer.y, z: S.hammer.z5, lo: -6, hi: 4 }];
      CL.forEach(function (c) { c.curve = roc(c.z, c.y); c.P = sum(c.y); c.N = c.y.length - c.P; c.auc = trapz(c.curve.fpr, c.curve.tpr); c.order = []; var i; for (i = 0; i < c.y.length; i++) if (c.y[i]) c.order.push(i); for (i = 0; i < c.y.length; i++) if (!c.y[i]) c.order.push(i); });
      var cls = 0, model = 2, node = Kit.$("ra-svg"), svg = d3.select(node), dots = [], geom = null, t = logit(0.1);
      Kit.chips(Kit.$("ra-class"), ["Ganges shark: 3 of 600", "Hammerhead: 57 of 600"], 0, function (i) { cls = i; setSlider(); build(); });
      Kit.chips(Kit.$("ra-model"), ["Never flags", "Flags all records", "Logistic regression"], 2, function (i) { model = i; sT.input.disabled = model !== 2; draw(); });
      var sT = Kit.slider(Kit.$("ra-controls"), { label: "Threshold of the logistic regression", min: -8, max: 1, step: 0.05, value: t, show: function (v) { var p = sigm(v); return Kit.fmt(p, p < 0.01 ? 4 : 2); } }, function (v) { t = v; draw(); });
      function setSlider() { var c = CL[cls]; sT.input.min = c.lo; sT.input.max = c.hi; t = Math.max(c.lo, Math.min(c.hi, cls === 0 ? logit(0.1) : logit(0.3))); sT.set(t); }
      function flagged(i) { var c = CL[cls]; return model === 0 ? false : (model === 1 ? true : c.z[i] >= t); }
      function build() {
        var c = CL[cls], P = Kit.palette(), W = Kit.width(node), cols = Math.max(24, Math.min(60, Math.round(W / 10.5))), s = W / cols, rows = Math.ceil(600 / cols), H = rows * s + 8;
        geom = { s: s, cols: cols, H: H };
        svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).selectAll("*").remove(); dots = [];
        c.order.forEach(function (rec, k) { dots.push({ rec: rec, el: svg.append("circle").attr("cx", (k % cols + 0.5) * s).attr("cy", (Math.floor(k / cols) + 0.5) * s) }); });
        draw();
      }
      function draw() {
        var c = CL[cls], P = Kit.palette(), g = geom, k = { tp: 0, fp: 0, tn: 0, fn: 0 };
        dots.forEach(function (d) { var f = flagged(d.rec), pos = c.y[d.rec]; if (pos) { if (f) k.tp++; else k.fn++; } else { if (f) k.fp++; else k.tn++; }
          d.el.attr("r", pos ? g.s * 0.5 : g.s * 0.34 + (f ? 0.6 : 0)).attr("fill", pos ? P.coral : P.navy).attr("stroke", f ? P.ink : "none").attr("stroke-width", f ? 1.6 : 0).attr("opacity", pos ? 1 : (f ? 1 : 0.55)); });
        var r = rates(k), lazy = (600 - c.P) / 600, base = c.P / 600, F = k.tp + k.fp;
        var boxes = [["acc", "accuracy"], ["rec", "recall"], ["prec", "precision"], ["f1", "F1"]], h = "";
        boxes.forEach(function (b) { h += '<div class="' + (b[0] === "rec" || b[0] === "prec" ? "hit" : "") + '"><b>' + pc(r[b[0]]) + '</b><span>' + b[1] + (b[0] === "prec" && r.prec == null ? " (0 of 0 flagged)" : "") + '</span></div>'; });
        Kit.$("ra-scores").innerHTML = h;
        Kit.$("ra-eqn").innerHTML = box("c-teal", String(k.tp + k.tn), "right calls: TP + TN") + op("&divide;") + box("", "600", "all records") + op("=") + box("c-sand", pc(r.acc, 1), "accuracy");
        Kit.$("ra-eqn2").innerHTML = box("c-coral", String(k.tp), "TP: " + c.pl + " found") + op("&divide;") + box("", String(c.P), "all " + c.pl) + op("=") + box("c-sand", pc(r.rec, 1), "recall");
        var wi = wilson(k.tp, c.P), txt = ["Never flags: accuracy " + pc(r.acc) + ", the same as the lazy model, and recall " + pc(r.rec) + ".", "Flags all records: recall is 100% and precision is " + pc(base) + ", the base rate.", "Logistic regression at threshold " + Kit.fmt(sigm(t), sigm(t) < 0.01 ? 4 : 2) + ": flags " + F + ", of which " + k.tp + " are " + c.pl + "."][model];
        Kit.setText("ra-status", txt + " Recall " + k.tp + " of " + c.P + " has a 95% interval of " + pc(wi[0], 0) + " to " + pc(wi[1], 0) + ". The lazy model scores " + pc(lazy) + " on accuracy.");
        // precision-recall plot
        var cv = c.curve, rx = cv.tpr.slice(1), py = cv.prec.slice(1), pt = (r.prec == null) ? null : { x: r.rec, y: r.prec };
        Kit.plot("ra-pr", function (P2) {
          var tr = [
            { x: [0, 1], y: [base, base], mode: "lines", line: { color: P2.muted, width: 1.6, dash: "dash" }, hoverinfo: "skip" },
            { x: rx, y: py, mode: "lines+markers", line: { color: P2.teal, width: 2.5 }, marker: { color: P2.teal, size: 5 }, hovertemplate: "recall %{x:.2f}, precision %{y:.2f}<extra></extra>" }
          ];
          if (pt) tr.push({ x: [pt.x], y: [pt.y], mode: "markers", marker: { color: P2.coral, size: 14, line: { color: P2.paper, width: 2 } }, hoverinfo: "skip" });
          return tr;
        }, function (P2) {
          return { xaxis: { title: { text: "recall" }, range: [-0.02, 1.02], dtick: 0.25 }, yaxis: { title: { text: "precision" }, range: [-0.02, 1.05], dtick: 0.25 }, margin: { l: 52, r: 12, t: 12, b: 46 },
            annotations: [{ x: 0.98, y: base, xref: "x", yref: "y", xanchor: "right", yanchor: "bottom", yshift: 4, showarrow: false, text: "random flagging: " + pc(base), font: { size: 12.5, color: P2.muted } }].concat(pt ? [] : [{ x: 0.5, y: 0.6, xref: "x", yref: "y", showarrow: false, text: "no records flagged: precision is 0 of 0", font: { size: 13.5, color: P2.coral } }]) };
        });
      }
      setSlider(); sT.input.disabled = false;
      theme(function () { build(); });
    });
  }
  REG.push(["ra", rareFig]);

  // ---------- 9. base rates in screening ----------
  function screenFig() {
    return load("data/clinic.json").then(function (C) {
      var n = C.y.length, node = Kit.$("sc-svg"), svg = d3.select(node), N = 1000, icons = [], tau = 0.5, prev = 34.9;
      var sP = Kit.slider(Kit.$("sc-controls"), { label: "Share of women who have diabetes where the model is used (prevalence)", min: 1, max: 50, step: 0.1, value: prev, show: function (v) { return Kit.fmt(v, 1) + "%"; } }, function (v) { prev = v; draw(); });
      var sT = Kit.slider(Kit.$("sc-controls"), { label: "Threshold of the model", min: 0.2, max: 0.8, step: 0.01, value: tau, show: function (v) { return Kit.fmt(v, 2); } }, function (v) { tau = v; draw(); });
      Kit.chips(Kit.$("sc-preset"), ["The Pima sample (34.9%)", "A town at 10% (made up)", "A town at 2% (made up)"], 0, function (i) { prev = [34.9, 10, 2][i]; sP.set(prev); draw(); });
      function operating() { var c = count(C.cv, C.y, logit(tau)); return rates(c); }
      function counts(r) { var sick = Math.round(N * prev / 100), tp = Math.round(sick * r.rec), fn = sick - tp, healthy = N - sick, fp = Math.round(healthy * (1 - r.spec)), tn = healthy - fp; return { sick: sick, tp: tp, fn: fn, fp: fp, tn: tn }; }
      function kind(k, c) { return k < c.tp ? 0 : k < c.sick ? 1 : k < c.sick + c.fp ? 2 : 3; }
      function build() {
        var P = Kit.palette(), W = Kit.width(node), cols = 40, cell = W / cols, H = N / cols * cell + 4;
        svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).selectAll("*").remove(); icons = [];
        for (var k = 0; k < N; k++) icons.push(svg.append("circle").attr("cx", (k % cols + 0.5) * cell).attr("cy", (Math.floor(k / cols) + 0.5) * cell).attr("r", Math.max(2.2, cell * 0.36)));
        draw();
      }
      function draw() {
        var P = Kit.palette(), r = operating(), c = counts(r), F = c.tp + c.fp, ppv = F ? c.tp / F : 0, npv = (c.tn + c.fn) ? c.tn / (c.tn + c.fn) : 0, p = prev / 100;
        icons.forEach(function (ic, k) { var t = kind(k, c), sick = t < 2, fl = t === 0 || t === 2, col = sick ? P.coral : P.navy; ic.attr("fill", fl ? col : "none").attr("stroke", col).attr("stroke-width", 1.5).attr("opacity", fl ? 0.95 : 0.7); });
        Kit.$("sc-counts").innerHTML = '<div class="hit"><b>' + c.tp + '</b><span>TP: flagged, has diabetes</span></div><div class="miss"><b>' + c.fp + '</b><span>FP: flagged, does not</span></div><div class="miss"><b>' + c.fn + '</b><span>FN: missed, has diabetes</span></div><div class="hit"><b>' + c.tn + '</b><span>TN: cleared, does not</span></div>';
        Kit.$("sc-eqn").innerHTML = box("c-coral", String(c.tp), "TP: flagged women who have diabetes") + op("&divide;") + box("", String(F), "TP + FP: all flagged women (" + c.tp + " + " + c.fp + ")") + op("=") + box("c-teal", pc(ppv), "precision: the chance a flagged woman has diabetes") +
          box("c-sand", pc(r.rec), "sensitivity at this threshold") + box("c-sand", pc(r.spec), "specificity at this threshold");
        var xs = [], ys = []; for (var q = 1; q <= 50; q += 0.5) { var pp = q / 100; xs.push(q); ys.push(r.rec * pp / (r.rec * pp + (1 - r.spec) * (1 - pp))); }
        Kit.plot("sc-plot", function (P2) { return [
          { x: xs, y: ys, mode: "lines", line: { color: P2.teal, width: 3 }, hoverinfo: "skip" },
          { x: [prev], y: [ppv], mode: "markers", marker: { color: P2.coral, size: 13, line: { color: P2.paper, width: 2 } }, hovertemplate: "prevalence %{x:.1f}%, precision %{y:.2f}<extra></extra>" }
        ]; }, function (P2) { return { xaxis: { title: { text: "share of women with diabetes (%)" }, range: [0, 51] }, yaxis: { title: { text: "precision (PPV)" }, range: [0, 1.02], tickformat: ".0%" }, margin: { l: 56, r: 12, t: 12, b: 46 },
          annotations: [{ x: 34.9, y: 0.04, xref: "x", yref: "y", ax: 0, ay: 0, showarrow: false, text: "Pima sample", textangle: -90, xanchor: "right", font: { size: 12, color: P2.muted } }],
          shapes: [{ type: "line", x0: 34.9, x1: 34.9, y0: 0, y1: 1.02, line: { color: P2.muted, width: 1, dash: "dot" } }] }; });
        Kit.setText("sc-status", "Of 1,000 women, " + c.sick + " have diabetes. The model flags " + F + ", and " + c.tp + " of them have it. With the same sensitivity and specificity, " + pc(ppv, 0) + " of flagged women have diabetes. Cleared women are right " + pc(npv, 0) + " of the time.");
      }
      theme(build);
    });
  }
  REG.push(["sc", screenFig]);

  // ---------- 10. subgroup analysis ----------
  function subgroupFig() {
    return load("data/clinic.json").then(function (C) {
      var n = C.y.length, G = [{ n: "All women", lo: 21, hi: 99 }, { n: "21 to 29", lo: 21, hi: 29 }, { n: "30 to 39", lo: 30, hi: 39 }, { n: "40 to 49", lo: 40, hi: 49 }, { n: "50 and over", lo: 50, hi: 99 }];
      var tau = 0.5, view = 0, gi = 4;
      Kit.chips(Kit.$("sg-view"), ["All 768 women (cross-validated)", "The 154 held-out women"], 0, function (i) { view = i; draw(); });
      var sT = Kit.slider(Kit.$("sg-controls"), { label: "One threshold for all groups", min: 0.2, max: 0.8, step: 0.01, value: tau, show: function (v) { return Kit.fmt(v, 2); } }, function (v) { tau = v; draw(); });
      Kit.chips(Kit.$("sg-pick"), G.map(function (g) { return g.n; }), 4, function (i) { gi = i; draw(); });
      function stats(g) {
        var z = [], y = [];
        for (var i = 0; i < n; i++) { if (view === 1 && !C.test[i]) continue; if (C.age[i] < g.lo || C.age[i] > g.hi) continue; z.push(view === 1 ? C.ho[i] : C.cv[i]); y.push(C.y[i]); }
        var c = count(z, y, logit(tau)), r = rates(c);
        return { c: c, r: r, ws: wilson(c.tp, r.P), wp: wilson(c.tn, r.N), z: z, y: y };
      }
      function draw() {
        var S = G.map(stats), P = Kit.palette();
        var narrow = Kit.width(Kit.$("sg-plot")) < 520, short = ["All", "21-29", "30-39", "40-49", "50+"];
        var labels = G.map(function (g, i) { return (narrow ? short[i] : g.n) + "<br>n = " + S[i].r.T; });
        function ser(key, ws, off) { return { x: G.map(function (g, i) { return i + off; }), y: S.map(function (s) { return s.r[key]; }), err: S.map(function (s) { return s[ws]; }) }; }
        var se = ser("rec", "ws", -0.13), sp = ser("spec", "wp", 0.13);
        Kit.plot("sg-plot", function (P2) {
          function tr(s, col, sym, name) { return { x: s.x, y: s.y, mode: "markers", name: name, marker: { color: col, size: 11, symbol: sym, line: { color: P2.paper, width: 1.5 } }, error_y: { type: "data", symmetric: false, array: s.err.map(function (w, i) { return Math.max(0, w[1] - s.y[i]); }), arrayminus: s.err.map(function (w, i) { return Math.max(0, s.y[i] - w[0]); }), color: col, thickness: 2, width: 5 },
            hovertemplate: name + " %{y:.1%}<extra></extra>" }; }
          return [tr(se, P2.teal, "circle", "sensitivity"), tr(sp, P2.navy, "diamond", "specificity")];
        }, function (P2) {
          return { xaxis: { range: [-0.6, 4.6], tickvals: [0, 1, 2, 3, 4], ticktext: labels, tickangle: 0, zeroline: false }, yaxis: { title: { text: "share, with 95% interval" }, range: [0, 1.04], tickformat: ".0%", dtick: 0.25 }, margin: { l: 62, r: 12, t: 12, b: 64 },
            shapes: [{ type: "line", x0: 0.5, x1: 0.5, y0: 0, y1: 1.04, line: { color: P2.rule, width: 1.5 } }] };
        });
        var h = '<thead><tr><th>Group</th><th>Women</th><th class="hide-s">With diabetes</th><th class="hide-s">Flagged</th><th>Sensitivity</th><th>Specificity</th></tr></thead><tbody>';
        G.forEach(function (g, i) { var s = S[i], r = s.r; h += '<tr' + (i === gi ? ' class="pick"' : '') + '><td>' + g.n + '</td><td>' + r.T + '</td><td class="hide-s">' + pc(r.P / r.T, 0) + '</td><td class="hide-s">' + pc(r.F / r.T, 0) + '</td><td>' + pc(r.rec, 0) + ' (' + Kit.fmt(s.ws[0] * 100, 0) + ' to ' + Kit.fmt(s.ws[1] * 100, 0) + ')<br><span class="cell-note">' + s.c.tp + ' of ' + r.P + '</span></td><td>' + pc(r.spec, 0) + ' (' + Kit.fmt(s.wp[0] * 100, 0) + ' to ' + Kit.fmt(s.wp[1] * 100, 0) + ')<br><span class="cell-note">' + s.c.tn + ' of ' + r.N + '</span></td></tr>'; });
        Kit.$("sg-table").innerHTML = h + '</tbody>';
        var s = S[gi], k = s.c.tp, m = s.r.P, ph = m ? k / m : 0, half = 1.96 * Math.sqrt(ph * (1 - ph) / m);
        Kit.$("sg-eqn").innerHTML = box("c-teal", k + " of " + m, "sensitivity in " + G[gi].n.toLowerCase() + ": " + pc(ph, 1)) + op("&rarr;") + box("c-sand", pc(Math.max(0, ph - half), 0) + " to " + pc(Math.min(1, ph + half), 0), "plain interval, " + Kit.fmt(ph, 3) + " &plusmn; 1.96 &times; SE") + box("c-coral", pc(s.ws[0], 0) + " to " + pc(s.ws[1], 0), "Wilson interval, better for small groups");
        var a = S[1], b = S[4];
        Kit.setText("sg-status", "At threshold " + Kit.fmt(tau, 2) + " the model finds " + pc(a.r.rec, 0) + " of the women aged 21 to 29 who have diabetes and " + pc(b.r.rec, 0) + " of those aged 50 and over. It raises a false alarm for " + pc(a.r.fpr, 0) + " of the healthy women in the first group and " + pc(b.r.fpr, 0) + " in the second.");
      }
      theme(draw);
    });
  }
  REG.push(["sg", subgroupFig]);

})();
