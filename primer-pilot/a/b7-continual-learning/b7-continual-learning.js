// Chapter B7: fine-tuning, forgetting and continual learning.
// All figures read data/b7.json, which data/make_b7.py-style offline runs wrote (small PyTorch models on the made-up spadenose records).
// The pure maths (logistic model, Fisher, descent) is exported under node for tests.
(function () {
  "use strict";
  var D = null;

  // ------------------------------------------------------------------ maths
  function sig(z) { return 1 / (1 + Math.exp(-z)); }
  // theta = [w (length), b (bias), c (clarity reading)]; a row is {x, v, y}
  function feat(r) { return [r.x, 1, r.v]; }
  function logit(th, r) { return th[0] * r.x + th[1] + th[2] * r.v; }
  function rowsOf(S) { var o = []; for (var i = 0; i < S.x.length; i++) o.push({ x: S.x[i], v: S.v[i], y: S.y[i] }); return o; }
  function lossRows(th, R) {
    var s = 0;
    for (var i = 0; i < R.length; i++) { var z = logit(th, R[i]); s += Math.max(z, 0) - z * R[i].y + Math.log(1 + Math.exp(-Math.abs(z))); }
    return s / R.length;
  }
  function gradRows(th, R) {
    var g = [0, 0, 0];
    for (var i = 0; i < R.length; i++) { var e = sig(logit(th, R[i])) - R[i].y, f = feat(R[i]); g[0] += e * f[0]; g[1] += e * f[1]; g[2] += e * f[2]; }
    return g.map(function (v) { return v / R.length; });
  }
  function accRows(th, R) { var k = 0; for (var i = 0; i < R.length; i++) if ((logit(th, R[i]) > 0 ? 1 : 0) === R[i].y) k++; return k / R.length; }
  function fisherDiag(th, R) {
    var F = [0, 0, 0];
    for (var i = 0; i < R.length; i++) { var e = sig(logit(th, R[i])) - R[i].y, f = feat(R[i]); for (var j = 0; j < 3; j++) F[j] += e * e * f[j] * f[j]; }
    return F.map(function (v) { return v / R.length; });
  }
  function klBern(p, q) { var e = 1e-12; p = Math.min(Math.max(p, e), 1 - e); q = Math.min(Math.max(q, e), 1 - e); return p * Math.log(p / q) + (1 - p) * Math.log((1 - p) / (1 - q)); }
  function klRows(th, th2, R) { var s = 0; for (var i = 0; i < R.length; i++) s += klBern(sig(logit(th, R[i])), sig(logit(th2, R[i]))); return s / R.length; }
  function hess(th, R) {
    var H = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    for (var i = 0; i < R.length; i++) { var p = sig(logit(th, R[i])), w = p * (1 - p), f = feat(R[i]); for (var a = 0; a < 3; a++) for (var b = 0; b < 3; b++) H[a][b] += w * f[a] * f[b]; }
    return H.map(function (r) { return r.map(function (v) { return v / R.length; }); });
  }
  function solve3(H, g) {
    var A = H.map(function (r, i) { return r.concat([g[i]]); });
    for (var i = 0; i < 3; i++) {
      var m = i; for (var k = i + 1; k < 3; k++) if (Math.abs(A[k][i]) > Math.abs(A[m][i])) m = k;
      var t = A[i]; A[i] = A[m]; A[m] = t;
      for (var k2 = i + 1; k2 < 3; k2++) { var f = A[k2][i] / A[i][i]; for (var j = i; j < 4; j++) A[k2][j] -= f * A[i][j]; }
    }
    var x = [0, 0, 0];
    for (var i2 = 2; i2 >= 0; i2--) { var s = A[i2][3]; for (var j2 = i2 + 1; j2 < 3; j2++) s -= A[i2][j2] * x[j2]; x[i2] = s / A[i2][i2]; }
    return x;
  }
  function newtonFit(R, ridge) {
    var th = [0, 0, 0], rg = (typeof ridge === "number") ? [ridge, ridge, ridge] : ridge;
    for (var it = 0; it < 60; it++) {
      var g = gradRows(th, R), H = hess(th, R);
      for (var j = 0; j < 3; j++) { H[j][j] += rg[j]; g[j] += rg[j] * th[j]; }
      var d = solve3(H, g); th = th.map(function (v, k) { return v - d[k]; });
    }
    return th;
  }
  // Gradient descent on L_B + (lam/2) sum_j k_j (theta_j - anchor_j)^2 inside a plane of two weights (indices pl = [i, j]); other weights stay at the anchor.
  function descend(R, anchor, pl, k, lam, steps) {
    var th = anchor.slice(), path = [[th[pl[0]], th[pl[1]]]], m2 = [0, 0, 0];
    R.forEach(function (r) { var f = feat(r); for (var j = 0; j < 3; j++) m2[j] += f[j] * f[j] / R.length; });
    var h = 0.25 * (m2[pl[0]] + m2[pl[1]]) + lam * Math.max(k[pl[0]], k[pl[1]]);   // the slope of the slope cannot exceed this
    var eta = 1.0 / Math.max(h, 1e-6);
    for (var s = 1; s <= steps; s++) {
      var g = gradRows(th, R);
      for (var q = 0; q < 2; q++) { var j = pl[q]; th[j] -= eta * (g[j] + lam * k[j] * (th[j] - anchor[j])); }
      if (s % 40 === 0 || s === steps) path.push([th[pl[0]], th[pl[1]]]);
    }
    return { path: path, theta: th };
  }
  function spearman(a, b) {
    function rank(v) { var idx = v.map(function (x, i) { return [x, i]; }).sort(function (p, q) { return p[0] - q[0]; }), r = new Array(v.length), i = 0; while (i < idx.length) { var j = i; while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++; var mr = (i + j) / 2 + 1; for (var k = i; k <= j; k++) r[idx[k][1]] = mr; i = j + 1; } return r; }
    var ra = rank(a), rb = rank(b), n = a.length, ma = 0, mb = 0; ra.forEach(function (v) { ma += v / n; }); rb.forEach(function (v) { mb += v / n; });
    var sab = 0, saa = 0, sbb = 0; for (var i = 0; i < n; i++) { sab += (ra[i] - ma) * (rb[i] - mb); saa += (ra[i] - ma) * (ra[i] - ma); sbb += (rb[i] - mb) * (rb[i] - mb); }
    return sab / Math.sqrt(saa * sbb);
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { sig: sig, lossRows: lossRows, gradRows: gradRows, accRows: accRows, fisherDiag: fisherDiag, klRows: klRows, hess: hess, newtonFit: newtonFit, descend: descend, rowsOf: rowsOf, spearman: spearman };
    return;
  }

  // ------------------------------------------------------------------ helpers
  var K = Kit, $ = K.$;
  function line(name, x, y, color, o) { o = o || {}; return { type: "scatter", mode: o.mode || "lines", name: name, x: x, y: y, line: { color: color, width: o.width || 2.2, dash: o.dash || "solid" }, marker: { color: color, size: o.size || 6 }, hoverinfo: o.hover || "skip", showlegend: false, text: o.text, textposition: o.tpos, textfont: { color: color, size: 13 } }; }
  function lab(x, y, text, color, pos) { return { type: "scatter", mode: "text", x: [x], y: [y], text: [text], textposition: pos || "top right", textfont: { color: color, size: 13 }, hoverinfo: "skip", showlegend: false }; }
  function pct(v) { return (100 * v).toFixed(0) + "%"; }
  function setHTML(id, h) { var n = $(id); if (n) n.innerHTML = h; }
  function tile(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + "</b><i>" + small + "</i></span>"; }
  var WATER_COL = ["teal", "coral", "sand", "navy"];

  // ------------------------------------------------------------------ 1. the four waters
  function waterFig() {
    var A = rowsOf(D.lr2d.A.test), st = D.lr2d.A_stats, fit = D.lr2d.A_fit;
    var thr0 = st.mu + st.sd * (-fit.b / fit.w);   // length (cm) where the clear-water rule says 50%
    var cur = 0;
    var bar = $("wt-bar");
    K.chips(bar, D.tasks.map(function (t) { return t.name; }), 0, function (i) { cur = i; draw(); });
    function draw() {
      var T = D.tasks[cur], f = T.factor, xs = [], ys = [], cs = [], rnd = K.rng(5), ok = 0;
      A.forEach(function (r) {
        var L = st.mu + st.sd * r.x, app = f * L; xs.push(app); ys.push(r.y + (rnd() - 0.5) * 0.5); cs.push(r.y);
        if ((app > thr0 ? 1 : 0) === r.y) ok++;
      });
      var acc = ok / A.length, best = f * thr0;
      K.plot("wt-plot", function (P) {
        return [
          { type: "scatter", mode: "markers", x: xs.filter(function (_, i) { return cs[i] === 0; }), y: ys.filter(function (_, i) { return cs[i] === 0; }), marker: { color: P.navy, size: 8, opacity: 0.75 }, hoverinfo: "skip", showlegend: false },
          { type: "scatter", mode: "markers", x: xs.filter(function (_, i) { return cs[i] === 1; }), y: ys.filter(function (_, i) { return cs[i] === 1; }), marker: { color: P.coral, size: 8, opacity: 0.75, symbol: "diamond" }, hoverinfo: "skip", showlegend: false },
          line("clear-water rule", [thr0, thr0], [-0.55, 1.55], P.ink2, { dash: "dash", width: 2 }),
          line("right rule here", [best, best], [-0.55, 1.55], P.teal, { width: 2.5 }),
          lab(thr0, 1.5, "clear-water rule", P.ink2, "top left"),
          lab(best, -0.58, "right cut", P.teal, best < 27 ? "top right" : "top left")
        ];
      }, { xaxis: { title: { text: "length as the camera reads it, cm" }, range: [8, 66] }, yaxis: { tickvals: [0, 1], ticktext: ["immature", "mature"], range: [-0.7, 1.8], zeroline: false }, margin: { l: 70, r: 12, t: 8, b: 46 } });
      setHTML("wt-read", "<span>Water: <b>" + T.name.toLowerCase() + "</b></span><span>Camera reads <b>" + Math.round(f * 100) + "%</b> of the true length</span><span>Clarity reading <b>" + T.clarity_m + " m</b></span><span>Clear-water rule is right for <b>" + pct(acc) + "</b> of the 108 test sharks</span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 2. regularisation, early stopping, augmentation
  function regFig() {
    var wd = 0, jit = 0, pat = 3, R = D.reg;
    var bar = $("rg-bar");
    var g1 = K.html("div", { class: "label" }, bar, "weight penalty");
    K.chips(bar, R.wd.map(function (v) { return v === 0 ? "none" : String(v); }), 0, function (i) { wd = i; draw(); });
    K.html("div", { class: "label" }, bar, "noise added to inputs");
    K.chips(bar, R.jit.map(function (v) { return v === 0 ? "none" : String(v); }), 0, function (i) { jit = i; draw(); });
    var sl = K.slider($("rg-controls"), { label: "Patience: stop after this many checks without a new best", min: 1, max: 12, step: 1, value: 3, digits: 0 }, function (v) { pat = v; draw(); });
    function draw() {
      var run = R.runs[R.wd[wd] + "_" + R.jit[jit]], ep = R.epochs, val = run.val, tr = run.train;
      var best = Infinity, bi = 0, cnt = 0, stop = val.length - 1, hit = false;
      for (var i = 0; i < val.length; i++) {
        if (val[i] < best - 1e-4) { best = val[i]; bi = i; cnt = 0; } else { cnt++; if (cnt >= pat) { stop = i; hit = true; break; } }
      }
      var gb = 0; for (var j = 1; j < val.length; j++) if (val[j] < val[gb]) gb = j;
      var top = Math.max(1.0, Math.min(2.8, Math.max.apply(null, val) * 1.05)), vEnd = val[val.length - 1], close = (val[74] - tr[74]) < top * 0.18;
      K.plot("rg-plot", function (P) {
        return [
          line("training loss", ep, tr, P.navy), line("validation loss", ep, val, P.coral),
          line("", [ep[bi], ep[bi]], [0, top], P.teal, { dash: "dot", width: 2 }),
          line("", [ep[stop], ep[stop]], [0, top], P.ink2, { dash: "dash", width: 1.5 }),
          lab(300, close ? tr[74] - top * 0.03 : tr[74] + top * 0.04, "training loss", P.navy, close ? "bottom center" : "top center"), lab(300, close ? val[74] + top * 0.04 : val[74] - top * 0.04, "validation loss", P.coral, close ? "top center" : "bottom center")
        ];
      }, function (P) {
        function note(x, y, t, c) { var right = x > 250; return { x: x, y: y, text: t, showarrow: false, xanchor: right ? "right" : "left", yanchor: "top", xshift: right ? -4 : 4, font: { color: c, size: 12.5 }, bgcolor: P.paper, borderpad: 1 }; }
        return { xaxis: { title: { text: "epoch (one pass over the 30 sharks)" }, range: [0, 400] }, yaxis: { title: { text: "log loss" }, range: [0, top] },
          annotations: [note(ep[bi], top * 0.99, "best weights", P.teal), note(ep[stop], top * 0.87, hit ? "training stops" : "no stop", P.ink2)] };
      });
      setHTML("rg-read", "<span>Lowest validation loss <b>" + val[gb].toFixed(2) + "</b> at epoch <b>" + ep[gb] + "</b></span><span>Validation loss at epoch 400 <b>" + val[val.length - 1].toFixed(2) + "</b></span><span>Early stopping, patience " + pat + (hit ? ": stops at epoch <b>" + ep[stop] + "</b>" : ": never fires") + ", keeps epoch <b>" + ep[bi] + "</b>, loss <b>" + val[bi].toFixed(2) + "</b></span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 3. pretraining and fine-tuning
  function transferFig() {
    var T = D.transfer, mode = 0, cols = ["navy", "teal", "sand"];
    K.chips($("tf-bar"), ["accuracy in the new, murky water", "accuracy back in clear water"], 0, function (i) { mode = i; draw(); });
    var sl = K.slider($("tf-controls"), { label: "Labelled sharks from the murky water", min: 0, max: T.nB.length - 1, step: 1, value: 1, show: function (i) { return String(T.nB[i]); } }, function () { draw(); });
    var names = { scratch: "from scratch", finetune: "pretrained, all weights trained", "last layer only": "pretrained, last layer only" };
    function draw() {
      var narrow = K.width($("tf-plot")) < 520;
      var i = sl.get(), src = mode === 0 ? T.B : T.A, base = mode === 0 ? T.none[0] : T.none[1];
      K.plot("tf-plot", function (P) {
        var tr = T.methods.map(function (m, k) { return line(names[m], T.nB, src[m], P[cols[k]], { mode: "lines+markers" }); });
        tr.push(line("pretrained, not trained further", [T.nB[0], T.nB[T.nB.length - 1]], [base, base], P.coral, { dash: "dash" }));
        tr.push({ type: "scatter", mode: "markers", x: [T.nB[i]], y: [src.finetune[i]], marker: { color: P.ink, size: 12, symbol: "circle-open", line: { width: 2 } }, hoverinfo: "skip", showlegend: false });
        var li = T.nB.length - 1, ends = [["from scratch", src.scratch[li], P.navy], ["all weights", src.finetune[li], P.teal], ["last layer only", src["last layer only"][li], P.sand]].sort(function (a, b) { return a[1] - b[1]; });
        var ys = ends.map(function (q) { return q[1]; });
        for (var k = 1; k < ys.length; k++) if (ys[k] - ys[k - 1] < 0.05) ys[k] = ys[k - 1] + 0.05;
        if (narrow) tr.slice(0, 3).forEach(function (t) { t.showlegend = true; });
        else ends.forEach(function (q, k) { tr.push({ type: "scatter", mode: "text", x: [T.nB[li]], y: [ys[k]], text: [q[0]], textposition: "middle right", cliponaxis: false, textfont: { color: q[2], size: 13 }, hoverinfo: "skip", showlegend: false }); });
        tr.push(lab(T.nB[1], base, narrow ? "no further training" : "pretrained, no further training", P.coral, "top right"));
        return tr;
      }, { xaxis: { type: "log", title: { text: narrow ? "labelled murky-water sharks" : "labelled murky-water sharks used for training" }, tickvals: T.nB, ticktext: T.nB.map(String) }, yaxis: { title: { text: "accuracy, 108 test sharks" }, range: [0.3, 0.95] }, showlegend: narrow, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 12 } }, margin: { l: 60, r: narrow ? 10 : 100, t: narrow ? 56 : 14, b: 46 } });
      setHTML("tf-read", "<span>With <b>" + T.nB[i] + "</b> sharks: from scratch <b>" + pct(src.scratch[i]) + "</b>, all weights <b>" + pct(src.finetune[i]) + "</b>, last layer only <b>" + pct(src["last layer only"][i]) + "</b></span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ weight-space figure with springs (forgetting and EWC)
  var PLANES = {
    "yes": { name: "bias b and clarity weight c", pl: [1, 2], keep: "the weight on length stays at its clear-water value", names: ["b, the bias", "c, the weight on the clarity reading"] },
    "no": { name: "length weight w and bias b", pl: [0, 1], keep: "the clarity weight stays at 0 (no clarity reading)", names: ["w, the weight on length", "b, the bias"] }
  };
  var LAMS = [0, 1, 3, 10, 30, 100, 300];
  function springFig(pre, o) {
    o = o || {};
    var RA = rowsOf(D.lr2d.A.train), RB0 = rowsOf(D.lr2d.B.train), RAt = rowsOf(D.lr2d.A.test), RBt0 = rowsOf(D.lr2d.B.test);
    var thA = newtonFit(RA, [1e-4, 1e-4, 40]);
    var st = { reading: "yes", li: o.ewc ? 4 : 0, uniform: false, k: 0 };
    var FA = fisherDiag(thA, RA), cache = {}, playStop = null, pathIdx = 0, anim = null;
    function viewRows(R, reading) { return reading === "yes" ? R : R.map(function (r) { return { x: r.x, v: 0, y: r.y }; }); }
    function plane() {
      var key = st.reading; if (cache[key]) return cache[key];
      var PL = PLANES[key], pl = PL.pl, RB = viewRows(RB0, key), RAv = viewRows(RA, key);
      var thAv = thA.slice(); if (key === "no") { thAv = newtonFit(RAv, [1e-4, 1e-4, 40]); thAv[2] = 0; }
      var FAv = fisherDiag(thAv, RAv);
      var plainR = descend(RB, thAv, pl, [0, 0, 0], 0, 6000), pe = plainR.path[plainR.path.length - 1], e3 = descend(RB, thAv, pl, FAv, 3, 6000).path.slice(-1)[0];
      var a0 = thAv[pl[0]], a1 = thAv[pl[1]], b0 = pe[0], b1 = pe[1];
      var lo0 = Math.min(a0, b0, e3[0]), hi0 = Math.max(a0, b0, e3[0]), lo1 = Math.min(a1, b1, e3[1]), hi1 = Math.max(a1, b1, e3[1]);
      var r0 = Math.max(hi0 - lo0, 2.5), r1 = Math.max(hi1 - lo1, 2.5);
      var x0 = lo0 - 0.55 * r0, x1 = hi0 + 0.55 * r0, y0 = lo1 - 0.55 * r1, y1 = hi1 + 0.55 * r1, n = 60, xs = [], ys = [], ZA = [], ZB = [];
      for (var i = 0; i < n; i++) { xs.push(x0 + (x1 - x0) * i / (n - 1)); ys.push(y0 + (y1 - y0) * i / (n - 1)); }
      var minA = Infinity, minB = Infinity;
      for (var j = 0; j < n; j++) { var ra = [], rb = []; for (var i2 = 0; i2 < n; i2++) { var th = thAv.slice(); th[pl[0]] = xs[i2]; th[pl[1]] = ys[j]; var la = lossRows(th, RAv), lb = lossRows(th, RB); ra.push(la); rb.push(lb); minA = Math.min(minA, la); minB = Math.min(minB, lb); } ZA.push(ra); ZB.push(rb); }
      return (cache[key] = { PL: PL, pl: pl, RB: RB, RAv: RAv, thA: thAv, plainR: plainR, F: FAv, x: xs, y: ys, ZA: ZA, ZB: ZB, minA: minA, minB: minB, RBt: viewRows(RBt0, key), RAt: viewRows(RAt, key) });
    }
    function stiff(S) {
      var lam = LAMS[st.li], k;
      if (st.uniform) { var m = (S.F[0] + S.F[1] + S.F[2]) / 3; k = [m, m, m]; } else k = S.F.slice();
      return { lam: lam, k: k };
    }
    function results(S) {
      var s = stiff(S), plain = S.plainR, ewc = descend(S.RB, S.thA, S.pl, s.k, s.lam, 6000);
      return { plain: plain, ewc: ewc, s: s };
    }
    function zig(p0, p1, along, amp, turns) {
      var xs = [], ys = [], n = turns * 2 + 2;
      for (var i = 0; i <= n; i++) {
        var t = i / n, off = (i === 0 || i === n) ? 0 : (i % 2 ? amp : -amp);
        xs.push(p0[0] + (p1[0] - p0[0]) * t + (along === "x" ? 0 : off)); ys.push(p0[1] + (p1[1] - p0[1]) * t + (along === "x" ? off : 0));
      }
      return { x: xs, y: ys };
    }
    function draw(pos) {
      var S = plane(), R = results(S), sl = R.s, pl = S.pl, path = R.ewc.path, plain = R.plain.path;
      var idx = pos == null ? path.length - 1 : pos, cur = path[idx];
      var span = [S.x[S.x.length - 1] - S.x[0], S.y[S.y.length - 1] - S.y[0]];
      var thc = S.thA.slice(); thc[pl[0]] = cur[0]; thc[pl[1]] = cur[1];
      var kmax = Math.max(sl.lam * Math.max(sl.k[pl[0]], sl.k[pl[1]]), 1e-9);
      function levels(minv, sc) { return { start: Math.log(0.015), end: Math.log(sc), size: (Math.log(sc) - Math.log(0.015)) / 9 }; }
      var zA = S.ZA.map(function (r) { return r.map(function (v) { return Math.log(Math.max(v - S.minA, 0) + 0.015); }); });
      var zB = S.ZB.map(function (r) { return r.map(function (v) { return Math.log(Math.max(v - S.minB, 0) + 0.015); }); });
      K.plot(pre + "-plot", function (P) {
        var tr = [];
        function cont(z, col) { return { type: "contour", x: S.x, y: S.y, z: z, showscale: false, hoverinfo: "skip", contours: { coloring: "lines", start: Math.log(0.03), end: Math.log(1.6), size: (Math.log(1.6) - Math.log(0.03)) / 5 }, line: { width: 1.8 }, colorscale: [[0, col], [1, col]], ncontours: 6 }; }
        tr.push(cont(zA, P.navy)); tr.push(cont(zB, P.coral));
        tr.push(line("plain", plain.map(function (p) { return p[0]; }), plain.map(function (p) { return p[1]; }), P.coral, { dash: "dot", width: 2.6, mode: "lines" }));
        if (sl.lam > 0 && o.ewc) tr.push(line("ewc", path.slice(0, idx + 1).map(function (p) { return p[0]; }), path.slice(0, idx + 1).map(function (p) { return p[1]; }), P.teal, { width: 3 }));
        // springs from the anchor lines to the current point
        if (o.ewc && sl.lam > 0) {
          [0, 1].forEach(function (q) {
            var kq = sl.lam * sl.k[pl[q]], w = 1 + 5 * Math.sqrt(kq / kmax);
            var a = S.thA[pl[q]];
            var p0 = q === 0 ? [a, cur[1]] : [cur[0], a], p1 = [cur[0], cur[1]];
            var z = zig(p0, p1, q === 0 ? "x" : "y", (q === 0 ? span[1] : span[0]) * 0.022, 6);
            tr.push({ type: "scatter", mode: "lines", x: z.x, y: z.y, line: { color: P.sand, width: w }, hoverinfo: "skip", showlegend: false });
          });
        }
        tr.push({ type: "scatter", mode: "markers", x: [S.thA[pl[0]]], y: [S.thA[pl[1]]], marker: { color: P.navy, size: 13 }, hoverinfo: "skip", showlegend: false });
        tr.push({ type: "scatter", mode: "markers", x: [cur[0]], y: [cur[1]], marker: { color: o.ewc && sl.lam > 0 ? P.teal : P.coral, size: 12, line: { color: P.ink, width: 2 } }, hoverinfo: "skip", showlegend: false });
        tr.push({ type: "scatter", mode: "markers+text", x: [plain[plain.length - 1][0]], y: [plain[plain.length - 1][1]], marker: { color: P.coral, size: 15, symbol: "star" }, hoverinfo: "skip", showlegend: false });
        return tr;
      }, function (P) {
        var pe = plain[plain.length - 1];
        function ann(x, y, t, c, ax, ay) { return { x: x, y: y, text: t, showarrow: true, arrowhead: 0, arrowcolor: c, ax: ax, ay: ay, font: { color: c, size: 12.5 }, bgcolor: P.paper, bordercolor: c, borderwidth: 1, borderpad: 2 }; }
        return { xaxis: { title: { text: S.PL.names[0] }, range: [S.x[0], S.x[S.x.length - 1]] }, yaxis: { title: { text: S.PL.names[1] }, range: [S.y[0], S.y[S.y.length - 1]] }, margin: { l: 56, r: 10, t: 8, b: 46 },
          annotations: [ann(S.thA[pl[0]], S.thA[pl[1]], "old optimum", P.navy, -46, -34), ann(pe[0], pe[1], "plain fine-tuning ends here", P.coral, -40, 38)].concat(K.width($(pre + "-plot")) > 520 ? [ann(S.x[2], S.y[S.y.length - 3], "blue: old water's loss", P.navy, 0, 0), ann(S.x[2], S.y[S.y.length - 8], "coral: new water's loss", P.coral, 0, 0)] : []) };
      });
      var aE = { A: accRows(thc, S.RAt), B: accRows(thc, S.RBt) }, thP = S.thA.slice(); thP[pl[0]] = plain[plain.length - 1][0]; thP[pl[1]] = plain[plain.length - 1][1];
      var aP = { A: accRows(thP, S.RAt), B: accRows(thP, S.RBt) };
      var html = "<span>Plain fine-tuning: clear water <b>" + pct(aP.A) + "</b>, murky water <b>" + pct(aP.B) + "</b>, loss on the old water <b>" + lossRows(thP, S.RAv).toFixed(2) + "</b></span>";
      if (o.ewc) html += "<span>With springs: clear water <b>" + pct(aE.A) + "</b>, murky water <b>" + pct(aE.B) + "</b>, loss on the old water <b>" + lossRows(thc, S.RAv).toFixed(2) + "</b></span>";
      setHTML(pre + "-read", html);
      if (o.ewc) {
        var kb = sl.lam * sl.k[pl[0]], kc = sl.lam * sl.k[pl[1]];
        setHTML(pre + "-eqn", tile("c-navy", "&lambda; = " + sl.lam, "strength you set") + '<span class="op">&times;</span>' + tile("c-sand", sl.k[pl[0]].toFixed(3), "F for " + S.PL.names[0].split(",")[0]) + '<span class="op">=</span>' + tile("c-teal", kb.toFixed(2), "spring 1 stiffness") + '<span class="op">|</span>' + tile("c-navy", "&lambda; = " + sl.lam, "strength") + '<span class="op">&times;</span>' + tile("c-sand", sl.k[pl[1]].toFixed(3), "F for " + S.PL.names[1].split(",")[0]) + '<span class="op">=</span>' + tile("c-teal", kc.toFixed(2), "spring 2 stiffness"));
        var penalty = 0.5 * sl.lam * (sl.k[pl[0]] * Math.pow(cur[0] - S.thA[pl[0]], 2) + sl.k[pl[1]] * Math.pow(cur[1] - S.thA[pl[1]], 2));
        setHTML(pre + "-status", "Loss on the new water <b>" + lossRows(thc, S.RB).toFixed(2) + "</b> plus spring penalty <b>" + penalty.toFixed(2) + "</b>.");
      }
    }
    function redraw(pos) { draw(pos); }
    var bar = $(pre + "-bar");
    K.chips(bar, ["SharkCam reads water clarity", "no clarity reading"], 0, function (i) { st.reading = i === 0 ? "yes" : "no"; cache[st.reading] = cache[st.reading]; if (anim) anim(); redraw(); $(pre + "-sub").textContent = "Plane shown: " + plane().PL.name + "; " + plane().PL.keep + "."; });
    if (o.ewc) {
      K.chips(bar, ["springs set by the Fisher diagonal", "same springs on all weights"], 0, function (i) { st.uniform = i === 1; if (anim) anim(); redraw(); });
      K.button(bar, "Play", function () {
        if (anim) anim(); var S = plane(), R = results(S), n = R.ewc.path.length;
        anim = K.animate(2600, function (p) { redraw(Math.min(n - 1, Math.round(K.ease(p) * (n - 1)))); });
      });
      K.slider($(pre + "-controls"), { label: "Spring strength λ", min: 0, max: LAMS.length - 1, step: 1, value: st.li, show: function (i) { return LAMS[i] === 0 ? "0 (no springs)" : String(LAMS[i]); } }, function (v) { st.li = v; if (anim) anim(); redraw(); });
    }
    $(pre + "-sub").textContent = "Plane shown: " + plane().PL.name + "; " + plane().PL.keep + ".";
    redraw();
    return { redraw: redraw };
  }

  // ------------------------------------------------------------------ 4/6. forgetting curve and EWC curves for the small network
  function forgetFig(pre, o) {
    var F = D.forget, st = { clar: "clarity", meth: o.ewc ? 0 : -1, li: 2 }, scrub;
    var tag = D.tracked;
    var avail = { clarity: ["ewc3", "ewc10", "ewc30", "ewc100", "ewc300"], noclarity: ["ewc10", "ewc30", "ewc100"] };
    function key() {
      if (!o.ewc || st.meth === -1) return "plain";
      if (st.meth === 1) return "uniform30"; if (st.meth === 2) return "replay";
      var a = avail[st.clar]; return a[Math.min(st.li, a.length - 1)];
    }
    var bar = $(pre + "-bar");
    if (o.ewc) {
      K.chips(bar, ["with a clarity reading", "no clarity reading"], 0, function (i) {
        var a0 = avail[st.clar], lam = parseInt(a0[Math.min(st.li, a0.length - 1)].replace("ewc", ""), 10);
        st.clar = i === 0 ? "clarity" : "noclarity"; if (st.clar === "noclarity" && st.meth === 1) st.meth = 0;
        var a1 = avail[st.clar], j = 0; a1.forEach(function (k, q) { if (Math.abs(parseInt(k.replace("ewc", ""), 10) - lam) < Math.abs(parseInt(a1[j].replace("ewc", ""), 10) - lam)) j = q; });
        setLam(j); draw();
      });
      K.chips(bar, ["Fisher springs", "same springs on all weights", "replay of 20 sharks"], 0, function (i) { st.meth = i; draw(); });
    }
    var lamSl = o.ewc ? K.slider($(pre + "-controls"), { label: "Spring strength λ", min: 0, max: 4, step: 1, value: 2, show: function (i) { var a = avail[st.clar]; return String(parseInt(a[Math.min(i, a.length - 1)].replace("ewc", ""), 10)); } }, function (v) { st.li = v; draw(); }) : null;
    function setLam(j) { if (lamSl) { lamSl.input.max = String(avail[st.clar].length - 1); lamSl.set(j); st.li = lamSl.get(); } }
    scrub = K.slider($(pre + "-scrub"), { label: "Steps of training on the murky water", min: 0, max: 200, step: 1, value: 200, digits: 0 }, function () { draw(); });
    K.button(bar, "Play", function () { var stop = K.animate(3200, function (p) { scrub.set(Math.round(K.ease(p) * 200)); draw(); }); });
    function draw() {
      var missing = !F[st.clar][key()], e = F[st.clar][key()] || F[st.clar].plain, base = F[st.clar].plain, s = scrub.get(), xs = e.accA.map(function (_, i) { return i; });
      var showBase = o.ewc && key() !== "plain";
      K.plot(pre + "-plot", function (P) {
        var tr = [];
        if (showBase) { tr.push(line("", xs, base.accA, P.coral, { dash: "dot", width: 1.6 })); tr.push(line("", xs, base.accB, P.teal, { dash: "dot", width: 1.6 })); }
        tr.push(line("clear water", xs, e.accA, P.coral, { width: 3 })); tr.push(line("murky water", xs, e.accB, P.teal, { width: 3 }));
        tr.push(line("", [s, s], [0.3, 1], P.ink2, { dash: "dash", width: 1.4 }));
        tr.push({ type: "scatter", mode: "markers", x: [s, s], y: [e.accA[s], e.accB[s]], marker: { color: [P.coral, P.teal], size: 11, line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip", showlegend: false });
        var up = e.accA[120] >= e.accB[120];
        tr.push(lab(125, e.accA[120] + (up ? 0.045 : -0.045), "old, clear water", P.coral, "middle center"));
        tr.push(lab(125, e.accB[120] + (up ? -0.045 : 0.045), "new, murky water", P.teal, "middle center"));
        return tr;
      }, { xaxis: { title: { text: "steps of training on the murky water" }, range: [0, 200] }, yaxis: { title: { text: "accuracy on 108 test sharks" }, range: [0.3, 1], tickformat: ".0%" } });
      if (missing) { setHTML(pre + "-read", "<span>This variant was not run for the case without a clarity reading. The plain fine-tuning curves are shown.</span>"); return; }
      setHTML(pre + "-read", "<span>Step <b>" + s + "</b>: clear water <b>" + pct(e.accA[s]) + "</b> (started at " + pct(e.accA[0]) + "), murky water <b>" + pct(e.accB[s]) + "</b></span><span>Tag " + tag.tag + " (" + tag.length_cm + " cm, immature): chance of 'mature' in clear water <b>" + e.tag_p[s].toFixed(2) + "</b> (was " + e.tag_p[0].toFixed(2) + ")</span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 5. what the Fisher information measures
  function fisherFig() {
    var RA = rowsOf(D.lr2d.A.train), th = newtonFit(RA, [1e-4, 1e-4, 40]), F = fisherDiag(th, RA), which = 0, names = ["w, the weight on length", "b, the bias", "c, the weight on the clarity reading"], short = ["w", "b", "c"], cols = ["navy", "coral", "teal"];
    var st = D.lr2d.A_stats;
    K.chips($("fi-bar"), names, 0, function (i) { which = i; draw(); });
    var sl = K.slider($("fi-controls"), { label: "Nudge the weight by δ", min: -2, max: 2, step: 0.05, value: 1, digits: 2 }, function () { draw(); });
    function draw() {
      var d = sl.get(), th2 = th.slice(); th2[which] += d;
      var xs = [], p0 = [], p1 = [];
      for (var i = 0; i <= 80; i++) { var L = 15 + i * 0.75, x = (L - st.mu) / st.sd; xs.push(L); p0.push(sig(th[0] * x + th[1])); p1.push(sig(th2[0] * x + th2[1] + th2[2] * (-0.0))); }
      // the clarity reading in clear water is about 0, so c has almost no effect on this curve
      var ds = [], kls = [[], [], []];
      for (var q = -2; q <= 2.0001; q += 0.1) { ds.push(q); for (var j = 0; j < 3; j++) { var t = th.slice(); t[j] += q; kls[j].push(klRows(th, t, RA)); } }
      var kl = klRows(th, th2, RA), pred = 0.5 * F[which] * d * d;
      K.plot("fi-curve", function (P) {
        return [line("before", xs, p0, P.ink2, { dash: "dash" }), line("after", xs, p1, P[cols[which]], { width: 3 }), lab(44, 0.5, "dashed: before", P.ink2, "middle right"), lab(44, 0.38, "solid: after", P[cols[which]], "middle right")];
      }, { xaxis: { title: { text: "length in clear water, cm" } }, yaxis: { title: { text: "chance of 'mature'" }, range: [-0.02, 1.02] }, margin: { l: 56, r: 10, t: 8, b: 46 } });
      K.plot("fi-kl", function (P) {
        var tr = [];
        for (var j = 0; j < 3; j++) tr.push(line(short[j], ds, kls[j], P[cols[j]], { width: j === which ? 3.4 : 1.6 }));
        tr.push({ type: "scatter", mode: "markers", x: [d], y: [kl], marker: { color: P[cols[which]], size: 12, line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip", showlegend: false });
        tr.push(lab(1.9, kls[0][ds.length - 1], "w", P.navy, "top left")); tr.push(lab(1.9, kls[1][ds.length - 1], "b", P.coral, "top left")); tr.push(lab(1.9, Math.max(kls[2][ds.length - 1], 0.0008), "c", P.teal, "top left"));
        return tr;
      }, { xaxis: { title: { text: "nudge δ" } }, yaxis: { title: { text: "average KL, nats" }, rangemode: "tozero" }, margin: { l: 66, r: 24, t: 8, b: 46 } });
      setHTML("fi-eqn", tile("c-sand", F[which].toFixed(3), "F for " + short[which]) + '<span class="op">&times; &frac12; &delta;&sup2; =</span>' + tile("c-navy", pred.toFixed(4), "&frac12; F δ², the prediction") + '<span class="op">vs</span>' + tile("c-teal", kl.toFixed(4), "measured change"));
      setHTML("fi-read", "<span>Fisher diagonal: F<sub>w</sub> <b>" + F[0].toFixed(3) + "</b>, F<sub>b</sub> <b>" + F[1].toFixed(3) + "</b>, F<sub>c</sub> <b>" + F[2].toFixed(4) + "</b></span>");
    }
    draw();
    return { th: th, F: F };
  }

  // ------------------------------------------------------------------ 7. Adam's v against the Fisher diagonal
  function barsFig() {
    var F = D.fisher, mode = 0, idx = [];
    F.live.forEach(function (l, i) { if (l) idx.push(i); });
    K.chips($("fb-bar"), ["raw values", "series scaled to sum to 1"], 0, function (i) { mode = i; draw(); });
    function draw() {
      var series = [["Fisher diagonal (average squared slope)", F.emp, "coral"], ["Adam's v after training", F.v_end, "navy"], ["v rebuilt in 200 tiny steps", F.v_rec, "teal"]];
      var labels = idx.map(function (i) { return F.names[i]; }), narrow = K.width($("fb-plot")) < 560;
      K.plot("fb-plot", function (P) {
        return series.map(function (s) {
          var vals = idx.map(function (i) { return s[1][i]; });
          if (mode === 1) { var tot = 0; idx.forEach(function (i) { tot += s[1][i]; }); vals = vals.map(function (v) { return v / tot; }); }
          return { type: "bar", name: s[0], x: labels, y: vals, marker: { color: P[s[2]] }, hovertemplate: "%{x}<br>" + s[0] + ": %{y:.4f}<extra></extra>" };
        });
      }, { barmode: "group", showlegend: true, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: narrow ? 11 : 12.5 } }, xaxis: { tickangle: -40, tickfont: { size: 11 } }, yaxis: { type: "log", title: { text: mode ? "share of total, log" : "value, log scale" }, tickvals: [1e-4, 1e-3, 1e-2, 1e-1, 1], ticktext: ["0.0001", "0.001", "0.01", "0.1", "1"], range: [-4.3, 0] }, margin: { l: 66, r: 10, t: narrow ? 96 : 40, b: 96 } });
      var ratio = 0, a = 0, b = 0; idx.forEach(function (i) { a += F.v_rec[i]; b += F.emp[i]; });
      setHTML("fb-read", "<span>Live weights shown: <b>" + idx.length + "</b> of " + F.n_weights + " (the other " + (F.n_weights - idx.length) + " belong to two hidden units that are off for all sharks, so all three series are 0)</span><span>Rank agreement (Spearman) with the Fisher diagonal, mean of 20 networks: Adam's v <b>" + F.rho_mean.emp_vend.toFixed(2) + "</b>, rebuilt v <b>" + F.rho_mean.emp_vrec.toFixed(2) + "</b></span><span>Sum of rebuilt v over sum of Fisher: <b>" + (a / b).toFixed(3) + "</b> (batches of 16)</span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 8. a timeline of waters
  function timelineFig() {
    var T = D.timeline, tasks = D.tasks, strategies = [["fine-tune only", "plain"], ["replay of 10 sharks per water", "replay"], ["EWC springs", "ewc"], ["separate weights per water", "separate"]];
    var st = { stage: 0, s: 0 };
    var box = $("tl-grid");
    K.chips($("tl-bar"), strategies.map(function (s) { return s[0]; }), 0, function (i) { st.s = i; draw(); });
    var sl = K.slider($("tl-controls"), { label: "Dive number (which water has just arrived)", min: 0, max: 3, step: 1, value: 0, show: function (i) { return String(i + 1) + " of 4"; } }, function (v) { st.stage = v; draw(); });
    K.button($("tl-bar"), "Next water", function () { sl.set((sl.get() + 1) % 4); st.stage = sl.get(); draw(); });
    var costs = { plain: "stores no extra data", replay: "stores 10 sharks per old water", ewc: "stores one copy of the weights and of F per old water", separate: "stores a full copy of the network per water, and needs to know which water it is in" };
    function mixc(P, v) { return "color-mix(in srgb, " + P.teal + " " + Math.round(Math.max(0, (v - 0.3) / 0.65) * 100) + "%, " + P.bg + ")"; }
    function draw() {
      var P = K.palette(), M = T[strategies[st.s][1]], stage = st.stage, h = '<div class="tl-row tl-head"><span></span>';
      tasks.forEach(function (t, j) { h += '<span class="tl-wh' + (j === stage ? " now" : "") + '"><b>' + t.name + "</b><i>clarity " + t.clarity_m + " m</i></span>"; });
      h += "</div>";
      for (var r = 0; r < 4; r++) {
        h += '<div class="tl-row' + (r === stage ? " on" : "") + '"><span class="tl-lab">after dive ' + (r + 1) + "</span>";
        for (var c = 0; c < 4; c++) {
          var v = M[r][c], seen = c <= r;
          h += '<span class="tl-c' + (seen ? "" : " unseen") + '" style="background:' + (seen ? mixc(P, v) : "transparent") + '">' + (seen ? "<b>" : "") + pct(v) + (seen ? "</b>" : "") + "</span>";
        }
        h += "</div>";
      }
      box.innerHTML = h;
      var seen = M[stage].slice(0, stage + 1), avg = seen.reduce(function (a, b) { return a + b; }, 0) / seen.length;
      var bars = strategies.map(function (s) { var m = T[s[1]][stage].slice(0, stage + 1), a = m.reduce(function (x, y) { return x + y; }, 0) / m.length; return [s[0], a]; });
      var bh = "";
      bars.forEach(function (b, i) { bh += '<div class="tl-bar' + (i === st.s ? " me" : "") + '"><span>' + b[0] + '</span><span class="tl-track"><span style="width:' + Math.round(((b[1] - 0.3) / 0.65) * 100) + '%"></span></span><b>' + pct(b[1]) + "</b></div>"; });
      $("tl-bars").innerHTML = bh;
      var joint = T.joint[0].reduce(function (a, b) { return a + b; }, 0) / 4;
      setHTML("tl-read", "<span>After dive " + (stage + 1) + ", average over the " + (stage + 1) + " water" + (stage ? "s" : "") + " seen: <b>" + pct(avg) + "</b></span><span>Cost: " + costs[strategies[st.s][1]] + "</span>" + (stage === 3 ? "<span>For reference, one network trained on all four waters at once: <b>" + pct(joint) + "</b></span>" : ""));
    }
    K.onTheme(draw); draw();
  }

  // ------------------------------------------------------------------ 9. adapting while in use
  function streamFig() {
    var S = D.lr2d.stream, st = D.lr2d.A_stats, fit = D.lr2d.A_fit, n = S.len_cm.length, win = 20;
    function predict(L, mu, sd) { var x = (L - mu) / sd; return sig(fit.w * x + fit.b) > 0.5 ? 1 : 0; }
    var frozen = [], adapt = [], mus = [], sds = [], m = 0, M2 = 0, cnt = 0;
    for (var i = 0; i < n; i++) {
      var L = S.len_cm[i];
      frozen.push(predict(L, st.mu, st.sd) === S.y[i] ? 1 : 0);
      cnt++; var d = L - m; m += d / cnt; M2 += d * (L - m);
      var mu = cnt >= 5 ? m : st.mu * S.factor * 0 + (cnt >= 1 ? (m * cnt + st.mu * (5 - cnt)) / 5 : st.mu), sd = cnt >= 5 ? Math.sqrt(M2 / cnt) : st.sd;
      mus.push(m);
      adapt.push(predict(L, mu, sd) === S.y[i] ? 1 : 0); sds.push(sd);
    }
    function roll(a, t) { var s = 0, c = 0; for (var i = Math.max(0, t - win + 1); i <= t; i++) { s += a[i]; c++; } return s / c; }
    var sl = K.slider($("st-controls"), { label: "Frames seen so far in the murky water", min: 5, max: n, step: 1, value: n, digits: 0 }, function () { draw(); });
    K.button($("st-bar"), "Play", function () { K.animate(4200, function (p) { sl.set(Math.max(5, Math.round(K.ease(p) * n))); draw(); }); });
    function draw() {
      var t = sl.get() - 1, xs = [], ya = [], yf = [];
      for (var i = 4; i <= t; i++) { xs.push(i + 1); ya.push(roll(adapt, i)); yf.push(roll(frozen, i)); }
      var trueMu = S.factor * st.mu;
      K.plot("st-acc", function (P) {
        return [line("", xs, yf, P.coral, { width: 3 }), line("", xs, ya, P.teal, { width: 3 }), lab(4, 0.06, "frozen clear-water rule", P.coral, "middle right"), lab(4, 1.09, "input scale re-estimated", P.teal, "middle right")];
      }, { xaxis: { title: { text: "frames seen" }, range: [0, n + 2] }, yaxis: { title: { text: "right in the last 20 frames" }, range: [0, 1.15], tickformat: ".0%", tickvals: [0, 0.2, 0.4, 0.6, 0.8, 1] }, margin: { l: 60, r: 10, t: 8, b: 46 } });
      var xm = [], ym = [];
      for (var j = 0; j <= t; j++) { xm.push(j + 1); ym.push(mus[j]); }
      K.plot("st-mu", function (P) {
        return [line("", xm, ym, P.teal, { width: 3 }), line("", [0, n + 2], [trueMu, trueMu], P.ink2, { dash: "dash", width: 1.5 }), line("", [0, n + 2], [st.mu, st.mu], P.coral, { dash: "dot", width: 1.5 }), lab(n - 4, trueMu, "water's true mean " + trueMu.toFixed(1) + " cm", P.ink2, "bottom left"), lab(n - 4, st.mu, "clear-water mean " + st.mu.toFixed(1) + " cm", P.coral, "top left")];
      }, { xaxis: { title: { text: "frames seen" }, range: [0, n + 2] }, yaxis: { title: { text: "running mean of the readings, cm" }, range: [15, 45] }, margin: { l: 60, r: 10, t: 8, b: 46 } });
      var a1 = 0, f1 = 0; for (var q = 0; q <= t; q++) { a1 += adapt[q]; f1 += frozen[q]; }
      setHTML("st-read", "<span>After " + (t + 1) + " frames: frozen model right in <b>" + pct(f1 / (t + 1)) + "</b>, adapting <b>" + pct(a1 / (t + 1)) + "</b></span><span>Running mean now <b>" + mus[t].toFixed(1) + " cm</b></span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    K.data("data/b7.json").then(function (d) {
      D = d;
      var jobs = [["wt-plot", waterFig], ["rg-plot", regFig], ["tf-plot", transferFig], ["fg-plot", function () { forgetFig("fg", { ewc: false }); }], ["sp0-plot", function () { springFig("sp0", { ewc: false }); }],
        ["fi-curve", fisherFig], ["sp-plot", function () { springFig("sp", { ewc: true }); }], ["ec-plot", function () { forgetFig("ec", { ewc: true }); }], ["fb-plot", barsFig], ["tl-grid", timelineFig], ["st-acc", streamFig]];
      jobs.forEach(function (j) {
        var node = $(j[0]); if (!node) return;
        try { j[1](); } catch (e) { console.error(j[0], e); }
      });
    }).catch(function (e) { console.error(e); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
