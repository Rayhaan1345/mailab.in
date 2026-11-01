// A3 Logistic regression and softmax: five figures.
//   1 sigmoidFig  a straight line or a sigmoid through juvenile and adult spadenose sharks, sliders for steepness and midpoint (Plotly)
//   2 boundaryFig a probability map on length and girth, a movable decision boundary, click to add sharks (Plotly)
//   3 lossFig     log loss against squared loss for one shark as the score slides from wrong to right (Plotly)
//   4 trainFig    gradient descent for logistic regression, the boundary moving over the map (Plotly)
//   5 softmaxFig  five species: a table of scores, a species map and softmax probabilities (Plotly and HTML)
(function () {
  "use strict";
  var REC = null, SP = [], L = [], G = [], Y = [], N = 0, ML = 0, SL = 1, MG = 0, SG = 1, SOFT = null;
  var SPECIES = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"];
  var NAME = { spadenose: "spadenose", milk: "milk shark", bamboo: "bamboo shark", hammerhead: "hammerhead", ganges: "Ganges shark" };
  var TAGS = { 3305: { L: 41.0, G: 11.5, sp: "spadenose", mature: 1 }, 4127: { L: 44.0, G: 12.4, sp: "bamboo", mature: 0 }, 5519: { L: 82.0, G: 27.6, sp: "hammerhead", mature: 0 } };

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("../data/sharks/catch.json"), Kit.data("data/softmax.json")]).then(function (r) {
      REC = r[0].records; SOFT = r[1];
      SP = REC.filter(function (q) { return q.species === "spadenose"; });
      L = SP.map(function (q) { return q.length_cm; }); G = SP.map(function (q) { return q.girth_cm; }); Y = SP.map(function (q) { return q.mature; });
      N = L.length; ML = mean(L); SL = sd(L); MG = mean(G); SG = sd(G);
      sigmoidFig(); boundaryFig(); lossFig(); trainFig(); softmaxFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- helpers ----------
  function mean(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function sd(a) { var m = mean(a); return Math.sqrt(mean(a.map(function (v) { return (v - m) * (v - m); }))); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function sig(z) { return 1 / (1 + Math.exp(-z)); }
  function box(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function op(s) { return '<span class="op">' + s + '</span>'; }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { var C = hexRGB(c); return "rgba(" + C[0] + "," + C[1] + "," + C[2] + "," + a + ")"; }
  function spCol(P) { return { spadenose: P.navy, milk: P.coral, bamboo: P.teal, hammerhead: P.sand, ganges: P.ink2 }; }
  function lossTerm(p, y) { p = clamp(p, 1e-12, 1 - 1e-12); return y ? -Math.log(p) : -Math.log(1 - p); }
  // solve A x = b by Gaussian elimination with partial pivoting
  function solve(A, b) {
    var n = b.length, i, j, k; A = A.map(function (r) { return r.slice(); }); b = b.slice();
    for (i = 0; i < n; i++) {
      var p = i; for (k = i + 1; k < n; k++) if (Math.abs(A[k][i]) > Math.abs(A[p][i])) p = k;
      var t = A[i]; A[i] = A[p]; A[p] = t; var tb = b[i]; b[i] = b[p]; b[p] = tb;
      for (k = i + 1; k < n; k++) { var f = A[k][i] / A[i][i]; for (j = i; j < n; j++) A[k][j] -= f * A[i][j]; b[k] -= f * b[i]; }
    }
    var x = new Array(n); for (i = n - 1; i >= 0; i--) { var s = b[i]; for (j = i + 1; j < n; j++) s -= A[i][j] * x[j]; x[i] = s / A[i][i]; } return x;
  }
  // Newton's method for logistic regression on rows Z (already standardised), labels y; the last entry of w is the bias
  function fitLogistic(Z, y, ridge) {
    var d = Z[0].length, w = new Array(d + 1).fill(0), it, i, a, c, n = Z.length;
    for (it = 0; it < 40; it++) {
      var g = new Array(d + 1).fill(0), H = []; for (a = 0; a <= d; a++) H.push(new Array(d + 1).fill(0));
      for (i = 0; i < n; i++) {
        var z = w[d]; for (a = 0; a < d; a++) z += w[a] * Z[i][a];
        var p = sig(z), r = p - y[i], q = p * (1 - p) + 1e-9;
        for (a = 0; a <= d; a++) { var xa = a < d ? Z[i][a] : 1; g[a] += r * xa / n; for (c = 0; c <= d; c++) H[a][c] += q * xa * (c < d ? Z[i][c] : 1) / n; }
      }
      for (a = 0; a < d; a++) { g[a] += ridge * w[a]; H[a][a] += ridge; }
      var st = solve(H, g), mx = 0; for (a = 0; a <= d; a++) { w[a] -= st[a]; mx = Math.max(mx, Math.abs(st[a])); }
      for (a = 0; a <= d; a++) w[a] = clamp(w[a], -25, 25);
      if (mx < 1e-9) break;
    }
    return w;
  }
  function typeset(node) { if (typeof renderMathInElement !== "undefined") renderMathInElement(node, { delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false }); }
  function hasWebGL() { try { var c = document.createElement("canvas"); return !!(c.getContext("webgl") || c.getContext("experimental-webgl")); } catch (e) { return false; } }

  // ---------- 1. from a score to a probability ----------
  function sigmoidFig() {
    var jit = Kit.rng(11), jy = L.map(function () { return (jit() - 0.5) * 0.07; });
    var extra = false, mode = 1, w = 0.15, L50 = 45;
    var X1 = function () { return extra ? L.concat(new Array(100).fill(80)) : L; };
    var Y1 = function () { return extra ? Y.concat(new Array(100).fill(1)) : Y; };
    var sw = Kit.slider(Kit.$("sg-controls"), { label: "Steepness w, per cm", min: 0.01, max: 1.5, step: 0.01, value: w, digits: 2 }, function (v) { w = v; draw(); });
    var sm = Kit.slider(Kit.$("sg-controls"), { label: "Midpoint L50: where the score is 0, cm", min: 20, max: 60, step: 0.1, value: L50, digits: 1 }, function (v) { L50 = v; draw(); });
    var bar = Kit.$("sg-bar");
    Kit.chips(bar, ["Straight line", "Sigmoid"], 1, function (i) { mode = i; draw(); });
    Kit.chips(bar, ["The 268 records", "Add 100 adults of 80 cm"], 0, function (i) { extra = i === 1; draw(); });
    Kit.button(bar, "Fit the best curve", function () { fitBest(); draw(); });
    function fitBest() {
      var xs = X1(), ys = Y1(), n = xs.length;
      if (mode === 1) {
        var m = mean(xs), s = sd(xs), Z = xs.map(function (v) { return [(v - m) / s]; }), f = fitLogistic(Z, ys, 1e-6);
        w = f[0] / s; L50 = m - f[1] / w; // z = w (L - L50)
      } else {
        var mx = mean(xs), my = mean(ys), sxy = 0, sxx = 0, i; for (i = 0; i < n; i++) { sxy += (xs[i] - mx) * (ys[i] - my); sxx += (xs[i] - mx) * (xs[i] - mx); }
        w = sxy / sxx; L50 = mx + (0.5 - my) / w;
      }
      w = clamp(w, 0.01, 1.5); L50 = clamp(L50, 20, 60); sw.set(w); sm.set(L50);
    }
    function pred(v) { return mode === 1 ? sig(w * (v - L50)) : 0.5 + w * (v - L50); }
    function draw() {
      var xs = X1(), ys = Y1(), n = xs.length, i, loss = 0, hits = 0, bad = false;
      for (i = 0; i < n; i++) { var p = pred(xs[i]); if (mode === 1) loss += lossTerm(p, ys[i]); else { loss += Math.pow(p - ys[i], 2); if (p < 0 || p > 1) bad = true; } if ((p >= 0.5) === (ys[i] === 1)) hits++; }
      loss /= n;
      var cx = lin(15, 85, 141), tag = TAGS[3305], pt = pred(tag.L), b = -w * L50;
      Kit.plot("sg-plot", function (P) {
        var ix0 = [], iy0 = [], ix1 = [], iy1 = [];
        for (i = 0; i < L.length; i++) { if (Y[i]) { ix1.push(L[i]); iy1.push(1 + jy[i]); } else { ix0.push(L[i]); iy0.push(0 + jy[i]); } }
        var tr = [
          { x: [15, 85], y: [0.5, 0.5], mode: "lines", line: { color: P.rule, width: 1.5, dash: "dot" }, hoverinfo: "skip" },
          { x: ix0, y: iy0, mode: "markers", marker: { color: P.navy, size: 6, symbol: "circle", opacity: 0.6 }, hovertemplate: "juvenile, %{x:.1f} cm<extra>made-up record</extra>" },
          { x: ix1, y: iy1, mode: "markers", marker: { color: P.coral, size: 7, symbol: "diamond", opacity: 0.6 }, hovertemplate: "adult, %{x:.1f} cm<extra>made-up record</extra>" }
        ];
        if (extra) tr.push({ x: new Array(100).fill(80), y: new Array(100).fill(1).map(function (v, k) { return 1 + (((k * 37) % 100) / 100 - 0.5) * 0.07; }), mode: "markers", marker: { color: P.coral, size: 7, symbol: "diamond", opacity: 0.35, line: { color: P.coral, width: 1 } }, hoverinfo: "skip" });
        tr.push({ x: cx, y: cx.map(pred), mode: "lines", line: { color: P.teal, width: 3.5 }, hoverinfo: "skip" });
        tr.push({ x: [tag.L], y: [pt], mode: "markers+text", text: ["Tag 3305"], textposition: "top left", textfont: { color: P.coral, size: 12.5 }, marker: { color: P.paper, size: 11, line: { color: P.coral, width: 3 } }, hoverinfo: "skip" });
        return tr;
      }, function (P) {
        return { margin: { l: 58, r: 12, t: 10, b: 46 }, dragmode: false, xaxis: { title: { text: "length, cm" }, range: [15, 85] }, yaxis: { title: { text: mode === 1 ? "probability of adult" : "model output" }, range: [-0.3, 1.3], tickvals: [0, 0.5, 1] },
          annotations: [{ x: 16, y: 1.12, xanchor: "left", showarrow: false, text: "adult (1)", font: { size: 12.5, color: P.coral } }, { x: 16, y: -0.12, xanchor: "left", showarrow: false, text: "juvenile (0)", font: { size: 12.5, color: P.navy } },
            { x: L50, y: -0.24, xanchor: "center", showarrow: false, text: "midpoint " + Kit.fmt(L50, 1) + " cm", font: { size: 12.5, color: P.teal } }],
          shapes: [{ type: "line", x0: L50, x1: L50, y0: -0.3, y1: 1.3, line: { color: P.teal, width: 1.5, dash: "dash" } }] };
      });
      Kit.setText("sg-L50", Kit.fmt(L50, 1) + " cm"); Kit.setText("sg-b", Kit.fmt(b, 2)); Kit.setText("sg-w", Kit.fmt(w, 2));
      Kit.setText("sg-loss-name", mode === 1 ? "Log loss" : "Mean squared error"); Kit.setText("sg-loss", Kit.fmt(loss, 3)); Kit.setText("sg-acc", Kit.pct(hits / n, 1));
      var z = mode === 1 ? w * tag.L + b : null;
      Kit.$("sg-eqn").innerHTML = mode === 1
        ? box("c-sand", Kit.fmt(w, 2), "w, per cm") + op("&times;") + box("c-navy", "41.0", "length of Tag 3305") + op("+") + box("c-sand", Kit.fmt(b, 2), "b = &minus;w &times; midpoint") + op("=") + box("c-coral", Kit.fmt(z, 2), "score z") + op("&rarr; &sigma;(z) =") + box("c-teal", Kit.fmt(pt, 3), "probability of adult (true label: adult)")
        : box("c-teal", Kit.fmt(pt, 3), "output for Tag 3305") + op("=") + box("c-sand", "0.5", "value at the midpoint") + op("+") + box("c-sand", Kit.fmt(w, 3), "slope, per cm") + op("&times;") + box("c-navy", Kit.fmt(tag.L - L50, 1), "length minus midpoint, cm");
      Kit.setText("sg-status", mode === 0 ? (bad ? "A straight line gives outputs below 0 and above 1 for many sharks. They cannot be probabilities." : "") : "");
      if (mode === 0) { var lo = pred(18), hi = pred(80); Kit.setText("sg-status", "At 18 cm the line says " + Kit.fmt(lo, 2) + " and at 80 cm it says " + Kit.fmt(hi, 2) + ". Neither can be a probability."); }
      else Kit.setText("sg-status", "The sigmoid stays between 0 and 1 at any length.");
    }
    fitBest(); w = 0.15; L50 = 45; sw.set(w); sm.set(L50); draw();
  }

  // ---------- shared: the probability map on (length, girth) ----------
  var MAP = { L0: 15, L1: 65, G0: 5, G1: 19 };
  function zOf(Lv, Gv) { return [(Lv - ML) / SL, (Gv - MG) / SG]; }
  function boundaryEnds(wv) {
    // points where w1 (L-ML)/SL + w2 (G-MG)/SG + b = 0 meets the map rectangle
    var pts = [], w1 = wv[0] / SL, w2 = wv[1] / SG, c = wv[2] - wv[0] * ML / SL - wv[1] * MG / SG; // w1 L + w2 G + c = 0
    function add(Lv, Gv) { if (Lv >= MAP.L0 - 1e-6 && Lv <= MAP.L1 + 1e-6 && Gv >= MAP.G0 - 1e-6 && Gv <= MAP.G1 + 1e-6) pts.push([Lv, Gv]); }
    if (Math.abs(w2) > 1e-9) { add(MAP.L0, -(w1 * MAP.L0 + c) / w2); add(MAP.L1, -(w1 * MAP.L1 + c) / w2); }
    if (Math.abs(w1) > 1e-9) { add(-(w2 * MAP.G0 + c) / w1, MAP.G0); add(-(w2 * MAP.G1 + c) / w1, MAP.G1); }
    if (pts.length < 2) return null;
    var best = [pts[0], pts[1]], dmax = 0, i, j; for (i = 0; i < pts.length; i++) for (j = i + 1; j < pts.length; j++) { var d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]); if (d > dmax) { dmax = d; best = [pts[i], pts[j]]; } }
    return best;
  }
  function probGrid(wv) {
    var xs = lin(MAP.L0, MAP.L1, 64), ys = lin(MAP.G0, MAP.G1, 44);
    return { xs: xs, ys: ys, z: ys.map(function (g) { return xs.map(function (l) { var u = zOf(l, g); return sig(wv[0] * u[0] + wv[1] * u[1] + wv[2]); }); }) };
  }
  function mapTraces(P, wv, pts, opts) {
    var g = probGrid(wv), tr = [];
    tr.push({ type: "contour", x: g.xs, y: g.ys, z: g.z, zmin: 0, zmax: 1, showscale: false, contours: { coloring: "heatmap", start: 0.1, end: 0.9, size: 0.1 }, line: { color: rgba(P.ink2, 0.35), width: 0.8 },
      colorscale: [[0, rgba(P.navy, 0.62)], [0.5, rgba(P.paper, 0.2)], [1, rgba(P.coral, 0.62)]], hovertemplate: "length %{x:.1f}, girth %{y:.1f}: p(adult) %{z:.2f}<extra></extra>" });
    var j0 = [], j1 = [];
    pts.forEach(function (q) { var z = wv[0] * (q.L - ML) / SL + wv[1] * (q.G - MG) / SG + wv[2]; q.wrong = (z >= 0) !== (q.y === 1); (q.y ? j1 : j0).push(q); });
    function trace(arr, col, sym, name) {
      return { x: arr.map(function (q) { return q.L; }), y: arr.map(function (q) { return q.G; }), mode: "markers", type: "scatter",
        marker: { color: col, symbol: sym, size: arr.map(function (q) { return q.mine ? 11 : 7; }), line: { color: arr.map(function (q) { return q.wrong ? P.ink : P.bg; }), width: arr.map(function (q) { return q.wrong ? 2.4 : (q.mine ? 2 : 1); }) } },
        hovertemplate: name + ", %{x:.1f} cm, girth %{y:.1f}<extra></extra>" };
    }
    tr.push(trace(j0, P.navy, "circle", "juvenile")); tr.push(trace(j1, P.coral, "diamond", "adult"));
    var tg = TAGS[3305];
    tr.push({ x: [tg.L], y: [tg.G], mode: "markers+text", text: ["Tag 3305"], textposition: "top left", textfont: { size: 12.5, color: P.ink }, marker: { color: "rgba(0,0,0,0)", size: 17, line: { color: P.ink, width: 2 } }, hoverinfo: "skip" });
    return tr;
  }
  function accOf(wv, pts) {
    var hits = 0, loss = 0; pts.forEach(function (q) { var z = wv[0] * (q.L - ML) / SL + wv[1] * (q.G - MG) / SG + wv[2]; if ((z >= 0) === (q.y === 1)) hits++; loss += lossTerm(sig(z), q.y); });
    return { acc: hits / pts.length, loss: loss / pts.length, wrong: pts.length - hits };
  }

  // ---------- 2. two inputs and a decision boundary ----------
  function boundaryFig() {
    var wv = [-1.5, 2.5, 0.0], mine = [], mode = 0, base = SP.map(function (q) { return { L: q.length_cm, G: q.girth_cm, y: q.mature }; });
    var s1 = Kit.slider(Kit.$("bd-controls"), { label: "Weight on length, w₁", min: -10, max: 10, step: 0.05, value: wv[0], digits: 2 }, function (v) { wv[0] = v; draw(); });
    var s2 = Kit.slider(Kit.$("bd-controls"), { label: "Weight on girth, w₂", min: -10, max: 10, step: 0.05, value: wv[1], digits: 2 }, function (v) { wv[1] = v; draw(); });
    var s3 = Kit.slider(Kit.$("bd-controls"), { label: "Bias b", min: -10, max: 10, step: 0.05, value: wv[2], digits: 2 }, function (v) { wv[2] = v; draw(); });
    var bar = Kit.$("bd-bar");
    Kit.chips(bar, ["Click moves the boundary", "Click adds a juvenile", "Click adds an adult"], 0, function (i) { mode = i; Kit.setText("bd-mode", ["Click the map to move the boundary there.", "Click the map to add a juvenile shark (made up by you).", "Click the map to add an adult shark (made up by you)."][i]); });
    Kit.button(bar, "Fit the best boundary", function () { fit(); });
    Kit.button(bar, "Remove my sharks", function () { mine = []; draw(); });
    Kit.button(bar, "Start again", function () { mine = []; wv = [-1.5, 2.5, 0]; sync(); draw(); });
    function sync() { s1.set(wv[0]); s2.set(wv[1]); s3.set(wv[2]); }
    function allPts() { return base.concat(mine); }
    function fit() {
      var pts = allPts(), Z = pts.map(function (q) { return zOf(q.L, q.G); }), yy = pts.map(function (q) { return q.y; }), f = fitLogistic(Z, yy, 1e-3);
      wv = [clamp(f[0], -10, 10), clamp(f[1], -10, 10), clamp(f[2], -10, 10)]; sync(); draw();
    }
    var node = Kit.$("bd-plot"), wired = false;
    function wire() {
      if (wired || !node.on) return; wired = true;
      node.on("plotly_click", function (e) {
        var p = e.points && e.points[0]; if (!p || p.x == null || p.y == null) return;
        var Lc = clamp(p.x, MAP.L0, MAP.L1), Gc = clamp(p.y, MAP.G0, MAP.G1);
        if (mode === 0) { var u = zOf(Lc, Gc); wv[2] = clamp(-(wv[0] * u[0] + wv[1] * u[1]), -10, 10); s3.set(wv[2]); draw(); }
        else { mine.push({ L: Lc, G: Gc, y: mode === 2 ? 1 : 0, mine: true }); draw(); }
      });
      node.on("plotly_relayout", function (e) {
        if (e["shapes[0].x0"] == null && e["shapes[0].y0"] == null && e["shapes[0].x1"] == null && e["shapes[0].y1"] == null) return;
        var sh = node.layout.shapes && node.layout.shapes[0]; if (!sh) return;
        var u0 = zOf(sh.x0, sh.y0), u1 = zOf(sh.x1, sh.y1), dx = u1[0] - u0[0], dy = u1[1] - u0[1], dl = Math.hypot(dx, dy); if (dl < 1e-6) return;
        var mag = Math.max(0.5, Math.hypot(wv[0], wv[1])), nx = -dy / dl, ny = dx / dl;
        if (nx * wv[0] + ny * wv[1] < 0) { nx = -nx; ny = -ny; }
        wv = [clamp(mag * nx, -10, 10), clamp(mag * ny, -10, 10), clamp(-mag * (nx * u0[0] + ny * u0[1]), -10, 10)]; sync(); draw();
      });
    }
    function draw() {
      var pts = allPts(), r = accOf(wv, pts), ends = boundaryEnds(wv), tg = TAGS[3305], u = zOf(tg.L, tg.G), z = wv[0] * u[0] + wv[1] * u[1] + wv[2];
      Kit.plot("bd-plot", function (P) { return mapTraces(P, wv, pts); }, function (P) {
        return { margin: { l: 56, r: 12, t: 10, b: 46 }, dragmode: false, hovermode: "closest", xaxis: { title: { text: "length, cm" }, range: [MAP.L0, MAP.L1] }, yaxis: { title: { text: "girth, cm" }, range: [MAP.G0, MAP.G1] },
          shapes: ends ? [{ type: "line", x0: ends[0][0], y0: ends[0][1], x1: ends[1][0], y1: ends[1][1], line: { color: P.ink, width: 3 }, editable: true }] : [] };
      }, { edits: { shapePosition: true } });
      wire();
      Kit.setText("bd-acc", Kit.pct(r.acc, 1)); Kit.setText("bd-wrong", String(r.wrong) + " of " + pts.length); Kit.setText("bd-loss", Kit.fmt(r.loss, 3));
      Kit.setText("bd-line", ends ? "" : "The boundary is outside the map: all sharks get the same call.");
      Kit.$("bd-eqn").innerHTML = box("c-sand", Kit.fmt(wv[0], 2), "w<sub>1</sub>") + op("&times;") + box("c-navy", Kit.fmt(u[0], 3), "length 41.0, scaled") + op("+") + box("c-sand", Kit.fmt(wv[1], 2), "w<sub>2</sub>") + op("&times;") + box("c-navy", Kit.fmt(u[1], 3), "girth 11.5, scaled") + op("+") + box("c-sand", Kit.fmt(wv[2], 2), "b") +
        op("=") + box("c-coral", Kit.fmt(z, 2), "score z") + op("&rarr;") + box("c-teal", Kit.pct(sig(z), 1), "probability of adult (true label: adult)");
    }
    sync(); draw();
  }

  // ---------- 3. log loss against squared loss ----------
  function lossFig() {
    var tag = 3305, z = 0;
    var sl = Kit.slider(Kit.$("ls-controls"), { label: "Score z the model gives the shark", min: -7, max: 7, step: 0.05, value: z, digits: 2 }, function (v) { z = v; draw(); });
    Kit.chips(Kit.$("ls-bar"), ["Tag 3305: adult (label 1)", "Tag 4127: juvenile (label 0)"], 0, function (i) { tag = i === 0 ? 3305 : 4127; z = i === 0 ? -5 : 5; sl.set(z); draw(); });
    z = -5; sl.set(z);
    var ps = lin(0.002, 0.998, 200), zs = lin(-7, 7, 200);
    function draw() {
      var y = TAGS[tag].mature, p = sig(z), lg = lossTerm(p, y), sq = Math.pow(p - y, 2), gl = p - y, gs = 2 * (p - y) * p * (1 - p);
      Kit.plot("ls-loss", function (P) {
        return [
          { x: ps, y: ps.map(function (q) { return lossTerm(q, y); }), mode: "lines", line: { color: P.coral, width: 3.5 }, hoverinfo: "skip" },
          { x: ps, y: ps.map(function (q) { return Math.pow(q - y, 2); }), mode: "lines", line: { color: P.navy, width: 3.5 }, hoverinfo: "skip" },
          { x: [p], y: [Math.min(lg, 5.6)], mode: "markers", marker: { color: P.coral, size: 12, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" },
          { x: [p], y: [sq], mode: "markers", marker: { color: P.navy, size: 12, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" }];
      }, function (P) {
        return { margin: { l: 56, r: 12, t: 10, b: 50 }, dragmode: false, xaxis: { title: { text: "probability of adult, p" }, range: [0, 1] }, yaxis: { title: { text: "loss for this shark" }, range: [-0.1, 5.8] },
          annotations: [{ x: y ? 0.3 : 0.7, y: y ? 3.3 : 3.3, xanchor: y ? "left" : "right", showarrow: false, text: "log loss: " + (y ? "\u2212ln p" : "\u2212ln(1 \u2212 p)"), font: { size: 13, color: P.coral } },
            { x: y ? 0.6 : 0.4, y: 1.0, xanchor: y ? "left" : "right", showarrow: false, text: "squared loss: (p \u2212 " + y + ")<sup>2</sup>", font: { size: 13, color: P.navy } }] };
      });
      Kit.plot("ls-slope", function (P) {
        return [
          { x: zs, y: zs.map(function (v) { return Math.abs(sig(v) - y); }), mode: "lines", line: { color: P.coral, width: 3.5 }, hoverinfo: "skip" },
          { x: zs, y: zs.map(function (v) { var q = sig(v); return Math.abs(2 * (q - y) * q * (1 - q)); }), mode: "lines", line: { color: P.navy, width: 3.5 }, hoverinfo: "skip" },
          { x: [z], y: [Math.abs(gl)], mode: "markers", marker: { color: P.coral, size: 12, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" },
          { x: [z], y: [Math.abs(gs)], mode: "markers", marker: { color: P.navy, size: 12, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" }];
      }, function (P) {
        var wrongSide = y ? 1 : 6.2, rightSide = y ? 6.2 : 1;
        return { margin: { l: 56, r: 12, t: 10, b: 50 }, dragmode: false, xaxis: { title: { text: "score z" }, range: [-7, 7] }, yaxis: { title: { text: "push on z (slope of the loss)" }, range: [-0.03, 1.1] },
          annotations: [{ x: y ? -6.8 : 6.8, y: 1.05, xanchor: y ? "left" : "right", showarrow: false, text: "confidently wrong", font: { size: 12.5, color: P.muted } }, { x: y ? 6.8 : -6.8, y: 0.45, xanchor: y ? "right" : "left", showarrow: false, text: "confidently right", font: { size: 12.5, color: P.muted } }] };
      });
      Kit.$("ls-eqn").innerHTML = box("c-sand", Kit.fmt(z, 2), "score z") + op("&rarr; &sigma;(z) =") + box("c-teal", Kit.fmt(p, p < 0.01 || p > 0.99 ? 4 : 3), "probability of adult p") + op("&rarr;") + box("c-coral", Kit.fmt(lg, 3), "log loss") + box("c-navy", Kit.fmt(sq, 3), "squared loss") +
        op("&rarr; slopes:") + box("c-coral", Kit.fmt(gl, 3), "log loss: p &minus; y") + box("c-navy", Kit.fmt(gs, 3), "squared: 2(p &minus; y) p (1 &minus; p)");
      Kit.setText("ls-status", (p >= 0.5) === (y === 1) ? "The model is on the right side. Both losses are small, and both slopes shrink." : (Math.abs(z) > 4 ? "Confidently wrong: log loss is large and keeps its full push. The squared loss is at most 1 and its push has nearly vanished." : "The model is on the wrong side."));
    }
    draw();
  }

  // ---------- 4. training by gradient descent ----------
  function trainFig() {
    var Z = SP.map(function (q) { return zOf(q.length_cm, q.girth_cm); }), pts = SP.map(function (q) { return { L: q.length_cm, G: q.girth_cm, y: q.mature }; });
    var eta = 0.5, start = [-1, 1, 1], traj = [], k = 0, stop = null, MAXS = 3000;
    var sl = Kit.slider(Kit.$("tr-controls"), { label: "Learning rate η", min: -2, max: 0.7, step: 0.01, value: Math.log10(eta), show: function (v) { return Kit.fmt(Math.pow(10, v), 2); } }, function (v) { eta = Math.pow(10, v); if (stop) stop(); recompute(); k = Math.min(k, MAXS); draw(); });
    var bar = Kit.$("tr-bar");
    Kit.button(bar, "Step", function () { if (stop) stop(); k = Math.min(MAXS, k + 1); draw(); });
    Kit.button(bar, "Run", function () { if (stop) stop(); var k0 = k; stop = Kit.animate(7000, function (p) { k = Math.min(MAXS, Math.round(k0 + (400 - k0) * p * p)); draw(); if (p >= 1) stop = null; }); });
    Kit.button(bar, "Jump to 3,000 steps", function () { if (stop) stop(); k = MAXS; draw(); });
    Kit.button(bar, "Reset", function () { if (stop) stop(); k = 0; draw(); });
    function recompute() {
      var w = start.slice(), n = Z.length; traj = [];
      for (var t = 0; t <= MAXS; t++) {
        var g = [0, 0, 0], loss = 0, hits = 0, i;
        for (i = 0; i < n; i++) { var z = w[0] * Z[i][0] + w[1] * Z[i][1] + w[2], p = sig(z), r = p - pts[i].y; g[0] += r * Z[i][0] / n; g[1] += r * Z[i][1] / n; g[2] += r / n; loss += lossTerm(p, pts[i].y); if ((z >= 0) === (pts[i].y === 1)) hits++; }
        traj.push({ w: w.slice(), g: g, loss: loss / n, acc: hits / n });
        w = [w[0] - eta * g[0], w[1] - eta * g[1], w[2] - eta * g[2]];
      }
    }
    recompute();
    function draw() {
      var cur = traj[k], wv = cur.w, ends = boundaryEnds(wv), ghosts = [0, 1, 3, 10, 30, 100, 300, 1000].filter(function (t) { return t < k; });
      Kit.plot("tr-map", function (P) { return mapTraces(P, wv, pts); }, function (P) {
        var shapes = ghosts.map(function (t) { var e = boundaryEnds(traj[t].w); return e ? { type: "line", x0: e[0][0], y0: e[0][1], x1: e[1][0], y1: e[1][1], line: { color: rgba(P.ink2, 0.55), width: 1.5, dash: "dot" } } : null; }).filter(Boolean);
        if (ends) shapes.push({ type: "line", x0: ends[0][0], y0: ends[0][1], x1: ends[1][0], y1: ends[1][1], line: { color: P.ink, width: 3 } });
        return { margin: { l: 56, r: 12, t: 10, b: 46 }, dragmode: false, xaxis: { title: { text: "length, cm" }, range: [MAP.L0, MAP.L1] }, yaxis: { title: { text: "girth, cm" }, range: [MAP.G0, MAP.G1] }, shapes: shapes };
      });
      var shown = traj.slice(0, k + 1), xmax = Math.max(10, k);
      Kit.plot("tr-loss", function (P) {
        var idx = []; for (var t = 0; t <= k; t += Math.max(1, Math.floor(k / 300))) idx.push(t); if (idx[idx.length - 1] !== k) idx.push(k);
        return [{ x: idx, y: idx.map(function (t) { return traj[t].loss; }), mode: "lines+markers", line: { color: P.coral, width: 2.5 }, marker: { color: P.coral, size: idx.length > 40 ? 0 : 6 }, hovertemplate: "step %{x}: log loss %{y:.4f}<extra></extra>" },
          { x: [0, xmax], y: [0.2318, 0.2318], mode: "lines", line: { color: P.teal, width: 1.5, dash: "dot" }, hoverinfo: "skip" }];
      }, { margin: { l: 56, r: 12, t: 10, b: 46 }, dragmode: false, xaxis: { title: { text: "step" }, range: [0, xmax] }, yaxis: { title: { text: "log loss" }, range: [0.2, 0.72] } });
      Kit.setText("tr-step", String(k)); Kit.setText("tr-loss-now", Kit.fmt(cur.loss, 4)); Kit.setText("tr-acc", Kit.pct(cur.acc, 1)); Kit.setText("tr-w", "(" + Kit.fmt(wv[0], 2) + ", " + Kit.fmt(wv[1], 2) + ")"); Kit.setText("tr-b", Kit.fmt(wv[2], 2));
      var nx = traj[Math.min(MAXS, k + 1)].w;
      Kit.$("tr-eqn").innerHTML = box("c-teal", Kit.fmt(nx[0], 3), "w<sub>1</sub> next") + op("=") + box("c-sand", Kit.fmt(wv[0], 3), "w<sub>1</sub> now") + op("&minus;") + box("c-navy", Kit.fmt(eta, 2), "&eta;") + op("&times;") + box("c-coral", Kit.fmt(cur.g[0], 4), "average of (p &minus; y) &times; scaled length");
      Kit.$("tr-eqn2").innerHTML = box("c-teal", Kit.fmt(nx[2], 3), "b next") + op("=") + box("c-sand", Kit.fmt(wv[2], 3), "b now") + op("&minus;") + box("c-navy", Kit.fmt(eta, 2), "&eta;") + op("&times;") + box("c-coral", Kit.fmt(cur.g[2], 4), "average of (p &minus; y)");
      Kit.setText("tr-status", k === 0 ? "Start: a boundary tilted the wrong way. The log loss is " + Kit.fmt(cur.loss, 3) + ", about the value for guessing 50% everywhere (ln 2 = 0.693) or worse." : (k >= MAXS ? "After 3,000 steps the loss is " + Kit.fmt(cur.loss, 4) + ", close to the best possible 0.2318." : "Step " + k + ": log loss " + Kit.fmt(cur.loss, 4) + ", accuracy " + Kit.pct(cur.acc, 1) + "."));
    }
    draw();
  }

  // ---------- 5. five species ----------
  function softmaxFig() {
    var S = SOFT, K = S.classes.length, ZOOM = [[15, 110, 5, 40], [15, 200, 5, 85]], zoom = 0, cur = { L: 41.0, G: 11.5 }, selTag = 3305, node = Kit.$("sm-map");
    var sL = Kit.slider(Kit.$("sm-controls"), { label: "Length, cm", min: 15, max: 200, step: 0.5, value: cur.L, digits: 1 }, function (v) { cur.L = v; selTag = 0; draw(); });
    var sG = Kit.slider(Kit.$("sm-controls"), { label: "Girth, cm", min: 5, max: 85, step: 0.1, value: cur.G, digits: 1 }, function (v) { cur.G = v; selTag = 0; draw(); });
    var bar = Kit.$("sm-bar");
    Kit.chips(bar, ["Zoom to the small species", "Whole range"], 0, function (i) { zoom = i; drawMap(); });
    [[3305, "Tag 3305"], [4127, "Tag 4127"], [5519, "Tag 5519"]].forEach(function (t) { Kit.button(bar, t[1], function () { cur = { L: TAGS[t[0]].L, G: TAGS[t[0]].G }; selTag = t[0]; sL.set(cur.L); sG.set(cur.G); draw(); }); });
    function scores(l, g) { return S.classes.map(function (c, k) { return S.w_length[k] * l + S.w_girth[k] * g + S.b[k]; }); }
    function soft(z) { var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - m); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (v) { return v / s; }); }
    function gridWin(zm) {
      var xs = lin(ZOOM[zm][0], ZOOM[zm][1], 150), ys = lin(ZOOM[zm][2], ZOOM[zm][3], 100);
      return { xs: xs, ys: ys, z: ys.map(function (g) { return xs.map(function (l) { var z = scores(l, g), b = 0, k; for (k = 1; k < K; k++) if (z[k] > z[b]) b = k; return b; }); }) };
    }
    var grids = [gridWin(0), gridWin(1)], sym = { spadenose: "circle", milk: "square", bamboo: "triangle-up", hammerhead: "diamond", ganges: "star" };
    function drawMap() {
      var P0 = Kit.palette(), gw = grids[zoom], Z = ZOOM[zoom];
      Kit.plot("sm-map", function (P) {
        var C = spCol(P), cs = SPECIES.map(function (s) { return rgba(C[s], 0.28); }), steps = [];
        cs.forEach(function (c, k) { steps.push([k / 5, c]); steps.push([(k + 1) / 5, c]); });
        var tr = [{ type: "heatmap", x: gw.xs, y: gw.ys, z: gw.z, zmin: 0, zmax: 4.9999, showscale: false, colorscale: steps, hoverinfo: "skip" }];
        SPECIES.forEach(function (s) {
          var rs = REC.filter(function (r) { return r.species === s && r.length_cm >= Z[0] && r.length_cm <= Z[1] && r.girth_cm >= Z[2] && r.girth_cm <= Z[3]; });
          tr.push({ x: rs.map(function (r) { return r.length_cm; }), y: rs.map(function (r) { return r.girth_cm; }), mode: "markers", marker: { color: C[s], symbol: sym[s], size: s === "ganges" ? 11 : 6, opacity: 0.75, line: { color: P.bg, width: 0.8 } }, hovertemplate: NAME[s] + ", %{x:.1f} cm, girth %{y:.1f}<extra>made-up record</extra>" });
        });
        tr.push({ x: [cur.L], y: [cur.G], mode: "markers", marker: { color: "rgba(0,0,0,0)", size: 18, line: { color: P.ink, width: 3 } }, hoverinfo: "skip" });
        return tr;
      }, function (P) {
        var C = spCol(P), ann = [], cen = SPECIES.map(function () { return { x: 0, y: 0, n: 0 }; });
        gw.z.forEach(function (row, j) { row.forEach(function (k, i) { cen[k].x += gw.xs[i]; cen[k].y += gw.ys[j]; cen[k].n++; }); });
        cen.forEach(function (c, k) { if (c.n > 25) ann.push({ x: c.x / c.n, y: c.y / c.n, text: NAME[SPECIES[k]], showarrow: false, font: { size: 13, color: P.ink }, bgcolor: rgba(P.paper, 0.75), borderpad: 2 }); });
        return { margin: { l: 56, r: 12, t: 10, b: 46 }, dragmode: false, hovermode: "closest", annotations: ann, xaxis: { title: { text: "length, cm" }, range: [ZOOM[zoom][0], ZOOM[zoom][1]] }, yaxis: { title: { text: "girth, cm" }, range: [ZOOM[zoom][2], ZOOM[zoom][3]] } };
      });
      if (node.on && !node._wired) { node._wired = true; node.on("plotly_click", function (e) { var p = e.points && e.points[0]; if (!p) return; cur = { L: clamp(p.x, 15, 200), G: clamp(p.y, 5, 85) }; selTag = 0; sL.set(cur.L); sG.set(cur.G); draw(); }); }
    }
    function draw() {
      var z = scores(cur.L, cur.G), p = soft(z), top = p.indexOf(Math.max.apply(null, p)), tb = Kit.$("sm-tbody"), P = Kit.palette(), C = spCol(P);
      tb.innerHTML = "";
      S.classes.forEach(function (c, k) {
        var tr = Kit.html("tr", { class: k === top ? "top" : "" }, tb);
        Kit.html("td", {}, tr, NAME[c]);
        Kit.html("td", {}, tr, Kit.fmt(S.w_length[k], 3) + " × " + cur.L.toFixed(1) + " = " + Kit.fmt(S.w_length[k] * cur.L, 2));
        Kit.html("td", {}, tr, Kit.fmt(S.w_girth[k], 3) + " × " + cur.G.toFixed(1) + " = " + Kit.fmt(S.w_girth[k] * cur.G, 2));
        Kit.html("td", {}, tr, Kit.fmt(S.b[k], 2));
        Kit.html("td", { class: "sc" }, tr, Kit.fmt(z[k], 2));
        var pc = Kit.html("td", { class: "pb" }, tr), b = Kit.html("span", { class: "bar" }, pc), f = Kit.html("span", { class: "fill" }, b);
        f.style.width = Math.max(1, p[k] * 100) + "%"; f.style.background = k === top ? "var(--teal)" : "var(--navy)";
        Kit.html("span", { class: "pv" }, pc, Kit.pct(p[k], p[k] < 0.01 ? 2 : 1));
      });
      var m = Math.max.apply(null, z), ex = z.map(function (v) { return Math.exp(v - m); }), tot = ex.reduce(function (a, b2) { return a + b2; }, 0);
      Kit.$("sm-eqn").innerHTML = box("c-coral", Kit.fmt(z[top], 2), "score of " + NAME[S.classes[top]]) + op("&rarr; e<sup>z</sup> &divide; (sum of e<sup>z</sup>) =") + box("c-navy", Kit.fmt(Math.exp(z[top] - m), 3), "e<sup>z &minus; z<sub>max</sub></sup> (shifted for safety)") + op("&divide;") + box("", Kit.fmt(tot, 3), "sum over the five species") + op("=") + box("c-teal", Kit.pct(p[top], 1), "probability of " + NAME[S.classes[top]]);
      var tag = selTag ? TAGS[selTag] : null;
      Kit.setText("sm-status", tag ? "Tag " + selTag + " is a " + NAME[tag.sp] + ". The model says " + NAME[S.classes[top]] + " with " + Kit.pct(p[top], 0) + (S.classes[top] === tag.sp ? ": right." : (": wrong. It gives the true species " + Kit.pct(p[S.classes.indexOf(tag.sp)], 0) + ".")) : "The model says " + NAME[S.classes[top]] + " with " + Kit.pct(p[top], 0) + " for a made-up shark at " + cur.L.toFixed(1) + " cm and girth " + cur.G.toFixed(1) + " cm.");
      var dot = node.data; drawMap();
    }
    draw();
  }
})();
