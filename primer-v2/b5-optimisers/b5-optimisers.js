// B5 Optimisers: figures on a narrow, tilted valley and on single weights (all numbers are made up or computed from the made-up catch table).
//   raceFig(cfg)  optimisers racing on the valley: Plotly contour with paths, loss curves, Run / Step / Reset, a step scrubber, live update boxes
//                 used four times: SGD alone, SGD against momentum, SGD against RMSProp, and all four
//   emaFig        m and v for one weight with and without bias correction
//   decayFig      L2 penalty inside Adam against decoupled weight decay (AdamW)
//   scheduleFig   warm-up and cosine decay, and the size of Adam's update under the schedule
//   stateFig      resume training with m and v reset to zero: the first steps jump
//   fisherFig     Adam's v against the mean squared slope and the curvature (data/fisher.json)
(function () {
  "use strict";

  // ---------- the valley: L = 1/2 theta^T H theta, curvatures 0.3 along the valley and 25 across it ----------
  var SLOW = 0.3, FAST = 25, NSTEP = 200, THR = 0.01;
  var KINDS = {
    sgd: { name: "SGD", color: "navy", symbol: "circle", dash: "solid" },
    mom: { name: "Momentum", color: "teal", symbol: "square", dash: "solid" },
    rms: { name: "RMSProp", color: "sand", symbol: "diamond", dash: "solid" },
    adam: { name: "Adam", color: "coral", symbol: "triangle-up", dash: "solid" }
  };
  function valley(phiDeg) {
    var p = phiDeg * Math.PI / 180, c = Math.cos(p), s = Math.sin(p);
    return { hxx: SLOW * c * c + FAST * s * s, hyy: SLOW * s * s + FAST * c * c, hxy: (SLOW - FAST) * c * s, start: [-2 * c - 0.45 * s, -2 * s + 0.45 * c], phi: phiDeg };
  }
  function lossV(V, x, y) { return 0.5 * (V.hxx * x * x + 2 * V.hxy * x * y + V.hyy * y * y); }
  function gradV(V, x, y) { return [V.hxx * x + V.hxy * y, V.hxy * x + V.hyy * y]; }
  // one optimiser from the start of the valley for n steps; noise holds 2n standard normals shared by all optimisers
  function runOpt(kind, V, cfg, n, noise) {
    var th = [V.start[0], V.start[1]], m = [0, 0], v = [0, 0], out = [], t, i, dead = false;
    out.push({ th: th.slice(), L: lossV(V, th[0], th[1]), g: [0, 0], m: [0, 0], v: [0, 0], mh: [0, 0], vh: [0, 0], st: [0, 0] });
    for (t = 1; t <= n; t++) {
      var prev = out[out.length - 1];
      if (dead) { out.push(prev); continue; }
      var g = gradV(V, th[0], th[1]); g[0] += cfg.sigma * noise[2 * t - 2]; g[1] += cfg.sigma * noise[2 * t - 1];
      var st = [0, 0], mh = [0, 0], vh = [0, 0];
      for (i = 0; i < 2; i++) {
        if (kind === "sgd") st[i] = cfg.eta * g[i];
        else if (kind === "mom") { m[i] = cfg.b1 * m[i] + (1 - cfg.b1) * g[i]; st[i] = cfg.eta * m[i]; }
        else if (kind === "rms") { v[i] = cfg.b2 * v[i] + (1 - cfg.b2) * g[i] * g[i]; st[i] = cfg.eta * g[i] / (Math.sqrt(v[i]) + cfg.eps); }
        else {
          m[i] = cfg.b1 * m[i] + (1 - cfg.b1) * g[i]; v[i] = cfg.b2 * v[i] + (1 - cfg.b2) * g[i] * g[i];
          mh[i] = m[i] / (1 - Math.pow(cfg.b1, t)); vh[i] = v[i] / (1 - Math.pow(cfg.b2, t));
          st[i] = cfg.eta * mh[i] / (Math.sqrt(vh[i]) + cfg.eps);
        }
      }
      th[0] -= st[0]; th[1] -= st[1];
      var L = lossV(V, th[0], th[1]);
      if (!isFinite(L) || L > 1e6) { dead = true; out.push({ th: prev.th, L: 1e6, g: g, m: m.slice(), v: v.slice(), mh: mh, vh: vh, st: st, diverged: true }); continue; }
      out.push({ th: th.slice(), L: L, g: g, m: m.slice(), v: v.slice(), mh: mh, vh: vh, st: st });
    }
    return out;
  }
  function firstBelow(run, thr) { for (var i = 0; i < run.length; i++) if (run[i].L < thr) return i; return -1; }

  // ---------- small helpers ----------
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { var C = hexRGB(c); return "rgba(" + C[0] + "," + C[1] + "," + C[2] + "," + a + ")"; }
  function mix(a, b, t) { var A = hexRGB(a), B = hexRGB(b); return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }).join(",") + ")"; }
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function nf(v) { if (!isFinite(v)) return "inf"; var a = Math.abs(v); return Kit.fmt(v, a >= 100 ? 1 : a >= 10 ? 2 : a >= 0.095 ? 3 : a === 0 ? 3 : a >= 0.0095 ? 4 : 5); }
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function noiseArr(seed, n) { var r = Kit.rng(seed), o = [], i; for (i = 0; i < n; i++) o.push(Kit.gauss(r)); return o; }
  function col(P, kind) { return P[KINDS[kind].color]; }

  // ---------- 1 to 4. the race ----------
  // cfg: {id, kinds, eta: {kind: value}, sliders: ["eta", "b1", "b2", "noise", "tilt"], b1, b2, noise, tilt, coord}
  function raceFig(cfg) {
    var id = cfg.id, kinds = cfg.kinds, N = NSTEP, eta = {}, b1 = cfg.b1 != null ? cfg.b1 : 0.9, b2 = cfg.b2 != null ? cfg.b2 : 0.99, sigma = cfg.noise != null ? cfg.noise : 0.1, tilt = cfg.tilt != null ? cfg.tilt : 25;
    var RANGE = { sgd: [0.005, 0.13, 0.005], mom: [0.02, 1.4, 0.01], rms: [0.005, 0.2, 0.005], adam: [0.01, 0.6, 0.01] };
    var DEF = { sgd: 0.07, mom: 0.3, rms: 0.03, adam: 0.2 };
    kinds.forEach(function (k) { eta[k] = cfg.eta && cfg.eta[k] != null ? cfg.eta[k] : DEF[k]; });
    var k = N, coord = 0, runs = {}, V = valley(tilt), grid = null, anim = null;
    var bar = Kit.$(id + "-bar"), ctrl = Kit.$(id + "-controls"), focus = kinds[kinds.length - 1];
    var noise = noiseArr(11, 2 * N);

    function compute() {
      V = valley(tilt);
      kinds.forEach(function (kd) { runs[kd] = runOpt(kd, V, { eta: eta[kd], b1: b1, b2: b2, eps: 1e-8, sigma: sigma }, N, noise); });
      grid = gridFor(V);
    }
    function gridFor(V) {
      var xs = lin(-2.6, 2.6, 105), ys = lin(-1.6, 1.6, 65), z = ys.map(function (y) { return xs.map(function (x) { return Math.log10(Math.max(lossV(V, x, y), 1e-3)); }); });
      return { x: xs, y: ys, z: z };
    }
    // controls
    var sliders = {};
    var want = cfg.sliders || ["eta"];
    if (want.indexOf("eta") >= 0) kinds.forEach(function (kd) {
      var R = RANGE[kd];
      sliders["eta" + kd] = Kit.slider(ctrl, { label: "Learning rate η, " + KINDS[kd].name, min: R[0], max: R[1], step: R[2], value: eta[kd], digits: R[2] < 0.01 ? 3 : 2 }, function (v) { eta[kd] = v; changed(); });
    });
    if (want.indexOf("b1") >= 0) sliders.b1 = Kit.slider(ctrl, { label: "β1, memory of the average slope", min: 0, max: 0.99, step: 0.01, value: b1, digits: 2 }, function (v) { b1 = v; changed(); });
    if (want.indexOf("b2") >= 0) sliders.b2 = Kit.slider(ctrl, { label: "β2, memory of the squared slope", min: 0.5, max: 0.999, step: 0.001, value: b2, digits: 3 }, function (v) { b2 = v; changed(); });
    if (want.indexOf("noise") >= 0) sliders.noise = Kit.slider(ctrl, { label: "Noise in the slope (made-up)", min: 0, max: 1, step: 0.05, value: sigma, digits: 2 }, function (v) { sigma = v; changed(); });
    if (want.indexOf("tilt") >= 0) sliders.tilt = Kit.slider(ctrl, { label: "Tilt of the valley, degrees", min: 0, max: 50, step: 1, value: tilt, digits: 0 }, function (v) { tilt = v; changed(); });
    var scrub = Kit.slider(ctrl, { label: "Step shown", min: 0, max: N, step: 1, value: k, digits: 0 }, function (v) { stopAnim(); k = v; draw(); });
    function changed() { stopAnim(); compute(); k = N; scrub.set(k); draw(); }
    function stopAnim() { if (anim) { anim(); anim = null; } }
    Kit.button(bar, "Run", function () {
      stopAnim(); k = 0; scrub.set(0);
      anim = Kit.animate(N * 20, function (p) { k = Math.round(p * N); scrub.set(k); draw(); });
    });
    Kit.button(bar, "Step", function () { stopAnim(); k = Math.min(N, k + 1); scrub.set(k); draw(); });
    Kit.button(bar, "Reset", function () { stopAnim(); k = 0; scrub.set(0); draw(); });
    Kit.chips(bar, ["θ1", "θ2"], 0, function (i) { coord = i; drawEq(); });

    function drawEq() {
      var box = Kit.$(id + "-eqs"); if (!box) return;
      var r = runs[focus], t = k;
      if (t < 1) { box.innerHTML = '<p class="status">Press Step or Run. The boxes below show one update of ' + KINDS[focus].name + ' for weight θ' + (coord + 1) + '.</p>'; return; }
      var a = r[t], p = r[t - 1], c = coord, w = "θ" + (c + 1), o = [];
      var rowEnd = function (tag, stepBox) { return stepBox; };
      if (focus === "sgd") {
        o.push(ebox("c-navy", nf(p.th[c]), w + ", before") + eop("&minus;") + ebox("c-sand", nf(eta.sgd), "η") + eop("&times;") + ebox("c-coral", nf(a.g[c]), "slope g") + eop("=") + ebox("c-teal", nf(a.th[c]), w + ", after"));
      } else if (focus === "mom") {
        o.push(ebox("c-sand", nf(b1), "β1") + eop("&times;") + ebox("c-navy", nf(p.m[c]), "m before") + eop("+") + ebox("c-sand", nf(1 - b1), "1 &minus; β1") + eop("&times;") + ebox("c-coral", nf(a.g[c]), "slope g") + eop("=") + ebox("c-teal", nf(a.m[c]), "m, the average slope"));
        o.push(ebox("c-navy", nf(p.th[c]), w + ", before") + eop("&minus;") + ebox("c-sand", nf(eta.mom), "η") + eop("&times;") + ebox("c-teal", nf(a.m[c]), "m") + eop("=") + ebox("c-teal", nf(a.th[c]), w + ", after"));
      } else if (focus === "rms") {
        o.push(ebox("c-sand", nf(b2), "β2") + eop("&times;") + ebox("c-navy", nf(p.v[c]), "v before") + eop("+") + ebox("c-sand", nf(1 - b2), "1 &minus; β2") + eop("&times;") + ebox("c-coral", nf(a.g[c] * a.g[c]), "g<sup>2</sup>") + eop("=") + ebox("c-teal", nf(a.v[c]), "v, the average g<sup>2</sup>"));
        o.push(ebox("c-navy", nf(p.th[c]), w + ", before") + eop("&minus;") + ebox("c-sand", nf(eta.rms), "η") + eop("&times;") + ebox("c-coral", nf(a.g[c]), "slope g") + eop("&divide;") + ebox("c-teal", nf(Math.sqrt(a.v[c])), "&radic;v") + eop("=") + ebox("c-teal", nf(a.th[c]), w + ", after"));
      } else {
        o.push(ebox("c-sand", nf(b1), "β1") + eop("&times;") + ebox("c-navy", nf(p.m[c]), "m before") + eop("+") + ebox("c-sand", nf(1 - b1), "1 &minus; β1") + eop("&times;") + ebox("c-coral", nf(a.g[c]), "slope g") + eop("=") + ebox("c-teal", nf(a.m[c]), "m") + eop("&divide;") + ebox("c-sand", nf(1 - Math.pow(b1, t)), "1 &minus; β1<sup>" + t + "</sup>") + eop("=") + ebox("c-teal", nf(a.mh[c]), "m̂"));
        o.push(ebox("c-sand", nf(b2), "β2") + eop("&times;") + ebox("c-navy", nf(p.v[c]), "v before") + eop("+") + ebox("c-sand", nf(1 - b2), "1 &minus; β2") + eop("&times;") + ebox("c-coral", nf(a.g[c] * a.g[c]), "g<sup>2</sup>") + eop("=") + ebox("c-teal", nf(a.v[c]), "v") + eop("&divide;") + ebox("c-sand", nf(1 - Math.pow(b2, t)), "1 &minus; β2<sup>" + t + "</sup>") + eop("=") + ebox("c-teal", nf(a.vh[c]), "v̂"));
        o.push(ebox("c-navy", nf(p.th[c]), w + ", before") + eop("&minus;") + ebox("c-sand", nf(eta.adam), "η") + eop("&times;") + ebox("c-teal", nf(a.mh[c]), "m̂") + eop("&divide;") + ebox("c-teal", nf(Math.sqrt(a.vh[c])), "&radic;v̂") + eop("=") + ebox("c-teal", nf(a.th[c]), w + ", after"));
      }
      box.innerHTML = '<p class="status">Step ' + t + ' of ' + KINDS[focus].name + ', weight ' + w + '.</p>' + o.map(function (r) { return '<div class="eqn">' + r + '</div>'; }).join("");
    }
    function draw() {
      var P = Kit.palette();
      Kit.plot(id + "-map", function (P) {
        var tr = [{ type: "contour", x: grid.x, y: grid.y, z: grid.z, showscale: false, hoverinfo: "skip", contours: { start: -2.4, end: 1.0, size: 0.4, coloring: "fill" }, colorscale: [[0, P.paper], [1, mix(P.paper, P.sand, 0.3)]], line: { width: 0.8, color: P.rule }, zmin: -2.4, zmax: 1.0 }];
        tr.push({ x: [V.start[0]], y: [V.start[1]], mode: "markers+text", text: ["start"], textposition: "top center", marker: { symbol: "star", size: 12, color: P.ink }, textfont: { color: P.ink2, size: 12 }, hoverinfo: "skip" });
        tr.push({ x: [0], y: [0], mode: "markers", marker: { symbol: "x", size: 10, color: P.ink }, hoverinfo: "skip", name: "lowest point" });
        kinds.forEach(function (kd) {
          var r = runs[kd], xs = [], ys = [], i;
          for (i = 0; i <= k; i++) { xs.push(clamp(r[i].th[0], -2.6, 2.6)); ys.push(clamp(r[i].th[1], -1.6, 1.6)); }
          tr.push({ x: xs, y: ys, mode: "lines+markers", line: { color: col(P, kd), width: 2, dash: KINDS[kd].dash }, marker: { size: 3.5, color: col(P, kd), symbol: KINDS[kd].symbol }, hoverinfo: "skip" });
          var last = r[k];
          tr.push({ x: [xs[xs.length - 1]], y: [ys[ys.length - 1]], mode: "markers+text", text: [KINDS[kd].name], textposition: kd === "sgd" ? "bottom right" : kd === "mom" ? "top right" : kd === "rms" ? "bottom left" : "top left", marker: { size: 12, color: col(P, kd), symbol: KINDS[kd].symbol, line: { color: P.paper, width: 2 } }, textfont: { color: col(P, kd), size: 13 }, hoverinfo: "skip" });
        });
        return tr;
      }, function (P) {
        return { margin: { l: 44, r: 12, t: 8, b: 40 }, xaxis: { range: [-2.6, 2.6], title: { text: "weight θ1" }, zeroline: false }, yaxis: { range: [-1.6, 1.6], title: { text: "weight θ2" }, scaleanchor: "x", zeroline: false }, showlegend: false };
      });
      Kit.plot(id + "-loss", function (P) {
        return kinds.map(function (kd) {
          var r = runs[kd], xs = [], ys = [], i;
          for (i = 0; i <= k; i++) { xs.push(i); ys.push(Math.max(r[i].L, 1e-9)); }
          return { x: xs, y: ys, mode: "lines", name: KINDS[kd].name, line: { color: col(P, kd), width: 2.6 }, hovertemplate: KINDS[kd].name + " step %{x}: loss %{y:.2e}<extra></extra>" };
        }).concat([{ x: [0, N], y: [THR, THR], mode: "lines", line: { color: P.muted, width: 1, dash: "dot" }, hoverinfo: "skip", showlegend: false }]);
      }, function (P) {
        return { margin: { l: 56, r: 12, t: 8, b: 40 }, xaxis: { range: [0, N], title: { text: "step" } }, yaxis: { type: "log", range: [-4.5, 1], title: { text: "loss" }, exponentformat: "power" }, showlegend: true, legend: { orientation: "h", y: 1.18, x: 0 },
          annotations: [{ x: N, y: Math.log10(THR), xref: "x", yref: "y", text: "0.01", showarrow: false, xanchor: "right", yanchor: "bottom", font: { size: 11, color: P.muted } }] };
      });
      var rd = kinds.map(function (kd) {
        var r = runs[kd], f = firstBelow(r, THR), L = r[k].L;
        return "<span>" + KINDS[kd].name + ": loss <b>" + (L >= 1e6 ? "diverged" : nf(L)) + "</b>, first below 0.01: <b>" + (f < 0 ? "never in " + N + " steps" : "step " + f) + "</b></span>";
      });
      Kit.$(id + "-read").innerHTML = rd.join("");
      Kit.$(id + "-map").setAttribute("aria-label", "Contour map of a narrow tilted valley with the paths of " + kinds.map(function (kd) { return KINDS[kd].name; }).join(", ") + " after " + k + " steps.");
      drawEq();
    }
    compute(); draw(); Kit.onTheme(draw);
    return { runs: function () { return runs; } };
  }

  // ---------- 5. m and v for one weight ----------
  function emaFig() {
    var T = 100, g = [], r = Kit.rng(5), i;
    for (i = 1; i <= T; i++) g.push(0.4 + Kit.gauss(r));
    var b1 = 0.9, b2 = 0.999, corr = true, t = 8, ctrl = Kit.$("em-controls");
    Kit.slider(ctrl, { label: "β1, memory of the average slope", min: 0, max: 0.99, step: 0.01, value: b1, digits: 2 }, function (v) { b1 = v; draw(); });
    Kit.slider(ctrl, { label: "β2, memory of the squared slope", min: 0.5, max: 0.999, step: 0.001, value: b2, digits: 3 }, function (v) { b2 = v; draw(); });
    var sT = Kit.slider(ctrl, { label: "Step t shown in the boxes", min: 1, max: T, step: 1, value: t, digits: 0 }, function (v) { t = v; draw(); });
    Kit.chips(Kit.$("em-bar"), ["Bias correction on", "Bias correction off"], 0, function (j) { corr = j === 0; draw(); });
    Kit.button(Kit.$("em-bar"), "Jump to t = 1", function () { t = 1; sT.set(1); draw(); });
    function series() {
      var m = 0, v = 0, mm = [], vv = [], mh = [], vh = [], ratio = [], j;
      for (j = 1; j <= T; j++) {
        m = b1 * m + (1 - b1) * g[j - 1]; v = b2 * v + (1 - b2) * g[j - 1] * g[j - 1];
        mm.push(m); vv.push(v); mh.push(m / (1 - Math.pow(b1, j))); vh.push(v / (1 - Math.pow(b2, j)));
        ratio.push(corr ? mh[j - 1] / (Math.sqrt(vh[j - 1]) + 1e-8) : m / (Math.sqrt(v) + 1e-8));
      }
      return { m: mm, v: vv, mh: mh, vh: vh, ratio: ratio };
    }
    function draw() {
      var S = series(), xs = lin(1, T, T), mline = corr ? S.mh : S.m, sline = (corr ? S.vh : S.v).map(Math.sqrt);
      Kit.plot("em-top", function (P) {
        return [
          { x: xs, y: g, mode: "markers", marker: { size: 5, color: rgba(P.ink2, 0.5) }, name: "slope g", hovertemplate: "step %{x}: g = %{y:.2f}<extra></extra>" },
          { x: xs, y: mline, mode: "lines", line: { color: P.teal, width: 3 }, name: corr ? "m̂, average slope" : "m, average slope", hovertemplate: "step %{x}: %{y:.3f}<extra></extra>" },
          { x: xs, y: sline, mode: "lines", line: { color: P.navy, width: 3, dash: "dash" }, name: corr ? "√v̂, typical size" : "√v, typical size", hovertemplate: "step %{x}: %{y:.3f}<extra></extra>" },
          { x: [t], y: [mline[t - 1]], mode: "markers", showlegend: false, marker: { size: 12, color: P.teal, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" },
          { x: [t], y: [sline[t - 1]], mode: "markers", showlegend: false, marker: { size: 12, color: P.navy, symbol: "square", line: { color: P.paper, width: 2 } }, hoverinfo: "skip" }
        ];
      }, function (P) {
        return { margin: { l: 50, r: 12, t: 8, b: 36 }, xaxis: { range: [0, T + 1], title: { text: "step t" } }, yaxis: { range: [-2.2, 3.2], title: { text: "slope of one weight" } }, showlegend: true, legend: { orientation: "h", y: 1.2, x: 0 }, shapes: [{ type: "line", x0: t, x1: t, y0: 0, y1: 1, yref: "paper", line: { color: P.muted, width: 1, dash: "dot" } }] };
      });
      Kit.plot("em-ratio", function (P) {
        return [{ x: xs, y: S.ratio, mode: "lines", line: { color: P.coral, width: 3 }, hovertemplate: "step %{x}: ratio %{y:.2f}<extra></extra>" }, { x: [t], y: [S.ratio[t - 1]], mode: "markers", marker: { size: 12, color: P.coral, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" }];
      }, function (P) {
        return { margin: { l: 50, r: 12, t: 8, b: 40 }, xaxis: { range: [0, T + 1], title: { text: "step t" } }, yaxis: { range: [-0.5, 3.6], title: { text: "step size ÷ η" } }, shapes: [{ type: "line", x0: t, x1: t, y0: 0, y1: 1, yref: "paper", line: { color: P.muted, width: 1, dash: "dot" } }, { type: "line", x0: 0, x1: T + 1, y0: 1, y1: 1, line: { color: P.muted, width: 1, dash: "dot" } }] };
      });
      var j = t - 1, pm = j > 0 ? S.m[j - 1] : 0, pv = j > 0 ? S.v[j - 1] : 0;
      Kit.$("em-eqn").innerHTML =
        '<div class="eqn">' + ebox("c-sand", nf(b1), "β1") + eop("&times;") + ebox("c-navy", nf(pm), "m before") + eop("+") + ebox("c-sand", nf(1 - b1), "1 &minus; β1") + eop("&times;") + ebox("c-coral", nf(g[j]), "slope g") + eop("=") + ebox("c-teal", nf(S.m[j]), "m") + eop("&divide;") + ebox("c-sand", nf(1 - Math.pow(b1, t)), "1 &minus; β1<sup>t</sup>") + eop("=") + ebox("c-teal", nf(S.mh[j]), "m̂") + '</div>' +
        '<div class="eqn">' + ebox("c-sand", nf(b2), "β2") + eop("&times;") + ebox("c-navy", nf(pv), "v before") + eop("+") + ebox("c-sand", nf(1 - b2), "1 &minus; β2") + eop("&times;") + ebox("c-coral", nf(g[j] * g[j]), "g<sup>2</sup>") + eop("=") + ebox("c-teal", nf(S.v[j]), "v") + eop("&divide;") + ebox("c-sand", nf(1 - Math.pow(b2, t)), "1 &minus; β2<sup>t</sup>") + eop("=") + ebox("c-teal", nf(S.vh[j]), "v̂") + '</div>' +
        '<div class="eqn">' + ebox("c-teal", nf(corr ? S.mh[j] : S.m[j]), corr ? "m̂" : "m") + eop("&divide;") + ebox("c-teal", nf(Math.sqrt(corr ? S.vh[j] : S.v[j])), corr ? "&radic;v̂" : "&radic;v") + eop("=") + ebox("c-coral", nf(S.ratio[j]), "step size in units of η") + '</div>';
      var first = (1 - b1) / Math.sqrt(1 - b2);
      Kit.$("em-read").innerHTML = "<span>First step without correction: (1 − β1) ÷ √(1 − β2) = <b>" + nf(first) + "</b> times η</span><span>First step with correction: <b>1.000</b> times η (the sign of g)</span><span>Average of g (made-up): <b>0.4</b>, typical size √(0.4² + 1²) = <b>1.08</b></span>";
      Kit.$("em-top").setAttribute("aria-label", "Dots are 100 noisy slopes of one weight with average 0.4. The teal line follows the average slope and the navy dashed line follows the typical size of the slope" + (corr ? " with bias correction, so both start at the right scale." : " without bias correction, so both start near zero."));
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 6. L2 penalty inside Adam against decoupled decay ----------
  function decayFig() {
    var om = 0.5, sn = 3, eta = 0.02, T = 400, S = 30, ctrl = Kit.$("dc-controls");
    Kit.slider(ctrl, { label: "Weight decay ω", min: 0, max: 1.5, step: 0.05, value: om, digits: 2 }, function (v) { om = v; draw(); });
    Kit.slider(ctrl, { label: "Noise of the noisy weight's slope (made-up)", min: 1, max: 6, step: 0.5, value: sn, digits: 1 }, function (v) { sn = v; draw(); });
    function sim(kind, s) {
      var acc = new Array(T + 1).fill(0), seed, t;
      for (seed = 0; seed < S; seed++) {
        var r = Kit.rng(100 + seed + (s > 1 ? 0 : 500)), th = 1, m = 0, v = 0; acc[0] += 1;
        for (t = 1; t <= T; t++) {
          var g = s * Kit.gauss(r); if (kind === "l2") g += om * th;
          m = 0.9 * m + 0.1 * g; v = 0.99 * v + 0.01 * g * g;
          th -= eta * (m / (1 - Math.pow(0.9, t))) / (Math.sqrt(v / (1 - Math.pow(0.99, t))) + 1e-8);
          if (kind === "w") th -= eta * om * th;
          acc[t] += Math.abs(th);
        }
      }
      return acc.map(function (a) { return a / S; });
    }
    function half(a) { for (var i = 0; i < a.length; i++) if (a[i] < 0.5) return i; return -1; }
    function draw() {
      var xs = lin(0, T, T + 1), A = sim("l2", sn), B = sim("l2", 0.05), C = sim("w", sn), D = sim("w", 0.05);
      function pl(id, noisy, quiet, title) {
        Kit.plot(id, function (P) {
          return [
            { x: xs, y: noisy, mode: "lines", name: "noisy weight", line: { color: P.navy, width: 3 }, hovertemplate: "step %{x}: |θ| = %{y:.2f}<extra>noisy weight</extra>" },
            { x: xs, y: quiet, mode: "lines", name: "quiet weight", line: { color: P.coral, width: 3, dash: "dash" }, hovertemplate: "step %{x}: |θ| = %{y:.2f}<extra>quiet weight</extra>" }
          ];
        }, function (P) {
          return { margin: { l: 50, r: 12, t: 26, b: 40 }, xaxis: { title: { text: "step" } }, yaxis: { range: [0, 1.1], title: { text: "size of the weight |θ|" } }, showlegend: true, legend: { orientation: "h", y: -0.32, x: 0 }, annotations: [{ x: 0, y: 1.0, xref: "paper", yref: "paper", xanchor: "left", yanchor: "bottom", text: title, showarrow: false, font: { size: 14, color: P.ink } }] };
        });
      }
      pl("dc-l2", A, B, "L2 penalty added to the slope");
      pl("dc-w", C, D, "AdamW: decoupled decay");
      var f = function (a) { var h = half(a); return h < 0 ? "more than " + T : String(h); };
      Kit.$("dc-read").innerHTML = "<span>Steps until a weight halves, L2 inside Adam: noisy <b>" + f(A) + "</b>, quiet <b>" + f(B) + "</b></span><span>AdamW: noisy <b>" + f(C) + "</b>, quiet <b>" + f(D) + "</b></span><span>Plain decay rate 1 − η·ω per step: <b>" + (1 - eta * om).toFixed(4) + "</b></span>";
      Kit.$("dc-l2").setAttribute("aria-label", "Two weights with no useful slope shrink under weight decay. With an L2 penalty inside Adam the noisy weight shrinks slowly and the quiet one quickly; with AdamW both shrink at the same rate.");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 7. schedules and warm-up ----------
  function scheduleFig() {
    var T = 1000, warm = 100, fin = 0.1, type = 2, b2 = 0.99, ctrl = Kit.$("sc-controls"), W = 300;
    Kit.slider(ctrl, { label: "Warm-up steps", min: 0, max: 300, step: 10, value: warm, digits: 0 }, function (v) { warm = v; draw(); });
    Kit.slider(ctrl, { label: "Final rate as a share of the peak", min: 0, max: 0.5, step: 0.01, value: fin, digits: 2 }, function (v) { fin = v; draw(); });
    Kit.slider(ctrl, { label: "β2", min: 0.9, max: 0.999, step: 0.001, value: b2, digits: 3 }, function (v) { b2 = v; ratioCache = null; draw(); });
    Kit.chips(Kit.$("sc-bar"), ["Constant", "Linear decay", "Cosine decay"], 2, function (i) { type = i; draw(); });
    var ratioCache = null;
    function ratio() {
      if (ratioCache) return ratioCache;
      var r = Kit.rng(21), acc = new Array(T + 1).fill(0), w, t;
      for (w = 0; w < W; w++) {
        var m = 0, v = 0;
        for (t = 1; t <= T; t++) {
          var g = 0.1 + Kit.gauss(r); m = 0.9 * m + 0.1 * g; v = b2 * v + (1 - b2) * g * g;
          acc[t] += Math.abs((m / (1 - Math.pow(0.9, t))) / (Math.sqrt(v / (1 - Math.pow(b2, t))) + 1e-8));
        }
      }
      ratioCache = acc.map(function (a) { return a / W; }); return ratioCache;
    }
    function lr(t, wu) {
      if (wu > 0 && t <= wu) return t / wu;
      var s = wu > 0 ? (t - wu) / Math.max(1, T - wu) : t / T;
      if (type === 0) return 1;
      if (type === 1) return fin + (1 - fin) * (1 - s);
      return fin + (1 - fin) * 0.5 * (1 + Math.cos(Math.PI * s));
    }
    function draw() {
      var R = ratio(), xs = lin(1, T, T), L = xs.map(function (t) { return lr(t, warm); }), L0 = xs.map(function (t) { return lr(t, 0); });
      var U = xs.map(function (t, i) { return L[i] * R[t]; }), U0 = xs.map(function (t, i) { return L0[i] * R[t]; });
      Kit.plot("sc-lr", function (P) {
        return [{ x: xs, y: L0, mode: "lines", name: "no warm-up", line: { color: P.muted, width: 2, dash: "dot" }, hovertemplate: "step %{x}: %{y:.3f}<extra>no warm-up</extra>" },
          { x: xs, y: L, mode: "lines", name: "your schedule", line: { color: P.teal, width: 3.5 }, hovertemplate: "step %{x}: η = %{y:.3f} × peak<extra></extra>" }];
      }, function (P) {
        return { margin: { l: 50, r: 12, t: 8, b: 40 }, xaxis: { title: { text: "step" } }, yaxis: { range: [0, 1.08], title: { text: "learning rate ÷ peak" } }, showlegend: true, legend: { orientation: "h", y: 1.2, x: 0 } };
      });
      Kit.plot("sc-upd", function (P) {
        return [{ x: xs, y: U0, mode: "lines", name: "no warm-up", line: { color: P.muted, width: 2, dash: "dot" }, hovertemplate: "step %{x}: %{y:.3f}<extra>no warm-up</extra>" },
          { x: xs, y: U, mode: "lines", name: "your schedule", line: { color: P.coral, width: 3 }, hovertemplate: "step %{x}: %{y:.3f} × peak<extra></extra>" }];
      }, function (P) {
        return { margin: { l: 50, r: 12, t: 8, b: 40 }, xaxis: { title: { text: "step" } }, yaxis: { range: [0, 1.08], title: { text: "average Adam step ÷ peak η" } }, showlegend: true, legend: { orientation: "h", y: 1.2, x: 0 } };
      });
      Kit.$("sc-read").innerHTML = "<span>Step 1 update: <b>" + U[0].toFixed(3) + "</b> × peak (no warm-up: <b>" + U0[0].toFixed(3) + "</b>)</span><span>Update at step 100: <b>" + U[99].toFixed(3) + "</b> × peak</span><span>Update at step " + T + ": <b>" + U[T - 1].toFixed(3) + "</b> × peak</span>";
      var s100 = lr(100, warm), s500 = lr(500, warm);
      Kit.$("sc-eqn").innerHTML = ebox("c-navy", "peak 1.0e&minus;3", "an example peak rate") + eop("&times;") + ebox("c-teal", nf(s100), "schedule at step 100") + eop("=") + ebox("c-teal", (s100 * 1e-3).toExponential(2).replace("e-", "e&minus;"), "η at step 100") + eop("|") + ebox("c-navy", "peak 1.0e&minus;3", "peak rate") + eop("&times;") + ebox("c-teal", nf(s500), "schedule at step 500") + eop("=") + ebox("c-teal", (s500 * 1e-3).toExponential(2).replace("e-", "e&minus;"), "η at step 500");
      Kit.$("sc-lr").setAttribute("aria-label", "Learning rate against step: a ramp over " + warm + " warm-up steps and then a " + ["constant", "linear", "cosine"][type] + " schedule.");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 8. lose the state ----------
  function stateFig() {
    var ck = 150, eta = 0.1, b2 = 0.99, sigma = 0.05, mode = 1, M = 40, V = valley(25), runs = null, anim = null, kk = M, ctrl = Kit.$("st-controls"), bar = Kit.$("st-bar");
    var NAMES = ["Keep m, v and t (warm start)", "Reset m, v and t to zero (cold start)", "Keep v and t, reset m"];
    var sCk = Kit.slider(ctrl, { label: "Checkpoint saved after this many steps", min: 10, max: 300, step: 10, value: ck, digits: 0 }, function (v) { ck = v; recompute(); });
    Kit.slider(ctrl, { label: "Learning rate η", min: 0.02, max: 0.3, step: 0.01, value: eta, digits: 2 }, function (v) { eta = v; recompute(); });
    Kit.slider(ctrl, { label: "β2", min: 0.9, max: 0.999, step: 0.001, value: b2, digits: 3 }, function (v) { b2 = v; recompute(); });
    Kit.slider(ctrl, { label: "Noise in the slope (made-up)", min: 0, max: 0.3, step: 0.01, value: sigma, digits: 2 }, function (v) { sigma = v; recompute(); });
    Kit.button(bar, "Resume training", function () { stop(); kk = 0; anim = Kit.animate(M * 90, function (p) { kk = Math.round(p * M); draw(); }); });
    Kit.button(bar, "Show all 40 steps", function () { stop(); kk = M; draw(); });
    function stop() { if (anim) { anim(); anim = null; } }
    function adamRun(start, st0, n, noise, t0) {
      var th = start.slice(), m = st0.m.slice(), v = st0.v.slice(), t = t0, out = [{ th: th.slice(), L: lossV(V, th[0], th[1]), s: 0 }], i, k;
      for (k = 0; k < n; k++) {
        var g = gradV(V, th[0], th[1]); g[0] += sigma * noise[2 * k]; g[1] += sigma * noise[2 * k + 1]; t++;
        var sz = 0;
        for (i = 0; i < 2; i++) {
          m[i] = 0.9 * m[i] + 0.1 * g[i]; v[i] = b2 * v[i] + (1 - b2) * g[i] * g[i];
          var step = eta * (m[i] / (1 - Math.pow(0.9, t))) / (Math.sqrt(v[i] / (1 - Math.pow(b2, t))) + 1e-8);
          th[i] -= step; sz += Math.abs(step) / eta / 2;
        }
        out.push({ th: th.slice(), L: lossV(V, th[0], th[1]), s: sz });
      }
      return { out: out, m: m, v: v, t: t };
    }
    function recompute() {
      stop();
      var pre = adamRun(V.start, { m: [0, 0], v: [0, 0] }, ck, noiseArr(11, 2 * ck), 0), nz = noiseArr(99, 2 * M), end = pre.out[pre.out.length - 1];
      runs = {
        pre: pre,
        warm: adamRun(end.th, { m: pre.m, v: pre.v }, M, nz, pre.t),
        cold: adamRun(end.th, { m: [0, 0], v: [0, 0] }, M, nz, 0),
        vonly: adamRun(end.th, { m: [0, 0], v: pre.v }, M, nz, pre.t)
      };
      kk = M; draw();
    }
    function draw() {
      var R = runs, tail = R.pre.out.slice(Math.max(0, R.pre.out.length - 25)), ends = R.pre.out[R.pre.out.length - 1].th;
      var series = [["warm", "Warm start", "teal", "circle"], ["cold", "Cold start", "coral", "square"], ["vonly", "Keep v and t only", "sand", "diamond"]];
      Kit.plot("st-map", function (P) {
        var tr = [{ x: tail.map(function (r) { return r.th[0]; }), y: tail.map(function (r) { return r.th[1]; }), mode: "lines+markers", line: { color: P.muted, width: 1.5 }, marker: { size: 4, color: P.muted }, hoverinfo: "skip", name: "before the checkpoint" }];
        series.forEach(function (s) {
          var r = R[s[0]].out.slice(0, kk + 1);
          tr.push({ x: r.map(function (q) { return q.th[0]; }), y: r.map(function (q) { return q.th[1]; }), mode: "lines+markers", line: { color: P[s[2]], width: 2 }, marker: { size: 5, color: P[s[2]], symbol: s[3] }, name: s[1], hoverinfo: "skip" });
        });
        tr.push({ x: [ends[0]], y: [ends[1]], mode: "markers", showlegend: false, marker: { symbol: "star", size: 13, color: P.ink }, hoverinfo: "skip" });
        tr.push({ x: [0], y: [0], mode: "markers", showlegend: false, marker: { symbol: "x", size: 9, color: P.ink }, hoverinfo: "skip" });
        return tr;
      }, function (P) {
        var rad = Math.max(0.12, 2.2 * eta);
        return { margin: { l: 48, r: 12, t: 8, b: 44 }, xaxis: { range: [ends[0] - rad * 1.9, ends[0] + rad * 1.9], title: { text: "weight θ1" }, zeroline: false }, yaxis: { range: [ends[1] - rad, ends[1] + rad], scaleanchor: "x", title: { text: "weight θ2" }, zeroline: false }, showlegend: true, legend: { orientation: "h", y: -0.28, x: 0 } };
      });
      Kit.plot("st-step", function (P) {
        return series.map(function (s) {
          var r = R[s[0]].out.slice(1, kk + 1);
          return { x: r.map(function (_, i) { return i + 1; }), y: r.map(function (q) { return Math.max(q.s, 1e-4); }), mode: "lines+markers", name: s[1], line: { color: P[s[2]], width: 2.6 }, marker: { size: 4, symbol: s[3] }, hovertemplate: s[1] + ", step %{x}: %{y:.3f} η<extra></extra>" };
        });
      }, function (P) {
        return { margin: { l: 56, r: 12, t: 8, b: 40 }, xaxis: { range: [0, M + 1], title: { text: "steps after resuming" } }, yaxis: { type: "log", range: [-3, 0.3], title: { text: "average step ÷ η" }, exponentformat: "power" }, showlegend: false };
      });
      Kit.plot("st-loss", function (P) {
        return series.map(function (s) {
          var r = R[s[0]].out.slice(0, kk + 1);
          return { x: r.map(function (_, i) { return i; }), y: r.map(function (q) { return Math.max(q.L, 1e-6); }), mode: "lines+markers", name: s[1], line: { color: P[s[2]], width: 2.6 }, marker: { size: 4, symbol: s[3] }, hovertemplate: s[1] + ", step %{x}: loss %{y:.2e}<extra></extra>" };
        });
      }, function (P) {
        return { margin: { l: 56, r: 12, t: 8, b: 40 }, xaxis: { range: [0, M + 1], title: { text: "steps after resuming" } }, yaxis: { type: "log", title: { text: "loss" }, exponentformat: "power" }, showlegend: false };
      });
      var w1 = R.warm.out[1], c1 = R.cold.out[1], v1 = R.vonly.out[1], pk = function (r) { return Math.max.apply(null, r.out.map(function (q) { return q.L; })); };
      Kit.$("st-read").innerHTML = "<span>First step ÷ η: warm <b>" + nf(w1.s) + "</b>, cold <b>" + nf(c1.s) + "</b>, v and t only <b>" + nf(v1.s) + "</b></span><span>Loss at the checkpoint: <b>" + nf(R.pre.out[R.pre.out.length - 1].L) + "</b></span><span>Highest loss in the 40 steps: warm <b>" + nf(pk(R.warm)) + "</b>, cold <b>" + nf(pk(R.cold)) + "</b></span>";
      Kit.$("st-eqn").innerHTML = ebox("c-coral", nf(c1.s), "cold start, first step ÷ η") + eop("=") + ebox("c-sand", "m̂ ÷ √v̂", "with m = 0.1 g and v = 0.01 g²") + eop("=") + ebox("c-coral", "g ÷ |g| = ±1", "the same size for all slopes") + eop("|") + ebox("c-teal", nf(w1.s), "warm start, first step ÷ η");
      Kit.$("st-map").setAttribute("aria-label", "Zoomed contour-free view near the lowest point of the valley. After the checkpoint, a warm start stays close and a cold start jumps away in its first steps.");
    }
    recompute(); Kit.onTheme(draw);
  }

  // ---------- 9. v against the squared slope and the curvature ----------
  function fisherFig() {
    var F = null, which = 0, idx = 59, ctrl = Kit.$("fi-controls");
    var sl = Kit.slider(ctrl, { label: "Step shown", min: 1, max: 60, step: 1, value: 60, digits: 0, show: function (v) { return String(v * 10); } }, function (v) { idx = v - 1; draw(); });
    Kit.chips(Kit.$("fi-bar"), ["length weight", "girth weight", "bias"], 0, function (i) { which = i; draw(); });
    Kit.data("data/fisher.json").then(function (J) { F = J; draw(); }).catch(function (e) { console.error(e); });
    function draw() {
      if (!F) return;
      var xs = F.t, v = F.vhat.map(function (r) { return r[which]; }), f = F.fisher.map(function (r) { return r[which]; }), h = F.hess.map(function (r) { return r[which]; });
      Kit.plot("fi-plot", function (P) {
        return [
          { x: xs, y: v, mode: "lines", name: "Adam's v (running average of g²)", line: { color: P.navy, width: 3 }, hovertemplate: "step %{x}: v̂ = %{y:.3f}<extra></extra>" },
          { x: xs, y: f, mode: "lines", name: "mean of g² over the data", line: { color: P.coral, width: 3, dash: "dash" }, hovertemplate: "step %{x}: %{y:.3f}<extra>mean g²</extra>" },
          { x: xs, y: h, mode: "lines", name: "curvature of the loss", line: { color: P.teal, width: 3, dash: "dot" }, hovertemplate: "step %{x}: %{y:.3f}<extra>curvature</extra>" },
          { x: [xs[idx]], y: [v[idx]], mode: "markers", showlegend: false, marker: { size: 11, color: P.navy, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" },
          { x: [xs[idx]], y: [f[idx]], mode: "markers", showlegend: false, marker: { size: 11, color: P.coral, symbol: "square", line: { color: P.paper, width: 2 } }, hoverinfo: "skip" },
          { x: [xs[idx]], y: [h[idx]], mode: "markers", showlegend: false, marker: { size: 11, color: P.teal, symbol: "diamond", line: { color: P.paper, width: 2 } }, hoverinfo: "skip" }
        ];
      }, function (P) {
        return { margin: { l: 56, r: 12, t: 8, b: 40 }, xaxis: { title: { text: "training step" } }, yaxis: { type: "log", title: { text: "size, for the " + ["length weight", "girth weight", "bias"][which] }, tickvals: [0.02, 0.05, 0.1, 0.2, 0.5], ticktext: ["0.02", "0.05", "0.1", "0.2", "0.5"], range: [Math.log10(0.02), Math.log10(0.5)] }, showlegend: true, legend: { orientation: "h", y: 1.25, x: 0 } };
      });
      var j = idx;
      Kit.$("fi-eqn").innerHTML = ebox("c-navy", nf(F.vhat[j][which]), "v̂ (Adam)") + eop("&asymp;") + ebox("c-coral", nf(F.fisher[j][which]), "mean g² over the 268 records") + eop("&asymp;") + ebox("c-teal", nf(F.hess[j][which]), "curvature of the loss") + eop("|") + ebox("", nf(F.loss[j]), "loss at step " + F.t[j]);
      Kit.$("fi-read").innerHTML = "<span>Weights at step " + F.t[j] + ": length <b>" + nf(F.theta[j][0]) + "</b>, girth <b>" + nf(F.theta[j][1]) + "</b>, bias <b>" + nf(F.theta[j][2]) + "</b></span><span>Ratio v̂ ÷ mean g²: <b>" + nf(F.vhat[j][which] / F.fisher[j][which]) + "</b></span>";
      Kit.$("fi-plot").setAttribute("aria-label", "Three curves on a log axis for one weight of a logistic regression trained with Adam: Adam's v, the mean squared slope over the data and the curvature of the loss. They have similar sizes and fall together as training proceeds.");
    }
    Kit.onTheme(draw);
  }

  // ---------- start ----------
  function init() {
    raceFig({ id: "r1", kinds: ["sgd"], sliders: ["eta", "noise"], noise: 0.1 });
    raceFig({ id: "r2", kinds: ["sgd", "mom"], eta: { sgd: 0.07, mom: 0.3 }, sliders: ["eta", "b1"], noise: 0.1 });
    raceFig({ id: "r3", kinds: ["sgd", "rms"], eta: { sgd: 0.07, rms: 0.03 }, sliders: ["eta", "b2", "tilt"], noise: 0.1, tilt: 0 });
    raceFig({ id: "r4", kinds: ["sgd", "mom", "rms", "adam"], sliders: ["eta", "b1", "b2", "noise", "tilt"], noise: 0.1 });
    emaFig(); decayFig(); scheduleFig(); stateFig(); fisherFig();
  }
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", init);
  if (typeof module !== "undefined") module.exports = { valley: valley, runOpt: runOpt, lossV: lossV, firstBelow: firstBelow, noiseArr: noiseArr };
})();
