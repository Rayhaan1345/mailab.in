// B6 Making deep networks trainable: five figures (all weights and values are random and made up).
//   layerFig   activation and slope histograms through 20 layers, menus for initialisation and activation (Plotly)
//   scaleFig   signal size after 20 layers against the weight spread, for three widths (Plotly, data/scales.json)
//   normFig    the norm-axes grid: which cells batch, layer, RMS and group norm average together (D3)
//   resFig     slope size reaching early layers with and without residual connections (Plotly)
//   dropFig    dropout on a small diagram, with the spread of outputs over many masks (D3 and Plotly)
(function () {
  "use strict";

  // ---------- helpers ----------
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
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function nf(v) { if (!isFinite(v)) return "inf"; var a = Math.abs(v); return Kit.fmt(v, a >= 100 ? 1 : a >= 10 ? 2 : a >= 0.095 ? 3 : a === 0 ? 3 : 4); }
  function sci(v) { if (v === 0) return "0"; if (!isFinite(v)) return "inf"; var e = Math.floor(Math.log10(Math.abs(v))), m = v / Math.pow(10, e); return (v < 0 ? "−" : "") + Math.abs(m).toFixed(1) + "e" + (e < 0 ? "−" : "+") + Math.abs(e); }

  // ---------- 1. activations and slopes through 20 layers ----------
  var NW = 100, DEPTH = 20, BATCH = 128, NB = 41, GB = 36;
  var RND = null;
  function rand() {
    if (RND) return RND;
    var r = Kit.rng(7), W = new Float64Array(DEPTH * NW * NW), X = new Float64Array(BATCH * NW), G = new Float64Array(BATCH * NW), i;
    for (i = 0; i < W.length; i++) W[i] = Kit.gauss(r);
    for (i = 0; i < X.length; i++) X[i] = Kit.gauss(r);
    for (i = 0; i < G.length; i++) G[i] = Kit.gauss(r);
    RND = { W: W, X: X, G: G }; return RND;
  }
  function simulate(act, sigma) {
    var R = rand(), W = R.W, h = Float64Array.from(R.X), zs = [], hs = [h], l, b, i, j;
    for (l = 0; l < DEPTH; l++) {
      var z = new Float64Array(BATCH * NW), a = new Float64Array(BATCH * NW), off = l * NW * NW;
      for (b = 0; b < BATCH; b++) {
        for (i = 0; i < NW; i++) { var hv = h[b * NW + i]; if (hv === 0) continue; for (j = 0; j < NW; j++) z[b * NW + j] += hv * W[off + i * NW + j]; }
      }
      for (i = 0; i < z.length; i++) { z[i] *= sigma; a[i] = act === "tanh" ? Math.tanh(z[i]) : (z[i] > 0 ? z[i] : 0); }
      zs.push(z); hs.push(a); h = a;
    }
    var out = { act: [], grad: [], ahist: [], ghist: [], sat: [], rmsA: [], rmsG: [] };
    var lo = act === "tanh" ? -1 : 0, hi = act === "tanh" ? 1 : 4;
    for (l = 0; l < DEPTH; l++) {
      var a2 = hs[l + 1], hist = new Array(NB).fill(0), s2 = 0, sat = 0;
      for (i = 0; i < a2.length; i++) {
        var v = a2[i]; s2 += v * v;
        var k = Math.floor((v - lo) / (hi - lo) * NB); hist[clamp(k, 0, NB - 1)]++;
        if (act === "tanh" ? Math.abs(v) > 0.97 : v === 0) sat++;
      }
      out.ahist.push(hist.map(function (c) { return c / a2.length; })); out.rmsA.push(Math.sqrt(s2 / a2.length)); out.sat.push(sat / a2.length);
    }
    var gh = Float64Array.from(R.G), ghists = new Array(DEPTH), rmsG = new Array(DEPTH);
    for (l = DEPTH - 1; l >= 0; l--) {
      var dz = new Float64Array(BATCH * NW), z2 = zs[l], a3 = hs[l + 1], s3 = 0, gh2 = new Float64Array(BATCH * NW), hist2 = new Array(GB).fill(0);
      for (i = 0; i < dz.length; i++) {
        dz[i] = gh[i] * (act === "tanh" ? 1 - a3[i] * a3[i] : (z2[i] > 0 ? 1 : 0)); s3 += dz[i] * dz[i];
        var lg = dz[i] === 0 ? -12 : Math.log10(Math.abs(dz[i])), kk = Math.floor((lg + 12) / 18 * GB); hist2[clamp(kk, 0, GB - 1)]++;
      }
      ghists[l] = hist2.map(function (c) { return c / dz.length; }); rmsG[l] = Math.sqrt(s3 / dz.length);
      off = l * NW * NW;
      for (b = 0; b < BATCH; b++) for (i = 0; i < NW; i++) { var s = 0; for (j = 0; j < NW; j++) s += dz[b * NW + j] * W[off + i * NW + j]; gh2[b * NW + i] = s * sigma; }
      gh = gh2;
    }
    out.ghist = ghists; out.rmsG = rmsG; out.lo = lo; out.hi = hi;
    return out;
  }
  var INIT = { zeros: 0, small: 0.01, large: 1, xavier: Math.sqrt(2 / (NW + NW)), he: Math.sqrt(2 / NW) };
  function layerFig() {
    var act = "tanh", sigma = INIT.he, which = "he", ctrl = Kit.$("ly-controls"), bar = Kit.$("ly-bar");
    var sel = Kit.select(ctrl, "Initialisation (weight spread)", [
      { value: "zeros", label: "zeros: spread 0" }, { value: "small", label: "too small: spread 0.01" }, { value: "large", label: "too large: spread 1" },
      { value: "xavier", label: "Xavier: spread 0.1" }, { value: "he", label: "He: spread 0.141" }, { value: "custom", label: "custom (use the slider)" }], which, function (v) {
      which = v; if (v !== "custom") { sigma = INIT[v]; sl.set(sigma); } draw();
    });
    var sl = Kit.slider(ctrl, { label: "Weight spread σ (width 100)", min: 0, max: 1.5, step: 0.005, value: sigma, digits: 3 }, function (v) { sigma = v; which = "custom"; sel.value = "custom"; draw(); });
    Kit.chips(bar, ["tanh", "ReLU"], 0, function (i) { act = i ? "relu" : "tanh"; draw(); });
    function draw() {
      var S = simulate(act, sigma), layers = lin(1, DEPTH, DEPTH);
      var ax = act === "tanh" ? lin(-1 + 1 / NB, 1 - 1 / NB, NB) : lin(2 / NB, 4 - 2 / NB, NB), gx = lin(-12 + 0.25, 6 - 0.25, GB);
      Kit.plot("ly-act", function (P) {
        return [{ type: "heatmap", x: ax, y: layers, z: S.ahist.map(function (r) { return r.map(Math.sqrt); }), colorscale: [[0, P.paper], [1, P.teal]], showscale: false, zmin: 0, zmax: 1, xgap: 0, ygap: 0, hovertemplate: "layer %{y}, activation near %{x:.2f}: %{customdata:.1%} of units<extra></extra>", customdata: S.ahist }];
      }, function (P) {
        return { margin: { l: 40, r: 10, t: 24, b: 44 }, xaxis: { title: { text: "activation value" }, range: act === "tanh" ? [-1, 1] : [0, 4] }, yaxis: { title: { text: "layer" }, autorange: "reversed", dtick: 3 },
          annotations: [{ x: 0, y: 1.02, xref: "paper", yref: "paper", xanchor: "left", yanchor: "bottom", text: "Activations", showarrow: false, font: { size: 14, color: P.ink } }] };
      });
      Kit.plot("ly-grad", function (P) {
        return [{ type: "heatmap", x: gx, y: layers, z: S.ghist.map(function (r) { return r.map(Math.sqrt); }), colorscale: [[0, P.paper], [1, P.coral]], showscale: false, zmin: 0, zmax: 1, xgap: 0, ygap: 0, hovertemplate: "layer %{y}, slope near 10^%{x:.1f}: %{customdata:.1%} of values<extra></extra>", customdata: S.ghist }];
      }, function (P) {
        return { margin: { l: 40, r: 10, t: 24, b: 44 }, xaxis: { title: { text: "slope size (power of ten)" }, range: [-12, 6], dtick: 6 }, yaxis: { title: { text: "layer" }, autorange: "reversed", dtick: 3 },
          annotations: [{ x: 0, y: 1.02, xref: "paper", yref: "paper", xanchor: "left", yanchor: "bottom", text: "Slopes on the way back", showarrow: false, font: { size: 14, color: P.ink } }] };
      });
      Kit.plot("ly-rms", function (P) {
        return [
          { x: layers, y: S.rmsA.map(function (v) { return Math.max(v, 1e-30); }), mode: "lines+markers", name: "activations", line: { color: P.navy, width: 3 }, marker: { size: 6, symbol: "circle" }, hovertemplate: "layer %{x}: %{y:.2e}<extra>activations</extra>" },
          { x: layers, y: S.rmsG.map(function (v) { return Math.max(v, 1e-30); }), mode: "lines+markers", name: "slopes", line: { color: P.coral, width: 3, dash: "dash" }, marker: { size: 6, symbol: "square" }, hovertemplate: "layer %{x}: %{y:.2e}<extra>slopes</extra>" },
          { x: [1, DEPTH], y: [1, 1], mode: "lines", line: { color: P.muted, width: 1, dash: "dot" }, showlegend: false, hoverinfo: "skip" }];
      }, function (P) {
        return { margin: { l: 56, r: 12, t: 8, b: 40 }, xaxis: { title: { text: "layer" }, dtick: 2, range: [0.5, DEPTH + 0.5] }, yaxis: { type: "log", automargin: true, title: { text: "typical size (RMS)" }, range: [-30, 30], dtick: 10, exponentformat: "power" }, showlegend: true, legend: { orientation: "h", y: 1.18, x: 0 } };
      });
      var a1 = S.rmsA[0], a20 = S.rmsA[DEPTH - 1], g1 = S.rmsG[0], g20 = S.rmsG[DEPTH - 1];
      var fa = a1 > 0 && a20 > 0 ? Math.pow(a20 / a1, 1 / (DEPTH - 1)) : 0, fg = g20 > 0 && g1 > 0 ? Math.pow(g1 / g20, 1 / (DEPTH - 1)) : 0;
      Kit.$("ly-eqn").innerHTML = ebox("c-navy", sci(a1), "activation size, layer 1") + eop("&rarr;") + ebox(a20 < 1e-3 ? "c-coral" : a20 > 1e3 ? "c-sand" : "c-teal", sci(a20), "activation size, layer 20") + eop("|") + ebox("c-coral", sci(g20), "slope size, layer 20") + eop("&rarr;") +
        ebox(g1 < 1e-3 ? "c-coral" : g1 > 1e3 ? "c-sand" : "c-teal", sci(g1), "slope size, layer 1") + eop("|") + ebox("c-teal", nf(fa), "measured factor per layer, forward");
      var msg;
      if (sigma === 0) msg = "All weights are zero. The activations are zero from layer 1 on, and the slopes are zero (or, for tanh, only the last layer gets one). No weight can learn, and units that start with equal weights stay equal.";
      else if (a20 < 1e-3) msg = "Vanishing: the activations shrink to " + sci(a20) + " by layer 20, so the last layers see almost no signal. The slopes shrink the same way going back (" + sci(g1) + " at layer 1).";
      else if (a20 > 1e3) msg = "Exploding: the activations grow to " + sci(a20) + " by layer 20, and the slopes at layer 1 are " + sci(g1) + " times those at layer 20.";
      else if (act === "tanh" && S.sat[DEPTH - 1] > 0.5) msg = "Saturating: " + Math.round(100 * S.sat[DEPTH - 1]) + "% of the units in layer 20 are in the flat tails of tanh, where the slope is near 0. The slope at layer 1 is " + sci(g1) + ".";
      else if (g1 < 1e-3 || g1 > 1e3) msg = "The activations look fine but the slope at layer 1 is " + sci(g1) + ", so the early layers learn at the wrong pace.";
      else msg = "Balanced: activations and slopes stay within a factor of about 10 of their starting size across 20 layers.";
      Kit.setText("ly-status", msg);
      Kit.$("ly-act").setAttribute("aria-label", "Histogram of activations through 20 layers for " + act + " units with weight spread " + nf(sigma) + ". Typical size at layer 20: " + sci(a20) + ".");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 2. the sweet spot for the spread ----------
  function scaleFig() {
    var D = null, n = "100", sigma = 0.141, act = "relu", ctrl = Kit.$("sx-controls");
    var sl = Kit.slider(ctrl, { label: "Weight spread σ (log scale)", min: -2, max: 0.3, step: 0.01, value: Math.log10(sigma), digits: 2, show: function (v) { return Math.pow(10, v).toFixed(3); } }, function (v) { sigma = Math.pow(10, v); draw(); });
    Kit.chips(Kit.$("sx-bar"), ["width 25", "width 100", "width 400"], 1, function (i) { n = ["25", "100", "400"][i]; draw(); });
    Kit.chips(Kit.$("sx-bar"), ["tanh", "ReLU"], 1, function (i) { act = i ? "relu" : "tanh"; draw(); });
    Kit.data("data/scales.json").then(function (J) { D = J; draw(); }).catch(function (e) { console.error(e); });
    function interp(xs, ys, x) {
      var i = 0; while (i < xs.length - 2 && xs[i + 1] < x) i++;
      var t = (Math.log10(x) - Math.log10(xs[i])) / (Math.log10(xs[i + 1]) - Math.log10(xs[i])); return ys[i] + (ys[i + 1] - ys[i]) * clamp(t, 0, 1);
    }
    function draw() {
      if (!D) return;
      var w = +n, R = D.widths[n], xav = 1 / Math.sqrt(w), he = Math.sqrt(2 / w), mk = interp(D.spreads, R[act].act, sigma), mg = interp(D.spreads, R[act].grad, sigma);
      function pl(id, key, title) {
        Kit.plot(id, function (P) {
          return [
            { x: D.spreads, y: R.tanh[key].map(function (v) { return Math.pow(10, clamp(v, -30, 30)); }), mode: "lines", name: "tanh", line: { color: P.navy, width: act === "tanh" ? 4 : 1.8 }, opacity: act === "tanh" ? 1 : 0.55, hovertemplate: "σ = %{x:.3f}: %{y:.1e}<extra>tanh</extra>" },
            { x: D.spreads, y: R.relu[key].map(function (v) { return Math.pow(10, clamp(v, -30, 30)); }), mode: "lines", name: "ReLU", line: { color: P.coral, width: act === "relu" ? 4 : 1.8, dash: "dash" }, opacity: act === "relu" ? 1 : 0.55, hovertemplate: "σ = %{x:.3f}: %{y:.1e}<extra>ReLU</extra>" },
            { x: [sigma], y: [Math.pow(10, clamp(key === "act" ? mk : mg, -30, 30))], mode: "markers", showlegend: false, marker: { size: 13, color: act === "tanh" ? P.navy : P.coral, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" }
          ];
        }, function (P) {
          return { margin: { l: 56, r: 12, t: 26, b: 44 }, xaxis: { type: "log", title: { text: "weight spread σ" }, range: [-2, 0.3] }, yaxis: { type: "log", automargin: true, title: { text: "typical size (RMS)" }, range: [-30, 30], dtick: 10, exponentformat: "power" }, showlegend: true, legend: { orientation: "h", y: -0.36, x: 0 },
            shapes: [{ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: -1, y1: 1, fillcolor: rgba(P.teal, 0.13), line: { width: 0 }, layer: "below" },
              { type: "line", xref: "x", x0: xav, x1: xav, yref: "paper", y0: 0, y1: 1, line: { color: P.muted, width: 1.5, dash: "dot" } }, { type: "line", xref: "x", x0: he, x1: he, yref: "paper", y0: 0, y1: 1, line: { color: P.ink2, width: 1.5, dash: "dash" } }],
            annotations: [{ x: 0, y: 1.02, xref: "paper", yref: "paper", xanchor: "left", yanchor: "bottom", text: title, showarrow: false, font: { size: 14, color: P.ink } },
              { x: Math.log10(xav), y: 1, xref: "x", yref: "paper", text: "Xavier", showarrow: false, xanchor: "right", yanchor: "top", font: { size: 11, color: P.muted } }, { x: Math.log10(he), y: 1, xref: "x", yref: "paper", text: "He", showarrow: false, xanchor: "left", yanchor: "top", font: { size: 11, color: P.ink2 } }] };
        });
      }
      pl("sx-act", "act", "Activations at layer 20 (input size 1)");
      pl("sx-grad", "grad", "Slopes at layer 1 (layer 20 size 1)");
      var fT = w * sigma * sigma, fR = w * sigma * sigma / 2;
      Kit.$("sx-eqn").innerHTML = ebox("c-navy", String(w), "inputs per unit, n") + eop("&times;") + ebox("c-navy", nf(sigma * sigma), "σ² (variance of a weight)") + eop("=") + ebox(Math.abs(Math.log10(fT)) < 0.2 ? "c-teal" : "c-coral", nf(fT), "variance factor per layer, tanh near 0") +
        eop("|") + ebox("c-navy", nf(fT), "n σ²") + eop("&divide; 2 =") + ebox(Math.abs(Math.log10(fR)) < 0.2 ? "c-teal" : "c-coral", nf(fR), "variance factor per layer, ReLU") + eop("&rarr; 20 layers &rarr;") + ebox("c-sand", sci(Math.pow(fR, 20)), "ReLU variance after 20 layers");
      Kit.$("sx-read").innerHTML = "<span>Width " + n + ": Xavier spread <b>" + nf(xav) + "</b>, He spread <b>" + nf(he) + "</b></span><span>At σ = " + nf(sigma) + ": activations at layer 20 are <b>" + sci(Math.pow(10, clamp(mk, -30, 30))) + "</b>, slopes at layer 1 are <b>" + sci(Math.pow(10, clamp(mg, -30, 30))) + "</b></span>";
      Kit.$("sx-act").setAttribute("aria-label", "On log axes, the size of the activations at layer 20 against the weight spread, for tanh and ReLU at width " + n + ". Sizes near 1 occur only close to the Xavier and He spreads; both are marked.");
    }
    Kit.onTheme(draw);
  }

  // ---------- 3. the norm-axes grid ----------
  var TAGS = [4127, 3305, 5519, 2509, 8722, 5577], NC = 8;
  var FEAT = { mean: [0.5, -1.2, 2.0, 0.0, 3.5, -0.4, 1.0, -2.5], sd: [0.5, 1.5, 2.0, 0.3, 1.0, 0.8, 2.5, 1.2], load: [0.8, -0.6, 0.7, 0.2, 0.9, -0.3, 0.5, -0.7] };
  function normData() {
    var r = Kit.rng(31), rows = [], i, c;
    function row() { var s = Kit.gauss(r), o = []; for (c = 0; c < NC; c++) { var v = FEAT.mean[c] + FEAT.sd[c] * (FEAT.load[c] * s + Math.sqrt(1 - FEAT.load[c] * FEAT.load[c]) * Kit.gauss(r)); o.push(Math.round(v * 10) / 10); } return o; }
    for (i = 0; i < TAGS.length; i++) rows.push(row());
    var more = [], run = { mean: [], vr: [] }; for (i = 0; i < 80; i++) more.push(row());
    for (c = 0; c < NC; c++) { var col = more.map(function (q) { return q[c]; }), m = col.reduce(function (a, b) { return a + b; }, 0) / col.length; run.mean.push(Math.round(m * 100) / 100); run.vr.push(Math.round(col.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / col.length * 100) / 100); }
    return { X: rows, run: run };
  }
  function normFig() {
    var D = normData(), mode = "bn", groups = 2, N = 6, view = 0, useRun = false, sel = [1, 2], gam = 1, bet = 0, EPS = 1e-5;
    var bar = Kit.$("nm-bar"), ctrl = Kit.$("nm-controls"), svgNode = Kit.$("nm-svg");
    var NAMES = { bn: "Batch norm", ln: "Layer norm", rms: "RMS norm", gn: "Group norm" };
    Kit.chips(bar, ["Batch norm", "Layer norm", "RMS norm", "Group norm"], 0, function (i) { mode = ["bn", "ln", "rms", "gn"][i]; draw(); });
    Kit.chips(bar, ["Raw values", "After normalising"], 0, function (i) { view = i; draw(); });
    Kit.chips(bar, ["2 groups", "4 groups"], 0, function (i) { groups = i ? 4 : 2; if (mode === "gn") draw(); });
    var slN = Kit.slider(ctrl, { label: "Sharks in the batch", min: 1, max: 6, step: 1, value: N, digits: 0 }, function (v) { N = v; sel[0] = Math.min(sel[0], N - 1); draw(); });
    Kit.slider(ctrl, { label: "Learned scale γ", min: 0.2, max: 3, step: 0.1, value: gam, digits: 1 }, function (v) { gam = v; draw(); });
    Kit.slider(ctrl, { label: "Learned shift β (not used by RMS norm)", min: -2, max: 2, step: 0.1, value: bet, digits: 1 }, function (v) { bet = v; draw(); });
    var runChip = Kit.chips(bar, ["Batch statistics", "Stored averages (test time)"], 0, function (i) { useRun = i === 1; draw(); });
    function cellsOf(r, c) {
      var o = [], i, k;
      if (mode === "bn") { for (i = 0; i < N; i++) o.push([i, c]); }
      else if (mode === "ln" || mode === "rms") { for (k = 0; k < NC; k++) o.push([r, k]); }
      else { var gs = NC / groups, g0 = Math.floor(c / gs) * gs; for (k = g0; k < g0 + gs; k++) o.push([r, k]); }
      return o;
    }
    function stats(cells, c) {
      var vals = cells.map(function (q) { return D.X[q[0]][q[1]]; }), m = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
      var vr = vals.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / vals.length, ms = vals.reduce(function (a, b) { return a + b * b; }, 0) / vals.length;
      if (mode === "bn" && useRun) { m = D.run.mean[c]; vr = D.run.vr[c]; }
      return { m: m, vr: vr, ms: ms };
    }
    function xhat(r, c) {
      var S = stats(cellsOf(r, c), c), x = D.X[r][c];
      return mode === "rms" ? x / Math.sqrt(S.ms + EPS) : (x - S.m) / Math.sqrt(S.vr + EPS);
    }
    function draw() {
      var P = Kit.palette(), W = Kit.width(svgNode), left = W < 480 ? 44 : 64, top = 30, bottom = mode === "bn" ? 74 : 12, cs = Math.min(58, Math.floor((W - left - 8) / NC)), H = top + N * cs + bottom;
      var svg = d3.select(svgNode); svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).selectAll("*").remove();
      runChip.forEach(function (b) { b.style.display = mode === "bn" ? "" : "none"; b.parentNode.style.display = mode === "bn" ? "" : "none"; });
      Array.from(bar.querySelectorAll(".group")).forEach(function (g, i) { if (i === 2) g.style.display = mode === "gn" ? "" : "none"; });
      var set = cellsOf(sel[0], sel[1]), inSet = {}; set.forEach(function (q) { inSet[q[0] + "," + q[1]] = true; });
      var vals = [], i, c;
      for (i = 0; i < N; i++) for (c = 0; c < NC; c++) vals.push(view ? xhat(i, c) : D.X[i][c]);
      var mx = Math.max.apply(null, vals.map(Math.abs)) || 1;
      for (c = 0; c < NC; c++) svg.append("text").attr("x", left + c * cs + cs / 2).attr("y", top - 10).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.muted).text("h" + (c + 1));
      for (i = 0; i < N; i++) {
        svg.append("text").attr("x", left - 6).attr("y", top + i * cs + cs / 2 + 4).attr("text-anchor", "end").attr("font-size", 12.5).attr("fill", P.muted).text((W < 480 ? "" : "Tag ") + TAGS[i]);
        for (c = 0; c < NC; c++) {
          var v = view ? xhat(i, c) : D.X[i][c], a = clamp(Math.abs(v) / mx, 0, 1), on = !!inSet[i + "," + c], isSel = i === sel[0] && c === sel[1];
          var g = svg.append("g").attr("tabindex", 0).attr("role", "button").attr("aria-label", "Row " + (i + 1) + ", feature " + (c + 1) + ", value " + nf(v)).style("cursor", "pointer").style("outline", "none").attr("opacity", on ? 1 : 0.4);
          g.append("rect").attr("x", left + c * cs + 1).attr("y", top + i * cs + 1).attr("width", cs - 2).attr("height", cs - 2).attr("rx", 5).attr("fill", v >= 0 ? rgba(P.coral, 0.1 + 0.6 * a) : rgba(P.navy, 0.1 + 0.6 * a)).attr("stroke", isSel ? P.ink : on ? P.teal : P.rule).attr("stroke-width", isSel ? 3 : on ? 2.2 : 1);
          g.append("text").attr("x", left + c * cs + cs / 2).attr("y", top + i * cs + cs / 2 + 4.5).attr("text-anchor", "middle").attr("font-size", cs < 40 ? 11.5 : 13.5).attr("font-weight", on ? 700 : 400).attr("fill", P.ink).text(Kit.fmt(v, view ? 2 : 1));
          (function (ii, cc) { g.on("click", function () { sel = [ii, cc]; draw(); }); g.on("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = [ii, cc]; draw(); } }); })(i, c);
        }
      }
      if (mode === "bn") {
        svg.append("text").attr("x", left - 6).attr("y", top + N * cs + 26).attr("text-anchor", "end").attr("font-size", 12).attr("fill", P.muted).text("stored mean");
        svg.append("text").attr("x", left - 6).attr("y", top + N * cs + 52).attr("text-anchor", "end").attr("font-size", 12).attr("fill", P.muted).text("stored var");
        for (c = 0; c < NC; c++) {
          svg.append("text").attr("x", left + c * cs + cs / 2).attr("y", top + N * cs + 26).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", useRun ? P.ink : P.muted).attr("font-weight", useRun && c === sel[1] ? 700 : 400).text(Kit.fmt(D.run.mean[c], 1));
          svg.append("text").attr("x", left + c * cs + cs / 2).attr("y", top + N * cs + 52).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", useRun ? P.ink : P.muted).attr("font-weight", useRun && c === sel[1] ? 700 : 400).text(Kit.fmt(D.run.vr[c], 1));
        }
      }
      // equation boxes for the selected cell
      var S = stats(set, sel[1]), x = D.X[sel[0]][sel[1]], den = mode === "rms" ? Math.sqrt(S.ms + EPS) : Math.sqrt(S.vr + EPS), xh = mode === "rms" ? x / den : (x - S.m) / den, y = gam * xh + (mode === "rms" ? 0 : bet);
      var who = mode === "bn" ? "feature h" + (sel[1] + 1) + " over " + N + " sharks" + (useRun ? " (stored)" : "") : mode === "ln" || mode === "rms" ? "row of Tag " + TAGS[sel[0]] + ", all 8 features" : "Tag " + TAGS[sel[0]] + ", group of " + NC / groups + " features";
      var rows = "<p class=\"status\">" + NAMES[mode] + ": the highlighted cells are averaged together (" + who + ").</p>";
      if (mode === "rms") {
        rows += '<div class="eqn">' + ebox("c-sand", nf(S.ms), "mean of x², over " + set.length + " cells") + eop("&rarr;") + ebox("c-sand", nf(den), "&radic;(mean x² + ε)") + '</div>';
        rows += '<div class="eqn">' + ebox("c-navy", nf(x), "x, Tag " + TAGS[sel[0]] + ", h" + (sel[1] + 1)) + eop("&divide;") + ebox("c-sand", nf(den), "root mean square") + eop("=") + ebox("c-teal", nf(xh), "x̂") + eop("&times;") + ebox("c-navy", nf(gam), "γ") + eop("=") + ebox("c-teal", nf(y), "output") + '</div>';
      } else {
        rows += '<div class="eqn">' + ebox("c-sand", nf(S.m), "mean μ of " + set.length + " cells") + ebox("c-sand", nf(S.vr), "variance") + eop("&rarr;") + ebox("c-sand", nf(den), "&radic;(variance + ε)") + '</div>';
        rows += '<div class="eqn">' + ebox("c-navy", nf(x), "x, Tag " + TAGS[sel[0]] + ", h" + (sel[1] + 1)) + eop("&minus;") + ebox("c-sand", nf(S.m), "μ") + eop("&divide;") + ebox("c-sand", nf(den), "&radic;(var + ε)") + eop("=") + ebox("c-teal", nf(xh), "x̂") + eop("&times;") + ebox("c-navy", nf(gam), "γ") + eop("+") + ebox("c-navy", nf(bet), "β") + eop("=") + ebox("c-teal", nf(y), "output") + '</div>';
      }
      Kit.$("nm-eqs").innerHTML = rows;
      var nh = set.map(function (q) { return xhat(q[0], q[1]); }), mh = nh.reduce(function (a, b) { return a + b; }, 0) / nh.length, vh = nh.reduce(function (a, b) { return a + (b - mh) * (b - mh); }, 0) / nh.length, rh = Math.sqrt(nh.reduce(function (a, b) { return a + b * b; }, 0) / nh.length);
      Kit.$("nm-read").innerHTML = "<span>Cells averaged together: <b>" + set.length + "</b></span><span>After normalising, those cells have mean <b>" + nf(mh) + "</b>, variance <b>" + nf(vh) + "</b>, root mean square <b>" + nf(rh) + "</b></span>";
      Kit.setText("nm-status", mode === "bn" && N === 1 && !useRun ? "With one shark in the batch, a feature has a single value, its variance is 0, and x − μ is 0 for all cells: batch norm outputs only β. Switch to layer norm or to the stored averages." : mode === "bn" && useRun ? "Test time: a single shark is normalised with averages stored during training (an exponential moving average over batches, as in 0.7)." : mode === "rms" ? "RMS norm keeps the mean: the normalised cells have root mean square 1, but their mean is not 0." : mode === "bn" ? "A column is normalised across the batch, so a shark's value depends on the other sharks that happen to share its batch." : "A row is normalised on its own, so the result does not depend on the other sharks.");
      svgNode.setAttribute("aria-label", NAMES[mode] + " on a grid of " + N + " sharks by 8 hidden values. The highlighted cells (" + who + ") share one mean and one variance.");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- 4. residual connections ----------
  function resFig() {
    var depth = 30, gain = 0.6, resid = false, n = 64, B = 16, ctrl = Kit.$("rs-controls");
    Kit.slider(ctrl, { label: "Number of blocks", min: 5, max: 50, step: 1, value: depth, digits: 0 }, function (v) { depth = v; draw(); });
    Kit.slider(ctrl, { label: "Weight scale of a block (1.0 is Xavier for tanh)", min: 0.2, max: 1.4, step: 0.05, value: gain, digits: 2 }, function (v) { gain = v; draw(); });
    Kit.chips(Kit.$("rs-bar"), ["Without residual connections", "With residual connections"], 0, function (i) { resid = i === 1; draw(); });
    var RN = null;
    function rnd() {
      if (RN) return RN; var r = Kit.rng(13), W = new Float64Array(50 * n * n), X = new Float64Array(B * n), G = new Float64Array(B * n), i;
      for (i = 0; i < W.length; i++) W[i] = Kit.gauss(r); for (i = 0; i < X.length; i++) X[i] = Kit.gauss(r); for (i = 0; i < G.length; i++) G[i] = Kit.gauss(r);
      RN = { W: W, X: X, G: G }; return RN;
    }
    function sim(res) {
      var R = rnd(), s = gain / Math.sqrt(n), h = Float64Array.from(R.X), zs = [], hs = [h], l, b, i, j;
      for (l = 0; l < depth; l++) {
        var z = new Float64Array(B * n), a = new Float64Array(B * n), off = l * n * n;
        for (b = 0; b < B; b++) for (i = 0; i < n; i++) { var hv = h[b * n + i]; for (j = 0; j < n; j++) z[b * n + j] += hv * R.W[off + i * n + j]; }
        for (i = 0; i < z.length; i++) { z[i] *= s; a[i] = Math.tanh(z[i]) + (res ? h[i] : 0); }
        zs.push(z); hs.push(a); h = a;
      }
      var gh = Float64Array.from(R.G), norms = new Array(depth + 1), nn = function (v) { var t = 0, q; for (b = 0; b < B; b++) { var u = 0; for (q = 0; q < n; q++) u += v[b * n + q] * v[b * n + q]; t += Math.sqrt(u); } return t / B; };
      var base = nn(gh); norms[depth] = 1;
      for (l = depth - 1; l >= 0; l--) {
        var th = new Float64Array(B * n), off2 = l * n * n, z2 = zs[l], out = new Float64Array(B * n);
        for (i = 0; i < th.length; i++) { var t = Math.tanh(z2[i]); th[i] = gh[i] * (1 - t * t); }
        for (b = 0; b < B; b++) for (i = 0; i < n; i++) { var sm = 0; for (j = 0; j < n; j++) sm += th[b * n + j] * R.W[off2 + i * n + j]; out[b * n + i] = sm * s + (res ? gh[b * n + i] : 0); }
        gh = out; norms[l] = nn(gh) / base;
      }
      var last = hs[depth], ra = 0; for (i = 0; i < last.length; i++) ra += last[i] * last[i];
      return { norms: norms, rms: Math.sqrt(ra / last.length) };
    }
    function draw() {
      var A = sim(false), Bm = sim(true), cur = resid ? Bm : A, oth = resid ? A : Bm, xs = lin(0, depth, depth + 1).map(function (v) { return v; });
      var ys = cur.norms.map(function (v) { return Math.max(v, 1e-30); }), yo = oth.norms.map(function (v) { return Math.max(v, 1e-30); });
      Kit.plot("rs-plot", function (P) {
        return [
          { type: "bar", x: xs, y: ys, marker: { color: resid ? P.teal : P.coral }, name: resid ? "with residual connections" : "without", hovertemplate: "block %{x}: %{y:.2e}<extra></extra>" },
          { x: xs, y: yo, mode: "lines", line: { color: P.muted, width: 2, dash: "dot" }, name: resid ? "without (for comparison)" : "with (for comparison)", hovertemplate: "block %{x}: %{y:.2e}<extra>other setting</extra>" }
        ];
      }, function (P) {
        return { margin: { l: 60, r: 12, t: 34, b: 44 }, xaxis: { title: { text: "block (0 is nearest the input)" }, dtick: depth > 25 ? 5 : 1 }, yaxis: { type: "log", automargin: true, title: { text: "slope size, output = 1" }, exponentformat: "power" }, bargap: 0.25, showlegend: true, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0 } };
      });
      var f = function (r) { return Math.pow(Math.max(r.norms[0], 1e-300), 1 / depth); };
      Kit.$("rs-eqn").innerHTML = ebox("c-coral", nf(f(A)), "slope factor per block, without") + eop("&rarr;") + ebox("c-coral", sci(A.norms[0]), "slope at block 0, without") + eop("|") + ebox(f(Bm) > 1.02 ? "c-sand" : "c-teal", nf(f(Bm)), "slope factor per block, with") + eop("&rarr;") + ebox(Bm.norms[0] > 1e3 ? "c-sand" : "c-teal", sci(Bm.norms[0]), "slope at block 0, with");
      Kit.$("rs-read").innerHTML = "<span>Slope reaching block 0, relative to the output: without <b>" + sci(A.norms[0]) + "</b>, with <b>" + sci(Bm.norms[0]) + "</b></span><span>Activation size at the last block: without <b>" + sci(A.rms) + "</b>, with <b>" + sci(Bm.rms) + "</b></span>";
      Kit.setText("rs-status", !resid ? (A.norms[0] < 1e-2 ? "Without shortcuts the slope reaching the first block is " + sci(A.norms[0]) + " of the slope at the output: the early blocks barely learn." : "At this weight scale the plain stack still passes the slope on. Lower the scale or add blocks.") : (Bm.norms[0] > 3 ? "With shortcuts the slope is not lost, but it grows (" + sci(Bm.norms[0]) + " at block 0) and the activations grow too. This is why deep residual stacks add normalisation layers or small branch weights." : "With shortcuts the slope reaching block 0 stays close to the output's slope (" + sci(Bm.norms[0]) + ")."));
      Kit.$("rs-plot").setAttribute("aria-label", "Bar chart on a log axis of the slope size at " + depth + " blocks, " + (resid ? "with" : "without") + " residual connections. At block 0 it is " + sci(cur.norms[0]) + " times the slope at the output.");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 5. dropout ----------
  function dropFig() {
    var p = 0.5, scale = true, test = false, ctr = 0, ctrl = Kit.$("dp-controls"), bar = Kit.$("dp-bar"), svgNode = Kit.$("dp-svg"), SZ = [4, 6, 6, 2], HID = 12;
    var r0 = Kit.rng(5), Wt = [], k, i, j;
    for (k = 0; k < 3; k++) { var M = []; for (i = 0; i < SZ[k + 1]; i++) { var row = []; for (j = 0; j < SZ[k]; j++) row.push(Kit.gauss(r0) * Math.sqrt(2 / SZ[k])); M.push(row); } Wt.push(M); }
    var xin = [0.9, -0.4, 1.2, 0.5];
    function forward(mask, sc) {
      var h = xin.slice(), l, zfirst = 0;
      for (l = 0; l < 3; l++) {
        var z = Wt[l].map(function (row) { return row.reduce(function (a, w, q) { return a + w * h[q]; }, 0); });
        if (l === 1) zfirst = z[0];
        if (l < 2) { h = z.map(function (v, q) { var a = Math.max(0, v); return mask ? (mask[l][q] ? a * sc : 0) : a; }); } else h = z;
      }
      var e0 = Math.exp(h[0] - Math.max(h[0], h[1])), e1 = Math.exp(h[1] - Math.max(h[0], h[1])); return { p: e0 / (e0 + e1), z: zfirst };
    }
    function maskAt(c) { var r = Kit.rng(1000 + c), m = [[], []], l, q; for (l = 0; l < 2; l++) for (q = 0; q < 6; q++) m[l].push(r() >= p); return m; }
    Kit.slider(ctrl, { label: "Drop probability p", min: 0, max: 0.9, step: 0.05, value: p, digits: 2 }, function (v) { p = v; draw(); });
    Kit.button(bar, "Resample the mask", function () { ctr++; test = false; chipsM[0].setAttribute("aria-pressed", "true"); chipsM[1].setAttribute("aria-pressed", "false"); draw(); });
    var chipsM = Kit.chips(bar, ["Training: units dropped", "Test: all units kept"], 0, function (i) { test = i === 1; draw(); });
    Kit.chips(bar, ["Scale kept units by 1/(1 − p)", "No scaling"], 0, function (i) { scale = i === 0; draw(); });
    function draw() {
      var P = Kit.palette(), W = Kit.width(svgNode), H = W < 480 ? 230 : 250, mask = test ? null : maskAt(ctr), sc = scale ? 1 / (1 - p) : 1, svg = d3.select(svgNode);
      svg.attr("viewBox", "0 0 " + W + " " + H).attr("height", H).selectAll("*").remove();
      var xs = lin(40, W - 40, 4), pos = [], l, q, kept = 0;
      for (l = 0; l < 4; l++) { pos.push(lin(26, H - 26, SZ[l] + 2).slice(1, -1).map(function (y) { return [xs[l], y]; })); }
      pos[0] = pos[0].map(function (c, q) { return [c[0], 40 + q * (H - 80) / 3]; });
      for (l = 0; l < 3; l++) pos[l + 1].forEach(function (b, qi) { pos[l].forEach(function (a, pi) {
        var dead = mask && ((l >= 1 && !mask[l - 1][pi]) || (l + 1 <= 2 && !mask[l][qi]));
        svg.append("line").attr("x1", a[0]).attr("y1", a[1]).attr("x2", b[0]).attr("y2", b[1]).attr("stroke", dead ? P.rule : P.muted).attr("stroke-width", dead ? 0.6 : 1).attr("opacity", dead ? 0.5 : 0.7);
      }); });
      for (l = 0; l < 4; l++) pos[l].forEach(function (c, qi) {
        var off = mask && (l === 1 || l === 2) && !mask[l - 1][qi]; if ((l === 1 || l === 2) && !off) kept++;
        svg.append("circle").attr("cx", c[0]).attr("cy", c[1]).attr("r", 13).attr("fill", off ? P.bg : l === 3 ? P.coral : l === 0 ? P.navy : P.teal).attr("stroke", off ? P.muted : P.paper).attr("stroke-width", off ? 1.6 : 2).attr("stroke-dasharray", off ? "3 3" : "");
        if (off) { svg.append("line").attr("x1", c[0] - 7).attr("y1", c[1] - 7).attr("x2", c[0] + 7).attr("y2", c[1] + 7).attr("stroke", P.muted).attr("stroke-width", 1.6); svg.append("line").attr("x1", c[0] - 7).attr("y1", c[1] + 7).attr("x2", c[0] + 7).attr("y2", c[1] - 7).attr("stroke", P.muted).attr("stroke-width", 1.6); }
      });
      ["input", "hidden 1", "hidden 2", "output"].forEach(function (t, li) { svg.append("text").attr("x", xs[li]).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.muted).text(t); });
      var ft = forward(null, 1), fm = forward(mask, sc), pt = ft.z, pm = fm.z;
      var outs = [], sum = 0, c2; for (c2 = 0; c2 < 2000; c2++) { var v = forward(maskAt(10000 + c2), sc).z; outs.push(v); sum += v; }
      var mean = sum / outs.length, lo = Math.min(Math.min.apply(null, outs), pt) - 0.2, hi = Math.max(Math.max.apply(null, outs), pt) + 0.2, nb = 24, bw = (hi - lo) / nb;
      var hist = new Array(nb).fill(0); outs.forEach(function (v) { hist[clamp(Math.floor((v - lo) / bw), 0, nb - 1)]++; });
      Kit.plot("dp-plot", function (P) {
        return [{ type: "bar", x: hist.map(function (_, i) { return lo + (i + 0.5) * bw; }), y: hist.map(function (h) { return h / outs.length; }), marker: { color: scale ? P.teal : P.sand }, hovertemplate: "total input near %{x:.2f}: %{y:.1%} of masks<extra></extra>" },
          { x: [pt, pt], y: [0, 1], mode: "lines", line: { color: P.navy, width: 3 }, hoverinfo: "skip" }, { x: [mean, mean], y: [0, 1], mode: "lines", line: { color: P.coral, width: 3, dash: "dash" }, hoverinfo: "skip" }];
      }, function (P) {
        return { margin: { l: 52, r: 12, t: 8, b: 44 }, xaxis: { title: { text: "total input to one unit, 2,000 masks" }, range: [lo, hi] }, yaxis: { title: { text: "share of masks" }, range: [0, Math.max(0.15, Math.max.apply(null, hist) / outs.length * 1.25)], tickformat: ".0%" }, bargap: 0.05,
          annotations: [{ x: pt, y: 1, yref: "paper", text: "test: " + pt.toFixed(2), showarrow: false, xanchor: "right", yanchor: "top", font: { size: 12, color: P.navy } }, { x: mean, y: 1, yref: "paper", text: "mean of masks: " + mean.toFixed(2), showarrow: false, xanchor: "left", yanchor: "top", font: { size: 12, color: P.coral } }] };
      });
      Kit.$("dp-read").innerHTML = "<span>Hidden units kept: <b>" + (test ? "all 12" : kept + " of 12") + "</b></span><span>Chance of adult with this mask: <b>" + nf(test ? ft.p : fm.p) + "</b> (test time: <b>" + nf(ft.p) + "</b>)</span><span>Total input to the unit, test time: <b>" + nf(pt) + "</b></span><span>Mean over 2,000 masks: <b>" + nf(mean) + "</b></span><span>Possible masks: 2<sup>12</sup> = <b>4,096</b></span>";
      var keep = 1 - p;
      Kit.$("dp-eqn").innerHTML = ebox("c-navy", "0.80", "a unit's value, before dropout") + eop("&times;") + ebox("c-sand", scale ? nf(1 / keep) : "1", scale ? "1 &divide; (1 &minus; p), if kept" : "no scaling") + eop("=") + ebox("c-teal", nf(0.8 * sc), "value if kept (chance " + nf(keep) + ")") + eop("|") + ebox("c-sand", "0", "value if dropped (chance " + nf(p) + ")") + eop("&rarr;") + ebox(scale ? "c-teal" : "c-coral", nf(keep * 0.8 * sc), "average value (true value 0.80)");
      svgNode.setAttribute("aria-label", "A small network with 4 inputs, two hidden layers of 6 units and 2 outputs. " + (test ? "All units are on." : kept + " of 12 hidden units are on; the others are crossed out."));
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- start ----------
  function init() { layerFig(); scaleFig(); normFig(); resFig(); dropFig(); }
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", init);
})();
