/* B3 Loss and gradient descent.
   Figures: (1) the push on the output for three tasks and two losses; (2) loss landscapes of a one-neuron model
   (cross-entropy and squared error) and of a one-hidden-unit network, with gradient descent paths;
   (3) batches and epochs on the 268 spadenose records; (4) mini-batch paths and the noise of the gradient.
   The model in (2) to (4): p = sigmoid(w x + b), x = scaled length, label 1 = mature (the made-up spadenose records of A3). */
(function () {
  "use strict";

  // ---------- helpers ----------
  function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function sig(z) { return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z)); }
  function hasWebGL() {
    try { var c = document.createElement("canvas"); return !!(c.getContext("webgl") || c.getContext("experimental-webgl")); } catch (e) { return false; }
  }
  function mix(a, b, t) {
    function p(h) { h = h.replace("#", ""); if (h.length === 3) h = h.replace(/(.)/g, "$1$1"); return [0, 2, 4].map(function (i) { return parseInt(h.substr(i, 2), 16); }); }
    var A = p(a), B = p(b); return "rgb(" + A.map(function (v, i) { return Math.round(v + (B[i] - v) * t); }).join(",") + ")";
  }
  var SPECIES = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"];

  // ---------- the data: spadenose records, scaled length x, label y = mature ----------
  var D = null, M = null;
  function prepare(C) {
    var rec = C.records.filter(function (r) { return r.species === "spadenose"; }), m = rec.length, i;
    var len = rec.map(function (r) { return r.length_cm; }), mean = len.reduce(function (a, b) { return a + b; }, 0) / m;
    var sd = Math.sqrt(len.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0) / m);
    var wt = rec.map(function (r) { return r.weight_kg; }), wm = wt.reduce(function (a, b) { return a + b; }, 0) / m, ws = Math.sqrt(wt.reduce(function (a, b) { return a + (b - wm) * (b - wm); }, 0) / m);
    D = { m: m, mean: mean, sd: sd, x: len.map(function (v) { return (v - mean) / sd; }), y: rec.map(function (r) { return r.mature; }), t: wt.map(function (v) { return (v - wm) / ws; }) };
    var x = D.x, y = D.y, t = D.t;
    function ce(w, b) { var s = 0, z; for (i = 0; i < m; i++) { z = w * x[i] + b; s += z >= 0 ? (1 - y[i]) * z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)) - y[i] * z; } return s / m; }
    function gce(w, b) { var gw = 0, gb = 0, p; for (i = 0; i < m; i++) { p = sig(w * x[i] + b) - y[i]; gw += p * x[i]; gb += p; } return [gw / m, gb / m]; }
    function mse(w, b) { var s = 0, p; for (i = 0; i < m; i++) { p = sig(w * x[i] + b) - y[i]; s += p * p; } return s / m; }
    function gmse(w, b) { var gw = 0, gb = 0, p, q; for (i = 0; i < m; i++) { p = sig(w * x[i] + b); q = 2 * (p - y[i]) * p * (1 - p); gw += q * x[i]; gb += q; } return [gw / m, gb / m]; }
    function acc(w, b) { var ok = 0; for (i = 0; i < m; i++) if ((w * x[i] + b > 0 ? 1 : 0) === y[i]) ok++; return ok / m; }
    function l2(w1, w2) { var s = 0, e; for (i = 0; i < m; i++) { e = w2 * Math.tanh(w1 * x[i]) - t[i]; s += e * e; } return s / m; }
    function g2(w1, w2) { var a = 0, c = 0, th, e; for (i = 0; i < m; i++) { th = Math.tanh(w1 * x[i]); e = 2 * (w2 * th - t[i]); a += e * w2 * (1 - th * th) * x[i]; c += e * th; } return [a / m, c / m]; }
    M = {
      ce: { L: ce, g: gce, dom: [[-6, 8], [-5, 6]], levels: { start: 0.25, end: 6, size: 0.5 }, zclip: 6, ax: ["weight w on scaled length", "bias b"], floor: [4.61, 2.48] },
      mse: { L: mse, g: gmse, dom: [[-6, 8], [-5, 6]], levels: { start: 0.075, end: 0.7, size: 0.05 }, zclip: 0.7, ax: ["weight w on scaled length", "bias b"], floor: [4.64, 2.55] },
      net: { L: l2, g: g2, dom: [[-2.5, 2.5], [-6, 6]], levels: { start: 0.1, end: 2.1, size: 0.1 }, zclip: 2.1, ax: ["weight w₁ (input)", "weight w₂ (output)"], floor: null },
      acc: acc
    };
  }
  function gdPath(model, start, eta, n) {
    var w = start[0], b = start[1], out = [[w, b]], k, g;
    for (k = 0; k < n; k++) {
      g = model.g(w, b); w -= eta * g[0]; b -= eta * g[1];
      if (!isFinite(w) || !isFinite(b) || Math.abs(w) > 60 || Math.abs(b) > 60) break;
      out.push([w, b]);
    }
    return out;
  }
  function gridOf(model, nx, ny) {
    var ws = lin(model.dom[0][0], model.dom[0][1], nx), bs = lin(model.dom[1][0], model.dom[1][1], ny), z = [], i, j, row;
    for (j = 0; j < ny; j++) { row = []; for (i = 0; i < nx; i++) row.push(Math.min(model.zclip, model.L(ws[i], bs[j]))); z.push(row); }
    return { ws: ws, bs: bs, z: z };
  }

  document.addEventListener("DOMContentLoaded", function () {
    costFig();
    Kit.data("../data/sharks/catch.json").then(function (C) { prepare(C); landFig(); valleyFig(); stripFig(); noiseFig(); })
      .catch(function () { ["ld", "vy", "st", "nz"].forEach(function (id) { var n = Kit.$(id); if (n) n.textContent = "The catch table did not load, so this figure cannot run."; }); });
  });

  // ======================================================================
  // 1. The push on the output: three tasks, two losses
  // ======================================================================
  function costFig() {
    var st = { task: 1, yhat: 0.3, z: 5, label: 0, spec: 3, zt: -0.4 }, BASE = [2.4, 1.2, 0.5, -0.4, -1.8], W_TRUE = 0.228;
    Kit.chips(Kit.$("co-task"), ["A number (weight, kg)", "Two classes (adult or not)", "Five classes (species)"], st.task, function (i) { st.task = i; show(); draw(); });
    var sHat = Kit.slider(Kit.$("co-controls"), { label: "network's prediction of the weight, kg", min: 0, max: 0.6, step: 0.005, value: st.yhat, digits: 3 }, function (v) { st.yhat = v; draw(); });
    var sZ = Kit.slider(Kit.$("co-controls"), { label: "network's score z for \"adult\"", min: -8, max: 8, step: 0.1, value: st.z, digits: 1 }, function (v) { st.z = v; draw(); });
    var sT = Kit.slider(Kit.$("co-controls"), { label: "score of the true species", min: -4, max: 6, step: 0.1, value: st.zt, digits: 1 }, function (v) { st.zt = v; draw(); });
    var lab = Kit.html("div", {}, Kit.$("co-bar")), spc = Kit.html("div", {}, Kit.$("co-bar"));
    Kit.chips(lab, ["true label: juvenile (0)", "true label: adult (1)"], st.label, function (i) { st.label = i; draw(); });
    Kit.chips(spc, SPECIES, st.spec, function (i) { st.spec = i; draw(); });
    function show() {
      sHat.input.closest("label").style.display = st.task === 0 ? "" : "none"; sZ.input.closest("label").style.display = st.task === 1 ? "" : "none"; sT.input.closest("label").style.display = st.task === 2 ? "" : "none";
      lab.style.display = st.task === 1 ? "" : "none"; spc.style.display = st.task === 2 ? "" : "none";
    }
    function softmax(z) { var mx = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - mx); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (v) { return v / s; }); }
    function draw() {
      var eq = Kit.$("co-eqn"), status, t = st.task;
      if (t === 0) {
        var y = W_TRUE, e = st.yhat - y, xs = lin(0, 0.6, 121);
        Kit.setText("co-lt", "Loss against the prediction (true weight 0.228 kg, Tag 3305)"); Kit.setText("co-rt", "Push on the prediction, dℓ/dŷ");
        Kit.plot("co-left", function (P) { return [{ type: "scatter", mode: "lines", x: xs, y: xs.map(function (v) { return (v - y) * (v - y); }), line: { color: P.coral, width: 3 }, hoverinfo: "skip" }, { type: "scatter", mode: "markers", x: [st.yhat], y: [e * e], marker: { size: 12, color: P.coral, line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" }]; },
          function () { return { margin: { l: 52, r: 12, t: 8, b: 48 }, dragmode: false, xaxis: { title: { text: "predicted weight ŷ, kg", standoff: 6 } }, yaxis: { title: { text: "squared error", standoff: 4 }, range: [0, 0.14] } }; });
        Kit.plot("co-right", function (P) { return [{ type: "scatter", mode: "lines", x: xs, y: xs.map(function (v) { return 2 * (v - y); }), line: { color: P.coral, width: 3 }, hoverinfo: "skip" }, { type: "scatter", mode: "markers", x: [st.yhat], y: [2 * e], marker: { size: 12, color: P.coral, line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" }]; },
          function () { return { margin: { l: 52, r: 12, t: 8, b: 48 }, dragmode: false, xaxis: { title: { text: "predicted weight ŷ, kg", standoff: 6 } }, yaxis: { title: { text: "push 2 (ŷ − y)", standoff: 4 }, range: [-0.5, 0.8] } }; });
        eq.innerHTML = "<span class='t c-teal'><b>" + Kit.fmt(st.yhat, 3) + "</b><i>prediction ŷ</i></span><span class='op'>&minus;</span><span class='t'><b>0.228</b><i>true weight y</i></span><span class='op'>=</span><span class='t c-sand'><b>" + Kit.fmt(e, 3) + "</b><i>error</i></span><span class='op'>&rarr; squared</span><span class='t c-coral'><b>" + Kit.fmt(e * e, 4) + "</b><i>loss ℓ</i></span><span class='op'>push</span><span class='t'><b>" + Kit.fmt(2 * e, 3) + "</b><i>2 × error</i></span>";
        status = "The loss is a parabola and the push is a straight line: the further the prediction is from 0.228 kg, the harder the push. There is no flat tail.";
      } else if (t === 1) {
        var yy = st.label, zs = lin(-8, 8, 161), p = sig(st.z);
        var ceL = function (z) { return z >= 0 ? (1 - yy) * z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z)) - yy * z; };
        var mseL = function (z) { var q = sig(z) - yy; return q * q; };
        var ceP = function (z) { return Math.abs(sig(z) - yy); }, mseP = function (z) { var q = sig(z); return Math.abs(2 * (q - yy) * q * (1 - q)); };
        Kit.setText("co-lt", "Loss against the score z (" + (yy ? "an adult" : "a juvenile, such as Tag 4127") + ")"); Kit.setText("co-rt", "Size of the push on the score, |dℓ/dz|");
        Kit.plot("co-left", function (P) { return [
          { type: "scatter", mode: "lines", x: zs, y: zs.map(ceL), line: { color: P.navy, width: 3 }, hoverinfo: "skip" }, { type: "scatter", mode: "lines", x: zs, y: zs.map(mseL), line: { color: P.coral, width: 3 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: [st.z, st.z], y: [ceL(st.z), mseL(st.z)], marker: { size: 11, color: [P.navy, P.coral], line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" }]; },
          function (P) { return { margin: { l: 52, r: 12, t: 8, b: 48 }, dragmode: false, xaxis: { title: { text: "score z", standoff: 6 } }, yaxis: { range: [0, 6.5], title: { text: "loss ℓ", standoff: 8 } }, annotations: [
            { x: yy ? -2.6 : 2.6, y: 4.9, xref: "x", yref: "y", text: "<b>cross-entropy</b>", font: { color: P.navy, size: 13 }, showarrow: false, xanchor: yy ? "left" : "right" },
            { x: yy ? 7.5 : -7.5, y: 0.7, xref: "x", yref: "y", text: "<b>squared error</b>", font: { color: P.coral, size: 13 }, showarrow: false, xanchor: yy ? "right" : "left" }] }; });
        Kit.plot("co-right", function (P) { return [
          { type: "scatter", mode: "lines", x: zs, y: zs.map(ceP), line: { color: P.navy, width: 3 }, hoverinfo: "skip" }, { type: "scatter", mode: "lines", x: zs, y: zs.map(mseP), line: { color: P.coral, width: 3 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: [st.z, st.z], y: [ceP(st.z), mseP(st.z)], marker: { size: 11, color: [P.navy, P.coral], line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" }]; },
          function () { return { margin: { l: 56, r: 12, t: 8, b: 48 }, dragmode: false, xaxis: { title: { text: "score z", standoff: 6 } }, yaxis: { range: [0, 1.05], title: { text: "push size", standoff: 8 } } }; });
        eq.innerHTML = "<span class='t c-teal'><b>" + Kit.fmt(p, 3) + "</b><i>probability p = σ(" + Kit.fmt(st.z, 1) + ")</i></span><span class='op'>label</span><span class='t'><b>" + yy + "</b><i>y</i></span><span class='op'>&rarr; cross-entropy</span><span class='t c-navy'><b>" + Kit.fmt(ceL(st.z), 3) + "</b><i>loss, push " + Kit.fmt(ceP(st.z), 3) + "</i></span><span class='op'>squared error</span><span class='t c-coral'><b>" + Kit.fmt(mseL(st.z), 3) + "</b><i>loss, push " + Kit.fmt(mseP(st.z), 3) + "</i></span>";
        var wrong = (st.z > 0 ? 1 : 0) !== yy;
        status = wrong && Math.abs(st.z) > 3 ? "The network is confidently wrong. Cross-entropy keeps a push of " + Kit.fmt(ceP(st.z), 2) + ". Squared error pushes with only " + Kit.fmt(mseP(st.z), 3) + ", because the sigmoid is flat out here." : "Move the score towards the wrong side, to +5 for a juvenile or −5 for an adult, and compare the two pushes.";
      } else {
        var z = BASE.slice(); z[st.spec] = st.zt; var pk = softmax(z), push = pk.map(function (v, k) { return v - (k === st.spec ? 1 : 0); });
        Kit.setText("co-lt", "Probabilities from softmax, p"); Kit.setText("co-rt", "Push on the five scores, p − y (true species: " + SPECIES[st.spec] + ")");
        Kit.plot("co-left", function (P) { return [{ type: "bar", x: SPECIES, y: pk, marker: { color: SPECIES.map(function (_, k) { return k === st.spec ? P.teal : P.sand; }) }, text: pk.map(function (v) { return Kit.fmt(v, 2); }), textposition: "outside", hoverinfo: "skip", cliponaxis: false }]; },
          function () { return { margin: { l: 44, r: 12, t: 18, b: 56 }, dragmode: false, xaxis: { tickangle: -25 }, yaxis: { range: [0, 1.12], title: { text: "probability", standoff: 4 } } }; });
        Kit.plot("co-right", function (P) { return [{ type: "bar", x: SPECIES, y: push, marker: { color: push.map(function (v) { return v < 0 ? P.teal : P.coral; }) }, text: push.map(function (v) { return Kit.fmt(v, 2); }), textposition: "outside", hoverinfo: "skip", cliponaxis: false }]; },
          function () { return { margin: { l: 44, r: 12, t: 18, b: 56 }, dragmode: false, xaxis: { tickangle: -25 }, yaxis: { range: [-1.25, 1.25], title: { text: "p − y", standoff: 4 } } }; });
        eq.innerHTML = "<span class='t c-teal'><b>" + Kit.fmt(pk[st.spec], 3) + "</b><i>p of the true species</i></span><span class='op'>&rarr; loss</span><span class='t c-navy'><b>" + Kit.fmt(-Math.log(pk[st.spec]), 3) + "</b><i>ℓ = −ln p</i></span><span class='op'>pushes add up to</span><span class='t'><b>" + Kit.fmt(push.reduce(function (a, b) { return a + b; }, 0), 2) + "</b><i>Σ (p − y)</i></span>";
        status = "The push on the true species is " + Kit.fmt(push[st.spec], 2) + ": it is told to rise. The other four are pushed down by their own probability. The pushes sum to 0, so probability moves from the wrong species to the true one.";
      }
      Kit.setText("co-status", status);
    }
    show(); draw();
  }

  // ======================================================================
  // 2. Loss landscapes with gradient descent paths
  // ======================================================================
  var PATH_STEPS = 700;
  function landFig() {
    var st = { start: [-4, 2], eta: 0.5, shown: 0, view: 0, stop: null }, gl = hasWebGL(), paths = {}, grids = {}, etas = [0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 3];
    var which = [["ce", "ld-ce"], ["mse", "ld-mse"]];
    grids.ce = gridOf(M.ce, 71, 56); grids.mse = gridOf(M.mse, 71, 56);
    function recompute() { paths.ce = gdPath(M.ce, st.start, st.eta, PATH_STEPS); paths.mse = gdPath(M.mse, st.start, st.eta, PATH_STEPS); }
    var se = Kit.slider(Kit.$("ld-controls"), { label: "learning rate η", min: 0, max: etas.length - 1, step: 1, value: etas.indexOf(st.eta), digits: 0, show: function (v) { return String(etas[v]); } }, function (v) { st.eta = etas[v]; reset(); });
    Kit.chips(Kit.$("ld-start"), ["start: confidently wrong (−4, 2)", "start: near the valley (2, 0)", "start: other side (−3, −3)"], 0, function (i) { st.start = [[-4, 2], [2, 0], [-3, -3]][i]; reset(); });
    Kit.chips(Kit.$("ld-view"), ["Contour map", "3D surface"], 0, function (i) { st.view = i; if (i === 1 && !gl) { Kit.setText("ld-status", "This browser cannot draw 3D graphics here. The contour map shows the same surface."); st.view = 0; return; } drawAll(); });
    Kit.button(Kit.$("ld-bar"), "One step", function () { halt(); st.shown = Math.min(st.shown + 1, maxShown()); drawAll(); });
    Kit.button(Kit.$("ld-bar"), "Run 300 steps", function () { halt(); var from = st.shown, to = Math.min(maxShown(), from + 300); st.stop = Kit.animate(300 + 5 * (to - from), function (p) { st.shown = Math.round(from + (to - from) * p); drawAll(true); if (p === 1) drawAll(); }); });
    Kit.button(Kit.$("ld-bar"), "Jump to step 700", function () { halt(); st.shown = maxShown(); drawAll(); });
    Kit.button(Kit.$("ld-bar"), "Reset", function () { reset(); });
    function halt() { if (st.stop) { st.stop(); st.stop = null; } }
    function maxShown() { return Math.min(paths.ce.length, paths.mse.length) - 1; }
    var accs = {};
    function reset() { halt(); recompute(); accs.ce = accSeries("ce"); accs.mse = accSeries("mse"); st.shown = 0; drawAll(); }
    function pathUpTo(key) { return paths[key].slice(0, Math.min(st.shown, paths[key].length - 1) + 1); }
    function tracesFor(key, P) {
      var mdl = M[key], g = grids[key], pu = pathUpTo(key), col = key === "ce" ? P.teal : P.coral, cur = pu[pu.length - 1];
      var cs = [[0, mix(P.paper, P.sand, 0.05)], [1, mix(P.paper, P.navy, 0.85)]];
      if (st.view === 0) {
        return [
          { type: "contour", x: g.ws, y: g.bs, z: g.z, colorscale: cs, zmin: 0, zmax: mdl.zclip, showscale: false, hoverinfo: "skip", contours: { coloring: "heatmap", start: mdl.levels.start, end: mdl.levels.end, size: mdl.levels.size, showlines: true }, line: { color: P.muted, width: 0.6, smoothing: 0.5 } },
          { type: "scatter", mode: "lines", x: pu.map(function (p) { return p[0]; }), y: pu.map(function (p) { return p[1]; }), line: { color: col, width: 2.6 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: [st.start[0]], y: [st.start[1]], marker: { size: 11, symbol: "circle-open", color: P.ink, line: { color: P.ink, width: 2.4 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: [cur[0]], y: [cur[1]], marker: { size: 11, color: col, line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers+text", x: [mdl.floor[0]], y: [mdl.floor[1]], text: ["valley floor"], textposition: "top left", textfont: { size: 12, color: P.ink2 }, marker: { size: 9, symbol: "x", color: P.ink }, hoverinfo: "skip" }];
      }
      return [
        { type: "surface", x: g.ws, y: g.bs, z: g.z, colorscale: cs, showscale: false, opacity: 0.93, hoverinfo: "skip", lighting: { ambient: 0.75, diffuse: 0.7, specular: 0.05, roughness: 0.8 }, contours: { z: { show: false } } },
        { type: "scatter3d", mode: "lines", x: pu.map(function (p) { return p[0]; }), y: pu.map(function (p) { return p[1]; }), z: pu.map(function (p) { return Math.min(mdl.zclip, mdl.L(p[0], p[1])) + 0.02 * mdl.zclip; }), line: { color: col, width: 6 }, hoverinfo: "skip" },
        { type: "scatter3d", mode: "markers", x: [cur[0]], y: [cur[1]], z: [Math.min(mdl.zclip, mdl.L(cur[0], cur[1])) + 0.03 * mdl.zclip], marker: { size: 5, color: col, line: { color: P.ink, width: 2 } }, hoverinfo: "skip" }];
    }
    function layoutFor(key) {
      var mdl = M[key];
      return function () {
        if (st.view === 0) return { margin: { l: 46, r: 10, t: 8, b: 46 }, dragmode: false, xaxis: { range: mdl.dom[0], title: { text: mdl.ax[0], standoff: 6 }, zeroline: false }, yaxis: { range: mdl.dom[1], title: { text: mdl.ax[1], standoff: 4 }, zeroline: false } };
        return { margin: { l: 0, r: 0, t: 0, b: 0 }, uirevision: "keep" + key, scene: { aspectmode: "manual", aspectratio: { x: 1.2, y: 1, z: 0.7 }, camera: { eye: { x: -1.5, y: -1.6, z: 0.9 } },
          xaxis: { range: mdl.dom[0], title: { text: "w" } }, yaxis: { range: mdl.dom[1], title: { text: "b" } }, zaxis: { range: [0, mdl.zclip], title: { text: "L" } } } };
      };
    }
    function accSeries(key) { return paths[key].map(function (p) { return M.acc(p[0], p[1]); }); }
    function plotAcc() {
      var n = st.shown, lz = D.y.reduce(function (a, b) { return a + b; }, 0) / D.m;      // share of adults: always answering "adult"
      Kit.plot("ld-acc", function (P) {
        var cut = function (k) { var a = accs[k].slice(0, n + 1); return { x: a.map(function (_, i) { return i; }), y: a }; }, c1 = cut("ce"), c2 = cut("mse");
        return [
          { type: "scatter", mode: "lines", x: [0, PATH_STEPS], y: [lz, lz], line: { color: P.muted, width: 1.4, dash: "dot" }, hoverinfo: "skip" },
          { type: "scatter", mode: "lines", x: c1.x, y: c1.y, line: { color: P.teal, width: 3 }, hoverinfo: "skip" },
          { type: "scatter", mode: "lines", x: c2.x, y: c2.y, line: { color: P.coral, width: 3 }, hoverinfo: "skip" }];
      }, function (P) {
        return { margin: { l: 52, r: 12, t: 8, b: 52 }, dragmode: false, xaxis: { range: [0, PATH_STEPS], title: { text: "gradient descent steps", standoff: 8 } }, yaxis: { range: [0.3, 1.02], tickformat: ".0%", tickvals: [0.4, 0.6, 0.8, 1] },
          annotations: [{ x: PATH_STEPS, y: lz, xref: "x", yref: "y", text: "always answering adult: " + Kit.pct(lz, 1), showarrow: false, font: { size: 12, color: P.muted }, xanchor: "right", yanchor: "top", yshift: -3 },
            { x: PATH_STEPS * 0.62, y: 0.925, xref: "x", yref: "y", text: "<b>cross-entropy</b>", showarrow: false, font: { size: 13, color: P.teal }, xanchor: "left", yanchor: "bottom" },
            { x: PATH_STEPS * 0.62, y: 0.52, xref: "x", yref: "y", text: "<b>squared error</b>", showarrow: false, font: { size: 13, color: P.coral }, xanchor: "left" }] };
      });
    }
    function drawAll(light) {
      which.forEach(function (w) {
        var key = w[0], id = w[1], div = Kit.$(id);
        if (light && div && div.data && div.data.length) {
          var pu = pathUpTo(key), cur = pu[pu.length - 1], mdl = M[key];
          if (st.view === 0) { Plotly.restyle(div, { x: [pu.map(function (p) { return p[0]; })], y: [pu.map(function (p) { return p[1]; })] }, [1]); Plotly.restyle(div, { x: [[cur[0]]], y: [[cur[1]]] }, [3]); }
          else { Plotly.restyle(div, { x: [pu.map(function (p) { return p[0]; })], y: [pu.map(function (p) { return p[1]; })], z: [pu.map(function (p) { return Math.min(mdl.zclip, mdl.L(p[0], p[1])) + 0.02 * mdl.zclip; })] }, [1]); }
        } else {
          Kit.plot(div, function (P) { return tracesFor(key, P); }, layoutFor(key));
        }
      });
      plotAcc();
      readouts();
    }
    function readouts() {
      var pc = pathUpTo("ce"), pm = pathUpTo("mse"), a = pc[pc.length - 1], b = pm[pm.length - 1], g = M.ce.g(a[0], a[1]), gm = M.mse.g(b[0], b[1]);
      Kit.setText("ld-step", String(st.shown));
      Kit.setText("ld-ce-loss", Kit.fmt(M.ce.L(a[0], a[1]), 3)); Kit.setText("ld-mse-loss", Kit.fmt(M.mse.L(b[0], b[1]), 3));
      Kit.setText("ld-ce-acc", Kit.pct(M.acc(a[0], a[1]), 1)); Kit.setText("ld-mse-acc", Kit.pct(M.acc(b[0], b[1]), 1));
      Kit.setText("ld-ce-wb", "(" + Kit.fmt(a[0], 2) + ", " + Kit.fmt(a[1], 2) + ")"); Kit.setText("ld-mse-wb", "(" + Kit.fmt(b[0], 2) + ", " + Kit.fmt(b[1], 2) + ")");
      Kit.setText("ld-gn-ce", Kit.fmt(Math.hypot(g[0], g[1]), 3)); Kit.setText("ld-gn-mse", Kit.fmt(Math.hypot(gm[0], gm[1]), 3));
      // the next update for the cross-entropy model, as boxes
      Kit.setText("ld-eq-w", Kit.fmt(a[0] - st.eta * g[0], 3)); Kit.setText("ld-eq-w0", Kit.fmt(a[0], 3)); Kit.setText("ld-eq-eta", String(st.eta)); Kit.setText("ld-eq-g", Kit.fmt(g[0], 3));
      Kit.setText("ld-eq-b", Kit.fmt(a[1] - st.eta * g[1], 3)); Kit.setText("ld-eq-b0", Kit.fmt(a[1], 3)); Kit.setText("ld-eq-eta2", String(st.eta)); Kit.setText("ld-eq-gb", Kit.fmt(g[1], 3));
      var stuck = st.shown >= 150 && M.mse.L(b[0], b[1]) > 0.2 && Math.abs(M.acc(b[0], b[1]) - (D.y.reduce(function (s, v) { return s + v; }, 0) / D.m)) < 0.001;
      Kit.setText("ld-status", st.shown === 0 ? "Both models start at the same weights. Press Run, or click a map to choose another start." :
        (stuck ? "Squared error is on a plateau: it calls all the sharks adults (the lazy answer) and its slope is almost zero. Cross-entropy left it long ago." : "Step " + st.shown + ": cross-entropy loss " + Kit.fmt(M.ce.L(a[0], a[1]), 3) + ", squared error loss " + Kit.fmt(M.mse.L(b[0], b[1]), 3) + "."));
    }
    // click on a map to choose a start
    function bindClicks() {
      which.forEach(function (w) {
        var div = Kit.$(w[1]); if (!div || !div.on || div.__bound === st.view) return;
        div.__bound = st.view;
        div.on("plotly_click", function (e) { if (st.view !== 0) return; var p = e.points && e.points[0]; if (!p) return; var m = M[w[0]]; st.start = [clamp(Math.round(p.x * 10) / 10, m.dom[0][0], m.dom[0][1]), clamp(Math.round(p.y * 10) / 10, m.dom[1][0], m.dom[1][1])]; reset(); });
      });
    }
    var drawAll0 = drawAll; drawAll = function (light) { drawAll0(light); bindClicks(); };
    reset();
  }

  // ======================================================================
  // 2b. A network's landscape has more than one valley
  // ======================================================================
  function valleyFig() {
    var mdl = M.net, st = { start: [1.5, 2], eta: 0.1, shown: 0, stop: null }, grid = gridOf(mdl, 71, 61), path = [], etas = [0.03, 0.1, 0.3, 0.5, 1];
    var N = 700;
    Kit.chips(Kit.$("vy-start"), ["start (1.5, 2)", "start (−1.5, 2)", "start (0.05, 0.05)", "start (0, 0)"], 0, function (i) { st.start = [[1.5, 2], [-1.5, 2], [0.05, 0.05], [0, 0]][i]; reset(); });
    Kit.slider(Kit.$("vy-controls"), { label: "learning rate η", min: 0, max: etas.length - 1, step: 1, value: etas.indexOf(st.eta), digits: 0, show: function (v) { return String(etas[v]); } }, function (v) { st.eta = etas[v]; reset(); });
    Kit.button(Kit.$("vy-bar"), "Run 700 steps", function () { halt(); st.stop = Kit.animate(2800, function (p) { st.shown = Math.round((path.length - 1) * p); draw(); }); });
    Kit.button(Kit.$("vy-bar"), "Reset", function () { reset(); });
    function halt() { if (st.stop) { st.stop(); st.stop = null; } }
    function reset() { halt(); path = gdPath(mdl, st.start, st.eta, N); st.shown = 0; draw(); }
    function fcurve(w1, w2) { var xs = lin(-2.6, 2.6, 80); return { x: xs, y: xs.map(function (v) { return w2 * Math.tanh(w1 * v); }) }; }
    var bound = false;
    function draw() {
      var pu = path.slice(0, st.shown + 1), cur = pu[pu.length - 1], cs = [[0, mix(Kit.palette().paper, Kit.palette().sand, 0.05)], [1, mix(Kit.palette().paper, Kit.palette().navy, 0.85)]];
      Kit.plot("vy-map", function (P) {
        var c2 = [[0, mix(P.paper, P.sand, 0.05)], [1, mix(P.paper, P.navy, 0.85)]];
        return [
          { type: "contour", x: grid.ws, y: grid.bs, z: grid.z, colorscale: c2, zmin: 0, zmax: mdl.zclip, showscale: false, hoverinfo: "skip", contours: { coloring: "heatmap", start: mdl.levels.start, end: mdl.levels.end, size: mdl.levels.size, showlines: true }, line: { color: P.muted, width: 0.6, smoothing: 0.5 } },
          { type: "scatter", mode: "lines", x: pu.map(function (p) { return p[0]; }), y: pu.map(function (p) { return p[1]; }), line: { color: P.teal, width: 2.6 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: [st.start[0]], y: [st.start[1]], marker: { size: 11, symbol: "circle-open", color: P.ink, line: { color: P.ink, width: 2.4 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: [cur[0]], y: [cur[1]], marker: { size: 11, color: P.teal, line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers+text", x: [0.25, -0.25, 0], y: [3.68, -3.68, 0], text: ["valley", "valley", "saddle"], textposition: ["top right", "bottom left", "top right"], textfont: { size: 12, color: P.ink2 }, marker: { size: 9, symbol: ["x", "x", "diamond-open"], color: P.ink, line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" }];
      }, function () { return { margin: { l: 46, r: 10, t: 8, b: 46 }, dragmode: false, xaxis: { range: mdl.dom[0], title: { text: mdl.ax[0], standoff: 6 }, zeroline: false }, yaxis: { range: mdl.dom[1], title: { text: mdl.ax[1], standoff: 4 }, zeroline: false } }; });
      Kit.plot("vy-fit", function (P) {
        var f = fcurve(cur[0], cur[1]);
        return [
          { type: "scatter", mode: "markers", x: D.x, y: D.t, marker: { size: 5, color: P.navy, opacity: 0.55 }, hoverinfo: "skip" },
          { type: "scatter", mode: "lines", x: f.x, y: f.y, line: { color: P.teal, width: 3.2 }, hoverinfo: "skip" }];
      }, function () { return { margin: { l: 46, r: 10, t: 8, b: 46 }, dragmode: false, xaxis: { title: { text: "scaled length x", standoff: 6 }, range: [-2.7, 2.7] }, yaxis: { title: { text: "scaled weight", standoff: 4 }, range: [-1.8, 4.5] } }; });
      var g = mdl.g(cur[0], cur[1]);
      Kit.setText("vy-step", String(st.shown)); Kit.setText("vy-w1", Kit.fmt(cur[0], 2)); Kit.setText("vy-w2", Kit.fmt(cur[1], 2)); Kit.setText("vy-loss", Kit.fmt(mdl.L(cur[0], cur[1]), 3));
      Kit.setText("vy-prod", Kit.fmt(cur[0] * cur[1], 2)); Kit.setText("vy-grad", Kit.fmt(Math.hypot(g[0], g[1]), 4));
      Kit.setText("vy-status", st.start[0] === 0 && st.start[1] === 0 ? "At (0, 0) the gradient is zero, so gradient descent never moves. The point is a saddle: downhill in some directions, flat for the gradient." : (st.shown === 0 ? "Press Run, or click the map to choose a start." : "Step " + st.shown + ": (w₁, w₂) = (" + Kit.fmt(cur[0], 2) + ", " + Kit.fmt(cur[1], 2) + "), loss " + Kit.fmt(mdl.L(cur[0], cur[1]), 3) + ", and the mirror pair of weights draws the same curve."));
      var div = Kit.$("vy-map");
      if (div && div.on && !div.__b) { div.__b = true; div.on("plotly_click", function (e) { var p = e.points && e.points[0]; if (!p) return; st.start = [clamp(Math.round(p.x * 20) / 20, mdl.dom[0][0], mdl.dom[0][1]), clamp(Math.round(p.y * 10) / 10, mdl.dom[1][0], mdl.dom[1][1])]; reset(); }); }
    }
    reset();
  }

  // ======================================================================
  // 3. Batches and epochs: which sharks make up a step
  // ======================================================================
  var BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 268];
  function stripFig() {
    var st = { bi: 4, order: [], step: 0, epoch: 1, timer: 0, seed: 11 };
    var sb = Kit.slider(Kit.$("st-controls"), { label: "batch size B", min: 0, max: BATCH_SIZES.length - 1, step: 1, value: st.bi, digits: 0, show: function (v) { return String(BATCH_SIZES[v]); } }, function (v) { st.bi = v; restart(); });
    var play = Kit.button(Kit.$("st-bar"), "Play", function () { if (st.timer) { stopPlay(); } else { play.textContent = "Pause"; st.timer = setInterval(function () { advance(); draw(); }, 260); } });
    Kit.button(Kit.$("st-bar"), "One step", function () { stopPlay(); advance(); draw(); });
    Kit.button(Kit.$("st-bar"), "Start over", function () { stopPlay(); restart(); });
    function stopPlay() { if (st.timer) { clearInterval(st.timer); st.timer = 0; } play.textContent = "Play"; }
    function B() { return BATCH_SIZES[st.bi]; }
    function shuffle() { var r = Kit.rng(st.seed++), a = D.x.map(function (_, i) { return i; }), i, j, t; for (i = a.length - 1; i > 0; i--) { j = Math.floor(r() * (i + 1)); t = a[i]; a[i] = a[j]; a[j] = t; } st.order = a; }
    function perEpoch() { return Math.ceil(D.m / B()); }
    function restart() { st.seed = 11; shuffle(); st.step = 0; st.epoch = 1; draw(); }
    function advance() { if (st.step >= perEpoch()) { st.step = 1; st.epoch++; shuffle(); } else st.step++; }
    function draw() {
      var svg = Kit.$("st-svg"), P = Kit.palette(), W = Math.max(280, Kit.width(svg)), cell = W < 520 ? 9 : 12, gap = W < 520 ? 3 : 3.5, cols = Math.max(10, Math.floor((W - 6) / (cell + gap))), rows = Math.ceil(D.m / cols), H = rows * (cell + gap) + 6, b = B();
      svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("height", H); svg.innerHTML = "";
      // cells in the order of the current epoch; the batch for the current step is outlined
      var lo = (st.step - 1) * b, hi = Math.min(D.m, st.step * b), i, pos, mature = 0, inb = 0;
      for (pos = 0; pos < D.m; pos++) {
        var idx = st.order[pos], col = pos % cols, row = Math.floor(pos / cols), used = st.step > 0 && pos < lo, cur = st.step > 0 && pos >= lo && pos < hi;
        var r = Kit.el("rect", { x: 3 + col * (cell + gap), y: 3 + row * (cell + gap), width: cell, height: cell, rx: 2, fill: D.y[idx] ? P.coral : P.navy, opacity: used ? 0.22 : 0.92 }, svg);
        if (cur) { r.setAttribute("stroke", P.ink); r.setAttribute("stroke-width", "2"); r.setAttribute("opacity", "1"); mature += D.y[idx]; inb++; }
      }
      var left = D.m - Math.min(D.m, st.step * b);
      Kit.setText("st-eq-b", String(b)); Kit.setText("st-eq-n", String(perEpoch())); Kit.setText("st-eq-m", String(D.m));
      Kit.setText("st-step", st.step + " of " + perEpoch()); Kit.setText("st-epoch", String(st.epoch)); Kit.setText("st-updates", String((st.epoch - 1) * perEpoch() + st.step));
      Kit.setText("st-batch", st.step ? String(inb) : "0"); Kit.setText("st-share", st.step && inb ? Kit.pct(mature / inb, 0) : "none yet");
      var last = D.m - (perEpoch() - 1) * b;
      Kit.setText("st-status", st.step === 0 ? "A square is one shark: coral for mature, navy for immature. Press One step or Play. The outlined squares are the sharks in the current step." :
        "Epoch " + st.epoch + ", step " + st.step + " of " + perEpoch() + ": the gradient is the average over the " + inb + " outlined sharks, of which " + Kit.pct(mature / inb, 0) + " are mature. All sharks: " + Kit.pct(D.y.reduce(function (a, c) { return a + c; }, 0) / D.m, 1) + ". " + (st.step === perEpoch() ? "That ends the epoch." + (last !== b ? " The last batch has only " + last + " sharks." : "") : left + " sharks wait their turn."));
    }
    shuffle(); draw();
    Kit.onTheme(draw);
    window.addEventListener("resize", Kit.debounce(draw, 250));
  }

  // ======================================================================
  // 4. Noisy steps: mini-batch paths and the noise of the gradient
  // ======================================================================
  function noiseFig() {
    var st = { bi: 4, eta: 0.3, epochs: 10, xmode: 0, seed: 5, start: [-4, 2], run: null }, etas = [0.05, 0.1, 0.2, 0.3, 0.5, 1];
    var grid = gridOf(M.ce, 71, 56), COMP = [1, 4, 16, 64, 268], sdv = Math.sqrt;
    var sb = Kit.slider(Kit.$("nz-controls"), { label: "batch size B", min: 0, max: BATCH_SIZES.length - 1, step: 1, value: st.bi, digits: 0, show: function (v) { return String(BATCH_SIZES[v]); } }, function (v) { st.bi = v; compute(); draw(); });
    Kit.slider(Kit.$("nz-controls"), { label: "learning rate η", min: 0, max: etas.length - 1, step: 1, value: etas.indexOf(st.eta), digits: 0, show: function (v) { return String(etas[v]); } }, function (v) { st.eta = etas[v]; compute(); draw(); });
    Kit.slider(Kit.$("nz-controls"), { label: "epochs", min: 1, max: 30, step: 1, value: st.epochs, digits: 0 }, function (v) { st.epochs = v; compute(); draw(); });
    Kit.chips(Kit.$("nz-bar"), ["Loss against epochs", "Loss against steps"], 0, function (i) { st.xmode = i; draw(); });
    Kit.button(Kit.$("nz-bar"), "Another shuffle", function () { st.seed++; compute(); draw(); });
    function B() { return BATCH_SIZES[st.bi]; }
    // one run of mini-batch gradient descent on the cross-entropy loss
    function sgd(b, eta, epochs, seed, start) {
      var r = Kit.rng(seed), m = D.m, x = D.x, y = D.y, idx = x.map(function (_, i) { return i; }), w = start[0], bb = start[1], path = [[w, bb]], ends = [0], losses = [M.ce.L(w, bb)], e, k, j, t, gw, gb, p, q;
      for (e = 0; e < epochs; e++) {
        for (j = m - 1; j > 0; j--) { q = Math.floor(r() * (j + 1)); t = idx[j]; idx[j] = idx[q]; idx[q] = t; }
        for (k = 0; k < m; k += b) {
          gw = 0; gb = 0; var hi = Math.min(m, k + b);
          for (j = k; j < hi; j++) { p = sig(w * x[idx[j]] + bb) - y[idx[j]]; gw += p * x[idx[j]]; gb += p; }
          w -= eta * gw / (hi - k); bb -= eta * gb / (hi - k); path.push([w, bb]);
        }
        ends.push(path.length - 1); losses.push(M.ce.L(w, bb));
      }
      return { path: path, ends: ends, losses: losses };
    }
    var runs = {};
    function compute() { COMP.concat([B()]).forEach(function (b) { runs[b] = sgd(b, st.eta, st.epochs, st.seed, st.start); }); }
    function plots() {
      var cur = runs[B()], perEp = Math.ceil(D.m / B());
      Kit.plot("nz-map", function (P) {
        var cs = [[0, mix(P.paper, P.sand, 0.05)], [1, mix(P.paper, P.navy, 0.85)]];
        return [
          { type: "contour", x: grid.ws, y: grid.bs, z: grid.z, colorscale: cs, zmin: 0, zmax: 6, showscale: false, hoverinfo: "skip", contours: { coloring: "heatmap", start: 0.25, end: 6, size: 0.5, showlines: true }, line: { color: P.muted, width: 0.6, smoothing: 0.5 } },
          { type: "scatter", mode: "lines", x: cur.path.map(function (p) { return p[0]; }), y: cur.path.map(function (p) { return p[1]; }), line: { color: P.teal, width: B() === 1 ? 1.1 : 1.8 }, opacity: 0.9, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: cur.ends.map(function (i) { return cur.path[i][0]; }), y: cur.ends.map(function (i) { return cur.path[i][1]; }), marker: { size: 7, color: P.coral, line: { color: P.ink, width: 1.2 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: [st.start[0]], y: [st.start[1]], marker: { size: 11, symbol: "circle-open", color: P.ink, line: { color: P.ink, width: 2.4 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers+text", x: [4.61], y: [2.48], text: ["valley floor"], textposition: "top left", textfont: { size: 12, color: P.ink2 }, marker: { size: 9, symbol: "x", color: P.ink }, hoverinfo: "skip" }];
      }, function () { return { margin: { l: 46, r: 10, t: 8, b: 46 }, dragmode: false, xaxis: { range: M.ce.dom[0], title: { text: M.ce.ax[0], standoff: 6 }, zeroline: false }, yaxis: { range: M.ce.dom[1], title: { text: M.ce.ax[1], standoff: 4 }, zeroline: false } }; });
      Kit.plot("nz-loss", function (P) {
        var tr = [], sel = B();
        COMP.concat(COMP.indexOf(sel) < 0 ? [sel] : []).forEach(function (b) {
          var r = runs[b], per = Math.ceil(D.m / b), xs = r.losses.map(function (_, i) { return st.xmode === 0 ? i : Math.max(1, i * per); }), isSel = b === sel;
          tr.push({ type: "scatter", mode: "lines", x: xs, y: r.losses, line: { color: isSel ? P.teal : P.muted, width: isSel ? 3.4 : 1.4 }, opacity: isSel ? 1 : 0.7, hoverinfo: "skip" });
        });
        return tr;
      }, function (P) {
        var ann = [], sel = B(), list = COMP.concat(COMP.indexOf(sel) < 0 ? [sel] : []).map(function (b) { var r = runs[b]; return { b: b, y: Math.log10(Math.max(0.2, r.losses[r.losses.length - 1])) }; }).sort(function (u, v) { return u.y - v.y; }), prevY = -9, shift = 0;
        list.forEach(function (it) { shift = it.y - prevY < 0.045 ? shift + 12 : 0; prevY = it.y; ann.push({ x: 0, y: it.y, xref: "x", yref: "y", text: "B = " + it.b, showarrow: false, font: { size: 12, color: it.b === sel ? P.teal : P.muted }, xanchor: "left", xshift: 4, yshift: shift }); });
        return { margin: { l: 52, r: 56, t: 8, b: 52 }, dragmode: false, annotations: st.xmode === 0 ? ann.map(function (a) { a.x = st.epochs; return a; }) : [], xaxis: st.xmode === 0 ? { range: [0, st.epochs], title: { text: "epochs (passes through the 268 sharks)", standoff: 8 } } : { type: "log", range: [0, Math.log10(Math.max(10, st.epochs * 268))], title: { text: "steps (updates), log scale", standoff: 8 } }, yaxis: { type: "log", range: [Math.log10(0.2), Math.log10(3.2)], title: { text: "cross-entropy loss, log scale", standoff: 4 }, tickvals: [0.25, 0.5, 1, 2], ticktext: ["0.25", "0.5", "1", "2"] } };
      });
      // the noise of the gradient on w at the valley floor
      var pts = [], theo = [], w0 = 4.61, b0 = 2.48, g1 = [], i, m = D.m;
      for (i = 0; i < m; i++) { var q = sig(w0 * D.x[i] + b0) - D.y[i]; g1.push(q * D.x[i]); }
      var mean = g1.reduce(function (a, c) { return a + c; }, 0) / m, s1 = sdv(g1.reduce(function (a, c) { return a + (c - mean) * (c - mean); }, 0) / m);
      var r = Kit.rng(99), idx = D.x.map(function (_, k) { return k; });
      BATCH_SIZES.slice(0, 8).forEach(function (b) {
        var vals = [], t, s, j, q2, tmp;
        for (t = 0; t < 400; t++) { s = 0; for (j = 0; j < b; j++) { q2 = j + Math.floor(r() * (m - j)); tmp = idx[j]; idx[j] = idx[q2]; idx[q2] = tmp; s += g1[idx[j]]; } vals.push(s / b); }
        var mv = vals.reduce(function (a, c) { return a + c; }, 0) / vals.length; pts.push(sdv(vals.reduce(function (a, c) { return a + (c - mv) * (c - mv); }, 0) / vals.length));
        theo.push(s1 / sdv(b) * sdv((m - b) / (m - 1)));
      });
      Kit.plot("nz-noise", function (P) { return [
        { type: "scatter", mode: "lines", x: BATCH_SIZES.slice(0, 8), y: theo, line: { color: P.sand, width: 2.4, dash: "dash" }, hoverinfo: "skip" },
        { type: "scatter", mode: "markers", x: BATCH_SIZES.slice(0, 8), y: pts, marker: { size: 9, color: P.navy, line: { color: P.paper, width: 1 } }, hoverinfo: "skip" },
        { type: "scatter", mode: "markers", x: B() < 268 ? [B()] : [], y: B() < 268 ? [pts[BATCH_SIZES.indexOf(B())]] : [], marker: { size: 14, symbol: "circle-open", color: P.coral, line: { color: P.coral, width: 3 } }, hoverinfo: "skip" }]; },
        function (P) { return { margin: { l: 62, r: 12, t: 8, b: 52 }, dragmode: false, xaxis: { type: "log", title: { text: "batch size B", standoff: 8 }, tickvals: BATCH_SIZES.slice(0, 8), ticktext: BATCH_SIZES.slice(0, 8).map(String), range: [-0.1, 2.25] }, yaxis: { type: "log", range: [-2.1, -0.6], title: { text: "gradient spread", standoff: 8 }, tickvals: [0.01, 0.02, 0.05, 0.1], ticktext: ["0.01", "0.02", "0.05", "0.1"] }, annotations: [{ x: Math.log10(2.2), y: Math.log10(theo[1]) + 0.26, xref: "x", yref: "y", text: "falls like 1/√B", showarrow: false, font: { size: 13, color: P.sand }, xanchor: "left" }] }; });
      var iB = BATCH_SIZES.indexOf(B());
      Kit.setText("nz-s1", Kit.fmt(s1, 3)); Kit.setText("nz-sb", iB < 8 ? Kit.fmt(pts[iB], 3) : "0"); Kit.setText("nz-b", String(B())); Kit.setText("nz-sq", Kit.fmt(sdv(B()), 2));
      Kit.setText("nz-theo", iB < 8 ? Kit.fmt(theo[iB], 3) : "0");
    }
    function draw() {
      var cur = runs[B()], per = Math.ceil(D.m / B()), last = cur.losses[cur.losses.length - 1];
      plots();
      Kit.setText("nz-per", String(per)); Kit.setText("nz-steps", String(cur.path.length - 1)); Kit.setText("nz-final", Kit.fmt(last, 3));
      Kit.setText("nz-status", "B = " + B() + " gives " + per + " step" + (per > 1 ? "s" : "") + " per epoch, so " + (cur.path.length - 1) + " steps in " + st.epochs + (st.epochs > 1 ? " epochs" : " epoch") + ". The cross-entropy loss after " + st.epochs + (st.epochs > 1 ? " epochs" : " epoch") + " is " + Kit.fmt(last, 3) + " (the valley floor is 0.232). The coral dots mark the ends of epochs.");
    }
    compute(); draw();
  }

})();
