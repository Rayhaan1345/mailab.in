/* B2 Why non-linearity matters.
   Figures: (1) the best straight line on blobs, XOR and circles; (2) circles lifted into 3D with a movable plane;
   (3) two layers with and without a bend between them; (4) a small network trained in the browser.
   The network in (4) is plain JavaScript (mini-batch gradient descent). How it computes its gradients is the
   topic of B4; here it is used as a black box. */
(function () {
  "use strict";

  // ---------- small helpers ----------
  function clamp(v, a, b) { return Math.min(b, Math.max(a, v)); }
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function hexRgb(c) {
    if (c.charAt(0) === "#") { c = c.replace("#", ""); if (c.length === 3) c = c.replace(/(.)/g, "$1$1"); return [0, 2, 4].map(function (i) { return parseInt(c.substr(i, 2), 16); }); }
    var m = c.match(/[\d.]+/g) || [0, 0, 0]; return [Number(m[0]), Number(m[1]), Number(m[2])];
  }
  function rgba(c, a) { var r = hexRgb(c); return "rgba(" + r[0] + "," + r[1] + "," + r[2] + "," + a + ")"; }
  function mixRgb(a, b, t) { var A = hexRgb(a), B = hexRgb(b); return [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }); }
  // value -> colour: navy below `mid`, neutral at `mid`, coral above; R is the half-range.
  function colorAt(P, v, mid, R) {
    var t = clamp((v - mid) / R, -1, 1);
    return t < 0 ? mixRgb(P.paper, P.navy, -t * 0.92) : mixRgb(P.paper, P.coral, t * 0.92);
  }
  // Paint an n by n grid of values (row 0 at the top) into a canvas, smoothed up to the canvas size.
  var scratch = document.createElement("canvas");
  function paintGrid(cv, vals, n, mid, R, P) {
    var ctx = cv.getContext("2d"), id, i, c;
    scratch.width = n; scratch.height = n;
    var sc = scratch.getContext("2d"); id = sc.createImageData(n, n);
    for (i = 0; i < n * n; i++) { c = colorAt(P, vals[i], mid, R); id.data[4 * i] = c[0]; id.data[4 * i + 1] = c[1]; id.data[4 * i + 2] = c[2]; id.data[4 * i + 3] = 255; }
    sc.putImageData(id, 0, 0);
    ctx.imageSmoothingEnabled = true; ctx.clearRect(0, 0, cv.width, cv.height); ctx.drawImage(scratch, 0, 0, cv.width, cv.height);
  }
  function hasWebGL() {
    try { var c = document.createElement("canvas"); return !!(c.getContext("webgl") || c.getContext("experimental-webgl")); } catch (e) { return false; }
  }
  function sizeCanvas(cv, cssW, cssH) {
    var d = Math.min(2, window.devicePixelRatio || 1);
    cv.style.width = cssW + "px"; cv.style.height = cssH + "px"; cv.width = Math.round(cssW * d); cv.height = Math.round(cssH * d);
    return d;
  }

  // half-plane {p : cx x + cy y >= cut} (sgn -1 for the other side) clipped to the square [-B, B]^2, as a polygon
  function halfPlane(cx, cy, cut, sgn, B, B2) {
    var x0 = -B, x1 = B, y0 = -B, y1 = B;
    if (B2) { x0 = B[0]; x1 = B[1]; y0 = B2[0]; y1 = B2[1]; }
    var sq = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]], out = [], i;
    function f(p) { return sgn * (cx * p[0] + cy * p[1] - cut); }
    for (i = 0; i < 4; i++) {
      var p = sq[i], q = sq[(i + 1) % 4], fp = f(p), fq = f(q);
      if (fp >= 0) out.push(p);
      if ((fp >= 0) !== (fq >= 0)) { var t = fp / (fp - fq); out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]); }
    }
    return out;
  }
  function lineEnds(cx, cy, cut, B) {
    var pts = [], t;
    if (Math.abs(cy) > 1e-9) { [-B, B].forEach(function (x) { t = (cut - cx * x) / cy; if (Math.abs(t) <= B + 1e-9) pts.push([x, t]); }); }
    if (Math.abs(cx) > 1e-9) { [-B, B].forEach(function (y) { t = (cut - cy * y) / cx; if (Math.abs(t) <= B + 1e-9) pts.push([t, y]); }); }
    return pts.length >= 2 ? [pts[0], pts[pts.length - 1]] : [[0, 0], [0, 0]];
  }

  // ---------- toy datasets in the square [-1, 1] by [-1, 1] (made up, not shark data) ----------
  var KINDS = ["blobs", "xor", "circles", "spiral"], KIND_NAME = { blobs: "Two blobs", xor: "XOR", circles: "Circles", spiral: "Spiral" };
  function makeData(kind, n, noise, seed) {
    var r = Kit.rng(seed), pts = [], i, c, x, y, t, rad, ang, h = n / 2, s;
    if (kind === "blobs") {
      for (i = 0; i < n; i++) { c = i < h ? 1 : 0; s = 0.17 + noise * 0.8; x = (c ? 0.5 : -0.5) + Kit.gauss(r) * s; y = (c ? 0.5 : -0.5) + Kit.gauss(r) * s; pts.push({ x: x, y: y, c: c }); }
    } else if (kind === "xor") {
      for (i = 0; i < n; i++) {
        do { x = r() * 2 - 1; y = r() * 2 - 1; } while (Math.abs(x) < 0.1 || Math.abs(y) < 0.1);
        c = x * y < 0 ? 1 : 0; x += Kit.gauss(r) * noise; y += Kit.gauss(r) * noise; pts.push({ x: x, y: y, c: c });
      }
    } else if (kind === "circles") {
      for (i = 0; i < n; i++) {
        c = i < h ? 0 : 1; rad = c ? 0.65 + r() * 0.3 : Math.sqrt(r()) * 0.42; ang = r() * 2 * Math.PI;
        x = rad * Math.cos(ang) + Kit.gauss(r) * noise * 0.6; y = rad * Math.sin(ang) + Kit.gauss(r) * noise * 0.6; pts.push({ x: x, y: y, c: c });
      }
    } else {
      for (i = 0; i < n; i++) {
        c = i < h ? 0 : 1; t = (i % h) / h; rad = 0.05 + t * 0.9; ang = 1.75 * t * 2 * Math.PI + (c ? Math.PI : 0);
        x = rad * Math.cos(ang) + Kit.gauss(r) * noise * 0.6; y = rad * Math.sin(ang) + Kit.gauss(r) * noise * 0.6; pts.push({ x: x, y: y, c: c });
      }
    }
    return pts;
  }
  // The best straight line: for 360 directions, the best cut along that direction. Returns {acc, theta, cut, side}.
  function bestLine(data) {
    var best = { acc: 0 }, a, n = data.length, tot1 = data.filter(function (p) { return p.c === 1; }).length;
    for (a = 0; a < 360; a += 1) {
      var th = a * Math.PI / 180, cx = Math.cos(th), cy = Math.sin(th);
      var pr = data.map(function (p) { return { s: p.x * cx + p.y * cy, c: p.c }; }).sort(function (u, v) { return u.s - v.s; });
      var below1 = 0, k, acc;
      for (k = 0; k <= n; k++) {
        acc = ((k - below1) + (tot1 - below1)) / n;     // points below the cut called 0, points above called 1
        if (acc > best.acc) best = { acc: acc, theta: th, cut: k > 0 && k < n ? (pr[k - 1].s + pr[k].s) / 2 : 0 };
        if (k < n && pr[k].c === 1) below1++;
      }
    }
    return best;
  }

  // ---------- a small network, trained by mini-batch gradient descent (see B4 for the gradients) ----------
  var ACT = {
    tanh: { f: Math.tanh, d: function (a) { return 1 - a * a; }, mid: 0, R: 1 },
    relu: { f: function (z) { return z > 0 ? z : 0; }, d: function (a) { return a > 0 ? 1 : 0; }, mid: 0, R: 1.5 },
    sigmoid: { f: function (z) { return 1 / (1 + Math.exp(-z)); }, d: function (a) { return a * (1 - a); }, mid: 0.5, R: 0.5 },
    none: { f: function (z) { return z; }, d: function () { return 1; }, mid: 0, R: 1.5 }
  };
  function Net(sizes, act, seed) {
    this.sizes = sizes; this.act = act; var r = Kit.rng(seed * 7919 + 13), l, i, j;
    this.W = []; this.b = [];
    for (l = 1; l < sizes.length; l++) {
      var fi = sizes[l - 1], fo = sizes[l], lim = (act === "relu" && l < sizes.length - 1) ? Math.sqrt(6 / fi) : Math.sqrt(6 / (fi + fo)), W = [];
      for (j = 0; j < fo; j++) { var row = []; for (i = 0; i < fi; i++) row.push((r() * 2 - 1) * lim); W.push(row); }
      this.W.push(W); this.b.push(new Array(fo).fill(0));
    }
  }
  Net.prototype.forward = function (x) {
    var a = [x], L = this.W.length, l, j, i, f = ACT[this.act].f;
    for (l = 0; l < L; l++) {
      var out = [], W = this.W[l], b = this.b[l], prev = a[l];
      for (j = 0; j < W.length; j++) {
        var z = b[j], row = W[j];
        for (i = 0; i < row.length; i++) z += row[i] * prev[i];
        out.push(l === L - 1 ? 1 / (1 + Math.exp(-z)) : f(z));
      }
      a.push(out);
    }
    return a;
  };
  // One mini-batch update. The output unit is a sigmoid with cross-entropy loss, so its error is p - y.
  Net.prototype.step = function (batch, lr) {
    var L = this.W.length, k, l, i, j, d, a, ex, W, prev, dn, d2 = ACT[this.act].d;
    var gW = this.W.map(function (W) { return W.map(function (r) { return r.map(function () { return 0; }); }); }), gb = this.b.map(function (b) { return b.map(function () { return 0; }); });
    for (k = 0; k < batch.length; k++) {
      ex = batch[k]; a = this.forward(ex.x); d = [a[L][0] - ex.c];
      for (l = L - 1; l >= 0; l--) {
        W = this.W[l]; prev = a[l]; dn = new Array(prev.length).fill(0);
        for (j = 0; j < W.length; j++) { gb[l][j] += d[j]; for (i = 0; i < prev.length; i++) { gW[l][j][i] += d[j] * prev[i]; dn[i] += d[j] * W[j][i]; } }
        if (l > 0) { for (i = 0; i < dn.length; i++) dn[i] *= d2(prev[i]); d = dn; }
      }
    }
    var s = lr / batch.length;
    for (l = 0; l < L; l++) for (j = 0; j < this.W[l].length; j++) { this.b[l][j] -= s * gb[l][j]; for (i = 0; i < this.W[l][j].length; i++) this.W[l][j][i] -= s * gW[l][j][i]; }
  };
  Net.prototype.params = function () { var n = 0, l; for (l = 1; l < this.sizes.length; l++) n += (this.sizes[l - 1] + 1) * this.sizes[l]; return n; };

  document.addEventListener("DOMContentLoaded", function () { oneFig(); liftFig(); bendFig(); playFig(); });

  // ======================================================================
  // 1. One neuron: the best straight line
  // ======================================================================
  function oneFig() {
    var st = { kind: "xor", seed: 3, theta: 40, cut: 0, flip: false, data: null, best: null };
    function newData() { st.data = makeData(st.kind, 200, 0.05, st.seed); st.best = bestLine(st.data); }
    newData();
    Kit.chips(Kit.$("on-kind"), ["Two blobs", "XOR", "Circles"], 1, function (i) { st.kind = ["blobs", "xor", "circles"][i]; newData(); st.flip = false; draw(); });
    var sa = Kit.slider(Kit.$("on-controls"), { label: "line direction, degrees", min: 0, max: 179, step: 1, value: st.theta, digits: 0 }, function (v) { st.theta = v; draw(); });
    var sc = Kit.slider(Kit.$("on-controls"), { label: "line position c", min: -1, max: 1, step: 0.01, value: st.cut, digits: 2 }, function (v) { st.cut = v; draw(); });
    Kit.button(Kit.$("on-bar"), "Flip the side", function () { st.flip = !st.flip; draw(); });
    Kit.button(Kit.$("on-bar"), "Find the best line", function () {
      var b = st.best, deg = b.theta * 180 / Math.PI, cut = b.cut, r0, a, c, f, out = { acc: -1 };
      if (deg >= 180) { deg -= 180; cut = -cut; }
      r0 = Math.round(deg);
      for (a = Math.max(0, r0 - 3); a <= Math.min(179, r0 + 3); a++) for (c = clamp(cut - 0.1, -1, 1); c <= clamp(cut + 0.1, -1, 1) + 1e-9; c += 0.01) for (f = 0; f < 2; f++) {
        var sc2 = score(a, Math.round(c * 100) / 100, !!f);
        if (sc2 > out.acc) out = { acc: sc2, theta: a, cut: Math.round(c * 100) / 100, flip: !!f };
      }
      st.theta = out.theta; st.cut = out.cut; st.flip = out.flip; sa.set(st.theta); sc.set(st.cut); draw();
    });
    Kit.button(Kit.$("on-bar"), "New points", function () { st.seed++; newData(); draw(); });
    function score(theta, cut, flip) {
      var th = theta * Math.PI / 180, cx = Math.cos(th), cy = Math.sin(th), ok = 0;
      st.data.forEach(function (p) { var pred = (p.x * cx + p.y * cy - cut >= 0) !== flip ? 1 : 0; if (pred === p.c) ok++; });
      return ok / st.data.length;
    }
    function draw() {
      var th = st.theta * Math.PI / 180, cx = Math.cos(th), cy = Math.sin(th), acc = score(st.theta, st.cut, st.flip), poly = halfPlane(cx, cy, st.cut, st.flip ? -1 : 1, 1.25), ends = lineEnds(cx, cy, st.cut, 1.25);
      var wrong = st.data.filter(function (p) { return (((p.x * cx + p.y * cy - st.cut >= 0) !== st.flip) ? 1 : 0) !== p.c; });
      Kit.plot("on-plot", function (P) {
        function cls(c) { var s = st.data.filter(function (p) { return p.c === c; }); return { x: s.map(function (p) { return p.x; }), y: s.map(function (p) { return p.y; }) }; }
        var c0 = cls(0), c1 = cls(1);
        return [
          { type: "scatter", mode: "lines", x: poly.map(function (p) { return p[0]; }).concat([poly[0][0]]), y: poly.map(function (p) { return p[1]; }).concat([poly[0][1]]), fill: "toself", fillcolor: rgba(P.coral, 0.13), line: { width: 0 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: c0.x, y: c0.y, marker: { symbol: "circle", size: 7, color: P.navy, opacity: 0.85, line: { color: P.paper, width: 0.8 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: c1.x, y: c1.y, marker: { symbol: "diamond", size: 8, color: P.coral, opacity: 0.9, line: { color: P.paper, width: 0.8 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: wrong.map(function (p) { return p.x; }), y: wrong.map(function (p) { return p.y; }), marker: { symbol: "circle-open", size: 14, color: P.ink, line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "lines", x: [ends[0][0], ends[1][0]], y: [ends[0][1], ends[1][1]], line: { color: P.ink, width: 2.4, dash: "dash" }, hoverinfo: "skip" }];
      }, function (P) {
        return { margin: { l: 36, r: 10, t: 8, b: 36 }, dragmode: false, xaxis: { range: [-1.25, 1.25], zeroline: false, constrain: "domain", title: { text: "x" } }, yaxis: { range: [-1.25, 1.25], zeroline: false, scaleanchor: "x", title: { text: "y" } }, annotations: [
          { x: poly.length ? poly.reduce(function (s, p) { return s + p[0]; }, 0) / poly.length : 0, y: poly.length ? poly.reduce(function (s, p) { return s + p[1]; }, 0) / poly.length : 0, xref: "x", yref: "y", text: "called 1", showarrow: false, font: { color: P.coral, size: 13 }, opacity: 0.9 }] };
      });
      var w1 = (st.flip ? -1 : 1) * cx, w2 = (st.flip ? -1 : 1) * cy, b = (st.flip ? 1 : -1) * st.cut;
      Kit.setText("on-w1", Kit.fmt(w1, 2)); Kit.setText("on-w2", Kit.fmt(w2, 2)); Kit.setText("on-b", Kit.fmt(b, 2));
      Kit.setText("on-acc", Kit.pct(acc, 1)); Kit.setText("on-best", Kit.pct(st.best.acc, 1)); Kit.setText("on-wrong", String(wrong.length));
      Kit.setText("on-status", st.kind === "blobs" ? "The two blobs are far apart, so one line separates them. Move the line until the score reaches 100%." :
        (acc >= st.best.acc - 0.0001 ? "This line is as good as any straight line can be for these points: " + Kit.pct(st.best.acc, 1) + ". Try to beat it." : "Drag the sliders, or press \"Find the best line\" to see how far one line can go."));
    }
    draw();
  }

  // ======================================================================
  // 2. Lifting the circles into 3D
  // ======================================================================
  function liftFig() {
    var data = makeData("circles", 200, 0.03, 5), st = { t: 0, c: 0.3 }, gl = hasWebGL(), stop = null;
    var pts = data.map(function (p) { return { x: p.x, y: p.y, c: p.c, r2: p.x * p.x + p.y * p.y }; });
    var A = { x: 0.3, y: 0.2, name: "A" }, B = { x: 0.8, y: -0.5, name: "B" };
    A.r2 = A.x * A.x + A.y * A.y; B.r2 = B.x * B.x + B.y * B.y;
    var sl = Kit.slider(Kit.$("lf-controls"), { label: "lift: how far the points rise", min: 0, max: 1, step: 0.01, value: st.t, digits: 2, show: function (v) { return Math.round(v * 100) + "%"; } }, function (v) { st.t = v; draw(); });
    var sc = Kit.slider(Kit.$("lf-controls"), { label: "plane height c", min: 0, max: 1.2, step: 0.01, value: st.c, digits: 2 }, function (v) { st.c = v; draw(); });
    Kit.button(Kit.$("lf-bar"), "Lift the points", function () { if (stop) stop(); var from = st.t; stop = Kit.animate(1800 * (1 - from) + 200, function (p) { st.t = from + (1 - from) * Kit.ease(p); sl.set(st.t); draw(); }); });
    Kit.button(Kit.$("lf-bar"), "Lower them", function () { if (stop) stop(); var from = st.t; stop = Kit.animate(900 * from + 100, function (p) { st.t = from * (1 - Kit.ease(p)); sl.set(st.t); draw(); }); });
    Kit.button(Kit.$("lf-bar"), "Best plane", function () {
      var vals = pts.map(function (p) { return { z: st.t * p.r2, c: p.c }; }).sort(function (a, b) { return a.z - b.z; }), n = vals.length, tot1 = vals.filter(function (v) { return v.c === 1; }).length, below1 = 0, k, bestA = -1, bestC = 0, acc;
      for (k = 0; k <= n; k++) {
        acc = ((k - below1) + (tot1 - below1)) / n;
        if (acc > bestA) { bestA = acc; bestC = k > 0 && k < n ? (vals[k - 1].z + vals[k].z) / 2 : (k === 0 ? vals[0].z - 0.01 : vals[n - 1].z + 0.01); }
        if (k < n && vals[k].c === 1) below1++;
      }
      st.c = clamp(Math.round(bestC * 100) / 100, 0, 1.2); sc.set(st.c); draw();
    });
    function draw() {
      var t = st.t, c = st.c, ok = 0, i, p, z;
      pts.forEach(function (q) { z = t * q.r2; if ((z >= c ? 1 : 0) === q.c) ok++; });
      var acc = ok / pts.length, rad = t > 1e-6 ? Math.sqrt(c / t) : Infinity;
      if (gl) {
        Kit.plot("lf-3d", function (P) {
          function cls(k) { var s = pts.filter(function (q) { return q.c === k; }); return { x: s.map(function (q) { return q.x; }), y: s.map(function (q) { return q.y; }), z: s.map(function (q) { return t * q.r2; }) }; }
          var c0 = cls(0), c1 = cls(1);
          return [
            { type: "scatter3d", mode: "markers", x: c0.x, y: c0.y, z: c0.z, marker: { size: 4.4, color: P.navy, symbol: "circle", opacity: 0.95 }, hoverinfo: "skip" },
            { type: "scatter3d", mode: "markers", x: c1.x, y: c1.y, z: c1.z, marker: { size: 4.4, color: P.coral, symbol: "diamond", opacity: 0.95 }, hoverinfo: "skip" },
            { type: "surface", x: [-1.1, 1.1], y: [-1.1, 1.1], z: [[c, c], [c, c]], showscale: false, opacity: 0.38, colorscale: [[0, P.teal], [1, P.teal]], hoverinfo: "skip" },
            { type: "scatter3d", mode: "markers+text", x: [A.x, B.x], y: [A.y, B.y], z: [t * A.r2, t * B.r2], text: ["A", "B"], textposition: "top center", textfont: { color: P.ink, size: 14 }, marker: { size: 6, color: P.paper, line: { color: P.ink, width: 3 } }, hoverinfo: "skip" }];
        }, function (P) {
          return { margin: { l: 0, r: 0, t: 0, b: 0 }, scene: { aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.8 }, camera: { eye: { x: 1.35, y: -1.45, z: 0.8 } },
            xaxis: { range: [-1.1, 1.1], title: { text: "x" } }, yaxis: { range: [-1.1, 1.1], title: { text: "y" } }, zaxis: { range: [0, 1.25], title: { text: "z" } } },
            uirevision: "keep" };
        }, { displayModeBar: false });
      } else {
        Kit.$("lf-3d").textContent = "This browser cannot draw 3D graphics here. The strip below shows the same heights.";
      }
      Kit.plot("lf-2d", function (P) {
        function cls(k) { var s = pts.filter(function (q) { return q.c === k; }); return { x: s.map(function (q) { return q.x; }), y: s.map(function (q) { return q.y; }) }; }
        var c0 = cls(0), c1 = cls(1), wrong = pts.filter(function (q) { return ((t * q.r2 >= c ? 1 : 0) !== q.c); });
        return [
          { type: "scatter", mode: "markers", x: c0.x, y: c0.y, marker: { symbol: "circle", size: 6, color: P.navy, opacity: 0.85 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: c1.x, y: c1.y, marker: { symbol: "diamond", size: 7, color: P.coral, opacity: 0.9 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: wrong.map(function (q) { return q.x; }), y: wrong.map(function (q) { return q.y; }), marker: { symbol: "circle-open", size: 13, color: P.ink, line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers+text", x: [A.x, B.x], y: [A.y, B.y], text: ["A", "B"], textposition: "top center", textfont: { color: P.ink, size: 14 }, marker: { size: 6, color: P.paper, line: { color: P.ink, width: 2.4 } }, hoverinfo: "skip" }];
      }, function (P) {
        var shapes = [];
        if (isFinite(rad) && rad < 1.6) shapes.push({ type: "circle", xref: "x", yref: "y", x0: -rad, x1: rad, y0: -rad, y1: rad, line: { color: P.teal, width: 3, dash: "dash" }, fillcolor: "rgba(0,0,0,0)" });
        return { margin: { l: 36, r: 10, t: 8, b: 36 }, dragmode: false, shapes: shapes, xaxis: { range: [-1.25, 1.25], zeroline: false, constrain: "domain", title: { text: "x" } }, yaxis: { range: [-1.25, 1.25], zeroline: false, scaleanchor: "x", title: { text: "y" } } };
      });
      Kit.plot("lf-strip", function (P) {
        var r = Kit.rng(2);
        function cls(k) { var s = pts.filter(function (q) { return q.c === k; }); return { x: s.map(function (q) { return t * q.r2; }), y: s.map(function () { return (r() - 0.5) * 0.7; }) }; }
        var c0 = cls(0), c1 = cls(1);
        return [
          { type: "scatter", mode: "markers", x: c0.x, y: c0.y, marker: { symbol: "circle", size: 6, color: P.navy, opacity: 0.8 }, hoverinfo: "skip" },
          { type: "scatter", mode: "markers", x: c1.x, y: c1.y, marker: { symbol: "diamond", size: 7, color: P.coral, opacity: 0.85 }, hoverinfo: "skip" },
          { type: "scatter", mode: "lines", x: [c, c], y: [-0.6, 0.6], line: { color: P.teal, width: 3 }, hoverinfo: "skip" }];
      }, function (P) {
        return { margin: { l: 14, r: 14, t: 6, b: 52 }, dragmode: false, xaxis: { range: [-0.03, 1.25], title: { text: "height z = lift × (x² + y²)", standoff: 8 }, zeroline: false }, yaxis: { range: [-0.7, 0.7], showticklabels: false, showgrid: false, zeroline: false },
          annotations: [{ x: c, y: 0.6, xref: "x", yref: "y", text: "plane", showarrow: false, font: { color: P.teal, size: 13 }, yanchor: "bottom", xshift: 18 }] };
      });
      // live boxes for the two marked points
      function box(id, pt) {
        var z = t * pt.r2;
        Kit.setText(id + "-x", Kit.fmt(pt.x, 2)); Kit.setText(id + "-y", Kit.fmt(pt.y, 2)); Kit.setText(id + "-z", Kit.fmt(z, 3));
        Kit.setText(id + "-call", z >= c ? "above: ring" : "below: inner");
      }
      box("lf-a", A); box("lf-b", B);
      Kit.setText("lf-eq-c", Kit.fmt(c, 2)); Kit.setText("lf-eq-c2", Kit.fmt(c, 2)); Kit.setText("lf-acc", Kit.pct(acc, 1)); Kit.setText("lf-wrong", String(pts.length - ok));
      Kit.setText("lf-rad", isFinite(rad) ? Kit.fmt(rad, 2) : "none");
      Kit.setText("lf-status", t < 0.98 ? "Raise the lift to 100% to see the flat plane cut the two layers apart." : (acc > 0.995 ? "The plane separates all 200 points. Seen from above, the cut is the dashed circle of radius √c = " + Kit.fmt(rad, 2) + "." : "Move the plane between the two layers of points."));
    }
    draw();
    window.addEventListener("resize", Kit.debounce(draw, 250));
  }

  // ======================================================================
  // 3. Two layers, with and without a bend between them
  // ======================================================================
  function bendFig() {
    var XOR = { W1: [[1, 1], [1, 1]], b1: [0, -1], W2: [1, -2], b2: 0 };
    var st = { W1: XOR.W1.map(function (r) { return r.slice(); }), b1: XOR.b1.slice(), W2: XOR.W2.slice(), b2: XOR.b2, x: [0, 0], relu: true };
    var corners = [[0, 0], [1, 0], [0, 1], [1, 1]], truth = [0, 1, 1, 0];
    var sx = [0, 1].map(function (i) { return Kit.slider(Kit.$("bd-controls"), { label: "input x" + (i + 1), min: 0, max: 1, step: 0.05, value: st.x[i], digits: 2 }, function (v) { st.x[i] = v; draw(); }); });
    Kit.chips(Kit.$("bd-corners"), ["(0, 0)", "(1, 0)", "(0, 1)", "(1, 1)"], 0, function (i) { st.x = corners[i].slice(); sx[0].set(st.x[0]); sx[1].set(st.x[1]); draw(); });
    Kit.chips(Kit.$("bd-act"), ["no bend between the layers", "ReLU between the layers"], 1, function (i) { st.relu = i === 1; draw(); });
    Kit.button(Kit.$("bd-bar"), "Reset to the XOR numbers", function () { st.W1 = XOR.W1.map(function (r) { return r.slice(); }); st.b1 = XOR.b1.slice(); st.W2 = XOR.W2.slice(); st.b2 = XOR.b2; fillInputs(); draw(); });
    Kit.button(Kit.$("bd-bar"), "Random numbers", function () {
      function rv() { return Math.round((Math.random() * 4 - 2) * 2) / 2; }
      st.W1 = [[rv(), rv()], [rv(), rv()]]; st.b1 = [rv(), rv()]; st.W2 = [rv(), rv()]; st.b2 = rv(); fillInputs(); draw();
    });
    // the boxes with numbers: matrices with editable cells
    var host = Kit.$("bd-mat");
    function cell(id, v) { return "<td><input class='mi' type='number' step='0.5' id='" + id + "' value='" + v + "' aria-label='" + id + "'></td>"; }
    function out(id) { return "<td><span id='" + id + "'></span></td>"; }
    function mx(rows) { return "<table class='mx'>" + rows.map(function (r) { return "<tr>" + r.join("") + "</tr>"; }).join("") + "</table>"; }
    function grp(label, html, cls) { return "<span class='mxg" + (cls ? " " + cls : "") + "'><span class='mxl'>" + label + "</span>" + html + "</span>"; }
    function op(t) { return "<span class='op'>" + t + "</span>"; }
    host.innerHTML =
      "<div class='mxrow'><span class='mxt'>Layer 1</span>" + grp("weights W¹", mx([[cell("w1_00", st.W1[0][0]), cell("w1_01", st.W1[0][1])], [cell("w1_10", st.W1[1][0]), cell("w1_11", st.W1[1][1])]])) + op("×") +
      grp("input x", mx([[out("x_0")], [out("x_1")]])) + op("+") + grp("bias b¹", mx([[cell("b1_0", st.b1[0])], [cell("b1_1", st.b1[1])]])) + op("=") +
      grp("sums z¹", mx([[out("z_0")], [out("z_1")]])) + "<span class='op' id='bd-g'></span>" + grp("hidden h", mx([[out("h_0")], [out("h_1")]]), "hi") + "</div>" +
      "<div class='mxrow'><span class='mxt'>Layer 2</span>" + grp("weights W²", mx([[cell("w2_0", st.W2[0]), cell("w2_1", st.W2[1])]])) + op("×") +
      grp("hidden h", mx([[out("h2_0")], [out("h2_1")]])) + op("+") + grp("bias b²", mx([[cell("b2_0", st.b2)]])) + op("=") + grp("output y", mx([[out("y_0")]]), "hi") + "</div>" +
      "<div class='mxrow'><span class='mxt'>One layer instead</span>" + grp("W² W¹", mx([[out("m_0"), out("m_1")]])) + op("×") + grp("input x", mx([[out("x2_0")], [out("x2_1")]])) + op("+") +
      grp("W² b¹ + b²", mx([[out("c_0")]])) + op("=") + grp("output y", mx([[out("yl_0")]]), "hi") + "</div>";
    function fillInputs() {
      var set = function (id, v) { Kit.$(id).value = v; };
      set("w1_00", st.W1[0][0]); set("w1_01", st.W1[0][1]); set("w1_10", st.W1[1][0]); set("w1_11", st.W1[1][1]); set("b1_0", st.b1[0]); set("b1_1", st.b1[1]); set("w2_0", st.W2[0]); set("w2_1", st.W2[1]); set("b2_0", st.b2);
    }
    host.addEventListener("input", function (e) {
      var id = e.target.id, v = parseFloat(e.target.value); if (isNaN(v) || !id) return;
      var m = { w1_00: function () { st.W1[0][0] = v; }, w1_01: function () { st.W1[0][1] = v; }, w1_10: function () { st.W1[1][0] = v; }, w1_11: function () { st.W1[1][1] = v; }, b1_0: function () { st.b1[0] = v; }, b1_1: function () { st.b1[1] = v; }, w2_0: function () { st.W2[0] = v; }, w2_1: function () { st.W2[1] = v; }, b2_0: function () { st.b2 = v; } };
      if (m[id]) { m[id](); draw(); }
    });
    function f(v, d) { return Kit.fmt(v, d == null ? 2 : d); }
    function net(x) {
      var z = [st.W1[0][0] * x[0] + st.W1[0][1] * x[1] + st.b1[0], st.W1[1][0] * x[0] + st.W1[1][1] * x[1] + st.b1[1]];
      var h = st.relu ? z.map(function (v) { return Math.max(0, v); }) : z.slice();
      return { z: z, h: h, y: st.W2[0] * h[0] + st.W2[1] * h[1] + st.b2 };
    }
    var cv = Kit.$("bd-map"), N = 40;
    function drawMap() {
      var P = Kit.palette(), d = sizeCanvas(cv, 300, 300), vals = new Float32Array(N * N), i, j, lo = -0.25, hi = 1.25, ctx = cv.getContext("2d");
      for (j = 0; j < N; j++) for (i = 0; i < N; i++) vals[j * N + i] = net([lo + (hi - lo) * (i + 0.5) / N, hi - (hi - lo) * (j + 0.5) / N]).y;
      paintGrid(cv, vals, N, 0.5, 1, P);
      function px(v) { return (v - lo) / (hi - lo) * cv.width; } function py(v) { return (hi - v) / (hi - lo) * cv.height; }
      ctx.strokeStyle = P.muted; ctx.lineWidth = d; ctx.strokeRect(0.5, 0.5, cv.width - 1, cv.height - 1);
      corners.forEach(function (c, k) {
        ctx.beginPath(); ctx.arc(px(c[0]), py(c[1]), 8 * d, 0, 6.2832); ctx.fillStyle = truth[k] ? P.coral : P.navy; ctx.fill(); ctx.lineWidth = 2 * d; ctx.strokeStyle = P.paper; ctx.stroke();
        ctx.fillStyle = P.paper; ctx.font = (11 * d) + "px Spectral, Georgia, serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(String(truth[k]), px(c[0]), py(c[1]) + 0.5 * d);
      });
      ctx.beginPath(); ctx.arc(px(st.x[0]), py(st.x[1]), 7 * d, 0, 6.2832); ctx.fillStyle = P.teal; ctx.fill(); ctx.lineWidth = 2.4 * d; ctx.strokeStyle = P.ink; ctx.stroke();
      ctx.fillStyle = P.ink2; ctx.font = (12.5 * d) + "px Spectral, Georgia, serif"; ctx.textAlign = "left"; ctx.textBaseline = "alphabetic"; ctx.fillText("x₁ →", cv.width - 38 * d, cv.height - 6 * d);
      ctx.textAlign = "left"; ctx.fillText("x₂ ↑", 6 * d, 16 * d);
    }
    function draw() {
      var o = net(st.x), M = [st.W2[0] * st.W1[0][0] + st.W2[1] * st.W1[1][0], st.W2[0] * st.W1[0][1] + st.W2[1] * st.W1[1][1]], cc = st.W2[0] * st.b1[0] + st.W2[1] * st.b1[1] + st.b2, yl = M[0] * st.x[0] + M[1] * st.x[1] + cc;
      var set = function (id, v) { Kit.setText(id, v); };
      set("x_0", f(st.x[0])); set("x_1", f(st.x[1])); set("x2_0", f(st.x[0])); set("x2_1", f(st.x[1]));
      set("z_0", f(o.z[0])); set("z_1", f(o.z[1])); set("h_0", f(o.h[0])); set("h_1", f(o.h[1])); set("h2_0", f(o.h[0])); set("h2_1", f(o.h[1]));
      set("y_0", f(o.y)); set("m_0", f(M[0])); set("m_1", f(M[1])); set("c_0", f(cc)); set("yl_0", f(yl));
      Kit.$("bd-g").textContent = st.relu ? "→ ReLU →" : "→ no bend →";
      var scores = corners.map(function (c, k) { return ((net(c).y >= 0.5 ? 1 : 0) === truth[k]); }), right = scores.filter(Boolean).length;
      Kit.setText("bd-right", right + " of 4");
      var same = Math.abs(o.y - yl) < 1e-9;
      Kit.setText("bd-status", (st.relu ? (same ? "At this input the two routes happen to agree: both hidden sums are positive or both are cut at 0 in the same way. Try the corner (0, 0)." : "The bend makes the two routes differ: the network gives " + f(o.y) + ", the single layer " + f(yl) + ".") :
        "Without a bend, the two routes always agree (" + f(o.y) + " and " + f(yl) + "). Two layers are one layer. Change any number and check.") + " The calls at the four corners: " + right + " right.");
      drawMap();
      var hp = corners.map(function (c) { return net(c).h; }), px = hp.map(function (h) { return h[0]; }), py = hp.map(function (h) { return h[1]; });
      var span = [Math.min(-0.5, Math.min.apply(null, px.concat([o.h[0]])) - 0.4), Math.max(1.5, Math.max.apply(null, px.concat([o.h[0]])) + 0.4)], spany = [Math.min(-0.5, Math.min.apply(null, py.concat([o.h[1]])) - 0.4), Math.max(1.5, Math.max.apply(null, py.concat([o.h[1]])) + 0.4)];
      Kit.plot("bd-hid", function (P) {
        var tr = [], B = Math.max(span[1] - span[0], spany[1] - spany[0]), cx = (span[0] + span[1]) / 2, cy = (spany[0] + spany[1]) / 2;
        var w1 = st.W2[0], w2 = st.W2[1], cut = 0.5 - st.b2;
        if (Math.abs(w1) + Math.abs(w2) > 1e-9) {
          var poly = halfPlane(w1, w2, cut, 1, span, spany);
          tr.push({ type: "scatter", mode: "lines", x: poly.map(function (p) { return p[0]; }).concat([poly[0][0]]), y: poly.map(function (p) { return p[1]; }).concat([poly[0][1]]), fill: "toself", fillcolor: rgba(P.coral, 0.12), line: { width: 0 }, hoverinfo: "skip" });
          var xs = [span[0], span[1]], ys = Math.abs(w2) > 1e-9 ? xs.map(function (x) { return (cut - w1 * x) / w2; }) : null;
          if (ys) tr.push({ type: "scatter", mode: "lines", x: xs, y: ys, line: { color: P.ink, width: 2.2, dash: "dash" }, hoverinfo: "skip" });
          else tr.push({ type: "scatter", mode: "lines", x: [cut / w1, cut / w1], y: spany, line: { color: P.ink, width: 2.2, dash: "dash" }, hoverinfo: "skip" });
        }
        var names = ["(0,0)", "(1,0)", "(0,1)", "(1,1)"], lab = names.slice();
        corners.forEach(function (c, i) { for (var j = 0; j < i; j++) if (Math.abs(px[i] - px[j]) < 1e-9 && Math.abs(py[i] - py[j]) < 1e-9) { lab[j] = lab[j] + " and " + names[i]; lab[i] = ""; } });
        tr.push({ type: "scatter", mode: "markers+text", x: px, y: py, text: lab, textposition: ["bottom right", "top center", "top center", "top left"], textfont: { size: 12, color: P.ink2 },
          marker: { size: 12, color: truth.map(function (t) { return t ? P.coral : P.navy; }), symbol: truth.map(function (t) { return t ? "diamond" : "circle"; }), line: { color: P.paper, width: 1.5 } }, hoverinfo: "skip" });
        tr.push({ type: "scatter", mode: "markers", x: [o.h[0]], y: [o.h[1]], marker: { size: 11, symbol: "star", color: P.teal, line: { color: P.ink, width: 1.6 } }, hoverinfo: "skip" });
        return tr;
      }, function () { return { margin: { l: 44, r: 10, t: 8, b: 42 }, dragmode: false, xaxis: { range: span, title: { text: "hidden unit 1" }, zeroline: true }, yaxis: { range: spany, title: { text: "hidden unit 2" }, zeroline: true } }; });
    }
    fillInputs(); draw();
    Kit.onTheme(drawMap);
    window.addEventListener("resize", Kit.debounce(draw, 250));
  }

  // ======================================================================
  // 4. The network playground
  // ======================================================================
  var FEATS = {
    xy: { n: 2, f: function (x, y) { return [x, y]; }, names: ["x", "y"], mid: [0, 0], R: [1.1, 1.1] },
    sq: { n: 5, f: function (x, y) { return [x, y, x * x, y * y, x * y]; }, names: ["x", "y", "x²", "y²", "xy"], mid: [0, 0, 0.6, 0.6, 0], R: [1.1, 1.1, 0.6, 0.6, 0.9] },
    sin: { n: 4, f: function (x, y) { return [x, y, Math.sin(3 * x), Math.sin(3 * y)]; }, names: ["x", "y", "sin 3x", "sin 3y"], mid: [0, 0, 0, 0], R: [1.1, 1.1, 1, 1] }
  };
  var EXT = 1.1, G = 40;

  function playFig() {
    var st = { kind: "circles", noise: 0.05, feat: "xy", nh: 1, units: [4, 3], act: "tanh", lr: 0.1, speed: 5, seed: 1, dseed: 7, running: false, epoch: 0, hist: [], loss: Math.LN2, acc: 0.5, hid: [0, 1] };
    var data = [], net = null, rngT = null, raf = 0, lastPlot = 0, visible = true;
    var unitVals = [], outVals = null, nodes = null, edges = null, svg = null;
    var big = Kit.$("pl-map"), cvSize = 320;

    // ----- controls -----
    Kit.chips(Kit.$("pl-kind"), KINDS.map(function (k) { return KIND_NAME[k]; }), KINDS.indexOf(st.kind), function (i) { st.kind = KINDS[i]; newData(); resetNet(); });
    Kit.chips(Kit.$("pl-feat"), ["x and y", "x, y, x², y², xy", "x, y, sin 3x, sin 3y"], 0, function (i) { st.feat = ["xy", "sq", "sin"][i]; resetNet(); });
    Kit.chips(Kit.$("pl-layers"), ["No hidden layer", "One hidden layer", "Two hidden layers"], st.nh, function (i) { st.nh = i; syncUnitSliders(); resetNet(); });
    var C = Kit.$("pl-controls");
    var sNoise = Kit.slider(C, { label: "noise in the points", min: 0, max: 0.3, step: 0.01, value: st.noise, digits: 2 }, function (v) { st.noise = v; newData(); resetNet(); });
    var sU1 = Kit.slider(C, { label: "units in hidden layer 1", min: 1, max: 8, step: 1, value: st.units[0], digits: 0 }, function (v) { st.units[0] = v; resetNet(); });
    var sU2 = Kit.slider(C, { label: "units in hidden layer 2", min: 1, max: 8, step: 1, value: st.units[1], digits: 0 }, function (v) { st.units[1] = v; resetNet(); });
    Kit.select(C, "hidden activation", [{ value: "tanh", label: "tanh" }, { value: "relu", label: "ReLU" }, { value: "sigmoid", label: "sigmoid" }, { value: "none", label: "none (linear)" }], st.act, function (v) { st.act = v; resetNet(); });
    Kit.select(C, "learning rate η", [{ value: "0.01", label: "0.01" }, { value: "0.03", label: "0.03" }, { value: "0.1", label: "0.1" }, { value: "0.3", label: "0.3" }, { value: "1", label: "1" }], String(st.lr), function (v) { st.lr = parseFloat(v); });
    Kit.select(C, "speed", [{ value: "1", label: "slow: 1 pass per frame" }, { value: "5", label: "medium: 5 passes per frame" }, { value: "20", label: "fast: 20 passes per frame" }], String(st.speed), function (v) { st.speed = parseInt(v, 10); });
    function syncUnitSliders() { sU1.input.closest("label").style.display = st.nh >= 1 ? "" : "none"; sU2.input.closest("label").style.display = st.nh >= 2 ? "" : "none"; }
    var bTrain = Kit.button(Kit.$("pl-bar"), "Train", function () { if (st.running) stop(); else start(); });
    Kit.button(Kit.$("pl-bar"), "One pass", function () { stop(); runEpochs(1); draw(true); });
    Kit.button(Kit.$("pl-bar"), "Restart with new weights", function () { st.seed++; resetNet(); });
    Kit.button(Kit.$("pl-bar"), "New points", function () { st.dseed++; newData(); resetNet(); });
    var selI = Kit.select(Kit.$("pl-hidctl"), "horizontal axis", [{ value: "0", label: "unit 1" }], "0", function (v) { st.hid[0] = parseInt(v, 10); draw(true); });
    var selJ = Kit.select(Kit.$("pl-hidctl"), "vertical axis", [{ value: "1", label: "unit 2" }], "1", function (v) { st.hid[1] = parseInt(v, 10); draw(true); });
    syncUnitSliders();

    // ----- data and network -----
    function newData() { data = makeData(st.kind, 200, st.noise, st.dseed); }
    function featOf(x, y) { return FEATS[st.feat].f(x, y); }
    function sizes() { var s = [FEATS[st.feat].n], l; for (l = 0; l < st.nh; l++) s.push(st.units[l]); s.push(1); return s; }
    function resetNet() {
      stop(); net = new Net(sizes(), st.act, st.seed); rngT = Kit.rng(st.seed * 131 + 5);
      data.forEach(function (p) { p.f = featOf(p.x, p.y); });
      st.epoch = 0; st.hist = []; evalNow(); st.hist.push(st.loss);
      var lastN = st.nh > 0 ? st.units[st.nh - 1] : 0;
      st.hid = [0, Math.min(1, Math.max(0, lastN - 1))];
      [selI, selJ].forEach(function (sel, k) {
        sel.innerHTML = ""; for (var u = 0; u < lastN; u++) { var o = document.createElement("option"); o.value = u; o.textContent = "unit " + (u + 1); sel.appendChild(o); }
        sel.value = String(st.hid[k]);
      });
      buildDiagram(); draw(true);
    }
    function evalNow() {
      var L = net.W.length, ok = 0, loss = 0;
      data.forEach(function (p) { var q = net.forward(p.f)[L][0]; q = clamp(q, 1e-9, 1 - 1e-9); loss -= p.c ? Math.log(q) : Math.log(1 - q); if ((q >= 0.5 ? 1 : 0) === p.c) ok++; });
      st.loss = loss / data.length; st.acc = ok / data.length;
    }
    function runEpochs(n) {
      var e, i, j, k, idx = data.map(function (_, q) { return q; });
      for (e = 0; e < n; e++) {
        for (i = idx.length - 1; i > 0; i--) { j = Math.floor(rngT() * (i + 1)); k = idx[i]; idx[i] = idx[j]; idx[j] = k; }
        for (i = 0; i < idx.length; i += 10) net.step(idx.slice(i, i + 10).map(function (q) { return { x: data[q].f, c: data[q].c }; }), st.lr);
        st.epoch++; evalNow(); st.hist.push(st.loss);
      }
    }
    function start() { st.running = true; bTrain.textContent = "Pause"; loop(); }
    function stop() { st.running = false; cancelAnimationFrame(raf); if (bTrain) bTrain.textContent = "Train"; }
    function loop() {
      if (!st.running) return;
      runEpochs(st.speed); draw(false);
      if (st.epoch >= 6000) { stop(); return; }
      raf = requestAnimationFrame(loop);
    }
    if ("IntersectionObserver" in window) new IntersectionObserver(function (es) { visible = es[0].isIntersecting; if (!visible && st.running) stop(); }, { threshold: 0.05 }).observe(Kit.$("pl"));

    // ----- the diagram: nodes are small canvases, edges are lines -----
    function buildDiagram() {
      var host = Kit.$("pl-net"), W = Math.max(260, host.clientWidth), ns = W < 420 ? 30 : 38, gap = 7, cols = st.nh + 2, sz = sizes(), maxN = Math.max.apply(null, sz), top = 20;
      var H = top + maxN * ns + (maxN - 1) * gap + 4;
      host.innerHTML = ""; host.style.height = H + "px";
      svg = Kit.el("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, "aria-hidden": "true" }, host); svg.style.position = "absolute"; svg.style.left = "0"; svg.style.top = "0";
      nodes = []; edges = [];
      var xs = [], c, u;
      for (c = 0; c < cols; c++) xs.push(cols === 1 ? W / 2 : ns / 2 + c * (W - ns) / (cols - 1));
      var ys = sz.map(function (n) { var tot = n * ns + (n - 1) * gap, y0 = top + (maxN * ns + (maxN - 1) * gap - tot) / 2; return lin(0, 1, Math.max(n, 2)).slice(0, n).map(function (_, i) { return y0 + i * (ns + gap) + ns / 2; }); });
      sz.forEach(function (n, col) {
        var t = Kit.html("div", { class: "pl-ct" }, host); t.style.width = "76px"; t.style.left = clamp(xs[col] - 38, 0, W - 76) + "px";
        t.textContent = col === 0 ? "inputs" : (col === sz.length - 1 ? "output" : "hidden " + col);
        nodes.push([]);
        for (u = 0; u < n; u++) {
          var cv = Kit.html("canvas", { class: "pl-node", "aria-hidden": "true" }, host); sizeCanvas(cv, ns, ns); cv.style.left = (xs[col] - ns / 2) + "px"; cv.style.top = (ys[col][u] - ns / 2) + "px";
          nodes[col].push({ cv: cv, x: xs[col], y: ys[col][u] });
          if (col === 0) { var lab = Kit.html("span", { class: "pl-nl" }, host, FEATS[st.feat].names[u]); lab.style.left = (xs[col] - ns / 2 - 2) + "px"; lab.style.top = (ys[col][u] - 7) + "px"; lab.style.transform = "translateX(-100%)"; }
        }
      });
      for (c = 1; c < sz.length; c++) {
        edges.push([]);
        for (u = 0; u < sz[c]; u++) { edges[c - 1].push([]); for (var i = 0; i < sz[c - 1]; i++) edges[c - 1][u].push(Kit.el("line", { x1: nodes[c - 1][i].x + ns / 2, y1: nodes[c - 1][i].y, x2: nodes[c][u].x - ns / 2, y2: nodes[c][u].y }, svg)); }
      }
      // unit value grids
      unitVals = []; for (c = 1; c < sz.length; c++) { unitVals.push([]); for (u = 0; u < sz[c]; u++) unitVals[c - 1].push(new Float32Array(G * G)); }
      outVals = unitVals[unitVals.length - 1][0];
      st.ns = ns;
    }
    function inputMaps(P) {
      var F = FEATS[st.feat], vals = new Float32Array(G * G), u, i, j;
      for (u = 0; u < F.n; u++) {
        for (j = 0; j < G; j++) for (i = 0; i < G; i++) vals[j * G + i] = F.f(-EXT + 2 * EXT * (i + 0.5) / G, EXT - 2 * EXT * (j + 0.5) / G)[u];
        paintGrid(nodes[0][u].cv, vals, G, F.mid[u], F.R[u], P);
        frame(nodes[0][u].cv, P);
      }
    }
    function frame(cv, P) { var ctx = cv.getContext("2d"); ctx.strokeStyle = P.muted; ctx.lineWidth = 1; ctx.strokeRect(0.5, 0.5, cv.width - 1, cv.height - 1); }

    // ----- drawing -----
    function computeGrids() {
      var L = net.W.length, i, j, a, l, u, k = 0;
      for (j = 0; j < G; j++) for (i = 0; i < G; i++) {
        a = net.forward(featOf(-EXT + 2 * EXT * (i + 0.5) / G, EXT - 2 * EXT * (j + 0.5) / G)); k = j * G + i;
        for (l = 1; l <= L; l++) for (u = 0; u < a[l].length; u++) unitVals[l - 1][u][k] = a[l][u];
      }
    }
    function draw(force) {
      if (!net) return;
      var P = Kit.palette(), L = net.W.length, l, u, i, ctx, d, now = performance.now();
      if (!visible && !force) return;
      computeGrids();
      var A = ACT[st.act];
      for (l = 1; l <= L; l++) for (u = 0; u < net.sizes[l]; u++) {
        var isOut = l === L;
        paintGrid(nodes[l][u].cv, unitVals[l - 1][u], G, isOut ? 0.5 : A.mid, isOut ? 0.5 : A.R, P); frame(nodes[l][u].cv, P);
      }
      if (force || !st.inputsPainted || st.pal !== P.navy) { inputMaps(P); st.inputsPainted = true; st.pal = P.navy; }
      // edges
      var maxW = 0; net.W.forEach(function (Wl) { Wl.forEach(function (r) { r.forEach(function (w) { maxW = Math.max(maxW, Math.abs(w)); }); }); });
      maxW = Math.max(maxW, 1);
      for (l = 0; l < L; l++) for (u = 0; u < net.W[l].length; u++) for (i = 0; i < net.W[l][u].length; i++) {
        var w = net.W[l][u][i], ln = edges[l][u][i];
        ln.setAttribute("stroke", w >= 0 ? P.navy : P.coral); ln.setAttribute("stroke-width", (0.4 + 3.8 * Math.min(1, Math.abs(w) / maxW)).toFixed(2));
        ln.setAttribute("stroke-opacity", "0.7"); ln.setAttribute("stroke-dasharray", w >= 0 ? "" : "5 3");
      }
      // big map
      d = sizeCanvas(big, cvSize, cvSize); ctx = big.getContext("2d");
      paintGrid(big, outVals, G, 0.5, 0.5, P);
      function px(v) { return (v + EXT) / (2 * EXT) * big.width; } function py(v) { return (EXT - v) / (2 * EXT) * big.height; }
      data.forEach(function (p) {
        ctx.beginPath();
        if (p.c) { var s = 5 * d; ctx.moveTo(px(p.x), py(p.y) - s); ctx.lineTo(px(p.x) + s, py(p.y)); ctx.lineTo(px(p.x), py(p.y) + s); ctx.lineTo(px(p.x) - s, py(p.y)); ctx.closePath(); ctx.fillStyle = P.coral; }
        else { ctx.arc(px(p.x), py(p.y), 4.2 * d, 0, 6.2832); ctx.fillStyle = P.navy; }
        ctx.fill(); ctx.lineWidth = 1.5 * d; ctx.strokeStyle = P.ink; ctx.stroke();
      });
      ctx.strokeStyle = P.muted; ctx.lineWidth = d; ctx.strokeRect(0.5, 0.5, big.width - 1, big.height - 1);
      // readouts
      Kit.setText("pl-epoch", String(st.epoch)); Kit.setText("pl-loss-now", Kit.fmt(st.loss, 3)); Kit.setText("pl-acc", Kit.pct(st.acc, 1));
      var sz = net.sizes, terms = [], total = 0;
      for (l = 1; l < sz.length; l++) { terms.push("(" + sz[l - 1] + " × " + sz[l] + " + " + sz[l] + ")"); total += (sz[l - 1] + 1) * sz[l]; }
      Kit.$("pl-eqn").innerHTML = "<span class='t c-sand'><b>" + total + "</b><i>weights and biases</i></span><span class='op'>=</span>" + terms.map(function (t, k) { return (k ? "<span class='op'>+</span>" : "") + "<span class='t'><b>" + t + "</b><i>" + (k === terms.length - 1 ? "output unit" : "layer " + (k + 1)) + "</i></span>"; }).join("");
      var msg;
      if (st.epoch === 0) msg = "Press Train. The coloured map is the network's answer at all points, from navy (class 0) to coral (class 1).";
      else if (st.nh === 0) msg = "No hidden layer: this is the single neuron of B1, and it can draw one straight line.";
      else if (st.act === "none") msg = "With the activation set to none, the layers collapse into one neuron (section 3). Expect one straight line at best.";
      else if (st.acc >= 0.99) msg = "Pass " + st.epoch + ": " + Kit.pct(st.acc, 1) + " of the 200 points are right, loss " + Kit.fmt(st.loss, 3) + ". Try the spiral, fewer units or more noise.";
      else msg = "Pass " + st.epoch + ": " + Kit.pct(st.acc, 1) + " right, loss " + Kit.fmt(st.loss, 3) + ". Training pauses when the figure leaves the screen.";
      Kit.setText("pl-status", msg);
      if (force || now - lastPlot > 160) { lastPlot = now; plots(P); }
    }
    function plots() {
      var hs = st.hist, stride = Math.max(1, Math.ceil(hs.length / 500)), xs = [], ys = [], i;
      for (i = 0; i < hs.length; i += stride) { xs.push(i); ys.push(hs[i]); }
      if ((hs.length - 1) % stride) { xs.push(hs.length - 1); ys.push(hs[hs.length - 1]); }
      Kit.plot("pl-loss", function (P) { return [{ type: "scatter", mode: "lines", x: xs, y: ys, line: { color: P.navy, width: 2.6 }, hoverinfo: "skip" }, { type: "scatter", mode: "lines", x: [0, Math.max(10, hs.length - 1)], y: [Math.LN2, Math.LN2], line: { color: P.muted, width: 1.2, dash: "dot" }, hoverinfo: "skip" }]; },
        function (P) { return { margin: { l: 48, r: 10, t: 8, b: 52 }, dragmode: false, xaxis: { title: { text: "passes through the data", standoff: 8 } , range: [0, Math.max(10, hs.length - 1)] }, yaxis: { title: { text: "cross-entropy loss", standoff: 4 }, range: [0, 0.85] },
          annotations: [{ x: Math.max(10, hs.length - 1), y: Math.LN2, xref: "x", yref: "y", text: "ln 2 = 0.69: guessing", showarrow: false, font: { size: 12, color: P.muted }, xanchor: "right", yanchor: "bottom" }] }; });
      var nHid = st.nh > 0 ? st.units[st.nh - 1] : 0;
      if (nHid === 0) {
        Kit.plot("pl-hid", function () { return [{ type: "scatter", x: [0], y: [0], mode: "markers", marker: { size: 1, opacity: 0 }, hoverinfo: "skip" }]; },
          function (P) { return { margin: { l: 10, r: 10, t: 8, b: 10 }, dragmode: false, xaxis: { visible: false }, yaxis: { visible: false }, annotations: [{ x: 0, y: 0, xref: "x", yref: "y", showarrow: false, text: "There is no hidden layer, so the inputs<br>feed the output unit directly.", font: { size: 14, color: P.ink2 } }] }; });
        return;
      }
      var L = net.W.length, i0 = st.hid[0], j0 = nHid > 1 ? st.hid[1] : -1;
      var rr = Kit.rng(3), pts = data.map(function (p) { var a = net.forward(p.f)[L - 1]; return { a: a[i0], b: j0 >= 0 ? a[j0] : (rr() - 0.5) * 0.6, c: p.c }; });
      Kit.plot("pl-hid", function (P) {
        function cl(c) { var s = pts.filter(function (q) { return q.c === c; }); return { x: s.map(function (q) { return q.a; }), y: s.map(function (q) { return q.b; }) }; }
        var c0 = cl(0), c1 = cl(1);
        return [{ type: "scatter", mode: "markers", x: c0.x, y: c0.y, marker: { symbol: "circle", size: 6, color: P.navy, opacity: 0.8 }, hoverinfo: "skip" }, { type: "scatter", mode: "markers", x: c1.x, y: c1.y, marker: { symbol: "diamond", size: 7, color: P.coral, opacity: 0.85 }, hoverinfo: "skip" }];
      }, function () { return { margin: { l: 48, r: 10, t: 8, b: 52 }, dragmode: false, xaxis: { title: { text: "output of hidden unit " + (i0 + 1) + (st.nh === 2 ? " (layer 2)" : ""), standoff: 8 } }, yaxis: j0 >= 0 ? { title: { text: "output of hidden unit " + (j0 + 1), standoff: 4 } } : { showticklabels: false, title: { text: "(spread for clarity)" }, range: [-0.5, 0.5] } }; });
    }
    Kit.onTheme(function () { st.inputsPainted = false; draw(true); });
    newData(); resetNet();
    window.addEventListener("resize", Kit.debounce(function () { buildDiagram(); st.inputsPainted = false; draw(true); }, 250));
  }

})();
