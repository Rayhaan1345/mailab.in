// Slopes and gradients (chapter 0.6): six figures.
//   1 slopeFig      drag a point along a curve: tangent, nudge triangle and the slope curve (D3)
//   2 activationFig sigmoid, tanh and ReLU with their slopes (Plotly)
//   3 partialFig    contour map with two frozen-variable slices (Plotly)
//   4 gradientFig   rotatable 3D surface beside a contour map with gradient arrows (Plotly)
//   5 chainFig      linked sliders: a nudge travels down a chain and the slopes multiply (D3)
//   6 curveFig      a sharp valley and a flat valley (Plotly)
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    slopeFig(); activationFig(); partialFig(); gradientFig(); chainFig(); curveFig();
  });

  // ---------- helpers ----------
  function sig(z) { return 1 / (1 + Math.exp(-z)); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function sgn(v) { return v < 0 ? "−" : "+"; }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function mix(a, b, t) { var A = hexRGB(a), B = hexRGB(b); return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }).join(",") + ")"; }
  function rgba(c, a) { var C = hexRGB(c); return "rgba(" + C[0] + "," + C[1] + "," + C[2] + "," + a + ")"; }
  function styleAxis(g, P) {
    g.selectAll("path,line").attr("stroke", P.muted);
    g.selectAll("text").attr("fill", P.muted).attr("font-size", 12.5);
  }

  // ---------- 1. a point on a curve, with its tangent ----------
  function slopeFig() {
    var F = [
      { name: "Shark weight", f: function (x) { return 4e-6 * x * x * x; }, d: function (x) { return 1.2e-5 * x * x; },
        x0: 30, x1: 110, xs: 80, hs: 8, hmax: 30, xd: 1, fd: 3, dd: 4, xlab: "length L, cm", ylab: "weight W, kg", dlab: "slope, kg per cm", unit: "Units: kg per cm.", ymin: 0 },
      { name: "x squared", f: function (x) { return x * x; }, d: function (x) { return 2 * x; },
        x0: -3, x1: 3, xs: 1, hs: 0.5, hmax: 3, xd: 2, fd: 2, dd: 3, xlab: "x", ylab: "f(x) = x²", dlab: "slope 2x", unit: "", ymin: 0 },
      { name: "Sigmoid", f: sig, d: function (x) { var s = sig(x); return s * (1 - s); },
        x0: -6, x1: 6, xs: -1, hs: 0.5, hmax: 4, xd: 2, fd: 3, dd: 4, xlab: "z", ylab: "σ(z)", dlab: "slope σ′(z)", unit: "" },
      { name: "ReLU", f: function (x) { return Math.max(0, x); }, d: function (x) { return x > 0 ? 1 : 0; },
        x0: -3, x1: 3, xs: 1.5, hs: 0.5, hmax: 3, xd: 2, fd: 2, dd: 3, xlab: "z", ylab: "ReLU(z)", dlab: "slope", unit: "", breaks: [0] },
      { name: "Logarithm", f: Math.log, d: function (x) { return 1 / x; },
        x0: 0.2, x1: 4, xs: 1.2, hs: 0.3, hmax: 1.5, xd: 2, fd: 3, dd: 3, xlab: "x", ylab: "ln x", dlab: "slope 1/x", unit: "" }
    ];
    F.forEach(function (c) {
      var xs = lin(c.x0, c.x1, 241), fmn = 1e9, fmx = -1e9, dmn = 0, dmx = 0;
      xs.forEach(function (x) { var y = c.f(x), d = c.d(x); fmn = Math.min(fmn, y); fmx = Math.max(fmx, y); dmn = Math.min(dmn, d); dmx = Math.max(dmx, d); });
      if (c.ymin != null) fmn = Math.min(fmn, c.ymin);
      var pf = (fmx - fmn) * 0.07, pd = (dmx - dmn) * 0.14;
      c.fmin = fmn - pf; c.fmax = fmx + pf; c.dmin = dmn - pd; c.dmax = dmx + pd;
      if (c.ymin === 0) c.fmin = 0;
    });
    var cur = F[0], x = cur.xs, h = cur.hs, W = 600;
    var node = Kit.$("sl-svg"), svg = d3.select(node);
    var xsl = Kit.slider(Kit.$("sl-controls"), { label: "Position", min: cur.x0, max: cur.x1, step: 0.05, value: x, show: function (v) { return Kit.fmt(v, cur.xd); } }, function (v) { x = v; draw(); });
    var hsl = Kit.slider(Kit.$("sl-controls"), { label: "Nudge h", min: 0.05, max: cur.hmax, step: 0.05, value: h, show: function (v) { return Kit.fmt(v, cur.hmax >= 10 ? 1 : 2); } }, function (v) { h = v; draw(); });
    Kit.chips(Kit.$("sl-fns"), F.map(function (c) { return c.name; }), 0, function (i) {
      cur = F[i]; x = cur.xs; h = cur.hs;
      xsl.input.min = cur.x0; xsl.input.max = cur.x1; xsl.input.step = (cur.x1 - cur.x0) / 400; xsl.set(x);
      hsl.input.max = cur.hmax; hsl.input.step = cur.hmax / 200; hsl.input.min = cur.hmax / 200; hsl.set(h);
      Kit.setText("sl-unit", cur.unit);
      draw();
    });
    Kit.setText("sl-unit", cur.unit);

    var dragging = false;
    function fromEvent(e) {
      var r = node.getBoundingClientRect(), px = (e.clientX - r.left) * (W / r.width);
      var sx = d3.scaleLinear().domain([cur.x0, cur.x1]).range([56, W - 16]);
      x = clamp(sx.invert(px), cur.x0, cur.x1); xsl.set(x); draw();
    }
    node.addEventListener("pointerdown", function (e) { dragging = true; try { node.setPointerCapture(e.pointerId); } catch (err) {} fromEvent(e); });
    node.addEventListener("pointermove", function (e) { if (dragging) fromEvent(e); });
    node.addEventListener("pointerup", function () { dragging = false; });
    node.addEventListener("pointercancel", function () { dragging = false; });

    function draw() {
      var P = Kit.palette(); W = Kit.width(node);
      var H = W < 480 ? 410 : 450, ml = 56, mr = 16, t0 = 14, t1 = Math.round(H * 0.55), b0 = t1 + 30, b1 = H - 42;
      var sx = d3.scaleLinear().domain([cur.x0, cur.x1]).range([ml, W - mr]);
      var sy = d3.scaleLinear().domain([cur.fmin, cur.fmax]).range([t1, t0]);
      var sd = d3.scaleLinear().domain([cur.dmin, cur.dmax]).range([b1, b0]);
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      var fx = cur.f(x), d = cur.d(x), fx2 = cur.f(x + h);
      // grid and axes
      var gg = svg.append("g");
      sy.ticks(5).forEach(function (v) { gg.append("line").attr("x1", ml).attr("x2", W - mr).attr("y1", sy(v)).attr("y2", sy(v)).attr("stroke", P.rule); });
      sd.ticks(3).forEach(function (v) { gg.append("line").attr("x1", ml).attr("x2", W - mr).attr("y1", sd(v)).attr("y2", sd(v)).attr("stroke", P.rule); });
      styleAxis(svg.append("g").attr("transform", "translate(" + ml + ",0)").call(d3.axisLeft(sy).ticks(5).tickSize(4)), P);
      styleAxis(svg.append("g").attr("transform", "translate(" + ml + ",0)").call(d3.axisLeft(sd).ticks(3).tickSize(4)), P);
      styleAxis(svg.append("g").attr("transform", "translate(0," + t1 + ")").call(d3.axisBottom(sx).ticks(6).tickSize(4).tickFormat(function () { return ""; })), P);
      styleAxis(svg.append("g").attr("transform", "translate(0," + b1 + ")").call(d3.axisBottom(sx).ticks(6).tickSize(4)), P);
      svg.append("text").attr("x", ml + 6).attr("y", t0 + 12).attr("font-size", 13.5).attr("fill", P.ink2).text(cur.ylab);
      svg.append("text").attr("x", ml + 6).attr("y", b0 + 12).attr("font-size", 13.5).attr("fill", P.ink2).text(cur.dlab);
      svg.append("text").attr("x", (ml + W - mr) / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.ink2).text(cur.xlab);
      svg.append("defs").append("clipPath").attr("id", "sl-clip").append("rect").attr("x", ml).attr("y", t0 - 4).attr("width", W - ml - mr).attr("height", t1 - t0 + 8);
      // curve and slope curve
      var xs = lin(cur.x0, cur.x1, 241);
      (cur.breaks || []).forEach(function (b) { xs.push(b - 1e-6, b + 1e-6); }); xs.sort(function (a, b) { return a - b; });
      var lineF = d3.line().x(function (v) { return sx(v); }).y(function (v) { return sy(cur.f(v)); });
      var lineD = d3.line().x(function (v) { return sx(v); }).y(function (v) { return sd(cur.d(v)); });
      var clip = svg.append("g").attr("clip-path", "url(#sl-clip)");
      clip.append("path").attr("d", lineF(xs)).attr("fill", "none").attr("stroke", P.navy).attr("stroke-width", 2.8);
      svg.append("path").attr("d", lineD(xs)).attr("fill", "none").attr("stroke", P.teal).attr("stroke-width", 2.6);
      // tangent
      var span = (cur.x1 - cur.x0) * 0.2, ta = Math.max(cur.x0, x - span), tb = Math.min(cur.x1, x + span);
      clip.append("line").attr("x1", sx(ta)).attr("y1", sy(fx + d * (ta - x))).attr("x2", sx(tb)).attr("y2", sy(fx + d * (tb - x)))
        .attr("stroke", P.teal).attr("stroke-width", 2.4);
      // nudge triangle
      clip.append("polygon").attr("points", [[x, fx], [x + h, fx], [x + h, fx2]].map(function (p) { return sx(p[0]) + "," + sy(p[1]); }).join(" ")).attr("fill", P.mark);
      clip.append("line").attr("x1", sx(x)).attr("y1", sy(fx)).attr("x2", sx(x + h)).attr("y2", sy(fx2)).attr("stroke", P.ink2).attr("stroke-width", 1.4).attr("stroke-dasharray", "5 4");
      clip.append("line").attr("x1", sx(x)).attr("y1", sy(fx)).attr("x2", sx(x + h)).attr("y2", sy(fx)).attr("stroke", P.navy).attr("stroke-width", 4);
      clip.append("line").attr("x1", sx(x + h)).attr("y1", sy(fx)).attr("x2", sx(x + h)).attr("y2", sy(fx2)).attr("stroke", P.coral).attr("stroke-width", 4);
      // labels on parts
      var runPx = Math.abs(sx(x + h) - sx(x)), risePx = Math.abs(sy(fx2) - sy(fx));
      if (runPx > 26) svg.append("text").attr("x", (sx(x) + sx(x + h)) / 2).attr("y", sy(fx) + (fx2 >= fx ? 17 : -8)).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", P.navy).text("run h");
      if (risePx > 22 && sx(x + h) < W - mr - 44) svg.append("text").attr("x", sx(x + h) + 8).attr("y", (sy(fx) + sy(fx2)) / 2 + 4).attr("font-size", 13).attr("fill", P.coral).text("rise");
      var xl = x - span * 0.7 > cur.x0 + (cur.x1 - cur.x0) * 0.05 ? x - span * 0.7 : Math.min(cur.x1 - span * 0.15, x + Math.max(span * 0.9, h * 1.4)), yt = fx + d * (xl - x);
      var below = cur.f(xl) >= yt;     // curve above the tangent: put the label under the tangent
      if (below && sy(yt) + 20 > t1 - 14) below = false;
      var yl = clamp(sy(yt) + (below ? 20 : -10), t0 + 26, t1 - 8);
      svg.append("text").attr("x", sx(xl)).attr("y", yl).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", P.teal).attr("font-weight", 600).text("tangent");
      // dots and the guide between the panels
      svg.append("line").attr("x1", sx(x)).attr("x2", sx(x)).attr("y1", sy(fx)).attr("y2", sd(d)).attr("stroke", P.muted).attr("stroke-dasharray", "2 4");
      svg.append("circle").attr("cx", sx(x)).attr("cy", sy(fx)).attr("r", 8).attr("fill", P.coral).attr("stroke", P.bg).attr("stroke-width", 2.5).style("cursor", "grab");
      svg.append("circle").attr("cx", sx(x)).attr("cy", sd(d)).attr("r", 6).attr("fill", P.coral).attr("stroke", P.bg).attr("stroke-width", 2);
      var right = sx(x) > W * 0.62;
      svg.append("text").attr("x", sx(x) + (right ? -12 : 12)).attr("y", sd(d) - 10 < b0 + 18 ? sd(d) + 22 : sd(d) - 10).attr("text-anchor", right ? "end" : "start").attr("font-size", 13).attr("fill", P.ink).attr("font-weight", 600).text("slope " + Kit.fmt(d, cur.dd));
      // readouts
      Kit.setText("sl-x", Kit.fmt(x, cur.xd)); Kit.setText("sl-fx", Kit.fmt(fx, cur.fd)); Kit.setText("sl-d", Kit.fmt(d, cur.dd));
      Kit.setText("sl-e-rise", Kit.fmt(fx2 - fx, cur.dd)); Kit.setText("sl-e-run", Kit.fmt(h, h < 1 ? 2 : 1));
      Kit.setText("sl-e-slope", Kit.fmt((fx2 - fx) / h, cur.dd)); Kit.setText("sl-e-rule", Kit.fmt(d, cur.dd));
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- 2. activation functions and their slopes ----------
  function activationFig() {
    var A = [
      { name: "Sigmoid", f: sig, d: function (z) { var s = sig(z); return s * (1 - s); }, y: [-0.15, 1.15], fl: "output σ(z)", dl: "slope σ′(z)", fz: 3.2, dz: 2.9 },
      { name: "tanh", f: Math.tanh, d: function (z) { var t = Math.tanh(z); return 1 - t * t; }, y: [-1.25, 1.25], fl: "output tanh z", dl: "slope 1 − tanh² z", fz: 2.4, dz: -2.2 },
      { name: "ReLU", f: function (z) { return Math.max(0, z); }, d: function (z) { return z > 0 ? 1 : 0; }, y: [-0.6, 6.4], fl: "output ReLU(z)", dl: "slope", fz: 4.6, dz: 3.0 }
    ];
    var cur = A[0];
    var zs = Kit.slider(Kit.$("ac-controls"), { label: "Input z", min: -6, max: 6, step: 0.05, value: 1 }, draw);
    Kit.chips(Kit.$("ac-fns"), A.map(function (a) { return a.name; }), 0, function (i) { cur = A[i]; draw(); });
    function draw() {
      var z = zs.get(), fz = cur.f(z), dz = cur.d(z), xs = lin(-6, 6, 241);
      if (cur.name === "ReLU") { xs.push(-1e-6, 1e-6); xs.sort(function (a, b) { return a - b; }); }
      Kit.plot("ac-plot", function (P) {
        return [
          { x: xs, y: xs.map(cur.f), mode: "lines", line: { color: P.navy, width: 3 }, hoverinfo: "skip" },
          { x: xs, y: xs.map(cur.d), mode: "lines", line: { color: P.coral, width: 3 }, hoverinfo: "skip" },
          { x: [z], y: [fz], mode: "markers", marker: { color: P.navy, size: 11, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" },
          { x: [z], y: [dz], mode: "markers", marker: { color: P.coral, size: 11, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" }
        ];
      }, function (P) {
        return {
          xaxis: { range: [-6, 6], title: { text: "z" } }, yaxis: { range: cur.y, title: { text: "value" } }, dragmode: false,
          shapes: [{ type: "line", x0: z, x1: z, y0: cur.y[0], y1: cur.y[1], line: { color: P.muted, width: 1.2, dash: "dot" } }],
          annotations: [
            { x: cur.fz, y: cur.f(cur.fz), text: cur.fl, showarrow: false, yshift: 16, font: { color: P.navy, size: 14 } },
            { x: cur.dz, y: cur.d(cur.dz), text: cur.dl, showarrow: false, yshift: 18, font: { color: P.coral, size: 14 } }
          ]
        };
      });
      var b = function (cls, v, lab) { return '<span class="t ' + cls + '"><b>' + v + '</b><i>' + lab + '</i></span>'; }, op = function (s) { return '<span class="op">' + s + '</span>'; };
      var html;
      if (cur.name === "Sigmoid") html = b("c-coral", Kit.fmt(dz, 4), "slope σ′(z)") + op("=") + b("c-navy", Kit.fmt(fz, 3), "output σ(z)") + op("&times;") + b("", Kit.fmt(1 - fz, 3), "1 &minus; σ(z)");
      else if (cur.name === "tanh") html = b("c-coral", Kit.fmt(dz, 4), "slope") + op("=") + b("", "1", "one") + op("&minus;") + b("c-navy", Kit.fmt(fz * fz, 3), "tanh&sup2; z");
      else html = b("", Kit.fmt(z, 2), "input z") + op("&rarr;") + b("c-navy", Kit.fmt(fz, 2), "output max(0, z)") + op("and") + b("c-coral", Kit.fmt(dz, 0), z > 0 ? "slope: z is positive" : "slope: z is negative");
      Kit.$("ac-eqn").innerHTML = html;
    }
    draw();
  }

  // ---------- surfaces shared by figures 3 and 4 ----------
  var LIM = 2.5;
  var SURF = {
    bowl: { name: "Bowl", start: [1.8, 1.2], eta: 0.3,
      L: function (a, b) { var u = a - 0.5, v = b + 0.5; return 0.5 * u * u + v * v + 0.6 * u * v; },
      g: function (a, b) { var u = a - 0.5, v = b + 0.5; return [u + 0.6 * v, 2 * v + 0.6 * u]; } },
    valley: { name: "Narrow valley", start: [-2, -0.6], eta: 0.8,
      L: function (a, b) { var p = (a + b) / Math.SQRT2, q = (a - b) / Math.SQRT2; return 0.05 * p * p + q * q; },
      g: function (a, b) { var p = (a + b) / Math.SQRT2, q = (a - b) / Math.SQRT2; return [(0.1 * p + 2 * q) / Math.SQRT2, (0.1 * p - 2 * q) / Math.SQRT2]; } },
    saddle: { name: "Saddle", start: [1.4, 0.15], eta: 0.3,
      L: function (a, b) { return 0.5 * a * a - 0.4 * b * b; },
      g: function (a, b) { return [a, -0.8 * b]; } },
    dips: { name: "Two dips", start: [0.55, 1.1], eta: 0.3,
      L: function (a, b) { return 0.25 * Math.pow(a * a - 1, 2) + 0.5 * b * b + 0.3 * a; },
      g: function (a, b) { return [a * (a * a - 1) + 0.3, b]; } }
  };
  function gridOf(S, n) {
    S.grids = S.grids || {};
    if (S.grids[n]) return S.grids[n];
    var xs = lin(-LIM, LIM, n), z = [], mn = 1e9, mx = -1e9, i, j;
    for (j = 0; j < n; j++) { var row = []; for (i = 0; i < n; i++) { var v = S.L(xs[i], xs[j]); row.push(v); mn = Math.min(mn, v); mx = Math.max(mx, v); } z.push(row); }
    S.grids[n] = { n: n, xs: xs, z: z, min: mn, max: mx };
    return S.grids[n];
  }
  function arrow(x0, y0, x1, y1, color, width, size) {
    return { x: x1, y: y1, ax: x0, ay: y0, xref: "x", yref: "y", axref: "x", ayref: "y", showarrow: true, arrowhead: 2, arrowsize: size || 1.1, arrowwidth: width, arrowcolor: color, text: "" };
  }
  function contourTrace(S, P) {
    var g = gridOf(S, 61);
    return { type: "contour", x: g.xs, y: g.xs, z: g.z, showscale: false, ncontours: 12,
      colorscale: [[0, rgba(P.navy, 0.03)], [1, rgba(P.navy, 0.5)]],
      contours: { coloring: "fill", showlabels: true, labelfont: { size: 11, color: P.muted } }, line: { color: P.muted, width: 0.8 },
      hovertemplate: "w₁ %{x:.2f}, w₂ %{y:.2f}, L %{z:.2f}<extra></extra>" };
  }
  function pointTrace(P, w) {
    return { x: [w[0]], y: [w[1]], mode: "markers", type: "scatter", marker: { color: P.coral, size: 12, line: { color: P.bg, width: 2.5 } }, hoverinfo: "skip" };
  }
  function hasWebGL() {
    try { var c = document.createElement("canvas"); return !!(c.getContext("webgl") || c.getContext("experimental-webgl")); } catch (e) { return false; }
  }
  function onMapClick(id, fn) {
    var d = Kit.$(id);
    if (d && d.on) d.on("plotly_click", function (e) { var p = e.points && e.points[0]; if (p) fn(clamp(p.x, -LIM, LIM), clamp(p.y, -LIM, LIM)); });
  }

  // ---------- 3. partial derivatives as slopes of slices ----------
  function partialFig() {
    var S = SURF.bowl, w = [1.5, 0.5], g = gridOf(S, 61);
    var s1 = Kit.slider(Kit.$("pt-controls"), { label: "w₁", min: -LIM, max: LIM, step: 0.05, value: w[0] }, function (v) { w[0] = v; draw(); });
    var s2 = Kit.slider(Kit.$("pt-controls"), { label: "w₂", min: -LIM, max: LIM, step: 0.05, value: w[1] }, function (v) { w[1] = v; draw(); });
    function draw() {
      var L0 = S.L(w[0], w[1]), gr = S.g(w[0], w[1]), xs = lin(-LIM, LIM, 101);
      Kit.plot("pt-map", function (P) { return [contourTrace(S, P), pointTrace(P, w)]; }, function (P) {
        return { margin: { l: 46, r: 10, t: 10, b: 44 }, dragmode: false,
          xaxis: { range: [-LIM, LIM], title: { text: "w₁" }, constrain: "domain" }, yaxis: { range: [-LIM, LIM], title: { text: "w₂" }, scaleanchor: "x" },
          shapes: [
            { type: "line", x0: -LIM, x1: LIM, y0: w[1], y1: w[1], line: { color: P.navy, width: 2.5 } },
            { type: "line", x0: w[0], x1: w[0], y0: -LIM, y1: LIM, line: { color: P.sand, width: 2.5 } }] };
      });
      function slice(id, k, xlab) {
        var curve = xs.map(function (v) { return k === 0 ? S.L(v, w[1]) : S.L(w[0], v); });
        var c = w[k], t0 = Math.max(-LIM, c - 0.9), t1 = Math.min(LIM, c + 0.9), slope = gr[k];
        Kit.plot(id, function (P) {
          return [
            { x: xs, y: curve, mode: "lines", line: { color: k === 0 ? P.navy : P.sand, width: 3 }, hoverinfo: "skip" },
            { x: [t0, t1], y: [L0 + slope * (t0 - c), L0 + slope * (t1 - c)], mode: "lines", line: { color: P.teal, width: 2.6 }, hoverinfo: "skip" },
            { x: [c], y: [L0], mode: "markers", marker: { color: P.coral, size: 11, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" }];
        }, function (P) {
          return { margin: { l: 46, r: 10, t: 10, b: 52 }, dragmode: false, xaxis: { range: [-LIM, LIM], title: { text: xlab, standoff: 6 } }, yaxis: { range: [0, Math.ceil(g.max)], title: { text: "L" } },
            annotations: [{ xref: "paper", yref: "paper", x: 0.98, y: 0.96, xanchor: "right", yanchor: "top", showarrow: false, align: "right", font: { size: 13, color: P.ink2 },
              text: (k === 0 ? "w₂" : "w₁") + " frozen at " + Kit.fmt(w[1 - k], 2) + "<br>slope " + Kit.fmt(slope, 2) }] };
        });
      }
      slice("pt-s1", 0, "w₁"); slice("pt-s2", 1, "w₂");
      Kit.setText("pt-e1", Kit.fmt(gr[0], 2)); Kit.setText("pt-e2", Kit.fmt(gr[1], 2)); Kit.setText("pt-ep", "(" + Kit.fmt(w[0], 2) + ", " + Kit.fmt(w[1], 2) + ")");
    }
    draw();
    onMapClick("pt-map", function (x, y) { w = [x, y]; s1.set(x); s2.set(y); draw(); });
  }

  // ---------- 4. the gradient: 3D surface beside a contour map ----------
  function gradientFig() {
    var keys = ["bowl", "valley", "saddle", "dips"], S = SURF.bowl;
    var st = { w: S.start.slice(), trail: [], field: false, eta: S.eta };
    var s1 = Kit.slider(Kit.$("gr-controls"), { label: "w₁", min: -LIM, max: LIM, step: 0.05, value: st.w[0] }, function (v) { st.w[0] = v; st.trail = []; msg(); draw(); });
    var s2 = Kit.slider(Kit.$("gr-controls"), { label: "w₂", min: -LIM, max: LIM, step: 0.05, value: st.w[1] }, function (v) { st.w[1] = v; st.trail = []; msg(); draw(); });
    var se = Kit.slider(Kit.$("gr-controls"), { label: "Step size η", min: 0.05, max: 1, step: 0.05, value: st.eta }, function (v) { st.eta = v; });
    Kit.chips(Kit.$("gr-surf"), keys.map(function (k) { return SURF[k].name; }), 0, function (i) { S = SURF[keys[i]]; reset(); });
    Kit.button(Kit.$("gr-btns"), "Step downhill", step);
    Kit.button(Kit.$("gr-btns"), "Reset", reset);
    Kit.chips(Kit.$("gr-btns"), ["Arrow field off", "Arrow field on"], 0, function (i) { st.field = i === 1; draw(); });
    function msg(t) { Kit.setText("gr-status", t || "Drag the 3D view to rotate it. Click the contour map to move the point. Arrow lengths are for display; the box below gives the true steepness."); }
    function reset() { st.w = S.start.slice(); st.trail = []; st.eta = S.eta; s1.set(st.w[0]); s2.set(st.w[1]); se.set(st.eta); msg(); draw(); }
    function step() {
      var g = S.g(st.w[0], st.w[1]), nw = [clamp(st.w[0] - st.eta * g[0], -LIM, LIM), clamp(st.w[1] - st.eta * g[1], -LIM, LIM)];
      st.trail.push(st.w.slice()); st.w = nw; s1.set(nw[0]); s2.set(nw[1]);
      var g2 = S.g(nw[0], nw[1]), L = S.L(nw[0], nw[1]), edge = Math.abs(nw[0]) >= LIM || Math.abs(nw[1]) >= LIM;
      msg(edge ? "Step " + st.trail.length + ": the point reached the edge of the map. On this surface the loss keeps falling beyond it." :
        "Step " + st.trail.length + ": loss " + Kit.fmt(L, 3) + (Math.hypot(g2[0], g2[1]) < 0.05 ? ". The gradient is nearly zero here. Is this the lowest point? Compare with the other surfaces and with a different start." : "."));
      draw();
    }
    function pathAlong(dir) {
      var g = S.g(st.w[0], st.w[1]), n = Math.hypot(g[0], g[1]), pts = { x: [], y: [], z: [] };
      if (n < 1e-6) return pts;
      for (var t = 0; t <= 1.2; t += 0.06) {
        var a = st.w[0] + dir * t * g[0] / n, b = st.w[1] + dir * t * g[1] / n;
        if (Math.abs(a) > LIM || Math.abs(b) > LIM) break;
        pts.x.push(a); pts.y.push(b); pts.z.push(S.L(a, b) + 0.12);
      }
      return pts;
    }
    var gl = hasWebGL();
    function draw() {
      var w = st.w, g = S.g(w[0], w[1]), n = Math.hypot(g[0], g[1]), L0 = S.L(w[0], w[1]), g41 = gridOf(S, 41);
      if (!gl) Kit.$("gr-3d").textContent = "This browser cannot draw 3D graphics here. The contour map on the right shows the same surface from above.";
      else Kit.plot("gr-3d", function (P) {
        var up = pathAlong(1), dn = pathAlong(-1);
        var tr = { x: st.trail.concat([w]).map(function (p) { return p[0]; }), y: st.trail.concat([w]).map(function (p) { return p[1]; }), z: st.trail.concat([w]).map(function (p) { return S.L(p[0], p[1]) + 0.12; }) };
        return [
          { type: "surface", x: g41.xs, y: g41.xs, z: g41.z, showscale: false, hoverinfo: "skip", opacity: 0.93,
            colorscale: [[0, mix(P.paper, P.navy, 0.12)], [1, mix(P.paper, P.navy, 0.95)]], lighting: { ambient: 0.75, diffuse: 0.7, specular: 0.08, roughness: 0.8 } },
          { type: "scatter3d", mode: "lines", x: up.x, y: up.y, z: up.z, line: { color: P.coral, width: 7 }, hoverinfo: "skip" },
          { type: "scatter3d", mode: "lines", x: dn.x, y: dn.y, z: dn.z, line: { color: P.teal, width: 7 }, hoverinfo: "skip" },
          { type: "scatter3d", mode: "lines+markers", x: tr.x, y: tr.y, z: tr.z, line: { color: P.ink2, width: 3 }, marker: { color: P.ink2, size: 3 }, hoverinfo: "skip" },
          { type: "scatter3d", mode: "markers", x: [w[0]], y: [w[1]], z: [L0 + 0.15], marker: { color: P.coral, size: 7, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" }];
      }, function () {
        return { margin: { l: 0, r: 0, t: 0, b: 0 }, uirevision: "t6-3d",
          scene: { xaxis: { title: { text: "w₁" }, range: [-LIM, LIM], nticks: 5 }, yaxis: { title: { text: "w₂" }, range: [-LIM, LIM], nticks: 5 }, zaxis: { title: { text: "L" }, nticks: 5 },
            aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.6 }, camera: { eye: Kit.width(Kit.$("gr-3d")) < 420 ? { x: 1.75, y: -1.85, z: 1.1 } : { x: 1.3, y: -1.4, z: 0.8 } } } };
      });
      Kit.plot("gr-map", function (P) {
        var tr = st.trail.concat([w]);
        return [contourTrace(S, P),
          { x: tr.map(function (p) { return p[0]; }), y: tr.map(function (p) { return p[1]; }), mode: "lines+markers", type: "scatter", line: { color: P.ink2, width: 1.5 }, marker: { color: P.ink2, size: 5 }, hoverinfo: "skip" },
          pointTrace(P, w)];
      }, function (P) {
        var ann = [], i, j;
        if (st.field) {
          var gs = lin(-2.1, 2.1, 9);
          for (i = 0; i < 9; i++) for (j = 0; j < 9; j++) {
            var q = S.g(gs[i], gs[j]), m = Math.hypot(q[0], q[1]); if (m < 1e-6) continue;
            var len = clamp(0.14 * m, 0.12, 0.42);
            ann.push(arrow(gs[i], gs[j], gs[i] + len * q[0] / m, gs[j] + len * q[1] / m, rgba(P.ink2, 0.75), 1.4, 0.8));
          }
        }
        if (n > 0.02) {
          var len2 = clamp(0.3 * n, 0.35, 1.3), ux = g[0] / n, uy = g[1] / n;
          ann.push(arrow(w[0], w[1], w[0] + len2 * ux, w[1] + len2 * uy, P.coral, 3.4, 1.2));
          ann.push(arrow(w[0], w[1], w[0] - len2 * ux, w[1] - len2 * uy, P.teal, 3.4, 1.2));
        }
        return { margin: { l: 46, r: 10, t: 10, b: 44 }, dragmode: false, annotations: ann,
          xaxis: { range: [-LIM, LIM], title: { text: "w₁" }, constrain: "domain" }, yaxis: { range: [-LIM, LIM], title: { text: "w₂" }, scaleanchor: "x" } };
      });
      Kit.setText("gr-e1", Kit.fmt(g[0], 2)); Kit.setText("gr-e2", Kit.fmt(g[1], 2)); Kit.setText("gr-e3", Kit.fmt(n, 2)); Kit.setText("gr-e4", Kit.fmt(L0, 2));
    }
    draw();
    onMapClick("gr-map", function (x, y) { st.w = [x, y]; st.trail = []; s1.set(x); s2.set(y); msg(); draw(); });
  }

  // ---------- 5. the chain rule as linked sliders ----------
  function chainFig() {
    var node = Kit.$("ch-svg"), svg = d3.select(node), nLinks = 2, x = 78, dx = 1, prog = 1, stop = null;
    var xs = Kit.slider(Kit.$("ch-controls"), { label: "Starting length x, cm", min: 40, max: 100, step: 0.5, value: x, digits: 1 }, function (v) { x = v; prog = 1; draw(); });
    var ds = Kit.slider(Kit.$("ch-controls"), { label: "Nudge Δx, cm", min: 0.1, max: 20, step: 0.1, value: dx, digits: 1 }, function (v) { dx = v; prog = 1; draw(); });
    Kit.chips(Kit.$("ch-links"), ["x → z → a", "add the loss ℓ"], 0, function (i) { nLinks = i === 0 ? 2 : 3; prog = 1; draw(); });
    Kit.button(Kit.$("ch-btn"), "Nudge", function () { if (stop) stop(); stop = Kit.animate(3200, function (p) { prog = p; draw(); }); });
    var ST = [
      { sym: "x", name: "length x, cm", min: 40, max: 100, f: function (v) { return v; }, dig: 1 },
      { sym: "z", name: "score z = 0.1 (x − 70)", min: -4, max: 4, f: function (v) { return 0.1 * (v - 70); }, s: function () { return 0.1; }, dig: 3 },
      { sym: "a", name: "probability of adult a = σ(z)", min: 0, max: 1, f: sig, s: function (v) { var a = sig(v); return a * (1 - a); }, dig: 4 },
      { sym: "ℓ", name: "loss ℓ = (a − 1)²", min: 0, max: 1, f: function (v) { return (v - 1) * (v - 1); }, s: function (v) { return 2 * (v - 1); }, dig: 4 }
    ];
    function chain() {
      var n = nLinks + 1, v0 = [x], v1 = [x + dx], sl = [], i;
      for (i = 1; i < n; i++) { v0.push(ST[i].f(v0[i - 1])); v1.push(ST[i].f(v1[i - 1])); sl.push(ST[i].s(v0[i - 1])); }
      var pred = [dx]; for (i = 1; i < n; i++) pred.push(pred[i - 1] * sl[i - 1]);
      return { n: n, v0: v0, v1: v1, sl: sl, pred: pred };
    }
    function draw() {
      var P = Kit.palette(), W = Kit.width(node), C = chain(), n = C.n, narrow = W < 560, rowH = narrow ? 112 : 96, H = (narrow ? 34 : 20) + n * rowH - 34, ml = 26, mr = 26, i;
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      var defs = svg.append("defs");
      defs.append("marker").attr("id", "ch-arrow").attr("viewBox", "0 0 10 10").attr("refX", 8).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
        .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", P.muted);
      var s = prog * n;
      for (i = 0; i < n; i++) {
        var y = (narrow ? 58 : 40) + i * rowH, S = ST[i], sc = d3.scaleLinear().domain([S.min, S.max]).range([ml, W - mr]);
        var r = Kit.ease(clamp(s - i, 0, 1)), a = clamp(C.v0[i], S.min, S.max), b = clamp(C.v1[i], S.min, S.max), bShown = a + (b - a) * (prog >= 1 ? 1 : r);
        svg.append("text").attr("x", ml).attr("y", narrow ? y - 36 : y - 17).attr("font-size", 14).attr("fill", P.ink2).text(S.name);
        var vt = svg.append("text").attr("x", narrow ? ml : W - mr).attr("y", y - 17).attr("text-anchor", narrow ? "start" : "end").attr("font-size", 14).attr("fill", P.ink).attr("font-weight", 600).text(S.sym + " = " + Kit.fmt(C.v0[i], S.dig));
        if (r > 0.02) vt.append("tspan").attr("fill", P.coral).text("   change " + sgn(C.v1[i] - C.v0[i]) + Kit.fmt(Math.abs((C.v1[i] - C.v0[i]) * (prog >= 1 ? 1 : r)), S.dig));
        svg.append("line").attr("x1", ml).attr("x2", W - mr).attr("y1", y).attr("y2", y).attr("stroke", P.rule).attr("stroke-width", 6).attr("stroke-linecap", "round");
        sc.ticks(5).forEach(function (t) { svg.append("line").attr("x1", sc(t)).attr("x2", sc(t)).attr("y1", y + 6).attr("y2", y + 10).attr("stroke", P.muted); svg.append("text").attr("x", sc(t)).attr("y", y + 23).attr("text-anchor", "middle").attr("font-size", 11.5).attr("fill", P.muted).text(d3.format("~g")(t)); });
        var bw = Math.max(3, Math.abs(sc(bShown) - sc(a)));
        svg.append("rect").attr("x", Math.min(sc(a), sc(bShown)) - (bw < 3 ? 1.5 : 0)).attr("y", y - 4).attr("width", bw).attr("height", 8).attr("rx", 3).attr("fill", P.coral).attr("opacity", r > 0 ? 0.95 : 0);
        svg.append("circle").attr("cx", sc(a)).attr("cy", y).attr("r", 9).attr("fill", P.navy).attr("stroke", P.bg).attr("stroke-width", 2.5);
        if (r > 0.02) svg.append("circle").attr("cx", sc(bShown)).attr("cy", y).attr("r", 6.5).attr("fill", P.coral).attr("stroke", P.bg).attr("stroke-width", 2);
        if (i < n - 1) {
          var ly = y + 38, lw = Math.min(W - 80, 330);
          svg.append("line").attr("x1", W / 2).attr("x2", W / 2).attr("y1", y + 26).attr("y2", y + rowH - (narrow ? 46 : 22)).attr("stroke", P.muted).attr("stroke-width", 1.6).attr("marker-end", "url(#ch-arrow)");
          svg.append("rect").attr("x", W / 2 - lw / 2).attr("y", ly - 14).attr("width", lw).attr("height", 24).attr("rx", 12).attr("fill", P.paper).attr("stroke", P.rule);
          svg.append("text").attr("x", W / 2).attr("y", ly + 3).attr("text-anchor", "middle").attr("font-size", 14).attr("fill", P.ink)
            .text("× " + Kit.fmt(C.sl[i], 4) + "  slope of " + ST[i + 1].sym + " with respect to " + ST[i].sym);
          var u = clamp((s - (i + 0.7)) / 0.7, 0, 1);
          if (u > 0 && u < 1 && prog < 1) svg.append("circle").attr("cx", W / 2).attr("cy", y + 26 + u * (rowH - (narrow ? 72 : 48))).attr("r", 5).attr("fill", P.coral);
        }
      }
      // equation boxes
      var b = function (cls, v, lab) { return '<span class="t ' + cls + '"><b>' + v + '</b><i>' + lab + '</i></span>'; }, op = function (t) { return '<span class="op">' + t + '</span>'; };
      var html = b("c-navy", Kit.fmt(dx, 2), "nudge &Delta;x, cm");
      for (i = 1; i < n; i++) html += op("&times;") + b("c-sand", Kit.fmt(C.sl[i - 1], 4), "slope of " + ST[i].sym + " w.r.t. " + ST[i - 1].sym) + op("=") + b("c-coral", Kit.fmt(C.pred[i], 4), "change in " + ST[i].sym) + (i < n - 1 ? "" : "");
      var prod = C.pred[n - 1] / dx, actual = (C.v1[n - 1] - C.v0[n - 1]) / dx;
      Kit.$("ch-eqn").innerHTML = html;
      Kit.$("ch-eqn2").innerHTML = b("c-coral", Kit.fmt(prod, 4), "product of local slopes") + op("vs") + b("c-teal", Kit.fmt(actual, 4), "actual change &divide; &Delta;x") + op("gap") + b("", Kit.fmt(Math.abs(prod - actual), 4), "small when the nudge is small");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- 6. a sharp valley and a flat valley ----------
  function curveFig() {
    var st = { ks: 6, kf: 0.5, delta: 0.5, eta: 0.25, trails: null };
    var sd = Kit.slider(Kit.$("cv-controls"), { label: "Distance from the bottom δ", min: 0, max: 2, step: 0.05, value: st.delta }, function (v) { st.delta = v; st.trails = null; draw(); });
    var ss = Kit.slider(Kit.$("cv-controls"), { label: "Sharp valley curvature", min: 2, max: 10, step: 0.5, value: st.ks, digits: 1 }, function (v) { st.ks = v; st.trails = null; draw(); });
    var sf = Kit.slider(Kit.$("cv-controls"), { label: "Flat valley curvature", min: 0.1, max: 1.5, step: 0.1, value: st.kf, digits: 1 }, function (v) { st.kf = v; st.trails = null; draw(); });
    var se = Kit.slider(Kit.$("cv-controls"), { label: "Step size η", min: 0.02, max: 0.6, step: 0.01, value: st.eta }, function (v) { st.eta = v; st.trails = null; draw(); });
    Kit.button(Kit.$("cv-btns"), "Take 8 steps", function () {
      function run(k) { var w = st.delta, o = [w], i; for (i = 0; i < 8; i++) { w = w - st.eta * k * w; o.push(w); } return o; }
      st.trails = { s: run(st.ks), f: run(st.kf) }; draw();
    });
    Kit.button(Kit.$("cv-btns"), "Clear steps", function () { st.trails = null; draw(); });
    var verdict = function (fac) {
      if (Math.abs(fac) < 0.02) return "reaches the bottom in one step";
      if (fac >= 0 && fac < 1) return fac > 0.8 ? "creeps towards the bottom" : "glides to the bottom";
      if (fac < 0 && fac > -1) return "bounces across the bottom and settles";
      if (fac >= 1) return "does not move closer";
      return "bounces out further and further";
    };
    function draw() {
      var xs = lin(-2.2, 2.2, 221), YMAX = 12;
      Kit.plot("cv-plot", function (P) {
        var tr = [
          { x: xs, y: xs.map(function (v) { return 0.5 * st.kf * v * v; }), mode: "lines", line: { color: P.navy, width: 3.4 }, hoverinfo: "skip" },
          { x: xs, y: xs.map(function (v) { return 0.5 * st.ks * v * v; }), mode: "lines", line: { color: P.coral, width: 3.4 }, hoverinfo: "skip" }];
        if (st.trails) {
          tr.push({ x: st.trails.f, y: st.trails.f.map(function (v) { return 0.5 * st.kf * v * v; }), mode: "lines+markers", line: { color: P.navy, width: 1.6, dash: "dot" }, marker: { color: P.navy, size: 7 }, hoverinfo: "skip" });
          tr.push({ x: st.trails.s, y: st.trails.s.map(function (v) { return 0.5 * st.ks * v * v; }), mode: "lines+markers", line: { color: P.coral, width: 1.6, dash: "dot" }, marker: { color: P.coral, size: 7 }, hoverinfo: "skip" });
        }
        tr.push({ x: [st.delta], y: [0.5 * st.kf * st.delta * st.delta], mode: "markers", marker: { color: P.navy, size: 13, line: { color: P.bg, width: 2.5 } }, hoverinfo: "skip" });
        tr.push({ x: [st.delta], y: [0.5 * st.ks * st.delta * st.delta], mode: "markers", marker: { color: P.coral, size: 13, line: { color: P.bg, width: 2.5 } }, hoverinfo: "skip" });
        return tr;
      }, function (P) {
        return { dragmode: false, margin: { l: 52, r: 16, t: 12, b: 58 }, xaxis: { range: [-2.2, 2.2], title: { text: "weight, as distance from the best value", standoff: 8 } }, yaxis: { range: [0, YMAX], title: { text: "loss" } },
          shapes: [{ type: "line", x0: st.delta, x1: st.delta, y0: 0, y1: YMAX, line: { color: P.muted, width: 1, dash: "dot" } }],
          annotations: [
            { x: 0, y: Math.min(9, 0.5 * st.ks * 0.7 * 0.7 + 3.4), text: "sharp valley", showarrow: false, font: { color: P.coral, size: 15 } },
            { x: -1.7, y: 0.5 * st.kf * 1.7 * 1.7 + 0.5, text: "flat valley", showarrow: false, xanchor: "center", yshift: 8, font: { color: P.navy, size: 15 } }] };
      });
      var d2 = st.delta * st.delta, rs = 0.5 * st.ks * d2, rf = 0.5 * st.kf * d2, fs = 1 - st.eta * st.ks, ff = 1 - st.eta * st.kf;
      var b = function (cls, v, lab) { return '<span class="t ' + cls + '"><b>' + v + '</b><i>' + lab + '</i></span>'; }, op = function (t) { return '<span class="op">' + t + '</span>'; };
      Kit.$("cv-eqn").innerHTML =
        b("c-coral", Kit.fmt(rs, 3), "loss rise, sharp: &frac12; &times; " + Kit.fmt(st.ks, 1) + " &times; &delta;&sup2;") + op("vs") +
        b("c-navy", Kit.fmt(rf, 3), "loss rise, flat: &frac12; &times; " + Kit.fmt(st.kf, 1) + " &times; &delta;&sup2;") + op("&rarr;") +
        b("", rf > 1e-9 ? "&times;" + Kit.fmt(rs / rf, 1) : "&ndash;", "sharp over flat") +
        '';
      Kit.$("cv-eqn2").innerHTML = b("c-coral", Kit.fmt(fs, 2), "step factor 1 &minus; &eta;k, sharp") + op("and") + b("c-navy", Kit.fmt(ff, 2), "step factor 1 &minus; &eta;k, flat") + op("&rarr;") + b("c-coral", Kit.fmt(2 / st.ks, 2), "largest safe step, sharp (2 &divide; k)") + op("and") + b("c-navy", Kit.fmt(2 / st.kf, 2), "largest safe step, flat");
      Kit.setText("cv-status", "Sharp valley: " + verdict(fs) + ". Flat valley: " + verdict(ff) + ".");
    }
    draw();
  }
})();
