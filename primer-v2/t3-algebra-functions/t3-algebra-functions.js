// Chapter 0.3 Algebra and functions: function machines, vertical line test, the function explorer
// (five families plus a compose mode), exponent and log rule labs, a power-versus-exponential race,
// a log finder, a log-log fit and an inverse-function mirror.
(function () {
  "use strict";
  var $ = Kit.$, fmt = Kit.fmt;

  // ---------- small helpers ----------
  function box(v, label, cls) { return '<span class="t' + (cls ? " c-" + cls : "") + '"><b>' + v + "</b><i>" + (label || "") + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function eqRow(parent, html) { var d = Kit.html("div", { class: "eqn" }, parent); d.innerHTML = html; return d; }
  function num(v, d) { return v == null || !isFinite(v) ? "undefined" : fmt(v, d == null ? 2 : d); }
  function sgn(v) { return v < 0 ? "−" : "+"; }
  function pretty(v) {
    if (v == null || !isFinite(v)) return "undefined";
    var a = Math.abs(v);
    if (a !== 0 && (a >= 1e7 || a < 1e-3)) return v.toExponential(2).replace("e", " × 10^").replace("^+", "^");
    return fmt(v, a >= 100 ? 0 : a >= 10 ? 1 : a >= 1 ? 2 : 4).replace(/(\.\d*?)0+$/, "$1").replace(/\.$/, "");
  }
  function tex(el, s) {
    if (window.katex) { try { window.katex.render(s, el, { throwOnError: false }); return; } catch (e) { /* fall through */ } }
    el.textContent = s;
  }
  function setChip(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function linspace(a, b, n) { var o = []; for (var i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function dashed(P, color) { return { color: color || P.muted, width: 1.5, dash: "dot" }; }

  document.addEventListener("DOMContentLoaded", function () {
    machine();
    vline();
    explorer("ex1", { toggle: false, mode: 0 });
    ruleLab("xr", EXP);
    race();
    logFinder();
    ruleLab("lr", LOG);
    loglog();
    inverse();
    explorer("ex2", { toggle: true, mode: 1, f: { fam: "line", p: { w: 1.5, b: -1 } }, g: { fam: "sigmoid", p: { w: 1, b: 0 } } });
  });

  // =====================================================================
  // 1. Function machines
  // =====================================================================
  function machine() {
    var M = [
      { name: "Length to weight", xl: "length L, cm", yl: "weight W, kg", min: 18, max: 60, step: 0.5, v: 41, d: 1, od: 3,
        f: function (x) { return 0.0044 * Math.pow(x, 2.923) / 1000; }, rule: "0.0044 × L^2.923 ÷ 1000", xr: [18, 60], yr: [0, 0.8], xn: "L" },
      { name: "Celsius to Fahrenheit", xl: "water temperature, °C", yl: "temperature, °F", min: 20, max: 34, step: 0.5, v: 28, d: 1, od: 1,
        f: function (x) { return 1.8 * x + 32; }, rule: "1.8 × C + 32", xr: [20, 34], yr: [60, 100], xn: "C" },
      { name: "Mature or immature?", xl: "length L, cm", yl: "output (1 mature, 0 immature)", min: 18, max: 60, step: 0.5, v: 41, d: 1, od: 0,
        f: function (x) { return x >= 35.8 ? 1 : 0; }, rule: "1 if L ≥ 35.8, else 0", xr: [18, 60], yr: [-0.2, 1.2], xn: "L", isStep: true }
    ];
    var cur = 0, sl = null, ctl = $("mc-controls");
    Kit.chips($("mc-bar"), M.map(function (m) { return m.name; }), 0, function (i) { cur = i; build(); });
    function build() {
      ctl.innerHTML = "";
      var m = M[cur];
      sl = Kit.slider(ctl, { label: "Input: " + m.xl, min: m.min, max: m.max, step: m.step, value: m.v, digits: m.d }, draw);
      draw();
    }
    function draw() {
      var m = M[cur], x = sl.get(), y = m.f(x);
      $("mc-eq").innerHTML = box(fmt(x, m.d), m.xl, "coral") + op("→") + box(m.rule, "the rule f") + op("→") + box(fmt(y, m.od), m.yl, "navy");
      var xs = linspace(m.xr[0], m.xr[1], m.isStep ? 181 : 120), ys = xs.map(m.f);
      Kit.plot("mc-plot", function (P) {
        return [
          { x: xs, y: ys, mode: "lines", line: { color: P.navy, width: 3.5, shape: m.isStep ? "hv" : "linear" }, hoverinfo: "skip" },
          { x: [x, x, m.xr[0]], y: [m.yr[0], y, y], mode: "lines", line: dashed(P), hoverinfo: "skip" },
          { x: [x], y: [y], mode: "markers", marker: { color: P.coral, size: 12, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" }
        ];
      }, { xaxis: { title: { text: m.xl }, range: m.xr, fixedrange: true }, yaxis: { title: { text: m.yl }, range: m.yr, fixedrange: true }, margin: { l: 56, r: 16, t: 10, b: 48 }, dragmode: false }, { staticPlot: true });
      $("mc-plot").setAttribute("aria-label", "Graph of the machine " + m.name + ". At input " + fmt(x, m.d) + " the output is " + fmt(y, m.od) + ".");
    }
    build();
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 1b. Vertical line test
  // =====================================================================
  function vline() {
    var C = [
      { name: "y = x² / 2", curve: function () { var t = linspace(-2, 2, 120); return { x: t, y: t.map(function (u) { return u * u / 2; }) }; },
        hits: function (x0) { return Math.abs(x0) <= 2 ? [x0 * x0 / 2] : []; } },
      { name: "x = y² / 2 (sideways)", curve: function () { var t = linspace(-2, 2, 120); return { x: t.map(function (u) { return u * u / 2; }), y: t }; },
        hits: function (x0) { return x0 < 0 || x0 > 2 ? [] : x0 === 0 ? [0] : [Math.sqrt(2 * x0), -Math.sqrt(2 * x0)]; } },
      { name: "circle x² + y² = 4", curve: function () { var t = linspace(0, 2 * Math.PI, 160); return { x: t.map(function (u) { return 2 * Math.cos(u); }), y: t.map(function (u) { return 2 * Math.sin(u); }) }; },
        hits: function (x0) { var r = 4 - x0 * x0; return r < 0 ? [] : r === 0 ? [0] : [Math.sqrt(r), -Math.sqrt(r)]; } }
    ];
    var cur = 0, sl = Kit.slider($("vl-controls"), { label: "Vertical line at x", min: -2.2, max: 2.2, step: 0.05, value: 0.8 }, draw);
    Kit.chips($("vl-bar"), C.map(function (c) { return c.name; }), 0, function (i) { cur = i; draw(); });
    function draw() {
      var c = C[cur], x0 = Math.round(sl.get() * 100) / 100, hs = c.hits(x0), cv = c.curve();
      $("vl-n").textContent = String(hs.length);
      var v = $("vl-verdict");
      v.textContent = hs.length > 1 ? "Two outputs for one input at this x. This curve fails the test." : hs.length === 1 ? "One output here." : "No output at this x: outside the domain.";
      v.style.color = hs.length > 1 ? "var(--coral)" : "var(--ink-2)";
      Kit.plot("vl-plot", function (P) {
        return [
          { x: cv.x, y: cv.y, mode: "lines", line: { color: P.navy, width: 3.5 }, hoverinfo: "skip" },
          { x: [x0, x0], y: [-2.4, 2.4], mode: "lines", line: { color: P.muted, width: 2, dash: "dash" }, hoverinfo: "skip" },
          { x: hs.map(function () { return x0; }), y: hs, mode: "markers", marker: { color: P.coral, size: 13, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" }
        ];
      }, { xaxis: { range: [-2.4, 2.4], fixedrange: true, title: { text: "x" }, constrain: "domain" }, yaxis: { range: [-2.4, 2.4], fixedrange: true, title: { text: "y" }, scaleanchor: "x", scaleratio: 1, constrain: "domain" }, margin: { l: 44, r: 16, t: 10, b: 44 }, dragmode: false }, { staticPlot: true });
      $("vl-plot").setAttribute("aria-label", "The curve " + c.name + " and a vertical line at x = " + fmt(x0, 2) + " that crosses it " + hs.length + " time" + (hs.length === 1 ? "" : "s") + ".");
    }
    draw();
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 2. Function explorer (five families, compose mode)
  // =====================================================================
  var FAM = {
    line: { label: "Line", init: { w: 1, b: 0 },
      sl: [["w", "slope w", -3, 3, 0.1], ["b", "bias b", -5, 5, 0.1]],
      f: function (x, p) { return p.w * x + p.b; },
      tex: function (p, xn, yn) { return yn + " = " + num(p.w, 1) + xn + " " + sgn(p.b) + " " + num(Math.abs(p.b), 1); },
      rows: function (p, x, y, io) {
        return [box(num(y), io.yl, io.yc) + op("=") + box(num(p.w, 1), "slope w") + op("×") + box(num(x), io.xl, io.xc) + op("+") + box(num(p.b, 1), "bias b")];
      } },
    quad: { label: "Quadratic", init: { a: 1, b: 0, c: -2 },
      sl: [["a", "a", -2, 2, 0.1], ["b", "b", -4, 4, 0.1], ["c", "c", -5, 5, 0.1]],
      f: function (x, p) { return p.a * x * x + p.b * x + p.c; },
      tex: function (p, xn, yn) { return yn + " = " + num(p.a, 1) + xn + "^2 " + sgn(p.b) + " " + num(Math.abs(p.b), 1) + xn + " " + sgn(p.c) + " " + num(Math.abs(p.c), 1); },
      rows: function (p, x, y, io) {
        return [box(num(y), io.yl, io.yc) + op("=") + box(num(p.a, 1), "a") + op("×") + box(num(x * x), io.xn + " squared") + op("+") + box(num(p.b, 1), "b") + op("×") + box(num(x), io.xl, io.xc) + op("+") + box(num(p.c, 1), "c")];
      } },
    exp: { label: "Exponential", init: { a: 1, r: 1.5 },
      sl: [["a", "scale a", -3, 3, 0.1], ["r", "growth factor r", 0.2, 3, 0.05]],
      f: function (x, p) { return p.a * Math.pow(p.r, x); },
      tex: function (p, xn, yn) { return yn + " = " + num(p.a, 1) + "\\cdot " + num(p.r, 2) + "^{" + xn + "}"; },
      rows: function (p, x, y, io) {
        return [box(num(y), io.yl, io.yc) + op("=") + box(num(p.a, 1), "scale a") + op("×") + box(num(Math.pow(p.r, x), 3), num(p.r, 2) + " raised to " + num(x))];
      } },
    log: { label: "Logarithm", init: { a: 1, B: 2, c: 0 },
      sl: [["a", "scale a", -3, 3, 0.1], ["B", "base B", 1.5, 10, 0.1], ["c", "shift c", -3, 3, 0.1]],
      f: function (x, p) { return p.a * Math.log(x) / Math.log(p.B) + p.c; },
      tex: function (p, xn, yn) { return yn + " = " + num(p.a, 1) + "\\log_{" + num(p.B, 1) + "} " + xn + " " + sgn(p.c) + " " + num(Math.abs(p.c), 1); },
      rows: function (p, x, y, io) {
        var l = x > 0 ? Math.log(x) / Math.log(p.B) : null;
        return [box(num(y), io.yl, io.yc) + op("=") + box(num(p.a, 1), "scale a") + op("×") + box(l == null ? "undefined" : num(l, 3), "log base " + num(p.B, 1) + " of " + num(x)) + op("+") + box(num(p.c, 1), "shift c")];
      } },
    sigmoid: { label: "Sigmoid", init: { w: 1, b: 0 },
      sl: [["w", "weight w", -4, 4, 0.1], ["b", "bias b", -6, 6, 0.1]],
      f: function (x, p) { return 1 / (1 + Math.exp(-(p.w * x + p.b))); },
      tex: function (p, xn, yn) { return yn + " = \\dfrac{1}{1+e^{-(" + num(p.w, 1) + xn + " " + sgn(p.b) + " " + num(Math.abs(p.b), 1) + ")}}"; },
      rows: function (p, x, y, io) {
        var z = p.w * x + p.b;
        return [
          box(num(z, 2), "z = w x + b") + op("=") + box(num(p.w, 1), "weight w") + op("×") + box(num(x), io.xl, io.xc) + op("+") + box(num(p.b, 1), "bias b"),
          box(num(y, 3), io.yl, io.yc) + op("=") + box("1") + op("÷") + op("(") + box("1") + op("+") + box(num(Math.exp(-z), 3), "e to the −z") + op(")")
        ];
      } }
  };
  var ORDER = ["line", "quad", "exp", "log", "sigmoid"];
  function ev(fam, p, x) { var y = FAM[fam].f(x, p); return isFinite(y) && Math.abs(y) < 1e6 ? y : null; }
  function cloneState(s) { var p = {}; for (var k in s.p) p[k] = s.p[k]; return { fam: s.fam, p: p }; }

  function explorer(id, opt) {
    var root = $(id);
    var st = {
      mode: opt.mode || 0,
      f: cloneState(opt.f || { fam: "line", p: FAM.line.init }),
      g: cloneState(opt.g || { fam: "quad", p: FAM.quad.init })
    };
    var XS = linspace(-5, 5, 401), chipsF, chipsG, chipsMode, probe;

    var top = Kit.html("div", { class: "bar" }, root);
    if (opt.toggle) {
      Kit.html("span", { class: "label" }, top, "Mode");
      chipsMode = Kit.chips(top, ["one function", "compose: f then g"], st.mode, function (i) { st.mode = i; sync(); });
      Kit.button(top, "swap f and g", function () { var t = st.f; st.f = st.g; st.g = t; buildPanel("f"); buildPanel("g"); draw(); });
    }
    var fBar = Kit.html("div", { class: "bar" }, root);
    Kit.html("span", { class: "label" }, fBar, "f");
    chipsF = Kit.chips(fBar, ORDER.map(function (k) { return FAM[k].label; }), ORDER.indexOf(st.f.fam), function (i) { st.f = { fam: ORDER[i], p: cloneP(FAM[ORDER[i]].init) }; buildPanel("f"); draw(); });
    var fCtl = Kit.html("div", { class: "controls" }, root);
    var gWrap = Kit.html("div", {}, root);
    var gBar = Kit.html("div", { class: "bar" }, gWrap);
    Kit.html("span", { class: "label" }, gBar, "g");
    chipsG = Kit.chips(gBar, ORDER.map(function (k) { return FAM[k].label; }), ORDER.indexOf(st.g.fam), function (i) { st.g = { fam: ORDER[i], p: cloneP(FAM[ORDER[i]].init) }; buildPanel("g"); draw(); });
    var gCtl = Kit.html("div", { class: "controls" }, gWrap);
    var formula = Kit.html("div", { class: "formula" }, root);
    var plotDiv = Kit.html("div", { class: "plot tall", id: id + "-plot", role: "img", "aria-label": "Graph of the chosen function" }, root);
    var pCtl = Kit.html("div", { class: "controls" }, root);
    probe = Kit.slider(pCtl, { label: "Probe input x", min: -5, max: 5, step: 0.05, value: 1.5 }, draw);
    var eqBox = Kit.html("div", {}, root);
    var status = Kit.html("p", { class: "status", "aria-live": "polite" }, root);

    function cloneP(p) { var o = {}; for (var k in p) o[k] = p[k]; return o; }
    function buildPanel(which) {
      var s = st[which], ctl = which === "f" ? fCtl : gCtl, bs = which === "f" ? chipsF : chipsG;
      setChip(bs, ORDER.indexOf(s.fam));
      ctl.innerHTML = "";
      FAM[s.fam].sl.forEach(function (d) {
        Kit.slider(ctl, { label: d[1], min: d[2], max: d[3], step: d[4], value: s.p[d[0]] }, function (v) { s.p[d[0]] = v; draw(); });
      });
    }
    function sync() {
      if (chipsMode) setChip(chipsMode, st.mode);
      gWrap.style.display = st.mode ? "" : "none";
      draw();
    }
    function labelPoint(xs, ys, target) {
      var best = -1, d = 1e9;
      for (var i = 0; i < xs.length; i++) if (ys[i] != null && Math.abs(ys[i]) < 5.2) { var e = Math.abs(xs[i] - target); if (e < d) { d = e; best = i; } }
      return best;
    }

    function draw() {
      var x = Math.round(probe.get() * 100) / 100, compose = !!st.mode;
      var F = FAM[st.f.fam], G = FAM[st.g.fam];
      var fy = XS.map(function (t) { return ev(st.f.fam, st.f.p, t); });
      var u = ev(st.f.fam, st.f.p, x), y1 = u;
      var gy = null, hy = null, yy = null;
      if (compose) {
        gy = XS.map(function (t) { return ev(st.g.fam, st.g.p, t); });
        hy = fy.map(function (v) { return v == null ? null : ev(st.g.fam, st.g.p, v); });
        yy = u == null ? null : ev(st.g.fam, st.g.p, u);
      }
      // formula line
      var tx = compose ? "\\begin{aligned} f(x) &= " + F.tex(st.f.p, "x", "").replace(/^ = /, "") + " \\\\ g(u) &= " + G.tex(st.g.p, "u", "").replace(/^ = /, "") + " \\end{aligned}"
                       : F.tex(st.f.p, "x", "y");
      tex(formula, tx);

      // equation boxes
      eqBox.innerHTML = "";
      var ioF = compose ? { xl: "input x", xc: "coral", xn: "x", yl: "u = f(x)", yc: "navy" } : { xl: "input x", xc: "coral", xn: "x", yl: "output y", yc: "navy" };
      F.rows(st.f.p, x, u, ioF).forEach(function (h) { eqRow(eqBox, h); });
      if (compose) {
        var ioG = { xl: "input u", xc: "navy", xn: "u", yl: "output y = g(u)", yc: "teal" };
        if (u != null) G.rows(st.g.p, u, yy, ioG).forEach(function (h) { eqRow(eqBox, h); });
        else eqRow(eqBox, box("undefined", "g needs an input u") );
      }
      // status line
      var msg = "";
      if (!compose) {
        msg = u == null ? "At x = " + num(x) + " this function has no output: the input is outside its domain." : "At x = " + num(x) + " the output is y = " + num(u, 3) + ".";
      } else if (u == null) msg = "f gives no output at this x, so g has no input to work on.";
      else if (yy == null) msg = "u = " + num(u, 3) + " is outside the domain of g. Try another x or another g.";
      else {
        msg = "x = " + num(x) + " goes into f, which gives u = " + num(u, 3) + ", and g turns that into y = " + num(yy, 3) + ".";
        if (st.f.fam === "line" && st.g.fam === "line") {
          var W = st.g.p.w * st.f.p.w, Bb = st.g.p.w * st.f.p.b + st.g.p.b;
          msg += " Two lines in a row make one line: slope " + num(W, 2) + " and bias " + num(Bb, 2) + ".";
        }
      }
      status.textContent = msg;

      // plot
      var lf = labelPoint(XS, fy, 4.4), lh = compose ? labelPoint(XS, hy, 3.0) : -1, lg = compose ? labelPoint(XS, gy, -4.0) : -1;
      Kit.plot(plotDiv, function (P) {
        var T = [];
        // asymptotes (single mode only)
        if (!compose) {
          if (st.f.fam === "exp") T.push({ x: [-5, 5], y: [0, 0], mode: "lines", line: dashed(P), hoverinfo: "skip" });
          if (st.f.fam === "sigmoid") { T.push({ x: [-5, 5], y: [0, 0], mode: "lines", line: dashed(P), hoverinfo: "skip" }); T.push({ x: [-5, 5], y: [1, 1], mode: "lines", line: dashed(P), hoverinfo: "skip" }); }
          if (st.f.fam === "log") T.push({ x: [0, 0], y: [-6, 6], mode: "lines", line: dashed(P), hoverinfo: "skip" });
        }
        T.push({ x: XS, y: fy, mode: "lines", connectgaps: false, line: { color: P.navy, width: compose ? 2.5 : 3.5, dash: compose ? "dot" : "solid" }, hoverinfo: "skip" });
        if (compose) {
          T.push({ x: XS, y: gy, mode: "lines", line: { color: P.sand, width: 2.5, dash: "dash" }, hoverinfo: "skip" });
          T.push({ x: XS, y: hy, mode: "lines", line: { color: P.teal, width: 4 }, hoverinfo: "skip" });
        }
        // probe guides and markers
        var top = compose ? yy : u;
        // Put the probe label on the side of the point where the curve is not.
        function probePos(px) {
          var arr = compose ? hy : fy, n = XS.length, k = Math.max(1, Math.min(n - 2, Math.round((px - XS[0]) / (XS[n - 1] - XS[0]) * (n - 1))));
          var d = (arr[k + 1] != null && arr[k - 1] != null) ? arr[k + 1] - arr[k - 1] : 0, left = px > XS[0] + 0.28 * (XS[n - 1] - XS[0]);
          return d >= 0 ? (left ? "top left" : "bottom right") : (left ? "bottom left" : "top right");
        }
        if (top != null) T.push({ x: [x, x], y: [0, top], mode: "lines", line: dashed(P, P.coral), hoverinfo: "skip" });
        if (compose && u != null) T.push({ x: [x], y: [u], mode: "markers", marker: { color: P.navy, size: 10, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" });
        if (top != null) T.push({ x: [x], y: [top], mode: "markers+text", text: ["(" + num(x, 1) + ", " + num(top, 2) + ")"], textposition: probePos(x), textfont: { color: P.coral, size: 13 }, cliponaxis: false, marker: { color: P.coral, size: 12, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" });
        // curve labels
        function lab(i, ys, text, color) { if (i >= 0) T.push({ x: [XS[i]], y: [ys[i]], mode: "text", text: [text], textposition: "top center", textfont: { color: color, size: 14 }, hoverinfo: "skip" }); }
        if (compose) { lab(lf, fy, "f(x)", P.navy); lab(lg, gy, "g(u)", P.sand); lab(lh, hy, "g(f(x))", P.teal); }
        return T;
      }, { xaxis: { range: [-5, 5], fixedrange: true, title: { text: "input x" }, zeroline: true }, yaxis: { range: [-6, 6], fixedrange: true, title: { text: compose ? "output" : "output y" }, zeroline: true }, margin: { l: 48, r: 16, t: 12, b: 46 }, dragmode: false }, { staticPlot: true });
      plotDiv.setAttribute("aria-label", compose
        ? "Graph of f, g and the composition g after f. At x = " + num(x) + " the final output is " + num(yy, 2) + "."
        : "Graph of the " + F.label.toLowerCase() + ". At x = " + num(x) + " the output is " + num(u, 2) + ".");
    }

    buildPanel("f"); buildPanel("g");
    sync();
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 3 and 4. Rule labs (exponent rules, log rules)
  // =====================================================================
  function pw(B, e) { return Math.pow(B, e); }
  var EXP = {
    vars: {
      B: { label: "base B", min: 2, max: 9, step: 1, v: 2 },
      m: { label: "exponent m", min: -3, max: 6, step: 1, v: 5 },
      n: { label: "exponent n", min: -3, max: 6, step: 1, v: 3 }
    },
    rules: [
      { name: "multiply", vars: ["B", "m", "n"], calc: function (v) {
          var l1 = pw(v.B, v.m), l2 = pw(v.B, v.n), lhs = l1 * l2, rhs = pw(v.B, v.m + v.n);
          return { rows: [
            box(pretty(l1), "B^m = " + v.B + "^" + v.m, "navy") + op("×") + box(pretty(l2), "B^n = " + v.B + "^" + v.n, "navy") + op("=") + box(pretty(lhs), "left side", "navy"),
            box(pretty(rhs), "B^(m+n) = " + v.B + "^" + (v.m + v.n), "teal") + op("=") + box(pretty(rhs), "right side", "teal")], l: lhs, r: rhs,
            text: v.m > 0 && v.n > 0 && v.m + v.n <= 9 ? "Written out: " + rep(v.B, v.m) + "  ×  " + rep(v.B, v.n) + "  =  " + rep(v.B, v.m + v.n) + ". Counting the " + v.B + "s gives " + v.m + " + " + v.n + " = " + (v.m + v.n) + "." : "Add the exponents: " + v.m + " + " + v.n + " = " + (v.m + v.n) + "." }; } },
      { name: "divide", vars: ["B", "m", "n"], calc: function (v) {
          var l1 = pw(v.B, v.m), l2 = pw(v.B, v.n), lhs = l1 / l2, rhs = pw(v.B, v.m - v.n);
          return { rows: [
            box(pretty(l1), "B^m = " + v.B + "^" + v.m, "navy") + op("÷") + box(pretty(l2), "B^n = " + v.B + "^" + v.n, "navy") + op("=") + box(pretty(lhs), "left side", "navy"),
            box(pretty(rhs), "B^(m−n) = " + v.B + "^" + (v.m - v.n), "teal") + op("=") + box(pretty(rhs), "right side", "teal")], l: lhs, r: rhs,
            text: "Subtract the exponents: " + v.m + " − " + v.n + " = " + (v.m - v.n) + ". A B on the bottom cancels one B on top." }; } },
      { name: "power of a power", vars: ["B", "m", "n"], calc: function (v) {
          var l1 = pw(v.B, v.m), lhs = pw(l1, v.n), rhs = pw(v.B, v.m * v.n);
          return { rows: [
            box(pretty(l1), "B^m = " + v.B + "^" + v.m, "navy") + op("raised to n =") + box(String(v.n), "n") + op("=") + box(pretty(lhs), "left side", "navy"),
            box(pretty(rhs), "B^(m×n) = " + v.B + "^" + (v.m * v.n), "teal") + op("=") + box(pretty(rhs), "right side", "teal")], l: lhs, r: rhs,
            text: "Multiply the exponents: " + v.m + " × " + v.n + " = " + (v.m * v.n) + "." }; } },
      { name: "zero and negative", vars: ["B", "n"], calc: function (v) {
          var lhs = pw(v.B, -v.n), rhs = 1 / pw(v.B, v.n);
          return { rows: [
            box("1", "B^0 = " + v.B + "^0", "navy") + op("and") + box(pretty(lhs), "B^(−n) = " + v.B + "^" + (-v.n), "navy"),
            box(pretty(rhs), "1 ÷ B^n = 1 ÷ " + pretty(pw(v.B, v.n)), "teal")], l: lhs, r: rhs,
            text: "A negative exponent divides instead of multiplying. Zero multiplications leave the starting number 1." }; } },
      { name: "roots", vars: ["B", "n"], calc: function (v) {
          var root = pw(v.B, 1 / v.n), back = pw(root, v.n);
          return { rows: [
            box(pretty(root), "B^(1/n) = " + v.B + "^(1/" + v.n + ")", "navy") + op("raised to n =") + box(String(v.n), "n") + op("=") + box(pretty(back), "left side", "navy"),
            box(pretty(v.B), "B itself", "teal")], l: back, r: v.B,
            text: "The exponent 1/n asks for the number that gives B when multiplied by itself n times." }; } }
    ]
  };
  function rep(B, k) { var a = []; for (var i = 0; i < k; i++) a.push(String(B)); return a.join("·"); }

  var LOG = {
    vars: {
      B: { label: "base B", min: 2, max: 10, step: 1, v: 2 },
      x: { label: "x", min: 1, max: 100, step: 1, v: 8 },
      y: { label: "y", min: 1, max: 100, step: 1, v: 32 },
      k: { label: "exponent k", min: -2, max: 5, step: 0.5, v: 3 }
    },
    rules: [
      { name: "product", vars: ["B", "x", "y"], calc: function (v) {
          var a = lg(v.x, v.B), b = lg(v.y, v.B), lhs = lg(v.x * v.y, v.B);
          return { rows: [
            box(pretty(a), "log base " + v.B + " of " + v.x, "navy") + op("+") + box(pretty(b), "log base " + v.B + " of " + v.y, "navy") + op("=") + box(pretty(a + b), "right side", "navy"),
            box(pretty(lhs), "log base " + v.B + " of " + (v.x * v.y), "teal") + op("=") + box(pretty(lhs), "left side", "teal")], l: a + b, r: lhs,
            text: "Multiplying x by y adds their logs. This is why products of many small probabilities are stored as sums of logs." }; } },
      { name: "quotient", vars: ["B", "x", "y"], calc: function (v) {
          var a = lg(v.x, v.B), b = lg(v.y, v.B), lhs = lg(v.x / v.y, v.B);
          return { rows: [
            box(pretty(a), "log base " + v.B + " of " + v.x, "navy") + op("−") + box(pretty(b), "log base " + v.B + " of " + v.y, "navy") + op("=") + box(pretty(a - b), "right side", "navy"),
            box(pretty(lhs), "log base " + v.B + " of " + pretty(v.x / v.y), "teal") + op("=") + box(pretty(lhs), "left side", "teal")], l: a - b, r: lhs,
            text: "Dividing subtracts the logs. When x = y the answer is log of 1, which is 0 in any base." }; } },
      { name: "power", vars: ["B", "x", "k"], calc: function (v) {
          var a = lg(v.x, v.B), lhs = lg(Math.pow(v.x, v.k), v.B);
          return { rows: [
            box(pretty(v.k), "exponent k") + op("×") + box(pretty(a), "log base " + v.B + " of " + v.x, "navy") + op("=") + box(pretty(v.k * a), "right side", "navy"),
            box(pretty(lhs), "log base " + v.B + " of " + pretty(Math.pow(v.x, v.k)), "teal") + op("=") + box(pretty(lhs), "left side", "teal")], l: v.k * a, r: lhs,
            text: "An exponent inside a log comes out front as a multiplier. A log of a cube is three times the log." }; } },
      { name: "change of base", vars: ["B", "x"], calc: function (v) {
          var num_ = Math.log(v.x), den = Math.log(v.B), lhs = lg(v.x, v.B);
          return { rows: [
            box(pretty(num_), "ln " + v.x, "navy") + op("÷") + box(pretty(den), "ln " + v.B, "navy") + op("=") + box(pretty(num_ / den), "right side", "navy"),
            box(pretty(lhs), "log base " + v.B + " of " + v.x, "teal") + op("=") + box(pretty(lhs), "left side", "teal")], l: num_ / den, r: lhs,
            text: "Any log can be computed from natural logs. This is how calculators and NumPy do it." }; } }
    ]
  };
  function lg(x, B) { return Math.log(x) / Math.log(B); }

  function ruleLab(id, cfg) {
    var root = $(id), cur = 0, vals = {};
    for (var k in cfg.vars) vals[k] = cfg.vars[k].v;
    var bar = Kit.html("div", { class: "bar" }, root), ctl = Kit.html("div", { class: "controls" }, root), eq = Kit.html("div", {}, root), out = Kit.html("p", { class: "lab-out", "aria-live": "polite" }, root);
    Kit.chips(bar, cfg.rules.map(function (r) { return r.name; }), 0, function (i) { cur = i; build(); });
    function build() {
      ctl.innerHTML = "";
      cfg.rules[cur].vars.forEach(function (k) {
        var d = cfg.vars[k];
        Kit.slider(ctl, { label: d.label, min: d.min, max: d.max, step: d.step, value: vals[k], digits: d.step < 1 ? 1 : 0 }, function (v) { vals[k] = v; draw(); });
      });
      draw();
    }
    function draw() {
      var r = cfg.rules[cur].calc(vals);
      eq.innerHTML = "";
      r.rows.forEach(function (h) { eqRow(eq, h); });
      var ok = Math.abs(r.l - r.r) <= 1e-9 * Math.max(1, Math.abs(r.l), Math.abs(r.r));
      out.innerHTML = '<span class="' + (ok ? "ok" : "bad") + '">' + (ok ? "Both sides agree: " + pretty(r.l) + "." : "The sides differ.") + "</span> " + r.text;
    }
    build();
  }

  // =====================================================================
  // 3b. Power versus exponential race
  // =====================================================================
  function race() {
    var pw_ = 3, axis = 0, sl = Kit.slider($("rc-controls"), { label: "Look as far as x =", min: 4, max: 24, step: 1, value: 12 }, draw);
    Kit.chips($("rc-bar"), ["x squared", "x cubed", "x to the fourth"], 1, function (i) { pw_ = i + 2; draw(); });
    Kit.chips($("rc-bar"), ["linear axis", "log axis"], 0, function (i) { axis = i; draw(); });
    var cross = { 2: "4", 3: "about 9.94", 4: "16" };
    function draw() {
      var xm = sl.get(), xs = linspace(0, xm, 200), a = xs.map(function (x) { return Math.pow(x, pw_); }), b = xs.map(function (x) { return Math.pow(2, x); });
      Kit.plot("rc-plot", function (P) {
        return [
          { x: xs, y: a, mode: "lines", line: { color: P.navy, width: 3.5 }, hoverinfo: "skip" },
          { x: xs, y: b, mode: "lines", line: { color: P.coral, width: 3.5 }, hoverinfo: "skip" },
          { x: [xm], y: [a[a.length - 1]], mode: "markers", marker: { color: P.navy, size: 9 }, hoverinfo: "skip", cliponaxis: false },
          { x: [xm], y: [b[b.length - 1]], mode: "markers", marker: { color: P.coral, size: 9 }, hoverinfo: "skip", cliponaxis: false }
        ];
      }, function (P) { return { annotations: [ { xref: "paper", yref: "paper", x: 0.02, y: 0.98, xanchor: "left", yanchor: "top", showarrow: false, text: "\u2501\u2501 x to the power " + pw_, font: { color: P.navy, size: 14 } }, { xref: "paper", yref: "paper", x: 0.02, y: 0.88, xanchor: "left", yanchor: "top", showarrow: false, text: "\u2501\u2501 2 to the power x", font: { color: P.coral, size: 14 } } ], xaxis: { title: { text: "x" }, fixedrange: true, range: [0, xm] }, yaxis: { title: { text: "value" }, type: axis ? "log" : "linear", fixedrange: true, range: axis ? [0, Math.log10(Math.max(1000, Math.pow(2, xm)) * 1.2)] : [0, Math.max(a[a.length - 1], b[b.length - 1]) * 1.1] }, margin: { l: 66, r: 16, t: 14, b: 46 }, dragmode: false }; }, { staticPlot: true });
      var r = b[b.length - 1] / a[a.length - 1];
      $("rc-read").innerHTML = "<span>At x = " + xm + ": x<sup>" + pw_ + "</sup> = <b>" + pretty(a[a.length - 1]) + "</b></span><span>2<sup>x</sup> = <b>" + pretty(b[b.length - 1]) + "</b></span><span>2<sup>x</sup> is <b>" + (r >= 1 ? pretty(r) + " times" : pretty(r) + " of") + "</b> the power</span><span>The exponential first overtakes the power for good at x = <b>" + cross[pw_] + "</b></span>";
      $("rc-plot").setAttribute("aria-label", "x to the power " + pw_ + " in navy and 2 to the power x in coral up to x = " + xm + ". Beyond x = " + cross[pw_] + " the exponential is larger.");
    }
    draw();
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 4a. Log finder
  // =====================================================================
  function logFinder() {
    var bases = [{ l: "base 2", B: 2 }, { l: "base e", B: Math.E }, { l: "base 10", B: 10 }], cur = 0;
    var sl = Kit.slider($("lf-controls"), { label: "The number N", min: 1, max: 1000, step: 1, value: 1000, digits: 0 }, draw);
    Kit.chips($("lf-bar"), bases.map(function (b) { return b.l; }), 0, function (i) { cur = i; draw(); });
    var xs = linspace(0.5, 1000, 500);
    function draw() {
      var B = bases[cur].B, N = sl.get(), y = Math.log(N) / Math.log(B), nm = cur === 1 ? "e" : String(B);
      Kit.plot("lf-plot", function (P) {
        return [
          { x: xs, y: xs.map(function (x) { return Math.log(x) / Math.log(B); }), mode: "lines", line: { color: P.navy, width: 3.5 }, hoverinfo: "skip" },
          { x: [N, N, 0], y: [0, y, y], mode: "lines", line: dashed(P, P.coral), hoverinfo: "skip" },
          { x: [N], y: [y], mode: "markers", marker: { color: P.coral, size: 12, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" }
        ];
      }, { xaxis: { title: { text: "N" }, range: [0, 1000], fixedrange: true }, yaxis: { title: { text: "log of N" }, range: [0, 10.5], fixedrange: true }, margin: { l: 50, r: 16, t: 10, b: 46 }, dragmode: false }, { staticPlot: true });
      $("lf-eq").innerHTML = box(nm === "e" ? "2.718" : nm, "base B") + op("<sup>raised to</sup>") + box(fmt(y, 3), "exponent y = log of N", "coral") + op("=") + box(String(N), "the number N", "navy");
      $("lf-plot").setAttribute("aria-label", "Logarithm of N in base " + nm + ". For N = " + N + " the logarithm is " + fmt(y, 2) + ".");
    }
    draw();
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 4b. Log-log fit
  // =====================================================================
  function loglog() {
    var SP = [["spadenose", "spadenose"], ["milk", "milk shark"], ["bamboo", "bamboo shark"]], cur = 0, mode = 0, recs = null, R = null;
    Kit.chips($("ll-bar"), SP.map(function (s) { return s[1]; }), 0, function (i) { cur = i; draw(); });
    Kit.chips($("ll-bar2"), ["linear axes", "log y only", "log x and log y"], 0, function (i) { mode = i; draw(); });
    $("ll-plot").textContent = "Loading the catch records...";
    Kit.data("../data/sharks/catch.json").then(function (d) { recs = d.records; $("ll-plot").textContent = ""; draw(); }, function () { $("ll-plot").textContent = "The catch records did not load. Check your connection."; });
    function fit(sp) {
      var rs = recs.filter(function (x) { return x.species === sp; }), L = rs.map(function (x) { return x.length_cm; }), W = rs.map(function (x) { return x.weight_kg; });
      var lx = L.map(Math.log), ly = W.map(Math.log), n = L.length, mx = 0, my = 0, sxy = 0, sxx = 0, j;
      for (j = 0; j < n; j++) { mx += lx[j] / n; my += ly[j] / n; }
      for (j = 0; j < n; j++) { sxy += (lx[j] - mx) * (ly[j] - my); sxx += (lx[j] - mx) * (lx[j] - mx); }
      var b = sxy / sxx;
      return { L: L, W: W, n: n, b: b, lna: my - b * mx, lo: Math.min.apply(null, L), hi: Math.max.apply(null, L), wlo: Math.min.apply(null, W), whi: Math.max.apply(null, W) };
    }
    function draw() {
      if (!recs) return;
      var f = fit(SP[cur][0]), xs = linspace(f.lo, f.hi, 80), line = xs.map(function (x) { return Math.exp(f.lna) * Math.pow(x, f.b); });
      Kit.plot("ll-plot", function (P) {
        return [
          { x: f.L, y: f.W, mode: "markers", marker: { color: P.navy, size: 7, opacity: 0.6 }, hovertemplate: "%{x:.1f} cm, %{y:.3f} kg<extra></extra>" },
          { x: xs, y: line, mode: "lines", line: { color: P.coral, width: 3 }, hoverinfo: "skip" }
        ];
      }, { xaxis: { title: { text: "length, cm" }, type: mode === 2 ? "log" : "linear", fixedrange: true, range: mode === 2 ? [Math.log10(f.lo * 0.95), Math.log10(f.hi * 1.05)] : [f.lo * 0.9, f.hi * 1.05] },
           yaxis: { title: { text: "weight, kg" }, type: mode >= 1 ? "log" : "linear", fixedrange: true, range: mode >= 1 ? [Math.log10(f.wlo * 0.8), Math.log10(f.whi * 1.25)] : [0, f.whi * 1.08] },
           margin: { l: 58, r: 16, t: 10, b: 48 }, dragmode: false });
      $("ll-eq").innerHTML = box("ln W", "log of weight", "navy") + op("=") + box(fmt(f.lna, 2), "ln a, the intercept") + op("+") + box(fmt(f.b, 3), "b, the slope (" + f.n + " records)", "coral") + op("×") + box("ln L", "log of length", "navy");
      $("ll-plot").setAttribute("aria-label", mode === 2 ? "On log axes the made-up " + SP[cur][1] + " records lie along a straight line of slope " + fmt(f.b, 2) + "." : "Weight against length for made-up " + SP[cur][1] + " records. The fitted curve bends upward on these axes.");
    }
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 5. Inverse functions
  // =====================================================================
  function inverse() {
    var F = [
      { name: "2x + 1", f: function (x) { return 2 * x + 1; }, inv: function (y) { return (y - 1) / 2; }, rule: "2x + 1", invRule: "(y − 1) ÷ 2", ok: true },
      { name: "x cubed", f: function (x) { return x * x * x / 4; }, inv: function (y) { return Math.cbrt(4 * y); }, rule: "x³ ÷ 4", invRule: "cube root of 4y", ok: true },
      { name: "e to the x", f: function (x) { return Math.exp(x); }, inv: function (y) { return Math.log(y); }, rule: "e^x", invRule: "ln y", ok: true },
      { name: "sigmoid", f: function (x) { return 1 / (1 + Math.exp(-x)); }, inv: function (y) { return Math.log(y / (1 - y)); }, rule: "1 ÷ (1 + e^−x)", invRule: "ln(y ÷ (1 − y))", ok: true },
      { name: "x squared (all x)", f: function (x) { return x * x / 2; }, inv: null, rule: "x² ÷ 2", ok: false },
      { name: "x squared (x ≥ 0)", f: function (x) { return x * x / 2; }, inv: function (y) { return Math.sqrt(2 * y); }, rule: "x² ÷ 2, x ≥ 0", invRule: "square root of 2y", ok: true, nonneg: true }
    ];
    var cur = 0, sl = null, ctl = $("iv-controls");
    Kit.chips($("iv-bar"), F.map(function (f) { return f.name; }), 0, function (i) { cur = i; build(); });
    function build() {
      var f = F[cur]; ctl.innerHTML = "";
      sl = Kit.slider(ctl, { label: "Probe input x", min: f.nonneg ? 0 : -3, max: 3, step: 0.05, value: f.nonneg ? 1.5 : 1.2 }, draw);
      draw();
    }
    function draw() {
      var f = F[cur], x = Math.round(sl.get() * 100) / 100, y = f.f(x);
      var lo = f.nonneg ? 0 : -4, xs = linspace(lo, 4, 300), ys = xs.map(f.f);
      Kit.plot("iv-plot", function (P) {
        var T = [
          { x: [-4, 4], y: [-4, 4], mode: "lines", line: { color: P.muted, width: 1.5, dash: "dash" }, hoverinfo: "skip" },
          { x: xs, y: ys, mode: "lines", line: { color: P.navy, width: 3.5 }, hoverinfo: "skip" },
          { x: ys, y: xs, mode: "lines", line: { color: P.coral, width: 3.5, dash: f.ok ? "solid" : "dot" }, hoverinfo: "skip" },
          { x: [x, y], y: [y, x], mode: "lines", line: dashed(P), hoverinfo: "skip" },
          { x: [x], y: [y], mode: "markers", marker: { color: P.navy, size: 11, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" },
          { x: [y], y: [x], mode: "markers", marker: { color: P.coral, size: 11, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" },
          { x: [3.2], y: [3.2 + 0.1], mode: "text", text: ["y = x"], textposition: "top left", textfont: { color: P.muted, size: 13 }, hoverinfo: "skip" }
        ];
        return T;
      }, { xaxis: { range: [-4, 4], fixedrange: true, title: { text: "x for f, y for the inverse" }, constrain: "domain" }, yaxis: { range: [-4, 4], fixedrange: true, title: { text: "f(x), or x for the inverse" }, scaleanchor: "x", scaleratio: 1 }, margin: { l: 52, r: 16, t: 10, b: 46 }, dragmode: false }, { staticPlot: true });
      var st = $("iv-status");
      if (f.inv) {
        var back = f.inv(y);
        $("iv-eq").innerHTML = box(fmt(x, 2), "input x", "coral") + op("→") + box(f.rule, "f") + op("→") + box(fmt(y, 3), "f(x)", "navy") + op("→") + box(f.invRule, "f inverse") + op("→") + box(fmt(back, 2), "back to x", "coral");
        st.textContent = "Round trip: " + fmt(x, 2) + " goes into f, gives " + fmt(y, 3) + ", and the inverse returns " + fmt(back, 2) + ".";
      } else {
        $("iv-eq").innerHTML = box(fmt(x, 2), "input x", "coral") + op("→") + box(f.rule, "f") + op("→") + box(fmt(y, 3), "f(x)", "navy") + op("→") + box("?", "which input?", "coral");
        st.textContent = "The inputs " + fmt(x, 2) + " and " + fmt(-x, 2) + " both give " + fmt(y, 3) + ". A function cannot send " + fmt(y, 3) + " back to two places, so there is no inverse unless you restrict the domain.";
      }
      $("iv-plot").setAttribute("aria-label", "The function " + f.name + " in navy and its mirror image across the line y equals x in coral" + (f.ok ? "." : ". The mirror bends back on itself, so it fails the vertical line test."));
    }
    build();
    Kit.onTheme(draw);
  }
})();
