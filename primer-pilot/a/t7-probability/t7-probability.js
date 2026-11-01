// Probability and statistics (chapter 0.7): eleven figures, built in this order.
//   spreadFig   drag dots: mean, deviations, standard deviation (D3)
//   normalFig   bell curve with mean and spread sliders and standardising (Plotly, shark catch records)
//   bayesFig    icon array of 1,000 patients: prevalence, sensitivity, specificity (D3)
//   softmaxFig  softmax bars with a temperature slider and sampling (D3)
//   crossFig    cross-entropy: probability bars and the -ln p curve (Plotly)
//   emaFig      exponential moving average with a beta slider (Plotly)
//   noiseFig    running mean of dive counts, with the 1/sqrt(n) funnel (Plotly)
//   ciFig       bootstrap of a test score, and 100 repeated studies (Plotly)
//   testFig     is model B better than A: the coin-flip test (Plotly)
//   causeFig    ice cream and shark sightings: confounding and intervention (Plotly)
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    spreadFig(); normalFig(); bayesFig(); softmaxFig(); crossFig(); emaFig(); noiseFig(); ciFig(); testFig(); causeFig();
  });

  // ---------- helpers ----------
  var NAMES = ["Spadenose", "Milk", "Grey bamboo", "Hammerhead", "Ganges"];
  var SCORES = [2.4, 1.2, 0.5, -0.4, -1.8];
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function sum(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s; }
  function mean(a) { return sum(a) / a.length; }
  function sd(a, ddof) { var m = mean(a); return Math.sqrt(sum(a.map(function (v) { return (v - m) * (v - m); })) / (a.length - (ddof || 0))); }
  function median(a) { var b = a.slice().sort(function (x, y) { return x - y; }), n = b.length; return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2; }
  function pctile(sorted, q) { var i = (sorted.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i); return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo); }
  function erf(x) { var s = x < 0 ? -1 : 1, t; x = Math.abs(x); t = 1 / (1 + 0.3275911 * x); return s * (1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x)); }
  function cdf(z) { return 0.5 * (1 + erf(z / Math.SQRT2)); }
  function npdf(x, m, s) { return Math.exp(-0.5 * Math.pow((x - m) / s, 2)) / (s * Math.sqrt(2 * Math.PI)); }
  function softmax(z, T) { var mx = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp((v - mx) / T); }), s = sum(e); return e.map(function (v) { return v / s; }); }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { var C = hexRGB(c); return "rgba(" + C[0] + "," + C[1] + "," + C[2] + "," + a + ")"; }
  function mixc(a, b, t) { var A = hexRGB(a), B = hexRGB(b); return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }).join(",") + ")"; }
  function styleAxis(g, P) { g.selectAll("path,line").attr("stroke", P.muted); g.selectAll("text").attr("fill", P.muted).attr("font-size", 12.5); }
  function logChoose(n, k) { var s = 0, i; for (i = 1; i <= k; i++) s += Math.log(n - k + i) - Math.log(i); return s; }
  function box(cls, v, lab) { return '<span class="t ' + cls + '"><b>' + v + '</b><i>' + lab + '</i></span>'; }
  function op(t) { return '<span class="op">' + t + '</span>'; }
  function cat(r, p) { var u = r(), c = 0, i; for (i = 0; i < p.length; i++) { c += p[i]; if (u < c) return i; } return p.length - 1; }

  // ---------- 1. mean and spread: drag the dots ----------
  function spreadFig() {
    var ORIG = [32.7, 33.2, 33.4, 33.9, 35.4, 36.8, 37.3, 44.1, 48.3], data = ORIG.slice(), X0 = 25, X1 = 100;
    var node = Kit.$("sp-svg"), svg = d3.select(node), els = {}, sx, W, rows = [82, 108, 134, 160];
    Kit.button(Kit.$("sp-bar"), "Reset", function () { data = ORIG.slice(); update(); });
    Kit.button(Kit.$("sp-bar"), "Add a typing slip: 48.3 becomes 95", function () { data[8] = 95; update(); });
    function build() {
      var P = Kit.palette(); W = Kit.width(node);
      var H = 252, ml = 22, mr = 22;
      sx = d3.scaleLinear().domain([X0, X1]).range([ml, W - mr]);
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      els.band = svg.append("rect").attr("y", 52).attr("height", 144).attr("fill", P.mark).attr("rx", 4);
      svg.append("text").attr("x", ml).attr("y", 20).attr("font-size", 13.5).attr("fill", P.muted).text(W < 520 ? "Nine spadenose lengths (made-up records)" : "Lengths of nine spadenose sharks (made-up catch records)");
      var ax = svg.append("g").attr("transform", "translate(0,206)").call(d3.axisBottom(sx).ticks(W < 500 ? 5 : 9).tickSize(5)); styleAxis(ax, P);
      svg.append("text").attr("x", W - mr).attr("y", 244).attr("text-anchor", "end").attr("font-size", 13.5).attr("fill", P.ink2).text("length, cm");
      els.seg = data.map(function (d, i) { return svg.append("line").attr("stroke", P.coral).attr("stroke-width", 2.2).attr("opacity", 0.75).attr("y1", rows[i % 4]).attr("y2", rows[i % 4]); });
      els.mean = svg.append("line").attr("y1", 50).attr("y2", 206).attr("stroke", P.teal).attr("stroke-width", 3);
      els.meanT = svg.append("text").attr("y", 44).attr("font-size", 14).attr("font-weight", 600).attr("fill", P.teal).attr("text-anchor", "middle");
      els.bandT = svg.append("text").attr("y", 68).attr("font-size", 12.5).attr("fill", P.ink2).attr("text-anchor", "start");
      els.med = svg.append("path").attr("d", "M0,0 L-7,-11 L7,-11 Z").attr("fill", P.sand).attr("transform", "translate(0,206)");
      els.medT = svg.append("text").attr("y", 198).attr("font-size", 12.5).attr("fill", P.sand).attr("text-anchor", "middle").attr("font-weight", 600);
      els.dot = data.map(function (d, i) {
        return svg.append("circle").attr("cy", rows[i % 4]).attr("r", W < 520 ? 6 : 8).attr("fill", P.navy).attr("stroke", P.bg).attr("stroke-width", 2).attr("tabindex", 0).attr("role", "slider")
          .attr("aria-label", "Shark " + (i + 1) + " length, arrow keys move it").attr("aria-valuemin", X0).attr("aria-valuemax", X1).style("cursor", "ew-resize")
          .on("keydown", function (e) { if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); data[i] = clamp(data[i] + (e.key === "ArrowLeft" ? -0.5 : 0.5), X0, X1); update(); } })
          .call(d3.drag().on("start", function () { d3.select(this).attr("fill", P.coral); }).on("drag", function (e) { data[i] = clamp(sx.invert(e.x), X0, X1); update(); }).on("end", function () { d3.select(this).attr("fill", P.navy); }));
      });
      update();
    }
    function update() {
      var m = mean(data), s = sd(data, 0), md = median(data), ss = sum(data.map(function (v) { return (v - m) * (v - m); })), x = function (v) { return sx(v); };
      els.band.attr("x", x(m - s)).attr("width", Math.max(0, x(m + s) - x(m - s)));
      els.mean.attr("x1", x(m)).attr("x2", x(m)); els.meanT.attr("x", x(m)).text("mean " + Kit.fmt(m, 1));
      els.bandT.attr("x", x(m - s) + 5).text("±1 SD");
      els.med.attr("transform", "translate(" + x(md) + ",206)"); els.medT.attr("x", x(md)).text("median " + Kit.fmt(md, 1));
      data.forEach(function (v, i) { els.seg[i].attr("x1", x(v)).attr("x2", x(m)); els.dot[i].attr("cx", x(v)).attr("aria-valuenow", Kit.fmt(v, 1)); });
      Kit.$("sp-eqn").innerHTML = box("c-coral", Kit.fmt(ss, 1), "sum of squared deviations, cm²") + op("&divide;") + box("", String(data.length), "n, the number of sharks") + op("=") +
        box("c-navy", Kit.fmt(ss / data.length, 1), "variance, cm²") + op("&rarr; square root &rarr;") + box("c-teal", Kit.fmt(s, 2), "standard deviation, cm");
      Kit.setText("sp-mean", Kit.fmt(m, 2)); Kit.setText("sp-med", Kit.fmt(md, 2)); Kit.setText("sp-sd", Kit.fmt(s, 2));
    }
    build(); Kit.onTheme(build); window.addEventListener("resize", Kit.debounce(build));
  }

  // ---------- 2. the normal curve and standardising ----------
  function normalFig() {
    var C = [
      { name: "Length, cm", key: "length_cm", lo: 20, hi: 60, mu: [20, 60, 0.1], sg: [2, 20, 0.1], xs: [10, 70, 0.1], d: 1, axis: [0, 70] },
      { name: "Girth, cm", key: "girth_cm", lo: 4, hi: 18, mu: [5, 17, 0.1], sg: [0.5, 6, 0.1], xs: [3, 19, 0.1], d: 1, axis: [0, 20] },
      { name: "Weight, kg", key: "weight_kg", lo: 0, hi: 0.8, mu: [0, 0.6, 0.005], sg: [0.02, 0.4, 0.005], xs: [0, 0.8, 0.005], d: 3, axis: [-0.25, 0.8] }
    ];
    var cur = 0, z = false, vals = null, fit = [];
    var sM = Kit.slider(Kit.$("nm-controls"), { label: "Mean μ", min: 20, max: 60, step: 0.1, value: 39.1, show: function (v) { return Kit.fmt(v, C[cur].d); } }, draw);
    var sS = Kit.slider(Kit.$("nm-controls"), { label: "Standard deviation σ", min: 2, max: 20, step: 0.1, value: 8.6, show: function (v) { return Kit.fmt(v, C[cur].d); } }, draw);
    var sX = Kit.slider(Kit.$("nm-controls"), { label: "A shark's value x", min: 10, max: 70, step: 0.1, value: 52, show: function (v) { return Kit.fmt(v, C[cur].d); } }, draw);
    Kit.chips(Kit.$("nm-cols"), C.map(function (c) { return c.name; }), 0, function (i) { cur = i; setup(); draw(); });
    Kit.chips(Kit.$("nm-mode"), ["Original units", "Standardised (z-scores)"], 0, function (i) { z = i === 1; draw(); });
    Kit.button(Kit.$("nm-mode"), "Fit to the data", function () { sM.set(fit[0]); sS.set(fit[1]); draw(); });
    function setup() {
      var c = C[cur], v = vals ? vals[c.key] : [1];
      fit = [mean(v), sd(v, 0)];
      [[sM, c.mu, fit[0]], [sS, c.sg, fit[1]], [sX, c.xs, mean(v) + 1.5 * sd(v, 0)]].forEach(function (a) { a[0].input.min = a[1][0]; a[0].input.max = a[1][1]; a[0].input.step = a[1][2]; a[0].set(Math.round(a[2] / a[1][2]) * a[1][2]); });
    }
    function draw() {
      if (!vals) return;
      var c = C[cur], v = vals[c.key], mu = sM.get(), s = sS.get(), x = sX.get(), zz = (x - mu) / s, n = v.length;
      var inside = function (k) { return v.filter(function (a) { return Math.abs(a - mu) <= k * s; }).length / n; };
      var i1 = inside(1), i2 = inside(2);
      Kit.plot("nm-plot", function (P) {
        var tr = [], bars, curve, xs, f;
        if (!z) {
          var lo = c.lo, hi = c.hi, nb = 14, w = (hi - lo) / nb, cnt = new Array(nb).fill(0);
          v.forEach(function (a) { var k = clamp(Math.floor((a - lo) / w), 0, nb - 1); cnt[k]++; });
          bars = { x: cnt.map(function (_, k) { return lo + (k + 0.5) * w; }), y: cnt.map(function (q) { return q / (n * w); }), w: w };
          xs = lin(c.axis[0], c.axis[1], 241); f = function (a) { return npdf(a, mu, s); };
        } else {
          var nb2 = 16, w2 = 0.5, cnt2 = new Array(nb2).fill(0);
          v.forEach(function (a) { var k = Math.floor(((a - mu) / s + 4) / w2); if (k >= 0 && k < nb2) cnt2[k]++; });
          bars = { x: cnt2.map(function (_, k) { return -4 + (k + 0.5) * w2; }), y: cnt2.map(function (q) { return q / (n * w2); }), w: w2 };
          xs = lin(-4, 4, 241); f = function (a) { return npdf(a, 0, 1); };
        }
        var m0 = z ? 0 : mu, s0 = z ? 1 : s, band = function (k) { var a = xs.filter(function (q) { return Math.abs(q - m0) <= k * s0; }); return { x: a, y: a.map(f) }; };
        var b2 = band(2), b1 = band(1);
        tr.push({ type: "bar", x: bars.x, y: bars.y, width: bars.w * 0.94, marker: { color: rgba(P.navy, 0.5) }, hovertemplate: "density %{y:.3f}<extra></extra>" });
        tr.push({ x: b2.x, y: b2.y, mode: "lines", fill: "tozeroy", fillcolor: rgba(P.coral, 0.12), line: { width: 0 }, hoverinfo: "skip" });
        tr.push({ x: b1.x, y: b1.y, mode: "lines", fill: "tozeroy", fillcolor: rgba(P.coral, 0.22), line: { width: 0 }, hoverinfo: "skip" });
        tr.push({ x: xs, y: xs.map(f), mode: "lines", line: { color: P.coral, width: 3 }, hoverinfo: "skip" });
        var px = z ? zz : x;
        tr.push({ x: [px], y: [f(px)], mode: "markers", marker: { color: P.teal, size: 12, line: { color: P.bg, width: 2.5 } }, hoverinfo: "skip" });
        return tr;
      }, function (P) {
        var px = z ? zz : x, ymax = z ? 0.62 : null;
        return { dragmode: false, margin: { l: 70, r: 14, t: 28, b: 52 },
          xaxis: { range: z ? [-4, 4] : c.axis, title: { text: z ? "z-score: standard deviations from the mean" : c.name, standoff: 6 } },
          yaxis: { title: { text: "density", standoff: 8 }, rangemode: "tozero", range: ymax ? [0, ymax] : undefined },
          shapes: [{ type: "line", x0: px, x1: px, y0: 0, y1: 1, yref: "paper", line: { color: P.teal, width: 1.8, dash: "dot" } }],
          annotations: [{ x: px, y: 1, yref: "paper", text: z ? "z = " + Kit.fmt(zz, 2) : "x = " + Kit.fmt(x, c.d), showarrow: false, yanchor: "bottom", xanchor: px > (z ? 2 : (c.axis[0] + c.axis[1]) * 0.6) ? "right" : "left", font: { color: P.teal, size: 14 } }] };
      });
      Kit.$("nm-eqn").innerHTML = box("c-teal", Kit.fmt(x, c.d), "x, the value") + op("&minus;") + box("c-coral", Kit.fmt(mu, c.d), "&mu;, the mean") + op("&divide;") + box("c-navy", Kit.fmt(s, c.d), "&sigma;, the standard deviation") + op("=") + box("", Kit.fmt(zz, 2), "z, the z-score");
      Kit.setText("nm-share", "Within 1 SD of the mean: data " + Kit.pct(i1, 1) + ", normal curve 68.3%. Within 2 SD: data " + Kit.pct(i2, 1) + ", normal curve 95.4%. (n = " + n + " spadenose sharks, made-up records.)");
    }
    Kit.data("../data/sharks/catch.json").then(function (J) {
      vals = {};
      ["length_cm", "girth_cm", "weight_kg"].forEach(function (k) { vals[k] = J.records.filter(function (r) { return r.species === "spadenose"; }).map(function (r) { return r[k]; }); });
      setup(); draw();
    }).catch(function () { Kit.$("nm-plot").textContent = "The catch records did not load. Check your connection."; });
  }

  // ---------- 3. icon array: base rates and Bayes' rule ----------
  function bayesFig() {
    var N = 1000, mode = 2, node = Kit.$("by-svg"), svg = d3.select(node), icons = [], keyEls = [], layout = {};
    var sp = Kit.slider(Kit.$("by-controls"), { label: "Share of patients with diabetes (prevalence)", min: 0.5, max: 50, step: 0.1, value: 8, show: function (v) { return Kit.fmt(v, 1) + "%"; } }, update);
    var ss = Kit.slider(Kit.$("by-controls"), { label: "Sensitivity of the test", min: 50, max: 100, step: 0.1, value: 90, show: function (v) { return Kit.fmt(v, 1) + "%"; } }, update);
    var sc = Kit.slider(Kit.$("by-controls"), { label: "Specificity of the test", min: 50, max: 100, step: 0.1, value: 90, show: function (v) { return Kit.fmt(v, 1) + "%"; } }, update);
    Kit.chips(Kit.$("by-preset"), ["Made-up clinic test", "Pima women, glucose rule", "Rare condition"], 0, function (i) {
      var P = [[8, 90, 90], [34.9, 50.8, 87.5], [1, 99, 95]][i]; sp.set(P[0]); ss.set(P[1]); sc.set(P[2]); update();
    });
    Kit.chips(Kit.$("by-mode"), ["All patients", "Who has diabetes", "Test result", "Only the flagged"], 2, function (i) { mode = i; style(); });
    function counts() {
      var sick = Math.round(N * sp.get() / 100), tp = Math.round(sick * ss.get() / 100), fn = sick - tp, healthy = N - sick, fp = Math.round(healthy * (1 - sc.get() / 100)), tn = healthy - fp;
      return { sick: sick, tp: tp, fn: fn, healthy: healthy, fp: fp, tn: tn };
    }
    function kind(k, c) { return k < c.tp ? 0 : k < c.sick ? 1 : k < c.sick + c.fp ? 2 : 3; }   // 0 TP, 1 FN, 2 FP, 3 TN
    function build() {
      var P = Kit.palette(), W = Kit.width(node), cols = W < 520 ? 25 : 50, rowsN = N / cols, m = 6, cell = (W - 2 * m) / cols, top = 8, gridH = rowsN * cell;
      layout = { cols: cols, cell: cell, m: m, top: top };
      var H = top + gridH + (W < 520 ? 92 : 56);
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      icons = [];
      for (var k = 0; k < N; k++) icons.push(svg.append("circle").attr("cx", m + (k % cols + 0.5) * cell).attr("cy", top + (Math.floor(k / cols) + 0.5) * cell).attr("r", Math.max(2.2, cell * 0.36)));
      // the key sits directly under the array
      var ky = top + gridH + 22, items = [["has it, flagged", 0], ["has it, missed", 1], ["no diabetes, flagged", 2], ["no diabetes, cleared", 3]];
      keyEls = items.map(function (it, i) {
        var col = W < 520 ? i % 2 : i, row = W < 520 ? Math.floor(i / 2) : 0, x0 = m + col * (W - 2 * m) / (W < 520 ? 2 : 4), y0 = ky + row * 28;
        var g = svg.append("g").attr("transform", "translate(" + x0 + "," + y0 + ")");
        g.append("circle").attr("cx", 7).attr("cy", -4).attr("r", 6).attr("data-k", it[1]);
        g.append("text").attr("x", 20).attr("y", 0).attr("font-size", 13.5).attr("fill", P.ink2);
        return { g: g, k: it[1], label: it[0] };
      });
      update();
    }
    function style() {
      var P = Kit.palette(), c = counts();
      icons.forEach(function (ic, k) {
        var t = kind(k, c), sick = t < 2, flagged = t === 0 || t === 2, col = sick ? P.coral : P.navy;
        var fill = P.muted, stroke = "none", op = 0.55;
        if (mode === 1) { fill = col; op = 0.9; }
        if (mode >= 2) { fill = flagged ? col : "none"; stroke = col; op = flagged ? 0.95 : 0.75; }
        if (mode === 3 && !flagged) op = 0.1;
        ic.attr("fill", fill).attr("stroke", stroke).attr("stroke-width", 1.6).attr("opacity", op);
      });
      keyEls.forEach(function (e) {
        var t = e.k, sick = t < 2, flagged = t === 0 || t === 2, col = sick ? P.coral : P.navy, ci = e.g.select("circle");
        ci.attr("fill", mode >= 2 ? (flagged ? col : "none") : (mode === 1 ? col : P.muted)).attr("stroke", mode >= 2 ? col : "none").attr("stroke-width", 1.8).attr("opacity", mode === 0 ? 0.55 : 0.95);
        e.g.select("text").text(e.label + " " + [c.tp, c.fn, c.fp, c.tn][t]);
      });
    }
    function update() {
      var c = counts(), flagged = c.tp + c.fp, ppv = flagged ? c.tp / flagged : 0, p = sp.get() / 100, se = ss.get() / 100, spc = sc.get() / 100;
      style();
      Kit.$("by-eqn").innerHTML = box("c-coral", String(c.tp), "flagged and have it") + op("&divide;") + box("", String(flagged), "all flagged (" + c.tp + " + " + c.fp + ")") + op("=") + box("c-teal", Kit.pct(ppv, 1), "chance a flagged patient has it");
      Kit.$("by-eqn2").innerHTML = box("c-coral", Kit.fmt(se * p, 4), "P(flagged and sick) = " + Kit.fmt(se, 3) + " &times; " + Kit.fmt(p, 3)) + op("&divide;") +
        box("", Kit.fmt(se * p + (1 - spc) * (1 - p), 4), "P(flagged) = " + Kit.fmt(se * p, 4) + " + " + Kit.fmt((1 - spc) * (1 - p), 4)) + op("=") + box("c-teal", Kit.pct((se * p) / (se * p + (1 - spc) * (1 - p)), 1), "same answer by Bayes' rule");
      Kit.setText("by-status", "Of 1,000 patients, " + c.sick + " have diabetes. The test flags " + flagged + ": " + c.tp + " who have it and " + c.fp + " who do not.");
    }
    build(); Kit.onTheme(build); window.addEventListener("resize", Kit.debounce(build));
  }

  // ---------- 4. softmax with temperature ----------
  function softmaxFig() {
    var node = Kit.$("sm-svg"), svg = d3.select(node), T = 1, sel = 0, tally = [0, 0, 0, 0, 0], drawn = 0, last = "", seed = 5, els = [], W, rowH = 46;
    var sT = Kit.slider(Kit.$("sm-controls"), { label: "Temperature T", min: 0.1, max: 5, step: 0.05, value: 1, show: function (v) { return Kit.fmt(v, 2); } }, function (v) { T = v; update(); });
    Kit.button(Kit.$("sm-bar"), "Draw 1", function () { draw(1); });
    Kit.button(Kit.$("sm-bar"), "Draw 100", function () { draw(100); });
    Kit.button(Kit.$("sm-bar"), "Clear draws", function () { tally = [0, 0, 0, 0, 0]; drawn = 0; last = ""; update(); });
    function draw(k) { var p = softmax(SCORES, T), r = Kit.rng(seed++ * 7919), i; for (i = 0; i < k; i++) { var c = cat(r, p); tally[c]++; last = NAMES[c]; } drawn += k; update(); }
    function build() {
      var P = Kit.palette(); W = Kit.width(node);
      var H = 24 + 5 * rowH + 8, nameW = W < 520 ? 96 : 150, barX = nameW + 56, barW = W - barX - 62;
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      svg.append("text").attr("x", 0).attr("y", 14).attr("font-size", 13).attr("fill", P.muted).text("species");
      svg.append("text").attr("x", nameW + 8).attr("y", 14).attr("font-size", 13).attr("fill", P.muted).text("score z");
      svg.append("text").attr("x", barX).attr("y", 14).attr("font-size", 13).attr("fill", P.muted).text(W < 520 ? "probability" : "probability (bar) and share of your draws (outline)");
      els = NAMES.map(function (nm, i) {
        var y = 24 + i * rowH, g = svg.append("g").attr("tabindex", 0).attr("role", "button").attr("aria-label", "Show the working for " + nm).style("cursor", "pointer");
        g.on("click", function () { sel = i; update(); }).on("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = i; update(); } });
        var hit = g.append("rect").attr("x", 0).attr("y", y).attr("width", W).attr("height", rowH - 4).attr("rx", 6).attr("fill", "none");
        g.append("text").attr("x", 4).attr("y", y + 26).attr("font-size", 15).attr("fill", P.ink).text(nm);
        g.append("text").attr("x", nameW + 8).attr("y", y + 26).attr("font-size", 14).attr("fill", P.ink2).text(Kit.fmt(SCORES[i], 1));
        g.append("rect").attr("x", barX).attr("y", y + 8).attr("height", 24).attr("width", barW).attr("fill", P.rule).attr("rx", 3);
        var bar = g.append("rect").attr("x", barX).attr("y", y + 8).attr("height", 24).attr("rx", 3).attr("fill", P.navy);
        var out = g.append("rect").attr("x", barX).attr("y", y + 5).attr("height", 30).attr("rx", 3).attr("fill", "none").attr("stroke", P.ink2).attr("stroke-width", 1.6).attr("stroke-dasharray", "4 3");
        var lab = g.append("text").attr("y", y + 26).attr("font-size", 14.5).attr("font-weight", 600).attr("fill", P.ink);
        return { hit: hit, bar: bar, out: out, lab: lab, barX: barX, barW: barW, y: y };
      });
      update();
    }
    function update() {
      var P = Kit.palette(), p = softmax(SCORES, T), top = p.indexOf(Math.max.apply(null, p));
      els.forEach(function (e, i) {
        e.bar.attr("width", Math.max(1, p[i] * e.barW)).attr("fill", i === top ? P.teal : P.navy);
        e.out.attr("width", drawn ? Math.max(2, tally[i] / drawn * e.barW) : 0).attr("opacity", drawn ? 1 : 0);
        e.lab.attr("x", e.barX + Math.max(1, p[i] * e.barW) + 8).text(Kit.pct(p[i], p[i] < 0.01 ? 2 : 1));
        e.hit.attr("stroke", i === sel ? P.coral : "none").attr("stroke-width", 1.6);
      });
      var ex = SCORES.map(function (v) { return Math.exp((v - Math.max.apply(null, SCORES)) / T); }), tot = sum(ex), k = sel;
      Kit.$("sm-eqn").innerHTML = box("c-navy", Kit.fmt(SCORES[k] / T, 2), "z &divide; T for " + NAMES[k]) + op("&rarr; e<sup>x</sup> &rarr;") + box("c-sand", Kit.fmt(ex[k] * Math.exp(Math.max.apply(null, SCORES) / T), 2), "e<sup>z/T</sup>") + op("&divide;") +
        box("", Kit.fmt(tot * Math.exp(Math.max.apply(null, SCORES) / T), 2), "total of e<sup>z/T</sup>, all five") + op("=") + box("c-teal", Kit.pct(p[k], 1), "probability of " + NAMES[k]);
      Kit.setText("sm-status", drawn ? drawn + " draw" + (drawn === 1 ? "" : "s") + " so far" + (last ? ". Last draw: " + last + "." : ".") : "Click a species to see its working. Draw species from the probabilities to compare the outline with the bar.");
    }
    build(); Kit.onTheme(build); window.addEventListener("resize", Kit.debounce(build));
  }

  // ---------- 5. cross-entropy: how the loss reacts to the probability ----------
  function crossFig() {
    var tru = 1, sc = SCORES[1];
    var s = Kit.slider(Kit.$("ce-controls"), { label: "Score the model gives the true species", min: -4, max: 6, step: 0.1, value: sc, show: function (v) { return Kit.fmt(v, 1); } }, function (v) { sc = v; draw(); });
    Kit.chips(Kit.$("ce-true"), NAMES, 1, function (i) { tru = i; sc = SCORES[i]; s.set(sc); draw(); });
    Kit.button(Kit.$("ce-bar"), "Make it confidently wrong", function () { sc = -4; s.set(sc); draw(); });
    Kit.button(Kit.$("ce-bar"), "Make it confidently right", function () { sc = 6; s.set(sc); draw(); });
    function draw() {
      var z = SCORES.slice(); z[tru] = sc; var p = softmax(z, 1), pt = p[tru], loss = -Math.log(pt), top = p.indexOf(Math.max.apply(null, p));
      Kit.plot("ce-bars", function (P) {
        return [{ type: "bar", orientation: "h", y: NAMES, x: p, marker: { color: NAMES.map(function (_, i) { return i === tru ? P.teal : P.navy; }) }, text: p.map(function (v) { return Kit.pct(v, 1); }), textposition: "outside", cliponaxis: false, hoverinfo: "skip" }];
      }, function () { return { dragmode: false, margin: { l: 96, r: 52, t: 10, b: 44 }, xaxis: { range: [0, 1.12], title: { text: "probability the model gives" } }, yaxis: { autorange: "reversed" } }; });
      var ps = lin(0.004, 1, 250);
      Kit.plot("ce-curve", function (P) {
        return [{ x: ps, y: ps.map(function (q) { return -Math.log(q); }), mode: "lines", line: { color: P.navy, width: 3 }, hoverinfo: "skip" },
          { x: [pt], y: [loss], mode: "markers", marker: { color: P.coral, size: 13, line: { color: P.bg, width: 2.5 } }, hoverinfo: "skip" }];
      }, function (P) {
        return { dragmode: false, margin: { l: 52, r: 14, t: 10, b: 52 }, xaxis: { range: [0, 1], title: { text: "probability given to the true species, p", standoff: 6 } }, yaxis: { range: [0, 5.6], title: { text: "loss = −ln p" } },
          annotations: [{ x: 0.06, y: 4.2, text: "confident and wrong:<br>loss is large", showarrow: false, xanchor: "left", font: { color: P.coral, size: 13 } },
            { x: 0.97, y: 1.5, text: "confident and right:<br>loss near 0", showarrow: false, xanchor: "right", font: { color: P.teal, size: 13 } }] };
      });
      Kit.$("ce-eqn").innerHTML = box("c-teal", Kit.fmt(pt, 3), "p, the probability for " + NAMES[tru]) + op("&rarr; &minus;ln &rarr;") + box("c-coral", Kit.fmt(loss, 3), "loss = &minus;ln p") + op("and accuracy says") + box("", top === tru ? "right" : "wrong", "top pick is " + NAMES[top]);
    }
    draw();
  }

  // ---------- 6. exponential moving average ----------
  function emaFig() {
    var T = 120, r = Kit.rng(11), series = [], run = [], i, acc = 0;
    for (i = 1; i <= T; i++) { series.push((i <= 60 ? 28.5 : 26.5) + 0.6 * Kit.gauss(r)); acc += series[i - 1]; run.push(acc / i); }
    var zero = false;
    var sb = Kit.slider(Kit.$("em-controls"), { label: "Decay β", min: 0, max: 0.99, step: 0.01, value: 0.9, show: function (v) { return Kit.fmt(v, 2); } }, draw);
    var sd_ = Kit.slider(Kit.$("em-controls"), { label: "Day t for the working", min: 1, max: T, step: 1, value: 80, digits: 0 }, draw);
    Kit.chips(Kit.$("em-start"), ["Start from the first reading", "Start at 0 (as Adam does)"], 0, function (k) { zero = k === 1; draw(); });
    function ema(b) { var o = [], m = zero ? 0 : series[0], k; for (k = 0; k < T; k++) { m = (k === 0 && !zero) ? series[0] : b * m + (1 - b) * series[k]; o.push(m); } return o; }
    function draw() {
      var b = sb.get(), t = sd_.get(), e = ema(b), days = lin(1, T, T), win = b < 0.995 ? 1 / (1 - b) : Infinity;
      Kit.plot("em-plot", function (P) {
        return [
          { x: days, y: series, mode: "markers", marker: { color: rgba(P.navy, 0.55), size: 6 }, hovertemplate: "day %{x}: %{y:.2f} °C<extra></extra>" },
          { x: days, y: run, mode: "lines", line: { color: P.sand, width: 3 }, hoverinfo: "skip" },
          { x: days, y: e, mode: "lines", line: { color: P.coral, width: 3.4 }, hoverinfo: "skip" },
          { x: [t], y: [e[t - 1]], mode: "markers", marker: { color: P.coral, size: 11, line: { color: P.bg, width: 2.5 } }, hoverinfo: "skip" }];
      }, function (P) {
        return { dragmode: false, margin: { l: 52, r: 14, t: 12, b: 50 }, xaxis: { range: [0, T + 1], title: { text: "day", standoff: 6 } }, yaxis: { range: [24.6, 30.4], title: { text: "water temperature, °C" } },
          shapes: [{ type: "line", x0: t, x1: t, y0: 0, y1: 1, yref: "paper", line: { color: P.muted, width: 1.2, dash: "dot" } }],
          annotations: [
            { x: 108, y: 26.0, text: "EMA", showarrow: false, font: { color: P.coral, size: 14 }, yshift: -14 },
            { x: 108, y: run[107], text: "running mean of all days", showarrow: false, xanchor: "right", yshift: 20, font: { color: P.sand, size: 13 } },
            { x: 8, y: 29.9, text: "readings (made-up)", showarrow: false, xanchor: "left", font: { color: P.navy, size: 13 } }].concat(zero ? [{ x: 14, y: 25.0, text: "average starts at 0 and climbs: biased low", showarrow: false, xanchor: "left", font: { color: P.coral, size: 13 } }] : []) };
      });
      var ks = lin(0, 39, 40);
      Kit.plot("em-w", function (P) { return [{ type: "bar", x: ks, y: ks.map(function (k) { return (1 - b) * Math.pow(b, k); }), marker: { color: P.teal }, hovertemplate: "%{x} days ago: weight %{y:.3f}<extra></extra>" }]; },
        function () { return { dragmode: false, margin: { l: 52, r: 14, t: 8, b: 50 }, xaxis: { title: { text: "how many days ago a reading was taken", standoff: 6 } }, yaxis: { title: { text: "weight" }, rangemode: "tozero" } }; });
      var prev = t === 1 ? (zero ? 0 : series[0]) : e[t - 2];
      Kit.$("em-eqn").innerHTML = box("c-navy", Kit.fmt(b, 2), "&beta;") + op("&times;") + box("c-sand", Kit.fmt(prev, 2), "average on day " + (t - 1)) + op("+") + box("", Kit.fmt(1 - b, 2), "1 &minus; &beta;") + op("&times;") + box("c-navy", Kit.fmt(series[t - 1], 2), "reading on day " + t) + op("=") + box("c-coral", Kit.fmt(e[t - 1], 2), "average on day " + t);
      Kit.setText("em-status", b === 0 ? "With β = 0 the average is just today's reading." : "Effective window: about " + (isFinite(win) ? Math.round(win) : "all") + " days (1 ÷ (1 − β)).");
    }
    draw();
  }

  // ---------- 7. expectation and sampling noise ----------
  function noiseFig() {
    var P_ = [0.5, 0.3, 0.15, 0.05], E = 0.75, SDX = Math.sqrt(0.7875), M = 2000, seed = 1, runs = [];
    var sN = Kit.slider(Kit.$("ns-controls"), { label: "Number of dives", min: 1, max: 3.3, step: 0.01, value: 2, show: function (v) { return String(Math.round(Math.pow(10, v))); } }, draw);
    Kit.button(Kit.$("ns-bar"), "New runs", function () { seed++; gen(); draw(); });
    function gen() {
      runs = [];
      for (var k = 0; k < 8; k++) { var r = Kit.rng(seed * 101 + k * 17), a = 0, o = [], i; for (i = 1; i <= M; i++) { a += cat(r, P_); o.push(a / i); } runs.push(o); }
    }
    function draw() {
      var N = Math.round(Math.pow(10, sN.get())), ns = [], i; for (i = 1; i <= N; i++) ns.push(i);
      var up = ns.map(function (n) { return E + 1.96 * SDX / Math.sqrt(n); }), lo = ns.map(function (n) { return E - 1.96 * SDX / Math.sqrt(n); });
      Kit.plot("ns-plot", function (P) {
        var tr = [{ x: ns, y: up, mode: "lines", line: { color: rgba(P.teal, 0.0), width: 0 }, hoverinfo: "skip" },
          { x: ns, y: lo, mode: "lines", fill: "tonexty", fillcolor: rgba(P.teal, 0.16), line: { color: P.teal, width: 1.4, dash: "dot" }, hoverinfo: "skip" },
          { x: ns, y: up, mode: "lines", line: { color: P.teal, width: 1.4, dash: "dot" }, hoverinfo: "skip" }];
        runs.forEach(function (o, k) { tr.push({ x: ns, y: o.slice(0, N), mode: "lines", line: { color: k === 0 ? P.coral : rgba(P.navy, 0.55), width: k === 0 ? 2.6 : 1.4 }, hoverinfo: "skip" }); });
        return tr;
      }, function (P) {
        return { dragmode: false, margin: { l: 52, r: 14, t: 12, b: 52 }, xaxis: { type: "log", range: [0, Math.log10(Math.max(N, 10))], tickvals: [1, 3, 10, 30, 100, 300, 1000, 2000], ticktext: ["1", "3", "10", "30", "100", "300", "1,000", "2,000"], title: { text: "number of dives so far (log scale)", standoff: 6 } }, yaxis: { range: [0, 2.2], title: { text: "average sharks seen per dive" } },
          shapes: [{ type: "line", x0: 0, x1: 1, xref: "paper", y0: E, y1: E, line: { color: P.ink2, width: 1.6 } }],
          annotations: [{ xref: "paper", x: 0.99, y: E, text: "expectation 0.75", showarrow: false, xanchor: "right", yshift: 12, bgcolor: rgba(P.bg, 0.85), font: { color: P.ink2, size: 13 } },
            { xref: "paper", x: 0.99, y: 2.1, text: "dotted funnel: where 95% of averages fall", showarrow: false, xanchor: "right", font: { color: P.teal, size: 13 } }] };
      });
      var se = SDX / Math.sqrt(N);
      Kit.$("ns-eqn").innerHTML = box("c-navy", Kit.fmt(SDX, 3), "&sigma;, the spread of one dive's count") + op("&divide;") + box("", Kit.fmt(Math.sqrt(N), 2), "&radic;n, with n = " + N) + op("=") + box("c-teal", Kit.fmt(se, 3), "standard error: typical wobble of the average") + op("so about 95% within") + box("c-coral", "±" + Kit.fmt(1.96 * se, 3), "of 0.75");
    }
    gen(); draw();
  }

  // ---------- 8. bootstrap and confidence interval ----------
  function ciFig() {
    var OBS = 121 / 154, mode = 0, boots = [], last = null, seed = 100, studies = null;
    var sn = Kit.slider(Kit.$("ci-controls"), { label: "Number of test women n (same 78.6% right)", min: 30, max: 1500, step: 1, value: 154, digits: 0 }, function () { boots = []; last = null; studies = null; draw(); });
    Kit.chips(Kit.$("ci-mode"), ["One test set, resampled", "100 repeated studies"], 0, function (i) { mode = i; draw(); });
    Kit.button(Kit.$("ci-bar"), "Resample once", function () { resample(1); });
    Kit.button(Kit.$("ci-bar"), "Resample 200", function () { resample(200); });
    Kit.button(Kit.$("ci-bar"), "Run 100 studies", function () { runStudies(); });
    Kit.button(Kit.$("ci-bar"), "Clear", function () { boots = []; last = null; studies = null; draw(); });
    function K() { return Math.round(OBS * sn.get()); }
    function binom(r, n, p) { var c = 0, i; for (i = 0; i < n; i++) if (r() < p) c++; return c; }
    function resample(m) { var n = sn.get(), p = K() / n, r = Kit.rng(seed++ * 977), i; mode = 0; Kit.$("ci-mode").querySelectorAll("button").forEach(function (b, j) { b.setAttribute("aria-pressed", String(j === 0)); }); for (i = 0; i < m; i++) { last = binom(r, n, p); boots.push(last / n * 100); } draw(); }
    function runStudies() {
      var n = sn.get(), r = Kit.rng(seed++ * 641), i; studies = [];
      for (i = 0; i < 100; i++) { var k = binom(r, n, OBS), ph = k / n, h = 1.96 * Math.sqrt(ph * (1 - ph) / n); studies.push([ (ph - h) * 100, (ph + h) * 100, ph * 100 ]); }
      mode = 1; Kit.$("ci-mode").querySelectorAll("button").forEach(function (b, j) { b.setAttribute("aria-pressed", String(j === 1)); }); draw();
    }
    function draw() {
      var n = sn.get(), k = K(), obs = k / n * 100, se = Math.sqrt(OBS * (1 - OBS) / n) * 100;
      if (mode === 0) {
        var sorted = boots.slice().sort(function (a, b) { return a - b; }), lo = sorted.length >= 100 ? pctile(sorted, 0.025) : null, hi = sorted.length >= 100 ? pctile(sorted, 0.975) : null;
        var step = 100 / n, bw = Math.max(1, Math.round(se / 3 / step)) * step, cnt = {}, xs = [], ys = [];
        boots.forEach(function (v) { var b = Math.round(v / bw) * bw; cnt[b.toFixed(3)] = (cnt[b.toFixed(3)] || 0) + 1; });
        Object.keys(cnt).forEach(function (b) { xs.push(+b); ys.push(cnt[b]); });
        Kit.plot("ci-plot", function (P) {
          var tr = [{ type: "bar", x: xs, y: ys, width: bw * 0.92, marker: { color: xs.map(function (x) { return lo !== null && x >= lo - bw / 2 && x <= hi + bw / 2 ? P.teal : rgba(P.navy, 0.55); }) }, hovertemplate: "%{x:.1f}% right: %{y} resamples<extra></extra>" }];
          return tr;
        }, function (P) {
          var sh = [{ type: "line", x0: obs, x1: obs, y0: 0, y1: 1, yref: "paper", line: { color: P.coral, width: 2.4 } }];
          return { dragmode: false, margin: { l: 52, r: 14, t: 28, b: 52 }, bargap: 0, xaxis: { range: [Math.max(0, obs - 5 * se - 2), Math.min(100, obs + 5 * se + 2)], title: { text: "share right in a resample (%)", standoff: 6 } },
            yaxis: { title: { text: "number of resamples" }, rangemode: "tozero" }, shapes: sh,
            annotations: [{ x: obs, y: 1, yref: "paper", text: "observed " + Kit.fmt(obs, 1) + "%", showarrow: false, yanchor: "bottom", xanchor: "left", xshift: 6, font: { color: P.coral, size: 14 } }].concat(!boots.length ? [{ xref: "paper", yref: "paper", x: 0.5, y: 0.5, text: "Press “Resample 200”", showarrow: false, font: { color: P.muted, size: 15 } }] : []) };
        });
        Kit.$("ci-eqn").innerHTML = box("c-coral", k + " of " + n, "right on the test set") + op("=") + box("c-coral", Kit.fmt(obs, 1) + "%", "observed score") + op("&rarr;") +
          box("c-teal", lo === null ? "&ndash;" : Kit.fmt(lo, 1) + "% to " + Kit.fmt(hi, 1) + "%", boots.length >= 100 ? "middle 95% of " + boots.length + " resamples" : "needs 100 or more resamples") + op("formula:") + box("", Kit.fmt(obs - 1.96 * se, 1) + "% to " + Kit.fmt(obs + 1.96 * se, 1) + "%", "score &plusmn; 1.96 &times; &radic;(p(1&minus;p)/n)");
        Kit.setText("ci-status", last === null ? "A resample draws " + n + " cases with replacement from your test set, then scores them." : "Last resample: " + Math.round(last) + " of " + n + " right (" + Kit.fmt(last / n * 100, 1) + "%). Resamples so far: " + boots.length + ".");
      } else {
        if (!studies) { runStudies(); return; }
        var T = OBS * 100, cover = studies.filter(function (s) { return s[0] <= T && s[1] >= T; }).length, ix = [], iy = [], mx = [], my = [];
        studies.forEach(function (s, i) { var tgt = (s[0] <= T && s[1] >= T) ? [ix, iy] : [mx, my]; tgt[0].push(s[0], s[1], null); tgt[1].push(i + 1, i + 1, null); });
        Kit.plot("ci-plot", function (P) {
          return [{ x: ix, y: iy, mode: "lines", line: { color: P.navy, width: 2 }, hoverinfo: "skip", connectgaps: false }, { x: mx, y: my, mode: "lines", line: { color: P.coral, width: 3 }, hoverinfo: "skip", connectgaps: false }];
        }, function (P) {
          return { dragmode: false, margin: { l: 52, r: 14, t: 16, b: 52 }, xaxis: { range: [T - 5 * se - 2, T + 5 * se + 2], title: { text: "95% interval for the score (%), one line per study", standoff: 6 } }, yaxis: { range: [0, 101], title: { text: "study number" } },
            shapes: [{ type: "line", x0: T, x1: T, y0: 0, y1: 101, line: { color: P.teal, width: 2.4 } }],
            annotations: [{ x: T, y: 101, text: "true score " + Kit.fmt(T, 1) + "% (known only in this simulation)", showarrow: false, yanchor: "bottom", font: { color: P.teal, size: 13 } }] };
        });
        Kit.$("ci-eqn").innerHTML = box("c-teal", String(cover), "intervals that contain the true score") + op("of") + box("", "100", "studies") + op("&rarr;") + box("c-coral", String(100 - cover), "intervals that miss (coral)");
        Kit.setText("ci-status", "The lines are separate studies: a new test set of " + n + " women, scored, with its own 95% interval. About 95 of 100 lines should cross the teal line.");
      }
    }
    resample(300); mode = 0; Kit.$("ci-mode").querySelectorAll("button").forEach(function (b, j) { b.setAttribute("aria-pressed", String(j === 0)); }); last = null; draw();
  }

  // ---------- 9. is model B better than model A: the coin-flip test ----------
  function testFig() {
    var seed = 3, sim = null;
    var sd_ = Kit.slider(Kit.$("ts-controls"), { label: "Frames where only one model is right", min: 4, max: 100, step: 1, value: 15, digits: 0 }, function () { var d = sd_.get(); sb.input.max = d; if (sb.get() > d) sb.set(d); sim = null; draw(); });
    var sb = Kit.slider(Kit.$("ts-controls"), { label: "Of those, frames won by model B", min: 0, max: 100, step: 1, value: 11, digits: 0 }, function () { sim = null; draw(); });
    sb.input.max = 15;
    Kit.button(Kit.$("ts-bar"), "Flip coins in 2,000 imaginary worlds", function () {
      var d = sd_.get(), r = Kit.rng(seed++ * 313), c = new Array(d + 1).fill(0), i, j, w; for (i = 0; i < 2000; i++) { w = 0; for (j = 0; j < d; j++) if (r() < 0.5) w++; c[w]++; }
      sim = c.map(function (v) { return v / 2000; }); draw();
    });
    Kit.button(Kit.$("ts-bar"), "Clear", function () { sim = null; draw(); });
    function draw() {
      var d = sd_.get(), b = sb.get(), ks = lin(0, d, d + 1), pm = ks.map(function (k) { return Math.exp(logChoose(d, k) - d * Math.LN2); }), dev = Math.abs(b - d / 2);
      var tail = ks.map(function (k) { return Math.abs(k - d / 2) >= dev - 1e-9; }), p = Math.min(1, sum(pm.filter(function (_, i) { return tail[i]; })));
      var both = 175 - d, a = both + d - b, bb = both + b;
      Kit.plot("ts-plot", function (P) {
        var tr = [{ type: "bar", x: ks, y: pm, marker: { color: tail.map(function (t) { return t ? P.coral : rgba(P.navy, 0.6); }) }, hovertemplate: "B wins %{x}: probability %{y:.3f}<extra></extra>" }];
        if (sim) tr.push({ x: ks, y: sim, mode: "markers", marker: { color: P.ink, size: 7, symbol: "diamond" }, hovertemplate: "simulated: %{y:.3f}<extra></extra>" });
        return tr;
      }, function (P) {
        return { dragmode: false, margin: { l: 52, r: 14, t: 28, b: 52 }, bargap: 0.08, xaxis: { range: [-0.7, d + 0.7], title: { text: "frames B wins if the models were equal", standoff: 6 } }, yaxis: { title: { text: "probability" }, rangemode: "tozero" },
          shapes: [{ type: "line", x0: b, x1: b, y0: 0, y1: 1, yref: "paper", line: { color: P.teal, width: 2.6 } }],
          annotations: [{ x: b, y: 1, yref: "paper", text: "observed: B won " + b, showarrow: false, yanchor: "bottom", xanchor: b > d / 2 ? "right" : "left", font: { color: P.teal, size: 14 } }] };
      });
      Kit.$("ts-eqn").innerHTML = box("c-navy", Kit.fmt(a / 2, 1) + "%", "model A, right on " + Math.round(a) + " of 200") + op("vs") + box("c-teal", Kit.fmt(bb / 2, 1) + "%", "model B, right on " + Math.round(bb) + " of 200") + op("&rarr; gap") + box("", (bb >= a ? "+" : "−") + Kit.fmt(Math.abs(bb - a) / 2, 1), "points, B minus A") + op("&rarr;") + box("c-coral", Kit.fmt(p, 3), "p-value: chance of a split this lopsided, or more, by luck");
      Kit.setText("ts-status", "Out of " + d + " frames where the models differ, B won " + b + " and A won " + (d - b) + ". If the models were equally good, a split at least this lopsided (either way) would happen about " + Kit.fmt(p * 100, 1) + "% of the time." + (sim ? " The diamonds are the coin flips you ran." : ""));
    }
    draw();
  }

  // ---------- 10. correlation and causation ----------
  function causeFig() {
    var r = Kit.rng(21), D = [], i, T, ice, sh;
    for (i = 0; i < 150; i++) {
      T = 27 + 2.4 * Math.sin(2 * Math.PI * i / 150 - 1) + 0.5 * Kit.gauss(r);
      ice = Math.max(0, 55 + 20 * (T - 27) + 9 * Kit.gauss(r));
      sh = Math.max(0, Math.round(2 + 1.1 * (T - 27) + 0.9 * Kit.gauss(r)));
      D.push({ T: T, ice: ice, sh: sh, jit: (r() - 0.5) * 0.28 });
    }
    var intervene = false;
    var sc = Kit.slider(Kit.$("ca-controls"), { label: "Window centre, water temperature", min: 24, max: 30.4, step: 0.1, value: 27, show: function (v) { return Kit.fmt(v, 1) + " °C"; } }, draw);
    var sw = Kit.slider(Kit.$("ca-controls"), { label: "Window width", min: 0.4, max: 8, step: 0.1, value: 8, show: function (v) { return v >= 7.95 ? "all days" : "±" + Kit.fmt(v / 2, 1) + " °C"; } }, draw);
    Kit.chips(Kit.$("ca-mode"), ["Watch what happens", "Ban ice cream"], 0, function (k) { intervene = k === 1; draw(); });
    Kit.button(Kit.$("ca-mode"), "Look at one temperature only", function () { sc.set(27); sw.set(0.8); draw(); });
    function corr(a, b) { if (a.length < 3) return NaN; var ma = mean(a), mb = mean(b), sa = sd(a, 0), sb = sd(b, 0); if (sa === 0 || sb === 0) return NaN; return mean(a.map(function (v, k) { return ((v - ma) / sa) * ((b[k] - mb) / sb); })); }
    function draw() {
      var c = sc.get(), w = sw.get(), all = w >= 7.95, inW = D.filter(function (d) { return all || Math.abs(d.T - c) <= w / 2; });
      var rAll = corr(D.map(function (d) { return d.ice; }), D.map(function (d) { return d.sh; })), rW = corr(inW.map(function (d) { return d.ice; }), inW.map(function (d) { return d.sh; }));
      var X = function (d) { return intervene ? 0 : d.ice; };
      Kit.plot("ca-plot", function (P) {
        var out = D.filter(function (d) { return !(all || Math.abs(d.T - c) <= w / 2); }), mk = function (arr, op_) {
          return { x: arr.map(function (d) { return X(d) + (intervene ? (d.jit * 18) : 0); }), y: arr.map(function (d) { return d.sh + d.jit; }), mode: "markers", marker: { size: 9, opacity: op_, color: arr.map(function (d) { return d.T; }), cmin: 24, cmax: 30.4, colorscale: [[0, P.navy], [1, P.coral]], line: { color: P.bg, width: 0.6 } }, text: arr.map(function (d) { return Kit.fmt(d.T, 1) + " °C"; }), hovertemplate: "ice cream %{x:.0f}, sightings %{y:.0f}, %{text}<extra></extra>" };
        };
        return [mk(out, 0.14), mk(inW, 0.95)];
      }, function (P) {
        return { dragmode: false, margin: { l: 52, r: 14, t: 14, b: 52 }, xaxis: { range: [intervene ? -20 : -5, 125], title: { text: "cones sold that day (made-up)", standoff: 6 } }, yaxis: { range: [-0.7, 8], title: { text: "shark sightings that day" } },
          annotations: [{ xref: "paper", yref: "paper", x: 0.01, y: 0.99, xanchor: "left", yanchor: "top", showarrow: false, text: "colour: water temperature<br>(cool navy to warm coral)", font: { color: P.ink2, size: 13 } }] };
      });
      var mean0 = mean(D.map(function (d) { return d.sh; }));
      Kit.$("ca-eqn").innerHTML = intervene ?
        box("c-navy", Kit.fmt(mean0, 2), "average sightings before the ban") + op("&rarr;") + box("c-teal", Kit.fmt(mean0, 2), "average sightings after: no change") + op("because") + box("", "0", "cones sold, but the water is as warm as before") :
        box("c-coral", Kit.fmt(rAll, 2), "r, all 150 days") + op("vs") + box("c-teal", isNaN(rW) ? "&ndash;" : Kit.fmt(rW, 2), "r, days inside the window") + op("with") + box("", String(inW.length), "days in the window");
      Kit.setText("ca-status", intervene ? "Selling no ice cream changes the x values only. Sightings follow the water, so they stay as they were." : (all ? "All days together show a strong link. Narrow the window and the link fades." : "Within a narrow temperature band, ice-cream sales say little about sightings."));
    }
    draw();
  }
})();
