// A2 Linear regression: eight figures, all on the made-up spadenose records.
//   1 lineFig     drag the line, see the error squares and the MSE (D3)
//   2 handFig     the best line by hand: a table of six sharks you can switch off (HTML and D3)
//   3 descentFig  gradient descent on the loss bowl: 3D surface, data and loss curve (Plotly); derivation stepper
//   4 rateFig     learning-rate slider: slow, good, bouncing and diverging runs (Plotly)
//   5 scaleFig    raw, centred and standardised length: the same loss, three shapes of bowl (Plotly)
//   6 residFig    residual plots for three models (Plotly)
//   7 logFig      the length-weight curve becomes a straight line on log axes (Plotly)
//   8 planeFig    two inputs, length and girth: a plane in 3D (Plotly)
(function () {
  "use strict";
  var REC = null, SP = [], X = [], Y = [], G = [], N = 0, MU = 0, SD = 1, YBAR = 0;
  var SAMPLE_IDS = [2509, 8722, 5577, 7674, 5284, 3581, 2590, 3305, 5093, 3626, 2407, 9585];
  var HAND_IDS = [2509, 5284, 3305, 5093, 2407, 9585];

  document.addEventListener("DOMContentLoaded", function () {
    Kit.data("../data/sharks/catch.json").then(function (J) {
      REC = J.records;
      SP = REC.filter(function (r) { return r.species === "spadenose"; }).sort(function (a, b) { return a.length_cm - b.length_cm; });
      X = SP.map(function (r) { return r.length_cm; }); Y = SP.map(function (r) { return r.weight_kg; }); G = SP.map(function (r) { return r.girth_cm; });
      N = X.length; MU = mean(X); SD = Math.sqrt(mean(X.map(function (v) { return (v - MU) * (v - MU); }))); YBAR = mean(Y);
      lineFig(); handFig(); descentFig(); derivation(); rateFig(); scaleFig(); residFig(); logFig(); planeFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- helpers ----------
  function mean(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function lin(a, b, n) { var o = [], i; for (i = 0; i < n; i++) o.push(a + (b - a) * i / (n - 1)); return o; }
  function box(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function op(s) { return '<span class="op">' + s + '</span>'; }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { var C = hexRGB(c); return "rgba(" + C[0] + "," + C[1] + "," + C[2] + "," + a + ")"; }
  function mix(a, b, t) { var A = hexRGB(a), B = hexRGB(b); return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }).join(",") + ")"; }
  function styleAxis(g, P) { g.selectAll("path,line").attr("stroke", P.muted); g.selectAll("text").attr("fill", P.muted).attr("font-size", 12.5); }
  function hasWebGL() { try { var c = document.createElement("canvas"); return !!(c.getContext("webgl") || c.getContext("experimental-webgl")); } catch (e) { return false; } }
  function sci(v, d) { return Kit.fmt(v, d == null ? 4 : d); }
  // least squares line through (x, y): returns {w, b, mse}
  function lsq(x, y) {
    var xb = mean(x), yb = mean(y), sxy = 0, sxx = 0, i;
    for (i = 0; i < x.length; i++) { sxy += (x[i] - xb) * (y[i] - yb); sxx += (x[i] - xb) * (x[i] - xb); }
    var w = sxx > 0 ? sxy / sxx : 0, b = yb - w * xb, e = 0;
    for (i = 0; i < x.length; i++) e += Math.pow(w * x[i] + b - y[i], 2);
    return { w: w, b: b, mse: e / x.length, xb: xb, yb: yb, sxy: sxy, sxx: sxx };
  }
  function sampleOf(ids) { return ids.map(function (id) { return SP.filter(function (r) { return r.record_id === id; })[0]; }).sort(function (a, b) { return a.length_cm - b.length_cm; }); }
  // second moments of (x, y): the loss of a line is a quadratic in (w, b) built from these
  function moments(x, y) {
    var n = x.length, M = { Ex: 0, Ex2: 0, Ey: 0, Exy: 0, Ey2: 0 }, i;
    for (i = 0; i < n; i++) { M.Ex += x[i]; M.Ex2 += x[i] * x[i]; M.Ey += y[i]; M.Exy += x[i] * y[i]; M.Ey2 += y[i] * y[i]; }
    for (var k in M) M[k] /= n; return M;
  }
  function lossAt(M, w, b) { return M.Ex2 * w * w + 2 * M.Ex * w * b + b * b - 2 * M.Exy * w - 2 * M.Ey * b + M.Ey2; }
  function gradAt(M, w, b) { return [2 * (M.Ex2 * w + M.Ex * b - M.Exy), 2 * (M.Ex * w + b - M.Ey)]; }
  function eigs(M) { var a = 2 * M.Ex2, c = 2 * M.Ex, d = 2, tr = a + d, det = a * d - c * c, s = Math.sqrt(Math.max(0, tr * tr / 4 - det)); return [tr / 2 + s, tr / 2 - s]; }
  function runGD(M, eta, w0, b0, n) {
    var p = [{ w: w0, b: b0, L: lossAt(M, w0, b0) }], w = w0, b = b0, i, g;
    for (i = 0; i < n; i++) {
      g = gradAt(M, w, b); w -= eta * g[0]; b -= eta * g[1];
      var L = lossAt(M, w, b);
      if (!isFinite(L) || L > 1e8) { p.push({ w: w, b: b, L: Infinity }); break; }
      p.push({ w: w, b: b, L: L });
    }
    return p;
  }
  function typeset(node) { if (typeof renderMathInElement !== "undefined") renderMathInElement(node, { delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false }); }
  var STD = null;
  function lossTicks(hi) { var v = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 10, 100, 1000].filter(function (t) { return t <= hi; }); return { tickvals: v, ticktext: v.map(String) }; }
  function stdMoments() { if (!STD) STD = moments(X.map(function (v) { return (v - MU) / SD; }), Y); return STD; }

  // ---------- 1. drag the line ----------
  function lineFig() {
    var S = sampleOf(SAMPLE_IDS), x = S.map(function (r) { return r.length_cm; }), y = S.map(function (r) { return r.weight_kg; }), n = x.length;
    var best = lsq(x, y), w = 0.006, b = 0.05, squares = true;
    var node = Kit.$("lf-svg"), svg = d3.select(node), W = 640, H = 360;
    var sw = Kit.slider(Kit.$("lf-controls"), { label: "Slope w, kg gained per cm", min: -0.005, max: 0.035, step: 0.0001, value: w, digits: 4 }, function (v) { w = v; draw(); });
    var sb = Kit.slider(Kit.$("lf-controls"), { label: "Bias b, kg (the line's height at length 0)", min: -1, max: 0.5, step: 0.005, value: b, digits: 3 }, function (v) { b = v; draw(); });
    var bar = Kit.$("lf-bar"), busy = null;
    Kit.button(bar, "Show the best line", function () {
      var w0 = w, b0 = b; if (busy) busy();
      busy = Kit.animate(900, function (p) { var e = Kit.ease(p); w = w0 + (best.w - w0) * e; b = b0 + (best.b - b0) * e; sw.set(w); sb.set(b); draw(); });
    });
    Kit.button(bar, "Reset", function () { if (busy) busy(); w = 0.006; b = 0.05; sw.set(w); sb.set(b); draw(); });
    Kit.chips(bar, ["Squares", "Misses only"], 0, function (i) { squares = i === 0; draw(); });
    var tb = Kit.$("lf-tbody");
    var X0 = 24, X1 = 54, XA = 18, XB = 60, YMAX = 0.7;
    function draw() {
      var P = Kit.palette(); W = Kit.width(node); H = W < 520 ? 320 : 380;
      var ml = 54, mr = 16, mt = 14, mb = 46, iw = W - ml - mr, ih = H - mt - mb;
      var sx = d3.scaleLinear().domain([XA, XB]).range([ml, ml + iw]), sy = d3.scaleLinear().domain([0, YMAX]).range([mt + ih, mt]);
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      svg.append("defs").append("clipPath").attr("id", "lf-clip").append("rect").attr("x", ml).attr("y", mt).attr("width", iw).attr("height", ih);
      var ax = svg.append("g").attr("transform", "translate(0," + (mt + ih) + ")").call(d3.axisBottom(sx).ticks(W < 480 ? 5 : 8)); styleAxis(ax, P);
      var ay = svg.append("g").attr("transform", "translate(" + ml + ",0)").call(d3.axisLeft(sy).ticks(7).tickSize(-iw)); styleAxis(ay, P);
      ay.selectAll("line").attr("stroke", P.rule); ay.select(".domain").attr("stroke", P.muted);
      svg.append("text").attr("x", ml + iw / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.ink2).text("length, cm");
      svg.append("text").attr("transform", "rotate(-90)").attr("x", -(mt + ih / 2)).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.ink2).text("weight, kg");
      var pxKg = ih / YMAX, sumSq = 0, big = { s: 0 }, i;
      var g = svg.append("g").attr("clip-path", "url(#lf-clip)");
      for (i = 0; i < n; i++) {
        var pred = w * x[i] + b, r = pred - y[i], s = Math.abs(r) * pxKg; sumSq += r * r;
        var px = sx(x[i]), top = sy(Math.max(pred, y[i]));
        if (squares) {
          var x0 = px + s > ml + iw ? px - s : px;
          g.append("rect").attr("x", x0).attr("y", top).attr("width", s).attr("height", s).attr("fill", rgba(P.coral, 0.2)).attr("stroke", P.coral).attr("stroke-width", 1.2);
          if (s > big.s) big = { s: s, x: x0 + s / 2, y: top + s / 2 };
        } else {
          g.append("line").attr("x1", px).attr("x2", px).attr("y1", sy(y[i])).attr("y2", sy(pred)).attr("stroke", P.coral).attr("stroke-width", 2.2);
        }
      }
      g.append("line").attr("x1", sx(XA)).attr("x2", sx(XB)).attr("y1", sy(w * XA + b)).attr("y2", sy(w * XB + b)).attr("stroke", P.teal).attr("stroke-width", 3.5);
      var ptsG = svg.append("g");
      for (i = 0; i < n; i++) {
        var tag = S[i].record_id === 3305;
        ptsG.append("circle").attr("cx", sx(x[i])).attr("cy", sy(y[i])).attr("r", tag ? 6.5 : 5).attr("fill", P.navy).attr("stroke", tag ? P.coral : P.bg).attr("stroke-width", tag ? 3 : 1.5);
        if (tag) ptsG.append("text").attr("x", sx(x[i]) - 10).attr("y", sy(y[i]) - 10).attr("text-anchor", "end").attr("font-size", 13).attr("font-weight", 600).attr("fill", P.coral).attr("paint-order", "stroke").attr("stroke", P.bg).attr("stroke-width", 4).text("Tag 3305");
      }
      if (squares && big.s > 22) svg.append("text").attr("x", big.x).attr("y", big.y + 4).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.coral).attr("font-weight", 600).text("miss²");
      svg.append("text").attr("x", ml + 8).attr("y", mt + 16).attr("font-size", 13.5).attr("fill", P.ink2).text("A coral square has a side equal to one residual.");
      // handles
      [X0, X1].forEach(function (xh, hi) {
        var yh = clamp(w * xh + b, -0.03, YMAX + 0.03);
        var hg = svg.append("g").attr("tabindex", 0).attr("role", "slider").attr("aria-label", (hi ? "Right" : "Left") + " end of the line").attr("aria-valuenow", (w * xh + b).toFixed(3)).style("cursor", "ns-resize").style("outline", "none");
        hg.append("circle").attr("cx", sx(xh)).attr("cy", sy(yh)).attr("r", 18).attr("fill", "transparent");
        hg.append("circle").attr("cx", sx(xh)).attr("cy", sy(yh)).attr("r", 9).attr("fill", P.paper).attr("stroke", P.teal).attr("stroke-width", 3.5);
        hg.append("text").attr("x", sx(xh)).attr("y", sy(yh) + (hi ? 26 : -16)).attr("text-anchor", "middle").attr("font-size", 12.5).attr("fill", P.teal).attr("font-weight", 600).attr("paint-order", "stroke").attr("stroke", P.bg).attr("stroke-width", 4).text("drag");
        function setY(yv) {
          var y0 = w * X0 + b, y1 = w * X1 + b; if (hi) y1 = yv; else y0 = yv;
          w = (y1 - y0) / (X1 - X0); b = y0 - w * X0; w = clamp(w, -0.005, 0.035); b = clamp(b, -1, 0.5); sw.set(w); sb.set(b);
        }
        hg.call(d3.drag().on("drag", function (e) { setY(clamp(sy.invert(e.y), -0.25, 1.0)); draw(); }));
        hg.on("keydown", function (e) {
          var d = e.key === "ArrowUp" ? 0.01 : e.key === "ArrowDown" ? -0.01 : 0; if (!d) return;
          e.preventDefault(); setY(w * xh + b + d); draw(); var gs = svg.selectAll("g[role=slider]").nodes(); if (gs[hi]) gs[hi].focus();
        });
      });
      var mse = sumSq / n;
      Kit.setText("lf-w", sci(w, 4)); Kit.setText("lf-b", sci(b, 3)); Kit.setText("lf-mse", mse.toFixed(5)); Kit.setText("lf-rmse", Kit.fmt(Math.sqrt(mse), 3) + " kg"); Kit.setText("lf-best", best.mse.toFixed(5));
      var ratio = mse / best.mse;
      Kit.setText("lf-status", ratio < 1.02 ? "That is the best line, to within 2%." : (ratio < 1.5 ? "Close: your MSE is " + ratio.toFixed(2) + " times the best possible." : "Your MSE is " + ratio.toFixed(1) + " times the best possible. Move the line or press Show the best line."));
      var p3 = w * 41.0 + b;
      Kit.$("lf-eqn").innerHTML = box("c-teal", Kit.fmt(p3, 3), "prediction for Tag 3305, kg") + op("=") + box("c-sand", sci(w, 4), "slope w") + op("&times;") + box("c-navy", "41.0", "its length, cm") + op("+") + box("c-sand", Kit.fmt(b, 3), "bias b") +
        op("&rarr;") + box("c-coral", Kit.fmt(Math.pow(p3 - 0.228, 2), 4), "its square: (miss)<sup>2</sup>, true weight 0.228");
      Kit.$("lf-eqn2").innerHTML = box("c-coral", sumSq.toFixed(4), "sum of the " + n + " squares (kg<sup>2</sup>)") + op("&divide;") + box("", String(n), "sharks") + op("=") + box("c-teal", mse.toFixed(5), "mean squared error, MSE") +
        op("&rarr;") + box("", Kit.fmt(Math.sqrt(mse), 3), "&radic;MSE: a typical miss in kg");
      tb.innerHTML = "";
      for (i = 0; i < n; i++) {
        var pr = w * x[i] + b, tr = Kit.html("tr", {}, tb);
        [S[i].record_id, x[i].toFixed(1), y[i].toFixed(3), pr.toFixed(3), Kit.fmt(y[i] - pr, 3), Math.pow(pr - y[i], 2).toFixed(4)].forEach(function (c) { Kit.html("td", {}, tr, String(c)); });
      }
      node.setAttribute("aria-label", "Twelve made-up spadenose sharks as dots, a draggable teal line and coral error squares. The mean squared error is " + mse.toFixed(5) + "; the best line scores " + best.mse.toFixed(5) + ".");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- 2. the best line by hand ----------
  function handFig() {
    var S = sampleOf(HAND_IDS), on = S.map(function () { return true; });
    var tb = Kit.$("bh-tbody"), tf = Kit.$("bh-tfoot"), node = Kit.$("bh-svg"), svg = d3.select(node), W = 560, H = 300;
    var rows = S.map(function (r, i) {
      var tr = Kit.html("tr", {}, tb), c = Kit.html("td", {}, tr), lab = Kit.html("label", {}, c), cb = Kit.html("input", { type: "checkbox", "aria-label": "Use Tag " + r.record_id }, lab);
      cb.checked = true; cb.addEventListener("change", function () { on[i] = cb.checked; update(); });
      var cells = [r.record_id, r.length_cm.toFixed(1), r.weight_kg.toFixed(3)].map(function (t) { return Kit.html("td", {}, tr, String(t)); });
      var d = [1, 2, 3, 4].map(function () { return Kit.html("td", { class: "calc" }, tr); });
      return { tr: tr, d: d };
    });
    function update() {
      var xs = [], ys = [];
      S.forEach(function (r, i) { if (on[i]) { xs.push(r.length_cm); ys.push(r.weight_kg); } });
      var f = xs.length >= 2 ? lsq(xs, ys) : null, xb = xs.length ? mean(xs) : 0, yb = ys.length ? mean(ys) : 0, sxy = 0, sxx = 0;
      S.forEach(function (r, i) {
        var dx = r.length_cm - xb, dy = r.weight_kg - yb;
        rows[i].tr.classList.toggle("off", !on[i]);
        if (on[i]) { sxy += dx * dy; sxx += dx * dx; rows[i].d[0].textContent = Kit.fmt(dx, 2); rows[i].d[1].textContent = Kit.fmt(dy, 3); rows[i].d[2].textContent = Kit.fmt(dx * dy, 3); rows[i].d[3].textContent = Kit.fmt(dx * dx, 2); }
        else rows[i].d.forEach(function (d) { d.textContent = "–"; });
      });
      tf.innerHTML = '<td></td><td>mean</td><td>' + xb.toFixed(2) + '</td><td>' + yb.toFixed(4) + '</td><td colspan="2" class="sum">sum of products</td><td class="sum"><b>' + sxy.toFixed(3) + '</b></td><td class="sum"><b>' + sxx.toFixed(2) + '</b></td>';
      var w = f ? f.w : NaN, b = f ? f.b : NaN;
      if (f && sxx > 0) {
        Kit.$("bh-eqn").innerHTML = box("c-coral", sxy.toFixed(3), "sum of (x &minus; mean x)(y &minus; mean y)") + op("&divide;") + box("c-navy", sxx.toFixed(2), "sum of (x &minus; mean x)<sup>2</sup>") + op("=") + box("c-teal", sci(w, 5), "slope w, kg per cm");
        Kit.$("bh-eqn2").innerHTML = box("c-navy", yb.toFixed(4), "mean weight") + op("&minus;") + box("c-teal", sci(w, 5), "w") + op("&times;") + box("c-navy", xb.toFixed(2), "mean length") + op("=") + box("c-teal", Kit.fmt(b, 4), "bias b, kg");
        Kit.$("bh-note").innerHTML = "With " + xs.length + " sharks the best line is weight = " + sci(w, 5) + " &times; length " + (b < 0 ? "&minus; " : "+ ") + Math.abs(b).toFixed(4) + " kg. The best line for all 268 spadenose records has slope 0.01545 and bias &minus;0.3789.";
      } else {
        Kit.$("bh-eqn").innerHTML = ""; Kit.$("bh-eqn2").innerHTML = "";
        Kit.$("bh-note").textContent = xs.length < 2 ? "A line needs at least two sharks." : "All the lengths are equal, so the slope is undefined.";
      }
      draw(f, xb, yb);
    }
    function draw(f, xb, yb) {
      var P = Kit.palette(); W = Kit.width(node); H = W < 520 ? 270 : 300;
      var ml = 52, mr = 14, mt = 12, mb = 44, iw = W - ml - mr, ih = H - mt - mb;
      var sx = d3.scaleLinear().domain([18, 60]).range([ml, ml + iw]), sy = d3.scaleLinear().domain([0, 0.7]).range([mt + ih, mt]);
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      svg.append("defs").append("clipPath").attr("id", "bh-clip").append("rect").attr("x", ml).attr("y", mt).attr("width", iw).attr("height", ih);
      var ax = svg.append("g").attr("transform", "translate(0," + (mt + ih) + ")").call(d3.axisBottom(sx).ticks(W < 480 ? 5 : 8)); styleAxis(ax, P);
      var ay = svg.append("g").attr("transform", "translate(" + ml + ",0)").call(d3.axisLeft(sy).ticks(7).tickSize(-iw)); styleAxis(ay, P); ay.selectAll("line").attr("stroke", P.rule);
      svg.append("text").attr("x", ml + iw / 2).attr("y", H - 8).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.ink2).text("length, cm");
      svg.append("text").attr("transform", "rotate(-90)").attr("x", -(mt + ih / 2)).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.ink2).text("weight, kg");
      var g = svg.append("g").attr("clip-path", "url(#bh-clip)");
      if (f && f.sxx > 0) {
        g.append("line").attr("x1", sx(18)).attr("x2", sx(60)).attr("y1", sy(f.w * 18 + f.b)).attr("y2", sy(f.w * 60 + f.b)).attr("stroke", P.teal).attr("stroke-width", 3.5);
        g.append("line").attr("x1", sx(xb)).attr("x2", sx(xb)).attr("y1", mt).attr("y2", mt + ih).attr("stroke", P.coral).attr("stroke-dasharray", "4 4").attr("opacity", 0.8);
        g.append("line").attr("x1", ml).attr("x2", ml + iw).attr("y1", sy(yb)).attr("y2", sy(yb)).attr("stroke", P.coral).attr("stroke-dasharray", "4 4").attr("opacity", 0.8);
        svg.append("circle").attr("cx", sx(xb)).attr("cy", sy(yb)).attr("r", 8).attr("fill", P.paper).attr("stroke", P.coral).attr("stroke-width", 3);
        svg.append("text").attr("x", sx(xb) + 12).attr("y", sy(yb) + 22).attr("font-size", 13).attr("font-weight", 600).attr("fill", P.coral).attr("paint-order", "stroke").attr("stroke", P.bg).attr("stroke-width", 4).text("average shark");
      }
      S.forEach(function (r, i) { svg.append("circle").attr("cx", sx(r.length_cm)).attr("cy", sy(r.weight_kg)).attr("r", on[i] ? 6 : 5).attr("fill", on[i] ? P.navy : "none").attr("stroke", on[i] ? P.bg : P.muted).attr("stroke-width", on[i] ? 1.5 : 1.8).attr("stroke-dasharray", on[i] ? null : "3 2"); });
      node.setAttribute("aria-label", "Six made-up sharks with their best-fit line and the point of average length and weight, through which the line passes.");
    }
    update(); Kit.onTheme(update); window.addEventListener("resize", Kit.debounce(update));
  }

  // ---------- 3. gradient descent on the loss bowl ----------
  function descentFig() {
    var M = stdMoments(), eta = 0.1, STARTS = [[0, 0.55], [0.4, 0.05], [-0.1, 0.1], [0.35, 0.6]], si = 0, timer = 0;
    var path = [], WG = lin(-0.15, 0.42, 38), BG = lin(0, 0.62, 38), Z = BG.map(function (b) { return WG.map(function (w) { return lossAt(M, w, b); }); });
    var wopt = (M.Exy - M.Ex * M.Ey) / (M.Ex2 - M.Ex * M.Ex), bopt = M.Ey - wopt * M.Ex, Lmin = lossAt(M, wopt, bopt), gl = hasWebGL();
    var bar = Kit.$("gd-bar"), runBtn;
    Kit.button(bar, "Step", function () { stopRun(); step(); });
    runBtn = Kit.button(bar, "Run", function () { if (timer) stopRun(); else startRun(); });
    Kit.button(bar, "Reset", function () { stopRun(); reset(); });
    Kit.button(bar, "New start", function () { stopRun(); si = (si + 1) % STARTS.length; reset(); });
    function stopRun() { if (timer) { clearInterval(timer); timer = 0; } runBtn.textContent = "Run"; }
    function startRun() { runBtn.textContent = "Pause"; timer = setInterval(function () { var last = path[path.length - 1], g = gradAt(M, last.w, last.b); if (path.length > 80 || Math.hypot(g[0], g[1]) < 4e-4) { stopRun(); return; } step(); }, 170); }
    function reset() { var s = STARTS[si]; path = [{ w: s[0], b: s[1], L: lossAt(M, s[0], s[1]) }]; draw(); }
    function step() { var l = path[path.length - 1], g = gradAt(M, l.w, l.b), w = l.w - eta * g[0], b = l.b - eta * g[1]; path.push({ w: w, b: b, L: lossAt(M, w, b) }); draw(); }
    function lineY(w, b, L) { return w * (L - MU) / SD + b; }
    function draw() {
      var cur = path[path.length - 1], st = path[0], g = gradAt(M, cur.w, cur.b), gn = Math.hypot(g[0], g[1]), t = path.length - 1, lx = [15, 65];
      if (!gl) Kit.$("gd-3d").textContent = "This browser cannot draw 3D graphics here. The data panel and the loss curve show the same run.";
      else Kit.plot("gd-3d", function (P) {
        return [
          { type: "surface", x: WG, y: BG, z: Z, showscale: false, hoverinfo: "skip", opacity: 0.92, colorscale: [[0, mix(P.paper, P.teal, 0.55)], [0.5, mix(P.paper, P.navy, 0.35)], [1, mix(P.paper, P.navy, 0.95)]],
            contours: { z: { show: true, usecolormap: false, color: P.paper, width: 1, project: { z: true } } }, lighting: { ambient: 0.8, diffuse: 0.6, specular: 0.05, roughness: 0.9 } },
          { type: "scatter3d", mode: "lines+markers", x: path.map(function (p) { return p.w; }), y: path.map(function (p) { return p.b; }), z: path.map(function (p) { return p.L + 0.004; }), line: { color: P.coral, width: 6 }, marker: { color: P.coral, size: 3.5 }, hoverinfo: "skip" },
          { type: "scatter3d", mode: "markers", x: [cur.w], y: [cur.b], z: [cur.L + 0.008], marker: { color: P.paper, size: 7, line: { color: P.coral, width: 3 } }, hoverinfo: "skip" },
          { type: "scatter3d", mode: "markers", x: [wopt], y: [bopt], z: [Lmin + 0.004], marker: { color: P.teal, size: 6, symbol: "diamond" }, hoverinfo: "skip" }
        ];
      }, function () {
        return { margin: { l: 14, r: 0, t: 0, b: 0 }, uirevision: "a2-gd-3d", scene: { xaxis: { title: { text: "slope w", font: { size: 12 } }, nticks: 4 }, yaxis: { title: { text: "bias b", font: { size: 12 } }, nticks: 4 }, zaxis: { title: { text: "L", font: { size: 12 } }, range: [0, 0.26], nticks: 4 },
          aspectmode: "manual", aspectratio: { x: 1, y: 1, z: 0.7 }, camera: { eye: Kit.width(Kit.$("gd-3d")) < 420 ? { x: 1.55, y: -1.65, z: 0.9 } : { x: 1.35, y: -1.5, z: 0.85 } } } };
      });
      Kit.plot("gd-data", function (P) {
        return [
          { x: X, y: Y, mode: "markers", marker: { color: P.navy, size: 5, opacity: 0.5 }, hoverinfo: "skip" },
          { x: lx, y: lx.map(function (L) { return lineY(st.w, st.b, L); }), mode: "lines", line: { color: P.muted, width: 2, dash: "dash" }, hoverinfo: "skip" },
          { x: lx, y: lx.map(function (L) { return lineY(wopt, bopt, L); }), mode: "lines", line: { color: P.sand, width: 2, dash: "dot" }, hoverinfo: "skip" },
          { x: lx, y: lx.map(function (L) { return lineY(cur.w, cur.b, L); }), mode: "lines", line: { color: P.teal, width: 3.5 }, hoverinfo: "skip" }
        ];
      }, function (P) {
        return { margin: { l: 52, r: 10, t: 10, b: 44 }, xaxis: { title: { text: "length, cm" }, range: [15, 65] }, yaxis: { title: { text: "weight, kg" }, range: [-0.1, 0.8] }, dragmode: false,
          annotations: [{ x: 16, y: lineY(st.w, st.b, 16), xanchor: "left", yanchor: "bottom", showarrow: false, text: "start", font: { size: 12.5, color: P.muted } }, { x: 64, y: lineY(cur.w, cur.b, 64), xanchor: "right", yanchor: "bottom", showarrow: false, text: "now", font: { size: 12.5, color: P.teal } }] };
      });
      Kit.plot("gd-loss", function (P) {
        return [{ x: path.map(function (p, i) { return i; }), y: path.map(function (p) { return p.L; }), mode: "lines+markers", line: { color: P.coral, width: 2.5 }, marker: { color: P.coral, size: 6 }, hovertemplate: "step %{x}: loss %{y:.4f}<extra></extra>" },
          { x: [0, Math.max(10, t + 2)], y: [Lmin, Lmin], mode: "lines", line: { color: P.teal, width: 1.5, dash: "dot" }, hoverinfo: "skip" }];
      }, { margin: { l: 56, r: 10, t: 10, b: 44 }, dragmode: false, xaxis: { title: { text: "step" }, range: [0, Math.max(10, t + 2)] }, yaxis: Object.assign({ title: { text: "loss L (log scale)" }, type: "log", range: [Math.log10(0.0015), Math.log10(0.3)] }, lossTicks(0.2)) });
      var wn = cur.w - eta * g[0], bn = cur.b - eta * g[1];
      Kit.$("gd-eqn").innerHTML = box("c-teal", Kit.fmt(wn, 4), "next w") + op("=") + box("c-sand", Kit.fmt(cur.w, 4), "w now") + op("&minus;") + box("c-navy", Kit.fmt(eta, 1), "step size &eta;") + op("&times;") + box("c-coral", Kit.fmt(g[0], 4), "slope of L along w");
      Kit.$("gd-eqn2").innerHTML = box("c-teal", Kit.fmt(bn, 4), "next b") + op("=") + box("c-sand", Kit.fmt(cur.b, 4), "b now") + op("&minus;") + box("c-navy", Kit.fmt(eta, 1), "step size &eta;") + op("&times;") + box("c-coral", Kit.fmt(g[1], 4), "slope of L along b");
      Kit.setText("gd-status", gn < 4e-3 ? "Step " + t + ": the slope is close to zero in both directions. The line is the best line (the dotted sand one)." : "Step " + t + ": loss " + Kit.fmt(cur.L, 4) + ". The best possible loss is " + Kit.fmt(Lmin, 4) + ".");
    }
    reset(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- the two partial derivatives, step by step ----------
  function derivation() {
    var M = stdMoments(), w = 0, b = 0.55, z = X.map(function (v) { return (v - MU) / SD; }), steps = [], shown = 1, i;
    for (i = 0; i < 6; i++) steps.push(Kit.$("dv-s" + i));
    var t = SP.filter(function (r) { return r.record_id === 3305; })[0], zt = (t.length_cm - MU) / SD, r = t.weight_kg - (w * zt + b);
    function Lfun(ww, bb) { var s = 0, k; for (k = 0; k < N; k++) s += Math.pow(ww * z[k] + bb - Y[k], 2); return s / N; }
    var h = 1e-5, nw = (Lfun(w + h, b) - Lfun(w - h, b)) / (2 * h), nb = (Lfun(w, b + h) - Lfun(w, b - h)) / (2 * h), g = gradAt(M, w, b);
    Kit.setText("dv-x", Kit.fmt(zt, 3)); Kit.setText("dv-r", Kit.fmt(r, 3));
    Kit.setText("dv-sq", Kit.fmt(2 * r, 3));
    Kit.setText("dv-c1", Kit.fmt(-2 * r * zt, 3)); Kit.setText("dv-c2", Kit.fmt(-2 * r, 3));
    Kit.setText("dv-a1", Kit.fmt(g[0], 4)); Kit.setText("dv-a2", Kit.fmt(g[1], 4));
    Kit.setText("dv-n1", Kit.fmt(nw, 4)); Kit.setText("dv-n2", Kit.fmt(nb, 4));
    function show() { steps.forEach(function (e, k) { e.style.display = k < shown ? "" : "none"; }); Kit.setText("dv-count", "Step " + shown + " of 6"); }
    var bar = Kit.$("dv-bar");
    Kit.button(bar, "Next step", function () { shown = Math.min(6, shown + 1); show(); });
    Kit.button(bar, "Back", function () { shown = Math.max(1, shown - 1); show(); });
    Kit.button(bar, "Show all", function () { shown = 6; show(); });
    show();
  }

  // ---------- 4. the learning rate ----------
  function rateFig() {
    var M = stdMoments(), w0 = 0, b0 = 0.55, NS = 40, eta = 0.1, shown = NS, stop = null;
    var wopt = (M.Exy - M.Ex * M.Ey) / (M.Ex2 - M.Ex * M.Ex), bopt = M.Ey - wopt * M.Ex, Lmin = lossAt(M, wopt, bopt);
    var WG = lin(-0.25, 0.5, 60), BG = lin(-0.1, 0.7, 60), Z = BG.map(function (b) { return WG.map(function (w) { return lossAt(M, w, b); }); });
    var sl = Kit.slider(Kit.$("lr-controls"), { label: "Learning rate η", min: -2.3, max: 0.12, step: 0.01, value: Math.log10(0.1), show: function (v) { return Kit.fmt(Math.pow(10, v), 3); } }, function (v) { eta = Math.pow(10, v); if (stop) stop(); shown = NS; draw(); });
    var PRE = [["Slow, 0.02", 0.02], ["Good, 0.3", 0.3], ["One jump, 0.5", 0.5], ["Bouncing, 0.9", 0.9], ["Diverging, 1.05", 1.05]];
    var bar = Kit.$("lr-bar");
    Kit.chips(bar, PRE.map(function (p) { return p[0]; }), 99, function (i) { eta = PRE[i][1]; sl.set(Math.log10(eta)); if (stop) stop(); shown = NS; draw(); });
    Kit.button(bar, "Run", function () { if (stop) stop(); shown = 0; stop = Kit.animate(NS * 90, function (p) { shown = Math.round(p * NS); draw(); if (p >= 1) stop = null; }); });
    function draw() {
      var path = runGD(M, eta, w0, b0, NS), k = Math.min(shown, path.length - 1), shownPath = path.slice(0, k + 1), last = shownPath[k];
      var hit = -1, i; for (i = 0; i < path.length; i++) if (path[i].L <= Lmin * 1.01) { hit = i; break; }
      var factor = 1 - 2 * eta, div = !isFinite(path[path.length - 1].L) || Math.abs(factor) > 1;
      Kit.plot("lr-map", function (P) {
        return [
          { type: "contour", x: WG, y: BG, z: Z, showscale: false, ncontours: 14, colorscale: [[0, rgba(P.navy, 0.03)], [1, rgba(P.navy, 0.5)]], contours: { coloring: "fill", showlabels: false }, line: { color: P.muted, width: 0.7 }, hoverinfo: "skip" },
          { x: shownPath.map(function (p) { return p.w; }), y: shownPath.map(function (p) { return p.b; }), mode: "lines+markers", line: { color: P.coral, width: 2 }, marker: { color: P.coral, size: 5 }, hoverinfo: "skip" },
          { x: [wopt], y: [bopt], mode: "markers", marker: { color: P.teal, size: 11, symbol: "diamond", line: { color: P.bg, width: 2 } }, hoverinfo: "skip" },
          { x: [w0], y: [b0], mode: "markers+text", text: ["start"], textposition: "top right", textfont: { size: 12.5, color: P.ink2 }, marker: { color: P.paper, size: 10, line: { color: P.coral, width: 3 } }, hoverinfo: "skip" }
        ];
      }, { margin: { l: 52, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { range: [-0.25, 0.5], title: { text: "slope w" }, constrain: "domain" }, yaxis: { range: [-0.1, 0.7], title: { text: "bias b" } } });
      Kit.plot("lr-loss", function (P) {
        return [{ x: shownPath.map(function (p, j) { return j; }), y: shownPath.map(function (p) { return Math.min(p.L, 1e3); }), mode: "lines+markers", line: { color: P.coral, width: 2.5 }, marker: { color: P.coral, size: 5 }, hovertemplate: "step %{x}: loss %{y:.4g}<extra></extra>" },
          { x: [0, NS], y: [Lmin, Lmin], mode: "lines", line: { color: P.teal, width: 1.5, dash: "dot" }, hoverinfo: "skip" }];
      }, { margin: { l: 56, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { range: [0, NS], title: { text: "step" } }, yaxis: Object.assign({ type: "log", range: [Math.log10(0.0015), 3], title: { text: "loss L (log scale)" } }, lossTicks(1000)) });
      var status;
      if (div) status = "Diverges: a step overshoots by more than it corrects, so the loss grows.";
      else if (Math.abs(factor) > 0.995 && hit < 0) status = "Too slow: after " + NS + " steps the loss is still far above the best.";
      else if (hit < 0) status = Math.abs(factor) >= 1 - 1e-9 ? "Bounces forever between two sides of the bowl." : "Converges, but slowly: not within 1% of the best loss in " + NS + " steps.";
      else if (factor < -0.05) status = "Bounces from side to side, and settles after " + hit + " steps.";
      else status = hit <= 1 ? "Reaches the bottom in " + hit + " step." : "Settles within 1% of the best loss in " + hit + " steps.";
      Kit.setText("lr-status", status);
      Kit.$("lr-eqn").innerHTML = box("c-navy", Kit.fmt(eta, 3), "learning rate &eta;") + op("&rarr;") + box("c-coral", Kit.fmt(factor, 3), "factor 1 &minus; 2&eta;") + op("&rarr;") + box("c-teal", Kit.fmt(Math.pow(Math.abs(factor), 10) * 100, Math.pow(Math.abs(factor), 10) > 1e4 ? 0 : 1) + "%", "distance left after 10 steps");
      Kit.setText("lr-loss-now", Kit.fmt(last.L, 4)); Kit.setText("lr-steps", hit < 0 ? "more than " + NS : String(hit));
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 5. feature scaling ----------
  function runThin(M, eta, w0, b0, n) {
    var out = [{ s: 0, w: w0, b: b0, L: lossAt(M, w0, b0) }], w = w0, b = b0, i, g, every = Math.max(1, Math.floor(n / 300));
    for (i = 1; i <= n; i++) {
      g = gradAt(M, w, b); w -= eta * g[0]; b -= eta * g[1];
      var L = lossAt(M, w, b);
      if (!isFinite(L) || L > 1e8) { out.push({ s: i, w: w, b: b, L: Infinity }); break; }
      if (i % every === 0 || i === n || i <= 40) out.push({ s: i, w: w, b: b, L: L });
    }
    return out;
  }
  function scaleFig() {
    var MODES = [
      { name: "Raw length, cm", xs: X, w: [-0.012, 0.042], b: [-1.0, 0.7], unit: "kg per cm" },
      { name: "Centred length", xs: X.map(function (v) { return v - MU; }), w: [-0.012, 0.042], b: [-0.1, 0.7], unit: "kg per cm" },
      { name: "Standardised length", xs: X.map(function (v) { return (v - MU) / SD; }), w: [-0.25, 0.5], b: [-0.1, 0.7], unit: "kg per spread" }
    ];
    MODES.forEach(function (m) {
      m.M = moments(m.xs, Y); m.eig = eigs(m.M); m.wopt = (m.M.Exy - m.M.Ex * m.M.Ey) / (m.M.Ex2 - m.M.Ex * m.M.Ex); m.bopt = m.M.Ey - m.wopt * m.M.Ex; m.Lmin = lossAt(m.M, m.wopt, m.bopt);
      var best = 1 / m.eig[0], w = 0, b = 0.55, k, found = -1;
      for (k = 1; k <= 1500000; k++) { var g = gradAt(m.M, w, b); w -= best * g[0]; b -= best * g[1]; if (lossAt(m.M, w, b) <= m.Lmin * 1.01) { found = k; break; } }
      m.stepsBest = found;
      m.grid = (function () { var WG = lin(m.w[0], m.w[1], 80), BG = lin(m.b[0], m.b[1], 80); return { WG: WG, BG: BG, Z: BG.map(function (b) { return WG.map(function (w) { return Math.log10(Math.max(1e-6, lossAt(m.M, w, b))); }); }) }; })();
    });
    var mode = 0, eta = 0.0005, nSteps = 40, NS = [40, 1000, 100000];
    var sl = Kit.slider(Kit.$("sc-controls"), { label: "Learning rate η (log scale)", min: -5.3, max: 0.3, step: 0.01, value: Math.log10(eta), show: function (v) { var e = Math.pow(10, v); return e < 0.01 ? e.toExponential(1) : Kit.fmt(e, 3); } }, function (v) { eta = Math.pow(10, v); draw(); });
    var bar = Kit.$("sc-bar");
    Kit.chips(bar, MODES.map(function (m) { return m.name; }), 0, function (i) { mode = i; draw(); });
    var bar2 = Kit.$("sc-bar2");
    Kit.button(bar2, "Best safe η for this scaling", function () { eta = 1 / MODES[mode].eig[0]; sl.set(Math.log10(eta)); draw(); });
    Kit.button(bar2, "Try η = 0.3", function () { eta = 0.3; sl.set(Math.log10(eta)); draw(); });
    Kit.chips(bar2, ["40 steps", "1,000 steps", "100,000 steps"], 0, function (i) { nSteps = NS[i]; draw(); });
    function draw() {
      var m = MODES[mode], etaMax = 2 / m.eig[0], pth = runThin(m.M, eta, 0, 0.55, nSteps), last = pth[pth.length - 1];
      Kit.plot("sc-map", function (P) {
        return [
          { type: "contour", x: m.grid.WG, y: m.grid.BG, z: m.grid.Z, showscale: false, contours: { coloring: "fill", start: -2.6, end: 0.2, size: 0.2 }, colorscale: [[0, rgba(P.navy, 0.03)], [1, rgba(P.navy, 0.5)]], line: { color: P.muted, width: 0.6 }, hoverinfo: "skip" },
          { x: pth.map(function (p) { return p.w; }), y: pth.map(function (p) { return p.b; }), mode: "lines+markers", line: { color: P.coral, width: 2 }, marker: { color: P.coral, size: pth.length > 60 ? 3 : 5 }, hoverinfo: "skip" },
          { x: [m.wopt], y: [m.bopt], mode: "markers", marker: { color: P.teal, size: 11, symbol: "diamond", line: { color: P.bg, width: 2 } }, hoverinfo: "skip" },
          { x: [0], y: [0.55], mode: "markers", marker: { color: P.paper, size: 10, line: { color: P.coral, width: 3 } }, hoverinfo: "skip" }
        ];
      }, { margin: { l: 56, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { range: m.w, title: { text: "slope w, " + m.unit }, tickformat: mode === 2 ? null : ".3f" }, yaxis: { range: m.b, title: { text: "bias b, kg" } } });
      var xmax = Math.max(10, last.s);
      Kit.plot("sc-loss", function (P) {
        return [{ x: pth.map(function (p) { return p.s; }), y: pth.map(function (p) { return Math.min(p.L, 1e3); }), mode: "lines", line: { color: P.coral, width: 2.5 }, hovertemplate: "step %{x}: loss %{y:.4g}<extra></extra>" },
          { x: [0, xmax], y: [m.Lmin, m.Lmin], mode: "lines", line: { color: P.teal, width: 1.5, dash: "dot" }, hoverinfo: "skip" }];
      }, { margin: { l: 56, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { range: [0, xmax], title: { text: "step" } }, yaxis: Object.assign({ type: "log", range: [Math.log10(0.0015), 3], title: { text: "loss L (log scale)" } }, lossTicks(1000)) });
      var div = !isFinite(last.L) || eta > etaMax, msg, eS = function (v) { return v < 0.01 ? v.toExponential(1) : Kit.fmt(v, 3); };
      if (div) msg = "η = " + eS(eta) + " is above the limit " + (etaMax < 0.01 ? etaMax.toExponential(2) : Kit.fmt(etaMax, 3)) + " for this scaling, so the loss grows.";
      else msg = "After " + last.s.toLocaleString("en-US") + " steps the loss is " + Kit.fmt(last.L, 4) + ". The best possible is " + Kit.fmt(m.Lmin, 4) + (last.L <= m.Lmin * 1.01 ? ". Within 1% of the best." : ".");
      Kit.setText("sc-status", msg);
      var fmtE = function (v) { return v < 0.01 ? v.toExponential(2) : v < 10 ? Kit.fmt(v, 3) : Kit.fmt(v, 0); };
      Kit.$("sc-eqn").innerHTML = box("c-coral", fmtE(m.eig[0]), "steepest curvature &lambda;<sub>max</sub>") + op("&amp;") + box("c-navy", fmtE(m.eig[1]), "flattest curvature &lambda;<sub>min</sub>") + op("&rarr;") + box("c-teal", fmtE(2 / m.eig[0]), "&eta; must stay below 2 &divide; &lambda;<sub>max</sub>") +
        op("&rarr;") + box("c-sand", m.stepsBest < 0 ? "over 1,500,000" : m.stepsBest.toLocaleString("en-US"), "steps to reach 1% above the best loss, at &eta; = 1 &divide; &lambda;<sub>max</sub>");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 6. residual plots ----------
  function residFig() {
    var lx = X.map(Math.log), ly = Y.map(Math.log), lf = lsq(lx, ly), line = lsq(X, Y);
    var MODELS = [
      { name: "Always the average", pred: function (v) { return YBAR; } },
      { name: "Straight line", pred: function (v) { return line.w * v + line.b; } },
      { name: "Power curve", pred: function (v) { return Math.exp(lf.b) * Math.pow(v, lf.w); } },
      { name: "Power curve, residuals in logs", pred: function (v) { return Math.exp(lf.b) * Math.pow(v, lf.w); }, log: true }
    ];
    var k = 1, third = [Math.floor(N / 3), Math.floor(2 * N / 3)];
    Kit.chips(Kit.$("rs-bar"), MODELS.map(function (m) { return m.name; }), 1, function (i) { k = i; draw(); });
    function sd(a) { var m = mean(a); return Math.sqrt(mean(a.map(function (v) { return (v - m) * (v - m); }))); }
    function draw() {
      var m = MODELS[k], res = [], i;
      for (i = 0; i < N; i++) res.push(m.log ? ly[i] - Math.log(m.pred(X[i])) : Y[i] - m.pred(X[i]));
      var parts = [res.filter(function (v, i) { return X[i] < 32; }), res.filter(function (v, i) { return X[i] >= 32 && X[i] < 42; }), res.filter(function (v, i) { return X[i] >= 42; })];
      var means = parts.map(mean), sds = parts.map(sd), rmse = Math.sqrt(mean(res.map(function (v) { return v * v; })));
      var neg = 0; for (i = 0; i < N; i++) if (m.pred(X[i]) < 0) neg++;
      var cx = lin(16, 62, 80), edges = [16, 30, 36, 42, 48, 62], bx = [], by = [];
      for (i = 0; i < edges.length - 1; i++) { var sel = []; for (var j = 0; j < N; j++) if (X[j] >= edges[i] && X[j] < edges[i + 1]) sel.push(res[j]); if (sel.length >= 3) { bx.push(mean(X.filter(function (v) { return v >= edges[i] && v < edges[i + 1]; }))); by.push(mean(sel)); } }
      Kit.plot("rs-fit", function (P) {
        return [{ x: X, y: Y, mode: "markers", marker: { color: P.navy, size: 5, opacity: 0.5 }, hoverinfo: "skip" },
          { x: cx, y: cx.map(m.pred), mode: "lines", line: { color: P.teal, width: 3.5 }, hoverinfo: "skip" }];
      }, { margin: { l: 52, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { title: { text: "length, cm" }, range: [16, 62] }, yaxis: { title: { text: "weight, kg" }, range: [-0.15, 0.85] } });
      Kit.plot("rs-res", function (P) {
        return [{ x: X, y: res, mode: "markers", marker: { color: P.navy, size: 5, opacity: 0.5 }, hovertemplate: "%{x:.1f} cm, residual %{y:.3f}<extra></extra>" },
          { x: [16, 62], y: [0, 0], mode: "lines", line: { color: P.muted, width: 1.5 }, hoverinfo: "skip" },
          { x: bx, y: by, mode: "lines+markers", line: { color: P.coral, width: 3 }, marker: { color: P.coral, size: 9, line: { color: P.bg, width: 2 } }, hovertemplate: "average residual %{y:.3f}<extra></extra>" }];
      }, { margin: { l: 56, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { title: { text: "length, cm" }, range: [16, 62] }, yaxis: { title: { text: m.log ? "residual, ln y − ln ŷ" : "residual y − ŷ, kg" }, range: m.log ? [-0.6, 0.6] : [-0.2, 0.25], zeroline: false } });
      var verdict = ["The residuals climb with length: the model ignores length, so long sharks are under-predicted and short sharks over-predicted. The leftovers contain the whole length-weight story.",
        "The residuals form a U: positive for short sharks, negative in the middle, positive for long ones. The line is missing a bend. It also predicts a negative weight for " + neg + " short sharks.",
        "The U is gone and the average residual stays near zero. The spread fans out, though: small for short sharks and large for long ones, because errors grow with the size of the shark.",
        "In logs the residuals spread evenly across lengths, so the model is wrong by about the same percentage for a small shark and a large one."][k];
      Kit.setText("rs-verdict", verdict);
      Kit.$("rs-eqn").innerHTML = box("c-coral", Kit.fmt(rmse, m.log ? 3 : 4), m.log ? "typical miss in logs" : "typical miss, kg (&radic;MSE)") + op("&rarr;") + box("c-navy", Kit.fmt(sds[0], m.log ? 3 : 3), "spread of residuals, under 32 cm") + box("c-navy", Kit.fmt(sds[1], 3), "32 to 42 cm") + box("c-navy", Kit.fmt(sds[2], 3), "42 cm and over") + op("&rarr;") + box("c-sand", Kit.fmt(means[0], 3) + " / " + Kit.fmt(means[1], 3) + " / " + Kit.fmt(means[2], 3), "average residual in the three bands");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 7. the curve becomes a line on log axes ----------
  function logFig() {
    var lx = X.map(Math.log), ly = Y.map(Math.log), fit = lsq(lx, ly), line = lsq(X, Y), lxb = mean(lx), lyb = mean(ly);
    var w = 2.0, b = lyb - w * lxb, pivot = true, logAxes = false, showLine = false, tag = SP.filter(function (r) { return r.record_id === 3305; })[0];
    var sw = Kit.slider(Kit.$("ll-controls"), { label: "Exponent w (the slope in logs)", min: 0.5, max: 4.5, step: 0.01, value: w, digits: 2 }, function (v) { w = v; if (pivot) { b = lyb - w * lxb; sb.set(b); } draw(); });
    var sb = Kit.slider(Kit.$("ll-controls"), { label: "Log-bias b (so a = e^b)", min: -20, max: -4, step: 0.01, value: b, digits: 2 }, function (v) { b = v; draw(); });
    var bar = Kit.$("ll-bar");
    Kit.chips(bar, ["Linear axes", "Log axes"], 0, function (i) { logAxes = i === 1; draw(); });
    Kit.chips(bar, ["Hide the straight line", "Show the straight line"], 0, function (i) { showLine = i === 1; draw(); });
    Kit.chips(bar, ["b follows w (pivot at the average shark)", "Set b yourself"], 0, function (i) { pivot = i === 0; if (pivot) { b = lyb - w * lxb; sb.set(b); } sb.input.disabled = pivot; draw(); });
    Kit.button(bar, "Fit the line in logs", function () { w = fit.w; b = fit.b; sw.set(w); sb.set(b); draw(); });
    sb.input.disabled = true;
    function draw() {
      var cx = lin(15, 65, 120), mseLog = 0, mseKg = 0, i;
      for (i = 0; i < N; i++) { mseLog += Math.pow(w * lx[i] + b - ly[i], 2); mseKg += Math.pow(Math.exp(w * lx[i] + b) - Y[i], 2); }
      mseLog /= N; mseKg /= N;
      var lineX = cx.filter(function (v) { return line.w * v + line.b > 0.005; });
      Kit.plot("ll-plot", function (P) {
        var tr = [{ x: X, y: Y, mode: "markers", marker: { color: P.navy, size: 5.5, opacity: 0.55 }, hovertemplate: "%{x:.1f} cm, %{y:.3f} kg<extra>made-up record</extra>" }];
        if (showLine) tr.push({ x: lineX, y: lineX.map(function (v) { return line.w * v + line.b; }), mode: "lines", line: { color: P.sand, width: 3, dash: "dash" }, hoverinfo: "skip" });
        tr.push({ x: cx, y: cx.map(function (v) { return Math.exp(w * Math.log(v) + b); }), mode: "lines", line: { color: P.teal, width: 3.5 }, hoverinfo: "skip" });
        tr.push({ x: [tag.length_cm], y: [tag.weight_kg], mode: "markers+text", text: ["Tag 3305"], textposition: "top left", textfont: { color: P.coral, size: 12.5 }, marker: { color: P.paper, size: 10, line: { color: P.coral, width: 3 } }, hoverinfo: "skip" });
        return tr;
      }, logAxes
        ? { margin: { l: 58, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { type: "log", range: [Math.log10(16), Math.log10(64)], title: { text: "length, cm (log axis)" }, tickvals: [20, 30, 40, 50, 60], ticktext: ["20", "30", "40", "50", "60"] }, yaxis: { type: "log", range: [Math.log10(0.012), Math.log10(1)], title: { text: "weight, kg (log axis)" }, tickvals: [0.02, 0.05, 0.1, 0.2, 0.5, 1], ticktext: ["0.02", "0.05", "0.1", "0.2", "0.5", "1"] } }
        : { margin: { l: 58, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { type: "linear", range: [15, 65], title: { text: "length, cm" } }, yaxis: { type: "linear", range: [-0.12, 0.85], title: { text: "weight, kg" } } });
      Kit.setText("ll-w", Kit.fmt(w, 3)); Kit.setText("ll-a", (Math.exp(b) * 1e6).toFixed(2) + " &times; 10⁻⁶"); Kit.$("ll-a").innerHTML = (Math.exp(b) * 1e6).toFixed(2) + "&times;10<sup>&minus;6</sup>";
      Kit.setText("ll-mse", mseLog.toFixed(4)); Kit.setText("ll-pct", Kit.fmt(Math.sqrt(mseLog) * 100, 0) + "%"); Kit.setText("ll-rmse", Kit.fmt(Math.sqrt(mseKg), 4) + " kg"); Kit.setText("ll-rmse-line", Kit.fmt(Math.sqrt(line.mse), 4) + " kg");
      var lt = Math.log(tag.length_cm), zt = w * lt + b;
      Kit.$("ll-eqn").innerHTML = box("c-navy", Kit.fmt(lt, 3), "ln 41.0, the log of Tag 3305's length") + op("&times;") + box("c-sand", Kit.fmt(w, 2), "exponent w") + op("+") + box("c-sand", Kit.fmt(b, 2), "log-bias b") + op("=") + box("c-teal", Kit.fmt(zt, 3), "ln of the predicted weight") +
        op("&rarr;") + box("c-teal", Kit.fmt(Math.exp(zt), 3), "predicted weight, kg (e to that power)") + op("vs") + box("c-coral", "0.228", "true weight, kg");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 8. two inputs: a plane ----------
  function planeFig() {
    var gl = hasWebGL(), tag = SP.filter(function (r) { return r.record_id === 3305; })[0];
    function fitGiven(w2) { var yy = Y.map(function (v, i) { return v - w2 * G[i]; }), f = lsq(X, yy); return { w1: f.w, b: f.b, rmse: Math.sqrt(f.mse) }; }
    var grid = lin(0, 0.08, 161).map(function (v) { var f = fitGiven(v); return { w2: v, w1: f.w1, b: f.b, rmse: f.rmse }; }), bestI = 0, i;
    for (i = 0; i < grid.length; i++) if (grid[i].rmse < grid[bestI].rmse) bestI = i;
    var best = grid[bestI], w2 = 0;
    var sl = Kit.slider(Kit.$("mx-controls"), { label: "Weight on girth, w₂ (w₁ and b re-fitted)", min: 0, max: 0.08, step: 0.0005, value: 0, digits: 4 }, function (v) { w2 = v; draw(); });
    var bar = Kit.$("mx-bar");
    Kit.button(bar, "Length only (w₂ = 0)", function () { w2 = 0; sl.set(0); draw(); });
    Kit.button(bar, "Best plane", function () { w2 = Math.round(best.w2 * 2000) / 2000; sl.set(w2); draw(); });
    function draw() {
      var f = fitGiven(w2), gl2 = lsq(X, G), Lr = [20, 32, 44, 58], T = [-2.5, 0, 2.5];
      var gx = T.map(function () { return Lr; }), gy = T.map(function (t) { return Lr.map(function (l) { return gl2.w * l + gl2.b + t; }); });
      var gz = T.map(function (t, r) { return Lr.map(function (l, c) { return f.w1 * l + w2 * gy[r][c] + f.b; }); });
      if (!gl) Kit.$("mx-3d").textContent = "This browser cannot draw 3D graphics here. The chart on the right shows the error for the weights on girth.";
      else Kit.plot("mx-3d", function (P) {
        return [
          { type: "surface", x: gx, y: gy, z: gz, showscale: false, opacity: 0.6, hoverinfo: "skip", colorscale: [[0, P.teal], [1, P.teal]] },
          { type: "scatter3d", mode: "markers", x: X, y: G, z: Y, marker: { color: P.navy, size: 2.8, opacity: 0.85 }, hoverinfo: "skip" },
          { type: "scatter3d", mode: "markers", x: [tag.length_cm], y: [tag.girth_cm], z: [tag.weight_kg], marker: { color: P.coral, size: 6, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" }
        ];
      }, function () {
        return { margin: { l: Kit.width(Kit.$("mx-3d")) < 420 ? 30 : 10, r: 34, t: 0, b: 0 }, uirevision: "a2-plane", scene: { xaxis: { title: { text: "length, cm", font: { size: 12 } }, range: [15, 62] }, yaxis: { title: { text: "girth, cm", font: { size: 12 } }, range: [5, 18] }, zaxis: { title: { text: "weight, kg", font: { size: 12 } }, range: [0, 0.8] },
          aspectmode: "manual", aspectratio: Kit.width(Kit.$("mx-3d")) < 420 ? { x: 0.9, y: 0.6, z: 0.75 } : { x: 1.1, y: 0.7, z: 0.9 }, camera: { eye: Kit.width(Kit.$("mx-3d")) < 420 ? { x: 1.9, y: -1.8, z: 0.7 } : { x: 1.55, y: -1.5, z: 0.5 } } } };
      });
      Kit.plot("mx-prof", function (P) {
        return [{ x: grid.map(function (g) { return g.w2; }), y: grid.map(function (g) { return g.rmse; }), mode: "lines", line: { color: P.navy, width: 2.5 }, hoverinfo: "skip" },
          { x: [w2], y: [f.rmse], mode: "markers", marker: { color: P.coral, size: 11, line: { color: P.bg, width: 2 } }, hoverinfo: "skip" },
          { x: [best.w2], y: [best.rmse], mode: "markers", marker: { color: "rgba(0,0,0,0)", size: 18, line: { color: P.teal, width: 3 } }, hoverinfo: "skip" }];
      }, { margin: { l: 58, r: 10, t: 10, b: 46 }, dragmode: false, xaxis: { title: { text: "weight on girth, w₂" }, range: [0, 0.08] }, yaxis: { title: { text: "typical miss √MSE, kg", standoff: 6 }, automargin: true, range: [0.036, 0.062] } });
      var p = f.w1 * tag.length_cm + w2 * tag.girth_cm + f.b;
      Kit.$("mx-eqn").innerHTML = box("c-sand", Kit.fmt(f.w1 * tag.length_cm, 3), "w<sub>1</sub> &times; 41.0: " + sci(f.w1, 4) + " &times; length") + op("+") + box("c-sand", Kit.fmt(w2 * tag.girth_cm, 3), "w<sub>2</sub> &times; 11.5: " + sci(w2, 4) + " &times; girth") + op("+") + box("c-sand", Kit.fmt(f.b, 3), "bias b") +
        op("=") + box("c-teal", Kit.fmt(p, 3), "predicted weight, kg") + op("vs") + box("c-coral", "0.228", "true weight, kg");
      Kit.setText("mx-rmse", Kit.fmt(f.rmse, 4) + " kg"); Kit.setText("mx-w1", sci(f.w1, 4)); Kit.setText("mx-w2", sci(w2, 4)); Kit.setText("mx-b", Kit.fmt(f.b, 3));
      Kit.setText("mx-status", w2 === 0 ? "Girth is ignored: this is the straight line of section 1." : (Math.abs(w2 - best.w2) < 0.0006 ? "This is the best plane. The typical miss falls from " + Kit.fmt(grid[0].rmse, 4) + " to " + Kit.fmt(best.rmse, 4) + " kg." : (w2 > best.w2 ? "Too much weight on girth: the miss grows again." : "Giving girth some weight lowers the miss.")));
    }
    draw(); Kit.onTheme(draw);
  }
})();
