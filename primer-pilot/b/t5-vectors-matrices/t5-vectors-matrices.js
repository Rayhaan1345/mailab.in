// Chapter 0.5 Vectors and matrices: arrows (add, scale, length), a weighted sum, dot product and cosine,
// a matrix as a layer and as a mix of columns, a matrix bending the plane (determinant, eigenvectors),
// the matrix product with a dimension checker, the outer product, the inverse, and an eigenvector spinner.
(function () {
  "use strict";
  var $ = Kit.$, fmt = Kit.fmt;

  // ---------- helpers ----------
  function box(v, label, cls) { return '<span class="t' + (cls ? " c-" + cls : "") + '"><b>' + v + "</b><i>" + (label || "") + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function n1(v) { return fmt(v, 1); }
  function n2(v) { return fmt(v, 2); }
  function vec(v, d) { return "(" + fmt(v[0], d == null ? 1 : d) + ", " + fmt(v[1], d == null ? 1 : d) + ")"; }
  function svgPoint(svg, evt) { var pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY; var m = svg.getScreenCTM(); return m ? pt.matrixTransform(m.inverse()) : { x: 0, y: 0 }; }
  function addToggle(parent, label, on, fn) {
    var b = Kit.html("button", { type: "button", class: "btn", "aria-pressed": String(on) }, parent, label);
    b.addEventListener("click", function () { on = !on; b.setAttribute("aria-pressed", String(on)); b.style.background = on ? "var(--ink)" : ""; b.style.color = on ? "var(--bg)" : ""; fn(on); });
    b.style.background = on ? "var(--ink)" : ""; b.style.color = on ? "var(--bg)" : "";
    return b;
  }
  function setChip(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function matHTML(rows, opt) {
    opt = opt || {};
    var h = '<table class="mat' + (opt.pick ? " pick" : "") + '">';
    rows.forEach(function (r, i) { h += "<tr>"; r.forEach(function (v, j) { var c = opt.cls ? opt.cls(i, j) : ""; h += '<td' + (c ? ' class="' + c + '"' : "") + (opt.pick ? ' data-i="' + i + '" data-j="' + j + '"' : "") + ">" + (opt.fmt ? opt.fmt(v) : v) + "</td>"; }); h += "</tr>"; });
    return h + "</table>";
  }
  // 2x2 matrices as [a, b, c, d] in row-major order
  function mulv(M, v) { return [M[0] * v[0] + M[1] * v[1], M[2] * v[0] + M[3] * v[1]]; }
  function det(M) { return M[0] * M[3] - M[1] * M[2]; }
  function inv(M) { var d = det(M); return Math.abs(d) < 1e-9 ? null : [M[3] / d, -M[1] / d, -M[2] / d, M[0] / d]; }
  function mm(A, B) { return [A[0] * B[0] + A[1] * B[2], A[0] * B[1] + A[1] * B[3], A[2] * B[0] + A[3] * B[2], A[2] * B[1] + A[3] * B[3]]; }
  function unit(v) { var l = Math.hypot(v[0], v[1]) || 1; return [v[0] / l, v[1] / l]; }
  function eig(M) {
    var a = M[0], b = M[1], c = M[2], d = M[3], T = a + d, D = a * d - b * c, disc = T * T / 4 - D;
    if (disc < -1e-9) return [];
    var s = Math.sqrt(Math.max(0, disc)), l1 = T / 2 + s, l2 = T / 2 - s;
    if (s < 1e-6) {
      if (Math.abs(b) < 1e-9 && Math.abs(c) < 1e-9) return [{ val: l1, vec: [1, 0], all: true }];
      return [{ val: l1, vec: unit(Math.abs(b) > 1e-9 ? [b, l1 - a] : [l1 - d, c]) }];
    }
    return [l1, l2].map(function (l) {
      var v = Math.abs(b) > 1e-9 ? [b, l - a] : Math.abs(c) > 1e-9 ? [l - d, c] : (Math.abs(l - a) < 1e-9 ? [1, 0] : [0, 1]);
      return { val: l, vec: unit(v) };
    });
  }

  // ---------- a square plane with data coordinates, a clipped drawing layer and a layer for drag handles ----------
  function Plane(svg, R) {
    var S = 420, pad = 12, sc = (S - 2 * pad) / (2 * R), NS = "http://www.w3.org/2000/svg";
    svg.setAttribute("viewBox", "0 0 " + S + " " + S);
    svg.innerHTML = "";
    var defs = Kit.el("defs", {}, svg), cp = Kit.el("clipPath", { id: "clip-" + svg.id }, defs);
    Kit.el("rect", { x: pad, y: pad, width: S - 2 * pad, height: S - 2 * pad }, cp);
    var P = { S: S, R: R, sc: sc, svg: svg };
    P.X = function (x) { return S / 2 + x * sc; };
    P.Y = function (y) { return S / 2 - y * sc; };
    P.fromEvent = function (e) { var p = svgPoint(svg, e); return [(p.x - S / 2) / sc, (S / 2 - p.y) / sc]; };
    P.frame = Kit.el("rect", { x: pad, y: pad, width: S - 2 * pad, height: S - 2 * pad, fill: "none", "stroke-width": 1.2 }, svg);
    P.dyn = Kit.el("g", { "clip-path": "url(#clip-" + svg.id + ")" }, svg);
    P.top = Kit.el("g", {}, svg);
    P.reset = function () { P.dyn.innerHTML = ""; P.frame.setAttribute("stroke", Kit.palette().rule); };
    P.line = function (x1, y1, x2, y2, color, w, dash, op_) {
      return Kit.el("line", { x1: P.X(x1), y1: P.Y(y1), x2: P.X(x2), y2: P.Y(y2), stroke: color, "stroke-width": w || 1.5, "stroke-dasharray": dash || "0", "stroke-opacity": op_ == null ? 1 : op_, "stroke-linecap": "round" }, P.dyn);
    };
    P.axes = function (Pal) {
      for (var k = -R; k <= R; k++) { P.line(k, -R, k, R, k === 0 ? Pal.muted : Pal.rule, k === 0 ? 1.6 : 1); P.line(-R, k, R, k, k === 0 ? Pal.muted : Pal.rule, k === 0 ? 1.6 : 1); }
    };
    P.arrow = function (x1, y1, x2, y2, color, w, label, labelOff, dash) {
      var dx = P.X(x2) - P.X(x1), dy = P.Y(y2) - P.Y(y1), len = Math.hypot(dx, dy);
      if (len < 2) return;
      var ux = dx / len, uy = dy / len, hl = Math.min(12, len * 0.5), hw = 5.5, bx = P.X(x2) - ux * hl, by = P.Y(y2) - uy * hl;
      Kit.el("line", { x1: P.X(x1), y1: P.Y(y1), x2: bx, y2: by, stroke: color, "stroke-width": w || 3.5, "stroke-linecap": "round", "stroke-dasharray": dash || "0" }, P.dyn);
      Kit.el("polygon", { points: P.X(x2) + "," + P.Y(y2) + " " + (bx - uy * hw) + "," + (by + ux * hw) + " " + (bx + uy * hw) + "," + (by - ux * hw), fill: color }, P.dyn);
      if (label) {
        // Keep the label inside the plot: estimate its width, then flip it to the left of the tip or move it down at an edge.
        var o = labelOff || [8, -8], lw = label.length * 8.6 + 6, lx = P.X(x2) + o[0], ly = P.Y(y2) + o[1];
        if (lx + lw > S - pad - 2) lx = P.X(x2) - o[0] - lw;
        if (lx < pad + 2) lx = pad + 2;
        if (ly < pad + 16) ly = P.Y(y2) + 22;
        if (ly > S - pad - 4) ly = S - pad - 4;
        var t = Kit.el("text", { x: lx, y: ly, "font-size": 16, "font-weight": 600, fill: color, stroke: Kit.palette().paper, "stroke-width": 4, "paint-order": "stroke" }, P.dyn);
        t.textContent = label;
      }
    };
    P.text = function (x, y, s, color, size, anchor) {
      var t = Kit.el("text", { x: P.X(x), y: P.Y(y), "font-size": size || 14, fill: color, "text-anchor": anchor || "start", stroke: Kit.palette().paper, "stroke-width": 4, "paint-order": "stroke" }, P.dyn); t.textContent = s; return t;
    };
    // a draggable handle: get() returns [x, y]; set([x, y]); colorFn(Pal) gives the stroke colour
    P.handle = function (label, colorFn, get, set) {
      var c = Kit.el("circle", { class: "drag", r: 11, tabindex: 0, role: "slider", "aria-label": label, "stroke-width": 3.5 }, P.top), on = false, lim = R - 0.15;
      function clamp(v) { return Math.max(-lim, Math.min(lim, v)); }
      c.addEventListener("pointerdown", function (e) { on = true; c.setPointerCapture(e.pointerId); e.preventDefault(); });
      c.addEventListener("pointermove", function (e) { if (!on) return; var p = P.fromEvent(e); set([clamp(Math.round(p[0] * 20) / 20), clamp(Math.round(p[1] * 20) / 20)]); });
      c.addEventListener("pointerup", function () { on = false; });
      c.addEventListener("pointercancel", function () { on = false; });
      c.addEventListener("keydown", function (e) {
        var m = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[e.key];
        if (m) { var g = get(); set([clamp(g[0] + m[0]), clamp(g[1] + m[1])]); e.preventDefault(); }
      });
      return { el: c, place: function (visible) {
        var g = get(), Pal = Kit.palette();
        c.setAttribute("cx", P.X(g[0])); c.setAttribute("cy", P.Y(g[1])); c.setAttribute("fill", Pal.paper); c.setAttribute("stroke", colorFn(Pal));
        c.style.display = visible === false ? "none" : "";
      } };
    };
    return P;
  }

  document.addEventListener("DOMContentLoaded", function () {
    arrows(); weighted(); dotcos(); matvec();
    GridFig("g1", { kind: "det", start: "stretch" });
    products(); outer();
    GridFig("g2", { kind: "inverse", start: "symmetric" });
    GridFig("g3", { kind: "eigen", start: "symmetric" });
  });

  // =====================================================================
  // 1. Arrows: add, scale, length
  // =====================================================================
  function arrows() {
    var u = [2, 1], v = [1, 2.5], c = 1.5, mode = 0, pl = Plane($("va-svg"), 5), hv;
    var cs = Kit.slider($("va-controls"), { label: "Scalar c", min: -2, max: 3, step: 0.1, value: c }, function (x) { c = x; draw(); });
    Kit.chips($("va-bar"), ["add: u + v", "scale: c u", "length of u"], 0, function (i) { mode = i; draw(); });
    var hu = pl.handle("Tip of arrow u, drag or use arrow keys", function (P) { return P.navy; }, function () { return u; }, function (p) { u = p; draw(); });
    hv = pl.handle("Tip of arrow v, drag or use arrow keys", function (P) { return P.coral; }, function () { return v; }, function (p) { v = p; draw(); });
    function draw() {
      var P = Kit.palette(); pl.reset(); pl.axes(P);
      $("va-controls").style.display = mode === 1 ? "" : "none";
      var s = [u[0] + v[0], u[1] + v[1]], len = Math.hypot(u[0], u[1]), eq = $("va-eq"), st = $("va-st");
      if (mode === 0) {
        pl.line(u[0], u[1], s[0], s[1], P.coral, 1.8, "5 4"); pl.line(v[0], v[1], s[0], s[1], P.navy, 1.8, "5 4");
        pl.arrow(0, 0, s[0], s[1], P.teal, 4, "u + v", [8, -10]);
        pl.arrow(0, 0, v[0], v[1], P.coral, 3.5, "v"); pl.arrow(0, 0, u[0], u[1], P.navy, 3.5, "u");
        eq.innerHTML = box(vec(u), "u", "navy") + op("+") + box(vec(v), "v", "coral") + op("=") + box(vec(s), "u + v", "teal");
        st.textContent = "Add matching components: " + n1(u[0]) + " + " + n1(v[0]) + " = " + n1(s[0]) + " and " + n1(u[1]) + " + " + n1(v[1]) + " = " + n1(s[1]) + ".";
      } else if (mode === 1) {
        var k = [c * u[0], c * u[1]];
        pl.line(-u[0] * 6, -u[1] * 6, u[0] * 6, u[1] * 6, P.muted, 1.2, "3 4");
        pl.arrow(0, 0, k[0], k[1], P.teal, 4.5, "c u", [8, -10]); pl.arrow(0, 0, u[0], u[1], P.navy, 3, "u");
        eq.innerHTML = box(n2(c), "scalar c") + op("×") + box(vec(u), "u", "navy") + op("=") + box(vec(k), "c u", "teal");
        st.textContent = c < 0 ? "A negative scalar flips the arrow to the other side of the origin." : Math.abs(c) < 1 ? "A scalar between 0 and 1 shrinks the arrow." : "A scalar above 1 stretches the arrow along its own line.";
      } else {
        pl.line(0, 0, u[0], 0, P.coral, 2.5, "6 4"); pl.line(u[0], 0, u[0], u[1], P.coral, 2.5, "6 4");
        pl.arrow(0, 0, u[0], u[1], P.navy, 4, "length " + n2(len), [10, -10]);
        pl.text(u[0] / 2, -0.3, "u₁ = " + n1(u[0]), P.coral, 14, "middle"); pl.text(u[0] + 0.12, u[1] / 2, "u₂ = " + n1(u[1]), P.coral, 14);
        eq.innerHTML = box(n2(u[0] * u[0]), "u₁ squared") + op("+") + box(n2(u[1] * u[1]), "u₂ squared") + op("=") + box(n2(len * len), "sum") + op("so") + box(n2(len), "length of u", "navy");
        st.textContent = "The arrow is the hypotenuse of a right triangle with sides " + n1(Math.abs(u[0])) + " and " + n1(Math.abs(u[1])) + ".";
      }
      hu.place(); hv.place(mode === 0);
      pl.svg.setAttribute("aria-label", "Arrow u from the origin to " + vec(u) + (mode === 0 ? " and arrow v to " + vec(v) + ". Their sum is " + vec(s) + "." : mode === 1 ? ", and the scaled arrow " + n2(c) + " times u." : ". Its length is " + n2(len) + "."));
    }
    draw(); Kit.onTheme(draw);
  }

  // =====================================================================
  // 2. Weighted sum
  // =====================================================================
  function weighted() {
    var SH = [{ n: "Tag 4127", x: [44.0, 0.281, 12.4] }, { n: "Tag 3305", x: [41.0, 0.228, 11.5] }, { n: "Tag 5519", x: [82.0, 2.61, 27.6] }], cur = 0;
    var W = [0.05, 0.1, 0.1], b = -6, NM = ["length", "weight", "girth"], UN = ["cm", "kg", "cm"];
    var chips = Kit.chips($("ws-bar"), SH.map(function (s) { return s.n; }), 0, function (i) { cur = i; draw(); });
    var ctl = $("ws-controls");
    [["Weight on length", -0.1, 0.1, 0.005], ["Weight on weight", -1, 1, 0.05], ["Weight on girth", -0.2, 0.2, 0.01]].forEach(function (d, j) {
      Kit.slider(ctl, { label: d[0], min: d[1], max: d[2], step: d[3], value: W[j], digits: d[3] < 0.01 ? 3 : 2 }, function (v) { W[j] = v; draw(); });
    });
    Kit.slider(ctl, { label: "Bias b", min: -10, max: 5, step: 0.5, value: b, digits: 1 }, function (v) { b = v; draw(); });
    var svg = $("ws-svg"), vw = 380, vh = 230; svg.setAttribute("viewBox", "0 0 " + vw + " " + vh);
    function score(x) { return W[0] * x[0] + W[1] * x[1] + W[2] * x[2] + b; }
    function draw() {
      var P = Kit.palette(), x = SH[cur].x, terms = [W[0] * x[0], W[1] * x[1], W[2] * x[2], b], z = score(x);
      svg.innerHTML = "";
      var x0 = 190, scl = 11;   // pixels per unit of score
      Kit.el("line", { x1: x0, y1: 14, x2: x0, y2: 206, stroke: P.muted, "stroke-width": 1.5 }, svg);
      var rows = [NM[0] + " " + fmt(x[0], 1) + " " + UN[0], NM[1] + " " + fmt(x[1], 3) + " " + UN[1], NM[2] + " " + fmt(x[2], 1) + " " + UN[2], "bias", "score z"];
      var vals = terms.concat([z]);
      vals.forEach(function (v, i) {
        var y = 22 + i * 38, w = Math.min(Math.abs(v) * scl, 170), left = v < 0;
        Kit.el("rect", { x: left ? x0 - w : x0, y: y, width: Math.max(w, 1), height: 22, rx: 3, fill: i === 4 ? P.teal : P.navy, "fill-opacity": i === 4 ? 0.9 : 0.75 }, svg);
        var t = Kit.el("text", { x: left ? x0 - w - 6 : x0 + w + 6, y: y + 16, "text-anchor": left ? "end" : "start", "font-size": 14, "font-weight": 600, fill: P.ink }, svg);
        t.textContent = (v > 0 ? "+" : "") + fmt(v, 2);
        var l = Kit.el("text", { x: left ? x0 + 6 : x0 - 6, y: y + 16, "text-anchor": left ? "start" : "end", "font-size": 13, fill: P.muted }, svg);
        l.textContent = rows[i];
        if (i === 3) Kit.el("line", { x1: 10, y1: y + 33, x2: vw - 10, y2: y + 33, stroke: P.rule, "stroke-width": 1 }, svg);
      });
      svg.setAttribute("aria-label", "Bars for the pieces of the score of " + SH[cur].n + ": length adds " + fmt(terms[0], 2) + ", weight " + fmt(terms[1], 2) + ", girth " + fmt(terms[2], 2) + ", bias " + fmt(b, 1) + ". The score is " + fmt(z, 2) + ".");
      $("ws-eq").innerHTML = box(fmt(W[0], 3), "weight on length", "") + op("×") + box(fmt(x[0], 1), "length, cm", "navy") + op("+") + box(fmt(W[1], 2), "weight on weight") + op("×") + box(fmt(x[1], 3), "weight, kg", "navy") + op("+") + box(fmt(W[2], 2), "weight on girth") + op("×") + box(fmt(x[2], 1), "girth, cm", "navy") + op("+") + box(fmt(b, 1), "bias b") + op("=") + box(fmt(z, 2), "score z", "teal");
      var tb = $("ws-table"); tb.innerHTML = "<thead><tr><th>made-up records</th><th>length</th><th>weight</th><th>girth</th><th>score z</th></tr></thead>";
      var body = document.createElement("tbody"); tb.appendChild(body);
      SH.forEach(function (s, i) {
        var tr = document.createElement("tr"); if (i === cur) tr.className = "on";
        tr.innerHTML = "<td>" + s.n + "</td><td>" + fmt(s.x[0], 1) + "</td><td>" + fmt(s.x[1], 3) + "</td><td>" + fmt(s.x[2], 1) + "</td><td>" + fmt(score(s.x), 2) + "</td>";
        tr.addEventListener("click", function () { cur = i; setChip(chips, i); draw(); }); body.appendChild(tr);
      });
    }
    draw(); Kit.onTheme(draw);
  }

  // =====================================================================
  // 3. Dot product, length, angle, cosine
  // =====================================================================
  function dotcos() {
    var u = [3, 1], v = [1, 3], pl = Plane($("dc-svg"), 5);
    var PRE = [["same direction", [2, 1.5], [3, 2.25]], ["perpendicular", [2, 1], [-1, 2]], ["opposite", [2, 1], [-2, -1]], ["start again", [3, 1], [1, 3]]];
    Kit.chips($("dc-bar"), PRE.map(function (p) { return p[0]; }), 3, function (i) { u = PRE[i][1].slice(); v = PRE[i][2].slice(); draw(); });
    var hu = pl.handle("Tip of arrow u, drag or use arrow keys", function (P) { return P.navy; }, function () { return u; }, function (p) { u = p; draw(); });
    var hv = pl.handle("Tip of arrow v, drag or use arrow keys", function (P) { return P.coral; }, function () { return v; }, function (p) { v = p; draw(); });
    function draw() {
      var P = Kit.palette(); pl.reset(); pl.axes(P);
      var lu = Math.hypot(u[0], u[1]), lv = Math.hypot(v[0], v[1]), dot = u[0] * v[0] + u[1] * v[1];
      var cos = lu > 1e-9 && lv > 1e-9 ? Math.max(-1, Math.min(1, dot / (lu * lv))) : null, th = cos == null ? null : Math.acos(cos) * 180 / Math.PI;
      if (lv > 1e-9) {
        var k = dot / (lv * lv), px = k * v[0], py = k * v[1];
        pl.line(-v[0] * 6, -v[1] * 6, v[0] * 6, v[1] * 6, P.muted, 1.2, "3 4");
        pl.line(u[0], u[1], px, py, P.muted, 1.8, "4 4");
        pl.arrow(0, 0, px, py, P.sand, 6, "", null);
      }
      // angle arc
      if (cos != null) {
        var a1 = Math.atan2(u[1], u[0]), a2 = Math.atan2(v[1], v[0]), r = 0.9, d = a2 - a1; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI;
        var pts = []; for (var i = 0; i <= 24; i++) { var a = a1 + d * i / 24; pts.push(pl.X(r * Math.cos(a)) + "," + pl.Y(r * Math.sin(a))); }
        Kit.el("polyline", { points: pts.join(" "), fill: "none", stroke: P.ink2, "stroke-width": 2 }, pl.dyn);
        var am = a1 + d / 2; pl.text(1.35 * Math.cos(am), 1.35 * Math.sin(am) - 0.1, "θ", P.ink2, 17, "middle");
      }
      pl.arrow(0, 0, v[0], v[1], P.coral, 3.5, "v"); pl.arrow(0, 0, u[0], u[1], P.navy, 3.5, "u");
      hu.place(); hv.place();
      $("dc-eq").innerHTML = box(n1(u[0]) + " × " + n1(v[0]) + " + " + n1(u[1]) + " × " + n1(v[1]), "u · v by components") + op("=") + box(n2(dot), "dot product u · v", "teal")
        + op("&nbsp;") + box(n2(lu), "length of u", "navy") + box(n2(lv), "length of v", "coral") + op("&nbsp;")
        + box(th == null ? "n/a" : fmt(th, 0) + "°", "angle θ") + box(cos == null ? "n/a" : n2(cos), "cosine similarity", "teal");
      var st = "";
      if (cos == null) st = "One arrow has length 0, so it has no direction and the cosine is undefined.";
      else if (cos > 0.995) st = "The arrows point the same way. The cosine is 1, whatever the lengths.";
      else if (cos < -0.995) st = "The arrows point in opposite directions. The cosine is −1.";
      else if (Math.abs(cos) < 0.02) st = "The arrows are perpendicular. The dot product is 0.";
      else st = (cos > 0 ? "The arrows lean the same way" : "The arrows lean apart") + ". The sand segment is the shadow of u on the line of v: its length is " + n2(lv > 1e-9 ? dot / lv : 0) + ".";
      $("dc-st").textContent = st;
      pl.svg.setAttribute("aria-label", "Arrows u " + vec(u) + " and v " + vec(v) + ". Dot product " + n2(dot) + ", cosine similarity " + (cos == null ? "undefined" : n2(cos)) + ".");
    }
    draw(); Kit.onTheme(draw);
  }

  // =====================================================================
  // 4. Matrix times vector: rows as neurons, columns as a mix
  // =====================================================================
  function matvec() {
    var A = [2, 1, 0, 3], x = [1, 2], ctl = $("mv-controls"), pl = Plane($("mv-plane"), 7), net = $("mv-net");
    net.setAttribute("viewBox", "0 0 300 230");
    [["a", 0], ["b", 1], ["c", 2], ["d", 3]].forEach(function (e) { Kit.slider(ctl, { label: "Matrix entry " + e[0], min: -3, max: 3, step: 0.1, value: A[e[1]] }, function (v) { A[e[1]] = v; draw(); }); });
    [["x₁", 0], ["x₂", 1]].forEach(function (e) { Kit.slider(ctl, { label: "Input " + e[0], min: -2, max: 2, step: 0.1, value: x[e[1]] }, function (v) { x[e[1]] = v; draw(); }); });
    function draw() {
      var P = Kit.palette(), y = mulv(A, x), c1 = [A[0], A[2]], c2 = [A[1], A[3]];
      // network of two neurons
      net.innerHTML = "";
      var inY = [70, 160], outY = [70, 160], xi = 52, xo = 248;
      [[0, 0, A[0]], [1, 0, A[1]], [0, 1, A[2]], [1, 1, A[3]]].forEach(function (e) {
        var w = e[2], tt = e[0] === e[1] ? 0.5 : 0.22, mid = [xi + 20 + tt * (xo - xi - 40), inY[e[0]] + tt * (outY[e[1]] - inY[e[0]]) - (e[0] === e[1] ? 9 : 0)];
        Kit.el("line", { x1: xi + 20, y1: inY[e[0]], x2: xo - 20, y2: outY[e[1]], stroke: e[0] === 0 ? P.navy : P.coral, "stroke-width": 1 + Math.min(Math.abs(w), 3) * 1.4, "stroke-dasharray": w < 0 ? "6 4" : "0", "stroke-opacity": 0.75 }, net);
        var t = Kit.el("text", { x: mid[0], y: mid[1] + 4, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: P.ink, stroke: P.paper, "stroke-width": 4, "paint-order": "stroke" }, net); t.textContent = n1(w);
      });
      [0, 1].forEach(function (i) {
        Kit.el("circle", { cx: xi, cy: inY[i], r: 20, fill: P.paper, stroke: i === 0 ? P.navy : P.coral, "stroke-width": 3 }, net);
        var t = Kit.el("text", { x: xi, y: inY[i] + 5, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: P.ink }, net); t.textContent = n1(x[i]);
        var l = Kit.el("text", { x: xi, y: inY[i] - 28, "text-anchor": "middle", "font-size": 13, fill: P.muted }, net); l.textContent = "x" + (i + 1);
        Kit.el("circle", { cx: xo, cy: outY[i], r: 20, fill: P.paper, stroke: P.teal, "stroke-width": 3 }, net);
        var t2 = Kit.el("text", { x: xo, y: outY[i] + 5, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: P.ink }, net); t2.textContent = n1(y[i]);
        var l2 = Kit.el("text", { x: xo, y: outY[i] - 28, "text-anchor": "middle", "font-size": 13, fill: P.muted }, net); l2.textContent = "y" + (i + 1) + " (neuron " + (i + 1) + ")";
      });
      var cap = Kit.el("text", { x: 150, y: 218, "text-anchor": "middle", "font-size": 13, fill: P.muted }, net); cap.textContent = "numbers on the lines are the matrix entries";
      net.setAttribute("aria-label", "Two inputs " + n1(x[0]) + " and " + n1(x[1]) + " feed two neurons whose weights are the rows of the matrix. The outputs are " + n1(y[0]) + " and " + n1(y[1]) + ".");
      // columns mixing
      pl.reset(); pl.axes(P);
      var s1 = [x[0] * c1[0], x[0] * c1[1]], s2 = [x[1] * c2[0], x[1] * c2[1]];
      pl.arrow(0, 0, c1[0], c1[1], P.navy, 1.8, "", null, "2 4"); pl.arrow(0, 0, c2[0], c2[1], P.coral, 1.8, "", null, "2 4");
      pl.arrow(0, 0, s1[0], s1[1], P.navy, 4, "x₁ × col 1", [6, 16]);
      pl.arrow(s1[0], s1[1], s1[0] + s2[0], s1[1] + s2[1], P.coral, 4, "x₂ × col 2", [6, -8]);
      pl.arrow(0, 0, y[0], y[1], P.teal, 4.5, "output", [10, -10]);
      pl.svg.setAttribute("aria-label", "The matrix columns as arrows. The output arrow " + vec(y) + " is x one times column one plus x two times column two.");
      $("mv-eq").innerHTML = "";
      var r1 = document.createElement("div"); r1.className = "eqn"; r1.innerHTML = box(n1(y[0]), "y₁", "teal") + op("=") + box(n1(A[0]), "a", "navy") + op("×") + box(n1(x[0]), "x₁", "navy") + op("+") + box(n1(A[1]), "b", "coral") + op("×") + box(n1(x[1]), "x₂", "coral");
      var r2 = document.createElement("div"); r2.className = "eqn"; r2.innerHTML = box(n1(y[1]), "y₂", "teal") + op("=") + box(n1(A[2]), "c", "navy") + op("×") + box(n1(x[0]), "x₁", "navy") + op("+") + box(n1(A[3]), "d", "coral") + op("×") + box(n1(x[1]), "x₂", "coral");
      $("mv-eq").appendChild(r1); $("mv-eq").appendChild(r2);
      $("mv-st").textContent = "Rows: y = (" + n1(y[0]) + ", " + n1(y[1]) + "). Columns: " + n1(x[0]) + " × " + vec(c1) + " + " + n1(x[1]) + " × " + vec(c2) + " = " + vec(y) + ". Same answer.";
    }
    draw(); Kit.onTheme(draw);
  }

  // =====================================================================
  // 5, 8, 9. Grid figure: a 2x2 matrix bends the plane (determinant, eigenvectors, inverse, test arrow)
  // =====================================================================
  var PRESETS = {
    "identity": [1, 0, 0, 1], "stretch": [2, 0, 0, 3], "shear": [1, 1, 0, 1], "rotate": [0.7, -0.7, 0.7, 0.7],
    "reflect": [-1, 0, 0, 1], "squash": [1, 1, 1, 1], "symmetric": [2, 1, 1, 2]
  };
  function matRow(label, M, d) { return '<div class="mrow"><span class="nm">' + label + "</span>" + matHTML([[fmt(M[0], d), fmt(M[1], d)], [fmt(M[2], d), fmt(M[3], d)]]) + "</div>"; }

  function GridFig(id, cfg) {
    var root = $(id), M = PRESETS[cfg.start || (cfg.kind === "det" ? "shear" : "identity")].slice(), t = 1, stage = 1, theta = 40, stopAnim = null, showEig = cfg.kind === "eigen", showCircle = false;
    var grid = Kit.html("div", { class: "mv-grid" }, root), left = Kit.html("div", {}, grid), right = Kit.html("div", {}, grid);
    var svg = Kit.el("svg", { id: id + "-svg", class: "sq", role: "img", "aria-label": "A grid bent by a 2 by 2 matrix" }, null); right.appendChild(svg);
    var pl = Plane(svg, 5);
    var bar = Kit.html("div", { class: "bar" }, left), chips = Kit.chips(bar, Object.keys(PRESETS), Object.keys(PRESETS).indexOf(cfg.start || (cfg.kind === "det" ? "shear" : "identity")), function (i) {
      M = PRESETS[Object.keys(PRESETS)[i]].slice(); t = 1; syncSliders(); draw();
    });
    var bar2 = Kit.html("div", { class: "bar" }, left);
    if (cfg.kind === "det") { Kit.button(bar2, "show the move", function () { play(); }); addToggle(bar2, "eigenvectors", false, function (on) { showEig = on; draw(); }); addToggle(bar2, "unit circle", false, function (on) { showCircle = on; draw(); }); }
    if (cfg.kind === "eigen") { Kit.button(bar2, "snap to eigenvector 1", function () { snap(0); }); Kit.button(bar2, "snap to eigenvector 2", function () { snap(1); }); }
    var ctl = Kit.html("div", { class: "controls" }, left), sl = [];
    ["a", "b", "c", "d"].forEach(function (nm, k) {
      sl.push(Kit.slider(ctl, { label: "Entry " + nm, min: -3, max: 3, step: 0.1, value: M[k] }, function (v) { M[k] = v; t = 1; if (stopAnim) { stopAnim(); stopAnim = null; } setChip(chips, -1); draw(); }));
    });
    var stageSl = null, thSl = null;
    if (cfg.kind === "inverse") stageSl = Kit.slider(ctl, { label: "Stage: 0 start, 1 after A, 2 after A and then its inverse", min: 0, max: 2, step: 0.02, value: 1, digits: 2 }, function (v) { stage = v; draw(); });
    if (cfg.kind === "eigen") thSl = Kit.slider(ctl, { label: "Angle of the test arrow, degrees", min: 0, max: 360, step: 1, value: theta, digits: 0 }, function (v) { theta = v; draw(); });
    var eqBox = Kit.html("div", {}, left), st = Kit.html("p", { class: "status", "aria-live": "polite" }, left);
    function syncSliders() { M.forEach(function (v, k) { sl[k].set(v); }); }
    function play() { if (stopAnim) stopAnim(); t = 0; stopAnim = Kit.animate(1200, function (p) { t = Kit.ease(p); draw(); }); }
    function snap(k) { var E = eig(M); if (!E[k]) { st.textContent = "This matrix has no eigenvector number " + (k + 1) + "."; return; } var a = Math.atan2(E[k].vec[1], E[k].vec[0]) * 180 / Math.PI; if (a < 0) a += 360; theta = Math.round(a); thSl.set(theta); draw(); }

    function effective() {
      if (cfg.kind === "inverse") {
        var Ai = inv(M);
        if (stage <= 1 || !Ai) return [1 + Math.min(stage, 1) * (M[0] - 1), Math.min(stage, 1) * M[1], Math.min(stage, 1) * M[2], 1 + Math.min(stage, 1) * (M[3] - 1)];
        var s = stage - 1, L = [1 + s * (Ai[0] - 1), s * Ai[1], s * Ai[2], 1 + s * (Ai[3] - 1)];
        return mm(M, L);
      }
      return [1 + t * (M[0] - 1), t * M[1], t * M[2], 1 + t * (M[3] - 1)];
    }
    function draw() {
      var P = Kit.palette(), Me = effective(), dt = det(Me), dM = det(M); pl.reset();
      pl.axes(P);
      var a = Me[0], b = Me[1], c = Me[2], d = Me[3], L = 14;
      // transformed grid
      for (var k = -9; k <= 9; k++) {
        var w = k === 0 ? 2.4 : 1.2, o = k === 0 ? 0.95 : (cfg.kind === "eigen" ? 0.22 : 0.5);
        pl.line(a * k - b * L, c * k - d * L, a * k + b * L, c * k + d * L, P.navy, w, "0", o);
        pl.line(-a * L + b * k, -c * L + d * k, a * L + b * k, c * L + d * k, P.navy, w, "0", o);
      }
      // unit square: original outline, transformed fill
      var sq = [[0, 0], [1, 0], [1, 1], [0, 1]], img = sq.map(function (p) { return [a * p[0] + b * p[1], c * p[0] + d * p[1]]; });
      Kit.el("polygon", { points: sq.map(function (p) { return pl.X(p[0]) + "," + pl.Y(p[1]); }).join(" "), fill: "none", stroke: P.muted, "stroke-width": 1.6, "stroke-dasharray": "5 4" }, pl.dyn);
      var pos = dt >= 0;
      Kit.el("polygon", { points: img.map(function (p) { return pl.X(p[0]) + "," + pl.Y(p[1]); }).join(" "), fill: pos ? P.teal : P.coral, "fill-opacity": 0.28, stroke: pos ? P.teal : P.coral, "stroke-width": 2.2 }, pl.dyn);
      if (showCircle) {
        var pts = []; for (var i = 0; i <= 72; i++) { var th = i / 72 * 2 * Math.PI, px = Math.cos(th), py = Math.sin(th); pts.push(pl.X(a * px + b * py) + "," + pl.Y(c * px + d * py)); }
        Kit.el("polyline", { points: pts.join(" "), fill: "none", stroke: P.sand, "stroke-width": 2.5 }, pl.dyn);
      }
      // eigen lines
      var E = eig(Me);
      if (showEig && cfg.kind !== "inverse") E.forEach(function (e, i) {
        pl.line(-e.vec[0] * L, -e.vec[1] * L, e.vec[0] * L, e.vec[1] * L, P.sand, 2.6, "7 5");
        var sg = 1; if (cfg.kind === "eigen") { var rd0 = theta * Math.PI / 180, tv0 = mulv(Me, [Math.cos(rd0), Math.sin(rd0)]); sg = (e.vec[0] * tv0[0] + e.vec[1] * tv0[1]) > 0 ? -1 : 1; } pl.text(sg * e.vec[0] * 3.3 + 0.12, sg * e.vec[1] * 3.3 + 0.12, "v" + (i + 1) + ", λ = " + n2(e.val), P.ink2, 13, sg * e.vec[0] > 0 ? "end" : "start");
      });
      // basis arrows
      pl.arrow(0, 0, 1, 0, P.coral, 1.5, "", null, "2 4"); pl.arrow(0, 0, 0, 1, P.teal, 1.5, "", null, "2 4");
      pl.arrow(0, 0, a, c, P.coral, 4, cfg.kind !== "det" ? "" : "col 1", [8, 14]); pl.arrow(0, 0, b, d, P.teal, 4, cfg.kind !== "det" ? "" : "col 2", [8, -8]);
      // eigen test arrow
      var res = "";
      if (cfg.kind === "eigen") {
        var rad = theta * Math.PI / 180, v = [1.2 * Math.cos(rad), 1.2 * Math.sin(rad)], Av = mulv(Me, v);
        pl.line(-v[0] * 6, -v[1] * 6, v[0] * 6, v[1] * 6, P.muted, 1.2, "3 4");
        pl.arrow(0, 0, Av[0], Av[1], P.coral, 4.5, "A v", [8, -8]); pl.arrow(0, 0, v[0], v[1], P.navy, 4, "v", [8, -8]);
        var lv = Math.hypot(v[0], v[1]), la = Math.hypot(Av[0], Av[1]), cs = la > 1e-9 ? (v[0] * Av[0] + v[1] * Av[1]) / (lv * la) : 1, ang = Math.acos(Math.max(-1, Math.min(1, cs))) * 180 / Math.PI;
        var aligned = la < 1e-9 || ang < 1.5 || ang > 178.5, lam = (v[0] * Av[0] + v[1] * Av[1]) / (lv * lv);
        eqBox.innerHTML = '<div class="eqn">' + box("(" + n2(v[0] / 1.2) + ", " + n2(v[1] / 1.2) + ")", "direction of v", "navy") + op("→") + box("(" + n2(Av[0] / 1.2) + ", " + n2(Av[1] / 1.2) + ")", "A v for unit v", "coral") + op("&nbsp;") + box(fmt(ang, 0) + "°", "angle between v and A v") + box(n2(la / lv), "stretch |A v| / |v|") + "</div>";
        if (aligned && la > 1e-9) eqBox.innerHTML += '<div class="eqn">' + box(vec([Av[0] / 1.2, Av[1] / 1.2], 2), "A v", "coral") + op("=") + box(n2(lam), "λ", "teal") + op("×") + box(vec([v[0] / 1.2, v[1] / 1.2], 2), "v", "navy") + "</div>";
        res = aligned && la > 1e-9 ? "On an eigenvector: A v points along v, and the stretch factor is λ = " + n2(lam) + "." : "Off an eigenvector: A v points " + fmt(ang, 0) + " degrees away from v.";
        if (aligned && la <= 1e-9) res = "A v is the zero arrow here, because this direction is squashed to a point.";
      }
      if (cfg.kind === "det") {
        eqBox.innerHTML = '<div class="eqn">' + box(n2(M[0]), "a", "coral") + op("×") + box(n2(M[3]), "d", "teal") + op("−") + box(n2(M[1]), "b", "teal") + op("×") + box(n2(M[2]), "c", "coral") + op("=") + box(n2(dM), "determinant", dM >= 0 ? "teal" : "coral") + "</div>";
        res = Math.abs(dM) < 0.05 ? "Determinant near 0: the grid collapses onto a line (or a point), so the unit square has no area." : "The unit square becomes a parallelogram of area " + n2(Math.abs(dM)) + (dM < 0 ? ", and the sign says the plane is flipped over." : ".");
        if (showEig) { var E0 = eig(M); res += E0.length ? " Eigenvectors (sand): " + E0.map(function (e, i) { return "v" + (i + 1) + " " + vec(e.vec, 2) + " with λ = " + n2(e.val); }).join("; ") + "." : " This matrix has no eigenvectors in the plane: it turns all directions."; }
      }
      if (cfg.kind === "inverse") {
        var Ai = inv(M), prod = Ai ? mm(Ai, M) : null;
        eqBox.innerHTML = matRow("A =", M, 2) + (Ai ? matRow("inverse of A =", Ai, 2) + matRow("inverse of A times A =", [Math.abs(prod[0]) < 5e-4 ? 0 : prod[0], Math.abs(prod[1]) < 5e-4 ? 0 : prod[1], Math.abs(prod[2]) < 5e-4 ? 0 : prod[2], Math.abs(prod[3]) < 5e-4 ? 0 : prod[3]], 2) : '<p class="status" style="color:var(--coral)">The determinant is ' + n2(dM) + ', so there is no inverse.</p>');
        if (!Ai && stageSl && stageSl.get() > 1) { stage = 1; stageSl.set(1); }
        res = !Ai ? "No inverse: the plane is squashed onto a line, and many points share one place." : stage <= 0.02 ? "Stage 0: the plane before any move." : stage < 1 ? "Bending towards A." : stage === 1 ? "Stage 1: the grid after A. Area scale: " + n2(dM) + "." : stage < 2 ? "Now applying the inverse of A. The grid is straightening." : "Stage 2: A and then its inverse give the identity. The grid is back.";
      }
      st.textContent = res;
      svg.setAttribute("aria-label", "A grid bent by the matrix with rows " + n2(M[0]) + ", " + n2(M[1]) + " and " + n2(M[2]) + ", " + n2(M[3]) + ". Determinant " + n2(dM) + ".");
    }
    draw(); Kit.onTheme(draw);
  }

  // =====================================================================
  // 6. Matrix product with a dimension checker
  // =====================================================================
  function products() {
    var root = $("mp"), S = { rA: 2, cA: 3, rB: 3, cB: 4, tA: false, tB: false, i: 0, j: 0 };
    var ctl = Kit.html("div", { class: "controls" }, root);
    [["Rows of A", "rA"], ["Columns of A", "cA"], ["Rows of B", "rB"], ["Columns of B", "cB"]].forEach(function (e) { Kit.slider(ctl, { label: e[0], min: 1, max: 4, step: 1, value: S[e[1]], digits: 0 }, function (v) { S[e[1]] = v; S.i = 0; S.j = 0; draw(); }); });
    var bar = Kit.html("div", { class: "bar" }, root);
    addToggle(bar, "transpose A", false, function (on) { S.tA = on; S.i = 0; S.j = 0; draw(); });
    addToggle(bar, "transpose B", false, function (on) { S.tB = on; S.i = 0; S.j = 0; draw(); });
    var view = Kit.html("div", { class: "mrow", style: "align-items:flex-start" }, root), eq = Kit.html("div", {}, root), st = Kit.html("p", { class: "status", "aria-live": "polite" }, root);
    function baseA(i, j) { return ((2 * i + 3 * j + 1) % 5) - 2; }
    function baseB(i, j) { return ((i + 2 * j + 2) % 4) - 1; }
    function build(r, c, f, tr) { var out = []; var R = tr ? c : r, C = tr ? r : c; for (var i = 0; i < R; i++) { out.push([]); for (var j = 0; j < C; j++) out[i].push(tr ? f(j, i) : f(i, j)); } return out; }
    function times(A, B) { var out = []; for (var i = 0; i < A.length; i++) { out.push([]); for (var j = 0; j < B[0].length; j++) { var s = 0; for (var k = 0; k < B.length; k++) s += A[i][k] * B[k][j]; out[i].push(s); } } return out; }
    function sizeTxt(M) { return M.length + " × " + M[0].length; }
    function draw() {
      var A = build(S.rA, S.cA, baseA, S.tA), B = build(S.rB, S.cB, baseB, S.tB), ok = A[0].length === B.length, C = ok ? times(A, B) : null;
      var i = Math.min(S.i, A.length - 1), j = ok ? Math.min(S.j, B[0].length - 1) : 0;
      var h = '<div><div class="nm">A' + (S.tA ? " transposed" : "") + "</div>" + matHTML(A, { cls: function (r) { return ok && r === i ? "hl" : ""; } }) + '<div class="nm" style="font-size:14px;color:var(--muted)">' + sizeTxt(A) + "</div></div>" + op("×")
        + '<div><div class="nm">B' + (S.tB ? " transposed" : "") + "</div>" + matHTML(B, { cls: function (r, c) { return ok && c === j ? "cs" : ""; } }) + '<div class="nm" style="font-size:14px;color:var(--muted)">' + sizeTxt(B) + "</div></div>" + op("=");
      if (ok) h += '<div><div class="nm">AB (click an entry)</div>' + matHTML(C, { pick: true, cls: function (r, c) { return r === i && c === j ? "sel" : ""; } }) + '<div class="nm" style="font-size:14px;color:var(--muted)">' + sizeTxt(C) + "</div></div>";
      else h += '<div style="color:var(--coral);font-weight:600;padding-top:1.4em">undefined</div>';
      view.innerHTML = h;
      view.querySelectorAll("td[data-i]").forEach(function (td) { td.addEventListener("click", function () { S.i = +td.getAttribute("data-i"); S.j = +td.getAttribute("data-j"); draw(); }); });
      if (ok) {
        var parts = "";
        for (var k = 0; k < B.length; k++) parts += (k ? op("+") : "") + box(A[i][k], "A[" + (i + 1) + "," + (k + 1) + "]", "coral") + op("×") + box(B[k][j], "B[" + (k + 1) + "," + (j + 1) + "]", "navy");
        eq.innerHTML = '<div class="eqn">' + box(C[i][j], "entry (" + (i + 1) + ", " + (j + 1) + ") of AB", "teal") + op("=") + parts + "</div>";
      } else eq.innerHTML = "";
      var msg;
      if (!ok) msg = "Sizes: (" + sizeTxt(A) + ") times (" + sizeTxt(B) + "). The inner sizes " + A[0].length + " and " + B.length + " differ, so AB does not exist.";
      else {
        msg = "Sizes: (" + sizeTxt(A) + ")(" + sizeTxt(B) + ") = (" + sizeTxt(C) + "). The inner sizes match.";
        if (B[0].length === A.length) { var D = times(B, A); msg += " BA exists too and is " + sizeTxt(D) + (D.length === C.length && D[0].length === C[0].length ? (JSON.stringify(D) === JSON.stringify(C) ? ", and here it equals AB." : ", and it differs from AB.") : ", a different size from AB."); }
        else msg += " BA does not exist: B has " + B[0].length + " columns and A has " + A.length + " rows.";
      }
      st.textContent = msg;
    }
    draw();
  }

  // =====================================================================
  // 7. Outer product
  // =====================================================================
  function outer() {
    var root = $("op"), U = [1, 2, 0], V = [3, 1, -1, 2], mode = 0, sel = [1, 0];
    var W0 = [[1, 0, 2, -1], [0, 1, 1, 2], [2, -1, 0, 1]];
    var bar = Kit.html("div", { class: "bar" }, root), ctl = Kit.html("div", { class: "controls" }, root);
    Kit.chips(bar, ["outer product u vT", "add it to a weight matrix W"], 0, function (i) { mode = i; draw(); });
    U.forEach(function (v, i) { Kit.slider(ctl, { label: "u" + (i + 1), min: -3, max: 3, step: 1, value: v, digits: 0 }, function (x) { U[i] = x; draw(); }); });
    V.forEach(function (v, j) { Kit.slider(ctl, { label: "v" + (j + 1), min: -3, max: 3, step: 1, value: v, digits: 0 }, function (x) { V[j] = x; draw(); }); });
    var svg = Kit.el("svg", { id: "op-svg", class: "sq", role: "img", "aria-label": "A table whose entries are the products of the numbers in u and v" }, null); root.appendChild(svg);
    var eq = Kit.html("div", { class: "eqn" }, root), st = Kit.html("p", { class: "status", "aria-live": "polite" }, root);
    var vw = 400, vh = 270, cw = 74, ch = 56, x0 = 70, y0 = 60;
    svg.setAttribute("viewBox", "0 0 " + vw + " " + vh);
    function draw() {
      var P = Kit.palette(); svg.innerHTML = "";
      V.forEach(function (v, j) { var t = Kit.el("text", { x: x0 + j * cw + cw / 2, y: 34, "text-anchor": "middle", "font-size": 18, "font-weight": 700, fill: P.coral }, svg); t.textContent = v; var l = Kit.el("text", { x: x0 + j * cw + cw / 2, y: 53, "text-anchor": "middle", "font-size": 12, fill: P.muted }, svg); l.textContent = "v" + (j + 1); });
      U.forEach(function (u, i) { var t = Kit.el("text", { x: 44, y: y0 + i * ch + ch / 2 + 6, "text-anchor": "end", "font-size": 18, "font-weight": 700, fill: P.navy }, svg); t.textContent = u; var l = Kit.el("text", { x: 54, y: y0 + i * ch + ch / 2 + 6, "font-size": 12, fill: P.muted }, svg); l.textContent = "u" + (i + 1); });
      var allZeroRows = 0;
      U.forEach(function (u, i) {
        if (u === 0) allZeroRows++;
        V.forEach(function (v, j) {
          var o = u * v, val = mode ? W0[i][j] + o : o, x = x0 + j * cw, y = y0 + i * ch, mag = Math.min(1, Math.abs(o) / 9);
          var r = Kit.el("rect", { x: x + 2, y: y + 2, width: cw - 4, height: ch - 4, rx: 6, fill: o === 0 ? P.paper : o > 0 ? P.teal : P.coral, "fill-opacity": o === 0 ? 1 : 0.15 + 0.7 * mag, stroke: sel[0] === i && sel[1] === j ? P.ink : P.rule, "stroke-width": sel[0] === i && sel[1] === j ? 3 : 1, style: "cursor:pointer" }, svg);
          r.addEventListener("click", function () { sel = [i, j]; draw(); });
          var t = Kit.el("text", { x: x + cw / 2, y: y + ch / 2 + (mode ? 2 : 7), "text-anchor": "middle", "font-size": 20, "font-weight": 700, fill: P.ink, "pointer-events": "none" }, svg); t.textContent = val;
          if (mode) { var s = Kit.el("text", { x: x + cw / 2, y: y + ch / 2 + 18, "text-anchor": "middle", "font-size": 12, fill: P.ink2, "pointer-events": "none" }, svg); s.textContent = "was " + W0[i][j] + (o ? ", " + (o > 0 ? "+" : "−") + Math.abs(o) : ""); }
        });
      });
      var i = sel[0], j = sel[1], o = U[i] * V[j];
      eq.innerHTML = box(U[i], "u" + (i + 1), "navy") + op("×") + box(V[j], "v" + (j + 1), "coral") + op("=") + box(o, "entry (" + (i + 1) + ", " + (j + 1) + ")", "teal") + (mode ? op("so") + box(W0[i][j], "old weight") + op("+") + box(o, "change") + op("=") + box(W0[i][j] + o, "new weight", "teal") : "");
      st.textContent = (allZeroRows ? "A zero in u gives a whole row of zeros. " : "") + "All rows are multiples of the row v. The table has 12 entries and is set by 7 numbers. Click a cell to see its product.";
      svg.setAttribute("aria-label", "A 3 by 4 table of the products of u and v. Rows are multiples of v.");
    }
    draw(); Kit.onTheme(draw);
  }
})();
