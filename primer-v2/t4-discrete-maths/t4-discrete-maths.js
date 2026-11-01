// Chapter 0.4 Discrete maths and logic: counting explorer, pairs grid, Venn diagram with draggable circles,
// truth-table builder with a separating-line game, and a graph you can build and walk.
(function () {
  "use strict";
  var $ = Kit.$, fmt = Kit.fmt;

  function box(v, label, cls) { return '<span class="t' + (cls ? " c-" + cls : "") + '"><b>' + v + "</b><i>" + (label || "") + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function commas(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function setChip(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function fact(n) { var r = 1; for (var i = 2; i <= n; i++) r *= i; return r; }
  function svgPoint(svg, evt) {
    var pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
    var m = svg.getScreenCTM(); return m ? pt.matrixTransform(m.inverse()) : { x: 0, y: 0 };
  }

  document.addEventListener("DOMContentLoaded", function () {
    counting(); pairs(); venn(); logic(); graphs();
  });

  // =====================================================================
  // 1. Counting explorer
  // =====================================================================
  function counting() {
    var ITEMS = ["L", "W", "G", "D", "T", "M"], NAMES = ["length", "weight", "girth", "depth", "temperature", "month"];
    var mode = 0, LIMIT = 60, chosen = -1;
    var nSl = Kit.slider($("ct-controls"), { label: "Items to choose from, n", min: 1, max: 6, step: 1, value: 4, digits: 0 }, function () { fixK(); chosen = -1; draw(); });
    var kSl = Kit.slider($("ct-controls"), { label: "Items picked, k", min: 0, max: 4, step: 1, value: 2, digits: 0 }, function () { chosen = -1; draw(); });
    Kit.chips($("ct-bar"), ["order matters (arrangements)", "order does not matter (groups)"], 0, function (i) { mode = i; chosen = -1; draw(); });
    var expand = Kit.html("div", { class: "expand", "aria-live": "polite" }, $("ct-list").parentNode);
    $("ct-list").parentNode.insertBefore(expand, $("ct-status"));
    expand.style.display = "none";
    function fixK() { kSl.input.max = nSl.get(); if (kSl.get() > nSl.get()) kSl.set(nSl.get()); else kSl.set(kSl.get()); }

    function item(i) { var s = document.createElement("i"); s.className = "it c" + (i % 5); s.textContent = ITEMS[i]; s.title = NAMES[i]; return s; }
    function perms(n, k, limit, pool) {
      var out = [], used = [], cur = [];
      (function rec() {
        if (out.length >= limit) return;
        if (cur.length === k) { out.push(cur.slice()); return; }
        for (var i = 0; i < pool.length; i++) {
          if (used[i]) continue;
          used[i] = true; cur.push(pool[i]); rec(); cur.pop(); used[i] = false;
          if (out.length >= limit) return;
        }
      })();
      return out;
    }
    function combs(n, k, limit) {
      var out = [], cur = [];
      (function rec(start) {
        if (out.length >= limit) return;
        if (cur.length === k) { out.push(cur.slice()); return; }
        for (var i = start; i < n; i++) { cur.push(i); rec(i + 1); cur.pop(); if (out.length >= limit) return; }
      })(0);
      return out;
    }
    function pill(arr, tag, onClick) {
      var el = document.createElement(tag);
      el.className = "arr";
      if (onClick) { el.type = "button"; el.addEventListener("click", onClick); }
      if (!arr.length) el.textContent = "empty";
      arr.forEach(function (i) { el.appendChild(item(i)); });
      el.setAttribute("aria-label", arr.map(function (i) { return NAMES[i]; }).join(", ") || "empty group");
      return el;
    }
    function draw() {
      var n = nSl.get(), k = kSl.get();
      var P = fact(n) / fact(n - k), C = P / fact(k);
      var pool = $("ct-pool"); pool.innerHTML = "";
      for (var i = 0; i < n; i++) pool.appendChild(item(i));
      pool.setAttribute("aria-label", "The " + n + " items you choose from: " + NAMES.slice(0, n).join(", "));
      var eq = $("ct-eq");
      if (mode === 0) eq.innerHTML = box(commas(fact(n)), n + " factorial") + op("÷") + box(commas(fact(n - k)), "(n − k) factorial = " + (n - k) + "!") + op("=") + box(commas(P), "arrangements P(n,k)", "coral");
      else eq.innerHTML = box(commas(P), "arrangements P(n,k)") + op("÷") + box(commas(fact(k)), "orders of a group, k! = " + k + "!") + op("=") + box(commas(C), "groups C(n,k)", "teal");
      var list = $("ct-list"); list.innerHTML = "";
      var shown = mode === 0 ? perms(n, k, LIMIT, Array.apply(null, { length: n }).map(function (_, j) { return j; })) : combs(n, k, LIMIT);
      var total = mode === 0 ? P : C;
      shown.forEach(function (arr, idx) {
        if (mode === 0) list.appendChild(pill(arr, "span"));
        else { var b = pill(arr, "button", function () { chosen = idx; showOrders(arr); draw2(); }); if (idx === chosen) b.classList.add("on"); list.appendChild(b); }
      });
      if (total > shown.length) { var more = document.createElement("span"); more.className = "arr-more"; more.textContent = "and " + commas(total - shown.length) + " more"; list.appendChild(more); }
      if (mode === 0 || chosen < 0) { expand.style.display = "none"; }
      var st = $("ct-status");
      if (mode === 0) st.textContent = commas(P) + " arrangements of " + k + (k === 1 ? " item" : " items") + " taken from " + n + (k >= 2 ? ". Order matters, so (" + ITEMS[0] + ", " + ITEMS[1] + ") and (" + ITEMS[1] + ", " + ITEMS[0] + ") count as two." : ".");
      else st.textContent = commas(C) + " groups of " + k + " from " + n + ". Click a group to see its " + commas(fact(k)) + " orders. Groups of any size from " + n + " items: 2^" + n + " = " + commas(Math.pow(2, n)) + ".";
    }
    function draw2() {
      // refresh highlight only
      var bs = $("ct-list").querySelectorAll("button.arr");
      bs.forEach(function (b, i) { b.classList.toggle("on", i === chosen); });
    }
    function showOrders(arr) {
      var orders = perms(arr.length, arr.length, 24, arr);
      expand.style.display = "";
      expand.innerHTML = "";
      var t = document.createElement("p"); t.style.margin = "0 0 6px";
      t.textContent = "This one group appears " + commas(fact(arr.length)) + (fact(arr.length) === 1 ? " time" : " times") + " among the arrangements" + (fact(arr.length) > 24 ? " (first 24 shown):" : ":");
      expand.appendChild(t);
      var l = document.createElement("div"); l.className = "arr-list"; expand.appendChild(l);
      orders.forEach(function (o) { l.appendChild(pill(o, "span")); });
    }
    fixK(); draw();
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 1b. Pairs grid
  // =====================================================================
  function pairs() {
    var mode = 0, svg = d3.select("#pr-svg");
    var sl = Kit.slider($("pr-controls"), { label: "Number of items n", min: 2, max: 30, step: 1, value: 12, digits: 0 }, draw);
    Kit.chips($("pr-bar"), ["all ordered pairs", "pairs counted once", "looking back only"], 0, function (i) { mode = i; draw(); });
    function count(n) { return mode === 0 ? n * n : mode === 1 ? n * (n - 1) / 2 : n * (n + 1) / 2; }
    function draw() {
      var n = sl.get(), P = Kit.palette(), S = 360, pad = 22, cell = (S - pad) / n;
      svg.attr("viewBox", "0 0 " + S + " " + S).style("max-width", "420px").style("margin", "0 auto").selectAll("*").remove();
      for (var i = 0; i < n; i++) for (var j = 0; j < n; j++) {
        var on = mode === 0 || (mode === 1 ? i > j : i >= j), diag = i === j;
        svg.append("rect").attr("x", pad + j * cell).attr("y", pad + i * cell).attr("width", cell - 1).attr("height", cell - 1).attr("rx", Math.min(3, cell / 5))
          .attr("fill", on ? (diag && mode !== 0 ? P.sand : P.teal) : P.rule).attr("opacity", on ? 0.9 : 0.55);
      }
      svg.append("text").attr("x", pad + (S - pad) / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", P.muted).text("item j (second)");
      svg.append("text").attr("transform", "translate(14," + (pad + (S - pad) / 2) + ") rotate(-90)").attr("text-anchor", "middle").attr("font-size", 13).attr("fill", P.muted).text("item i (first)");
      var c = count(n), c2 = count(2 * n);
      var eq = $("pr-eq");
      eq.innerHTML = (mode === 0 ? box(n, "n") + op("×") + box(n, "n") : mode === 1 ? box(n, "n") + op("×") + box(n - 1, "n − 1") + op("÷") + box(2, "two orders") : box(n, "n") + op("×") + box(n + 1, "n + 1") + op("÷") + box(2, "half the grid"))
        + op("=") + box(commas(c), "cells counted", "teal") + op("&nbsp;&nbsp;&nbsp;") + box(commas(c2), "if n doubles to " + (2 * n), "coral") + op("=") + box(fmt(c2 / c, 2) + " ×", "times as many", "coral");
      $("pr-svg").setAttribute("aria-label", "A " + n + " by " + n + " grid of cells, one per pair of items. " + commas(c) + " cells are counted. Doubling n would give " + commas(c2) + ".");
    }
    draw();
    Kit.onTheme(draw);
    window.addEventListener("resize", Kit.debounce(draw));
  }

  // =====================================================================
  // 2. Venn diagram with draggable circles
  // =====================================================================
  function venn() {
    var W = 640, H = 330, svg = $("vn-svg"), r = Kit.rng(5), dots = [];
    var A = { x: 250, y: 165, r: 100 }, B = { x: 390, y: 165, r: 100 }, region = 0, words = 0;
    // Place 30 dots so that the starting circles show all four regions: 5 in both, 9 in A only, 7 in B only, 9 in neither.
    [["both", 5], ["a", 9], ["b", 7], ["n", 9]].forEach(function (q) {
      var k = 0, tries = 0;
      while (k < q[1] && tries < 9000) {
        tries++;
        var p = { x: 22 + r() * 596, y: 22 + r() * 286 }, da = Math.hypot(p.x - A.x, p.y - A.y), db = Math.hypot(p.x - B.x, p.y - B.y);
        if (Math.abs(da - A.r) < 10 || Math.abs(db - B.r) < 10) continue;
        var key = da < A.r && db < B.r ? "both" : da < A.r ? "a" : db < B.r ? "b" : "n";
        if (key !== q[0]) continue;
        if (dots.some(function (d) { return Math.hypot(d.x - p.x, d.y - p.y) < 30; })) continue;
        dots.push(p); k++;
      }
    });
    var REG = ["all dots", "A ∩ B (both)", "A ∪ B (either)", "A only", "B only", "neither"];
    var sA = Kit.slider($("vn-controls"), { label: "Size of circle A", min: 45, max: 135, step: 1, value: A.r, digits: 0 }, function (v) { A.r = v; update(); });
    var sB = Kit.slider($("vn-controls"), { label: "Size of circle B", min: 45, max: 135, step: 1, value: B.r, digits: 0 }, function (v) { B.r = v; update(); });
    Kit.chips($("vn-bar"), REG, 0, function (i) { region = i; update(); });
    Kit.chips($("vn-bar2"), ["plain set names", "classifier words"], 0, function (i) { words = i; update(); });

    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var NS = "http://www.w3.org/2000/svg";
    var uni = Kit.el("rect", { x: 4, y: 4, width: W - 8, height: H - 8, rx: 8, fill: "none", "stroke-width": 1.5 }, svg);
    var gDots = Kit.el("g", {}, svg), cA = Kit.el("circle", { class: "drag", tabindex: 0, role: "slider", "aria-label": "Circle A, drag or use arrow keys" }, svg), cB = Kit.el("circle", { class: "drag", tabindex: 0, role: "slider", "aria-label": "Circle B, drag or use arrow keys" }, svg);
    var tA = Kit.el("text", { "text-anchor": "middle", "font-size": 16, "font-weight": 600, "pointer-events": "none" }, svg), tB = Kit.el("text", { "text-anchor": "middle", "font-size": 16, "font-weight": 600, "pointer-events": "none" }, svg);
    var tU = Kit.el("text", { x: 14, y: 22, "font-size": 13 }, svg);
    var dEls = dots.map(function (d) { return Kit.el("circle", { r: 7, cx: d.x, cy: d.y, "pointer-events": "none" }, gDots); });
    [[cA, A], [cB, B]].forEach(function (pr) { drag(pr[0], pr[1]); });

    function drag(el, C) {
      var off = null;
      el.addEventListener("pointerdown", function (e) { var p = svgPoint(svg, e); off = { x: C.x - p.x, y: C.y - p.y }; el.setPointerCapture(e.pointerId); e.preventDefault(); });
      el.addEventListener("pointermove", function (e) { if (!off) return; var p = svgPoint(svg, e); C.x = Math.max(10, Math.min(W - 10, p.x + off.x)); C.y = Math.max(10, Math.min(H - 10, p.y + off.y)); update(); });
      el.addEventListener("pointerup", function () { off = null; });
      el.addEventListener("pointercancel", function () { off = null; });
      el.addEventListener("keydown", function (e) {
        var s = 10, m = { ArrowLeft: [-s, 0], ArrowRight: [s, 0], ArrowUp: [0, -s], ArrowDown: [0, s] }[e.key];
        if (m) { C.x = Math.max(10, Math.min(W - 10, C.x + m[0])); C.y = Math.max(10, Math.min(H - 10, C.y + m[1])); update(); e.preventDefault(); }
      });
    }
    function inC(d, C) { var dx = d.x - C.x, dy = d.y - C.y; return dx * dx + dy * dy <= C.r * C.r; }
    function update() {
      var P = Kit.palette(), cnt = { both: 0, aOnly: 0, bOnly: 0, none: 0 }, hit = [];
      dots.forEach(function (d, i) {
        var a = inC(d, A), b = inC(d, B), key = a && b ? "both" : a ? "aOnly" : b ? "bOnly" : "none";
        cnt[key]++;
        var inReg = [true, a && b, a || b, a && !b, b && !a, !a && !b][region];
        var col = key === "both" ? P.teal : key === "aOnly" ? P.navy : key === "bOnly" ? P.coral : P.muted;
        var el = dEls[i];
        el.setAttribute("fill", col); el.setAttribute("opacity", inReg ? 1 : 0.25);
        el.setAttribute("stroke", region && inReg ? P.ink : "none"); el.setAttribute("stroke-width", region && inReg ? 2.5 : 0);
        el.setAttribute("r", region && inReg ? 8 : 7);
        if (inReg) hit.push(i);
      });
      uni.setAttribute("stroke", P.rule);
      [[cA, A, P.navy], [cB, B, P.coral]].forEach(function (t) { t[0].setAttribute("cx", t[1].x); t[0].setAttribute("cy", t[1].y); t[0].setAttribute("r", t[1].r); t[0].setAttribute("fill", t[2]); t[0].setAttribute("fill-opacity", 0.1); t[0].setAttribute("stroke", t[2]); t[0].setAttribute("stroke-width", 2.5); });
      var la = words ? "A: flagged" : "A", lb = words ? "B: Ganges shark" : "B";
      tA.setAttribute("x", A.x); tA.setAttribute("y", A.y - A.r - 8 < 20 ? A.y + A.r + 20 : A.y - A.r - 8); tA.setAttribute("fill", P.navy); tA.textContent = la;
      tB.setAttribute("x", B.x); tB.setAttribute("y", B.y - B.r - 8 < 20 ? B.y + B.r + 20 : B.y - B.r - 8); tB.setAttribute("fill", P.coral); tB.textContent = lb;
      tU.setAttribute("fill", P.muted); tU.textContent = "all 30 sharks (made-up)";
      var a = cnt.both + cnt.aOnly, b = cnt.both + cnt.bOnly, u = cnt.both + cnt.aOnly + cnt.bOnly;
      $("vn-eq").innerHTML = box(a, "|A|, in circle A", "navy") + op("+") + box(b, "|B|, in circle B", "coral") + op("−") + box(cnt.both, "|A ∩ B|, in both", "teal") + op("=") + box(u, "|A ∪ B|, in either", "sand") + op("&nbsp;&nbsp;") + box(30 - u, "in neither: 30 − |A ∪ B|");
      var names = words ? ["all 30 sharks", "flagged and a Ganges shark (true positives)", "flagged or a Ganges shark", "flagged but not a Ganges shark (false positives)", "Ganges sharks that were not flagged (false negatives)", "not flagged and not a Ganges shark (true negatives)"]
                        : ["all 30 sharks", "in both A and B", "in A or B or both", "in A but not B", "in B but not A", "in neither A nor B"];
      var inCount = [30, cnt.both, u, cnt.aOnly, cnt.bOnly, cnt.none][region];
      $("vn-status").textContent = names[region] + ": " + inCount + (region === 0 ? "." : " of 30.") + (words ? " (The names come back in chapter A4.)" : "");
      svg.setAttribute("aria-label", "Thirty dots for sharks with two circles. " + a + " in A, " + b + " in B, " + cnt.both + " in both, " + cnt.none + " in neither.");
    }
    update();
    Kit.onTheme(update);
  }

  // =====================================================================
  // 3. Truth-table builder
  // =====================================================================
  var NAMES16 = {
    "0000": "always 0", "0001": "A AND B", "0010": "A AND NOT B", "0011": "A (B is ignored)", "0100": "NOT A AND B", "0101": "B (A is ignored)",
    "0110": "A XOR B", "0111": "A OR B", "1000": "A NOR B, which is NOT (A OR B)", "1001": "A XNOR B, the inputs match", "1010": "NOT B (A is ignored)",
    "1011": "B IMPLIES A", "1100": "NOT A (B is ignored)", "1101": "A IMPLIES B", "1110": "A NAND B, which is NOT (A AND B)", "1111": "always 1"
  };
  var PT = [[0, 0], [0, 1], [1, 0], [1, 1]];
  // score of a line (w1, w2, b) on a table: number of points with (w.p + b > 0) equal to the output
  function scoreLine(out, w1, w2, b) { var s = 0; for (var i = 0; i < 4; i++) s += ((w1 * PT[i][0] + w2 * PT[i][1] + b > 0) === !!out[i]) ? 1 : 0; return s; }
  function bestScore(out) {
    var best = 0, ws = [-1, 0, 1], bs = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5];
    ws.forEach(function (w1) { ws.forEach(function (w2) { bs.forEach(function (b) { best = Math.max(best, scoreLine(out, w1, w2, b)); }); }); });
    return best;
  }

  function logic() {
    var out = [0, 0, 0, 1], sel = 3, flip = false, lo = -0.6, hi = 1.6;
    var H1 = { x: 0.2, y: 1.7 }, H2 = { x: 1.7, y: 0.2 };
    var PRE = [["AND", [0, 0, 0, 1]], ["OR", [0, 1, 1, 1]], ["NOT A", [1, 1, 0, 0]], ["XOR", [0, 1, 1, 0]]];
    var chips = Kit.chips($("lg-bar"), PRE.map(function (p) { return p[0]; }), 0, function (i) { out = PRE[i][1].slice(); draw(); });
    Kit.button($("lg-bar2"), "flip side", function () { flip = !flip; draw(); });
    Kit.button($("lg-bar2"), "search for the best line", search);
    Kit.button($("lg-bar2"), "reset line", function () { H1 = { x: 0.2, y: 1.7 }; H2 = { x: 1.7, y: 0.2 }; flip = false; draw(); });

    var svg = $("lg-svg"), SW = 360, SH = 340, NS = "http://www.w3.org/2000/svg";
    svg.setAttribute("viewBox", "0 0 " + SW + " " + SH);
    var px = function (a) { return 34 + (a - lo) / (hi - lo) * 300; }, py = function (b) { return 306 - (b - lo) / (hi - lo) * 276; };
    var defs = Kit.el("defs", {}, svg), clip = Kit.el("clipPath", { id: "lg-clip" }, defs); Kit.el("rect", { x: 34, y: 30, width: 300, height: 276 }, clip);
    var frame = Kit.el("rect", { x: 34, y: 30, width: 300, height: 276, fill: "none", "stroke-width": 1.2 }, svg);
    var shade = Kit.el("polygon", { "clip-path": "url(#lg-clip)" }, svg), line = Kit.el("line", { "clip-path": "url(#lg-clip)", "stroke-width": 3 }, svg);
    var axes = Kit.el("g", {}, svg);
    var pts = PT.map(function (p, i) {
      var g = Kit.el("g", { style: "cursor:pointer" }, svg);
      var c = Kit.el("circle", { cx: px(p[0]), cy: py(p[1]), r: 17, "stroke-width": 3 }, g), t = Kit.el("text", { x: px(p[0]), y: py(p[1]) + 6, "text-anchor": "middle", "font-size": 17, "font-weight": 700 }, g);
      var m = Kit.el("text", { x: px(p[0]) + 24, y: py(p[1]) - 16, "font-size": 15, "font-weight": 700 }, g);
      g.addEventListener("click", function () { out[i] = 1 - out[i]; sel = i; draw(); });
      return { g: g, c: c, t: t, m: m };
    });
    var h1 = Kit.el("circle", { class: "drag", r: 10, tabindex: 0, role: "slider", "aria-label": "Line handle one" }, svg), h2 = Kit.el("circle", { class: "drag", r: 10, tabindex: 0, role: "slider", "aria-label": "Line handle two" }, svg);
    function dragH(el, H) {
      var on = false;
      el.addEventListener("pointerdown", function (e) { on = true; el.setPointerCapture(e.pointerId); e.preventDefault(); });
      el.addEventListener("pointermove", function (e) { if (!on) return; var p = svgPoint(svg, e); H.x = Math.max(lo, Math.min(hi, lo + (p.x - 34) / 300 * (hi - lo))); H.y = Math.max(lo, Math.min(hi, lo + (306 - p.y) / 276 * (hi - lo))); draw(); });
      el.addEventListener("pointerup", function () { on = false; });
      el.addEventListener("pointercancel", function () { on = false; });
      el.addEventListener("keydown", function (e) {
        var s = 0.05, m = { ArrowLeft: [-s, 0], ArrowRight: [s, 0], ArrowUp: [0, s], ArrowDown: [0, -s] }[e.key];
        if (m) { H.x = Math.max(lo, Math.min(hi, H.x + m[0])); H.y = Math.max(lo, Math.min(hi, H.y + m[1])); draw(); e.preventDefault(); }
      });
    }
    dragH(h1, H1); dragH(h2, H2);

    function lineFn() {
      var dx = H2.x - H1.x, dy = H2.y - H1.y, len = Math.hypot(dx, dy) || 1e-9, nx = -dy / len, ny = dx / len, s = flip ? -1 : 1;
      return { nx: nx * s, ny: ny * s, dx: dx / len, dy: dy / len };
    }
    function side(L, p) { return (p[0] - H1.x) * L.nx + (p[1] - H1.y) * L.ny > 0 ? 1 : 0; }
    function draw() {
      var P = Kit.palette(), L = lineFn(), key = out.join(""), score = 0, known = -1;
      PRE.forEach(function (p, i) { if (p[1].join("") === key) known = i; });
      setChip(chips, known);
      // axes
      axes.innerHTML = "";
      [0, 1].forEach(function (v) {
        Kit.el("text", { x: px(v), y: 322, "text-anchor": "middle", "font-size": 13, fill: P.muted }, axes).textContent = v;
        Kit.el("text", { x: 26, y: py(v) + 4, "text-anchor": "end", "font-size": 13, fill: P.muted }, axes).textContent = v;
      });
      Kit.el("text", { x: 184, y: 333, "text-anchor": "middle", "font-size": 14, fill: P.ink2 }, axes).textContent = "input A";
      var yl = Kit.el("text", { transform: "translate(15,168) rotate(-90)", "text-anchor": "middle", "font-size": 14, fill: P.ink2 }, axes); yl.textContent = "input B";
      frame.setAttribute("stroke", P.rule);
      // shaded "1 side": polygon in data space, mapped to pixels
      var big = 8, ax = H1.x - L.dx * big, ay = H1.y - L.dy * big, bx = H1.x + L.dx * big, by = H1.y + L.dy * big;
      var poly = [[ax, ay], [bx, by], [bx + L.nx * big, by + L.ny * big], [ax + L.nx * big, ay + L.ny * big]].map(function (q) { return px(q[0]) + "," + py(q[1]); }).join(" ");
      shade.setAttribute("points", poly); shade.setAttribute("fill", P.teal); shade.setAttribute("fill-opacity", 0.14);
      line.setAttribute("x1", px(ax)); line.setAttribute("y1", py(ay)); line.setAttribute("x2", px(bx)); line.setAttribute("y2", py(by)); line.setAttribute("stroke", P.ink);
      h1.setAttribute("cx", px(H1.x)); h1.setAttribute("cy", py(H1.y)); h2.setAttribute("cx", px(H2.x)); h2.setAttribute("cy", py(H2.y));
      [h1, h2].forEach(function (h) { h.setAttribute("fill", P.paper); h.setAttribute("stroke", P.ink); h.setAttribute("stroke-width", 2.5); });
      pts.forEach(function (o, i) {
        var one = out[i] === 1, pred = side(L, PT[i]), ok = (pred === 1) === one; if (ok) score++;
        o.c.setAttribute("fill", one ? P.coral : P.navy); o.c.setAttribute("stroke", ok ? P.teal : P.ink); o.c.setAttribute("stroke-dasharray", ok ? "0" : "4 3");
        o.t.setAttribute("fill", P.bg); o.t.textContent = out[i];
        o.m.setAttribute("fill", ok ? P.teal : P.coral); o.m.textContent = ok ? "✓" : "✗";
      });
      // table
      var tb = $("lg-table");
      tb.innerHTML = "<thead><tr><th>A</th><th>B</th><th>output</th></tr></thead>";
      var body = document.createElement("tbody"); tb.appendChild(body);
      PT.forEach(function (p, i) {
        var tr = document.createElement("tr"); if (i === sel) tr.className = "sel";
        tr.innerHTML = "<td>" + p[0] + "</td><td>" + p[1] + "</td>";
        var td = document.createElement("td"), b = document.createElement("button"); b.type = "button"; b.textContent = out[i]; b.className = out[i] ? "one" : ""; b.setAttribute("aria-label", "Output for A " + p[0] + " and B " + p[1] + " is " + out[i] + ". Press to flip.");
        b.addEventListener("click", function (e) { e.stopPropagation(); out[i] = 1 - out[i]; sel = i; draw(); });
        td.appendChild(b); tr.appendChild(td); tr.addEventListener("click", function () { sel = i; draw(); }); body.appendChild(tr);
      });
      $("lg-name").innerHTML = "This table is <b>" + NAMES16[key] + "</b>.";
      var best = bestScore(out);
      $("lg-status").innerHTML = "<b>" + score + " of 4</b> points are on the right side of your line (ticks mark the right ones). Best possible with one straight line: <b>" + best + " of 4</b>" + (best === 4 ? "." : ". This table cannot be separated by any line.");
      // equation: any table is c0 + c1 A + c2 B + c3 A B
      var c0 = out[0], c1 = out[2] - out[0], c2 = out[1] - out[0], c3 = out[3] - out[2] - out[1] + out[0], a = PT[sel][0], b2 = PT[sel][1];
      $("lg-eq").innerHTML = box(out[sel], "output for (A, B) = (" + a + ", " + b2 + ")", "coral") + op("=") + box(c0, "constant") + op("+") + box(c1, "weight on A") + op("×") + box(a, "A", "navy") + op("+") + box(c2, "weight on B") + op("×") + box(b2, "B", "navy") + op("+") + box(c3, "product term", c3 ? "coral" : "") + op("×") + box(a * b2, "A × B", "navy");
      svg.setAttribute("aria-label", "The four inputs of a truth table as points on a grid. " + score + " of 4 are on the correct side of the line. The best possible is " + best + " of 4.");
    }
    function clipT(p, d) {
      var t0 = -1e9, t1 = 1e9;
      [["x", "dx"], ["y", "dy"]].forEach(function (c) {
        var pv = p[c[0]], dv = d[c[1]];
        if (Math.abs(dv) < 1e-9) return;
        var a = (lo + 0.05 - pv) / dv, b = (hi - 0.05 - pv) / dv;
        t0 = Math.max(t0, Math.min(a, b)); t1 = Math.min(t1, Math.max(a, b));
      });
      return [t0, t1];
    }
    function search() {
      var r = Kit.rng(2024), best = -1, bp = null, bd = null, bflip = false;
      for (var i = 0; i < 20000; i++) {
        var p = { x: -0.3 + 1.6 * r(), y: -0.3 + 1.6 * r() }, th = Math.PI * r(), d = { dx: Math.cos(th), dy: Math.sin(th) };
        var nx = -d.dy, ny = d.dx;
        var pred = PT.map(function (q) { return (q[0] - p.x) * nx + (q[1] - p.y) * ny > 0 ? 1 : 0; });
        var s = 0; for (var k = 0; k < 4; k++) s += pred[k] === out[k] ? 1 : 0;
        var f = false; if (4 - s > s) { s = 4 - s; f = true; }
        if (s > best) { best = s; bp = p; bd = d; bflip = f; if (s === 4) break; }
      }
      var tt = clipT(bp, bd);
      H1 = { x: bp.x + bd.dx * tt[0], y: bp.y + bd.dy * tt[0] }; H2 = { x: bp.x + bd.dx * tt[1], y: bp.y + bd.dy * tt[1] };
      flip = bflip; draw();
      $("lg-status").innerHTML += " The search found a line with <b>" + best + " of 4</b>.";
    }
    draw();
    Kit.onTheme(draw);
  }

  // =====================================================================
  // 4. Graph builder and walker
  // =====================================================================
  function graphs() {
    var svg = $("gr-svg"), W = 640, H = 360, R = 21;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var nodes = [], edges = [], directed = false, mode = 0, sel = null, walk = [], lit = null, litBack = false, timer = 0, drag = null, note = "";
    var MODES = ["Move", "Add node", "Add edge", "Delete", "Walk"];
    Kit.chips($("gr-bar"), MODES, 0, function (i) { mode = i; sel = null; stopOrder(); if (i !== 4) walk = []; render(); });
    var tchips = Kit.chips($("gr-bar"), ["undirected", "directed"], 0, function (i) { directed = i === 1; stopOrder(); render(); });
    Kit.button($("gr-bar2"), "Tree", function () { preset("tree"); });
    Kit.button($("gr-bar2"), "DAG", function () { preset("dag"); });
    Kit.button($("gr-bar2"), "Cycle", function () { preset("cycle"); });
    Kit.button($("gr-bar2"), "Clear", function () { preset("clear"); });
    Kit.button($("gr-bar2"), "Order", function () { runOrder(false); });
    Kit.button($("gr-bar2"), "Backward order", function () { runOrder(true); });

    function addNode(x, y, label) { nodes.push({ label: label || nextLabel(), x: x, y: y }); }
    function nextLabel() { var used = nodes.map(function (n) { return n.label; }); for (var c = 65; c < 91; c++) if (used.indexOf(String.fromCharCode(c)) < 0) return String.fromCharCode(c); return "?"; }
    function hasEdge(a, b) { return edges.some(function (e) { return directed ? (e.a === a && e.b === b) : ((e.a === a && e.b === b) || (e.a === b && e.b === a)); }); }
    function preset(kind) {
      stopOrder(); nodes = []; edges = []; walk = []; sel = null; note = "";
      if (kind === "tree") {
        directed = false;
        [[320, 50, "A"], [180, 140, "B"], [460, 140, "C"], [110, 250, "D"], [250, 250, "E"], [390, 250, "F"], [530, 250, "G"]].forEach(function (p) { addNode(p[0], p[1], p[2]); });
        [[0, 1], [0, 2], [1, 3], [1, 4], [2, 5], [2, 6]].forEach(function (p) { edges.push({ a: p[0], b: p[1] }); });
      } else if (kind === "dag") {
        directed = true;
        [[110, 90, "x"], [110, 230, "w"], [310, 160, "m"], [310, 300, "b"], [520, 230, "z"]].forEach(function (p) { addNode(p[0], p[1], p[2]); });
        [[0, 2], [1, 2], [2, 4], [3, 4]].forEach(function (p) { edges.push({ a: p[0], b: p[1] }); });
        note = "DAG preset: m = w times x, then z = m + b.";
      } else if (kind === "cycle") {
        directed = true;
        [[320, 60, "A"], [470, 260, "B"], [170, 260, "C"]].forEach(function (p) { addNode(p[0], p[1], p[2]); });
        [[0, 1], [1, 2], [2, 0]].forEach(function (p) { edges.push({ a: p[0], b: p[1] }); });
      }
      setChip(tchips, directed ? 1 : 0); render();
    }

    // ---- analysis: degrees, components, cycles, topological order ----
    function analyse() {
      var n = nodes.length, e = edges.length, E = edges.map(function (ed) { return [ed.a, ed.b]; });
      var deg = nodes.map(function () { return 0; }), indeg = deg.slice();
      E.forEach(function (p) { deg[p[0]]++; deg[p[1]]++; indeg[p[1]]++; });
      function reach(from, to, skip, dir) {
        var seen = {}, st = [from]; seen[from] = 1;
        while (st.length) {
          var u = st.pop(); if (u === to) return true;
          E.forEach(function (p, k) {
            if (k === skip) return;
            if (p[0] === u && !seen[p[1]]) { seen[p[1]] = 1; st.push(p[1]); }
            if (!dir && p[1] === u && !seen[p[0]]) { seen[p[0]] = 1; st.push(p[0]); }
          });
        }
        return false;
      }
      var cyc = {}, ncyc = 0;
      E.forEach(function (p, k) { if (directed ? reach(p[1], p[0], -1, true) : reach(p[1], p[0], k, false)) { cyc[k] = 1; ncyc++; } });
      var comp = nodes.map(function (_, i) { return i; });
      function find(x) { while (comp[x] !== x) { comp[x] = comp[comp[x]]; x = comp[x]; } return x; }
      var undirCycle = false;
      E.forEach(function (p) { var a = find(p[0]), b = find(p[1]); if (a === b) undirCycle = true; else comp[a] = b; });
      var roots = {}; nodes.forEach(function (_, i) { roots[find(i)] = 1; });
      var comps = Object.keys(roots).length, order = null;
      if (directed && ncyc === 0 && n > 0) {
        var ind = indeg.slice(), ready = [], res = [];
        ind.forEach(function (v, i) { if (v === 0) ready.push(i); });
        while (ready.length) { ready.sort(function (a, b) { return a - b; }); var u = ready.shift(); res.push(u); E.forEach(function (p) { if (p[0] === u && --ind[p[1]] === 0) ready.push(p[1]); }); }
        if (res.length === n) order = res;
      }
      return { n: n, e: e, deg: deg, indeg: indeg, comps: comps, undirCycle: undirCycle, isTree: n > 0 && comps === 1 && !undirCycle, dirCycle: directed && ncyc > 0, order: order, cyc: cyc };
    }

    // ---- drawing ----
    function render() {
      var P = Kit.palette(), A = analyse();
      svg.innerHTML = "";
      var defs = Kit.el("defs", {}, svg);
      [["m1", P.muted], ["m2", P.coral]].forEach(function (m) {
        var mk = Kit.el("marker", { id: m[0], viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 8, markerHeight: 8, orient: "auto" }, defs);
        Kit.el("path", { d: "M0,1 L9,5 L0,9 z", fill: m[1] }, mk);
      });
      var bg = Kit.el("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }, svg);
      bg.addEventListener("click", function (e) {
        if (mode !== 1) return;
        if (nodes.length >= 12) { setStatus("The canvas allows 12 nodes at most. Delete one first."); return; }
        var p = svgPoint(svg, e); note = ""; addNode(Math.max(R, Math.min(W - R, p.x)), Math.max(R, Math.min(H - R, p.y))); stopOrder(); render();
      });
      var eg = Kit.el("g", {}, svg), ng = Kit.el("g", {}, svg);
      edges.forEach(function (ed, k) {
        var a = nodes[ed.a], b = nodes[ed.b], dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1, ux = dx / len, uy = dy / len;
        var rev = directed && edges.some(function (o) { return o.a === ed.b && o.b === ed.a; }), off = rev ? 16 : 0, nx = -uy * off, ny = ux * off;
        var gap = R + (directed ? 3 : 0);
        var x1 = a.x + ux * R + nx * 0.5, y1 = a.y + uy * R + ny * 0.5, x2 = b.x - ux * gap + nx * 0.5, y2 = b.y - uy * gap + ny * 0.5;
        var d = "M" + x1 + "," + y1 + " Q" + ((a.x + b.x) / 2 + nx) + "," + ((a.y + b.y) / 2 + ny) + " " + x2 + "," + y2;
        var hot = !!A.cyc[k];
        Kit.el("path", { d: d, fill: "none", stroke: hot ? P.coral : P.muted, "stroke-width": hot ? 3.5 : 2.5, "marker-end": directed ? "url(#" + (hot ? "m2" : "m1") + ")" : "none" }, eg);
        var hit = Kit.el("path", { d: d, fill: "none", stroke: "transparent", "stroke-width": 16, style: mode === 3 ? "cursor:pointer" : "" }, eg);
        hit.addEventListener("click", function (e) { e.stopPropagation(); if (mode === 3) { edges.splice(k, 1); note = ""; stopOrder(); render(); } });
      });
      nodes.forEach(function (nd, i) {
        var g = Kit.el("g", { style: "cursor:" + (mode === 0 ? "grab" : "pointer"), "data-i": i, role: "button", "aria-label": "Node " + nd.label }, ng);
        var cur = walk.length && walk[walk.length - 1] === i, vis = walk.indexOf(i) >= 0, pos = lit && lit[i];
        Kit.el("circle", { cx: nd.x, cy: nd.y, r: R, fill: cur ? P.teal : pos ? P.sand : P.paper, stroke: sel === i ? P.coral : vis ? P.teal : P.navy, "stroke-width": sel === i || vis ? 4 : 2.5, style: mode === 0 ? "touch-action:none" : "" }, g);
        var t = Kit.el("text", { x: nd.x, y: nd.y + 5.5, "text-anchor": "middle", "font-size": 17, "font-weight": 600, fill: cur ? P.bg : P.ink }, g); t.textContent = nd.label;
        var dg = Kit.el("text", { x: nd.x, y: nd.y + R + 15, "text-anchor": "middle", "font-size": 12, fill: P.muted, stroke: P.paper, "stroke-width": 4, "paint-order": "stroke" }, g);
        dg.textContent = directed ? "in " + A.indeg[i] + ", out " + (A.deg[i] - A.indeg[i]) : "degree " + A.deg[i];
        if (pos) { var bd = Kit.el("text", { x: nd.x + R - 2, y: nd.y - R + 2, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: P.coral }, g); bd.textContent = pos; }
        g.addEventListener("pointerdown", function (e) { if (mode !== 0) return; drag = i; svg.setPointerCapture(e.pointerId); e.preventDefault(); });
        g.addEventListener("click", function (e) { e.stopPropagation(); onNode(i); });
      });
      var n = A.n, e = A.e, sumdeg = A.deg.reduce(function (a, b) { return a + b; }, 0);
      $("gr-eq").innerHTML = box(n, "nodes n") + op("&nbsp;") + box(e, "edges e") + op("&nbsp;") + box(Math.max(0, n - 1), "n − 1, the edges of a tree", A.isTree ? "teal" : "") + op("&nbsp;") + box(sumdeg, "sum of degrees") + op("=") + box("2 × " + e, "twice the edges");
      var s = "";
      if (n === 0) s = "Pick \"Add node\" and click the canvas to place nodes, or load a preset.";
      else if (directed) s = A.dirCycle ? "A cycle follows the arrows (coral), so there is no forward order. This is not a DAG." : "A DAG: no cycle follows the arrows" + (A.isTree ? ", and without the arrows it is a tree" : "") + ". Press Order for a forward order.";
      else s = A.isTree ? "A tree: connected, with no cycle, and " + e + " = " + n + " − 1 edges." : A.undirCycle ? "There is a cycle (coral)." + (A.comps > 1 ? " There are also " + A.comps + " separate pieces." : "") : "No cycle, but " + A.comps + " separate pieces, so this is a forest and not a tree.";
      if (walk.length) s += " Walk: " + walk.map(function (i) { return nodes[i].label; }).join(" → ") + (mode === 4 ? ". Click a neighbour to step on." : ".");
      else if (mode === 4 && n) s += " Click a node to start a walk.";
      if (mode === 2 && sel !== null) s += " Click a second node to join it to " + nodes[sel].label + ".";
      if (lit) { s = (litBack ? "Backward order so far: " : "Forward order so far: ") + Object.keys(lit).sort(function (x, y) { return lit[x] - lit[y]; }).map(function (i) { return nodes[+i].label; }).join(" → ") + "."; }
      setStatus((note && !lit ? note + " " : "") + s);
      svg.setAttribute("aria-label", "A graph with " + n + " nodes and " + e + " edges. " + (directed ? (A.dirCycle ? "It has a directed cycle." : "It is a DAG.") : (A.isTree ? "It is a tree." : A.undirCycle ? "It has a cycle." : "It is a forest.")));
    }
    function setStatus(t) { $("gr-status").textContent = t; }
    function onNode(i) {
      if (mode === 2) {
        if (sel === null) sel = i;
        else if (sel === i) sel = null;
        else if (hasEdge(sel, i)) { sel = null; setStatus("That edge exists already."); return; }
        else { edges.push({ a: sel, b: i }); sel = null; note = ""; stopOrder(); }
        render();
      } else if (mode === 3) {
        nodes.splice(i, 1); note = "";
        edges = edges.filter(function (e) { return e.a !== i && e.b !== i; }).map(function (e) { return { a: e.a > i ? e.a - 1 : e.a, b: e.b > i ? e.b - 1 : e.b }; });
        stopOrder(); render();
      } else if (mode === 4) {
        if (!walk.length) walk = [i];
        else {
          var cur = walk[walk.length - 1];
          var ok = edges.some(function (e) { return directed ? (e.a === cur && e.b === i) : ((e.a === cur && e.b === i) || (e.a === i && e.b === cur)); });
          if (ok) walk.push(i);
          else if (i === cur) walk = [i];
          else { setStatus("There is no " + (directed ? "arrow from " : "edge between ") + nodes[cur].label + (directed ? " to " : " and ") + nodes[i].label + ". Walk: " + walk.map(function (k) { return nodes[k].label; }).join(" → ") + "."); return; }
        }
        render();
      }
    }
    svg.addEventListener("pointermove", function (e) { if (drag === null) return; var p = svgPoint(svg, e); nodes[drag].x = Math.max(R, Math.min(W - R, p.x)); nodes[drag].y = Math.max(R, Math.min(H - R, p.y)); render(); });
    svg.addEventListener("pointerup", function () { drag = null; });
    svg.addEventListener("pointercancel", function () { drag = null; });

    function stopOrder() { if (timer) clearInterval(timer); timer = 0; lit = null; }
    function runOrder(back) {
      stopOrder();
      var A = analyse();
      if (!directed) { setStatus("Switch to directed edges and build a DAG first, or load the DAG preset."); return; }
      if (!A.order) { setStatus(A.n ? "There is no forward order because the arrows contain a cycle." : "Add some nodes first."); return; }
      var seq = back ? A.order.slice().reverse() : A.order.slice(), k = 0;
      walk = []; litBack = back;
      function step() {
        k++; lit = {}; seq.slice(0, k).forEach(function (i, p) { lit[i] = p + 1; });
        render();
        if (k >= seq.length) { clearInterval(timer); timer = 0; }
      }
      step();
      if (seq.length > 1) timer = setInterval(step, 700);
    }
    preset("tree");
    Kit.onTheme(render);
  }
})();
