// A1 What a model is: four figures.
//   1 dialFig    a one-dial model, weight = a * length^3, against the lazy average (Plotly)
//   2 columnFig  choose the question and see the label, the inputs and the task type (HTML table)
//   3 lazyFig    "beat the lazy model": cut-offs on one input, a species for each part (D3)
//   4 searchFig  trying dial settings at random, and how fast the number of settings grows (Plotly)
(function () {
  "use strict";
  var SPECIES = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"];
  var NAME = { spadenose: "spadenose", milk: "milk shark", bamboo: "bamboo shark", hammerhead: "hammerhead", ganges: "Ganges shark" };
  var REC = null;

  document.addEventListener("DOMContentLoaded", function () {
    Kit.data("../data/sharks/catch.json").then(function (J) {
      REC = J.records;
      dialFig(); columnFig(); lazyFig(); searchFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- helpers ----------
  function mean(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s / a.length; }
  function spCol(P) { return { spadenose: P.navy, milk: P.coral, bamboo: P.teal, hammerhead: P.sand, ganges: P.ink2 }; }
  function box(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function op(s) { return '<span class="op">' + s + '</span>'; }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { var C = hexRGB(c); return "rgba(" + C[0] + "," + C[1] + "," + C[2] + "," + a + ")"; }
  function thousands(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }

  // ---------- 1. a model with one dial ----------
  function dialFig() {
    var sp = REC.filter(function (r) { return r.species === "spadenose"; });
    var L = sp.map(function (r) { return r.length_cm; }), W = sp.map(function (r) { return r.weight_kg; });
    var avgW = mean(W), lazyMiss = mean(W.map(function (w) { return Math.abs(w - avgW); }));
    var tag = sp.filter(function (r) { return r.record_id === 3305; })[0];
    var a = 7;
    var sl = Kit.slider(Kit.$("dl-controls"), { label: "Dial a, in millionths of a kg per cm³", min: 0, max: 8, step: 0.05, value: a, show: function (v) { return Kit.fmt(v, 2); } }, function (v) { a = v; draw(); });
    var bar = Kit.$("dl-bar");
    Kit.button(bar, "Set a to 7", function () { a = 7; sl.set(a); draw(); });
    Kit.button(bar, "Set a to 0", function () { a = 0; sl.set(a); draw(); });
    Kit.button(bar, "Show the best a", function () { a = 3.3; sl.set(a); draw(); });
    function miss(av) { var s = 0, i; for (i = 0; i < L.length; i++) s += Math.abs(W[i] - av * 1e-6 * L[i] * L[i] * L[i]); return s / L.length; }
    function draw() {
      var av = a * 1e-6, gx = [], gy = [], i;
      for (i = 0; i < L.length; i++) { var p = av * L[i] * L[i] * L[i]; gx.push(L[i], L[i], null); gy.push(W[i], p, null); }
      var cx = [], cy = [];
      for (var l = 15; l <= 65; l += 0.5) { cx.push(l); cy.push(av * l * l * l); }
      Kit.plot("dl-plot", function (P) {
        return [
          { x: gx, y: gy, mode: "lines", line: { color: P.muted, width: 1 }, opacity: 0.45, hoverinfo: "skip" },
          { x: L, y: W, mode: "markers", marker: { color: P.navy, size: 6, opacity: 0.6 }, hovertemplate: "%{x:.1f} cm, %{y:.3f} kg<extra>made-up record</extra>" },
          { x: [15, 65], y: [avgW, avgW], mode: "lines", line: { color: P.ink2, width: 2, dash: "dash" }, hoverinfo: "skip" },
          { x: cx, y: cy, mode: "lines", line: { color: P.teal, width: 3.5 }, hoverinfo: "skip" },
          { x: [tag.length_cm], y: [tag.weight_kg], mode: "markers+text", text: ["Tag 3305"], textposition: "top left", textfont: { color: P.coral, size: 13 }, marker: { color: P.paper, size: 11, line: { color: P.coral, width: 3 } }, hovertemplate: "Tag 3305: 41.0 cm, 0.228 kg<extra></extra>" }
        ];
      }, function (P) {
        return {
          margin: { l: 56, r: 12, t: 12, b: 46 },
          xaxis: { title: { text: "length, cm" }, range: [15, 65] }, yaxis: { title: { text: "weight, kg" }, range: [0, 0.85] },
          annotations: [
            { x: 17, y: avgW, xanchor: "left", yanchor: "bottom", text: "lazy model: always " + Kit.fmt(avgW, 3) + " kg", showarrow: false, font: { color: P.ink2, size: 13 } },
            (function () { var xl = av > 0 ? Math.min(60, Math.pow(0.7 / av, 1 / 3)) : 55; xl = Math.max(30, xl); var yl = av * xl * xl * xl; return { x: xl, y: yl, xanchor: "right", yanchor: "bottom", xshift: -6, text: "weight = a × length³", showarrow: false, font: { color: P.teal, size: 13 } }; })()
          ]
        };
      });
      var m = miss(a);
      Kit.setText("dl-a", Kit.fmt(a, 2));
      Kit.setText("dl-miss", Kit.fmt(m, 3) + " kg");
      Kit.setText("dl-lazy", Kit.fmt(lazyMiss, 3) + " kg");
      Kit.setText("dl-verdict", m < lazyMiss ? "Better than the lazy model." : "Not better than the lazy model.");
      var pred = av * Math.pow(tag.length_cm, 3);
      Kit.$("dl-eqn").innerHTML = box("c-teal", Kit.fmt(pred, 3), "predicted weight, kg") + op("=") + box("c-sand", Kit.fmt(a, 2) + "&times;10<sup>&minus;6</sup>", "the dial a") + op("&times;") + box("c-navy", thousands(Math.pow(tag.length_cm, 3)), "41.0<sup>3</sup>, the length cubed") +
        op("&rarr;") + box("c-coral", Kit.fmt(Math.abs(pred - tag.weight_kg), 3), "miss on Tag 3305 (true weight 0.228)");
    }
    draw();
  }

  // ---------- 2. which column is the answer ----------
  function columnFig() {
    var ids = [3305, 4127, 5519, 1071, 1052, 2137];
    var rows = ids.map(function (id) { return REC.filter(function (r) { return r.record_id === id; })[0]; });
    var newShark = { record_id: "new", species: null, length_cm: 47.0, weight_kg: 0.36, girth_cm: 13.0, depth_m: 20.0, month: 3, mature: null };
    var cols = [
      { k: "record_id", h: "tag", fmt: function (v) { return String(v); } },
      { k: "species", h: "species", fmt: function (v) { return v === "milk" ? "milk" : v === "bamboo" ? "bamboo" : v === "ganges" ? "Ganges" : v; } },
      { k: "length_cm", h: "length, cm", fmt: function (v) { return v.toFixed(1); } },
      { k: "weight_kg", h: "weight, kg", fmt: function (v) { return v.toFixed(3); } },
      { k: "girth_cm", h: "girth, cm", fmt: function (v) { return v.toFixed(1); } },
      { k: "depth_m", h: "depth, m", fmt: function (v) { return v.toFixed(1); } },
      { k: "month", h: "month", fmt: function (v) { return String(v); } },
      { k: "mature", h: "adult?", fmt: function (v) { return v ? "adult" : "juvenile"; } }
    ];
    var labelKey = "species", used = { length_cm: true, weight_kg: true, girth_cm: true, depth_m: true, month: true, species: true, mature: true };
    var Q = [["species", "Which species is it?"], ["weight_kg", "How much does it weigh?"], ["mature", "Is it an adult?"]];
    var chipsBox = Kit.$("cp-chips");
    Kit.chips(chipsBox, Q.map(function (q) { return q[1]; }), 0, function (i) { labelKey = Q[i][0]; draw(); });
    function draw() {
      var t = Kit.$("cp-table"); t.innerHTML = "";
      var thead = Kit.html("thead", {}, t), r1 = Kit.html("tr", {}, thead), r2 = Kit.html("tr", { class: "pick" }, thead);
      var d = 0;
      cols.forEach(function (c) {
        var role = c.k === "record_id" ? "name" : (c.k === labelKey ? "label" : (used[c.k] ? "input" : "off"));
        if (role === "input") d++;
        var th = Kit.html("th", { class: "role-" + role, scope: "col" }, r1, c.h);
        Kit.html("small", {}, th, role === "label" ? "the answer" : role === "input" ? "input" : role === "name" ? "name, not an input" : "not used");
        var td = Kit.html("td", { class: "role-" + role }, r2);
        if (role === "input" || role === "off") {
          var lab = Kit.html("label", {}, td), cb = Kit.html("input", { type: "checkbox" }, lab);
          cb.checked = role === "input"; lab.appendChild(document.createTextNode(" use"));
          cb.addEventListener("change", function () { used[c.k] = cb.checked; draw(); });
        }
      });
      var tb = Kit.html("tbody", {}, t);
      rows.concat([newShark]).forEach(function (r, ri) {
        var tr = Kit.html("tr", { class: ri === rows.length ? "newrow" : "" }, tb);
        cols.forEach(function (c) {
          var role = c.k === "record_id" ? "name" : (c.k === labelKey ? "label" : (used[c.k] ? "input" : "off"));
          var txt = r[c.k] == null ? "?" : c.fmt(r[c.k]);
          if (ri === rows.length && c.k === "record_id") txt = "new";
          Kit.html("td", { class: "role-" + role + (r[c.k] == null ? " unknown" : "") }, tr, txt);
        });
      });
      var isNum = labelKey === "weight_kg";
      Kit.setText("cp-task", isNum ? "Regression: the answer is a number." : (labelKey === "species" ? "Classification: the answer is one of five species." : "Classification: the answer is yes or no."));
      Kit.setText("cp-m", "600");
      Kit.setText("cp-d", String(d));
      Kit.setText("cp-x", d ? "(" + cols.filter(function (c) { return c.k !== labelKey && used[c.k] && c.k !== "record_id"; }).map(function (c) { return c.fmt(rows[0][c.k]); }).join(", ") + ")" : "( )");
      Kit.setText("cp-y", cols.filter(function (c) { return c.k === labelKey; })[0].fmt(rows[0][labelKey]));
      var note = "";
      if (d === 0) note = "With no inputs, the best a model can do is say the same thing for all sharks. That is the lazy model of the next section.";
      else if (labelKey === "weight_kg" && used.girth_cm) note = "Warning: in this made-up table, girth was computed from weight. An input that is built from the answer makes a model look better than it is. A4 calls this a leak.";
      else if (labelKey === "mature" && used.mature) note = "";
      else note = "The tag number is a name. It has no link to the animal's size or species, so it is never an input.";
      Kit.setText("cp-note", note);
    }
    draw();
  }

  // ---------- 3. beat the lazy model ----------
  function lazyFig() {
    var FE = [
      { k: "length_cm", name: "length, cm", d: 0 }, { k: "weight_kg", name: "weight, kg", d: 2 }, { k: "girth_cm", name: "girth, cm", d: 1 },
      { k: "depth_m", name: "depth, m", d: 1 }, { k: "temp_c", name: "water temperature, °C", d: 1 }, { k: "month", name: "month of the catch", d: 0 }
    ];
    var feat = 0, nCuts = 1, cuts = [], labels = ["spadenose", "spadenose", "spadenose"];
    var counts = {}; SPECIES.forEach(function (s) { counts[s] = REC.filter(function (r) { return r.species === s; }).length; });
    var lazyHits = counts.spadenose, M = REC.length;
    var node = Kit.$("lz-svg"), svg = d3.select(node), W = 600, H = 300, sx, geom = {};
    var ctl = Kit.$("lz-controls");
    var fsel = Kit.select(ctl, "Input the rule looks at", FE.map(function (f, i) { return { value: String(i), label: f.name }; }), "0", function (v) { feat = +v; reset(); });
    var bar = Kit.$("lz-bar");
    Kit.chips(bar, ["One cut-off", "Two cut-offs"], 0, function (i) { nCuts = i + 1; placeCuts(); draw(); });
    Kit.button(bar, "Fill the parts with their commonest species", function () { fillCommon(); draw(); });
    Kit.button(bar, "Back to the lazy model", function () { labels = ["spadenose", "spadenose", "spadenose"]; draw(); });
    Kit.button(bar, "Show the best rule for this input", function () { best(); draw(); });
    var sels = Kit.$("lz-parts");

    function vals() { return REC.map(function (r) { return r[FE[feat].k]; }); }
    function range() { var v = vals(); return [Math.min.apply(null, v), Math.max.apply(null, v)]; }
    function quantile(q) { var v = vals().slice().sort(function (a, b) { return a - b; }); return v[Math.min(v.length - 1, Math.floor(q * v.length))]; }
    function placeCuts() { cuts = nCuts === 1 ? [quantile(0.5)] : [quantile(0.4), quantile(0.75)]; if (feat === 5) cuts = cuts.map(function (c) { return Math.round(c) + 0.5; }); }
    function reset() { placeCuts(); labels = ["spadenose", "spadenose", "spadenose"]; draw(); }
    function seg(v, cs) { var i = 0; while (i < cs.length && v >= cs[i]) i++; return i; }
    function cutsSorted() { return cuts.slice(0, nCuts).sort(function (a, b) { return a - b; }); }
    function commonest(cs, v) {
      var tab = []; for (var s = 0; s <= cs.length; s++) tab.push({ spadenose: 0, milk: 0, bamboo: 0, hammerhead: 0, ganges: 0 });
      for (var i = 0; i < REC.length; i++) tab[seg(v[i], cs)][REC[i].species]++;
      var lab = [], hits = 0;
      tab.forEach(function (t) { var b = "spadenose"; SPECIES.forEach(function (sp) { if (t[sp] > t[b]) b = sp; }); lab.push(b); hits += t[b]; });
      return { labels: lab, hits: hits };
    }
    function fillCommon() { var r = commonest(cutsSorted(), vals()); labels = r.labels.concat(["spadenose", "spadenose"]).slice(0, 3); }
    function best() {
      var v = vals(), u = Array.from(new Set(v)).sort(function (a, b) { return a - b; }), cand = [], i, j, bestHits = -1, bc = null;
      var step = Math.max(1, Math.floor(u.length / 60));
      for (i = 0; i < u.length - 1; i += step) cand.push((u[i] + u[Math.min(u.length - 1, i + 1)]) / 2);
      if (nCuts === 1) { cand.forEach(function (c) { var r = commonest([c], v); if (r.hits > bestHits) { bestHits = r.hits; bc = [c]; } }); }
      else { for (i = 0; i < cand.length; i++) for (j = i + 1; j < cand.length; j++) { var r = commonest([cand[i], cand[j]], v); if (r.hits > bestHits) { bestHits = r.hits; bc = [cand[i], cand[j]]; } } }
      cuts = bc; fillCommon();
    }
    function fmtV(v) { var d = FE[feat].d; return feat === 5 ? "month " + Math.round(v - 0.5) + "/" + Math.round(v + 0.5) : v.toFixed(d === 0 && feat === 0 ? 1 : d + (feat === 1 ? 1 : 0)); }

    function build() {
      var P = Kit.palette(), C = spCol(P), cs = cutsSorted();
      W = Kit.width(node); H = W < 520 ? 300 : 320;
      var ml = 44, mr = 12, mt = 46, mb = 44, iw = W - ml - mr, ih = H - mt - mb;
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      var v = vals(), rg = range(), cap = feat === 5 ? 12.5 : quantile(0.985) + 1e-9, nb = feat === 5 ? 12 : 36, lo = feat === 5 ? 0.5 : rg[0], hi = feat === 5 ? 12.5 : cap, bw = (hi - lo) / nb;
      sx = d3.scaleLinear().domain([lo, hi]).range([ml, ml + iw]);
      var bins = []; for (var b = 0; b < nb; b++) bins.push({ x0: lo + b * bw, parts: [] });
      var tally = bins.map(function () { var o = {}; SPECIES.forEach(function (s) { o[s + "1"] = 0; o[s + "0"] = 0; }); return o; });
      var hits = 0, perSp = {}; SPECIES.forEach(function (s) { perSp[s] = 0; });
      for (var i = 0; i < REC.length; i++) {
        var bi = Math.min(nb - 1, Math.max(0, Math.floor((v[i] - lo) / bw))), ok = labels[seg(v[i], cs)] === REC[i].species;
        tally[bi][REC[i].species + (ok ? "1" : "0")]++; if (ok) { hits++; perSp[REC[i].species]++; }
      }
      var maxN = 0; tally.forEach(function (t) { var s = 0; SPECIES.forEach(function (sp) { s += t[sp + "1"] + t[sp + "0"]; }); maxN = Math.max(maxN, s); });
      var sy = d3.scaleLinear().domain([0, maxN * 1.05]).range([mt + ih, mt]);
      geom = { ml: ml, mr: mr, mt: mt, ih: ih, iw: iw, lo: lo, hi: hi, sy: sy };
      // axes
      var ax = svg.append("g").attr("transform", "translate(0," + (mt + ih) + ")").call(d3.axisBottom(sx).ticks(W < 480 ? 5 : 8).tickFormat(feat === 5 ? d3.format("d") : null));
      ax.selectAll("path,line").attr("stroke", P.muted); ax.selectAll("text").attr("fill", P.muted).attr("font-size", 12.5);
      var ay = svg.append("g").attr("transform", "translate(" + ml + ",0)").call(d3.axisLeft(sy).ticks(4));
      ay.selectAll("path,line").attr("stroke", P.muted); ay.selectAll("text").attr("fill", P.muted).attr("font-size", 12.5);
      svg.append("text").attr("x", ml + iw / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.ink2).text(FE[feat].name + (feat === 5 ? "" : " (last bin: this value and above)"));
      svg.append("text").attr("transform", "rotate(-90)").attr("x", -(mt + ih / 2)).attr("y", 12).attr("text-anchor", "middle").attr("font-size", 13.5).attr("fill", P.ink2).text("sharks in the bin");
      // segment bands
      var edges = [lo].concat(cs).concat([hi]);
      for (var s = 0; s < edges.length - 1; s++) {
        svg.append("rect").attr("x", sx(edges[s])).attr("y", mt).attr("width", Math.max(0, sx(edges[s + 1]) - sx(edges[s]))).attr("height", ih).attr("fill", s % 2 ? rgba(P.muted, 0.08) : "none");
      }
      // bars
      var best = {}; SPECIES.forEach(function (sp) { best[sp] = { h: 0 }; });
      bins.forEach(function (bn, b) {
        var y = 0, t = tally[b];
        SPECIES.forEach(function (sp) {
          [1, 0].forEach(function (okf) {
            var n = t[sp + okf]; if (!n) return;
            var y0 = sy(y), y1 = sy(y + n);
            svg.append("rect").attr("x", sx(bn.x0) + 0.6).attr("width", Math.max(1, sx(bn.x0 + bw) - sx(bn.x0) - 1.2)).attr("y", y1).attr("height", Math.max(0.5, y0 - y1))
              .attr("fill", okf ? C[sp] : rgba(C[sp], 0.22)).attr("stroke", okf ? "none" : C[sp]).attr("stroke-width", okf ? 0 : 0.8).attr("stroke-dasharray", okf ? null : "2 2");
            if (okf && y0 - y1 > best[sp].h) best[sp] = { h: y0 - y1, x: sx(bn.x0 + bw / 2), y: (y0 + y1) / 2 };
            y += n;
          });
        });
      });
      // direct labels on the largest solid part of a species when it is tall enough
      SPECIES.forEach(function (sp) {
        var bsp = best[sp]; if (!bsp.x || bsp.h < 15) return;
        svg.append("text").attr("x", bsp.x).attr("y", bsp.y + 4).attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.paper).attr("paint-order", "stroke").attr("stroke", C[sp]).attr("stroke-width", 3).text(NAME[sp]);
      });
      // band captions
      for (var q = 0; q < edges.length - 1; q++) {
        var cx = (sx(edges[q]) + sx(edges[q + 1])) / 2;
        svg.append("text").attr("x", cx).attr("y", mt - 22).attr("text-anchor", "middle").attr("font-size", 13).attr("fill", P.ink).attr("font-weight", 600).text("says: " + NAME[labels[q]]);
      }
      // cut handles
      cuts.slice(0, nCuts).forEach(function (c, ci) {
        var g = svg.append("g").attr("tabindex", 0).attr("role", "slider").attr("aria-label", "Cut-off " + (ci + 1)).attr("aria-valuenow", c.toFixed(2)).style("cursor", "ew-resize").style("outline", "none");
        g.append("line").attr("x1", sx(c)).attr("x2", sx(c)).attr("y1", mt - 8).attr("y2", mt + ih).attr("stroke", P.coral).attr("stroke-width", 2.5);
        g.append("circle").attr("cx", sx(c)).attr("cy", mt - 8).attr("r", 8).attr("fill", P.paper).attr("stroke", P.coral).attr("stroke-width", 3);
        g.append("line").attr("x1", sx(c)).attr("x2", sx(c)).attr("y1", mt - 14).attr("y2", mt + ih).attr("stroke", "transparent").attr("stroke-width", 22);
        g.append("text").attr("x", sx(c)).attr("y", mt + ih + 16).attr("text-anchor", "middle").attr("font-size", 12.5).attr("font-weight", 600).attr("fill", P.coral).attr("paint-order", "stroke").attr("stroke", P.bg).attr("stroke-width", 4).text(fmtV(c));
        var step = (hi - lo) / 200;
        g.on("keydown", function (e) {
          var dlt = e.key === "ArrowLeft" ? -step : e.key === "ArrowRight" ? step : 0; if (!dlt) return;
          e.preventDefault(); cuts[ci] = Math.max(lo + step, Math.min(hi - step, cuts[ci] + dlt)); draw(); var gs = svg.selectAll("g[role=slider]").nodes(); if (gs[ci]) gs[ci].focus();
        });
        g.call(d3.drag().on("start", function () { g.raise(); }).on("drag", function (e) {
          var xv = sx.invert(Math.max(ml, Math.min(ml + iw, e.x))); if (feat === 5) xv = Math.round(xv - 0.5) + 0.5;
          cuts[ci] = Math.max(lo + 1e-6, Math.min(hi - 1e-6, xv)); drawFast();
        }).on("end", function () { draw(); }));
      });
      return { hits: hits, perSp: perSp, cs: cs };
    }
    var drawTimer = 0;
    function drawFast() { draw(true); }
    function draw(fast) {
      var r = build();
      // part selectors
      if (!fast) {
        sels.innerHTML = "";
        var cs = r.cs, edges = [null].concat(cs).concat([null]);
        for (var s = 0; s < cs.length + 1; s++) {
          (function (s) {
            var lab = Kit.html("label", {}, sels), nm = s === 0 ? "Below " + fmtV(cs[0]) : (s === cs.length ? "From " + fmtV(cs[cs.length - 1]) + " up" : fmtV(cs[s - 1]) + " to " + fmtV(cs[s]));
            Kit.html("span", {}, lab, nm + ": say");
            var sel = Kit.html("select", {}, lab);
            SPECIES.forEach(function (sp) { var o = Kit.html("option", { value: sp }, sel, NAME[sp]); if (labels[s] === sp) o.selected = true; });
            sel.addEventListener("change", function () { labels[s] = sel.value; draw(); });
          })(s);
        }
      }
      var acc = r.hits / M, lazy = lazyHits / M;
      Kit.$("lz-eqn").innerHTML = box("c-coral", thousands(r.hits), "your rule is right") + op("&divide;") + box("", String(M), "sharks") + op("=") + box("c-teal", Kit.pct(acc, 1), "your accuracy") +
        op("&nbsp;&nbsp;vs") + box("c-navy", Kit.pct(lazy, 1), "lazy model: " + lazyHits + " of " + M + ", always spadenose");
      var diff = (acc - lazy) * 100;
      Kit.setText("lz-status", diff > 0.05 ? "Your rule beats the lazy model by " + diff.toFixed(1) + " points." : (diff < -0.05 ? "Your rule is " + (-diff).toFixed(1) + " points below the lazy model." : "Your rule ties with the lazy model."));
      Kit.$("lz-sp").innerHTML = SPECIES.map(function (sp) { return '<div><b>' + r.perSp[sp] + ' of ' + counts[sp] + '</b><span>' + NAME[sp] + ' right</span></div>'; }).join("");
      node.setAttribute("aria-label", "Stacked histogram of the " + FE[feat].name + " of 600 made-up sharks, coloured by species, with " + nCuts + " cut-off" + (nCuts > 1 ? "s" : "") + ". Your rule scores " + Kit.pct(acc, 1) + ". The lazy model scores " + Kit.pct(lazy, 1) + ".");
    }
    placeCuts(); draw();
    Kit.onTheme(function () { draw(); });
    window.addEventListener("resize", Kit.debounce(function () { draw(); }));
  }

  // ---------- 4. trying dial settings at random ----------
  function searchFig() {
    var sp = REC.filter(function (r) { return r.species === "spadenose"; });
    var L = sp.map(function (r) { return r.length_cm; }), Wt = sp.map(function (r) { return r.weight_kg; });
    function miss(a) { var s = 0, i; for (i = 0; i < L.length; i++) s += Math.abs(Wt[i] - a * 1e-6 * L[i] * L[i] * L[i]); return s / L.length; }
    var grid = [], curve = [], i; for (i = 0; i <= 160; i++) { grid.push(i * 0.05); curve.push(miss(i * 0.05)); }
    var tries = [], reveal = false, rnd = Kit.rng(2024);
    var bar = Kit.$("sr-bar");
    Kit.button(bar, "Try one random a", function () { tryA(rnd() * 8); });
    Kit.button(bar, "Try 10", function () { for (var k = 0; k < 10; k++) tryA(rnd() * 8); });
    Kit.button(bar, "Try 10 near the best so far", function () { var b = bestTry(); if (!b) return; for (var k = 0; k < 10; k++) tryA(Math.max(0, Math.min(8, b.a + (rnd() - 0.5) * 0.8))); });
    Kit.button(bar, "Clear", function () { tries = []; draw(); });
    Kit.chips(bar, ["Hide the curve", "Show the whole curve"], 0, function (j) { reveal = j === 1; draw(); });
    function tryA(a) { tries.push({ a: a, m: miss(a) }); draw(); }
    function bestTry() { var b = null; tries.forEach(function (t) { if (!b || t.m < b.m) b = t; }); return b; }
    function draw() {
      var b = bestTry();
      Kit.plot("sr-plot", function (P) {
        var tr = [];
        if (reveal) tr.push({ x: grid, y: curve, mode: "lines", line: { color: P.navy, width: 2.5 }, hoverinfo: "skip" });
        tr.push({ x: tries.map(function (t) { return t.a; }), y: tries.map(function (t) { return t.m; }), mode: "markers", marker: { color: P.coral, size: 9, line: { color: P.paper, width: 1.5 } }, hovertemplate: "a = %{x:.2f}, average miss %{y:.3f} kg<extra>a try</extra>" });
        if (b) tr.push({ x: [b.a], y: [b.m], mode: "markers", marker: { color: "rgba(0,0,0,0)", size: 19, line: { color: P.teal, width: 3 } }, hoverinfo: "skip" });
        return tr;
      }, { margin: { l: 56, r: 12, t: 12, b: 46 }, xaxis: { title: { text: "dial a, millionths of a kg per cm³" }, range: [0, 8] }, yaxis: { title: { text: "average miss, kg" }, range: [0, 0.42] } });
      Kit.setText("sr-n", String(tries.length));
      Kit.setText("sr-best", b ? "a = " + Kit.fmt(b.a, 2) + " (average miss " + Kit.fmt(b.m, 3) + " kg)" : "none yet");
      Kit.setText("sr-true", reveal ? "The lowest point of the curve is a = 3.30, average miss 0.025 kg." : "The curve is hidden. A search normally sees only the settings it has tried.");
    }
    tries = [{ a: 1.2, m: miss(1.2) }, { a: 5.6, m: miss(5.6) }, { a: 6.9, m: miss(6.9) }];
    draw();
    var dsl = Kit.slider(Kit.$("sr-controls"), { label: "Number of dials d", min: 1, max: 30, step: 1, value: 3 }, upd);
    function upd() {
      var d = dsl.get(), n = Math.pow(10, d), secs = n / 1e6, txt;
      if (secs < 1) txt = "under a second";
      else if (secs < 3600) txt = Math.round(secs).toLocaleString("en-US") + " seconds";
      else if (secs < 86400) txt = (secs / 3600).toFixed(1) + " hours";
      else if (secs < 3.156e7) txt = (secs / 86400).toFixed(1) + " days";
      else if (secs < 4.36e17) txt = thousands(secs / 3.156e7) + " years";
      else txt = "longer than the age of the universe";
      var big = d <= 6 ? thousands(n) : "10<sup>" + d + "</sup>";
      Kit.$("sr-eqn").innerHTML = box("c-sand", "10", "settings tried per dial") + op("<sup>d</sup> =") + box("c-navy", big, "combinations with d = " + d) + op("&rarr;") + box("c-coral", txt, "at a million tries per second");
    }
    upd();
  }
})();
