// Chapter 0.8 Reading charts and tables: figures on the made-up SharkCam catch table. Pure helpers first (exported for tests).
(function () {
  "use strict";
  var SPECIES = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"];
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  function mean(a) { return a.reduce(function (s, x) { return s + x; }, 0) / a.length; }
  function sd(a) { var m = mean(a); return Math.sqrt(a.reduce(function (s, x) { return s + (x - m) * (x - m); }, 0) / (a.length - 1)); }
  function quantile(a, q) { var s = a.slice().sort(function (x, y) { return x - y; }), p = (s.length - 1) * q, i = Math.floor(p); return s[i] + (s[Math.min(i + 1, s.length - 1)] - s[i]) * (p - i); }
  function median(a) { return quantile(a, 0.5); }
  function ols(xs, ys) {
    var mx = mean(xs), my = mean(ys), sxx = 0, sxy = 0, syy = 0;
    xs.forEach(function (x, i) { sxx += (x - mx) * (x - mx); sxy += (x - mx) * (ys[i] - my); syy += (ys[i] - my) * (ys[i] - my); });
    var b = sxy / sxx; return { b: b, a: my - b * mx, r2: sxy * sxy / (sxx * syy), r: sxy / Math.sqrt(sxx * syy) };
  }
  function lieFactor(sf, ss, tf, ts) { return (Math.abs(ss - sf) / sf) / (Math.abs(ts - tf) / tf); }
  function monthly(recs) {
    var n = [], t = [];
    for (var m = 1; m <= 12; m++) { var r = recs.filter(function (x) { return x.month === m; }); n.push(r.length); t.push(r.length ? mean(r.map(function (x) { return x.temp_c; })) : null); }
    return { n: n, t: t };
  }
  function commas(x) { return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  if (typeof module !== "undefined" && module.exports) module.exports = { mean: mean, sd: sd, median: median, quantile: quantile, ols: ols, lieFactor: lieFactor, monthly: monthly };
  if (typeof document === "undefined") return;

  var H = function (tag, attrs, parent, text) { return Kit.html(tag, attrs, parent, text); };
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function spCol(P) { return { spadenose: P.navy, milk: P.coral, bamboo: P.teal, hammerhead: P.sand, ganges: P.ink2 }; }
  function item(parent, label, value) { var s = H("span", {}, parent, label + " "); H("b", {}, s, value); return s; }
  function box(parent, cls, b, i) { var t = H("span", { class: "t " + cls }, parent); H("b", {}, t, b); H("i", {}, t, i); }

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("../data/sharks/catch.json"), Kit.data("data/results.json")]).then(function (d) {
      var R = d[0].records, res = d[1];
      types(R); scales(R); logGap(); gallery(R); matrix(res); results(res);
    }).catch(function (e) { console.error(e); [].forEach.call(document.querySelectorAll(".plot"), function (p) { p.textContent = "The data did not load."; }); });
  });

  // ======================= 1. chart types =======================
  function types(R) {
    var view = 0, by = 0, qs = ["How are the lengths spread out, and where are most of the sharks?", "How do the species compare in typical length and in spread?", "Does weight rise with length, and how fast?", "How many records does a species have?"];
    var L = R.map(function (r) { return r.length_cm; });
    var bw = Kit.slider(Kit.$("ty-controls"), { label: "bin width, cm (histogram)", min: 2, max: 40, step: 1, value: 20, digits: 0 }, draw);
    Kit.chips(Kit.$("ty-bar"), ["Histogram", "Box plot", "Scatter plot", "Bar chart"], 0, function (i) { view = i; draw(); });
    var cc = H("span", {}, Kit.$("ty-bar")); Kit.chips(cc, ["One colour", "Colour by species"], 0, function (i) { by = i; draw(); });
    function draw() {
      var w = bw.get(), ro = Kit.$("ty-read"); clear(ro);
      Kit.$("ty-q").textContent = "Question this chart answers: " + qs[view];
      Kit.plot("ty-plot", function (P) {
        var C = spCol(P), tr = [];
        if (view === 0) {
          if (by === 0) tr.push({ type: "histogram", x: L, xbins: { start: 0, end: 250, size: w }, autobinx: false, marker: { color: P.navy, line: { color: P.paper, width: 1 } }, hovertemplate: "%{x} cm: %{y} records<extra></extra>" });
          else SPECIES.forEach(function (s) { tr.push({ type: "histogram", name: s, x: R.filter(function (r) { return r.species === s; }).map(function (r) { return r.length_cm; }), xbins: { start: 0, end: 250, size: w }, autobinx: false, marker: { color: C[s], line: { color: P.paper, width: 1 } }, hovertemplate: s + ", %{x} cm: %{y} records<extra></extra>" }); });
        } else if (view === 1) {
          SPECIES.forEach(function (s) { tr.push({ type: "box", name: s, y: R.filter(function (r) { return r.species === s; }).map(function (r) { return r.length_cm; }), marker: { color: C[s], size: 4 }, line: { color: C[s] }, boxpoints: "outliers" }); });
        } else if (view === 2) {
          if (by === 0) tr.push({ type: "scatter", mode: "markers", x: L, y: R.map(function (r) { return r.weight_kg; }), marker: { color: P.navy, size: 6, opacity: 0.6 }, text: R.map(function (r) { return r.species + ", tag " + r.record_id; }), hovertemplate: "%{text}<br>%{x} cm, %{y} kg<extra></extra>" });
          else SPECIES.forEach(function (s) { var rr = R.filter(function (r) { return r.species === s; }); tr.push({ type: "scatter", mode: "markers", name: s, x: rr.map(function (r) { return r.length_cm; }), y: rr.map(function (r) { return r.weight_kg; }), marker: { color: C[s], size: 6, opacity: 0.7 }, text: rr.map(function (r) { return "tag " + r.record_id; }), hovertemplate: s + ", %{text}<br>%{x} cm, %{y} kg<extra></extra>" }); });
        } else {
          var cnt = SPECIES.map(function (s) { return R.filter(function (r) { return r.species === s; }).length; });
          tr.push({ type: "bar", x: SPECIES, y: cnt, marker: { color: SPECIES.map(function (s) { return C[s]; }) }, text: cnt.map(String), textposition: "outside", cliponaxis: false, hovertemplate: "%{x}: %{y} records<extra></extra>" });
        }
        return tr;
      }, function (P) {
        var lay = { barmode: "stack", showlegend: (view === 0 || view === 2) && by === 1, legend: { orientation: "h", y: 1.12, x: 0 }, margin: { l: 56, r: 12, t: 24, b: 48 } };
        if (view === 0) { lay.xaxis = { title: { text: "length, cm" }, range: [0, 250] }; lay.yaxis = { title: { text: "records" } }; }
        else if (view === 1) { lay.xaxis = { title: { text: "species" } }; lay.yaxis = { title: { text: "length, cm" } }; }
        else if (view === 2) { lay.xaxis = { title: { text: "length, cm" } }; lay.yaxis = { title: { text: "weight, kg" } }; }
        else { lay.xaxis = { title: { text: "species" } }; lay.yaxis = { title: { text: "records" }, range: [0, 320] }; }
        return lay;
      });
      item(ro, "records:", String(R.length));
      if (view === 0) { item(ro, "bin width:", w + " cm"); item(ro, "bins:", String(Math.ceil(250 / w))); item(ro, "median:", median(L).toFixed(1) + " cm"); item(ro, "mean:", mean(L).toFixed(1) + " cm"); }
      else if (view === 1) { SPECIES.forEach(function (s) { item(ro, s + " median:", median(R.filter(function (r) { return r.species === s; }).map(function (r) { return r.length_cm; })).toFixed(1)); }); }
      else if (view === 2) item(ro, "longest:", Math.max.apply(null, L).toFixed(1) + " cm");
      else item(ro, "spadenose share:", Kit.pct(268 / 600, 1));
    }
    draw();
  }

  // ======================= 2. scales =======================
  function scales(R) {
    var ax = 0, by = 0;
    var X = R.map(function (r) { return r.length_cm; }), Y = R.map(function (r) { return r.weight_kg; });
    var fit = ols(X.map(Math.log), Y.map(Math.log));
    Kit.chips(Kit.$("sc-bar"), ["Linear axes", "Log x", "Log y", "Log-log"], 0, function (i) { ax = i; draw(); });
    var cc = H("span", {}, Kit.$("sc-bar")); Kit.chips(cc, ["One colour", "Colour by species"], 0, function (i) { by = i; draw(); });
    function draw() {
      var lx = ax === 1 || ax === 3, ly = ax === 2 || ax === 3;
      Kit.plot("sc-plot", function (P) {
        var C = spCol(P), tr = [];
        if (by === 0) tr.push({ type: "scatter", mode: "markers", x: X, y: Y, marker: { color: P.navy, size: 6, opacity: 0.55 }, text: R.map(function (r) { return r.species + ", tag " + r.record_id; }), hovertemplate: "%{text}<br>%{x} cm, %{y} kg<extra></extra>" });
        else SPECIES.forEach(function (s) { var rr = R.filter(function (r) { return r.species === s; }); tr.push({ type: "scatter", mode: "markers", name: s, x: rr.map(function (r) { return r.length_cm; }), y: rr.map(function (r) { return r.weight_kg; }), marker: { color: C[s], size: 6, opacity: 0.7 }, text: rr.map(function (r) { return "tag " + r.record_id; }), hovertemplate: s + ", %{text}<br>%{x} cm, %{y} kg<extra></extra>" }); });
        var xs = [], ys = []; for (var i = 0; i <= 60; i++) { var x = 15 * Math.pow(250 / 15, i / 60); xs.push(x); ys.push(Math.exp(fit.a) * Math.pow(x, fit.b)); }
        tr.push({ type: "scatter", mode: "lines", x: xs, y: ys, line: { color: P.coral, width: 2.5, dash: "dash" }, hoverinfo: "skip", name: "fitted power law" });
        return tr;
      }, function (P) { return { showlegend: by === 1, legend: { orientation: "h", y: 1.1, x: 0 }, xaxis: { title: { text: "length, cm" }, type: lx ? "log" : "linear" }, yaxis: { title: { text: "weight, kg" }, type: ly ? "log" : "linear" }, margin: { l: 60, r: 12, t: 24, b: 48 } }; });
      var eq = Kit.$("sc-eqn"); clear(eq);
      box(eq, "c-coral", fit.b.toFixed(2), "fitted exponent b, the slope on log-log axes"); H("span", { class: "op" }, eq, "→"); box(eq, "c-navy", Math.pow(2, fit.b).toFixed(1) + "×", "weight when length doubles, 2 to the power b"); H("span", { class: "op" }, eq, "→"); box(eq, "c-teal", fit.r2.toFixed(3), "R squared of the line in the logs");
    }
    draw();
  }

  function logGap() {
    var log = false;
    Kit.chips(Kit.$("lg-bar"), ["Linear axis", "Log axis"], 0, function (i) { log = i === 1; draw(); });
    function draw() {
      Kit.plot("lg-plot", function (P) {
        return [{ type: "scatter", mode: "markers+text", x: [1, 10, 100, 1000], y: [0, 0, 0, 0], text: ["1", "10", "100", "1,000"], textposition: ["bottom center", "top center", "top center", "top center"], marker: { color: P.navy, size: 13 }, hoverinfo: "skip" },
          { type: "scatter", mode: "lines+text", x: [1, 10], y: [0.95, 0.95], text: ["gap A", ""], textposition: "top right", line: { color: P.coral, width: 3 }, hoverinfo: "skip" },
          { type: "scatter", mode: "lines+text", x: [100, 1000], y: [0.5, 0.5], text: ["gap B", ""], textposition: "top right", line: { color: P.teal, width: 3 }, hoverinfo: "skip" }];
      }, function (P) { return { xaxis: { type: log ? "log" : "linear", range: log ? [-0.2, 3.3] : [-50, 1130], title: { text: log ? "log axis: equal ratios" : "linear axis: equal differences" } }, yaxis: { visible: false, range: [-0.5, 1.2] }, margin: { l: 20, r: 20, t: 10, b: 52 } }; });
    }
    draw();
  }

  // ======================= 3. five misleading charts =======================
  function gallery(R) {
    var mo = monthly(R), milk = R.filter(function (r) { return r.species === "milk"; });
    var fem = milk.filter(function (r) { return r.sex === "F"; }).map(function (r) { return r.length_cm; }), mal = milk.filter(function (r) { return r.sex === "M"; }).map(function (r) { return r.length_cm; });
    var mf = mean(fem), mm = mean(mal), sdMilk = sd(milk.map(function (r) { return r.length_cm; }));
    var wmean = {}; SPECIES.forEach(function (s) { wmean[s] = mean(R.filter(function (r) { return r.species === s; }).map(function (r) { return r.weight_kg; })); });
    var cnt = {}; SPECIES.forEach(function (s) { cnt[s] = R.filter(function (r) { return r.species === s; }).length; });
    var cur = 0, vw = 0, plotDiv = Kit.$("mg-plot"), svg = Kit.$("mg-svg");
    var tabs = ["Truncated axis", "Pictogram", "Log scale", "Chosen window", "Two axes"];
    var heads = ["Headline: male milk sharks are far longer than females.", "Headline: spadenose landings dwarf bamboo shark landings.", "Headline: the heaviest species is only a few times heavier than the lightest.", "Headline: landings have quintupled since August.", "Headline: as the water warms, the sharks leave."];
    var quiz = [
      { q: "What is wrong with this chart?", o: ["The chart is fine: the bars show the two means correctly.", "The axis starts at 64 cm, so a difference of about 3% looks like more than double.", "The bars should be drawn as a line.", "The sample of milk sharks is far too big."], a: 1, why: "Bar heights are read from the axis start, so cutting the axis at 64 cm makes the difference look large. The means are 65.8 and 67.9 cm." },
      { q: "What is wrong with this chart?", o: ["The icons are drawn too small.", "The icons grow in height and width together, so their area grows with the square of the count.", "The two counts are wrong.", "Pictures should never be used in a chart."], a: 1, why: "The eye reads area. A count 3.7 times larger drawn with a height and a width 3.7 times larger fills 13.9 times the area." },
      { q: "What is wrong with this chart?", o: ["Bars on a log axis with no tick labels: the bar lengths do not show how many times heavier a species is.", "The weights are not in kilograms.", "A log scale is never allowed.", "The species are in the wrong order."], a: 0, why: "On a log axis a bar's length depends on where the axis starts, and without tick labels the reader cannot see that one gridline step is a factor of 10." },
      { q: "What is wrong with this chart?", o: ["It shows only the months after the closure, when landings recover, and hides the rest of the year.", "Landings cannot be counted by month.", "A line chart should be a bar chart.", "The tick labels are too small."], a: 0, why: "August to November follows the monsoon closure of June and July. A chart of the whole year shows that the rise is a recovery." },
      { q: "What is wrong with this chart?", o: ["The right axis range is chosen so that the lines cross, which invites a story that the data do not support.", "Temperature should be shown in Fahrenheit.", "Two lines can never be compared.", "The months are in the wrong order."], a: 0, why: "Moving the range of the right axis moves one line through the other without changing a number. The ten months also share a season and a closure." }
    ];
    var tabBox = Kit.$("mg-tabs");
    var tabChips = Kit.chips(tabBox, tabs, 0, function (i) { cur = i; vw = 0; show(); });
    function views() {
      var v = Kit.$("mg-views"); [].slice.call(v.querySelectorAll(".group")).forEach(function (g) { g.parentNode.removeChild(g); });
      var labels = [["As published (axis from 64 cm)", "Fair (axis from 0)"], ["As published (height and width scaled)", "Fair: area matches the count", "Fair: bars"], ["As published: log bars, no tick labels", "Dots on a labelled log axis", "Bars on a linear axis"], ["As published: August to November", "Fair: the whole year", "Choose a window"], ["As published: two value axes", "Fair: two panels"]][cur];
      Kit.chips(v, labels, vw, function (i) { vw = i; if (cur === 0) axisMin.set(i === 0 ? 64 : 0); render(); });
    }
    var axisMin, w0, w1, rmin, rmax;
    function controls() {
      var c = Kit.$("mg-controls"); clear(c);
      if (cur === 0) axisMin = Kit.slider(c, { label: "value axis starts at, cm", min: 0, max: 66, step: 1, value: 64, digits: 0 }, function () { vw = axisMin.get() === 0 ? 1 : 0; views(); render(); });
      if (cur === 3) { w0 = Kit.slider(c, { label: "window starts at month", min: 1, max: 12, step: 1, value: 8, digits: 0, show: function (v) { return MONTHS[v - 1]; } }, render); w1 = Kit.slider(c, { label: "window ends at month", min: 1, max: 12, step: 1, value: 11, digits: 0, show: function (v) { return MONTHS[v - 1]; } }, render); }
      if (cur === 4) { rmin = Kit.slider(c, { label: "right axis from, degrees C", min: 20, max: 27, step: 0.5, value: 24.5, digits: 1 }, render); rmax = Kit.slider(c, { label: "right axis to, degrees C", min: 29, max: 40, step: 0.5, value: 30, digits: 1 }, render); }
    }
    function show() {
      Kit.$("mg-head").textContent = heads[cur];
      clear(Kit.$("mg-quiz")); Kit.quiz("mg-quiz", [quiz[cur]]);
      controls(); views(); render();
    }
    function setMode(usePlot) { plotDiv.style.display = usePlot ? "" : "none"; svg.style.display = usePlot ? "none" : "block"; }
    function render() {
      var ro = Kit.$("mg-read"), eq = Kit.$("mg-eqn"), ex = Kit.$("mg-explain"); clear(ro); clear(eq); clear(ex); eq.style.display = (cur <= 1) ? "" : "none";
      if (cur === 0) truncated(ro, eq, ex);
      else if (cur === 1) pictogram(ro, eq, ex);
      else if (cur === 2) logbars(ro, ex);
      else if (cur === 3) windowed(ro, ex);
      else dual(ro, ex);
    }
    function ex_(parent, bold, text) { var p = H("p", {}, parent); H("b", {}, p, bold + " "); p.appendChild(document.createTextNode(text)); }

    function truncated(ro, eq, ex) {
      setMode(true); var a0 = axisMin.get(), hf = mf - a0, hm = mm - a0, shown = (hm - hf) / hf, trueE = (mm - mf) / mf, lf = shown / trueE;
      Kit.plot("mg-plot", function (P) { return [{ type: "bar", x: ["female (n = " + fem.length + ")", "male (n = " + mal.length + ")"], y: [mf, mm], marker: { color: [P.sand, P.navy] }, text: [mf.toFixed(1) + " cm", mm.toFixed(1) + " cm"], textposition: "outside", cliponaxis: false, hoverinfo: "skip" }]; },
        { yaxis: { title: { text: "mean length of milk sharks, cm" }, range: [a0, 72] }, xaxis: {}, margin: { l: 60, r: 12, t: 20, b: 40 } });
      box(eq, "c-navy", hf.toFixed(1) + " : " + hm.toFixed(1), "bar heights above the axis"); H("span", { class: "op" }, eq, "→"); box(eq, "c-coral", "+" + (100 * shown).toFixed(0) + "%", "effect shown"); H("span", { class: "op" }, eq, "÷"); box(eq, "c-sand", "+" + (100 * trueE).toFixed(1) + "%", "effect in the data"); H("span", { class: "op" }, eq, "="); box(eq, "c-teal", lf.toFixed(1), "lie factor");
      item(ro, "Difference of the means:", (mm - mf).toFixed(2) + " cm"); item(ro, "Spread of milk shark lengths (standard deviation):", sdMilk.toFixed(1) + " cm");
      ex_(ex, "What changed.", "The same two means are drawn, but the axis starts at " + a0 + " cm. Bar heights are read from the axis start, so the male bar is " + (hm / hf).toFixed(1) + " times as tall as the female bar, while the means differ by " + (100 * trueE).toFixed(1) + "%.");
      ex_(ex, "Also ask.", "The difference of " + (mm - mf).toFixed(1) + " cm is small next to the spread between single sharks, about " + sdMilk.toFixed(0) + " cm. A bar chart of means hides that spread.");
    }
    function pictogram(ro, eq, ex) {
      var c1 = cnt.bamboo, c2 = cnt.spadenose, ratio = c2 / c1;
      if (vw === 2) { setMode(true); Kit.plot("mg-plot", function (P) { return [{ type: "bar", x: ["bamboo", "spadenose"], y: [c1, c2], marker: { color: [P.teal, P.navy] }, text: [String(c1), String(c2)], textposition: "outside", cliponaxis: false, hoverinfo: "skip" }]; }, { yaxis: { title: { text: "records in the made-up table" }, range: [0, 310] }, margin: { l: 60, r: 12, t: 20, b: 40 } }); }
      else {
        setMode(false); var P = Kit.palette(), W = Kit.width(svg), Hh = 230; svg.setAttribute("viewBox", "0 0 " + W + " " + Hh); clear(svg);
        var base = 52, k = vw === 0 ? ratio : Math.sqrt(ratio), h1 = base, h2 = base * k, wk = 0.8;
        function fin(cx, h, col, label) { var w = h * wk; Kit.el("path", { d: "M" + (cx - w / 2) + "," + (Hh - 34) + " L" + (cx + w / 2) + "," + (Hh - 34) + " Q" + (cx - w * 0.05) + "," + (Hh - 34 - h * 0.55) + " " + (cx - w * 0.28) + "," + (Hh - 34 - h) + " Z", fill: col, "fill-opacity": 0.75, stroke: col, "stroke-width": 2 }, svg); var t = Kit.el("text", { x: cx, y: Hh - 12, "text-anchor": "middle", "font-size": 14, fill: P.ink2 }, svg); t.textContent = label; }
        fin(W * 0.3, h1, P.teal, "bamboo: " + c1 + " records"); fin(W * 0.65, h2, P.navy, "spadenose: " + c2 + " records");
      }
      var lf = lieFactor(1, vw === 0 ? ratio * ratio : ratio, c1, c2);
      box(eq, "c-navy", (vw === 0 ? ratio * ratio : ratio).toFixed(1) + "×", "area of the second mark, relative to the first"); H("span", { class: "op" }, eq, "→"); box(eq, "c-coral", "+" + (100 * ((vw === 0 ? ratio * ratio : ratio) - 1)).toFixed(0) + "%", "effect shown"); H("span", { class: "op" }, eq, "÷"); box(eq, "c-sand", "+" + (100 * (ratio - 1)).toFixed(0) + "%", "effect in the data"); H("span", { class: "op" }, eq, "="); box(eq, "c-teal", lf.toFixed(1), "lie factor");
      item(ro, "Count ratio:", ratio.toFixed(2) + " to 1");
      ex_(ex, "What changed.", vw === 0 ? "The count is 3.7 times larger, so the icon is drawn 3.7 times taller and 3.7 times wider. The eye compares the area, which is 13.9 times larger. The lie factor is above 4." : vw === 1 ? "The sides of the icon grow by the square root of the count ratio, so the area is 3.7 times larger and matches the data. Readers still judge area poorly, so label the counts." : "Bars use one dimension only, so their length is proportional to the count and the comparison is easy to read.");
    }
    function logbars(ro, ex) {
      var sp = ["spadenose", "bamboo", "milk", "hammerhead", "ganges"], y = sp.map(function (s) { return wmean[s]; }), ratioData = y[4] / y[0], hts = y.map(function (v) { return Math.log10(v / 0.1); });
      setMode(true);
      Kit.plot("mg-plot", function (P) {
        var C = spCol(P);
        if (vw === 1) return [{ type: "scatter", mode: "markers+text", x: sp, y: y, marker: { color: sp.map(function (s) { return C[s]; }), size: 14 }, text: y.map(function (v) { return v < 10 ? v.toPrecision(2) + " kg" : v.toFixed(0) + " kg"; }), textposition: "top center", hoverinfo: "skip" }];
        return [{ type: "bar", x: sp, y: y, marker: { color: sp.map(function (s) { return C[s]; }) }, text: vw === 0 ? null : y.map(function (v) { return v < 10 ? v.toPrecision(2) : v.toFixed(0); }), textposition: "outside", cliponaxis: false, hoverinfo: "skip" }];
      }, function (P) { return { yaxis: { title: { text: "mean weight, kg" }, type: vw === 2 ? "linear" : "log", range: vw === 2 ? [0, 45] : [-1, 2], showticklabels: vw !== 0, tickvals: vw === 0 ? [] : [0.1, 1, 10, 100], ticktext: ["0.1", "1", "10", "100"] }, xaxis: { title: { text: "species" } }, margin: { l: 60, r: 12, t: 20, b: 50 } }; });
      item(ro, "Heaviest to lightest, in the data:", ratioData.toFixed(0) + " times"); item(ro, "Tallest to shortest bar, as published:", (hts[4] / hts[0]).toFixed(1) + " times");
      ex_(ex, "What changed.", vw === 0 ? "A bar on a log axis starts at the bottom of the axis, which is 0.1 kg here. The bar for the Ganges shark is " + (hts[4] / hts[0]).toFixed(0) + " times as long as the one for the spadenose shark, but the weights differ by " + ratioData.toFixed(0) + " times. With no tick labels a reader cannot tell that one gridline step is a factor of 10." : vw === 1 ? "Dots do not pretend to start at zero, and the tick labels say 0.1, 1, 10 and 100 kg. The distance from the spadenose dot to the Ganges dot is a little over two tenfold steps, which is a factor of about 170." : "The linear axis tells the truth about the size of the gap: the Ganges mean of 38 kg dwarfs the others. It also shows why a log axis is useful, because the four smaller species are squeezed into the bottom. Only 3 Ganges records exist, so that mean rests on very little.");
    }
    function windowed(ro, ex) {
      setMode(true); var s, e;
      if (vw === 0) { s = 8; e = 11; } else if (vw === 1) { s = 1; e = 12; } else { s = Math.min(w0.get(), w1.get()); e = Math.max(w0.get(), w1.get()); }
      var xs = [], ys = []; for (var m = s; m <= e; m++) { xs.push(m); ys.push(mo.n[m - 1]); }
      var fit = xs.length > 1 ? ols(xs, ys) : { b: 0, a: ys[0] };
      Kit.plot("mg-plot", function (P) {
        var tr = [{ type: "scatter", mode: "lines+markers", x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], y: mo.n, line: { color: P.navy, width: 3 }, marker: { size: 9 }, hovertemplate: "%{text}: %{y} records<extra></extra>", text: MONTHS }];
        if (xs.length > 1) tr.push({ type: "scatter", mode: "lines", x: [s, e], y: [fit.a + fit.b * s, fit.a + fit.b * e], line: { color: P.coral, width: 2.5, dash: "dash" }, hoverinfo: "skip" });
        return tr;
      }, function (P) {
        var lay = { xaxis: { title: { text: "month" }, tickvals: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], ticktext: MONTHS, range: vw === 1 ? [0.5, 12.5] : [s - 0.4, e + 0.4] }, yaxis: { title: { text: "records per month" }, range: [-5, 100] }, margin: { l: 56, r: 12, t: 20, b: 50 } };
        if (vw === 1) { lay.shapes = [{ type: "rect", xref: "x", yref: "paper", x0: 5.5, x1: 7.5, y0: 0, y1: 1, fillcolor: P.sand, opacity: 0.18, line: { width: 0 } }]; lay.annotations = [{ x: 6.5, y: 60, text: "no landings: monsoon closure", showarrow: false, font: { size: 13, color: P.ink2 } }]; }
        return lay;
      });
      var first = ys[0], last = ys[ys.length - 1];
      item(ro, "Window:", MONTHS[s - 1] + " to " + MONTHS[e - 1]); item(ro, "First month:", String(first)); item(ro, "Last month:", String(last));
      item(ro, "Change:", first === 0 ? (last === 0 ? "none" : "from zero, no percentage") : (last >= first ? "+" : "−") + Math.abs(100 * (last - first) / first).toFixed(0) + "%"); item(ro, "Trend line:", (fit.b >= 0 ? "+" : "−") + Math.abs(fit.b).toFixed(1) + " records per month");
      ex_(ex, "What changed.", vw === 0 ? "August to November follows the closure of June and July, when no landings are recorded, so landings recover from 18 to 88. That is a rise of 389% in this window." : vw === 1 ? "The full year shows a season: 74 records in February, a slow fall to 31 in May, the closure, and a recovery to 88 in November. The rise in the published window is the end of the closure, and the year ends near where it began." : "Slide the two month sliders. Find a window that shows landings rising, and another that shows them falling, from the same twelve numbers.");
    }
    function dual(ro, ex) {
      setMode(true); var tt = mo.t, cx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], lo = rmin ? rmin.get() : 24.5, hi = rmax ? rmax.get() : 30;
      if (hi <= lo + 1) hi = lo + 1;
      var tcl = tt.map(function (t) { return t === null ? null : 100 * (t - lo) / (hi - lo); }), cross = [];
      for (var m = 0; m < 11; m++) { if (tcl[m] !== null && tcl[m + 1] !== null) { var d0 = mo.n[m] - tcl[m], d1 = mo.n[m + 1] - tcl[m + 1]; if (d0 * d1 < 0) cross.push(MONTHS[m] + " and " + MONTHS[m + 1]); } }
      var open = mo.n.map(function (n, i) { return i; }).filter(function (i) { return mo.n[i] > 0; }), rr = ols(open.map(function (i) { return tt[i]; }), open.map(function (i) { return mo.n[i]; })).r;
      if (vw === 0) {
        Kit.plot("mg-plot", function (P) { return [
          { type: "scatter", mode: "lines+markers", x: cx, y: mo.n, name: "records per month", line: { color: P.navy, width: 3 }, marker: { size: 8 }, hovertemplate: "%{x}: %{y} records<extra></extra>" },
          { type: "scatter", mode: "lines+markers", x: cx, y: tt, name: "water temperature", yaxis: "y2", line: { color: P.coral, width: 3 }, marker: { size: 8 }, connectgaps: false, hovertemplate: "%{x}: %{y:.1f} C<extra></extra>" }]; },
          function (P) { return { xaxis: { title: { text: "month" }, tickvals: cx, ticktext: MONTHS }, yaxis: { title: { text: "records per month (left axis)" }, range: [0, 100] }, yaxis2: { title: { text: "mean water temperature, degrees C (right axis)", font: { color: P.coral } }, overlaying: "y", side: "right", range: [lo, hi], gridcolor: "rgba(0,0,0,0)", tickfont: { color: P.coral }, linecolor: P.muted, zeroline: false }, margin: { l: 60, r: 64, t: 20, b: 50 }, showlegend: true, legend: { orientation: "h", y: 1.1, x: 0 } }; });
        item(ro, "The two lines cross between:", cross.length ? cross.join("; and between ") : "no months at this setting");
      } else {
        Kit.plot("mg-plot", function (P) { return [
          { type: "scatter", mode: "lines+markers", x: cx, y: mo.n, name: "records per month", line: { color: P.navy, width: 3 }, marker: { size: 8 }, hovertemplate: "%{x}: %{y} records<extra></extra>" },
          { type: "scatter", mode: "lines+markers", x: cx, y: tt, name: "water temperature", xaxis: "x", yaxis: "y2", line: { color: P.coral, width: 3 }, marker: { size: 8 }, hovertemplate: "%{x}: %{y:.1f} C<extra></extra>" }]; },
          function (P) { return { xaxis: { title: { text: "month" }, tickvals: cx, ticktext: MONTHS, anchor: "y2" }, yaxis: { title: { text: "records per month" }, range: [0, 100], domain: [0.56, 1] }, yaxis2: { title: { text: "water, degrees C" }, range: [23, 31], domain: [0, 0.44], gridcolor: P.rule, linecolor: P.muted, tickfont: { color: P.muted }, zerolinecolor: P.rule }, margin: { l: 60, r: 12, t: 12, b: 50 }, showlegend: true, legend: { orientation: "h", y: 1.1, x: 0 }, shapes: [{ type: "rect", xref: "x", yref: "paper", x0: 5.5, x1: 7.5, y0: 0, y1: 1, fillcolor: P.sand, opacity: 0.15, line: { width: 0 } }] }; });
        item(ro, "Closure:", "June and July are shaded: no landings");
      }
      item(ro, "Correlation r over the 10 open months:", Kit.fmt(rr, 2));
      ex_(ex, "What changed.", vw === 0 ? "The two lines use different axes, and the author chooses both ranges. Move the right axis range and the lines cross at another month, or never. A crossing looks like a threshold, but no number changed." : "Two panels give both series their own axis and the same months, so no reader is asked to compare heights across axes. The shaded months are the closure. Temperature and landings both follow the season and the closure, and ten months are too few to say that one drives the other (see 0.7 on correlation).");
    }
    window.addEventListener("resize", Kit.debounce(function () { if (cur === 1 && vw < 2) render(); }));
    Kit.onTheme(function () { if (cur === 1 && vw < 2) render(); });
    show();
  }

  // ======================= 4. confusion matrix =======================
  function matrix(res) {
    var mi = 0, view = 0, sel = null, keys = ["tree3", "boost", "lazy"], names = ["Depth-3 tree", "Boosting, 100 trees of depth 3", "Always spadenose"], classes = res.classes, counts = res.counts;
    var cms = { tree3: res.confusion.tree3, boost: res.confusion.boost, lazy: classes.map(function (c, i) { return classes.map(function (d, j) { return j === 0 ? counts[i] : 0; }); }) };
    Kit.chips(Kit.$("cm-bar"), names, 0, function (i) { mi = i; sel = null; draw(); });
    Kit.chips(Kit.$("cm-view"), ["Counts", "Share of the row (recall on the diagonal)", "Share of the column (precision on the diagonal)"], 0, function (i) { view = i; draw(); });
    function draw() {
      var cm = cms[keys[mi]], n = classes.length, rt = cm.map(function (r) { return r.reduce(function (s, x) { return s + x; }, 0); }), ct = classes.map(function (c, j) { return cm.reduce(function (s, r) { return s + r[j]; }, 0); }), total = rt.reduce(function (s, x) { return s + x; }, 0);
      var tb = Kit.$("cm-table"); clear(tb);
      var hr = H("tr", {}, H("thead", {}, tb)); H("th", { class: "corner" }, hr, "true ↓  predicted →"); classes.forEach(function (c) { H("th", {}, hr, c); });
      var body = H("tbody", {}, tb), correct = 0;
      cm.forEach(function (row, i) {
        var tr = H("tr", {}, body); H("th", { class: "rowh" }, tr, classes[i] + " (" + rt[i] + ")");
        row.forEach(function (v, j) {
          var share = view === 2 ? (ct[j] ? v / ct[j] : 0) : (rt[i] ? v / rt[i] : 0), diag = i === j; if (diag) correct += v;
          var txt = view === 0 ? String(v) : (view === 1 ? (rt[i] ? Math.round(100 * v / rt[i]) + "%" : "") : (ct[j] ? Math.round(100 * v / ct[j]) + "%" : "–"));
          var td = H("td", { class: (diag ? "diag " : "") + (v === 0 ? "zero " : "") + (sel && sel[0] === i && sel[1] === j ? "sel" : ""), tabindex: "0", role: "gridcell", "aria-label": "true " + classes[i] + ", predicted " + classes[j] + ": " + v }, tr, txt);
          td.style.background = "color-mix(in srgb, " + (diag ? "var(--teal)" : "var(--coral)") + " " + Math.round(70 * share) + "%, var(--bg))";
          td.addEventListener("click", function () { sel = [i, j]; draw(); });
        });
      });
      var fr = H("tr", {}, body); H("th", { class: "rowh" }, fr, "column total (" + total + ")"); ct.forEach(function (v) { H("td", { class: "tot" }, fr, String(v)); });
      var eq = Kit.$("cm-eqn"); clear(eq);
      box(eq, "c-teal", String(correct), "on the diagonal"); H("span", { class: "op" }, eq, "÷"); box(eq, "c-navy", String(total), "all records"); H("span", { class: "op" }, eq, "="); box(eq, "c-coral", Kit.pct(correct / total, 1), "accuracy");
      H("span", { class: "op" }, eq, " "); box(eq, "c-teal", String(cm[4][4]), "Ganges sharks found"); H("span", { class: "op" }, eq, "÷"); box(eq, "c-navy", String(rt[4]), "Ganges sharks in all"); H("span", { class: "op" }, eq, "="); box(eq, "c-coral", Kit.pct(cm[4][4] / rt[4], 0), "recall for Ganges");
      var st = Kit.$("cm-status");
      if (!sel) st.textContent = "Click a cell to hear it read aloud.";
      else {
        var i = sel[0], j = sel[1], v = cm[i][j];
        if (i === j) st.textContent = "True " + classes[i] + ", predicted " + classes[j] + ": " + v + " records, the diagonal. That is " + (rt[i] ? Math.round(100 * v / rt[i]) : 0) + "% of the " + rt[i] + " true " + classes[i] + " sharks (recall), and " + (ct[j] ? Math.round(100 * v / ct[j]) : 0) + "% of the " + ct[j] + " records the model labelled " + classes[j] + " (precision).";
        else st.textContent = "In " + v + " records the shark was a " + classes[i] + " and the model said " + classes[j] + ". That is " + (rt[i] ? Math.round(100 * v / rt[i]) : 0) + "% of the " + rt[i] + " true " + classes[i] + " sharks.";
      }
    }
    draw();
  }

  // ======================= 5. results table =======================
  function results(res) {
    var seed = 5, sortKey = null, models = res.models, names = ["seed 4127", "seed 1", "seed 2", "seed 3", "seed 4", "mean of 5"];
    Kit.chips(Kit.$("rt-bar"), names, 5, function (i) { seed = i; draw(); });
    var tb = Kit.$("rt-table");
    function cv(m) { return seed === 5 ? mean(m.cv) : m.cv[seed]; }
    function draw() {
      clear(tb);
      var hr = H("tr", {}, H("thead", {}, tb)); var cols = [["Model", null], ["Train accuracy", "train"], ["Test accuracy (cross-validated)", "cv"], ["Range over 5 seeds", null], ["Ganges sharks found of 3", null]];
      cols.forEach(function (c) { var th = H("th", {}, hr, c[0] + (sortKey === c[1] && c[1] ? " ↓" : "")); if (c[1]) th.addEventListener("click", function () { sortKey = sortKey === c[1] ? null : c[1]; draw(); }); });
      var order = models.slice(); if (sortKey === "train") order.sort(function (a, b) { return b.train - a.train; }); if (sortKey === "cv") order.sort(function (a, b) { return cv(b) - cv(a); });
      var best = Math.max.apply(null, models.map(cv)), body = H("tbody", {});
      tb.appendChild(body);
      order.forEach(function (m) {
        var tr = H("tr", { class: m.id === "lazy" ? "base" : "" }, body);
        H("td", {}, tr, m.name + (m.id === "lazy" ? " (baseline)" : ""));
        H("td", { class: m.train - cv(m) > 0.15 ? "warn" : "" }, tr, Kit.pct(m.train, 1));
        H("td", { class: Math.abs(cv(m) - best) < 1e-9 ? "best" : "" }, tr, Kit.pct(cv(m), 1));
        var td = H("td", {}, tr, Kit.pct(Math.min.apply(null, m.cv), 1) + " to " + Kit.pct(Math.max.apply(null, m.cv), 1));
        var gf = seed === 5 ? m.ganges_found.map(function (x) { return x; }) : [m.ganges_found[seed]];
        H("td", {}, tr, seed === 5 ? Math.min.apply(null, gf) + " to " + Math.max.apply(null, gf) : String(gf[0]));
      });
      var winner = models.filter(function (m) { return Math.abs(cv(m) - best) < 1e-9; })[0];
      Kit.$("rt-status").textContent = (seed === 5 ? "Averaged over five seeds, " : "On " + names[seed] + ", ") + winner.name + " has the highest test accuracy, " + Kit.pct(best, 1) + ". Compare the 'Range' column before you believe the order of close rows. Coral in the train column marks a train score more than 15 points above the test score. In the plot, dots are the five seeds and the diamond is the selected view.";
      Kit.plot("rt-plot", function (P) {
        var C = spCol(P), tr = [], short = { lazy: "Always spadenose", logistic: "Logistic", tree3: "Tree, depth 3", treeFull: "Tree, no limit", boost: "Boosting" }, ys = models.map(function (m) { return short[m.id]; });
        models.forEach(function (m, k) { tr.push({ type: "scatter", mode: "markers", x: m.cv, y: m.cv.map(function () { return short[m.id]; }), marker: { color: P.navy, size: 10, opacity: 0.6 }, hovertemplate: m.name + ": %{x:.3f}<extra></extra>" }); });
        tr.push({ type: "scatter", mode: "markers", x: models.map(cv), y: ys, marker: { color: P.coral, size: 15, symbol: "diamond", line: { color: P.paper, width: 2 } }, hovertemplate: "%{y}: %{x:.3f}<extra></extra>" });
        return tr;
      }, { xaxis: { title: { text: "test accuracy" }, range: [0.4, 0.84], tickformat: ".0%" }, yaxis: { automargin: true, autorange: "reversed" }, margin: { l: 110, r: 12, t: 10, b: 50 } });
      var eq = Kit.$("rt-eqn"); clear(eq);
      var t3 = models.filter(function (m) { return m.id === "tree3"; })[0], bo = models.filter(function (m) { return m.id === "boost"; })[0];
      box(eq, "c-navy", Kit.pct(mean(t3.cv), 1) + " ± " + (100 * sd(t3.cv)).toFixed(1), "depth-3 tree: mean and spread over 5 seeds"); H("span", { class: "op" }, eq, "versus"); box(eq, "c-coral", Kit.pct(mean(bo.cv), 1) + " ± " + (100 * sd(bo.cv)).toFixed(1), "boosting: mean and spread over 5 seeds");
    }
    window.addEventListener("resize", Kit.debounce(draw));
    draw();
  }
})();
