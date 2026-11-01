// F1 Data, fairness and privacy. Six figures: a visitor against the Pima women (mi), a two-group threshold explorer (gp),
// a printed probability under a different base rate (rl), a gender direction in word vectors (bi), a small table that re-identifies
// people (pv) and a count of unique Pima women (pu). Pure helpers are exported with module.exports for node tests.
(function () {
  "use strict";

  // ------------------------------------------------------------------ helpers
  function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }
  function logit(p) { return Math.log(p / (1 - p)); }
  function wilson(k, n, z) {
    z = z || 1.96;
    if (!n) return [0, 1];
    var p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d, h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [Math.max(0, c - h), Math.min(1, c + h)];
  }
  // Confusion counts for the records in idx at a log-odds threshold t (flag when the score is at least t).
  function confusion(z, y, idx, t) {
    var tp = 0, fp = 0, fn = 0, tn = 0;
    for (var k = 0; k < idx.length; k++) {
      var i = idx[k], f = z[i] >= t;
      if (y[i]) { if (f) tp++; else fn++; } else { if (f) fp++; else tn++; }
    }
    return { tp: tp, fp: fp, fn: fn, tn: tn };
  }
  function rates(c) {
    var pos = c.tp + c.fn, neg = c.fp + c.tn, n = pos + neg, fl = c.tp + c.fp;
    return { n: n, pos: pos, neg: neg, flagged: fl, base: n ? pos / n : NaN, sens: pos ? c.tp / pos : NaN, fpr: neg ? c.fp / neg : NaN, ppv: fl ? c.tp / fl : NaN, tp: c.tp, fp: c.fp, fn: c.fn, tn: c.tn };
  }
  // ROC curve of the records in idx: arrays of false positive rate and sensitivity, from (0,0) to (1,1).
  function roc(z, y, idx) {
    var order = idx.slice().sort(function (a, b) { return z[b] - z[a]; });
    var pos = 0, neg = 0, k;
    for (k = 0; k < idx.length; k++) { if (y[idx[k]]) pos++; else neg++; }
    var fx = [0], ty = [0], tp = 0, fp = 0;
    for (k = 0; k < order.length; k++) {
      if (y[order[k]]) tp++; else fp++;
      if (k === order.length - 1 || z[order[k + 1]] !== z[order[k]]) { fx.push(fp / neg); ty.push(tp / pos); }
    }
    return { fpr: fx, tpr: ty };
  }
  function auc(c) { var a = 0; for (var i = 1; i < c.fpr.length; i++) a += (c.fpr[i] - c.fpr[i - 1]) * (c.tpr[i] + c.tpr[i - 1]) / 2; return a; }
  // Share of a metric closest to a target: choose the threshold (probability, 0.05 to 0.95) for group idx that brings metric(r) nearest to target.
  function matchThreshold(z, y, idx, metric, target) {
    var best = 0.5, bd = Infinity;
    for (var s = 5; s <= 95; s++) {
      var t = s / 100, r = rates(confusion(z, y, idx, logit(t)));
      if (metric === "ppv" && r.flagged < 5) continue;
      var v = r[metric];
      if (isNaN(v)) continue;
      var d = Math.abs(v - target);
      if (d < bd - 1e-12) { bd = d; best = t; }
    }
    return best;
  }
  // Prior shift: the same evidence in a place where the base rate is pNew instead of pOld.
  function shiftLogit(z, pOld, pNew) { return z + logit(pNew) - logit(pOld); }
  function pct(v, d) { return isNaN(v) ? "n/a" : (100 * v).toFixed(d == null ? 0 : d) + "%"; }
  function hexA(hex, a) {
    var h = hex.replace("#", ""); if (h.length === 3) h = h.replace(/(.)/g, "$1$1");
    return "rgba(" + parseInt(h.slice(0, 2), 16) + "," + parseInt(h.slice(2, 4), 16) + "," + parseInt(h.slice(4, 6), 16) + "," + a + ")";
  }
  function el(id) { return document.getElementById(id); }
  function row(parent, cls) { var d = document.createElement("div"); if (cls) d.className = cls; parent.appendChild(d); return d; }

  // the model of the visitor figure: standardise, weight, add the intercept
  function modelLogit(M, x) {
    var z = M.intercept;
    for (var j = 0; j < M.coef.length; j++) z += M.coef[j] * (x[j] - M.mean[j]) / M.std[j];
    return z;
  }

  var ROWS = [
    ["Asha", 34, "F", "Vashi", "teacher", 0], ["Bilal", 52, "M", "Nerul", "driver", 1], ["Chitra", 29, "F", "Vashi", "nurse", 0], ["Deepak", 61, "M", "Belapur", "shopkeeper", 1],
    ["Esha", 24, "F", "Kharghar", "student", 1], ["Farhan", 47, "M", "Juhu", "fisher", 1], ["Gauri", 38, "F", "Versova", "tailor", 0], ["Hari", 33, "M", "Andheri", "clerk", 1],
    ["Isha", 41, "F", "Dadar", "teacher", 0], ["Jatin", 36, "M", "Sion", "driver", 0], ["Kavita", 58, "F", "Dadar", "shopkeeper", 1], ["Lata", 66, "F", "Sion", "retired", 0],
    ["Manoj", 44, "M", "Nerul", "clerk", 0], ["Neha", 31, "F", "Belapur", "nurse", 0], ["Omkar", 27, "M", "Kharghar", "student", 0], ["Pooja", 35, "F", "Vashi", "tailor", 1],
    ["Qasim", 49, "M", "Versova", "fisher", 1], ["Rekha", 53, "F", "Juhu", "tailor", 0], ["Sagar", 39, "M", "Andheri", "driver", 0], ["Tara", 30, "F", "Dadar", "clerk", 0],
    ["Uday", 72, "M", "Sion", "retired", 1], ["Vimal", 55, "M", "Belapur", "driver", 1], ["Yash", 22, "M", "Vashi", "student", 0], ["Zoya", 83, "F", "Juhu", "retired", 1]
  ];
  var ZONE = { Vashi: "Navi Mumbai", Nerul: "Navi Mumbai", Belapur: "Navi Mumbai", Kharghar: "Navi Mumbai", Juhu: "Western Mumbai", Versova: "Western Mumbai", Andheri: "Western Mumbai", Dadar: "Central Mumbai", Sion: "Central Mumbai" };
  function ageLabel(a, band) { if (band === 1) return String(a); var lo = Math.floor(a / band) * band; return lo + " to " + (lo + band - 1); }
  function areaLabel(a, lvl) { return lvl === 0 ? a : lvl === 1 ? ZONE[a] : "Mumbai region"; }
  function groupKey(r, c) {
    var k = [];
    if (c.age) k.push(ageLabel(r[1], c.ageBand)); if (c.sex) k.push(r[2]); if (c.area) k.push(areaLabel(r[3], c.areaLvl)); if (c.job) k.push(r[4]);
    return k.join("|");
  }
  function releaseStats(rows, c) {
    var groups = {}, i;
    rows.forEach(function (r) { var k = groupKey(r, c); (groups[k] = groups[k] || []).push(r); });
    var unique = 0, revealed = 0, kmin = Infinity;
    rows.forEach(function (r) {
      var g = groups[groupKey(r, c)]; if (g.length === 1) unique++;
      if (g.length < kmin) kmin = g.length;
      if (g.every(function (q) { return q[5] === g[0][5]; })) revealed++;
    });
    return { groups: groups, unique: unique, k: kmin, revealed: revealed, n: rows.length, nGroups: Object.keys(groups).length };
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { sigmoid: sigmoid, logit: logit, wilson: wilson, confusion: confusion, rates: rates, roc: roc, auc: auc, matchThreshold: matchThreshold, shiftLogit: shiftLogit, modelLogit: modelLogit, releaseStats: releaseStats, ROWS: ROWS };
  }
  if (typeof document === "undefined") return;

  var CL = null, BI = null;
  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/clinic.json"), Kit.data("data/bias.json")]).then(function (r) {
      CL = r[0]; BI = r[1];
      visitorFig(); groupFig(); reliabilityFig(); biasFig(); privacyFig(); uniqueFig();
    }).catch(function (e) { console.error(e); var s = document.createElement("p"); s.className = "status"; s.textContent = "The data for these figures did not load."; document.querySelector("main").appendChild(s); });
  });

  // ================================================================== mi: a visitor against the Pima women
  function visitorFig() {
    var n = CL.y.length, M = CL.model, pi0 = CL.y.reduce(function (a, b) { return a + b; }, 0) / n;
    var typical = { preg: M.median[0], bp: M.median[2], dpf: M.median[4] };
    var PRESETS = [
      { label: "Woman A (Pima, 28)", age: 28, bmi: 31.6, glu: 158 },
      { label: "Visitor 1 (made up)", age: 45, bmi: 24, glu: 125 },
      { label: "Visitor 2 (made up)", age: 62, bmi: 22.5, glu: 140 },
      { label: "Visitor 3 (made up)", age: 24, bmi: 40, glu: 100 }
    ];
    var bar = el("mi-bar"), ctl = el("mi-controls"), ctl2 = el("mi-controls2");
    var sAge = Kit.slider(ctl, { label: "Age, years", min: 21, max: 81, step: 1, value: 45 }, draw);
    var sBmi = Kit.slider(ctl, { label: "BMI, kg per square metre", min: 15, max: 50, step: 0.5, value: 24, digits: 1 }, draw);
    var sGlu = Kit.slider(ctl, { label: "Glucose, mg/dl", min: 60, max: 200, step: 1, value: 125 }, draw);
    var sPi = Kit.slider(ctl2, { label: "Share with diabetes where the model is used", min: 0.02, max: 0.6, step: 0.001, value: pi0, show: function (v) { return Kit.pct(v, 1); } }, draw);
    var sWa = Kit.slider(ctl2, { label: "Neighbourhood: age within, years", min: 2, max: 10, step: 1, value: 5 }, draw);
    var sWb = Kit.slider(ctl2, { label: "Neighbourhood: BMI within", min: 1, max: 6, step: 0.5, value: 3, digits: 1 }, draw);
    Kit.chips(bar, PRESETS.map(function (p) { return p.label; }), 1, function (i) { var p = PRESETS[i]; sAge.set(p.age); sBmi.set(p.bmi); sGlu.set(p.glu); draw(); });
    var bmiSorted = CL.bmi.filter(function (v) { return v != null; }).sort(function (a, b) { return a - b; });
    function draw() {
      var age = sAge.get(), bmi = sBmi.get(), glu = sGlu.get(), pi = sPi.get(), wa = sWa.get(), wb = sWb.get();
      var z = modelLogit(M, [typical.preg, glu, typical.bp, bmi, typical.dpf, age]), p = sigmoid(z);
      var zs = shiftLogit(z, pi0, pi), ps = sigmoid(zs);
      var nS = 0, pos = 0;
      for (var i = 0; i < n; i++) {
        if (CL.bmi[i] == null) continue;
        if (Math.abs(CL.age[i] - age) <= wa && Math.abs(CL.bmi[i] - bmi) <= wb) { nS++; pos += CL.y[i]; }
      }
      var below = 0; while (below < bmiSorted.length && bmiSorted[below] < bmi) below++;
      Kit.plot("mi-plot", function (P) {
        var xs0 = [], ys0 = [], xs1 = [], ys1 = [];
        for (var i = 0; i < n; i++) { if (CL.bmi[i] == null) continue; if (CL.y[i]) { xs1.push(CL.age[i]); ys1.push(CL.bmi[i]); } else { xs0.push(CL.age[i]); ys0.push(CL.bmi[i]); } }
        return [
          { x: xs0, y: ys0, mode: "markers", marker: { color: P.navy, size: 6, opacity: 0.5 }, hoverinfo: "skip", name: "no diabetes" },
          { x: xs1, y: ys1, mode: "markers", marker: { color: P.coral, size: 6, symbol: "diamond", opacity: 0.6 }, hoverinfo: "skip", name: "diabetes" },
          { x: [age], y: [bmi], mode: "markers+text", text: ["visitor"], textposition: "top right", textfont: { color: P.ink, size: 14 }, marker: { color: P.teal, size: 17, symbol: "star", line: { color: P.ink, width: 1.5 } }, hovertemplate: "visitor: age %{x}, BMI %{y}<extra></extra>" }
        ];
      }, function (P) {
        return {
          xaxis: { title: { text: "age, years" }, range: [19, 83] }, yaxis: { title: { text: "BMI" }, range: [14, 52] },
          shapes: [
            { type: "rect", x0: age - wa, x1: age + wa, y0: bmi - wb, y1: bmi + wb, line: { color: P.teal, width: 2 }, fillcolor: hexA(P.teal, 0.12) },
            { type: "line", x0: 19, x1: 83, y0: 23, y1: 23, line: { color: P.sand, width: 1.5, dash: "dot" } },
            { type: "line", x0: 19, x1: 83, y0: 27.5, y1: 27.5, line: { color: P.sand, width: 1.5, dash: "dot" } }
          ],
          annotations: [
            { x: 83, y: 23, xanchor: "right", yanchor: "top", text: "WHO 23", showarrow: false, font: { size: 12, color: P.muted } },
            { x: 83, y: 27.5, xanchor: "right", yanchor: "bottom", text: "WHO 27.5", showarrow: false, font: { size: 12, color: P.muted } }
          ],
          margin: { l: 52, r: 12, t: 12, b: 46 }
        };
      });
      Kit.setText("mi-p", Kit.pct(p, 0));
      Kit.setText("mi-ps", Kit.pct(ps, 0));
      Kit.setText("mi-n", String(nS));
      Kit.setText("mi-share", nS ? Kit.pct(pos / nS, 0) : "none");
      Kit.setText("mi-eq-z", Kit.fmt(z, 2)); Kit.setText("mi-eq-old", Kit.fmt(logit(pi0), 2)); Kit.setText("mi-eq-new", Kit.fmt(logit(pi), 2));
      Kit.setText("mi-eq-zs", Kit.fmt(zs, 2)); Kit.setText("mi-eq-ps", Kit.pct(ps, 0)); Kit.setText("mi-eq-p", Kit.pct(p, 0));
      var low = bmiSorted.length ? Math.round(100 * below / bmiSorted.length) : 0;
      var msg = "In the data, " + low + "% of the women with a measured BMI are below " + Kit.fmt(bmi, 1) + ". ";
      if (nS === 0) msg += "No woman lies in the neighbourhood, so the model has no similar women to compare this visitor with.";
      else if (nS < 20) msg += "Only " + nS + " women lie in the neighbourhood: treat the score as a guess.";
      else msg += nS + " women lie in the neighbourhood, enough to compare.";
      Kit.setText("mi-status", msg);
    }
    draw();
  }

  // ================================================================== gp: two groups, one threshold each
  function groupFig() {
    var n = CL.y.length;
    var SPLIT = [
      { name: "age", a: "under 30", b: "30 and over", f: function (i) { return CL.age[i] < 30 ? 0 : 1; } },
      { name: "BMI", a: "BMI under 30", b: "BMI 30 and over", f: function (i) { return (CL.bmi[i] == null ? 32.3 : CL.bmi[i]) < 30 ? 0 : 1; } },
      { name: "pregnancies", a: "0 to 2 pregnancies", b: "3 or more", f: function (i) { return CL.preg[i] <= 2 ? 0 : 1; } }
    ];
    var MODES = ["Free", "Same threshold", "Match sensitivity", "Match precision", "Match false positive rate"];
    var st = { split: 0, sees: true, mode: 0, tA: 0.5, tB: 0.5 };
    var idxA, idxB, z, rocA, rocB;
    function setup() {
      z = st.sees ? CL.zcv : CL.zna; idxA = []; idxB = [];
      for (var i = 0; i < n; i++) (SPLIT[st.split].f(i) ? idxB : idxA).push(i);
      rocA = roc(z, CL.y, idxA); rocB = roc(z, CL.y, idxB);
    }
    var bar = el("gp-bar"), bar2 = el("gp-bar2"), ctl = el("gp-controls");
    Kit.html("span", { class: "label" }, bar, "Split the women by");
    Kit.chips(bar, ["Age", "BMI", "Pregnancies"], 0, function (i) { st.split = i; apply(); });
    var lab = Kit.html("label", { class: "chk" }, bar); var cb = Kit.html("input", { type: "checkbox" }, lab); cb.checked = true; lab.appendChild(document.createTextNode(" the model sees age"));
    cb.addEventListener("change", function () { st.sees = cb.checked; apply(); });
    Kit.html("span", { class: "label" }, bar2, "Threshold for group B");
    var modeBtns = Kit.chips(bar2, MODES, 0, function (i) { st.mode = i; apply(); });
    var sA = Kit.slider(ctl, { label: "Threshold for group A", min: 0.05, max: 0.95, step: 0.01, value: 0.5, show: function (v) { return v.toFixed(2); } }, function (v) { st.tA = v; apply(); });
    var sB = Kit.slider(ctl, { label: "Threshold for group B", min: 0.05, max: 0.95, step: 0.01, value: 0.5, show: function (v) { return v.toFixed(2); } }, function (v) { st.tB = v; apply(); });
    function apply() {
      setup();
      var cA = rates(confusion(z, CL.y, idxA, logit(st.tA)));
      if (st.mode === 1) st.tB = st.tA;
      else if (st.mode === 2) st.tB = matchThreshold(z, CL.y, idxB, "sens", cA.sens);
      else if (st.mode === 3) st.tB = matchThreshold(z, CL.y, idxB, "ppv", isNaN(cA.ppv) ? cA.base : cA.ppv);
      else if (st.mode === 4) st.tB = matchThreshold(z, CL.y, idxB, "fpr", cA.fpr);
      sB.set(st.tB); sB.input.disabled = st.mode !== 0;
      draw(cA, rates(confusion(z, CL.y, idxB, logit(st.tB))));
    }
    function gapCell(a, b) {
      var g = 100 * (b - a), cls = Math.abs(g) >= 5 ? " class='gap'" : "";
      return "<td" + cls + ">" + (isNaN(g) ? "n/a" : (g > 0.5 ? "+" : g < -0.5 ? "−" : "") + Math.abs(g).toFixed(0)) + "</td>";
    }
    function ci(c, k, m) { var w = wilson(k, m); return " <span class='cell-note'>(" + Math.round(100 * w[0]) + " to " + Math.round(100 * w[1]) + ")</span>"; }
    function draw(A, B) {
      var S = SPLIT[st.split];
      var rowsHtml = [
        ["Women", String(A.n), String(B.n), "<td></td>"],
        ["Base rate", pct(A.base), pct(B.base), gapCell(A.base, B.base)],
        ["Flagged", pct(A.flagged / A.n), pct(B.flagged / B.n), gapCell(A.flagged / A.n, B.flagged / B.n)],
        ["Sensitivity", pct(A.sens) + ci(A, A.tp, A.pos), pct(B.sens) + ci(B, B.tp, B.pos), gapCell(A.sens, B.sens)],
        ["Precision", pct(A.ppv), pct(B.ppv), gapCell(A.ppv, B.ppv)],
        ["False positive rate", pct(A.fpr), pct(B.fpr), gapCell(A.fpr, B.fpr)]
      ].map(function (r) { return "<tr><td>" + r[0] + "</td><td>" + r[1] + "</td><td>" + r[2] + "</td>" + r[3] + "</tr>"; }).join("");
      el("gp-table").innerHTML = "<thead><tr><th></th><th>A: " + S.a + "</th><th>B: " + S.b + "</th><th>B − A, points</th></tr></thead><tbody>" + rowsHtml + "</tbody>";
      // ROC plot with the two curves, two operating points and lines of equal precision
      var T = isNaN(A.ppv) ? A.base : A.ppv;
      function ray(base) { var k = (1 - base) * T / (base * (1 - T)); return k > 0 && isFinite(k) ? { x: [0, Math.min(1, 1 / k)], y: [0, Math.min(1, k)] } : { x: [0, 0], y: [0, 0] }; }
      var rayA = ray(A.base), rayB = ray(B.base);
      Kit.plot("gp-plot", function (P) {
        return [
          { x: rocA.fpr, y: rocA.tpr, mode: "lines", line: { color: P.navy, width: 2.5 }, hoverinfo: "skip", name: "A" },
          { x: rocB.fpr, y: rocB.tpr, mode: "lines", line: { color: P.coral, width: 2.5 }, hoverinfo: "skip", name: "B" },
          { x: rayA.x, y: rayA.y, mode: "lines", line: { color: P.navy, width: 1.5, dash: "dash" }, hoverinfo: "skip" },
          { x: rayB.x, y: rayB.y, mode: "lines", line: { color: P.coral, width: 1.5, dash: "dash" }, hoverinfo: "skip" },
          { x: [A.fpr], y: [A.sens], mode: "markers+text", text: ["A"], textposition: "top left", textfont: { color: P.navy, size: 15 }, marker: { color: P.navy, size: 15, line: { color: P.bg, width: 2 } }, hovertemplate: "A: FPR %{x:.2f}, sensitivity %{y:.2f}<extra></extra>" },
          { x: [B.fpr], y: [B.sens], mode: "markers+text", text: ["B"], textposition: "bottom right", textfont: { color: P.coral, size: 15 }, marker: { color: P.coral, size: 15, symbol: "diamond", line: { color: P.bg, width: 2 } }, hovertemplate: "B: FPR %{x:.2f}, sensitivity %{y:.2f}<extra></extra>" }
        ];
      }, function (P) {
        return {
          xaxis: { title: { text: "false positive rate" }, range: [0, 1] }, yaxis: { title: { text: "sensitivity" }, range: [0, 1.02], scaleanchor: "x", scaleratio: 1 },
          annotations: [
            { x: 0.98, xref: "paper", y: 0.16, yref: "paper", xanchor: "right", yanchor: "bottom", text: "dashed, A: precision " + (isNaN(A.ppv) ? "n/a" : pct(T)), showarrow: false, font: { size: 12, color: P.navy } },
            { x: 0.98, xref: "paper", y: 0.06, yref: "paper", xanchor: "right", yanchor: "bottom", text: "dashed, B: the same precision", showarrow: false, font: { size: 12, color: P.coral } }
          ],
          margin: { l: 52, r: 12, t: 12, b: 46 }
        };
      });
      // the identity FPR = p/(1-p) * (1-PPV)/PPV * sensitivity, group by group
      function eqRow(id, R, cls) {
        var odds = R.base / (1 - R.base), pr = isNaN(R.ppv) ? NaN : (1 - R.ppv) / R.ppv;
        el(id).innerHTML = "<span class='t " + cls + "'><b>" + Kit.fmt(odds, 2) + "</b><i>odds of diabetes, p/(1−p)</i></span><span class='op'>&times;</span>" +
          "<span class='t " + cls + "'><b>" + (isNaN(pr) ? "n/a" : Kit.fmt(pr, 2)) + "</b><i>(1−precision)/precision</i></span><span class='op'>&times;</span>" +
          "<span class='t " + cls + "'><b>" + Kit.fmt(R.sens, 2) + "</b><i>sensitivity</i></span><span class='op'>=</span>" +
          "<span class='t " + cls + "'><b>" + Kit.fmt(odds * pr * R.sens, 3) + "</b><i>false positive rate</i></span>" +
          "<span class='op'>(measured " + Kit.fmt(R.fpr, 3) + ")</span>";
      }
      eqRow("gp-eqA", A, "c-navy"); eqRow("gp-eqB", B, "c-coral");
      var oddsR = (B.base / (1 - B.base)) / (A.base / (1 - A.base));
      var precR = ((1 - B.ppv) / B.ppv) / ((1 - A.ppv) / A.ppv), sensR = B.sens / A.sens, fprR = B.fpr / A.fpr;
      el("gp-eqR").innerHTML = "<span class='t c-sand'><b>" + Kit.fmt(oddsR, 2) + "</b><i>odds of B over odds of A</i></span><span class='op'>&times;</span>" +
        "<span class='t c-sand'><b>" + (isFinite(precR) ? Kit.fmt(precR, 2) : "n/a") + "</b><i>precision term, B over A</i></span><span class='op'>&times;</span>" +
        "<span class='t c-sand'><b>" + (isFinite(sensR) ? Kit.fmt(sensR, 2) : "n/a") + "</b><i>sensitivity, B over A</i></span><span class='op'>=</span>" +
        "<span class='t c-teal'><b>" + (isFinite(fprR) ? Kit.fmt(fprR, 2) : "n/a") + "</b><i>false positive rate, B over A</i></span>";
      var eq = function (a, b) { return Math.abs(a - b) < 0.015; };
      var parts = [];
      parts.push("Sensitivity " + (eq(A.sens, B.sens) ? "is matched" : "differs by " + Math.abs(Math.round(100 * (B.sens - A.sens))) + " points"));
      parts.push("precision " + (isNaN(A.ppv) || isNaN(B.ppv) ? "cannot be computed for a group that flags nobody" : eq(A.ppv, B.ppv) ? "is matched" : "differs by " + Math.abs(Math.round(100 * (B.ppv - A.ppv))) + " points"));
      parts.push("the false positive rate " + (eq(A.fpr, B.fpr) ? "is matched" : "differs by " + Math.abs(Math.round(100 * (B.fpr - A.fpr))) + " points"));
      Kit.setText("gp-status", parts.join(", ") + ". Group B has " + (B.base > A.base ? "the higher" : "the lower") + " base rate (" + pct(B.base) + " against " + pct(A.base) + ").");
      el("gp-aucs").textContent = "AUC: group A " + auc(rocA).toFixed(2) + ", group B " + auc(rocB).toFixed(2);
    }
    apply();
  }

  // ================================================================== rl: a printed probability in a place with another base rate
  function reliabilityFig() {
    var n = CL.y.length, pi0 = CL.y.reduce(function (a, b) { return a + b; }, 0) / n;
    var PRE = [["Pima sample, 34.9%", pi0], ["20% (made up)", 0.2], ["India survey, 11.4% (adults)", 0.114], ["5% (made up)", 0.05]];
    var bar = el("rl-bar"), ctl = el("rl-controls");
    Kit.html("span", { class: "label" }, bar, "Share of women with diabetes where the model is used");
    var sPi = Kit.slider(ctl, { label: "Share with diabetes where used", min: 0.02, max: 0.6, step: 0.001, value: pi0, show: function (v) { return Kit.pct(v, 1); } }, draw);
    var sP = Kit.slider(ctl, { label: "Chance printed by the model", min: 0.05, max: 0.95, step: 0.01, value: 0.7, show: function (v) { return Kit.pct(v, 0); } }, draw);
    Kit.chips(bar, PRE.map(function (p) { return p[0]; }), 0, function (i) { sPi.set(PRE[i][1]); draw(); });
    var prob = CL.zcv.map(sigmoid);
    function bins(pi) {
      var sw = new Array(10).fill(0), swy = new Array(10).fill(0), cnt = new Array(10).fill(0);
      for (var i = 0; i < n; i++) {
        var b = Math.min(9, Math.floor(prob[i] * 10)), w = CL.y[i] ? pi / pi0 : (1 - pi) / (1 - pi0);
        sw[b] += w; swy[b] += w * CL.y[i]; cnt[b]++;
      }
      return { obs: swy.map(function (v, k) { return sw[k] > 0 ? v / sw[k] : null; }), cnt: cnt, wt: sw };
    }
    function draw() {
      var pi = sPi.get(), p = sP.get(), B0 = bins(pi0), B1 = bins(pi);
      var xs = [], k;
      for (k = 0; k < 10; k++) xs.push(0.05 + 0.1 * k);
      Kit.plot("rl-plot", function (P) {
        var sz0 = B0.cnt.map(function (c) { return 6 + 1.4 * Math.sqrt(c); });
        var sz1 = B1.wt.map(function (c) { return 6 + 1.4 * Math.sqrt(c); });
        return [
          { x: [0, 1], y: [0, 1], mode: "lines", line: { color: P.muted, width: 1.5, dash: "dot" }, hoverinfo: "skip" },
          { x: xs, y: B1.obs, mode: "lines+markers", line: { color: P.coral, width: 2.5 }, marker: { color: P.coral, size: sz1, symbol: "diamond" }, hovertemplate: "printed %{x:.0%}: %{y:.0%} would have diabetes (new share)<extra></extra>" },
          { x: xs, y: B0.obs, mode: "lines+markers", line: { color: P.navy, width: 2.5 }, marker: { color: P.navy, size: sz0 }, hovertemplate: "printed %{x:.0%}: %{y:.0%} had diabetes (Pima)<extra></extra>" },
          { x: [p, p], y: [0, 1], mode: "lines", line: { color: P.teal, width: 2 }, hoverinfo: "skip" }
        ];
      }, function (P) {
        return {
          xaxis: { title: { text: "chance printed by the model" }, range: [0, 1], tickformat: ".0%" }, yaxis: { title: { text: "share who have diabetes" }, range: [0, 1.02], tickformat: ".0%" },
          annotations: [
            { x: 0.97, y: 0.12, xanchor: "right", text: "Pima women", showarrow: false, font: { color: P.navy, size: 13 } },
            { x: 0.97, y: 0.04, xanchor: "right", text: "same women at " + Kit.pct(pi, 1), showarrow: false, font: { color: P.coral, size: 13 } }
          ],
          margin: { l: 56, r: 12, t: 12, b: 46 }
        };
      });
      var b = Math.min(9, Math.floor(p * 10)), zs = shiftLogit(logit(p), pi0, pi);
      Kit.setText("rl-eq-p", Kit.pct(p, 0)); Kit.setText("rl-eq-z", Kit.fmt(logit(p), 2)); Kit.setText("rl-eq-shift", Kit.fmt(logit(pi) - logit(pi0), 2));
      Kit.setText("rl-eq-zs", Kit.fmt(zs, 2)); Kit.setText("rl-eq-ps", Kit.pct(sigmoid(zs), 0));
      Kit.setText("rl-status", "Among Pima women whose printed chance fell between " + Kit.pct(b / 10, 0) + " and " + Kit.pct((b + 1) / 10, 0) + " (" + B0.cnt[b] + " women), " + Kit.pct(B0.obs[b], 0) + " had diabetes. Reweighted so that " + Kit.pct(pi, 1) + " of all the women have diabetes, " + Kit.pct(B1.obs[b], 0) + " of this group do.");
    }
    draw();
  }

  // ================================================================== bi: a gender direction in word vectors
  function biasFig() {
    var dirs = BI.directions, occ = BI.occupations, cur = 2, removed = false, band = BI.null_sd, iN = occ.indexOf("nurse"), iD = occ.indexOf("doctor");
    var bar = el("bi-bar"), bar2 = el("bi-bar2");
    Kit.html("span", { class: "label" }, bar, "Direction built from");
    Kit.chips(bar, dirs, 2, function (i) { cur = i; draw(); });
    var lab = Kit.html("label", { class: "chk" }, bar2); var cb = Kit.html("input", { type: "checkbox" }, lab);
    lab.appendChild(document.createTextNode(" remove the direction from all the vectors"));
    cb.addEventListener("change", function () { removed = cb.checked; draw(); });
    Kit.html("span", { class: "label", style: "margin-left:14px" }, bar2, "Look inside");
    var nearBox = Kit.html("span", {}, bar2);
    Kit.chips(nearBox, ["nurse", "doctor", "secretary", "manager", "architect"], 0, function (i) { showNear(["nurse", "doctor", "secretary", "manager", "architect"][i]); });
    function showNear(w) {
      var l = BI.near[w].map(function (a) { return a[0] + " (" + a[1].toFixed(2) + ")"; }).join(", ");
      Kit.setText("bi-near", "Nearest words to “" + w + "” among all " + BI.vocab.toLocaleString("en-US") + ": " + l + ".");
    }
    showNear("nurse");
    function draw() {
      var sc = BI.scores[dirs[cur]], ord = occ.map(function (w, i) { return i; }).sort(function (a, b) { return sc[a] - sc[b]; });
      var names = ord.map(function (i) { return occ[i]; }), xs = ord.map(function (i) { return removed ? 0 : sc[i]; });
      var out = ord.filter(function (i) { return Math.abs(sc[i]) > band; }).length;
      Kit.plot("bi-plot", function (P) {
        var colors = ord.map(function (i) { return sc[i] < 0 ? P.coral : P.navy; });
        var fills = ord.map(function (i, k) { return Math.abs(sc[i]) > band && !removed ? colors[k] : "rgba(0,0,0,0)"; });
        return [{ x: xs, y: names, mode: "markers", marker: { size: 11, color: fills, line: { color: colors, width: 2 }, symbol: ord.map(function (i) { return sc[i] < 0 ? "diamond" : "circle"; }) },
          hovertemplate: "%{y}: %{customdata:.2f}<extra></extra>", customdata: ord.map(function (i) { return sc[i]; }) }];
      }, function (P) {
        return {
          xaxis: { title: { text: removed ? "score after removing the direction" : Kit.width(el("bi-plot")) < 520 ? "leans to she (left) or he (right)" : "leans towards she (left) or he (right)" }, range: [-0.5, 0.5], zeroline: true },
          yaxis: { categoryorder: "array", categoryarray: names, autorange: "reversed", tickfont: { size: 12 }, automargin: true, gridcolor: "rgba(0,0,0,0)" },
          shapes: [{ type: "rect", xref: "x", yref: "paper", x0: -band, x1: band, y0: 0, y1: 1, fillcolor: hexA(P.sand, 0.22), line: { width: 0 }, layer: "below" }],
          annotations: [{ x: 0, y: 1.0, yref: "paper", yanchor: "bottom", text: Kit.width(el("bi-plot")) < 520 ? "grey band: random directions" : "inside the band: no clearer than a random direction", showarrow: false, font: { size: 12, color: P.muted } }],
          margin: { l: 90, r: 12, t: 26, b: 46 }
        };
      });
      Kit.setText("bi-out", removed ? "0 of " + occ.length : out + " of " + occ.length);
      Kit.setText("bi-nd", Kit.fmt(removed ? 0 : sc[iD] - sc[iN], 2));
      Kit.setText("bi-rho", Kit.fmt(BI.rank_corr[0][cur], 2));
      Kit.setText("bi-status", removed ? "All scores are now 0 by construction. After the ten-pair direction is removed, a linear classifier can still tell which side the 1,000 most gendered words came from, with " + Kit.pct(BI.debias.acc_after, 0) + " accuracy (chance is 50%)." :
        out + " of the " + occ.length + " occupations lie outside the grey band. The band is one standard deviation of a single word's score over 2,000 random directions (" + band.toFixed(3) + ").");
      el("bi-v1").style.width = (100 * (removed ? BI.debias.acc_after : BI.debias.acc_before)) + "%"; Kit.setText("bi-v1n", Kit.pct(removed ? BI.debias.acc_after : BI.debias.acc_before, 0));
    }
    // worked example from the shipped vectors: score of nurse on the direction he - she
    (function () {
      var V = BI.vec, he = V.he, she = V.she, nu = V.nurse, dc = V.doctor;
      var d = he.map(function (v, i) { return v - she[i]; });
      var nd = Math.sqrt(d.reduce(function (a, v) { return a + v * v; }, 0));
      function cosd(w) { var dot = 0, nw = 0; for (var i = 0; i < w.length; i++) { dot += w[i] * d[i]; nw += w[i] * w[i]; } return { dot: dot, nw: Math.sqrt(nw), cos: dot / (Math.sqrt(nw) * nd) }; }
      var a = cosd(nu), b = cosd(dc);
      Kit.setText("bi-eq-d", Kit.fmt(nd, 2)); Kit.setText("bi-eq-dot", Kit.fmt(a.dot, 2)); Kit.setText("bi-eq-nw", Kit.fmt(a.nw, 2)); Kit.setText("bi-eq-cos", Kit.fmt(a.cos, 2));
      Kit.setText("bi-eq-dcos", Kit.fmt(b.cos, 2));
    })();
    draw();
  }

  // ================================================================== pv: a small table that points to people (made-up rows)
  function privacyFig() {
    var c = { age: true, ageBand: 1, sex: true, area: true, areaLvl: 0, job: false }, names = true, pick = -1;
    var ctl = el("pv-controls"), bar = el("pv-bar");
    function chk(label, key, on) {
      var l = Kit.html("label", { class: "chk" }, bar), i = Kit.html("input", { type: "checkbox" }, l); i.checked = on;
      l.appendChild(document.createTextNode(" " + label)); i.addEventListener("change", function () { c[key] = i.checked; draw(); }); return i;
    }
    Kit.html("span", { class: "label" }, bar, "Release these columns");
    chk("age", "age", true); chk("sex", "sex", true); chk("area", "area", true); chk("job", "job", false);
    var nl = Kit.html("label", { class: "chk" }, bar), ni = Kit.html("input", { type: "checkbox" }, nl); ni.checked = true; nl.appendChild(document.createTextNode(" names removed"));
    ni.addEventListener("change", function () { names = ni.checked; draw(); });
    Kit.select(ctl, "Age as", [{ value: "1", label: "single years" }, { value: "5", label: "5-year bands" }, { value: "10", label: "10-year bands" }, { value: "20", label: "20-year bands" }], "1", function (v) { c.ageBand = Number(v); draw(); });
    Kit.select(ctl, "Area as", [{ value: "0", label: "neighbourhood" }, { value: "1", label: "zone (3 zones)" }, { value: "2", label: "whole region" }], "0", function (v) { c.areaLvl = Number(v); draw(); });
    function draw() {
      var S = releaseStats(ROWS, c), tb = el("pv-table"), html = "<thead><tr><th>Name</th>";
      if (c.age) html += "<th>Age</th>"; if (c.sex) html += "<th>Sex</th>"; if (c.area) html += "<th>Area</th>"; if (c.job) html += "<th>Job</th>";
      html += "<th>Group size</th><th>Diabetes</th></tr></thead><tbody>";
      var pickKey = pick >= 0 ? groupKey(ROWS[pick], c) : null;
      ROWS.forEach(function (r, i) {
        var g = S.groups[groupKey(r, c)], cls = [];
        if (g.length === 1) cls.push("uniq"); else if (g.every(function (q) { return q[5] === g[0][5]; })) cls.push("homo");
        if (pickKey !== null && groupKey(r, c) === pickKey) cls.push("pick");
        html += "<tr data-i='" + i + "' class='" + cls.join(" ") + "' tabindex='0'><td>" + (names ? "<s>" + r[0] + "</s>" : r[0]) + "</td>";
        if (c.age) html += "<td>" + ageLabel(r[1], c.ageBand) + "</td>"; if (c.sex) html += "<td>" + r[2] + "</td>"; if (c.area) html += "<td>" + areaLabel(r[3], c.areaLvl) + "</td>"; if (c.job) html += "<td>" + r[4] + "</td>";
        html += "<td>" + g.length + "</td><td>" + (r[5] ? "yes" : "no") + "</td></tr>";
      });
      tb.innerHTML = html + "</tbody>";
      tb.querySelectorAll("tr[data-i]").forEach(function (tr) {
        function go() { pick = Number(tr.getAttribute("data-i")); draw(); }
        tr.addEventListener("click", go); tr.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(); } });
      });
      Kit.setText("pv-unique", S.unique + " of " + S.n); Kit.setText("pv-k", String(S.k)); Kit.setText("pv-groups", String(S.nGroups)); Kit.setText("pv-rev", S.revealed + " of " + S.n);
      var msg;
      if (pick < 0) msg = "Click a row to play the neighbour who knows that person.";
      else {
        var r = ROWS[pick], g = S.groups[pickKey];
        var who = c.sex ? (r[2] === "F" ? "a woman" : "a man") : "a person";
        if (c.age) who += " aged " + ageLabel(r[1], c.ageBand);
        if (c.area) who += " who lives in " + areaLabel(r[3], c.areaLvl);
        if (c.job) who += (c.area ? " and works" : " who works") + " as " + r[4];
        var yes = g.filter(function (q) { return q[5] === 1; }).length;
        msg = "The neighbour knows " + (names ? "" : r[0] + ", ") + who + ". " + g.length + " row" + (g.length === 1 ? "" : "s") + " in the table match. " +
          (yes === g.length ? "All of them have diabetes: the neighbour learns the diagnosis." : yes === 0 ? "None of them has diabetes: the neighbour learns that too." : yes + " of " + g.length + " have diabetes: the neighbour can only guess.");
      }
      Kit.setText("pv-status", msg);
    }
    draw();
  }

  // ================================================================== pu: unique women in the published Pima table
  function uniqueFig() {
    var n = CL.y.length, on = { age: true, preg: true, bmi: false, glu: false, bp: false }, ageBand = 1;
    var bar = el("pu-bar"), svg = el("pu-svg");
    Kit.html("span", { class: "label" }, bar, "Columns someone knows");
    function chk(label, key) { var l = Kit.html("label", { class: "chk" }, bar), i = Kit.html("input", { type: "checkbox" }, l); i.checked = on[key]; l.appendChild(document.createTextNode(" " + label)); i.addEventListener("change", function () { on[key] = i.checked; draw(); }); return i; }
    var boxes = { age: chk("age", "age"), preg: chk("pregnancies", "preg"), bmi: chk("BMI, to the nearest whole number", "bmi"), glu: chk("glucose, in bands of 10", "glu"), bp: chk("blood pressure, in bands of 10", "bp") };
    var bar2 = el("pu-bar2");
    Kit.html("span", { class: "label" }, bar2, "Presets");
    function setAll(s) { Object.keys(on).forEach(function (k) { on[k] = !!s[k]; boxes[k].checked = !!s[k]; }); draw(); }
    Kit.chips(bar2, ["Age only", "Age + pregnancies", "+ BMI", "+ glucose"], 1, function (i) {
      setAll([{ age: 1 }, { age: 1, preg: 1 }, { age: 1, preg: 1, bmi: 1 }, { age: 1, preg: 1, bmi: 1, glu: 1 }][i]);
    });
    Kit.select(bar2, "Age as", [{ value: "1", label: "single years" }, { value: "5", label: "5-year bands" }, { value: "10", label: "10-year bands" }], "1", function (v) { ageBand = Number(v); draw(); });
    function key(i) {
      var k = [];
      if (on.age) k.push(Math.floor(CL.age[i] / ageBand)); if (on.preg) k.push(CL.preg[i]);
      if (on.bmi) k.push(CL.bmi[i] == null ? "x" : Math.round(CL.bmi[i])); if (on.glu) k.push(CL.glucose[i] == null ? "x" : Math.floor(CL.glucose[i] / 10)); if (on.bp) k.push(CL.bp[i] == null ? "x" : Math.floor(CL.bp[i] / 10));
      return k.join("|");
    }
    function draw() {
      var cnt = {}, i;
      for (i = 0; i < n; i++) { var k = key(i); cnt[k] = (cnt[k] || 0) + 1; }
      var uniq = 0, kmin = Infinity, flags = [];
      for (i = 0; i < n; i++) { var c = cnt[key(i)]; flags.push(c === 1); if (c === 1) uniq++; if (c < kmin) kmin = c; }
      var P = Kit.palette(), W = Kit.width(svg), cols = 48, rows = 16, step = W / cols, r = Math.max(2.2, step * 0.36), H = Math.round(rows * step + 4);
      svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.innerHTML = "";
      var order = flags.map(function (f, i) { return i; }).sort(function (a, b) { return (flags[b] ? 1 : 0) - (flags[a] ? 1 : 0); });
      order.forEach(function (idx, j) {
        var cx = (j % cols + 0.5) * step, cy = (Math.floor(j / cols) + 0.5) * step + 2;
        Kit.el("circle", { cx: cx, cy: cy, r: r, fill: flags[idx] ? P.coral : "none", stroke: flags[idx] ? P.coral : P.navy, "stroke-width": 1.4 }, svg);
      });
      Kit.setText("pu-unique", uniq + " of " + n); Kit.setText("pu-share", Kit.pct(uniq / n, uniq < n && uniq / n > 0.995 ? 1 : 0)); Kit.setText("pu-groups", String(Object.keys(cnt).length)); Kit.setText("pu-k", String(kmin));
      Kit.setText("pu-status", uniq === 0 ? "No woman is alone in her group with these columns." : "With these columns, " + uniq + " women are the only one with their combination. Anyone who knew these values for one of them could pick her row.");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }
})();
