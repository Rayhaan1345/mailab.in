// A6 Ensembles and boosting: wires the BoostLab figures to the made-up SharkCam catch table and the exported XGBoost model.
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("../data/sharks/catch.json"), Kit.data("data/xgb6.json")]).then(function (r) { start(r[0], r[1]); }).catch(function (e) { console.error(e); });
  });
  function start(C, M) {
    var rows = C.records, note = "Made-up records from the SharkCam catch table, not field data.";
    var base = {
      rows: rows, features: ["length_cm", "depth_m"], featureLabels: ["length, cm", "depth, m"], featureNames: ["length", "depth"], units: ["cm", "m"],
      label: function (r) { return r.species === "spadenose" ? 1 : 0; }, classNames: ["other shark", "spadenose"], noun: "shark", nouns: "sharks",
      heldOut: function (r) { return r.record_id % 10 < 3; }, ranges: [[15, 125], [0, 62]], ids: "record_id", idLabel: "Tag", digits: [1, 1],
      footnote: note + " Coral diamonds are spadenose sharks and navy dots are other sharks."
    };
    function opt(extra) { var o = {}; Object.keys(base).forEach(function (k) { o[k] = base[k]; }); Object.keys(extra || {}).forEach(function (k) { o[k] = extra[k]; }); return o; }
    if (Kit.$("fo")) BoostLab.mountForest(Kit.$("fo"), opt({ trees: 100, depth: 3, point: [41, 18], aria: "A heat map of the share of trees voting spadenose, over length and depth" }));
    if (Kit.$("b1")) {
      var sp = rows.filter(function (r) { return r.species === "spadenose"; }), idx = {}; sp.forEach(function (r, i) { idx[r.record_id] = i; });
      BoostLab.mount1D(Kit.$("b1"), { rows: sp, x: "length_cm", y: "weight_kg", xLabel: "length, cm", yLabel: "weight, kg", residLabel: "left over, kg", xUnit: "cm", yUnit: "kg", noun: "shark", nouns: "sharks",
        heldOut: function (r) { return r.record_id % 10 < 3; }, xMin: 15, xMax: 65, yMax: 0.9, maxTrees: 100, eta: 0.3, depth: 1, lambda: 0, ids: "record_id", idLabel: "Tag",
        probes: [{ name: "Tag 3305", row: idx[3305] }, { name: "Tag 3646", row: idx[3646] }], law: function (L) { return 0.0044 * Math.pow(L, 2.923) / 1000; }, lawName: "the rule behind the data",
        footnote: "Made-up spadenose records from the SharkCam catch table, not field data. The rule is the published length-weight law that the generator used." });
    }
    if (Kit.$("lf")) BoostLab.mountLeaf(Kit.$("lf"), { leaves: [{ name: "A leaf with 2 sharks", n: 2, sum: 0.1 }, { name: "A leaf with 12 sharks", n: 12, sum: 0.6 }, { name: "A leaf with 120 sharks", n: 120, sum: 6 }], noun: "shark", nouns: "sharks", unit: "kg", lambda: 3, footnote: "Made-up left-over values: all three leaves have an average of 0.05 kg." });
    if (Kit.$("bd")) BoostLab.mountBoundary(Kit.$("bd"), opt({ maxTrees: 300, eta: 0.1, depth: 3, lambda: 1, reference: { label: "One depth-2 tree (A5)", held: 0.8598 } }));
    var names = ["length", "weight", "girth", "depth", "temperature", "month"];
    if (Kit.$("im")) BoostLab.mountImportance(Kit.$("im"), { model: M, featureLabels: names, hot: 0, footnote: note + " Model: 100 trees of depth 3, learning rate 0.1, trained on the 436 training sharks." });
    if (Kit.$("wf")) BoostLab.mountWaterfall(Kit.$("wf"), { model: M, featureLabels: names, units: ["cm", "kg", "cm", "m", "°C", ""], digits: [1, 3, 1, 1, 1, 0], classNames: ["other shark", "spadenose"], noun: "shark",
      picks: [{ name: "Tag 3305", id: 3305 }, { name: "Tag 4127", id: 4127 }, { name: "Tag 5519", id: 5519 }], footnote: note });
  }
})();
