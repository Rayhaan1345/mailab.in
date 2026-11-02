// A5 Decision trees: wires the TreeLab figures to the made-up SharkCam catch table.
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () {
    Kit.data("../data/sharks/catch.json").then(start).catch(function (e) { console.error(e); });
  });
  function start(C) {
    var rows = C.records;
    var base = {
      rows: rows, features: ["length_cm", "depth_m"], featureLabels: ["length, cm", "depth, m"], featureNames: ["length", "depth"], units: ["cm", "m"],
      label: function (r) { return r.species === "spadenose" ? 1 : 0; }, classNames: ["other shark", "spadenose"], noun: "shark", nouns: "sharks",
      heldOut: function (r) { return r.record_id % 10 < 3; }, ranges: [[15, 125], [0, 62]], ids: "record_id", idLabel: "Tag", digits: [1, 1], footnote: "Made-up records from the SharkCam catch table, not field data. Coral diamonds are spadenose sharks and navy dots are other sharks."
    };
    function opt(extra) { var o = {}; Object.keys(base).forEach(function (k) { o[k] = base[k]; }); Object.keys(extra || {}).forEach(function (k) { o[k] = extra[k]; }); return o; }
    var byTag = {}; rows.forEach(function (r) { byTag[r.record_id] = r; });
    function tagX(id) { return [byTag[id].length_cm, byTag[id].depth_m]; }
    if (Kit.$("rd")) TreeLab.mountRead(Kit.$("rd"), opt({ tree: { depth: 2 }, picks: [{ name: "Tag 3305", x: tagX(3305) }, { name: "Tag 4127", x: tagX(4127) }, { name: "Tag 5519", x: tagX(5519) }],
      aria: "Length and depth of the training sharks, divided into boxes by a small decision tree, with a ringed point you can move", treeAria: "The decision tree with the path of the ringed shark highlighted" }));
    if (Kit.$("gi")) {
      var tr = rows.filter(function (r) { return r.record_id % 10 >= 3 && r.length_cm < 100; }), r = Kit.rng(4127), pick = [];
      while (pick.length < 24) { var j = Math.floor(r() * tr.length); if (pick.indexOf(tr[j]) < 0) pick.push(tr[j]); }
      TreeLab.mountGini(Kit.$("gi"), { values: pick.map(function (q) { return q.length_cm; }), labels: pick.map(function (q) { return q.species === "spadenose" ? 1 : 0; }),
        classNames: base.classNames, noun: "shark", nouns: "sharks", unit: "cm", featureLabel: "length, cm", start: 12, footnote: base.footnote });
    }
    if (Kit.$("gr")) TreeLab.mountGrow(Kit.$("gr"), opt({ maxDepth: 4, aria: "Length and depth of the training sharks with the boxes of your tree behind them", treeAria: "Your tree, drawn from left to right" }));
    if (Kit.$("dp")) TreeLab.mountDepth(Kit.$("dp"), opt({ maxDepth: 12, aria: "The boxes of a tree at the chosen depth limit" }));
    if (Kit.$("rg")) {
      var sp = rows.filter(function (r) { return r.species === "spadenose"; });
      TreeLab.mountRegression(Kit.$("rg"), { rows: sp, x: "length_cm", y: "weight_kg", xLabel: "length, cm", yLabel: "weight, kg", yUnit: "kg", xUnit: "cm", xName: "length", noun: "shark", nouns: "sharks",
        heldOut: function (r) { return r.record_id % 10 < 3; }, xMin: 15, xMax: 70, yMax: 1.2, probe: 45, depth: 3,
        law: function (L) { return 0.0044 * Math.pow(L, 2.923) / 1000; }, lawName: "the rule behind the data", footnote: "Made-up spadenose records from the SharkCam catch table, not field data. The rule is the published length-weight law that the generator used." });
    }
  }
})();
