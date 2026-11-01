// Starter chapter script: one Plotly figure with live equation boxes, one D3 diagram, one quiz.
// It goes with the example sections of page.html. Replace both with your own figures.
(function () {
  "use strict";
  document.addEventListener("DOMContentLoaded", function () { weight(); graph(); });

  // ---------- weight from length (Plotly) ----------
  function weight() {
    var r = Kit.rng(7), L = [], W = [];
    for (var i = 0; i < 60; i++) { var l = 35 + 70 * r(); L.push(l); W.push(0.004 * Math.pow(l, 3.05) / 1000 * Math.exp(0.12 * Kit.gauss(r))); }
    var pick = 0;
    var b = Kit.slider(Kit.$("lw-controls"), { label: "Exponent b", min: 1, max: 4, step: 0.05, value: 2 }, draw);
    function draw() {
      var bb = b.get(), xs = [], ys = [], err = 0;
      for (var l = 30; l <= 110; l += 2) { xs.push(l); ys.push(0.004 * Math.pow(l, bb) / 1000); }
      L.forEach(function (l, i) { var e = 0.004 * Math.pow(l, bb) / 1000 - W[i]; err += e * e; });
      Kit.plot("lw-plot", function (P) {
        return [
          { x: L, y: W, mode: "markers", marker: { color: P.navy, size: 8, opacity: 0.8 }, hovertemplate: "%{x:.0f} cm, %{y:.2f} kg<extra></extra>" },
          { x: xs, y: ys, mode: "lines", line: { color: P.coral, width: 3 }, hoverinfo: "skip" }
        ];
      }, { xaxis: { title: { text: "length, cm" } }, yaxis: { title: { text: "weight, kg" }, range: [0, 6] } });
      Kit.setText("lw-err", Kit.fmt(err / L.length, 3));
      Kit.setText("lw-b", Kit.fmt(bb, 2));
      Kit.setText("lw-eq-l", L[pick].toFixed(0));
      Kit.setText("lw-eq-w", (0.004 * Math.pow(L[pick], bb) / 1000).toFixed(2));
    }
    draw();
    Kit.$("lw-plot").on && Kit.$("lw-plot").on("plotly_hover", function (e) { if (e.points[0].curveNumber === 0) { pick = e.points[0].pointIndex; draw(); } });
  }

  // ---------- a small D3 diagram that follows the theme ----------
  function graph() {
    var svg = d3.select("#gr-svg"), step = 0;
    Kit.button(Kit.$("gr-bar"), "Step", function () { step = (step + 1) % 3; draw(); });
    function draw() {
      var P = Kit.palette(), W = Kit.width(Kit.$("gr-svg")), H = 120;
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      var xs = [W * 0.2, W * 0.5, W * 0.8];
      svg.selectAll("line").data([0, 1]).enter().append("line")
        .attr("x1", function (d) { return xs[d] + 22; }).attr("x2", function (d) { return xs[d + 1] - 22; }).attr("y1", 60).attr("y2", 60)
        .attr("stroke", P.muted).attr("stroke-width", 2);
      svg.selectAll("circle").data(xs).enter().append("circle")
        .attr("cx", function (d) { return d; }).attr("cy", 60).attr("r", 20)
        .attr("fill", function (d, i) { return i === step ? P.teal : P.paper; }).attr("stroke", P.navy).attr("stroke-width", 2);
      svg.selectAll("text").data(["input", "unit", "output"]).enter().append("text")
        .attr("x", function (d, i) { return xs[i]; }).attr("y", 104).attr("text-anchor", "middle").attr("font-size", 14).attr("fill", P.ink2).text(function (d) { return d; });
    }
    draw();
    Kit.onTheme(draw);
    window.addEventListener("resize", Kit.debounce(draw));
    Kit.quiz("gr-quiz", [{ q: "Which node is highlighted after one press of Step?", o: ["input", "unit", "output"], a: 1, why: "Step moves the highlight one node to the right." }]);
  }
})();
