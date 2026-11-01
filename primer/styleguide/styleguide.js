// Style guide: mounts the live demos with the Kit helpers. It adds no library feature.
(function () {
  "use strict";
  var $ = Kit.$;

  // Palette swatches: read the CSS variables of the current theme and print their values.
  function paintTokens() {
    document.querySelectorAll(".gd-sw").forEach(function (sw) {
      sw.querySelector(".gd-hex").textContent = Kit.css(sw.getAttribute("data-token"));
    });
  }
  paintTokens();
  Kit.onTheme(paintTokens);

  // Type scale: read the size, weight and family of the samples, again when the window changes (the phone sizes differ).
  function measure() {
    document.querySelectorAll("[data-measure]").forEach(function (el) {
      var cs = getComputedStyle(el), fam = cs.fontFamily.split(",")[0].replace(/["']/g, "").trim();
      el.closest(".gd-row").querySelector(".gd-size").textContent = Math.round(parseFloat(cs.fontSize) * 10) / 10 + " px, weight " + cs.fontWeight + ", " + fam;
    });
  }
  measure();
  window.addEventListener("resize", Kit.debounce(measure, 150));

  // Widths and fills that a chapter script would set: data-px for spacing bars, data-w (percent) and data-fill (a palette colour).
  document.querySelectorAll("[data-px]").forEach(function (e) { e.style.width = e.getAttribute("data-px") + "px"; });
  document.querySelectorAll("[data-w]").forEach(function (e) { e.style.width = e.getAttribute("data-w") + "%"; });
  document.querySelectorAll("[data-fill]").forEach(function (e) { e.style.background = "var(--" + e.getAttribute("data-fill") + ")"; });

  // The footer demo toggles the theme of the page.
  document.querySelectorAll(".gd-demo .theme-toggle").forEach(function (b) {
    b.addEventListener("click", function () { $("theme-toggle").click(); });
    Kit.onTheme(function () { b.textContent = $("theme-toggle").textContent; });
  });

  // Practice options: the picked option gets the on class, as the practice scripts do.
  document.querySelectorAll(".gd-demo .opts:not([disabled]) input").forEach(function (inp) {
    inp.addEventListener("change", function () {
      inp.closest(".opts").querySelectorAll(".opt").forEach(function (o) { o.classList.toggle("on", o.contains(inp)); });
    });
  });

  // Card with controls and a plot.
  var LEN = [], OBS = [], r = Kit.rng(7), i, L;
  for (i = 0; i < 14; i++) { L = 42 + i * 6; LEN.push(L); OBS.push(2.05 * Math.pow(L / 80, 3) * (1 + 0.09 * Kit.gauss(r))); }
  var show = 2, colour = "coral", b;
  function curve(l) { return 2.05 * Math.pow(l / 80, b.get()); }
  function draw() {
    var err = 0, xs = [], ys = [], ex = [], ey = [];
    for (i = 0; i < LEN.length; i++) {
      err += Math.abs(curve(LEN[i]) - OBS[i]) / LEN.length;
      ex.push(LEN[i], LEN[i], null); ey.push(OBS[i], curve(LEN[i]), null);
    }
    for (L = 40; L <= 120; L += 4) { xs.push(L); ys.push(curve(L)); }
    Kit.setText("gd-err", Kit.fmt(err, 2)); Kit.setText("gd-exp", Kit.fmt(b.get(), 1));
    Kit.setText("gd-status", Math.abs(b.get() - 3) < 0.15 ? "At this exponent the curve follows the sharks." : "The curve misses the sharks by " + Kit.fmt(err, 2) + " kg on average.");
    Kit.plot("gd-plot", function (P) {
      var t = [];
      if (show !== 1) t.push({ x: LEN, y: OBS, mode: "markers", marker: { color: P.navy, size: 8 }, hoverinfo: "skip" });
      if (show !== 0) t.push({ x: xs, y: ys, mode: "lines", line: { color: P[colour], width: 3 }, hoverinfo: "skip" });
      if ($("gd-resid").checked && show !== 1) t.push({ x: ex, y: ey, mode: "lines", line: { color: P.sand, width: 2 }, hoverinfo: "skip" });
      return t;
    }, { xaxis: { title: { text: "length, cm" }, range: [35, 125] }, yaxis: { title: { text: "weight, kg" }, range: [0, 8] }, margin: { l: 52, r: 12, t: 8, b: 46 } });
  }
  b = Kit.slider($("gd-controls"), { label: "Exponent b", min: 1, max: 4, step: 0.1, value: 3 }, draw);
  Kit.select($("gd-controls"), "Curve colour", [{ value: "coral", label: "coral" }, { value: "teal", label: "teal" }, { value: "navy", label: "navy" }], "coral", function (v) { colour = v; draw(); });
  Kit.chips($("gd-bar"), ["Points", "Curve", "Both"], 2, function (k) { show = k; draw(); });
  Kit.button($("gd-bar"), "Play", function () {
    Kit.animate(1600, function (p) { b.set(1 + 3 * Kit.ease(p)); draw(); });
  });
  $("gd-resid").addEventListener("change", draw);
  draw();

  // The five series colours.
  var xs2 = [], k;
  for (k = 0; k <= 40; k++) xs2.push(k / 4);
  var names = ["navy", "coral", "teal", "sand", "ink-2"];
  Kit.plot("gd-plot2", function (P) {
    return P.series.map(function (c, j) {
      return { x: xs2, y: xs2.map(function (x) { return Math.sin(x / (1 + 0.35 * j)) + 0.9 * j; }), mode: "lines", name: names[j], line: { color: c, width: 3 }, hoverinfo: "skip" };
    });
  }, { showlegend: true, xaxis: { title: { text: "x" } }, yaxis: { title: { text: "y" } }, margin: { l: 52, r: 12, t: 8, b: 70 } });

  // Multiple choice.
  Kit.quiz("gd-quiz", [
    { q: "A shark is 80 cm long. Another is 160 cm. How much heavier is the second?", o: ["2 times", "4 times", "8 times"], a: 2, why: "Weight grows like length cubed, and 2 cubed is 8." },
    { q: "What does a residual measure?", o: ["The distance between the data and the curve", "The slope of the curve"], a: 0, why: "A residual is the observed value minus the predicted value." }
  ]);

  // Pixel canvases: the same 12 by 6 grid, with hard edges (fig) and smoothed (fig photo).
  function pixels() {
    ["gd-cv1", "gd-cv2"].forEach(function (id) {
      var cv = $(id), ctx = cv.getContext("2d"), x, y;
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.fillStyle = Kit.css("navy");
      for (y = 0; y < cv.height; y++) for (x = 0; x < cv.width; x++) {
        ctx.globalAlpha = 0.08 + 0.92 * (0.5 + 0.5 * Math.sin(x * 0.9) * Math.cos(y * 1.3));
        ctx.fillRect(x, y, 1, 1);
      }
      ctx.globalAlpha = 1;
    });
  }
  pixels();
  Kit.onTheme(pixels);
})();
