// Chapter B8: debugging neural networks. Figures read data/b8.json and data/b8_pipeline.json (offline PyTorch runs on the made-up catch records).
(function () {
  "use strict";
  var K = Kit, $ = K.$, D = null, PIPE = null;

  function line(name, x, y, color, o) { o = o || {}; return { type: "scatter", mode: o.mode || "lines", name: name, x: x, y: y, connectgaps: false, line: { color: color, width: o.width || 2.2, dash: o.dash || "solid" }, marker: { color: color, size: o.size || 6 }, hoverinfo: "skip", showlegend: !!o.legend }; }
  function lab(x, y, text, color, pos) { return { type: "scatter", mode: "text", x: [x], y: [y], text: [text], textposition: pos || "top right", textfont: { color: color, size: 13 }, hoverinfo: "skip", showlegend: false }; }
  function pct(v) { return (100 * v).toFixed(0) + "%"; }
  function setHTML(id, h) { var n = $(id); if (n) n.innerHTML = h; }
  function tile(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + "</b><i>" + small + "</i></span>"; }
  function fmtSci(v) { if (v == null) return "NaN"; if (v === 0) return "0"; var e = Math.floor(Math.log10(Math.abs(v))); var m = v / Math.pow(10, e); return Math.abs(e) < 3 ? v.toPrecision(3).replace(/\.?0+$/, "") : m.toFixed(1) + "e" + e; }
  var STEPTICKS = { tickvals: [0.7, 1, 3, 10, 30, 100, 300], ticktext: ["start", "1", "3", "10", "30", "100", "300"] };
  function stepTicks(id) { return K.width($(id)) < 520 ? { tickvals: [0.7, 10, 100], ticktext: ["start", "10", "100"], tickangle: 0 } : STEPTICKS; }
  var LOGTICKS = { tickvals: [-8, -6, -4, -2, 0, 2], ticktext: ["1e-8", "1e-6", "1e-4", "0.01", "1", "100"] };

  // ------------------------------------------------------------------ 1. the recipe and the starting loss
  var RECIPE = [
    ["Look at the data", "Print rows with their labels. Plot the inputs. Check one shark you know: Tag 5519 should be a hammerhead.", "Wrong labels, odd scales, outliers (a 58 kg shark among 1 kg sharks), duplicates.", "A shark you know has the label you expect."],
    ["Get dumb baselines", "Score a model that always says the commonest species, then a simple model. A rough first system quickly, then improve it.", "A network that is no better than the lazy model, or a bug that makes it worse.", "Lazy model 48% on the 120 validation sharks; a small tree about 80%."],
    ["Check the starting loss", "Before any training, compute the loss on one batch.", "A last layer that starts too large, or a loss that does not match the output.", "About ln 5 = 1.61 for five species with random scores."],
    ["Overfit one small batch", "Train on 16 sharks until the loss is near 0.", "Bugs in the model, the loss, the optimiser step or zero_grad.", "Loss below 0.1 within about 200 steps."],
    ["Add one thing at a time", "Add inputs, depth, a penalty or augmentation one at a time, and plot training and validation loss.", "Overfitting, leakage, underfitting; which change helped.", "A gap between the curves that you can explain."],
    ["Read the gradients", "Print the gradient norm of the layers and the update-to-weight ratio.", "Vanishing and exploding slopes, a learning rate far off.", "Layer norms within a factor of 10 of one another; a ratio near 1e-3."],
    ["Tune, then look at mistakes", "Run a learning-rate range test, then list the shark photos or records the model gets wrong.", "A rate that is too high or too low; a type of error you can fix with data.", "A clear region of rates where the loss falls."]
  ];
  function recipeFig() {
    var cur = 0;
    K.chips($("rc-bar"), RECIPE.map(function (r, i) { return "Step " + (i + 1) + ": " + r[0]; }), 0, function (i) { cur = i; draw(); });
    function draw() {
      var r = RECIPE[cur];
      setHTML("rc-panel", "<p><b>What you do.</b> " + r[1] + "</p><p><b>What it catches.</b> " + r[2] + "</p><p><b>What you should see.</b> " + r[3] + "</p>");
    }
    draw();
  }
  function startFig() {
    var SL = D.startloss;
    var sl = K.slider($("sl-controls"), { label: "Typical size of the network's five output scores at the start", min: 0, max: SL.length - 1, step: 1, value: 0, show: function (i) { return String(SL[i][0]); } }, function () { draw(); });
    function draw() {
      var i = sl.get(), ent = D.entropy;
      K.plot("sl-plot", function (P) {
        return [line("", SL.map(function (r) { return r[0]; }), SL.map(function (r) { return r[1]; }), P.teal, { width: 3, mode: "lines+markers" }),
          line("", [0.005, 6], [Math.log(5), Math.log(5)], P.ink2, { dash: "dash", width: 1.5 }), line("", [0.005, 6], [ent, ent], P.sand, { dash: "dot", width: 1.5 }),
          { type: "scatter", mode: "markers", x: [SL[i][0]], y: [SL[i][1]], marker: { color: P.coral, size: 13, line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip", showlegend: false },
          lab(0.006, Math.log(5) + 0.35, "ln 5 = 1.61: random guessing", P.ink2, "middle right"), lab(0.006, ent - 0.35, "base rates only: 1.23", P.sand, "middle right")];
      }, { xaxis: { type: "log", title: { text: "size of the output scores (log scale)" }, range: [Math.log10(0.005), Math.log10(6)] }, yaxis: { title: { text: "loss on the first batch" }, range: [0, 6.5] } });
      setHTML("sl-eqn", tile("c-navy", "ln 5", "five equal scores") + '<span class="op">+ &frac12; s&sup2; &times; 0.8 &asymp;</span>' + tile("c-teal", (Math.log(5) + 0.5 * SL[i][0] * SL[i][0] * 0.8).toFixed(2), "small-s approximation, s = " + SL[i][0]) + '<span class="op">vs</span>' + tile("c-coral", SL[i][1].toFixed(2), "simulated, 40,000 draws"));
      setHTML("sl-read", "<span>Scores of size <b>" + SL[i][0] + "</b>: the first loss is <b>" + SL[i][1].toFixed(2) + "</b>, " + (SL[i][1] < 1.75 ? "close to ln 5, so the last layer starts small" : "well above ln 5, so the last layer starts too large") + "</span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 2. the broken-run lab
  var FAULTS = [
    { key: "nan", chip: "The loss turns into NaN", cause: 0, symptom: "The loss climbs from 1.5 to 6, then 47, then thousands, and a few steps later the loss is no longer a number.", cName: "The learning rate is too high",
      cause_text: "A step overshoots the valley, the weights grow, the loss overflows. Plain SGD at rate 2.5 on this network does that in 10 steps.",
      look: "In the gradient plot, layer norms leap upward before the loss breaks. In the ratio plot the update is about as big as the weight itself (ratio near 1 instead of 0.001).",
      notes: ["Lower rate (0.05): the run trains and reaches 75% on validation.", "Clipping at norm 1.0 stops the NaN, but the step is still too big. Units die (the last layer is 100% dead) and the loss stays at the base-rate value 1.23.", "A wider network makes it worse: the NaN comes at step 6."] },
    { key: "init", chip: "The loss does not move from 1.6", cause: 1, symptom: "The loss starts at 1.61 and settles at 1.23, the entropy of the labels. Validation accuracy is 48%, the lazy model's.", cName: "The weights start at a bad scale",
      cause_text: "With weight spread 0.005 a layer shrinks the signal by a large factor (B6). The slope reaching the early layers is tens of thousands of times smaller than at the output.",
      look: "In the gradient plot, layer 1 is near 1e-6 and layer 5 near 0.02. Update ratios in layers 1 to 3 are around 1e-7, a factor 10,000 below the usual 1e-3.",
      notes: ["He initialisation (B6): the run trains, 75% on validation.", "Adam (rate 0.01) trains too, because it divides a slope by its own typical size. It works round the problem, and the cause remains.", "A rate of 0.5 does not give the early layers a usable slope, and it kills units: the loss stays at 1.23."] },
    { key: "raw", chip: "The loss starts huge, then freezes", cause: 2, symptom: "The first loss is 9.1, far above ln 5 = 1.61. Within a few steps it falls to 1.23 and stays there.", cName: "The inputs are not standardised",
      cause_text: "Weight in kg, length in cm and depth in m have sizes from 1 to 240, so the first scores are large and the first steps are large. Most units die.",
      look: "The starting loss is the clue: it should be near 1.61. The dead-unit bars show 96% of layer 1 off.",
      notes: ["Standardising (A2, B6): the loss starts at 1.48 and the run trains to 75%.", "A lower rate (0.005) rescues the run (74%) but the starting loss is still 9.1: the data problem is hidden, not fixed.", "A wider network diverges at step 5."] },
    { key: "labels", chip: "Training improves, validation does not", cause: 3, symptom: "Training loss falls from 1.5 to 1.1. Validation loss rises and validation accuracy is 32%, below the lazy model's 48%.", cName: "The labels do not match the inputs",
      cause_text: "A bug shuffled the labels apart from their rows. The network memorises the pairs it sees, and what it learns does not transfer (Zhang and others, 2017, show that networks can fit random labels).",
      look: "Gradient norms and update ratios look healthy: all numbers inside the training loop look normal. Only the validation score, below the lazy baseline, gives it away.",
      notes: ["Fixing the alignment: 75% on validation.", "A lower rate changes the curve a little. Validation accuracy stays at 34%.", "A wider network fits the training set better (loss 1.02) and validates worse (1.54)."] },
    { key: "spike", chip: "The loss leaps around, norms jump 100 times", cause: 4, symptom: "Squared error on weight in kg jumps between 0.3 and 40, and the gradient norms jump between batches. Validation error is 1.8 kg².", cName: "A few huge gradients (no clipping)",
      cause_text: "Squared error on raw weights: one 58 kg shark in a batch gives a gradient 100 times bigger than a normal batch. Without a cap, one such step moves the weights far.",
      look: "Per-layer gradient norms swing by two orders of magnitude (1 to 70). The update-to-weight ratio shows the effect of clipping: capped near 1e-3 instead of jumping.",
      notes: ["Clipping at norm 1.0 tames the jumps: validation error 0.13 against 1.78.", "A 10 times lower rate (0.0003) does not fix it: validation error 2.95. The steps are smaller and the gradients are just as uneven.", "Predicting the log of the weight (A2) removes the heavy tail, and the loss falls smoothly with no clipping. Its units are log kg squared, so compare the curves and not the numbers."] },
    { key: "dead", chip: "The loss sticks at 1.23, gradients are 0", cause: 5, symptom: "Loss 1.25, validation accuracy 48%. Gradient norms of layers 1 to 4 are 0, so the lines are missing in the plot.", cName: "Dead ReLUs after a too large step",
      cause_text: "A learning rate of 0.5 pushes biases so far negative that units output 0 for all sharks. A ReLU with a negative input has slope 0, so it cannot recover.",
      look: "The dead-unit bars: 100% of the units in layers 3 and 4 output 0 for all 480 training sharks. Gradient norms of those layers are 0.",
      notes: ["Rate 0.02: trains, 75%.", "Clipping at norm 1.0 helps a little (70%) but 92% of layers 3 and 4 stay dead.", "Leaky ReLU (slope 0.1 below zero) keeps the gradient alive, and at rate 0.5 the run reaches NaN at step 48. The rate has to come down as well."] }
  ];
  var CAUSES = ["learning rate too high (the loss overflows)", "weights start at a bad scale", "inputs not standardised", "labels do not match the inputs", "a few huge gradients, no clipping", "dead ReLU units"];
  function labFig() {
    var st = { f: 0, bits: [0, 0, 0], guess: "" };
    var bar = $("lb-bar"), togs = $("lb-toggles");
    K.chips(bar, FAULTS.map(function (f) { return f.chip; }), 0, function (i) { st.f = i; st.bits = [0, 0, 0]; st.guess = ""; buildToggles(); draw(); });
    var tbtn = [];
    function buildToggles() {
      togs.innerHTML = ""; tbtn = [];
      var F = D.lab[FAULTS[st.f].key], g = K.html("div", { class: "group", role: "group" }, togs);
      F.toggles.forEach(function (t, j) {
        var b = K.html("button", { type: "button", "aria-pressed": "false" }, g, t.label);
        b.addEventListener("click", function () { st.bits[j] = 1 - st.bits[j]; b.setAttribute("aria-pressed", String(!!st.bits[j])); draw(); });
        tbtn.push(b);
      });
      K.html("button", { type: "button", class: "btn" }, togs, "Reset").addEventListener("click", function () { st.bits = [0, 0, 0]; tbtn.forEach(function (b) { b.setAttribute("aria-pressed", "false"); }); draw(); });
    }
    var dg = $("lb-diag"), sel = K.html("select", { "aria-label": "Your diagnosis" }, K.html("label", {}, dg, "Your diagnosis: "));
    K.html("option", { value: "" }, sel, "(choose a cause)"); CAUSES.forEach(function (c, i) { K.html("option", { value: String(i) }, sel, c); });
    sel.addEventListener("change", function () { st.guess = sel.value; verdict(); });
    var vd = K.html("span", { class: "verdict", "aria-live": "polite" }, dg, "");
    function verdict() {
      var F = FAULTS[st.f];
      if (st.guess === "") { vd.textContent = ""; return; }
      vd.textContent = Number(st.guess) === F.cause ? " Right. " + F.cName + "." : " Not this one. Look at the plots again, or try a fix.";
      vd.style.color = Number(st.guess) === F.cause ? "var(--teal)" : "var(--coral)";
    }
    buildToggles();
    function layerSeries(r, key) {
      var out = [[], [], [], [], []], xs = [];
      for (var i = 1; i < r.step.length; i++) { xs.push(r.step[i]); for (var l = 0; l < 5; l++) out[l].push(r[key][i] ? r[key][i][l] : null); }
      return { x: xs, y: out };
    }
    function draw() {
      var F = FAULTS[st.f], data = D.lab[F.key], r = data.runs[st.bits.join("")], reg = data.task === "reg";
      var xs = r.step.map(function (s) { return s === 0 ? 0.7 : s; });
      var lossRange = []; r.train.concat(r.val).forEach(function (v) { if (v != null && v > 0) lossRange.push(v); });
      var hiL = Math.min(1e6, Math.max.apply(null, lossRange)), loL = Math.min.apply(null, lossRange);
      K.plot("lb-loss", function (P) {
        var tr = [line("training loss", xs, r.train, P.navy, { width: 2.6 }), line("validation loss", xs, r.val, P.coral, { width: 2.6 })];
        if (!reg) { tr.push(line("", [0.7, 375], [Math.log(5), Math.log(5)], P.ink2, { dash: "dash", width: 1.2 })); tr.push(line("", [0.7, 375], [D.entropy, D.entropy], P.sand, { dash: "dot", width: 1.2 })); tr.push(lab(380, Math.log(5), "ln 5", P.ink2, "middle left")); tr.push(lab(380, D.entropy, "1.23", P.sand, "bottom left")); }
        if (r.diverged) tr.push(lab(r.last + 0.3, r.train[r.train.length - 1] || 1, "NaN at step " + r.last, P.coral, "top left"));
        tr.push(lab(0.7, r.train[0], "start " + fmtSci(r.train[0]), P.navy, "top right"));
        return tr;
      }, { xaxis: Object.assign({ type: "log", title: { text: "training step" }, range: [Math.log10(0.6), Math.log10(700)] }, stepTicks("lb-loss")), yaxis: { type: "log", title: { text: reg ? "squared error, kg²" : "log loss" } }, margin: { l: 58, r: 40, t: 8, b: 44 } });
      var gs = layerSeries(r, "g"), rs = layerSeries(r, "r"), cols = ["navy", "coral", "teal", "sand", "ink2"];
      function layersPlot(id, S, ytitle, extra) {
        K.plot(id, function (P) {
          var tr = S.y.map(function (y, l) { return line("L" + (l + 1), S.x, y, P[cols[l]], { width: 2, legend: true }); });
          var any = S.y.some(function (y) { return y.some(function (v) { return v != null; }); });
          if (extra) tr = tr.concat(extra(P));
          if (!any) tr.push(lab(10, 0, "no line: the value is 0 or the run ended", P.coral, "middle center"));
          return tr;
        }, { showlegend: true, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 12 } }, xaxis: Object.assign({ type: "log", title: { text: "training step" }, range: [Math.log10(0.8), Math.log10(500)] }, { tickvals: [1, 3, 10, 30, 100, 300], ticktext: ["1", "3", "10", "30", "100", "300"] }), yaxis: Object.assign({ title: { text: ytitle }, range: [-8.5, 2.5] }, LOGTICKS), margin: { l: 58, r: 12, t: K.width($(id)) < 520 ? 46 : 30, b: 44 } });
      }
      layersPlot("lb-grad", gs, "gradient norm of the layer");
      layersPlot("lb-ratio", rs, "update size ÷ weight size", function (P) { return [line("", [0.8, 500], [-3, -3], P.ink2, { dash: "dash", width: 1.2 }), lab(0.9, -3, "1e-3", P.ink2, "top right")]; });
      var dead = r.dead[r.dead.length - 1] || [], names = ["layer 1", "layer 2", "layer 3", "layer 4"];
      K.plot("lb-dead", function (P) { return [{ type: "bar", x: names, y: dead, marker: { color: dead.map(function (v) { return v > 0.5 ? P.coral : P.teal; }) }, text: dead.map(function (v) { return Math.round(v * 100) + "%"; }), textposition: "outside", hoverinfo: "skip" }]; },
        { yaxis: { title: { text: "units off for all sharks" }, range: [0, 1.15], tickformat: ".0%" }, xaxis: {}, margin: { l: 58, r: 12, t: 8, b: 40 } });
      var acc = r.acc[r.acc.length - 1], vf = r.val[r.val.length - 1];
      var good = !r.diverged && ((reg && vf != null && vf < 0.2) || (!reg && acc != null && acc >= 0.7));
      var parts = ["<span>Symptom: " + F.symptom + "</span>", "<span>Start loss <b>" + fmtSci(r.train[0]) + "</b>" + (reg ? "" : " (expected 1.61)") + "</span>"];
      parts.push(r.diverged ? "<span>Run ended: <b>NaN at step " + r.last + "</b></span>" : "<span>After 375 steps: validation " + (reg ? "squared error <b>" + fmtSci(vf) + (st.bits[2] ? " (log kg)²</b>" : " kg²</b>") : "accuracy <b>" + pct(acc) + "</b> (lazy model " + pct(D.lazy_val) + ")") + "</span>");
      parts.push(good ? "<span><b>Healthy run.</b></span>" : "<span><b>Still broken.</b></span>");
      setHTML("lb-read", parts.join(""));
      var notes = ""; st.bits.forEach(function (b, j) { if (b) notes += "<p><b>" + data.toggles[j].label + ".</b> " + F.notes[j] + "</p>"; });
      setHTML("lb-notes", notes || "<p>Pick a fix to see what it does. A note describes that fix on its own.</p>");
      setHTML("lb-look", "<p><b>Where to look.</b> " + F.look + "</p>" + (st.guess !== "" && Number(st.guess) === F.cause ? "<p><b>Cause.</b> " + F.cause_text + "</p>" : ""));
      verdict();
    }
    draw();
  }

  // ------------------------------------------------------------------ 3. update-to-weight ratio calculator
  function ratioFig() {
    var lrS = K.slider($("ra-controls"), { label: "Learning rate η (log scale: slide to the exponent)", min: -4, max: 0, step: 0.1, value: -1.7, show: function (v) { return Math.pow(10, v).toPrecision(2); } }, draw);
    var gS = K.slider($("ra-controls"), { label: "Gradient norm ‖g‖ of the layer (log)", min: -3, max: 2, step: 0.1, value: 0, show: function (v) { return Math.pow(10, v).toPrecision(2); } }, draw);
    var wS = K.slider($("ra-controls"), { label: "Weight norm ‖θ‖ of the layer (log)", min: -1, max: 2, step: 0.1, value: 1, show: function (v) { return Math.pow(10, v).toPrecision(2); } }, draw);
    function draw() {
      var lr = Math.pow(10, lrS.get()), g = Math.pow(10, gS.get()), w = Math.pow(10, wS.get()), r = lr * g / w, lg = Math.log10(r);
      var verdict = lg < -4 ? "far too small: the layer is barely learning" : lg < -3.5 ? "small" : lg <= -2.5 ? "in the usual range" : lg <= -1.5 ? "large: watch the loss" : "too large: a step rewrites the layer";
      K.plot("ra-plot", function (P) {
        return [{ type: "bar", orientation: "h", x: [1.5, 1, 1, 1.5, 1.5, 1.5, 1], y: ["a"], marker: { color: P.rule }, hoverinfo: "skip" }].slice(0, 0).concat([
          line("", [-6, 0], [0, 0], P.rule, { width: 14 }), line("", [-3.5, -2.5], [0, 0], P.teal, { width: 14 }),
          { type: "scatter", mode: "markers+text", x: [Math.max(-6, Math.min(0, lg))], y: [0], marker: { color: P.coral, size: 18, symbol: "diamond", line: { color: P.ink, width: 1.5 } }, text: [r.toExponential(1)], textposition: lg > -1.2 ? "top left" : lg < -5.2 ? "top right" : "top center", textfont: { color: P.ink, size: 14 }, hoverinfo: "skip", showlegend: false },
          lab(-3, -0.35, "about 1e-3", P.teal, "middle center")]);
      }, { xaxis: { title: { text: "update size ÷ weight size (log scale)" }, tickvals: [-6, -5, -4, -3, -2, -1, 0], ticktext: ["1e-6", "1e-5", "1e-4", "1e-3", "0.01", "0.1", "1"], range: [-6.2, 0.2] }, yaxis: { visible: false, range: [-0.9, 0.9] }, margin: { l: 14, r: 14, t: 8, b: 46 } });
      setHTML("ra-eqn", tile("c-navy", lr.toPrecision(2), "learning rate") + '<span class="op">&times;</span>' + tile("c-coral", g.toPrecision(2), "gradient norm") + '<span class="op">&divide;</span>' + tile("c-sand", w.toPrecision(2), "weight norm") + '<span class="op">=</span>' + tile("c-teal", r.toExponential(1), verdict));
    }
    draw();
  }

  // ------------------------------------------------------------------ 4. clipping geometry and the spiky run
  function clipFig() {
    var svg = $("cl-svg"), W = 520, H = 360, cx = 260, cy = 180, S = 120;   // 1 unit = 120 px
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var st = { g: [1.9, 0.9], c: 1.0, mode: 0 };
    var bar = $("cl-bar");
    K.chips(bar, ["clip by norm (keeps the direction)", "clip all components (changes it)"], 0, function (i) { st.mode = i; draw(); });
    var sl = K.slider($("cl-controls"), { label: "Threshold c", min: 0.3, max: 2.5, step: 0.05, value: 1.0, digits: 2 }, function (v) { st.c = v; draw(); });
    var dragging = false;
    function toData(ev) { var r = svg.getBoundingClientRect(), k = W / r.width; return [((ev.clientX - r.left) * k - cx) / S, -((ev.clientY - r.top) * k - cy) / S]; }
    svg.addEventListener("pointerdown", function (ev) { dragging = true; try { svg.setPointerCapture(ev.pointerId); } catch (e) {} move(ev); });
    svg.addEventListener("pointermove", function (ev) { if (dragging) move(ev); });
    svg.addEventListener("pointerup", function () { dragging = false; });
    function move(ev) { var p = toData(ev); st.g = [Math.max(-2.4, Math.min(2.4, p[0])), Math.max(-1.45, Math.min(1.45, p[1]))]; draw(); }
    svg.setAttribute("tabindex", "0");
    svg.addEventListener("keydown", function (ev) { var d = { ArrowLeft: [-0.1, 0], ArrowRight: [0.1, 0], ArrowUp: [0, 0.1], ArrowDown: [0, -0.1] }[ev.key]; if (d) { ev.preventDefault(); st.g = [st.g[0] + d[0], st.g[1] + d[1]]; draw(); } });
    function arrow(P, x0, y0, x1, y1, col, wdt, dash) {
      var g = K.el("g", {}, svg);
      K.el("line", { x1: cx + x0 * S, y1: cy - y0 * S, x2: cx + x1 * S, y2: cy - y1 * S, stroke: col, "stroke-width": wdt, "stroke-dasharray": dash || "" }, g);
      var a = Math.atan2(-(y1 - y0), x1 - x0), hx = cx + x1 * S, hy = cy - y1 * S;
      K.el("polygon", { points: hx + "," + hy + " " + (hx - 11 * Math.cos(a - 0.4)) + "," + (hy - 11 * Math.sin(a - 0.4)) + " " + (hx - 11 * Math.cos(a + 0.4)) + "," + (hy - 11 * Math.sin(a + 0.4)), fill: col }, g);
    }
    function draw() {
      var P = K.palette(); svg.innerHTML = "";
      var kf = Math.max(1, W / Math.max(svg.getBoundingClientRect().width || W, 220)), f1 = Math.round(13 * kf), f2 = Math.round(12 * kf);
      K.el("rect", { x: 0, y: 0, width: W, height: H, fill: "none" }, svg);
      K.el("line", { x1: 10, y1: cy, x2: W - 10, y2: cy, stroke: P.rule }, svg); K.el("line", { x1: cx, y1: 8, x2: cx, y2: H - 8, stroke: P.rule }, svg);
      if (st.mode === 0) K.el("circle", { cx: cx, cy: cy, r: st.c * S, fill: P.mark, stroke: P.teal, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
      else K.el("rect", { x: cx - st.c * S, y: cy - st.c * S, width: 2 * st.c * S, height: 2 * st.c * S, fill: P.mark, stroke: P.teal, "stroke-width": 1.5, "stroke-dasharray": "5 4" }, svg);
      var g = st.g, n = Math.hypot(g[0], g[1]), c = st.c, cl;
      if (st.mode === 0) { var f = Math.min(1, c / Math.max(n, 1e-9)); cl = [g[0] * f, g[1] * f]; } else cl = [Math.max(-c, Math.min(c, g[0])), Math.max(-c, Math.min(c, g[1]))];
      arrow(P, 0, 0, g[0], g[1], P.coral, 3);
      if (Math.hypot(cl[0] - g[0], cl[1] - g[1]) > 1e-6) arrow(P, 0, 0, cl[0], cl[1], P.teal, 3.5);
      var anc1 = (g[0] >= 0 && cx + g[0] * S < W - 110) ? "start" : "end", t1 = K.el("text", { x: cx + g[0] * S + (anc1 === "start" ? 8 : -8), y: Math.max(f1 + 2, cy - g[1] * S - 8), "text-anchor": anc1, fill: P.coral, "font-size": f1 }, svg); t1.textContent = "gradient g";
      var anc2 = (cl[0] >= 0 && cx + cl[0] * S < W - 120) ? "start" : "end", t2 = K.el("text", { x: cx + cl[0] * S + (anc2 === "start" ? 8 : -8), y: Math.min(H - 4, cy - cl[1] * S + 18), "text-anchor": anc2, fill: P.teal, "font-size": f1 }, svg); t2.textContent = "after clipping";
      var t3 = K.el("text", { x: cx + 6, y: 16, fill: P.muted, "font-size": f2 }, svg); t3.textContent = "weight 2 slope";
      var t4 = K.el("text", { x: W - 12, y: cy - 6, "text-anchor": "end", fill: P.muted, "font-size": f2 }, svg); t4.textContent = "weight 1 slope";
      var n2 = Math.hypot(cl[0], cl[1]), ang = Math.atan2(g[1], g[0]) * 180 / Math.PI, ang2 = Math.atan2(cl[1], cl[0]) * 180 / Math.PI;
      setHTML("cl-eqn", tile("c-coral", n.toFixed(2), "‖g‖ before") + '<span class="op">&times; min(1, ' + c.toFixed(2) + " &divide; " + n.toFixed(2) + ") =</span>" + tile("c-teal", n2.toFixed(2), "‖g‖ after") + '<span class="op">|</span>' + tile("c-navy", ang.toFixed(0) + "°", "direction before") + '<span class="op">&rarr;</span>' + tile(Math.abs(ang - ang2) > 0.5 ? "c-coral" : "c-teal", ang2.toFixed(0) + "°", "direction after"));
    }
    K.onTheme(draw); draw();
  }
  function clipRunFig() {
    var R = D.lab.spike.runs, mode = 0;
    K.chips($("cr-bar"), ["validation error", "update size ÷ weight size, layer 1", "training loss"], 0, function (i) { mode = i; draw(); });
    function draw() {
      var a = R["000"], b = R["100"];
      K.plot("cr-plot", function (P) {
        var xa = a.step.map(function (s) { return s === 0 ? 0.7 : s; });
        if (mode === 0) return [line("", xa, a.val, P.coral, { width: 2.4 }), line("", xa, b.val, P.teal, { width: 2.8 }), lab(2, a.val[0] * 1.5, "without clipping", P.coral, "top right"), lab(60, 0.09, "clipped at norm 1.0", P.teal, "bottom center")];
        if (mode === 2) return [line("", xa, a.train, P.coral, { width: 2.4 }), line("", xa, b.train, P.teal, { width: 2.8 }), lab(2, a.train[0] * 0.9, "without clipping", P.coral, "bottom right"), lab(100, b.train[b.train.length - 20] * 0.4, "clipped at norm 1.0", P.teal, "bottom center")];
        var sa = [], ya = [], yb = []; for (var i = 1; i < a.step.length; i++) { sa.push(a.step[i]); ya.push(a.r[i] ? a.r[i][0] : null); yb.push(b.r[i] ? b.r[i][0] : null); }
        return [line("", sa, ya, P.coral, { width: 2.4 }), line("", sa, yb, P.teal, { width: 2.8 }), line("", [1, 375], [-3, -3], P.ink2, { dash: "dash", width: 1.2 }), lab(3, -2.35, "without clipping", P.coral, "middle right"), lab(60, -4.3, "clipped", P.teal, "middle right"), lab(250, -3, "1e-3", P.ink2, "top center")];
      }, mode !== 1 ? { xaxis: Object.assign({ type: "log", title: { text: "training step" }, range: [Math.log10(0.6), Math.log10(500)] }, stepTicks("cr-plot")), yaxis: { type: "log", title: { text: (mode === 0 ? "validation" : "training") + " squared error, kg²" } } }
        : { xaxis: { type: "log", title: { text: "training step" }, range: [0, Math.log10(500)], tickvals: [1, 3, 10, 30, 100, 300] }, yaxis: Object.assign({ title: { text: "update size ÷ weight size" }, range: [-5, -1.5] }, { tickvals: [-5, -4, -3, -2], ticktext: ["1e-5", "1e-4", "1e-3", "0.01"] }) });
      setHTML("cr-read", "<span>After 375 steps: validation squared error <b>" + fmtSci(a.val[a.val.length - 1]) + " kg²</b> without clipping, <b>" + fmtSci(b.val[b.val.length - 1]) + " kg²</b> with clipping at norm 1.0</span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 5. learning-rate range test
  function lrFig() {
    var T = D.lrtest, keys = Object.keys(T.runs), sel = 5;
    var lrs = keys.map(Number);
    var sl = K.slider($("lr-controls"), { label: "Choose a learning rate (the nearest of ten trained runs is shown)", min: 0, max: keys.length - 1, step: 1, value: sel, show: function (i) { return String(lrs[i]); } }, function (v) { sel = v; draw(); });
    function draw() {
      var run = T.runs[keys[sel]], lr0 = lrs[sel];
      var idx = []; T.lr.forEach(function (v, i) { if (T.ema[i] != null) idx.push(i); });
      var first = !lrFig.done;
      var pr = K.plot("lr-range", function (P) {
        var xs = idx.map(function (i) { return T.lr[i]; }), raw = idx.map(function (i) { return T.loss[i]; }), em = idx.map(function (i) { return T.ema[i]; });
        return [line("", xs, raw, P.rule, { width: 1.4 }), line("", xs, em, P.navy, { width: 3 }),
          { type: "scatter", mode: "lines", x: [0.003, 0.03, 0.03, 0.003, 0.003], y: [0.25, 0.25, 3, 3, 0.25], fill: "toself", fillcolor: P.mark, line: { width: 0 }, hoverinfo: "skip", showlegend: false },
          line("", [lr0, lr0], [0.25, 3], P.coral, { dash: "dash", width: 2 }), lab(0.00001, 1.75, "flat: too small to learn", P.ink2, "middle right"), lab(0.0035, 0.36, "steep fall: pick here", P.teal, "middle right"), lab(0.07, 0.6, "lowest", P.navy, "middle center"), lab(0.3, 2.7, "blows up", P.coral, "middle center")];
      }, { xaxis: { type: "log", title: { text: "learning rate, raised from 1e-5 to 3 (log)" }, range: [-5, 0.5] }, yaxis: { title: { text: "loss (smoothed)" }, range: [Math.log10(0.25), Math.log10(3)], type: "log", tickvals: [0.3, 0.5, 1, 2], ticktext: ["0.3", "0.5", "1", "2"] } });
      if (first) { lrFig.done = true; Promise.resolve(pr).then(function () { var gd = $("lr-range"); if (gd && gd.on) gd.on("plotly_click", function (e) { var x = e.points[0].x, best = 0, bd = 1e9; lrs.forEach(function (v, i) { var d = Math.abs(Math.log(v / x)); if (d < bd) { bd = d; best = i; } }); sl.set(best); sel = best; draw(); }); }); }
      var xs2 = run.step.map(function (s) { return s; });
      K.plot("lr-run", function (P) { return [line("", xs2, run.train, P.navy, { width: 2.6 }), line("", xs2, run.val, P.coral, { width: 2.6 }), lab(xs2[2], run.train[2], "training", P.navy, "top right"), lab(xs2[xs2.length - 3], run.val[run.val.length - 1], "validation", P.coral, "top left")]; },
        { xaxis: { title: { text: "training step at rate " + lr0 } }, yaxis: { type: "log", title: { text: "log loss" }, range: [Math.log10(0.2), Math.log10(2.2)], tickvals: [0.3, 0.5, 1, 2], ticktext: ["0.3", "0.5", "1", "2"] } });
      var a = run.acc[run.acc.length - 1];
      setHTML("lr-read", "<span>At <b>" + lr0 + "</b>: final validation accuracy <b>" + pct(a) + "</b> (lazy model " + pct(D.lazy_val) + "), validation loss <b>" + run.val[run.val.length - 1].toFixed(2) + "</b></span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 6. learning curves, bias and variance
  function curvesFig() {
    var C = D.curves, kind = "simple", target = 0.20, names = { simple: "too simple", medium: "about right", flexible: "too flexible" };
    K.chips($("lc-bar"), ["too simple: length only", "about right: 16 units and a penalty", "too flexible: 128 + 128 units"], 0, function (i) { kind = ["simple", "medium", "flexible"][i]; draw(); });
    var sl = K.slider($("lc-controls"), { label: "Training sharks", min: 0, max: C.n.length - 1, step: 1, value: 5, show: function (i) { return String(C.n[i]); } }, function () { draw(); });
    function draw() {
      var k = C.kinds[kind], i = sl.get(), tr = k.train[i], va = k.val[i], gap = va - tr;
      K.plot("lc-plot", function (P) {
        return [line("", C.n, k.train, P.navy, { width: 3, mode: "lines+markers" }), line("", C.n, k.val, P.coral, { width: 3, mode: "lines+markers" }), line("", [C.n[0], C.n[C.n.length - 1]], [target, target], P.teal, { dash: "dash", width: 1.8 }),
          { type: "scatter", mode: "markers", x: [C.n[i], C.n[i]], y: [tr, va], marker: { color: [P.navy, P.coral], size: 13, line: { color: P.ink, width: 1.5 } }, hoverinfo: "skip", showlegend: false },
          lab(C.n[2], k.train[2] - 0.035, "training error", P.navy, "middle center"), lab(C.n[3], k.val[3] + 0.035, "validation error", P.coral, "middle center"), lab(C.n[4], target + 0.03, "target 0.20 (a made-up goal)", P.teal, "middle center")];
      }, { xaxis: { type: "log", title: { text: "training sharks" }, tickvals: C.n, ticktext: C.n.map(String) }, yaxis: { title: { text: "error rate (1 − accuracy)" }, range: [-0.02, 0.5] } });
      var av = Math.max(0, tr - target), diag = tr > target + 0.02 && gap < 0.1 ? "High avoidable bias: training error itself is above the target and the gap is small. A bigger model or better inputs would help, more data would not." : gap > 0.12 ? "High variance: training error is far below validation error. More data, a penalty, augmentation or a smaller model would help." : "Neither is large at this size: the curves are close to the target.";
      setHTML("lc-eqn", tile("c-navy", tr.toFixed(2), "training error") + '<span class="op">&minus;</span>' + tile("c-teal", target.toFixed(2), "target") + '<span class="op">=</span>' + tile("c-sand", av.toFixed(2), "avoidable bias (0 if below)") + '<span class="op">|</span>' + tile("c-coral", va.toFixed(2), "validation error") + '<span class="op">&minus;</span>' + tile("c-navy", tr.toFixed(2), "training error") + '<span class="op">=</span>' + tile("c-coral", gap.toFixed(2), "variance"));
      setHTML("lc-read", "<span><b>" + names[kind] + "</b> with " + C.n[i] + " sharks. " + diag + "</span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 7. reproducibility
  function reproFig() {
    var R = D.repro, pair = true;
    K.chips($("rp-bar"), ["pair the two rates by seed", "show the runs unpaired"], 0, function (i) { pair = i === 0; draw(); });
    function draw() {
      var n = R.A.length, xa = [], xb = [];
      for (var i = 0; i < n; i++) { xa.push(1 + (Math.sin(i * 12.9898) * 0.5) * 0.18); xb.push(2 + (Math.sin(i * 78.233) * 0.5) * 0.18); }
      var diff = R.A.map(function (a, i) { return R.B[i] - a; }), md = diff.reduce(function (a, b) { return a + b; }, 0) / n, sd = Math.sqrt(diff.reduce(function (a, b) { return a + (b - md) * (b - md); }, 0) / (n - 1)), se = sd / Math.sqrt(n);
      K.plot("rp-plot", function (P) {
        var tr = [];
        if (pair) for (var i = 0; i < n; i++) tr.push(line("", [1, 2], [R.A[i], R.B[i]], P.rule, { width: 1.2 }));
        tr.push({ type: "scatter", mode: "markers", x: pair ? R.A.map(function () { return 1; }) : xa, y: R.A, marker: { color: P.navy, size: 9 }, hoverinfo: "skip", showlegend: false });
        tr.push({ type: "scatter", mode: "markers", x: pair ? R.B.map(function () { return 2; }) : xb, y: R.B, marker: { color: P.coral, size: 9, symbol: "diamond" }, hoverinfo: "skip", showlegend: false });
        tr.push(line("", [0.75, 1.25], [R.meanA, R.meanA], P.ink, { width: 3 })); tr.push(line("", [1.75, 2.25], [R.meanB, R.meanB], P.ink, { width: 3 }));
        return tr;
      }, { xaxis: { tickvals: [1, 2], ticktext: ["learning rate " + R.lrA, "learning rate " + R.lrB], range: [0.5, 2.5] }, yaxis: { title: { text: "validation accuracy, 120 sharks" }, range: [0.62, 0.82], tickformat: ".0%" } });
      setHTML("rp-read", "<span>Rate " + R.lrA + ": mean <b>" + pct(R.meanA) + "</b>, spread (sd) <b>" + (100 * R.sdA).toFixed(1) + " points</b></span><span>Rate " + R.lrB + ": mean <b>" + pct(R.meanB) + "</b>, sd <b>" + (100 * R.sdB).toFixed(1) + " points</b></span><span>Paired difference: <b>" + K.fmt(100 * md, 1) + " &plusmn; " + (100 * 1.96 * se).toFixed(1) + " points</b> (95% interval, 20 seeds)</span>");
    }
    draw();
  }

  // ------------------------------------------------------------------ 8. test one component at a time
  var TESTS = [
    ["data", "Test 1: look at one shark you know", "fail", 1, "The row for Tag 5519 (82.0 cm, 2.61 kg) should have the label hammerhead. It says spadenose. The labels do not belong to their rows."],
    ["start", "Test 2: starting loss is about ln 5", "pass", 0, "1.613 against ln 5 = 1.609. This passes, and it proves little: most bugs leave the starting loss alone."],
    ["lazy", "Test 3: beat the lazy baseline", "fail", 0, "The model scores 0.45, the same as always saying 'spadenose'. It has learned no more than a guess. Something is wrong, and this test cannot say what."],
    ["overfit", "Test 4: overfit one small batch", "fail", 2, "The loss wanders between 1.4 and 1.65 and never approaches 0. The step, the loss or the model is broken. (The fixed pipeline gets to 0.07.)"],
    ["grad", "Test 5: same batch twice, gradients", "fail", 3, "The gradient norm doubles and triples when backward is called again without zero_grad: gradients are being added up."],
    ["eval", "Test 6: is the model in evaluation mode?", "fail", 4, "model.training is True when validation runs, and two passes over the same sharks give different outputs: dropout is still on."],
    ["shapes", "Test 7: sizes and types of the arrays", "pass", 0, "The inputs are 480 by 6, the labels have type int64 and run from 0 to 4. All as expected."],
    ["output", "Test 8: print the network's output for two rows", "fail", 5, "The outputs are probabilities (one-hot, sum 1). nn.CrossEntropyLoss expects raw scores and applies softmax itself."]
  ];
  var BUGS = ["labels shuffled apart from their rows", "softmax inside the model and again in the loss", "optimizer.zero_grad() missing", "dropout left on during validation"];
  var BUGOF = { data: 0, overfit: [1, 2], grad: 2, eval: 3, output: 1 };
  function checkFig() {
    var done = {}, found = {};
    var list = $("ck-list");
    TESTS.forEach(function (t) {
      var row = K.html("label", { class: "ck-row" }, list), cb = K.html("input", { type: "checkbox" }, row); K.html("span", {}, row, t[1]);
      cb.addEventListener("change", function () { done[t[0]] = cb.checked; draw(); });
    });
    function draw() {
      var h = "", bugs = {};
      TESTS.forEach(function (t) {
        if (!done[t[0]]) return;
        var T = PIPE.tests[t[0]], res = t[2];
        h += '<div class="ck-res ' + res + '"><p><b>' + t[1].replace(/^Test \d+: /, "") + "</b> <span class=\"ck-badge\">" + (res === "pass" ? "passes" : "fails") + "</span></p><pre class=\"code\"><code>" + escapeHtml(T.code) + "</code></pre><pre class=\"out\">" + escapeHtml(T.output) + "</pre><p>" + t[4] + "</p></div>";
        var b = BUGOF[t[0]]; if (b != null) [].concat(b).forEach(function (x) { bugs[x] = true; });
      });
      $("ck-out").innerHTML = h || "<p>Tick a test to run it on the pipeline above. The outputs are from running the code above.</p>";
      var n = Object.keys(bugs).length;
      setHTML("ck-status", "<span>Tests run: <b>" + Object.keys(done).filter(function (k) { return done[k]; }).length + " of 8</b></span><span>Suspects found: <b>" + n + " of 4</b>" + (n ? " (" + Object.keys(bugs).map(function (i) { return BUGS[i]; }).join("; ") + ")" : "") + "</span>");
      $("ck-fix").style.display = n === 4 ? "" : "none";
    }
    draw();
  }
  function escapeHtml(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }

  // ------------------------------------------------------------------ boot
  function boot() {
    Promise.all([K.data("data/b8.json"), K.data("data/b8_pipeline.json")]).then(function (r) {
      D = r[0]; PIPE = r[1];
      [["rc-bar", recipeFig], ["sl-plot", startFig], ["lb-loss", labFig], ["ra-plot", ratioFig], ["cl-svg", clipFig], ["cr-plot", clipRunFig], ["lr-range", lrFig], ["lc-plot", curvesFig], ["rp-plot", reproFig], ["ck-list", checkFig]].forEach(function (j) {
        if ($(j[0])) { try { j[1](); } catch (e) { console.error(j[0], e); } }
      });
    }).catch(function (e) { console.error(e); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
