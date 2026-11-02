// F2 AI on a robot. Four figures: one SharkCam frame through camera, model, decision and motor (lp) with a pointing lag
// animation (aim); a latency and memory calculator (bu); weight quantisation of the C2 digit reader with measured accuracy (qz);
// and a ladder of model sizes (ld). Pure helpers are exported with module.exports for node tests.
(function () {
  "use strict";

  // ------------------------------------------------------------------ pure helpers
  var GB = 1e9;
  var DEVICES = [
    { id: "pi4", name: "Raspberry Pi 4 (4 GB)", ram: 4 * GB, speed8: 300774272 / 0.0264 / 1e9 },   // MobileNetV2, 8-bit, 26.4 ms per frame (PyTorch tutorial)
    { id: "pi5", name: "Raspberry Pi 5 (8 GB)", ram: 8 * GB, speed8: 300774272 / 0.0211 / 1e9 }    // from about 41 frames per second, minus 3.3 ms outside the model (estimate)
  ];
  var MODEL_META = [
    { id: "c2", name: "C2 digit reader", short: "C2 CNN", unit: "image" },
    { id: "mobilenet", name: "MobileNetV2", short: "MobileNetV2", unit: "frame" },
    { id: "resnet18", name: "ResNet-18", short: "ResNet-18", unit: "frame" },
    { id: "gpt2", name: "GPT-2 small", short: "GPT-2", unit: "token" },
    { id: "llama70", name: "a 70-billion-parameter model", short: "70B model", unit: "token", params: 70e9, macs: 70e9 }
  ];
  function speedAt(speed8, bits, s8) { return bits <= 8 ? speed8 : speed8 / s8; }          // multiply-adds per second, in giga
  function timeMs(macs, speed8, bits, s8) { return macs / (speedAt(speed8, bits, s8) * 1e9) * 1000; }
  function memBytes(params, bits) { return params * bits / 8; }
  function fmtBytes(b) {
    var u = ["B", "KB", "MB", "GB", "TB"], i = 0;
    while (b >= 1000 && i < u.length - 1) { b /= 1000; i++; }
    return (b >= 100 ? b.toFixed(0) : b >= 10 ? b.toFixed(1) : b.toFixed(2)) + " " + u[i];
  }
  function fmtMs(ms) { return ms >= 1000 ? (ms / 1000).toFixed(ms >= 10000 ? 0 : 1) + " s" : ms >= 100 ? ms.toFixed(0) + " ms" : ms >= 10 ? ms.toFixed(1) + " ms" : ms.toFixed(2) + " ms"; }
  function fmtFps(f) { return f >= 100 ? f.toFixed(0) : f >= 10 ? f.toFixed(1) : f >= 1 ? f.toFixed(2) : f >= 0.01 ? f.toFixed(3) : f.toExponential(1); }
  function groupInt(n) { return Math.round(n).toLocaleString("en-US"); }

  // uniform quantisation of one tensor: scale s, zero point z, codes q in 0 .. 2^bits - 1
  function quantParams(arr, bits) {
    var lo = Infinity, hi = -Infinity;
    for (var i = 0; i < arr.length; i++) { if (arr[i] < lo) lo = arr[i]; if (arr[i] > hi) hi = arr[i]; }
    var s = (hi - lo) / (Math.pow(2, bits) - 1);
    return { s: s, z: Math.round(-lo / s), lo: lo, hi: hi };
  }
  function quantise(arr, bits) {
    var p = quantParams(arr, bits), top = Math.pow(2, bits) - 1, q = new Float64Array(arr.length), d = new Float64Array(arr.length);
    for (var i = 0; i < arr.length; i++) {
      var c = Math.round(arr[i] / p.s) + p.z; c = c < 0 ? 0 : c > top ? top : c;
      q[i] = c; d[i] = p.s * (c - p.z);
    }
    return { q: q, deq: d, s: p.s, z: p.z, lo: p.lo, hi: p.hi };
  }

  // the C2 digit reader: conv 1 to 6 (3x3), ReLU, pool, conv 6 to 12 (3x3), ReLU, pool, dense 300 to 10
  function forward(img, P) {
    var p1 = new Float64Array(6 * 13 * 13), o, i, j, a, b, c, s, v;
    for (o = 0; o < 6; o++) for (i = 0; i < 13; i++) for (j = 0; j < 13; j++) {
      var best = -Infinity;
      for (a = 0; a < 2; a++) for (b = 0; b < 2; b++) {
        s = P.conv1_b[o];
        for (var ka = 0; ka < 3; ka++) for (var kb = 0; kb < 3; kb++) s += img[(2 * i + a + ka) * 28 + 2 * j + b + kb] * P.conv1_w[o * 9 + ka * 3 + kb];
        if (s < 0) s = 0;
        if (s > best) best = s;
      }
      p1[(o * 13 + i) * 13 + j] = best;
    }
    var flat = new Float64Array(300);
    for (o = 0; o < 12; o++) for (i = 0; i < 5; i++) for (j = 0; j < 5; j++) {
      var best2 = -Infinity;
      for (a = 0; a < 2; a++) for (b = 0; b < 2; b++) {
        s = P.conv2_b[o];
        for (c = 0; c < 6; c++) for (var ka2 = 0; ka2 < 3; ka2++) for (var kb2 = 0; kb2 < 3; kb2++) s += p1[(c * 13 + 2 * i + a + ka2) * 13 + 2 * j + b + kb2] * P.conv2_w[(o * 6 + c) * 9 + ka2 * 3 + kb2];
        if (s < 0) s = 0;
        if (s > best2) best2 = s;
      }
      flat[o * 25 + i * 5 + j] = best2;
    }
    var out = new Float64Array(10);
    for (var k = 0; k < 10; k++) { v = P.fc_b[k]; for (var m = 0; m < 300; m++) v += flat[m] * P.fc_w[k * 300 + m]; out[k] = v; }
    return out;
  }
  function argmax(a) { var k = 0; for (var i = 1; i < a.length; i++) if (a[i] > a[k]) k = i; return k; }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { speedAt: speedAt, timeMs: timeMs, memBytes: memBytes, quantParams: quantParams, quantise: quantise, forward: forward, argmax: argmax, DEVICES: DEVICES, fmtBytes: fmtBytes };
  }
  if (typeof document === "undefined") return;

  // ------------------------------------------------------------------ shared UI helpers
  function el(id) { return document.getElementById(id); }
  function hexA(hex, a) {
    var h = hex.replace("#", ""); if (h.length === 3) h = h.replace(/(.)/g, "$1$1");
    return "rgba(" + parseInt(h.slice(0, 2), 16) + "," + parseInt(h.slice(2, 4), 16) + "," + parseInt(h.slice(4, 6), 16) + "," + a + ")";
  }
  function setPressed(btns, i) { btns.forEach(function (b, k) { b.setAttribute("aria-pressed", String(k === i)); }); }

  var COUNTS = null, MODELS = null, Q = null, SPRITE = null;
  document.addEventListener("DOMContentLoaded", function () {
    var sprite = new Promise(function (ok, fail) { var im = new Image(); im.onload = function () { ok(im); }; im.onerror = fail; im.src = "data/digits.png"; });
    Promise.all([Kit.data("data/counts.json"), Kit.data("data/quant.json"), sprite]).then(function (r) {
      COUNTS = r[0]; Q = r[1];
      MODELS = MODEL_META.map(function (m) {
        var c = COUNTS[m.id] || {};
        return { id: m.id, name: m.name, short: m.short, unit: m.unit, params: m.params || c.params, macs: m.macs || c.macs };
      });
      var cv = document.createElement("canvas"); cv.width = r[2].width; cv.height = r[2].height;
      var cx = cv.getContext("2d"); cx.drawImage(r[2], 0, 0);
      var px = cx.getImageData(0, 0, cv.width, cv.height).data, n = Q.subset.n, per = cv.width / 28;
      SPRITE = new Float64Array(n * 784);
      for (var d = 0; d < n; d++) { var rr = Math.floor(d / per), cc = d % per; for (var y = 0; y < 28; y++) for (var x = 0; x < 28; x++) SPRITE[d * 784 + y * 28 + x] = px[((rr * 28 + y) * cv.width + cc * 28 + x) * 4] / 255; }
      loopFig(); calcFig(); quantFig(); ladderFig();
    }).catch(function (e) { console.error(e); var s = document.createElement("p"); s.className = "status"; s.textContent = "The data for these figures did not load."; document.querySelector("main").appendChild(s); });
  });
  function model(id) { return MODELS.filter(function (m) { return m.id === id; })[0]; }

  // ================================================================== lp: one frame through the loop, and the pointing lag
  function loopFig() {
    var S = { model: "resnet18", bits: 8, dev: 0, pipe: false, v: 1, d: 3, tol: 5, step: 0 };
    var CAM = 1000 / 30 + 3.5, DEC = 1, MOTOR = 60;      // camera frame interval plus read and resize, decision, a small servo (assumed)
    var bar = el("lp-bar"), ctl = el("lp-controls");
    var mChips = [], bChips = [], dChips = [];
    Kit.html("span", { class: "label" }, bar, "Model");
    mChips = Kit.chips(bar, ["MobileNetV2", "ResNet-18"], 1, function (i) { S.model = ["mobilenet", "resnet18"][i]; reset(); });
    Kit.html("span", { class: "label", style: "margin-left:8px" }, bar, "Bits");
    bChips = Kit.chips(bar, ["32", "8"], 1, function (i) { S.bits = [32, 8][i]; reset(); });
    Kit.html("span", { class: "label", style: "margin-left:8px" }, bar, "Computer");
    dChips = Kit.chips(bar, ["Pi 4", "Pi 5"], 0, function (i) { S.dev = i; reset(); });
    var lab = Kit.html("label", { class: "chk", style: "margin-left:8px" }, bar), cb = Kit.html("input", { type: "checkbox" }, lab);
    lab.appendChild(document.createTextNode(" pipelined"));
    cb.addEventListener("change", function () { S.pipe = cb.checked; reset(); });
    Kit.slider(ctl, { label: "Shark speed, m/s", min: 0.2, max: 2, step: 0.1, value: 1, digits: 1 }, function (v) { S.v = v; reset(); });
    Kit.slider(ctl, { label: "Distance to the shark, m", min: 1, max: 6, step: 0.5, value: 3, digits: 1 }, function (v) { S.d = v; reset(); });
    Kit.slider(ctl, { label: "Allowed pointing error, degrees", min: 2, max: 15, step: 1, value: 5 }, function (v) { S.tol = v; reset(); });
    var bar2 = el("lp-bar2");
    Kit.button(bar2, "Step", function () { stop(); S.step = S.step >= 4 ? 0 : S.step + 1; cursor = S.step ? stageEnd(S.step - 1) : 0; draw(); });
    var playBtn = Kit.button(bar2, "Play", play);
    Kit.button(bar2, "Reset", function () { stop(); S.step = 0; cursor = 0; draw(); });
    var cursor = 0, stopFn = null, aim = { t: 0, max: 0, stop: null };

    function calc() {
      var m = model(S.model), dv = DEVICES[S.dev];
      var tm = timeMs(m.macs, dv.speed8, S.bits, 2.5);
      var st = [{ k: "Camera", t: CAM }, { k: "Model", t: tm }, { k: "Decision", t: DEC }, { k: "Motor", t: MOTOR }];
      var L = st.reduce(function (a, b) { return a + b.t; }, 0);
      var T = S.pipe ? Math.max(1000 / 30, tm) : L;
      var omega = S.v / S.d, omegaDeg = omega * 180 / Math.PI, lag = L + T;
      return { st: st, L: L, T: T, tm: tm, omegaDeg: omegaDeg, lag: lag, errDeg: omegaDeg * lag / 1000, budget: S.tol / omegaDeg * 1000, m: m, dv: dv };
    }
    function stageEnd(k) { var C = calc(), a = 0; for (var i = 0; i <= k; i++) a += C.st[i].t; return a; }
    function reset() { stop(); S.step = 0; cursor = 0; if (aim.stop) aim.stop(); aim.t = 0; aim.max = 0; draw(); drawAim(); }
    function stop() { if (stopFn) stopFn(); stopFn = null; }
    function play() {
      stop(); var C = calc();
      stopFn = Kit.animate(3200, function (p) {
        cursor = p * C.L; var a = 0; S.step = 0;
        for (var i = 0; i < 4; i++) { if (cursor > a) S.step = i + 1; a += C.st[i].t; }
        if (p >= 1) S.step = 4;
        draw();
      });
    }
    var NOTE = ["Light from the shark falls on the sensor for one frame interval, 33 ms at 30 frames per second. Then the Pi reads and resizes the picture, which takes 3.5 ms in the PyTorch tutorial's Pi 4 run.",
      "The network turns the picture into scores. This is the only stage that depends on the model size, the bits and the computer.",
      "A few lines of code: if the shark score is above the threshold, work out how far to turn. About 1 ms (assumed).",
      "The motor receives the command and turns the camera. A small hobby servo needs about 60 ms for a short turn (assumed)."];
    function draw() {
      var C = calc(), P = Kit.palette(), svg = el("lp-svg"), W = Kit.width(svg), H = 214;
      svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.innerHTML = "";
      var cols = [P.sand, P.teal, P.navy, P.coral], gap = 22, bw = (W - 3 * gap - 8) / 4, y0 = 14, bh = 70, i;
      for (i = 0; i < 4; i++) {
        var x = 4 + i * (bw + gap), on = S.step === i + 1;
        Kit.el("rect", { x: x, y: y0, width: bw, height: bh, rx: 8, fill: on ? hexA(cols[i], 0.22) : P.paper, stroke: cols[i], "stroke-width": on ? 3 : 1.5 }, svg);
        var t1 = Kit.el("text", { x: x + bw / 2, y: y0 + 26, "text-anchor": "middle", "font-size": 15, "font-weight": 600, fill: P.ink }, svg); t1.textContent = C.st[i].k;
        var t2 = Kit.el("text", { x: x + bw / 2, y: y0 + 52, "text-anchor": "middle", "font-size": 14, fill: P.ink2 }, svg); t2.textContent = fmtMs(C.st[i].t);
        if (i < 3) {
          Kit.el("line", { x1: x + bw + 3, x2: x + bw + gap - 3, y1: y0 + bh / 2, y2: y0 + bh / 2, stroke: P.muted, "stroke-width": 2 }, svg);
          Kit.el("path", { d: "M" + (x + bw + gap - 3) + " " + (y0 + bh / 2) + " l-6 -4 l0 8 z", fill: P.muted }, svg);
        }
      }
      if (S.step >= 1 && S.step <= 4) {
        var tx = 4 + (S.step - 1) * (bw + gap) + bw / 2;
        Kit.el("circle", { cx: tx, cy: y0 + bh + 12, r: 7, fill: P.teal, stroke: P.ink, "stroke-width": 1.5 }, svg);
      }
      // timeline: the four stages to scale, the wait for the next frame, and the budget
      var padL = 4, padR = 4, bx0 = padL, bw0 = W - padL - padR, total = Math.max(C.L + C.T, C.budget) * 1.06, sc = bw0 / total, by = 142, bhh = 26;
      var lbl = Kit.el("text", { x: bx0, y: by - 10, "font-size": 13, fill: P.muted }, svg); lbl.textContent = "time from the light reaching the lens (to scale)";
      var a = 0;
      C.st.forEach(function (s, k) {
        Kit.el("rect", { x: bx0 + a * sc, y: by, width: Math.max(1, s.t * sc), height: bhh, fill: cols[k], opacity: 0.85 }, svg);
        if (s.t * sc > 34) { var tt = Kit.el("text", { x: bx0 + (a + s.t / 2) * sc, y: by + 17, "text-anchor": "middle", "font-size": 12, fill: P.bg }, svg); tt.textContent = Math.round(s.t); }
        a += s.t;
      });
      Kit.el("rect", { x: bx0 + a * sc, y: by, width: C.T * sc, height: bhh, fill: "none", stroke: P.muted, "stroke-width": 1.5, "stroke-dasharray": "4 3" }, svg);
      var wt = Kit.el("text", { x: bx0 + (a + C.T / 2) * sc, y: by + 17, "text-anchor": "middle", "font-size": 12, fill: P.ink2 }, svg);
      if (C.T * sc > 70) wt.textContent = "wait " + Math.round(C.T) + " ms";
      var bxp = bx0 + C.budget * sc, over = C.L + C.T > C.budget;
      Kit.el("line", { x1: bxp, x2: bxp, y1: by - 6, y2: by + bhh + 22, stroke: over ? P.coral : P.teal, "stroke-width": 2.5 }, svg);
      var bt = Kit.el("text", { x: Math.min(bxp, W - 4), y: by + bhh + 38, "text-anchor": bxp > W * 0.7 ? "end" : "start", "font-size": 13, "font-weight": 600, fill: over ? P.coral : P.link }, svg);
      bt.textContent = "budget " + Math.round(C.budget) + " ms";
      if (S.step > 0) Kit.el("line", { x1: bx0 + cursor * sc, x2: bx0 + cursor * sc, y1: by - 6, y2: by + bhh + 6, stroke: P.ink, "stroke-width": 2 }, svg);
      el("lp-note").textContent = S.step === 0 ? "Press Step to follow one frame, or Play to run it through in slow motion. Change the model, the bits or the computer and watch the model box." : "Stage " + S.step + " of 4, " + C.st[S.step - 1].k + ", " + fmtMs(C.st[S.step - 1].t) + ". " + NOTE[S.step - 1];
      Kit.setText("lp-L", fmtMs(C.L)); Kit.setText("lp-T", fmtMs(C.T)); Kit.setText("lp-lag", fmtMs(C.lag)); Kit.setText("lp-err", Kit.fmt(C.errDeg, 1) + "°");
      el("lp-errbox").className = C.errDeg > S.tol ? "miss" : "hit";
      Kit.setText("lp-eq-cam", C.st[0].t.toFixed(1)); Kit.setText("lp-eq-mod", C.st[1].t < 10 ? C.st[1].t.toFixed(1) : C.st[1].t.toFixed(0)); Kit.setText("lp-eq-dec", C.st[2].t.toFixed(0)); Kit.setText("lp-eq-mot", C.st[3].t.toFixed(0));
      Kit.setText("lp-eq-L", C.L.toFixed(0)); Kit.setText("lp-eq-T", C.T.toFixed(0)); Kit.setText("lp-eq-w", Kit.fmt(C.omegaDeg, 1)); Kit.setText("lp-eq-lag", C.lag.toFixed(0)); Kit.setText("lp-eq-err", Kit.fmt(C.errDeg, 1));
      Kit.setText("lp-eq-bud", C.budget.toFixed(0));
      Kit.setText("lp-status", "On " + C.dv.name + ", " + C.m.name + " at " + S.bits + " bits needs " + fmtMs(C.tm) + " per frame. The robot " + (S.pipe ? "keeps the camera running and gets a new command after " + fmtMs(C.T) : "finishes one frame before it takes the next, so it gets a new command after " + fmtMs(C.T)) + ". The camera points up to " + Kit.fmt(C.errDeg, 1) + " degrees behind the shark, against an allowed " + S.tol + ". " + (C.errDeg > S.tol ? "That is too slow for this shark." : "That is fast enough for this shark."));
      setPressed(mChips, S.model === "mobilenet" ? 0 : 1); setPressed(bChips, S.bits === 32 ? 0 : 1); setPressed(dChips, S.dev);
    }
    // the aim animation: the camera points where the shark was one latency ago, updated once per period
    var abar = el("aim-bar"), aplay = Kit.button(abar, "Play the pass", function () { aimPlay(); }); Kit.button(abar, "Reset", function () { if (aim.stop) aim.stop(); aim.t = 0; aim.max = 0; drawAim(); });
    function thS(C, t) { var Xm = S.d * Math.tan(30 * Math.PI / 180); return Math.atan((-Xm + S.v * t / 1000) / S.d); }   // shark angle at time t (ms)
    function thA(C, t) { var k = Math.floor((t - C.L) / C.T); return k < 0 ? thS(C, 0) : thS(C, k * C.T); }
    function aimPlay() {
      if (aim.stop) aim.stop();
      var C = calc(), dur = 2 * S.d * Math.tan(30 * Math.PI / 180) / S.v * 1000, slow = Math.max(1, 3200 / dur), play = dur * slow;
      aim.max = 0; aim.slow = slow;
      aim.stop = Kit.animate(play, function (p) { aim.t = p * dur; var e = Math.abs(thS(C, aim.t) - thA(C, aim.t)) * 180 / Math.PI; if (e > aim.max) aim.max = e; drawAim(); });
    }
    function drawAim() {
      var C = calc(), P = Kit.palette(), svg = el("aim-svg"), W = Kit.width(svg), H = 270;
      svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.innerHTML = "";
      var cx = W / 2, cy = H - 22, Xm = S.d * Math.tan(32 * Math.PI / 180), scale = Math.min((W / 2 - 8) / (1.09 * S.d), (cy - 62) / S.d);
      var py = cy - S.d * scale;
      Kit.el("line", { x1: cx - Xm * scale, x2: cx + Xm * scale, y1: py, y2: py, stroke: P.rule, "stroke-width": 2, "stroke-dasharray": "5 4" }, svg);
      var ts = thS(C, aim.t), ta = thA(C, aim.t), R = S.d * scale / Math.cos(ta) * 1.04;
      var fov = 30 * Math.PI / 180, ax = cx + Math.sin(ta) * R, ay = cy - Math.cos(ta) * R;
      var wedge = "M" + cx + " " + cy + " L" + (cx + Math.sin(ta - fov) * R) + " " + (cy - Math.cos(ta - fov) * R) + " L" + (cx + Math.sin(ta + fov) * R) + " " + (cy - Math.cos(ta + fov) * R) + " Z";
      Kit.el("path", { d: wedge, fill: hexA(P.teal, 0.1), stroke: P.teal, "stroke-width": 1 }, svg);
      Kit.el("line", { x1: cx, y1: cy, x2: ax, y2: ay, stroke: P.teal, "stroke-width": 3 }, svg);
      var sx = cx + S.d * scale * Math.tan(ts), sy = py;
      Kit.el("line", { x1: cx, y1: cy, x2: sx, y2: sy, stroke: P.coral, "stroke-width": 1.8, "stroke-dasharray": "5 3" }, svg);
      Kit.el("ellipse", { cx: sx, cy: sy, rx: 15, ry: 6, fill: P.navy, stroke: P.bg, "stroke-width": 1.5 }, svg);
      Kit.el("path", { d: "M" + (sx - 10) + " " + sy + " l-9 -7 l0 14 z", fill: P.navy }, svg);
      Kit.el("rect", { x: cx - 11, y: cy - 8, width: 22, height: 16, rx: 3, fill: P.ink }, svg);
      var e = Math.abs(ts - ta) * 180 / Math.PI;
      var lab = Kit.el("text", { x: 10, y: 20, "font-size": 14, fill: aim.max > S.tol || e > S.tol ? P.coral : P.ink2, "font-weight": 600 }, svg);
      lab.textContent = "error now " + e.toFixed(1) + "°, largest " + aim.max.toFixed(1) + "°, allowed " + S.tol + "°";
      var l2 = Kit.el("text", { x: W - 10, y: 20, "text-anchor": "end", "font-size": 13, fill: P.muted }, svg); l2.textContent = aim.slow > 1 ? "slowed " + aim.slow.toFixed(0) + " times" : "";
      Kit.el("text", { x: sx, y: sy - 12, "text-anchor": "middle", "font-size": 12, fill: P.ink2 }, svg).textContent = "shark";
      var mx = cx + Math.sin(ta) * R * 0.5, my = cy - Math.cos(ta) * R * 0.5, right = ta >= 0;
      Kit.el("text", { x: right ? Math.min(mx + 8, W - 128) : Math.max(mx - 8, 128), y: my, "text-anchor": right ? "start" : "end", "font-size": 12, fill: P.link, stroke: P.bg, "stroke-width": 3, "paint-order": "stroke" }, svg).textContent = "where the camera points";
      Kit.setText("aim-status", aim.max > 0 ? "Largest pointing error in this pass: " + aim.max.toFixed(1) + " degrees, with an allowed " + S.tol + ". " + (aim.max > S.tol ? "The shark drifted out of the target zone." : "The camera kept up.") : "Press Play the pass: the shark swims across at the speed you set, and the camera aims where the shark was one lag ago.");
    }
    draw(); drawAim(); Kit.onTheme(function () { draw(); drawAim(); });
    window.addEventListener("resize", Kit.debounce(function () { draw(); drawAim(); }));
  }

  // ================================================================== bu: the calculator
  function calcFig() {
    var S = { m: 2, bits: 8, dev: 0, ram: DEVICES[0].ram, speed8: DEVICES[0].speed8, s8: 2.5, need: 10 }, devChips;
    var bar = el("bu-bar"), ctl = el("bu-controls");
    Kit.html("span", { class: "label" }, bar, "Model");
    var mChips = Kit.chips(bar, MODELS.map(function (m) { return m.short; }), 2, function (i) { S.m = i; draw(); });
    var bar2 = el("bu-bar2");
    Kit.html("span", { class: "label" }, bar2, "Bits per weight");
    var bChips = Kit.chips(bar2, ["32", "16", "8", "4"], 2, function (i) { S.bits = [32, 16, 8, 4][i]; draw(); });
    Kit.html("span", { class: "label", style: "margin-left:10px" }, bar2, "Computer");
    devChips = Kit.chips(bar2, ["Pi 4 (4 GB)", "Pi 5 (8 GB)"], 0, function (i) { S.dev = i; S.ram = DEVICES[i].ram; S.speed8 = DEVICES[i].speed8; sSp.set(Math.log10(S.speed8)); draw(); });
    var sSp = Kit.slider(ctl, { label: "Speed at 8 bits, billion multiply-adds per second", min: -0.3, max: 2.3, step: 0.001, value: Math.log10(S.speed8), show: function (v) { return Math.pow(10, v).toFixed(1); } }, function (v) { S.speed8 = Math.pow(10, v); S.dev = -1; draw(); });
    Kit.slider(ctl, { label: "Speed-up of 8 bits over 32 bits", min: 1, max: 4, step: 0.1, value: 2.5, digits: 1, show: function (v) { return v.toFixed(1) + " times"; } }, function (v) { S.s8 = v; draw(); });
    Kit.slider(ctl, { label: "Frames per second the robot needs", min: 1, max: 30, step: 1, value: 10 }, function (v) { S.need = v; draw(); });
    function tms(m, bits) { return timeMs(m.macs, S.speed8, bits, S.s8); }
    function draw() {
      var m = MODELS[S.m], ram = S.ram, mem = memBytes(m.params, S.bits), t = tms(m, S.bits), f = 1000 / t;
      var spd = speedAt(S.speed8, S.bits, S.s8);
      var narrow = Kit.width(el("bu-plot")) < 520;
      Kit.plot("bu-plot", function (P) {
        var xs = MODELS.map(function (q) { return memBytes(q.params, S.bits) / 1e6; }), ys = MODELS.map(function (q) { return 1000 / tms(q, S.bits); });
        return [
          { x: xs, y: ys, mode: "markers+text", text: MODELS.map(function (q) { return q.short; }), textposition: narrow ? ["top right", "top left", "bottom right", "top right", "top left"] : ["top right", "top right", "top right", "bottom right", "top left"],
            textfont: { size: narrow ? 11 : 12, color: P.ink2 }, marker: { size: MODELS.map(function (q, k) { return k === S.m ? 16 : 10; }), color: MODELS.map(function (q, k) { return k === S.m ? P.teal : P.navy; }), line: { color: P.ink, width: MODELS.map(function (q, k) { return k === S.m ? 2 : 0; }) } },
            hovertemplate: "%{text}: %{x:.3g} MB, %{y:.3g} per second<extra></extra>" }
        ];
      }, function (P) {
        return {
          xaxis: { title: { text: "memory for the weights, MB" }, type: "log", range: [-3.2, 5.6], tickangle: 0, automargin: true, tickvals: narrow ? [0.001, 0.1, 10, 1000, 100000] : [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], ticktext: narrow ? ["0.001", "0.1", "10", "1,000", "100,000"] : ["0.001", "0.01", "0.1", "1", "10", "100", "1,000", "10,000", "100,000"] }, yaxis: { title: { text: "frames (or tokens) per second" }, type: "log", range: [-2, 5.4], automargin: true, tickvals: narrow ? [0.01, 1, 100, 10000] : [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000], ticktext: narrow ? ["0.01", "1", "100", "10,000"] : ["0.01", "0.1", "1", "10", "100", "1,000", "10,000", "100,000"] },
          shapes: [
            { type: "line", x0: ram / 1e6, x1: ram / 1e6, y0: 0, y1: 1, yref: "paper", line: { color: P.coral, width: 2, dash: "dash" } },
            { type: "line", x0: 0, x1: 1, xref: "paper", y0: S.need, y1: S.need, line: { color: P.sand, width: 2, dash: "dash" } }
          ],
          annotations: [
            { x: Math.log10(ram / 1e6), y: 0.98, yref: "paper", xanchor: "right", yanchor: "top", text: "memory of the computer", showarrow: false, font: { size: 12, color: P.coral } },
            { x: 0.01, xref: "paper", y: Math.log10(S.need), xanchor: "left", yanchor: "bottom", text: "frames per second needed", showarrow: false, font: { size: 12, color: P.muted } }
          ],
          margin: { l: 56, r: 12, t: 12, b: 46 }
        };
      });
      Kit.setText("bu-mem", fmtBytes(mem)); Kit.setText("bu-memn", Kit.pct(mem / ram, mem / ram < 0.1 ? 2 : 0) + " of the RAM");
      Kit.setText("bu-spd", spd.toFixed(1)); Kit.setText("bu-time", fmtMs(t)); Kit.setText("bu-fps", fmtFps(f));
      Kit.setText("bu-need", f >= S.need ? "enough for " + S.need + " per second" : "below the " + S.need + " per second needed");
      el("bu-fpsbox").className = f >= S.need ? "hit" : "miss"; el("bu-membox").className = mem <= ram ? "hit" : "miss";
      Kit.setText("bu-eq-N", groupInt(m.params)); Kit.setText("bu-eq-b", String(S.bits)); Kit.setText("bu-eq-M", fmtBytes(mem));
      Kit.setText("bu-eq-O", groupInt(m.macs)); Kit.setText("bu-eq-R", spd.toFixed(1)); Kit.setText("bu-eq-t", fmtMs(t)); Kit.setText("bu-eq-f", fmtFps(f));
      var tb = el("bu-table");
      tb.innerHTML = "<thead><tr><th>Model</th><th>Weights</th><th>Multiply-adds per frame or token</th><th>Multiply-adds per weight</th></tr></thead><tbody>" +
        MODELS.map(function (q, k) { return "<tr class='" + (k === S.m ? "pick" : "") + "'><td>" + q.name + "</td><td>" + (q.params >= 1e9 ? (q.params / 1e9).toFixed(0) + " billion" : groupInt(q.params)) + "</td><td>" + (q.macs >= 1e9 ? (q.macs / 1e9).toFixed(2) + " billion" : groupInt(q.macs)) + "</td><td>" + (q.macs / q.params).toFixed(q.macs / q.params < 10 ? 2 : 0) + "</td></tr>"; }).join("") + "</tbody>";
      setPressed(mChips, S.m); setPressed(bChips, [32, 16, 8, 4].indexOf(S.bits)); setPressed(devChips, S.dev);
      Kit.setText("bu-status", m.name + " at " + S.bits + " bits: " + fmtBytes(mem) + " of weights " + (mem <= ram ? "fit" : "do not fit") + " in " + (ram / GB) + " GB, and the computer reaches " + fmtFps(f) + " per second. " + (m.macs / m.params < 2 ? "A weight is used about once per " + m.unit + ", so memory traffic, not arithmetic, is likely to set the actual speed." : ""));
    }
    draw();
  }

  // ================================================================== qz: quantise the digit reader and measure
  function quantFig() {
    var LAY = [{ k: "conv1_w", name: "conv 1" }, { k: "conv2_w", name: "conv 2" }, { k: "fc_w", name: "dense" }];
    var W0 = {}; ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"].forEach(function (k) { W0[k] = Float64Array.from(Q.w[k]); });
    var S = { bits: 8, layer: 2, zoom: false }, BITS = [2, 3, 4, 5, 6, 8, 12, 16, 32];
    var bar = el("qz-bar"), bar2 = el("qz-bar2");
    Kit.html("span", { class: "label" }, bar, "Bits per weight");
    var bChips = Kit.chips(bar, BITS.map(String), 5, function (i) { S.bits = BITS[i]; draw(); });
    Kit.html("span", { class: "label" }, bar2, "Look at the weights of");
    var lChips = Kit.chips(bar2, LAY.map(function (l) { return l.name; }), 2, function (i) { S.layer = i; draw(); });
    Kit.html("span", { class: "label", style: "margin-left:10px" }, bar2, "Accuracy axis");
    var zChips = Kit.chips(bar2, ["Whole range", "Zoom in"], 0, function (i) { S.zoom = i === 1; draw(); });
    var nTest = Q.subset.n, labels = Q.subset.labels.split("").map(Number);
    var cache = {};
    function weightsAt(bits) {
      if (bits >= 32) return W0;
      if (cache[bits]) return cache[bits];
      var P = Object.assign({}, W0);
      LAY.forEach(function (l) { P[l.k] = quantise(W0[l.k], bits).deq; });
      return (cache[bits] = P);
    }
    var accCache = {};
    function liveAcc(bits) {
      if (accCache[bits]) return accCache[bits];
      var P = weightsAt(bits), ok = 0, preds = new Array(nTest);
      for (var i = 0; i < nTest; i++) { var p = argmax(forward(SPRITE.subarray(i * 784, (i + 1) * 784), P)); preds[i] = p; if (p === labels[i]) ok++; }
      return (accCache[bits] = { acc: ok / nTest, preds: preds });
    }
    function hist(vals, lo, hi, nb) {
      var c = new Array(nb).fill(0), w = (hi - lo) / nb;
      for (var i = 0; i < vals.length; i++) { var b = Math.min(nb - 1, Math.max(0, Math.floor((vals[i] - lo) / w))); c[b]++; }
      return { c: c, x: c.map(function (v, k) { return lo + (k + 0.5) * w; }), w: w };
    }
    // digit thumbnails: the first 40 test digits and what the quantised network says
    var grid = el("qz-grid"), thumbs = [];
    for (var g = 0; g < 40; g++) {
      var cell = Kit.html("div", { class: "dg" }, grid), cvs = Kit.html("canvas", { width: 28, height: 28 }, cell), cap = Kit.html("span", {}, cell, "");
      var ctx = cvs.getContext("2d"), im = ctx.createImageData(28, 28);
      for (var q = 0; q < 784; q++) { var v = 255 - Math.round(SPRITE[g * 784 + q] * 255); im.data[q * 4] = im.data[q * 4 + 1] = im.data[q * 4 + 2] = v; im.data[q * 4 + 3] = 255; }
      ctx.putImageData(im, 0, 0); thumbs.push({ cell: cell, cap: cap });
    }
    function draw() {
      var b = S.bits, L = LAY[S.layer], w = W0[L.k], isQ = b < 32, qr = isQ ? quantise(w, b) : null;
      var lo = Math.min.apply(null, w), hi = Math.max.apply(null, w), H = hist(w, lo, hi, 40);
      Kit.plot("qz-w", function (P) {
        return [{ x: H.x, y: H.c, type: "bar", marker: { color: P.navy, opacity: 0.85 }, width: H.w * 0.92, hovertemplate: "%{y} weights near %{x:.2f}<extra></extra>" }];
      }, function (P) {
        var shapes = [];
        if (isQ && b <= 5) { var seen = {}; for (var i = 0; i < w.length; i++) { var lv = qr.deq[i]; if (!seen[lv.toFixed(6)]) { seen[lv.toFixed(6)] = 1; shapes.push({ type: "line", x0: lv, x1: lv, y0: 0, y1: 1, yref: "paper", line: { color: P.coral, width: 1.5 } }); } } }
        return { xaxis: { title: { text: "weight value, " + L.name }, range: [lo - 0.05, hi + 0.05] }, yaxis: { title: { text: "number of weights" } }, shapes: shapes, bargap: 0.05,
          annotations: [{ x: 0.01, xref: "paper", y: 0.98, yref: "paper", xanchor: "left", yanchor: "top", showarrow: false, font: { size: 12, color: P.coral }, text: isQ && b <= 5 ? (Math.pow(2, b) + " allowed values (coral lines)") : isQ ? (Math.pow(2, b).toLocaleString("en-US") + " allowed values, too close to draw") : "32 bits: a different value for nearly all weights" }],
          margin: { l: 52, r: 12, t: 12, b: 46 } };
      });
      if (isQ) {
        var err = new Float64Array(w.length), mx = 0;
        for (var j = 0; j < w.length; j++) { err[j] = w[j] - qr.deq[j]; if (Math.abs(err[j]) > mx) mx = Math.abs(err[j]); }
        var E = hist(err, -qr.s / 2, qr.s / 2, 20);
        Kit.plot("qz-e", function (P) { return [{ x: E.x, y: E.c, type: "bar", marker: { color: P.coral, opacity: 0.85 }, width: E.w * 0.92, hovertemplate: "%{y} weights with error near %{x:.4f}<extra></extra>" }]; },
          function (P) { return { xaxis: { title: { text: "rounding error" }, range: [-qr.s * 0.6, qr.s * 0.6], exponentformat: "none" }, yaxis: { title: { text: "number of weights" } }, bargap: 0.05,
            shapes: [{ type: "line", x0: -qr.s / 2, x1: -qr.s / 2, y0: 0, y1: 1, yref: "paper", line: { color: P.muted, width: 1.5, dash: "dot" } }, { type: "line", x0: qr.s / 2, x1: qr.s / 2, y0: 0, y1: 1, yref: "paper", line: { color: P.muted, width: 1.5, dash: "dot" } }],
            annotations: [{ x: 0.5, xref: "paper", y: 1.0, yref: "paper", yanchor: "bottom", showarrow: false, font: { size: 12, color: P.muted }, text: "no error can exceed half a step (" + (qr.s / 2).toPrecision(2) + ")" }], margin: { l: 52, r: 12, t: 24, b: 46 } }; });
        Kit.setText("qz-eq-w", Kit.fmt(w[0], 4)); Kit.setText("qz-eq-s", qr.s.toPrecision(3)); Kit.setText("qz-eq-z", String(qr.z)); Kit.setText("qz-eq-q", String(qr.q[0]));
        Kit.setText("qz-eq-wh", Kit.fmt(qr.deq[0], 4)); Kit.setText("qz-eq-e", Kit.fmt(w[0] - qr.deq[0], 4)); Kit.setText("qz-eq-max", mx.toPrecision(2));
      } else {
        Kit.plot("qz-e", [], function (P) { return { xaxis: { visible: false }, yaxis: { visible: false }, annotations: [{ x: 0.5, y: 0.5, xref: "paper", yref: "paper", showarrow: false, text: "32 bits is the reference: no rounding error here", font: { size: 14, color: P.muted } }], margin: { l: 10, r: 10, t: 10, b: 10 } }; });
        ["w", "s", "z", "q", "wh", "e", "max"].forEach(function (k) { Kit.setText("qz-eq-" + k, "n/a"); });
        Kit.setText("qz-eq-w", Kit.fmt(w[0], 4));
      }
      var R = Q.results, idx = R.bits.indexOf(b), live = liveAcc(b);
      Kit.setText("qz-mem", fmtBytes(3702 * Math.min(b, 32) / 8 + 112)); Kit.setText("qz-live", Kit.pct(live.acc, 1));
      Kit.setText("qz-off", idx >= 0 ? Kit.pct(R.w_acc[idx], 2) : Kit.pct(Q.base_acc, 2));
      Kit.setText("qz-flips", idx >= 0 ? R.w_flips[idx].toLocaleString("en-US") + " of 10,000" : "0 of 10,000");
      // accuracy chart
      var narrowQ = Kit.width(el("qz-acc")) < 520;
      Kit.plot("qz-acc", function (P) {
        var xs = R.bits.concat([17]), y1 = R.w_acc.map(function (v) { return 100 * v; }).concat([100 * Q.base_acc]), y2 = R.wa_acc.map(function (v) { return 100 * v; }).concat([100 * Q.base_acc]);
        var cx = b >= 32 ? 17 : b, cy = b >= 32 ? 100 * Q.base_acc : 100 * R.w_acc[idx];
        return [
          { x: xs, y: y2, name: "weights and activations rounded", mode: "lines+markers", line: { color: P.sand, width: 2, dash: "dot" }, marker: { size: 6, color: P.sand }, hovertemplate: "%{x} bits, weights and activations: %{y:.2f}%<extra></extra>" },
          { x: xs, y: y1, name: "weights rounded", mode: "lines+markers", line: { color: P.navy, width: 2.5 }, marker: { size: 7, color: P.navy }, hovertemplate: "%{x} bits, weights only: %{y:.2f}%<extra></extra>" },
          { x: [cx], y: [cy], name: "your choice", mode: "markers", marker: { size: 16, color: "rgba(0,0,0,0)", line: { color: P.teal, width: 3 } }, hoverinfo: "skip" }
        ];
      }, function (P) {
        return { xaxis: { title: { text: "bits per weight" }, tickvals: narrowQ ? [2, 4, 8, 12, 17] : [2, 4, 6, 8, 12, 16, 17], ticktext: narrowQ ? ["2", "4", "8", "12", "32"] : ["2", "4", "6", "8", "12", "16", "32"], range: [1.5, 17.6] },
          yaxis: { title: { text: "accuracy, %" }, range: S.zoom ? [96, 99.5] : [0, 104] }, showlegend: true, legend: { orientation: "h", y: -0.4, font: { size: 13 } },
          annotations: [{ x: 17, y: 100 * Q.base_acc, xanchor: "right", yanchor: "top", yshift: -10, showarrow: false, text: "32 bits: " + (100 * Q.base_acc).toFixed(2) + "%", font: { size: 12, color: P.muted } }], margin: { l: 56, r: 12, t: 12, b: 100 } };
      });
      // digits
      thumbs.forEach(function (t, i) {
        var p = live.preds[i], bad = p !== labels[i];
        t.cap.textContent = "" + p; t.cell.className = "dg" + (bad ? " bad" : ""); t.cell.setAttribute("title", "digit " + labels[i] + ", read as " + p);
      });
      var nb = thumbs.filter(function (t, i) { return live.preds[i] !== labels[i]; }).length;
      Kit.setText("qz-bad", nb + (nb === 1 ? " of the first 40 digits is misread" : " of the first 40 digits are misread") + ". The label under a picture is the network's answer.");
      setPressed(bChips, BITS.indexOf(b)); setPressed(lChips, S.layer); setPressed(zChips, S.zoom ? 1 : 0);
      Kit.setText("qz-status", b >= 32 ? "The reference: 3,702 weights in 32 bits, " + (100 * Q.base_acc).toFixed(2) + "% on the 10,000 test digits." :
        b + " bits: " + fmtBytes(3702 * b / 8 + 112) + " of weights against " + fmtBytes(3702 * 4 + 112) + " at 32 bits. Accuracy on all 10,000 test digits is " + (idx >= 0 ? (100 * R.w_acc[idx]).toFixed(2) : "") + "% and " + (idx >= 0 ? R.w_flips[idx].toLocaleString("en-US") : 0) + " digits change their answer. On the 400 digits loaded in your browser it is " + (100 * live.acc).toFixed(1) + "% (" + (100 * liveAcc(32).acc).toFixed(1) + "% at 32 bits). A gap of one digit in 400 is 0.25 points, so small sample differences are noise.");
    }
    draw();
  }

  // ================================================================== ld: the ladder of model sizes
  function ladderFig() {
    var BITS = [32, 16, 8, 4], show = -1, bar = el("ld-bar");
    Kit.html("span", { class: "label" }, bar, "Show");
    var chips = Kit.chips(bar, ["All four", "32 bits", "16 bits", "8 bits", "4 bits"], 0, function (i) { show = i - 1; draw(); });
    function draw() {
      Kit.plot("ld-plot", function (P) {
        var cols = [P.navy, P.teal, P.sand, P.coral];
        return BITS.map(function (b, k) {
          return { x: MODELS.map(function (m) { return m.short; }), y: MODELS.map(function (m) { return memBytes(m.params, b) / 1e6; }), type: "bar", name: b + " bits", marker: { color: cols[k], opacity: show >= 0 && show !== k ? 0.12 : 0.95 },
            hovertemplate: "%{x}, " + b + " bits: %{y:,.3g} MB<extra></extra>" };
        });
      }, function (P) {
        var lines = [[4e3, "4 GB"], [8e3, "8 GB"], [16e3, "16 GB"]];
        return { barmode: "group", showlegend: true, legend: { orientation: "h", y: Kit.width(el("ld-plot")) < 520 ? -0.4 : -0.22, font: { size: 13 } },
          xaxis: { title: { text: "" }, tickfont: { size: 12 }, automargin: true }, yaxis: { title: { text: "memory for the weights, MB" }, type: "log", range: [-3.5, 6], tickvals: [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000], ticktext: ["0.001", "0.01", "0.1", "1", "10", "100", "1,000", "10,000", "100,000", "1,000,000"] },
          shapes: lines.map(function (l) { return { type: "line", x0: 0, x1: 1, xref: "paper", y0: l[0], y1: l[0], yref: "y", line: { color: P.muted, width: 1.5, dash: "dash" } }; }),
          annotations: [{ x: 0.005, xref: "paper", y: Math.log10(16e3), yref: "y", xanchor: "left", yanchor: "bottom", showarrow: false, text: "dashed: 4, 8, 16 GB of memory", font: { size: 12, color: P.muted } }],
          margin: { l: 66, r: 12, t: 12, b: Kit.width(el("ld-plot")) < 520 ? 120 : 64 } };
      });
      el("ld-table").innerHTML = "<thead><tr><th>Model</th><th>Weights</th>" + BITS.map(function (b) { return "<th>" + b + " bits</th>"; }).join("") + "</tr></thead><tbody>" +
        MODELS.map(function (m) { return "<tr><td>" + m.name + "</td><td>" + (m.params >= 1e9 ? (m.params / 1e9).toFixed(0) + " billion" : groupInt(m.params)) + "</td>" + BITS.map(function (b) { var by = memBytes(m.params, b); return "<td class='" + (by <= 4 * GB ? "okc" : "nocc") + "'>" + fmtBytes(by) + "</td>"; }).join("") + "</tr>"; }).join("") + "</tbody>";
      setPressed(chips, show + 1);
    }
    draw();
  }
})();
