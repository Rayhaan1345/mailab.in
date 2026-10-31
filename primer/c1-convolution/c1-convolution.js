// C1 Images as numbers and convolution.
//   pixelFig    a handwritten tag digit as a grid of numbers, with a magnifier, paint mode and a pixel shuffle
//   channelFig  a shark photo as three grids (red, green, blue), a depth slider, and a live weight count
//   slideFig    SIGNATURE: a filter slides over the digit and fills a feature map, with the dot product in boxes
//   sizeFig     stride and padding sliders with the output-size formula
//   poolFig     max (or average) pooling and its small tolerance to shifts
// Pure functions are exported with module.exports for the node test (conv2d, outSize, pool2d, degradeData).
(function () {
  "use strict";

  // ---------- pure maths ----------
  function outSize(n, k, s, p) { var t = n + 2 * p - k; return t < 0 ? 0 : Math.floor(t / s) + 1; }
  // Cross-correlation of a square image (n by n, row-major array) with a k by k kernel, as deep-learning libraries call convolution.
  function conv2d(img, n, ker, k, s, p) {
    s = s || 1; p = p || 0;
    var m = outSize(n, k, s, p), out = new Float32Array(m * m), i, j, u, v, acc, r, c;
    for (i = 0; i < m; i++) for (j = 0; j < m; j++) {
      acc = 0;
      for (u = 0; u < k; u++) for (v = 0; v < k; v++) {
        r = i * s + u - p; c = j * s + v - p;
        if (r >= 0 && r < n && c >= 0 && c < n) acc += ker[u * k + v] * img[r * n + c];
      }
      out[i * m + j] = acc;
    }
    return { out: out, m: m };
  }
  // Pooling with block size q and stride st over an m by m map. kind: "max" or "avg".
  function pool2d(map, m, q, st, kind) {
    var o = outSize(m, q, st, 0), out = new Float32Array(o * o), i, j, u, v, best, sum;
    for (i = 0; i < o; i++) for (j = 0; j < o; j++) {
      best = -Infinity; sum = 0;
      for (u = 0; u < q; u++) for (v = 0; v < q; v++) { var x = map[(i * st + u) * m + j * st + v]; if (x > best) best = x; sum += x; }
      out[i * o + j] = kind === "avg" ? sum / (q * q) : best;
    }
    return { out: out, m: o };
  }
  function shiftImg(img, n, dx) {
    var out = new Float32Array(n * n), i, j;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) { var jj = j - dx; if (jj >= 0 && jj < n) out[i * n + j] = img[i * n + jj]; }
    return out;
  }
  function relu(a) { var o = new Float32Array(a.length), i; for (i = 0; i < a.length; i++) o[i] = a[i] > 0 ? a[i] : 0; return o; }
  function relDiff(a, b) { var d = 0, nb = 0, i; for (i = 0; i < a.length; i++) { d += (a[i] - b[i]) * (a[i] - b[i]); nb += b[i] * b[i]; } return nb > 0 ? Math.sqrt(d / nb) : 0; }

  // The depth model of depth.py (Jaffe-style: colour loss, darkening with auto exposure, haze), pixel by pixel on 0..255 sRGB arrays.
  var K_RGB = [0.54, 0.256, 0.209], BETA = 0.06, VEIL = [0.08, 0.50, 0.55], EXPO = 0.6;
  var LIN = new Float32Array(256);
  (function () { for (var v = 0; v < 256; v++) { var x = v / 255; LIN[v] = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); } })();
  function toSrgb(x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return 255 * (x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055); }
  function degradeData(src, dst, z, o) {
    o = o || {};
    var k = o.k || K_RGB, beta = o.beta != null ? o.beta : BETA, veil = o.veil || VEIL, expo = o.expo != null ? o.expo : EXPO;
    var kmin = Math.min(k[0], k[1], k[2]), kmean = (k[0] + k[1] + k[2]) / 3;
    var V = 1 - Math.exp(-beta * z), dark = Math.exp(-kmin * z), gain = Math.exp(expo * kmean * z);
    var mult = [0, 1, 2].map(function (c) { return Math.exp(-(k[c] - kmin) * z) * dark * gain; });
    var veilV = [0, 1, 2].map(function (c) { return veil[c] * V; });
    var bright = o.bright != null ? o.bright : 1;
    for (var i = 0; i < src.length; i += 4) {
      for (var c = 0; c < 3; c++) dst[i + c] = Math.round(toSrgb((LIN[src[i + c]] * (1 - V) + veilV[c]) * mult[c] * bright));
      dst[i + 3] = 255;
    }
  }

  if (typeof module !== "undefined") { module.exports = { outSize: outSize, conv2d: conv2d, pool2d: pool2d, shiftImg: shiftImg, relu: relu, relDiff: relDiff, degradeData: degradeData }; return; }

  // ---------- helpers for the page ----------
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function mixRGB(a, b, t) { var A = hexRGB(a), B = hexRGB(b); return [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }); }
  function css(c) { return "rgb(" + c[0] + "," + c[1] + "," + c[2] + ")"; }
  function lum(c) { return (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255; }
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function decode(b64) { var s = atob(b64), a = new Float32Array(784), i; for (i = 0; i < 784; i++) a[i] = s.charCodeAt(i); return a; }
  function num(v, d) { return Kit.fmt(v, d == null ? 0 : d); }
  function setPressed(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function label(parent, text, cls) { return Kit.html("label", cls ? { class: cls } : {}, parent, text); }
  // Canvas that fits its CSS width, with crisp cells.
  function prep(cvs, nx, ny, cell) { cvs.width = nx * cell; cvs.height = ny * cell; return cvs.getContext("2d"); }
  function inkColour(P, v) { return css(mixRGB(P.paper, P.ink, Math.min(1, Math.max(0, v / 255)))); }
  function divColour(P, v, vmax) {
    if (vmax <= 0) return P.paper;
    var t = Math.max(-1, Math.min(1, v / vmax));
    return css(t >= 0 ? mixRGB(P.paper, P.teal, t) : mixRGB(P.paper, P.coral, -t));
  }

  var DIGITS = null, TAGS = { "Tag 4127": [4, 1, 2, 7], "Tag 3305": [3, 3, 0, 5], "Tag 5519": [5, 5, 1, 9] };
  function digitArr(label, which) {
    var seen = 0, i;
    for (i = 0; i < DIGITS.length; i++) if (DIGITS[i].label === label) { if (seen === (which || 0)) return decode(DIGITS[i].b64); seen++; }
    return new Float32Array(784);
  }

  document.addEventListener("DOMContentLoaded", function () {
    Kit.data("data/digits.json").then(function (d) {
      DIGITS = d.digits;
      pixelFig(); slideFig(); sizeFig(); poolFig();
    }).catch(function (e) { console.error(e); });
    channelFig();
  });

  // ---------- 1. a digit as a grid of numbers ----------
  function pixelFig() {
    var N = 28, CELL = 20, cvs = Kit.$("px-cvs"), ctx = prep(cvs, N, N, CELL);
    var tag = "Tag 4127", pos = 0, mode = 0, shuffled = false, hover = null, img = null, orig = null;
    var perm = (function () { var r = Kit.rng(42), a = [], i, j, t; for (i = 0; i < 784; i++) a.push(i); for (i = 783; i > 0; i--) { j = Math.floor(r() * (i + 1)); t = a[i]; a[i] = a[j]; a[j] = t; } return a; })();
    var bar = Kit.$("px-bar"), tagBar = Kit.html("div", { class: "bar-row" }, bar), posBar = Kit.html("div", { class: "bar-row" }, bar), modeBar = Kit.html("div", { class: "bar-row" }, bar);
    var tagChips = Kit.chips(tagBar, Object.keys(TAGS), 0, function (i) { tag = Object.keys(TAGS)[i]; load(); buildPos(); });
    var posChips = [];
    function buildPos() {
      posBar.innerHTML = ""; posChips = Kit.chips(posBar, TAGS[tag].map(function (d, i) { return "digit " + (i + 1) + ": " + d; }), pos, function (i) { pos = i; load(); });
    }
    Kit.chips(modeBar, ["Look", "Add ink", "Erase"], 0, function (i) { mode = i; cvs.style.touchAction = i ? "none" : "auto"; cvs.style.cursor = i ? "cell" : "crosshair"; });
    var shufLab = label(modeBar, "", "check"), shuf = Kit.html("input", { type: "checkbox" }, shufLab); shufLab.appendChild(document.createTextNode(" Shuffle the squares"));
    shuf.addEventListener("change", function () { shuffled = shuf.checked; draw(); });
    Kit.button(modeBar, "Reset", function () { load(); });
    function load() { orig = digitArr(TAGS[tag][pos], 0); img = new Float32Array(orig); draw(); }
    function shown() { if (!shuffled) return img; var o = new Float32Array(784), i; for (i = 0; i < 784; i++) o[i] = img[perm[i]]; return o; }
    function cellAt(e) {
      var r = cvs.getBoundingClientRect(), x = Math.floor((e.clientX - r.left) / r.width * N), y = Math.floor((e.clientY - r.top) / r.height * N);
      return x < 0 || y < 0 || x >= N || y >= N ? null : [y, x];
    }
    function paint(e) {
      if (!mode || shuffled) return;
      var c = cellAt(e); if (!c) return;
      img[c[0] * N + c[1]] = mode === 1 ? 255 : 0; draw();
    }
    cvs.addEventListener("pointermove", function (e) { hover = cellAt(e); if (e.buttons) paint(e); else draw(); });
    cvs.addEventListener("pointerdown", function (e) { hover = cellAt(e); if (mode && !shuffled) { try { cvs.setPointerCapture(e.pointerId); } catch (err) {} paint(e); } else draw(); });
    cvs.addEventListener("pointerleave", function () { hover = null; draw(); });
    cvs.addEventListener("keydown", function (e) {
      var d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key]; if (!d) return;
      e.preventDefault(); var h = hover || [14, 14]; hover = [Math.max(0, Math.min(27, h[0] + d[0])), Math.max(0, Math.min(27, h[1] + d[1]))]; draw();
    });
    function draw() {
      var P = Kit.palette(), a = shown(), i, j;
      for (i = 0; i < N; i++) for (j = 0; j < N; j++) { ctx.fillStyle = inkColour(P, a[i * N + j]); ctx.fillRect(j * CELL, i * CELL, CELL, CELL); }
      ctx.strokeStyle = P.rule; ctx.lineWidth = 1; ctx.globalAlpha = 0.5;
      for (i = 0; i <= N; i++) { ctx.beginPath(); ctx.moveTo(0, i * CELL + 0.5); ctx.lineTo(N * CELL, i * CELL + 0.5); ctx.stroke(); ctx.beginPath(); ctx.moveTo(i * CELL + 0.5, 0); ctx.lineTo(i * CELL + 0.5, N * CELL); ctx.stroke(); }
      ctx.globalAlpha = 1;
      var fr = hover || [14, 14]; ctx.strokeStyle = P.coral; ctx.lineWidth = 3; ctx.setLineDash(hover ? [] : [6, 4]); ctx.strokeRect((fr[1] - 2) * CELL + 1.5, (fr[0] - 2) * CELL + 1.5, 5 * CELL - 3, 5 * CELL - 3); ctx.setLineDash([]);
      magnifier(P, a);
      var zeros = 0; for (i = 0; i < 784; i++) if (a[i] === 0) zeros++;
      Kit.$("px-eqn").innerHTML = ebox("c-navy", "28", "rows") + eop("&times;") + ebox("c-navy", "28", "columns") + eop("&times;") + ebox("c-sand", "1", "channel (grey)") + eop("=") + ebox("c-coral", "784", "numbers") + eop("|") +
        ebox("c-teal", String(zeros), "of them are 0") + eop("=") + ebox("c-teal", Kit.pct(zeros / 784, 0), "of the picture is empty paper");
    }
    function magnifier(P, a) {
      var box = Kit.$("px-mag"), h = hover || [14, 14], t = '<table aria-label="The 5 by 5 squares around the pointer, with their values">', i, j;
      for (i = -2; i <= 2; i++) {
        t += "<tr>";
        for (j = -2; j <= 2; j++) {
          var r = h[0] + i, c = h[1] + j, ok = r >= 0 && r < N && c >= 0 && c < N, v = ok ? a[r * N + c] : null;
          var bg = ok ? mixRGB(P.paper, P.ink, v / 255) : mixRGB(P.paper, P.paper, 0), fg = ok && lum(bg) < 0.5 ? "#fff" : "#17191c";
          if (!ok) fg = P.muted;
          t += '<td' + (i === 0 && j === 0 ? ' class="mid"' : "") + ' style="background:' + css(bg) + ";color:" + fg + '">' + (ok ? v : "") + "</td>";
        }
        t += "</tr>";
      }
      box.innerHTML = t + "</table>";
      var rd = hover ? "Row " + hover[0] + ", column " + hover[1] + ": value " + a[hover[0] * N + hover[1]] : "Move over the picture, or tap a square. Arrow keys also work.";
      Kit.$("px-read").textContent = rd;
    }
    load(); buildPos();
    Kit.onTheme(draw);
  }

  // ---------- 2. colour channels of a shark photo ----------
  function channelFig() {
    var PH = [
      { file: "../data/sharks/photos/hammerhead-1.jpg", name: "Scalloped hammerhead", credit: "Scalloped hammerhead, Jarvis Island. Photo: Kevin Lino, NOAA, public domain, via Wikimedia Commons." },
      { file: "../data/sharks/photos/bamboo-1.jpg", name: "Young grey bamboo shark", credit: "Young grey bamboo shark in an aquarium. Photo: Mbrickn, CC BY 4.0, via Wikimedia Commons." },
      { file: "../data/sharks/photos/spadenose-1.jpg", name: "Spadenose shark", credit: "A spadenose shark laid on a white board with a ruler for scale. Photo: Hariprasad Shetty10, CC0, via Wikimedia Commons." }
    ];
    var W = 320, cur = 0, depth = 0, on = [true, true, true], src = null, H = 0, probe = null, k = 3, nf = 16;
    var cv = ["mix", "r", "g", "b"].map(function (n) { return Kit.$("ch-" + n); });
    var ctxs = [], base = document.createElement("canvas"), bctx = base.getContext("2d");
    var bar = Kit.$("ch-bar"), ctrl = Kit.$("ch-controls");
    Kit.chips(bar, PH.map(function (p) { return p.name; }), 0, function (i) { cur = i; load(); });
    ["Red", "Green", "Blue"].forEach(function (n, i) {
      var l = label(bar, "", "check"), c = Kit.html("input", { type: "checkbox", checked: "checked" }, l); l.appendChild(document.createTextNode(" " + n)); l.className = "check ch-" + i;
      c.addEventListener("change", function () { on[i] = c.checked; draw(); });
    });
    var dsl = Kit.slider(ctrl, { label: "Depth, m", min: 0, max: 20, step: 0.5, value: 0, digits: 1 }, function (v) { depth = v; draw(); });
    var ksel = Kit.select(ctrl, "Filter size", [{ value: 1, label: "1 by 1" }, { value: 3, label: "3 by 3" }, { value: 5, label: "5 by 5" }, { value: 7, label: "7 by 7" }], 3, function (v) { k = +v; count(); });
    var fsl = Kit.slider(ctrl, { label: "Number of filters", min: 1, max: 64, step: 1, value: 16, digits: 0 }, function (v) { nf = v; count(); });
    function load() {
      var im = new Image(); im.onload = function () {
        H = Math.round(W * im.naturalHeight / im.naturalWidth); base.width = W; base.height = H; bctx.drawImage(im, 0, 0, W, H); src = bctx.getImageData(0, 0, W, H);
        cv.forEach(function (c, i) { c.width = W; c.height = H; ctxs[i] = c.getContext("2d"); });
        draw();
      };
      im.src = PH[cur].file; Kit.$("ch-credit").textContent = PH[cur].credit + " Depth effect made with depth.py, a simple model.";
    }
    var out = null;
    function draw() {
      if (!src) return;
      out = out && out.data.length === src.data.length ? out : new ImageData(W, H);
      degradeData(src.data, out.data, depth);
      var d = out.data, n = W * H, i, c;
      var planes = [0, 1, 2, 3].map(function () { return new ImageData(W, H); });
      var sums = [0, 0, 0];
      for (i = 0; i < n; i++) {
        for (c = 0; c < 3; c++) {
          var v = d[4 * i + c]; sums[c] += v;
          planes[0].data[4 * i + c] = on[c] ? v : 0;
          planes[c + 1].data[4 * i + c] = v; // plane tinted with its own colour
        }
        planes[0].data[4 * i + 3] = 255;
        for (c = 1; c < 4; c++) planes[c].data[4 * i + 3] = 255;
      }
      for (c = 0; c < 4; c++) ctxs[c].putImageData(planes[c], 0, 0);
      var means = sums.map(function (s) { return s / n; });
      Kit.$("ch-means").innerHTML = ebox("c-coral", means[0].toFixed(0), "mean red, of 255") + eop("|") + ebox("c-teal", means[1].toFixed(0), "mean green") + eop("|") + ebox("c-navy", means[2].toFixed(0), "mean blue") + eop("|") + ebox("c-sand", Kit.pct(means[0] / Math.max(1, means[2]), 0), "red as a share of blue");
      if (probe) mark(probe[0], probe[1], d);
      else readout(null);
    }
    function readout(px) {
      var box = Kit.$("ch-px");
      if (!px) { box.innerHTML = '<span class="hintline">Move over any picture to read one pixel.</span>'; return; }
      box.innerHTML = ebox("c-sand", "row " + px.r + ", col " + px.c, "pixel position") + eop("=") + ebox("c-coral", px.v[0], "red") + eop(",") + ebox("c-teal", px.v[1], "green") + eop(",") + ebox("c-navy", px.v[2], "blue");
    }
    function mark(x, y, d) {
      var P = Kit.palette(), i = (y * W + x) * 4;
      cv.forEach(function (c, j) { c.classList.add("probe"); });
      readout({ r: y, c: x, v: [d[i], d[i + 1], d[i + 2]] });
      cv.forEach(function (c, j) { var cx = ctxs[j]; cx.strokeStyle = "#fff"; cx.lineWidth = 2; cx.strokeRect(x - 5, y - 5, 10, 10); cx.strokeStyle = P.coral; cx.lineWidth = 1; cx.strokeRect(x - 6, y - 6, 12, 12); });
    }
    cv.forEach(function (c) {
      function at(e) { var r = c.getBoundingClientRect(); return [Math.max(0, Math.min(W - 1, Math.floor((e.clientX - r.left) / r.width * W))), Math.max(0, Math.min(H - 1, Math.floor((e.clientY - r.top) / r.height * H)))]; }
      function go(e) { var p = at(e); probe = p; draw(); }
      c.addEventListener("pointermove", go); c.addEventListener("pointerdown", go);
    });
    function count() {
      var w = k * k * 3 * nf, tot = (k * k * 3 + 1) * nf;
      Kit.$("ch-eqn").innerHTML = ebox("c-navy", k + "&times;" + k, "filter size") + eop("&times;") + ebox("c-coral", "3", "channels in") + eop("+") + ebox("c-sand", "1", "bias") + eop("=") + ebox("c-teal", String(k * k * 3 + 1), "numbers in one filter") + eop("&times;") + ebox("c-navy", String(nf), "filters") + eop("=") + ebox("c-coral", tot.toLocaleString("en-US"), "numbers to learn");
    }
    load(); count();
    Kit.onTheme(draw);
  }

  // ---------- 3. a filter slides (signature) ----------
  var PRESETS = [
    { id: "lr", name: "Edge left-right", w: [-1, 0, 1, -2, 0, 2, -1, 0, 1], note: "Right column minus left column. Flat patches give 0." },
    { id: "tb", name: "Edge top-bottom", w: [-1, -2, -1, 0, 0, 0, 1, 2, 1], note: "Bottom row minus top row. Flat patches give 0." },
    { id: "blur", name: "Blur", w: [1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9], note: "The average of the nine squares." },
    { id: "sharp", name: "Sharpen", w: [0, -1, 0, -1, 5, -1, 0, -1, 0], note: "Five times the middle minus its four neighbours." },
    { id: "id", name: "Identity", w: [0, 0, 0, 0, 1, 0, 0, 0, 0], note: "Copies the middle square." },
    { id: "own", name: "Your own", w: [0, 1, 0, 1, -4, 1, 0, 1, 0], note: "Type nine numbers. The start is a known edge finder (it adds the four neighbours and subtracts four times the middle)." }
  ];
  function wfmt(v) { var a = Math.abs(v); var s = a >= 100 ? a.toFixed(0) : a >= 10 ? a.toFixed(1) : (Math.round(a * 100) / 100).toString(); return (v < 0 ? "−" : "") + s; }

  function slideFig() {
    var N = 28, M = 26, CELL = 12, cin = Kit.$("sl-in"), cout = Kit.$("sl-out"), ci = prep(cin, N, N, CELL), co = prep(cout, M, M, CELL);
    var digit = 4, which = 0, preset = 0, w = PRESETS[0].w.slice(), relu_ = false, img = digitArr(digit, 0), full = null, vmax = 1;
    var cur = 0, revealed = M * M, playing = false, speed = 60, raf = 0, last = 0, acc = 0;
    var bar = Kit.$("sl-bar"), bar2 = Kit.$("sl-bar2");
    var dBar = Kit.html("div", { class: "bar-row" }, bar), fBar = Kit.html("div", { class: "bar-row" }, bar);
    var dchips = Kit.chips(dBar, ["4", "1", "2", "7", "3", "5"].map(function (d) { return "digit " + d; }), 0, function (i) { digit = +["4", "1", "2", "7", "3", "5"][i]; img = digitArr(digit, 0); compute(true); });
    var fchips = Kit.chips(fBar, PRESETS.map(function (p) { return p.name; }), 0, function (i) { pickFilter(i); });
    var bPlay = Kit.button(bar2, "Play", function () { if (playing) stop(); else start(); });
    Kit.button(bar2, "Step", function () { stop(); advance(1); });
    Kit.button(bar2, "Step a row", function () { stop(); advance(M); });
    Kit.button(bar2, "Show all", function () { stop(); revealed = M * M; if (cur < 0) cur = 0; draw(); });
    Kit.button(bar2, "Clear", function () { stop(); cur = 0; revealed = 0; draw(); });
    var pace = Kit.html("label", { class: "pace" }, bar2); pace.appendChild(document.createTextNode("speed "));
    var sp = Kit.html("input", { type: "range", min: 1, max: 4, step: 0.05, value: 1.78 }, pace); var spo = Kit.html("output", {}, pace);
    function setSpeed() { speed = Math.round(Math.pow(10, +sp.value)); spo.textContent = speed + " squares per second"; }
    sp.addEventListener("input", setSpeed); setSpeed();
    var rl = label(bar2, "", "check"), rlc = Kit.html("input", { type: "checkbox" }, rl); rl.appendChild(document.createTextNode(" then ReLU"));
    rlc.addEventListener("change", function () { relu_ = rlc.checked; compute(false); });

    // the equation row: pixels under the window, the editable weights, the products and their sum (built once, then updated)
    var fin = [], pixTd = [], prdTd = [], sumBox, reluBox;
    (function () {
      var seq = Kit.html("div", { class: "seq" }, Kit.$("sl-math"));
      function part(title, cls) { var d = Kit.html("div", { class: "part" + (cls ? " " + cls : "") }, seq); Kit.html("div", { class: "gh" }, d, title); return d; }
      function table(parent, cls, cells, inputs) {
        var t = Kit.html("table", { class: "g3 " + cls }, parent), r, c;
        for (r = 0; r < 3; r++) {
          var tr = Kit.html("tr", {}, t);
          for (c = 0; c < 3; c++) {
            var td = Kit.html("td", {}, tr), idx = r * 3 + c;
            if (inputs) {
              var inp = Kit.html("input", { type: "number", step: "any", "aria-label": "weight row " + (r + 1) + ", column " + (c + 1) }, td); fin.push(inp);
              (function (idx, inp) { inp.addEventListener("input", function () { var v = parseFloat(inp.value); if (isFinite(v)) { w[idx] = v; if (preset !== 5) { preset = 5; setPressed(fchips, 5); Kit.$("sl-note").textContent = PRESETS[5].note; } compute(false); } }); })(idx, inp);
            } else cells.push(td);
          }
        }
      }
      table(part("pixels under the window"), "pix", pixTd);
      Kit.html("div", { class: "opx" }, seq, "\u00d7");
      table(part("filter weights (type to edit)"), "inp", null, true);
      Kit.html("div", { class: "opx" }, seq, "=");
      table(part("nine products"), "prd", prdTd);
      Kit.html("div", { class: "opx" }, seq, "\u03a3");
      var res = part("sum of the nine", "res"); sumBox = Kit.html("div", { class: "big" }, res); reluBox = Kit.html("div", { class: "gh" }, res);
    })();
    function pickFilter(i) {
      preset = i; if (i !== 5) w = PRESETS[i].w.slice(); setPressed(fchips, i); Kit.$("sl-note").textContent = PRESETS[i].note; syncInputs(); compute(false);
    }
    function syncInputs() { fin.forEach(function (inp, i) { inp.value = Math.round(w[i] * 1000) / 1000; }); }

    function compute(newDigit) {
      full = conv2d(img, N, w, 3, 1, 0);
      var o = full.out, i, mx = 0, bi = 0;
      var shownV = relu_ ? relu(o) : o;
      for (i = 0; i < o.length; i++) { var a = Math.abs(shownV[i]); if (a > mx) { mx = a; bi = i; } }
      vmax = mx || 1;
      if (newDigit) { cur = bi; revealed = M * M; }
      draw();
    }
    function advance(n) { if (revealed >= M * M && cur >= M * M - 1 && n > 0) { cur = 0; revealed = 0; } var target = Math.min(M * M, Math.max(revealed, 0) + n); revealed = target; cur = Math.max(0, target - 1); draw(); }
    function start() {
      if (revealed >= M * M) { cur = 0; revealed = 0; }
      playing = true; bPlay.textContent = "Pause"; last = 0; acc = 0; raf = requestAnimationFrame(tick);
    }
    function stop() { playing = false; bPlay.textContent = "Play"; cancelAnimationFrame(raf); }
    function tick(ts) {
      if (!playing) return;
      if (!last) last = ts; acc += (ts - last) / 1000 * speed; last = ts;
      var n = Math.floor(acc); if (n > 0) { acc -= n; revealed = Math.min(M * M, revealed + n); cur = Math.max(0, revealed - 1); draw(); if (revealed >= M * M) { stop(); return; } }
      raf = requestAnimationFrame(tick);
    }
    function point(cvs, n, e) { var r = cvs.getBoundingClientRect(); return [Math.floor((e.clientY - r.top) / r.height * n), Math.floor((e.clientX - r.left) / r.width * n)]; }
    function pick(e, isIn) {
      var p = point(isIn ? cin : cout, isIn ? N : M, e), r = Math.max(0, Math.min(M - 1, p[0])), c = Math.max(0, Math.min(M - 1, p[1]));
      stop(); cur = r * M + c; if (revealed < cur + 1) revealed = cur + 1; draw();
    }
    [[cin, true], [cout, false]].forEach(function (a) {
      a[0].addEventListener("pointerdown", function (e) { try { a[0].setPointerCapture(e.pointerId); } catch (err) {} pick(e, a[1]); });
      a[0].addEventListener("pointermove", function (e) { if (e.buttons) pick(e, a[1]); });
    });
    function draw() {
      var P = Kit.palette(), i, j, o = relu_ ? relu(full.out) : full.out;
      for (i = 0; i < N; i++) for (j = 0; j < N; j++) { ci.fillStyle = inkColour(P, img[i * N + j]); ci.fillRect(j * CELL, i * CELL, CELL, CELL); }
      var r = Math.floor(cur / M), c = cur % M;
      ci.fillStyle = "rgba(194,86,58,0.18)"; ci.fillRect(c * CELL, r * CELL, 3 * CELL, 3 * CELL);
      ci.strokeStyle = P.coral; ci.lineWidth = 3; ci.strokeRect(c * CELL + 1.5, r * CELL + 1.5, 3 * CELL - 3, 3 * CELL - 3);
      for (i = 0; i < M; i++) for (j = 0; j < M; j++) {
        var idx = i * M + j; co.fillStyle = idx < revealed ? divColour(P, o[idx], vmax) : P.paper; co.fillRect(j * CELL, i * CELL, CELL, CELL);
        if (idx >= revealed) { co.strokeStyle = P.rule; co.lineWidth = 1; co.strokeRect(j * CELL + 0.5, i * CELL + 0.5, CELL - 1, CELL - 1); }
      }
      co.strokeStyle = P.coral; co.lineWidth = 3; co.strokeRect(c * CELL + 1.5, r * CELL + 1.5, CELL - 3, CELL - 3);
      equation(r, c, o);
      Kit.$("sl-status").textContent = (revealed >= M * M ? "All " : revealed + " of ") + (M * M) + " squares of the feature map are filled. Window at row " + r + ", column " + c + ". Colour scale: coral below 0, teal above 0, full colour at " + wfmt(vmax) + ".";
    }
    function equation(r, c, o) {
      var prod = [], sum = 0, u, v, i = 0;
      for (u = 0; u < 3; u++) for (v = 0; v < 3; v++) { var x = img[(r + u) * N + c + v], pr = x * w[u * 3 + v]; pixTd[i].textContent = x.toFixed(0); prdTd[i].textContent = Math.abs(pr) < 0.05 ? "0" : wfmt(Math.round(pr * 10) / 10); sum += pr; i++; }
      var sr = Math.round(sum * 10) / 10;
      sumBox.textContent = wfmt(sr); sumBox.className = "big " + (sr < 0 ? "neg" : "pos");
      reluBox.textContent = relu_ ? "after ReLU: " + wfmt(Math.max(0, sr)) : "";
    }
    pickFilter(0); compute(true);
    Kit.onTheme(draw);
  }

  // ---------- 4. stride and padding ----------
  function sizeFig() {
    var n = 7, k = 3, s = 1, p = 0, sel = 0, reads = false, playing = false, timer = 0;
    var ctrl = Kit.$("sz-controls"), svgI = Kit.$("sz-in"), svgO = Kit.$("sz-out");
    var sn = Kit.slider(ctrl, { label: "Input size n", min: 4, max: 12, step: 1, value: 7, digits: 0 }, function (v) { n = v; sel = 0; draw(); });
    var sk = Kit.slider(ctrl, { label: "Filter size k", min: 1, max: 5, step: 1, value: 3, digits: 0 }, function (v) { k = v; sel = 0; draw(); });
    var ss = Kit.slider(ctrl, { label: "Stride s", min: 1, max: 4, step: 1, value: 1, digits: 0 }, function (v) { s = v; sel = 0; draw(); });
    var sp = Kit.slider(ctrl, { label: "Padding p", min: 0, max: 3, step: 1, value: 0, digits: 0 }, function (v) { p = v; sel = 0; draw(); });
    var bar = Kit.$("sz-bar");
    Kit.button(bar, "Play", function () { var b = this; if (playing) { clearInterval(timer); playing = false; b.textContent = "Play"; return; } playing = true; b.textContent = "Pause"; timer = setInterval(function () { var m = outSize(n, k, s, p); if (m > 0) { sel = (sel + 1) % (m * m); draw(); } }, 450); });
    var rl = label(bar, "", "check"), rc = Kit.html("input", { type: "checkbox" }, rl); rl.appendChild(document.createTextNode(" Count how often a square is read"));
    rc.addEventListener("change", function () { reads = rc.checked; draw(); });
    Kit.button(bar, "Same size (p = (k − 1) / 2, s = 1)", function () { s = 1; ss.set(1); p = Math.floor((k - 1) / 2); sp.set(p); sel = 0; draw(); });
    function draw() {
      var P = Kit.palette(), m = outSize(n, k, s, p), T = n + 2 * p, i, j, u, v;
      if (m > 0 && sel >= m * m) sel = 0;
      var count = []; for (i = 0; i < T * T; i++) count.push(0);
      for (i = 0; i < m; i++) for (j = 0; j < m; j++) for (u = 0; u < k; u++) for (v = 0; v < k; v++) count[(i * s + u) * T + j * s + v]++;
      var c1 = 26, pad = 6;
      function grid(svg, rows, cellfn, extra) {
        svg.innerHTML = ""; var wpx = rows * c1 + 2 * pad; svg.setAttribute("viewBox", "0 0 " + wpx + " " + wpx);
        svg.setAttribute("role", "img");
        for (var a = 0; a < rows; a++) for (var b = 0; b < rows; b++) cellfn(a, b, svg);
        if (extra) extra(svg);
      }
      var maxc = Math.max.apply(null, count.concat([1]));
      grid(svgI, T, function (a, b, svg) {
        var isPad = a < p || b < p || a >= p + n || b >= p + n, cnt = count[a * T + b], x = pad + b * c1, y = pad + a * c1;
        var shade = reads ? Math.min(1, cnt / Math.max(1, maxc)) : 0;
        var fill = isPad ? "none" : reads ? css(mixRGB(P.paper, P.teal, 0.12 + 0.7 * shade)) : P.bg;
        var r = Kit.el("rect", { x: x + 1, y: y + 1, width: c1 - 2, height: c1 - 2, rx: 3, fill: fill, stroke: isPad ? P.muted : P.sand, "stroke-width": 1.2, "stroke-dasharray": isPad ? "3 3" : "none" }, svg);
        var txt = reads ? String(cnt) : isPad ? "0" : "";
        if (txt) { var t = Kit.el("text", { x: x + c1 / 2, y: y + c1 / 2 + 4, "text-anchor": "middle", "font-size": 12, fill: isPad ? P.muted : P.ink, "font-weight": cnt === 0 && !isPad ? 700 : 400 }, svg); t.textContent = txt; }
        if (!isPad && cnt === 0) { Kit.el("line", { x1: x + 4, y1: y + 4, x2: x + c1 - 4, y2: y + c1 - 4, stroke: P.coral, "stroke-width": 2 }, svg); }
      }, function (svg) {
        if (m <= 0) return;
        var r = Math.floor(sel / m), c = sel % m;
        Kit.el("rect", { x: pad + c * s * c1, y: pad + r * s * c1, width: k * c1, height: k * c1, rx: 4, fill: "rgba(194,86,58,0.15)", stroke: P.coral, "stroke-width": 3 }, svg);
        for (var a = 0; a < m; a++) for (var b = 0; b < m; b++) Kit.el("circle", { cx: pad + b * s * c1 + c1 / 2, cy: pad + a * s * c1 + c1 / 2, r: 2.6, fill: a === r && b === c ? P.coral : P.navy, opacity: a === r && b === c ? 1 : 0.55 }, svg);
      });
      svgI.setAttribute("aria-label", "Input grid with padding: " + n + " by " + n + " with " + p + " rings of zeros");
      if (m <= 0) { svgO.innerHTML = ""; svgO.setAttribute("viewBox", "0 0 100 40"); var tt = Kit.el("text", { x: 50, y: 24, "text-anchor": "middle", "font-size": 8, fill: P.coral }, svgO); tt.textContent = "No window fits"; }
      else grid(svgO, m, function (a, b, svg) {
        var x = pad + b * c1, y = pad + a * c1, on = a * m + b === sel;
        var rr = Kit.el("rect", { x: x + 1, y: y + 1, width: c1 - 2, height: c1 - 2, rx: 3, fill: on ? "rgba(194,86,58,0.25)" : css(mixRGB(P.paper, P.teal, 0.25)), stroke: on ? P.coral : P.teal, "stroke-width": on ? 3 : 1.2, style: "cursor:pointer" }, svg);
        rr.addEventListener("click", function () { sel = a * m + b; draw(); });
      });
      svgO.setAttribute("aria-label", "Output grid: " + m + " by " + m);
      var num = n + 2 * p - k, q = m > 0 ? Math.floor(num / s) : 0, exact = num >= 0 && num % s === 0;
      Kit.$("sz-eqn").innerHTML = eop("&lfloor;(") + ebox("c-navy", String(n), "n, input size") + eop("+ 2 &times;") + ebox("c-sand", String(p), "p, padding") + eop("&minus;") + ebox("c-teal", String(k), "k, filter size") + eop(") &divide;") + ebox("c-coral", String(s), "s, stride") + eop("&rfloor; + 1 =") + ebox("c-coral", m > 0 ? String(m) : "none", "output size") +
        eop("|") + ebox("c-navy", num >= 0 ? (num + "/" + s + " = " + (num / s).toFixed(2).replace(/\.00$/, "")) : String(num), "before the floor");
      var nout = m * m, mult = nout * k * k;
      var uncovered = 0; for (i = 0; i < T * T; i++) { var a2 = Math.floor(i / T), b2 = i % T; if (!(a2 < p || b2 < p || a2 >= p + n || b2 >= p + n) && count[i] === 0) uncovered++; }
      Kit.$("sz-status").textContent = m > 0 ? "Output " + m + " by " + m + " = " + nout + " numbers, from " + nout + " window positions times " + k * k + " multiplications = " + mult + " multiplications." + (uncovered ? " " + uncovered + " input squares (crossed out) are never read, because the stride does not divide n + 2p − k evenly: the floor drops them." : "") : "The filter is larger than the padded input, so no window fits.";
    }
    draw();
    Kit.onTheme(draw);
  }

  // ---------- 5. pooling ----------
  function poolFig() {
    var N = 28, M = 26, CELL = 12, cm = Kit.$("pl-map"), cp = Kit.$("pl-pool"), xm = prep(cm, M, M, CELL);
    var digit = 4, filt = 0, kind = "max", q = 2, dx = 0, hov = null, img0 = digitArr(4, 0), F0, P0, Fs, Ps, po, pm;
    var DIG = [4, 1, 2, 7, 3, 5], FL = [0, 1, 2];
    var bar = Kit.$("pl-bar");
    var r1 = Kit.html("div", { class: "bar-row" }, bar), r2 = Kit.html("div", { class: "bar-row" }, bar);
    Kit.chips(r1, DIG.map(function (d) { return "digit " + d; }), 0, function (i) { digit = DIG[i]; img0 = digitArr(digit, 0); calc(); });
    Kit.chips(r1, ["Edge left-right", "Edge top-bottom", "Blur"], 0, function (i) { filt = i; calc(); });
    Kit.chips(r2, ["Max pooling", "Average pooling"], 0, function (i) { kind = i ? "avg" : "max"; calc(); });
    Kit.chips(r2, ["2 by 2, stride 2", "3 by 3, stride 3", "2 by 2, stride 1"], 0, function (i) { q = [2, 3, 2][i]; stp = [2, 3, 1][i]; calc(); });
    var stp = 2;
    var sh = Kit.slider(Kit.$("pl-controls"), { label: "Move the digit to the right, in squares", min: -3, max: 3, step: 1, value: 0, digits: 0 }, function (v) { dx = v; calc(); });
    function feat(img) { var f = conv2d(img, N, PRESETS[filt].w, 3, 1, 0).out; return filt === 2 ? f : relu(f); }
    var xp = null;
    function calc() {
      F0 = feat(img0); Fs = feat(shiftImg(img0, N, dx));
      po = pool2d(F0, M, q, stp, kind); pm = pool2d(Fs, M, q, stp, kind);
      var o = pm.m; cell2h = Math.max(4, Math.floor(M * CELL / o)); cp.width = o * cell2h; cp.height = o * cell2h; xp = cp.getContext("2d");
      hov = null; draw();
    }
    var cell2h = 0;
    function mx(a) { var m = 0, i; for (i = 0; i < a.length; i++) if (Math.abs(a[i]) > m) m = Math.abs(a[i]); return m || 1; }
    cp.addEventListener("pointermove", function (e) { var r = cp.getBoundingClientRect(), o = pm.m; var i = Math.floor((e.clientY - r.top) / r.height * o), j = Math.floor((e.clientX - r.left) / r.width * o); hov = i >= 0 && j >= 0 && i < o && j < o ? [i, j] : null; draw(); });
    cp.addEventListener("pointerleave", function () { hov = null; draw(); });
    cm.addEventListener("pointermove", function (e) { var r = cm.getBoundingClientRect(), i = Math.floor((e.clientY - r.top) / r.height * M), j = Math.floor((e.clientX - r.left) / r.width * M), o = pm.m; var bi = Math.floor(i / stp), bj = Math.floor(j / stp); hov = bi < o && bj < o && bi >= 0 && bj >= 0 ? [bi, bj] : null; draw(); });
    cm.addEventListener("pointerleave", function () { hov = null; draw(); });
    function draw() {
      var P = Kit.palette(), i, j, v0 = mx(F0), o = pm.m;
      var vm = Math.max(mx(Fs), v0);
      for (i = 0; i < M; i++) for (j = 0; j < M; j++) { xm.fillStyle = divColour(P, Fs[i * M + j], vm); xm.fillRect(j * CELL, i * CELL, CELL, CELL); }
      for (i = 0; i < o; i++) for (j = 0; j < o; j++) { xp.fillStyle = divColour(P, pm.out[i * o + j], vm); xp.fillRect(j * cell2h, i * cell2h, cell2h, cell2h); }
      if (hov) {
        xm.strokeStyle = P.coral; xm.lineWidth = 3; xm.strokeRect(hov[1] * stp * CELL + 1.5, hov[0] * stp * CELL + 1.5, q * CELL - 3, q * CELL - 3);
        xp.strokeStyle = P.coral; xp.lineWidth = 3; xp.strokeRect(hov[1] * cell2h + 1.5, hov[0] * cell2h + 1.5, cell2h - 3, cell2h - 3);
        var vals = [], u, v; for (u = 0; u < q; u++) for (v = 0; v < q; v++) vals.push(Fs[(hov[0] * stp + u) * M + hov[1] * stp + v]);
        var shownVals = vals.map(function (x) { return Math.round(x); });
        var res = pm.out[hov[0] * o + hov[1]];
        Kit.$("pl-eqn").innerHTML = eop(kind === "max" ? "max(" : "average(") + ebox("c-navy", shownVals.join(", "), "the " + q * q + " values in the block") + eop(") =") + ebox("c-coral", String(Math.round(res * 10) / 10), kind === "max" ? "the largest" : "their mean") + eop("at output row " + hov[0] + ", column " + hov[1]);
      } else Kit.$("pl-eqn").innerHTML = '<span class="hintline">Move over either map to see one block and its result.</span>';
      var dm = relDiff(Fs, F0), dp = relDiff(pm.out, po.out);
      Kit.$("pl-read").innerHTML = '<span>Feature map: <b>' + o + " by " + o + " after pooling</b> (from " + M + " by " + M + ")</span><span>Change of the map when the digit moves " + Math.abs(dx) + " square" + (Math.abs(dx) === 1 ? "" : "s") + ": <b>" + Kit.pct(dm, 0) + "</b> before pooling, <b>" + Kit.pct(dp, 0) + "</b> after</span>";
      Kit.$("pl-pooltitle").textContent = (kind === "max" ? "After max pooling, " : "After average pooling, ") + o + " by " + o;
    }
    calc();
    Kit.onTheme(draw);
  }
})();
