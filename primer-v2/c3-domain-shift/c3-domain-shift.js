// C3 When the water changes.
//   viewerFig  SIGNATURE: a depth slider degrades a shark photo (the model of depth.py, run in JS) while a pretrained ResNet-18's
//              shark-class confidence (precomputed, data/resnet.json) is plotted; the second copy has an "adapt" switch
//   augFig     random water for training: eight draws on a photo and on digits
//   curveFig   accuracy of a small batch-norm CNN on 10,000 digits against depth (data/murk.json): augmentation and adaptation
//   bnFig      stored batch-norm statistics against the statistics of the murky images, per channel (data/resnet.json)
//   loopFig    read, update, predict: an online stream of frames (data/murk.json, "stream")
// Pure functions are exported with module.exports for the node test (degradeData, attenuation).
(function () {
  "use strict";

  // ---------- the depth model of depth.py, pixel by pixel on 0..255 sRGB arrays ----------
  var K_RGB = [0.54, 0.256, 0.209], BETA = 0.06, VEIL = [0.08, 0.50, 0.55], EXPO = 0.6;
  var K_B = [0.30, 0.50, 0.85], VEIL_B = [0.35, 0.30, 0.15];   // water B: a second, brownish water type (made up)
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
    for (var i = 0; i < src.length; i += 4) {
      for (var c = 0; c < 3; c++) dst[i + c] = Math.round(toSrgb((LIN[src[i + c]] * (1 - V) + veilV[c]) * mult[c]));
      dst[i + 3] = 255;
    }
  }
  // the share of surface light left at depth z for each colour (direct light only): exp(-K z)
  function attenuation(z, k) { k = k || K_RGB; return k.map(function (kc) { return Math.exp(-kc * z); }); }
  function entropy(p) { var h = 0; p.forEach(function (v) { if (v > 0) h -= v * Math.log(v); }); return h; }

  if (typeof module !== "undefined") { module.exports = { degradeData: degradeData, attenuation: attenuation, entropy: entropy }; return; }

  // ---------- page helpers ----------
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op' + (s.length > 14 ? " wrap" : "") + '">' + s + '</span>'; }
  function label(parent, text, cls) { return Kit.html("label", cls ? { class: cls } : {}, parent, text); }
  function nf(v, d) { return Kit.fmt(v, d == null ? 2 : d); }
  function setPressed(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function decode(b64) { var s = atob(b64), a = new Uint8Array(784), i; for (i = 0; i < 784; i++) a[i] = s.charCodeAt(i); return a; }
  // a colour key under a chart: items are {color, text, style} with style "", "dash" or "dot"
  function setKey(id, items) {
    var box = Kit.$(id); if (!box) return; box.innerHTML = "";
    items.forEach(function (it) { var sp = Kit.html("span", {}, box), sw = Kit.html("i", { class: it.style || "" }, sp); sw.style.borderTopColor = it.color; Kit.html("em", {}, sp, it.text); });
  }
  function median(a) { var b = a.slice().sort(function (x, y) { return x - y; }); return b[Math.floor(b.length / 2)]; }

  var RES = null, MURK = null, CREDITS = {}, DIG = null;
  var PHOTOS = [
    { key: "hammerhead-1", name: "Hammerhead, Jarvis Island", shark: true },
    { key: "hammerhead-2", name: "Hammerhead, Costa Rica", shark: true },
    { key: "milk-1", name: "Milk sharks, Mangalore", shark: true },
    { key: "milk-2", name: "Milk sharks, Kuala Lumpur", shark: true },
    { key: "spadenose-1", name: "Spadenose shark on a board", shark: true },
    { key: "dolphin-1", name: "Not a shark: dolphin", shark: false },
    { key: "reeffish-1", name: "Not a shark: clownfish", shark: false },
    { key: "turtle-1", name: "Not a shark: sea turtle", shark: false }
  ];
  var ALPHA_NAMES = [{ a: "0.0", label: "Stored statistics (adaptation off)" }, { a: "0.25", label: "Mix: 25% new images" }, { a: "1.0", label: "Re-estimated on the new images" }];

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/resnet.json"), Kit.data("data/murk.json"), Kit.data("data/digits10.json"), Kit.data("../data/sharks/photos.json")]).then(function (r) {
      RES = r[0]; MURK = r[1]; DIG = r[2].digits;
      r[3].photos.forEach(function (p) { CREDITS[p.file.replace("photos/", "").replace(".jpg", "")] = p.credit_line; });
      viewerFig({ id: "v1", adapt: false }); viewerFig({ id: "v2", adapt: true });
      augFig(); curveFig({ id: "ca", adapt: false }); curveFig({ id: "cb", adapt: true }); bnFig(); loopFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1 and 3. depth slider on a photo, confidence curve, optional adaptation ----------
  function viewerFig(cfg) {
    var id = cfg.id, cur = 3, depth = cfg.adapt ? 10 : 12, mode = 2, W = 440, src = null, H = 0, base = document.createElement("canvas"), bctx = base.getContext("2d", { willReadFrequently: true });
    var cv = Kit.$(id + "-cv"), ctx = cv.getContext("2d"), out = null;
    var bar = Kit.$(id + "-bar");
    var r1 = Kit.html("div", { class: "bar-row" }, bar);
    Kit.chips(r1, PHOTOS.map(function (p) { return p.name; }), cur, function (i) { cur = i; loadPhoto(); });
    if (cfg.adapt) {
      var r2 = Kit.html("div", { class: "bar-row" }, bar); Kit.html("span", { class: "label" }, r2, "Statistics used by the network:");
      Kit.chips(r2, ALPHA_NAMES.map(function (a) { return a.label; }), 2, function (i) { mode = i; draw(); });
    }
    var sl = Kit.slider(Kit.$(id + "-controls"), { label: "Depth, m", min: 0, max: 30, step: 0.5, value: depth, digits: 1 }, function (v) { depth = v; draw(); });
    function loadPhoto() {
      var P = PHOTOS[cur], im = new Image();
      im.onload = function () {
        H = Math.round(W * im.naturalHeight / im.naturalWidth); base.width = W; base.height = H; bctx.drawImage(im, 0, 0, W, H); src = bctx.getImageData(0, 0, W, H);
        cv.width = W; cv.height = H; out = new ImageData(W, H); draw();
      };
      im.src = "../data/sharks/photos/" + P.key + ".jpg";
      Kit.$(id + "-credit").textContent = (CREDITS[P.key] || "") + " Depth effect made with depth.py, a simple model.";
      cv.setAttribute("aria-label", P.name + " at the chosen depth");
    }
    function curve(photo, a, field) { return RES.curves[photo][a][field]; }
    function nonShark(a) {
      var keys = PHOTOS.filter(function (p) { return !p.shark; }).map(function (p) { return p.key; });
      return RES.depths.map(function (d, i) { var s = 0; keys.forEach(function (k) { s += curve(k, a, "shark")[i]; }); return s / keys.length; });
    }
    function draw() {
      if (!src) return;
      degradeData(src.data, out.data, depth); ctx.putImageData(out, 0, 0);
      var P = PHOTOS[cur], a = cfg.adapt ? ALPHA_NAMES[mode].a : "0.0", gi = Math.round(depth * 2), xs = RES.depths;
      var stored = curve(P.key, "0.0", "shark"), chosen = curve(P.key, a, "shark"), non = nonShark(a);
      Kit.plot(id + "-plot", function (Q) {
        var tr = [];
        if (cfg.adapt && mode > 0) tr.push({ x: xs, y: stored, mode: "lines", line: { color: Q.navy, width: 2, dash: "dot" }, hovertemplate: "%{x} m: %{y:.2f}, stored statistics<extra></extra>" });
        tr.push({ x: xs, y: chosen, mode: "lines", line: { color: cfg.adapt && mode > 0 ? Q.teal : Q.navy, width: 3.5 }, hovertemplate: "%{x} m: %{y:.2f}<extra>this photo</extra>" });
        tr.push({ x: xs, y: non, mode: "lines", line: { color: Q.sand, width: 2.5, dash: "dash" }, hovertemplate: "%{x} m: %{y:.2f}<extra>mean of the 3 non-shark photos</extra>" });
        tr.push({ x: [xs[gi]], y: [chosen[gi]], mode: "markers", marker: { color: Q.coral, size: 11, line: { color: "#fff", width: 2 } }, hoverinfo: "skip" });
        return tr;
      }, function (Q) {
        var ann = [];
        return { xaxis: { title: { text: "depth, m" }, range: [0, 30] }, yaxis: { title: { text: "shark-class confidence" }, range: [-0.03, 1.05] }, annotations: ann, margin: { l: 58, r: 14, t: 12, b: 46 } };
      });
      Kit.$(id + "-plot").setAttribute("aria-label", "Shark-class confidence of the network against depth for " + P.name + ", with the current depth marked");
      var Qp = Kit.palette(), keyItems = [{ color: cfg.adapt && mode > 0 ? Qp.teal : Qp.navy, text: "this photo (" + (P.shark ? "a shark" : "not a shark") + ")" }, { color: Qp.sand, text: "mean of the three non-shark photos", style: "dash" }];
      if (cfg.adapt && mode > 0) keyItems.push({ color: Qp.navy, text: "this photo with stored statistics", style: "dot" });
      setKey(id + "-key", keyItems);
      // readout: top three calls with the stored statistics, and the confidence under the chosen mode
      var t3 = RES.top3_stored[P.key][gi].map(function (t) { return RES.classes[t[0]] + " " + nf(t[1], 2); }).join(", ");
      var kv = attenuation(depth), V = 1 - Math.exp(-BETA * depth);
      Kit.$(id + "-read").innerHTML = '<span>At <b>' + nf(depth, 1) + ' m</b>, shark-class confidence: <b>' + nf(stored[gi], 2) + '</b> with the stored statistics' + (cfg.adapt && mode > 0 ? ', <b>' + nf(chosen[gi], 2) + '</b> with ' + (mode === 1 ? "25% new statistics" : "re-estimated statistics") : "") + '.</span><span>Top three calls (stored): ' + t3 + '.</span>';
      Kit.$(id + "-eqn").innerHTML = eop("direct light left, e<sup>&minus;K z</sup>:") + ebox("c-coral", nf(kv[0], 3), "red, K = 0.54") + ebox("c-teal", nf(kv[1], 3), "green, K = 0.256") + ebox("c-navy", nf(kv[2], 3), "blue, K = 0.209") + eop("|") + ebox("c-sand", Kit.pct(V, 0), "haze share V = 1 &minus; e<sup>&minus;0.06 z</sup>");
    }
    loadPhoto();
    Kit.onTheme(draw);
  }

  // ---------- 2. random water for training ----------
  function augFig() {
    var cur = 0, zmax = 15, variety = 1, flip = true, seed = 1, n = 8, W = 200, src = null, H = 0;
    var base = document.createElement("canvas"), bctx = base.getContext("2d", { willReadFrequently: true });
    var files = ["hammerhead-1", "milk-2", "spadenose-1"], names = ["Hammerhead", "Milk sharks", "Spadenose"];
    var bar = Kit.$("ag-bar");
    Kit.chips(bar, names, 0, function (i) { cur = i; loadPhoto(); });
    Kit.button(bar, "New draw", function () { seed++; draw(); });
    var fl = label(bar, "", "check"), fc = Kit.html("input", { type: "checkbox", checked: "checked" }, fl); fl.appendChild(document.createTextNode(" Random flip and crop"));
    fc.addEventListener("change", function () { flip = fc.checked; draw(); });
    var ctrl = Kit.$("ag-controls");
    Kit.slider(ctrl, { label: "Deepest water drawn, m", min: 0, max: 25, step: 1, value: 15, digits: 0 }, function (v) { zmax = v; draw(); });
    Kit.slider(ctrl, { label: "Variety of the water (0 = one fixed water)", min: 0, max: 1, step: 0.05, value: 1, digits: 2 }, function (v) { variety = v; draw(); });
    var grid = Kit.$("ag-grid"), dgrid = Kit.$("ag-digits"), cells = [], dcells = [];
    for (var i = 0; i < n; i++) {
      var f = Kit.html("figure", { class: "ag" }, grid), c = Kit.html("canvas", { class: "fig photo", role: "img", "aria-label": "augmented training photo " + (i + 1) }, f); Kit.html("figcaption", {}, f, ""); cells.push(c);
      var f2 = Kit.html("figure", { class: "ag" }, dgrid), c2 = Kit.html("canvas", { class: "fig", width: 28, height: 28, role: "img", "aria-label": "augmented training digit " + (i + 1) }, f2); Kit.html("figcaption", {}, f2, ""); dcells.push(c2);
    }
    function loadPhoto() {
      var im = new Image(); im.onload = function () { H = Math.round(W * im.naturalHeight / im.naturalWidth); base.width = W; base.height = H; bctx.drawImage(im, 0, 0, W, H); src = bctx.getImageData(0, 0, W, H); draw(); };
      im.src = "../data/sharks/photos/" + files[cur] + ".jpg";
      Kit.$("ag-credit").textContent = (CREDITS[files[cur]] || "") + " Depth effects made with depth.py, a simple model.";
    }
    function draws() {
      var r = Kit.rng(seed * 7919 + 13), out = [];
      for (var i = 0; i < n; i++) {
        var z = r() * zmax, s = 0.2 * variety;
        out.push({ z: z, k: K_RGB.map(function (kc) { return kc * Math.exp(s * Kit.gauss(r)); }), beta: BETA * (1 + (r() * 2 - 1) * 0.4 * variety), expo: EXPO + (r() * 2 - 1) * 0.3 * variety, flip: flip && r() < 0.5, crop: flip ? 0.7 + 0.3 * r() : 1, cx: r(), cy: r() });
      }
      return out;
    }
    function draw() {
      if (!src) return;
      var D = draws(), i, j;
      var tmp = new ImageData(W, H);
      D.forEach(function (d, i) {
        degradeData(src.data, tmp.data, d.z, { k: d.k, beta: d.beta, expo: d.expo });
        var c = cells[i], ctx = c.getContext("2d"); c.width = W; c.height = H;
        var t = document.createElement("canvas"); t.width = W; t.height = H; t.getContext("2d").putImageData(tmp, 0, 0);
        var cw = W * d.crop, ch = H * d.crop, sx = (W - cw) * d.cx, sy = (H - ch) * d.cy;
        ctx.save(); if (d.flip) { ctx.translate(W, 0); ctx.scale(-1, 1); } ctx.drawImage(t, sx, sy, cw, ch, 0, 0, W, H); ctx.restore();
        c.parentNode.querySelector("figcaption").textContent = nf(d.z, 1) + " m, K " + d.k.map(function (v) { return nf(v, 2); }).join(" / ") + (d.flip ? ", flipped" : "");
        // the same water on a digit (white on black, three equal channels)
        var dg = decode(DIG[(i * 3 + seed) % 10].b64), s4 = new Uint8ClampedArray(784 * 4), o4 = new Uint8ClampedArray(784 * 4);
        for (j = 0; j < 784; j++) { s4[4 * j] = s4[4 * j + 1] = s4[4 * j + 2] = dg[j]; s4[4 * j + 3] = 255; }
        degradeData(s4, o4, d.z, { k: d.k, beta: d.beta, expo: d.expo });
        var dc = dcells[i], dctx = dc.getContext("2d"), im = dctx.createImageData(28, 28); im.data.set(o4); dctx.putImageData(im, 0, 0);
        dc.parentNode.querySelector("figcaption").textContent = nf(d.z, 1) + " m";
      });
    }
    loadPhoto();
  }

  // ---------- 2 and 3. accuracy of a batch-norm CNN on 10,000 digits against depth ----------
  function curveFig(cfg) {
    var id = cfg.id, water = "A", depth = 10, bar = Kit.$(id + "-bar");
    Kit.chips(bar, ["Blue-green water (the water of the training draws)", "Brown river water (never seen in training, made up)"], 0, function (i) { water = i ? "B" : "A"; draw(); });
    var sl = Kit.slider(Kit.$(id + "-controls"), { label: "Depth for the readout and the digits, m", min: 0, max: 30, step: 1, value: 10, digits: 0 }, function (v) { depth = v; draw(); });
    var strip = Kit.$(id + "-digits"), dcs = [];
    for (var q = 0; q < 8; q++) { var c = Kit.html("canvas", { class: "fig dg", width: 28, height: 28, role: "img", "aria-label": "a digit in murky water" }, strip); dcs.push(c); }
    function digitsAt(z) {
      var o = water === "A" ? {} : { k: K_B, veil: VEIL_B };
      dcs.forEach(function (c, i) {
        var dg = decode(DIG[[7, 2, 4, 1, 9, 3, 0, 5][i]].b64), s4 = new Uint8ClampedArray(784 * 4), o4 = new Uint8ClampedArray(784 * 4), j;
        for (j = 0; j < 784; j++) { s4[4 * j] = s4[4 * j + 1] = s4[4 * j + 2] = dg[j]; s4[4 * j + 3] = 255; }
        degradeData(s4, o4, z, o); var ctx = c.getContext("2d"), im = ctx.createImageData(28, 28); im.data.set(o4); ctx.putImageData(im, 0, 0);
      });
    }
    function draw() {
      var M = MURK[water], xs = MURK.depths, gi = xs.indexOf(depth) >= 0 ? xs.indexOf(depth) : xs.reduce(function (b, v, i) { return Math.abs(v - depth) < Math.abs(xs[b] - depth) ? i : b; }, 0);
      Kit.plot(id + "-plot", function (Q) {
        var tr = [{ x: xs, y: M.clean, mode: "lines+markers", line: { color: Q.navy, width: 3 }, marker: { size: 6 }, hovertemplate: "%{x} m: %{y:.1%}<extra>trained on clean digits</extra>" }];
        if (!cfg.adapt) tr.push({ x: xs, y: M.aug, mode: "lines+markers", line: { color: Q.teal, width: 3 }, marker: { size: 6 }, hovertemplate: "%{x} m: %{y:.1%}<extra>trained with random water</extra>" });
        else {
          tr.push({ x: xs, y: M.clean_adapt, mode: "lines+markers", line: { color: Q.teal, width: 3 }, marker: { size: 6 }, hovertemplate: "%{x} m: %{y:.1%}<extra>clean-trained, statistics re-estimated</extra>" });
          tr.push({ x: xs, y: M.aug, mode: "lines", line: { color: Q.sand, width: 2.5, dash: "dash" }, hovertemplate: "%{x} m: %{y:.1%}<extra>trained with random water</extra>" });
        }
        tr.push({ x: [xs[gi], xs[gi]], y: [0, 1], mode: "lines", line: { color: Q.coral, width: 2, dash: "dot" }, hoverinfo: "skip" });
        return tr;
      }, function (Q) {
        var ann = [];
        var shapes = cfg.adapt ? [] : [{ type: "rect", xref: "x", yref: "paper", x0: 0, x1: 15, y0: 0, y1: 1, fillcolor: Q.mark, line: { width: 0 }, layer: "below" }];
        if (!cfg.adapt) ann.push({ x: 7.5, y: 0.04, xref: "x", yref: "y", showarrow: false, text: "depths used in training", font: { color: Q.muted, size: 12 } });
        return { xaxis: { title: { text: "depth, m" }, range: [0, 30] }, yaxis: { title: { text: "accuracy on 10,000 test digits" }, range: [0, 1.02], tickformat: ".0%" }, annotations: ann, shapes: shapes, margin: { l: 62, r: 14, t: 12, b: 46 } };
      });
      Kit.$(id + "-plot").setAttribute("aria-label", "Accuracy of the digit reader against depth in " + (water === "A" ? "blue-green" : "brown") + " water");
      var Qp = Kit.palette();
      setKey(id + "-key", cfg.adapt ? [{ color: Qp.teal, text: "clean-trained, new statistics" }, { color: Qp.navy, text: "clean-trained, stored statistics" }, { color: Qp.sand, text: "trained with random water", style: "dash" }] : [{ color: Qp.teal, text: "trained with random water, 0 to 15 m" }, { color: Qp.navy, text: "trained on clean digits only" }]);
      var sd = M.clean_adapt_sd[gi];
      Kit.$(id + "-eqn").innerHTML = ebox("c-navy", Kit.pct(M.clean[gi], 1), "clean-trained, stored statistics") + (cfg.adapt ? ebox("c-teal", Kit.pct(M.clean_adapt[gi], 1), "statistics re-estimated (&plusmn;" + (100 * sd).toFixed(1) + " points over 5 draws)") + ebox("c-sand", Kit.pct(M.aug[gi], 1), "trained with random water") + ebox("c-coral", Kit.pct(M.aug_adapt[gi], 1), "random water and re-estimated statistics") : ebox("c-teal", Kit.pct(M.aug[gi], 1), "trained with random water")) + eop("at " + xs[gi] + " m");
      digitsAt(xs[gi]);
    }
    draw();
    Kit.onTheme(draw);
  }

  // ---------- 3. stored batch-norm statistics against the statistics of the murky images ----------
  function bnFig() {
    var layers = ["bn1", "layer4.1.bn2"], names = ["First layer (the 64 channels after the first convolution)", "Last layer (512 channels, layer 4)"], li = 0, di = 3, ch = -1;
    var bar = Kit.$("bn-bar"), B = RES.bnstats;
    Kit.chips(bar, names, 0, function (i) { li = i; ch = -1; draw(); });
    Kit.html("span", { class: "label" }, bar, "Depth of the new images:");
    var dchips = Kit.chips(bar, B.depths.map(function (d) { return d + " m"; }), di, function (i) { di = i; draw(); });
    function draw() {
      var L = B.layers[layers[li]], sm = L.stored_mean, sv = L.stored_var, nm = L.mean[di], nv = L.var[di], n = sm.length, i;
      var med1 = median(sm.map(function (m, k) { return Math.abs(nm[k] - m) / Math.sqrt(sv[k] + 1e-5); })), med2 = median(sv.map(function (v, k) { return nv[k] / (v + 1e-12); }));
      var cols = sm.map(function (_, k) { return k === ch ? 1 : 0; });
      function mm(a) { return [Math.min.apply(null, a), Math.max.apply(null, a)]; }
      var lo = Math.min(mm(sm)[0], mm(nm)[0]), hi = Math.max(mm(sm)[1], mm(nm)[1]), pad = (hi - lo) * 0.05;
      var vl = Math.min.apply(null, sv.concat(nv).filter(function (v) { return v > 0; })), vh = Math.max.apply(null, sv.concat(nv));
      var p1 = Kit.plot("bn-mean", function (Q) {
        return [{ x: [lo - pad, hi + pad], y: [lo - pad, hi + pad], mode: "lines", line: { color: Q.muted, width: 1.5, dash: "dot" }, hoverinfo: "skip" },
          { x: sm, y: nm, mode: "markers", marker: { size: sm.map(function (_, k) { return k === ch ? 12 : 6; }), color: sm.map(function (_, k) { return k === ch ? Q.coral : Q.navy; }), opacity: 0.8 }, text: sm.map(function (_, k) { return "channel " + k; }), hovertemplate: "%{text}<br>stored %{x:.2f}, now %{y:.2f}<extra></extra>" }];
      }, { xaxis: { title: { text: "stored mean of the channel" }, range: [lo - pad, hi + pad] }, yaxis: { title: { text: "mean on the new images" }, range: [lo - pad, hi + pad] }, margin: { l: 58, r: 12, t: 10, b: 46 } });
      var p2 = Kit.plot("bn-var", function (Q) {
        return [{ x: [vl / 1.5, vh * 1.5], y: [vl / 1.5, vh * 1.5], mode: "lines", line: { color: Q.muted, width: 1.5, dash: "dot" }, hoverinfo: "skip" },
          { x: sv, y: nv, mode: "markers", marker: { size: sv.map(function (_, k) { return k === ch ? 12 : 6; }), color: sv.map(function (_, k) { return k === ch ? Q.coral : Q.teal; }), opacity: 0.8 }, text: sv.map(function (_, k) { return "channel " + k; }), hovertemplate: "%{text}<br>stored %{x:.3g}, now %{y:.3g}<extra></extra>" }];
      }, { xaxis: { type: "log", title: { text: "stored variance of the channel" } }, yaxis: { type: "log", title: { text: "variance on the new images" } }, margin: { l: 58, r: 12, t: 10, b: 46 } });
      [[p1, "bn-mean"], [p2, "bn-var"]].forEach(function (q) {
        q[0].then(function () {
          var gd = Kit.$(q[1]); if (gd.on && !gd._c3) { gd._c3 = true; gd.on("plotly_click", function (e) { var p = e.points[0]; if (p.curveNumber === 1) { ch = p.pointIndex; draw(); } }); gd.on("plotly_hover", function (e) { var p = e.points[0]; if (p.curveNumber === 1 && p.pointIndex !== ch) { ch = p.pointIndex; draw(); } }); }
        });
      });
      var c = ch >= 0 ? ch : (function () { var b = 0, bv = -1; sm.forEach(function (m, k) { var s = Math.abs(nm[k] - m) / Math.sqrt(sv[k] + 1e-5); if (s > bv) { bv = s; b = k; } }); return b; })();
      var mu = sm[c], va = sv[c], mu2 = nm[c], va2 = nv[c], x = mu2 + Math.sqrt(va2), h1 = (x - mu) / Math.sqrt(va + 1e-5), h2 = (x - mu2) / Math.sqrt(va2 + 1e-5);
      Kit.$("bn-eqn").innerHTML = eop("channel " + c + (ch >= 0 ? "" : " (the most shifted; click a dot for another)") + ":") + ebox("c-navy", nf(mu, 2) + ", " + nf(va, 3), "stored &mu; and variance") + ebox("c-teal", nf(mu2, 2) + ", " + nf(va2, 3), "&mu; and variance on the new images") + eop("|") + ebox("c-sand", nf(x, 2), "a value one sd above the new mean") + eop("&rarr;") + ebox("c-coral", nf(h1, 2), "normalised with the stored values") + ebox("c-coral", nf(h2, 2), "normalised with the new values");
      Kit.$("bn-read").innerHTML = "<span>Median shift of a channel's mean: <b>" + nf(med1, 2) + "</b> stored standard deviations.</span><span>Median ratio of the variance to the stored one: <b>" + nf(med2, 2) + "</b>.</span>";
      Kit.$("bn-mean").setAttribute("aria-label", "Mean of the channels on the new images against the stored mean"); Kit.$("bn-var").setAttribute("aria-label", "Variance of the channels on the new images against the stored variance");
    }
    draw();
  }

  // ---------- 4. read, update, predict ----------
  function loopFig() {
    var S = MURK.stream, N = S.n_chunks, t = N, stage = 0, playing = false, timer = 0, bar = Kit.$("lp-bar"), svg = Kit.$("lp-svg");
    var bPlay = Kit.button(bar, "Play", function () { if (playing) stop(); else start(); });
    Kit.button(bar, "Step", function () { stop(); step(); });
    Kit.button(bar, "Reset", function () { stop(); t = 0; stage = 0; draw(); });
    Kit.button(bar, "Show all", function () { stop(); t = N; stage = 0; draw(); });
    function start() { if (t >= N) { t = 0; stage = 0; } playing = true; bPlay.textContent = "Pause"; timer = setInterval(function () { step(); if (t >= N) stop(); }, 140); }
    function stop() { playing = false; bPlay.textContent = "Play"; clearInterval(timer); }
    function step() { if (t >= N) { t = 0; stage = 0; } stage++; if (stage > 2) { stage = 0; t++; } draw(); }
    function diagram() {
      var P = Kit.palette(), Wd = Math.max(300, Kit.width(svg)), narrow = Wd < 560;
      svg.innerHTML = ""; var Hh = narrow ? 330 : 172; svg.setAttribute("viewBox", "0 0 " + Wd + " " + Hh);
      var steps = [["1 Read", "200 new frames arrive, no labels"], ["2 Update", "statistics, then scale and shift by one step on the entropy of the predictions"], ["3 Predict", "the updated network labels the next frames"]];
      var bw = narrow ? Wd - 30 : (Wd - 60) / 3, bh = narrow ? 76 : 100;
      steps.forEach(function (s, i) {
        var x = narrow ? 15 : 15 + i * (bw + 15), y = narrow ? 8 + i * (bh + 12) : 14, on = stage === i && t < N;
        Kit.el("rect", { x: x, y: y, width: bw, height: bh, rx: 8, fill: on ? P.mark : P.bg, stroke: on ? P.teal : P.rule, "stroke-width": on ? 3 : 1.5 }, svg);
        var tt = Kit.el("text", { x: x + 12, y: y + 26, "font-size": 16, "font-weight": 700, fill: on ? P.ink : P.ink2 }, svg); tt.textContent = s[0];
        wrap(svg, s[1], x + 12, y + 46, bw - 24, 13.5, P.ink2);
        if (i < 2) { if (narrow) Kit.el("path", { d: "M" + (Wd / 2) + " " + (y + bh) + " l0 10 l-5 -6 m5 6 l5 -6", stroke: P.muted, "stroke-width": 2, fill: "none" }, svg); else Kit.el("path", { d: "M" + (x + bw + 1) + " " + (y + bh / 2) + " l12 0 l-6 -5 m6 5 l-6 5", stroke: P.muted, "stroke-width": 2, fill: "none" }, svg); }
      });
      if (!narrow) { var lx = 15 + 2 * (bw + 15) + bw / 2; Kit.el("path", { d: "M" + lx + " 118 L" + lx + " 138 L" + (15 + bw / 2) + " 138 L" + (15 + bw / 2) + " 118", stroke: P.muted, "stroke-width": 2, fill: "none", "stroke-dasharray": "5 4" }, svg); var tl = Kit.el("text", { x: Wd / 2, y: 146, "text-anchor": "middle", "font-size": 13, fill: P.muted }, svg); tl.textContent = "next chunk"; }
    }
    function wrap(svgEl, text, x, y, w, size, fill) {
      var words = text.split(" "), line = "", ln = 0, per = Math.max(10, Math.floor(w / (size * 0.5)));
      words.forEach(function (wd) { if ((line + " " + wd).length > per) { var e = Kit.el("text", { x: x, y: y + ln * (size + 3), "font-size": size, fill: fill }, svgEl); e.textContent = line; line = wd; ln++; } else line = line ? line + " " + wd : wd; });
      var e2 = Kit.el("text", { x: x, y: y + ln * (size + 3), "font-size": size, fill: fill }, svgEl); e2.textContent = line;
    }
    function draw() {
      diagram();
      var xs = S.zs.map(function (z, i) { return i + 1; }), k = Math.max(0, t);
      function cut(a) { return a.slice(0, k); }
      Kit.plot("lp-plot", function (Q) {
        return [
          { x: cut(xs), y: cut(S.frozen), mode: "lines", line: { color: Q.navy, width: 3 }, hovertemplate: "chunk %{x}: %{y:.1%}<extra>frozen</extra>" },
          { x: cut(xs), y: cut(S.stats), mode: "lines", line: { color: Q.sand, width: 3 }, hovertemplate: "chunk %{x}: %{y:.1%}<extra>statistics only</extra>" },
          { x: cut(xs), y: cut(S.tent), mode: "lines", line: { color: Q.teal, width: 3.5 }, hovertemplate: "chunk %{x}: %{y:.1%}<extra>statistics, scale and shift</extra>" }
        ];
      }, function (Q) {
        return { xaxis: { title: { text: "chunk of 200 frames (depth 0 to 12 m)" }, range: [0.5, N + 0.5] }, yaxis: { title: { text: "accuracy on the chunk" }, range: [0, 1.02], tickformat: ".0%" },
          annotations: [], margin: { l: 58, r: 14, t: 10, b: 50 } };
      });
      var i = Math.min(N, Math.max(1, k)) - 1;
      var avg = function (a) { var s = 0, j; for (j = 0; j < N; j++) s += a[j]; return s / N; };
      Kit.$("lp-eqn").innerHTML = (k === 0 ? eop("Press Play or Step. The chart fills one chunk at a time.") : eop("chunk " + (i + 1) + " of " + N + ", depth " + nf(S.zs[i], 1) + " m:") + ebox("c-navy", Kit.pct(S.frozen[i], 0), "frozen") + ebox("c-sand", Kit.pct(S.stats[i], 0), "statistics only") + ebox("c-teal", Kit.pct(S.tent[i], 0), "statistics, scale and shift")) + eop("|") + eop("mean over all " + N + " chunks:") + ebox("c-navy", Kit.pct(avg(S.frozen), 1), "frozen") + ebox("c-sand", Kit.pct(avg(S.stats), 1), "statistics only") + ebox("c-teal", Kit.pct(avg(S.tent), 1), "statistics, scale and shift");
      Kit.$("lp-plot").setAttribute("aria-label", "Accuracy per chunk of frames as the dive gets deeper, for a frozen network and two ways of updating it");
      var Qp = Kit.palette(); setKey("lp-key", [{ color: Qp.teal, text: "update statistics, scale and shift" }, { color: Qp.sand, text: "update statistics only" }, { color: Qp.navy, text: "frozen, no update" }]);
    }
    draw();
    Kit.onTheme(draw);
    window.addEventListener("resize", Kit.debounce(draw));
  }
})();
