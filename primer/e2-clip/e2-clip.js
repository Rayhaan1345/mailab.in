// E2 Encoders and CLIP.
//   encFig      a photo and a caption become vectors (first 40 of 512 numbers drawn); their cosine (data/clip.json)
//   trainFig    three ways to train an encoder: labels, masked prediction (MAE reconstructions), contrastive pairs
//   matrixFig   SIGNATURE: choose a batch of shark photos; the similarity matrix with its diagonal, the softmax of rows and columns, the loss in boxes
//   tempFig     the temperature T sharpens or flattens the softmax over captions
//   spaceFig    photos and captions in one 2D map (PCA computed offline), with the modality gap and a switch that removes it
//   zeroFig     zero-shot classification: pick the captions, the template and the photo
//   murkyFig    the same classification as the depth slider makes the water murkier (depth.py in the browser; CLIP numbers precomputed)
// Pure helpers are exported with module.exports for the node test.
(function () {
  "use strict";

  // ---------- pure helpers ----------
  function softmax(z) { var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - m); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (v) { return v / s; }); }
  function entropy(p) { var h = 0; p.forEach(function (v) { if (v > 0) h -= v * Math.log(v); }); return h; }
  // symmetric contrastive loss of an n by n cosine matrix S (rows photos, columns captions) at temperature T
  function clipLoss(S, T) {
    var n = S.length, rows = [], cols = [], i, j, row, col;
    for (i = 0; i < n; i++) {
      row = softmax(S[i].map(function (v) { return v / T; })); rows.push(-Math.log(row[i]));
      col = softmax(S.map(function (r) { return r[i] / T; })); cols.push(-Math.log(col[i]));
    }
    var a = rows.reduce(function (x, y) { return x + y; }, 0) / n, b = cols.reduce(function (x, y) { return x + y; }, 0) / n;
    return { rows: rows, cols: cols, img: a, txt: b, total: (a + b) / 2 };
  }
  // the depth model of depth.py (copied from chapter C3), on 0..255 RGBA arrays
  var K_RGB = [0.54, 0.256, 0.209], BETA = 0.06, VEIL = [0.08, 0.50, 0.55], EXPO = 0.6;
  var LIN = new Float32Array(256);
  (function () { for (var v = 0; v < 256; v++) { var x = v / 255; LIN[v] = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); } })();
  function toSrgb(x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return 255 * (x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055); }
  function degradeData(src, dst, z) {
    var k = K_RGB, kmin = Math.min(k[0], k[1], k[2]), kmean = (k[0] + k[1] + k[2]) / 3, V = 1 - Math.exp(-BETA * z), dark = Math.exp(-kmin * z), gain = Math.exp(EXPO * kmean * z);
    var mult = [0, 1, 2].map(function (c) { return Math.exp(-(k[c] - kmin) * z) * dark * gain; }), veilV = [0, 1, 2].map(function (c) { return VEIL[c] * V; });
    for (var i = 0; i < src.length; i += 4) {
      for (var c = 0; c < 3; c++) dst[i + c] = Math.round(toSrgb((LIN[src[i + c]] * (1 - V) + veilV[c]) * mult[c]));
      dst[i + 3] = 255;
    }
  }
  if (typeof module !== "undefined") { module.exports = { softmax: softmax, entropy: entropy, clipLoss: clipLoss, degradeData: degradeData }; return; }

  // ---------- page helpers ----------
  var D = {}, CREDITS = {}, IMG = {};
  var KEYS = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"];
  var NICE = ["Spadenose, board", "Spadenose, basket", "Milk, wharf", "Milk, market", "Bamboo, tank", "Bamboo, rocks", "Hammerhead, blue", "Hammerhead, Costa Rica", "Ganges, 1839 drawing", "Ganges, line drawing", "Eagle ray", "Dolphin", "Clownfish", "Sea turtle"];
  var SHARK_CLASSES = [0, 1, 2, 3, 4];
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function nf(v, d) { return Kit.fmt(v, d == null ? 2 : d); }
  function comma(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function mix(a, b, t) { return [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t), Math.round(a[2] + (b[2] - a[2]) * t)]; }
  function rgb(c) { return "rgb(" + c.join(",") + ")"; }
  function loadImg(src) { return new Promise(function (ok, fail) { var im = new Image(); im.onload = function () { ok(im); }; im.onerror = function () { fail(new Error("image " + src)); }; im.src = src; }); }
  function fitCanvas(cv, w, h) {
    var d = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(w * d); cv.height = Math.round(h * d); cv.style.width = w + "px"; cv.style.height = h + "px";
    var ctx = cv.getContext("2d"); ctx.setTransform(d, 0, 0, d, 0, 0); return ctx;
  }
  function onResize(fn) { window.addEventListener("resize", Kit.debounce(fn, 120)); Kit.onTheme(fn); }
  function credit(i) { return CREDITS[KEYS[i]] || ""; }
  var TL = ["name only", "a photo of a \u2026", "a photo of a \u2026 in murky water"];
  function creditsList(id) {
    var ul = Kit.$(id); if (!ul) return; ul.innerHTML = "";
    KEYS.forEach(function (k, i) { Kit.html("li", {}, ul, NICE[i] + ": " + (CREDITS[k] || "")); });
  }
  function setKey(id, items) {
    var box = Kit.$(id); if (!box) return; box.innerHTML = "";
    items.forEach(function (it) { var sp = Kit.html("span", {}, box), sw = Kit.html("i", { class: it.style || "" }, sp); sw.style.borderTopColor = it.color; Kit.html("em", {}, sp, it.text); });
  }
  function thumb(parent, i, size, cls) { var b = Kit.html("button", { type: "button", class: "thumb " + (cls || ""), "aria-pressed": "false", title: NICE[i] }, parent), im = Kit.html("img", { src: "../data/sharks/photos/" + KEYS[i] + ".jpg", alt: NICE[i], width: size, height: size }, b); return b; }
  function softBars(box, labels, probs, opts) {
    opts = opts || {}; box.innerHTML = ""; var order = probs.map(function (p, i) { return i; }); if (opts.sort !== false) order.sort(function (a, b) { return probs[b] - probs[a]; });
    order.forEach(function (i) {
      var r = Kit.html("div", { class: "rs-row" + (opts.mark === i ? " tr" : "") }, box); Kit.html("span", { class: "rs-name" }, r, labels[i]);
      var tr = Kit.html("span", { class: "rs-track" }, r), f = Kit.html("i", {}, tr); f.style.width = (100 * probs[i]).toFixed(1) + "%";
      if (opts.good === i) f.style.background = "color-mix(in srgb, var(--teal) 80%, transparent)"; else if (opts.bad === i) f.style.background = "color-mix(in srgb, var(--coral) 75%, transparent)";
      Kit.html("b", {}, r, probs[i] >= 0.995 ? "100%" : (100 * probs[i] < 0.5 ? (100 * probs[i]).toFixed(1) + "%" : Math.round(100 * probs[i]) + "%"));
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    var imgs = KEYS.map(function (k) { return loadImg("../data/sharks/photos/" + k + ".jpg"); });
    Promise.all([Kit.data("data/clip.json"), Kit.data("data/mae.json"), Kit.data("../data/sharks/photos.json")].concat(imgs)).then(function (r) {
      D.clip = r[0]; D.mae = r[1];
      r[2].photos.forEach(function (p) { CREDITS[p.file.replace("photos/", "").replace(".jpg", "")] = p.credit_line; });
      KEYS.forEach(function (k, i) { IMG[k] = r[3 + i]; });
      encFig(); trainFig(); matrixFig(); tempFig(); spaceFig(); zeroFig(); murkyFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. encoders: a photo and a caption become vectors ----------
  function encFig() {
    var C = D.clip, M = C.map, st = { ph: 6, tx: 31 }, cv = Kit.$("en-cv");
    var grid = Kit.$("en-thumbs"), btns = KEYS.map(function (k, i) { var b = thumb(grid, i, 44); b.addEventListener("click", function () { st.ph = i; draw(); }); return b; });
    var items = []; M.items.forEach(function (it, i) { if (i >= 14) items.push({ i: i, label: (it.kind === "caption" ? "caption: " : it.kind === "prompt" ? "prompt: " : "generic: ") + it.label }); });
    var sel = Kit.select(Kit.$("en-controls"), "Text", items.map(function (o) { return { value: o.i, label: o.label }; }), String(st.tx), function (v) { st.tx = +v; draw(); });
    function strip(cvs, vals, P, label) {
      var cs = 6, cols = 40, ctx = fitCanvas(cvs, cols * cs, cs * 2 + 2), i; ctx.clearRect(0, 0, cols * cs, 14);
      for (i = 0; i < cols; i++) { var col = mix(hexRGB(P.paper), hexRGB(vals[i] >= 0 ? P.teal : P.coral), Math.min(1, Math.abs(vals[i]) / 0.12)); ctx.fillStyle = rgb(col); ctx.fillRect(i * cs, 0, cs - 1, cs * 2); }
      cvs.setAttribute("aria-label", label);
    }
    function draw() {
      var P = Kit.palette(), ph = st.ph, tx = st.tx, it = M.items[tx], c = M.cos[ph][tx];
      btns.forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === ph)); });
      var W = Math.min(224, Kit.width(Kit.$("en-photo"))), ctx = fitCanvas(cv, W, W); ctx.drawImage(IMG[KEYS[ph]], 0, 0, W, W);
      strip(Kit.$("en-sv"), C.strip.photo[ph], P, "The first 40 of 512 numbers of the photo vector");
      var tstrip = it.kind === "caption" ? C.strip.caption[tx - 14] : it.kind === "prompt" ? C.strip.prompt[tx - 28] : null;
      var tcv = Kit.$("en-st");
      if (tstrip) { strip(tcv, tstrip, P, "The first 40 of 512 numbers of the text vector"); tcv.style.opacity = 1; } else { fitCanvas(tcv, 240, 14).clearRect(0, 0, 240, 14); tcv.setAttribute("aria-label", "Numbers of this text vector are not drawn"); }
      Kit.$("en-text").textContent = "“" + it.label + "”";
      Kit.$("en-eq").innerHTML = ebox("c-navy", "1.00", "length of the photo vector, |u|") + eop("&times;") + ebox("c-coral", "1.00", "length of the text vector, |v|") + eop("&times;") + ebox("c-sand", nf(c, 3), "cos of the angle between them") + eop("=") + ebox("c-teal", nf(c, 3), "u &middot; v, the dot product");
      // ranking of the nine species prompts for this photo
      var labels = C.class_names, vals = M.cos[ph].slice(28, 37);
      var tr = Kit.$("en-rank"); tr.innerHTML = "";
      vals.map(function (v, k) { return k; }).sort(function (a, b) { return vals[b] - vals[a]; }).forEach(function (k) {
        var r = Kit.html("div", { class: "rs-row" + (28 + k === tx ? " tr" : "") }, tr); Kit.html("span", { class: "rs-name" }, r, labels[k]);
        var t = Kit.html("span", { class: "rs-track" }, r), f = Kit.html("i", {}, t); f.style.width = Math.max(0, Math.min(100, (vals[k] - 0.1) / 0.25 * 100)) + "%"; Kit.html("b", {}, r, nf(vals[k], 3));
      });
      sel.value = String(tx);
      Kit.$("en-credit").textContent = credit(ph);
      cv.setAttribute("aria-label", NICE[ph] + " photo");
    }
    creditsList("en-credits");
    onResize(draw); draw();
  }

  // ---------- 2. three ways to train an encoder ----------
  function trainFig() {
    var Mae = D.mae, st = { tab: 1, ph: 0, ri: 3, p: 0.88 }, panes = [Kit.$("tr-lab"), Kit.$("tr-mask"), Kit.$("tr-con")];
    var tabs = Kit.chips(Kit.$("tr-tabs"), ["Labels", "Masked prediction", "Contrastive pairs"], 1, function (i) { st.tab = i; show(); });
    function show() { panes.forEach(function (p, i) { p.hidden = i !== st.tab; }); draw(); }
    // labels
    var ls = Kit.slider(Kit.$("tr-lab-controls"), { label: "Probability the network gives to the true label", min: 0.01, max: 0.99, step: 0.01, value: 0.88, digits: 2 }, function (v) { st.p = v; draw(); });
    // masked
    var phb = Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("tr-mask-bar")), ["Hammerhead, Costa Rica", "Grey bamboo shark, tank"], 0, function (i) { st.ph = i; draw(); });
    var RA = [0, 25, 50, 75, 90];
    var rs = Kit.slider(Kit.$("tr-mask-controls"), { label: "Share of patches hidden", min: 0, max: 4, step: 1, value: 3, show: function (i) { return RA[i] + "%"; } }, function (v) { st.ri = v; draw(); });
    var rec = new Image(), orig = {}, masked = Kit.$("tr-in"), out = Kit.$("tr-out");
    var names = Mae.photos;
    var cache = {};
    function get(src) { if (!cache[src]) { cache[src] = new Image(); cache[src].onload = draw; cache[src].src = src; } return cache[src]; }
    function draw() {
      var P = Kit.palette();
      if (st.tab === 2) Kit.$("tr-con-credit").textContent = ["hammerhead-1", "milk-1", "turtle-1"].map(function (k) { return CREDITS[k] || ""; }).join(" ");
      if (st.tab === 0) {
        var loss = -Math.log(st.p), classes = [["hammerhead", st.p], ["other classes together", 1 - st.p]];
        Kit.$("tr-lab-eq").innerHTML = ebox("c-teal", nf(st.p, 2), "probability on the true label, p") + eop("&rarr;") + ebox("c-coral", nf(loss, 2), "loss = &minus;ln p") + eop("|") + ebox("c-navy", nf(-Math.log(1 / 1000), 2), "loss of a network that guesses among 1,000 classes");
        return;
      }
      if (st.tab === 1) {
        var key = names[st.ph], r = RA[st.ri], keep = r === 0 ? 196 : Mae.keep[st.ri - 1], W = Math.min(240, Math.floor((Kit.width(Kit.$("tr-mask")) - 14) / 2));
        var oi = get("data/p224/" + key + ".jpg"), c1 = fitCanvas(masked, W, W), c2 = fitCanvas(out, W, W), cs = W / 14, i;
        if (oi.complete && oi.naturalWidth) {
          c1.drawImage(oi, 0, 0, W, W);
          var tc = hexRGB(P.paper);
          for (i = 0; i < 196; i++) if (Mae.rank[i] >= keep) { c1.fillStyle = rgb(mix(tc, hexRGB(P.muted), 0.35)); c1.fillRect((i % 14) * cs, Math.floor(i / 14) * cs, cs + 0.5, cs + 0.5); }
        }
        var ri = r === 0 ? oi : get("data/mae/" + key + "_" + r + ".jpg");
        if (ri.complete && ri.naturalWidth) c2.drawImage(ri, 0, 0, W, W);
        var err = r === 0 ? 0 : Mae.mae_abs_error[key][st.ri - 1];
        Kit.$("tr-mask-eq").innerHTML = ebox("c-sand", 196 - keep, "patches hidden, of 196") + eop("|") + ebox("c-navy", keep, "patches the encoder sees") + eop("|") + ebox("c-coral", r === 0 ? "none" : nf(err, 1), "average error of the guess, on a 0 to 255 scale");
        Kit.$("tr-mask-credit").textContent = (key === "hammerhead-2" ? CREDITS["hammerhead-2"] : CREDITS["bamboo-1"]) + " Cropped and resized to 224 by 224 pixels. Reconstruction by facebook/vit-mae-base (MAE, He and others, 2022; licences in the sources).";
        masked.setAttribute("aria-label", "The photo with " + (196 - keep) + " patches hidden"); out.setAttribute("aria-label", "The masked autoencoder's guess for the hidden patches");
      }
    }
    onResize(draw); show();
  }

  // ---------- 3. the similarity matrix and the contrastive loss ----------
  function matrixFig() {
    var C = D.clip, S = C.cos_caption, st = { batch: [0, 2, 4, 6, 8, 10, 11, 13], T: 0.01, view: 0, row: 3 }, box = Kit.$("mx-box"), svg = Kit.$("mx-svg");
    var thumbs = Kit.$("mx-thumbs"), btns = KEYS.map(function (k, i) {
      var b = thumb(thumbs, i, 44); b.addEventListener("click", function () {
        var at = st.batch.indexOf(i); if (at >= 0) { if (st.batch.length > 2) st.batch.splice(at, 1); } else { st.batch.push(i); st.batch.sort(function (a, b2) { return a - b2; }); }
        if (st.row >= st.batch.length) st.row = 0; draw();
      }); return b;
    });
    Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("mx-bar")), ["Cosines", "Softmax over captions (rows)", "Softmax over photos (columns)"], 0, function (i) { st.view = i; draw(); });
    var tb = Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("mx-bar")), ["T = 0.01 (learned)", "T = 0.1", "T = 1"], 0, function (i) { st.T = [0.01, 0.1, 1][i]; draw(); });
    Kit.button(Kit.$("mx-bar"), "Pick all 14", function () { st.batch = KEYS.map(function (k, i) { return i; }); draw(); });
    Kit.button(Kit.$("mx-bar"), "Pick 6 different animals", function () { st.batch = [0, 2, 4, 6, 10, 13]; st.row = 0; draw(); });
    function sub() { var b = st.batch; return b.map(function (i) { return b.map(function (j) { return S[i][j]; }); }); }
    function draw() {
      var P = Kit.palette(), b = st.batch, n = b.length, M = sub(), T = st.T, L = clipLoss(M, T), W = Math.min(640, Kit.width(box)), lab = 68, top = 24, cs = Math.max(20, Math.min(54, Math.floor((W - lab - 4) / n)));
      btns.forEach(function (bt, i) { bt.setAttribute("aria-pressed", String(b.indexOf(i) >= 0)); });
      svg.setAttribute("viewBox", "0 0 " + (lab + n * cs + 4) + " " + (top + n * cs + 6)); svg.style.maxWidth = (lab + n * cs + 4) + "px"; svg.innerHTML = "";
      var lo = Math.min.apply(null, M.map(function (r) { return Math.min.apply(null, r); })), hi = Math.max.apply(null, M.map(function (r) { return Math.max.apply(null, r); }));
      var rowP = M.map(function (r) { return softmax(r.map(function (v) { return v / T; })); }), colP = M.map(function (r, i) { return softmax(M.map(function (q) { return q[i] / T; })); });   // colP[j][i] = P(photo i | caption j)
      var base = hexRGB(P.paper), tgt = hexRGB(P.teal), i, j;
      for (j = 0; j < n; j++) { var t = Kit.el("text", { x: lab + j * cs + cs / 2, y: top - 8, "text-anchor": "middle", "font-size": 12, fill: P.ink2 }, svg); t.textContent = "c" + (j + 1); }
      for (i = 0; i < n; i++) {
        Kit.el("image", { href: "../data/sharks/photos/" + KEYS[b[i]] + ".jpg", x: 2, y: top + i * cs + 2, width: cs - 4 < 36 ? cs - 4 : 36, height: cs - 4 < 36 ? cs - 4 : 36, preserveAspectRatio: "xMidYMid slice" }, svg);
        var rt = Kit.el("text", { x: lab - 8, y: top + i * cs + cs / 2 + 4, "text-anchor": "end", "font-size": 12, fill: P.ink2 }, svg); rt.textContent = "p" + (i + 1);
        for (j = 0; j < n; j++) {
          var v = st.view === 0 ? (M[i][j] - lo) / (hi - lo + 1e-9) : st.view === 1 ? rowP[i][j] : colP[j][i];
          var g = Kit.el("g", { transform: "translate(" + (lab + j * cs) + "," + (top + i * cs) + ")", "data-r": i }, svg);
          var rc = Kit.el("rect", { x: 0.5, y: 0.5, width: cs - 1, height: cs - 1, rx: 3, fill: rgb(mix(base, tgt, Math.max(0.03, v))), stroke: i === st.row ? P.ink : "none", "stroke-width": 1.2 }, g);
          var val = st.view === 0 ? nf(M[i][j], 2) : (v >= 0.995 ? "1" : v < 0.005 ? "0" : nf(v, 2));
          if (cs >= 30) { var tx = Kit.el("text", { x: cs / 2, y: cs / 2 + 4, "text-anchor": "middle", "font-size": cs >= 40 ? 12.5 : 11, fill: v > 0.55 ? "#fff" : P.ink }, g); tx.textContent = val; }
          var ti = Kit.el("title", {}, g); ti.textContent = "photo " + (i + 1) + ", caption " + (j + 1) + ": cosine " + nf(M[i][j], 3) + ", softmax over captions " + nf(rowP[i][j], 3) + ", over photos " + nf(colP[j][i], 3);
          if (i === j) Kit.el("rect", { x: 1.5, y: 1.5, width: cs - 3, height: cs - 3, rx: 3, fill: "none", stroke: P.coral, "stroke-width": 2.4 }, g);
          g.style.cursor = "pointer"; (function (ii) { g.addEventListener("click", function () { st.row = ii; draw(); }); })(i);
        }
      }
      // caption legend
      var leg = Kit.$("mx-legend"); leg.innerHTML = "";
      b.forEach(function (idx, k) { var li = Kit.html("li", {}, leg); Kit.html("b", {}, li, "c" + (k + 1) + " "); li.appendChild(document.createTextNode("“" + C.caption[idx] + "”")); });
      // loss boxes
      var avgImg = L.img, avgTxt = L.txt;
      Kit.$("mx-loss").innerHTML = ebox("c-navy", nf(L.img, 3), "photo to caption, mean of &minus;ln p over the " + n + " rows") + eop("+") + ebox("c-coral", nf(L.txt, 3), "caption to photo, mean over the " + n + " columns") + eop("&divide; 2 =") + ebox("c-teal", nf(L.total, 3), "loss for this batch") + eop("|") + ebox("c-sand", nf(Math.log(n), 3), "chance level, ln " + n);
      // worked row
      var r = st.row, p = rowP[r][r], pc = colP[r][r];
      Kit.$("mx-work").innerHTML = eop("Row p" + (r + 1) + ":") + ebox("c-navy", nf(M[r][r], 3), "cosine on the diagonal") + eop("&divide; T =") + ebox("c-navy", nf(M[r][r] / T, 1), "logit, T = " + T) + eop("&rarr; softmax over the row &rarr;") + ebox("c-teal", p >= 0.9995 ? "1.000" : nf(p, 3), "p of its own caption") + eop("&rarr;") + ebox("c-coral", nf(-Math.log(p), 3), "loss of the row, &minus;ln p") + eop("|") + ebox("c-sand", nf(-Math.log(pc), 3), "loss of column c" + (r + 1));
      var best = M[r].indexOf(Math.max.apply(null, M[r]));
      Kit.$("mx-status").textContent = "Batch of " + n + " pairs. Click a row to see its working. In row p" + (r + 1) + " the highest cosine is in column c" + (best + 1) + (best === r ? ", which is its own caption." : ", which is not its own caption (c" + (r + 1) + "): the loss is high.") + " Own caption is the top choice in " + M.filter(function (row, k) { return row.indexOf(Math.max.apply(null, row)) === k; }).length + " of " + n + " rows.";
      svg.setAttribute("aria-label", "Similarity matrix of " + n + " photos and " + n + " captions with the diagonal outlined; loss " + nf(L.total, 2));
    }
    creditsList("mx-credits");
    onResize(draw); draw();
  }

  // ---------- 4. temperature ----------
  function tempFig() {
    var C = D.clip, st = { ph: 6, lt: -2 }, S = C.cos_species[1], names = C.class_names, truth = C.cls.map(function (c) { return C.classes.indexOf(c); });
    var pick = [6, 2, 4, 11, 13, 8];
    var row = Kit.html("div", { class: "bar-row" }, Kit.$("tp-bar"));
    Kit.chips(row, pick.map(function (i) { return NICE[i]; }), 0, function (i) { st.ph = pick[i]; draw(); });
    var sl = Kit.slider(Kit.$("tp-controls"), { label: "Temperature T (log scale)", min: -3, max: 0, step: 0.05, value: -2, show: function (v) { return "T = " + (Math.pow(10, v) < 0.01 ? Math.pow(10, v).toFixed(4) : Math.pow(10, v).toFixed(3)); } }, function (v) { st.lt = v; draw(); });
    Kit.button(Kit.$("tp-bar"), "Set T = 0.01 (the learned value)", function () { st.lt = -2; sl.set(-2); draw(); });
    function draw() {
      var T = Math.pow(10, st.lt), cos = S[st.ph], z = cos.map(function (v) { return v / T; }), p = softmax(z), ti = truth[st.ph], H = entropy(p), top = p.indexOf(Math.max.apply(null, p));
      softBars(Kit.$("tp-bars"), names, p, { good: ti, mark: top });
      var ord = cos.map(function (v, i) { return i; }).sort(function (a, b) { return cos[b] - cos[a]; });
      Kit.$("tp-eq").innerHTML = ebox("c-navy", nf(cos[ord[0]], 3), "cosine of the top caption") + eop("&minus;") + ebox("c-navy", nf(cos[ord[1]], 3), "cosine of the second") + eop("=") + ebox("c-sand", nf(cos[ord[0]] - cos[ord[1]], 3), "gap in cosine") + eop("&divide; T =") + ebox("c-coral", nf((cos[ord[0]] - cos[ord[1]]) / T, 2), "gap in logits") + eop("&rarr; odds") + ebox("c-teal", Math.exp((cos[ord[0]] - cos[ord[1]]) / T) > 9999 ? "&gt; 9,999" : nf(Math.exp((cos[ord[0]] - cos[ord[1]]) / T), 2), "top : second, e<sup>gap</sup>");
      Kit.$("tp-read").innerHTML = "<span>Top caption: <b>" + names[top] + "</b> at <b>" + Kit.pct(p[top], 1) + "</b>. Effective number of captions in play, e<sup>H</sup>: <b>" + nf(Math.exp(H), 2) + "</b> of 9 (<a href=\"../d3-next-token/#distribution\">D3</a>).</span>";
      Kit.$("tp-status").textContent = "The teal bar is the caption that matches the photo, if it is among the nine. At T = 1 the nine probabilities are close to equal; at T = 0.001 the top caption takes nearly everything.";
    }
    draw();
  }

  // ---------- 5. the shared space ----------
  function spaceFig() {
    var C = D.clip, M = C.map, items = M.items, st = { mode: 0, sel: 6, label: false }, svg = d3.select("#sp-svg");
    // three layouts: as computed, zoomed on the photos (same map, rescaled), each group centred (a second PCA)
    var raw = M.raw, ph = raw.slice(0, 14), xs = ph.map(function (p) { return p[0]; }), ys = ph.map(function (p) { return p[1]; });
    var cx = (Math.min.apply(null, xs) + Math.max.apply(null, xs)) / 2, cy = (Math.min.apply(null, ys) + Math.max.apply(null, ys)) / 2;
    var hh = Math.max(Math.max.apply(null, xs) - Math.min.apply(null, xs), Math.max.apply(null, ys) - Math.min.apply(null, ys)) / 2 * 1.3;
    var LAY = [raw.map(function (p) { return p.slice(); }), raw.map(function (p) { return [(p[0] - cx) / hh, (p[1] - cy) / hh]; }), M.centred.map(function (p) { return p.slice(); })];
    var cur = LAY[0].map(function (p) { return p.slice(); }), stopA = null;
    Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("sp-bar")), ["As computed", "Zoom on the photos", "Groups centred"], 0, function (i) { st.mode = i; move(); });
    var lab = Kit.html("label", { class: "check" }, Kit.$("sp-bar")), chk = Kit.html("input", { type: "checkbox" }, lab); lab.appendChild(document.createTextNode(" label all texts"));
    chk.addEventListener("change", function () { st.label = chk.checked; draw(); });
    function move() {
      if (stopA) stopA(); var from = cur.map(function (p) { return p.slice(); }), to = LAY[st.mode];
      stopA = Kit.animate(700, function (p) { var e = Kit.ease(p); cur = from.map(function (f, i) { return [f[0] + (to[i][0] - f[0]) * e, f[1] + (to[i][1] - f[1]) * e]; }); draw(); });
      Kit.$("sp-read").innerHTML = [
        "<span>The first axis keeps <b>" + Kit.pct(M.var_raw[0], 0) + "</b> of the spread and separates photos from texts. The photos are so alike that they pile up on the left. The average photo vector and the average text vector are <b>" + nf(M.gap, 2) + "</b> apart (unit vectors are at most 2 apart).</span>",
        "<span>The same map, rescaled so that the photos fill the frame. The texts are far to the right, out of view. The photos are in one tight cloud: photo to photo cosines average <b>" + nf(C.photo_photo_mean_offdiag, 2) + "</b>.</span>",
        "<span>With the average of the group removed, the first axis keeps <b>" + Kit.pct(M.var_centred[0], 0) + "</b> of the spread. Photos and texts about the same subject now lie nearer each other. This is a second PCA, computed on the centred vectors.</span>"][st.mode];
    }
    function draw() {
      var P = Kit.palette(), W = Math.min(680, Kit.width(Kit.$("sp-svg").parentNode)), H = Math.round(Math.min(440, W * 0.75)), pad = 30;
      svg.attr("viewBox", "0 0 " + W + " " + H).style("overflow", "hidden").selectAll("*").remove();
      var X = function (v) { return pad + (v + 1) / 2 * (W - 2 * pad); }, Y = function (v) { return pad + (1 - (v + 1) / 2) * (H - 2 * pad - 14); };
      var g = svg.append("g");
      g.append("line").attr("x1", pad).attr("x2", W - pad).attr("y1", Y(0)).attr("y2", Y(0)).attr("stroke", P.rule);
      g.append("line").attr("x1", X(0)).attr("x2", X(0)).attr("y1", pad).attr("y2", H - pad).attr("stroke", P.rule);
      var near = nearest(st.sel, 3), nearIds = {}; near.forEach(function (n) { nearIds[n.i] = true; });
      near.forEach(function (n) { var x2 = X(cur[n.i][0]), y2 = Y(cur[n.i][1]); if (x2 < -20 || x2 > W + 20 || y2 < -20 || y2 > H + 20) return; g.append("line").attr("x1", X(cur[st.sel][0])).attr("y1", Y(cur[st.sel][1])).attr("x2", x2).attr("y2", y2).attr("stroke", P.teal).attr("stroke-width", 2).attr("stroke-dasharray", "4 3"); });
      var boxes = [], lblOK = {};   // labels are placed one by one and skipped when they would cover an earlier label
      function tryLabel(i, force) {
        var it = items[i]; if (it.kind === "photo") return;
        var x = X(cur[i][0]), y = Y(cur[i][1]); if (x < -30 || x > W + 30 || y < -30 || y > H + 30) return;
        var short = it.label.replace("a photo of a ", "").replace("a photo of an ", ""); if (short.length > 26) short = short.slice(0, 25) + "\u2026";
        var w = 6.4 * short.length + 6, r = i === st.sel ? 9 : 7, flip = x > W - 140, b = flip ? [x - r - 3 - w, y - 10, x - r - 3, y + 5] : [x + r + 3, y - 10, x + r + 3 + w, y + 5];
        if (!force) for (var q = 0; q < boxes.length; q++) if (b[0] < boxes[q][2] && b[2] > boxes[q][0] && b[1] < boxes[q][3] && b[3] > boxes[q][1]) return;
        boxes.push(b); lblOK[i] = true;
      }
      tryLabel(st.sel, true); near.forEach(function (n) { tryLabel(n.i, false); });
      if (st.mode === 2) items.forEach(function (it, i) { if (it.kind === "prompt") tryLabel(i, false); });
      if (st.label) items.forEach(function (it, i) { tryLabel(i, false); });
      items.forEach(function (it, i) {
        var x = X(cur[i][0]), y = Y(cur[i][1]);
        if (x < -30 || x > W + 30 || y < -30 || y > H + 30) return;
        var node = g.append("g").style("cursor", "pointer").on("click", function () { st.sel = i; draw(); info(); });
        node.append("title").text(it.kind + ": " + it.label);
        if (it.kind === "photo") {
          node.append("circle").attr("cx", x).attr("cy", y).attr("r", i === st.sel ? 18 : 15).attr("fill", P.paper).attr("stroke", i === st.sel ? P.coral : P.navy).attr("stroke-width", i === st.sel ? 3 : 2);
          node.append("clipPath").attr("id", "sp-c" + i).append("circle").attr("cx", x).attr("cy", y).attr("r", 13);
          node.append("image").attr("href", "../data/sharks/photos/" + it.label + ".jpg").attr("x", x - 16).attr("y", y - 16).attr("width", 32).attr("height", 32).attr("preserveAspectRatio", "xMidYMid slice").attr("clip-path", "url(#sp-c" + i + ")");
        } else {
          var fill = it.kind === "caption" ? P.coral : it.kind === "prompt" ? P.sand : P.muted, r = i === st.sel ? 9 : 7;
          node.append("path").attr("d", "M" + x + "," + (y - r) + " L" + (x + r) + "," + y + " L" + x + "," + (y + r) + " L" + (x - r) + "," + y + " Z").attr("fill", fill).attr("stroke", i === st.sel ? P.ink : "none").attr("stroke-width", 2);
          if (lblOK[i]) {
            var short = it.label.replace("a photo of a ", "").replace("a photo of an ", ""); if (short.length > 26) short = short.slice(0, 25) + "\u2026";
            var flip = x > W - 140;
            node.append("text").attr("x", flip ? x - r - 3 : x + r + 3).attr("y", y + 4).attr("text-anchor", flip ? "end" : "start").attr("font-size", 12).attr("fill", P.ink2).attr("paint-order", "stroke").attr("stroke", P.bg).attr("stroke-width", 3).text(short);
          }
        }
      });
      var ky = H - 6, kx = pad; [["photo", P.navy, "circle"], ["batch caption", P.coral, "d"], ["prompt", P.sand, "d"], ["other text", P.muted, "d"]].forEach(function (k, n) {
        if (k[2] === "circle") svg.append("circle").attr("cx", kx).attr("cy", ky - 4).attr("r", 5).attr("fill", "none").attr("stroke", k[1]).attr("stroke-width", 2); else svg.append("path").attr("d", "M" + kx + "," + (ky - 9) + " l5,5 l-5,5 l-5,-5 Z").attr("fill", k[1]);
        svg.append("text").attr("x", kx + 9).attr("y", ky).attr("font-size", 12).attr("fill", P.ink2).text(k[0]);
        kx += 9 + 6.6 * k[0].length + 16;
      });
      svg.attr("aria-label", "Two-dimensional map of 14 photos and 32 texts. " + ["Photos and texts form two separate groups, and the photos pile up.", "Zoomed on the photos.", "Photos and texts are mixed after the average of the groups is removed."][st.mode]);
    }
    function nearest(i, k) {
      var kindSel = items[i].kind === "photo", out = [];
      items.forEach(function (it, j) { if (j === i) return; if ((it.kind === "photo") === kindSel) return; out.push({ i: j, c: M.cos[i][j] }); });
      out.sort(function (a, b) { return b.c - a.c; }); return out.slice(0, k);
    }
    function info() {
      var it = items[st.sel], n = nearest(st.sel, 5), title = it.kind === "photo" ? NICE[KEYS.indexOf(it.label)] + " (photo)" : "\u201c" + it.label + "\u201d (" + (it.kind === "caption" ? "batch caption" : it.kind === "prompt" ? "prompt" : "other text") + ")";
      var html = "<p class=\"sel-h\"><b>" + title + "</b></p><p>Nearest " + (it.kind === "photo" ? "texts" : "photos") + " by cosine of the 512 numbers:</p><ol>";
      n.forEach(function (x) { var o = items[x.i]; html += "<li>" + (o.kind === "photo" ? NICE[KEYS.indexOf(o.label)] : "\u201c" + o.label + "\u201d") + " <span>" + nf(x.c, 3) + "</span></li>"; });
      Kit.$("sp-info").innerHTML = html + "</ol>";
    }
    creditsList("sp-credits");
    onResize(draw); move(); info();
  }

  // ---------- 6. zero-shot classification ----------
  function zeroFig() {
    var C = D.clip, st = { ph: 6, t: 1, on: C.classes.map(function () { return true; }).concat(C.generic.map(function () { return false; })) }, truth = C.cls.map(function (c) { return C.classes.indexOf(c); });
    var thumbs = Kit.$("zs-thumbs"), btns = KEYS.map(function (k, i) { var b = thumb(thumbs, i, 46, "mark"); b.addEventListener("click", function () { st.ph = i; draw(); }); return b; });
    Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("zs-bar")), TL, 1, function (i) { st.t = i; draw(); });
    var pr = Kit.html("div", { class: "bar-row" }, Kit.$("zs-bar")); Kit.html("span", { class: "label" }, pr, "Presets:");
    function preset(on) { st.on = C.classes.map(function (c, i) { return on.indexOf(i) >= 0; }).concat(C.generic.map(function (g, i) { return on.indexOf(100 + i) >= 0; })); sync(); draw(); }
    Kit.button(pr, "All nine", function () { preset([0, 1, 2, 3, 4, 5, 6, 7, 8]); });
    Kit.button(pr, "Five sharks", function () { preset([0, 1, 2, 3, 4]); });
    Kit.button(pr, "Shark, fish or person", function () { preset([100, 101, 108]); });
    Kit.button(pr, "Nine and two kinds of water", function () { preset([0, 1, 2, 3, 4, 5, 6, 7, 8, 106, 107]); });
    var boxes = Kit.$("zs-checks"), checks = [];
    C.classes.forEach(function (c, i) { checks.push(mk(C.class_names[i], i)); });
    C.generic.forEach(function (g, i) { checks.push(mk(g, 9 + i)); });
    function mk(text, idx) { var l = Kit.html("label", { class: "check" }, boxes), c = Kit.html("input", { type: "checkbox" }, l); l.appendChild(document.createTextNode(" " + text)); c.addEventListener("change", function () { st.on[idx] = c.checked; draw(); }); return c; }
    function sync() { checks.forEach(function (c, i) { c.checked = st.on[i]; }); }
    function cand() { var out = [], i; for (i = 0; i < 9; i++) if (st.on[i]) out.push({ name: C.class_names[i], cls: i, label: C.species_texts[st.t][i], col: i }); for (i = 0; i < 9; i++) if (st.on[9 + i]) out.push({ name: C.generic[i], cls: -1, label: C.generic[i], gen: i }); return out; }
    function cosOf(ph, c) { return c.gen != null ? C.cos_generic[ph][c.gen] : C.cos_species[st.t][ph][c.col]; }
    function probs(ph, cs) { return softmax(cs.map(function (c) { return cosOf(ph, c) / 0.01; })); }
    function draw() {
      var cs = cand(), P = Kit.palette();
      if (cs.length < 2) { Kit.$("zs-bars").innerHTML = "<p class=\"hintline\">Tick at least two captions.</p>"; Kit.$("zs-eq").innerHTML = ""; Kit.$("zs-read").innerHTML = ""; return; }
      var p = probs(st.ph, cs), ti = -1; cs.forEach(function (c, i) { if (c.cls === truth[st.ph]) ti = i; });
      var top = p.indexOf(Math.max.apply(null, p));
      softBars(Kit.$("zs-bars"), cs.map(function (c) { return c.gen != null ? c.name : c.label; }), p, { good: ti, mark: top, bad: ti >= 0 && top !== ti ? top : undefined });
      // right or wrong for all photos with these captions
      var right = 0, tot = 0;
      btns.forEach(function (b, i) {
        b.setAttribute("aria-pressed", String(i === st.ph));
        var ix = -1; cs.forEach(function (c, k) { if (c.cls === truth[i]) ix = k; });
        b.classList.remove("ok", "no", "na");
        if (ix < 0) { b.classList.add("na"); b.title = NICE[i] + ": its true caption is not in the list"; return; }
        var pp = probs(i, cs), t = pp.indexOf(Math.max.apply(null, pp)); tot++; if (t === ix) { right++; b.classList.add("ok"); b.title = NICE[i] + ": right"; } else { b.classList.add("no"); b.title = NICE[i] + ": wrong, the top caption is " + cs[t].name; }
      });
      var s = cs.slice().map(function (c, i) { return { c: c, i: i, v: cosOf(st.ph, c) }; }).sort(function (a, b) { return b.v - a.v; });
      Kit.$("zs-eq").innerHTML = ebox("c-navy", nf(s[0].v, 3), "cosine of the top caption") + eop("&divide; 0.01 =") + ebox("c-coral", nf(s[0].v / 0.01, 1), "logit") + eop("&rarr; softmax over " + cs.length + " captions &rarr;") + ebox("c-teal", Kit.pct(p[s[0].i], 0), "probability of the top caption");
      Kit.$("zs-read").innerHTML = tot === 0 ? "<span>None of the 14 photos has its true class on this list, so no dot is drawn. The model still has to pick one of the captions.</span>" : "<span>With these captions and this template the model is right on <b>" + right + "</b> of <b>" + tot + "</b> photos whose true caption is on the list. Teal dots mark right calls, coral dots wrong ones, no dot means the true caption is not offered.</span>";
      Kit.$("zs-status").textContent = ti < 0 ? "The true class of this photo is not among the captions, so the model must choose a wrong one." : (top === ti ? "The model picks the right caption." : "The model picks “" + cs[top].name + "” but the photo shows " + C.class_names[truth[st.ph]] + ".");
      Kit.$("zs-credit").textContent = credit(st.ph);
    }
    creditsList("zs-credits");
    sync(); draw();
  }

  // ---------- 7. the same in murky water ----------
  function murkyFig() {
    var C = D.clip, Mu = C.murky, st = { ph: 0, di: 0, t: 1 }, truth = { "hammerhead-1": "hammerhead", "hammerhead-2": "hammerhead", "milk-1": "milk", "milk-2": "milk", "spadenose-1": "spadenose", "dolphin-1": "dolphin", "reeffish-1": "clownfish", "turtle-1": "turtle" };
    var names = ["Hammerhead, Jarvis Island", "Hammerhead, Costa Rica", "Milk sharks, Mangalore", "Milk sharks, Kuala Lumpur", "Spadenose on a board", "Dolphin (not a shark)", "Clownfish (not a shark)", "Sea turtle (not a shark)"];
    var cv = Kit.$("mu-cv"), work = document.createElement("canvas"), W0 = 360, src = null, out = null;
    Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("mu-bar")), names, 0, function (i) { st.ph = i; load(); });
    Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("mu-bar")), TL, 1, function (i) { st.t = i; draw(); });
    var sl = Kit.slider(Kit.$("mu-controls"), { label: "Depth, m", min: 0, max: 15, step: 1, value: 0, show: function (i) { return Mu.depths[i] + " m"; } }, function (i) { st.di = i; draw(); });
    function load() {
      var im = IMG[Mu.photos[st.ph]], H = Math.round(W0 * im.naturalHeight / im.naturalWidth); work.width = W0; work.height = H; var c = work.getContext("2d", { willReadFrequently: true }); c.drawImage(im, 0, 0, W0, H);
      src = c.getImageData(0, 0, W0, H); out = new ImageData(W0, H); draw();
    }
    function probsAt(ph, di, t) { return softmax(Mu.cos_species[t][ph][di].map(function (v) { return v / 0.01; })); }
    function trueIdx(ph) { return C.classes.indexOf(truth[Mu.photos[ph]]); }
    function sharkMass(p) { return SHARK_CLASSES.reduce(function (s, i) { return s + p[i]; }, 0); }
    function draw() {
      if (!src) return;
      var P = Kit.palette(), z = Mu.depths[st.di], H = src.height; degradeData(src.data, out.data, z);
      var ctx = fitCanvas(cv, Math.min(W0, Kit.width(Kit.$("mu-box")) - 4), Math.round(Math.min(W0, Kit.width(Kit.$("mu-box")) - 4) * H / W0)); var tmp = document.createElement("canvas"); tmp.width = W0; tmp.height = H; tmp.getContext("2d").putImageData(out, 0, 0); ctx.drawImage(tmp, 0, 0, parseFloat(cv.style.width), parseFloat(cv.style.height));
      var p = probsAt(st.ph, st.di, st.t), ti = trueIdx(st.ph), top = p.indexOf(Math.max.apply(null, p)), ph = Mu.photos[st.ph];
      softBars(Kit.$("mu-bars"), C.class_names, p, { good: ti, mark: top, bad: top !== ti ? top : undefined });
      var xs = Mu.depths, clipTrue = xs.map(function (d, i) { return probsAt(st.ph, i, st.t)[ti]; }), clipShark = xs.map(function (d, i) { return sharkMass(probsAt(st.ph, i, st.t)); });
      Kit.plot("mu-plot", function (Q) {
        return [
          { x: xs, y: clipTrue, mode: "lines+markers", line: { color: Q.teal, width: 3 }, marker: { size: 5 }, hovertemplate: "%{x} m: %{y:.2f}<extra>CLIP, true species</extra>" },
          { x: xs, y: clipShark, mode: "lines", line: { color: Q.navy, width: 2.5, dash: "dot" }, hovertemplate: "%{x} m: %{y:.2f}<extra>CLIP, any shark caption</extra>" },
          { x: xs, y: Mu.resnet_shark[ph], mode: "lines", line: { color: Q.sand, width: 2.5, dash: "dash" }, hovertemplate: "%{x} m: %{y:.2f}<extra>ResNet-18 shark classes (C3)</extra>" },
          { x: [z], y: [clipTrue[st.di]], mode: "markers", marker: { color: Q.coral, size: 11, line: { color: "#fff", width: 2 } }, hoverinfo: "skip" }
        ];
      }, function (Q) { return { xaxis: { title: { text: "depth, m" }, range: [-0.5, 30.5] }, yaxis: { title: { text: "probability" }, range: [-0.03, 1.05] }, margin: { l: 52, r: 12, t: 10, b: 46 } }; });
      // accuracy of the eight photos by depth for the three templates
      var acc = [0, 1, 2].map(function (t) { return xs.map(function (d, i) { var ok = 0; Mu.photos.forEach(function (k, j) { var pp = probsAt(j, i, t); if (pp.indexOf(Math.max.apply(null, pp)) === trueIdx(j)) ok++; }); return 100 * ok / Mu.photos.length; }); });
      Kit.plot("mu-plot2", function (Q) {
        return acc.map(function (a, t) { return { x: xs, y: a, mode: "lines+markers", line: { color: [Q.sand, Q.teal, Q.coral][t], width: t === st.t ? 3.5 : 2, dash: t === st.t ? "solid" : "dot" }, marker: { size: 5 }, hovertemplate: "%{x} m: %{y:.0f}%<extra>" + TL[t] + "</extra>" }; });
      }, function (Q) { return { xaxis: { title: { text: "depth, m" }, range: [-0.5, 30.5] }, yaxis: { title: { text: "species right, % of 8 photos" }, range: [-3, 103] }, margin: { l: 56, r: 12, t: 10, b: 46 } }; });
      var Qp = Kit.palette();
      setKey("mu-key1", [{ color: Qp.teal, text: "CLIP: probability of the true species" }, { color: Qp.navy, text: "CLIP: any shark caption", style: "dot" }, { color: Qp.sand, text: "ResNet-18: shark classes (C3)", style: "dash" }]);
      setKey("mu-key2", TL.map(function (t, k) { return { color: [Qp.sand, Qp.teal, Qp.coral][k], text: t, style: k === st.t ? "" : "dot" }; }));
      var G = Mu.cos_generic[st.ph][st.di];
      Kit.$("mu-read").innerHTML = "<span>At <b>" + z + " m</b> the top caption is <b>" + C.class_names[top] + "</b> (" + Kit.pct(p[top], 0) + "). Probability on the true species: <b>" + Kit.pct(p[ti], 0) + "</b>. Any shark caption: <b>" + Kit.pct(sharkMass(p), 0) + "</b>. ResNet-18 shark classes: <b>" + Kit.pct(Mu.resnet_shark[ph][st.di], 0) + "</b>.</span><span>Cosine with “a murky underwater photo”: <b>" + nf(G[7], 3) + "</b>; with “a clear blue underwater photo”: <b>" + nf(G[6], 3) + "</b>.</span>";
      Kit.$("mu-credit").textContent = CREDITS[ph] + " Depth effect made with depth.py, a simple model.";
      cv.setAttribute("aria-label", names[st.ph] + " at " + z + " m");
    }
    load(); onResize(draw);
  }
})();
