// E3 Cross-attention.
//   sideFig    SIGNATURE: self-attention and cross-attention side by side, queries from the words, keys and values from the words or the photo, step by step (BLIP weights)
//   workedFig  four patches by hand: edit the query of a word and watch the weights, the output and the equation boxes change
//   heatFig    SIGNATURE: click a word of a caption, see its weights over the 24 by 24 patches of the photo (BLIP, 12 layers, layer stops chosen)
//   emptyFig   the 'empty' patches: share of weight against share of what is actually added, per layer
//   headFig    the 12 heads of the last layer for one word
//   occFig     does the attention say which patches matter? hide patches and watch the word's log probability
//   costFig    what the image costs: image tokens in the sequence against cross-attention
// Pure helpers are exported with module.exports for the node test.
(function () {
  "use strict";

  // ---------- pure helpers ----------
  function softmax(z) { var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - m); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (v) { return v / s; }); }
  function dot(a, b) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  // one query against a list of keys and values: scores, weights, output (the formula of the page)
  function attendOne(q, K, V, scale) {
    var dk = q.length, sc = K.map(function (k) { return dot(q, k) / (scale ? Math.sqrt(dk) : 1); }), a = softmax(sc), out = [], j, i;
    for (i = 0; i < V[0].length; i++) { var s = 0; for (j = 0; j < V.length; j++) s += a[j] * V[j][i]; out.push(s); }
    return { dots: K.map(function (k) { return dot(q, k); }), scores: sc, weights: a, out: out };
  }
  function dequant(u, mx, levels) { var x = u / levels; return mx * x * x; }
  function neff(p) { var s = 0, i, h = 0; for (i = 0; i < p.length; i++) s += p[i]; for (i = 0; i < p.length; i++) { if (p[i] > 0) { var q = p[i] / s; h -= q * Math.log(q); } } return Math.exp(h); }
  // share of a weight vector in a set (mask) and outside, with the 'empty' patches optionally set aside
  function shares(w, empty, animal, hide) {
    var e = 0, a = 0, o = 0, i;
    for (i = 0; i < w.length; i++) { if (empty[i]) e += w[i]; else if (animal[i]) a += w[i]; else o += w[i]; }
    var tot = hide ? a + o : e + a + o;
    return { empty: hide ? 0 : e / tot, animal: a / tot, other: o / tot };
  }
  // pairs and sizes for the cost figure: image tokens in the sequence against cross-attention (per layer, per head)
  function costs(nImg, nTxt, d) {
    var seqPairs = nImg * nImg + nTxt * nImg + nTxt * (nTxt + 1) / 2;
    var crossPairs = nTxt * nImg + nTxt * (nTxt + 1) / 2;
    var seqMA = nImg * 12 * d * d;                       // the image tokens go through a whole block: attention matrices 4 d^2 and MLP 8 d^2 per token
    var crossMA = (nImg + nTxt) * 2 * d * d;             // keys and values of the image (2 d^2 per image token), query and output of the words (2 d^2 per word)
    return { seqPairs: seqPairs, crossPairs: crossPairs, seqMA: seqMA, crossMA: crossMA, kv: 2 * nImg * d };
  }
  if (typeof module !== "undefined") { module.exports = { softmax: softmax, dot: dot, attendOne: attendOne, dequant: dequant, neff: neff, shares: shares, costs: costs }; return; }

  // ---------- page helpers ----------
  var PH = [
    { key: "hammerhead-1", name: "Hammerhead, open water" },
    { key: "bamboo-1", name: "Bamboo shark, tank" },
    { key: "spadenose-1", name: "Spadenose, white board" },
    { key: "turtle-1", name: "Turtle, beach" }
  ];
  var DATA = {}, IMG = {}, TILE = {}, HEAD = {}, OTILE = {}, CREDITS = {};
  var MAXW = 560;
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function nf(v, d) { return Kit.fmt(v, d == null ? 2 : d); }
  function pc(v, d) { return (100 * v).toFixed(d == null ? 0 : d) + "%"; }
  function comma(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")"; }
  function mix(a, b, t) { return [Math.round(a[0] + (b[0] - a[0]) * t), Math.round(a[1] + (b[1] - a[1]) * t), Math.round(a[2] + (b[2] - a[2]) * t)]; }
  function loadImg(src) { return new Promise(function (ok, fail) { var im = new Image(); im.onload = function () { ok(im); }; im.onerror = function () { fail(new Error("image " + src)); }; im.src = src; }); }
  function loadGray(src) {
    return loadImg(src).then(function (im) {
      var c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight;
      var x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(im, 0, 0);
      var d = x.getImageData(0, 0, c.width, c.height).data, a = new Uint8Array(c.width * c.height), i;
      for (i = 0; i < a.length; i++) a[i] = d[i * 4];
      return { w: c.width, h: c.height, a: a };
    });
  }
  function fitCanvas(cv, w, h) {
    var d = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(w * d); cv.height = Math.round(h * d); cv.style.width = w + "px"; cv.style.height = h + "px";
    var ctx = cv.getContext("2d"); ctx.setTransform(d, 0, 0, d, 0, 0); return ctx;
  }
  function setPressed(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function onResize(fn) { window.addEventListener("resize", Kit.debounce(fn, 120)); Kit.onTheme(fn); }
  function setById(id) { return DATA.blip.sets.filter(function (s) { return s.id === id; })[0]; }
  function setFor(photo, who) { return setById(photo + "-" + who); }
  function wordLabel(w) { return w === "end" ? "[end]" : w; }
  var NOUNS = { hammerhead: 1, shark: 1, fish: 1, turtle: 1 };
  function nounIndex(S) { var i; for (i = 0; i < S.words.length; i++) if (NOUNS[S.words[i]]) return i; return 0; }
  // 576 weights of one tile (row r, layer stop s) of a set
  function tile(S, r, s) {
    var T = TILE[S.id], out = new Float64Array(576), i, j, x0 = s * 24, y0 = r * 24, mx = S.mx[r][s];
    for (i = 0; i < 24; i++) for (j = 0; j < 24; j++) out[i * 24 + j] = dequant(T.a[(y0 + i) * T.w + x0 + j], mx, 255);
    return out;
  }
  function headTile(S, r, h) {
    var T = HEAD[S.id], out = new Float64Array(576), i, j, x0 = h * 24, y0 = r * 24, mx = S.hmx[r][0][h];
    for (i = 0; i < 24; i++) for (j = 0; j < 24; j++) out[i * 24 + j] = dequant(Math.round(T.a[(y0 + i) * T.w + x0 + j] / 17), mx, 15);
    return out;
  }
  function flags(photo) {
    var P = DATA.blip.photos[photo], e = new Uint8Array(576), a = new Uint8Array(576);
    P.empty.forEach(function (i) { e[i] = 1; }); P.animal.forEach(function (i) { a[i] = 1; });
    return { empty: e, animal: a };
  }
  // draw a photo into a canvas at a width, return {w, h}
  function drawPhoto(cv, photo, maxw) {
    var im = IMG[photo], w = Math.min(maxw, Kit.width(cv.parentNode)), h = Math.round(w * im.naturalHeight / im.naturalWidth), ctx = fitCanvas(cv, w, h);
    ctx.drawImage(im, 0, 0, w, h); return { ctx: ctx, w: w, h: h };
  }
  // heat overlay of a 24 by 24 grid on a canvas of w by h
  function heatOver(ctx, w, h, vals, empty, hide, P, opts) {
    opts = opts || {};
    var mx = 0, i, r, c, col = hexRGB(opts.color || P.coral), cw = w / 24, ch = h / 24;
    for (i = 0; i < 576; i++) if (!(hide && empty[i]) && vals[i] > mx) mx = vals[i];
    for (i = 0; i < 576; i++) {
      r = Math.floor(i / 24); c = i % 24;
      if (hide && empty[i]) { ctx.strokeStyle = rgba(hexRGB(P.bg), 0.55); ctx.lineWidth = 1; ctx.strokeRect(c * cw + 1.5, r * ch + 1.5, cw - 3, ch - 3); continue; }
      var t = mx > 0 ? Math.sqrt(vals[i] / mx) : 0;
      if (t < 0.02) continue;
      ctx.fillStyle = rgba(col, Math.min(0.9, 0.9 * t)); ctx.fillRect(c * cw, r * ch, cw + 0.6, ch + 0.6);
    }
    return mx;
  }
  function outline(ctx, w, h, poly, P) {
    ctx.save(); ctx.setLineDash([5, 3]); ctx.lineWidth = 1.6; ctx.strokeStyle = P.teal; ctx.beginPath();
    poly.forEach(function (p, i) { var x = p[0] / 24 * w, y = p[1] / 24 * h; if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
    ctx.closePath(); ctx.stroke(); ctx.restore();
  }

  document.addEventListener("DOMContentLoaded", function () {
    var imgs = PH.map(function (p) { return loadImg("../data/sharks/photos/" + p.key + ".jpg"); });
    Kit.data("data/blip.json").then(function (B) {
      DATA.blip = B;
      var loads = [Kit.data("data/occ.json"), Kit.data("../data/sharks/photos.json")];
      B.sets.forEach(function (s) { loads.push(loadGray("data/t-" + s.id + ".png").then(function (g) { TILE[s.id] = g; })); loads.push(loadGray("data/h-" + s.id + ".png").then(function (g) { HEAD[s.id] = g; })); });
      PH.forEach(function (p) { loads.push(loadGray("data/o-" + p.key + ".png").then(function (g) { OTILE[p.key] = g; })); });
      return Promise.all([Promise.all(imgs), Promise.all(loads)]);
    }).then(function (r) {
      PH.forEach(function (p, i) { IMG[p.key] = r[0][i]; });
      DATA.occ = r[1][0];
      r[1][1].photos.forEach(function (p) { CREDITS[p.file.replace("photos/", "").replace(".jpg", "")] = p.credit_line; });
      sideFig(); workedFig(); heatFig(); emptyFig(); headFig(); occFig(); costFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. self-attention and cross-attention side by side ----------
  var SIDE_STEPS = [
    "Step 1. Queries. The words ask. A word becomes a query vector in both layers: Q = X W_Q. The rows of the tables are the queries, named by the word they write.",
    "Step 2. Keys. A query is compared with a key for all items it may read. On the left the keys come from the words. On the right they come from the photo: 576 patch tokens, one key apiece.",
    "Step 3. Weights. A softmax per row turns the scaled dot products of queries and keys into weights that add up to 1. These are the weights BLIP computed. The left table is a triangle because of the causal mask.",
    "Step 4. Values. The values come from the same place as the keys. A word receives the weighted sum of the values of its row: a mix of earlier words on the left, a mix of patches on the right."
  ];
  function sideFig() {
    var host = Kit.$("sx-box"), cv = Kit.$("sx-cv"), st = { set: "hammerhead-1-page", stop: 4, step: 2, row: 7, reveal: 1, playing: false };
    var chipsRow = Kit.html("div", { class: "bar-row" }, Kit.$("sx-bar"));
    var capChips = Kit.chips(chipsRow, ["Caption written for this page", "BLIP's own caption"], 0, function (i) { st.set = "hammerhead-1-" + (i ? "blip" : "page"); st.row = Math.min(st.row, setById(st.set).words.length - 1); buildWords(); draw(); });
    var layerRow = Kit.html("div", { class: "bar-row" }, Kit.$("sx-bar"));
    Kit.html("span", { class: "hintline" }, layerRow, "Decoder layer:");
    var STOPS = DATA.blip.sets[0].stops, lab = STOPS.map(function (s) { return s === "mean" ? "mean of 12" : s; });
    var layerChips = Kit.chips(layerRow, lab, st.stop, function (i) { st.stop = i; draw(); });
    var wordRow = Kit.html("div", { class: "bar-row" }, Kit.$("sx-words")), wordChips = [];
    var stepBar = Kit.$("sx-btns"), status = Kit.$("sx-status"), timer = 0;
    var back = Kit.button(stepBar, "Back", function () { stop(); go(st.step - 1); });
    var next = Kit.button(stepBar, "Next step", function () { stop(); go((st.step + 1) % 4); });
    var play = Kit.button(stepBar, "Play", function () { if (st.playing) { stop(); return; } st.playing = true; play.textContent = "Pause"; if (st.step >= 3) go(0); tick(); });
    Kit.button(stepBar, "All steps", function () { stop(); go(3); });
    function stop() { st.playing = false; play.textContent = "Play"; clearTimeout(timer); }
    function tick() { timer = setTimeout(function () { if (!st.playing) return; if (st.step >= 3) { stop(); return; } go(st.step + 1); tick(); }, 2400); }
    function go(s) {
      st.step = Math.max(0, Math.min(3, s)); status.textContent = SIDE_STEPS[st.step];
      if (st.step === 2) { st.reveal = 0; Kit.animate(900, function (p) { st.reveal = Kit.ease(p); draw(); }); } else { st.reveal = 1; draw(); }
    }
    function buildWords() {
      wordRow.innerHTML = ""; var S = setById(st.set);
      Kit.html("span", { class: "hintline" }, wordRow, "Row, the word being written:");
      wordChips = S.words.map(function (w, i) {
        var b = Kit.html("button", { type: "button", "aria-pressed": String(i === st.row), class: "wchip" }, wordRow, wordLabel(w));
        b.addEventListener("click", function () { st.row = i; setPressed(wordChips, i); draw(); }); return b;
      });
    }
    // geometry for a canvas width
    function geom(W) {
      var wide = W >= 700, gap = 18, pw = wide ? (W - gap) / 2 : W, LW = W < 420 ? 70 : 96, HH = 74, MY = 54 + HH, rowH = W < 420 ? 17 : 19, S = setById(st.set), n = S.words.length;
      var mh = n * rowH, ph = MY + mh + 66;
      return { wide: wide, pw: pw, gap: gap, LW: LW, HH: HH, MY: MY, rowH: rowH, n: n, mh: mh, ph: ph, H: wide ? ph : 2 * ph + 12, mw: pw - LW - 26 };
    }
    var G, SELF, CROSS, STRIP;
    function patchStrip() {
      if (STRIP) return STRIP;
      var c = document.createElement("canvas"); c.width = 24; c.height = 24; var x = c.getContext("2d", { willReadFrequently: true });
      x.drawImage(IMG["hammerhead-1"], 0, 0, 24, 24); STRIP = x.getImageData(0, 0, 24, 24).data; return STRIP;
    }
    function draw() {
      var P = Kit.palette(), W = Math.min(980, Kit.width(host.querySelector(".sx-wrap"))), S = setById(st.set), n = S.words.length, i, j, r;
      G = geom(W); var ctx = fitCanvas(cv, W, G.H), ink = P.ink, muted = P.muted, strip = patchStrip();
      ctx.clearRect(0, 0, W, G.H);
      ctx.font = '600 15px Spectral, Georgia, serif'; ctx.textBaseline = "alphabetic";
      var self = S.self[st.stop], cross = new Array(n), mxS = 0, mxC = 0;
      for (r = 0; r < n; r++) { cross[r] = tile(S, r, st.stop); for (i = 0; i < 576; i++) if (cross[r][i] > mxC) mxC = cross[r][i]; for (j = 0; j < self[r].length; j++) if (self[r][j] > mxS) mxS = self[r][j]; }
      var teal = hexRGB(P.teal), coral = hexRGB(P.coral), navy = hexRGB(P.navy), sand = hexRGB(P.sand), paper = hexRGB(P.paper);
      var fl = flags("hammerhead-1");
      for (var pi = 0; pi < 2; pi++) {
        var ox = G.wide ? pi * (G.pw + G.gap) : 0, oy = G.wide ? 0 : pi * (G.ph + 12), X = ox + G.LW, Y = oy + G.MY, cross_ = pi === 1;
        ctx.fillStyle = ink; ctx.font = '600 15px Spectral, Georgia, serif'; ctx.textAlign = "left";
        ctx.fillText(cross_ ? "Cross-attention" : "Self-attention", ox + 2, oy + 16);
        ctx.fillStyle = muted; ctx.font = '13px Spectral, Georgia, serif';
        ctx.fillText(cross_ ? "Q from the words, K and V from the photo" : "Q, K and V all from the words", ox + 2, oy + 34);
        // row labels (queries)
        var qHot = st.step === 0;
        for (r = 0; r < n; r++) {
          var y = Y + r * G.rowH;
          ctx.fillStyle = r === st.row ? rgba(coral, 0.22) : (qHot ? rgba(coral, 0.1) : "rgba(0,0,0,0)"); ctx.fillRect(ox, y, G.LW - 4, G.rowH - 1);
          ctx.strokeStyle = qHot || r === st.row ? P.coral : P.rule; ctx.lineWidth = qHot ? 1.5 : 1; ctx.strokeRect(ox + 0.5, y + 0.5, G.LW - 5, G.rowH - 2);
          ctx.fillStyle = r === st.row || qHot ? ink : muted; var fs = 12.5; ctx.textAlign = "right";
          ctx.font = (r === st.row ? "600 " : "") + fs + 'px Spectral, Georgia, serif';
          while (ctx.measureText(wordLabel(S.words[r])).width > G.LW - 16 && fs > 9) { fs -= 0.5; ctx.font = (r === st.row ? "600 " : "") + fs + 'px Spectral, Georgia, serif'; }
          ctx.fillText(wordLabel(S.words[r]), ox + G.LW - 9, y + G.rowH - 5);
        }
        ctx.textAlign = "left"; ctx.fillStyle = qHot ? P.coral : muted; ctx.font = '600 12px Spectral, Georgia, serif'; ctx.fillText("queries", ox + 2, Y - 6);
        // key header
        var kHot = st.step === 1;
        if (!cross_) {
          var cw = G.mw / n, rd = S.reads;
          for (j = 0; j < n; j++) {
            ctx.save(); ctx.translate(X + (j + 0.5) * cw, Y - 8); ctx.rotate(-Math.PI / 3); ctx.fillStyle = kHot ? ink : muted; ctx.font = (kHot ? "600 " : "") + '12px Spectral, Georgia, serif'; ctx.textAlign = "left";
            ctx.fillText(rd[j] === "start" ? "[start]" : rd[j], 0, 0); ctx.restore();
          }
          if (kHot) { ctx.strokeStyle = P.navy; ctx.lineWidth = 2; ctx.strokeRect(X - 1, Y - 12, G.mw + 2, 5); }
        } else {
          var sy = Y - 34, sw = G.mw / 576;
          for (j = 0; j < 576; j++) { ctx.fillStyle = "rgb(" + strip[j * 4] + "," + strip[j * 4 + 1] + "," + strip[j * 4 + 2] + ")"; ctx.fillRect(X + j * sw, sy, Math.ceil(sw) + 0.5, 16); }
          for (j = 0; j < 576; j++) if (fl.empty[j]) { ctx.fillStyle = P.coral; ctx.fillRect(X + j * sw, sy + 17, Math.max(1, sw), 3); }
          ctx.fillStyle = kHot ? P.navy : muted; ctx.font = (kHot ? "600 " : "") + '12px Spectral, Georgia, serif'; ctx.textAlign = "left";
          ctx.fillText("Patch tokens: 576 (coral ticks: empty)", X, sy - 5);
          if (kHot) { ctx.strokeStyle = P.navy; ctx.lineWidth = 2; ctx.strokeRect(X - 1, sy - 1, G.mw + 2, 18); }
        }
        ctx.textAlign = "left"; ctx.fillStyle = kHot ? P.navy : muted; ctx.font = '600 12px Spectral, Georgia, serif'; ctx.fillText("keys", X, Y - G.HH + 14);
        // matrix
        ctx.fillStyle = rgba(paper, 1); ctx.fillRect(X, Y, G.mw, G.mh);
        if (st.step >= 2) {
          var shown = Math.ceil(st.reveal * n);
          for (r = 0; r < shown; r++) {
            var yy = Y + r * G.rowH;
            if (!cross_) {
              var cw2 = G.mw / n;
              for (j = 0; j < n; j++) {
                if (j > r) { ctx.fillStyle = "rgba(128,128,128,0.10)"; ctx.fillRect(X + j * cw2, yy, cw2 - 1, G.rowH - 1); continue; }
                ctx.fillStyle = rgba(teal, Math.min(1, 0.04 + 0.96 * Math.sqrt(self[r][j] / mxS))); ctx.fillRect(X + j * cw2, yy, cw2 - 1, G.rowH - 1);
              }
            } else {
              var sw2 = G.mw / 576;
              for (j = 0; j < 576; j++) { var t = Math.sqrt(cross[r][j] / mxC); if (t < 0.03) continue; ctx.fillStyle = rgba(teal, Math.min(1, 0.05 + 0.95 * t)); ctx.fillRect(X + j * sw2, yy, Math.ceil(sw2) + 0.5, G.rowH - 1); }
            }
          }
        }
        if (st.step >= 2) {
          // mask note
          if (!cross_) { ctx.fillStyle = muted; ctx.font = '12px Spectral, Georgia, serif'; ctx.textAlign = "right"; ctx.fillText("grey: masked (the word cannot read ahead)", X + G.mw, Y + G.mh + 15); }
          else { ctx.fillStyle = muted; ctx.font = '12px Spectral, Georgia, serif'; ctx.textAlign = "right"; ctx.fillText("no mask: any word may read all 576 patches", X + G.mw, Y + G.mh + 15); }
        }
        ctx.strokeStyle = P.rule; ctx.lineWidth = 1; ctx.strokeRect(X + 0.5, Y + 0.5, G.mw, G.mh);
        // selected row outline
        ctx.strokeStyle = P.coral; ctx.lineWidth = 2; ctx.strokeRect(X, Y + st.row * G.rowH, G.mw, G.rowH - 1);
        // output column
        if (st.step >= 3) {
          var vc = cross_ ? navy : sand;
          for (r = 0; r < n; r++) { ctx.fillStyle = rgba(vc, 0.85); ctx.fillRect(X + G.mw + 6, Y + r * G.rowH, 14, G.rowH - 1); }
          ctx.fillStyle = ink; ctx.font = '600 12px Spectral, Georgia, serif'; ctx.textAlign = "right"; var outT = "new word vectors, the weighted sum of values from " + (cross_ ? "the photo" : "the words"); if (ctx.measureText(outT).width > X + G.mw + 20 - ox) outT = "new word vectors (weighted values)"; ctx.fillText(outT, X + G.mw + 20, Y + G.mh + 52);
        }
        // sizes
        ctx.fillStyle = muted; ctx.font = '12.5px Spectral, Georgia, serif'; ctx.textAlign = "left";
        var szLong = cross_ ? n + " queries × 577 keys = " + comma(n * 577) + " weights (class token not drawn)" : n + " queries × " + n + " keys, " + (n * (n + 1) / 2) + " allowed by the mask", szShort = cross_ ? n + " × 577 = " + comma(n * 577) + " weights" : n + " × " + n + ", " + (n * (n + 1) / 2) + " allowed";
        ctx.fillText(ctx.measureText(szLong).width <= G.pw - G.LW - 4 ? szLong : szShort, X, Y + G.mh + 15 + (st.step >= 2 ? 18 : 0));
      }
      readouts();
    }
    function readouts() {
      var S = setById(st.set), r = st.row, n = S.words.length, selfRow = S.self[st.stop][r], t = tile(S, r, st.stop), fl = flags("hammerhead-1"), sh = shares(t, fl.empty, fl.animal, false), best = 0, bi = 0, i, bj = 0;
      for (i = 0; i < selfRow.length; i++) if (selfRow[i] > selfRow[bj]) bj = i;
      for (i = 0; i < 576; i++) if (t[i] > best) { best = t[i]; bi = i; }
      Kit.$("sx-eq").innerHTML =
        ebox("c-coral", wordLabel(S.words[r]), "the word written, query row " + (r + 1)) + eop("&rarr;") +
        ebox("c-teal", (r + 1) + " keys", "self: the row reads " + (r + 1) + " of " + n + " words") + eop("|") +
        ebox("c-teal", pc(selfRow[bj], 0), "self: largest weight, on " + (S.reads[bj] === "start" ? "[start]" : S.reads[bj])) + eop("|") +
        ebox("c-navy", "577 keys", "cross: 576 patches and the class token, no mask") + eop("|") +
        ebox("c-teal", pc(best, 1), "cross: largest weight, on patch " + Math.floor(bi / 24) + "," + (bi % 24)) + eop("|") +
        ebox("c-sand", pc(sh.empty, 0), "cross: weight on the 27 empty patches");
      Kit.$("sx-dims").innerHTML =
        ebox("c-coral", n + " × 64", "Q, the words") + eop("&times;") + ebox("c-navy", "64 × 577", "K transposed, the photo") + eop("=") + ebox("c-teal", n + " × 577", "cross scores") + eop("|") +
        ebox("c-coral", n + " × 64", "Q, the words") + eop("&times;") + ebox("c-navy", "64 × " + n, "K transposed, the words") + eop("=") + ebox("c-teal", n + " × " + n, "self scores");
    }
    cv.addEventListener("click", function (ev) {
      var rc = cv.getBoundingClientRect(), x = ev.clientX - rc.left, y = ev.clientY - rc.top, pi = G.wide ? (x > G.pw + G.gap / 2 ? 1 : 0) : (y > G.ph + 6 ? 1 : 0), oy = G.wide ? 0 : pi * (G.ph + 12);
      var r = Math.floor((y - oy - G.MY) / G.rowH); if (r >= 0 && r < G.n) { st.row = r; setPressed(wordChips, r); draw(); }
    });
    buildWords(); go(2);
    onResize(draw);
  }

  // ---------- 2. four patches by hand ----------
  var WK_PATCHES = [
    { name: "water", cell: [3, 4], k: [-1.2, 0.3], v: [0.2, -0.9] },
    { name: "fin", cell: [5, 12], k: [0.2, 1.4], v: [-0.7, 0.8] },
    { name: "head", cell: [10, 7], k: [1.3, 0.4], v: [0.9, 0.7] },
    { name: "body", cell: [12, 15], k: [0.5, -1.1], v: [0.6, -0.5] }
  ];
  var WK_WORDS = [{ w: "water", q: [-1.5, 0.3] }, { w: "fin", q: [0.3, 1.5] }, { w: "hammerhead", q: [1.5, 0.6] }, { w: "body", q: [0.9, -1.2] }];
  function workedFig() {
    var st = { q: [0.3, 1.5], order: [0, 1, 2, 3], scale: true, prevOut: null };
    var ctl = Kit.$("wk-controls"), plane = Kit.$("wk-plane"), tilesBox = Kit.$("wk-tiles");
    Kit.html("span", { class: "hintline" }, Kit.$("wk-bar"), "A word's query:");
    var wchips = Kit.chips(Kit.$("wk-bar"), WK_WORDS.map(function (x) { return x.w; }), 1, function (i) { st.q = WK_WORDS[i].q.slice(); s1.set(st.q[0]); s2.set(st.q[1]); refresh(); });
    var s1 = Kit.slider(ctl, { label: "Query, first number", min: -2, max: 2, step: 0.1, value: st.q[0] }, function (v) { st.q[0] = v; wchips.forEach(function (b) { b.setAttribute("aria-pressed", "false"); }); refresh(); });
    var s2 = Kit.slider(ctl, { label: "Query, second number", min: -2, max: 2, step: 0.1, value: st.q[1] }, function (v) { st.q[1] = v; wchips.forEach(function (b) { b.setAttribute("aria-pressed", "false"); }); refresh(); });
    var chk = Kit.html("label", { class: "check" }, Kit.$("wk-bar")), cb = Kit.html("input", { type: "checkbox", checked: "checked" }, chk); chk.appendChild(document.createTextNode(" divide by √2"));
    cb.addEventListener("change", function () { st.scale = cb.checked; refresh(); });
    Kit.button(Kit.$("wk-bar"), "Shuffle the patches", function () {
      var o = st.order.slice(), r = Kit.rng(4127 + o.join("") * 1 + Math.floor(Math.random() * 1000)), i; for (i = o.length - 1; i > 0; i--) { var j = Math.floor(r() * (i + 1)), t = o[i]; o[i] = o[j]; o[j] = t; }
      st.order = o; refresh(true);
    });
    Kit.button(Kit.$("wk-bar"), "Reset order", function () { st.order = [0, 1, 2, 3]; refresh(true); });
    // canvases for the four patches
    var tcv = WK_PATCHES.map(function () { var c = Kit.html("canvas", { width: 56, height: 56 }); return c; });
    function drawTiles() {
      var im = IMG["hammerhead-1"], cw = im.naturalWidth / 24, ch = im.naturalHeight / 24;
      tilesBox.innerHTML = "";
      st.order.forEach(function (pi, slot) {
        var p = WK_PATCHES[pi], col = Kit.html("div", { class: "wk-col" }, tilesBox), c = Kit.html("canvas", {}, col), ctx = fitCanvas(c, 56, 56);
        ctx.imageSmoothingEnabled = true; ctx.drawImage(im, p.cell[1] * cw, p.cell[0] * ch, cw, ch, 0, 0, 56, 56);
        Kit.html("div", { class: "wk-name" }, col, p.name);
        col.id = "wk-c" + slot; Kit.html("div", { class: "wk-score" }, col); Kit.html("div", { class: "wk-bar" }, col).appendChild(Kit.html("i", {}, null));
        Kit.html("div", { class: "wk-w" }, col);
      });
    }
    function run() { var P = st.order.map(function (i) { return WK_PATCHES[i]; }); return attendOne(st.q, P.map(function (p) { return p.k; }), P.map(function (p) { return p.v; }), st.scale); }
    function drawPlane() {
      var P = Kit.palette(); plane.innerHTML = ""; var W = 260, H = 260, cx = 130, cy = 130, u = 52;
      plane.setAttribute("viewBox", "0 0 " + W + " " + H);
      var X = function (v) { return cx + v * u; }, Y = function (v) { return cy - v * u; };
      Kit.el("line", { x1: 8, y1: cy, x2: W - 8, y2: cy, stroke: P.rule }, plane); Kit.el("line", { x1: cx, y1: 8, x2: cx, y2: H - 8, stroke: P.rule }, plane);
      WK_PATCHES.forEach(function (p) {
        Kit.el("line", { x1: cx, y1: cy, x2: X(p.k[0]), y2: Y(p.k[1]), stroke: P.navy, "stroke-width": 2 }, plane);
        Kit.el("circle", { cx: X(p.k[0]), cy: Y(p.k[1]), r: 5, fill: P.navy }, plane);
      });
      Kit.el("line", { x1: cx, y1: cy, x2: X(st.q[0]), y2: Y(st.q[1]), stroke: P.coral, "stroke-width": 3 }, plane);
      var h = Kit.el("circle", { cx: X(st.q[0]), cy: Y(st.q[1]), r: 9, fill: P.coral, stroke: P.bg, "stroke-width": 2, style: "cursor:grab", tabindex: 0, role: "slider", "aria-label": "Tip of the query. Drag it, or use the sliders." }, plane);
      var drag = false;
      function set(ev) { var r = plane.getBoundingClientRect(), x = (ev.clientX - r.left) / r.width * W, y = (ev.clientY - r.top) / r.height * H, a = Math.max(-2, Math.min(2, Math.round((x - cx) / u * 10) / 10)), b = Math.max(-2, Math.min(2, Math.round((cy - y) / u * 10) / 10)); st.q = [a, b]; s1.set(a); s2.set(b); wchips.forEach(function (bt) { bt.setAttribute("aria-pressed", "false"); }); refresh(); }
      h.addEventListener("pointerdown", function (ev) { drag = true; h.setPointerCapture(ev.pointerId); });
      h.addEventListener("pointermove", function (ev) { if (drag) set(ev); });
      h.addEventListener("pointerup", function () { drag = false; });
      WK_PATCHES.forEach(function (p) {
        var t = Kit.el("text", { x: X(p.k[0]) + (p.k[0] < 0 ? -8 : 8), y: Y(p.k[1]) + (p.k[1] < 0 ? 15 : -8), fill: P.ink2, stroke: P.bg, "stroke-width": 3.5, "stroke-linejoin": "round", "paint-order": "stroke", "font-size": 13, "text-anchor": p.k[0] < 0 ? "end" : "start", "font-family": "Spectral, Georgia, serif", style: "pointer-events:none" }, plane); t.textContent = "key: " + p.name;
      });
      var tq = Kit.el("text", { x: 10, y: H - 8, fill: P.coral, "font-size": 13, "font-weight": 600, "font-family": "Spectral, Georgia, serif" }, plane); tq.textContent = "coral: the query (" + nf(st.q[0], 1) + ", " + nf(st.q[1], 1) + ")";
    }
    function refresh(reorder) {
      if (reorder !== undefined) drawTiles();
      var P = run(), order = st.order, i, besti = 0;
      for (i = 0; i < 4; i++) if (P.weights[i] > P.weights[besti]) besti = i;
      order.forEach(function (pi, slot) {
        var col = Kit.$("wk-c" + slot);
        col.querySelector(".wk-score").innerHTML = "<b>" + nf(P.dots[slot], 2) + "</b><span>q·k</span>";
        col.querySelector(".wk-bar i").style.width = (100 * P.weights[slot]).toFixed(1) + "%";
        col.querySelector(".wk-w").innerHTML = "<b>" + pc(P.weights[slot], 0) + "</b>";
        col.classList.toggle("top", slot === besti);
      });
      var dim = st.scale ? "&divide;1.41" : "";
      var scoreBoxes = order.map(function (pi, slot) { return ebox("c-navy", nf(P.scores[slot], 2), "q·k " + dim + ", " + WK_PATCHES[pi].name); }).join(eop(","));
      var e = P.weights.map(function (a, slot) { return Math.exp(P.scores[slot]); }), z = e.reduce(function (a, b) { return a + b; }, 0);
      var wBoxes = order.map(function (pi, slot) { return ebox("c-teal", pc(P.weights[slot], 0), "e^" + nf(P.scores[slot], 2) + " = " + nf(e[slot], 2) + " of " + nf(z, 2)); }).join(eop("+"));
      var oBoxes = order.map(function (pi, slot) { var p = WK_PATCHES[pi]; return ebox("c-sand", "(" + nf(P.weights[slot] * p.v[0], 2) + ", " + nf(P.weights[slot] * p.v[1], 2) + ")", nf(P.weights[slot], 2) + " × (" + nf(p.v[0], 1) + ", " + nf(p.v[1], 1) + ")"); }).join(eop("+"));
      Kit.$("wk-eq1").innerHTML = ebox("c-coral", "(" + nf(st.q[0], 1) + ", " + nf(st.q[1], 1) + ")", "query of the word") + eop("&rarr;") + scoreBoxes;
      Kit.$("wk-eq2").innerHTML = eop("softmax &rarr;") + wBoxes + eop("= 100%");
      Kit.$("wk-eq3").innerHTML = eop("output =") + oBoxes + eop("=") + ebox("c-teal", "(" + nf(P.out[0], 2) + ", " + nf(P.out[1], 2) + ")", "new vector of the word");
      var same = st.prevOut && Math.abs(st.prevOut[0] - P.out[0]) < 1e-9 && Math.abs(st.prevOut[1] - P.out[1]) < 1e-9;
      Kit.$("wk-status").textContent = "Largest weight: " + pc(P.weights[besti], 0) + " on the " + WK_PATCHES[order[besti]].name + " patch. " + (reorder !== undefined && st.prevOut ? (same ? "Order changed, output unchanged." : "") : "");
      if (reorder === undefined || !st.prevOut) st.prevOut = P.out.slice(); else if (reorder !== undefined) st.prevOut = P.out.slice();
      drawPlane();
    }
    drawTiles(); refresh(); onResize(function () { drawTiles(); refresh(); });
  }

  // ---------- 3. click a word, see the photo ----------
  function heatFig() {
    var st = { photo: 0, who: "page", word: 1, stop: 4, hide: false, outline: true };
    var cv = Kit.$("hm-cv"), photoRow = Kit.html("div", { class: "bar-row" }, Kit.$("hm-bar")), capRow = Kit.html("div", { class: "bar-row" }, Kit.$("hm-bar")), layerRow = Kit.html("div", { class: "bar-row" }, Kit.$("hm-bar")), wordRow = Kit.html("div", { class: "bar-row" }, Kit.$("hm-words")), wchips = [];
    var pchips = Kit.chips(photoRow, PH.map(function (p) { return p.name; }), 0, function (i) { st.photo = i; st.word = nounIndex(curSet()); buildWords(); draw(); });
    var cchips = Kit.chips(capRow, ["Caption written for this page", "BLIP's own caption"], 0, function (i) { st.who = i ? "blip" : "page"; st.word = nounIndex(curSet()); buildWords(); draw(); });
    Kit.html("span", { class: "hintline" }, layerRow, "Decoder layer:");
    var STOPS = DATA.blip.sets[0].stops, lchips = Kit.chips(layerRow, STOPS.map(function (s) { return s === "mean" ? "mean of 12" : s; }), st.stop, function (i) { st.stop = i; draw(); });
    var opt = Kit.html("div", { class: "bar-row" }, Kit.$("hm-bar"));
    var l1 = Kit.html("label", { class: "check" }, opt), c1 = Kit.html("input", { type: "checkbox" }, l1); l1.appendChild(document.createTextNode(" set the empty patches aside"));
    var l2 = Kit.html("label", { class: "check" }, opt), c2 = Kit.html("input", { type: "checkbox", checked: "checked" }, l2); l2.appendChild(document.createTextNode(" outline the animal (marked by hand)"));
    c1.addEventListener("change", function () { st.hide = c1.checked; draw(); }); c2.addEventListener("change", function () { st.outline = c2.checked; draw(); });
    function curSet() { return setFor(PH[st.photo].key, st.who); }
    function buildWords() {
      wordRow.innerHTML = ""; Kit.html("span", { class: "hintline" }, wordRow, "Click the word being written:");
      wchips = curSet().words.map(function (w, i) { var b = Kit.html("button", { type: "button", "aria-pressed": String(i === st.word), class: "wchip" }, wordRow, wordLabel(w)); b.addEventListener("click", function () { st.word = i; setPressed(wchips, i); draw(); }); return b; });
    }
    function draw() {
      var P = Kit.palette(), S = curSet(), key = PH[st.photo].key, ph = drawPhoto(cv, key, 520), fl = flags(key), t = tile(S, st.word, st.stop), i;
      var mx = heatOver(ph.ctx, ph.w, ph.h, t, fl.empty, st.hide, P);
      if (st.outline) outline(ph.ctx, ph.w, ph.h, DATA.blip.photos[key].poly, P);
      var sh = shares(t, fl.empty, fl.animal, st.hide), nA = DATA.blip.photos[key].animal.length, nE = DATA.blip.photos[key].n_empty;
      var tt = st.hide ? Array.prototype.filter.call(t, function (v, k) { return !fl.empty[k]; }) : t, ne = neff(Array.prototype.slice.call(tt));
      var best = 0, bi = 0; for (i = 0; i < 576; i++) if (!(st.hide && fl.empty[i]) && t[i] > best) { best = t[i]; bi = i; }
      var tot = 0; for (i = 0; i < 576; i++) if (!(st.hide && fl.empty[i])) tot += t[i];
      var lp = S.logp[st.word];
      Kit.$("hm-eq").innerHTML =
        ebox("c-coral", wordLabel(S.words[st.word]), "word written; BLIP gave it " + (Math.exp(lp) < 0.001 ? "under 0.1%" : pc(Math.exp(lp), Math.exp(lp) < 0.1 ? 1 : 0))) + eop("|") +
        ebox("c-sand", pc(sh.empty, 0), "weight on the " + nE + " empty patches") + eop("|") +
        ebox("c-teal", pc(sh.animal, 0), "weight inside the animal (" + pc(nA / 576, 0) + " of the patches)") + eop("|") +
        ebox("c-navy", pc(best / tot, 1), "largest patch, row " + Math.floor(bi / 24) + ", column " + (bi % 24)) + eop("|") +
        ebox("c-navy", nf(ne, 0), "effective number of patches (of " + (st.hide ? 576 - nE : 576) + ")");
      Kit.$("hm-credit").textContent = CREDITS[key] || "";
      cv.setAttribute("aria-label", "Photo with a heat map of the weights of the word " + wordLabel(S.words[st.word]) + " over the 24 by 24 patches, " + pc(sh.empty, 0) + " on empty patches, " + pc(sh.animal, 0) + " inside the animal");
      var bar = Kit.$("hm-share"); bar.innerHTML = "";
      [["empty", sh.empty, "c-navy"], ["animal", sh.animal, "c-teal"], ["other", sh.other, "c-sand"]].forEach(function (x) { if (x[1] > 0.001) { var s = Kit.html("span", { class: "seg " + x[2], style: "width:" + (100 * x[1]).toFixed(1) + "%" }, bar); s.textContent = (x[1] > 0.07 ? x[0] + " " + pc(x[1], 0) : ""); } });
    }
    buildWords(); draw(); onResize(draw);
  }

  // ---------- 4. the empty patches ----------
  function emptyFig() {
    var st = { photo: 0, layer: 11 };
    var pchips = Kit.chips(Kit.$("em-bar"), PH.map(function (p) { return p.name; }), 0, function (i) { st.photo = i; draw(); });
    var sl = Kit.slider(Kit.$("em-controls"), { label: "Decoder layer", min: 0, max: 11, step: 1, value: 11 }, function (v) { st.layer = v; draw(); });
    var cv = Kit.$("em-cv");
    function draw() {
      var key = PH[st.photo].key, S = setFor(key, "page"), P = Kit.palette(), ph = drawPhoto(cv, key, 360), fl = flags(key), i, L = st.layer, C = S.contrib;
      var cw = ph.w / 24, ch = ph.h / 24;
      for (i = 0; i < 576; i++) if (fl.empty[i]) { ph.ctx.fillStyle = rgba(hexRGB(P.coral), 0.8); ph.ctx.fillRect((i % 24) * cw, Math.floor(i / 24) * ch, cw + 0.5, ch + 0.5); }
      var ph2 = DATA.blip.photos[key];
      var x = C.map(function (c, l) { return l; });
      Kit.plot("em-plot", function (Pl) {
        return [
          { x: x, y: C.map(function (c) { return 100 * c[1]; }), type: "scatter", mode: "lines+markers", name: "weight on the empty patches", line: { color: Pl.navy, width: 2.5 }, marker: { size: 6 }, hovertemplate: "layer %{x}: %{y:.0f}% of the weight<extra></extra>" },
          { x: x, y: C.map(function (c) { return 100 * c[0]; }), type: "scatter", mode: "lines+markers", name: "share of what is added", line: { color: Pl.coral, width: 2.5, dash: "dash" }, marker: { size: 6, symbol: "square" }, hovertemplate: "layer %{x}: %{y:.1f}% of the sum of weight times value length<extra></extra>" },
          { x: [L, L], y: [0, 100], type: "scatter", mode: "lines", line: { color: Pl.muted, width: 1.5, dash: "dot" }, hoverinfo: "skip" }
        ];
      }, function (Pl) {
        return { xaxis: { title: { text: "decoder layer", standoff: 10 }, dtick: 1, range: [-0.5, 11.5] }, yaxis: { title: { text: "percent of the total" }, range: [0, 100] }, margin: { l: 52, r: 12, t: 10, b: 52 },
          annotations: [{ x: 6, y: C[6][1] * 100, text: "weight", showarrow: false, yshift: 14, font: { color: Pl.navy, size: 13 } }, { x: 6, y: C[6][0] * 100, text: "what is added", showarrow: false, yshift: 14, font: { color: Pl.coral, size: 13 } }] };
      });
      Kit.$("em-eq").innerHTML =
        ebox("c-sand", ph2.n_empty + " of 576", "empty patches (" + pc(ph2.n_empty / 576, 1) + ")") + eop("|") +
        ebox("c-navy", nf(ph2.token_len_empty, 1) + " against " + nf(ph2.token_len_other, 1), "length of an image token: empty, other") + eop("|") +
        ebox("c-navy", nf(C[L][2], 2) + " against " + nf(C[L][3], 1), "length of a value, layer " + L + ": empty, other") + eop("|") +
        ebox("c-teal", pc(C[L][1], 0), "weight on the empty patches") + eop("&rarr;") + ebox("c-coral", pc(C[L][0], 1), "share of what is added");
    }
    onResize(draw); draw();
  }

  // ---------- 5. the heads of the last layer ----------
  function headFig() {
    var st = { photo: 0, word: 1, hide: true, sort: false };
    var pchips = Kit.chips(Kit.$("hd-bar"), PH.map(function (p) { return p.name; }), 0, function (i) { st.photo = i; st.word = nounIndex(curSet()); buildWords(); draw(); });
    var wordRow = Kit.html("div", { class: "bar-row" }, Kit.$("hd-words")), wchips = [];
    var opt = Kit.html("div", { class: "bar-row" }, Kit.$("hd-bar"));
    var l1 = Kit.html("label", { class: "check" }, opt), c1 = Kit.html("input", { type: "checkbox", checked: "checked" }, l1); l1.appendChild(document.createTextNode(" set the empty patches aside"));
    var l2 = Kit.html("label", { class: "check" }, opt), c2 = Kit.html("input", { type: "checkbox" }, l2); l2.appendChild(document.createTextNode(" sort the heads by weight inside the animal"));
    c1.addEventListener("change", function () { st.hide = c1.checked; draw(); }); c2.addEventListener("change", function () { st.sort = c2.checked; draw(); });
    function curSet() { return setFor(PH[st.photo].key, "page"); }
    function buildWords() {
      wordRow.innerHTML = ""; Kit.html("span", { class: "hintline" }, wordRow, "Word written:");
      wchips = curSet().words.map(function (w, i) { var b = Kit.html("button", { type: "button", "aria-pressed": String(i === st.word), class: "wchip" }, wordRow, wordLabel(w)); b.addEventListener("click", function () { st.word = i; setPressed(wchips, i); draw(); }); return b; });
    }
    function draw() {
      var P = Kit.palette(), S = curSet(), key = PH[st.photo].key, fl = flags(key), grid = Kit.$("hd-grid"), im = IMG[key], h, list = [];
      var ar = im.naturalHeight / im.naturalWidth, w = Math.max(70, Math.min(150, Math.floor((Kit.width(grid) - 30) / (window.innerWidth < 600 ? 3 : 6))));
      for (h = 0; h < 12; h++) { var t = headTile(S, st.word, h), hr = S.hread[st.word][0][h]; list.push({ h: h, t: t, inside: hr[1], empty: hr[0] }); }
      if (st.sort) list.sort(function (a, b) { return b.inside - a.inside; });
      grid.innerHTML = "";
      list.forEach(function (x) {
        var box = Kit.html("div", { class: "hd-cell" }, grid), c = Kit.html("canvas", {}, box), ctx = fitCanvas(c, w, Math.round(w * ar));
        ctx.drawImage(im, 0, 0, w, Math.round(w * ar)); heatOver(ctx, w, Math.round(w * ar), x.t, fl.empty, st.hide, P);
        Kit.html("div", { class: "hd-lab" }, box).innerHTML = "<b>head " + x.h + "</b><br>" + pc(x.inside, 0) + " inside, " + pc(x.empty, 0) + " empty";
      });
      var ins = list.map(function (x) { return x.inside; }), mean = ins.reduce(function (a, b) { return a + b; }, 0) / 12, nA = DATA.blip.photos[key].animal.length / 576, hi = list.reduce(function (a, b) { return b.inside > a.inside ? b : a; });
      Kit.$("hd-eq").innerHTML = ebox("c-coral", wordLabel(S.words[st.word]), "word written, layer 11") + eop("|") + ebox("c-sand", pc(nA, 0), "area of the animal") + eop("|") +
        ebox("c-teal", pc(mean, 0), "mean of the 12 heads, inside the animal") + eop("|") + ebox("c-teal", pc(hi.inside, 0), "best head (" + hi.h + "), inside the animal") + eop("|") + ebox("c-navy", pc(Math.min.apply(null, ins), 0), "weakest head, inside");
      Kit.$("hd-credit").textContent = CREDITS[key] || "";
    }
    buildWords(); draw(); onResize(draw);
  }

  // ---------- 6. does attention show what matters? ----------
  function occFig() {
    var st = { photo: 2, word: 9, rule: 1, k: 3 }, KS = DATA.occ.ks;
    var pchips = Kit.chips(Kit.$("oc-bar"), PH.map(function (p) { return p.name; }), 2, function (i) { st.photo = i; st.word = nounIndex(setById(O().set)); buildWords(); draw(); });
    var wordRow = Kit.html("div", { class: "bar-row" }, Kit.$("oc-words")), wchips = [];
    var ctl = Kit.$("oc-controls");
    Kit.select(ctl, "Hide the patches chosen by", [{ value: "0", label: "attention, as drawn" }, { value: "1", label: "attention, empty patches set aside" }, { value: "2", label: "the single-patch test" }, { value: "3", label: "chance (random patches)" }], "1", function (v) { st.rule = +v; draw(); });
    Kit.slider(ctl, { label: "Patches hidden, k", min: 0, max: KS.length - 1, step: 1, value: st.k, show: function (i) { return KS[i] + " (" + pc(KS[i] / 576, 1) + ")"; } }, function (i) { st.k = i; draw(); });
    function O() { return DATA.occ.photos[PH[st.photo].key]; }
    function buildWords() {
      wordRow.innerHTML = ""; Kit.html("span", { class: "hintline" }, wordRow, "Word written:");
      wchips = O().words.map(function (w, i) { var b = Kit.html("button", { type: "button", "aria-pressed": String(i === st.word), class: "wchip" }, wordRow, wordLabel(w)); b.addEventListener("click", function () { st.word = i; setPressed(wchips, i); draw(); }); return b; });
    }
    function oTile(key, r, mx) { var T = OTILE[key], out = new Float64Array(576), i, j; for (i = 0; i < 24; i++) for (j = 0; j < 24; j++) out[i * 24 + j] = dequant(T.a[(r * 24 + i) * T.w + j], mx, 255); return out; }
    function rank(vals, skip) { var idx = [], i; for (i = 0; i < 576; i++) if (!(skip && skip[i])) idx.push(i); idx.sort(function (a, b) { return vals[b] - vals[a]; }); return idx; }
    function draw() {
      var P = Kit.palette(), key = PH[st.photo].key, o = O(), S = setById(o.set), fl = flags(key), r = st.word, k = KS[st.k];
      var att = tile(S, r, S.stops.length - 1), one = oTile(key, r, o.omx[r]);
      var w = Math.min(300, Math.floor((Kit.width(Kit.$("oc-maps")) - 24) / (window.innerWidth < 700 ? 1 : 3))), ids = ["oc-a", "oc-o", "oc-p"], ph = [];
      ids.forEach(function (id) { ph.push(drawPhoto(Kit.$(id), key, w)); });
      heatOver(ph[0].ctx, ph[0].w, ph[0].h, att, fl.empty, true, P); heatOver(ph[1].ctx, ph[1].w, ph[1].h, one, fl.empty, false, P, { color: P.teal });
      var chosen;
      if (st.rule === 0) chosen = rank(att, null).slice(0, k); else if (st.rule === 1) chosen = rank(att, fl.empty).slice(0, k); else if (st.rule === 2) chosen = rank(one, null).slice(0, k);
      else { var rr = Kit.rng(4127), pool = []; for (var i = 0; i < 576; i++) pool.push(i); for (i = 575; i > 0; i--) { var j = Math.floor(rr() * (i + 1)), tmp = pool[i]; pool[i] = pool[j]; pool[j] = tmp; } chosen = pool.slice(0, k); }
      var c = ph[2], cw = c.w / 24, ch = c.h / 24; c.ctx.fillStyle = rgba(hexRGB(P.bg), 0.9);
      chosen.forEach(function (i) { c.ctx.fillRect((i % 24) * cw, Math.floor(i / 24) * ch, cw + 0.5, ch + 0.5); c.ctx.strokeStyle = P.muted; c.ctx.lineWidth = 0.5; c.ctx.strokeRect((i % 24) * cw, Math.floor(i / 24) * ch, cw, ch); });
      var cr = o.curves[r], names = ["attention", "attention_set_aside", "one_patch", "random"], lab = ["attention, as drawn", "attention, empty patches set aside", "single-patch test", "random patches"], colors = [P.navy, P.sand, P.teal, P.muted];
      var allv = [].concat.apply([], names.map(function (n) { return cr[n]; })), lo = Math.min(0, Math.min.apply(null, allv)), hi = Math.max(0.5, Math.max.apply(null, allv));
      Kit.plot("oc-plot", function (Pl) {
        var cols = [Pl.navy, Pl.sand, Pl.teal, Pl.muted], tr = names.map(function (n, i) {
          return { x: KS, y: cr[n], type: "scatter", mode: "lines+markers", name: lab[i], line: { color: cols[i], width: i === st.rule ? 3.5 : 1.8, dash: i === 3 ? "dot" : "solid" }, marker: { size: i === st.rule ? 8 : 5 }, hovertemplate: lab[i] + ", %{x} patches hidden: %{y:.2f} nats<extra></extra>" };
        });
        tr.push({ x: [k, k], y: [lo, hi], type: "scatter", mode: "lines", line: { color: Pl.coral, width: 1.5, dash: "dot" }, hoverinfo: "skip", showlegend: false });
        return tr;
      }, function (Pl) { var narrow = Kit.width(Kit.$("oc-plot")) < 520; Kit.$("oc-plot").style.height = narrow ? "340px" : ""; return { xaxis: { title: { text: "patches hidden, k" }, type: "log", tickvals: KS, ticktext: KS.map(String) }, yaxis: { title: { text: "fall (nats)" }, zeroline: true }, margin: { l: 56, r: 12, t: 10, b: narrow ? 150 : 54 }, showlegend: true, legend: narrow ? { orientation: "v", yref: "container", y: 0, yanchor: "bottom", x: 0, font: { size: 12 } } : { orientation: "h", y: -0.5, font: { size: 12 } } }; });
      var nowFall = cr[names[st.rule]][st.k];
      Kit.$("oc-eq").innerHTML =
        ebox("c-coral", wordLabel(o.words[r]), "word; log probability " + nf(o.base[r], 2)) + eop("|") +
        ebox("c-teal", nf(nowFall, 2) + " nats", "fall when " + k + " patches are hidden by this rule") + eop("|") +
        ebox("c-navy", nf(o.rho[r], 2), "rank correlation of attention and the single-patch map") + eop("|") +
        ebox("c-sand", pc(o.overlap32[r], 0), "of the best 32 by attention are among the best 32 by test") + eop("|") +
        ebox("c-teal", nf(o.animal_fall[r], 2) + " against " + nf(o.background_fall[r], 2), "hiding all " + o.n_animal + " animal patches, against as many background patches");
      Kit.$("oc-credit").textContent = CREDITS[key] || "";
    }
    buildWords(); draw(); onResize(draw);
  }

  // ---------- 7. what the image costs ----------
  function costFig() {
    var st = { side: 384, p: 16, nTxt: 12, d: 768 };
    var ctl = Kit.$("co-controls"), PS = [8, 14, 16, 32, 64];
    Kit.slider(ctl, { label: "Photo side S, pixels", min: 224, max: 1024, step: 32, value: 384, show: function (v) { return v + " px"; } }, function (v) { st.side = v; draw(); });
    Kit.slider(ctl, { label: "Patch side P", min: 0, max: PS.length - 1, step: 1, value: 2, show: function (i) { return PS[i] + " px"; } }, function (i) { st.p = PS[i]; draw(); });
    Kit.slider(ctl, { label: "Words written, n", min: 1, max: 200, step: 1, value: 12, show: function (v) { return v + " words"; } }, function (v) { st.nTxt = v; draw(); });
    function draw() {
      var g = Math.floor(st.side / st.p), nImg = g * g, c = costs(nImg, st.nTxt, st.d), box = Kit.$("co-bars");
      var rows = [["pairs per head and layer: image tokens in the sequence", c.seqPairs, "c-navy"], ["pairs per head and layer: cross-attention", c.crossPairs, "c-teal"], ["multiply-adds per layer: image tokens in the sequence", c.seqMA, "c-navy"], ["multiply-adds per layer: cross-attention", c.crossMA, "c-teal"]];
      var mx1 = Math.max(rows[0][1], rows[1][1]), mx2 = Math.max(rows[2][1], rows[3][1]); box.innerHTML = "";
      rows.forEach(function (r, i) {
        var m = i < 2 ? mx1 : mx2, row = Kit.html("div", { class: "co-row" }, box);
        Kit.html("span", { class: "co-name" }, row, r[0]); var tr = Kit.html("span", { class: "co-track" }, row), b = Kit.html("i", { class: r[2] }, tr); b.style.width = Math.max(0.5, 100 * r[1] / m) + "%"; Kit.html("b", {}, row, comma(r[1]));
      });
      Kit.$("co-eq").innerHTML =
        ebox("c-sand", g + " × " + g, "patches per side, (S / P)") + eop("=") + ebox("c-coral", comma(nImg), "image tokens N") + eop("|") +
        ebox("c-navy", nf(c.seqPairs / c.crossPairs, 1) + "×", "pairs: sequence over cross") + eop("|") +
        ebox("c-teal", nf(c.seqMA / c.crossMA, 1) + "×", "multiply-adds: sequence over cross") + eop("|") + ebox("c-sand", comma(2 * (nImg + 1) * st.d), "numbers of keys and values of the photo per layer (patches and class token)");
    }
    draw();
  }
})();
