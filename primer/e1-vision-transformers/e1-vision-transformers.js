// E1 Vision transformers and image tokens.
//   cutFig      SIGNATURE: a patch-size slider cuts a shark photo into a grid; Step and Play lift the patches into a row of tokens with position numbers
//   embedFig    one patch times a matrix: eight rows of the matrix of a trained ViT-B/16, computed live from the pixels you click (data/vit.json)
//   posFig      position embeddings of the same ViT (cosine map) and a patch-moving test for the ViT and for ResNet-18 (data/shuffle.json)
//   clsFig      the class token: where it reads in each of the 12 blocks (mean over the 12 heads)
//   attendFig   click a patch, see where six chosen heads of the trained ViT send its attention (data/attn/*.png, 16 levels)
//   reachFig    what a CNN window and a ViT block can reach, measured attention distance of all 144 heads, parameter counts
//   vqFig       a patch replaced by its nearest code-book entry (k-means code books, data/vq.json)
// Pure helpers are exported with module.exports for the node test.
(function () {
  "use strict";
  var S = 224;

  // ---------- pure helpers ----------
  function numPatches(s, p) { var g = Math.floor(s / p); return g * g; }
  function patchify(rgb, s, p) {            // rgb: flat array of s*s*3 numbers (row, column, colour); returns N arrays of p*p*3 numbers
    var g = Math.floor(s / p), out = [], r, c, i, j, k, v, o;
    for (r = 0; r < g; r++) for (c = 0; c < g; c++) {
      v = new Array(p * p * 3); k = 0;
      for (i = 0; i < p; i++) for (j = 0; j < p; j++) { o = ((r * p + i) * s + c * p + j) * 3; v[k++] = rgb[o]; v[k++] = rgb[o + 1]; v[k++] = rgb[o + 2]; }
      out.push(v);
    }
    return out;
  }
  function dot(a, b) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i] * b[i]; return s; }
  // move patches: the first k patches of `order` are passed along in a circle (patch order[i] gets the pixels of order[(i+1) mod k])
  function moveRGBA(src, s, p, order, k) {
    var out = new Uint8ClampedArray(src), g = s / p, t, a, b, i, j, o1, o2, c;
    if (k < 2) return out;
    for (t = 0; t < k; t++) {
      a = order[t]; b = order[(t + 1) % k];
      for (i = 0; i < p; i++) for (j = 0; j < p; j++) {
        o1 = (((Math.floor(a / g)) * p + i) * s + (a % g) * p + j) * 4; o2 = (((Math.floor(b / g)) * p + i) * s + (b % g) * p + j) * 4;
        for (c = 0; c < 4; c++) out[o1 + c] = src[o2 + c];
      }
    }
    return out;
  }
  function nearest(vec, book, K, dim) {      // index and squared distance of the closest of K code-book rows
    var best = 0, bd = Infinity, k, i, d, e, o;
    for (k = 0; k < K; k++) {
      o = k * dim; d = 0;
      for (i = 0; i < dim; i++) { e = vec[i] - book[o + i]; d += e * e; if (d >= bd) break; }
      if (d < bd) { bd = d; best = k; }
    }
    return { k: best, d2: bd };
  }
  function dequant(v, rowmax) { var u = v / 15; return u * u * rowmax; }
  function windowCount(r, c, g, half) {
    var r0 = Math.max(0, r - half), r1 = Math.min(g - 1, r + half), c0 = Math.max(0, c - half), c1 = Math.min(g - 1, c + half);
    return (r1 - r0 + 1) * (c1 - c0 + 1);
  }
  if (typeof module !== "undefined") { module.exports = { numPatches: numPatches, patchify: patchify, dot: dot, moveRGBA: moveRGBA, nearest: nearest, dequant: dequant, windowCount: windowCount }; return; }

  // ---------- page helpers ----------
  var PH = [
    { key: "hammerhead-2", name: "Hammerhead, Costa Rica", src: "hammerhead-2", row: 6, col: 6 },
    { key: "bamboo-1", name: "Grey bamboo shark, tank", src: "bamboo-1", row: 7, col: 4 },
    { key: "spadenose-2", name: "Spadenose sharks, basket", src: "spadenose-2", row: 6, col: 5 }
  ];
  var DATA = {}, IM = {}, CREDITS = {};
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function nf(v, d) { return Kit.fmt(v, d == null ? 2 : d); }
  function comma(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")"; }
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
  function cell(cv, ev, x0, size, g) {       // the grid cell under a pointer
    var r = cv.getBoundingClientRect(), x = (ev.clientX - r.left) - x0, y = ev.clientY - r.top;
    var c = Math.floor(x / (size / g)), rr = Math.floor(y / (size / g));
    if (c < 0 || rr < 0 || c >= g || rr >= g) return null; return { r: rr, c: c };
  }
  function divergeRGB(v, lim, P) {            // paper at 0, teal for positive, coral for negative
    var base = hexRGB(P.paper), t = Math.max(-1, Math.min(1, v / lim)), tgt = hexRGB(t >= 0 ? P.teal : P.coral), a = Math.abs(t);
    return [Math.round(base[0] + (tgt[0] - base[0]) * a), Math.round(base[1] + (tgt[1] - base[1]) * a), Math.round(base[2] + (tgt[2] - base[2]) * a)];
  }
  function pixelData(im) {
    var c = document.createElement("canvas"); c.width = S; c.height = S; var x = c.getContext("2d", { willReadFrequently: true });
    x.drawImage(im, 0, 0, S, S); return x.getImageData(0, 0, S, S);
  }
  function creditFor(key) { return (CREDITS[key] || "") + " Cropped and resized to 224 by 224 pixels."; }
  function photoChips(parent, initial, onPick) {
    var row = Kit.html("div", { class: "bar-row" }, parent);
    return Kit.chips(row, PH.map(function (p) { return p.name; }), initial, onPick);
  }
  function nextFrame(fn) { requestAnimationFrame(fn); }
  function onResize(fn) { window.addEventListener("resize", Kit.debounce(fn, 120)); Kit.onTheme(fn); }

  document.addEventListener("DOMContentLoaded", function () {
    var imgs = PH.map(function (p) { return loadImg("data/p224/" + p.src + ".jpg"); });
    Promise.all([Kit.data("data/vit.json"), Kit.data("data/shuffle.json"), Kit.data("data/vq.json"), Kit.data("../data/sharks/photos.json"), loadGray("data/possim.png")].concat(imgs)).then(function (r) {
      DATA.vit = r[0]; DATA.sh = r[1]; DATA.vq = r[2]; DATA.sim = r[4];
      r[3].photos.forEach(function (p) { var k = p.file.replace("photos/", "").replace(".jpg", ""); CREDITS[k] = p.credit_line; });
      PH.forEach(function (p, i) { var im = r[5 + i]; IM[p.key] = { img: im, data: pixelData(im) }; });
      cutFig(); embedFig(); simFig(); shuffleFig(); clsFig(); attendFig(); reachFig(); paramFig(); vqFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. cut a photo into patches, lift them into a row of tokens ----------
  function cutFig() {
    var PS = [112, 56, 32, 28, 16, 14, 8], st = { pi: 4, ph: 2, cls: false, placed: 0, anim: null, playing: false };
    var cv = Kit.$("cut-cv"), box = Kit.$("cut-box"), stepBtn, playBtn;
    photoChips(Kit.$("cut-photo"), 2, function (i) { st.ph = i; showAll(); });
    Kit.slider(Kit.$("cut-controls"), { label: "Patch side P", min: 0, max: 6, step: 1, value: 4, show: function (i) { return PS[i] + " px"; } }, function (i) { st.pi = i; showAll(); });
    var bar = Kit.$("cut-btns");
    stepBtn = Kit.button(bar, "Step", function () { if (st.placed >= slotList().length && !st.anim) st.placed = 0; fly(); });
    playBtn = Kit.button(bar, "Play", play);
    Kit.button(bar, "Show all", showAll);
    var lab = Kit.html("label", { class: "check" }, bar), chk = Kit.html("input", { type: "checkbox" }, lab); lab.appendChild(document.createTextNode(" add the class token"));
    chk.addEventListener("change", function () { st.cls = chk.checked; showAll(); });
    function geom() { var P = PS[st.pi]; return { P: P, g: S / P, N: (S / P) * (S / P) }; }
    function slotList() { var N = geom().N, a = [], i; if (N <= 11) { for (i = 0; i < N; i++) a.push(i); } else { for (i = 0; i < 9; i++) a.push(i); a.push(-1); a.push(N - 1); } if (st.cls) a.unshift(-2); return a; }
    function stop() { st.playing = false; playBtn.textContent = "Play"; }
    function showAll() { stop(); st.anim = null; st.placed = slotList().length; draw(); status(); eq(); }
    function fly(done) {
      if (st.anim) return;
      var s = slotList(); while (s[st.placed] === -1 || s[st.placed] === -2) st.placed++;
      if (st.placed >= s.length) { if (done) done(); draw(); status(); return; }
      var tok = s[st.placed]; st.anim = { tok: tok, slot: st.placed, p: 0 };
      Kit.animate(430, function (p) {
        st.anim.p = Kit.ease(p);
        if (p >= 1) { st.placed++; st.anim = null; var s2 = slotList(); while (s2[st.placed] === -1) st.placed++; status(); }
        draw(); if (p >= 1 && done) done();
      });
    }
    function play() {
      if (st.playing) { stop(); return; }
      if (st.placed >= slotList().length) st.placed = 0;
      st.playing = true; playBtn.textContent = "Pause";
      (function next() {
        if (!st.playing) return;
        if (st.placed >= slotList().length) { stop(); return; }
        fly(function () { setTimeout(next, 90); });
      })();
    }
    function status() {
      var G = geom(), s = slotList(), msg;
      if (st.placed >= s.length) msg = "All " + comma(G.N) + " patches are now tokens. Position numbers run in reading order, from the top left to the bottom right" + (st.cls ? ", and the class token takes position 0." : ".");
      else if (st.placed === 0 && !st.anim) msg = "Press Step to lift patch 0 into the row, or Play to lift the first nine and the last.";
      else { var t = st.anim ? st.anim.tok : s[st.placed - 1]; msg = "Token " + (t + (st.cls ? 1 : 0)) + " comes from grid row " + Math.floor(t / G.g) + ", column " + (t % G.g) + "."; }
      Kit.$("cut-status").textContent = msg;
    }
    function eq() {
      var G = geom(), n = G.N + 1, per = G.P * G.P * 3;
      Kit.$("cut-eqn").innerHTML = ebox("c-navy", S / G.P, "patches across") + eop("&times;") + ebox("c-navy", S / G.P, "patches down") + eop("=") + ebox("c-coral", comma(G.N), "patches, N") + eop("|") +
        ebox("c-sand", comma(per), "numbers in a patch, P &times; P &times; 3") + eop("|") + ebox("c-teal", comma(n * n), "attention pairs per head, (N + 1)<sup>2</sup>");
      Kit.$("cut-cv").setAttribute("aria-label", "The photo cut into " + G.N + " patches of " + G.P + " by " + G.P + " pixels, with the first patches lifted into a row of tokens");
    }
    function draw() {
      var G = geom(), P = Kit.palette(), W = Math.min(640, Kit.width(box)), sz = Math.min(W - 4, 300), x0 = Math.round((W - sz) / 2), y0 = 4;
      var s = slotList(), ns = s.length, gap = 6, t = Math.max(20, Math.min(56, Math.floor((W - 8 - gap * (ns - 1)) / ns)));
      var rowY = y0 + sz + 26, H = rowY + t + 30, im = IM[PH[st.ph].key].img, cs = sz / G.g, i, ctx = fitCanvas(cv, W, H);
      ctx.drawImage(im, x0, y0, sz, sz);
      ctx.lineWidth = 1;
      for (i = 0; i <= G.g; i++) {
        var p = Math.round(i * cs) + 0.5;
        ctx.strokeStyle = "rgba(255,255,255,0.8)"; ctx.beginPath(); ctx.moveTo(x0 + p, y0); ctx.lineTo(x0 + p, y0 + sz); ctx.moveTo(x0, y0 + p); ctx.lineTo(x0 + sz, y0 + p); ctx.stroke();
        ctx.strokeStyle = "rgba(0,0,0,0.3)"; ctx.beginPath(); ctx.moveTo(x0 + p + 1, y0); ctx.lineTo(x0 + p + 1, y0 + sz); ctx.moveTo(x0, y0 + p + 1); ctx.lineTo(x0 + sz, y0 + p + 1); ctx.stroke();
      }
      var teal = hexRGB(P.teal), coral = hexRGB(P.coral);
      function tokCell(tok) { return { x: x0 + (tok % G.g) * cs, y: y0 + Math.floor(tok / G.g) * cs }; }
      s.forEach(function (tok, si) {
        if (tok < 0 || si >= st.placed) return;
        var c = tokCell(tok); ctx.strokeStyle = rgba(teal, 1); ctx.lineWidth = 2; ctx.strokeRect(c.x + 1, c.y + 1, cs - 2, cs - 2);
        if (cs >= 14) { ctx.fillStyle = "rgba(0,0,0,0.55)"; ctx.fillRect(c.x + 1, c.y + 1, Math.min(cs - 2, 26), 13); ctx.fillStyle = "#fff"; ctx.font = "11px ui-monospace, Menlo, monospace"; ctx.textAlign = "left"; ctx.fillText(String(tok + (st.cls ? 1 : 0)), c.x + 3, c.y + 11); }
      });
      var total = ns * t + (ns - 1) * gap, xr = Math.round((W - total) / 2);
      function drawTile(tok, sx, sy, tt, flying) {
        ctx.imageSmoothingEnabled = G.P < tt;
        var c = { r: Math.floor(tok / G.g), c: tok % G.g };
        ctx.drawImage(im, c.c * G.P, c.r * G.P, G.P, G.P, sx, sy, tt, tt); ctx.imageSmoothingEnabled = true;
        ctx.lineWidth = 2; ctx.strokeStyle = rgba(flying ? coral : teal, 1); ctx.strokeRect(sx, sy, tt, tt);
      }
      ctx.textAlign = "center"; ctx.font = "12px ui-monospace, Menlo, monospace";
      s.forEach(function (tok, si) {
        var sx = xr + si * (t + gap), filled = si < st.placed;
        if (tok === -1) {
          ctx.fillStyle = P.muted; ctx.font = "20px Spectral, Georgia, serif"; if (filled) ctx.fillText("···", sx + t / 2, rowY + t / 2 + 4); ctx.font = "12px ui-monospace, Menlo, monospace"; return;
        }
        if (!filled) { ctx.setLineDash([4, 3]); ctx.strokeStyle = P.rule; ctx.lineWidth = 1.5; ctx.strokeRect(sx + 0.5, rowY + 0.5, t - 1, t - 1); ctx.setLineDash([]); return; }
        if (tok === -2) { ctx.setLineDash([4, 3]); ctx.strokeStyle = rgba(coral, 1); ctx.lineWidth = 2; ctx.strokeRect(sx + 1, rowY + 1, t - 2, t - 2); ctx.setLineDash([]); ctx.fillStyle = P.coral; ctx.font = "600 " + Math.max(10, Math.floor(t / 3.4)) + "px Spectral, Georgia, serif"; ctx.fillText("CLS", sx + t / 2, rowY + t / 2 + 4); ctx.font = "12px ui-monospace, Menlo, monospace"; ctx.fillStyle = P.ink2; ctx.fillText("0", sx + t / 2, rowY + t + 16); return; }
        drawTile(tok, sx, rowY, t, false);
        ctx.fillStyle = P.ink2; ctx.fillText(String(tok + (st.cls ? 1 : 0)), sx + t / 2, rowY + t + 16);
      });
      ctx.fillStyle = P.muted; ctx.font = "13px Spectral, Georgia, serif"; ctx.textAlign = "left"; ctx.fillText("token row, with the position numbers below", xr, rowY - 8);
      if (st.anim) {
        var a = st.anim, c0 = tokCell(a.tok), sx2 = xr + a.slot * (t + gap), e = a.p;
        drawTile(a.tok, c0.x + (sx2 - c0.x) * e, c0.y + (rowY - c0.y) * e, cs + (t - cs) * e, true);
      }
    }
    function credit() { Kit.$("cut-credit").textContent = creditFor(PH[st.ph].key); }
    onResize(draw);
    showAll(); credit();
    Kit.$("cut-photo").addEventListener("click", function () { setTimeout(credit, 0); });
  }

  // ---------- 2. a patch times a matrix gives a token ----------
  function embedFig() {
    var V = DATA.vit, F = V.filters, nF = F.index.length, st = { ph: 0, r: 6, c: 6, P: 16 };
    var cv = Kit.$("em-cv"), box = Kit.$("em-photo");
    photoChips(Kit.$("em-bar"), 0, function (i) { st.ph = i; st.r = PH[i].row; st.c = PH[i].col; draw(); credit(); });
    var pc = Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("em-bar")), ["P = 8", "P = 16", "P = 32"], 1, function (i) { st.P = [8, 16, 32][i]; dims(); });
    var xs = Kit.$("em-x"), pcv = Kit.$("em-patch"), rows = Kit.$("em-rows"), cells = [];
    function filtCanvas(j) {
      var c = document.createElement("canvas"); c.width = 16; c.height = 16; c.className = "em-f"; c.setAttribute("role", "img"); c.setAttribute("aria-label", "Row " + F.index[j] + " of the matrix drawn as a 16 by 16 colour picture");
      var x = c.getContext("2d"), id = x.createImageData(16, 16), w = F.w[j], mn = Infinity, mx = -Infinity, i;
      for (i = 0; i < w.length; i++) { if (w[i] < mn) mn = w[i]; if (w[i] > mx) mx = w[i]; }
      for (i = 0; i < 256; i++) { for (var k = 0; k < 3; k++) id.data[i * 4 + k] = Math.round(255 * (w[i * 3 + k] - mn) / (mx - mn)); id.data[i * 4 + 3] = 255; }
      x.putImageData(id, 0, 0); return c;
    }
    for (var j = 0; j < nF; j++) {
      var line = Kit.html("div", { class: "em-line" }, rows);
      line.appendChild(filtCanvas(j));
      Kit.html("span", { class: "em-t" }, line, "row " + F.index[j] + " · x + b =");
      var z = Kit.html("b", { class: "em-z" }, line, "0"), bar = Kit.html("span", { class: "em-bar" }, line), fill = Kit.html("i", {}, bar);
      cells.push({ z: z, fill: fill });
    }
    function patchX() {
      var d = IM[PH[st.ph].key].data.data, x = new Array(768), k = 0, i, jj, c, o;
      for (i = 0; i < 16; i++) for (jj = 0; jj < 16; jj++) for (c = 0; c < 3; c++) {
        o = ((st.r * 16 + i) * S + st.c * 16 + jj) * 4 + c; x[k++] = (d[o] / 255 - V.mean[c]) / V.std[c];
      }
      return x;
    }
    function draw() {
      var P = Kit.palette(), W = Math.min(300, Kit.width(box)), ctx = fitCanvas(cv, W, W), cs = W / 14, im = IM[PH[st.ph].key].img, i;
      ctx.drawImage(im, 0, 0, W, W); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(255,255,255,0.45)";
      for (i = 0; i <= 14; i++) { var p = Math.round(i * cs) + 0.5; ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, W); ctx.moveTo(0, p); ctx.lineTo(W, p); ctx.stroke(); }
      ctx.lineWidth = 3; ctx.strokeStyle = P.coral; ctx.strokeRect(st.c * cs + 1.5, st.r * cs + 1.5, cs - 3, cs - 3);
      // the patch magnified
      var pctx = pcv.getContext("2d"); pcv.width = 96; pcv.height = 96; pctx.imageSmoothingEnabled = false;
      pctx.drawImage(im, st.c * 16, st.r * 16, 16, 16, 0, 0, 96, 96);
      // x laid out in 16 rows of 48 cells: three cells (red, green, blue) for a pixel
      var x = patchX(), xc = xs.getContext("2d"); xs.width = 240; xs.height = 80;
      for (var k = 0; k < 768; k++) { var rr = Math.floor(k / 48), cc = k % 48, col = divergeRGB(x[k], 2.2, P); xc.fillStyle = "rgb(" + col.join(",") + ")"; xc.fillRect(cc * 5, rr * 5, 5, 5); }
      for (var j = 0; j < nF; j++) {
        var zv = dot(F.w[j], x) + F.bias[j]; cells[j].z.textContent = nf(zv, 2);
        var f = Math.min(1, Math.abs(zv) / 0.35); cells[j].fill.style.width = Math.round(50 * f) + "%"; cells[j].fill.style.left = zv >= 0 ? "50%" : (50 - Math.round(50 * f)) + "%";
        cells[j].fill.style.background = zv >= 0 ? P.teal : P.coral;
      }
      Kit.$("em-status").textContent = "Patch at grid row " + st.r + ", column " + st.c + " (position " + (st.r * 14 + st.c + 1) + " with the class token in front). Click another patch or use the arrow keys.";
      cv.setAttribute("aria-label", "Photo with patch row " + st.r + ", column " + st.c + " selected");
    }
    function dims() {
      var P = st.P, per = P * P * 3, d = 768, N = (S / P) * (S / P), w = d * per;
      Kit.$("em-dims").innerHTML = ebox("c-navy", d + " &times; " + comma(per), "W<sub>patch</sub>: d rows, P&sup2;&middot;3 columns") + eop("&times;") + ebox("c-coral", comma(per) + " &times; 1", "x, the flattened patch") + eop("+") + ebox("c-sand", d + " &times; 1", "b, the bias") + eop("=") + ebox("c-teal", d + " &times; 1", "e, the token");
      Kit.$("em-count").innerHTML = ebox("c-navy", comma(w), "weights, d &times; P&sup2;&middot;3") + eop("+") + ebox("c-sand", d, "biases") + eop("=") + ebox("c-teal", comma(w + d), "numbers learned") + eop("|") +
        ebox("c-coral", comma(w), "multiply-adds for one patch") + eop("&times;") + ebox("c-coral", comma(N), "patches") + eop("=") + ebox("c-teal", comma(w * N), "multiply-adds for the photo");
      Kit.$("em-note").textContent = P === 16 ? "This is the size of the trained model on this page, and its numbers are live in the figure above." : "The trained model here has P = 16, so the rows above keep showing P = 16. These boxes show the sizes that a model with this patch side would need.";
    }
    cv.setAttribute("tabindex", "0");
    cv.addEventListener("click", function (ev) { var W = Math.min(300, Kit.width(box)), c = cell(cv, ev, 0, W, 14); if (c) { st.r = c.r; st.c = c.c; draw(); } });
    cv.addEventListener("keydown", function (ev) {
      var m = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[ev.key]; if (!m) return; ev.preventDefault();
      st.r = Math.max(0, Math.min(13, st.r + m[0])); st.c = Math.max(0, Math.min(13, st.c + m[1])); draw();
    });
    function credit() { Kit.$("em-credit").textContent = creditFor(PH[st.ph].key); }
    onResize(draw); draw(); dims(); credit();
  }

  // ---------- 3a. similarity of the learned position embeddings ----------
  function simFig() {
    var st = { r: 6, c: 6 }, A = DATA.sim, cv = Kit.$("ps-map"), ph = Kit.$("ps-photo"), box = Kit.$("ps-box");
    function cosine(a, b) { return A.a[a * 196 + b] / 255 * 2 - 1; }
    function draw() {
      var P = Kit.palette(), W = Math.min(260, Math.floor((Kit.width(box) - 16) / 2)), cs = W / 14, i, j, sel = st.r * 14 + st.c;
      var c1 = fitCanvas(cv, W, W), c2 = fitCanvas(ph, W, W), im = IM[PH[0].key].img;
      for (i = 0; i < 196; i++) {
        var col = divergeRGB(cosine(sel, i), 1, P); c1.fillStyle = "rgb(" + col.join(",") + ")"; c1.fillRect((i % 14) * cs, Math.floor(i / 14) * cs, cs + 0.5, cs + 0.5);
      }
      c1.lineWidth = 3; c1.strokeStyle = P.ink; c1.strokeRect(st.c * cs + 1.5, st.r * cs + 1.5, cs - 3, cs - 3);
      c2.drawImage(im, 0, 0, W, W); c2.lineWidth = 3; c2.strokeStyle = P.coral; c2.strokeRect(st.c * cs + 1.5, st.r * cs + 1.5, cs - 3, cs - 3);
      var right = st.c < 13 ? cosine(sel, sel + 1) : cosine(sel, sel - 1), r3 = st.c + 3 < 14 ? cosine(sel, sel + 3) : cosine(sel, sel - 3);
      var far = [[0, 0], [0, 13], [13, 0], [13, 13]].map(function (q) { return { d: Math.hypot(q[0] - st.r, q[1] - st.c), v: cosine(sel, q[0] * 14 + q[1]) }; }).sort(function (a, b) { return b.d - a.d; })[0];
      Kit.$("ps-eq").innerHTML = ebox("c-teal", nf(right, 2), "cosine to the next patch along the row") + ebox("c-navy", nf(r3, 2), "to the patch 3 along the row") + ebox("c-coral", nf(far.v, 2), "to the farthest corner, " + nf(far.d, 1) + " patches away");
      Kit.$("ps-status").textContent = "Position " + (sel + 1) + " (row " + st.r + ", column " + st.c + ") compared with the 196 patch positions. Teal is similar, coral is opposite.";
      cv.setAttribute("aria-label", "Map of the cosine similarity of the position embedding of row " + st.r + ", column " + st.c + " with all other positions");
    }
    function pick(ev) { var W = Math.min(260, Math.floor((Kit.width(box) - 16) / 2)), c = cell(ev.currentTarget, ev, 0, W, 14); if (c) { st.r = c.r; st.c = c.c; draw(); } }
    [cv, ph].forEach(function (n) {
      n.setAttribute("tabindex", "0"); n.addEventListener("click", pick);
      n.addEventListener("keydown", function (ev) { var m = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[ev.key]; if (!m) return; ev.preventDefault(); st.r = Math.max(0, Math.min(13, st.r + m[0])); st.c = Math.max(0, Math.min(13, st.c + m[1])); draw(); });
    });
    Kit.$("ps-credit").textContent = creditFor("hammerhead-2");
    onResize(draw); draw();
  }

  // ---------- 3b. move patches, watch the call ----------
  function shuffleFig() {
    var SH = DATA.sh, st = { ph: 0, ki: 4 }, cv = Kit.$("sh-cv"), box = Kit.$("sh-box"), order = SH.order;
    photoChips(Kit.$("sh-bar"), 0, function (i) { st.ph = i; draw(); credit(); });
    Kit.slider(Kit.$("sh-controls"), { label: "Patches moved, k", min: 0, max: SH.ks.length - 1, step: 1, value: 4, show: function (i) { return SH.ks[i] + " of 196"; } }, function (i) { st.ki = i; draw(); });
    var work = document.createElement("canvas"); work.width = S; work.height = S;
    function draw() {
      var P = Kit.palette(), k = SH.ks[st.ki], key = PH[st.ph].key, W = Math.min(260, Math.floor(Kit.width(box) * 0.5)), ctx = fitCanvas(cv, W, W);
      var src = IM[key].data, out = new ImageData(moveRGBA(src.data, S, 16, order, k), S, S); work.getContext("2d").putImageData(out, 0, 0);
      ctx.drawImage(work, 0, 0, W, W);
      var t3 = SH.shown[key][st.ki];
      Kit.$("sh-top").innerHTML = "<b>Top three calls of the ViT</b><ol>" + t3.map(function (t) { return "<li>" + t[0] + " <span>" + Kit.pct(t[1], 0) + "</span></li>"; }).join("") + "</ol>";
      var ix = SH.ks[st.ki];
      Kit.plot("sh-plot", function (Q) {
        return [
          { x: SH.ks, y: SH.keep_vit.map(function (v) { return 100 * v; }), mode: "lines+markers", line: { color: Q.teal, width: 3 }, marker: { size: 6 }, hovertemplate: "%{x} patches moved: %{y:.0f}% keep the call<extra>ViT-B/16</extra>" },
          { x: SH.ks, y: SH.keep_resnet.map(function (v) { return 100 * v; }), mode: "lines+markers", line: { color: Q.navy, width: 3, dash: "dash" }, marker: { size: 6 }, hovertemplate: "%{x} patches moved: %{y:.0f}% keep the call<extra>ResNet-18</extra>" },
          { x: [ix, ix], y: [SH.keep_vit[st.ki] * 100, SH.keep_resnet[st.ki] * 100], mode: "markers", marker: { color: Q.coral, size: 11, line: { color: "#fff", width: 2 } }, hoverinfo: "skip" }
        ];
      }, function (Q) {
        return { xaxis: { title: { text: "patches moved, k" }, range: [-4, 202] }, yaxis: { title: { text: "top call unchanged, %" }, range: [-4, 104] }, margin: { l: 58, r: 12, t: 10, b: 46 },
          annotations: [{ x: 96, y: SH.keep_vit[7] * 100, text: "ViT-B/16", showarrow: false, yshift: 16, font: { color: Q.teal, size: 13 } }, { x: 64, y: SH.keep_resnet[6] * 100, text: "ResNet-18", showarrow: false, yshift: -16, font: { color: Q.navy, size: 13 } }] };
      });
      Kit.$("sh-read").innerHTML = "<span>At <b>" + k + "</b> patches moved, the top call is unchanged for <b>" + Math.round(100 * SH.keep_vit[st.ki]) + "%</b> of the runs of the ViT and <b>" + Math.round(100 * SH.keep_resnet[st.ki]) + "%</b> of ResNet-18 (" + SH.n_photos + " photos &times; " + SH.n_orders + " random orders).</span>";
      cv.setAttribute("aria-label", "The photo with " + k + " patches moved");
    }
    function credit() { Kit.$("sh-credit").textContent = creditFor(PH[st.ph].key); }
    onResize(draw); draw(); credit();
  }

  // ---------- 4. the class token ----------
  function clsFig() {
    var V = DATA.vit, st = { ph: 0, l: 11 }, cv = Kit.$("cl-cv"), box = Kit.$("cl-box"), dg = Kit.$("cl-diagram");
    photoChips(Kit.$("cl-bar"), 0, function (i) { st.ph = i; draw(); credit(); });
    Kit.slider(Kit.$("cl-controls"), { label: "Block", min: 0, max: 11, step: 1, value: 11, show: function (v) { return "block " + v + " of 0 to 11"; } }, function (v) { st.l = v; draw(); });
    function diagram(P) {
      dg.setAttribute("viewBox", "0 0 360 150"); dg.innerHTML = "";
      function rect(x, y, w, h, fill, stroke, dash) { return Kit.el("rect", { x: x, y: y, width: w, height: h, rx: 4, fill: fill, stroke: stroke, "stroke-width": 1.8, "stroke-dasharray": dash || "" }, dg); }
      function text(x, y, s, fill, size, anchor, weight) { var t = Kit.el("text", { x: x, y: y, fill: fill, "font-size": size || 12, "text-anchor": anchor || "middle", "font-weight": weight || 400 }, dg); t.textContent = s; return t; }
      rect(6, 8, 40, 32, "none", P.coral, "4 3"); text(26, 29, "CLS", P.coral, 12, "middle", 600);
      for (var i = 0; i < 6; i++) rect(60 + i * 26, 8, 22, 32, P.paper, P.navy);
      text(224, 30, "\u2026", P.ink2, 16); rect(238, 8, 22, 32, P.paper, P.navy);
      text(26, 54, "class token", P.muted, 11); text(150, 54, "196 patch tokens (token e + position)", P.muted, 11, "start");
      text(6, 78, "12 blocks, one recipe, the same weights for all tokens", P.muted, 11, "start");
      for (var b = 0; b < 12; b++) { var on = b === st.l; rect(6 + b * 28, 86, 24, 26, on ? P.teal : P.paper, on ? P.teal : P.muted); text(18 + b * 28, 103, String(b), on ? P.bg : P.ink2, 11); }
      var top = V.photos[PH[st.ph].key].top5[0];
      rect(6, 120, 40, 26, "none", P.coral, "4 3"); text(26, 137, "CLS", P.coral, 12, "middle", 600);
      text(54, 137, "after block 11, then the linear head: " + top[0] + " " + Kit.pct(top[1], 0), P.teal, 12, "start", 600);
    }
    function draw() {
      var P = Kit.palette(), W = Math.min(300, Kit.width(box)), ctx = fitCanvas(cv, W, W), key = PH[st.ph].key, im = IM[key].img, cs = W / 14, i;
      var att = V.cls_by_layer[key][st.l], mx = Math.max.apply(null, att), selfShare = V.cls_to_cls[key][st.l];
      ctx.drawImage(im, 0, 0, W, W);
      for (i = 0; i < 196; i++) { var u = Math.sqrt(att[i] / mx); ctx.fillStyle = "rgba(0,0,0," + (0.62 * (1 - u)).toFixed(3) + ")"; ctx.fillRect((i % 14) * cs, Math.floor(i / 14) * cs, cs + 0.5, cs + 0.5); }
      var sumAtt = att.reduce(function (a, b) { return a + b; }, 0), tot = 1 - selfShare, sorted = att.slice().sort(function (a, b) { return b - a; }), cum = 0, need = 0;   // the stored maps are rounded, so the patch share is 1 minus the share on the class token itself
      for (i = 0; i < 196; i++) { cum += sorted[i]; if (cum >= 0.5 * sumAtt) { need = i + 1; break; } }
      Kit.$("cl-eq").innerHTML = ebox("c-sand", Kit.pct(selfShare, 0), "weight on itself") + eop("+") + ebox("c-teal", Kit.pct(tot, 0), "weight on the 196 patches") + eop("=") + ebox("c-navy", "100%", "mean over the 12 heads") + eop("|") + ebox("c-coral", need, "patches have half of the patch weight");
      Kit.$("cl-status").textContent = "Block " + st.l + ": the class token reads mostly " + (tot > selfShare ? "the patches" : "itself") + ". Bright patches get most of its weight; the brightest patch has " + nf(100 * mx * tot / sumAtt, 1) + "% of all its weight.";
      cv.setAttribute("aria-label", "Photo shaded by how much attention the class token gives a patch in block " + st.l);
      diagram(P);
    }
    function credit() { Kit.$("cl-credit").textContent = creditFor(PH[st.ph].key); }
    onResize(draw); draw(); credit();
  }

  // ---------- 5. click a patch, see where the head looks ----------
  var ATT = {};
  function loadAtt(key, l, h) {
    var id = key + "_L" + l + "H" + h;
    if (!ATT[id]) ATT[id] = loadGray("data/attn/" + id + ".png").then(function (g) { for (var i = 0; i < g.a.length; i++) g.a[i] = Math.round(g.a[i] / 17); return g; });
    return ATT[id];
  }
  function attendFig() {
    var V = DATA.vit, H = V.heads, st = { ph: 0, hd: 1, r: 6, c: 6 }, cv = Kit.$("at-cv"), box = Kit.$("at-box"), cur = null;
    var phs = [0, 1];
    var pr = Kit.html("div", { class: "bar-row" }, Kit.$("at-bar"));
    Kit.chips(pr, phs.map(function (i) { return PH[i].name; }), 0, function (i) { st.ph = phs[i]; st.r = PH[st.ph].row; st.c = PH[st.ph].col; load(); credit(); });
    var hr = Kit.html("div", { class: "bar-row" }, Kit.$("at-bar"));
    var hb = Kit.chips(hr, H.map(function (h) { return "L" + h.layer + "H" + h.head; }), 1, function (i) { st.hd = i; load(); });
    function load() {
      var h = H[st.hd], key = PH[st.ph].key; cur = null;
      loadAtt(key, h.layer, h.head).then(function (g) { if (key === PH[st.ph].key && h === H[st.hd]) { cur = g; draw(); } });
    }
    function row(q) {   // weights of query patch q (0-based) over the 197 keys, with the dequantised values rescaled to sum to 1
      var h = H[st.hd], rm = V.rowmax[PH[st.ph].key + "_L" + h.layer + "H" + h.head][q + 1], w = new Array(197), s = 0, k;
      for (k = 0; k < 197; k++) { w[k] = dequant(cur.a[(q + 1) * 197 + k], rm); s += w[k]; }
      for (k = 0; k < 197; k++) w[k] /= s; return { w: w, rm: rm };
    }
    function draw() {
      if (!cur) return;
      var P = Kit.palette(), W = Math.min(340, Kit.width(box)), ctx = fitCanvas(cv, W, W), cs = W / 14, q = st.r * 14 + st.c, R = row(q), h = H[st.hd], key = PH[st.ph].key, i;
      ctx.drawImage(IM[key].img, 0, 0, W, W);
      var mx = 0; for (i = 1; i < 197; i++) if (R.w[i] > mx) mx = R.w[i];
      var tealC = hexRGB(P.teal);
      for (i = 0; i < 196; i++) { var u = Math.sqrt(R.w[i + 1] / mx); ctx.fillStyle = "rgba(0,0,0," + (0.55 * (1 - u)).toFixed(3) + ")"; ctx.fillRect((i % 14) * cs, Math.floor(i / 14) * cs, cs + 0.5, cs + 0.5); ctx.fillStyle = rgba(tealC, (0.5 * u * u).toFixed(3)); ctx.fillRect((i % 14) * cs, Math.floor(i / 14) * cs, cs + 0.5, cs + 0.5); }
      ctx.lineWidth = 3; ctx.strokeStyle = P.coral; ctx.strokeRect(st.c * cs + 1.5, st.r * cs + 1.5, cs - 3, cs - 3);
      var self = R.w[q + 1], near = 0, far = 0, d = 0, nbr = 0, best = 1;
      for (i = 0; i < 196; i++) {
        var rr = Math.floor(i / 14), cc = i % 14, cheb = Math.max(Math.abs(rr - st.r), Math.abs(cc - st.c));
        d += R.w[i + 1] * Math.hypot(rr - st.r, cc - st.c);
        if (i === q) continue; if (cheb <= 1) near += R.w[i + 1]; else far += R.w[i + 1];
        if (R.w[i + 1] > R.w[best]) best = i + 1;
      }
      var cls = R.w[0], patchSum = 1 - cls, dmean = d / patchSum;
      Kit.$("at-eq").innerHTML = ebox("c-coral", Kit.pct(self, 0), "on itself") + eop("+") + ebox("c-teal", Kit.pct(near, 0), "on the 8 neighbours") + eop("+") + ebox("c-navy", Kit.pct(far, 0), "on patches farther away") + eop("+") + ebox("c-sand", Kit.pct(cls, 0), "on the class token") + eop("=") + ebox("", "100%", "of this head's weight");
      Kit.$("at-eq2").innerHTML = ebox("c-navy", nf(dmean, 1), "mean distance of this patch's attention, in patches") + eop("|") + ebox("c-sand", nf(h.dist, 1), "mean distance of this head over 14 photos") + eop("|") + ebox("c-teal", Kit.pct(R.w[best], 0), "largest weight, on row " + Math.floor((best - 1) / 14) + ", column " + ((best - 1) % 14));
      Kit.$("at-status").textContent = "Layer " + h.layer + ", head " + h.head + " (counting from 0), " + ({ wide: "a head whose weight spreads over patches across the photo", local: "a head whose weight stays on the neighbours", body: "a head whose weight falls mostly on the animal's body", scattered: "a head whose weight goes to a few scattered patches" }[h.kind] || "") + ", for this query. Query patch: row " + st.r + ", column " + st.c + ".";
      cv.setAttribute("aria-label", "Photo shaded by the attention of layer " + h.layer + " head " + h.head + " from the selected patch: " + Kit.pct(self, 0) + " on itself, " + Kit.pct(near, 0) + " on neighbours, " + Kit.pct(far, 0) + " farther away");
    }
    cv.setAttribute("tabindex", "0");
    cv.addEventListener("click", function (ev) { var W = Math.min(340, Kit.width(box)), c = cell(cv, ev, 0, W, 14); if (c) { st.r = c.r; st.c = c.c; draw(); } });
    cv.addEventListener("keydown", function (ev) {
      var m = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[ev.key]; if (!m) return; ev.preventDefault();
      st.r = Math.max(0, Math.min(13, st.r + m[0])); st.c = Math.max(0, Math.min(13, st.c + m[1])); draw();
    });
    function credit() { Kit.$("at-credit").textContent = creditFor(PH[st.ph].key); }
    onResize(draw); load(); credit();
  }

  // ---------- 6a. what a CNN window and a ViT block can reach ----------
  function reachFig() {
    var V = DATA.vit, st = { k: 1, r: 6, c: 6, half: 1 }, cv = Kit.$("rc-cv"), box = Kit.$("rc-box");
    Kit.slider(Kit.$("rc-controls"), { label: "Layers stacked", min: 1, max: 12, step: 1, value: 1, show: function (v) { return v + (v === 1 ? " layer" : " layers"); } }, function (v) { st.k = v; draw(); });
    var hr = Kit.html("div", { class: "bar-row" }, Kit.$("rc-bar")); Kit.html("span", { class: "label" }, hr, "Window in the second chart:");
    Kit.chips(hr, ["3 by 3", "5 by 5", "7 by 7"], 0, function (i) { st.half = i + 1; charts(); });
    function draw() {
      var P = Kit.palette(), W2 = Math.min(250, Math.floor((Kit.width(box) - 12) / 2)), ctx = fitCanvas(cv, W2 * 2 + 12, W2 + 24), cs = W2 / 14, key = PH[0].key, im = IM[key].img, side = 2 * st.k + 1, i, pane;
      var half = st.k, win = windowCount(st.r, st.c, 14, half);
      for (pane = 0; pane < 2; pane++) {
        var ox = pane * (W2 + 12);
        ctx.drawImage(im, ox, 0, W2, W2);
        for (i = 0; i < 196; i++) {
          var rr = Math.floor(i / 14), cc = i % 14, inside = pane === 0 ? (Math.abs(rr - st.r) <= half && Math.abs(cc - st.c) <= half) : true;
          ctx.fillStyle = inside ? (pane === 1 ? rgba(hexRGB(P.coral), 0.18) : "rgba(0,0,0,0)") : "rgba(0,0,0,0.62)"; ctx.fillRect(ox + cc * cs, rr * cs, cs + 0.5, cs + 0.5);
        }
        ctx.lineWidth = 3; ctx.strokeStyle = P.coral; ctx.strokeRect(ox + st.c * cs + 1.5, st.r * cs + 1.5, cs - 3, cs - 3);
        ctx.fillStyle = P.ink2; ctx.font = "14px Spectral, Georgia, serif"; ctx.textAlign = "left";
        ctx.fillText(pane === 0 ? "CNN window " + side + " by " + side : "ViT: reads all 196", ox, W2 + 18);
      }
      Kit.$("rc-eq").innerHTML = ebox("c-navy", win, "patches in the CNN window, of 196") + eop("|") + ebox("c-coral", "197", "tokens a ViT head can read in block 1") + eop("|") + ebox("c-teal", nf(st.k * Math.SQRT2, 1), "farthest diagonal reach of the stack, patches");
      Kit.$("rc-status").textContent = "Click a patch to move the window. A stack of " + st.k + (st.k === 1 ? " layer" : " layers") + " of 3 by 3 reads at most " + win + " of the 196 patches around it.";
      cv.setAttribute("aria-label", "Two copies of the photo: the window a stack of " + st.k + " 3 by 3 layers sees, and the whole photo that a ViT block can read");
      charts();
    }
    function charts() {
      var xs = [], ys = [], names = [], l, h;
      for (l = 0; l < 12; l++) for (h = 0; h < 12; h++) { xs.push(l + (h - 5.5) * 0.04); ys.push(V.dist_patches[l][h]); names.push("L" + l + "H" + h); }
      Kit.plot("rc-plot1", function (Q) {
        var lx = [], ly = []; for (var k = 1; k <= 12; k++) { lx.push(k - 1); ly.push(k * Math.SQRT2); }
        return [
          { x: xs, y: ys, text: names, mode: "markers", marker: { color: Q.navy, size: 7, opacity: 0.75 }, hovertemplate: "%{text}: %{y:.1f} patches<extra></extra>" },
          { x: lx, y: ly, mode: "lines", line: { color: Q.coral, width: 3 }, hovertemplate: "%{y:.1f} patches<extra>farthest reach of k 3 by 3 layers</extra>" },
          { x: [-0.5, 11.5], y: [7.28, 7.28], mode: "lines", line: { color: Q.sand, width: 2, dash: "dash" }, hovertemplate: "7.3 patches<extra>even attention on all patches</extra>" }
        ];
      }, function (Q) { return { xaxis: { title: { text: "block (0 to 11)" }, range: [-0.6, 11.6], dtick: 1 }, yaxis: { title: { text: "mean attention distance, patches" }, range: [0, 12] }, margin: { l: 56, r: 12, t: 10, b: 46 },
        annotations: [{ x: 5.0, y: 11.3, text: "CNN reach", showarrow: false, font: { color: Q.coral, size: 13 } }, { x: 9.6, y: 8.6, text: "even attention", showarrow: false, font: { color: Q.sand, size: 13 } }, { x: 4.2, y: 0.8, text: "12 heads per block", showarrow: false, font: { color: Q.navy, size: 13 } }] }; });
      var share = V.window_share.map(function (r) { return 100 * r[st.half]; }), uni = 0, rr, cc;
      for (rr = 0; rr < 14; rr++) for (cc = 0; cc < 14; cc++) uni += windowCount(rr, cc, 14, st.half) / 196; uni = 100 * uni / 196;
      Kit.plot("rc-plot2", function (Q) {
        return [{ x: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], y: share, type: "bar", marker: { color: Q.teal }, hovertemplate: "block %{x}: %{y:.0f}% of the weight<extra></extra>" },
          { x: [-0.5, 11.5], y: [uni, uni], mode: "lines", line: { color: Q.sand, width: 2, dash: "dash" }, hovertemplate: "%{y:.1f}%<extra>even attention on all patches</extra>" }];
      }, function (Q) { return { xaxis: { title: { text: "block (0 to 11)" }, dtick: 1, range: [-0.6, 11.6] }, yaxis: { title: { text: "attention inside the window, %" }, range: [0, 60] }, margin: { l: 56, r: 12, t: 10, b: 46 }, bargap: 0.25 }; });
      Kit.$("rc-read").innerHTML = "<span>Block 0: <b>" + nf(100 * V.window_share[0][st.half], 1) + "%</b> of the weight falls in the " + (2 * st.half + 1) + " by " + (2 * st.half + 1) + " window. Block 3: <b>" + nf(100 * V.window_share[3][st.half], 1) + "%</b>. Block 11: <b>" + nf(100 * V.window_share[11][st.half], 1) + "%</b>. Even attention would give " + nf(uni, 1) + "%.</span>";
    }
    cv.setAttribute("tabindex", "0");
    cv.addEventListener("click", function (ev) { var W2 = Math.min(250, Math.floor((Kit.width(box) - 12) / 2)), c = cell(cv, ev, 0, W2, 14); if (!c) { var r = cv.getBoundingClientRect(); c = cell(cv, ev, W2 + 12, W2, 14); } if (c) { st.r = c.r; st.c = c.c; draw(); } });
    Kit.$("rc-credit").textContent = creditFor("hammerhead-2");
    onResize(draw); draw();
  }

  // ---------- 6b. parameter counts ----------
  function paramFig() {
    var V = DATA.vit, Pm = V.params, mode = 0;
    Kit.chips(Kit.$("pm-bar"), ["Whole models", "Inside ViT-B/16"], 0, function (i) { mode = i; draw(); });
    function draw() {
      var items;
      if (mode === 0) items = [
        ["ResNet-18", Pm.resnet18, Pm.resnet18_gflops, Pm.resnet18_top1, "cnn"], ["ResNet-50", Pm.resnet50, Pm.resnet50_gflops, Pm.resnet50_top1, "cnn"],
        ["ViT-B/32", Pm.vit_b_32, Pm.vit_b_32_gflops, Pm.vit_b_32_top1, "vit"], ["ViT-B/16", Pm.vit_b_16, Pm.vit_b_16_gflops, Pm.vit_b_16_top1, "vit"], ["ViT-L/16", Pm.vit_l_16, Pm.vit_l_16_gflops, Pm.vit_l_16_top1, "vit"]];
      else items = [["12 blocks", Pm.blocks12, 0, 0, "vit"], ["class head", Pm.head, 0, 0, "vit"], ["patch matrix", Pm.patch_embed, 0, 0, "vit"], ["position table", Pm.pos, 0, 0, "vit"], ["final norm", Pm.ln, 0, 0, "vit"], ["class token", Pm.cls, 0, 0, "vit"]];
      var names = items.map(function (t) { return t[0]; }), vals = items.map(function (t) { return t[1]; });
      Kit.plot("pm-plot", function (Q) {
        return [{ type: "bar", orientation: "h", y: names, x: vals, marker: { color: items.map(function (t) { return t[4] === "cnn" ? Q.navy : Q.teal; }) }, text: vals.map(function (v) { return v >= 1e6 ? nf(v / 1e6, 1) + " M" : comma(v); }), textposition: "outside", cliponaxis: false,
          hovertemplate: items.map(function (t) { return t[0] + ": " + comma(t[1]) + " numbers" + (t[2] ? ", " + t[2] + " GFLOPs, ImageNet top-1 " + t[3] + "%" : "") + "<extra></extra>"; }) }];
      }, function (Q) { var nar = Kit.width(Kit.$("pm-plot")) < 470; return { xaxis: { type: "log", tickangle: 0, title: { text: "numbers learned (log scale)" }, range: mode === 0 ? [6.5, 8.9] : [2.5, 8.6], tickvals: mode === 0 ? (nar ? [1e7, 1e8] : [1e7, 3e7, 1e8, 3e8]) : (nar ? [1e3, 1e5, 1e7] : [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]), ticktext: mode === 0 ? (nar ? ["10 M", "100 M"] : ["10 M", "30 M", "100 M", "300 M"]) : (nar ? ["1 k", "100 k", "10 M"] : ["1 k", "10 k", "100 k", "1 M", "10 M", "100 M"]), minor: { showgrid: false } }, yaxis: { autorange: "reversed" }, margin: { l: 100, r: 56, t: 8, b: 56 } }; });
      Kit.$("pm-read").innerHTML = mode === 0
        ? "<span>Torchvision's own counts: ResNet-18 <b>" + nf(Pm.resnet18_gflops, 1) + "</b> GFLOPs, ViT-B/32 <b>" + nf(Pm.vit_b_32_gflops, 1) + "</b>, ViT-B/16 <b>" + nf(Pm.vit_b_16_gflops, 1) + "</b>. ViT-B/32 and ViT-B/16 have almost the same number of weights and a fourfold difference in work.</span>"
        : "<span>The 12 blocks contain <b>" + Kit.pct(Pm.blocks12 / Pm.vit_b_16, 1) + "</b> of the numbers. The patch matrix has <b>" + Kit.pct(Pm.patch_embed / Pm.vit_b_16, 1) + "</b>.</span>";
      Kit.$("pm-plot").setAttribute("aria-label", mode === 0 ? "Bar chart of the number of weights of five networks on a log scale" : "Bar chart of where the 86.6 million numbers of ViT-B/16 are");
    }
    draw();
  }

  // ---------- 7. discrete image tokens by vector quantisation ----------
  function vqFig() {
    var VQ = DATA.vq, books = {}, Ks = [4, 16, 64, 256], st = { ph: 0, ki: 3, r: 10, c: 12 }, DIM = 192, G = 28, cv = Kit.$("vq-orig"), cv2 = Kit.$("vq-rec"), cb = Kit.$("vq-book"), box = Kit.$("vq-box"), codes = null, stats = null;
    Ks.forEach(function (K) { var s = atob(VQ.books[String(K)]), a = new Uint8Array(s.length), i; for (i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); books[K] = a; });
    photoChips(Kit.$("vq-bar"), 0, function (i) { st.ph = i; run(); credit(); });
    Kit.chips(Kit.html("div", { class: "bar-row" }, Kit.$("vq-bar")), Ks.map(function (k) { return k + " codes"; }), 3, function (i) { st.ki = i; run(); });
    function patchVec(d, r, c) { var v = new Array(DIM), k = 0, i, j, o, ch; for (i = 0; i < 8; i++) for (j = 0; j < 8; j++) { o = ((r * 8 + i) * S + c * 8 + j) * 4; for (ch = 0; ch < 3; ch++) v[k++] = d[o + ch]; } return v; }
    function run() {
      var K = Ks[st.ki], book = books[K], d = IM[PH[st.ph].key].data.data, r, c, tot = 0, used = {}, n = 0;
      codes = new Array(G * G); var rec = new ImageData(S, S);
      for (r = 0; r < G; r++) for (c = 0; c < G; c++) {
        var v = patchVec(d, r, c), nr = nearest(v, book, K, DIM); codes[r * G + c] = nr; tot += nr.d2; used[nr.k] = 1;
        for (var i = 0; i < 8; i++) for (var j = 0; j < 8; j++) { var o = ((r * 8 + i) * S + c * 8 + j) * 4, q = nr.k * DIM + (i * 8 + j) * 3; rec.data[o] = book[q]; rec.data[o + 1] = book[q + 1]; rec.data[o + 2] = book[q + 2]; rec.data[o + 3] = 255; }
      }
      for (var u in used) n++;
      var mse = tot / (G * G * DIM); stats = { mse: mse, psnr: 10 * Math.log10(255 * 255 / mse), used: n, rec: rec, K: K };
      draw();
    }
    function draw() {
      if (!codes) return;
      var P = Kit.palette(), K = stats.K, W = Math.min(250, Math.floor((Kit.width(box) - 12) / 2)), cs = W / G, key = PH[st.ph].key;
      var c1 = fitCanvas(cv, W, W), c2 = fitCanvas(cv2, W, W), tmp = document.createElement("canvas"); tmp.width = S; tmp.height = S; tmp.getContext("2d").putImageData(stats.rec, 0, 0);
      c1.drawImage(IM[key].img, 0, 0, W, W); c2.drawImage(tmp, 0, 0, W, W);
      var sel = codes[st.r * G + st.c];
      [c1, c2].forEach(function (x) { x.lineWidth = 2.5; x.strokeStyle = P.coral; x.strokeRect(st.c * cs + 1, st.r * cs + 1, Math.max(cs - 2, 4), Math.max(cs - 2, 4)); });
      // the code book as tiles
      var cols = Math.ceil(Math.sqrt(K)), tile = Math.max(14, Math.min(40, Math.floor(Math.min(250, Kit.width(box)) / cols))), bw = cols * tile, bh = Math.ceil(K / cols) * tile, cx = fitCanvas(cb, bw, bh), book = books[K], k, i, j;
      var timg = cx.createImageData(8, 8), tc = document.createElement("canvas"); tc.width = 8; tc.height = 8;
      var count = new Array(K).fill(0); codes.forEach(function (q) { count[q.k]++; });
      for (k = 0; k < K; k++) {
        for (i = 0; i < 64; i++) { timg.data[i * 4] = book[k * DIM + i * 3]; timg.data[i * 4 + 1] = book[k * DIM + i * 3 + 1]; timg.data[i * 4 + 2] = book[k * DIM + i * 3 + 2]; timg.data[i * 4 + 3] = 255; }
        tc.getContext("2d").putImageData(timg, 0, 0); cx.imageSmoothingEnabled = false; cx.drawImage(tc, (k % cols) * tile, Math.floor(k / cols) * tile, tile, tile);
        if (count[k] === 0) { cx.fillStyle = "rgba(0,0,0,0.55)"; cx.fillRect((k % cols) * tile, Math.floor(k / cols) * tile, tile, tile); }
      }
      cx.lineWidth = 2.5; cx.strokeStyle = P.coral; cx.strokeRect((sel.k % cols) * tile + 1, Math.floor(sel.k / cols) * tile + 1, tile - 2, tile - 2);
      // inspector: the patch and its code, magnified
      var d = IM[key].data.data, pv = patchVec(d, st.r, st.c), pi = Kit.$("vq-p"), pc = Kit.$("vq-c"), a = pi.getContext("2d"), b = pc.getContext("2d");
      pi.width = 8; pi.height = 8; pc.width = 8; pc.height = 8;
      var id1 = a.createImageData(8, 8), id2 = b.createImageData(8, 8);
      for (i = 0; i < 64; i++) { for (j = 0; j < 3; j++) { id1.data[i * 4 + j] = pv[i * 3 + j]; id2.data[i * 4 + j] = book[sel.k * DIM + i * 3 + j]; } id1.data[i * 4 + 3] = 255; id2.data[i * 4 + 3] = 255; }
      a.putImageData(id1, 0, 0); b.putImageData(id2, 0, 0);
      var bits = Math.log2(K), tokens = G * G, rawBits = S * S * 3 * 8;
      Kit.$("vq-eq").innerHTML = ebox("c-coral", "#" + sel.k, "nearest code of the chosen patch") + eop("|") + ebox("c-navy", comma(sel.d2), "squared distance, summed over 192 numbers") + eop("|") + ebox("c-sand", nf(Math.sqrt(sel.d2 / DIM), 1), "typical gap per number, of 255");
      Kit.$("vq-eq2").innerHTML = ebox("c-teal", bits, "bits per token, log<sub>2</sub> " + K) + eop("&times;") + ebox("c-teal", comma(tokens), "tokens") + eop("=") + ebox("c-coral", comma(bits * tokens), "bits for the photo") + eop("|") + ebox("c-navy", comma(rawBits), "bits for the raw pixels") + eop("|") + ebox("c-sand", nf(rawBits / (bits * tokens), 0) + " &times;", "smaller");
      Kit.$("vq-read").innerHTML = "<span>Picture quality: PSNR <b>" + nf(stats.psnr, 1) + "</b> dB (higher is closer; mean squared error " + nf(stats.mse, 0) + " on a 0 to 255 scale). Codes used: <b>" + stats.used + "</b> of " + K + ". Dark tiles in the code book are unused for this photo.</span>";
      Kit.$("vq-status").textContent = "Patch at row " + st.r + ", column " + st.c + " of the 28 by 28 grid becomes token #" + sel.k + ". Click a patch in either picture to inspect another.";
      cv.setAttribute("aria-label", "The photo, with an 8 by 8 patch selected"); cv2.setAttribute("aria-label", "The photo rebuilt from " + K + " codes");
    }
    function pick(ev) { var W = Math.min(250, Math.floor((Kit.width(box) - 12) / 2)), c = cell(ev.currentTarget, ev, 0, W, G); if (c) { st.r = c.r; st.c = c.c; draw(); } }
    [cv, cv2].forEach(function (n) {
      n.setAttribute("tabindex", "0"); n.addEventListener("click", pick);
      n.addEventListener("keydown", function (ev) { var m = { ArrowLeft: [0, -1], ArrowRight: [0, 1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[ev.key]; if (!m) return; ev.preventDefault(); st.r = Math.max(0, Math.min(G - 1, st.r + m[0])); st.c = Math.max(0, Math.min(G - 1, st.c + m[1])); draw(); });
    });
    function credit() { Kit.$("vq-credit").textContent = creditFor(PH[st.ph].key) + " This photo was not used to make the code books."; }
    onResize(draw); run(); credit();
  }
})();
