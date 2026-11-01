// Chapter B9: explaining a network. Figures read data/b9.json and the small images in data/ (ResNet-18 and ViT-B/16 outputs computed offline).
(function () {
  "use strict";
  var K = Kit, $ = K.$, D = null;
  var NAMES = { h2: "hammerhead, Costa Rica", h1: "hammerhead, Jarvis Island", dol: "dolphin, Oman", milk: "market sharks, Kuala Lumpur" };
  var KEYS = ["h2", "h1", "dol", "milk"];

  function pct(v) { return (100 * v).toFixed(0) + "%"; }
  function setHTML(id, h) { var n = $(id); if (n) n.innerHTML = h; }
  function tile(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + "</b><i>" + small + "</i></span>"; }
  function rgb(hex) {
    hex = (hex || "#c2563a").trim();
    if (hex[0] === "#") { if (hex.length === 4) hex = "#" + hex[1] + hex[1] + hex[2] + hex[2] + hex[3] + hex[3]; return [parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16), parseInt(hex.slice(5, 7), 16)]; }
    var m = hex.match(/\d+/g) || [194, 86, 58]; return [+m[0], +m[1], +m[2]];
  }
  var imgCache = {};
  function loadImg(url) {
    if (!imgCache[url]) imgCache[url] = new Promise(function (ok, fail) { var im = new Image(); im.onload = function () { ok(im); }; im.onerror = fail; im.src = url; });
    return imgCache[url];
  }
  var mapCache = {};
  function loadMap(url) {   // grayscale PNG -> {w, h, v: Float32Array in 0..1}
    if (!mapCache[url]) mapCache[url] = loadImg(url).then(function (im) {
      var c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight; var x = c.getContext("2d"); x.drawImage(im, 0, 0);
      var d = x.getImageData(0, 0, c.width, c.height).data, v = new Float32Array(c.width * c.height);
      for (var i = 0; i < v.length; i++) v[i] = d[4 * i] / 255;
      return { w: c.width, h: c.height, v: v };
    });
    return mapCache[url];
  }
  // paint an overlay: values in 0..1 at gw x gh, colour col, onto ctx covering the canvas size (smooth upscale)
  function paintOverlay(ctx, W, H, gw, gh, alphaOf, col, offset) {
    var oc = document.createElement("canvas"); oc.width = gw; oc.height = gh; var ox = oc.getContext("2d"), id = ox.createImageData(gw, gh);
    for (var i = 0; i < gw * gh; i++) { var a = alphaOf(i); id.data[4 * i] = a[1] ? a[1][0] : col[0]; id.data[4 * i + 1] = a[1] ? a[1][1] : col[1]; id.data[4 * i + 2] = a[1] ? a[1][2] : col[2]; id.data[4 * i + 3] = Math.round(255 * a[0]); }
    ox.putImageData(id, 0, 0); ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = "high";
    if (offset) ctx.drawImage(oc, offset[0], offset[1], offset[2], offset[3]); else ctx.drawImage(oc, 0, 0, W, H);
  }
  function drawBase(ctx, im, W, H, dim) {
    ctx.clearRect(0, 0, W, H); ctx.drawImage(im, 0, 0, W, H);
    if (dim) { ctx.fillStyle = "rgba(0,0,0," + dim + ")"; ctx.fillRect(0, 0, W, H); }
  }
  function sampleGrid(g, x, y) {   // bilinear lookup in a grid {ny, nx, v}, x and y in grid units
    var xx = Math.max(0, Math.min(g.nx - 1, x)), yy = Math.max(0, Math.min(g.ny - 1, y)), x0 = Math.floor(xx), y0 = Math.floor(yy), x1 = Math.min(g.nx - 1, x0 + 1), y1 = Math.min(g.ny - 1, y0 + 1), fx = xx - x0, fy = yy - y0;
    function at(a, b) { return g.v[b * g.nx + a] / 1000; }
    return at(x0, y0) * (1 - fx) * (1 - fy) + at(x1, y0) * fx * (1 - fy) + at(x0, y1) * (1 - fx) * fy + at(x1, y1) * fx * fy;
  }
  function creditLine(key) { return D.photos[key].credit; }
  function stageWidth(node, maxw) { return Math.min(maxw || 640, Math.max(260, Math.floor(node.clientWidth || node.parentNode.clientWidth || 600))); }

  // ------------------------------------------------------------------ 1. occlusion
  function occlusionFig() {
    var st = { key: "h2", s: 72, show: false, x: 100, y: 60 };
    var bar = $("oc-bar");
    K.chips(bar, KEYS.map(function (k) { return NAMES[k]; }), 0, function (i) { st.key = KEYS[i]; load(); });
    K.html("div", { style: "flex-basis:100%;height:0" }, bar);
    K.html("div", { class: "label" }, bar, "patch");
    K.chips(bar, ["40 px", "72 px", "112 px"], 1, function (i) { st.s = [40, 72, 112][i]; clampPos(); draw(); });
    K.chips(bar, ["map hidden", "map shown"], 0, function (i) { st.show = i === 1; draw(); });
    K.button(bar, "Find the worst spot", function () { var g = D.photos[st.key].occ[st.s], b = 0; g.v.forEach(function (v, i) { if (v < g.v[b]) b = i; }); st.x = (b % g.nx) * 8; st.y = Math.floor(b / g.nx) * 8; draw(); });
    var stage = $("oc-stage"), img = K.html("img", { alt: "", draggable: "false" }, stage), heat = K.html("canvas", { class: "heat" }, stage), patch = K.html("div", { class: "patch", tabindex: "0", role: "slider", "aria-label": "Grey patch: drag it over the photo, or use the arrow keys" }, stage);
    var P0, scale = 1;
    function clampPos() { P0 = D.photos[st.key]; st.x = Math.max(0, Math.min(P0.w - st.s, st.x)); st.y = Math.max(0, Math.min(P0.h - st.s, st.y)); }
    function load() {
      P0 = D.photos[st.key]; img.src = "data/" + P0.file; img.alt = P0.credit; st.x = P0.w - st.s - 12; st.y = 8;
      loadImg("data/" + P0.file).then(function () { clampPos(); draw(); });
      setHTML("oc-credit", P0.credit);
    }
    function scoreAt() { var g = P0.occ[st.s]; return sampleGrid(g, st.x / 8, st.y / 8); }
    function draw() {
      clampPos();
      var wpx = stageWidth(stage, 640); scale = wpx / P0.w; stage.style.width = wpx + "px"; stage.style.height = Math.round(P0.h * scale) + "px";
      img.style.width = wpx + "px"; img.style.height = Math.round(P0.h * scale) + "px";
      patch.style.width = patch.style.height = Math.round(st.s * scale) + "px"; patch.style.left = Math.round(st.x * scale) + "px"; patch.style.top = Math.round(st.y * scale) + "px";
      heat.width = wpx; heat.height = Math.round(P0.h * scale); heat.style.display = st.show ? "block" : "none";
      if (st.show) {
        var g = P0.occ[st.s], base = P0.p_target, col = rgb(K.css("coral")), ctx = heat.getContext("2d"); ctx.clearRect(0, 0, heat.width, heat.height);
        paintOverlay(ctx, heat.width, heat.height, g.nx, g.ny, function (i) { var d = Math.max(0, (base - g.v[i] / 1000) / base); return [Math.min(0.85, d * 1.1), null]; }, col, [(st.s / 2 - 4) * scale, (st.s / 2 - 4) * scale, g.nx * 8 * scale, g.ny * 8 * scale]);
      }
      var sc = scoreAt(), base2 = P0.p_target;
      setHTML("oc-read", "<span>Score for <b>" + P0.target_name + "</b>: with the patch here <b>" + pct(sc) + "</b>, without it <b>" + pct(base2) + "</b></span><span>Patch at x = " + Math.round(st.x) + ", y = " + Math.round(st.y) + " (image is " + P0.w + " by " + P0.h + " px)</span>");
      var fill = $("oc-bar-fill"); if (fill) { fill.style.width = (100 * sc).toFixed(1) + "%"; $("oc-bar-base").style.left = (100 * base2).toFixed(1) + "%"; }
      var big = P0.occ[st.s], mn = Math.min.apply(null, big.v) / 1000;
      setHTML("oc-eqn", tile("c-navy", pct(base2), "score with the whole photo") + '<span class="op">&minus;</span>' + tile("c-coral", pct(sc), "score with the patch") + '<span class="op">=</span>' + tile("c-teal", (Math.round(100 * (base2 - sc)) || 0) + " points", "drop caused by this patch") + '<span class="op">|</span>' + tile("c-sand", pct(mn), "lowest score of any " + st.s + " px patch"));
    }
    var dragging = false;
    function move(ev) { var r = stage.getBoundingClientRect(); st.x = (ev.clientX - r.left) / scale - st.s / 2; st.y = (ev.clientY - r.top) / scale - st.s / 2; draw(); }
    stage.addEventListener("pointerdown", function (ev) { dragging = true; try { stage.setPointerCapture(ev.pointerId); } catch (e) {} move(ev); });
    stage.addEventListener("pointermove", function (ev) { if (dragging) move(ev); });
    stage.addEventListener("pointerup", function () { dragging = false; });
    patch.addEventListener("keydown", function (ev) { var d = { ArrowLeft: [-8, 0], ArrowRight: [8, 0], ArrowUp: [0, -8], ArrowDown: [0, 8] }[ev.key]; if (d) { ev.preventDefault(); st.x += d[0]; st.y += d[1]; draw(); } });
    window.addEventListener("resize", K.debounce(draw, 120));
    load();
  }

  // ------------------------------------------------------------------ map panels (saliency, Grad-CAM, integrated gradients, occlusion)
  var METHODS = {
    sal: { title: "Saliency", get: function (k) { return loadMap("data/maps/" + k + "_sal.png"); } },
    salr: { title: "Saliency, random weights", get: function (k) { return loadMap("data/maps/" + k + "_salr.png"); } },
    ig: { title: "Integrated gradients", get: function (k) { return loadMap("data/maps/" + k + "_ig.png"); }, signed: true }
  };
  function drawMapPanel(canvas, key, method, strength, layer) {
    var P0 = D.photos[key], W = canvas.width, H = Math.round(W * P0.h / P0.w); canvas.height = H;
    var ctx = canvas.getContext("2d"), pal = K.palette(), coral = rgb(pal.coral), teal = rgb(pal.teal), navy = rgb(pal.navy);
    return loadImg("data/" + P0.file).then(function (im) {
      drawBase(ctx, im, W, H, 0.35);
      if (method === "cam" || method === "camr") {
        var g = layer === 3 && method === "cam" ? P0.cam3 : (method === "cam" ? P0.cam4 : P0.cam4r);
        paintOverlay(ctx, W, H, g.nx, g.ny, function (i) { return [Math.min(0.9, Math.pow(g.v[i], 1.1) * strength), null]; }, coral);
        return;
      }
      if (method === "occ") {
        var o = P0.occ[72], base = P0.p_target, s = 72, sc = W / P0.w;
        paintOverlay(ctx, W, H, o.nx, o.ny, function (i) { return [Math.min(0.9, Math.max(0, (base - o.v[i] / 1000) / base) * strength * 1.6), null]; }, coral, [(s / 2 - 4) * sc, (s / 2 - 4) * sc, o.nx * 8 * sc, o.ny * 8 * sc]);
        return;
      }
      var M = METHODS[method];
      return M.get(key).then(function (m) {
        paintOverlay(ctx, W, H, m.w, m.h, function (i) {
          var v = m.v[i];
          if (M.signed) { var s2 = (v - 0.5) * 2; return s2 >= 0 ? [Math.min(0.9, Math.pow(s2, 0.7) * strength), coral] : [Math.min(0.9, Math.pow(-s2, 0.7) * strength * 0.8), navy]; }
          return [Math.min(0.9, Math.pow(v, 0.8) * strength), coral];
        }, coral);
      });
    });
  }
  function panelsFig(pre, methodList, opts) {
    opts = opts || {};
    var st = { key: "h2", strength: 1, layer: 4 }, box = $(pre + "-panels");
    var bar = $(pre + "-bar");
    K.chips(bar, KEYS.map(function (k) { return NAMES[k]; }), 0, function (i) { st.key = KEYS[i]; draw(); });
    if (opts.layerChips) K.chips(bar, ["Grad-CAM from the last block (7 by 12)", "Grad-CAM from an earlier block (14 by 24)"], 0, function (i) { st.layer = i === 0 ? 4 : 3; draw(); });
    K.slider($(pre + "-controls"), { label: "Strength of the overlay", min: 0.2, max: 1.6, step: 0.1, value: 1.0, digits: 1 }, function (v) { st.strength = v; draw(); });
    var cvs = {};
    methodList.forEach(function (m) {
      var f = K.html("figure", { class: "pn" }, box), c = K.html("canvas", { role: "img", "aria-label": m[1] }, f); K.html("figcaption", {}, f, m[1]); cvs[m[0]] = c;
    });
    function draw() {
      var w = Math.max(150, Math.floor((box.clientWidth || 600) / Math.min(methodList.length, window.innerWidth < 700 ? 2 : methodList.length)) - 12);
      methodList.forEach(function (m) { cvs[m[0]].width = Math.min(w, 420); drawMapPanel(cvs[m[0]], st.key, m[0], st.strength, st.layer); });
      var P0 = D.photos[st.key];
      setHTML(pre + "-read", "<span>Class explained: <b>" + P0.target_name + "</b>, score " + pct(P0.p_target) + " (the network's top choice is <b>" + P0.top5[0][0] + "</b>)</span><span class=\"note\">" + P0.credit + "</span>");
      if (opts.after) opts.after(st.key);
    }
    window.addEventListener("resize", K.debounce(draw, 150));
    draw();
  }

  // ------------------------------------------------------------------ 5. deletion test
  function deletionFig() {
    var Dl = D.deletion, st = { key: "h2", method: "integrated gradients", q: 4 }, order = ["random order", "saliency", "Grad-CAM", "integrated gradients", "occlusion"];
    var bar = $("dl-bar");
    K.chips(bar, ["h2", "h1"].map(function (k) { return NAMES[k]; }), 0, function (i) { st.key = ["h2", "h1"][i]; draw(); });
    var sel = K.select($("dl-controls"), "Remove the blocks that this map ranks highest", order.slice(1).map(function (m) { return { value: m, label: m }; }), st.method, function (v) { st.method = v; draw(); });
    var sl = K.slider($("dl-controls"), { label: "Share of the photo removed, in blocks of 8 by 8 pixels", min: 0, max: Dl.h2.q.length - 1, step: 1, value: 4, show: function (i) { return Dl.h2.q[i] + "%"; } }, function (v) { st.q = v; draw(); });
    var cv = $("dl-canvas");
    function draw() {
      var R = Dl[st.key], P0 = D.photos[st.key], q = R.q[st.q], W = Math.min(520, Math.max(260, ($("dl-stage").clientWidth || 500)));
      cv.width = W; cv.height = Math.round(W * P0.h / P0.w);
      loadImg("data/" + P0.file).then(function (im) {
        var ctx = cv.getContext("2d"); drawBase(ctx, im, cv.width, cv.height, 0);
        var ny = R.blocks[0], nx = R.blocks[1], str = R.order[st.method], bw = cv.width / nx, bh = cv.height / ny;
        ctx.fillStyle = "rgb(124,116,104)";
        for (var b = 0; b < ny * nx; b++) { var r = (str.charCodeAt(b) - 33) / 93; if (r < q / 100) ctx.fillRect(Math.floor((b % nx) * bw), Math.floor(Math.floor(b / nx) * bh), Math.ceil(bw), Math.ceil(bh)); }
      });
      K.plot("dl-plot", function (P) {
        var cols = { "random order": P.muted, "saliency": P.sand, "Grad-CAM": P.navy, "integrated gradients": P.teal, "occlusion": P.coral };
        var tr = order.map(function (m) { return { type: "scatter", mode: "lines+markers", name: m, x: R.q, y: R.curves[m], line: { color: cols[m], width: m === st.method ? 3.4 : 1.8, dash: m === "random order" ? "dot" : "solid" }, marker: { size: m === st.method ? 7 : 4 }, hoverinfo: "skip", showlegend: true }; });
        tr.push({ type: "scatter", mode: "markers", x: [q], y: [R.curves[st.method][st.q]], marker: { color: P.ink, size: 13, symbol: "circle-open", line: { width: 2 } }, hoverinfo: "skip", showlegend: false });
        return tr;
      }, { showlegend: true, legend: { orientation: "h", y: 1.02, yanchor: "bottom", x: 0, font: { size: 12 } }, margin: { l: 60, r: 10, t: K.width($("dl-plot")) < 520 ? 98 : 54, b: 46 }, xaxis: { title: { text: "share of the photo removed (%)" }, range: [0, 75] }, yaxis: { title: { text: "score" }, range: [0, 1], tickformat: ".0%" } });
      setHTML("dl-read", "<span>Removing <b>" + q + "%</b> by <b>" + st.method + "</b>: score <b>" + pct(R.curves[st.method][st.q]) + "</b> (from <b>" + pct(R.curves[st.method][0]) + "</b>). Removing the same share in random order: <b>" + pct(R.curves["random order"][st.q]) + "</b></span><span class=\"note\">" + P0.credit + "</span>");
    }
    window.addEventListener("resize", K.debounce(draw, 150));
    draw();
  }

  // ------------------------------------------------------------------ 6. cue lab
  var CUES = {
    hammerhead: ["h_orig", "h_erased", "h_water", "h_sand", "h_card"],
    dolphin: ["d_orig", "d_covered", "d_water", "d_grey", "d_blue", "d_sand"]
  };
  var CUELAB = { h_orig: "photo as it is", h_erased: "shark painted out", h_water: "shark on green water", h_sand: "shark on dark sand", h_card: "shark on a white card", d_orig: "photo as it is", d_covered: "dolphin covered with water", d_water: "a strip of water only", d_grey: "dolphin on grey", d_blue: "dolphin on deep blue", d_sand: "dolphin on dark sand" };
  function cueFig() {
    var st = { subj: "dolphin", cue: "d_orig" };
    var bar = $("cu-bar"), chipsBox = $("cu-chips"), chipsCtl = null;
    K.chips(bar, ["the dolphin photo", "the hammerhead photo"], 0, function (i) { st.subj = i === 0 ? "dolphin" : "hammerhead"; st.cue = CUES[st.subj][0]; build(); draw(); });
    function build() { chipsBox.innerHTML = ""; chipsCtl = K.chips(chipsBox, CUES[st.subj].map(function (c) { return CUELAB[c]; }), 0, function (i) { st.cue = CUES[st.subj][i]; draw(); }); }
    function draw() {
      if (!D.cue[st.cue]) { st.cue = CUES[st.subj][0]; }
      var C = D.cue[st.cue], img = $("cu-img"); img.src = "data/" + C.file; img.alt = C.caption;
      var rows = C.top5, h = "";
      rows.forEach(function (r) { var shark = /shark|hammerhead/.test(r[0]); h += '<div class="cu-row' + (shark ? " shark" : "") + '"><span>' + r[0] + '</span><span class="cu-track"><span style="width:' + Math.round(r[1] * 100) + '%"></span></span><b>' + pct(r[1]) + "</b></div>"; });
      $("cu-bars").innerHTML = h;
      var t = '<table class="sv-table"><thead><tr><th>edit</th><th>great white</th><th>tiger</th><th>hammerhead</th><th>top choice</th></tr></thead><tbody>';
      CUES[st.subj].forEach(function (c) { var X = D.cue[c]; t += '<tr data-c="' + c + '"' + (c === st.cue ? ' class="on"' : "") + "><td>" + CUELAB[c] + "</td><td>" + pct(X.gw) + "</td><td>" + pct(X.tiger) + "</td><td>" + pct(X.hammer) + "</td><td>" + X.top5[0][0] + "</td></tr>"; });
      t += "</tbody></table>"; $("cu-table").innerHTML = t;
      [].forEach.call($("cu-table").querySelectorAll("tr[data-c]"), function (tr) { tr.style.cursor = "pointer"; tr.addEventListener("click", function () { st.cue = tr.getAttribute("data-c"); var i = CUES[st.subj].indexOf(st.cue); [].forEach.call(chipsBox.querySelectorAll("button"), function (b, j) { b.setAttribute("aria-pressed", String(j === i)); }); draw(); }); });
      var srcCredit = st.subj === "dolphin" ? D.photos.dol.credit : D.photos.h2.credit, sandCredit = (st.cue === "d_sand" || st.cue === "h_sand") ? " The dark sand is a crop of a photo of an olive ridley turtle by Brad Flickinger, CC BY 2.0, via Wikimedia Commons." : "";
      setHTML("cu-read", "<span>" + C.caption + ". Top choice: <b>" + C.top5[0][0] + " " + pct(C.top5[0][1]) + "</b></span><span class=\"note\">" + (/_orig$/.test(st.cue) ? "" : "Edited by us from this photo. ") + srcCredit + sandCredit + "</span>");
    }
    build(); draw();
  }

  // ------------------------------------------------------------------ 7. attention maps of a vision transformer
  function attnFig() {
    var V = D.vit, st = { key: "h2", head: -1, uniform: false };
    var vk = { h2: "h2", dol: "dol" }, bar = $("at-bar");
    K.chips(bar, ["hammerhead, Costa Rica", "dolphin, Oman"], 0, function (i) { st.key = i === 0 ? "h2" : "dol"; draw(); });
    K.select($("at-controls"), "Which head of the last layer", [{ value: "-1", label: "mean of the 12 heads" }].concat([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11].map(function (i) { return { value: String(i), label: "head " + (i + 1) }; })), "-1", function (v) { st.head = Number(v); draw(); });
    K.chips(bar, ["attention as computed", "attention replaced by equal weights"], 0, function (i) { st.uniform = i === 1; draw(); });
    var cv = $("at-canvas");
    function draw() {
      var R = V[st.key], P0 = D.photos[st.key], vals = st.head < 0 ? R.mean : R.heads[st.head], mx = Math.max.apply(null, vals), W = Math.min(520, Math.max(260, ($("at-stage").clientWidth || 500)));
      cv.width = W; cv.height = Math.round(W * P0.h / P0.w);
      loadImg("data/" + P0.file).then(function (im) {
        var ctx = cv.getContext("2d"), col = rgb(K.css("coral")); drawBase(ctx, im, cv.width, cv.height, 0.3);
        paintOverlay(ctx, cv.width, cv.height, 14, 14, function (i) { return [st.uniform ? 0.08 : Math.min(0.9, Math.pow(vals[i] / mx, 0.9)), null]; }, col);
      });
      var rho = st.head < 0 ? R.rho_mean_vs_saliency : R.rho_head_vs_saliency[st.head], pu = st.head < 0 ? R.p_uniform : R.p_head_uniform[st.head];
      setHTML("at-read", "<span>Class: <b>" + R.cls + "</b>, score <b>" + pct(R.p) + "</b></span><span>With " + (st.head < 0 ? "equal weights in all 12 heads" : "equal weights in head " + (st.head + 1) + " only") + " the score is <b>" + pct(pu) + "</b>" + (st.head < 0 ? " (and <b>" + pct(R.p_shuffled) + "</b> with the weights shuffled across the photo)" : "") + "</span><span>Rank agreement of this map with the gradient saliency: <b>" + K.fmt(rho, 2) + "</b></span><span class=\"note\">" + P0.credit + "</span>");
    }
    window.addEventListener("resize", K.debounce(draw, 150));
    draw();
  }

  // ------------------------------------------------------------------ boot
  function boot() {
    K.data("data/b9.json").then(function (d) {
      D = d;
      var jobs = [
        ["oc-stage", occlusionFig],
        ["sa-panels", function () { panelsFig("sa", [["sal", "trained ResNet-18"], ["salr", "same network, random weights"]]); }],
        ["cm-panels", function () { panelsFig("cm", [["sal", "saliency (gradient)"], ["cam", "Grad-CAM"], ["ig", "integrated gradients"], ["occ", "occlusion map"]], { layerChips: true }); }],
        ["dl-stage", deletionFig], ["cu-bars", cueFig], ["at-stage", attnFig]
      ];
      jobs.forEach(function (j) { if ($(j[0])) { try { j[1](); } catch (e) { console.error(j[0], e); } } });
    }).catch(function (e) { console.error(e); });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
})();
