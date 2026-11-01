// C2 Convolutional networks.
//   archFig    a layer builder: output sizes, numbers learned and receptive field per layer, with the working for one layer
//   readFig    SIGNATURE: draw a digit; a small CNN (weights trained offline in PyTorch, data/cnn.json) runs in plain JS and shows
//              the feature maps of both conv layers and the softmax probabilities
//   rfFig      receptive-field highlighter: click a cell in a deeper map, see the input pixels it can depend on and the ones that matter
//   paramFig   dense layer, locally connected layer and conv layer on the same picture: parameter counts (Plotly, log axis)
//   limitFig   move or turn a digit and watch the network; measured accuracy curves for two networks
// Pure functions are exported with module.exports for the node test (outSize, forward, rfOf, prepare helpers that need no DOM).
(function () {
  "use strict";

  // ---------- pure maths ----------
  function outSize(n, k, s, p) { var t = n + 2 * p - k; return t < 0 ? 0 : Math.floor(t / s) + 1; }
  var LAYERS = [
    { id: "conv1", name: "Conv 1", short: "conv 1", k: 3, s: 1, kind: "conv" },
    { id: "pool1", name: "Pool 1", short: "pool 1", k: 2, s: 2, kind: "pool" },
    { id: "conv2", name: "Conv 2", short: "conv 2", k: 3, s: 1, kind: "conv" },
    { id: "pool2", name: "Pool 2", short: "pool 2", k: 2, s: 2, kind: "pool" }
  ];
  // receptive field r and jump j after each layer, from r_out = r_in + (k - 1) j_in and j_out = s j_in
  function rfTable() {
    var r = 1, j = 1, rows = [{ name: "Input", r: 1, j: 1, k: 0, s: 0, rin: 1, jin: 1 }];
    LAYERS.forEach(function (L) { var rin = r, jin = j; r = rin + (L.k - 1) * jin; j = jin * L.s; rows.push({ name: L.name, r: r, j: j, k: L.k, s: L.s, rin: rin, jin: jin }); });
    return rows;
  }

  var NET = null;
  function loadNet(J) {
    function flat(a) { var o = []; (function rec(x) { if (Array.isArray(x)) x.forEach(rec); else o.push(x); })(a); return Float32Array.from(o); }
    NET = { c1w: flat(J.conv1_w), c1b: flat(J.conv1_b), c2w: flat(J.conv2_w), c2b: flat(J.conv2_b), fcw: flat(J.fc_w), fcb: flat(J.fc_b) };
    return NET;
  }
  // x: Float32Array(784) with values 0..1. upto: "conv1" | "pool1" | "conv2" | "pool2" | undefined (all, with logits)
  function forward(x, upto) {
    var N = NET, f, c, i, j, u, v, acc, idx;
    var a1 = new Float32Array(6 * 676);
    for (f = 0; f < 6; f++) for (i = 0; i < 26; i++) for (j = 0; j < 26; j++) {
      acc = N.c1b[f];
      for (u = 0; u < 3; u++) for (v = 0; v < 3; v++) acc += N.c1w[f * 9 + u * 3 + v] * x[(i + u) * 28 + j + v];
      a1[f * 676 + i * 26 + j] = acc > 0 ? acc : 0;
    }
    var out = { a1: a1 };
    if (upto === "conv1") return out;
    var p1 = new Float32Array(6 * 169);
    for (f = 0; f < 6; f++) for (i = 0; i < 13; i++) for (j = 0; j < 13; j++) {
      idx = f * 676 + 2 * i * 26 + 2 * j;
      p1[f * 169 + i * 13 + j] = Math.max(a1[idx], a1[idx + 1], a1[idx + 26], a1[idx + 27]);
    }
    out.p1 = p1;
    if (upto === "pool1") return out;
    var a2 = new Float32Array(12 * 121);
    for (f = 0; f < 12; f++) for (i = 0; i < 11; i++) for (j = 0; j < 11; j++) {
      acc = N.c2b[f];
      for (c = 0; c < 6; c++) for (u = 0; u < 3; u++) for (v = 0; v < 3; v++) acc += N.c2w[((f * 6 + c) * 3 + u) * 3 + v] * p1[c * 169 + (i + u) * 13 + j + v];
      a2[f * 121 + i * 11 + j] = acc > 0 ? acc : 0;
    }
    out.a2 = a2;
    if (upto === "conv2") return out;
    var p2 = new Float32Array(12 * 25);
    for (f = 0; f < 12; f++) for (i = 0; i < 5; i++) for (j = 0; j < 5; j++) {
      idx = f * 121 + 2 * i * 11 + 2 * j;
      p2[f * 25 + i * 5 + j] = Math.max(a2[idx], a2[idx + 1], a2[idx + 11], a2[idx + 12]);
    }
    out.p2 = p2;
    if (upto === "pool2") return out;
    var z = new Float32Array(10), mx = -Infinity, sum = 0, pr = new Float32Array(10);
    for (var k = 0; k < 10; k++) { acc = N.fcb[k]; for (i = 0; i < 300; i++) acc += N.fcw[k * 300 + i] * p2[i]; z[k] = acc; if (acc > mx) mx = acc; }
    for (k = 0; k < 10; k++) { pr[k] = Math.exp(z[k] - mx); sum += pr[k]; }
    for (k = 0; k < 10; k++) pr[k] /= sum;
    out.z = z; out.p = pr; out.expSum = sum; out.zmax = mx;
    return out;
  }
  var LAYER_SIZE = { conv1: 26, pool1: 13, conv2: 11, pool2: 5 }, LAYER_CH = { conv1: 6, pool1: 6, conv2: 12, pool2: 12 }, LAYER_KEY = { conv1: "a1", pool1: "p1", conv2: "a2", pool2: "p2" };
  // value of one cell, and its slope with respect to every input pixel inside the receptive field (central differences)
  function cellValue(x, layer, ch, i, j) { var o = forward(x, layer), n = LAYER_SIZE[layer]; return o[LAYER_KEY[layer]][ch * n * n + i * n + j]; }
  function influence(x, layer, ch, i, j, box, h) {
    h = h || 0.1; var g = new Float32Array(784), r, c, xp = Float32Array.from(x), v1, v2;
    for (r = box.y0; r <= box.y1; r++) for (c = box.x0; c <= box.x1; c++) {
      var old = x[r * 28 + c]; xp[r * 28 + c] = old + h; v1 = cellValue(xp, layer, ch, i, j); xp[r * 28 + c] = old - h; v2 = cellValue(xp, layer, ch, i, j); xp[r * 28 + c] = old;
      g[r * 28 + c] = (v1 - v2) / (2 * h);
    }
    return g;
  }
  function rfBox(layer, i, j) {
    var T = rfTable(), k = LAYERS.map(function (L) { return L.id; }).indexOf(layer) + 1, row = T[k], start = row.j; // cells of a layer are j input pixels apart
    return { y0: i * start, x0: j * start, y1: i * start + row.r - 1, x1: j * start + row.r - 1, r: row.r, j: row.j, start: start };
  }
  // dense, locally connected and conv layers that write the same maps from the same picture
  function paramCounts(n, cin, F, k, same) {
    var m = outSize(n, k, 1, same ? Math.floor((k - 1) / 2) : 0), outs = m * m * F;
    return { m: m, outputs: outs, inputs: n * n * cin, dense: (n * n * cin + 1) * outs, local: (k * k * cin + 1) * outs, conv: (k * k * cin + 1) * F };
  }
  // the transformations of the limits figure (the same convention as the offline measurement: roll right by dx; rotate with affine_grid and bilinear sampling)
  function shiftDigit(x, dx) { var o = new Float32Array(784), i, j; for (i = 0; i < 28; i++) for (j = 0; j < 28; j++) { var jj = ((j - dx) % 28 + 28) % 28; o[i * 28 + j] = x[i * 28 + jj]; } return o; }
  function rotateDigit(x, deg) {
    var a = deg * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a), o = new Float32Array(784), i, j;
    for (i = 0; i < 28; i++) for (j = 0; j < 28; j++) {
      var xo = (2 * j + 1) / 28 - 1, yo = (2 * i + 1) / 28 - 1, xs = ca * xo - sa * yo, ys = sa * xo + ca * yo;
      var fx = ((xs + 1) * 28 - 1) / 2, fy = ((ys + 1) * 28 - 1) / 2, x0 = Math.floor(fx), y0 = Math.floor(fy), tx = fx - x0, ty = fy - y0, v = 0;
      [[0, 0, (1 - tx) * (1 - ty)], [1, 0, tx * (1 - ty)], [0, 1, (1 - tx) * ty], [1, 1, tx * ty]].forEach(function (q) {
        var xx = x0 + q[0], yy = y0 + q[1]; if (xx >= 0 && xx < 28 && yy >= 0 && yy < 28) v += q[2] * x[yy * 28 + xx];
      });
      o[i * 28 + j] = v;
    }
    return o;
  }

  if (typeof module !== "undefined") { module.exports = { outSize: outSize, loadNet: loadNet, forward: forward, rfTable: rfTable, rfBox: rfBox, paramCounts: paramCounts, influence: influence, cellValue: cellValue, shiftDigit: shiftDigit, rotateDigit: rotateDigit }; return; }

  // ---------- page helpers ----------
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function mixRGB(a, b, t) { var A = hexRGB(a), B = hexRGB(b); return [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }); }
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function decode(b64) { var s = atob(b64), a = new Float32Array(784), i; for (i = 0; i < 784; i++) a[i] = s.charCodeAt(i) / 255; return a; }
  function setPressed(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function label(parent, text, cls) { return Kit.html("label", cls ? { class: cls } : {}, parent, text); }
  function big(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(n >= 1e13 ? 0 : 1) + " trillion";
    if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + " billion";
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 1 : 2) + " million";
    return Math.round(n).toLocaleString("en-US");
  }
  function drawMap(cv, data, off, n, vmax, P, mode) {
    // mode "teal": values >= 0 mapped from paper to teal. canvas has n by n pixels, scaled up by CSS.
    if (cv.width !== n || cv.height !== n) { cv.width = n; cv.height = n; }
    var ctx = cv.getContext("2d"), im = ctx.createImageData(n, n), i, c;
    var bg = hexRGB(P.paper), fg = mode === "ink" ? hexRGB(P.ink) : hexRGB(P.teal);
    for (i = 0; i < n * n; i++) {
      var t = vmax > 0 ? Math.min(1, Math.max(0, data[off + i] / vmax)) : 0;
      for (c = 0; c < 3; c++) im.data[4 * i + c] = Math.round(bg[c] + (fg[c] - bg[c]) * t);
      im.data[4 * i + 3] = 255;
    }
    ctx.putImageData(im, 0, 0);
  }

  var J = null, EX = [];
  var S = { x: new Float32Array(784), listeners: [], set: function (x, src) { S.x = x; S.listeners.forEach(function (f) { f(src); }); } };
  document.addEventListener("DOMContentLoaded", function () {
    Kit.data("data/cnn.json").then(function (j) {
      J = j; loadNet(j); EX = j.examples;
      readFig(); rfFig(); limitFig();
    }).catch(function (e) { console.error(e); });
    archFig(); paramFig();
  });

  // ---------- 1. the layer builder ----------
  function archFig() {
    var INPUTS = [{ n: 28, c: 1, name: "28 by 28 digit (1 channel)" }, { n: 64, c: 3, name: "64 by 64 colour crop (3 channels)" }, { n: 224, c: 3, name: "224 by 224 photo (3 channels)" }];
    var inp = 0, layers = [], sel = -1;
    function preset(kind) {
      if (kind === 0) layers = [{ t: "conv", k: 3, s: 1, same: false, f: 6 }, { t: "pool", k: 2, s: 2 }, { t: "conv", k: 3, s: 1, same: false, f: 12 }, { t: "pool", k: 2, s: 2 }, { t: "dense", f: 10 }];
      else layers = [{ t: "conv", k: 3, s: 1, same: true, f: 16 }, { t: "pool", k: 2, s: 2 }, { t: "conv", k: 3, s: 1, same: true, f: 32 }, { t: "pool", k: 2, s: 2 }, { t: "conv", k: 3, s: 1, same: true, f: 64 }, { t: "pool", k: 2, s: 2 }, { t: "conv", k: 3, s: 1, same: true, f: 128 }, { t: "pool", k: 2, s: 2 }, { t: "dense", f: 10 }];
      sel = layers.length - 2; draw();
    }
    var bar = Kit.$("ar-bar"), r1 = Kit.html("div", { class: "bar-row" }, bar), r2 = Kit.html("div", { class: "bar-row" }, bar);
    var ichips = Kit.chips(r1, INPUTS.map(function (o) { return o.name; }), 0, function (i) { inp = i; draw(); });
    var pchips = Kit.chips(r1, ["Our digit reader", "A deeper network"], 0, function (i) { if (i === 1 && inp === 0) { inp = 1; setPressed(ichips, 1); } setPressed(pchips, i); preset(i); });
    Kit.button(r2, "Add conv 3 by 3", function () { insert({ t: "conv", k: 3, s: 1, same: true, f: 16 }); });
    Kit.button(r2, "Add max pool 2 by 2", function () { insert({ t: "pool", k: 2, s: 2 }); });
    Kit.button(r2, "Add dense layer", function () { insert({ t: "dense", f: 10 }); });
    Kit.button(r2, "Remove last", function () { layers.pop(); sel = Math.min(sel, layers.length - 1); draw(); });
    function insert(L) { var at = layers.length && layers[layers.length - 1].t === "dense" && L.t !== "dense" ? layers.length - 1 : layers.length; layers.splice(at, 0, L); sel = at; draw(); }
    // returns rows with sizes
    function compute() {
      var n = INPUTS[inp].n, c = INPUTS[inp].c, r = 1, j = 1, flat = false, rows = [], total = 0, ok = true;
      rows.push({ name: "Input", out: n + " &times; " + n + " &times; " + c, params: 0, r: 1, j: 1, ok: true, note: "" });
      layers.forEach(function (L, idx) {
        var row = { L: L, idx: idx, ok: ok, params: 0, note: "" };
        if (!ok) { row.name = L.t; row.out = "&ndash;"; row.r = ""; row.j = ""; rows.push(row); return; }
        if (L.t === "conv" || L.t === "pool") {
          if (flat) { row.ok = false; ok = false; row.out = "after a dense layer only dense layers fit"; row.name = L.t; rows.push(row); return; }
          var p = L.t === "conv" && L.same ? Math.floor((L.k - 1) / 2) : 0, m = outSize(n, L.k, L.s, p);
          if (m <= 0) { row.ok = false; ok = false; row.out = "too small"; row.name = L.t === "conv" ? "conv " + L.k + "&times;" + L.k : "max pool " + L.k + "&times;" + L.k; rows.push(row); return; }
          row.q = { n: n, k: L.k, s: L.s, p: p, m: m, cin: c, rin: r, jin: j };
          if (L.t === "conv") { row.params = (L.k * L.k * c + 1) * L.f; row.name = "conv " + L.k + "&times;" + L.k + (L.same ? ", same" : ""); c = L.f; }
          else row.name = "max pool " + L.k + "&times;" + L.k;
          r = r + (L.k - 1) * j; j = j * L.s; n = m;
          row.out = n + " &times; " + n + " &times; " + c; row.r = r; row.j = j;
        } else {
          var fin = flat ? c : n * n * c; row.q = { fin: fin, f: L.f };
          row.params = (fin + 1) * L.f; row.name = "dense " + L.f; flat = true; n = 1; c = L.f; row.out = "" + L.f; row.r = "whole input"; row.j = "&ndash;";
        }
        total += row.params; rows.push(row);
      });
      return { rows: rows, total: total, ok: ok };
    }
    function draw() {
      var res = compute(), box = Kit.$("ar-table"), html = '<table class="arch"><thead><tr><th></th><th>Layer</th><th>Output size</th><th>Numbers learned</th><th>Receptive field</th><th></th></tr></thead><tbody>';
      res.rows.forEach(function (row, i) {
        var L = row.L;
        html += '<tr class="' + (i - 1 === sel ? "sel " : "") + (row.ok === false ? "bad" : "") + '" data-i="' + (i - 1) + '"><td>' + (i === 0 ? "" : i) + "</td><td>" + row.name;
        if (L && L.t === "conv" && row.ok !== false) html += ' <input type="number" class="nf" min="1" max="512" value="' + L.f + '" data-i="' + (i - 1) + '" aria-label="number of filters"> filters';
        if (L && L.t === "dense" && row.ok !== false) html += ' <input type="number" class="nf" min="1" max="4096" value="' + L.f + '" data-i="' + (i - 1) + '" aria-label="number of units"> units';
        html += "</td><td>" + (row.out || "") + "</td><td>" + (row.params ? row.params.toLocaleString("en-US") : (i === 0 ? "" : "0")) + "</td><td>" + (row.r !== undefined && row.r !== "" ? (typeof row.r === "number" ? row.r + " &times; " + row.r + " pixels" : row.r) : "") + "</td><td>" + (L ? '<button type="button" class="x" data-rm="' + (i - 1) + '" aria-label="remove this layer">&times;</button>' : "") + "</td></tr>";
      });
      box.innerHTML = html + "</tbody></table>";
      box.querySelectorAll("tr[data-i]").forEach(function (tr) { tr.addEventListener("click", function (e) { if (e.target.tagName === "INPUT" || e.target.tagName === "BUTTON") return; sel = +tr.getAttribute("data-i"); draw(); }); });
      box.querySelectorAll("input.nf").forEach(function (e) { e.addEventListener("change", function () { layers[+e.getAttribute("data-i")].f = Math.max(1, Math.min(4096, Math.round(+e.value) || 1)); draw(); }); });
      box.querySelectorAll("button.x").forEach(function (b) { b.addEventListener("click", function () { var i = +b.getAttribute("data-rm"); layers.splice(i, 1); sel = Math.min(sel, layers.length - 1); draw(); }); });
      var n0 = INPUTS[inp].n * INPUTS[inp].c * INPUTS[inp].n, plain = (INPUTS[inp].n * INPUTS[inp].n * INPUTS[inp].c + 1) * 128 + 129 * 10;
      Kit.$("ar-total").innerHTML = ebox("c-coral", res.total.toLocaleString("en-US"), "numbers learned in all layers") + eop("|") + ebox("c-navy", plain.toLocaleString("en-US"), "a plain network with one hidden layer of 128 units and 10 outputs on the same input") + (res.ok ? "" : eop("a layer does not fit: remove it or change the input"));
      working(res);
    }
    function working(res) {
      var box = Kit.$("ar-eqn"); var row = res.rows[sel + 1];
      if (!row || !row.L || row.ok === false) { box.innerHTML = '<span class="hintline">Click a layer to see the working for its size, its numbers and its receptive field.</span>'; return; }
      var L = row.L, q = row.q, h = "";
      if (L.t === "dense") {
        h = eop("inputs") + ebox("c-navy", q.fin.toLocaleString("en-US"), "all numbers of the last maps, flattened") + eop("+ 1 bias, &times;") + ebox("c-teal", String(q.f), "units") + eop("=") + ebox("c-coral", row.params.toLocaleString("en-US"), "numbers learned");
      } else {
        h = eop("&lfloor;(") + ebox("c-navy", String(q.n), "n") + eop("+ 2 &times;") + ebox("c-sand", String(q.p), "p") + eop("&minus;") + ebox("c-teal", String(q.k), "k") + eop(") &divide;") + ebox("c-coral", String(q.s), "s") + eop("&rfloor; + 1 =") + ebox("c-coral", String(q.m), "output side");
        if (L.t === "conv") h += eop("|") + ebox("c-navy", "(" + q.k + "&times;" + q.k + "&times;" + q.cin + " + 1)", "numbers in one filter") + eop("&times;") + ebox("c-teal", String(L.f), "filters") + eop("=") + ebox("c-coral", row.params.toLocaleString("en-US"), "numbers learned");
        h += eop("|") + ebox("c-navy", q.rin + " + (" + q.k + " &minus; 1) &times; " + q.jin, "r in + (k &minus; 1) &times; j in") + eop("=") + ebox("c-coral", String(row.r), "receptive field") + eop(",") + ebox("c-sand", q.jin + " &times; " + q.s + " = " + row.j, "jump");
      }
      box.innerHTML = h;
    }
    preset(0);
  }

  // ---------- 2. draw a digit (signature) ----------
  function readFig() {
    var PAD = 280, pad = Kit.$("rd-pad"), pctx = pad.getContext("2d"), strokes = [], drawing = false, example = null, centre = true;
    pad.width = PAD; pad.height = PAD;
    var off = document.createElement("canvas"); off.width = PAD; off.height = PAD; var octx = off.getContext("2d");
    var c28 = document.createElement("canvas"); c28.width = 28; c28.height = 28; var x28 = c28.getContext("2d", { willReadFrequently: true });
    var t28 = document.createElement("canvas"); t28.width = 28; t28.height = 28; var xt = t28.getContext("2d", { willReadFrequently: true });
    var bar = Kit.$("rd-bar"), b1 = Kit.html("div", { class: "bar-row" }, bar), b2 = Kit.html("div", { class: "bar-row" }, bar), b3 = Kit.html("div", { class: "bar-row" }, bar);
    var mapMode = 0, state = null, picked = -1;
    Kit.button(b1, "Clear", function () { strokes = []; example = null; update(true); });
    var cl = label(b1, "", "check"), cc = Kit.html("input", { type: "checkbox", checked: "checked" }, cl); cl.appendChild(document.createTextNode(" Centre and scale my drawing like MNIST"));
    cc.addEventListener("change", function () { centre = cc.checked; update(false); });
    Kit.chips(b2, ["after ReLU", "after pooling"], 0, function (i) { mapMode = i; render(); });
    var labels = []; for (var d = 0; d < 10; d++) labels.push("test " + d);
    Kit.html("span", { class: "label" }, b3, "Load a test digit:");
    var exChips = Kit.chips(b3, labels, -1, function (i) { loadExample(i); });
    var misBtns = (J.misread || []).map(function (m, i) { return Kit.button(b3, "misread " + m.label + " as " + m.pred, function () { loadMis(i); }); });
    function loadExample(d) { var e = EX.filter(function (q) { return q.label === d; })[0]; example = decode(e.b64); strokes = []; setPressed(exChips, d); update(true); }
    function loadMis(i) { example = decode(J.misread[i].b64); strokes = []; setPressed(exChips, -1); update(true); }

    function inkFill(P) { return P.ink; }
    function redrawPad() {
      var P = Kit.palette(); pctx.fillStyle = P.paper; pctx.fillRect(0, 0, PAD, PAD);
      pctx.strokeStyle = P.rule; pctx.lineWidth = 1; pctx.strokeRect(0.5, 0.5, PAD - 1, PAD - 1);
      if (example) {
        var cell = PAD / 28, i, j; for (i = 0; i < 28; i++) for (j = 0; j < 28; j++) { var v = example[i * 28 + j]; if (v > 0) { pctx.fillStyle = "rgb(" + mixRGB(P.paper, P.ink, v).join(",") + ")"; pctx.fillRect(j * cell, i * cell, cell + 0.5, cell + 0.5); } }
      }
      pctx.lineCap = "round"; pctx.lineJoin = "round"; pctx.lineWidth = 18; pctx.strokeStyle = P.ink; pctx.fillStyle = P.ink;
      strokes.forEach(function (s) { pctx.beginPath(); pctx.moveTo(s[0][0], s[0][1]); if (s.length === 1) pctx.lineTo(s[0][0] + 0.1, s[0][1]); s.forEach(function (p) { pctx.lineTo(p[0], p[1]); }); pctx.stroke(); });
      if (!example && !strokes.length) { pctx.fillStyle = P.muted; pctx.font = "16px Spectral, Georgia, serif"; pctx.textAlign = "center"; pctx.fillText("Draw a digit here", PAD / 2, PAD / 2); }
    }
    function pt(e) { var r = pad.getBoundingClientRect(); return [(e.clientX - r.left) / r.width * PAD, (e.clientY - r.top) / r.height * PAD]; }
    pad.addEventListener("pointerdown", function (e) { try { pad.setPointerCapture(e.pointerId); } catch (err) {} drawing = true; if (example) { example = null; } strokes.push([pt(e)]); update(false); e.preventDefault(); });
    pad.addEventListener("pointermove", function (e) { if (!drawing) return; strokes[strokes.length - 1].push(pt(e)); schedule(); });
    function end() { drawing = false; }
    pad.addEventListener("pointerup", end); pad.addEventListener("pointercancel", end);
    var pending = false; function schedule() { if (pending) return; pending = true; requestAnimationFrame(function () { pending = false; update(false); }); }

    // the same steps as MNIST: fit the ink in a 20 by 20 box, then put its centre of mass in the middle of the 28 by 28 picture
    function prepare() {
      if (example) return example;
      octx.fillStyle = "#000"; octx.fillRect(0, 0, PAD, PAD);
      octx.lineCap = "round"; octx.lineJoin = "round"; octx.lineWidth = 18; octx.strokeStyle = "#fff"; octx.fillStyle = "#fff";
      strokes.forEach(function (s) { octx.beginPath(); octx.moveTo(s[0][0], s[0][1]); if (s.length === 1) octx.lineTo(s[0][0] + 0.1, s[0][1]); s.forEach(function (p) { octx.lineTo(p[0], p[1]); }); octx.stroke(); });
      x28.imageSmoothingEnabled = true; x28.imageSmoothingQuality = "high"; x28.fillStyle = "#000"; x28.fillRect(0, 0, 28, 28);
      if (!strokes.length) return new Float32Array(784);
      if (!centre) { x28.drawImage(off, 0, 0, 28, 28); }
      else {
        var d = octx.getImageData(0, 0, PAD, PAD).data, x0 = PAD, y0 = PAD, x1 = -1, y1 = -1, i, j;
        for (i = 0; i < PAD; i++) for (j = 0; j < PAD; j++) if (d[(i * PAD + j) * 4] > 60) { if (j < x0) x0 = j; if (j > x1) x1 = j; if (i < y0) y0 = i; if (i > y1) y1 = i; }
        if (x1 < 0) return new Float32Array(784);
        var bw = x1 - x0 + 1, bh = y1 - y0 + 1, sc = 20 / Math.max(bw, bh), w = Math.max(1, bw * sc), h = Math.max(1, bh * sc);
        x28.drawImage(off, x0, y0, bw, bh, (28 - w) / 2, (28 - h) / 2, w, h);
        var im = x28.getImageData(0, 0, 28, 28).data, m = 0, mx = 0, my = 0;
        for (i = 0; i < 28; i++) for (j = 0; j < 28; j++) { var v = im[(i * 28 + j) * 4]; m += v; mx += v * j; my += v * i; }
        if (m > 0) {
          var sx = 13.5 - mx / m, sy = 13.5 - my / m;
          xt.fillStyle = "#000"; xt.fillRect(0, 0, 28, 28); xt.imageSmoothingEnabled = true; xt.drawImage(c28, sx, sy);
          x28.fillStyle = "#000"; x28.fillRect(0, 0, 28, 28); x28.drawImage(t28, 0, 0);
        }
      }
      var data = x28.getImageData(0, 0, 28, 28).data, o = new Float32Array(784), k;
      for (k = 0; k < 784; k++) o[k] = data[k * 4] / 255;
      return o;
    }
    var maps = { c1: [], c2: [] }, kern = [];
    function build() {
      var P = Kit.palette();
      var g1 = Kit.$("rd-c1"), g2 = Kit.$("rd-c2"); g1.innerHTML = ""; g2.innerHTML = "";
      for (var f = 0; f < 6; f++) {
        var cell = Kit.html("figure", { class: "mp" }, g1), kc = Kit.html("canvas", { class: "fig kern", "aria-label": "filter " + (f + 1) + " of layer 1, 3 by 3 weights" }, cell);
        drawKernel(kc, f, P); var cv = Kit.html("canvas", { class: "fig", role: "img", "aria-label": "map " + (f + 1) + " of layer 1" }, cell); Kit.html("figcaption", {}, cell, "filter " + (f + 1)); maps.c1.push(cv);
      }
      for (f = 0; f < 12; f++) { var cell2 = Kit.html("figure", { class: "mp" }, g2), cv2 = Kit.html("canvas", { class: "fig", role: "img", "aria-label": "map " + (f + 1) + " of layer 2" }, cell2); Kit.html("figcaption", {}, cell2, "map " + (f + 1)); maps.c2.push(cv2); }
      var pb = Kit.$("rd-bars"); pb.innerHTML = "";
      for (var k = 0; k < 10; k++) {
        var row = Kit.html("div", { class: "pb", "data-k": k, tabindex: "0", role: "button", "aria-label": "digit " + k }, pb);
        Kit.html("span", { class: "d" }, row, String(k)); var tr = Kit.html("span", { class: "tr" }, row); Kit.html("span", { class: "fill" }, tr); Kit.html("span", { class: "v" }, row, "0.00");
        (function (k, row) { row.addEventListener("click", function () { picked = picked === k ? -1 : k; render(); }); row.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); picked = picked === k ? -1 : k; render(); } }); })(k, row);
      }
    }
    function drawKernel(cv, f, P) {
      cv.width = 3; cv.height = 3; var ctx = cv.getContext("2d"), w = [], i, mx = 0;
      for (i = 0; i < 9; i++) { w.push(NET.c1w[f * 9 + i]); mx = Math.max(mx, Math.abs(w[i])); }
      for (i = 0; i < 9; i++) { var t = w[i] / mx; ctx.fillStyle = "rgb(" + (t >= 0 ? mixRGB(P.paper, P.teal, t) : mixRGB(P.paper, P.coral, -t)).join(",") + ")"; ctx.fillRect(i % 3, Math.floor(i / 3), 1, 1); }
    }
    function update(fresh) {
      var x = prepare(); S.set(x, "read");
      redrawPad(); render();
      if (fresh) { }
    }
    function render() {
      if (!NET) return;
      var P = Kit.palette(), x = S.x, st = forward(x); state = st;
      redrawPad();
      var c28v = Kit.$("rd-in"); drawMap(c28v, x, 0, 28, 1, P, "ink");
      var key1 = mapMode ? "p1" : "a1", n1 = mapMode ? 13 : 26, key2 = mapMode ? "p2" : "a2", n2 = mapMode ? 5 : 11, i, m1 = 0, m2 = 0;
      for (i = 0; i < st[key1].length; i++) if (st[key1][i] > m1) m1 = st[key1][i];
      for (i = 0; i < st[key2].length; i++) if (st[key2][i] > m2) m2 = st[key2][i];
      for (i = 0; i < 6; i++) drawMap(maps.c1[i], st[key1], i * n1 * n1, n1, m1, P);
      for (i = 0; i < 12; i++) drawMap(maps.c2[i], st[key2], i * n2 * n2, n2, m2, P);
      Kit.$("rd-h1").textContent = "Layer 1: 6 filters, maps " + n1 + " by " + n1 + (mapMode ? " after pooling" : " after ReLU");
      Kit.$("rd-h2").textContent = "Layer 2: 12 filters, maps " + n2 + " by " + n2 + (mapMode ? " after pooling" : " after ReLU");
      var best = 0; for (i = 1; i < 10; i++) if (st.p[i] > st.p[best]) best = i;
      var ink = 0; for (i = 0; i < 784; i++) ink += x[i];
      var rows = Kit.$("rd-bars").children;
      for (i = 0; i < 10; i++) { var r = rows[i]; r.querySelector(".fill").style.width = (st.p[i] * 100).toFixed(1) + "%"; r.querySelector(".v").textContent = st.p[i].toFixed(2); r.classList.toggle("top", i === best); r.classList.toggle("pick", i === picked); }
      Kit.$("rd-ans").innerHTML = ink < 3 ? "Draw something." : "The network reads <b>" + best + "</b> with probability <b>" + st.p[best].toFixed(2) + "</b>";
      var k = picked >= 0 ? picked : best, sumP = 0; for (i = 0; i < 10; i++) sumP += st.p[i];
      Kit.$("rd-eqn").innerHTML = ebox("c-navy", Kit.fmt(st.z[k], 2), "score z for " + k) + eop("&rarr;") + ebox("c-sand", Kit.fmt(Math.exp(st.z[k] - st.zmax), 3), "e<sup>z &minus; z max</sup>") + eop("&divide;") + ebox("c-sand", Kit.fmt(st.expSum, 3), "sum over all ten") + eop("=") + ebox("c-coral", st.p[k].toFixed(3), "probability of " + k) + eop("|") + ebox("c-teal", sumP.toFixed(3), "ten probabilities add up to");
    }
    build(); loadExample(4);
    Kit.onTheme(function () { render(); });
    window.addEventListener("resize", Kit.debounce(redrawPad));
  }

  // ---------- 3. receptive field ----------
  function rfFig() {
    var layer = "conv2", ch = 0, cell = [3, 3], showInf = true, cvIn = Kit.$("rf-in"), cvMap = Kit.$("rf-map"), CELL = 12;
    var ictx = null;
    var bar = Kit.$("rf-bar"), r1 = Kit.html("div", { class: "bar-row" }, bar), r2 = Kit.html("div", { class: "bar-row" }, bar), chBar = null;
    var lchips = Kit.chips(r1, LAYERS.map(function (L) { return L.name + " (" + LAYER_SIZE[L.id] + " by " + LAYER_SIZE[L.id] + ")"; }), 2, function (i) { layer = LAYERS[i].id; ch = bestChannel(); cell = bestCell(); buildCh(); draw(); });
    var il = label(r1, "", "check"), ic = Kit.html("input", { type: "checkbox", checked: "checked" }, il); il.appendChild(document.createTextNode(" Show which pixels matter (nudge test)"));
    ic.addEventListener("change", function () { showInf = ic.checked; draw(); });
    function buildCh() { r2.innerHTML = ""; Kit.html("span", { class: "label" }, r2, "Map:"); var n = LAYER_CH[layer], labs = []; for (var i = 0; i < n; i++) labs.push(String(i + 1)); chBar = Kit.chips(r2, labs, ch, function (i) { ch = i; cell = bestCell(); draw(); }); }
    function bestCell() {
      var n = LAYER_SIZE[layer], d = forward(S.x, layer)[LAYER_KEY[layer]], off = ch * n * n, b = 0, i;
      for (i = 1; i < n * n; i++) if (d[off + i] > d[off + b]) b = i;
      return [Math.floor(b / n), b % n];
    }
    function bestChannel() {
      var n = LAYER_SIZE[layer], d = forward(S.x, layer)[LAYER_KEY[layer]], best = 0, bv = -1, f, i;
      for (f = 0; f < LAYER_CH[layer]; f++) { var m = 0; for (i = 0; i < n * n; i++) if (d[f * n * n + i] > m) m = d[f * n * n + i]; if (m > bv) { bv = m; best = f; } }
      return best;
    }
    function pick(e) {
      var r = cvMap.getBoundingClientRect(), n = LAYER_SIZE[layer], i = Math.floor((e.clientY - r.top) / r.height * n), j = Math.floor((e.clientX - r.left) / r.width * n);
      cell = [Math.max(0, Math.min(n - 1, i)), Math.max(0, Math.min(n - 1, j))]; draw();
    }
    cvMap.addEventListener("pointerdown", function (e) { try { cvMap.setPointerCapture(e.pointerId); } catch (err) {} pick(e); });
    cvMap.addEventListener("pointermove", function (e) { if (e.buttons) pick(e); });
    cvMap.addEventListener("keydown", function (e) {
      var d = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] }[e.key]; if (!d) return; e.preventDefault(); var n = LAYER_SIZE[layer];
      cell = [Math.max(0, Math.min(n - 1, cell[0] + d[0])), Math.max(0, Math.min(n - 1, cell[1] + d[1]))]; draw();
    });
    function draw() {
      if (!NET) return;
      var P = Kit.palette(), x = S.x, n = LAYER_SIZE[layer], o = forward(x, layer), data = o[LAYER_KEY[layer]], i, j;
      var cv = cvIn; cv.width = 28 * CELL; cv.height = 28 * CELL; var ctx = cv.getContext("2d");
      for (i = 0; i < 28; i++) for (j = 0; j < 28; j++) { ctx.fillStyle = "rgb(" + mixRGB(P.paper, P.ink, x[i * 28 + j]).join(",") + ")"; ctx.fillRect(j * CELL, i * CELL, CELL, CELL); }
      var box = rfBox(layer, cell[0], cell[1]), b = { y0: Math.max(0, box.y0), x0: Math.max(0, box.x0), y1: Math.min(27, box.y1), x1: Math.min(27, box.x1) };
      var g = null, mxg = 0, count = 0;
      if (showInf) {
        g = influence(x, layer, ch, cell[0], cell[1], b); for (i = 0; i < 784; i++) { var a = Math.abs(g[i]); if (a > mxg) mxg = a; }
        for (i = b.y0; i <= b.y1; i++) for (j = b.x0; j <= b.x1; j++) { var a2 = Math.abs(g[i * 28 + j]); if (mxg > 0 && a2 > 0.1 * mxg) count++; }
        for (i = b.y0; i <= b.y1; i++) for (j = b.x0; j <= b.x1; j++) { var t = mxg > 0 ? Math.min(1, Math.abs(g[i * 28 + j]) / mxg) : 0; if (t > 0.02) { ctx.fillStyle = "rgba(" + hexRGB(P.teal).join(",") + "," + (0.15 + 0.6 * t).toFixed(2) + ")"; ctx.fillRect(j * CELL, i * CELL, CELL, CELL); } }
      }
      ctx.fillStyle = "rgba(" + hexRGB(P.coral).join(",") + ",0.10)"; ctx.fillRect(b.x0 * CELL, b.y0 * CELL, (b.x1 - b.x0 + 1) * CELL, (b.y1 - b.y0 + 1) * CELL);
      ctx.strokeStyle = P.coral; ctx.lineWidth = 3; ctx.strokeRect(b.x0 * CELL + 1.5, b.y0 * CELL + 1.5, (b.x1 - b.x0 + 1) * CELL - 3, (b.y1 - b.y0 + 1) * CELL - 3);
      // the map, enlarged
      var mc = cvMap, mcell = Math.max(8, Math.floor(336 / n)); mc.width = n * mcell; mc.height = n * mcell; var mx = mc.getContext("2d"), mm = 0, off = ch * n * n;
      for (i = 0; i < n * n; i++) if (data[off + i] > mm) mm = data[off + i];
      for (i = 0; i < n; i++) for (j = 0; j < n; j++) { var tt = mm > 0 ? data[off + i * n + j] / mm : 0; mx.fillStyle = "rgb(" + mixRGB(P.paper, P.teal, Math.min(1, tt)).join(",") + ")"; mx.fillRect(j * mcell, i * mcell, mcell, mcell); mx.strokeStyle = P.rule; mx.lineWidth = 1; mx.strokeRect(j * mcell + 0.5, i * mcell + 0.5, mcell - 1, mcell - 1); }
      mx.strokeStyle = P.coral; mx.lineWidth = 3; mx.strokeRect(cell[1] * mcell + 1.5, cell[0] * mcell + 1.5, mcell - 3, mcell - 3);
      var val = data[off + cell[0] * n + cell[1]];
      var T = rfTable(), li = LAYERS.map(function (L) { return L.id; }).indexOf(layer) + 1, row = T[li], prev = T[li - 1];
      Kit.$("rf-eqn").innerHTML = ebox("c-navy", prev.r + " + (" + row.k + " &minus; 1) &times; " + prev.j, "r in + (k &minus; 1) &times; j in") + eop("=") + ebox("c-coral", row.r + " &times; " + row.r, "receptive field, pixels") + eop("|") + ebox("c-sand", prev.j + " &times; " + row.s + " = " + row.j, "jump: j in &times; s") + eop("|") + ebox("c-teal", "rows " + box.y0 + " to " + Math.min(27, box.y1) + ", columns " + box.x0 + " to " + Math.min(27, box.x1), "cell (" + cell[0] + ", " + cell[1] + ") reads these input pixels");
      Kit.$("rf-status").textContent = "Value of the selected cell: " + val.toFixed(2) + "." + (showInf ? " Of the " + (b.y1 - b.y0 + 1) * (b.x1 - b.x0 + 1) + " pixels in its receptive field, " + count + " change the value by more than a tenth of the largest change when nudged (teal)." : "");
      var tb = Kit.$("rf-table"), h = '<table class="arch"><thead><tr><th>Layer</th><th>k</th><th>s</th><th>r in</th><th>j in</th><th>r out</th><th>j out</th></tr></thead><tbody>';
      T.forEach(function (rw, idx) { h += '<tr class="' + (idx === li ? "sel" : "") + '"><td>' + rw.name + "</td><td>" + (rw.k || "") + "</td><td>" + (rw.s || "") + "</td><td>" + (idx ? rw.rin : "") + "</td><td>" + (idx ? rw.jin : "") + "</td><td>" + rw.r + "</td><td>" + rw.j + "</td></tr>"; });
      tb.innerHTML = h + "</tbody></table>";
    }
    var first = true;
    function init() { if (first) { ch = bestChannel(); cell = bestCell(); buildCh(); first = false; } draw(); }
    init();
    S.listeners.push(function (src) { if (src === "read" && !first) { /* a new drawing: keep the cell if it is still active, else pick the best */ var n = LAYER_SIZE[layer], d = forward(S.x, layer)[LAYER_KEY[layer]]; if (!(d[ch * n * n + cell[0] * n + cell[1]] > 0)) { ch = bestChannel(); cell = bestCell(); buildCh(); } } draw(); });
    Kit.onTheme(draw);
  }

  // ---------- 4. parameter counts ----------
  function paramFig() {
    var n = 28, cin = 1, F = 6, k = 3, same = false;
    var bar = Kit.$("pm-bar"), ctrl = Kit.$("pm-controls");
    var sn, sf, sk, sc, ss;
    var pch = Kit.chips(bar, ["Digit: 28 by 28, 1 channel, 6 filters", "Colour crop: 64 by 64, 3 channels, 16 filters", "Photo: 224 by 224, 3 channels, 64 filters"], 0, function (i) {
      var P = [[28, 1, 6, 3, false], [64, 3, 16, 3, true], [224, 3, 64, 3, true]][i]; n = P[0]; cin = P[1]; F = P[2]; k = P[3]; same = P[4]; sn.set(n); sc.set(cin); sf.set(F); sk.set(k); ss.value = same ? "same" : "valid"; draw();
    });
    sn = Kit.slider(ctrl, { label: "Picture side n", min: 8, max: 256, step: 1, value: 28, digits: 0 }, function (v) { n = v; draw(); });
    sc = Kit.slider(ctrl, { label: "Channels in", min: 1, max: 3, step: 2, value: 1, digits: 0 }, function (v) { cin = v; draw(); });
    sf = Kit.slider(ctrl, { label: "Filters (maps out)", min: 1, max: 128, step: 1, value: 6, digits: 0 }, function (v) { F = v; draw(); });
    sk = Kit.slider(ctrl, { label: "Filter side k", min: 1, max: 7, step: 2, value: 3, digits: 0 }, function (v) { k = v; draw(); });
    ss = Kit.select(ctrl, "Padding", [{ value: "valid", label: "none (valid)" }, { value: "same", label: "same" }], "valid", function (v) { same = v === "same"; draw(); });
    function draw() {
      var c = paramCounts(n, cin, F, k, same);
      Kit.plot("pm-plot", function (P) {
        return [{ type: "bar", x: ["Dense<br>layer", "Locally<br>connected", "Convolution<br>layer"], y: [c.dense, c.local, c.conv], marker: { color: [P.coral, P.sand, P.teal] }, text: [big(c.dense), big(c.local), big(c.conv)], textposition: "outside", cliponaxis: false, hovertemplate: "%{x}: %{y:,.0f} numbers<extra></extra>" }];
      }, { yaxis: { type: "log", title: { text: "numbers to learn (log scale)" }, range: [0, Math.max(3, Math.ceil(Math.log10(c.dense)) + 1)], automargin: true }, xaxis: { tickfont: { size: 12 }, tickangle: 0, automargin: true }, margin: { l: 62, r: 16, t: 28, b: 56 } });
      Kit.$("pm-plot").setAttribute("aria-label", "Parameter counts on a log axis: dense layer " + big(c.dense) + ", locally connected " + big(c.local) + ", convolution " + big(c.conv));
      var bytes = c.dense * 4, bs = bytes >= 1e12 ? (bytes / 1e12).toFixed(1) + " TB" : bytes >= 1e9 ? (bytes / 1e9).toFixed(1) + " GB" : bytes >= 1e6 ? (bytes / 1e6).toFixed(1) + " MB" : (bytes / 1e3).toFixed(0) + " KB";
      Kit.$("pm-eqn").innerHTML = ebox("c-navy", c.inputs.toLocaleString("en-US"), "inputs: n &times; n &times; channels") + eop("&rarr;") + ebox("c-navy", c.outputs.toLocaleString("en-US"), "outputs: " + c.m + " &times; " + c.m + " &times; " + F) +
        eop("|") + ebox("c-coral", "(" + c.inputs.toLocaleString("en-US") + " + 1) &times; " + c.outputs.toLocaleString("en-US"), "dense: inputs + 1, times outputs") + eop("=") + ebox("c-coral", c.dense.toLocaleString("en-US"), "numbers (" + bs + " in 32-bit floats)");
      Kit.$("pm-eqn2").innerHTML = ebox("c-sand", "(" + k + "&times;" + k + "&times;" + cin + " + 1) &times; " + c.outputs.toLocaleString("en-US"), "local: one filter per output square") + eop("=") + ebox("c-sand", c.local.toLocaleString("en-US"), "numbers") + eop("|") + ebox("c-teal", "(" + k + "&times;" + k + "&times;" + cin + " + 1) &times; " + F, "conv: one set of filters, shared") + eop("=") + ebox("c-teal", c.conv.toLocaleString("en-US"), "numbers") + eop("|") + ebox("c-navy", big(c.dense / c.conv), "times more numbers in the dense layer than in conv");
    }
    draw();
  }

  // ---------- 5. move or turn a digit ----------
  function limitFig() {
    var mode = 0, amount = 0, dig = 7, bar = Kit.$("lm-bar"), cv = Kit.$("lm-cv");
    Kit.chips(bar, ["Move sideways", "Turn"], 0, function (i) { mode = i; amount = 0; sl.input.max = i ? 90 : 8; sl.input.step = i ? 10 : 1; sl.set(0); sl.label.textContent = i ? "Turn, degrees" : "Move right, squares"; draw(); });
    Kit.select(bar, "Digit", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(function (d) { return { value: d, label: String(d) }; }), 7, function (v) { dig = +v; draw(); });
    var sl = Kit.slider(Kit.$("lm-controls"), { label: "Move right, squares", min: 0, max: 8, step: 1, value: 0, digits: 0 }, function (v) { amount = v; draw(); });
    var R = J.robust;
    function draw() {
      var P = Kit.palette(), e = EX.filter(function (q) { return q.label === dig; })[0], x = decode(e.b64), y = mode ? rotateDigit(x, amount) : shiftDigit(x, amount);
      drawMap(cv, y, 0, 28, 1, P, "ink");
      var st = forward(y), best = 0, i; for (i = 1; i < 10; i++) if (st.p[i] > st.p[best]) best = i;
      Kit.$("lm-ans").innerHTML = "The network reads <b>" + best + "</b> (probability " + st.p[best].toFixed(2) + "). Probability of the true digit " + dig + ": <b>" + st.p[dig].toFixed(2) + "</b>.";
      var xs = mode ? R.aug.rot_deg : R.aug.shift_px, plain = mode ? R.plain.rot_acc : R.plain.shift_acc, aug = mode ? R.aug.rot_acc : R.aug.shift_acc;
      var Qk = Kit.palette(), kb = Kit.$("lm-key"); kb.innerHTML = "";
      [[Qk.teal, "trained with small shifts, turns and zoom (this chapter's network)", ""], [Qk.navy, "trained on centred digits only", ""], [Qk.coral, "the slider", "dot"]].forEach(function (it) { var sp = Kit.html("span", {}, kb), sw = Kit.html("i", { class: it[2] }, sp); sw.style.borderTopColor = it[0]; Kit.html("em", {}, sp, it[1]); });
      Kit.plot("lm-plot", function (Q) {
        return [
          { x: xs, y: plain, mode: "lines+markers", line: { color: Q.navy, width: 3 }, marker: { size: 7, symbol: "square" }, hovertemplate: "%{y:.1%}, trained on centred digits only<extra></extra>" },
          { x: xs, y: aug, mode: "lines+markers", line: { color: Q.teal, width: 3 }, marker: { size: 7 }, hovertemplate: "%{y:.1%}, trained with small shifts, turns and zoom<extra></extra>" },
          { x: [amount, amount], y: [0, 1], mode: "lines", line: { color: Q.coral, width: 2, dash: "dot" }, hoverinfo: "skip" }
        ];
      }, function (Q) {
        return { xaxis: { title: { text: mode ? "turn of the whole test set, degrees" : "shift of the whole test set, squares" }, dtick: mode ? 10 : 1, automargin: true }, yaxis: { title: { text: "accuracy on 10,000 test digits" }, range: [0, 1.02], tickformat: ".0%", automargin: true },
          annotations: [] };
      });
    }
    draw();
    Kit.onTheme(draw);
  }
})();
