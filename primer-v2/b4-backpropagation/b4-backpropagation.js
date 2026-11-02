// B4 Backpropagation: figures for the tiny network on one shark (made-up weights, Tag 3305).
//   1 graphFig("g1")   forward pass only: values fill the graph node by node (D3-free SVG)
//   2 graphFig("g2")   the signature figure: Forward, Backward, Step, Back, Update, with equation boxes and a loss history
//   3 pathsFig         one value used twice: the slopes along two paths add up
//   4 layerFig         a whole layer at once: dW is an outer product, the slope to the inputs is W-transpose times delta
//   5 vanishFig        a chain of one-neuron layers: the slope reaching each layer (Plotly)
//   6 checkFig         gradient checking: backprop against finite differences, with a bug switch (Plotly)
(function () {
  "use strict";

  // ---------- the tiny network: two inputs, two hidden units, one output, log loss ----------
  var ACT = {
    tanh: { name: "tanh", g: Math.tanh, dg: function (h) { return 1 - h * h; }, word: "tanh" },
    relu: { name: "ReLU", g: function (z) { return Math.max(0, z); }, dg: function (h, z) { return z > 0 ? 1 : 0; }, word: "max(0, " },
    sigmoid: { name: "sigmoid", g: function (z) { return 1 / (1 + Math.exp(-z)); }, dg: function (h) { return h * (1 - h); }, word: "sigma" }
  };
  var PARAMS = ["w11", "w12", "w21", "w22", "b1", "b2", "v1", "v2", "c"];
  var NAMES = { x1: "x1", x2: "x2", w11: "w11", w12: "w12", w21: "w21", w22: "w22", b1: "b1", b2: "b2", v1: "v1", v2: "v2", c: "c" };
  function newNet() {
    return { x: [0.22, 0.21], y: 1, W: [[0.8, -0.6], [-0.4, 0.7]], b: [0.1, -0.2], v: [0.9, -0.8], c: -0.9, act: "tanh" };
  }
  function cloneNet(N) { return { x: N.x.slice(), y: N.y, W: [N.W[0].slice(), N.W[1].slice()], b: N.b.slice(), v: N.v.slice(), c: N.c, act: N.act }; }
  function getP(N, k) {
    switch (k) {
      case "w11": return N.W[0][0]; case "w12": return N.W[0][1]; case "w21": return N.W[1][0]; case "w22": return N.W[1][1];
      case "b1": return N.b[0]; case "b2": return N.b[1]; case "v1": return N.v[0]; case "v2": return N.v[1]; case "c": return N.c;
      case "x1": return N.x[0]; case "x2": return N.x[1];
    }
  }
  function setP(N, k, val) {
    switch (k) {
      case "w11": N.W[0][0] = val; break; case "w12": N.W[0][1] = val; break; case "w21": N.W[1][0] = val; break; case "w22": N.W[1][1] = val; break;
      case "b1": N.b[0] = val; break; case "b2": N.b[1] = val; break; case "v1": N.v[0] = val; break; case "v2": N.v[1] = val; break; case "c": N.c = val; break;
      case "x1": N.x[0] = val; break; case "x2": N.x[1] = val; break;
    }
  }
  function sigm(z) { return 1 / (1 + Math.exp(-z)); }
  function forward(N) {
    var A = ACT[N.act], z = [0, 1].map(function (j) { return N.W[j][0] * N.x[0] + N.W[j][1] * N.x[1] + N.b[j]; });
    var h = z.map(A.g), zo = N.v[0] * h[0] + N.v[1] * h[1] + N.c, p = sigm(zo);
    // stable log loss written with the output score: softplus(zo) - y * zo
    var L = Math.max(zo, 0) + Math.log1p(Math.exp(-Math.abs(zo))) - N.y * zo;
    return { z: z, h: h, zo: zo, p: p, L: L };
  }
  function lossOf(N) { return forward(N).L; }
  // bug: "none", "noact" (forgot the hidden slope), "sign" (wrong sign on p - y), "swap" (used v instead of h for the slope of v)
  function backward(N, F, bug) {
    bug = bug || "none";
    var A = ACT[N.act], p = F.p, y = N.y;
    var dp = y === 1 ? -1 / p : 1 / (1 - p);
    var dzo = dp * p * (1 - p);
    if (bug === "sign") dzo = -dzo;
    var dv = [dzo * (bug === "swap" ? N.v[0] : F.h[0]), dzo * (bug === "swap" ? N.v[1] : F.h[1])], dc = dzo;
    var dh = [dzo * N.v[0], dzo * N.v[1]];
    var loc = F.h.map(function (h, j) { return bug === "noact" ? 1 : A.dg(h, F.z[j]); });
    var dz = [dh[0] * loc[0], dh[1] * loc[1]];
    var dW = [[dz[0] * N.x[0], dz[0] * N.x[1]], [dz[1] * N.x[0], dz[1] * N.x[1]]];
    var dx = [N.W[0][0] * dz[0] + N.W[1][0] * dz[1], N.W[0][1] * dz[0] + N.W[1][1] * dz[1]];
    return { dp: dp, dzo: dzo, dv: dv, dc: dc, dh: dh, loc: loc, dz: dz, dW: dW, db: dz.slice(), dx: dx };
  }
  function gradVec(N, B) { return [B.dW[0][0], B.dW[0][1], B.dW[1][0], B.dW[1][1], B.db[0], B.db[1], B.dv[0], B.dv[1], B.dc]; }
  function numGrad(N, key, h, central) {
    var M = cloneNet(N), w0 = getP(N, key), l0 = lossOf(N), up, dn;
    setP(M, key, w0 + h); up = lossOf(M);
    if (!central) return (up - l0) / h;
    setP(M, key, w0 - h); dn = lossOf(M);
    return (up - dn) / (2 * h);
  }

  // ---------- small helpers ----------
  function nf(v) { if (!isFinite(v)) return "inf"; var a = Math.abs(v); return Kit.fmt(v, a >= 10 ? 2 : a >= 0.095 ? 3 : a === 0 ? 3 : 4); }
  function sci(v) { if (v === 0) return "0"; if (!isFinite(v)) return "inf"; var e = Math.floor(Math.log10(Math.abs(v))), m = v / Math.pow(10, e); if (Math.abs(m) >= 9.95) { m = m / 10; e += 1; } return (v < 0 ? "−" : "") + Math.abs(m).toFixed(1) + "e" + (e < 0 ? "−" : "+") + Math.abs(e); }
  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { var C = hexRGB(c); return "rgba(" + C[0] + "," + C[1] + "," + C[2] + "," + a + ")"; }
  function mix(a, b, t) { var A = hexRGB(a), B = hexRGB(b); return "rgb(" + [0, 1, 2].map(function (i) { return Math.round(A[i] + (B[i] - A[i]) * t); }).join(",") + ")"; }
  function typeset(node) { if (typeof renderMathInElement !== "undefined") renderMathInElement(node, { delimiters: [{ left: "\\[", right: "\\]", display: true }, { left: "\\(", right: "\\)", display: false }], throwOnError: false }); }
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  // a row of equation boxes from tokens: {b: big text, i: caption, c: colour class} or {o: operator}
  function rowHTML(tokens) {
    var aria = tokens.map(function (t) { return t.o ? t.o.replace(/&[a-z]+;/g, " ") : (t.i ? t.i.replace(/<[^>]+>/g, "") + " " : "") + t.b; }).join(" ");
    return '<div class="eqn" role="img" aria-label="' + aria.replace(/"/g, "'") + '">' + tokens.map(function (t) { return t.o ? eop(t.o) : ebox(t.c || "", t.b, t.i || ""); }).join("") + '</div>';
  }
  function bx(v, cap, cls) { return { b: nf(v), i: cap, c: cls }; }
  function op(s) { return { o: s }; }

  // ---------- the graph, drawn in SVG; two layouts, wide and tall ----------
  var EDGES = [
    { a: "x1", b: "z1", chip: "w11" }, { a: "x2", b: "z1", chip: "w12" }, { a: "b1", b: "z1" },
    { a: "x1", b: "z2", chip: "w21" }, { a: "x2", b: "z2", chip: "w22" }, { a: "b2", b: "z2" },
    { a: "z1", b: "h1" }, { a: "z2", b: "h2" },
    { a: "h1", b: "zo", chip: "v1" }, { a: "h2", b: "zo", chip: "v2" }, { a: "c", b: "zo" },
    { a: "zo", b: "p" }, { a: "p", b: "L" }, { a: "y", b: "L" }
  ];
  var CHIPS = PARAMS;
  function layout(vertical) {
    var pos = {}, vb, nw, nh, cw, ch;
    function lerp(a, b, t) { return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t }; }
    if (!vertical) {
      vb = [900, 450]; nw = 92; nh = 58; cw = 80; ch = 42;
      pos.x1 = { x: 62, y: 150 }; pos.x2 = { x: 62, y: 350 }; pos.z1 = { x: 275, y: 150 }; pos.z2 = { x: 275, y: 350 };
      pos.h1 = { x: 435, y: 150 }; pos.h2 = { x: 435, y: 350 }; pos.zo = { x: 595, y: 250 }; pos.p = { x: 710, y: 250 }; pos.L = { x: 826, y: 250 }; pos.y = { x: 826, y: 130 };
      pos.w11 = lerp(pos.x1, pos.z1, 0.5); pos.w22 = lerp(pos.x2, pos.z2, 0.5);
      pos.w21 = lerp(pos.x1, pos.z2, 0.36); pos.w12 = lerp(pos.x2, pos.z1, 0.36);
      pos.b1 = { x: 275, y: 56 }; pos.b2 = { x: 275, y: 424 }; pos.c = { x: 595, y: 130 };
      pos.v1 = lerp(pos.h1, pos.zo, 0.5); pos.v2 = lerp(pos.h2, pos.zo, 0.5);
    } else {
      vb = [420, 790]; nw = 84; nh = 56; cw = 68; ch = 40;
      pos.x1 = { x: 130, y: 44 }; pos.x2 = { x: 290, y: 44 }; pos.z1 = { x: 130, y: 210 }; pos.z2 = { x: 290, y: 210 };
      pos.h1 = { x: 130, y: 350 }; pos.h2 = { x: 290, y: 350 }; pos.zo = { x: 210, y: 490 }; pos.p = { x: 210, y: 600 }; pos.L = { x: 210, y: 712 }; pos.y = { x: 346, y: 712 };
      pos.w11 = lerp(pos.x1, pos.z1, 0.7); pos.w22 = lerp(pos.x2, pos.z2, 0.7);
      pos.w21 = lerp(pos.x1, pos.z2, 0.3); pos.w12 = lerp(pos.x2, pos.z1, 0.3);
      pos.b1 = { x: 40, y: 210 }; pos.b2 = { x: 380, y: 210 }; pos.c = { x: 350, y: 490 };
      pos.v1 = lerp(pos.h1, pos.zo, 0.5); pos.v2 = lerp(pos.h2, pos.zo, 0.5);
    }
    CHIPS.forEach(function (k) { pos[k].hw = cw / 2; pos[k].hh = ch / 2; });
    ["x1", "x2", "z1", "z2", "h1", "h2", "zo", "p", "L", "y"].forEach(function (k) { pos[k].hw = nw / 2; pos[k].hh = nh / 2; });
    pos.y.hh = 24; pos.y.hw = 44;
    return { pos: pos, vb: vb };
  }
  function clipPt(from, to) {
    // point where the segment from `from` centre toward `to` centre leaves from's rectangle
    var dx = to.x - from.x, dy = to.y - from.y, t = Math.min(dx ? from.hw / Math.abs(dx) : 1e9, dy ? from.hh / Math.abs(dy) : 1e9);
    return { x: from.x + dx * t, y: from.y + dy * t };
  }

  // ---------- the step script for one network state ----------
  function buildSteps(N) {
    var F = forward(N), B = backward(N, F, "none"), A = ACT[N.act], S = [];
    var yw = N.y === 1 ? "adult" : "juvenile";
    function actWord(zid) { return N.act === "tanh" ? "tanh(" + zid + ")" : N.act === "relu" ? "max(0, " + zid + ")" : "σ(" + zid + ")"; }
    // forward steps: z1, h1, z2, h2, zo, p, L
    [0, 1].forEach(function (j) {
      var n = j + 1, zid = "z" + n, hid = "h" + n, xi = ["x1", "x2"];
      var setz = {}; setz[zid] = F.z[j];
      S.push({
        phase: "fwd", edges: ["x1>" + zid, "x2>" + zid, "b" + n + ">" + zid], hi: [zid], val: setz, grad: {},
        text: "Forward. The sum node " + zid + " adds the inputs times their weights, plus the bias.",
        rows: [[bx(N.W[j][0], "w" + n + "1", "c-navy"), op("&times;"), bx(N.x[0], "x1", "c-navy"), op("+"), bx(N.W[j][1], "w" + n + "2", "c-navy"), op("&times;"), bx(N.x[1], "x2", "c-navy"), op("+"), bx(N.b[j], "b" + n, "c-navy"), op("="), bx(F.z[j], zid, "c-teal")]]
      });
      var seth = {}; seth[hid] = F.h[j];
      S.push({
        phase: "fwd", edges: [zid + ">" + hid], hi: [hid], val: seth, grad: {},
        text: "Forward. The activation " + A.name + " turns " + zid + " into " + hid + "." + (N.act === "relu" && F.z[j] <= 0 ? " " + zid + " is not above 0, so the unit gives 0." : ""),
        rows: [[{ o: N.act === "tanh" ? "tanh(" : N.act === "relu" ? "max(0," : "σ(" }, bx(F.z[j], zid, "c-navy"), op(")"), op("="), bx(F.h[j], hid, "c-teal")]]
      });
    });
    S.push({
      phase: "fwd", edges: ["h1>zo", "h2>zo", "c>zo"], hi: ["zo"], val: { zo: F.zo }, grad: {},
      text: "Forward. The output score zo adds the hidden values times their weights, plus the bias c.",
      rows: [[bx(N.v[0], "v1", "c-navy"), op("&times;"), bx(F.h[0], "h1", "c-navy"), op("+"), bx(N.v[1], "v2", "c-navy"), op("&times;"), bx(F.h[1], "h2", "c-navy"), op("+"), bx(N.c, "c", "c-navy"), op("="), bx(F.zo, "zo", "c-teal")]]
    });
    S.push({
      phase: "fwd", edges: ["zo>p"], hi: ["p"], val: { p: F.p }, grad: {},
      text: "Forward. The sigmoid turns the score into a probability that the shark is an adult.",
      rows: [[{ o: "σ(" }, bx(F.zo, "zo", "c-navy"), op(")"), op("="), bx(F.p, "p, chance of adult", "c-teal")]]
    });
    S.push({
      phase: "fwd", edges: ["p>L", "y>L"], hi: ["L"], val: { L: F.L }, grad: {},
      text: "Forward. The loss is large when the network gave a low chance to the true label (here " + yw + ", y = " + N.y + ").",
      rows: [[{ o: N.y === 1 ? "&minus;ln(" : "&minus;ln(1 &minus;" }, bx(F.p, "p", "c-navy"), op(")"), op("="), bx(F.L, "loss L", "c-teal")]]
    });
    // backward steps
    var dpLocal = N.y === 1 ? -1 / F.p : 1 / (1 - F.p);
    S.push({
      phase: "bwd", edges: ["p>L"], hi: ["L", "p"], val: {}, grad: { L: 1, p: B.dp },
      text: "Backward 1. The loss slope with respect to itself is 1. The local slope of the loss with respect to p is " + (N.y === 1 ? "−1/p" : "1/(1−p)") + ".",
      rows: [[bx(1, "upstream ∂L/∂L", "c-coral"), op("&times;"), bx(dpLocal, "local ∂L/∂p", "c-navy"), op("="), bx(B.dp, "∂L/∂p", "c-coral")]]
    });
    S.push({
      phase: "bwd", edges: ["zo>p"], hi: ["p", "zo"], val: {}, grad: { zo: B.dzo },
      text: "Backward 2. Through the sigmoid the local slope is p(1−p). The product simplifies to p − y, the slope A3 found for log loss.",
      rows: [[bx(B.dp, "upstream ∂L/∂p", "c-coral"), op("&times;"), bx(F.p * (1 - F.p), "local p(1&minus;p)", "c-navy"), op("="), bx(B.dzo, "∂L/∂zo", "c-coral"), op("="), bx(F.p - N.y, "p &minus; y", "c-sand")]]
    });
    [0, 1].forEach(function (j) {
      var n = j + 1, hid = "h" + n, vid = "v" + n, g = {}; g[hid] = B.dh[j]; g[vid] = B.dv[j];
      S.push({
        phase: "bwd", edges: [hid + ">zo"], hi: [hid, vid, "zo"], val: {}, grad: g,
        text: "Backward " + (2 + n) + ". A product node: the slope to " + hid + " uses the other factor " + vid + ", and the slope to " + vid + " uses " + hid + ".",
        rows: [
          [bx(B.dzo, "upstream ∂L/∂zo", "c-coral"), op("&times;"), bx(N.v[j], "local ∂zo/∂" + hid + " = " + vid, "c-navy"), op("="), bx(B.dh[j], "∂L/∂" + hid, "c-coral")],
          [bx(B.dzo, "upstream ∂L/∂zo", "c-coral"), op("&times;"), bx(F.h[j], "local ∂zo/∂" + vid + " = " + hid, "c-navy"), op("="), bx(B.dv[j], "∂L/∂" + vid, "c-coral")]
        ]
      });
    });
    S.push({
      phase: "bwd", edges: ["c>zo"], hi: ["c", "zo"], val: {}, grad: { c: B.dc },
      text: "Backward 5. The bias is added, so its local slope is 1 and the slope passes through unchanged.",
      rows: [[bx(B.dzo, "upstream ∂L/∂zo", "c-coral"), op("&times;"), bx(1, "local ∂zo/∂c", "c-navy"), op("="), bx(B.dc, "∂L/∂c", "c-coral")]]
    });
    [0, 1].forEach(function (j) {
      var n = j + 1, zid = "z" + n, hid = "h" + n, g = {}; g[zid] = B.dz[j];
      var zero = B.loc[j] === 0;
      S.push({
        phase: "bwd", edges: [zid + ">" + hid], hi: [zid, hid], val: {}, grad: g,
        text: "Backward " + (5 + n) + ". Through the activation the local slope is " + (N.act === "tanh" ? "1 − h²" : N.act === "relu" ? "1 if z is above 0, else 0" : "h(1−h)") + "." + (zero ? " Here it is 0, so no slope flows back through this unit." : ""),
        rows: [[bx(B.dh[j], "upstream ∂L/∂" + hid, "c-coral"), op("&times;"), bx(B.loc[j], "local " + (N.act === "tanh" ? "1 &minus; " + hid + "<sup>2</sup>" : N.act === "relu" ? "ReLU slope" : hid + "(1&minus;" + hid + ")"), "c-navy"), op("="), bx(B.dz[j], "∂L/∂" + zid, "c-coral")]]
      });
    });
    // the six edges into z1 and z2, with the running sum for x1 and x2
    var xs = { x1: 0, x2: 0 }, stepNo = 8;
    [0, 1].forEach(function (j) {
      var n = j + 1, zid = "z" + n;
      [0, 1].forEach(function (i) {
        var xid = "x" + (i + 1), wid = "w" + n + (i + 1), g = {}, prev = xs[xid], contrib = B.dz[j] * N.W[j][i];
        xs[xid] = prev + contrib; g[wid] = B.dW[j][i]; g[xid] = xs[xid];
        var rows = [
          [bx(B.dz[j], "upstream ∂L/∂" + zid, "c-coral"), op("&times;"), bx(N.x[i], "local ∂" + zid + "/∂" + wid + " = " + xid, "c-navy"), op("="), bx(B.dW[j][i], "∂L/∂" + wid, "c-coral")],
          j === 0 ? [bx(B.dz[j], "upstream ∂L/∂" + zid, "c-coral"), op("&times;"), bx(N.W[j][i], "local ∂" + zid + "/∂" + xid + " = " + wid, "c-navy"), op("="), bx(contrib, "path through " + zid, "c-coral"), op("&rarr;"), bx(xs[xid], "∂L/∂" + xid + " so far", "c-coral")]
            : [bx(B.dz[j], "upstream ∂L/∂" + zid, "c-coral"), op("&times;"), bx(N.W[j][i], "local ∂" + zid + "/∂" + xid + " = " + wid, "c-navy"), op("="), bx(contrib, "path through " + zid, "c-coral"), op("+"), bx(prev, "first path, step " + (8 + i), "c-sand"), op("="), bx(xs[xid], "∂L/∂" + xid + " in total", "c-coral")]
        ];
        S.push({
          phase: "bwd", edges: [xid + ">" + zid], hi: [zid, wid, xid], val: {}, grad: g,
          text: "Backward " + (stepNo++) + ". " + (j === 0 ? "The weight " + wid + " gets slope × input. The input " + xid + " gets slope × weight; it is used in two places, so this is only its first path." : "Same again. The input " + xid + " is also used by z1, so its slope is the sum of two paths."),
          rows: rows
        });
      });
      var gb = {}; gb["b" + n] = B.db[j];
      S.push({
        phase: "bwd", edges: ["b" + n + ">" + zid], hi: [zid, "b" + n], val: {}, grad: gb,
        text: "Backward " + (stepNo++) + ". The bias is added, so its slope equals the slope arriving at " + zid + ".",
        rows: [[bx(B.dz[j], "upstream ∂L/∂" + zid, "c-coral"), op("&times;"), bx(1, "local ∂" + zid + "/∂b" + n, "c-navy"), op("="), bx(B.db[j], "∂L/∂b" + n, "c-coral")]]
      });
    });
    S.F = F; S.B = B;
    return S;
  }
  var NFWD = 7;

  // ---------- figure: the graph with a script ----------
  // opts: {full: show backward buttons, update and history}
  function graphFig(id, opts) {
    var full = !!opts.full, N = newNet(), NET0 = newNet(), steps = buildSteps(N), k = 0, timer = 0, hist = [lossOf(N)], sel = full ? "w11" : "x1";
    var box = Kit.$(id), svgNode = Kit.$(id + "-svg"), bar = Kit.$(id + "-bar"), status = Kit.$(id + "-status"), eqs = Kit.$(id + "-eqs"), edit = Kit.$(id + "-edit"), read = Kit.$(id + "-read");
    var eta = 0.3, vertical = false, lastLoss = null, flowStop = null, delay = 780;
    var api = { net: N, listeners: [], steps: function () { return steps; } };
    function changed() { api.listeners.forEach(function (f) { f(); }); }

    // controls
    var btnF = Kit.button(bar, "Forward", function () { play(NFWD); });
    var btnB = full ? Kit.button(bar, "Backward", function () { if (k < NFWD) k = NFWD; play(steps.length); }) : null;
    var btnS = Kit.button(bar, "Step", function () { stop(); if (k < steps.length && (full || k < NFWD)) { k++; render(true); } });
    var btnBack = Kit.button(bar, "Back", function () { stop(); if (k > 0) { k--; render(false); } });
    var btnU = full ? Kit.button(bar, "Update", update) : null;
    Kit.button(bar, "Reset", function () { stop(); var keepAct = N.act, keepY = N.y; N = newNet(); N.act = keepAct; N.y = keepY; api.net = N; hist = [lossOf(N)]; lastLoss = null; rebuild(); syncEditor(); changed(); });
    var actChips = Kit.chips(bar, ["tanh", "ReLU", "sigmoid"], 0, function (i) { stop(); N.act = ["tanh", "relu", "sigmoid"][i]; lastLoss = null; hist = [lossOf(N)]; rebuild(); changed(); });
    var yChips = Kit.chips(bar, ["y = 1, adult", "y = 0, juvenile"], 0, function (i) { stop(); N.y = i === 0 ? 1 : 0; lastLoss = null; hist = [lossOf(N)]; rebuild(); changed(); });
    if (full) Kit.chips(bar, ["Slow", "Fast"], 0, function (i) { delay = i ? 170 : 780; });
    var etaSl = full ? Kit.slider(edit, { label: "Learning rate η for Update", min: 0.05, max: 2, step: 0.05, value: eta, digits: 2 }, function (v) { eta = v; }) : null;
    var opts2 = ["x1", "x2"].concat(PARAMS).map(function (key) { return { value: key, label: key + (key === "x1" ? " (scaled length)" : key === "x2" ? " (scaled girth)" : "") }; });
    var selBox = Kit.select(edit, "Edit a value", opts2, sel, function (v) { sel = v; syncEditor(); draw(); });
    var valSl = Kit.slider(edit, { label: "Value", min: -2, max: 2, step: 0.01, value: getP(N, sel), digits: 2 }, function (v) {
      stop(); setP(N, sel, v); lastLoss = null; hist = [lossOf(N)]; rebuild(); changed();
    });
    function syncEditor() { selBox.value = sel; valSl.set(getP(N, sel)); }
    function stop() { if (timer) { clearTimeout(timer); timer = 0; } if (flowStop) { flowStop(); flowStop = null; } }
    function rebuild() {
      steps = buildSteps(N); k = 0;
      status.textContent = "Values changed or reset. Press Forward to run the network."; render(false); draw();
      if (full) drawHist();
    }
    function play(target) {
      stop();
      if (k >= target) { if (target <= NFWD && k >= NFWD) { status.textContent = "The forward pass is done. Press Backward."; } return; }
      (function tick() {
        if (k >= target) { timer = 0; return; }
        k++; render(true);
        timer = setTimeout(tick, delay);
      })();
    }
    function update() {
      stop();
      var B = steps.B, g = gradVec(N, B), before = steps.F.L;
      PARAMS.forEach(function (key, i) { setP(N, key, getP(N, key) - eta * g[i]); });
      var after = lossOf(N); hist.push(after); lastLoss = { before: before, after: after };
      steps = buildSteps(N); k = NFWD; render(false); draw(); drawHist(); syncEditor(); changed();
      status.textContent = "Update applied: all weights moved by −η × its slope (η = " + eta.toFixed(2) + "). The loss went from " + nf(before) + " to " + nf(after) + ". Press Backward to get the new slopes.";
    }

    // state shown after k steps
    function shown() {
      var val = {}, grad = {}, i;
      for (i = 0; i < k && i < steps.length; i++) { var s = steps[i], q; for (q in s.val) val[q] = s.val[q]; for (q in s.grad) grad[q] = s.grad[q]; }
      return { val: val, grad: grad };
    }
    function render(flow) {
      var cur = k > 0 ? steps[k - 1] : null;
      if (flow && cur) { draw(); animateFlow(cur); } else draw();
      eqs.innerHTML = "";
      if (cur) { cur.rows.forEach(function (r) { eqs.insertAdjacentHTML("beforeend", rowHTML(r)); }); }
      if (cur) status.textContent = cur.text + " (Step " + k + " of " + (full ? steps.length : NFWD) + ".)";
      else if (!status.textContent || /^Step|^Forward\./.test(status.textContent)) status.textContent = "Press Forward to run the network, or Step to go one node at a time.";
      var F = steps.F;
      btnBack.disabled = k === 0;
      if (btnU) btnU.disabled = false;
      var done = k >= steps.length;
      if (!full) { btnS.disabled = k >= NFWD; btnF.disabled = k >= NFWD; } else { btnS.disabled = done; }
      var cp = k >= NFWD ? F.p : null, cl = k >= NFWD ? F.L : null;
      read.innerHTML = "<span>Chance of adult p: <b>" + (cp == null ? "?" : nf(cp)) + "</b></span><span>Loss L: <b>" + (cl == null ? "?" : nf(cl)) + "</b></span>" +
        (full && lastLoss ? "<span>Last update: <b>" + nf(lastLoss.before) + " → " + nf(lastLoss.after) + "</b></span>" : "") +
        (full ? "<span>Updates so far: <b>" + (hist.length - 1) + "</b></span>" : "");
    }

    // drawing
    function draw() {
      var P = Kit.palette(), W = Kit.width(svgNode); vertical = W < 600;
      var lay = layout(vertical), pos = lay.pos, sh = shown(), cur = k > 0 ? steps[k - 1] : null;
      svgNode.setAttribute("viewBox", "0 0 " + lay.vb[0] + " " + lay.vb[1]);
      svgNode.setAttribute("aria-label", "Computation graph of a tiny network: two inputs, two hidden units, one output probability and a loss. " + (k >= NFWD ? "Forward pass done: p is " + nf(steps.F.p) + " and the loss is " + nf(steps.F.L) + "." : "Forward pass not finished.") + (k > NFWD ? " Backward steps shown: " + (k - NFWD) + " of " + (steps.length - NFWD) + "." : ""));
      svgNode.innerHTML = "";
      var defs = Kit.el("defs", {}, svgNode);
      [["ah-" + id, P.muted], ["af-" + id, P.teal], ["ab-" + id, P.coral]].forEach(function (a) {
        var m = Kit.el("marker", { id: a[0], viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" }, defs);
        Kit.el("path", { d: "M0,1 L9,5 L0,9 z", fill: a[1] }, m);
      });
      var activeEdges = {}; if (cur) cur.edges.forEach(function (e) { activeEdges[e] = true; });
      var hiSet = {}; if (cur) cur.hi.forEach(function (n) { hiSet[n] = true; });
      var gE = Kit.el("g", {}, svgNode), gC = Kit.el("g", {}, svgNode), gN = Kit.el("g", {}, svgNode);
      EDGES.forEach(function (e) {
        var A = pos[e.a], B = pos[e.b], key = e.a + ">" + e.b, on = !!activeEdges[key];
        var p0 = clipPt(A, B), p1 = clipPt(B, A);
        var col = on ? (cur.phase === "fwd" ? P.teal : P.coral) : P.muted, mk = on ? (cur.phase === "fwd" ? "af-" + id : "ab-" + id) : "ah-" + id;
        var ln = Kit.el("line", { x1: p0.x, y1: p0.y, x2: p1.x, y2: p1.y, stroke: col, "stroke-width": on ? 3.2 : 1.5, "marker-end": on && cur.phase === "bwd" ? "" : "url(#" + mk + ")", opacity: on ? 1 : 0.7 }, gE);
        if (on && cur.phase === "bwd") { ln.setAttribute("marker-start", "url(#" + mk + ")"); }
      });
      // parameter chips
      CHIPS.forEach(function (key) {
        var c = pos[key], hasG = sh.grad[key] != null, hot = !!hiSet[key];
        var g = Kit.el("g", { transform: "translate(" + c.x + "," + c.y + ")", tabindex: 0, role: "button", "aria-label": "Parameter " + key + " = " + nf(getP(N, key)) + (hasG ? ", slope " + nf(sh.grad[key]) : "") + ". Click to edit.", style: "cursor:pointer;outline:none" }, gC);
        Kit.el("rect", { x: -c.hw, y: -c.hh, width: 2 * c.hw, height: 2 * c.hh, rx: 7, fill: P.bg, stroke: hot ? (cur.phase === "bwd" ? P.coral : P.teal) : sel === key ? P.teal : P.navy, "stroke-width": hot || sel === key ? 3 : 1.6 }, g);
        var t1 = Kit.el("text", { x: 0, y: full ? -3 : 5, "text-anchor": "middle", "font-size": vertical ? 11.5 : 13, "font-weight": 600, fill: P.navy }, g); t1.textContent = key + " " + nf(getP(N, key));
        if (full) { var t2 = Kit.el("text", { x: 0, y: 13, "text-anchor": "middle", "font-size": vertical ? 11 : 12, "font-weight": 600, fill: hasG ? P.coral : P.muted }, g); t2.textContent = hasG ? "∂L " + nf(sh.grad[key]) : "∂L ?"; }
        g.addEventListener("click", function () { sel = key; syncEditor(); draw(); });
        g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = key; syncEditor(); draw(); } });
      });
      // nodes
      var titles = { x1: "x1 length", x2: "x2 girth", z1: "z1", z2: "z2", h1: "h1", h2: "h2", zo: "zo", p: "p", L: "loss L", y: "label" };
      Object.keys(titles).forEach(function (key) {
        var c = pos[key], isIn = key === "x1" || key === "x2", isY = key === "y";
        var hasV = isIn ? true : isY ? true : sh.val[key] != null, hasG = sh.grad[key] != null, hot = !!hiSet[key];
        var v = isIn ? getP(N, key) : isY ? N.y : sh.val[key];
        var g = Kit.el("g", { transform: "translate(" + c.x + "," + c.y + ")", style: (isIn || isY) ? "cursor:pointer;outline:none" : "outline:none" }, gN);
        if (isIn || isY) { g.setAttribute("tabindex", 0); g.setAttribute("role", "button"); g.setAttribute("aria-label", isY ? "Label y = " + N.y + ". Click to flip." : "Input " + key + " = " + nf(v) + ". Click to edit."); }
        Kit.el("rect", { x: -c.hw, y: -c.hh, width: 2 * c.hw, height: 2 * c.hh, rx: 9, fill: P.bg, stroke: hot ? (cur.phase === "bwd" ? P.coral : P.teal) : (isIn && sel === key) ? P.teal : hasV ? P.ink2 : P.muted, "stroke-width": hot || (isIn && sel === key) ? 3.2 : 1.6, "stroke-dasharray": hasV ? "" : "5 4" }, g);
        var th = Kit.el("text", { x: 0, y: isY ? -4 : -c.hh + 15, "text-anchor": "middle", "font-size": 11.5, fill: P.muted }, g); th.textContent = titles[key];
        var tv = Kit.el("text", { x: 0, y: isY ? 14 : 4, "text-anchor": "middle", "font-size": 15.5, "font-weight": 700, fill: hasV ? P.ink : P.muted }, g); tv.textContent = isY ? "y = " + N.y : hasV ? nf(v) : "?";
        if (full && !isY) { var tg = Kit.el("text", { x: 0, y: c.hh - 7, "text-anchor": "middle", "font-size": 12.5, "font-weight": 600, fill: hasG ? P.coral : P.muted }, g); tg.textContent = hasG ? "∂L " + nf(sh.grad[key]) : "∂L ?"; }
        if (isIn) { g.addEventListener("click", function () { sel = key; syncEditor(); draw(); }); g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sel = key; syncEditor(); draw(); } }); }
        if (isY) {
          var flip = function () { stop(); N.y = N.y === 1 ? 0 : 1; yChips.forEach(function (b, i) { b.setAttribute("aria-pressed", String((i === 0) === (N.y === 1))); }); lastLoss = null; hist = [lossOf(N)]; rebuild(); changed(); };
          g.addEventListener("click", flip); g.addEventListener("keydown", function (e) { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
        }
      });
      api.lay = lay;
    }
    function animateFlow(s) {
      if (flowStop) { flowStop(); flowStop = null; }
      var lay = api.lay, P = Kit.palette(), dots = [];
      s.edges.forEach(function (key) {
        var parts = key.split(">"), a = lay.pos[parts[0]], b = lay.pos[parts[1]];
        var from = s.phase === "fwd" ? a : b, to = s.phase === "fwd" ? b : a;
        var d = Kit.el("circle", { r: 6, fill: s.phase === "fwd" ? P.teal : P.coral, stroke: P.bg, "stroke-width": 2 }, svgNode);
        dots.push({ d: d, from: from, to: to });
      });
      flowStop = Kit.animate(520, function (p) {
        var e = Kit.ease(p);
        dots.forEach(function (o) { o.d.setAttribute("cx", o.from.x + (o.to.x - o.from.x) * e); o.d.setAttribute("cy", o.from.y + (o.to.y - o.from.y) * e); o.d.setAttribute("opacity", p < 1 ? 1 : 0); });
        if (p >= 1) dots.forEach(function (o) { if (o.d.parentNode) o.d.parentNode.removeChild(o.d); });
      });
    }
    function drawHist() {
      var hd = Kit.$(id + "-hist"); if (!hd) return;
      Kit.plot(hd, function (P) {
        return [{ x: hist.map(function (_, i) { return i; }), y: hist, mode: "lines+markers", line: { color: P.teal, width: 3 }, marker: { size: 9, color: P.teal, line: { color: P.paper, width: 2 } }, hovertemplate: "after %{x} updates: loss %{y:.4f}<extra></extra>" }];
      }, function (P) {
        return { margin: { l: 52, r: 16, t: 10, b: 44 }, xaxis: { title: { text: "updates applied" }, dtick: 1, range: [-0.3, Math.max(4, hist.length - 0.7)] }, yaxis: { title: { text: "loss L" }, rangemode: "tozero" } };
      });
    }
    window.addEventListener("resize", Kit.debounce(draw));
    Kit.onTheme(draw);
    rebuild(); status.textContent = "Press Forward to run the network, or Step to go one node at a time.";
    api.rebuild = rebuild; api.draw = draw; api.reset = function () { N = newNet(); api.net = N; };
    api.getNet = function () { return N; };
    api.getSteps = function () { return steps; };
    return api;
  }

  // ---------- 3. one value used twice ----------
  function pathsFig() {
    var node = Kit.$("pa-svg"), x = 3, mode = 2, h = 0.01;
    var bar = Kit.$("pa-bar"), ctrl = Kit.$("pa-controls");
    var sl = Kit.slider(ctrl, { label: "Value of x", min: -3, max: 4, step: 0.1, value: x, digits: 1 }, function (v) { x = v; draw(); });
    Kit.chips(bar, ["Path through p", "Path through q", "Both paths"], 2, function (i) { mode = i; draw(); });
    function draw() {
      var P = Kit.palette(), W = Kit.width(node), H = W < 520 ? 250 : 230, svg = d3.select(node);
      var p = 2 * x, q = x * x, y = p * q, dpx = 2, dqx = 2 * x, dyp = q, dyq = p;
      var path1 = dyp * dpx, path2 = dyq * dqx, total = path1 + path2;
      var y2 = 2 * Math.pow(x + h, 3), num = (y2 - y) / h;
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      var wide = W >= 520, cx = [W * 0.1, W * 0.4, W * 0.4, W * 0.88], cy = [H / 2, H * 0.22, H * 0.78, H / 2];
      var nodes = { x: [cx[0], cy[0]], p: [cx[1] + (wide ? W * 0.1 : 0), cy[1]], q: [cx[2] + (wide ? W * 0.1 : 0), cy[2]], y: [cx[3], cy[3]] };
      var defs = svg.append("defs");
      [["pa-ac", P.coral], ["pa-an", P.navy], ["pa-ag", P.muted]].forEach(function (a) { defs.append("marker").attr("id", a[0]).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto").append("path").attr("d", "M0,1 L9,5 L0,9 z").attr("fill", a[1]); });
      var E = [["x", "p", dpx, "∂p/∂x", 1], ["p", "y", dyp, "∂y/∂p", 1], ["x", "q", dqx, "∂q/∂x", 2], ["q", "y", dyq, "∂y/∂q", 2]];
      E.forEach(function (e) {
        var a = nodes[e[0]], b = nodes[e[1]], on = mode === 2 || mode === e[4] - 1;
        var col = !on ? P.muted : e[4] === 1 ? P.coral : P.navy, dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy), ux = dx / len, uy = dy / len;
        svg.append("line").attr("x1", a[0] + ux * 30).attr("y1", a[1] + uy * 22).attr("x2", b[0] - ux * 34).attr("y2", b[1] - uy * 22).attr("stroke", col).attr("stroke-width", on ? 3 : 1.4).attr("opacity", on ? 1 : 0.6).attr("marker-end", "url(#" + (on ? (e[4] === 1 ? "pa-ac" : "pa-an") : "pa-ag") + ")");
        var mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2 + (e[4] === 1 ? -9 : 17);
        svg.append("text").attr("x", mx).attr("y", my).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 600).attr("fill", col).attr("paint-order", "stroke").attr("stroke", P.paper).attr("stroke-width", 4).text(e[3] + " = " + Kit.fmt(e[2], 1));
      });
      [["x", "x", x], ["p", "p = 2x", p], ["q", "q = x²", q], ["y", "y = p × q", y]].forEach(function (nd) {
        var c = nodes[nd[0]], g = svg.append("g").attr("transform", "translate(" + c[0] + "," + c[1] + ")");
        g.append("rect").attr("x", -34).attr("y", -22).attr("width", 68).attr("height", 44).attr("rx", 9).attr("fill", P.bg).attr("stroke", P.ink2).attr("stroke-width", 1.6);
        g.append("text").attr("y", -6).attr("text-anchor", "middle").attr("font-size", 11).attr("fill", P.muted).text(nd[1]);
        g.append("text").attr("y", 13).attr("text-anchor", "middle").attr("font-size", 15).attr("font-weight", 700).attr("fill", P.ink).text(Kit.fmt(nd[2], 1));
      });
      var on1 = mode === 0 || mode === 2, on2 = mode === 1 || mode === 2;
      var parts = [];
      if (on1) parts.push(ebox("c-coral", Kit.fmt(dyp, 1) + " &times; " + Kit.fmt(dpx, 1) + " = " + Kit.fmt(path1, 1), "path through p: ∂y/∂p &times; ∂p/∂x"));
      if (on1 && on2) parts.push(eop("+"));
      if (on2) parts.push(ebox("c-navy", Kit.fmt(dyq, 1) + " &times; " + Kit.fmt(dqx, 1) + " = " + Kit.fmt(path2, 1), "path through q: ∂y/∂q &times; ∂q/∂x"));
      parts.push(eop("="));
      parts.push(ebox(mode === 2 ? "c-teal" : "", Kit.fmt(mode === 2 ? total : on1 ? path1 : path2, 1), mode === 2 ? "∂y/∂x, both paths added" : "only part of ∂y/∂x"));
      Kit.$("pa-eqn").innerHTML = parts.join("");
      Kit.$("pa-check").innerHTML = ebox("", Kit.fmt(num, 2), "(y(x + 0.01) &minus; y(x)) &divide; 0.01, the nudge test") + eop("&asymp;") + ebox("c-teal", Kit.fmt(6 * x * x, 1), "6x&sup2;, the slope of y = 2x&sup3;");
      node.setAttribute("aria-label", "A small graph: x feeds p equals 2x and q equals x squared, and y is p times q. At x = " + Kit.fmt(x, 1) + " the path through p contributes " + Kit.fmt(path1, 1) + " and the path through q contributes " + Kit.fmt(path2, 1) + ", a total of " + Kit.fmt(total, 1) + ".");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- 4. a whole layer at once ----------
  function layerFig() {
    var Wm = [[0.5, -0.3, 0.8], [0.2, 0.7, -0.4]];
    var delta = [-0.5, 0.2], hin = [1.0, 2.0, 0.0], extra = [{ d: [0.3, -0.4], h: [0.5, -1.0, 1.5] }, { d: [-0.2, 0.6], h: [1.5, 0.0, -0.5] }], batch = 0;
    var ctrl = Kit.$("ly-controls"), out = Kit.$("ly-out");
    var sl = [];
    sl.push(Kit.slider(ctrl, { label: "δ1: slope arriving at neuron 1", min: -1, max: 1, step: 0.1, value: delta[0], digits: 1 }, function (v) { delta[0] = v; draw(); }));
    sl.push(Kit.slider(ctrl, { label: "δ2: slope arriving at neuron 2", min: -1, max: 1, step: 0.1, value: delta[1], digits: 1 }, function (v) { delta[1] = v; draw(); }));
    [0, 1, 2].forEach(function (i) { sl.push(Kit.slider(ctrl, { label: "h" + (i + 1) + ": input to the layer", min: -2, max: 2, step: 0.1, value: hin[i], digits: 1 }, function (v) { hin[i] = v; draw(); })); });
    Kit.chips(Kit.$("ly-bar"), ["One shark", "Three sharks, averaged"], 0, function (i) { batch = i; draw(); });
    function outer(d, h) { return d.map(function (a) { return h.map(function (b) { return a * b; }); }); }
    function heat(M, P, lab, cls) {
      var mx = 1.6, t = '<table class="ly-mat ' + (cls || "") + '" aria-label="' + lab + '"><tbody>';
      M.forEach(function (r) {
        t += "<tr>"; r.forEach(function (v) { var a = clamp(Math.abs(v) / mx, 0, 1), c = v >= 0 ? P.navy : P.coral; t += '<td style="background:' + rgba(c, 0.12 + 0.55 * a) + '">' + Kit.fmt(v, 2) + "</td>"; }); t += "</tr>";
      });
      return t + "</tbody></table>";
    }
    function draw() {
      var P = Kit.palette(), sets = [{ d: delta, h: hin }].concat(batch ? extra : []), mats = sets.map(function (s) { return outer(s.d, s.h); });
      var avg = [0, 1].map(function (i) { return [0, 1, 2].map(function (j) { return mats.reduce(function (a, m) { return a + m[i][j]; }, 0) / mats.length; }); });
      var WtD = [0, 1, 2].map(function (j) { return Wm[0][j] * delta[0] + Wm[1][j] * delta[1]; });
      var html = '<div class="ly-row">';
      html += '<div class="ly-col"><div class="ly-h">δ (one per neuron)</div>' + heat([[delta[0]], [delta[1]]], P, "delta") + "</div>";
      html += '<div class="ly-op">&times;</div><div class="ly-col"><div class="ly-h">hᵀ (inputs)</div>' + heat([hin], P, "h transposed") + "</div>";
      html += '<div class="ly-op">=</div><div class="ly-col"><div class="ly-h">∂L/∂W for this shark</div>' + heat(mats[0], P, "outer product") + "</div></div>";
      if (batch) {
        html += '<div class="ly-row"><div class="ly-col"><div class="ly-h">shark 2: δ hᵀ</div>' + heat(mats[1], P, "outer product of shark 2") + "</div>" + '<div class="ly-op">+</div>' +
          '<div class="ly-col"><div class="ly-h">shark 3: δ hᵀ</div>' + heat(mats[2], P, "outer product of shark 3") + "</div>" + '<div class="ly-op">+ shark 1, divide by 3 =</div>' +
          '<div class="ly-col"><div class="ly-h">average: ∂L/∂W for the batch</div>' + heat(avg, P, "average gradient") + "</div></div>";
      }
      html += '<div class="ly-row"><div class="ly-col"><div class="ly-h">W (made-up, fixed)</div>' + heat(Wm, P, "weights") + '</div><div class="ly-op">transposed, times δ &rarr;</div><div class="ly-col"><div class="ly-h">Wᵀδ: slope sent to the three inputs</div>' + heat([WtD], P, "slope to inputs") + "</div></div>";
      out.innerHTML = html;
      var e = mats[0];
      Kit.$("ly-eqn").innerHTML = ebox("c-coral", Kit.fmt(delta[0], 1), "δ1") + eop("&times;") + ebox("c-navy", Kit.fmt(hin[1], 1), "h2") + eop("=") + ebox("c-coral", Kit.fmt(e[0][1], 2), "∂L/∂W row 1, column 2") +
        eop("&nbsp;&nbsp;|&nbsp;&nbsp;") + ebox("c-navy", Kit.fmt(Wm[0][1], 1), "W row 1, col 2") + eop("&times;") + ebox("c-coral", Kit.fmt(delta[0], 1), "δ1") + eop("+") + ebox("c-navy", Kit.fmt(Wm[1][1], 1), "W row 2, col 2") + eop("&times;") + ebox("c-coral", Kit.fmt(delta[1], 1), "δ2") + eop("=") + ebox("c-teal", Kit.fmt(WtD[1], 2), "slope sent to input 2");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 5. a chain of one-neuron layers: vanishing and exploding ----------
  function vanishFig() {
    var n = 12, w = 1.0, act = "sigmoid", a0 = 0.5, ctrl = Kit.$("vn-controls");
    Kit.slider(ctrl, { label: "Number of layers", min: 2, max: 30, step: 1, value: n, digits: 0 }, function (v) { n = v; draw(); });
    Kit.slider(ctrl, { label: "Weight in all layers", min: 0.2, max: 5, step: 0.05, value: w, digits: 2 }, function (v) { w = v; draw(); });
    Kit.chips(Kit.$("vn-bar"), ["sigmoid", "tanh", "ReLU"], 0, function (i) { act = ["sigmoid", "tanh", "relu"][i]; draw(); });
    var A = { sigmoid: ACT.sigmoid, tanh: ACT.tanh, relu: ACT.relu };
    // chain: h[0] = a0 is the input; layer k computes z[k] = w * h[k-1] and h[k] = g(z[k]).
    // slope[k] = d h[n] / d h[k]: the slope that arrives at the output of layer k on the way back.
    function chain() {
      var h = [a0], z = [], loc = [], slope = [], i, a = A[act];
      for (i = 1; i <= n; i++) { z[i] = w * h[i - 1]; h[i] = a.g(z[i]); loc[i] = a.dg(h[i], z[i]); }
      slope[n] = 1; for (i = n - 1; i >= 1; i--) slope[i] = slope[i + 1] * w * loc[i + 1];
      return { slope: slope, h: h, loc: loc };
    }
    function draw() {
      var C = chain(), xs = [], ys = [], i;
      for (i = 1; i <= n; i++) { xs.push(i); ys.push(Math.max(C.slope[i], 1e-300)); }
      var first = C.slope[1], factor = w * (C.loc[Math.min(n, 2)] || 0);
      Kit.plot("vn-plot", function (P) {
        return [{ type: "bar", x: xs, y: ys, marker: { color: ys.map(function (v) { return v < 1e-3 ? P.coral : v > 1e3 ? P.sand : P.teal; }) }, hovertemplate: "layer %{x}: slope %{y:.2e}<extra></extra>" }];
      }, function (P) {
        return { margin: { l: 64, r: 16, t: 10, b: 46 }, xaxis: { title: { text: "layer (1 is nearest the input)" }, dtick: n > 16 ? 5 : 1 }, yaxis: { type: "log", title: { text: "slope arriving at this layer" }, exponentformat: "power" }, bargap: 0.2 };
      });
      Kit.$("vn-eqn").innerHTML = ebox("c-navy", Kit.fmt(w, 2), "weight w") + eop("&times;") + ebox("c-navy", Kit.fmt(C.loc[Math.min(n, 2)] || 0, 3), "local slope g&prime;, layer 2") + eop("=") + ebox("c-teal", Kit.fmt(factor, 3), "factor per layer") +
        eop("&rarr;") + ebox("c-coral", sci(first), "slope that reaches layer 1 (the output layer gets 1)");
      Kit.setText("vn-status", first < 1e-3 ? "Vanishing: layer 1 receives " + sci(first) + " of the slope the output layer starts with, so its weights barely move." : first > 1e3 ? "Exploding: layer 1 receives " + sci(first) + " times the slope the output layer starts with, so one step would throw its weights far away." : "Balanced: the slope keeps a similar size from the output layer back to layer 1.");
      Kit.$("vn-plot").setAttribute("aria-label", "Bar chart on a log axis of the slope arriving at " + n + " layers for " + act + " units with weight " + Kit.fmt(w, 2) + ". Layer 1 receives " + sci(first) + " of the slope at the output layer.");
    }
    draw(); Kit.onTheme(draw);
  }

  // ---------- 6. gradient checking ----------
  function checkFig(g2) {
    var key = "w11", hexp = -5, central = true, bug = "none", ctrl = Kit.$("ck-controls"), bar = Kit.$("ck-bar");
    var slH = Kit.slider(ctrl, { label: "Nudge size h, as a power of ten", min: -15, max: 0, step: 0.25, value: hexp, digits: 2, show: function (v) { return "1e" + Kit.fmt(v, 2); } }, function (v) { hexp = v; draw(); });
    Kit.select(ctrl, "Parameter for the error curve", PARAMS.map(function (k) { return { value: k, label: k }; }), key, function (v) { key = v; draw(); });
    Kit.select(ctrl, "Bug in the backward code", [{ value: "none", label: "none" }, { value: "noact", label: "forgot the hidden activation slope" }, { value: "sign", label: "wrong sign on p − y" }, { value: "swap", label: "used v instead of h for the slope of v" }], bug, function (v) { bug = v; draw(); });
    Kit.chips(bar, ["Central: (L(w+h) − L(w−h)) / 2h", "Forward: (L(w+h) − L(w)) / h"], 0, function (i) { central = i === 0; draw(); });
    function relErr(a, b) { var d = Math.abs(a) + Math.abs(b); return d === 0 ? 0 : Math.abs(a - b) / d; }
    function verdict(e) { return e < 1e-7 ? "great" : e < 1e-5 ? "fine" : e < 1e-3 ? "look closer" : "wrong"; }
    function draw() {
      var N = g2.getNet(), F = forward(N), B = backward(N, F, bug), bp = gradVec(N, B), h = Math.pow(10, hexp), i;
      var num = PARAMS.map(function (k) { return numGrad(N, k, h, central); });
      var errs = PARAMS.map(function (k, j) { return relErr(bp[j], num[j]); });
      var va = 0, vb = 0, vd = 0; for (i = 0; i < 9; i++) { va += bp[i] * bp[i]; vb += num[i] * num[i]; vd += Math.pow(bp[i] - num[i], 2); }
      var whole = Math.sqrt(vd) / (Math.sqrt(va) + Math.sqrt(vb) || 1);
      var idx = PARAMS.indexOf(key), hs = [], ec = [], ef = [], e;
      for (e = -15; e <= 0.001; e += 0.25) {
        var hh = Math.pow(10, e);
        hs.push(hh); ec.push(Math.max(relErr(bp[idx], numGrad(N, key, hh, true)), 1e-17)); ef.push(Math.max(relErr(bp[idx], numGrad(N, key, hh, false)), 1e-17));
      }
      Kit.plot("ck-plot", function (P) {
        return [
          { x: hs, y: ef, mode: "lines", name: "forward", line: { color: P.sand, width: 3, dash: "dot" }, hovertemplate: "h = %{x:.0e}: error %{y:.1e}<extra>forward</extra>" },
          { x: hs, y: ec, mode: "lines", name: "central", line: { color: P.navy, width: 3 }, hovertemplate: "h = %{x:.0e}: error %{y:.1e}<extra>central</extra>" },
          { x: [h], y: [Math.max(errs[idx], 1e-17)], showlegend: false, mode: "markers+text", text: ["your h"], textposition: h > 0.003 ? "top left" : "top center", marker: { size: 13, color: P.coral, line: { color: P.paper, width: 2 } }, textfont: { color: P.coral }, hovertemplate: "h = %{x:.0e}: error %{y:.1e}<extra></extra>" }
        ];
      }, function (P) {
        return {
          margin: { l: 66, r: 16, t: 10, b: 50 }, showlegend: true, legend: { orientation: "h", y: 1.12, x: 0 },
          xaxis: { type: "log", title: { text: "nudge size h" }, exponentformat: "power", range: [-15.2, 0.2] },
          yaxis: { type: "log", title: { text: "relative error for " + key }, exponentformat: "power", range: [-17.3, 0.3] },
          annotations: [{ x: -2.5, y: -1.2, xref: "x", yref: "y", text: "h too large: the curve bends", showarrow: false, font: { size: 12, color: P.muted } }, { x: -13.2, y: -14.2, xref: "x", yref: "y", text: "h too small: digits cancel", showarrow: false, font: { size: 12, color: P.muted } }]
        };
      });
      var rows = "";
      PARAMS.forEach(function (k, j) { rows += "<tr><td>" + k + "</td><td>" + nf(bp[j]) + "</td><td>" + nf(num[j]) + "</td><td>" + sci(errs[j]) + "</td><td>" + verdict(errs[j]) + "</td></tr>"; });
      Kit.$("ck-tbody").innerHTML = rows;
      Kit.$("ck-eqn").innerHTML = ebox("c-coral", nf(bp[idx]), "backprop slope of " + key) + eop("vs") + ebox("c-navy", nf(num[idx]), (central ? "central" : "forward") + " estimate at h = " + sci(h)) + eop("&rarr;") + ebox(verdict(errs[idx]) === "great" || verdict(errs[idx]) === "fine" ? "c-teal" : "c-coral", sci(errs[idx]), "|a &minus; b| &divide; (|a| + |b|)");
      Kit.setText("ck-status", "All nine slopes together: relative error " + sci(whole) + ", so the check says " + verdict(whole) + ". " + (bug !== "none" ? "A bug is switched on: the error stays large whatever h is." : ""));
      Kit.$("ck-cost").innerHTML = "<span>Passes through the network for this check: <b>" + (central ? 18 : 10) + "</b> (" + (central ? "two" : "one extra") + " per weight" + (central ? "" : ", plus the base run") + ")</span><span>Passes for backprop: <b>1 forward + 1 backward</b></span>";
      Kit.$("ck-plot").setAttribute("aria-label", "Log-log plot of the relative error of a finite-difference slope for " + key + " against the nudge size h. The error is large for big h, falls to a minimum near 1e-5 for the central difference, and rises again for tiny h.");
    }
    g2.listeners.push(draw);
    draw(); Kit.onTheme(draw);
  }

  // ---------- start ----------
  function init() {
    var g1 = graphFig("g1", { full: false });
    var g2 = graphFig("g2", { full: true });
    pathsFig(); layerFig(); vanishFig(); checkFig(g2);
  }
  if (typeof document !== "undefined") document.addEventListener("DOMContentLoaded", init);
  if (typeof module !== "undefined") module.exports = { newNet: newNet, forward: forward, backward: backward, gradVec: gradVec, numGrad: numGrad, lossOf: lossOf, cloneNet: cloneNet, PARAMS: PARAMS, setP: setP, getP: getP };
})();
