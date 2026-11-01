// Chapter 0.1 Python: figures. Plain JS on top of Kit. Pure logic is at the top and exported for tests.
(function () {
  "use strict";

  // ======================= pure logic (no DOM) =======================
  function prod(a) { return a.reduce(function (x, y) { return x * y; }, 1); }
  function dimsText(d) { return d.length === 0 ? "()" : d.length === 1 ? "(" + d[0] + ",)" : "(" + d.join(", ") + ")"; }
  function parseDims(s) { var m = s.replace(/[()\s]/g, ""); return m === "" ? [] : m.split(",").filter(function (x) { return x !== ""; }).map(Number); }

  // Broadcasting rule: compare dimensions from the right.
  function broadcast(a, b) {
    var n = Math.max(a.length, b.length), rows = [], res = [], ok = true;
    for (var j = 1; j <= n; j++) {
      var x = a.length - j >= 0 ? a[a.length - j] : null, y = b.length - j >= 0 ? b[b.length - j] : null;
      var xv = x === null ? 1 : x, yv = y === null ? 1 : y, verdict, r = null, good = true;
      if (x !== null && y !== null && x === y) { r = x; verdict = "equal, no stretching needed"; }
      else if (xv === 1 && x !== null) { r = yv; verdict = "A is 1, so it is stretched to " + yv; }
      else if (yv === 1 && y !== null) { r = xv; verdict = "B is 1, so it is stretched to " + xv; }
      else if (x === null) { r = yv; verdict = "A has no such dimension. It counts as 1 and is stretched to " + yv; }
      else if (y === null) { r = xv; verdict = "B has no such dimension. It counts as 1 and is stretched to " + xv; }
      else { good = false; ok = false; verdict = "sizes " + xv + " and " + yv + " differ and neither is 1"; }
      rows.push({ j: j, a: x, b: y, r: r, good: good, verdict: verdict });
      if (good) res.unshift(r);
    }
    return { ok: ok, dims: ok ? res : null, rows: rows };
  }

  // Matrix product rule, following NumPy's matmul for up to 3 dimensions.
  function matmulDims(a, b) {
    if (a.length === 0 || b.length === 0) return { ok: false, msg: "The @ operator needs arrays with at least one dimension. Use * for a single number." };
    var A = a.length === 1 ? [1, a[0]] : a.slice(), B = b.length === 1 ? [b[0], 1] : b.slice();
    var k1 = A[A.length - 1], k2 = B[B.length - 2];
    if (k1 !== k2) return { ok: false, msg: "The last dimension of A is " + k1 + ", but the second-to-last dimension of B is " + k2 + ". These two must be equal.", k1: k1, k2: k2 };
    var bb = broadcast(A.slice(0, -2), B.slice(0, -2));
    if (!bb.ok) return { ok: false, msg: "The leading (batch) dimensions of A and B cannot be broadcast together." };
    var res = bb.dims.concat([A[A.length - 2], B[B.length - 1]]);
    if (a.length === 1) res.splice(res.length - 2, 1);
    if (b.length === 1) res.splice(res.length - 1, 1);
    return { ok: true, dims: res, A: A, B: B, k: k1, batch: bb.dims };
  }

  // Row-major helpers.
  function strides(d) { var s = [], p = 1; for (var i = d.length - 1; i >= 0; i--) { s[i] = p; p *= d[i]; } return s; }
  function unflat(i, d) { var idx = []; for (var j = d.length - 1; j >= 0; j--) { idx[j] = i % d[j]; i = Math.floor(i / d[j]); } return idx; }
  function flat(idx, d) { var s = strides(d), t = 0; for (var j = 0; j < d.length; j++) t += idx[j] * s[j]; return t; }
  function makeValues(d, kind) { var n = prod(d), out = []; for (var i = 0; i < n; i++) out.push(kind === "a" ? ((i * 3) % 5) + 1 : ((i * 2) % 3) + 1); return out; }

  // Elementwise operation with broadcasting: returns result values and, for each operand, which result cells are copies.
  function elementwise(da, va, db, vb, op) {
    var bc = broadcast(da, db);
    if (!bc.ok) return bc;
    var R = bc.dims, n = prod(R), out = [], copyA = [], copyB = [], shA = [], shB = [];
    var pa = new Array(R.length - da.length).fill(1).concat(da), pb = new Array(R.length - db.length).fill(1).concat(db);
    for (var i = 0; i < n; i++) {
      var idx = unflat(i, R), ia = [], ib = [], ca = false, cb = false;
      for (var j = 0; j < R.length; j++) {
        ia.push(pa[j] === 1 ? 0 : idx[j]); ib.push(pb[j] === 1 ? 0 : idx[j]);
        if (pa[j] === 1 && R[j] > 1 && idx[j] > 0) ca = true;
        if (pb[j] === 1 && R[j] > 1 && idx[j] > 0) cb = true;
      }
      var x = va[flat(ia, pa)], y = vb[flat(ib, pb)];
      out.push(op === "*" ? x * y : x + y); copyA.push(ca); copyB.push(cb); shA.push(x); shB.push(y);
    }
    return { ok: true, dims: R, values: out, copyA: copyA, copyB: copyB, shownA: shA, shownB: shB, rows: bc.rows };
  }

  // Matrix product values. Operands are normalised to at least 2 dimensions; `cells` maps a result cell to the inputs it used.
  function matmul(da, va, db, vb) {
    var md = matmulDims(da, db);
    if (!md.ok) return md;
    var A = md.A, B = md.B, nb = md.batch.length, m = A[A.length - 2], k = md.k, n = B[B.length - 1];
    var batchA = A.slice(0, -2), batchB = B.slice(0, -2);
    var pa = new Array(nb - batchA.length).fill(1).concat(batchA), pb = new Array(nb - batchB.length).fill(1).concat(batchB);
    var nbatch = prod(md.batch), vals = [], uses = [];
    for (var q = 0; q < nbatch; q++) {
      var bi = unflat(q, md.batch), ia = bi.map(function (v, j) { return pa[j] === 1 ? 0 : v; }), ib = bi.map(function (v, j) { return pb[j] === 1 ? 0 : v; });
      var offA = flat(ia, batchA.length ? pa : []) * m * k, offB = flat(ib, batchB.length ? pb : []) * k * n;
      for (var i = 0; i < m; i++) for (var j = 0; j < n; j++) {
        var s = 0, terms = [];
        for (var t = 0; t < k; t++) { var x = va[offA + i * k + t], y = vb[offB + t * n + j]; s += x * y; terms.push(x + "×" + y); }
        vals.push(s); uses.push({ batchA: ia.length ? flat(ia, pa) : 0, batchB: ib.length ? flat(ib, pb) : 0, i: i, j: j, text: terms.join(" + ") + " = " + s });
      }
    }
    return { ok: true, dims: md.dims, values: vals, uses: uses, norm: { A: A, B: B, m: m, k: k, n: n, nbatch: nbatch, batchA: batchA, batchB: batchB } };
  }

  // Python slice semantics for a list of length n.
  function sliceIdx(start, stop, step, n) {
    step = step == null ? 1 : step;
    var lo, hi;
    if (step > 0) { lo = start == null ? 0 : start < 0 ? Math.max(0, start + n) : Math.min(start, n); hi = stop == null ? n : stop < 0 ? Math.max(0, stop + n) : Math.min(stop, n); }
    else { lo = start == null ? n - 1 : start < 0 ? Math.max(-1, start + n) : Math.min(start, n - 1); hi = stop == null ? -1 : stop < 0 ? Math.max(-1, stop + n) : Math.min(stop, n - 1); }
    var out = [];
    if (step > 0) for (var i = lo; i < hi; i += step) out.push(i); else for (var j = lo; j > hi; j += step) out.push(j);
    return out;
  }
  // Parse one axis of an index such as "1:4", ":", "-1", "::2". Returns {int:i} or {list:[...]} or null.
  function parseAxis(s, n) {
    s = s.trim();
    if (s === "") return null;
    if (/^-?\d+$/.test(s)) { var v = parseInt(s, 10); if (v < -n || v >= n) return { err: "index " + v + " is out of range for an axis of size " + n }; return { int: v < 0 ? v + n : v }; }
    var m = s.match(/^(-?\d*)\s*:\s*(-?\d*)\s*(?::\s*(-?\d*))?$/);
    if (!m) return null;
    var f = function (t) { return t === undefined || t === "" ? null : parseInt(t, 10); };
    var step = f(m[3]); if (step === 0) return { err: "the step cannot be 0" };
    return { list: sliceIdx(f(m[1]), f(m[2]), step, n) };
  }

  function fmtNum(x) { return String(+Number(x).toPrecision(7)); }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { broadcast: broadcast, matmulDims: matmulDims, elementwise: elementwise, matmul: matmul, sliceIdx: sliceIdx, parseAxis: parseAxis, parseDims: parseDims, dimsText: dimsText, makeValues: makeValues };
  }
  if (typeof document === "undefined") return;

  // ======================= helpers =======================
  var H = function (tag, attrs, parent, text) { return Kit.html(tag, attrs, parent, text); };
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function codeLines(box, lines, state) {
    clear(box);
    lines.forEach(function (t, i) {
      var d = H("span", { class: "ln" + (state && state[i] ? " " + state[i] : "") }, box);
      if (t === "") { d.appendChild(document.createTextNode("\u00a0")); return; }
      var m = t.match(/^(.*?)(\s+#.*)?$/);
      d.appendChild(document.createTextNode(m[1]));
      if (m[2]) H("span", { class: "cm" }, d, m[2]);
    });
  }
  function varBox(parent, name, value, type, cls) {
    var b = H("div", { class: "vb" + (cls ? " " + cls : "") }, parent);
    H("small", {}, b, name); H("b", {}, b, value); if (type) H("i", {}, b, type);
    return b;
  }

  document.addEventListener("DOMContentLoaded", function () {
    values(); loops(); slices(); functions(); containers(); venn(); classes(); tables(); arrays(); broadcastFig(); seeds(); autograd();
  });

  // ======================= 1. values and decisions =======================
  function values() {
    var kind = 0, len, c1, c2;
    var kinds = ["float", "int", "text"];
    len = Kit.slider(Kit.$("vf-controls"), { label: "length, cm", min: 20, max: 120, step: 0.5, value: 62.5, digits: 1 }, draw);
    c1 = Kit.slider(Kit.$("vf-controls"), { label: "first cut-off", min: 30, max: 60, step: 1, value: 45, digits: 0 }, draw);
    c2 = Kit.slider(Kit.$("vf-controls"), { label: "second cut-off", min: 55, max: 95, step: 1, value: 70, digits: 0 }, draw);
    Kit.chips(Kit.$("vf-bar"), ["a decimal", "a whole number", "text in quotes"], 0, function (i) { kind = i; draw(); });
    function draw() {
      var L = len.get(), a = c1.get(), b = c2.get(), shown, num;
      if (kind === 0) { shown = Number.isInteger(L) ? L.toFixed(1) : String(L); num = L; }
      else if (kind === 1) { shown = String(Math.round(L)); num = Math.round(L); }
      else { shown = '"' + (Number.isInteger(L) ? L.toFixed(1) : String(L)) + '"'; num = null; }
      var lines = ["length = " + shown, "if length < " + a + ":", '    stage = "juvenile"', "elif length < " + b + ":", '    stage = "sub-adult"', "else:", '    stage = "adult"', "print(stage)"];
      var st = [], stage = null, err = null, tests = [];
      st[0] = "on";
      if (num === null) { st[1] = "err"; err = "TypeError: '<' not supported between instances of 'str' and 'int'"; for (var i = 2; i < 8; i++) st[i] = "skip"; }
      else {
        var t1 = num < a; tests.push(["length < " + a, t1]);
        st[1] = "on";
        if (t1) { st[2] = "on"; stage = "juvenile"; st[3] = st[4] = st[5] = st[6] = "skip"; }
        else {
          st[2] = "skip"; st[3] = "on"; var t2 = num < b; tests.push(["length < " + b, t2]);
          if (t2) { st[4] = "on"; stage = "sub-adult"; st[5] = st[6] = "skip"; }
          else { st[4] = "skip"; st[5] = "on"; st[6] = "on"; stage = "adult"; }
        }
        st[7] = "on";
      }
      codeLines(Kit.$("vf-code"), lines, st);
      var vars = Kit.$("vf-vars"); clear(vars);
      varBox(vars, "length", shown, kinds[kind] === "text" ? "str" : kinds[kind]);
      if (stage) varBox(vars, "stage", '"' + stage + '"', "str", "new"); else varBox(vars, "stage", err ? "not set" : "", "", err ? "gone" : "");
      var ul = Kit.$("vf-tests"); clear(ul);
      tests.forEach(function (t) { var li = H("li", {}, ul); li.appendChild(document.createTextNode(t[0] + " is ")); H("span", { class: t[1] ? "t" : "f" }, li, t[1] ? "True" : "False"); });
      if (err) H("li", { class: "f" }, ul, err);
      Kit.$("vf-status").textContent = err ? "Python stops at line 2 with an error, so stage is never set." : "Output: " + stage;
    }
    draw();
  }

  // ======================= 2. lists and loops =======================
  var LENS = [42.4, 45.1, 74.5, 41.0, 36.1, 44.0, 64.3, 46.0];
  function cellRow(box, vals, cls) {
    clear(box);
    return vals.map(function (v, i) {
      var c = H("div", { class: "c" }, box); H("small", {}, c, "index " + i); H("b", {}, c, v.toFixed(1)); return c;
    });
  }
  function loops() {
    var progs = {
      total: { name: "Total", lines: ["total = 0", "for length in lengths:", "    total = total + length"] },
      count: { name: "Count over 60", lines: ["count = 0", "for length in lengths:", "    if length > 60:", "        count = count + 1"] },
      longest: { name: "Longest", lines: ["longest = lengths[0]", "for length in lengths:", "    if length > longest:", "        longest = length"] },
      keep: { name: "Keep over 60", lines: ["big = []", "for length in lengths:", "    if length > 60:", "        big.append(length)"] }
    };
    var keys = ["total", "count", "longest", "keep"], cur = "total", frames = [], pos = 0, timer = 0;
    function r1(x) { return Math.round(x * 10) / 10; }
    function build(k) {
      var f = [], kept = [];
      function push(line, vars, cell, text) { f.push({ line: line, vars: JSON.parse(JSON.stringify(vars)), cell: cell, kept: kept.slice(), text: text }); }
      if (k === "total") {
        var t = 0; push(0, { total: 0 }, -1, "Start the total at 0.");
        LENS.forEach(function (x, i) {
          push(1, { total: r1(t), length: x }, i, "The loop takes the item at index " + i + ": length = " + x.toFixed(1) + ".");
          var nt = r1(t + x);
          push(2, { total: nt, length: x }, i, "total = " + r1(t).toFixed(1) + " + " + x.toFixed(1) + " = " + nt.toFixed(1) + ".");
          t = nt;
        });
        push(-1, { total: r1(t) }, -1, "The list is used up. total = 393.4. Python prints 393.40000000000003 here, because decimals are stored in binary. Round before you show it.");
      } else if (k === "count") {
        var c = 0; push(0, { count: 0 }, -1, "Start the count at 0.");
        LENS.forEach(function (x, i) {
          push(1, { count: c, length: x }, i, "The loop takes the item at index " + i + ": length = " + x.toFixed(1) + ".");
          var yes = x > 60; push(2, { count: c, length: x }, i, x.toFixed(1) + " > 60 is " + (yes ? "True." : "False, so the body is skipped."));
          if (yes) { c++; kept.push(i); push(3, { count: c, length: x }, i, "count = " + (c - 1) + " + 1 = " + c + "."); }
        });
        push(-1, { count: c }, -1, "The list is used up. count = " + c + ".");
      } else if (k === "longest") {
        var m = LENS[0]; push(0, { longest: m }, -1, "Start with the first item as the longest so far.");
        kept.push(0);
        LENS.forEach(function (x, i) {
          push(1, { longest: m, length: x }, i, "The loop takes the item at index " + i + ": length = " + x.toFixed(1) + ".");
          var yes = x > m; push(2, { longest: m, length: x }, i, x.toFixed(1) + " > " + m.toFixed(1) + " is " + (yes ? "True." : "False, so longest stays."));
          if (yes) { m = x; kept.length = 0; kept.push(i); push(3, { longest: m, length: x }, i, "longest = " + x.toFixed(1) + "."); }
        });
        push(-1, { longest: m }, -1, "The list is used up. longest = " + m.toFixed(1) + ".");
      } else {
        var big = []; push(0, { big: "[]" }, -1, "Start with an empty list.");
        LENS.forEach(function (x, i) {
          push(1, { big: "[" + big.map(function (v) { return v.toFixed(1); }).join(", ") + "]", length: x }, i, "The loop takes the item at index " + i + ": length = " + x.toFixed(1) + ".");
          var yes = x > 60; push(2, { big: "[" + big.map(function (v) { return v.toFixed(1); }).join(", ") + "]", length: x }, i, x.toFixed(1) + " > 60 is " + (yes ? "True." : "False, so the body is skipped."));
          if (yes) { big.push(x); kept.push(i); push(3, { big: "[" + big.map(function (v) { return v.toFixed(1); }).join(", ") + "]", length: x }, i, "append(" + x.toFixed(1) + ") adds it to the end of the new list."); }
        });
        push(-1, { big: "[" + big.map(function (v) { return v.toFixed(1); }).join(", ") + "]" }, -1, "The list is used up. big has " + big.length + " items.");
      }
      return f;
    }
    var bar = Kit.$("lf-bar");
    var chipBox = H("span", {}, bar);
    Kit.chips(chipBox, keys.map(function (k) { return progs[k].name; }), 0, function (i) { stop(); cur = keys[i]; frames = build(cur); pos = 0; draw(); });
    var bStep = Kit.button(bar, "Step", function () { stop(); step(); });
    var bPlay = Kit.button(bar, "Play", function () { if (timer) { stop(); return; } if (pos >= frames.length - 1) pos = 0; bPlay.textContent = "Pause"; timer = setInterval(function () { if (pos >= frames.length - 1) { stop(); return; } step(); }, 650); });
    Kit.button(bar, "Reset", function () { stop(); pos = 0; draw(); });
    function stop() { if (timer) clearInterval(timer); timer = 0; bPlay.textContent = "Play"; }
    function step() { if (pos < frames.length - 1) { pos++; draw(); } }
    frames = build(cur);
    function draw() {
      var fr = frames[pos], cells = cellRow(Kit.$("lf-cells"), LENS);
      cells.forEach(function (c, i) { if (i === fr.cell) c.classList.add("cur"); else if (fr.kept.indexOf(i) >= 0) c.classList.add("kept"); else if (fr.cell >= 0 && i < fr.cell || fr.cell < 0 && pos > 0) c.classList.add("done"); });
      var st = []; if (fr.line >= 0) st[fr.line] = "on";
      codeLines(Kit.$("lf-code"), progs[cur].lines, st);
      var vb = Kit.$("lf-vars"); clear(vb);
      var names = Object.keys(fr.vars);
      names.forEach(function (n) { varBox(vb, n, typeof fr.vars[n] === "number" ? (fr.vars[n] === 0 || n === "count" ? String(fr.vars[n]) : fr.vars[n].toFixed(1)) : fr.vars[n], "", (pos > 0 && frames[pos - 1].vars[n] !== fr.vars[n]) ? "new" : ""); });
      Kit.$("lf-status").textContent = "Step " + pos + " of " + (frames.length - 1) + ". " + fr.text;
    }
    draw();
  }
  function slices() {
    var s = Kit.slider(Kit.$("sf-controls"), { label: "start", min: -8, max: 8, step: 1, value: 2, digits: 0 }, draw);
    var e = Kit.slider(Kit.$("sf-controls"), { label: "stop", min: -8, max: 8, step: 1, value: 5, digits: 0 }, draw);
    function draw() {
      var a = s.get(), b = e.get(), idx = sliceIdx(a, b, 1, 8), cells = cellRow(Kit.$("sf-cells"), LENS);
      cells.forEach(function (c, i) { if (idx.indexOf(i) >= 0) c.classList.add("sel"); });
      Kit.setText("sf-expr", "lengths[" + a + ":" + b + "]");
      Kit.setText("sf-res", "[" + idx.map(function (i) { return LENS[i].toFixed(1); }).join(", ") + "]" + (idx.length ? "" : "  (empty)"));
    }
    draw();
  }

  // ======================= 3. functions =======================
  function functions() {
    var len = Kit.slider(Kit.$("ff-controls"), { label: "length_cm", min: 20, max: 130, step: 0.5, value: 44, digits: 1 }, draw);
    var callSel = Kit.select(Kit.$("ff-controls"), "call", [{ value: "0", label: "weight_kg(L), the bamboo defaults" }, { value: "1", label: "weight_kg(L, 0.0044, 2.923), by position" }, { value: "2", label: "weight_kg(L, a=0.0033, b=3.051), by name" }], "0", function () { draw(); });
    var phase = 0, tryPrint = false;
    Kit.chips(Kit.$("ff-bar"), ["1 Call", "2 Inside", "3 Return", "4 After"], 0, function (i) { phase = i; if (i < 3) tryPrint = false; draw(); });
    Kit.button(Kit.$("ff-bar"), "Try print(grams) after the call", function () { phase = 3; tryPrint = true; var bs = Kit.$("ff-bar").querySelectorAll(".group button"); bs.forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === 3)); }); draw(); });
    function draw() {
      var L = len.get(), ci = Number(callSel.value), a = [0.00398, 0.0044, 0.0033][ci], b = [2.95, 2.923, 3.051][ci];
      var grams = a * Math.pow(L, b), ret = grams / 1000;
      var call = ["weight_kg(" + L.toFixed(1) + ")", "weight_kg(" + L.toFixed(1) + ", 0.0044, 2.923)", "weight_kg(" + L.toFixed(1) + ", a=0.0033, b=3.051)"][ci];
      var lines = ["def weight_kg(length_cm, a=0.00398, b=2.95):", "    grams = a * length_cm ** b", "    return grams / 1000", "", "answer = " + call, "print(answer)"];
      if (tryPrint) lines.push("print(grams)");
      var st = [];
      if (phase === 0) st[4] = "on"; else if (phase === 1) st[1] = "on"; else if (phase === 2) st[2] = "on"; else { st[5] = "on"; if (tryPrint) { st[5] = "dim"; st[6] = "err"; } }
      codeLines(Kit.$("ff-code"), lines, st);
      var box = Kit.$("ff-frames"); clear(box); box.className = "frames";
      var inside = H("div", { class: "fbox" }, box); H("p", { class: "vbox-title" }, inside, phase === 3 ? "Inside the call (gone)" : "Inside the call");
      var iv = H("div", { class: "vars" }, inside);
      var cls = phase === 3 ? "gone" : "";
      varBox(iv, "length_cm", fmtNum(L), "float", cls); varBox(iv, "a", fmtNum(a), "float", cls); varBox(iv, "b", fmtNum(b), "float", cls);
      if (phase >= 1) varBox(iv, "grams", fmtNum(grams), "float", phase === 3 ? "gone" : (phase === 1 ? "new" : ""));
      if (phase === 2) varBox(iv, "returns", fmtNum(ret), "float", "new");
      var outside = H("div", { class: "fbox" }, box); H("p", { class: "vbox-title" }, outside, "Outside, in the program");
      var ov = H("div", { class: "vars" }, outside);
      if (phase === 3) { varBox(ov, "answer", fmtNum(ret), "float", "new"); if (tryPrint) varBox(ov, "grams", "NameError: name 'grams' is not defined", "", "err"); }
      else H("small", { class: "note" }, ov, "answer is not set yet");
      Kit.$("ff-status").textContent = ["The call passes " + fmtNum(L) + " to length_cm. " + (ci === 0 ? "The names a and b take their defaults." : "The call also sets a and b."), "The body runs with its own names. grams = a * length_cm ** b = " + fmtNum(grams) + ".", "return sends grams / 1000 = " + fmtNum(ret) + " back to the caller.", tryPrint ? "grams was local to the call. The program cannot see it, so Python raises a NameError." : "The call is finished and its names are gone. The program keeps only the returned value in answer."][phase];
    }
    draw();
  }

  // ======================= 4. containers =======================
  function containers() {
    Kit.data("data/containers.json").then(function (D) {
      var grid = Kit.$("cf-grid"), detail = Kit.$("cf-detail"), score = 0, asked = 0, guess = {}, sel = null;
      var bar = Kit.$("cf-bar");
      var scoreEl = H("span", { class: "label" }, bar, "");
      Kit.button(bar, "Reveal all", function () { D.tasks.forEach(function (t) { D.containers.forEach(function (c) { if (!guess[t.id + c.id]) guess[t.id + c.id] = { shown: true }; }); }); draw(); });
      Kit.button(bar, "Reset", function () { guess = {}; score = 0; asked = 0; sel = null; draw(); });
      function draw() {
        clear(grid);
        grid.style.gridTemplateColumns = "";
        H("div", { class: "h" }, grid, "");
        D.containers.forEach(function (c) { var h = H("div", { class: "h" }, grid); H("b", {}, h, c.label); });
        D.tasks.forEach(function (t) {
          H("div", { class: "rowh" }, grid, t.label);
          D.containers.forEach(function (c) {
            var key = t.id + c.id, g = guess[key], cell = D.cells[t.id][c.id];
            var b = H("button", { type: "button", class: "cc" + (g ? (cell.ok ? " ok" : " bad") : "") + (sel === key ? " sel" : ""), "aria-label": t.label + " on a " + c.label + (g ? (cell.ok ? ": works" : ": error") : ": not revealed yet") }, grid, g ? (cell.ok ? "✓ works" : "✗ error") : "?");
            b.addEventListener("click", function () { sel = key; draw(); });
          });
        });
        scoreEl.textContent = asked ? "Predictions right: " + score + " of " + asked : "Click a cell. Say whether it works, then see what Python does.";
        clear(detail);
        if (!sel) { detail.textContent = "Pick a cell to see the code."; return; }
        var tid = sel.replace(/(list|tuple|dict|set)$/, ""), cid = sel.slice(tid.length), cell = D.cells[tid][cid], t = D.tasks.filter(function (x) { return x.id === tid; })[0], c = D.containers.filter(function (x) { return x.id === cid; })[0];
        var g = guess[sel];
        H("p", {}, detail).innerHTML = "Task: <b>" + t.label + "</b> on a <b>" + c.label + "</b>.";
        var pre = H("pre", {}, detail, c.setup + "\n" + cell.code);
        if (!g) {
          H("p", {}, detail, "Will this run without an error?");
          var row = H("div", { class: "btnrow" }, detail);
          [["Works", true], ["Raises an error", false]].forEach(function (o) {
            Kit.button(row, o[0], function () { asked++; var right = o[1] === cell.ok; if (right) score++; guess[sel] = { said: o[1], right: right }; draw(); });
          });
        } else {
          H("pre", {}, detail, cell.out);
          var p = H("p", {}, detail);
          if (g.said !== undefined) { var s = H("span", { class: g.right ? "good" : "poor" }, p, g.right ? "Right. " : "Not quite. "); p.appendChild(document.createTextNode(cell.ok ? "This works." : "Python stops with an error.")); }
          else p.textContent = cell.ok ? "This works." : "Python stops with an error.";
        }
      }
      draw();
    }).catch(function () { Kit.$("cf-grid").textContent = "The figure data did not load."; });
  }

  // ======================= 4b. set operators =======================
  function venn() {
    var A = ["bamboo", "hammerhead", "milk", "spadenose"], B = ["bamboo", "ganges", "hammerhead", "milk", "spadenose"];
    var ops = [["dive | landing", "union"], ["dive & landing", "both"], ["dive - landing", "onlyA"], ["landing - dive", "onlyB"], ["dive ^ landing", "xor"]];
    var cur = 1, svg = Kit.$("vn-svg");
    Kit.chips(Kit.$("vn-bar"), ops.map(function (o) { return o[0]; }), 1, function (i) { cur = i; draw(); });
    function inSet(kind, w) {
      var inA = A.indexOf(w) >= 0, inB = B.indexOf(w) >= 0;
      return kind === "union" ? inA || inB : kind === "both" ? inA && inB : kind === "onlyA" ? inA && !inB : kind === "onlyB" ? inB && !inA : inA !== inB;
    }
    function draw() {
      var P = Kit.palette(), W = Kit.width(svg), Hh = 230, cx1 = W / 2 - 44, cx2 = W / 2 + 44, cy = 118, r = 92;
      svg.setAttribute("viewBox", "0 0 " + W + " " + Hh); clear(svg);
      var kind = ops[cur][1];
      var defs = Kit.el("defs", {}, svg);
      var cpA = Kit.el("clipPath", { id: "vn-cpa" }, defs); Kit.el("circle", { cx: cx1, cy: cy, r: r }, cpA);
      var mk = Kit.el("mask", { id: "vn-mk" }, defs); Kit.el("rect", { x: 0, y: 0, width: W, height: Hh, fill: "#000" }, mk); Kit.el("circle", { cx: cx1, cy: cy, r: r, fill: "#fff" }, mk); Kit.el("circle", { cx: cx2, cy: cy, r: r, fill: "#000" }, mk);
      var mk2 = Kit.el("mask", { id: "vn-mk2" }, defs); Kit.el("rect", { x: 0, y: 0, width: W, height: Hh, fill: "#000" }, mk2); Kit.el("circle", { cx: cx2, cy: cy, r: r, fill: "#fff" }, mk2); Kit.el("circle", { cx: cx1, cy: cy, r: r, fill: "#000" }, mk2);
      var fillc = P.teal, op = 0.3;
      if (kind === "union" || kind === "xor" || kind === "onlyA") Kit.el("circle", { cx: cx1, cy: cy, r: r, fill: fillc, "fill-opacity": op, mask: "url(#vn-mk)" }, svg);
      if (kind === "union" || kind === "xor" || kind === "onlyB") Kit.el("circle", { cx: cx2, cy: cy, r: r, fill: fillc, "fill-opacity": op, mask: "url(#vn-mk2)" }, svg);
      if (kind === "union" || kind === "both") Kit.el("circle", { cx: cx2, cy: cy, r: r, fill: fillc, "fill-opacity": op, "clip-path": "url(#vn-cpa)" }, svg);
      Kit.el("circle", { cx: cx1, cy: cy, r: r, fill: "none", stroke: P.navy, "stroke-width": 2 }, svg);
      Kit.el("circle", { cx: cx2, cy: cy, r: r, fill: "none", stroke: P.sand, "stroke-width": 2 }, svg);
      var t1 = Kit.el("text", { x: cx1 - 30, y: 18, "text-anchor": "middle", "font-size": 14, fill: P.navy, "font-weight": 600 }, svg); t1.textContent = "dive";
      var t2 = Kit.el("text", { x: cx2 + 30, y: 18, "text-anchor": "middle", "font-size": 14, fill: P.ink2, "font-weight": 600 }, svg); t2.textContent = "landing";
      var spots = { "bamboo": [W / 2, cy - 33], "hammerhead": [W / 2, cy - 11], "milk": [W / 2, cy + 11], "spadenose": [W / 2, cy + 33], "ganges": [W / 2 + 92, cy] };
      Object.keys(spots).forEach(function (w) {
        var on = inSet(kind, w), t = Kit.el("text", { x: spots[w][0], y: spots[w][1], "text-anchor": "middle", "font-size": 13.5, fill: on ? P.ink : P.muted, "font-weight": on ? 600 : 400, "text-decoration": on ? "none" : "line-through" }, svg);
        t.textContent = w;
      });
      var members = Object.keys(spots).filter(function (w) { return inSet(kind, w); }).sort();
      Kit.setText("vn-code", "sorted(" + ops[cur][0] + ")");
      Kit.setText("vn-res", "[" + members.map(function (m) { return "'" + m + "'"; }).join(", ") + "]");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw));
  }

  // ======================= 5. classes =======================
  function classes() {
    var CLS = ["class Shark:", "    def __init__(self, tag, species, length_cm):", "        self.tag = tag", "        self.species = species", "        self.length_cm = length_cm", "", "    def weight_kg(self):", "        a, b = LAW[self.species]", "        return a * self.length_cm ** b / 1000", "", "    def grow(self, cm):", "        self.length_cm = self.length_cm + cm"];
    var names = ["a", "b", "c", "d"], LAW = { spadenose: [0.0044, 2.923], milk: [0.0033, 3.051], bamboo: [0.00398, 2.95] }, tags = [3305, 2194, 9242], objs = [{ tag: 4127, species: "bamboo", len: 44 }], nextTag = 0, hot = [], flash = -1;
    var sp = Kit.select(Kit.$("kf-controls"), "species", ["spadenose", "milk", "bamboo"].map(function (s) { return { value: s, label: s }; }), "spadenose", function () {});
    var ln = Kit.slider(Kit.$("kf-controls"), { label: "length_cm", min: 25, max: 100, step: 0.5, value: 41, digits: 1 }, function () {});
    var make = Kit.button(Kit.$("kf-bar"), "Make a Shark", function () {
      if (objs.length >= 4) { Kit.$("kf-status").textContent = "The shelf is full at four sharks. Reload the page to start again."; return; }
      var o = { tag: tags[nextTag++ % tags.length], species: sp.value, len: ln.get() }; objs.push(o); hot = [0, 1, 2, 3, 4]; flash = objs.length - 1;
      status(names[objs.length - 1] + " = Shark(" + o.tag + ', "' + o.species + '", ' + o.len.toFixed(1) + ") runs __init__, which stores three attributes on the new object."); draw();
    });
    function status(t) { Kit.$("kf-status").textContent = t; }
    function f1(x) { return Number.isInteger(x) ? x.toFixed(1) : String(x); }
    function draw() {
      var st = []; hot.forEach(function (i) { st[i] = "on"; });
      codeLines(Kit.$("kf-code"), CLS, st);
      var shelf = Kit.$("kf-shelf"); clear(shelf);
      objs.forEach(function (o, i) {
        var c = H("div", { class: "scard" + (i === flash ? " flash" : "") }, shelf), h = H("h4", {}, c); h.appendChild(document.createTextNode("Shark object ")); H("span", {}, h, names[i]);
        var tb = H("table", {}, c);
        [["tag", String(o.tag)], ["species", '"' + o.species + '"'], ["length_cm", f1(o.len)]].forEach(function (r) { var tr = H("tr", {}, tb); H("td", {}, tr, "self." + r[0]); H("td", {}, tr, r[1]); });
        var bt = H("div", { class: "btns" }, c);
        var b1 = H("button", { type: "button" }, bt, ".grow(5)"), b2 = H("button", { type: "button" }, bt, ".weight_kg()");
        b1.addEventListener("click", function () { o.len += 5; hot = [10, 11]; flash = i; status(names[i] + ".grow(5) changes length_cm on " + names[i] + " only. It is now " + f1(o.len) + ". The other sharks are unchanged."); draw(); });
        b2.addEventListener("click", function () { hot = [6, 7, 8]; flash = i; var lw = LAW[o.species], w = lw[0] * Math.pow(o.len, lw[1]) / 1000; status(names[i] + ".weight_kg() looks up a and b for " + o.species + " and returns " + lw[0] + " * " + f1(o.len) + " ** " + lw[1] + " / 1000 = " + fmtNum(Math.round(w * 1e4) / 1e4) + ". No attribute is changed."); draw(); });
      });
    }
    status("Tag 4127, the Juhu bamboo shark, is already made. Make a second shark, then press .grow(5) on only one of them.");
    draw();
  }

  // ======================= 6. pandas-style query builder =======================
  var SAMPLE_IDS = [1134, 1683, 2194, 3305, 3395, 4127, 4595, 5330, 5519, 5804, 5884, 8512, 9116, 9226, 9242, 9347];
  function tables() {
    fetch("../data/sharks/catch.csv").then(function (r) { return r.text(); }).then(function (txt) {
      var lines = txt.trim().split(/\r?\n/), head = lines[0].split(",").map(function (h) { return h.trim(); });
      var all = lines.slice(1).map(function (l) { var p = l.split(","), o = {}; head.forEach(function (h, i) { o[h] = p[i]; }); o.id = +o.record_id; o.length_cm = +o.length_cm; o.weight_kg = +o.weight_kg; o.mature = +o.mature; return o; });
      var rows = SAMPLE_IDS.map(function (id) { return all.filter(function (r) { return r.id === id; })[0]; });
      var species = ["any species", "spadenose", "milk", "bamboo", "hammerhead"];
      var sp = Kit.select(Kit.$("qf-controls"), "species", species.map(function (s) { return { value: s, label: s }; }), "milk", draw);
      var mn = Kit.slider(Kit.$("qf-controls"), { label: "length over, cm", min: 0, max: 90, step: 5, value: 60, digits: 0 }, draw);
      var mt = Kit.select(Kit.$("qf-controls"), "mature", [{ value: "any", label: "any maturity" }, { value: "1", label: "mature = 1" }, { value: "0", label: "mature = 0" }], "any", draw);
      var table = Kit.$("qf-table");
      function draw() {
        var conds = [], tests = [];
        if (sp.value !== "any species") { conds.push('df["species"] == "' + sp.value + '"'); tests.push(function (r) { return r.species === sp.value; }); }
        var m = mn.get(); if (m > 0) { conds.push('df["length_cm"] > ' + m); tests.push(function (r) { return r.length_cm > m; }); }
        if (mt.value !== "any") { conds.push('df["mature"] == ' + mt.value); tests.push(function (r) { return r.mature === Number(mt.value); }); }
        var code = conds.length === 0 ? 'mask = pd.Series(True, index=df.index)   # no condition: all rows are True' : conds.length === 1 ? "mask = " + conds[0] : "mask = " + conds.map(function (c) { return "(" + c + ")"; }).join(" & ");
        Kit.$("qf-code").textContent = code + "\ndf[mask]";
        clear(table);
        var thead = H("thead", {}, table), tr = H("tr", {}, thead);
        ["record_id", "species", "length_cm", "weight_kg", "mature", "mask"].forEach(function (h) { H("th", { class: h === "mask" ? "mask" : "" }, tr, h); });
        var tb = H("tbody", {}, table), kept = 0, sum = 0;
        rows.forEach(function (r) {
          var ok = tests.every(function (t) { return t(r); }); if (ok) { kept++; sum += r.length_cm; }
          var row = H("tr", { class: ok ? "" : "off" }, tb);
          [r.id, r.species, r.length_cm.toFixed(1), r.weight_kg.toFixed(3), r.mature].forEach(function (v) { H("td", {}, row, String(v)); });
          H("td", { class: "mask " + (ok ? "t" : "f") }, row, ok ? "True" : "False");
        });
        Kit.setText("qf-n", String(kept));
        Kit.setText("qf-mean", kept ? (sum / kept).toFixed(2) + " cm" : "no rows");
      }
      draw();
    }).catch(function () { Kit.$("qf-code").textContent = "The catch table did not load."; });
  }

  // ======================= 7. array indexing =======================
  var AM = [[42.4, 0.267, 33.3, 25.2], [45.1, 0.242, 10.3, 25.4], [74.5, 2.128, 10.7, 28.4], [41.0, 0.228, 18.0, 25.4], [36.1, 0.164, 22.2, 28.6], [44.0, 0.281, 0.4, 28.6]];
  var ACOL = ["length", "weight", "depth", "temp"];
  function r2(x) { return String(Math.round(x * 100) / 100); }
  function arrays() {
    var mode = "idx", rowIn, colIn, msg = null;
    var presets = [
      { l: "a[0]", r: "0", c: ":" }, { l: "a[:, 0]", r: ":", c: "0" }, { l: "a[1:4, :2]", r: "1:4", c: ":2" }, { l: "a[-1]", r: "-1", c: ":" }, { l: "a[::2]", r: "::2", c: ":" },
      { l: "a[a[:, 0] > 44]", m: "bool" }, { l: "a.sum(axis=0)", m: "sum0" }, { l: "a.sum(axis=1)", m: "sum1" }
    ];
    var chipBox = H("span", {}, Kit.$("af-bar"));
    var chips = Kit.chips(chipBox, presets.map(function (p) { return p.l; }), 2, function (i) { var p = presets[i]; mode = p.m || "idx"; if (!p.m) { rowIn.value = p.r; colIn.value = p.c; } draw(); });
    var box = H("div", { class: "mono", style: "display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-size:15px;margin:6px 0 0" }, Kit.$("af-controls"));
    H("span", {}, box, "a[");
    rowIn = H("input", { type: "text", value: "1:4", size: "6", "aria-label": "rows part of the index", style: "font:inherit;padding:2px 6px;border:1px solid var(--rule);border-radius:4px;background:var(--bg);color:var(--ink)" }, box);
    H("span", {}, box, ",");
    colIn = H("input", { type: "text", value: ":2", size: "6", "aria-label": "columns part of the index", style: "font:inherit;padding:2px 6px;border:1px solid var(--rule);border-radius:4px;background:var(--bg);color:var(--ink)" }, box);
    H("span", {}, box, "]");
    [rowIn, colIn].forEach(function (i) { i.addEventListener("input", function () { mode = "idx"; chips.forEach(function (b) { b.setAttribute("aria-pressed", "false"); }); draw(); }); });
    function grid(parent, hdr, rowsHdr, cells, cols, small) {
      clear(parent); parent.style.gridTemplateColumns = "repeat(" + (cols + 1) + ", auto)";
      H("div", { class: "ac hd" }, parent, "");
      hdr.forEach(function (h) { H("div", { class: "ac hd" }, parent, h); });
      cells.forEach(function (row, i) {
        H("div", { class: "ac hd" }, parent, rowsHdr[i]);
        row.forEach(function (c) { H("div", { class: "ac" + (c.sel ? " sel" : "") }, parent, c.t); });
      });
    }
    function draw() {
      var R = [0, 1, 2, 3, 4, 5], C = [0, 1, 2, 3], selR = [], selC = [], res = null, dims = "", expr = "", err = null, extra = null;
      if (mode === "idx") {
        var r = parseAxis(rowIn.value, 6), c = parseAxis(colIn.value, 4);
        if (!r || !c) err = "This index is not one of the forms this figure understands. Try 1:4, :, -1, ::2 or a number.";
        else if (r.err || c.err) err = "Python raises an IndexError: " + (r.err || c.err) + ".";
        else {
          selR = r.int !== undefined ? [r.int] : r.list; selC = c.int !== undefined ? [c.int] : c.list;
          var vals = selR.map(function (i) { return selC.map(function (j) { return AM[i][j]; }); });
          if (r.int !== undefined && c.int !== undefined) { res = { type: "0", v: vals[0][0] }; dims = "()"; }
          else if (r.int !== undefined) { res = { type: "1", v: vals[0], hdr: selC.map(function (j) { return ACOL[j]; }) }; dims = "(" + selC.length + ",)"; }
          else if (c.int !== undefined) { res = { type: "1", v: vals.map(function (x) { return x[0]; }), hdr: selR.map(function (i) { return "row " + i; }) }; dims = "(" + selR.length + ",)"; }
          else { res = { type: "2", v: vals, rowsH: selR.map(function (i) { return "row " + i; }), hdr: selC.map(function (j) { return ACOL[j]; }) }; dims = "(" + selR.length + ", " + selC.length + ")"; }
          expr = colIn.value.trim() === ":" ? "a[" + rowIn.value.trim() + "]" : "a[" + rowIn.value.trim() + ", " + colIn.value.trim() + "]";
        }
      } else if (mode === "bool") {
        selR = R.filter(function (i) { return AM[i][0] > 44; }); selC = C; extra = R.map(function (i) { return AM[i][0] > 44 ? "True" : "False"; });
        res = { type: "2", v: selR.map(function (i) { return AM[i]; }), rowsH: selR.map(function (i) { return "row " + i; }), hdr: ACOL }; dims = "(" + selR.length + ", 4)"; expr = "a[a[:, 0] > 44]";
      } else if (mode === "sum0") { selR = R; selC = C; res = { type: "1", v: C.map(function (j) { return R.reduce(function (s, i) { return s + AM[i][j]; }, 0); }), hdr: ACOL }; dims = "(4,)"; expr = "a.sum(axis=0)"; }
      else { selR = R; selC = C; res = { type: "1", v: R.map(function (i) { return AM[i].reduce(function (s, x) { return s + x; }, 0); }), hdr: R.map(function (i) { return "row " + i; }) }; dims = "(6,)"; expr = "a.sum(axis=1)"; }
      var main = Kit.$("af-grid"); clear(main); main.style.gridTemplateColumns = "repeat(" + (extra ? 6 : 5) + ", auto)";
      H("div", { class: "ac hd" }, main, ""); ACOL.forEach(function (h) { H("div", { class: "ac hd" }, main, h); }); if (extra) H("div", { class: "ac hd" }, main, "a[:, 0] > 44");
      AM.forEach(function (row, i) {
        H("div", { class: "ac hd" }, main, "row " + i);
        row.forEach(function (v, j) { var on = !err && (mode === "sum0" || mode === "sum1" ? true : selR.indexOf(i) >= 0 && selC.indexOf(j) >= 0); H("div", { class: "ac" + (on ? " sel" : "") }, main, String(v)); });
        if (extra) H("div", { class: "ac hd" }, main, extra[i]);
      });
      Kit.$("af-expr").textContent = err ? "(not valid)" : expr;
      Kit.$("af-dims").textContent = err ? "none" : dims;
      var out = Kit.$("af-result"); clear(out);
      if (err) { out.style.display = "block"; out.textContent = err; return; }
      out.style.display = "inline-grid";
      if (res.type === "0") { out.style.gridTemplateColumns = "auto"; H("div", { class: "ac" }, out, String(res.v)); }
      else if (res.type === "1") { out.style.gridTemplateColumns = "repeat(" + res.v.length + ", auto)"; res.hdr.forEach(function (h) { H("div", { class: "ac hd" }, out, h); }); res.v.forEach(function (x) { H("div", { class: "ac" }, out, r2(x)); }); }
      else { out.style.gridTemplateColumns = "repeat(" + (res.hdr.length + 1) + ", auto)"; H("div", { class: "ac hd" }, out, ""); res.hdr.forEach(function (h) { H("div", { class: "ac hd" }, out, h); }); res.v.forEach(function (row, i) { H("div", { class: "ac hd" }, out, res.rowsH[i]); row.forEach(function (x) { H("div", { class: "ac" }, out, r2(x)); }); }); }
    }
    draw();
  }

  // ======================= 8. the array-dimensions explorer =======================
  var PRESET_DIMS = ["()", "(3,)", "(4,)", "(1,3)", "(3,1)", "(4,1)", "(2,3)", "(3,2)", "(4,3)", "(3,4)", "(4,2)", "(2,1,3)", "(2,3,4)"];
  function drawArray(parent, dims, vals, cls, copies, onCell) {
    var cells = [], n = prod(dims);
    function mk(host, i) {
      var c = H("div", { class: "bc " + cls + (copies && copies[i] ? " fade" : "") + (onCell ? " clickable" : "") }, host, fmtNum(vals[i]));
      cells[i] = c;
      if (onCell) { c.addEventListener("mouseenter", function () { onCell(i, true); }); c.addEventListener("click", function () { onCell(i, true); }); c.addEventListener("mouseleave", function () { onCell(i, false); }); }
    }
    if (dims.length === 0) { var h = H("div", { class: "bslice", style: "grid-template-columns:auto" }, parent); mk(h, 0); }
    else if (dims.length === 1) { var h1 = H("div", { class: "bslice", style: "grid-template-columns:repeat(" + dims[0] + ",auto)" }, parent); for (var i = 0; i < n; i++) mk(h1, i); }
    else if (dims.length === 2) { var h2 = H("div", { class: "bslice", style: "grid-template-columns:repeat(" + dims[1] + ",auto)" }, parent); for (var j = 0; j < n; j++) mk(h2, j); }
    else {
      var wrap = H("div", { class: "bslices" }, parent), per = dims[1] * dims[2];
      for (var s = 0; s < dims[0]; s++) {
        var col = H("div", {}, wrap); H("small", { class: "note" }, col, "slice " + s);
        var g = H("div", { class: "bslice", style: "grid-template-columns:repeat(" + dims[2] + ",auto)" }, col);
        for (var q = 0; q < per; q++) mk(g, s * per + q);
      }
    }
    return cells;
  }
  function broadcastFig() {
    var A = "(4,3)", B = "(3,)", op = "+", hide = false, revealed = false;
    var opts = PRESET_DIMS.map(function (d) { return { value: d, label: dimsText(parseDims(d)) }; });
    var selA = Kit.select(Kit.$("bf-controls"), "A has dimensions", opts, A, function (v) { A = v; revealed = false; draw(); });
    var selB = Kit.select(Kit.$("bf-controls"), "B has dimensions", opts, B, function (v) { B = v; revealed = false; draw(); });
    var opChips = Kit.chips(Kit.$("bf-bar"), ["A + B", "A * B", "A @ B"], 0, function (i) { op = ["+", "*", "@"][i]; revealed = false; draw(); });
    var hideChips = Kit.chips(Kit.$("bf-bar"), ["Show the answer", "Hide it until I reveal"], 0, function (i) { hide = i === 1; revealed = false; draw(); });
    var revealBtn = Kit.button(Kit.$("bf-bar"), "Reveal", function () { revealed = true; draw(); });
    var tries = [["(4,3)", "(3,)", "+"], ["(4,3)", "(4,)", "+"], ["(4,3)", "(4,1)", "+"], ["(4,1)", "(1,3)", "+"], ["(2,1,3)", "(4,1)", "+"], ["(4,3)", "(3,2)", "@"], ["(4,3)", "(4,2)", "@"], ["(2,3,4)", "(4,2)", "@"]];
    tries.forEach(function (t) {
      Kit.button(Kit.$("bf-try"), dimsText(parseDims(t[0])) + " " + t[2] + " " + dimsText(parseDims(t[1])), function () {
        A = t[0]; B = t[1]; op = t[2]; selA.value = A; selB.value = B; revealed = false;
        var oi = ["+", "*", "@"].indexOf(op); opChips.forEach(function (b, i) { b.setAttribute("aria-pressed", String(i === oi)); }); draw();
      });
    });
    function dt(d) { return dimsText(d); }
    function panel(title, dimsLabel, cls, dims, vals, copies, onCell) {
      var p = H("div", { class: "bpanel", role: "img", "aria-label": title + " with dimensions " + dimsLabel }, Kit.$("bf-grids"));
      var t = H("div", { class: "ttl" }, p); H("b", {}, t, title); t.appendChild(document.createTextNode(" ")); H("i", {}, t, dimsLabel);
      return { el: p, cells: drawArray(p, dims, vals, cls, copies, onCell) };
    }
    function sym(t) { H("div", { class: "bsyms" }, Kit.$("bf-grids"), t); }
    function draw() {
      var dA = parseDims(A), dB = parseDims(B), vA = makeValues(dA, "a"), vB = makeValues(dB, "b");
      var res = op === "@" ? matmul(dA, vA, dB, vB) : elementwise(dA, vA, dB, vB, op);
      var wait = hide && !revealed;
      revealBtn.style.display = hide && !revealed ? "" : "none";
      var grids = Kit.$("bf-grids"); clear(grids);
      var eq = Kit.$("bf-eqn"); clear(eq);
      function box(cls, b, i) { var t = H("span", { class: "t " + cls }, eq); H("b", {}, t, b); H("i", {}, t, i); }
      box("c-navy", dt(dA), "A"); H("span", { class: "op" }, eq, op === "@" ? "@" : op === "*" ? "×" : "+"); box("c-sand", dt(dB), "B"); H("span", { class: "op" }, eq, "→");
      if (wait) box("c-teal", "?", "result: your turn"); else if (res.ok) box("c-teal", dt(res.dims), "result"); else box("c-coral", "error", "no result");
      var status = Kit.$("bf-status"), rule = Kit.$("bf-rule"); clear(rule);
      if (op !== "@") {
        var showA = res.ok ? dA : dA;
        var pa, pb, pr;
        if (res.ok) {
          pa = panel("A", dt(dA) + (dt(dA) !== dt(res.dims) ? ", stretched to " + dt(res.dims) : ""), "a", res.dims, res.shownA, res.copyA);
          sym(op === "*" ? "×" : "+");
          pb = panel("B", dt(dB) + (dt(dB) !== dt(res.dims) ? ", stretched to " + dt(res.dims) : ""), "b", res.dims, res.shownB, res.copyB);
        } else {
          panel("A", dt(dA), "a", dA, vA, null); sym(op === "*" ? "×" : "+"); panel("B", dt(dB), "b", dB, vB, null);
        }
        sym("=");
        if (wait) { var pq = panel("Result", "?", "q", [1], ["?"], null); }
        else if (res.ok) panel("Result", dt(res.dims), "r", res.dims, res.values, null);
        else panel("Result", "none", "x", [1], ["×"], null);
        // rule table
        var tbl = H("table", {}, rule), th = H("tr", {}, H("thead", {}, tbl));
        ["from the right", "A", "B", "result", "what happens"].forEach(function (h) { H("th", {}, th, h); });
        var tb = H("tbody", {}, tbl), bc = broadcast(dA, dB);
        bc.rows.forEach(function (r) {
          var tr = H("tr", {}, tb); H("td", {}, tr, String(r.j)); H("td", {}, tr, r.a === null ? "missing" : String(r.a)); H("td", {}, tr, r.b === null ? "missing" : String(r.b));
          H("td", {}, tr, wait ? "?" : r.good ? String(r.r) : "–"); H("td", { class: wait ? "" : r.good ? "ok" : "no" }, tr, wait ? "work it out" : (r.good ? "✓ " : "✗ ") + r.verdict);
        });
        if (bc.rows.length === 0) { var tr0 = H("tr", {}, tb); H("td", { colspan: 5 }, tr0, "Both are single numbers, so there are no dimensions to compare. The result is a single number."); }
        status.textContent = wait ? "Predict the result's dimensions, or the error, then press Reveal." : res.ok ? dt(dA) + " " + op + " " + dt(dB) + " works. The result has dimensions " + dt(res.dims) + "." + (dt(dA) !== dt(res.dims) || dt(dB) !== dt(res.dims) ? " The faded cells are the stretched copies." : "") : "NumPy raises an error: " + bc.rows.filter(function (r) { return !r.good; })[0].verdict + ", counting dimensions from the right.";
      } else {
        var mm = matmul(dA, vA, dB, vB), cellsA, cellsB, cellsR;
        var pA = panel("A", dt(dA), "a", dA, vA, null), pBp = (sym("@"), panel("B", dt(dB), "b", dB, vB, null)); sym("=");
        cellsA = pA.cells; cellsB = pBp.cells;
        var pR;
        if (wait) pR = panel("Result", "?", "q", [1], ["?"], null);
        else if (mm.ok) {
          pR = panel("Result", dt(mm.dims), "r", mm.dims, mm.values, null, function (i, on) {
            cellsA.concat(cellsB).forEach(function (c) { c.classList.remove("hl"); });
            if (!on) { status.textContent = dt(dA) + " @ " + dt(dB) + " works. The result has dimensions " + dt(mm.dims) + ". Point at a result cell to see what it used."; return; }
            var u = mm.uses[i], N = mm.norm;
            for (var t = 0; t < N.k; t++) { cellsA[u.batchA * N.m * N.k + u.i * N.k + t].classList.add("hl"); cellsB[u.batchB * N.k * N.n + t * N.n + u.j].classList.add("hl"); }
            status.textContent = "This cell is a row of A times a column of B, added up: " + u.text + ".";
          });
        } else pR = panel("Result", "none", "x", [1], ["×"], null);
        var tbl2 = H("table", {}, rule), th2 = H("tr", {}, H("thead", {}, tbl2));
        ["part", "from A", "from B", "what happens"].forEach(function (h) { H("th", {}, th2, h); });
        var tb2 = H("tbody", {}, tbl2);
        function row(a, b, c, d, cls) { var tr = H("tr", {}, tb2); H("td", {}, tr, a); H("td", {}, tr, b); H("td", {}, tr, c); H("td", { class: wait ? "" : cls || "" }, tr, wait ? "work it out" : d); }
        if (dA.length === 0 || dB.length === 0) row("scalar", dA.length === 0 ? "a single number" : "", dB.length === 0 ? "a single number" : "", "✗ @ needs at least one dimension. Use * for a single number.", "no");
        else {
          var Aa = dA.length === 1 ? [1, dA[0]] : dA, Bb = dB.length === 1 ? [dB[0], 1] : dB, k1 = Aa[Aa.length - 1], k2 = Bb[Bb.length - 2];
          row("shared size k", "last: " + k1, "second-to-last: " + k2, k1 === k2 ? "✓ equal, so k = " + k1 + " and it disappears from the result" : "✗ " + k1 + " and " + k2 + " must be equal", k1 === k2 ? "ok" : "no");
          row("result rows", "second-to-last: " + Aa[Aa.length - 2], "", "kept as the rows of the result" + (dA.length === 1 ? " (A is 1-dimensional, so this 1 is dropped)" : ""), "ok");
          row("result columns", "", "last: " + Bb[Bb.length - 1], "kept as the columns of the result" + (dB.length === 1 ? " (B is 1-dimensional, so this 1 is dropped)" : ""), "ok");
          if (Aa.length > 2 || Bb.length > 2) row("leading dimensions", Aa.length > 2 ? dt(Aa.slice(0, -2)) : "none", Bb.length > 2 ? dt(Bb.slice(0, -2)) : "none", "broadcast as in the rule above, so a 2-dimensional B is reused for all slices", "ok");
        }
        status.textContent = wait ? "Predict the result's dimensions, or the error, then press Reveal." : mm.ok ? dt(dA) + " @ " + dt(dB) + " works. The result has dimensions " + dt(mm.dims) + ". Point at a result cell to see what it used." : "NumPy raises an error. " + mm.msg;
      }
    }
    draw();
  }

  // ======================= 9. seeds =======================
  function seeds() {
    Kit.data("data/seeds.json").then(function (S) {
      var mode = 0, batch = 0, run = 0, cur = null, seen = {};
      var sl = Kit.slider(Kit.$("rf-controls"), { label: "seed", min: 0, max: 30, step: 1, value: 7, digits: 0 }, function () {});
      Kit.chips(Kit.$("rf-bar"), ["Fixed seed", "No seed"], 0, function (i) { mode = i; });
      Kit.button(Kit.$("rf-bar"), "Run script", function () { run++; batch = 0; show(false); });
      Kit.button(Kit.$("rf-bar"), "Draw 8 more", function () { if (!cur) { run++; batch = 0; show(false); return; } if (mode === 0 && batch >= 2) { addLog("This demo stores three draws per seed. Press Run script to start again.", false, true); return; } batch++; show(true); });
      Kit.button(Kit.$("rf-bar"), "Clear log", function () { clear(Kit.$("rf-log")); seen = {}; run = 0; });
      function vals() {
        if (mode === 0) return S[String(sl.get())][batch];
        var r = []; for (var i = 0; i < 8; i++) r.push(30 + Math.floor(Math.random() * 91)); return r;
      }
      function addLog(text, same, plain) {
        var d = H("div", { class: same ? "same" : "" }, Kit.$("rf-log"), text); if (same) H("b", {}, d, same);
        Kit.$("rf-log").scrollTop = Kit.$("rf-log").scrollHeight; return d;
      }
      function show(more) {
        cur = vals(); var key = cur.join(",");
        var label = (mode === 0 ? "seed " + sl.get() : "no seed") + ", run " + run + (more ? ", draw " + (batch + 1) : "");
        var same = seen[key] ? "same as " + seen[key] : null; if (!seen[key]) seen[key] = "run " + run + (more ? ", draw " + (batch + 1) : "");
        addLog(label + ": [" + cur.join(", ") + "]", same);
        plot();
      }
      function plot() {
        Kit.plot("rf-bars", function (P) { return [{ type: "bar", x: cur.map(function (_, i) { return i + 1; }), y: cur, marker: { color: P.navy }, text: cur.map(String), textposition: "outside", hovertemplate: "draw %{x}: %{y} cm<extra></extra>", cliponaxis: false }]; },
          { xaxis: { title: { text: "which draw" }, dtick: 1 }, yaxis: { title: { text: "length, cm" }, range: [0, 140] }, margin: { l: 52, r: 12, t: 12, b: 44 } });
      }
      run = 1; show(false);
    }).catch(function () { Kit.$("rf-log").textContent = "The seed table did not load."; });
  }

  // ======================= 10. autograd =======================
  function autograd() {
    var X = [0.41, 0.44, 0.82], Y = [0.228, 0.281, 2.610], W = 1.0, B = 0.0;
    var wS = Kit.slider(Kit.$("gf-controls"), { label: "w (slope)", min: -2, max: 10, step: 0.1, value: 1, digits: 1 }, function (v) { W = v; draw(); });
    var bS = Kit.slider(Kit.$("gf-controls"), { label: "b (intercept)", min: -2, max: 2, step: 0.1, value: 0, digits: 1 }, function (v) { B = v; draw(); });
    var eS = Kit.slider(Kit.$("gf-controls"), { label: "step size", min: 0.05, max: 1.5, step: 0.05, value: 0.5, digits: 2 }, function () {});
    Kit.button(Kit.$("gf-bar"), "Gradient step: w = w - step * w.grad", function () { var g = grads(); W -= eS.get() * g.dw; B -= eS.get() * g.db; wS.set(Math.max(-2, Math.min(10, W))); bS.set(Math.max(-2, Math.min(2, B))); draw(); });
    Kit.button(Kit.$("gf-bar"), "Reset", function () { W = 1; B = 0; wS.set(1); bS.set(0); draw(); });
    function grads() {
      var m = X.length, L = 0, dw = 0, db = 0;
      X.forEach(function (x, i) { var e = W * x + B - Y[i]; L += e * e; dw += 2 * e * x; db += 2 * e; });
      return { L: L / m, dw: dw / m, db: db / m };
    }
    var eq = Kit.$("gf-eqn");
    function box(cls, id, label) { var t = H("span", { class: "t " + cls }, eq); H("b", { id: id }, t, ""); H("i", {}, t, label); }
    box("c-coral", "gf-w", "w"); H("span", { class: "op" }, eq, ","); box("c-coral", "gf-b", "b"); H("span", { class: "op" }, eq, "→"); box("c-navy", "gf-L", "loss L"); H("span", { class: "op" }, eq, "→"); box("c-teal", "gf-dw", "w.grad"); box("c-teal", "gf-db", "b.grad");
    function draw() {
      var g = grads(), segX = [], segY = [];
      X.forEach(function (x, i) { segX.push(x, x, null); segY.push(Y[i], W * x + B, null); });
      Kit.plot("gf-plot", function (P) {
        return [
          { x: segX, y: segY, mode: "lines", line: { color: P.muted, width: 2, dash: "dot" }, hoverinfo: "skip" },
          { x: [0, 1], y: [B, W + B], mode: "lines", line: { color: P.coral, width: 3 }, hoverinfo: "skip" },
          { x: X, y: Y, mode: "markers", marker: { color: P.navy, size: 11 }, hovertemplate: "%{x} m, %{y} kg<extra></extra>" }
        ];
      }, { xaxis: { title: { text: "length, m" }, range: [0, 1] }, yaxis: { title: { text: "weight, kg" }, range: [-1, 4] }, margin: { l: 52, r: 12, t: 12, b: 46 } });
      Kit.setText("gf-w", Kit.fmt(W, 2)); Kit.setText("gf-b", Kit.fmt(B, 2)); Kit.setText("gf-L", Kit.fmt(g.L, 4)); Kit.setText("gf-dw", Kit.fmt(g.dw, 4)); Kit.setText("gf-db", Kit.fmt(g.db, 4));
    }
    draw();
  }
})();
