// Chapter 0.2 Algorithms and data structures: figures. Pure logic first (exported for tests), then the figures.
(function () {
  "use strict";

  // ======================= pure logic =======================
  var DEFAULT16 = [42.4, 45.1, 74.5, 41.0, 36.1, 44.0, 64.3, 46.0, 82.0, 28.5, 87.2, 42.4, 75.0, 62.2, 54.4, 52.0];
  var TAGS = [4127, 3305, 5519, 1134, 1683, 2194, 3395, 4595];

  // Insertion sort by adjacent swaps. One frame per comparison; frame.a is the list after the swap, frame.cmp the two positions compared.
  function insertionFrames(a0) {
    var a = a0.slice(), f = [{ a: a.slice(), cmp: null, count: 0, front: 0 }], c = 0;
    for (var i = 1; i < a.length; i++) {
      var j = i;
      while (j > 0) {
        c++;
        var sw = a[j - 1] > a[j], cmp = [j - 1, j];
        if (sw) { var t = a[j - 1]; a[j - 1] = a[j]; a[j] = t; }
        f.push({ a: a.slice(), cmp: cmp, swapped: sw, count: c, front: i });
        if (sw) j--; else break;
      }
    }
    f[f.length - 1].done = true;
    return f;
  }
  // Top-down merge sort. One frame per comparison; frame.a shows the merged part first, then what is left of each half.
  function mergeFrames(a0) {
    var a = a0.slice(), f = [{ a: a.slice(), cmp: null, count: 0, win: null }], c = 0;
    function rec(lo, hi) {
      if (hi - lo <= 1) return;
      var mid = (lo + hi) >> 1; rec(lo, mid); rec(mid, hi);
      var L = a.slice(lo, mid), R = a.slice(mid, hi), out = [], i = 0, j = 0;
      while (i < L.length && j < R.length) {
        c++;
        var disp = out.concat(L.slice(i), R.slice(j));
        f.push({ a: a.slice(0, lo).concat(disp, a.slice(hi)), cmp: [lo + out.length, lo + out.length + (L.length - i)], count: c, win: [lo, hi] });
        if (L[i] <= R[j]) { out.push(L[i]); i++; } else { out.push(R[j]); j++; }
      }
      var res = out.concat(L.slice(i), R.slice(j));
      for (var k = 0; k < res.length; k++) a[lo + k] = res[k];
    }
    rec(0, a.length);
    f.push({ a: a.slice(), cmp: null, count: c, win: null, done: true });
    return f;
  }
  // Top-k by a scan with a board of the best k so far (the same counting as the code in the chapter).
  function topkFrames(items, k) {
    var board = [], c = 0, f = [{ i: -1, board: [], c: 0, text: "The board starts empty." }];
    items.forEach(function (x, i) {
      var text;
      if (board.length === k) {
        c++;
        if (x <= board[board.length - 1]) { f.push({ i: i, board: board.slice(), c: c, kind: "rejected", text: x.toFixed(1) + " is not above the weakest board item, " + board[board.length - 1].toFixed(1) + ". One comparison rejects it." }); return; }
        var out = board.pop(); text = x.toFixed(1) + " beats " + out.toFixed(1) + ", which leaves the board. ";
      } else text = "The board is not full yet, so " + x.toFixed(1) + " goes on it. ";
      var j = board.length, pc = 0;
      while (j > 0) { c++; pc++; if (board[j - 1] >= x) break; j--; }
      board.splice(j, 0, x);
      f.push({ i: i, board: board.slice(), c: c, kind: "entered", text: text + "Finding its place took " + pc + " comparison" + (pc === 1 ? "." : "s.") });
    });
    return f;
  }
  function bsFrames(items, target) {
    var lo = 0, hi = items.length - 1, f = [];
    while (lo <= hi) {
      var mid = (lo + hi) >> 1, v = items[mid], verdict = v === target ? "found" : v < target ? "low" : "high";
      f.push({ lo: lo, hi: hi, mid: mid, verdict: verdict });
      if (verdict === "found") return f;
      if (verdict === "low") lo = mid + 1; else hi = mid - 1;
    }
    f.push({ lo: lo, hi: hi, mid: -1, verdict: "empty" });
    return f;
  }
  // A small hash table with chaining, to count real comparisons for the lookup race.
  function hashOf(k, M) { return (Math.imul(k, 2654435761) >>> 0) % M; }
  function buildTable(N) {
    var M = 2 * N, head = new Int32Array(M).fill(-1), next = new Int32Array(N), keys = new Int32Array(N);
    for (var i = 0; i < N; i++) { var key = 4101 + i; keys[i] = key; var h = hashOf(key, M); next[i] = head[h]; head[h] = i; }
    return { M: M, head: head, next: next, keys: keys, N: N };
  }
  function lookupCount(T, key) {
    var c = 0, i = T.head[hashOf(key, T.M)];
    while (i >= 0) { c++; if (T.keys[i] === key) return c; i = T.next[i]; }
    return c;
  }
  // Recursion: tree of calls and the event list.
  function simulate(kind, n) {
    var nodes = [], events = [];
    function rec(k, parent) {
      var node = { id: nodes.length, k: k, parent: parent, children: [], value: null }; nodes.push(node);
      if (parent !== null) nodes[parent].children.push(node.id);
      events.push({ t: "call", id: node.id });
      var v;
      if (kind === "total") v = k === 0 ? 0 : k + rec(k - 1, node.id);
      else v = k < 2 ? k : rec(k - 1, node.id) + rec(k - 2, node.id);
      node.value = v; events.push({ t: "return", id: node.id, value: v });
      return v;
    }
    rec(n, null);
    return { nodes: nodes, events: events };
  }
  function layoutTree(nodes) {
    var leaf = 0, maxD = 0;
    function go(id, d) {
      var nd = nodes[id]; nd.d = d; maxD = Math.max(maxD, d);
      if (nd.children.length === 0) { nd.x = leaf++; return; }
      nd.children.forEach(function (c) { go(c, d + 1); });
      nd.x = (nodes[nd.children[0]].x + nodes[nd.children[nd.children.length - 1]].x) / 2;
    }
    go(0, 0);
    return { leaves: leaf, depth: maxD };
  }
  function genList(n, seed, order) {
    var r = Kit_rng(seed), a = [];
    for (var i = 0; i < n; i++) a.push(Math.round((30 + 100 * r()) * 10) / 10);
    if (order === 0) return a;
    a.sort(function (x, y) { return x - y; });
    if (order === 2) a.reverse();
    if (order === 1) { for (var s = 0; s < Math.max(1, Math.round(n / 8)); s++) { var p = Math.floor(r() * (n - 1)); var t = a[p]; a[p] = a[p + 1]; a[p + 1] = t; } }
    return a;
  }
  function Kit_rng(seed) {
    var a = seed >>> 0;
    return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; var t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function commas(x) { return Math.round(x).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function sci(x) { return x >= 1e12 ? (x / 1e12).toFixed(1) + " trillion" : x >= 1e9 ? (x / 1e9).toFixed(1) + " billion" : x >= 1e6 ? (x / 1e6).toFixed(1) + " million" : commas(x); }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { insertionFrames: insertionFrames, mergeFrames: mergeFrames, topkFrames: topkFrames, bsFrames: bsFrames, buildTable: buildTable, lookupCount: lookupCount, simulate: simulate, layoutTree: layoutTree, genList: genList, DEFAULT16: DEFAULT16 };
  }
  if (typeof document === "undefined") return;

  // ======================= helpers =======================
  var H = function (tag, attrs, parent, text) { return Kit.html(tag, attrs, parent, text); };
  function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
  function codeLines(box, lines, state) {
    clear(box);
    lines.forEach(function (t, i) {
      var d = H("span", { class: "ln" + (state && state[i] ? " " + state[i] : "") }, box);
      if (t === "") { d.appendChild(document.createTextNode(" ")); return; }
      var m = t.match(/^(.*?)(\s+#.*)?$/);
      d.appendChild(document.createTextNode(m[1]));
      if (m[2]) H("span", { class: "cm" }, d, m[2]);
    });
  }
  function playLoop(btnHost, stepFn, doneFn, resetFn, ms) {
    var timer = 0, play;
    function stop() { if (timer) clearInterval(timer); timer = 0; play.textContent = "Play"; }
    var stepB = Kit.button(btnHost, "Step", function () { stop(); stepFn(); });
    play = Kit.button(btnHost, "Play", function () {
      if (timer) { stop(); return; }
      if (doneFn()) resetFn();
      play.textContent = "Pause";
      timer = setInterval(function () { if (doneFn()) { stop(); return; } stepFn(); }, ms());
    });
    Kit.button(btnHost, "Reset", function () { stop(); resetFn(); });
    return { stop: stop };
  }
  function svgSetup(svg, W, Hh) { svg.setAttribute("viewBox", "0 0 " + W + " " + Hh); clear(svg); }

  document.addEventListener("DOMContentLoaded", function () {
    stacks(); lookup(); search(); sorting(); topk(); recursion(); toyTree(); bigO(); attention(); tracing();
  });

  // ======================= 1. stacks and queues =======================
  function stacks() {
    var stack = [4127, 3305, 5519], queue = [4127, 3305, 5519], sOut = [], qOut = [], next = 3, MAX = 8;
    var bar = Kit.$("sq-bar");
    Kit.button(bar, "Add the next tag to both", function () { if (next >= TAGS.length || stack.length >= MAX) { say(next >= TAGS.length ? "All eight tags have been added. Press Reset to start again." : "Both are full for this demo. Remove a tag first."); return; } var t = TAGS[next++]; stack.push(t); queue.push(t); say("Tag " + t + " went on top of the stack and to the back of the queue."); draw(); });
    Kit.button(bar, "Remove from the stack", function () { if (!stack.length) { say("The stack is empty. A pop here would raise an IndexError in Python."); return; } var t = stack.pop(); sOut.push(t); say("pop() gave back " + t + ", the newest tag in the stack."); draw(); });
    Kit.button(bar, "Remove from the queue", function () { if (!queue.length) { say("The queue is empty."); return; } var t = queue.shift(); qOut.push(t); say("popleft() gave back " + t + ", the oldest tag in the queue."); draw(); });
    Kit.button(bar, "Reset", function () { stack = [4127, 3305, 5519]; queue = [4127, 3305, 5519]; sOut = []; qOut = []; next = 3; say("The three tags are back in. Remove two from both and compare."); draw(); });
    function say(t) { Kit.$("sq-status").textContent = t; }
    function draw() {
      var sb = Kit.$("sq-stack"), qb = Kit.$("sq-queue"); clear(sb); clear(qb);
      stack.forEach(function (t, i) { H("div", { class: "it" + (i === stack.length - 1 ? " top" : "") }, sb, String(t)); });
      queue.forEach(function (t, i) { H("div", { class: "it" + (i === 0 ? " front" : "") }, qb, String(t)); });
      if (queue.length) { var l = H("span", { class: "lbl" }, qb, "← front is the left end"); }
      Kit.$("sq-sout").textContent = sOut.length ? sOut.join(", ") : "none yet";
      Kit.$("sq-qout").textContent = qOut.length ? qOut.join(", ") : "none yet";
    }
    say("Tags 4127, 3305 and 5519 went in, in that order. Remove two from the stack and two from the queue, then compare.");
    draw();
  }

  // ======================= 2. lookup race =======================
  function lookup() {
    var sizes = [10, 100, 1000, 10000, 100000, 1000000], N = 10000, T = buildTable(N), cells = 40, press = 0, stopAnim = null;
    var rng = Kit_rng(5), pending = 0;
    var chipBox = H("span", {}, Kit.$("lk-bar"));
    Kit.chips(chipBox, sizes.map(commas), 3, function (i) { N = sizes[i]; T = buildTable(N); reset(); });
    Kit.button(Kit.$("lk-bar2"), "Find a tag", function () { go(); });
    Kit.button(Kit.$("lk-bar2"), "Run 1,000 lookups", function () { avg(); });
    var nc = function () { return Math.min(cells, N); };
    function strips(scanFrac, hitCell, slotCell) {
      var ls = Kit.$("lk-list"), ds = Kit.$("lk-dict"); clear(ls); clear(ds);
      var n = nc();
      for (var i = 0; i < n; i++) { H("i", { class: i < scanFrac ? "scan" : "" }, ls); }
      if (hitCell >= 0) ls.children[Math.min(n - 1, hitCell)].className = "hit";
      for (var j = 0; j < n; j++) H("i", {}, ds);
      if (slotCell >= 0) ds.children[Math.min(n - 1, slotCell)].className = "hit";
    }
    function reset() { if (stopAnim) stopAnim(); strips(0, -1, -1); Kit.setText("lk-lc", "0"); Kit.setText("lk-dc", "0"); Kit.$("lk-avg").textContent = ""; var note = N > cells ? "One cell represents " + commas(N / cells) + " tags." : "One cell is one tag."; Kit.$("lk-avg").textContent = note; }
    function go() {
      if (stopAnim) stopAnim();
      var pos = Math.floor(rng() * N), key = 4101 + pos, lc = pos + 1, dc = lookupCount(T, key), n = nc();
      var hitCell = Math.floor(pos / N * n), slotCell = Math.floor(hashOf(key, T.M) / T.M * n);
      Kit.setText("lk-dc", "0");
      Kit.$("lk-avg").textContent = "Target: tag " + commas(key) + ". It is at position " + commas(pos) + " in the list. " + (N > cells ? "One cell represents " + commas(N / cells) + " tags." : "");
      stopAnim = Kit.animate(Math.min(1400, 400 + n * 20), function (p) {
        var s = Math.ceil(p * lc), cellsScanned = Math.ceil(p * (hitCell + 1));
        strips(cellsScanned, p >= 1 ? hitCell : -1, p >= 1 ? slotCell : -1);
        Kit.setText("lk-lc", commas(s));
        if (p >= 1) Kit.setText("lk-dc", String(dc));
      });
    }
    function avg() {
      var r = Kit_rng(100 + (++press)), sl = 0, sd = 0, M = 1000;
      for (var i = 0; i < M; i++) { var pos = Math.floor(r() * N); sl += pos + 1; sd += lookupCount(T, 4101 + pos); }
      Kit.$("lk-avg").innerHTML = "";
      var a = H("span", {}, Kit.$("lk-avg")); a.appendChild(document.createTextNode("Average over 1,000 random tags: list ")); H("b", {}, a, (sl / M).toLocaleString("en-US", { maximumFractionDigits: 1 })); a.appendChild(document.createTextNode(" comparisons, dictionary ")); H("b", {}, a, (sd / M).toFixed(2)); a.appendChild(document.createTextNode(" comparisons."));
    }
    reset();
    Kit.onVisible(Kit.$("lk"), go, 0.4);
  }

  // ======================= 3. binary search =======================
  function search() {
    var sizes = [16, 32, 64], n = 16, items = DEFAULT16.slice().sort(function (a, b) { return a - b; }), target = 62.2, frames = [], pos = 0, timer = 0;
    var tsel, wrap = Kit.$("bs-svg-wrap"), svg = Kit.$("bs-svg");
    var ctl = H("div", { style: "display:grid;gap:4px;justify-items:start;font-size:14.5px;color:var(--ink-2)" }, Kit.$("bs-controls")); H("span", {}, ctl, "List size");
    Kit.chips(ctl, sizes.map(String), 0, function (i) { n = sizes[i]; setItems(); });
    function setItems() {
      stop();
      if (n === 16) items = DEFAULT16.slice().sort(function (a, b) { return a - b; });
      else { var r = Kit_rng(n * 7), a = []; for (var i = 0; i < n; i++) a.push(Math.round((30 + 95 * r()) * 10) / 10); a.sort(function (x, y) { return x - y; }); for (var j = 1; j < n; j++) if (a[j] <= a[j - 1]) a[j] = Math.round((a[j - 1] + 0.1) * 10) / 10; items = a; }
      target = items[Math.floor(n / 2) + Math.floor(n / 8)];
      buildSel(); setTarget();
    }
    function buildSel() {
      var host = Kit.$("bs-controls"); if (tsel && tsel.parentNode.parentNode === host) host.removeChild(tsel.parentNode);
      var opts = [], seen = {};
      items.forEach(function (v, i) { if (seen[v] === undefined) { seen[v] = opts.length; opts.push({ value: String(v), label: v.toFixed(1) + " (position " + i + ")" }); } else opts[seen[v]].label = v.toFixed(1) + " (positions " + items.indexOf(v) + " and " + i + ")"; });
      opts.push({ value: "absent", label: "a length that is not in the list" });
      tsel = Kit.select(host, "Find the length", opts, String(target), function (v) { target = v === "absent" ? (items[Math.floor(n / 3)] + items[Math.floor(n / 3) + 1]) / 2 + 0.01 : Number(v); setTarget(); });
    }
    var bar = Kit.$("bs-bar");
    var pl = playLoop(bar, function () { if (pos < frames.length - 1) { pos++; draw(); } }, function () { return pos >= frames.length - 1; }, function () { pos = 0; draw(); }, function () { return 900; });
    function stop() { pl.stop(); }
    function setTarget() { frames = bsFrames(items, target); pos = 0; draw(); }
    function draw() {
      var P = Kit.palette(), W = Kit.width(wrap), Hh = 210, f = frames[pos], bw = W / n, maxV = 130, base = Hh - 44;
      svgSetup(svg, W, Hh);
      items.forEach(function (v, i) {
        var inW = i >= f.lo && i <= f.hi, col = P.rule, op = 1;
        if (inW) col = P.navy; else { col = P.muted; op = 0.3; }
        if (i === f.mid && f.verdict !== "empty") col = f.verdict === "found" ? P.teal : P.coral;
        var h = Math.max(4, v / maxV * (base - 20));
        Kit.el("rect", { x: i * bw + 1, y: base - h, width: Math.max(2, bw - 2), height: h, rx: 2, fill: col, opacity: op }, svg);
        if (n <= 16) { var t = Kit.el("text", { x: i * bw + bw / 2, y: base + 15, "text-anchor": "middle", "font-size": 12, fill: P.muted }, svg); t.textContent = i; }
      });
      function tag(i, txt, color, dy) { var tx = Kit.el("text", { x: i * bw + bw / 2, y: base + dy, "text-anchor": "middle", "font-size": 13, "font-weight": 600, fill: color }, svg); tx.textContent = txt; }
      if (f.verdict !== "empty") tag(f.mid, "mid", f.verdict === "found" ? P.teal : P.coral, 31);
      if (f.lo <= f.hi) { if (f.lo !== f.mid) tag(f.lo, "lo", P.navy, 31); if (f.hi !== f.mid && f.hi !== f.lo) tag(f.hi, "hi", P.navy, 31); }
      // target line
      var tl = Kit.el("text", { x: W - 4, y: 14, "text-anchor": "end", "font-size": 13, fill: P.ink2 }, svg); tl.textContent = "target " + target.toFixed(1) + " cm";
      var linear = items.indexOf(target) >= 0 ? items.indexOf(target) + 1 : n;
      var note;
      if (f.verdict === "found") note = "Step " + (pos + 1) + ": the middle item, position " + f.mid + ", is " + items[f.mid].toFixed(1) + ". That is the target. Binary search used " + (pos + 1) + " step" + (pos ? "s" : "") + "; linear search would have used " + linear + ".";
      else if (f.verdict === "low") note = "Step " + (pos + 1) + ": the middle item, position " + f.mid + ", is " + items[f.mid].toFixed(1) + ", below the target. Positions " + f.lo + " to " + f.mid + " are discarded, so lo becomes " + (f.mid + 1) + ".";
      else if (f.verdict === "high") note = "Step " + (pos + 1) + ": the middle item, position " + f.mid + ", is " + items[f.mid].toFixed(1) + ", above the target. Positions " + f.mid + " to " + f.hi + " are discarded, so hi becomes " + (f.mid - 1) + ".";
      else note = "The window is empty (lo is " + f.lo + ", hi is " + f.hi + "), so the target is not in the list. Binary search used " + pos + " steps; linear search would have used " + n + ".";
      Kit.$("bs-status").textContent = note;
      var eq = Kit.$("bs-eqn"); clear(eq);
      function box(cls, b, i) { var t2 = H("span", { class: "t " + cls }, eq); H("b", {}, t2, b); H("i", {}, t2, i); }
      var lg = Math.log2(n);
      box("c-navy", String(n), "items, n"); H("span", { class: "op" }, eq, "→"); box("c-sand", Kit.fmt(lg, 2), "log₂ n"); H("span", { class: "op" }, eq, "→"); box("c-teal", String(Math.floor(lg) + 1), "at most this many steps"); H("span", { class: "op" }, eq, "·"); box("c-coral", String(f.verdict === "empty" ? pos : pos + 1), "steps shown so far");
    }
    window.addEventListener("resize", Kit.debounce(draw)); Kit.onTheme(draw);
    buildSel(); setTarget();
  }

  // ======================= 4. sorting race =======================
  function sorting() {
    var n = 16, order = 0, seed = 0, list = DEFAULT16.slice(), fi = [], fm = [], t = 0, timer = 0, speed;
    var sizes = [8, 16, 32, 64];
    var ctl = Kit.$("sr-controls");
    var ordSel = Kit.select(ctl, "Starting order", ["Shuffled", "Nearly sorted", "Reversed", "Already sorted"].map(function (s, i) { return { value: String(i), label: s }; }), "0", function (v) { order = Number(v); newList(); });
    speed = Kit.slider(ctl, { label: "speed, comparisons per second", min: 2, max: 60, step: 1, value: 10, digits: 0 }, function () { if (timer) { stopT(); startT(); } });
    var bar = Kit.$("sr-bar");
    var chipBox = H("span", {}, bar);
    Kit.chips(chipBox, sizes.map(function (s) { return s + " sharks"; }), 1, function (i) { n = sizes[i]; seed = 0; newList(); });
    Kit.button(bar, "New shuffle", function () { seed++; newList(); });
    var playB = Kit.button(bar, "Play", function () { if (timer) { stopT(); return; } if (done()) { t = 0; } startT(); });
    Kit.button(bar, "Step", function () { stopT(); tick(); });
    Kit.button(bar, "Reset", function () { stopT(); t = 0; draw(); });
    function done() { return t >= Math.max(fi.length, fm.length) - 1; }
    function startT() { playB.textContent = "Pause"; timer = setInterval(function () { if (done()) { stopT(); return; } tick(); }, 1000 / speed.get()); }
    function stopT() { if (timer) clearInterval(timer); timer = 0; playB.textContent = "Play"; }
    function tick() { if (!done()) { t++; draw(); } }
    function newList() { stopT(); list = (n === 16 && order === 0 && seed === 0) ? DEFAULT16.slice() : genList(n, 1000 + seed * 31 + n, order); fi = insertionFrames(list); fm = mergeFrames(list); t = 0; draw(); }
    function bars(svgId, f, kind, finished) {
      var svg = Kit.$(svgId), P = Kit.palette(), W = Kit.width(svg), Hh = 170, bw = W / n, base = Hh - 6, maxV = 130;
      svgSetup(svg, W, Hh);
      if (kind === "m" && f.win) Kit.el("rect", { x: f.win[0] * bw, y: 2, width: (f.win[1] - f.win[0]) * bw, height: Hh - 4, rx: 4, fill: P.sand, opacity: 0.22 }, svg);
      f.a.forEach(function (v, i) {
        var col = P.navy, op = 0.9;
        if (finished) { col = P.teal; op = 0.9; }
        else if (kind === "i" && i <= f.front) { col = P.teal; op = 0.45; }
        if (!finished && f.cmp && (i === f.cmp[0] || i === f.cmp[1])) { col = P.coral; op = 1; }
        var h = Math.max(3, v / maxV * (base - 12));
        Kit.el("rect", { x: i * bw + 0.5, y: base - h, width: Math.max(1.5, bw - 1.5), height: h, rx: Math.min(2, bw / 4), fill: col, opacity: op }, svg);
      });
    }
    function draw() {
      var a = fi[Math.min(t, fi.length - 1)], b = fm[Math.min(t, fm.length - 1)], ia = t >= fi.length - 1, ib = t >= fm.length - 1;
      bars("sr-ins", a, "i", ia); bars("sr-mrg", b, "m", ib);
      Kit.setText("sr-ic", commas(a.count)); Kit.setText("sr-mc", commas(b.count));
      Kit.$("sr-id").textContent = ia ? "sorted" : ""; Kit.$("sr-md").textContent = ib ? "sorted" : "";
      var ci = fi[fi.length - 1].count, cm = fm[fm.length - 1].count;
      var st = Kit.$("sr-status");
      if (ia && ib) st.textContent = ci === cm ? "Both used " + commas(ci) + " comparisons." : (cm < ci ? "Merge sort finished with " + commas(cm) + " comparisons and insertion sort with " + commas(ci) + "." : "Insertion sort finished with " + commas(ci) + " comparisons and merge sort with " + commas(cm) + ".") + " The same " + n + " lengths, sorted by two methods.";
      else if (ib) st.textContent = "Merge sort is finished with " + commas(cm) + " comparisons. Insertion sort is still working.";
      else if (ia) st.textContent = "Insertion sort is finished with " + commas(ci) + " comparisons. Merge sort is still working.";
      else st.textContent = t === 0 ? "Press Play or Step. One tick is one comparison for both algorithms." : "Comparison " + t + ". The two bars in coral are the ones being compared.";
      var eq = Kit.$("sr-eqn"); clear(eq);
      function box(cls, v, l) { var s = H("span", { class: "t " + cls }, eq); H("b", {}, s, v); H("i", {}, s, l); }
      box("c-navy", String(n), "items, n"); H("span", { class: "op" }, eq, "→");
      box("c-sand", commas(n * (n - 1) / 2), "n(n−1)/2, insertion worst case"); box("c-sand", commas(n * Math.ceil(Math.log2(n))), "n ⌈log₂ n⌉, merge bound"); H("span", { class: "op" }, eq, "→");
      box("c-coral", ia ? commas(ci) : "…", "insertion, this list"); box("c-teal", ib ? commas(cm) : "…", "merge, this list");
    }
    window.addEventListener("resize", Kit.debounce(draw)); Kit.onTheme(draw);
    newList();
  }

  // ======================= 5. top-k =======================
  function topk() {
    var k = 3, fr = [], pos = 0;
    var ks = Kit.slider(Kit.$("tk-controls"), { label: "k, how many to keep", min: 1, max: 8, step: 1, value: 3, digits: 0 }, function (v) { k = v; reset(); });
    var bar = Kit.$("tk-bar");
    var pl = playLoop(bar, function () { if (pos < fr.length - 1) { pos++; draw(); } }, function () { return pos >= fr.length - 1; }, function () { pos = 0; draw(); }, function () { return 600; });
    var sortCount = mergeFrames(DEFAULT16)[mergeFrames(DEFAULT16).length - 1].count;
    function reset() { pl.stop(); fr = topkFrames(DEFAULT16, k); pos = 0; draw(); }
    function draw() {
      var f = fr[pos], box = Kit.$("tk-cells"); clear(box);
      DEFAULT16.forEach(function (v, i) {
        var c = H("div", { class: "c" + (i === f.i ? " cur" : "") + (f.board.indexOf(v) >= 0 && i <= f.i ? " kept" : "") + (i < f.i && f.board.indexOf(v) < 0 ? " done" : "") }, box); H("small", {}, c, "item " + i); H("b", {}, c, v.toFixed(1));
      });
      var bd = Kit.$("tk-board"); clear(bd);
      for (var j = 0; j < k; j++) { var b = H("div", { class: "vb" + (j < f.board.length ? "" : " gone") }, bd); H("small", {}, b, j === 0 ? "best" : "place " + (j + 1)); H("b", {}, b, j < f.board.length ? f.board[j].toFixed(1) : "empty"); }
      Kit.setText("tk-c", String(f.c)); Kit.setText("tk-sort", String(sortCount));
      Kit.$("tk-status").textContent = (pos === fr.length - 1 ? "Done. The board has the " + k + " largest lengths, best first. " : "Item " + (f.i + 1) + " of 16. ") + f.text;
    }
    reset();
  }

  // ======================= 6. recursion =======================
  function recursion() {
    var kind = "total", sim, lay, ev = 0, timer = 0;
    var bar = Kit.$("rc-bar"), wrap = Kit.$("rc-svg-wrap"), svg = Kit.$("rc-svg");
    var ctl = Kit.$("rc-controls");
    var sel = Kit.select(ctl, "function", [{ value: "total", label: "total(n) = n + total(n - 1)" }, { value: "fib", label: "fib(n) = fib(n - 1) + fib(n - 2)" }], "total", function (v) { kind = v; setN(); });
    var ns = Kit.slider(ctl, { label: "n", min: 0, max: 8, step: 1, value: 4, digits: 0 }, function () { build(); });
    function setN() { ns.input.max = kind === "fib" ? 6 : 8; if (ns.get() > Number(ns.input.max)) ns.set(Number(ns.input.max)); if (kind === "fib" && ns.get() === 4) {} build(); }
    var pl = playLoop(bar, function () { if (ev < sim.events.length) { ev++; draw(); } }, function () { return ev >= sim.events.length; }, function () { ev = 0; draw(); }, function () { return 550; });
    Kit.button(bar, "Run to the end", function () { pl.stop(); ev = sim.events.length; draw(); });
    function build() { pl.stop(); sim = simulate(kind, ns.get()); lay = layoutTree(sim.nodes); ev = 0; draw(); }
    function draw() {
      var P = Kit.palette(), W = Kit.width(wrap), dx = Math.min(54, (W - 40) / Math.max(1, lay.leaves)), dy = 44, Hh = (lay.depth + 1) * dy + 12;
      svgSetup(svg, W, Hh);
      var ox = (W - lay.leaves * dx) / 2 + dx / 2;
      var called = {}, returned = {}, stack = [], calls = 0, deepest = 0, last = null;
      for (var e = 0; e < ev; e++) {
        var evn = sim.events[e];
        if (evn.t === "call") { called[evn.id] = true; stack.push(evn.id); calls++; deepest = Math.max(deepest, stack.length); }
        else { returned[evn.id] = true; stack.pop(); }
        last = evn;
      }
      function px(nd) { return ox + nd.x * dx; } function py(nd) { return 24 + nd.d * dy; }
      sim.nodes.forEach(function (nd) { if (nd.parent !== null) { var pn = sim.nodes[nd.parent]; Kit.el("line", { x1: px(pn), y1: py(pn) + 14, x2: px(nd), y2: py(nd) - 14, stroke: P.muted, "stroke-width": 1.5, opacity: called[nd.id] ? 1 : 0.18 }, svg); } });
      sim.nodes.forEach(function (nd) {
        if (!called[nd.id]) { Kit.el("circle", { cx: px(nd), cy: py(nd), r: 14, fill: "none", stroke: P.muted, "stroke-width": 1.2, "stroke-dasharray": "3 3", opacity: 0.4 }, svg); return; }
        var on = stack.indexOf(nd.id) >= 0, top = stack[stack.length - 1] === nd.id, g = Kit.el("g", { class: "node" }, svg);
        Kit.el("circle", { cx: px(nd), cy: py(nd), r: 14, fill: returned[nd.id] ? P.teal : P.paper, "fill-opacity": returned[nd.id] ? 0.28 : 1, stroke: top ? P.coral : returned[nd.id] ? P.teal : P.navy, "stroke-width": top ? 3 : 1.8 }, g);
        var t = Kit.el("text", { x: px(nd), y: py(nd) + 5, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: P.ink }, g); t.textContent = nd.k;
        if (returned[nd.id]) { var v = Kit.el("text", { x: px(nd) + 18, y: py(nd) + 5, "font-size": 13, fill: P.teal, "font-weight": 600 }, g); v.textContent = "= " + nd.value; }
      });
      Kit.$("rc-legend").textContent = "A circle marked k is the call " + kind + "(k). Dashed circles are calls not yet made. A value appears when the call returns.";
      var st = Kit.$("rc-stack"); clear(st); st.className = "callstack";
      if (!stack.length && !(last && last.t === "return")) H("div", { class: "empty" }, st, "empty: no call is waiting");
      stack.forEach(function (id, i) { H("div", { class: i === stack.length - 1 ? "top" : "" }, st, kind + "(" + sim.nodes[id].k + ")" + (i === stack.length - 1 ? "  ← running" : "  waiting")); });
      if (last && last.t === "return") H("div", { class: "ret" }, st, kind + "(" + sim.nodes[last.id].k + ") returns " + last.value);
      Kit.setText("rc-n", String(calls)); Kit.setText("rc-d", String(deepest)); Kit.setText("rc-a", ev >= sim.events.length ? String(sim.nodes[0].value) : "?");
      Kit.$("rc-svg").setAttribute("aria-label", "The tree of calls for " + kind + "(" + ns.get() + "), with " + sim.nodes.length + " calls in all");
    }
    window.addEventListener("resize", Kit.debounce(draw)); Kit.onTheme(draw);
    build();
  }

  function toyTree() {
    var svg = Kit.$("tr-svg"), wrap = Kit.$("tr-svg-wrap");
    var L = Kit.slider(Kit.$("tr-controls"), { label: "length_cm", min: 30, max: 120, step: 0.5, value: 62.5, digits: 1 }, draw);
    var D = Kit.slider(Kit.$("tr-controls"), { label: "depth_m", min: 0, max: 80, step: 1, value: 22, digits: 0 }, draw);
    function draw() {
      var P = Kit.palette(), W = Kit.width(wrap), Hh = 250, l = L.get(), d = D.get();
      svgSetup(svg, W, Hh);
      var pos = { r: [W * 0.5, 36], a: [W * 0.17, 200], b: [W * 0.665, 110], c: [W * 0.5, 205], e: [W * 0.83, 205] };
      var lbl = { r: "length < 58?", b: "depth < 30?", a: "juvenile", c: "adult, inshore", e: "adult, offshore" };
      var yes1 = l < 58, yes2 = d < 30;
      var path = yes1 ? ["r", "a"] : yes2 ? ["r", "b", "c"] : ["r", "b", "e"];
      var edges = [["r", "a", "yes"], ["r", "b", "no"], ["b", "c", "yes"], ["b", "e", "no"]];
      edges.forEach(function (e) {
        var on = path.indexOf(e[0]) >= 0 && path.indexOf(e[1]) >= 0 && path.indexOf(e[1]) === path.indexOf(e[0]) + 1;
        Kit.el("line", { x1: pos[e[0]][0], y1: pos[e[0]][1] + 14, x2: pos[e[1]][0], y2: pos[e[1]][1] - 14, stroke: on ? P.coral : P.muted, "stroke-width": on ? 3 : 1.5 }, svg);
        var t = Kit.el("text", { x: (pos[e[0]][0] + pos[e[1]][0]) / 2 + (e[2] === "yes" ? -10 : 10), y: (pos[e[0]][1] + pos[e[1]][1]) / 2, "text-anchor": "middle", "font-size": 13, fill: on ? P.coral : P.muted, "font-weight": on ? 600 : 400 }, svg); t.textContent = e[2];
      });
      Object.keys(pos).forEach(function (k) {
        var leaf = k === "a" || k === "c" || k === "e", on = path.indexOf(k) >= 0, w = leaf ? Math.min(118, W * 0.3) : Math.min(108, W * 0.3);
        Kit.el("rect", { x: pos[k][0] - w / 2, y: pos[k][1] - 15, width: w, height: 30, rx: leaf ? 15 : 5, fill: leaf && on ? P.teal : P.paper, "fill-opacity": leaf && on ? 0.3 : 1, stroke: on ? P.coral : P.navy, "stroke-width": on ? 2.6 : 1.6 }, svg);
        var t = Kit.el("text", { x: pos[k][0], y: pos[k][1] + 5, "text-anchor": "middle", "font-size": 14, fill: P.ink, "font-weight": on ? 600 : 400 }, svg); t.textContent = lbl[k];
      });
      var st = Kit.$("tr-stack"); clear(st);
      var lines = [];
      lines.push("predict(root, shark): tests length_cm < 58. " + l.toFixed(1) + " < 58 is " + (yes1 ? "True" : "False") + ", so it calls predict on the '" + (yes1 ? "yes" : "no") + "' tree.");
      if (!yes1) lines.push("predict(second test, shark): tests depth_m < 30. " + d + " < 30 is " + (yes2 ? "True" : "False") + ", so it calls predict on the '" + (yes2 ? "yes" : "no") + "' tree.");
      var leaf = yes1 ? "juvenile" : yes2 ? "adult, inshore" : "adult, offshore";
      lines.push("predict(leaf, shark): a leaf is the base case. It returns '" + leaf + "'.");
      lines.forEach(function (t, i) { H("div", { class: i === lines.length - 1 ? "ret" : "" }, st, t); });
      svg.setAttribute("aria-label", "A small decision tree; a shark of length " + l + " cm at depth " + d + " m is sent to the leaf " + leaf);
    }
    window.addEventListener("resize", Kit.debounce(draw)); Kit.onTheme(draw);
    draw();
  }

  // ======================= 7. Big-O =======================
  function bigO() {
    var xmax = 100, log = false, c = 1;
    var ctl = Kit.$("bo-controls");
    var ns = Kit.slider(ctl, { label: "input size n", min: 1, max: 1000, step: 1, value: 60, digits: 0 }, draw);
    var cs = Kit.slider(ctl, { label: "constant factor c on the n log n curve", min: 1, max: 100, step: 1, value: 1, digits: 0 }, function (v) { c = v; draw(); });
    var bar = Kit.$("bo-bar");
    var g1 = H("span", {}, bar); Kit.chips(g1, ["linear axes", "log-log axes"], 0, function (i) { log = i === 1; draw(); });
    var lab = H("span", { class: "label" }, bar, "Show n up to"); var g2 = H("span", {}, bar); Kit.chips(g2, ["10", "100", "1,000"], 1, function (i) { xmax = [10, 100, 1000][i]; draw(); });
    function f(kind, n) { return kind === 0 ? Math.log2(n) : kind === 1 ? n : kind === 2 ? c * n * Math.log2(n) : n * n; }
    function cross() { var last = 1; for (var n = 2; n < 300000; n++) if (!(n > c * Math.log2(n))) last = n; return last + 1; }
    function draw() {
      var n = Math.min(ns.get(), xmax), names = ["log n", "n", c === 1 ? "n log n" : c + " n log n", "n²"], xs = [], i;
      var N = 160; for (i = 0; i < N; i++) xs.push(log ? Math.pow(xmax, i / (N - 1)) : 1 + (xmax - 1) * i / (N - 1));
      ns.input.max = xmax; if (ns.get() > xmax) ns.set(xmax);
      Kit.plot("bo-plot", function (P) {
        var cols = [P.sand, P.navy, P.teal, P.coral], tr = [];
        for (var k = 0; k < 4; k++) tr.push({ x: xs, y: xs.map(function (x) { return f(k, x); }), mode: "lines", line: { color: cols[k], width: 3 }, hovertemplate: names[k] + ": %{y:,.0f} steps at n = %{x:.0f}<extra></extra>" });
        tr.push({ x: [n, n, n, n], y: [0, 1, 2, 3].map(function (k) { return Math.max(f(k, n), 0.0001); }), mode: "markers", marker: { color: cols, size: 11, line: { color: P.paper, width: 2 } }, hoverinfo: "skip" });
        return tr;
      }, function (P) {
        var cols = [P.sand, P.navy, P.teal, P.coral];
        // On linear axes the three lower curves end close together, so spread their end labels at least 15 px apart.
        var plotH = Math.max(160, (Kit.$("bo-plot").clientHeight || 340) - 60), ytop = f(3, xmax) * 1.05, nudge = [0, 0, 0, 0];
        if (!log) {
          var px = [0, 1, 2, 3].map(function (k) { return f(k, xmax) / ytop * plotH; }), prev = -6;
          for (var j = 0; j < 4; j++) { var want = Math.max(px[j], prev + 15); nudge[j] = want - px[j]; prev = want; }
        }
        var ann = [0, 1, 2, 3].map(function (k) { var y = f(k, xmax); return { x: log ? Math.log10(xmax) : xmax, y: log ? Math.log10(Math.max(y, 1e-3)) : y, xanchor: "left", yanchor: "middle", text: names[k], showarrow: false, font: { size: 13, color: cols[k] }, xshift: 6, yshift: nudge[k] }; });
        return { xaxis: { title: { text: "input size n" }, type: log ? "log" : "linear" }, yaxis: { title: { text: "steps" }, type: log ? "log" : "linear", rangemode: log ? "normal" : "tozero" }, margin: { l: 60, r: 78, t: 12, b: 48 }, annotations: ann };
      });
      var ro = Kit.$("bo-read"); clear(ro);
      function item(l, v) { var s = H("span", {}, ro, l + " "); H("b", {}, s, v); }
      item("at n = " + n + ":", ""); item("log n", commas(f(0, n))); item("n", commas(f(1, n))); item("c n log n", commas(f(2, n))); item("n²", commas(f(3, n)));
      var r = H("span", {}, ro, "Doubling n multiplies the steps by: "); H("b", {}, r, "log n +1, n ×2, n log n ×" + (f(2, 2 * n) / f(2, n)).toFixed(2) + ", n² ×4");
      var x = cross(); var cr = H("span", {}, ro, "n² is above c n log n from n = "); H("b", {}, cr, x > xmax ? commas(x) + " (off this chart)" : commas(x));
      var eq = Kit.$("bo-eqn"); clear(eq);
      function box(cls, b, i) { var t = H("span", { class: "t " + cls }, eq); H("b", {}, t, b); H("i", {}, t, i); }
      var a = 3 * n * n, b2 = 5 * n, tot = a + b2 + 2;
      box("c-coral", commas(a), "3n²"); H("span", { class: "op" }, eq, "+"); box("c-navy", commas(b2), "5n"); H("span", { class: "op" }, eq, "+"); box("c-sand", "2", "constant"); H("span", { class: "op" }, eq, "="); box("c-teal", commas(tot), "T(n) steps"); H("span", { class: "op" }, eq, "→"); box("c-coral", (100 * a / tot).toFixed(1) + "%", "share of the n² term, n = " + n);
    }
    draw();
  }

  function attention() {
    var s = Kit.slider(Kit.$("at-controls"), { label: "prompt length n, in tokens", min: 2, max: 5.3, step: 0.01, value: 3.6, digits: 2, show: function (v) { return commas(Math.pow(10, v)); } }, draw);
    function draw() {
      var n = Math.round(Math.pow(10, s.get())), pairs = n * n, ratio = pairs / 1e6;
      var eq = Kit.$("at-eqn"); clear(eq);
      function box(cls, b, i) { var t = H("span", { class: "t " + cls }, eq); H("b", {}, t, b); H("i", {}, t, i); }
      box("c-navy", commas(n), "tokens, n"); H("span", { class: "op" }, eq, "→"); box("c-coral", sci(pairs), "pairs, n²"); H("span", { class: "op" }, eq, "→"); box("c-teal", "×" + (ratio >= 100 ? commas(ratio) : ratio.toFixed(2)), "versus a 1,000-token prompt");
    }
    draw();
  }

  // ======================= 8. tracing by hand =======================
  function tracing() {
    Kit.data("data/traces.json").then(function (D) {
      var progs = D.programs, cur = 0, k, state, rows, outText, score, asked, tries, ended, pendingWrong;
      var bar = Kit.$("tc-bar"), chipBox = H("span", {}, bar);
      Kit.chips(chipBox, progs.map(function (p, i) { return (i + 1) + ". " + p.title; }), 0, function (i) { cur = i; start(); });
      Kit.button(bar, "Restart", function () { start(); });
      function P() { return progs[cur]; }
      function start() { k = 0; state = {}; rows = []; outText = []; score = 0; asked = 0; tries = 0; ended = false; autoRun(); draw(); }
      function scalar(v) { return typeof v === "string"; }
      function apply(s) {
        Object.keys(s.changes).forEach(function (n) { if (scalar(s.changes[n])) state[n] = s.changes[n]; });
        if (s.out !== null) outText.push(s.out);
        rows.push({ line: s.line, state: Object.assign({}, state), changed: Object.keys(s.changes).filter(function (n) { return scalar(s.changes[n]); }), out: s.out, next: s.next });
        k++; tries = 0;
      }
      function autoRun() { while (k < P().steps.length && P().steps[k].auto) apply(P().steps[k]); }
      function fmtV(v) { return typeof v === "number" && !Number.isInteger(v) ? String(v) : typeof v === "number" ? String(v) : String(v); }
      function eqv(a, b) { var x = parseFloat(a), y = parseFloat(b); if (!isNaN(x) && !isNaN(y) && /^-?[\d.]+$/.test(String(a).trim())) return Math.abs(x - y) < 1e-9; return String(a).trim().replace(/\s+/g, " ") === String(b).trim().replace(/\s+/g, " "); }
      function options(s) {
        var lines = P().lines, set = {}, hdr = null;
        for (var i = s.line - 2; i >= 0; i--) if (/^\s*(while|for)\b/.test(lines[i])) { hdr = i + 1; break; }
        set[s.line + 1] = 1; set[s.next] = 1; if (hdr) set[hdr] = 1; if (Object.keys(set).length < 3) set[lines.length] = 1;
        return Object.keys(set).map(Number).sort(function (a, b) { return a - b; }).slice(0, 4);
      }
      var inputs = {};
      function draw() {
        var p = P(), s = k < p.steps.length ? p.steps[k] : null;
        var st = []; if (s) st[s.line - 1] = "on";
        codeLines(Kit.$("tc-code"), p.lines.map(function (t, i) { return t; }), st);
        // prefix line numbers
        [].forEach.call(Kit.$("tc-code").children, function (n, i) { n.insertBefore(document.createTextNode(String(i + 1).padStart(2, " ") + "  "), n.firstChild); });
        var tb = Kit.$("tc-table"); clear(tb);
        var th = H("tr", {}, H("thead", {}, tb));
        ["step", "line"].concat(p.vars).concat(["prints", "next line"]).forEach(function (h) { H("th", {}, th, h); });
        var body = H("tbody", {}, tb);
        rows.forEach(function (r, ri) {
          var tr = H("tr", {}, body); H("td", {}, tr, String(ri + 1)); H("td", {}, tr, String(r.line));
          p.vars.forEach(function (v) { H("td", { class: r.changed.indexOf(v) >= 0 ? "new" : "" }, tr, v in r.state ? fmtV(r.state[v]) : ""); });
          H("td", {}, tr, r.out !== null ? r.out : ""); H("td", {}, tr, r.next ? String(r.next) : "end");
        });
        inputs = {};
        if (s) {
          var tr = H("tr", {}, body); H("td", { class: "pend" }, tr, String(rows.length + 1)); H("td", { class: "pend" }, tr, String(s.line));
          var prev = rows.length ? rows[rows.length - 1].state : {};
          p.vars.forEach(function (v) {
            var td = H("td", { class: "pend" }, tr);
            if (s.kind === "assign" && v in s.changes && scalar(s.changes[v])) { inputs[v] = H("input", { type: "text", "aria-label": "new value of " + v, inputmode: "decimal" }, td); }
            else td.textContent = v in prev ? fmtV(prev[v]) : "";
          });
          var tdp = H("td", { class: "pend" }, tr); if (s.kind === "print") inputs.__out = H("input", { type: "text", "aria-label": "what is printed" }, tdp);
          var tdn = H("td", { class: "pend nl" }, tr);
          if (s.kind === "branch") options(s).forEach(function (ln) { var b = H("button", { type: "button" }, tdn, String(ln)); b.addEventListener("click", function () { guessNext(ln); }); });
          else tdn.textContent = s.next ? String(s.next) : "end";
          var first = Object.keys(inputs)[0]; if (first && document.activeElement === document.body) {}
        }
        var panel = Kit.$("tc-panel"); clear(panel);
        var box = H("div", { class: "scorebox" }, panel);
        if (s) {
          var q = s.kind === "assign" ? "Type the new value in the highlighted row." : s.kind === "print" ? "What does this line print? Type it in the highlighted row." : "Which line runs next? Press its number in the highlighted row.";
          H("p", { style: "margin:0 0 4px" }, box, "Line " + s.line + " runs:");
          H("pre", { style: "margin:0 0 8px;padding:4px 10px;background:var(--bg);border:1px solid var(--rule);border-radius:4px;font-size:13.5px;white-space:pre-wrap" }, box, p.lines[s.line - 1].trim());
          H("p", { style: "margin:0 0 6px" }, box, q);
          var br = H("div", { class: "btnrow" }, box);
          if (s.kind !== "branch") Kit.button(br, "Check", check);
          Kit.button(br, "Show answer", reveal);
          if (tries) H("p", { style: "margin:8px 0 0;color:var(--coral)" }, box, "Not yet. Re-read the line and the values in the row above.");
        } else {
          ended = true;
          H("p", { style: "margin:0 0 6px" }, box, "The program ended. Output: " + (outText.join(" | ") || "none") + ".");
          H("p", { style: "margin:0 0 6px" }, box, p.note);
        }
        H("p", { style: "margin:8px 0 0;font-size:14px;color:var(--muted)" }, box, asked ? "Predicted right on the first try: " + score + " of " + asked + "." : "No predictions yet.");
        Kit.$("tc-status").textContent = s ? "Step " + (rows.length + 1) + " of " + p.steps.length + ". Line " + s.line + " is next." : "Done after " + rows.length + " steps.";
      }
      function check() {
        var s = P().steps[k], ok = true;
        Object.keys(inputs).forEach(function (n) {
          var el = inputs[n], expected = n === "__out" ? s.out : s.changes[n];
          var good = eqv(el.value, expected); el.classList.toggle("bad", !good); if (!good) ok = false;
        });
        if (ok) { if (tries === 0) score++; asked++; apply(s); autoRun(); draw(); }
        else { tries++; if (tries === 1) {} draw2(); }
      }
      function draw2() { var vals = {}; Object.keys(inputs).forEach(function (n) { vals[n] = inputs[n].value; }); draw(); Object.keys(vals).forEach(function (n) { if (inputs[n]) { inputs[n].value = vals[n]; inputs[n].classList.add("bad"); } }); }
      function guessNext(ln) { var s = P().steps[k]; if (ln === s.next) { if (tries === 0) score++; asked++; apply(s); autoRun(); draw(); } else { tries++; draw(); } }
      function reveal() { var s = P().steps[k]; if (tries === 0) asked++; apply(s); autoRun(); draw(); }
      start();
    }).catch(function (e) { Kit.$("tc-status").textContent = "The trace data did not load."; });
  }
})();
