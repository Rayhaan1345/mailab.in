// D1 Tokenisation. Figures: units (characters, words, pieces), a byte-pair-encoding trainer, a live tokenizer,
// a vocabulary lookup, padding with special tokens, a bytes explorer and a four-language token comparison.
// The pure logic (train, tokenise, vocabulary) is exported with module.exports for node tests.
(function () {
  "use strict";
  var SP = "▁"; // the mark for "a space came before this word"
  var BASE = "abcdefghijklmnopqrstuvwxyz0123456789.,'!?-".split("").concat([SP]); // 43 symbols
  var SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"];
  // Word counts of a tiny, made-up SharkCam corpus.
  var CORPUS = [["shark", 10], ["sharks", 7], ["sharkcam", 5], ["hammerhead", 4], ["hammerheads", 3], ["bamboo", 3], ["dock", 5], ["docks", 3],
    ["dive", 4], ["dives", 3], ["diver", 2], ["tag", 4], ["tags", 3], ["tagged", 2], ["tide", 3]];

  // ---------- byte-pair encoding: training ----------
  function pairCounts(segs) {
    var map = {}, order = [];
    segs.forEach(function (s, wi) {
      var c = CORPUS[wi][1];
      for (var i = 0; i + 1 < s.length; i++) {
        var key = s[i] + "\u0001" + s[i + 1];
        if (!map[key]) { map[key] = { a: s[i], b: s[i + 1], count: 0, first: order.length, parts: {} }; order.push(key); }
        map[key].count += c; map[key].parts[wi] = (map[key].parts[wi] || 0) + 1;
      }
    });
    var list = order.map(function (k) { return map[k]; });
    list.sort(function (x, y) { return y.count - x.count || x.first - y.first; }); // ties: the pair met first wins
    return list;
  }
  function applyMerge(seg, a, b) {
    var out = [], i = 0;
    while (i < seg.length) {
      if (i + 1 < seg.length && seg[i] === a && seg[i + 1] === b) { out.push(a + b); i += 2; } else { out.push(seg[i]); i++; }
    }
    return out;
  }
  function train() {
    var states = [CORPUS.map(function (w) { return [SP].concat(Array.from(w[0])); })], merges = [];
    for (;;) {
      var cur = states[states.length - 1], list = pairCounts(cur);
      if (!list.length) break;
      var best = list[0];
      merges.push({ a: best.a, b: best.b, tok: best.a + best.b, count: best.count });
      states.push(cur.map(function (s) { return applyMerge(s, best.a, best.b); }));
    }
    return { states: states, merges: merges };
  }
  var T = train();
  var MAXK = T.merges.length;

  // ---------- tokenising new text with the first k merges ----------
  function utf8(ch) { return Array.from(new TextEncoder().encode(ch)); }
  function hex2(b) { return (b < 16 ? "0" : "") + b.toString(16).toUpperCase(); }
  function pretokens(text) {
    var out = [], re = /(\s*)([\p{L}\p{M}]+|\p{N}+|[^\s\p{L}\p{M}\p{N}])/gu, m;
    while ((m = re.exec(text))) out.push({ s: m[2], space: m[1].length > 0 || out.length === 0 });
    return out;
  }
  function vocabulary(k) {
    var tokens = SPECIAL.slice(), i;
    for (i = 0; i < 256; i++) tokens.push("<0x" + hex2(i) + ">");
    tokens = tokens.concat(BASE);
    for (i = 0; i < k; i++) tokens.push(T.merges[i].tok);
    var id = {};
    tokens.forEach(function (t, j) { id[t] = j; });
    return { tokens: tokens, id: id };
  }
  // Returns [{t, kind, id}] with kind "tok", "unk" or "byte".
  function tokenise(text, k, opts) {
    opts = opts || {};
    if (opts.lower !== false) text = text.toLowerCase();
    var V = vocabulary(k), baseSet = {}, out = [];
    BASE.forEach(function (c) { baseSet[c] = true; });
    pretokens(text).forEach(function (p) {
      var chars = (p.space ? [SP] : []).concat(Array.from(p.s)), seg = [], fixed = [];
      chars.forEach(function (c) {
        if (baseSet[c]) { seg.push(c); return; }
        if (opts.unknown === "unk") { seg.push("\u0002" + fixed.length); fixed.push({ t: "<unk>", kind: "unk" }); return; }
        utf8(c).forEach(function (b) { seg.push("\u0002" + fixed.length); fixed.push({ t: "<0x" + hex2(b) + ">", kind: "byte" }); });
      });
      for (var i = 0; i < k; i++) seg = applyMerge(seg, T.merges[i].a, T.merges[i].b);
      seg.forEach(function (s) {
        if (s.charAt(0) === "\u0002") { var f = fixed[Number(s.slice(1))]; out.push({ t: f.t, kind: f.kind, id: V.id[f.t] }); } else out.push({ t: s, kind: "tok", id: V.id[s] });
      });
    });
    return out;
  }
  function decode(ids, k) {
    var V = vocabulary(k), bytes = [], s = "";
    function flush() { if (bytes.length) { s += new TextDecoder("utf-8", { fatal: false }).decode(new Uint8Array(bytes)); bytes = []; } }
    ids.forEach(function (i) {
      var t = V.tokens[i], m = /^<0x([0-9A-F]{2})>$/.exec(t || "");
      if (m) { bytes.push(parseInt(m[1], 16)); return; }
      flush();
      if (t === undefined || i < SPECIAL.length) { if (i === 1) s += "⁇"; return; }
      s += t;
    });
    flush();
    return s.split(SP).join(" ");
  }

  if (typeof module !== "undefined" && module.exports) {
    module.exports = { T: T, CORPUS: CORPUS, pairCounts: pairCounts, tokenise: tokenise, vocabulary: vocabulary, decode: decode, pretokens: pretokens, BASE: BASE };
  }
  if (typeof document === "undefined") return;

  // ---------- shared state and helpers ----------
  var S = { k: 5 }, listeners = [], DATA = null;
  function setK(k) { S.k = Math.max(0, Math.min(MAXK, k)); listeners.forEach(function (f) { f(); }); }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function showTok(t) { return t.charAt(0) === " " ? SP + t.slice(1) : t; }
  function el(tag, cls, parent, html) { var n = document.createElement(tag); if (cls) n.className = cls; if (html != null) n.innerHTML = html; if (parent) parent.appendChild(n); return n; }
  function nf(n) { return Number(n).toLocaleString("en-US"); }
  function chip(t, cls, id) { return '<span class="tk ' + cls + '">' + esc(t) + (id != null ? "<i>" + id + "</i>" : "") + "</span>"; }

  document.addEventListener("DOMContentLoaded", function () {
    Kit.data("data/tokens.json").then(function (d) {
      DATA = d;
      unitsFig(); trainerFig(); liveFig(); vocabFig(); specialFig(); bytesFig(); langFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. characters, words or pieces ----------
  function unitsFig() {
    var root = Kit.$("un"), pick = 0, unit = 2;
    var bar = Kit.$("un-bar"), out = Kit.$("un-toks"), read = Kit.$("un-read"), st = Kit.$("un-status");
    var known = {};
    DATA.messages.filter(function (m) { return m.language === "English"; }).forEach(function (m) {
      (m.text.toLowerCase().match(/[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu) || []).forEach(function (w) { known[w] = true; });
    });
    var nKnown = Object.keys(known).length;
    var sentences = DATA.presets.map(function (p) { return p.text; });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Sharkcams", "Tag 4128", "Sharkcamera", "Hinglish", "Hindi"], 0, function (i) { pick = i; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Characters", "Words", "GPT-2 pieces"], 2, function (i) { unit = i; draw(); });
    function draw() {
      var text = sentences[pick], html = "", n = 0, unk = 0, need, note;
      if (unit === 0) {
        Array.from(text).forEach(function (c) { html += chip(c === " " ? "␣" : c, "c" + (n % 3)); n++; });
        need = "about 100 letters, digits and signs for English; Unicode lists over 100,000 characters in all";
        note = "All characters are tokens, so no token is unknown, but the list is long.";
      } else if (unit === 1) {
        (text.match(/[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]/gu) || []).forEach(function (w) {
          var ok = !!known[w.toLowerCase()];
          html += chip(ok ? w : "<unk>", ok ? "c" + (n % 3) : "unk", ok ? null : "?"); n++; if (!ok) unk++;
        });
        need = "grows with the data: names, numbers, typos and spellings are all new words (this list has " + nKnown + " entries)";
        note = unk ? unk + " of " + n + " tokens are unknown. The list has no entry for them, and the model never sees what they were." : "All words are in the list, so none is unknown.";
      } else {
        DATA.presets[pick].gpt2.forEach(function (p) {
          var kind = p[1] === 1 ? "by" : "c" + (n % 3);
          html += chip(p[1] === 1 ? p[0] : showTok(p[0]), kind, p[2]); n++;
        });
        need = "50,257 pieces for GPT-2, learned from text";
        note = pick === 4 ? "Devanagari letters are cut into bytes. The coral pieces are parts of letters, shown as hex." : "Common words are one piece. New words such as sharkcamera split into known pieces, so none is unknown.";
      }
      out.innerHTML = html;
      read.innerHTML = '<span>Tokens: <b>' + n + '</b></span><span>Unknown: <b>' + unk + '</b></span><span>Vocabulary the unit needs: <b>' + need + "</b></span>";
      st.textContent = note;
    }
    draw();
  }

  // ---------- 2. the byte-pair-encoding trainer ----------
  function trainerFig() {
    var bar = Kit.$("tr-bar"), corpusBox = Kit.$("tr-corpus"), vocabBox = Kit.$("tr-vocab"), st = Kit.$("tr-status"), eq = Kit.$("tr-eqn"), read = Kit.$("tr-read");
    var timer = 0, playBtn, startLen = corpusLen(0);
    var back = Kit.button(bar, "Back", function () { stop(); setK(S.k - 1); });
    var step = Kit.button(bar, "Step", function () { stop(); setK(S.k + 1); });
    playBtn = Kit.button(bar, "Play", function () { if (timer) stop(); else play(); });
    Kit.button(bar, "Reset", function () { stop(); setK(0); });
    var sl = Kit.slider(Kit.$("tr-controls"), { label: "Merges done", min: 0, max: MAXK, step: 1, value: S.k, digits: 0 }, function (v) { stop(); setK(v); });
    function stop() { if (timer) { clearInterval(timer); timer = 0; playBtn.textContent = "Play"; } }
    function play() {
      if (S.k >= MAXK) setK(0);
      playBtn.textContent = "Pause";
      timer = setInterval(function () { if (S.k >= MAXK) { stop(); return; } setK(S.k + 1); }, 700);
    }
    function corpusLen(k) { var n = 0; T.states[k].forEach(function (s, i) { n += s.length * CORPUS[i][1]; }); return n; }
    function chars() { var n = 0; CORPUS.forEach(function (w) { n += (Array.from(w[0]).length + 1) * w[1]; }); return n; }
    function draw() {
      var k = S.k, segs = T.states[k], list = pairCounts(segs), next = list[0], last = k > 0 ? T.merges[k - 1] : null;
      sl.set(k); back.disabled = k === 0; step.disabled = k >= MAXK;
      var html = "";
      segs.forEach(function (s, wi) {
        var h = "";
        s.forEach(function (p, i) {
          var cls = "c" + (i % 3);
          if (last && p === last.tok) cls = "new";
          if (next && i + 1 < s.length && s[i] === next.a && s[i + 1] === next.b) cls += " nx";
          else if (next && i > 0 && s[i - 1] === next.a && s[i] === next.b) cls += " nx";
          h += chip(p, cls);
        });
        html += '<div class="crow"><span class="cnt">' + CORPUS[wi][1] + "&times;</span><span class=\"toks\">" + h + "</span></div>";
      });
      corpusBox.innerHTML = html;
      // pair counts
      var top = list.slice(0, 8);
      Kit.plot("tr-pairs", function (P) {
        return [{ type: "bar", orientation: "h", x: top.map(function (p) { return p.count; }).reverse(), y: top.map(function (p) { return p.a + " + " + p.b; }).reverse(),
          text: top.map(function (p) { return p.count; }).reverse(), textposition: "outside", cliponaxis: false,
          marker: { color: top.map(function (p, i) { return i === 0 ? P.coral : P.navy; }).reverse() }, hovertemplate: "%{y}: %{x} times<extra></extra>" }];
      }, { margin: { l: 96, r: 30, t: 6, b: 54 }, xaxis: { title: { text: "times the pair occurs in the corpus", standoff: 10 }, range: [0, top.length ? top[0].count * 1.18 : 1] }, yaxis: { automargin: true, tickfont: { family: "ui-monospace, Menlo, monospace", size: 12 } } });
      // vocabulary
      var v = '<span class="vlab">Alphabet (' + BASE.length + ')</span>';
      BASE.forEach(function (c) { v += chip(c, "base"); });
      v += '<span class="vlab">Merged pieces (' + k + ")</span>";
      for (var i = 0; i < k; i++) v += chip(T.merges[i].tok, i === k - 1 ? "new" : "mrg");
      if (!k) v += '<span class="vnone">none yet</span>';
      vocabBox.innerHTML = v;
      // status and equation
      if (next) {
        var parts = [], total = 0;
        Object.keys(next.parts).forEach(function (wi) { var c = CORPUS[wi][1] * next.parts[wi]; total += c; parts.push([CORPUS[wi][0], c]); });
        var e = "";
        parts.forEach(function (p, i) { e += (i ? '<span class="op">+</span>' : "") + '<span class="t c-navy"><b>' + p[1] + "</b><i>" + esc(p[0]) + "</i></span>"; });
        e += '<span class="op">=</span><span class="t c-coral"><b>' + total + "</b><i>count of " + esc(next.a) + " + " + esc(next.b) + "</i></span>";
        eq.innerHTML = e;
        eq.setAttribute("aria-label", "The pair " + next.a + " and " + next.b + " occurs " + total + " times, adding the counts of the words that contain it");
      } else { eq.innerHTML = '<span class="t c-teal"><b>0</b><i>pairs left</i></span>'; }
      var msg = k === 0 ? "Start: the words are rows of single characters. The dashed pairs are the next merge." : "Merge " + k + ": " + esc(last.a) + " + " + esc(last.b) + " became " + esc(last.tok) + " (" + last.count + " times).";
      msg += next ? " Next: " + esc(next.a) + " + " + esc(next.b) + ", seen " + next.count + " times." : " No pair is left: all words are one piece.";
      st.innerHTML = msg;
      var n = corpusLen(k);
      read.innerHTML = '<span>Alphabet + merges: <b>' + (BASE.length + k) + "</b></span><span>Corpus length: <b>" + nf(n) + " pieces</b> (from " + nf(startLen) + ")</span><span>Characters per piece: <b>" + (chars() / n).toFixed(2) + "</b></span>";
    }
    listeners.push(draw);
    draw();
    Kit.onTheme(draw);
  }

  // ---------- 3. the live tokenizer ----------
  function liveFig() {
    var input = Kit.$("lv-text"), out = Kit.$("lv-toks"), read = Kit.$("lv-read"), st = Kit.$("lv-status"), lower = Kit.$("lv-lower"), unk = Kit.$("lv-unk");
    var sl = Kit.slider(Kit.$("lv-controls"), { label: "Merges used", min: 0, max: MAXK, step: 1, value: S.k, digits: 0 }, function (v) { setK(v); });
    var presets = ["hammerheads at the dock", "bamboo shark tag 4127", "sharkcam dives", "spadenose", "छोटी बांस शार्क", "Sassoon Dock"];
    var pb = Kit.$("lv-bar");
    presets.forEach(function (p) { Kit.button(pb, p, function () { input.value = p; draw(); }); });
    Kit.button(pb, "Use all " + MAXK, function () { setK(MAXK); });
    [input, lower, unk].forEach(function (n) { n.addEventListener("input", draw); n.addEventListener("change", draw); });
    function draw() {
      var k = S.k; sl.set(k);
      var toks = tokenise(input.value, k, { lower: lower.checked, unknown: unk.value }), html = "";
      toks.slice(0, 300).forEach(function (t, i) {
        html += chip(t.t, t.kind === "tok" ? "c" + (i % 3) : t.kind === "unk" ? "unk" : "by", t.id);
      });
      if (toks.length > 300) html += "<span class=\"vnone\">and " + (toks.length - 300) + " more</span>";
      out.innerHTML = html || '<span class="vnone">Type something above.</span>';
      var nch = Array.from(input.value).length, nbytes = utf8(input.value).length;
      read.innerHTML = '<span>Characters: <b>' + nch + "</b></span><span>Tokens: <b>" + toks.length + "</b></span><span>Characters per token: <b>" + (toks.length ? (nch / toks.length).toFixed(2) : "0") + "</b></span><span>UTF-8 bytes: <b>" + nbytes + "</b></span>";
      var nUnk = toks.filter(function (t) { return t.kind === "unk"; }).length, nByte = toks.filter(function (t) { return t.kind === "byte"; }).length;
      st.textContent = nUnk ? nUnk + " characters are not in the alphabet, so they became <unk>. The model cannot tell them apart." : nByte ? nByte + " byte tokens replace characters that are not in the alphabet. No information is lost, but a character can cost several tokens." : k === 0 ? "No merges yet: one token per character, plus a space mark." : "Merges are applied in the order they were learned.";
    }
    listeners.push(draw);
    draw();
  }

  // ---------- 4. vocabulary: token to ID and back ----------
  function vocabFig() {
    var tbl = Kit.$("vc-table"), q = Kit.$("vc-q"), res = Kit.$("vc-res"), enc = Kit.$("vc-text"), pipe = Kit.$("vc-pipe"), cmp = Kit.$("vc-cmp"), sel = null;
    function drawTable() {
      var V = vocabulary(S.k), h = "<table class=\"vt\"><tbody>";
      function row(i) { h += '<tr data-i="' + i + '"' + (sel === i ? ' class="on"' : "") + "><td>" + i + "</td><td>" + esc(V.tokens[i]) + "</td></tr>"; }
      for (var i = 0; i < 4; i++) row(i);
      h += '<tr class="gap"><td>4 to 259</td><td>256 byte tokens, &lt;0x00&gt; to &lt;0xFF&gt;</td></tr>';
      for (i = 260; i < V.tokens.length; i++) row(i);
      tbl.innerHTML = h + "</tbody></table>";
      var on = tbl.querySelector("tr.on"); if (on && on.scrollIntoView) { var t = on.offsetTop - tbl.clientHeight / 2; tbl.scrollTop = Math.max(0, t); }
    }
    tbl.addEventListener("click", function (e) { var tr = e.target.closest("tr[data-i]"); if (tr) { q.value = tr.cells[1].textContent; sel = Number(tr.dataset.i); lookup(); } });
    function lookup() {
      var V = vocabulary(S.k), s = q.value.trim(), h;
      if (s === "") { res.textContent = "Type a token such as ▁shark, or an ID such as 300."; cmp.innerHTML = ""; return; }
      var tokenForm = s.replace(/^_/, SP);
      if (/^\d+$/.test(s) && Number(s) < V.tokens.length) {
        sel = Number(s); h = "ID <b>" + s + "</b> is the token <b>" + esc(V.tokens[sel]) + "</b>. Going from an ID to a token is a position in a list: one step.";
        cmp.innerHTML = "";
      } else if (V.id[tokenForm] !== undefined) {
        sel = V.id[tokenForm]; h = "<b>" + esc(tokenForm) + "</b> has ID <b>" + sel + "</b>.";
        cmp.innerHTML = '<span class="t c-navy"><b>' + (sel + 1) + '</b><i>comparisons if a list were scanned</i></span><span class="op">vs</span><span class="t c-teal"><b>1</b><i>step for a dictionary</i></span>';
      } else { sel = null; h = "<b>" + esc(s) + "</b> is not in the vocabulary. A tokenizer would split it into smaller pieces, or return <b>&lt;unk&gt;</b> (ID 1)."; cmp.innerHTML = ""; }
      res.innerHTML = h; drawTable();
    }
    q.addEventListener("input", lookup);
    function drawPipe() {
      var toks = tokenise(enc.value, S.k, { unknown: "bytes" }), ids = toks.map(function (t) { return t.id; });
      var back = decode(ids, S.k).replace(/^ /, ""), same = back === enc.value.toLowerCase().replace(/\s+/g, " ").trim();
      pipe.innerHTML = '<p><b>1. Tokens.</b> ' + toks.slice(0, 40).map(function (t, i) { return chip(t.t, t.kind === "tok" ? "c" + (i % 3) : "by"); }).join("") + (toks.length > 40 ? " &hellip;" : "") + "</p>" +
        '<p><b>2. IDs.</b> <code>[' + ids.slice(0, 40).join(", ") + (ids.length > 40 ? ", &hellip;" : "") + "]</code></p>" +
        "<p><b>3. Back to text.</b> <code>" + esc(back) + "</code> " + (same ? "(the same text as the input, in lower case)" : "(different from the input)") + "</p>";
    }
    enc.addEventListener("input", drawPipe);
    listeners.push(function () { lookup(); drawTable(); drawPipe(); });
    drawTable(); drawPipe(); lookup();
  }

  // ---------- 5. special tokens: padding a batch ----------
  function specialFig() {
    var svg = d3.select("#sp-svg"), st = Kit.$("sp-status"), read = Kit.$("sp-read"), bar = Kit.$("sp-bar");
    var tk = "gpt2", ends = true, pad = true, topic = 2;
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["GPT-2", "Qwen2.5"], 0, function (i) { tk = i ? "qwen" : "gpt2"; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Add start and end", "Do not add"], 0, function (i) { ends = i === 0; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Pad to the longest", "No padding"], 0, function (i) { pad = i === 0; draw(); });
    function draw() {
      var P = Kit.palette(), W = Kit.width(Kit.$("sp-svg")), rowH = 34, lab = W < 520 ? 74 : 96, H = rowH * 4 + 38, names = ["English", "Hindi", "Marathi", "Hinglish"];
      var msgs = [0, 4, 8, 12].map(function (b) { return DATA.messages[b + topic]; });
      var lens = msgs.map(function (m) { return m.tok[tk].text.length + (ends ? 2 : 0); }), mx = Math.max.apply(null, lens);
      var cw = (W - lab - 8) / mx, used = 0, slots = 0, padN = 0;
      svg.attr("viewBox", "0 0 " + W + " " + H).selectAll("*").remove();
      msgs.forEach(function (m, r) {
        var y = 8 + r * rowH, n = m.tok[tk].text.length, x = lab;
        svg.append("text").attr("x", 0).attr("y", y + 20).attr("font-size", 14).attr("fill", P.ink2).text(names[r]);
        function seg(count, fill, stroke, label) {
          if (count <= 0) return;
          svg.append("rect").attr("x", x).attr("y", y).attr("width", Math.max(1, count * cw - (cw > 5 ? 1 : 0))).attr("height", 24).attr("rx", 2).attr("fill", fill).attr("stroke", stroke || "none");
          if (label && count * cw > 38) svg.append("text").attr("x", x + count * cw / 2).attr("y", y + 16).attr("text-anchor", "middle").attr("font-size", 12).attr("fill", P.ink).text(label);
          x += count * cw;
        }
        if (ends) seg(1, P.coral, null, "");
        seg(n, P.teal, null, n + " tokens");
        if (ends) seg(1, P.coral, null, "");
        var mine = n + (ends ? 2 : 0), padding = pad ? mx - mine : 0;
        seg(padding, P.paper, P.muted, padding > 0 ? padding + " pad" : "");
        used += mine; slots += pad ? mx : mine; padN += padding;
      });
      svg.append("text").attr("x", 0).attr("y", H - 8).attr("font-size", 12.5).attr("fill", P.muted).text(ends ? "coral: start, end   teal: message   outline: padding" : "teal: message   outline: padding");
      read.innerHTML = '<span>Slots in the batch: <b>' + nf(slots) + "</b></span><span>Tokens in use: <b>" + nf(used) + "</b></span><span>Padding: <b>" + nf(padN) + " (" + (slots ? Math.round(100 * padN / slots) : 0) + "%)</b></span>";
      st.textContent = pad ? "The model must ignore the outlined slots, which it does with a mask (D4)." : "Without padding the rows have different lengths, so the batch cannot be one rectangle of numbers.";
    }
    draw();
    Kit.onTheme(draw);
    window.addEventListener("resize", Kit.debounce(draw));
  }

  // ---------- 6. bytes ----------
  function bytesFig() {
    var input = Kit.$("by-text"), tbl = Kit.$("by-table"), read = Kit.$("by-read"), bar = Kit.$("by-bar");
    var pre = [["English m01", DATA.messages[0].text], ["Hindi m05", DATA.messages[4].text], ["Marathi m09", DATA.messages[8].text], ["Hinglish m13", DATA.messages[12].text], ["शार्क", "शार्क"], ["Mumbai ❤", "Mumbai ❤"]];
    pre.forEach(function (p) { Kit.button(bar, p[0], function () { input.value = p[1]; draw(); }); });
    input.addEventListener("input", draw);
    function draw() {
      var chs = Array.from(input.value), total = 0, h = "<table class=\"bt\"><thead><tr><th>Character</th><th>Code point</th><th>UTF-8 bytes</th><th>Count</th></tr></thead><tbody>";
      chs.slice(0, 40).forEach(function (c) {
        var b = utf8(c); total += b.length;
        h += "<tr><td class=\"ch\">" + (c === " " ? "␣" : esc(c)) + "</td><td>U+" + c.codePointAt(0).toString(16).toUpperCase().padStart(4, "0") + "</td><td><code>" + b.map(hex2).join(" ") + "</code></td><td>" + b.length + "</td></tr>";
      });
      if (chs.length > 40) { chs.slice(40).forEach(function (c) { total += utf8(c).length; }); h += "<tr><td colspan=\"4\" class=\"vnone\">and " + (chs.length - 40) + " more characters</td></tr>"; }
      tbl.innerHTML = h + "</tbody></table>";
      read.innerHTML = '<span>Characters: <b>' + chs.length + "</b></span><span>UTF-8 bytes: <b>" + total + "</b></span><span>Bytes per character: <b>" + (chs.length ? (total / chs.length).toFixed(2) : "0") + "</b></span>";
    }
    draw();
  }

  // ---------- 7. four languages ----------
  function langFig() {
    var bar = Kit.$("lg-bar"), rows = Kit.$("lg-rows"), eq = Kit.$("lg-eqn"), st = Kit.$("lg-status");
    var topic = 1, tk = "gpt2", sel = 1, names = ["English", "Hindi", "Marathi", "Hinglish"];
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["Spadenose count", "Tag 4127 at Juhu", "Young hammerhead", "Water at Uran"], 1, function (i) { topic = i; draw(); });
    Kit.chips(Kit.html("div", { class: "row-chips" }, bar), ["GPT-2 (50,257 tokens)", "Qwen2.5 (151,643 tokens)"], 0, function (i) { tk = i ? "qwen" : "gpt2"; draw(); });
    rows.addEventListener("click", function (e) { var r = e.target.closest(".lrow"); if (r && Number(r.dataset.r) > 0) { sel = Number(r.dataset.r); draw(); } });
    function ratio(m) { var g = m.tok[tk].gloss_n; return g ? m.tok[tk].text.length / g : null; }
    function draw() {
      var msgs = [0, 4, 8, 12].map(function (b) { return DATA.messages[b + topic]; }), mx = Math.max.apply(null, msgs.map(function (m) { return m.tok[tk].text.length; })), h = "";
      msgs.forEach(function (m, r) {
        var t = m.tok[tk].text, g = m.tok[tk].gloss_n, ra = ratio(m), toks = "";
        t.forEach(function (p, i) { toks += chip(p[1] === 1 ? p[0] : showTok(p[0]), p[1] === 1 ? "by" : "c" + (i % 3)); });
        var flag = m.language !== "English" ? '<span class="flag">invented; not yet checked by a native speaker</span>' : '<span class="flag">invented</span>';
        h += '<div class="lrow' + (r === sel ? " sel" : "") + '" data-r="' + r + '"' + (r > 0 ? ' tabindex="0" role="button" aria-label="Show the working for ' + names[r] + '"' : "") + '><div class="lhead"><b>' + names[r] + "</b> <span class=\"mid\">" + m.id + "</span> " + flag + "</div>" +
          '<div class="lbar"><span class="fill" style="width:' + (100 * t.length / mx).toFixed(1) + '%"></span><b>' + t.length + " tokens</b></div>" +
          '<div class="lmeta">' + m.chars + " characters, " + m.bytes + " bytes, " + m.words + " words" + (g ? "; the English meaning takes " + g + " tokens, so the cost is <b>&times;" + ra.toFixed(1) + "</b>" : "") + "</div>" +
          '<div class="toks scroll">' + toks + "</div></div>";
      });
      rows.innerHTML = h;
      var m = msgs[sel], t = m.tok[tk].text.length, g = m.tok[tk].gloss_n;
      eq.innerHTML = '<span class="t c-coral"><b>' + t + '</b><i>tokens, ' + names[sel] + " " + m.id + '</i></span><span class="op">&divide;</span><span class="t c-navy"><b>' + g + '</b><i>tokens, its English meaning</i></span><span class="op">=</span><span class="t c-teal"><b>' + (t / g).toFixed(1) + '</b><i>cost ratio &rho;</i></span>' +
        '<span class="op">|</span><span class="t c-coral"><b>' + t + '</b><i>tokens</i></span><span class="op">&divide;</span><span class="t c-navy"><b>' + m.words + '</b><i>words</i></span><span class="op">=</span><span class="t c-teal"><b>' + (t / m.words).toFixed(1) + '</b><i>tokens per word</i></span>' +
        '<span class="op">|</span><span class="t c-sand"><b>' + Math.floor(1024 / t) + " vs " + Math.floor(1024 / g) + '</b><i>reports like this that fit in a 1,024-token window, ' + names[sel] + " vs English</i></span>";
      eq.setAttribute("aria-label", names[sel] + " report " + m.id + " takes " + t + " tokens against " + g + " for its English meaning, a cost ratio of " + (t / g).toFixed(1));
      st.textContent = "Click a Hindi, Marathi or Hinglish row to see its working. All sixteen messages are invented; the Hindi, Marathi and Hinglish texts have not yet been checked by a native speaker.";
      // bar chart of all twelve cost ratios
      var lang = ["Hindi", "Marathi", "Hinglish"], traces = [];
      lang.forEach(function (L, li) {
        var ms = DATA.messages.filter(function (x) { return x.language === L; });
        traces.push({ type: "bar", name: L, x: ms.map(function (x) { return x.id; }), y: ms.map(function (x) { return ratio(x); }),
          text: ms.map(function (x) { return ratio(x).toFixed(1); }), textposition: "outside", cliponaxis: false,
          hovertemplate: "%{x} " + L + ": %{y:.1f} times the English tokens<extra></extra>" });
      });
      Kit.plot("lg-plot", function (P) {
        traces.forEach(function (tr, i) { tr.marker = { color: [P.navy, P.teal, P.sand][i], opacity: DATA.messages.filter(function (x) { return x.language === tr.name; }).map(function (x, j) { return j === topic ? 1 : 0.45; }) }; });
        return traces;
      }, { barmode: "overlay", margin: { l: 48, r: 10, t: 34, b: 44 }, yaxis: { title: { text: "cost ratio ρ" }, range: [0, tk === "gpt2" ? 7.6 : 5] }, xaxis: { title: { text: "message", standoff: 6 }, tickangle: 0, tickfont: { size: 10 }, automargin: true }, showlegend: true, legend: { orientation: "h", y: 1.14, x: 0 } });
    }
    draw();
    Kit.onTheme(draw);
  }
})();
