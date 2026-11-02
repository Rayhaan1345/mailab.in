// D7 From language model to assistant. Figures: one question at three training stages (a real GPT-2 reply and scripted replies, labelled), which tokens count
// in the loss, learning from preferences (a Bradley-Terry reward and a policy on a leash), the chat template of an assistant model, in-context learning with
// GPT-2 small (exact next-token probabilities for every choice of prompt), the tool loop with a tool that runs on the made-up tables, skills loaded in stages,
// and how a fluent next token can be false. The pure logic is exported with module.exports for node tests.
(function () {
  "use strict";
  // ---------- pure logic ----------
  function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
  // One step on the Bradley-Terry loss -ln sigma(r_w - r_l): both scores move by eta (1 - sigma). Returns the probability and the loss before the step.
  function btStep(r, w, l, eta) {
    var s = sigmoid(r[w] - r[l]);
    r[w] += eta * (1 - s); r[l] -= eta * (1 - s);
    return { p: s, loss: -Math.log(s) };
  }
  function softmaxArr(z) { var m = Math.max.apply(null, z), e = z.map(function (x) { return Math.exp(x - m); }), s = e.reduce(function (a, b) { return a + b; }, 0); return e.map(function (x) { return x / s; }); }
  // The policy that maximises (expected reward - beta KL) is proportional to ref * exp(r / beta).
  function policy(ref, r, beta) { return softmaxArr(ref.map(function (p, i) { return Math.log(p) + r[i] / beta; })); }
  function kl(p, q) { var s = 0; for (var i = 0; i < p.length; i++) if (p[i] > 0) s += p[i] * Math.log(p[i] / q[i]); return s; }
  function expect(p, v) { var s = 0; for (var i = 0; i < p.length; i++) s += p[i] * v[i]; return s; }
  function rng(seed) { var a = seed >>> 0; return function () { a |= 0; a = (a + 0x6d2b79f5) | 0; var t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  // Tools that run on the made-up tables. Returns the text a tool returns, or an error text.
  var SPECIES = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"];
  function runTool(name, species, catchRecords, events) {
    var allowed = name === "dive_events" ? SPECIES.slice(0, 4) : SPECIES;
    if (allowed.indexOf(species) < 0) return 'error: unknown species "' + species + '". Choose from: ' + allowed.join(", ");
    if (name === "count_records") return String(catchRecords.filter(function (r) { return r.species === species; }).length);
    if (name === "mean_length") {
      var L = catchRecords.filter(function (r) { return r.species === species; }).map(function (r) { return r.length_cm; });
      return (L.reduce(function (a, b) { return a + b; }, 0) / L.length).toFixed(1) + " cm (" + L.length + " records)";
    }
    var ev = events.filter(function (e) { return e.kind === "shark" && e.species === species; });
    return ev.map(function (e) { return e.start_s + " s to " + e.end_s + " s"; }).join("; ");
  }
  if (typeof module !== "undefined" && module.exports) { module.exports = { sigmoid: sigmoid, btStep: btStep, policy: policy, kl: kl, expect: expect, runTool: runTool }; }
  if (typeof document === "undefined") return;

  // ---------- helpers ----------
  var ST = null, IC = null, HL = null, CATCH = null, DIVE = null;
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function nf(n) { return Number(n).toLocaleString("en-US"); }
  function box(val, cap, cls) { return '<span class="t ' + (cls || "") + '"><b>' + val + "</b><i>" + cap + "</i></span>"; }
  function op(s) { return '<span class="op">' + s + "</span>"; }
  function sig2(x) { var e = Math.pow(10, Math.floor(Math.log(x) / Math.LN10) - 1); return Math.round(x / e) * e; }
  function pc(x) { return x <= 0 ? "0%" : x < 0.0005 ? (100 * x).toPrecision(1) + "%" : x < 0.01 ? (100 * x).toFixed(2) + "%" : x < 0.1 ? (100 * x).toFixed(1) + "%" : (100 * x).toFixed(0) + "%"; }
  function rowChips(bar, labels, init, fn) { return Kit.chips(Kit.html("div", { class: "row-chips" }, bar), labels, init, fn); }
  function showTok(t) { t = t.replace(/\n/g, "↵"); return t.charAt(0) === " " ? "▁" + t.slice(1) : t; }
  function svgSize(svg, w, h) { svg.setAttribute("viewBox", "0 0 " + w + " " + h); svg.setAttribute("width", w); svg.setAttribute("height", h); svg.style.maxWidth = w + "px"; }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function svgText(parent, x, y, txt, attrs) { var t = Kit.el("text", Object.assign({ x: x, y: y }, attrs || {}), parent); t.textContent = txt; return t; }
  function checkbox(parent, label, checked, fn) {
    var lab = Kit.html("label", { class: "chk" }, parent), inp = Kit.html("input", { type: "checkbox" }, lab);
    inp.checked = !!checked; lab.appendChild(document.createTextNode(" " + label));
    inp.addEventListener("change", function () { fn(inp.checked); });
    return inp;
  }

  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/stages.json"), Kit.data("data/icl.json"), Kit.data("data/halluc.json"), Kit.data("../data/sharks/catch.json"), Kit.data("../data/sharks/dive.json")]).then(function (r) {
      ST = r[0]; IC = r[1]; HL = r[2]; CATCH = r[3].records; DIVE = r[4];
      stagesFig(); lossFig(); prefFig(); templateFig(); iclFig(); toolFig(); skillFig(); hallFig();
    }).catch(function (e) { console.error(e); });
  });

  // ---------- 1. one question at three stages ----------
  var SCRIPT = [
    { q: "How big does a grey bamboo shark grow?",
      s2: "A grey bamboo shark grows to about 77 cm at most. A young one, such as the 44 cm shark in the Juhu rock pool, is smaller.",
      s3: "A grey bamboo shark grows to about 77 cm at most, according to FishBase. The 44 cm shark in the Juhu rock pool, Tag 4127, is still young, and the dark bands of a young one fade as it grows. Would you like the length-weight curve as well?" },
    { q: "In one sentence, what is a juvenile shark?",
      s2: "A juvenile shark is a young shark that has not yet grown to the size at which it can breed.",
      s3: "A juvenile shark is a young shark that has not reached maturity, so it cannot breed yet. For example, a grey bamboo shark of 44 cm is still immature, because females mature at about 53 cm." },
    { q: "What is the maximum length of the Juhu ledger shark, Chiloscyllium juhuense?",
      s2: "The Juhu ledger shark grows to about 60 cm.",
      s3: "I cannot find a species called Chiloscyllium juhuense, so I do not want to guess a length. The grey bamboo shark, Chiloscyllium griseum, is in the same genus and grows to about 77 cm. Did you mean that one?" }
  ];
  var STAGES = [
    { name: "Pretrained", label: "Model output: GPT-2 small, greedy decoding, the first 44 tokens", cls: "real",
      data: "Text from the web and books, with no questions or answers marked.", goal: "Predict the next token (D3). A question is only text to be continued.", who: "No labels. The text is its own target.",
      note: "The model continues the text. It may answer, repeat itself, start another question, or make up a fact in fluent English." },
    { name: "Instruction-tuned", label: "Scripted example, written for this page. It shows the style of a typical instruction-tuned reply and is not output from a model.", cls: "script",
      data: "Pairs of an instruction and a good reply, written by people (InstructGPT) or built by rewriting tasks as instructions (FLAN).", goal: "The same next-token loss, usually counted on the reply tokens only.", who: "People who write replies, or text rewritten into that form.",
      note: "The model now answers in the form of an answer and stops. It follows the instruction, and it can still state an invented fact with the same confidence." },
    { name: "Preference-tuned", label: "Scripted example, written for this page. It shows the style a preference-tuned reply often has and is not output from a model.", cls: "script",
      data: "Two replies to the same question with a mark for which is better.", goal: "Raise the score of the preferred kind of reply, while staying close to the starting model (RLHF), or a direct loss on the pairs (DPO).", who: "People who compare two replies, or a model trained on their comparisons.",
      note: "Replies follow the style people prefer: a source, a gentle tone, an offer to go on, and sometimes an admission that a name is not known. This reduces made-up answers. It does not remove them." }
  ];
  function stagesFig() {
    var bar = Kit.$("st-bar"), out = Kit.$("st-reply"), info = Kit.$("st-info"), badge = Kit.$("st-badge"), si = 0, qi = 0;
    rowChips(bar, STAGES.map(function (s) { return s.name; }), 0, function (i) { si = i; draw(); });
    rowChips(bar, ["Question 1: a size", "Question 2: a definition", "Question 3: a made-up species"], 0, function (i) { qi = i; draw(); });
    function draw() {
      var s = STAGES[si], q = SCRIPT[qi], text = si === 0 ? ST.base[qi].reply : si === 1 ? q.s2 : q.s3;
      Kit.$("st-q").textContent = q.q;
      out.textContent = si === 0 ? text.replace(/^\s+/, "").replace(/\n+/g, " ↵ ") + " …" : text;
      badge.textContent = s.label; badge.className = "badge " + s.cls;
      info.innerHTML = '<div class="stgrid"><div><b>Data</b>' + esc(s.data) + "</div><div><b>Goal</b>" + esc(s.goal) + "</div><div><b>Who supplies the labels</b>" + esc(s.who) + '</div></div><p class="status">' + esc(s.note) + (qi === 2 && si === 1 ? " This reply is a made-up failure: the species does not exist and the number is invented." : "") + "</p>";
    }
    draw();
  }

  // ---------- 2. which tokens count in the loss ----------
  function lossFig() {
    var bar = Kit.$("lm-bar"), box_ = Kit.$("lm-toks"), mode = 0, L = ST.loss;
    rowChips(bar, ["Pretraining: all tokens count", "Instruction tuning: only the reply counts"], 0, function (i) { mode = i; draw(); });
    function draw() {
      var h = "", sum = 0, cnt = 0;
      L.tokens.forEach(function (t) {
        var counts = t.loss != null && (mode === 0 || t.part === "reply");
        if (counts) { sum += t.loss; cnt++; }
        var a = t.loss == null ? 0 : Math.min(1, t.loss / 10);
        h += '<span class="tk lt' + (counts ? "" : " off") + '" style="background:' + (counts ? "color-mix(in srgb, var(--coral) " + Math.round(8 + 70 * a) + "%, transparent)" : "none") + '" title="' + (t.loss == null ? "first token, no loss" : "loss " + t.loss + " nats") + '">' + esc(showTok(t.t)) + (t.loss == null ? "" : "<sub>" + t.loss.toFixed(1) + "</sub>") + "</span>";
      });
      box_.innerHTML = h;
      Kit.$("lm-read").innerHTML = "<span>Tokens that count: <b>" + cnt + "</b> of " + (L.tokens.length - 1) + "</span><span>Mean loss on them: <b>" + (sum / cnt).toFixed(2) + "</b> nats</span><span>Question tokens alone: <b>" + L.mean_prompt.toFixed(2) + "</b>, reply tokens alone: <b>" + L.mean_reply.toFixed(2) + "</b></span>";
      Kit.$("lm-eq").innerHTML = '<div class="eqn">' + box(cnt, "tokens that count", "c-sand") + op("&times;") + box((sum / cnt).toFixed(2), "mean loss", "c-coral") + op("=") + box(sum.toFixed(1), "total loss, nats", "c-teal") + "</div>";
    }
    draw();
  }

  // ---------- 3. learning from preferences ----------
  var ANS = [
    { id: "A", text: "About 77 cm at most (FishBase). The 44 cm shark from the Juhu rock pool, Tag 4127, is still young.", ref: 0.22, use: 0.9 },
    { id: "B", text: "They are big.", ref: 0.18, use: 0.1 },
    { id: "C", text: "77 cm. You could have looked that up.", ref: 0.10, use: 0.5 },
    { id: "D", text: "Sharks are fascinating animals found in many oceans, and the size of any one of them depends on many things, including food, water and age, so it is hard to give a single number, but in general bamboo sharks are small to medium sized sharks that can be quite long.", ref: 0.15, use: 0.3 },
    { id: "E", text: "About the size of a small car.", ref: 0.35, use: 0.0 }
  ];
  function prefFig() {
    var bar = Kit.$("pf-bar"), ctl = Kit.$("pf-controls"), r = [0, 0, 0, 0, 0], n = 0, last = null, ETA = 0.5, rnd = rng(4127), vr = rng(99), pairs = [], k, l, cur = 0;
    for (k = 0; k < 5; k++) for (l = k + 1; l < 5; l++) pairs.push([k, l]);
    var order = pairs.slice(); for (k = order.length - 1; k > 0; k--) { var m = Math.floor(rnd() * (k + 1)), t = order[k]; order[k] = order[m]; order[m] = t; }
    var words = ANS.map(function (a) { return a.text.split(/\s+/).length; }), ref = ANS.map(function (a) { return a.ref; }), use = ANS.map(function (a) { return a.use; });
    var beta = Kit.slider(ctl, { label: "Leash β (small = free to move)", min: 0.1, max: 3, step: 0.05, value: 1 }, draw);
    var left = Kit.$("pf-left"), right = Kit.$("pf-right");
    var b = Kit.html("div", { class: "row-chips" }, bar);
    function vote(w, lo) { last = btStep(r, w, lo, ETA); last.w = w; last.l = lo; n++; cur = (cur + 1) % order.length; draw(); }
    Kit.button(b, "Careful rater votes 10 times", function () { for (var i = 0; i < 10; i++) { var p = pairs[Math.floor(vr() * pairs.length)]; vote(use[p[0]] >= use[p[1]] ? p[0] : p[1], use[p[0]] >= use[p[1]] ? p[1] : p[0]); } });
    Kit.button(b, "Rater who likes long answers votes 10 times", function () { for (var i = 0; i < 10; i++) { var p = pairs[Math.floor(vr() * pairs.length)]; vote(words[p[0]] >= words[p[1]] ? p[0] : p[1], words[p[0]] >= words[p[1]] ? p[1] : p[0]); } });
    Kit.button(b, "Reset", function () { r = [0, 0, 0, 0, 0]; n = 0; last = null; vr = rng(99); draw(); });
    left.addEventListener("click", function () { var p = order[cur]; vote(p[0], p[1]); });
    right.addEventListener("click", function () { var p = order[cur]; vote(p[1], p[0]); });
    function draw() {
      var P = Kit.palette(), p = order[cur], pol = policy(ref, r, beta.get()), ids = ANS.map(function (a) { return a.id; });
      Kit.$("pf-a").textContent = ANS[p[0]].id + ". " + ANS[p[0]].text; Kit.$("pf-b").textContent = ANS[p[1]].id + ". " + ANS[p[1]].text;
      Kit.plot("pf-reward", function (P2) { return [{ type: "bar", x: ids, y: r, marker: { color: r.map(function (v) { return v >= 0 ? P2.teal : P2.coral; }) }, hovertemplate: "answer %{x}: reward %{y:.2f}<extra></extra>" }]; },
        { xaxis: { title: { text: "answer" }, automargin: true }, yaxis: { title: { text: "reward score r" }, zeroline: true, automargin: true }, margin: { l: 54, r: 8, t: 8, b: 40 } });
      Kit.plot("pf-policy", function (P2) { return [
        { type: "bar", name: "before", x: ids, y: ref, marker: { color: P2.sand }, hovertemplate: "answer %{x}: before %{y:.1%}<extra></extra>" },
        { type: "bar", name: "after", x: ids, y: pol, marker: { color: P2.teal }, hovertemplate: "answer %{x}: after %{y:.1%}<extra></extra>" }]; },
        { barmode: "group", xaxis: { title: { text: "answer" }, automargin: true }, yaxis: { title: { text: "probability" }, tickformat: ".0%", automargin: true }, margin: { l: 54, r: 8, t: 8, b: 40 }, showlegend: true, legend: { orientation: "h", y: 1.12 } });
      var eu0 = expect(ref, use), eu1 = expect(pol, use);
      Kit.$("pf-read").innerHTML = "<span>Votes so far: <b>" + n + "</b></span><span>Usefulness (page author's made-up score, 0 to 1) before: <b>" + eu0.toFixed(2) + "</b>, after: <b>" + eu1.toFixed(2) + "</b></span><span>Distance from the start, KL: <b>" + Math.max(0, kl(pol, ref)).toFixed(2) + "</b> nats</span><span>Answer E (a made-up size) before: <b>" + pc(ref[4]) + "</b>, after: <b>" + pc(pol[4]) + "</b></span>";
      Kit.$("pf-eq").innerHTML = last ? '<div class="eqn">' + box(ANS[last.w].id + " &gt; " + ANS[last.l].id, "last vote", "c-sand") + op("&rarr;") + box(last.p.toFixed(2), "σ(r_w − r_l) before the step", "c-navy") + op("&rarr;") + box(last.loss.toFixed(2), "loss −ln σ", "c-coral") + op("&rarr;") + box("+" + (ETA * (1 - last.p)).toFixed(2), "r_w up, r_l down by this much", "c-teal") + "</div>" : '<p class="status">Vote for the better answer, or let a simulated rater vote. The reward scores move after a vote.</p>';
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 4. the chat template ----------
  function templateFig() {
    var bar = Kit.$("ct-bar"), pre = Kit.$("ct-text"), withSys = 0;
    rowChips(bar, ["No system message", "With a system message"], 0, function (i) { withSys = i; draw(); });
    function draw() {
      var c = withSys ? ST.chat.system : ST.chat.no_system, parts = c.text.split(/(<\|im_start\|>|<\|im_end\|>)/), h = "";
      parts.forEach(function (p) { if (!p) return; if (p === "<|im_start|>" || p === "<|im_end|>") h += '<span class="sp">' + esc(p) + "</span>"; else h += esc(p).replace(/\n/g, "↵\n"); });
      pre.innerHTML = h;
      var id = ST.chat.special_ids;
      Kit.$("ct-read").innerHTML = "<span>Tokens the model reads: <b>" + c.tokens + "</b></span><span>&lt;|im_start|&gt; is token <b>" + id["<|im_start|>"] + "</b>, &lt;|im_end|&gt; is token <b>" + id["<|im_end|>"] + "</b> (Qwen2.5 tokenizer)</span><span>The model writes after the last line and stops by writing &lt;|im_end|&gt;</span>";
    }
    draw();
  }

  // ---------- 5. in-context learning ----------
  function iclFig() {
    var bar = Kit.$("ic-bar"), body = Kit.$("ic-body"), mode = 0, qF = 2, qR = 0, flip = false, onF = [false, false, true, false, false], onR = [true, true, true, true, true, true];
    rowChips(bar, ["Facts in the prompt", "A rule from examples"], 0, function (i) { mode = i; build(); });
    function build() {
      clear(body);
      var pick = Kit.html("div", { class: "bar" }, body), qs = Kit.html("div", { class: "bar" }, body), out = Kit.html("div", { class: "ic-out" }, body);
      if (mode === 0) {
        IC.facts.pieces.forEach(function (p, i) { checkbox(pick, p, onF[i], function (v) { onF[i] = v; draw(); }); });
        rowChips(qs, IC.facts.questions.map(function (q) { return q.split("\n")[0].replace("Q: ", ""); }), qF, function (i) { qF = i; draw(); });
      } else {
        IC.rule.examples.forEach(function (e, i) { checkbox(pick, e[0] + " → " + e[1], onR[i], function (v) { onR[i] = v; draw(); }); });
        checkbox(pick, "Flip all the labels", flip, function (v) { flip = v; draw(); });
        rowChips(qs, IC.rule.queries.map(function (q) { return q[0].length > 36 ? q[0].slice(0, 34) + "…" : q[0]; }), qR, function (i) { qR = i; draw(); });
      }
      body._out = out; draw();
    }
    function mask(a) { var m = 0; a.forEach(function (v, i) { if (v) m |= 1 << i; }); return m; }
    function draw() {
      var out = body._out, h = "";
      if (mode === 0) {
        var m = mask(onF), row = IC.facts.results[String(m)][qF], row0 = IC.facts.results["0"][qF], inc = IC.facts.pieces.filter(function (_, i) { return onF[i]; });
        var prompt = (inc.length ? inc.join("\n") + "\n" : "") + IC.facts.questions[qF];
        h += '<pre class="prompt">' + esc(prompt) + '<span class="cursor">▌</span></pre>';
        h += '<p class="side-h">GPT-2 small: the five most likely next tokens</p>' + bars(row.top);
        h += '<div class="eqn">' + box(pc(row0.p), "answer “" + esc(row.first.trim()) + "” with no fact", "c-sand") + op("&rarr;") + box(pc(row.p), "with the facts above", "c-teal") + op("=") + box(row0.p > 0 ? "about ×" + nf(sig2(row.p / row0.p)) : "—", "change", "c-coral") + "</div>";
        h += '<p class="status">' + (inc.length === 0 ? "With no fact in the prompt the model has no way to know the answer, and the probability of the right first token is tiny." : onF[qF] ? "The fact for this question is in the prompt. The model finds it and copies from it. No weight changed." : "The fact for this question is not in the prompt, so the other facts do not help.") + (inc.length > 1 && onF[qF] ? " With more facts in the prompt, GPT-2 small finds the right one less often: it is a small model." : "") + "</p>";
      } else {
        var mR = mask(onR), resT = IC.rule.results[flip ? "flip" : "true"][String(mR)], q = IC.rule.queries[qR], r0 = resT[qR], inc2 = IC.rule.examples.filter(function (_, i) { return onR[i]; });
        var prompt2 = inc2.map(function (e) { return e[0] + " Shark seen: " + (flip ? (e[1] === "yes" ? "no" : "yes") : e[1]) + "\n"; }).join("") + q[0] + " Shark seen:";
        h += '<pre class="prompt">' + esc(prompt2) + '<span class="cursor">▌</span></pre>';
        var other = Math.max(0, 1 - r0[0] - r0[1]);
        h += '<div class="eqn">' + box(pc(r0[0]), "yes", q[1] === "yes" ? "c-teal" : "c-navy") + box(pc(r0[1]), "no", q[1] === "no" ? "c-teal" : "c-navy") + box(pc(other), "any other token", "c-sand") + "</div>";
        var right = 0, mass = 0; resT.forEach(function (x, i) { if ((x[0] > x[1]) === (IC.rule.queries[i][1] === "yes")) right++; mass += x[0] + x[1]; });
        h += '<p class="status">The true answer for this report is <b>' + q[1] + "</b>. Over the six test reports with these examples, GPT-2 small gives more weight to the right label for <b>" + right + "</b> of 6, and puts <b>" + (mass / 6 >= 0.1 ? (100 * mass / 6).toFixed(1) + "%" : pc(mass / 6)) + "</b> of its probability on yes or no" + (inc2.length === 0 ? " (with no examples it does not even know that yes or no is the form of the answer)." : ".") + "</p>";
      }
      out.innerHTML = h;
    }
    function bars(top) { return top.map(function (t) { return '<div class="rs-row"><span class="rs-tok">' + esc(showTok(t[0])) + '</span><span class="rs-track"><i style="width:' + Math.max(1, Math.round(100 * t[1])) + '%"></i></span><b>' + pc(t[1]) + "</b></div>"; }).join(""); }
    build();
  }

  // ---------- 6. the tool loop ----------
  function toolFig() {
    var bar = Kit.$("tl-bar"), ctxEl = Kit.$("tl-ctx"), who = Kit.$("tl-who"), qi = 0, typo = false, step = 0, T = ST.chat.trips;
    rowChips(bar, ["How many Ganges shark records?", "Mean length of bamboo sharks?", "When was a hammerhead seen?"], 0, function (i) { qi = i; step = 0; draw(); });
    rowChips(bar, ["The model writes the call correctly", "The model misspells the species"], 0, function (i) { typo = i === 1; step = 0; draw(); });
    var b = Kit.html("div", { class: "row-chips" }, bar);
    Kit.button(b, "Back", function () { step = Math.max(0, step - 1); draw(); });
    Kit.button(b, "Next step", function () { step = Math.min(steps().length - 1, step + 1); draw(); });
    Kit.button(b, "Reset", function () { step = 0; draw(); });
    function steps() {
      return typo ? [["ask", "ask"], ["call_bad", "call_bad"], ["run_bad", "call_bad"], ["error", "error"], ["call_good", "call_good"], ["run", "call_good"], ["result", "result"], ["answer", "answer"]]
        : [["ask", "ask"], ["call", "call"], ["run", "call"], ["result", "result"], ["answer", "answer"]];
    }
    var WHO = {
      ask: ["You and the application", "You send a question. The application (the program around the model) builds the text the model reads: the system message, the three tool definitions and your question."],
      call: ["Model (scripted)", "The model writes a tool call as text, in the format the template defines, and stops. It does not run anything."],
      call_bad: ["Model (scripted)", "The model writes a tool call, but spells the species wrong. This step is scripted to show a failure."],
      call_good: ["Model (scripted)", "Told what went wrong, the model writes the call again with the right name."],
      run: ["Application", "The application reads the call, finds the tool and runs it. This is ordinary code, and in this page it runs on the made-up tables."],
      run_bad: ["Application", "The application runs the tool. The species is not in its list, so the tool returns an error text."],
      error: ["Application", "The error text is added to the text as a tool response, and the model is asked to continue."],
      result: ["Application", "The result is added to the text as a tool response, and the model is asked to continue."],
      answer: ["Model (scripted)", "The model reads the result in its context and writes the answer. The tool result is the source of the number."]
    };
    function draw() {
      var t = T[qi], seq = steps(), cur = seq[step], key = cur[1], set = typo ? t.retry : t.normal, c = set[key], prev = step > 0 ? set[seq[step - 1][1]] : null, kind = cur[0];
      var text = c.text, hd = text.indexOf("<|im_start|>user"), oldLen = (prev && text.indexOf(prev.text) === 0) ? prev.text.length : hd;
      ctxEl.innerHTML = '<details class="sysd"><summary>System message and three tool definitions: ' + nf(ST.chat.tool_head.tokens) + ' tokens (open to read)</summary><pre>' + esc(text.slice(0, hd)) + "</pre></details>" +
        '<span class="old">' + esc(text.slice(hd, oldLen)) + "</span><mark>" + esc(text.slice(oldLen)) + "</mark>" + (oldLen === text.length && step > 0 ? '<span class="old">(no new text at this step)</span>' : "");
      var w = WHO[kind], live = "";
      if (kind === "run" || kind === "run_bad") {
        var arg = kind === "run" ? t.species : t.typo, res = runTool(t.name, arg, CATCH, DIVE.events);
        live = '<pre class="run">' + esc(t.name + '(species="' + arg + '")') + "\n" + esc("→ " + res) + "</pre>";
        if (kind === "run" && res !== t.result) console.error("tool result differs from the shipped one", res, t.result);
        if (kind === "run_bad" && res !== t.error) console.error("tool error differs from the shipped one", res, t.error);
      }
      who.innerHTML = '<p class="who"><b>' + w[0] + ".</b> " + esc(w[1]) + "</p>" + live;
      Kit.$("tl-read").innerHTML = "<span>Step <b>" + (step + 1) + "</b> of " + seq.length + "</span><span>Text the model reads at this step: <b>" + nf(c.tokens) + "</b> tokens (Qwen2.5 tokenizer)</span><span>Of these, the three tool definitions take about <b>" + nf(ST.chat.tool_head.tokens - ST.chat.tool_head_no_tools.tokens) + "</b></span>";
      ctxEl.setAttribute("aria-label", "The text of the conversation at step " + (step + 1) + ", with the new part highlighted.");
    }
    draw();
  }

  // ---------- 7. skills load in stages ----------
  function skillFig() {
    var bar = Kit.$("sk-bar"), svg = Kit.$("sk-svg"), mi = 0;
    var MSG = [
      { t: "I have to solve 2x² − 5x − 3 = 0 for my homework. Can you help?", skill: "socratic-tutor", why: "The description of socratic-tutor matches help with homework, so Claude reads its SKILL.md. The question ladder in references/ opens only if the instructions point to it." },
      { t: "Quiz me on the spadenose shark.", skill: "quiz-me", why: "The description of quiz-me matches a request to be quizzed, so Claude reads that SKILL.md instead." },
      { t: "What is a spadenose shark?", skill: null, why: "A definition is a lookup. The tutor skill excludes definitions, so no skill body is read and the reply is plain." }
    ];
    rowChips(bar, MSG.map(function (m, i) { return "Message " + (i + 1); }), 0, function (i) { mi = i; draw(); });
    function draw() {
      var P = Kit.palette(), W = Math.max(300, Kit.width(svg.parentNode) - 8), H = 186, m = MSG[mi], scale = (W - 20) / 5600;
      svgSize(svg, W, H); clear(svg);
      var rows = [["always in the text", "two names and descriptions", 200, P.sand, true], ["when a skill matches", m.skill ? m.skill + " SKILL.md, up to 5,000 tokens" : "no skill body is read", m.skill ? 5000 : 0, P.coral, !!m.skill], ["only if the instructions point to it", "extra files: 0 tokens until read", 0, P.teal, false]];
      rows.forEach(function (r, i) {
        var y = 14 + i * 58;
        svgText(svg, 10, y, r[0] + ":", { "font-size": 12.5, fill: P.ink2 });
        svgText(svg, 10, y + 15, r[1], { "font-size": 12.5, fill: P.ink2 });
        Kit.el("rect", { x: 10, y: y + 22, width: Math.max(3, r[2] * scale), height: 14, rx: 3, fill: r[3], "fill-opacity": r[4] ? 0.8 : 0.2, stroke: r[4] ? "none" : P.muted, "stroke-dasharray": r[4] ? "0" : "3 2" }, svg);
      });
      Kit.$("sk-text").innerHTML = '<b>' + esc(m.t) + "</b> " + esc(m.why);
      svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "Bars of the text added by a skill: " + (m.skill ? m.skill + " is read in full" : "no skill is read") + ".");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }

  // ---------- 8. how a fluent token can be false ----------
  function hallFig() {
    var bar = Kit.$("hl-bar"), ctl = Kit.$("hl-controls"), ci = 1, rnd = rng(4127), draws = null;
    rowChips(bar, ["A fact in many texts", "A true fact few texts state", "A made-up species: a length", "A made-up species: a name"], 1, function (i) { ci = i; draws = null; draw(); });
    var tem = Kit.slider(ctl, { label: "Temperature T", min: 0.2, max: 2, step: 0.05, value: 1 }, function () { draws = null; draw(); });
    var b = Kit.html("div", { class: "row-chips" }, bar);
    Kit.button(b, "Draw 200 continuations", function () { draws = sample(200); draw(); });
    function dist(c, T) {
      var rows = c.tokens.map(function (t) { return { t: t[0], w: Math.exp(t[1] / T), cls: t[2] }; }), tail = 0, cn = c.tail.counts, i;
      for (i = 0; i < cn.length; i++) tail += cn[i] * Math.exp(c.tail.m[i] / T) * (1 + c.tail.v[i] / (2 * T * T));
      var Z = tail; rows.forEach(function (r) { Z += r.w; });
      rows.forEach(function (r) { r.p = r.w / Z; });
      var H = 0; rows.forEach(function (r) { if (r.p > 0) H -= r.p * Math.log(r.p); });
      for (i = 0; i < cn.length; i++) { var pz = Math.exp(c.tail.m[i] / T) * (1 + c.tail.v[i] / (2 * T * T)) / Z; if (pz > 0 && cn[i] > 0) H -= cn[i] * pz * Math.log(pz); }
      return { rows: rows, tail: tail / Z, H: H };
    }
    function sample(n) {
      var d = dist(HL.contexts[ci], tem.get()), cnt = { true: 0, like: 0, other: 0 }, k;
      for (k = 0; k < n; k++) {
        var u = rnd(), acc = 0, cls = "other";
        for (var i = 0; i < d.rows.length; i++) { acc += d.rows[i].p; if (u < acc) { cls = d.rows[i].cls; break; } }
        cnt[cls]++;
      }
      return cnt;
    }
    function draw() {
      var c = HL.contexts[ci], T = tem.get(), d = dist(c, T), P = Kit.palette(), show = d.rows.slice(0, 10);
      var pTrue = 0, pLike = 0; d.rows.forEach(function (r) { if (r.cls === "true") pTrue += r.p; else if (r.cls === "like") pLike += r.p; });
      var y = show.map(function (r) { return showTok(r.t); }).reverse(), x = show.map(function (r) { return r.p; }).reverse(), cols = show.map(function (r) { return r.cls === "true" ? P.teal : r.cls === "like" ? P.coral : P.sand; }).reverse();
      Kit.plot("hl-plot", [{ type: "bar", orientation: "h", y: y.map(function (s, i) { return s + " "; }), x: x, marker: { color: cols }, text: x.map(function (v) { return pc(v); }), textposition: "outside", cliponaxis: false, hovertemplate: "%{y}: %{x:.1%}<extra></extra>" }],
        { xaxis: { title: { text: "probability of the next token" }, tickformat: ".0%", automargin: true, range: [0, Math.min(1, Math.max.apply(null, x) * 1.25 + 0.03)] }, yaxis: { automargin: true, tickfont: { family: "ui-monospace, Menlo, monospace", size: 13 } }, margin: { l: 90, r: 30, t: 8, b: 44 } });
      var top = d.rows[0], kind = c.truth ? "A true answer exists: <b>" + showTok(c.truth) + "</b>." : "No true answer exists, because the species is made up.";
      Kit.$("hl-read").innerHTML = "<span>" + esc(c.kind) + ". " + kind + "</span>";
      Kit.$("hl-prompt").textContent = c.text;
      Kit.$("hl-eq").innerHTML = '<div class="eqn">' + (c.truth ? box(pc(pTrue), "true token", "c-teal") : "") + box(pc(pLike), c.truth ? "answer-like but not true" : "answer-like, all unsupported", "c-coral") + box(pc(Math.max(0, 1 - pTrue - pLike)), "other tokens", "c-sand") + box(d.H.toFixed(2) + " nats", "entropy, about " + (Math.exp(d.H)).toFixed(0) + " equally likely choices", "c-navy") + "</div>";
      var greedy = top.cls === "true" ? "true" : top.cls === "like" ? "answer-like but " + (c.truth ? "false" : "unsupported") : "not an answer, a word that starts a longer sentence";
      var s = "Greedy choice: <b>" + esc(showTok(top.t)) + "</b> (" + pc(top.p) + "), " + greedy + ".";
      if (draws) s += " In 200 draws: <b>" + draws.true + "</b> true, <b>" + draws.like + "</b> answer-like but " + (c.truth ? "false" : "unsupported") + ", <b>" + draws.other + "</b> other.";
      Kit.$("hl-status").innerHTML = s;
      Kit.$("hl-plot").setAttribute("aria-label", "Bars of the ten most likely next tokens for the context: " + c.text + ". The true token, if there is one, is teal, other answer-like tokens are coral.");
    }
    draw(); Kit.onTheme(draw); window.addEventListener("resize", Kit.debounce(draw, 200));
  }
})();
