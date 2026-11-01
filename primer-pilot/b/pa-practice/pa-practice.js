// Practice set A.P: timer, saved answers and scoring for the questions on Part A, learning from data.
// Answers are checked in the browser. KEY holds a hash of each right answer, so the page itself never shows an answer.
// To change an answer, recompute its hash with hash(id + "|" + normalised answer) and replace the entry in KEY.
(function () {
  "use strict";

  var KEY = {
    "q01": "1d5lo3wu7ej",
    "q02": "22ht0j3omw7",
    "q03": "ubp98rs1w5",
    "q04": "28h3s7sxj5w",
    "q05": "9hs7xjowg9",
    "q06": "4nt39o30uf",
    "q07": "1so32m0j64w",
    "q08": "15s044ne9h8",
    "q09": "9rcs7egpak",
    "q10": "pnz7xwjovl",
    "q11": "twbfrqlgwa",
    "q12": "1ct3mbxzud7",
    "q13": "2bjp0qt36b",
    "q14": "v5owncnpqu",
    "q15": "109detq4pox",
    "q16": "1p2d4v8nd2v",
    "q17": "1y946b8yhe6",
    "q18": "23vqdtugstq",
    "q19": "fj1hq3z7x7",
    "q20": "1qjqh093ap",
    "q21": "1qippqmtja5",
    "q22": "10e00thmcwh",
    "q23": "15b9dya2vpg",
    "q24": "1zevn2seary",
    "q25": "11m2xo6gujc"
  };
  var STORE = "primer-pa-practice-v1";
  var AREAS = [
    { id: "models", name: "Models and baselines", links: [["A1 What a model is", "../a1-what-is-a-model/"]] },
    { id: "linear", name: "Linear regression", links: [["A2 Linear regression", "../a2-linear-regression/"]] },
    { id: "logistic", name: "Logistic regression", links: [["A3 Logistic regression and softmax", "../a3-logistic-regression/"]] },
    { id: "grading", name: "Grading a model", links: [["A4 Grading a model fairly", "../a4-grading-models/"]] },
    { id: "trees", name: "Decision trees", links: [["A5 Decision trees", "../a5-decision-trees/"]] },
    { id: "boosting", name: "Boosting", links: [["A6 Ensembles and boosting", "../a6-boosting/"]] }
  ];
  var MINUTES = [45, 60, 75];
  var DEFAULT_MIN = 60;

  // A small string hash (cyrb53, public domain). It returns a base-36 string.
  function hash(str) {
    var h1 = 0xdeadbeef, h2 = 0x41c6ce57;
    for (var i = 0; i < str.length; i++) {
      var ch = str.charCodeAt(i);
      h1 = Math.imul(h1 ^ ch, 2654435761);
      h2 = Math.imul(h2 ^ ch, 1597334677);
    }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(36);
  }

  // Read a typed number: a minus sign, a decimal point, a fraction such as 1/4, or a trailing percent sign.
  function parseNum(raw) {
    var s = String(raw).trim().replace(/\u2212/g, "-").replace(/\s+/g, ""), pct = false;
    if (/^-?\d{1,3}(,\d{3})+(\.\d+)?%?$/.test(s)) s = s.replace(/,/g, "");
    if (/%$/.test(s)) { pct = true; s = s.slice(0, -1); }
    var m = s.match(/^(-?\d*\.?\d+)\/(-?\d*\.?\d+)$/), x;
    if (m) x = parseFloat(m[1]) / parseFloat(m[2]);
    else if (/^-?(\d+\.?\d*|\.\d+)$/.test(s)) x = parseFloat(s);
    else return null;
    return isFinite(x) ? { x: x, pct: pct } : null;
  }

  // The form of an answer that the key was made from. Returns null when the answer is blank or not a number.
  function normal(q, raw) {
    if (raw == null || String(raw).trim() === "") return null;
    if (q.kind === "mc") return String(raw);
    var p = parseNum(raw);
    if (!p) return null;
    var x = p.x;
    if (q.pct && !p.pct && x > 0 && x <= 1) x = x * 100;
    if (q.dp === "int") return Math.abs(x - Math.round(x)) > 1e-9 ? null : String(Math.round(x));
    var s = x.toFixed(Number(q.dp));
    return /^-0(\.0+)?$/.test(s) ? s.slice(1) : s;
  }

  document.addEventListener("DOMContentLoaded", function () {
    var $ = Kit.$;
    var qs = Array.prototype.map.call(document.querySelectorAll(".pq"), function (el) {
      return {
        el: el, id: el.getAttribute("data-q"), kind: el.getAttribute("data-kind"), dp: el.getAttribute("data-dp"),
        pct: el.hasAttribute("data-pct"), area: el.closest("section").id, n: Number(el.id.slice(1)),
        inputs: Array.prototype.slice.call(el.querySelectorAll("input")),
        st: el.querySelector(".pq-st"), rev: el.querySelector(".revise")
      };
    });
    var state = { a: {}, min: DEFAULT_MIN, el: 0, used: false, done: false, first: null };
    var canStore = true;

    // ---------- saved state ----------
    function load() {
      try {
        var s = JSON.parse(localStorage.getItem(STORE));
        if (s && s.v === 1 && s.d) {
          state.a = s.d.a || {}; state.min = MINUTES.indexOf(s.d.min) >= 0 ? s.d.min : DEFAULT_MIN; state.el = Number(s.d.el) || 0;
          state.used = !!s.d.used; state.done = !!s.d.done; state.first = s.d.first == null ? null : Number(s.d.first);
        }
      } catch (e) {}
    }
    function save() {
      try { localStorage.setItem(STORE, JSON.stringify({ v: 1, d: state })); } catch (e) { canStore = false; noteStore(); }
    }
    function checkStore() {
      try { localStorage.setItem(STORE + "-probe", "1"); localStorage.removeItem(STORE + "-probe"); } catch (e) { canStore = false; }
      noteStore();
    }
    function noteStore() {
      if (!canStore) Kit.setText("store-note", "This browser blocks saving, so keep this page open until you have scored your answers.");
    }

    // ---------- answers ----------
    function answerOf(q) {
      if (q.kind === "mc") {
        var on = q.inputs.filter(function (i) { return i.checked; })[0];
        return on ? on.value : "";
      }
      return q.inputs[0].value;
    }
    function setAnswer(q, v) {
      if (q.kind === "mc") q.inputs.forEach(function (i) { i.checked = String(i.value) === String(v); });
      else q.inputs[0].value = v == null ? "" : v;
      paintChoice(q);
    }
    function paintChoice(q) {
      if (q.kind !== "mc") return;
      q.inputs.forEach(function (i) { i.parentNode.classList.toggle("on", i.checked); });
    }
    function answered(q) { return String(answerOf(q)).trim() !== ""; }
    function unreadable(q) { return q.kind === "num" && answered(q) && !parseNum(answerOf(q)) || !answered(q); }
    function isRight(q) {
      var v = normal(q, answerOf(q));
      return v !== null && hash(q.id + "|" + v) === KEY[q.id];
    }
    function progress() {
      var n = qs.filter(answered).length;
      Kit.setText("pb-n", String(n));
      return n;
    }

    // ---------- timer ----------
    var disp = $("pb-time"), go = $("pb-go"), running = false, t0 = 0, base = 0, ticks = 0;
    function elapsed() { return running ? base + (Date.now() - t0) : base; }
    function clock(ms) {
      var s = Math.floor(ms / 1000);
      var m = Math.floor(s / 60);
      return (m < 10 ? "0" : "") + m + ":" + (s % 60 < 10 ? "0" : "") + (s % 60);
    }
    function paint() {
      var ms = elapsed(), left = state.min * 60 * 1000 - ms, over = left < 0;
      disp.textContent = (over ? "+" : "") + clock(Math.abs(left) + (over ? 0 : 999));
      disp.classList.toggle("over", over);
      go.textContent = state.done ? "Timer stopped" : running ? "Pause" : (ms > 0 ? "Resume" : "Start timer");
      go.disabled = state.done;
    }
    function start() { if (!running) { running = true; t0 = Date.now(); state.used = true; paint(); } }
    function pause() { if (running) { base = elapsed(); running = false; state.el = base; paint(); save(); } }
    function resetTimer() { running = false; base = 0; state.el = 0; state.used = false; paint(); save(); }
    go.addEventListener("click", function () { if (running) pause(); else start(); });
    Kit.$("tm-reset").addEventListener("click", resetTimer);
    setInterval(function () {
      if (!running) return;
      paint();
      if (++ticks % 8 === 0) { state.el = elapsed(); save(); }
    }, 250);

    // ---------- scoring ----------
    var armed = false, armTimer = 0;
    function lockAll(on) { qs.forEach(function (q) { q.inputs.forEach(function (i) { i.disabled = on; }); }); }
    function markQuestions() {
      var right = 0;
      qs.forEach(function (q) {
        var ok = isRight(q);
        right += ok ? 1 : 0;
        q.el.classList.toggle("right", ok);
        q.el.classList.toggle("missed", !ok);
        q.st.hidden = false; q.st.textContent = ok ? "Right" : "Missed";
        q.rev.hidden = ok;
        q.inputs.forEach(function (i) { i.disabled = true; });
      });
      return right;
    }
    function areaBadges() {
      AREAS.forEach(function (a) {
        var h = $(a.id + "-h"), old = h.querySelector(".area-badge");
        if (old) old.remove();
        var mine = qs.filter(function (q) { return q.area === a.id; });
        Kit.html("span", { class: "area-badge" }, h, mine.filter(function (q) { return q.el.classList.contains("right"); }).length + " of " + mine.length);
      });
    }
    function showResults(fresh) {
      var right = markQuestions();
      if (state.first == null) state.first = right;
      areaBadges();
      var box = $("results"), h = [];
      var pct = right / qs.length;
      h.push('<h3 id="results-h">Your score</h3>');
      h.push('<p class="big"><b>' + right + '</b> of ' + qs.length + '</p>');
      var sub = [];
      if (state.first !== right) sub.push("First attempt " + state.first + " of " + qs.length + ".");
      if (state.used && state.el > 0) sub.push("Time used " + clock(state.el) + " of " + clock(state.min * 60000) + ".");
      sub.push(pct >= 0.8 ? "Part A is in good order. Revise only the sections below."
        : pct >= 0.5 ? "Revise the sections below, then try the missed questions again."
        : "Work through the chapters for the areas with the lowest scores, then attempt the whole set again in a few days.");
      h.push('<p class="sub">' + sub.join(" ") + '</p>');
      h.push('<h4>By area</h4>');
      AREAS.forEach(function (a) {
        var mine = qs.filter(function (q) { return q.area === a.id; });
        var got = mine.filter(function (q) { return q.el.classList.contains("right"); }).length;
        var seen = {}, links = [];
        mine.filter(function (q) { return q.el.classList.contains("missed"); }).forEach(function (q) {
          Array.prototype.forEach.call(q.rev.querySelectorAll("a"), function (a2) {
            var href = a2.getAttribute("href");
            if (!seen[href]) { seen[href] = 1; links.push('<a href="' + href + '">' + a2.textContent + '</a>'); }
          });
        });
        var chap = a.links.map(function (l) { return '<a href="' + l[1] + '">' + l[0] + '</a>'; }).join(", ");
        h.push('<div class="arow"><span class="an">' + a.name + '</span><span class="as">' + got + ' of ' + mine.length + '</span>' +
          '<div class="meter" role="img" aria-label="' + got + ' of ' + mine.length + ' right"><i style="width:' + Math.round(100 * got / mine.length) + '%"></i></div>' +
          '<span class="ar-rev">' + (links.length ? "Revise " + links.join(", ") + "." : "All right. Chapter: " + chap + ".") + '</span></div>');
      });
      var missed = qs.filter(function (q) { return q.el.classList.contains("missed"); });
      if (missed.length) {
        h.push('<h4>Questions to try again</h4><ol class="miss">');
        missed.forEach(function (q) {
          var tag = q.el.querySelector(".pq-tag").textContent;
          h.push('<li value="' + q.n + '"><a href="#q' + q.n + '">Question ' + q.n + '</a> (' + tag + ')</li>');
        });
        h.push('</ol>');
      } else {
        h.push('<p class="sub">Full marks. Come back in a week, press Start over, and attempt the set again from a blank sheet.</p>');
      }
      h.push('<div class="acts">' + (missed.length ? '<button type="button" class="btn btn-main" id="btn-retry">Try the missed questions again</button>' : "") +
        '<button type="button" class="btn" id="btn-over">Start over</button></div>');
      h.push('<p class="note">The page does not show correct answers. After you revise, try the missed questions again. The first-attempt score stays on this page.</p>');
      box.innerHTML = h.join("");
      box.hidden = false;
      $("btn-score").disabled = true;
      Kit.setText("score-note", "");
      var again = $("btn-retry");
      if (again) again.addEventListener("click", retry);
      wireOver($("btn-over"));
      document.querySelector(".pb-prog").innerHTML = "Scored <b>" + right + "</b> of " + qs.length;
      paint();
      if (fresh) {
        box.scrollIntoView({ behavior: Kit.reduce ? "auto" : "smooth", block: "start" });
        box.focus({ preventScroll: true });
      }
    }
    function score() {
      var blank = qs.filter(unreadable).length;
      if (blank && !armed) {
        armed = true;
        Kit.setText("score-note", blank + (blank === 1 ? " question is" : " questions are") + " blank or cannot be read as a number, and will count as missed. Press the button again to score anyway.");
        clearTimeout(armTimer);
        armTimer = setTimeout(function () { armed = false; Kit.setText("score-note", ""); }, 8000);
        return;
      }
      armed = false;
      pause();
      state.done = true;
      qs.forEach(function (q) { state.a[q.id] = answerOf(q); });
      save();
      showResults(true);
    }
    function retry() {
      qs.forEach(function (q) {
        if (q.el.classList.contains("missed")) {
          setAnswer(q, ""); delete state.a[q.id];
          q.el.classList.remove("missed");
          q.st.hidden = true; q.rev.hidden = true;
          q.inputs.forEach(function (i) { i.disabled = false; });
        }
      });
      state.done = false;
      save();
      $("results").hidden = true;
      $("btn-score").disabled = false;
      AREAS.forEach(function (a) { var b = $(a.id + "-h").querySelector(".area-badge"); if (b) b.remove(); });
      document.querySelector(".pb-prog").innerHTML = '<b id="pb-n">0</b> of ' + qs.length + '<span class="wide"> answered</span>';
      progress(); paint();
      var first = qs.filter(function (q) { return !q.el.classList.contains("right"); })[0];
      if (first) first.el.scrollIntoView({ behavior: Kit.reduce ? "auto" : "smooth", block: "start" });
    }
    function wireOver(btn) {
      var wait = 0;
      btn.addEventListener("click", function () {
        if (!wait) {
          btn.textContent = "Press again to clear all answers";
          wait = setTimeout(function () { wait = 0; btn.textContent = "Start over"; }, 4000);
          return;
        }
        clearTimeout(wait);
        startOver();
      });
    }
    function startOver() {
      running = false; base = 0;
      state = { a: {}, min: state.min, el: 0, used: false, done: false, first: null };
      qs.forEach(function (q) {
        setAnswer(q, "");
        q.el.classList.remove("right", "missed");
        q.st.hidden = true; q.rev.hidden = true;
      });
      lockAll(false);
      $("results").hidden = true;
      $("btn-score").disabled = false;
      AREAS.forEach(function (a) { var b = $(a.id + "-h").querySelector(".area-badge"); if (b) b.remove(); });
      document.querySelector(".pb-prog").innerHTML = '<b id="pb-n">0</b> of ' + qs.length + '<span class="wide"> answered</span>';
      progress(); paint(); save();
      window.scrollTo({ top: 0, behavior: Kit.reduce ? "auto" : "smooth" });
    }
    $("btn-score").addEventListener("click", score);

    // ---------- wiring ----------
    qs.forEach(function (q) {
      q.inputs.forEach(function (i) {
        i.addEventListener(q.kind === "mc" ? "change" : "input", function () {
          paintChoice(q);
          state.a[q.id] = answerOf(q);
          progress();
          save();
        });
      });
    });
    load();
    Kit.chips($("tm-chips"), MINUTES.map(function (m) { return m + " minutes"; }), MINUTES.indexOf(state.min), function (i) { state.min = MINUTES[i]; paint(); save(); });
    qs.forEach(function (q) { if (state.a[q.id] != null) setAnswer(q, state.a[q.id]); });
    base = state.el;
    checkStore();
    progress();
    paint();
    if (state.done) showResults(false);
  });
})();
