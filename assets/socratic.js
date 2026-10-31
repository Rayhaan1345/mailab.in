// Socratic Tutor report: the Meno scene, the study charts, the plugin file viewer,
// the skill-loading demonstration, the tutor simulator and the subject notes.
// Numbers come from the cited papers; tutor rules come from the plugin's SKILL.md files.
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  function $(id) { return document.getElementById(id); }
  function el(name, attrs, parent) {
    var n = document.createElementNS(NS, name);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function text(parent, x, y, s, attrs) {
    var a = { x: x, y: y };
    for (var k in attrs) a[k] = attrs[k];
    var t = el("text", a, parent);
    t.textContent = s;
    return t;
  }
  function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); }
  function width(svg) { return Math.max(280, Math.round(svg.getBoundingClientRect().width || svg.parentNode.clientWidth)); }
  function debounce(fn) { var t = 0; return function () { clearTimeout(t); t = setTimeout(fn, 150); }; }
  function ease(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }
  function tween(ms, fn) {
    if (reduce) { fn(1); return; }
    var t0 = 0;
    requestAnimationFrame(function f(ts) {
      if (!t0) t0 = ts;
      var p = Math.min(1, (ts - t0) / ms);
      fn(ease(p));
      if (p < 1) requestAnimationFrame(f);
    });
  }
  function press(buttons, on) { buttons.forEach(function (b) { b.setAttribute("aria-pressed", String(b === on)); }); }
  function turn(list, who, label, say, tag, tagClass) {
    var li = document.createElement("li");
    li.className = who;
    var w = document.createElement("span"); w.className = "who"; w.textContent = label;
    var s = document.createElement("span"); s.className = "say";
    if (tag) { var t = document.createElement("span"); t.className = "tag" + (tagClass ? " " + tagClass : ""); t.textContent = tag; s.appendChild(t); }
    s.appendChild(document.createTextNode(say));
    li.appendChild(w); li.appendChild(s); list.appendChild(li);
    return li;
  }

  // ---------- Meno: doubling the square ----------

  (function meno() {
    var svg = $("meno-svg");
    if (!svg) return;
    var list = $("meno-turns"), no = $("meno-no"), prev = $("meno-prev"), next = $("meno-next");
    var U = 60, O = 30;
    // a 4 x 4 grid of feet
    for (var i = 0; i <= 4; i++) {
      el("line", { x1: O + i * U, x2: O + i * U, y1: O, y2: O + 4 * U, style: "stroke:var(--hair)", "stroke-width": 1 }, svg);
      el("line", { y1: O + i * U, y2: O + i * U, x1: O, x2: O + 4 * U, style: "stroke:var(--hair)", "stroke-width": 1 }, svg);
    }
    function sq(attrs, fill) {
      var a = { "class": "sq", style: "fill:" + fill + ";fill-opacity:0.18;stroke:" + fill, "stroke-width": 3, opacity: 0 };
      for (var k in attrs) a[k] = attrs[k];
      return el(attrs.points ? "polygon" : "rect", a, svg);
    }
    var S = {
      s4: sq({ x: O, y: O, width: 4 * U, height: 4 * U }, "var(--bar-c)"),
      s3: sq({ x: O, y: O, width: 3 * U, height: 3 * U }, "var(--bar-a)"),
      s2: sq({ x: O, y: O, width: 2 * U, height: 2 * U }, "var(--orange)"),
      tilt: sq({ points: [O + 2 * U, O, O + 4 * U, O + 2 * U, O + 2 * U, O + 4 * U, O, O + 2 * U].join(" ") }, "var(--orange)")
    };
    var mid = el("g", { opacity: 0, "class": "sq" }, svg);
    el("line", { x1: O, y1: O + 2 * U, x2: O + 4 * U, y2: O + 2 * U, style: "stroke:var(--bar-c)", "stroke-width": 2 }, mid);
    el("line", { x1: O + 2 * U, y1: O, x2: O + 2 * U, y2: O + 4 * U, style: "stroke:var(--bar-c)", "stroke-width": 2 }, mid);
    var L = {
      a4: text(svg, O + U, O + U + 6, "4", { "text-anchor": "middle", "font-size": 26, "font-weight": 800, style: "fill:var(--orange-ink)", opacity: 0, "class": "sq" }),
      a16: text(svg, O + 3.3 * U, O + 3.6 * U, "16", { "text-anchor": "middle", "font-size": 26, "font-weight": 800, style: "fill:var(--ink-2)", opacity: 0, "class": "sq" }),
      a9: text(svg, O + 2.5 * U, O + 2.65 * U, "9", { "text-anchor": "middle", "font-size": 26, "font-weight": 800, style: "fill:var(--bar-a)", opacity: 0, "class": "sq" }),
      a8: text(svg, O + 2 * U, O + 2 * U + 9, "8", { "text-anchor": "middle", "font-size": 34, "font-weight": 800, style: "fill:var(--orange-ink)", opacity: 0, "class": "sq" })
    };
    var STEPS = [
      { show: ["s2", "a4"], talk: [["Socrates", "This square has sides of 2 feet, so its area is 4 square feet. How long is the side of a square with twice the area?"], ["Boy", "Twice as long, Socrates. Sides of 4 feet."]] },
      { show: ["s4", "s2", "a16", "mid"], talk: [["Socrates", "Here is the square with sides of 4. How many of the first squares fit inside it?"], ["Boy", "Four of them. So its area is 16, four times as big."]] },
      { show: ["s3", "s2", "a9"], talk: [["Socrates", "So the side must be longer than 2 and shorter than 4. What would you try?"], ["Boy", "Three feet."], ["Socrates", "How big is a square with sides of 3?"], ["Boy", "Nine square feet. Still too big. I don't know what the side should be."]] },
      { show: ["s4", "mid", "tilt"], talk: [["Socrates", "Draw a line from corner to corner across the four small squares, as here. What part of a small square lies inside the tilted figure?"], ["Boy", "Half of it."]] },
      { show: ["s4", "mid", "tilt", "a8", "fill"], talk: [["Socrates", "There are four halves, and a small square has area 4. What is the area of the tilted square?"], ["Boy", "Eight. Its side is the line across the first square."]] }
    ];
    var step = 0;
    function paint() {
      var st = STEPS[step];
      for (var k in S) S[k].setAttribute("opacity", st.show.indexOf(k) >= 0 ? 1 : 0);
      for (var j in L) L[j].setAttribute("opacity", st.show.indexOf(j) >= 0 ? 1 : 0);
      mid.setAttribute("opacity", st.show.indexOf("mid") >= 0 ? 1 : 0);
      S.tilt.style.fillOpacity = st.show.indexOf("fill") >= 0 ? "0.45" : "0.08";
      clear(list);
      st.talk.forEach(function (t) { turn(list, t[0] === "Socrates" ? "tutor" : "student", t[0], t[1]); });
      no.textContent = "Step " + (step + 1) + " of " + STEPS.length;
      prev.disabled = step === 0;
      next.textContent = step === STEPS.length - 1 ? "Start again" : "Next question";
    }
    prev.addEventListener("click", function () { if (step > 0) { step--; paint(); } });
    next.addEventListener("click", function () { step = step === STEPS.length - 1 ? 0 : step + 1; paint(); });
    paint();
  })();

  // ---------- Bastani et al.: practice against exam ----------

  (function bastani() {
    var svg = $("bastani-svg");
    if (!svg) return;
    var note = $("bastani-note"), mode = "practice", W = 0, bars = [];
    var ROWS = [["Plain GPT-4 chat", "var(--bar-c)"], ["Hint-giving GPT-4 tutor", "var(--orange)"]];
    var VAL = { practice: [48, 127], exam: [-17, 0] };
    var NOTE = {
      practice: "During practice, compared with students who used notes and a textbook, scores were 48% higher with the plain chat and 127% higher with the hint-giving tutor.",
      exam: "On the exam without AI, students who had the plain chat scored 17% lower than the notes-and-textbook group. Students who had the hint-giving tutor scored about the same as that group."
    };
    var btns = [].slice.call(document.querySelectorAll("#bastani [data-bv]"));
    btns.forEach(function (b) { b.addEventListener("click", function () { press(btns, b); mode = b.dataset.bv; update(); }); });
    var X;
    function build() {
      clear(svg); bars = [];
      W = width(svg);
      var narrow = W < 520, lab = narrow ? 0 : 190, top = 28, rh = narrow ? 64 : 46;
      var x0 = lab + 10, x1 = W - 52, lo = -30, hi = 140;
      X = function (v) { return x0 + (x1 - x0) * (v - lo) / (hi - lo); };
      var H = top + ROWS.length * rh + 34;
      [-25, 0, 25, 50, 75, 100, 125].forEach(function (v) {
        el("line", { x1: X(v), x2: X(v), y1: top - 24, y2: H - 26, style: v === 0 ? "stroke:var(--ink)" : "stroke:var(--hair)", "stroke-width": v === 0 ? 2 : 1 }, svg);
        text(svg, X(v), H - 8, (v > 0 ? "+" : v < 0 ? "−" : "") + Math.abs(v) + "%", { "text-anchor": "middle", "font-size": 13, style: "fill:var(--muted)" });
      });
      text(svg, X(0) + 6, top - 12, "notes and textbook", { "font-size": 12.5, style: "fill:var(--muted)" });
      ROWS.forEach(function (r, i) {
        var y = top + i * rh + (narrow ? 22 : 10);
        text(svg, narrow ? 0 : lab, narrow ? y - 6 : y + 17, r[0], { "text-anchor": narrow ? "start" : "end", "font-size": 15, "font-weight": 600, style: "fill:var(--ink)" });
        var rect = el("rect", { x: X(0), y: y, width: 0, height: 24, style: "fill:" + r[1] }, svg);
        var t = text(svg, X(0), y + 17, "", { "font-size": 14.5, "font-weight": 700, style: "fill:var(--ink)" });
        bars.push({ rect: rect, t: t, v: 0 });
      });
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    }
    function update() {
      var to = VAL[mode], from = bars.map(function (b) { return b.v; });
      tween(600, function (e) {
        bars.forEach(function (b, i) {
          var v = from[i] + (to[i] - from[i]) * e;
          b.v = v;
          var a = X(Math.min(0, v)), w = Math.abs(X(v) - X(0));
          b.rect.setAttribute("x", a.toFixed(1)); b.rect.setAttribute("width", Math.max(w, 0).toFixed(1));
          var label = to[i] === 0 ? "about the same" : (to[i] > 0 ? "+" : "−") + Math.abs(to[i]) + "%";
          b.t.textContent = e < 1 ? "" : label;
          // a negative value is labelled to the right of the zero line, clear of the row name
          b.t.setAttribute("x", (v >= 0 ? X(v) + 6 : X(0) + 6).toFixed(1));
          b.t.setAttribute("text-anchor", "start");
        });
      });
      note.textContent = NOTE[mode];
    }
    build(); update();
    window.addEventListener("resize", debounce(function () { if (width(svg) !== W) { build(); bars.forEach(function (b) { b.v = 0; }); update(); } }));
  })();

  // ---------- Kestin et al.: median scores ----------

  (function kestin() {
    var svg = $("kestin-svg");
    if (!svg) return;
    var W = 0;
    function build() {
      clear(svg);
      W = width(svg);
      var narrow = W < 520, lab = narrow ? 0 : 150, x0 = lab + 14, x1 = W - 20, lo = 2.5, hi = 5;
      var X = function (v) { return x0 + (x1 - x0) * (v - lo) / (hi - lo); };
      var rows = [["After the class", 3.5, "var(--bar-c)"], ["After the AI tutor", 4.5, "var(--orange)"]];
      var top = 30, rh = narrow ? 58 : 44, H = top + rows.length * rh + 40;
      text(svg, x1, H + 12, "median score on the test after the lesson", { "text-anchor": "end", "font-size": 13, style: "fill:var(--muted)" });
      for (var v = 2.5; v <= 5.001; v += 0.5) {
        el("line", { x1: X(v), x2: X(v), y1: top - 8, y2: H - 30, style: "stroke:var(--hair)" }, svg);
        text(svg, X(v), H - 12, v.toFixed(1), { "text-anchor": "middle", "font-size": 13, style: "fill:var(--muted)" });
      }
      el("line", { x1: X(2.75), x2: X(2.75), y1: top - 12, y2: H - 30, style: "stroke:var(--ink)", "stroke-dasharray": "4 4", "stroke-width": 1.5 }, svg);
      text(svg, X(2.75) + 6, top - 14, "pre-test 2.75", { "font-size": 13, style: "fill:var(--ink-2)" });
      rows.forEach(function (r, i) {
        var y = top + i * rh + (narrow ? 26 : 16);
        text(svg, narrow ? 0 : lab, narrow ? y - 14 : y + 5, r[0], { "text-anchor": narrow ? "start" : "end", "font-size": 15, "font-weight": 600, style: "fill:var(--ink)" });
        el("line", { x1: X(2.75), x2: X(r[1]), y1: y, y2: y, style: "stroke:" + r[2], "stroke-width": 8, "stroke-linecap": "round" }, svg);
        el("circle", { cx: X(r[1]), cy: y, r: 9, style: "fill:" + r[2] }, svg);
        text(svg, X(r[1]) + (r[1] > 4.7 ? -14 : 14), y + 5, String(r[1]), { "text-anchor": r[1] > 4.7 ? "end" : "start", "font-size": 15, "font-weight": 700, style: "fill:var(--ink)" });
      });
      svg.setAttribute("viewBox", "0 0 " + W + " " + (H + 18));
      svg.setAttribute("aria-label", "Median scores on the post-tests: 3.5 after the class and 4.5 after the AI tutor, from a pre-test median of 2.75");
    }
    build();
    window.addEventListener("resize", debounce(function () { if (width(svg) !== W) build(); }));
  })();

  // ---------- Roediger and Karpicke: rereading against recall ----------

  (function recall() {
    var svg = $("rk-svg");
    if (!svg) return;
    var note = $("rk-note"), state = 0, W = 0, bars = [];
    var G = [["Read 4 times", "read 14.2 times", "var(--bar-c)"], ["Read 3, recall 1", "read 10.3 times", "var(--bar-a)"], ["Read 1, recall 3", "read 3.4 times", "var(--orange)"]];
    var VAL = [[83, 78, 71], [40, 56, 61]];
    var NOTE = [
      "Five minutes after learning, the rereading group recalled the most: 83% of the ideas in the passage, against 78% and 71%.",
      "A week later the order flipped. The group that recalled three times kept 61% of the ideas, against 40% for the group that only reread."
    ];
    var btns = [].slice.call(document.querySelectorAll("#testing [data-rk]"));
    btns.forEach(function (b) { b.addEventListener("click", function () { press(btns, b); state = Number(b.dataset.rk); update(); }); });
    var Y, base;
    function build() {
      clear(svg); bars = [];
      W = width(svg);
      var H = 280, m = { l: 40, r: 10, t: 14, b: 58 };
      base = H - m.b;
      Y = function (v) { return base - (base - m.t) * v / 100; };
      [0, 25, 50, 75, 100].forEach(function (v) {
        el("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v), style: "stroke:var(--hair)" }, svg);
        text(svg, m.l - 8, Y(v) + 4, v + "%", { "text-anchor": "end", "font-size": 12.5, style: "fill:var(--muted)" });
      });
      var slot = (W - m.l - m.r) / 3, bw = Math.min(110, slot * 0.6);
      G.forEach(function (g, i) {
        var cx = m.l + slot * (i + 0.5);
        var rect = el("rect", { x: cx - bw / 2, y: base, width: bw, height: 0, style: "fill:" + g[2] }, svg);
        var t = text(svg, cx, base - 8, "", { "text-anchor": "middle", "font-size": 18, "font-weight": 800, style: "fill:var(--ink)" });
        text(svg, cx, base + 22, g[0], { "text-anchor": "middle", "font-size": W < 420 ? 13 : 15, "font-weight": 600, style: "fill:var(--ink)" });
        text(svg, cx, base + 42, g[1], { "text-anchor": "middle", "font-size": 13, style: "fill:var(--muted)" });
        bars.push({ rect: rect, t: t, v: 0 });
      });
      svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    }
    function update() {
      var to = VAL[state], from = bars.map(function (b) { return b.v; });
      tween(650, function (e) {
        bars.forEach(function (b, i) {
          var v = from[i] + (to[i] - from[i]) * e;
          b.v = v;
          b.rect.setAttribute("y", Y(v).toFixed(1)); b.rect.setAttribute("height", (base - Y(v)).toFixed(1));
          b.t.setAttribute("y", (Y(v) - 8).toFixed(1));
          b.t.textContent = Math.round(v) + "%";
        });
      });
      note.textContent = NOTE[state];
    }
    build(); update();
    window.addEventListener("resize", debounce(function () { if (width(svg) !== W) { build(); update(); } }));
  })();

  // ---------- the plugin's files ----------

  (function tree() {
    var list = $("tree-list");
    if (!list) return;
    var what = $("tree-what");
    var WHAT = {
      plugin: "The manifest. It gives the plugin its name, version and description.",
      market: "The marketplace file. It lets people add this repository as a catalog of plugins and install from it.",
      skill: "The tutor skill. Claude reads the description in the header to decide when to load it. The text below the header is the instructions.",
      ladder: "A reference file. SKILL.md points to it, and Claude opens it when it needs the ladder of questions.",
      notes: "A reference file of subject notes: what to ask first, common blockers and what to answer plainly.",
      quiz: "The second skill, for short recall quizzes.",
      eval: "A test case: a student's message and the checks the reply must pass.",
      privacy: "The privacy policy. The plugin collects no data."
    };
    var btns = [].slice.call(list.querySelectorAll("button[data-file]"));
    var pres = [].slice.call(document.querySelectorAll("#tree pre[data-file]"));
    function show(key) {
      btns.forEach(function (b) { b.setAttribute("aria-selected", String(b.dataset.file === key)); });
      pres.forEach(function (p) { p.hidden = p.dataset.file !== key; });
      what.textContent = WHAT[key];
    }
    btns.forEach(function (b) { b.addEventListener("click", function () { show(b.dataset.file); }); });
    show("skill");
  })();

  // ---------- which skill loads ----------

  (function loader() {
    var box = $("ld-msgs");
    if (!box) return;
    var MSGS = [
      { m: "I have to solve 2x² − 5x − 3 = 0 for my homework. Can you help?", load: "socratic-tutor. The student is studying and asks for help with homework.", open: "references/question-ladder.md, when it picks the next question.", reply: "One question about what the student has tried. It does not give x = 3 or x = −½." },
      { m: "My Python code should add up a list but prints the wrong total. Can you fix it?", load: "socratic-tutor. The student is debugging code they wrote.", open: "references/subject-notes.md, for the programming notes.", reply: "One question that points at where total is set to 0. It never pastes a corrected program." },
      { m: "Bhai mera maths homework hai, 3x + 7 = 22 solve karna hai.", load: "socratic-tutor. This is homework, written in Hinglish.", open: "references/question-ladder.md, when it picks the next question.", reply: "A reply in Hinglish with one question to get started. It does not give x = 5." },
      { m: "Quiz me on Newton's laws, five questions. I'm in grade 11.", load: "quiz-me. The student asked to be quizzed.", open: "No extra files. The quiz rules are in its SKILL.md.", reply: "One quiz question about Newton's laws, then it waits for the answer." },
      { m: "What does homeostasis mean? One line is fine.", load: "Neither skill. The tutor's description rules out definitions.", open: "No files.", reply: "A plain definition in a sentence or two, with no question first.", none: true },
      { m: "What is the capital of Australia?", load: "Neither skill. This is a factual question.", open: "No files.", reply: "Canberra, in the first sentence.", none: true }
    ];
    var steps = [$("ld-1"), $("ld-2"), $("ld-3"), $("ld-4")];
    var btns = MSGS.map(function (d, i) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = "“" + d.m + "”"; b.setAttribute("aria-pressed", String(i === 0));
      b.addEventListener("click", function () { press(btns, b); show(d); });
      box.appendChild(b);
      return b;
    });
    var timers = [];
    function show(d) {
      timers.forEach(clearTimeout); timers = [];
      $("ld-load").textContent = d.load; $("ld-open").textContent = d.open; $("ld-reply").textContent = d.reply;
      var cls = [d.none ? "" : "hit", d.none ? "skip" : "hit", d.none ? "skip" : "hit", "hit"];
      steps.forEach(function (s) { s.className = ""; });
      steps.forEach(function (s, i) {
        if (reduce) { s.className = cls[i]; return; }
        timers.push(setTimeout(function () { s.className = cls[i]; }, 180 * i));
      });
    }
    show(MSGS[0]);
  })();

  // ---------- tutor simulator ----------
  // State: the rung on the question ladder, the hint level on that rung, and the run of right answers without a hint.

  (function simulator() {
    var list = $("sim-turns");
    if (!list) return;
    var state = $("sim-state");
    var rungEls = [].slice.call(document.querySelectorAll("#sim-rungs li"));
    var hintEls = [].slice.call(document.querySelectorAll("#sim-hints li"));
    var acts = [].slice.call(document.querySelectorAll("#sim-controls [data-act]"));
    var NAMES = ["Clarify", "Recall", "Simpler case", "Apply", "Check", "Generalise"];
    var HINT_TAGS = ["Nudge", "Concept", "Worked step"];
    var R = [
      { q: "What is the question asking you to find, in your own words?",
        right: "The values of x that make the left side equal zero.",
        wrong: "The value of 2x² − 5x − 3?",
        hints: ["Look at the = 0 at the end. Are you finding a number for x, or a value for the whole expression?",
                "Solving means finding the values of x that make the equation true. For which x would the left side come out as 0?",
                "Try x = 1. The left side is 2 − 5 − 3 = −6, which is not 0, so x = 1 is not a solution. What are we looking for, then?"] },
      { q: "What methods do you know for solving a quadratic?",
        right: "Factorising, or the quadratic formula.",
        wrong: "Divide both sides by x?",
        hints: ["Think about how your class solved x² − 9 = 0. What did you do to the left side?",
                "The two standard methods are factorising and the quadratic formula. Which of the two have you practised more?",
                "Factorising writes the left side as two brackets multiplied together. If two brackets multiply to 0, what must be true of one of them?"] },
      { q: "Try an easier one first. Which two numbers multiply to 6 and add to −5?",
        faded: "Try the easier x² − 5x + 6 = 0 first. How does it factorise?",
        right: "−2 and −3, so x² − 5x + 6 = (x − 2)(x − 3).",
        wrong: "−1 and −6?",
        hints: ["The two numbers add to a negative total and multiply to a positive one. What does that tell you about their signs?",
                "For x² + bx + c you need two numbers that multiply to c and add to b. Which pairs of whole numbers multiply to 6?",
                "The pairs are 1 and 6, and 2 and 3. With minus signs, −2 and −3 add to −5. How would you write the brackets?"] },
      { q: "Now the original. With a leading coefficient of 2, how does 2x² − 5x − 3 factorise?",
        faded: "Now factorise the original equation. What do you get?",
        right: "(2x + 1)(x − 3), so x = 3 or x = −½.",
        wrong: "(2x − 3)(x + 1)?",
        hints: ["Look at the constant term. What are its factor pairs?",
                "This is the product-sum method with a leading coefficient. The product you need is 2 times −3. Which two numbers multiply to −6 and add to −5?",
                "The product is −6 and the sum is −5, so the pair is −6 and +1. Split the middle term: 2x² − 6x + x − 3. Can you finish?"] },
      { q: "Put x = 3 back into the original equation. What do you get?",
        faded: "How can you check your roots?",
        right: "Substitute x = 3: 2(9) − 15 − 3 = 0, so it works.",
        wrong: "18 + 15 − 3 = 30?",
        hints: ["Replace x with 3 in the middle term. What is −5 times 3?",
                "A root makes the left side zero, so substituting is a check. What is 2 × 3² on its own?",
                "Work it out in pieces: 2 × 9 = 18 and −5 × 3 = −15, so the left side is 18 − 15 − 3. What does that come to?"] },
      { q: "When would you choose the formula over factorising?",
        faded: "When would this method not work?",
        right: "When I can't find whole-number factors quickly, I'd use the formula.",
        wrong: "Never. Factorising always works.",
        hints: ["Try 2x² − 5x − 1 = 0. Can you find whole numbers that work?",
                "The formula works on any quadratic, and factorising is quicker when the factors are whole numbers. Which would you try on 2x² − 5x − 1 = 0?",
                "For 2x² − 5x − 1 = 0, no whole numbers multiply to −2 and add to −5, so factorising stalls. What would you reach for?"] }
    ];
    var FULL = "(2x + 1)(x − 3) = 0, so x = 3 or x = −½. Why did we need the product to be −6 rather than −3?";
    var r, h, streak, done, last;

    function tutor(say, tag, cls) { turn(list, "tutor", "Tutor", say, tag, cls); }
    function student(say) { turn(list, "student", "Student", say); }
    function ask(prefix) {
      var faded = streak >= 2 && R[r].faded;
      tutor((prefix || "") + (faded ? R[r].faded : R[r].q), faded ? "Less guidance" : "Rung " + (r + 1) + " · " + NAMES[r], faded ? "" : "rung");
      last = faded ? "Two right answers in a row without a hint, so this question gives less guidance." : "";
    }
    function paint() {
      rungEls.forEach(function (li) {
        var n = Number(li.dataset.r) - 1;
        li.classList.toggle("on", !done && n === r);
        li.classList.toggle("done", n < r || (done && n <= r));
      });
      hintEls.forEach(function (li) { li.classList.toggle("on", Number(li.dataset.h) <= h); });
      acts.forEach(function (b) { if (b.dataset.act !== "reset") b.disabled = done; });
      state.textContent = done ? "Session finished. Start again to try another path."
        : "Rung " + (r + 1) + " of 6. Hints on this rung: " + h + ". Right in a row without a hint: " + streak + "." + (last ? " " + last : "");
      list.scrollTop = list.scrollHeight;
    }
    function reset() {
      clear(list);
      r = 0; h = 0; streak = 0; done = false; last = "";
      tutor("This is a quadratic with a leading coefficient of 2, which is where most people slip. What have you tried so far, even if it went nowhere?");
      student("I haven't tried anything yet. I'm not sure where to start.");
      ask();
      paint();
    }
    function stall(say) {
      student(say);
      streak = 0; last = "";
      h++;
      if (h <= 3) { tutor(R[r].hints[h - 1], HINT_TAGS[h - 1]); last = "A stall moves one step up the hint scale."; }
      else { tutor(FULL, "Full answer"); done = true; }
    }
    function act(a) {
      if (a === "reset") { reset(); return; }
      if (done) return;
      if (a === "right") {
        student(R[r].right);
        streak = h === 0 ? streak + 1 : 0;
        h = 0;
        if (r === R.length - 1) {
          tutor("Right. Put the method in your own words, in two sentences.", "Summary");
          student("Multiply the first and last coefficients and find two numbers with that product that add to the middle one. Split the middle term, factorise in pairs, then check by substituting.");
          tutor("Right. That is the method.");
          r = R.length - 1; done = true;
        } else {
          r++;
          ask("Right. ");
        }
      } else if (a === "wrong") { stall(R[r].wrong); }
      else if (a === "idk") { stall("I don't know."); }
      else if (a === "tell") {
        student("Just tell me.");
        h = 4; tutor(FULL, "Answer"); done = true;
      }
      paint();
    }
    acts.forEach(function (b) { b.addEventListener("click", function () { act(b.dataset.act); }); });
    reset();
  })();

  // ---------- subject notes ----------

  (function notes() {
    var box = $("nt-subjects");
    if (!box) return;
    var N = [
      ["Mathematics", "Which method, and why that one?", "Formulae they are expected to look up, standard identities, the names of methods, arithmetic they have set up correctly and asked you to check.",
        [["Sign errors when moving terms", "2x - 5 = 3 becomes 2x = -2", "What did you do to both sides to get rid of the -5?"],
         ["Fractions", "Adds numerators and denominators separately", "If you had 1/2 of a pizza and 1/3 of another, is that 2/5 of a pizza?"],
         ["Derivative as a rule, not a rate", "Differentiates correctly and cannot interpret the result", "Your answer is 6. Six of what, per what?"]]],
      ["Physics", "What is given, what is unknown, and can you draw it?", "Constants, unit conversions, the statement of a law they ask for by name.",
        [["Force and motion", "Thinks a moving object needs a force to keep moving", "A puck slides on ice with no friction. What force keeps it going?"],
         ["Circuits", "Thinks current is used up along a series circuit", "If current were used up, what would happen to the charge that flows into the bulb?"],
         ["Vectors", "Adds magnitudes of vectors at an angle", "If you walk 3 m east and 4 m north, how far are you from the start?"]]],
      ["Chemistry", "What is reacting, and what is the balanced equation?", "Atomic masses, constants, the formula of a named compound, names of apparatus.",
        [["Balancing", "Changes subscripts to balance", "If you change H2O to H2O2, is it still water?"],
         ["Equilibrium", "Thinks equilibrium means equal amounts", "What is equal at equilibrium: the amounts, or the rates?"],
         ["Acids and bases", "Equates strong with concentrated", "Can you have a dilute solution of a strong acid? What would its pH depend on?"]]],
      ["Biology", "Which system or process is this about, and what is it for?", "Definitions, the function of a named structure, the names of stages.",
        [["Natural selection", "Says organisms adapt because they need to", "Does the giraffe stretch its neck, or do longer-necked giraffes leave more offspring?"],
         ["Genetics crosses", "Gives a gamete two alleles of one gene", "How many alleles for this gene end up in a gamete?"],
         ["Experiments", "Cannot name the variables", "What did you change, what did you measure, and what did you keep the same?"]]],
      ["Programming", "What do you expect this to print on one small input, and what does it print?", "Syntax, the name of a library function, what an error message means, how to print a variable.",
        [["Accumulator reset", "Total is reset inside the loop", "What is total after the first iteration? And the second?"],
         ["Off-by-one", "Loop runs one time too many or too few", "What is the value of the loop variable on the last pass? Is that the index you meant?"],
         ["Reading errors", "Pastes the traceback without reading it", "Which line does the error name, and what does the last sentence of the message say?"]]],
      ["Writing", "What is your claim, in one sentence, and what is the strongest evidence for it?", "Spelling, a word's meaning, a citation format, the convention of a genre they ask about.",
        [["No thesis", "The introduction describes the topic and takes no position", "If someone disagreed with your essay, what would they be disagreeing with?"],
         ["Summary instead of argument", "Paragraphs retell the source", "This paragraph tells me what happens. What does it show?"],
         ["Vague words", "“Many factors”, “very important”, “society”", "Which factors? Important to whom?"]]],
      ["Languages", "Try the sentence first, then we fix it.", "Vocabulary, conjugation tables, pronunciation. Vocabulary is recall.",
        [["Word-for-word translation", "Native-language word order", "In the language you are learning, where does the verb go in this kind of sentence?"],
         ["Tense", "Uses present for a completed action", "When did this happen: now, before, or after? Which tense marks that?"]]],
      ["History and economics", "What is the question actually asking: describe, explain, compare, or evaluate?", "Dates, names, definitions, the content of a named treaty or policy.",
        [["Single cause", "One reason for a complex event", "If that cause were removed, would the event still have happened? What else was needed?"],
         ["Source as fact", "Treats a source as true because it is old or official", "Who wrote this, for whom, and what did they want the reader to think?"],
         ["Economics diagrams", "Draws the shift without saying why", "What changed for buyers or sellers that moves this curve?"]]]
    ];
    var cur = 0, k = 0;
    var btns = N.map(function (d, i) {
      var b = document.createElement("button");
      b.type = "button"; b.textContent = d[0]; b.setAttribute("aria-pressed", String(i === 0));
      b.addEventListener("click", function () { press(btns, b); cur = i; k = 0; paint(); });
      box.appendChild(b);
      return b;
    });
    function paint() {
      var d = N[cur], row = d[3][k];
      $("nt-first").textContent = "“" + d[1] + "”";
      $("nt-plain").textContent = d[2];
      $("nt-blocker").textContent = row[0];
      $("nt-looks").textContent = "What it looks like: " + row[1];
      $("nt-q").textContent = "“" + row[2] + "”";
    }
    $("nt-next").addEventListener("click", function () { k = (k + 1) % N[cur][3].length; paint(); });
    paint();
  })();
})();
