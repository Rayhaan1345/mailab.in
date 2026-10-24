(function () {
  "use strict";

  // Scripted replies for the hint ladder on 2x^2 - 5x - 3 = 0.
  var HINTS = [
    {
      tag: "Nudge",
      tutor: "Look at the first and last coefficients. Which two numbers multiply to 2 × (−3) = −6 and add to −5?",
      student: "−6 and 1? I'm not sure what to do with them."
    },
    {
      tag: "Concept",
      tutor: "Yes. Use them to split the middle term: write −5x as −6x + x. Then factorise the first two terms and the last two terms separately.",
      student: "2x² − 6x + x − 3. I'm not sure how to group it."
    },
    {
      tag: "Worked step",
      tutor: "2x² − 6x + x − 3 = 2x(x − 3) + 1(x − 3). What do the two groups have in common?",
      student: "Both have (x − 3). What do I do next?"
    },
    {
      tag: "Full answer",
      tutor: "So the equation is (2x + 1)(x − 3) = 0, which gives x = 3 or x = −½. Substitute x = 3 back into the equation. Do you get zero?",
      student: null
    }
  ];

  var turns = document.getElementById("turns");
  if (!turns) return;
  var next = document.getElementById("hint-next");
  var tell = document.getElementById("hint-tell");
  var reset = document.getElementById("hint-reset");
  var count = document.getElementById("hint-count");
  var rungs = document.querySelectorAll(".rungs li");
  var initial = turns.innerHTML;
  var used = 0;

  function turn(who, text, tag) {
    var li = document.createElement("li");
    li.className = who;
    var w = document.createElement("span");
    w.className = "who";
    w.textContent = who === "tutor" ? "Tutor" : "Student";
    var s = document.createElement("span");
    s.className = "say";
    if (tag) {
      var t = document.createElement("span");
      t.className = "tag";
      t.textContent = tag;
      s.appendChild(t);
    }
    s.appendChild(document.createTextNode(text));
    li.appendChild(w);
    li.appendChild(s);
    turns.appendChild(li);
  }

  function update() {
    rungs.forEach(function (r) { r.classList.toggle("on", Number(r.dataset.rung) <= used); });
    count.textContent = "Hints used: " + used + " of 4";
    next.disabled = used >= 4;
    tell.disabled = used >= 4;
  }

  next.addEventListener("click", function () {
    if (used >= 4) return;
    if (used === 0) turn("student", "I'm stuck. Can I have a hint?");
    var h = HINTS[used];
    used++;
    turn("tutor", h.tutor, h.tag);
    if (h.student) turn("student", h.student);
    update();
  });

  tell.addEventListener("click", function () {
    if (used >= 4) return;
    turn("student", "Just tell me.");
    used = 4;
    turn("tutor", HINTS[3].tutor, "Answer");
    update();
  });

  reset.addEventListener("click", function () {
    turns.innerHTML = initial;
    used = 0;
    update();
  });

  update();
})();
