// E4 Vision-language models.
//   pipeFig    SIGNATURE: a photo and a question step through a small open model (Qwen3-VL-2B-Instruct): patches, encoder, projector, prompt, answer, with counts
//   designFig  two designs as two diagrams that light up step by step: image tokens in the sequence (LLaVA) and cross-attention layers (Flamingo)
//   trainFig   training stages: which parts learn, what the loss counts
//   tokFig     image tokens against photo size for four real processors, and the formula
//   failFig    gallery of failure probes with the model's outputs: counting, small text, names and facts, murky water
//   adaptFig   what may change at test time, and a stream of frames read in chunks (bridge to In-Place TTT)
// Pure helpers are exported with module.exports for the node test.
(function () {
  "use strict";

  // ---------- pure helpers ----------
  function roundHalfEven(x) { var f = Math.floor(x), d = x - f; if (d > 0.5) return f + 1; if (d < 0.5) return f; return f % 2 === 0 ? f : f + 1; }
  // the resize rule of the Qwen2-VL family (smart_resize): sides become multiples of factor, and the area stays between min and max
  function smartResize(h, w, factor, minPx, maxPx) {
    var hh = Math.max(factor, roundHalfEven(h / factor) * factor), ww = Math.max(factor, roundHalfEven(w / factor) * factor), beta;
    if (hh * ww > maxPx) { beta = Math.sqrt(h * w / maxPx); hh = Math.floor(h / beta / factor) * factor; ww = Math.floor(w / beta / factor) * factor; }
    else if (hh * ww < minPx) { beta = Math.sqrt(minPx / (h * w)); hh = Math.ceil(h * beta / factor) * factor; ww = Math.ceil(w * beta / factor) * factor; }
    return { h: hh, w: ww };
  }
  function qwenTokens(w, h) { var r = smartResize(h, w, 32, 65536, 16777216); return { tokens: (r.h / 32) * (r.w / 32), h: r.h, w: r.w, patches: (r.h / 16) * (r.w / 16) }; }
  function mergeShapes(rows, cols, dv, dlm) { return { patches: rows * cols, tokens: (rows / 2) * (cols / 2), merged: 4 * dv, out: dlm }; }
  function kvBytesPerToken(layers, kvHeads, headDim, bytes) { return 2 * layers * kvHeads * headDim * bytes; }
  function mlpParams(din, dh, dout, lnDim) { return lnDim * 2 + din * dh + dh + dh * dout + dout; }
  if (typeof module !== "undefined") { module.exports = { roundHalfEven: roundHalfEven, smartResize: smartResize, qwenTokens: qwenTokens, mergeShapes: mergeShapes, kvBytesPerToken: kvBytesPerToken, mlpParams: mlpParams }; return; }

  // ---------- page helpers ----------
  var D = {}, IMG = {}, CREDITS = {};
  var K_RGB = [0.54, 0.256, 0.209], BETA = 0.06, VEIL = [0.08, 0.50, 0.55], EXPO = 0.6, LIN = new Float32Array(256);
  (function () { for (var v = 0; v < 256; v++) { var x = v / 255; LIN[v] = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); } })();
  function toSrgb(x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return 255 * (x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055); }
  function degradeData(src, dst, z) {      // the depth model of depth.py (as in C3), on 0..255 RGBA arrays
    var k = K_RGB, kmin = Math.min(k[0], k[1], k[2]), kmean = (k[0] + k[1] + k[2]) / 3, V = 1 - Math.exp(-BETA * z), dark = Math.exp(-kmin * z), gain = Math.exp(EXPO * kmean * z);
    var mult = [0, 1, 2].map(function (c) { return Math.exp(-(k[c] - kmin) * z) * dark * gain; }), veilV = [0, 1, 2].map(function (c) { return VEIL[c] * V; });
    for (var i = 0; i < src.length; i += 4) { for (var c = 0; c < 3; c++) dst[i + c] = Math.round(toSrgb((LIN[src[i + c]] * (1 - V) + veilV[c]) * mult[c])); dst[i + 3] = 255; }
  }
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function nf(v, d) { return Kit.fmt(v, d == null ? 2 : d); }
  function pc(v, d) { return (100 * v).toFixed(d == null ? 0 : d) + "%"; }
  function comma(n) { return String(Math.round(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ","); }
  function big(v) { return v >= 1000 ? comma(v) : nf(v, 1); }
  function hexRGB(h) {
    h = (h || "#888888").trim();
    if (h.charAt(0) === "#") { if (h.length === 4) h = "#" + h[1] + h[1] + h[2] + h[2] + h[3] + h[3]; return [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]; }
    var m = h.match(/\d+/g) || [136, 136, 136]; return [+m[0], +m[1], +m[2]];
  }
  function rgba(c, a) { return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")"; }
  function loadImg(src) { return new Promise(function (ok, fail) { var im = new Image(); im.onload = function () { ok(im); }; im.onerror = function () { fail(new Error("image " + src)); }; im.src = src; }); }
  function fitCanvas(cv, w, h) {
    var d = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(w * d); cv.height = Math.round(h * d); cv.style.width = w + "px"; cv.style.height = h + "px";
    var ctx = cv.getContext("2d"); ctx.setTransform(d, 0, 0, d, 0, 0); return ctx;
  }
  function setPressed(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }
  function onResize(fn) { window.addEventListener("resize", Kit.debounce(fn, 120)); Kit.onTheme(fn); }
  function nameOf(k) { return { "hammerhead-1": "Hammerhead, open water", "bamboo-1": "Bamboo shark, tank", "spadenose-1": "Spadenose, white board", "milk-1": "Milk sharks, rocks", "turtle-1": "Turtle, beach", "dolphin-1": "Dolphin", "reeffish-1": "Clownfish", "ray-1": "Eagle ray" }[k] || k; }
  var NEED = ["hammerhead-1", "bamboo-1", "spadenose-1", "milk-1", "turtle-1", "dolphin-1", "reeffish-1", "ray-1"];
  var FACT = {
    "hammerhead-1": { photo: "scalloped hammerhead (Sphyrna lewini)", iucn: "Critically Endangered (2018)", extra: "Indian law: the scalloped hammerhead is in Appendix II of Schedule IV of the Wild Life (Protection) Act as amended in 2022. That is the list of species whose international trade is regulated (Gazette text of the Act; ICAR-CMFRI poster 69, 2025). The great hammerhead that the model named is on Schedule II." },
    "bamboo-1": { photo: "grey bamboo shark (Chiloscyllium griseum)", iucn: "Vulnerable (2020)", extra: "" },
    "spadenose-1": { photo: "spadenose shark (Scoliodon laticaudus)", iucn: "Near Threatened (2020)", extra: "" },
    "milk-1": { photo: "milk shark (Rhizoprionodon acutus)", iucn: "Vulnerable (2020)", extra: "" }
  };

  document.addEventListener("DOMContentLoaded", function () {
    var jobs = [Kit.data("data/vlm.json"), Kit.data("data/vlm2.json"), Kit.data("data/tokens.json"), Kit.data("data/facts.json"), Kit.data("data/murky2.json"), Kit.data("../data/sharks/photos.json"), Kit.data("data/tags.json")];
    NEED.forEach(function (k) { jobs.push(loadImg("../data/sharks/photos/" + k + ".jpg").then(function (im) { IMG[k] = im; })); });
    Promise.all(jobs).then(function (r) {
      D.v = r[0]; D.v2 = r[1]; D.tok = r[2]; D.f = r[3]; D.m2 = r[4]; D.tags = r[6];
      r[5].photos.forEach(function (p) { CREDITS[p.file.replace("photos/", "").replace(".jpg", "")] = p.credit_line; });
      pipeFig(); designFig(); trainFig(); tokFig(); failFig(); adaptFig();
    }).catch(function (e) { console.error(e); });
  });
  function meanColours(img, cols, rows) {
    var c = document.createElement("canvas"); c.width = cols; c.height = rows; var x = c.getContext("2d", { willReadFrequently: true });
    x.drawImage(img, 0, 0, cols, rows); return x.getImageData(0, 0, cols, rows).data;
  }
  function creditLine(k) { return CREDITS[k] || ""; }

  // ---------- 1. photo to answer ----------
  var PIPE_NAMES = ["Photo and question", "Cut into patches", "Vision encoder", "Projector", "The prompt", "The answer"];
  function pipeFig() {
    var st = { ph: 0, step: 0, typed: 0, playing: false, timer: 0, stopTyping: null };
    var P = D.v.pipeline, keys = ["hammerhead-1", "bamboo-1", "spadenose-1", "milk-1"], cards = Kit.$("pp-cards"), status = Kit.$("pp-status"), cvs = [], ns = [], textBox;
    Kit.chips(Kit.$("pp-bar"), keys.map(nameOf), 0, function (i) { st.ph = i; reset(); });
    var btns = Kit.$("pp-btns");
    Kit.button(btns, "Back", function () { stop(); go(st.step - 1); });
    var nextB = Kit.button(btns, "Next step", function () { stop(); go(st.step + 1); });
    var playB = Kit.button(btns, "Play", function () { if (st.playing) { stop(); return; } st.playing = true; playB.textContent = "Pause"; if (st.step >= 5) go(0); tick(); });
    Kit.button(btns, "Reset", function () { stop(); reset(); });
    function stop() { st.playing = false; playB.textContent = "Play"; clearTimeout(st.timer); }
    function tick() { st.timer = setTimeout(function () { if (!st.playing) return; if (st.step >= 5) { stop(); return; } go(st.step + 1); tick(); }, st.step === 4 ? 2600 : 2000); }
    PIPE_NAMES.forEach(function (n, i) {
      var c = Kit.html("div", { class: "pp-card", id: "pp-c" + i }, cards);
      Kit.html("div", { class: "pp-h" }, c).innerHTML = "<span>" + (i + 1) + "</span> " + n;
      if (i === 5) { textBox = Kit.html("div", { class: "pp-text", "aria-live": "off" }, c); cvs.push(null); }
      else cvs.push(Kit.html("canvas", {}, c));
      ns.push(Kit.html("div", { class: "pp-n" }, c));
    });
    function reset() { st.step = 0; st.typed = 0; if (st.stopTyping) st.stopTyping(); render(); }
    function go(s) {
      s = Math.max(0, Math.min(5, s)); var was = st.step; st.step = s;
      if (st.stopTyping) st.stopTyping();
      if (s === 5) { st.typed = 0; var L = P[keys[st.ph]].a.length; st.stopTyping = Kit.animate(Math.min(9000, 900 + L * 12), function (p) { st.typed = Math.round(p * L); updateAnswer(); }); }
      else st.typed = 0;
      render();
    }
    function sizes() { var w = Kit.width(cvs[0].parentNode) - 24; return Math.max(110, Math.min(220, w)); }
    function updateAnswer() { var r = P[keys[st.ph]], t = r.a.slice(0, st.typed); textBox.textContent = t + (st.typed < r.a.length ? "▌" : ""); var tk = Math.round(r.n_ans * st.typed / r.a.length); ns[5].innerHTML = "<b>" + tk + " of " + r.n_ans + "</b> answer tokens, one at a time" + (r.finished ? "" : "; stopped at the limit"); }
    function render() {
      var Pl = Kit.palette(), key = keys[st.ph], r = P[key], im = IMG[key], w = sizes(), i, s = st.step;
      for (i = 0; i < 6; i++) { var c = Kit.$("pp-c" + i); c.classList.toggle("on", i === s); c.classList.toggle("off", i > s); }
      var thH = 92;
      // 1 photo
      var ctx = fitCanvas(cvs[0], w, thH), ar = im.naturalWidth / im.naturalHeight, dw = Math.min(w, thH * ar), dh = dw / ar; ctx.drawImage(im, (w - dw) / 2, (thH - dh) / 2, dw, dh);
      ns[0].innerHTML = "<b>" + r.size[0] + " &times; " + r.size[1] + "</b> pixels<br>question: " + r.n_question + " tokens";
      // 2 patches: resized to multiples of 32 and cut into 16 pixel patches
      var rs = qwenTokens(r.size[0], r.size[1]), gw = rs.w / 16, gh = rs.h / 16; ctx = fitCanvas(cvs[1], w, thH); var cw = w / gw, ch = Math.min(thH / gh, cw * 0.9), ph2 = ch * gh, oy = (thH - ph2) / 2;
      ctx.drawImage(im, 0, oy, w, ph2); ctx.strokeStyle = "rgba(255,255,255,0.45)"; ctx.lineWidth = 0.5; ctx.beginPath();
      for (i = 0; i <= gw; i++) { ctx.moveTo(i * cw, oy); ctx.lineTo(i * cw, oy + ph2); } for (i = 0; i <= gh; i++) { ctx.moveTo(0, oy + i * ch); ctx.lineTo(w, oy + i * ch); } ctx.stroke();
      ns[1].innerHTML = "<b>" + rs.w + " &times; " + rs.h + "</b> after the resize<br><b>" + gh + " &times; " + gw + " = " + comma(r.patches) + "</b> patches of 16 &times; 16<br>1,536 numbers each";
      // 3 encoder: strip of patch colours over 24 blocks
      var col = meanColours(im, gw, gh); ctx = fitCanvas(cvs[2], w, thH); var n = gw * gh, sw = w / n, b;
      for (i = 0; i < n; i++) { ctx.fillStyle = "rgb(" + col[i * 4] + "," + col[i * 4 + 1] + "," + col[i * 4 + 2] + ")"; ctx.fillRect(i * sw, 0, Math.ceil(sw) + 0.5, 14); }
      for (b = 0; b < 24; b++) { ctx.fillStyle = rgba(hexRGB(Pl.navy), 0.35 + 0.65 * (b / 23)); ctx.fillRect(w * 0.06, 20 + b * 3, w * 0.88, 2.4); }
      ctx.fillStyle = Pl.ink2; ctx.font = "12px Spectral, Georgia, serif"; ctx.fillText("blocks: 24, width 1,024", w * 0.06, 20 + 24 * 3 + 12);
      ns[2].innerHTML = "<b>" + comma(r.patches) + " &times; 1,024</b> numbers out<br>mean length of a token " + comma(r.raw_len);
      // 4 merge: tokens of 2 by 2 patches
      var mc = meanColours(im, gw / 2, gh / 2), mw = gw / 2, mh = gh / 2; ctx = fitCanvas(cvs[3], w, thH); var cw2 = w / mw, ch2 = Math.min(thH / mh, cw2 * 0.9), oy2 = (thH - ch2 * mh) / 2;
      for (i = 0; i < mw * mh; i++) { ctx.fillStyle = "rgb(" + mc[i * 4] + "," + mc[i * 4 + 1] + "," + mc[i * 4 + 2] + ")"; ctx.fillRect((i % mw) * cw2, oy2 + Math.floor(i / mw) * ch2, cw2 - 0.6, ch2 - 0.6); }
      ns[3].innerHTML = "<b>" + comma(r.patches) + " &times; 1,024</b> &rarr; 4 patches joined: <b>" + r.n_img + " &times; 4,096</b> &rarr; two layers &rarr; <b>" + r.n_img + " &times; 2,048</b>";
      // 5 prompt strip
      ctx = fitCanvas(cvs[4], w, thH); var tot = r.n_prompt, segs = [["template", 3, Pl.sand], ["image", r.n_img, Pl.navy], ["question", r.n_question, Pl.coral], ["template", r.n_other - 3, Pl.sand]], x = 0;
      segs.forEach(function (sg) { var ww = Math.max(2, sg[1] / tot * w); ctx.fillStyle = sg[2]; ctx.fillRect(x, 14, ww, 22); x += ww; });
      ctx.fillStyle = Pl.ink2; ctx.font = "12px Spectral, Georgia, serif"; ctx.fillText("image " + r.n_img + " tokens", w * 0.18, 56); ctx.fillStyle = Pl.coral; ctx.fillText("question " + r.n_question, w * 0.18, 72); ctx.fillStyle = Pl.ink2; ctx.fillText("template " + r.n_other, w * 0.18, 88);
      ns[4].innerHTML = "<b>" + r.n_prompt + "</b> tokens in all<br>the photo costs " + nf(r.n_img / r.n_question, 0) + " times the question";
      if (s >= 5) updateAnswer(); else { textBox.textContent = ""; ns[5].innerHTML = "<b>" + r.n_ans + "</b> tokens will be written"; }
      var msgs = [
        "The photo and the question go in. The question is " + r.n_question + " tokens long, and the prompt gets a few special tokens around it (template).",
        "The processor resizes the photo so that both sides are multiples of 32, and cuts it into patches of 16 by 16 pixels (E1). A patch is flattened to 1,536 numbers: 3 colours, 2 frames (a photo is copied to make 2) and 256 pixels.",
        "A vision transformer with 24 blocks (E1) turns the " + comma(r.patches) + " patches into " + comma(r.patches) + " vectors of 1,024 numbers.",
        "The projector joins 2 by 2 neighbouring patches into one vector of 4,096 numbers and passes it through two layers, to the width of the language model, 2,048. " + r.n_img + " image tokens come out.",
        "The language model reads the prompt: " + r.n_img + " image tokens, " + r.n_question + " question tokens and " + r.n_other + " template tokens. Image tokens are the same kind of vector as word tokens, and attention treats them alike.",
        "The model writes the answer one token at a time, " + r.n_ans + " tokens here. The text below is model output (greedy decoding), not written by a person."
      ];
      status.textContent = "Step " + (s + 1) + " of 6. " + msgs[s];
      var eq = Kit.$("pp-eq"), kv = D.v.params;
      var boxes = [
        [ebox("c-navy", r.size[0] + " &times; " + r.size[1], "pixels"), ebox("c-coral", String(r.n_question), "question tokens")],
        [ebox("c-navy", rs.w + " &times; " + rs.h, "after the resize"), eop("&divide;16 &rarr;"), ebox("c-coral", gh + " &times; " + gw, "patches"), eop("="), ebox("c-teal", comma(r.patches), "patches")],
        [ebox("c-navy", comma(r.patches) + " &times; 1,536", "numbers in"), eop("&rarr;"), ebox("c-teal", comma(r.patches) + " &times; 1,024", "numbers out"), eop("|"), ebox("c-sand", comma(kv.vision_blocks), "numbers in the 24 blocks")],
        [ebox("c-navy", comma(r.patches) + " &times; 1,024", "from the encoder"), eop("&rarr;"), ebox("c-coral", r.n_img + " &times; 4,096", "patches joined, 2 by 2"), eop("&rarr;"), ebox("c-teal", r.n_img + " &times; 2,048", "image tokens"), eop("|"), ebox("c-sand", comma(kv.merger), "numbers in the projector"), eop("|"), ebox("c-navy", nf(r.merged_len, 1) + " against " + nf(r.word_len, 2), "mean length: image token, word token")],
        [ebox("c-navy", String(r.n_img), "image tokens"), eop("+"), ebox("c-coral", String(r.n_question), "question"), eop("+"), ebox("c-sand", String(r.n_other), "template"), eop("="), ebox("c-teal", String(r.n_prompt), "prompt tokens"), eop("|"), ebox("c-sand", nf(r.n_prompt * 114688 / 1e6, 1) + " MB", "key-value cache of the prompt, 16 bits")],
        [ebox("c-teal", String(r.n_ans), "answer tokens"), eop("|"), ebox("c-navy", String(r.n_prompt + r.n_ans), "tokens in all"), eop("|"), ebox("c-sand", pc(r.n_img / (r.n_prompt + r.n_ans), 0), "of them are the photo")]
      ];
      eq.innerHTML = boxes[s].join("");
      Kit.$("pp-credit").textContent = creditLine(key);
      renderCheck();
    }
    function renderCheck() {
      var key = keys[st.ph], f = FACT[key], r = P[key], box = Kit.$("pp-check"), s = st.step;
      box.hidden = s < 5;
      var name = { "hammerhead-1": ["A hammerhead shark: right. Species named: great hammerhead (Sphyrna mokarran): the photo shows a scalloped hammerhead.", "IUCN category given: Vulnerable. For the scalloped hammerhead of the photo it is Critically Endangered (2018), and the great hammerhead is Critically Endangered too.", "Laws named: the US Endangered Species Act and the Marine Mammal Protection Act. NOAA Fisheries lists four populations of the scalloped hammerhead under the Endangered Species Act (two endangered, two threatened). For the shark in the photo that claim is partly right, although it concerns US law. The great hammerhead that the model named is not listed under it. The second law covers marine mammals such as whales, seals and manatees, and sharks are fish (NOAA), so that claim is wrong.", "Not mentioned: Indian law, which the question did not name. " + f.extra],
        "bamboo-1": ["Species named: banded shark (Mustelus mustelus). The photo shows a " + f.photo + ". Wrong.", "The answer says the shark is \"not protected\" in most countries. The IUCN lists the grey bamboo shark as " + f.iucn + ". Risk of extinction and legal protection are different things, so the two statements do not check one another."],
        "spadenose-1": ["Species named: Cape shark (Carcharhinus harriettae). The photo shows a " + f.photo + ". I could not match the name to a species. Wrong.", "The answer says \"not protected\". The IUCN lists the spadenose shark as " + f.iucn + ". The answer does not say which law it means."],
        "milk-1": ["Species named: broadnose shark (Rhizoprionodon porosus). The photo shows a " + f.photo + ". Wrong species, same genus.", "The answer says the Marine Mammal Protection Act protects \"marine mammals, including sharks\". Sharks are fish, and the Act covers whales, seals, manatees and other marine mammals (NOAA)."] }[key];
      box.innerHTML = "<p class=\"pane-h\">Checking the answer against what is known (the photo's species is from its Wikimedia description)</p><ul>" + name.map(function (t) { return "<li>" + t + "</li>"; }).join("") + "</ul>";
    }
    reset(); onResize(render);
  }

  // ---------- 2. two designs ----------
  var DESIGN_STEPS = [
    "Step 1. Look. In both designs a vision encoder (E1) turns the photo into one vector per patch. The language model has not been used yet.",
    "Step 2. Connect. LLaVA multiplies the patch vectors by a matrix (a small MLP in LLaVA-1.5), which gives one vector of the language model's width per patch. Flamingo squeezes the patch vectors into 64 tokens with a resampler.",
    "Step 3. Enter the language model. In LLaVA the image tokens join the text tokens in one sequence, and self-attention (D4) reads all of it. In Flamingo the text stays alone, and extra cross-attention layers (E3) read the 64 tokens.",
    "Step 4. Train. In LLaVA the connector learns first, then the language model joins it. In Flamingo the encoder and the language model stay frozen, and only the resampler and the cross-attention layers learn."
  ];
  function designFig() {
    var st = { step: 0, playing: false, timer: 0 }, svgA = Kit.$("ds-a"), svgB = Kit.$("ds-b"), status = Kit.$("ds-status");
    var btns = Kit.$("ds-btns");
    Kit.button(btns, "Back", function () { stop(); go(st.step - 1); });
    Kit.button(btns, "Next step", function () { stop(); go((st.step + 1) % 4); });
    var playB = Kit.button(btns, "Play", function () { if (st.playing) { stop(); return; } st.playing = true; playB.textContent = "Pause"; if (st.step >= 3) go(0); tick(); });
    function stop() { st.playing = false; playB.textContent = "Play"; clearTimeout(st.timer); }
    function tick() { st.timer = setTimeout(function () { if (!st.playing) return; if (st.step >= 3) { stop(); return; } go(st.step + 1); tick(); }, 2600); }
    function go(s) { st.step = Math.max(0, Math.min(3, s)); draw(); }
    function box(svg, P, x, y, w, h, label, at, cls, sub) {
      var on = st.step >= at, now = st.step === at, g = Kit.el("g", { opacity: on ? 1 : 0.22 }, svg), col = cls === "coral" ? P.coral : cls === "teal" ? P.teal : cls === "sand" ? P.sand : cls === "navy" ? P.navy : P.rule;
      Kit.el("rect", { x: x, y: y, width: w, height: h, rx: 6, fill: rgba(hexRGB(col), cls ? 0.14 : 0.06), stroke: now ? P.coral : col, "stroke-width": now ? 2.6 : 1.4 }, g);
      var t = Kit.el("text", { x: x + w / 2, y: y + h / 2 + (sub ? -3 : 5), "text-anchor": "middle", fill: P.ink, "font-size": 13.5, "font-family": "Spectral, Georgia, serif" }, g); t.textContent = label;
      if (sub) { var t2 = Kit.el("text", { x: x + w / 2, y: y + h / 2 + 13, "text-anchor": "middle", fill: P.ink2, "font-size": 12, "font-family": "Spectral, Georgia, serif" }, g); t2.textContent = sub; }
      return g;
    }
    function arrow(svg, P, x1, y1, x2, y2, at, col) {
      var on = st.step >= at; Kit.el("line", { x1: x1, y1: y1, x2: x2, y2: y2, stroke: col || P.ink2, "stroke-width": 1.8, opacity: on ? 1 : 0.22, "marker-end": "url(#ah" + (svg.id) + ")" }, svg);
    }
    function label(svg, P, x, y, txt, at, col, anchor) { var t = Kit.el("text", { x: x, y: y, fill: col || P.ink2, "font-size": 12.5, "text-anchor": anchor || "middle", "font-family": "Spectral, Georgia, serif", opacity: st.step >= at ? 1 : 0.22 }, svg); t.textContent = txt; }
    function defs(svg, P) { var d = Kit.el("defs", {}, svg), m = Kit.el("marker", { id: "ah" + svg.id, viewBox: "0 0 10 10", refX: 8, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto" }, d); Kit.el("path", { d: "M0,0 L10,5 L0,10 z", fill: P.ink2 }, m); }
    function tokens(svg, P, x, y, n, cls, at, fill) { var on = st.step >= at; for (var i = 0; i < n; i++) Kit.el("rect", { x: x + i * 13, y: y, width: 10, height: 14, rx: 2, fill: fill, opacity: on ? 0.9 : 0.22 }, svg); }
    function draw() {
      var P = Kit.palette(), W = 330, H = 470, now;
      [svgA, svgB].forEach(function (s) { s.innerHTML = ""; s.setAttribute("viewBox", "0 0 " + W + " " + H); defs(s, P); });
      // A: LLaVA-style
      var s = svgA;
      box(s, P, 105, 8, 120, 34, "Photo", 0); arrow(s, P, 165, 42, 165, 62, 0);
      box(s, P, 65, 62, 200, 44, "Vision encoder", 0, "navy", "one vector per patch"); arrow(s, P, 165, 106, 165, 130, 1);
      box(s, P, 65, 130, 200, 44, "Projector (matrix product)", 1, "coral", "patch vector to token vector"); arrow(s, P, 165, 174, 165, 204, 2);
      tokens(s, P, 38, 208, 8, "navy", 2, P.navy); tokens(s, P, 38 + 8 * 13 + 10, 208, 6, "coral", 2, P.coral);
      label(s, P, 38 + 4 * 13, 240, "image tokens", 2, P.navy); label(s, P, 38 + 8 * 13 + 10 + 3 * 13, 240, "text tokens", 2, P.coral);
      arrow(s, P, 165, 246, 165, 270, 2);
      box(s, P, 40, 270, 250, 130, "Language model", 2, "teal");
      label(s, P, 165, 296, "self-attention over the whole sequence", 2, P.ink); arrow(s, P, 165, 400, 165, 424, 2);
      box(s, P, 90, 424, 150, 32, "answer tokens", 2);
      label(s, P, 165, 463, st.step >= 3 ? "stage 1: projector learns. stage 2: projector and language model" : "", 3, P.coral);
      // B: Flamingo-style
      s = svgB;
      box(s, P, 105, 8, 120, 34, "Photo", 0); arrow(s, P, 165, 42, 165, 62, 0);
      box(s, P, 65, 62, 200, 44, "Vision encoder", 0, "navy", "one vector per patch"); arrow(s, P, 165, 106, 165, 130, 1);
      box(s, P, 65, 130, 200, 44, "Resampler", 1, "coral", "64 tokens");
      // language model column on the right, cross-attention boxes joined by arrows
      var ys = [214, 262, 310, 358];
      ys.forEach(function (y, k) {
        var x = 150, isX = k % 2 === 0; box(s, P, x + 20, y, 150, 34, isX ? "gated cross-attention" : "language model layer", 2, isX ? "teal" : "sand");
      });
      label(s, P, 60, 184, "64 tokens", 2, P.coral, "start");
      arrow(s, P, 165, 174, 165, 190, 2); Kit.el("line", { x1: 165, y1: 190, x2: 40, y2: 190, stroke: P.ink2, "stroke-width": 1.8, opacity: st.step >= 2 ? 1 : 0.22 }, s); Kit.el("line", { x1: 40, y1: 190, x2: 40, y2: 380, stroke: P.ink2, "stroke-width": 1.8, opacity: st.step >= 2 ? 1 : 0.22 }, s);
      [0, 2].forEach(function (k) { arrow(s, P, 40, ys[k] + 17, 168, ys[k] + 17, 2); });
      
      arrow(s, P, 245, 396, 245, 424, 2);
      box(s, P, 170, 424, 150, 32, "answer tokens", 2);
      label(s, P, 165, 463, st.step >= 3 ? "learn: resampler and cross-attention. frozen: the rest" : "", 3, P.coral);
      if (st.step >= 3) {
        [[svgA, 272, 84, "frozen"], [svgA, 272, 152, "learns"], [svgB, 272, 84, "frozen"], [svgB, 272, 152, "learns"]].forEach(function (b) { var t = Kit.el("text", { x: b[1], y: b[2], fill: b[3] === "frozen" ? P.muted : P.coral, "font-size": 12.5, "font-weight": 600, "font-family": "Spectral, Georgia, serif" }, b[0]); t.textContent = b[3]; });
        var tl = Kit.el("text", { x: 165, y: 376, "text-anchor": "middle", fill: P.coral, "font-size": 12.5, "font-weight": 600, "font-family": "Spectral, Georgia, serif" }, svgA); tl.textContent = "stage 1 frozen, stage 2 learns";
        [214, 310].forEach(function (y) { var t = Kit.el("text", { x: 176, y: y - 3, fill: P.coral, "font-size": 12, "font-weight": 600, "font-family": "Spectral, Georgia, serif" }, svgB); t.textContent = "learns"; });
        [262, 358].forEach(function (y) { var t = Kit.el("text", { x: 176, y: y - 3, fill: P.muted, "font-size": 12, "font-weight": 600, "font-family": "Spectral, Georgia, serif" }, svgB); t.textContent = "frozen"; });
      }
      status.textContent = DESIGN_STEPS[st.step];
    }
    draw(); onResize(draw);
  }

  // ---------- 3. training stages ----------
  function trainFig() {
    var st = { recipe: 0, stage: 0 };
    var RECIPES = [
      { name: "LLaVA (2023)", stages: [{ title: "Stage 1: align", train: { enc: 0, con: 1, lm: 0 }, data: "Data: 595,000 image and caption pairs (a filtered part of CC3M), turned into a request to describe the image briefly with the caption as the answer", loss: "the caption" }, { title: "Stage 2: instruction-tune", train: { enc: 0, con: 1, lm: 1 }, data: "Data: 158,000 language-image instruction examples: conversations, detailed descriptions and reasoning questions, written with the help of GPT-4", loss: "the answer" }] },
      { name: "LLaVA-1.5 (2023)", stages: [{ title: "Stage 1: align", train: { enc: 0, con: 1, lm: 0 }, data: "Data: 558,000 image and caption pairs", loss: "the caption" }, { title: "Stage 2: instruction-tune", train: { enc: 0, con: 1, lm: 1 }, data: "Data: 665,000 examples, with short-answer questions and a two-layer MLP projector in place of one matrix", loss: "the answer" }] }
    ];
    Kit.chips(Kit.$("tr-bar"), RECIPES.map(function (r) { return r.name; }), 0, function (i) { st.recipe = i; draw(); });
    Kit.chips(Kit.$("tr-bar2"), ["Stage 1: align", "Stage 2: instruction-tune"], 0, function (i) { st.stage = i; draw(); });
    function draw() {
      var P = Kit.palette(), R = RECIPES[st.recipe], S = R.stages[st.stage], pr = D.v.params, f = D.f, conn = pr.merger + pr.deepstack_mergers, tot = pr.total;
      var parts = [["Vision encoder", "enc", pr.vision_tower - pr.merger - pr.deepstack_mergers], ["Connector", "con", conn], ["Language model", "lm", pr.language_model]];
      var box = Kit.$("tr-parts"); box.innerHTML = "";
      var learn = 0;
      parts.forEach(function (p) {
        var on = S.train[p[1]], c = Kit.html("div", { class: "tr-part " + (on ? "learn" : "frozen") }, box);
        Kit.html("b", {}, c, p[0]); Kit.html("span", {}, c, on ? "learns" : "frozen"); Kit.html("i", {}, c, comma(p[2]) + " numbers");
        if (on) learn += p[2];
      });
      var strip = Kit.$("tr-strip"); strip.innerHTML = "";
      var r = D.v.pipeline["hammerhead-1"], nImg = r.n_img, nQ = r.n_question + r.n_other, nA = st.stage === 0 ? 12 : r.n_ans, tt = nImg + nQ + nA;
      [["image tokens", nImg, 0, "c-sand"], ["prompt", nQ, 0, "c-sand"], [S.loss, nA, 1, "c-teal"]].forEach(function (sg) { var sp = Kit.html("span", { class: "tk " + sg[3], style: "width:" + Math.max(3, 100 * sg[1] / tt) + "%" }, strip); sp.textContent = sg[1] > tt * 0.12 ? sg[0] + " " + sg[1] : ""; sp.title = sg[0] + ": " + sg[1] + " tokens" + (sg[2] ? ", in the loss" : ", not in the loss"); });
      Kit.$("tr-data").textContent = S.data + ".";
      Kit.$("tr-eq").innerHTML = ebox("c-coral", pc(learn / tot, 1), "of the numbers of a model of the size of Qwen3-VL-2B learn in this stage") + eop("|") + ebox("c-teal", nA + " of " + tt, "tokens in the loss (a made-up example of the same size)") + eop("|") + ebox("c-sand", comma(learn), "numbers that move");
      Kit.$("tr-status").textContent = S.title + ". " + (st.stage === 0 ? "The projector starts as random numbers, so the image tokens have no meaning for the language model yet. The language model stays as it is." : "The projector keeps learning and the language model joins it, so that it learns to follow questions about photos. The vision encoder stays frozen.");
    }
    draw(); onResize(draw);
  }

  // ---------- 4. token cost ----------
  function tokFig() {
    var st = { w: 640, h: 362 }, sl = Kit.$("tk-controls");
    var cps, sw = Kit.slider(sl, { label: "Photo width, pixels", min: 112, max: 4000, step: 16, value: 640, show: function (v) { return v + " px"; } }, function (v) { st.w = v; unpress(); draw(); });
    var sh = Kit.slider(sl, { label: "Photo height, pixels", min: 112, max: 3000, step: 2, value: 362, show: function (v) { return v + " px"; } }, function (v) { st.h = v; unpress(); draw(); });
    function unpress() { if (cps) cps.forEach(function (b) { b.setAttribute("aria-pressed", "false"); }); }
    cps = Kit.chips(Kit.$("tk-bar"), ["Our photo, 640 × 362", "Phone, 4000 × 3000", "Video frame, 1920 × 1080", "Thumbnail, 224 × 224"], 0, function (i) { var z = [[640, 362], [4000, 3000], [1920, 1080], [224, 224]][i]; st.w = z[0]; st.h = z[1]; sw.set(z[0]); sh.set(z[1]); draw(); });
    // the formula against the processor on the measured sizes
    var ok = D.tok.processors["qwen3-vl"].every(function (r) { return qwenTokens(r.w, r.h).tokens === r.tokens; });
    function curve() {
      var xs = [], ys = [], i; for (i = 112; i <= 4000; i += 64) { xs.push(i); ys.push(qwenTokens(i, Math.round(i * st.h / st.w)).tokens); } return { x: xs, y: ys };
    }
    function draw() {
      var q = qwenTokens(st.w, st.h), Pl = Kit.palette(), r = D.v.pipeline["hammerhead-1"], cv = curve(), nq = r.n_question;
      var names = { "qwen3-vl": "Qwen3-VL (processor)", "llava-1.5": "LLaVA-1.5 (processor)", "llava-next": "LLaVA-NeXT (processor)", "smolvlm": "SmolVLM-256M (processor)" }, cols = [Pl.teal, Pl.navy, Pl.coral, Pl.sand];
      Kit.plot("tk-plot", function (P) {
        var tr = Object.keys(names).map(function (k, i) { var rows = D.tok.processors[k]; return { x: rows.map(function (a) { return a.w * a.h / 1e6; }), y: rows.map(function (a) { return a.tokens; }), type: "scatter", mode: "lines+markers", name: names[k], line: { color: [P.teal, P.navy, P.coral, P.sand][i], width: 2 }, marker: { size: 6 }, hovertemplate: names[k] + ": %{y} tokens for a photo of %{x:.2f} megapixels<extra></extra>" }; });
        tr.push({ x: [st.w * st.h / 1e6], y: [q.tokens], type: "scatter", mode: "markers", name: "your photo (formula)", marker: { size: 13, color: P.coral, symbol: "diamond", line: { color: P.bg, width: 2 } }, hovertemplate: "your photo: %{y} tokens<extra></extra>" });
        return tr;
      }, function () { var narrow = Kit.width(Kit.$("tk-plot")) < 560; Kit.$("tk-plot").style.height = narrow ? "430px" : ""; return { xaxis: { title: { text: "photo area, megapixels" }, type: "log", tickvals: [0.0125, 0.05, 0.2, 0.8, 3.2, 8.3], ticktext: ["0.0125", "0.05", "0.2", "0.8", "3.2", "8.3"] }, yaxis: { title: { text: "image tokens", standoff: 8 }, type: "log", tickvals: [64, 256, 1024, 4096], ticktext: ["64", "256", "1,024", "4,096"], automargin: true }, margin: { l: 64, r: 12, t: 10, b: narrow ? 170 : 60 }, showlegend: true, legend: narrow ? { orientation: "v", yref: "container", y: 0, yanchor: "bottom", x: 0, font: { size: 12 } } : { orientation: "h", y: -0.32, yanchor: "top", font: { size: 12 } } }; });
      var kvB = kvBytesPerToken(28, 8, 128, 2);
      Kit.$("tk-eq").innerHTML =
        ebox("c-navy", q.w + " &times; " + q.h, "after the resize to multiples of 32") + eop("&divide;32 &rarr;") + ebox("c-coral", (q.h / 32) + " &times; " + (q.w / 32), "tokens down and across") + eop("=") + ebox("c-teal", comma(q.tokens), "image tokens") + eop("|") +
        ebox("c-sand", nf(q.tokens / nq, 0) + " &times;", "the question, " + nq + " tokens") + eop("|") + ebox("c-navy", big(q.tokens * kvB / 1e6) + " MB", "key-value cache at 16 bits") + eop("|") + ebox("c-teal", comma(Math.floor(262144 / (q.tokens + 20))), "such photos fit in the 262,144 token window");
      Kit.$("tk-note").textContent = "The formula agrees with the processor on all " + D.tok.processors["qwen3-vl"].length + " measured sizes: " + (ok ? "yes" : "no") + ". A 2,400 frame dive at this size is " + comma(2400 * q.tokens) + " image tokens, and a cache of " + big(2400 * q.tokens * kvB / 1e9) + " GB at 16 bits.";
    }
    draw();
  }

  // ---------- 5. failure gallery ----------
  function failFig() {
    var st = { tab: 0 }, tabs = ["Counting", "Small text", "Names and facts", "Murky water", "What passed"], panes = [];
    var bar = Kit.$("fl-bar"); Kit.chips(bar, tabs, 0, function (i) { st.tab = i; show(); });
    tabs.forEach(function (t, i) { panes.push(Kit.$("fl-p" + i)); });
    function show() { panes.forEach(function (p, i) { p.hidden = i !== st.tab; }); if (st.tab === 0) cntDraw(); if (st.tab === 1) txtDraw(); if (st.tab === 2) nameDraw(); if (st.tab === 3) murkDraw(); if (st.tab === 4) passDraw(); }
    // counting
    var NS = [1, 2, 3, 4, 6, 8, 12, 16, 24], ci = 4;
    Kit.slider(Kit.$("fl-c-ctl"), { label: "Copies of the shark in the picture", min: 0, max: NS.length - 1, step: 1, value: ci, show: function (i) { return NS[i] + " copies"; } }, function (i) { ci = i; cntDraw(); });
    function cntDraw() {
      var N = NS[ci], rows = D.v2.count.filter(function (r) { return r.N === N; }), im = Kit.$("fl-c-img");
      im.src = "data/count-" + N + ".jpg"; im.alt = "A white board with " + N + " copies of one shark pasted on it (made-up composite)";
      Kit.$("fl-c-eq").innerHTML = ebox("c-coral", String(N), "sharks in the picture") + eop("|") + rows.map(function (r, i) { return ebox(r.a === String(N) ? "c-teal" : "c-coral", r.a, "answer, layout " + (i + 1) + (r.a === String(N) ? "" : ", wrong")); }).join(eop(",")) + eop("|") + ebox("c-sand", String(rows[0].n_img), "image tokens");
      Kit.plot("fl-c-plot", function (P) {
        return [{ x: [0, 26], y: [0, 26], type: "scatter", mode: "lines", line: { color: P.muted, dash: "dot", width: 1.5 }, hoverinfo: "skip", name: "right answer" }].concat([0, 1, 2].map(function (k) { var rr = D.v2.count.filter(function (r) { return r.layout === k; }); return { x: rr.map(function (r) { return r.N; }), y: rr.map(function (r) { return +r.a; }), type: "scatter", mode: "markers", name: "layout " + (k + 1), marker: { size: 9 - k * 1.5, color: [P.navy, P.coral, P.teal][k], symbol: ["circle", "square", "diamond"][k] }, hovertemplate: "%{x} copies: answer %{y}<extra></extra>" }; }));
      }, function () { return { xaxis: { title: { text: "sharks in the picture" }, range: [0, 26] }, yaxis: { title: { text: "model's answer" }, range: [0, 32] }, margin: { l: 48, r: 12, t: 10, b: 64 }, showlegend: true, legend: { orientation: "h", y: -0.62, yanchor: "top", font: { size: 12 } } }; });
    }
    // small text
    var EMS = [3, 4, 5, 6, 7, 8, 11, 14, 18, 24, 32, 48], ti = 3, scale = 1, kind = "short", LONG = [6, 8, 11, 14, 18];
    var tchips = Kit.chips(Kit.$("fl-t-bar"), ["Photo as it is (640 px wide)", "Photo at twice the size"], 0, function (i) { scale = i ? 2 : 1; txtDraw(); });
    Kit.chips(Kit.$("fl-t-bar"), ["Short tag", "Long tag with several numbers"], 0, function (i) { kind = i ? "long" : "short"; ti = Math.min(ti, kind === "long" ? LONG.length - 1 : EMS.length - 1); sl.input.max = kind === "long" ? LONG.length - 1 : EMS.length - 1; sl.set(ti); txtDraw(); });
    var sl = Kit.slider(Kit.$("fl-t-ctl"), { label: "Letter size", min: 0, max: EMS.length - 1, step: 1, value: ti, show: function (i) { var em = (kind === "long" ? LONG : EMS)[i]; return "em " + em; } }, function (i) { ti = i; txtDraw(); });
    function findText(em) {
      var all = (D.v.text.filter(function (r) { return r.photo === "hammerhead-1"; })).concat(D.v2.small.filter(function (r) { return r.kind !== "long" && r.photo === "hammerhead-1"; }));
      if (kind === "short") return all.filter(function (r) { return r.em === em && r.scale === scale; })[0];
      return D.v2.small.filter(function (r) { return r.kind === "long" && r.em === em && r.scale === scale; })[0];
    }
    function txtDraw() {
      var list = kind === "long" ? LONG : EMS, em = list[Math.min(ti, list.length - 1)], r = findText(em), im = Kit.$("fl-t-img"), nm = "tag-" + kind + "-" + scale + "-" + em + ".png", dim = D.tags[nm] || [100, 30];
      im.src = "data/" + nm; im.style.width = (dim[0] / scale * 2) + "px"; im.alt = "The tag at letter size " + em + ", as the model saw it";
      var ok = r && r.a.replace(/\s/g, "") === r.truth;
      Kit.$("fl-t-eq").innerHTML = ebox("c-sand", r ? nf(r.cap_height, 1) + " px" : "-", "height of a capital letter, in a photo 640 pixels wide") + eop("|") + ebox("c-navy", r ? String(r.n_img) : "-", "image tokens (one per 32 &times; 32 pixels of the resized photo)") + eop("|") + ebox(ok ? "c-teal" : "c-coral", r ? "\"" + r.a + "\"" : "-", "the model's answer") + eop("|") + ebox("c-teal", r ? r.truth : "-", "the tag says");
      var rows = EMS.map(function (e) { var q = (D.v.text.filter(function (x) { return x.photo === "hammerhead-1" && x.em === e && x.scale === scale; }).concat(D.v2.small.filter(function (x) { return x.kind === "short" && x.em === e && x.scale === scale; })))[0]; return q ? (q.a.replace(/\s/g, "") === "4127" ? "yes" : "no") : "-"; });
      Kit.$("fl-t-tab").textContent = kind === "short" ? "Short tag read right at sizes em " + EMS.filter(function (e, i) { return rows[i] === "yes"; }).join(", ") + (rows.indexOf("no") >= 0 ? ". Wrong at em " + EMS.filter(function (e, i) { return rows[i] === "no"; }).join(", ") + "." : ".") : (function () { var ok = LONG.filter(function (e) { var q = D.v2.small.filter(function (x) { return x.kind === "long" && x.em === e && x.scale === scale; })[0]; return q && q.a.replace(/\s/g, "") === q.truth; }); var bad = LONG.filter(function (e) { return ok.indexOf(e) < 0; }); return "Long tag, the weight 0.281: read right at em " + ok.join(", ") + (bad.length ? ". Wrong at em " + bad.join(", ") + "." : "."); })();
    }
    // names and facts
    var ni = 0, nk = ["hammerhead-1", "bamboo-1", "spadenose-1", "milk-1"];
    Kit.chips(Kit.$("fl-n-bar"), nk.map(nameOf), 0, function (i) { ni = i; nameDraw(); });
    function nameDraw() {
      var k = nk[ni], run = D.v2.sample.runs[k], f = FACT[k], im = Kit.$("fl-n-img");
      im.src = "../data/sharks/photos/" + k + ".jpg"; im.alt = nameOf(k);
      var counts = {}; run.sampled.forEach(function (a) { var t = a.toLowerCase(); counts[t] = (counts[t] || 0) + 1; });
      var list = Object.keys(counts).sort(function (a, b) { return counts[b] - counts[a]; }), box = Kit.$("fl-n-list"); box.innerHTML = "";
      list.forEach(function (a) { var row = Kit.html("div", { class: "fl-row" }, box); Kit.html("span", {}, row, a); var tr = Kit.html("span", { class: "fl-track" }, row); Kit.html("i", { style: "width:" + (100 * counts[a] / 8) + "%" }, tr); Kit.html("b", {}, row, counts[a] + " of 8"); });
      Kit.$("fl-n-eq").innerHTML = ebox("c-teal", f.photo.split(" (")[0], "what the photo shows (Wikimedia description)") + eop("|") + ebox("c-navy", run.greedy, "greedy answer") + eop("|") + ebox("c-coral", String(list.length), "different answers in 8 samples");
      Kit.$("fl-n-credit").textContent = creditLine(k);
    }
    // murky
    var mi = 0, zi = 0, mk = ["hammerhead-1", "bamboo-1", "spadenose-1", "turtle-1", "dolphin-1", "reeffish-1", "ray-1"], ZS = [0, 5, 10, 15, 20, 30];
    Kit.chips(Kit.$("fl-m-bar"), mk.map(nameOf), 0, function (i) { mi = i; murkDraw(); });
    Kit.slider(Kit.$("fl-m-ctl"), { label: "Depth", min: 0, max: ZS.length - 1, step: 1, value: 0, show: function (i) { return ZS[i] + " m"; } }, function (i) { zi = i; murkDraw(); });
    function murkDraw() {
      var k = mk[mi], z = ZS[zi], im = IMG[k], cv = Kit.$("fl-m-cv"), w = Math.min(420, Kit.width(cv.parentNode)), h = Math.round(w * im.naturalHeight / im.naturalWidth), ctx = fitCanvas(cv, w, h);
      var c = document.createElement("canvas"); c.width = im.naturalWidth; c.height = im.naturalHeight; var x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(im, 0, 0);
      var d = x.getImageData(0, 0, c.width, c.height), o = x.createImageData(c.width, c.height); degradeData(d.data, o.data, z); x.putImageData(o, 0, 0); ctx.drawImage(c, 0, 0, w, h);
      var isShark = mi < 3, row = isShark ? D.v.murky.filter(function (r) { return r.photo === k && r.depth === z; })[0] : null, m2 = isShark ? null : D.m2.filter(function (r) { return r.photo === k && r.depth === z; })[0];
      cv.setAttribute("aria-label", nameOf(k) + " at " + z + " metres in the simple depth model");
      if (!isShark && !m2) { Kit.$("fl-m-eq").innerHTML = ebox("c-sand", "not run", "the no-shark photos were tested at 0, 10, 20 and 30 m only; move the depth to one of those") + eop("|") + ebox("c-sand", nameOf(k), "the photo shows"); Kit.$("fl-m-credit").textContent = creditLine(k) + " Made murky with depth.py (a simple model, not a measurement of Mumbai water)."; return; }
      Kit.$("fl-m-eq").innerHTML = isShark ?
        ebox("c-navy", "\"" + row.which.replace(/\.$/, "") + "\"", "answer to: Which shark is this?") + eop("|") + ebox("c-teal", row.yes, "answer to: Is there a shark in this photo?") + eop("|") + ebox("c-sand", FACT[k].photo.split(" (")[0], "the photo shows") :
        ebox("c-teal", m2.a, "answer to: Is there a shark in this photo? (no shark here)") + eop("|") + ebox("c-sand", nameOf(k), "the photo shows");
      Kit.$("fl-m-credit").textContent = creditLine(k) + " Made murky with depth.py (a simple model, not a measurement of Mumbai water).";
    }
    // what passed
    function passDraw() {
      var obj = D.v.objects, right = obj.filter(function (r) { var a = r.a.trim().toLowerCase().indexOf("yes") === 0; return a === r.truth; }).length, cnt = D.v.count.filter(function (r) { return String(r.truth) === r.a.trim(); }).length;
      var cr = D.v2.count.filter(function (r) { return r.N <= 8 && r.a === String(r.N); }).length, cr12 = D.v2.count.filter(function (r) { return r.N >= 12 && r.a === String(r.N); }).length;
      var t = D.m2.filter(function (r) { return r.a.toLowerCase() === "no"; }).length;
      Kit.$("fl-s-eq").innerHTML = ebox("c-teal", cnt + " of " + D.v.count.length, "photos: number of sharks right (0 to 3)") + eop("|") + ebox("c-teal", right + " of " + obj.length, "yes or no: is there a diver, dolphin, turtle, ruler, shark?") + eop("|") + ebox("c-teal", cr + " of 18", "composites with 1 to 8 copies counted right") + eop("|") + ebox("c-coral", cr12 + " of 9", "composites with 12 to 24 copies counted right") + eop("|") + ebox("c-teal", t + " of " + D.m2.length, "no-shark photos in murky water: no false alarm");
    }
    show();
  }

  // ---------- 6. adapting at test time ----------
  function adaptFig() {
    var st = { opt: 3, chunk: 4096, depth: 1 }, pr = D.v.params, f = D.f, tot = pr.total;
    var OPTS = [
      { name: "No change (frozen)", n: 0, note: "No learning after deployment." },
      { name: "Norm scales", n: f.norm_lm + f.norm_vision, note: "The scale numbers of the norm layers. A ViT uses layer norm and this language model uses RMS norm. Both compute their statistics per token, so there are no stored batch statistics to re-estimate as in C3." },
      { name: "Projector", n: pr.merger + pr.deepstack_mergers, note: "The connector between the encoder and the language model." },
      { name: "Fast weights: 5 down-projections", n: 5 * f.down_proj_one, note: "The down projection of the MLP in blocks 0, 6, 12, 18 and 24. This applies the stride 6 of the text-model study of D6 to this 28-block model, as an illustration." },
      { name: "The whole language model", n: pr.language_model, note: "Full fine-tuning of the language side." },
      { name: "Everything", n: tot, note: "All numbers, as in training." }
    ];
    var bar = Kit.$("ad-bar"); Kit.chips(bar, OPTS.map(function (o) { return o.name; }), st.opt, function (i) { st.opt = i; draw(); });
    var ctl = Kit.$("ad-controls");
    Kit.slider(ctl, { label: "Chunk size, tokens", min: 9, max: 14, step: 1, value: 12, show: function (v) { return comma(Math.pow(2, v)); } }, function (v) { st.chunk = Math.pow(2, v); draw(); });
    Kit.slider(ctl, { label: "Frames in the dive", min: 100, max: 2400, step: 100, value: 2400, show: function (v) { return comma(v) + " frames"; } }, function (v) { st.frames = v; draw(); });
    st.frames = 2400;
    function draw() {
      var o = OPTS[st.opt], P = Kit.palette(), tpf = 220, perChunk = st.chunk / tpf, updates = Math.floor(st.frames * tpf / st.chunk);
      var b = Kit.$("ad-share"); b.innerHTML = ""; var span = Kit.html("span", { class: "seg c-teal", style: "width:" + Math.max(0.4, 100 * o.n / tot) + "%" }, b); span.textContent = o.n / tot > 0.12 ? pc(o.n / tot, 0) : "";
      Kit.$("ad-note").textContent = o.note;
      // strip of frames, murkier along the dive
      var cv = Kit.$("ad-cv"), W = Math.min(900, Kit.width(cv.parentNode)), H = 96, ctx = fitCanvas(cv, W, H), nF = Math.max(6, Math.min(16, Math.floor(W / 56))), im = IMG["hammerhead-1"], fw = (W - 6) / nF - 4, fh = fw * im.naturalHeight / im.naturalWidth;
      var c = document.createElement("canvas"); c.width = 160; c.height = Math.round(160 * im.naturalHeight / im.naturalWidth); var x = c.getContext("2d", { willReadFrequently: true }); x.drawImage(im, 0, 0, c.width, c.height);
      var d = x.getImageData(0, 0, c.width, c.height), tmp = document.createElement("canvas"); tmp.width = c.width; tmp.height = c.height; var tx = tmp.getContext("2d");
      for (var i = 0; i < nF; i++) { var z = 30 * i / (nF - 1), o2 = tx.createImageData(c.width, c.height); degradeData(d.data, o2.data, z); tx.putImageData(o2, 0, 0); ctx.drawImage(tmp, 3 + i * (fw + 4), 16, fw, fh); }
      // chunk brackets: a chunk covers perChunk frames, drawn on the strip scaled to the chosen number of frames
      var k, pxPerFrame = (W - 6) / st.frames;
      ctx.strokeStyle = P.coral; ctx.fillStyle = P.coral; ctx.lineWidth = 2;
      for (k = 1; k * perChunk < st.frames; k++) { var xx = 3 + k * perChunk * pxPerFrame; if (xx > W - 3) break; ctx.beginPath(); ctx.moveTo(xx, 14); ctx.lineTo(xx + 3.5, 6); ctx.lineTo(xx - 3.5, 6); ctx.closePath(); ctx.fill(); }
      ctx.fillStyle = P.ink2; ctx.font = "12px Spectral, Georgia, serif"; ctx.fillText("surface", 3, H - 6); ctx.textAlign = "right"; ctx.fillText("the water gets murkier along the dive, down to 30 m", W - 3, H - 6); ctx.textAlign = "left";
      Kit.$("ad-eq").innerHTML = ebox("c-coral", comma(o.n), "numbers that may change") + eop("=") + ebox("c-teal", pc(o.n / tot, 2), "of " + comma(tot)) + eop("|") + ebox("c-navy", nf(perChunk, 1), "frames of " + tpf + " tokens per chunk of " + comma(st.chunk)) + eop("|") + ebox("c-sand", comma(updates), "updates over " + comma(st.frames) + " frames, one per chunk");
    }
    draw(); onResize(draw);
  }
})();
