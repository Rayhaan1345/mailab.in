// F3 SharkCam, end to end. Figures: system diagram with step mode, loop budget, trace of Tag 4127,
// failure gallery, six equations, book map. The made-up records, messages and dive log are labelled where they appear.
// Pure helpers are exported with module.exports so that node can test them.
(function () {
  "use strict";

  // ---------- pure helpers ----------
  var SPECIES = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"];
  var NAMES9 = ["spadenose shark", "milk shark", "grey bamboo shark", "scalloped hammerhead shark", "Ganges shark", "spotted eagle ray", "dolphin", "clownfish", "sea turtle"];
  var SHORT5 = ["Spadenose", "Milk", "Grey bamboo", "Hammerhead", "Ganges"];

  function sum(a) { var s = 0, i; for (i = 0; i < a.length; i++) s += a[i]; return s; }
  function softmax(z) { var m = Math.max.apply(null, z), e = z.map(function (v) { return Math.exp(v - m); }), s = sum(e); return e.map(function (v) { return v / s; }); }
  function argmax(a) { var b = 0, i; for (i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }
  function clipProbs(cosRow, scale) { return softmax(cosRow.map(function (c) { return c * (scale || 100); })); }
  // chances read from the species a message names (idx 0..4), or none (-1): trust for the named one, the rest shared
  function msgProbs(named, trust) {
    if (named < 0) return [0.2, 0.2, 0.2, 0.2, 0.2];
    return SPECIES.map(function (_, i) { return i === named ? trust : (1 - trust) / 4; });
  }
  function fuse(pTab, pClip5, pMsg) {
    var raw = pTab.map(function (p, i) { return p * pClip5[i] * pMsg[i]; }), s = sum(raw);
    if (s <= 0) return { raw: raw, fused: raw.map(function () { return 0.2; }) };
    return { raw: raw, fused: raw.map(function (v) { return v / s; }) };
  }
  // The rule of the trace: accept when the winner's fused chance reaches tau, at least two of the three sources name it,
  // and CLIP puts at least half of its chance on the five sharks. A message at the lowest trust (0.2) gives every species
  // the same chance, so it does not count as a source that names the winner.
  function decide(pTab, pClip9, pMsg, named, tau) {
    var pc5 = pClip9.slice(0, 5), F = fuse(pTab, pc5, pMsg), w = argmax(F.fused), shark = sum(pc5);
    var votes = [argmax(pTab) === w, argmax(pc5) === w, named === w && pMsg[w] > 0.2 + 1e-9], agree = votes.filter(Boolean).length;
    var ok = F.fused[w] >= tau && agree >= 2 && shark >= 0.5;
    return { fused: F.fused, raw: F.raw, win: w, agree: agree, votes: votes, shark: shark, accept: ok, top: F.fused[w] };
  }
  // F2's loop: latency L, update period T (one frame at a time: T = L; pipelined: T = max(camera interval, model time)).
  var ROBOT = { cam: 36.8, dec: 1.0, motor: 60.0, interval: 33.3, speed: 11.4, omega: 19.1, allowed: 5.0 };
  function loop(macs, pipelined) {
    var tm = macs / (ROBOT.speed * 1e9) * 1000, L = ROBOT.cam + tm + ROBOT.dec + ROBOT.motor, T = pipelined ? Math.max(ROBOT.interval, tm) : L;
    return { model: tm, L: L, T: T, err: ROBOT.omega * (L + T) / 1000 };
  }
  function clipLoss(S, T) {
    var n = S.length, lr = 0, lc = 0, i, j;
    for (i = 0; i < n; i++) {
      var row = S[i].map(function (v) { return v / T; }), col = S.map(function (r) { return r[i] / T; });
      lr += -(row[i] - Math.log(sum(row.map(Math.exp)))); lc += -(col[i] - Math.log(sum(col.map(Math.exp))));
    }
    return (lr + lc) / (2 * n);
  }
  // depth.py on 0..255 sRGB pixels (same model as C3 and E2)
  var K_RGB = [0.54, 0.256, 0.209], KAPPA = 0.06, VEIL = [0.08, 0.50, 0.55], EXPO = 0.6;
  var LIN = new Float32Array(256);
  (function () { for (var v = 0; v < 256; v++) { var x = v / 255; LIN[v] = x <= 0.04045 ? x / 12.92 : Math.pow((x + 0.055) / 1.055, 2.4); } })();
  function toSrgb(x) { x = x < 0 ? 0 : x > 1 ? 1 : x; return 255 * (x <= 0.0031308 ? x * 12.92 : 1.055 * Math.pow(x, 1 / 2.4) - 0.055); }
  function degradeData(src, dst, z) {
    var k = K_RGB, kmin = Math.min(k[0], k[1], k[2]), kmean = (k[0] + k[1] + k[2]) / 3;
    var V = 1 - Math.exp(-KAPPA * z), dark = Math.exp(-kmin * z), gain = Math.exp(EXPO * kmean * z);
    var mult = [0, 1, 2].map(function (c) { return Math.exp(-(k[c] - kmin) * z) * dark * gain; });
    var veilV = [0, 1, 2].map(function (c) { return VEIL[c] * V; });
    for (var i = 0; i < src.length; i += 4) {
      for (var c = 0; c < 3; c++) dst[i + c] = Math.round(toSrgb((LIN[src[i + c]] * (1 - V) + veilV[c]) * mult[c]));
      dst[i + 3] = 255;
    }
  }

  if (typeof module !== "undefined") {
    module.exports = { softmax: softmax, clipProbs: clipProbs, msgProbs: msgProbs, fuse: fuse, decide: decide, loop: loop, clipLoss: clipLoss, argmax: argmax, ROBOT: ROBOT, degradeData: degradeData };
    return;
  }

  // ---------- page helpers ----------
  function nf(v, d) { return Kit.fmt(v, d == null ? 2 : d); }
  function pc(v, d) { return (100 * v).toFixed(d == null ? 0 : d) + "%"; }
  function pt(v) { return pc(v, Math.abs(100 * v - Math.round(100 * v)) > 0.001 ? 1 : 0); }
  function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;"); }
  function tex(s, display) {
    if (typeof katex === "undefined") return esc(s);
    try { return katex.renderToString(s, { displayMode: !!display, throwOnError: false }); } catch (e) { return esc(s); }
  }
  function ebox(cls, big, small) { return '<span class="t ' + cls + '"><b>' + big + '</b><i>' + small + '</i></span>'; }
  function eop(s) { return '<span class="op">' + s + '</span>'; }
  function decodeDigit(b64) { var s = atob(b64), a = new Uint8Array(784), i; for (i = 0; i < 784; i++) a[i] = s.charCodeAt(i); return a; }
  function link(href, text) { return '<a href="' + href + '">' + text + '</a>'; }
  function setPressed(bs, i) { bs.forEach(function (b, j) { b.setAttribute("aria-pressed", String(i === j)); }); }

  var D = null, BD = null;      // f3.json and bamboo_depth.json

  // =====================================================================================
  // 1. The system diagram
  // =====================================================================================
  var NODES = {
    dive: { name: "Dive footage", ch: "D6, F2", kind: "in", runs: "robot",
      takes: "Light through the camera lens during a dive.", gives: "One frame a second: 2,400 frames for a 40-minute dive (a made-up log).", model: "None. This is the raw material for the stages below.",
      number: "A dive of 2,400 frames makes 470,400 image tokens at 196 a frame. The same dive written as sentences is 78,163 GPT-2 tokens.", cite: "D6", citeHref: "../d6-long-context/#cost",
      fail: "Cost. Attention over all those tokens grows with the square of their number, so a model cannot simply read the whole dive at once.", href: "../d6-long-context/#cost", hrefText: "D6: what a longer text costs" },
    photo: { name: "Photo", ch: "C1, C3", kind: "in", runs: "shore",
      takes: "A photo from a phone or from the robot.", gives: "Pixels: three grids of numbers, one for red, green and blue. A 224 by 224 photo has 150,528 of them (E1).", model: "None.",
      number: "In the depth model of C3, red light falls to 1% of its surface value at about 8.5 m.", cite: "C3", citeHref: "../c3-domain-shift/#water",
      fail: "Water changes the numbers before any model sees them. The same shark looks different at 10 m.", href: "../c1-convolution/#channels", hrefText: "C1: colour is three grids" },
    tag: { name: "Tag", ch: "C2", kind: "in", runs: "shore",
      takes: "A phone photo of a handwritten tag, cut into one digit at a time.", gives: "A 28 by 28 grid of grey values for a digit.", model: "None.",
      number: "The digit reader of C2 was trained with shifts of up to 2 squares, turns of up to 10 degrees and zoom between 0.9 and 1.1.", cite: "C2", citeHref: "../c2-cnns/#read",
      fail: "A digit cut off-centre or turned far more than 10 degrees is outside what the reader was trained on.", href: "../c2-cnns/#read", hrefText: "C2: reading a handwritten tag" },
    msg: { name: "Message", ch: "D1", kind: "in", runs: "shore",
      takes: "A short report from a fisher or a researcher, in English, Hindi, Marathi or Hinglish.", gives: "Text: a string of characters.", model: "None.",
      number: "The four versions of the Tag 4127 report are 155, 153, 158 and 144 characters long (made-up messages).", cite: "D1", citeHref: "../d1-tokenisation/#languages",
      fail: "Script and spelling vary. Hinglish is written in Latin letters and Hindi in Devanagari, so a tokenizer sees four different inputs for one fact.", href: "../d1-tokenisation/#languages", hrefText: "D1: why some languages cost more" },
    cam: { name: "Robot camera and loop", short: ["Robot camera", "F2"], ch: "F2", kind: "reader", runs: "robot",
      takes: "Frames, about one per 33 ms.", gives: "A picture for the model and a command for the motor.", model: "A small network of the MobileNetV2 size, in 8 bits, on a Raspberry Pi 4.",
      number: "With MobileNetV2 the loop takes 124 ms and the camera points 4.7 degrees behind the shark, where 5 are allowed. ResNet-18 takes 257 ms and gives 9.8 degrees.", cite: "F2", citeHref: "../f2-robots/#loop",
      fail: "A model that is too slow points where the shark was. A model with too few bits changes answers (F2).", href: "../f2-robots/#loop", hrefText: "F2: sense, decide, act" },
    adapt: { name: "Murky-water adaptation", short: ["Murky water", "C3, B7"], ch: "C3, B7", kind: "reader", runs: "both",
      takes: "Frames from water of any clarity.", gives: "Frames, statistics or weights that suit the water now.", model: "Random-water augmentation in training; then batch-norm statistics and a few weights updated while the robot works (test-time training).",
      number: "A digit reader fed 8,000 digits from 0 to 12 m of water: 35.8% with frozen statistics, 92.6% with updated statistics and 97.3% when scale and shift also move.", cite: "C3", citeHref: "../c3-domain-shift/#ttt",
      fail: "It can hurt. In C3, re-estimating statistics on 14 photos changed the answers at 0 m, where the stored statistics were right. Learning a new water can also erase the old one: clear-water accuracy fell from 88% to 73% in B7.", href: "../c3-domain-shift/#adapt", hrefText: "C3: adapting to water you cannot predict" },
    tagreader: { name: "Tag reader", short: ["Tag reader", "C2"], ch: "C2", kind: "reader", runs: "shore",
      takes: "A 28 by 28 picture of one digit.", gives: "Ten chances, one per digit.", model: "A convolutional network with 3,730 numbers: two convolution layers, pooling and one dense layer.",
      number: "Right on 98.89% of the 10,000 test digits. Four digits in a row: 95.6% if errors are independent (computed here).", cite: "C2", citeHref: "../c2-cnns/#read",
      fail: "Shifted digits. A reader trained only on centred digits fell from 98.7% to 87.9% when the test digits moved 3 squares, and to 44.0% at 5 squares.", href: "../c2-cnns/#limits", hrefText: "C2: where the reader fails" },
    msgreader: { name: "Message reader", short: ["Message reader", "D1 to D3, D7"], ch: "D1 to D3, D7", kind: "reader", runs: "shore",
      takes: "Text in English, Hindi, Marathi or Hinglish.", gives: "Fields (tag, length, species, place) and a reply to a question.", model: "A tokenizer (byte-pair encoding, D1) and a language model that predicts the next token (D3), used through a prompt and tools (D7).",
      number: "The Tag 4127 report costs 40 GPT-2 tokens in English and 227 in Hindi.", cite: "D1", citeHref: "../d1-tokenisation/#languages",
      fail: "Hallucination. Asked for the year of the 2016 Ganges shark record, GPT-2 gave 16.7% to the year 2000 and 2.2% to 2016.", href: "../d7-assistants/#halluc", hrefText: "D7: why a model makes things up" },
    photomodel: { name: "Photo model", short: ["Photo model", "C1 to C3, E1"], ch: "C1 to C3, E1", kind: "model", runs: "both",
      takes: "A photo cut into pixels (a CNN) or patches (a vision transformer).", gives: "Scores over a fixed list of classes, or features for later stages.", model: "ResNet-18 (CNN) or ViT-B/16, both trained on the 1,000 classes of ImageNet.",
      number: "ViT-B/16 calls a hammerhead crop \"hammerhead\" at 88% and a grey bamboo shark \"electric ray\" at 56%, because ImageNet has no bamboo shark class.", cite: "E1", citeHref: "../e1-vision-transformers/#cls",
      fail: "A closed list always answers with something on the list. The background can decide as well: with the dolphin painted out, a great white shark score rose from 41% to 93% (B9).", href: "../e1-vision-transformers/#cls", hrefText: "E1: the class token" },
    rec: { name: "Catch records and tabular models", short: ["Catch records", "A2 to A6"], ch: "A2 to A6", kind: "model", runs: "shore",
      takes: "A row of six measurements: length, weight, girth, depth, temperature and month.", gives: "A weight check, species chances and a maturity score.", model: "Linear and logistic regression (A2, A3), trees (A5) and boosted trees (A6).",
      number: "XGBoost scored 85.4% on 164 held-out sharks for \"spadenose or not\". The lazy answer scores 56.1%.", cite: "A6, A5", citeHref: "../a6-boosting/#boundary",
      fail: "Leakage: a tree with no depth limit scored 94.7% when the same sharks were in both parts and 83.6% on new sharks (A4). A rare class hides behind accuracy: a model that never says Ganges is right on 597 of the 600 records, 99.5%, and finds none of the three.", href: "../a6-boosting/#importance", hrefText: "A6: which inputs matter" },
    clip: { name: "CLIP search and zero-shot", short: ["CLIP search", "E2"], ch: "E2", kind: "model", runs: "shore",
      takes: "A photo and a list of captions.", gives: "One similarity per caption, and chances that add to 100%.", model: "CLIP ViT-B/32: an image encoder and a text encoder, 151,277,313 numbers in all (counted for this chapter from the released weights).",
      number: "With \"a photo of a ...\" it names the right species for 5 of 10 shark photos, and separates shark from not shark on 14 of 14.", cite: "E2", citeHref: "../e2-clip/#zeroshot",
      fail: "If the true species is missing from the list, it still picks a caption. In murky water the species call fell from 88% right on 8 photos at 10 m to 50% at 30 m.", href: "../e2-clip/#zeroshot", hrefText: "E2: zero-shot classification" },
    vlm: { name: "Vision-language model", short: ["VLM answers", "E3, E4"], ch: "E3, E4", kind: "model", runs: "shore",
      takes: "A photo as image tokens and a question as text tokens.", gives: "A sentence that answers the question.", model: "An image encoder, a connection (a projector or cross-attention) and a language model.",
      number: "A 224 by 224 photo becomes 196 image tokens (E1). A 40-minute dive at one frame a second would be 470,400 (D6).", cite: "E1, D6", citeHref: "../e1-vision-transformers/#cut",
      fail: "Hallucination again. The sentence can name a status or a species that the photo does not support. Check its claims against a table (D7).", href: "../e4-vlms/#fail", hrefText: "E4: vision-language models", href2: "../e3-cross-attention/#occ", hrefText2: "E3: cross-attention" },
    decide: { name: "Decision", short: ["Decision", "rules"], ch: "A4, D7, F1", kind: "answer", runs: "shore",
      takes: "Species chances from three sources, a maturity score and the results of the checks.", gives: "A record to write, or a case sent to a person.", model: "A rule, not a trained model: multiply the chances, then accept or ask.",
      number: "For Tag 4127 the product gives about 99.6% for the grey bamboo shark (computed in section 2, with the controls as they load).", cite: "section 2", citeHref: "#trace",
      fail: "Over-trust. The product assumes the sources fail separately, and they may not. In F1 a printed 70% became 36% when the base rate fell from 34.9% to 11.4%.", href: "../f1-fairness/#trust", hrefText: "F1: a number that looks certain" },
    check: { name: "Fairness and privacy checks", short: ["Fairness and privacy checks", "F1"], ch: "F1", kind: "check", runs: "shore",
      takes: "Scores, labels and a group for a case (site, language, water, age), and the columns of a table before release.", gives: "Sensitivity and precision by group with intervals, and a count of rows that are unique.", model: "None. This is a procedure.",
      number: "At one threshold, sensitivity was 46% for women under 30 and 62% for women 30 and over (clinic data). In a made-up table of 24 patients, all 24 rows were unique when age was given in years, with sex and neighbourhood.", cite: "F1", citeHref: "../f1-fairness/#groups",
      fail: "A model fitted on one population and used on another: 34.9% of the Pima sample had diabetes, against 11.4% of India's adults.", href: "../f1-fairness/#groups", hrefText: "F1: fair for whom?" }
  };
  var ORDER = ["dive", "photo", "tag", "msg", "cam", "adapt", "tagreader", "msgreader", "photomodel", "rec", "clip", "vlm", "decide", "check"];
  var KIND_NAME = { "in": "input", reader: "reader", model: "model", answer: "answer", check: "check" };

  // geometry: x, y, w, h of every node, in the viewBox units of each layout
  var WIDE = { w: 940, h: 590, pos: {
    dive: [6, 50, 108, 38], photo: [6, 150, 108, 38], tag: [6, 360, 108, 38], msg: [6, 460, 108, 38],
    cam: [150, 38, 166, 64], adapt: [150, 138, 166, 64], tagreader: [150, 348, 166, 64], msgreader: [150, 448, 166, 64],
    photomodel: [356, 138, 166, 64], rec: [356, 398, 166, 64], clip: [562, 138, 166, 64], vlm: [562, 248, 166, 64],
    decide: [768, 200, 166, 80], check: [356, 520, 578, 56] },
    edges: [
      ["dive", "cam", "M114,69 H150"], ["photo", "adapt", "M114,169 H150"], ["tag", "tagreader", "M114,379 H150"], ["msg", "msgreader", "M114,479 H150"],
      ["cam", "adapt", "M233,102 V138"], ["adapt", "photomodel", "M316,170 H356"], ["photomodel", "clip", "M522,170 H562"], ["clip", "vlm", "M645,202 V248"],
      ["tagreader", "rec", "M316,380 L356,418"], ["msgreader", "rec", "M316,480 L356,442"],
      ["clip", "decide", "M728,160 H851 V200"], ["vlm", "decide", "M728,280 L768,260"], ["rec", "decide", "M522,430 H810 V280"],
      ["decide", "check", "M890,280 V520"]
    ], dashed: [["rec", "check", "M439,462 V520"]] };
  var TALL = { w: 340, h: 540, pos: {
    dive: [0, 22, 78, 40], photo: [87, 22, 78, 40], tag: [175, 22, 78, 40], msg: [262, 22, 78, 40],
    cam: [0, 110, 165, 54], adapt: [175, 110, 165, 54], tagreader: [0, 172, 165, 54], msgreader: [175, 172, 165, 54],
    photomodel: [0, 262, 165, 54], rec: [175, 262, 165, 54], clip: [0, 324, 165, 54], vlm: [175, 324, 165, 54],
    decide: [0, 414, 340, 54], check: [0, 478, 340, 54] },
    labels: [[8, "Comes in"], [96, "Readers"], [248, "Models and records"], [400, "Answer"]], arrows: [76, 234, 386] };

  function sysFig() {
    var svgNode = Kit.$("sys-svg"), svg = d3.select(svgNode), bar = Kit.$("sys-bar"), status = Kit.$("sys-status"), detail = Kit.$("sys-detail");
    var sel = "photomodel", path = [], k = 0, timer = 0, colorMode = 0, pathName = "";
    var f = D.clip.cos_species[1][4], pcl = clipProbs(f), dec0 = decide(D.softmax.probs, pcl, msgProbs(2, 0.8), 2, 0.9);
    NODES.decide.number = "For Tag 4127 the product gives " + pc(dec0.top, 1) + " for the grey bamboo shark (computed in section 2, with the controls as they load).";

    var PATHS = {
      photo: { label: "Photo", ids: ["photo", "adapt", "photomodel", "clip", "vlm", "decide", "check"], text: [
        "The photo of Tag 4127 comes in. This page uses a placeholder, a young grey bamboo shark in an aquarium (the credit is under the trace).",
        "C3, B7. The rock pool was clear, so the water step leaves the photo as it is. In murkier water this box would update statistics or a few weights.",
        "E1. ViT-B/16 scores 1,000 ImageNet classes. It has no bamboo shark, so its top call for this photo is electric ray at 56%.",
        "E2. CLIP compares the photo with nine captions: grey bamboo shark " + pc(pcl[2]) + ", milk shark " + pc(pcl[1]) + ".",
        "E3, E4. A vision-language model reads the photo as 196 image tokens and answers the question. On this page its answer is scripted.",
        "Rules. With the controls of section 2 as they load, the photo's chances are multiplied with the tabular chances and the message: " + pc(dec0.top, 1) + " for the grey bamboo shark. The rule accepts.",
        "F1. The checks ask who is missing: by site, water and language, and which columns could identify a person." ] },
      tag: { label: "Tag", ids: ["tag", "tagreader", "rec", "decide", "check"], text: [
        "A phone picture of the handwritten tag arrives. On this page, four MNIST digits (4, 1, 2, 7) replace the handwriting.",
        "C2. The digit reader gives 4, 1, 2 and 7, all above 99.9%. On the 10,000 test digits it is right 98.89% of the time, so a four-digit tag is right about 95.6% of the time.",
        "A2 to A6. The tag number 4127 picks the row: a female of 44.0 cm, 12.4 cm girth, found at 0.4 m at Juhu in December.",
        "Rules. The row gives the chances from the tabular models. A wrong tag would put this call into another animal's row.",
        "F1. A check compares the length in the message and the length in the row, and counts rows that could identify a fisher or a place." ] },
      msg: { label: "Message", ids: ["msg", "msgreader", "rec", "decide", "check"], text: [
        "A report arrives: \"Found a young grey bamboo shark in a rock pool at Juhu ... Length 44 cm ... Tag 4127.\" (a made-up message in four languages).",
        "D1 to D3, D7. The tokenizer cuts the message into pieces (40 GPT-2 tokens in English, 227 in Hindi). A language model with a prompt pulls out tag, length, species and place. On this page the fields are scripted.",
        "A2 to A6. The fields fill or check a row of the table: the tag, a length of 44 cm and the species the message names.",
        "Rules. The message is one of three sources of the species call. It names the grey bamboo shark.",
        "F1. The message names a place and a tag. The checks decide how much of that may be published." ] },
      dive: { label: "Dive footage", ids: ["dive", "cam", "adapt", "photomodel", "clip", "decide", "check"], text: [
        "A 40-minute dive makes 2,400 frames at one frame a second (a made-up log).",
        "F2. The camera hands over a frame about once per 33 ms. A small model keeps the loop at 124 ms and the pointing error at 4.7 degrees.",
        "C3, B7. Water of any clarity: the frame is passed on as it is, or the model's statistics and a few weights are updated.",
        "C1 to C3, E1. A small photo model flags frames that may show a shark. The large one runs later on shore.",
        "E2. CLIP compares the flagged frames with the species captions, after the dive.",
        "Rules. The chances of a frame are combined with the records of the day.",
        "F1. The checks look at the groups: dive sites, water types, species that are rare." ] }
    };

    // controls
    var pathChips = Kit.chips(bar, ["Photo", "Tag", "Message", "Dive footage"], -1, function (i) { pickPath(["photo", "tag", "msg", "dive"][i]); });
    var back = Kit.button(bar, "Back", function () { stop(); if (k > 0) { k--; show(); } });
    var step = Kit.button(bar, "Step", function () { stop(); if (!path.length) pickPath("photo"); if (k < path.length) { k++; show(); } });
    var play = Kit.button(bar, "Play", function () { stop(); if (!path.length) pickPath("photo"); if (k >= path.length) k = 0; (function tick() { if (k >= path.length) { timer = 0; return; } k++; show(); timer = setTimeout(tick, 1500); })(); });
    Kit.button(bar, "Reset", function () { stop(); path = []; k = 0; pathName = ""; setPressed(pathChips, -1); show(); });
    var colorChips = Kit.chips(bar, ["Colour by kind", "Colour by where it runs"], 0, function (i) { colorMode = i; draw(); });
    function stop() { if (timer) { clearTimeout(timer); timer = 0; } }
    function pickPath(name) {
      stop(); pathName = name; path = PATHS[name].ids; k = 0; setPressed(pathChips, ["photo", "tag", "msg", "dive"].indexOf(name)); show();
    }
    function show() {
      if (k > 0) sel = path[k - 1];
      status.textContent = k > 0 ? "Step " + k + " of " + path.length + ". " + PATHS[pathName].text[k - 1] : (path.length ? "Press Step to follow the " + PATHS[pathName].label.toLowerCase() + " through the system." : "Choose what comes in, then press Step. Or click any box.");
      back.disabled = k === 0; draw(); fill();
    }
    function fill() {
      var n = NODES[sel];
      var runsText = n.runs === "robot" ? "the robot, inside the loop" : n.runs === "shore" ? "the shore computer, after the dive or when a message arrives" : "a small version on the robot and a large one on shore";
      var linkHtml = link(n.href, n.hrefText) + (n.href2 ? " &middot; " + link(n.href2, n.hrefText2) : "");
      detail.innerHTML = '<h3>' + esc(n.name) + '<small>' + esc(KIND_NAME[n.kind]) + ' &middot; ' + esc(n.ch) + '</small></h3><dl>' +
        '<dt>Takes in</dt><dd>' + esc(n.takes) + '</dd><dt>Gives out</dt><dd>' + esc(n.gives) + '</dd><dt>Model</dt><dd>' + esc(n.model) + '</dd>' +
        '<dt>Measured number</dt><dd>' + esc(n.number) + ' <span class="cite">(' + link(n.citeHref, esc(n.cite)) + ')</span></dd>' +
        '<dt>Where it breaks</dt><dd>' + esc(n.fail) + '</dd><dt>Runs on</dt><dd><span class="runs ' + n.runs + '">' + esc(n.runs === "both" ? "robot or shore" : n.runs) + '</span> ' + esc(runsText) + '</dd>' +
        '<dt>Read more</dt><dd>' + linkHtml + '</dd></dl>';
    }

    function draw() {
      var P = Kit.palette(), W = Kit.width(svgNode), tall = W < 600, L = tall ? TALL : WIDE;
      var kindCol = { "in": P.sand, reader: P.navy, model: P.teal, answer: P.coral, check: P.ink2 };
      var runCol = { robot: P.teal, shore: P.navy, both: P.sand };
      var lit = {}, curId = k > 0 ? path[k - 1] : null, order = {};
      path.slice(0, k).forEach(function (id, i) { lit[id] = true; order[id] = i + 1; });
      var litEdge = {}; for (var i = 1; i < k; i++) litEdge[path[i - 1] + ">" + path[i]] = true;
      svgNode.setAttribute("viewBox", "0 0 " + L.w + " " + L.h);
      svgNode.setAttribute("aria-label", "Diagram of SharkCam with 4 inputs and 10 stages, ending in a decision. Photo, tag and message pass through readers, then models and records, then the decision, with the fairness and privacy checks last. Only the robot camera and a small photo model run inside the quarter-second loop." + (k > 0 ? " Showing step " + k + " of the " + PATHS[pathName].label.toLowerCase() + " path: " + NODES[path[k - 1]].name + "." : ""));
      svg.selectAll("*").remove();
      var defs = svg.append("defs");
      [["m-mut", P.muted], ["m-lit", P.teal]].forEach(function (m) {
        defs.append("marker").attr("id", m[0]).attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5).attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
          .append("path").attr("d", "M0,1 L9,5 L0,9 z").attr("fill", m[1]);
      });
      if (!tall) {
        L.edges.forEach(function (e) {
          var on = !!litEdge[e[0] + ">" + e[1]];
          svg.append("path").attr("d", e[2]).attr("fill", "none").attr("stroke", on ? P.teal : P.muted).attr("stroke-width", on ? 3.2 : 1.6).attr("opacity", on ? 1 : 0.75).attr("marker-end", "url(#" + (on ? "m-lit" : "m-mut") + ")");
        });
        L.dashed.forEach(function (e) { svg.append("path").attr("d", e[2]).attr("fill", "none").attr("stroke", P.muted).attr("stroke-width", 1.4).attr("stroke-dasharray", "5 4").attr("marker-end", "url(#m-mut)"); });
      } else {
        L.labels.forEach(function (l) { svg.append("text").attr("x", 0).attr("y", l[0] + 8).attr("font-size", 12).attr("fill", P.muted).attr("letter-spacing", "0.04em").text(l[1].toUpperCase()); });
        L.arrows.forEach(function (y) { svg.append("path").attr("d", "M165,{y} l10,0 l-5,8 z".replace("{y}", y)).attr("fill", P.muted); });
      }
      ORDER.forEach(function (id) {
        var n = NODES[id], p = L.pos[id], x = p[0], y = p[1], w = p[2], h = p[3], isPill = n.kind === "in";
        var c = colorMode === 1 ? runCol[n.runs] : kindCol[n.kind];
        var g = svg.append("g").attr("class", "node").attr("transform", "translate(" + x + "," + y + ")").attr("tabindex", 0).attr("role", "button")
          .attr("aria-label", n.name + ", chapters " + n.ch + ", " + KIND_NAME[n.kind] + (order[id] ? ", step " + order[id] : "") + ". Press to see details.")
          .attr("aria-pressed", String(sel === id));
        g.append("rect").attr("width", w).attr("height", h).attr("rx", isPill ? 19 : 9).attr("fill", P.bg);
        g.append("rect").attr("width", w).attr("height", h).attr("rx", isPill ? 19 : 9).attr("fill", (lit[id] && colorMode === 0) ? P.teal : c).attr("opacity", (lit[id] && colorMode === 0) ? 0.22 : (colorMode === 1 ? (lit[id] ? 0.32 : 0.2) : 0.1));
        g.append("rect").attr("class", "edge").attr("width", w).attr("height", h).attr("rx", isPill ? 19 : 9).attr("fill", "none")
          .attr("stroke", curId === id ? P.coral : sel === id ? P.ink : c).attr("stroke-width", curId === id ? 4 : sel === id ? 3 : 1.8)
          .attr("stroke-dasharray", n.kind === "check" ? "6 4" : null);
        var t1 = isPill ? n.name : (n.short ? n.short[0] : n.name), t2 = isPill ? null : (n.short ? n.short[1] : n.ch);
        var fs = tall ? 13.5 : 15;
        if (isPill) {
          g.append("text").attr("x", w / 2).attr("y", h / 2 + 5).attr("text-anchor", "middle").attr("font-size", tall ? 12 : 13.5).attr("font-weight", 600).attr("fill", P.ink).text(t1);
        } else {
          g.append("text").attr("x", 10).attr("y", h / 2 - (tall ? 1 : 2)).attr("font-size", fs).attr("font-weight", 600).attr("fill", P.ink).text(t1);
          g.append("text").attr("x", 10).attr("y", h / 2 + (tall ? 15 : 18)).attr("font-size", tall ? 12 : 13).attr("fill", P.muted).text(t2);
          if (colorMode === 1) g.append("text").attr("x", w - 8).attr("y", 15).attr("text-anchor", "end").attr("font-size", 11.5).attr("font-weight", 600).attr("fill", P.ink2).text(n.runs === "both" ? "both" : n.runs);
        }
        if (order[id]) {
          g.append("circle").attr("cx", w - (isPill ? 4 : 12)).attr("cy", isPill ? 4 : h - 12).attr("r", 11).attr("fill", curId === id ? P.coral : P.teal);
          g.append("text").attr("x", w - (isPill ? 4 : 12)).attr("y", (isPill ? 4 : h - 12) + 4.5).attr("text-anchor", "middle").attr("font-size", 13).attr("font-weight", 700).attr("fill", "#fff").text(order[id]);
        }
        g.on("click", function () { stop(); sel = id; draw(); fill(); });
        g.on("keydown", function (ev) { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); stop(); sel = id; draw(); fill(); } });
      });
    }
    Kit.onTheme(draw);
    window.addEventListener("resize", Kit.debounce(draw, 120));
    show();
  }

  // =====================================================================================
  // 1b. Which stages fit the loop?
  // =====================================================================================
  function budgetFig() {
    var bar = Kit.$("bd-bar"), eq = Kit.$("bd-eq"), counts = Kit.$("bd-counts"), status = Kit.$("bd-status");
    var M = [["MobileNetV2", 300774272], ["ResNet-18", 1814073344], ["CLIP image encoder (ViT-B/32 size)", 4.409e9]];
    var pick = 0, pipe = false;
    Kit.chips(bar, M.map(function (m) { return m[0].split(" (")[0]; }), pick, function (i) { pick = i; upd(); });
    Kit.chips(bar, ["One frame at a time", "Pipelined"], 0, function (i) { pipe = i === 1; upd(); });
    function upd() {
      var r = loop(M[pick][1], pipe), ok = r.err <= ROBOT.allowed;
      eq.innerHTML = ebox("c-sand", nf(ROBOT.cam, 1), "camera, ms") + eop("+") + ebox("c-teal", nf(r.model, 0), "model, ms") + eop("+") + ebox("c-navy", nf(ROBOT.dec, 0), "decision, ms") + eop("+") + ebox("c-coral", nf(ROBOT.motor, 0), "motor, ms") + eop("=") + ebox("", nf(r.L, 0), "latency L, ms") +
        eop("&rarr;") + ebox("", nf(r.L + r.T, 0), "L + T, ms") + eop("&times;") + ebox("c-sand", nf(ROBOT.omega, 1), "degrees per second") + eop("=") + ebox(ok ? "c-teal" : "c-coral", nf(r.err, 1), "pointing error, degrees");
      counts.innerHTML = '<div><b>' + nf(M[pick][1] / 1e9, 2) + ' billion</b><span>multiply-adds per frame</span></div><div><b>' + nf(r.model, 0) + ' ms</b><span>time in the model at 11.4 billion per second</span></div>' +
        '<div class="' + (ok ? "hit" : "miss") + '"><b>' + nf(r.err, 1) + '&deg;</b><span>pointing error, 5 allowed</span></div><div class="' + (ok ? "hit" : "miss") + '"><b>' + (ok ? "fits" : "too slow") + '</b><span>inside the quarter-second loop?</span></div>';
      status.textContent = M[pick][0] + (pipe ? ", pipelined: " : ", one frame at a time: ") + "latency " + nf(r.L, 0) + " ms, a new command after " + nf(r.T, 0) + " ms, camera " + nf(r.err, 1) + " degrees behind the shark, " + (ok ? "inside" : "outside") + " the 5 degrees allowed.";
    }
    upd();
  }

  // =====================================================================================
  // 2. The trace of Tag 4127
  // =====================================================================================
  var S = { lang: 0, photo: 4, depth: 0, tmpl: 1, named: 2, trust: 0.8, tau: 0.9, nTag: 4, vlmBad: false };
  var photoImg = null;

  function bars(names, probs, topIdx, wrongIdx) {
    return '<div class="pbars">' + names.map(function (n, i) {
      var cls = "pbar" + (i === topIdx ? " top" : "") + (i === wrongIdx ? " wrong" : "");
      return '<div class="' + cls + '"><span class="n" title="' + esc(n) + '">' + esc(n) + '</span><span class="tr"><i style="width:' + (100 * probs[i]).toFixed(1) + '%"></i></span><b>' + pc(probs[i], probs[i] < 0.1 && probs[i] > 0 ? 1 : 0) + '</b></div>';
    }).join("") + '</div>';
  }
  function kv(model, output, conf) { return '<div class="kv"><div><i>Model</i><span>' + model + '</span></div><div><i>Output</i><span>' + output + '</span></div><div><i>Confidence</i><span>' + conf + '</span></div></div>'; }
  function badge(kind, text) { return '<span class="badge ' + kind + '">' + text + '</span>'; }
  function head(n, title, chs, badges) { return '<div class="step-head"><span class="num">' + n + '</span><b>' + title + '</b><span class="chap">' + chs + '</span>' + badges + '</div>'; }

  function traceState() {
    var bam = S.photo === 4, cosRow = bam ? BD.cos_species[S.tmpl][S.depth / 2] : D.clip.cos_species[S.tmpl][S.photo];
    var pcl = clipProbs(cosRow), pmsg = msgProbs(S.named, S.trust), dec = decide(D.softmax.probs, pcl, pmsg, S.named, S.tau);
    return { bam: bam, pcl: pcl, pmsg: pmsg, dec: dec, depthOk: bam };
  }

  function renderTrace() {
    var st = traceState(), dec = st.dec, rec = D.record, msg = D.messages[S.lang], x = D.xgb, mat = D.maturity;
    var h = "";
    // 1 tag reader
    var tagP = D.tag.digits.map(function (d) { return d.p_top; }), whole = tagP.reduce(function (a, b) { return a * b; }, 1);
    h += '<div class="step">' + head(1, "Tag reader", link("../c2-cnns/#read", "C2"), badge("comp", "computed here")) + '<div class="step-body">' +
      '<div class="digits">' + D.tag.digits.map(function (d, i) { return '<div class="digit"><canvas id="dg' + i + '" width="28" height="28" role="img" aria-label="MNIST digit ' + d.label + '"></canvas><b>' + d.top + '</b>' + pc(d.p_top, 2) + '</div>'; }).join("") + '</div>' +
      '<p>The reader gives 4, 1, 2 and 7, so the tag is 4127. The four pictures are MNIST test digits that replace the handwriting. The product of the four chances is ' + pc(whole, 2) + '. Over the 10,000 test digits the reader is right ' + pc(D.tag.test_accuracy, 2) + ' of the time, so a tag of ' + S.nTag + ' digits is right about ' + pc(Math.pow(D.tag.test_accuracy, S.nTag), 1) + ' of the time if errors are independent.</p></div>' +
      kv("C2 digit reader, a CNN with 3,730 numbers", "digits 4, 1, 2, 7: Tag 4127", "above 99.9% per digit; 98.89% right on average") +
      '<p class="src">Source: ' + esc(D.tag.source) + '.</p></div>';

    // 2 message reader
    var isNative = msg.needs_native_check;
    h += '<div class="step">' + head(2, "Message reader", link("../d1-tokenisation/#languages", "D1") + ", " + link("../d7-assistants/#tools", "D7"), badge("stored", "token counts stored") + " " + badge("script", "fields scripted")) + '<div class="step-body two"><div>' +
      '<div class="msg-box" lang="' + ({ English: "en", Hindi: "hi", Marathi: "mr", Hinglish: "en" })[msg.language] + '">' + esc(msg.text) + '<small>' + esc(msg.language) + ', ' + esc(msg.speaker) + ' at ' + esc(msg.place) + '. Invented message ' + esc(msg.id) + '.' + (msg.gloss_en ? ' Meaning in English: ' + esc(msg.gloss_en) : '') + '</small></div>' +
      (isNative ? '<p class="mini-note">This Hindi or Marathi text was written for the book and has not been checked by a native speaker.</p>' : '') +
      '</div><div><p>Tokens: <b>' + msg.tokens_gpt2 + '</b> with GPT-2 and <b>' + msg.tokens_qwen + '</b> with Qwen2.5, against 40 and 45 for the English version. That is ' + nf(msg.tokens_gpt2 / D.messages[0].tokens_gpt2, 1) + ' times the GPT-2 cost.</p>' +
      '<p>Fields pulled out (scripted):</p><div class="ck"><b>tag</b> 4127 ' + (rec.record_id === 4127 ? '<b class="ok">matches the digit reader</b>' : '') + '</div><div class="ck"><b>length</b> 44 cm <b class="ok">matches the record</b></div><div class="ck"><b>species</b> grey bamboo shark (named in the message)</div><div class="ck"><b>place</b> Juhu, rock pool at low tide</div>' +
      '<p class="mini-note">Sentence vectors of the four versions (D2) are close: English to Hindi ' + nf(D.message_cos_vs_m02.m06, 2) + ', to Marathi ' + nf(D.message_cos_vs_m02.m10, 2) + ', to Hinglish ' + nf(D.message_cos_vs_m02.m14, 2) + '.</p></div></div>' +
      kv("Tokenizer (byte-pair encoding) and a language model with a prompt and tools", "tag, length, species, place", "none to report: a language model gives token chances, not the chance that a field is right") +
      '<p class="src">Source: ' + esc(D.messages_source) + '. The fields are scripted for this page.</p></div>';

    // 3 record and tabular models
    var sm = D.softmax, sp = sm.probs.slice(), topSm = argmaxL(sp);
    var shapNames = x.features.map(function (f) { return f.replace("_cm", "").replace("_kg", "").replace("_m", "").replace("_c", ""); });
    h += '<div class="step">' + head(3, "Catch record and tabular models", link("../a2-linear-regression/#loglog", "A2") + " to " + link("../a6-boosting/#shap", "A6"), badge("stored", "A6 stored") + " " + badge("comp", "A3 and maturity computed here")) + '<div class="step-body two"><div>' +
      '<table class="tbl words" aria-label="The record of Tag 4127"><tbody><tr><td>Tag</td><td>' + rec.record_id + ' (made up)</td></tr><tr><td>Length, weight, girth</td><td>' + nf(rec.length_cm, 1) + ' cm, ' + nf(rec.weight_kg, 3) + ' kg, ' + nf(rec.girth_cm, 1) + ' cm</td></tr><tr><td>Sex, source</td><td>' + (rec.sex === "F" ? "female" : "male") + ', a ' + esc(rec.source) + ' at ' + esc(rec.site) + '</td></tr><tr><td>Depth, water, month</td><td>' + nf(rec.depth_m, 1) + ' m, ' + nf(rec.temp_c, 1) + ' C, month ' + rec.month + '</td></tr></tbody></table>' +
      '<p class="mini-note">Weight check (A2): the bamboo law W = 0.00398 L<sup>2.95</sup> gives ' + nf(0.00398 * Math.pow(rec.length_cm, 2.95), 0) + ' g for 44 cm. The record says ' + nf(rec.weight_kg * 1000, 0) + ' g. The row is consistent.</p></div><div>' +
      '<p><b>Species from length and girth (A3 softmax)</b></p>' + bars(SHORT5, sp, topSm, topSm !== 2 ? topSm : -1) +
      '<p style="margin-top:8px"><b>Spadenose or not, six measurements (A6 XGBoost)</b>: chance of spadenose <b>' + pc(x.p_spadenose, 1) + '</b>. The biggest push is depth, ' + nf(x.shap[3], 2) + ' in score units. At 0.4 m the animal is shallower than all 268 spadenose sharks of the table (counted from the table: the shallowest is at 5.6 m).</p>' +
      '<p><b>Mature? (A3 idea, bamboo-only model on length)</b>: ' + pc(mat.p_mature_44, 1) + ' mature at 44 cm (L50 ' + nf(mat.L50, 1) + ' cm). The spadenose model of A3 would say ' + pc(mat.spadenose_model_p_adult, 1) + ' adult.</p></div></div>' +
      kv("Softmax classifier (A3), XGBoost with 100 trees (A6), logistic regression for bamboo maturity", "spadenose " + pc(sp[0]) + " (wrong top call), not spadenose " + pc(1 - x.p_spadenose, 1) + ", juvenile", "the classifier got 3 of the 72 bamboo records right (A3); XGBoost 85.4% on held-out sharks (A6)") +
      '<p class="src">Sources: ' + esc(sm.source) + '. ' + esc(x.source) + '. ' + esc(mat.source) + '.</p></div>';

    // 4 photo
    var nm = NAMES9, ptop = argmaxL(st.pcl);
    var depthTxt = st.bam ? (S.depth + " m") : "the surface (depth variants exist for the placeholder photo only)";
    var vitHtml = (st.bam && S.depth === 0 && D.vit) ? '<p><b>ViT-B/16 (E1, stored)</b>: ' + D.vit.top5.slice(0, 3).map(function (t) { return esc(t[0]) + ' ' + pc(t[1]); }).join(", ") + '. ImageNet has no bamboo shark class.</p>' : '<p class="mini-note">ViT-B/16 outputs (E1) are stored for the clear placeholder photo only.</p>';
    h += '<div class="step">' + head(4, "Photo, water and CLIP", link("../c3-domain-shift/#water", "C3") + ", " + link("../e1-vision-transformers/#cls", "E1") + ", " + link("../e2-clip/#zeroshot", "E2"), badge("stored", "E1, E2 stored") + (st.bam && S.depth > 0 ? " " + badge("comp", "CLIP at depth computed for this chapter") : "")) + '<div class="step-body two"><div class="photo-box">' +
      (st.bam ? '<canvas id="ph-canvas" width="360" height="270" role="img" aria-label="The placeholder photo of a young grey bamboo shark at ' + S.depth + ' m of water, drawn with the depth model of C3"></canvas>' : '<img src="data/thumb/' + D.clip.photos[S.photo] + '.jpg" alt="Thumbnail of the chosen photo">') +
      '<p class="credit">' + esc(D.clip.credits[S.photo]) + (st.bam ? ' Shown here at ' + depthTxt + ' with the depth model of C3. A placeholder: this is not Tag 4127.' : '') + '</p></div><div>' +
      vitHtml + '<p><b>CLIP ViT-B/32 (E2), prompt ' + esc(D.clip.templates[S.tmpl].replace("{}", "...")) + '</b></p>' + bars(["Spadenose", "Milk", "Grey bamboo", "Hammerhead", "Ganges", "Eagle ray", "Dolphin", "Clownfish", "Sea turtle"], st.pcl, ptop, -1) +
      '<p class="mini-note">Chance on the five sharks: ' + pc(dec.shark, 1) + '. The photo is shown at ' + depthTxt + '.</p></div></div>' +
      kv("ViT-B/16 for 1,000 ImageNet classes; CLIP ViT-B/32 for nine captions", esc(nm[ptop]) + " at " + pc(st.pcl[ptop], 1), "softmax of 100 times the cosines, " + pc(st.pcl[ptop], 1) + " on the top caption") +
      '<p class="src">Sources: ' + esc(D.clip.source) + '; ' + esc(D.vit.source) + (st.bam && S.depth > 0 ? '; data/bamboo_depth.json (CLIP run offline on the placeholder at this depth, same method as E2).' : '.') + '</p></div>';

    // 5 vlm
    var vlmOK = S.photo === 4, spC = D.species.bamboo, say = "This looks like a young grey bamboo shark, Chiloscyllium griseum. The dark bands across its back fit a young animal. The IUCN Red List lists the species as " + (S.vlmBad ? "Endangered" : "Vulnerable") + ".";
    var claimSpecies = dec.win === 2, claimStatus = !S.vlmBad && spC.iucn === "Vulnerable";
    h += '<div class="step">' + head(5, "Vision-language model", link("../e3-cross-attention/#sources", "E3") + ", " + link("../e4-vlms/#pipeline", "E4"), badge("script", "scripted")) + '<div class="step-body">' +
      '<p>Question: "Which shark is this, and how threatened is it?"</p>' +
      (vlmOK ? '<div class="say">' + esc(say) + '</div><p class="ck"><b class="' + (claimSpecies ? "ok" : "no") + '">' + (claimSpecies ? "pass" : "fail") + '</b> The species in the answer ' + (claimSpecies ? "matches" : "does not match") + ' the fused call of step 6 (' + esc(D.species[SPECIES[dec.win]].common) + ').</p>' +
        '<p class="ck"><b class="' + (claimStatus ? "ok" : "no") + '">' + (claimStatus ? "pass" : "fail") + '</b> The status in the answer is checked against the species table: ' + esc(spC.common) + ', ' + esc(spC.iucn) + ' (' + esc(spC.iucn_code) + ', ' + esc(spC.criteria) + ', assessed ' + esc(String(spC.assessed).slice(0, 4)) + ').</p>' :
        '<p class="mini-note">No scripted answer is written for this photo. Choose the placeholder photo of the grey bamboo shark to see the step.</p>') +
      '<p class="mini-note">The answer is scripted. A full-size model reads the photo as 196 image tokens (E1) and writes the sentence one token at a time (D3). Use the switch above to let the model state a wrong status and see the check catch it.</p></div>' +
      kv("A vision-language model (E4): image encoder, projector or cross-attention, language model", vlmOK ? "a sentence of three claims" : "none for this photo", "no single chance for a sentence; claims are checked against a table") +
      '<p class="src">Species table: data/sharks/species.json (read through ' + esc(D.species.bamboo.common) + '). The answer is scripted.</p></div>';

    // 6 decision
    var rows = SPECIES.map(function (s, i) {
      var tabv = D.softmax.probs[i], clv = st.pcl[i], mv = st.pmsg[i], fv = dec.fused[i], cell = function (v, cls) { return '<td class="cell ' + (cls || "") + '"><i style="width:' + (100 * v).toFixed(1) + '%"></i><span>' + pc(v, v > 0 && v < 0.1 ? 1 : 0) + '</span></td>'; };
      var cellP = function (v) { return '<td class="cell fz"><i style="width:' + (100 * v).toFixed(1) + '%"></i><span>' + pc(v, 1) + '</span></td>'; };
      return '<tr class="' + (i === dec.win ? "win" : "") + '"><td>' + SHORT5[i] + '</td>' + cell(tabv) + cell(clv) + cell(mv) + cellP(fv) + '</tr>';
    }).join("");
    var wI = dec.win, runner = argmaxL(dec.fused.map(function (v, i) { return i === wI ? -1 : v; }));
    var why = [];
    if (dec.top < S.tau) why.push("the winner's chance is under " + pt(S.tau));
    if (dec.agree < 2) why.push("only " + dec.agree + " of 3 sources name the winner");
    if (dec.shark < 0.5) why.push("CLIP puts under half its chance on sharks");
    var mat2 = wI === 2 ? ("juvenile (" + pc(mat.p_mature_44, 1) + " mature)") : "maturity model not fitted for this species here";
    h += '<div class="step">' + head(6, "Species call and decision", link("../a4-grading-models/#threshold", "A4") + ", " + link("../d7-assistants/#halluc", "D7") + ", " + link("../f1-fairness/#trust", "F1"), badge("comp", "computed here")) + '<div class="step-body">' +
      '<div class="tbl-wrap"><table class="fuse" aria-label="Chances for five species from three sources and the product"><thead><tr><th>Species</th><th>Table A3</th><th>CLIP E2</th><th>Message</th><th>Product</th></tr></thead><tbody>' + rows + '</tbody></table></div>' +
      '<p>Winner: <b>' + esc(D.species[SPECIES[wI]].common) + '</b> at <b>' + pc(dec.top, 1) + '</b>. Sources that name it: <b>' + dec.agree + ' of 3</b> (' + [dec.votes[0] ? "tabular" : null, dec.votes[1] ? "CLIP" : null, dec.votes[2] ? "message" : null].filter(Boolean).join(", ") + (dec.agree ? "" : "none") + '). CLIP puts <b>' + pc(dec.shark, 0) + '</b> on the five sharks. Rule: accept if the winner reaches <b>' + pt(S.tau) + '</b>, at least two sources name it, and CLIP puts at least 50% on sharks.</p>' +
      '<div class="verdict ' + (dec.accept ? "" : "ask") + '"><b>' + (dec.accept ? "Accept." : "Ask a person.") + '</b> ' + (dec.accept ? "Write the record: " + esc(D.species[SPECIES[wI]].common) + ", " + mat2 + "." : "The rule is not met: " + why.join(", and ") + ". The best guess is " + esc(D.species[SPECIES[wI]].common) + " at " + pc(dec.top, 1) + ", with " + esc(D.species[SPECIES[runner]].common) + " at " + pc(dec.fused[runner], 1) + ".") + '</div></div>' +
      kv("A rule (multiply, rescale, accept or ask), not a trained model", dec.accept ? "record written" : "held for a person", "the winner's chance is " + pc(dec.top, 1) + ", an upper bound because the sources are not independent") +
      '<p class="src">Inputs: the three columns above. The trust slider sets the message column. The rule is scripted.</p></div>';

    // 7 budget and checks
    var r1 = loop(300774272, false), r2 = loop(1814073344, false), r3 = loop(4.409e9, false), f2lnk = link("../f2-robots/#loop", "F2");
    h += '<div class="step">' + head(7, "Budget and checks", f2lnk + ", " + link("../f1-fairness/#groups", "F1"), badge("stored", "F2 values") + " " + badge("script", "checks scripted")) + '<div class="step-body two"><div>' +
      '<p><b>Where the models run (one frame at a time, Pi 4, 8 bits)</b></p><div class="tbl-wrap"><table class="tbl"><thead><tr><th>Model</th><th>Latency</th><th>Pointing error</th></tr></thead><tbody>' +
      '<tr><td>MobileNetV2</td><td>' + nf(r1.L, 0) + ' ms</td><td>' + nf(r1.err, 1) + '&deg;</td></tr><tr><td>ResNet-18</td><td>' + nf(r2.L, 0) + ' ms</td><td>' + nf(r2.err, 1) + '&deg;</td></tr><tr class="pick"><td>CLIP image encoder</td><td>' + nf(r3.L, 0) + ' ms</td><td>' + nf(r3.err, 1) + '&deg;</td></tr></tbody></table></div>' +
      '<p class="mini-note">The budget is 5 degrees. The small model points the camera; CLIP and the vision-language model run on shore. The CLIP time is an estimate (section 1).</p></div><div>' +
      '<p><b>Checks before the record is trusted</b></p><p class="ck"><b>Groups.</b> Compare recall by site (Juhu dive against Sassoon landing) and by message language. Show intervals (<a href="../f1-fairness/#groups">F1</a>).</p>' +
      '<p class="ck"><b>Privacy.</b> The message names a place and a tag. Publish a coarse place for a Vulnerable species (<a href="../f1-fairness/#privacy">F1</a>).</p>' +
      '<p class="ck"><b>Trust.</b> The fused chance assumes independent sources. Test the accept rule on held-out cases before relying on it (<a href="../f1-fairness/#trust">F1</a>).</p></div></div>' +
      kv("None: a procedure and a table of times", "loop times, three checks to run", "the pointing error is an estimate from F2's assumed times") + '</div>';

    Kit.$("tr-steps").innerHTML = h;
    D.tag.digits.forEach(function (d, i) { drawDigit(Kit.$("dg" + i), decodeDigit(d.b64)); });
    if (st.bam) drawPhoto();
    // final record
    var out = Kit.$("tr-record");
    out.innerHTML = '<div class="rec-out ' + (dec.accept ? "" : "ask") + '"><h3>' + (dec.accept ? "What SharkCam writes" : "What SharkCam keeps for a person") + '</h3><div class="row"><span><b>Tag</b> 4127</span><span><b>Species</b> ' + esc(D.species[SPECIES[wI]].common) + ' (' + pc(dec.top, 1) + ')</span><span><b>Length</b> 44.0 cm</span><span><b>Maturity</b> ' + mat2 + '</span><span><b>Place</b> Juhu, month 12</span></div><p>' +
      (dec.accept ? "The record is added to the made-up catch table with the photo, the message and the three source chances attached. The shark was put back, as the message says." : "The record waits for a person to look at the photo and the message. The table stays unchanged until then.") + '</p></div>';
    // equation boxes
    var wi = dec.win, rawW = dec.raw[wi], rawR = dec.raw[runner];
    Kit.$("tr-eqn").innerHTML = ebox("c-navy", nf(D.softmax.probs[wi], 3), "tabular, " + SHORT5[wi].toLowerCase()) + eop("&times;") + ebox("c-teal", nf(st.pcl[wi], 3), "CLIP") + eop("&times;") + ebox("c-sand", nf(st.pmsg[wi], 3), "message") + eop("=") + ebox("", nf(rawW, 4), "product, winner") +
      eop("|") + ebox("", nf(rawR, 4), "product, runner-up (" + SHORT5[runner].toLowerCase() + ")") + eop("&rarr;") + ebox("c-coral", pc(dec.top, 1), "winner after rescaling");
    Kit.$("tr-chain").innerHTML = ebox("c-navy", nf(D.tag.test_accuracy, 4), "right per digit") + eop("<sup>" + S.nTag + "</sup>") + eop("=") + ebox("c-coral", nf(Math.pow(D.tag.test_accuracy, S.nTag), 3), "whole tag right, n = " + S.nTag) + eop("|") + ebox("", nf(1000 * (1 - Math.pow(D.tag.test_accuracy, S.nTag)), 0), "wrong in 1,000 tags");
  }
  function argmaxL(a) { return argmax(a); }
  function drawDigit(cv, px) {
    if (!cv) return; var ctx = cv.getContext("2d"), im = ctx.createImageData(28, 28), i;
    for (i = 0; i < 784; i++) { var v = 255 - px[i]; im.data[4 * i] = v; im.data[4 * i + 1] = v; im.data[4 * i + 2] = v; im.data[4 * i + 3] = 255; }
    ctx.putImageData(im, 0, 0);
  }
  function drawPhoto() {
    var cv = Kit.$("ph-canvas"); if (!cv || !photoImg) return;
    var ctx = cv.getContext("2d"), off = document.createElement("canvas"); off.width = cv.width; off.height = cv.height;
    var oc = off.getContext("2d"); oc.drawImage(photoImg, 0, 0, cv.width, cv.height);
    var id = oc.getImageData(0, 0, cv.width, cv.height), out = ctx.createImageData(cv.width, cv.height);
    degradeData(id.data, out.data, S.depth); ctx.putImageData(out, 0, 0);
  }

  function traceFig() {
    var box = Kit.$("tr-controls");
    var r1 = Kit.html("div", { class: "row" }, box); Kit.html("span", { class: "label" }, r1, "Message");
    var langChips = Kit.chips(r1, ["English", "Hindi", "Marathi", "Hinglish"], 0, function (i) { S.lang = i; renderTrace(); });
    var r2 = Kit.html("div", { class: "row" }, box); Kit.html("span", { class: "label" }, r2, "Photo");
    var th = Kit.html("div", { class: "thumbs", role: "group", "aria-label": "Choose the photo that arrives with the report" }, r2), tbs = [];
    D.clip.photos.forEach(function (n, i) {
      var b = Kit.html("button", { type: "button", "aria-pressed": String(i === S.photo), title: D.clip.short[i] }, th); b.innerHTML = '<img src="data/thumb/' + n + '.jpg" alt="' + esc(D.clip.short[i]) + '"><span>' + esc(D.clip.short[i].split(",")[0]) + '</span>';
      b.addEventListener("click", function () { S.photo = i; tbs.forEach(function (x, j) { x.setAttribute("aria-pressed", String(j === i)); }); depthSl.input.disabled = i !== 4; if (i !== 4) { S.depth = 0; depthSl.set(0); } renderTrace(); });
      tbs.push(b);
    });
    var r3 = Kit.html("div", { class: "controls" }, box);
    var depthSl = Kit.slider(r3, { label: "Water depth for the placeholder photo, m", min: 0, max: 30, step: 2, value: 0, digits: 0 }, function (v) { S.depth = v; renderTrace(); });
    Kit.slider(r3, { label: "Digits in the tag, n", min: 1, max: 8, step: 1, value: 4, digits: 0 }, function (v) { S.nTag = v; renderTrace(); });
    Kit.slider(r3, { label: "Trust in the message", min: 0.2, max: 0.99, step: 0.01, value: 0.8, digits: 2 }, function (v) { S.trust = v; renderTrace(); });
    Kit.slider(r3, { label: "Accept at a fused chance of", min: 0.5, max: 0.999, step: 0.001, value: 0.9, digits: 3 }, function (v) { S.tau = v; renderTrace(); });
    var r4 = Kit.html("div", { class: "controls" }, box);
    Kit.select(r4, "Message names", [{ value: "2", label: "grey bamboo shark (as written)" }, { value: "0", label: "spadenose shark" }, { value: "1", label: "milk shark" }, { value: "3", label: "scalloped hammerhead" }, { value: "4", label: "Ganges shark" }, { value: "-1", label: "no species" }], "2", function (v) { S.named = +v; renderTrace(); });
    Kit.select(r4, "CLIP prompt", [{ value: "0", label: "name only" }, { value: "1", label: "a photo of a ..." }, { value: "2", label: "a photo of a ... in murky water" }], "1", function (v) { S.tmpl = +v; renderTrace(); });
    var r5 = Kit.html("div", { class: "row" }, box); Kit.html("span", { class: "label" }, r5, "Scripted VLM");
    Kit.chips(r5, ["As scripted", "States a wrong status"], 0, function (i) { S.vlmBad = i === 1; renderTrace(); });
    var cr = Kit.html("details", { class: "mini-note" }, box); cr.innerHTML = "<summary>Credits for the 14 photos</summary>" + D.clip.credits.map(function (c, i) { return "<p style=\"margin:3px 0\">" + esc(D.clip.photos[i]) + ": " + esc(c) + "</p>"; }).join("");
    Kit.html("p", { class: "mini-note" }, box, "All records, messages and the dive log are made up. The photo is a placeholder from an aquarium. The four digits are MNIST test digits.");
    var img = new Image(); img.onload = function () { photoImg = img; renderTrace(); }; img.src = "data/bamboo-1.jpg";
    renderTrace();
  }

  // =====================================================================================
  // 3. Failure gallery
  // =====================================================================================
  var TOOLS = {
    T1: { t: "Split by animal or by dive, then fit all the steps on the training part only", c: "A4" },
    T2: { t: "Plot training error against validation error (learning curves)", c: "B8" },
    T3: { t: "Test at several depths, then train on random water", c: "C3" },
    T4: { t: "Edit the background, cover the animal and score again", c: "B9" },
    T5: { t: "Ask for evidence: look the claim up in a table you trust", c: "D7" },
    T6: { t: "Score the groups on their own, with intervals", c: "F1" },
    T7: { t: "Count the answers that changed on a fixed test set", c: "F2" },
    T8: { t: "Test the old task again after the update", c: "B7" }
  };
  var FAILS = [
    { id: "leak", title: "Leakage", ch: "A4", href: "../a4-grading-models/#leak", right: "T1", opts: ["T2", "T1", "T6", "T7"],
      story: "A tree with no depth limit is tested on photos of sharks it has already seen frames of. It scores 94.7%. On new sharks it scores 83.6%.",
      figs: [["94.7%", "same sharks in both parts", "miss"], ["83.6%", "new sharks", "hit"]],
      cause: "The test part held sharks the tree had memorised. The score measured memory.",
      tip: "Split by animal or by dive before anything else. Fit scalers, selectors and thresholds on the training part only." },
    { id: "over", title: "Overfitting", ch: "A4, B8", href: "../a4-grading-models/#overfit", right: "T2", opts: ["T1", "T3", "T2", "T8"],
      story: "A degree-9 polynomial is fitted to 14 sharks. It misses the training sharks by 0.023 kg and the test sharks by about 92 kg. A degree-3 curve misses the test sharks by 0.039 kg.",
      figs: [["0.023 kg", "training error, degree 9", "hit"], ["92 kg", "test error, degree 9", "miss"], ["0.039 kg", "test error, degree 3", "hit"]],
      cause: "The model has more dials than data and fits the noise. The training error tells you little about new sharks.",
      tip: "Plot training and validation error together. A model that fits training data almost perfectly and fails on validation needs fewer dials, more data or a brake (B7)." },
    { id: "water", title: "Murky water", ch: "C3", href: "../c3-domain-shift/#water", right: "T3", opts: ["T3", "T2", "T4", "T7"],
      story: "A digit reader trained only on clean digits is tested after the digits are put through the depth model of C3.",
      figs: [["98.6%", "at 0 m", "hit"], ["68.6%", "at 2 m", "miss"], ["10.6%", "at 10 m", "miss"]],
      cause: "Water dims the picture and veils it with haze, so the pixels the model sees change. The clean training set had no pictures like this.",
      tip: "Test at several depths, not at one. Train with random water (augmentation) and, where you can, adapt on unlabelled frames while the robot works." },
    { id: "cue", title: "Spurious cues", ch: "B9", href: "../b9-explaining/#shortcut", right: "T4", opts: ["T1", "T4", "T3", "T5"],
      story: "ResNet-18 sees a dolphin and gives a great white shark score of 41%. With the dolphin painted out and the water left, the score rises.",
      figs: [["41%", "dolphin in the photo", "hit"], ["93%", "dolphin painted out", "miss"]],
      cause: "The score follows the water around the animal, a cue that came with sharks in the training photos.",
      tip: "Edit the photo to remove the animal and keep the background. If the score stays high, the model reads the background. Check an explanation map against a control." },
    { id: "hall", title: "Hallucination", ch: "D7, E4", href: "../d7-assistants/#halluc", right: "T5", opts: ["T4", "T6", "T5", "T8"],
      story: "GPT-2 is asked for the year of the Ganges shark record at Sassoon Dock. It puts 16.7% on the year 2000 and 2.2% on 2016, which is the right year.",
      figs: [["16.7%", "year 2000", "miss"], ["2.2%", "year 2016, the true one", "hit"]],
      cause: "The model has learned what answers look like, and a year-like token is likely. A fluent answer is not evidence. A vision-language model can do the same with a status or a species.",
      tip: "Ask where the claim comes from. Check names, numbers and conservation status against a table with a tool call (D7), as step 5 of the trace does." },
    { id: "group", title: "Subgroup gaps", ch: "F1, A4", href: "../f1-fairness/#groups", right: "T6", opts: ["T1", "T5", "T6", "T2"],
      story: "A clinic model with one threshold of 0.5 is scored on women under 30 and on women 30 and over.",
      figs: [["46%", "sensitivity, under 30", "miss"], ["62%", "sensitivity, 30 and over", "hit"]],
      cause: "The groups have different base rates and different score spreads, so one threshold gives different sensitivity and precision. The overall number hides it.",
      tip: "Report sensitivity and precision per group, with intervals. Look at the base rate of the groups before you move a threshold." },
    { id: "quant", title: "Quantisation flips", ch: "F2", href: "../f2-robots/#quant", right: "T7", opts: ["T3", "T8", "T7", "T6"],
      story: "The weights of the digit reader are rounded to fewer bits. Accuracy on 10,000 test digits barely moves at first.",
      figs: [["98.87%", "at 8 bits, 6 answers changed", "hit"], ["97.67%", "at 4 bits, 184 answers changed", "miss"], ["92.16%", "at 3 bits, 761 changed", "miss"]],
      cause: "Rounding moves weights a little. Digits close to a boundary flip, in both directions, while the average stays put.",
      tip: "Run the full model and the rounded model on the same inputs and count the answers that differ. Then look at the classes that matter most." },
    { id: "forget", title: "Forgetting", ch: "B7", href: "../b7-continual-learning/#forget", right: "T8", opts: ["T2", "T3", "T7", "T8"],
      story: "A network is trained on clear water and then fine-tuned on murky water.",
      figs: [["88%", "clear water, before", "hit"], ["73%", "clear water, after", "miss"], ["86%", "murky water, after", "hit"]],
      cause: "The new task moved weights that the old task needed. The new data gives the model no reason to protect them.",
      tip: "Keep a held-out set from the earlier tasks and test it after the update. Replay a few old examples or add springs to the important weights (EWC)." }
  ];
  function failFig() {
    var grid = Kit.$("fl-grid"), scoreRoot = Kit.$("fl-score"), tried = {}, firstRight = 0, done = 0;
    var scoreEl = Kit.html("span", {}, scoreRoot);
    function upd() { scoreEl.innerHTML = 'First choices right: <b>' + firstRight + ' of ' + FAILS.length + '</b> &middot; Cards answered: <b>' + done + '</b>'; }
    var btnAll = Kit.html("button", { type: "button", class: "btn" }, scoreRoot, "Reveal all"); btnAll.addEventListener("click", function () { FAILS.forEach(function (F) { if (!tried[F.id]) reveal(F, null); }); });
    FAILS.forEach(function (F) {
      var card = Kit.html("div", { class: "fcard", id: "fc-" + F.id }, grid);
      card.innerHTML = '<h3>' + esc(F.title) + ' <span class="ch">' + link(F.href, esc(F.ch)) + '</span></h3><p>' + esc(F.story) + '</p><div class="counts three">' +
        F.figs.map(function (f) { return '<div class="' + f[2] + '"><b>' + esc(f[0]) + '</b><span>' + esc(f[1]) + '</span></div>'; }).join("") + '</div><p class="q">Which tool would you reach for?</p><div class="opts" role="group" aria-label="Tools for ' + esc(F.title) + '"></div><div class="rv"></div>';
      var opts = card.querySelector(".opts"), rv = card.querySelector(".rv");
      F.opts.forEach(function (key) {
        var b = Kit.html("button", { type: "button", "data-k": key }, opts); b.innerHTML = esc(TOOLS[key].t) + ' <em>(' + esc(TOOLS[key].c) + ')</em>';
        b.addEventListener("click", function () { if (tried[F.id]) return; reveal(F, key); });
      });
      F.card = card; F.rv = rv; F.opts_el = opts;
    });
    function reveal(F, key) {
      tried[F.id] = true; done++;
      var good = key === F.right; if (key !== null && good) firstRight++;
      F.opts_el.querySelectorAll("button").forEach(function (b) { b.disabled = true; if (b.getAttribute("data-k") === F.right) b.classList.add("right"); else if (b.getAttribute("data-k") === key) b.classList.add("wrong"); });
      F.rv.innerHTML = '<div class="reveal"><p><b>' + (key === null ? "Answer." : good ? "Right." : "Not this one.") + '</b> The tool is: ' + esc(TOOLS[F.right].t) + ' (' + esc(TOOLS[F.right].c) + ').</p><p><b>Cause.</b> ' + esc(F.cause) + '</p><p><b>Debug tip.</b> ' + esc(F.tip) + '</p><p>' + link(F.href, "Open " + esc(F.ch)) + '</p></div>';
      upd();
    }
    upd();
  }

  // =====================================================================================
  // 4. Six equations
  // =====================================================================================
  var B4 = { x: [0.22, 0.21], W1: [[0.8, -0.6], [-0.4, 0.7]], b1: [0.1, -0.2], v: [0.9, -0.8], c: -0.9, y: 1 };
  function b4Net() {
    var z = [0, 1].map(function (j) { return B4.W1[j][0] * B4.x[0] + B4.W1[j][1] * B4.x[1] + B4.b1[j]; }), h = z.map(Math.tanh);
    var zo = B4.v[0] * h[0] + B4.v[1] * h[1] + B4.c, p = 1 / (1 + Math.exp(-zo));
    return { z: z, h: h, zo: zo, p: p };
  }
  var EQS = [
    { id: "lin", name: "Linear model and squared error", ch: "A2", href: "../a2-linear-regression/#line", prompt: "A line with two dials predicts a shark's weight from its length. Write the prediction and the loss that training makes small.",
      tex: ["\\hat y = w\\,x + b", "L = \\frac{1}{m}\\sum_{i=1}^{m}\\left(y^{(i)} - \\hat y^{(i)}\\right)^2"],
      notes: ["<b>\\(x\\)</b> is the length in cm, <b>\\(w\\)</b> the slope (kg per cm) and <b>\\(b\\)</b> the bias (kg). <b>\\(\\hat y\\)</b> is the predicted weight and <b>\\(y\\)</b> the true weight.", "<b>\\(m\\)</b> is the number of examples. <b>\\(L\\)</b> is the mean squared error (MSE), the average of the squared misses."],
      live: function (box) {
        var w = 0.01545, b = -0.3789, out = Kit.html("div", { class: "eqn" }, box), sl;
        function upd(x) { var yh = w * x + b; out.innerHTML = ebox("c-navy", nf(w, 5), "w, kg per cm") + eop("&times;") + ebox("c-coral", nf(x, 0), "x, cm") + eop("+") + ebox("c-sand", nf(b, 4), "b, kg") + eop("=") + ebox("c-teal", nf(yh, 3), "prediction, kg") + (x === 41 ? eop("|") + ebox("", "0.228", "Tag 3305, true kg") + eop("&rarr;") + ebox("", nf(0.228 - yh, 3), "miss, y minus prediction") : ""); }
        sl = Kit.slider(box, { label: "Length x, cm (41 is Tag 3305; A2's line for 268 spadenose sharks)", min: 18, max: 60, step: 1, value: 41, digits: 0 }, upd); box.insertBefore(sl.input.parentNode, out); upd(41);
      } },
    { id: "sig", name: "Sigmoid and log loss", ch: "A3", href: "../a3-logistic-regression/#logloss", prompt: "Turn a score z into a chance of 'adult', then write the loss for one shark with label y.",
      tex: ["z = \\mathbf{w}\\cdot\\mathbf{x} + b \\qquad p = \\sigma(z) = \\frac{1}{1+e^{-z}}", "\\ell = -\\bigl[\\,y\\ln p + (1-y)\\ln(1-p)\\,\\bigr]"],
      notes: ["<b>\\(z\\)</b> is the score, <b>\\(\\mathbf{w}\\)</b> the weights, <b>\\(\\mathbf{x}\\)</b> the inputs and <b>\\(b\\)</b> the bias. <b>\\(\\sigma(z)\\)</b> squeezes the score into a chance <b>\\(p\\)</b> between 0 and 1.", "<b>\\(y\\)</b> is the label (1 for adult, 0 for juvenile). <b>\\(\\ell\\)</b> is the loss for one shark: large when the model is sure and wrong."],
      live: function (box) {
        var out = Kit.html("div", { class: "eqn" }, box), sl;
        function upd(z) { var p = 1 / (1 + Math.exp(-z)); out.innerHTML = ebox("c-navy", nf(z, 2), "score z") + eop("&rarr;") + ebox("c-teal", nf(p, 3), "p = &sigma;(z)") + eop("|") + ebox("c-coral", nf(-Math.log(p), 2), "loss if y = 1") + ebox("c-coral", nf(-Math.log(1 - p), 2), "loss if y = 0"); }
        sl = Kit.slider(box, { label: "Score z (3.48 is Tag 3305; 5.08 is Tag 4127 under the spadenose model)", min: -6, max: 6, step: 0.01, value: 5.08, digits: 2 }, upd); box.insertBefore(sl.input.parentNode, out); upd(5.08);
      } },
    { id: "gd", name: "Gradient descent update", ch: "A2, B3", href: "../a2-linear-regression/#descent", prompt: "Write the update that moves a weight downhill, and say what the learning rate does.",
      tex: ["w \\leftarrow w - \\eta\\,\\frac{\\partial L}{\\partial w}"],
      notes: ["<b>\\(w\\)</b> is a weight and <b>\\(L\\)</b> the loss. <b>\\(\\partial L/\\partial w\\)</b> is the slope of the loss along that weight.", "<b>\\(\\eta\\)</b> is the learning rate, the size of the step. Too small is slow, too large overshoots (A2, B5)."],
      live: function (box) {
        var out = Kit.html("div", { class: "eqn" }, box), sl;
        function upd(e) { var g = -0.682; out.innerHTML = ebox("c-navy", "&minus;4", "w before") + eop("&minus;") + ebox("c-sand", nf(e, 2), "&eta;") + eop("&times;") + ebox("c-coral", "&minus;0.682", "slope of L") + eop("=") + ebox("c-teal", nf(-4 - e * g, 3), "w after"); }
        sl = Kit.slider(box, { label: "Learning rate η (B3: one neuron on 268 spadenose sharks, start w = −4, slope −0.682)", min: 0.05, max: 2, step: 0.05, value: 0.5, digits: 2 }, upd); box.insertBefore(sl.input.parentNode, out); upd(0.5);
      } },
    { id: "chain", name: "Chain rule through a network", ch: "B4", href: "../b4-backpropagation/#backward", prompt: "For the tiny network of B4, write the slope of the loss with respect to a first-layer weight as a product of four factors.",
      tex: ["\\frac{\\partial L}{\\partial w_{ji}} = (p - y)\\;\\cdot\\; v_j\\;\\cdot\\; g'(z_j)\\;\\cdot\\; x_i"],
      notes: ["<b>\\(p-y\\)</b> is the slope of the loss at the output score. <b>\\(v_j\\)</b> is the output weight from hidden unit \\(j\\), and <b>\\(g'(z_j)\\)</b> the slope of its activation (for tanh, \\(1-h_j^2\\)).", "<b>\\(x_i\\)</b> is the input that the weight \\(w_{ji}\\) multiplies. The product follows the path from the loss back to the weight."],
      live: function (box) {
        var N = b4Net(), out = Kit.html("div", { class: "eqn" }, box), bar = Kit.html("div", { class: "bar" }, box); box.insertBefore(bar, out);
        var names = ["w11", "w12", "w21", "w22"], idx = [[0, 0], [0, 1], [1, 0], [1, 1]];
        function upd(i) { var j = idx[i][0], c = idx[i][1], d = N.p - B4.y, gp = 1 - N.h[j] * N.h[j], x = B4.x[c], g = d * B4.v[j] * gp * x;
          out.innerHTML = ebox("c-coral", nf(d, 4), "p &minus; y") + eop("&times;") + ebox("c-navy", nf(B4.v[j], 1), "v" + (j + 1)) + eop("&times;") + ebox("c-sand", nf(gp, 4), "1 &minus; h" + (j + 1) + "&sup2;") + eop("&times;") + ebox("c-teal", nf(x, 2), "x" + (c + 1)) + eop("=") + ebox("", nf(g, 4), "slope of " + names[i]); }
        Kit.chips(bar, names, 0, upd); upd(0);
      } },
    { id: "att", name: "Attention", ch: "D4", href: "../d4-self-attention/#score", prompt: "Write attention in one line from the queries Q, the keys K and the values V.",
      tex: ["\\mathrm{Attention}(Q,K,V) = \\mathrm{softmax}\\!\\left(\\frac{QK^{\\top}}{\\sqrt{d_k}}\\right)V"],
      notes: ["<b>\\(Q\\)</b>, <b>\\(K\\)</b> and <b>\\(V\\)</b> contain the queries, keys and values of the tokens. <b>\\(QK^{\\top}\\)</b> is the dot product of all queries with all keys.", "<b>\\(d_k\\)</b> is the key size. Dividing by \\(\\sqrt{d_k}\\) keeps the scores from growing with it, so the softmax does not saturate. The result is a weighted sum of values."],
      live: function (box) {
        var dots = [0.11, 1.41, 1.98, 0.80], toks = ["the", "young", "hammerhead", "it"], out = Kit.html("div", { class: "eqn" }, box), bar = Kit.html("div", { class: "bar" }, box); box.insertBefore(bar, out);
        function upd(i) { var sc = i === 0 ? dots.map(function (v) { return v / Math.sqrt(2); }) : dots, w = softmax(sc);
          out.innerHTML = toks.map(function (t, j) { return ebox(j === 2 ? "c-coral" : "c-navy", pc(w[j]), t + " (score " + nf(sc[j], 2) + ")"); }).join("") + eop("&larr;") + '<span class="op" style="font-size:14px">weights of the query "it", d<sub>k</sub> = 2' + (i === 0 ? ", divided by &radic;2" : ", no division") + '</span>'; }
        Kit.chips(bar, ["Divide by √dₖ", "No division"], 0, upd); upd(0);
      } },
    { id: "clip", name: "Contrastive loss (CLIP)", ch: "E2", href: "../e2-clip/#matrix", prompt: "Write CLIP's loss for a batch of N photo and caption pairs, using the cosines between photos and captions.",
      tex: ["s_{ij} = \\cos(\\mathbf u_i, \\mathbf v_j)", "\\ell_i^{\\text{photo}} = -\\ln\\frac{e^{s_{ii}/T}}{\\sum_{j=1}^{N} e^{s_{ij}/T}}", "L = \\frac{1}{2N}\\sum_{i=1}^{N}\\left(\\ell_i^{\\text{photo}} + \\ell_i^{\\text{caption}}\\right)"],
      notes: ["<b>\\(\\mathbf u_i\\)</b> is the vector of photo \\(i\\) and <b>\\(\\mathbf v_j\\)</b> the vector of caption \\(j\\). <b>\\(s_{ij}\\)</b> is their cosine. <b>\\(N\\)</b> is the number of pairs in the batch.", "<b>\\(T\\)</b> is the temperature. <b>\\(\\ell_i^{\\text{caption}}\\)</b> is the same loss read down the column of caption \\(i\\). The loss is small when a photo is closest to its own caption and a caption is closest to its own photo."],
      live: function (box) {
        var out = Kit.html("div", { class: "eqn" }, box), bar = Kit.html("div", { class: "bar" }, box); box.insertBefore(bar, out); var Ts = [0.01, 0.1, 1];
        function upd(i) { var T = Ts[i], L = clipLoss(D.clip.cos_caption, T); out.innerHTML = ebox("c-navy", "14", "photo and caption pairs") + eop("&rarr;") + ebox("c-teal", nf(T, 2), "temperature T") + eop("&rarr;") + ebox("c-coral", nf(L, 2), "loss L") + eop("|") + ebox("", nf(Math.log(14), 2), "ln 14, loss of a blind guess"); }
        Kit.chips(bar, ["T = 0.01", "T = 0.1", "T = 1"], 0, upd); upd(0);
      } }
  ];
  function eqFig() {
    var grid = Kit.$("eq-grid"), sum = Kit.$("eq-summary"), rating = {}, cards = [];
    function upd() {
      var g = 0, p = 0, m = 0, miss = [];
      EQS.forEach(function (E) { var r = rating[E.id]; if (r === "got") g++; else if (r === "part") { p++; miss.push(E); } else if (r === "miss") { m++; miss.push(E); } });
      sum.innerHTML = '<span>Had it: <b>' + g + '</b></span><span>Some of it: <b>' + p + '</b></span><span>Missed: <b>' + m + '</b></span><span>Not yet rated: <b>' + (EQS.length - g - p - m) + '</b></span>' +
        (miss.length ? '<span>Revisit: ' + miss.map(function (E) { return link(E.href, esc(E.ch)); }).join(", ") + '</span>' : '');
    }
    Kit.button(Kit.$("eq-bar"), "Hide all", function () { cards.forEach(function (c) { c.hide(); }); });
    Kit.button(Kit.$("eq-bar"), "Shuffle the order", function () { var arr = cards.slice().sort(function () { return Math.random() - 0.5; }); arr.forEach(function (c) { grid.appendChild(c.el); }); });
    EQS.forEach(function (E, n) {
      var el = Kit.html("div", { class: "eqcard" }, grid);
      el.innerHTML = '<h3>' + (n + 1) + '. ' + esc(E.name) + ' <span class="ch">' + link(E.href, esc(E.ch)) + '</span></h3><p>' + esc(E.prompt) + '</p><textarea aria-label="Your attempt at ' + esc(E.name) + ', not stored" placeholder="Write it here. The text is not saved."></textarea><div class="acts"><button type="button" class="btn rv">Reveal</button></div><div class="shown" hidden></div>';
      var shown = el.querySelector(".shown"), btn = el.querySelector(".rv"), built = false;
      function show() {
        if (!built) {
          built = true;
          shown.innerHTML = E.tex.map(function (q) { return '<div class="math">' + tex(q, true) + '</div>'; }).join("") + '<ul class="math-notes">' + E.notes.map(function (t) { return '<li>' + t.replace(/\\\((.+?)\\\)/g, function (_, m) { return tex(m, false); }) + '</li>'; }).join("") + '</ul><div class="live"></div><div class="acts rate"><span class="rate-res">How close were you?</span></div>';
          E.live(shown.querySelector(".live"));
          var acts = shown.querySelector(".rate");
          [["got", "Had it"], ["part", "Some of it"], ["miss", "Missed it"]].forEach(function (r) {
            var b = Kit.html("button", { type: "button", class: "btn", "aria-pressed": "false" }, acts, r[1]);
            b.addEventListener("click", function () { rating[E.id] = r[0]; acts.querySelectorAll("button").forEach(function (x) { x.setAttribute("aria-pressed", String(x === b)); }); upd(); });
          });
        }
        shown.hidden = false; btn.textContent = "Hide";
      }
      function hide() { shown.hidden = true; btn.textContent = "Reveal"; }
      btn.addEventListener("click", function () { if (shown.hidden) show(); else hide(); });
      cards.push({ el: el, hide: hide });
    });
    upd();
  }

  // =====================================================================================
  // 5. Book map and projects
  // =====================================================================================
  var PARTS = [
    ["Toolkit", [["t1", "0.1"], ["t2", "0.2"], ["t3", "0.3"], ["t4", "0.4"], ["t5", "0.5"], ["t6", "0.6"], ["t7", "0.7"], ["t8", "0.8"]]],
    ["A Learning from data", [["a1", "A1"], ["a2", "A2"], ["a3", "A3"], ["a4", "A4"], ["a5", "A5"], ["a6", "A6"]]],
    ["B Neural networks", [["b1", "B1"], ["b2", "B2"], ["b3", "B3"], ["b4", "B4"], ["b5", "B5"], ["b6", "B6"], ["b7", "B7"], ["b8", "B8"], ["b9", "B9"]]],
    ["C Images", [["c1", "C1"], ["c2", "C2"], ["c3", "C3"]]],
    ["D Language", [["d1", "D1"], ["d2", "D2"], ["d3", "D3"], ["d4", "D4"], ["d5", "D5"], ["d6", "D6"], ["d7", "D7"]]],
    ["E Vision and language", [["e1", "E1"], ["e2", "E2"], ["e3", "E3"], ["e4", "E4"]]],
    ["F AI in the world", [["f1", "F1"], ["f2", "F2"], ["f3", "F3"]]]
  ];
  var SLUG = { t1: "t1-python", t2: "t2-algorithms", t3: "t3-algebra-functions", t4: "t4-discrete-maths", t5: "t5-vectors-matrices", t6: "t6-gradients", t7: "t7-probability", t8: "t8-charts",
    a1: "a1-what-is-a-model", a2: "a2-linear-regression", a3: "a3-logistic-regression", a4: "a4-grading-models", a5: "a5-decision-trees", a6: "a6-boosting",
    b1: "b1-one-neuron", b2: "b2-non-linearity", b3: "b3-loss-gradient-descent", b4: "b4-backpropagation", b5: "b5-optimisers", b6: "b6-deep-networks", b7: "b7-continual-learning", b8: "b8-debugging", b9: "b9-explaining",
    c1: "c1-convolution", c2: "c2-cnns", c3: "c3-domain-shift", d1: "d1-tokenisation", d2: "d2-embeddings", d3: "d3-next-token", d4: "d4-self-attention", d5: "d5-transformer", d6: "d6-long-context", d7: "d7-assistants",
    e1: "e1-vision-transformers", e2: "e2-clip", e3: "e3-cross-attention", e4: "e4-vlms", f1: "f1-fairness", f2: "f2-robots", f3: "f3-sharkcam" };
  var TITLE = { t1: "Python", t2: "Algorithms and data structures", t3: "Algebra and functions", t4: "Discrete maths and logic", t5: "Vectors and matrices", t6: "Slopes and gradients", t7: "Probability and statistics", t8: "Reading charts and tables",
    a1: "What a model is", a2: "Linear regression", a3: "Logistic regression and softmax", a4: "Grading a model fairly", a5: "Decision trees", a6: "Ensembles and boosting",
    b1: "One neuron", b2: "Why non-linearity matters", b3: "Loss and gradient descent", b4: "Backpropagation", b5: "Optimisers", b6: "Making deep networks trainable", b7: "Fine-tuning, forgetting and continual learning", b8: "Debugging neural networks", b9: "Explaining a network",
    c1: "Images as numbers and convolution", c2: "Convolutional networks", c3: "When the water changes", d1: "Tokenisation", d2: "Embeddings", d3: "Predicting the next token", d4: "Self-attention", d5: "The transformer", d6: "Long context", d7: "From language model to assistant",
    e1: "Vision transformers and image tokens", e2: "Encoders and CLIP", e3: "Cross-attention", e4: "Vision-language models", f1: "Data, fairness and privacy", f2: "AI on a robot", f3: "SharkCam, end to end" };
  var PROJECTS = [
    { id: "dia", name: "Diabetes screening", chapters: { a2: "fit a line and read the misses", a3: "a sigmoid and a log loss for a yes-or-no risk", a4: "grade the model by threshold, ROC and subgroup", a5: "trees on the clinic columns", a6: "XGBoost and feature importance", b1: "the neuron behind the small network", b2: "hidden layers", b3: "loss and gradient descent for networks", f1: "who is in the data and who is missing" },
      facts: ["I built and deployed a diabetes risk screening app on the Pima dataset, 768 women of Pima heritage measured in the United States.", "The final model in my report was XGBoost with feature-importance explanations, and I checked results across demographic subgroups for bias. The deployed web app runs a small fastai neural network.", "A rebuilt XGBoost model with 100 trees of depth 3, a learning rate of 0.01 and an 80/20 split gets 78.6% of 154 held-out women right.", "My lesson: a model trained on one population may not suit another. The app flagged my mother, who is not from that population."],
      links: [["Project page", "https://rayhaan1345.github.io/projects/diabetes-screening/"], ["Primer", "https://rayhaan1345.github.io/projects/diabetes-screening/primer/"]] },
    { id: "opt", name: "Optimizer state and Fisher", chapters: { b5: "Adam's m and v, and what is lost without the optimiser state", b7: "the Fisher information, elastic weight consolidation and v as a Fisher proxy" },
      facts: ["Public checkpoints share a model's weights but usually not the optimizer state, so continued training often starts unstable. The work asks whether that state can be rebuilt from public data. It is a first-author manuscript under review at TMLR, done at ADIA Lab.", "AdamW keeps a first moment (m) and a second moment (v) per weight. The paper rebuilds v from public data and uses it as a Fisher proxy: v tracks squared gradients and is related to the diagonal of the Fisher information. This also helps limit forgetting through an EWC-style penalty.", "I wrote a pipeline that logged optimizer buffers while a CNN trained on CIFAR-10 and measured how closely a rebuilt state matched the original. The work was extended to language models (GPT-2, OLMo-2)."],
      links: [["Project page", "https://rayhaan1345.github.io/projects/optimizer-state/"], ["Primer", "https://rayhaan1345.github.io/projects/optimizer-state/primer/"]] },
    { id: "ttt", name: "In-Place TTT and the VLM port", chapters: { c3: "domain shift and test-time training for the shark", d4: "attention", d5: "the MLP block and its down projection", d6: "test-time training and long context", e4: "vision-language models" },
      facts: ["Test-time training updates some of a model's weights while it is being used. In-Place TTT, from ByteDance Seed, treats the down projection of selected MLP blocks as fast weights. It updates them chunk by chunk as the model reads, with a target built from the input. Generation never pauses.", "I am porting the method from text models to a Qwen vision-language model, so that it can adapt to an input after deployment.", "My motivation is the question of this book: recognising a shark when the water, the light and the animals change between dives. This work is at MBZUAI, in Prof. Salman Khan's group."],
      links: [["Project page", "https://rayhaan1345.github.io/projects/in-place-ttt/"], ["Primer", "https://rayhaan1345.github.io/projects/in-place-ttt/primer/"]] },
    { id: "soc", name: "Socratic Tutor", chapters: { d7: "prompts, tools and skills" },
      facts: ["The Socratic Tutor is an open-source plugin from Mumbai AI Lab that makes Claude tutor by asking questions. It has two skills, socratic-tutor and quiz-me."],
      links: [["Report", "https://mailab.in/socratic-tutor/"]] },
    { id: "rob", name: "Robot", chapters: { f2: "the loop, the latency budget, memory and quantisation" },
      facts: ["I have built a small robot that ran a vision model on a Raspberry Pi."], links: [] }
  ];
  function mapFig() {
    var bar = Kit.$("mp-bar"), tiles = Kit.$("mp-tiles"), info = Kit.$("mp-info"), cur = 0, selTile = null, tileEls = {};
    var chips = Kit.chips(bar, PROJECTS.map(function (p) { return p.name; }), 0, function (i) { cur = i; selTile = null; upd(); });
    PARTS.forEach(function (P) {
      var row = Kit.html("div", { class: "bm-row" }, tiles); Kit.html("span", {}, row, P[0]);
      var tw = Kit.html("div", { class: "bm-tiles" }, row);
      P[1].forEach(function (c) {
        var b = Kit.html("button", { type: "button", class: "tile", title: TITLE[c[0]] }, tw, c[1]); tileEls[c[0]] = b;
        b.addEventListener("click", function () { selTile = c[0]; upd(); });
      });
    });
    function upd() {
      var p = PROJECTS[cur];
      Object.keys(tileEls).forEach(function (id) { var b = tileEls[id]; b.classList.toggle("lit", !!p.chapters[id]); b.classList.toggle("sel", selTile === id); b.setAttribute("aria-pressed", String(!!p.chapters[id])); b.setAttribute("aria-label", TITLE[id] + (p.chapters[id] ? ", used by " + p.name : "")); });
      var h = "";
      if (selTile) {
        var users = PROJECTS.filter(function (q) { return q.chapters[selTile]; });
        h += '<h3>' + link("../" + SLUG[selTile] + "/", esc(TITLE[selTile])) + '</h3>' + (users.length ? '<p>Used by: ' + users.map(function (q) { return '<b>' + esc(q.name) + '</b> (' + esc(q.chapters[selTile]) + ')'; }).join("; ") + '.</p>' : '<p>None of the five projects lists this chapter. It supports the chapters around it.</p>');
        h += '<p class="mini-note">Pick a project chip to go back to its chapters.</p>';
      } else {
        h += '<h3>' + esc(p.name) + '</h3>' + p.facts.map(function (t) { return '<p>' + esc(t) + '</p>'; }).join("") + '<p><b>Chapters behind it</b></p><ul>' + Object.keys(p.chapters).map(function (id) { return '<li>' + link("../" + SLUG[id] + "/", esc(PARTS.reduce(function (a, P) { var f = P[1].filter(function (c) { return c[0] === id; })[0]; return f ? f[1] : a; }, id)) + " " + esc(TITLE[id])) + ': ' + esc(p.chapters[id]) + '</li>'; }).join("") + '</ul>';
        if (p.links.length) h += '<p>' + p.links.map(function (l) { return '<a href="' + l[1] + '" rel="noopener">' + esc(l[0]) + '</a>'; }).join(" &middot; ") + '</p>';
      }
      info.innerHTML = h;
    }
    upd();
  }

  // =====================================================================================
  // 6. What next: a short quiz
  // =====================================================================================
  function nextQuiz() {
    Kit.quiz("nx-quiz", [
      { q: "SharkCam should find the nearest known photo of a shark for any new photo. Which left-out topic is closest?", o: ["Generative adversarial networks", "k-nearest neighbours", "Mixture of experts"], a: 1, why: "A nearest-neighbour search compares a new vector with stored ones, as D2 and E2 did with word and photo vectors." },
      { q: "The team wants to make clear-water photos look like murky-water photos to train on, without a physics model. Which topic fits?", o: ["Support vector machines", "Principal component analysis", "Diffusion and image generation"], a: 2, why: "Image generation models learn to produce new photos. The depth model of C3 is a hand-built version of the same job." },
      { q: "The robot should learn when to move its camera from trial and error, without labelled examples. Which topic fits?", o: ["Distributed training", "Reinforcement learning", "k-nearest neighbours"], a: 1, why: "Reinforcement learning trains a policy from rewards, with no labelled answer per step." }
    ]);
  }

  // ---------- start ----------
  document.addEventListener("DOMContentLoaded", function () {
    Promise.all([Kit.data("data/f3.json"), Kit.data("data/bamboo_depth.json")]).then(function (r) {
      D = r[0]; BD = r[1];
      var run = function (name, fn) { try { fn(); } catch (e) { console.error(name, e); } };
      run("sys", sysFig); run("budget", budgetFig); run("trace", traceFig); run("fail", failFig); run("eq", eqFig); run("map", mapFig); run("next", nextQuiz);
    }).catch(function (e) { console.error("F3 data failed to load", e); });
  });
})();
