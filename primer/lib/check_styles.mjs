#!/usr/bin/env node
// Style regression check for the primer. Needs Node 22 and Google Chrome. No other dependencies.
//
// It loads a page in headless Chrome, waits for the figures to draw, and records the computed style of all
// elements. An element is stored as a stable path (tag plus nth-child chain from <html>) and a hash of its full
// getComputedStyle, with the ::before and ::after styles when they draw something. Console errors are listed too.
// Two runs can then be compared, which shows whether a change to the library CSS left all elements as they were.
//
// One page, printed as JSON:
//   node check_styles.mjs http://127.0.0.1:8891/primer/b1-one-neuron/ [--width 1100] [--dark] [--mobile] [--wait 9000] [--only PATH,PATH]
//
// Snapshot all pages into a folder (one JSON file per page and state):
//   node check_styles.mjs --snap http://127.0.0.1:8891 OUTDIR [--jobs 4] [--pages b1-one-neuron,index] [--states light,dark,mobile]
//
// Compare two servers (for example the old copy on 8890 and the new copy on 8891) in one go:
//   node check_styles.mjs --compare http://127.0.0.1:8890 http://127.0.0.1:8891 [--jobs 4] [--out DIR] [--retries 2]
//
// Compare two snapshot folders that are already on disk:
//   node check_styles.mjs --diff DIR_A DIR_B
//
// Property-level differences for one page and state (loads both servers twice: once to find the differing
// elements, once to read their styles):
//   node check_styles.mjs --detail http://127.0.0.1:8890 http://127.0.0.1:8891 b1-one-neuron dark [--limit 40]
//
// States: light = width 1100 with a light scheme; dark = width 1100 with prefers-color-scheme dark; mobile = width 390, light.
// The default is light,dark,mobile. Add print (width 800, print media) with --states light,dark,mobile,print.
// Pages are the folders next to this script's parent that contain an index.html, plus the primer index ("index").
// Start the servers yourself, in the folder that contains primer/, for example: python3 -m http.server 8890
//
// Notes on what is and is not compared:
// - The contents of <head> are skipped. The head contains links and scripts, which have no visible style, and a build
//   changes them. <html> and the whole body are recorded.
// - Motion is forced to no-preference so figures that autoplay behave the same in all runs.
// - Google web fonts are blocked (pass --webfonts to allow them). Their load time changes where Plotly puts labels,
//   which makes two runs of one page differ by a few SVG elements. Fallback fonts give the same layout in all runs.
// - Ids that Plotly makes up at random (clip paths) are replaced by a fixed text before hashing.
// - After a first pass, --compare loads the page states that differ again, up to --retries times, and reports only
//   the elements that differ in all passes. A genuine difference repeats. Timing noise does not.
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, readdirSync, readFileSync, existsSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const HERE = dirname(fileURLToPath(import.meta.url));
const STATES = {
  light: { width: 1100, dark: false, mobile: false },
  dark: { width: 1100, dark: true, mobile: false },
  mobile: { width: 390, dark: false, mobile: true },
  print: { width: 800, dark: false, mobile: false, print: true },   // optional: checks the @media print rules
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------- command line ----------
const argv = process.argv.slice(2);
const flag = (name) => argv.includes("--" + name);
const opt = (name, dflt) => { const i = argv.indexOf("--" + name); return i >= 0 && i + 1 < argv.length ? argv[i + 1] : dflt; };
const VALUE_OPTIONS = ["width", "wait", "jobs", "pages", "states", "out", "root", "only", "limit", "retries"];
const positional = () => { const out = []; for (let i = 0; i < argv.length; i++) { if (argv[i].startsWith("--")) { if (VALUE_OPTIONS.includes(argv[i].slice(2))) i++; continue; } out.push(argv[i]); } return out; };

// ---------- the in-page recorder ----------
// Called with null, or with a list of paths whose full styles should be returned as well.
const RECORDER = `(function (only) {
  function h53(s) {
    var h1 = 0xdeadbeef, h2 = 0x41c6ce57, ch;
    for (var i = 0; i < s.length; i++) { ch = s.charCodeAt(i); h1 = Math.imul(h1 ^ ch, 2654435761); h2 = Math.imul(h2 ^ ch, 1597334677); }
    h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
    h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
    return (4294967296 * (2097151 & h2) + (h1 >>> 0)).toString(16);
  }
  var ID = /url\\(["']?#[^)"']*["']?\\)/g;
  function props(cs) { var o = {}; for (var i = 0; i < cs.length; i++) { var n = cs[i]; o[n] = cs.getPropertyValue(n).replace(ID, "url(#)"); } return o; }
  function text(o) { var s = "", ks = Object.keys(o).sort(); for (var i = 0; i < ks.length; i++) s += ks[i] + ":" + o[ks[i]] + ";"; return s; }
  var styles = {}, detail = {}, desc = {}, count = 0, onlySet = only ? new Set(only) : null;
  function visit(el, path) {
    count++;
    var parts = { "": props(getComputedStyle(el)) };
    ["::before", "::after"].forEach(function (pe) {
      var cs = getComputedStyle(el, pe);
      if (cs.content && cs.content !== "none" && cs.content !== "normal") parts[pe] = props(cs);
    });
    var s = ""; for (var pe in parts) s += "|" + pe + "|" + text(parts[pe]);
    styles[path] = h53(s);
    if (onlySet && onlySet.has(path)) {
      detail[path] = parts;
      desc[path] = el.tagName.toLowerCase() + (el.id ? "#" + el.id : "") + (typeof el.className === "string" && el.className ? "." + el.className.trim().split(/\\s+/).join(".") : "");
    }
    var kids = el.children;
    for (var i = 0; i < kids.length; i++) {
      if (el === document.documentElement && kids[i].tagName === "HEAD") continue;
      visit(kids[i], path + ">" + kids[i].tagName.toLowerCase() + ":" + (i + 1));
    }
  }
  visit(document.documentElement, "html");
  return { elements: count, styles: styles, detail: onlySet ? detail : undefined, desc: onlySet ? desc : undefined };
})`;

// ---------- Chrome ----------
let nextPort = 9300 + Math.floor(Math.random() * 400);
async function launch() {
  const port = nextPort++;
  const dir = mkdtempSync(join(tmpdir(), "primer-check-"));
  const chrome = spawn(CHROME, ["--headless=new", `--remote-debugging-port=${port}`, "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
    "--hide-scrollbars", "--no-first-run", `--user-data-dir=${dir}`, "--window-size=1100,900", "about:blank"], { stdio: "ignore" });
  let ws;
  for (let k = 0; k < 75 && !ws; k++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      const page = list.find((t) => t.type === "page");
      if (page) { const s = new WebSocket(page.webSocketDebuggerUrl); await new Promise((r, j) => { s.onopen = r; s.onerror = j; }); ws = s; }
    } catch (e) { await sleep(200); }
  }
  if (!ws) { chrome.kill("SIGKILL"); throw new Error("Chrome did not start (is it installed at " + CHROME + "?)"); }
  let id = 0; const pending = new Map(); let errors = [], warnings = 0;
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d.result || d.error); pending.delete(d.id); return; }
    if (d.method === "Runtime.exceptionThrown") errors.push("exception: " + (d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text).split("\n")[0].slice(0, 200));
    else if (d.method === "Runtime.consoleAPICalled") { if (d.params.type === "error") errors.push("console.error: " + d.params.args.map((a) => a.value ?? a.description).join(" ").slice(0, 200)); else if (d.params.type === "warning") warnings++; }
    else if (d.method === "Log.entryAdded") {
      const e = d.params.entry;
      if (e.level === "error" && /ERR_BLOCKED_BY_CLIENT/.test(e.text + (e.url || ""))) { /* the web fonts blocked on purpose */ }
      else if (e.level === "error") errors.push("log: " + e.text.slice(0, 160) + (e.url ? " " + e.url : ""));
      else if (e.level === "warning") warnings++;
    }
  };
  const send = (method, params = {}) => new Promise((resolve) => { const i = ++id; pending.set(i, resolve); ws.send(JSON.stringify({ id: i, method, params })); });
  await send("Runtime.enable"); await send("Log.enable"); await send("Page.enable");
  if (!flag("webfonts")) { await send("Network.enable"); await send("Network.setBlockedURLs", { urls: ["*://fonts.googleapis.com/*", "*://fonts.gstatic.com/*"] }); }
  return {
    send,
    take() { const e = [...new Set(errors)]; const w = warnings; errors = []; warnings = 0; return { errors: e, warnings: w }; },
    async close() { try { ws.close(); } catch (e) {} chrome.kill("SIGKILL"); await sleep(100); try { rmSync(dir, { recursive: true, force: true }); } catch (e) {} },
  };
}

async function record(browser, url, state, { wait = 9000, only = null } = {}) {
  const s = typeof state === "string" ? STATES[state] : state;
  if (!s) throw new Error("unknown state " + state);
  await browser.send("Emulation.setDeviceMetricsOverride", { width: s.width, height: 900, deviceScaleFactor: 1, mobile: s.mobile });
  await browser.send("Emulation.setEmulatedMedia", { media: s.print ? "print" : "", features: [{ name: "prefers-color-scheme", value: s.dark ? "dark" : "light" }, { name: "prefers-reduced-motion", value: "no-preference" }] });
  await browser.send("Page.navigate", { url: "about:blank" });
  browser.take();
  await browser.send("Page.navigate", { url });
  await sleep(wait);
  const r = await browser.send("Runtime.evaluate", { expression: RECORDER + "(" + JSON.stringify(only) + ")", returnByValue: true, timeout: 60000 });
  if (!r.result || r.exceptionDetails) throw new Error("recorder failed: " + JSON.stringify(r.exceptionDetails || r).slice(0, 300));
  const { errors, warnings } = browser.take();
  return { url, state: typeof state === "string" ? state : "custom", ...r.result.value, errors, warnings };
}

// ---------- page list ----------
function pageList(root) {
  const dir = resolve(root || join(HERE, ".."));
  const out = ["index"];
  for (const name of readdirSync(dir, { withFileTypes: true }).filter((d) => d.isDirectory()).map((d) => d.name).sort()) {
    if (["assets", "data", "lib", "_template"].includes(name) || name.startsWith(".")) continue;
    if (existsSync(join(dir, name, "index.html"))) out.push(name);
  }
  return out;
}
const pageUrl = (base, page) => base.replace(/\/$/, "") + "/primer/" + (page === "index" ? "" : page + "/");

// ---------- running many ----------
// tasks: array of [page, state]. Writes OUTDIR/<page>.<state>.json. Returns the list of failures.
async function snapTasks(base, outDir, tasks, jobs, wait, label) {
  mkdirSync(outDir, { recursive: true });
  let next = 0, done = 0; const failed = [];
  async function worker() {
    const b = await launch();
    try {
      while (next < tasks.length) {
        const [p, s] = tasks[next++];
        const file = join(outDir, p + "." + s + ".json");
        for (let attempt = 0; attempt < 2; attempt++) {
          try { const rec = await record(b, pageUrl(base, p), s, { wait }); writeFileSync(file, JSON.stringify({ page: p, ...rec })); break; }
          catch (e) { if (attempt) failed.push(p + " " + s + ": " + e.message); }
        }
        done++; if (done % 10 === 0) process.stderr.write(`  ${label || base}: ${done}/${tasks.length}\n`);
      }
    } finally { await b.close(); }
  }
  await Promise.all(Array.from({ length: Math.max(1, Math.min(jobs, tasks.length)) }, worker));
  return failed;
}
const allTasks = (pages, states) => pages.flatMap((p) => states.map((s) => [p, s]));

// Compare one file from two folders. Returns sets of differing and one-sided paths.
function compareFile(a, b, f) {
  const A = JSON.parse(readFileSync(join(a, f), "utf8")), B = JSON.parse(readFileSync(join(b, f), "utf8"));
  const ka = Object.keys(A.styles);
  return {
    elementsA: A.elements, elementsB: B.elements, errorsA: A.errors, errorsB: B.errors,
    diff: new Set(ka.filter((k) => k in B.styles && A.styles[k] !== B.styles[k])),
    onlyA: new Set(ka.filter((k) => !(k in B.styles))), onlyB: new Set(Object.keys(B.styles).filter((k) => !(k in A.styles))),
  };
}
const intersect = (x, y) => new Set([...x].filter((v) => y.has(v)));
function jsonFiles(dir) { return readdirSync(dir).filter((f) => f.endsWith(".json") && f !== "diff.json").sort(); }

function summarise(rows) {
  const bad = rows.filter((r) => r.missing || r.diff.size || r.onlyA.size || r.onlyB.size);
  return { total: rows.length, same: rows.length - bad.length, errorPages: rows.filter((r) => r.errorsB && r.errorsB.length).length };
}
function printRows(rows, title) {
  const sum = summarise(rows);
  console.log(`\n${title}: ${sum.total} page states compared, ${sum.same} identical, ${sum.total - sum.same} with differences, ${sum.errorPages} with console errors in the second set`);
  for (const r of rows) {
    if (r.missing) { console.log("  MISSING in second set: " + r.name); continue; }
    if (r.diff.size || r.onlyA.size || r.onlyB.size) console.log(`  DIFF ${r.name}: ${r.diff.size} elements differ, ${r.onlyA.size} only in first, ${r.onlyB.size} only in second (elements ${r.elementsA} vs ${r.elementsB})\n       e.g. ${[...r.diff, ...r.onlyA, ...r.onlyB].slice(0, 3).join("\n            ")}`);
    if (r.errorsB.length) console.log(`  ERRORS ${r.name}: ${r.errorsB.join(" | ").slice(0, 300)}`);
    if (r.errorsA.length && JSON.stringify(r.errorsA) !== JSON.stringify(r.errorsB)) console.log(`  (errors in first set: ${r.errorsA.join(" | ").slice(0, 200)})`);
  }
  return sum;
}
function diffFolders(a, b) {
  return jsonFiles(a).map((f) => existsSync(join(b, f)) ? { name: f.replace(/\.json$/, ""), ...compareFile(a, b, f) } : { name: f.replace(/\.json$/, ""), missing: true });
}
const rowsForJson = (rows) => rows.map((r) => r.missing ? r : { ...r, diff: [...r.diff].slice(0, 200), onlyA: [...r.onlyA].slice(0, 50), onlyB: [...r.onlyB].slice(0, 50), nDiff: r.diff.size });

// ---------- property-level detail ----------
function propDiff(A, B) {
  const out = [];
  for (const path of Object.keys(A.detail)) {
    const pa = A.detail[path], pb = B.detail[path] || {}, changes = [];
    for (const pe of new Set([...Object.keys(pa), ...Object.keys(pb)])) {
      const x = pa[pe] || {}, y = pb[pe] || {};
      for (const k of new Set([...Object.keys(x), ...Object.keys(y)])) if (x[k] !== y[k]) changes.push(`${pe || "self"} ${k}: ${x[k]} -> ${y[k]}`);
    }
    out.push({ path, element: A.desc[path], changes });
  }
  return out;
}

// ---------- main ----------
async function main() {
  const wait = Number(opt("wait", 9000)), jobs = Number(opt("jobs", 3));
  const pos = positional();
  const pagesOpt = () => (opt("pages") ? opt("pages").split(",") : pageList(opt("root")));
  const statesOpt = () => (opt("states") || "light,dark,mobile").split(",");

  if (flag("snap")) {
    const [base, out] = pos; if (!base || !out) throw new Error("usage: --snap BASEURL OUTDIR");
    const failed = await snapTasks(base, resolve(out), allTasks(pagesOpt(), statesOpt()), jobs, wait);
    console.log(`snapshots written to ${resolve(out)}` + (failed.length ? "\nFAILED: " + failed.join("; ") : ""));
    process.exit(failed.length ? 2 : 0);
  }
  if (flag("diff")) {
    const [a, b] = pos; if (!a || !b) throw new Error("usage: --diff DIR_A DIR_B");
    const sum = printRows(diffFolders(resolve(a), resolve(b)), "diff");
    process.exit(sum.same === sum.total && !sum.errorPages ? 0 : 1);
  }
  if (flag("compare")) {
    const [baseA, baseB] = pos; if (!baseA || !baseB) throw new Error("usage: --compare BASEURL_A BASEURL_B");
    const out = resolve(opt("out", mkdtempSync(join(tmpdir(), "primer-compare-"))));
    const tasks = allTasks(pagesOpt(), statesOpt()), retries = Number(opt("retries", 2));
    const halfB = Math.max(1, Math.floor(jobs / 2)), halfA = Math.max(1, jobs - halfB);
    const run = async (pass, subset) => {
      const suffix = pass ? String(pass) : "";
      const fails = await Promise.all([snapTasks(baseA, join(out, "A" + suffix), subset, halfA, wait, "A" + suffix), snapTasks(baseB, join(out, "B" + suffix), subset, halfB, wait, "B" + suffix)]);
      return fails.flat();
    };
    let failed = await run(0, tasks);
    let rows = diffFolders(join(out, "A"), join(out, "B"));
    let cleared = 0;
    for (let pass = 1; pass <= retries; pass++) {
      const bad = rows.filter((r) => !r.missing && (r.diff.size || r.onlyA.size || r.onlyB.size));
      if (!bad.length) break;
      console.log(`pass ${pass}: loading the ${bad.length} differing page states again`);
      failed = failed.concat(await run(pass, bad.map((r) => r.name.split(".")).map(([p, s]) => [p, s])));
      for (const r of bad) {
        const f = r.name + ".json";
        if (!existsSync(join(out, "A" + pass, f)) || !existsSync(join(out, "B" + pass, f))) continue;
        const again = compareFile(join(out, "A" + pass), join(out, "B" + pass), f);
        const before = r.diff.size + r.onlyA.size + r.onlyB.size;
        r.diff = intersect(r.diff, again.diff); r.onlyA = intersect(r.onlyA, again.onlyA); r.onlyB = intersect(r.onlyB, again.onlyB);
        r.errorsA = [...new Set([...r.errorsA, ...again.errorsA])]; r.errorsB = [...new Set([...r.errorsB, ...again.errorsB])];
        if (before && !(r.diff.size + r.onlyA.size + r.onlyB.size)) cleared++;
      }
    }
    if (failed.length) console.log("FAILED loads: " + failed.join("; "));
    const sum = printRows(rows, "compare");
    if (cleared) console.log(`(${cleared} page states differed in the first pass only; the difference did not repeat, so it is timing noise)`);
    writeFileSync(join(out, "diff.json"), JSON.stringify(rowsForJson(rows), null, 1));
    console.log(`snapshots and diff.json are in ${out}`);
    process.exit(sum.same === sum.total && !sum.errorPages && !failed.length ? 0 : 1);
  }
  if (flag("detail")) {
    const [baseA, baseB, page, state = "light"] = pos; if (!baseA || !baseB || !page) throw new Error("usage: --detail BASEURL_A BASEURL_B PAGE [STATE]");
    const [ba, bb] = await Promise.all([launch(), launch()]);
    try {
      const [A0, B0] = await Promise.all([record(ba, pageUrl(baseA, page), state, { wait }), record(bb, pageUrl(baseB, page), state, { wait })]);
      const paths = Object.keys(A0.styles).filter((k) => k in B0.styles && A0.styles[k] !== B0.styles[k]);
      const onlyA = Object.keys(A0.styles).filter((k) => !(k in B0.styles)), onlyB = Object.keys(B0.styles).filter((k) => !(k in A0.styles));
      let details = [];
      if (paths.length) {
        const pick = paths.slice(0, Number(opt("limit", 40)));
        const [A, B] = await Promise.all([record(ba, pageUrl(baseA, page), state, { wait, only: pick }), record(bb, pageUrl(baseB, page), state, { wait, only: pick })]);
        details = propDiff(A, B).filter((d) => d.changes.length);
      }
      console.log(JSON.stringify({ page, state, differingFirstPass: paths.length, onlyInFirst: onlyA.slice(0, 10), onlyInSecond: onlyB.slice(0, 10), confirmedInSecondPass: details.length, details }, null, 1));
    } finally { await ba.close(); await bb.close(); }
    process.exit(0);
  }
  // one page, printed as JSON
  const url = pos[0];
  if (!url) { console.error(readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").slice(1, 34).join("\n")); process.exit(2); }
  const state = flag("dark") ? { width: Number(opt("width", 1100)), dark: true, mobile: flag("mobile") } : flag("mobile") ? { width: Number(opt("width", 390)), dark: false, mobile: true } : { width: Number(opt("width", 1100)), dark: false, mobile: false };
  const b = await launch();
  try { const rec = await record(b, url, state, { wait, only: opt("only") ? opt("only").split(",") : null }); console.log(JSON.stringify(rec)); } finally { await b.close(); }
  process.exit(0);
}
export { launch, record, RECORDER, STATES };
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch((e) => { console.error(e.message); process.exit(2); });
