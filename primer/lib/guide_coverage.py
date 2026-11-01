#!/usr/bin/env python3
"""Check that the style guide (primer/styleguide/page.html) shows the components of the library.

    python3 primer/lib/build.py --guide-coverage        (calls this module)
    python3 primer/lib/guide_coverage.py [-v]

It does three checks and writes no files.

1. Coverage. Every class that a selector in lib/css/*.css defines must appear in the guide. A class counts as
   "shown" when a demo (a real element of the page) carries it in a class attribute. It is "in markup only" when it
   appears in a class attribute inside an escaped markup block, but no demo carries it (for example classes that a
   Kit helper writes). It is "mentioned" when it appears as .name inside a code element. A class that is none of these
   is "missing", unless it is in INTERNAL below, with the reason that a page never writes it.
2. Sync. In a block that is not marked data-nosync, the markup block (the pre with code[data-hl=html]) must equal the
   demo before it (same tags, attributes and text; white space and the snippet tokens as written are compared after
   the same normalisation as lib/compare_pages.py). A writer who copies the markup then gets what the demo shows.
3. Ids. The ids of the guide's own elements must be unique.

Exit status 1 when a class is missing, a block is out of sync or an id repeats. Standard library only.
"""
import glob
import html
import os
import re
import sys
from html.parser import HTMLParser

LIB = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.path.dirname(LIB)
sys.dont_write_bytecode = True
sys.path.insert(0, LIB)
import compare_pages  # noqa: E402

GUIDE = os.path.join(PRIMER, "styleguide", "page.html")

# Classes that a page never writes. Keep this list short. Each entry needs a reason.
INTERNAL = {
    "kit-copy": "the copy button that Kit.js adds to every pre.code",
    "kit-run": "the run button that Kit.js adds to a pre.code with data-run",
    "katex": "added by KaTeX to the maths it typesets (the maths demos show it)",
    "katex-display": "added by KaTeX around a displayed formula (the maths demos show it)",
}


PRE = re.compile(r"<pre\b[^>]*>(.*?)</pre>", re.S)


def css_classes():
    """Return {class name: [css files]} for all classes in the selectors of lib/css/*.css."""
    found = {}
    for path in sorted(glob.glob(os.path.join(LIB, "css", "*.css"))):
        text = open(path, encoding="utf-8").read()
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
        for m in re.finditer(r"([^{}]+)\{", text):
            sel = m.group(1).strip()
            if sel.startswith("@"):
                continue
            for c in re.findall(r"\.([A-Za-z_][\w-]*)", sel):
                found.setdefault(c, set()).add(os.path.basename(path))
    return found


class Scan(HTMLParser):
    """Walks the guide page: classes and ids of real elements, and the demo and markup of each block."""

    def __init__(self, source):
        super().__init__(convert_charrefs=True)
        self.src = source
        self.starts = [0]
        for line in source.split("\n"):
            self.starts.append(self.starts[-1] + len(line) + 1)
        self.classes, self.ids, self.dup_ids = set(), set(), set()
        self.blocks = []            # dicts: id, demo (raw source), markup (text), nosync
        self.block = None
        self.demo_depth = 0
        self.demo_from = None
        self.in_markup = False

    def abs_pos(self):
        line, col = self.getpos()
        return self.starts[line - 1] + col

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        for c in (a.get("class") or "").split():
            self.classes.add(c)
        if a.get("id"):
            (self.dup_ids if a["id"] in self.ids else self.ids).add(a["id"])
        cls = (a.get("class") or "").split()
        if tag == "div" and "gd-block" in cls:
            self.block = {"id": a.get("id", "?"), "demo": None, "markup": None, "nosync": False}
            self.blocks.append(self.block)
        elif self.block is not None and tag == "div" and "gd-demo" in cls and self.demo_depth == 0:
            self.demo_depth = 1
            self.demo_from = self.abs_pos() + len(self.get_starttag_text())
            self.block["nosync"] = "data-nosync" in a
        elif self.demo_depth and tag == "div":
            self.demo_depth += 1
        if self.block is not None and tag == "code" and a.get("data-hl") == "html" and self.block["markup"] is None:
            self.in_markup = True
            self.block["markup"] = ""

    def handle_endtag(self, tag):
        if self.demo_depth and tag == "div":
            self.demo_depth -= 1
            if self.demo_depth == 0:
                self.block["demo"] = self.src[self.demo_from:self.abs_pos()]
        if tag == "code":
            self.in_markup = False

    def handle_data(self, data):
        if self.in_markup:
            self.block["markup"] += data


def analyse(path=GUIDE):
    src = open(path, encoding="utf-8").read()
    body = src.split("\n---\n", 1)[1] if src.startswith("---") else src
    scan = Scan(body)
    scan.feed(body)
    scan.close()
    in_markup = set()
    for m in re.finditer(r'class\s*=\s*"([^"]*)"', body):
        in_markup.update(m.group(1).split())
    mentioned = set()
    for m in re.finditer(r"<code>([^<]*)</code>", body):
        mentioned.update(re.findall(r"(?<![\w-])\.([A-Za-z_][\w-]*)", html.unescape(m.group(1))))
    return scan, in_markup, mentioned


def main(argv=None):
    args = argv if argv is not None else sys.argv[1:]
    verbose = "-v" in args
    if not os.path.isfile(GUIDE):
        print("guide coverage: %s does not exist" % os.path.relpath(GUIDE, PRIMER))
        return 1
    defined = css_classes()
    scan, in_markup, mentioned = analyse()
    shown = {c for c in defined if c in scan.classes}
    markup_only = {c for c in defined if c not in shown and c in in_markup}
    mention_only = {c for c in defined if c not in shown and c not in in_markup and c in mentioned}
    rest = {c for c in defined if c not in shown | markup_only | mention_only}
    internal = {c for c in rest if c in INTERNAL}
    missing = sorted(rest - internal)
    stale = sorted(c for c in INTERNAL if c not in defined)
    redundant = sorted(c for c in INTERNAL if c in defined and c not in rest)

    problems = []
    print("guide coverage: %d classes defined in lib/css" % len(defined))
    print("  shown in a demo        : %d" % len(shown))
    print("  in markup text only    : %d%s" % (len(markup_only), "  " + " ".join(sorted(markup_only)) if markup_only else ""))
    print("  mentioned in text only : %d%s" % (len(mention_only), "  " + " ".join(sorted(mention_only)) if mention_only else ""))
    print("  internal (see INTERNAL): %d" % len(internal))
    for c in sorted(internal):
        print("      .%-14s %s" % (c, INTERNAL[c]))
    print("  missing                : %d%s" % (len(missing), "  " + " ".join(missing) if missing else ""))
    if missing:
        problems.append("%d classes are not in the guide. Add a demo to styleguide/page.html, or add the class to INTERNAL in lib/guide_coverage.py with a reason" % len(missing))
    for c in stale:
        problems.append("INTERNAL lists .%s, which no style defines any more. Remove it" % c)
    for c in redundant:
        print("  note: .%s is in INTERNAL but the guide shows it. Remove it from INTERNAL" % c)

    out_of_sync = []
    checked = 0
    for b in scan.blocks:
        if b["nosync"] or b["demo"] is None or b["markup"] is None:
            if b["demo"] is not None and b["markup"] is None and not b["nosync"]:
                out_of_sync.append((b["id"], "no markup block"))
            continue
        checked += 1
        if compare_pages.canon(b["demo"]) != compare_pages.canon(b["markup"]):
            out_of_sync.append((b["id"], "the markup block differs from the demo"))
        elif [html.unescape(x) for x in PRE.findall(b["demo"])] != [html.unescape(x) for x in PRE.findall(b["markup"])]:
            out_of_sync.append((b["id"], "white space inside a pre differs between the demo and the markup block (do not indent the lines of a pre)"))
    print("  blocks                 : %d, %d checked for demo and markup in sync" % (len(scan.blocks), checked))
    for bid, why in out_of_sync:
        problems.append("block #%s: %s" % (bid, why))
    for i in sorted(scan.dup_ids):
        problems.append("the id %r is used twice in the guide" % i)

    if verbose:
        print("\nclasses by file that the guide shows only by markup or mention:")
        for c in sorted(markup_only | mention_only):
            print("  .%-14s %s" % (c, ", ".join(sorted(defined[c]))))
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print("  " + p)
        return 1
    print("  ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
