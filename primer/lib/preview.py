#!/usr/bin/env python3
"""Make a preview copy of the primer that search engines ignore and that links back to the live version.

    python3 primer/lib/preview.py OUTDIR

Steps:
  1. Runs the build check (python3 primer/lib/build.py --check). A stale build stops the preview.
  2. Copies primer/ to OUTDIR (created, or replaced if it contains an earlier preview). lib/, the page.html
     sources, .DS_Store and __pycache__ are left out.
  3. In the copy only, adds <meta name="robots" content="noindex"> and a canonical link to the live page
     in the head of all pages, and a small fixed pill at the bottom right that links to the live page.

Serve OUTDIR from a folder named primer-v2 (or any name). Pages use relative links, so the folder name
does not matter. The script lists any absolute /primer/ reference it finds in the copy.
Standard library only.
"""
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True  # keep __pycache__ out of the repo
sys.path.insert(0, HERE)
import build  # noqa: E402

PRIMER = build.PRIMER
LIVE = "https://mailab.in/primer/"
MARKER = ".primer-preview"
SKIP_TOP = {"lib"}
SKIP_ANY = {".DS_Store", "__pycache__", "page.html"}   # page.html is the source of index.html, not a page of the preview

PILL = (
    '<a id="primer-preview-pill" href="{href}" style="position:fixed;right:14px;bottom:14px;z-index:2147483647;'
    "display:block;margin:0;padding:7px 13px;border:1px solid #5b6470;border-radius:999px;background:#fffdf7;"
    "background-image:none;color:#14181c;font:13px/1.3 -apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;"
    'letter-spacing:0;text-decoration:none;box-shadow:0 2px 8px rgba(0,0,0,0.3);transition:none;">'
    "Preview v2 &middot; open current version</a>\n"
)


def live_url(rel):
    rel = rel.replace(os.sep, "/")
    if rel == "index.html":
        return LIVE
    if rel.endswith("/index.html"):
        return LIVE + rel[: -len("index.html")]
    return LIVE + rel


def ignore(src, names):
    skipped = set(n for n in names if n in SKIP_ANY)
    if os.path.abspath(src) == PRIMER:
        skipped |= set(n for n in names if n in SKIP_TOP)
    return skipped


def inject(path, rel):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    url = live_url(rel)
    head = '<link rel="canonical" href="%s">\n' % url
    if 'name="robots"' not in text:          # the style guide already says noindex
        head = '<meta name="robots" content="noindex">\n' + head
    if "</head>" not in text or "</body>" not in text:
        return False
    text = text.replace("</head>", head + "</head>", 1)
    i = text.rfind("</body>")
    text = text[:i] + PILL.format(href=url) + text[i:]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    return True


def absolute_refs(root):
    pat = re.compile(r"""(?:href|src|action)\s*=\s*["']/(?!/)|["'`]/primer/|https?://(?:www\.)?mailab\.in/primer""", re.I)
    found = []
    for dirpath, _, files in os.walk(root):
        for name in files:
            if not name.endswith((".html", ".js", ".css")):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for n, line in enumerate(f, 1):
                    if pat.search(line):
                        found.append("%s:%d  %s" % (os.path.relpath(path, root), n, " ".join(line.split())[:110]))
    return found


def safe_out(out):
    out = os.path.realpath(out)
    home = os.path.realpath(os.path.expanduser("~"))
    if out in (os.path.sep, home) or PRIMER == out or PRIMER.startswith(out + os.sep) or out.startswith(PRIMER + os.sep):
        raise SystemExit("Refusing to use %s as the preview folder." % out)
    if os.path.exists(out):
        if not os.path.isdir(out):
            raise SystemExit("%s exists and is not a folder." % out)
        if os.listdir(out) and not os.path.exists(os.path.join(out, MARKER)):
            raise SystemExit("%s is not empty and is not an earlier preview. Remove it yourself or choose another folder." % out)
    return out


def main():
    if len(sys.argv) != 2 or sys.argv[1].startswith("-"):
        print(__doc__)
        return 2
    out = safe_out(sys.argv[1])
    code = build.main(["--check"])
    if code != 0:
        print("\nThe build is stale or has problems. Run python3 primer/lib/build.py and try again.")
        return code
    if os.path.exists(out):
        shutil.rmtree(out)
    shutil.copytree(PRIMER, out, ignore=ignore)
    with open(os.path.join(out, MARKER), "w") as f:
        f.write("Created by primer/lib/preview.py. It is safe to delete and recreate this folder.\n")
    absolute = absolute_refs(out)
    pages = 0
    for dirpath, _, files in os.walk(out):
        for name in files:
            if name.endswith(".html"):
                path = os.path.join(dirpath, name)
                if inject(path, os.path.relpath(path, out)):
                    pages += 1
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(out) for f in fs)
    print("\npreview written to %s: %d pages marked noindex with a canonical link and the pill, %.1f MB" % (out, pages, size / 1e6))
    if absolute:
        print("references to /primer/ or https://mailab.in/primer found in the pages (check that none is a file the page loads):")
        for a in absolute[:20]:
            print("  " + a)
    else:
        print("no absolute /primer/ references: the copy works from a folder of any name")
    return 0


if __name__ == "__main__":
    sys.exit(main())
