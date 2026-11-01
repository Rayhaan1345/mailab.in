#!/usr/bin/env python3
"""Start a new chapter from the starter in lib/starter/.

    python3 primer/lib/new_chapter.py b10-something

Creates primer/<slug>/ with page.html (front matter and a small working example), <slug>.js and <slug>.css.
It does not edit chapters.json. It prints the entry to add there. After that, run python3 primer/lib/build.py
to build the page, then open primer/<slug>/ in a browser. Standard library only.
"""
import json
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.path.dirname(HERE)
STARTER = os.path.join(HERE, "starter")


def main(argv):
    if len(argv) != 2 or argv[1].startswith("-"):
        print(__doc__)
        return 2
    slug = argv[1].strip("/")
    if not re.fullmatch(r"[a-z][a-z0-9]*-[a-z0-9]+(?:-[a-z0-9]+)*", slug):
        print("The slug must look like b10-something: the chapter id, a dash, then lowercase words joined by dashes.")
        return 2
    dest = os.path.join(PRIMER, slug)
    if os.path.exists(dest):
        print("primer/%s already exists. Choose another slug or remove it first." % slug)
        return 1
    os.makedirs(dest)
    for src, name in (("page.html", "page.html"), ("starter.js", slug + ".js"), ("starter.css", slug + ".css")):
        shutil.copyfile(os.path.join(STARTER, src), os.path.join(dest, name))
    cid = slug.split("-")[0]
    with open(os.path.join(PRIMER, "chapters.json"), encoding="utf-8") as f:
        data = json.load(f)
    known = any(c["slug"] == slug for c in data["chapters"])
    print("Created primer/%s/ with page.html, %s.js and %s.css." % (slug, slug, slug))
    print()
    if known:
        print("chapters.json already lists %s." % slug)
    else:
        part = cid[0].upper() if cid[0].upper() in data["parts"] else ""
        entry = {"id": cid, "num": "", "slug": slug, "title": "", "blurb": "", "part": part, "live": False}
        print("Next: add this to the 'chapters' list in primer/chapters.json, in reading order, and fill in the blanks")
        print("(num is the number shown to readers, for example B10; title and blurb are plain text):")
        print()
        print("  " + json.dumps(entry, indent=1).replace("\n", "\n  "))
        print()
    print("Then: edit page.html (description, subtitle, libs and the content), edit %s.js, and run" % slug)
    print("  python3 primer/lib/build.py")
    print("The front page shows the chapter as coming soon until its 'live' flag in chapters.json is true.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
