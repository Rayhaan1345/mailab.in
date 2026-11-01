"""Page assembly for the primer: front matter, the template engine and the page renderer.

build.py calls this module. It has no other users, except new_chapter.py. Standard library only.

A page is a folder with a hand-edited source file, page.html, and a generated page, index.html:

    ---
    description: One sentence for search results.
    subtitle: What the reader can do after this chapter.
    libs: plotly-cartesian, katex
    ---
    <div class="tldr">...</div>
    ...the page's own content...

render_page() joins this with lib/template.html (or lib/template-index.html for the front page),
the partials in lib/partials/, the libraries in lib/libs.json and the record of the page in chapters.json.
The result has no version stamps yet: build.py adds them.
"""
import html
import json
import os
import re

LIB = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.path.dirname(LIB)
PARTIALS = os.path.join(LIB, "partials")
TEMPLATES = {"chapter": "template.html", "practice": "template.html", "index": "template-index.html"}

# Keys that a page.html front matter may carry. Everything else is an error, so a typo is caught.
KEYS = {
    "kind": "chapter (default), practice or index",
    "description": "the meta description (required)",
    "subtitle": "the line under the title (required for chapter and practice, and for index)",
    "kicker": "overrides the line above the title; the default is 'Part B &middot; Chapter B1'",
    "topbar_part": "overrides the part name in the top bar",
    "libs": "the libraries to load, comma separated; the names are in lib/libs.json",
    "scripts": "extra scripts of this page, comma separated, loaded after kit.js and before <slug>.js",
    "byline": "index only: the line under the subtitle",
    "og_title": "optional: a social card title",
    "og_description": "optional: a social card description",
    "og_url": "optional: the address of the page for social cards",
    "author": "optional: meta author",
}
REQUIRED = {"chapter": ("description", "subtitle"), "practice": ("description", "subtitle"),
            "index": ("description", "subtitle", "kicker", "byline")}

# Elements that belong to the shell. A page.html that carries one of them would show it twice.
SHELL_MARKUP = re.compile(
    r"<!doctype|<html\b|<head\b|<body\b|<main\b|</main>|<footer\b|<header\b[^>]*\bhero\b|"
    r"""<nav\b[^>]*\b(?:topbar|chap-nav)\b|<script\b[^>]*\bsrc\s*=|<link\b[^>]*\bstylesheet""", re.I)


class PageError(Exception):
    pass


# ---------- front matter ----------

def split_front_matter(text, where="page.html"):
    """Return (meta dict, content text). Front matter is 'key: value' lines between two '---' lines."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != "---":
        raise PageError("%s: the file must start with a line of three dashes (front matter)" % where)
    meta, end = {}, None
    for i in range(1, len(lines)):
        line = lines[i]
        if line.strip() == "---":
            end = i
            break
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if ":" not in line:
            raise PageError("%s: front matter line %d is not 'key: value': %r" % (where, i + 1, line))
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key not in KEYS:
            raise PageError("%s: unknown front matter key %r (known: %s)" % (where, key, ", ".join(KEYS)))
        if key in meta:
            raise PageError("%s: front matter key %r appears twice" % (where, key))
        meta[key] = value
    if end is None:
        raise PageError("%s: the front matter has no closing line of three dashes" % where)
    content = "\n".join(lines[end + 1:]).strip("\n")
    return meta, content


def join_front_matter(meta, content):
    """The inverse of split_front_matter, with the keys in a fixed order."""
    out = ["---"]
    for key in KEYS:
        if key in meta:
            out.append("%s: %s" % (key, meta[key]))
    out.append("---")
    return "\n".join(out) + "\n" + content.strip("\n") + "\n"


# ---------- libraries ----------

def load_libs():
    with open(os.path.join(LIB, "libs.json"), encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def resolve_libs(names, libs, where):
    """Return (css urls, js urls) for the named libraries, in the order of lib/libs.json."""
    for n in names:
        if n not in libs:
            raise PageError("%s: unknown library %r in 'libs' (known: %s)" % (where, n, ", ".join(libs)))
    chosen = [k for k in libs if k in names]
    groups = {}
    for k in chosen:
        g = libs[k].get("group")
        if g:
            if g in groups:
                raise PageError("%s: 'libs' lists both %s and %s; use one %s bundle" % (where, groups[g], k, g))
            groups[g] = k
    css, js = [], []
    for k in chosen:
        v = libs[k]["version"]
        css += [u.replace("{v}", v) for u in libs[k].get("css", [])]
        js += [u.replace("{v}", v) for u in libs[k].get("js", [])]
    return css, js


# ---------- the template engine ----------
# {{name}}                     a value (raw text; derived values are escaped before they get here)
# {{#if name}}..{{else}}..{{/if}}   when the value is not empty
# {{#for name}}..{{.}}..{{/for}}   repeats the text for the items of a list
# {{> name}}                   the partial lib/partials/name.html
# A tag alone on its line takes the whole line with it, so the template stays readable.

TAG = re.compile(r"\{\{\s*([#/>]?)\s*([^}\s]*)\s*([^}]*?)\s*\}\}")


def tokenize(src, name):
    tokens, pos = [], 0
    for m in TAG.finditer(src):
        kind, word, rest = m.group(1), m.group(2), m.group(3)
        start, end = m.start(), m.end()
        if kind == "" and word == "else":
            kind = "else"
        standalone = False
        line_start = src.rfind("\n", 0, start) + 1
        line_end = src.find("\n", end)
        line_end = len(src) if line_end < 0 else line_end + 1
        if not src[line_start:start].strip() and not src[end:line_end].strip():
            standalone = True
        control = kind in ("#", "/", ">", "else")
        if standalone and (control or kind == ""):
            start, end = line_start, line_end
        if start > pos:
            tokens.append(("text", src[pos:start]))
        if kind == "#":
            tokens.append((word, rest))      # 'if' or 'for', with the name
        elif kind == "/":
            tokens.append(("end" + word, ""))
        elif kind == ">":
            tokens.append(("partial", word))
        elif kind == "else":
            tokens.append(("else", ""))
        else:
            tokens.append(("var", word, standalone and kind == ""))
        pos = end
    if pos < len(src):
        tokens.append(("text", src[pos:]))
    return tokens


def parse(tokens, name, i=0, until=()):
    """Build a tree from tokens. Returns (nodes, next index, terminator)."""
    nodes = []
    while i < len(tokens):
        t = tokens[i]
        k = t[0]
        if k == "text":
            nodes.append(("text", t[1]))
        elif k == "var":
            nodes.append(("var", t[1], t[2]))
        elif k == "partial":
            nodes.append(("partial", t[1]))
        elif k == "if":
            body, i, term = parse(tokens, name, i + 1, ("else", "endif"))
            other = []
            if term == "else":
                other, i, term = parse(tokens, name, i + 1, ("endif",))
            if term != "endif":
                raise PageError("%s: {{#if %s}} is not closed" % (name, t[1]))
            nodes.append(("if", t[1], body, other))
        elif k == "for":
            body, i, term = parse(tokens, name, i + 1, ("endfor",))
            if term != "endfor":
                raise PageError("%s: {{#for %s}} is not closed" % (name, t[1]))
            nodes.append(("for", t[1], body))
        elif k in until:
            return nodes, i, k
        else:
            raise PageError("%s: unexpected {{%s}}" % (name, k))
        i += 1
    return nodes, i, None


def load_template(fname):
    path = os.path.join(LIB, fname) if not os.path.isabs(fname) else fname
    with open(path, encoding="utf-8") as f:
        return f.read(), os.path.relpath(path, PRIMER)


def render_nodes(nodes, ctx, name, item=None):
    out = []
    for n in nodes:
        k = n[0]
        if k == "text":
            out.append(n[1])
        elif k == "var":
            if n[1] == ".":
                value = item
            elif n[1] in ctx:
                value = ctx[n[1]]
            else:
                raise PageError("%s: {{%s}} is not a known value" % (name, n[1]))
            if isinstance(value, (list, tuple)):
                raise PageError("%s: {{%s}} is a list; use {{#for}}" % (name, n[1]))
            value = "" if value is None or value is False else str(value)
            if n[2]:                      # alone on its line: an empty value takes the line with it
                out.append(value + "\n" if value else "")
            else:
                out.append(value)
        elif k == "partial":
            src, pname = load_template(os.path.join(PARTIALS, n[1] + ".html"))
            sub, _, _ = parse(tokenize(src, pname), pname)
            out.append(render_nodes(sub, ctx, pname, item))
        elif k == "if":
            if n[1] not in ctx:
                raise PageError("%s: {{#if %s}} is not a known value" % (name, n[1]))
            out.append(render_nodes(n[2] if ctx[n[1]] else n[3], ctx, name, item))
        elif k == "for":
            if n[1] not in ctx or not isinstance(ctx[n[1]], (list, tuple)):
                raise PageError("%s: {{#for %s}} needs a list" % (name, n[1]))
            for it in ctx[n[1]]:
                out.append(render_nodes(n[2], ctx, name, it))
    return "".join(out)


def render_template(fname, ctx):
    src, name = load_template(fname)
    nodes, _, _ = parse(tokenize(src, name), name)
    return render_nodes(nodes, ctx, name)


# ---------- chapters.json ----------

def load_chapters():
    with open(os.path.join(PRIMER, "chapters.json"), encoding="utf-8") as f:
        data = json.load(f)
    return data, {c["slug"]: c for c in data["chapters"]}


def esc(s):
    return html.escape(s, quote=False)


# ---------- one page ----------

def render_page(page_path, text=None, chapters=None, libs=None):
    """Return the unstamped index.html text for the page.html at page_path."""
    page_dir = os.path.dirname(os.path.abspath(page_path))
    is_root = os.path.abspath(page_dir) == PRIMER
    slug = os.path.basename(page_dir)
    where = os.path.relpath(page_path, PRIMER)
    if text is None:
        with open(page_path, encoding="utf-8") as f:
            text = f.read()
    data, by_slug = chapters if chapters else load_chapters()
    libs = libs or load_libs()
    meta, content = split_front_matter(text, where)

    kind = meta.get("kind", "index" if is_root else "chapter")
    if kind not in TEMPLATES:
        raise PageError("%s: kind must be one of %s, not %r" % (where, ", ".join(TEMPLATES), kind))
    if is_root != (kind == "index"):
        raise PageError("%s: kind: index belongs to primer/page.html only" % where)
    for key in REQUIRED[kind]:
        if not meta.get(key):
            raise PageError("%s: front matter needs '%s'" % (where, key))
    bad = SHELL_MARKUP.search(content)
    if bad:
        raise PageError("%s: the content carries shell markup (%s). The template supplies it; remove it from page.html" % (where, bad.group(0).strip()))

    ctx = {k: "" for k in KEYS}
    ctx.update(meta)
    names = [x.strip() for x in meta.get("libs", "").split(",") if x.strip()]
    ctx["lib_css"], ctx["lib_js"] = resolve_libs(names, libs, where)
    ctx["scripts"] = [x.strip() for x in meta.get("scripts", "").split(",") if x.strip()]
    ctx["content"] = content
    ctx["is_index"] = kind == "index"
    ctx["is_practice"] = kind == "practice"
    ctx["assets"] = "assets/" if is_root else "../assets/"
    if is_root:
        ctx["title"] = esc(data["title"])
        ctx["slug"] = ""
        ctx["own_css"] = ctx["own_js"] = False
        ctx["ready"] = sum(1 for c in data["chapters"] if c.get("live"))
        ctx["total"] = len(data["chapters"])
        ctx.update({"id": "", "num": "", "part": "", "part_name": ""})
    else:
        if slug not in by_slug:
            raise PageError("%s: %s is not in chapters.json. Add the chapter there first" % (where, slug))
        c = by_slug[slug]
        for field in ("id", "num", "title", "part"):
            if not c.get(field):
                raise PageError("%s: chapters.json has no %r for %s. Fill it in" % (where, field, slug))
        if c["part"] not in data["parts"]:
            raise PageError("%s: part %r of %s is not in chapters.json 'parts'" % (where, c["part"], slug))
        ctx["slug"] = slug
        ctx["id"] = c["id"]
        ctx["num"] = esc(c["num"])
        ctx["part"] = esc(c["part"])
        ctx["part_name"] = esc(data["parts"][c["part"]])
        ctx["title"] = esc(c["title"])
        ctx["own_css"] = os.path.isfile(os.path.join(page_dir, slug + ".css"))
        ctx["own_js"] = os.path.isfile(os.path.join(page_dir, slug + ".js"))
    return render_template(TEMPLATES[kind], ctx)
