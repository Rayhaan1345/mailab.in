#!/usr/bin/env python3
"""Compare the built pages of two copies of the primer as normalised streams of tags, attributes and text.

    python3 primer/lib/compare_pages.py OLD_PRIMER NEW_PRIMER [--changes CHANGES.md] [--show]

OLD_PRIMER and NEW_PRIMER are two primer/ folders (for example a copy from before a step and the folder after it).
Every index.html is parsed with html.parser and turned into a stream: comments and the doctype are ignored,
attributes are sorted, ?v=hash stamps are removed from href and src, and white space in text is collapsed.
Pages that are byte for byte equal, pages whose streams are equal, and pages that differ are listed.

--show     print the differing words of each changed page, with a little context.
--changes  read a CHANGES file (### slug, then "- was: `...`" and "now: `...`" lines in backticks) and apply the
           entries to the old stream of each page. The result must equal the new stream. Then every difference
           belongs to an entry, and every entry belongs to a difference. Exit 1 if not.
Standard library only.
"""
import difflib
import glob
import html
import os
import re
import sys
from html.parser import HTMLParser

sys.dont_write_bytecode = True  # keep __pycache__ out of the repo


class Norm(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.atoms = []
        self.buf = []
    def flush(self):
        if self.buf:
            t = re.sub(r'\s+', ' ', ''.join(self.buf))
            self.atoms.extend(re.findall(r' |[^ ]+', t))
            self.buf = []
    def handle_starttag(self, tag, attrs):
        self.flush()
        self.atoms.append(self.fmt(tag, attrs, False))
    def handle_startendtag(self, tag, attrs):
        self.flush()
        self.atoms.append(self.fmt(tag, attrs, True))
    def fmt(self, tag, attrs, selfclose):
        out = []
        for k, v in sorted(attrs, key=lambda a: a[0]):
            if v is None:
                out.append(k)
                continue
            if k in ('href', 'src'):
                v = re.sub(r'\?v=[0-9a-f]{8}(?=$|#|&)', '', v)
            v = re.sub(r'\s+', ' ', v).strip() if k == 'class' else v
            out.append('%s="%s"' % (k, v.replace('"', '&quot;')))
        return '<%s%s%s>' % (tag, (' ' + ' '.join(out)) if out else '', '/' if selfclose else '')
    def handle_endtag(self, tag):
        self.flush()
        self.atoms.append('</%s>' % tag)
    def handle_data(self, data):
        self.buf.append(data)
    def handle_comment(self, data):
        pass
    def handle_decl(self, decl):
        pass
    def handle_pi(self, data):
        pass
    def unknown_decl(self, data):
        pass


def atoms(text):
    p = Norm()
    p.feed(text)
    p.close()
    p.flush()
    return p.atoms


def canon(text):
    return ''.join(atoms(text)).strip()


def pages(root):
    out = {}
    top = os.path.join(root, 'index.html')
    if os.path.isfile(top):
        out['index'] = top
    for f in sorted(glob.glob(os.path.join(root, '*', 'index.html'))):
        out[os.path.basename(os.path.dirname(f))] = f
    return out


def hunks(a, b, ctx=0):
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != 'equal':
            out.append((i1, i2, j1, j2))
    return out


def parse_changes(path):
    entries = {}
    cur = None
    was = None
    for line in open(path, encoding='utf-8'):
        line = line.rstrip('\n')
        m = re.match(r'^###\s+(\S+)', line)
        if m:
            cur = m.group(1)
            entries.setdefault(cur, [])
            continue
        m = re.match(r'^\s*-\s*was:\s*`(.*)`\s*$', line)
        if m and cur:
            was = m.group(1)
            continue
        m = re.match(r'^\s*now:\s*`(.*)`\s*$', line)
        if m and cur and was is not None:
            entries[cur].append((was, m.group(1)))
            was = None
    return entries


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    base, new = args[0], args[1]
    changes = None
    if '--changes' in sys.argv:
        changes = parse_changes(sys.argv[sys.argv.index('--changes') + 1])
        args = [a for a in args if a != sys.argv[sys.argv.index('--changes') + 1]]
    show = '--show' in sys.argv
    B, N = pages(base), pages(new)
    problems = 0
    same_bytes = same_struct = 0
    changed = []
    for slug in sorted(set(B) | set(N)):
        if slug not in B or slug not in N:
            print('MISSING page', slug, 'in', 'new' if slug not in N else 'base')
            problems += 1
            continue
        tb, tn = open(B[slug], encoding='utf-8').read(), open(N[slug], encoding='utf-8').read()
        if tb == tn:
            same_bytes += 1
            if changes and changes.get(slug):
                print('ENTRY WITHOUT CHANGE: %s has entries but the page is identical' % slug)
                problems += 1
            continue
        ab, an = atoms(tb), atoms(tn)
        if ab == an:
            same_struct += 1
            if changes and changes.get(slug):
                print('ENTRY WITHOUT CHANGE: %s has entries but the streams are equal' % slug)
                problems += 1
            continue
        changed.append(slug)
        if show:
            print('=' * 8, slug)
            for i1, i2, j1, j2 in hunks(ab, an):
                lo = max(0, i1 - 8)
                print(' ...', ''.join(ab[lo:i1]))
                print(' WAS:', ''.join(ab[i1:i2]))
                print(' NOW:', ''.join(an[j1:j2]))
                print(' ...', ''.join(ab[i2:i2 + 6]))
        if changes is not None:
            S = ''.join(ab)
            ents = changes.get(slug)
            if not ents:
                print('NO ENTRY for changed page', slug)
                problems += 1
                continue
            for was, now in ents:
                wc, nc = canon(was), canon(now)
                n = S.count(wc)
                if n != 1:
                    print('ENTRY DOES NOT MATCH ONCE in %s (%d times): %s' % (slug, n, was[:100]))
                    problems += 1
                    continue
                S = S.replace(wc, nc)
            if S != ''.join(an).strip() and S.strip() != ''.join(an).strip():
                print('ENTRIES DO NOT EXPLAIN ALL DIFFERENCES in', slug)
                sm = difflib.SequenceMatcher(None, re.findall(r' |[^ ]+', S), ''.join(an) and re.findall(r' |[^ ]+', ''.join(an)), autojunk=False)
                k = 0
                for tag, i1, i2, j1, j2 in sm.get_opcodes():
                    if tag != 'equal' and k < 4:
                        k += 1
                        print('   left over WAS:', ''.join(re.findall(r' |[^ ]+', S)[i1:i2])[:300])
                        print('   left over NOW:', ''.join(re.findall(r' |[^ ]+', ''.join(an))[j1:j2])[:300])
                problems += 1
    if changes is not None:
        extra = [s for s in changes if changes[s] and s not in changed]
        for s in extra:
            if s not in B:
                print('ENTRY for unknown page', s)
                problems += 1
    print('pages: %d identical bytes, %d equal streams, %d changed (%s)' % (same_bytes, same_struct, len(changed), ' '.join(changed)))
    print('PROBLEMS: %d' % problems)
    return 1 if problems else 0

if __name__ == '__main__':
    sys.exit(main())
