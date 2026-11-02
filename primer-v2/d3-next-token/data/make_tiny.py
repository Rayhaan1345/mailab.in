"""Offline step for D3: a tiny word model for the generation-loop figure. It writes tiny.json.
It is built by counting word triples in 500 short MADE-UP reports generated from sentence templates in the style of the invented
SharkCam messages (seed 4127). It is a counting model with a context of two words, not a neural network.
Run with: python3 -I make_tiny.py"""
import json, os, random
from collections import defaultdict, Counter
HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(4127)
def pick(options):
    items, w = zip(*options); return rng.choices(items, weights=w)[0]
def sp(): return pick([("spadenose", 5), ("milk", 3), ("bamboo", 2)])
def place(): return pick([("sassoon dock", 4), ("versova", 2), ("juhu", 2), ("uran", 2)])
def num(): return pick([("ten", 2), ("twelve", 2), ("fifteen", 4), ("twenty", 2)])
def cm(): return pick([(str(x), 1) for x in [25, 30, 40, 44, 50, 60, 80]])
def S1(): return f"{num()} {sp()} sharks came in at {place()} this morning ."
def S2(): return f"the biggest was about {cm()} cm and the smallest about {cm()} cm ."
def S3(): return f"prices were {pick([('low', 3), ('high', 1)])} because the auction started {pick([('late', 3), ('early', 1)])} ."
def S4(): return f"we found a {pick([('young', 4), ('small', 2), ('big', 1)])} {sp()} shark in a rock pool at {place()} at low tide ."
def S5(): return f"length {cm()} cm , dark bands across its back ."
def S6(): return f"we {pick([('photographed', 3), ('measured', 2), ('weighed', 1)])} it and {pick([('put it back', 4), ('let it go', 3)])} ."
def S7(): return f"our gillnet caught one hammerhead near {place()} , about {cm()} cm long ."
def S8(): return f"it looked young , so we {pick([('let it go', 4), ('put it back in the water', 2)])} ."
def S9(): return f"the water at {place()} was {pick([('twenty eight', 4), ('twenty nine', 2), ('twenty seven', 2)])} degrees at {pick([('six', 3), ('ten', 2), ('four', 1)])} metres ."
def S10(): return f"we could see about {pick([('four', 3), ('five', 2), ('three', 1)])} metres ahead ."
def S11(): return f"{pick([('two', 3), ('three', 1)])} milk sharks swam past the rocks ."
report_kinds = [
  ([S1, S2, S3], 4), ([S4, S5, S6], 4), ([S7, S8], 3), ([S9, S10, S11], 3), ([S1, S3], 1), ([S4, S6], 1),
]
docs = []
for _ in range(500):
    seq = pick(report_kinds)
    docs.append(" ".join(f() for f in seq).split())
tri = defaultdict(Counter)
for d in docs:
    toks = ["<s>", "<s>"] + d + ["</s>"]
    for a, b, c in zip(toks, toks[1:], toks[2:]): tri[a + "|" + b][c] += 1
vocab = sorted({w for d in docs for w in d})
out = {"about": "Counts of word triples in 500 MADE-UP reports built from templates (seed 4127), by make_tiny.py. Not field data.",
       "docs": len(docs), "words": sum(len(d) for d in docs), "vocab": len(vocab),
       "contexts": {k: sorted(v.items(), key=lambda kv: (-kv[1], kv[0])) for k, v in sorted(tri.items())},
       "examples": [" ".join(d) for d in docs[:3]]}
json.dump(out, open(os.path.join(HERE, "tiny.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("contexts", len(tri), "vocab", len(vocab), "words", out["words"], "bytes", os.path.getsize(os.path.join(HERE, "tiny.json")))
print(out["examples"])
print(out["contexts"]["<s>|<s>"], out["contexts"]["at|sassoon"], out["contexts"]["in|at"])
