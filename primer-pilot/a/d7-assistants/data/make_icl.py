"""Offline step for D7: what does a prompt change in GPT-2 small, with no change to its weights?
Two experiments, each over every subset of a small pool of prompt pieces (exact next-token probabilities from GPT-2 small,
Radford and others 2019, MIT licence):
  facts: five made-up facts, each can be in the prompt or not; one question per fact.
  rule:  six labelled example reports (yes or no: was a shark seen?), each in or out, true labels or flipped labels; six new reports.
Run with: venv-d/bin/python -I make_icl.py   (writes icl.json)"""
import json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
def probs(prompt):
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    with torch.no_grad(): lg = model(ids).logits[0, -1]
    return torch.softmax(lg.double(), -1)
def top(p, k=5): t = torch.topk(p, k); return [[tok.decode(int(i)), round(float(v), 4)] for v, i in zip(t.values, t.indices)]

FACTS = [
 ("The SharkCam robot is called Finwick.", "Q: What is the SharkCam robot called?\nA: It is called", " Finwick"),
 ("The dive buddy of the Juhu team is Farida.", "Q: Who is the dive buddy of the Juhu team?\nA:", " Farida"),
 ("The reef we dive on is called Velkar Ledge.", "Q: What is the reef we dive on called?\nA: It is called", " Velkar"),
 ("Tag 4127 was found at Juhu.", "Q: Where was Tag 4127 found?\nA: It was found at", " Juhu"),
 ("Tag 3305 was landed at Sassoon Dock.", "Q: Where was Tag 3305 landed?\nA: It was landed at", " Sassoon"),
]
facts = {"pieces": [f[0] for f in FACTS], "questions": [f[1] for f in FACTS], "answers": [f[2].strip() for f in FACTS], "results": {}}
for mask in range(2 ** len(FACTS)):
    inc = [FACTS[i][0] for i in range(len(FACTS)) if mask >> i & 1]
    row = []
    for (_, q, a) in FACTS:
        prompt = ("\n".join(inc) + "\n" if inc else "") + q
        p = probs(prompt); tid = tok.encode(a)[0]
        row.append({"top": top(p), "p": float("%.4g" % float(p[tid])), "first": tok.decode(tid), "rank": int((p > p[tid]).sum()) + 1})
    facts["results"][str(mask)] = row

EX = [
 ("A hammerhead swam past the camera at 12 metres.", "yes"), ("Only sand and a few small fish at the reef today.", "no"),
 ("We saw a young bamboo shark in the rock pool.", "yes"), ("The water was too murky to see anything.", "no"),
 ("A milk shark came right up to the lens.", "yes"), ("The battery died after ten minutes.", "no"),
]
QS = [("A big hammerhead turned toward the camera near the reef.", "yes"), ("Nothing but sand and a plastic bag.", "no"),
      ("A spadenose shark hid under a ledge.", "yes"), ("The lens was covered in silt.", "no"),
      ("We spotted a bamboo shark sleeping on the sand.", "yes"), ("It rained all day and we stayed ashore.", "no")]
FMT = "{t} Shark seen:{l}\n"
YES, NO = tok.encode(" yes")[0], tok.encode(" no")[0]
rule = {"format": "text + ' Shark seen:' + ' yes' or ' no'", "examples": [list(e) for e in EX], "queries": [list(q) for q in QS], "results": {"true": {}, "flip": {}}}
for flip in (False, True):
    for mask in range(2 ** len(EX)):
        inc = [EX[i] for i in range(len(EX)) if mask >> i & 1]
        row = []
        for q, ans in QS:
            prompt = "".join(FMT.format(t=t, l=" " + (("no" if l == "yes" else "yes") if flip else l)) for t, l in inc) + FMT.format(t=q, l="").rstrip("\n")
            p = probs(prompt); t1 = top(p, 1)[0]
            row.append([round(float(p[YES]), 4), round(float(p[NO]), 4), t1[0], t1[1]])
        rule["results"]["flip" if flip else "true"][str(mask)] = row
json.dump({"about": "GPT-2 small next-token probabilities for every subset of prompt pieces, written by make_icl.py.", "facts": facts, "rule": rule}, open(os.path.join(HERE, "icl.json"), "w"), separators=(",", ":"))
# a few numbers for the text
f = facts["results"]
print("facts: P(answer) with no fact vs only its own fact vs all five")
for q in range(5):
    print(q, facts["answers"][q], f["0"][q]["p"], f[str(1 << q)][q]["p"], f[str(31)][q]["p"])
print("rule: mass on yes+no, no examples:", sum(a + b for a, b, *_ in rule["results"]["true"]["0"]) / 6, " six examples:", sum(a + b for a, b, *_ in rule["results"]["true"]["63"]) / 6)
def acc(res): return sum(((a > b) == (QS[i][1] == "yes")) for i, (a, b, *_) in enumerate(res)) / 6
print("rule accuracy with six true / six flipped labels:", acc(rule["results"]["true"]["63"]), acc(rule["results"]["flip"]["63"]))
print("mean P(correct label | yes or no) with six true / six flipped:",
      sum((a if QS[i][1] == "yes" else b) / (a + b) for i, (a, b, *_) in enumerate(rule["results"]["true"]["63"])) / 6,
      sum((a if QS[i][1] == "yes" else b) / (a + b) for i, (a, b, *_) in enumerate(rule["results"]["flip"]["63"])) / 6)
