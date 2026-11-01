"""Offline step for D7: next-token scores of GPT-2 small (Radford and others 2019, MIT licence) for four contexts, to show how a fluent token can be false.
For each context: the 100 highest scores (minus the largest score), a histogram of the other 50,157 scores (bins of 0.5, with the mean and variance of the scores in each bin), the true token if there is one,
and a class for each shown token: true, answer-like (the kind of token that would fill the gap, but not the true one) or other.
Run with: venv-d/bin/python -I make_halluc.py   (writes halluc.json)"""
import json, os, re, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
STATES = {" Maharashtra", " Gujarat", " Mad", " Karn", " Tamil", " Uttar", " Raj", " J", " Kerala", " Andhra", " West", " Bih", " Punjab", " Orissa", " Odisha", " Telangana", " Assam", " Goa", " Har", " Him", " Mah", " Guj", " Delhi", " Kar", " Utt", " Chhat", " Jhar", " Ut"}
CONTEXTS = [
 {"key": "known", "text": "Mumbai is the capital of the Indian state of", "truth": " Maharashtra", "kind": "A fact that appears in many texts",
  "like": lambda t: t in STATES},
 {"key": "rare", "text": "The Ganges shark was recorded at Sassoon Dock in Mumbai in the year", "truth": " 2016", "kind": "A true fact that few texts state",
  "like": lambda t: bool(re.fullmatch(r" (1[5-9]\d\d|20\d\d|1[5-9]|20)", t))},
 {"key": "number", "text": "The Juhu ledger shark grows to a maximum length of", "truth": None, "kind": "A species name made up for this page",
  "like": lambda t: bool(re.fullmatch(r" \d+(\.\d+)?", t))},
 {"key": "name", "text": "The Juhu ledger shark, Chiloscyllium juhuense, was first described by", "truth": None, "kind": "A species name made up for this page",
  "like": lambda t: bool(re.fullmatch(r" [A-Z][a-z]+", t)) and t not in {" The", " A", " An", " British", " In", " This", " It", " Chinese", " French", " German", " Japanese", " Italian", " American", " Australian", " Indian", " Dutch", " Swedish", " Danish", " English", " Russian", " Spanish", " Scottish", " Norwegian"}},
]
out = {"about": "GPT-2 small next-token scores for four contexts, written by make_halluc.py.", "model": "GPT-2 small (124,439,808 weights, 50,257 tokens)", "contexts": []}
for c in CONTEXTS:
    ids = tok(c["text"], return_tensors="pt")["input_ids"]
    with torch.no_grad(): lg = model(ids).logits[0, -1].double()
    p = torch.softmax(lg, -1); z = (lg - lg.max())
    order = torch.argsort(lg, descending=True)
    top = order[:100]
    tail = z[order[100:]].numpy()
    hist, edges = np.histogram(tail, bins=np.arange(math.floor(tail.min() * 2) / 2, 0.5 + 0.0, 0.5)) if False else np.histogram(tail, bins=np.arange(math.floor(tail.min() * 2) / 2, math.ceil(tail.max() * 2) / 2 + 0.5, 0.5))
    idx = np.clip(np.digitize(tail, edges) - 1, 0, len(hist) - 1)
    tail_m = [round(float(tail[idx == b].mean()), 3) if (idx == b).any() else 0.0 for b in range(len(hist))]
    tail_v = [round(float(tail[idx == b].var()), 4) if (idx == b).any() else 0.0 for b in range(len(hist))]
    toks = []
    for i in top.tolist():
        t = tok.decode(i)
        cls = "true" if c["truth"] and t == c["truth"] else ("like" if c["like"](t) else "other")
        toks.append([t, round(float(z[i]), 3), cls])
    H = float(-(p * torch.log(p + 1e-30)).sum())
    entry = {"key": c["key"], "text": c["text"], "kind": c["kind"], "truth": c["truth"], "tokens": toks,
             "tail": {"start": float(edges[0]), "width": 0.5, "counts": hist.astype(int).tolist(), "m": tail_m, "v": tail_v},
             "entropy": round(H, 3), "effective": round(math.exp(H), 1)}
    if c["truth"]:
        tid = tok.encode(c["truth"])[0]
        entry.update(p_true=round(float(p[tid]), 4), rank_true=int((p > p[tid]).sum()) + 1, z_true=round(float(z[tid]), 3))
    # mass check
    pm = {"true": 0.0, "like": 0.0}
    for t, zz, cls in toks:
        if cls in pm: pm[cls] += math.exp(zz)
    Z = float(torch.exp(z).sum())
    entry["mass"] = {k: round(v / Z, 4) for k, v in pm.items()}
    entry["top100_mass"] = round(float(torch.exp(z[top]).sum() / Z), 4)
    out["contexts"].append(entry)
    print(c["key"], "H %.2f eff %.0f" % (H, math.exp(H)), entry.get("p_true"), entry.get("rank_true"), entry["mass"], "top100", entry["top100_mass"], [(t, round(math.exp(zz) / Z, 3), cl) for t, zz, cl in toks[:6]])
json.dump(out, open(os.path.join(HERE, "halluc.json"), "w"), separators=(",", ":"))
