"""Computes the data behind the practice set pd-practice (Part D, language) and prints key values. With the argument --write it also writes figdata.json next to this file.
The small calculations of the questions themselves (merges, cosines, softmax, attention, parameter counts, rotations, cache sizes) are done in questions.py and checked again,
with separate code (PyTorch layers and autograd, a second BPE trainer, brute force counting), in verify.py. This file reads the data files of chapters D1 to D7 (read only) and prepares the charts and tables:
  Q3   cost ratio of the twelve Hindi, Marathi and Hinglish messages under GPT-2 and Qwen2.5       (d1-tokenisation/data/tokens.json)
  Q4   token counts of the Marathi report m12 and its English meaning under GPT-2                  (d1-tokenisation/data/tokens.json)
  Q6   the GPT-2 pieces of a 20-token note about two tags                                          (GPT-2 tokenizer from the transformers library, cached on disk)
  Q7   accuracy of the analogy test by relation                                                    (d2-embeddings/data/words.json)
  Q8   the four highest GPT-2 scores for "Mumbai is the capital of the Indian state of"            (d3-next-token/data/contexts.json)
  Q11  perplexity per token and bits per character of the 16 messages                              (d3-next-token/data/loss.json)
  Q14  loss curves of the two tiny models with and without the causal mask                         (d4-self-attention/data/leak.json)
  Q20  GPT-2 small loss against window size, with and without the first token                      (d6-long-context/data/gpt2win.json)
  Q21  per-token loss of GPT-2 small on a question and a reply                                     (d7-assistants/data/stages.json)
  Q23  next-token probabilities of GPT-2 small for six labelled examples, true and flipped labels  (d7-assistants/data/icl.json)
  Q24  the ten highest probabilities for the year of the Ganges shark record                       (d7-assistants/data/halluc.json)
Needs numpy and the transformers library (the scratchpad venv-d Python):  python -I figdata.py [--write]
The environment variable PRIMER can point to the primer folder; the default is the repository of the site."""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.environ.get("PRIMER") or (os.path.abspath(os.path.join(HERE, "..", "..")) if os.path.exists(os.path.join(HERE, "..", "..", "data", "sharks", "catch.csv")) else
          "/private/tmp/claude-502/-Users-shadabkhan-Documents-rayhaan-application/9307b589-bce9-48cb-b0f8-a7d270ba25b8/scratchpad/work/mailab.in/primer")
J = lambda p: json.load(open(os.path.join(PRIMER, p), encoding="utf-8"))
D = {}

# ---------------- Q3 and Q4: cost ratios (chapter D1) ----------------
tk = J("d1-tokenisation/data/tokens.json")
rows = []
for m in tk["messages"]:
    if m["language"] == "English": continue
    g, q = len(m["tok"]["gpt2"]["text"]), len(m["tok"]["qwen"]["text"])
    gg, qg = m["tok"]["gpt2"]["gloss_n"], m["tok"]["qwen"]["gloss_n"]
    rows.append({"id": m["id"], "lang": m["language"], "g": g, "gg": gg, "q": q, "qg": qg, "rg": g / gg, "rq": q / qg})
D["q3"] = rows
m12 = [m for m in tk["messages"] if m["id"] == "m12"][0]
D["q4"] = {"id": "m12", "n": len(m12["tok"]["gpt2"]["text"]), "n_en": m12["tok"]["gpt2"]["gloss_n"]}

# ---------------- Q6: the pieces of a note, from the GPT-2 tokenizer ----------------
NOTE = "Tag 4127 was found at Juhu. Tag 3305 was caught at Sassoon Dock."
try:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ids = tok.encode(NOTE)
    pieces = [tok.decode(i) for i in ids]
except Exception as e:                       # fall back to the stored result when the library is missing
    old = json.load(open(os.path.join(HERE, "figdata.json"))) if os.path.exists(os.path.join(HERE, "figdata.json")) else None
    if not old: raise
    pieces = old["q6"]["pieces"]; ids = old["q6"]["ids"]
D["q6"] = {"text": NOTE, "pieces": pieces, "ids": ids}

# ---------------- Q7: the analogy test by relation (chapter D2) ----------------
at = J("d2-embeddings/data/words.json")["analogy_test"]
S = {s[0]: (s[1], s[2]) for s in at["sections"]}
pick = [("gram6-nationality-adjective", "Nationality adjective"), ("family", "Family"), ("capital-world", "Capital of a country in the world"), ("gram8-plural", "Plural noun"),
        ("gram7-past-tense", "Past tense"), ("city-in-state", "City in a US state"), ("currency", "Currency"), ("gram2-opposite", "Opposite")]
D["q7"] = {"rows": [{"name": nm, "acc": S[k][0], "n": S[k][1], "key": k} for k, nm in pick], "overall": at["overall"], "total": S["Total accuracy"][1]}

# ---------------- Q8: GPT-2 scores for the Maharashtra context (chapter D3) ----------------
ctx = [c for c in J("d3-next-token/data/contexts.json")["contexts"] if c["key"] == "capital"][0]
D["q8"] = {"text": ctx["text"], "top": ctx["top"][:4]}

# ---------------- Q11: perplexity and bits per character of the 16 messages (chapter D3) ----------------
ms = []
for m in J("d3-next-token/data/loss.json")["messages"]:
    L = [t[1] for t in m["tokens"]]; n = len(L); tot = sum(L)
    ms.append({"id": m["id"], "lang": m["language"], "n": n, "chars": m["chars"], "total": tot, "ppl": math.exp(tot / n), "bpc": tot / math.log(2) / m["chars"]})
D["q11"] = ms

# ---------------- Q14: loss curves with and without the mask (chapter D4) ----------------
lk = J("d4-self-attention/data/leak.json")
D["q14"] = {"mask": lk["mask"], "nomask": lk["nomask"], "uniform": lk["uniform"], "vocab": lk["vocab"]}

# ---------------- Q20: window experiment on GPT-2 small (chapter D6) ----------------
gw = J("d6-long-context/data/gpt2win.json")
D["q20"] = {"windows": gw["windows"], "window": gw["loss_window"], "first": gw["loss_window_first"], "full": gw["loss_full"]}

# ---------------- Q21: loss of GPT-2 small on a question and a reply (chapter D7) ----------------
st = J("d7-assistants/data/stages.json")["loss"]
D["q21"] = {"tokens": [[t["t"], t["loss"], t["part"]] for t in st["tokens"]], "mean_reply": st["mean_reply"], "mean_prompt": st["mean_prompt"]}

# ---------------- Q23: six labelled examples, true and flipped (chapter D7) ----------------
ic = J("d7-assistants/data/icl.json")["rule"]
def pc(e, lab): return (e[0] if lab == "yes" else e[1]) / (e[0] + e[1])
D["q23"] = {"labels": [q[1] for q in ic["queries"]],
            "mass0": [e[0] + e[1] for e in ic["results"]["true"]["0"]], "mass6": [e[0] + e[1] for e in ic["results"]["true"]["63"]],
            "p_true": [pc(e, q[1]) for e, q in zip(ic["results"]["true"]["63"], ic["queries"])], "p_flip": [pc(e, q[1]) for e, q in zip(ic["results"]["flip"]["63"], ic["queries"])]}

# ---------------- Q24: the year of the Ganges shark record (chapter D7) ----------------
hl = J("d7-assistants/data/halluc.json")["contexts"][1]
assert hl["key"] == "rare"
Z = sum(math.exp(t[1]) for t in hl["tokens"]) + sum(c * math.exp(m) * (1 + v / 2) for c, m, v in zip(hl["tail"]["counts"], hl["tail"]["m"], hl["tail"]["v"]))
D["q24"] = {"text": hl["text"], "truth": hl["truth"], "rank_true": hl["rank_true"], "p_true": hl["p_true"], "Z": Z,
            "top": [[t[0].strip(), math.exp(t[1]) / Z, t[2]] for t in hl["tokens"][:10]], "effective": hl["effective"]}

if "--write" in sys.argv:
    json.dump(D, open(os.path.join(HERE, "figdata.json"), "w"), separators=(",", ":"))
    print("written figdata.json")
print("Q3 cost ratios GPT-2:", [round(r["rg"], 2) for r in D["q3"]], "Qwen2.5:", [round(r["rq"], 2) for r in D["q3"]])
print("Q4 m12 tokens", D["q4"])
print("Q6 pieces", len(D["q6"]["pieces"]), D["q6"]["pieces"])
print("Q7 rows", [(r["name"], round(r["acc"], 3), r["n"], round(r["acc"] * r["n"])) for r in D["q7"]["rows"]])
print("Q8 top four", D["q8"]["top"])
print("Q11 perplexity and bits per character", [(m["id"], round(m["ppl"], 1), round(m["bpc"], 2)) for m in D["q11"]])
print("Q14 final losses (mask: training, held-out; no mask: training, held-out)", D["q14"]["mask"][-1], D["q14"]["nomask"][-1])
print("Q20 windows", D["q20"]["windows"], "without", D["q20"]["window"], "with first", D["q20"]["first"], "full", D["q20"]["full"])
print("Q23 mass on yes or no with no example and six", [round(x, 3) for x in D["q23"]["mass0"]], [round(x, 3) for x in D["q23"]["mass6"]], "P(correct) true", [round(x, 3) for x in D["q23"]["p_true"]], "flipped", [round(x, 3) for x in D["q23"]["p_flip"]])
print("Q24 top ten", [(t[0], round(100 * t[1], 1), t[2]) for t in D["q24"]["top"]], "Z", round(Z, 3), "ten bars add up to", round(sum(t[1] for t in D["q24"]["top"]), 3))
