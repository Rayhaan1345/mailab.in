"""Offline step for D5: the same sentence read by an encoder-only model (BERT base, uncased; Devlin et al. 2019; Apache 2.0 licence) and a
decoder-only model (GPT-2 small). For each word, BERT sees the whole sentence with that word replaced by [MASK] and guesses the word;
GPT-2 sees only the words before it and guesses the next word. Writes models.json.
Run with: venv-d/bin/python -I make_bert.py"""
import json, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
HERE = os.path.dirname(os.path.abspath(__file__))
SENT = "We photographed the young shark and put it back."
bt = AutoTokenizer.from_pretrained("bert-base-uncased"); bm = AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval()
gt = AutoTokenizer.from_pretrained("gpt2"); gm = AutoModelForCausalLM.from_pretrained("gpt2").eval()
bids = bt(SENT, return_tensors="pt")["input_ids"][0]; btoks = [bt.decode(int(i)) for i in bids]
gids = gt(SENT, return_tensors="pt")["input_ids"][0]; gtoks = [gt.decode(int(i)) for i in gids]
print(btoks); print(gtoks)
words = SENT.replace(".", " .").split()
assert len(btoks) == len(words) + 2 and len(gtoks) == len(words), (len(btoks), len(gtoks), len(words))
out = {"about": "The same sentence read by BERT (sees both sides) and GPT-2 (sees the left side only), by make_bert.py.", "sentence": SENT, "words": words, "bert_model": "bert-base-uncased", "gpt_model": "gpt2", "rows": []}
for i, w in enumerate(words):
    row = {"i": i, "word": w}
    # BERT: mask word i (position i+1 because of [CLS])
    ids = bids.clone(); ids[i + 1] = bt.mask_token_id
    with torch.no_grad(): lg = bm(ids[None]).logits[0, i + 1]
    p = torch.softmax(lg.double(), -1); top = torch.topk(p, 5)
    tid = int(bids[i + 1])
    row["bert"] = {"top": [[bt.decode(int(j)), round(float(q), 4)] for q, j in zip(top.values, top.indices)], "p_true": round(float(p[tid]), 4), "rank_true": int((p > p[tid]).sum()) + 1}
    # GPT-2: predict word i from words 0..i-1
    if i >= 1:
        with torch.no_grad(): lg = gm(gids[None, :i]).logits[0, -1]
        p = torch.softmax(lg.double(), -1); top = torch.topk(p, 5); tid = int(gids[i])
        row["gpt"] = {"top": [[gt.decode(int(j)).strip(), round(float(q), 4)] for q, j in zip(top.values, top.indices)], "p_true": round(float(p[tid]), 4), "rank_true": int((p > p[tid]).sum()) + 1}
    out["rows"].append(row)
    print(i, w, "BERT", row["bert"]["top"][:3], row["bert"]["p_true"], row["bert"]["rank_true"], "| GPT", row.get("gpt", {}).get("top", [])[:3], row.get("gpt", {}).get("p_true"), row.get("gpt", {}).get("rank_true"))
json.dump(out, open(os.path.join(HERE, "models.json"), "w"), ensure_ascii=False, separators=(",", ":"))
