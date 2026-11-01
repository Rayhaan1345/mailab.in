"""Offline step for D4: attention weights of GPT-2 small (Radford et al. 2019; MIT licence; 12 layers, 12 heads per layer, key size 64)
for three short sentences. Writes attention.json (four featured heads on three sentences, with the weights under the causal mask, the
weights if the mask were removed, and the scaled scores) and allheads.json (all 144 heads on sentence 1, plus three averages per head).
Run with: venv-d/bin/python -I make_attention.py
The scores are q.k / sqrt(64) computed from the layer's own input, so 'weights if the mask were removed' means: the same scores, softmax over
all positions. The model itself was trained and run with the mask, so this is not what GPT-2 would compute if it were run without one."""
import json, os, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
SENT = [
 ("s1", "Tag 4127 is a young bamboo shark. We put it back and wrote down Tag 4127."),
 ("s2", "The fisherman caught a hammerhead and he let it go because it was young."),
 ("s3", "Fifteen spadenose sharks came in at Sassoon Dock this morning."),
]
EXTRA = ["Fifteen spadenose sharks came in at Sassoon Dock this morning. The biggest was about 50 cm and the smallest about 25 cm. Prices were low because the auction started late.",
 "Found a young grey bamboo shark in a rock pool at Juhu at low tide. Length 44 cm, dark bands across its back. We photographed it and put it back. Tag 4127.",
 "Our gillnet caught one hammerhead near Versova, about 80 cm long. It looked young, so we let it go.",
 "Water at Uran was 28 degrees at 6 metres. We could see about 4 metres ahead. Two milk sharks swam past the rocks."]
FEATURED = [
 (4, 11, "previous token", "Almost all of its weight goes to the token just before."),
 (0, 1, "itself and its twin", "Most weight goes to the token itself; when the same token appeared earlier, that copy gets a share."),
 (5, 5, "after the earlier twin", "When a token appeared before, it looks at the token that came after that earlier copy. Elsewhere it rests on the first token."),
 (1, 10, "spread out", "No sharp peak: the weight is spread over the nearest words, with a little more on the first nouns."),
]

def run(text):
    ids = tok(text, return_tensors="pt")
    with torch.no_grad(): o = model(**ids, output_attentions=True, output_hidden_states=True)
    return ids["input_ids"][0], o

def scores(o, layer, n):
    """scaled scores for all heads of one layer, shape (12, n, n), from the layer's own input"""
    blk = model.transformer.h[layer]
    h = o.hidden_states[layer]
    a = blk.ln_1(h)
    qkv = blk.attn.c_attn(a)[0]  # n, 2304
    q, k, v = qkv.split(768, dim=-1)
    q = q.view(n, 12, 64).transpose(0, 1); k = k.view(n, 12, 64).transpose(0, 1)
    return (q @ k.transpose(1, 2) / math.sqrt(64))

out = {"about": "Attention of GPT-2 small on three short sentences written for this primer, computed by make_attention.py.", "model": "gpt2",
       "heads": [{"id": f"L{l}H{h}", "layer": l, "head": h, "name": nm, "note": nt} for l, h, nm, nt in FEATURED], "sentences": []}

# induction test: 20 random token sequences of length 40, repeated twice. A token in the second copy is scored by the weight it puts on the
# token that FOLLOWED its earlier copy (position i+1 for the token at position 40+i).
rng = np.random.default_rng(0)
ind = np.zeros((12, 12)); T = 40; NREP = 20
for _ in range(NREP):
    seq = rng.integers(1000, 30000, size=T)
    ids = torch.tensor(np.concatenate([seq, seq]))[None]
    with torch.no_grad(): o = model(ids, output_attentions=True)
    A = torch.stack([a[0] for a in o.attentions]).numpy()
    for i in range(T - 1): ind += A[:, :, T + i, i + 1]
ind /= NREP * (T - 1)
for hd in out["heads"]: hd["induction"] = round(float(ind[hd["layer"], hd["head"]]), 2)
print("induction scores of the featured heads:", [(h["id"], h["induction"]) for h in out["heads"]])
for sid, text in SENT:
    ids, o = run(text); n = len(ids)
    rec = {"id": sid, "text": text, "tokens": [tok.decode(int(i)) for i in ids], "heads": {}}
    for l, h, nm, nt in FEATURED:
        S = scores(o, l, n)[h].detach().numpy()
        mask = np.tril(np.ones((n, n), bool))
        Sm = np.where(mask, S, -np.inf)
        W = np.exp(Sm - Sm.max(1, keepdims=True)); W /= W.sum(1, keepdims=True)
        Wn = np.exp(S - S.max(1, keepdims=True)); Wn /= Wn.sum(1, keepdims=True)
        ref = o.attentions[l][0, h].numpy()
        assert np.abs(W - ref).max() < 2e-4, (sid, l, h, np.abs(W - ref).max())
        rec["heads"][f"L{l}H{h}"] = {"w": np.round(W, 3).tolist(), "wn": np.round(Wn, 3).tolist(), "s": np.round(S, 1).tolist()}
    out["sentences"].append(rec)
json.dump(out, open(os.path.join(HERE, "attention.json"), "w"), separators=(",", ":"))
print("attention.json bytes", os.path.getsize(os.path.join(HERE, "attention.json")))

# all 144 heads on sentence 1, as whole percents of the lower triangle, and three averages over eight texts
ids, o = run(SENT[0][1]); n = len(ids)
A = torch.stack([a[0] for a in o.attentions]).numpy()  # 12,12,n,n
tri = [[[ [int(round(100 * A[l, h, i, j])) for j in range(i + 1)] for i in range(n)] for h in range(12)] for l in range(12)]
first = np.zeros((12, 12)); prev = np.zeros((12, 12)); selfw = np.zeros((12, 12)); cnt = 0
for text in [t for _, t in SENT] + EXTRA:
    ids2, o2 = run(text); B = torch.stack([a[0] for a in o2.attentions]).numpy()
    for i in range(2, len(ids2)):
        first += B[:, :, i, 0]; prev += B[:, :, i, i - 1]; selfw += B[:, :, i, i]; cnt += 1
stats = {"first": np.round(first / cnt, 2).tolist(), "prev": np.round(prev / cnt, 2).tolist(), "self": np.round(selfw / cnt, 2).tolist()}
json.dump({"about": "All 144 heads of GPT-2 small on sentence s1 (lower triangle, whole percents) and the mean weight each head puts on the first token, the previous token and the token itself, over 7 texts (tokens from the third on).",
           "tokens": [tok.decode(int(i)) for i in ids], "tri": tri, "stats": stats, "ntexts": 7, "ntokens": cnt},
          open(os.path.join(HERE, "allheads.json"), "w"), separators=(",", ":"))
print("allheads.json bytes", os.path.getsize(os.path.join(HERE, "allheads.json")), "tokens counted", cnt)
fm = np.array(stats["first"]); print("heads with first-token share above 0.8:", int((fm > 0.8).sum()), "of 144; layer 0:", np.round(fm[0], 2))
print("L4H11 prev", stats["prev"][4][11], "L0H1 self", stats["self"][0][1], "L5H5 first", stats["first"][5][5], "L1H10 first/prev/self", stats["first"][1][10], stats["prev"][1][10], stats["self"][1][10])
