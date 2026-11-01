"""Offline step for D5: numbers from GPT-2 small (Radford et al. 2019; MIT licence; 12 blocks, width 768, 12 heads) for the residual-stream
figure (size of the stream and of each sub-layer's output at the last token, and the 'logit lens' guesses after each block), the learned
position vectors, the key-value cache (measured) and a greedy continuation for the generation figure. Writes gpt2.json.
Run with: venv-d/bin/python -I make_gpt2.py"""
import json, os, math
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
cfg = model.config
def vis(t): return t.replace("\n", "\\n")
CONTEXTS = [
 ("capital", "Mumbai is the capital of the Indian state of", " Maharashtra"),
 ("nets", "The fishermen at Sassoon Dock cast their", " nets"),
 ("put", "Found a young grey bamboo shark in a rock pool at Juhu at low tide. Length 44 cm, dark bands across its back. We photographed it and put it", " back"),
]
blocks = model.transformer.h
store = {}
def hook_for(name):
    def f(mod, inp, out): store[name] = (out[0] if isinstance(out, tuple) else out).detach()
    return f
for l, b in enumerate(blocks):
    b.attn.register_forward_hook(hook_for(f"attn{l}")); b.mlp.register_forward_hook(hook_for(f"mlp{l}"))
W = model.lm_head.weight  # tied to wte
out = {"about": "GPT-2 small numbers for D5, by make_gpt2.py.", "config": {"n_layer": cfg.n_layer, "n_embd": cfg.n_embd, "n_head": cfg.n_head, "n_positions": cfg.n_positions, "vocab": cfg.vocab_size}, "contexts": []}
for key, text, truth in CONTEXTS:
    ids = tok(text, return_tensors="pt")["input_ids"]
    with torch.no_grad(): o = model(ids, output_hidden_states=True)
    hs = o.hidden_states  # 13 entries: input to block 0 ... output of block 11 (after ln_f for the last in HF)
    n = ids.shape[1]; tid = tok.encode(truth)[0]
    rows = []
    # hidden_states[l] is the stream entering block l (l = 0..11); hidden_states[12] is after the final layer norm. Rebuild the raw stream after block 11.
    raw = [hs[l][0, -1] for l in range(12)]
    # stream after block 11 = input of block 11 + attn11 + mlp11
    raw.append(raw[11] + store["attn11"][0, -1] + store["mlp11"][0, -1])
    final = None
    for l in range(13):
        s = raw[l]
        logits = model.transformer.ln_f(s) @ W.T
        p = torch.softmax(logits.double(), -1)
        top = torch.topk(p, 5)
        rows.append({"layer": l, "norm": round(float(s.norm()), 2),
                     "attn": round(float(store[f"attn{l}"][0, -1].norm()), 2) if l < 12 else None,
                     "mlp": round(float(store[f"mlp{l}"][0, -1].norm()), 2) if l < 12 else None,
                     "top": [[vis(tok.decode(int(i))), round(float(pp), 4)] for pp, i in zip(top.values, top.indices)],
                     "p_true": round(float(p[tid]), 4), "rank_true": int((p > p[tid]).sum()) + 1})
    # check against the model's own output
    pf = torch.softmax(o.logits[0, -1].double(), -1)
    assert abs(float(pf[tid]) - rows[12]["p_true"]) < 1e-3, (float(pf[tid]), rows[12]["p_true"])
    out["contexts"].append({"key": key, "text": text, "tokens": n, "true": vis(truth), "layers": rows})
    print(key, "final p_true", rows[12]["p_true"], "layer-by-layer p_true", [r["p_true"] for r in rows])

# learned position vectors: cosine similarity of the first 64 positions, and the first 16 numbers of three positions
P = model.transformer.wpe.weight.detach()
Pn = P[:64] / P[:64].norm(dim=1, keepdim=True); S = (Pn @ Pn.T).numpy()
out["wpe"] = {"n": 64, "cos": np.round(S, 2).tolist(), "norm": [round(float(x), 2) for x in P[:64].norm(dim=1)], "total": int(P.shape[0]), "dim": int(P.shape[1])}

# parameters of one block, counted from the model
b = blocks[0]; cnt = lambda m: sum(p.numel() for p in m.parameters())
out["params"] = {"attn": cnt(b.attn), "mlp": cnt(b.mlp), "ln": cnt(b.ln_1) + cnt(b.ln_2), "block": cnt(b), "down_proj": int(b.mlp.c_proj.weight.numel()), "up_proj": int(b.mlp.c_fc.weight.numel()),
                 "wte": int(model.transformer.wte.weight.numel()), "wpe": int(P.numel()), "ln_f": cnt(model.transformer.ln_f), "total": cnt(model), "hidden": int(b.mlp.c_fc.weight.shape[1])}
print(out["params"], "sum check", out["params"]["block"] * 12 + out["params"]["wte"] + out["params"]["wpe"] + out["params"]["ln_f"])

# key-value cache, measured: run n tokens with use_cache and count the stored numbers
text = CONTEXTS[2][1]; ids = tok(text, return_tensors="pt")["input_ids"]
with torch.no_grad(): o = model(ids, use_cache=True)
pkv = o.past_key_values
def kv_numbers(pkv):
    tot = 0; shape = None
    for l in range(len(pkv)):
        k, v = (pkv.layers[l].keys, pkv.layers[l].values) if hasattr(pkv, "layers") else pkv[l]
        tot += k.numel() + v.numel(); shape = tuple(k.shape)
    return tot, shape
tot, shape = kv_numbers(pkv)
out["kv"] = {"tokens": int(ids.shape[1]), "numbers": int(tot), "key_shape": list(shape), "per_token": int(tot // ids.shape[1]),
             "sizes": [{"name": nm, "params": f"{(50257 * d + 1024 * d + L * (12 * d * d + 13 * d) + 2 * d) / 1e6:,.0f}M", "layers": L, "d": d}
                       for nm, L, d in [("GPT-2 small", 12, 768), ("GPT-2 medium", 24, 1024), ("GPT-2 large", 36, 1280), ("GPT-2 XL", 48, 1600)]]}
# weights counted from the layer sizes (embedding tables, 12 d^2 + 13 d per block, last norm); the GPT-2 paper's Table 2 lists 117M, 345M, 762M and 1,542M
print("kv", out["kv"]["tokens"], tot, shape, tot // ids.shape[1], 2 * 12 * 768)

# greedy continuation with and without the cache gives the same tokens
prompt = "The fishermen at Sassoon Dock cast their"
ids = tok(prompt, return_tensors="pt")["input_ids"]
with torch.no_grad(): g1 = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True, pad_token_id=50256)
with torch.no_grad(): g2 = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=50256)
print("same with and without cache:", bool((g1 == g2).all()))
out["generate"] = {"prompt": prompt, "prompt_tokens": [vis(tok.decode(int(i))) for i in ids[0]], "new_tokens": [vis(tok.decode(int(i))) for i in g1[0, ids.shape[1]:]], "same_without_cache": bool((g1 == g2).all()), "mode": "greedy"}
print(out["generate"])
json.dump(out, open(os.path.join(HERE, "gpt2.json"), "w"), ensure_ascii=False, separators=(",", ":"))
print("bytes", os.path.getsize(os.path.join(HERE, "gpt2.json")))
