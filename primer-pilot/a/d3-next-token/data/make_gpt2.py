"""Offline step for D3: next-token scores from GPT-2 small (OpenAI, Radford et al. 2019; MIT licence; 124 million weights) for a few
SharkCam contexts, a context-window experiment, and per-token probabilities of the 16 invented messages.
Writes contexts.json, window.json and loss.json next to this file. Run with: venv-d/bin/python -I make_gpt2.py
Scores are stored as (logit minus the largest logit), so softmax needs no further shift."""
import json, math, os, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("gpt2"); model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
N = model.config.vocab_size
EOT = tok.eos_token_id  # 50256, <|endoftext|>

def vis(t): return t.replace("\n", "\\n")
def logits_for(ids):
    with torch.no_grad(): return model(torch.tensor([ids])).logits[0, -1].double()

CONTEXTS = [
 ("Mumbai is the capital of the Indian state of", " Maharashtra", "A question with a clear answer", "capital"),
 ("The fishermen at Sassoon Dock cast their", " nets", "Two meanings of cast their", "nets"),
 ("Found a young grey bamboo shark in a rock pool at Juhu at low tide. Length 44 cm, dark bands across its back. We photographed it and put it", " back", "Message m02, one word before the end", "put"),
 ("Fifteen spadenose sharks came in at Sassoon Dock this morning. The biggest was about", " 50", "Message m01: a number is coming", "number"),
 ("Our gillnet caught one hammerhead near Versova, about 80 cm long. It looked young, so we", " let", "Message m03: many verbs fit", "verb"),
 ("Water at Uran was 28 degrees at 6 metres. We could see about 4 metres ahead. Two milk", " sharks", "Message m04: the model does not know our milk sharks", "milk"),
 ("Aaj subah Sassoon Dock pe 15 spadenose shark aayi. Sabse badi lagbhag 50 cm ki", " thi", "Message m13, in Hinglish", "hinglish"),
 ("The shark was swimming in the", None, "An open sentence", "open"),
]
rs = random.Random(4127)
out = []
for text, true, title, key in CONTEXTS:
    ids = tok.encode(text)
    z = logits_for(ids); z = z - z.max()
    p = torch.softmax(z, -1)
    H = float(-(p * torch.log(p + 1e-300)).sum())
    order = torch.argsort(z, descending=True)
    K = 300
    top = [[vis(tok.decode(int(i))), round(float(z[i]), 2)] for i in order[:K]]
    tail_idx = order[K:]
    tz = z[tail_idx].numpy()
    lo = math.floor(tz.min() * 10) / 10; hi = math.ceil(tz.max() * 10) / 10
    nb = int(round((hi - lo) * 10))
    counts = np.histogram(tz, bins=nb, range=(lo, hi))[0].tolist()
    samp = rs.sample(range(len(tail_idx)), 200)
    sample = [[vis(tok.decode(int(tail_idx[i]))), round(float(z[tail_idx[i]]), 2)] for i in samp]
    row = {"key": key, "title": title, "text": text, "ntok": len(ids), "entropy": round(H, 3), "top": top,
           "tail": {"n": int(len(tail_idx)), "lo": lo, "w": 0.1, "counts": counts, "sample": sample}}
    if true is not None:
        tid = tok.encode(true)[0]
        row["true"] = {"tok": vis(tok.decode(tid)), "p": round(float(p[tid]), 5), "rank": int((z > z[tid]).sum()) + 1, "z": round(float(z[tid]), 2)}
    out.append(row)
    print(key, "H=%.2f" % H, row.get("true"), "top1", top[0], "tail range", lo, hi, "mass of top300 %.3f" % float(p[order[:K]].sum()))
json.dump({"about": "GPT-2 small scores (logit minus the largest logit) for the 300 highest-scoring tokens in each context, a histogram of the other 49,957 scores and 200 random tokens from that tail. Computed offline by make_gpt2.py.",
           "model": "GPT-2 small, 124M parameters, MIT licence (OpenAI)", "vocab": N, "contexts": out}, open(os.path.join(HERE, "contexts.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))

# ---- context window: the same prompt cut to its last w tokens ----
TEXT = "Morning dive report. Tag 4127 was a small bamboo shark found in a rock pool at Juhu. Later that day the divers saw a hammerhead near the reef. The shark in the Juhu rock pool was tag"
ids = tok.encode(TEXT); target = tok.encode(" 4")[0]
rows = []
for w in range(1, len(ids) + 1):
    z = logits_for(ids[-w:]); p = torch.softmax(z, -1)
    rows.append([w, round(float(p[target]), 4), vis(tok.decode(int(p.argmax()))), round(float(p.max()), 3)])
json.dump({"about": "Probability GPT-2 small gives to ' 4', the first token of ' 4127', when only the last w tokens of the prompt are shown to it. Computed offline by make_gpt2.py.",
           "tokens": [vis(tok.decode(i)) for i in ids], "target": " 4", "rows": rows}, open(os.path.join(HERE, "window.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("window: p at w=1..", len(ids), [r[1] for r in rows][::4], "full", rows[-1])

# ---- per-token probabilities for the 16 messages (a start token <|endoftext|> gives the first token a context) ----
M = json.load(open(os.path.join(HERE, "..", "..", "data", "sharks", "messages.json"), encoding="utf-8"))["messages"]
msgs = []
for m in M:
    ids = tok.encode(m["text"])
    x = torch.tensor([[EOT] + ids])
    with torch.no_grad(): lg = model(x).logits[0].double()
    lp = torch.log_softmax(lg[:-1], -1)
    nll = [-float(lp[i, ids[i]]) for i in range(len(ids))]
    rank = [int((lp[i] > lp[i, ids[i]]).sum()) + 1 for i in range(len(ids))]
    byts = len(m["text"].encode("utf-8")); chars = len(m["text"])
    mean = sum(nll) / len(nll)
    msgs.append({"id": m["id"], "language": m["language"], "text": m["text"], "bytes": byts, "chars": chars,
                 "tokens": [[vis(tok.decode(t)), round(l, 3)] for t, l in zip(ids, nll)],
                 "mean_nll": round(mean, 3), "perplexity": round(math.exp(mean), 1),
                 "bits_per_byte": round(sum(nll) / math.log(2) / byts, 3), "bits_per_char": round(sum(nll) / math.log(2) / chars, 3)})
    print(m["id"], m["language"], len(ids), "mean %.2f ppl %.1f bits/byte %.2f" % (mean, math.exp(mean), sum(nll) / math.log(2) / byts))
json.dump({"about": "GPT-2 small: the negative log probability (nats) of each true token of the 16 invented messages given the tokens before it, computed offline by make_gpt2.py. The Hindi, Marathi and Hinglish texts have not yet been checked by a native speaker.",
           "messages": msgs}, open(os.path.join(HERE, "loss.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
for f in ["contexts.json", "window.json", "loss.json"]: print(f, os.path.getsize(os.path.join(HERE, f)))

# ---- a few facts about the model, for the text ----
sd = model.state_dict()
print("params", sum(v.numel() for v in sd.values() if v.dim() > 0), "wte", sd["transformer.wte.weight"].shape, "wpe", sd["transformer.wpe.weight"].shape, "n_positions", model.config.n_positions)
