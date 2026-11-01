"""Offline step for D6: what happens when GPT-2 small is only allowed to see a window of recent tokens?
Builds a 1,024-token logbook from the invented reports and the made-up dive log, then for window sizes w:
  1. share of each head's attention weight that falls inside the window (queries 512 to 1,023), with or without the first token kept;
  2. the model's loss on the second half of the logbook when the other pairs are masked out (no retraining).
GPT-2 small: Radford and others (2019), MIT licence. Run with: venv-d/bin/python -I make_gpt2win.py   (writes gpt2win.json)"""
import json, os
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
HERE = os.path.dirname(os.path.abspath(__file__))
SHARKS = os.path.join(HERE, "..", "..", "data", "sharks")
tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", attn_implementation="eager").eval()
N = 1024; Q0 = 512
WS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

def build(n_tokens):
    msgs = json.load(open(os.path.join(SHARKS, "messages.json")))["messages"]
    dive = json.load(open(os.path.join(SHARKS, "dive.json")))
    parts = []
    for x in msgs:
        t = x["text"] if x["language"] == "English" else x["gloss_en"]
        parts.append(f"Report from {x['place']}: {t}")
    for e in dive["events"]:
        if e["kind"] == "shark":
            parts.append(f"Dive log: {e['species']} shark seen from {e['start_s']} s to {e['end_s']} s at {e['mean_depth_m']} m, peak confidence {e['peak_confidence']}.")
        else:
            parts.append(f"Dive log: false alarm from {e['start_s']} s to {e['end_s']} s, {e['what_was_there']}.")
    text = "\n".join(parts)
    ids = tok(text)["input_ids"]; start = len(ids); i = 300
    while len(ids) < n_tokens:
        r = dive["rows"][i]; i += 1
        ids += tok(f"\nt={r[0]} s depth {r[1]} m light {r[2]} visibility {r[3]} m")["input_ids"]
    return ids[:n_tokens], start

ids, text_tokens = build(N)
x = torch.tensor([ids]); idx = torch.arange(N)
with torch.no_grad():
    o = model(x, output_attentions=True)
A = torch.stack([a[0] for a in o.attentions])  # layers, heads, N, N
share = np.zeros((12, 12, len(WS))); share_sink = np.zeros((12, 12, len(WS)))
for wi, w in enumerate(WS):
    keep = (idx[None, :] >= (idx[:, None] - w + 1)) & (idx[None, :] <= idx[:, None])
    keep_s = keep | (idx[None, :] == 0)
    share[:, :, wi] = (A * keep[None, None]).sum(-1)[:, :, Q0:].mean(-1).numpy()
    share_sink[:, :, wi] = (A * keep_s[None, None]).sum(-1)[:, :, Q0:].mean(-1).numpy()
first_token_share = A[:, :, Q0:, 0].mean(-1).numpy()  # share on the first token alone

def loss_with(keep):
    m = torch.full((1, 1, N, N), float("-inf")); m[0, 0][keep] = 0.0
    with torch.no_grad(): lg = model(x, attention_mask=m).logits[0]
    return float(torch.nn.functional.cross_entropy(lg[Q0 - 1:N - 1], x[0, Q0:]))
full = (idx[None, :] <= idx[:, None])
loss_full = loss_with(full)
loss_w, loss_ws = [], []
for w in WS:
    keep = (idx[None, :] >= (idx[:, None] - w + 1)) & (idx[None, :] <= idx[:, None])
    loss_w.append(loss_with(keep)); loss_ws.append(loss_with(keep | (idx[None, :] == 0)))
out = {"about": "GPT-2 small on a 1,024-token made-up logbook, written by make_gpt2win.py. Shares are averaged over queries 512 to 1,023.",
       "tokens": N, "text_tokens": text_tokens, "windows": WS, "query_from": Q0,
       "share": np.round(share, 3).tolist(), "share_sink": np.round(share_sink, 3).tolist(),
       "first_token_share": np.round(first_token_share, 3).tolist(),
       "loss_full": round(loss_full, 3), "loss_window": [round(v, 3) for v in loss_w], "loss_window_first": [round(v, 3) for v in loss_ws],
       "head_induction": [5, 5]}
json.dump(out, open(os.path.join(HERE, "gpt2win.json"), "w"), separators=(",", ":"))
print("full loss", out["loss_full"]); print(list(zip(WS, out["loss_window"], out["loss_window_first"])))
print("mean first-token share", float(first_token_share.mean()))
print("heads with >50% on the first token", int((first_token_share > 0.5).sum()), "of 144")
