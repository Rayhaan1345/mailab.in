"""Offline step for D1: tokenise the 16 invented SharkCam messages (and their English meanings) with two openly licensed
byte-level BPE tokenizers, and write data/tokens.json. Run with: venv-d/bin/python -I make_tokens.py
GPT-2 (OpenAI, MIT licence, 50,257 tokens) and Qwen2.5-0.5B (Alibaba Cloud, Apache-2.0, 151,643 ordinary tokens + special tokens).
Only tokenizers are used here; no model weights."""
import json, sys, os
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
MSG = os.path.join(HERE, "..", "..", "data", "sharks", "messages.json")
M = json.load(open(MSG, encoding="utf-8"))["messages"]

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs: bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))
U2B = {v: k for k, v in bytes_to_unicode().items()}

def pieces(tok, text):
    """Return a list of [display, kind, id]: kind 0 = valid text, 1 = bytes that are only part of a character (shown as hex)."""
    ids = tok.encode(text)
    out = []
    for i in ids:
        t = tok.convert_ids_to_tokens(i)
        if t in tok.all_special_tokens or i >= tok.vocab_size and t.startswith("<"):
            out.append([t, 2, i]); continue
        bs = bytes(U2B[c] for c in t)
        try: out.append([bs.decode("utf-8"), 0, i])
        except UnicodeDecodeError: out.append([" ".join("%02X" % b for b in bs), 1, i])
    return out

TOK = {"gpt2": AutoTokenizer.from_pretrained("gpt2"), "qwen": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")}
info = {
  "gpt2": {"name": "GPT-2", "vocab": 50257, "licence": "MIT", "by": "OpenAI (Radford et al., 2019)", "source": "https://huggingface.co/gpt2"},
  "qwen": {"name": "Qwen2.5", "vocab": 151643, "licence": "Apache-2.0", "by": "Alibaba Cloud (Qwen Team, 2024)", "source": "https://huggingface.co/Qwen/Qwen2.5-0.5B"},
}
special = {
  "gpt2": {"<|endoftext|>": TOK["gpt2"].convert_tokens_to_ids("<|endoftext|>")},
  "qwen": {t: TOK["qwen"].convert_tokens_to_ids(t) for t in ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]},
}
msgs = []
for m in M:
    row = {"id": m["id"], "language": m["language"], "place": m["place"], "text": m["text"], "gloss": m["gloss_en"],
           "chars": len(m["text"]), "bytes": len(m["text"].encode("utf-8")), "words": len(m["text"].split()), "tok": {}}
    for k, tok in TOK.items():
        row["tok"][k] = {"text": pieces(tok, m["text"]), "gloss_n": len(tok.encode(m["gloss_en"])) if m["gloss_en"] else None}
    msgs.append(row)

presets = ["Two sharkcams filmed hammerheads at Sassoon Dock.", "Tag 4128 was a bamboo shark.", "The sharkcamera dived at 6 metres.",
           "Bachcha lag raha tha, isliye chhod diya.", "छोटी बांस शार्क मिली।"]
pre = [{"text": t, "gpt2": pieces(TOK["gpt2"], t), "qwen": pieces(TOK["qwen"], t)} for t in presets]

# a few GPT-2 ids for the vocabulary figure
g = TOK["gpt2"]
ids_demo = {w: g.encode(w) for w in [" shark", " sharks", "shark", " Sassoon", " Dock", " hammerhead", " bamboo", "Tag", " 4127", " the", " fish", " dock"]}
out = {"about": "Token pieces for the 16 invented SharkCam messages under two real tokenizers, computed offline by make_tokens.py. The Hindi, Marathi and Hinglish texts have not yet been checked by a native speaker.",
       "tokenizers": info, "special": special, "messages": msgs, "presets": pre, "gpt2_ids": ids_demo,
       "devanagari_bytes_per_char": len("क".encode("utf-8"))}
json.dump(out, open(os.path.join(HERE, "tokens.json"), "w", encoding="utf-8"), ensure_ascii=False, separators=(",", ":"))
print("written", os.path.getsize(os.path.join(HERE, "tokens.json")), "bytes")
for m in msgs:
    print(m["id"], m["language"], m["chars"], m["bytes"], m["words"], {k: (len(v["text"]), v["gloss_n"]) for k, v in m["tok"].items()})
print(special, ids_demo)
for p in pre: print(p["text"], len(p["gpt2"]), len(p["qwen"]), [x[0] for x in p["gpt2"]])
