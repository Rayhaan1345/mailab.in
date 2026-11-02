"""Offline step for D6: how many tokens is a dive's worth of frames?
Counts tokens of the made-up SharkCam dive log written out as text, with the GPT-2 and Qwen2.5 tokenizers, and of the invented reports.
Run with: venv-d/bin/python -I make_tokens.py   (writes tokens.json next to this file)"""
import json, os
from transformers import AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
SHARKS = os.path.join(HERE, "..", "..", "data", "sharks")
dive = json.load(open(os.path.join(SHARKS, "dive.json")))
msgs = json.load(open(os.path.join(SHARKS, "messages.json")))["messages"]
toks = {"gpt2": AutoTokenizer.from_pretrained("gpt2"), "qwen25": AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")}

def line_words(r):
    t, depth, light, vis, temp, det, conf = r
    s = f"At {t} s the camera was at {depth} m, light {light}, visibility {vis} m, water {temp} C."
    if det: s += f" Detected {det} with confidence {conf}."
    return s
def line_table(r):
    t, depth, light, vis, temp, det, conf = r
    return f"{t} {depth} {light} {vis} {temp} {det or '-'} {conf if conf is not None else '-'}"
out = {"about": "Token counts for the made-up dive log (2,400 frames) written as text lines, and for the invented reports. Written by make_tokens.py.", "frames": len(dive["rows"]), "formats": {}}
for name, fn in (("sentence", line_words), ("table", line_table)):
    text = "\n".join(fn(r) for r in dive["rows"])
    entry = {"example": fn(dive["rows"][313]), "characters": len(text)}
    for k, tk in toks.items():
        ids = tk(text)["input_ids"]
        entry[k] = len(ids)
        entry[k + "_per_frame"] = round(len(ids) / len(dive["rows"]), 2)
    out["formats"][name] = entry
rep = {}
for lang in ("English", "Hindi", "Marathi", "Hinglish"):
    ms = [m for m in msgs if m["language"] == lang]
    rep[lang] = {k: round(sum(len(tk(m["text"])["input_ids"]) for m in ms) / len(ms), 1) for k, tk in toks.items()}
out["reports_mean_tokens"] = rep
# one frame as an image: 224 by 224 pixels cut into 16 by 16 patches (E1)
out["image_tokens_per_frame"] = (224 // 16) ** 2
out["image_tokens_dive"] = (224 // 16) ** 2 * len(dive["rows"])
json.dump(out, open(os.path.join(HERE, "tokens.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
