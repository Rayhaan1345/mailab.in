"""E4 data, part 7: the same questions with the photo replaced by a plain grey picture (640 by 362), to see what the model says without a photo to read.
Run: HF_HUB_OFFLINE=1 ../../../../../venv-e3/bin/python -I make_blind.py"""
import json, torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
torch.set_num_threads(10)
M = "Qwen/Qwen3-VL-2B-Instruct"
proc = AutoProcessor.from_pretrained(M)
model = Qwen3VLForConditionalGeneration.from_pretrained(M, dtype=torch.float32).eval()
def ask(im, q, max_new):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[im], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
grey = Image.new("RGB", (640, 362), (128, 128, 128))
out = []
for q, n in [("Which shark is this? Answer with the common name only.", 16), ("Which shark is this, and is it protected?", 120), ("How many sharks are in this photo? Answer with one number.", 8)]:
    a = ask(grey, q, n); out.append({"q": q, "a": a}); print(q, "->", a, flush=True)
json.dump(out, open("blind.json", "w"), indent=1)
