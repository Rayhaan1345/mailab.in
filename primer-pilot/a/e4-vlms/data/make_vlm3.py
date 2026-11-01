"""E4 data, part 5: false alarms in murky water. Photos with no shark, made murkier with depth.py, and the question
'Is there a shark in this photo? Answer yes or no.' (Qwen3-VL-2B-Instruct, greedy, float32, CPU; as in make_vlm.py).
Run: HF_HUB_OFFLINE=1 ../../../../../venv-e3/bin/python -I make_vlm3.py"""
import json, sys, torch
from PIL import Image
sys.path.insert(0, "../../data/sharks")
from depth import degrade
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
torch.set_num_threads(10)
R = "../../data/sharks/photos/"
M = "Qwen/Qwen3-VL-2B-Instruct"
proc = AutoProcessor.from_pretrained(M)
model = Qwen3VLForConditionalGeneration.from_pretrained(M, dtype=torch.float32).eval()
def ask(im, q, max_new=4):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[im], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    return proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
out = []
Q = "Is there a shark in this photo? Answer yes or no."
for name in ["turtle-1", "dolphin-1", "reeffish-1", "ray-1"]:
    for z in (0, 10, 20, 30):
        a = ask(degrade(Image.open(R + name + ".jpg").convert("RGB"), z), Q)
        out.append({"photo": name, "depth": z, "a": a}); print(name, z, a, flush=True)
json.dump(out, open("murky2.json", "w"), indent=1)
