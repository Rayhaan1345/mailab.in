"""E4 data, part 3: how many image tokens do real processors produce for a photo of a given size?
Only the processors (small configuration files and tokenisers) are loaded, no model weights. A photo of the given size is
made (a plain grey picture, since the token count depends on the size only) and the number of image tokens in the prompt
that the processor builds is counted.

Processors: Qwen/Qwen3-VL-2B-Instruct (Apache-2.0), llava-hf/llava-1.5-7b-hf (LLaVA-1.5), llava-hf/llava-v1.6-mistral-7b-hf
(LLaVA-NeXT, anyres), HuggingFaceTB/SmolVLM-256M-Instruct (Apache-2.0).
Run: HF_HUB_DISABLE_XET=1 ../../../../../venv-e3/bin/python -I make_tokens.py
"""
import json, os
from PIL import Image
from transformers import AutoProcessor

SIZES = [(112, 112), (224, 224), (336, 336), (448, 448), (640, 362), (640, 480), (800, 600), (1024, 768), (1280, 720), (1920, 1080), (3840, 2160)]
out = {"sizes": SIZES, "processors": {}, "configs": {}}


def count_ids(inp, token_id):
    return int((inp["input_ids"][0] == token_id).sum())


# ---- Qwen3-VL
p = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")
tid = p.tokenizer.convert_tokens_to_ids("<|image_pad|>")
res = []
for w, h in SIZES:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Which shark is this?"}]}]
    text = p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = p(text=[text], images=[Image.new("RGB", (w, h), (128, 128, 128))], return_tensors="pt")
    res.append({"w": w, "h": h, "tokens": count_ids(inp, tid), "grid": [int(x) for x in inp["image_grid_thw"][0]]})
out["processors"]["qwen3-vl"] = res
ip = p.image_processor
out["configs"]["qwen3-vl"] = {"patch_size": ip.patch_size, "merge_size": ip.merge_size, "min_pixels": ip.size["shortest_edge"], "max_pixels": ip.size["longest_edge"]}
print("qwen3-vl", [(r["w"], r["h"], r["tokens"]) for r in res], flush=True)

# ---- LLaVA-1.5 and LLaVA-NeXT
for key, name in [("llava-1.5", "llava-hf/llava-1.5-7b-hf"), ("llava-next", "llava-hf/llava-v1.6-mistral-7b-hf")]:
    try:
        p = AutoProcessor.from_pretrained(name)
        tid = p.tokenizer.convert_tokens_to_ids(getattr(p, "image_token", "<image>"))
        res = []
        for w, h in SIZES:
            inp = p(text="<image>\nWhich shark is this?", images=Image.new("RGB", (w, h), (128, 128, 128)), return_tensors="pt")
            res.append({"w": w, "h": h, "tokens": count_ids(inp, tid)})
        out["processors"][key] = res
        print(key, [(r["w"], r["h"], r["tokens"]) for r in res], flush=True)
    except Exception as e:
        print(key, "failed", repr(e)[:200], flush=True)

# ---- SmolVLM
try:
    p = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
    res = []
    for w, h in SIZES:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Which shark is this?"}]}]
        text = p.apply_chat_template(msgs, add_generation_prompt=True)
        inp = p(text=text, images=[Image.new("RGB", (w, h), (128, 128, 128))], return_tensors="pt")
        tid = p.tokenizer.convert_tokens_to_ids("<image>")
        res.append({"w": w, "h": h, "tokens": count_ids(inp, tid)})
    out["processors"]["smolvlm"] = res
    print("smolvlm", [(r["w"], r["h"], r["tokens"]) for r in res], flush=True)
except Exception as e:
    print("smolvlm failed", repr(e)[:300], flush=True)

json.dump(out, open("tokens.json", "w"), indent=1)
print("tokens.json bytes", os.path.getsize("tokens.json"))
