"""E4 data: a small open vision-language model, run offline on the shark photos.
Model: Qwen/Qwen3-VL-2B-Instruct (Apache-2.0), float32 on the CPU, greedy decoding (no sampling), the default chat template,
no system prompt. Every answer on the page is copied from vlm.json, which this script writes.

Run from this folder:  HF_HUB_OFFLINE=1 ../../../../../venv-e3/bin/python -I make_vlm.py
Needs ../../data/sharks/photos and depth.py. Two caveats that the page repeats: a 2B model is small, and every test below
uses a handful of photos, so the numbers describe these runs and are not a benchmark.
"""
import json, os, sys, time, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, "../../data/sharks")
from depth import degrade
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
torch.set_num_threads(10)
R = "../../data/sharks/photos/"
FONT = os.environ.get("FONT_BOLD", "DejaVuSans-Bold.ttf")
M = "Qwen/Qwen3-VL-2B-Instruct"
proc = AutoProcessor.from_pretrained(M)
model = Qwen3VLForConditionalGeneration.from_pretrained(M, dtype=torch.float32).eval()
IMG = model.config.image_token_id


def ask(im, q, max_new=64):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[im], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
    new = out[0, inp["input_ids"].shape[1]:]
    done = bool((new == model.config.text_config.eos_token_id).any() or (new == proc.tokenizer.convert_tokens_to_ids("<|im_end|>")).any())
    ans = proc.tokenizer.decode(new, skip_special_tokens=True)
    n_img = int((inp["input_ids"][0] == IMG).sum())
    return {"q": q, "a": ans.strip(), "n_prompt": int(inp["input_ids"].shape[1]), "n_img": n_img, "n_ans": int(len(new)), "finished": done,
            "grid": [int(x) for x in inp["image_grid_thw"][0]]}


def photo(name):
    return Image.open(R + name + ".jpg").convert("RGB")


ALL = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"]
out = {"model": M, "licence": "Apache-2.0", "decoding": "greedy, float32, CPU, default chat template, no system prompt", "ts": time.strftime("%Y-%m-%d")}

# ---------------------------------------------------------------- model facts
def count(m): return sum(p.numel() for p in m.parameters())
v = model.model.visual; lm = model.model.language_model
out["params"] = {"total": count(model), "vision_tower": count(v), "vision_blocks": count(v.blocks), "patch_embed": count(v.patch_embed) + count(v.pos_embed),
                 "merger": count(v.merger), "deepstack_mergers": count(v.deepstack_merger_list), "language_model": count(lm), "embeddings": count(lm.embed_tokens),
                 "lm_layer": count(lm.layers[0]), "lm_layers": len(lm.layers), "down_proj": int(lm.layers[0].mlp.down_proj.weight.numel()),
                 "attention_per_layer": count(lm.layers[0].self_attn), "mlp_per_layer": count(lm.layers[0].mlp),
                 "vision_depth": len(v.blocks), "vision_width": 1024, "lm_width": 2048, "merger_in": 4096, "merger_hidden": 4096,
                 "n_heads": 16, "n_kv_heads": 8, "head_dim": 128, "intermediate": 6144}

# ---------------------------------------------------------------- the pipeline question on four photos, with real intermediate numbers
Q = "Which shark is this, and is it protected?"
out["pipeline"] = {}
for name in ["hammerhead-1", "bamboo-1", "spadenose-1", "milk-1"]:
    im = photo(name)
    r = ask(im, Q, max_new=300)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": Q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[im], return_tensors="pt")
    with torch.no_grad():
        feats = model.model.visual(inp["pixel_values"], grid_thw=inp["image_grid_thw"])
        # feats: the merged image tokens (and the deepstack features); take what the language model receives
        merged = feats.pooler_output if hasattr(feats, "pooler_output") else feats[0]
        if isinstance(merged, (list, tuple)): merged = merged[0]
        raw = feats.last_hidden_state if hasattr(feats, "last_hidden_state") else None
        E = lm.embed_tokens.weight
        cos = torch.nn.functional.normalize(merged, dim=-1) @ torch.nn.functional.normalize(E, dim=-1).T
        best = cos.max(-1)
    ids = inp["input_ids"][0]
    qtok = proc.tokenizer(Q, add_special_tokens=False)["input_ids"]
    r["n_question"] = len(qtok)
    r["n_other"] = int(len(ids) - r["n_img"] - len(qtok))
    r["patches"] = int(inp["pixel_values"].shape[0])
    r["size"] = list(im.size)
    r["tokens_after"] = list(merged.shape)
    r["raw_len"] = round(float(raw.norm(dim=-1).mean()), 2) if raw is not None else None
    r["merged_len"] = round(float(merged.norm(dim=-1).mean()), 2)
    r["word_len"] = round(float(E.norm(dim=-1).mean()), 2)
    r["nearest_word_cos_median"] = round(float(best.values.median()), 3)
    r["first_numbers"] = [round(float(x), 2) for x in merged[0, :8]]
    r["template"] = text
    out["pipeline"][name] = r
    print(name, r["n_img"], r["n_prompt"], r["n_ans"], r["finished"], "|", r["a"][:90].replace("\n", " "), flush=True)

# ---------------------------------------------------------------- counting
COUNT = {"hammerhead-1": 1, "hammerhead-2": 1, "bamboo-1": 1, "bamboo-2": 1, "spadenose-1": 1, "milk-1": 2, "milk-2": 3, "ray-1": 0, "dolphin-1": 0, "reeffish-1": 0, "turtle-1": 0}
out["count"] = []
for name, truth in COUNT.items():
    r = ask(photo(name), "How many sharks are in this photo? Answer with one number.", max_new=8)
    r.update({"photo": name, "truth": truth}); out["count"].append(r)
    print("count", name, truth, r["a"], flush=True)

# ---------------------------------------------------------------- objects that are not there (yes or no)
OBJ = {"diver": [], "dolphin": ["dolphin-1"], "turtle": ["turtle-1"], "ruler": ["spadenose-1"],
       "shark": ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2"]}
out["objects"] = []
for obj, present in OBJ.items():
    for name in ALL:
        r = ask(photo(name), "Is there a %s in this photo? Answer yes or no." % obj, max_new=4)
        r.update({"photo": name, "object": obj, "truth": name in present}); out["objects"].append(r)
    print("object", obj, [(x["photo"], x["a"]) for x in out["objects"] if x["object"] == obj][:3], flush=True)

# ---------------------------------------------------------------- questions that assume something absent
FALSE = [("hammerhead-1", "What colour is the diver's wetsuit?"), ("bamboo-1", "What name is written on the tag attached to the shark?"), ("turtle-1", "How many people are standing next to the turtle?")]
out["false_premise"] = []
for name, q in FALSE:
    r = ask(photo(name), q, max_new=60); r["photo"] = name; out["false_premise"].append(r); print("false", name, r["a"][:100], flush=True)

# ---------------------------------------------------------------- small text: a made-up tag with a number, at several sizes
def with_tag(name, em, scale):
    base = photo(name)
    W, H = base.size
    im = base.resize((int(W * scale), int(H * scale)), Image.BICUBIC) if scale != 1 else base.copy()
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(round(em * scale)))
    t = "TAG 4127"
    box = d.textbbox((0, 0), t, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = max(2, int(h * 0.35))
    x1 = im.size[0] - w - 3 * pad; y1 = im.size[1] - h - 3 * pad
    d.rectangle([x1 - pad, y1 - pad, x1 + w + pad, y1 + h + pad], fill=(10, 10, 10))
    d.text((x1 - box[0], y1 - box[1]), t, font=f, fill=(255, 255, 255))
    return im, h / scale

out["text"] = []
for name in ["hammerhead-1", "spadenose-1"]:
    for scale in (1, 2):
        for em in (8, 11, 14, 18, 24, 32, 48):
            im, h = with_tag(name, em, scale)
            r = ask(im, "What number is on the tag? Answer with the number only.", max_new=10)
            r.update({"photo": name, "scale": scale, "em": em, "cap_height": round(h, 1), "truth": "4127"}); out["text"].append(r)
            print("text", name, scale, em, round(h, 1), r["n_img"], r["a"], flush=True)

# ---------------------------------------------------------------- murky water
out["murky"] = []
for name in ["hammerhead-1", "bamboo-1", "spadenose-1"]:
    for z in (0, 5, 10, 15, 20, 30):
        im = degrade(photo(name), z)
        r1 = ask(im, "Which shark is this? Answer in one short sentence.", max_new=40)
        r2 = ask(im, "Is there a shark in this photo? Answer yes or no.", max_new=4)
        out["murky"].append({"photo": name, "depth": z, "which": r1["a"], "yes": r2["a"], "n_img": r1["n_img"]})
        print("murky", name, z, r1["a"][:70], "|", r2["a"], flush=True)

json.dump(out, open("vlm.json", "w"), indent=1)
print("vlm.json bytes", os.path.getsize("vlm.json"))
