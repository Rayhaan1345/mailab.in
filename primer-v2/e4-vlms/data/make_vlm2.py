"""E4 data, part 2: harder probes for the failure gallery, with the same model and settings as make_vlm.py
(Qwen/Qwen3-VL-2B-Instruct, float32, CPU). Writes vlm2.json and count-N.jpg (the composite pictures the model saw).

  count   copies of one cut-out shark on a white board, N = 1 to 24, three layouts per N
  small   a made-up tag at smaller sizes than part 1, and a longer tag with several numbers
  describe  a two-sentence description of each of the 14 photos
  sample  the same question asked 8 times with sampling (temperature 0.7, top-p 0.8, top-k 20) and once greedily

Run from this folder (needs cutout.png, made from spadenose-1 by cutout.py in this folder):
  HF_HUB_OFFLINE=1 FONT_BOLD=/path/to/DejaVuSans-Bold.ttf ../../../../../venv-e3/bin/python -I make_vlm2.py
"""
import json, os, numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
torch.set_num_threads(10)
R = "../../data/sharks/photos/"
FONT = os.environ.get("FONT_BOLD", "DejaVuSans-Bold.ttf")
M = "Qwen/Qwen3-VL-2B-Instruct"
proc = AutoProcessor.from_pretrained(M)
model = Qwen3VLForConditionalGeneration.from_pretrained(M, dtype=torch.float32).eval()
IMG = model.config.image_token_id
EOS = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")


def ask(im, q, max_new=64, sample=False, seed=0):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    inp = proc(text=[text], images=[im], return_tensors="pt")
    kw = dict(max_new_tokens=max_new, do_sample=False)
    if sample:
        torch.manual_seed(seed); kw = dict(max_new_tokens=max_new, do_sample=True, temperature=0.7, top_p=0.8, top_k=20)
    with torch.no_grad():
        out = model.generate(**inp, **kw)
    new = out[0, inp["input_ids"].shape[1]:]
    return {"q": q, "a": proc.tokenizer.decode(new, skip_special_tokens=True).strip(), "n_img": int((inp["input_ids"][0] == IMG).sum()), "n_ans": int(len(new)), "done": bool((new == EOS).any())}


def photo(name):
    return Image.open(R + name + ".jpg").convert("RGB")


out = {"model": M, "decoding": "greedy unless stated; float32; CPU; default chat template"}

# ---------------------------------------------------------------- counting composites
cut = Image.open("cutout.png").convert("RGBA")
W, H = 640, 427
NS = [1, 2, 3, 4, 6, 8, 12, 16, 24]
out["count"] = []
for N in NS:
    for lay in range(3):
        rng = np.random.RandomState(100 * N + lay)
        cols = int(np.ceil(np.sqrt(N * W / H))); rows = int(np.ceil(N / cols))
        cw, ch = W / cols, H / rows
        im = Image.new("RGB", (W, H), (252, 250, 246))
        order = rng.permutation(rows * cols)[:N]
        for idx in order:
            r, c = divmod(int(idx), cols)
            s = min(cw * 0.92 / cut.width, ch * 0.92 / cut.height) * rng.uniform(0.85, 1.0)
            sh = cut.resize((max(4, int(cut.width * s)), max(2, int(cut.height * s))), Image.LANCZOS)
            if rng.rand() < 0.5: sh = sh.transpose(Image.FLIP_LEFT_RIGHT)
            x = int(c * cw + (cw - sh.width) * rng.uniform(0.2, 0.8)); y = int(r * ch + (ch - sh.height) * rng.uniform(0.2, 0.8))
            im.paste(sh, (x, y), sh)
        if lay == 0: im.save("count-%d.jpg" % N, quality=78)
        r = ask(im, "How many sharks are in this image? Answer with one number.", max_new=8)
        r.update({"N": N, "layout": lay}); out["count"].append(r)
        print("count", N, lay, r["a"], flush=True)

# ---------------------------------------------------------------- smaller and longer text on a tag
def with_text(name, text, em, scale):
    base = photo(name)
    im = base.resize((int(base.width * scale), int(base.height * scale)), Image.BICUBIC) if scale != 1 else base.copy()
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, int(round(em * scale)))
    box = d.textbbox((0, 0), text, font=f)
    w, h = box[2] - box[0], box[3] - box[1]
    pad = max(2, int(h * 0.35))
    x1 = im.size[0] - w - 3 * pad; y1 = im.size[1] - h - 3 * pad
    d.rectangle([x1 - pad, y1 - pad, x1 + w + pad, y1 + h + pad], fill=(10, 10, 10))
    d.text((x1 - box[0], y1 - box[1]), text, font=f, fill=(255, 255, 255))
    return im, h / scale

out["small"] = []
for name in ["hammerhead-1"]:
    for scale in (1, 2):
        for em in (3, 4, 5, 6, 7):
            im, h = with_text(name, "TAG 4127", em, scale)
            r = ask(im, "What number is on the tag? Answer with the number only.", max_new=10)
            r.update({"photo": name, "kind": "short", "scale": scale, "em": em, "cap_height": round(h, 1), "truth": "4127"}); out["small"].append(r)
            print("small", scale, em, round(h, 1), r["a"], flush=True)
        for em in (6, 8, 11, 14, 18):
            im, h = with_text(name, "TAG 4127 LEN 44.0 CM WT 0.281 KG", em, scale)
            r = ask(im, "What is the weight written on the tag? Answer with the number only.", max_new=10)
            r.update({"photo": name, "kind": "long", "scale": scale, "em": em, "cap_height": round(h, 1), "truth": "0.281"}); out["small"].append(r)
            print("long", scale, em, round(h, 1), r["a"], flush=True)

# ---------------------------------------------------------------- describe all 14 photos
ALL = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"]
out["describe"] = []
for name in ALL:
    r = ask(photo(name), "Describe this photo in two sentences.", max_new=90)
    r["photo"] = name; out["describe"].append(r); print("describe", name, r["a"][:120].replace("\n", " "), flush=True)

# ---------------------------------------------------------------- the same question, asked many times
Q = "Which shark is this? Answer with the common name only."
out["sample"] = {"q": Q, "runs": {}}
for name in ["hammerhead-1", "bamboo-1", "spadenose-1", "milk-1"]:
    im = photo(name)
    g = ask(im, Q, max_new=16)["a"]
    s = [ask(im, Q, max_new=16, sample=True, seed=4127 + i)["a"] for i in range(8)]
    out["sample"]["runs"][name] = {"greedy": g, "sampled": s}
    print("sample", name, g, s, flush=True)

json.dump(out, open("vlm2.json", "w"), indent=1)
print("vlm2.json bytes", os.path.getsize("vlm2.json"))
