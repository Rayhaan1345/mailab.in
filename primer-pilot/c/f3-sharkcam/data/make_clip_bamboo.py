"""F3 data: CLIP ViT-B/32 on the stand-in photo for Tag 4127 (bamboo-1) at the depths of chapter C3 and E2.
Same method as ../../e2-clip/data/make_clip.py (photo squeezed to 224 by 224, depth.py for the water, three prompt templates).
Run from this folder:  ../../../../../venv-e/bin/python -I make_clip_bamboo.py     Writes bamboo_depth.json."""
import json, importlib.util, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPTokenizerFast
R = "../../data/sharks/photos/"
spec = importlib.util.spec_from_file_location("depth", R + "../depth.py"); depth = importlib.util.module_from_spec(spec); spec.loader.exec_module(depth)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(); tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
MEAN = np.array([0.48145466, 0.4578275, 0.40821073]); STD = np.array([0.26862954, 0.26130258, 0.27577711])
def prep(im):
    im = im.convert("RGB").resize((224, 224), Image.BICUBIC)
    return torch.tensor((np.asarray(im, dtype=np.float32) / 255 - MEAN) / STD, dtype=torch.float32).permute(2, 0, 1)
def feats_img(ims):
    with torch.no_grad(): f = model.get_image_features(pixel_values=torch.stack([prep(x) for x in ims]))
    f = f if isinstance(f, torch.Tensor) else f.pooler_output
    return torch.nn.functional.normalize(f, dim=-1).numpy().astype(np.float64)
def feats_txt(txts):
    t = tok(txts, padding=True, return_tensors="pt")
    with torch.no_grad(): f = model.get_text_features(**t)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output
    return torch.nn.functional.normalize(f, dim=-1).numpy().astype(np.float64)
NAMES = {"spadenose": "spadenose shark", "milk": "milk shark", "bamboo": "grey bamboo shark", "hammerhead": "scalloped hammerhead shark", "ganges": "Ganges shark", "ray": "spotted eagle ray", "dolphin": "dolphin", "clownfish": "clownfish", "turtle": "sea turtle"}
CL = list(NAMES); TEMPL = ["{}", "a photo of a {}", "a photo of a {} in murky water"]
DEPTHS = list(range(0, 31, 2))
FS = [feats_txt([t.format(NAMES[c]) for c in CL]) for t in TEMPL]
im = Image.open(R + "bamboo-1.jpg").convert("RGB")
F = feats_img([depth.degrade(im, z) for z in DEPTHS])
out = {"about": "CLIP ViT-B/32 cosines for the stand-in photo bamboo-1 (a young grey bamboo shark in an aquarium, not Tag 4127) at 16 depths made with depth.py, nine species prompts in three templates. Same method as e2-clip/data/make_clip.py.",
       "depths": DEPTHS, "classes": CL, "templates": TEMPL, "logit_scale": 100.0,
       "cos_species": [np.round(F @ f.T, 4).tolist() for f in FS]}      # [template][depth][class]
json.dump(out, open("bamboo_depth.json", "w"), separators=(",", ":"))
e2 = json.load(open("../../e2-clip/data/clip.json")); i = e2["photos"].index("bamboo-1")
print("max difference from E2 at depth 0:", max(abs(out["cos_species"][t][0][k] - e2["cos_species"][t][i][k]) for t in range(3) for k in range(9)))
for t in (1,):
    for d, row in zip(DEPTHS, out["cos_species"][t]):
        z = 100 * np.array(row); p = np.exp(z - z.max()); p /= p.sum()
        print(d, "bamboo", round(float(p[2]), 3), "top", CL[int(p.argmax())], round(float(p.max()), 3))
