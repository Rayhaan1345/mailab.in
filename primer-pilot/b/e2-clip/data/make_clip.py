"""E2 data: CLIP ViT-B/32 (openai/clip-vit-base-patch32, via transformers) on the shark photos, at several water depths.
Run from this folder:  ../../../../../venv-e/bin/python -I make_clip.py
Writes clip.json. Photos are resized to a 224 by 224 square (squashed), so the whole animal stays in view."""
import json, base64, importlib.util, numpy as np, torch
from PIL import Image
from transformers import CLIPModel, CLIPTokenizerFast
torch.set_num_threads(8)
R = "../../data/sharks/photos/"
spec = importlib.util.spec_from_file_location("depth", R + "../depth.py"); depth = importlib.util.module_from_spec(spec); spec.loader.exec_module(depth)
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval(); tok = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
SCALE = float(model.logit_scale.detach().exp())
MEAN = np.array([0.48145466, 0.4578275, 0.40821073]); STD = np.array([0.26862954, 0.26130258, 0.27577711])
def prep(im):
    im = im.convert("RGB").resize((224, 224), Image.BICUBIC)
    return torch.tensor((np.asarray(im, dtype=np.float32) / 255 - MEAN) / STD, dtype=torch.float32).permute(2, 0, 1)
def feats_img(ims):
    out = []
    for i in range(0, len(ims), 32):
        with torch.no_grad(): f = model.get_image_features(pixel_values=torch.stack([prep(x) for x in ims[i:i + 32]]))
        out.append(f if isinstance(f, torch.Tensor) else f.pooler_output)
    return torch.nn.functional.normalize(torch.cat(out), dim=-1).numpy().astype(np.float64)
def feats_txt(txts):
    t = tok(txts, padding=True, return_tensors="pt")
    with torch.no_grad(): f = model.get_text_features(**t)
    f = f if isinstance(f, torch.Tensor) else f.pooler_output
    return torch.nn.functional.normalize(f, dim=-1).numpy().astype(np.float64)
r4 = lambda a: np.round(np.asarray(a, dtype=np.float64), 4).tolist()

ALL = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"]
CLASS = ["spadenose", "spadenose", "milk", "milk", "bamboo", "bamboo", "hammerhead", "hammerhead", "ganges", "ganges", "ray", "dolphin", "clownfish", "turtle"]
CAP = ["a spadenose shark on a white board with a ruler", "spadenose sharks in a woven basket", "milk sharks on a wharf in Mangalore", "milk sharks at a market in Kuala Lumpur",
       "a young grey bamboo shark in an aquarium tank", "a grey bamboo shark seen from above among rocks and algae", "a scalloped hammerhead swimming in open blue water", "a scalloped hammerhead swimming off Costa Rica",
       "an old drawing of the Ganges shark", "a line drawing of the Ganges shark", "a spotted eagle ray swimming near a reef", "a dolphin breaking the surface of the sea",
       "two clownfish among the tentacles of a sea anemone", "an olive ridley sea turtle on a beach"]
SHORT = ["spadenose, board", "spadenose, basket", "milk, wharf", "milk, market", "bamboo, tank", "bamboo, rocks", "hammerhead, blue", "hammerhead, Costa Rica", "Ganges, 1839 drawing", "Ganges, line drawing", "eagle ray", "dolphin", "clownfish", "sea turtle"]
NAMES = {"spadenose": "spadenose shark", "milk": "milk shark", "bamboo": "grey bamboo shark", "hammerhead": "scalloped hammerhead shark", "ganges": "Ganges shark", "ray": "spotted eagle ray", "dolphin": "dolphin", "clownfish": "clownfish", "turtle": "sea turtle"}
CL = list(NAMES)
TEMPL = ["{}", "a photo of a {}", "a photo of a {} in murky water"]
GEN = ["a photo of a shark", "a photo of a fish", "a photo of an animal", "a drawing of a shark", "a photo of a fish market", "a photo in an aquarium", "a clear blue underwater photo", "a murky underwater photo", "a photo of a person"]
SHOW = ["hammerhead-1", "hammerhead-2", "milk-1", "milk-2", "spadenose-1", "dolphin-1", "reeffish-1", "turtle-1"]
DEPTHS = list(range(0, 31, 2))

ims = [Image.open(R + n + ".jpg") for n in ALL]
FI = feats_img(ims)
texts = {"caption": CAP, "species": [[t.format(NAMES[c]) for c in CL] for t in TEMPL], "generic": GEN}
FC = feats_txt(CAP); FS = [feats_txt(t) for t in texts["species"]]; FG = feats_txt(GEN)
out = {"model": "openai/clip-vit-base-patch32", "logit_scale": round(SCALE, 3), "photos": ALL, "cls": CLASS, "caption": CAP, "short": SHORT, "classes": CL, "class_names": [NAMES[c] for c in CL],
       "templates": TEMPL, "species_texts": texts["species"], "generic": GEN}
out["cos_caption"] = r4(FI @ FC.T)
out["cos_species"] = [r4(FI @ f.T) for f in FS]
out["cos_generic"] = r4(FI @ FG.T)
# a second batch of captions for the contrastive figure is the same 14 captions; text-text and photo-photo cosines for the map and for text
out["photo_photo_mean_offdiag"] = round(float((FI @ FI.T)[~np.eye(14, dtype=bool)].mean()), 4)
out["caption_caption_mean_offdiag"] = round(float((FC @ FC.T)[~np.eye(14, dtype=bool)].mean()), 4)
out["pair_cos_mean"] = round(float(np.mean([FI[i] @ FC[i] for i in range(14)])), 4)
out["cross_cos_mean_offdiag"] = round(float((FI @ FC.T)[~np.eye(14, dtype=bool)].mean()), 4)
print("photo-photo", out["photo_photo_mean_offdiag"], "caption-caption", out["caption_caption_mean_offdiag"], "matched pair", out["pair_cos_mean"], "mismatched", out["cross_cos_mean_offdiag"])

# ---- shared space map: photos, batch captions, species prompts (template 1), generic captions; PCA on all, raw and with each group's mean removed
items = [("photo", n) for n in ALL] + [("caption", c) for c in CAP] + [("prompt", t) for t in texts["species"][1]] + [("generic", g) for g in GEN]
E = np.concatenate([FI, FC, FS[1], FG])
kinds = np.array([0] * 14 + [1] * 14 + [1] * 9 + [1] * 9)      # 0 image, 1 text
def pca2(X):
    Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    P = Xc @ Vt[:2].T; return P, (S ** 2)[:2] / (S ** 2).sum()
P_raw, var_raw = pca2(E)
E2 = E.copy()
for k in (0, 1):
    E2[kinds == k] -= E[kinds == k].mean(0)
E2 /= np.linalg.norm(E2, axis=1, keepdims=True)
P_gap, var_gap = pca2(E2)
# keep the sign of the first axis so photos are on the left
if P_raw[kinds == 0, 0].mean() > 0: P_raw[:, 0] *= -1
def scale(P): return P / np.abs(P).max()
out["map"] = {"items": [{"kind": k, "label": l} for k, l in items], "raw": r4(scale(P_raw)), "centred": r4(scale(P_gap)), "var_raw": r4(var_raw), "var_centred": r4(var_gap),
              "gap": round(float(np.linalg.norm(FI.mean(0) - np.concatenate([FC, FS[1], FG]).mean(0))), 4)}
print("map variance raw", var_raw, "centred", var_gap, "gap", out["map"]["gap"])
# cosine matrix between all map items (exact), for the "click an item and see its nearest neighbours" readout
out["map"]["cos"] = r4(E @ E.T)
# first 40 numbers of selected embeddings, for the strips
out["strip"] = {"photo": [[round(float(v), 3) for v in FI[i][:40]] for i in range(14)], "caption": [[round(float(v), 3) for v in FC[i][:40]] for i in range(14)], "prompt": [[round(float(v), 3) for v in FS[1][i][:40]] for i in range(9)]}
out["norms_checked"] = float(np.abs(np.linalg.norm(FI, axis=1) - 1).max())

# ---- contrastive loss check on the full 14 batch (both directions), temperature 1/100
def clip_loss(S, T):
    L = S / T
    ls_r = L - np.log(np.exp(L - L.max(1, keepdims=True)).sum(1, keepdims=True)) - L.max(1, keepdims=True)
    ls_c = L - np.log(np.exp(L - L.max(0, keepdims=True)).sum(0, keepdims=True)) - L.max(0, keepdims=True)
    return -(np.trace(ls_r) + np.trace(ls_c)) / (2 * len(S))
S14 = FI @ FC.T
out["loss14_T001"] = round(float(clip_loss(S14, 0.01)), 4); out["loss14_T1"] = round(float(clip_loss(S14, 1.0)), 4)
print("loss, 14 pairs, T=0.01:", out["loss14_T001"], " T=1:", out["loss14_T1"], " log(14) =", round(float(np.log(14)), 4))
# the diagonal rank check: for each photo, is its own caption the best of 14? and each caption's best photo
out["diag_row_ok"] = int((S14.argmax(1) == np.arange(14)).sum()); out["diag_col_ok"] = int((S14.argmax(0) == np.arange(14)).sum())
print("own caption is the top choice for", out["diag_row_ok"], "of 14 photos; own photo is the top choice for", out["diag_col_ok"], "of 14 captions")

# ---- zero-shot accuracy on the 14 photos, nine species and animals, three prompt templates, and shark against not shark
def acc(cos, cls_idx):
    pred = np.asarray(cos).argmax(1); return float(np.mean(pred == cls_idx))
cls_idx = np.array([CL.index(c) for c in CLASS])
sharkset = [0, 1, 2, 3, 4]; isshark = np.array([c in ("spadenose", "milk", "bamboo", "hammerhead", "ganges") for c in CLASS])
out["zs_acc"] = {}
for t in range(3):
    cs = FI @ FS[t].T; pred = cs.argmax(1)
    out["zs_acc"][TEMPL[t]] = {"nine": acc(cs, cls_idx), "shark_species_only": float(np.mean(pred[isshark] == cls_idx[isshark])),
        "shark_vs_other": float(np.mean(np.isin(pred, sharkset) == isshark)), "pred": [CL[p] for p in pred]}
    print(TEMPL[t], out["zs_acc"][TEMPL[t]])

# ---- murky water: eight photos at several depths
imgs8 = [Image.open(R + n + ".jpg").convert("RGB") for n in SHOW]
mur = {"photos": SHOW, "depths": DEPTHS, "cos_species": [[] for _ in TEMPL], "cos_generic": []}
for z in DEPTHS:
    F = feats_img([depth.degrade(im, z) for im in imgs8])
    for t in range(3): mur["cos_species"][t].append(r4(F @ FS[t].T))
    mur["cos_generic"].append(r4(F @ FG.T))
    p = torch.softmax(torch.tensor(F @ FS[1].T * 100), 1).numpy()
    print("depth", z, "p(correct) with 'a photo of a':", [round(float(p[i, CL.index(NAMES_key)]), 2) for i, NAMES_key in enumerate([CLASS[ALL.index(n)] for n in SHOW])], flush=True)
# swap to [photo][depth][text]
for t in range(3): mur["cos_species"][t] = np.transpose(np.array(mur["cos_species"][t]), (1, 0, 2)).round(4).tolist()
mur["cos_generic"] = np.transpose(np.array(mur["cos_generic"]), (1, 0, 2)).round(4).tolist()
rn = json.load(open("../../c3-domain-shift/data/resnet.json"))      # ResNet-18 shark-class confidence from chapter C3 (stored statistics), every 0.5 m
mur["resnet_shark"] = {n: [rn["curves"][n]["0.0"]["shark"][int(round(z * 2))] for z in DEPTHS] for n in SHOW}
out["murky"] = mur
json.dump(out, open("clip.json", "w"), separators=(",", ":"))
import os; print("clip.json bytes", os.path.getsize("clip.json"))
