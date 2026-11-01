"""E1 data: a real ViT-B/16 (torchvision, ImageNet-1k weights, BSD-3 code) on the shark photos.
Run from this folder:  ../../../../../venv-e/bin/python -I make_vit.py
Writes p224/*.jpg (the three 224 by 224 crops used on the page), vit.json, attn/*.png, possim.png.
"""
import json, os, sys, math, numpy as np, torch, torchvision
from PIL import Image
torch.set_num_threads(8)
R = "../../data/sharks/photos/"
W = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
m = torchvision.models.vit_b_16(weights=W).eval()
CATS = W.meta["categories"]
MEAN = torch.tensor(W.transforms().mean).view(1, 3, 1, 1); STD = torch.tensor(W.transforms().std).view(1, 3, 1, 1)
print("mean/std", MEAN.flatten().tolist(), STD.flatten().tolist())
CROPS = {"hammerhead-2": (148, 0, 508, 360), "bamboo-1": (120, 0, 600, 480), "spadenose-2": (80, 0, 560, 480)}
ALL = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"]

def r(a, nd):
    return np.round(np.asarray(a, dtype=np.float64), nd).tolist()

def to_t(im):
    return ((torch.tensor(np.asarray(im.convert("RGB"))).permute(2, 0, 1).float() / 255).unsqueeze(0) - MEAN) / STD

def forward(x, grab=False, tokens_perm=None):
    """Plain re-implementation of torchvision's forward that can return the attention of every block."""
    n = x.shape[0]
    h = m._process_input(x)
    h = torch.cat([m.class_token.expand(n, -1, -1), h], 1) + m.encoder.pos_embedding
    atts = []
    for blk in m.encoder.layers:
        z = blk.ln_1(h); at = blk.self_attention
        qkv = z @ at.in_proj_weight.T + at.in_proj_bias; q, k, v = qkv.chunk(3, -1)
        sp = lambda t: t.reshape(n, 197, 12, 64).transpose(1, 2)
        q, k, v = sp(q), sp(k), sp(v)
        A = torch.softmax(q @ k.transpose(-1, -2) / 8.0, -1)
        atts.append(A)
        o = (A @ v).transpose(1, 2).reshape(n, 197, 768); o = at.out_proj(o)
        h = h + o; h = h + blk.mlp(blk.ln_2(h))
    h = m.encoder.ln(h)
    lg = m.heads(h[:, 0])
    return (lg, atts) if grab else lg

# ---- the three crops shown on the page
imgs = {}
for name, box in CROPS.items():
    im = Image.open(R + name + ".jpg").convert("RGB").crop(box).resize((224, 224), Image.BICUBIC)
    im.save("p224/%s.jpg" % name, quality=92)
    imgs[name] = Image.open("p224/%s.jpg" % name).convert("RGB")      # decode the saved file: the browser decodes the same bytes

out = {"model": "torchvision vit_b_16, IMAGENET1K_V1", "photos": {}, "crops": CROPS}
with torch.no_grad():
    x = torch.cat([to_t(imgs[n]) for n in CROPS])
    ref = m(x); lg, atts = forward(x, grab=True)
    print("max logit diff vs torchvision forward:", float((ref - lg).abs().max()))
    pr = torch.softmax(lg, 1)
    for i, n in enumerate(CROPS):
        t5 = pr[i].topk(5)
        out["photos"][n] = {"top5": [[CATS[int(j)], round(float(v), 3)] for v, j in zip(t5.values, t5.indices)]}
        print(n, out["photos"][n]["top5"])

# ---- mean attention distance of all 144 heads, over the 14 photos (standard Resize 256 + CenterCrop 224)
tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)])
gy, gx = np.divmod(np.arange(196), 14)
D = np.sqrt((gy[:, None] - gy[None, :]) ** 2 + (gx[:, None] - gx[None, :]) ** 2)       # distance between patch centres, in patches
Dm = torch.tensor(D, dtype=torch.float32)
cheb = np.maximum(np.abs(gy[:, None] - gy[None, :]), np.abs(gx[:, None] - gx[None, :]))  # window distance (Chebyshev)
dist = np.zeros((12, 12)); cls_share = np.zeros((12, 12)); win = np.zeros((12, 12, 14))     # window share by half-width 0..13
cnt = 0
all_x = torch.cat([to_t(tf(Image.open(R + n + ".jpg"))) for n in ALL])
with torch.no_grad():
    lg14, att14 = forward(all_x, grab=True)
p14 = torch.softmax(lg14, 1)
for l in range(12):
    A = att14[l]                                   # 14 x 12 x 197 x 197
    P = A[:, :, 1:, 1:]                            # patch to patch
    mass = P.sum(-1)                               # attention of a patch query that lands on patches (rest on the class token)
    dist[l] = ((P * Dm).sum(-1) / mass).mean((0, 2)).numpy()      # distance averaged over the patch-attention mass
    cls_share[l] = A[:, :, 1:, 0].mean((0, 2)).numpy()
    for w in range(14):
        msk = torch.tensor(cheb <= w, dtype=torch.float32)
        win[l, :, w] = ((P * msk).sum(-1).mean((0, 2))).numpy()   # share of all attention (class token included in the total of 1) within window w
out["dist_patches"] = r(dist, 2)
out["cls_share"] = r(cls_share, 3)
out["window_share"] = r(win.mean(1), 3)          # layer x half-width: mean over heads
out["window_share_heads"] = {str(w): r(win[:, :, w], 3) for w in (1, 2)}   # per head for half-width 1 and 2
print("mean distance by layer (heads min, mean, max):")
for l in range(12): print(l, dist[l].min().round(2), dist[l].mean().round(2), dist[l].max().round(2), "cls share", cls_share[l].mean().round(3), "win1", win[l, :, 1].mean().round(3), "win3", win[l, :, 3].mean().round(3))

# ---- heads to export as click-a-patch attention (chosen by looking at all 144 maps for one query patch)
sel = [[0, 6, "wide"], [2, 5, "local"], [3, 3, "body"], [5, 3, "body"], [7, 6, "scattered"], [11, 7, "body"]]
out["heads"] = [{"layer": l, "head": h, "kind": k, "dist": round(float(dist[l, h]), 2), "cls": round(float(cls_share[l, h]), 3)} for l, h, k in sel]
SHOWP = ["hammerhead-2", "bamboo-1"]
os.makedirs("attn", exist_ok=True)
for f in os.listdir("attn"): os.remove("attn/" + f)
rowmax = {}
for i, n in enumerate(CROPS):
    if n not in SHOWP: continue
    for l, h, k in sel:
        A = atts[l][i, h].numpy()                      # 197 x 197
        rm = A.max(1)                                   # per query row
        q = np.round(15 * np.sqrt(A / rm[:, None])).astype(np.uint8)     # 16 levels of the square root; the browser undoes both
        im = Image.fromarray(q, "P"); im.putpalette([v for g in range(16) for v in (g * 17, g * 17, g * 17)] + [0] * (3 * 240))
        im.save("attn/%s_L%dH%d.png" % (n, l, h), optimize=True, bits=4)
        rowmax["%s_L%dH%d" % (n, l, h)] = r(rm, 3)
out["rowmax"] = rowmax
# class-token attention by layer (mean over heads), 12 x 196, per displayed photo
out["cls_by_layer"] = {n: r(np.stack([atts[l][i, :, 0, 1:].mean(0).numpy() for l in range(12)]), 3) for i, n in enumerate(CROPS)}
out["cls_to_cls"] = {n: [round(float(atts[l][i, :, 0, 0].mean()), 3) for l in range(12)] for i, n in enumerate(CROPS)}

# ---- patch embedding: eight rows of E, bias, and the exact outputs for every patch of the three crops
cw = m.conv_proj.weight.detach()                   # 768 x 3 x 16 x 16
cb = m.conv_proj.bias.detach()
E = cw.permute(0, 2, 3, 1).reshape(768, 768)       # rows follow the order (row i, column j, colour c) of patch.reshape(-1)
norms = E.norm(dim=1)
# choose filters: spread over the norm ranking and over colour-ness
col = (cw.mean((2, 3)).abs().sum(1) / (cw.abs().mean((2, 3)).sum(1) + 1e-9))
print("norm range", float(norms.min()), float(norms.max()))
FILT = [143, 734, 63, 97, 658, 402, 238, 51]
out["filters"] = {"index": FILT, "bias": [round(float(cb[i]), 4) for i in FILT], "w": [r(E[i].numpy(), 4) for i in FILT]}
out["mean"] = MEAN.flatten().tolist(); out["std"] = STD.flatten().tolist()
with torch.no_grad():
    tok = m._process_input(x)       # 3 x 196 x 768 (output of conv_proj, flattened): the patch tokens before position embeddings
    ref_vals = {n: r(tok[i][:, FILT].numpy(), 3) for i, n in enumerate(CROPS)}
json.dump(ref_vals, open("check_patch_tokens.json", "w"), separators=(",", ":"))   # exact tokens, to test the page JS against (not shipped)
out["cls_token_head"] = [round(float(v), 3) for v in m.class_token.detach()[0, 0, :16]]
out["params"] = {"vit_b_16": sum(p.numel() for p in m.parameters()), "patch_embed": sum(p.numel() for p in m.conv_proj.parameters()),
   "pos": int(m.encoder.pos_embedding.numel()), "cls": int(m.class_token.numel()),
   "block": sum(p.numel() for p in m.encoder.layers[0].parameters()), "blocks12": sum(p.numel() for p in m.encoder.layers.parameters()),
   "head": sum(p.numel() for p in m.heads.parameters()), "ln": sum(p.numel() for p in m.encoder.ln.parameters())}
for nm, fn in (("resnet18", torchvision.models.resnet18), ("resnet50", torchvision.models.resnet50), ("vit_b_32", torchvision.models.vit_b_32), ("vit_l_16", torchvision.models.vit_l_16)):
    out["params"][nm] = sum(p.numel() for p in fn().parameters())
tv = torchvision.models
for nm, wt in (("resnet18", tv.ResNet18_Weights.IMAGENET1K_V1), ("resnet50", tv.ResNet50_Weights.IMAGENET1K_V1), ("vit_b_32", tv.ViT_B_32_Weights.IMAGENET1K_V1), ("vit_b_16", tv.ViT_B_16_Weights.IMAGENET1K_V1), ("vit_l_16", tv.ViT_L_16_Weights.IMAGENET1K_V1)):
    out["params"][nm + "_gflops"] = wt.meta["_ops"]; out["params"][nm + "_top1"] = wt.meta["_metrics"]["ImageNet-1K"]["acc@1"]
print(out["params"])

# ---- position embedding similarity (cosine, 196 x 196) as a grey PNG, value = (cos + 1) / 2
pe = m.encoder.pos_embedding[0, 1:].detach()
pn = pe / pe.norm(dim=1, keepdim=True); S = (pn @ pn.T).numpy()
print("pos cos: min", S.min(), "max off-diagonal", (S - 2 * np.eye(196)).max())
Image.fromarray(np.round((S + 1) / 2 * 255).astype(np.uint8), "L").save("possim.png", optimize=True)
out["possim_range"] = [float(S.min()), 1.0]
nb = {}
# cosine of neighbours (1 patch apart in a row), 5 and 10 patches apart
for dd in (1, 3, 6, 10):
    vals = [S[a, b] for a in range(196) for b in range(196) if a < b and np.hypot(gy[a] - gy[b], gx[a] - gx[b]) == dd]
    nb[str(dd)] = round(float(np.mean(vals)), 3) if vals else None
out["pos_cos_by_distance"] = nb; print("pos cos by distance", nb)
out["pos_row_col"] = {"same_row_other_col": round(float(np.mean([S[a, b] for a in range(196) for b in range(196) if a != b and gy[a] == gy[b] and abs(gx[a] - gx[b]) == 3])), 3),
                       "not_same_row_dist3": round(float(np.mean([S[a, b] for a in range(196) for b in range(196) if gy[a] != gy[b] and gx[a] != gx[b] and np.hypot(gy[a] - gy[b], gx[a] - gx[b]) < 3.3 and np.hypot(gy[a] - gy[b], gx[a] - gx[b]) > 2.5])), 3)}

json.dump(out, open("vit.json", "w"), separators=(",", ":"))
print("saved", os.path.getsize("vit.json"))
