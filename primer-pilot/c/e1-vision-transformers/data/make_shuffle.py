"""E1 data: what happens to a classifier when the patches of a photo are moved to other places.
A fixed order pi of the 196 patches is stored. 'k patches moved' means the first k patches of pi are passed along in a circle
(patch pi[i] gets the pixels of patch pi[(i+1) mod k]). Position embeddings stay where they are.
Part 1: top-3 of ViT-B/16 for the three crops shown on the page, with that single order, for each k.
Part 2: over the 14 photos (Resize 256, CenterCrop 224) and 16 random orders per photo, how often the top-1 class of the unmoved photo is kept,
        for ViT-B/16 and for ResNet-18 (both torchvision, ImageNet-1k weights)."""
import json, numpy as np, torch, torchvision
from PIL import Image
torch.set_num_threads(8)
R = "../../data/sharks/photos/"
vw = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1; vit = torchvision.models.vit_b_16(weights=vw).eval()
rw = torchvision.models.ResNet18_Weights.IMAGENET1K_V1; rn = torchvision.models.resnet18(weights=rw).eval()
CATS = vw.meta["categories"]
mean_v = torch.tensor(vw.transforms().mean).view(1, 3, 1, 1); std_v = torch.tensor(vw.transforms().std).view(1, 3, 1, 1)
ALL = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"]
KS = [0, 4, 8, 16, 32, 48, 64, 96, 128, 196]
def move(a, order, k):        # a: H x W x 3 uint8 (224), patches of 16
    if k < 2: return a.copy()
    out = a.copy(); idx = order[:k]
    P = lambda i: (slice((i // 14) * 16, (i // 14) * 16 + 16), slice((i % 14) * 16, (i % 14) * 16 + 16))
    for t in range(k):
        out[P(idx[t])] = a[P(idx[(t + 1) % k])]
    return out
def tens(a, mean, std): return ((torch.tensor(a).permute(2, 0, 1).float() / 255).unsqueeze(0) - mean) / std
res = {"ks": KS}
order = np.random.RandomState(4127).permutation(196)
res["order"] = [int(i) for i in order]
res["shown"] = {}
for n in ["hammerhead-2", "bamboo-1", "spadenose-2"]:
    a = np.asarray(Image.open("p224/%s.jpg" % n).convert("RGB"))
    rows = []
    for k in KS:
        with torch.no_grad(): p = torch.softmax(vit(tens(move(a, order, k), mean_v, std_v)), 1)[0]
        t = p.topk(3); rows.append([[CATS[int(j)], round(float(v), 3)] for v, j in zip(t.values, t.indices)])
    res["shown"][n] = rows
    print(n, [r[0] for r in rows])
tf = torchvision.transforms.Compose([torchvision.transforms.Resize(256), torchvision.transforms.CenterCrop(224)])
mean_r = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); std_r = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
arrs = [np.asarray(tf(Image.open(R + n + ".jpg").convert("RGB"))) for n in ALL]
with torch.no_grad():
    top_v = [int(vit(tens(a, mean_v, std_v)).argmax()) for a in arrs]; top_r = [int(rn(tens(a, mean_r, std_r)).argmax()) for a in arrs]
keep_v = np.zeros(len(KS)); keep_r = np.zeros(len(KS)); pv = np.zeros(len(KS)); pr = np.zeros(len(KS)); N = 0
NS = 16
for ia, a in enumerate(arrs):
    for s in range(NS):
        o = np.random.RandomState(1000 * ia + s).permutation(196)
        xs_v = torch.cat([tens(move(a, o, k), mean_v, std_v) for k in KS]); xs_r = torch.cat([tens(move(a, o, k), mean_r, std_r) for k in KS])
        with torch.no_grad(): lv = torch.softmax(vit(xs_v), 1); lr = torch.softmax(rn(xs_r), 1)
        keep_v += (lv.argmax(1).numpy() == top_v[ia]); keep_r += (lr.argmax(1).numpy() == top_r[ia])
        pv += lv[:, top_v[ia]].numpy(); pr += lr[:, top_r[ia]].numpy(); N += 1
    print(ALL[ia], flush=True)
res["keep_vit"] = np.round(keep_v / N, 3).tolist(); res["keep_resnet"] = np.round(keep_r / N, 3).tolist()
res["p_vit"] = np.round(pv / N, 3).tolist(); res["p_resnet"] = np.round(pr / N, 3).tolist()
res["n_runs"] = N; res["n_photos"] = len(ALL); res["n_orders"] = NS
res["unmoved_top1"] = {"vit": [CATS[i] for i in top_v], "resnet": [rw.meta["categories"][i] for i in top_r]}
print("ks", KS); print("vit keep", res["keep_vit"]); print("resnet keep", res["keep_resnet"]); print("p vit", res["p_vit"]); print("p resnet", res["p_resnet"])
json.dump(res, open("shuffle.json", "w"), separators=(",", ":"))
