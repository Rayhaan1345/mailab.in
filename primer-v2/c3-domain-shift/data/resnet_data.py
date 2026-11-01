"""Shark-class confidence of a pretrained ResNet-18 against depth, with stored and re-estimated batch-norm statistics; per-channel statistics of two BN layers."""
import sys, json, copy, importlib.util, numpy as np, torch, torchvision
from PIL import Image
import torchvision.transforms as T
R = "../../data/sharks/"   # relative to this folder
spec = importlib.util.spec_from_file_location("depth", R + "depth.py"); depth = importlib.util.module_from_spec(spec); spec.loader.exec_module(depth)
torch.set_num_threads(8)
base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1); base.eval()
cats = torchvision.models.ResNet18_Weights.IMAGENET1K_V1.meta["categories"]
norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
aug = T.Compose([T.RandomResizedCrop(224, scale=(0.15, 1.0)), T.RandomHorizontalFlip(), T.ToTensor(), norm])
ALL = ["spadenose-1", "spadenose-2", "milk-1", "milk-2", "bamboo-1", "bamboo-2", "hammerhead-1", "hammerhead-2", "ganges-1", "ganges-2", "ray-1", "dolphin-1", "reeffish-1", "turtle-1"]
SHOW = ["hammerhead-1", "hammerhead-2", "milk-1", "milk-2", "spadenose-1", "dolphin-1", "reeffish-1", "turtle-1"]
imgs = {n: Image.open(R + "photos/%s.jpg" % n).convert("RGB") for n in ALL}
depths = [i * 0.5 for i in range(0, 61)]       # 0 to 30 m
ALPHAS = [0.0, 0.25, 1.0]
NCROP = 16
bn_names = [n for n, m in base.named_modules() if isinstance(m, torch.nn.BatchNorm2d)]
KEEP = ["bn1", "layer4.1.bn2"]
stored = {n: {"mean": dict(base.named_modules())[n].running_mean.numpy().round(4).tolist(), "var": dict(base.named_modules())[n].running_var.numpy().round(4).tolist()} for n in KEEP}

def adapt_models(pool):
    m = copy.deepcopy(base)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d): mod.reset_running_stats(); mod.momentum = None
    m.train()
    with torch.no_grad():
        for i in range(0, len(pool), 64): m(pool[i:i + 64])
    out = {}
    for al in ALPHAS:
        mm = copy.deepcopy(m)
        for a, b in zip(mm.modules(), base.modules()):
            if isinstance(a, torch.nn.BatchNorm2d):
                a.running_mean.mul_(al).add_((1 - al) * b.running_mean); a.running_var.mul_(al).add_((1 - al) * b.running_var)
        mm.eval(); out[al] = mm
    return out, m

res = {"depths": depths, "photos": SHOW, "alphas": ALPHAS, "classes": {}, "curves": {n: {str(a): {"shark": [], "top": [], "ptop": []} for a in ALPHAS} for n in SHOW}, "top3_stored": {n: [] for n in SHOW}}
stat_depths = [0, 2.5, 5, 10, 15, 20, 30]
res["bnstats"] = {"depths": stat_depths, "layers": {n: {"stored_mean": stored[n]["mean"], "stored_var": stored[n]["var"], "mean": [], "var": []} for n in KEEP}}
names_used = set()
for z in depths:
    pool = []
    for n in ALL:
        d = depth.degrade(imgs[n], z)
        for k in range(NCROP):
            torch.manual_seed(100 + k); pool.append(aug(d))
    pool = torch.stack(pool)
    models, mfull = adapt_models(pool)
    xs = torch.stack([tf(depth.degrade(imgs[n], z)) for n in SHOW])
    for al in ALPHAS:
        with torch.no_grad(): p = torch.softmax(models[al](xs), 1)
        for i, n in enumerate(SHOW):
            c = res["curves"][n][str(al)]
            c["shark"].append(round(float(p[i, 2:5].sum()), 4)); top = int(p[i].argmax()); c["top"].append(top); c["ptop"].append(round(float(p[i, top]), 3)); names_used.add(top)
            if al == 0.0:
                t3 = p[i].topk(3); res["top3_stored"][n].append([[int(a), round(float(b), 3)] for b, a in zip(t3.values, t3.indices)]); [names_used.add(int(a)) for a in t3.indices]
    if z in stat_depths:
        # per-channel mean and variance of the BN inputs, measured on the pool with the stored statistics in the layers before
        hooks = {}; store = {}
        def mk(name):
            def f(mod, inp): 
                x = inp[0]; store[name] = (x.mean((0, 2, 3)).numpy(), x.var((0, 2, 3), unbiased=False).numpy())
            return f
        mm = copy.deepcopy(base); mm.eval()
        mods = dict(mm.named_modules())
        hs = [mods[n].register_forward_pre_hook(mk(n)) for n in KEEP]
        with torch.no_grad():
            mm(pool)   # all 224 crops of the 14 photos
        [h.remove() for h in hs]
        for n in KEEP:
            res["bnstats"]["layers"][n]["mean"].append(np.round(store[n][0], 3).tolist()); res["bnstats"]["layers"][n]["var"].append(np.round(store[n][1], 3).tolist())
    print("depth", z, [round(res["curves"][n]["0.0"]["shark"][-1], 2) for n in SHOW], [round(res["curves"][n]["1.0"]["shark"][-1], 2) for n in SHOW], flush=True)
res["classes"] = {str(i): cats[i] for i in sorted(names_used)}
json.dump(res, open("out/resnet_data.json", "w"), separators=(",", ":"))
import os; print(os.path.getsize("out/resnet_data.json"))
