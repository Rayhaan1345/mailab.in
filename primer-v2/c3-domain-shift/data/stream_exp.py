import sys; sys.path.insert(0, '.')
import json, copy, numpy as np, torch, torch.nn.functional as F
from murk import degrade
from murk_exp import BNNet, Xte, yte
torch.set_num_threads(8)
ck = torch.load("out/murk_models.pt"); m0 = BNNet(); m0.load_state_dict(ck["clean"]); m0.eval()
NCH, CH = 40, 200
g = torch.Generator().manual_seed(21); order = torch.randperm(10000, generator=g)[:NCH * CH]
zs = [12.0 * t / (NCH - 1) for t in range(NCH)]
def chunk(t): idx = order[t * CH:(t + 1) * CH]; return degrade(Xte[idx], zs[t]), yte[idx]
def acc(m, x, y):
    m.eval()
    with torch.no_grad(): return float((m(x).argmax(1) == y).float().mean())
def run(mode, lr=1e-3, momentum=0.5):
    m = copy.deepcopy(m0)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d): mod.momentum = momentum
    params = [p for mod in m.modules() if isinstance(mod, torch.nn.BatchNorm2d) for p in (mod.weight, mod.bias)]
    for p in m.parameters(): p.requires_grad_(False)
    for p in params: p.requires_grad_(True)
    opt = torch.optim.Adam(params, lr=lr) if mode == "tent" else None
    out = []
    for t in range(NCH):
        x, y = chunk(t)
        out.append(acc(m, x, y))                       # predict with what was learned from earlier chunks
        if mode == "frozen": continue
        m.train()                                       # read the chunk: batch statistics, running statistics move toward them
        if mode == "stats":
            with torch.no_grad(): m(x)
        else:
            o = m(x); p = o.softmax(1); ent = -(p * torch.log(p + 1e-8)).sum(1).mean()
            opt.zero_grad(); ent.backward(); opt.step()
    return out
res = {"zs": zs, "chunk": CH, "n_chunks": NCH}
res["frozen"] = run("frozen"); res["stats"] = run("stats"); res["tent_1e-3"] = run("tent", 1e-3); res["tent_1e-2"] = run("tent", 1e-2)
for k in ("frozen", "stats", "tent_1e-3", "tent_1e-2"): print(k, "mean accuracy over the stream", round(float(np.mean(res[k])), 4), "last 10 chunks", round(float(np.mean(res[k][-10:])), 4), "first 5", round(float(np.mean(res[k][:5])), 4))
json.dump(res, open("out/stream_res.json", "w"))
