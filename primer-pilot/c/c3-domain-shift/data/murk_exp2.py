import sys; sys.path.insert(0, '.')
import json, copy, numpy as np, torch
from murk import degrade, K_RGB, VEIL
import murk_exp as E
from murk_exp import BNNet, accuracy, reestimate, Xtr, Xte, ytr, yte
torch.manual_seed(3)
ck = torch.load("out/murk_models.pt"); m0 = BNNet(); m0.load_state_dict(ck["clean"]); m1 = BNNet(); m1.load_state_dict(ck["aug"]); m0.eval(); m1.eval()
depths = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
# water B: a second, brownish water type (made up): red lasts longer, blue is absorbed fast, tan veil
K_B = torch.tensor([0.30, 0.50, 0.85]); VEIL_B = torch.tensor([0.35, 0.30, 0.15])
def deg(x, z, water):
    return degrade(x, z) if water == "A" else degrade(x, z, k=K_B, veil=VEIL_B)
res = {"depths": depths, "A": {}, "B": {}}
def pool_from(idx, z, water): return deg(Xtr[idx], z, water)
rs = np.random.default_rng(0)
pools = [torch.from_numpy(rs.choice(60000, 256, replace=False)) for _ in range(5)]
for water in ("A", "B"):
    R = {k: [] for k in ("clean", "clean_adapt", "clean_adapt_sd", "aug", "aug_adapt", "aug_adapt_sd")}
    for z in depths:
        Xz = torch.cat([deg(Xte[i:i + 2000], z, water) for i in range(0, 10000, 2000)])
        R["clean"].append(accuracy(m0, Xz, yte)); R["aug"].append(accuracy(m1, Xz, yte))
        a0 = [accuracy(reestimate(m0, pool_from(p, z, water)), Xz, yte) for p in pools]
        a1 = [accuracy(reestimate(m1, pool_from(p, z, water)), Xz, yte) for p in pools]
        R["clean_adapt"].append(float(np.mean(a0))); R["clean_adapt_sd"].append(float(np.std(a0)))
        R["aug_adapt"].append(float(np.mean(a1))); R["aug_adapt_sd"].append(float(np.std(a1)))
        print(water, z, {k: round(v[-1], 4) for k, v in R.items()}, flush=True)
    res[water] = R
# pool size sweep at 10 m, water A, clean model; and pools of one class
z = 10; Xz = torch.cat([deg(Xte[i:i + 2000], z, "A") for i in range(0, 10000, 2000)])
sizes = [2, 4, 8, 16, 32, 64, 128, 256, 1024]
sw = []
for n in sizes:
    acc = []
    for s in range(8):
        idx = torch.from_numpy(np.random.default_rng(100 + s).choice(60000, n, replace=False))
        acc.append(accuracy(reestimate(m0, pool_from(idx, z, "A")), Xz, yte))
    sw.append([n, float(np.mean(acc)), float(np.std(acc))]); print("pool", n, sw[-1], flush=True)
res["pool_sweep"] = {"depth": 10, "rows": sw}
# mixes
mixes = []
pool = pool_from(pools[0], z, "A")
for al in (0, 0.25, 0.5, 0.75, 1.0):
    mixes.append([al, accuracy(reestimate(m0, pool, alpha=al), Xz, yte)]); print("alpha", mixes[-1])
res["mix"] = mixes
# unbalanced pools
ytr_np = ytr.numpy()
ub = {}
for name, classes in (("only_ones", [1]), ("zeros_to_four", [0, 1, 2, 3, 4]), ("all_ten", list(range(10)))):
    idx = np.where(np.isin(ytr_np, classes))[0]
    pick = torch.from_numpy(np.random.default_rng(7).choice(idx, 256, replace=False))
    ub[name] = accuracy(reestimate(m0, pool_from(pick, z, "A")), Xz, yte); print(name, ub[name])
res["unbalanced"] = ub
# clean pool at 0 m: adaptation changes nothing?
X0 = Xte
res["no_shift_check"] = {"stored": accuracy(m0, X0, yte), "adapted_on_clean_pool": accuracy(reestimate(m0, Xtr[pools[0]]), X0, yte)}
print(res["no_shift_check"])
json.dump(res, open("out/murk_res2.json", "w"))
