import sys; sys.path.insert(0, '.')
import json, copy, math, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from murk import degrade, K_RGB
torch.manual_seed(11); np.random.seed(11)
tr = torchvision.datasets.MNIST("mnist", train=True); te = torchvision.datasets.MNIST("mnist", train=False)
Xtr = tr.data.float().div(255).unsqueeze(1).repeat(1, 3, 1, 1); ytr = tr.targets
Xte = te.data.float().div(255).unsqueeze(1).repeat(1, 3, 1, 1); yte = te.targets

class BNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1); self.b1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1); self.b2 = nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.b1(self.c1(x))), 2)
        x = F.max_pool2d(F.relu(self.b2(self.c2(x))), 2)
        return self.fc(x.flatten(1))

def rand_water(n, zmax):
    z = torch.rand(n) * zmax
    k = K_RGB.view(1, 3) * torch.exp(0.2 * torch.randn(n, 3))
    beta = 0.06 * torch.empty(n).uniform_(0.6, 1.4); expo = torch.empty(n).uniform_(0.3, 0.9)
    return z, k, beta, expo

def train(aug, epochs=4):
    m = BNNet(); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(60000)
        for i in range(0, 60000, 128):
            idx = perm[i:i + 128]; xb = Xtr[idx]; yb = ytr[idx]
            if aug:
                z, k, beta, expo = rand_water(len(idx), 15.0)
                half = len(idx) // 2     # half of every batch stays clean
                xb = torch.cat([xb[:half], degrade(xb[half:], z[half:], k=k[half:], beta=beta[half:], exposure=expo[half:])])
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
    return m

def accuracy(m, X, y, bs=1000):
    m.eval(); c = 0
    with torch.no_grad():
        for i in range(0, len(y), bs): c += (m(X[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    return c / len(y)

def reestimate(m, pool, alpha=1.0, bs=128):
    """AdaBN: replace the stored mean and variance of every batch-norm layer by those of the new images (no labels, no weight update)."""
    a = copy.deepcopy(m)
    src = [(b.running_mean.clone(), b.running_var.clone()) for b in (m.b1, m.b2)]
    for b in (a.b1, a.b2): b.reset_running_stats(); b.momentum = None
    a.train()
    with torch.no_grad():
        for i in range(0, len(pool), bs): a(pool[i:i + bs])
    for b, (mu, var) in zip((a.b1, a.b2), src):
        b.running_mean.mul_(alpha).add_((1 - alpha) * mu); b.running_var.mul_(alpha).add_((1 - alpha) * var)
    a.eval(); return a

def entropy_steps(m, pool, steps=1, lr=1e-3, bs=64):
    """Tent (Wang et al. 2021): normalise with batch statistics and update only the batch-norm scale and shift to lower the entropy of the predictions."""
    a = copy.deepcopy(m); a.train()
    params = [p for b in (a.b1, a.b2) for p in (b.weight, b.bias)]
    for p_ in a.parameters(): p_.requires_grad_(False)
    for p_ in params: p_.requires_grad_(True)
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9)
    for s in range(steps):
        for i in range(0, len(pool) - bs + 1, bs):
            out = a(pool[i:i + bs]); p = out.softmax(1)
            ent = -(p * torch.log(p + 1e-8)).sum(1).mean()
            opt.zero_grad(); ent.backward(); opt.step()
    # after the updates: re-estimate the statistics on the same pool, then evaluate in eval mode
    for p_ in a.parameters(): p_.requires_grad_(False)
    for b in (a.b1, a.b2): b.reset_running_stats(); b.momentum = None
    with torch.no_grad():
        for i in range(0, len(pool), 128): a(pool[i:i + 128])
    a.eval(); return a

if __name__ == "__main__":
    depths = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
    m0 = train(False); m1 = train(True)
    res = {"depths": depths, "clean": [], "clean_adapt": [], "clean_tent": [], "aug": [], "aug_adapt": [], "adapt_clean_pool_at_0": None}
    g = torch.Generator().manual_seed(5)
    pool_idx = torch.randperm(60000, generator=g)[:256]
    for z in depths:
        Xz = torch.cat([degrade(Xte[i:i + 2000], z) for i in range(0, 10000, 2000)])
        pool = degrade(Xtr[pool_idx], z)
        r0 = accuracy(m0, Xz, yte); r0a = accuracy(reestimate(m0, pool), Xz, yte)
        r0t = accuracy(entropy_steps(m0, pool), Xz, yte)
        r1 = accuracy(m1, Xz, yte); r1a = accuracy(reestimate(m1, pool), Xz, yte)
        res["clean"].append(r0); res["clean_adapt"].append(r0a); res["clean_tent"].append(r0t); res["aug"].append(r1); res["aug_adapt"].append(r1a)
        print(z, round(r0, 4), round(r0a, 4), round(r0t, 4), round(r1, 4), round(r1a, 4), flush=True)
    json.dump(res, open("out/murk_res.json", "w"))
    torch.save({"clean": m0.state_dict(), "aug": m1.state_dict()}, "out/murk_models.pt")
