import torch, torch.nn as nn, torch.nn.functional as F, torchvision, numpy as np
torch.manual_seed(0)
tr = torchvision.datasets.MNIST("mnist", train=True); te = torchvision.datasets.MNIST("mnist", train=False)
Xtr = tr.data.float().div(255).flatten(1); ytr = tr.targets; Xte = te.data.float().div(255).flatten(1); yte = te.targets
perm = torch.from_numpy(np.random.default_rng(42).permutation(784))
def run(P):
    torch.manual_seed(1)
    m = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)); opt = torch.optim.Adam(m.parameters(), 1e-3)
    A = Xtr[:, P]; B = Xte[:, P]
    for ep in range(5):
        pm = torch.randperm(60000)
        for i in range(0, 60000, 128):
            idx = pm[i:i + 128]; loss = F.cross_entropy(m(A[idx]), ytr[idx]); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad(): return float((m(B).argmax(1) == yte).float().mean())
print("normal", run(torch.arange(784)), "shuffled", run(perm))
n = sum(p.numel() for p in nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).parameters()); print("params", n)
