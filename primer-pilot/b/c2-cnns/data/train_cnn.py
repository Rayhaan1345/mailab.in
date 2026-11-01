"""Train the small CNN for chapter C2 on MNIST and export weights to JSON."""
import json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision
torch.manual_seed(7); np.random.seed(7)
tr = torchvision.datasets.MNIST("mnist", train=True); te = torchvision.datasets.MNIST("mnist", train=False)
Xtr = tr.data.float().div(255).unsqueeze(1); ytr = tr.targets
Xte = te.data.float().div(255).unsqueeze(1); yte = te.targets

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)      # 28 -> 26
        self.conv2 = nn.Conv2d(6, 12, 3)     # 13 -> 11
        self.fc = nn.Linear(12 * 5 * 5, 10)  # 5 x 5 after the second pool
    def maps(self, x):
        a1 = F.relu(self.conv1(x)); p1 = F.max_pool2d(a1, 2)
        a2 = F.relu(self.conv2(p1)); p2 = F.max_pool2d(a2, 2)
        return a1, p1, a2, p2
    def forward(self, x):
        return self.fc(self.maps(x)[3].flatten(1))

def affine(x, max_shift=2.0, max_rot=10.0, scale=(0.9, 1.1)):
    n = x.shape[0]
    ang = (torch.rand(n) * 2 - 1) * max_rot * math.pi / 180
    sc = torch.empty(n).uniform_(*scale)
    tx = (torch.rand(n) * 2 - 1) * max_shift * 2 / 28; ty = (torch.rand(n) * 2 - 1) * max_shift * 2 / 28
    th = torch.zeros(n, 2, 3)
    th[:, 0, 0] = torch.cos(ang) / sc; th[:, 0, 1] = -torch.sin(ang) / sc; th[:, 0, 2] = tx
    th[:, 1, 0] = torch.sin(ang) / sc; th[:, 1, 1] = torch.cos(ang) / sc; th[:, 1, 2] = ty
    g = F.affine_grid(th, x.shape, align_corners=False)
    return F.grid_sample(x, g, padding_mode="zeros", align_corners=False)

def acc(model, X, y):
    model.eval()
    with torch.no_grad(): return float((model(X).argmax(1) == y).float().mean())

def train(use_aug, epochs=10):
    m = Net(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=6e-3, total_steps=epochs * (60000 // 128 + 1))
    for ep in range(epochs):
        m.train(); perm = torch.randperm(60000)
        for i in range(0, 60000, 128):
            idx = perm[i:i + 128]; xb = Xtr[idx]; yb = ytr[idx]
            if use_aug: xb = affine(xb)
            loss = F.cross_entropy(m(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        print("aug" if use_aug else "plain", ep, round(loss.item(), 4), round(acc(m, Xte, yte), 4), flush=True)
    return m

mA = train(True); mP = train(False)
def shift(X, dx, dy): return torch.roll(X, shifts=(dy, dx), dims=(2, 3))
def rotate(X, deg):
    n = X.shape[0]; a = deg * math.pi / 180
    th = torch.tensor([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0]], dtype=torch.float32).expand(n, 2, 3)
    return F.grid_sample(X, F.affine_grid(th, X.shape, align_corners=False), padding_mode="zeros", align_corners=False)
def curves(m):
    sh = [round(acc(m, shift(Xte, d, 0), yte), 4) for d in range(0, 9)]
    ro = [round(acc(m, rotate(Xte, d), yte), 4) for d in range(0, 91, 10)]
    return sh, ro
out = {}
for name, m in (("aug", mA), ("plain", mP)):
    sh, ro = curves(m); out[name] = {"test_acc": round(acc(m, Xte, yte), 4), "shift_px": list(range(9)), "shift_acc": sh, "rot_deg": list(range(0, 91, 10)), "rot_acc": ro}
    print(name, out[name])
torch.save({"aug": mA.state_dict(), "plain": mP.state_dict()}, "out/c2_models.pt")
json.dump(out, open("out/c2_robustness.json", "w"))
