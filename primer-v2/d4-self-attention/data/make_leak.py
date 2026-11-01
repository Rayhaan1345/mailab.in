"""Offline step for D4: what goes wrong without the causal mask. Two copies of one tiny attention model (word embeddings, learned positions, one
attention head, one output layer) are trained on 400 MADE-UP reports (the same templates and seed as D3's make_tiny.py; 100 more are held out).
One copy is trained with the causal mask, one without. Writes leak.json: loss every 20 steps on the training reports (with the mask the
model was trained with) and on the held-out reports with the mask on, which is the condition at generation time.
Run with: venv-d/bin/python -I make_leak.py"""
import json, random, math, torch, torch.nn as nn, torch.nn.functional as F
rng = random.Random(4127)
def pick(options):
    items, w = zip(*options); return rng.choices(items, weights=w)[0]
def sp(): return pick([("spadenose", 5), ("milk", 3), ("bamboo", 2)])
def place(): return pick([("sassoon dock", 4), ("versova", 2), ("juhu", 2), ("uran", 2)])
def num(): return pick([("ten", 2), ("twelve", 2), ("fifteen", 4), ("twenty", 2)])
def cm(): return pick([(str(x), 1) for x in [25, 30, 40, 44, 50, 60, 80]])
def S1(): return f"{num()} {sp()} sharks came in at {place()} this morning ."
def S2(): return f"the biggest was about {cm()} cm and the smallest about {cm()} cm ."
def S3(): return f"prices were {pick([('low', 3), ('high', 1)])} because the auction started {pick([('late', 3), ('early', 1)])} ."
def S4(): return f"we found a {pick([('young', 4), ('small', 2), ('big', 1)])} {sp()} shark in a rock pool at {place()} at low tide ."
def S5(): return f"length {cm()} cm , dark bands across its back ."
def S6(): return f"we {pick([('photographed', 3), ('measured', 2), ('weighed', 1)])} it and {pick([('put it back', 4), ('let it go', 3)])} ."
def S7(): return f"our gillnet caught one hammerhead near {place()} , about {cm()} cm long ."
def S8(): return f"it looked young , so we {pick([('let it go', 4), ('put it back in the water', 2)])} ."
def S9(): return f"the water at {place()} was {pick([('twenty eight', 4), ('twenty nine', 2), ('twenty seven', 2)])} degrees at {pick([('six', 3), ('ten', 2), ('four', 1)])} metres ."
def S10(): return f"we could see about {pick([('four', 3), ('five', 2), ('three', 1)])} metres ahead ."
def S11(): return f"{pick([('two', 3), ('three', 1)])} milk sharks swam past the rocks ."
kinds = [([S1, S2, S3], 4), ([S4, S5, S6], 4), ([S7, S8], 3), ([S9, S10, S11], 3), ([S1, S3], 1), ([S4, S6], 1)]
docs = []
for _ in range(500):
    seq = pick(kinds); docs.append(" ".join(f() for f in seq).split())
vocab = ["<pad>", "<s>"] + sorted({w for d in docs for w in d}); ix = {w: i for i, w in enumerate(vocab)}
V = len(vocab); T = max(len(d) for d in docs) + 1
def enc(ds):
    X = torch.zeros(len(ds), T, dtype=torch.long); M = torch.zeros(len(ds), T)
    for i, d in enumerate(ds):
        ids = [ix["<s>"]] + [ix[w] for w in d]; X[i, :len(ids)] = torch.tensor(ids); M[i, :len(ids)] = 1
    return X, M
train, test = docs[:400], docs[400:]
Xtr, Mtr = enc(train); Xte, Mte = enc(test)
class Tiny(nn.Module):
    def __init__(s, d=48):
        super().__init__(); s.E = nn.Embedding(V, d); s.P = nn.Embedding(T, d); s.q = nn.Linear(d, d, bias=False); s.k = nn.Linear(d, d, bias=False); s.v = nn.Linear(d, d, bias=False); s.o = nn.Linear(d, V)
        s.d = d
    def forward(s, x, causal, shift=False):
        n = x.shape[1]; h = s.E(x) + s.P(torch.arange(n))[None]
        sc = s.q(h) @ s.k(h).transpose(1, 2) / math.sqrt(s.d)
        # padding keys are never attended
        pad = (x == 0)[:, None, :]
        sc = sc.masked_fill(pad, float("-inf"))
        if causal: sc = sc.masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1)[None], float("-inf"))
        a = sc.softmax(-1) @ s.v(h)
        return s.o(h + a)
def loss_of(model, X, M, causal):
    lg = model(X[:, :-1], causal)  # predict token t+1 from positions <= t (or any position if not causal)
    tgt = X[:, 1:]; msk = M[:, 1:]
    l = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    return (l * msk).sum() / msk.sum()
res = {}; models = {}
for name, causal in [("mask", True), ("nomask", False)]:
    torch.manual_seed(4127); m = Tiny(); opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    curve = []
    for step in range(401):
        if step % 20 == 0:
            with torch.no_grad(): curve.append([step, round(float(loss_of(m, Xtr, Mtr, causal)), 3), round(float(loss_of(m, Xte, Mte, True)), 3)])
        opt.zero_grad(); l = loss_of(m, Xtr, Mtr, causal); l.backward(); opt.step()
    res[name] = curve; models[name] = m
    with torch.no_grad(): print(name, "train(own mask)", float(loss_of(m, Xtr, Mtr, causal)), "test(mask on)", float(loss_of(m, Xte, Mte, True)), "test(own)", float(loss_of(m, Xte, Mte, causal)))
print("uniform", math.log(V), "V", V, "T", T)
import os
# Where does the unmasked model look? Weight that position i puts on position i+1 (the answer), over the training reports, for all positions and
# for the positions where the masked model finds the next word hard to guess (its loss above 1 nat).
with torch.no_grad():
    x = Xtr[:, :-1]; n = x.shape[1]; tgt = Xtr[:, 1:]; msk = Mtr[:, 1:]
    lg = models["mask"](x, True); l = F.cross_entropy(lg.reshape(-1, V), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
    m = models["nomask"]; h = m.E(x) + m.P(torch.arange(n))[None]
    sc = m.q(h) @ m.k(h).transpose(1, 2) / math.sqrt(m.d); sc = sc.masked_fill((x == 0)[:, None, :], float("-inf")); a = sc.softmax(-1)
    nxt = torch.diagonal(a, offset=1, dim1=1, dim2=2)
    L = l[:, :-1]; valid = (msk[:, :-1] > 0) & (x[:, 1:] != 0); hard = valid & (L > 1.0)
    peek = {"all": round(float(nxt[valid].mean()), 3), "hard": round(float(nxt[hard].mean()), 3), "hard_n": int(hard.sum()), "all_n": int(valid.sum())}
print("peek", peek)
out = {"about": "Loss in nats per word of two tiny attention models on MADE-UP reports, by make_leak.py. Rows are [step, training loss, held-out loss with the mask on].", "vocab": V, "uniform": round(math.log(V), 3), "train_reports": len(train), "test_reports": len(test), "peek": peek, "mask": res["mask"], "nomask": res["nomask"]}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "leak.json"), "w"), separators=(",", ":"))
print(res["mask"][-3:], res["nomask"][-3:])
