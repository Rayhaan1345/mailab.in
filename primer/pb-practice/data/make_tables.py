"""Computes the chart data behind the practice set pb-practice (Part B, neural networks) and writes figdata.json next to this file.
The small calculations of the questions themselves (tanh of a weighted sum, one Adam step, an EWC penalty and so on) are done in questions.py and checked again,
with separate code, in verify.py. This file prepares only the charts that come from data:
  Q5  made-up XOR points and the best straight line (as in chapter B2),
  Q7  mini-batch runs on the 268 made-up spadenose sharks (as in chapter B3),
  Q9  gradient checking of the tiny network of chapter B4,
  Q18 forgetting curves from chapter B7 (data/b7.json),
  Q22 the badly initialised run of the broken-run lab of chapter B8 (data/b8.json),
  Q24 and Q25 the deletion curves and the cue table of chapter B9 (data/b9.json).
Run with numpy and pandas (the scratchpad Python):  python -I figdata.py
The environment variable PRIMER can point to the primer folder; the default is the repository of the site."""
import json, math, os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.environ.get("PRIMER") or (os.path.abspath(os.path.join(HERE, "..", "..")) if os.path.exists(os.path.join(HERE, "..", "..", "data", "sharks", "catch.csv")) else
          "/private/tmp/claude-502/-Users-shadabkhan-Documents-rayhaan-application/9307b589-bce9-48cb-b0f8-a7d270ba25b8/scratchpad/work/mailab.in/primer")
D = {}
sig = lambda z: 1 / (1 + np.exp(-z))

# ---------------- Q5: XOR points (made up) and the best straight line ----------------
rng = np.random.default_rng(11)
P = rng.uniform(-1, 1, size=(150, 2))
Y = (P[:, 0] * P[:, 1] < 0).astype(int)          # class 1 when x and y have different signs (chapter B2)
best = (0, None)
for k in range(360):
    th = 2 * math.pi * k / 360
    n = np.array([math.cos(th), math.sin(th)])
    proj = P @ n
    for c in np.unique(np.concatenate([proj - 1e-9, proj + 1e-9])):
        side = (proj >= c).astype(int)
        acc = max(np.mean(side == Y), np.mean(side != Y))
        if acc > best[0] + 1e-12: best = (acc, (th, float(c)))
D["q5"] = {"pts": np.round(P, 3).tolist(), "y": Y.tolist(), "acc": float(best[0]), "theta": best[1][0], "c": best[1][1]}

# ---------------- Q7: mini-batch runs on the 268 spadenose sharks ----------------
df = pd.read_csv(PRIMER + "/data/sharks/catch.csv")  # made-up records, not field data
sp = df[df.species == "spadenose"].reset_index(drop=True)
L = sp.length_cm.values; x = (L - L.mean()) / L.std(); yy = sp.mature.values.astype(float); m = len(x)
assert m == 268
def ce(w, b):
    z = w * x + b
    return float(np.mean(np.where(z >= 0, (1 - yy) * z + np.log1p(np.exp(-z)), np.log1p(np.exp(z)) - yy * z)))
def run(B, eta=0.3, epochs=10, seed=3, start=(-4.0, 2.0)):
    r = np.random.default_rng(seed); w, b = start; idx = np.arange(m); out = []
    for e in range(epochs):
        r.shuffle(idx)
        for k in range(0, m, B):
            j = idx[k:k + B]; p = sig(w * x[j] + b) - yy[j]
            w -= eta * np.mean(p * x[j]); b -= eta * np.mean(p)
        out.append(ce(w, b))
    return out
D["q7"] = {"mean": float(L.mean()), "sd": float(L.std()), "start_loss": ce(-4.0, 2.0),
           "runs": {str(B): {"steps_per_epoch": math.ceil(m / B), "loss": [round(v, 4) for v in run(B)]} for B in (1, 16, 64, 268)}}

# ---------------- Q9: gradient checking of the tiny network of chapter B4 ----------------
xin = np.array([0.22, 0.21]); ylab = 1.0
th0 = np.concatenate([np.array([[0.8, -0.6], [-0.4, 0.7]]).ravel(), [0.1, -0.2], [0.9, -0.8], [-0.9]])
def fwd(t):
    W1 = t[:4].reshape(2, 2); b1 = t[4:6]; W2 = t[6:8]; b2 = t[8]
    h = np.tanh(W1 @ xin + b1); p = sig(W2 @ h + b2)
    return -math.log(p), h, p
def bp(t, bug=None):
    W2 = t[6:8]; L_, h, p = fwd(t); d2 = p - ylab
    d1 = d2 * W2 * (1 - h ** 2) if bug != "noact" else d2 * W2
    return np.concatenate([np.outer(d1, xin).ravel(), d1, d2 * h, [d2]])
def numgrad(t, h):
    g = np.zeros(9)
    for i in range(9):
        a = t.copy(); c = t.copy(); a[i] += h; c[i] -= h
        g[i] = (fwd(a)[0] - fwd(c)[0]) / (2 * h)
    return g
rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b)))
D["q9"] = {"e": list(range(1, 16)),
           "good": [rel(bp(th0), numgrad(th0, 10.0 ** -e)) for e in range(1, 16)],
           "bug": [rel(bp(th0, "noact"), numgrad(th0, 10.0 ** -e)) for e in range(1, 16)]}

# ---------------- Q18: forgetting curves from chapter B7 ----------------
b7 = json.load(open(PRIMER + "/b7-continual-learning/data/b7.json"))
F = b7["forget"]["clarity"]
D["q18"] = {k: {"A": F[k]["accA"], "B": F[k]["accB"]} for k in ("plain", "ewc30", "uniform30")}

# ---------------- Q22: the badly initialised run of the lab of chapter B8 ----------------
b8 = json.load(open(PRIMER + "/b8-debugging/data/b8.json"))
r = b8["lab"]["init"]["runs"]["000"]
i120 = r["step"].index(120)
D["q22"] = {"step": r["step"], "train": r["train"], "val": r["val"], "acc": r["acc"], "g120": r["g"][i120], "entropy": b8["entropy"], "lazy_val": b8["lazy_val"]}

# ---------------- Q24 and Q25: chapter B9 ----------------
b9 = json.load(open(PRIMER + "/b9-explaining/data/b9.json"))
dl = b9["deletion"]["h2"]
D["q24"] = {"q": dl["q"], "curves": {k: dl["curves"][k] for k in ("saliency", "Grad-CAM", "random order")}}
D["q25"] = {k: {"gw": v["gw"], "top5": v["top5"][:2], "caption": v["caption"]} for k, v in b9["cue"].items()}

if os.path.basename(__file__) == "figdata.py":   # the copy shipped with the page (make_tables.py) only prints
    json.dump(D, open(os.path.join(HERE, "figdata.json"), "w"))
print("Q5 best straight line:", round(D["q5"]["acc"], 4))
print("Q7 start loss", round(D["q7"]["start_loss"], 3), {B: (v["steps_per_epoch"], v["loss"][-1]) for B, v in D["q7"]["runs"].items()})
print("Q9 good", ["%.1e" % v for v in D["q9"]["good"]]); print("Q9 bug ", ["%.1e" % v for v in D["q9"]["bug"]])
print("Q18 ends", {k: (v["A"][-1], v["B"][-1]) for k, v in D["q18"].items()})
print("Q22 g120", D["q22"]["g120"])
