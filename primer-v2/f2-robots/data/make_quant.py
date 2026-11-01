"""Offline step for F2: quantise the weights of the C2 digit reader and measure what happens. Run with python -I (numpy, Pillow).
Reads the C2 weights (c2-cnns/data/cnn.json) and the MNIST test digits (LeCun et al. 1998). Writes quant.json and digits.png.
Scheme: uniform quantisation of each weight tensor on its own (scale s and zero point z), biases kept in 32 bits, activations in 32 bits ('weights') or
rounded to the same number of bits with a range calibrated on 500 training digits ('weights+acts')."""
import json, os, struct, gzip, sys
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "..", ".."))
MN = sys.argv[1]            # folder with the MNIST raw files
def idx(path):
    d = open(path, "rb").read()
    if d[:2] == b"\x1f\x8b": d = gzip.decompress(d)
    nd = d[3]; shape = struct.unpack(">" + "I" * nd, d[4:4 + 4 * nd]); return np.frombuffer(d, np.uint8, offset=4 + 4 * nd).reshape(shape)
Xte = idx(f"{MN}/t10k-images-idx3-ubyte").astype(np.float32) / 255; yte = idx(f"{MN}/t10k-labels-idx1-ubyte")
Xtr = idx(f"{MN}/train-images-idx3-ubyte").astype(np.float32)[:500] / 255
J = json.load(open(f"{R}/c2-cnns/data/cnn.json"))
W = {k: np.array(J[k], np.float32) for k in ["conv1_w", "conv1_b", "conv2_w", "conv2_b", "fc_w", "fc_b"]}

def conv(x, w, b):              # x (N,C,H,W), w (O,C,3,3)
    N, C, H, Wd = x.shape; O = w.shape[0]; oh, ow = H - 2, Wd - 2
    cols = np.stack([x[:, :, i:i + oh, j:j + ow] for i in range(3) for j in range(3)], 2)      # N,C,9,oh,ow
    return np.einsum("nckhw,ock->nohw", cols, w.reshape(O, C, 9), optimize=True) + b[None, :, None, None]
def pool(x):
    N, C, H, Wd = x.shape; return x[:, :, :H // 2 * 2, :Wd // 2 * 2].reshape(N, C, H // 2, 2, Wd // 2, 2).max((3, 5))
def forward(x, P, actq=None):
    q = (lambda a, i: a) if actq is None else actq
    a = q(pool(np.maximum(conv(x, P["conv1_w"], P["conv1_b"]), 0)), 0)
    a = q(pool(np.maximum(conv(a, P["conv2_w"], P["conv2_b"]), 0)), 1)
    return a.reshape(len(a), -1) @ P["fc_w"].T + P["fc_b"]
def acc(P, actq=None, X=Xte, y=yte, bs=1000):
    pred = np.concatenate([forward(X[i:i + bs, None], P, actq).argmax(1) for i in range(0, len(X), bs)]); return pred
base_pred = acc(W); base = float((base_pred == yte).mean())
print("float32 accuracy", base, "recorded", J["test_acc"])

def quant_params(w, bits):
    lo, hi = float(w.min()), float(w.max()); levels = 2 ** bits - 1
    s = (hi - lo) / levels; z = int(round(-lo / s)); return s, z
def quantise(w, bits):
    s, z = quant_params(w, bits); q = np.clip(np.round(w / s) + z, 0, 2 ** bits - 1); return q.astype(np.int64), s, z
def dequant(q, s, z): return (s * (q - z)).astype(np.float32)
def qweights(bits):
    P = dict(W)
    for k in ["conv1_w", "conv2_w", "fc_w"]:
        q, s, z = quantise(W[k], bits); P[k] = dequant(q, s, z)
    return P
# activation ranges from 500 training digits
cal = []; Pf = W
a = pool(np.maximum(conv(Xtr[:, None], Pf["conv1_w"], Pf["conv1_b"]), 0)); cal.append(float(a.max()))
a = pool(np.maximum(conv(a, Pf["conv2_w"], Pf["conv2_b"]), 0)); cal.append(float(a.max()))
def actq_for(bits):
    def f(a, i):
        s = cal[i] / (2 ** bits - 1); return np.clip(np.round(a / s), 0, 2 ** bits - 1) * s
    return f
def f16(P): return {k: v.astype(np.float16).astype(np.float32) if k.endswith("_w") else v for k, v in P.items()}
def bf16(P):
    out = dict(P)
    for k in ["conv1_w", "conv2_w", "fc_w"]:
        u = W[k].view(np.uint32); r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32); out[k] = r.view(np.float32)
    return out
BITS = list(range(2, 17))
res = {"bits": BITS, "w_acc": [], "w_flips": [], "wa_acc": [], "wa_flips": [], "mean_abs_err": []}
for b in BITS:
    P = qweights(b); p = acc(P)
    res["w_acc"].append(round(float((p == yte).mean()), 4)); res["w_flips"].append(int((p != base_pred).sum()))
    pa = acc(P, actq_for(b)); res["wa_acc"].append(round(float((pa == yte).mean()), 4)); res["wa_flips"].append(int((pa != base_pred).sum()))
    res["mean_abs_err"].append(float(np.mean(np.concatenate([np.abs(P[k] - W[k]).ravel() for k in ["conv1_w", "conv2_w", "fc_w"]]))))
    print(b, res["w_acc"][-1], res["w_flips"][-1], res["wa_acc"][-1], res["wa_flips"][-1])
p16 = acc(f16(W)); pb = acc(bf16(W))
res["f16"] = {"acc": round(float((p16 == yte).mean()), 4), "flips": int((p16 != base_pred).sum())}
res["bf16"] = {"acc": round(float((pb == yte).mean()), 4), "flips": int((pb != base_pred).sum())}
print("f16", res["f16"], "bf16", res["bf16"])
# weights to ship (rounded as in c2; they are the 32-bit reference)
tensors = {k: W[k].ravel().round(4).tolist() for k in W}
dims = {"conv1_w": [6, 1, 3, 3], "conv2_w": [12, 6, 3, 3], "fc_w": [10, 300]}
# first 400 test digits as a sprite 20 x 20 of 28 x 28
N = 400; sprite = np.zeros((20 * 28, 20 * 28), np.uint8)
for i in range(N): r, c = divmod(i, 20); sprite[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = (Xte[i] * 255).round().astype(np.uint8)
Image.fromarray(sprite).save(f"{HERE}/digits.png", optimize=True)
sub_base = float((base_pred[:N] == yte[:N]).mean())
sub = {"labels": "".join(str(int(v)) for v in yte[:N]), "n": N, "w_acc": [], "wa_acc": []}
for b in BITS:
    sub["w_acc"].append(round(float((acc(qweights(b), X=Xte[:N], y=yte[:N]) == yte[:N]).mean()), 4))
    sub["wa_acc"].append(round(float((acc(qweights(b), actq_for(b), X=Xte[:N], y=yte[:N]) == yte[:N]).mean()), 4))
print("subset base", sub_base, sub["w_acc"])
# the worked example: first weight of the dense layer
w0 = float(W["fc_w"].ravel()[0]); s, z = quant_params(W["fc_w"], 8); q, _, _ = quantise(W["fc_w"], 8)
ex = {"layer": "fc_w", "w0": w0, "min": float(W["fc_w"].min()), "max": float(W["fc_w"].max()), "s8": s, "z8": z, "q0": int(q.ravel()[0]), "back": float(dequant(q, s, z).ravel()[0])}
print(ex)
out = {"about": "Weights of the C2 digit reader (3,730 numbers; trained offline on MNIST, rounded to 4 decimals) and measured results of weight quantisation on the 10,000 MNIST test digits.",
       "dims": dims, "w": tensors, "base_acc": round(base, 4), "base_acc_subset": round(sub_base, 4), "results": res, "subset": sub, "example": ex, "cal": cal,
       "bias_note": "Biases stay in 32 bits."}
json.dump(out, open(f"{HERE}/quant.json", "w"), separators=(",", ":"))
print(os.path.getsize(f"{HERE}/quant.json"), os.path.getsize(f"{HERE}/digits.png"))
