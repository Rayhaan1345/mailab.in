"""Computes the data behind the practice set pc-practice (Part C, images) and prints key values. With the argument --write it also writes figdata.json next to this file.
The small calculations of the questions themselves (sizes, parameter counts, receptive fields, one softmax) are done in questions.py and checked again,
with separate code (PyTorch layers and autograd), in verify.py. This file prepares the grids and the charts:
  Q1  a 6 by 6 crop of an MNIST digit 7 (chapter C1, data/digits.json),
  Q2  mean red, green and blue of the hammerhead photo at 0, 5 and 10 m (depth.py, the model of chapters C1 and C3),
  Q3, Q6, Q7, Q13  small made-up grids (the feature map of Q3 and Q13 is computed here),
  Q10 parameter counts of three kinds of layer against picture side,
  Q14 shift and turn curves of two digit readers (chapter C2, data/cnn.json),
  Q15 the ten scores of a misread digit (the network of chapter C2, rerun with PyTorch from data/cnn.json),
  Q17 ResNet-18 shark-class confidence against depth (chapter C3, data/resnet.json),
  Q18, Q20, Q21 digit-reader curves and the stream (chapter C3, data/murk.json).
Needs numpy, Pillow and PyTorch (the scratchpad Python):  python -I figdata.py [--write]
The environment variable PRIMER can point to the primer folder; the default is the repository of the site."""
import base64, json, math, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.environ.get("PRIMER") or (os.path.abspath(os.path.join(HERE, "..", "..")) if os.path.exists(os.path.join(HERE, "..", "..", "data", "sharks", "catch.csv")) else
          "/private/tmp/claude-502/-Users-shadabkhan-Documents-rayhaan-application/9307b589-bce9-48cb-b0f8-a7d270ba25b8/scratchpad/work/mailab.in/primer")
J = lambda p: json.load(open(os.path.join(PRIMER, p)))
D = {}

# ---------------- Q1: a crop of the digit 7 of Tag 4127 (MNIST) ----------------
dig = J("c1-convolution/data/digits.json")["digits"][21]
assert dig["label"] == 7
img28 = np.frombuffer(base64.b64decode(dig["b64"]), dtype=np.uint8).reshape(28, 28)
D["q1"] = {"crop": img28[9:15, 16:22].astype(int).tolist(), "row0": 9, "col0": 16, "row": 2, "col": 4}

# ---------------- Q2: mean R, G, B of the hammerhead photo at three depths ----------------
sys.path.insert(0, os.path.join(PRIMER, "data", "sharks"))
from depth import degrade
from PIL import Image
photo = Image.open(os.path.join(PRIMER, "data", "sharks", "photos", "hammerhead-1.jpg")).convert("RGB")
D["q2"] = {"depths": [0, 5, 10], "means": [np.array(degrade(photo, z)).reshape(-1, 3).mean(0).round(1).tolist() for z in (0, 5, 10)]}

# ---------------- Q3: a convolution by hand (made-up grey values 0 to 9) ----------------
def conv(img, K, s=1, p=0):
    img = np.pad(img, p); k = K.shape[0]; n = (img.shape[0] - k) // s + 1
    return np.array([[(img[i * s:i * s + k, j * s:j * s + k] * K).sum() for j in range(n)] for i in range(n)])
TB = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]); LR = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
BL = np.ones((3, 3)) / 9; SH = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
I3 = np.array([[0, 0, 5, 9, 2], [0, 0, 7, 8, 0], [0, 3, 9, 1, 0], [0, 6, 8, 0, 0], [2, 9, 3, 0, 0]])
D["q3"] = {"img": I3.tolist(), "K": TB.tolist(), "b": -2, "map": conv(I3, TB).tolist()}

# ---------------- Q5: windows on a 10 by 10 picture ----------------
D["q5"] = {"n": 10, "k": 3, "s": 3}

# ---------------- Q6 and Q7: pooling ----------------
D["q6"] = {"map": [[1, 0, 3, 2, 4], [0, 5, 1, 0, 2], [2, 1, 0, 6, 1], [3, 0, 2, 1, 0], [1, 7, 0, 9, 2]]}
m7 = np.zeros((8, 8), int); m7[1, 2] = 9; m7[3, 4] = 6; m7[4, 6] = 7; m7[6, 2] = 8
D["q7"] = {"map": m7.tolist()}

# ---------------- Q10: parameter counts against picture side ----------------
ns = [8, 16, 32, 64]; cin, k, F = 3, 3, 16
D["q10"] = {"n": ns, "dense": [(n * n * cin + 1) * n * n * F for n in ns], "local": [(k * k * cin + 1) * n * n * F for n in ns], "conv": [(k * k * cin + 1) * F for n in ns]}

# ---------------- Q13: which filter made the map ----------------
I13 = np.array([[0, 0, 9, 9, 0, 0], [0, 0, 9, 9, 0, 0], [0, 0, 9, 9, 0, 0], [0, 0, 9, 9, 9, 9], [0, 0, 9, 9, 9, 9], [0, 0, 0, 0, 0, 0]])
maps = {n: np.round(conv(I13, K)).astype(int).tolist() for n, K in (("LR", LR), ("TB", TB), ("BL", BL), ("SH", SH))}
D["q13"] = {"img": I13.tolist(), "map": maps["TB"], "all": maps}

# ---------------- Q14: shift and turn curves (chapter C2) ----------------
rb = J("c2-cnns/data/cnn.json")["robust"]
D["q14"] = {"shift": {"x": rb["plain"]["shift_px"], "plain": rb["plain"]["shift_acc"], "aug": rb["aug"]["shift_acc"]},
            "rot": {"x": rb["plain"]["rot_deg"], "plain": rb["plain"]["rot_acc"], "aug": rb["aug"]["rot_acc"]}}

# ---------------- Q15: the ten scores of a misread digit ----------------
import torch, torch.nn.functional as Fn
cnn = J("c2-cnns/data/cnn.json")
T = lambda k: torch.tensor(cnn[k], dtype=torch.float32)
def scores(a):
    x = torch.tensor(a, dtype=torch.float32)[None, None] / 255.0
    h = Fn.max_pool2d(Fn.relu(Fn.conv2d(x, T("conv1_w"), T("conv1_b"))), 2)
    h = Fn.max_pool2d(Fn.relu(Fn.conv2d(h, T("conv2_w"), T("conv2_b"))), 2)
    return (h.flatten(1) @ T("fc_w").T + T("fc_b"))[0]
mis = cnn["misread"][0]
am = np.frombuffer(base64.b64decode(mis["b64"]), dtype=np.uint8).reshape(28, 28)
z = scores(am)
assert mis["label"] == 3 and mis["pred"] == 8 and int(z.argmax()) == 8
D["q15"] = {"digit": am.astype(int).tolist(), "z": [round(float(v), 2) for v in z], "label": 3}

# ---------------- Q17: ResNet-18 shark-class confidence against depth ----------------
rn = J("c3-domain-shift/data/resnet.json")
dep = rn["depths"]; cv = rn["curves"]
sharks = ["hammerhead-1", "hammerhead-2", "milk-1", "milk-2", "spadenose-1"]; non = ["dolphin-1", "reeffish-1", "turtle-1"]
mean = lambda names, a: [float(np.mean([cv[p][a]["shark"][i] for p in names])) for i in range(len(dep))]
D["q17"] = {"depths": dep, "hammerhead": cv["hammerhead-1"]["0.0"]["shark"], "milk": cv["milk-2"]["0.0"]["shark"], "spadenose": cv["spadenose-1"]["0.0"]["shark"],
            "non": mean(non, "0.0")}
D["q20r"] = {"depths": dep, "stored": mean(sharks, "0.0"), "mix25": mean(sharks, "0.25"), "all": mean(sharks, "1.0")}

# ---------------- Q18, Q20, Q21: the digit reader (chapter C3) ----------------
mk = J("c3-domain-shift/data/murk.json")
D["q18"] = {"depths": mk["depths"], "A": {"clean": mk["A"]["clean"], "aug": mk["A"]["aug"]}, "B": {"clean": mk["B"]["clean"], "aug": mk["B"]["aug"]}}
D["q20d"] = {"depths": mk["depths"], "stored": mk["A"]["clean"], "adapt": mk["A"]["clean_adapt"]}
st = mk["stream"]
D["q21"] = {"zs": st["zs"], "frozen": st["frozen"], "stats": st["stats"], "tent": st["tent"]}

if "--write" in sys.argv:
    json.dump(D, open(os.path.join(HERE, "figdata.json"), "w"), separators=(",", ":"))
    print("written figdata.json")
print("Q1 crop", D["q1"]["crop"])
print("Q2 mean R, G, B at 0, 5, 10 m", D["q2"]["means"])
print("Q3 feature map before the bias", D["q3"]["map"])
print("Q10 counts (dense, local, conv) at n = 8, 16, 32, 64", list(zip(D["q10"]["dense"], D["q10"]["local"], D["q10"]["conv"])))
print("Q13 map", D["q13"]["map"])
print("Q14 shift curves", D["q14"]["shift"]["plain"], D["q14"]["shift"]["aug"])
print("Q15 scores", D["q15"]["z"])
print("Q17 at 0, 5, 10, 15, 30 m: hammerhead, milk, spadenose, non-sharks", [[round(D["q17"][k][D["q17"]["depths"].index(z)], 3) for z in (0.0, 5.0, 10.0, 15.0, 30.0)] for k in ("hammerhead", "milk", "spadenose", "non")])
print("Q20 five-shark means at 0, 5, 10, 15 m (stored, 25% new, all new)", [[round(D["q20r"][k][D["q20r"]["depths"].index(z)], 3) for z in (0.0, 5.0, 10.0, 15.0)] for k in ("stored", "mix25", "all")])
print("Q21 stream means (frozen, statistics, statistics plus scale and shift)", [round(float(np.mean(D["q21"][k])), 3) for k in ("frozen", "stats", "tent")])
