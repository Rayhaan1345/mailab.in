import json, base64, numpy as np, torch, torchvision
exec(open("train_cnn.py").read().split("mA = train(True)")[0])   # reuse Net and data
ck = torch.load("out/c2_models.pt"); m = Net(); m.load_state_dict(ck["aug"]); m.eval()
def r(t, d=4): return np.round(t.detach().numpy(), d).tolist()
W = {"conv1_w": r(m.conv1.weight), "conv1_b": r(m.conv1.bias), "conv2_w": r(m.conv2.weight), "conv2_b": r(m.conv2.bias),
     "fc_w": r(m.fc.weight), "fc_b": r(m.fc.bias)}
npar = sum(p.numel() for p in m.parameters())
with torch.no_grad(): logits = m(Xte); pr = logits.softmax(1); pred = logits.argmax(1)
acc = float((pred == yte).float().mean())
# example digits: 2 confident per class, and the first 3 misread ones
ex = []
for c in range(10):
    n = 0
    for i in range(len(yte)):
        if yte[i] == c and pred[i] == c and pr[i, c] > 0.995 and Xte[i].sum() > 60 and Xte[i].sum() < 130:
            ex.append({"label": c, "pred": c, "idx": i, "b64": base64.b64encode(bytes((Xte[i, 0] * 255).round().byte().flatten().tolist())).decode()}); n += 1
            if n == 2: break
bad = []
for i in range(len(yte)):
    if pred[i] != yte[i]:
        bad.append({"label": int(yte[i]), "pred": int(pred[i]), "idx": i, "b64": base64.b64encode(bytes((Xte[i, 0] * 255).round().byte().flatten().tolist())).decode()})
        if len(bad) == 3: break
rob = json.load(open("out/c2_robustness.json"))
J = {"about": "Small CNN for chapter C2, trained offline in PyTorch on MNIST (LeCun et al. 1998) with random shifts up to 2 px, rotations up to 10 degrees and zoom 0.9 to 1.1. Layout: conv 1->6 (3x3), ReLU, max-pool 2, conv 6->12 (3x3), ReLU, max-pool 2, flatten 300, dense 10, softmax. Weights rounded to 4 decimals.",
     "params": npar, "test_acc": round(acc, 4), "test_n": len(yte), **W, "examples": ex, "misread": bad, "robust": rob}
json.dump(J, open("out/c2_cnn.json", "w"), separators=(",", ":"))
print(npar, acc, len(ex), len(bad), [ (b["label"], b["pred"]) for b in bad])
# reference activations for the JS test
x = Xte[:5]; 
with torch.no_grad(): a1, p1, a2, p2 = m.maps(x); lg = m(x)
json.dump({"idx": list(range(5)), "logits": lg.numpy().tolist(), "a1_sum": a1.sum((1,2,3)).tolist(), "a2_sum": a2.sum((1,2,3)).tolist()}, open("out/c2_ref.json", "w"))
# C1 digits: 3 per class from the test set (clean ones), 28x28 bytes
c1 = []
for c in range(10):
    n = 0
    for i in range(len(yte)):
        if yte[i] == c and pred[i] == c and pr[i, c] > 0.999 and 70 < Xte[i].sum() < 120:
            c1.append({"label": c, "b64": base64.b64encode(bytes((Xte[i, 0] * 255).round().byte().flatten().tolist())).decode()}); n += 1
            if n == 3: break
json.dump({"about": "Handwritten digits from the MNIST test set (LeCun, Bottou, Bengio and Haffner, 1998): 28 by 28 pixels, values 0 to 255.", "digits": c1}, open("out/c1_digits.json", "w"), separators=(",", ":"))
import os; print(os.path.getsize("out/c2_cnn.json"), os.path.getsize("out/c1_digits.json"))
