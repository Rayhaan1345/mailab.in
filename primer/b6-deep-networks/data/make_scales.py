# Signal size after 20 layers against the spread of the weights, for tanh and ReLU and three layer widths.
# For each spread s: RMS of the activations at layer 20 (input RMS 1) and RMS of the slope at layer 1 (slope RMS 1 at layer 20).
# Weights are normal with spread s; batch 128; mean of 3 seeds in log10. Run: python -I make_scales.py
import json, numpy as np
D, B = 20, 128
spreads = np.logspace(-2, 0.3, 36)
out = {"about": "Made-up random inputs and slopes (normal, size 1). Entries are log10 of the RMS size at layer 20 (activations) and at layer 1 (slopes). Computed with NumPy, 3 seeds, 20 layers, batch 128.", "spreads": [round(float(s), 5) for s in spreads], "widths": {}}
for n in (25, 100, 400):
    res = {"tanh": {"act": [], "grad": []}, "relu": {"act": [], "grad": []}}
    for act in ("tanh", "relu"):
        for s in spreads:
            la, lg = [], []
            for seed in range(3):
                rs = np.random.RandomState(seed)
                x = rs.randn(B, n); Ws = [rs.randn(n, n) * s for _ in range(D)]; g = rs.randn(B, n)
                h, zs, hs = x, [], [x]
                for W in Ws:
                    z = h @ W; h = np.tanh(z) if act == "tanh" else np.maximum(z, 0); zs.append(z); hs.append(h)
                gh = g
                for l in reversed(range(D)):
                    dz = gh * ((1 - hs[l + 1] ** 2) if act == "tanh" else (zs[l] > 0))
                    gh = dz @ Ws[l].T
                    if l == 0: first = dz
                la.append(np.log10(max(np.sqrt(np.mean(hs[-1] ** 2)), 1e-300))); lg.append(np.log10(max(np.sqrt(np.mean(first ** 2)), 1e-300)))
            res[act]["act"].append(round(float(np.mean(la)), 3)); res[act]["grad"].append(round(float(np.mean(lg)), 3))
    out["widths"][str(n)] = res
json.dump(out, open("scales.json", "w"), separators=(",", ":"))
for n in ("25", "100", "400"):
    r = out["widths"][n]
    print(n, "tanh act at spreads", [(out["spreads"][i], r["tanh"]["act"][i]) for i in (0, 12, 18, 24, 35)])
    print(n, "relu act", [(out["spreads"][i], r["relu"]["act"][i]) for i in (0, 12, 18, 24, 35)])
