# Adam's v against the empirical Fisher diagonal and the curvature, for logistic regression on the 268 made-up spadenose records
# (inputs: standardised length and girth; label: mature). Run: python -I make_fisher.py   Writes fisher.json.
import json, numpy as np
d = json.load(open("../../data/sharks/catch.json"))
R = [r for r in d["records"] if r["species"] == "spadenose"]
L = np.array([r["length_cm"] for r in R]); G = np.array([r["girth_cm"] for r in R]); y = np.array([r["mature"] for r in R], float)
Z = np.c_[(L - L.mean()) / L.std(), (G - G.mean()) / G.std(), np.ones(len(L))]
sig = lambda z: 1 / (1 + np.exp(-z))
th = np.array([-1.0, 1.0, 1.0]); rs = np.random.RandomState(4)
m = np.zeros(3); v = np.zeros(3); eta, b1, b2 = 0.02, 0.9, 0.99
rows = {"t": [], "theta": [], "loss": [], "vhat": [], "fisher": [], "hess": []}
for t in range(1, 601):
    i = rs.randint(len(y)); g = (sig(Z[i] @ th) - y[i]) * Z[i]       # one example per step
    m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
    th = th - eta * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + 1e-8)
    if t % 10 == 0:
        p = sig(Z @ th)
        rows["t"].append(t); rows["theta"].append(np.round(th, 4).tolist())
        rows["loss"].append(round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4))
        rows["vhat"].append(np.round(v / (1 - b2 ** t), 5).tolist())
        rows["fisher"].append(np.round(np.mean(((p - y) ** 2)[:, None] * Z ** 2, axis=0), 5).tolist())   # mean squared per-example slope
        rows["hess"].append(np.round(np.mean((p * (1 - p))[:, None] * Z ** 2, axis=0), 5).tolist())    # curvature of the loss
out = {"about": "Made-up spadenose records (catch table). Logistic regression on standardised length, girth and a bias, Adam with one example per step (eta 0.02, beta1 0.9, beta2 0.99, seed 4). vhat is Adam's bias-corrected v; fisher is the mean squared per-example slope at the current weights; hess is the second derivative of the loss for each weight.", "params": ["length weight", "girth weight", "bias"], **rows}
json.dump(out, open("fisher.json", "w"), separators=(",", ":"))
print(len(rows["t"]), "rows")
for k in (0, 5, 20, 59): print(rows["t"][k], rows["vhat"][k], rows["fisher"][k], rows["hess"][k])
