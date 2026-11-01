"""Offline step for D5: the weights and the true numbers of the tiny transformer block used in the step-through figure and in the NumPy
code of the chapter. All numbers are MADE UP (random weights, seed 4127), not a trained model. Writes block.json.
Run with: python3 -I make_block.py   (needs NumPy)"""
import json, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(4127)
d, H, dk, dff = 4, 2, 2, 8
X = rng.normal(0, 1, (3, d)).round(1)
Wq, Wk, Wv, Wo = (rng.normal(0, 0.5, (d, d)).round(2) for _ in range(4))
W1 = rng.normal(0, 0.5, (d, dff)).round(2)
W2 = rng.normal(0, 0.5, (dff, d)).round(2)

def layer_norm(x, eps=1e-5):
    mu = x.mean(-1, keepdims=True); var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)
def gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
def softmax(s):
    e = np.exp(s - s.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)
def attention(h, trace=None):
    n = len(h); q, k, v = h @ Wq, h @ Wk, h @ Wv
    heads, ws = [], []
    for i in range(H):
        sl = slice(i * dk, (i + 1) * dk)
        s = q[:, sl] @ k[:, sl].T / np.sqrt(dk)
        s = np.where(np.tril(np.ones((n, n), bool)), s, -np.inf)
        w = softmax(s); ws.append(w); heads.append(w @ v[:, sl])
    if trace is not None: trace["weights"] = ws
    return np.concatenate(heads, axis=1) @ Wo
def mlp(h): return gelu(h @ W1) @ W2

tr = {}
h1 = layer_norm(X); a = attention(h1, tr); x1 = X + a
h2 = layer_norm(x1); m = mlp(h2); x2 = x1 + m
# the original 2017 order: sublayer first, then add, then norm
a_p = attention(X); y1 = layer_norm(X + a_p); m_p = mlp(y1); y2 = layer_norm(y1 + m_p)
out = {"about": "A tiny transformer block with MADE-UP random weights (numpy default_rng(4127)), three tokens of four numbers, two heads of two numbers, an MLP with 8 hidden units. By make_block.py.",
       "d": d, "H": H, "dk": dk, "dff": dff, "tokens": ["the", "young", "hammerhead"],
       "X": X.tolist(), "Wq": Wq.tolist(), "Wk": Wk.tolist(), "Wv": Wv.tolist(), "Wo": Wo.tolist(), "W1": W1.tolist(), "W2": W2.tolist(),
       "pre": {"h1": h1.round(4).tolist(), "attn": a.round(4).tolist(), "x1": x1.round(4).tolist(), "h2": h2.round(4).tolist(), "mlp": m.round(4).tolist(), "x2": x2.round(4).tolist(),
               "weights": [w.round(4).tolist() for w in tr["weights"]]},
       "post": {"attn": a_p.round(4).tolist(), "y1": y1.round(4).tolist(), "mlp": m_p.round(4).tolist(), "y2": y2.round(4).tolist()}}
json.dump(out, open(os.path.join(HERE, "block.json"), "w"), separators=(",", ":"))
np.set_printoptions(precision=3, suppress=True)
print("X\n", X); print("h1\n", h1); print("attn\n", a); print("x1\n", x1); print("h2\n", h2); print("mlp\n", m); print("x2\n", x2)
print("weights head0\n", tr["weights"][0]); print("weights head1\n", tr["weights"][1])
print("post y2\n", y2)
for i in range(3): print("cos(x2,X) tok", i, float(x2[i] @ X[i] / np.linalg.norm(x2[i]) / np.linalg.norm(X[i])), "norms", np.linalg.norm(X[i]), np.linalg.norm(a[i]), np.linalg.norm(m[i]), np.linalg.norm(x2[i]))
