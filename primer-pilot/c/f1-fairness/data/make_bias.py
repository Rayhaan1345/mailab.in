"""Offline step for F1, embedding part. Run with: scratchpad/venv-d/bin/python -I make_bias.py
Uses the same GloVe vectors as chapter D2 (glove-wiki-gigaword-50, Pennington, Socher and Manning 2014; Public Domain Dedication and License).
Writes bias.json: the score of 42 occupation words on four gender directions, the spread of scores on random directions (the null band),
a test of whether a few words differ more than chance, and a Gonen and Goldberg style check that removing the direction does not remove the signal."""
import json, os
import numpy as np
from huggingface_hub import hf_hub_download
from gensim.models import KeyedVectors
from scipy.stats import spearmanr
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
HERE = os.path.dirname(os.path.abspath(__file__))
p = hf_hub_download("fse/glove-wiki-gigaword-50", "glove-wiki-gigaword-50.model")
hf_hub_download("fse/glove-wiki-gigaword-50", "glove-wiki-gigaword-50.model.vectors.npy")
kv = KeyedVectors.load(p)
unit = lambda v: v / np.linalg.norm(v)
OCC = ("midwife maid receptionist nurse babysitter hairdresser dancer singer dentist photographer tailor cashier librarian teacher cleaner journalist "
       "doctor fisherman chef farmer pharmacist soldier surgeon accountant electrician pilot mechanic driver painter plumber carpenter professor judge "
       "scientist lawyer banker programmer politician engineer secretary architect manager").split()
assert all(w in kv.key_to_index for w in OCC)
PAIRS = [("she", "he"), ("her", "his"), ("woman", "man"), ("mary", "john"), ("herself", "himself"), ("daughter", "son"), ("mother", "father"), ("gal", "guy"), ("girl", "boy"), ("female", "male")]
D = {"he - she": unit(kv["he"] - kv["she"]), "man - woman": unit(kv["man"] - kv["woman"])}
D["mean of 10 pairs"] = unit(np.mean([unit(kv[b] - kv[a]) for a, b in PAIRS], 0))
cent = []
for a, b in PAIRS:
    m = (kv[a] + kv[b]) / 2; cent += [kv[a] - m, kv[b] - m]
vt = np.linalg.svd(np.array(cent), full_matrices=False)[2]
pc = vt[0] if vt[0] @ D["mean of 10 pairs"] > 0 else -vt[0]
D["first component of the pairs"] = unit(pc)
names = list(D)
S = {n: [round(float(unit(kv[w]) @ D[n]), 3) for w in OCC] for n in names}       # positive = leans towards he, man
rho = [[round(float(spearmanr(S[a], S[b])[0]), 2) for b in names] for a in names]
# null: random directions made from two random frequent words
rng = np.random.RandomState(4127); top = kv.index_to_key[:20000]
Ou = np.array([unit(kv[w]) for w in OCC])
rand_scores = []
for t in range(2000):
    a, b = rng.choice(top, 2, replace=False)
    rand_scores.append(Ou @ unit(kv[a] - kv[b]))
rand_scores = np.array(rand_scores)
null_sd = float(rand_scores.std(0).mean())                                          # typical spread of one word's score over random directions
iN, iD = OCC.index("nurse"), OCC.index("doctor")
gap = S["mean of 10 pairs"][iD] - S["mean of 10 pairs"][iN]
gap_null = np.abs(rand_scores[:, iD] - rand_scores[:, iN])
p_gap = float((gap_null >= abs(gap)).mean())
# Gonen and Goldberg check
d = D["mean of 10 pairs"]
gendered = set(w for pr in PAIRS for w in pr) | set("he she his her him himself herself man woman men women boy girl boys girls male female mr mrs ms miss father mother son daughter brother sister husband wife king queen lord lady".split())
words = [w for w in kv.index_to_key[:50000] if w.isalpha() and w not in gendered]
Wm = np.array([unit(kv[w]) for w in words]); proj = Wm @ d; order = np.argsort(proj)
fem, mal = order[:500], order[-500:]
idx = np.concatenate([fem, mal]); lab = np.array([0] * 500 + [1] * 500)
Xo = Wm[idx]; Xd = Xo - np.outer(Xo @ d, d)
acc_o = float(cross_val_score(LinearSVC(C=1.0, max_iter=20000), Xo, lab, cv=5).mean())
acc_d = float(cross_val_score(LinearSVC(C=1.0, max_iter=20000), Xd, lab, cv=5).mean())
near = {w: [[a, round(float(b), 2)] for a, b in kv.most_similar(w, topn=6)] for w in ["secretary", "manager", "architect", "nurse", "doctor"]}
out = {"about": "Scores are cosine similarity between an occupation word and a direction, using 50-number GloVe vectors (the vectors of chapter D2). Positive leans towards he, man. The directions are built from the listed word pairs. null_sd is the typical spread of one word's score over 2,000 random directions (each the difference of two random words among the 20,000 most frequent).",
  "occupations": OCC, "directions": names, "pairs": PAIRS, "scores": S, "rank_corr": rho, "null_sd": round(null_sd, 3),
  "nurse_doctor": {"gap": round(float(gap), 3), "p_random": round(p_gap, 4)},
  "debias": {"n_words": 1000, "pool": len(words), "acc_before": round(acc_o, 3), "acc_after": round(acc_d, 3), "female_side": [words[i] for i in fem[:12]], "male_side": [words[i] for i in mal[::-1][:12]]},
  "near": near, "vocab": len(kv.key_to_index),
  "vec": {w: [round(float(v), 3) for v in kv[w]] for w in ["he", "she", "nurse", "doctor"]}}
json.dump(out, open(f"{HERE}/bias.json", "w"), separators=(",", ":"))
print("wrote bias.json", os.path.getsize(f"{HERE}/bias.json"), "bytes")
print("null_sd", null_sd, "nurse-doctor gap", gap, "p", p_gap, "acc", acc_o, acc_d); print(rho); print(near)
