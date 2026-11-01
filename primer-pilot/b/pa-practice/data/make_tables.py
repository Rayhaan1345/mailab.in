"""Computes the numbers and chart data behind the questions of the practice set pa-practice from the made-up
catch table, the dive log and the published Pima data. Writes figdata.json next to this file when run in the generator folder, and prints the numbers otherwise.
Run with the scratchpad Python (numpy, pandas, scikit-learn, xgboost):  python -I figdata.py"""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PRIMER = os.environ.get("PRIMER") or (os.path.abspath(os.path.join(HERE, "..", "..")) if os.path.exists(os.path.join(HERE, "..", "..", "data", "sharks", "catch.csv")) else
          "/private/tmp/claude-502/-Users-shadabkhan-Documents-rayhaan-application/9307b589-bce9-48cb-b0f8-a7d270ba25b8/scratchpad/work/mailab.in/primer")
D = {}
df = pd.read_csv(PRIMER + "/data/sharks/catch.csv")  # made-up records, not field data
pima = json.load(open(PRIMER + "/data/clinic/pima.json"))
PX, PY, PT = np.array(pima["X"]), np.array(pima["y"]), np.array(pima["test"]).astype(bool)
PF = pima["features"]
sig = lambda z: 1 / (1 + np.exp(-z))
sp = df[df.species == "spadenose"].reset_index(drop=True)

# ---- Q2: species counts by source
D["q2"] = {"landing": df[df.source == "landing"].species.value_counts().to_dict(), "dive": df[df.source == "dive"].species.value_counts().to_dict()}

# ---- Q3 and Q7: the 268 spadenose records
D["spadenose"] = {"L": sp.length_cm.round(1).tolist(), "W": sp.weight_kg.round(3).tolist()}
D["q3"] = {"w": 0.0155, "b": -0.38, "A": [30, 0.085], "B": [50, 0.395], "tag": 1079}
r = sp[sp.record_id == 1079].iloc[0]
D["q3"]["shark"] = [float(r.length_cm), float(r.weight_kg)]
slope, icpt = np.polyfit(np.log(sp.length_cm), np.log(sp.weight_kg), 1)
D["q7"] = {"exp": float(slope), "a": float(np.exp(icpt))}
for L in (25, 50, 100):
    D["q7"]["W%d" % L] = float(np.exp(icpt) * L ** slope)

# ---- Q4 table
g = sp.assign(g=(sp.weight_kg * 1000).round().astype(int))
q4 = []
for t in (1419, 5787, 3526, 6101, 5524):
    r = g[g.record_id == t].iloc[0]
    q4.append({"tag": t, "L": float(r.length_cm), "g": int(r.g), "pred": int(round(15.5 * r.length_cm - 380))})
D["q4"] = q4

# ---- Q5 table
q5 = []
for t in (3232, 1534, 1714, 4006):
    r = sp[sp.record_id == t].iloc[0]
    q5.append({"tag": t, "L": float(r.length_cm), "x": (float(r.length_cm) - 40) / 5, "y": float(r.weight_kg)})
D["q5"] = q5

# ---- Q6: loss curves for four learning rates, 268 spadenose standardised, start (w, b) = (0, 0.55)
z = (sp.length_cm.values - sp.length_cm.mean()) / sp.length_cm.std(ddof=0)
y = sp.weight_kg.values
curves = {}
for eta in (0.02, 0.3, 0.9, 1.05):
    w, b, ls = 0.0, 0.55, []
    for s in range(21):
        rr = y - (w * z + b)
        ls.append(float((rr ** 2).mean()))
        w += eta * 2 * (rr * z).mean(); b += eta * 2 * rr.mean()
    curves[str(eta)] = ls
D["q6"] = curves

# ---- Q8: bamboo maturity model
from sklearn.linear_model import LogisticRegression
bm = df[df.species == "bamboo"]
m = LogisticRegression(C=1e6, max_iter=10000).fit(bm[["length_cm"]], bm.mature)
D["q8"] = {"w": float(m.coef_[0][0]), "L50": float(-m.intercept_[0] / m.coef_[0][0])}

# ---- Q9: Pima logistic regression on glucose and BMI (training women, zeros dropped)
G, B = PX[:, 1], PX[:, 5]
ok = (G > 0) & (B > 0) & ~PT
m = LogisticRegression(C=1e6, max_iter=10000).fit(np.c_[G[ok], B[ok]], PY[ok])
D["q9"] = {"fit": [float(m.coef_[0][0]), float(m.coef_[0][1]), float(m.intercept_[0])], "used": [0.04, 0.09, -8.3],
           "pts": [[float(a), float(b_), int(c)] for a, b_, c in zip(G[ok], B[ok], PY[ok])], "womanA": [float(PX[31, 1]), float(PX[31, 5])]}

# ---- Q10: log loss table from the A3 spadenose model (length and girth)
w3 = np.array([4.732, -0.1215]); b3 = 2.4811; mu = np.array([39.138, 10.993]); sd = np.array([8.614, 2.455])
q10 = []
for t in (6297, 9840, 6406, 5910):
    r = sp[sp.record_id == t].iloc[0]
    zz = ((np.array([r.length_cm, r.girth_cm]) - mu) / sd) @ w3 + b3
    q10.append({"tag": t, "L": float(r.length_cm), "G": float(r.girth_cm), "y": int(r.mature), "p": round(float(sig(zz)), 2)})
D["q10"] = q10

# ---- Q11: softmax scores of Tag 3483 from the A3 softmax classifier
s3 = json.load(open(PRIMER + "/a3-logistic-regression/data/softmax.json"))
r = df[df.record_id == 3483].iloc[0]
zs = np.array(s3["w_length"]) * r.length_cm + np.array(s3["w_girth"]) * r.girth_cm + np.array(s3["b"])
D["q11"] = {"classes": s3["classes"], "z": [round(float(v), 2) for v in zs], "L": float(r.length_cm), "G": float(r.girth_cm), "true": r.species}

# ---- Q12: polynomial fits, 15 training sharks (seed 2), test on the other spadenose records
Lc, Wc = sp.length_cm.values, sp.weight_kg.values
idx = np.random.RandomState(2).permutation(len(sp)); tr, te = idx[:15], idx[15:]
rm = lambda a, b_: float(np.sqrt(np.mean((a - b_) ** 2)))
q12 = []
for M in range(1, 10):
    p = np.polynomial.Polynomial.fit(Lc[tr], Wc[tr], M)
    q12.append([M, rm(p(Lc[tr]), Wc[tr]), rm(p(Lc[te]), Wc[te])])
D["q12"] = q12

# ---- Q13: dive log, tree on depth, light, visibility and temperature; shuffled folds against time blocks
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_predict
dive = json.load(open(PRIMER + "/data/sharks/dive.json"))
dd = pd.DataFrame(dive["rows"], columns=dive["columns"])
yd = dd.detection.notna().astype(int).values
Xd = dd[["depth_m", "light", "visibility_m", "temp_c"]].values
q13 = {"lazy": float(1 - yd.mean()), "frames": len(yd), "with_detection": int(yd.sum())}
for name, cv in (("random", KFold(5, shuffle=True, random_state=4127)), ("blocks", KFold(5, shuffle=False))):
    pr = cross_val_predict(DecisionTreeClassifier(random_state=0), Xd, yd, cv=cv)
    q13[name] = {"acc": float((pr == yd).mean()), "recall": float(pr[yd == 1].mean())}
D["q13"] = q13

# ---- hammerhead detector scores (out-of-fold, five measurements)
sc = json.load(open(PRIMER + "/a4-grading-models/data/scores.json"))
assert sc["ids"] == df.record_id.tolist()
df["ph"] = sig(np.array(sc["hammer"]["z5"]))
yh = (df.species == "hammerhead").astype(int).values
def cm(t):
    f = df.ph.values >= t
    return {"t": t, "TP": int((f & (yh == 1)).sum()), "FP": int((f & (yh == 0)).sum()), "FN": int((~f & (yh == 1)).sum()), "TN": int((~f & (yh == 0)).sum())}
D["q14"] = cm(0.5)
D["q16"] = [cm(t) for t in (0.5, 0.3, 0.1, 0.05)]

# ---- Q15: ten records ranked by detector score (4 hammerheads, 6 other sharks)
H, N = df[df.species == "hammerhead"], df[df.species != "hammerhead"]
used, q15 = [], []
for kind, frame, tgt in [("H", H, .92), ("N", N, .78), ("H", H, .63), ("N", N, .50), ("H", H, .31), ("N", N, .24), ("N", N, .13), ("H", H, .09), ("N", N, .05), ("N", N, .02)]:
    d_ = (frame.ph - tgt).abs().copy()
    for u in used: d_[frame.record_id == u] = 9
    r = frame.loc[d_.idxmin()]; used.append(r.record_id)
    q15.append({"kind": kind, "tag": int(r.record_id), "species": r.species, "p": round(float(r.ph), 3)})
D["q15"] = q15

# ---- Q17 and Q18: Pima logistic regression, cross-validated (A4 data)
cl = json.load(open(PRIMER + "/a4-grading-models/data/clinic.json"))
cy, cage, ccv, cho = np.array(cl["y"]), np.array(cl["age"]), np.array(cl["cv"]), np.array(cl["ho"])
fl = ccv >= np.log(0.3 / 0.7)
D["q17"] = {"t": 0.3, "TP": int((fl & (cy == 1)).sum()), "FN": int((~fl & (cy == 1)).sum()), "FP": int((fl & (cy == 0)).sum()), "TN": int((~fl & (cy == 0)).sum()), "prev": 0.08}
q18 = []
for lo, hi in ((21, 29), (30, 39), (40, 49), (50, 99)):
    mk = PT & (cage >= lo) & (cage <= hi)
    q18.append({"lo": lo, "hi": hi, "n": int(mk.sum()), "pos": int(((cy == 1) & mk).sum()), "found": int(((cho >= 0) & (cy == 1) & mk).sum())})
D["q18"] = q18

# ---- Q19: 20 training sharks, spadenose or not, split at length 41.5 cm (neighbours in a lane are at least 1.5 cm apart)
tr_ = df[df.record_id % 10 >= 3]
def nearest(is_spade, targets):
    f = tr_[(tr_.species == "spadenose") == is_spade]
    out, used = [], set()
    for t_ in targets:
        d_ = (f.length_cm - t_).abs()
        d_[f.record_id.isin(used)] = 99
        r = f.loc[d_.idxmin()]; used.add(r.record_id); out.append(r)
    return out
picked = (nearest(True, [31, 32.7, 34.4, 36.1, 37.8, 39.5, 41.2]) + nearest(False, [36.0]) + nearest(True, [43.5, 47, 50.5, 54])
          + nearest(False, [44, 47.5, 51, 54.5, 58, 61.5, 65, 68.5]))
picked.sort(key=lambda r: r.length_cm)
D["q19"] = [{"tag": int(r.record_id), "species": r.species, "L": float(r.length_cm)} for r in picked]

# ---- Q20 and Q21: Pima trees on glucose and BMI (training median fills the zeros)
Z = PX[:, [1, 5]].copy()
for j in range(2):
    Z[Z[:, j] == 0, j] = np.median(Z[(Z[:, j] > 0) & ~PT, j])
t2 = DecisionTreeClassifier(max_depth=2, random_state=0).fit(Z[~PT], PY[~PT])
tt = t2.tree_
leaf_of = t2.apply(Z[~PT])
leaves = [[int(((leaf_of == nd) & (PY[~PT] == 0)).sum()), int(((leaf_of == nd) & (PY[~PT] == 1)).sum())] for nd in range(tt.node_count) if tt.children_left[nd] == -1]
D["q20"] = {"thr": [float(tt.threshold[0]), float(tt.threshold[tt.children_left[0]]), float(tt.threshold[tt.children_right[0]])], "leaves": leaves,
            "womanA": [float(PX[31, 1]), float(PX[31, 5])]}
depth = []
for dep in range(1, 13):
    mt = DecisionTreeClassifier(max_depth=dep, random_state=0).fit(Z[~PT], PY[~PT])
    depth.append([dep, float(mt.score(Z[~PT], PY[~PT])), float(mt.score(Z[PT], PY[PT])), int(mt.get_n_leaves())])
D["q21"] = {"rows": depth, "lazy": float(1 - PY[PT].mean()), "n": int(PT.sum())}

# ---- Q23, Q24, Q25: boosting on the Pima data
import xgboost as xgb
curves = {}
for eta in (0.3, 0.1, 0.03):
    mx = xgb.XGBClassifier(n_estimators=200, max_depth=3, learning_rate=eta, random_state=42, eval_metric="logloss", tree_method="exact")
    mx.fit(PX[~PT], PY[~PT], eval_set=[(PX[PT], PY[PT])], verbose=False)
    curves[str(eta)] = [float(v) for v in mx.evals_result()["validation_0"]["logloss"]]
D["q23"] = curves
noise = np.random.RandomState(7).normal(size=(768, 1))
X2 = np.c_[PX, noise]; F2 = PF + ["Random numbers"]
mx = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, tree_method="exact").fit(X2[~PT], PY[~PT])
bst = mx.get_booster(); cnt = bst.get_score(importance_type="weight")
base = float((mx.predict(X2[PT]) == PY[PT]).mean()); rows = []
for j, f in enumerate(F2):
    dr = []
    for s in range(50):
        Xp = X2[PT].copy(); Xp[:, j] = np.random.RandomState(100 + s).permutation(Xp[:, j]); dr.append(base - float((mx.predict(Xp) == PY[PT]).mean()))
    rows.append({"name": f, "count": int(cnt.get("f%d" % j, 0)), "shuffle": 100 * float(np.mean(dr)), "sd": 100 * float(np.std(dr))})
D["q24"] = {"rows": rows, "acc": base}
m25 = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42, tree_method="exact").fit(PX[~PT], PY[~PT])
b25 = m25.get_booster(); dm = xgb.DMatrix(PX[31:32])
con = b25.predict(dm, pred_contribs=True)[0]; marg = float(b25.predict(dm, output_margin=True)[0])
D["q25"] = {"shares": {f: float(v) for f, v in zip(PF, con[:-1])}, "avg": float(con[-1]), "score": marg, "p": float(sig(marg)), "x": [float(v) for v in PX[31]], "features": PF}

if os.path.exists(os.path.join(HERE, "build.py")):  # only in the generator folder; the copy in the site folder just prints
    json.dump(D, open(os.path.join(HERE, "figdata.json"), "w"))
    print("written figdata.json")
for k in ("q4", "q5", "q8", "q10", "q11", "q13", "q14", "q16", "q17", "q18", "q19", "q20", "q21", "q24", "q25"):
    print(k, json.dumps(D[k])[:700])
