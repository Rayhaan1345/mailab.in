"""Offline work for A6: trains XGBoost on the made-up SharkCam catch table (not field data) and exports the trees,
importances and SHAP values to xgb6.json. Label: spadenose or not. Training sharks: tags not ending in 0, 1, 2.
Run from anywhere with: python -I make_xgb6.py   (needs numpy, pandas, xgboost)"""
import json, numpy as np, pandas as pd, xgboost as xgb
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "..", "..") + os.sep      # the primer folder
OUT = HERE + os.sep
df = pd.read_csv(R + "data/sharks/catch.csv")
df["y"] = (df.species == "spadenose").astype(int)
df["held"] = (df.record_id % 10 < 3)
F = ["length_cm", "weight_kg", "girth_cm", "depth_m", "temp_c", "month"]
tr, te = df[~df.held], df[df.held]
P = dict(n_estimators=100, max_depth=3, learning_rate=0.1, tree_method="exact", random_state=0)
m = xgb.XGBClassifier(**P).fit(tr[F], tr.y)
bst = m.get_booster()
cfg = json.loads(bst.save_config())
base_score = float(str(cfg["learner"]["learner_model_param"]["base_score"]).strip("[]"))
base_margin = float(np.log(base_score / (1 - base_score)))
print("base_score", base_score, "margin", base_margin)
td = bst.trees_to_dataframe()
trees = []
for t, g in td.groupby("Tree"):
    g = g.reset_index(drop=True); idx = {nid: i for i, nid in enumerate(g.ID)}
    f, th, l, r, v, c = [], [], [], [], [], []
    for _, row in g.iterrows():
        leaf = row.Feature == "Leaf"
        f.append(-1 if leaf else F.index(row.Feature)); th.append(0 if leaf else float(np.float32(row.Split)))
        l.append(-1 if leaf else idx[row.Yes]); r.append(-1 if leaf else idx[row.No])
        v.append(round(float(row.Gain), 6) if leaf else 0); c.append(round(float(row.Cover), 4))
    trees.append(dict(f=f, t=th, l=l, r=r, v=v, c=c))
def margin_py(X):
    return bst.predict(xgb.DMatrix(X[F]), output_margin=True)
# importances
imp_w = bst.get_score(importance_type="weight"); imp_g = bst.get_score(importance_type="gain"); imp_tg = bst.get_score(importance_type="total_gain")
rng = np.random.default_rng(4127)
base_acc = m.score(te[F], te.y)
shuf = {}
for f in F:
    drops = []
    for _ in range(50):
        b = te[F].copy(); b[f] = rng.permutation(b[f].values); drops.append(base_acc - m.score(b, te.y))
    shuf[f] = [float(np.mean(drops)), float(np.std(drops))]
sh_te = bst.predict(xgb.DMatrix(te[F]), pred_contribs=True)
mean_abs = np.abs(sh_te[:, :-1]).mean(0)
imp = {f: dict(count=int(imp_w.get(f, 0)), gain=float(imp_g.get(f, 0)), total_gain=float(imp_tg.get(f, 0)), shuffle=shuf[f][0], shuffle_sd=shuf[f][1], shap=float(mean_abs[i])) for i, f in enumerate(F)}
# rows: the three named tags and all held-out sharks
named = {3305: "Tag 3305", 4127: "Tag 4127", 5519: "Tag 5519"}
def rowrec(r):
    return dict(id=int(r.record_id), x=[float(r[f]) for f in F], y=int(r.y), species=r.species)
rows = [rowrec(r) for _, r in df[df.record_id.isin(named)].iterrows()] + [rowrec(r) for _, r in te.iterrows()]
for r in rows: r["held"] = bool(r["id"] % 10 < 3)
sh_rows = bst.predict(xgb.DMatrix(pd.DataFrame([r["x"] for r in rows], columns=F)), pred_contribs=True)
mg_rows = bst.predict(xgb.DMatrix(pd.DataFrame([r["x"] for r in rows], columns=F)), output_margin=True)
for r, s, mg in zip(rows, sh_rows, mg_rows):
    r["shap"] = [round(float(v), 5) for v in s[:-1]]; r["bias"] = round(float(s[-1]), 5); r["margin"] = round(float(mg), 5)
# the model's own average of the training margins is the SHAP base value
out = dict(about="XGBoost trained offline on the made-up SharkCam catch table (not field data). Label: spadenose or not. Training sharks: tags not ending 0, 1, 2.",
           features=F, params=dict(n_estimators=100, max_depth=3, learning_rate=0.1), base_margin=base_margin, trees=trees,
           importance=imp, held_accuracy=base_acc, train_accuracy=float(m.score(tr[F], tr.y)), n_train=int(len(tr)), n_held=int(len(te)), rows=rows)
json.dump(out, open(OUT + "xgb6.json", "w"), separators=(",", ":"))
print("held acc", base_acc, "train acc", out["train_accuracy"], "size", os.path.getsize(OUT + "xgb6.json"))
for f in F: print(f, imp[f])
# check: margins from the exported trees
def walk(tree, x):
    i = 0
    while tree["f"][i] >= 0: i = tree["l"][i] if np.float32(x[tree["f"][i]]) < tree["t"][i] else tree["r"][i]
    return tree["v"][i]
mx = max(abs(base_margin + sum(walk(t, r["x"]) for t in trees) - r["margin"]) for r in rows); print("max margin diff", mx)
