"""Offline script for results.json: cross-validated models on the made-up catch table. Run with python -I from this folder (needs scikit-learn)."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import os
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))   # the primer folder
df = pd.read_csv(f"{R}/data/sharks/catch.csv")
feats = ["length_cm", "weight_kg", "girth_cm", "depth_m", "temp_c", "month"]
X = df[feats].to_numpy(); y = df.species.to_numpy()
classes = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"]
SEEDS = [4127, 1, 2, 3, 4]
def mk(name, seed):
    return {"lazy": lambda: DummyClassifier(strategy="most_frequent"),
            "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
            "tree3": lambda: DecisionTreeClassifier(max_depth=3, random_state=seed),
            "treeFull": lambda: DecisionTreeClassifier(random_state=seed),
            "boost": lambda: GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=seed)}[name]()
names = {"lazy": "Always spadenose", "logistic": "Logistic regression", "tree3": "Decision tree, depth 3", "treeFull": "Decision tree, no depth limit", "boost": "Boosting, 100 trees of depth 3"}
out = {"about": "Computed offline from the made-up SharkCam catch table (not field data). Five-fold cross-validation with shuffled folds, one run per seed; seed 4127 matches reference_models in summary.json.", "classes": classes, "counts": [int((y == c).sum()) for c in classes], "seeds": SEEDS, "models": [], "confusion": {}}
for key in ["lazy", "logistic", "tree3", "treeFull", "boost"]:
    train = float((mk(key, 4127).fit(X, y).predict(X) == y).mean())
    cv, ganges = [], []
    for sd in SEEDS:
        kf = KFold(n_splits=5, shuffle=True, random_state=sd)
        pred = cross_val_predict(mk(key, sd), X, y, cv=kf)
        cv.append(round(float((pred == y).mean()), 4))
        ganges.append(round(float(((pred == "ganges") & (y == "ganges")).sum()), 0))
        if sd == 4127:
            cm = [[int(((y == a) & (pred == b)).sum()) for b in classes] for a in classes]
            out["confusion"][key] = cm
    out["models"].append({"id": key, "name": names[key], "train": round(train, 4), "cv": cv, "ganges_found": ganges})
    print(key, "train", round(train, 3), "cv", cv, "ganges found", ganges)
json.dump(out, open(os.path.join(HERE, "results.json"), "w"))
print(np.array(out["confusion"]["tree3"]))
print(np.array(out["confusion"]["boost"]))
