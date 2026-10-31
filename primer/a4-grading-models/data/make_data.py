"""Offline script for the A4 chapter data. Run with python -I from this folder (needs numpy, pandas, scikit-learn).
Writes scores.json, clinic.json, splits.json and leak.json. Everything is computed from the made-up SharkCam catch table
(not field data) and from the published Pima data (768 women, United States). Seed 4127 where a single seed is needed."""
import json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(HERE, "..", ".."))        # the primer folder
df = pd.read_csv(f"{R}/data/sharks/catch.csv")
pima = json.load(open(f"{R}/data/clinic/pima.json"))
which = sys.argv[1:] or ["scores", "clinic", "splits", "leak"]

def dump(name, obj):
    with open(os.path.join(HERE, name), "w") as f:
        json.dump(obj, f, separators=(",", ":"))
    print("wrote", name, os.path.getsize(os.path.join(HERE, name)), "bytes")

# ---------------------------------------------------------------- scores.json: hammerhead and Ganges scorers
if "scores" in which:
    X5 = np.column_stack([np.log(df.length_cm), np.log(df.weight_kg), np.log(df.girth_cm), df.depth_m, df.temp_c])
    def oof(y, cols, seed=4127):
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        return cross_val_predict(m, X5[:, cols], y, cv=cv, method="decision_function")
    yh = (df.species == "hammerhead").astype(int).values
    yg = (df.species == "ganges").astype(int).values
    zh5 = oof(yh, [0, 1, 2, 3, 4]); zh1 = oof(yh, [0]); zg5 = oof(yg, [0, 1, 2, 3, 4])
    print("hammerhead AUC", roc_auc_score(yh, zh5), "length only", roc_auc_score(yh, zh1), "ganges AUC", roc_auc_score(yg, zg5))
    r3 = lambda a: [round(float(v), 3) for v in a]
    dump("scores.json", {
        "about": "Out-of-fold log-odds from five-fold stratified cross-validation (seed 4127) of a logistic regression on the made-up SharkCam catch table (not field data). Rows follow catch.csv. The five measurements are log length, log weight, log girth, depth and temperature.",
        "ids": [int(v) for v in df.record_id], "species": [int(["spadenose", "milk", "bamboo", "hammerhead", "ganges"].index(s)) for s in df.species],
        "length": [float(v) for v in df.length_cm],
        "hammer": {"y": [int(v) for v in yh], "z5": r3(zh5), "z1": r3(zh1)},
        "ganges": {"y": [int(v) for v in yg], "z5": r3(zg5)}})

# ---------------------------------------------------------------- clinic.json: Pima logistic regression
if "clinic" in which:
    X = np.array(pima["X"]); y = np.array(pima["y"]); te = np.array(pima["test"]).astype(bool)
    Xn = X.copy()
    for j in [1, 2, 3, 4, 5]:
        Xn[Xn[:, j] == 0, j] = np.nan          # zeros mean "not measured"
    cols = [0, 1, 2, 5, 6, 7]                   # Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigree, Age
    mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
    m = mk().fit(Xn[~te][:, cols], y[~te])
    zho = m.decision_function(Xn[:, cols])
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    zcv = cross_val_predict(mk(), Xn[:, cols], y, cv=cv, method="decision_function")
    print("heldout acc", ((zho[te] >= 0) == y[te]).mean(), "auc", roc_auc_score(y[te], zho[te]), "cv auc", roc_auc_score(y, zcv))
    r3 = lambda a: [round(float(v), 3) for v in a]
    dump("clinic.json", {
        "about": "Published Pima data (768 women of Pima heritage, measured in the United States; not data from Mumbai). Logistic regression on Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction and Age, with zeros in Glucose, BloodPressure and BMI treated as not measured and filled with the training median. ho: log-odds from a model fitted on the 614 training women, for all rows (use rows with test = 1). cv: out-of-fold log-odds from five-fold stratified cross-validation (seed 42) over all 768 women.",
        "age": [int(v) for v in X[:, 7]], "y": [int(v) for v in y], "test": [int(v) for v in te],
        "ho": r3(zho), "cv": r3(zcv)})

# ---------------------------------------------------------------- splits.json: hold-out and k-fold, 24 shuffles
if "splits" in which:
    from joblib import Parallel, delayed
    feats = ["length_cm", "weight_kg", "girth_cm", "depth_m", "temp_c", "month"]
    Xs = df[feats].to_numpy(); ys = df.species.to_numpy(); n = len(df)
    classes = ["spadenose", "milk", "bamboo", "hammerhead", "ganges"]
    SEEDS = [4127] + list(range(1, 24))
    MODELS = ["lazy", "logistic", "tree3", "treeFull"]
    def mk(name, seed):
        return {"lazy": lambda: DummyClassifier(strategy="most_frequent"),
                "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
                "tree3": lambda: DecisionTreeClassifier(max_depth=3, random_state=seed),
                "treeFull": lambda: DecisionTreeClassifier(random_state=seed)}[name]()
    def stratified_order(seed):
        """Records ordered so that dealing them in turn spreads every species evenly: shuffle inside each species."""
        rs = np.random.RandomState(seed); out = []
        for c in classes:
            idx = np.where(ys == c)[0].copy(); rs.shuffle(idx); out.append(idx)
        return out
    def holdout_assign(seed, ratio, strat):
        a = np.zeros(n, dtype=int)
        f = np.array(ratio, dtype=float) / sum(ratio)
        if not strat:
            perm = np.random.RandomState(seed).permutation(n)
            n_tr = int(round(f[0] * n)); n_va = int(round(f[1] * n))
            a[perm[n_tr:n_tr + n_va]] = 1; a[perm[n_tr + n_va:]] = 2
        else:
            rs = np.random.RandomState(seed)
            for c in classes:
                idx = np.where(ys == c)[0].copy(); rs.shuffle(idx); m = len(idx)
                raw = f * m; k = np.floor(raw).astype(int); rem = raw - k
                left = m - k.sum()
                order = np.argsort(-(rem + 1e-9 * rs.random_sample(3)))      # largest remainder first, ties at random
                for j in order[:left]: k[j] += 1
                a[idx[k[0]:k[0] + k[1]]] = 1; a[idx[k[0] + k[1]:]] = 2
        return a
    def kfold_assign(seed, K, strat):
        a = np.zeros(n, dtype=int)
        if not strat:
            idx = np.arange(n); np.random.RandomState(seed).shuffle(idx)       # the same shuffle as KFold(shuffle=True, random_state=seed)
            for k, part in enumerate(np.array_split(idx, K)): a[part] = k
        else:
            rs = np.random.RandomState(seed); off = 0
            for c in classes:
                idx = np.where(ys == c)[0].copy(); rs.shuffle(idx)
                for i, r in enumerate(idx): a[r] = (off + i) % K
                off += len(idx)
        return a
    def acc(model, tr, te_, seed):
        m = mk(model, seed).fit(Xs[tr], ys[tr]); return float((m.predict(Xs[te_]) == ys[te_]).mean())
    def job_hold(seed, ri, strat):
        ratio = [(50, 25, 25), (60, 20, 20), (80, 10, 10)][ri]
        a = holdout_assign(seed, ratio, strat)
        tr, va, te_ = (np.where(a == k)[0] for k in range(3))
        res = {m: [round(acc(m, tr, tr, seed), 4), round(acc(m, tr, va, seed), 4), round(acc(m, tr, te_, seed), 4)] for m in MODELS}
        ga = [int(((ys == "ganges") & (a == k)).sum()) for k in range(3)]
        ha = [int(((ys == "hammerhead") & (a == k)).sum()) for k in range(3)]
        return {"a": "".join(map(str, a)), "acc": res, "ganges": ga, "hammer": ha}
    def job_kf(seed, K, strat):
        a = kfold_assign(seed, K, strat)
        res = {m: [round(acc(m, np.where(a != k)[0], np.where(a == k)[0], seed), 4) for k in range(K)] for m in MODELS}
        return {"a": "".join(chr(48 + v) if v < 10 else chr(55 + v) for v in a), "acc": res}
    out = {"about": "Computed offline from the made-up SharkCam catch table (not field data). Task: species from length, weight, girth, depth, temperature and month, with the models of the t8 results table. a is one character per record in catch.csv order: hold-out 0 training, 1 validation, 2 test; k-fold the fold number (0 to 9, characters 0-9). Plain shuffles use RandomState(seed).",
           "seeds": SEEDS, "models": MODELS, "ratios": ["50/25/25", "60/20/20", "80/10/10"], "holdout": {}, "kfold": {}}
    for strat in (False, True):
        key = "strat" if strat else "plain"
        res = Parallel(n_jobs=6)(delayed(job_hold)(s, ri, strat) for ri in range(3) for s in SEEDS)
        out["holdout"][key] = [res[ri * len(SEEDS):(ri + 1) * len(SEEDS)] for ri in range(3)]
        for K in (5, 10):
            res = Parallel(n_jobs=6)(delayed(job_kf)(s, K, strat) for s in SEEDS)
            out["kfold"].setdefault(key, {})[str(K)] = res
    # checks against the t8 results table (seed 4127, plain 5-fold): logistic 0.7783, tree3 0.8017, treeFull 0.705
    k = out["kfold"]["plain"]["5"][0]["acc"]
    print("5-fold seed 4127 plain: ", {m: round(float(np.mean(v)), 4) for m, v in k.items()})
    dump("splits.json", out)

# ---------------------------------------------------------------- leak.json: three leaks and their fair twins
if "leak" in which:
    from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit
    sp = df[df.species == "spadenose"].reset_index(drop=True)             # 268 sharks, the maturity task of A3
    base = np.column_stack([sp.length_cm, sp.girth_cm, sp.depth_m, sp.temp_c, sp.month]); ym = sp.mature.values
    FRAMES, REPS = 4, 200
    def logreg(scale=True):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)) if scale else LogisticRegression(max_iter=2000)
    out = {"about": "Computed offline on the made-up SharkCam catch table (not field data). Mean test accuracy over 200 random splits, with the standard deviation across splits. Each of the 268 spadenose sharks gets 4 made-up frames: length, girth, depth and temperature re-measured with noise (SD 0.4 cm, 0.25 cm, 0.3 m and 0.05 C; month unchanged). Task: mature or not from length, girth, depth, temperature and month.", "reps": REPS}
    # frame table
    rng = np.random.RandomState(4127)
    Xf = np.vstack([base + rng.normal(0, [0.4, 0.25, 0.3, 0.05, 0.0], base.shape) for _ in range(FRAMES)])
    yf = np.tile(ym, FRAMES); gf = np.tile(np.arange(len(ym)), FRAMES)
    # 1. scaling: fit the scaler on every row (leak) or on training rows only (fair); one frame per shark
    res = {}
    for name, mkm in [("logistic", lambda: LogisticRegression(max_iter=2000)), ("tree", lambda: DecisionTreeClassifier(random_state=0))]:
        leaky, fair = [], []
        for r in range(REPS):
            tr, te_ = next(ShuffleSplit(1, test_size=0.2, random_state=r).split(base))
            sc_all = StandardScaler().fit(base); sc_tr = StandardScaler().fit(base[tr])
            leaky.append((mkm().fit(sc_all.transform(base[tr]), ym[tr]).predict(sc_all.transform(base[te_])) == ym[te_]).mean())
            fair.append((mkm().fit(sc_tr.transform(base[tr]), ym[tr]).predict(sc_tr.transform(base[te_])) == ym[te_]).mean())
        res[name] = {"leaky": [float(np.mean(leaky)), float(np.std(leaky))], "fair": [float(np.mean(fair)), float(np.std(fair))], "diff": float(np.mean(np.array(leaky) - np.array(fair)))}
    out["scaling"] = res
    # 2. same shark in both sets: split frames at random (leak) or split sharks (fair)
    res = {}
    for name, mkm in [("logistic", lambda: logreg()), ("tree", lambda: DecisionTreeClassifier(random_state=0))]:
        leaky, fair = [], []
        for r in range(REPS):
            tr, te_ = next(ShuffleSplit(1, test_size=0.2, random_state=r).split(Xf))
            leaky.append((mkm().fit(Xf[tr], yf[tr]).predict(Xf[te_]) == yf[te_]).mean())
            tr, te_ = next(GroupShuffleSplit(1, test_size=0.2, random_state=r).split(Xf, yf, gf))
            fair.append((mkm().fit(Xf[tr], yf[tr]).predict(Xf[te_]) == yf[te_]).mean())
        res[name] = {"leaky": [float(np.mean(leaky)), float(np.std(leaky))], "fair": [float(np.mean(fair)), float(np.std(fair))]}
    out["duplicates"] = res
    # 3. peeking at labels: 80 sharks with coin-flip labels and 200 filler readings; keep the 10 readings that best match the labels
    from sklearn.feature_selection import SelectKBest, f_classif
    res = {"leaky": [], "fair": []}
    for r in range(REPS):
        rs = np.random.RandomState(1000 + r)
        Xn = rs.normal(size=(80, 200)); yn = rs.randint(0, 2, 80)
        kf = StratifiedKFold(5, shuffle=True, random_state=r)
        Xsel = SelectKBest(f_classif, k=10).fit_transform(Xn, yn)                 # chosen with every label
        res["leaky"].append(float(cross_val_predict(logreg(), Xsel, yn, cv=kf).__eq__(yn).mean()))
        pipe = make_pipeline(SelectKBest(f_classif, k=10), StandardScaler(), LogisticRegression(max_iter=2000))   # chosen inside each training fold
        res["fair"].append(float(cross_val_predict(pipe, Xn, yn, cv=kf).__eq__(yn).mean()))
    out["peeking"] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in res.items()}
    # honest numbers for the text: logistic regression and tree on the one-frame-per-shark table, fair 5-fold CV
    print(json.dumps(out, indent=1))
    dump("leak.json", out)

# ---------------------------------------------------------------- perm.json: shuffles of the 268 spadenose records for the polynomial figure
if "perm" in which:
    nsp = int((df.species == "spadenose").sum())
    dump("perm.json", {"about": "RandomState(seed).permutation(268) over the spadenose rows of catch.csv, in file order. The polynomial figure uses the first n as the training sample and the rest as test records.",
                       "n": nsp, "seeds": {str(s): [int(v) for v in np.random.RandomState(s).permutation(nsp)] for s in range(1, 13)}})
