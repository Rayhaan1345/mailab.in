"""Offline step for F1, clinic part. Run with python -I from this folder (numpy, scikit-learn).
Reads the published Pima data (768 women of Pima heritage, measured in the United States; not data from Mumbai) and writes clinic.json:
the columns the figures need, the out-of-fold log-odds of the A4 model (same recipe, same seed 42: it must equal a4-grading-models/data/clinic.json),
the same model without the Age column, and the coefficients of the model fitted on all 768 women (for the visitor figure)."""
import json, os
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.abspath(os.path.join(HERE, "..", ".."))
pima = json.load(open(f"{R}/data/clinic/pima.json")); a4 = json.load(open(f"{R}/a4-grading-models/data/clinic.json"))
X = np.array(pima["X"], float); y = np.array(pima["y"]); Xn = X.copy()
for j in [1, 2, 3, 4, 5]: Xn[Xn[:, j] == 0, j] = np.nan          # zeros mean "not measured"
mk = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=2000))
cv = StratifiedKFold(5, shuffle=True, random_state=42)
six = [0, 1, 2, 5, 6, 7]; five = [0, 1, 2, 5, 6]                  # Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigree, (Age)
z6 = cross_val_predict(mk(), Xn[:, six], y, cv=cv, method="decision_function")
z5 = cross_val_predict(mk(), Xn[:, five], y, cv=cv, method="decision_function")
assert np.abs(np.round(z6, 3) - np.array(a4["cv"])).max() == 0, "does not match A4"
m = mk().fit(Xn[:, six], y)
imp, sc, lr = m.steps[0][1], m.steps[1][1], m.steps[2][1]
r = lambda a, d=3: [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), d) for v in a]
out = {
 "about": "Published Pima data (768 women of Pima heritage, measured in the United States; not data from Mumbai). Zeros in Glucose, BloodPressure and BMI mean not measured and are stored as null. zcv: out-of-fold log-odds of a logistic regression on Pregnancies, Glucose, BloodPressure, BMI, DiabetesPedigreeFunction and Age (five-fold stratified, seed 42, identical to the A4 chapter). zna: the same without Age. model: the same regression fitted on all 768 women.",
 "age": [int(v) for v in X[:, 7]], "preg": [int(v) for v in X[:, 0]],
 "glucose": r(Xn[:, 1], 0), "bp": r(Xn[:, 2], 0), "bmi": r(Xn[:, 5], 1), "dpf": r(X[:, 6], 3),
 "y": [int(v) for v in y], "zcv": r(z6), "zna": r(z5),
 "model": {"features": ["Pregnancies", "Glucose", "BloodPressure", "BMI", "DiabetesPedigreeFunction", "Age"],
           "median": r(imp.statistics_, 3), "mean": r(sc.mean_, 4), "std": r(sc.scale_, 4), "coef": r(lr.coef_[0], 4), "intercept": round(float(lr.intercept_[0]), 4)}}
json.dump(out, open(f"{HERE}/clinic.json", "w"), separators=(",", ":"))
print("wrote clinic.json", os.path.getsize(f"{HERE}/clinic.json"), "bytes"); print(out["model"])
age = X[:, 7]
for name, mask in [("under 30", age < 30), ("30 and over", age >= 30)]:
    print(name, mask.sum(), "base rate", round(y[mask].mean(), 3))
