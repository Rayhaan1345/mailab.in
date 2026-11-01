"""Reproduces the numbers behind questions 17, 18 and 19 of the practice set from the made-up catch table.
Run from this folder with:  python -I make_tables.py   (needs pandas and scikit-learn)."""
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
df = pd.read_csv("../../data/sharks/catch.csv")  # made-up records, not field data

# Question 17: records per month. October has 86 and November has 88.
print("records per month:", df.groupby("month").size().to_dict())
print("lie factor of bars drawn from an axis starting at 84:", ((4 - 2) / 2) / ((88 - 86) / 86))

# Question 18: depth-2 tree on length, three species, five-fold cross-validated predictions (shuffled folds, seed 4127).
three = df[df.species.isin(["spadenose", "milk", "bamboo"])]
kf = KFold(5, shuffle=True, random_state=4127)
pred = cross_val_predict(DecisionTreeClassifier(max_depth=2, random_state=0), three[["length_cm"]].values, three.species.values, cv=kf)
cm = confusion_matrix(three.species.values, pred, labels=["spadenose", "milk", "bamboo"])
print(cm.tolist(), "precision of spadenose:", round(cm[0, 0] / cm[:, 0].sum(), 4))

# Question 19: is a spadenose shark mature? Models on length, five-fold cross-validation repeated for seeds 1 to 5.
sp = df[df.species == "spadenose"]
X, y = sp[["length_cm"]].values, sp.mature.values
models = {"Always says mature": DummyClassifier(strategy="most_frequent"), "Logistic regression": LogisticRegression(max_iter=5000),
          "Tree, depth 1": DecisionTreeClassifier(max_depth=1, random_state=0), "Tree, depth 3": DecisionTreeClassifier(max_depth=3, random_state=0),
          "Tree, no depth limit": DecisionTreeClassifier(random_state=0)}
for name, m in models.items():
    test, train = [], []
    for seed in (1, 2, 3, 4, 5):
        r = cross_validate(m, X, y, cv=KFold(5, shuffle=True, random_state=seed), return_train_score=True)
        test.append(r["test_score"].mean()); train.append(r["train_score"].mean())
    print(f"{name:22s} training {100 * np.mean(train):.1f}  new mean {100 * np.mean(test):.1f}  new SD {100 * np.std(test, ddof=1):.1f}")
