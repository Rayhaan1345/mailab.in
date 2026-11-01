#!/usr/bin/env python3
"""
make_catch.py  -  builds the SharkCam Mumbai catch table.

*** MADE-UP RECORDS. They are generated from published length-weight
*** relationships and maturity lengths. They are NOT field data, and no animal
*** in this table was measured. Any page that shows them must say so.

What it does
------------
- Reads species.json (length-weight laws, maturity lengths, sources).
- Draws 600 sharks: species, month, length, sex, depth, water
   temperature and source ("landing" at Sassoon Dock, or "dive").
- Computes weight from the species' length-weight law with multiplicative
   noise, girth from weight with a simple body model, and maturity from a
   logistic curve centred on the sex-specific length at 50% maturity (L50).
- Pins three named animals that chapters reuse (Tag 4127 and two others).
- Writes catch.csv, catch.json and summary.json next to this script.

Run it with:   python -I make_catch.py
Needs numpy and pandas. scikit-learn is optional (reference models in summary.json).
The seed is fixed, so the output is the same on all runs.

Columns of catch.csv / catch.json
---------------------------------
record_id   4-digit tag number written on the animal's tag (unique)
species     species id: spadenose, milk, bamboo, hammerhead, ganges
month       month of year, 1 to 12
length_cm   total length in cm
weight_kg   weight in kg
girth_cm    girth just in front of the first dorsal fin, in cm
sex         F or M
mature      1 if mature, 0 if immature
depth_m     depth of capture (landing) or depth of the animal (dive), in metres
temp_c      water temperature in degrees Celsius
source      "landing" (measured at Sassoon Dock) or "dive" (measured in the
            water or in a pool by researchers, then released)
site        where: Sassoon Dock for landings; Juhu, Versova or Uran for dives
            (this column is an extra, not in the first specification)

Assumptions (all made up unless a source is named)
--------------------------------------------------
* Monsoon closure. Fishing is closed in Maharashtra from 1 June to 31 July
  (61 days) under the national uniform ban for the west coast: Order No.
  j-2103035/1/2026-Fy (E-27594), 16 March 2026, Department of Fisheries, Ministry
  of Animal Husbandry and Dairying, quoted in the Puducherry Gazette of 10 April
  2026. Maharashtra extended the ban for mechanised and motorised boats to 15
  August 2026 (Deccan Chronicle, 29 June 2026). So the table has NO landings in
  June and July, and few in August. Traditional non-motorised boats are exempt,
  which is why a few August landings remain. Dives are not scheduled from June
  to September because the monsoon sea is rough (an operating assumption).
* Water temperature. Monthly mean sea surface temperature off Mumbai comes from
  NOAA daily OISST v2.1 (Reynolds et al. (2007), doi:10.1175/2007JCLI1824.1;
  Huang et al. (2021), doi:10.1175/JCLI-D-20-0166.1), grid cell 18.875 N,
  72.625 E, 2019 to 2023 (1,812 daily values). Daily values ranged from 24.1 to
  31.2 C and the day-to-day scatter around the monthly mean was 0.7 C.
  A mild fall with depth (0.03 C per metre) and measurement noise are added.
  Water in very shallow pools is warmed by 1 C.
* Length distributions are truncated normal curves chosen so that the size range
  of the catch matches the studies in species.json and so that both mature and
  immature animals appear. Some lower limits are wider than the published
  samples (noted below) because otherwise a species would have almost no juveniles.
* Weight noise. W = a * L^b * exp(e), e ~ Normal(0, 0.14). Published fits have
  r-squared between 0.82 and 0.999, so 0.14 is a middle value.
* Girth. A shark is treated as a tapered cylinder with density 1.05 g per cubic
  cm and mean cross-section equal to 0.5 of the largest cross-section.
  Then girth = sqrt(4 * pi * W / (density * 0.5 * L)), times a 4% reading error.
  This is a modelling assumption with no published source. It gives girth near
  0.28 to 0.34 of body length for the small species, which looks plausible for a slim shark,
  and about 0.45 for the Ganges shark, whose weight comes from a stout bull shark law. Because
  girth is built from weight, it adds little to a maturity model once length is known.
* Maturity. P(mature) = 1 / (1 + exp(-(L - L50) / s)) with s = 0.05 * L50,
  using the sex-specific L50 from species.json. The slope s is an assumption.
* Sex ratio. Milk shark 53% female (male to female 1 : 1.13, Dash et al. (2018)).
  Hammerhead 56.5% female (F:M 1.3 : 1, Thomas et al. (2021)). Others 50%.
"""

import json
import math
import os
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 4127
rng = np.random.default_rng(SEED)

with open(os.path.join(HERE, "species.json"), encoding="utf8") as f:
    SPECIES = {s["id"]: s for s in json.load(f)["species"]}

# ---------------------------------------------------------------- settings --

# Number of records per species (total 600). Ganges shark is kept very rare on purpose.
COUNTS = {"spadenose": 268, "milk": 200, "bamboo": 72, "hammerhead": 57, "ganges": 3}

# Share of a species logged on dives (the rest are landings at Sassoon Dock).
DIVE_SHARE = {"spadenose": 0.03, "milk": 0.03, "bamboo": 0.45, "hammerhead": 0.14, "ganges": 0.0}

# Share of females.
FEMALE_SHARE = {"spadenose": 0.50, "milk": 0.53, "bamboo": 0.50, "hammerhead": 0.565, "ganges": 0.50}

# Relative number of landings in a month (index 0 = January).
# June and July are zero because of the monsoon fishing closure (see header).
# August is low because the ban on mechanised boats ran to 15 August.
# The other values are made up: busiest just after the season reopens.
LANDING_MONTH_WEIGHT = [1.0, 1.0, 0.9, 0.8, 0.6, 0.0, 0.0, 0.3, 1.1, 1.2, 1.2, 1.1]
# Dives run from October to May.
DIVE_MONTHS = [10, 11, 12, 1, 2, 3, 4, 5]

# Monthly mean sea surface temperature off Mumbai, deg C (NOAA OISST v2.1, 2019 to 2023).
SST = [25.9, 25.8, 27.3, 29.3, 30.2, 30.1, 29.4, 28.5, 28.9, 29.2, 28.2, 27.6]

# Length distribution per species and source: (mean, sd, lowest, highest) in cm.
# Published landing ranges are in species.json. Where a lower limit is wider than the
# published sample, the reason is given.
LENGTH = {
    # Gladston et al. (2018): 18 to 61 cm. Gujarat mean 40.5 cm (Dash et al. (2018)).
    "spadenose": {"landing": (40.0, 9.0, 18.0, 61.0), "dive": (38.0, 9.0, 18.0, 61.0)},
    # Gladston et al. (2018): 55 to 97 cm. Lower limit widened to 45 cm because L50 is 61 cm.
    "milk": {"landing": (67.0, 9.0, 45.0, 97.0), "dive": (62.0, 9.0, 45.0, 97.0)},
    # Eswaran et al. (2023): 32.0 to 62.6 cm. Dive animals include smaller juveniles (down to 25 cm).
    "bamboo": {"landing": (51.0, 7.0, 32.0, 66.0), "dive": (40.0, 8.0, 25.0, 62.0)},
    # Hammerheads use a two-part mixture, see draw_length().
    "hammerhead": {"landing": None, "dive": None},
    "ganges": {"landing": None, "dive": None},
}

# Depth in metres: (mean, sd, lowest, highest) for landings (depth of capture, as told by the skipper).
LANDING_DEPTH = {
    "spadenose": (22.0, 10.0, 5.0, 60.0),   # IUCN: 10 to 75 m, usually less than 50 m
    "milk": (28.0, 12.0, 5.0, 80.0),        # FishBase: 1 to 200 m
    "bamboo": (20.0, 10.0, 3.0, 50.0),      # FishBase: 5 to 100 m
    "hammerhead": (18.0, 8.0, 4.0, 45.0),   # FishBase: usually 0 to 25 m
    "ganges": (10.0, 5.0, 3.0, 25.0),       # no published depth, shallow estuarine water assumed
}
# Dive sites: uniform depth range in metres, and how likely a site is for a species.
DIVE_SITE_DEPTH = {"Juhu": (0.3, 4.0), "Versova": (3.0, 10.0), "Uran": (5.0, 18.0)}
DIVE_SITE_WEIGHT = {
    "spadenose": {"Versova": 0.5, "Uran": 0.5},
    "milk": {"Versova": 0.5, "Uran": 0.5},
    "bamboo": {"Juhu": 0.5, "Versova": 0.25, "Uran": 0.25},
    "hammerhead": {"Versova": 0.5, "Uran": 0.5},
    "ganges": {"Versova": 1.0},
}

WEIGHT_NOISE_SD = 0.14      # sd of ln(weight) around the law
GIRTH_NOISE_SD = 0.04       # reading error of the girth tape
DENSITY = 1.05              # g per cubic cm
TAPER = 0.5                 # mean cross-section / largest cross-section
SLOPE_FRACTION = 0.05       # maturity curve slope s = 0.05 * L50
DEPTH_TEMP_DROP = 0.03      # deg C lost per metre of depth
TEMP_NOISE_SD = 0.4         # deg C
SHALLOW_WARMING = 1.0       # extra deg C in water shallower than 1.5 m

# Animals that chapters reuse. eps = 0 puts the weight on the law's curve with no noise.
# (record_id, species, sex, length_cm, month, source, site, depth_m, mature)
PINNED = [
    # Tag 4127: a juvenile grey bamboo shark found in a Juhu rock pool at low tide.
    dict(record_id=4127, species="bamboo", sex="F", length_cm=44.0, month=12, source="dive",
         site="Juhu", depth_m=0.4, mature=0),
    # Tag 3305: an adult spadenose shark from a Sassoon Dock landing.
    dict(record_id=3305, species="spadenose", sex="F", length_cm=41.0, month=1, source="landing",
         site="Sassoon Dock", depth_m=18.0, mature=1),
    # Tag 5519: a young scalloped hammerhead from a Sassoon Dock landing.
    dict(record_id=5519, species="hammerhead", sex="F", length_cm=82.0, month=11, source="landing",
         site="Sassoon Dock", depth_m=14.0, mature=0),
]
# The three made-up Ganges shark records (set by hand so the rare class is stable).
# They are NOT the 2016 specimen recorded at Sassoon Docks, which was a 266 cm female.
GANGES = [
    dict(sex="M", length_cm=121.0, month=10, depth_m=9.0, mature=0),
    dict(sex="F", length_cm=156.5, month=2, depth_m=6.0, mature=0),
    dict(sex="F", length_cm=191.0, month=11, depth_m=12.0, mature=1),
]

# ----------------------------------------------------------------- helpers --


def trunc_normal(mean, sd, lo, hi):
    """Draw from a normal curve, redrawing until the value is inside [lo, hi]."""
    while True:
        x = rng.normal(mean, sd)
        if lo <= x <= hi:
            return x


def draw_length(sp, source):
    if sp == "hammerhead":
        # About 81% are young animals of 38 to 100 cm (Thomas et al. (2021): 80% of landed
        # hammerheads measured 40 to 100 cm). The rest are larger animals, 100 cm and up.
        if rng.random() < 0.81:
            return trunc_normal(68.0, 17.0, 38.0, 100.0)
        while True:
            x = 100.0 + rng.exponential(45.0)
            if x <= 260.0:
                return x
    if sp == "ganges":
        raise ValueError("Ganges records are pinned")
    mean, sd, lo, hi = LENGTH[sp][source]
    return trunc_normal(mean, sd, lo, hi)


def law_weight_g(sp, sex, length_cm):
    """Weight in grams from the species' length-weight law (species.json)."""
    lw = SPECIES[sp]["length_weight"]
    ab = lw["by_sex"]["female" if sex == "F" else "male"] if lw.get("by_sex") else lw
    return ab["a"] * length_cm ** ab["b"]


def l50(sp, sex):
    by_sex = SPECIES[sp].get("l50_by_sex_cm")
    if by_sex:
        return by_sex["female" if sex == "F" else "male"]
    return SPECIES[sp]["l50_cm"]


def p_mature(sp, sex, length_cm):
    L50 = l50(sp, sex)
    s = SLOPE_FRACTION * L50
    return 1.0 / (1.0 + math.exp(-(length_cm - L50) / s))


def girth_cm(weight_g, length_cm, noise):
    """Tapered-cylinder body model, see header."""
    return math.sqrt(4.0 * math.pi * weight_g / (DENSITY * TAPER * length_cm)) * noise


def water_temp(month, depth_m, noise):
    t = SST[month - 1] - DEPTH_TEMP_DROP * depth_m + noise
    if depth_m < 1.5:
        t += SHALLOW_WARMING
    return t


def choice(options, weights):
    w = np.array(weights, dtype=float)
    return options[int(rng.choice(len(options), p=w / w.sum()))]


def make_record(sp, source, length_cm=None, sex=None, month=None, site=None, depth_m=None,
                mature=None, exact=False):
    """Build one record. exact=True removes the random noise (used for pinned animals)."""
    sex = sex or ("F" if rng.random() < FEMALE_SHARE[sp] else "M")
    length = length_cm if length_cm is not None else draw_length(sp, source)
    if month is None:
        if source == "landing":
            month = int(choice(list(range(1, 13)), LANDING_MONTH_WEIGHT))
        else:
            month = int(rng.choice(DIVE_MONTHS))
    if source == "landing":
        site = site or "Sassoon Dock"
        depth = depth_m
        if depth is None:
            m, sd, lo, hi = LANDING_DEPTH[sp]
            depth = trunc_normal(m, sd, lo, hi)
    else:
        if site is None:
            sites = DIVE_SITE_WEIGHT[sp]
            site = choice(list(sites), list(sites.values()))
        depth = depth_m
        if depth is None:
            lo, hi = DIVE_SITE_DEPTH[site]
            depth = rng.uniform(lo, hi)
    eps = 0.0 if exact else rng.normal(0.0, WEIGHT_NOISE_SD)
    weight_g = law_weight_g(sp, sex, length) * math.exp(eps)
    g_noise = 1.0 if exact else math.exp(rng.normal(0.0, GIRTH_NOISE_SD))
    girth = girth_cm(weight_g, length, g_noise)
    if mature is None:
        mature = int(rng.random() < p_mature(sp, sex, length))
    t_noise = 0.0 if exact else rng.normal(0.0, TEMP_NOISE_SD)
    temp = water_temp(month, depth, t_noise)
    return dict(species=sp, month=month, length_cm=round(length, 1), weight_kg=round(weight_g / 1000.0, 3),
                girth_cm=round(girth, 1), sex=sex, mature=int(mature), depth_m=round(depth, 1),
                temp_c=round(temp, 1), source=source, site=site)


# ----------------------------------------------------------------- build ----


def build():
    rows = []
    # Pinned animals first, so the random draws below fill the rest of each species' count.
    pinned_per_species = {}
    for p in PINNED:
        p = dict(p)
        rid = p.pop("record_id")
        sp = p.pop("species")
        rec = make_record(sp, p.pop("source"), exact=True, **p)
        rec["record_id"] = rid
        rows.append(rec)
        pinned_per_species[sp] = pinned_per_species.get(sp, 0) + 1

    for g in GANGES:
        rec = make_record("ganges", "landing", **g)
        rows.append(rec)
        pinned_per_species["ganges"] = pinned_per_species.get("ganges", 0) + 1

    for sp, n in COUNTS.items():
        for _ in range(n - pinned_per_species.get(sp, 0)):
            source = "dive" if rng.random() < DIVE_SHARE[sp] else "landing"
            rows.append(make_record(sp, source))
    # Exact dive count per species is not fixed; counts of species are.

    # Unique 4-digit tag numbers for the random rows, then sort everything by tag.
    taken = {r["record_id"] for r in rows if "record_id" in r}
    pool = [i for i in range(1001, 9999) if i not in taken]
    need = sum(1 for r in rows if "record_id" not in r)
    tags = sorted(rng.choice(pool, size=need, replace=False).tolist())
    # Shuffle which row gets which tag so tags are unrelated to species.
    order = rng.permutation(need)
    k = 0
    for r in rows:
        if "record_id" not in r:
            r["record_id"] = int(tags[order[k]])
            k += 1
    rows.sort(key=lambda r: r["record_id"])
    cols = ["record_id", "species", "month", "length_cm", "weight_kg", "girth_cm", "sex", "mature",
            "depth_m", "temp_c", "source", "site"]
    return pd.DataFrame(rows)[cols]


def checks(df):
    assert len(df) == sum(COUNTS.values()) == 600
    assert df.record_id.is_unique
    assert df.groupby("species").size().to_dict() == COUNTS
    land = df[df.source == "landing"]
    assert not land.month.isin([6, 7]).any(), "landings found in the closed season"
    assert (df.species == "ganges").sum() == 3


def fit_logistic_l50(sub):
    """L50 from a logistic fit of mature on length (needs both classes, enough of each)."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None
    if sub.mature.sum() < 10 or (1 - sub.mature).sum() < 10:
        return None
    X = sub[["length_cm"]].to_numpy()
    m = LogisticRegression(C=1e6, max_iter=1000).fit(X, sub.mature.to_numpy())
    return round(float(-m.intercept_[0] / m.coef_[0][0]), 1)


def summarise(df):
    n = len(df)
    out = {
        "about": "Recurring numbers for the SharkCam Mumbai catch table. The records are made-up, "
                 "generated from published length-weight relationships; not field data. "
                 "Numbers here are computed from catch.csv by make_catch.py.",
        "seed": SEED,
        "n_records": n,
        "columns": list(df.columns),
    }
    per = {}
    for sp, g in df.groupby("species"):
        ll = np.log(g.length_cm.to_numpy())
        lw = np.log(g.weight_kg.to_numpy() * 1000.0)
        entry = {
            "n": int(len(g)),
            "share": round(len(g) / n, 4),
            "n_landing": int((g.source == "landing").sum()),
            "n_dive": int((g.source == "dive").sum()),
            "length_cm": {"mean": round(g.length_cm.mean(), 1), "min": float(g.length_cm.min()),
                          "max": float(g.length_cm.max())},
            "weight_kg": {"mean": round(g.weight_kg.mean(), 3), "min": float(g.weight_kg.min()),
                          "max": float(g.weight_kg.max())},
            "girth_cm": {"mean": round(g.girth_cm.mean(), 1), "min": float(g.girth_cm.min()),
                         "max": float(g.girth_cm.max())},
            "share_female": round(float((g.sex == "F").mean()), 3),
            "n_mature": int(g.mature.sum()),
            "share_mature": round(float(g.mature.mean()), 3),
            "law_b_published": SPECIES[sp]["length_weight"]["b"]
            if not SPECIES[sp]["length_weight"].get("by_sex") else None,
        }
        if len(g) >= 10:
            b, ln_a = np.polyfit(ll, lw, 1)
            pred = ln_a + b * ll
            r2 = 1 - ((lw - pred) ** 2).sum() / ((lw - lw.mean()) ** 2).sum()
            entry["loglog_fit"] = {"b": round(float(b), 3), "a_grams_cm": round(float(math.exp(ln_a)), 5),
                                   "r2": round(float(r2), 4)}
            entry["observed_l50_cm_logistic_on_length"] = fit_logistic_l50(g)
        else:
            entry["loglog_fit"] = None
            entry["observed_l50_cm_logistic_on_length"] = None
        per[sp] = entry
    out["per_species"] = per

    # Whole-table numbers.
    out["share_mature_overall"] = round(float(df.mature.mean()), 3)
    out["count_ganges"] = int((df.species == "ganges").sum())
    out["base_rate_ganges"] = round(3 / n, 4)
    out["base_rate_hammerhead"] = round(float((df.species == "hammerhead").mean()), 4)
    out["accuracy_if_never_predicting_ganges"] = round(1 - 3 / n, 4)
    top = per["spadenose"]["share"]
    out["majority_class"] = {"species": "spadenose", "accuracy_if_always_predicting_it": top}
    out["source_counts"] = df.source.value_counts().to_dict()
    out["site_counts"] = df.site.value_counts().to_dict()
    out["records_per_month"] = {int(m): int(c) for m, c in df.groupby("month").size().items()}
    out["landing_records_per_month"] = {int(m): int(c) for m, c in df[df.source == "landing"].groupby("month").size().items()}
    out["landing_records_in_june_and_july"] = int(((df.source == "landing") & df.month.isin([6, 7])).sum())
    out["temp_c"] = {"mean": round(df.temp_c.mean(), 2), "min": float(df.temp_c.min()), "max": float(df.temp_c.max())}
    out["depth_m"] = {"mean": round(df.depth_m.mean(), 1), "min": float(df.depth_m.min()), "max": float(df.depth_m.max())}
    out["length_cm_overall"] = {"mean": round(df.length_cm.mean(), 1), "min": float(df.length_cm.min()),
                                "max": float(df.length_cm.max())}
    out["weight_kg_overall"] = {"mean": round(df.weight_kg.mean(), 3), "min": float(df.weight_kg.min()),
                                "max": float(df.weight_kg.max())}

    # Recurring individuals.
    names = {4127: ("Tag 4127", "the Juhu bamboo shark", "a juvenile grey bamboo shark found in a rock pool at Juhu at low tide"),
             3305: ("Tag 3305", "the Sassoon spadenose", "an adult female spadenose shark from a Sassoon Dock landing"),
             5519: ("Tag 5519", "the young hammerhead", "a young female scalloped hammerhead from a Sassoon Dock landing")}
    ind = []
    for rid, (name, nickname, story) in names.items():
        r = df[df.record_id == rid].iloc[0].to_dict()
        r = {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
             for k, v in r.items()}
        ind.append({"name": name, "nickname": nickname, "story": story, "record_id": rid, "row": r})
    out["recurring_individuals"] = ind

    # Reference results from simple models, for chapters that quote them.
    out["reference_models"] = reference_models(df)
    return out


def reference_models(df):
    try:
        import sklearn
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
    except ImportError:
        return None
    warnings.filterwarnings("ignore")
    res = {"note": "Five-fold cross-validation with shuffled folds, seed 4127. Reference numbers only; "
                   "a chapter that trains its own model may get slightly different values.",
           "scikit_learn_version": sklearn.__version__}
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Weight from length, spadenose only (lazy model: predict the mean weight).
    s = df[df.species == "spadenose"]
    y = s.weight_kg.to_numpy()
    X = s[["length_cm"]].to_numpy()
    lazy_rmse = float(np.sqrt(((y - y.mean()) ** 2).mean()))
    lin = LinearRegression().fit(X, y)
    rmse = float(np.sqrt(((y - lin.predict(X)) ** 2).mean()))
    res["spadenose_weight_from_length"] = {
        "lazy_rmse_kg": round(lazy_rmse, 4), "linear_rmse_kg": round(rmse, 4),
        "linear_slope_kg_per_cm": round(float(lin.coef_[0]), 5), "linear_intercept_kg": round(float(lin.intercept_), 4),
        "linear_r2": round(float(lin.score(X, y)), 4)}

    # Maturity, spadenose only, from length and girth.
    y = s.mature.to_numpy()
    X = s[["length_cm", "girth_cm"]].to_numpy()
    pred = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=kf)
    res["spadenose_maturity_from_length_and_girth"] = {
        "share_mature": round(float(y.mean()), 3),
        "lazy_accuracy_always_mature": round(float(y.mean()), 3),
        "logistic_cv_accuracy": round(float((pred == y).mean()), 3)}

    # Species from the measurements.
    feats = ["length_cm", "weight_kg", "girth_cm", "depth_m", "temp_c", "month"]
    X = df[feats].to_numpy()
    y = df.species.to_numpy()
    out = {}
    for name, model in [("decision_tree_depth3", DecisionTreeClassifier(max_depth=3, random_state=SEED)),
                        ("gradient_boosting_100x3", GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                                                 random_state=SEED))]:
        pred = cross_val_predict(model, X, y, cv=kf)
        rec = {sp: round(float(((pred == sp) & (y == sp)).sum() / (y == sp).sum()), 3) for sp in COUNTS}
        out[name] = {"cv_accuracy": round(float((pred == y).mean()), 3), "recall_by_species": rec}
    out["features"] = feats
    out["lazy_accuracy_always_spadenose"] = round(float((y == "spadenose").mean()), 3)
    res["species_from_measurements"] = out
    return res


def main():
    df = build()
    checks(df)
    df.to_csv(os.path.join(HERE, "catch.csv"), index=False)
    doc = {
        "about": "Made-up records generated from published length-weight relationships; not field data. "
                 "The animals, tags, dates and places in this table are invented for teaching. "
                 "Do not present them as measurements.",
        "generator": "make_catch.py, seed %d" % SEED,
        "columns": list(df.columns),
        "units": {"month": "1 to 12", "length_cm": "total length, cm", "weight_kg": "kg", "girth_cm": "cm",
                  "depth_m": "metres", "temp_c": "degrees Celsius", "mature": "1 mature, 0 immature",
                  "source": "landing or dive", "species": "see species.json"},
        "closed_season_note": "No landings in June or July (monsoon fishing closure) and few in August.",
        "records": json.loads(df.to_json(orient="records")),
    }
    with open(os.path.join(HERE, "catch.json"), "w", encoding="utf8") as f:
        json.dump(doc, f, separators=(",", ":"), ensure_ascii=False)
    summary = summarise(df)
    with open(os.path.join(HERE, "summary.json"), "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("wrote catch.csv, catch.json, summary.json;", len(df), "records")


if __name__ == "__main__":
    main()
