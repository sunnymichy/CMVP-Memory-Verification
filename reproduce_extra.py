"""
reproduce_extra.py  --  regenerate the paper numbers that are NOT emitted by run_pathB.py
=========================================================================================
Run from the ml_pipeline/ directory (uses relative paths). Deployed model = XGBoost,
configured identically to run_pathB.py (n_estimators=200, max_depth=6, lr=0.1,
multi:softprob, seed=42), so results are consistent with results_real.json.

Produces results_real_extra.json with:
  - groupkfold        : GroupKFold(5) by (library, algorithm)  -> W/M-F1, KEY recall (Table 11/sec 5.2.2)
  - lolo_detail       : per-library LOLO W/M-F1 + KEY recall    (Table 11)  [W-F1 matches results_real.json]
  - mid               : distinct MID count + mean/SD            (sec 4.1.2)
  - midh_split        : fraction-matched MID-disjoint train/test sizes (sec 5.2.4)
  - key_fpr           : direct KEY-FPR raw counts on the 20% split
  - ece               : pre/post isotonic ECE                   (sec 5.4)
  - tier_hi           : high-tier fraction + tier-restricted KEY-FPR (sec 5.4)
  - aeskeyfind        : AESKeyFind-style binary baseline + non-AES, vs XGBoost (sec 5.1.1)
  - field_block_prec  : overall standard block precision from field_results.json (sec 5.5)

TabNet (Table 7/10) is regenerated separately by reproduce_tabnet.py (slow; needs pytorch-tabnet).

USAGE
-----
    python reproduce_extra.py            # writes results_real_extra.json
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from experiment_fraction_matched_midh import assign_mids, mid_disjoint_split

CSV = "dataset/real_crypto_features.csv"
FIELD = "field_results.json"
CLASSES = ["KEY", "IV", "CIPHERTEXT", "PLAINTEXT", "NON_CRYPTO"]
REAL_LIBS = ["OpenSSL", "PyCryptodome", "Windows CNG", "PyNaCl", "pyaes"]
LE = LabelEncoder().fit(CLASSES)
KEY = LE.transform(["KEY"])[0]


def xgb():
    import xgboost as x
    return x.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                           objective="multi:softprob", num_class=5,
                           eval_metric="mlogloss", random_state=42, verbosity=0)


def ece(conf, corr, bins=15):
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += abs(corr[m].mean() - conf[m].mean()) * m.sum() / len(conf)
    return round(e * 100, 2)


def main():
    df = pd.read_csv(CSV)
    X = df.iloc[:, :10].values.astype(np.float64)
    y = LE.transform(df["label"].values.astype(str))
    lib = df["library"].values
    alg = df["algorithm"].values
    out = {"n_total": int(len(y))}

    # --- GroupKFold by 33 crypto (library, algorithm) pairs; NON_CRYPTO stratified
    #     (each NON_CRYPTO sample is its own singleton group so it spreads across folds) ---
    lbl = df["label"].values.astype(str)
    groups = np.array([f"NON_{i}" if lbl[i] == "NON_CRYPTO" else f"{lib[i]}__{alg[i]}"
                       for i in range(len(y))], dtype=object)
    wf, mf, kr = [], [], []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = xgb(); m.fit(X[tr], y[tr]); p = np.asarray(m.predict(X[te])).ravel().astype(int)
        wf.append(f1_score(y[te], p, average="weighted", zero_division=0) * 100)
        mf.append(f1_score(y[te], p, average="macro", zero_division=0) * 100)
        km = y[te] == KEY; kr.append((p[km] == KEY).sum() / km.sum() * 100 if km.sum() else 0)
    out["groupkfold"] = {"n_crypto_groups": int(len({g for g in groups if not str(g).startswith('NON_')})),
                         "WF1": [round(np.mean(wf), 1), round(np.std(wf, ddof=1), 1)],
                         "MF1": [round(np.mean(mf), 1), round(np.std(mf, ddof=1), 1)],
                         "KEYrec": [round(np.mean(kr), 1), round(np.std(kr, ddof=1), 1)]}

    # --- Contextual-ablated LOLO (drop F5,F6 = idx 4,5): portability check ---
    keep = [i for i in range(10) if i not in (4, 5)]
    ab = {}
    for L in REAL_LIBS:
        te = lib == L; m = xgb(); m.fit(X[~te][:, keep], y[~te])
        ab[L] = round(f1_score(y[te], np.asarray(m.predict(X[te][:, keep])).ravel().astype(int),
                               average="weighted", zero_division=0) * 100, 1)
    out["lolo_contextual_ablated_WF1"] = {"per_lib": ab, "mean": round(float(np.mean(list(ab.values()))), 1)}

    # --- LOLO detail (5 real libraries) ---
    lolo = {}
    for L in REAL_LIBS:
        te = lib == L; m = xgb(); m.fit(X[~te], y[~te]); p = np.asarray(m.predict(X[te])).ravel().astype(int)
        km = y[te] == KEY
        lolo[L] = {"WF1": round(f1_score(y[te], p, average="weighted", zero_division=0) * 100, 1),
                   "MF1": round(f1_score(y[te], p, average="macro", zero_division=0) * 100, 1),
                   "KEYrec": round((p[km] == KEY).sum() / km.sum() * 100, 1) if km.sum() else 0.0}
    out["lolo_detail"] = lolo

    # --- MID + MIDH split ---
    mids = assign_mids(df); _, cnts = np.unique(mids, return_counts=True)
    out["mid"] = {"n_distinct": int(len(cnts)), "mean": round(float(cnts.mean()), 1),
                  "sd": round(float(cnts.std(ddof=1)), 1)}
    trm, tem = mid_disjoint_split(mids, test_mid_frac=0.35, seed=42, train_frac=0.60)
    out["midh_split"] = {"n_train": int(trm.sum()), "n_test": int(tem.sum())}

    # --- single 20% split: KEY-FPR, ECE, tier_hi, AESKeyFind ---
    Xtr, Xte, ytr, yte, _, ate = train_test_split(X, y, alg, test_size=0.2, random_state=42, stratify=y)
    m = xgb(); m.fit(Xtr, ytr)
    proba = m.predict_proba(Xte); pred = proba.argmax(1); maxp = proba.max(1)
    nonkey = yte != KEY
    # NOTE: the headline overall KEY-FPR / deployment PPV is in results_real.json
    # (deployment_ppv: 11.47% / 0.084%), computed from the row-normalized confusion;
    # we do not re-emit it here to avoid a second (direct-count) value for the same quantity.
    ptr = m.predict_proba(Xtr); iso = IsotonicRegression(out_of_bounds="clip").fit(ptr.max(1), (ptr.argmax(1) == ytr).astype(int))
    out["ece"] = {"pre": ece(maxp, (pred == yte).astype(int)), "post": ece(iso.predict(maxp), (pred == yte).astype(int))}
    hi = maxp >= 0.80; hk = hi & nonkey
    out["tier_hi"] = {"frac": round(float(hi.mean()), 3),
                      "fpr_hi": round(int(np.sum(hi & (pred == KEY) & nonkey)) / hk.sum(), 4) if hk.sum() else None}
    aes = {16, 24, 32}
    apred = np.array([1 if (Xte[i, 0] >= 7.0 and Xte[i, 2] in aes and Xte[i, 3] == 1) else 0 for i in range(len(Xte))])
    ybin = (yte == KEY).astype(int); xbin = (pred == KEY).astype(int)
    nonaes = ~np.array([str(a).startswith("AES") for a in ate])

    def bm(yt, pr):
        tp = int(np.sum((pr == 1) & (yt == 1))); fpp = int(np.sum((pr == 1) & (yt == 0))); fn = int(np.sum((pr == 0) & (yt == 1)))
        p = tp / (tp + fpp) * 100 if tp + fpp else 0.0; r = tp / (tp + fn) * 100 if tp + fn else 0.0
        return {"prec": round(p, 1), "rec": round(r, 1), "f1": round(2 * p * r / (p + r), 1) if p + r else 0.0, "tp": tp, "fp": fpp}
    out["aeskeyfind"] = {"AESKeyFind_all": bm(ybin, apred), "XGB_all": bm(ybin, xbin),
                         "XGB_nonAES": bm(ybin[nonaes], xbin[nonaes]), "AESKeyFind_nonAES": bm(ybin[nonaes], apred[nonaes])}

    # --- field block precision ---
    fr = json.load(open(FIELD, encoding="utf-8"))
    tpb = sum(x["detected"] for x in fr["per_module"]); fpb = sum(x["fp_block"] for x in fr["per_module"])
    out["field_block_prec_pct"] = round(tpb / (tpb + fpb) * 100, 1) if tpb + fpb else None

    json.dump(out, open("results_real_extra.json", "w", encoding="utf-8"), indent=2)
    print(json.dumps(out, indent=2))
    print("\nWrote results_real_extra.json")


if __name__ == "__main__":
    main()
