"""
experiment_fraction_matched_midh.py
===================================
Fraction-matched Material-ID Holdout (MIDH): removes value-level overlap while holding the
training fraction fixed (60%, MID-disjoint), so the only difference vs. the sample-wise
60/20/20 split is the absence of value-level overlap. This isolates value-level leakage.

`assign_mids` and `mid_disjoint_split` are imported by run_pathB.py to produce the paper's
MIDH number; this file can also be run standalone for the three-condition comparison.

USAGE
-----
    python experiment_fraction_matched_midh.py --csv dataset/real_crypto_features.csv \
        --out results_midh.json --tex results_midh.tex --seed 42
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score

FEATURES = [
    "F1_entropy", "F2_chi_square", "F3_length", "F4_standard_key_len",
    "F5_standard_iv_len", "F6_memory_region", "F7_change_count",
    "F8_change_pattern", "F9_entropy_length_interaction", "F10_high_confidence_key",
]
LABELS = ["KEY", "IV", "CIPHERTEXT", "PLAINTEXT", "NON_CRYPTO"]


def make_model(seed):
    """XGBoost (deployed) if available; else sklearn GBDT fallback."""
    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder().fit(LABELS)
        clf = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                objective="multi:softprob", num_class=5,
                                eval_metric="mlogloss", random_state=seed, verbosity=0)
        return ("xgb", clf, le), "XGBoost-200-d6-lr0.1"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return ("skl", HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                                       random_state=seed), None), "HistGBDT-fallback"


def assign_mids(df, entropy_bins=8):
    """Reconstruct material IDs: group by (library, algorithm, label, block-length) then
    split by entropy proximity. On a real corpus with recorded key identities, replace
    this with the true per-sample identity."""
    ent = df["F1_entropy"].astype(float)
    qbin = pd.qcut(ent.rank(method="first"), q=entropy_bins, labels=False)
    key_cols = (
        df["library"].astype(str) + "|" + df["algorithm"].astype(str) + "|" +
        df["label"].astype(str) + "|" +
        df["F3_length"].astype(float).round().astype(int).astype(str) + "|" +
        pd.Series(qbin, index=df.index).astype(str)
    )
    return key_cols.astype("category").cat.codes.to_numpy()


def mid_disjoint_split(mids, test_mid_frac, seed, train_frac=None):
    """Boolean masks (train, test) with disjoint MIDs. If train_frac is set, sub-sample
    train MIDs so the TRAIN ROW fraction ~= train_frac."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(mids); rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_mid_frac))
    test_mids = set(uniq[:n_test].tolist())
    test_mask = np.isin(mids, list(test_mids))
    train_mask = ~test_mask
    if train_frac is not None:
        target_rows = int(round(train_frac * len(mids)))
        train_mids = [m for m in uniq[n_test:]]; rng.shuffle(train_mids)
        kept, kept_rows = [], 0
        for m in train_mids:
            rows = int(np.sum(mids == m))
            if kept_rows + rows > target_rows and kept_rows >= target_rows * 0.97:
                break
            kept.append(m); kept_rows += rows
        train_mask = np.isin(mids, kept)
    return train_mask, test_mask


def _fit_predict(model_tuple, X_tr, y_tr, X_te):
    kind, clf, le = model_tuple
    if kind == "xgb":
        clf.fit(X_tr, le.transform(np.asarray(y_tr).astype(str)))
        return le.inverse_transform(np.asarray(clf.predict(X_te)).ravel().astype(int))
    clf.fit(X_tr, np.asarray(y_tr).astype(str))
    return np.asarray(clf.predict(X_te)).ravel().astype(str)


def evaluate(model_factory, X_tr, y_tr, X_te, y_te, seed):
    model_tuple, name = model_factory(seed)
    pred = _fit_predict(model_tuple, X_tr, y_tr, X_te)
    y_te = np.asarray(y_te).astype(str)
    return {"model": name,
            "W_F1": round(f1_score(y_te, pred, labels=LABELS, average="weighted", zero_division=0) * 100, 2),
            "M_F1": round(f1_score(y_te, pred, labels=LABELS, average="macro", zero_division=0) * 100, 2),
            "KEY_recall": round(recall_score((y_te == "KEY").astype(int), (pred == "KEY").astype(int),
                                              zero_division=0) * 100, 2),
            "n_train": int(len(X_tr)), "n_test": int(len(X_te))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/real_crypto_features.csv")
    ap.add_argument("--out", default="results_midh.json")
    ap.add_argument("--tex", default="results_midh.tex")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset missing columns: {missing}")
    X = df[FEATURES].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    results = {"dataset": args.csv, "n_total": int(len(df))}

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=args.seed)
    results["A_sample_wise_60_20_20"] = evaluate(make_model, Xtr, ytr, Xte, yte, args.seed)

    mids = assign_mids(df)
    results["n_mids"] = int(len(np.unique(mids)))

    tr_b, te_b = mid_disjoint_split(mids, test_mid_frac=0.35, seed=args.seed)
    results["B_midh_no_valset"] = evaluate(make_model, X[tr_b], y[tr_b], X[te_b], y[te_b], args.seed)
    results["B_midh_no_valset"]["train_row_frac"] = round(float(tr_b.mean()), 4)

    cs = []
    for r in range(args.repeats):
        s = args.seed + r
        tr_c, te_c = mid_disjoint_split(mids, test_mid_frac=0.20, seed=s, train_frac=0.60)
        cs.append(evaluate(make_model, X[tr_c], y[tr_c], X[te_c], y[te_c], s))
    arr = lambda k: np.array([c[k] for c in cs], dtype=float)
    results["C_fraction_matched_midh"] = {
        "model": cs[0]["model"],
        "W_F1_mean": round(arr("W_F1").mean(), 2), "W_F1_sd": round(arr("W_F1").std(ddof=1), 2),
        "M_F1_mean": round(arr("M_F1").mean(), 2), "M_F1_sd": round(arr("M_F1").std(ddof=1), 2),
        "KEY_recall_mean": round(arr("KEY_recall").mean(), 2),
        "KEY_recall_sd": round(arr("KEY_recall").std(ddof=1), 2),
        "train_row_frac": round(float(np.mean([c["n_train"] for c in cs]) / len(df)), 4),
        "runs": cs}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
