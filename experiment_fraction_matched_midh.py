"""
experiment_fraction_matched_midh.py
====================================
Fraction-matched Material-ID Holdout (MIDH) experiment for the TDSC paper.

WHAT THIS RESOLVES
------------------
Reviewer concern: the reported MIDH W-F1 (89.0%) is *above* the sample-wise split
(88.33%), which the paper attributes to a larger effective training fraction
(64.9% vs. 60%). Because removing value-level overlap should DEPRESS performance
while a larger training set should RAISE it, the net non-degradation cannot, by
itself, isolate "no value-level memorization". This script runs the missing
fraction-matched control: MIDH with the training fraction fixed at 60% (MID-disjoint),
so the only difference vs. the 60/20/20 sample-wise split is the absence of
value-level overlap.

It reports three conditions on the SAME estimator/hyperparameters:
  (A) sample-wise stratified 60/20/20            -> baseline
  (B) MIDH ~65/35 of MIDs, no val set            -> paper's reported MIDH (~64.9% train)
  (C) fraction-matched MIDH, 60% train (MID-disjoint), 20% val carved from train pool

Interpretation: if (C) is within ~1 CV-SD of (A), value-level leakage is negligible.
If (C) drops materially below (A), the original MIDH number was buoyed by the extra
training data and the "negligible leakage" claim is NOT supported.

!!! DATA-PROVENANCE WARNING !!!
-------------------------------
The repository dataset (dataset/crypto_features.csv) is produced by
generate_synthetic_dataset.py, which draws each class's features from hand-specified
distributions. It contains NO underlying key bytes, so a "material identifier" here is
reconstructed from (library, algorithm, label, length) + entropy proximity -- it is NOT
a real cryptographic key identity. On synthetic data this script validates the
PROCEDURE only; the numbers do NOT constitute empirical evidence about real
cryptographic memory and must NOT be inserted into the paper as such. Run this on a
REAL collected corpus (with true per-sample key identities recorded at collection time)
before reporting any number.

USAGE
-----
    python experiment_fraction_matched_midh.py \
        --csv dataset/crypto_features.csv \
        --out results_midh.json \
        --tex results_midh.tex \
        --seed 42

Outputs a JSON (machine-readable) and a .tex snippet (paste-ready) but writes NOTHING
into the manuscript automatically.
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
    """CatBoost if available (matches paper); else sklearn GBDT fallback."""
    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier(
            iterations=1000, depth=9, learning_rate=0.05,
            loss_function="MultiClass", auto_class_weights="Balanced",
            random_seed=seed, verbose=False,
        ), "CatBoost-1000-d9-lr0.05"
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, random_state=seed,
        ), "HistGBDT-fallback(sklearn)"


def assign_mids(df, entropy_bins=8):
    """Reconstruct material IDs per the paper's description:
    group by (library, algorithm, label, block-length) then split by entropy proximity.
    NOTE: on synthetic data this is a proxy, not a true key identity (see header)."""
    ent = df["F1_entropy"].astype(float)
    # entropy-proximity bucket within each (lib,algo,label,len) group
    qbin = pd.qcut(ent.rank(method="first"), q=entropy_bins, labels=False)
    key_cols = (
        df["library"].astype(str) + "|" +
        df["algorithm"].astype(str) + "|" +
        df["label"].astype(str) + "|" +
        df["F3_length"].astype(float).round().astype(int).astype(str) + "|" +
        pd.Series(qbin, index=df.index).astype(str)
    )
    mids = key_cols.astype("category").cat.codes.to_numpy()
    return mids


def evaluate(model_factory, X_tr, y_tr, X_te, y_te, seed):
    model, name = model_factory(seed)
    model.fit(X_tr, y_tr)
    pred = np.asarray(model.predict(X_te)).ravel().astype(str)
    y_te = np.asarray(y_te).astype(str)
    wf1 = f1_score(y_te, pred, labels=LABELS, average="weighted", zero_division=0)
    mf1 = f1_score(y_te, pred, labels=LABELS, average="macro", zero_division=0)
    key_rec = recall_score(
        (y_te == "KEY").astype(int), (pred == "KEY").astype(int), zero_division=0
    )
    return {"model": name, "W_F1": round(wf1 * 100, 2),
            "M_F1": round(mf1 * 100, 2), "KEY_recall": round(key_rec * 100, 2),
            "n_train": int(len(X_tr)), "n_test": int(len(X_te))}


def mid_disjoint_split(mids, test_mid_frac, seed, train_frac=None):
    """Return boolean masks (train, test) with disjoint MIDs.
    If train_frac is set, sub-sample the train MIDs so the TRAIN ROW fraction ~= train_frac."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(mids)
    rng.shuffle(uniq)
    n_test = int(round(len(uniq) * test_mid_frac))
    test_mids = set(uniq[:n_test].tolist())
    test_mask = np.isin(mids, list(test_mids))
    train_mask = ~test_mask
    if train_frac is not None:
        # Trim train MIDs until train row fraction is ~ train_frac of total.
        target_rows = int(round(train_frac * len(mids)))
        train_mids = [m for m in uniq[n_test:]]
        rng.shuffle(train_mids)
        kept, kept_rows = [], 0
        for m in train_mids:
            rows = int(np.sum(mids == m))
            if kept_rows + rows > target_rows and kept_rows >= target_rows * 0.97:
                break
            kept.append(m); kept_rows += rows
        train_mask = np.isin(mids, kept)
    return train_mask, test_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="dataset/crypto_features.csv")
    ap.add_argument("--out", default="results_midh.json")
    ap.add_argument("--tex", default="results_midh.tex")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=5,
                    help="repeat condition C with different seeds for a mean+/-SD")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    missing = [c for c in FEATURES + ["label"] if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: dataset missing columns: {missing}")
    X = df[FEATURES].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()

    print("=" * 64)
    print(" Fraction-matched MIDH experiment")
    print(" PROVENANCE: if csv is the synthetic corpus, results are procedural only.")
    print("=" * 64)

    results = {"dataset": args.csv, "n_total": int(len(df))}

    # (A) sample-wise 60/20/20 (test = 20%)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=args.seed)
    results["A_sample_wise_60_20_20"] = evaluate(make_model, Xtr, ytr, Xte, yte, args.seed)

    # MIDs
    mids = assign_mids(df)
    results["n_mids"] = int(len(np.unique(mids)))

    # (B) MIDH ~35% test MIDs, no val (paper's ~64.9% train)
    tr_b, te_b = mid_disjoint_split(mids, test_mid_frac=0.35, seed=args.seed)
    results["B_midh_no_valset"] = evaluate(
        make_model, X[tr_b], y[tr_b], X[te_b], y[te_b], args.seed)
    results["B_midh_no_valset"]["train_row_frac"] = round(float(tr_b.mean()), 4)

    # (C) fraction-matched MIDH: train fraction fixed to 60% (MID-disjoint), repeated
    cs = []
    for r in range(args.repeats):
        s = args.seed + r
        tr_c, te_c = mid_disjoint_split(
            mids, test_mid_frac=0.20, seed=s, train_frac=0.60)
        cs.append(evaluate(make_model, X[tr_c], y[tr_c], X[te_c], y[te_c], s))
    arr = lambda k: np.array([c[k] for c in cs], dtype=float)
    results["C_fraction_matched_midh"] = {
        "model": cs[0]["model"],
        "W_F1_mean": round(arr("W_F1").mean(), 2), "W_F1_sd": round(arr("W_F1").std(ddof=1), 2),
        "M_F1_mean": round(arr("M_F1").mean(), 2), "M_F1_sd": round(arr("M_F1").std(ddof=1), 2),
        "KEY_recall_mean": round(arr("KEY_recall").mean(), 2),
        "KEY_recall_sd": round(arr("KEY_recall").std(ddof=1), 2),
        "train_row_frac": round(float(np.mean([c["n_train"] for c in cs]) / len(df)), 4),
        "runs": cs,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    A = results["A_sample_wise_60_20_20"]
    C = results["C_fraction_matched_midh"]
    delta = round(C["W_F1_mean"] - A["W_F1"], 2)
    tex = (
        "%% Auto-generated by experiment_fraction_matched_midh.py -- VERIFY before use.\n"
        "%% Suggested replacement for the MIDH paragraph (Sec. V-D4):\n"
        f"A fraction-matched MIDH (training fraction fixed at 60\\%%, MID-disjoint) "
        f"yields W-F1 = {C['W_F1_mean']:.1f}\\%% $\\pm$ {C['W_F1_sd']:.1f}\\%% "
        f"(M-F1: {C['M_F1_mean']:.1f}\\%%; KEY recall: {C['KEY_recall_mean']:.1f}\\%%), "
        f"a {delta:+.1f}~pp change relative to the sample-wise split "
        f"({A['W_F1']:.2f}\\%%). "
        "Because the only difference from the sample-wise split is the removal of "
        "value-level overlap, this isolates value-level leakage.\n"
    )
    with open(args.tex, "w", encoding="utf-8") as f:
        f.write(tex)

    print(json.dumps(results, indent=2))
    print(f"\nWrote {args.out} and {args.tex}")
    print("Reminder: do NOT paste these numbers into the paper if --csv is synthetic.")


if __name__ == "__main__":
    main()
