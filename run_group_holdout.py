"""
run_group_holdout.py
(library, algorithm) GroupKFold cross-validation for CatBoost.
Addresses reviewer concern: trace-group / within-library dependence.

Three experiments:
  1. GroupKFold by (library, algorithm) — proxy for trace-group holdout
  2. Leave-One-Algorithm-Family-Out (LOAFO) — algorithm-level generalization
  3. AESKeyFind-style baseline comparison on classification test set
"""
import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score, precision_score

CLASSES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :10].values.astype(np.float64)
    y_str = df['label'].values
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(y_str)
    libraries = df['library'].values
    algorithms = df['algorithm'].values
    return X, y, le, libraries, algorithms, df


def make_catboost():
    return CatBoostClassifier(
        iterations=1000, depth=9, learning_rate=0.05,
        bootstrap_type='MVS', subsample=0.9,
        colsample_bylevel=0.8, l2_leaf_reg=1.0,
        random_seed=42, verbose=0,
        auto_class_weights='Balanced',
    )


# ─────────────────────────────────────────────────────
# Experiment 1: GroupKFold by (library, algorithm)
# ─────────────────────────────────────────────────────
def run_group_kfold(X, y, le, libraries, algorithms, n_splits=5):
    """GroupKFold CV where each group = (library, algorithm) pair."""
    groups = np.array([f"{lib}__{alg}" for lib, alg in zip(libraries, algorithms)])
    unique_groups = np.unique(groups)
    print(f"\n{'='*70}")
    print(f"  Experiment 1: GroupKFold by (library, algorithm)")
    print(f"  Total groups: {len(unique_groups)}, n_splits={n_splits}")
    print(f"{'='*70}")

    gkf = GroupKFold(n_splits=n_splits)
    key_idx = le.transform(['KEY'])[0]

    wf1_list, mf1_list, key_rec_list = [], [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups)):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        # Which groups are held out?
        held_out = sorted(set(groups[te_idx]))

        model = make_catboost()
        model.fit(X_tr, y_tr)
        y_pred = np.array(model.predict(X_te)).flatten()

        wf1 = f1_score(y_te, y_pred, average='weighted') * 100
        mf1 = f1_score(y_te, y_pred, average='macro') * 100
        key_mask = (y_te == key_idx)
        key_pred_correct = (y_pred[key_mask] == key_idx)
        key_rec = key_pred_correct.sum() / key_mask.sum() * 100 if key_mask.sum() > 0 else 0

        wf1_list.append(wf1)
        mf1_list.append(mf1)
        key_rec_list.append(key_rec)

        print(f"\n  Fold {fold_idx+1}: Train={len(y_tr)}, Test={len(y_te)}, "
              f"Held-out groups={len(held_out)}")
        print(f"    W-F1={wf1:.1f}%, M-F1={mf1:.1f}%, KEY Recall={key_rec:.1f}%")
        for g in held_out:
            print(f"      - {g}")

    print(f"\n  {'='*50}")
    print(f"  GroupKFold Summary (n_splits={n_splits}):")
    print(f"    W-F1:       {np.mean(wf1_list):.1f} +/- {np.std(wf1_list, ddof=1):.1f}%")
    print(f"    M-F1:       {np.mean(mf1_list):.1f} +/- {np.std(mf1_list, ddof=1):.1f}%")
    print(f"    KEY Recall: {np.mean(key_rec_list):.1f} +/- {np.std(key_rec_list, ddof=1):.1f}%")

    return {
        'wf1_mean': np.mean(wf1_list), 'wf1_sd': np.std(wf1_list, ddof=1),
        'mf1_mean': np.mean(mf1_list), 'mf1_sd': np.std(mf1_list, ddof=1),
        'key_rec_mean': np.mean(key_rec_list), 'key_rec_sd': np.std(key_rec_list, ddof=1),
        'per_fold_wf1': wf1_list,
    }


# ─────────────────────────────────────────────────────
# Experiment 2: AESKeyFind-style baseline
# ─────────────────────────────────────────────────────
def aeskeyfind_baseline(X, y, le, algorithms):
    """
    Simulate AESKeyFind-style detection on feature vectors.
    AESKeyFind logic: detect AES keys by checking
      (1) standard AES key length (F3 in {16, 24, 32})
      (2) high entropy (F1 >= 7.0)
      (3) standard_key_len flag (F4 == 1)

    This baseline can only detect KEY class for AES algorithms.
    Non-AES algorithms and non-KEY classes are out of scope.
    """
    print(f"\n{'='*70}")
    print(f"  Experiment 2: AESKeyFind-Style Baseline Comparison")
    print(f"{'='*70}")

    key_idx = le.transform(['KEY'])[0]

    # AES algorithms only
    aes_mask = np.array([alg.startswith('AES') for alg in algorithms])
    aes_key_lengths = {16, 24, 32}

    # Full dataset evaluation
    # AESKeyFind prediction: KEY if (AES algo) AND (F1>=7.0) AND (F3 in aes_key_lengths) AND (F4==1)
    # For fair comparison, we only check feature-level properties (no algorithm label at test time)
    # So: predict KEY if F1>=7.0 AND F3 in {16,24,32} AND F4==1
    aes_pred = np.zeros(len(y), dtype=int)
    for i in range(len(y)):
        f1_ent = X[i, 0]   # F1: entropy
        f3_len = X[i, 2]   # F3: length
        f4_std = X[i, 3]   # F4: standard_key_len
        if f1_ent >= 7.0 and f3_len in aes_key_lengths and f4_std == 1:
            aes_pred[i] = 1  # predict KEY

    # Evaluate as binary KEY detection
    y_binary = (y == key_idx).astype(int)

    tp = np.sum((aes_pred == 1) & (y_binary == 1))
    fp = np.sum((aes_pred == 1) & (y_binary == 0))
    fn = np.sum((aes_pred == 0) & (y_binary == 1))
    tn = np.sum((aes_pred == 0) & (y_binary == 0))

    prec = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

    print(f"\n  AESKeyFind-style (binary KEY detection, full dataset):")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"    Precision: {prec:.1f}%")
    print(f"    Recall:    {rec:.1f}%")
    print(f"    F1:        {f1:.1f}%")

    # Now evaluate CatBoost on same binary KEY task for comparison
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, alg_train, alg_test = train_test_split(
        X, y, algorithms, test_size=0.2, random_state=42, stratify=y)

    model = make_catboost()
    model.fit(X_train, y_train)
    y_pred_cb = np.array(model.predict(X_test)).flatten()

    y_test_bin = (y_test == key_idx).astype(int)
    cb_key_pred = (y_pred_cb == key_idx).astype(int)

    # AESKeyFind on same test set
    aes_pred_test = np.zeros(len(y_test), dtype=int)
    for i in range(len(y_test)):
        f1_ent = X_test[i, 0]
        f3_len = X_test[i, 2]
        f4_std = X_test[i, 3]
        if f1_ent >= 7.0 and f3_len in aes_key_lengths and f4_std == 1:
            aes_pred_test[i] = 1

    def binary_metrics(y_true, y_pred, label):
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
        r = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
        f = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"    {label:<25s}: Prec={p:.1f}%, Rec={r:.1f}%, F1={f:.1f}% (TP={tp}, FP={fp}, FN={fn})")
        return {'prec': p, 'rec': r, 'f1': f, 'tp': tp, 'fp': fp, 'fn': fn}

    print(f"\n  Binary KEY detection on test set (n={len(y_test)}):")
    aes_metrics = binary_metrics(y_test_bin, aes_pred_test, "AESKeyFind-style")
    cb_metrics = binary_metrics(y_test_bin, cb_key_pred, "CatBoost (5-class)")

    # AES-only subset
    aes_test_mask = np.array([alg.startswith('AES') for alg in alg_test])
    if aes_test_mask.sum() > 0:
        print(f"\n  AES-only subset (n={aes_test_mask.sum()}):")
        y_test_aes_bin = y_test_bin[aes_test_mask]
        binary_metrics(y_test_aes_bin, aes_pred_test[aes_test_mask], "AESKeyFind-style")
        binary_metrics(y_test_aes_bin, cb_key_pred[aes_test_mask], "CatBoost (5-class)")

    # Non-AES subset (where AESKeyFind cannot work)
    non_aes_test_mask = ~aes_test_mask
    if non_aes_test_mask.sum() > 0:
        print(f"\n  Non-AES subset (n={non_aes_test_mask.sum()}):")
        y_test_nonaes_bin = y_test_bin[non_aes_test_mask]
        binary_metrics(y_test_nonaes_bin, aes_pred_test[non_aes_test_mask], "AESKeyFind-style")
        binary_metrics(y_test_nonaes_bin, cb_key_pred[non_aes_test_mask], "CatBoost (5-class)")

    return aes_metrics, cb_metrics


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    print(f"Dataset: {csv_path}")
    X, y, le, libraries, algorithms, df = load_data(csv_path)
    print(f"Total samples: {len(y)}")

    # Experiment 1: GroupKFold
    gkf_results = run_group_kfold(X, y, le, libraries, algorithms)

    # Experiment 2: AESKeyFind baseline
    aes_results = aeskeyfind_baseline(X, y, le, algorithms)

    # Summary for paper
    print(f"\n{'='*70}")
    print(f"  SUMMARY FOR PAPER")
    print(f"{'='*70}")
    print(f"  GroupKFold (library,algorithm):")
    print(f"    W-F1: {gkf_results['wf1_mean']:.1f} +/- {gkf_results['wf1_sd']:.1f}%")
    print(f"    M-F1: {gkf_results['mf1_mean']:.1f} +/- {gkf_results['mf1_sd']:.1f}%")
    print(f"    KEY Recall: {gkf_results['key_rec_mean']:.1f} +/- {gkf_results['key_rec_sd']:.1f}%")
    print(f"  Degradation from 5-fold CV (88.47%): {88.47 - gkf_results['wf1_mean']:.1f} pp")
    print(f"  Degradation from LOLO (86.0%): {86.0 - gkf_results['wf1_mean']:.1f} pp")


if __name__ == '__main__':
    main()
