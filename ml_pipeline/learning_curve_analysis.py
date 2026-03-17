"""
learning_curve_analysis.py
Paper Section 4: Learning Curve Analysis — Verifying performance saturation with respect to dataset size.

Gradually increases the training set size and measures the primary classifier
(CatBoost)'s Weighted F1, Macro F1, and KEY Recall using 5-fold stratified CV.

Usage:
  python learning_curve_analysis.py [dataset_path]
  (default: dataset/crypto_features.csv)

Output:
  1. Console table (numerical values for the paper)
  2. TikZ coordinates (can be directly inserted into paper figures)
  3. Saturation analysis (delta F1 over the last 3 intervals)
"""

import sys
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score
from scipy import stats

CLASSES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']

# Training set sizes used for the learning curve (10 steps)
TRAIN_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


def load_data(csv_path):
    """Load the CSV dataset."""
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :10].values.astype(np.float64)
    y_str = df['label'].values
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(y_str)
    return X, y, le


def make_catboost():
    """Create a CatBoost model with the optimal hyperparameters from paper Table 5."""
    return CatBoostClassifier(
        iterations=1000,
        depth=9,
        learning_rate=0.05,
        bootstrap_type='MVS',
        subsample=0.9,
        colsample_bylevel=0.8,
        l2_leaf_reg=1.0,
        random_seed=42,
        verbose=0,
        auto_class_weights='Balanced',
    )


def run_learning_curve(X, y, le, train_sizes=TRAIN_SIZES, n_folds=5):
    """
    Perform 5-fold stratified CV at each train_size and return
    Weighted F1, Macro F1, and KEY Recall.

    When train_size is smaller than the full dataset, stratified
    subsampling is applied within each fold to reduce the training set.
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    key_idx = le.transform(['KEY'])[0]

    results = []

    for size in train_sizes:
        fold_wf1 = []
        fold_mf1 = []
        fold_key_recall = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_tr_full, X_te = X[train_idx], X[test_idx]
            y_tr_full, y_te = y[train_idx], y[test_idx]

            # Use full data as-is; otherwise apply stratified subsampling
            if size >= len(y_tr_full):
                X_tr, y_tr = X_tr_full, y_tr_full
            else:
                # stratified subsampling
                rng = np.random.RandomState(42 + fold_idx)
                sub_idx = []
                classes_in_fold = np.unique(y_tr_full)
                for cls in classes_in_fold:
                    cls_mask = np.where(y_tr_full == cls)[0]
                    n_cls = max(1, int(size * len(cls_mask) / len(y_tr_full)))
                    n_cls = min(n_cls, len(cls_mask))
                    sub_idx.extend(rng.choice(cls_mask, n_cls, replace=False))
                sub_idx = np.array(sub_idx)
                rng.shuffle(sub_idx)
                X_tr = X_tr_full[sub_idx]
                y_tr = y_tr_full[sub_idx]

            # Train CatBoost
            model = make_catboost()
            model.fit(X_tr, y_tr, verbose=0)
            y_pred = model.predict(X_te)

            wf1 = f1_score(y_te, y_pred, average='weighted', zero_division=0) * 100
            mf1 = f1_score(y_te, y_pred, average='macro', zero_division=0) * 100

            # KEY recall
            key_mask = (y_te == key_idx)
            if key_mask.sum() > 0:
                key_rec = recall_score(
                    y_te[key_mask] == key_idx,
                    y_pred[key_mask] == key_idx,
                    zero_division=0,
                ) * 100
            else:
                key_rec = 0.0

            fold_wf1.append(wf1)
            fold_mf1.append(mf1)
            fold_key_recall.append(key_rec)

        results.append({
            'size': size,
            'wf1_mean': np.mean(fold_wf1),
            'wf1_std': np.std(fold_wf1, ddof=1),
            'mf1_mean': np.mean(fold_mf1),
            'mf1_std': np.std(fold_mf1, ddof=1),
            'key_recall_mean': np.mean(fold_key_recall),
            'key_recall_std': np.std(fold_key_recall, ddof=1),
        })

        print(f"  N={size:>5d}: WF1={np.mean(fold_wf1):.2f}±{np.std(fold_wf1, ddof=1):.2f}  "
              f"MF1={np.mean(fold_mf1):.2f}±{np.std(fold_mf1, ddof=1):.2f}  "
              f"KEY_Rec={np.mean(fold_key_recall):.1f}%")

    return results


def saturation_analysis(results):
    """Analyze saturation by examining the delta F1 changes over the last 3-4 intervals."""
    print("\n" + "=" * 70)
    print("  Saturation Analysis")
    print("=" * 70)

    # Delta between consecutive intervals
    print(f"\n  {'Interval':<20s} {'ΔWF1':>8s} {'ΔMWF1':>8s}")
    print(f"  {'-'*40}")
    deltas_wf1 = []
    for i in range(1, len(results)):
        prev = results[i - 1]
        curr = results[i]
        d_wf1 = curr['wf1_mean'] - prev['wf1_mean']
        d_mf1 = curr['mf1_mean'] - prev['mf1_mean']
        deltas_wf1.append(d_wf1)
        interval = f"{prev['size']}→{curr['size']}"
        print(f"  {interval:<20s} {d_wf1:>+7.2f} {d_mf1:>+7.2f}")

    # Average delta over the last 3 intervals
    last_3_deltas = deltas_wf1[-3:]
    avg_delta = np.mean(last_3_deltas)
    print(f"\n  Average ΔWF1 over the last 3 intervals: {avg_delta:+.2f} pp")

    if abs(avg_delta) < 0.5:
        print("  → Saturation confirmed: average change over the last 3 intervals < 0.5 pp")
        print("    (Limited room for performance improvement with additional data)")
    elif abs(avg_delta) < 1.0:
        print("  → Near-saturation: average change over the last 3 intervals is 0.5–1.0 pp")
    else:
        print("  → Not saturated: additional data may still contribute to performance improvement")

    # Marginal gain analysis: last vs first half
    first_half_gain = results[len(results) // 2]['wf1_mean'] - results[0]['wf1_mean']
    second_half_gain = results[-1]['wf1_mean'] - results[len(results) // 2]['wf1_mean']
    print(f"\n  First-half total gain (N={results[0]['size']}→{results[len(results)//2]['size']}): "
          f"+{first_half_gain:.2f} pp")
    print(f"  Second-half total gain (N={results[len(results)//2]['size']}→{results[-1]['size']}): "
          f"+{second_half_gain:.2f} pp")
    ratio = second_half_gain / first_half_gain if first_half_gain > 0 else 0
    print(f"  Second-half/first-half ratio: {ratio:.2f} (below 1.0 indicates diminishing returns)")

    return avg_delta, first_half_gain, second_half_gain


def generate_tikz_coordinates(results):
    """Generate TikZ coordinate strings for insertion into paper figures."""
    print("\n" + "=" * 70)
    print("  TikZ Coordinates (for paper figure insertion)")
    print("=" * 70)

    # Weighted F1
    print("\n  % Weighted F1 (mean)")
    print("  \\addplot[blue, mark=*, thick] coordinates {")
    for r in results:
        print(f"    ({r['size']},{r['wf1_mean']:.2f})")
    print("  };")

    # Weighted F1 error bars (±SD)
    print("\n  % Weighted F1 error bars")
    print("  \\addplot[blue, mark=*, thick, error bars/.cd, y dir=both, y explicit]")
    print("    coordinates {")
    for r in results:
        print(f"    ({r['size']},{r['wf1_mean']:.2f}) +- (0,{r['wf1_std']:.2f})")
    print("  };")

    # Macro F1
    print("\n  % Macro F1 (mean)")
    print("  \\addplot[red, mark=triangle*, thick, dashed] coordinates {")
    for r in results:
        print(f"    ({r['size']},{r['mf1_mean']:.2f})")
    print("  };")

    # Macro F1 error bars
    print("\n  % Macro F1 error bars")
    print("  \\addplot[red, mark=triangle*, thick, dashed, error bars/.cd, y dir=both, y explicit]")
    print("    coordinates {")
    for r in results:
        print(f"    ({r['size']},{r['mf1_mean']:.2f}) +- (0,{r['mf1_std']:.2f})")
    print("  };")

    # KEY Recall
    print("\n  % KEY Recall (mean)")
    print("  \\addplot[green!60!black, mark=square*, thick, dotted] coordinates {")
    for r in results:
        print(f"    ({r['size']},{r['key_recall_mean']:.2f})")
    print("  };")


def print_paper_table(results):
    """Print a table formatted for paper insertion."""
    print("\n" + "=" * 70)
    print("  Learning Curve Table (for paper)")
    print("=" * 70)

    print(f"\n  {'N':>6s} {'WF1 Mean':>10s} {'WF1 SD':>8s} {'MF1 Mean':>10s} "
          f"{'MF1 SD':>8s} {'KEY Rec':>8s} {'ΔWF1':>8s}")
    print(f"  {'-'*62}")

    for i, r in enumerate(results):
        delta = '' if i == 0 else f"{r['wf1_mean'] - results[i-1]['wf1_mean']:+.2f}"
        print(f"  {r['size']:>6d} {r['wf1_mean']:>9.2f}% {r['wf1_std']:>7.2f} "
              f"{r['mf1_mean']:>9.2f}% {r['mf1_std']:>7.2f} "
              f"{r['key_recall_mean']:>7.1f}% {delta:>8s}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    print(f"Dataset: {csv_path}")

    X, y, le = load_data(csv_path)
    print(f"Total samples: {len(y)}")
    print(f"Class distribution: {dict(zip(*np.unique(le.inverse_transform(y), return_counts=True)))}")

    print("\n" + "=" * 70)
    print("  Learning Curve Analysis (CatBoost, 5-fold stratified CV)")
    print("=" * 70)

    results = run_learning_curve(X, y, le)

    print_paper_table(results)
    avg_delta, first_gain, second_gain = saturation_analysis(results)
    generate_tikz_coordinates(results)

    print("\n" + "=" * 70)
    print("  Done")
    print("=" * 70)


if __name__ == '__main__':
    main()
