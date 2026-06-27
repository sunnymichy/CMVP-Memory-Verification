"""
run_full_evaluation.py
6개 모델(CatBoost, XGBoost, RF, LightGBM, TabNet, MLP) 전체 평가.
5-fold CV + single split + LOLO for all models.
"""
import sys
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, confusion_matrix
)

CLASSES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :10].values.astype(np.float64)
    y_str = df['label'].values
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(y_str)
    libraries = df['library'].values if 'library' in df.columns else None
    return X, y, le, libraries


def make_models():
    """Return dict of (model, needs_scaler) for all 6 models."""
    return {
        'CatBoost': (CatBoostClassifier(
            iterations=1000, depth=9, learning_rate=0.05,
            bootstrap_type='MVS', subsample=0.9,
            colsample_bylevel=0.8, l2_leaf_reg=1.0,
            random_seed=42, verbose=0,
            auto_class_weights='Balanced',
        ), False),
        'XGBoost': (xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective='multi:softprob', num_class=5,
            eval_metric='mlogloss', use_label_encoder=False,
            random_state=42, verbosity=0,
        ), False),
        'RandomForest': (RandomForestClassifier(
            n_estimators=100, max_depth=None,
            random_state=42, n_jobs=-1,
        ), False),
        'LightGBM': (lgb.LGBMClassifier(
            n_estimators=200, num_leaves=31,
            learning_rate=0.1, random_state=42, verbosity=-1,
        ), False),
        'TabNet': (TabNetClassifier(
            n_d=8, n_a=8, n_steps=3, gamma=1.3, seed=42, verbose=0,
        ), True),
        'MLP': (MLPClassifier(
            hidden_layer_sizes=(64, 32, 16), activation='relu',
            solver='adam', max_iter=500, early_stopping=True,
            validation_fraction=0.2, random_state=42,
        ), True),
    }


def fit_predict(name, model, needs_scaler, X_tr, y_tr, X_te):
    if needs_scaler:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        model.fit(X_tr_s, y_tr)
        return model.predict(X_te_s)
    else:
        model.fit(X_tr, y_tr)
        return model.predict(X_te)


def run_5fold_cv(X, y, n_folds=5):
    """5-fold CV for all 6 models."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = {name: [] for name in make_models()}

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        print(f"  Fold {fold_idx+1}/{n_folds}...")

        for name, (model_template, needs_scaler) in make_models().items():
            y_pred = fit_predict(name, model_template, needs_scaler, X_tr, y_tr, X_te)
            wf1 = f1_score(y_te, y_pred, average='weighted') * 100
            results[name].append(wf1)
            print(f"    {name}: WF1={wf1:.2f}%")

    print("\n" + "=" * 70)
    print("  5-Fold CV Results (for paper Table)")
    print("=" * 70)
    for name in ['CatBoost', 'XGBoost', 'RandomForest', 'LightGBM', 'TabNet', 'MLP']:
        vals = results[name]
        mean_v = np.mean(vals)
        sd_v = np.std(vals, ddof=1)
        print(f"  {name:<15s}: {mean_v:.2f} +/- {sd_v:.2f}")

    # Paired t-tests vs CatBoost
    from scipy import stats
    print("\n  Paired t-tests vs CatBoost:")
    cat_vals = np.array(results['CatBoost'])
    for name in ['XGBoost', 'RandomForest', 'LightGBM', 'TabNet', 'MLP']:
        other_vals = np.array(results[name])
        diff = other_vals - cat_vals
        delta = np.mean(diff)
        t_stat, p_val = stats.ttest_rel(other_vals, cat_vals)
        d_z = abs(t_stat) / np.sqrt(n_folds)
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
        print(f"  {name:<15s}: delta={delta:+.2f}, t={t_stat:.2f}, p={p_val:.4f}{sig}, d={d_z:.2f}")

    return results


def run_single_split(X, y, le):
    """60/20/20 single split with per-class results."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\n  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    print("\n" + "=" * 70)
    print("  Single Split Per-Class F1 (for paper Table)")
    print("=" * 70)

    header = f"  {'Model':<15s}"
    for cls in CLASSES:
        header += f" {cls[:5]:>7s}"
    header += f" {'W-F1':>7s} {'M-F1':>7s}"
    print(header)
    print("  " + "-" * 75)

    for name, (model_template, needs_scaler) in make_models().items():
        y_pred = fit_predict(name, model_template, needs_scaler,
                             X_train, y_train, X_test)
        wf1 = f1_score(y_test, y_pred, average='weighted') * 100
        mf1 = f1_score(y_test, y_pred, average='macro') * 100

        # Per-class F1
        per_class = {}
        for cls_name in CLASSES:
            cls_idx = le.transform([cls_name])[0]
            mask_true = (y_test == cls_idx)
            mask_pred = (y_pred == cls_idx)
            tp = np.sum(mask_true & mask_pred)
            fp = np.sum(~mask_true & mask_pred)
            fn = np.sum(mask_true & ~mask_pred)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            per_class[cls_name] = f1 * 100

        row = f"  {name:<15s}"
        for cls in CLASSES:
            row += f" {per_class[cls]:>6.1f}%"
        row += f" {wf1:>6.2f}% {mf1:>6.2f}%"
        print(row)

        # Print confusion matrix for CatBoost
        if name == 'CatBoost':
            cm = confusion_matrix(y_test, y_pred)
            # Normalize by row
            cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            print(f"\n  CatBoost Confusion Matrix (normalized, n={len(y_test)}):")
            print(f"  {'':>15s}", end="")
            for cls in CLASSES:
                print(f" {cls[:5]:>7s}", end="")
            print()
            for i, cls in enumerate(CLASSES):
                print(f"  {cls:<15s}", end="")
                for j in range(len(CLASSES)):
                    print(f" {cm_norm[i,j]:>7.3f}", end="")
                print(f"  Recall={cm_norm[i,i]*100:.1f}%")
            print(f"  {'Precision':>15s}", end="")
            for j in range(len(CLASSES)):
                col_sum = cm[:, j].sum()
                prec = cm[j, j] / col_sum * 100 if col_sum > 0 else 0
                print(f" {prec:>6.1f}%", end="")
            print()

    # Test set class counts
    print(f"\n  Test set class counts (n={len(y_test)}):")
    for cls_name in CLASSES:
        cls_idx = le.transform([cls_name])[0]
        count = np.sum(y_test == cls_idx)
        print(f"    {cls_name}: {count}")


def run_lolo(X, y, le, libraries):
    """Leave-One-Library-Out CV for CatBoost."""
    if libraries is None:
        print("  No library column found, skipping LOLO.")
        return

    unique_libs = np.unique(libraries)
    print("\n" + "=" * 70)
    print("  LOLO CV Results (CatBoost)")
    print("=" * 70)

    header = f"  {'Library':<15s} {'Train':>6s} {'Test':>6s} {'W-F1':>7s} {'M-F1':>7s} {'KEY-Rec':>8s}"
    print(header)
    print("  " + "-" * 55)

    wf1_list = []
    mf1_list = []
    key_idx = le.transform(['KEY'])[0]

    for lib in unique_libs:
        mask = libraries == lib
        X_tr, y_tr = X[~mask], y[~mask]
        X_te, y_te = X[mask], y[mask]

        model = CatBoostClassifier(
            iterations=1000, depth=9, learning_rate=0.05,
            bootstrap_type='MVS', subsample=0.9,
            colsample_bylevel=0.8, l2_leaf_reg=1.0,
            random_seed=42, verbose=0,
            auto_class_weights='Balanced',
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)

        wf1 = f1_score(y_te, y_pred, average='weighted') * 100
        mf1 = f1_score(y_te, y_pred, average='macro') * 100
        key_mask = (y_te == key_idx)
        key_rec = recall_score(y_te[key_mask] == key_idx,
                               y_pred[key_mask] == key_idx) * 100 if key_mask.sum() > 0 else 0

        wf1_list.append(wf1)
        mf1_list.append(mf1)
        print(f"  {lib:<15s} {len(y_tr):>6d} {len(y_te):>6d} {wf1:>6.1f}% {mf1:>6.1f}% {key_rec:>7.1f}%")

    print("  " + "-" * 55)
    print(f"  {'Mean +/- SD':<15s} {'':>6s} {'':>6s} {np.mean(wf1_list):>5.1f}+/-{np.std(wf1_list,ddof=1):.1f} "
          f"{np.mean(mf1_list):>5.1f}+/-{np.std(mf1_list,ddof=1):.1f}")


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    print(f"Dataset: {csv_path}")
    X, y, le, libraries = load_data(csv_path)
    print(f"Total samples: {len(y)}")

    # Single split evaluation (per-class F1 + confusion matrix)
    run_single_split(X, y, le)

    # 5-fold CV
    print("\n" + "=" * 70)
    print("  5-Fold Stratified CV")
    print("=" * 70)
    run_5fold_cv(X, y)

    # LOLO
    run_lolo(X, y, le, libraries)

    print("\n  Done.")


if __name__ == '__main__':
    main()
