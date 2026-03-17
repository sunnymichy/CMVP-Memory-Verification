"""
evaluate.py
Paper Section 4: Experimental Result Evaluation.

1) Overall performance comparison (Paper Table 3)
2) Hybrid ensemble evaluation
3) Ablation Study (Paper Table 4) - Contribution analysis per feature group
4) Statistical significance validation (10-repeat paired t-test)
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from scipy import stats

from heuristic_scorer import (
    compute_heuristic_score,
    heuristic_to_class,
    CLASS_NAMES,
)
from hybrid_ensemble import HybridEnsemble
from model_trainer import load_dataset, split_dataset, CLASSES


# ════════════════════════════════════════════════════════════
# 1. Ablation Study (Paper Table 4)
# ════════════════════════════════════════════════════════════

# Feature group definitions (0-indexed)
FEATURE_GROUPS = {
    'F1 (Entropy)':              [0],
    'F2 (Chi-squared)':          [1],
    'F3 (Length)':                [2],
    'F4-F5 (Standard length)':   [3, 4],
    'F6 (Memory region)':        [5],
    'F7-F8 (Temporal pattern)':  [6, 7],
    'F9-F10 (Cross-feature)':    [8, 9],
}


def _train_xgb(X_train, y_train):
    """Lightweight XGBoost training for ablation."""
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=len(CLASSES),
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    return model


def ablation_study(X_train, y_train, X_test, y_test):
    """
    Sequentially remove each feature group and measure the F1 drop.
    Reproduces Paper Table 4.
    """
    print("\n" + "=" * 60)
    print("  Ablation Study (Feature Contribution Analysis)")
    print("=" * 60)

    # Baseline: all 10 features
    full_model = _train_xgb(X_train, y_train)
    baseline_f1 = f1_score(y_test, full_model.predict(X_test), average='weighted')
    print(f"\n  Baseline F1 (all features): {baseline_f1:.4f}")
    print(f"\n  {'Removed feature':<25s} {'F1':>8s} {'Drop':>10s} {'Contrib.':>8s}")
    print(f"  {'-'*53}")

    drops = {}
    for name, indices in FEATURE_GROUPS.items():
        mask = [i for i in range(10) if i not in indices]
        X_train_r = X_train[:, mask]
        X_test_r = X_test[:, mask]

        model = xgb.XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.1,
            objective='multi:softprob',
            num_class=len(CLASSES),
            use_label_encoder=False, random_state=42, verbosity=0,
        )
        model.fit(X_train_r, y_train)
        reduced_f1 = f1_score(y_test, model.predict(X_test_r), average='weighted')
        drops[name] = baseline_f1 - reduced_f1

    # Contribution = individual drop / total drop sum
    total_drop = sum(drops.values())
    for name, drop in sorted(drops.items(), key=lambda x: -x[1]):
        contrib = (drop / total_drop * 100) if total_drop > 0 else 0
        print(f"  {name:<25s} {baseline_f1 - drop:>7.4f} {-drop*100:>+9.1f}%p {contrib:>7.1f}%")

    return drops, baseline_f1


# ════════════════════════════════════════════════════════════
# 2. Hybrid Ensemble Evaluation
# ════════════════════════════════════════════════════════════

def evaluate_hybrid_ensemble(xgb_model, X_test, y_test, le):
    """Hybrid ensemble vs. standalone XGBoost comparison."""
    print("\n" + "=" * 60)
    print("  Hybrid Ensemble Evaluation")
    print("=" * 60)

    # Standalone XGBoost
    y_xgb = xgb_model.predict(X_test)
    xgb_f1 = f1_score(y_test, y_xgb, average='weighted')
    print(f"\n  [Standalone XGBoost]")
    print(classification_report(y_test, y_xgb, target_names=le.classes_,
                                digits=4))

    # Hybrid ensemble
    ensemble = HybridEnsemble(xgb_model, sensitivity='normal')
    y_hybrid, confidences, paths = ensemble.predict(X_test)
    hybrid_f1 = f1_score(y_test, y_hybrid, average='weighted')

    print(f"  [Hybrid Ensemble]")
    print(classification_report(y_test, y_hybrid, target_names=le.classes_,
                                digits=4))

    # Path distribution
    path_counts = {}
    for p in paths:
        path_counts[p] = path_counts.get(p, 0) + 1
    print("  Distribution by path:")
    for path, count in sorted(path_counts.items()):
        print(f"    {path:<15s}: {count:>5d} ({count/len(paths)*100:.1f}%)")

    # Confidence statistics
    print(f"\n  Mean confidence: {np.mean(confidences):.4f}")
    print(f"  Confidence > 90%: {np.sum(confidences >= 0.9)}/{len(confidences)}")

    print(f"\n  XGBoost F1:  {xgb_f1:.4f}")
    print(f"  Ensemble F1: {hybrid_f1:.4f}")
    print(f"  Improvement: {(hybrid_f1 - xgb_f1)*100:+.2f}%p")

    return hybrid_f1


# ════════════════════════════════════════════════════════════
# 3. Statistical Significance Validation (10 repeats)
# ════════════════════════════════════════════════════════════

def statistical_validation(X, y, n_repeats: int = 10):
    """
    Paper Section 4.3: 10-repeat random split experiment + paired sample t-test.
    """
    print("\n" + "=" * 60)
    print(f"  Statistical Significance Validation ({n_repeats} repeats)")
    print("=" * 60)

    ml_f1s = []
    heuristic_f1s = []
    ensemble_f1s = []

    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )

        # XGBoost
        model = _train_xgb(X_train, y_train)
        ml_f1 = f1_score(y_test, model.predict(X_test), average='weighted')
        ml_f1s.append(ml_f1)

        # Hybrid ensemble
        ens = HybridEnsemble(model, sensitivity='normal')
        y_ens, _, _ = ens.predict(X_test)
        ens_f1 = f1_score(y_test, y_ens, average='weighted')
        ensemble_f1s.append(ens_f1)

        # Heuristic baseline
        h_preds = []
        for feat in X_test:
            score = compute_heuristic_score(feat)
            h_preds.append(heuristic_to_class(score, feat))
        h_f1 = f1_score(y_test, h_preds, average='weighted')
        heuristic_f1s.append(h_f1)

        print(f"  Repeat {seed+1:>2d}: ML={ml_f1:.4f}  Ensemble={ens_f1:.4f}  "
              f"Heuristic={h_f1:.4f}")

    ml_f1s = np.array(ml_f1s)
    heuristic_f1s = np.array(heuristic_f1s)
    ensemble_f1s = np.array(ensemble_f1s)

    # Paired sample t-test (ML vs. Heuristic)
    t_stat, p_value = stats.ttest_rel(ml_f1s, heuristic_f1s)

    print(f"\n  ── Summary ──")
    print(f"  ML F1:        {ml_f1s.mean()*100:.2f}% +/- {ml_f1s.std()*100:.2f}%")
    print(f"  Ensemble F1:  {ensemble_f1s.mean()*100:.2f}% +/- {ensemble_f1s.std()*100:.2f}%")
    print(f"  Heuristic F1: {heuristic_f1s.mean()*100:.2f}% +/- {heuristic_f1s.std()*100:.2f}%")
    print(f"  ML-Heuristic diff: +{(ml_f1s.mean()-heuristic_f1s.mean())*100:.2f}%p")
    print(f"  t-statistic:  {t_stat:.4f}")
    print(f"  p-value:      {p_value:.6f} {'*** (p<0.001, significant)' if p_value < 0.001 else ''}")

    return {
        'ml_f1s': ml_f1s,
        'ensemble_f1s': ensemble_f1s,
        'heuristic_f1s': heuristic_f1s,
        't_stat': t_stat,
        'p_value': p_value,
    }


# ════════════════════════════════════════════════════════════
# 4. Heuristic Baseline Evaluation
# ════════════════════════════════════════════════════════════

def evaluate_heuristic_baseline(X_test, y_test, le):
    """Pure heuristic score-based classification performance (baseline)."""
    print("\n" + "=" * 60)
    print("  Heuristic Baseline Evaluation")
    print("=" * 60)

    y_pred = []
    scores = []
    for feat in X_test:
        s = compute_heuristic_score(feat)
        scores.append(s)
        y_pred.append(heuristic_to_class(s, feat))

    y_pred = np.array(y_pred)
    scores = np.array(scores)

    print(classification_report(y_test, y_pred, target_names=le.classes_,
                                digits=4))
    print(f"  Mean score: {scores.mean():.1f}")
    print(f"  Score distribution: min={scores.min():.0f} / "
          f"median={np.median(scores):.0f} / max={scores.max():.0f}")

    return f1_score(y_test, y_pred, average='weighted')


# ════════════════════════════════════════════════════════════
# Main Execution
# ════════════════════════════════════════════════════════════

def run_full_evaluation(csv_path: str):
    """Run the full evaluation pipeline."""
    print("Loading data...")
    X, y, le = load_dataset(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # Train XGBoost
    print("Training XGBoost...")
    xgb_model = _train_xgb(X_train, y_train)

    # (1) Heuristic baseline
    evaluate_heuristic_baseline(X_test, y_test, le)

    # (2) Hybrid ensemble
    evaluate_hybrid_ensemble(xgb_model, X_test, y_test, le)

    # (3) Ablation Study
    ablation_study(X_train, y_train, X_test, y_test)

    # (4) Statistical significance
    statistical_validation(X, y, n_repeats=10)

    print("\nFull evaluation complete.")


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    run_full_evaluation(csv_path)
