"""
run_all_evaluations.py
논문의 모든 성능 수치를 재생성하는 종합 평가 스크립트.

출력:
  1. 단일 분할 (60/20/20) 성능 - tab:performance, tab:class_performance, tab:confusion
  2. 계층적 5-겹 교차 검증 - tab:kfold
  3. 10회 반복 무작위 분할 - subsec:results_stats
  4. LOLO CV - tab:lolo
  5. 모델 간 쌍별 t-검정 - tab:pairwise
  6. 휴리스틱 성능 비교
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from scipy import stats
import sys
import json

sys.path.insert(0, '.')
from heuristic_scorer import compute_heuristic_score, heuristic_to_class

CLASSES = ['KEY', 'IV', 'CIPHERTEXT', 'PLAINTEXT', 'NON_CRYPTO']

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :10].values.astype(np.float64)
    y_str = df['label'].values
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(y_str)
    library = df['library'].values if 'library' in df.columns else None
    return X, y, le, library


def make_models():
    """3개 ML 모델 + 고정 하이퍼파라미터."""
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=5,
        eval_metric='mlogloss', use_label_encoder=False,
        random_state=42, verbosity=0,
    )
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=None, random_state=42, n_jobs=-1,
    )
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), activation='relu',
        solver='adam', max_iter=500, early_stopping=True,
        validation_fraction=0.2, random_state=42,
    )
    return xgb_model, rf_model, mlp_model


def heuristic_predict(X):
    """휴리스틱 기반 예측."""
    preds = []
    for row in X:
        score = compute_heuristic_score(row)
        pred = heuristic_to_class(score, row)
        preds.append(pred)
    return np.array(preds)


# ═══════════════════════════════════════════════════════════
# 1. 단일 분할 (60/20/20)
# ═══════════════════════════════════════════════════════════
def run_single_split(X, y, le):
    print("\n" + "="*70)
    print("  1. 단일 분할 (60/20/20)")
    print("="*70)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    xgb_m, rf_m, mlp_m = make_models()

    # XGBoost
    xgb_m.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred_xgb = xgb_m.predict(X_test)

    # Random Forest
    rf_m.fit(X_train, y_train)
    y_pred_rf = rf_m.predict(X_test)

    # MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    mlp_m.fit(X_train_s, y_train)
    y_pred_mlp = mlp_m.predict(scaler.transform(X_test))

    # Heuristic
    y_pred_heur = heuristic_predict(X_test)

    results = {}
    for name, y_pred in [('XGBoost', y_pred_xgb), ('RandomForest', y_pred_rf),
                          ('MLP', y_pred_mlp), ('Heuristic', y_pred_heur)]:
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        f1m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        results[name] = {'prec': prec, 'rec': rec, 'f1w': f1w, 'f1m': f1m, 'y_pred': y_pred}

    # tab:performance
    print("\n  [tab:performance] Single Split Performance:")
    print(f"  {'Model':<20s} {'Prec':>8s} {'Recall':>8s} {'F1_w':>8s} {'F1_m':>8s}")
    print(f"  {'-'*50}")
    for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']:
        r = results[name]
        print(f"  {name:<20s} {r['prec']:>7.1%} {r['rec']:>7.1%} {r['f1w']:>7.1%} {r['f1m']:>7.1%}")

    # tab:class_performance (XGBoost detailed)
    print(f"\n  [tab:class_performance] Per-Class (Test n={len(y_test)}):")
    report_xgb = classification_report(y_test, y_pred_xgb, target_names=le.classes_,
                                        output_dict=True, zero_division=0)
    report_rf = classification_report(y_test, y_pred_rf, target_names=le.classes_,
                                       output_dict=True, zero_division=0)
    report_mlp = classification_report(y_test, y_pred_mlp, target_names=le.classes_,
                                        output_dict=True, zero_division=0)

    print(f"  {'Class':<15s} {'Prec':>8s} {'Recall':>8s} {'F1(XGB)':>8s} {'F1(RF)':>8s} {'F1(MLP)':>8s} {'Support':>8s}")
    print(f"  {'-'*65}")
    for cls in CLASSES:
        xgb_r = report_xgb[cls]
        rf_r = report_rf[cls]
        mlp_r = report_mlp[cls]
        print(f"  {cls:<15s} {xgb_r['precision']:>7.1%} {xgb_r['recall']:>7.1%} "
              f"{xgb_r['f1-score']:>7.1%} {rf_r['f1-score']:>7.1%} {mlp_r['f1-score']:>7.1%} "
              f"{int(xgb_r['support']):>7d}")

    # Weighted and Macro averages
    xgb_wa = report_xgb['weighted avg']
    rf_wa = report_rf['weighted avg']
    mlp_wa = report_mlp['weighted avg']
    print(f"  {'Weighted Avg':<15s} {xgb_wa['precision']:>7.1%} {xgb_wa['recall']:>7.1%} "
          f"{xgb_wa['f1-score']:>7.1%} {rf_wa['f1-score']:>7.1%} {mlp_wa['f1-score']:>7.1%} "
          f"{int(xgb_wa['support']):>7d}")
    xgb_ma = report_xgb['macro avg']
    rf_ma = report_rf['macro avg']
    mlp_ma = report_mlp['macro avg']
    print(f"  {'Macro Avg':<15s} {xgb_ma['precision']:>7.1%} {xgb_ma['recall']:>7.1%} "
          f"{xgb_ma['f1-score']:>7.1%} {rf_ma['f1-score']:>7.1%} {mlp_ma['f1-score']:>7.1%} "
          f"{int(xgb_ma['support']):>7d}")

    # tab:confusion
    cm = confusion_matrix(y_test, y_pred_xgb)
    print(f"\n  [tab:confusion] XGBoost Confusion Matrix (n={len(y_test)}):")
    print(f"  {'':>15s}", end='')
    for cls in CLASSES:
        print(f" {cls[:5]:>6s}", end='')
    print(f" {'Recall':>7s} {'F1':>7s}")
    for i, cls in enumerate(CLASSES):
        support = cm[i].sum()
        rec = cm[i][i] / support if support > 0 else 0
        f1c = report_xgb[cls]['f1-score']
        print(f"  {cls:>15s}", end='')
        for j in range(len(CLASSES)):
            print(f" {cm[i][j]:>6d}", end='')
        print(f" {rec:>6.1%} {f1c:>6.1%}")
    print(f"  {'Precision':>15s}", end='')
    for j in range(len(CLASSES)):
        col_sum = cm[:, j].sum()
        prec = cm[j][j] / col_sum if col_sum > 0 else 0
        print(f" {prec:>5.1%}", end=' ')
    print()
    print(f"  Overall accuracy: {np.trace(cm)}/{len(y_test)} = {np.trace(cm)/len(y_test):.1%}")

    return results, y_test, len(y_test)


# ═══════════════════════════════════════════════════════════
# 2. 계층적 5-겹 교차 검증
# ═══════════════════════════════════════════════════════════
def run_5fold_cv(X, y, le):
    print("\n" + "="*70)
    print("  2. 계층적 5-겹 교차 검증")
    print("="*70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = {name: [] for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']}
    fold_results_macro = {name: [] for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']}
    per_class_f1 = {name: {cls: [] for cls in CLASSES} for name in ['XGBoost', 'RandomForest', 'MLP']}

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        xgb_m, rf_m, mlp_m = make_models()

        # XGBoost (no separate val split for CV)
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred_xgb = xgb_m.predict(X_te)

        rf_m.fit(X_tr, y_tr)
        y_pred_rf = rf_m.predict(X_te)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        mlp_m.fit(X_tr_s, y_tr)
        y_pred_mlp = mlp_m.predict(scaler.transform(X_te))

        y_pred_heur = heuristic_predict(X_te)

        for name, y_pred in [('XGBoost', y_pred_xgb), ('RandomForest', y_pred_rf),
                              ('MLP', y_pred_mlp), ('Heuristic', y_pred_heur)]:
            f1w = f1_score(y_te, y_pred, average='weighted', zero_division=0)
            f1m = f1_score(y_te, y_pred, average='macro', zero_division=0)
            fold_results[name].append(f1w * 100)
            fold_results_macro[name].append(f1m * 100)

        # Per-class F1 for ML models
        for name, y_pred in [('XGBoost', y_pred_xgb), ('RandomForest', y_pred_rf), ('MLP', y_pred_mlp)]:
            report = classification_report(y_te, y_pred, target_names=le.classes_,
                                            output_dict=True, zero_division=0)
            for cls in CLASSES:
                per_class_f1[name][cls].append(report[cls]['f1-score'] * 100)

    # tab:kfold
    print(f"\n  [tab:kfold] 5-Fold CV Results:")
    print(f"  {'Model':<15s}", end='')
    for i in range(5):
        print(f" {'F'+str(i+1):>6s}", end='')
    print(f" {'Wt Mean±SD':>14s} {'Mac Mean':>10s}")
    print(f"  {'-'*70}")

    cv_means = {}
    for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']:
        scores = fold_results[name]
        mac_scores = fold_results_macro[name]
        mean_w = np.mean(scores)
        std_w = np.std(scores, ddof=1)
        mean_m = np.mean(mac_scores)
        cv_means[name] = (mean_w, std_w)
        print(f"  {name:<15s}", end='')
        for s in scores:
            print(f" {s:>6.1f}", end='')
        print(f"   {mean_w:.2f}±{std_w:.2f}   {mean_m:.2f}")

    # Per-class F1 averages
    print(f"\n  Per-class 5-fold CV F1 (mean±sd):")
    for name in ['XGBoost', 'RandomForest', 'MLP']:
        print(f"    {name}:", end='')
        for cls in CLASSES:
            vals = per_class_f1[name][cls]
            print(f" {cls[:5]}={np.mean(vals):.1f}±{np.std(vals, ddof=1):.1f}", end='')
        print()

    return fold_results, fold_results_macro, cv_means, per_class_f1


# ═══════════════════════════════════════════════════════════
# 3. 10회 반복 무작위 분할
# ═══════════════════════════════════════════════════════════
def run_10repeat(X, y):
    print("\n" + "="*70)
    print("  3. 10회 반복 무작위 분할")
    print("="*70)

    repeat_results = {name: [] for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']}

    for seed in range(10):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y)

        xgb_m, rf_m, mlp_m = make_models()

        xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred_xgb = xgb_m.predict(X_te)

        rf_m.fit(X_tr, y_tr)
        y_pred_rf = rf_m.predict(X_te)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        mlp_m.fit(X_tr_s, y_tr)
        y_pred_mlp = mlp_m.predict(scaler.transform(X_te))

        y_pred_heur = heuristic_predict(X_te)

        for name, y_pred in [('XGBoost', y_pred_xgb), ('RandomForest', y_pred_rf),
                              ('MLP', y_pred_mlp), ('Heuristic', y_pred_heur)]:
            f1w = f1_score(y_te, y_pred, average='weighted', zero_division=0)
            repeat_results[name].append(f1w * 100)

    print(f"\n  10-Repeat Results:")
    for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']:
        scores = repeat_results[name]
        print(f"  {name:<15s}: {np.mean(scores):.2f}% ± {np.std(scores, ddof=1):.2f}%")

    return repeat_results


# ═══════════════════════════════════════════════════════════
# 4. LOLO CV
# ═══════════════════════════════════════════════════════════
def run_lolo_cv(X, y, le, library):
    print("\n" + "="*70)
    print("  4. Leave-One-Library-Out CV")
    print("="*70)

    libs = ['OpenSSL', 'PyCryptodome', 'Windows CNG', 'PyNaCl', 'pyaes']
    lolo_results = {name: [] for name in ['XGBoost', 'RandomForest', 'MLP']}
    lolo_details = []

    for lib in libs:
        test_mask = library == lib
        train_mask = ~test_mask
        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        xgb_m, rf_m, mlp_m = make_models()

        xgb_m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        y_pred_xgb = xgb_m.predict(X_te)

        rf_m.fit(X_tr, y_tr)
        y_pred_rf = rf_m.predict(X_te)

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        mlp_m.fit(X_tr_s, y_tr)
        y_pred_mlp = mlp_m.predict(scaler.transform(X_te))

        f1_xgb = f1_score(y_te, y_pred_xgb, average='weighted', zero_division=0) * 100
        f1_rf = f1_score(y_te, y_pred_rf, average='weighted', zero_division=0) * 100
        f1_mlp = f1_score(y_te, y_pred_mlp, average='weighted', zero_division=0) * 100

        lolo_results['XGBoost'].append(f1_xgb)
        lolo_results['RandomForest'].append(f1_rf)
        lolo_results['MLP'].append(f1_mlp)

        lolo_details.append({
            'library': lib, 'train': len(y_tr), 'test': len(y_te),
            'XGB': f1_xgb, 'RF': f1_rf, 'MLP': f1_mlp,
        })

    print(f"\n  [tab:lolo] LOLO CV Results:")
    print(f"  {'Test Library':<15s} {'Train':>6s} {'Test':>6s} {'XGB':>7s} {'RF':>7s} {'MLP':>7s}")
    print(f"  {'-'*50}")
    for d in lolo_details:
        print(f"  {d['library']:<15s} {d['train']:>6d} {d['test']:>6d} "
              f"{d['XGB']:>6.1f} {d['RF']:>6.1f} {d['MLP']:>6.1f}")

    print(f"  {'-'*50}")
    for name in ['XGBoost', 'RandomForest', 'MLP']:
        scores = lolo_results[name]
        print(f"  Mean±SD ({name}): {np.mean(scores):.1f} ± {np.std(scores, ddof=1):.1f}")

    return lolo_results, lolo_details


# ═══════════════════════════════════════════════════════════
# 5. 쌍별 t-검정
# ═══════════════════════════════════════════════════════════
def run_pairwise_tests(fold_results):
    print("\n" + "="*70)
    print("  5. 쌍별 t-검정 (5-Fold CV)")
    print("="*70)

    pairs = [
        ('XGBoost', 'RandomForest'),
        ('XGBoost', 'MLP'),
        ('RandomForest', 'MLP'),
        ('XGBoost', 'Heuristic'),
    ]

    print(f"\n  {'Comparison':<30s} {'ΔF1(%p)':>10s} {'t-stat':>8s} {'p-value':>10s}")
    print(f"  {'-'*60}")

    pairwise_results = []
    for a, b in pairs:
        scores_a = np.array(fold_results[a])
        scores_b = np.array(fold_results[b])
        diff = np.mean(scores_a) - np.mean(scores_b)
        t_stat, p_val = stats.ttest_rel(scores_a, scores_b)
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.0167 else '*' if p_val < 0.05 else 'ns'
        print(f"  {a+' vs '+b:<30s} {diff:>+9.2f} {t_stat:>8.2f} {p_val:>9.3f}{sig}")
        pairwise_results.append({
            'a': a, 'b': b, 'delta': diff, 't': t_stat, 'p': p_val,
        })

    return pairwise_results


# ═══════════════════════════════════════════════════════════
# 6. Baseline 재현
# ═══════════════════════════════════════════════════════════
def run_baselines(X, y, le):
    print("\n" + "="*70)
    print("  6. Baseline 재현 (Entropy-only, Entropy+Length)")
    print("="*70)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Entropy-only (F1만)
    X_tr_e = X_train[:, [0]]
    X_te_e = X_test[:, [0]]
    X_va_e = X_val[:, [0]]
    m_e = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              objective='multi:softprob', num_class=5,
                              eval_metric='mlogloss', use_label_encoder=False,
                              random_state=42, verbosity=0)
    m_e.fit(X_tr_e, y_train, eval_set=[(X_va_e, y_val)], verbose=False)
    y_pred_e = m_e.predict(X_te_e)
    f1w_e = f1_score(y_test, y_pred_e, average='weighted', zero_division=0)
    prec_e = precision_score(y_test, y_pred_e, average='weighted', zero_division=0)
    rec_e = recall_score(y_test, y_pred_e, average='weighted', zero_division=0)

    # Entropy+Length (F1, F3)
    X_tr_el = X_train[:, [0, 2]]
    X_te_el = X_test[:, [0, 2]]
    X_va_el = X_val[:, [0, 2]]
    m_el = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               objective='multi:softprob', num_class=5,
                               eval_metric='mlogloss', use_label_encoder=False,
                               random_state=42, verbosity=0)
    m_el.fit(X_tr_el, y_train, eval_set=[(X_va_el, y_val)], verbose=False)
    y_pred_el = m_el.predict(X_te_el)
    f1w_el = f1_score(y_test, y_pred_el, average='weighted', zero_division=0)
    prec_el = precision_score(y_test, y_pred_el, average='weighted', zero_division=0)
    rec_el = recall_score(y_test, y_pred_el, average='weighted', zero_division=0)

    print(f"  Entropy-only:  Prec={prec_e:.1%} Recall={rec_e:.1%} F1_w={f1w_e:.1%}")
    print(f"  Entropy+Length: Prec={prec_el:.1%} Recall={rec_el:.1%} F1_w={f1w_el:.1%}")

    return {
        'entropy_only': {'prec': prec_e, 'rec': rec_e, 'f1w': f1w_e},
        'entropy_length': {'prec': prec_el, 'rec': rec_el, 'f1w': f1w_el},
    }


# ═══════════════════════════════════════════════════════════
# 7. 3특징 vs 10특징 비교
# ═══════════════════════════════════════════════════════════
def run_3feat_vs_10feat(X, y):
    print("\n" + "="*70)
    print("  7. 3특징(F1,F3,F6) vs 10특징 비교 (5-fold CV)")
    print("="*70)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_3 = []
    scores_10 = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        # 10특징
        m10 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                  objective='multi:softprob', num_class=5,
                                  eval_metric='mlogloss', use_label_encoder=False,
                                  random_state=42, verbosity=0)
        m10.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
        scores_10.append(f1_score(y_te, m10.predict(X_te), average='weighted', zero_division=0) * 100)

        # 3특징 (F1=entropy, F3=length, F6=region → indices 0, 2, 5)
        m3 = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                 objective='multi:softprob', num_class=5,
                                 eval_metric='mlogloss', use_label_encoder=False,
                                 random_state=42, verbosity=0)
        X_tr_3 = X_tr[:, [0, 2, 5]]
        X_te_3 = X_te[:, [0, 2, 5]]
        m3.fit(X_tr_3, y_tr, eval_set=[(X_te_3, y_te)], verbose=False)
        scores_3.append(f1_score(y_te, m3.predict(X_te_3), average='weighted', zero_division=0) * 100)

    mean_3 = np.mean(scores_3)
    std_3 = np.std(scores_3, ddof=1)
    mean_10 = np.mean(scores_10)
    std_10 = np.std(scores_10, ddof=1)
    delta = mean_10 - mean_3
    t_stat, p_val = stats.ttest_rel(scores_10, scores_3)

    print(f"  3특징: {mean_3:.2f}% ± {std_3:.2f}%")
    print(f"  10특징: {mean_10:.2f}% ± {std_10:.2f}%")
    print(f"  ΔF1 = +{delta:.2f}%p, t={t_stat:.2f}, p={p_val:.3f}")

    return {
        'mean_3': mean_3, 'std_3': std_3,
        'mean_10': mean_10, 'std_10': std_10,
        'delta': delta, 't': t_stat, 'p': p_val,
    }


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    print(f"데이터셋: {csv_path}")

    X, y, le, library = load_data(csv_path)
    print(f"총 표본: {len(y)}")

    # 1. 단일 분할
    single_results, y_test_single, n_test = run_single_split(X, y, le)

    # 2. 5-fold CV
    fold_results, fold_results_macro, cv_means, per_class_f1 = run_5fold_cv(X, y, le)

    # 3. 10회 반복
    repeat_results = run_10repeat(X, y)

    # 4. LOLO CV
    if library is not None:
        lolo_results, lolo_details = run_lolo_cv(X, y, le, library)
    else:
        print("\n  LOLO CV: library 열 없음, 건너뜀")
        lolo_results, lolo_details = None, None

    # 5. 쌍별 t-검정
    pairwise_results = run_pairwise_tests(fold_results)

    # 6. Baselines
    baselines = run_baselines(X, y, le)

    # 7. 3특징 vs 10특징
    feat_comparison = run_3feat_vs_10feat(X, y)

    # ── 최종 요약 ──
    print("\n" + "="*70)
    print("  최종 요약 (논문 수치 갱신용)")
    print("="*70)

    print(f"\n  단일 분할 (n_test={n_test}):")
    for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']:
        r = single_results[name]
        print(f"    {name}: Prec={r['prec']:.1%} Rec={r['rec']:.1%} F1w={r['f1w']:.1%} F1m={r['f1m']:.1%}")

    print(f"\n  5-fold CV:")
    for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']:
        m, s = cv_means[name]
        print(f"    {name}: {m:.2f}% ± {s:.2f}%")

    print(f"\n  10회 반복:")
    for name in ['XGBoost', 'RandomForest', 'MLP', 'Heuristic']:
        scores = repeat_results[name]
        print(f"    {name}: {np.mean(scores):.2f}% ± {np.std(scores, ddof=1):.2f}%")

    if lolo_results:
        print(f"\n  LOLO CV:")
        for name in ['XGBoost', 'RandomForest', 'MLP']:
            scores = lolo_results[name]
            print(f"    {name}: {np.mean(scores):.1f}% ± {np.std(scores, ddof=1):.1f}%")

        print(f"\n  LOLO-to-5fold gap:")
        for name in ['XGBoost', 'RandomForest', 'MLP']:
            lolo_mean = np.mean(lolo_results[name])
            cv_mean = cv_means[name][0]
            gap = lolo_mean - cv_mean
            print(f"    {name}: {gap:+.1f}%p")
