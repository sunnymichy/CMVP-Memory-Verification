"""
evaluate.py
논문 섹션 4장: 실험 결과 평가.

1) 전체 성능 비교 (논문 표3)
2) 하이브리드 앙상블 평가
3) Ablation Study (논문 표4) - 특징 그룹별 기여도 분석
4) 통계적 유의성 검증 (10회 반복 t-검정)
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
# 1. Ablation Study (논문 표4)
# ════════════════════════════════════════════════════════════

# 특징 그룹 정의 (0-indexed)
FEATURE_GROUPS = {
    'F1 (엔트로피)':           [0],
    'F2 (카이제곱)':           [1],
    'F3 (길이)':               [2],
    'F4-F5 (표준길이)':        [3, 4],
    'F6 (메모리영역)':         [5],
    'F7-F8 (시간패턴)':        [6, 7],
    'F9-F10 (교차특징)':       [8, 9],
}


def _train_xgb(X_train, y_train):
    """ablation용 XGBoost 간이 학습."""
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
    각 특징 그룹을 순차적으로 제거하여 F1 감소량을 측정한다.
    논문 표4 재현.
    """
    print("\n" + "=" * 60)
    print("  Ablation Study (특징별 기여도 분석)")
    print("=" * 60)

    # Baseline: 전체 10개 특징
    full_model = _train_xgb(X_train, y_train)
    baseline_f1 = f1_score(y_test, full_model.predict(X_test), average='weighted')
    print(f"\n  Baseline F1 (전체 특징): {baseline_f1:.4f}")
    print(f"\n  {'제거 특징':<25s} {'F1':>8s} {'감소량':>10s} {'기여도':>8s}")
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

    # 기여도 = 개별 감소량 / 전체 감소량 합
    total_drop = sum(drops.values())
    for name, drop in sorted(drops.items(), key=lambda x: -x[1]):
        contrib = (drop / total_drop * 100) if total_drop > 0 else 0
        print(f"  {name:<25s} {baseline_f1 - drop:>7.4f} {-drop*100:>+9.1f}%p {contrib:>7.1f}%")

    return drops, baseline_f1


# ════════════════════════════════════════════════════════════
# 2. 하이브리드 앙상블 평가
# ════════════════════════════════════════════════════════════

def evaluate_hybrid_ensemble(xgb_model, X_test, y_test, le):
    """하이브리드 앙상블 vs XGBoost 단독 비교."""
    print("\n" + "=" * 60)
    print("  하이브리드 앙상블 평가")
    print("=" * 60)

    # XGBoost 단독
    y_xgb = xgb_model.predict(X_test)
    xgb_f1 = f1_score(y_test, y_xgb, average='weighted')
    print(f"\n  [XGBoost 단독]")
    print(classification_report(y_test, y_xgb, target_names=le.classes_,
                                digits=4))

    # 하이브리드 앙상블
    ensemble = HybridEnsemble(xgb_model, sensitivity='normal')
    y_hybrid, confidences, paths = ensemble.predict(X_test)
    hybrid_f1 = f1_score(y_test, y_hybrid, average='weighted')

    print(f"  [하이브리드 앙상블]")
    print(classification_report(y_test, y_hybrid, target_names=le.classes_,
                                digits=4))

    # 경로 분포
    path_counts = {}
    for p in paths:
        path_counts[p] = path_counts.get(p, 0) + 1
    print("  경로별 분포:")
    for path, count in sorted(path_counts.items()):
        print(f"    {path:<15s}: {count:>5d} ({count/len(paths)*100:.1f}%)")

    # 신뢰도 통계
    print(f"\n  평균 신뢰도: {np.mean(confidences):.4f}")
    print(f"  신뢰도 > 90%: {np.sum(confidences >= 0.9)}/{len(confidences)}")

    print(f"\n  XGBoost F1:  {xgb_f1:.4f}")
    print(f"  앙상블 F1:   {hybrid_f1:.4f}")
    print(f"  개선:        {(hybrid_f1 - xgb_f1)*100:+.2f}%p")

    return hybrid_f1


# ════════════════════════════════════════════════════════════
# 3. 통계적 유의성 검증 (10회 반복)
# ════════════════════════════════════════════════════════════

def statistical_validation(X, y, n_repeats: int = 10):
    """
    논문 섹션 4.3: 10회 무작위 분할 반복 실험 + 대응 표본 t-검정.
    """
    print("\n" + "=" * 60)
    print(f"  통계적 유의성 검증 ({n_repeats}회 반복)")
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

        # 하이브리드 앙상블
        ens = HybridEnsemble(model, sensitivity='normal')
        y_ens, _, _ = ens.predict(X_test)
        ens_f1 = f1_score(y_test, y_ens, average='weighted')
        ensemble_f1s.append(ens_f1)

        # 휴리스틱 baseline
        h_preds = []
        for feat in X_test:
            score = compute_heuristic_score(feat)
            h_preds.append(heuristic_to_class(score, feat))
        h_f1 = f1_score(y_test, h_preds, average='weighted')
        heuristic_f1s.append(h_f1)

        print(f"  반복 {seed+1:>2d}: ML={ml_f1:.4f}  앙상블={ens_f1:.4f}  "
              f"휴리스틱={h_f1:.4f}")

    ml_f1s = np.array(ml_f1s)
    heuristic_f1s = np.array(heuristic_f1s)
    ensemble_f1s = np.array(ensemble_f1s)

    # 대응 표본 t-검정 (ML vs 휴리스틱)
    t_stat, p_value = stats.ttest_rel(ml_f1s, heuristic_f1s)

    print(f"\n  ── 결과 요약 ──")
    print(f"  ML F1:        {ml_f1s.mean()*100:.2f}% +/- {ml_f1s.std()*100:.2f}%")
    print(f"  앙상블 F1:    {ensemble_f1s.mean()*100:.2f}% +/- {ensemble_f1s.std()*100:.2f}%")
    print(f"  휴리스틱 F1:  {heuristic_f1s.mean()*100:.2f}% +/- {heuristic_f1s.std()*100:.2f}%")
    print(f"  ML-휴리스틱 차이: +{(ml_f1s.mean()-heuristic_f1s.mean())*100:.2f}%p")
    print(f"  t-statistic:  {t_stat:.4f}")
    print(f"  p-value:      {p_value:.6f} {'*** (p<0.001, 유의)' if p_value < 0.001 else ''}")

    return {
        'ml_f1s': ml_f1s,
        'ensemble_f1s': ensemble_f1s,
        'heuristic_f1s': heuristic_f1s,
        't_stat': t_stat,
        'p_value': p_value,
    }


# ════════════════════════════════════════════════════════════
# 4. 휴리스틱 baseline 평가
# ════════════════════════════════════════════════════════════

def evaluate_heuristic_baseline(X_test, y_test, le):
    """순수 휴리스틱 점수 기반 분류 성능 (baseline)."""
    print("\n" + "=" * 60)
    print("  휴리스틱 Baseline 평가")
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
    print(f"  평균 점수: {scores.mean():.1f}")
    print(f"  점수 분포: min={scores.min():.0f} / "
          f"median={np.median(scores):.0f} / max={scores.max():.0f}")

    return f1_score(y_test, y_pred, average='weighted')


# ════════════════════════════════════════════════════════════
# 메인 실행
# ════════════════════════════════════════════════════════════

def run_full_evaluation(csv_path: str):
    """전체 평가 파이프라인 실행."""
    print("데이터 로드 중...")
    X, y, le = load_dataset(csv_path)

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    # XGBoost 학습
    print("XGBoost 학습 중...")
    xgb_model = _train_xgb(X_train, y_train)

    # (1) 휴리스틱 baseline
    evaluate_heuristic_baseline(X_test, y_test, le)

    # (2) 하이브리드 앙상블
    evaluate_hybrid_ensemble(xgb_model, X_test, y_test, le)

    # (3) Ablation Study
    ablation_study(X_train, y_train, X_test, y_test)

    # (4) 통계적 유의성
    statistical_validation(X, y, n_repeats=10)

    print("\n전체 평가 완료.")


if __name__ == '__main__':
    import sys
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset/crypto_features.csv'
    run_full_evaluation(csv_path)
