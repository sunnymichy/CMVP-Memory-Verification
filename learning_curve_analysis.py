"""
learning_curve_analysis.py
논문 섹션 4: Learning Curve 분석 — 데이터셋 크기에 따른 성능 포화 검증.

훈련 셋 크기를 점진적으로 증가시키며 CatBoost(주 분류기)의
Weighted F1, Macro F1, KEY Recall을 5-fold stratified CV로 측정한다.

사용법:
  python learning_curve_analysis.py [dataset_path]
  (기본값: dataset/crypto_features.csv)

출력:
  1. 콘솔 테이블 (논문용 수치)
  2. TikZ 좌표 (논문 Figure에 직접 삽입 가능)
  3. 포화 분석 (마지막 3 구간의 ΔF1)
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

# 학습 곡선에 사용할 훈련 셋 크기 (10개 단계)
TRAIN_SIZES = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]


def load_data(csv_path):
    """CSV 데이터셋 로드."""
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :10].values.astype(np.float64)
    y_str = df['label'].values
    le = LabelEncoder()
    le.fit(CLASSES)
    y = le.transform(y_str)
    return X, y, le


def make_catboost():
    """논문 Table 5 최적 하이퍼파라미터의 CatBoost 모델 생성."""
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
    각 train_size에서 5-fold stratified CV를 수행하고
    Weighted F1, Macro F1, KEY Recall을 반환한다.

    train_size가 전체 데이터보다 작을 경우, 각 fold 내에서
    stratified subsampling으로 축소한다.
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

            # 전체 데이터 사용 시 그대로, 아닐 경우 stratified subsampling
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

            # CatBoost 학습
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
    """마지막 3~4 구간에서의 ΔF1 변화량 분석으로 포화 여부 판정."""
    print("\n" + "=" * 70)
    print("  포화 분석 (Saturation Analysis)")
    print("=" * 70)

    # 연속 구간 간 delta
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

    # 마지막 3 구간의 평균 delta
    last_3_deltas = deltas_wf1[-3:]
    avg_delta = np.mean(last_3_deltas)
    print(f"\n  마지막 3 구간 평균 ΔWF1: {avg_delta:+.2f} pp")

    if abs(avg_delta) < 0.5:
        print("  → 포화 상태 확인: 마지막 3 구간 평균 변화 < 0.5 pp")
        print("    (추가 데이터에 의한 성능 향상 여지가 제한적)")
    elif abs(avg_delta) < 1.0:
        print("  → 준-포화 상태: 마지막 3 구간 평균 변화 0.5~1.0 pp")
    else:
        print("  → 미포화: 추가 데이터가 성능 향상에 기여할 가능성 있음")

    # Marginal gain analysis: last vs first half
    first_half_gain = results[len(results) // 2]['wf1_mean'] - results[0]['wf1_mean']
    second_half_gain = results[-1]['wf1_mean'] - results[len(results) // 2]['wf1_mean']
    print(f"\n  전반부 총 gain (N={results[0]['size']}→{results[len(results)//2]['size']}): "
          f"+{first_half_gain:.2f} pp")
    print(f"  후반부 총 gain (N={results[len(results)//2]['size']}→{results[-1]['size']}): "
          f"+{second_half_gain:.2f} pp")
    ratio = second_half_gain / first_half_gain if first_half_gain > 0 else 0
    print(f"  후반부/전반부 비율: {ratio:.2f} (1.0 미만이면 수확 체감)")

    return avg_delta, first_half_gain, second_half_gain


def generate_tikz_coordinates(results):
    """논문 Figure에 삽입할 TikZ 좌표 문자열을 생성."""
    print("\n" + "=" * 70)
    print("  TikZ 좌표 (논문 Figure 삽입용)")
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
    """논문 삽입용 테이블 출력."""
    print("\n" + "=" * 70)
    print("  논문용 Learning Curve 테이블")
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
    print(f"데이터셋: {csv_path}")

    X, y, le = load_data(csv_path)
    print(f"총 표본: {len(y)}")
    print(f"클래스 분포: {dict(zip(*np.unique(le.inverse_transform(y), return_counts=True)))}")

    print("\n" + "=" * 70)
    print("  Learning Curve 분석 (CatBoost, 5-fold stratified CV)")
    print("=" * 70)

    results = run_learning_curve(X, y, le)

    print_paper_table(results)
    avg_delta, first_gain, second_gain = saturation_analysis(results)
    generate_tikz_coordinates(results)

    print("\n" + "=" * 70)
    print("  완료")
    print("=" * 70)


if __name__ == '__main__':
    main()
