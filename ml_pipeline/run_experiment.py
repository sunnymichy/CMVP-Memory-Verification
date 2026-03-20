"""
run_experiment.py
전체 실험 파이프라인을 일괄 실행하는 메인 스크립트.

실행 순서:
  1) 합성 데이터셋 생성 (실제 데이터가 없을 때)
  2) 3개 모델 학습 + 비교
  3) 하이브리드 앙상블 평가
  4) Ablation Study
  5) 통계적 유의성 검증

사용법:
  python run_experiment.py                        # 합성 데이터 사용
  python run_experiment.py dataset/real_data.csv  # 실제 데이터 사용
"""

import sys
import os

# ml_pipeline 디렉토리 기준 실행
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from generate_synthetic_dataset import generate_dataset
from model_trainer import train_and_evaluate_all
from evaluate import (
    run_full_evaluation,
    ablation_study,
    evaluate_hybrid_ensemble,
    statistical_validation,
)


def main():
    csv_path = 'dataset/crypto_features.csv'

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        if not os.path.exists(csv_path):
            print(f"파일을 찾을 수 없음: {csv_path}")
            sys.exit(1)
        print(f"실제 데이터 사용: {csv_path}")
    else:
        if not os.path.exists(csv_path):
            print("=" * 60)
            print("  [Step 0] 합성 데이터셋 생성")
            print("=" * 60)
            generate_dataset(csv_path)
        else:
            print(f"기존 데이터셋 사용: {csv_path}")

    # ── Step 1: 3개 모델 학습 + 비교 (논문 표3) ──
    print("\n" + "=" * 60)
    print("  [Step 1] 모델 학습 및 성능 비교")
    print("=" * 60)
    results, X_train, X_val, X_test, y_train, y_val, y_test, le = \
        train_and_evaluate_all(csv_path, save_dir='models')

    # ── Step 2: 전체 평가 (앙상블 + ablation + t-검정) ──
    print("\n" + "=" * 60)
    print("  [Step 2] 전체 평가 실행")
    print("=" * 60)
    run_full_evaluation(csv_path)

    print("\n" + "=" * 60)
    print("  전체 실험 완료")
    print("=" * 60)
    print("\n  생성된 파일:")
    print("    models/xgb_classifier.pkl   - XGBoost 모델")
    print("    models/rf_classifier.pkl    - Random Forest 모델")
    print("    models/mlp_classifier.pkl   - MLP 모델")
    print("    models/mlp_scaler.pkl       - MLP용 StandardScaler")
    print("    models/label_encoder.pkl    - 레이블 인코더")
    print(f"    {csv_path}    - 데이터셋")


if __name__ == '__main__':
    main()
