"""
hybrid_ensemble.py
논문 섹션 3.4.5: 하이브리드 앙상블 기법.

기계학습 확률 예측과 규칙 기반 휴리스틱을 결합하여 경계 사례에 대한 탐지 성능을 보완한다.

3경로 의사결정:
  경로 1: ML 신뢰도 >= 90% → ML 결과 채택
  경로 2: ML 신뢰도 70~90% → ML과 휴리스틱 비교, 일치 시 채택
  경로 3: ML 신뢰도 < 70% → 휴리스틱 규칙 기반 결정
"""

import numpy as np
from typing import Tuple

from heuristic_scorer import (
    compute_heuristic_score,
    heuristic_to_class,
    check_short_block_threshold,
    CLASS_NAMES,
)


class HybridEnsemble:
    """하이브리드 앙상블 분류기."""

    def __init__(self, ml_model, tau_high: float = 0.90, tau_low: float = 0.70,
                 sensitivity: str = 'normal'):
        """
        Args:
            ml_model: predict_proba()를 지원하는 학습된 ML 모델 (XGBoost 등)
            tau_high: 고신뢰 임계값 (기본 90%)
            tau_low: 저신뢰 임계값 (기본 70%)
            sensitivity: 탐지 감도 ('high' / 'normal' / 'low')
        """
        self.ml_model = ml_model
        self.tau_high = tau_high
        self.tau_low = tau_low
        self.sensitivity = sensitivity

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, list]:
        """
        하이브리드 앙상블 예측.

        Args:
            X: (n_samples, 10) 특징 벡터 배열

        Returns:
            predictions: (n_samples,) 예측 클래스 인덱스
            confidences: (n_samples,) 신뢰도 (0.0~1.0)
            paths: 각 표본이 통과한 경로 ('ml_high' / 'ml_agree' /
                   'ml_disagree' / 'heuristic')
        """
        proba = self.ml_model.predict_proba(X)  # (n, 5)

        predictions = []
        confidences = []
        paths = []

        for i in range(len(X)):
            features = X[i]
            ml_class = int(np.argmax(proba[i]))
            ml_conf = float(proba[i][ml_class])

            h_score = compute_heuristic_score(features)
            h_class = heuristic_to_class(h_score, features)

            if ml_conf >= self.tau_high:
                # ── 경로 1: 고신뢰 → ML 결과 직접 채택 ──
                predictions.append(ml_class)
                confidences.append(ml_conf)
                paths.append('ml_high')

            elif ml_conf >= self.tau_low:
                # ── 경로 2: 중간 신뢰 → ML-휴리스틱 비교 ──
                if ml_class == h_class:
                    predictions.append(ml_class)
                    confidences.append(ml_conf)
                    paths.append('ml_agree')
                else:
                    # 불일치: ML 클래스 유지, 신뢰도 하향 조정
                    predictions.append(ml_class)
                    confidences.append(0.50)
                    paths.append('ml_disagree')

            else:
                # ── 경로 3: 저신뢰 → 휴리스틱 규칙 기반 ──
                predictions.append(h_class)
                confidences.append(h_score / 100.0)
                paths.append('heuristic')

        return (
            np.array(predictions),
            np.array(confidences),
            paths,
        )

    def predict_with_threshold(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        탐지 임계값을 적용한 최종 분류.
        단축 블록 보정 포함.

        Returns:
            final_predictions: 임계값 미달 시 NON_CRYPTO(4)로 변경
            confidences: 신뢰도
        """
        predictions, confidences, paths = self.predict(X)

        for i in range(len(X)):
            features = X[i]
            score = compute_heuristic_score(features)

            # 단축 블록 검사 (8~15바이트)
            if not check_short_block_threshold(features, score, self.sensitivity):
                predictions[i] = 4  # NON_CRYPTO
                confidences[i] = min(confidences[i], 0.30)

        return predictions, confidences

    def predict_detail(self, X: np.ndarray) -> list:
        """
        상세 예측 결과를 딕셔너리 리스트로 반환 (디버깅/리포트 용).
        """
        proba = self.ml_model.predict_proba(X)
        predictions, confidences, paths = self.predict(X)

        details = []
        for i in range(len(X)):
            features = X[i]
            h_score = compute_heuristic_score(features)

            details.append({
                'index': i,
                'ml_class': CLASS_NAMES[int(np.argmax(proba[i]))],
                'ml_confidence': float(np.max(proba[i])),
                'ml_proba': {CLASS_NAMES[j]: float(proba[i][j]) for j in range(5)},
                'heuristic_score': h_score,
                'heuristic_class': CLASS_NAMES[heuristic_to_class(h_score, features)],
                'final_class': CLASS_NAMES[predictions[i]],
                'final_confidence': float(confidences[i]),
                'decision_path': paths[i],
                'features': {
                    'entropy': features[0],
                    'chi2': features[1],
                    'length': features[2],
                    'is_key_len': bool(features[3]),
                    'is_iv_len': bool(features[4]),
                    'memory_region': int(features[5]),
                    'change_count': int(features[6]),
                    'pattern': int(features[7]),
                },
            })

        return details


# ─── 단독 실행 시 테스트 ───
if __name__ == '__main__':
    import joblib
    import os

    model_path = 'models/xgb_classifier.pkl'
    if not os.path.exists(model_path):
        print(f"학습된 모델 없음: {model_path}")
        print("model_trainer.py를 먼저 실행하세요.")
        exit(1)

    xgb_model = joblib.load(model_path)
    ensemble = HybridEnsemble(xgb_model, sensitivity='normal')

    # 테스트 데이터 (수동 생성)
    test_features = np.array([
        # AES-256 키: 고엔트로피, 표준길이, DLL
        [7.9, 250.0, 32.0, 1.0, 0.0, 1.0, 18.0, 3.0, 39.5, 1.0],
        # IV: 중간 엔트로피, IV길이, 힙
        [7.2, 0.0, 16.0, 1.0, 1.0, 2.0, 0.0, 0.0, 28.8, 0.0],
        # 평문: 저엔트로피
        [4.2, 800.0, 64.0, 1.0, 0.0, 2.0, 0.0, 0.0, 17.6, 0.0],
    ])

    details = ensemble.predict_detail(test_features)
    for d in details:
        print(f"\n--- 표본 {d['index']} ---")
        print(f"  ML:         {d['ml_class']} ({d['ml_confidence']:.1%})")
        print(f"  Heuristic:  {d['heuristic_class']} (점수 {d['heuristic_score']:.0f})")
        print(f"  최종 결정:  {d['final_class']} ({d['final_confidence']:.1%})")
        print(f"  경로:       {d['decision_path']}")
